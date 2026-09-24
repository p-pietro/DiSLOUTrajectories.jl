@doc raw"""
    dislou_solve(H, ψ0, tlist, c_ops; gauge_set, <keyword arguments>)

Quantum trajectories of the time-independent Lindblad equation

```math
\frac{d\hat{\rho}}{dt} = -i[\hat{H}, \hat{\rho}]
+ \sum_\mu \left( \hat{C}_\mu \hat{\rho} \hat{C}_\mu^\dagger
- \frac{1}{2} \{\hat{C}_\mu^\dagger \hat{C}_\mu, \hat{\rho}\} \right),
```

with the Diagonal, Switching, and Locally Optimal Unraveling (DiSLOU). The
trajectories are run by QuantumToolbox's `mcsolve`, which receives all the other
keyword arguments and returns its usual `TimeEvolutionMCSol`.

A gauge ``g`` (column `g` of `gauge_set`) shifts the collapse operators and the
Hamiltonian without changing the master equation (Eq. 9):

```math
\hat{C}_\mu^{(g)} = \hat{C}_\mu + \zeta_\mu^{(g)}, \qquad
\hat{H}^{(g)} = \hat{H} + \frac{i}{2} \sum_\mu \left( \zeta_\mu^{(g)} \hat{C}_\mu^\dagger
- \zeta_\mu^{(g)*} \hat{C}_\mu \right).
```

- **Layer I**: each trajectory starts in the gauge with the lowest jump activity
  ``A_g(\psi) = \sum_\mu \| \hat{C}_\mu^{(g)} \psi \|^2``. After each jump, it moves to
  the least active gauge if that activity is below `hysteresis` times the current
  one (Eqs. 11–13).
- **Layer II**: between jumps, the state evolves exactly in the eigenbasis of the
  effective Hamiltonian of the gauge, with [`GaugeEigenExponential`](@ref). Jump
  times are found by root finding on this exact solution.
- **Layer III** (optional): after a jump, a state within `residual_tolerance` of the
  `layer3_sizes` slowest eigenmodes of its gauge is projected on them, and evolves
  in that smaller basis until the next jump (Eqs. 20–21).

# Arguments

- `H`: Time-independent Hamiltonian ``\hat{H}``, a `QuantumObject` operator.
- `ψ0`: Initial state, a `QuantumObject` ket.
- `tlist`: Times at which the expectation values are computed.
- `c_ops`: Nonempty vector of time-independent collapse operators ``\hat{C}_\mu``.
- `gauge_set`: `Nc × Ng` matrix of shifts ``\zeta_\mu^{(g)}``, with one row per collapse
  operator and one column per gauge, or the result of [`discover_gauges`](@ref).
- `hysteresis`: The threshold ``\eta \in (0, 1]`` for switching gauge. Default `0.5`.
- `layer3_sizes`: Number of slow modes kept for Layer III, one integer for all gauges
  or one per gauge. Modes degenerate with the kept ones are added. Default `nothing`,
  which disables Layer III.
- `residual_tolerance`: Largest relative distance ``r_{\rm tol}`` of the state from
  the slow modes for Layer III. Default `1e-3`.
- `kwargs`: Keyword arguments of `mcsolve`, such as `e_ops`, `ntraj`, `rng`,
  `ensemblealg`, `saveat`, `keep_runs_results`, `progress_bar` or `callback`.

# Notes

- `dislou_solve` sets the `alg` and `jump_callback` of `mcsolve` itself.
- Jump records (`col_times`, `col_which`) refer to the shifted operators of the gauge
  that was active at each jump.
- Layers I and II are exact up to floating-point errors. Layer III replaces the state
  by its projection after each accepted jump, an error below `residual_tolerance`.
- Each gauge is diagonalized once, which takes ``O(N^3)`` time and ``O(N^2)`` memory for
  an ``N``-dimensional Hilbert space. With `e_ops = nothing`, `sol.expect` is an empty
  matrix instead of `nothing`.

# Returns

- `sol::TimeEvolutionMCSol`: the result of `mcsolve`.

# Examples

A driven cavity relaxes to the coherent state ``|\alpha\rangle``. With a gauge centered
on it, jumps become rare while the photon number follows the exact result:

```jldoctest
julia> using QuantumToolbox, DiSLOUTrajectories

julia> N, κ, α = 15, 1.0, 1.0;

julia> a = destroy(N);

julia> H = 0.5im * κ * (α * a' - conj(α) * a);

julia> tlist = 0:0.5:5;

julia> sol = dislou_solve(H, fock(N, 0), tlist, [sqrt(κ) * a];
           gauge_set = fill(-sqrt(κ) * α, 1, 1), e_ops = [a' * a], ntraj = 20,
           progress_bar = Val(false));

julia> isapprox(real(sol.expect[1, :]), abs2.(α .* (1 .- exp.(-κ .* tlist ./ 2))); atol = 1e-6)
true
```
"""
function dislou_solve(
        H::QuantumObject{Operator}, ψ0::QuantumObject{Ket}, tlist::AbstractVector,
        c_ops::AbstractVector{<:QuantumObject{Operator}};
        gauge_set,
        hysteresis::Real = 0.5,
        layer3_sizes = nothing,
        residual_tolerance::Real = 1.0e-3,
        e_ops = nothing,
        callback = nothing,
        tstops = Float64[],
        kwargs...,
    )
    isempty(c_ops) && throw(ArgumentError("dislou_solve needs at least one collapse operator"))
    0 < hysteresis <= 1 || throw(ArgumentError("hysteresis must be in (0, 1], got $hysteresis"))
    residual_tolerance > 0 || throw(ArgumentError("residual_tolerance must be positive, got $residual_tolerance"))
    for key in (:alg, :jump_callback)
        haskey(kwargs, key) && throw(ArgumentError("dislou_solve sets `$key` itself"))
    end

    shifts = _gauge_shifts(gauge_set, length(c_ops))
    gauges = [_shifted_operators(H, c_ops, shifts[:, g]) for g in axes(shifts, 2)]
    alg = GaugeEigenExponential(gauges; layer3_sizes)

    # Initial gauge (paper Eq. 12) and, with Layer III, projection of the initial state.
    # The state has the array type and precision of the eigenbases.
    ψ = eltype(first(alg.bases).λ).(to_dense(ψ0.data))
    C = [op.data for op in c_ops]
    g0 = argmin(_gauge_activities(C, shifts, ψ, similar(ψ)))
    reduced0 = !isempty(alg.reduced_bases) &&
        _project!(ψ, alg.reduced_bases[g0], residual_tolerance, similar(ψ), similar(ψ))

    # Jump operators of each gauge, built as QuantumToolbox builds those of `mcsolve`
    # (from version 0.47.2; earlier versions store plain matrices).
    jump_ops = [map(op -> get_data(cache_operator(QobjEvo(op), ψ)), Cg) for (_, Cg) in gauges]
    router = GaugeRouter(C, shifts, jump_ops, float(hysteresis), float(residual_tolerance), g0, reduced0)
    router_callback = _router_callback(router)
    callback = callback === nothing ? router_callback : CallbackSet(router_callback, callback)

    # QuantumToolbox's mcsolve fails when `e_ops = nothing` comes with an extra callback;
    # an empty list of observables gives the same result.
    e_ops = something(e_ops, typeof(H)[])

    # The norm decreases monotonically between jumps, so checking the step ends is
    # enough to detect a jump (the default only from QuantumToolbox 0.49).
    jump_callback = ContinuousLindbladJumpCallback(interp_points = 0)
    # Exact steps can be arbitrarily long. Stopping at each time of `tlist` keeps the
    # jump-time search within one interval, and results are saved at step ends.
    tstops = sort!(unique!(vcat(collect(Float64, tlist), tstops)))

    H0, C0 = gauges[g0]
    ψ0 = QuantumObject(ψ; type = Ket(), dims = ψ0.dimensions)
    return mcsolve(H0, ψ0, tlist, C0; alg, e_ops, callback, jump_callback, tstops, kwargs...)
end
