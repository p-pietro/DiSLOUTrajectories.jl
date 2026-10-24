# Layers II and III: the no-jump state evolves exactly in the eigenbasis of the
# effective Hamiltonian. This is written as an OrdinaryDiffEq algorithm, so that
# `mcsolve` keeps handling the jump times, saving and ensembles.

# Modes of the no-jump generator L = -i H_eff: eigenvalues `λ` and right
# eigenvectors `V` (columns), either all of them (Layer II) or only the slowest
# ones (Layer III). With V = Q R, the coordinates c = R⁻¹ Q†ψ of a state solve
# V c = ψ without inverting V, Q Q† projects on the span of the modes (Layer III),
# and Q†ψ = R c has the norm of ψ.
# The arrays have the type of the model's arrays, for example GPU arrays.
struct EigenBasis{T <: Number, VT <: AbstractVector{T}, MT <: AbstractMatrix{T}}
    λ::VT
    V::MT
    Q::MT
    R::UpperTriangular{T, MT}
    G::MT                      # V†V = R†R, shared by survival and its derivative
end

function EigenBasis(λ::AbstractVector, V::AbstractMatrix)
    F = qr(V)
    Q = lmul!(F.Q, _identity_like(V))   # the Q factor, with the array type of V
    R = UpperTriangular(F.R)
    return EigenBasis(λ, V, Q, R, R' * R)
end

# Identity matrix with the size and array type of `A`.
function _identity_like(A::AbstractMatrix)
    id = fill!(similar(A), 0)
    view(id, diagind(id)) .= 1
    return id
end

Base.length(basis::EigenBasis) = length(basis.λ)

# Coordinates c with V c = ψ (the least-squares fit if ψ is outside the span of V).
_coordinates!(c, basis::EigenBasis, ψ) = ldiv!(basis.R, mul!(c, basis.Q', ψ))

# The state V c or, when `coordinates`, its coordinates Q†ψ = R c followed by zeros.
function _write_state!(out, basis::EigenBasis, c, coordinates::Bool)
    coordinates || return mul!(out, basis.V, c)
    n = length(basis)
    mul!(view(out, 1:n), basis.R, c)
    fill!(view(out, (n + 1):length(out)), 0)
    return out
end

_effective_hamiltonian(H, c_ops) = H - im * sum(C' * C for C in c_ops) / 2

# All the modes of H_eff (paper Eqs. 15–16). `eigen` runs where the data of H_eff
# lives, for example on the GPU.
function _full_basis(Heff::AbstractMatrix)
    F = eigen(to_dense(Heff))
    basis = EigenBasis(-im .* F.values, F.vectors)
    # The coordinates of a state lose at most log10(κ) digits, κ being the condition
    # number of V (estimated on the CPU, where it is cheap); physical states usually
    # lose far fewer. Near an exceptional point κ grows, up to κ ≈ 1/ε when H_eff has
    # no eigenbasis at all.
    κ = cond(UpperTriangular(collect(basis.R.data)), 1)
    ϵ = eps(real(eltype(basis.λ)))
    κ < 1 / ϵ || throw(
        ArgumentError(
            "the effective Hamiltonian is not diagonalizable: its eigenvectors have condition \
            number ≈ $(round(κ; sigdigits = 2)), as at an exceptional point"
        )
    )
    κ < 1 / sqrt(ϵ) || @warn "The eigenvectors of the effective Hamiltonian have condition number \
        ≈ $(round(κ; sigdigits = 2)), so the propagation can lose up to $(round(Int, log10(κ))) digits."
    return basis
end

# Layer III (paper Eq. 20): the `m` slowest modes, plus any mode degenerate with them.
# The modes are selected on the CPU, then taken from the arrays of `full`.
function _slow_basis(full::EigenBasis, m::Integer)
    1 <= m <= length(full) || throw(ArgumentError("layer3_sizes must be in 1:$(length(full)), got $m"))
    λ = collect(full.λ)
    order = sortperm(λ; by = x -> -real(x))   # slowest decay first
    slow = order[1:m]
    rtol = sqrt(eps(real(eltype(λ))))
    degenerate = filter(j -> any(i -> isapprox(λ[j], λ[i]; rtol), slow), order[(m + 1):end])
    modes = sort!(vcat(slow, degenerate))
    return EigenBasis(full.λ[modes], full.V[:, modes])
end

_layer3_sizes(m::Integer, ngauges) = fill(m, ngauges)
function _layer3_sizes(m::AbstractVector{<:Integer}, ngauges)
    length(m) == ngauges || throw(ArgumentError("layer3_sizes must have one entry per gauge ($ngauges)"))
    return m
end

# Layer III (paper Eq. 21): whether the relative residual ‖ψ - QQ†ψ‖/‖ψ‖ of ψ on the
# slow modes `basis` is at most `tolerance`. If so, `q` holds the coordinates Q†ψ of the
# projection, rescaled to the norm of ψ.
function _project!(q, basis::EigenBasis, ψ, tolerance, residual)
    mul!(q, basis.Q', ψ)
    mul!(copyto!(residual, ψ), basis.Q, q, -1, 1)
    nψ = norm(ψ)
    norm(residual) <= tolerance * nψ || return false
    rmul!(q, nψ / norm(q))
    return true
end

@doc raw"""
    GaugeEigenExponential(H, c_ops)

Exact propagator for the no-jump evolution of `mcsolve` with a time-independent
Hamiltonian `H` and collapse operators `c_ops`:

```math
|\psi(t + \tau)\rangle = e^{-i H_{\rm eff} \tau} |\psi(t)\rangle
= V e^{\Lambda \tau} V^{-1} |\psi(t)\rangle,
\qquad H_{\rm eff} = H - \frac{i}{2} \sum_\mu C_\mu^\dagger C_\mu .
```

The decomposition ``-i H_{\rm eff} = V \Lambda V^{-1}`` is computed once. Each step,
and each evaluation of the dense output used to locate the jumps and to save
results, is exact, so steps can be as long as the time between jumps. The arrays
follow those of `H` and `c_ops`: with GPU operators, for example, the decomposition
and the propagation run on the GPU.

Pass it to `mcsolve` together with the same `H` and `c_ops`, which it also uses for
the jumps, and with the times of `tlist` as stops. `mcsolve` computes the expectation
values at these times after the jumps of their step, which can change the basis of the
dense output, so they must be step ends. The stops also shorten the search for each jump:

```julia
mcsolve(H, ψ0, tlist, c_ops; alg = GaugeEigenExponential(H, c_ops), tstops = tlist)
```

[`dislou_solve`](@ref) builds it with one eigenbasis per gauge, plus the slow modes of
Layer III when requested. Each trajectory then switches gauge and basis at its jumps.

!!! note
    Jump location and saving are supported only during integration. Post-solve
    interpolation of a direct ODE solution, such as `sol(t)` between saved times,
    is unsupported. Request the required times through `saveat`.

!!! warning
    The algorithm never evaluates the ODE function. It must be built from the same
    `H` and `c_ops` that are given to `mcsolve`.
"""
struct GaugeEigenExponential{B <: EigenBasis, S, TC} <: OrdinaryDiffEqCore.OrdinaryDiffEqLinearExponentialAlgorithm
    bases::Vector{B}             # all modes, one basis per gauge (Layer II)
    slow::Vector{S}              # slowest modes, one `SlowModes` per gauge (Layer III), or empty
    C::Vector{TC}                # collapse operators C_μ
    shifts::Matrix{ComplexF64}   # ζ[μ, g]: gauge g jumps with C_μ + ζ[μ, g] (Layer I)
    hysteresis::Float64          # η
    residual_tolerance::Float64  # Layer III r_tol
end

GaugeEigenExponential(H::QuantumObject, c_ops) =
    GaugeEigenExponential(H, c_ops, zeros(ComplexF64, length(c_ops), 1))

# One basis per gauge (column of `shifts`) and, for Layer III, `layer3_sizes` slow modes
# per gauge, with their matrices for the jumps and for the expectation values of `e_ops`.
function GaugeEigenExponential(
        H::QuantumObject, c_ops, shifts::AbstractMatrix;
        hysteresis = 1, layer3_sizes = nothing, residual_tolerance = 1.0e-3, e_ops = ()
    )
    gauges = [_shifted_operators(H, c_ops, ζ) for ζ in eachcol(shifts)]
    bases = [_full_basis(_effective_hamiltonian(Hg, Cg).data) for (Hg, Cg) in gauges]
    C = [op.data for op in c_ops]
    slow = layer3_sizes === nothing ? SlowModes{eltype(bases), typeof(first(bases).V)}[] :
        _slow_modes(bases, _layer3_sizes(layer3_sizes, length(bases)), C, shifts, e_ops)
    return GaugeEigenExponential(
        bases, slow, C, Matrix{ComplexF64}(shifts), float(hysteresis), float(residual_tolerance)
    )
end

function Base.show(io::IO, alg::GaugeEigenExponential)
    print(io, "GaugeEigenExponential(", length(alg.bases), " gauge(s), ", length(first(alg.bases)), " modes")
    isempty(alg.slow) || print(io, ", Layer III modes ", [length(s.basis) for s in alg.slow])
    return print(io, ")")
end

# Per-trajectory state: the gauge, and the basis of the current step.
OrdinaryDiffEqCore.@cache mutable struct GaugeEigenExponentialCache{uType, B <: EigenBasis, tType, rType} <: OrdinaryDiffEqCore.OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType        # scratch lent to callbacks (`get_tmp_cache`)
    c::uType          # coordinates of `uprev` in `basis`
    c_next::uType     # coordinates of `u`, the end of the step
    c_scratch::uType
    ulast::uType      # `u` as this algorithm left it, to detect changes made by callbacks
    basis::B          # basis of the current step
    activities::Vector{Float64}   # activity of each gauge
    gauge::Int
    reduced::Bool     # whether `basis` holds the slow modes of the gauge (Layer III)
    coordinates::Bool # whether `u` holds the coordinates Q†ψ on these modes instead of ψ
    condition_time::tType
    survival::rType
    rate::rType
    condition_valid::Bool
end

function OrdinaryDiffEqCore.alg_cache(
        alg::GaugeEigenExponential, u, rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t, dt,
        reltol, p, calck, ::Val{true}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    return GaugeEigenExponentialCache(
        u, uprev, zero(u), zero(u), zero(u), zero(u), zero(u),
        first(alg.bases), zeros(length(alg.bases)), 1, false, false,
        t, zero(real(eltype(u))), zero(real(eltype(u))), false
    )
end

OrdinaryDiffEqCore.alg_order(::GaugeEigenExponential) = 1
OrdinaryDiffEqCore.isfsal(::GaugeEigenExponential) = false
OrdinaryDiffEqCore.dt_required(::GaugeEigenExponential) = false   # each step goes to the next stop time
OrdinaryDiffEqCore.get_fsalfirstlast(::GaugeEigenExponentialCache, u) = (nothing, nothing)

# The trajectory starts in the least active gauge (paper Eq. 12).
function OrdinaryDiffEqCore.initialize!(integrator, cache::GaugeEigenExponentialCache)
    integrator.kshortsize = 0   # the dense output needs no stage derivatives
    resize!(integrator.k, 0)
    (; alg, u) = integrator
    A = _gauge_activities!(cache.activities, alg.C, alg.shifts, u, cache.tmp)
    _enter_gauge!(cache, alg, argmin(A), u)
    copyto!(cache.ulast, u)
    return nothing
end

# The gauge after a jump, or after another change of the state ψ (paper Eq. 13).
function _switch_gauge!(cache, alg, ψ)
    A = _gauge_activities!(cache.activities, alg.C, alg.shifts, ψ, cache.tmp)
    return _enter_gauge!(cache, alg, _next_gauge(A, cache.gauge, alg.hysteresis), ψ)
end

# Move to the gauge `g` at the state ψ: on its slow modes if ψ is close to them (Layer
# III), otherwise on all its modes.
function _enter_gauge!(cache, alg, g, ψ)
    cache.coordinates = false
    if !isempty(alg.slow)
        basis = alg.slow[g].basis
        q = view(cache.c_scratch, 1:length(basis))
        if _project!(q, basis, ψ, alg.residual_tolerance, cache.tmp)
            return _enter_slow_modes!(cache, alg, g, q)
        end
    end
    return _enter_all_modes!(cache, alg, g, ψ)
end

# The next step starts on the slow modes of the gauge `g`, from the coordinates q = Q†ψ.
function _enter_slow_modes!(cache, alg, g, q)
    cache.condition_valid = false
    basis = alg.slow[g].basis
    ldiv!(view(cache.c_next, 1:length(basis)), basis.R, q)
    cache.gauge = g
    cache.basis = basis
    cache.reduced = true
    return nothing
end

# The next step starts on all the modes of the gauge `g`, from the state ψ.
function _enter_all_modes!(cache, alg, g, ψ)
    cache.condition_valid = false
    basis = alg.bases[g]
    _coordinates!(view(cache.c_next, 1:length(basis)), basis, ψ)
    cache.gauge = g
    cache.basis = basis
    cache.reduced = false
    return nothing
end

function OrdinaryDiffEqCore.perform_step!(integrator, cache::GaugeEigenExponentialCache, repeat_step = false)
    cache.condition_valid = false
    (; alg, dt, uprev, u) = integrator
    uprev == cache.ulast || _switch_gauge!(cache, alg, uprev)   # changed by a callback
    basis = cache.basis
    c = view(cache.c, 1:length(basis))
    c_next = view(cache.c_next, 1:length(basis))
    copyto!(c, c_next)
    # On the slow modes, the state is stored as its coordinates, except where it is saved.
    cache.coordinates = cache.reduced && !_saves_step_end(integrator)
    @. c_next = exp(basis.λ * dt) * c
    _write_state!(u, basis, c_next, cache.coordinates)
    copyto!(cache.ulast, u)
    return nothing
end

# Whether the state at the end of the step is saved, by `saveat`, `save_end` or `save_everystep`.
function _saves_step_end(integrator)
    opts = integrator.opts
    opts.save_on || return false
    opts.save_everystep && return true
    t = integrator.t + integrator.dt
    saved(s) = abs(s - t) <= 100 * eps(abs(s))
    return (!isempty(opts.saveat) && saved(integrator.tdir * first(opts.saveat))) ||
        (opts.save_end && saved(last(integrator.sol.prob.tspan)))
end

# Exact dense output of the current step: ψ(t + Θ dt) = V e^{Λ Θ dt} c, or its time
# derivative, stored as the end of the step is.
function _dense_output!(out, Θ, dt, y₀, cache, derivative::Bool)
    y₀ === cache.uprev || throw(ArgumentError("GaugeEigenExponential can only interpolate within the current step"))
    basis = cache.basis
    c = view(cache.c, 1:length(basis))
    w = view(cache.c_scratch, 1:length(basis))
    if derivative
        @. w = basis.λ * exp(basis.λ * (Θ * dt)) * c
    else
        @. w = exp(basis.λ * (Θ * dt)) * c
    end
    return _write_state!(out, basis, w, cache.coordinates)
end

function OrdinaryDiffEqCore._ode_interpolant!(
        out, Θ, dt, y₀, y₁, k, cache::GaugeEigenExponentialCache, idxs::Nothing,
        ::Type{Val{0}}, differential_vars
    )
    return _dense_output!(out, Θ, dt, y₀, cache, false)
end

function OrdinaryDiffEqCore._ode_interpolant!(
        out, Θ, dt, y₀, y₁, k, cache::GaugeEigenExponentialCache, idxs::Nothing,
        ::Type{Val{1}}, differential_vars
    )
    return _dense_output!(out, Θ, dt, y₀, cache, true)
end

OrdinaryDiffEqCore._ode_addsteps!(
    k, t, uprev, u, dt, f, p, cache::GaugeEigenExponentialCache,
    always_calc_begin = false, allow_calc_end = true, force_calc_end = false
) = nothing
