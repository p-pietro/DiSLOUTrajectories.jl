# Layers II and III: the no-jump state evolves exactly in the eigenbasis of the
# effective Hamiltonian. This is written as an OrdinaryDiffEq algorithm, so that
# `mcsolve` keeps handling jumps, saving and ensembles.

# Modes of the no-jump generator L = -i H_eff: eigenvalues `λ` and right
# eigenvectors `V` (columns), either all of them (Layer II) or only the slowest
# ones (Layer III). The factorization V = Q R gives the coordinates of a state.
struct EigenBasis
    λ::Vector{ComplexF64}
    V::Matrix{ComplexF64}
    Q::Matrix{ComplexF64}
    R::UpperTriangular{ComplexF64, Matrix{ComplexF64}}
end

function EigenBasis(λ::AbstractVector, V::AbstractMatrix)
    F = qr(V)
    return EigenBasis(λ, V, Matrix(F.Q), UpperTriangular(Matrix(F.R)))
end

Base.length(basis::EigenBasis) = length(basis.λ)

# Coordinates c with V c = ψ (the least-squares fit if ψ is outside the span of V).
_coordinates!(c, basis::EigenBasis, ψ) = ldiv!(basis.R, mul!(c, basis.Q', ψ))

_effective_hamiltonian(H, c_ops) = H - (im / 2) * sum(C' * C for C in c_ops)

# All the modes of H_eff (paper Eqs. 15–16).
function _full_basis(Heff)
    E, V = _eigen_decomposition(Matrix{ComplexF64}(Heff))
    basis = EigenBasis(-im .* E, V)
    κ = cond(basis.R, 1)
    κ < 1.0e8 || @warn "The eigenvectors of the effective Hamiltonian are ill-conditioned \
        (condition number ≈ $(round(κ; sigdigits = 2))), so the propagation loses accuracy." maxlog = 1
    return basis
end

# Layer III (paper Eq. 20): the `m` slowest modes, plus any mode degenerate with them.
function _slow_basis(full::EigenBasis, m::Integer)
    1 <= m <= length(full) || throw(ArgumentError("layer3_sizes must be in 1:$(length(full)), got $m"))
    λ = full.λ
    order = sortperm(λ; by = x -> -real(x))   # slowest decay first
    slow = order[1:m]
    degenerate = filter(j -> any(i -> isapprox(λ[j], λ[i]; rtol = sqrt(eps())), slow), order[(m + 1):end])
    modes = sort!(vcat(slow, degenerate))
    return EigenBasis(λ[modes], full.V[:, modes])
end

_layer3_sizes(m::Integer, ngauges) = fill(m, ngauges)
function _layer3_sizes(m::AbstractVector{<:Integer}, ngauges)
    length(m) == ngauges || throw(ArgumentError("layer3_sizes must have one entry per gauge ($ngauges)"))
    return m
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
results, is exact, so steps can be as long as the time between jumps.

Pass it to `mcsolve` together with the same `H` and `c_ops`. Stopping at the times of
`tlist` keeps the search for each jump time within one interval:

```julia
mcsolve(H, ψ0, tlist, c_ops; alg = GaugeEigenExponential(H, c_ops), tstops = tlist)
```

[`dislou_solve`](@ref) builds it with one eigenbasis per gauge, plus the slow-mode
bases of Layer III when requested. Each trajectory then switches basis after its jumps.

!!! warning
    The algorithm never evaluates the ODE function. It must be built from the same
    `H` and `c_ops` that are given to `mcsolve`.
"""
struct GaugeEigenExponential <: OrdinaryDiffEqCore.OrdinaryDiffEqLinearExponentialAlgorithm
    bases::Vector{EigenBasis}           # all modes, one basis per gauge (Layer II)
    reduced_bases::Vector{EigenBasis}   # slowest modes, one basis per gauge (Layer III), or empty
end

GaugeEigenExponential(H::QuantumObject, c_ops) = GaugeEigenExponential([(H, c_ops)])

# One basis per gauge `(H_g, C_g)`, and `layer3_sizes` slow modes per gauge for Layer III.
function GaugeEigenExponential(gauges::AbstractVector; layer3_sizes = nothing)
    bases = [_full_basis(_effective_hamiltonian(Hg, Cg).data) for (Hg, Cg) in gauges]
    reduced_bases = layer3_sizes === nothing ? EigenBasis[] :
        map(_slow_basis, bases, _layer3_sizes(layer3_sizes, length(bases)))
    return GaugeEigenExponential(bases, reduced_bases)
end

function Base.show(io::IO, alg::GaugeEigenExponential)
    print(io, "GaugeEigenExponential(", length(alg.bases), " gauge(s), ", length(first(alg.bases)), " modes")
    isempty(alg.reduced_bases) || print(io, ", Layer III modes ", length.(alg.reduced_bases))
    return print(io, ")")
end

# Per-trajectory state. `gauge` and `reduced` select the active basis; the gauge
# router changes them after jumps.
OrdinaryDiffEqCore.@cache mutable struct GaugeEigenExponentialCache{uType} <: OrdinaryDiffEqCore.OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    tmp::uType        # scratch lent to callbacks (`get_tmp_cache`)
    c::uType          # coordinates of `uprev` in `basis`
    c_next::uType     # coordinates of `u`, the end of the step
    c_scratch::uType
    ulast::uType      # `u` as this algorithm left it, to detect changes made by callbacks
    basis::EigenBasis # basis of the current step
    gauge::Int
    reduced::Bool     # whether the gauge uses its Layer III basis
end

_active_basis(alg::GaugeEigenExponential, cache) =
    cache.reduced ? alg.reduced_bases[cache.gauge] : alg.bases[cache.gauge]

function OrdinaryDiffEqCore.alg_cache(
        alg::GaugeEigenExponential, u, rate_prototype, ::Type{uEltypeNoUnits},
        ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t, dt,
        reltol, p, calck, ::Val{true}, verbose
    ) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    ulast = fill!(similar(u), NaN)   # matches no state, so the first step computes coordinates
    return GaugeEigenExponentialCache(
        u, uprev, zero(u), zero(u), zero(u), zero(u), ulast, first(alg.bases), 1, false
    )
end

OrdinaryDiffEqCore.alg_order(::GaugeEigenExponential) = 1
OrdinaryDiffEqCore.isfsal(::GaugeEigenExponential) = false
OrdinaryDiffEqCore.dt_required(::GaugeEigenExponential) = false   # each step goes to the next stop time
OrdinaryDiffEqCore.get_fsalfirstlast(::GaugeEigenExponentialCache, u) = (nothing, nothing)

function OrdinaryDiffEqCore.initialize!(integrator, cache::GaugeEigenExponentialCache)
    integrator.kshortsize = 0   # the dense output needs no stage derivatives
    resize!(integrator.k, 0)
    return nothing
end

function OrdinaryDiffEqCore.perform_step!(integrator, cache::GaugeEigenExponentialCache, repeat_step = false)
    (; dt, uprev, u) = integrator
    basis = _active_basis(integrator.alg, cache)
    c = view(cache.c, 1:length(basis))
    c_next = view(cache.c_next, 1:length(basis))
    # The coordinates reached by the previous step are still valid, unless a
    # callback changed the state (a jump) or the basis (a gauge switch).
    if basis === cache.basis && uprev == cache.ulast
        copyto!(c, c_next)
    else
        _coordinates!(c, basis, uprev)
        cache.basis = basis
    end
    @. c_next = exp(basis.λ * dt) * c
    mul!(u, basis.V, c_next)
    copyto!(cache.ulast, u)
    return nothing
end

# Exact dense output of the current step: ψ(t + Θ dt) = V e^{Λ Θ dt} c, or its time derivative.
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
    return mul!(out, basis.V, w)
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
