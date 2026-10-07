# Gauge discovery from stable semiclassical fixed points (paper Appendix A.1).

const _SEMICLASSICAL_POLICY = (
    starts_per_coordinate = 5,
    abstol = 1.0e-10,
    maxiters = 80,
    stability_cutoff = -1.0e-8,
    occupation_slack = 1.0e-8,
    deduplication_tolerance = 1.0e-7,
)

# The unknowns are real, u = (Re α₁, …, Re α_m, Im α₁, …, Im α_m): the drift depends on α and
# α*, so Newton's method needs the real 2m × 2m Jacobian and not a complex one.
_amplitudes(u) = complex.(view(u, 1:(length(u) ÷ 2)), view(u, (length(u) ÷ 2 + 1):length(u)))

# Paper: the coherent product state |α⟩ truncated to the Fock cutoffs `dims`.
_coherent_product(dims, α) = reduce(kron, map((d, αⱼ) -> exp(-abs2(αⱼ) / 2) .* cumprod(vcat(one(αⱼ), αⱼ ./ sqrt.(1:(d - 1)))), dims, α))

# Paper: L†(a_j) = i[H, a_j] + Σ_μ (C_μ† a_j C_μ - {C_μ†C_μ, a_j}/2), whose expectation is d⟨a_j⟩/dt.
_heisenberg_drift(H, c_ops, a) = im * commutator(H, a) +
    sum(C' * a * C - commutator(C' * C, a; anti = true) / 2 for C in c_ops)

# Paper: (Re F, Im F) with F_j(α) = ⟨α|L†(a_j)|α⟩ = dα_j/dt (Eqs. A.1–A.3). The parameters `p`
# hold the Fock cutoffs `dims` and the matrices `drifts` of L†(a_j).
function _meanfield_drift(u, p)
    ψ = _coherent_product(p.dims, _amplitudes(u))
    F = [dot(ψ, D * ψ) for D in p.drifts]
    return [real(F); imag(F)]
end

# Occupations rounded, because symmetric branches tie up to rounding errors.
function _root_key(u)
    α = _amplitudes(u)
    return [round.(abs2.(α); digits = 6); real(α); imag(α)]
end

# The n^length(radii) points of a regular grid with n points per coordinate in the box
# |u_k| ≤ radii[k]. The digits of s in base n are the grid indices of its s-th point.
function _grid(radii, n)
    axis = range(-1, 1; length = n)
    return [radii .* axis[digits(s; base = n, pad = length(radii)) .+ 1] for s in 0:(n^length(radii) - 1)]
end

# Paper: the fixed points x^(g) of the drift inside the occupation bounds (Appendix A.1), found
# by a trust-region Newton method from a regular grid of starting points.
function _fixed_points(p, limits, ensemblealg)
    policy = _SEMICLASSICAL_POLICY
    seeds = _grid(repeat(sqrt.(limits); outer = 2), policy.starts_per_coordinate)

    problem = SciMLBase.NonlinearProblem{false}(_meanfield_drift, first(seeds), p)
    ensemble = SciMLBase.EnsembleProblem(problem; prob_func = (prob, ctx) -> SciMLBase.remake(prob; u0 = seeds[ctx.sim_id]))
    solutions = SciMLBase.solve(
        ensemble, SimpleTrustRegion(), ensemblealg; trajectories = length(seeds),
        abstol = policy.abstol, maxiters = policy.maxiters,
    )

    # Typed, because the element type of an ensemble solution is not inferred (it depends on `prob_func`).
    converged = Vector{Float64}[solution.u for solution in solutions.u if SciMLBase.successful_retcode(solution)]
    roots = Vector{Float64}[]
    for u in converged
        inside = all(abs2.(_amplitudes(u)) .<= limits .+ policy.occupation_slack)
        inside && !any(v -> norm(u - v) <= policy.deduplication_tolerance, roots) && push!(roots, u)
    end
    return sort!(roots; by = _root_key)
end

# Paper: max_ℓ Re λ_ℓ of the Jacobian of the drift, which decides the stability (Eq. A.4).
function _stability(u, p)
    # ForwardDiff chooses its chunk size from length(u) at run time, so its result type is not inferred.
    jacobian = ForwardDiff.jacobian(Base.Fix2(_meanfield_drift, p), u)::Matrix{Float64}
    λmax = maximum(real, eigvals(jacobian))
    return (; stable = λmax < _SEMICLASSICAL_POLICY.stability_cutoff, max_real_eigenvalue = λmax, residual = norm(_meanfield_drift(u, p)))
end

_amplitude_matrix(roots, nmodes) = ComplexF64[u[mode] + im * u[nmodes + mode] for mode in 1:nmodes, u in roots]

# Paper: ζ_μ^(g) = -C_{μ,sc}(α^(g), α^(g)*) (Eq. A.5), from the stable fixed points of the drift.
function _discover_gauges(
        ::Val{:semiclassical}, H, c_ops;
        mode_ops, mode_dims, limits, ensemblealg::EnsembleAlgorithm = EnsembleThreads()
    )
    _check_operators(H, c_ops, mode_ops, mode_dims)
    nmodes = length(mode_ops)
    limits isa Union{Tuple, AbstractVector} && length(limits) == nmodes && all(b -> b isa Real && isfinite(b) && b > 0, limits) ||
        throw(ArgumentError("limits must be a tuple or vector with one finite positive occupation bound per mode, got $limits"))

    p = (; dims = collect(Int, mode_dims), drifts = [_heisenberg_drift(H, c_ops, a).data for a in mode_ops])
    roots = _fixed_points(p, collect(Float64, limits), ensemblealg)
    stability = [_stability(u, p) for u in roots]
    isstable = [s.stable for s in stability]
    stable = roots[isstable]
    isempty(stable) && throw(ArgumentError("semiclassical discovery found no stable fixed points"))

    ψs = [_coherent_product(p.dims, _amplitudes(u)) for u in stable]
    shifts = ComplexF64[-dot(ψ, C.data * ψ) for C in c_ops, ψ in ψs]
    diagnostics = (;
        roots = _amplitude_matrix(roots, nmodes), stability,
        rejected = _amplitude_matrix(roots[.!isstable], nmodes), stable = trues(length(stable)),
    )
    return (;
        shifts, method = :semiclassical, centers = _amplitude_matrix(stable, nmodes),
        weights = fill(1 / length(stable), length(stable)), diagnostics,
    )
end
