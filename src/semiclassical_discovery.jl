# Gauge discovery from stable semiclassical fixed points (paper Appendix A.1).

const _SEMICLASSICAL_POLICY = (
    starts_per_coordinate = 5,
    max_iterations = 80,
    residual_tolerance = 1.0e-10,
    stability_cutoff = -1.0e-8,
    occupation_slack = 1.0e-8,
    deduplication_tolerance = 1.0e-7,
)

# Paper: |α⟩ = ⊗_j |α_j⟩ for x = (Re α₁, Im α₁, …), truncated to the Fock dimensions.
function _coherent_product(mode_dims, x)
    amplitudes = map(enumerate(mode_dims)) do (mode, dim)
        α = complex(x[2mode - 1], x[2mode])
        exp(-abs2(α) / 2) .* cumprod(vcat(one(α), α ./ sqrt.(1:(dim - 1))))
    end
    return reduce(kron, amplitudes)
end

# Paper: the Heisenberg-picture generator L†(a_j) = i[H, a_j] + Σ_μ (C_μ† a_j C_μ - ½{C_μ†C_μ, a_j}).
# Its expectation value is d⟨a_j⟩/dt.
_heisenberg_drift(H, c_ops, a) = im * commutator(H, a) +
    sum(C' * a * C - commutator(C' * C, a; anti = true) / 2 for C in c_ops)

# Paper: R(x) = (Re F₁, Im F₁, …) with F_j(α) = ⟨α|L†(a_j)|α⟩ (Eqs. A.1–A.3).
function _residual(x, mode_dims, drifts)
    ψ = _coherent_product(mode_dims, x)
    return [part for D in drifts for part in reim(dot(ψ, D * ψ))]
end

# Paper: x^(g) with R(x^(g)) = 0 (Eqs. A.2–A.3), approached by a trust-region Newton method.
function _newton_root(residual, seed)
    policy = _SEMICLASSICAL_POLICY
    problem = SciMLBase.NonlinearProblem{false}((x, _) -> residual(x), seed)
    return SciMLBase.solve(problem, SimpleTrustRegion(); abstol = policy.residual_tolerance, maxiters = policy.max_iterations).u
end

# Paper: candidate fixed points α^(g) inside the occupation bounds (Appendix A.1), found
# from a regular grid of starting points, with their stability (Eq. A.4).
function _phase_space_points(residual, bounds)
    policy = _SEMICLASSICAL_POLICY
    npoints = policy.starts_per_coordinate
    radii = repeat(sqrt.(bounds); inner = 2)
    grid = range(-1, 1; length = npoints)
    seeds = (radii .* grid[digits(s; base = npoints, pad = length(radii)) .+ 1] for s in 0:(npoints^length(radii) - 1))
    roots = Vector{Float64}[]
    for seed in seeds
        x = _newton_root(residual, seed)
        converged = norm(residual(x)) <= policy.residual_tolerance
        inside = all(m -> x[2m - 1]^2 + x[2m]^2 <= bounds[m] + policy.occupation_slack, eachindex(bounds))
        converged && inside && !any(y -> norm(x - y) <= policy.deduplication_tolerance, roots) && push!(roots, x)
    end

    points = map(roots) do x
        λmax = maximum(real, eigvals(ForwardDiff.jacobian(residual, x)::Matrix{Float64}))
        α = complex.(x[1:2:end], x[2:2:end])
        (; x, amplitudes = α, stable = λmax < policy.stability_cutoff, max_real_eigenvalue = λmax, residual = norm(residual(x)))
    end
    return sort!(points; by = point -> [round.(abs2.(point.amplitudes); digits = 6); real.(point.amplitudes); imag.(point.amplitudes)])
end

_amplitude_matrix(points, nmodes) = ComplexF64[point.amplitudes[mode] for mode in 1:nmodes, point in points]

# Paper: ζ_μ^(g) = -C_{μ,sc}(α^(g), α^(g)*) (Eq. A.5), from the fixed points of the mean-field drift.
function _discover_gauges(::Val{:semiclassical}, H, c_ops; mode_ops, mode_dims, limits)
    _check_operators(H, c_ops, mode_ops, mode_dims)
    nmodes = length(mode_ops)
    limits isa Union{Tuple, AbstractVector} && length(limits) == nmodes && all(b -> b isa Real && isfinite(b) && b > 0, limits) ||
        throw(ArgumentError("limits must be a tuple or vector with one finite positive occupation bound per mode, got $limits"))

    drifts = [_heisenberg_drift(H, c_ops, a).data for a in mode_ops]
    residual = x -> _residual(x, mode_dims, drifts)
    points = _phase_space_points(residual, collect(Float64, limits))
    stable = filter(point -> point.stable, points)
    isempty(stable) && throw(ArgumentError("semiclassical discovery found no stable fixed points"))

    ψs = [_coherent_product(mode_dims, point.x) for point in stable]
    shifts = ComplexF64[-dot(ψ, C.data * ψ) for C in c_ops, ψ in ψs]
    diagnostics = (;
        roots = _amplitude_matrix(points, nmodes),
        stability = [(; point.stable, point.max_real_eigenvalue, point.residual) for point in points],
        rejected = _amplitude_matrix(filter(point -> !point.stable, points), nmodes),
        stable = trues(length(stable)),
    )
    return (;
        shifts, method = :semiclassical, centers = _amplitude_matrix(stable, nmodes),
        weights = fill(1 / length(stable), length(stable)), diagnostics,
    )
end
