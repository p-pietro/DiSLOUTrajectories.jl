# Gauge discovery: shifts ζ_μ^(g) from clusters of preliminary trajectories
# (paper Appendix A.2) or from stable semiclassical fixed points (Appendix A.1).

"""
    discover_gauges(H, c_ops; <keyword arguments>)

Discover the paper's shifts `ζ_μ^(g)` (Eq. 9) for [`dislou_solve`](@ref).

The default `method=:trajectories` clusters terminal amplitudes from preliminary
quantum trajectories. With `method=:semiclassical`, it instead finds the stable
fixed points of the mean-field equations of the bosonic modes.

See also [`dislou_solve`](@ref).

# Examples

Find the single vacuum gauge of a damped mode with trajectory discovery:

```jldoctest
julia> using DiSLOUTrajectories, QuantumToolbox

julia> a = destroy(2);

julia> gauges = discover_gauges(0.0 * a, [a];
           mode_ops = [a], mode_dims = [2], discovery_time = 1.0,
           seed_radii = [0.0], cluster_scales = [1.0], step = 1.0,
           nseeds = 2, min_neighbors = 1, min_weight = 0.0);

julia> (gauges.method, size(gauges.shifts), all(iszero, gauges.shifts))
(:trajectories, (1, 1), true)
```

Find the same vacuum fixed point from the mean-field equations:

```jldoctest
julia> using DiSLOUTrajectories, QuantumToolbox

julia> a = destroy(6);

julia> gauges = discover_gauges(a' * a, [a];
           method = :semiclassical, mode_ops = [a], mode_dims = [6], limits = (1.0,));

julia> (gauges.method, size(gauges.shifts), all(z -> abs(z) < 1e-9, gauges.shifts))
(:semiclassical, (1, 1), true)
```

# Extended help

Trajectory discovery starts in random coherent product states. Each retained
cluster defines shifts from the negative cluster averages of the physical
collapse-operator expectations. Semiclassical discovery solves the first-order
mean-field equations `dα_j/dt = ⟨α|L†(a_j)|α⟩ = 0` for the amplitudes `α` of the modes,
where `|α⟩` is a coherent product state and `L†(a_j) = i[H, a_j] + Σ_μ (C_μ† a_j C_μ -
{C_μ†C_μ, a_j} / 2)` is the Heisenberg-picture generator. It keeps the stable solutions
and evaluates the collapse-operator expectations `⟨α|C_μ|α⟩` at them. The operators
are the truncated matrices of `H` and `c_ops`, so the occupations of the solutions must
stay well below the Fock cutoffs.

# Arguments

The accepted arguments depend on `method`. The trajectory method is the default;
the semiclassical method must be selected explicitly.

## Trajectory discovery

- `H`: Time-independent Hermitian Hamiltonian as a
  `QuantumToolbox.QuantumObject` operator. Its tensor dimensions must match
  `mode_dims`. This method uses QuantumToolbox's `mcsolve` for the preliminary
  trajectories.
- `c_ops`: Vector of time-independent `QuantumToolbox.QuantumObject` collapse
  operators with the same tensor dimensions as `H`. Include the square root of
  each physical decay rate in the corresponding operator.
- `method::Symbol`: Set to `:trajectories` (the default).
- `mode_ops`: Nonempty collection of bosonic annihilation operators, embedded
  in the full tensor-product Hilbert space. Each operator must have the same
  dimensions as `H`, and `mode_ops[j]` must act on the `j`-th tensor factor. Their
  order defines the rows of the returned centers.
- `mode_dims`: One integer Hilbert-space truncation dimension per mode, in
  tensor-product order. Together they must span the whole space of `H`.
- `discovery_time`: Positive, finite duration of each preliminary trajectory,
  which starts at time zero.
- `seed_radii`: One finite, nonnegative coherent-amplitude radius per mode.
  Initial amplitudes are sampled uniformly in each complex disk.
- `cluster_scales`: Vector of positive, finite amplitude scales, one per mode.
  Real and imaginary parts of each terminal mode mean are divided by its
  scale before Euclidean DBSCAN clustering.
- `step`: Positive, finite spacing of the preliminary observation grid.
  Defaults to `discovery_time / 40`. The endpoint `discovery_time` is always
  included.
- `nseeds::Int`: Positive number of initial coherent product states, with one
  trajectory per state. Defaults to `600`.
- `terminal_window`: Finite, nonnegative duration at the end of the run over
  which sampled mode, occupation, and collapse expectations are averaged.
  Defaults to `0.0`, which uses the final sample. Values greater than
  `discovery_time` use the whole observation grid.
- `preliminary_shifts`: Optional vector of finite complex shifts, one per
  collapse channel, applied during the preliminary runs with a compensating
  Hamiltonian shift. Defaults to `nothing`, meaning zero shifts. Returned
  gauges use expectations of the original physical collapse operators.
- `dbscan_radius`: Positive, finite DBSCAN neighborhood radius in the scaled
  real-imaginary feature space. Defaults to `1.5`.
- `min_neighbors::Int`: Positive DBSCAN core-point threshold: the number of points
  within `dbscan_radius`, the point itself included. Defaults to `10`.
- `min_weight`: Minimum retained cluster population as a fraction of all
  `nseeds` samples. Must be finite and in `[0, 1)`; defaults to `0.02`.
- `rng::AbstractRNG`: Random number generator for the initial amplitudes and
  the preliminary trajectory streams. Defaults to `Random.default_rng()`.
- `save_preliminary_trajectories::Int`: Nonnegative number of preliminary
  trajectories whose state, expectation, and jump traces are retained in the
  diagnostics. Defaults to `0`; values above `nseeds` retain all trajectories.
- `ensemblealg`: How the preliminary trajectories run: `EnsembleThreads()` (default),
  `EnsembleSerial()`, or `EnsembleDistributed()`. Distributed execution requires
  worker processes with DiSLOUTrajectories and QuantumToolbox loaded.

## Semiclassical discovery

- `H`, `c_ops`, `mode_ops`, `mode_dims`: As for trajectory discovery. The collapse
  operators may be nonlinear in the modes, such as `a^2` or `a' * a`.
- `method::Symbol`: Set to `:semiclassical`.
- `limits`: Nonempty tuple or vector of positive, finite occupation bounds,
  one per mode. Accepted fixed points satisfy `abs2(α[m]) ≤ limits[m]` up to
  numerical tolerance. These bounds define the search region, and they should
  stay a few standard deviations `√limits[m]` below `mode_dims[m]`; operators of
  higher degree in the modes need more margin.
- `ensemblealg`: How the Newton solves from the grid of starting points run:
  `EnsembleThreads()` (default), `EnsembleSerial()`, or `EnsembleDistributed()`.
  Distributed execution requires worker processes with DiSLOUTrajectories loaded.

# Notes

- The two methods have separate keyword sets. Arbitrary `mcsolve` or ODE
  solver options are not forwarded.
- Trajectory cluster weights are fractions of all preliminary samples,
  including discarded samples in the denominator. Their sum can be less than
  one. Clusters are ordered by decreasing weight, then by their centers.
  Noise points and samples from discarded clusters receive label `0`.
- Stability of the semiclassical solution is determined from the real
  mean-field Jacobian. The search starts from a regular grid of `5^(2m)` points
  for `m` modes, so it can miss roots and its cost grows rapidly with the number
  of modes. It uses the truncated matrices of `H` and `c_ops`, which reproduce
  the mean-field equations only while the occupations stay well below the cutoffs.
- Discovery raises an `ArgumentError` if no trajectory cluster or stable
  semiclassical fixed point is retained.

# Returns

- `gauges::NamedTuple`: `(; shifts, method, centers, weights, diagnostics)`.
  With `Nc` collapse channels, `Nm` modes, and `Ng` retained gauges:
  - `shifts::Matrix{ComplexF64}`: `Nc × Ng` shifts `ζ_μ^(g)`; column `g` defines
    `C_μ^(g) = C_μ + ζ_μ^(g) I` (Eq. 9) for each channel `μ`.
  - `method::Symbol`: `:trajectories` or `:semiclassical`.
  - `centers::Matrix{ComplexF64}`: `Nm × Ng` cluster-mean mode amplitudes
    or stable fixed-point amplitudes.
  - `weights::Vector{Float64}`: `Ng` cluster population fractions, as described
    above, or the uniform weights `1/Ng` of the semiclassical method.
  - `diagnostics::NamedTuple`: Method-specific discovery data described below.

For trajectory discovery, `diagnostics` contains:

- `counts` and `labels`: Cluster sizes of length `Ng` and assignments of
  length `nseeds`. Positive labels index columns of `centers`.
- `terminal_means` and `terminal_occupations`: `Nm × nseeds` terminal-window
  averages of mode amplitudes and occupations.
- `terminal_collapse_means`: `Nc × nseeds` terminal-window averages of the
  physical collapse-operator expectations.
- `times`: Preliminary observation grid, including zero and `discovery_time`.
- `preliminary_traces`: `(; indices, states, means, occupations)` for the
  retained preliminary runs. `indices` contains their one-based seed indices;
  the other fields are vectors of matrices. For `Nt = length(times)` and
  `N = prod(mode_dims)`, each state matrix is `N × Nt`, and each mode-mean
  or occupation matrix is `Nm × Nt`. These vectors are empty by default.
- `preliminary_jump_times` and `preliminary_jump_channels`: Vectors
  of event-time and one-based collapse-channel vectors for the retained runs.
- The resolved discovery settings: `mode_dims`, `discovery_time`, `seed_radii`,
  `cluster_scales`, `step`, `nseeds`, `terminal_window`, `preliminary_shifts`,
  `dbscan_radius`, `min_neighbors`, `min_weight`, `save_preliminary_trajectories`,
  and `ensemblealg`. The `save_preliminary_trajectories` field records the retained
  count, capped at `nseeds`.

For semiclassical discovery, `diagnostics` contains:

- `roots`: Matrix with one column per distinct fixed point found, including
  unstable points, and one row per mode.
- `stability`: One `(; stable, max_real_eigenvalue, residual)` record per
  column of `roots`. `max_real_eigenvalue` is the largest real part of the
  Jacobian eigenvalues, and `residual` is the norm of the mean-field drift.
- `rejected`: Matrix of the unstable fixed points excluded from `centers`.
- `stable`: Boolean vector of length `Ng`, with every entry `true`.
"""
discover_gauges(H, c_ops; method::Symbol = :trajectories, kwargs...) =
    _discover_gauges(Val(method), H, c_ops; kwargs...)

# Checks that H, the collapse operators and the bosonic mode operators (one per tensor
# factor, in order) live in the tensor product of Fock spaces of dimensions `mode_dims`.
function _check_operators(H, c_ops, mode_ops, mode_dims)
    nmodes = length(mode_ops)
    nmodes > 0 && length(mode_dims) == nmodes ||
        throw(DimensionMismatch("mode_ops needs at least one operator, with one entry of mode_dims each"))
    isempty(c_ops) && throw(ArgumentError("need at least one collapse operator"))
    dims = Tuple(Int.(mode_dims))
    has_dims(op) = size(op.data) == (prod(dims), prod(dims)) && Tuple(first(op.dims)) == dims && Tuple(last(op.dims)) == dims
    all(has_dims, (H, mode_ops..., c_ops...)) ||
        throw(DimensionMismatch("H, mode_ops and c_ops must have tensor dimensions mode_dims=$dims"))
    return nothing
end

function _discover_gauges(::Val{method}, args...; kwargs...) where {method}
    throw(ArgumentError("unknown gauge discovery method :$method; use :trajectories or :semiclassical"))
end
