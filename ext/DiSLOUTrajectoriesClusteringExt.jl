module DiSLOUTrajectoriesClusteringExt

import Clustering
import DiSLOUTrajectories
import DiSLOUTrajectories: _run_preliminary_trajectories
import DiSLOUTrajectories: EnsembleAlgorithm, EnsembleSerial, EnsembleThreads, EnsembleDistributed
using Distances
using Distributed: nprocs
using Random: AbstractRNG, default_rng
using Statistics

const CF = ComplexF64

# Paper: rescaled (Re ᾱ_j^(r), Im ᾱ_j^(r)) (Appendix A.2).
function _scaled_features(points::AbstractMatrix{<:Complex}, scales::AbstractVector)
    nmodes, npoints = size(points)
    features = Matrix{Float64}(undef, 2nmodes, npoints)
    @inbounds for point in 1:npoints, mode in 1:nmodes
        z = points[mode, point] / scales[mode]
        features[2mode - 1, point] = real(z)
        features[2mode, point] = imag(z)
    end
    return features
end

_center_sort_key(center) = Tuple(Iterators.flatten((real(z), imag(z)) for z in center))

# Paper: clusters 𝓘_g and centers of ᾱ_j^(r) (Appendix A.2).
function _cluster_terminal_means(
        points::AbstractMatrix{<:Complex};
        cluster_scales::AbstractVector, dbscan_radius::Real,
        min_neighbors::Int, min_weight::Real
    )
    nmodes, npoints = size(points)
    length(cluster_scales) == nmodes || throw(DimensionMismatch("need one clustering scale per mode"))
    all(scale -> isfinite(scale) && scale > 0, cluster_scales) || throw(ArgumentError("cluster_scales must be finite and positive"))
    isfinite(dbscan_radius) && dbscan_radius > 0 || throw(ArgumentError("dbscan_radius must be finite and positive"))
    min_neighbors >= 1 || throw(ArgumentError("min_neighbors must be positive"))
    0 <= min_weight < 1 || throw(ArgumentError("min_weight must lie in [0, 1)"))
    all(isfinite, points) || throw(ArgumentError("terminal means must be finite"))
    npoints == 0 && return (; centers = zeros(CF, nmodes, 0), weights = Float64[], counts = Int[], labels = Int[])
    features = _scaled_features(points, cluster_scales)
    result = Clustering.dbscan(features, dbscan_radius; min_neighbors)
    labels = copy(Clustering.assignments(result))
    for (label, cluster) in enumerate(result.clusters), point in cluster.boundary_indices
        any(
            core -> evaluate(Euclidean(), @view(features[:, point]), @view(features[:, core])) <= dbscan_radius,
            cluster.core_indices
        ) || (labels[point] = 0)
    end
    records = NamedTuple[]
    for old in sort!(unique(filter(>(0), labels)))
        members = findall(==(old), labels)
        weight = length(members) / npoints
        weight < min_weight && continue
        center = CF.(vec(mean(@view(points[:, members]); dims = 2)))
        push!(records, (; old, count = length(members), weight, center))
    end
    sort!(records; by = record -> (-record.weight, _center_sort_key(record.center)))
    remap = Dict(record.old => index for (index, record) in enumerate(records))
    final_labels = [get(remap, label, 0) for label in labels]
    centers = isempty(records) ? zeros(CF, nmodes, 0) : hcat((copy(record.center) for record in records)...)
    return (; centers, weights = Float64[record.weight for record in records], counts = Int[record.count for record in records], labels = final_labels)
end

# Paper Eq. (A.7): ζ_μ^(g) from clusters of preliminary trajectories.
function DiSLOUTrajectories._discover_gauges(
        ::Val{:trajectories}, H, c_ops;
        mode_ops, mode_dims, discovery_time, seed_radii, cluster_scales,
        step = discovery_time / 40, nseeds::Int = 600, terminal_window = 0.0,
        preliminary_shifts = nothing, dbscan_radius = 1.5,
        min_neighbors::Int = 10, min_weight = 0.02, rng::AbstractRNG = default_rng(),
        save_preliminary_trajectories::Int = 0, ensemblealg::EnsembleAlgorithm = EnsembleThreads()
    )
    shifts = _check_discovery_inputs(
        H, c_ops, mode_ops, mode_dims, discovery_time, seed_radii, step, nseeds,
        terminal_window, preliminary_shifts, save_preliminary_trajectories, ensemblealg
    )
    # Checks the clustering options before any run.
    _cluster_terminal_means(zeros(ComplexF64, length(mode_ops), 0); cluster_scales, dbscan_radius, min_neighbors, min_weight)
    data = _run_preliminary_trajectories(
        H, c_ops, shifts, mode_ops, mode_dims, discovery_time, seed_radii, step, nseeds,
        terminal_window, rng, save_preliminary_trajectories, ensemblealg
    )
    clusters = _cluster_terminal_means(
        data.terminal_means;
        cluster_scales, dbscan_radius, min_neighbors, min_weight
    )
    isempty(clusters.counts) &&
        throw(ArgumentError("trajectory discovery found no retained DBSCAN clusters"))

    # ζ_μ^(g) = -⟨C_μ⟩, averaged over the trajectories of cluster g.
    gauges = axes(clusters.centers, 2)
    gauge_shifts = stack(-vec(mean(data.terminal_collapse_means[:, clusters.labels .== g]; dims = 2)) for g in gauges)
    diagnostics = (;
        counts = clusters.counts, labels = clusters.labels,
        data.terminal_means, data.terminal_occupations, data.terminal_collapse_means,
        times = data.tlist,
        preliminary_traces = (;
            indices = collect(1:data.nsave),
            states = data.states, means = data.traces, occupations = data.occupations,
        ),
        preliminary_jump_times = data.jump_times,
        preliminary_jump_channels = data.jump_channels,
        mode_dims = collect(Int, mode_dims), discovery_time = float(discovery_time),
        seed_radii = Float64.(seed_radii), cluster_scales = Float64.(cluster_scales),
        step = float(step), nseeds, terminal_window = float(terminal_window),
        preliminary_shifts = shifts, dbscan_radius = float(dbscan_radius),
        min_neighbors, min_weight = float(min_weight),
        save_preliminary_trajectories = data.nsave, ensemblealg,
    )
    return (;
        shifts = gauge_shifts, method = :trajectories, centers = clusters.centers,
        weights = clusters.weights, diagnostics,
    )
end

# Validate the inputs of trajectory discovery; returns the preliminary shifts.
function _check_discovery_inputs(
        H, c_ops, mode_ops, mode_dims, discovery_time, seed_radii, step, nseeds,
        terminal_window, preliminary_shifts, save_preliminary_trajectories, ensemblealg
    )
    nmodes = length(mode_ops)
    nmodes > 0 || throw(ArgumentError("need at least one mode operator"))
    length(mode_dims) == nmodes || throw(DimensionMismatch("need one subsystem dimension per mode"))
    all(>=(2), mode_dims) || throw(ArgumentError("mode dimensions must be at least 2"))
    length(seed_radii) == nmodes || throw(DimensionMismatch("need one seed radius per mode"))
    all(r -> isfinite(r) && r >= 0, seed_radii) ||
        throw(ArgumentError("seed_radii must be finite and nonnegative"))
    isfinite(discovery_time) && discovery_time > 0 ||
        throw(ArgumentError("discovery_time must be finite and positive"))
    isfinite(step) && step > 0 || throw(ArgumentError("step must be finite and positive"))
    isfinite(terminal_window) && terminal_window >= 0 ||
        throw(ArgumentError("terminal_window must be finite and nonnegative"))
    nseeds >= 1 || throw(ArgumentError("nseeds must be positive"))
    save_preliminary_trajectories >= 0 ||
        throw(ArgumentError("save_preliminary_trajectories must be nonnegative"))
    ensemblealg isa Union{EnsembleSerial, EnsembleThreads, EnsembleDistributed} ||
        throw(ArgumentError("ensemblealg must be EnsembleSerial(), EnsembleThreads(), or EnsembleDistributed()"))
    ensemblealg isa EnsembleDistributed && nprocs() <= 1 &&
        throw(ArgumentError("EnsembleDistributed() requires worker processes"))

    dims = Tuple(Int.(mode_dims))
    has_dims(op) = size(op.data) == (prod(dims), prod(dims)) &&
        Tuple(first(op.dims)) == dims && Tuple(last(op.dims)) == dims
    has_dims(H) || throw(DimensionMismatch("mode_dims=$dims do not match H tensor dimensions $(H.dims)"))
    all(has_dims, mode_ops) || throw(DimensionMismatch("every mode operator must have tensor dimensions $dims"))
    all(has_dims, c_ops) || throw(DimensionMismatch("every collapse operator must have tensor dimensions $dims"))

    shifts = preliminary_shifts === nothing ? zeros(ComplexF64, length(c_ops)) : Vector{ComplexF64}(preliminary_shifts)
    length(shifts) == length(c_ops) ||
        throw(DimensionMismatch("preliminary_shifts must contain one shift per collapse channel"))
    all(isfinite, shifts) || throw(ArgumentError("preliminary_shifts must be finite"))
    return shifts
end

end
