# DBSCAN clustering of the terminal amplitudes of the preliminary trajectories
# (paper Appendix A.2).

_sqdist(X, i, j) = sum(k -> abs2(X[k, i] - X[k, j]), axes(X, 1))

# Labels of the columns of `X`, with 0 for noise. A point with at least `min_neighbors`
# points within `radius`, itself included, is a core point. Clusters are the connected
# sets of core points, and any other point joins the cluster of its nearest core point
# within `radius`.
function _dbscan(X::AbstractMatrix{<:Real}, radius::Real, min_neighbors::Integer)
    npoints = size(X, 2)
    near = [_sqdist(X, i, j) <= radius^2 for i in 1:npoints, j in 1:npoints]
    core = vec(sum(near; dims = 1)) .>= min_neighbors
    labels = zeros(Int, npoints)
    nclusters = 0
    for seed in findall(core)
        labels[seed] == 0 || continue
        nclusters += 1
        labels[seed] = nclusters
        frontier = [seed]
        while !isempty(frontier)
            point = pop!(frontier)
            for neighbor in findall(@view near[:, point])
                core[neighbor] && labels[neighbor] == 0 || continue
                labels[neighbor] = nclusters
                push!(frontier, neighbor)
            end
        end
    end
    for point in findall(!, core)
        reachable = filter(j -> core[j] && near[j, point], 1:npoints)
        isempty(reachable) || (labels[point] = labels[argmin(j -> _sqdist(X, point, j), reachable)])
    end
    return labels
end

# Paper: rescaled (Re ᾱ_j^(r), Im ᾱ_j^(r)), one column per point.
_scaled_features(points, scales) = reinterpret(Float64, ComplexF64.(points ./ scales))

_mean_columns(A, columns) = vec(sum(@view(A[:, columns]); dims = 2)) / length(columns)

# Paper: clusters 𝓘_g and centers of ᾱ_j^(r). Clusters with less than a fraction
# `min_weight` of the points are discarded; the others are ordered by decreasing
# weight, then by center. Discarded and noise points get the label 0.
function _cluster_terminal_means(points::AbstractMatrix{<:Complex}; cluster_scales, dbscan_radius, min_neighbors, min_weight)
    nmodes, npoints = size(points)
    labels = _dbscan(_scaled_features(points, cluster_scales), dbscan_radius, min_neighbors)
    members = [findall(==(g), labels) for g in 1:maximum(labels; init = 0)]
    centers = [_mean_columns(points, group) for group in members]
    kept = filter(g -> length(members[g]) / npoints >= min_weight, eachindex(members))
    sort!(kept; by = g -> (-length(members[g]), collect(reinterpret(Float64, centers[g]))))
    relabel = zeros(Int, length(members) + 1)   # relabel[g + 1] is the final label of cluster g
    relabel[kept .+ 1] = eachindex(kept)
    return (;
        centers = ComplexF64[centers[g][mode] for mode in 1:nmodes, g in kept],
        weights = [length(members[g]) / npoints for g in kept],
        counts = [length(members[g]) for g in kept],
        labels = relabel[labels .+ 1],
    )
end
