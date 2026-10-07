# Gauge discovery from preliminary trajectories (paper Appendix A.2).

# Paper Eq. (A.7): ζ_μ^(g) from clusters of preliminary trajectories.
function _discover_gauges(
        ::Val{:trajectories}, H, c_ops;
        mode_ops, mode_dims, discovery_time, seed_radii, cluster_scales,
        step = discovery_time / 40, nseeds::Int = 600, terminal_window = 0.0,
        preliminary_shifts = nothing, dbscan_radius = 1.5,
        min_neighbors::Int = 10, min_weight = 0.02, rng::AbstractRNG = Random.default_rng(),
        save_preliminary_trajectories::Int = 0, ensemblealg::EnsembleAlgorithm = EnsembleThreads()
    )
    shifts = _check_model(H, c_ops, mode_ops, mode_dims, seed_radii, cluster_scales, preliminary_shifts)
    _check_options(;
        discovery_time, step, terminal_window, dbscan_radius, nseeds, min_neighbors,
        min_weight, save_preliminary_trajectories, ensemblealg
    )
    data = _run_preliminary_trajectories(
        H, c_ops, shifts, mode_ops, mode_dims, discovery_time, seed_radii, step, nseeds,
        terminal_window, rng, save_preliminary_trajectories, ensemblealg
    )
    clusters = _cluster_terminal_means(data.terminal_means; cluster_scales, dbscan_radius, min_neighbors, min_weight)
    isempty(clusters.counts) && throw(ArgumentError("trajectory discovery found no retained DBSCAN clusters"))

    # ζ_μ^(g) = -⟨C_μ⟩, averaged over the trajectories of cluster g.
    gauge_shifts = stack(
        -_mean_columns(data.terminal_collapse_means, findall(==(g), clusters.labels))
            for g in eachindex(clusters.counts)
    )
    diagnostics = (;
        counts = clusters.counts, clusters.labels,
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
    return (; shifts = gauge_shifts, method = :trajectories, clusters.centers, clusters.weights, diagnostics)
end

# Checks the operators and per-mode inputs; returns the preliminary shifts.
function _check_model(H, c_ops, mode_ops, mode_dims, seed_radii, cluster_scales, preliminary_shifts)
    nmodes = length(mode_ops)
    nmodes > 0 && length(mode_dims) == length(seed_radii) == length(cluster_scales) == nmodes ||
        throw(DimensionMismatch("mode_ops needs at least one operator, with one entry of mode_dims, seed_radii and cluster_scales each"))
    all(r -> isfinite(r) && r >= 0, seed_radii) || throw(ArgumentError("seed_radii must be finite and nonnegative"))
    all(s -> isfinite(s) && s > 0, cluster_scales) || throw(ArgumentError("cluster_scales must be finite and positive"))

    dims = Tuple(Int.(mode_dims))
    has_dims(op) = size(op.data) == (prod(dims), prod(dims)) && Tuple(first(op.dims)) == dims && Tuple(last(op.dims)) == dims
    all(has_dims, (H, mode_ops..., c_ops...)) ||
        throw(DimensionMismatch("H, mode_ops and c_ops must have tensor dimensions mode_dims=$dims"))

    shifts = preliminary_shifts === nothing ? zeros(ComplexF64, length(c_ops)) : Vector{ComplexF64}(preliminary_shifts)
    length(shifts) == length(c_ops) ||
        throw(DimensionMismatch("preliminary_shifts must contain one shift per collapse channel"))
    all(isfinite, shifts) || throw(ArgumentError("preliminary_shifts must be finite"))
    return shifts
end

function _check_options(;
        discovery_time, step, terminal_window, dbscan_radius, nseeds, min_neighbors,
        min_weight, save_preliminary_trajectories, ensemblealg
    )
    for (name, value) in (:discovery_time => discovery_time, :step => step, :dbscan_radius => dbscan_radius)
        isfinite(value) && value > 0 || throw(ArgumentError("$name must be finite and positive"))
    end
    isfinite(terminal_window) && terminal_window >= 0 ||
        throw(ArgumentError("terminal_window must be finite and nonnegative"))
    for (name, value) in (:nseeds => nseeds, :min_neighbors => min_neighbors)
        value >= 1 || throw(ArgumentError("$name must be positive"))
    end
    0 <= min_weight < 1 || throw(ArgumentError("min_weight must lie in [0, 1)"))
    save_preliminary_trajectories >= 0 ||
        throw(ArgumentError("save_preliminary_trajectories must be nonnegative"))
    ensemblealg isa Union{EnsembleSerial, EnsembleThreads, EnsembleDistributed} ||
        throw(ArgumentError("ensemblealg must be EnsembleSerial(), EnsembleThreads(), or EnsembleDistributed()"))
    ensemblealg isa EnsembleDistributed && Distributed.nprocs() <= 1 &&
        throw(ArgumentError("EnsembleDistributed() requires worker processes"))
    return nothing
end

# Paper Eqs. (A.6–A.7): one trajectory from each random coherent state, averaging
# the mode amplitudes and the collapse expectations over the terminal window.
function _run_preliminary_trajectories(
        H, c_ops, shifts, mode_ops, mode_dims, discovery_time, seed_radii, step, nseeds,
        terminal_window, rng, save_preliminary_trajectories, ensemblealg
    )
    tlist = collect(0.0:float(step):float(discovery_time))
    last(tlist) < discovery_time && push!(tlist, float(discovery_time))
    start = discovery_time - terminal_window
    tail = findall(t -> t >= start || t ≈ start, tlist)   # the window, up to rounding of `start`

    Hrun, Crun = all(iszero, shifts) ? (H, c_ops) : _shifted_operators(H, c_ops, shifts)
    nmodes = length(mode_ops)
    e_ops = vcat(collect(mode_ops), [op' * op for op in mode_ops], collect(c_ops))
    modes, occupations, collapses = 1:nmodes, nmodes .+ (1:nmodes), 2nmodes .+ eachindex(c_ops)

    # Initial amplitudes uniform in a disk of radius `seed_radii[mode]`, and one seed per run.
    amplitudes = [seed_radii[mode] * sqrt(rand(rng)) * cis(2π * rand(rng)) for mode in 1:nmodes, _ in 1:nseeds]
    seeds = [rand(rng, UInt64) for _ in 1:nseeds]
    nsave = min(save_preliminary_trajectories, nseeds)

    function relax(point)
        ψ0 = tensor((coherent(Int(mode_dims[mode]), amplitudes[mode, point]) for mode in 1:nmodes)...)
        sol = mcsolve(
            Hrun, ψ0, tlist, Crun; e_ops, ntraj = 1, rng = Xoshiro(seeds[point]), saveat = tlist,
            keep_runs_results = Val(true), ensemblealg = EnsembleSerial(), progress_bar = Val(false)
        )
        expect = sol.expect[:, 1, :]
        terminal = vec(sum(expect[:, tail]; dims = 2)) / length(tail)
        point <= nsave || return (; terminal, trace = nothing)
        states = stack(ψ.data for ψ in vec(sol.states))
        return (; terminal, trace = (; states, expect, col_times = sol.col_times[1], col_which = sol.col_which[1]))
    end

    results = _map_seeds(relax, nseeds, ensemblealg)
    averages = stack(result.terminal for result in results)
    traces = [results[point].trace for point in 1:nsave]
    return (;
        tlist, nsave,
        terminal_means = averages[modes, :],
        terminal_occupations = real.(averages[occupations, :]),
        terminal_collapse_means = averages[collapses, :],
        states = [trace.states for trace in traces],
        traces = [trace.expect[modes, :] for trace in traces],
        occupations = [real.(trace.expect[occupations, :]) for trace in traces],
        jump_times = [trace.col_times for trace in traces],
        jump_channels = [trace.col_which for trace in traces],
    )
end

# Run `f(i)` for `i in 1:n` as requested by `ensemblealg`.
_map_seeds(f, n, ::EnsembleSerial) = map(f, 1:n)
_map_seeds(f, n, ::EnsembleThreads) = fetch.([Threads.@spawn f(i) for i in 1:n])
_map_seeds(f, n, ::EnsembleDistributed) = Distributed.pmap(f, 1:n)
