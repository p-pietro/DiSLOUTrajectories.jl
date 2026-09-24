@testset "trajectory gauge discovery" begin
    d = 6
    a = destroy(d)
    H = 0.2 * num(d)
    c_ops = [sqrt(0.4) * a]
    preliminary = ComplexF64[0.15 + 0.05im]
    kw = (;
        method = :trajectories, mode_ops = [a], mode_dims = [d],
        discovery_time = 0.2, seed_radii = [sqrt(d - 1)], cluster_scales = [1.0],
        step = 0.02, nseeds = 12, terminal_window = 0.04, preliminary_shifts = preliminary,
        dbscan_radius = 10.0, min_neighbors = 1, min_weight = 0.0,
        save_preliminary_trajectories = 2,
    )
    result = discover_gauges(H, c_ops; kw..., rng = Xoshiro(19), ensemblealg = EnsembleSerial())

    @testset "result" begin
        @test propertynames(result) == (:shifts, :method, :centers, :weights, :diagnostics)
        @test result.method === :trajectories
        @test size(result.shifts, 1) == length(c_ops)
        @test size(result.shifts, 2) == size(result.centers, 2) == length(result.weights)
        @test issorted(result.weights; rev = true)
        for g in axes(result.shifts, 2)
            members = result.diagnostics.labels .== g
            @test result.shifts[:, g] ≈ -vec(mean(result.diagnostics.terminal_collapse_means[:, members]; dims = 2))
        end
        @test result.diagnostics.preliminary_shifts == preliminary
    end

    @testset "reproducible with threads" begin
        threaded = discover_gauges(H, c_ops; kw..., rng = Xoshiro(19), ensemblealg = EnsembleThreads())
        @test threaded.shifts == result.shifts
        @test threaded.diagnostics.labels == result.diagnostics.labels
    end

    # Each seed runs mcsolve from a random coherent state, with the shifted
    # operators, and averages the physical ⟨C⟩ over the terminal window.
    @testset "preliminary runs replay with mcsolve" begin
        rng = Xoshiro(19)
        amplitudes = [sqrt(d - 1) * sqrt(rand(rng)) * cis(2π * rand(rng)) for _ in 1:kw.nseeds]
        seeds = [rand(rng, UInt64) for _ in 1:kw.nseeds]
        Hrun, Crun = SM._shifted_operators(H, c_ops, preliminary)
        tlist = collect(0.0:kw.step:kw.discovery_time)
        tail = tlist .>= kw.discovery_time - kw.terminal_window
        expected = map(1:kw.nseeds) do seed
            sol = mcsolve(
                Hrun, coherent(d, amplitudes[seed]), tlist, Crun; e_ops = c_ops, ntraj = 1,
                rng = Xoshiro(seeds[seed]), keep_runs_results = Val(true), quiet...
            )
            mean(sol.expect[1, 1, tail])
        end
        @test vec(result.diagnostics.terminal_collapse_means) ≈ expected
    end

    @testset "saved preliminary trajectories" begin
        traces = result.diagnostics.preliminary_traces
        nt = length(result.diagnostics.times)
        @test traces.indices == [1, 2]
        @test all(size(states) == (d, nt) for states in traces.states)
        @test all(size(means) == (1, nt) for means in traces.means)
        @test length(result.diagnostics.preliminary_jump_times) == 2
    end

    @testset "DBSCAN clustering" begin
        # Two clusters of three core points and one boundary point each, and an outlier.
        points = ComplexF64[
        1.0 + 1.0im 1.04 + 0.98im 0.97 + 1.02im 1.12 + 1.0im -1.0 - 1.0im -0.98 - 1.04im -1.03 - 0.97im -0.9 - 1.06im 5.0 + 5.0im
        ]
        clusters = ClusteringExt._cluster_terminal_means(
            points; cluster_scales = [0.2], dbscan_radius = 0.5, min_neighbors = 3, min_weight = 0.0
        )
        @test clusters.counts == [4, 4]
        @test clusters.weights == [4 / 9, 4 / 9]
        @test clusters.labels == [2, 2, 2, 2, 1, 1, 1, 1, 0]   # equal weights are ordered by center
        @test clusters.centers ≈ ComplexF64[(-3.91 - 4.07im) / 4 (4.13 + 4.0im) / 4]
    end

    @testset "input errors" begin
        @test_throws ArgumentError discover_gauges(H, c_ops; kw..., nseeds = 0)
        @test_throws DimensionMismatch discover_gauges(H, c_ops; kw..., mode_dims = [d + 1])
        @test_throws "unknown gauge discovery method" discover_gauges(H, c_ops; kw..., method = :unknown)
        @test_throws "using QuantumCumulants" discover_gauges(H, c_ops; kw..., method = :semiclassical)
    end
end
