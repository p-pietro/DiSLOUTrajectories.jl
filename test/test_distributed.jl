using Distributed
import SciMLBase: EnsembleDistributed

@testset "gauge discovery needs workers for EnsembleDistributed" begin
    @test nprocs() == 1
    a = destroy(2)
    @test_throws ArgumentError discover_gauges(
        num(2), [a];
        method = :trajectories, mode_ops = [a], mode_dims = [2],
        discovery_time = 0.1, seed_radii = [0.1], cluster_scales = [1.0], step = 0.1,
        nseeds = 1, dbscan_radius = 1.0, min_neighbors = 1, min_weight = 0.0,
        ensemblealg = EnsembleDistributed()
    )
end

@testset "distributed runs match serial runs" begin
    project = Base.active_project()
    workers_added = addprocs(2; exeflags = `--project=$project --startup-file=no`)
    try
        @everywhere using DiSLOUTrajectories, QuantumToolbox, Clustering
        p = DrivenKerrParams(N = 20)
        kerr = driven_kerr_model(p)
        shifts = -sqrt(kerr.κ) * ComplexF64[kerr.αlow kerr.αhigh]
        # QuantumToolbox 0.49 needs the progress bar with EnsembleDistributed().
        run(ensemblealg) = dislou_solve(
            kerr.H, coherent(p.N, kerr.αmid), 0:0.5:5, kerr.c_ops; gauge_set = shifts,
            e_ops = [kerr.nop], ntraj = 40, rng = Xoshiro(11), layer3_sizes = 10, ensemblealg
        )
        serial = run(EnsembleSerial())
        distributed = run(EnsembleDistributed())
        @test distributed.col_times == serial.col_times
        @test distributed.expect == serial.expect

        a = destroy(p.N)
        discover(ensemblealg) = discover_gauges(
            kerr.H, kerr.c_ops;
            method = :trajectories, mode_ops = [a], mode_dims = [p.N],
            discovery_time = 1.0, seed_radii = [4.0], cluster_scales = [1.0], step = 0.25,
            nseeds = 8, dbscan_radius = 2.0, min_neighbors = 1, min_weight = 0.0,
            rng = Xoshiro(2), ensemblealg
        )
        @test discover(EnsembleDistributed()).shifts == discover(EnsembleSerial()).shifts
    finally
        rmprocs(workers_added)
    end
end
