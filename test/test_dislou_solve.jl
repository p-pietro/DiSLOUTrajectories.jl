@testset "dislou_solve" begin
    @testset "driven cavity" begin
        m = driven_cavity()
        tlist = range(0, 10, 100)
        kw = (; e_ops = [m.a' * m.a], ntraj = 100)
        sol_zero = cavity_solve(m, tlist; kw..., rng = MersenneTwister(1))

        @testset "a zero gauge reproduces mcsolve" begin
            sol_mc = mcsolve(m.H, m.ψ0, tlist, m.c_ops; kw..., rng = MersenneTwister(1), quiet...)
            sol_me = mesolve(m.H, m.ψ0, tlist, m.c_ops; e_ops = kw.e_ops, quiet...)
            @test sol_zero isa TimeEvolutionMCSol
            @test sum(length, sol_zero.col_times) == sum(length, sol_mc.col_times)
            @test all(isapprox(sol_zero.col_times[i], sol_mc.col_times[i]; atol = 1.0e-4) for i in 1:100)
            @test sol_zero.col_which == sol_mc.col_which
            # A driven cavity stays in a coherent state, so every trajectory gives the exact ⟨n⟩.
            @test real.(sol_zero.expect[1, :]) ≈ real.(sol_me.expect[1, :]) atol = 1.0e-5
        end

        @testset "a gauge at the steady state removes most jumps" begin
            sol = cavity_solve(m, tlist; kw..., rng = MersenneTwister(1), gauge_set = m.steady_gauge)
            # About 250 jumps against 725 on average: the shifted jump rate κ|α(t) - α|² decays.
            @test 2 * sum(length, sol.col_times) < sum(length, sol_zero.col_times)
            @test sol.expect ≈ sol_zero.expect atol = 1.0e-8
        end
    end

    # Bistable Kerr resonator started between its two branches, with one gauge per branch.
    @testset "driven Kerr resonator" begin
        p = DrivenKerrParams()
        kerr = driven_kerr_model(p)
        ψ0 = coherent(p.N, kerr.αmid)
        tlist = range(0, 10, 51)
        shifts = -sqrt(kerr.κ) * ComplexF64[kerr.αlow kerr.αhigh]
        kerr_solve(; kw...) = dislou_solve(
            kerr.H, ψ0, tlist, kerr.c_ops; gauge_set = shifts, e_ops = [kerr.nop],
            ntraj = 300, rng = Xoshiro(7), quiet..., kw...
        )
        n_mesolve = real.(mesolve(kerr.H, ψ0, tlist, kerr.c_ops; e_ops = [kerr.nop], quiet...).expect[1, :])
        recorder, gauges, _ = gauge_recorder()
        sol_kerr = kerr_solve(; keep_runs_results = Val(true), ensemblealg = EnsembleSerial(), callback = recorder)

        @testset "gauge switching agrees with mesolve" begin
            @test within_errors(sol_kerr, n_mesolve; atol = 1.0e-3)
            @test 1 in gauges && 2 in gauges
            sol_mc = mcsolve(kerr.H, ψ0, tlist, kerr.c_ops; ntraj = 300, rng = Xoshiro(7), quiet...)
            @test 2 * sum(length, sol_kerr.col_times) < sum(length, sol_mc.col_times)
        end

        @testset "threads and serial runs are identical" begin
            sol_threads = kerr_solve(; keep_runs_results = Val(true), ensemblealg = EnsembleThreads())
            @test sol_threads.expect == sol_kerr.expect
            @test sol_threads.col_times == sol_kerr.col_times
        end

        @testset "Layer III" begin
            # A projection that is never accepted leaves the trajectories unchanged.
            sol_never = kerr_solve(; keep_runs_results = Val(true), layer3_sizes = 10, residual_tolerance = 1.0e-14)
            @test sol_never.col_times == sol_kerr.col_times
            @test sol_never.expect == sol_kerr.expect

            recorder, _, reduced = gauge_recorder()
            sol = kerr_solve(;
                layer3_sizes = [12, 16], keep_runs_results = Val(true),
                ensemblealg = EnsembleSerial(), callback = recorder
            )
            @test sprint(show, sol.alg) == "GaugeEigenExponential(2 gauge(s), 30 modes, Layer III modes [12, 16])"
            @test count(reduced) > length(reduced) / 2
            @test within_errors(sol, n_mesolve; atol = 1.0e-2)
        end
    end

    @testset "mcsolve options pass through" begin
        m = driven_cavity(N = 20)
        tlist = [0.0, 0.5, 1.0]
        run(; kw...) = cavity_solve(m, tlist; gauge_set = m.steady_gauge, ntraj = 8, rng = Xoshiro(3), kw...)
        sol = run()   # no e_ops: the states are saved at tlist
        @test sol.times_states == tlist
        @test isempty(sol.expect)
        saveat = [0.25, 0.75]   # between the steps, so the states come from the dense output
        sol = run(; e_ops = [m.a], saveat, keep_runs_results = Val(true))
        @test sol.times_states == saveat
        @test size(sol.expect) == (1, 8, 3)
        @test size(sol.states) == (8, 2)
        # Every trajectory is the coherent state |β(t)⟩ of the classical field.
        β(t) = m.α * (1 - exp(-(m.κ / 2 + 0.5im) * t))
        @test all(abs2(dot(sol.states[i, j].data, coherent(20, β(saveat[j])).data)) ≈ 1 for i in 1:8, j in 1:2)

        ncalls = Ref(0)
        counter = SciMLBase.DiscreteCallback((u, t, integrator) -> (ncalls[] += 1; false), identity)
        run(callback = counter, ensemblealg = EnsembleSerial())
        @test ncalls[] > 0
    end

    @testset "gauge_set can be a discover_gauges result" begin
        m = driven_cavity(N = 10)
        found = (; shifts = m.steady_gauge, method = :manual, weights = [1.0])
        run(gauge_set) = cavity_solve(m, [0.0, 1.0]; gauge_set, ntraj = 4, rng = Xoshiro(1))
        @test run(found).col_times == run(found.shifts).col_times
    end
end
