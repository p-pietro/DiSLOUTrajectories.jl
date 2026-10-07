@testset "dislou_solve" begin
    @testset "analytical jump finding" begin
        N, κ = 6, 0.7
        a = destroy(N)
        H, c_ops = 0 * a, [sqrt(κ) * a]
        for reduced in (false, true), logarithm in (false, true)
            alg = GaugeEigenExponential(H, c_ops, zeros(1, 1); layer3_sizes = reduced ? N : nothing)
            prob = mcsolveProblem(
                H, fock(N, 1), [0.0, 2000.0], c_ops;
                e_ops = [], save_end = false, jump_derivative = true, jump_log = logarithm
            )
            jump = QuantumToolbox._mc_get_jump_callback(prob.prob.kwargs[:callback])
            @test jump.condition isa QuantumToolbox.ConditionWithDerivative
            # PR 792 draws the target when integration starts.
            prob = SciMLBase.remake(prob.prob; callback = QuantumToolbox._modify_field(jump, :initialize, (cb, u, t, integrator) -> (cb.affect!.random_n[] = 0.0; nothing)))
            integrator = SciMLBase.init(prob, alg; saveat = Float64[], save_everystep = false, save_end = false)
            SciMLBase.step!(integrator)
            @test integrator.cache.coordinates == reduced
            t = 0.9
            # Use the callback interpolation buffer: the derivative must not overwrite it.
            u = integrator(integrator.cache.tmp, t)
            before = copy(u)
            s = exp(-κ * t)
            @test QuantumToolbox._mcsolve_continuous_derivative(u, t, integrator, Val(false)) ≈ κ * s rtol = 1.0e-11
            @test jump.condition.derivative(u, t, integrator) ≈
                (logarithm ? κ : κ * s) rtol = 1.0e-11
            @test u == before
            @test jump.condition(u, t, integrator) < 0   # zero threshold has no crossing

            # Fix the threshold and locate its known crossing, even though the
            # survival probability underflows at the end of the step.
            r = 0.37
            prob = mcsolveProblem(
                H, fock(N, 1), [0.0, 2000.0], c_ops;
                e_ops = [], save_end = false, jump_derivative = true, jump_log = logarithm
            )
            jump = QuantumToolbox._mc_get_jump_callback(prob.prob.kwargs[:callback])
            prob = SciMLBase.remake(prob.prob; callback = QuantumToolbox._modify_field(jump, :initialize, (cb, u, t, integrator) -> (cb.affect!.random_n[] = r; nothing)))
            SciMLBase.solve(prob, alg; saveat = Float64[], save_everystep = false, save_end = false)
            @test jump.affect!.col_times_which_idx[] == 2
            @test jump.affect!.col_times[1] ≈ -log(r) / κ atol = 1.0e-11
        end

        # Shifted operators and changing gauges must give the same events in both modes.
        m = driven_cavity(N = 20)
        run(logarithm, ensemblealg) = cavity_solve(
            m, range(0, 10, 21); gauge_set = hcat(zeros(1), m.steady_gauge),
            e_ops = [m.a' * m.a], ntraj = 12, rng = Xoshiro(2),
            jump_log = logarithm, ensemblealg
        )
        plain = run(false, EnsembleSerial())
        logarithmic = run(true, EnsembleSerial())
        @test plain.col_which == logarithmic.col_which
        @test all(isapprox(x, y; atol = 1.0e-9) for (x, y) in zip(plain.col_times, logarithmic.col_times))
        @test plain.expect ≈ logarithmic.expect atol = 1.0e-9
        @test run(true, EnsembleThreads()).col_times == logarithmic.col_times
    end

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
            # The final states are kets, not coordinates on the slow modes.
            @test [expect(kerr.nop, ψ) for ψ in sol.states[:, end]] ≈ sol.expect[1, :, end]

            # With the states saved at every time, the steps end on kets instead of coordinates.
            sol_states = kerr_solve(; layer3_sizes = [12, 16], keep_runs_results = Val(true), saveat = tlist)
            @test sol_states.col_which == sol.col_which
            @test all(isapprox(sol_states.col_times[i], sol.col_times[i]; atol = 1.0e-8) for i in 1:300)
            @test sol_states.expect ≈ sol.expect atol = 1.0e-8
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

    # The solver keeps the array types of the model, here its single precision.
    @testset "single precision is kept" begin
        m = driven_cavity(N = 10)
        f32(A) = QuantumObject(ComplexF32.(A.data); type = A.type, dims = A.dimensions)
        m32 = (; m..., H = f32(m.H), ψ0 = f32(m.ψ0), c_ops = f32.(m.c_ops))
        sol = cavity_solve(m32, [0.0, 1.0]; gauge_set = m.steady_gauge, ntraj = 2, rng = Xoshiro(1))
        @test eltype(first(sol.alg.bases).V) == ComplexF32
        @test eltype(first(sol.states).data) == ComplexF32
    end
end
