@testset "GaugeEigenExponential" begin
    rng = Xoshiro(1)
    N = 6
    A = randn(rng, ComplexF64, N, N)
    H = QuantumObject((A + A') / 2)
    c_ops = [QuantumObject(randn(rng, ComplexF64, N, N) / 3) for _ in 1:2]
    L = -im * Matrix((H - 0.5im * sum(C' * C for C in c_ops)).data)   # non-normal generator
    ψ0 = normalize(randn(rng, ComplexF64, N))
    exact(t) = exp(L * t) * ψ0
    alg = GaugeEigenExponential(H, c_ops)

    @testset "steps and dense output are exact" begin
        prob = SciMLBase.ODEProblem((du, u, p, t) -> mul!(du, L, u), ψ0, (0.0, 2.0))
        # Steps stop at the tstops; saveat times in between use the dense output.
        sol = SciMLBase.solve(prob, alg; tstops = [0.5, 0.7, 1.4], saveat = [0.3, 0.7, 1.1, 2.0])
        @test sol.t == [0.3, 0.7, 1.1, 2.0]
        @test all(isapprox(sol.u[i], exact(sol.t[i]); rtol = 1.0e-11) for i in eachindex(sol.t))

        integrator = SciMLBase.init(prob, alg)
        SciMLBase.step!(integrator)
        @test integrator.t == 2.0
        @test integrator(0.9) ≈ exact(0.9) rtol = 1.0e-11
        @test integrator(0.9, Val{1}) ≈ L * exact(0.9) rtol = 1.0e-11
    end

    @testset "mcsolve with the exact propagator" begin
        m = driven_cavity(N = 20)
        tlist = range(0, 10, 51)
        kw = (; e_ops = [m.a' * m.a], ntraj = 40, quiet...)
        sol_dp5 = mcsolve(m.H, m.ψ0, tlist, m.c_ops; rng = Xoshiro(2), kw...)
        sol = mcsolve(m.H, m.ψ0, tlist, m.c_ops; alg = GaugeEigenExponential(m.H, m.c_ops), rng = Xoshiro(2), kw...)
        @test length.(sol.col_times) == length.(sol_dp5.col_times)
        @test all(isapprox(sol.col_times[i], sol_dp5.col_times[i]; atol = 1.0e-4) for i in 1:40)
    end

    @testset "a failed GPU diagonalization falls back to LAPACK" begin
        # Without the CUDA extension, the GPU diagonalization fails with a MethodError.
        SM._enable_cuda_diagonalization!()
        gpu = @test_logs (:warn, r"CUDA diagonalization failed") GaugeEigenExponential(H, c_ops)
        @test !backend_info().cuda_enabled
        @test first(gpu.bases).λ == first(alg.bases).λ
    end

    @test sprint(show, alg) == "GaugeEigenExponential(1 gauge(s), 6 modes)"
    @test SM.OrdinaryDiffEqCore.alg_order(alg) == 1
end
