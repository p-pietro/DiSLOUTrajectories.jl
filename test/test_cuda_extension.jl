if Sys.islinux() || Sys.iswindows()
    using CUDA

    @testset "CUDA extension" begin
        @test Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesCUDAExt) !== nothing
        @test backend_info().cuda_enabled == CUDA.functional()

        if CUDA.functional()
            rng = Xoshiro(92)
            Heff = randn(rng, ComplexF64, 8, 8)
            E, V = DiSLOUTrajectories._cuda_eigen(Heff)
            @test E isa Vector{ComplexF64} && V isa Matrix{ComplexF64}
            @test Heff * V ≈ V * Diagonal(E)

            # The solver gives the same trajectories with GPU and CPU eigensystems.
            m = driven_cavity(N = 20)
            run() = dislou_solve(
                m.H, m.ψ0, 0:0.5:5, m.c_ops; gauge_set = zeros(ComplexF64, 1, 1),
                e_ops = [m.a' * m.a], ntraj = 20, rng = Xoshiro(1), quiet...
            )
            on_gpu = run()
            DiSLOUTrajectories._disable_cuda_diagonalization!()
            try
                on_cpu = run()
                @test length.(on_gpu.col_times) == length.(on_cpu.col_times)
                @test on_gpu.expect ≈ on_cpu.expect
            finally
                DiSLOUTrajectories._enable_cuda_diagonalization!()
            end
        end
    end
else
    @testset "CUDA extension" begin
        @test_skip "CUDA is unsupported on this platform"
    end
end
