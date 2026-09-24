# dislou_solve keeps the array types of its inputs; with a CUDA GPU, the same
# trajectories run on GPU arrays.
if Sys.islinux() || Sys.iswindows()
    using CUDA
end

@testset "GPU arrays" begin
    if isdefined(Main, :CUDA) && CUDA.functional()
        m = driven_cavity(N = 20)
        run(to) = dislou_solve(
            to(m.H), to(m.ψ0), 0:0.5:5, to.(m.c_ops);
            gauge_set = m.steady_gauge, ntraj = 20, rng = Xoshiro(1), quiet...
        )
        cpu = run(identity)
        gpu = run(cu)
        @test first(gpu.alg.bases).V isa CuMatrix
        @test gpu.col_which == cpu.col_which
        @test all(isapprox(g, c; atol = 1.0e-8) for (g, c) in zip(gpu.col_times, cpu.col_times))
    else
        @test_skip "no functional CUDA GPU"
    end
end
