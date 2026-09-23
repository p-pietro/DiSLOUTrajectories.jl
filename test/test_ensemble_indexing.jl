@testset "trajectory seeds follow QuantumToolbox.mcsolve" begin
    master = MersenneTwister(23)
    seeds = SM._trajectory_seeds(MersenneTwister(23), 5)
    rand(master)
    @test seeds == [rand(master, UInt64) for _ in 1:5]
end

@testset "trajectory streams derive only from the rng state" begin
    d = 4
    a = destroy(d)
    H = 0.15 * (a + a')
    psi0 = fock(d, 3)
    tlist = collect(0.0:0.2:1.0)
    common = (;
        e_ops = [num(d)],
        gauge_set = ComplexF64[0 0.6],
        ntraj = 12,
        ensemblealg = :serial,
        saveat = tlist,
        save_trajectories = true,
        save_final_states = true,
    )
    first_run = dislou_solve(H, psi0, tlist, [1.2 * a]; common..., rng = Xoshiro(0x1234))
    repeated = dislou_solve(H, psi0, tlist, [1.2 * a]; common..., rng = Xoshiro(0x1234))
    another_seed = dislou_solve(H, psi0, tlist, [1.2 * a]; common..., rng = Xoshiro(0x1235))

    @test first_run.col_times == repeated.col_times
    @test first_run.col_which == repeated.col_which
    @test first_run.col_gauge == repeated.col_gauge
    @test first_run.trajectory_expect == repeated.trajectory_expect
    @test first_run.final_states == repeated.final_states
    @test (first_run.col_times, first_run.col_which, first_run.col_gauge) !=
        (another_seed.col_times, another_seed.col_which, another_seed.col_gauge)
end

# Mutation caught: any change to how trajectory streams are derived from `rng`,
# or to the order of threshold and channel draws within a trajectory.
@testset "zero-shift trajectories reproduce mcsolve jumps" begin
    d = 12
    a = destroy(d)
    H = 0.5 * num(d) + 0.6 * (a + a')
    c_ops = [sqrt(0.3) * a, sqrt(0.1) * a' * a]
    tlist = range(0, 8, 41)
    ntraj = 24
    for make_rng in (() -> MersenneTwister(1), () -> Xoshiro(7))
        reference = mcsolve(
            H, fock(d, 0), tlist, c_ops;
            ntraj, rng = make_rng(), progress_bar = Val(false)
        )
        sol = dislou_solve(
            H, fock(d, 0), tlist, c_ops;
            gauge_set = zeros(ComplexF64, 2, 1), ntraj, rng = make_rng()
        )
        @test length.(sol.col_times) == length.(reference.col_times)
        @test sol.col_which == reference.col_which
        @test all(
            isapprox(x, y; atol = 1.0e-4, rtol = 0)
                for (x, y) in zip(sol.col_times, reference.col_times)
        )
    end
end
