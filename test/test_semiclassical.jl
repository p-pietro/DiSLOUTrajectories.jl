include("fixtures/two_mode_diamond.jl")

@testset "semiclassical gauge discovery" begin
    @testset "driven Kerr resonator matches its analytic branches" begin
        kerr = driven_kerr_model()
        semiclassical(; kw...) = discover_gauges(
            kerr.H, kerr.c_ops; method = :semiclassical, mode_ops = [kerr.a],
            mode_dims = [kerr.p.N], limits = (20.0,), kw...
        )
        result = semiclassical()

        @test propertynames(result) == (:shifts, :method, :centers, :weights, :diagnostics)
        @test result.method === :semiclassical
        @test result.centers ≈ [kerr.αlow kerr.αhigh] atol = 1.0e-7
        @test result.diagnostics.rejected ≈ [kerr.αmid] atol = 1.0e-7
        @test result.shifts ≈ -sqrt(kerr.κ) * result.centers atol = 1.0e-7
        @test result.weights == [0.5, 0.5]
        @test length(result.diagnostics.stability) == size(result.diagnostics.roots, 2) == 3
        @test all(result.diagnostics.stable)
        @test semiclassical(ensemblealg = EnsembleSerial()).diagnostics.roots == result.diagnostics.roots

        sol = dislou_solve(
            kerr.H, kerr.ψ0, [0.0, 0.5], kerr.c_ops; gauge_set = result, ntraj = 2, rng = Xoshiro(5), quiet...
        )
        @test length(sol.alg.bases) == 2
    end

    # Two stable branches and a saddle of the memory, with the buffer adiabatically following.
    @testset "two-mode diamond" begin
        cfg = TWO_MODE_DIAMOND
        model = two_mode_diamond_model((cfg.Na, cfg.Nb))
        result = discover_gauges(
            model.H, model.c_ops; method = :semiclassical, model.mode_ops,
            mode_dims = [cfg.Na, cfg.Nb], limits = (cfg.Na - 1, cfg.Nb - 1)
        )
        @test size(result.diagnostics.roots) == (2, 5)
        @test count(s -> s.stable, result.diagnostics.stability) == size(result.centers, 2) == 3
        @test result.shifts ≈ -sqrt.([cfg.κa, cfg.κb]) .* result.centers
    end

    # The shifts are the collapse operators evaluated on the fixed point, ζ = -C(α, α*).
    @testset "nonlinear collapse operators" begin
        dims = (12, 8)
        a = tensor(destroy(dims[1]), qeye(dims[2]))
        b = tensor(qeye(dims[1]), destroy(dims[2]))
        H = 0.8 * a' * a + 0.3 * b' * b + 0.5 * (a + a') + 0.2 * (a' * b + b' * a)
        c_ops = [sqrt(0.4) * a, sqrt(0.3) * b, sqrt(0.05) * a^2, sqrt(0.02) * (a' * a)]
        result = discover_gauges(
            H, c_ops; method = :semiclassical, mode_ops = [a, b], mode_dims = collect(dims), limits = (6.0, 3.0)
        )
        α, β = result.centers[1, :], result.centers[2, :]
        @test size(result.shifts) == (4, size(result.centers, 2))
        @test result.shifts[1, :] ≈ -sqrt(0.4) * α
        @test result.shifts[2, :] ≈ -sqrt(0.3) * β
        @test result.shifts[3, :] ≈ -sqrt(0.05) * α .^ 2
        @test result.shifts[4, :] ≈ -sqrt(0.02) * abs2.(α)
        @test maximum(s -> s.residual, result.diagnostics.stability) < 1.0e-9
    end

    # The coherent-state expectation is the normal-ordered symbol, whatever the order in which
    # an operator is written, so nonlinear dissipators give the usual mean-field terms.
    @testset "drift of nonlinear dissipators" begin
        N, α = 40, 0.9 + 0.5im
        n, a = abs2(α), destroy(N)
        function drift(c_ops)
            p = (; dims = [N], drifts = [SM._heisenberg_drift(0 * a, c_ops, a).data])
            F = SM._meanfield_drift([real(α), imag(α)], p)
            return complex(F[1], F[2])
        end
        @test drift([sqrt(0.3) * a^2]) ≈ -0.3 * n * α
        @test drift([sqrt(0.4) * a' * a]) ≈ -0.2 * α
        @test drift([sqrt(0.25) * a']) ≈ 0.125 * α
        @test drift([sqrt(0.1) * a^3]) ≈ -0.15 * n^2 * α

        ψ = SM._coherent_product([N], [α])
        @test dot(ψ, (a * a').data * ψ) ≈ n + 1
        @test dot(ψ, (a * a' * a * a').data * ψ) ≈ n^2 + 3n + 1
    end

    @testset "input errors" begin
        a = destroy(6)
        discover(; kw...) = discover_gauges(a' * a, [a]; method = :semiclassical, mode_ops = [a], mode_dims = [6], limits = (1.0,), kw...)
        for limits in (nothing, 1.0, (), (0.0,), (-1.0,), (Inf,), (NaN,), (1.0, 1.0))
            @test_throws "occupation bound" discover(; limits)
        end
        @test_throws DimensionMismatch discover(mode_dims = [7])
        @test_throws UndefKeywordError discover_gauges(a' * a, [a]; method = :semiclassical)
        # Gain only: the fixed point at the origin is unstable.
        @test_throws "no stable fixed points" discover_gauges(
            0 * a, [a']; method = :semiclassical, mode_ops = [a], mode_dims = [6], limits = (1.0,)
        )
    end
end
