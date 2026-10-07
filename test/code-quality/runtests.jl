using Aqua
using DiSLOUTrajectories
using JET
using LinearAlgebra
using QuantumToolbox
using Random
using Test

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

include("../reporting/check.jl")

@testset "Code quality" begin
    @testset "Aqua" begin
        # Aqua's subprocess probe flakes on hosted Julia 1.12
        Aqua.test_all(DiSLOUTrajectories; persistent_tasks = false)
    end
    @testset "JET" begin
        JET.test_package(DiSLOUTrajectories; target_modules = (DiSLOUTrajectories,), ignore_missing_comparison = true, mode = :typo)

        # Error and optimization analysis of the kernels that run at every step or jump.
        N = 8
        rng = Xoshiro(1)
        A = randn(rng, ComplexF64, N, N)
        H = QuantumObject((A + A') / 2)
        c_ops = [QuantumObject(randn(rng, ComplexF64, N, N) / 3)]
        alg = GaugeEigenExponential(H, c_ops)
        basis = first(alg.bases)
        ψ = normalize(randn(rng, ComplexF64, N))
        C = [c.data for c in c_ops]
        a = destroy(N)
        drifts = [DiSLOUTrajectories._heisenberg_drift(a' * a, [a], a).data]
        residual = let dims = [N], drifts = drifts
            x -> DiSLOUTrajectories._residual(x, dims, drifts)
        end
        for (f, args) in (
                (DiSLOUTrajectories._coordinates!, (similar(ψ), basis, ψ)),
                (DiSLOUTrajectories._project!, (copy(ψ), basis, 1.0e-3, similar(ψ), similar(ψ))),
                (DiSLOUTrajectories._gauge_activities!, (zeros(2), C, zeros(ComplexF64, 1, 2), ψ, similar(ψ))),
                (DiSLOUTrajectories._dbscan, (randn(rng, 2, 20), 1.0, 3)),
                (DiSLOUTrajectories._coherent_product, ([N], zeros(2))),
                (DiSLOUTrajectories._residual, (zeros(2), [N], drifts)),
                (DiSLOUTrajectories._newton_root, (residual, zeros(2))),
                (DiSLOUTrajectories._phase_space_points, (residual, [1.0])),
            )
            @testset "$(nameof(f))" begin
                JET.test_call(f, typeof.(args); target_modules = (DiSLOUTrajectories,), mode = :basic)
                JET.test_opt(f, typeof.(args); target_modules = (DiSLOUTrajectories,))
            end
        end

        @testset "_cluster_terminal_means" begin
            points = randn(rng, ComplexF64, 2, 20)
            JET.@test_opt target_modules = (DiSLOUTrajectories,) DiSLOUTrajectories._cluster_terminal_means(
                points; cluster_scales = [1.0, 1.0], dbscan_radius = 1.0, min_neighbors = 3, min_weight = 0.1,
            )
        end
    end

    @testset "every export has a docstring" begin
        # Inspect registered docstrings directly; Docs.hasdoc requires Julia 1.11.
        undocumented = sort!(
            String[
                string(name) for name in names(DiSLOUTrajectories; all = false, imported = false)
                    if name != :DiSLOUTrajectories &&   # the module, which has no docstring
                    !haskey(Docs.meta(DiSLOUTrajectories), Docs.Binding(DiSLOUTrajectories, name))
            ]
        )
        @test isempty(undocumented)
    end

    # Documenter is configured to fail on undocumented exports, failing
    # doctests, and broken links; assert that configuration is still in place
    # rather than re-implementing those checks here.
    @testset "documentation build stays strict" begin
        makefile = read(joinpath(ROOT, "docs", "make.jl"), String)
        @test occursin(r"checkdocs\s*=\s*:exports", makefile)
        @test occursin(r"doctest\s*=\s*true", makefile)
        @test !occursin("warnonly", makefile)
    end

    # A new test_*.jl file that nobody wired into runtests.jl runs nowhere.
    # The GPU and multi-process suites are owned by dedicated CI jobs.
    @testset "every package test file runs somewhere" begin
        extended = Set(["test_gpu.jl", "test_distributed.jl"])
        runtests = read(joinpath(ROOT, "test", "runtests.jl"), String)
        included = Set(
            match.captures[1] for match in
                eachmatch(r"""include\("([^"]+)"\)""", runtests)
        )
        @test all(path -> isfile(joinpath(ROOT, "test", path)), included)
        present = Set(
            filter(
                path -> startswith(path, "test_") && endswith(path, ".jl"),
                readdir(joinpath(ROOT, "test")),
            )
        )
        @test setdiff(present, included) == extended
        @test all(path -> isfile(joinpath(ROOT, "test", path)), extended)
    end
end
