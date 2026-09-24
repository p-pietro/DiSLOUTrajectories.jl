@testset "input validation" begin
    m = driven_cavity(N = 5)
    run(; kw...) = cavity_solve(m, [0.0, 1.0]; ntraj = 2, kw...)

    @test_throws UndefKeywordError dislou_solve(m.H, m.ψ0, [0.0, 1.0], m.c_ops)
    @test_throws DimensionMismatch run(gauge_set = zeros(ComplexF64, 2, 1))
    @test_throws ArgumentError run(gauge_set = zeros(ComplexF64, 1, 0))
    @test_throws ArgumentError run(gauge_set = fill(NaN + 0im, 1, 1))
    @test_throws ArgumentError run(gauge_set = 1.0)
    @test_throws ArgumentError run(hysteresis = 0)
    @test_throws ArgumentError run(hysteresis = 1.5)
    @test_throws ArgumentError run(residual_tolerance = 0)
    @test_throws ArgumentError run(layer3_sizes = 0)
    @test_throws ArgumentError run(layer3_sizes = 6)
    @test_throws ArgumentError run(layer3_sizes = [1, 2])
    @test_throws ArgumentError run(alg = GaugeEigenExponential(m.H, m.c_ops))
    @test_throws ArgumentError run(jump_callback = ContinuousLindbladJumpCallback())
    @test_throws ArgumentError dislou_solve(
        m.H, m.ψ0, [0.0, 1.0], typeof(m.H)[]; gauge_set = zeros(ComplexF64, 0, 1)
    )
end

@testset "package information" begin
    info = sprint(SM.versioninfo)
    @test occursin("DiSLOUTrajectories   Ver. $(pkgversion(SM))", info)
    @test occursin("Clustering", info)   # loaded extension
    @test occursin("QuantumToolbox.jl", info)
    @test sprint(SM.about) == info
    @test occursin("@article", sprint(SM.cite))
end
