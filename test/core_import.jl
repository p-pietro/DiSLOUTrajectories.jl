using Test
using DiSLOUTrajectories

@testset "Core-only import" begin
    @test Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesClusteringExt) === nothing
    @test Base.get_extension(DiSLOUTrajectories, :DiSLOUTrajectoriesQuantumCumulantsExt) === nothing
    loaded = Set(pkg.name for pkg in keys(Base.loaded_modules))
    @test "Clustering" ∉ loaded
    @test "Distances" ∉ loaded
    @test "QuantumCumulants" ∉ loaded
    @test "ModelingToolkitBase" ∉ loaded
    for (method, message) in (
            :trajectories => "using Clustering",
            :semiclassical => "using QuantumCumulants",
            :unknown => "unknown gauge discovery method :unknown",
        )
        @test_throws message discover_gauges(nothing, []; method)
    end
end
