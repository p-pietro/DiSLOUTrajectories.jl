include("testsetup.jl")

@testset "DiSLOUTrajectories" begin
    include("test_eigen_exponential.jl")
    include("test_dislou_solve.jl")
    include("test_inputs.jl")
    include("test_gauge_discovery.jl")
    include("test_semiclassical.jl")
    include("test_distributed.jl")
    include("test_gpu.jl")   # skipped without a functional CUDA GPU
end
