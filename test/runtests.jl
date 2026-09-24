if haskey(ENV, "DISLOU_APPLE_ACCELERATE_ENV")
    push!(LOAD_PATH, ENV["DISLOU_APPLE_ACCELERATE_ENV"])
    using AppleAccelerate, LinearAlgebra
    @assert any(lib -> occursin("Accelerate", lib.libname), BLAS.get_config().loaded_libs)
    @info "CI BLAS backend" config = BLAS.get_config()
end

include("testsetup.jl")

# The optional-dependency and multi-process suites (test_semiclassical.jl,
# test_gpu.jl, test_distributed.jl) run in dedicated CI jobs.
@testset "DiSLOUTrajectories" begin
    include("test_eigen_exponential.jl")
    include("test_dislou_solve.jl")
    include("test_inputs.jl")
    include("test_gauge_discovery.jl")
end
