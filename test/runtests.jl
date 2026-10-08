# On macOS, run the tests with Apple's Accelerate as the BLAS and LAPACK backend, as
# users can, to keep the package working with it.
if Sys.isapple()
    using AppleAccelerate
end

include("testsetup.jl")

function run_tests()
    return @testset "DiSLOUTrajectories" begin
        if Sys.isapple()
            @testset "Accelerate is the BLAS backend" begin
                @test any(lib -> occursin("Accelerate", lib.libname), BLAS.get_config().loaded_libs)
            end
        end
        include("test_eigen_exponential.jl")
        include("test_dislou_solve.jl")
        include("test_inputs.jl")
        include("test_gauge_discovery.jl")
        include("test_semiclassical.jl")
        include("test_distributed.jl")
        include("test_gpu.jl")   # skipped without a functional CUDA GPU
    end
end

if haskey(ENV, "JUNIT_OUTPUT")
    include("reporting.jl")
    with_junit(run_tests, ENV["JUNIT_OUTPUT"])
else
    run_tests()
end
