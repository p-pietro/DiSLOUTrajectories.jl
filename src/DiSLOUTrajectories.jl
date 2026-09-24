"""
    DiSLOUTrajectories

Diagonal, Switching, and Locally Optimal Unraveling (DiSLOU) of time-independent
Lindblad models, built on QuantumToolbox's `mcsolve`.

[`dislou_solve`](@ref) runs `mcsolve` with the exact propagator
[`GaugeEigenExponential`](@ref) and a callback that switches gauge after each
jump. [`discover_gauges`](@ref) finds suitable gauges.
"""
module DiSLOUTrajectories

using LinearAlgebra
using Random
import Distributed
using QuantumToolbox
import SciMLBase
import SciMLBase: DiscreteCallback, CallbackSet, derivative_discontinuity!
import SciMLBase: EnsembleAlgorithm, EnsembleSerial, EnsembleThreads, EnsembleDistributed
import OrdinaryDiffEqCore
import SciMLOperators: cache_operator

export dislou_solve, discover_gauges, GaugeEigenExponential
export about, backend_info, cite, versioninfo

include("backend.jl")
include("gauges.jl")
include("eigen_exponential.jl")
include("gauge_router.jl")
include("solve.jl")
include("gauge_discovery.jl")

__init__() = Base.Experimental.register_error_hint(_discovery_error_hint, MethodError)

end # module
