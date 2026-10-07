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
import ForwardDiff
import SimpleNonlinearSolve: SimpleTrustRegion

export dislou_solve, discover_gauges, GaugeEigenExponential

include("info.jl")
include("gauges.jl")
include("eigen_exponential.jl")
include("gauge_router.jl")
include("solve.jl")
include("gauge_discovery.jl")
include("dbscan.jl")
include("trajectory_discovery.jl")
include("semiclassical_discovery.jl")

end # module
