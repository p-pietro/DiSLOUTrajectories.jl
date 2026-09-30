using Test
using DiSLOUTrajectories
import Clustering  # activates the extension used by trajectory gauge discovery
using QuantumToolbox
using LinearAlgebra
using Random
using Statistics
import SciMLBase
import SciMLBase: EnsembleSerial, EnsembleThreads

const SM = DiSLOUTrajectories
const ClusteringExt = Base.get_extension(SM, :DiSLOUTrajectoriesClusteringExt)

include(joinpath(@__DIR__, "fixtures", "driven_kerr_model.jl"))

quiet = (progress_bar = Val(false),)

# Resonantly driven cavity, initially empty; `α` is its steady-state amplitude, and
# `steady_gauge` the gauge centered on it.
function driven_cavity(; N = 50, F = 1.0, Δ = 0.5, κ = 0.1)
    a = destroy(N)
    H = Δ * a' * a + F * (a + a')
    α = -im * F / (κ / 2 + im * Δ)
    return (; a, H, c_ops = [sqrt(κ) * a], ψ0 = fock(N, 0), α, κ, steady_gauge = fill(-sqrt(κ) * α, 1, 1))
end

# dislou_solve on the driven cavity `m`, with the zero gauge unless `gauge_set` is given.
cavity_solve(m, tlist; kw...) =
    dislou_solve(m.H, m.ψ0, tlist, m.c_ops; gauge_set = zeros(ComplexF64, 1, 1), quiet..., kw...)

# Whether the average of the first observable of `sol` is within 5 standard errors
# of `reference`, up to `atol`.
function within_errors(sol, reference; atol)
    n = real.(average_expect(sol)[1, :])
    sem = std_expect(sol)[1, :] ./ sqrt(sol.ntraj)
    return all(abs.(n .- reference) .<= 5 .* sem .+ atol)
end

# The exception thrown by `f()`, or `nothing` if it returns.
function thrown(f)
    try
        f()
    catch err
        return err
    end
    return nothing
end

# Record the gauge (and the Layer III flag) of each trajectory after every step.
# The callback runs after the gauge router; use it with `EnsembleSerial()`.
function gauge_recorder()
    gauges = Int[]
    reduced = Bool[]
    affect!(integrator) = begin
        push!(gauges, integrator.cache.gauge)
        push!(reduced, integrator.cache.reduced)
        SciMLBase.derivative_discontinuity!(integrator, false)
    end
    callback = SciMLBase.DiscreteCallback((u, t, integrator) -> true, affect!; save_positions = (false, false))
    return callback, gauges, reduced
end
