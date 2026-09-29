# Layer I at the jumps. `mcsolve` finds the jump times from the norm of the state, and
# these methods give the jump weights and the jump, with the collapse operators of the
# current gauge, for the integrators of `GaugeEigenExponential`.

const _GaugeIntegrator = SciMLBase.DEIntegrator{<:GaugeEigenExponential}

# ‖C_μ^(g) ψ‖² in the current gauge g.
function QuantumToolbox._mcsolve_jump_weights!(weights, _, cache_mc, integrator::_GaugeIntegrator)
    (; alg, cache, u) = integrator
    g = cache.gauge
    for μ in eachindex(weights)
        ψ′ = _jump_image!(cache_mc, alg.C[μ], alg.shifts[μ, g], u)
        weights[μ] = real(dot(ψ′, ψ′))
    end
    return weights
end

function QuantumToolbox._mcsolve_jump!(integrator::_GaugeIntegrator, _, μ, cache_mc)
    (; alg, cache, u) = integrator
    copyto!(u, normalize!(_jump_image!(cache_mc, alg.C[μ], alg.shifts[μ, cache.gauge], u)))
    _switch_gauge!(cache, alg, u)
    copyto!(cache.ulast, u)
    return nothing
end

# (C + ζ) ψ, the jump by C in a gauge that shifts it by ζ.
function _jump_image!(out, C, ζ, ψ)
    mul!(out, C, ψ)
    @. out += ζ * ψ
    return out
end
