# Layers I and III at the jumps. `mcsolve` finds the jump times from the norm of the
# state, and these methods give the jump weights, the jump and the expectation values for
# the integrators of `GaugeEigenExponential`.

const _GaugeIntegrator = SciMLBase.DEIntegrator{<:GaugeEigenExponential}

# Root conditions need only the spectral norm, not an interpolated physical state.
QuantumToolbox._mcsolve_jump_condition_state(integrator::_GaugeIntegrator, callback, t) = nothing

function QuantumToolbox._mcsolve_jump_survival(u, t, integrator::_GaugeIntegrator)
    cache = integrator.cache
    cache.condition_valid && cache.condition_time == t && return cache.survival
    t == integrator.t && return real(dot(integrator.u, integrator.u))
    t == integrator.tprev && return real(dot(integrator.uprev, integrator.uprev))
    return first(_jump_survival_and_rate!(integrator, t))
end

QuantumToolbox._mcsolve_jump_rate(jump, u, t, integrator::_GaugeIntegrator) =
    last(_jump_survival_and_rate!(integrator, t))

# One Gd product for survival; its derivative adds only an O(n) contraction.
function _jump_survival_and_rate!(integrator, t)
    cache = integrator.cache
    cache.condition_valid && cache.condition_time == t && return (cache.survival, cache.rate)
    basis = cache.basis
    n = length(basis)
    d = view(cache.c_scratch, 1:n)
    if t == integrator.t
        copyto!(d, view(cache.c_next, 1:n))
    else
        c = view(cache.c, 1:n)
        @. d = exp(basis.λ * (t - integrator.tprev)) * c
    end
    jump = QuantumToolbox._mc_get_jump_callback(integrator).affect!
    z = mul!(view(jump.cache_mc, 1:n), basis.G, d)
    cache.survival, cache.rate = _survival_and_rate!(d, z, basis.λ)
    cache.condition_time = t
    cache.condition_valid = true
    return (cache.survival, cache.rate)
end

function _survival_and_rate!(d, z, λ)
    survival = real(dot(d, z))
    @. d *= λ
    return (survival, -2 * real(dot(z, d)))
end

# ‖C_μ^(g) ψ‖² in the current gauge g.
function QuantumToolbox._mcsolve_jump_weights!(weights, _, cache_mc, integrator::_GaugeIntegrator)
    (; alg, cache, u) = integrator
    g = cache.gauge
    if cache.coordinates
        slow = alg.slow[g]
        n = length(slow.basis)
        q = view(u, 1:n)
        tmp = view(cache.tmp, 1:n)
        for μ in eachindex(weights, slow.weights)
            weights[μ] = _quadratic(slow.weights[μ], q, tmp)
        end
    else
        for μ in eachindex(weights)
            ψ′ = _jump_image!(cache_mc, alg.C[μ], alg.shifts[μ, g], u)
            weights[μ] = real(dot(ψ′, ψ′))
        end
    end
    return weights
end

function QuantumToolbox._mcsolve_jump!(integrator::_GaugeIntegrator, _, μ, cache_mc)
    (; alg, cache, u) = integrator
    if cache.coordinates
        _slow_jump!(cache, alg, u, μ)
    else
        copyto!(u, normalize!(_jump_image!(cache_mc, alg.C[μ], alg.shifts[μ, cache.gauge], u)))
        _switch_gauge!(cache, alg, u)
    end
    copyto!(cache.ulast, u)
    return nothing
end

# ⟨ψ|O|ψ⟩ / ⟨ψ|ψ⟩, or q† (Q†OQ) q / ‖q‖² from the coordinates q = Q†ψ.
function QuantumToolbox._mcsolve_expect!(expvals, e_ops, u, integrator::_GaugeIntegrator)
    # Inside a step, the dense output may already follow a later jump of that step.
    u === integrator.u || throw(
        ArgumentError(
            "GaugeEigenExponential computes expectation values only at step ends: pass the \
            times of `tlist` as `tstops`"
        )
    )
    (; alg, cache) = integrator
    cache.coordinates || return expvals .= dot.(Ref(u), e_ops, Ref(u)) ./ real(dot(u, u))
    slow = alg.slow[cache.gauge]
    n = length(slow.basis)
    q = view(u, 1:n)
    tmp = view(cache.c_scratch, 1:n)
    norm2 = real(dot(q, q))
    for k in eachindex(expvals, slow.e_ops)
        expvals[k] = dot(q, mul!(tmp, slow.e_ops[k], q)) / norm2
    end
    return expvals
end

# The jump μ from the coordinates q on the slow modes of the gauge g, in O(m²). The new
# state X_μ q is built only when it is too far from the slow modes of its next gauge d.
function _slow_jump!(cache, alg, u, μ)
    g = cache.gauge
    slow = alg.slow[g]
    n = length(slow.basis)
    q = view(u, 1:n)
    tmp = view(cache.tmp, 1:n)
    weight = _quadratic(slow.weights[μ], q, tmp)
    A = cache.activities
    for d in eachindex(A)
        A[d] = _quadratic(slow.activities[μ, d], q, tmp) / weight
    end
    d = _next_gauge(A, g, alg.hysteresis)
    if _quadratic(slow.residuals[μ, d], q, tmp) <= alg.residual_tolerance^2 * weight
        q′ = view(cache.c_scratch, 1:length(alg.slow[d].basis))
        _enter_slow_modes!(cache, alg, d, normalize!(mul!(q′, slow.coordinates[μ, d], q)))
        _write_state!(u, cache.basis, view(cache.c_next, 1:length(q′)), true)
        cache.coordinates = true
    else
        ψ = mul!(cache.tmp, slow.basis.Q, q)
        normalize!(_jump_image!(u, alg.C[μ], alg.shifts[μ, g], ψ))
        _enter_all_modes!(cache, alg, d, u)
        cache.coordinates = false
    end
    return nothing
end

# (C + ζ) ψ, the jump by C in a gauge that shifts it by ζ.
function _jump_image!(out, C, ζ, ψ)
    mul!(out, C, ψ)
    @. out += ζ * ψ
    return out
end

_quadratic(A, x, tmp) = real(dot(x, mul!(tmp, A, x)))
