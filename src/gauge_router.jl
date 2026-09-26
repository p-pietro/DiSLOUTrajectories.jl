# Layer I, together with the projections of Layer III: after each quantum jump,
# choose the gauge and the basis used until the next jump. The router is a
# `DiscreteCallback` shared by all trajectories, while the choice of each
# trajectory lives in its `GaugeEigenExponential` cache.

struct GaugeRouter{TC, TJ}
    C::Vector{TC}                 # physical collapse operators, for the activities
    shifts::Matrix{ComplexF64}    # ζ[μ, g]
    jump_ops::Vector{TJ}          # C^(g) of each gauge, in the form used by the jump callback
    hysteresis::Float64           # η
    residual_tolerance::Float64   # Layer III r_tol
    initial_gauge::Int
    initial_reduced::Bool
end

# State of QuantumToolbox's jump callback. It is internal to QuantumToolbox, and
# `dsf_mcsolve` accesses it in the same way.
_jump_state(integrator) = QuantumToolbox._mc_get_jump_callback(integrator).affect!

# Condition: QuantumToolbox has just recorded a jump at the current time.
function (router::GaugeRouter)(u, t, integrator)
    jump = _jump_state(integrator)
    n = jump.col_times_which_idx[] - 1
    return n > 0 && jump.col_times[n] == t
end

# Affect: switch gauge (paper Eq. 13) and, with Layer III, project on its slow modes.
function (router::GaugeRouter)(integrator)
    cache = integrator.cache
    A = _gauge_activities!(cache.activities, router.C, router.shifts, integrator.u, cache.tmp)
    gauge = _next_gauge(A, cache.gauge, router.hysteresis)
    if gauge != cache.gauge
        cache.gauge = gauge
        copyto!(_jump_state(integrator).c_ops, router.jump_ops[gauge])
    end
    reduced_bases = integrator.alg.reduced_bases
    cache.reduced = !isempty(reduced_bases) && _project!(
        integrator.u, reduced_bases[gauge], router.residual_tolerance, cache.c_scratch, cache.tmp
    )
    derivative_discontinuity!(integrator, cache.reduced)
    return nothing
end

function _initialize_router(callback, u, t, integrator)
    router = callback.affect!
    integrator.cache.gauge = router.initial_gauge
    integrator.cache.reduced = router.initial_reduced
    copyto!(_jump_state(integrator).c_ops, router.jump_ops[router.initial_gauge])
    return nothing
end

_router_callback(router::GaugeRouter) =
    DiscreteCallback(router, router; initialize = _initialize_router, save_positions = (false, false))

# Layer III (paper Eq. 21): if the relative residual ‖ψ - QQ†ψ‖/‖ψ‖ is at most
# `tolerance`, replace ψ by its projection QQ†ψ on the slow modes, rescaled to the
# norm of ψ (1 after a jump). Returns whether ψ was projected.
function _project!(ψ, basis::EigenBasis, tolerance, coefficients, residual)
    q = view(coefficients, 1:length(basis))
    mul!(q, basis.Q', ψ)
    mul!(copyto!(residual, ψ), basis.Q, q, -1, 1)
    nψ = norm(ψ)
    norm(residual) <= tolerance * nψ || return false
    mul!(ψ, basis.Q, rmul!(q, nψ / norm(q)))
    return true
end
