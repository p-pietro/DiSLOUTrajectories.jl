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

# Whether QuantumToolbox has just recorded a jump at time `t`.
function _jumped_at(integrator, t)
    jump = _jump_state(integrator)
    n = jump.col_times_which_idx[] - 1
    return n > 0 && jump.col_times[n] == t
end

# Condition: a jump has just happened, or is due now.
(router::GaugeRouter)(u, t, integrator) = _jumped_at(integrator, t) || t == integrator.cache.jump_time

# Affect: jump if the jump callback has not, switch gauge (paper Eq. 13) and, with
# Layer III, project on its slow modes.
function (router::GaugeRouter)(integrator)
    cache = integrator.cache
    _jumped_at(integrator, integrator.t) || _jump_state(integrator)(integrator)
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
    _schedule_jump!(integrator)
    return nothing
end

function _initialize_router(callback, u, t, integrator)
    router = callback.affect!
    integrator.cache.gauge = router.initial_gauge
    integrator.cache.reduced = router.initial_reduced
    copyto!(_jump_state(integrator).c_ops, router.jump_ops[router.initial_gauge])
    _schedule_jump!(integrator)
    return nothing
end

# Layers II and III: the next jump happens when ‖ψ‖² falls to the random threshold r of
# the jump callback. Until then ψ(t + τ) = V e^{Λτ} c with V = QR, so ‖ψ‖ = ‖R e^{Λτ} c‖:
# the jump time is found on the m coordinates, at O(m²) per evaluation instead of O(Nm)
# for ψ, and becomes a stop time. There, the discrete jump callback of `mcsolve` jumps
# if ‖ψ‖² < r, and the router otherwise, since rounding can leave ‖ψ‖² just above r.
# The coordinates are handed to the next step.
function _schedule_jump!(integrator)
    cache = integrator.cache
    basis = _active_basis(integrator.alg, cache)
    c = _coordinates!(view(cache.c_next, 1:length(basis)), basis, integrator.u)
    cache.basis_next = basis
    copyto!(cache.ulast, integrator.u)
    r = _jump_state(integrator).random_n[]
    y = view(cache.c_scratch, 1:length(basis))
    excess(τ, _ = nothing) = sum(abs2, lmul!(basis.R, (@. y = exp(basis.λ * τ) * c))) - r
    horizon = last(integrator.sol.prob.tspan) - integrator.t
    cache.jump_time = excess(horizon) < 0 < excess(0) ?
        integrator.t + solve(IntervalNonlinearProblem{false}(excess, (zero(horizon), horizon)), ModAB()).right : Inf
    isfinite(cache.jump_time) && add_tstop!(integrator, cache.jump_time)
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
