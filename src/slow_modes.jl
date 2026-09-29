# Layer III: the slow modes of a gauge g, and the m × m matrices that give a jump and the
# expectation values from the coordinates q = Q†ψ, in O(m²) operations (paper Appendix C).
# The jump by C_μ^(g) = C_μ + ζ_μg gives C_μ^(g) ψ = X_μ q, with X_μ = C_μ^(g) Q.
struct SlowModes{B <: EigenBasis, M <: AbstractMatrix}
    basis::B
    weights::Vector{M}       # [μ]: X_μ†X_μ, so that ‖C_μ^(g) ψ‖² = q† X_μ†X_μ q
    activities::Matrix{M}    # [μ, d]: activity of the gauge d after the jump μ, times ‖X_μ q‖²
    residuals::Matrix{M}     # [μ, d]: ‖(1 - Q_d Q_d†) X_μ q‖², Q_d being the slow modes of d
    coordinates::Matrix{M}   # [μ, d]: Q_d† X_μ, the coordinates of X_μ q on the slow modes of d
    e_ops::Vector{M}         # [k]: Q† O_k Q
end

function _slow_modes(bases, sizes, C, shifts, e_ops)
    slow = map(_slow_basis, bases, sizes)
    return [SlowModes(slow, g, C, shifts, e_ops) for g in eachindex(slow)]
end

# The matrices of the gauge `g`, given the slow modes `slow` of all the gauges.
function SlowModes(slow::AbstractVector{<:EigenBasis}, g::Integer, C, shifts, e_ops)
    Q = slow[g].Q
    M = typeof(Q)
    ζ = convert(Matrix{eltype(Q)}, shifts)
    X = [C[μ] * Q + ζ[μ, g] * Q for μ in eachindex(C)]
    coordinates = M[s.Q' * Xμ for Xμ in X, s in slow]
    return SlowModes(
        slow[g],
        M[Xμ' * Xμ for Xμ in X],
        M[_activity_matrix(Xμ, C, view(ζ, :, d)) for Xμ in X, d in eachindex(slow)],
        M[_residual_matrix(X[μ], s.Q, coordinates[μ, d]) for μ in eachindex(X), (d, s) in enumerate(slow)],
        coordinates,
        M[Q' * (O * Q) for O in e_ops],
    )
end

# Σ_ν ‖(C_ν + ζ_ν) X q‖² = q† (Σ_ν Y_ν†Y_ν) q, with Y_ν = (C_ν + ζ_ν) X (paper Eq. 11).
_activity_matrix(X, C, ζ) = sum(eachindex(C)) do ν
    Y = C[ν] * X + ζ[ν] * X
    return Y' * Y
end

# ‖(1 - Q Q†) X q‖² = q† Y†Y q, with Y = X - Q (Q†X) computed first to keep the digits of
# small residuals.
function _residual_matrix(X, Q, QX)
    Y = X - Q * QX
    return Y' * Y
end
