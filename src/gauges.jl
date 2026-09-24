# Layer I: gauges (paper Eq. 9) and the activity used to choose between them
# (Eqs. 11–13).

# Shift matrix ζ[μ, g] (one row per collapse operator, one column per gauge),
# given directly or as the result of `discover_gauges`.
function _gauge_shifts(gauge_set, nchannels::Int)
    raw = gauge_set isa AbstractMatrix ? gauge_set :
        hasproperty(gauge_set, :shifts) ? gauge_set.shifts :
        throw(ArgumentError("gauge_set must be a shift matrix or a discover_gauges result"))
    shifts = Matrix{ComplexF64}(raw)
    size(shifts, 1) == nchannels || throw(
        DimensionMismatch(
            "gauge_set must have one row per collapse operator ($nchannels), got $(size(shifts, 1))"
        )
    )
    size(shifts, 2) >= 1 || throw(ArgumentError("gauge_set must contain at least one gauge (column)"))
    all(isfinite, shifts) || throw(ArgumentError("gauge_set must contain only finite shifts"))
    return shifts
end

# Paper Eq. (9): C_μ^(g) = C_μ + ζ_μ and H^(g) = H + (i/2) Σ_μ (ζ_μ C_μ† - ζ_μ* C_μ).
# The shifted operators generate the same Lindblad equation as (H, c_ops), with
# the array types and precision of H.
function _shifted_operators(H, c_ops, ζ)
    ζ = convert(Vector{complex(eltype(H.data))}, ζ)
    Hg = H + im * sum(ζ[μ] * c_ops[μ]' - conj(ζ[μ]) * c_ops[μ] for μ in eachindex(c_ops)) / 2
    Cg = [c_ops[μ] + ζ[μ] * qeye_like(c_ops[μ]) for μ in eachindex(c_ops)]
    return Hg, Cg
end

# Paper Eq. (11) for all gauges. With the moments of the physical operators C_μ,
# A_g(ψ) = Σ_μ ‖(C_μ + ζ_μg) ψ‖² = Σ_μ (⟨C_μ†C_μ⟩ + 2 Re(ζ_μg* ⟨C_μ⟩) + |ζ_μg|²).
function _gauge_activities(C, shifts, ψ, tmp)
    norm2 = real(dot(ψ, ψ))
    A = zeros(size(shifts, 2))
    for μ in eachindex(C)
        mul!(tmp, C[μ], ψ)
        mean = dot(ψ, tmp) / norm2
        second = real(dot(tmp, tmp)) / norm2
        for g in eachindex(A)
            ζ = shifts[μ, g]
            A[g] += second + 2 * real(conj(ζ) * mean) + abs2(ζ)
        end
    end
    return A
end

# Paper Eq. (13): move to the least active gauge only if its activity is below
# η times the activity of the current gauge.
function _next_gauge(A, current, η)
    best = argmin(A)
    return A[best] < η * A[current] ? best : current
end
