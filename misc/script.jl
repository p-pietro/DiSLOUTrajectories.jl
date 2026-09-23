using QuantumToolbox
using DiSLOUTrajectories
using Random

# %%

N = 50
F = 1.0
Δ = 0.5
κ = 0.1

a = destroy(N)

tlist = range(0, 10, 100)

ψ0 = fock(N, 0)

# %%

H = Δ * a' * a + F * (a + a')
c_ops = [sqrt(κ) * a]

sol_me = mesolve(H, ψ0, tlist, c_ops; e_ops = [a' * a])

# %%

αss = -im * F / (κ / 2 + im * Δ)

gauges = fill(-sqrt(κ) * αss, 1, 1) .* 0

sol_mc = mcsolve(H, ψ0, tlist, c_ops; e_ops = [a' * a], ntraj = 100, rng = MersenneTwister(1))
sol_dislo = dislou_solve(H, ψ0, tlist, c_ops; e_ops = [a' * a], gauge_set = gauges, ntraj = 100, rng = MersenneTwister(1))

@show sum(length, sol_mc.col_times)
@show sum(length, sol_dislo.col_times)
