# API reference

## Paper notation

The notation follows that of the [reference paper](./getting_started/cite.md) for the DiSLOU method.

| Code | Paper quantity |
| :----- | :--------------- |
| `gauge_set[μ, g]` or `gauges.shifts[μ, g]` | ``ζ_μ^{(g)}`` (Eq. 9) |
| `hysteresis` | ``η`` (Eq. 13) |
| `layer3_sizes[g]` | Requested ``m^{(g)}``; degenerate modes can increase it (Eq. 20) |
| `residual_tolerance` | ``r_{tol}``, applied to ``r_m^{(g)}(ψ)`` for normalized states (Eq. 21) |
| `sol.states`, `average_states(sol)` | ``ρ_{MC}(t)`` (Section 4.1) |
| `sol.expect` | Ensemble ``⟨O⟩(t)`` |

## Solver

```@docs
dislou_solve
GaugeEigenExponential
```

### Examples

[`dislou_solve`](@ref) accepts the keyword arguments of QuantumToolbox's `mcsolve`
and returns its `TimeEvolutionMCSol`. Here the states are saved at two times, and
the results of every trajectory are kept so that their spread can be computed.

```jldoctest
using QuantumToolbox, DiSLOUTrajectories, Random

N, κ, α = 15, 1.0, 1.0
a = destroy(N)
H = 0.5im * κ * (α * a' - conj(α) * a)
tlist = 0:0.5:10

sol = dislou_solve(H, fock(N, 0), tlist, [sqrt(κ) * a];
                   gauge_set = fill(-sqrt(κ) * α, 1, 1), e_ops = [a' * a],
                   ntraj = 50, rng = Xoshiro(1), saveat = [5.0, 10.0],
                   keep_runs_results = Val(true), progress_bar = Val(false))

(size(sol.expect), size(sol.states), length(average_states(sol)), size(std_expect(sol)))

# output

((1, 50, 21), (50, 2), 2, (1, 21))
```

## Gauge discovery

```@docs
discover_gauges
```

## Package information

```@docs
DiSLOUTrajectories.versioninfo
DiSLOUTrajectories.about
DiSLOUTrajectories.cite
```
