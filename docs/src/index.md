# DiSLOUTrajectories.jl: An efficient Julia implementation of the Diagonal, Switching, and Locally Optimal Unraveling (DiSLOUTrajectories) algorithm for quantum trajectories

```DiSLOUTrajectories.jl``` provides an efficient implementation of the
*Diagonal, Switching, and Locally Optimal Unraveling* method in ```Julia```. This approach is useful for the simulation of metastable open quanum systems using the quantum trajectories framework.

The package is built on top of [`QuantumToolbox.jl`](https://github.com/qutip/QuantumToolbox.jl): the solver runs ```mcsolve``` with an exact propagator, so it takes the same input objects and keyword arguments and returns the same solution:

```julia
sol = dislou_solve(H, ψ0, tlist, c_ops; gauge_set, e_ops)
```

```DiSLOUTrajectories.jl``` implements both [evolution](@ref Solver) according to the three layers that make up the algorithm, keeping Layer III approximate propagation optional, and [two approaches](@ref Gauge-discovery) to identify the most efficient gauges.

## Where to use DiSLOU

DiSLOU supports open systems described by a finite-dimensional, time-independent Lindblad, for which the diagonalization of the effective Hamiltonian fits in memory.
For the best performance, the system should exhibit metastability, and the quantum trajectories should remain close to the metastable states during most of the evolution.  A prototypical example of such a system is the driven-dissipative Kerr resonator in the bistable regime.

The required gauges can be found through automatic discovery through short initial trajectories, semiclassical discovery from symbolic bosonic constructors, or they can be provided manually. The solver (`dislou_solve`) returns expectation values, states and jump records,
averaged or for every trajectory, as `mcsolve` does. Refer to the [API reference](api.md) for further details.

## Read next

- [Read the paper where DiSLOU is introduced](https://arxiv.org/)
- [Install DiSLOUTrajectories.jl](getting_started/installation.md).
- [Run a first solve](getting_started/quickstart.md).
- [Explore the examples](examples.md).
- [Read the API reference](api.md).

The source code is available at <https://github.com/p-pietro/DiSLOUTrajectories.jl>.
