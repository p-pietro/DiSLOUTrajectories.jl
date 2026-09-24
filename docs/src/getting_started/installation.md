# Installation

`DiSLOUTrajectories.jl` requires Julia 1.10 or later and QuantumToolbox 0.47.2 to 0.49. To install it, run in an interactive session (REPL):

```julia
using Pkg
Pkg.add("DiSLOUTrajectories")
```

Alternatively, this can also be done in Julia's [Pkg REPL](https://julialang.github.io/Pkg.jl/v1/getting-started/) by pressing the key `]` in the REPL to use the package mode, and then type the following command:

```julia-REPL
(1.10) pkg> add DiSLOUTrajectories
```

More information about `Julia`'s package manager can be found at [`Pkg.jl`](https://julialang.github.io/Pkg.jl/v1/).

To load the package and check the version information, use either [`DiSLOUTrajectories.versioninfo()`](@ref) or [`DiSLOUTrajectories.about()`](@ref), namely

```julia
using DiSLOUTrajectories
DiSLOUTrajectories.versioninfo()
DiSLOUTrajectories.about()
```

## [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl)

`DiSLOUTrajectories.jl` is built upon `QuantumToolbox.jl`, which is a cutting-edge Julia package designed for quantum physics simulations, closely emulating the popular Python [`QuTiP`](https://qutip.org/) package. It provides many useful functions to create arbitrary quantum states and operators which can be combined in all the expected ways. It uniquely combines the simplicity and power of Julia with advanced features like GPU acceleration and distributed computing, making simulation of quantum systems more accessible and efficient.

## Optional extensions

Automatic gauge discovery via initial trajectories clusters the terminal amplitudes of its preliminary runs
with DBSCAN. Install [```Clustering.jl```](https://github.com/JuliaStats/Clustering.jl) and load it to activate this method:

```julia
using Pkg
Pkg.add("Clustering")
using Clustering
```

Semiclassical gauge discovery requires instead [```QuantumCumulants.jl```](https://qojulia.github.io/QuantumCumulants.jl) to derive the semiclassical equations of motion.
Install ```QuantumCumulants.jl``` and load it to activate this method:

```julia
using Pkg
Pkg.add("QuantumCumulants")
using QuantumCumulants
```

QuantumCumulants 0.7 currently requires QuantumToolbox 0.47, so installing it
selects that version.

## GPUs

`dislou_solve` keeps the array types of its inputs, so GPUs need no extension. With
[```CUDA.jl```](https://cuda.juliagpu.org/stable/), for example, QuantumToolbox's `cu`
moves the operators and the initial state to the GPU:

```julia
using CUDA
sol = dislou_solve(cu(H), cu(ψ0), tlist, cu.(c_ops); gauge_set)
```

The eigendecomposition of each gauge and the propagation of the trajectories then
run on the GPU. This relies on the dense linear algebra of the GPU package (`eigen`,
`qr` and triangular solves) and on GPU support in QuantumToolbox's `mcsolve`. It is
not tested in CI, which has no GPU.

Continue with the [quick start](quickstart.md).
