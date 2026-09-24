"""
    DiSLOUTrajectories.versioninfo(io::IO = stdout)

Command line output of the versions of DiSLOUTrajectories.jl and of its loaded extensions,
followed by `QuantumToolbox.versioninfo`, with the dependencies and the system. Same as
[`DiSLOUTrajectories.about`](@ref).
"""
function versioninfo(io::IO = stdout)
    println(
        io,
        "\n",
        " DiSLOUTrajectories.jl: Diagonal, Switching, and Locally Optimal Unraveling\n",
        "≡"^76, "\n",
        "Pietro Pacchioni and Fabrizio Minganti\n",
    )
    println(io, rpad("DiSLOUTrajectories", 20), " Ver. ", pkgversion(DiSLOUTrajectories))
    for (extension, package) in (
            :DiSLOUTrajectoriesClusteringExt => :Clustering,
            :DiSLOUTrajectoriesQuantumCumulantsExt => :QuantumCumulants,
        )
        ext = Base.get_extension(DiSLOUTrajectories, extension)
        ext === nothing || println(io, rpad(package, 20), " Ver. ", pkgversion(getproperty(ext, package)))
    end
    QuantumToolbox.versioninfo(io)
    println(
        io,
        "+-------------------------------------------------------+\n",
        "| Please cite DiSLOUTrajectories.jl in your publication |\n",
        "+-------------------------------------------------------+\n",
        "For your convenience, a bibtex reference can be easily generated using `DiSLOUTrajectories.cite()`.",
    )
    return nothing
end

"""
    DiSLOUTrajectories.about(io::IO = stdout)

Same as [`DiSLOUTrajectories.versioninfo`](@ref).
"""
about(io::IO = stdout) = versioninfo(io)

"""
    DiSLOUTrajectories.cite(io::IO = stdout)

Command line output of the BibTeX entries of DiSLOUTrajectories.jl and of its method paper.
"""
function cite(io::IO = stdout)
    citation = raw"""
    @software{Pacchioni2026DiSLOU,
      author = {Pietro Pacchioni and Fabrizio Minganti},
      title = {DiSLOUTrajectories.jl},
      year = {2026},
      license = {BSD-3-Clause},
      url = {https://github.com/p-pietro/DiSLOUTrajectories.jl}
    }

    @article{Pacchioni2026Diagonal,
      author = {Pietro Pacchioni and Patrick Winkel and Fabrizio Minganti},
      title = {Diagonal, Switching, and Locally Optimal Unraveling efficient quantum trajectories in metastable open quantum systems},
      year = {2026}
    }
    """
    return print(io, citation)
end
