"""
    DiSLOUTrajectories.versioninfo(io::IO = stdout)

Command line output of the version of DiSLOUTrajectories.jl, followed by
`QuantumToolbox.versioninfo`, with the dependencies and the system. Same as
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

Command line output of the BibTeX entry for the DiSLOUTrajectories.jl publication.
"""
function cite(io::IO = stdout)
    citation = raw"""
    @article{Pacchioni2026Diagonal,
      title = {Diagonal, Switching, and Locally Optimal Unraveling for efficient quantum trajectories in metastable open quantum systems},
      author = {Pacchioni, Pietro and Mercurio, Alberto and Winkel, Patrick and Minganti, Fabrizio},
      journal = {arXiv preprint arXiv:2610.36067},
      year = {2026},
      eprint = {2610.36067},
      archivePrefix = {arXiv},
      primaryClass = {quant-ph},
      doi = {10.48550/arXiv.2610.36067},
      url = {https://arxiv.org/abs/2610.36067}
    }
    """
    return print(io, citation)
end
