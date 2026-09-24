const _REPORT_EXTENSIONS = (
    (:DiSLOUTrajectoriesClusteringExt, :Clustering, :Distances),
    (:DiSLOUTrajectoriesQuantumCumulantsExt, :QuantumCumulants, :ModelingToolkitBase),
)

_report_extension_loaded(name::Symbol) = Base.get_extension(DiSLOUTrajectories, name) !== nothing

function _report_optional_versions(io::IO)
    for (extension, packages...) in _REPORT_EXTENSIONS
        module_ = Base.get_extension(DiSLOUTrajectories, extension)
        module_ === nothing && continue
        for package in packages
            println(io, "  ", rpad("$(package):", 20), Base.pkgversion(getproperty(module_, package)))
        end
    end
    return
end

function _report_blas_library()
    libraries = LinearAlgebra.BLAS.get_config().loaded_libs
    return isempty(libraries) ? "unknown" : join(string.(libraries), ", ")
end

const _CITATION_BIBTEX = raw"""@software{Pacchioni2026DiSLOU,
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
}"""

"""
    versioninfo(io::IO = stdout)

Print DiSLOUTrajectories.jl's version, runtime environment, optional-extension status, and
citation reminder to `io`.
"""
function versioninfo(io::IO = stdout)
    println(io, "DiSLOUTrajectories.jl — v", Base.pkgversion(DiSLOUTrajectories))
    println(io, "Diagonal, Switching, and Locally Optimal Unraveling")
    println(io, "Pietro Pacchioni and Fabrizio Minganti · BSD-3-Clause license")

    println(io, "\nVersions")
    println(io, "  ", rpad("Julia:", 20), VERSION)
    println(io, "  ", rpad("QuantumToolbox:", 20), Base.pkgversion(QuantumToolbox))
    _report_optional_versions(io)

    println(io, "\nSystem")
    println(io, "  ", rpad("OS / architecture:", 20), Sys.KERNEL, " / ", Sys.ARCH)
    println(io, "  ", rpad("CPU:", 20), Sys.CPU_NAME, ", ", Sys.CPU_THREADS, " logical cores")
    println(io, "  ", rpad("Memory:", 20), round(Sys.total_memory() / 2^30; digits = 1), " GiB")

    println(io, "\nExecution")
    println(io, "  ", rpad("Julia threads:", 20), Threads.nthreads())
    println(io, "  ", rpad("Distributed workers:", 20), Distributed.nworkers())
    println(io, "  ", rpad("BLAS:", 20), _report_blas_library(), ", ", LinearAlgebra.BLAS.get_num_threads(), " threads")

    println(io, "\nOptional features")
    println(io, "  ", rpad("Clustering extension:", 28), _report_extension_loaded(:DiSLOUTrajectoriesClusteringExt) ? "loaded" : "not loaded")
    println(io, "  ", rpad("Semiclassical extension:", 28), _report_extension_loaded(:DiSLOUTrajectoriesQuantumCumulantsExt) ? "loaded" : "not loaded")

    println(io, "\nDocumentation: https://p-pietro.github.io/DiSLOUTrajectories.jl/")
    println(io, "Repository:    https://github.com/p-pietro/DiSLOUTrajectories.jl")
    println(io, "Citation:      run cite() for BibTeX.")
    return nothing
end

"""
    about(io::IO = stdout)

Print the same report as [`versioninfo`](@ref).
"""
about(io::IO = stdout) = versioninfo(io)

"""
    cite(io::IO = stdout)

Print BibTeX entries for DiSLOUTrajectories.jl and its method paper to `io`.
"""
function cite(io::IO = stdout)
    println(io, _CITATION_BIBTEX)
    return nothing
end
