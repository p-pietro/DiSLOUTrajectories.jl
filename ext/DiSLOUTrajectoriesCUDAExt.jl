module DiSLOUTrajectoriesCUDAExt

using LinearAlgebra
import CUDACore
import cuSOLVER  # provides eigen/lu on CuMatrix and loads cuBLAS for `*`
import DiSLOUTrajectories

# Diagonalize H_eff on the GPU; the eigensystem is returned to the CPU.
function DiSLOUTrajectories._cuda_eigen(Heff::Matrix{ComplexF64})
    F = eigen(CUDACore.CuArray(Heff))
    CUDACore.synchronize()
    return Array(F.values), Array(F.vectors)
end

function __init__()
    CUDACore.functional() && DiSLOUTrajectories._enable_cuda_diagonalization!()
    return nothing
end

end
