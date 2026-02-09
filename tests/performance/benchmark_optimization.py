
import time
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from qiskit.quantum_info import SparsePauliOp

def benchmark():
    N = 8
    dim = 2**N
    print(f"Benchmarking for N={N} (dim={dim})")
    
    op_list = []
    for i in range(N):
        p = ["I"] * N
        p[i] = "Y"
        op_list.append("".join(reversed(p)))
    
    op = SparsePauliOp(op_list)
    mat = op.to_matrix()
    vec = np.random.rand(dim) + 1j * np.random.rand(dim)
    vec /= np.linalg.norm(vec)

    theta = 0.5
    exponent = 1j * theta * mat
    
    # 1. expm (dense)
    t0 = time.time()
    U = scipy.linalg.expm(exponent)
    res_expm = np.dot(U, vec)
    t1 = time.time()
    print(f"expm (dense) time: {t1-t0:.6f} s")
    
    # 2. expm_multiply (dense)
    t0 = time.time()
    res_expm_mul = scipy.sparse.linalg.expm_multiply(exponent, vec)
    t1 = time.time()
    print(f"expm_multiply (dense) time: {t1-t0:.6f} s")

    # 3. expm_multiply (sparse)
    sparse_exponent = scipy.sparse.csr_matrix(exponent)
    t0 = time.time()
    res_expm_mul_sparse = scipy.sparse.linalg.expm_multiply(sparse_exponent, vec)
    t1 = time.time()
    print(f"expm_multiply (sparse) time: {t1-t0:.6f} s")
    
    # 5. Diagonalization
    t0 = time.time()
    # Pre-diagonalize once
    vals, vecs = np.linalg.eigh(mat)
    t1 = time.time()
    print(f"One-time diagonalization time: {t1-t0:.6f} s")
    
    t0 = time.time()
    # Apply using eigh
    # U psi = V exp(i theta D) V* psi
    v1 = np.dot(vecs.conj().T, vec)
    v2 = np.exp(1j * theta * vals) * v1
    res_eigh = np.dot(vecs, v2)
    t1 = time.time()
    # 6. Analytic (Split)
    t0 = time.time()
    current_vec = vec.copy()
    for i in range(N):
        # Apply term i
        c = np.cos(theta)
        s = np.sin(theta)
        # Using mat[i] would be better, but we simulate overhead
        # In reality we'd fetch the i-th term's matrix.
        # Let's just do one mat-vec for timing
        current_vec = c * current_vec + 1j * s * (mat @ current_vec)
    t1 = time.time()
    print(f"Analytic Split (N terms) time: {t1-t0:.6f} s")

if __name__ == "__main__":
    benchmark()
