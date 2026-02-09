import time
from qiskit.quantum_info import SparsePauliOp
import numpy as np

N = 12
terms = []
coeffs = []
for i in range(N):
    p = ["I"] * N; p[i] = "X"
    terms.append("".join(p))
    coeffs.append(1.0)

op = SparsePauliOp.from_list(list(zip(terms, coeffs)))

start = time.time()
print(f"Starting to_matrix for N={N}...")
mat = op.to_matrix(sparse=True)
print(f"Done in {time.time() - start:.4f}s")
print(f"Matrix shape: {mat.shape}")
