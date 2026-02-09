"""Quick SU(2) test"""
import sys
print("Starting SU(2) analysis...", flush=True)

from src.models import SU2GaugeModel
print("Imported SU2GaugeModel", flush=True)

model = SU2GaugeModel(num_sites=4, g=1.0, a=1.0, pbc=True)
print(f"Created model: {model}", flush=True)

H = model.build_hamiltonian()
print(f"Built Hamiltonian: {H.num_qubits} qubits", flush=True)

import numpy as np
mat = H.to_matrix()
print(f"Matrix shape: {mat.shape}", flush=True)

eigs = np.linalg.eigvalsh(mat)
print(f"Ground state energy: {eigs[0]:.4f}", flush=True)
print(f"Gap: {eigs[1] - eigs[0]:.4f}", flush=True)

print("✓ SU(2) analysis test complete!", flush=True)
