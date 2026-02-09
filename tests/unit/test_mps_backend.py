import numpy as np
from src.backends.quimb_backend import QuimbBackend
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.models.ising_1d import IsingModel1D
from qiskit.quantum_info import SparsePauliOp

def test_mps_vs_dense():
    num_sites = 8
    model = IsingModel1D(num_sites=num_sites, g_x=1.25, g_z=0.15)
    ham = model.build_hamiltonian()
    
    dense_backend = QuimbBackend()
    mps_backend = QuimbMPSBackend(max_bond_dim=10)
    
    # 1. State Preparation
    psi_dense = dense_backend.get_reference_state(num_sites)
    psi_mps = mps_backend.get_reference_state(num_sites)
    
    # 2. Expectation Value
    e_dense = dense_backend.compute_expectation_value(psi_dense, ham)
    e_mps = mps_backend.compute_expectation_value(psi_mps, ham)
    
    print(f"Energy (Reference State): Dense={e_dense:.6f}, MPS={e_mps:.6f}")
    assert np.isclose(e_dense, e_mps)
    
    # 3. Evolution
    op = model.build_operator_pool()[0] # O1 = sum Y
    theta = 0.5
    
    psi_dense_evolved = dense_backend.apply_operator(psi_dense, op, theta)
    psi_mps_evolved = mps_backend.apply_operator(psi_mps, op, theta)
    
    e_dense_evol = dense_backend.compute_expectation_value(psi_dense_evolved, ham)
    e_mps_evol = mps_backend.compute_expectation_value(psi_mps_evolved, ham)
    
    print(f"Energy (Evolved): Dense={e_dense_evol:.6f}, MPS={e_mps_evol:.6f}")
    assert np.isclose(e_dense_evol, e_mps_evol)
    
    print("Overlap check...")
    # Convert MPS to dense vector for overlap
    vec_mps = psi_mps_evolved.to_dense()
    overlap = np.abs(np.vdot(psi_dense_evolved, vec_mps))**2
    print(f"Overlap: {overlap:.6f}")
    assert np.isclose(overlap, 1.0, atol=1e-5)

if __name__ == "__main__":
    test_mps_vs_dense()
