"""
Test Quimb Backend for ADAPT-VQE
================================

Validates the new QuimbBackend using the known reliable 1D Ising case.
"""

from src.models.ising_1d import IsingModel1D
from src.backends.quimb_backend import QuimbBackend
from src.simulation.adapt_vqe import ADAPTVQESolver
import numpy as np

def test_quimb_backend():
    print("Initializing QuimbBackend...")
    # Try with GPU if available, else CPU
    backend = QuimbBackend(use_gpu=True, verbose=True)
    if backend.use_gpu:
        print("-> Using GPU (cupy)")
    else:
        print("-> Using CPU (numpy/scipy)")
        
    print("\n--- Testing 1D Ising N=4, gx=1.0 ---")
    model = IsingModel1D(num_sites=4, g_x=1.0, g_z=1.0)
    
    # Run ADAPT-VQE
    solver = ADAPTVQESolver(model, backend=backend, tolerance=1e-3, max_iters=10)
    
    # Initialize with W-state to match previous validation (Quimb needs vector)
    # W-state for N=4: |0001> + |0010> + ...
    # Quimb basis order might differ from Qiskit.
    # Qiskit: q3 q2 q1 q0. |0001> means q0=1.
    # Quimb: usually left-to-right 0 1 2 3.
    # Let's use reference state |0000> first to minimize basis confusion issues.
    # If 1D Ising works with |0...0>, then great.
    # Actually, previous 1D runs worked fine with W-state.
    # Let's stick to default initialization (reference state) for simplicity in backend test.
    solver.initial_state = None 
    
    print("Running solver...")
    results = solver.run()
    
    # DEBUG: Reconstruct state and check manually
    print("\n--- DEBUG: Manual State Check ---")
    final_params = solver.parameters
    psi = solver._prepare_ansatz_state(final_params)
    psi_vec = backend.get_statevector(psi)
    
    # compute exact GS
    H_mat = model.build_hamiltonian().to_matrix()
    eigs, vecs = np.linalg.eigh(H_mat)
    gs = vecs[:, 0]
    
    overlap_calc = np.abs(np.vdot(psi_vec, gs))**2
    print(f"Manual Overlap Calc: {overlap_calc:.6f}")
    
    # Check norms
    print(f"Psi norm: {np.linalg.norm(psi_vec)}")
    print(f"GS norm: {np.linalg.norm(gs)}")
    
    # Check first few elements
    print(f"Psi[:4]: {psi_vec[:4]}")
    print(f"GS[:4]:  {gs[:4]}")
    
    # Check results
    overlap = results.get('overlap', 0.0)
    energy_err = results.get('energy_error', 1.0)
    
    print(f"\nFinal Overlap: {overlap:.6f}")
    print(f"Energy Error: {energy_err:.2e}")
    
    if overlap > 0.99 and energy_err < 1e-6:
        print("✅ QuimbBackend VALIDATED")
    else:
        print("❌ QuimbBackend FAILED Validation")

if __name__ == "__main__":
    test_quimb_backend()
