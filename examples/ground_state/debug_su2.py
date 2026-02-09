"""
Debug SU(2) Convergence
=======================

Quick script to debug why SU(2) N=4, g=1.0 gets stuck at 71% overlap with W-state.
Try:
1. Reference state |0000>
2. W-state |0001> + ...
3. Random state (if possible)
4. Check energy landscape vs g (strong vs weak coupling)
"""

import numpy as np
from qiskit.quantum_info import Statevector
from src.models.su2 import SU2GaugeModel
from src.simulation.adapt_vqe import ADAPTVQESolver

def run_test(initial_state_name='reference', g=1.0):
    print(f"\n--- Testing SU(2) N=4, g={g} with {initial_state_name} initialization ---")
    
    model = SU2GaugeModel(num_sites=4, g=g, a=1.0) # 2 sites = 4 qubits? No, num_sites=4 is 4 sites.
    # Wait, SU(2) num_sites means lattice sites.
    # N=4 sites means 4 physical sites.
    # Mapped to qubits? One qubit per link? Or per site?
    # Usually SU(2) lattice gauge theory with truncation uses multiple qubits per link.
    # But let's assume num_sites=4 corresponds to system size.
    
    print(f"Couplings: {model.coupling_constants}")
    
    solver = ADAPTVQESolver(model, tolerance=1e-3, max_iters=20)
    
    # Prepare initial state
    if initial_state_name == 'w_state':
        # Create W-state for N qubits
        N = model.num_sites
        w_vec = np.zeros(2**N)
        for i in range(N):
            s = ['0']*N
            s[i] = '1'
            idx = int("".join(s), 2)
            w_vec[idx] = 1.0
        w_vec /= np.linalg.norm(w_vec)
        solver.initial_state = Statevector(w_vec)
    elif initial_state_name == 'reference':
        solver.initial_state = None # Uses |0...0>
        
    results = solver.run()
    return results

if __name__ == "__main__":
    # Test 1: Reference State (All Z+?)
    run_test('reference', g=1.0)
    
    # Test 2: W-State (reproduce failure)
    run_test('w_state', g=1.0)
    
    # Test 3: Strong coupling (g=10.0) -> effectively Ising ferromagnet
    run_test('reference', g=10.0)
