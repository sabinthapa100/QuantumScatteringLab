"""
Demo of ADAPT-VQE on SU(2) Gauge Theory.
Compares Qiskit (Exact) and Quimb (MPS) solvers.
"""

from src.models.su2 import SU2GaugeModel
from src.backends.qiskit_backend import QiskitBackend
from src.backends.quimb_backend import QuimbBackend
from src.simulation.adapt_vqe import ADAPTVQESolver
import matplotlib.pyplot as plt


def run_demo(num_sites=4):
    print(f"=== ADAPT-VQE Demo (N={num_sites}) ===")
    
    # 1. Physics Model
    # Use standard coupling from paper
    model = SU2GaugeModel(num_sites=num_sites, g=1.0, a=1.0, pbc=True)
    
    # 2. Solver with Qiskit Backend
    print("\n--- Running with Qiskit Backend (Exact) ---")
    q_backend = QiskitBackend()
    q_solver = ADAPTVQESolver(model, q_backend, tolerance=1e-3, max_iters=10)
    q_result = q_solver.run()
    
    # 3. Solver with Quimb Backend
    print("\n--- Running with Quimb Backend (MPS) ---")
    m_backend = QuimbBackend()
    m_solver = ADAPTVQESolver(model, m_backend, tolerance=1e-3, max_iters=10)
    m_result = m_solver.run()
    
    # 4. Verification
    print("\n--- Summary ---")
    print(f"Qiskit Final Energy: {q_result['energy']:.6f}")
    print(f"Quimb Final Energy:  {m_result['energy']:.6f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(q_result['energy_history'], 'o-', label='Qiskit (Exact)')
    plt.plot(m_result['energy_history'], 's--', label='Quimb (MPS)')
    plt.xlabel('ADAPT Iteration')
    plt.ylabel('Energy')
    plt.title(f'ADAPT-VQE Convergence (SU2 N={num_sites})')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/adapt_vqe_comparison.png')
    print("\nPlot saved to outputs/adapt_vqe_comparison.png")


if __name__ == "__main__":
    run_demo(num_sites=4)
