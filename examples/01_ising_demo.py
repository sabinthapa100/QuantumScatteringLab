import sys
import os
# Add src to path
sys.path.append(os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt
from src.models.ising_1d import IsingModel1D
from src.simulation.adapt_vqe import ADAPTVQESolver
from src.simulation.exact import ExactSolver
from src.backends.qiskit_backend import QiskitBackend

def main():
    print("========================================")
    print("      ADAPT-VQE Demo: 1D Ising Model    ")
    print("      (Using QiskitBackend)             ")
    print("========================================")

    # 1. Define Model
    num_sites = 4
    model = IsingModel1D(num_sites=num_sites, j_int=1.0, g_x=1.5, pbc=True)
    print(f"Model: N={num_sites} Ising Chain (J=1.0, g=1.5)")

    # 2. Define Backend
    backend = QiskitBackend()
    print("Backend: QiskitBackend (Statevector)")

    # 3. Ground Truth (Exact Diagonalization)
    print("\n[Running Exact Solver...]")
    exact_solver = ExactSolver(model)
    exact_result = exact_solver.solve()
    E_exact = exact_result['energy']
    print(f"Exact Ground State Energy: {E_exact:.6f}")

    # 4. Run ADAPT-VQE Machine
    print("\n[Running ADAPT-VQE Machine...]")
    # Now passing backend explicitly
    solver = ADAPTVQESolver(model, backend=backend, tolerance=1e-5, max_iters=20)
    result = solver.run()

    E_vqe = result['energy']
    print("\n----------------------------------------")
    print(f"Final VQE Energy: {E_vqe:.6f}")
    print(f"Exact Energy:     {E_exact:.6f}")
    print(f"Error:            {abs(E_vqe - E_exact):.2e}")
    print("----------------------------------------")
    print("Ansatz Operators Selected:")
    for i, op in enumerate(result['ansatz_ops']):
        print(f"  {i+1}. {op}")

    # 5. Visualization
    try:
        plt.figure(figsize=(10, 6))
        energies = result['energy_history']
        plt.plot(energies, 'o-', label='ADAPT-VQE')
        plt.axhline(E_exact, color='r', linestyle='--', label='Exact')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title(f'ADAPT-VQE Convergence (N={num_sites})')
        plt.legend()
        plt.grid(True)
        plt.savefig('examples/01_ising_convergence.png')
        print("\nConvergence plot saved to 'examples/01_ising_convergence.png'")
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    main()
