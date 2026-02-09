"""
2D Ising ADAPT-VQE Analysis
===========================

Comprehensive analysis of the 2D Ising model using ADAPT-VQE.
Validating convergence on square lattices.

Lattices: 2x2 (4 qubits), 2x3 (6 qubits)
Couplings (gx): 2.0, 3.0, 4.0 (around critical point g_x ~ 3.044)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import time

from src.models.ising_2d import IsingModel2D
from src.simulation.adapt_vqe import ADAPTVQESolver
from src.simulation.initialization import prepare_w_state
from src.backends.qiskit_backend import QiskitBackend

# Output directory setup
OUTPUT_DIR = Path("results/adapt_vqe_ising2d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
LATTICES = [(2, 2), (2, 3)]
COUPLINGS = [2.0, 3.044, 4.0] 
MAX_ITERS = 30
TOLERANCE = 1e-4

def run_analysis(Lx: int, Ly: int, gx: float) -> Dict[str, Any]:
    """Run ADAPT-VQE for a specific 2D configuration."""
    print(f"\n--- Analysis: {Lx}x{Ly}, gx={gx} ---")
    
    # Model Setup
    model = IsingModel2D(Lx=Lx, Ly=Ly, g_x=gx, g_z=0.0, pbc=True)
    
    # We'll use W-state as it was successful in 1D
    # For 2D, we just flatten the indices or use the site list
    num_sites = Lx * Ly
    initial_state = prepare_w_state(num_sites)
    
    # Backend
    backend = QiskitBackend()
    
    # Solver
    solver = ADAPTVQESolver(
        model=model,
        backend=backend,
        tolerance=TOLERANCE,
        max_iters=MAX_ITERS,
        convergence_type='norm',
        initial_state=initial_state,
        pool_type='global'
    )
    
    # Run - This will use the enhanced logging we implemented
    start_time = time.time()
    result = solver.run()
    elapsed = time.time() - start_time
    
    result['total_time'] = elapsed
    result['Lx'] = Lx
    result['Ly'] = Ly
    result['gx'] = gx
    
    return result

def plot_convergence(results_list: List[Dict]):
    """Generate convergence plots for 2D."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for res in results_list:
        label = f"{res['Lx']}x{res['Ly']}, gx={res['gx']}"
        
        # Energy Error
        exact_E = res['exact_energy']
        energies = res['energy_history']
        errors = [abs(e - exact_E) + 1e-15 for e in energies]
        
        ax1.semilogy(range(len(errors)), errors, '.-', label=label)
        
        # Gradient Norm
        grads = res['gradient_history']
        ax2.semilogy(range(len(grads)), grads, '.-', label=label)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy Error |E - E_exact|')
    ax1.set_title('2D Ising Energy Convergence')
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.legend()
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('2D Ising Gradient Convergence')
    ax2.grid(True, which="both", ls="-", alpha=0.4)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ising2d_convergence.png", dpi=300)
    plt.close()

def main():
    print("="*80)
    print("ADAPT-VQE 2D ISING ANALYSIS STARTED")
    print("="*80)
    
    results = []
    for Lx, Ly in LATTICES:
        for gx in COUPLINGS:
            res = run_analysis(Lx, Ly, gx)
            results.append(res)
    
    plot_convergence(results)
    
    # Summary
    summary_path = OUTPUT_DIR / "ising2d_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ADAPT-VQE 2D ISING SUMMARY\n")
        f.write("="*40 + "\n\n")
        for res in results:
            f.write(f"Config: {res['Lx']}x{res['Ly']}, gx={res['gx']}\n")
            f.write(f"  Converged: {res['converged']}\n")
            f.write(f"  Iterations: {res['iterations']}\n")
            f.write(f"  Error: {res['energy_error']:.2e}\n")
            f.write(f"  Overlap: {res['overlap']:.6f}\n")
            f.write("-" * 40 + "\n")

if __name__ == "__main__":
    main()
