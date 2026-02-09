"""
SU(2) Gauge Theory ADAPT-VQE Analysis
=====================================

Analysis of convergence for SU(2) gauge theory using ADAPT-VQE.
Focusing on small systems to validate the gauge theory specific pool.

Chain Length (N): 4, 6
Couplings (g): 0.5, 1.0, 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import time

from src.models.su2 import SU2GaugeModel
from src.simulation.adapt_vqe import ADAPTVQESolver
from src.simulation.initialization import prepare_w_state
from src.backends.qiskit_backend import QiskitBackend

# Output directory setup
OUTPUT_DIR = Path("results/adapt_vqe_su2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SIZES = [4, 6]
COUPLINGS = [0.5, 1.0, 2.0]
MAX_ITERS = 40
TOLERANCE = 1e-4

def run_analysis(num_sites: int, g: float) -> Dict[str, Any]:
    """Run ADAPT-VQE for a specific SU2 configuration."""
    print(f"\n--- Analysis: SU2 N={num_sites}, g={g} ---")
    
    # Model Setup
    model = SU2GaugeModel(num_sites=num_sites, g=g, a=1.0, pbc=True)
    
    # Use W-state for initialization
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
    
    start_time = time.time()
    result = solver.run()
    elapsed = time.time() - start_time
    
    result['total_time'] = elapsed
    result['N'] = num_sites
    result['g'] = g
    
    return result

def plot_convergence(results_list: List[Dict]):
    """Generate convergence plots for SU2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for res in results_list:
        label = f"N={res['N']}, g={res['g']}"
        
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
    ax1.set_title('SU2 Gauge Theory Energy Convergence')
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.legend()
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('SU2 Gauge Theory Gradient Convergence')
    ax2.grid(True, which="both", ls="-", alpha=0.4)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "su2_convergence.png", dpi=300)
    plt.close()

def main():
    print("="*80)
    print("ADAPT-VQE SU(2) GAUGE THEORY ANALYSIS STARTED")
    print("="*80)
    
    results = []
    for N in SIZES:
        for g in COUPLINGS:
            res = run_analysis(N, g)
            results.append(res)
    
    plot_convergence(results)
    
    # Summary
    summary_path = OUTPUT_DIR / "su2_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ADAPT-VQE SU(2) SUMMARY\n")
        f.write("="*40 + "\n\n")
        for res in results:
            f.write(f"Config: N={res['N']}, g={res['g']}\n")
            f.write(f"  Converged: {res['converged']}\n")
            f.write(f"  Iterations: {res['iterations']}\n")
            f.write(f"  Error: {res['energy_error']:.2e}\n")
            f.write(f"  Overlap: {res['overlap']:.6f}\n")
            f.write("-" * 40 + "\n")

if __name__ == "__main__":
    main()
