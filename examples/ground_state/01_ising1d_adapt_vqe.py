"""
1D Ising ADAPT-VQE Analysis
===========================

Comprehensive analysis of the 1D Ising model using ADAPT-VQE.
Comparing convergence, energy accuracy, and state overlap for:
- Different system sizes (N)
- Different coupling strengths (gx)
- Initial state preparation (Reference |0> vs W-state)

Outputs:
- Energy convergence plots
- Fidelity/Overlap plots
- Operator usage statistics
- Comparison with exact diagonalization results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import json
import time

from src.models import IsingModel1D
from src.simulation.adapt_vqe import ADAPTVQESolver
from src.simulation.initialization import prepare_w_state
from src.backends.qiskit_backend import QiskitBackend

# Output directory setup
OUTPUT_DIR = Path("results/adapt_vqe_ising1d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
SYSTEM_SIZES = [4, 6, 8]  # Keep small for rapid testing, can extend to 10
COUPLINGS = [0.5, 1.0, 2.0]  # Paramagnetic, Critical, Ferromagnetic
MAX_ITERS = 20
TOLERANCE = 1e-4  # Convergence tolerance


def run_analysis(N: int, gx: float, initial_state_type: str) -> Dict[str, Any]:
    """Run ADAPT-VQE for a specific configuration."""
    print(f"\n--- Analysis: N={N}, gx={gx}, Init={initial_state_type} ---")
    
    # Model Setup
    model = IsingModel1D(num_sites=N, g_x=gx, g_z=0.0, pbc=True)
    
    # Initial State
    if initial_state_type == 'w_state':
        initial_state = prepare_w_state(N)
    else:
        initial_state = None  # Default |0...0>
        
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
    
    # Run
    start_time = time.time()
    result = solver.run()
    elapsed = time.time() - start_time
    
    result['total_time'] = elapsed
    result['N'] = N
    result['gx'] = gx
    result['initial_state_type'] = initial_state_type
    
    return result

def plot_convergence(results_list: List[Dict], filename_suffix: str):
    """Generate convergence plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for res in results_list:
        label = f"N={res['N']}, gx={res['gx']}, {res['initial_state_type']}"
        
        # Energy Error vs Iteration
        exact_E = res['exact_energy']
        energies = res['energy_history']
        errors = [abs(e - exact_E) + 1e-15 for e in energies] # constant for log plot stability
        
        ax1.semilogy(range(len(errors)), errors, '.-', label=label)
        
        # Gradient Norm vs Iteration
        grads = res['gradient_history']
        ax2.semilogy(range(len(grads)), grads, '.-', label=label)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy Error |E - E_exact|')
    ax1.set_title('Energy Convergence')
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.legend()
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Convergence')
    ax2.grid(True, which="both", ls="-", alpha=0.4)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"convergence_{filename_suffix}.png", dpi=300)
    plt.close()
    print(f"Saved convergence plot: convergence_{filename_suffix}.png")

def main():
    print("="*80)
    print("ADAPT-VQE 1D ISING ANALYSIS STARTED")
    print("="*80)
    
    all_results = []
    
    # 1. Compare Initial States (fixed N=6, gx=1.0)
    print("\n>>> EXPERIMENT 1: Initial State Comparison")
    results_init_comp = []
    for init_type in ['reference', 'w_state']:
        res = run_analysis(N=6, gx=1.0, initial_state_type=init_type)
        results_init_comp.append(res)
        all_results.append(res)
    plot_convergence(results_init_comp, "initial_state_comp")
    
    # 2. System Size Scaling (fixed gx=1.0, w_state)
    print("\n>>> EXPERIMENT 2: System Size Scaling")
    results_size_scaling = []
    for N in SYSTEM_SIZES:
        # Check if already run
        existing = next((r for r in all_results if r['N'] == N and r['gx'] == 1.0 and r['initial_state_type'] == 'w_state'), None)
        if existing:
            results_size_scaling.append(existing)
        else:
            res = run_analysis(N=N, gx=1.0, initial_state_type='w_state')
            results_size_scaling.append(res)
            all_results.append(res)
    plot_convergence(results_size_scaling, "size_scaling")
    
    # 3. Coupling Strength Scan (fixed N=6, w_state)
    print("\n>>> EXPERIMENT 3: Coupling Strength Scan")
    results_coupling_scan = []
    for gx in COUPLINGS:
        # Check if already run
        existing = next((r for r in all_results if r['N'] == 6 and r['gx'] == gx and r['initial_state_type'] == 'w_state'), None)
        if existing:
            results_coupling_scan.append(existing)
        else:
            res = run_analysis(N=6, gx=gx, initial_state_type='w_state')
            results_coupling_scan.append(res)
            all_results.append(res)
    plot_convergence(results_coupling_scan, "coupling_scan")

    # Summary Report
    summary_path = OUTPUT_DIR / "analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("ADAPT-VQE 1D ISING ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        
        for res in all_results:
            f.write(f"Config: N={res['N']}, gx={res['gx']}, Init={res['initial_state_type']}\n")
            f.write(f"  Converged: {res['converged']}\n")
            f.write(f"  Iterations: {res['iterations']}\n")
            f.write(f"  Final Energy Error: {res['energy_error']:.2e}\n")
            f.write(f"  Final Overlap: {res['overlap']:.6f}\n")
            f.write(f"  Time: {res['total_time']:.2f}s\n")
            f.write("-" * 40 + "\n")
            
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
