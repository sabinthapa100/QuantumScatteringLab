"""
Unified ADAPT-VQE Multi-Model Validation
========================================

Small-scale validation of ADAPT-VQE across three different models:
1. 1D Ising Model (N=4)
2. 2D Ising Model (2x2 square lattice, N=4)
3. SU(2) Gauge Theory (N=4 sites)

This script demonstrates convergence and provides real-time logging of:
- Energy levels
- Gradient norms
- Wavefunction overlap with exact ground state
- Optimization timing
"""

import numpy as np
from pathlib import Path
import time

from src.models.ising_1d import IsingModel1D
from src.models.ising_2d import IsingModel2D
from src.models.su2 import SU2GaugeModel
from src.simulation.adapt_vqe import ADAPTVQESolver
from src.simulation.initialization import prepare_w_state
from src.backends.qiskit_backend import QiskitBackend

# Output directory
OUTPUT_DIR = Path("results/unified_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_experiment(model, name, initial_state=None):
    print(f"\n" + "#"*80)
    print(f" EXPERIMENT: {name}")
    print("#"*80)
    
    backend = QiskitBackend()
    
    # Solver setup
    solver = ADAPTVQESolver(
        model=model,
        backend=backend,
        tolerance=1e-3,
        max_iters=10,
        convergence_type='max',
        initial_state=initial_state
    )
    
    results = solver.run()
    return results

def main():
    print("="*80)
    print("UNIFIED ADAPT-VQE VALIDATION")
    print("="*80)
    
    # 1. 1D Ising Validation
    ising1d = IsingModel1D(num_sites=4, g_x=1.0, pbc=True)
    w_state_1d = prepare_w_state(4)
    run_experiment(ising1d, "1D Ising (N=4, gx=1.0, W-state init)", initial_state=w_state_1d)
    
    # 2. 2D Ising Validation
    ising2d = IsingModel2D(Lx=2, Ly=2, g_x=3.044, pbc=True) # Near critical point
    w_state_2d = prepare_w_state(4)
    run_experiment(ising2d, "2D Ising (2x2, gx=3.044, W-state init)", initial_state=w_state_2d)
    
    # 3. SU(2) Gauge Theory Validation
    su2 = SU2GaugeModel(num_sites=4, g=1.0, a=1.0, pbc=True)
    w_state_su2 = prepare_w_state(4)
    run_experiment(su2, "SU(2) Gauge Theory (N=4, g=1.0, W-state init)", initial_state=w_state_su2)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()
