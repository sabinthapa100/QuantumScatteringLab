"""
Quantum Scattering Lab - Production CLI
========================================
Scalable engine for Ising and Gauge Theory scattering.

Author: Sabin Thapa (Senior Scientist Edition)
"""

import sys
import os
import time
import json
import argparse
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project Path Setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ising_1d import IsingModel1D
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.scattering import ScatteringSimulator
from src.simulation.initialization import prepare_wavepacket_mps
from src.simulation.adapt_vqe import ADAPTVQESolver

# Master Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("production_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def plot_heatmap(data, t_max, L, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, aspect='auto', origin='lower', extent=[0, L, 0, t_max], cmap='magma')
    plt.colorbar(label='Energy Density')
    plt.xlabel('Site')
    plt.ylabel('Time')
    plt.title(f'Spacetime Evolution (L={L})')
    plt.savefig(filename, dpi=150)
    plt.close()

def run_scatter(args):
    """Production Scattering Workflow."""
    logger.info(f"STARTING SCATTERING: L={args.L}, D={args.bond_dim}, GPU={args.gpu}")
    
    backend = QuimbMPSBackend(max_bond_dim=args.bond_dim, use_gpu=args.gpu)
    model = IsingModel1D(num_sites=args.L, g_x=1.25, g_z=0.15, j_int=1.0)
    sim = ScatteringSimulator(model, backend)
    
    # 1. Vacuum
    sim.set_vacuum_reference(method="dmrg")
    
    # 2. Wavepackets (Launch at x=L/4 and 3L/4)
    psi_vac = backend.get_reference_state(args.L) # Dummy to start, set_vacuum_reference handled DMRG
    # Actually, we should use the DMRG state for wavepacket prep if possible
    # For now, prepare_wavepacket_mps handles its own ref if needed, but let's be scientific:
    psi_gs = backend.get_ground_state(model)
    
    k_val = 0.28 * np.pi
    psi_1 = prepare_wavepacket_mps(args.L, x0=args.L//4, k0=k_val, sigma=4.0, reference_state=psi_gs)
    psi_launch = prepare_wavepacket_mps(args.L, x0=3*args.L//4, k0=-k_val, sigma=4.0, reference_state=psi_1)
    
    # 3. Evolution
    results, _ = sim.run(
        initial_state=psi_launch, 
        t_max=60.0, 
        dt=0.1, 
        progress_bar=True,
        return_final_state=True
    )
    
    # 4. Persistence
    output_dir = "data/production/inelastic_collision"
    os.makedirs(output_dir, exist_ok=True)
    
    tensor_data = np.array(results["energy_density"])
    np.save(os.path.join(output_dir, "energy_density.npy"), tensor_data)
    plot_heatmap(tensor_data, 60.0, args.L, os.path.join(output_dir, "heatmap.png"))
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
        
    logger.info(f"Results archived in {output_dir}")

def run_vqe(args):
    """Ground State Preparation Phase."""
    logger.info(f"STARTING VQE: L={args.L}, D={args.bond_dim}")
    model = IsingModel1D(num_sites=args.L, g_x=1.25, g_z=0.15)
    backend = QuimbMPSBackend(max_bond_dim=args.bond_dim, use_gpu=args.gpu)
    solver = ADAPTVQESolver(model=model, backend=backend, tolerance=1e-4)
    results = solver.run()
    logger.info(f"VQE Complete. Final Energy: {results['energy']:.10f}")

def main():
    parser = argparse.ArgumentParser(description="Quantum Scattering Lab: Production CLI")
    parser.add_argument("--mode", type=str, choices=['spectrum', 'vqe', 'scatter'], default='scatter')
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument("--bond-dim", type=int, default=128)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    
    print("==================================================")
    print("     QUANTUM SCATTERING LAB: PRODUCTION CLI       ")
    print("==================================================")
    
    if args.mode == 'scatter':
        run_scatter(args)
    elif args.mode == 'vqe':
        run_vqe(args)
    elif args.mode == 'spectrum':
        print("Spectrum analysis script point. (See src/analysis/spectrum.py)")

if __name__ == "__main__":
    main()
