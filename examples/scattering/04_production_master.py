"""
Quantum Scattering Lab: Master Production Suite
================================================
Perform high-resolution simulations for the final report.
Includes:
- Elastic Scattering (Integrable TFIM)
- Inelastic Scattering (Non-Integrable TFIM)
- Entropy Scrambling Analysis
- Particle Production Metrics
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# High-Res Constants
L = 60
T_MAX = 15.0
DT = 0.2
MAX_BOND = 64

# Import internal modules
sys.path.insert(0, os.path.abspath("."))
from src.models.ising_1d import IsingModel1D
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state
import quimb.tensor as qtn

def run_production(g_z=0.0, label="elastic"):
    print(f"\n🚀 Running Production Run: {label.upper()} (L={L}, gz={g_z})")
    
    # 1. Initialization
    model = IsingModel1D(num_sites=L, g_x=1.2, g_z=g_z, pbc=True)
    backend = QuimbMPSBackend(max_bond_dim=MAX_BOND)
    
    # Accurate Vacuum via DMRG
    print("Preparing accurate vacuum via DMRG...")
    psi_vac = backend.get_ground_state(model)
    vac_E = [backend.compute_expectation_value(psi_vac, model.get_local_hamiltonian(i)) for i in range(L)]
    
    # Scalable Wavepackets via Direct MPS Construction
    print("Initializing wavepackets (MPS)...")
    psi = prepare_two_wavepacket_state(
        L, x1=L/4, k1=0.5*np.pi, sigma1=2.0,
           x2=3*L/4, k2=-0.5*np.pi, sigma2=2.0,
        backend_type="mps",
        reference_state=psi_vac # Pass vacuum to ensure it sits on top of the correct sector
    )
    psi.compress(max_bond=MAX_BOND)
    
    # Evolution
    layers = model.get_trotter_layers()
    energy_heatmap = []
    entropy_profiles = []
    particle_counts = []
    
    num_steps = int(T_MAX / DT)
    
    for t in tqdm(range(num_steps), desc=f"Evolving {label}"):
        # 1. Measurement: Energy Density
        row_e = []
        for i in range(L):
            val = backend.compute_expectation_value(psi, model.get_local_hamiltonian(i))
            row_e.append(val - vac_E[i])
        energy_heatmap.append(row_e)
        
        # 2. Measurement: Entropy Profile
        row_s = [backend.entanglement_entropy(psi, i) for i in range(L-1)]
        entropy_profiles.append(row_s)
        
        # 3. Measurement: Particle count (Integrated excitation)
        # For simplicity, we use integrated positive energy density as a proxy for particle number
        particle_counts.append(np.sum(np.clip(row_e, 0, None)))
        
        # 4. Step
        psi = backend.evolve_state_trotter(psi, layers, DT)
        
    return {
        "energy": np.array(energy_heatmap),
        "entropy": np.array(entropy_profiles),
        "particles": np.array(particle_counts)
    }

def plot_master_results(results_list, labels):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(len(results_list), 2, figsize=(18, 10 * len(results_list)))
    
    for idx, (res, label) in enumerate(zip(results_list, labels)):
        # Energy Heatmap
        ax_e = axes[idx][0]
        im_e = ax_e.imshow(res["energy"], aspect='auto', origin='lower', extent=[0, L, 0, T_MAX], cmap='inferno', interpolation='gaussian')
        ax_e.set_title(f"Energy Density Evolution ({label.capitalize()})", fontsize=16, color='cyan')
        ax_e.set_ylabel("Time (t)", fontsize=14)
        ax_e.set_xlabel("Lattice Site (n)", fontsize=14)
        plt.colorbar(im_e, ax=ax_e, label="E - E_vac")
        
        # Entropy Heatmap
        ax_s = axes[idx][1]
        im_s = ax_s.imshow(res["entropy"], aspect='auto', origin='lower', extent=[0, L-1, 0, T_MAX], cmap='viridis', interpolation='gaussian')
        ax_s.set_title(f"Entanglement Entropy Growth ({label.capitalize()})", fontsize=16, color='magenta')
        ax_s.set_ylabel("Time (t)", fontsize=14)
        ax_s.set_xlabel("Bond Index", fontsize=14)
        plt.colorbar(im_s, ax=ax_s, label="S_vn")

    plt.tight_layout()
    os.makedirs("results/final_production", exist_ok=True)
    plt.savefig("results/final_production/master_scattering_analysis.png", dpi=300)
    print("\n✅ Master Production Figure saved to results/final_production/master_scattering_analysis.png")

if __name__ == "__main__":
    el = run_production(g_z=0.0, label="elastic")
    inel = run_production(g_z=0.2, label="inelastic")
    
    plot_master_results([el, inel], ["elastic (g_z=0)", "inelastic (g_z=0.2)"])
