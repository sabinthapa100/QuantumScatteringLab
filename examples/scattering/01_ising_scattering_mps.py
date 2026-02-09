"""
Experiment 01: 1D Ising Mode Scattering (MPS)
=============================================
Simulates the collision of two wavepackets in the Transverse Field Ising Model (TFIM).
Compares Elastic (g_z=0) vs Inelastic (g_z>0) scattering regimes.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.models.ising_1d import IsingModel1D
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state
import quimb.tensor as qtn
import os

def run_experiment(L=40, T=40, dt=0.1, g_z=0.0, label="elastic"):
    print(f"--- Running {label} scattering (L={L}, g_z={g_z}) ---")
    
    # 1. Setup
    model = IsingModel1D(num_sites=L, g_x=1.2, g_z=g_z, pbc=True)
    backend = QuimbMPSBackend(max_bond_dim=48, verbose=True)
    
    # 2. Vacuum (Approximate |0...0> for strong field)
    # For more precision, we would run DMRG or imaginary time evolution here.
    psi_vac = backend.get_reference_state(L)
    vac_E = [backend.compute_expectation_value(psi_vac, model.get_local_hamiltonian(i)) for i in range(L)]
    
    # 3. Initial State (Wavepackets)
    print("Preparing wavepackets...")
    psi_dense = prepare_two_wavepacket_state(
        L, x1=10, k1=0.5*np.pi, sigma1=1.5,
           x2=30, k2=-0.5*np.pi, sigma2=1.5,
        backend_type="numpy"
    )
    psi = qtn.MatrixProductState.from_dense(psi_dense, [2]*L)
    psi.compress(max_bond=48)
    
    # 4. Evolution
    layers = model.get_trotter_layers()
    heatmap = []
    
    for t in range(int(T/dt)):
        if t % 10 == 0: print(f"Step {t} / {int(T/dt)}")
        
        # Measure Energy Density
        row = []
        for i in range(L):
            val = backend.compute_expectation_value(psi, model.get_local_hamiltonian(i))
            row.append(val - vac_E[i])
        heatmap.append(row)
        
        # Evolve
        psi = backend.evolve_state_trotter(psi, layers, dt)
        
    return np.array(heatmap)

if __name__ == "__main__":
    os.makedirs("results/scattering", exist_ok=True)
    
    # 1. Elastic Scattering
    data_elastic = run_experiment(g_z=0.0, label="Elastic")
    
    # 2. Inelastic Scattering (Particle Production)
    # Turn on longitudinal field to break integrability
    data_inelastic = run_experiment(g_z=0.2, label="Inelastic")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(data_elastic, aspect='auto', origin='lower', cmap='inferno')
    ax1.set_title("Elastic (Integrable)")
    ax1.set_xlabel("Site")
    ax1.set_ylabel("Time")
    
    ax2.imshow(data_inelastic, aspect='auto', origin='lower', cmap='inferno')
    ax2.set_title("Inelastic (Non-Integrable)")
    ax2.set_xlabel("Site")
    
    plt.tight_layout()
    plt.savefig("results/scattering/01_ising_comparison.png")
    print("Saved comparison figure to results/scattering/01_ising_comparison.png")
