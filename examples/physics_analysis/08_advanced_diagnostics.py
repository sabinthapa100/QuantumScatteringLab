"""
Experiment 03: Advanced Diagnostics (Entanglement Entropy)
==========================================================
Simulates scattering and measures the Von Neumann entanglement entropy growth.
Verifies Area Law vs Volume Law behavior during collision.
S_vn = -Tr(rho_A log rho_A)
"""

import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn
from src.models.ising_1d import IsingModel1D
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state

def run_entropy_analysis(L=30, T=30, dt=0.1):
    print(f"--- Running Entanglement Entropy Analysis (L={L}) ---")
    
    # Setup
    model = IsingModel1D(num_sites=L, g_x=1.5, g_z=0.1) # Inelastic
    backend = QuimbMPSBackend(max_bond_dim=64)
    
    # Initial State (Wavepackets)
    psi_dense = prepare_two_wavepacket_state(
        L, 
        x1=8, k1=0.6*np.pi, sigma1=1.5,
        x2=22, k2=-0.6*np.pi, sigma2=1.5,
        backend_type="numpy"
    )
    psi = qtn.MatrixProductState.from_dense(psi_dense, [2]*L)
    psi.compress(max_bond=64)
    
    layers = model.get_trotter_layers()
    
    # Track entropy at the center cut (L/2)
    # and profile across all cuts
    entropy_evolution = []
    
    steps = int(T/dt)
    for t in range(steps):
        if t % 5 == 0: print(f"Step {t}/{steps}")
        
        # Measure Entropy across all bonds
        # S(x) profile
        profile = []
        for i in range(L-1):
            # Bond between i and i+1
            S = backend.entanglement_entropy(psi, i)
            profile.append(S)
        entropy_evolution.append(profile)
        
        psi = backend.evolve_state_trotter(psi, layers, dt)
        
    return np.array(entropy_evolution)

if __name__ == "__main__":
    import os
    os.makedirs("results/diagnostics", exist_ok=True)
    
    S_data = run_entropy_analysis()
    
    # 1. 2D Map of Entropy
    plt.figure(figsize=(10, 6))
    plt.imshow(S_data, aspect='auto', origin='lower', cmap='magma')
    plt.title("Entanglement Entropy Evolution S(x, t)")
    plt.xlabel("Bond Index (x)")
    plt.ylabel("Time Step")
    plt.colorbar(label="Von Neumann Entropy")
    plt.savefig("results/diagnostics/entropy_map.png")
    
    # 2. Central Cut Evolution
    plt.figure(figsize=(8, 4))
    L = S_data.shape[1]
    center = L // 2
    plt.plot(S_data[:, center], label=f"Center Cut (x={center})", color='cyan')
    plt.title(f"Entanglement Growth (Bond Dimension={64})")
    plt.ylabel("S_vn")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("results/diagnostics/entropy_growth.png")
    
    print("Figures saved to results/diagnostics/")
