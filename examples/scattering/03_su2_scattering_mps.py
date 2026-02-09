"""
Experiment 02b: SU(2) Lattice Gauge Theory Scattering (MPS)
==========================================================
Simulates "glueball" scattering on a chain of SU(2) plaquettes.
The model is mapped to a spin chain where "plaquette flips" correspond to local magnetic excitations.
"""

import numpy as np
import matplotlib.pyplot as plt
import quimb.tensor as qtn
import os
from src.models.su2 import SU2GaugeModel
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state

def run_experiment(L=20, T=30, dt=0.05, g=1.5):
    print(f"--- Running SU(2) Gauge Theory Scattering (L={L}, g={g}) ---")
    
    # 1. Setup Model (Plaquette Chain)
    # g is coupling constant. Large g -> Strong Coupling (Vacuum ~ |0>)
    model = SU2GaugeModel(num_sites=L, g=g, a=1.0, pbc=True)
    backend = QuimbMPSBackend(max_bond_dim=64, verbose=True)
    
    # 2. Vacuum State
    # In strong coupling, vacuum is close to flux-free |0...0>
    # In weak coupling, we'd need VQE or imaginary time evolution.
    # Here we assume strong coupling limit for clean wavepackets.
    psi_vac = backend.get_reference_state(L)
    vac_E = [backend.compute_expectation_value(psi_vac, model.get_local_hamiltonian(i)) for i in range(L)]

    # 3. Initial State (Meson/Glueball Wavepackets)
    # Excitations are localized plaquette flips (magnetic flux)
    print("Injecting glueballs...")
    psi_dense = prepare_two_wavepacket_state(
        L, 
        x1=5, k1=0.8*np.pi, sigma1=1.0,  # Left glueball
        x2=15, k2=-0.8*np.pi, sigma2=1.0, # Right glueball
        backend_type="numpy"
    )
    psi = qtn.MatrixProductState.from_dense(psi_dense, [2]*L)
    psi.compress(max_bond=64)
    
    # 4. Evolution
    layers = model.get_trotter_layers()
    heatmap = []
    
    steps = int(T/dt)
    for t in range(steps):
        if t % 5 == 0: print(f"Step {t}/{steps}")
        
        # Measure local Electric/Magnetic energy density
        # For simplicity, just total H density
        row = []
        for i in range(L):
            val = backend.compute_expectation_value(psi, model.get_local_hamiltonian(i))
            row.append(val - vac_E[i])
        heatmap.append(row)
        
        psi = backend.evolve_state_trotter(psi, layers, dt)
        
    return np.array(heatmap)

if __name__ == "__main__":
    os.makedirs("results/su2", exist_ok=True)
    
    # Run simulation
    data = run_experiment(g=1.2) # Intermediate coupling
    
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.imshow(data, aspect='auto', origin='lower', cmap='plasma')
    plt.title("SU(2) Glueball Scattering Density")
    plt.xlabel("Plaquette Index")
    plt.ylabel("Time Step")
    plt.colorbar(label="Energy Density")
    
    plt.savefig("results/su2/scattering_map.png")
    print("Saved to results/su2/scattering_map.png")
