"""
Scattering of Gauge Excitations (Glueballs) in SU(2) Lattice Gauge Theory.
Using MPS backend for 1D plaquette chain.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Ensure project root is in path
sys.path.append(os.path.abspath("."))

from src.models.su2 import SU2GaugeModel
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state
import quimb.tensor as qtn

def run_su2_scattering(num_sites=16, g=2.0, num_steps=80, dt=0.1):
    print(f"\n--- SU(2) Gauge Scattering: L={num_sites}, g={g} ---")
    # For large g, vacuum is highly polarized.
    model = SU2GaugeModel(num_sites=num_sites, g=g, pbc=True)
    backend = QuimbMPSBackend(max_bond_dim=32)
    
    # 1. Vacuum
    # Computational '0' is the strong coupling vacuum for SU(2).
    psi_vac = backend.get_reference_state(num_sites)
    print("Measuring vacuum energy density...")
    vac_energy_density = []
    for n in range(num_sites):
        En_op = model.get_local_hamiltonian(n)
        val = backend.compute_expectation_value(psi_vac, En_op)
        vac_energy_density.append(val)
    vac_energy_density = np.array(vac_energy_density)
    
    # 2. Initial State
    print("Preparing initial state (Two Glueball Wavepackets)...")
    # Wavepackets of electric flux excitations.
    x1, x2 = num_sites / 4, 3 * num_sites / 4
    psi_np = prepare_two_wavepacket_state(
        num_sites,
        x1=x1, k1=0.4*np.pi, sigma1=1.0,
        x2=x2, k2=-0.4*np.pi, sigma2=1.0,
        backend_type="numpy"
    )
    current_psi = qtn.MatrixProductState.from_dense(psi_np, [2]*num_sites)
    current_psi.compress(max_bond=32)
    
    # 3. Evolution
    heatmap_data = np.zeros((num_steps, num_sites))
    layers = model.get_trotter_layers()
    
    print(f"Evolving for {num_steps} steps...")
    start_time = time.time()
    for i in range(num_steps):
        if i % 10 == 0:
            print(f"  Step {i}/{num_steps} (Time: {time.time()-start_time:.1f}s)")
            
        # Measure Energy Density
        for n in range(num_sites):
            En_op = model.get_local_hamiltonian(n)
            val = backend.compute_expectation_value(current_psi, En_op)
            heatmap_data[i, n] = val - vac_energy_density[n]
            
        current_psi = backend.evolve_state_trotter(current_psi, layers, dt)
        
    return heatmap_data

def plot_su2_heatmap(data, filename, num_sites, num_steps, dt):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, extent=[0, num_sites-1, num_steps*dt, 0], aspect='auto', cmap='inferno')
    plt.colorbar(label='Glueball Energy Density (En - Evac)')
    plt.xlabel('Plaquette Index n')
    plt.ylabel('Time t')
    plt.title(f'SU(2) Glueball Scattering (L={num_sites}, g=2.0)')
    
    output_dir = "results/scattering"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    print(f"Saved SU(2) plot to {output_dir}/{filename}")

def main():
    h_su2 = run_su2_scattering(num_sites=16, g=2.0, num_steps=60)
    plot_su2_heatmap(h_su2, "su2_scattering.png", 16, 60, 0.1)

if __name__ == "__main__":
    main()
