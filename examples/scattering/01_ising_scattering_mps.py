"""
Scattering in 1D Ising Model using MPS Backend.
Reproducing Farrell et al. (2025) Figure 2 and Figure 4.
Cases:
1. Elastic (gz = 0.0)
2. Inelastic (gz = 0.15)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Ensure project root is in path
sys.path.append(os.path.abspath("."))

from src.models.ising_1d import IsingModel1D
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state
from qiskit.quantum_info import SparsePauliOp
import quimb.tensor as qtn

def run_scattering_sim(num_sites=20, g_x=1.25, g_z=0.0, num_steps=80, dt=0.125):
    print(f"\n--- Starting Simulation: L={num_sites}, gx={g_x}, gz={g_z} ---")
    model = IsingModel1D(num_sites=num_sites, g_x=g_x, g_z=g_z, pbc=True)
    backend = QuimbMPSBackend(max_bond_dim=32)
    
    # 1. Vacuum Measurements
    # For large gx, vacuum is approx |0...0> in X-basis.
    # Our code uses X-basis for transverse field.
    psi_vac = backend.get_reference_state(num_sites)
    
    # Measure vacuum energy density at each site
    print("Measuring vacuum energy density...")
    vac_energy_density = []
    for n in range(num_sites):
        En_op = model.get_local_hamiltonian(n)
        val = backend.compute_expectation_value(psi_vac, En_op)
        vac_energy_density.append(val)
    vac_energy_density = np.array(vac_energy_density)
    
    # 2. Initial State Preparation
    print("Preparing initial state (Two Wavepackets)...")
    # Centers at L/4 and 3L/4
    x1, x2 = num_sites / 4, 3 * num_sites / 4
    k1, k2 = 0.4 * np.pi, -0.4 * np.pi
    sigma = 1.0  # Slightly wider than the paper (0.8) for stability
    
    psi_np = prepare_two_wavepacket_state(
        num_sites,
        x1=x1, k1=k1, sigma1=sigma,
        x2=x2, k2=k2, sigma2=sigma,
        backend_type="numpy"
    )
    
    # Convert to MPS
    current_psi = qtn.MatrixProductState.from_dense(psi_np, [2]*num_sites)
    current_psi.compress(max_bond=32)
    
    # 3. Evolution
    heatmap_data = np.zeros((num_steps, num_sites))
    particle_num = []
    
    # Particle number operator N = sum 0.5(I - X_n)
    # We'll measure local occupation <n_i> = <0.5(I - X_i)>
    def get_local_occupation(psi):
        occ = []
        for i in range(num_sites):
            # 0.5 * (I - X_i)
            p = ["I"] * num_sites; p[i] = "X"
            op = SparsePauliOp.from_list([("".join(reversed(p)), -0.5), ("I"*num_sites, 0.5)])
            occ.append(backend.compute_expectation_value(psi, op))
        return np.array(occ)

    layers = model.get_trotter_layers()
    
    print(f"Evolving for {num_steps} steps...")
    start_time = time.time()
    for i in range(num_steps):
        if i % 10 == 0:
            print(f"  Step {i}/{num_steps} (Time: {time.time()-start_time:.1f}s)")
            
        # Measure Energy Density (En - Evac)
        for n in range(num_sites):
            En_op = model.get_local_hamiltonian(n)
            val = backend.compute_expectation_value(current_psi, En_op)
            heatmap_data[i, n] = val - vac_energy_density[n]
            
        # Measure Particle Number
        occ = get_local_occupation(current_psi)
        particle_num.append(np.sum(occ))
        
        # Evolve
        current_psi = backend.evolve_state_trotter(current_psi, layers, dt)
        
    return heatmap_data, particle_num

def plot_results(data, particle_num, filename, title, num_sites, num_steps, dt):
    # Set dark background style matching the dashboard theme
    plt.style.use('dark_background')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Heatmap with 'inferno' for high contrast
    # Use 'bicubic' interpolation for smooth "premium" look, or 'nearest' for scientific accuracy
    # Let's use 'gaussian' for a glowing effect
    im = ax1.imshow(data, extent=[0, num_sites-1, num_steps*dt, 0], aspect='auto', cmap='inferno', interpolation='gaussian')
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Energy Density (En - Evac)', fontsize=12)
    
    ax1.set_xlabel('Lattice Site (n)', fontsize=12)
    ax1.set_ylabel('Time (t)', fontsize=12)
    ax1.set_title(f'Energy Density: {title}', fontsize=14, fontweight='bold', color='cyan')
    
    # Particle Number
    times = np.arange(num_steps) * dt
    ax2.plot(times, particle_num, 'o-', color='#00f2ff', markersize=4, linewidth=2, label='Excitations <N>')
    
    # Reference lines
    ax2.axhline(y=2.0, color='#ff0055', linestyle='--', alpha=0.8, label='N=2 (Target)')
    
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Total Particle Number <N>', fontsize=12)
    ax2.set_title(f'Integrated Particle Count: {title}', fontsize=14, fontweight='bold', color='magenta')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    output_dir = "results/scattering"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved premium plot to {save_path}")

def main():
    # 1. Elastic Case (gz=0.0)
    h_elastic, n_elastic = run_scattering_sim(num_sites=20, g_x=1.25, g_z=0.0)
    plot_results(h_elastic, n_elastic, "ising1d_elastic.png", "Elastic (gz=0.0)", 20, 80, 0.125)
    
    # 2. Inelastic Case (gz=0.15)
    h_inelastic, n_inelastic = run_scattering_sim(num_sites=20, g_x=1.25, g_z=0.15)
    plot_results(h_inelastic, n_inelastic, "ising1d_inelastic.png", "Inelastic (gz=0.15)", 20, 80, 0.125)

if __name__ == "__main__":
    main()
