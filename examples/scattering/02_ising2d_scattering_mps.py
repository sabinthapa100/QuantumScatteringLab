"""
Scattering in 2D Ising Model using MPS.
Two wavepackets colliding in a 2D lattice.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

# Ensure project root is in path
sys.path.append(os.path.abspath("."))

from src.models.ising_2d import IsingModel2D
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state_2d
import quimb.tensor as qtn

def run_2d_scattering(Lx=6, Ly=4, g_x=3.5, g_z=0.0, num_steps=50, dt=0.1):
    print(f"\n--- 2D Scattering: {Lx}x{Ly}, gx={g_x}, gz={g_z} ---")
    model = IsingModel2D(Lx=Lx, Ly=Ly, g_x=g_x, g_z=g_z, pbc=True)
    num_sites = Lx * Ly
    backend = QuimbMPSBackend(max_bond_dim=64)
    
    # 1. Vacuum
    psi_vac = backend.get_reference_state(num_sites)
    print("Measuring vacuum energy density...")
    vac_energy = np.zeros((Ly, Lx))
    for ny in range(Ly):
        for nx in range(Lx):
            op = model.get_local_hamiltonian(nx, ny)
            vac_energy[ny, nx] = backend.compute_expectation_value(psi_vac, op)
            
    # 2. Initial State
    print("Preparing initial state (2D Wavepackets)...")
    # WP1: Bottom left, moving up-right
    # WP2: Top right, moving down-left
    psi_np = prepare_two_wavepacket_state_2d(
        Lx, Ly,
        x1=1.0, y1=1.0, kx1=0.4*np.pi, ky1=0.2*np.pi, sigma1=0.8,
        x2=5.0, y2=3.0, kx2=-0.4*np.pi, ky2=-0.2*np.pi, sigma2=0.8,
        backend_type="numpy"
    )
    
    current_psi = qtn.MatrixProductState.from_dense(psi_np, [2]*num_sites)
    current_psi.compress(max_bond=64)
    
    # 3. Evolution
    print(f"Evolving for {num_steps} steps...")
    layers = model.get_trotter_layers()
    
    # We'll save energy density at t=0, t=mid, t=end
    frames = []
    capture_steps = [0, num_steps//2, num_steps-1]
    
    for i in range(num_steps):
        if i in capture_steps:
            print(f"  Capturing frame at step {i}")
            frame = np.zeros((Ly, Lx))
            for ny in range(Ly):
                for nx in range(Lx):
                    op = model.get_local_hamiltonian(nx, ny)
                    val = backend.compute_expectation_value(current_psi, op)
                    frame[ny, nx] = val - vac_energy[ny, nx]
            frames.append(frame)
            
        if i % 10 == 0:
            print(f"  Step {i}/{num_steps}")
            
        current_psi = backend.evolve_state_trotter(current_psi, layers, dt)
        
    return frames

def plot_2d_frames(frames, filename):
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1: axes = [axes]
    
    titles = ["Initial (t=0)", "Mid-Collision", "Final"]
    for i, frame in enumerate(frames):
        im = axes[i].imshow(frame, origin='lower', cmap='hot', interpolation='gaussian')
        axes[i].set_title(titles[i])
        plt.colorbar(im, ax=axes[i])
        
    plt.tight_layout()
    output_dir = "results/scattering"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    print(f"Saved 2D plot to {output_dir}/{filename}")

def main():
    frames = run_2d_scattering(Lx=6, Ly=4, g_x=3.5, g_z=0.0, num_steps=40)
    plot_2d_frames(frames, "ising2d_scattering.png")

if __name__ == "__main__":
    main()
