"""
Generate Simulation Data for the QSL Dashboard
==============================================

Runs a 1D Ising scattering simulation and exports the trajectory to JSON.
"""

import numpy as np
import json
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("."))

from src.models.ising_1d import IsingModel1D
from src.backends.quimb_backend import QuimbBackend
from src.simulation.initialization import prepare_two_wavepacket_state

def run_simulation(num_sites=14, steps=100, dt=0.1):
    print(f"Running simulation: L={num_sites}, T={steps*dt:.2f}")
    
    # 1. Setup Model
    model = IsingModel1D(num_sites=num_sites, g_x=1.25, g_z=0.15, pbc=True)
    backend = QuimbBackend(use_gpu=False)
    
    # 2. Vacuum subtraction - Use sparse solver
    from scipy.sparse.linalg import eigsh
    H_sparse = backend._pauli_to_matrix(model.build_hamiltonian())
    # We want Ground State. H_sparse is sparse matrix.
    # Note: for large systems this is slow, but for L=14 it's fine (~ok).
    evals, evecs = eigsh(H_sparse, k=1, which='SA')
    psi_vacuum = evecs[:, 0]
    
    vac_densities = []
    for n in range(num_sites):
        op = model.get_local_hamiltonian(n)
        vac_densities.append(backend.compute_expectation_value(psi_vacuum, op))
    vac_densities = np.array(vac_densities)

    # 3. Initial State
    # Create two wavepackets colliding
    psi = prepare_two_wavepacket_state(
        num_sites,
        x1=3.0, k1=0.35*np.pi, sigma1=1.0,
        x2=num_sites-4, k2=-0.35*np.pi, sigma2=1.0,
        backend_type="numpy"
    )
    
    # 4. Evolution
    trajectory = []
    layers = model.get_trotter_layers()
    
    current_psi = psi
    for t in range(steps):
        if t % 10 == 0: print(f"Step {t}/{steps}")
        
        # Measure local energy density
        densities = []
        for n in range(num_sites):
            op = model.get_local_hamiltonian(n)
            val = backend.compute_expectation_value(current_psi, op)
            densities.append(val - vac_densities[n])
        
        # Store data (convert to float for JSON)
        trajectory.append({
            "t": t * dt,
            "densities": [float(d) for d in densities]
        })
        
        # Trotter step
        current_psi = backend.evolve_state_trotter(current_psi, layers, dt)
        
    return {
        "num_sites": num_sites,
        "steps": steps,
        "dt": dt,
        "trajectory": trajectory
    }

if __name__ == "__main__":
    # Ensure dashboard data directory exists
    output_dir = "dashboard/public/data"
    os.makedirs(output_dir, exist_ok=True)
    
    data = run_simulation()
    
    output_path = os.path.join(output_dir, "scattering_data.json")
    
    with open(output_path, "w") as f:
        json.dump(data, f)
        
    print(f"Data exported to {output_path}")
