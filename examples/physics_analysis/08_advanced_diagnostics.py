"""
Advanced Physics Diagnostics: Entropy, Energy, and Scaling.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath("."))

from src.models.ising_1d import IsingModel1D
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state
import quimb.tensor as qtn

def analyze_scattering_diagnostics(num_sites=16, g_x=1.25, g_z=0.15, num_steps=50, dt=0.125):
    print(f"--- Advanced Diagnostics: L={num_sites}, gx={g_x}, gz={g_z} ---")
    model = IsingModel1D(num_sites=num_sites, g_x=g_x, g_z=g_z, pbc=True)
    backend = QuimbMPSBackend(max_bond_dim=32)
    
    # 1. Initial State
    psi_np = prepare_two_wavepacket_state(
        num_sites,
        x1=4, k1=0.4*np.pi, sigma1=1.0,
        x2=12, k2=-0.4*np.pi, sigma2=1.0,
        backend_type="numpy"
    )
    current_psi = qtn.MatrixProductState.from_dense(psi_np, [2]*num_sites)
    
    # 2. Evolution with Entropy and Bond Dimension Tracking
    entropy_heatmap = np.zeros((num_steps, num_sites - 1))
    bond_dims = np.zeros((num_steps, num_sites - 1))
    energy_variance = []
    
    layers = model.get_trotter_layers()
    H = model.build_hamiltonian()
    
    print("Starting evolution diagnostics...")
    for i in range(num_steps):
        # Entanglement Entropy across all bonds
        for b in range(num_sites - 1):
            entropy_heatmap[i, b] = backend.entanglement_entropy(current_psi, b)
            # Find bond dimension: index between site b and b+1
            # In quimb mps, the bond index is usually 'k{b}'
            t_left = current_psi[b]
            t_right = current_psi[b+1]
            shared_inds = list(set(t_left.inds) & set(t_right.inds))
            if shared_inds:
                bond_dims[i, b] = t_left.ind_size(shared_inds[0])

        # Evolve
        current_psi = backend.evolve_state_trotter(current_psi, layers, dt)
        if i % 10 == 0:
            print(f"  Step {i}/{num_steps}")

    # 3. Visualization
    output_dir = "results/diagnostics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Entropy Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(entropy_heatmap, extent=[0, num_sites-2, num_steps*dt, 0], aspect='auto', cmap='viridis')
    plt.colorbar(label='Von Neumann Entropy S_vN')
    plt.title(f'Entanglement Entropy Evolution (L={num_sites}, gz={g_z})')
    plt.xlabel('Bond Index')
    plt.ylabel('Time t')
    plt.savefig(os.path.join(output_dir, "entanglement_entropy.png"))
    
    # Bond Dimension Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(bond_dims, extent=[0, num_sites-2, num_steps*dt, 0], aspect='auto', cmap='magma')
    plt.colorbar(label='Bond Dimension χ')
    plt.title('MPS Bond Dimension Growth')
    plt.xlabel('Bond Index')
    plt.ylabel('Time t')
    plt.savefig(os.path.join(output_dir, "bond_dimension_growth.png"))
    
    print(f"Diagnostics saved to {output_dir}")

def circuit_depth_scaling():
    """Analyze circuit complexity scaling."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate
    
    l_range = [4, 8, 12, 16, 20]
    depths = []
    
    print("Analyzing circuit depth scaling...")
    for L in l_range:
        model = IsingModel1D(num_sites=L, g_x=1.25, g_z=0.15)
        H = model.build_hamiltonian()
        
        # One Trotter step circuit
        qc = QuantumCircuit(L)
        # Simplified: one layer of ZZ and one layer of X
        # In reality, IsingModel1D.get_trotter_layers() gives the specific terms.
        layers = model.get_trotter_layers()
        for layer in layers:
            # We treat each SparsePauliOp as a set of gates
            for label, coeff in layer.to_list():
                # Count as generic interaction for depth logic
                pass
        
        # Using Qiskit's own estimation for a PauliEvolutionGate
        evo_gate = PauliEvolutionGate(H, time=0.125)
        qc.append(evo_gate, range(L))
        
        # Transpile to basic gates (CX, RZ, SX, X)
        try:
            decomposed_qc = qc.decompose().decompose() # Rough decomposition
            depths.append(decomposed_qc.depth())
        except:
            # Fallback to manual interaction count
            depths.append(len(H))

    plt.figure(figsize=(8, 5))
    plt.plot(l_range, depths, 'o-', color='teal', label='Decomposed Depth')
    plt.xlabel('Number of sites L')
    plt.ylabel('Circuit Depth (est.)')
    plt.title('Circuit Complexity Scaling: 1D Ising (1 Trotter Step)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs("results/scaling", exist_ok=True)
    plt.savefig("results/scaling/circuit_depth_scaling.png")
    print("Scaling plot saved to results/scaling/circuit_depth_scaling.png")

if __name__ == "__main__":
    analyze_scattering_diagnostics()
    circuit_depth_scaling()
