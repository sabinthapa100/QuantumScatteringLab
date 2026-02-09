#!/usr/bin/env python3
"""
Phase 0: Model Validation & Spectrum Analysis
==============================================

Systematically validates each physics model by computing:
1. Energy spectrum vs coupling parameter
2. Ground state characterization
3. Phase transition markers
4. Entanglement entropy

Author: Sabin Thapa
Date: 2026-02-09
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple

from src.models.ising_1d import IsingModel1D
from src.models.ising_2d import IsingModel2D
from src.models.su2 import SU2GaugeModel
from src.backends.quimb_mps_backend import QuimbMPSBackend


def setup_output_dirs():
    """Create output directories for results."""
    dirs = [
        "outputs/phase0_validation",
        "outputs/phase0_validation/spectrum",
        "outputs/phase0_validation/groundstates",
        "outputs/phase0_validation/data"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Output directories created")


def exact_diagonalization_1d_ising(L: int, g_x: float, g_z: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact diagonalization for small 1D Ising systems.
    Returns eigenvalues and eigenvectors.
    """
    from scipy.sparse.linalg import eigsh
    model = IsingModel1D(num_sites=L, g_x=g_x, g_z=g_z, pbc=True)
    H = model.build_hamiltonian()
    
    # Convert to dense matrix (only for small L!)
    H_dense = H.to_matrix()
    
    # Get lowest 5 eigenvalues
    evals, evecs = eigsh(H_dense, k=min(5, 2**L - 1), which='SA')
    return evals, evecs


def compute_spectrum_vs_coupling(model_type: str, sizes: List[int], 
                                  coupling_range: Tuple[float, float, int],
                                  use_dmrg: bool = True) -> Dict:
    """
    Compute energy spectrum as function of coupling parameter.
    
    Args:
        model_type: 'ising_1d', 'ising_2d', or 'su2_gauge'
        sizes: List of system sizes to test
        coupling_range: (min, max, num_points) for transverse field
        use_dmrg: Use DMRG (True) or exact diag (False, small L only)
    
    Returns:
        dict with results for each size
    """
    g_min, g_max, n_points = coupling_range
    g_values = np.linspace(g_min, g_max, n_points)
    
    results = {}
    backend = QuimbMPSBackend(max_bond_dim=64, cutoff=1e-10)
    
    for L in sizes:
        print(f"\n{'='*60}")
        print(f"Testing {model_type}, L={L}")
        print(f"{'='*60}")
        
        energies = []
        entropies = []
        
        for g_x in g_values:
            if model_type == "ising_1d":
                model = IsingModel1D(num_sites=L, g_x=g_x, g_z=0.0, pbc=True)
            elif model_type == "ising_2d":
                # For 2D, use square lattice
                Lx = Ly = int(np.sqrt(L))
                if Lx * Ly != L:
                    print(f"⚠️  Skipping L={L} (not a perfect square)")
                    continue
                model = IsingModel2D(Lx=Lx, Ly=Ly, g_x=g_x, g_z=0.0, pbc=False)
            elif model_type == "su2_gauge":
                model = SU2GaugeModel(num_sites=L, g=g_x, a=1.0, pbc=True)
            else:
                raise ValueError(f"Unknown model: {model_type}")
            
            if use_dmrg or L > 12:
                # DMRG for larger systems
                psi_gs = backend.get_ground_state(model)
                H_total = model.build_hamiltonian()
                E0 = backend.compute_expectation_value(psi_gs, H_total)
                
                # Compute entanglement entropy at center
                S_vn = psi_gs.entropy(L // 2)
            else:
                # Exact diagonalization for small systems
                if model_type == "ising_1d":
                    evals, _ = exact_diagonalization_1d_ising(L, g_x, 0.0)
                    E0 = evals[0]
                    S_vn = 0.0  # Would need eigenvector for this
                else:
                    print("⚠️  Exact diag only implemented for 1D Ising")
                    continue
            
            energies.append(E0 / L)  # Energy per site
            entropies.append(S_vn)
            print(f"  g_x={g_x:.3f}: E/L={E0/L:.6f}, S_vN={S_vn:.4f}")
        
        results[f"L{L}"] = {
            "g_values": g_values.tolist(),
            "energies_per_site": energies,
            "entropies": entropies
        }
    
    return results


def plot_spectrum_analysis(results: Dict, model_type: str, save_path: str):
    """Generate publication-quality plots of spectrum analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Energy vs coupling
    ax = axes[0]
    for key, data in results.items():
        L = int(key[1:])  # Extract L from "L8", "L10", etc.
        ax.plot(data["g_values"], data["energies_per_site"], 
                marker='o', label=f'L={L}', linewidth=2)
    
    ax.set_xlabel(r'Transverse Field $g_x$', fontsize=12)
    ax.set_ylabel(r'Ground State Energy $E_0/L$', fontsize=12)
    ax.set_title(f'{model_type.upper()}: Energy Spectrum', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mark critical point for Ising
    if "ising" in model_type:
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Critical Point')
    
    # Plot 2: Entanglement entropy
    ax = axes[1]
    for key, data in results.items():
        L = int(key[1:])
        ax.plot(data["g_values"], data["entropies"], 
                marker='s', label=f'L={L}', linewidth=2)
    
    ax.set_xlabel(r'Transverse Field $g_x$', fontsize=12)
    ax.set_ylabel(r'Half-Chain Entanglement $S_{vN}$', fontsize=12)
    ax.set_title(f'{model_type.upper()}: Quantum Correlations', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if "ising" in model_type:
        ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    plt.close()


def detect_phase_transition(results: Dict, model_type: str) -> Dict:
    """
    Detect phase transition by analyzing energy gap and entropy scaling.
    """
    analysis = {}
    
    for key, data in results.items():
        L = int(key[1:])
        g_vals = np.array(data["g_values"])
        energies = np.array(data["energies_per_site"])
        
        # Find minimum energy (most negative)
        min_idx = np.argmin(energies)
        
        # Compute derivative (gap indicator)
        dE_dg = np.gradient(energies, g_vals)
        
        # Peak in |dE/dg| indicates transition
        critical_idx = np.argmax(np.abs(dE_dg))
        g_critical = g_vals[critical_idx]
        
        analysis[f"L{L}"] = {
            "estimated_g_critical": g_critical,
            "energy_min": energies[min_idx],
            "max_derivative": float(np.max(np.abs(dE_dg)))
        }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Model Validation")
    parser.add_argument("--model", type=str, default="ising_1d",
                       choices=["ising_1d", "ising_2d", "su2_gauge"],
                       help="Model to test")
    parser.add_argument("--sizes", type=str, default="8,10,12",
                       help="Comma-separated list of system sizes")
    parser.add_argument("--coupling_range", type=str, default="0.0,2.0,21",
                       help="g_x range: min,max,num_points")
    parser.add_argument("--use_dmrg", action="store_true", default=True,
                       help="Use DMRG (default) instead of exact diag")
    
    args = parser.parse_args()
    
    # Parse inputs
    sizes = [int(s) for s in args.sizes.split(",")]
    g_min, g_max, n_pts = [float(x) for x in args.coupling_range.split(",")]
    n_pts = int(n_pts)
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║       QUANTUM SCATTERING LAB - Phase 0 Validation         ║
║                   Model: {args.model.upper():20s}          ║
╚════════════════════════════════════════════════════════════╝

Configuration:
  System sizes: {sizes}
  Coupling range: g_x ∈ [{g_min}, {g_max}] ({n_pts} points)
  Method: {'DMRG' if args.use_dmrg else 'Exact Diagonalization'}
    """)
    
    # Setup
    setup_output_dirs()
    
    # Run spectrum analysis
    print("\n[1/4] Computing energy spectrum...")
    results = compute_spectrum_vs_coupling(
        args.model, sizes, (g_min, g_max, n_pts), args.use_dmrg
    )
    
    # Generate plots
    print("\n[2/4] Generating plots...")
    plot_path = f"outputs/phase0_validation/spectrum/{args.model}_spectrum.png"
    plot_spectrum_analysis(results, args.model, plot_path)
    
    # Phase transition analysis
    print("\n[3/4] Analyzing phase transitions...")
    transition_analysis = detect_phase_transition(results, args.model)
    
    # Save data
    print("\n[4/4] Saving results...")
    output_data = {
        "model": args.model,
        "sizes": sizes,
        "coupling_range": [g_min, g_max, n_pts],
        "results": results,
        "transition_analysis": transition_analysis
    }
    
    data_path = f"outputs/phase0_validation/data/{args.model}_analysis.json"
    with open(data_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Data saved to {data_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for L_key, analysis in transition_analysis.items():
        print(f"\n{L_key}:")
        print(f"  Critical point g_c ≈ {analysis['estimated_g_critical']:.3f}")
        print(f"  Ground state energy: {analysis['energy_min']:.6f}")
    
    if "ising" in args.model:
        print("\n✓ For Ising model, critical point should be near g_x = 1.0")
    
    print(f"\n✅ Phase 0 validation complete for {args.model}")
    print(f"📊 Results saved in outputs/phase0_validation/")


if __name__ == "__main__":
    main()
