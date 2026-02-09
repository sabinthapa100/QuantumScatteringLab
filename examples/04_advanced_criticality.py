"""
Advanced Criticality Visualizations: Phase Diagrams, Scaling Collapse, Entanglement.

This script generates state-of-the-art visualizations for understanding quantum criticality.
"""
import sys
import os
sys.path.append(os.path.abspath("."))

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from src.models.ising_1d import IsingModel1D
from src.analysis.spectrum import SpectrumAnalyzer, scan_parameter
from src.analysis.criticality import (
    EntanglementAnalyzer, 
    extract_central_charge,
    ScalingCollapseAnalyzer,
    compute_correlation_length
)


def plot_2d_phase_diagram():
    """
    Create 2D phase diagram in (g_x, g_z) parameter space.
    Shows energy gap with critical line at g_x = 1.
    """
    print("=== 2D Phase Diagram: Ising Model ===")
    
    num_sites = 8
    
    # Parameter grid
    g_x_values = np.linspace(0.3, 1.7, 60)
    g_z_values = np.linspace(-0.3, 0.3, 50)
    
    # Compute gap on grid
    gap_grid = np.zeros((len(g_z_values), len(g_x_values)))
    
    print(f"Computing {len(g_x_values)} x {len(g_z_values)} = {len(g_x_values)*len(g_z_values)} points...")
    
    for i, g_z in enumerate(g_z_values):
        for j, g_x in enumerate(g_x_values):
            model = IsingModel1D(num_sites=num_sites, j_int=1.0, g_x=g_x, g_z=g_z, pbc=True)
            analyzer = SpectrumAnalyzer(model)
            result = analyzer.compute_spectrum(num_states=2)
            gap_grid[i, j] = result['gap']
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(g_z_values)} rows")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot gap as color map
    extent = [g_x_values[0], g_x_values[-1], g_z_values[0], g_z_values[-1]]
    im = ax.imshow(gap_grid, aspect='auto', origin='lower', extent=extent,
                   cmap='viridis', interpolation='bilinear')
    
    # Mark critical line g_x = 1
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=3, label='Critical line ($g_x = 1$)')
    
    # Overlay eta_latt contours
    G_X, G_Z = np.meshgrid(g_x_values, g_z_values)
    
    # Avoid division by zero
    eta_grid = np.zeros_like(G_X)
    valid_mask = np.abs(G_Z) > 1e-6
    eta_grid[valid_mask] = (G_X[valid_mask] - 1.0) / np.abs(G_Z[valid_mask])**(8.0/15.0)
    
    # Plot eta contours
    eta_levels = [-5, -2, -1, 0, 1, 2, 5]
    contours = ax.contour(G_X, G_Z, eta_grid, levels=eta_levels, 
                          colors='white', linewidths=1.5, alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=9, fmt=r'$\eta = %.1f$')
    
    # Mark paper parameters
    ax.scatter([1.25], [0.15], color='yellow', s=300, marker='*', 
               edgecolors='black', linewidths=2, label='Paper params', zorder=10)
    
    # Labels and aesthetics
    ax.set_xlabel(r'$g_x$ (Transverse Field)', fontsize=14)
    ax.set_ylabel(r'$g_z$ (Longitudinal Field)', fontsize=14)
    ax.set_title(f'Ising Model Phase Diagram (N={num_sites})\nColor = Energy Gap', fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'Gap $\Delta E$', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('examples/04_phase_diagram_2d.png', dpi=150)
    print("Saved: examples/04_phase_diagram_2d.png")
    plt.show()


def plot_scaling_collapse():
    """
    Demonstrate finite-size scaling collapse at criticality.
    
    Theory: Delta E = L^(-z) * f((g - g_c) * L^(1/nu))
    
    For 1D Ising: nu = 1, z = 1
    """
    print("\n=== Scaling Collapse Demonstration ===")
    
    sizes = [4, 6, 8, 10, 12]
    g_x_values = np.linspace(0.7, 1.3, 80)
    g_c = 1.0
    nu = 1.0
    z = 1.0
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Raw data
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))
    
    for idx, N in enumerate(sizes):
        result = scan_parameter(
            model_class=IsingModel1D,
            param_name='g_x',
            param_values=g_x_values,
            fixed_params={'num_sites': N, 'j_int': 1.0, 'g_z': 0.0, 'pbc': True},
            num_states=4
        )
        ax1.plot(result['param_values'], result['gaps'], 
                 'o-', color=colors[idx], label=f'L={N}', markersize=4, linewidth=2)
    
    ax1.axvline(x=g_c, color='red', linestyle='--', alpha=0.7, label='$g_c = 1$')
    ax1.set_xlabel(r'$g_x$', fontsize=13)
    ax1.set_ylabel(r'Gap $\Delta E$', fontsize=13)
    ax1.set_title('Raw Data (No Collapse)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Scaled data (collapse)
    ax2 = axes[1]
    
    for idx, N in enumerate(sizes):
        result = scan_parameter(
            model_class=IsingModel1D,
            param_name='g_x',
            param_values=g_x_values,
            fixed_params={'num_sites': N, 'j_int': 1.0, 'g_z': 0.0, 'pbc': True},
            num_states=4
        )
        
        # Apply scaling
        x_scaled = (result['param_values'] - g_c) * (N ** (1.0 / nu))
        y_scaled = result['gaps'] * (N ** z)
        
        ax2.plot(x_scaled, y_scaled, 
                 'o', color=colors[idx], label=f'L={N}', markersize=5, alpha=0.7)
    
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_xlabel(r'$(g_x - g_c) \times L^{1/\nu}$', fontsize=13)
    ax2.set_ylabel(r'$\Delta E \times L^z$', fontsize=13)
    ax2.set_title(r'Scaled Data (Collapse with $\nu=1, z=1$)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-15, 15)
    
    plt.tight_layout()
    plt.savefig('examples/04_scaling_collapse.png', dpi=150)
    print("Saved: examples/04_scaling_collapse.png")
    plt.show()


def plot_entanglement_entropy():
    """
    Compute and visualize entanglement entropy at criticality.
    Extract central charge c from S(ell) = (c/3) * log(ell) + const.
    """
    print("\n=== Entanglement Entropy Analysis ===")
    
    # Analyze at critical point
    num_sites = 12
    model = IsingModel1D(num_sites=num_sites, j_int=1.0, g_x=1.0, g_z=0.0, pbc=True)
    
    print(f"Computing ground state for N={num_sites} at criticality...")
    analyzer_spectrum = SpectrumAnalyzer(model)
    E_gs, psi_gs = analyzer_spectrum.compute_ground_state()
    
    print("Computing entanglement entropy...")
    analyzer_ent = EntanglementAnalyzer(num_sites)
    entropies = analyzer_ent.scan_subsystem_sizes(psi_gs)
    
    # Convert to arrays
    ell_values = np.array(sorted(entropies.keys()))
    S_values = np.array([entropies[ell] for ell in ell_values])
    
    # Extract central charge (fit middle region to avoid edge effects)
    fit_start = 2
    fit_end = num_sites - 3
    c, offset = extract_central_charge(
        ell_values, S_values, 
        fit_range=(fit_start, fit_end)
    )
    
    print(f"Extracted central charge: c = {c:.4f} (Theory: c = 0.5 for Ising CFT)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Data points
    ax.plot(ell_values, S_values, 'o', markersize=10, label='Numerical data', color='blue')
    
    # Fit curve
    ell_fit = np.linspace(ell_values[0], ell_values[-1], 100)
    S_fit = (c / 3.0) * np.log(ell_fit) + offset
    ax.plot(ell_fit, S_fit, '--', linewidth=2, color='red', 
            label=f'Fit: $S = ({c:.3f}/3)\log(\ell) + {offset:.2f}$')
    
    ax.set_xlabel(r'Subsystem size $\ell$', fontsize=13)
    ax.set_ylabel(r'Entanglement entropy $S(\ell)$', fontsize=13)
    ax.set_title(f'Entanglement Entropy at Criticality ($g_x=1.0$, N={num_sites})\n' + 
                 f'Central Charge: $c = {c:.4f}$ (Theory: $c=0.5$)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/04_entanglement_entropy.png', dpi=150)
    print("Saved: examples/04_entanglement_entropy.png")
    plt.show()
    
    return c


def plot_entanglement_across_transition():
    """
    Show how entanglement entropy changes across the phase transition.
    """
    print("\n=== Entanglement Across Phase Transition ===")
    
    num_sites = 10
    subsystem_size = num_sites // 2  # Half-chain cut
    
    g_x_values = np.linspace(0.5, 1.5, 30)
    entropies = []
    
    print(f"Computing entanglement for {len(g_x_values)} points...")
    
    for g_x in g_x_values:
        model = IsingModel1D(num_sites=num_sites, j_int=1.0, g_x=g_x, g_z=0.0, pbc=True)
        analyzer_spectrum = SpectrumAnalyzer(model)
        E_gs, psi_gs = analyzer_spectrum.compute_ground_state()
        
        analyzer_ent = EntanglementAnalyzer(num_sites)
        S = analyzer_ent.compute_entanglement_entropy(psi_gs, subsystem_size)
        entropies.append(S)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(g_x_values, entropies, 'o-', linewidth=2, markersize=8, color='purple')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Critical point')
    
    ax.set_xlabel(r'$g_x$ (Transverse Field)', fontsize=13)
    ax.set_ylabel(rf'Entanglement Entropy $S_{{\ell={subsystem_size}}}$', fontsize=13)
    ax.set_title(f'Entanglement Entropy Across Phase Transition (N={num_sites})', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/04_entanglement_transition.png', dpi=150)
    print("Saved: examples/04_entanglement_transition.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("     ADVANCED CRITICALITY VISUALIZATIONS")
    print("     Phase Diagrams | Scaling Collapse | Entanglement")
    print("=" * 70)
    
    # 1. 2D Phase Diagram
    plot_2d_phase_diagram()
    
    # 2. Scaling Collapse
    plot_scaling_collapse()
    
    # 3. Entanglement Entropy at Criticality
    c_extracted = plot_entanglement_entropy()
    
    # 4. Entanglement Across Transition
    plot_entanglement_across_transition()
    
    print("\n" + "=" * 70)
    print("     ANALYSIS COMPLETE")
    print(f"     Extracted Central Charge: c = {c_extracted:.4f}")
    print(f"     Theory Prediction: c = 0.5 (Ising CFT)")
    print("=" * 70)
