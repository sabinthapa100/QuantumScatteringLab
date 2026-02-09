"""
Ising Model Spectrum Analysis.

This script visualizes:
1. Energy levels as a function of transverse field g_x.
2. The energy gap closing at the critical point g_x = 1.
3. The effect of the longitudinal field g_z on the spectrum.

Based on the Ising Field Theory from arXiv:2505.03111v2.
"""
import sys
import os
sys.path.append(os.path.abspath("."))

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.models.ising_1d import IsingModel1D
from src.analysis.spectrum import SpectrumAnalyzer, scan_parameter


def plot_spectrum_vs_gx(num_sites: int = 8, g_z: float = 0.0):
    """
    Plot energy levels as a function of transverse field g_x.
    The critical point is at g_x = 1 (for g_z = 0).
    """
    print(f"=== Ising Spectrum vs g_x (N={num_sites}, g_z={g_z}) ===")
    
    g_x_values = np.linspace(0.1, 2.0, 50)
    
    result = scan_parameter(
        model_class=IsingModel1D,
        param_name='g_x',
        param_values=g_x_values,
        fixed_params={'num_sites': num_sites, 'j_int': 1.0, 'g_z': g_z, 'pbc': True},
        num_states=8
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Energy levels
    ax1 = axes[0]
    for i in range(result['energies'].shape[1]):
        ax1.plot(result['param_values'], result['energies'][:, i], 
                 label=f'$E_{i}$', linewidth=1.5)
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Critical Point')
    ax1.set_xlabel(r'$g_x$ (Transverse Field)', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'Ising Energy Spectrum (N={num_sites})', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Energy Gap
    ax2 = axes[1]
    ax2.plot(result['param_values'], result['gaps'], 'b-', linewidth=2)
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Critical Point')
    ax2.set_xlabel(r'$g_x$ (Transverse Field)', fontsize=12)
    ax2.set_ylabel(r'Gap $\Delta E = E_1 - E_0$', fontsize=12)
    ax2.set_title('Energy Gap (Closes at Criticality)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Highlight the gap minimum
    min_idx = np.argmin(result['gaps'])
    ax2.scatter([result['param_values'][min_idx]], [result['gaps'][min_idx]], 
                color='red', s=100, zorder=5, label=f'Min at g_x={result["param_values"][min_idx]:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('examples/02_ising_spectrum_gx.png', dpi=150)
    print("Saved: examples/02_ising_spectrum_gx.png")
    plt.show()
    
    return result


def plot_spectrum_vs_gz(num_sites: int = 8, g_x: float = 1.25):
    """
    Plot energy levels as a function of longitudinal field g_z.
    This breaks the Z2 symmetry.
    """
    print(f"\n=== Ising Spectrum vs g_z (N={num_sites}, g_x={g_x}) ===")
    
    g_z_values = np.linspace(-0.5, 0.5, 50)
    
    result = scan_parameter(
        model_class=IsingModel1D,
        param_name='g_z',
        param_values=g_z_values,
        fixed_params={'num_sites': num_sites, 'j_int': 1.0, 'g_x': g_x, 'pbc': True},
        num_states=8
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Energy levels
    ax1 = axes[0]
    for i in range(result['energies'].shape[1]):
        ax1.plot(result['param_values'], result['energies'][:, i], 
                 label=f'$E_{i}$', linewidth=1.5)
    ax1.axvline(x=0.0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel(r'$g_z$ (Longitudinal Field)', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'Ising Energy Spectrum (N={num_sites}, $g_x$={g_x})', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Energy Gap
    ax2 = axes[1]
    ax2.plot(result['param_values'], result['gaps'], 'b-', linewidth=2)
    ax2.set_xlabel(r'$g_z$ (Longitudinal Field)', fontsize=12)
    ax2.set_ylabel(r'Gap $\Delta E = E_1 - E_0$', fontsize=12)
    ax2.set_title('Energy Gap vs Longitudinal Field', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/02_ising_spectrum_gz.png', dpi=150)
    print("Saved: examples/02_ising_spectrum_gz.png")
    plt.show()
    
    return result


def plot_criticality_scaling(sizes: List[int] = [4, 6, 8, 10]):
    """
    Demonstrate that the gap closes as 1/N at the critical point.
    This is a hallmark of criticality.
    """
    print("\n=== Finite-Size Scaling at Criticality ===")
    
    g_x_values = np.linspace(0.5, 1.5, 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gaps_at_critical = []
    
    for N in sizes:
        result = scan_parameter(
            model_class=IsingModel1D,
            param_name='g_x',
            param_values=g_x_values,
            fixed_params={'num_sites': N, 'j_int': 1.0, 'g_z': 0.0, 'pbc': True},
            num_states=4
        )
        ax.plot(result['param_values'], result['gaps'], label=f'N={N}', linewidth=2)
        
        # Find gap at critical point (g_x = 1)
        critical_idx = np.argmin(np.abs(result['param_values'] - 1.0))
        gaps_at_critical.append((N, result['gaps'][critical_idx]))
    
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Critical')
    ax.set_xlabel(r'$g_x$ (Transverse Field)', fontsize=12)
    ax.set_ylabel(r'Gap $\Delta E$', fontsize=12)
    ax.set_title('Finite-Size Scaling: Gap Closes at $g_x = 1$', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/02_ising_fss.png', dpi=150)
    print("Saved: examples/02_ising_fss.png")
    plt.show()
    
    # Plot gap vs 1/N at critical point
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    Ns = [x[0] for x in gaps_at_critical]
    gaps = [x[1] for x in gaps_at_critical]
    ax2.plot([1/n for n in Ns], gaps, 'o-', markersize=10, linewidth=2)
    ax2.set_xlabel(r'$1/N$', fontsize=12)
    ax2.set_ylabel(r'Gap $\Delta E$ at $g_x = 1$', fontsize=12)
    ax2.set_title(r'Gap Scaling: $\Delta E \sim 1/N$ at Criticality', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/02_ising_gap_scaling.png', dpi=150)
    print("Saved: examples/02_ising_gap_scaling.png")
    plt.show()


def plot_paper_parameters():
    """
    Visualize the spectrum at the paper's parameter values:
    g_x = 1.25, g_z = 0.15
    
    At these values, there are two stable particles with masses:
    m1 = 1.59, m2 = 2.98
    """
    print("\n=== Paper Parameter Values (g_x=1.25, g_z=0.15) ===")
    
    model = IsingModel1D(num_sites=10, j_int=1.0, g_x=1.25, g_z=0.15, pbc=True)
    analyzer = SpectrumAnalyzer(model)
    
    # Full spectrum for a small system
    spectrum = analyzer.compute_full_spectrum()
    
    # Shift so ground state is at 0
    spectrum_shifted = spectrum - spectrum[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot energy levels as horizontal lines
    for i, E in enumerate(spectrum_shifted[:20]):  # First 20 levels
        ax.axhline(y=E, color='blue', alpha=0.6, linewidth=1)
        if i < 5:
            ax.text(1.02, E, f'$E_{i}$', fontsize=9, va='center')
    
    # Mark the expected particle masses (approximate, for visualization)
    # These are NOT exact matches but illustrate the concept
    ax.set_xlim(0, 1.2)
    ax.set_ylabel('Energy (shifted)', fontsize=12)
    ax.set_title(r'Ising Spectrum at $g_x=1.25$, $g_z=0.15$', fontsize=14)
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/02_ising_paper_params.png', dpi=150)
    print("Saved: examples/02_ising_paper_params.png")
    plt.show()
    
    print(f"\nFirst 10 energy levels (shifted from GS=0):")
    for i, E in enumerate(spectrum_shifted[:10]):
        print(f"  E_{i} = {E:.4f}")




if __name__ == "__main__":
    # Run all spectrum analyses
    print("=" * 60)
    print("     ISING MODEL SPECTRUM ANALYSIS")
    print("     Based on arXiv:2505.03111v2")
    print("=" * 60)
    
    # 1. Spectrum vs g_x (shows critical point)
    plot_spectrum_vs_gx(num_sites=8)
    
    # 2. Spectrum vs g_z (shows symmetry breaking)
    plot_spectrum_vs_gz(num_sites=8)
    
    # 3. Finite-size scaling
    plot_criticality_scaling(sizes=[4, 6, 8, 10])
    
    # 4. Paper's exact parameters
    plot_paper_parameters()
    
    print("\n" + "=" * 60)
    print("     ANALYSIS COMPLETE")
    print("=" * 60)
