"""
SU(2) Gauge Model Spectrum Analysis.

This script visualizes:
1. Energy levels as a function of coupling g.
2. The phase transition between strong and weak coupling regimes.
3. Comparison with exact diagonalization.

Based on Kogut-Susskind Hamiltonian for SU(2) lattice gauge theory.
"""
import sys
import os
sys.path.append(os.path.abspath("."))

from typing import List
import numpy as np
import matplotlib.pyplot as plt

from src.models.su2 import SU2GaugeModel
from src.analysis.spectrum import SpectrumAnalyzer, scan_parameter


def plot_spectrum_vs_g(num_sites: int = 4):
    """
    Plot energy levels as a function of coupling g.
    
    For SU(2) gauge theory:
    - Strong coupling (g >> 1): Electric term dominates, confining
    - Weak coupling (g << 1): Magnetic term dominates
    """
    print(f"=== SU(2) Spectrum vs g (N={num_sites}) ===")
    
    g_values = np.linspace(0.5, 3.0, 40)
    
    result = scan_parameter(
        model_class=SU2GaugeModel,
        param_name='g',
        param_values=g_values,
        fixed_params={'num_sites': num_sites},
        num_states=8
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Energy levels
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, result['energies'].shape[1]))
    for i in range(result['energies'].shape[1]):
        ax1.plot(result['param_values'], result['energies'][:, i], 
                 color=colors[i], label=f'$E_{i}$', linewidth=1.5)
    ax1.set_xlabel(r'$g$ (Coupling)', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'SU(2) Energy Spectrum (N={num_sites})', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Energy Gap
    ax2 = axes[1]
    ax2.plot(result['param_values'], result['gaps'], 'b-', linewidth=2)
    ax2.set_xlabel(r'$g$ (Coupling)', fontsize=12)
    ax2.set_ylabel(r'Gap $\Delta E = E_1 - E_0$', fontsize=12)
    ax2.set_title('Energy Gap vs Coupling', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/03_su2_spectrum_g.png', dpi=150)
    print("Saved: examples/03_su2_spectrum_g.png")
    plt.show()
    
    return result


def plot_full_spectrum(num_sites: int = 3, g: float = 1.0):
    """
    Visualize the full energy spectrum for a small system.
    """
    print(f"\n=== SU(2) Full Spectrum (N={num_sites}, g={g}) ===")
    
    model = SU2GaugeModel(num_sites=num_sites, g=g)
    analyzer = SpectrumAnalyzer(model)
    
    spectrum = analyzer.compute_full_spectrum()
    spectrum_shifted = spectrum - spectrum[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot energy levels as horizontal lines with degeneracy coloring
    unique_energies, counts = np.unique(np.round(spectrum_shifted[:30], 6), return_counts=True)
    
    for i, (E, count) in enumerate(zip(unique_energies, counts)):
        color = 'blue' if count == 1 else ('green' if count == 2 else 'red')
        ax.axhline(y=E, color=color, alpha=0.7, linewidth=2)
        ax.text(1.02, E, f'deg={count}', fontsize=8, va='center')
    
    ax.set_xlim(0, 1.15)
    ax.set_ylabel('Energy (shifted from GS)', fontsize=12)
    ax.set_title(f'SU(2) Spectrum (N={num_sites}, g={g})\nBlue=1, Green=2, Red=3+ degeneracy', fontsize=12)
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/03_su2_full_spectrum.png', dpi=150)
    print("Saved: examples/03_su2_full_spectrum.png")
    plt.show()
    
    print(f"\nFirst 10 energy levels (shifted):")
    for i, E in enumerate(spectrum_shifted[:10]):
        print(f"  E_{i} = {E:.4f}")


def compare_sizes():
    """
    Compare spectra for different system sizes.
    """
    print("\n=== SU(2) Finite-Size Comparison ===")
    
    g = 1.0
    sizes = [2, 3, 4]
    
    fig, axes = plt.subplots(1, len(sizes), figsize=(15, 5))
    
    for ax, N in zip(axes, sizes):
        model = SU2GaugeModel(num_sites=N, g=g)
        analyzer = SpectrumAnalyzer(model)
        
        try:
            spectrum = analyzer.compute_full_spectrum()
            spectrum_shifted = spectrum - spectrum[0]
            
            # Plot first 15 levels
            for i, E in enumerate(spectrum_shifted[:15]):
                ax.axhline(y=E, color='blue', alpha=0.6, linewidth=1.5)
            
            ax.set_ylabel('Energy' if N == sizes[0] else '')
            ax.set_title(f'N={N} sites', fontsize=12)
            ax.set_xticks([])
            ax.grid(True, axis='y', alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', transform=ax.transAxes,
                   ha='center', va='center')
    
    plt.suptitle(f'SU(2) Spectrum Comparison (g={g})', fontsize=14)
    plt.tight_layout()
    plt.savefig('examples/03_su2_size_comparison.png', dpi=150)
    print("Saved: examples/03_su2_size_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("     SU(2) GAUGE MODEL SPECTRUM ANALYSIS")
    print("     Kogut-Susskind Formulation")
    print("=" * 60)
    
    # 1. Spectrum vs coupling g
    plot_spectrum_vs_g(num_sites=4)
    
    # 2. Full spectrum at specific g
    plot_full_spectrum(num_sites=3, g=1.0)
    
    # 3. Size comparison
    compare_sizes()
    
    print("\n" + "=" * 60)
    print("     ANALYSIS COMPLETE")
    print("=" * 60)
