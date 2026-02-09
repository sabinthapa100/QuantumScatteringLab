"""
Ising Model: PBC vs OBC Comparison

Demonstrates the effects of boundary conditions on:
1. Energy spectra
2. Entanglement entropy and central charge extraction
3. Phase diagrams

Shows that OBC gives correct central charge c=0.5 at criticality.
"""
import sys
import os
sys.path.append(os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt

from src.models.ising_1d import IsingModel1D
from src.analysis.framework import (
    AnalysisConfig,
    BoundaryCondition,
    ModelAnalyzer
)
from src.visualization.phase_diagrams import (
    plot_entanglement_scaling,
    compare_boundary_conditions
)


def compare_entanglement_pbc_vs_obc():
    """
    Compare entanglement entropy between PBC and OBC at criticality.
    This is the KEY comparison - OBC should give c = 0.5.
    """
    print("=" * 70)
    print("  Entanglement Entropy: PBC vs OBC at Criticality")
    print("=" * 70)
    
    # Critical parameters
    critical_params = {'g_x': 1.0, 'g_z': 0.0, 'j_int': 1.0}
    system_size = 14
    
    # PBC Analysis
    print("\n[1/2] Computing with PBC...")
    config_pbc = AnalysisConfig(
        model_class=IsingModel1D,
        fixed_params=critical_params,
        boundary_condition=BoundaryCondition.PBC,
        system_sizes=[system_size]
    )
    analyzer_pbc = ModelAnalyzer(config_pbc)
    ent_data_pbc = analyzer_pbc.compute_entanglement_at_criticality(critical_params, system_size)
    
    print(f"  PBC: c = {ent_data_pbc.central_charge:.4f}")
    
    # OBC Analysis
    print("\n[2/2] Computing with OBC...")
    config_obc = AnalysisConfig(
        model_class=IsingModel1D,
        fixed_params=critical_params,
        boundary_condition=BoundaryCondition.OBC,
        system_sizes=[system_size]
    )
    analyzer_obc = ModelAnalyzer(config_obc)
    ent_data_obc = analyzer_obc.compute_entanglement_at_criticality(critical_params, system_size)
    
    print(f"  OBC: c = {ent_data_obc.central_charge:.4f}")
    print(f"  Theory: c = 0.5 (Majorana fermion CFT)")
    
    # Plot individual results
    plot_entanglement_scaling(
        ent_data_pbc,
        save_path='examples/05_ising_entanglement_pbc.png',
        title=f'Entanglement Entropy with PBC (N={system_size})'
    )
    
    plot_entanglement_scaling(
        ent_data_obc,
        save_path='examples/05_ising_entanglement_obc.png',
        title=f'Entanglement Entropy with OBC (N={system_size})'
    )
    
    # Side-by-side comparison
    compare_boundary_conditions(
        ent_data_pbc,
        ent_data_obc,
        comparison_type='entanglement',
        save_path='examples/05_ising_pbc_vs_obc_comparison.png',
        suptitle=f'Ising Model: PBC vs OBC at Criticality (N={system_size})'
    )
    
    return ent_data_pbc, ent_data_obc


def compare_spectra_pbc_vs_obc():
    """
    Compare energy spectra between PBC and OBC.
    """
    print("\n" + "=" * 70)
    print("  Energy Spectrum: PBC vs OBC")
    print("=" * 70)
    
    num_sites = 10
    g_x_values = np.linspace(0.5, 1.5, 40)
    
    # PBC
    print("\n[1/2] Computing spectrum with PBC...")
    config_pbc = AnalysisConfig(
        model_class=IsingModel1D,
        fixed_params={'j_int': 1.0, 'g_z': 0.0},
        boundary_condition=BoundaryCondition.PBC
    )
    analyzer_pbc = ModelAnalyzer(config_pbc)
    _, gaps_pbc = analyzer_pbc.compute_1d_phase_scan('g_x', g_x_values, num_sites)
    
    # OBC
    print("[2/2] Computing spectrum with OBC...")
    config_obc = AnalysisConfig(
        model_class=IsingModel1D,
        fixed_params={'j_int': 1.0, 'g_z': 0.0},
        boundary_condition=BoundaryCondition.OBC
    )
    analyzer_obc = ModelAnalyzer(config_obc)
    _, gaps_obc = analyzer_obc.compute_1d_phase_scan('g_x', g_x_values, num_sites)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(g_x_values, gaps_pbc, 'o-', label='PBC', linewidth=2, markersize=4)
    ax.plot(g_x_values, gaps_obc, 's-', label='OBC', linewidth=2, markersize=4, alpha=0.7)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Critical')
    
    ax.set_xlabel(r'$g_x$ (Transverse Field)', fontsize=13)
    ax.set_ylabel(r'Gap $\Delta E$', fontsize=13)
    ax.set_title(f'Ising Model Energy Gap: PBC vs OBC (N={num_sites})', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/05_ising_gap_pbc_vs_obc.png', dpi=150)
    print("Saved: examples/05_ising_gap_pbc_vs_obc.png")
    plt.show()


def analyze_finite_size_scaling_obc():
    """
    Demonstrate finite-size scaling with OBC.
    """
    print("\n" + "=" * 70)
    print("  Finite-Size Scaling with OBC")
    print("=" * 70)
    
    config = AnalysisConfig(
        model_class=IsingModel1D,
        fixed_params={'j_int': 1.0, 'g_z': 0.0},
        boundary_condition=BoundaryCondition.OBC,
        system_sizes=[6, 8, 10, 12, 14]
    )
    
    analyzer = ModelAnalyzer(config)
    
    g_x_values = np.linspace(0.7, 1.3, 60)
    
    print(f"Computing scaling data for {len(config.system_sizes)} system sizes...")
    scaling_data = analyzer.compute_scaling_collapse(
        param_name='g_x',
        param_values=g_x_values,
        param_critical=1.0,
        nu=1.0,
        z=1.0
    )
    
    from src.visualization.phase_diagrams import plot_scaling_collapse
    
    plot_scaling_collapse(
        scaling_data,
        save_path='examples/05_ising_scaling_obc.png',
        xlabel_raw=r'$g_x$',
        title_scaled=r'Scaled Data (OBC, $\nu=1, z=1$)'
    )


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ISING MODEL: PBC vs OBC ANALYSIS" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 1. Entanglement comparison
    ent_pbc, ent_obc = compare_entanglement_pbc_vs_obc()
    
    # 2. Spectrum comparison
    compare_spectra_pbc_vs_obc()
    
    # 3. Finite-size scaling with OBC
    analyze_finite_size_scaling_obc()
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  PBC Central Charge:  c = {ent_pbc.central_charge:.4f}")
    print(f"  OBC Central Charge:  c = {ent_obc.central_charge:.4f}")
    print(f"  Theoretical Value:   c = 0.5000")
    print(f"  ")
    print(f"  → OBC gives much better central charge extraction!")
    print("=" * 70)
