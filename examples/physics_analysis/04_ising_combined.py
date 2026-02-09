"""
Phase 1 Analysis: 1D + 2D Ising Models (Combined)
==================================================

Comprehensive analysis of both 1D and 2D Ising models with:
- Extended system sizes
- Critical point verification
- Finite-size scaling
- Comparison between 1D and 2D physics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.linalg import eigh

from src.models import IsingModel1D, IsingModel2D

# Output directory
OUTPUT_DIR = Path("results/phase1_ising_combined")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 1 ANALYSIS: 1D + 2D Ising Models (Combined)")
print("="*80)
print()

# ============================================================================
# PART 1: 1D Ising - Extended Finite-Size Scaling
# ============================================================================
print("PART 1: 1D Ising - Extended Finite-Size Scaling")
print("-" * 80)

system_sizes_1d = [4, 6, 8, 10, 12, 14, 16, 18, 20]
gaps_1d = []
energies_1d = []

print("Computing gaps for N = 4 to 20...")
for N in system_sizes_1d:
    model = IsingModel1D(num_sites=N, g_x=1.0, g_z=0.0, pbc=True)
    H = model.build_hamiltonian()
    mat = H.to_matrix()
    eigs = np.linalg.eigvalsh(mat)
    
    gap = eigs[1] - eigs[0]
    E0 = eigs[0] / N
    
    gaps_1d.append(gap)
    energies_1d.append(E0)
    
    print(f"  N={N:2d}: Gap = {gap:.6f}, E0/N = {E0:.6f}")

# Fit gap ~ A/N
log_N = np.log(system_sizes_1d)
log_gap = np.log(gaps_1d)
coeffs = np.polyfit(log_N, log_gap, 1)
alpha_1d = -coeffs[0]
A_1d = np.exp(coeffs[1])

print(f"\nGap scaling: Δ ~ {A_1d:.3f} / N^{alpha_1d:.3f}")
print(f"Expected: α = 1.0 (CFT)")
print()

# ============================================================================
# PART 2: 2D Ising - Finite-Size Scaling
# ============================================================================
print("PART 2: 2D Ising - Finite-Size Scaling")
print("-" * 80)

lattice_sizes_2d = [2, 3, 4]  # L×L lattices
gaps_2d = []
energies_2d = []
system_sizes_2d = []

print("Computing gaps for L×L lattices...")
for L in lattice_sizes_2d:
    N = L * L
    model = IsingModel2D(Lx=L, Ly=L, g_x=3.04438, g_z=0.0, pbc=True)
    H = model.build_hamiltonian()
    mat = H.to_matrix()
    eigs = np.linalg.eigvalsh(mat)
    
    gap = eigs[1] - eigs[0]
    E0 = eigs[0] / N
    
    gaps_2d.append(gap)
    energies_2d.append(E0)
    system_sizes_2d.append(N)
    
    print(f"  L={L} (N={N:2d}): Gap = {gap:.6f}, E0/N = {E0:.6f}")

# Fit gap ~ A/N^α
log_N_2d = np.log(system_sizes_2d)
log_gap_2d = np.log(gaps_2d)
coeffs_2d = np.polyfit(log_N_2d, log_gap_2d, 1)
alpha_2d = -coeffs_2d[0]
A_2d = np.exp(coeffs_2d[1])

print(f"\nGap scaling: Δ ~ {A_2d:.3f} / N^{alpha_2d:.3f}")
print(f"Note: 2D has stronger finite-size effects")
print()

# ============================================================================
# PART 3: Critical Point Detection (1D)
# ============================================================================
print("PART 3: Critical Point Detection (1D)")
print("-" * 80)

gx_vals = np.linspace(0.5, 1.5, 40)
gaps_scan = []

print("Scanning gx from 0.5 to 1.5...")
for gx in gx_vals:
    model = IsingModel1D(num_sites=12, g_x=gx, g_z=0.0, pbc=True)
    H = model.build_hamiltonian()
    mat = H.to_matrix()
    eigs = np.linalg.eigvalsh(mat)
    gap = eigs[1] - eigs[0]
    gaps_scan.append(gap)

idx_min = np.argmin(gaps_scan)
gx_crit = gx_vals[idx_min]

print(f"Critical point (gap minimum): gx = {gx_crit:.4f}")
print(f"Expected: gx = 1.0")
print(f"Finite-size shift: Δgx = {abs(gx_crit - 1.0):.4f}")
print()

# ============================================================================
# PART 4: Critical Point Detection (2D)
# ============================================================================
print("PART 4: Critical Point Detection (2D)")
print("-" * 80)

gx_vals_2d = np.linspace(2.0, 4.0, 30)
gaps_scan_2d = []

print("Scanning gx from 2.0 to 4.0...")
for gx in gx_vals_2d:
    model = IsingModel2D(Lx=3, Ly=3, g_x=gx, g_z=0.0, pbc=True)
    H = model.build_hamiltonian()
    mat = H.to_matrix()
    eigs = np.linalg.eigvalsh(mat)
    gap = eigs[1] - eigs[0]
    gaps_scan_2d.append(gap)

idx_min_2d = np.argmin(gaps_scan_2d)
gx_crit_2d = gx_vals_2d[idx_min_2d]

print(f"Critical point (gap minimum): gx = {gx_crit_2d:.4f}")
print(f"Expected: gx = 3.04438")
print(f"Finite-size shift: Δgx = {abs(gx_crit_2d - 3.04438):.4f}")
print()

# ============================================================================
# PART 5: Plotting
# ============================================================================
print("PART 5: Generating Plots")
print("-" * 80)

# Plot 1: Gap Scaling Comparison
fig1, ax1 = plt.subplots(figsize=(10, 7))

ax1.loglog(system_sizes_1d, gaps_1d, 'o-', markersize=8, linewidth=2, label='1D Ising')
ax1.loglog(system_sizes_2d, gaps_2d, 's-', markersize=8, linewidth=2, label='2D Ising')

# Reference lines
N_ref = np.linspace(4, 20, 100)
ax1.loglog(N_ref, 1.0 / N_ref, ':', linewidth=2, alpha=0.5, label='1/N (CFT)')

ax1.set_xlabel('System Size N', fontsize=14)
ax1.set_ylabel('Energy Gap Δ', fontsize=14)
ax1.set_title('Finite-Size Scaling: 1D vs 2D Ising', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gap_scaling_comparison.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

print("✓ Gap scaling comparison saved")

# Plot 2: Critical Point Scans
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

# 1D
ax2a.plot(gx_vals, gaps_scan, 'o-', linewidth=2, markersize=4)
ax2a.axvline(1.0, color='r', linestyle='--', label='Expected (gx=1.0)')
ax2a.axvline(gx_crit, color='g', linestyle=':', label=f'Observed (gx={gx_crit:.3f})')
ax2a.set_xlabel('gx', fontsize=14)
ax2a.set_ylabel('Energy Gap', fontsize=14)
ax2a.set_title('1D Ising Critical Point (N=12)', fontsize=14)
ax2a.legend()
ax2a.grid(True, alpha=0.3)

# 2D
ax2b.plot(gx_vals_2d, gaps_scan_2d, 'o-', linewidth=2, markersize=4)
ax2b.axvline(3.04438, color='r', linestyle='--', label='Expected (gx=3.044)')
ax2b.axvline(gx_crit_2d, color='g', linestyle=':', label=f'Observed (gx={gx_crit_2d:.3f})')
ax2b.set_xlabel('gx', fontsize=14)
ax2b.set_ylabel('Energy Gap', fontsize=14)
ax2b.set_title('2D Ising Critical Point (3×3)', fontsize=14)
ax2b.legend()
ax2b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "critical_points.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

print("✓ Critical point scans saved")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY: 1D + 2D Ising Analysis")
print("="*80)
print()

summary = f"""
1D ISING MODEL:
  System sizes:      N = 4 to 20
  Gap scaling:       Δ ~ N^(-{alpha_1d:.3f})
  Expected:          α = 1.0 (CFT)
  Critical point:    gx = {gx_crit:.4f} (expected: 1.0)
  Universality:      c = 1/2 Majorana CFT

2D ISING MODEL:
  Lattice sizes:     L = 2, 3, 4 (N = 4, 9, 16)
  Gap scaling:       Δ ~ N^(-{alpha_2d:.3f})
  Critical point:    gx = {gx_crit_2d:.4f} (expected: 3.044)
  Universality:      3D Ising class

KEY DIFFERENCES:
  - 1D has logarithmic entanglement (volume law)
  - 2D has area-law entanglement
  - 2D has stronger finite-size effects
  - Different critical exponents (same c though!)

FILES GENERATED:
  - gap_scaling_comparison.png
  - critical_points.png
"""

print(summary)

with open(OUTPUT_DIR / "summary_combined.txt", "w") as f:
    f.write(summary)

print(f"✓ Summary saved to {OUTPUT_DIR / 'summary_combined.txt'}")
print()
print("="*80)
print("COMBINED 1D + 2D ISING ANALYSIS COMPLETE!")
print("="*80)
