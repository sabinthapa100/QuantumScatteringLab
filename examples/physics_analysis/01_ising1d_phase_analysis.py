"""
Phase 1 Analysis: 1D Ising Model
=================================

Comprehensive physics analysis:
1. Phase diagram (gx, gz) parameter space
2. Critical point verification (gx=1, gz=0)
3. Finite-size scaling (gap ~ 1/N)
4. Central charge extraction (c = 1/2 for Majorana CFT)
5. Order parameters and phase transitions

Expected Results:
- Ferromagnetic phase: gx >> 1, ⟨X⟩ → 1
- Paramagnetic phase: gx << 1, degeneracy
- Critical point: gx = 1, gz = 0
- CFT: c = 0.5 (Majorana fermion)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import IsingModel1D
from src.analysis.phase_diagram import (
    PhaseDiagramScanner, 
    plot_phase_diagram_1d,
    plot_phase_diagram_2d
)
from src.analysis.criticality import (
    CriticalityAnalyzer,
    plot_gap_scaling,
    plot_entanglement_scaling
)

# Output directory
OUTPUT_DIR = Path("results/phase1_ising1d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PHASE 1 ANALYSIS: 1D Transverse-Field Ising Model")
print("="*70)
print()

# ============================================================================
# PART 1: Phase Diagram - Scan gx at gz=0
# ============================================================================
print("PART 1: Phase Diagram (gx scan, gz=0)")
print("-" * 70)

N = 8  # System size
scanner = PhaseDiagramScanner(IsingModel1D, num_sites=N, pbc=True, g_z=0.0)

# Scan gx from 0.1 to 3.0
results_1d = scanner.scan_1d('g_x', (0.1, 3.0, 30))

# Find critical point (where ⟨X⟩ ≈ 0.5)
critical_indices = scanner.find_critical_points(results_1d, order_param='<X>', threshold=0.5)
if critical_indices:
    idx = critical_indices[0]
    gx_crit = results_1d[idx].params['g_x']
    print(f"✓ Critical point found: gx ≈ {gx_crit:.3f} (expected: 1.0)")
else:
    print("⚠ No critical point found in scan range")

# Plot
fig1 = plot_phase_diagram_1d(results_1d, 'g_x', order_param='<X>', 
                             save_path=OUTPUT_DIR / "phase_diagram_gx.png")
plt.close(fig1)

print(f"✓ Phase diagram saved to {OUTPUT_DIR / 'phase_diagram_gx.png'}")
print()

# ============================================================================
# PART 2: 2D Phase Diagram - Scan (gx, gz)
# ============================================================================
print("PART 2: 2D Phase Diagram (gx, gz)")
print("-" * 70)

N_2d = 6  # Smaller for 2D scan (computational cost)
scanner_2d = PhaseDiagramScanner(IsingModel1D, num_sites=N_2d, pbc=True)

# Scan (gx, gz) space
results_2d = scanner_2d.scan_2d(
    'g_x', (0.2, 2.0, 15),
    'g_z', (0.0, 1.0, 10)
)

# Plot
fig2 = plot_phase_diagram_2d(results_2d, 'g_x', 'g_z', order_param='<X>',
                             save_path=OUTPUT_DIR / "phase_diagram_2d.png")
plt.close(fig2)

print(f"✓ 2D phase diagram saved to {OUTPUT_DIR / 'phase_diagram_2d.png'}")
print()

# ============================================================================
# PART 3: Finite-Size Scaling at Criticality
# ============================================================================
print("PART 3: Finite-Size Scaling (gx=1.0, gz=0.0)")
print("-" * 70)

# Compute gaps for N = 4, 6, 8, 10, 12
system_sizes = [4, 6, 8, 10, 12]
data_critical = CriticalityAnalyzer.compute_gap_scaling(
    IsingModel1D, 
    system_sizes,
    g_x=1.0,
    g_z=0.0,
    pbc=True
)

# Fit gap ~ A / N^α
fit_params = CriticalityAnalyzer.fit_gap_scaling(data_critical)
alpha = fit_params['alpha']
r2 = fit_params['r_squared']

print(f"Gap scaling: Δ ~ N^(-{alpha:.3f})")
print(f"Expected: α = 1.0 (CFT)")
print(f"Fit quality: R² = {r2:.4f}")

if abs(alpha - 1.0) < 0.1:
    print("✓ Consistent with CFT scaling!")
else:
    print(f"⚠ Deviation from CFT: Δα = {abs(alpha - 1.0):.3f}")

# Plot
fig3 = plot_gap_scaling(data_critical, fit_params,
                       save_path=OUTPUT_DIR / "gap_scaling.png")
plt.close(fig3)

print(f"✓ Gap scaling plot saved to {OUTPUT_DIR / 'gap_scaling.png'}")
print()

# ============================================================================
# PART 4: Central Charge Extraction
# ============================================================================
print("PART 4: Central Charge Extraction")
print("-" * 70)

# Compute entanglement entropy for N = 4, 6, 8, 10
system_sizes_ee = [4, 6, 8, 10]  # Smaller range (expensive)
cft_data = CriticalityAnalyzer.extract_central_charge(
    IsingModel1D,
    system_sizes_ee,
    g_x=1.0,
    g_z=0.0,
    pbc=True
)

c = cft_data['c']
r2_ee = cft_data['r_squared']

print(f"Central charge: c = {c:.3f}")
print(f"Expected: c = 0.5 (Majorana CFT)")
print(f"Fit quality: R² = {r2_ee:.4f}")

if abs(c - 0.5) < 0.15:
    print("✓ Consistent with Majorana CFT!")
else:
    print(f"⚠ Deviation from c=0.5: Δc = {abs(c - 0.5):.3f}")

# Plot
fig4 = plot_entanglement_scaling(
    system_sizes_ee,
    cft_data['entropies'],
    c,
    save_path=OUTPUT_DIR / "entanglement_scaling.png"
)
plt.close(fig4)

print(f"✓ Entanglement scaling plot saved to {OUTPUT_DIR / 'entanglement_scaling.png'}")
print()

# ============================================================================
# PART 5: Off-Critical Scaling (Comparison)
# ============================================================================
print("PART 5: Off-Critical Scaling (gx=0.5)")
print("-" * 70)

data_off_critical = CriticalityAnalyzer.compute_gap_scaling(
    IsingModel1D,
    system_sizes,
    g_x=0.5,
    g_z=0.0,
    pbc=True
)

fit_off = CriticalityAnalyzer.fit_gap_scaling(data_off_critical)
alpha_off = fit_off['alpha']

print(f"Off-critical scaling: Δ ~ N^(-{alpha_off:.3f})")
print(f"Critical scaling:     Δ ~ N^(-{alpha:.3f})")

if alpha_off < alpha:
    print("✓ Off-critical gap decays slower (expected)")
else:
    print("⚠ Unexpected: off-critical should have α < α_critical")

print()

# ============================================================================
# PART 6: Summary Table
# ============================================================================
print("="*70)
print("SUMMARY: 1D Ising Model Physics")
print("="*70)
print()

summary = f"""
Critical Point:
  Location:        gx = {gx_crit:.3f} ± 0.1 (expected: 1.0)
  Gap at N=8:      {results_1d[critical_indices[0]].gap:.4f}
  
Finite-Size Scaling:
  Exponent α:      {alpha:.3f} (expected: 1.0)
  Fit quality:     R² = {r2:.4f}
  
CFT Data:
  Central charge:  c = {c:.3f} (expected: 0.5)
  Fit quality:     R² = {r2_ee:.4f}
  
Phase Identification:
  gx < 1:  Paramagnetic (⟨Z⟩ ordered, degenerate)
  gx = 1:  Critical (CFT, c=1/2)
  gx > 1:  Ferromagnetic (⟨X⟩ ordered)

Files Generated:
  - phase_diagram_gx.png       (1D scan)
  - phase_diagram_2d.png       (2D heatmap)
  - gap_scaling.png            (finite-size scaling)
  - entanglement_scaling.png   (CFT verification)
"""

print(summary)

# Save summary
with open(OUTPUT_DIR / "summary.txt", "w") as f:
    f.write(summary)

print(f"✓ Summary saved to {OUTPUT_DIR / 'summary.txt'}")
print()
print("="*70)
print("PHASE 1 ANALYSIS COMPLETE!")
print("="*70)
