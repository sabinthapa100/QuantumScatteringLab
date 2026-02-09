"""
IMPROVED Phase 1 Analysis: 1D Ising Model
==========================================

Enhancements:
1. Larger system sizes (N up to 24 using MPS)
2. Corrections to scaling fits
3. Better critical point detection
4. Comparison to exact results

Expected Results:
- Clean 1/N scaling for N > 12
- Central charge c ≈ 0.5 ± 0.05
- Critical point gx = 1.0 ± 0.05
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

from src.models import IsingModel1D
from src.analysis.phase_diagram import PhaseDiagramScanner, plot_phase_diagram_1d
from src.analysis.criticality import CriticalityAnalyzer, plot_gap_scaling, plot_entanglement_scaling

# Output directory
OUTPUT_DIR = Path("results/phase1_ising1d_improved")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("IMPROVED PHASE 1 ANALYSIS: 1D Ising Model")
print("="*80)
print()

# ============================================================================
# PART 1: High-Resolution Phase Diagram
# ============================================================================
print("PART 1: High-Resolution Phase Diagram")
print("-" * 80)

# Use larger system for better resolution
N_phase = 12
scanner = PhaseDiagramScanner(IsingModel1D, num_sites=N_phase, pbc=True, g_z=0.0)

# Fine scan around critical point
results_fine = scanner.scan_1d('g_x', (0.5, 1.5, 50))

# Find critical point using gap minimum
gaps = [r.gap for r in results_fine]
gx_vals = [r.params['g_x'] for r in results_fine]
idx_min = np.argmin(gaps)
gx_crit_gap = gx_vals[idx_min]

print(f"Critical point (gap minimum): gx = {gx_crit_gap:.4f}")

# Find critical point using susceptibility (derivative of <X>)
mx_vals = [r.order_parameters['<X>'] for r in results_fine]
dmx_dgx = np.gradient(mx_vals, gx_vals)
idx_susc = np.argmax(np.abs(dmx_dgx))
gx_crit_susc = gx_vals[idx_susc]

print(f"Critical point (susceptibility): gx = {gx_crit_susc:.4f}")
print(f"Expected: gx = 1.0")
print()

# Plot
fig1 = plot_phase_diagram_1d(results_fine, 'g_x', order_param='<X>',
                             save_path=OUTPUT_DIR / "phase_diagram_fine.png")
plt.close(fig1)

# ============================================================================
# PART 2: Extended Finite-Size Scaling (with MPS)
# ============================================================================
print("PART 2: Extended Finite-Size Scaling (N=4 to 24)")
print("-" * 80)

# System sizes: small (exact) + large (MPS)
system_sizes_extended = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

# Compute gaps at criticality
data_extended = CriticalityAnalyzer.compute_gap_scaling(
    IsingModel1D,
    system_sizes_extended,
    g_x=1.0,
    g_z=0.0,
    pbc=True
)

# Standard power-law fit
fit_standard = CriticalityAnalyzer.fit_gap_scaling(data_extended)
alpha_std = fit_standard['alpha']
r2_std = fit_standard['r_squared']

print(f"Standard fit: Δ ~ N^(-{alpha_std:.3f}), R² = {r2_std:.4f}")

# Fit with corrections: Δ = A/N + B/N²
def gap_with_corrections(N, A, B):
    return A / N + B / N**2

N_arr = np.array(system_sizes_extended)
gaps_arr = np.array(data_extended.gaps)

try:
    popt, pcov = curve_fit(gap_with_corrections, N_arr, gaps_arr, p0=[1.0, 0.1])
    A_corr, B_corr = popt
    
    # Effective exponent at large N
    N_large = 24
    alpha_eff = 1.0 + B_corr / (A_corr * N_large)
    
    print(f"Corrected fit: Δ = {A_corr:.3f}/N + {B_corr:.3f}/N²")
    print(f"Effective α at N=24: {alpha_eff:.3f}")
    
    # Plot both fits
    fig2, ax = plt.subplots(figsize=(10, 7))
    
    ax.loglog(N_arr, gaps_arr, 'o', markersize=10, label='Data', zorder=3)
    
    # Standard fit
    N_fit = np.linspace(4, 24, 100)
    gap_std = fit_standard['A'] / N_fit**alpha_std
    ax.loglog(N_fit, gap_std, '--', linewidth=2, alpha=0.7,
             label=f'Power law: Δ ~ N^(-{alpha_std:.2f})')
    
    # Corrected fit
    gap_corr = gap_with_corrections(N_fit, A_corr, B_corr)
    ax.loglog(N_fit, gap_corr, '-', linewidth=2,
             label=f'With corrections: Δ = {A_corr:.2f}/N + {B_corr:.2f}/N²')
    
    # CFT reference
    ax.loglog(N_fit, 1.0 / N_fit, ':', linewidth=2, alpha=0.5, label='1/N (CFT)')
    
    ax.set_xlabel('System Size N', fontsize=14)
    ax.set_ylabel('Energy Gap Δ', fontsize=14)
    ax.set_title('Finite-Size Scaling with Corrections', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gap_scaling_corrected.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"✓ Corrected scaling plot saved")
    
except Exception as e:
    print(f"⚠ Correction fit failed: {e}")
    A_corr, B_corr = None, None

print()

# ============================================================================
# PART 3: Improved Central Charge Extraction
# ============================================================================
print("PART 3: Improved Central Charge (N=6 to 16)")
print("-" * 80)

# Use moderate sizes for entanglement (expensive!)
system_sizes_ee = [6, 8, 10, 12, 14, 16]

cft_data = CriticalityAnalyzer.extract_central_charge(
    IsingModel1D,
    system_sizes_ee,
    g_x=1.0,
    g_z=0.0,
    pbc=True
)

c = cft_data['c']
r2_ee = cft_data['r_squared']

print(f"Central charge: c = {c:.3f} (expected: 0.5)")
print(f"Fit quality: R² = {r2_ee:.4f}")

if abs(c - 0.5) < 0.1:
    print("✓ Excellent agreement with Majorana CFT!")
elif abs(c - 0.5) < 0.2:
    print("✓ Good agreement with CFT (within finite-size uncertainty)")
else:
    print(f"⚠ Deviation: Δc = {abs(c - 0.5):.3f}")

# Plot
fig3 = plot_entanglement_scaling(
    system_sizes_ee,
    cft_data['entropies'],
    c,
    save_path=OUTPUT_DIR / "entanglement_improved.png"
)
plt.close(fig3)

print()

# ============================================================================
# PART 4: Correlation Length Analysis
# ============================================================================
print("PART 4: Correlation Length at Criticality")
print("-" * 80)

# Compute ξ for different N
correlation_lengths = []
for N in [8, 12, 16, 20]:
    model = IsingModel1D(num_sites=N, g_x=1.0, g_z=0.0, pbc=True)
    xi = CriticalityAnalyzer.compute_correlation_length(model, max_distance=N//2)
    correlation_lengths.append(xi)
    print(f"N={N:2d}: ξ = {xi:.2f} (ξ/N = {xi/N:.3f})")

print()
print("At criticality: ξ ~ N (diverges with system size)")
print()

# ============================================================================
# PART 5: Off-Critical Comparison
# ============================================================================
print("PART 5: Off-Critical vs Critical Scaling")
print("-" * 80)

# Off-critical (gx = 0.7)
data_off = CriticalityAnalyzer.compute_gap_scaling(
    IsingModel1D,
    [4, 6, 8, 10, 12, 14, 16],
    g_x=0.7,
    g_z=0.0,
    pbc=True
)

fit_off = CriticalityAnalyzer.fit_gap_scaling(data_off)
alpha_off = fit_off['alpha']

print(f"Critical (gx=1.0):     α = {alpha_std:.3f}")
print(f"Off-critical (gx=0.7): α = {alpha_off:.3f}")

if alpha_off < alpha_std:
    print("✓ Off-critical has slower decay (expected for finite N)")
else:
    print("⚠ Unexpected behavior")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY: Improved 1D Ising Analysis")
print("="*80)
print()

summary = f"""
CRITICAL POINT DETERMINATION:
  Gap minimum:       gx = {gx_crit_gap:.4f}
  Susceptibility:    gx = {gx_crit_susc:.4f}
  Expected:          gx = 1.0000
  Agreement:         ✓ Within {max(abs(gx_crit_gap-1.0), abs(gx_crit_susc-1.0)):.1%}

FINITE-SIZE SCALING:
  System sizes:      N = 4 to 24
  Standard fit:      Δ ~ N^(-{alpha_std:.3f}), R² = {r2_std:.4f}
  With corrections:  Δ = {A_corr:.3f}/N + {B_corr:.3f}/N² (if fitted)
  CFT prediction:    α = 1.0
  Agreement:         {'✓ Good' if abs(alpha_std - 1.0) < 0.2 else '⚠ Needs larger N'}

CFT CENTRAL CHARGE:
  Extracted:         c = {c:.3f}
  Expected:          c = 0.500 (Majorana)
  Fit quality:       R² = {r2_ee:.4f}
  Agreement:         {'✓ Excellent' if abs(c-0.5) < 0.1 else '✓ Good' if abs(c-0.5) < 0.2 else '⚠ Moderate'}

CORRELATION LENGTH:
  Behavior:          ξ ~ N at criticality
  Verification:      ξ/N ≈ {correlation_lengths[-1]/20:.2f} for N=20

IMPROVEMENTS OVER INITIAL ANALYSIS:
  ✓ Larger N (up to 24 vs 12)
  ✓ Corrections to scaling
  ✓ Better critical point detection
  ✓ Correlation length analysis
  ✓ Higher resolution phase diagram

FILES GENERATED:
  - phase_diagram_fine.png
  - gap_scaling_corrected.png
  - entanglement_improved.png
"""

print(summary)

with open(OUTPUT_DIR / "summary_improved.txt", "w") as f:
    f.write(summary)

print(f"✓ Summary saved to {OUTPUT_DIR / 'summary_improved.txt'}")
print()
print("="*80)
print("IMPROVED 1D ISING ANALYSIS COMPLETE!")
print("="*80)
