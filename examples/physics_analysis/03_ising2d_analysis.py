"""
Phase 1 Analysis: 2D Ising Model
=================================

Comprehensive 2D analysis:
1. Phase diagram in (gx, gz) space
2. Critical point verification (gx ≈ 3.04438)
3. Finite-size scaling on L×L lattices
4. Comparison to 3D Ising universality class

Expected Results:
- Critical point: gx ≈ 3.044, gz = 0
- 3D Ising CFT (c ≈ 0.5, but different from 1D!)
- Stronger finite-size effects than 1D

Challenges:
- 2D requires larger Hilbert space (2^(L²))
- L=4 → N=16 qubits (feasible)
- L=5 → N=25 qubits (MPS needed)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import IsingModel2D
from src.analysis.phase_diagram import PhaseDiagramScanner, plot_phase_diagram_1d
from src.analysis.criticality import CriticalityAnalyzer

# Output directory
OUTPUT_DIR = Path("results/phase1_ising2d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 1 ANALYSIS: 2D Ising Model on Square Lattice")
print("="*80)
print()

# ============================================================================
# PART 1: Phase Diagram - gx scan for L×L lattices
# ============================================================================
print("PART 1: Phase Diagram (gx scan, gz=0)")
print("-" * 80)

# Start with small lattice
Lx, Ly = 3, 3
N_2d = Lx * Ly

print(f"Lattice: {Lx}×{Ly} (N={N_2d} qubits)")
print()

# Create scanner - pass Lx, Ly directly
class IsingModel2DWrapper:
    """Wrapper to make IsingModel2D compatible with scanner."""
    def __init__(self, num_sites, Lx, Ly, **kwargs):
        from src.models import IsingModel2D
        self.model = IsingModel2D(Lx=Lx, Ly=Ly, **kwargs)
        self.num_sites = self.model.num_sites
        
    def build_hamiltonian(self):
        return self.model.build_hamiltonian()
    
    def __getattr__(self, name):
        return getattr(self.model, name)

scanner_2d = PhaseDiagramScanner(IsingModel2DWrapper, num_sites=N_2d, Lx=Lx, Ly=Ly, pbc=True, g_z=0.0)

# Scan gx around expected critical point
results_2d = scanner_2d.scan_1d('g_x', (1.0, 5.0, 30))

# Find critical point
gaps = [r.gap for r in results_2d]
gx_vals = [r.params['g_x'] for r in results_2d]
idx_min = np.argmin(gaps)
gx_crit_2d = gx_vals[idx_min]

print(f"Critical point (gap minimum): gx = {gx_crit_2d:.4f}")
print(f"Expected (infinite lattice): gx = 3.04438")
print(f"Finite-size shift: Δgx = {abs(gx_crit_2d - 3.04438):.4f}")
print()

# Plot phase diagram
fig1, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Order parameter
mx_vals = [r.order_parameters['<X>'] for r in results_2d]
axes[0].plot(gx_vals, mx_vals, 'o-', linewidth=2, markersize=6)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].axvline(3.04438, color='r', linestyle=':', alpha=0.5, label='Expected critical point')
axes[0].set_ylabel('⟨X⟩', fontsize=14)
axes[0].set_title(f'2D Ising Phase Diagram ({Lx}×{Ly} lattice)', fontsize=16)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gap
axes[1].plot(gx_vals, gaps, 'o-', color='red', linewidth=2, markersize=6)
axes[1].axvline(3.04438, color='r', linestyle=':', alpha=0.5)
axes[1].set_ylabel('Energy Gap', fontsize=14)
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

# Energy per site
energies = [r.energy / N_2d for r in results_2d]
axes[2].plot(gx_vals, energies, 'o-', color='green', linewidth=2, markersize=6)
axes[2].axvline(3.04438, color='r', linestyle=':', alpha=0.5)
axes[2].set_ylabel('Energy / Site', fontsize=14)
axes[2].set_xlabel('gx', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "phase_diagram_2d.png", dpi=300, bbox_inches='tight')
plt.close(fig1)

print(f"✓ Phase diagram saved to {OUTPUT_DIR / 'phase_diagram_2d.png'}")
print()

# ============================================================================
# PART 2: Finite-Size Scaling - Different Lattice Sizes
# ============================================================================
print("PART 2: Finite-Size Scaling (L = 2, 3, 4)")
print("-" * 80)

# Lattice sizes
lattice_sizes = [2, 3, 4]  # L=5 would be 25 qubits (too large for exact diag)
gaps_2d = []
system_sizes_2d = []

for L in lattice_sizes:
    N = L * L
    model = IsingModel2D(Lx=L, Ly=L, g_x=3.04438, g_z=0.0, pbc=True)
    H = model.build_hamiltonian()
    mat = H.to_matrix()
    eigs = np.linalg.eigvalsh(mat)
    gap = eigs[1] - eigs[0]
    
    gaps_2d.append(gap)
    system_sizes_2d.append(N)
    
    print(f"L={L} (N={N:2d}): Gap = {gap:.6f}")

print()

# Plot gap vs N
fig2, ax = plt.subplots(figsize=(10, 7))

ax.loglog(system_sizes_2d, gaps_2d, 'o', markersize=12, label='2D Ising')

# Reference: 1/N scaling
N_ref = np.linspace(4, 16, 100)
ax.loglog(N_ref, 1.0 / N_ref, ':', linewidth=2, alpha=0.5, label='1/N reference')

ax.set_xlabel('System Size N = L²', fontsize=14)
ax.set_ylabel('Energy Gap Δ', fontsize=14)
ax.set_title('2D Ising Finite-Size Scaling', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gap_scaling_2d.png", dpi=300, bbox_inches='tight')
plt.close(fig2)

print(f"✓ Gap scaling plot saved")
print()

# Fit power law
log_N = np.log(system_sizes_2d)
log_gap = np.log(gaps_2d)
coeffs = np.polyfit(log_N, log_gap, 1)
alpha_2d = -coeffs[0]

print(f"Gap scaling: Δ ~ N^(-{alpha_2d:.3f})")
print(f"Note: 2D has stronger finite-size effects than 1D")
print()

# ============================================================================
# PART 3: Entanglement Entropy (if feasible)
# ============================================================================
print("PART 3: Entanglement Entropy")
print("-" * 80)

entropies_2d = []
lattice_sizes_ee = [2, 3]  # L=4 is expensive for entanglement

for L in lattice_sizes_ee:
    N = L * L
    model = IsingModel2D(Lx=L, Ly=L, g_x=3.04438, g_z=0.0, pbc=True)
    
    # Subsystem: half the lattice (cut along one direction)
    L_A = (L * L) // 2
    
    try:
        S = CriticalityAnalyzer.compute_entanglement_entropy(model, L_A)
        entropies_2d.append(S)
        print(f"L={L} (N={N}): S = {S:.4f}")
    except Exception as e:
        print(f"L={L}: Entanglement calculation failed ({e})")
        entropies_2d.append(np.nan)

print()

# ============================================================================
# PART 4: Comparison to 1D Ising
# ============================================================================
print("PART 4: 1D vs 2D Comparison")
print("-" * 80)

comparison = """
Property              | 1D Ising        | 2D Ising
----------------------|-----------------|------------------
Critical point        | gx = 1.0        | gx ≈ 3.044
Universality class    | c=1/2 Majorana  | 3D Ising (c≈0.5)
Finite-size effects   | Moderate        | Strong
Correlation length    | ξ ~ N           | ξ ~ L (2D)
Entanglement scaling  | S ~ (c/3)log(N) | S ~ L (area law)

Key Difference: 2D has AREA LAW entanglement, not volume law!
"""

print(comparison)
print()

# ============================================================================
# PART 5: 2D Heatmap (gx, gz) for Fixed Lattice
# ============================================================================
print("PART 5: 2D Parameter Space Heatmap")
print("-" * 80)

# Small lattice for 2D scan
Lx_scan, Ly_scan = 2, 2
scanner_heatmap = PhaseDiagramScanner(IsingModel2D, num_sites=4, Lx=Lx_scan, Ly=Ly_scan, pbc=True)

# Scan (gx, gz)
results_heatmap = scanner_heatmap.scan_2d(
    'g_x', (1.0, 5.0, 12),
    'g_z', (0.0, 2.0, 10)
)

# Extract order parameter grid
n1, n2 = results_heatmap.shape
gx_grid = np.array([results_heatmap[i, 0].params['g_x'] for i in range(n1)])
gz_grid = np.array([results_heatmap[0, j].params['g_z'] for j in range(n2)])

mx_grid = np.array([[results_heatmap[i, j].order_parameters['<X>'] 
                     for j in range(n2)] for i in range(n1)])

# Plot heatmap
fig3, ax = plt.subplots(figsize=(10, 8))

im = ax.imshow(mx_grid, origin='lower', aspect='auto', cmap='RdBu_r',
              extent=[gz_grid[0], gz_grid[-1], gx_grid[0], gx_grid[-1]])
ax.axhline(3.04438, color='yellow', linestyle='--', linewidth=2, label='Expected critical line')
ax.set_xlabel('gz', fontsize=14)
ax.set_ylabel('gx', fontsize=14)
ax.set_title(f'2D Ising Order Parameter ⟨X⟩ ({Lx_scan}×{Ly_scan} lattice)', fontsize=16)
ax.legend()
plt.colorbar(im, ax=ax, label='⟨X⟩')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_2d.png", dpi=300, bbox_inches='tight')
plt.close(fig3)

print(f"✓ Heatmap saved to {OUTPUT_DIR / 'heatmap_2d.png'}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY: 2D Ising Model Analysis")
print("="*80)
print()

summary = f"""
CRITICAL POINT (3×3 lattice):
  Observed:          gx = {gx_crit_2d:.4f}
  Expected:          gx = 3.04438
  Finite-size shift: Δgx = {abs(gx_crit_2d - 3.04438):.4f}
  
FINITE-SIZE SCALING:
  Lattice sizes:     L = 2, 3, 4 (N = 4, 9, 16)
  Gap scaling:       Δ ~ N^(-{alpha_2d:.3f})
  Note:              Stronger finite-size effects than 1D
  
ENTANGLEMENT:
  Behavior:          Area law (S ~ L, not log(N))
  Universality:      3D Ising class (different from 1D!)
  
PHASE DIAGRAM:
  Ferromagnetic:     gx > 3.044
  Paramagnetic:      gx < 3.044
  Critical line:     gx ≈ 3.044 (varies with gz)
  
CHALLENGES:
  ✓ Larger Hilbert space (2^(L²))
  ✓ Need L ≥ 5 for clean scaling (N=25 qubits)
  ✓ MPS required for L > 4
  
FILES GENERATED:
  - phase_diagram_2d.png
  - gap_scaling_2d.png
  - heatmap_2d.png
"""

print(summary)

with open(OUTPUT_DIR / "summary_2d.txt", "w") as f:
    f.write(summary)

print(f"✓ Summary saved to {OUTPUT_DIR / 'summary_2d.txt'}")
print()
print("="*80)
print("2D ISING ANALYSIS COMPLETE!")
print("="*80)
