"""
Phase 1C: Operator Pool Discovery
==================================

Discover operator pools for all three models and compare to reference pools.
"""

import numpy as np
from pathlib import Path

from src.models import IsingModel1D, IsingModel2D, SU2GaugeModel
from src.analysis.pool_discovery import OperatorPoolDiscovery, visualize_pool_structure

# Output directory
OUTPUT_DIR = Path("results/phase1c_pool_discovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 1C: Operator Pool Discovery")
print("="*80)
print()

# ============================================================================
# PART 1: 1D Ising Model
# ============================================================================
print("PART 1: 1D Ising Model Pool Discovery")
print("-" * 80)

model_1d = IsingModel1D(num_sites=4, g_x=1.0, g_z=0.0, pbc=True)
discovery_1d = OperatorPoolDiscovery(model_1d)

# Discover pool
pool_1d_discovered = discovery_1d.generate_pool(max_order=3, global_only=True, max_pool_size=50)

# Get reference pool
pool_1d_reference = model_1d.build_operator_pool(pool_type='global')

# Compare
comparison_1d = discovery_1d.compare_to_reference(pool_1d_discovered, pool_1d_reference)

print(f"\nDiscovered pool size: {comparison_1d['discovered_size']}")
print(f"Reference pool size:  {comparison_1d['reference_size']}")
print(f"Intersection:         {comparison_1d['intersection_size']}")
print(f"Jaccard similarity:   {comparison_1d['jaccard_similarity']:.3f}")
print(f"Recall:               {comparison_1d['recall']:.3f}")
print(f"Precision:            {comparison_1d['precision']:.3f}")

if comparison_1d['missing_from_discovered']:
    print(f"\nMissing from discovered pool:")
    for op_str in list(comparison_1d['missing_from_discovered'])[:3]:
        print(f"  {op_str}")

if comparison_1d['extra_in_discovered']:
    print(f"\nExtra in discovered pool:")
    for op_str in list(comparison_1d['extra_in_discovered'])[:3]:
        print(f"  {op_str}")

# Visualize
fig_1d = visualize_pool_structure(pool_1d_discovered, model_1d,
                                  save_path=OUTPUT_DIR / "pool_1d_ising.png")

print()

# ============================================================================
# PART 2: 2D Ising Model
# ============================================================================
print("PART 2: 2D Ising Model Pool Discovery")
print("-" * 80)

model_2d = IsingModel2D(Lx=2, Ly=2, g_x=3.04438, g_z=0.0, pbc=True)
discovery_2d = OperatorPoolDiscovery(model_2d)

# Discover pool
pool_2d_discovered = discovery_2d.generate_pool(max_order=2, global_only=True, max_pool_size=30)

# Get reference pool
pool_2d_reference = model_2d.build_operator_pool(pool_type='global')

# Compare
comparison_2d = discovery_2d.compare_to_reference(pool_2d_discovered, pool_2d_reference)

print(f"\nDiscovered pool size: {comparison_2d['discovered_size']}")
print(f"Reference pool size:  {comparison_2d['reference_size']}")
print(f"Jaccard similarity:   {comparison_2d['jaccard_similarity']:.3f}")

# Visualize
fig_2d = visualize_pool_structure(pool_2d_discovered, model_2d,
                                  save_path=OUTPUT_DIR / "pool_2d_ising.png")

print()

# ============================================================================
# PART 3: SU(2) Gauge Model
# ============================================================================
print("PART 3: SU(2) Gauge Model Pool Discovery")
print("-" * 80)

model_su2 = SU2GaugeModel(num_sites=4, g=1.0, a=1.0, mag_case=3, pbc=True)
discovery_su2 = OperatorPoolDiscovery(model_su2)

# Discover pool
pool_su2_discovered = discovery_su2.generate_pool(max_order=2, global_only=True, max_pool_size=30)

# Get reference pool
pool_su2_reference = model_su2.build_operator_pool(pool_type='global')

# Compare
comparison_su2 = discovery_su2.compare_to_reference(pool_su2_discovered, pool_su2_reference)

print(f"\nDiscovered pool size: {comparison_su2['discovered_size']}")
print(f"Reference pool size:  {comparison_su2['reference_size']}")
print(f"Jaccard similarity:   {comparison_su2['jaccard_similarity']:.3f}")

# Visualize
fig_su2 = visualize_pool_structure(pool_su2_discovered, model_su2,
                                   save_path=OUTPUT_DIR / "pool_su2_gauge.png")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("SUMMARY: Pool Discovery Results")
print("="*80)
print()

summary = f"""
1D ISING MODEL:
  Discovered:     {comparison_1d['discovered_size']} operators
  Reference:      {comparison_1d['reference_size']} operators
  Match:          {comparison_1d['jaccard_similarity']:.1%}
  Assessment:     {'✓ Excellent' if comparison_1d['jaccard_similarity'] > 0.8 else '~ Good' if comparison_1d['jaccard_similarity'] > 0.5 else '⚠ Needs improvement'}

2D ISING MODEL:
  Discovered:     {comparison_2d['discovered_size']} operators
  Reference:      {comparison_2d['reference_size']} operators
  Match:          {comparison_2d['jaccard_similarity']:.1%}
  Assessment:     {'✓ Excellent' if comparison_2d['jaccard_similarity'] > 0.8 else '~ Good' if comparison_2d['jaccard_similarity'] > 0.5 else '⚠ Needs improvement'}

SU(2) GAUGE THEORY:
  Discovered:     {comparison_su2['discovered_size']} operators
  Reference:      {comparison_su2['reference_size']} operators
  Match:          {comparison_su2['jaccard_similarity']:.1%}
  Assessment:     {'✓ Excellent' if comparison_su2['jaccard_similarity'] > 0.8 else '~ Good' if comparison_su2['jaccard_similarity'] > 0.5 else '⚠ Needs improvement'}

KEY INSIGHTS:
- Commutator expansion successfully generates operators
- Global symmetry filtering reduces pool size
- Reference pools from papers are well-optimized
- Discovered pools may include additional useful operators

FILES GENERATED:
  - pool_1d_ising.png
  - pool_2d_ising.png
  - pool_su2_gauge.png
"""

print(summary)

with open(OUTPUT_DIR / "pool_discovery_summary.txt", "w") as f:
    f.write(summary)

print(f"✓ Summary saved to {OUTPUT_DIR / 'pool_discovery_summary.txt'}")
print()
print("="*80)
print("POOL DISCOVERY COMPLETE!")
print("="*80)
