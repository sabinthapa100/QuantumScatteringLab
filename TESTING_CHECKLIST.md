# Phase 0 Testing Checklist
## Systematic Model Validation

**Author**: Sabin Thapa  
**Date**: 2026-02-09  
**Status**: IN PROGRESS

---

## ✅ Completed Tests

### 1D Ising Model
- [x] L=8 spectrum analysis (g_x: 0.5 → 1.5)
- [x] L=10 spectrum analysis  
- [x] Energy per site convergence
- [x] Entanglement entropy measurement
- [ ] L=12 validation (TODO)
- [ ] L=14 validation (TODO)

**Key Findings**:
- Energy spectrum computed successfully via DMRG  
- Critical behavior observed near g_x ~ 1.0 (expected)  
- Entanglement scaling: S_vN ∝ log(L) near critical point

**Outputs**:
- `outputs/phase0_validation/spectrum/ising_1d_spectrum.png`
- `outputs/phase0_validation/data/ising_1d_analysis.json`

---

## 🔄 Next Tests (Phase 0 Cont.)

### 2D Ising Model
- [ ] Test perfect squares: L=16 (4x4), L=25 (5x5), L=36 (6x6)
- [ ] Validate snake-chain mapping
- [ ] Compare with known 2D Ising critical point

### SU(2) Gauge Theory
- [ ] L=8, 10, 12 spectrum analysis
- [ ] Validate coupling J=-3g²/16, h_z=3g²/8
- [ ] Check lattice spacing dependence (a)

---

## 📊 Validation Criteria

Before moving to Phase 1 (ADAPT-VQE), we need:

1. **Convergence**: Energy/L converges with system size
2. **Critical Points**: Match known values (g_c ≈ 1.0 for Ising)
3. **Entanglement**: Logarithmic scaling near criticality
4. **Cross-validation**: DMRG vs Exact Diag for L ≤ 12

---

## 🚀 Phase 1 Readiness Checklist

- [ ] All 3 models validated (1D Ising, 2D Ising, SU(2))
- [ ] Phase transition points documented
- [ ] DMRG accuracy benchmarked against exact diag
- [ ] Plots generated and saved
- [ ] Results committed to git

---

## Commands to Run

```bash
# 1D Ising - Full validation
python scripts/phase0_validate_models.py --model ising_1d --sizes 8,10,12,14 --coupling_range 0.2,1.8,17

# 2D Ising - Perfect squares only
python scripts/phase0_validate_models.py --model ising_2d --sizes 16,25,36 --coupling_range 0.2,1.8,17

# SU(2) Gauge
python scripts/phase0_validate_models.py --model su2_gauge --sizes 8,10,12 --coupling_range 0.5,2.0,16
```

---

## Git Workflow

After each successful validation:
```bash
git add outputs/phase0_validation/
git commit -m "test(phase0): Validate [MODEL] for L=[SIZES]"
git push origin main
```

**Note**: Large output files (`.npy`, `.npz`) are gitignored. Only JSON data and plots are tracked.
