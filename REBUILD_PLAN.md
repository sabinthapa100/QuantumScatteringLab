# QUANTUM SCATTERING LAB: SYSTEMATIC REBUILD PLAN

**Date**: 2026-02-09  
**Status**: Refactoring from GUI-first to Physics-first approach

---

## CURRENT STATE ASSESSMENT

### ✅ **What's Working**
1. **Model Definitions** (`src/models/`)
   - `IsingModel1D`: Parameters: `num_sites`, `g_x`, `g_z`, `pbc`
   - `IsingModel2D`: Parameters: `Lx`, `Ly`, `g_x`, `g_z`, `pbc`
   - `SU2GaugeModel`: Parameters: `num_sites`, `g` (gauge coupling), `a` (lattice spacing), `pbc`
   - All models correctly implement `build_hamiltonian()`, `get_trotter_layers()`, `build_operator_pool()`

2. **MPS Backend** (`src/backends/quimb_mps_backend.py`)
   - DMRG ground state solver implemented
   - MPS time evolution via Trotterization
   - Expectation value computation

3. **Direct MPS Wavepacket Initialization** (`src/simulation/initialization.py`)
   - `prepare_two_wavepacket_mps()` - bypasses 2^N memory limit
   - `prepare_three_wavepacket_mps()` - 3-particle states

4. **Test Suite** (✅ All passing):
   - Model integrity tests
   - MPS vs dense backend comparison
   - Time evolution accuracy

---

### ❌ **What's Broken / Missing**

#### **1. GUI Parameter Mapping (CRITICAL BUG)**
**Problem**: 
- The GUI mixes parameters across models
- Added a meaningless `mass` parameter that gets added to `g_z`
- LaTeX doesn't update when model changes
- Parameter labels are wrong ("Quark Mass" for Ising models)

**Correct Parameter Sets per Model**:
```python
# 1D Ising
{
    "num_sites": int,
    "g_x": float,      # Transverse field
    "g_z": float,      # Longitudinal field
    "pbc": bool
}

# 2D Ising
{
    "Lx": int,
    "Ly": int,
    "g_x": float,
    "g_z": float,
    "pbc": bool
}

# SU(2) Gauge
{
    "num_sites": int,
    "g": float,        # Gauge coupling
    "a": float,        # Lattice spacing
    "pbc": bool
}
```

**Fix Required**:
- Remove `mass` parameter entirely
- Implement model-specific parameter forms in GUI
- Make LaTeX dynamic based on selected model
- Update `dashboard/server.py` to handle model-specific params

---

#### **2. Missing Physics Analysis Tools**

**NOT YET IMPLEMENTED**:
- [ ] Energy spectrum calculation (exact diagonalization for small L)
- [ ] Phase transition detection (critical point at g_x = 1 for Ising)
- [ ] Ground state entanglement entropy
- [ ] Correlation functions
- [ ] S-matrix extraction from scattering data

---

#### **3. ADAPT-VQE Not Implemented**
- The arxiv paper in `docs/` has ADAPT-VQE results we need to reproduce
- Need to implement:
  - ADAPT-VQE algorithm
  - Gradient-based operator selection
  - Comparison with VQE
  - Publication plots

---

#### **4. Trotterization Order Comparison Missing**
- Currently only 1st-order Trotter
- Need: 2nd-order, 4th-order comparisons
- Error analysis vs exact evolution
- Time step convergence studies

---

## PHASE-BY-PHASE REBUILD PLAN

### **PHASE 0: MODEL VALIDATION & ANALYSIS** (Do First!)

**Goal**: Verify physics before building GUI

#### 0.1: Spectrum Analysis Scripts
Create `scripts/analyze_spectrum.py`:
```python
# For each model, compute:
# 1. Energy eigenvalues vs coupling parameter
# 2. Level spacing statistics
# 3. Phase transition markers (gap closing)
# 4. Save plots to outputs/spectrum/
```

**Deliverables**:
- `outputs/spectrum/ising_1d_spectrum.png`
- `outputs/spectrum/ising_2d_spectrum.png`
- `outputs/spectrum/su2_spectrum.png`

#### 0.2: Ground State Characterization
Create `scripts/groundstate_analysis.py`:
```python
# For each model:
# 1. DMRG ground state
# 2. Entanglement entropy profile
# 3. Correlation functions
# 4. Compare with exact diag (small L)
```

#### 0.3: Critical Point Detection
For Ising models:
- Sweep g_x from 0 to 2
- Plot energy gap, entropy, magnetization
- Verify critical point at g_x ≈ 1.0

---

### **PHASE 1: ADAPT-VQE IMPLEMENTATION**

**Goal**: Reproduce arxiv paper figures

#### 1.1: Core Algorithm
Create `src/vqe/adaptvqe.py`:
- Gradient calculation
- Operator selection from pool
- Parameter optimization
- Convergence criteria

#### 1.2: Benchmarking
- Compare ADAPT-VQE vs standard VQE
- Plot ansatz depth vs accuracy
- Generate paper Fig. 3, Fig. 4 equivalents

---

### **PHASE 2: SCATTERING SIMULATIONS**

**Goal**: Validate time evolution

#### 2.1: Trotterization Order Study
Create `scripts/trotter_convergence.py`:
- 1st, 2nd, 4th order comparison
- Energy conservation plots
- Time step convergence

#### 2.2: Wavepacket Collision Analysis
- Two-particle scattering
- S-matrix extraction (reflection/transmission)
- Phase shift calculation

#### 2.3: Three-Particle Dynamics
- Use `prepare_three_wavepacket_mps()`
- Study bound state formation
- Multi-particle interference

---

### **PHASE 3: GUI REBUILD** (LAST!)

**Only after physics is validated**

#### 3.1: Fix Parameter Handling
- Remove `mass` parameter
- Model-specific parameter forms
- Dynamic LaTeX rendering

#### 3.2: Add Analysis Panels
- Live spectrum display
- Entanglement entropy evolution
- Phase space visualization

#### 3.3: Interactive Phase Diagrams
- Clickable phase diagrams
- Hover-to-see parameter values
- Export data for publications

---

## IMMEDIATE ACTION ITEMS (Next Steps)

### 🔧 **Critical Fixes (Do Now)**
1. Fix GUI parameter mapping in `dashboard/server.py`
2. Fix LaTeX rendering in `dashboard/src/App.jsx`
3. Remove `mass` parameter completely

### 📊 **Week 1: Spectrum Analysis**
1. Create `scripts/analyze_spectrum.py`
2. Run for all three models
3. Generate plots in `outputs/spectrum/`
4. Verify critical points

### 🧮 **Week 2: ADAPT-VQE**
1. Implement core algorithm
2. Run on 1D Ising (L=8, 10, 12)
3. Compare with exact diagonalization
4. Generate arxiv paper figures

### ⚛️ **Week 3: Scattering**
1. Trotter order convergence study
2. Two-particle S-matrix extraction
3. Three-particle dynamics

### 🎨 **Week 4: GUI**
1. Rebuild with correct parameters
2. Add analysis visualizations
3. Polish for research use

---

## TESTING STRATEGY

After each phase:
```bash
# Run full test suite
pytest tests/ -v

# Run specific physics validation
python scripts/validate_phase_N.py

# Generate comparison plots
python scripts/compare_with_exact.py
```

---

## NOTES & QUESTIONS

**Q: Why did the original approach fail?**  
A: We built the GUI before validating the physics. This led to:
- Parameter confusion (mixing g_z, mass, etc.)
- No way to verify correctness
- Misleading visualizations

**Q: What's the minimum viable product (MVP)?**  
A: Phase 0 + Phase 1 (spectrum analysis + ADAPT-VQE). This gives us:
- Verified ground states
- Validated evolution
- Publication-ready figures

**Q: Can we skip ADAPT-VQE?**  
A: No - it's in the arxiv paper and validates our quantum circuit approach.

---

## REFERENCES

- **Paper**: `docs/QuantumScatteringAdaptVQEarxiv.pdf`
- **Models**: `src/models/*.py`
- **Tests**: `tests/`
- **Current Dashboard**: `dashboard/` (needs rebuild)
