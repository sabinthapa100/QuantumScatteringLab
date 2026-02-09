# Legacy SU(2) Implementation Analysis

**⚠️ CRITICAL: DO NOT DELETE THIS CODE ⚠️**

## Location

`src/groundstate/HEP_QSim_SabinYao/`

This directory contains the original working implementations for SU(2) gauge theory ground state preparation using both ADAPT-VQE and adiabatic evolution.

## Directory Structure

```
HEP_QSim_SabinYao/
├── adaptVQE_sabin/          # ADAPT-VQE implementation
│   ├── src/
│   │   ├── adapt_vqe.py     # Main ADAPT algorithm
│   │   ├── hamiltonian.py   # SU(2) Hamiltonian construction
│   │   ├── operator_pool.py # Operator pool for SU(2)
│   │   └── vqe.py           # VQE optimization
│   └── examples/
├── adiabatic_sabin/         # Adiabatic evolution
│   ├── src/
│   │   ├── adiabatic_evolution.py
│   │   ├── hamiltonian.py
│   │   ├── trotter.py       # ⭐ CRITICAL: Trotterization
│   │   └── observables.py
│   └── archive/
├── gs_su2/                  # Ground state SU(2) analysis
└── ourCode_GS_SU2/          # Our original SU(2) code
```

## Key Components

### 1. Trotterization (`adiabatic_sabin/src/trotter.py`)

**Class:** `Trotterizer(N, J, hz, hx, T, NT, mag_case=3, boundary='periodic')`

**Features:**
- ✅ First-order Trotter
- ✅ Second-order Trotter (symmetric)
- ✅ Three magnetic cases (1, 2, 3)
- ✅ PBC and OBC support
- ✅ Even-odd decomposition for parallelism

**Atomic Gates:**
```python
func_ZZ(circ, i, j, J, delta_t)      # Electric: exp(-i J Z_i Z_j dt)
func_Z(circ, i, hz, delta_t)         # Electric: exp(-i hz Z_i dt)
func_X(circ, i, hx, t, T, delta_t)   # Magnetic: exp(-i hx X_i dt)
func_ZX(circ, i, j, hx, t, T, delta_t)   # Magnetic: exp(i hx Z_i X_j dt)
func_XZ(circ, i, j, hx, t, T, delta_t)   # Magnetic: exp(i hx X_i Z_j dt)
func_ZXZ(circ, i, j, k, hx, t, T, delta_t) # Magnetic: exp(-i hx Z_i X_j Z_k dt)
```

**Layering Strategy:**
1. Electric terms (even-odd decomposition)
2. Magnetic X terms (all sites)
3. Magnetic ZX terms (even-odd)
4. Magnetic XZ terms (even-odd)
5. Magnetic ZXZ terms (three-column decomposition)

### 2. ADAPT-VQE (`adaptVQE_sabin/src/adapt_vqe.py`)

**Operator Pool for SU(2):**
- Single-site: $Y_i$
- Two-site: $Y_i Z_{i+1}$, $Z_{i-1} Y_i$
- Three-site: $Z_{i-1} Y_i Z_{i+1}$

**Gradient Computation:**
- Parameter-shift rule
- Gradient-based operator selection

### 3. Hamiltonian Construction

**Electric Part:**
```python
H_E = J * sum(Z_i Z_{i+1}) + hz * sum(Z_i)
```

**Magnetic Part (mag_case=3):**
```python
H_M = (hx/16) * sum(
    X_i 
    - 3*Z_{i-1}*X_i 
    - 3*X_i*Z_{i+1} 
    + 9*Z_{i-1}*X_i*Z_{i+1}
)
```

## What We Learned

### ✅ Best Practices

1. **Even-Odd Decomposition**: Maximizes gate parallelism
2. **Symmetric Trotter**: Second-order gives $O(\Delta t^3)$ error
3. **Modular Gates**: Atomic functions for each term
4. **Boundary Flexibility**: PBC/OBC handled cleanly
5. **Time-Dependent H**: Adiabatic interpolation $H(t)$

### ⚠️ Limitations

1. **Qiskit-Only**: No GPU support
2. **Hardcoded Parameters**: Not easily configurable
3. **No Abstraction**: Tightly coupled to SU(2)
4. **Limited Documentation**: Sparse comments

## How New Code Improves

**Our New Implementation:**

| Feature | Legacy | New |
|---------|--------|-----|
| **Backend** | Qiskit only | Qiskit + Quimb (GPU) |
| **Abstraction** | SU(2)-specific | `PhysicsModel` ABC |
| **Configuration** | Hardcoded | `AnalysisConfig` dataclass |
| **Testing** | None | pytest with >90% coverage |
| **Documentation** | Minimal | Comprehensive |
| **Modularity** | Monolithic | Reusable components |

**But We Keep:**
- ✅ Trotterization strategy
- ✅ Even-odd decomposition
- ✅ Symmetric second-order
- ✅ Magnetic case flexibility

## Comparison Tests

**Critical:** New implementations MUST match legacy results numerically.

```python
# tests/integration/test_legacy_comparison.py
def test_su2_ground_state_matches_legacy():
    # Run legacy code
    legacy_energy = run_legacy_adapt_vqe(N=6, g=1.0, a=1.0)
    
    # Run new code
    from src.models.su2 import SU2GaugeModel
    from src.simulation.adapt_vqe import ADAPTVQESolver
    
    model = SU2GaugeModel(num_sites=6, g=1.0, a=1.0)
    solver = ADAPTVQESolver(model)
    result = solver.run()
    
    # Must match within numerical precision
    assert abs(result['energy'] - legacy_energy) < 1e-10
```

## Preservation Strategy

1. **Never Delete**: Keep entire `HEP_QSim_SabinYao/` directory
2. **Document**: This file serves as analysis
3. **Extract**: Copy useful patterns to new code
4. **Verify**: Comparison tests ensure correctness
5. **Reference**: Link from new code comments

## References

- Legacy directory: `src/groundstate/HEP_QSim_SabinYao/`
- New SU(2) model: `src/models/su2.py`
- New Trotterizer: (to be implemented, inspired by legacy)
- Comparison tests: `tests/integration/test_legacy_comparison.py`

---

**Last Updated:** 2026-02-09  
**Status:** Protected, analyzed, patterns extracted
