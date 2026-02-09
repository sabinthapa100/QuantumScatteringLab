# Physics Knowledge: SU(2) Gauge Theory

## Model Definition (Kogut-Susskind)

**Total Hamiltonian:**
$$H = H_E + H_M$$

### Electric Part
$$H_E = J \sum_i Z_i Z_{i+1} + h_z \sum_i Z_i$$

where:
- $J = -\frac{3g^2}{16}$
- $h_z = \frac{3g^2}{8}$

### Magnetic Part
$$H_M = \frac{h_x}{16} \sum_i \left( X_i - 3Z_{i-1}X_i - 3X_i Z_{i+1} + 9Z_{i-1}X_i Z_{i+1} \right)$$

where:
- $h_x = -\frac{2}{(ag)^2}$

**Parameters:**
- $g$: Gauge coupling
- $a$: Lattice spacing

## Magnetic Cases

**Case 1:** Only $X_i$ terms  
**Case 2:** $X_i + Z_{i-1}X_i + X_i Z_{i+1}$  
**Case 3:** Full (all four terms) ← **Default**

## Phase Diagram

**Strong Coupling** ($g \to \infty$): Electric term dominates → Ising-like  
**Weak Coupling** ($g \to 0$): Magnetic term dominates  
**Critical Point**: Confinement-deconfinement transition at $g = g_c$ (TBD)

## Mapping to Heisenberg

In certain limits, SU(2) gauge theory maps to XXZ/Heisenberg spin chains.

## Operator Pool (ADAPT-VQE)

$$\{Y_i, Y_i Z_{i+1}, Z_{i-1} Y_i, Z_{i-1} Y_i Z_{i+1}\}$$

## Trotterization Strategy

**Layers:**
1. $H_E$ (diagonal, commuting)
2. $H_M$ Even sites
3. $H_M$ Odd sites

## Tasks

- [ ] Find critical coupling $g_c$
- [ ] Characterize CFT at $g_c$
- [ ] Compare with legacy implementation
- [ ] Verify confinement in strong coupling

## References

- Our implementation: `src/models/su2.py`
- Legacy code: `src/groundstate/HEP_QSim_SabinYao/`
- Trotterization: `HEP_QSim_SabinYao/adiabatic_sabin/src/trotter.py`
