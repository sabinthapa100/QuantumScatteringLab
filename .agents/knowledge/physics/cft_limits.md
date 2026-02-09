# Conformal Field Theory Limits

## What is CFT?

**Conformal Field Theory** describes quantum systems at critical points where:
1. Correlation length $\xi \to \infty$
2. System is **massless**
3. **Scale invariance**: Physics looks the same at all length scales
4. **Conformal invariance**: Scale + Lorentz symmetry

## When Do We Reach CFT?

### 1D Transverse-Field Ising

**Critical Point:** $g_x = 1, g_z = 0$

**CFT:** Free Majorana fermion  
**Central Charge:** $c = 1/2$

**How to Verify:**
- Energy gap closes: $\Delta E \to 0$
- Entanglement entropy: $S(\ell) = \frac{c}{3}\log(\ell) + \text{const}$
- Correlation function: $\langle Z_i Z_j \rangle \sim |i-j|^{-2\Delta}$ (power law)

### 2D Ising Model

**Critical Point:** $h_x = h_{x,c}, h_z = 0$

**CFT:** 3D Ising CFT  
**Central Charge:** $c = 1/2$ (in 3D)  
**Scaling Dimensions:**
- Energy: $\Delta_\epsilon = 1.413$
- Spin: $\Delta_\sigma = 0.518$

### SU(2) Gauge Theory

**Critical Point:** Confinement-deconfinement transition at $g = g_c$ (unknown)

**CFT:** Gauge CFT (to be determined)

**How to Find:**
1. Scan coupling $g$
2. Look for gap closing
3. Compute entanglement entropy
4. Extract central charge

## Mussardo Chapter 10 Key Quote

> "Right at the critical point, the correlation length is infinite: 
> the corresponding field theory is therefore massless and becomes 
> invariant under a dilation of the length-scales"

## Universal Properties

At CFT:
- **Massless**: $m = 0$
- **Gapless**: $\Delta E \to 0$
- **Power-law correlations**: $\langle O(x) O(0) \rangle \sim |x|^{-2\Delta}$
- **Logarithmic entanglement**: $S \sim \frac{c}{3}\log(\ell)$
- **Finite-size scaling**: $\Delta E \sim L^{-z}$

## Verification Strategy

**For Each Model:**
1. Scan parameters to find gap minimum
2. Compute entanglement entropy
3. Fit $S(\ell) = (c/3)\log(\ell) + b$
4. Extract central charge $c$
5. Compare with theory

**Status:**
- ✅ 1D Ising: Found critical point, $c \approx 0.02$ (finite-size, need N>20)
- 🚧 2D Ising: Not yet implemented
- 🚧 SU(2): Critical point unknown

## References

- Mussardo, Statistical Field Theory, Chapters 9-10
- arXiv:2505.03111v2
- Our analysis: `examples/04_advanced_criticality.py`
