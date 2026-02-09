# Physics Knowledge: 1D Ising Model Criticality

## Model Definition

**Hamiltonian:**
$$H = -J \sum_{i} Z_i Z_{i+1} - g_x \sum_i X_i - g_z \sum_i Z_i$$

**Parameters:**
- $J$: Ising coupling (typically $J=1$)
- $g_x$: Transverse field
- $g_z$: Longitudinal field

## Critical Point

**Location:** $g_x = 1, g_z = 0$

**CFT Description:** Free Majorana fermion with central charge $c = 1/2$

## Phase Diagram

```
         g_x
          ^
          |
    2.0   |  Paramagnetic (Disordered)
          |
    1.0   |━━━━━━━━━━━ Critical Line ━━━━━━━━━━━
          |
    0.0   |  Ferromagnetic (Ordered)
          +──────────────────────────────> g_z
         -0.5        0.0         0.5
```

**Phases:**
- **$g_x < 1$**: Ordered (ferromagnetic), $\langle Z \rangle \neq 0$
- **$g_x = 1$**: Critical (CFT), gapless
- **$g_x > 1$**: Disordered (paramagnetic), $\langle Z \rangle = 0$

## Scaling Parameter

$$\eta_{\text{latt}} = \frac{g_x - 1}{|g_z|^{8/15}}$$

Controls the family of massive QFTs near criticality.

## Critical Exponents

- **Correlation length**: $\nu = 1$
- **Dynamical**: $z = 1$
- **Central charge**: $c = 1/2$

## Verification Checklist

- [x] Reproduce phase diagram (examples/04_phase_diagram_2d.png)
- [x] Verify scaling collapse with $\nu=1, z=1$
- [ ] Extract $c=0.5$ with OBC (currently get $c \approx 0.02$ due to finite size)
- [ ] Increase system size to N>20 for accurate central charge

## References

- Paper: arXiv:2505.03111v2
- Mussardo Chapter 9: Phase Transitions
- Our implementation: `src/models/ising_1d.py`
