# Workflow: Run Spectrum Analysis

**Purpose:** Compute and visualize energy spectrum for any model

## Prerequisites

```bash
# Ensure environment is activated
source .venv/bin/activate  # or conda activate quantumscattering

# Verify installation
python -c "from src.models.ising_1d import IsingModel1D; print('OK')"
```

## Quick Start

```bash
# Run Ising spectrum demo
python examples/02_ising_spectrum.py
```

## Custom Analysis

### Step 1: Choose Model

```python
from src.models.ising_1d import IsingModel1D
from src.models.su2 import SU2GaugeModel

# Option A: 1D Ising
model = IsingModel1D(
    num_sites=8,
    g_x=1.0,      # Transverse field
    g_z=0.0,      # Longitudinal field
    pbc=False     # Open boundaries
)

# Option B: SU(2) Gauge
model = SU2GaugeModel(
    num_sites=6,
    g=1.0,        # Gauge coupling
    a=1.0,        # Lattice spacing
    pbc=True
)
```

### Step 2: Compute Spectrum

```python
from src.analysis.spectrum import SpectrumAnalyzer

analyzer = SpectrumAnalyzer(model)
result = analyzer.compute_spectrum(num_states=10)

print(f"Ground state energy: {result['energies'][0]:.6f}")
print(f"Energy gap: {result['gap']:.6f}")
```

### Step 3: Visualize

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(result['energies'], 'o-', markersize=8)
plt.xlabel('State index')
plt.ylabel('Energy')
plt.title(f'Energy Spectrum (N={model.num_sites})')
plt.grid(True, alpha=0.3)
plt.savefig('spectrum.png', dpi=150)
plt.show()
```

## Parameter Scan

```python
from src.analysis.spectrum import scan_parameter
import numpy as np

# Scan transverse field
g_x_values = np.linspace(0.5, 1.5, 50)

result = scan_parameter(
    model_class=IsingModel1D,
    param_name='g_x',
    param_values=g_x_values,
    fixed_params={'num_sites': 8, 'g_z': 0.0, 'pbc': False},
    num_states=4
)

# Plot gap vs parameter
plt.plot(result['param_values'], result['gaps'])
plt.axvline(x=1.0, color='red', linestyle='--', label='Critical')
plt.xlabel('$g_x$')
plt.ylabel('Gap $\\Delta E$')
plt.legend()
plt.savefig('gap_scan.png', dpi=150)
```

## Advanced: Phase Diagram

```python
from src.analysis.framework import AnalysisConfig, BoundaryCondition, ModelAnalyzer

config = AnalysisConfig(
    model_class=IsingModel1D,
    fixed_params={'j_int': 1.0},
    boundary_condition=BoundaryCondition.OBC,
    system_sizes=[8]
)

analyzer = ModelAnalyzer(config)

phase_data = analyzer.compute_2d_phase_diagram(
    param1_name='g_x',
    param1_values=np.linspace(0.3, 1.7, 60),
    param2_name='g_z',
    param2_values=np.linspace(-0.3, 0.3, 50),
    system_size=8
)

from src.visualization import plot_2d_phase_diagram

plot_2d_phase_diagram(
    phase_data,
    critical_line={'type': 'vertical', 'value': 1.0, 'label': 'Critical'},
    save_path='phase_diagram.png'
)
```

## Troubleshooting

**Issue:** `ImportError: No module named 'src'`
```bash
# Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or install package
pip install -e .
```

**Issue:** Computation too slow
```bash
# Reduce system size or use fewer states
num_sites = 6  # instead of 12
num_states = 4  # instead of 10
```

**Issue:** Out of memory
```bash
# Use sparse methods (already default)
# Or reduce system size
```

## Expected Output

- **Console**: Energy values, gap, convergence info
- **Files**: `spectrum.png`, `gap_scan.png`, `phase_diagram.png`
- **Time**: ~1-30 seconds depending on system size

## Next Steps

- Criticality analysis: `examples/04_advanced_criticality.py`
- PBC vs OBC: `examples/05_ising_pbc_vs_obc.py`
- Ground state prep: `examples/01_ising_demo.py`
