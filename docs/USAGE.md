# Usage Guide - QuantumScatteringLab

This guide shows you how to use the `QuantumScatteringLab` package step-by-step.

## Quick Start

### 1. Installation

```bash
cd /home/sawin/Desktop/QuantumComputing/quantumscatteringlab

# Install the package in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[all]"
```

### 2. Basic Model Usage

#### Creating a 1D Ising Model

```python
from src.models.ising_1d import IsingModel1D

# Create a model at criticality
model = IsingModel1D(
    num_sites=8,      # 8 qubits
    j_int=1.0,        # Ising coupling
    g_x=1.0,          # Transverse field (critical point!)
    g_z=0.0,          # No longitudinal field
    pbc=True          # Periodic boundary conditions
)

# Inspect the model
print(model)
print(f"Symmetries: {model.get_symmetries()}")
print(f"Metadata: {model.get_metadata()}")

# Build the Hamiltonian
H = model.build_hamiltonian()
print(f"Hamiltonian has {len(H)} terms")

# Get operator pool for ADAPT-VQE
pool = model.build_operator_pool()
print(f"Operator pool has {len(pool)} operators")

# Get Trotter layers for time evolution
layers = model.get_trotter_layers()
print(f"Trotter decomposition has {len(layers)} layers")
```

#### Creating a Heisenberg Model

```python
from src.models.heisenberg import HeisenbergModel

# XXX (isotropic) model
model_xxx = HeisenbergModel(
    num_sites=6,
    j_x=1.0, j_y=1.0, j_z=1.0,  # Isotropic
    h=0.0,
    pbc=True
)

# XXZ model
model_xxz = HeisenbergModel(
    num_sites=6,
    j_x=1.0, j_y=1.0, j_z=2.0,  # Anisotropic
    h=0.1,
    pbc=False
)
```

#### Creating an SU(2) Gauge Model

```python
from src.models.su2 import SU2GaugeModel

model = SU2GaugeModel(
    num_sites=6,
    g=1.0,    # Gauge coupling
    a=1.0,    # Lattice spacing
    pbc=True
)
```

### 3. Using Backends

#### Qiskit Backend (Exact, CPU)

```python
from src.backends.qiskit_backend import QiskitBackend
from src.models.ising_1d import IsingModel1D

# Create backend and model
backend = QiskitBackend()
model = IsingModel1D(num_sites=8, g_x=1.0, pbc=True)

# Get initial state |00...0⟩
state = backend.get_reference_state(model.num_sites)
print(f"State type: {type(state)}")

# Compute energy
H = model.build_hamiltonian()
energy = backend.compute_expectation_value(state, H)
print(f"Ground state energy estimate: {energy:.6f}")

# Apply an operator (for VQE)
pool = model.build_operator_pool()
state_new = backend.apply_operator(state, pool[0], parameter=0.1)

# Time evolution (Trotter)
layers = model.get_trotter_layers()
state_evolved = backend.evolve_state_trotter(state, layers, time_step=0.01)
```

#### Quimb Backend (MPS, CPU/GPU)

```python
from src.backends.quimb_backend import QuimbBackend

# Create MPS backend (CPU by default)
backend = QuimbBackend(use_gpu=False)

# Use same interface as Qiskit!
state = backend.get_reference_state(num_sites=20)  # Can handle larger systems
# ... rest is identical
```

### 4. Analysis Framework

#### Computing Phase Diagrams

```python
from src.analysis.framework import AnalysisConfig, BoundaryCondition, ModelAnalyzer
from src.models.ising_1d import IsingModel1D
import numpy as np

# Configure analysis
config = AnalysisConfig(
    model_class=IsingModel1D,
    fixed_params={'j_int': 1.0, 'g_z': 0.0},
    boundary_condition=BoundaryCondition.OBC,
    system_sizes=[6, 8, 10, 12]
)

analyzer = ModelAnalyzer(config)

# 1D scan: Gap vs g_x
g_x_values = np.linspace(0.5, 1.5, 50)
param_vals, gaps = analyzer.compute_1d_phase_scan(
    param_name='g_x',
    param_values=g_x_values,
    system_size=8
)

import matplotlib.pyplot as plt
plt.plot(param_vals, gaps)
plt.xlabel('g_x')
plt.ylabel('Energy Gap')
plt.title('Ising Model Gap vs Transverse Field')
plt.show()
```

#### Computing Entanglement Entropy

```python
# Compute entanglement at critical point
ent_data = analyzer.compute_entanglement_at_criticality(
    critical_params={'g_x': 1.0},
    system_size=12
)

print(f"Central charge: {ent_data.central_charge:.3f}")
print(f"Expected for Ising CFT: 0.5")

# Plot entanglement entropy
plt.plot(ent_data.subsystem_sizes, ent_data.entropies, 'o-')
plt.xlabel('Subsystem size ℓ')
plt.ylabel('Entanglement Entropy S(ℓ)')
plt.show()
```

## Running Tests

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # or xdg-open on Linux
```

### Integration Tests

```bash
# Test backend compatibility
pytest tests/integration/test_backends.py -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

### Manual Testing

```bash
# Test a specific model
python3 -c "
from src.models.ising_1d import IsingModel1D
model = IsingModel1D(num_sites=4, g_x=1.0)
print(model.build_hamiltonian())
"

# Run an example script
python examples/01_ising_demo.py
python examples/02_ising_spectrum.py
```

## Examples

### Example 1: Ground State Energy

```python
from src.models.ising_1d import IsingModel1D
from src.analysis.spectrum import SpectrumAnalyzer

model = IsingModel1D(num_sites=8, g_x=1.0, pbc=True)
analyzer = SpectrumAnalyzer(model)

result = analyzer.compute_ground_state()
print(f"Ground state energy: {result['energy']:.6f}")
```

### Example 2: Spectrum Analysis

```python
from src.analysis.spectrum import SpectrumAnalyzer

analyzer = SpectrumAnalyzer(model)
spectrum = analyzer.compute_spectrum(num_states=6)

print("Low-energy spectrum:")
for i, E in enumerate(spectrum['energies']):
    print(f"  E_{i} = {E:.6f}")
```

### Example 3: Parameter Scan

```python
from src.analysis.spectrum import scan_parameter
import numpy as np

result = scan_parameter(
    model_class=IsingModel1D,
    param_name='g_x',
    param_values=np.linspace(0.5, 1.5, 20),
    fixed_params={'num_sites': 8, 'j_int': 1.0, 'g_z': 0.0, 'pbc': True},
    num_states=4
)

# Plot results
import matplotlib.pyplot as plt
plt.plot(result['param_values'], result['gaps'])
plt.xlabel('g_x')
plt.ylabel('Energy Gap')
plt.show()
```

## Debugging Tips

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all debug messages will be printed
from src.models.ising_1d import IsingModel1D
model = IsingModel1D(num_sites=4, g_x=1.0)
```

### Check Model Properties

```python
model = IsingModel1D(num_sites=8, g_x=1.0, pbc=True)

# Inspect properties
print(f"Number of sites: {model.num_sites}")
print(f"Number of bonds: {model.num_bonds}")
print(f"Boundary conditions: {'PBC' if model.pbc else 'OBC'}")
print(f"Symmetries: {model.get_symmetries()}")

# Check Hamiltonian
H = model.build_hamiltonian()
print(f"Hamiltonian terms: {len(H)}")
print(f"Is Hermitian: {H.is_hermitian()}")
```

### Verify Backend Compatibility

```python
from src.backends.qiskit_backend import QiskitBackend

backend = QiskitBackend()
backend.verify_compatibility(num_sites=20)  # Will warn if too large
```

## Common Workflows

### Workflow 1: Find Critical Point

```python
from src.models.ising_1d import IsingModel1D
from src.analysis.spectrum import scan_parameter
import numpy as np

# Scan transverse field
g_x_values = np.linspace(0.8, 1.2, 50)
result = scan_parameter(
    model_class=IsingModel1D,
    param_name='g_x',
    param_values=g_x_values,
    fixed_params={'num_sites': 10, 'j_int': 1.0, 'g_z': 0.0, 'pbc': True},
    num_states=3
)

# Find minimum gap
min_idx = np.argmin(result['gaps'])
g_x_critical = result['param_values'][min_idx]
print(f"Critical point: g_x ≈ {g_x_critical:.4f}")
```

### Workflow 2: Scaling Collapse

```python
from src.analysis.framework import ModelAnalyzer, AnalysisConfig, BoundaryCondition
import numpy as np

config = AnalysisConfig(
    model_class=IsingModel1D,
    fixed_params={'j_int': 1.0, 'g_z': 0.0},
    boundary_condition=BoundaryCondition.OBC,
    system_sizes=[6, 8, 10, 12, 14]
)

analyzer = ModelAnalyzer(config)
scaling_data = analyzer.compute_scaling_collapse(
    param_name='g_x',
    param_values=np.linspace(0.8, 1.2, 30),
    param_critical=1.0,
    nu=1.0,  # Correlation length exponent
    z=1.0    # Dynamical exponent
)

# Plot scaling collapse
# ... (see examples/04_advanced_criticality.py)
```

## Next Steps

1. **Explore Examples**: Check `examples/` directory for complete scripts
2. **Read Documentation**: See `docs/` for detailed physics background
3. **Run Tests**: Verify your installation with `pytest tests/`
4. **Contribute**: See `docs/CONTRIBUTING.md` for guidelines

## Troubleshooting

### Import Errors

```bash
# Make sure package is installed
pip install -e .

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"
```

### Qiskit Issues

```bash
# Reinstall Qiskit
pip install --upgrade qiskit qiskit-aer
```

### Quimb Issues

```bash
# Install Quimb
pip install quimb

# For GPU support (optional)
pip install cupy-cuda11x  # Replace with your CUDA version
```

---

**For more help**: See the full documentation in `docs/` or open an issue on GitHub.
