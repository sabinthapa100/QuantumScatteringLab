# Testing Guide - QuantumScatteringLab

Quick reference for running tests and debugging.

## Running Tests

### All Tests
```bash
cd /home/sawin/Desktop/QuantumComputing/quantumscatteringlab
python3 -m pytest tests/ -v
```

### Specific Test File
```bash
# Test models only
python3 -m pytest tests/unit/test_models.py -v

# Test backends only
python3 -m pytest tests/integration/test_backends.py -v
```

### Specific Test Class or Method
```bash
# Test only Ising model
python3 -m pytest tests/unit/test_models.py::TestIsingModel1D -v

# Test specific method
python3 -m pytest tests/unit/test_models.py::TestIsingModel1D::test_hamiltonian_hermiticity -v
```

### With Coverage
```bash
# Generate coverage report
python3 -m pytest tests/ --cov=src --cov-report=html

# View report
xdg-open htmlcov/index.html  # Linux
# or
open htmlcov/index.html      # Mac
```

### Skip Slow Tests
```bash
python3 -m pytest tests/ -m "not slow" -v
```

## Quick Manual Tests

### Test a Model
```bash
python3 -c "
from src.models.ising_1d import IsingModel1D

model = IsingModel1D(num_sites=4, g_x=1.0, pbc=True)
print(f'Model: {model}')
print(f'Symmetries: {model.get_symmetries()}')

H = model.build_hamiltonian()
print(f'Hamiltonian: {len(H)} terms')
print('✅ Ising model works!')
"
```

### Test Heisenberg Model
```bash
python3 -c "
from src.models.heisenberg import HeisenbergModel

model = HeisenbergModel(num_sites=4, j_x=1.0, j_y=1.0, j_z=1.0, pbc=True)
print(f'Model: {model}')
print(f'Metadata: {model.get_metadata().name}')
print('✅ Heisenberg model works!')
"
```

### Test Backend
```bash
python3 -c "
from src.backends.qiskit_backend import QiskitBackend
from src.models.ising_1d import IsingModel1D

backend = QiskitBackend()
model = IsingModel1D(num_sites=4, g_x=1.0)

state = backend.get_reference_state(model.num_sites)
H = model.build_hamiltonian()
energy = backend.compute_expectation_value(state, H)

print(f'Energy: {energy:.6f}')
print('✅ Backend works!')
"
```

## Test Results Summary

### Current Status (as of last run)
```
tests/unit/test_models.py ............................ 29 passed ✅
```

### Test Coverage

#### IsingModel1D (12 tests)
- ✅ Basic instantiation
- ✅ Parameter validation
- ✅ Hamiltonian Hermiticity
- ✅ PBC vs OBC
- ✅ Operator pool size and Hermiticity
- ✅ Trotter layers
- ✅ Symmetry detection (PBC, OBC, Z2)
- ✅ Metadata
- ✅ Properties (num_bonds)

#### HeisenbergModel (11 tests)
- ✅ XXX instantiation
- ✅ XXZ instantiation
- ✅ Parameter validation
- ✅ Hamiltonian Hermiticity and terms
- ✅ Operator pool Hermiticity
- ✅ Trotter layers
- ✅ Symmetry detection (SU2, U1)
- ✅ Metadata (XXX, XXZ)

#### Cross-Model Tests (2 tests)
- ✅ Ising-Heisenberg limit
- ✅ Reference state labels

#### Edge Cases (4 tests)
- ✅ Minimal system size (2 sites)
- ✅ Large system instantiation (100 sites)
- ✅ Zero field Ising
- ✅ String representations

## Debugging Tips

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your code
from src.models.ising_1d import IsingModel1D
model = IsingModel1D(num_sites=4, g_x=1.0)
```

### Check Test Output
```bash
# Show full traceback
python3 -m pytest tests/unit/test_models.py -v --tb=long

# Show only failed tests
python3 -m pytest tests/unit/test_models.py -v --tb=short -x
```

### Run Single Test with Print Statements
```bash
# Add print() in your test, then:
python3 -m pytest tests/unit/test_models.py::TestIsingModel1D::test_basic_instantiation -v -s
```

### Interactive Debugging
```python
# Add this in your test where you want to debug:
import pdb; pdb.set_trace()

# Then run:
python3 -m pytest tests/unit/test_models.py::TestIsingModel1D::test_basic_instantiation -v -s
```

## Common Issues

### Import Errors
```bash
# Make sure package is installed
pip install -e .

# Or reinstall
pip install -e . --force-reinstall
```

### Qiskit Version Issues
```bash
# Check version
python3 -c "import qiskit; print(qiskit.__version__)"

# Update if needed
pip install --upgrade qiskit
```

### Test Discovery Issues
```bash
# Make sure __init__.py exists in test directories
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
```

## Writing New Tests

### Template for Model Tests
```python
def test_my_new_feature(self):
    \"\"\"Test description.\"\"\"
    # Arrange
    model = IsingModel1D(num_sites=4, g_x=1.0)
    
    # Act
    result = model.some_method()
    
    # Assert
    assert result == expected_value
```

### Template for Backend Tests
```python
def test_backend_feature(self):
    \"\"\"Test backend functionality.\"\"\"
    backend = QiskitBackend()
    model = IsingModel1D(num_sites=4)
    
    state = backend.get_reference_state(4)
    # ... test logic
    
    assert condition
```

## Continuous Integration

### Pre-commit Checks
```bash
# Run before committing
python3 -m pytest tests/ -v
python3 -m pytest tests/ --cov=src --cov-report=term-missing
```

### Quick Smoke Test
```bash
# Fast check (< 1 second)
python3 -m pytest tests/unit/test_models.py::TestIsingModel1D::test_basic_instantiation -v
```

## Performance Testing

### Timing Tests
```bash
# Show test durations
python3 -m pytest tests/ -v --durations=10
```

### Profile a Test
```python
import cProfile
import pstats

def test_performance():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    model = IsingModel1D(num_sites=20)
    H = model.build_hamiltonian()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime')
    stats.print_stats(10)
```

---

**Next Steps**: See `docs/USAGE.md` for full usage examples.
