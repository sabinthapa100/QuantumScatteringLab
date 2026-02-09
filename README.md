# Quantum Scattering Lab (QSL)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Qiskit](https://img.shields.io/badge/Framework-Qiskit-blueviolet.svg)](https://qiskit.org/)
[![Enginge: Quimb](https://img.shields.io/badge/Engine-Quimb-green.svg)](https://quimb.readthedocs.io/)

A production-grade Object-Oriented framework for simulating quantum scattering and many-body dynamics in Lattice Field Theories (LFTs).

---

## 🔬 Scientific Overview

QuantumScatteringLab provides the computational infrastructure to study real-time dynamics of gauge theories and spin systems. The framework is designed to bridge the gap between classical tensor network simulations (MPS) and near-term quantum hardware execution (Qiskit).

### Core Physics Models

1.  **1D Transverse-Field Ising Model (TFIM)**
    $$H = -J \sum_{i} Z_i Z_{i+1} - g_x \sum_i X_i - g_z \sum_i Z_i$$
    *   **Phase Physics:** Supports periodic (PBC) and open (OBC) boundaries. Focus on $(1+1)D$ criticality at $g_x=1$.

2.  **Heisenberg XXX/XXZ Spin Chain**
    $$H = \sum_i [J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1}] + h \sum_i Z_i$$
    *   **Symmetry:** Isotropic $SU(2)$ for $J_x=J_y=J_z$ and $U(1)$ for XXZ.

3.  **SU(2) Lattice Gauge Theory (Kogut-Susskind mapping)**
    $$H_E = J \sum Z_i Z_{i+1} + h_z \sum Z_i, \quad H_M = \frac{h_x}{16} \sum [ X_i - 3 Z_{i-1} X_i - 3 X_i Z_{i+1} + 9 Z_{i-1} X_i Z_{i+1} ]$$
    *   **Implementation:** Rigorous mapping from gauge links to site qubits, including boundary-correct plaquette handling.

---

## 🏗️ Architecture

The project follows a modular, extensible design pattern to ensure research reproducibility and scalability.

-   **`src/models/`**: Physics definitions. Generates Hamiltonians and ADAPT-VQE operator pools using `SparsePauliOp`.
-   **`src/backends/`**: Simulation engines.
    -   `QiskitBackend`: Exact statevector simulation (best for $N \le 20$).
    -   `QuimbBackend`: Matrix Product States (MPS) for large-scale $1D$ chains ($N \gg 20$).
-   **`src/simulation/`**: State preparation algorithms.
    -   `ADAPTVQESolver`: Implementation of Grimsley et al. with customizable gradient selection.
    -   `AdiabaticSolver`: Time-dependent Hamiltonian evolution.
-   **`src/analysis/`**: Scientific processing.
    -   Energy spectrum, entanglement entropy, Phase diagrams, and scaling collapse.

---

## 🚀 Installation

```bash
# Development setup
git clone https://github.com/sabinthapa100/quantumscattering.git
cd quantumscattering
pip install -e ".[all]"
```

---

## 🧪 Quality Assurance

We maintain strict verification standards using `pytest`:

1.  **Unit Tests**: Verify Hamiltonian Hermiticity, Symmetry sectors, and Operator pool integrity.
2.  **Integration Tests**: Ensure Qiskit (Exact) and Quimb (MPS) produce numerically consistent results.

```bash
# Run the complete test suite
python3 -m pytest tests/ -v
```

---

## 📈 Research Workflows

### 1. Ground State Preparation (ADAPT-VQE)
The framework implements the **Adaptive Ansatz Construction** to minimize circuit depth while hitting scientific accuracy targets.

```python
from src.models.ising_1d import IsingModel1D
from src.simulation.adapt_vqe import ADAPTVQESolver
from src.backends.qiskit_backend import QiskitBackend

model = IsingModel1D(num_sites=10, g_x=1.0)
solver = ADAPTVQESolver(model, backend=QiskitBackend())
result = solver.run()
print(f"Ground state energy: {result['energy']}")
```

### 2. Time Evolution (Trotterization)
Real-time scattering is simulated via multi-layer Trotter decomposition.

```python
layers = model.get_trotter_layers()
final_state = backend.evolve_state_trotter(initial_state, layers, time_step=0.01, steps=100)
```

---

## 📚 References & Archive

The `archive/` folder contains original research scripts and Jupyter notebooks that serve as benchmarks. Key references:
- **Grimsley et al.**, "An adaptive variational algorithm for exact molecular simulations on a quantum computer" (arXiv:1812.11173).
- **Ebner et al.**, "Eigenstate Thermalization in 2+1 dimensional SU(2) Lattice Gauge Theory" (arXiv:2308.16202).
- **Farrell et al.**, "Digital quantum simulations of scattering in quantum field theories using W states" (arXiv:2505.03111v2).

---

**Developed for Advanced Quantum Simulations.**

**Principal Author:** Sabin Thapa ([sthapa3@kent.edu](mailto:sthapa3@kent.edu))  
**Research Area:** Quantum Simulation for Lattice Field Theory.
