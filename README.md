# Quantum Scattering Lab (QSL)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Framework: Quimb](https://img.shields.io/badge/Engine-Quimb-green.svg)](https://quimb.readthedocs.io/)

A production-grade quantum simulation framework designed to explore high-energy physics phenomena using quantum information theoretic tools. This repository focuses on simulating **scattering dynamics in 1D/2D Ising Models** and **SU(2) Lattice Gauge Theories**, implementing rigorous **Matrix Product State (MPS)** methods and **ADAPT-VQE** state preparation algorithms.

## 🔬 Scientific Objectives

My work in this repository targets three core research areas inspired by *Farrell et al. (2025)*:

1.  **Vacuum & State Preparation**:
    - Utilizing **ADAPT-VQE** to construct accurate ground states and single-particle excitations (magnons/mesons) on quantum hardware.
    - Benchmarking ansatz depth connectivity against exact diagonalization.

2.  **Real-Time Scattering Dynamics**:
    - Simulating wavepacket collisions to observe **elastic vs. inelastic scattering** regimes.
    - Quantifying particle production and entanglement entropy growth during high-energy collision events.
    - **Models**:
        - **1D Ising**: Transverse ($g_x$) and Longitudinal ($g_z$) fields.
        - **SU(2) Gauge Theory**: Plaquette chain mapping for "glueball" dynamics.

3.  **Interactive Visualization**:
    - A custom-built **React + FastAPI** dashboard to visualize energy density evolution in real-time, allowing for rapid parameter tuning and intuition building.

## 🏗️ Architecture

The codebase is structured for scalability and reproducibility:

- **`src/`**: The Core Engine.
    - **`models/`**: rigorous Hamiltonian definitions (`IsingModel1D`, `SU2GaugeModel`) with explicit symmetry sectors.
    - **`backends/`**: 
        - `QuimbMPSBackend`: Optimized Tensor Network backend for $L > 20$ sites.
        - `QiskitBackend`: Exact statevector simulation for validation.
    - **`simulation/`**: Trotterized time-evolution and ADAPT-VQE solvers.
    
- **`examples/`**: Research scripts.
    - `scattering/`: Production scripts for generating scattering heatmaps.
    - `ground_state/`: Converging vacuum states using variational methods.
    
- **`dashboard/`**: Interactive GUI.
    - Full-stack visualization tool for presenting results dynamically.

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/sabinthapa100/quantumscattering.git
cd quantumscattering

# Setup Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

# Install Core & Dev Dependencies
pip install -e ".[all]"
```

### 2. Running Simulations

Generate a basic scattering trajectory (Ising Model):

```bash
python examples/generate_gui_data.py
```
*Output will be saved to `dashboard/public/data/scattering_data.json`.*

Run the ADAPT-VQE verification:

```bash
python examples/ground_state/01_ising1d_adapt_vqe.py
```

### 3. Launching the Dashboard

To verify and inspect the simulation data interactively:

```bash
# Terminal 1: Backend API
cd dashboard
../.venv/bin/uvicorn server:app --reload

# Terminal 2: Frontend Client
cd dashboard
npm install
npm run dev
```

## 📈 Results & Analysis

Key results include:
- **Vacuum Subtraction**: Clean signal extraction of scattering wavepackets.
- **Entanglement Entropy**: Verification of area law vs. volume law growth during collisions.
- **Phase Diagrams**: Mapping confinement-deconfinement transitions in the SU(2) model.

---

**Author**: Sabin Thapa  
**Focus**: Quantum Simulation, Lattice Field Theory, Tensor Networks.
