# Professional Research Guide: Quantum Scattering Lab

This guide outlines the complete workflow for performing quantum scattering simulations, from model investigation to advanced physical measurements.

## Table of Contents
1. [Environment Setup](#1-environment-setup-critical)
2. [Basic Model Diagnostics](#2-basic-model-diagnostics)
3. [Ground State Preparation](#3-ground-state-preparation)
4. [Scattering Simulation Workflow](#4-scattering-simulation-workflow)
5. [Advanced Visualization & GUI](#5-advanced-visualization--gui)
6. [Expert Analysis: Entropy & Complexity](#6-expert-analysis-entropy--complexity)

---

## 1. Environment Setup (CRITICAL)
To avoid system package conflicts (`externally-managed-environment`), you **must** use a virtual environment.

### Step 1.1: Create & Activate Virtual Environment
Run these commands in the project root:
```bash
# multiple python installations?
# sudo apt install python3-venv  # (Only if venv is missing)

# 1. Create the virtual environment (.venv)
python3 -m venv .venv

# 2. Activate it (You must do this every time you open a new terminal)
source .venv/bin/activate
```
*You will know it's active when you see `(.venv)` at the start of your terminal prompt.*

### Step 1.2: Install Dependencies
```bash
# 3. Install core libraries
pip install -r requirements.txt

# 4. Install dashboard dependencies (FastAPI, Uvicorn)
pip install fastapi uvicorn pydantic
```

### Step 1.3: Set Python Path
To ensure imports work correctly:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

---

## 2. Basic Model Diagnostics
Before scattering, verify the model's Hamiltonian and Trotter decomposition.
- **Ising 1D/2D**: Check critical field values ($g_x \approx 1.0$).
- **SU(2) Gauge**: Verify the electric ($H_E$) and magnetic ($H_M$) terms.

Run integration tests:
```bash
pytest tests/test_ising.py tests/test_su2.py
```

## 3. Ground State Preparation
We provide two primary methods for preparing the vacuum/ground state:

### Method A: ADAPT-VQE (Recommended)
Automatically grows the circuit to find the ground state efficiently.
```bash
python3 examples/ground_state/01_ising_adapt_vqe.py
```

### Method B: Adiabatic State Preparation (ASP)
Evolves the state from a simple initial Hamiltonian to the target Hamiltonian.
*Implementation found in `src/simulation/adiabatic.py`.*

## 4. Scattering Simulation Workflow
### Step 1: Initialization
We create Gaussian wavepackets using the `prepare_two_wavepacket_state` function.
Parameters:
- $x_0$: Center position.
- $k_0$: Momentum.
- $\sigma$: Spatial spread.

### Step 2: Time Evolution
Using the `QuimbMPSBackend` for large systems ($L > 20$):
```bash
python3 examples/scattering/01_ising_scattering_mps.py
```
This script computes the energy density at each step: $\mathcal{E}_n(t) = \langle \Psi(t) | H_n | \Psi(t) \rangle - \langle \Psi_{vac} | H_n | \Psi_{vac} \rangle$.

## 5. Advanced Visualization & GUI
For interactive exploration, use the **Scattering Lab Dashboard**.

**Terminal 1 (Backend Engine):**
```bash
# Ensure venv is active
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd dashboard
../.venv/bin/uvicorn server:app --port 8000
```

**Terminal 2 (Frontend Interface):**
```bash
cd dashboard
npm install       # (Only the first time)
npm run dev
```

**Navigate**: Open `http://localhost:5173/` in your browser.

**Visualization Metrics**:
- **Heatmaps**: Time vs. Position energy density.
- **Particle Number**: Total excitation count $\langle N \rangle$.

## 6. Expert Analysis: Entropy & Complexity
To understand information dynamics, we measure the **Entanglement Entropy** $S_{vN} = -\text{Tr}(\rho \ln \rho)$ at each bond.

Run the advanced diagnostics:
```bash
python3 examples/physics_analysis/08_advanced_diagnostics.py
```

**New Measured Quantities**:
- **Bond Dimension Growth**: χ increases over time, indicating entanglement growth.
- **Circuit Depth Scaling**: Analyzes how many gates are required as $L$ grows.
- **Error Analysis**: Compares Trotter steps (Order 1 vs Order 2) for precision tracking.

---
*Created by Antigravity AI for Quantum Scattering Research.*
