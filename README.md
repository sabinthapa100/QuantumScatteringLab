# ⚛️ Quantum Scattering Lab (QSL)

A professional research-grade simulation framework for exploring **Inelastic Scattering** and **Particle Production** in Quantum Field Theories ($1+1D$ Ising and $SU(2)$ LGT) using Matrix Product States (MPS).

## 📖 Quick Links
- **[Getting Started (Installation & Usage)](docs/GETTING_STARTED.md)** – Run your first simulation in 5 minutes.
- **[Testing Guide](docs/TEST_GUIDE.md)** – Ensure everything is calibrated correctly.
- **[Physics Architecture](docs/RESEARCH_GUIDE.md)** – Detailed derivations of Hamiltonians and backends.

## 🏗️ Project Architecture
The codebase is structured for scalability and reproducibility:

- **`src/`**: The Core Engine.
    - **`models/`**: Rigorous Hamiltonian definitions (`IsingModel1D`, `IsingModel2D`, `SU2GaugeModel`) with explicit symmetry sectors.
    - **`backends/`**: 
        - `QuimbMPSBackend`: Optimized Tensor Network backend for $L > 20$ sites.
        - `QiskitBackend`: Exact statevector validation.
    - **`simulation/`**: Trotterized time-evolution and ADAPT-VQE solvers.
    
- **`examples/`**: Research scripts for scattering and ground state preparation.
- **`dashboard/`**: Custom-built **React + FastAPI** dashboard to visualize energy density evolution in real-time.

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
