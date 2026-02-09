# ⚛️ QuantumScatteringLab GUI Documentation

This document provides detailed information on the newly implemented "Senior Scientist" GUI dashboard for the QuantumScatteringLab.

## 🛠️ Installation & Setup

The GUI is designed to be self-contained within the `.agents/` directory to preserve the integrity of the core project.

### 1. Prerequisites
Ensure you have Python 3.12+ installed. The GUI uses a dedicated virtual environment to manage its scientific dependencies (`streamlit`, `plotly`, `pandas`, `scipy`).

### 2. Environment Activation
```bash
# From the project root
source .agents/venv/bin/activate
```

### 3. Running the App
```bash
streamlit run .agents/gui_app.py
```

---

## 🚀 Feature Set

### 1. Hardware-Aware Backend
- **CPU Mode**: Uses standard sparse matrix solvers for smaller systems.
- **GPU Mode (High Performance)**: Leverages `Quimb` with `cupy` for Matrix Product State (MPS) simulations. This is essential for large system sizes where exact diagonalization becomes prohibitive.

### 2. Model Laboratory
- **Flexible Parameters**: Every physical parameter (coupling $g$, transverse field $g_x$, etc.) is exposed via the sidebar.
- **Automatic Scaling**: The UI dynamically adjusts its control panels based on the selected model (`Ising`, `Heisenberg`, or `SU2 Gauge`).

### 3. Integrated Analysis Suites
- **Spectroscopy**: Perform high-resolution sweeps to find phase transition points.
- **Quantum Information**: Automatically compute entanglement entropy and the Central Charge—a fundamental metric in conformal field theory (CFT).
- **Criticality Mapping**: Use finite-size scaling to collapse data from different system sizes into a single universal curve.

---

## 📈 Future Improvements & Roadmap

The current implementation is a high-fidelity prototype. Based on "Senior Scientist" requirements, the following enhancements are proposed:

### 🔬 Physics & Algorithms
- **TMRG/DMRG Integration**: Direct support for Density Matrix Renormalization Group (DMRG) via `quimb` for ground state search on lattices > 50 sites.
- **Time Evolution Panel**: Add a dedicated tab for real-time dynamics (Trotterization visualization).
- **Variational Circuits**: Visualize the ADAPT-VQE ansatz expansion in real-time.

### 🍱 UI & UX
- **Persistence Layer**: Implement a database (SQLite) in `.agents/db/` to store simulation history for cross-session comparisons.
- **3D Lattice Viewer**: Interactive 3D visualization of the spin-chain or SU(2) plaquette configurations.
- **Parallel Sweeps**: Use `multiprocessing` to run multi-parameter sweeps in parallel on the CPU backend.

### 📤 Export & Reporting
- **LaTeX Integration**: Generate ready-to-use LaTeX tables for computed spectral data.
- **Interactive Reports**: Export the entire dashboard state as a standalone HTML scientific report.
