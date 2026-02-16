# ⚛️ Quantum Scattering Lab (QSL)

A state-of-the-art simulation framework for investigating **Inelastic Scattering**, **Particle Production**, and **Non-Integrable Dynamics** using Matrix Product States (MPS).

---

## ⚡ Performance Metrics (NVIDIA RTX 4070)

The engine is mathematically optimized for $O(L)$ scaling across all measurement phases. 

| Phase | Legacy Engine ($O(L^2)$) | Optimized Engine ($O(L)$) | Speedup |
| :--- | :--- | :--- | :--- |
| **Evolution (L=128, D=128)** | 53.9s / iter | **0.82s / iter** | **~65x** |
| **Measurement (L=128)** | O(L^2) | **O(L) (Canonical)** | $\gg 100x$ |
| **Vacuum Preparation** | Manual SVD | **Hybrid CPU-DMRG** | Stability+ |

---

## 🛠️ The Unified Production Workflow

Everything is driven by the production CLI at `scripts/run_production_scattering.py`.

### 1. Landscape Analysis (`--mode spectrum`)
Map the dispersion relation to identify the mass gap $M$ and particle sectors.

### 2. Ground State Preparation (`--mode vqe`)
Construct the interacting vacuum via the **ADAPT-VQE** solver.
```bash
python scripts/run_production_scattering.py --mode vqe --L 128 --gpu
```

### 3. Production Scattering Run (`--mode scatter`)
Launch relativistic wavepackets and record energy density trajectories.
```bash
python scripts/run_production_scattering.py --mode scatter --L 128 --bond-dim 128 --gpu
```

---

## 📖 Essential Documentation
- **[Installation & Quick Start](docs/GETTING_STARTED.md)** – Run your first simulation in 5 minutes.
- **[Scientific Usage & Math](docs/SCIENTIFIC_USAGE.md)** – Derivations of the $O(L)$ optimization and Trotterization metrics.
- **[Testing & Calibration](docs/TEST_GUIDE.md)** – Ensure the backend is mathematically verified.

---

## 🏗️ Technical Architecture

- **`src/models/`**: Rigorous definitions of $H$ with support for $g_z$ (confinement) and $j_{int}$ (coupling).
- **`src/backends/`**: Specialized `QuimbMPSBackend` with deferred compression and CPU-to-GPU hybrid stability.
- **`src/simulation/`**: Trotterized evolution and ADAPT-VQE solvers.
- **`dashboard/`**: React + FastAPI dashboard for visual data inspection (Legacy/Interactive).

---

## 🚀 Quick Start (Development)

```bash
# Setup Environment
pip install -e ".[all]"

# Run Validation
pytest tests/test_backends.py

# Generate Sample Data
python examples/scripts/generate_gui_data.py

# Launch Dashboard
python run_lab.py
```

---

## 📊 Data Hygiene
Large binary outputs (`*.npy`, `*.png`, `data/`) are strictly excluded from Git tracking to prevent repository bloat. Production results are stored in `data/production/` for local analysis.

---
**Author**: Sabin Thapa  
**Status**: Production Ready (Optimized for Large-Scale Research)
