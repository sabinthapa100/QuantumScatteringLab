# Getting Started: Quantum Scattering Lab

Welcome to the Quantum Scattering Lab! This guide will take you from scratch to running your first quantum scattering simulation.

## 🛠️ Installation from Scratch

1. **Clone the Repository** (If you haven't already):
   ```bash
   git clone <repo-url>
   cd quantumscatteringlab
   ```

2. **Setup Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

4. **Install Dashboard Dependencies**:
   ```bash
   cd dashboard
   npm install
   cd ..
   ```

## 🚀 Running Your First Simulation

### Option A: The Interactive Dashboard (Recommended)
Launch the full stack with one command:
```bash
python3 run_lab.py
```
*Go to http://localhost:5173 to play with sliders and see scattering in real-time.*

### Option B: High-Performance CLI (For Production)
Run a script that generates a publication-quality figure:
```bash
python3 examples/scattering/01_ising_scattering_mps.py
```
*Results will be saved in the `results/` folder.*

## 📖 Key Concepts
- **Ising 1D/2D**: Models spin chains and lattices. $g_x$ controls quantum fluctuations.
- **SU(2) Gauge Theory**: Simulates gluons on a lattice.
- **Wavepackets**: Particles with momentum $k_0$ and width $\sigma$.
- **Evolution**: We use Trotter decomposition and Matrix Product States (MPS) for scalability.

---
*For deep technical details, see the paper: [arXiv:2505.03111](https://arxiv.org/abs/2501.03111)*
