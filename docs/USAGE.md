# Quantum Scattering Lab: Usage Guide

This guide explains how to use the full capabilities of the lab, from high-performance CLI production runs to interactive dashboard visualization.

## 1. 🚀 The Easy Way: Interactive Dashboard

We have built a unified launcher that starts both the Physics Engine (FastAPI) and the Visualization Client (React).

**Command:**
```bash
python3 run_lab.py
```
*This will open http://localhost:5173 in your browser automatically.*

### Dashboard Features
- **Model Selection**: Choose between Ising 1D (Spin Chain) and SU(2) Gauge Theory (Plaquette Chain).
- **Parameters**:
  - `g_x`: Transverse field or Coupling strength.
  - `g_z`: Longitudinal field (Simulates confinement/inelasticity).
  - `L`: System size (Sites).
- **Wavepackets**: Configure initial momentum `k` and position `x0` for scattering particles.
- **Visualization**: Real-time Heatmap of Energy Density ($ \langle H_x \rangle - E_{vac} $).

---

## 2. 🧪 The Pro Way: Production Scripts

For publication-quality data generation and analysis, use the specialized scripts in `examples/`. These scripts use the scalable MPS backend (`QuimbMPSBackend`) capable of simulating 100+ sites.

### Experiment A: Elastic vs Inelastic Scattering (Ising 1D)
Compare integrable vs non-integrable dynamics.
```bash
python3 examples/scattering/01_ising_scattering_mps.py
```
- **Output**: `results/scattering/01_ising_comparison.png`
- **Key Physics**: Observe particle production when $g_z > 0$.

### Experiment B: SU(2) Glueball Scattering
Simulate collision of magnetic flux excitations in a Lattice Gauge Theory.
```bash
python3 examples/scattering/03_su2_scattering_mps.py
```
- **Output**: `results/su2/scattering_map.png`
- **Key Physics**: Observe meson-like bound states (glueballs).

### Experiment C: Entanglement Entropy Diagnostics
Analyze the growth of Von Neumann entropy during collision to verify Area Law violations.
```bash
python3 examples/physics_analysis/08_advanced_diagnostics.py
```
- **Output**: `results/diagnostics/entropy_map.png`
- **Key Physics**: Volume law scaling after quench/collision.

---

## 3. 🛠️ Development & Extension

### Modifying the Physics Model
Edit `src/models/*.py`. All models inherit from `PhysicsModel` and must implement `build_hamiltonian()` and `get_local_hamiltonian(i)`.

### Adding a New Backend
Implement the `QuantumBackend` interface in `src/backends/base.py`. Currently supported:
- `QiskitBackend`: Exact statevector (for small N).
- `QuimbMPSBackend`: Matrix Product States (for large N).

### Custom Simulation Logic
Use the modular `ScatteringSimulator` in your own scripts:
```python
from src.simulation.scattering import ScatteringSimulator
from src.backends.quimb_mps_backend import QuimbMPSBackend

backend = QuimbMPSBackend()
sim = ScatteringSimulator(model, backend)
results = sim.run(psi_init, t_max=10, dt=0.1)
```
