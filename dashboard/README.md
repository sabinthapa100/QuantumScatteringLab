# Quantum Scattering Dashboard

Interactive visualization interface for the Quantum Scattering Lab (QSL). This dashboard provides real-time control and visualization of quantum field simulations, leveraging the project's optimized tensor network backend.

## Architecture

- **Frontend**: React + Vite (Fast HMR) using `recharts` and `plotly.js` or `canvas` for heatmaps.
- **Backend**: FastAPI (`server.py`) wrapping the QSL core engine (`src` package).
- **Communication**: REST API for simulation control and data polling.

## Quick Start

### 1. Prerequisites
Ensure you have the Python environment set up at the project root.

```bash
cd ../
source .venv/bin/activate
pip install fastapi uvicorn
```

### 2. Start Simulation Server
The Python backend handles the heavy lifting (Trotter evolution, Tensor contractions).

```bash
# Inside dashboard/ directory
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Visualization Client
In a new terminal:

```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to access the dashboard.

## Features

- **Real-time Heatmaps**: Visualize energy density evolution $|\langle \psi(t) | H_x | \psi(t) \rangle - E_{vac}|$.
- **Parameter Control**: Adjust couplings $g_x, g_z$ and time step $dt$ on the fly.
- **Wavepacket Builder**: Configure initial momentum $k$ and position $x_0$ for scattering experiments.
