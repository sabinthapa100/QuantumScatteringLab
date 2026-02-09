from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(".."))

from src.models.ising_1d import IsingModel1D
from src.models.ising_2d import IsingModel2D
from src.models.su2 import SU2GaugeModel
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state, prepare_two_wavepacket_state_2d
import quimb.tensor as qtn

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class WavepacketParams(BaseModel):
    x0: float
    k0: float
    sigma: float

class SimParams(BaseModel):
    model: str
    num_sites: int
    g_x: float
    g_z: float
    dt: float
    steps: int
    wp1: WavepacketParams
    wp2: WavepacketParams

@app.post("/simulate")
async def simulate(params: SimParams):
    print(f"Received simulation request for {params.model}")
    
    try:
        if params.model == "1d_ising":
            model = IsingModel1D(num_sites=params.num_sites, g_x=params.g_x, g_z=params.g_z, pbc=True)
            backend = QuimbMPSBackend(max_bond_dim=32)
            
            # Preparation
            psi_np = prepare_two_wavepacket_state(
                params.num_sites,
                x1=params.wp1.x0, k1=params.wp1.k0 * np.pi, sigma1=params.wp1.sigma,
                x2=params.wp2.x0, k2=params.wp2.k0 * np.pi, sigma2=params.wp2.sigma,
                backend_type="numpy"
            )
            current_psi = qtn.MatrixProductState.from_dense(psi_np, [2]*params.num_sites)
            
            # Vacuum subtraction
            psi_vac = backend.get_reference_state(params.num_sites)
            vac_e = [backend.compute_expectation_value(psi_vac, model.get_local_hamiltonian(n)) for n in range(params.num_sites)]
            
            # Evolution
            heatmap = []
            layers = model.get_trotter_layers()
            for i in range(params.steps):
                row = []
                for n in range(params.num_sites):
                    val = backend.compute_expectation_value(current_psi, model.get_local_hamiltonian(n))
                    row.append(float(val - vac_e[n]))
                heatmap.append(row)
                current_psi = backend.evolve_state_trotter(current_psi, layers, params.dt)
                
            return {"heatmap": heatmap}

        elif params.model == "2d_ising":
            Lx = int(np.sqrt(params.num_sites))
            Ly = params.num_sites // Lx
            model = IsingModel2D(Lx=Lx, Ly=Ly, g_x=params.g_x, g_z=params.g_z, pbc=True)
            backend = QuimbMPSBackend(max_bond_dim=32)
            
            # 2D Wavepackets
            # We use Lx/2, Ly/2 centers
            psi_np = prepare_two_wavepacket_state_2d(
                Lx, Ly,
                x1=Lx/4, y1=Ly/2, kx1=params.wp1.k0 * np.pi, ky1=0.0, sigma1=params.wp1.sigma,
                x2=3*Lx/4, y2=Ly/2, kx2=params.wp2.k0 * np.pi, ky2=0.0, sigma2=params.wp2.sigma,
                backend_type="numpy"
            )
            current_psi = qtn.MatrixProductState.from_dense(psi_np, [2]*params.num_sites)
            
            # Vacuum
            psi_vac = backend.get_reference_state(params.num_sites)
            vac_e = {}
            for ny in range(Ly):
                for nx in range(Lx):
                    vac_e[(nx, ny)] = backend.compute_expectation_value(psi_vac, model.get_local_hamiltonian(nx, ny))
            
            # Evolution (Return 2D averaged rows for heatmap simplicity or full grid)
            heatmap = []
            layers = model.get_trotter_layers()
            for i in range(params.steps):
                if i % 2 == 0:
                    row = []
                    # Average over Y to show a 1D-like projection or just return a row?
                    # Let's return the middle row of X to keep heatmap visualization consistent
                    mid_y = Ly // 2
                    for nx in range(Lx):
                        val = backend.compute_expectation_value(current_psi, model.get_local_hamiltonian(nx, mid_y))
                        row.append(float(val - vac_e[(nx, mid_y)]))
                    heatmap.append(row)
                current_psi = backend.evolve_state_trotter(current_psi, layers, params.dt)
            return {"heatmap": heatmap}

        elif params.model == "su2_gauge":
            # Similar logic for SU2
            model = SU2GaugeModel(num_sites=params.num_sites, g=params.g_x, pbc=True)
            backend = QuimbMPSBackend(max_bond_dim=32)
            psi_np = prepare_two_wavepacket_state(
                params.num_sites,
                x1=params.wp1.x0, k1=params.wp1.k0 * np.pi, sigma1=params.wp1.sigma,
                x2=params.wp2.x0, k2=params.wp2.k0 * np.pi, sigma2=params.wp2.sigma,
                backend_type="numpy"
            )
            current_psi = qtn.MatrixProductState.from_dense(psi_np, [2]*params.num_sites)
            psi_vac = backend.get_reference_state(params.num_sites)
            vac_e = [backend.compute_expectation_value(psi_vac, model.get_local_hamiltonian(n)) for n in range(params.num_sites)]
            
            heatmap = []
            layers = model.get_trotter_layers()
            # Run fewer steps or skip to save time in UI demo
            for i in range(params.steps):
                if i % 2 == 0: # Downsample
                    row = [float(backend.compute_expectation_value(current_psi, model.get_local_hamiltonian(n)) - vac_e[n]) for n in range(params.num_sites)]
                    heatmap.append(row)
                current_psi = backend.evolve_state_trotter(current_psi, layers, params.dt)
            return {"heatmap": heatmap}

        else:
            raise HTTPException(status_code=400, detail="Model not yet supported in GUI")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
