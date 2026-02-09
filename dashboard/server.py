import sys
import os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import quimb.tensor as qtn

from src.models.ising_1d import IsingModel1D
from src.models.ising_2d import IsingModel2D
from src.models.su2 import SU2GaugeModel
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WavepacketParams(BaseModel):
    x0: float
    k0: float
    sigma: float

class SimulationParams(BaseModel):
    model_type: str
    num_sites: int = 20
    g_x: float = 1.0
    g_z: float = 0.0
    dt: float = 0.1
    steps: int = 50
    wp1: WavepacketParams
    wp2: WavepacketParams

@app.get("/")
def root():
    return {"status": "QSL API running"}

@app.get("/model-info")
def get_model_info(model_type: str):
    """Return LaTeX string for the Hamiltonian."""
    if model_type == "ising_1d":
        return {
            "name": "1D Transverse Field Ising Model",
            "latex": r"H = -J \sum_{i} Z_i Z_{i+1} - g_x \sum_{i} X_i - g_z \sum_{i} Z_i",
            "description": "1D spin chain with nearest-neighbor interactions."
        }
    elif model_type == "ising_2d":
        return {
            "name": "2D Transverse Field Ising Model",
            "latex": r"H = -J \sum_{\langle i,j \rangle} Z_i Z_j - g_x \sum_{i} X_i - g_z \sum_{i} Z_i",
            "description": "Square lattice mapped to 1D chain."
        }
    elif model_type == "su2_gauge":
        return {
            "name": "SU(2) Lattice Gauge Theory",
            "latex": r"H = \frac{g^2}{2} \sum_{l} E_l^2 - \frac{1}{2g^2} \sum_{p} (\mathrm{Tr} \, U_p + \mathrm{h.c.})",
            "description": "Kogut-Susskind on 1D plaquette chain."
        }
    else:
        return {"latex": "H = ?", "name": "Unknown"}

@app.post("/simulate")
def simulate_scattering(params: SimulationParams):
    try:
        print(f"Simulating: {params.model_type}, L={params.num_sites}, gx={params.g_x}, gz={params.g_z}")
        
        backend = QuimbMPSBackend(max_bond_dim=64)
        
        if params.model_type == "ising_1d":
            model = IsingModel1D(num_sites=params.num_sites, g_x=params.g_x, g_z=params.g_z, pbc=True)
        elif params.model_type == "ising_2d":
            import math
            L = params.num_sites
            Lx = int(math.sqrt(L))
            Ly = L // Lx
            model = IsingModel2D(Lx=Lx, Ly=Ly, g_x=params.g_x, g_z=params.g_z, pbc=False)
        elif params.model_type == "su2_gauge":
            model = SU2GaugeModel(num_sites=params.num_sites, g=params.g_x, a=1.0, pbc=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {params.model_type}")

        # Vacuum reference
        psi_vac = backend.get_reference_state(model.num_sites)
        vac_e = [backend.compute_expectation_value(psi_vac, model.get_local_hamiltonian(n)) 
                 for n in range(model.num_sites)]

        # Initial state
        psi_dense = prepare_two_wavepacket_state(
            model.num_sites,
            x1=params.wp1.x0, k1=params.wp1.k0 * np.pi, sigma1=params.wp1.sigma,
            x2=params.wp2.x0, k2=params.wp2.k0 * np.pi, sigma2=params.wp2.sigma,
            backend_type="numpy"
        )
        
        current_psi = qtn.MatrixProductState.from_dense(psi_dense, [2] * model.num_sites)
        current_psi.compress(max_bond=64)

        # Evolution
        from src.simulation.scattering import ScatteringSimulator
        simulator = ScatteringSimulator(model, backend)
        simulator.vac_energy_profile = vac_e 
        
        result = simulator.run(
            initial_state=current_psi,
            t_max=params.steps * params.dt,
            dt=params.dt,
            observables=["energy_density"]
        )
        
        return {
            "heatmap": result["energy_density"],
            "t_max": params.steps * params.dt,
            "num_sites": model.num_sites
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
