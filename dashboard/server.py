import sys
import os
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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
    model: str
    num_sites: int = 20
    g_x: float = 1.0
    g_z: float = 0.0
    dt: float = 0.1
    steps: int = 50
    wp1: WavepacketParams
    wp2: WavepacketParams
    wp3: Optional[WavepacketParams] = None
    # Model-specific params (optional)
    g: Optional[float] = None  # For SU(2) gauge coupling
    a: Optional[float] = None  # For SU(2) lattice spacing
    Lx: Optional[int] = None   # For 2D Ising
    Ly: Optional[int] = None   # For 2D Ising

@app.get("/")
def root():
    return {"status": "QSL API running"}

def analyze_s_matrix(heatmap: List[List[float]]) -> dict:
    """
    Very basic S-matrix analysis based on energy density flows.
    """
    data = np.array(heatmap)
    initial = data[0]
    final = data[-1]
    
    mid = len(initial) // 2
    # Probability in left/right halves
    trans = np.sum(final[mid:]) / (np.sum(initial) + 1e-12)
    refl = np.sum(final[:mid]) / (np.sum(initial) + 1e-12)
    
    return {
        "transmission": float(trans),
        "reflection": float(refl),
        "inelasticity": float(max(0, 1.0 - (trans + refl)))
    }

@app.get("/model-info")
def get_model_info(model_type: str):
    """Return LaTeX string and available parameters for the Hamiltonian."""
    if model_type == "ising_1d":
        return {
            "name": "1D Transverse Field Ising Model",
            "latex": r"H = -\sum_i \left[\frac{1}{2} Z_i Z_{i+1} + g_x X_i + g_z Z_i\right]",
            "description": "1D spin chain with nearest-neighbor interactions.",
            "parameters": ["num_sites", "g_x", "g_z", "pbc"]
        }
    elif model_type == "ising_2d":
        return {
            "name": "2D Transverse Field Ising Model",
            "latex": r"H = -\sum_{\langle i,j \rangle} \left[\frac{1}{2} Z_i Z_j + g_x X_i + g_z Z_i\right]",
            "description": "Square lattice mapped to 1D chain (snake pattern).",
            "parameters": ["Lx", "Ly", "g_x", "g_z", "pbc"]
        }
    elif model_type == "su2_gauge":
        return {
            "name": "SU(2) Lattice Gauge Theory",
            "latex": r"H = J\sum_i Z_i Z_{i+1} + h_z\sum_i Z_i + \frac{h_x}{16}\sum_i \left(X_i - 3Z_{i-1}X_i - 3X_iZ_{i+1} + 9Z_{i-1}X_iZ_{i+1}\right)",
            "description": f"Kogut-Susskind formulation. J=-3g²/16, h_z=3g²/8, h_x=-2/(ag)²",
            "parameters": ["num_sites", "g", "a", "pbc"]
        }
    else:
        return {"latex": "H = ?", "name": "Unknown", "parameters": []}

@app.post("/simulate")
def simulate_scattering(params: SimulationParams):
    try:
        backend = QuimbMPSBackend(max_bond_dim=64)
        
        if params.model == "ising_1d":
            print(f"[1D ISING] L={params.num_sites}, g_x={params.g_x}, g_z={params.g_z}")
            model = IsingModel1D(num_sites=params.num_sites, g_x=params.g_x, g_z=params.g_z, pbc=True)
        elif params.model == "ising_2d":
            # Use explicit Lx, Ly if provided, otherwise infer from num_sites
            if params.Lx and params.Ly:
                Lx, Ly = params.Lx, params.Ly
            else:
                import math
                L = params.num_sites
                Lx = int(math.sqrt(L))
                Ly = L // Lx
            print(f"[2D ISING] Lx={Lx}, Ly={Ly}, g_x={params.g_x}, g_z={params.g_z}")
            model = IsingModel2D(Lx=Lx, Ly=Ly, g_x=params.g_x, g_z=params.g_z, pbc=False)
        elif params.model == "su2_gauge":
            # Use g and a if provided, otherwise defaults
            g_coupling = params.g if params.g is not None else 1.0
            lattice_spacing = params.a if params.a is not None else 1.0
            print(f"[SU(2) GAUGE] L={params.num_sites}, g={g_coupling}, a={lattice_spacing}")
            model = SU2GaugeModel(num_sites=params.num_sites, g=g_coupling, a=lattice_spacing, pbc=True)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {params.model}")

        # 1. Vacuum Preparation (High Precision via DMRG)
        psi_vac = backend.get_ground_state(model)
        
        # 2. Local Energy Profile (Vacuum subtraction baseline)
        vac_e = [backend.compute_expectation_value(psi_vac, model.get_local_hamiltonian(n)) 
                 for n in range(model.num_sites)]

        # 3. Wavepacket Initialization (Direct MPS, no 2^N)
        if params.wp3:
            from src.simulation.initialization import prepare_three_wavepacket_mps
            wps = [
                {"x0": params.wp1.x0, "k0": params.wp1.k0 * np.pi, "sigma": params.wp1.sigma},
                {"x0": params.wp2.x0, "k0": params.wp2.k0 * np.pi, "sigma": params.wp2.sigma},
                {"x0": params.wp3.x0, "k0": params.wp3.k0 * np.pi, "sigma": params.wp3.sigma},
            ]
            current_psi = prepare_three_wavepacket_mps(model.num_sites, wps)
        else:
            current_psi = prepare_two_wavepacket_state(
                model.num_sites,
                x1=params.wp1.x0, k1=params.wp1.k0 * np.pi, sigma1=params.wp1.sigma,
                x2=params.wp2.x0, k2=params.wp2.k0 * np.pi, sigma2=params.wp2.sigma,
                backend_type="mps"
            )
        
        # 4. Evolution
        from src.simulation.scattering import ScatteringSimulator
        simulator = ScatteringSimulator(model, backend)
        simulator.vac_energy_profile = vac_e 
        
        def progress_callback(msg: str):
            print(f"[{params.model.upper()}] {msg}")
            
        result = simulator.run(
            initial_state=current_psi,
            t_max=params.steps * params.dt,
            dt=params.dt,
            observables=["energy_density", "entropy"],
            callback=progress_callback
        )
        
        analysis = analyze_s_matrix(result["energy_density"])
        
        return {
            "heatmap": result["energy_density"],
            "entropy": result["entropy"],
            "analysis": analysis,
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
