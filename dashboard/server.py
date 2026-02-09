import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import quimb.tensor as qtn

# Import core QSL modules
from src.models.ising_1d import IsingModel1D
from src.models.su2 import SU2GaugeModel
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.simulation.initialization import prepare_two_wavepacket_state

app = FastAPI()

# Enable CORS for frontend
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
    model_type: str  # "ising_1d" or "su2_gauge"
    num_sites: int = 20
    g_x: float = 1.0  # Transverse field / Coupling g
    g_z: float = 0.0  # Longitudinal field / unused
    dt: float = 0.1
    steps: int = 50
    wp1: WavepacketParams
    wp2: WavepacketParams

@app.get("/")
def read_root():
    return {"status": "Quantum Scattering Lab API is running"}

@app.post("/simulate")
def simulate_scattering(params: SimulationParams):
    try:
        print(f"Starting simulation: {params.model_type}, L={params.num_sites}")
        
        # 1. Initialize Backend
        # Use MPS for scalability
        backend = QuimbMPSBackend(max_bond_dim=32)
        
        # 2. Initialize Model & Vacuum
        vac_e = np.zeros(params.num_sites)
        
        if params.model_type == "ising_1d":
            model = IsingModel1D(num_sites=params.num_sites, g_x=params.g_x, g_z=params.g_z, pbc=True)
            
            # Approximate vacuum |0...0> (Good for strong field)
            # For more accuracy, we could run imaginary time evolution here, 
            # but for GUI responsiveness, reference state is often acceptable or pre-calculated.
            psi_vac = backend.get_reference_state(params.num_sites)
            
            # If g_z is small/zero, |0> is not eigenstate of X term.
            # Let's do a quick imaginary time evolution if system is small?
            # Or just subtract expectation of reference.
            
            # Better: Calculate "Vacuum" energy density on the fly for subtraction
            # For Ising at g=1, vacuum is complex. 
            # Let's use the reference state expectation for now to keep it fast.
            # Or better: don't subtract if unsafe. 
            # Let's try to get a better vacuum if L is small.
            if params.num_sites <= 12:
                # Use dense exact diag for better vacuum
                from src.backends.quimb_backend import QuimbBackend
                dense_backend = QuimbBackend()
                H_dense = dense_backend._pauli_to_matrix(model.build_hamiltonian())
                from scipy.sparse.linalg import eigsh
                evals, evecs = eigsh(H_dense, k=1, which='SA')
                # But we need MPS state.
                # Skip perfect vacuum for speed in GUI demo.
                pass

        elif params.model_type == "su2_gauge":
            # Map parameters: g -> g_x
            model = SU2GaugeModel(num_sites=params.num_sites, g=params.g_x, a=1.0, pbc=True)
            psi_vac = backend.get_reference_state(params.num_sites)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {params.model_type}")

        # Compute vacuum energy profile for subtraction
        # (MPS computation)
        vac_e = []
        for n in range(params.num_sites):
             op = model.get_local_hamiltonian(n)
             vac_e.append(backend.compute_expectation_value(psi_vac, op))

        # 3. Prepare Initial State (Two Wavepackets)
        # Using the helper from src
        # Note: helper returns dense vector usually, need to convert to MPS
        psi_dense = prepare_two_wavepacket_state(
            params.num_sites,
            x1=params.wp1.x0, k1=params.wp1.k0 * np.pi, sigma1=params.wp1.sigma,
            x2=params.wp2.x0, k2=params.wp2.k0 * np.pi, sigma2=params.wp2.sigma,
            backend_type="numpy"
        )
        
        # Convert dense to MPS
        current_psi = qtn.MatrixProductState.from_dense(psi_dense, [2] * params.num_sites)
        current_psi.compress(max_bond=32)

        # 4. Use Modular Simulation Engine
        from src.simulation.scattering import ScatteringSimulator
        simulator = ScatteringSimulator(model, backend)
        # Set vacuum for subtraction manually or let it compute
        simulator.vac_energy_profile = vac_e 
        
        simulation_result = simulator.run(
            initial_state=current_psi,
            t_max=params.steps * params.dt,
            dt=params.dt,
            observables=["energy_density"]
        )
        
        return {
            "heatmap": simulation_result["energy_density"],
            "t_max": params.steps * params.dt,
            "sites": list(range(params.num_sites))
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
