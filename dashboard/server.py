import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import quimb.tensor as qtn

# Import core QSL modules
from src.models.ising_2d import IsingModel2D

@app.get("/model-info")
def get_model_info(model_type: str):
    """Return LaTeX string for the Hamiltonian."""
    if model_type == "ising_1d":
        return {
            "name": "1D Transverse Field Ising Model",
            "latex": r"H = -J \sum_{i} Z_i Z_{i+1} - g_x \sum_{i} X_i - g_z \sum_{i} Z_i",
            "description": "Standard 1D spin chain with nearest-neighbor interactions."
        }
    elif model_type == "ising_2d":
        return {
            "name": "2D Transverse Field Ising Model",
            "latex": r"H = -J \sum_{\langle i,j \rangle} Z_i Z_j - g_x \sum_{i} X_i - g_z \sum_{i} Z_i",
            "description": "Square lattice model mapped to 1D via snake-path ordering."
        }
    elif model_type == "su2_gauge":
        return {
            "name": "SU(2) Lattice Gauge Theory",
            "latex": r"H = \frac{g^2}{2} \sum_{l} E_l^2 - \frac{1}{2g^2} \sum_{p} (\text{Tr } U_p + h.c.)",
            "description": "Kogut-Susskind Hamiltonian on a 1D plaquette chain (dual formulation)."
        }
    else:
        return {"latex": "H = ?", "name": "Unknown"}

@app.post("/simulate")
def simulate_scattering(params: SimulationParams):
    try:
        print(f"Starting simulation: {params.model_type}, L={params.num_sites}")
        
        # 1. Initialize Backend
        backend = QuimbMPSBackend(max_bond_dim=48)
        
        # 2. Initialize Model & Vacuum
        vac_e = np.zeros(params.num_sites)
        model = None
        
        if params.model_type == "ising_1d":
            model = IsingModel1D(num_sites=params.num_sites, g_x=params.g_x, g_z=params.g_z, pbc=True)
            psi_vac = backend.get_reference_state(params.num_sites)
            
        elif params.model_type == "ising_2d":
            # Map Lx * Ly = num_sites. Try to find square-ish factors
            import math
            L = params.num_sites
            Lx = int(math.sqrt(L))
            Ly = L // Lx
            # Adjust L to match rectangle
            real_L = Lx * Ly
            if real_L != L:
                print(f"Adjusting 2D Lattice to {Lx}x{Ly} = {real_L}")
            
            model = IsingModel2D(Lx=Lx, Ly=Ly, g_x=params.g_x, g_z=params.g_z, pbc=False)
            psi_vac = backend.get_reference_state(real_L)
            
            # Update params sites for return
            params.num_sites = real_L

        elif params.model_type == "su2_gauge":
            model = SU2GaugeModel(num_sites=params.num_sites, g=params.g_x, a=1.0, pbc=True)
            psi_vac = backend.get_reference_state(params.num_sites)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {params.model_type}")

        # Compute vacuum energy profile for subtraction
        vac_e = []
        # Note: calling get_local_hamiltonian for 2D might be slow if loop is large
        # But for L=20 it's fast.
        for n in range(model.num_sites):
             op = model.get_local_hamiltonian(n)
             vac_e.append(backend.compute_expectation_value(psi_vac, op))

        # 3. Prepare Initial State (Two Wavepackets)
        psi_dense = prepare_two_wavepacket_state(
            model.num_sites,
            x1=params.wp1.x0, k1=params.wp1.k0 * np.pi, sigma1=params.wp1.sigma,
            x2=params.wp2.x0, k2=params.wp2.k0 * np.pi, sigma2=params.wp2.sigma,
            backend_type="numpy"
        )
        
        # Convert dense to MPS
        current_psi = qtn.MatrixProductState.from_dense(psi_dense, [2] * model.num_sites)
        current_psi.compress(max_bond=48)

        # 4. Use Modular Simulation Engine
        from src.simulation.scattering import ScatteringSimulator
        simulator = ScatteringSimulator(model, backend)
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
            "sites": list(range(model.num_sites)),
            "real_shape": f"{Lx}x{Ly}" if params.model_type == "ising_2d" else f"{model.num_sites}x1"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return simple string error to frontend to avoid [object Object] confusion
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
