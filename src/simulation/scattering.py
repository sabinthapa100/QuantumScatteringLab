"""
Scattering Simulation Engine
============================
Core logic for time-evolving wavepackets and extracting observables.
Designed to be backend-agnostic but optimized for MPS.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import quimb.tensor as qtn
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.models.base import PhysicsModel
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ScatteringSimulator:
    """
    Orchestrates the scattering experiment:
    1. Prepare Initial State (Wavepackets)
    2. Evolve in Time (Trotter)
    3. Measure Observables (Energy Density, Entropy)
    """
    
    def __init__(self, model: PhysicsModel, backend: QuimbMPSBackend):
        self.model = model
        self.backend = backend
        self.layers = model.get_trotter_layers()
        self.vac_energy_profile = None

    def set_vacuum_reference(self, method: str = "dmrg"):
        """
        Compute vacuum energy profile for subtraction.
        """
        logger.info(f"Computing vacuum reference using {method}...")
        if method == "dmrg":
            psi_vac = self.backend.get_ground_state(self.model)
        else:
            psi_vac = self.backend.get_reference_state(self.model.num_sites)
        
        # Performance optimization: Canonicalize once!
        psi_vac.canonize(0)
        
        self.vac_energy_profile = []
        for n in range(self.model.num_sites):
             op = self.model.get_local_hamiltonian(n)
             val = self.backend.compute_expectation_value(psi_vac, op)
             self.vac_energy_profile.append(val)
             
        return self.vac_energy_profile

    def measure_energy_density(self, state: Any) -> np.ndarray:
        """
        Measure local energy density at each site, subtracted by vacuum.
        Highly optimized O(L) implementation utilizing MPS canonical forms.
        """
        if self.vac_energy_profile is None:
            self.set_vacuum_reference(method="dmrg")
            
        L = self.model.num_sites
        energy_densities = np.zeros(L)
        
        # Performance optimization: Canonicalize once!
        # This makes subsequent local_expectation(site) O(D^3) instead of O(L*D^3)
        state.canonize(0)
        
        for n in range(L):
            op = self.model.get_local_hamiltonian(n)
            # compute_expectation_value now benefits from the canonized state
            val = self.backend.compute_expectation_value(state, op)
            energy_densities[n] = float(val - self.vac_energy_profile[n])
            
        return energy_densities

    def run(self, initial_state: Any, t_max: float, dt: float, 
            observables: List[str] = ["energy_density"], 
            progress_bar: bool = False,
            return_final_state: bool = False) -> Union[Dict[str, List[float]], Tuple[Dict[str, List[float]], Any]]:
        """
        Run time evolution and return trajectory.
        """
        num_steps = int(t_max / dt)
        results = {obs: [] for obs in observables}
        results["time"] = []
        
        current_psi = initial_state
        
        # Ensure vacuum is set
        if self.vac_energy_profile is None:
            self.set_vacuum_reference(method="dmrg")
            
        iterator = range(num_steps)
        if progress_bar:
            iterator = tqdm(iterator, desc="Time Evolution")
            
        current_time = 0.0
        for _ in iterator:
            # 1. Evolve (Evolution happens first in some conventions, here it doesn't matter much)
            current_psi = self.backend.evolve_state_trotter(current_psi, self.layers, dt)
            current_time += dt
            
            # 2. Measure
            if "energy_density" in observables:
                row = self.measure_energy_density(current_psi)
                results["energy_density"].append(row.tolist())
                
            results["time"].append(current_time)
            
        if return_final_state:
            return results, current_psi
        return results
