"""
Scattering Simulation Engine
============================
Core logic for time-evolving wavepackets and extracting observables.
Designed to be backend-agnostic but optimized for MPS.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import quimb.tensor as qtn
from src.backends.quimb_mps_backend import QuimbMPSBackend
from src.models.base import PhysicsModel
import logging

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

    def set_vacuum_reference(self, method: str = "approximate"):
        """
        Compute vacuum energy profile for subtraction.
        """
        # For simplicity in this demo, use the reference state |00...0>
        # In production, use VQE or Imaginary Time Evolution
        psi_vac = self.backend.get_reference_state(self.model.num_sites)
        
        self.vac_energy_profile = []
        for n in range(self.model.num_sites):
             op = self.model.get_local_hamiltonian(n)
             val = self.backend.compute_expectation_value(psi_vac, op)
             self.vac_energy_profile.append(val)
             
        return self.vac_energy_profile

    def run(self, initial_state: qtn.MatrixProductState, t_max: float, dt: float, 
            observables: List[str] = ["energy_density"]) -> Dict[str, Any]:
        """
        Run time evolution and return trajectory.
        """
        steps = int(t_max / dt)
        results = {obs: [] for obs in observables}
        results["time"] = []
        
        current_psi = initial_state
        
        # Ensure vacuum is set
        if self.vac_energy_profile is None:
            self.set_vacuum_reference()
            
        for t in range(steps):
            # 1. Measure
            if "energy_density" in observables:
                row = []
                for n in range(self.model.num_sites):
                    op = self.model.get_local_hamiltonian(n)
                    val = self.backend.compute_expectation_value(current_psi, op)
                    row.append(float(val - self.vac_energy_profile[n]))
                results["energy_density"].append(row)
                
            if "entropy" in observables:
                # Expensive! Maybe only measure every k steps?
                # For now, skip to keep GUI fast.
                pass
                
            results["time"].append(t * dt)
            
            # 2. Evolve
            current_psi = self.backend.evolve_state_trotter(current_psi, self.layers, dt)
            
        return results
