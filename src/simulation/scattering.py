"""
Scattering Simulation Module
============================

Simulate wavepacket dynamics and scattering in 1D quantum systems.
- Gaussian wavepacket initialization.
- Time evolution using Trotterization.
- Analysis of transmission and reflection coefficients.
"""

import numpy as np
from typing import Any, List, Optional, Tuple
from qiskit.quantum_info import SparsePauliOp

class ScatteringSimulator:
    def __init__(self, model: Any, backend: Any):
        self.model = model
        self.backend = backend
        self.num_sites = model.num_sites

    def create_gaussian_wavepacket(
        self,
        x0: float,
        sigma: float,
        k0: float,
        excitation_op: str = "X"
    ) -> Any:
        """
        Create a Gaussian wavepacket of single-site excitations.
        psi = sum_x exp(i k0 x) exp(-(x-x0)^2 / (2 sigma^2)) |x>
        where |x> = Op_x |0>
        """
        # 1. Create coefficients
        x = np.arange(self.num_sites)
        coeffs = np.exp(1j * k0 * x) * np.exp(-(x - x0)**2 / (2 * sigma**2))
        coeffs /= np.linalg.norm(coeffs) # Normalize
        
        # 2. Construct state in the full Hilbert space
        # We start with the reference state (vacuum)
        vacuum = self.backend.get_reference_state(self.num_sites)
        
        # We need to construct the state as a superposition of Op_x |vac>
        # This is psi = (sum_x coeffs[x] Op_x) |vac>
        
        # Build the creator operator
        creator_terms = []
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites
            p[i] = excitation_op
            creator_terms.append(("".join(reversed(p)), coeffs[i]))
            
        creator_op = SparsePauliOp.from_list(creator_terms)
        
        # Apply the creator operator to the vacuum
        # apply_operator applies exp(i theta P). We want to apply P directly?
        # QuimbBackend doesn't have a direct 'apply_linear_operator'?
        # Actually, we can use compute_expectation_value logic to apply it.
        # mat = backend._pauli_to_matrix(creator_op)
        # psi = mat @ vacuum
        
        # Let's add 'apply_matrix' or similar to backend or use private method
        if hasattr(self.backend, "_pauli_to_matrix"):
            mat = self.backend._pauli_to_matrix(creator_op)
            state = mat @ vacuum
            # Re-normalize just in case (e.g. if vacuum was not exactly 0)
            if self.backend.use_gpu:
                import cupy
                norm = cupy.linalg.norm(state)
                state /= norm
            else:
                state /= np.linalg.norm(state)
            return state
        else:
            raise NotImplementedError("Backend must support _pauli_to_matrix for scattering init.")

    def evolve(self, state: Any, dt: float, num_steps: int) -> List[Any]:
        """Evolve state and return trajectory."""
        trajectory = [state]
        current_state = state
        
        # Get trotter layers from model
        layers = self.model.get_trotter_layers()
        
        for _ in range(num_steps):
            current_state = self.backend.evolve_state_trotter(current_state, layers, dt)
            trajectory.append(current_state)
            
        return trajectory

    def compute_density_profile(self, state: Any) -> np.ndarray:
        """Compute <Z_i> or population at each site."""
        profile = []
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites; p[i] = "Z"
            op = SparsePauliOp.from_list([("".join(reversed(p)), 1.0)])
            val = self.backend.compute_expectation_value(state, op)
            profile.append(val)
        return np.array(profile)
