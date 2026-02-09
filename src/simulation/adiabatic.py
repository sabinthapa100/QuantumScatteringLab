from typing import List, Dict, Optional
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ..models.base import PhysicsModel
from ..backends.base import QuantumBackend
from ..backends.qiskit_backend import QiskitBackend

class AdiabaticSolver:
    """
    Adiabatic State Preparation Solver.
    
    Evolves the system from the ground state of an initial Hamiltonian H_0 
    to the target Hamiltonian H_T over time T.
    
    H(t) = (1 - t/T) H_0 + (t/T) H_T
    """
    
    def __init__(self, model_target: PhysicsModel, model_initial: Optional[PhysicsModel] = None, backend: Optional[QuantumBackend] = None):
        self.model_target = model_target
        self.H_target = model_target.build_hamiltonian()
        
        # If no initial model, assume pure Transverse Field H = - sum X
        # For now, let's just assume the user provides it or we use the Field part of Target.
        # This is a simplification. Ideally, H_initial is simpler.
        self.H_initial = model_initial.build_hamiltonian() if model_initial else self._build_default_H0(model_target.num_sites)
        
        self.backend = backend if backend else QiskitBackend()
        
    def _build_default_H0(self, N: int) -> SparsePauliOp:
        # Default H0 = - sum X (Transverse field)
        # Ground state is |+> ? No, if coeff is negative (-X), ground state of -X is |+>.
        # Reference state |0> is ground state of -Z.
        # Let's use H0 = - sum Z so |00...0> is ground state.
        terms = []
        for i in range(N):
            p = ["I"] * N
            p[i] = "Z"
            terms.append(("".join(reversed(p)), -1.0))
        return SparsePauliOp.from_list(terms)

    def run(self, total_time: float = 10.0, dt: float = 0.1) -> Dict:
        """
        Run adiabatic evolution.
        """
        steps = int(total_time / dt)
        state = self.backend.get_reference_state(self.model_target.num_sites)
        
        energy_history = []
        
        for step in range(steps):
            t = step * dt
            s = t / total_time # Interpolation parameter [0, 1]
            
            # H(s) = (1-s)H0 + sH1
            # We need to Trotterize H(s).
            # This requires mixing the layers of H0 and H1.
            # Simplified: Just compute the full H(s) and evolve using backend's generic evolve if supported,
            # or approximate H(s) ~ H_target (if s close to 1) + perturbation?
            
            # Better: H(s) is a SparsePauliOp.
            # Backend might support direct evolution of arbitrary H.
            # QiskitBackend does via PauliEvolutionGate.
            
            H_current = (1 - s) * self.H_initial + s * self.H_target
            
            # Evolve by exp(-i * H(s) * dt)
            # We treat H_current as a single layer for simplicity in this demo.
            # Backend should handle it.
            # Note: evolve_state_trotter expects LAYERS. 
            # If we pass [H_current], QiskitBackend will use PauliEvolutionGate on sum.
            
            state = self.backend.evolve_state_trotter(state, [H_current], dt)
            
            # Measure expectation of Target H
            E = self.backend.compute_expectation_value(state, self.H_target)
            energy_history.append(E)
            
        return {
            "final_energy": energy_history[-1],
            "energy_history": energy_history,
            "final_state": state
        }
