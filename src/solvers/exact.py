import numpy as np
from qiskit.quantum_info import SparsePauliOp
from ..models.base import PhysicsModel

class ExactSolver:
    """
    Exact Diagonalization solver for benchmarking.
    """
    
    def __init__(self, model: PhysicsModel):
        self.hamiltonian = model.build_hamiltonian()
        
    def solve(self):
        """
        Diagonalize the Hamiltonian and return the ground state energy.
        """
        # Convert SparsePauliOp to dense matrix
        H_matrix = self.hamiltonian.to_matrix()
        
        # Diagonalize
        evals, evecs = np.linalg.eigh(H_matrix)
        
        return {
            "energy": evals[0],
            "excited_states": evals[1:5],
            "ground_state_vector": evecs[:, 0]
        }
