"""
Spectrum Analyzer for Quantum Models.

Computes exact eigenvalue spectra using scipy.sparse.linalg for models
implementing the PhysicsModel interface.
"""
import numpy as np
from scipy.sparse.linalg import eigsh
from typing import List, Dict, Any, Optional, Tuple

from ..models.base import PhysicsModel


class SpectrumAnalyzer:
    """
    Computes and analyzes eigenvalue spectra for PhysicsModel instances.
    """
    
    def __init__(self, model: PhysicsModel):
        """
        Initialize with a PhysicsModel.
        
        Args:
            model: A PhysicsModel instance with build_hamiltonian() method.
        """
        self.model = model
        self._hamiltonian_matrix = None
    
    def _get_hamiltonian_matrix(self) -> np.ndarray:
        """Get the Hamiltonian as a sparse matrix."""
        if self._hamiltonian_matrix is None:
            H_op = self.model.build_hamiltonian()
            # Convert SparsePauliOp to scipy sparse matrix
            self._hamiltonian_matrix = H_op.to_matrix(sparse=True)
        return self._hamiltonian_matrix
    
    def compute_ground_state(self) -> Tuple[float, np.ndarray]:
        """
        Compute the ground state energy and wavefunction.
        
        Returns:
            Tuple of (ground_state_energy, ground_state_vector)
        """
        H = self._get_hamiltonian_matrix()
        # Use eigsh for sparse Hermitian matrices, k=1 for ground state
        eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
        return eigenvalues[0].real, eigenvectors[:, 0]
    
    def compute_spectrum(self, num_states: int = 10) -> Dict[str, Any]:
        """
        Compute the lowest num_states eigenvalues.
        
        Args:
            num_states: Number of lowest eigenvalues to compute.
            
        Returns:
            Dict with 'energies' (array) and 'gap' (E1 - E0).
        """
        H = self._get_hamiltonian_matrix()
        
        # Ensure we don't ask for more states than the Hilbert space dimension
        max_states = min(num_states, H.shape[0] - 2)
        
        eigenvalues, _ = eigsh(H, k=max_states, which='SA')
        eigenvalues = np.sort(eigenvalues.real)
        
        gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0
        
        return {
            'energies': eigenvalues,
            'ground_state_energy': eigenvalues[0],
            'gap': gap,
            'num_states': len(eigenvalues)
        }
    
    def compute_full_spectrum(self) -> np.ndarray:
        """
        Compute the full eigenvalue spectrum (for small systems only).
        Uses dense diagonalization.
        
        Returns:
            Sorted array of all eigenvalues.
        """
        H = self._get_hamiltonian_matrix()
        H_dense = H.toarray()
        eigenvalues = np.linalg.eigvalsh(H_dense)
        return np.sort(eigenvalues)


def scan_parameter(
    model_class,
    param_name: str,
    param_values: np.ndarray,
    fixed_params: Dict[str, Any],
    num_states: int = 6
) -> Dict[str, np.ndarray]:
    """
    Scan a parameter and compute the spectrum at each point.
    
    Args:
        model_class: The PhysicsModel class (e.g., IsingModel1D).
        param_name: Name of the parameter to scan (e.g., 'g_x').
        param_values: Array of parameter values to scan.
        fixed_params: Dict of other fixed parameters (e.g., {'num_sites': 8, 'j_int': 1.0}).
        num_states: Number of eigenvalues to compute at each point.
        
    Returns:
        Dict with 'param_values', 'energies' (2D array), 'gaps'.
    """
    all_energies = []
    all_gaps = []
    
    for val in param_values:
        # Construct model with this parameter value
        params = fixed_params.copy()
        params[param_name] = val
        
        model = model_class(**params)
        analyzer = SpectrumAnalyzer(model)
        result = analyzer.compute_spectrum(num_states=num_states)
        
        all_energies.append(result['energies'])
        all_gaps.append(result['gap'])
    
    # Pad energies to same length if necessary
    max_len = max(len(e) for e in all_energies)
    padded_energies = np.full((len(param_values), max_len), np.nan)
    for i, e in enumerate(all_energies):
        padded_energies[i, :len(e)] = e
    
    return {
        'param_values': param_values,
        'energies': padded_energies,
        'gaps': np.array(all_gaps)
    }
