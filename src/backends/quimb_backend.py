"""
Backend using Quimb for Tensor Network and high-performance Statevector simulation.

Supports:
- Dense statevector simulation (numpy/scipy/cupy)
- Tensor Network (MPS/MPO) via quimb.tensor (Planned)
- GPU acceleration via cupy (if installed)
"""

import numpy as np
from typing import Any, List, Dict, Optional, Union
from qiskit.quantum_info import SparsePauliOp
import quimb as qu
import logging
from .base import QuantumBackend, BackendError

logger = logging.getLogger(__name__)

class QuimbBackend(QuantumBackend):
    """
    High-performance simulation backend using Quimb.
    
    Defaults to dense statevector simulation which is very fast for N < 20.
    Can leverage GPU if cupy is installed and configured in quimb.
    """
    
    def __init__(self, use_gpu: bool = False, verbose: bool = False):
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        # Check for GPU support
        if self.use_gpu:
            try:
                import cupy
                logger.info("QuimbBackend: GPU (cupy) enabled.")
            except ImportError:
                logger.warning("QuimbBackend: GPU requested but cupy not found. Falling back to CPU.")
                self.use_gpu = False
                
    def get_statevector(self, state: Any) -> np.ndarray:
        """Return state as a numpy array (moves from GPU if needed)."""
        arr = state
        if self.use_gpu:
            import cupy
            if isinstance(state, cupy.ndarray):
                arr = state.get()
        
        arr = np.asarray(arr).flatten() # Ensure 1D
        # logger.debug(f"QuimbBackend state shape: {arr.shape}, norm: {np.linalg.norm(arr)}")
        return arr

    def get_reference_state(self, num_sites: int) -> Any:
        """Returns |0...0> as a quimb qarray (dense vector)."""
        # quimb uses '0' for |0>
        # qarray is a subclass of numpy.ndarray
        psi = qu.computational_state('0' * num_sites)
        
        # Flatten initially? Qiskit backend uses 1D.
        # But for quimb operations (matmul), (N,1) might be better?
        # Let's keep (N,1) internally if quimb produces it, but flatten in get_statevector.
        
        if self.use_gpu:
            import cupy
            return cupy.asarray(psi)
        return psi


    def _pauli_to_matrix(self, operator: SparsePauliOp) -> Any:
        """Convert Qiskit SparsePauliOp to a dense complex matrix."""
        # This is efficient for small-ish systems (N < 14) but scales poorly O(2^N).
        # For larger systems, we should handle sparse or TN.
        # But ADAPT-VQE typically needs the full operator for evolution.
        
        # This conversion is tricky because qiskit vs quimb basis conventions/ordering might differ.
        # Qiskit: q_n ... q_0 order? No, qiskit is typically little-endian for printing but standard for matrix.
        # Quimb: qu.kron(A, B) -> A x B. A corresponds to qubit 0 (leftmost) if we index 0,1,2.
        # Qiskit 'XYZ' string implies Qubit (N-1) ... Qubit 0.
        
        # Solution: Convert SparsePauliOp to matrix using qiskit, then get the numpy array.
        # This ensures correctness of the matrix representation.
        try:
            mat = operator.to_matrix()
            if self.use_gpu:
                import cupy
                return cupy.asarray(mat)
            return mat
        except Exception as e:
            raise BackendError(f"Failed to convert operator to matrix: {e}")

    def compute_expectation_value(self, state: Any, operator: SparsePauliOp) -> float:
        """Compute <psi|O|psi> using quimb/numpy."""
        # Convert operator to matrix
        # For N=12, matrix is 4096 x 4096. Manageable.
        H_mat = self._pauli_to_matrix(operator)
        
        # Expectation: <psi|H|psi>
        # qu.expec(op, state) usually handles density matrices or vectors
        if self.use_gpu:
            import cupy
            # cupy.vdot for inner product
            # H|psi>
            H_psi = cupy.matmul(H_mat, state)
            # <psi|H|psi>
            val = cupy.vdot(state, H_psi)
            return float(val.real)
        else:
            return qu.expec(H_mat, state).real

    def apply_operator(self, state: Any, operator: SparsePauliOp, parameter: float = 1.0) -> Any:
        """
        Apply U = exp(i * parameter * operator) to state.
        state -> exp(i * theta * P) |psi>
        """
        # Convert P to matrix
        P_mat = self._pauli_to_matrix(operator)
        
        # Construct exponent: i * theta * P
        # For Hermitian P, exp(i theta P) is unitary.
        exponent = 1j * parameter * P_mat
        
        # Expm
        if self.use_gpu:
            import cupy
            from cupyx.scipy.linalg import expm
            U = expm(exponent)
            new_state = cupy.matmul(U, state)
        else:
            from scipy.linalg import expm
            U = expm(exponent)
            new_state = np.dot(U, state)
            
        return new_state

    def evolve_state_trotter(self, state: Any, hamiltonian_layers: List[SparsePauliOp], time_step: float) -> Any:
        """First order Trotter evolution."""
        current_state = state
        
        for layer_op in hamiltonian_layers:
            # U = exp(-i * H * t)
            # Here apply_operator applies exp(i * theta * P), so set theta = -t * coeff?
            # Wait, apply_operator takes SparsePauliOp.
            # layer_op is a sum of terms.
            # We assume layer_op terms commute or we approximate.
            # evolve_state_exact handles the full layer exponential.
            
            # Since layer_op comes from get_trotter_layers, we treat it as a single block Hamiltonian.
            # U = exp(-i * H_layer * dt)
            # apply_operator implementation uses expm(i * theta * P)
            # So pass operator=layer_op, parameter = -time_step
            current_state = self.apply_operator(current_state, layer_op, parameter=-time_step)
            
        return current_state
