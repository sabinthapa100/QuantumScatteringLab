"""
Backend using Quimb for Tensor Network and high-performance Statevector simulation.

Supports:
- Dense statevector simulation (numpy/scipy/cupy)
- Tensor Network (MPS/MPO) via quimb.tensor (Planned)
- GPU acceleration via cupy (if installed)
"""

import numpy as np
from typing import Any, List, Dict, Optional
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
    
    Includes optional caching of dense operator matrices to avoid repeated
    `SparsePauliOp.to_matrix()` calls in ADAPT-VQE loops for moderate system
    sizes (e.g. N≈8–12), with safeguards to avoid unbounded memory growth.
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        verbose: bool = False,
        cache_operators: bool = True,
        max_cache_size: Optional[int] = 256,
        max_cache_dim: Optional[int] = 4096,
    ):
        self.use_gpu = use_gpu
        self.verbose = verbose
        
        # Operator -> dense matrix cache (keyed by a stable structural key and
        # device type). This is purely a performance optimisation; it must not
        # change physics.
        self.cache_operators = cache_operators
        self._max_cache_size = max_cache_size
        self._max_cache_dim = max_cache_dim
        # Cache for operator matrices (CPU or GPU array / sparse matrix),
        # keyed by (structural_key, use_gpu_flag).
        self._op_cache: Dict[tuple[Any, bool], Any] = {}
        # Backwards-compatible alias (in case old code references _matrix_cache).
        self._matrix_cache = self._op_cache
        
        # Check for GPU support
        if self.use_gpu:
            try:
                import cupy  # type: ignore[unused-import]
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
        
        if self.use_gpu:
            import cupy
            return cupy.asarray(psi)
        return psi

    def _get_operator_cache_key(self, operator: SparsePauliOp) -> tuple[Any, bool]:
        """Construct a stable cache key for ``operator``."""
        device_flag = bool(self.use_gpu)
        
        # Fast path: reuse a precomputed structural key if available.
        base_key: Any
        try:
            existing_key = getattr(operator, "_cache_key", None)
        except Exception:
            existing_key = None
        
        if existing_key is not None:
            base_key = existing_key
        else:
            try:
                # Qiskit SparsePauliOp.to_list() -> List[(label, coeff)]
                op_list = operator.to_list()
                # Make hashable: ((label, complex(coeff)), ...)
                base_key = tuple((label, complex(coeff)) for label, coeff in op_list)
                # Store on the operator instance for O(1) reuse.
                try:
                    setattr(operator, "_cache_key", base_key)
                except Exception:
                    pass
            except Exception:
                base_key = id(operator)
        
        return (base_key, device_flag)

    def _pauli_to_matrix(self, operator: SparsePauliOp) -> Any:
        """Convert Qiskit SparsePauliOp to a (potentially sparse) matrix with optional caching.
        
        For performance, we request a sparse matrix from Qiskit and cache the
        resulting SciPy CSR matrix (or GPU equivalent) keyed by a structural
        operator hash and device flag.
        """
        try:
            if not self.cache_operators:
                mat = operator.to_matrix(sparse=True)
                if self.use_gpu:
                    from cupyx.scipy.sparse import csr_matrix  # type: ignore[import]
                    return csr_matrix(mat)
                return mat
            
            key = self._get_operator_cache_key(operator)
            cached = self._op_cache.get(key)
            if cached is not None:
                return cached
            
            # Convert to sparse matrix (CSR) on CPU
            mat = operator.to_matrix(sparse=True)
            
            # Move to appropriate device
            if self.use_gpu:
                from cupyx.scipy.sparse import csr_matrix  # type: ignore[import]
                arr = csr_matrix(mat)
            else:
                arr = mat
            
            # Cache if within limits
            dim = arr.shape[0] if hasattr(arr, "shape") and len(arr.shape) > 0 else None
            if (
                self.cache_operators
                and dim is not None
                and (self._max_cache_dim is None or dim <= self._max_cache_dim)
            ):
                if self._max_cache_size is None or len(self._op_cache) < self._max_cache_size:
                    self._op_cache[key] = arr
            
            return arr
        except Exception as e:
            raise BackendError(f"Failed to convert operator to matrix: {e}")

    def compute_expectation_value(self, state: Any, operator: SparsePauliOp) -> float:
        """Compute ⟨ψ|O|ψ⟩ using cached (sparse) matrix representation."""
        H_mat = self._pauli_to_matrix(operator)
        
        if self.use_gpu:
            import cupy
            H_psi = H_mat @ state
            val = cupy.vdot(state, H_psi)
            return float(val.real)
        else:
            H_psi = H_mat @ state
            return float(np.vdot(state, H_psi).real)

    def apply_operator(self, state: Any, operator: SparsePauliOp, parameter: float = 1.0) -> Any:
        """
        Apply U = exp(i * parameter * operator) to state.
        state -> exp(i * θ * P) |ψ⟩.
        
        We use scipy/cupyx ``expm_multiply`` on the (cached) sparse matrix
        representation of the operator, which applies the matrix exponential
        directly to the vector without forming a dense U, preserving physics
        while significantly reducing time and memory for moderate sizes.
        """
        if abs(parameter) < 1e-15:
            return state
        
        P_mat = self._pauli_to_matrix(operator)
        exponent = 1j * parameter * P_mat
        
        if self.use_gpu:
            import cupy
            try:
                from cupyx.scipy.sparse.linalg import expm_multiply  # type: ignore[import]
                return expm_multiply(exponent, state)
            except Exception:
                # Fallback: dense expm on GPU if sparse expm_multiply unavailable
                from cupyx.scipy.linalg import expm  # type: ignore[import]
                U = expm(exponent.toarray())
                return cupy.matmul(U, state)
        else:
            from scipy.sparse.linalg import expm_multiply
            return expm_multiply(exponent, state)

    def evolve_state_trotter(self, state: Any, hamiltonian_layers: List[SparsePauliOp], time_step: float) -> Any:
        """First order Trotter evolution."""
        current_state = state
        for layer_op in hamiltonian_layers:
            current_state = self.apply_operator(current_state, layer_op, parameter=-time_step)
        return current_state
