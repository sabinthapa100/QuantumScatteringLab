"""
Abstract backend interface for quantum simulation.

This module defines the QuantumBackend ABC that allows seamless switching
between different simulation methods:
    - Qiskit Statevector (exact, CPU)
    - Quimb MPS/TN (approximate, GPU-capable)
    - Future: Real hardware, noise models

Design Principles:
    - Backends operate on opaque state types (Statevector, MPS, etc.)
    - All operations are defined in terms of SparsePauliOp for portability
    - Error handling with informative messages
"""

from abc import ABC, abstractmethod
from typing import Any, List, Union, Optional, Dict
import logging

from qiskit.quantum_info import SparsePauliOp

# Configure module-level logger
logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Exception raised for backend-specific errors."""
    pass


class QuantumBackend(ABC):
    """
    Abstract interface for quantum simulation backends.
    
    Allows switching between Qiskit (CPU exact) and Quimb (GPU-capable MPS)
    without changing the simulation code.
    
    Subclasses must implement:
        - get_reference_state(num_sites) -> state
        - compute_expectation_value(state, operator) -> float
        - apply_operator(state, operator, parameter) -> state
        - evolve_state_trotter(state, layers, dt) -> state
        
    The state type is backend-specific (Statevector, MPS, etc.) and should
    be treated as opaque by the calling code.
    
    Example:
        >>> backend = QiskitBackend()
        >>> state = backend.get_reference_state(8)
        >>> energy = backend.compute_expectation_value(state, hamiltonian)
    """
    
    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return self.__class__.__name__
    
    @abstractmethod
    def get_reference_state(self, num_sites: int) -> Any:
        """
        Returns the initial reference state |00...0⟩.
        
        Args:
            num_sites: Number of qubits in the system.
            
        Returns:
            Backend-specific state representation (Statevector, MPS, etc.).
            
        Raises:
            BackendError: If state creation fails.
        """
        pass

    @abstractmethod
    def compute_expectation_value(self, state: Any, operator: SparsePauliOp) -> float:
        """
        Compute the expectation value ⟨ψ|O|ψ⟩.
        
        Args:
            state: The quantum state (backend-specific type).
            operator: Observable as SparsePauliOp.
            
        Returns:
            Real-valued expectation value.
            
        Raises:
            BackendError: If computation fails.
            
        Note:
            The returned value is real (imaginary part discarded for Hermitian O).
        """
        pass

    @abstractmethod
    def apply_operator(
        self, 
        state: Any, 
        operator: SparsePauliOp, 
        parameter: float = 1.0
    ) -> Any:
        """
        Apply the unitary exp(i θ P) to the state.
        
        This is the fundamental operation for ADAPT-VQE ansatz construction.
        For a Hermitian operator P, exp(i θ P) is unitary.
        
        Args:
            state: The input quantum state.
            operator: Hermitian operator P (typically a Pauli string).
            parameter: The angle θ in the exponential.
            
        Returns:
            The evolved state exp(i θ P)|ψ⟩.
            
        Note:
            Qiskit's PauliEvolutionGate computes exp(-i H t), so we use t = -θ.
        """
        pass
        
    @abstractmethod
    def evolve_state_trotter(
        self, 
        state: Any, 
        hamiltonian_layers: List[SparsePauliOp], 
        time_step: float
    ) -> Any:
        """
        Evolve the state by one Trotter step.
        
        Implements first-order Trotter:
            U(dt) = exp(-i H_1 dt) exp(-i H_2 dt) ... exp(-i H_N dt)
        
        For second-order (symmetric) Trotter, call twice with dt/2.
        
        Args:
            state: The input quantum state.
            hamiltonian_layers: List [H_1, H_2, ...] of commuting sublayers.
            time_step: The time step dt.
            
        Returns:
            The time-evolved state.
        """
        pass

    def compute_observable_statistics(
        self, 
        state: Any, 
        observables: Dict[str, SparsePauliOp]
    ) -> Dict[str, float]:
        """
        Compute expectation values for multiple observables.
        
        Convenience method for measuring several observables at once.
        
        Args:
            state: The quantum state.
            observables: Dictionary mapping names to operators.
            
        Returns:
            Dictionary mapping names to expectation values.
        """
        results = {}
        for name, op in observables.items():
            results[name] = self.compute_expectation_value(state, op)
            logger.debug(f"Observable '{name}': {results[name]:.6f}")
        return results

    def verify_compatibility(self, num_sites: int) -> None:
        """
        Verify that this backend can handle the given system size.
        
        Args:
            num_sites: Number of qubits.
            
        Raises:
            BackendError: If system is too large for this backend.
        """
        # Default: no limits (subclasses can override)
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

