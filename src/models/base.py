"""
Base classes for physics models in QuantumScatteringLab.

This module defines the abstract interface that all physics models must implement.
Models encapsulate:
    1. The Hamiltonian of the system.
    2. Operator pool for variational algorithms (ADAPT-VQE).
    3. Trotter decomposition for time evolution.

Design Principles:
    - Models are backend-agnostic (they produce SparsePauliOp, not circuits).
    - Models are immutable after construction (thread-safe).
    - Models validate parameters on construction.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from qiskit.quantum_info import SparsePauliOp

# Configure module-level logger
logger = logging.getLogger(__name__)


class Symmetry(Enum):
    """Symmetries that a physics model may possess."""
    TRANSLATION = auto()      # Discrete translation invariance
    PARITY = auto()           # Spatial reflection symmetry
    TIME_REVERSAL = auto()    # Time-reversal invariance
    Z2 = auto()               # Z_2 spin-flip symmetry
    U1 = auto()               # U(1) charge/magnetization conservation
    SU2 = auto()              # SU(2) spin rotation invariance


@dataclass(frozen=True)
class ModelMetadata:
    """
    Immutable metadata about a physics model.
    
    Attributes:
        name: Human-readable model name.
        description: Brief physics description.
        critical_points: Known critical parameter values (for reference).
        citation: Optional paper reference.
    """
    name: str
    description: str
    critical_points: Dict[str, float] = field(default_factory=dict)
    citation: Optional[str] = None


class PhysicsModel(ABC):
    """
    Abstract Base Class for all physical models in the Quantum Scattering framework.
    
    This class enforces a standard interface for defining Hamiltonians, 
    operator pools for VQE, and Trotterization strategies.
    Decoupling the model definition from the simulation backend ensures 
    reusability and easier debugging.
    
    Subclasses must implement:
        - build_hamiltonian() -> SparsePauliOp
        - build_operator_pool() -> List[SparsePauliOp]
        - get_trotter_layers() -> List[SparsePauliOp]
        
    Optional overrides:
        - get_symmetries() -> List[Symmetry]
        - get_metadata() -> ModelMetadata
        - validate_parameters() -> None (called in __init__)
    
    Example:
        >>> model = IsingModel1D(num_sites=8, g_x=1.0, pbc=True)
        >>> H = model.build_hamiltonian()
        >>> print(f"Hamiltonian has {len(H)} terms")
    """

    def __init__(self, num_sites: int, pbc: bool = True):
        """
        Initialize the model.
        
        Args:
            num_sites: Number of lattice sites (qubits). Must be >= 2.
            pbc: If True, use periodic boundary conditions.
            
        Raises:
            ValueError: If num_sites < 2.
        """
        if num_sites < 2:
            raise ValueError(f"num_sites must be >= 2, got {num_sites}")
        
        self._num_sites = num_sites
        self._pbc = pbc
        
        # Validate model-specific parameters (implemented by subclasses)
        self._validate_parameters()
        
        logger.debug(f"Initialized {self.__class__.__name__} with {num_sites} sites, PBC={pbc}")

    @property
    def num_sites(self) -> int:
        """Number of lattice sites (qubits)."""
        return self._num_sites
    
    @property
    def pbc(self) -> bool:
        """Whether periodic boundary conditions are used."""
        return self._pbc
    
    @property
    def num_bonds(self) -> int:
        """Number of nearest-neighbor bonds."""
        if self._pbc:
            return self._num_sites
        else:
            return self._num_sites - 1

    def _validate_parameters(self) -> None:
        """
        Validate model-specific parameters.
        
        Override this in subclasses to add parameter checks.
        Called automatically in __init__ after base validation.
        
        Raises:
            ValueError: If any parameter is invalid.
        """
        pass  # Default: no additional validation

    @abstractmethod
    def build_hamiltonian(self) -> SparsePauliOp:
        """
        Construct and return the full Hamiltonian of the system.
        
        The Hamiltonian is returned as a SparsePauliOp, which is a sum of
        weighted Pauli strings. This representation is backend-agnostic and
        can be converted to matrices, circuits, or tensor networks as needed.
        
        Returns:
            SparsePauliOp: The total Hamiltonian H.
            
        Example:
            >>> H = model.build_hamiltonian()
            >>> # H is now sum_i c_i P_i where P_i are Pauli strings
        """
        pass

    @abstractmethod
    def build_operator_pool(self) -> List[SparsePauliOp]:
        """
        Construct the operator pool for ADAPT-VQE.
        
        The pool consists of Hermitian operators {P_k} that generate
        parameterized unitaries exp(-i θ_k P_k). ADAPT-VQE iteratively
        selects operators from this pool based on gradient magnitude.
        
        For efficiency, operators should be:
            1. Hermitian (so exp(-iθP) is unitary)
            2. Local or quasi-local (for efficient circuit/MPS implementation)
            3. Spanning the relevant Hilbert space sector
        
        Returns:
            List[SparsePauliOp]: A list of Hermitian operators {P_k}.
            
        Note:
            The pool should NOT contain the identity operator.
        """
        pass

    @abstractmethod
    def get_trotter_layers(self) -> List[SparsePauliOp]:
        """
        Return the Hamiltonian split into layers for Trotterization.
        
        The Hamiltonian H = H_1 + H_2 + ... is decomposed such that terms
        within each layer H_k commute (or are trivially diagonalizable).
        This enables efficient time evolution:
            exp(-iHt) ≈ exp(-i H_1 t) exp(-i H_2 t) ... + O(t^2)
        
        Common decompositions:
            - Even/Odd bond decomposition for 1D chains
            - Color decomposition for higher dimensions
        
        Returns:
            List[SparsePauliOp]: The list of Hamiltonian layers [H_1, H_2, ...].
        """
        pass

    def get_symmetries(self) -> List[Symmetry]:
        """
        Return the symmetries of this model.
        
        Knowing symmetries allows:
            - Exploitation for computational speedup
            - Validation of simulation results
            - Understanding phase structure
        
        Returns:
            List[Symmetry]: List of symmetries present in the model.
            Empty list if unknown or no special symmetries.
        """
        return []  # Default: no known symmetries

    def get_metadata(self) -> ModelMetadata:
        """
        Return metadata about this model.
        
        Useful for documentation, provenance tracking, and display.
        
        Returns:
            ModelMetadata: Information about the model.
        """
        return ModelMetadata(
            name=self.__class__.__name__,
            description="Generic physics model"
        )

    def get_reference_state_label(self) -> str:
        """
        Return the computational basis label for the default initial state.
        
        This is typically |00...0⟩ for most applications, but some models
        may require a different starting point (e.g., Néel state).
        
        Returns:
            str: Binary string of length num_sites (e.g., "00000000").
        """
        return "0" * self._num_sites

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_sites={self._num_sites}, pbc={self._pbc})"

    def __str__(self) -> str:
        metadata = self.get_metadata()
        return f"{metadata.name}: {self._num_sites} sites, {'PBC' if self._pbc else 'OBC'}"

