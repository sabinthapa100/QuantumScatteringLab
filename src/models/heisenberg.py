"""
Heisenberg XXX/XXZ spin chain model.

The Heisenberg model is a fundamental model in quantum magnetism describing
interacting spin-1/2 particles on a lattice.
"""

from typing import List
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .base import PhysicsModel, Symmetry, ModelMetadata


class HeisenbergModel(PhysicsModel):
    """
    1D Heisenberg XXX/XXZ spin chain.
    
    Hamiltonian:
        H = sum_i [J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1}] + h sum_i Z_i
    
    Special Cases:
        - XXX (isotropic): J_x = J_y = J_z
        - XXZ: J_x = J_y ≠ J_z
        - Ising limit: J_x = J_y = 0
    
    The XXX model has SU(2) symmetry and is exactly solvable via Bethe ansatz.
    The XXZ model has U(1) symmetry (total S^z conservation).
    
    Attributes:
        j_x (float): XX coupling strength.
        j_y (float): YY coupling strength.
        j_z (float): ZZ coupling strength.
        h (float): Longitudinal magnetic field.
    """
    
    def __init__(
        self,
        num_sites: int,
        j_x: float = 1.0,
        j_y: float = 1.0,
        j_z: float = 1.0,
        h: float = 0.0,
        pbc: bool = True
    ):
        """
        Initialize the Heisenberg model.
        
        Args:
            num_sites: Number of lattice sites.
            j_x: XX coupling (default: 1.0).
            j_y: YY coupling (default: 1.0).
            j_z: ZZ coupling (default: 1.0).
            h: Magnetic field (default: 0.0).
            pbc: Use periodic boundary conditions (default: True).
        """
        self.j_x = j_x
        self.j_y = j_y
        self.j_z = j_z
        self.h = h
        
        super().__init__(num_sites, pbc)
    
    def _validate_parameters(self) -> None:
        """Validate Heisenberg model parameters."""
        if self.j_x == 0 and self.j_y == 0 and self.j_z == 0 and self.h == 0:
            raise ValueError("All coupling constants cannot be zero")
    
    def build_hamiltonian(self) -> SparsePauliOp:
        """Construct the Heisenberg Hamiltonian."""
        terms = []
        coeffs = []
        
        # Interaction terms
        for i in range(self.num_bonds):
            next_i = (i + 1) % self.num_sites
            
            # XX term
            if not np.isclose(self.j_x, 0.0):
                pauli_str = ["I"] * self.num_sites
                pauli_str[i] = "X"
                pauli_str[next_i] = "X"
                terms.append("".join(reversed(pauli_str)))
                coeffs.append(self.j_x)
            
            # YY term
            if not np.isclose(self.j_y, 0.0):
                pauli_str = ["I"] * self.num_sites
                pauli_str[i] = "Y"
                pauli_str[next_i] = "Y"
                terms.append("".join(reversed(pauli_str)))
                coeffs.append(self.j_y)
            
            # ZZ term
            if not np.isclose(self.j_z, 0.0):
                pauli_str = ["I"] * self.num_sites
                pauli_str[i] = "Z"
                pauli_str[next_i] = "Z"
                terms.append("".join(reversed(pauli_str)))
                coeffs.append(self.j_z)
        
        # Magnetic field
        if not np.isclose(self.h, 0.0):
            for i in range(self.num_sites):
                pauli_str = ["I"] * self.num_sites
                pauli_str[i] = "Z"
                terms.append("".join(reversed(pauli_str)))
                coeffs.append(self.h)
        
        return SparsePauliOp.from_list(list(zip(terms, coeffs)))
    
    def build_operator_pool(self) -> List[SparsePauliOp]:
        """
        Construct operator pool for ADAPT-VQE.
        
        Pool contains:
        - Single-site rotations: X_i, Y_i
        - Two-site interactions: X_i X_{i+1}, Y_i Y_{i+1}, Z_i Z_{i+1}
        - Mixed terms: X_i Y_{i+1}, Y_i X_{i+1}, etc.
        """
        pool = []
        
        # Single-site operators
        for i in range(self.num_sites):
            # X_i
            pauli_str = ["I"] * self.num_sites
            pauli_str[i] = "X"
            pool.append(SparsePauliOp.from_list([("".join(reversed(pauli_str)), 1.0)]))
            
            # Y_i
            pauli_str = ["I"] * self.num_sites
            pauli_str[i] = "Y"
            pool.append(SparsePauliOp.from_list([("".join(reversed(pauli_str)), 1.0)]))
        
        # Two-site operators
        for i in range(self.num_bonds):
            next_i = (i + 1) % self.num_sites
            
            # XX
            p1 = ["I"] * self.num_sites
            p1[i] = "X"
            p1[next_i] = "X"
            pool.append(SparsePauliOp.from_list([("".join(reversed(p1)), 1.0)]))
            
            # YY
            p2 = ["I"] * self.num_sites
            p2[i] = "Y"
            p2[next_i] = "Y"
            pool.append(SparsePauliOp.from_list([("".join(reversed(p2)), 1.0)]))
            
            # ZZ
            p3 = ["I"] * self.num_sites
            p3[i] = "Z"
            p3[next_i] = "Z"
            pool.append(SparsePauliOp.from_list([("".join(reversed(p3)), 1.0)]))
            
            # Mixed: XY, YX, XZ, ZX, YZ, ZY
            for pauli_pair in [("X", "Y"), ("Y", "X"), ("X", "Z"), ("Z", "X"), ("Y", "Z"), ("Z", "Y")]:
                p = ["I"] * self.num_sites
                p[i] = pauli_pair[0]
                p[next_i] = pauli_pair[1]
                pool.append(SparsePauliOp.from_list([("".join(reversed(p)), 1.0)]))
        
        return pool
    
    def get_trotter_layers(self) -> List[SparsePauliOp]:
        """
        Trotter decomposition using even/odd bond splitting.
        
        Layers:
        1. Odd bonds (0-1, 2-3, ...)
        2. Even bonds (1-2, 3-4, ...)
        3. Magnetic field
        """
        odd_terms = []
        odd_coeffs = []
        even_terms = []
        even_coeffs = []
        field_terms = []
        field_coeffs = []
        
        # Bonds
        for i in range(self.num_bonds):
            next_i = (i + 1) % self.num_sites
            
            target_terms = odd_terms if (i % 2 == 1) else even_terms
            target_coeffs = odd_coeffs if (i % 2 == 1) else even_coeffs
            
            # XX
            if not np.isclose(self.j_x, 0.0):
                pauli_str = ["I"] * self.num_sites
                pauli_str[i] = "X"
                pauli_str[next_i] = "X"
                target_terms.append("".join(reversed(pauli_str)))
                target_coeffs.append(self.j_x)
            
            # YY
            if not np.isclose(self.j_y, 0.0):
                pauli_str = ["I"] * self.num_sites
                pauli_str[i] = "Y"
                pauli_str[next_i] = "Y"
                target_terms.append("".join(reversed(pauli_str)))
                target_coeffs.append(self.j_y)
            
            # ZZ
            if not np.isclose(self.j_z, 0.0):
                pauli_str = ["I"] * self.num_sites
                pauli_str[i] = "Z"
                pauli_str[next_i] = "Z"
                target_terms.append("".join(reversed(pauli_str)))
                target_coeffs.append(self.j_z)
        
        # Magnetic field
        if not np.isclose(self.h, 0.0):
            for i in range(self.num_sites):
                pauli_str = ["I"] * self.num_sites
                pauli_str[i] = "Z"
                field_terms.append("".join(reversed(pauli_str)))
                field_coeffs.append(self.h)
        
        layers = []
        if odd_terms:
            layers.append(SparsePauliOp.from_list(list(zip(odd_terms, odd_coeffs))))
        if even_terms:
            layers.append(SparsePauliOp.from_list(list(zip(even_terms, even_coeffs))))
        if field_terms:
            layers.append(SparsePauliOp.from_list(list(zip(field_terms, field_coeffs))))
        
        return layers
    
    def get_symmetries(self) -> List[Symmetry]:
        """
        Return symmetries of the Heisenberg model.
        
        - XXX (isotropic): SU(2) symmetry
        - XXZ: U(1) symmetry (S^z conservation)
        - Translation: Present if PBC
        - Parity: Always present
        """
        symmetries = [Symmetry.PARITY]
        
        if self.pbc:
            symmetries.append(Symmetry.TRANSLATION)
        
        # Check for isotropic XXX
        if np.isclose(self.j_x, self.j_y) and np.isclose(self.j_y, self.j_z) and np.isclose(self.h, 0.0):
            symmetries.append(Symmetry.SU2)
        # Check for XXZ
        elif np.isclose(self.j_x, self.j_y) and not np.isclose(self.h, 0.0):
            symmetries.append(Symmetry.U1)
        
        return symmetries
    
    def get_metadata(self) -> ModelMetadata:
        """Return metadata about the Heisenberg model."""
        # Determine model type
        if np.isclose(self.j_x, self.j_y) and np.isclose(self.j_y, self.j_z):
            model_type = "XXX (isotropic)"
        elif np.isclose(self.j_x, self.j_y):
            model_type = "XXZ"
        else:
            model_type = "XYZ (general)"
        
        return ModelMetadata(
            name=f"1D Heisenberg {model_type} Model",
            description=(
                f"Heisenberg model with J_x={self.j_x}, J_y={self.j_y}, J_z={self.j_z}, h={self.h}. "
                f"{'PBC' if self.pbc else 'OBC'} on {self.num_sites} sites."
            ),
            critical_points={},
            citation="Bethe (1931), arXiv:cond-mat/9809163"
        )
