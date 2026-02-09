"""
2D Ising Model on a Square Lattice.
Mapped to 1D chain via snake-like path for MPS simulation.
"""

from typing import List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .base import PhysicsModel, ModelMetadata

class IsingModel2D(PhysicsModel):
    """
    2D Transverse Field Ising Model.
    H = - J sum_<ij> Z_i Z_j - g_x sum_i X_i - g_z sum_i Z_i
    
    Mapped to 1D array of size N = Lx * Ly.
    """
    
    def __init__(self, Lx: int, Ly: int, J: float = 1.0, g_x: float = 1.0, g_z: float = 0.0, pbc: bool = False):
        self.Lx = Lx
        self.Ly = Ly
        self.J = J
        self.g_x = g_x
        self.g_z = g_z
        super().__init__(Lx * Ly, pbc)
        
    def _validate_parameters(self) -> None:
        if self.Lx < 2 or self.Ly < 2:
            raise ValueError("Lattice dimensions must be >= 2")

    def _coord_to_index(self, x: int, y: int) -> int:
        """Map (x,y) to 1D index using snake pattern to minimize bond distance."""
        # Snake path:
        # y=0: 0 -> 1 -> ... -> Lx-1
        # y=1: Lx-1 -> ... -> 0 (if valid, helps connectivity?)
        # Standard row-major is simplest for now: i = y * Lx + x
        # But snake might be better for MPS. Let's stick to row-major for simplicity of debugging first.
        return y * self.Lx + x

    def build_hamiltonian(self) -> SparsePauliOp:
        terms = []
        coeffs = []
        
        # Horizontal bonds (x-direction)
        for y in range(self.Ly):
            for x in range(self.Lx):
                # Right neighbor
                if x + 1 < self.Lx:
                    i = self._coord_to_index(x, y)
                    j = self._coord_to_index(x + 1, y)
                    p = ["I"] * self.num_sites
                    p[i] = "Z"; p[j] = "Z"
                    terms.append("".join(reversed(p)))
                    coeffs.append(-self.J)
                elif self.pbc: # Horizontal PBC
                    i = self._coord_to_index(x, y)
                    j = self._coord_to_index(0, y)
                    p = ["I"] * self.num_sites
                    p[i] = "Z"; p[j] = "Z"
                    terms.append("".join(reversed(p)))
                    coeffs.append(-self.J)

        # Vertical bonds (y-direction)
        for y in range(self.Ly):
            for x in range(self.Lx):
                # Top neighbor
                if y + 1 < self.Ly:
                    i = self._coord_to_index(x, y)
                    j = self._coord_to_index(x, y + 1)
                    p = ["I"] * self.num_sites
                    p[i] = "Z"; p[j] = "Z"
                    terms.append("".join(reversed(p)))
                    coeffs.append(-self.J)
                elif self.pbc: # Vertical PBC
                    i = self._coord_to_index(x, y)
                    j = self._coord_to_index(x, 0)
                    p = ["I"] * self.num_sites
                    p[i] = "Z"; p[j] = "Z"
                    terms.append("".join(reversed(p)))
                    coeffs.append(-self.J)
                    
        # Fields
        for i in range(self.num_sites):
            # Transverse X
            p = ["I"] * self.num_sites
            p[i] = "X"
            terms.append("".join(reversed(p)))
            coeffs.append(-self.g_x)
            
            # Longitudinal Z
            if abs(self.g_z) > 1e-12:
                p = ["I"] * self.num_sites
                p[i] = "Z"
                terms.append("".join(reversed(p)))
                coeffs.append(-self.g_z)
                
        return SparsePauliOp.from_list(list(zip(terms, coeffs))).simplify()

    def get_local_hamiltonian(self, n: int) -> SparsePauliOp:
        """
        Local energy density at site n.
        Includes onsite fields and HALF of the coupling to neighbors.
        """
        # Inverse map index n -> (x, y)
        y = n // self.Lx
        x = n % self.Lx
        
        terms = []
        coeffs = []
        
        # Fields
        p = ["I"] * self.num_sites; p[n] = "X"
        terms.append("".join(reversed(p))); coeffs.append(-self.g_x)
        
        if abs(self.g_z) > 1e-12:
            p = ["I"] * self.num_sites; p[n] = "Z"
            terms.append("".join(reversed(p))); coeffs.append(-self.g_z)
            
        # Bonds (Horizontal)
        # Right
        if x + 1 < self.Lx:
            nj = self._coord_to_index(x+1, y)
            p = ["I"] * self.num_sites; p[n]="Z"; p[nj]="Z"
            terms.append("".join(reversed(p))); coeffs.append(-0.5 * self.J)
        # Left
        if x - 1 >= 0:
            nj = self._coord_to_index(x-1, y)
            p = ["I"] * self.num_sites; p[nj]="Z"; p[n]="Z"
            terms.append("".join(reversed(p))); coeffs.append(-0.5 * self.J)
            
        # Bonds (Vertical)
        # Top
        if y + 1 < self.Ly:
            nj = self._coord_to_index(x, y+1)
            p = ["I"] * self.num_sites; p[n]="Z"; p[nj]="Z"
            terms.append("".join(reversed(p))); coeffs.append(-0.5 * self.J)
        # Bottom
        if y - 1 >= 0:
            nj = self._coord_to_index(x, y-1)
            p = ["I"] * self.num_sites; p[nj]="Z"; p[n]="Z"
            terms.append("".join(reversed(p))); coeffs.append(-0.5 * self.J)
            
        return SparsePauliOp.from_list(list(zip(terms, coeffs))).simplify()

    def get_trotter_layers(self) -> List[SparsePauliOp]:
        """
        Trotter layers:
        1. X terms + Z fields (Onsite)
        2. Horizontal ZZ bonds (Even)
        3. Horizontal ZZ bonds (Odd)
        4. Vertical ZZ bonds (Even)
        5. Vertical ZZ bonds (Odd)
        """
        layers = []
        
        # 1. Onsite
        onsite = []
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites; p[i] = "X"
            onsite.append(("".join(reversed(p)), -self.g_x))
            if abs(self.g_z) > 1e-12:
                 p_z = ["I"] * self.num_sites; p_z[i] = "Z"
                 onsite.append(("".join(reversed(p_z)), -self.g_z))
        if onsite: layers.append(SparsePauliOp.from_list(onsite).simplify())
        
        # 2. Horizontal Bonds
        h_even = []
        h_odd = []
        for y in range(self.Ly):
            for x in range(self.Lx - 1): # No PBC support in trotter helper for now for simplicity
                i = self._coord_to_index(x, y)
                j = self._coord_to_index(x+1, y)
                p = ["I"] * self.num_sites; p[i]="Z"; p[j]="Z"
                term = ("".join(reversed(p)), -self.J)
                if x % 2 == 0: h_even.append(term)
                else: h_odd.append(term)
        if h_even: layers.append(SparsePauliOp.from_list(h_even).simplify())
        if h_odd: layers.append(SparsePauliOp.from_list(h_odd).simplify())
        
        # 3. Vertical Bonds
        v_even = []
        v_odd = []
        for x in range(self.Lx):
            for y in range(self.Ly - 1):
                i = self._coord_to_index(x, y)
                j = self._coord_to_index(x, y+1)
                p = ["I"] * self.num_sites; p[i]="Z"; p[j]="Z"
                term = ("".join(reversed(p)), -self.J)
                if y % 2 == 0: v_even.append(term)
                else: v_odd.append(term)
        if v_even: layers.append(SparsePauliOp.from_list(v_even).simplify())
        if v_odd: layers.append(SparsePauliOp.from_list(v_odd).simplify())
        
        return layers

    def build_operator_pool(self) -> List[SparsePauliOp]:
        """
        Operator pool for 2D Ising ADAPT-VQE.
        Standard pool: Single-site X and Nearest-neighbor ZZ.
        """
        pool = []
        
        # 1. Single-site X operators
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites
            p[i] = "X"
            pool.append(SparsePauliOp("".join(reversed(p))))
            
        # 2. Horizontal ZZ bonds
        for y in range(self.Ly):
            for x in range(self.Lx - 1):
                i = self._coord_to_index(x, y)
                j = self._coord_to_index(x+1, y)
                p = ["I"] * self.num_sites
                p[i] = "Z"; p[j] = "Z"
                pool.append(SparsePauliOp("".join(reversed(p))))
                
        # 3. Vertical ZZ bonds
        for x in range(self.Lx):
            for y in range(self.Ly - 1):
                i = self._coord_to_index(x, y)
                j = self._coord_to_index(x, y+1)
                p = ["I"] * self.num_sites
                p[i] = "Z"; p[j] = "Z"
                pool.append(SparsePauliOp("".join(reversed(p))))
                
        return pool

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="2D Ising Model",
            description=f"2D Square Lattice {self.Lx}x{self.Ly}. Gx={self.g_x}, Gz={self.g_z}",
            citation="arXiv:2505.03111"
        )
