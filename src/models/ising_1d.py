"""
1D Transverse-Field Ising Model.

Hamiltonian (Paper Convention):
H = - sum_n [ 0.5 * Z_n Z_{n+1} + g_x * X_n + g_z * Z_n ]
"""

from typing import List, Optional
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .base import PhysicsModel, Symmetry, ModelMetadata


class IsingModel1D(PhysicsModel):
    """
    1D Ising Model with Transverse and Longitudinal fields.
    
    Hamiltonian is normalized according to Farrell et al. (2025):
    H = - sum [ 0.5*Z_i*Z_{i+1} + g_x*X_i + g_z*Z_i ]
    """
    
    def __init__(self, num_sites: int, g_x: float = 1.0, g_z: float = 0.0, pbc: bool = True):
        self.g_x = g_x
        self.g_z = g_z
        # Note: j_int is fixed at 0.5 in this convention
        self.j_int = 0.5
        super().__init__(num_sites, pbc)

    def _validate_parameters(self) -> None:
        if self.num_sites < 2:
            raise ValueError("System size must be >= 2")

    def build_hamiltonian(self) -> SparsePauliOp:
        terms = []
        coeffs = []
        
        # ZZ interaction: -0.5 * sum Z_i Z_{i+1}
        for i in range(self.num_bonds):
            next_i = (i + 1) % self.num_sites
            p = ["I"] * self.num_sites
            p[i] = "Z"
            p[next_i] = "Z"
            terms.append("".join(reversed(p)))
            coeffs.append(-self.j_int)
            
        # Transverse Field: -g_x * sum X_i
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites
            p[i] = "X"
            terms.append("".join(reversed(p)))
            coeffs.append(-self.g_x)
            
        # Longitudinal Field: -g_z * sum Z_i
        if abs(self.g_z) > 1e-12:
            for i in range(self.num_sites):
                p = ["I"] * self.num_sites
                p[i] = "Z"
                terms.append("".join(reversed(p)))
                coeffs.append(-self.g_z)
                
        return SparsePauliOp.from_list(list(zip(terms, coeffs))).simplify()

    def build_operator_pool(self, pool_type: str = "global") -> List[SparsePauliOp]:
        """
        Construct operator pool for ADAPT-VQE.
        
        Symmetry-preserving pool from Farrell et al. (2025):
        O1 = sum Y_n
        O2 = sum Z_n Y_{n+1} Z_{n+2}
        O3 = sum (Y_n Z_{n+1} + Z_n Y_{n+1})
        O4 = sum (Y_n X_{n+1} + X_n Y_{n+1})
        O5 = sum (Z_n X_{n+1} Y_{n+2} + Y_n X_{n+1} Z_{n+2})
        """
        if pool_type == "local":
            # Fallback to site-local operators if needed for breaking symmetry
            pool = []
            for i in range(self.num_sites):
                p = ["I"] * self.num_sites; p[i] = "Y"
                pool.append(SparsePauliOp.from_list([("".join(reversed(p)), 1.0)]))
            return pool

        # Global Sums (Translationally Invariant)
        pool_labels = []
        
        # O1: sum Y_n
        o1_terms = []
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites; p[i] = "Y"
            o1_terms.append(("".join(reversed(p)), 1.0))
        pool_labels.append(SparsePauliOp.from_list(o1_terms).simplify())
        
        # O3: sum (Y_n Z_{n+1} + Z_n Y_{n+1})
        o3_terms = []
        for i in range(self.num_sites):
            next_i = (i + 1) % self.num_sites
            if not self.pbc and i == self.num_sites - 1: continue
            p1 = ["I"] * self.num_sites; p1[i] = "Y"; p1[next_i] = "Z"
            p2 = ["I"] * self.num_sites; p2[i] = "Z"; p2[next_i] = "Y"
            o3_terms.append(("".join(reversed(p1)), 1.0))
            o3_terms.append(("".join(reversed(p2)), 1.0))
        pool_labels.append(SparsePauliOp.from_list(o3_terms).simplify())
        
        # O4: sum (Y_n X_{n+1} + X_n Y_{n+1})
        o4_terms = []
        for i in range(self.num_sites):
            next_i = (i + 1) % self.num_sites
            if not self.pbc and i == self.num_sites - 1: continue
            p1 = ["I"] * self.num_sites; p1[i] = "Y"; p1[next_i] = "X"
            p2 = ["I"] * self.num_sites; p2[i] = "X"; p2[next_i] = "Y"
            o4_terms.append(("".join(reversed(p1)), 1.0))
            o4_terms.append(("".join(reversed(p2)), 1.0))
        pool_labels.append(SparsePauliOp.from_list(o4_terms).simplify())

        # O2: sum Z_n Y_{n+1} Y_{n+2}
        if self.num_sites >= 3:
            o2_terms = []
            for i in range(self.num_sites):
                i1, i2 = (i+1)%self.num_sites, (i+2)%self.num_sites
                if not self.pbc and i >= self.num_sites - 2: continue
                p = ["I"] * self.num_sites; p[i]="Z"; p[i1]="Y"; p[i2]="Z"
                o2_terms.append(("".join(reversed(p)), 1.0))
            pool_labels.append(SparsePauliOp.from_list(o2_terms).simplify())
            
            # O5: sum (Z_n X_{n+1} Y_{n+2} + Y_n X_{n+1} Z_{n+2})
            o5_terms = []
            for i in range(self.num_sites):
                i1, i2 = (i+1)%self.num_sites, (i+2)%self.num_sites
                if not self.pbc and i >= self.num_sites - 2: continue
                p1 = ["I"] * self.num_sites; p1[i]="Z"; p1[i1]="X"; p1[i2]="Y"
                p2 = ["I"] * self.num_sites; p2[i]="Y"; p2[i1]="X"; p2[i2]="Z"
                o5_terms.append(("".join(reversed(p1)), 1.0))
                o5_terms.append(("".join(reversed(p2)), 1.0))
            pool_labels.append(SparsePauliOp.from_list(o5_terms).simplify())

        return pool_labels

    def get_trotter_layers(self) -> List[SparsePauliOp]:
        # Odd-Even bond split
        layers = []
        
        # Z-terms (Diagonal)
        if abs(self.g_z) > 1e-12:
            diag_terms = []
            for i in range(self.num_sites):
                p = ["I"] * self.num_sites; p[i] = "Z"
                diag_terms.append(("".join(reversed(p)), -self.g_z))
            if diag_terms:
                layers.append(SparsePauliOp.from_list(diag_terms).simplify())
            
        # X-terms (Off-diagonal, all commute)
        x_terms = []
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites; p[i] = "X"
            x_terms.append(("".join(reversed(p)), -self.g_x))
        if x_terms:
            layers.append(SparsePauliOp.from_list(x_terms).simplify())
            
        # ZZ-terms (Non-commuting across bonds)
        # Layer even bonds
        even_zz = []
        for i in range(0, self.num_bonds, 2):
            next_i = (i + 1) % self.num_sites
            p = ["I"] * self.num_sites; p[i] = "Z"; p[next_i] = "Z"
            even_zz.append(("".join(reversed(p)), -self.j_int))
        if even_zz: layers.append(SparsePauliOp.from_list(even_zz).simplify())
        
        # Layer odd bonds
        odd_zz = []
        for i in range(1, self.num_bonds, 2):
            next_i = (i + 1) % self.num_sites
            p = ["I"] * self.num_sites; p[i] = "Z"; p[next_i] = "Z"
            odd_zz.append(("".join(reversed(p)), -self.j_int))
        if odd_zz: layers.append(SparsePauliOp.from_list(odd_zz).simplify())
            
        return layers

    def get_local_hamiltonian(self, n: int) -> SparsePauliOp:
        """
        Returns the local energy density operator E_n.
        E_n = - [ 0.25*(Z_n Z_{n+1} + Z_{n-1} Z_n) + g_x*X_n + g_z*Z_n ]
        This ensures sum E_n = H.
        """
        terms = []
        coeffs = []
        
        # Site index n
        n0 = n
        n1 = (n + 1) % self.num_sites
        nm1 = (n - 1) % self.num_sites
        
        # Interactions (half of each bond touching site n)
        # Bond n -- n+1
        p_fwd = ["I"] * self.num_sites; p_fwd[n0] = "Z"; p_fwd[n1] = "Z"
        terms.append("".join(reversed(p_fwd)))
        coeffs.append(-0.5 * self.j_int * 0.5) # Factor of 0.5 because bond is shared
        
        # Bond n-1 -- n
        p_bck = ["I"] * self.num_sites; p_bck[nm1] = "Z"; p_bck[n0] = "Z"
        terms.append("".join(reversed(p_bck)))
        coeffs.append(-0.5 * self.j_int * 0.5)
        
        # Field terms
        p_x = ["I"] * self.num_sites; p_x[n0] = "X"
        terms.append("".join(reversed(p_x)))
        coeffs.append(-self.g_x)
        
        if abs(self.g_z) > 1e-12:
            p_z = ["I"] * self.num_sites; p_z[n0] = "Z"
            terms.append("".join(reversed(p_z)))
            coeffs.append(-self.g_z)
            
        return SparsePauliOp.from_list(list(zip(terms, coeffs))).simplify()

    def get_symmetries(self) -> List[Symmetry]:
        syms = [Symmetry.PARITY]
        if self.pbc: syms.append(Symmetry.TRANSLATION)
        if abs(self.g_z) < 1e-12: syms.append(Symmetry.Z2)
        return syms

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="1D Ising Model (Farrell Convention)",
            description=f"TFIM with gx={self.g_x}, gz={self.g_z}, JZZ=0.5. {'PBC' if self.pbc else 'OBC'}.",
            critical_points={"gx": 1.0, "gz": 0.0},
            citation="arXiv:2505.03111"
        )
