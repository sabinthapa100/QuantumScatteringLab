"""
SU(2) Lattice Gauge Theory model.

Based on Kogut-Susskind formulation mapped to a plaquette chain.
Reference: arXiv:2308.16202 / Yao et al. (2023)
"""

from typing import List, Optional
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .base import PhysicsModel, Symmetry, ModelMetadata


class SU2GaugeModel(PhysicsModel):
    """
    SU(2) Lattice Gauge Theory mapped to a 1D spin chain.
    
    Hamiltonian Parameters (Yao/Thapa Convention):
        J = -3 * g^2 / 16
        h_z = 3 * g^2 / 8
        h_x = -2 / (a * g)^2
    
    Hamiltonian:
        H_E = J * sum(Z_i Z_{i+1}) + h_z * sum(Z_i)
        H_M = (h_x/16) * sum( X_i - 3 Z_{i-1} X_i - 3 X_i Z_{i+1} + 9 Z_{i-1} X_i Z_{i+1} )
    """
    
    def __init__(self, num_sites: int, g: float = 1.0, a: float = 1.0, pbc: bool = True):
        self.g = g
        self.a = a
        super().__init__(num_sites, pbc)
    
    def _validate_parameters(self) -> None:
        if self.g <= 0 or self.a <= 0:
            raise ValueError("Coupling g and lattice spacing a must be positive.")

    @property
    def coupling_constants(self) -> dict:
        """Derived couplings from LaTeX source."""
        g2 = self.g**2
        return {
            "J": -3 * g2 / 16.0,
            "h_z": 3 * g2 / 8.0,
            "h_x": -2.0 / ((self.a * self.g)**2)
        }

    def build_hamiltonian(self) -> SparsePauliOp:
        c = self.coupling_constants
        J, h_z, h_x = c["J"], c["h_z"], c["h_x"]
        
        terms = []
        coeffs = []
        
        # H_E Electric
        for i in range(self.num_bonds):
            next_i = (i + 1) % self.num_sites
            p = ["I"] * self.num_sites; p[i] = "Z"; p[next_i] = "Z"
            terms.append("".join(reversed(p))); coeffs.append(J)
            
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites; p[i] = "Z"
            terms.append("".join(reversed(p))); coeffs.append(h_z)
            
        # H_M Magnetic
        hx_scaled = h_x / 16.0
        for i in range(self.num_sites):
            # Base X: X_i
            p = ["I"] * self.num_sites; p[i] = "X"
            terms.append("".join(reversed(p))); coeffs.append(hx_scaled)
            
            prev_i = (i - 1) % self.num_sites if self.pbc else (i - 1 if i > 0 else None)
            next_i = (i + 1) % self.num_sites if self.pbc else (i + 1 if i < self.num_sites - 1 else None)
            
            if prev_i is not None:
                p = ["I"] * self.num_sites; p[prev_i] = "Z"; p[i] = "X"
                terms.append("".join(reversed(p))); coeffs.append(-3 * hx_scaled)
            if next_i is not None:
                p = ["I"] * self.num_sites; p[i] = "X"; p[next_i] = "Z"
                terms.append("".join(reversed(p))); coeffs.append(-3 * hx_scaled)
            if prev_i is not None and next_i is not None:
                p = ["I"] * self.num_sites; p[prev_i] = "Z"; p[i] = "X"; p[next_i] = "Z"
                terms.append("".join(reversed(p))); coeffs.append(9 * hx_scaled)
                
        return SparsePauliOp.from_list(list(zip(terms, coeffs))).simplify()

    def build_operator_pool(self, pool_type: str = "global") -> List[SparsePauliOp]:
        """
        Global Symmetry-Respecting Pool (LaTeX):
        O1 = sum Y_i
        O2 = sum Y_i Z_{i+1}
        O3 = sum Z_i Y_{i+1}
        O4 = sum Z_{i-1} Y_i Z_{i+1}
        """
        if pool_type == "local":
            # Site-local fallback
            pool = []
            for i in range(self.num_sites):
                p = ["I"] * self.num_sites; p[i] = "Y"
                pool.append(SparsePauliOp.from_list([("".join(reversed(p)), 1.0)]))
            return pool

        pool = []
        
        # O1: sum Y_i
        pool.append(SparsePauliOp.from_list([
            ("".join(reversed(["Y" if j==i else "I" for j in range(self.num_sites)])), 1.0) 
            for i in range(self.num_sites)
        ]).simplify())
        
        # O2: sum Y_i Z_{i+1}
        o2_list = []
        for i in range(self.num_sites):
            next_i = (i + 1) % self.num_sites
            if not self.pbc and i == self.num_sites - 1: continue
            p = ["I"] * self.num_sites; p[i] = "Y"; p[next_i] = "Z"
            o2_list.append(("".join(reversed(p)), 1.0))
        if o2_list: pool.append(SparsePauliOp.from_list(o2_list).simplify())
        
        # O3: sum Z_i Y_{i+1}
        o3_list = []
        for i in range(self.num_sites):
            next_i = (i + 1) % self.num_sites
            if not self.pbc and i == self.num_sites - 1: continue
            p = ["I"] * self.num_sites; p[i] = "Z"; p[next_i] = "Y"
            o3_list.append(("".join(reversed(p)), 1.0))
        if o3_list: pool.append(SparsePauliOp.from_list(o3_list).simplify())
        
        # O4: sum Z_{i-1} Y_i Z_{i+1}
        o4_list = []
        for i in range(self.num_sites):
            prev_i = (i - 1) % self.num_sites
            next_i = (i + 1) % self.num_sites
            if not self.pbc and (i == 0 or i == self.num_sites - 1): continue
            p = ["I"] * self.num_sites; p[prev_i] = "Z"; p[i] = "Y"; p[next_i] = "Z"
            o4_list.append(("".join(reversed(p)), 1.0))
        if o4_list: pool.append(SparsePauliOp.from_list(o4_list).simplify())
        
        return pool

    def get_trotter_layers(self) -> List[SparsePauliOp]:
        # H_E layer, H_M_Even layer, H_M_Odd layer
        c = self.coupling_constants
        J, h_z, h_x = c["J"], c["h_z"], c["h_x"]
        
        # Layer 1: H_E (All Z terms)
        he_terms = []
        for i in range(self.num_bonds):
            next_i = (i + 1) % self.num_sites
            p = ["I"] * self.num_sites; p[i] = "Z"; p[next_i] = "Z"
            he_terms.append(("".join(reversed(p)), J))
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites; p[i] = "Z"
            he_terms.append(("".join(reversed(p)), h_z))
        
        layers = [SparsePauliOp.from_list(he_terms).simplify()]
        
        # Layer 2 & 3: H_M Even/Odd
        hx_scaled = h_x / 16.0
        for parity in [0, 1]:
            layer_terms = []
            for i in range(parity, self.num_sites, 2):
                p = ["I"] * self.num_sites; p[i] = "X"
                layer_terms.append(("".join(reversed(p)), hx_scaled))
                
                prev_i = (i - 1) % self.num_sites if self.pbc else (i - 1 if i > 0 else None)
                next_i = (i + 1) % self.num_sites if self.pbc else (i + 1 if i < self.num_sites - 1 else None)
                
                if prev_i is not None:
                    p = ["I"] * self.num_sites; p[prev_i] = "Z"; p[i] = "X"
                    layer_terms.append(("".join(reversed(p)), -3 * hx_scaled))
                if next_i is not None:
                    p = ["I"] * self.num_sites; p[i] = "X"; p[next_i] = "Z"
                    layer_terms.append(("".join(reversed(p)), -3 * hx_scaled))
                if prev_i is not None and next_i is not None:
                    p = ["I"] * self.num_sites; p[prev_i] = "Z"; p[i] = "X"; p[next_i] = "Z"
                    layer_terms.append(("".join(reversed(p)), 9 * hx_scaled))
            
            if layer_terms:
                layers.append(SparsePauliOp.from_list(layer_terms).simplify())
                
        return layers

    def get_symmetries(self) -> List[Symmetry]:
        return [Symmetry.PARITY, Symmetry.TRANSLATION] if self.pbc else [Symmetry.PARITY]

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="SU(2) Plaquette Chain",
            description=f"Mapped gauge theory with g={self.g}, a={self.a}. hz coeff follows Yao et al. (3g^2/8).",
            critical_points={},
            citation="arXiv:2308.16202"
        )
