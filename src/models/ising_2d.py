"""
2D Ising Field Theory Model.

Square lattice with Periodic Boundary Conditions (PBC).
Maps (nx, ny) to index = ny * Lx + nx.

Reference: arXiv:2505.03111 / Farrell et al. (2025)
"""

from typing import List, Optional
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .base import PhysicsModel, Symmetry, ModelMetadata


class IsingModel2D(PhysicsModel):
    """
    2D Ising Model on a square lattice.
    
    Hamiltonian:
    H = - sum_{<i,j>} Z_i Z_j - sum_i [ g_x X_i + g_z Z_i ]
    
    Parameters:
        num_sites (int): Total spins (Nx * Ny).
        Lx (int): Length in x direction.
        g_x (float): Transverse field.
        g_z (float): Longitudinal field.
    """
    
    def __init__(self, Lx: int, Ly: int, g_x: float = 3.04438, g_z: float = 0.0, pbc: bool = True):
        self.Lx = Lx
        self.Ly = Ly
        self.g_x = g_x
        self.g_z = g_z
        num_sites = Lx * Ly
        super().__init__(num_sites, pbc)

    def _validate_parameters(self) -> None:
        if self.Lx < 2 or self.Ly < 2:
            raise ValueError("2D lattice dimensions must be >= 2x2")

    def _get_idx(self, nx: int, ny: int) -> int:
        """Map 2D coordinates to 1D index."""
        nx = nx % self.Lx
        ny = ny % self.Ly
        return ny * self.Lx + nx

    def build_hamiltonian(self) -> SparsePauliOp:
        terms = []
        coeffs = []
        
        # ZZ Nearest Neighbors (Horizontal and Vertical)
        for ny in range(self.Ly):
            for nx in range(self.Lx):
                i = self._get_idx(nx, ny)
                
                # Horizontal neighbor
                j_x = self._get_idx(nx + 1, ny)
                p = ["I"] * self.num_sites; p[i] = "Z"; p[j_x] = "Z"
                terms.append("".join(reversed(p))); coeffs.append(-1.0)
                
                # Vertical neighbor
                j_y = self._get_idx(nx, ny + 1)
                p = ["I"] * self.num_sites; p[i] = "Z"; p[j_y] = "Z"
                terms.append("".join(reversed(p))); coeffs.append(-1.0)
                
        # Fields
        for i in range(self.num_sites):
            pX = ["I"] * self.num_sites; pX[i] = "X"
            terms.append("".join(reversed(pX))); coeffs.append(-self.g_x)
            
            if abs(self.g_z) > 1e-12:
                pZ = ["I"] * self.num_sites; pZ[i] = "Z"
                terms.append("".join(reversed(pZ))); coeffs.append(-self.g_z)
                
        return SparsePauliOp.from_list(list(zip(terms, coeffs))).simplify()

    def get_local_hamiltonian(self, nx: int, ny: int) -> SparsePauliOp:
        """
        Returns the local energy density operator E_{nx,ny}.
        Includes 0.5 of each bond touching site (nx,ny).
        """
        idx = self._get_idx(nx, ny)
        neighbor_indices = [
            self._get_idx(nx + 1, ny),
            self._get_idx(nx - 1, ny),
            self._get_idx(nx, ny + 1),
            self._get_idx(nx, ny - 1)
        ]
        
        terms = []
        coeffs = []
        
        # Interactions (0.5 for each shared bond)
        for j in neighbor_indices:
            p = ["I"] * self.num_sites; p[idx] = "Z"; p[j] = "Z"
            terms.append("".join(reversed(p))); coeffs.append(-0.5)
            
        # Fields
        pX = ["I"] * self.num_sites; pX[idx] = "X"
        terms.append("".join(reversed(pX))); coeffs.append(-self.g_x)
        
        if abs(self.g_z) > 1e-12:
            pZ = ["I"] * self.num_sites; pZ[idx] = "Z"
            terms.append("".join(reversed(pZ))); coeffs.append(-self.g_z)
            
        return SparsePauliOp.from_list(list(zip(terms, coeffs))).simplify()

    def build_operator_pool(self, pool_type: str = "global") -> List[SparsePauliOp]:
        """
        8-operator Global Symmetry-Preserving Pool (Farrell et al. 2025).
        """
        if pool_type == "local":
            return [SparsePauliOp.from_list([("".join(reversed(["Y" if j==i else "I" for j in range(self.num_sites)])), 1.0)]) for i in range(self.num_sites)]

        pool = []
        
        # O1: sum Y_{nx,ny}
        o1 = []
        for i in range(self.num_sites):
            p = ["I"] * self.num_sites; p[i] = "Y"
            o1.append(("".join(reversed(p)), 1.0))
        pool.append(SparsePauliOp.from_list(o1).simplify())
        
        # O2: sum (Y Z_x + Z Y_x + Y Z_y + Z Y_y)
        o2 = []
        for ny in range(self.Ly):
            for nx in range(self.Lx):
                idx = self._get_idx(nx, ny)
                idx_x = self._get_idx(nx + 1, ny)
                idx_y = self._get_idx(nx, ny + 1)
                for i1, i2 in [(idx, idx_x), (idx_x, idx), (idx, idx_y), (idx_y, idx)]:
                    p = ["I"] * self.num_sites; p[i1] = "Y"; p[i2] = "Z"
                    o2.append(("".join(reversed(p)), 1.0))
        pool.append(SparsePauliOp.from_list(o2).simplify())
        
        # O3: sum (Y X_x + X Y_x + Y X_y + X Y_y)
        o3 = []
        for ny in range(self.Ly):
            for nx in range(self.Lx):
                idx = self._get_idx(nx, ny)
                idx_x = self._get_idx(nx + 1, ny)
                idx_y = self._get_idx(nx, ny + 1)
                for i1, i2 in [(idx, idx_x), (idx_x, idx), (idx, idx_y), (idx_y, idx)]:
                    p = ["I"] * self.num_sites; p[i1] = "Y"; p[i2] = "X"
                    o3.append(("".join(reversed(p)), 1.0))
        pool.append(SparsePauliOp.from_list(o3).simplify())
        
        # O4: sum (Z Y_x_x' Z + Z Y_y_y' Z) -> Z_n Y_{n+1} Z_{n+2} in x and y
        o4 = []
        for ny in range(self.Ly):
            for nx in range(self.Lx):
                i0 = self._get_idx(nx, ny)
                # x-direction
                i1, i2 = self._get_idx(nx + 1, ny), self._get_idx(nx + 2, ny)
                p = ["I"] * self.num_sites; p[i0] = "Z"; p[i1] = "Y"; p[i2] = "Z"
                o4.append(("".join(reversed(p)), 1.0))
                # y-direction
                i1, i2 = self._get_idx(nx, ny + 1), self._get_idx(nx, ny + 2)
                p = ["I"] * self.num_sites; p[i0] = "Z"; p[i1] = "Y"; p[i2] = "Z"
                o4.append(("".join(reversed(p)), 1.0))
        pool.append(SparsePauliOp.from_list(o4).simplify())

        # O5: sum (Z X Y + Y X Z) - simplified to Z_n X_{n+1} Y_{n+2} pattern
        o5 = []
        for ny in range(self.Ly):
            for nx in range(self.Lx):
                i0 = self._get_idx(nx, ny)
                for dx, dy in [(1, 0), (0, 1)]: # x and y directions
                    i1, i2 = self._get_idx(nx+dx, ny+dy), self._get_idx(nx+2*dx, ny+2*dy)
                    p1 = ["I"] * self.num_sites; p1[i0]="Z"; p1[i1]="X"; p1[i2]="Y"
                    p2 = ["I"] * self.num_sites; p2[i0]="Y"; p2[i1]="X"; p2[i2]="Z"
                    o5.extend([("".join(reversed(p1)), 1.0), ("".join(reversed(p2)), 1.0)])
        pool.append(SparsePauliOp.from_list(o5).simplify())

        # O6: sum (Z Y Z + Z Y Z) - L-shapes (4 orientations)
        o6 = []
        for ny in range(self.Ly):
            for nx in range(self.Lx):
                i00 = self._get_idx(nx, ny)
                # 1. Z(x,y) Y(x+1,y) Z(x+1,y+1)
                i10 = self._get_idx(nx+1, ny)
                i11 = self._get_idx(nx+1, ny+1)
                p = ["I"] * self.num_sites; p[i00]="Z"; p[i10]="Y"; p[i11]="Z"
                o6.append(("".join(reversed(p)), 1.0))
                
                # 2. Z(x,y) Y(x,y+1) Z(x-1,y+1)
                i01 = self._get_idx(nx, ny+1)
                im11 = self._get_idx(nx-1, ny+1)
                p = ["I"] * self.num_sites; p[i00]="Z"; p[i01]="Y"; p[im11]="Z"
                o6.append(("".join(reversed(p)), 1.0))
                
                # 3. Z(x,y) Y(x-1,y) Z(x-1,y-1)
                im10 = self._get_idx(nx-1, ny)
                im1m1 = self._get_idx(nx-1, ny-1)
                p = ["I"] * self.num_sites; p[i00]="Z"; p[im10]="Y"; p[im1m1]="Z"
                o6.append(("".join(reversed(p)), 1.0))
                
                # 4. Z(x,y) Y(x,y-1) Z(x+1,y-1)
                i0m1 = self._get_idx(nx, ny-1)
                i1m1 = self._get_idx(nx+1, ny-1)
                p = ["I"] * self.num_sites; p[i00]="Z"; p[i0m1]="Y"; p[i1m1]="Z"
                o6.append(("".join(reversed(p)), 1.0))
        pool.append(SparsePauliOp.from_list(o6).simplify())

        # O7: sum (Z X Y + Y X Z) - L-shapes with X middle (8 orientations)
        o7 = []
        for ny in range(self.Ly):
            for nx in range(self.Lx):
                idx = self._get_idx(nx, ny)
                
                # Relative coordinates for the 8 L-shape patterns around (nx,ny)
                # Patterns from LaTeX lines 189-196
                # 1. Z(xy) X(x+1,y) Y(x+1,y+1)
                # 2. Z(xy) X(x,y+1) Y(x-1,y+1)
                # 3. Z(xy) X(x-1,y) Y(x-1,y-1)
                # 4. Z(xy) X(x,y-1) Y(x+1,y-1)
                # 5-8. Swap Y and Z in 1-4
                
                shifts = [
                    ((1,0), (1,1)),   # 1
                    ((0,1), (-1,1)),  # 2
                    ((-1,0), (-1,-1)),# 3
                    ((0,-1), (1,-1))  # 4
                ]
                
                for (dx1, dy1), (dx2, dy2) in shifts:
                    i1 = self._get_idx(nx+dx1, ny+dy1)
                    i2 = self._get_idx(nx+dx2, ny+dy2)
                    
                    # Z X Y
                    p1 = ["I"] * self.num_sites; p1[idx]="Z"; p1[i1]="X"; p1[i2]="Y"
                    o7.append(("".join(reversed(p1)), 1.0))
                    
                    # Y X Z
                    p2 = ["I"] * self.num_sites; p2[idx]="Y"; p2[i1]="X"; p2[i2]="Z"
                    o7.append(("".join(reversed(p2)), 1.0))
        pool.append(SparsePauliOp.from_list(o7).simplify())
        
        # O8: sum (Y Z Z + Z Z Y + Z Y) - 3-body clusters + 2-body
        o8 = []
        for ny in range(self.Ly):
            for nx in range(self.Lx):
                idx = self._get_idx(nx, ny)
                
                # Patterns from LaTeX lines 200-205
                # 1. Y(xy) Z(x+1,y) Z(x-1,y) (Horizontal sandwich)
                i1, i2 = self._get_idx(nx+1, ny), self._get_idx(nx-1, ny)
                p = ["I"] * self.num_sites; p[idx]="Y"; p[i1]="Z"; p[i2]="Z"
                o8.append(("".join(reversed(p)), 1.0))
                
                # 2. Y(xy) Z(x,y+1) Z(x,y-1) (Vertical sandwich)
                i1, i2 = self._get_idx(nx, ny+1), self._get_idx(nx, ny-1)
                p = ["I"] * self.num_sites; p[idx]="Y"; p[i1]="Z"; p[i2]="Z"
                o8.append(("".join(reversed(p)), 1.0))
                
                # 3. Y(xy) Z(x+1,y) Z(x,y+1) (Corner 1)
                i1, i2 = self._get_idx(nx+1, ny), self._get_idx(nx, ny+1)
                p = ["I"] * self.num_sites; p[idx]="Y"; p[i1]="Z"; p[i2]="Z"
                o8.append(("".join(reversed(p)), 1.0))
                
                # 4. Y(xy) Z(x-1,y) Z(x,y-1) (Corner 2)
                i1, i2 = self._get_idx(nx-1, ny), self._get_idx(nx, ny-1)
                p = ["I"] * self.num_sites; p[idx]="Y"; p[i1]="Z"; p[i2]="Z"
                o8.append(("".join(reversed(p)), 1.0))
                
                # 5. Y(xy) Z(x-1,y) Z(x,y+1) (Corner 3)
                i1, i2 = self._get_idx(nx-1, ny), self._get_idx(nx, ny+1)
                p = ["I"] * self.num_sites; p[idx]="Y"; p[i1]="Z"; p[i2]="Z"
                o8.append(("".join(reversed(p)), 1.0))
                
                # 6. Z(xy) Y(x,y-1) (Likely redundant with O2 but included for completeness)
                i1 = self._get_idx(nx, ny-1)
                p = ["I"] * self.num_sites; p[idx]="Z"; p[i1]="Y"
                o8.append(("".join(reversed(p)), 1.0))
                
        pool.append(SparsePauliOp.from_list(o8).simplify())

        return pool

    def get_trotter_layers(self) -> List[SparsePauliOp]:
        # X and Z fields (Layer 1)
        diag = []
        for i in range(self.num_sites):
            pX = ["I"] * self.num_sites; pX[i]="X"
            diag.append(("".join(reversed(pX)), -self.g_x))
            if abs(self.g_z) > 1e-12:
                pZ = ["I"] * self.num_sites; pZ[i]="Z"
                diag.append(("".join(reversed(pZ)), -self.g_z))
        
        layers = [SparsePauliOp.from_list(diag).simplify()]
        
        # ZZ interaction layers (Coloring the lattice bonds)
        # 4 layers needed for square lattice (Even-X, Odd-X, Even-Y, Odd-Y)
        for direction in ['x', 'y']:
            for parity in [0, 1]:
                zz_terms = []
                for ny in range(self.Ly):
                    for nx in range(self.Lx):
                        if direction == 'x' and nx % 2 == parity:
                            i, j = self._get_idx(nx, ny), self._get_idx(nx+1, ny)
                            p = ["I"] * self.num_sites; p[i]="Z"; p[j]="Z"
                            zz_terms.append(("".join(reversed(p)), -1.0))
                        elif direction == 'y' and ny % 2 == parity:
                            i, j = self._get_idx(nx, ny), self._get_idx(nx, ny+1)
                            p = ["I"] * self.num_sites; p[i]="Z"; p[j]="Z"
                            zz_terms.append(("".join(reversed(p)), -1.0))
                if zz_terms:
                    layers.append(SparsePauliOp.from_list(zz_terms).simplify())
        return layers

    def get_symmetries(self) -> List[Symmetry]:
        return [Symmetry.TRANSLATION, Symmetry.PARITY]

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="2D Ising Model",
            description=f"{self.Lx}x{self.Ly} square lattice. gx={self.g_x}. Maps nx,ny -> index ny*Lx + nx.",
            critical_points={"gx": 3.044},
            citation="Farrell et al. (2025)"
        )
