"""
Unit tests for Physics Models.
Updated to Farrell et al. (2025) and Yao et al. (2023) conventions.
"""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from src.models.ising_1d import IsingModel1D
from src.models.ising_2d import IsingModel2D
from src.models.su2 import SU2GaugeModel
from src.models.base import Symmetry


def is_hermitian(op: SparsePauliOp, atol: float = 1e-10) -> bool:
    mat = op.to_matrix()
    return np.allclose(mat, mat.conj().T, atol=atol)


class TestIsingModel1D:
    def test_hamiltonian_normalization(self):
        # Farrell et al. 2025: H = - sum [0.5 ZZ + gx X + gz Z]
        model = IsingModel1D(num_sites=2, g_x=1.0, g_z=0.0, pbc=False)
        H = model.build_hamiltonian()
        
        # Site 0,1: -0.5 Z0 Z1 - 1.0 X0 - 1.0 X1
        # Qiskit label Z0 Z1 is "ZZ"
        terms = H.to_list()
        labels = [t[0] for t in terms]
        coeffs = [t[1].real for t in terms]
        
        assert "ZZ" in labels
        idx = labels.index("ZZ")
        assert np.isclose(coeffs[idx], -0.5)

    def test_global_pool_size(self):
        # Always 5 operators for N>=3
        model = IsingModel1D(num_sites=4, pbc=True)
        pool = model.build_operator_pool(pool_type="global")
        assert len(pool) == 5
        for op in pool:
            # Operators should be Hermitian (sums of Paulis)
            assert is_hermitian(op)
            # Should be sums of terms
            assert len(op.paulis) > 1

class TestIsingModel2D:
    def test_2d_instantiation(self):
        model = IsingModel2D(Lx=3, Ly=3, g_x=3.044)
        assert model.num_sites == 9
        H = model.build_hamiltonian()
        assert is_hermitian(H)
        
    def test_2d_pool(self):
        model = IsingModel2D(Lx=2, Ly=2)
        pool = model.build_operator_pool()
        assert len(pool) >= 4 # At least O1-O4
        for op in pool:
            assert is_hermitian(op)

class TestSU2GaugeModel:
    def test_hz_normalization(self):
        # Yao et al. 2023: hz = 3g^2/8
        model = SU2GaugeModel(num_sites=2, g=1.0, a=1.0)
        c = model.coupling_constants
        assert np.isclose(c["h_z"], 3/8.0)
        
    def test_global_pool(self):
        # O1, O2, O3, O4
        model = SU2GaugeModel(num_sites=4, pbc=True)
        pool = model.build_operator_pool(pool_type="global")
        assert len(pool) == 4
