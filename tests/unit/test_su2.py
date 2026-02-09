"""
Unit tests for SU(2) Gauge Model.
"""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from src.models.su2 import SU2GaugeModel
from src.models.base import Symmetry

def is_hermitian(op: SparsePauliOp, atol: float = 1e-10) -> bool:
    mat = op.to_matrix()
    return np.allclose(mat, mat.conj().T, atol=atol)

class TestSU2GaugeModel:
    
    def test_instantiation(self):
        model = SU2GaugeModel(num_sites=4, g=1.5, a=0.5, pbc=True)
        assert model.num_sites == 4
        assert model.g == 1.5
        assert model.a == 0.5
        assert model.pbc == True
        
    def test_parameter_validation(self):
        with pytest.raises(ValueError):
            SU2GaugeModel(num_sites=4, g=-1.0)
        with pytest.raises(ValueError):
            SU2GaugeModel(num_sites=4, a=0.0)

    def test_coupling_constants(self):
        g, a = 1.0, 1.0
        model = SU2GaugeModel(num_sites=4, g=g, a=a)
        c = model.coupling_constants
        
        expected_J = -3 * a * g**2 / 16.0
        assert np.isclose(c["J"], expected_J)
        
    def test_hamiltonian_hermiticity(self):
        model = SU2GaugeModel(num_sites=4, pbc=True)
        H = model.build_hamiltonian()
        assert is_hermitian(H)
        
    def test_operator_pool_hermiticity(self):
        model = SU2GaugeModel(num_sites=3, pbc=False)
        pool = model.build_operator_pool()
        for op in pool:
            # Pool operators are Y-like (anti-hermitian usually in ADAPT? No, usually Pauli strings are Hermitian or anti-Hermitian)
            # Y is Hermitian. YZ is Hermitian.
            # Wait, ADAPT-VQE usually uses A_k such that U = exp(theta * A_k)
            # If A_k is anti-Hermitian (e.g. iY), then U is unitary.
            # If A_k is Hermitian (e.g. Y), then U = exp(i * theta * A_k).
            # SparsePauliOp stores Pauli strings, which are Hermitian (Y is Hermitian: [[0, -i], [i, 0]]).
            assert is_hermitian(op)
            
    def test_operator_pool_size(self):
        # Implementation uses Global Symmetry-Respecting Pool (sums of terms).
        # We expect 4 global operators (O1, O2, O3, O4) for sufficient system size.
        
        model = SU2GaugeModel(num_sites=4, pbc=True)
        pool = model.build_operator_pool()
        # Should be exactly 4 global operators
        assert len(pool) == 4
        
        # For OBC on 3 sites (0, 1, 2):
        # O1 (Y) - yes
        # O2 (YZ) - yes (0-1, 1-2)
        # O3 (ZY) - yes (0-1, 1-2)
        # O4 (ZYZ) - yes (0-1-2)
        model_obc = SU2GaugeModel(num_sites=3, pbc=False)
        pool_obc = model_obc.build_operator_pool()
        assert len(pool_obc) == 4
        
    def test_trotter_layers(self):
        model = SU2GaugeModel(num_sites=5, pbc=True)
        layers = model.get_trotter_layers()
        # Should have [H_E, H_M_Even, H_M_Odd]
        assert len(layers) == 3 or len(layers) == 2 # If empty
        for layer in layers:
            assert is_hermitian(layer)
