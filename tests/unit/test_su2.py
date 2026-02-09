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
        # PBC: 4 ops per site (Y, YZ, ZY, ZYZ) * N terms
        # But wait, pool is individual terms.
        # terms per site i:
        # Y_i (always)
        # Y_i Z_{i+1} (if right neighbor)
        # Z_{i-1} Y_i (if left neighbor)
        # Z_{i-1} Y_i Z_{i+1} (if both)
        
        # For PBC on 3 sites:
        # All sites have left and right neighbors.
        # So 4 terms per site * 3 sites = 12 terms.
        model = SU2GaugeModel(num_sites=3, pbc=True)
        pool = model.build_operator_pool()
        assert len(pool) == 12
        
        # For OBC on 3 sites (0, 1, 2):
        # Site 0:
        # - Left: None. Right: 1.
        # - Y_0 (yes)
        # - Y_0 Z_1 (yes)
        # - Z_{-1} Y_0 (no)
        # - Z_{-1} Y_0 Z_1 (no)
        # -> 2 terms
        
        # Site 1:
        # - Left: 0. Right: 2.
        # - All 4 valid.
        
        # Site 2:
        # - Left: 1. Right: None.
        # - Y_2 (yes)
        # - Y_2 Z_3 (no)
        # - Z_1 Y_2 (yes)
        # - Z_1 Y_2 Z_3 (no)
        # -> 2 terms
        
        # Total: 2 + 4 + 2 = 8 terms.
        model_obc = SU2GaugeModel(num_sites=3, pbc=False)
        pool_obc = model_obc.build_operator_pool()
        assert len(pool_obc) == 8
        
    def test_trotter_layers(self):
        model = SU2GaugeModel(num_sites=5, pbc=True)
        layers = model.get_trotter_layers()
        # Should have [H_E, H_M_Even, H_M_Odd]
        assert len(layers) == 3 or len(layers) == 2 # If empty
        for layer in layers:
            assert is_hermitian(layer)
