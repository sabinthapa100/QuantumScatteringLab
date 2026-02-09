"""
Integration tests for backends.
Compares Qiskit (Exact) and Quimb (MPS) results.
"""

import pytest
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from src.backends.qiskit_backend import QiskitBackend
from src.backends.quimb_backend import QuimbBackend
from src.models.ising_1d import IsingModel1D


class TestBackendComparison:
    
    @pytest.fixture
    def model(self):
        # Use 4 sites for quick exact comparison
        return IsingModel1D(num_sites=4, g_x=1.0, pbc=True)
        
    @pytest.fixture
    def qiskit_backend(self):
        return QiskitBackend()
        
    @pytest.fixture
    def quimb_backend(self):
        return QuimbBackend()

    def test_reference_state(self, model, qiskit_backend, quimb_backend):
        """Test that reference states produce same expectation values."""
        H = model.build_hamiltonian()
        
        state_qiskit = qiskit_backend.get_reference_state(model.num_sites)
        state_quimb = quimb_backend.get_reference_state(model.num_sites)
        
        val_qiskit = qiskit_backend.compute_expectation_value(state_qiskit, H)
        val_quimb = quimb_backend.compute_expectation_value(state_quimb, H)
        
        assert np.isclose(val_qiskit, val_quimb)

    def test_expectation_values(self, model, qiskit_backend, quimb_backend):
        """Test expectation values of different Paulis."""
        # Create a non-trivial state (e.g. apply some gates)
        H = model.build_hamiltonian()
        pool = model.build_operator_pool()
        
        # Apply Y gate to both
        s_q = qiskit_backend.get_reference_state(model.num_sites)
        s_m = quimb_backend.get_reference_state(model.num_sites)
        
        # op is Y_0
        op = pool[0] 
        s_q = qiskit_backend.apply_operator(s_q, op, parameter=0.5)
        s_m = quimb_backend.apply_operator(s_m, op, parameter=0.5)
        
        v_q = qiskit_backend.compute_expectation_value(s_q, H)
        v_m = quimb_backend.compute_expectation_value(s_m, H)
        
        assert np.isclose(v_q, v_m)

    def test_multi_site_pauli_expectation(self, model, qiskit_backend, quimb_backend):
        """Test expectation values for multi-site Paulis (ZZ terms)."""
        # ZZ on site 0, 1
        op = SparsePauliOp.from_list([("ZZII", 1.0)]) # Qiskit order
        # Wait, ZZII means Z on q3, Z on q2 (if little-endian 3210).
        # In our backends we handle reversing. 
        # Z_{0} Z_{1} in my models corresponds to label "IIZZ" reversed -> "ZZII" labels?
        # Let's just use what model produces.
        
        s_q = qiskit_backend.get_reference_state(model.num_sites)
        s_m = quimb_backend.get_reference_state(model.num_sites)
        
        # Apply H to make it non-Z-eigenstate
        from qiskit.circuit.library import HGate
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(4)
        qc.h(0)
        s_q = s_q.evolve(qc)
        
        # For MPS, apply H gate manually
        h_mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        s_m.gate_(h_mat, 0)
        
        v_q = qiskit_backend.compute_expectation_value(s_q, op)
        v_m = quimb_backend.compute_expectation_value(s_m, op)
        
        assert np.isclose(v_q, v_m)

    def test_trotter_evolution(self, model, qiskit_backend, quimb_backend):
        """Test that time evolution results match."""
        layers = model.get_trotter_layers()
        dt = 0.1
        
        s_q = qiskit_backend.get_reference_state(model.num_sites)
        s_m = quimb_backend.get_reference_state(model.num_sites)
        
        # One trotter step
        s_q = qiskit_backend.evolve_state_trotter(s_q, layers, dt)
        s_m = quimb_backend.evolve_state_trotter(s_m, layers, dt)
        
        # Compare energies
        H = model.build_hamiltonian()
        v_q = qiskit_backend.compute_expectation_value(s_q, H)
        v_m = quimb_backend.compute_expectation_value(s_m, H)
        
        # Higher tolerance for Trotter + MPS truncation if N was large, 
        # but for N=4 it should be very close.
        assert np.isclose(v_q, v_m, atol=1e-8)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
