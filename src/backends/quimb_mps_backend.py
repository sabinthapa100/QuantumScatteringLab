"""
MPS Backend using Quimb.
Internal implementation for large scale 1D simulation.
"""

import numpy as np
from typing import Any, List, Dict, Optional, Union
from qiskit.quantum_info import SparsePauliOp
import quimb as qu
import quimb.tensor as qtn
import logging
from .base import QuantumBackend, BackendError

logger = logging.getLogger(__name__)

class QuimbMPSBackend(QuantumBackend):
    """
    Scalable MPS simulation backend using Quimb.
    Ideal for N > 20 in 1D systems.
    """
    
    def __init__(self, max_bond_dim: int = 64, cutoff: float = 1e-10, verbose: bool = False):
        self.max_bond_dim = max_bond_dim
        self.cutoff = cutoff
        self.verbose = verbose
        
    def get_reference_state(self, num_sites: int) -> qtn.MatrixProductState:
        # Default Z+ state |00...0>
        return qtn.MPS_computational_state("0" * num_sites)
        
    def compute_expectation_value(self, state: qtn.MatrixProductState, operator: SparsePauliOp) -> float:
        """Compute <psi|O|psi> using MPS contraction."""
        total_expec = 0.0
        
        # Ensure state is normalized? MPS usually is.
        # But for expectation values, we should check?
        # quimb methods usually handle it if we use local_expectation.
        
        for label, coeff in operator.to_list():
            active_sites = []
            pauli_gates = []
            for i, char in enumerate(reversed(label)):
                if char != 'I':
                    active_sites.append(i)
                    pauli_gates.append(qu.pauli(char))
            
            if not active_sites:
                total_expec += coeff.real
                continue
            
            if len(active_sites) == 1:
                op_mat = pauli_gates[0]
            else:
                op_mat = pauli_gates[0]
                for p in pauli_gates[1:]:
                    op_mat = np.kron(op_mat, p)
            
            # Use local_expectation_exact which computes <psi|O|psi>
            # Note: exact contraction for many sites scaling is exp(k).
            # But for k=1 sum Y_i or k=2 sum Z Z, it is efficient.
            val = state.local_expectation_exact(op_mat, active_sites)
            total_expec += coeff.real * val.real
            
        return total_expec

    def apply_operator(self, state: qtn.MatrixProductState, operator: SparsePauliOp, parameter: float = 1.0) -> qtn.MatrixProductState:
        """
        Apply U = exp(i * parameter * operator) to MPS.
        Assumes terms in operator commute (standard for Trotter layers).
        """
        if abs(parameter) < 1e-15:
            return state
            
        mps = state.copy()
        
        # Iterate over terms and apply gate exp(i * theta * term)
        for label, coeff in operator.to_list():
            theta = (parameter * coeff).real
            
            active_sites = []
            pauli_chars = []
            for i, char in enumerate(reversed(label)):
                if char != 'I':
                    active_sites.append(i)
                    pauli_chars.append(char)
            
            if not active_sites:
                # Global phase factor, can ignore for MPS usually unless tracking phase?
                # MPS structure doesn't easily store global phase factor without aux tensor.
                continue
            
            if len(active_sites) == 1:
                # Single site rotation
                u = qu.expm(1j * theta * qu.pauli(pauli_chars[0]))
                # Gate on single site
                mps.gate(u, active_sites[0], contract=True, inplace=True)
                
            else:
                # Multi-site gate (e.g. ZZ)
                # Construct U = exp(i theta P1 tensor ... Pk)
                # For ZZ: [Z, Z]
                p_mats = [qu.pauli(p) for p in pauli_chars]
                mat = p_mats[0]
                for p in p_mats[1:]:
                    mat = np.kron(mat, p)
                
                u = qu.expm(1j * theta * mat)
                
                # Apply using swap+split (standard for nearest neighbor)
                # If non-nearest neighbor, this might be expensive (swap gates inserted)
                # But for 1D physics models (ZZ), it's NN.
                mps.gate(u, active_sites, contract='swap+split', inplace=True)
            
            # Compress after each gate or after layer?
            # Compressing after each gate keeps bond dimension from exploding during the sequence
            mps.compress(max_bond=self.max_bond_dim, cutoff=self.cutoff)
            
        return mps

    def evolve_state_trotter(self, state: qtn.MatrixProductState, hamiltonian_layers: List[SparsePauliOp], time_step: float) -> qtn.MatrixProductState:
        """Evolve state by one Trotter step."""
        current_state = state
        for layer_op in hamiltonian_layers:
            current_state = self.apply_operator(current_state, layer_op, parameter=-time_step)
        return current_state

    def to_dense(self, state: qtn.MatrixProductState) -> np.ndarray:
        return state.to_dense()
