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
    
    def __init__(self, max_bond_dim: int = 64, cutoff: float = 1e-10, verbose: bool = False, use_gpu: bool = False):
        self.max_bond_dim = max_bond_dim
        self.cutoff = cutoff
        self.verbose = verbose
        self.use_gpu = use_gpu
        self._xp = np
        
        if self.use_gpu:
            try:
                import cupy as cp
                self._xp = cp
            except ImportError:
                import sys
                print("⚠️ Warning: GPU acceleration requested but CuPy not found. Falling back to CPU.", file=sys.stderr)
                self.use_gpu = False

    def _ensure_backend(self, tn: qtn.TensorNetwork) -> qtn.TensorNetwork:
        """Ensure TensorNetwork tensors are on the correct device (CPU/GPU)."""
        if self.use_gpu:
            import cupy as cp
            for t in tn:
                if not hasattr(t.data, 'device'):
                    t.modify(data=cp.asarray(t.data))
        else:
            for t in tn:
                if hasattr(t.data, 'device'):
                    t.modify(data=t.data.get())
        return tn
        
    def get_reference_state(self, num_sites: int) -> qtn.MatrixProductState:
        # Default Z+ state |00...0>
        mps = qtn.MPS_computational_state("0" * num_sites)
        return self._ensure_backend(mps)
        
    def compute_expectation_value(self, state: qtn.MatrixProductState, operator: SparsePauliOp) -> float:
        """Compute <psi|O|psi> using MPS contraction."""
        self._ensure_backend(state)
        total_expec = 0.0
        
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
            
            if self.use_gpu:
                op_mat = self._xp.asarray(op_mat)

            # Use local_expectation_exact which computes <psi|O|psi>
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
            
        # Optimization: Modify in-place to avoid expensive copies if possible
        # But for correctness in some higher level loops, we copy here
        mps = state.copy()
        self._ensure_backend(mps)
        
        for label, coeff in operator.to_list():
            # exponent = i * parameter * coeff * P
            theta = parameter * coeff
            
            active_sites = []
            pauli_chars = []
            for i, char in enumerate(reversed(label)):
                if char != 'I':
                    active_sites.append(i)
                    pauli_chars.append(char)
            
            if not active_sites:
                continue
            
            if len(active_sites) == 1:
                u = qu.expm(1j * theta * qu.pauli(pauli_chars[0]))
                if self.use_gpu:
                    u = self._xp.asarray(u)
                mps.gate(u, active_sites[0], contract=True, inplace=True)
            else:
                p_mats = [qu.pauli(p) for p in pauli_chars]
                mat = p_mats[0]
                for p in p_mats[1:]:
                    mat = np.kron(mat, p)
                
                u = qu.expm(1j * theta * mat)
                if self.use_gpu:
                    u = self._xp.asarray(u)
                
                # Apply using split (standard for nearest neighbor ZZ)
                mps.gate(u, active_sites, contract='split', inplace=True)
            
        # DRAMATIC SPEEDUP: Compress ONCE per layer instead of per gate
        mps.compress(max_bond=self.max_bond_dim, cutoff=self.cutoff)
        return mps

    def evolve_state_trotter(self, state: qtn.MatrixProductState, hamiltonian_layers: List[SparsePauliOp], time_step: float) -> qtn.MatrixProductState:
        """Evolve state by one Trotter step."""
        current_state = state
        self._ensure_backend(current_state)
        for layer_op in hamiltonian_layers:
            current_state = self.apply_operator(current_state, layer_op, parameter=-time_step)
        return current_state

    def compute_overlap(self, state1: qtn.MatrixProductState, state2: qtn.MatrixProductState) -> float:
        """Compute the fidelity overlap |<psi1|psi2>|^2."""
        # Normalize and compute inner product
        inner = (state1.H @ state2)
        return abs(inner)**2

    def get_ground_state(self, model: Any) -> qtn.MatrixProductState:
        """
        Compute the ground state of the model using DMRG.
        """
        import quimb as qu
        num_sites = model.num_sites
        
        # Build Hamiltonian MPO
        builder = qtn.SpinHam1D(S=0.5)
        j_int = getattr(model, 'j_int', 1.0)
        builder += -j_int, 'Z', 'Z'
        builder += -model.g_x, 'X'
        if abs(model.g_z) > 1e-12:
            builder += -model.g_z, 'Z'
        
        H_mpo = builder.build_mpo(L=num_sites)
        
        # Hybrid Approach: DMRG solve on CPU for stability
        # Many GPU pathfinding issues in quimb's DMRG-solve blocks
        dmrg = qtn.DMRG2(H_mpo, bond_dims=[self.max_bond_dim])
        print(f"  Running DMRG (L={num_sites}, D={self.max_bond_dim}) [CPU Solver]...")
        dmrg.solve(verbosity=1, tol=1e-8)
        
        # Migrate result to execution device
        return self._ensure_backend(dmrg.state)

    def to_dense(self, state: qtn.MatrixProductState) -> np.ndarray:
        return state.to_dense()

    def entanglement_entropy(self, state: qtn.MatrixProductState, site_index: int) -> float:
        """
        Calculate von Neumann entanglement entropy at the bond after site_index.
        """
        # site_index is 0 to L-2.
        # Returns entropy between sites [0...site_index] and [site_index+1...L-1]
        return state.entropy(site_index + 1)
