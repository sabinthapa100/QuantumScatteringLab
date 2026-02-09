import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import scipy.optimize
from qiskit.quantum_info import SparsePauliOp

from ..models.base import PhysicsModel
from ..backends.base import QuantumBackend
from ..backends.qiskit_backend import QiskitBackend

class ADAPTVQESolver:
    """
    A robust, reusable ADAPT-VQE engine.
    
    This class manages the adaptive ansatz construction and VQE optimization loop.
    It is decoupled from the specific physics model AND the simulation backend.
    
    Convergence Criteria:
    Matches original implementation (Grimsley et al. / vqe_methods.py):
    - 'norm': Euclidean norm of gradient vector < tolerance.
    - 'max': Max absolute gradient < tolerance.
    """
    
    def __init__(self, model: PhysicsModel, backend: Optional[QuantumBackend] = None, 
                 tolerance: float = 1e-6, max_iters: int = 20, convergence_type: str = 'norm',
                 initial_state: Optional[Any] = None, pool_type: str = 'global'):
        """
        Initialize the solver.
        
        Args:
            model (PhysicsModel): The physics model.
            backend (QuantumBackend): The simulation backend.
            tolerance (float): Gradient norm tolerance.
            max_iters (int): Max ADAPT steps.
            convergence_type (str): 'norm' (Euclidean) or 'max' (Infinity).
            initial_state (Any): Optional starting state (e.g. W-state). Defaults to backend's reference state.
            pool_type (str): 'global' or 'local' pool from model.
        """
        self.model = model
        self.backend = backend if backend else QiskitBackend()
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.convergence_type = convergence_type
        self.initial_state = initial_state
        self.pool_type = pool_type
        
        # Build problem
        self.hamiltonian = model.build_hamiltonian()
        self.pool = model.build_operator_pool(pool_type=pool_type)
        
        # State
        self.ansatz_ops: List[SparsePauliOp] = []
        self.parameters: List[float] = []
        self.energy_history: List[float] = []
        self.gradient_history: List[float] = []
        
    def _prepare_ansatz_state(self, params: List[float]) -> Any:
        """Helper to prepare state |psi(theta)> = U(theta) |initial>."""
        if self.initial_state is not None:
            state = self.initial_state
        else:
            state = self.backend.get_reference_state(self.model.num_sites)
        
        # Apply ansatz operators in order
        for op, theta in zip(self.ansatz_ops, params):
            # U = exp(i * theta * P)
            state = self.backend.apply_operator(state, op, theta)
            
        return state

    def compute_gradients(self, params: List[float]) -> List[float]:
        """
        Computes gradient for each pool operator A_k = i P_k.
        grad = <psi| [H, A_k] |psi> = i <psi| [H, P_k] |psi>.
        """
        psi = self._prepare_ansatz_state(params)
        
        grads = []
        for op in self.pool:
            # We want expectation of commutator O = i[H, P]
            # Since backends usually expect SparsePauliOp, we construct it.
            # Qiskit SparsePauliOp supports composition.
            # commutator = H @ P - P @ H
            comm = self.hamiltonian.compose(op) - op.compose(self.hamiltonian)
            grad_op = 1j * comm
            
            val = self.backend.compute_expectation_value(psi, grad_op)
            grads.append(val)
            
        return grads
            
    def run(self) -> Dict:
        """
        Run ADAPT-VQE with comprehensive logging and tracking.
        
        Returns:
            Dictionary with results and iteration data
        """
        import time
        
        # Compute exact ground state if possible (for validation)
        exact_gs_energy = None
        exact_gs_state = None
        try:
            H_mat = self.hamiltonian.to_matrix()
            eigs, vecs = np.linalg.eigh(H_mat)
            exact_gs_energy = eigs[0]
            exact_gs_state = vecs[:, 0]
        except:
            pass  # Too large for exact diagonalization
        
        # Initialize tracking
        iteration_data = []
        current_params = []
        prev_energy = None
        last_selected_op = None
        
        # Print header
        print("="*100)
        print(f"ADAPT-VQE: N={self.model.num_sites}, Pool size={len(self.pool)}, "
              f"Tolerance={self.tolerance}, Max iters={self.max_iters}")
        print("="*100)
        
        if exact_gs_energy is not None:
            print(f"Exact ground state energy: {exact_gs_energy:.10f}")
        print()
        
        # Table header
        print(f"{'Iter':<6} {'Op':<6} {'||grad||':<12} {'max|g|':<12} "
              f"{'Energy':<16} {'ΔE':<12} {'#P':<5} {'Overlap':<10} {'Time(s)':<8}")
        print("-"*100)
        
        # Main ADAPT-VQE loop
        for iteration in range(self.max_iters):
            iter_start = time.time()
            
            # 1. Compute gradients for all pool operators
            grads = self.compute_gradients(current_params)
            
            # Compute gradient metrics
            grad_norm = np.linalg.norm(grads)
            grad_abs = np.abs(grads)
            max_grad_idx = np.argmax(grad_abs)
            max_grad = grad_abs[max_grad_idx]
            
            # Check for repeated operator selection (from archive code)
            if last_selected_op is not None and max_grad_idx == last_selected_op:
                # Select second-best to avoid repetition
                sorted_indices = np.argsort(grad_abs)[::-1]
                for idx in sorted_indices:
                    if idx != last_selected_op:
                        max_grad_idx = idx
                        max_grad = grad_abs[idx]
                        print(f"  (Avoiding repeat: selected 2nd best operator)")
                        break
            
            # Check convergence BEFORE adding operator
            converged = False
            if self.convergence_type == 'norm':
                converged = grad_norm < self.tolerance
            elif self.convergence_type == 'max':
                converged = max_grad < self.tolerance
            
            if converged:
                # Compute final state for overlap
                psi_final = self._prepare_ansatz_state(current_params)
                overlap_final = None
                if exact_gs_state is not None:
                    psi_vec = self.backend.get_statevector(psi_final) if hasattr(self.backend, 'get_statevector') else psi_final.data
                    overlap_final = np.abs(np.vdot(psi_vec, exact_gs_state))**2
                
                iter_time = time.time() - iter_start
                
                # Print final iteration
                energy_str = f"{prev_energy:.10f}" if prev_energy is not None else "N/A"
                overlap_str = f"{overlap_final:.6f}" if overlap_final is not None else "N/A"
                print(f"{iteration:<6} {'--':<6} {grad_norm:<12.6e} {max_grad:<12.6e} "
                      f"{energy_str:<16} {'--':<12} {len(current_params):<5} "
                      f"{overlap_str:<10} {iter_time:<8.2f}")
                
                print("-"*100)
                print(f"✓ CONVERGED: Gradient {'norm' if self.convergence_type == 'norm' else 'max'} "
                      f"= {grad_norm if self.convergence_type == 'norm' else max_grad:.2e} < {self.tolerance:.2e}")
                break
            
            # 2. Add selected operator to ansatz
            selected_op = self.pool[max_grad_idx]
            self.ansatz_ops.append(selected_op)
            current_params.append(0.0)  # Initialize new parameter
            last_selected_op = max_grad_idx
            
            # 3. Optimize all parameters
            def cost_fn(params):
                psi = self._prepare_ansatz_state(params)
                E = self.backend.compute_expectation_value(psi, self.hamiltonian)
                return np.real(E)
            
            # Use COBYLA (from archive) or BFGS
            opt_result = scipy.optimize.minimize(
                cost_fn, current_params, method='COBYLA',
                options={'maxiter': 1000, 'tol': 1e-8}
            )
            
            current_params = opt_result.x.tolist()
            current_energy = opt_result.fun
            
            # Compute energy change
            delta_E = current_energy - prev_energy if prev_energy is not None else 0.0
            prev_energy = current_energy
            
            # Compute overlap with exact GS (if available)
            overlap = None
            if exact_gs_state is not None:
                psi = self._prepare_ansatz_state(current_params)
                # Get statevector
                if hasattr(self.backend, 'get_statevector'):
                    psi_vec = self.backend.get_statevector(psi)
                elif hasattr(psi, 'data'):
                    psi_vec = psi.data
                else:
                    psi_vec = psi
                overlap = np.abs(np.vdot(psi_vec, exact_gs_state))**2
            
            iter_time = time.time() - iter_start
            
            # Store iteration data
            iter_info = {
                'iteration': iteration,
                'selected_op_idx': max_grad_idx,
                'grad_norm': grad_norm,
                'max_grad': max_grad,
                'energy': current_energy,
                'delta_E': delta_E,
                'num_params': len(current_params),
                'overlap': overlap,
                'time': iter_time,
                'converged': False
            }
            iteration_data.append(iter_info)
            
            # Update histories
            self.energy_history.append(current_energy)
            self.gradient_history.append(grad_norm)
            
            # Print iteration row
            overlap_str = f"{overlap:.6f}" if overlap is not None else "N/A"
            print(f"{iteration:<6} {max_grad_idx:<6} {grad_norm:<12.6e} {max_grad:<12.6e} "
                  f"{current_energy:<16.10f} {delta_E:<12.6e} {len(current_params):<5} "
                  f"{overlap_str:<10} {iter_time:<8.2f}")
            
            # Check energy convergence (additional criterion)
            if exact_gs_energy is not None:
                energy_error = abs(current_energy - exact_gs_energy)
                if energy_error < 1e-8:
                    print(f"\n✓ Energy converged to exact GS (error = {energy_error:.2e})")
                    break
        
        # Final summary
        print("="*100)
        print("ADAPT-VQE COMPLETE")
        print("="*100)
        
        final_energy = self.energy_history[-1] if self.energy_history else None
        
        if final_energy is not None:
            print(f"Final energy:        {final_energy:.10f}")
            if exact_gs_energy is not None:
                error = abs(final_energy - exact_gs_energy)
                rel_error = error / abs(exact_gs_energy) if exact_gs_energy != 0 else error
                print(f"Exact energy:        {exact_gs_energy:.10f}")
                print(f"Energy error:        {error:.2e}")
                print(f"Relative error:      {rel_error:.2e}")
        
        if overlap is not None:
            print(f"Final overlap:       {overlap:.6f} ({overlap*100:.2f}%)")
        
        print(f"Total iterations:    {len(self.energy_history)}")
        print(f"Operators used:      {len(self.ansatz_ops)}")
        print(f"Total parameters:    {len(current_params)}")
        print()
        
        # Store final parameters
        self.parameters = current_params
        
        return {
            'energy': final_energy,
            'exact_energy': exact_gs_energy,
            'energy_error': abs(final_energy - exact_gs_energy) if exact_gs_energy and final_energy else None,
            'overlap': overlap,
            'iterations': len(self.energy_history),
            'converged': converged if 'converged' in locals() else False,
            'parameters': current_params,
            'energy_history': self.energy_history,
            'gradient_history': self.gradient_history,
            'iteration_data': iteration_data,
            'selected_operators': [i['selected_op_idx'] for i in iteration_data],
            'ansatz_ops': self.ansatz_ops
        }
