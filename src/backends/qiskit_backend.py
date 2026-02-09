"""
Backend using Qiskit's Statevector simulator.
"""

from typing import Any, List, Dict, Optional
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import QuantumCircuit
from .base import QuantumBackend, BackendError


class QiskitBackend(QuantumBackend):
    """
    Backend using Qiskit's Statevector.
    
    Best for small systems (N < 20 qubits) where full statevector 
    simulation is feasible on a single CPU.
    """
    
    @property
    def name(self) -> str:
        return "Qiskit-Statevector"

    def verify_compatibility(self, num_sites: int) -> None:
        """Check if system size is reasonable for statevector simulation."""
        if num_sites > 25:
            # 2^25 states is ~512MB for complex128, but operations can be heavy
            raise BackendError(
                f"System size {num_sites} is too large for Qiskit-Statevector backend. "
                "Consider using QuimbBackend (MPS)."
            )

    def get_reference_state(self, num_sites: int) -> Statevector:
        """Returns the |00...0> reference state."""
        return Statevector.from_label('0' * num_sites)

    def compute_expectation_value(self, state: Statevector, operator: SparsePauliOp) -> float:
        """Compute <psi| O |psi>."""
        try:
            return float(np.real(state.expectation_value(operator)))
        except Exception as e:
            raise BackendError(f"Failed to compute expectation value: {str(e)}")

    def compute_observable_statistics(self, state: Statevector, observables: Dict[str, SparsePauliOp]) -> Dict[str, float]:
        """Compute multiple expectation values efficiently."""
        results = {}
        for name, op in observables.items():
            results[name] = self.compute_expectation_value(state, op)
        return results
    
    def get_statevector(self, state: Statevector) -> np.ndarray:
        """Extract numpy array from Statevector."""
        return state.data

    def apply_operator(self, state: Statevector, operator: SparsePauliOp, parameter: float = 1.0) -> Statevector:
        """
        Applies U(theta) = exp(i * theta * P).
        Note: Qiskit's PauliEvolutionGate(P, t) implements exp(-i * P * t).
        So we set t = -parameter.
        """
        try:
            # Construct a circuit for the evolution
            qc = QuantumCircuit(state.num_qubits)
            # Qiskit uses little-endian qubit ordering (q0 is rightmost)
            qc.append(PauliEvolutionGate(operator, time=-parameter), qc.qubits)
            return state.evolve(qc)
        except Exception as e:
            raise BackendError(f"Failed to apply operator: {str(e)}")

    def evolve_state_trotter(
        self, 
        state: Statevector, 
        hamiltonian_layers: List[SparsePauliOp], 
        time_step: float,
        steps: int = 1
    ) -> Statevector:
        """
        Time evolution using Trotter decomposition.
        |psi(T)> = [exp(-i H_n dt) ... exp(-i H_1 dt)]^steps |psi(0)>
        """
        try:
            qc = QuantumCircuit(state.num_qubits)
            for _ in range(steps):
                for layer in hamiltonian_layers:
                    qc.append(PauliEvolutionGate(layer, time=time_step), qc.qubits)
            
            return state.evolve(qc)
        except Exception as e:
            raise BackendError(f"Failed to evolve state: {str(e)}")
