"""
Initial state preparation for scattering wavepackets.

Reference: arXiv:2401.08044 / Farrell et al. (2025)
"""

from typing import Any, Optional, List
import numpy as np
from qiskit.quantum_info import Statevector

def prepare_w_state(num_sites: int, sites: Optional[List[int]] = None) -> Statevector:
    """
    Construct a uniform W-state: |W> = 1/sqrt(d) * sum_{n in sites} |2^n>.
    """
    if sites is None:
        sites = list(range(num_sites))
    
    d = len(sites)
    vec = np.zeros(2**num_sites, dtype=complex)
    for s in sites:
        # |2^s> has bit s set. 
        # Finite bitstring: 00...010...0
        idx = 1 << s
        vec[idx] = 1.0 / np.sqrt(d)
        
    return Statevector(vec)

def prepare_wavepacket_state(
    num_sites: int, 
    x0: float, 
    k0: float, 
    sigma: float,
    backend_type: str = "qiskit"
) -> Any:
    """
    Prepare a Gaussian wavepacket (W-state generalization).
    |W(k0)> = sum_n c_n e^{i phi_n} |2^n>
    
    Args:
        num_sites: Total qubits.
        x0: Center of wavepacket (site index).
        k0: Momentum of wavepacket.
        sigma: Spatial width.
    """
    # Spatial amplitudes
    n_range = np.arange(num_sites)
    # Gaussian envelope centered at x0
    amplitudes = np.exp(-(n_range - x0)**2 / (4 * sigma**2))
    # Phase factors for momentum k0
    phases = np.exp(1j * k0 * n_range)
    
    # Combined coefficients
    coeffs = amplitudes * phases
    norm = np.linalg.norm(coeffs)
    coeffs /= norm
    
    if backend_type == "qiskit":
        vec = np.zeros(2**num_sites, dtype=complex)
        for n, c in enumerate(coeffs):
            idx = 1 << n
            vec[idx] = c
        return Statevector(vec)
    
    elif backend_type == "quimb":
        try:
            import quimb.tensor as qtn
            # Construct as sum of computational states (slow for large N but OK for initialization)
            # Better: construct MPS directly
            states = []
            for n, c in enumerate(coeffs):
                if abs(c) > 1e-10:
                    bitstr = ['0'] * num_sites
                    bitstr[n] = '1'
                    state = qtn.MPS_computational_state("".join(bitstr))
                    # Scale state
                    for t in state.tensors:
                        # This is a bit hacky to scale an MPS, usually we'd multiply the whole network
                        pass
                # A better way in Quimb to create sum of product states is more involved.
            
            # Simple fallback: convert from vector
            # But that defeats the purpose of MPS for large N.
            # Usually for scattering, N=20-30, so vector conversion is borderline.
            pass
            
            # For now, return the coefficients and norm for the user to use in Quimb operations
            return coeffs
        except ImportError:
            # Quimb not available, return coefficients
            return coeffs
    
    return coeffs
