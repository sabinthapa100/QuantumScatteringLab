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
    
    # Default to numpy array
    vec = np.zeros(2**num_sites, dtype=complex)
    for n, c in enumerate(coeffs):
        idx = 1 << n
        vec[idx] = c
    return vec

def prepare_two_wavepacket_state(
    num_sites: int,
    x1: float, k1: float, sigma1: float,
    x2: float, k2: float, sigma2: float,
    backend_type: str = "qiskit"
) -> Any:
    """
    Prepare a state with TWO wavepackets in the 2-excitation sector.
    |W1, W2> = sum_{n<m} (c1_n c2_m + c1_m c2_n) |2^n + 2^m>
    (assuming they represent additive excitations like spin-flips)
    """
    n_range = np.arange(num_sites)
    c1 = np.exp(-(n_range - x1)**2 / (4 * sigma1**2)) * np.exp(1j * k1 * n_range)
    c2 = np.exp(-(n_range - x2)**2 / (4 * sigma2**2)) * np.exp(1j * k2 * n_range)
    
    vec = np.zeros(2**num_sites, dtype=complex)
    for n in range(num_sites):
        for m in range(n + 1, num_sites):
            # Normalization might be slightly off if they overlap, but usually they don't.
            coeff = c1[n] * c2[m] + c1[m] * c2[n]
            idx = (1 << n) | (1 << m)
            vec[idx] = coeff
            
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
        
    if backend_type == "qiskit":
        return Statevector(vec)
    return vec
def prepare_2d_wavepacket_state(
    Lx: int, Ly: int,
    x0: float, y0: float,
    kx: float, ky: float,
    sigma: float,
    backend_type: str = "numpy"
) -> Any:
    """
    Gaussian wavepacket in 2D.
    """
    num_sites = Lx * Ly
    vec = np.zeros(2**num_sites, dtype=complex)
    
    for ny in range(Ly):
        for nx in range(Lx):
            idx = ny * Lx + nx
            # Amplitude
            amp = np.exp(-((nx - x0)**2 + (ny - y0)**2) / (4 * sigma**2))
            # Phase
            phase = np.exp(1j * (kx * nx + ky * ny))
            vec[1 << idx] = amp * phase
            
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
        
    if backend_type == "qiskit":
        return Statevector(vec)
    return vec

def prepare_two_wavepacket_state_2d(
    Lx: int, Ly: int,
    x1: float, y1: float, kx1: float, ky1: float, sigma1: float,
    x2: float, y2: float, kx2: float, ky2: float, sigma2: float,
    backend_type: str = "numpy"
) -> Any:
    """
    Two wavepackets in 2D in the 2-excitation sector.
    """
    num_sites = Lx * Ly
    vec = np.zeros(2**num_sites, dtype=complex)
    
    # Precompute c1, c2
    c1 = np.zeros(num_sites, dtype=complex)
    c2 = np.zeros(num_sites, dtype=complex)
    for ny in range(Ly):
        for nx in range(Lx):
            idx = ny * Lx + nx
            c1[idx] = np.exp(-((nx - x1)**2 + (ny - y1)**2) / (4 * sigma1**2)) * np.exp(1j * (kx1 * nx + ky1 * ny))
            c2[idx] = np.exp(-((nx - x2)**2 + (ny - y2)**2) / (4 * sigma2**2)) * np.exp(1j * (kx2 * nx + ky2 * ny))
            
    for i in range(num_sites):
        for j in range(i + 1, num_sites):
            coeff = c1[i] * c2[j] + c1[j] * c2[i]
            vec[(1 << i) | (1 << j)] = coeff
            
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
        
    if backend_type == "qiskit":
        return Statevector(vec)
    return vec
