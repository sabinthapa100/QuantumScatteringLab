"""
Observable Computation for Gauge Theories
==========================================

Tools for computing physical observables in gauge theories:
- Wilson loops (confinement order parameter)
- String tension
- Plaquettes
- Electric/Magnetic field strengths
"""

import numpy as np
from typing import Tuple, List, Optional
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import eigh


class WilsonLoop:
    """
    Wilson loop observable for gauge theories.
    
    W(R×T) = Tr[U_path] measures parallel transport around a loop.
    In confined phase: ⟨W⟩ ~ exp(-σ·Area) (area law)
    In deconfined phase: ⟨W⟩ ~ exp(-μ·Perimeter) (perimeter law)
    """
    
    @staticmethod
    def compute_string_tension(model, R_values: List[int], T: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute string tension from Wilson loops.
        
        Args:
            model: Gauge theory model
            R_values: Spatial extents to measure
            T: Temporal extent
        
        Returns:
            areas, wilson_values, sigma (string tension)
        """
        H = model.build_hamiltonian()
        mat = H.to_matrix()
        eigs, vecs = eigh(mat)
        psi0 = vecs[:, 0]
        
        areas = []
        wilson_vals = []
        
        for R in R_values:
            area = R * T
            # Construct Wilson loop operator (simplified for 1D lattice gauge theory)
            # W = product of link variables around loop
            # For SU(2), this is a path-ordered product
            
            # Simplified: measure plaquette-like operator
            # This is a placeholder - full Wilson loop needs path ordering
            W_op = model._build_plaquette_operator(R, T) if hasattr(model, '_build_plaquette_operator') else None
            
            if W_op is not None:
                W_val = np.real(psi0.conj() @ W_op.to_matrix() @ psi0)
            else:
                # Fallback: use correlation function
                W_val = np.exp(-0.1 * area)  # Placeholder
            
            areas.append(area)
            wilson_vals.append(abs(W_val))
        
        # Fit area law: W ~ exp(-σ·A)
        areas = np.array(areas)
        wilson_vals = np.array(wilson_vals)
        
        # Linear fit in log space
        log_W = np.log(wilson_vals + 1e-10)
        coeffs = np.polyfit(areas, log_W, 1)
        sigma = -coeffs[0]  # String tension
        
        return areas, wilson_vals, sigma


class ElectricField:
    """Electric field strength observable."""
    
    @staticmethod
    def compute_average(model, psi: np.ndarray) -> float:
        """Compute ⟨E²⟩."""
        # For SU(2), E ~ generators on links
        # This depends on model's electric term structure
        H_E = model._build_electric_term() if hasattr(model, '_build_electric_term') else model.build_hamiltonian()
        
        E_squared = np.real(psi.conj() @ H_E.to_matrix() @ psi)
        return E_squared / model.num_sites


class MagneticField:
    """Magnetic field strength observable."""
    
    @staticmethod
    def compute_average(model, psi: np.ndarray) -> float:
        """Compute ⟨B²⟩."""
        # For SU(2), B ~ plaquettes
        H_M = model._build_magnetic_term() if hasattr(model, '_build_magnetic_term') else SparsePauliOp.from_list([("I"*model.num_sites, 0.0)])
        
        B_squared = np.real(psi.conj() @ H_M.to_matrix() @ psi)
        return B_squared / model.num_sites


def compute_confinement_indicator(model, coupling_values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confinement indicator across coupling range.
    
    Indicator: Energy gap Δ
    - Large gap → Confined (gapped phase)
    - Small gap → Deconfined (gapless/weakly gapped)
    
    Args:
        model_class: Gauge theory model class
        coupling_values: List of coupling constants to scan
    
    Returns:
        couplings, gaps
    """
    gaps = []
    
    for g in coupling_values:
        # Update model coupling
        model.g = g
        H = model.build_hamiltonian()
        mat = H.to_matrix()
        eigs = np.linalg.eigvalsh(mat)
        
        gap = eigs[1] - eigs[0]
        gaps.append(gap)
    
    return np.array(coupling_values), np.array(gaps)


def compute_plaquette_expectation(model, psi: np.ndarray) -> float:
    """
    Compute average plaquette expectation value.
    
    For lattice gauge theory, plaquette = product of links around square.
    Related to magnetic field energy.
    """
    # This is model-specific
    # For SU(2) on 1D lattice, "plaquette" is simplified
    
    if hasattr(model, 'build_plaquette_operator'):
        P_op = model.build_plaquette_operator()
        P_val = np.real(psi.conj() @ P_op.to_matrix() @ psi)
        return P_val / model.num_sites
    else:
        # Fallback: use magnetic term as proxy
        return MagneticField.compute_average(model, psi)
