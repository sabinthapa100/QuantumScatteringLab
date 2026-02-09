"""
Criticality and Finite-Size Scaling Analysis
=============================================

Extract critical exponents, central charge, and scaling behavior.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.linalg import eigh
from qiskit.quantum_info import SparsePauliOp
from src.models.base import PhysicsModel


@dataclass
class CriticalityData:
    """Data for criticality analysis."""
    system_sizes: List[int]
    gaps: List[float]
    energies: List[float]
    correlations: List[float]
    entanglement_entropies: Optional[List[float]] = None


class CriticalityAnalyzer:
    """
    Analyze critical behavior and extract CFT data.
    
    Key Concepts:
    - Finite-size scaling: gap ~ 1/N at criticality
    - Entanglement entropy: S ~ (c/3) log(N) for CFT
    - Correlation length: ξ ~ N at criticality
    """
    
    @staticmethod
    def compute_gap_scaling(model_class: type, 
                           system_sizes: List[int],
                           **model_params) -> CriticalityData:
        """
        Compute energy gap for different system sizes.
        
        Args:
            model_class: PhysicsModel subclass
            system_sizes: List of N values
            **model_params: Fixed model parameters (e.g., g_x=1.0)
        
        Returns:
            CriticalityData with gaps, energies, etc.
        """
        gaps = []
        energies = []
        correlations = []
        
        for N in system_sizes:
            model = model_class(num_sites=N, **model_params)
            H = model.build_hamiltonian()
            mat = H.to_matrix()
            eigs = np.linalg.eigvalsh(mat)
            
            E0 = eigs[0]
            gap = eigs[1] - eigs[0]
            
            # Compute correlation at half-chain
            if N >= 4:
                vecs = eigh(mat)[1]
                psi0 = vecs[:, 0]
                r = N // 2
                op_str = ['I'] * N
                op_str[0], op_str[r] = 'Z', 'Z'
                corr_op = SparsePauliOp.from_list([("".join(reversed(op_str)), 1.0)])
                corr = np.real(psi0.conj() @ corr_op.to_matrix() @ psi0)
            else:
                corr = 0.0
            
            gaps.append(gap)
            energies.append(E0 / N)  # Energy per site
            correlations.append(corr)
            
        return CriticalityData(
            system_sizes=system_sizes,
            gaps=gaps,
            energies=energies,
            correlations=correlations
        )
    
    @staticmethod
    def fit_gap_scaling(data: CriticalityData) -> Dict[str, float]:
        """
        Fit gap ~ A / N^α to extract scaling exponent.
        
        At criticality: α = 1 (CFT prediction)
        Off-critical: α < 1 (exponential gap)
        
        Returns:
            {'A': prefactor, 'alpha': exponent, 'r_squared': fit quality}
        """
        N = np.array(data.system_sizes)
        gaps = np.array(data.gaps)
        
        # Log-log fit: log(gap) = log(A) - α log(N)
        log_N = np.log(N)
        log_gap = np.log(gaps)
        
        # Linear fit
        coeffs = np.polyfit(log_N, log_gap, 1)
        alpha = -coeffs[0]
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # R-squared
        fit_vals = log_A - alpha * log_N
        ss_res = np.sum((log_gap - fit_vals)**2)
        ss_tot = np.sum((log_gap - np.mean(log_gap))**2)
        r_squared = 1 - ss_res / ss_tot
        
        return {
            'A': A,
            'alpha': alpha,
            'r_squared': r_squared
        }
    
    @staticmethod
    def compute_entanglement_entropy(model: PhysicsModel, 
                                    subsystem_size: int) -> float:
        """
        Compute entanglement entropy S_A for subsystem A.
        
        For CFT: S_A = (c/3) log(N sin(πL_A/N)) + const
        where c is the central charge.
        
        Args:
            model: PhysicsModel instance
            subsystem_size: Size of subsystem A
        
        Returns:
            Entanglement entropy (von Neumann)
        """
        H = model.build_hamiltonian()
        mat = H.to_matrix()
        eigs, vecs = eigh(mat)
        psi0 = vecs[:, 0]
        
        # Reshape to matrix for partial trace
        N = model.num_sites
        L_A = subsystem_size
        L_B = N - L_A
        
        # Reshape state: |ψ⟩ → ψ[i_A, i_B]
        psi_matrix = psi0.reshape(2**L_A, 2**L_B)
        
        # Reduced density matrix: ρ_A = Tr_B(|ψ⟩⟨ψ|)
        rho_A = psi_matrix @ psi_matrix.conj().T
        
        # Eigenvalues of ρ_A
        eigs_rho = np.linalg.eigvalsh(rho_A)
        eigs_rho = eigs_rho[eigs_rho > 1e-12]  # Remove numerical zeros
        
        # von Neumann entropy: S = -Tr(ρ log ρ)
        S = -np.sum(eigs_rho * np.log(eigs_rho))
        
        return S
    
    @staticmethod
    def extract_central_charge(model_class: type,
                               system_sizes: List[int],
                               **model_params) -> Dict[str, float]:
        """
        Extract CFT central charge from entanglement scaling.
        
        S_A = (c/3) log(N sin(πL_A/N)) + const
        
        For L_A = N/2: S ~ (c/3) log(N)
        
        Returns:
            {'c': central_charge, 'const': additive_constant, 'r_squared': fit}
        """
        entropies = []
        
        for N in system_sizes:
            model = model_class(num_sites=N, **model_params)
            L_A = N // 2
            S = CriticalityAnalyzer.compute_entanglement_entropy(model, L_A)
            entropies.append(S)
        
        # Fit S = (c/3) log(N) + const
        N = np.array(system_sizes)
        S = np.array(entropies)
        log_N = np.log(N)
        
        coeffs = np.polyfit(log_N, S, 1)
        c_over_3 = coeffs[0]
        c = 3 * c_over_3
        const = coeffs[1]
        
        # R-squared
        fit_vals = c_over_3 * log_N + const
        ss_res = np.sum((S - fit_vals)**2)
        ss_tot = np.sum((S - np.mean(S))**2)
        r_squared = 1 - ss_res / ss_tot
        
        return {
            'c': c,
            'const': const,
            'r_squared': r_squared,
            'entropies': entropies
        }
    
    @staticmethod
    def compute_correlation_length(model: PhysicsModel, 
                                   max_distance: Optional[int] = None) -> float:
        """
        Compute correlation length from exponential decay.
        
        C(r) = ⟨Z_0 Z_r⟩ ~ exp(-r/ξ)
        
        Returns:
            Correlation length ξ
        """
        N = model.num_sites
        if max_distance is None:
            max_distance = N // 2
        
        H = model.build_hamiltonian()
        mat = H.to_matrix()
        eigs, vecs = eigh(mat)
        psi0 = vecs[:, 0]
        
        # Compute C(r) for r = 1, 2, ..., max_distance
        distances = []
        correlations = []
        
        for r in range(1, min(max_distance + 1, N)):
            op_str = ['I'] * N
            op_str[0], op_str[r] = 'Z', 'Z'
            corr_op = SparsePauliOp.from_list([("".join(reversed(op_str)), 1.0)])
            C_r = np.real(psi0.conj() @ corr_op.to_matrix() @ psi0)
            
            distances.append(r)
            correlations.append(abs(C_r))
        
        # Fit log(C) = log(C0) - r/ξ
        r = np.array(distances)
        log_C = np.log(np.array(correlations) + 1e-12)
        
        # Linear fit
        coeffs = np.polyfit(r, log_C, 1)
        xi = -1.0 / coeffs[0] if coeffs[0] < 0 else np.inf
        
        return xi


def plot_gap_scaling(data: CriticalityData, 
                    fit_params: Dict[str, float],
                    save_path: Optional[str] = None):
    """Plot gap vs system size with power-law fit."""
    import matplotlib.pyplot as plt
    
    N = np.array(data.system_sizes)
    gaps = np.array(data.gaps)
    
    # Fit curve
    A, alpha = fit_params['A'], fit_params['alpha']
    N_fit = np.linspace(N[0], N[-1], 100)
    gap_fit = A / N_fit**alpha
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Data points
    ax.loglog(N, gaps, 'o', markersize=10, label='Data')
    
    # Fit
    ax.loglog(N_fit, gap_fit, '--', linewidth=2, 
             label=f'Fit: Δ = {A:.3f} / N^{alpha:.3f}\n(R² = {fit_params["r_squared"]:.4f})')
    
    # Reference lines
    ax.loglog(N_fit, A / N_fit, ':', alpha=0.5, label='1/N (CFT)')
    
    ax.set_xlabel('System Size N', fontsize=14)
    ax.set_ylabel('Energy Gap Δ', fontsize=14)
    ax.set_title('Finite-Size Scaling at Criticality', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved gap scaling plot to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_entanglement_scaling(system_sizes: List[int],
                              entropies: List[float],
                              central_charge: float,
                              save_path: Optional[str] = None):
    """Plot entanglement entropy vs log(N)."""
    import matplotlib.pyplot as plt
    
    N = np.array(system_sizes)
    S = np.array(entropies)
    log_N = np.log(N)
    
    # Fit line
    c_over_3 = central_charge / 3
    const = S[0] - c_over_3 * log_N[0]  # Approximate
    S_fit = c_over_3 * log_N + const
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(log_N, S, 'o', markersize=10, label='Data')
    ax.plot(log_N, S_fit, '--', linewidth=2, 
           label=f'CFT: S = (c/3) log(N) + const\nc = {central_charge:.3f}')
    
    ax.set_xlabel('log(N)', fontsize=14)
    ax.set_ylabel('Entanglement Entropy S', fontsize=14)
    ax.set_title('Entanglement Scaling (CFT)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved entanglement scaling plot to {save_path}")
    else:
        plt.show()
    
    return fig
