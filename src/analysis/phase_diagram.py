"""
Phase Diagram Analysis Tools
=============================

Scan parameter space to identify phases and phase transitions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.linalg import eigh
from qiskit.quantum_info import SparsePauliOp
from src.models.base import PhysicsModel


@dataclass
class PhasePoint:
    """Single point in phase diagram."""
    params: Dict[str, float]  # e.g., {'g_x': 1.0, 'g_z': 0.0}
    energy: float
    gap: float
    order_parameters: Dict[str, float]  # e.g., {'<X>': 0.99, '<Z>': 0.01}
    degeneracy: int


class PhaseDiagramScanner:
    """
    Scan parameter space and compute observables.
    
    Example:
        scanner = PhaseDiagramScanner(IsingModel1D, num_sites=8)
        results = scanner.scan_2d('g_x', (0.1, 3.0, 20), 'g_z', (0.0, 1.0, 10))
        scanner.plot_phase_diagram(results, order_param='<X>')
    """
    
    def __init__(self, model_class: type, num_sites: int, **fixed_params):
        """
        Args:
            model_class: PhysicsModel subclass (e.g., IsingModel1D)
            num_sites: System size
            **fixed_params: Fixed model parameters (e.g., pbc=True)
        """
        self.model_class = model_class
        self.num_sites = num_sites
        self.fixed_params = fixed_params
        
    def compute_observables(self, model: PhysicsModel) -> Dict[str, float]:
        """Compute order parameters for a given model."""
        H = model.build_hamiltonian()
        mat = H.to_matrix()
        eigs, vecs = eigh(mat)
        
        # Ground state
        psi0 = vecs[:, 0]
        E0 = eigs[0]
        
        # Energy gap
        gap = eigs[1] - eigs[0]
        
        # Degeneracy (count states within 1e-6 of E0)
        degeneracy = np.sum(np.abs(eigs - E0) < 1e-6)
        
        # Order parameters
        order_params = {}
        
        # Magnetization <X>, <Z>
        for pauli in ['X', 'Z']:
            mag = 0.0
            for i in range(model.num_sites):
                op_str = ['I'] * model.num_sites
                op_str[i] = pauli
                op = SparsePauliOp.from_list([("".join(reversed(op_str)), 1.0)])
                mag += np.real(psi0.conj() @ op.to_matrix() @ psi0)
            order_params[f'<{pauli}>'] = mag / model.num_sites
        
        # Two-point correlations <Z_0 Z_r>
        if model.num_sites >= 2:
            r = model.num_sites // 2  # Half-chain distance
            op_str = ['I'] * model.num_sites
            op_str[0], op_str[r] = 'Z', 'Z'
            corr_op = SparsePauliOp.from_list([("".join(reversed(op_str)), 1.0)])
            order_params['<Z_0 Z_r>'] = np.real(psi0.conj() @ corr_op.to_matrix() @ psi0)
        
        return {
            'energy': E0,
            'gap': gap,
            'degeneracy': int(degeneracy),
            **order_params
        }
    
    def scan_1d(self, param_name: str, param_range: Tuple[float, float, int]) -> List[PhasePoint]:
        """
        Scan 1D parameter space.
        
        Args:
            param_name: Parameter to vary (e.g., 'g_x')
            param_range: (min, max, num_points)
        
        Returns:
            List of PhasePoint objects
        """
        min_val, max_val, num_points = param_range
        param_values = np.linspace(min_val, max_val, num_points)
        
        results = []
        for val in param_values:
            # Create model with this parameter value
            params = {**self.fixed_params, param_name: val}
            model = self.model_class(num_sites=self.num_sites, **params)
            
            # Compute observables
            obs = self.compute_observables(model)
            
            # Store result
            point = PhasePoint(
                params={param_name: val},
                energy=obs['energy'],
                gap=obs['gap'],
                order_parameters={k: v for k, v in obs.items() if k.startswith('<')},
                degeneracy=obs['degeneracy']
            )
            results.append(point)
            
        return results
    
    def scan_2d(self, 
                param1: str, range1: Tuple[float, float, int],
                param2: str, range2: Tuple[float, float, int]) -> np.ndarray:
        """
        Scan 2D parameter space.
        
        Returns:
            Array of PhasePoint objects with shape (n1, n2)
        """
        min1, max1, n1 = range1
        min2, max2, n2 = range2
        
        vals1 = np.linspace(min1, max1, n1)
        vals2 = np.linspace(min2, max2, n2)
        
        results = np.empty((n1, n2), dtype=object)
        
        for i, v1 in enumerate(vals1):
            for j, v2 in enumerate(vals2):
                params = {**self.fixed_params, param1: v1, param2: v2}
                model = self.model_class(num_sites=self.num_sites, **params)
                obs = self.compute_observables(model)
                
                results[i, j] = PhasePoint(
                    params={param1: v1, param2: v2},
                    energy=obs['energy'],
                    gap=obs['gap'],
                    order_parameters={k: v for k, v in obs.items() if k.startswith('<')},
                    degeneracy=obs['degeneracy']
                )
                
        return results
    
    def find_critical_points(self, results: List[PhasePoint], 
                            order_param: str = '<X>',
                            threshold: float = 0.5) -> List[int]:
        """
        Find critical points where order parameter crosses threshold.
        
        Returns:
            Indices of critical points
        """
        critical_indices = []
        
        for i in range(len(results) - 1):
            val_i = results[i].order_parameters.get(order_param, 0)
            val_ip1 = results[i+1].order_parameters.get(order_param, 0)
            
            # Check if crosses threshold
            if (val_i - threshold) * (val_ip1 - threshold) < 0:
                critical_indices.append(i)
                
        return critical_indices


def plot_phase_diagram_1d(results: List[PhasePoint], 
                          param_name: str,
                          order_param: str = '<X>',
                          save_path: Optional[str] = None):
    """Plot 1D phase diagram."""
    import matplotlib.pyplot as plt
    
    param_vals = [p.params[param_name] for p in results]
    order_vals = [p.order_parameters.get(order_param, 0) for p in results]
    gaps = [p.gap for p in results]
    energies = [p.energy for p in results]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Order parameter
    axes[0].plot(param_vals, order_vals, 'o-', linewidth=2, markersize=6)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel(order_param, fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Phase Diagram (N={results[0].params.get("num_sites", "?")})', fontsize=16)
    
    # Energy gap
    axes[1].plot(param_vals, gaps, 'o-', color='red', linewidth=2, markersize=6)
    axes[1].set_ylabel('Energy Gap', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Ground state energy
    axes[2].plot(param_vals, energies, 'o-', color='green', linewidth=2, markersize=6)
    axes[2].set_ylabel('Ground State Energy', fontsize=14)
    axes[2].set_xlabel(param_name, fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved phase diagram to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_phase_diagram_2d(results: np.ndarray,
                          param1: str, param2: str,
                          order_param: str = '<X>',
                          save_path: Optional[str] = None):
    """Plot 2D phase diagram as heatmap."""
    import matplotlib.pyplot as plt
    
    n1, n2 = results.shape
    
    # Extract grid values
    vals1 = np.array([results[i, 0].params[param1] for i in range(n1)])
    vals2 = np.array([results[0, j].params[param2] for j in range(n2)])
    
    # Extract order parameter
    order_grid = np.array([[results[i, j].order_parameters.get(order_param, 0) 
                           for j in range(n2)] for i in range(n1)])
    
    gap_grid = np.array([[results[i, j].gap for j in range(n2)] for i in range(n1)])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Order parameter heatmap
    im1 = axes[0].imshow(order_grid, origin='lower', aspect='auto', cmap='RdBu_r',
                        extent=[vals2[0], vals2[-1], vals1[0], vals1[-1]])
    axes[0].set_xlabel(param2, fontsize=14)
    axes[0].set_ylabel(param1, fontsize=14)
    axes[0].set_title(f'{order_param}', fontsize=16)
    plt.colorbar(im1, ax=axes[0])
    
    # Gap heatmap (log scale)
    im2 = axes[1].imshow(np.log10(gap_grid + 1e-10), origin='lower', aspect='auto', cmap='viridis',
                        extent=[vals2[0], vals2[-1], vals1[0], vals1[-1]])
    axes[1].set_xlabel(param2, fontsize=14)
    axes[1].set_ylabel(param1, fontsize=14)
    axes[1].set_title('log₁₀(Gap)', fontsize=16)
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D phase diagram to {save_path}")
    else:
        plt.show()
    
    return fig
