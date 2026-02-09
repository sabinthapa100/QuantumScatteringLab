"""
Operator Pool Discovery Tool
=============================

Automated generation of operator pools via commutator expansion.

Theory:
-------
Given Hamiltonian H, generate pool by computing nested commutators:
    [H, O], [H, [H, O]], [H, [H, [H, O]]], ...

Keep unique, Hermitian operators up to desired order.

References:
- Grimsley et al. (2019) - ADAPT-VQE
- Farrell et al. (2025) - Global symmetry-preserving pools
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from qiskit.quantum_info import SparsePauliOp, Pauli
from dataclasses import dataclass
from src.models.base import PhysicsModel


@dataclass
class PoolOperator:
    """Single operator in the pool."""
    operator: SparsePauliOp
    order: int  # Commutator order (0=initial, 1=[H,O], 2=[[H,O],H], etc.)
    parent_idx: Optional[int] = None  # Index of parent operator
    
    def __hash__(self):
        # Hash based on Pauli string representation
        return hash(str(self.operator.paulis))
    
    def __eq__(self, other):
        if not isinstance(other, PoolOperator):
            return False
        # Two operators are equal if they have same Pauli structure
        return str(self.operator.paulis) == str(other.operator.paulis)


class OperatorPoolDiscovery:
    """
    Discover operator pools via commutator expansion.
    
    Example:
        model = IsingModel1D(num_sites=4)
        discovery = OperatorPoolDiscovery(model)
        pool = discovery.generate_pool(max_order=3, global_only=True)
    """
    
    def __init__(self, model: PhysicsModel, tolerance: float = 1e-10):
        """
        Args:
            model: PhysicsModel instance
            tolerance: Threshold for considering operators as zero
        """
        self.model = model
        self.tolerance = tolerance
        self.H = model.build_hamiltonian()
        
    def commutator(self, A: SparsePauliOp, B: SparsePauliOp) -> SparsePauliOp:
        """Compute [A, B] = AB - BA."""
        comm = (A @ B - B @ A).simplify()
        # Remove small coefficients
        if len(comm.paulis) > 0:
            mask = np.abs(comm.coeffs) > self.tolerance
            if np.any(mask):
                return SparsePauliOp(comm.paulis[mask], comm.coeffs[mask])
        return SparsePauliOp.from_list([("I" * self.model.num_sites, 0.0)])
    
    def is_hermitian(self, op: SparsePauliOp) -> bool:
        """Check if operator is Hermitian."""
        # For Pauli operators, Hermitian means real coefficients
        return np.allclose(op.coeffs.imag, 0, atol=self.tolerance)
    
    def is_global(self, op: SparsePauliOp) -> bool:
        """
        Check if operator is translationally invariant (global).
        
        A global operator is a sum over all sites with same local structure.
        """
        # This is a heuristic: check if operator acts on all sites equally
        # Count how many terms act on each site
        site_counts = [0] * self.model.num_sites
        
        for pauli_str in op.paulis:
            pauli_list = list(str(pauli_str))[::-1]  # Reverse for qubit ordering
            for i, p in enumerate(pauli_list):
                if p != 'I':
                    site_counts[i] += 1
        
        # Global if all sites appear roughly equally
        if max(site_counts) == 0:
            return False
        return max(site_counts) - min(site_counts) <= 1
    
    def normalize_operator(self, op: SparsePauliOp) -> SparsePauliOp:
        """Normalize operator to unit norm."""
        norm = np.linalg.norm(op.coeffs)
        if norm > self.tolerance:
            return SparsePauliOp(op.paulis, op.coeffs / norm)
        return op
    
    def generate_initial_operators(self) -> List[PoolOperator]:
        """
        Generate initial operator set.
        
        For ADAPT-VQE, start with single-site Pauli operators.
        """
        initial_ops = []
        
        # Single-site Pauli operators
        for i in range(self.model.num_sites):
            for pauli in ['X', 'Y', 'Z']:
                pauli_str = ['I'] * self.model.num_sites
                pauli_str[i] = pauli
                op = SparsePauliOp.from_list([("".join(reversed(pauli_str)), 1.0)])
                initial_ops.append(PoolOperator(op, order=0))
        
        return initial_ops
    
    def generate_pool(self, 
                     max_order: int = 3,
                     global_only: bool = False,
                     max_pool_size: int = 100) -> List[SparsePauliOp]:
        """
        Generate operator pool via commutator expansion.
        
        Args:
            max_order: Maximum commutator order
            global_only: If True, keep only translationally invariant operators
            max_pool_size: Maximum number of operators to keep
        
        Returns:
            List of SparsePauliOp operators
        """
        # Start with initial operators
        current_ops = self.generate_initial_operators()
        all_ops = set(current_ops)
        
        print(f"Starting with {len(current_ops)} initial operators")
        
        # Iteratively compute commutators
        for order in range(1, max_order + 1):
            print(f"\nOrder {order}: Computing commutators...")
            new_ops = []
            
            for op_data in current_ops:
                # Compute [H, O]
                comm = self.commutator(self.H, op_data.operator)
                
                # Skip if zero or not Hermitian
                if len(comm.paulis) == 0:
                    continue
                if not self.is_hermitian(comm):
                    continue
                
                # Normalize
                comm = self.normalize_operator(comm)
                
                # Check if global (if required)
                if global_only and not self.is_global(comm):
                    continue
                
                # Create new operator
                new_op = PoolOperator(comm, order=order, parent_idx=None)
                
                # Check if unique
                if new_op not in all_ops:
                    new_ops.append(new_op)
                    all_ops.add(new_op)
            
            print(f"  Found {len(new_ops)} new unique operators")
            current_ops = new_ops
            
            if len(all_ops) >= max_pool_size:
                print(f"  Reached max pool size ({max_pool_size})")
                break
        
        # Convert to list of SparsePauliOp
        pool = [op.operator for op in all_ops]
        
        # Sort by order
        pool_with_order = [(op.operator, op.order) for op in all_ops]
        pool_with_order.sort(key=lambda x: x[1])
        pool = [op for op, _ in pool_with_order]
        
        print(f"\nFinal pool size: {len(pool)}")
        return pool
    
    def compare_to_reference(self, discovered_pool: List[SparsePauliOp],
                            reference_pool: List[SparsePauliOp]) -> dict:
        """
        Compare discovered pool to reference pool.
        
        Returns:
            Dictionary with comparison metrics
        """
        # Convert to sets of Pauli strings
        discovered_set = set(str(op.paulis) for op in discovered_pool)
        reference_set = set(str(op.paulis) for op in reference_pool)
        
        # Compute overlap
        intersection = discovered_set & reference_set
        union = discovered_set | reference_set
        
        return {
            'discovered_size': len(discovered_pool),
            'reference_size': len(reference_pool),
            'intersection_size': len(intersection),
            'jaccard_similarity': len(intersection) / len(union) if union else 0,
            'recall': len(intersection) / len(reference_set) if reference_set else 0,
            'precision': len(intersection) / len(discovered_set) if discovered_set else 0,
            'missing_from_discovered': reference_set - discovered_set,
            'extra_in_discovered': discovered_set - reference_set
        }


def visualize_pool_structure(pool: List[SparsePauliOp], 
                             model: PhysicsModel,
                             save_path: Optional[str] = None):
    """Visualize operator pool structure."""
    import matplotlib.pyplot as plt
    
    # Analyze pool
    num_ops = len(pool)
    num_terms = [len(op.paulis) for op in pool]
    max_weight = [max([sum(1 for p in str(pauli) if p != 'I') 
                      for pauli in op.paulis]) if len(op.paulis) > 0 else 0 
                 for op in pool]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Number of terms per operator
    axes[0].hist(num_terms, bins=20, edgecolor='black')
    axes[0].set_xlabel('Number of Pauli Terms', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Pool Operator Complexity', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Maximum weight
    axes[1].hist(max_weight, bins=range(model.num_sites + 2), edgecolor='black')
    axes[1].set_xlabel('Maximum Pauli Weight', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Operator Locality', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Cumulative pool size
    axes[2].plot(range(1, num_ops + 1), range(1, num_ops + 1), 'o-')
    axes[2].set_xlabel('Operator Index', fontsize=12)
    axes[2].set_ylabel('Cumulative Pool Size', fontsize=12)
    axes[2].set_title(f'Total Pool Size: {num_ops}', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved pool visualization to {save_path}")
    else:
        plt.show()
    
    return fig
