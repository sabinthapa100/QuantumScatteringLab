import unittest
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from src.models.ising_1d import IsingModel1D

class TestIsingModel1D(unittest.TestCase):
    
    def test_hamiltonian_open_boundary(self):
        """Test H construction for N=3 Open BC."""
        N = 3
        # Constructor signature: num_sites, g_x, g_z, pbc
        # j_int is fixed at 0.5 internally
        model = IsingModel1D(num_sites=N, g_x=0.5, g_z=0.0, pbc=False)
        H = model.build_hamiltonian()
        
        # Expected terms:
        # Bonds: 0-1, 1-2 (2 terms) -> coeff -0.5
        # Fields (X): 0, 1, 2 (3 terms) -> coeff -0.5
        # Total 5 terms
        self.assertEqual(len(H.paulis), 5)
        
        # Verify coefficients
        # We can sum the coeffs and check
        # 2 bonds * (-0.5) = -1.0
        # 3 fields * (-0.5) = -1.5
        # Total sum of coeffs = -2.5
        self.assertAlmostEqual(np.sum(H.coeffs).real, -2.5)
        
    def test_hamiltonian_periodic_boundary(self):
        """Test H construction for N=3 PBC."""
        N = 3
        model = IsingModel1D(num_sites=N, g_x=0.0, g_z=0.0, pbc=True)
        H = model.build_hamiltonian()
        
        # Only ZZ bonds: 0-1, 1-2, 2-0 (3 terms)
        # g_x=0, g_z=0
        self.assertEqual(len(H.paulis), 3)
        self.assertAlmostEqual(np.sum(H.coeffs).real, -1.5) # 3 * -0.5

    def test_operator_pool(self):
        """Test Operator Pool generation."""
        N = 3
        model = IsingModel1D(num_sites=N, pbc=False)
        pool = model.build_operator_pool()
        
        # The pool returns 'Global Sum' operators (generators)
        # For N=3: O1, O3, O4, O2, O5 should be present because N>=3
        # Total 5 generators
        self.assertEqual(len(pool), 5)
        
    def test_trotter_layers(self):
        """Test Trotter layer splitting."""
        N = 4
        # g_z=0 by default, so Z-layer (diag) is empty/skipped
        # g_x=1 by default, so X-layer is present
        # Even bonds: 0-1, 2-3
        # Odd bonds: 1-2, 3-0 (if pbc)
        
        # Test 1: OBC, g_z=0
        model = IsingModel1D(num_sites=N, g_x=1.0, g_z=0.0, pbc=False)
        layers = model.get_trotter_layers()
        
        # Layers expected:
        # 1. X terms (fields)
        # 2. Even ZZ bonds
        # 3. Odd ZZ bonds
        # Total 3 layers
        self.assertEqual(len(layers), 3) 
        
        # Test 2: PBC, g_z=0.5
        model2 = IsingModel1D(num_sites=N, g_x=1.0, g_z=0.5, pbc=True)
        layers2 = model2.get_trotter_layers()
        
        # Layers expected:
        # 1. Z terms (diagonal)
        # 2. X terms (fields)
        # 3. Even ZZ bonds
        # 4. Odd ZZ bonds
        # Total 4 layers
        self.assertEqual(len(layers2), 4)

if __name__ == '__main__':
    unittest.main()
