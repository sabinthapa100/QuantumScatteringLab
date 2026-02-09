import unittest
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from src.models.ising_1d import IsingModel1D

class TestIsingModel1D(unittest.TestCase):
    
    def test_hamiltonian_open_boundary(self):
        """Test H construction for N=3 Open BC."""
        N = 3
        J = 1.0
        gk = 0.5
        model = IsingModel1D(num_sites=N, j_int=J, g_x=gk, pbc=False)
        H = model.build_hamiltonian()
        
        # Expected terms:
        # Bonds: 0-1, 1-2 (2 terms)
        # Fields: 0, 1, 2 (3 terms)
        # Total 5 terms
        self.assertEqual(len(H.paulis), 5)
        
        # Verify coeff of Z0 Z1 (index 0 and 1)
        # Note: Qiskit SparsePauliOp stores labels.
        # We need to find "IZZ" (if Z0 Z1) or "ZZI" depending on ordering.
        # My implementation: term[i]="Z". reversed() for Qiskit.
        # i=0: "ZII..." reversed -> "...IIZ"
        # Wait, if site 0 is rightmost in Qiskit notation (standard), then reversed(["Z", "I", "I"]) = "IIZ".
        # Let's inspect the terms.
        
        # Filter for ZZ terms
        zz_ops = [op for op in H if "Z" in op.paulis[0].to_label() and "X" not in op.paulis[0].to_label() and op.paulis[0].to_label().count("Z") == 2]
        self.assertEqual(len(zz_ops), 2)
        
    def test_hamiltonian_periodic_boundary(self):
        """Test H construction for N=3 PBC."""
        N = 3
        model = IsingModel1D(num_sites=N, j_int=1.0, g_x=0.0, pbc=True)
        H = model.build_hamiltonian()
        
        # Bonds: 0-1, 1-2, 2-0 (3 terms)
        self.assertEqual(len(H.paulis), 3)

    def test_operator_pool(self):
        """Test Operator Pool generation."""
        N = 3
        model = IsingModel1D(num_sites=N, pbc=False)
        pool = model.build_operator_pool()
        
        # N=3 OBC
        # Y terms: 3
        # YZ terms: 2 (0-1, 1-2)
        # ZY terms: 2 (0-1, 1-2)
        # Total: 7 terms
        self.assertEqual(len(pool), 7)
        
    def test_trotter_layers(self):
        """Test Trotter layer splitting."""
        N = 4
        # Even bonds: 0-1, 2-3
        # Odd bonds: 1-2
        # Fields: 0, 1, 2, 3
        model = IsingModel1D(num_sites=N, pbc=False)
        layers = model.get_trotter_layers()
        
        # Should have 3 layers: Odd, Even, Field (if coeffs != 0)
        # Default g_x=1.0, g_z=0.0
        self.assertEqual(len(layers), 3) 
        
        # Check commutativity within layers?
        # Even bonds (0-1, 2-3) commute.
        even_layer = layers[1] # Assuming order [Odd, Even, Fields] based on implementation implementation:
        # Implementation: 
        # if odd_bonds: append. (starts i=0..N. i=0 is Even. i=1 is Odd.)
        # Logic in code:
        # if i % 2 == 0: even_bonds.append
        # else: odd_bonds.append
        # Then: if odd_bonds: layers.append. if even_bonds: layers.append.
        # So Layer 0 is Odd, Layer 1 is Even.
        
        # Verify Odd layer has 1 term (1-2)
        self.assertEqual(len(layers[0].paulis), 1)
        # Verify Even layer has 2 terms (0-1, 2-3)
        self.assertEqual(len(layers[1].paulis), 2)

if __name__ == '__main__':
    unittest.main()
