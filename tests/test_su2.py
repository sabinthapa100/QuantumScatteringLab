import unittest
from qiskit.quantum_info import SparsePauliOp
from src.models.su2 import SU2GaugeModel

class TestSU2GaugeModel(unittest.TestCase):
    
    def test_pool_size(self):
        """Test if operator pool has correct size for N=3 PBC."""
        # N=3 sites.
        # Pool types: Y (3), YZ (3), ZY (3), ZYZ (3 terms)
        # Total = 12 terms
        model = SU2GaugeModel(num_sites=3, pbc=True)
        pool = model.build_operator_pool()
        self.assertEqual(len(pool), 12)
        
    def test_trotter_layers(self):
        """Test Trotter layer composition."""
        N = 4
        model = SU2GaugeModel(num_sites=N)
        layers = model.get_trotter_layers()
        
        # Expect 3 layers: H_E, H_M_Even, H_M_Odd
        self.assertEqual(len(layers), 3)
        
        # Verify H_E contains only Zs
        he = layers[0]
        for term in he.paulis:
            label = term.to_label()
            self.assertTrue("X" not in label)
            self.assertTrue("Y" not in label)
            
        # Verify H_M layers contain Xs (since M terms have X)
        hm_even = layers[1]
        has_x_even = any("X" in term.to_label() for term in hm_even.paulis)
        self.assertTrue(has_x_even)

if __name__ == '__main__':
    unittest.main()
