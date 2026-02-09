"""
W-State Validation Tests
=========================

Comprehensive tests for W-state preparation:
1. Normalization: ⟨W|W⟩ = 1
2. Structure: Equal amplitudes on single-excitation states
3. Energy expectation: ⟨W|H|W⟩ for Ising Hamiltonian
4. Overlap with ground state: |⟨W|ψ₀⟩|²

Reference: arXiv:2505.03111v2 (Farrell et al.)
"""

import pytest
import numpy as np
from qiskit.quantum_info import Statevector

from src.simulation.initialization import prepare_w_state
from src.models import IsingModel1D


class TestWStateNormalization:
    """Test W-state normalization."""
    
    @pytest.mark.parametrize("N", [4, 6, 8, 10])
    def test_normalization(self, N):
        """Verify ⟨W|W⟩ = 1 for different system sizes."""
        W = prepare_w_state(N)
        norm = np.abs(W.inner(W))
        
        print(f"\n✓ N={N}: ||W|| = {norm:.10f}")
        assert np.isclose(norm, 1.0, atol=1e-10), f"N={N}: norm={norm} != 1.0"
    
    def test_normalization_subset(self):
        """Test W-state on subset of sites."""
        N = 6
        sites = [0, 2, 4]  # Only even sites
        
        W = prepare_w_state(N, sites=sites)
        norm = np.abs(W.inner(W))
        
        print(f"\n✓ N={N}, sites={sites}: ||W|| = {norm:.10f}")
        assert np.isclose(norm, 1.0, atol=1e-10)


class TestWStateStructure:
    """Test W-state amplitude structure."""
    
    def test_single_excitation_amplitudes(self):
        """Verify equal amplitudes on single-excitation states."""
        N = 4
        W = prepare_w_state(N)
        vec = W.data
        
        expected_amp = 1.0 / np.sqrt(N)
        
        print(f"\n{'State':<10} {'Amplitude':<15} {'Expected':<15} {'Match':<10}")
        print("-" * 50)
        
        for i in range(N):
            idx = 1 << i  # 2^i
            amp = np.abs(vec[idx])
            match = "✓" if np.isclose(amp, expected_amp, atol=1e-10) else "✗"
            
            print(f"|2^{i}⟩{'':<6} {amp:<15.10f} {expected_amp:<15.10f} {match:<10}")
            assert np.isclose(amp, expected_amp, atol=1e-10), \
                f"|2^{i}⟩: amplitude={amp} != {expected_amp}"
    
    def test_zero_on_other_states(self):
        """Verify all non-single-excitation states have zero amplitude."""
        N = 4
        W = prepare_w_state(N)
        vec = W.data
        
        non_zero_count = 0
        for idx in range(2**N):
            if bin(idx).count('1') != 1:  # Not single-excitation
                amp = np.abs(vec[idx])
                if amp > 1e-10:
                    non_zero_count += 1
                    print(f"Warning: Non-zero amplitude at idx={idx} (binary: {bin(idx)}): {amp}")
                assert amp < 1e-10, f"Non-zero amplitude at idx={idx}: {amp}"
        
        print(f"\n✓ All {2**N - N} non-single-excitation states have zero amplitude")


class TestWStateEnergy:
    """Test W-state energy expectation values."""
    
    @pytest.mark.parametrize("gx", [0.5, 1.0, 2.0])
    def test_energy_expectation(self, gx):
        """Compute ⟨W|H|W⟩ for different coupling values."""
        N = 4
        
        # Prepare model and Hamiltonian
        model = IsingModel1D(num_sites=N, g_x=gx, g_z=0.0, pbc=True)
        H = model.build_hamiltonian()
        
        # W-state energy
        W = prepare_w_state(N)
        E_W = W.expectation_value(H).real
        
        # Exact ground state energy
        mat = H.to_matrix()
        E_0 = np.linalg.eigvalsh(mat)[0]
        
        # Reference state |0⟩ energy
        ref_state = Statevector.from_label('0' * N)
        E_ref = ref_state.expectation_value(H).real
        
        delta_E = E_W - E_0
        
        print(f"\ngx={gx}:")
        print(f"  E_ref = {E_ref:>10.6f}")
        print(f"  E_W   = {E_W:>10.6f}")
        print(f"  E_0   = {E_0:>10.6f}")
        print(f"  ΔE    = {delta_E:>10.6f}")
        
        # W-state should have higher energy than ground state
        assert E_W > E_0, f"W-state energy ({E_W}) should be > ground state ({E_0})"
        
        # W-state should be better than reference state
        assert E_W < E_ref, f"W-state energy ({E_W}) should be < reference state ({E_ref})"
    
    def test_energy_vs_system_size(self):
        """Test how W-state energy scales with system size."""
        gx = 1.0
        sizes = [4, 6, 8]
        
        print(f"\n{'N':<6} {'E_W':<15} {'E_0':<15} {'ΔE':<15} {'ΔE/N':<15}")
        print("-" * 66)
        
        for N in sizes:
            model = IsingModel1D(num_sites=N, g_x=gx, g_z=0.0, pbc=True)
            H = model.build_hamiltonian()
            
            W = prepare_w_state(N)
            E_W = W.expectation_value(H).real
            E_0 = np.linalg.eigvalsh(H.to_matrix())[0]
            
            delta_E = E_W - E_0
            delta_E_per_site = delta_E / N
            
            print(f"{N:<6} {E_W:<15.6f} {E_0:<15.6f} {delta_E:<15.6f} {delta_E_per_site:<15.6f}")


class TestWStateOverlap:
    """Test W-state overlap with ground state."""
    
    @pytest.mark.parametrize("gx", [0.5, 1.0, 2.0])
    def test_overlap_with_ground_state(self, gx):
        """Compute |⟨W|ψ₀⟩|² for different coupling values."""
        N = 4
        
        # Prepare model
        model = IsingModel1D(num_sites=N, g_x=gx, g_z=0.0, pbc=True)
        H = model.build_hamiltonian()
        
        # Exact ground state
        mat = H.to_matrix()
        eigs, vecs = np.linalg.eigh(mat)
        psi_0 = vecs[:, 0]
        
        # W-state
        W = prepare_w_state(N)
        W_vec = W.data
        
        # Overlap
        overlap = np.abs(np.vdot(W_vec, psi_0))**2
        
        print(f"\ngx={gx}: |⟨W|ψ₀⟩|² = {overlap:.6f}")
        
        # Expect reasonable overlap (typically > 0.3 for Ising models)
        assert overlap > 0.1, f"Overlap too small: {overlap}"
        assert overlap < 1.0, f"Overlap should be < 1.0: {overlap}"
    
    def test_overlap_vs_system_size(self):
        """Test how overlap changes with system size."""
        gx = 1.0
        sizes = [4, 6, 8, 10]
        
        print(f"\n{'N':<6} {'Overlap':<15} {'Quality':<15}")
        print("-" * 36)
        
        for N in sizes:
            model = IsingModel1D(num_sites=N, g_x=gx, g_z=0.0, pbc=True)
            H = model.build_hamiltonian()
            
            mat = H.to_matrix()
            _, vecs = np.linalg.eigh(mat)
            psi_0 = vecs[:, 0]
            
            W = prepare_w_state(N)
            overlap = np.abs(np.vdot(W.data, psi_0))**2
            
            quality = "Excellent" if overlap > 0.7 else "Good" if overlap > 0.5 else "Fair" if overlap > 0.3 else "Poor"
            
            print(f"{N:<6} {overlap:<15.6f} {quality:<15}")
            
            assert overlap > 0.1, f"N={N}: Overlap too small: {overlap}"


class TestWStatePhysics:
    """Test physical properties of W-state."""
    
    def test_magnetization(self):
        """Test magnetization ⟨X⟩ and ⟨Z⟩ for W-state."""
        N = 4
        W = prepare_w_state(N)
        
        from qiskit.quantum_info import SparsePauliOp
        
        # Compute ⟨X⟩
        X_total = 0.0
        for i in range(N):
            op_str = ['I'] * N
            op_str[i] = 'X'
            X_op = SparsePauliOp.from_list([("".join(reversed(op_str)), 1.0)])
            X_total += W.expectation_value(X_op).real
        
        avg_X = X_total / N
        
        # Compute ⟨Z⟩
        Z_total = 0.0
        for i in range(N):
            op_str = ['I'] * N
            op_str[i] = 'Z'
            Z_op = SparsePauliOp.from_list([("".join(reversed(op_str)), 1.0)])
            Z_total += Z_op.expectation_value(W).real
        
        avg_Z = Z_total / N
        
        print(f"\nW-state magnetization (N={N}):")
        print(f"  ⟨X⟩ = {avg_X:.6f}")
        print(f"  ⟨Z⟩ = {avg_Z:.6f}")
        
        # W-state should have zero Z magnetization (equal superposition)
        assert np.abs(avg_Z) < 1e-10, f"⟨Z⟩ should be ~0: {avg_Z}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
