"""
Test Enhanced ADAPT-VQE Implementation
=======================================

Quick test to validate the enhanced ADAPT-VQE with:
- Comprehensive logging
- Overlap tracking
- Timing information
- Formatted table output

This tests on a small 1D Ising model.
"""

import numpy as np
from pathlib import Path

from src.models import IsingModel1D
from src.simulation.adapt_vqe import ADAPTVQESolver
from src.simulation.initialization import prepare_w_state
from src.backends.qiskit_backend import QiskitBackend

print("="*80)
print("TESTING ENHANCED ADAPT-VQE")
print("="*80)
print()

# Small test case
N = 4
gx = 1.0

print(f"Test case: 1D Ising, N={N}, gx={gx}")
print()

# Create model
model = IsingModel1D(num_sites=N, g_x=gx, g_z=0.0, pbc=True)

# Prepare W-state as initial state
W_state = prepare_w_state(N)
print(f"✓ W-state prepared: ||W|| = {np.linalg.norm(W_state.data):.10f}")
print()

# Create backend
backend = QiskitBackend()

# Create ADAPT-VQE solver
solver = ADAPTVQESolver(
    model=model,
    backend=backend,
    tolerance=1e-3,  # Relaxed for quick test
    max_iters=10,
    convergence_type='max',
    initial_state=W_state,
    pool_type='global'
)

print(f"✓ ADAPT-VQE solver created")
print(f"  Pool size: {len(solver.pool)}")
print(f"  Tolerance: {solver.tolerance}")
print(f"  Max iterations: {solver.max_iters}")
print()

# Run ADAPT-VQE
print("Running ADAPT-VQE...")
print()

results = solver.run()

# Print results summary
print()
print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"Final energy:      {results['energy']:.10f}")
print(f"Exact energy:      {results['exact_energy']:.10f}")
print(f"Energy error:      {results['energy_error']:.2e}")
print(f"Final overlap:     {results['overlap']:.6f} ({results['overlap']*100:.2f}%)")
print(f"Iterations:        {results['iterations']}")
print(f"Converged:         {results['converged']}")
print(f"Operators used:    {len(results['ansatz_ops'])}")
print()

# Verify convergence
if results['converged']:
    print("✓ ADAPT-VQE converged successfully!")
else:
    print("⚠ ADAPT-VQE did not converge (may need more iterations)")

if results['energy_error'] < 1e-6:
    print("✓ Energy matches exact ground state!")
else:
    print(f"⚠ Energy error: {results['energy_error']:.2e}")

if results['overlap'] > 0.99:
    print("✓ High overlap with exact ground state!")
elif results['overlap'] > 0.9:
    print("✓ Good overlap with exact ground state")
else:
    print(f"⚠ Overlap: {results['overlap']:.2%}")

print()
print("="*80)
print("TEST COMPLETE!")
print("="*80)
