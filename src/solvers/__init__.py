"""
Solvers Module - Ground State Computation
==========================================

Consolidated module for all ground state solvers:
- Exact diagonalization
- ADAPT-VQE
- Standard VQE
- Adiabatic evolution

Replaces: simulation/exact.py, groundstate/*, simulation/adapt_vqe.py
"""

from .exact import ExactSolver
from .adapt_vqe import ADAPTVQESolver
from .adiabatic import AdiabaticSolver

__all__ = ['ExactSolver', 'ADAPTVQESolver', 'AdiabaticSolver']
