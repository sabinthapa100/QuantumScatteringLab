"""Pytest configuration and shared fixtures."""
import pytest
import numpy as np
from typing import Generator


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def ising_model_critical():
    """1D Ising model at critical point (g_x=1, g_z=0)."""
    from src.models.ising_1d import IsingModel1D
    return IsingModel1D(num_sites=8, j_int=1.0, g_x=1.0, g_z=0.0, pbc=False)


@pytest.fixture
def ising_model_ordered():
    """1D Ising model in ordered phase (g_x<1)."""
    from src.models.ising_1d import IsingModel1D
    return IsingModel1D(num_sites=6, j_int=1.0, g_x=0.5, g_z=0.0, pbc=True)


@pytest.fixture
def ising_model_disordered():
    """1D Ising model in disordered phase (g_x>1)."""
    from src.models.ising_1d import IsingModel1D
    return IsingModel1D(num_sites=6, j_int=1.0, g_x=1.5, g_z=0.0, pbc=True)


@pytest.fixture
def su2_model():
    """SU(2) gauge model with default parameters."""
    from src.models.su2 import SU2GaugeModel
    return SU2GaugeModel(num_sites=6, g=1.0, a=1.0, pbc=True)


# ============================================================================
# Backend Fixtures
# ============================================================================

@pytest.fixture
def qiskit_backend():
    """Qiskit statevector backend."""
    from src.backends.qiskit_backend import QiskitBackend
    return QiskitBackend()


@pytest.fixture
def quimb_backend_cpu():
    """Quimb backend (CPU mode)."""
    pytest.importorskip("quimb")
    from src.backends.quimb_backend import QuimbBackend
    return QuimbBackend(use_gpu=False)


@pytest.fixture(params=['qiskit'])
def backend(request):
    """Parametrized backend fixture (currently only Qiskit)."""
    if request.param == 'qiskit':
        from src.backends.qiskit_backend import QiskitBackend
        return QiskitBackend()
    elif request.param == 'quimb':
        pytest.skip("Quimb backend not fully implemented")
    else:
        raise ValueError(f"Unknown backend: {request.param}")


# ============================================================================
# Analysis Fixtures
# ============================================================================

@pytest.fixture
def spectrum_analyzer(ising_model_critical):
    """Spectrum analyzer for critical Ising model."""
    from src.analysis.spectrum import SpectrumAnalyzer
    return SpectrumAnalyzer(ising_model_critical)


@pytest.fixture
def entanglement_analyzer():
    """Entanglement analyzer."""
    from src.analysis.criticality import EntanglementAnalyzer
    return EntanglementAnalyzer(num_sites=10)


# ============================================================================
# Reference Data Fixtures
# ============================================================================

@pytest.fixture
def known_ising_ground_state_energy():
    """Known ground state energy for 1D Ising at criticality.
    
    For N=8, g_x=1.0, g_z=0.0, OBC:
    Exact value from analytical solution.
    """
    return -8.0  # Approximate, should be computed exactly


@pytest.fixture
def tolerance():
    """Numerical tolerance for comparisons."""
    return 1e-10


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "legacy: marks tests comparing with legacy code"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests."""
    for item in items:
        if "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


# ============================================================================
# Utility Functions
# ============================================================================

@pytest.fixture
def assert_hermitian():
    """Helper function to assert matrix is Hermitian."""
    def _assert_hermitian(matrix, tol=1e-10):
        assert np.allclose(matrix, matrix.conj().T, atol=tol), \
            "Matrix is not Hermitian"
    return _assert_hermitian


@pytest.fixture
def assert_unitary():
    """Helper function to assert matrix is unitary."""
    def _assert_unitary(matrix, tol=1e-10):
        identity = np.eye(matrix.shape[0])
        product = matrix @ matrix.conj().T
        assert np.allclose(product, identity, atol=tol), \
            "Matrix is not unitary"
    return _assert_unitary


@pytest.fixture
def random_state():
    """Random number generator with fixed seed for reproducibility."""
    return np.random.RandomState(42)
