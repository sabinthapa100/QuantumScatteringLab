"""
Backend Configuration and Auto-Detection
=========================================

Flexible backend system that auto-detects available resources:
- Quimb GPU (fastest)
- Quimb CPU (fast)
- NumPy (fallback, slower)

User can override with environment variable or config.
"""

import os
import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass


@dataclass
class BackendConfig:
    """Backend configuration."""
    backend: Literal['quimb_gpu', 'quimb_cpu', 'numpy']
    device: Optional[str] = None  # 'cuda:0', 'cpu', etc.
    max_qubits_exact: int = 20  # Max for exact diagonalization
    
    def __str__(self):
        if self.backend == 'quimb_gpu':
            return f"Quimb GPU ({self.device})"
        elif self.backend == 'quimb_cpu':
            return "Quimb CPU"
        else:
            return "NumPy (exact diagonalization)"


def detect_backend(prefer: Optional[str] = None) -> BackendConfig:
    """
    Auto-detect best available backend.
    
    Priority:
    1. User preference (if specified)
    2. Quimb GPU (if available)
    3. Quimb CPU (if available)
    4. NumPy (always available)
    
    Args:
        prefer: 'gpu', 'cpu', or 'numpy' to override auto-detection
    
    Returns:
        BackendConfig with detected/chosen backend
    """
    # Check environment variable
    env_backend = os.environ.get('QUANTUM_BACKEND', None)
    if env_backend:
        prefer = env_backend.lower()
    
    # Try Quimb GPU
    if prefer in [None, 'gpu', 'quimb_gpu']:
        try:
            import quimb.tensor as qtn
            import torch
            
            if torch.cuda.is_available():
                device = 'cuda:0'
                print(f"✓ Detected: Quimb with GPU ({torch.cuda.get_device_name(0)})")
                return BackendConfig(backend='quimb_gpu', device=device, max_qubits_exact=30)
        except ImportError:
            pass
    
    # Try Quimb CPU
    if prefer in [None, 'cpu', 'quimb_cpu', 'quimb']:
        try:
            import quimb.tensor as qtn
            print("✓ Detected: Quimb CPU")
            return BackendConfig(backend='quimb_cpu', device='cpu', max_qubits_exact=25)
        except ImportError:
            pass
    
    # Fallback to NumPy
    print("✓ Using: NumPy (exact diagonalization)")
    print("  Note: Limited to ~20 qubits. Install Quimb for larger systems.")
    return BackendConfig(backend='numpy', device='cpu', max_qubits_exact=20)


def get_backend_info() -> dict:
    """Get information about available backends."""
    info = {
        'numpy': True,  # Always available
        'quimb_cpu': False,
        'quimb_gpu': False,
        'gpu_name': None
    }
    
    try:
        import quimb
        info['quimb_cpu'] = True
        
        try:
            import torch
            if torch.cuda.is_available():
                info['quimb_gpu'] = True
                info['gpu_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
    except ImportError:
        pass
    
    return info


def print_backend_status():
    """Print status of all backends."""
    info = get_backend_info()
    
    print("="*70)
    print("BACKEND STATUS")
    print("="*70)
    print(f"NumPy:      {'✓ Available' if info['numpy'] else '✗ Not available'}")
    print(f"Quimb CPU:  {'✓ Available' if info['quimb_cpu'] else '✗ Not available (pip install quimb)'}")
    print(f"Quimb GPU:  {'✓ Available' if info['quimb_gpu'] else '✗ Not available'}")
    if info['gpu_name']:
        print(f"  GPU: {info['gpu_name']}")
    print("="*70)
    print()
    
    # Recommendation
    if info['quimb_gpu']:
        print("Recommendation: Use Quimb GPU for best performance")
        print("  Set: export QUANTUM_BACKEND=gpu")
    elif info['quimb_cpu']:
        print("Recommendation: Use Quimb CPU for larger systems")
        print("  Set: export QUANTUM_BACKEND=cpu")
    else:
        print("Recommendation: Install Quimb for better performance")
        print("  pip install quimb")
    print()


# Global backend configuration
_BACKEND_CONFIG: Optional[BackendConfig] = None


def get_backend(prefer: Optional[str] = None) -> BackendConfig:
    """Get or create backend configuration."""
    global _BACKEND_CONFIG
    
    if _BACKEND_CONFIG is None or prefer is not None:
        _BACKEND_CONFIG = detect_backend(prefer)
    
    return _BACKEND_CONFIG


def set_backend(backend: str):
    """Manually set backend."""
    global _BACKEND_CONFIG
    _BACKEND_CONFIG = detect_backend(prefer=backend)
    print(f"Backend set to: {_BACKEND_CONFIG}")


if __name__ == "__main__":
    print_backend_status()
    config = get_backend()
    print(f"Active backend: {config}")
