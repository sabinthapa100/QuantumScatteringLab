from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SimulationConfig:
    """
    Central configuration for the Quantum Scattering simulation.
    Designed to be easily serialized/deserialized for a Web App or GUI.
    """
    # System Parameters
    model_type: str = "ising"  # "ising" or "su2"
    num_sites: int = 4
    boundary_condition: str = "pbc"  # "pbc" or "obc"
    
    # Model Specific Parameters
    # Ising
    coupling_j: float = 1.0
    field_g: float = 1.0 # Or g_x
    field_h: float = 0.0 # Or g_z
    
    # SU(2)
    su2_g: float = 1.0
    su2_a: float = 1.0
    
    # Simulation Parameters
    backend_type: str = "qiskit"  # "qiskit" or "quimb"
    use_gpu: bool = False
    
    # Solver Parameters
    solver_type: str = "adapt-vqe" # "adapt-vqe" or "adiabatic"
    vqe_tolerance: float = 1e-6
    vqe_max_iters: int = 20
    
    # Evolution
    dt: float = 0.1
    total_time: float = 1.0
    
    def __post_init__(self):
        # Validation logic can go here
        if self.num_sites < 2:
            raise ValueError("Number of sites must be >= 2")
