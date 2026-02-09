"""
Generic analysis framework for quantum criticality.

Provides high-level orchestration for multi-model analysis with:
- Configuration management
- Boundary condition handling
- Reusable analysis pipelines
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Type, Dict, List, Optional, Any, Tuple
from enum import Enum

from ..models.base import PhysicsModel
from .spectrum import SpectrumAnalyzer, scan_parameter
from .criticality import (
    EntanglementAnalyzer,
    extract_central_charge,
    ScalingCollapseAnalyzer
)


class BoundaryCondition(Enum):
    """Boundary condition types for lattice models."""
    PBC = "periodic"
    OBC = "open"
    
    def __str__(self):
        return self.value


@dataclass
class PhaseData:
    """Data from phase diagram computation."""
    param1_values: np.ndarray
    param2_values: np.ndarray
    gap_grid: np.ndarray
    param1_name: str
    param2_name: str
    boundary_condition: BoundaryCondition
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingData:
    """Data from scaling collapse analysis."""
    sizes: List[int]
    param_values: np.ndarray
    gaps: List[np.ndarray]  # One array per size
    param_critical: float
    nu: float
    z: float
    boundary_condition: BoundaryCondition
    model_name: str


@dataclass
class EntanglementData:
    """Data from entanglement entropy analysis."""
    subsystem_sizes: np.ndarray
    entropies: np.ndarray
    central_charge: Optional[float] = None
    fit_offset: Optional[float] = None
    boundary_condition: BoundaryCondition = BoundaryCondition.OBC
    num_sites: int = 0


@dataclass
class AnalysisConfig:
    """
    Configuration for model analysis.
    
    Example:
        config = AnalysisConfig(
            model_class=IsingModel1D,
            fixed_params={'j_int': 1.0},
            boundary_condition=BoundaryCondition.OBC,
            system_sizes=[4, 6, 8, 10]
        )
    """
    model_class: Type[PhysicsModel]
    fixed_params: Dict[str, Any]
    boundary_condition: BoundaryCondition = BoundaryCondition.PBC
    system_sizes: List[int] = field(default_factory=lambda: [4, 6, 8, 10])
    num_states: int = 6
    
    def __post_init__(self):
        """Validate configuration."""
        if not issubclass(self.model_class, PhysicsModel):
            raise TypeError(f"{self.model_class} must be a PhysicsModel subclass")
        
        if not isinstance(self.fixed_params, dict):
            raise TypeError("fixed_params must be a dictionary")
        
        if len(self.system_sizes) == 0:
            raise ValueError("system_sizes must not be empty")
    
    def create_model(self, num_sites: int, **override_params) -> PhysicsModel:
        """
        Create a model instance with the configured parameters.
        
        Args:
            num_sites: Number of lattice sites.
            **override_params: Parameters to override from fixed_params.
            
        Returns:
            Model instance.
        """
        params = self.fixed_params.copy()
        params['num_sites'] = num_sites
        params['pbc'] = (self.boundary_condition == BoundaryCondition.PBC)
        params.update(override_params)
        
        return self.model_class(**params)


class ModelAnalyzer:
    """
    High-level analyzer for any PhysicsModel.
    
    Provides reusable analysis methods:
    - Phase diagrams
    - Scaling collapse
    - Entanglement entropy
    """
    
    def __init__(self, config: AnalysisConfig):
        """
        Args:
            config: Analysis configuration.
        """
        self.config = config
        self.model_name = config.model_class.__name__
    
    def compute_1d_phase_scan(
        self,
        param_name: str,
        param_values: np.ndarray,
        system_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan a single parameter and compute gaps.
        
        Args:
            param_name: Name of parameter to scan.
            param_values: Parameter values.
            system_size: System size (uses first in config if None).
            
        Returns:
            (param_values, gaps)
        """
        if system_size is None:
            system_size = self.config.system_sizes[0]
        
        result = scan_parameter(
            model_class=self.config.model_class,
            param_name=param_name,
            param_values=param_values,
            fixed_params={
                **self.config.fixed_params,
                'num_sites': system_size,
                'pbc': (self.config.boundary_condition == BoundaryCondition.PBC)
            },
            num_states=self.config.num_states
        )
        
        return result['param_values'], result['gaps']
    
    def compute_2d_phase_diagram(
        self,
        param1_name: str,
        param1_values: np.ndarray,
        param2_name: str,
        param2_values: np.ndarray,
        system_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> PhaseData:
        """
        Compute 2D phase diagram.
        
        Args:
            param1_name: First parameter name (x-axis).
            param1_values: First parameter values.
            param2_name: Second parameter name (y-axis).
            param2_values: Second parameter values.
            system_size: System size (uses first in config if None).
            progress_callback: Optional callback(current, total).
            
        Returns:
            PhaseData object.
        """
        if system_size is None:
            system_size = self.config.system_sizes[0]
        
        gap_grid = np.zeros((len(param2_values), len(param1_values)))
        total_points = len(param1_values) * len(param2_values)
        current_point = 0
        
        for i, param2_val in enumerate(param2_values):
            for j, param1_val in enumerate(param1_values):
                model = self.config.create_model(
                    num_sites=system_size,
                    **{param1_name: param1_val, param2_name: param2_val}
                )
                
                analyzer = SpectrumAnalyzer(model)
                result = analyzer.compute_spectrum(num_states=2)
                gap_grid[i, j] = result['gap']
                
                current_point += 1
                if progress_callback is not None:
                    progress_callback(current_point, total_points)
        
        return PhaseData(
            param1_values=param1_values,
            param2_values=param2_values,
            gap_grid=gap_grid,
            param1_name=param1_name,
            param2_name=param2_name,
            boundary_condition=self.config.boundary_condition,
            model_name=self.model_name,
            metadata={'system_size': system_size}
        )
    
    def compute_scaling_collapse(
        self,
        param_name: str,
        param_values: np.ndarray,
        param_critical: float,
        nu: float = 1.0,
        z: float = 1.0
    ) -> ScalingData:
        """
        Compute data for scaling collapse analysis.
        
        Args:
            param_name: Parameter to scan.
            param_values: Parameter values.
            param_critical: Critical parameter value.
            nu: Correlation length exponent.
            z: Dynamical critical exponent.
            
        Returns:
            ScalingData object.
        """
        gaps_list = []
        
        for size in self.config.system_sizes:
            result = scan_parameter(
                model_class=self.config.model_class,
                param_name=param_name,
                param_values=param_values,
                fixed_params={
                    **self.config.fixed_params,
                    'num_sites': size,
                    'pbc': (self.config.boundary_condition == BoundaryCondition.PBC)
                },
                num_states=self.config.num_states
            )
            gaps_list.append(result['gaps'])
        
        return ScalingData(
            sizes=self.config.system_sizes,
            param_values=param_values,
            gaps=gaps_list,
            param_critical=param_critical,
            nu=nu,
            z=z,
            boundary_condition=self.config.boundary_condition,
            model_name=self.model_name
        )
    
    def compute_entanglement_at_criticality(
        self,
        critical_params: Dict[str, Any],
        system_size: Optional[int] = None,
        fit_range: Optional[Tuple[int, int]] = None
    ) -> EntanglementData:
        """
        Compute entanglement entropy at a critical point.
        
        Args:
            critical_params: Parameters at criticality.
            system_size: System size (default: largest in config).
            fit_range: Optional (start, end) for central charge fit.
            
        Returns:
            EntanglementData object.
        """
        if system_size is None:
            system_size = max(self.config.system_sizes)
        
        # Create model at critical point
        model = self.config.create_model(num_sites=system_size, **critical_params)
        
        # Get ground state
        spectrum_analyzer = SpectrumAnalyzer(model)
        E_gs, psi_gs = spectrum_analyzer.compute_ground_state()
        
        # Compute entanglement
        ent_analyzer = EntanglementAnalyzer(system_size)
        entropies_dict = ent_analyzer.scan_subsystem_sizes(psi_gs)
        
        # Convert to arrays
        subsystem_sizes = np.array(sorted(entropies_dict.keys()))
        entropies = np.array([entropies_dict[ell] for ell in subsystem_sizes])
        
        # Extract central charge
        if fit_range is None:
            fit_range = (2, system_size - 3)
        
        c, offset = extract_central_charge(subsystem_sizes, entropies, fit_range)
        
        return EntanglementData(
            subsystem_sizes=subsystem_sizes,
            entropies=entropies,
            central_charge=c,
            fit_offset=offset,
            boundary_condition=self.config.boundary_condition,
            num_sites=system_size
        )
