"""
Reusable visualization functions for quantum criticality analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List
from matplotlib.colors import LinearSegmentedColormap

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analysis.framework import (
    PhaseData, 
    ScalingData, 
    EntanglementData,
    BoundaryCondition
)


def plot_2d_phase_diagram(
    phase_data: PhaseData,
    critical_line: Optional[Dict[str, Any]] = None,
    contours: Optional[Dict[str, Any]] = None,
    markers: Optional[List[Dict[str, Any]]] = None,
    figsize: Tuple[float, float] = (12, 9),
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot 2D phase diagram from PhaseData.
    
    Args:
        phase_data: PhaseData object.
        critical_line: Dict with 'type' ('vertical' or 'horizontal'), 'value', 'label'.
        contours: Dict with 'grid', 'levels', 'labels', 'fmt'.
        markers: List of dicts with 'x', 'y', 'label', 'color', 'marker'.
        figsize: Figure size.
        save_path: Optional path to save figure.
        **kwargs: Additional plot customization.
        
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot gap as color map
    extent = [
        phase_data.param1_values[0],
        phase_data.param1_values[-1],
        phase_data.param2_values[0],
        phase_data.param2_values[-1]
    ]
    
    cmap = kwargs.get('cmap', 'viridis')
    im = ax.imshow(
        phase_data.gap_grid,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=cmap,
        interpolation='bilinear'
    )
    
    # Add critical line if specified
    if critical_line is not None:
        if critical_line['type'] == 'vertical':
            ax.axvline(
                x=critical_line['value'],
                color='red',
                linestyle='--',
                linewidth=3,
                label=critical_line.get('label', 'Critical line')
            )
        elif critical_line['type'] == 'horizontal':
            ax.axhline(
                y=critical_line['value'],
                color='red',
                linestyle='--',
                linewidth=3,
                label=critical_line.get('label', 'Critical line')
            )
    
    # Add contours if specified
    if contours is not None:
        G1, G2 = np.meshgrid(phase_data.param1_values, phase_data.param2_values)
        contour_plot = ax.contour(
            G1, G2, contours['grid'],
            levels=contours.get('levels', 10),
            colors='white',
            linewidths=1.5,
            alpha=0.6
        )
        if contours.get('labels', True):
            ax.clabel(contour_plot, inline=True, fontsize=9, 
                     fmt=contours.get('fmt', '%.1f'))
    
    # Add markers if specified
    if markers is not None:
        for marker in markers:
            ax.scatter(
                [marker['x']], [marker['y']],
                color=marker.get('color', 'yellow'),
                s=marker.get('size', 300),
                marker=marker.get('marker', '*'),
                edgecolors='black',
                linewidths=2,
                label=marker.get('label', ''),
                zorder=10
            )
    
    # Labels and formatting
    ax.set_xlabel(kwargs.get('xlabel', f'${phase_data.param1_name}$'), fontsize=14)
    ax.set_ylabel(kwargs.get('ylabel', f'${phase_data.param2_name}$'), fontsize=14)
    
    title = kwargs.get('title', 
                      f'{phase_data.model_name} Phase Diagram ({phase_data.boundary_condition})')
    ax.set_title(title, fontsize=16)
    
    if critical_line is not None or markers is not None:
        ax.legend(fontsize=12, loc='upper right')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(kwargs.get('cbar_label', r'Gap $\Delta E$'), fontsize=12)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=kwargs.get('dpi', 150))
        print(f"Saved: {save_path}")
    
    return fig


def plot_scaling_collapse(
    scaling_data: ScalingData,
    figsize: Tuple[float, float] = (16, 6),
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot scaling collapse demonstration.
    
    Args:
        scaling_data: ScalingData object.
        figsize: Figure size.
        save_path: Optional path to save figure.
        **kwargs: Additional customization.
        
    Returns:
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(scaling_data.sizes)))
    
    # Left: Raw data
    ax1 = axes[0]
    for idx, (size, gaps) in enumerate(zip(scaling_data.sizes, scaling_data.gaps)):
        ax1.plot(
            scaling_data.param_values, gaps,
            'o-', color=colors[idx],
            label=f'L={size}',
            markersize=4,
            linewidth=2
        )
    
    ax1.axvline(
        x=scaling_data.param_critical,
        color='red',
        linestyle='--',
        alpha=0.7,
        label=f'Critical'
    )
    ax1.set_xlabel(kwargs.get('xlabel_raw', 'Parameter'), fontsize=13)
    ax1.set_ylabel(r'Gap $\Delta E$', fontsize=13)
    ax1.set_title('Raw Data (No Collapse)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Scaled data
    ax2 = axes[1]
    for idx, (size, gaps) in enumerate(zip(scaling_data.sizes, scaling_data.gaps)):
        x_scaled = (scaling_data.param_values - scaling_data.param_critical) * \
                   (size ** (1.0 / scaling_data.nu))
        y_scaled = gaps * (size ** scaling_data.z)
        
        ax2.plot(
            x_scaled, y_scaled,
            'o', color=colors[idx],
            label=f'L={size}',
            markersize=5,
            alpha=0.7
        )
    
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_xlabel(
        rf'$(g - g_c) \times L^{{1/\nu}}$ ($\nu={scaling_data.nu}$)',
        fontsize=13
    )
    ax2.set_ylabel(rf'$\Delta E \times L^z$ ($z={scaling_data.z}$)', fontsize=13)
    ax2.set_title(kwargs.get('title_scaled', 'Scaled Data (Collapse)'), fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=kwargs.get('dpi', 150))
        print(f"Saved: {save_path}")
    
    return fig


def plot_entanglement_scaling(
    ent_data: EntanglementData,
    figsize: Tuple[float, float] = (10, 7),
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot entanglement entropy with central charge fit.
    
    Args:
        ent_data: EntanglementData object.
        figsize: Figure size.
        save_path: Optional path to save figure.
        **kwargs: Additional customization.
        
    Returns:
        Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Data points
    ax.plot(
        ent_data.subsystem_sizes,
        ent_data.entropies,
        'o',
        markersize=10,
        label='Numerical data',
        color='blue'
    )
    
    # Fit curve
    if ent_data.central_charge is not None:
        ell_fit = np.linspace(
            ent_data.subsystem_sizes[0],
            ent_data.subsystem_sizes[-1],
            100
        )
        S_fit = (ent_data.central_charge / 3.0) * np.log(ell_fit) + ent_data.fit_offset
        
        ax.plot(
            ell_fit, S_fit,
            '--',
            linewidth=2,
            color='red',
            label=rf'Fit: $S = ({ent_data.central_charge:.3f}/3)\log(\ell) + {ent_data.fit_offset:.2f}$'
        )
    
    ax.set_xlabel(r'Subsystem size $\ell$', fontsize=13)
    ax.set_ylabel(r'Entanglement entropy $S(\ell)$', fontsize=13)
    
    title = kwargs.get('title', 
                      f'Entanglement Entropy ({ent_data.boundary_condition}, N={ent_data.num_sites})')
    if ent_data.central_charge is not None:
        title += f'\nCentral Charge: $c = {ent_data.central_charge:.4f}$'
    
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=kwargs.get('dpi', 150))
        print(f"Saved: {save_path}")
    
    return fig


def compare_boundary_conditions(
    data_pbc: Any,
    data_obc: Any,
    comparison_type: str = 'entanglement',
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Compare PBC vs OBC results side-by-side.
    
    Args:
        data_pbc: Data with PBC.
        data_obc: Data with OBC.
        comparison_type: 'entanglement', 'scaling', or 'phase'.
        figsize: Figure size.
        save_path: Optional path to save figure.
        **kwargs: Additional customization.
        
    Returns:
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if comparison_type == 'entanglement':
        # Plot PBC
        axes[0].plot(data_pbc.subsystem_sizes, data_pbc.entropies, 'o-', 
                    markersize=8, label=f'c = {data_pbc.central_charge:.3f}')
        axes[0].set_title(f'PBC (N={data_pbc.num_sites})', fontsize=14)
        axes[0].set_xlabel(r'$\ell$', fontsize=12)
        axes[0].set_ylabel(r'$S(\ell)$', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot OBC
        axes[1].plot(data_obc.subsystem_sizes, data_obc.entropies, 'o-', 
                    markersize=8, color='orange', label=f'c = {data_obc.central_charge:.3f}')
        axes[1].set_title(f'OBC (N={data_obc.num_sites})', fontsize=14)
        axes[1].set_xlabel(r'$\ell$', fontsize=12)
        axes[1].set_ylabel(r'$S(\ell)$', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(kwargs.get('suptitle', 'PBC vs OBC Comparison'), fontsize=16)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=kwargs.get('dpi', 150))
        print(f"Saved: {save_path}")
    
    return fig
