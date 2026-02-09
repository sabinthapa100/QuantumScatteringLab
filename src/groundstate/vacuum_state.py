"""
Approximate vacuum preparation circuit for the Ising MPS simulations.

The design follows the qualitative guidance from ``docs/ising_mps_specs.md``:

- For the ordered (interaction-dominated) regime begin in a computational-basis
  ferromagnet and add weak transverse rotations.
- For the near-critical / paramagnetic regime bias toward the ``|+...+>`` state
  and dress it with shallow entangling layers that mimic ADAPT-VQE corrections.

The helper below keeps the circuit intentionally lightweight so it can act as a
reference vacuum for matrix-product-state benchmarks without reproducing the
full ADAPT-VQE workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import numpy as np
from qiskit import QuantumCircuit


@dataclass
class VacuumAnsatzConfig:
    """Configuration container for the approximate vacuum circuit."""

    interaction_strength: float = 1.0  # J
    transverse_field: float = 1.25  # g_x
    longitudinal_field: float = 0.15  # g_z
    num_layers: int = 3
    start_in_plus: bool | None = None
    entangling_scale: float = 0.8
    rotation_scale: float = 1.0
    include_periodic_entangler: bool = True

    @classmethod
    def from_mapping(cls, params: Mapping[str, float] | None) -> "VacuumAnsatzConfig":
        if params is None:
            return cls()

        kwargs: MutableMapping[str, float] = {}
        for key in (
            "interaction_strength",
            "transverse_field",
            "longitudinal_field",
            "entangling_scale",
            "rotation_scale",
        ):
            if key in params:
                kwargs[key] = float(params[key])
        if "num_layers" in params:
            kwargs["num_layers"] = int(params["num_layers"])

        if "start_in_plus" in params:
            kwargs["start_in_plus"] = bool(params["start_in_plus"])
        if "include_periodic_entangler" in params:
            kwargs["include_periodic_entangler"] = bool(params["include_periodic_entangler"])

        return cls(**kwargs)

    @property
    def field_ratio(self) -> float:
        denom = max(abs(self.interaction_strength), 1e-6)
        return float(abs(self.transverse_field) / denom)

    @property
    def prefers_transverse(self) -> bool:
        threshold = 0.9
        return self.start_in_plus if self.start_in_plus is not None else self.field_ratio >= threshold


def _initial_reference_state(circuit: QuantumCircuit, cfg: VacuumAnsatzConfig) -> None:
    """Prepare the ordered product state referenced in the specs."""

    if cfg.prefers_transverse:
        # Ordered along +X = |+> to match the field-dominated regime.
        for qubit in range(circuit.num_qubits):
            circuit.h(qubit)
    else:
        # Start from |0...0> (or |1...1> if longitudinal field flips the bias).
        if cfg.longitudinal_field < 0:
            for qubit in range(circuit.num_qubits):
                circuit.x(qubit)


def _layer_single_qubit_rotations(circuit: QuantumCircuit, cfg: VacuumAnsatzConfig, layer_idx: int) -> None:
    """Apply global single-qubit rotations that mimic ADAPT-VQE Y layers."""

    # Larger transverse field => larger mixing angle.
    base_angle = np.arctan2(cfg.transverse_field, cfg.interaction_strength + 1e-9)
    alternating = 1.0 if layer_idx % 2 == 0 else -0.6  # small alternation to break symmetry.
    theta = 2.0 * cfg.rotation_scale * base_angle * alternating

    # Longitudinal field induces a Z bias; treat it as a small corrective phase.
    z_angle = 2.0 * cfg.longitudinal_field * 0.2

    for qubit in range(circuit.num_qubits):
        if abs(theta) > 1e-9:
            circuit.ry(theta, qubit)
        if abs(z_angle) > 1e-9:
            circuit.rz(z_angle, qubit)


def _layer_entanglers(circuit: QuantumCircuit, cfg: VacuumAnsatzConfig, layer_idx: int) -> None:
    """Add shallow ZZ entanglers to capture short-range correlations."""

    strength = cfg.entangling_scale * cfg.interaction_strength
    decay = 1.0 / max(cfg.num_layers, 1)
    angle = 2.0 * (strength * (1.0 - decay * layer_idx)) / (cfg.interaction_strength + cfg.transverse_field + 1e-9)

    for qubit in range(circuit.num_qubits - 1):
        circuit.rzz(angle, qubit, qubit + 1)
    if cfg.include_periodic_entangler and circuit.num_qubits > 2:
        circuit.rzz(angle, circuit.num_qubits - 1, 0)


def build_approx_vacuum_circuit(
    num_sites: int,
    params: Mapping[str, float] | None = None,
) -> QuantumCircuit:
    """
    Construct a lightweight circuit that approximates the Ising vacuum.

    Parameters
    ----------
    num_sites:
        Number of lattice sites / qubits.
    params:
        Optional dictionary containing both physical couplings (``interaction_strength``,
        ``transverse_field``, ``longitudinal_field``) and ansatz tweaks (``num_layers``,
        ``rotation_scale``, ``entangling_scale``, ``start_in_plus``).

    Returns
    -------
    QuantumCircuit
        Circuit that prepares the approximate vacuum when applied to ``|0...0>``.
    """

    if num_sites < 2:
        raise ValueError("num_sites must be at least 2.")

    cfg = VacuumAnsatzConfig.from_mapping(params)
    circuit = QuantumCircuit(num_sites, name="approx_vac")

    _initial_reference_state(circuit, cfg)

    for layer in range(max(cfg.num_layers, 1)):
        _layer_single_qubit_rotations(circuit, cfg, layer)
        _layer_entanglers(circuit, cfg, layer)

    return circuit


__all__ = ["VacuumAnsatzConfig", "build_approx_vacuum_circuit"]


