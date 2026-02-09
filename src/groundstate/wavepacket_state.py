"""
Approximate wavepacket preparation routines layered on top of the vacuum circuit.

The helper exposed here keeps the structure intentionally lightweight so the
resulting states remain compatible with classical matrix-product-state (MPS)
simulations.  Each requested packet is injected by applying localized Y
rotations (to create an excitation envelope) and position-dependent Z rotations
that imprint the desired momentum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableSequence, Sequence

import numpy as np
from qiskit import QuantumCircuit

from .vacuum_state import build_approx_vacuum_circuit


@dataclass(frozen=True)
class WavepacketConfig:
    """Container describing a single localized packet."""

    center: float
    width: float = 2.0
    momentum: float = 0.0
    amplitude: float = 0.35
    phase_offset: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "WavepacketConfig":
        if "center" not in data:
            raise ValueError("Each packet configuration must supply a 'center'.")
        return cls(
            center=float(data["center"]),
            width=float(data.get("width", cls.width)),
            momentum=float(data.get("momentum", cls.momentum)),
            amplitude=float(data.get("amplitude", cls.amplitude)),
            phase_offset=float(data.get("phase_offset", cls.phase_offset)),
        )


def _coerce_packet_configs(configs: Iterable[Mapping[str, float] | WavepacketConfig]) -> list[WavepacketConfig]:
    packets: MutableSequence[WavepacketConfig] = []
    for cfg in configs:
        if isinstance(cfg, WavepacketConfig):
            packets.append(cfg)
        else:
            packets.append(WavepacketConfig.from_mapping(cfg))
    return list(packets)


def _gaussian_envelope(num_sites: int, packet: WavepacketConfig) -> np.ndarray:
    coords = np.arange(num_sites, dtype=float)
    width = max(float(abs(packet.width)), 1e-6)
    envelope = np.exp(-0.5 * ((coords - packet.center) / width) ** 2)
    peak = float(envelope.max()) if envelope.size else 0.0
    if peak > 0.0:
        envelope /= peak
    return envelope


def _apply_packet_layers(circuit: QuantumCircuit, packet: WavepacketConfig) -> None:
    """Dress the circuit with localized rotations for a single packet."""

    num_sites = circuit.num_qubits
    coords = np.arange(num_sites, dtype=float)
    envelope = _gaussian_envelope(num_sites, packet)

    # Keep rotations bounded to avoid numerical instability.
    amp = float(np.clip(packet.amplitude, -np.pi, np.pi))
    rotation_angles = 2.0 * amp * envelope

    # Momentum determines how phases wind across the lattice.
    base_phases = packet.phase_offset + packet.momentum * (coords - packet.center)
    phase_angles = envelope * base_phases

    for site, theta in enumerate(rotation_angles):
        if abs(theta) > 1e-9:
            circuit.ry(theta, site)
        phi = phase_angles[site]
        if abs(phi) > 1e-9:
            circuit.rz(phi, site)


def build_wavepacket_circuit(
    num_sites: int,
    vacuum_params: Mapping[str, float] | None,
    packet_configs: Sequence[Mapping[str, float] | WavepacketConfig] | None,
) -> QuantumCircuit:
    """
    Assemble a circuit containing the approximate vacuum plus optional packets.

    Parameters
    ----------
    num_sites:
        Number of qubits in the chain.
    vacuum_params:
        Parameter dictionary forwarded to ``build_approx_vacuum_circuit``.
    packet_configs:
        Sequence describing each packet.  Each entry may be either a plain
        mapping with the keys ``center``, ``width``, ``amplitude``, ``momentum``,
        ``phase_offset`` or an already-instantiated ``WavepacketConfig``.
    """

    if num_sites < 2:
        raise ValueError("num_sites must be at least 2.")

    circuit = build_approx_vacuum_circuit(num_sites, vacuum_params)

    if not packet_configs:
        return circuit

    packets = _coerce_packet_configs(packet_configs)
    for packet in packets:
        _apply_packet_layers(circuit, packet)

    return circuit


__all__ = ["WavepacketConfig", "build_wavepacket_circuit"]


