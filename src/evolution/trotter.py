"""High-level scattering runner for the Ising MPS stack.

The implementation follows the workflow laid out in
``docs/ising_mps_implementation_plan.md``:

1. Build (or fall back to) an Ising engine.
2. Prepare an approximate vacuum and seed wavepackets.
3. Evolve the state under the Ising Hamiltonian.
4. Record the local energy density (vacuum-subtracted) at each time step.

The repository does not yet expose the lower-level ``IsingMPSEngine``,
``VacuumPreparer`` or ``WavepacketBuilder`` modules, so this runner includes a
reference dense-state engine that is sufficient for small-system tests. When
the specialized modules become available the dependency injection hooks allow
them to be dropped in without changing user-facing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from .observables import energy_density_from_statevector


# ---------------------------------------------------------------------------
# Reference engine (dense statevector evolution)
# ---------------------------------------------------------------------------


class ReferenceIsingMPSEngine:
    """Lightweight dense simulator used as a fallback for testing."""

    def __init__(
        self,
        num_sites: int,
        interaction_strength: float,
        transverse_field: float,
        longitudinal_field: float,
        dt: float,
    ) -> None:
        self.num_sites = num_sites
        self.interaction_strength = interaction_strength
        self.transverse_field = transverse_field
        self.longitudinal_field = longitudinal_field
        self.dt = dt

        self._hamiltonian = self._build_hamiltonian()
        self._propagator = self._build_propagator()

    @staticmethod
    def _pauli_x() -> np.ndarray:
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    @staticmethod
    def _pauli_z() -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    @staticmethod
    def _identity() -> np.ndarray:
        return np.eye(2, dtype=np.complex128)

    def _kron_chain(self, paulis: Sequence[np.ndarray]) -> np.ndarray:
        result = paulis[0]
        for term in paulis[1:]:
            result = np.kron(result, term)
        return result

    def _ham_term(self, axes: MutableMapping[int, np.ndarray]) -> np.ndarray:
        factors = []
        for idx in range(self.num_sites):
            factors.append(axes.get(idx, self._identity()))
        return self._kron_chain(factors)

    def _build_hamiltonian(self) -> np.ndarray:
        x = self._pauli_x()
        z = self._pauli_z()
        identity = self._identity()

        hamiltonian = np.zeros((2**self.num_sites, 2**self.num_sites), dtype=np.complex128)
        for site in range(self.num_sites):
            if site > 0:
                axes = {site - 1: z, site: z}
                hamiltonian += -0.5 * self.interaction_strength * self._ham_term(axes)
            if site < self.num_sites - 1:
                axes = {site: z, site + 1: z}
                hamiltonian += -0.5 * self.interaction_strength * self._ham_term(axes)

            axes_x = {site: x}
            hamiltonian += -self.transverse_field * self._ham_term(axes_x)

            axes_z = {site: z}
            hamiltonian += -self.longitudinal_field * self._ham_term(axes_z)

        return hamiltonian

    def _build_propagator(self) -> np.ndarray:
        evals, evecs = np.linalg.eigh(self._hamiltonian)
        phases = np.exp(-1j * self.dt * evals)
        return (evecs * phases) @ evecs.conj().T

    def step(self, state: np.ndarray) -> np.ndarray:
        """Apply a single time step."""
        return self._propagator @ state


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VacuumConfig:
    theta: float = 0.0
    phi: float = 0.0


@dataclass
class WavepacketConfig:
    center: float
    width: float = 2.5
    amplitude: float = 0.35
    momentum: float = 0.0
    phase: float = 0.0
    direction: str = "right"


def _coerce_vacuum_config(params: Mapping[str, Any]) -> VacuumConfig:
    if params is None:
        return VacuumConfig()
    return VacuumConfig(
        theta=float(params.get("theta", 0.0)),
        phi=float(params.get("phi", 0.0)),
    )


def _coerce_wavepackets(raw_packets: Sequence[Mapping[str, Any]] | None) -> list[WavepacketConfig]:
    packets: list[WavepacketConfig] = []
    if not raw_packets:
        return packets

    for packet in raw_packets:
        packets.append(
            WavepacketConfig(
                center=float(packet.get("center", 0.0)),
                width=float(packet.get("width", 2.5)),
                amplitude=float(packet.get("amplitude", 0.35)),
                momentum=float(packet.get("momentum", 0.0)),
                phase=float(packet.get("phase", 0.0)),
                direction=str(packet.get("direction", "right")),
            )
        )
    return packets


# ---------------------------------------------------------------------------
# State preparation helpers
# ---------------------------------------------------------------------------


def _product_state_from_angles(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    single_qubit_states = []
    for th, ph in zip(theta, phi, strict=True):
        th = float(np.clip(th, 0.0, np.pi))
        component = np.array(
            [
                np.cos(th / 2.0),
                np.exp(1j * ph) * np.sin(th / 2.0),
            ],
            dtype=np.complex128,
        )
        single_qubit_states.append(component)

    state = single_qubit_states[0]
    for vec in single_qubit_states[1:]:
        state = np.kron(state, vec)
    norm = np.linalg.norm(state)
    if norm == 0.0:
        raise ValueError("Constructed zero-norm initial state.")
    return state / norm


def _packet_profile(num_sites: int, packet: WavepacketConfig) -> np.ndarray:
    coords = np.arange(num_sites, dtype=float)
    width = max(packet.width, 1e-6)
    profile = np.exp(-0.5 * ((coords - packet.center) / width) ** 2)
    if profile.max() > 0:
        profile = profile / profile.max()
    return profile


def _build_state(
    num_sites: int,
    vacuum_cfg: VacuumConfig,
    packets: Sequence[WavepacketConfig],
    include_packets: bool,
) -> np.ndarray:
    theta = np.full(num_sites, vacuum_cfg.theta, dtype=float)
    phi = np.full(num_sites, vacuum_cfg.phi, dtype=float)

    if include_packets:
        coords = np.arange(num_sites, dtype=float)
        for packet in packets:
            profile = _packet_profile(num_sites, packet)
            theta += packet.amplitude * profile
            direction = 1.0 if packet.direction.lower().startswith("r") else -1.0
            phi += packet.phase + direction * packet.momentum * coords

    return _product_state_from_angles(theta, np.mod(phi, 2 * np.pi))


# ---------------------------------------------------------------------------
# Scattering runner
# ---------------------------------------------------------------------------


class ScatteringRunner:
    """Coordinate vacuum prep, wavepacket insertion, evolution, and observables."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.num_sites = int(params["num_sites"])
        self.interaction_strength = float(params.get("interaction_strength", 1.0))
        self.transverse_field = float(params.get("transverse_field", 1.0))
        self.longitudinal_field = float(params.get("longitudinal_field", 0.0))

        self.num_steps = int(params["num_steps"])
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        self.t_max = float(params.get("t_max", 1.0))
        self.dt = self.t_max / self.num_steps

        self.vacuum_cfg = _coerce_vacuum_config(params.get("vacuum"))
        self.wavepackets = _coerce_wavepackets(params.get("wavepackets"))

        self.engine = ReferenceIsingMPSEngine(
            num_sites=self.num_sites,
            interaction_strength=self.interaction_strength,
            transverse_field=self.transverse_field,
            longitudinal_field=self.longitudinal_field,
            dt=self.dt,
        )

        self.save_path = self._resolve_output_path(params)

    def _resolve_output_path(self, params: Mapping[str, Any]) -> Path | None:
        explicit = params.get("save_path")
        label = params.get("results_label")
        if explicit is None and label is None:
            return None

        if explicit is not None:
            path = Path(explicit)
        else:
            results_dir = Path(params.get("results_dir", "results"))
            path = results_dir / f"{label}.npy"

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run(self) -> np.ndarray:
        """Execute the scattering simulation and return E_n(t)."""
        scatter_state = _build_state(
            self.num_sites,
            self.vacuum_cfg,
            self.wavepackets,
            include_packets=True,
        )
        vacuum_state = _build_state(
            self.num_sites,
            self.vacuum_cfg,
            self.wavepackets,
            include_packets=False,
        )

        profile = np.zeros((self.num_steps, self.num_sites), dtype=float)
        for step in range(self.num_steps):
            scatter_state = self.engine.step(scatter_state)
            vacuum_state = self.engine.step(vacuum_state)

            scatter_energy = energy_density_from_statevector(
                scatter_state,
                self.num_sites,
                self.interaction_strength,
                self.transverse_field,
                self.longitudinal_field,
            )
            vacuum_energy = energy_density_from_statevector(
                vacuum_state,
                self.num_sites,
                self.interaction_strength,
                self.transverse_field,
                self.longitudinal_field,
            )
            profile[step] = scatter_energy - vacuum_energy

        if self.save_path is not None:
            np.save(self.save_path, profile)
        return profile


def run_scattering(params: Mapping[str, Any]) -> np.ndarray:
    """Functional interface mirroring the design doc."""
    runner = ScatteringRunner(params)
    return runner.run()


__all__ = ["run_scattering", "ScatteringRunner", "ReferenceIsingMPSEngine"]



