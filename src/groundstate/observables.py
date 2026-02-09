"""Utilities for constructing and evaluating Ising energy-density observables.

The observable of interest is the local energy density

    H_n = -0.5 * J * (Z_{n-1} Z_n + Z_n Z_{n+1}) - g_x X_n - g_z Z_n

in accordance with the specs described in ``docs/ising_mps_implementation_plan.md``.
This module provides two complementary estimation paths:

1. Deterministic expectation values using Qiskit's ``SparsePauliOp`` plus a
   statevector / circuit.
2. Estimates derived from raw measurement counts (e.g., device samples).

For classical reference simulations (e.g., MPS, statevector), the expectation
value route avoids sampling noise and should therefore be preferred.
"""

from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np

try:
    from qiskit.quantum_info import SparsePauliOp, Statevector
except Exception:  # pragma: no cover - Qiskit might not be installed.
    SparsePauliOp = None  # type: ignore[assignment]
    Statevector = None  # type: ignore[assignment]

PauliLabel = MutableMapping[int, str]


def _pauli_label(num_sites: int, axes: PauliLabel) -> str:
    """Return a Qiskit-style Pauli label string for the provided axes."""
    if num_sites <= 0:
        raise ValueError("num_sites must be positive.")

    label = ["I"] * num_sites
    for idx, axis in axes.items():
        if idx < 0 or idx >= num_sites:
            raise ValueError(f"Qubit index {idx} out of range.")
        axis = axis.upper()
        if axis not in {"X", "Y", "Z"}:
            raise ValueError(f"Unsupported axis {axis}.")
        # Qiskit uses little-endian ordering in Pauli strings.
        label[num_sites - 1 - idx] = axis
    return "".join(label)


def local_energy_density_ops(
    num_sites: int,
    interaction_strength: float,
    transverse_field: float,
    longitudinal_field: float = 0.0,
):
    """Construct per-site SparsePauliOp objects for the local energy density."""
    if SparsePauliOp is None:  # pragma: no cover - depends on Qiskit install.
        raise RuntimeError(
            "Qiskit is required to build SparsePauliOp instances. "
            "Install qiskit-terra>=0.24."
        )

    ops = {}
    for site in range(num_sites):
        pauli_terms = []
        if site > 0:
            label = _pauli_label(num_sites, {site - 1: "Z", site: "Z"})
            pauli_terms.append((label, -0.5 * interaction_strength))
        if site < num_sites - 1:
            label = _pauli_label(num_sites, {site: "Z", site + 1: "Z"})
            pauli_terms.append((label, -0.5 * interaction_strength))

        label = _pauli_label(num_sites, {site: "X"})
        pauli_terms.append((label, -transverse_field))

        label = _pauli_label(num_sites, {site: "Z"})
        pauli_terms.append((label, -longitudinal_field))

        ops[site] = SparsePauliOp.from_list(pauli_terms)

    return ops


def evaluate_energy_density_with_state(
    state_or_circuit,
    ops: Mapping[int, "SparsePauliOp"],
    subtract_profile: Sequence[float] | None = None,
) -> np.ndarray:
    """Evaluate local energy density using Qiskit's expectation machinery."""
    if SparsePauliOp is None or Statevector is None:  # pragma: no cover
        raise RuntimeError("Qiskit is required for expectation evaluation.")

    if isinstance(state_or_circuit, Statevector):
        state = state_or_circuit
    elif hasattr(state_or_circuit, "num_qubits"):
        state = Statevector.from_instruction(state_or_circuit)
    else:
        state = Statevector(state_or_circuit)

    values = np.array(
        [np.real(state.expectation_value(ops[idx])) for idx in sorted(ops)],
        dtype=float,
    )
    if subtract_profile is not None:
        values = values - np.asarray(subtract_profile, dtype=float)
    return values


def _counts_to_expectation(
    counts: Mapping[str, int],
    qubit_indices: Iterable[int],
) -> float:
    """Compute an expectation value from raw measurement counts."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Cannot compute expectation from zero shots.")

    expectation = 0.0
    for bitstring, freq in counts.items():
        # Reverse string: Qiskit records qubit-0 in the right-most bit.
        bits = bitstring[::-1]
        parity = 1.0
        for idx in qubit_indices:
            if idx >= len(bits):
                raise ValueError("Bitstring shorter than expected.")
            parity *= 1.0 if bits[idx] == "0" else -1.0
        expectation += parity * freq
    return expectation / total


def energy_density_from_counts(
    num_sites: int,
    interaction_strength: float,
    transverse_field: float,
    longitudinal_field: float,
    counts_z: Mapping[str, int],
    counts_x: Mapping[str, int],
    counts_zz: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Estimate the energy density profile from measurement counts."""

    counts_zz = counts_zz or counts_z

    exp_z = np.array(
        [_counts_to_expectation(counts_z, [site]) for site in range(num_sites)],
        dtype=float,
    )
    exp_x = np.array(
        [_counts_to_expectation(counts_x, [site]) for site in range(num_sites)],
        dtype=float,
    )
    exp_zz = np.array(
        [
            _counts_to_expectation(counts_zz, [site, site + 1])
            for site in range(num_sites - 1)
        ],
        dtype=float,
    )
    return combine_local_energy_terms(
        exp_z=exp_z,
        exp_x=exp_x,
        exp_zz=exp_zz,
        interaction_strength=interaction_strength,
        transverse_field=transverse_field,
        longitudinal_field=longitudinal_field,
    )


def combine_local_energy_terms(
    exp_z: np.ndarray,
    exp_x: np.ndarray,
    exp_zz: np.ndarray,
    interaction_strength: float,
    transverse_field: float,
    longitudinal_field: float,
) -> np.ndarray:
    """Combine expectation values into the per-site energy density."""
    num_sites = exp_z.size
    if exp_x.size != num_sites:
        raise ValueError("exp_x must match exp_z in length.")
    if exp_zz.size != max(0, num_sites - 1):
        raise ValueError("exp_zz must have length num_sites-1.")

    profile = np.zeros(num_sites, dtype=float)
    for site in range(num_sites):
        value = 0.0
        if site > 0:
            value += -0.5 * interaction_strength * exp_zz[site - 1]
        if site < num_sites - 1:
            value += -0.5 * interaction_strength * exp_zz[site]
        value += -transverse_field * exp_x[site]
        value += -longitudinal_field * exp_z[site]
        profile[site] = value
    return profile


def _z_expectations_from_state(state: np.ndarray, num_sites: int) -> np.ndarray:
    """Return <Z_i> for i in range(num_sites)."""
    probabilities = np.abs(state) ** 2
    basis_indices = np.arange(state.size, dtype=np.uint64)
    expectations = np.empty(num_sites, dtype=float)
    for site in range(num_sites):
        parity = 1.0 - 2.0 * ((basis_indices >> site) & 1)
        expectations[site] = float(np.sum(parity * probabilities))
    return expectations


def _zz_expectations_from_state(state: np.ndarray, num_sites: int) -> np.ndarray:
    """Return <Z_i Z_{i+1}> expectations."""
    if num_sites < 2:
        return np.zeros(0, dtype=float)
    probabilities = np.abs(state) ** 2
    basis_indices = np.arange(state.size, dtype=np.uint64)
    expectations = np.empty(num_sites - 1, dtype=float)
    for site in range(num_sites - 1):
        parity = (1.0 - 2.0 * ((basis_indices >> site) & 1)) * (
            1.0 - 2.0 * ((basis_indices >> (site + 1)) & 1)
        )
        expectations[site] = float(np.sum(parity * probabilities))
    return expectations


def _x_expectations_from_state(state: np.ndarray, num_sites: int) -> np.ndarray:
    """Return <X_i> for i in range(num_sites)."""
    expectations = np.empty(num_sites, dtype=float)
    size = state.size
    for site in range(num_sites):
        mask = 1 << site
        total = 0.0 + 0.0j
        for idx in range(size):
            flipped = idx ^ mask
            total += np.conj(state[idx]) * state[flipped]
        expectations[site] = float(total.real)
    return expectations


def energy_density_from_statevector(
    state: np.ndarray,
    num_sites: int,
    interaction_strength: float,
    transverse_field: float,
    longitudinal_field: float = 0.0,
) -> np.ndarray:
    """Compute the energy density directly from a classical statevector."""
    exp_z = _z_expectations_from_state(state, num_sites)
    exp_x = _x_expectations_from_state(state, num_sites)
    exp_zz = _zz_expectations_from_state(state, num_sites)
    return combine_local_energy_terms(
        exp_z,
        exp_x,
        exp_zz,
        interaction_strength,
        transverse_field,
        longitudinal_field,
    )


__all__ = [
    "combine_local_energy_terms",
    "energy_density_from_counts",
    "energy_density_from_statevector",
    "evaluate_energy_density_with_state",
    "local_energy_density_ops",
]

