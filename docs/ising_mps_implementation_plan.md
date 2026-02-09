## Ising MPS Implementation Plan

Reference specs: `@ising_mps_specs.md`.

### 1. Proposed Python Package Layout

- `src/`
  - `__init__.py`
  - `ising_engine.py`
  - `vacuum_state.py`
  - `wavepacket_state.py`
  - `scattering_sim.py`
  - `observables.py`
  - `config.py` (central place for lattice size, couplings, evolution parameters, backend toggles)
- `tests/`
  - `test_ising_engine.py`
  - `test_vacuum_state.py`
  - `test_wavepacket_state.py`
  - `test_scattering_sim.py`
  - `test_observables.py`

### 2. Modules, Classes, and Key Functions

#### `ising_engine.py`

- `class IsingMPSEngine`
  - `__init__(self, n_sites: int, mass: float, coupling: float, field: float, dt: float, trotter_steps: int, backend_cfg: BackendConfig)`  
    Stores lattice/problem params, builds common Pauli operator cache, and records backend settings (`shots`, `precision`, GPU flags).
  - `build_hamiltonian(self) -> SparsePauliOp | list[Operator]`  
    Generates transverse-field Ising Hamiltonian in the convention from the specs; returns both diagonal and interaction terms for reuse.
  - `build_evo_gate(self) -> QuantumCircuit`  
    Produces a first-order (extendable to second-order) Trotterized circuit with `n_sites` qubits using Pauli rotations; reusable per step.
  - `evolve_circuit(self, init_circ: QuantumCircuit, n_steps: int | None = None) -> QuantumCircuit`  
    Appends the evolution block `n_steps` (default `trotter_steps`) times to the initial circuit/state.
  - `run_mps(self, circuit: QuantumCircuit, observables: dict[str, SparsePauliOp], shots: int | None = None) -> dict[str, np.ndarray]`  
    Executes on Qiskit Aer `AerSimulator(method="matrix_product_state", device="GPU")`, returns expectation values/time traces needed downstream.
  - `configure_backend(self, **overrides)`  
    Convenience method to swap Aer settings (precision, blocking) without rebuilding the class.

Support dataclasses:
- `@dataclass BackendConfig` inside `config.py` with fields `simulator="aer"`, `method="matrix_product_state"`, `device="GPU"`, `max_bond_dim`, `seed`, etc.

#### `vacuum_state.py`

- `class VacuumPreparer`
  - `__init__(self, engine: IsingMPSEngine, variational_params: VacuumParams)`  
    Holds reference to engine for operator reuse; parameter dataclass includes ADAPT depth, rotation angles, and symmetry flags.
  - `build_reference_state(self) -> QuantumCircuit`  
    Returns |0...0⟩ or a product state matching the weak/strong coupling branch discussed in the specs.
  - `build_vacuum_circuit(self) -> QuantumCircuit`  
    Implements the approximate ADAPT-like circuit (few layers of alternating ZZ / X rotations) to approximate `|Ω⟩`.
  - `prepare_statevector(self) -> np.ndarray`  
    Optionally returns the vector (via Aer statevector snapshot) for debugging/validation.

#### `wavepacket_state.py`

- `class WavepacketBuilder`
  - `__init__(self, engine: IsingMPSEngine, packet_specs: list[WavepacketSpec], vacuum: QuantumCircuit)`  
    Each `WavepacketSpec` (dataclass) captures center site, momentum `k0`, width, amplitude, and particle number (1 or 2).
  - `add_single_packet(self, circ: QuantumCircuit, spec: WavepacketSpec) -> QuantumCircuit`  
    Implements the simple approximate method from the specs: apply localized rotations and phase shifts that approximate discrete Fourier modes.
  - `add_two_packet(self, circ: QuantumCircuit, left_spec: WavepacketSpec, right_spec: WavepacketSpec) -> QuantumCircuit`  
    Guarantees packets are separated spatially; returns updated circuit.
  - `build_initial_state(self) -> QuantumCircuit`  
    Clones the vacuum circuit and sequences packet insertions.

#### `scattering_sim.py`

- `class ScatteringSim`
  - `__init__(self, engine: IsingMPSEngine, vacuum_builder: VacuumPreparer, wp_builder_cls: type[WavepacketBuilder], observables: list[str], measure_times: np.ndarray)`  
    Wires together subsystems; `observables` names map to helper constructors in `observables.py`.
  - `prepare_initial_circuit(self, wp_specs: list[WavepacketSpec]) -> QuantumCircuit`  
    Builds vacuum, injects packets.
  - `run(self, wp_specs: list[WavepacketSpec], total_steps: int | None = None) -> ScatteringResult`  
    Calls `IsingMPSEngine.evolve_circuit` and `run_mps`, collects expectation values over `measure_times`. Returns a dataclass containing time grid, spatial grid, and arrays (e.g., `energy_density[t, x]`).
  - `analyze(self, result: ScatteringResult) -> dict[str, np.ndarray]`  
    Optional post-processing (momentum distributions, inelasticity metrics).

Support dataclasses:
- `ScatteringResult` with `time_axis`, `space_axis`, `observables`.

#### `observables.py`

- Helper constructors:  
  - `energy_density_ops(n_sites: int, mass: float, coupling: float) -> dict[int, SparsePauliOp]`  
    Local energy density terms aligned with specs (on-site + nearest-neighbor).
  - `magnetization_ops(n_sites: int) -> SparsePauliOp`.
- `process_expectations(raw_expectations: dict[str, np.ndarray], normalization: float) -> dict[str, np.ndarray]`  
  Converts expectation values into physical units (divide by lattice spacing, energy rescaling in specs).
- `measure_energy_density(engine: IsingMPSEngine, circuit: QuantumCircuit) -> np.ndarray`  
  Convenience for single-shot diagnostic runs.

### 3. Backend Strategy

- Baseline implementation targets **Qiskit Aer** with `method="matrix_product_state"` and `device="GPU"` to leverage the RTX 4070. The engine config ensures `precision="double"` and `max_bond_dimension` sized per specs.
- To migrate to **CUDA-Q / cuQuantum**, keep each module’s public API stable. Implement a `CuQuantumMPSEngine` subclass (or adapter) that matches `IsingMPSEngine`’s interface; only swap instantiation inside `config.py`. Higher-level modules (`vacuum_state`, `wavepacket_state`, `scattering_sim`, `observables`) remain unchanged since they consume abstract engine methods.

### 4. Testing Plan (`tests/`)

- `test_ising_engine.py`
  - Validate Hamiltonian matrix is Hermitian and matches small-system analytic values.
  - Confirm Trotterization depth produces circuits with correct gate counts and parameter wires.
  - Run a one-step evolution of |00…0⟩ and assert statevector norm is 1 (using Aer statevector mode).
- `test_vacuum_state.py`
  - Verify `build_vacuum_circuit` produces a normalized state close to expected energy (compare to classical diagonalization for `n_sites ≤ 6`).
  - Check parameter sweeping preserves symmetries specified in the specs (e.g., parity).
- `test_wavepacket_state.py`
  - Inspect real-space probability distribution to ensure localization near target centers.
  - Fourier transform the single-packet state to confirm peak near `k0`.
- `test_scattering_sim.py`
  - Smoke test assembling vacuum + two packets and running a few time steps without errors.
  - Compare energy density traces between symmetric packet setups to ensure conservation (within tolerance).
- `test_observables.py`
  - Ensure local energy density operators sum to the global Hamiltonian.
  - Check `process_expectations` respects normalization and tolerates batch inputs.

This plan should allow a follow-up agent to implement the modules incrementally while keeping the MPS backend swappable and ensuring physics validation at each layer.

