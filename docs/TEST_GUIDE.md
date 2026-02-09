# Project Testing Guide

Follow these steps to ensure the Quantum Scattering Lab is functioning correctly after any changes.

## 1. Unit Tests (Model & Backend Logic)
Run the core physics tests to verify Hamiltonian construction and operator application.
```bash
# Run all unit tests
pytest tests/unit

# Specific tests
pytest tests/unit/test_ising_1d.py
pytest tests/unit/test_su2.py
pytest tests/unit/test_mps_backend.py
```

## 2. Integration Tests (Full Evolution)
Verify that the backend can evolve a state and produce correct expectation values over time.
```bash
pytest tests/integration/test_backends.py
```

## 3. Performance Benchmarks
Measure the speed and scalability of the simulation.
```bash
python3 tests/performance/benchmark_optimization.py
```

## 4. Manual Dashboard Verification
Test the interactive GUI to ensure the frontend-backend communication is stable.
1. Start the lab: `python3 run_lab.py`
2. Open `http://localhost:5173`
3. Select "1D Ising Model"
4. Adjust $g_x$ and $g_z$ sliders.
5. Click **Run Simulation**.
6. Verify that the Atom Logo ⚛️ is rotating and the Heatmap populates with data.

---
### Correcting Discrepancies
If the simulation fails with a Pydantic validation error (e.g., "Field required"), ensure that the frontend state names in `App.jsx` exactly match the `BaseModel` in `server.py`.
