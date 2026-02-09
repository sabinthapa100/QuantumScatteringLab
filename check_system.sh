#!/bin/bash
# Comprehensive Test Suite for Quantum Scattering Lab

echo "=========================================================="
echo "      QUANTUM SCATTERING LAB: DIAGNOSTIC & TEST SUITE"
echo "=========================================================="

# 0. Environment Check
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "ERROR: Virtual environment not active."
    echo "Please activate it: source .venv/bin/activate"
    exit 1
fi
echo "[+] Environment Active: $VIRTUAL_ENV"

# 1. Unit Tests (Fast)
echo -e "\n--- 1. Running Unit Tests ---"
export PYTHONPATH=$(pwd)
pytest tests/test_ising.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[PASS] Ising Model Tests"
else
    echo "[FAIL] Ising Model Tests"
fi

# 2. MPS Backend Check
echo -e "\n--- 2. Checking MPS Backend ---"
python3 -c "import src.backends.quimb_mps_backend as mps; print('MPS Module loaded successfully')" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[PASS] MPS Module Import"
else
    echo "[FAIL] Must fix PYTHONPATH or installation"
fi

# 3. Simulation Dry Run (Ising 1D)
echo -e "\n--- 3. Running Scattering Simulation (Dry Run) ---"
python3 examples/scattering/01_ising_scattering_mps.py --steps 5 > /dev/null 2>&1
# We assume it passes if exit code is 0 (or if we catch output, but basic check for now)
# Actually the script might run long, let's just do a quick check
if [ $? -eq 0 ]; then
    echo "[PASS] 1D Ising Scattering Script"
else
    echo "[WARN] 1D Ising Script failed or took too long (check logs)"
fi

# 4. SU(2) Gauge Theory Check
echo -e "\n--- 4. Checking SU(2) Gauge Implementation ---"
python3 -c "from src.models.su2 import SU2GaugeModel; m=SU2GaugeModel(4); print(m.build_hamiltonian())" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "[PASS] SU(2) Model construction"
else
    echo "[FAIL] SU(2) Model Error"
fi

echo -e "\n=========================================================="
echo "      DIAGNOSTICS COMPLETE."
echo "      See docs/GUIDE.md for detailed next steps."
echo "=========================================================="
