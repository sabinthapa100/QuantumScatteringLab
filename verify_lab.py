"""
Automated System Verification Script
Checks all core components: Backend, Model, and Integration.
"""

import sys
import os
import subprocess

def run_test(name, command):
    print(f"\n--- Testing {name} ---")
    venv_python = ".venv/bin/python3"
    venv_pytest = ".venv/bin/pytest"
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    # Prepend venv if it exists
    if os.path.exists(venv_pytest) and "pytest" in command:
        command = command.replace("pytest", venv_pytest)
    elif os.path.exists(venv_python) and "python3" in command:
        command = command.replace("python3", venv_python)
        
    try:
        subprocess.run(command, shell=True, check=True, env=env)
        print(f"✅ {name} PASSED")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {name} FAILED")
        return False

def main():
    print("⚛️  QUANTUM SCATTERING LAB: SYSTEM CHECK\n")
    
    overall_pass = True
    
    # 1. Unit Tests
    overall_pass &= run_test("Model Integrity", "pytest tests/unit/test_models.py")
    overall_pass &= run_test("MPS Backend Logic", "pytest tests/unit/test_mps_backend.py")
    
    # 2. Integration Tests
    overall_pass &= run_test("Time Evolution Accuracy", "pytest tests/integration/test_backends.py")
    
    # 3. GUI Data Generation
    overall_pass &= run_test("GUI Data Pipeline", "python3 examples/scripts/generate_gui_data.py")
    
    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    if overall_pass:
        print("ALL SYSTEMS OPERATIONAL. You are ready for research! ✨")
    else:
        print("SOME TESTS FAILED. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
