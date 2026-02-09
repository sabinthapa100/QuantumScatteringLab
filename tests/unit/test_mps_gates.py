import quimb as qu
import quimb.tensor as qtn
import numpy as np

def test_auto_split():
    L = 4
    mps = qtn.MPS_computational_state("0000")
    u = qu.expm(1j * 0.1 * np.kron(qu.pauli('Z'), qu.pauli('Z')))
    
    print("Testing 2-site gate with contract='auto-split-gate'...")
    try:
        mps.gate(u, (0, 1), contract='auto-split-gate', inplace=True)
        print(f"Site 0 tensors: {len(mps.select_tensors('I0'))}")
        print(f"mps[0] type: {type(mps[0])}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_auto_split()
