import quimb as qu
import quimb.tensor as qtn

def test_apply_mpo():
    L = 4
    mps = qtn.MPS_computational_state("0000")
    mpo = qtn.MPO_identity(L)
    
    print("Testing mps.apply(mpo)...")
    try:
        res = mpo.apply(mps)
        print(f"Result type: {type(res)}")
        # Check if we can compress
        res.compress(max_bond=2)
        print("Compress success")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_apply_mpo()
