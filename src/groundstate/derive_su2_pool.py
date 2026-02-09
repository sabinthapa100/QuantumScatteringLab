
import sys

# Define Pauli algebra
# Representation: (coeff, {site: 'type'})
# type can be 'X', 'Y', 'Z', 'I'

def mul_pauli(p1, p2):
    # p1, p2 are chars 'I','X','Y','Z'
    if p1 == 'I': return 1, p2
    if p2 == 'I': return 1, p1
    if p1 == p2: return 1, 'I'
    if p1 == 'X' and p2 == 'Y': return 1j, 'Z'
    if p1 == 'Y' and p2 == 'Z': return 1j, 'X'
    if p1 == 'Z' and p2 == 'X': return 1j, 'Y'
    if p1 == 'Y' and p2 == 'X': return -1j, 'Z'
    if p1 == 'Z' and p2 == 'Y': return -1j, 'X'
    if p1 == 'X' and p2 == 'Z': return -1j, 'Y'
    return 0, 'I' # Should not happen

def mul_term(t1, t2):
    # t1: (coeff1, dict1)
    # t2: (coeff2, dict2)
    c1, d1 = t1
    c2, d2 = t2
    
    new_coeff = c1 * c2
    new_dict = d1.copy()
    
    for site, p2 in d2.items():
        p1 = new_dict.get(site, 'I')
        phase, res = mul_pauli(p1, p2)
        new_coeff *= phase
        if res == 'I':
            if site in new_dict:
                del new_dict[site]
        else:
            new_dict[site] = res
            
    return new_coeff, new_dict

def commutator(op1, op2):
    # op1, op2 are lists of terms
    # [A, B] = AB - BA
    
    # Compute AB
    res = {}
    
    for t1 in op1:
        for t2 in op2:
            # AB
            c_ab, d_ab = mul_term(t1, t2)
            # BA
            c_ba, d_ba = mul_term(t2, t1)
            
            # AB - BA
            diff = c_ab - c_ba
            
            if abs(diff) > 1e-10:
                # Convert dict to frozen set for hashing
                key = frozenset(d_ab.items())
                res[key] = res.get(key, 0) + diff

    # Format result
    final_terms = []
    for k, v in res.items():
        if abs(v) > 1e-10:
            final_terms.append((v, dict(k)))
            
    return final_terms

def to_string(terms):
    if not terms:
        return "0"
    s = []
    for coeff, d in terms:
        # Sort by site
        sites = sorted(d.keys())
        term_s = ""
        for site in sites:
            term_s += f"{d[site]}_{site} "
        s.append(f"{term_s.strip()}")
    # unique terms
    return list(set(s))

# Define Hamiltonian terms (implicitly translationally invariant, so we just calculate for a central range)
# H_E = J Σ Z_i Z_{i+1} + h_z Σ Z_i
# H_M ~ Σ X_i + ...
# We want to generate the pool from commutators with H_E and H_M.
# Actually, ADAPT-VQE typically builds the pool from [H, A_k] where A_k are "base" operators like X_i, or from the gradients.
# Usually we take the pool to be "operators that appear in the commutators [H, O_current]".
# Start with fundamental magnetic operators (since H_E is diagonal).
# Let's compute [H_E, H_M terms] to see what generated operators (anti-hermitian) look like.

# Definitions for N sites
N = 5 # Small window sufficient to see local structure

# H_E terms (local)
H_E = []
# Z_i Z_{i+1} at site 2
H_E.append((1.0, {2: 'Z', 3: 'Z'}))
H_E.append((1.0, {1: 'Z', 2: 'Z'}))
H_E.append((1.0, {3: 'Z', 4: 'Z'}))
# Z_i at site 2
H_E.append((1.0, {2: 'Z'}))

# H_M terms (local)
# X_i
H_M_X = (1.0, {2: 'X'})
# Z_{i-1} X_i
H_M_ZX = (1.0, {1: 'Z', 2: 'X'})
# X_i Z_{i+1}
H_M_XZ = (1.0, {2: 'X', 3: 'Z'})
# Z_{i-1} X_i Z_{i+1}
H_M_ZXZ = (1.0, {1: 'Z', 2: 'X', 3: 'Z'})

print("--- Commutators [H_E, H_M terms] ---")

c1 = commutator(H_E, [H_M_X])
print(f"[H_E, X_2]: {to_string(c1)}")

c2 = commutator(H_E, [H_M_ZX])
print(f"[H_E, Z_1 X_2]: {to_string(c2)}")

c3 = commutator(H_E, [H_M_XZ])
print(f"[H_E, X_2 Z_3]: {to_string(c3)}")

c4 = commutator(H_E, [H_M_ZXZ])
print(f"[H_E, Z_1 X_2 Z_3]: {to_string(c4)}")

print("\n--- Higher Order: [H_M, [H_E, H_M]] type checks ---")
# Let's take the result of c1 (Y type) and commute with H_M terms
# Y_2 from c1
op_Y2 = [(1.0, {2: 'Y'})]
c_nested = commutator([H_M_ZXZ], op_Y2)
print(f"[Z_1 X_2 Z_3, Y_2]: {to_string(c_nested)}")

# Y_2 Z_3 from c1/c2
op_Y2Z3 = [(1.0, {2: 'Y', 3: 'Z'})]
c_nested_2 = commutator([H_M_ZX], op_Y2Z3) # [Z_1 X_2, Y_2 Z_3]
print(f"[Z_1 X_2, Y_2 Z_3]: {to_string(c_nested_2)}")

