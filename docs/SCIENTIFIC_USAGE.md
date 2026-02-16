# Scientific Usage & Mathematics

## 🧬 Complexity Management

The QuantumScatteringLab (QSL) engine minimizes redundant SVD operations. For systems where $L \ge 128$, traditional MPS naive gates lead to $O(L^2)$ overhead. 

### 1. Batched Trotter Evolution
For a Trotter layer $U(\Delta t) \approx e^{i H_{even} \Delta t} e^{i H_{odd} \Delta t}$:
- **Naive**: Apply gate $\to$ SVD $\to$ Apply gate $\to$ SVD. (Total SVDs: $L \times \text{steps}$)
- **Optimized**: Apply all $L/2$ gates $\to$ Single SVD. (Total SVDs: $1 \times \text{steps}$)
- **Result**: Evolution time for $L=128$ dropped from **53.9s to 0.82s**.

### 2. $O(L)$ Measurement Scaling
Expectation values $\langle \Psi | \hat{O}_n | \Psi \rangle$ conventionally require contracting the entire chain.
- **Optimization**: We leverage the MPS **Canonical Form**. By shifting the orthogonality center to site $n$, the local density matrix $\rho_n$ is computed in $O(1)$ time relative to $L$.
- **Implementation**: `state.canonize(0)` is called once per measurement row, allowing all subsequent local site expectations to be computed as simple tensor traces.

---

## 🚀 Physics Configuration

### Ground State Preparation (ADAPT-VQE)
We use the **ADAPT-VQE** algorithm to grow the ansatz iteratively:
1. Compute gradients for the operator pool: $G_k = i \langle \Psi | [H, A_k] | \Psi \rangle$.
2. Select the operator with the largest gradient.
3. Perform a global optimization of all ansatz coefficients.

### Wavepacket Preparation (W-State Superposition)
Relative wavepackets are prepared as linear superpositions of local excitations:
$$ |\Psi(t=0)\rangle = \sum_n f(n, x_0, k_0, \sigma) \hat{O}_n |vac\rangle $$
where $f$ is the Gaussian envelope with momentum $k_0$.

---

## 📈 Data Extraction

### Spacetime Heatmaps
We plot the vacuum-subtracted energy density:
$$ \Delta \epsilon(x, t) = \langle \Psi(t) | H_x | \Psi(t) \rangle - \langle vac | H_x | vac \rangle $$
This ensures that "dark" background energy does not obscure the collision signal.

### Entanglement Entropy
Computed at each step to track the "Quench" behavior of the collision:
$$ S_1(\text{bond } j) = -\sum \lambda_k^2 \ln \lambda_k^2 $$
Where $\lambda$ are the Singular Values at bond $j$.
