# Ising MPS Simulation Specs

## 1. Ising Hamiltonian
- Eq. (5) defines the 1D Ising field-theory Hamiltonian \(\hat{H}=-\sum_n \left[\tfrac{1}{2}(\hat{Z}_{n-1}\hat{Z}_n+\hat{Z}_n\hat{Z}_{n+1}) + g_x\hat{X}_n + g_z\hat{Z}_n\right]\), i.e., nearest-neighbor Ising interactions (ferromagnetic \(ZZ\) couplings split equally between left/right neighbors), a transverse field \(g_x \hat{X}\) that drives paramagnetic behavior, and a longitudinal field \(g_z \hat{Z}\) that breaks the \(Z_2\) symmetry.
- The Pauli operators are the conventional single-site \(\hat{X}_n, \hat{Y}_n, \hat{Z}_n\); only \(\hat{X}\) and \(\hat{Z}\) appear explicitly in Eq. (5), while \(\hat{Y}\) enters later through commutators in the ADAPT-VQE operator pool.
- Open boundaries drop the wrap-around \(\hat{Z}_0\hat{Z}_{L-1}\) term (and end sites keep only one \(\tfrac{1}{2}\hat{Z}\hat{Z}\) contribution), whereas periodic boundaries include both neighbors and enable exact momentum blocks; the wavepacket optimization is always done assuming PBCs, then truncated operators are applied when switching to OBC hardware layouts.
- The “energy density” operator is simply \(\hat{H}_n = -\tfrac{1}{2}(\hat{Z}_{n-1}\hat{Z}_n+\hat{Z}_n\hat{Z}_{n+1}) - g_x\hat{X}_n - g_z\hat{Z}_n\); observables subtract the vacuum contribution site-by-site via \(E_n = \langle \psi_{\text{state}}|\hat{H}_n(t)|\psi_{\text{state}}\rangle - \langle \psi_{\text{vac}}|\hat{H}_n(t)|\psi_{\text{vac}}\rangle\).

## 2. MPS Simulation Parameters
- Primary “Appendix D” (Methods Sec. C.3) setup: \(L = 256\) with PBCs, \(g_x = 1.25\), \(g_z = 0.15\), light-particle mass gap \(m_1 \approx 1.59\), heavy particle \(m_2 \approx 2.98\), wavepacket width \(\sigma = 0.13\) (in momentum space), truncated spatial support \(d = 22\), and maximum MPS bond dimension 350 for accurate dynamics.
- Two Gaussian packets are initialized 10 sites apart, momenta \(k_0 \in \{0.18\pi, 0.20\pi, 0.28\pi, 0.32\pi, 0.36\pi\}\); \(k_0 \lesssim 0.22\pi\) yields elastic \(11 \rightarrow 11\), while \(k_0 \gtrsim 0.24\pi\) enters the \(11 \rightarrow 12\) inelastic channel (threshold from Appendix B kinematics).
- Real-time evolution uses second-order Trotterization with ordering \(\{R_X, R_Z, R_{ZZ}, R_X\}\), time step \(\delta t = 1/16\), and typically 30–40 steps to reach \(t \sim 25\); larger \(\delta t\) visibly distorts velocities and washes out the inelastic bump (Appendix D.1).
- Variational circuits use 8 ADAPT-VQE layers (10 for the lowest \(k_0\)) derived from the operator pool in Eq. (29); the same circuit parameters work for both \(\pm k_0\) wavepackets due to reality/time-reversal symmetry.
- Recommended small-size GPU test: \(L = 64\) (or even 32) with PBCs, same couplings \((g_x, g_z) = (1.25, 0.15)\), \(\sigma = 0.13\), \(d = 22\) (trim to 16 if memory requires), wavepackets centered \(20\) sites apart, TEBD/TDVP time step \(\delta t = 1/16\), max bond dimension 200; this reproduces qualitative elastic vs inelastic behavior while keeping tensors light enough for a single GPU.

## 3. Vacuum / Ground State
- The paper prepares the vacuum by running the same ADAPT-VQE circuit used for wavepackets on the trivial product state: \(|\psi_{\text{vac}}\rangle \approx \hat{U}(\vec{\theta}_\star)|0\rangle^{\otimes L}\); Appendix F shows this “wavepacket-derived” vacuum beats a dedicated vacuum-only ADAPT run because the greedy ordering favors improving the local energy density where the wavepacket sits.
- Physically the vacuum corresponds to the near-critical paramagnetic phase of the Ising QFT with finite longitudinal bias; correlations fall off exponentially with correlation length \(\xi \sim 1/m_1 \approx 0.6\) in lattice units, so local patches far from the packets look thermalized to the true ordered-phase vacuum.
- For classical MPS reproductions, a simple starting vacuum is the uniform \(|+\rangle^{\otimes L}\) state (eigenstate of the transverse field), optionally followed by one or two imaginary-time TEBD sweeps using the pure Ising Hamiltonian; this captures most of the transverse order and can be refined variationally if higher fidelity to the ADAPT vacuum is needed.

## 4. Wavepacket Preparation ( \( |W(k_0)\rangle \) )
- Algorithm (Methods B.1 + Appendix C + H): (1) build a truncated Gaussian superposition of single-spin excitations \( |W(k_0)\rangle = \sum_n e^{ik_0 n} e^{-(n-x_0)^2/(2\sigma_x^2)} |2^n\rangle \) using either the constant-depth MCM-FF circuit (Fig. 4) or the deterministic unitary ladder (Fig. 12/Appendix C); (2) apply translationally invariant ADAPT-VQE layers generated from the operator pool \( \{ \hat{Y}, \hat{Y}\hat{Z}+\hat{Z}\hat{Y}, \hat{Z}\hat{Y}\hat{Z}, \hat{Z}\hat{X}\hat{Y}+\hat{Y}\hat{X}\hat{Z}, \hat{Y}\hat{X}+\hat{X}\hat{Y} \} \) to project into the single-particle sector while preserving the initialized momentum amplitudes.
- Numerical parameters: \(d = 22\) sites (successfully suppresses truncation errors for \(\sigma = 0.13\)), packet centers separated by 10 sites with \(x_{0,\mathrm{L/R}} = L/2 \mp 5\), momenta as listed above (elastic vs inelastic probes), and two simultaneous packets (one \(+k_0\) one \(-k_0\)); Appendix H tabulates all ADAPT angles for \(k_0 = 0.18\pi \ldots 0.36\pi\).
- Simpler classical recipe: prepare a Gaussian-weighted single-spin-flip superposition directly in MPS (set amplitudes \(c_n\) explicitly), optionally imprint momentum via site-dependent phases \(e^{ik_0 n}\), and run a short TEBD imaginary-time projection restricted to the single-magnon subspace; this avoids MCM-FF and large operator pools while keeping the envelope, velocity, and dispersion consistent with the paper’s construction.

## 5. Observables
- The reported diagnostic is the vacuum-subtracted energy density \(E_n = \langle \hat{H}_n \rangle_{\text{state}} - \langle \hat{H}_n \rangle_{\text{vac}}\) (Eq. 6): compute \(\langle \hat{Z}_{n-1}\hat{Z}_n \rangle\), \(\langle \hat{Z}_n\hat{Z}_{n+1} \rangle\), \(\langle \hat{X}_n \rangle\), and \(\langle \hat{Z}_n \rangle\) locally, plug into \(\hat{H}_n\), and subtract the same combination measured in the time-evolved vacuum reference simulation.
- In circuits, measure each term separately (basis rotations for \(\hat{X}\)), enforce reflection symmetry \(E_n = E_{L-1-n}\), and optionally apply mitigation factors \(p_{\hat{O}}\) extracted from vacuum-only runs; in MPS, evolve both the two-packet state and the approximate vacuum with identical TEBD settings so that subtracting tensors removes background energy inflations from Trotterization.

## 6. Implementation Checklist
- Ising Hamiltonian builder (supports both PBC and OBC versions of Eq. (5) plus per-site \(\hat{H}_n\)).
- Real-time MPS evolution (second-order Trotter/TEBD with configurable \(\delta t\), ordering, and bond growth controls).
- Approximate vacuum preparation workflow (ADAPT-inspired circuit or classical TEBD warm-up, plus vacuum evolution for subtraction).
- Approximate wavepacket initialization (Gaussian \( |W(k_0)\rangle \), truncation to \(d\) sites, phase imprinting, optional ADAPT refinement).
- Energy-density measurement pipeline (local expectation calculator, vacuum subtraction, skewness diagnostic if desired).
- Parameter catalog for elastic vs inelastic runs (set of \((L, d, \sigma, k_0, g_x, g_z, \delta t, n_T)\) tuples, including small-lattice GPU-friendly defaults).



