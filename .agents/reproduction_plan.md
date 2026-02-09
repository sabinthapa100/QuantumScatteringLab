# Reproduction Plan

## Hamiltonian (Eq. 5)
The paper defines the lattice Hamiltonian as
\[
\hat{H} = - \sum_{n=0}^{L-1} \left[\tfrac{1}{2}\left(\hat{Z}_{n-1}\hat{Z}_{n} + \hat{Z}_{n}\hat{Z}_{n+1}\right) + g_x \hat{X}_{n} + g_z \hat{Z}_{n}\right],
\]
with periodic boundary conditions implied unless otherwise stated. The interaction term couples neighboring sites through \(\hat{Z}\hat{Z}\), i.e., the longitudinal Ising interaction is in the \(Z\) basis, while the transverse field couples through \(\hat{X}\).

For consistency with standard TFIM notation, we can take \(J=1\) (the prefactor of each nearest-neighbor \(\hat{Z}\hat{Z}\) term after accounting for the double counting in the sum), \(h = g_x\), and \(h_z = g_z\).

## Simulation Parameters
- **Elastic regime:** Uses the same couplings as the inelastic case—\(J=1\), \(h = g_x = 1.25\), and \(h_z = g_z = 0.15\)—but launches lower-energy wavepackets with momentum \(k_0 = 0.18\pi\) (discussed alongside Fig. 14 in the Methods section on MPS scattering), remaining below the \(11 \to 12\) production threshold.
- **Inelastic regime:** Keeps \(J=1\), \(h = 1.25\), \(h_z = 0.15\), and excites higher-momentum packets, e.g. \(k_0 = 0.32\pi\), yielding total energy \(E_{\text{tot}}/E_{\text{thr}} \approx 1.2\) and showing clear \(11 \to 12\) signals (main text Sec. 3 and Fig. 3).
- **Shared settings:** Lattice sizes \(L=104\) (hardware) and \(L=256\) (MPS reference), Trotter step \(\delta t = 0.55\) on hardware vs. \(1/16\) for high-accuracy MPS runs, and wavepacket width parameter \(\sigma = 0.13\) with support \(d \approx 21\text{–}22\) sites.

## Initial State (Wavepacket / W-state)
Wavepackets start from the generalized W-state ansatz
\[
|W(k_0)\rangle = \sum_{n=0}^{d-1} e^{i \phi_n} c_n |2^n\rangle,
\]
where \(c_n\) follow a Gaussian envelope centered at \(x_0\) with spread \(\sigma\), and \(\phi_n = k_0 n\) encodes momentum. Two such packets with momenta \(\pm k_0\) are prepared, separated by at least two vacant regions and projected into low-energy single-particle eigenstates using translationally invariant ADAPT-VQE circuits (Methods §§\ref{sec:WPsummary}, \ref{sec:IsingStatevector}, \ref{sec:qcirc_scatt}).

For the scattering shots highlighted in the figures:
- Momentum: \(k_0 = 0.32\pi\) (inelastic) and \(k_0 = 0.18\pi\) (elastic benchmark).
- Width: \(\sigma = 0.13\), leading to spatial support \(d \gtrsim 21\) qubits per packet.
- Positioning: Packets are initialized about 10 sites apart on the MPS lattice (or ~30 sites from each boundary for \(L=104\) with OBCs).

## Observables
The key observable is the vacuum-subtracted energy density,
\[
E_n(t) = \langle \psi_{\text{2wp}} | \hat{H}_n(t) | \psi_{\text{2wp}} \rangle - \langle \psi_{\text{vac}} | \hat{H}_n(t) | \psi_{\text{vac}} \rangle,
\]
with local term
\[
\hat{H}_n = -\tfrac{1}{2}\left(\hat{Z}_{n-1}\hat{Z}_n + \hat{Z}_n\hat{Z}_{n+1}\right) - g_x \hat{X}_n - g_z \hat{Z}_n.
\]
These local energies form the “light cone” heatmaps distinguishing elastic versus inelastic scattering (Figures 2 & 3). Skewness of \(E_n\) in each half-lattice further diagnoses heavy-particle production (Methods §\ref{sec:skew}).

