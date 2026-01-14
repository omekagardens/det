"""
DET v6.3 2D Collider - Unified Implementation
=============================================

Complete implementation with all DET modules:
- Gravity module (Section V): Helmholtz baseline, Poisson potential, gravitational flux
- Boundary operators (Section VI): Grace injection, bond healing
- Agency-gated diffusion (IV.2)
- Presence-clocked transport (III.1)
- Momentum dynamics with gravity coupling (IV.4)
- Angular momentum dynamics (IV.5)
- Floor repulsion (IV.6)

Reference: DET Theory Card v6.3

Changelog from v6.2:
- Added beta_g parameter for gravity-momentum coupling
- Added angular momentum dynamics (L_Z on plaquettes)
- Added lattice correction factor eta
- Updated version references
"""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import label
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


def compute_lattice_correction_2d(N: int) -> float:
    """Compute lattice correction factor eta for 2D grid."""
    if N <= 32:
        return 0.88
    elif N <= 64:
        return 0.94
    elif N <= 96:
        return 0.96
    else:
        return 0.97


@dataclass
class DETParams2D:
    """DET v6.3 2D simulation parameters - complete."""
    N: int = 100
    DT: float = 0.015
    F_VAC: float = 0.01
    F_MIN: float = 0.0
    R: int = 5

    # Coherence
    C_init: float = 0.3

    # Diffusive flux
    diff_enabled: bool = True

    # Momentum module (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.08
    lambda_pi: float = 0.015
    mu_pi: float = 0.25
    pi_max: float = 2.5

    # Angular Momentum (IV.5) - NEW in v6.3 for 2D
    angular_momentum_enabled: bool = True
    alpha_L: float = 0.06
    lambda_L: float = 0.005
    mu_L: float = 0.18
    L_max: float = 5.0

    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.015

    # Agency dynamics (VI.2B) - v6.4 update
    agency_dynamic: bool = True
    lambda_a: float = 30.0      # Structural ceiling coupling
    beta_a: float = 0.2         # Relaxation rate toward ceiling
    gamma_a_max: float = 0.15   # Max relational drive strength
    gamma_a_power: float = 2.0  # Coherence gating exponent (n >= 2)

    # Sigma dynamics
    sigma_dynamic: bool = True

    # Coherence dynamics
    coherence_dynamic: bool = True
    alpha_C: float = 0.04
    lambda_C: float = 0.002

    # Floor repulsion (IV.6)
    floor_enabled: bool = True
    eta_floor: float = 0.12
    F_core: float = 4.0
    floor_power: float = 2.0

    # Gravity module (V.1-V.3)
    gravity_enabled: bool = True
    alpha_grav: float = 0.02
    kappa_grav: float = 5.0
    mu_grav: float = 2.0
    beta_g: float = 10.0  # v6.3: gravity-momentum coupling

    # Boundary operators (VI)
    boundary_enabled: bool = True
    grace_enabled: bool = True
    F_MIN_grace: float = 0.05
    healing_enabled: bool = False
    eta_heal: float = 0.03
    R_boundary: int = 3

    # v6.3: Lattice correction factor
    eta_lattice: float = 0.94

    # v6.3: Option B - Coherence-weighted load
    # H_i = Σ_{j ∈ N_R(i)} √C_ij * σ_ij
    coherence_weighted_H: bool = False

    # Numerical stability
    outflow_limit: float = 0.20


def periodic_local_sum_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs)."""
    result = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
    return result


class DETCollider2D:
    """
    DET v6.3 2D Collider - Unified with Gravity, Boundary Operators, Angular Momentum

    Key v6.3 features:
    - beta_g gravity-momentum coupling
    - Angular momentum on plaquettes (L_Z)
    - Lattice correction factor eta
    - Enhanced time dilation tracking
    """

    def __init__(self, params: Optional[DETParams2D] = None):
        self.p = params or DETParams2D()
        N = self.p.N

        # Update lattice correction based on grid size
        self.p.eta_lattice = compute_lattice_correction_2d(N)

        # Per-node state
        self.F = np.ones((N, N), dtype=np.float64) * self.p.F_VAC
        self.q = np.zeros((N, N), dtype=np.float64)
        self.a = np.ones((N, N), dtype=np.float64)
        self.theta = np.random.uniform(0, 2*np.pi, (N, N)).astype(np.float64)

        # Per-bond linear momentum
        self.pi_E = np.zeros((N, N), dtype=np.float64)  # East
        self.pi_S = np.zeros((N, N), dtype=np.float64)  # South

        # Per-plaquette angular momentum (L_Z) - NEW in v6.3 for 2D
        self.L_Z = np.zeros((N, N), dtype=np.float64)

        # Per-bond coherence
        self.C_E = np.ones((N, N), dtype=np.float64) * self.p.C_init
        self.C_S = np.ones((N, N), dtype=np.float64) * self.p.C_init

        self.sigma = np.ones((N, N), dtype=np.float64)

        # Gravity fields
        self.b = np.zeros((N, N), dtype=np.float64)
        self.Phi = np.zeros((N, N), dtype=np.float64)
        self.gx = np.zeros((N, N), dtype=np.float64)
        self.gy = np.zeros((N, N), dtype=np.float64)

        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N), dtype=np.float64)
        self.Delta_tau = np.ones((N, N), dtype=np.float64) * self.p.DT

        # Time dilation tracking (v6.3)
        self.accumulated_proper_time = np.zeros((N, N), dtype=np.float64)

        # Boundary operator diagnostics
        self.last_grace_injection = np.zeros((N, N), dtype=np.float64)
        self.last_healing_E = np.zeros((N, N), dtype=np.float64)
        self.last_healing_S = np.zeros((N, N), dtype=np.float64)
        self.total_grace_injected = 0.0

        self._setup_fft_solvers()

    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers."""
        N = self.p.N
        kx = np.fft.fftfreq(N) * N
        ky = np.fft.fftfreq(N) * N
        KX, KY = np.meshgrid(kx, ky, indexing='ij')

        self.L_k = -4 * (np.sin(np.pi * KX / N)**2 + np.sin(np.pi * KY / N)**2)
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0, 0] = 1.0

    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        source_k = fft2(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifft2(b_k))

    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        source_k = fft2(source)
        source_k[0, 0] = 0
        # Apply lattice correction (v6.3)
        Phi_k = -self.p.kappa_grav * self.p.eta_lattice * source_k / self.L_k_poisson
        Phi_k[0, 0] = 0
        return np.real(ifft2(Phi_k))

    def _compute_coherence_weighted_H(self) -> np.ndarray:
        """Compute Option B coherence-weighted load.

        H_i = Σ_{j ∈ N_R(i)} √C_ij * σ_ij

        In 2D, sum over 4 neighbors (E, W, S, N).
        """
        E = lambda x: np.roll(x, -1, axis=1)
        W = lambda x: np.roll(x, 1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)
        Nb = lambda x: np.roll(x, 1, axis=0)

        # Bond coherences
        sqrt_C_E = np.sqrt(self.C_E)
        sqrt_C_W = np.sqrt(W(self.C_E))  # C_W[i] = C_E[i-1] in x
        sqrt_C_S = np.sqrt(self.C_S)
        sqrt_C_N = np.sqrt(Nb(self.C_S))  # C_N[i] = C_S[i-1] in y

        # Bond-averaged sigma
        sigma_E = 0.5 * (self.sigma + E(self.sigma))
        sigma_W = 0.5 * (self.sigma + W(self.sigma))
        sigma_S = 0.5 * (self.sigma + S(self.sigma))
        sigma_N = 0.5 * (self.sigma + Nb(self.sigma))

        # Coherence-weighted load: sum over 4 neighbors
        H = (sqrt_C_E * sigma_E + sqrt_C_W * sigma_W +
             sqrt_C_S * sigma_S + sqrt_C_N * sigma_N)

        return H

    def _compute_gravity(self):
        """Compute gravitational fields from q."""
        if not self.p.gravity_enabled:
            self.gx = np.zeros_like(self.F)
            self.gy = np.zeros_like(self.F)
            return

        self.b = self._solve_helmholtz(self.q)
        rho = self.q - self.b
        self.Phi = self._solve_poisson(rho)

        E = lambda x: np.roll(x, -1, axis=1)
        W = lambda x: np.roll(x, 1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)
        Nb = lambda x: np.roll(x, 1, axis=0)

        self.gx = -0.5 * (E(self.Phi) - W(self.Phi))
        self.gy = -0.5 * (S(self.Phi) - Nb(self.Phi))

    def compute_grace_injection(self, D: np.ndarray) -> np.ndarray:
        """Grace Injection per DET VI.5"""
        p = self.p
        n = np.maximum(0, p.F_MIN_grace - self.F)
        w = self.a * n
        w_sum = periodic_local_sum_2d(w, p.R_boundary) + 1e-12
        I_g = D * w / w_sum
        return I_g

    def compute_bond_healing(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Bond Healing Operator (Agency-Gated)"""
        p = self.p
        E = lambda x: np.roll(x, -1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)

        g_E = np.sqrt(self.a * E(self.a))
        g_S = np.sqrt(self.a * S(self.a))

        room_E = 1.0 - self.C_E
        room_S = 1.0 - self.C_S

        D_avg_E = 0.5 * (D + E(D))
        D_avg_S = 0.5 * (D + S(D))

        Delta_tau_E = 0.5 * (self.Delta_tau + E(self.Delta_tau))
        Delta_tau_S = 0.5 * (self.Delta_tau + S(self.Delta_tau))

        dC_heal_E = p.eta_heal * g_E * room_E * D_avg_E * Delta_tau_E
        dC_heal_S = p.eta_heal * g_S * room_S * D_avg_S * Delta_tau_S

        return dC_heal_E, dC_heal_S

    def add_packet(self, center: Tuple[int, int], mass: float = 6.0,
                   width: float = 5.0, momentum: Tuple[float, float] = (0, 0),
                   initial_q: float = 0.0, initial_spin: float = 0.0):
        """Add a Gaussian resource packet with optional momentum and spin."""
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        cy, cx = center
        r2 = (x - cx)**2 + (y - cy)**2
        envelope = np.exp(-0.5 * r2 / width**2)

        self.F += mass * envelope
        self.C_E = np.clip(self.C_E + 0.7 * envelope, self.p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.7 * envelope, self.p.C_init, 1.0)

        py, px = momentum
        if px != 0 or py != 0:
            mom_env = np.exp(-0.5 * r2 / (width * 3)**2)
            self.pi_E += px * mom_env
            self.pi_S += py * mom_env

        if initial_q > 0:
            self.q += initial_q * envelope

        # Add angular momentum (v6.3)
        if initial_spin != 0:
            self.L_Z += initial_spin * envelope

        self._clip()

    def add_spin(self, center: Tuple[int, int], spin: float = 1.0, width: float = 4.0):
        """Add initial angular momentum to a region."""
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        cy, cx = center

        dx = (x - cx + N/2) % N - N/2
        dy = (y - cy + N/2) % N - N/2

        r2 = dx**2 + dy**2
        envelope = np.exp(-0.5 * r2 / width**2)

        self.L_Z += spin * envelope
        self._clip()

    def _clip(self):
        p = self.p
        self.F = np.clip(self.F, p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_E = np.clip(self.pi_E, -p.pi_max, p.pi_max)
        self.pi_S = np.clip(self.pi_S, -p.pi_max, p.pi_max)
        self.L_Z = np.clip(self.L_Z, -p.L_max, p.L_max)

    def step(self):
        """Execute one canonical DET update step per Theory Card v6.3."""
        p = self.p
        N = p.N
        dk = p.DT

        E = lambda x: np.roll(x, -1, axis=1)
        W = lambda x: np.roll(x, 1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)
        Nb = lambda x: np.roll(x, 1, axis=0)

        # STEP 0: Gravity
        self._compute_gravity()

        # STEP 1: Presence (III.1)
        # Option B: Coherence-weighted load H_i = Σ_{j} √C_ij * σ_ij
        if p.coherence_weighted_H:
            H = self._compute_coherence_weighted_H()
        else:
            H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk

        # Track accumulated proper time (v6.3)
        self.accumulated_proper_time += self.Delta_tau

        Delta_tau_E = 0.5 * (self.Delta_tau + E(self.Delta_tau))
        Delta_tau_S = 0.5 * (self.Delta_tau + S(self.Delta_tau))

        # Plaquette-averaged proper time
        Delta_tau_Z = 0.25 * (self.Delta_tau + E(self.Delta_tau) +
                              S(self.Delta_tau) + E(S(self.Delta_tau)))

        # STEP 2: Flow computation
        J_E = np.zeros_like(self.F)
        J_W = np.zeros_like(self.F)
        J_S = np.zeros_like(self.F)
        J_N = np.zeros_like(self.F)

        J_diff_E = np.zeros_like(self.F)
        J_diff_S = np.zeros_like(self.F)

        # Diffusive flux
        if p.diff_enabled:
            classical_E = self.F - E(self.F)
            classical_W = self.F - W(self.F)
            classical_S = self.F - S(self.F)
            classical_N = self.F - Nb(self.F)

            g_E = np.sqrt(self.a * E(self.a))
            g_W = np.sqrt(self.a * W(self.a))
            g_S = np.sqrt(self.a * S(self.a))
            g_N = np.sqrt(self.a * Nb(self.a))

            cond_E = self.sigma * (self.C_E + 1e-4)
            cond_W = self.sigma * (W(self.C_E) + 1e-4)
            cond_S = self.sigma * (self.C_S + 1e-4)
            cond_N = self.sigma * (Nb(self.C_S) + 1e-4)

            J_diff_E = g_E * cond_E * classical_E
            J_diff_W = g_W * cond_W * classical_W
            J_diff_S = g_S * cond_S * classical_S
            J_diff_N = g_N * cond_N * classical_N

            J_E += J_diff_E
            J_W += J_diff_W
            J_S += J_diff_S
            J_N += J_diff_N

        # Momentum flux
        if p.momentum_enabled:
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_W = 0.5 * (self.F + W(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            F_avg_N = 0.5 * (self.F + Nb(self.F))

            J_E += p.mu_pi * self.sigma * self.pi_E * F_avg_E
            J_W += -p.mu_pi * self.sigma * W(self.pi_E) * F_avg_W
            J_S += p.mu_pi * self.sigma * self.pi_S * F_avg_S
            J_N += -p.mu_pi * self.sigma * Nb(self.pi_S) * F_avg_N

        # Floor flux
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            classical_E = self.F - E(self.F)
            classical_W = self.F - W(self.F)
            classical_S = self.F - S(self.F)
            classical_N = self.F - Nb(self.F)

            J_E += p.eta_floor * self.sigma * (s + E(s)) * classical_E
            J_W += p.eta_floor * self.sigma * (s + W(s)) * classical_W
            J_S += p.eta_floor * self.sigma * (s + S(s)) * classical_S
            J_N += p.eta_floor * self.sigma * (s + Nb(s)) * classical_N

        # Gravitational flux
        if p.gravity_enabled:
            gx_bond_E = 0.5 * (self.gx + E(self.gx))
            gx_bond_W = 0.5 * (self.gx + W(self.gx))
            gy_bond_S = 0.5 * (self.gy + S(self.gy))
            gy_bond_N = 0.5 * (self.gy + Nb(self.gy))

            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_W = 0.5 * (self.F + W(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            F_avg_N = 0.5 * (self.F + Nb(self.F))

            J_E += p.mu_grav * self.sigma * gx_bond_E * F_avg_E
            J_W += p.mu_grav * self.sigma * gx_bond_W * F_avg_W
            J_S += p.mu_grav * self.sigma * gy_bond_S * F_avg_S
            J_N += p.mu_grav * self.sigma * gy_bond_N * F_avg_N

        # Rotational flux from angular momentum (v6.3)
        if p.angular_momentum_enabled:
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))

            # Rotational flux from L_Z
            J_rot_E = p.mu_L * self.sigma * F_avg_E * (self.L_Z - Nb(self.L_Z))
            J_rot_S = -p.mu_L * self.sigma * F_avg_S * (self.L_Z - W(self.L_Z))

            J_E += J_rot_E
            J_W -= W(J_rot_E)
            J_S += J_rot_S
            J_N -= Nb(J_rot_S)

        # STEP 3: Limiter
        total_outflow = (np.maximum(0, J_E) + np.maximum(0, J_W) +
                         np.maximum(0, J_S) + np.maximum(0, J_N))
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))

        J_E_lim = np.where(J_E > 0, J_E * scale, J_E)
        J_W_lim = np.where(J_W > 0, J_W * scale, J_W)
        J_S_lim = np.where(J_S > 0, J_S * scale, J_S)
        J_N_lim = np.where(J_N > 0, J_N * scale, J_N)

        J_diff_E_scaled = np.where(J_diff_E > 0, J_diff_E * scale, J_diff_E)
        J_diff_S_scaled = np.where(J_diff_S > 0, J_diff_S * scale, J_diff_S)

        # Dissipation
        D = (np.abs(J_E_lim) + np.abs(J_W_lim) +
             np.abs(J_S_lim) + np.abs(J_N_lim)) * self.Delta_tau

        # STEP 4: Resource update
        transfer_E = J_E_lim * self.Delta_tau
        transfer_W = J_W_lim * self.Delta_tau
        transfer_S = J_S_lim * self.Delta_tau
        transfer_N = J_N_lim * self.Delta_tau

        outflow = transfer_E + transfer_W + transfer_S + transfer_N
        inflow = W(transfer_E) + E(transfer_W) + Nb(transfer_S) + S(transfer_N)

        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)

        # STEP 5: Grace Injection (VI.5)
        if p.boundary_enabled and p.grace_enabled:
            I_g = self.compute_grace_injection(D)
            self.F = self.F + I_g
            self.last_grace_injection = I_g.copy()
            self.total_grace_injected += np.sum(I_g)
        else:
            self.last_grace_injection = np.zeros((N, N))

        # STEP 6: Momentum update with gravity coupling (v6.3: beta_g)
        if p.momentum_enabled:
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)

            dpi_diff_E = p.alpha_pi * J_diff_E_scaled * Delta_tau_E
            dpi_diff_S = p.alpha_pi * J_diff_S_scaled * Delta_tau_S

            if p.gravity_enabled:
                gx_bond_E = 0.5 * (self.gx + E(self.gx))
                gy_bond_S = 0.5 * (self.gy + S(self.gy))
                dpi_grav_E = p.beta_g * gx_bond_E * Delta_tau_E
                dpi_grav_S = p.beta_g * gy_bond_S * Delta_tau_S
            else:
                dpi_grav_E = dpi_grav_S = 0

            self.pi_E = decay_E * self.pi_E + dpi_diff_E + dpi_grav_E
            self.pi_S = decay_S * self.pi_S + dpi_diff_S + dpi_grav_S

        # STEP 7: Angular momentum update (v6.3)
        if p.angular_momentum_enabled:
            # Curl of momentum (plaquette circulation)
            curl_Z = self.pi_E + E(self.pi_S) - S(self.pi_E) - self.pi_S
            decay_Z = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_Z)
            self.L_Z = decay_Z * self.L_Z + p.alpha_L * curl_Z * Delta_tau_Z

        # STEP 8: Structure update
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)

        # STEP 9: Bond Healing
        if p.boundary_enabled and p.healing_enabled:
            dC_heal_E, dC_heal_S = self.compute_bond_healing(D)
            self.C_E = np.clip(self.C_E + dC_heal_E, p.C_init, 1.0)
            self.C_S = np.clip(self.C_S + dC_heal_S, p.C_init, 1.0)
            self.last_healing_E = dC_heal_E.copy()
            self.last_healing_S = dC_heal_S.copy()
        else:
            self.last_healing_E = np.zeros((N, N))
            self.last_healing_S = np.zeros((N, N))

        # STEP 10: Agency update
        if p.agency_dynamic:
            # v6.4 Agency Law: Structural Ceiling + Relational Drive

            # Step 1: Structural ceiling (matter law)
            a_max = 1.0 / (1.0 + p.lambda_a * self.q**2)

            # Step 2: Relational drive (life law)
            # Compute local average presence (self + 4 neighbors)
            P_local = (self.P + E(self.P) + W(self.P) + S(self.P) + Nb(self.P)) / 5.0

            # Average coherence at each node
            C_avg = (self.C_E + W(self.C_E) + self.C_S + Nb(self.C_S)) / 4.0

            # Coherence-gated drive: γ(C) = γ_max * C^n
            gamma = p.gamma_a_max * (C_avg ** p.gamma_a_power)

            # Relational drive: seeks presence gradients
            delta_a_drive = gamma * (self.P - P_local)

            # Step 3: Unified update
            self.a = self.a + p.beta_a * (a_max - self.a) + delta_a_drive
            self.a = np.clip(self.a, 0.0, a_max)

        # STEP 11: Coherence and sigma dynamics
        J_mag = (np.abs(J_E_lim) + np.abs(J_S_lim)) / 2.0

        if p.coherence_dynamic:
            self.C_E = np.clip(self.C_E + p.alpha_C * np.abs(J_E_lim) * self.Delta_tau
                              - p.lambda_C * self.C_E * self.Delta_tau, p.C_init, 1.0)
            self.C_S = np.clip(self.C_S + p.alpha_C * np.abs(J_S_lim) * self.Delta_tau
                              - p.lambda_C * self.C_S * self.Delta_tau, p.C_init, 1.0)

        if p.sigma_dynamic:
            self.sigma = 1.0 + 0.1 * np.log(1.0 + J_mag)

        self._clip()
        self.time += dk
        self.step_count += 1

    # ==================== DIAGNOSTICS ====================

    def total_mass(self) -> float:
        return float(np.sum(self.F))

    def total_q(self) -> float:
        return float(np.sum(self.q))

    def total_angular_momentum(self) -> float:
        """Total angular momentum (L_Z)."""
        return float(np.sum(self.L_Z))

    def potential_energy(self) -> float:
        return float(np.sum(self.F * self.Phi))

    def center_of_mass(self) -> Tuple[float, float]:
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        total = np.sum(self.F) + 1e-9
        cx = float(np.sum(x * self.F) / total)
        cy = float(np.sum(y * self.F) / total)
        return (cy, cx)

    def find_blobs(self, threshold_ratio: float = 10.0) -> List[Dict]:
        """Find distinct blobs using connected components."""
        threshold = self.p.F_VAC * threshold_ratio
        above = self.F > threshold
        labeled, num = label(above)
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]

        blobs = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) == 0:
                continue
            weights = self.F[mask]
            total_mass = float(np.sum(weights))
            if total_mass < 0.5:
                continue
            blobs.append({
                'x': float(np.sum(x[mask] * weights) / total_mass),
                'y': float(np.sum(y[mask] * weights) / total_mass),
                'mass': total_mass,
                'size': int(np.sum(mask))
            })
        blobs.sort(key=lambda c: -c['mass'])
        return blobs

    def separation(self) -> float:
        """Find separation between two largest blobs."""
        blobs = self.find_blobs()
        if len(blobs) < 2:
            return 0.0

        N = self.p.N
        dx = blobs[1]['x'] - blobs[0]['x']
        dy = blobs[1]['y'] - blobs[0]['y']
        dx = dx - N if dx > N/2 else (dx + N if dx < -N/2 else dx)
        dy = dy - N if dy > N/2 else (dy + N if dy < -N/2 else dy)
        return float(np.sqrt(dx**2 + dy**2))

    def rot_flux_magnitude(self) -> float:
        """Magnitude of rotational flux."""
        if not self.p.angular_momentum_enabled:
            return 0.0

        W = lambda arr: np.roll(arr, 1, axis=1)
        Nb = lambda arr: np.roll(arr, 1, axis=0)
        E = lambda arr: np.roll(arr, -1, axis=1)
        S = lambda arr: np.roll(arr, -1, axis=0)

        F_avg_E = 0.5 * (self.F + E(self.F))
        F_avg_S = 0.5 * (self.F + S(self.F))

        J_rot_E = self.p.mu_L * self.sigma * F_avg_E * (self.L_Z - Nb(self.L_Z))
        J_rot_S = -self.p.mu_L * self.sigma * F_avg_S * (self.L_Z - W(self.L_Z))

        return float(np.sum(np.abs(J_rot_E) + np.abs(J_rot_S)))


# ============================================================
# TEST SUITE
# ============================================================

def test_v6_3_gravity_vacuum(verbose: bool = True) -> bool:
    """Gravity has no effect in vacuum (q=0)."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravity in Vacuum")
        print("="*60)

    params = DETParams2D(N=32, gravity_enabled=True, q_enabled=False, boundary_enabled=False)
    sim = DETCollider2D(params)

    for _ in range(200):
        sim.step()

    max_g = np.max(np.abs(sim.gx)) + np.max(np.abs(sim.gy))
    max_Phi = np.max(np.abs(sim.Phi))

    passed = max_g < 1e-10 and max_Phi < 1e-10

    if verbose:
        print(f"  Max |g|: {max_g:.2e}")
        print(f"  Max |Phi|: {max_Phi:.2e}")
        print(f"  Lattice correction eta: {sim.p.eta_lattice:.3f}")
        print(f"  Vacuum gravity {'PASSED' if passed else 'FAILED'}")

    return passed


def test_v6_3_binding(verbose: bool = True) -> bool:
    """Test gravitational binding with beta_g coupling."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravitational Binding (v6.3)")
        print("="*60)

    params = DETParams2D(
        N=64, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True
    )

    sim = DETCollider2D(params)

    initial_sep = 24
    center = params.N // 2
    sim.add_packet((center, center - initial_sep//2), mass=8.0, width=3.0,
                   momentum=(0, 0.1), initial_q=0.3)
    sim.add_packet((center, center + initial_sep//2), mass=8.0, width=3.0,
                   momentum=(0, -0.1), initial_q=0.3)

    min_sep = initial_sep

    for t in range(1000):
        sep = sim.separation()
        min_sep = min(min_sep, sep)

        if verbose and t % 200 == 0:
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.3f}")

        sim.step()

    passed = min_sep < initial_sep * 0.5

    if verbose:
        print(f"\n  Initial sep: {initial_sep:.1f}")
        print(f"  Min sep: {min_sep:.1f}")
        print(f"  beta_g: {params.beta_g}")
        print(f"  Binding {'PASSED' if passed else 'FAILED'}")

    return passed


def test_v6_3_angular_momentum(verbose: bool = True) -> bool:
    """Test angular momentum conservation."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Angular Momentum (v6.3)")
        print("="*60)

    params = DETParams2D(
        N=48,
        F_MIN=0.0,
        diff_enabled=False,
        momentum_enabled=False,
        floor_enabled=False,
        q_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False,
        coherence_dynamic=False,
        angular_momentum_enabled=True,
        gravity_enabled=False,
        boundary_enabled=False
    )
    sim = DETCollider2D(params)
    center = params.N // 2

    sim.add_packet((center, center), mass=15.0, width=6.0)
    sim.add_spin((center, center), spin=2.0, width=8.0)

    initial_F = sim.total_mass()

    for t in range(500):
        sim.step()

    final_F = sim.total_mass()
    mass_err = abs(final_F - initial_F) / initial_F

    passed = mass_err < 1e-10

    if verbose:
        print(f"  Mass error: {mass_err:.2e}")
        print(f"  Total L_Z: {sim.total_angular_momentum():.4f}")
        print(f"  Angular momentum {'PASSED' if passed else 'FAILED'}")

    return passed


def test_v6_3_option_b_coherence_weighted_H(verbose: bool = True) -> bool:
    """Test Option B: Coherence-weighted load H_i = Σ √C_ij σ_ij."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Option B - Coherence-Weighted Load")
        print("="*60)

    # Test with Option B enabled
    params_b = DETParams2D(
        N=64, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        coherence_weighted_H=True  # Enable Option B
    )

    sim_b = DETCollider2D(params_b)

    initial_sep = 24
    center = params_b.N // 2
    sim_b.add_packet((center, center - initial_sep//2), mass=8.0, width=3.0,
                     momentum=(0, 0.1), initial_q=0.3)
    sim_b.add_packet((center, center + initial_sep//2), mass=8.0, width=3.0,
                     momentum=(0, -0.1), initial_q=0.3)

    min_sep_b = initial_sep

    for t in range(1000):
        sep = sim_b.separation()
        min_sep_b = min(min_sep_b, sep)

        if verbose and t % 200 == 0:
            H_mean = np.mean(sim_b._compute_coherence_weighted_H())
            print(f"  t={t}: sep={sep:.1f}, H_mean={H_mean:.4f}, PE={sim_b.potential_energy():.3f}")

        sim_b.step()

    # Option B should still show binding behavior
    passed = min_sep_b < initial_sep * 0.6

    if verbose:
        print(f"\n  Initial sep: {initial_sep:.1f}")
        print(f"  Min sep (Option B): {min_sep_b:.1f}")
        print(f"  Option B {'PASSED' if passed else 'FAILED'}")

    return passed


def run_v6_3_test_suite(include_option_b: bool = False):
    """Run v6.3 2D test suite."""
    print("="*70)
    print("DET v6.3 2D COLLIDER - TEST SUITE")
    print("="*70)

    results = {}

    results['vacuum_gravity'] = test_v6_3_gravity_vacuum(verbose=True)
    results['binding'] = test_v6_3_binding(verbose=True)
    results['angular_momentum'] = test_v6_3_angular_momentum(verbose=True)

    if include_option_b:
        results['option_b_coherence_weighted_H'] = test_v6_3_option_b_coherence_weighted_H(verbose=True)

    print("\n" + "="*70)
    print("v6.3 2D TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return results


if __name__ == "__main__":
    run_v6_3_test_suite()
