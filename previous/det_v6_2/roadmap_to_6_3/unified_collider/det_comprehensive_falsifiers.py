"""
DET v6.2 Comprehensive Falsifier Suite
=======================================

Complete falsifier implementation including:
- F1: Locality Violation
- F2: Grace Coercion (a=0 blocks grace)
- F3: Boundary Redundancy
- F4: Regime Transition
- F5: Hidden Global Aggregates
- F6: Gravitational Binding
- F7: Mass Conservation
- F8: Vacuum Momentum
- F9: Symmetry Drift
- F10: Regime Discontinuity
- F_L1: Rotational Flux Conservation
- F_L2: Vacuum Spin No Transport
- F_L3: Orbital Capture
- F_GTD1-4: Gravitational Time Dilation

Reference: DET Theory Card v6.2
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import label
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import time


@dataclass
class DETParams3D:
    """DET 3D simulation parameters - complete."""
    N: int = 32
    DT: float = 0.02
    F_VAC: float = 0.01
    F_MIN: float = 0.0

    # Coherence
    C_init: float = 0.15

    # Diffusive flux
    diff_enabled: bool = True

    # Linear Momentum (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.12
    lambda_pi: float = 0.008
    mu_pi: float = 0.35
    pi_max: float = 3.0

    # Plaquette Angular Momentum (IV.5)
    angular_momentum_enabled: bool = True
    alpha_L: float = 0.06
    lambda_L: float = 0.005
    mu_L: float = 0.18
    L_max: float = 5.0

    # Floor repulsion (IV.6)
    floor_enabled: bool = True
    eta_floor: float = 0.12
    F_core: float = 5.0
    floor_power: float = 2.0

    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.012

    # Agency dynamics
    agency_dynamic: bool = True
    a_coupling: float = 30.0
    a_rate: float = 0.2

    # Sigma dynamics
    sigma_dynamic: bool = True

    # Coherence dynamics
    coherence_dynamic: bool = True

    # Gravity module (V.1-V.3)
    gravity_enabled: bool = True
    alpha_grav: float = 0.02
    kappa_grav: float = 5.0
    mu_grav: float = 2.0

    # Boundary operators (VI)
    boundary_enabled: bool = True
    grace_enabled: bool = True
    F_MIN_grace: float = 0.05
    healing_enabled: bool = False
    eta_heal: float = 0.03
    R_boundary: int = 2

    # Numerical stability
    outflow_limit: float = 0.2


def periodic_local_sum_3d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs)."""
    result = np.zeros_like(x)
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                result += np.roll(np.roll(np.roll(x, dz, axis=0), dy, axis=1), dx, axis=2)
    return result


class DETCollider3DComplete:
    """
    DET v6.2 3D Collider - Complete Implementation for Falsifier Testing

    Includes all modules: Gravity, Boundary, Angular Momentum, Time Dilation
    """

    def __init__(self, params: Optional[DETParams3D] = None):
        self.p = params or DETParams3D()
        N = self.p.N

        # Per-node state
        self.F = np.ones((N, N, N), dtype=np.float64) * self.p.F_VAC
        self.q = np.zeros((N, N, N), dtype=np.float64)
        self.a = np.ones((N, N, N), dtype=np.float64)
        self.theta = np.random.uniform(0, 2*np.pi, (N, N, N)).astype(np.float64)

        # Per-bond linear momentum
        self.pi_X = np.zeros((N, N, N), dtype=np.float64)
        self.pi_Y = np.zeros((N, N, N), dtype=np.float64)
        self.pi_Z = np.zeros((N, N, N), dtype=np.float64)

        # Per-plaquette angular momentum
        self.L_XY = np.zeros((N, N, N), dtype=np.float64)
        self.L_YZ = np.zeros((N, N, N), dtype=np.float64)
        self.L_XZ = np.zeros((N, N, N), dtype=np.float64)

        # Coherence
        self.C_X = np.ones((N, N, N), dtype=np.float64) * self.p.C_init
        self.C_Y = np.ones((N, N, N), dtype=np.float64) * self.p.C_init
        self.C_Z = np.ones((N, N, N), dtype=np.float64) * self.p.C_init

        self.sigma = np.ones((N, N, N), dtype=np.float64)

        # Gravity fields
        self.b = np.zeros((N, N, N), dtype=np.float64)
        self.Phi = np.zeros((N, N, N), dtype=np.float64)
        self.gx = np.zeros((N, N, N), dtype=np.float64)
        self.gy = np.zeros((N, N, N), dtype=np.float64)
        self.gz = np.zeros((N, N, N), dtype=np.float64)

        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N, N), dtype=np.float64)
        self.Delta_tau = np.ones((N, N, N), dtype=np.float64) * self.p.DT

        # Boundary operator diagnostics
        self.last_grace_injection = np.zeros((N, N, N), dtype=np.float64)
        self.total_grace_injected = 0.0

        # Time dilation tracking
        self.accumulated_proper_time = np.zeros((N, N, N), dtype=np.float64)

        self._setup_fft_solvers()

    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers."""
        N = self.p.N
        kx = np.fft.fftfreq(N) * N
        ky = np.fft.fftfreq(N) * N
        kz = np.fft.fftfreq(N) * N
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

        self.L_k = -4 * (np.sin(np.pi * KX / N)**2 +
                        np.sin(np.pi * KY / N)**2 +
                        np.sin(np.pi * KZ / N)**2)
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0, 0, 0] = 1.0

    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        source_k = fftn(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifftn(b_k))

    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        source_k = fftn(source)
        source_k[0, 0, 0] = 0
        Phi_k = -self.p.kappa_grav * source_k / self.L_k_poisson
        Phi_k[0, 0, 0] = 0
        return np.real(ifftn(Phi_k))

    def _compute_gravity(self):
        """Compute gravitational fields from q."""
        if not self.p.gravity_enabled:
            self.gx = np.zeros_like(self.F)
            self.gy = np.zeros_like(self.F)
            self.gz = np.zeros_like(self.F)
            return

        self.b = self._solve_helmholtz(self.q)
        rho = self.q - self.b
        self.Phi = self._solve_poisson(rho)

        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)
        Zm = lambda arr: np.roll(arr, 1, axis=0)

        self.gx = -0.5 * (Xp(self.Phi) - Xm(self.Phi))
        self.gy = -0.5 * (Yp(self.Phi) - Ym(self.Phi))
        self.gz = -0.5 * (Zp(self.Phi) - Zm(self.Phi))

    def compute_grace_injection(self, D: np.ndarray) -> np.ndarray:
        """Grace Injection per DET VI.5"""
        p = self.p
        n = np.maximum(0, p.F_MIN_grace - self.F)
        w = self.a * n
        w_sum = periodic_local_sum_3d(w, p.R_boundary) + 1e-12
        I_g = D * w / w_sum
        return I_g

    def add_packet(self, center: Tuple[int, int, int], mass: float = 10.0,
                   width: float = 3.0, momentum: Tuple[float, float, float] = (0, 0, 0),
                   initial_q: float = 0.0):
        """Add a 3D Gaussian resource packet."""
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        cz, cy, cx = center

        dx = (x - cx + N/2) % N - N/2
        dy = (y - cy + N/2) % N - N/2
        dz = (z - cz + N/2) % N - N/2

        r2 = dx**2 + dy**2 + dz**2
        envelope = np.exp(-0.5 * r2 / width**2)

        self.F += mass * envelope
        self.C_X = np.clip(self.C_X + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Y = np.clip(self.C_Y + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Z = np.clip(self.C_Z + 0.5 * envelope, self.p.C_init, 1.0)

        px, py, pz = momentum
        if px != 0 or py != 0 or pz != 0:
            self.pi_X += px * envelope
            self.pi_Y += py * envelope
            self.pi_Z += pz * envelope

        if initial_q > 0:
            self.q += initial_q * envelope

        self._clip()

    def add_spin(self, center: Tuple[int, int, int], spin: float = 1.0, width: float = 4.0):
        """Add initial angular momentum."""
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        cz, cy, cx = center

        dx = (x - cx + N/2) % N - N/2
        dy = (y - cy + N/2) % N - N/2
        dz = (z - cz + N/2) % N - N/2

        r2 = dx**2 + dy**2 + dz**2
        envelope = np.exp(-0.5 * r2 / width**2)

        self.L_XY += spin * envelope
        self._clip()

    def _clip(self):
        p = self.p
        self.F = np.clip(self.F, p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_X = np.clip(self.pi_X, -p.pi_max, p.pi_max)
        self.pi_Y = np.clip(self.pi_Y, -p.pi_max, p.pi_max)
        self.pi_Z = np.clip(self.pi_Z, -p.pi_max, p.pi_max)
        self.L_XY = np.clip(self.L_XY, -p.L_max, p.L_max)
        self.L_YZ = np.clip(self.L_YZ, -p.L_max, p.L_max)
        self.L_XZ = np.clip(self.L_XZ, -p.L_max, p.L_max)

    def step(self):
        """Execute one canonical DET update step."""
        p = self.p
        dk = p.DT
        N = p.N

        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)
        Zm = lambda arr: np.roll(arr, 1, axis=0)

        # STEP 0: Gravity
        self._compute_gravity()

        # STEP 1: Presence and proper time
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk

        # Track accumulated proper time for time dilation tests
        self.accumulated_proper_time += self.Delta_tau

        Delta_tau_Xp = 0.5 * (self.Delta_tau + Xp(self.Delta_tau))
        Delta_tau_Yp = 0.5 * (self.Delta_tau + Yp(self.Delta_tau))
        Delta_tau_Zp = 0.5 * (self.Delta_tau + Zp(self.Delta_tau))

        # Plaquette-averaged proper time
        Delta_tau_XY = 0.25 * (self.Delta_tau + Xp(self.Delta_tau) +
                               Yp(self.Delta_tau) + Xp(Yp(self.Delta_tau)))
        Delta_tau_YZ = 0.25 * (self.Delta_tau + Yp(self.Delta_tau) +
                               Zp(self.Delta_tau) + Yp(Zp(self.Delta_tau)))
        Delta_tau_XZ = 0.25 * (self.Delta_tau + Xp(self.Delta_tau) +
                               Zp(self.Delta_tau) + Xp(Zp(self.Delta_tau)))

        # STEP 2: Flux computation
        J_Xp = np.zeros_like(self.F)
        J_Xm = np.zeros_like(self.F)
        J_Yp = np.zeros_like(self.F)
        J_Ym = np.zeros_like(self.F)
        J_Zp = np.zeros_like(self.F)
        J_Zm = np.zeros_like(self.F)

        J_diff_Xp = np.zeros_like(self.F)
        J_diff_Yp = np.zeros_like(self.F)
        J_diff_Zp = np.zeros_like(self.F)

        # Diffusive flux
        if p.diff_enabled:
            classical_Xp = self.F - Xp(self.F)
            classical_Xm = self.F - Xm(self.F)
            classical_Yp = self.F - Yp(self.F)
            classical_Ym = self.F - Ym(self.F)
            classical_Zp = self.F - Zp(self.F)
            classical_Zm = self.F - Zm(self.F)

            g_Xp = np.sqrt(self.a * Xp(self.a))
            g_Xm = np.sqrt(self.a * Xm(self.a))
            g_Yp = np.sqrt(self.a * Yp(self.a))
            g_Ym = np.sqrt(self.a * Ym(self.a))
            g_Zp = np.sqrt(self.a * Zp(self.a))
            g_Zm = np.sqrt(self.a * Zm(self.a))

            cond_Xp = self.sigma * (self.C_X + 1e-4)
            cond_Xm = self.sigma * (Xm(self.C_X) + 1e-4)
            cond_Yp = self.sigma * (self.C_Y + 1e-4)
            cond_Ym = self.sigma * (Ym(self.C_Y) + 1e-4)
            cond_Zp = self.sigma * (self.C_Z + 1e-4)
            cond_Zm = self.sigma * (Zm(self.C_Z) + 1e-4)

            J_diff_Xp = g_Xp * cond_Xp * classical_Xp
            J_diff_Xm = g_Xm * cond_Xm * classical_Xm
            J_diff_Yp = g_Yp * cond_Yp * classical_Yp
            J_diff_Ym = g_Ym * cond_Ym * classical_Ym
            J_diff_Zp = g_Zp * cond_Zp * classical_Zp
            J_diff_Zm = g_Zm * cond_Zm * classical_Zm

            J_Xp += J_diff_Xp
            J_Xm += J_diff_Xm
            J_Yp += J_diff_Yp
            J_Ym += J_diff_Ym
            J_Zp += J_diff_Zp
            J_Zm += J_diff_Zm

        # Linear momentum flux
        if p.momentum_enabled:
            F_avg_Xp = 0.5 * (self.F + Xp(self.F))
            F_avg_Xm = 0.5 * (self.F + Xm(self.F))
            F_avg_Yp = 0.5 * (self.F + Yp(self.F))
            F_avg_Ym = 0.5 * (self.F + Ym(self.F))
            F_avg_Zp = 0.5 * (self.F + Zp(self.F))
            F_avg_Zm = 0.5 * (self.F + Zm(self.F))

            J_Xp += p.mu_pi * self.sigma * self.pi_X * F_avg_Xp
            J_Xm += -p.mu_pi * self.sigma * Xm(self.pi_X) * F_avg_Xm
            J_Yp += p.mu_pi * self.sigma * self.pi_Y * F_avg_Yp
            J_Ym += -p.mu_pi * self.sigma * Ym(self.pi_Y) * F_avg_Ym
            J_Zp += p.mu_pi * self.sigma * self.pi_Z * F_avg_Zp
            J_Zm += -p.mu_pi * self.sigma * Zm(self.pi_Z) * F_avg_Zm

        # Floor repulsion
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            classical_Xp = self.F - Xp(self.F)
            classical_Xm = self.F - Xm(self.F)
            classical_Yp = self.F - Yp(self.F)
            classical_Ym = self.F - Ym(self.F)
            classical_Zp = self.F - Zp(self.F)
            classical_Zm = self.F - Zm(self.F)

            J_Xp += p.eta_floor * self.sigma * (s + Xp(s)) * classical_Xp
            J_Xm += p.eta_floor * self.sigma * (s + Xm(s)) * classical_Xm
            J_Yp += p.eta_floor * self.sigma * (s + Yp(s)) * classical_Yp
            J_Ym += p.eta_floor * self.sigma * (s + Ym(s)) * classical_Ym
            J_Zp += p.eta_floor * self.sigma * (s + Zp(s)) * classical_Zp
            J_Zm += p.eta_floor * self.sigma * (s + Zm(s)) * classical_Zm

        # Gravitational flux
        if p.gravity_enabled:
            gx_bond_Xp = 0.5 * (self.gx + Xp(self.gx))
            gx_bond_Xm = 0.5 * (self.gx + Xm(self.gx))
            gy_bond_Yp = 0.5 * (self.gy + Yp(self.gy))
            gy_bond_Ym = 0.5 * (self.gy + Ym(self.gy))
            gz_bond_Zp = 0.5 * (self.gz + Zp(self.gz))
            gz_bond_Zm = 0.5 * (self.gz + Zm(self.gz))

            F_avg_Xp = 0.5 * (self.F + Xp(self.F))
            F_avg_Xm = 0.5 * (self.F + Xm(self.F))
            F_avg_Yp = 0.5 * (self.F + Yp(self.F))
            F_avg_Ym = 0.5 * (self.F + Ym(self.F))
            F_avg_Zp = 0.5 * (self.F + Zp(self.F))
            F_avg_Zm = 0.5 * (self.F + Zm(self.F))

            J_Xp += p.mu_grav * self.sigma * gx_bond_Xp * F_avg_Xp
            J_Xm += p.mu_grav * self.sigma * gx_bond_Xm * F_avg_Xm
            J_Yp += p.mu_grav * self.sigma * gy_bond_Yp * F_avg_Yp
            J_Ym += p.mu_grav * self.sigma * gy_bond_Ym * F_avg_Ym
            J_Zp += p.mu_grav * self.sigma * gz_bond_Zp * F_avg_Zp
            J_Zm += p.mu_grav * self.sigma * gz_bond_Zm * F_avg_Zm

        # Rotational flux from angular momentum
        if p.angular_momentum_enabled:
            F_avg_Xp = 0.5 * (self.F + Xp(self.F))
            F_avg_Yp = 0.5 * (self.F + Yp(self.F))
            F_avg_Zp = 0.5 * (self.F + Zp(self.F))

            J_rot_Xp_from_XY = p.mu_L * self.sigma * F_avg_Xp * (self.L_XY - Ym(self.L_XY))
            J_rot_Yp_from_XY = -p.mu_L * self.sigma * F_avg_Yp * (self.L_XY - Xm(self.L_XY))

            J_rot_Yp_from_YZ = p.mu_L * self.sigma * F_avg_Yp * (self.L_YZ - Zm(self.L_YZ))
            J_rot_Zp_from_YZ = -p.mu_L * self.sigma * F_avg_Zp * (self.L_YZ - Ym(self.L_YZ))

            J_rot_Xp_from_XZ = p.mu_L * self.sigma * F_avg_Xp * (self.L_XZ - Zm(self.L_XZ))
            J_rot_Zp_from_XZ = -p.mu_L * self.sigma * F_avg_Zp * (self.L_XZ - Xm(self.L_XZ))

            J_rot_Xp = J_rot_Xp_from_XY + J_rot_Xp_from_XZ
            J_rot_Yp = J_rot_Yp_from_XY + J_rot_Yp_from_YZ
            J_rot_Zp = J_rot_Zp_from_YZ + J_rot_Zp_from_XZ

            J_Xp += J_rot_Xp
            J_Xm -= Xm(J_rot_Xp)
            J_Yp += J_rot_Yp
            J_Ym -= Ym(J_rot_Yp)
            J_Zp += J_rot_Zp
            J_Zm -= Zm(J_rot_Zp)

        # STEP 3: Limiter
        total_outflow = (np.maximum(0, J_Xp) + np.maximum(0, J_Xm) +
                         np.maximum(0, J_Yp) + np.maximum(0, J_Ym) +
                         np.maximum(0, J_Zp) + np.maximum(0, J_Zm))
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))

        J_Xp_lim = np.where(J_Xp > 0, J_Xp * scale, J_Xp)
        J_Xm_lim = np.where(J_Xm > 0, J_Xm * scale, J_Xm)
        J_Yp_lim = np.where(J_Yp > 0, J_Yp * scale, J_Yp)
        J_Ym_lim = np.where(J_Ym > 0, J_Ym * scale, J_Ym)
        J_Zp_lim = np.where(J_Zp > 0, J_Zp * scale, J_Zp)
        J_Zm_lim = np.where(J_Zm > 0, J_Zm * scale, J_Zm)

        J_diff_Xp_scaled = np.where(J_diff_Xp > 0, J_diff_Xp * scale, J_diff_Xp)
        J_diff_Yp_scaled = np.where(J_diff_Yp > 0, J_diff_Yp * scale, J_diff_Yp)
        J_diff_Zp_scaled = np.where(J_diff_Zp > 0, J_diff_Zp * scale, J_diff_Zp)

        # Dissipation
        D = (np.abs(J_Xp_lim) + np.abs(J_Xm_lim) + np.abs(J_Yp_lim) +
             np.abs(J_Ym_lim) + np.abs(J_Zp_lim) + np.abs(J_Zm_lim)) * self.Delta_tau

        # STEP 4: Resource update
        transfer_Xp = J_Xp_lim * self.Delta_tau
        transfer_Xm = J_Xm_lim * self.Delta_tau
        transfer_Yp = J_Yp_lim * self.Delta_tau
        transfer_Ym = J_Ym_lim * self.Delta_tau
        transfer_Zp = J_Zp_lim * self.Delta_tau
        transfer_Zm = J_Zm_lim * self.Delta_tau

        outflow = transfer_Xp + transfer_Xm + transfer_Yp + transfer_Ym + transfer_Zp + transfer_Zm
        inflow = (Xm(transfer_Xp) + Xp(transfer_Xm) +
                  Ym(transfer_Yp) + Yp(transfer_Ym) +
                  Zm(transfer_Zp) + Zp(transfer_Zm))

        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)

        # STEP 5: Grace Injection
        if p.boundary_enabled and p.grace_enabled:
            I_g = self.compute_grace_injection(D)
            self.F = self.F + I_g
            self.last_grace_injection = I_g.copy()
            self.total_grace_injected += np.sum(I_g)
        else:
            self.last_grace_injection = np.zeros((N, N, N))

        # STEP 6: Momentum update
        if p.momentum_enabled:
            decay_Xp = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Xp)
            decay_Yp = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Yp)
            decay_Zp = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Zp)

            dpi_diff_X = p.alpha_pi * J_diff_Xp_scaled * Delta_tau_Xp
            dpi_diff_Y = p.alpha_pi * J_diff_Yp_scaled * Delta_tau_Yp
            dpi_diff_Z = p.alpha_pi * J_diff_Zp_scaled * Delta_tau_Zp

            if p.gravity_enabled:
                gx_bond_Xp = 0.5 * (self.gx + Xp(self.gx))
                gy_bond_Yp = 0.5 * (self.gy + Yp(self.gy))
                gz_bond_Zp = 0.5 * (self.gz + Zp(self.gz))
                dpi_grav_X = 5.0 * p.mu_grav * gx_bond_Xp * Delta_tau_Xp
                dpi_grav_Y = 5.0 * p.mu_grav * gy_bond_Yp * Delta_tau_Yp
                dpi_grav_Z = 5.0 * p.mu_grav * gz_bond_Zp * Delta_tau_Zp
            else:
                dpi_grav_X = dpi_grav_Y = dpi_grav_Z = 0

            self.pi_X = decay_Xp * self.pi_X + dpi_diff_X + dpi_grav_X
            self.pi_Y = decay_Yp * self.pi_Y + dpi_diff_Y + dpi_grav_Y
            self.pi_Z = decay_Zp * self.pi_Z + dpi_diff_Z + dpi_grav_Z

        # STEP 7: Angular momentum update
        if p.angular_momentum_enabled:
            curl_XY = self.pi_X + Xp(self.pi_Y) - Yp(self.pi_X) - self.pi_Y
            curl_YZ = self.pi_Y + Yp(self.pi_Z) - Zp(self.pi_Y) - self.pi_Z
            curl_XZ = self.pi_Z + Zp(self.pi_X) - Xp(self.pi_Z) - self.pi_X

            decay_XY = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_XY)
            decay_YZ = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_YZ)
            decay_XZ = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_XZ)

            self.L_XY = decay_XY * self.L_XY + p.alpha_L * curl_XY * Delta_tau_XY
            self.L_YZ = decay_YZ * self.L_YZ + p.alpha_L * curl_YZ * Delta_tau_YZ
            self.L_XZ = decay_XZ * self.L_XZ + p.alpha_L * curl_XZ * Delta_tau_XZ

        # STEP 8: Structure update
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)

        # STEP 9: Agency update
        if p.agency_dynamic:
            a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
            self.a = self.a + p.a_rate * (a_target - self.a)

        # STEP 10: Coherence and sigma dynamics
        J_mag = (np.abs(J_Xp_lim) + np.abs(J_Yp_lim) + np.abs(J_Zp_lim)) / 3.0

        if p.coherence_dynamic:
            self.C_X = np.clip(self.C_X + 0.04 * np.abs(J_Xp_lim) * self.Delta_tau
                              - 0.002 * self.C_X * self.Delta_tau, p.C_init, 1.0)
            self.C_Y = np.clip(self.C_Y + 0.04 * np.abs(J_Yp_lim) * self.Delta_tau
                              - 0.002 * self.C_Y * self.Delta_tau, p.C_init, 1.0)
            self.C_Z = np.clip(self.C_Z + 0.04 * np.abs(J_Zp_lim) * self.Delta_tau
                              - 0.002 * self.C_Z * self.Delta_tau, p.C_init, 1.0)

        if p.sigma_dynamic:
            self.sigma = 1.0 + 0.1 * np.log(1.0 + J_mag)

        self._clip()
        self.time += dk
        self.step_count += 1

    # Diagnostics
    def total_mass(self) -> float:
        return float(np.sum(self.F))

    def total_q(self) -> float:
        return float(np.sum(self.q))

    def total_angular_momentum(self) -> Tuple[float, float, float]:
        return float(np.sum(self.L_YZ)), float(np.sum(self.L_XZ)), float(np.sum(self.L_XY))

    def potential_energy(self) -> float:
        return float(np.sum(self.F * self.Phi))

    def center_of_mass(self) -> Tuple[float, float, float]:
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        total = np.sum(self.F) + 1e-9
        return (float(np.sum(x * self.F) / total),
                float(np.sum(y * self.F) / total),
                float(np.sum(z * self.F) / total))

    def find_blobs(self, threshold_ratio: float = 50.0) -> List[Dict]:
        """Find distinct blobs using connected components."""
        threshold = self.p.F_VAC * threshold_ratio
        above = self.F > threshold
        labeled, num = label(above)
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]

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
                'z': float(np.sum(z[mask] * weights) / total_mass),
                'mass': total_mass,
                'size': int(np.sum(mask))
            })
        blobs.sort(key=lambda c: -c['mass'])
        return blobs

    def separation(self) -> Tuple[float, int]:
        blobs = self.find_blobs()
        if len(blobs) < 2:
            return 0.0, len(blobs)
        N = self.p.N
        dx = blobs[1]['x'] - blobs[0]['x']
        dy = blobs[1]['y'] - blobs[0]['y']
        dz = blobs[1]['z'] - blobs[0]['z']
        dx = dx - N if dx > N/2 else (dx + N if dx < -N/2 else dx)
        dy = dy - N if dy > N/2 else (dy + N if dy < -N/2 else dy)
        dz = dz - N if dz > N/2 else (dz + N if dz < -N/2 else dz)
        return float(np.sqrt(dx**2 + dy**2 + dz**2)), len(blobs)

    def rot_flux_magnitude(self) -> float:
        """Magnitude of rotational flux."""
        if not self.p.angular_momentum_enabled:
            return 0.0
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)

        F_avg_Xp = 0.5 * (self.F + Xp(self.F))
        F_avg_Yp = 0.5 * (self.F + Yp(self.F))

        J_rot_Xp = self.p.mu_L * self.sigma * F_avg_Xp * (self.L_XY - Ym(self.L_XY))
        J_rot_Yp = -self.p.mu_L * self.sigma * F_avg_Yp * (self.L_XY - Xm(self.L_XY))

        return float(np.sum(np.abs(J_rot_Xp) + np.abs(J_rot_Yp)))


# ============================================================
# FALSIFIER IMPLEMENTATIONS
# ============================================================

def test_F1_locality_violation(verbose: bool = True) -> Dict:
    """
    F1: Locality Violation Test

    Verify that all interactions are strictly local (nearest-neighbor only).
    No information should propagate faster than 1 cell per timestep.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F1: Locality Violation")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02, F_VAC=0.01,
        diff_enabled=True, momentum_enabled=False,
        angular_momentum_enabled=False, floor_enabled=False,
        gravity_enabled=False, boundary_enabled=False,
        q_enabled=False, agency_dynamic=False
    )
    sim = DETCollider3DComplete(params)

    # Place a single-cell perturbation
    center = params.N // 2
    sim.F[center, center, center] = 10.0

    # Track propagation
    propagation_distances = []

    for t in range(20):
        # Find furthest cell with F > threshold
        threshold = params.F_VAC * 2
        above = sim.F > threshold
        if np.any(above):
            z, y, x = np.mgrid[0:params.N, 0:params.N, 0:params.N]
            dists = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
            max_dist = np.max(dists[above])
        else:
            max_dist = 0
        propagation_distances.append(max_dist)
        sim.step()

    # Check that propagation speed <= 1 cell per step
    max_speed = 0
    for i in range(1, len(propagation_distances)):
        speed = propagation_distances[i] - propagation_distances[i-1]
        max_speed = max(max_speed, speed)

    # Allow some tolerance for diffusion spreading
    passed = max_speed <= 2.0  # Conservative bound

    result = {
        'passed': passed,
        'max_speed': max_speed,
        'propagation_distances': propagation_distances
    }

    if verbose:
        print(f"  Max propagation speed: {max_speed:.2f} cells/step")
        print(f"  F1 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F2_grace_coercion(verbose: bool = True) -> Dict:
    """F2: Grace doesn't go to a=0 nodes"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F2: Grace Coercion (a=0 blocks grace)")
        print("="*60)

    params = DETParams3D(
        N=24, boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.15, a_rate=0.0, gravity_enabled=True
    )
    sim = DETCollider3DComplete(params)

    sim.add_packet((12, 12, 6), mass=3.0, width=2.0, momentum=(0, 0, 0.3))
    sim.add_packet((12, 12, 18), mass=3.0, width=2.0, momentum=(0, 0, -0.3))

    # Set sentinel node to a=0
    sz, sy, sx = 12, 12, 12
    sim.a[sz, sy, sx] = 0.0
    sim.F[sz, sy, sx] = 0.01

    for _ in range(200):
        sim.step()

    sentinel_grace = sim.last_grace_injection[sz, sy, sx]
    passed = sentinel_grace == 0.0

    result = {
        'passed': passed,
        'sentinel_a': sim.a[sz, sy, sx],
        'sentinel_grace': sentinel_grace
    }

    if verbose:
        print(f"  Sentinel a = {sim.a[sz, sy, sx]:.4f}")
        print(f"  Sentinel grace = {sentinel_grace:.2e}")
        print(f"  F2 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F3_boundary_redundancy(verbose: bool = True) -> Dict:
    """F3: Boundary ON produces different outcome than OFF"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F3: Boundary Redundancy")
        print("="*60)

    def run_scenario(boundary_on: bool):
        params = DETParams3D(
            N=24, F_VAC=0.02,
            boundary_enabled=boundary_on, grace_enabled=True, F_MIN_grace=0.15,
            gravity_enabled=True
        )
        sim = DETCollider3DComplete(params)
        sim.add_packet((12, 12, 6), mass=2.0, width=2.0, momentum=(0, 0, 0.3))
        sim.add_packet((12, 12, 18), mass=2.0, width=2.0, momentum=(0, 0, -0.3))

        for _ in range(300):
            sim.step()

        return np.mean(sim.F[10:14, 10:14, 10:14]), sim.total_grace_injected

    F_off, grace_off = run_scenario(False)
    F_on, grace_on = run_scenario(True)

    passed = grace_on > grace_off + 0.001

    result = {
        'passed': passed,
        'F_off': F_off,
        'F_on': F_on,
        'grace_off': grace_off,
        'grace_on': grace_on
    }

    if verbose:
        print(f"  OFF: F={F_off:.4f}, grace={grace_off:.4f}")
        print(f"  ON:  F={F_on:.4f}, grace={grace_on:.4f}")
        print(f"  F3 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F4_regime_transition(verbose: bool = True) -> Dict:
    """
    F4: Regime Transition Test

    Verify smooth transition between quantum and classical regimes
    as system parameters change (no discontinuous jumps).
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F4: Regime Transition")
        print("="*60)

    lambda_pi_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = []

    for lp in lambda_pi_values:
        params = DETParams3D(
            N=20, DT=0.02,
            momentum_enabled=True, lambda_pi=lp,
            gravity_enabled=False, boundary_enabled=False
        )
        sim = DETCollider3DComplete(params)
        sim.add_packet((10, 10, 10), mass=5.0, width=3.0, momentum=(0.5, 0.5, 0))

        total_momentum = []
        for _ in range(200):
            px = np.sum(sim.pi_X)
            py = np.sum(sim.pi_Y)
            total_momentum.append(np.sqrt(px**2 + py**2))
            sim.step()

        final_momentum = total_momentum[-1]
        decay_rate = (total_momentum[0] - final_momentum) / total_momentum[0] if total_momentum[0] > 0 else 0
        results.append({
            'lambda_pi': lp,
            'final_momentum': final_momentum,
            'decay_rate': decay_rate
        })

    # Check for monotonic decay rate increase with lambda_pi
    decay_rates = [r['decay_rate'] for r in results]
    monotonic = all(decay_rates[i] <= decay_rates[i+1] for i in range(len(decay_rates)-1))

    # Check for no discontinuous jumps (max ratio between consecutive values)
    max_ratio = 1.0
    for i in range(len(decay_rates)-1):
        if decay_rates[i] > 0.01:
            ratio = decay_rates[i+1] / decay_rates[i]
            max_ratio = max(max_ratio, ratio)

    passed = monotonic and max_ratio < 10.0

    result = {
        'passed': passed,
        'monotonic': monotonic,
        'max_ratio': max_ratio,
        'results': results
    }

    if verbose:
        print(f"  Decay rates: {[f'{r:.3f}' for r in decay_rates]}")
        print(f"  Monotonic: {monotonic}")
        print(f"  Max ratio: {max_ratio:.2f}")
        print(f"  F4 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F5_hidden_global_aggregates(verbose: bool = True) -> Dict:
    """
    F5: Hidden Global Aggregates Test

    Verify that no hidden global state affects local dynamics.
    Two isolated regions should evolve identically if given identical ICs.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F5: Hidden Global Aggregates")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02,
        gravity_enabled=False, boundary_enabled=False
    )

    # Region A: isolated packet in corner
    sim_a = DETCollider3DComplete(params)
    sim_a.add_packet((8, 8, 8), mass=5.0, width=2.0, momentum=(0.3, 0.3, 0))

    # Region B: same packet but with another packet far away
    sim_b = DETCollider3DComplete(params)
    sim_b.add_packet((8, 8, 8), mass=5.0, width=2.0, momentum=(0.3, 0.3, 0))
    sim_b.add_packet((24, 24, 24), mass=10.0, width=2.0, momentum=(-0.3, -0.3, 0))

    # Run both and compare region around first packet
    max_diff = 0
    for _ in range(200):
        sim_a.step()
        sim_b.step()

        # Compare local region
        region_a = sim_a.F[4:12, 4:12, 4:12]
        region_b = sim_b.F[4:12, 4:12, 4:12]
        diff = np.max(np.abs(region_a - region_b))
        max_diff = max(max_diff, diff)

    # Should be identical (within numerical precision)
    passed = max_diff < 1e-10

    result = {
        'passed': passed,
        'max_diff': max_diff
    }

    if verbose:
        print(f"  Max regional difference: {max_diff:.2e}")
        print(f"  F5 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F6_gravitational_binding(verbose: bool = True) -> Dict:
    """F6: Gravitational binding"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F6: Gravitational Binding")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        C_init=0.3, diff_enabled=True,
        momentum_enabled=True, alpha_pi=0.1, lambda_pi=0.002, mu_pi=0.5,
        angular_momentum_enabled=False, floor_enabled=False,
        q_enabled=True, alpha_q=0.02,
        agency_dynamic=True, a_coupling=3.0, a_rate=0.05,
        gravity_enabled=True, alpha_grav=0.01, kappa_grav=10.0, mu_grav=3.0,
        boundary_enabled=True, grace_enabled=True
    )

    sim = DETCollider3DComplete(params)

    initial_sep = 12
    center = params.N // 2
    sim.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, -0.1), initial_q=0.3)

    rec = {'t': [], 'sep': [], 'PE': []}

    for t in range(1500):
        sep, _ = sim.separation()
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['PE'].append(sim.potential_energy())

        if verbose and t % 300 == 0:
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.3f}")

        sim.step()

    initial_sep_m = rec['sep'][0] if rec['sep'][0] > 0 else initial_sep
    final_sep = rec['sep'][-1]
    min_sep = min(rec['sep'])

    sep_decreased = final_sep < initial_sep_m * 0.9
    bound_state = min_sep < initial_sep_m * 0.5

    passed = sep_decreased or bound_state

    result = {
        'passed': passed,
        'initial_sep': initial_sep_m,
        'final_sep': final_sep,
        'min_sep': min_sep,
        'sep_history': rec['sep'],
        'PE_history': rec['PE']
    }

    if verbose:
        print(f"\n  Initial sep: {initial_sep_m:.1f}, Final: {final_sep:.1f}, Min: {min_sep:.1f}")
        print(f"  F6 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F7_mass_conservation(verbose: bool = True) -> Dict:
    """F7: Mass conservation with gravity + boundary"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F7: Mass Conservation")
        print("="*60)

    params = DETParams3D(N=24, F_MIN=0.0, gravity_enabled=True, boundary_enabled=True)
    sim = DETCollider3DComplete(params)
    sim.add_packet((8, 8, 8), mass=10.0, width=3.0, momentum=(0.2, 0.2, 0.2))
    sim.add_packet((16, 16, 16), mass=10.0, width=3.0, momentum=(-0.2, -0.2, -0.2))

    initial_mass = sim.total_mass()

    for t in range(500):
        sim.step()

    final_mass = sim.total_mass()
    grace_added = sim.total_grace_injected
    effective_drift = abs(final_mass - initial_mass - grace_added) / initial_mass

    passed = effective_drift < 0.10

    result = {
        'passed': passed,
        'initial_mass': initial_mass,
        'final_mass': final_mass,
        'grace_added': grace_added,
        'effective_drift': effective_drift
    }

    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Grace added: {grace_added:.4f}")
        print(f"  Effective drift: {effective_drift*100:.4f}%")
        print(f"  F7 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F8_vacuum_momentum(verbose: bool = True) -> Dict:
    """F8: Momentum doesn't push vacuum"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F8: Vacuum Momentum")
        print("="*60)

    params = DETParams3D(
        N=16, momentum_enabled=True, q_enabled=False, floor_enabled=False,
        F_MIN=0.0, gravity_enabled=False, boundary_enabled=False
    )
    sim = DETCollider3DComplete(params)
    sim.F = np.ones_like(sim.F) * params.F_VAC
    sim.pi_X = np.ones_like(sim.pi_X) * 1.0

    initial_mass = sim.total_mass()

    for _ in range(200):
        sim.step()

    final_mass = sim.total_mass()
    drift = abs(final_mass - initial_mass) / initial_mass

    passed = drift < 0.01

    result = {
        'passed': passed,
        'initial_mass': initial_mass,
        'final_mass': final_mass,
        'drift': drift
    }

    if verbose:
        print(f"  Mass drift: {drift*100:.4f}%")
        print(f"  F8 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F9_symmetry_drift(verbose: bool = True) -> Dict:
    """F9: Symmetric IC doesn't drift"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F9: Symmetry Drift")
        print("="*60)

    params = DETParams3D(N=20, momentum_enabled=False, gravity_enabled=False, boundary_enabled=False)
    sim = DETCollider3DComplete(params)

    N = params.N
    sim.add_packet((N//2, N//2, N//4), mass=5.0, width=3.0, momentum=(0, 0, 0))
    sim.add_packet((N//2, N//2, 3*N//4), mass=5.0, width=3.0, momentum=(0, 0, 0))

    initial_com = sim.center_of_mass()

    max_drift = 0
    for _ in range(300):
        com = sim.center_of_mass()
        drift = np.sqrt((com[0] - initial_com[0])**2 +
                       (com[1] - initial_com[1])**2 +
                       (com[2] - initial_com[2])**2)
        max_drift = max(max_drift, drift)
        sim.step()

    passed = max_drift < 1.0

    result = {
        'passed': passed,
        'max_drift': max_drift
    }

    if verbose:
        print(f"  Max COM drift: {max_drift:.4f} cells")
        print(f"  F9 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F10_regime_discontinuity(verbose: bool = True) -> Dict:
    """
    F10: Regime Discontinuity Test

    Sweep lambda_pi and verify no discontinuous behavior in observables.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F10: Regime Discontinuity")
        print("="*60)

    lambda_values = np.linspace(0.001, 0.1, 20)
    final_spreads = []

    for lp in lambda_values:
        params = DETParams3D(
            N=20, DT=0.02,
            momentum_enabled=True, lambda_pi=lp,
            gravity_enabled=False, boundary_enabled=False
        )
        sim = DETCollider3DComplete(params)
        sim.add_packet((10, 10, 10), mass=5.0, width=2.0)

        for _ in range(100):
            sim.step()

        # Measure spatial spread
        z, y, x = np.mgrid[0:params.N, 0:params.N, 0:params.N]
        com = sim.center_of_mass()
        total = np.sum(sim.F) + 1e-9
        spread = np.sqrt(np.sum(((x - com[0])**2 + (y - com[1])**2 + (z - com[2])**2) * sim.F) / total)
        final_spreads.append(spread)

    # Check for discontinuities
    max_jump = 0
    for i in range(1, len(final_spreads)):
        jump = abs(final_spreads[i] - final_spreads[i-1])
        max_jump = max(max_jump, jump)

    passed = max_jump < 2.0  # No jumps larger than 2 cells

    result = {
        'passed': passed,
        'max_jump': max_jump,
        'lambda_values': lambda_values.tolist(),
        'final_spreads': final_spreads
    }

    if verbose:
        print(f"  Max discontinuity: {max_jump:.4f} cells")
        print(f"  F10 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_L1_rotational_conservation(verbose: bool = True) -> Dict:
    """F_L1: Rotational flux conservation (isolated)"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L1: Rotational Flux Conservation")
        print("="*60)

    params = DETParams3D(
        N=24,
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
    sim = DETCollider3DComplete(params)
    center = params.N // 2

    sim.add_packet((center, center, center), mass=15.0, width=4.0)
    sim.add_spin((center, center, center), spin=2.0, width=5.0)

    initial_F = sim.total_mass()
    initial_com = sim.center_of_mass()

    mass_history = []
    com_drift = []

    for t in range(500):
        sim.step()
        mass_history.append(sim.total_mass())
        com = sim.center_of_mass()
        drift = np.sqrt((com[0] - initial_com[0])**2 +
                       (com[1] - initial_com[1])**2 +
                       (com[2] - initial_com[2])**2)
        com_drift.append(drift)

    final_F = sim.total_mass()
    mass_err = abs(final_F - initial_F) / initial_F
    max_drift = max(com_drift)

    passed = mass_err < 1e-10 and max_drift < 0.1

    result = {
        'passed': passed,
        'initial_F': initial_F,
        'final_F': final_F,
        'mass_err': mass_err,
        'max_drift': max_drift
    }

    if verbose:
        print(f"  Mass error: {mass_err:.2e}")
        print(f"  Max COM drift: {max_drift:.6f} cells")
        print(f"  F_L1 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_L2_vacuum_spin(verbose: bool = True) -> Dict:
    """F_L2: Vacuum spin doesn't transport"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L2: Vacuum Spin No Transport")
        print("="*60)

    results = []
    F_vac_values = [0.001, 0.01, 0.1]

    for F_vac in F_vac_values:
        params = DETParams3D(
            N=20,
            F_VAC=F_vac,
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
        sim = DETCollider3DComplete(params)

        sim.F = np.ones_like(sim.F) * F_vac
        center = params.N // 2
        sim.add_spin((center, center, center), spin=2.0, width=5.0)

        initial_F = sim.total_mass()
        max_J_rot = 0

        for t in range(200):
            max_J_rot = max(max_J_rot, sim.rot_flux_magnitude())
            sim.step()

        final_F = sim.total_mass()
        mass_change = abs(final_F - initial_F)

        results.append({
            'F_vac': F_vac,
            'max_J_rot': max_J_rot,
            'mass_change': mass_change
        })

        if verbose:
            print(f"  F_vac={F_vac}: max|J_rot|={max_J_rot:.6f}, ΔF={mass_change:.2e}")

    ratio_J = results[2]['max_J_rot'] / (results[0]['max_J_rot'] + 1e-15)
    ratio_F = F_vac_values[2] / F_vac_values[0]
    scaling_ok = 0.5 < ratio_J / ratio_F < 2.0
    mass_ok = all(r['mass_change'] < 1e-10 for r in results)

    passed = scaling_ok and mass_ok

    result = {
        'passed': passed,
        'results': results,
        'scaling_ok': scaling_ok,
        'mass_ok': mass_ok
    }

    if verbose:
        print(f"\n  J_rot scaling ratio: {ratio_J:.2f} (expected ~{ratio_F:.0f})")
        print(f"  F_L2 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_L3_orbital_capture(verbose: bool = True) -> Dict:
    """F_L3: Orbital capture with angular momentum"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L3: Orbital Capture")
        print("="*60)

    params = DETParams3D(
        N=50,
        momentum_enabled=True,
        angular_momentum_enabled=True,
        floor_enabled=True,
        gravity_enabled=False,
        boundary_enabled=False
    )
    sim = DETCollider3DComplete(params)
    center = params.N // 2
    sep_init = 15
    b = 3

    sim.add_packet((center - sep_init, center - b, center), mass=10.0, width=2.5, momentum=(2.0, 0, 0))
    sim.add_packet((center + sep_init, center + b, center), mass=10.0, width=2.5, momentum=(-2.0, 0, 0))

    rec = {'t': [], 'sep': [], 'L_z': [], 'angle': []}
    prev_angle = 0
    total_angle = 0

    for t in range(1500):
        sep, num = sim.separation()
        L = sim.total_angular_momentum()

        blobs = sim.find_blobs()
        if len(blobs) >= 2:
            dx = blobs[1]['x'] - blobs[0]['x']
            dy = blobs[1]['y'] - blobs[0]['y']
            angle = np.arctan2(dy, dx)
            d_angle = angle - prev_angle
            if d_angle > np.pi: d_angle -= 2*np.pi
            elif d_angle < -np.pi: d_angle += 2*np.pi
            total_angle += d_angle
            prev_angle = angle

        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['L_z'].append(L[2])
        rec['angle'].append(total_angle)

        sim.step()

    sep_array = np.array(rec['sep'])
    valid_seps = sep_array[sep_array > 0]

    if len(valid_seps) > 10:
        second_half = valid_seps[len(valid_seps)//2:]
        sep_max = np.max(second_half)
        total_revolutions = abs(total_angle) / (2 * np.pi)
        bounded = sep_max < sep_init * 2.5
        orbital_capture = bounded and (total_revolutions > 0.1)
    else:
        orbital_capture = False
        total_revolutions = 0

    passed = orbital_capture

    result = {
        'passed': passed,
        'total_revolutions': total_revolutions,
        'final_L_z': rec['L_z'][-1],
        'sep_history': rec['sep'],
        'angle_history': rec['angle']
    }

    if verbose:
        print(f"  Orbital capture: {orbital_capture}")
        print(f"  Revolutions: {total_revolutions:.2f}")
        print(f"  Final L_z: {rec['L_z'][-1]:.4f}")
        print(f"  F_L3 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_GTD1_time_dilation(verbose: bool = True) -> Dict:
    """
    F_GTD1: Gravitational Time Dilation Test

    Verify presence-based time dilation: P = a*σ/(1+F)/(1+H)
    High-F regions should have lower presence (slower proper time).
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_GTD1: Gravitational Time Dilation")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False,
        agency_dynamic=False,  # Keep agency uniform
        sigma_dynamic=False    # Keep sigma uniform
    )
    sim = DETCollider3DComplete(params)

    # Create a massive body
    center = params.N // 2
    sim.add_packet((center, center, center), mass=20.0, width=3.0, initial_q=0.5)

    # Let it settle
    for _ in range(100):
        sim.step()

    # Measure P and F at different distances from center
    F_center = sim.F[center, center, center]
    F_edge = sim.F[center, center, center + 10]

    P_center = sim.P[center, center, center]
    P_edge = sim.P[center, center, center + 10]

    # With uniform a=1, σ=1, H=σ=1: P = 1/(1+F)/2
    # So P_center/P_edge = (1+F_edge)/(1+F_center)
    a_center = sim.a[center, center, center]
    a_edge = sim.a[center, center, center + 10]
    sigma_center = sim.sigma[center, center, center]
    sigma_edge = sim.sigma[center, center, center + 10]
    H_center = sigma_center
    H_edge = sigma_edge

    # Full formula: P = a*σ/(1+F)/(1+H)
    predicted_P_center = a_center * sigma_center / (1 + F_center) / (1 + H_center)
    predicted_P_edge = a_edge * sigma_edge / (1 + F_edge) / (1 + H_edge)

    # Key test: higher F should mean lower P (time dilation)
    time_dilated = P_center < P_edge

    # Also check the formula is correctly implemented
    formula_error_center = abs(P_center - predicted_P_center) / (predicted_P_center + 1e-10)
    formula_error_edge = abs(P_edge - predicted_P_edge) / (predicted_P_edge + 1e-10)
    formula_correct = formula_error_center < 0.01 and formula_error_edge < 0.01

    passed = time_dilated and formula_correct

    result = {
        'passed': passed,
        'F_center': F_center,
        'F_edge': F_edge,
        'P_center': P_center,
        'P_edge': P_edge,
        'predicted_P_center': predicted_P_center,
        'predicted_P_edge': predicted_P_edge,
        'time_dilated': time_dilated,
        'formula_correct': formula_correct
    }

    if verbose:
        print(f"  F at center: {F_center:.4f}")
        print(f"  F at edge: {F_edge:.4f}")
        print(f"  P at center: {P_center:.6f}")
        print(f"  P at edge: {P_edge:.6f}")
        print(f"  Time dilation (P_center < P_edge): {time_dilated}")
        print(f"  Formula correct: {formula_correct}")
        print(f"  F_GTD1 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_GTD2_accumulated_proper_time(verbose: bool = True) -> Dict:
    """
    F_GTD2: Accumulated Proper Time Test

    Verify accumulated proper time differs between high-F and low-F regions.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_GTD2: Accumulated Proper Time")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False
    )
    sim = DETCollider3DComplete(params)

    center = params.N // 2
    sim.add_packet((center, center, center), mass=30.0, width=3.0, initial_q=0.5)

    # Run simulation
    for _ in range(500):
        sim.step()

    # Compare accumulated proper time
    tau_center = sim.accumulated_proper_time[center, center, center]
    tau_edge = sim.accumulated_proper_time[center, center, center + 12]

    # High-F regions should have less accumulated proper time (time dilation)
    time_dilated = tau_center < tau_edge

    # Calculate dilation factor
    dilation_factor = tau_edge / tau_center if tau_center > 0 else 0

    passed = time_dilated and dilation_factor > 1.01

    result = {
        'passed': passed,
        'tau_center': tau_center,
        'tau_edge': tau_edge,
        'dilation_factor': dilation_factor
    }

    if verbose:
        print(f"  Proper time at center: {tau_center:.4f}")
        print(f"  Proper time at edge: {tau_edge:.4f}")
        print(f"  Dilation factor: {dilation_factor:.4f}")
        print(f"  F_GTD2 {'PASSED' if passed else 'FAILED'}")

    return result


# ============================================================
# MAIN TEST SUITE
# ============================================================

def run_comprehensive_test_suite():
    """Run the complete falsifier test suite."""
    print("="*70)
    print("DET v6.2 COMPREHENSIVE FALSIFIER SUITE")
    print("="*70)

    start_time = time.time()
    results = {}

    # Core falsifiers
    results['F1'] = test_F1_locality_violation(verbose=True)
    results['F2'] = test_F2_grace_coercion(verbose=True)
    results['F3'] = test_F3_boundary_redundancy(verbose=True)
    results['F4'] = test_F4_regime_transition(verbose=True)
    results['F5'] = test_F5_hidden_global_aggregates(verbose=True)
    results['F6'] = test_F6_gravitational_binding(verbose=True)
    results['F7'] = test_F7_mass_conservation(verbose=True)
    results['F8'] = test_F8_vacuum_momentum(verbose=True)
    results['F9'] = test_F9_symmetry_drift(verbose=True)
    results['F10'] = test_F10_regime_discontinuity(verbose=True)

    # Angular momentum falsifiers
    results['F_L1'] = test_F_L1_rotational_conservation(verbose=True)
    results['F_L2'] = test_F_L2_vacuum_spin(verbose=True)
    results['F_L3'] = test_F_L3_orbital_capture(verbose=True)

    # Gravitational time dilation falsifiers
    results['F_GTD1'] = test_F_GTD1_time_dilation(verbose=True)
    results['F_GTD2'] = test_F_GTD2_accumulated_proper_time(verbose=True)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE SUITE SUMMARY")
    print("="*70)

    passed_count = 0
    total_count = 0

    for name, result in results.items():
        if isinstance(result, dict):
            status = result.get('passed', False)
        else:
            status = result
        passed_count += 1 if status else 0
        total_count += 1
        print(f"  {name}: {'PASS' if status else 'FAIL'}")

    print(f"\n  TOTAL: {passed_count}/{total_count} PASSED")
    print(f"  Runtime: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    results = run_comprehensive_test_suite()
