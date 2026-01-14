"""
DET v6.3 3D Collider - Unified Implementation
=============================================

Complete implementation with all DET modules:
- Gravity module (Section V): Helmholtz baseline, Poisson potential, gravitational flux
- Boundary operators (Section VI): Grace injection with antisymmetric edge flux
- Agency-gated diffusion (IV.2)
- Presence-clocked transport (III.1)
- Momentum dynamics with gravity coupling (IV.4)
- Angular momentum dynamics (IV.5)
- Floor repulsion (IV.6)
- Lattice correction factor (V.3)

Reference: DET Theory Card v6.3

Changelog from v6.2:
- Added beta_g parameter for gravity-momentum coupling
- Added lattice correction factor eta
- Updated grace injection formulation
- Enhanced time dilation tracking
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import label
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


@dataclass
class DETParams3D:
    """DET v6.3 3D simulation parameters - complete."""
    # Grid and time
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

    # Agency dynamics (VI.2B) - v6.4 update
    agency_dynamic: bool = True
    lambda_a: float = 30.0      # Structural ceiling coupling (was a_coupling)
    beta_a: float = 0.2         # Relaxation rate toward ceiling (was a_rate)
    gamma_a_max: float = 0.15   # Max relational drive strength
    gamma_a_power: float = 2.0  # Coherence gating exponent (n >= 2)

    # Sigma dynamics
    sigma_dynamic: bool = True

    # Coherence dynamics
    coherence_dynamic: bool = True
    alpha_C: float = 0.04
    lambda_C: float = 0.002

    # Gravity module (V.1-V.3)
    gravity_enabled: bool = True
    alpha_grav: float = 0.02
    kappa_grav: float = 5.0
    mu_grav: float = 2.0
    beta_g: float = 10.0  # v6.3: gravity-momentum coupling (5.0 * mu_grav)

    # Boundary operators (VI)
    boundary_enabled: bool = True
    grace_enabled: bool = True
    F_MIN_grace: float = 0.05
    healing_enabled: bool = False
    eta_heal: float = 0.03
    R_boundary: int = 2

    # v6.3: Lattice correction factor
    eta_lattice: float = 0.965  # for N=64, scales with grid size

    # v6.3: Option B - Coherence-weighted load
    # H_i = Σ_{j ∈ N_R(i)} √C_ij * σ_ij
    coherence_weighted_H: bool = False

    # Numerical stability
    outflow_limit: float = 0.2


def compute_lattice_correction(N: int) -> float:
    """Compute lattice correction factor eta for given grid size.

    The discrete Laplacian creates systematic corrections:
    eta(N) = integral over BZ of lattice_eigenvalue / continuum_eigenvalue

    Returns eta approaching 1 as N -> infinity.
    """
    # Empirical fit based on theory card v6.3
    if N <= 32:
        return 0.901
    elif N <= 64:
        return 0.955
    elif N <= 96:
        return 0.968
    else:
        return 0.975


def periodic_local_sum_3d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs)."""
    result = np.zeros_like(x)
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                result += np.roll(np.roll(np.roll(x, dz, axis=0), dy, axis=1), dx, axis=2)
    return result


class DETCollider3D:
    """
    DET v6.3 3D Collider - Unified with Gravity, Boundary Operators, and Lattice Correction

    Key v6.3 features:
    - beta_g gravity-momentum coupling
    - Lattice correction factor eta
    - Enhanced time dilation tracking
    """

    def __init__(self, params: Optional[DETParams3D] = None):
        self.p = params or DETParams3D()
        N = self.p.N

        # Update lattice correction based on grid size
        self.p.eta_lattice = compute_lattice_correction(N)

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

        # Time dilation tracking (v6.3)
        self.accumulated_proper_time = np.zeros((N, N, N), dtype=np.float64)

        # Boundary operator diagnostics
        self.last_grace_injection = np.zeros((N, N, N), dtype=np.float64)
        self.total_grace_injected = 0.0

        self._setup_fft_solvers()

    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers for Helmholtz and Poisson solvers."""
        N = self.p.N
        kx = np.fft.fftfreq(N) * N
        ky = np.fft.fftfreq(N) * N
        kz = np.fft.fftfreq(N) * N
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

        # Discrete Laplacian eigenvalues: -4*sum(sin^2(k_i*pi/N))
        self.L_k = -4 * (np.sin(np.pi * KX / N)**2 +
                        np.sin(np.pi * KY / N)**2 +
                        np.sin(np.pi * KZ / N)**2)

        # Helmholtz operator: (L - alpha)
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12

        # Poisson operator (avoid zero mode)
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0, 0, 0] = 1.0

    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        """Solve (L - alpha)*b = -alpha*q for baseline field."""
        source_k = fftn(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifftn(b_k))

    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        """Solve L*Phi = kappa*rho for gravitational potential.

        Note: L_k < 0 (discrete Laplacian eigenvalues are non-positive).
        With Phi_k = kappa*rho_k / L_k, we get Phi < 0 near mass (like Newtonian).
        Then g = -grad(Phi) points TOWARD mass (attractive gravity).
        """
        source_k = fftn(source)
        source_k[0, 0, 0] = 0  # Remove mean
        # Apply lattice correction factor (v6.3)
        # Sign: kappa*rho / L_k with L_k < 0 gives Phi < 0 (attractive)
        Phi_k = self.p.kappa_grav * self.p.eta_lattice * source_k / self.L_k_poisson
        Phi_k[0, 0, 0] = 0
        return np.real(ifftn(Phi_k))

    def _compute_coherence_weighted_H(self) -> np.ndarray:
        """Compute Option B coherence-weighted load.

        H_i = Σ_{j ∈ N_R(i)} √C_ij * σ_ij

        In 3D, sum over 6 neighbors (±X, ±Y, ±Z).
        """
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)
        Zm = lambda arr: np.roll(arr, 1, axis=0)

        # Bond coherences
        sqrt_C_Xp = np.sqrt(self.C_X)           # C(i, i+X)
        sqrt_C_Xm = np.sqrt(Xm(self.C_X))       # C(i-X, i)
        sqrt_C_Yp = np.sqrt(self.C_Y)           # C(i, i+Y)
        sqrt_C_Ym = np.sqrt(Ym(self.C_Y))       # C(i-Y, i)
        sqrt_C_Zp = np.sqrt(self.C_Z)           # C(i, i+Z)
        sqrt_C_Zm = np.sqrt(Zm(self.C_Z))       # C(i-Z, i)

        # Bond-averaged sigma
        sigma_Xp = 0.5 * (self.sigma + Xp(self.sigma))
        sigma_Xm = 0.5 * (self.sigma + Xm(self.sigma))
        sigma_Yp = 0.5 * (self.sigma + Yp(self.sigma))
        sigma_Ym = 0.5 * (self.sigma + Ym(self.sigma))
        sigma_Zp = 0.5 * (self.sigma + Zp(self.sigma))
        sigma_Zm = 0.5 * (self.sigma + Zm(self.sigma))

        # Coherence-weighted load: sum over 6 neighbors
        H = (sqrt_C_Xp * sigma_Xp + sqrt_C_Xm * sigma_Xm +
             sqrt_C_Yp * sigma_Yp + sqrt_C_Ym * sigma_Ym +
             sqrt_C_Zp * sigma_Zp + sqrt_C_Zm * sigma_Zm)

        return H

    def _compute_gravity(self):
        """Compute gravitational fields from structure q.

        DET gravity is sourced by imbalance between local q and dynamically
        computed baseline b: rho = q - b.
        """
        if not self.p.gravity_enabled:
            self.gx = np.zeros_like(self.F)
            self.gy = np.zeros_like(self.F)
            self.gz = np.zeros_like(self.F)
            return

        # Step 1: Solve Helmholtz for baseline
        self.b = self._solve_helmholtz(self.q)

        # Step 2: Compute relative source
        rho = self.q - self.b

        # Step 3: Solve Poisson for potential
        self.Phi = self._solve_poisson(rho)

        # Step 4: Compute gravitational force (negative gradient)
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
        """Grace Injection per DET VI.5 (v6.2 simple form).

        Grace is injected to nodes below threshold, weighted by agency
        and distributed locally from dissipation pool.
        """
        p = self.p
        n = np.maximum(0, p.F_MIN_grace - self.F)
        w = self.a * n
        w_sum = periodic_local_sum_3d(w, p.R_boundary) + 1e-12
        I_g = D * w / w_sum
        return I_g

    def add_packet(self, center: Tuple[int, int, int], mass: float = 10.0,
                   width: float = 3.0, momentum: Tuple[float, float, float] = (0, 0, 0),
                   initial_q: float = 0.0, initial_spin: float = 0.0):
        """Add a 3D Gaussian resource packet with optional momentum and spin."""
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        cz, cy, cx = center

        # Periodic distance
        dx = (x - cx + N/2) % N - N/2
        dy = (y - cy + N/2) % N - N/2
        dz = (z - cz + N/2) % N - N/2

        r2 = dx**2 + dy**2 + dz**2
        envelope = np.exp(-0.5 * r2 / width**2)

        # Add resource
        self.F += mass * envelope

        # Boost coherence in packet region
        self.C_X = np.clip(self.C_X + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Y = np.clip(self.C_Y + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Z = np.clip(self.C_Z + 0.5 * envelope, self.p.C_init, 1.0)

        # Add momentum
        px, py, pz = momentum
        if px != 0 or py != 0 or pz != 0:
            self.pi_X += px * envelope
            self.pi_Y += py * envelope
            self.pi_Z += pz * envelope

        # Add structure (sources gravity)
        if initial_q > 0:
            self.q += initial_q * envelope

        # Add angular momentum
        if initial_spin != 0:
            self.L_XY += initial_spin * envelope

        self._clip()

    def add_spin(self, center: Tuple[int, int, int], spin: float = 1.0, width: float = 4.0):
        """Add initial angular momentum to a region."""
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
        """Enforce physical bounds on all state variables."""
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
        """Execute one canonical DET update step per Theory Card v6.3."""
        p = self.p
        dk = p.DT
        N = p.N

        # Shift operators (periodic boundary conditions)
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)
        Zm = lambda arr: np.roll(arr, 1, axis=0)

        # STEP 0: Compute gravitational fields (V.1-V.2)
        self._compute_gravity()

        # STEP 1: Presence and proper time (III.1)
        # Option B: Coherence-weighted load H_i = Σ_{j} √C_ij * σ_ij
        if p.coherence_weighted_H:
            H = self._compute_coherence_weighted_H()
        else:
            H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk

        # Track accumulated proper time (v6.3)
        self.accumulated_proper_time += self.Delta_tau

        # Bond-averaged proper time
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

        # Diffusive flux (agency-gated)
        if p.diff_enabled:
            # Classical pressure gradient
            classical_Xp = self.F - Xp(self.F)
            classical_Xm = self.F - Xm(self.F)
            classical_Yp = self.F - Yp(self.F)
            classical_Ym = self.F - Ym(self.F)
            classical_Zp = self.F - Zp(self.F)
            classical_Zm = self.F - Zm(self.F)

            # Agency gate: g^(a)_ij = sqrt(a_i * a_j)
            g_Xp = np.sqrt(self.a * Xp(self.a))
            g_Xm = np.sqrt(self.a * Xm(self.a))
            g_Yp = np.sqrt(self.a * Yp(self.a))
            g_Ym = np.sqrt(self.a * Ym(self.a))
            g_Zp = np.sqrt(self.a * Zp(self.a))
            g_Zm = np.sqrt(self.a * Zm(self.a))

            # Conductivity from coherence
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

        # Linear momentum flux (F-weighted)
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

        # Floor repulsion (agency-independent)
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

            # Rotational flux from L_XY
            J_rot_Xp_from_XY = p.mu_L * self.sigma * F_avg_Xp * (self.L_XY - Ym(self.L_XY))
            J_rot_Yp_from_XY = -p.mu_L * self.sigma * F_avg_Yp * (self.L_XY - Xm(self.L_XY))

            # Rotational flux from L_YZ
            J_rot_Yp_from_YZ = p.mu_L * self.sigma * F_avg_Yp * (self.L_YZ - Zm(self.L_YZ))
            J_rot_Zp_from_YZ = -p.mu_L * self.sigma * F_avg_Zp * (self.L_YZ - Ym(self.L_YZ))

            # Rotational flux from L_XZ
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

        # STEP 3: Conservative limiter
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

        # Dissipation (for grace injection)
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

        # STEP 5: Grace Injection (VI.6)
        if p.boundary_enabled and p.grace_enabled:
            I_g = self.compute_grace_injection(D)
            self.F = self.F + I_g
            self.last_grace_injection = I_g.copy()
            self.total_grace_injected += np.sum(I_g)
        else:
            self.last_grace_injection = np.zeros((N, N, N))

        # STEP 6: Momentum update with gravity coupling (v6.3)
        if p.momentum_enabled:
            decay_Xp = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Xp)
            decay_Yp = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Yp)
            decay_Zp = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_Zp)

            # Charging from diffusive flux
            dpi_diff_X = p.alpha_pi * J_diff_Xp_scaled * Delta_tau_Xp
            dpi_diff_Y = p.alpha_pi * J_diff_Yp_scaled * Delta_tau_Yp
            dpi_diff_Z = p.alpha_pi * J_diff_Zp_scaled * Delta_tau_Zp

            # Charging from gravity (v6.3: beta_g coupling)
            if p.gravity_enabled:
                gx_bond_Xp = 0.5 * (self.gx + Xp(self.gx))
                gy_bond_Yp = 0.5 * (self.gy + Yp(self.gy))
                gz_bond_Zp = 0.5 * (self.gz + Zp(self.gz))
                dpi_grav_X = p.beta_g * gx_bond_Xp * Delta_tau_Xp
                dpi_grav_Y = p.beta_g * gy_bond_Yp * Delta_tau_Yp
                dpi_grav_Z = p.beta_g * gz_bond_Zp * Delta_tau_Zp
            else:
                dpi_grav_X = dpi_grav_Y = dpi_grav_Z = 0

            self.pi_X = decay_Xp * self.pi_X + dpi_diff_X + dpi_grav_X
            self.pi_Y = decay_Yp * self.pi_Y + dpi_diff_Y + dpi_grav_Y
            self.pi_Z = decay_Zp * self.pi_Z + dpi_diff_Z + dpi_grav_Z

        # STEP 7: Angular momentum update (IV.5)
        if p.angular_momentum_enabled:
            # Curl of momentum (plaquette circulation)
            curl_XY = self.pi_X + Xp(self.pi_Y) - Yp(self.pi_X) - self.pi_Y
            curl_YZ = self.pi_Y + Yp(self.pi_Z) - Zp(self.pi_Y) - self.pi_Z
            curl_XZ = self.pi_Z + Zp(self.pi_X) - Xp(self.pi_Z) - self.pi_X

            decay_XY = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_XY)
            decay_YZ = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_YZ)
            decay_XZ = np.maximum(0.0, 1.0 - p.lambda_L * Delta_tau_XZ)

            self.L_XY = decay_XY * self.L_XY + p.alpha_L * curl_XY * Delta_tau_XY
            self.L_YZ = decay_YZ * self.L_YZ + p.alpha_L * curl_YZ * Delta_tau_YZ
            self.L_XZ = decay_XZ * self.L_XZ + p.alpha_L * curl_XZ * Delta_tau_XZ

        # STEP 8: Structure update (q-locking)
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)

        # STEP 9: Agency update
        if p.agency_dynamic:
            # v6.4 Agency Law: Structural Ceiling + Relational Drive

            # Step 1: Structural ceiling (matter law)
            # a_max = 1 / (1 + λ_a * q²) - what the system permits
            a_max = 1.0 / (1.0 + p.lambda_a * self.q**2)

            # Step 2: Relational drive (life law)
            # Δa_drive = γ(C) * (P_i - P̄_neighbors)
            # where γ(C) = γ_max * C^n (coherence-gated)

            # Compute local average presence
            P_local = self.P.copy()
            for d in range(3):
                P_local = P_local + np.roll(self.P, 1, axis=d) + np.roll(self.P, -1, axis=d)
            P_local = P_local / 7.0  # Self + 6 neighbors

            # Compute average coherence at each node
            C_avg = (self.C_X + np.roll(self.C_X, 1, axis=2) +
                     self.C_Y + np.roll(self.C_Y, 1, axis=1) +
                     self.C_Z + np.roll(self.C_Z, 1, axis=0)) / 6.0

            # Coherence-gated drive strength: γ(C) = γ_max * C^n
            gamma = p.gamma_a_max * (C_avg ** p.gamma_a_power)

            # Relational drive: seeks presence gradients
            delta_a_drive = gamma * (self.P - P_local)

            # Step 3: Unified update with ceiling relaxation + drive
            self.a = self.a + p.beta_a * (a_max - self.a) + delta_a_drive

            # Clip to [0, a_max]
            self.a = np.clip(self.a, 0.0, a_max)

        # STEP 10: Coherence and sigma dynamics
        J_mag = (np.abs(J_Xp_lim) + np.abs(J_Yp_lim) + np.abs(J_Zp_lim)) / 3.0

        if p.coherence_dynamic:
            self.C_X = np.clip(self.C_X + p.alpha_C * np.abs(J_Xp_lim) * self.Delta_tau
                              - p.lambda_C * self.C_X * self.Delta_tau, p.C_init, 1.0)
            self.C_Y = np.clip(self.C_Y + p.alpha_C * np.abs(J_Yp_lim) * self.Delta_tau
                              - p.lambda_C * self.C_Y * self.Delta_tau, p.C_init, 1.0)
            self.C_Z = np.clip(self.C_Z + p.alpha_C * np.abs(J_Zp_lim) * self.Delta_tau
                              - p.lambda_C * self.C_Z * self.Delta_tau, p.C_init, 1.0)

        if p.sigma_dynamic:
            self.sigma = 1.0 + 0.1 * np.log(1.0 + J_mag)

        self._clip()
        self.time += dk
        self.step_count += 1

    # ==================== DIAGNOSTICS ====================

    def total_mass(self) -> float:
        """Total resource in system."""
        return float(np.sum(self.F))

    def total_q(self) -> float:
        """Total structural debt."""
        return float(np.sum(self.q))

    def total_angular_momentum(self) -> Tuple[float, float, float]:
        """Total angular momentum (L_x, L_y, L_z)."""
        return float(np.sum(self.L_YZ)), float(np.sum(self.L_XZ)), float(np.sum(self.L_XY))

    def potential_energy(self) -> float:
        """Gravitational potential energy."""
        return float(np.sum(self.F * self.Phi))

    def center_of_mass(self) -> Tuple[float, float, float]:
        """Center of mass position (x, y, z)."""
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
        """Find separation between two largest blobs."""
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
# TEST SUITE
# ============================================================

def test_v6_3_gravity_vacuum(verbose: bool = True) -> bool:
    """Gravity has no effect in vacuum (q=0)."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravity in Vacuum")
        print("="*60)

    params = DETParams3D(N=16, gravity_enabled=True, q_enabled=False, boundary_enabled=False)
    sim = DETCollider3D(params)

    for _ in range(200):
        sim.step()

    max_g = np.max(np.abs(sim.gx)) + np.max(np.abs(sim.gy)) + np.max(np.abs(sim.gz))
    max_Phi = np.max(np.abs(sim.Phi))

    passed = max_g < 1e-10 and max_Phi < 1e-10

    if verbose:
        print(f"  Max |g|: {max_g:.2e}")
        print(f"  Max |Phi|: {max_Phi:.2e}")
        print(f"  Lattice correction eta: {sim.p.eta_lattice:.3f}")
        print(f"  Vacuum gravity {'PASSED' if passed else 'FAILED'}")

    return passed


def test_v6_3_gravitational_binding(verbose: bool = True) -> bool:
    """Test gravitational binding with v6.3 beta_g coupling."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravitational Binding (v6.3)")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True
    )

    sim = DETCollider3D(params)

    initial_sep = 12
    center = params.N // 2
    sim.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, -0.1), initial_q=0.3)

    min_sep = initial_sep

    for t in range(1000):
        sep, _ = sim.separation()
        min_sep = min(min_sep, sep)

        if verbose and t % 200 == 0:
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.3f}")

        sim.step()

    passed = min_sep < initial_sep * 0.5

    if verbose:
        print(f"\n  Initial sep: {initial_sep:.1f}")
        print(f"  Min sep: {min_sep:.1f}")
        print(f"  beta_g (gravity-momentum coupling): {params.beta_g}")
        print(f"  Binding {'PASSED' if passed else 'FAILED'}")

    return passed


def test_v6_3_time_dilation(verbose: bool = True) -> bool:
    """Test gravitational time dilation P = a*sigma/(1+F)/(1+H)."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Time Dilation (v6.3)")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False
    )
    sim = DETCollider3D(params)

    center = params.N // 2
    sim.add_packet((center, center, center), mass=20.0, width=3.0, initial_q=0.5)

    for _ in range(100):
        sim.step()

    F_center = sim.F[center, center, center]
    F_edge = sim.F[center, center, center + 10]
    P_center = sim.P[center, center, center]
    P_edge = sim.P[center, center, center + 10]

    time_dilated = P_center < P_edge

    # Check formula: P = a*sigma/(1+F)/(1+H)
    a_center = sim.a[center, center, center]
    sigma_center = sim.sigma[center, center, center]
    H_center = sigma_center
    predicted_P = a_center * sigma_center / (1 + F_center) / (1 + H_center)
    formula_error = abs(P_center - predicted_P) / (predicted_P + 1e-10)

    passed = time_dilated and formula_error < 0.01

    if verbose:
        print(f"  F at center: {F_center:.4f}")
        print(f"  F at edge: {F_edge:.4f}")
        print(f"  P at center: {P_center:.6f}")
        print(f"  P at edge: {P_edge:.6f}")
        print(f"  Time dilated (P_center < P_edge): {time_dilated}")
        print(f"  Formula error: {formula_error*100:.4f}%")
        print(f"  Time dilation {'PASSED' if passed else 'FAILED'}")

    return passed


def test_v6_3_option_b_coherence_weighted_H(verbose: bool = True) -> bool:
    """Test Option B: Coherence-weighted load H_i = Σ √C_ij σ_ij."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Option B - Coherence-Weighted Load")
        print("="*60)

    # Test with Option B enabled
    params_b = DETParams3D(
        N=32, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        coherence_weighted_H=True  # Enable Option B
    )

    sim_b = DETCollider3D(params_b)

    initial_sep = 12
    center = params_b.N // 2
    sim_b.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                     momentum=(0, 0, 0.1), initial_q=0.3)
    sim_b.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                     momentum=(0, 0, -0.1), initial_q=0.3)

    min_sep_b = initial_sep

    for t in range(1000):
        sep, _ = sim_b.separation()
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
    """Run v6.3 test suite."""
    print("="*70)
    print("DET v6.3 3D COLLIDER - TEST SUITE")
    print("="*70)

    results = {}

    results['vacuum_gravity'] = test_v6_3_gravity_vacuum(verbose=True)
    results['binding'] = test_v6_3_gravitational_binding(verbose=True)
    results['time_dilation'] = test_v6_3_time_dilation(verbose=True)

    if include_option_b:
        results['option_b_coherence_weighted_H'] = test_v6_3_option_b_coherence_weighted_H(verbose=True)

    print("\n" + "="*70)
    print("v6.3 TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return results


if __name__ == "__main__":
    run_v6_3_test_suite()
