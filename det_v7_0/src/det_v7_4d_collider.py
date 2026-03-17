"""
DET v7.0 4D Collider (Unified Mutable Structural Debt)
======================================================

Canonical 4D extension of the DET v7 collider family.

Patch alignment:
- Unified structural debt: q only
- Structural drag in presence law: D = 1/(1 + lambda_P*q)
- Agency-first update without structural ceiling
- Jubilee reduces total q (lawful recovery)

4D topology:
- 8-neighbor adjacency: ±X, ±Y, ±Z, ±W (face-sharing hypercubes)
- 4 bond momenta: pi_X, pi_Y, pi_Z, pi_W
- 6 plaquette angular momenta: L_XY, L_XZ, L_XW, L_YZ, L_YW, L_ZW
- 4 bond coherences: C_X, C_Y, C_Z, C_W
- Periodic boundary conditions on all 4 axes
- FFT-based Helmholtz/Poisson solvers generalized to 4D

Research target:
- Test whether DET canonical laws remain stable under 4D adjacency
- Probe binding, orbit persistence, diffusion, identity, recovery, projection
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import label
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List


# ============================================================
# 4D TOPOLOGY CONSTANTS
# ============================================================

# In 4D, a node has 8 face-sharing neighbors (±X, ±Y, ±Z, ±W).
# There are 4 bond directions and C(4,2) = 6 plaquette planes.
AXES_4D = ('X', 'Y', 'Z', 'W')
AXIS_INDICES = {'X': 3, 'Y': 2, 'Z': 1, 'W': 0}  # numpy axis mapping
PLANES_4D = ('XY', 'XZ', 'XW', 'YZ', 'YW', 'ZW')
NUM_NEIGHBORS_4D = 8
NUM_BONDS_4D = 4
NUM_PLAQUETTES_4D = 6


@dataclass
class DETParams4D:
    """DET v7.0 4D simulation parameters.

    Extends DETParams3D to 4D while preserving all canonical laws.
    Default grid size is smaller (N=16) due to O(N^4) memory scaling.
    """
    # Grid and time
    N: int = 16
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

    # Structure (q-locking / mutable recovery)
    q_enabled: bool = True
    alpha_q: float = 0.012

    # Presence drag (unified-q patch)
    lambda_P: float = 3.0
    gamma_v: float = 1.0

    # Structural Debt Couplings (v6.4 extension)
    debt_conductivity_enabled: bool = False
    xi_conductivity: float = 2.0
    debt_temporal_enabled: bool = False
    zeta_temporal: float = 0.5
    debt_decoherence_enabled: bool = False
    theta_decoherence: float = 1.0

    # Agency dynamics (v7 canonical)
    agency_dynamic: bool = True
    lambda_a: float = 30.0      # Deprecated (kept for backward compatibility)
    beta_a: float = 0.2
    gamma_a_max: float = 0.15
    gamma_a_power: float = 2.0
    a0: float = 1.0
    epsilon_a: float = 0.0

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
    beta_g: float = 10.0

    # Boundary operators (VI)
    boundary_enabled: bool = True
    grace_enabled: bool = True
    F_MIN_grace: float = 0.05
    healing_enabled: bool = False
    eta_heal: float = 0.03
    R_boundary: int = 2

    # Lattice correction factor
    eta_lattice: float = 0.965

    # Coherence-weighted load
    coherence_weighted_H: bool = False

    # Boundary Jubilee operator (q recovery)
    jubilee_enabled: bool = False
    delta_q: float = 0.001
    n_q: float = 2.0
    D_0: float = 0.05
    jubilee_energy_coupling: bool = True

    # Numerical stability
    outflow_limit: float = 0.2

    def __post_init__(self):
        self.alpha_q = float(self.alpha_q)


def compute_lattice_correction_4d(N: int) -> float:
    """Compute lattice correction factor eta for 4D grid.

    4D discrete Laplacian has stronger finite-size corrections than 3D.
    """
    if N <= 8:
        return 0.850
    elif N <= 12:
        return 0.880
    elif N <= 16:
        return 0.901
    elif N <= 24:
        return 0.935
    elif N <= 32:
        return 0.955
    else:
        return 0.970


def periodic_local_sum_4d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs) in 4D.

    Sums over a (2*radius+1)^4 hypercube neighborhood.
    """
    result = np.zeros_like(x)
    for dw in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    result += np.roll(
                        np.roll(
                            np.roll(
                                np.roll(x, dw, axis=0),
                                dz, axis=1),
                            dy, axis=2),
                        dx, axis=3)
    return result


# ============================================================
# SHIFT OPERATORS
# ============================================================

def _shift_plus(arr: np.ndarray, axis_name: str) -> np.ndarray:
    """Shift array +1 along named axis (periodic)."""
    return np.roll(arr, -1, axis=AXIS_INDICES[axis_name])

def _shift_minus(arr: np.ndarray, axis_name: str) -> np.ndarray:
    """Shift array -1 along named axis (periodic)."""
    return np.roll(arr, 1, axis=AXIS_INDICES[axis_name])

# Convenience aliases
Xp = lambda arr: _shift_plus(arr, 'X')
Xm = lambda arr: _shift_minus(arr, 'X')
Yp = lambda arr: _shift_plus(arr, 'Y')
Ym = lambda arr: _shift_minus(arr, 'Y')
Zp = lambda arr: _shift_plus(arr, 'Z')
Zm = lambda arr: _shift_minus(arr, 'Z')
Wp = lambda arr: _shift_plus(arr, 'W')
Wm = lambda arr: _shift_minus(arr, 'W')

# Map axis name to shift functions
SHIFT_PLUS = {'X': Xp, 'Y': Yp, 'Z': Zp, 'W': Wp}
SHIFT_MINUS = {'X': Xm, 'Y': Ym, 'Z': Zm, 'W': Wm}


class DETCollider4D:
    """
    DET v7.0 4D Collider - Canonical Extension to Four Spatial Dimensions

    Preserves all DET v7 canonical laws:
    - Strict locality (8-neighbor face-sharing adjacency)
    - Agency-first invariance (no structural ceiling on a)
    - Presence/drag expression through q
    - Canonical 15-step update order
    - Boundary operator discipline (Grace, Healing, Jubilee)

    4D-specific features:
    - 4 bond momenta (pi_X, pi_Y, pi_Z, pi_W)
    - 6 plaquette angular momenta (L_XY, L_XZ, L_XW, L_YZ, L_YW, L_ZW)
    - 4 bond coherences (C_X, C_Y, C_Z, C_W)
    - 4D FFT-based Helmholtz and Poisson solvers
    - 4D-to-3D projection readouts
    """

    def __init__(self, params: Optional[DETParams4D] = None):
        self.p = params or DETParams4D()
        N = self.p.N

        # Update lattice correction based on grid size
        self.p.eta_lattice = compute_lattice_correction_4d(N)

        # Shape for all 4D fields: (W, Z, Y, X)
        shape = (N, N, N, N)

        # Per-node state
        self.F = np.ones(shape, dtype=np.float64) * self.p.F_VAC
        self.q = np.zeros(shape, dtype=np.float64)
        self.a = np.ones(shape, dtype=np.float64)
        self.theta = np.random.uniform(0, 2 * np.pi, shape).astype(np.float64)

        # Per-bond linear momentum (4 axes)
        self.pi_X = np.zeros(shape, dtype=np.float64)
        self.pi_Y = np.zeros(shape, dtype=np.float64)
        self.pi_Z = np.zeros(shape, dtype=np.float64)
        self.pi_W = np.zeros(shape, dtype=np.float64)

        # Per-plaquette angular momentum (6 planes in 4D)
        self.L_XY = np.zeros(shape, dtype=np.float64)
        self.L_XZ = np.zeros(shape, dtype=np.float64)
        self.L_XW = np.zeros(shape, dtype=np.float64)
        self.L_YZ = np.zeros(shape, dtype=np.float64)
        self.L_YW = np.zeros(shape, dtype=np.float64)
        self.L_ZW = np.zeros(shape, dtype=np.float64)

        # Bond coherences (4 axes)
        self.C_X = np.ones(shape, dtype=np.float64) * self.p.C_init
        self.C_Y = np.ones(shape, dtype=np.float64) * self.p.C_init
        self.C_Z = np.ones(shape, dtype=np.float64) * self.p.C_init
        self.C_W = np.ones(shape, dtype=np.float64) * self.p.C_init

        self.sigma = np.ones(shape, dtype=np.float64)

        # Gravity fields (4D gradient has 4 components)
        self.b = np.zeros(shape, dtype=np.float64)
        self.Phi = np.zeros(shape, dtype=np.float64)
        self.gx = np.zeros(shape, dtype=np.float64)
        self.gy = np.zeros(shape, dtype=np.float64)
        self.gz = np.zeros(shape, dtype=np.float64)
        self.gw = np.zeros(shape, dtype=np.float64)

        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones(shape, dtype=np.float64)
        self.Delta_tau = np.ones(shape, dtype=np.float64) * self.p.DT

        # Time dilation tracking
        self.accumulated_proper_time = np.zeros(shape, dtype=np.float64)

        # Boundary operator diagnostics
        self.last_grace_injection = np.zeros(shape, dtype=np.float64)
        self.last_jubilee = np.zeros(shape, dtype=np.float64)
        self.total_grace_injected = 0.0
        self.total_jubilee = 0.0

        self._setup_fft_solvers()

    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers for 4D Helmholtz and Poisson solvers."""
        N = self.p.N
        # 4D wavenumber grids
        kw = np.fft.fftfreq(N) * N
        kz = np.fft.fftfreq(N) * N
        ky = np.fft.fftfreq(N) * N
        kx = np.fft.fftfreq(N) * N
        KW, KZ, KY, KX = np.meshgrid(kw, kz, ky, kx, indexing='ij')

        # 4D discrete Laplacian eigenvalues:
        # L_k = -4 * sum_{d} sin^2(pi * k_d / N)
        # = -2 * sum_{d} (1 - cos(2*pi*k_d/N))
        self.L_k = -4 * (np.sin(np.pi * KX / N)**2 +
                         np.sin(np.pi * KY / N)**2 +
                         np.sin(np.pi * KZ / N)**2 +
                         np.sin(np.pi * KW / N)**2)

        # Helmholtz operator: (L - alpha)
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12

        # Poisson operator (avoid zero mode)
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0, 0, 0, 0] = 1.0

    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        """Solve (L - alpha)*b = -alpha*q for baseline field in 4D."""
        source_k = fftn(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifftn(b_k))

    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        """Solve L*Phi = kappa*rho for gravitational potential in 4D."""
        source_k = fftn(source)
        source_k[0, 0, 0, 0] = 0  # Remove mean
        Phi_k = self.p.kappa_grav * self.p.eta_lattice * source_k / self.L_k_poisson
        Phi_k[0, 0, 0, 0] = 0
        return np.real(ifftn(Phi_k))

    def _compute_coherence_weighted_H(self) -> np.ndarray:
        """Compute Option B coherence-weighted load in 4D.

        H_i = sum_{j in N(i)} sqrt(C_ij) * sigma_ij
        In 4D, sum over 8 neighbors (±X, ±Y, ±Z, ±W).
        """
        H = np.zeros_like(self.F)
        for axis_name, C_bond in zip(AXES_4D, [self.C_X, self.C_Y, self.C_Z, self.C_W]):
            sp = SHIFT_PLUS[axis_name]
            sm = SHIFT_MINUS[axis_name]
            # Forward bond
            sigma_p = 0.5 * (self.sigma + sp(self.sigma))
            H += np.sqrt(C_bond) * sigma_p
            # Backward bond
            sigma_m = 0.5 * (self.sigma + sm(self.sigma))
            H += np.sqrt(sm(C_bond)) * sigma_m
        return H

    def _compute_gravity(self):
        """Compute gravitational fields from structure q in 4D.

        DET gravity is sourced by imbalance between local structure
        and Helmholtz baseline: rho = q - b.
        """
        if not self.p.gravity_enabled:
            self.gx = np.zeros_like(self.F)
            self.gy = np.zeros_like(self.F)
            self.gz = np.zeros_like(self.F)
            self.gw = np.zeros_like(self.F)
            return

        # Step 1: Solve Helmholtz for baseline
        self.b = self._solve_helmholtz(self.q)

        # Step 2: Compute relative source
        rho = self.q - self.b

        # Step 3: Solve Poisson for potential
        self.Phi = self._solve_poisson(rho)

        # Step 4: Compute gravitational force (negative gradient) in 4D
        self.gx = -0.5 * (Xp(self.Phi) - Xm(self.Phi))
        self.gy = -0.5 * (Yp(self.Phi) - Ym(self.Phi))
        self.gz = -0.5 * (Zp(self.Phi) - Zm(self.Phi))
        self.gw = -0.5 * (Wp(self.Phi) - Wm(self.Phi))

    def _sync_legacy_q_field(self):
        """Clip externally-written q to physical bounds."""
        self.q = np.clip(self.q, 0, 1)

    def _compute_drag_factor(self) -> np.ndarray:
        """Structural drag multiplier D = 1/(1 + lambda_P*q)."""
        return 1.0 / (1.0 + self.p.lambda_P * self.q)

    def compute_grace_injection(self, D: np.ndarray) -> np.ndarray:
        """Grace Injection per DET VI.5 (v6.2 simple form) in 4D."""
        p = self.p
        n = np.maximum(0, p.F_MIN_grace - self.F)
        w = self.a * n
        w_sum = periodic_local_sum_4d(w, p.R_boundary) + 1e-12
        I_g = D * w / w_sum
        return I_g

    def compute_jubilee(self, D: np.ndarray) -> np.ndarray:
        """Jubilee operator: reduces total q (never agency) in 4D.

        Node coherence proxy averages over all 8 bond coherences.
        """
        p = self.p

        # Node coherence proxy from local bonds (8 bonds in 4D)
        C_i = (
            self.C_X + Xm(self.C_X) +
            self.C_Y + Ym(self.C_Y) +
            self.C_Z + Zm(self.C_Z) +
            self.C_W + Wm(self.C_W)
        ) / 8.0

        # Local activation
        S_i = self.a * (C_i ** p.n_q) * (D / (D + p.D_0))
        dq = p.delta_q * S_i * self.Delta_tau

        if p.jubilee_energy_coupling:
            F_op = np.maximum(self.F - p.F_VAC, 0.0)
            energy_cap = F_op / (1.0 + F_op)
            dq = np.minimum(dq, energy_cap)
            dq = np.minimum(dq, self.q)

        return dq

    def add_packet(self, center: Tuple[int, int, int, int], mass: float = 10.0,
                   width: float = 3.0,
                   momentum: Tuple[float, float, float, float] = (0, 0, 0, 0),
                   initial_q: float = 0.0, initial_spin: float = 0.0):
        """Add a 4D Gaussian resource packet with optional momentum and spin.

        center: (w, z, y, x) coordinates
        momentum: (px, py, pz, pw) components
        """
        N = self.p.N
        w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
        cw, cz, cy, cx = center

        # Periodic distance in 4D
        dx = (x - cx + N / 2) % N - N / 2
        dy = (y - cy + N / 2) % N - N / 2
        dz = (z - cz + N / 2) % N - N / 2
        dw = (w - cw + N / 2) % N - N / 2

        r2 = dx**2 + dy**2 + dz**2 + dw**2
        envelope = np.exp(-0.5 * r2 / width**2)

        # Add resource
        self.F += mass * envelope

        # Boost coherence in packet region
        self.C_X = np.clip(self.C_X + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Y = np.clip(self.C_Y + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_Z = np.clip(self.C_Z + 0.5 * envelope, self.p.C_init, 1.0)
        self.C_W = np.clip(self.C_W + 0.5 * envelope, self.p.C_init, 1.0)

        # Add momentum
        px, py, pz, pw = momentum
        if px != 0 or py != 0 or pz != 0 or pw != 0:
            self.pi_X += px * envelope
            self.pi_Y += py * envelope
            self.pi_Z += pz * envelope
            self.pi_W += pw * envelope

        # Add structure (sources gravity)
        if initial_q > 0:
            q_add = initial_q * envelope
            self.q += q_add
            self.q = np.clip(self.q, 0, 1)

        # Add angular momentum (default: XY plane)
        if initial_spin != 0:
            self.L_XY += initial_spin * envelope

        self._clip()

    def add_spin(self, center: Tuple[int, int, int, int], spin: float = 1.0,
                 width: float = 4.0, plane: str = 'XY'):
        """Add initial angular momentum to a region in a specified plane.

        plane: one of 'XY', 'XZ', 'XW', 'YZ', 'YW', 'ZW'
        """
        N = self.p.N
        w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
        cw, cz, cy, cx = center

        dx = (x - cx + N / 2) % N - N / 2
        dy = (y - cy + N / 2) % N - N / 2
        dz = (z - cz + N / 2) % N - N / 2
        dw = (w - cw + N / 2) % N - N / 2

        r2 = dx**2 + dy**2 + dz**2 + dw**2
        envelope = np.exp(-0.5 * r2 / width**2)

        L_field = getattr(self, f'L_{plane}')
        L_field += spin * envelope
        setattr(self, f'L_{plane}', L_field)
        self._clip()

    def _clip(self):
        """Enforce physical bounds on all state variables."""
        p = self.p
        self.F = np.clip(self.F, p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)

        # Bond momenta
        for pi_name in ['pi_X', 'pi_Y', 'pi_Z', 'pi_W']:
            setattr(self, pi_name, np.clip(getattr(self, pi_name), -p.pi_max, p.pi_max))

        # Plaquette angular momenta
        for L_name in ['L_XY', 'L_XZ', 'L_XW', 'L_YZ', 'L_YW', 'L_ZW']:
            setattr(self, L_name, np.clip(getattr(self, L_name), -p.L_max, p.L_max))

    def step(self):
        """Execute one canonical DET v7 update step in 4D.

        Follows the exact 15-step canonical update order from the theory card.
        """
        p = self.p
        dk = p.DT
        N = p.N
        shape = (N, N, N, N)

        # Sync external legacy writes before canonical updates
        self._sync_legacy_q_field()

        # ============================================================
        # STEP 1-2: Compute gravitational baseline/potential
        # ============================================================
        self._compute_gravity()

        # ============================================================
        # STEP 3-4: Presence and proper time
        # ============================================================
        if p.coherence_weighted_H:
            H = self._compute_coherence_weighted_H()
        else:
            H = self.sigma

        if p.debt_temporal_enabled:
            debt_temporal_factor = 1.0 + p.zeta_temporal * self.q
        else:
            debt_temporal_factor = 1.0

        drag = self._compute_drag_factor()
        gamma_v = max(p.gamma_v, 1e-12)
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H) / gamma_v / debt_temporal_factor
        self.P = self.P * drag
        self.Delta_tau = self.P * dk

        # Track accumulated proper time
        self.accumulated_proper_time += self.Delta_tau

        # Bond-averaged proper time for each axis
        Delta_tau_bond = {}
        for ax in AXES_4D:
            sp = SHIFT_PLUS[ax]
            Delta_tau_bond[ax] = 0.5 * (self.Delta_tau + sp(self.Delta_tau))

        # Plaquette-averaged proper time for each plane
        Delta_tau_plaq = {}
        for plane in PLANES_4D:
            ax1, ax2 = plane[0], plane[1]
            sp1, sp2 = SHIFT_PLUS[ax1], SHIFT_PLUS[ax2]
            Delta_tau_plaq[plane] = 0.25 * (
                self.Delta_tau + sp1(self.Delta_tau) +
                sp2(self.Delta_tau) + sp1(sp2(self.Delta_tau))
            )

        # ============================================================
        # STEP 5: Flux computation (4D: 8 directional fluxes)
        # ============================================================
        # Initialize directional fluxes
        J_plus = {ax: np.zeros(shape, dtype=np.float64) for ax in AXES_4D}
        J_minus = {ax: np.zeros(shape, dtype=np.float64) for ax in AXES_4D}
        J_diff_plus = {ax: np.zeros(shape, dtype=np.float64) for ax in AXES_4D}

        # --- Diffusive flux (agency-gated) ---
        if p.diff_enabled:
            for ax in AXES_4D:
                sp = SHIFT_PLUS[ax]
                sm = SHIFT_MINUS[ax]
                C_bond = getattr(self, f'C_{ax}')

                # Classical pressure gradient
                classical_p = self.F - sp(self.F)
                classical_m = self.F - sm(self.F)

                # Agency gate: g^(a)_ij = sqrt(a_i * a_j)
                g_p = np.sqrt(self.a * sp(self.a))
                g_m = np.sqrt(self.a * sm(self.a))

                # Conductivity from coherence
                cond_p = self.sigma * (C_bond + 1e-4)
                cond_m = self.sigma * (sm(C_bond) + 1e-4)

                # Debt conductivity gate
                if p.debt_conductivity_enabled:
                    g_q_p = 1.0 / (1.0 + p.xi_conductivity * (self.q + sp(self.q)))
                    g_q_m = 1.0 / (1.0 + p.xi_conductivity * (self.q + sm(self.q)))
                    cond_p *= g_q_p
                    cond_m *= g_q_m

                J_diff_p = g_p * cond_p * classical_p
                J_diff_m = g_m * cond_m * classical_m

                J_diff_plus[ax] = J_diff_p
                J_plus[ax] += J_diff_p
                J_minus[ax] += J_diff_m

        # --- Linear momentum flux (F-weighted) ---
        if p.momentum_enabled:
            for ax in AXES_4D:
                sp = SHIFT_PLUS[ax]
                sm = SHIFT_MINUS[ax]
                pi_bond = getattr(self, f'pi_{ax}')

                F_avg_p = 0.5 * (self.F + sp(self.F))
                F_avg_m = 0.5 * (self.F + sm(self.F))

                J_plus[ax] += p.mu_pi * self.sigma * pi_bond * F_avg_p
                J_minus[ax] += -p.mu_pi * self.sigma * sm(pi_bond) * F_avg_m

        # --- Floor repulsion (agency-independent) ---
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            for ax in AXES_4D:
                sp = SHIFT_PLUS[ax]
                sm = SHIFT_MINUS[ax]
                classical_p = self.F - sp(self.F)
                classical_m = self.F - sm(self.F)
                J_plus[ax] += p.eta_floor * self.sigma * (s + sp(s)) * classical_p
                J_minus[ax] += p.eta_floor * self.sigma * (s + sm(s)) * classical_m

        # --- Gravitational flux ---
        if p.gravity_enabled:
            g_fields = {'X': self.gx, 'Y': self.gy, 'Z': self.gz, 'W': self.gw}
            for ax in AXES_4D:
                sp = SHIFT_PLUS[ax]
                sm = SHIFT_MINUS[ax]
                g_ax = g_fields[ax]

                g_bond_p = 0.5 * (g_ax + sp(g_ax))
                g_bond_m = 0.5 * (g_ax + sm(g_ax))

                F_avg_p = 0.5 * (self.F + sp(self.F))
                F_avg_m = 0.5 * (self.F + sm(self.F))

                J_plus[ax] += p.mu_grav * self.sigma * g_bond_p * F_avg_p
                J_minus[ax] += p.mu_grav * self.sigma * g_bond_m * F_avg_m

        # --- Rotational flux from angular momentum ---
        if p.angular_momentum_enabled:
            # For each plaquette plane (a,b), angular momentum L_ab generates
            # rotational flux in the a-direction from curl along b, and vice versa.
            # J_rot_a += mu_L * sigma * F_avg_a * (L_ab - shift_minus_b(L_ab))
            # J_rot_b -= mu_L * sigma * F_avg_b * (L_ab - shift_minus_a(L_ab))
            J_rot = {ax: np.zeros(shape, dtype=np.float64) for ax in AXES_4D}

            for plane in PLANES_4D:
                ax_a, ax_b = plane[0], plane[1]
                L_ab = getattr(self, f'L_{plane}')
                sm_a = SHIFT_MINUS[ax_a]
                sm_b = SHIFT_MINUS[ax_b]
                sp_a = SHIFT_PLUS[ax_a]
                sp_b = SHIFT_PLUS[ax_b]

                F_avg_a = 0.5 * (self.F + sp_a(self.F))
                F_avg_b = 0.5 * (self.F + sp_b(self.F))

                J_rot[ax_a] += p.mu_L * self.sigma * F_avg_a * (L_ab - sm_b(L_ab))
                J_rot[ax_b] -= p.mu_L * self.sigma * F_avg_b * (L_ab - sm_a(L_ab))

            for ax in AXES_4D:
                sm = SHIFT_MINUS[ax]
                J_plus[ax] += J_rot[ax]
                J_minus[ax] -= sm(J_rot[ax])

        # ============================================================
        # STEP 6: Conservative limiter
        # ============================================================
        total_outflow = sum(np.maximum(0, J_plus[ax]) + np.maximum(0, J_minus[ax])
                           for ax in AXES_4D)
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))

        J_plus_lim = {}
        J_minus_lim = {}
        J_diff_plus_scaled = {}
        for ax in AXES_4D:
            J_plus_lim[ax] = np.where(J_plus[ax] > 0, J_plus[ax] * scale, J_plus[ax])
            J_minus_lim[ax] = np.where(J_minus[ax] > 0, J_minus[ax] * scale, J_minus[ax])
            J_diff_plus_scaled[ax] = np.where(
                J_diff_plus[ax] > 0, J_diff_plus[ax] * scale, J_diff_plus[ax])

        # Dissipation (for grace injection)
        D_dissip = sum(np.abs(J_plus_lim[ax]) + np.abs(J_minus_lim[ax])
                       for ax in AXES_4D) * self.Delta_tau

        # ============================================================
        # STEP 7: Resource update
        # ============================================================
        outflow = sum(J_plus_lim[ax] * self.Delta_tau + J_minus_lim[ax] * self.Delta_tau
                      for ax in AXES_4D)
        inflow = np.zeros(shape, dtype=np.float64)
        for ax in AXES_4D:
            sm = SHIFT_MINUS[ax]
            sp = SHIFT_PLUS[ax]
            inflow += sm(J_plus_lim[ax] * self.Delta_tau)
            inflow += sp(J_minus_lim[ax] * self.Delta_tau)

        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)

        # ============================================================
        # STEP 8: Boundary Grace operator
        # ============================================================
        if p.boundary_enabled and p.grace_enabled:
            I_g = self.compute_grace_injection(D_dissip)
            self.F = self.F + I_g
            self.last_grace_injection = I_g.copy()
            self.total_grace_injected += np.sum(I_g)
        else:
            self.last_grace_injection = np.zeros(shape)

        # ============================================================
        # STEP 9: Momentum update (4 axes)
        # ============================================================
        if p.momentum_enabled:
            g_fields = {'X': self.gx, 'Y': self.gy, 'Z': self.gz, 'W': self.gw}
            for ax in AXES_4D:
                sp = SHIFT_PLUS[ax]
                pi_bond = getattr(self, f'pi_{ax}')
                dt_bond = Delta_tau_bond[ax]

                decay = np.maximum(0.0, 1.0 - p.lambda_pi * dt_bond)

                # Charging from diffusive flux
                dpi_diff = p.alpha_pi * J_diff_plus_scaled[ax] * dt_bond

                # Charging from gravity
                if p.gravity_enabled:
                    g_ax = g_fields[ax]
                    g_bond = 0.5 * (g_ax + sp(g_ax))
                    dpi_grav = p.beta_g * g_bond * dt_bond
                else:
                    dpi_grav = 0

                setattr(self, f'pi_{ax}', decay * pi_bond + dpi_diff + dpi_grav)

        # ============================================================
        # STEP 10: Angular momentum update (6 planes)
        # ============================================================
        if p.angular_momentum_enabled:
            for plane in PLANES_4D:
                ax_a, ax_b = plane[0], plane[1]
                sp_a = SHIFT_PLUS[ax_a]
                sp_b = SHIFT_PLUS[ax_b]
                pi_a = getattr(self, f'pi_{ax_a}')
                pi_b = getattr(self, f'pi_{ax_b}')

                # Curl of momentum (plaquette circulation)
                curl = pi_a + sp_a(pi_b) - sp_b(pi_a) - pi_b

                L_ab = getattr(self, f'L_{plane}')
                dt_plaq = Delta_tau_plaq[plane]
                decay = np.maximum(0.0, 1.0 - p.lambda_L * dt_plaq)

                setattr(self, f'L_{plane}', decay * L_ab + p.alpha_L * curl * dt_plaq)

        # ============================================================
        # STEP 11: Structure update (q accumulation)
        # ============================================================
        if p.q_enabled:
            dq_lock = p.alpha_q * np.maximum(0, -dF)
            self.q = np.clip(self.q + dq_lock, 0, 1)

        # ============================================================
        # STEP 12: Boundary Jubilee / Forgiveness (q)
        # ============================================================
        if p.jubilee_enabled:
            dq_jubilee = self.compute_jubilee(D_dissip)
            self.q = np.clip(self.q - dq_jubilee, 0, 1)
            self.last_jubilee = dq_jubilee.copy()
            self.total_jubilee += float(np.sum(dq_jubilee))
        else:
            self.last_jubilee = np.zeros(shape)

        # ============================================================
        # STEP 13: Agency update (v7 canonical, no structural ceiling)
        # ============================================================
        if p.agency_dynamic:
            # Compute local average presence (self + 8 neighbors in 4D)
            P_local = self.P.copy()
            for ax in AXES_4D:
                P_local = P_local + np.roll(self.P, 1, axis=AXIS_INDICES[ax]) + \
                          np.roll(self.P, -1, axis=AXIS_INDICES[ax])
            P_local = P_local / 9.0  # Self + 8 neighbors

            # Compute average coherence at each node (8 bonds)
            C_avg = (
                self.C_X + Xm(self.C_X) +
                self.C_Y + Ym(self.C_Y) +
                self.C_Z + Zm(self.C_Z) +
                self.C_W + Wm(self.C_W)
            ) / 8.0

            # Coherence-gated drive strength
            gamma = p.gamma_a_max * (C_avg ** p.gamma_a_power)

            # Relational drive
            delta_a_drive = gamma * (self.P - P_local)

            if p.epsilon_a > 0:
                xi = np.clip(
                    np.random.normal(0.0, p.epsilon_a, size=self.a.shape),
                    -p.epsilon_a, p.epsilon_a)
            else:
                xi = 0.0

            self.a = self.a + p.beta_a * (p.a0 - self.a) + delta_a_drive + xi
            self.a = np.clip(self.a, 0.0, 1.0)

        # ============================================================
        # STEP 14: Coherence and sigma dynamics
        # ============================================================
        J_mag = sum(np.abs(J_plus_lim[ax]) for ax in AXES_4D) / 4.0

        if p.coherence_dynamic:
            for ax in AXES_4D:
                sp = SHIFT_PLUS[ax]
                C_bond = getattr(self, f'C_{ax}')

                if p.debt_decoherence_enabled:
                    q_bond = 0.5 * (self.q + sp(self.q))
                    lambda_C_ax = p.lambda_C * (1.0 + p.theta_decoherence * q_bond)
                else:
                    lambda_C_ax = p.lambda_C

                C_new = np.clip(
                    C_bond + p.alpha_C * np.abs(J_plus_lim[ax]) * self.Delta_tau
                    - lambda_C_ax * C_bond * self.Delta_tau,
                    p.C_init, 1.0)
                setattr(self, f'C_{ax}', C_new)

        if p.sigma_dynamic:
            self.sigma = 1.0 + 0.1 * np.log(1.0 + J_mag)

        # ============================================================
        # STEP 15: Clip and advance
        # ============================================================
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

    def total_angular_momentum(self) -> Dict[str, float]:
        """Total angular momentum for each plane."""
        return {plane: float(np.sum(getattr(self, f'L_{plane}')))
                for plane in PLANES_4D}

    def potential_energy(self) -> float:
        """Gravitational potential energy."""
        return float(np.sum(self.F * self.Phi))

    def center_of_mass(self) -> Tuple[float, float, float, float]:
        """Center of mass position (x, y, z, w)."""
        N = self.p.N
        w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
        total = np.sum(self.F) + 1e-9
        return (float(np.sum(x * self.F) / total),
                float(np.sum(y * self.F) / total),
                float(np.sum(z * self.F) / total),
                float(np.sum(w * self.F) / total))

    def find_blobs(self, threshold_ratio: float = 50.0) -> List[Dict]:
        """Find distinct blobs using connected components in 4D."""
        threshold = self.p.F_VAC * threshold_ratio
        above = self.F > threshold
        # 4D connectivity structure (face-sharing = 8 neighbors)
        struct = np.zeros((3, 3, 3, 3), dtype=int)
        struct[1, 1, 1, :] = 1
        struct[1, 1, :, 1] = 1
        struct[1, :, 1, 1] = 1
        struct[:, 1, 1, 1] = 1
        labeled, num = label(above, structure=struct)
        N = self.p.N
        w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]

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
                'w': float(np.sum(w[mask] * weights) / total_mass),
                'mass': total_mass,
                'size': int(np.sum(mask))
            })
        blobs.sort(key=lambda c: -c['mass'])
        return blobs

    def separation(self) -> Tuple[float, int]:
        """Find 4D separation between two largest blobs."""
        blobs = self.find_blobs()
        if len(blobs) < 2:
            return 0.0, len(blobs)
        N = self.p.N
        dx = blobs[1]['x'] - blobs[0]['x']
        dy = blobs[1]['y'] - blobs[0]['y']
        dz = blobs[1]['z'] - blobs[0]['z']
        dw = blobs[1]['w'] - blobs[0]['w']
        dx = dx - N if dx > N / 2 else (dx + N if dx < -N / 2 else dx)
        dy = dy - N if dy > N / 2 else (dy + N if dy < -N / 2 else dy)
        dz = dz - N if dz > N / 2 else (dz + N if dz < -N / 2 else dz)
        dw = dw - N if dw > N / 2 else (dw + N if dw < -N / 2 else dw)
        return float(np.sqrt(dx**2 + dy**2 + dz**2 + dw**2)), len(blobs)

    def rot_flux_magnitude(self) -> float:
        """Magnitude of rotational flux across all 6 planes."""
        if not self.p.angular_momentum_enabled:
            return 0.0
        total = 0.0
        for plane in PLANES_4D:
            ax_a, ax_b = plane[0], plane[1]
            L_ab = getattr(self, f'L_{plane}')
            sm_b = SHIFT_MINUS[ax_b]
            sp_a = SHIFT_PLUS[ax_a]
            F_avg_a = 0.5 * (self.F + sp_a(self.F))
            J_rot = self.p.mu_L * self.sigma * F_avg_a * (L_ab - sm_b(L_ab))
            total += float(np.sum(np.abs(J_rot)))
        return total

    # ==================== 4D-to-3D PROJECTION ====================

    def project_3d_slice(self, w_index: int = 0) -> Dict[str, np.ndarray]:
        """Extract a 3D slice at fixed w-coordinate.

        Returns dict of 3D arrays for visualization/analysis.
        """
        return {
            'F': self.F[w_index].copy(),
            'q': self.q[w_index].copy(),
            'a': self.a[w_index].copy(),
            'P': self.P[w_index].copy(),
            'Phi': self.Phi[w_index].copy(),
            'sigma': self.sigma[w_index].copy(),
        }

    def project_3d_max(self) -> Dict[str, np.ndarray]:
        """Maximum-intensity projection along W axis.

        Returns 3D arrays where each voxel is the max over all W values.
        """
        return {
            'F': np.max(self.F, axis=0),
            'q': np.max(self.q, axis=0),
            'a': np.min(self.a, axis=0),  # min agency is more informative
            'P': np.max(self.P, axis=0),
            'Phi': np.min(self.Phi, axis=0),  # min potential (deepest well)
        }

    def project_3d_sum(self) -> Dict[str, np.ndarray]:
        """Sum projection along W axis (column-density analogue).

        Returns 3D arrays where each voxel is the sum over all W values.
        """
        return {
            'F': np.sum(self.F, axis=0),
            'q': np.sum(self.q, axis=0),
            'P': np.sum(self.P, axis=0),
        }

    def w_profile(self, x: int, y: int, z: int) -> Dict[str, np.ndarray]:
        """Extract 1D profile along W at fixed (x, y, z).

        Useful for studying 4th-dimension structure.
        """
        return {
            'F': self.F[:, z, y, x].copy(),
            'q': self.q[:, z, y, x].copy(),
            'a': self.a[:, z, y, x].copy(),
            'P': self.P[:, z, y, x].copy(),
        }

    def dimension_comparison(self) -> Dict[str, float]:
        """Compare gradient magnitudes across dimensions.

        Useful for testing whether 4D dynamics are isotropic or
        show dimensional asymmetry.
        """
        grad_x = float(np.mean(np.abs(self.F - Xp(self.F))))
        grad_y = float(np.mean(np.abs(self.F - Yp(self.F))))
        grad_z = float(np.mean(np.abs(self.F - Zp(self.F))))
        grad_w = float(np.mean(np.abs(self.F - Wp(self.F))))
        return {'grad_X': grad_x, 'grad_Y': grad_y,
                'grad_Z': grad_z, 'grad_W': grad_w}


# ============================================================
# BUILT-IN TEST SUITE
# ============================================================

def test_4d_gravity_vacuum(verbose: bool = True) -> bool:
    """Gravity has no effect in vacuum (q=0) in 4D."""
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: 4D Gravity in Vacuum")
        print("=" * 60)

    params = DETParams4D(N=8, gravity_enabled=True, q_enabled=False, boundary_enabled=False)
    sim = DETCollider4D(params)

    for _ in range(100):
        sim.step()

    max_g = (np.max(np.abs(sim.gx)) + np.max(np.abs(sim.gy)) +
             np.max(np.abs(sim.gz)) + np.max(np.abs(sim.gw)))
    max_Phi = np.max(np.abs(sim.Phi))

    passed = max_g < 1e-10 and max_Phi < 1e-10

    if verbose:
        print(f"  Max |g|: {max_g:.2e}")
        print(f"  Max |Phi|: {max_Phi:.2e}")
        print(f"  Lattice correction eta: {sim.p.eta_lattice:.3f}")
        print(f"  Vacuum gravity {'PASSED' if passed else 'FAILED'}")

    return passed


def test_4d_gravitational_binding(verbose: bool = True) -> bool:
    """Test gravitational binding in 4D.

    In 4D, higher connectivity (8 neighbors vs 6) causes faster diffusion.
    Binding is tested by:
    1. Placing two well-separated packets with opposing transverse momenta
    2. Verifying that gravitational potential energy becomes increasingly negative
    3. Verifying that the system remains gravitationally bound (PE stays negative)

    This is a key 4D research observable: 4D adjacency increases diffusion,
    making binding harder. The test checks whether gravity can still overcome
    the enhanced dispersion.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: 4D Gravitational Binding")
        print("=" * 60)

    params = DETParams4D(
        N=16, DT=0.015, F_VAC=0.001, F_MIN=0.0,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        kappa_grav=7.0, mu_grav=2.5, beta_g=12.5,
    )

    sim = DETCollider4D(params)

    center = params.N // 2
    # Place packets with structure to source gravity
    sim.add_packet((center, center, center, center - 3),
                   mass=12.0, width=2.0,
                   momentum=(0, 0, 0, 0.15), initial_q=0.5)
    sim.add_packet((center, center, center, center + 3),
                   mass=12.0, width=2.0,
                   momentum=(0, 0, 0, -0.15), initial_q=0.5)

    pe_values = []
    for t in range(400):
        pe = sim.potential_energy()
        pe_values.append(pe)

        if verbose and t % 100 == 0:
            sep, nblobs = sim.separation()
            print(f"  t={t}: PE={pe:.3f}, blobs={nblobs}, total_q={sim.total_q():.3f}")

        sim.step()

    pe_values = np.array(pe_values)
    # Binding criterion: PE should be significantly negative and deepening
    pe_late = float(np.mean(pe_values[-100:]))
    pe_early = float(np.mean(pe_values[:50]))
    pe_deepened = pe_late < pe_early
    pe_negative = pe_late < -1.0

    passed = pe_negative and pe_deepened

    if verbose:
        print(f"\n  Early PE: {pe_early:.3f}")
        print(f"  Late PE: {pe_late:.3f}")
        print(f"  PE deepened: {pe_deepened}")
        print(f"  PE negative: {pe_negative}")
        print(f"  Binding {'PASSED' if passed else 'FAILED'}")

    return passed


def test_4d_time_dilation(verbose: bool = True) -> bool:
    """Test gravitational time dilation in 4D."""
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: 4D Time Dilation")
        print("=" * 60)

    params = DETParams4D(
        N=12, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False
    )
    sim = DETCollider4D(params)

    center = params.N // 2
    sim.add_packet((center, center, center, center), mass=20.0, width=2.0, initial_q=0.5)

    for _ in range(50):
        sim.step()

    F_center = sim.F[center, center, center, center]
    F_edge = sim.F[center, center, center, min(center + 4, params.N - 1)]
    P_center = sim.P[center, center, center, center]
    P_edge = sim.P[center, center, center, min(center + 4, params.N - 1)]

    time_dilated = P_center < P_edge

    passed = time_dilated

    if verbose:
        print(f"  F at center: {F_center:.4f}")
        print(f"  F at edge: {F_edge:.4f}")
        print(f"  P at center: {P_center:.6f}")
        print(f"  P at edge: {P_edge:.6f}")
        print(f"  Time dilated (P_center < P_edge): {time_dilated}")
        print(f"  Time dilation {'PASSED' if passed else 'FAILED'}")

    return passed


def test_4d_conservation(verbose: bool = True) -> bool:
    """Test resource conservation in 4D (no boundary operators)."""
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: 4D Resource Conservation")
        print("=" * 60)

    params = DETParams4D(
        N=8, DT=0.02,
        gravity_enabled=False, q_enabled=False,
        boundary_enabled=False,
        momentum_enabled=False, angular_momentum_enabled=False,
        floor_enabled=False
    )
    sim = DETCollider4D(params)

    center = params.N // 2
    sim.add_packet((center, center, center, center), mass=10.0, width=1.5)

    initial_mass = sim.total_mass()

    for _ in range(200):
        sim.step()

    final_mass = sim.total_mass()
    rel_error = abs(final_mass - initial_mass) / initial_mass

    passed = rel_error < 0.01

    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Relative error: {rel_error:.6f}")
        print(f"  Conservation {'PASSED' if passed else 'FAILED'}")

    return passed


def test_4d_isotropy(verbose: bool = True) -> bool:
    """Test that 4D dynamics are isotropic (no preferred dimension)."""
    if verbose:
        print("\n" + "=" * 60)
        print("TEST: 4D Isotropy")
        print("=" * 60)

    params = DETParams4D(
        N=10, DT=0.02,
        gravity_enabled=False, q_enabled=False,
        boundary_enabled=False,
        momentum_enabled=False, angular_momentum_enabled=False,
        floor_enabled=False
    )
    sim = DETCollider4D(params)

    center = params.N // 2
    sim.add_packet((center, center, center, center), mass=10.0, width=2.0)

    for _ in range(100):
        sim.step()

    grads = sim.dimension_comparison()

    # All gradient magnitudes should be similar (within 20%)
    vals = list(grads.values())
    mean_grad = np.mean(vals)
    max_dev = max(abs(v - mean_grad) / (mean_grad + 1e-12) for v in vals)

    passed = max_dev < 0.20

    if verbose:
        for k, v in grads.items():
            print(f"  {k}: {v:.6f}")
        print(f"  Max deviation from mean: {max_dev * 100:.1f}%")
        print(f"  Isotropy {'PASSED' if passed else 'FAILED'}")

    return passed


def run_4d_test_suite():
    """Run the complete 4D collider test suite."""
    print("=" * 70)
    print("DET v7.0 4D COLLIDER - TEST SUITE")
    print("=" * 70)

    results = {}
    results['vacuum_gravity'] = test_4d_gravity_vacuum(verbose=True)
    results['conservation'] = test_4d_conservation(verbose=True)
    results['isotropy'] = test_4d_isotropy(verbose=True)
    results['time_dilation'] = test_4d_time_dilation(verbose=True)
    results['binding'] = test_4d_gravitational_binding(verbose=True)

    print("\n" + "=" * 70)
    print("4D TEST SUMMARY")
    print("=" * 70)

    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return results


if __name__ == "__main__":
    run_4d_test_suite()
