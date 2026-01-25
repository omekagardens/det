"""
Lattice Substrate for DET Collider
===================================

Native DET lattice implementation for physics simulation.
This provides the substrate infrastructure for ColliderCreature.ex.

The lattice is a regular grid of nodes with periodic boundary conditions.
Each node has DET state (F, a, q, sigma, P, tau) and bonds to neighbors
carry momentum (pi) and coherence (C).

Design:
    - Structure-of-Arrays layout matching C substrate (substrate_types.h)
    - Per-dimension bond arrays (1D: 1, 2D: 2, 3D: 3 directions)
    - Periodic boundary conditions via modular arithmetic
    - FFT-based gravity solver using NumPy (upgrade to Metal MPS later)

Architecture Note:
    This is a reference implementation in Python/NumPy. The design matches
    the C substrate's NodeArrays/BondArrays layout so it can be ported to
    C/Metal when performance optimization is needed. The fields are:

    NodeArrays: F, q, a, sigma, P, tau, cos_theta, sin_theta, k, r, flags
    BondArrays: node_i, node_j, C, pi, sigma, flags

    For lattice topology, we use dense arrays indexed by grid position
    rather than sparse bond lists. This is more efficient for regular grids.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import IntEnum


class LatticeDirection(IntEnum):
    """Direction indices for neighbor/bond access."""
    # 1D
    RIGHT = 0    # +X
    LEFT = 1     # -X (implicit, use RIGHT with roll)
    # 2D adds
    DOWN = 1     # +Y (in 2D/3D, reuse index 1)
    UP = 2       # -Y
    # 3D adds
    FORWARD = 2  # +Z
    BACK = 3     # -Z


@dataclass
class LatticeParams:
    """DET v6.3 physics parameters for lattice simulation.

    Default values match Theory Card v6.3 Section VII.2.
    """
    # Grid
    dim: int = 1
    N: int = 100
    dt: float = 0.02  # DT from VII.2

    # Core state
    F_VAC: float = 0.01  # VII.2
    F_MIN: float = 0.0   # VII.2
    C_init: float = 0.15  # VII.2 (was 0.3)

    # Momentum (IV.4) - VII.2 defaults
    momentum_enabled: bool = True
    alpha_pi: float = 0.12  # VII.2 (was 0.10)
    lambda_pi: float = 0.008  # VII.2 (was 0.02)
    mu_pi: float = 0.35  # VII.2 (was 0.30)
    pi_max: float = 3.0  # VII.2

    # Structure (q-locking) - VII.2
    q_enabled: bool = True
    alpha_q: float = 0.012  # VII.2 (was 0.015)

    # Agency (VI.2B v6.4) - VII.2 defaults
    agency_enabled: bool = True
    lambda_a: float = 30.0  # VII.2
    beta_a: float = 0.2  # VII.2
    gamma_a_max: float = 0.15  # VII.2
    gamma_a_power: float = 2.0  # VII.2

    # Floor repulsion (IV.6) - VII.2 defaults
    floor_enabled: bool = True
    eta_floor: float = 0.12  # VII.2 (was 0.15)
    F_core: float = 5.0  # VII.2
    floor_power: float = 2.0  # VII.2

    # Gravity (V.1-V.3) - VII.2 defaults
    gravity_enabled: bool = True
    alpha_grav: float = 0.02  # VII.2 (was 0.05) - screening parameter
    kappa_grav: float = 5.0  # VII.2 (was 2.0) - Poisson coupling
    mu_grav: float = 2.0  # VII.2 (was 1.0) - gravity mobility
    beta_g: float = 10.0  # VII.2: 5.0 × μ_g = 10.0 (was 5.0)

    # Grace (VI.6) - VII.2 defaults
    grace_enabled: bool = True
    F_MIN_grace: float = 0.05  # VII.2
    R_grace: int = 2  # VII.2: R_boundary = 2 (was 3)

    # Coherence dynamics - VII.2 defaults
    coherence_enabled: bool = True
    alpha_C: float = 0.04  # VII.2
    lambda_C: float = 0.002  # VII.2

    # Numerical
    outflow_limit: float = 0.25


def compute_lattice_correction(dim: int, N: int) -> float:
    """Compute lattice correction factor eta."""
    if dim == 1:
        if N <= 64: return 0.92
        if N <= 128: return 0.96
        if N <= 256: return 0.98
        return 0.99
    elif dim == 2:
        if N <= 32: return 0.88
        if N <= 64: return 0.94
        if N <= 96: return 0.96
        return 0.97
    else:  # dim == 3
        if N <= 32: return 0.901
        if N <= 64: return 0.955
        if N <= 96: return 0.968
        return 0.975


class DETLattice:
    """
    Native DET Lattice - Structure-of-Arrays implementation.

    This is the substrate layer for collider physics. All arrays are
    shaped for the lattice dimension:
        1D: (N,)
        2D: (N, N)
        3D: (N, N, N)

    Bond arrays have an extra leading dimension for direction:
        1D: (1, N)     - bonds in +X direction
        2D: (2, N, N)  - bonds in +X, +Y directions
        3D: (3, N, N, N) - bonds in +X, +Y, +Z directions
    """

    def __init__(self, params: Optional[LatticeParams] = None):
        self.p = params or LatticeParams()
        self.dim = self.p.dim
        self.N = self.p.N

        # Compute lattice correction
        self.eta = compute_lattice_correction(self.dim, self.N)

        # Node shape
        if self.dim == 1:
            shape = (self.N,)
        elif self.dim == 2:
            shape = (self.N, self.N)
        else:
            shape = (self.N, self.N, self.N)
        self.shape = shape

        # Per-node state (Structure-of-Arrays)
        self.F = np.full(shape, self.p.F_VAC, dtype=np.float64)
        self.q = np.zeros(shape, dtype=np.float64)
        self.a = np.ones(shape, dtype=np.float64)
        self.sigma = np.ones(shape, dtype=np.float64)
        self.P = np.ones(shape, dtype=np.float64)
        self.Delta_tau = np.full(shape, self.p.dt, dtype=np.float64)

        # Per-bond state (one array per direction)
        bond_shape = (self.dim,) + shape
        self.pi = np.zeros(bond_shape, dtype=np.float64)  # Momentum
        self.C = np.full(bond_shape, self.p.C_init, dtype=np.float64)  # Coherence

        # Gravity fields
        self.Phi = np.zeros(shape, dtype=np.float64)  # Potential
        self.g = np.zeros(bond_shape, dtype=np.float64)  # Gradient (per direction)

        # Diagnostics
        self.step_count = 0
        self.time = 0.0
        self.total_grace = 0.0

        # Precompute FFT kernels
        self._setup_fft_kernels()

    def _setup_fft_kernels(self):
        """Precompute FFT wavenumber arrays for gravity solver."""
        N = self.N
        dim = self.dim

        if dim == 1:
            k = np.fft.fftfreq(N) * N
            self.L_k = -4 * np.sin(np.pi * k / N)**2
        elif dim == 2:
            kx = np.fft.fftfreq(N) * N
            ky = np.fft.fftfreq(N) * N
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            self.L_k = -4 * (np.sin(np.pi * KX / N)**2 + np.sin(np.pi * KY / N)**2)
        else:  # dim == 3
            kx = np.fft.fftfreq(N) * N
            ky = np.fft.fftfreq(N) * N
            kz = np.fft.fftfreq(N) * N
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            self.L_k = -4 * (np.sin(np.pi * KX / N)**2 +
                            np.sin(np.pi * KY / N)**2 +
                            np.sin(np.pi * KZ / N)**2)

        # Helmholtz kernel
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12

        # Poisson kernel
        self.L_k_poisson = self.L_k.copy()
        if dim == 1:
            self.L_k_poisson[0] = 1.0
        elif dim == 2:
            self.L_k_poisson[0, 0] = 1.0
        else:
            self.L_k_poisson[0, 0, 0] = 1.0

    def _fft_forward(self, field: np.ndarray) -> np.ndarray:
        """Forward FFT on lattice field."""
        if self.dim == 1:
            return np.fft.fft(field)
        elif self.dim == 2:
            return np.fft.fft2(field)
        else:
            return np.fft.fftn(field)

    def _fft_inverse(self, field_k: np.ndarray) -> np.ndarray:
        """Inverse FFT on lattice field."""
        if self.dim == 1:
            return np.real(np.fft.ifft(field_k))
        elif self.dim == 2:
            return np.real(np.fft.ifft2(field_k))
        else:
            return np.real(np.fft.ifftn(field_k))

    def _solve_gravity(self):
        """Compute gravitational fields from structure q."""
        if not self.p.gravity_enabled:
            self.g.fill(0)
            return

        # 1. Helmholtz baseline: (L - alpha)b = -alpha * q
        q_k = self._fft_forward(self.q)
        b_k = -self.p.alpha_grav * q_k / self.H_k
        b = self._fft_inverse(b_k)

        # 2. Relative source: rho = q - b
        rho = self.q - b

        # 3. Poisson potential with lattice correction
        rho_k = self._fft_forward(rho)
        if self.dim == 1:
            rho_k[0] = 0
        elif self.dim == 2:
            rho_k[0, 0] = 0
        else:
            rho_k[0, 0, 0] = 0

        Phi_k = self.p.kappa_grav * self.eta * rho_k / self.L_k_poisson
        if self.dim == 1:
            Phi_k[0] = 0
        elif self.dim == 2:
            Phi_k[0, 0] = 0
        else:
            Phi_k[0, 0, 0] = 0

        self.Phi = self._fft_inverse(Phi_k)

        # 4. Gradient for g (central difference)
        for d in range(self.dim):
            self.g[d] = -0.5 * (np.roll(self.Phi, -1, axis=d) -
                                np.roll(self.Phi, 1, axis=d))

    def _roll(self, arr: np.ndarray, shift: int, dim: int) -> np.ndarray:
        """Roll array along dimension (periodic BC)."""
        return np.roll(arr, shift, axis=dim)

    def add_packet(self, center: Tuple, mass: float = 5.0, width: float = 5.0,
                   momentum: Optional[Tuple] = None, initial_q: float = 0.0):
        """Add a Gaussian resource packet."""
        # Create coordinate grids
        if self.dim == 1:
            x = np.arange(self.N)
            r2 = (x - center[0])**2
        elif self.dim == 2:
            y, x = np.mgrid[0:self.N, 0:self.N]
            r2 = (x - center[1])**2 + (y - center[0])**2
        else:
            z, y, x = np.mgrid[0:self.N, 0:self.N, 0:self.N]
            r2 = (x - center[2])**2 + (y - center[1])**2 + (z - center[0])**2

        envelope = np.exp(-0.5 * r2 / width**2)

        # Add resource
        self.F += mass * envelope

        # Boost coherence
        for d in range(self.dim):
            self.C[d] = np.clip(self.C[d] + 0.5 * envelope, self.p.C_init, 1.0)

        # Add momentum
        if momentum is not None:
            mom_env = np.exp(-0.5 * r2 / (width * 2)**2)
            for d in range(min(len(momentum), self.dim)):
                self.pi[d] += momentum[d] * mom_env

        # Add structure
        if initial_q > 0:
            self.q += initial_q * envelope

        self._clip()

    def _clip(self):
        """Enforce physical bounds."""
        p = self.p
        self.F = np.clip(self.F, p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi = np.clip(self.pi, -p.pi_max, p.pi_max)
        self.C = np.clip(self.C, 0, 1)

    def step(self):
        """Execute one physics step (DET v6.3/v6.4)."""
        p = self.p
        dim = self.dim
        dt = p.dt

        # STEP 0: Gravity
        self._solve_gravity()

        # STEP 1: Presence
        # P = a·σ / (1 + F) / (1 + H)
        H = self.sigma  # Simplified load (Option A)
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dt

        # STEP 2: Compute fluxes in BOTH directions (matching det_v6_3)
        # J_R[d] = flux from node i to i+1 in dimension d
        # J_L[d] = flux from node i to i-1 in dimension d
        J_R = np.zeros((dim,) + self.shape, dtype=np.float64)  # +d direction
        J_L = np.zeros((dim,) + self.shape, dtype=np.float64)  # -d direction
        J_diff_R = np.zeros((dim,) + self.shape, dtype=np.float64)

        for d in range(dim):
            # Neighbors
            F_right = self._roll(self.F, -1, d)  # F at i+1
            F_left = self._roll(self.F, 1, d)    # F at i-1
            a_right = self._roll(self.a, -1, d)
            a_left = self._roll(self.a, 1, d)

            # Classical gradients
            grad_R = self.F - F_right  # F[i] - F[i+1]
            grad_L = self.F - F_left   # F[i] - F[i-1]

            # Coherence on bonds
            C_R = self.C[d]                  # Coherence on (i, i+1)
            C_L = self._roll(self.C[d], 1, d)  # Coherence on (i-1, i)
            sqrt_C_R = np.sqrt(np.maximum(C_R, 0.01))
            sqrt_C_L = np.sqrt(np.maximum(C_L, 0.01))

            # Classical drive
            drive_R = (1.0 - sqrt_C_R) * grad_R
            drive_L = (1.0 - sqrt_C_L) * grad_L

            # Agency gates
            g_R = np.sqrt(np.maximum(self.a * a_right, 0))
            g_L = np.sqrt(np.maximum(self.a * a_left, 0))

            # Conductivity
            cond_R = self.sigma * (C_R + 1e-4)
            cond_L = self.sigma * (C_L + 1e-4)

            # Diffusive flux
            J_diff_R[d] = g_R * cond_R * drive_R
            J_diff_L = g_L * cond_L * drive_L

            flux_R = J_diff_R[d].copy()
            flux_L = J_diff_L.copy()

            # Momentum flux (IV.4)
            if p.momentum_enabled:
                F_avg_R = 0.5 * (self.F + F_right)
                F_avg_L = 0.5 * (self.F + F_left)
                flux_R += p.mu_pi * self.sigma * self.pi[d] * F_avg_R
                # Note: pi for leftward flux is the rolled pi
                flux_L += -p.mu_pi * self.sigma * self._roll(self.pi[d], 1, d) * F_avg_L

            # Floor flux (IV.6)
            if p.floor_enabled:
                s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
                s_right = self._roll(s, -1, d)
                s_left = self._roll(s, 1, d)
                flux_R += p.eta_floor * self.sigma * (s + s_right) * grad_R
                flux_L += p.eta_floor * self.sigma * (s + s_left) * grad_L

            # Gravity flux (V.2)
            if p.gravity_enabled:
                g_bond_R = 0.5 * (self.g[d] + self._roll(self.g[d], -1, d))
                g_bond_L = 0.5 * (self.g[d] + self._roll(self.g[d], 1, d))
                F_avg_R = 0.5 * (self.F + F_right)
                F_avg_L = 0.5 * (self.F + F_left)
                flux_R += p.mu_grav * self.sigma * g_bond_R * F_avg_R
                flux_L += p.mu_grav * self.sigma * g_bond_L * F_avg_L

            J_R[d] = flux_R
            J_L[d] = flux_L

        # STEP 3: Limiter (matching det_v6_3)
        total_outflow = np.zeros(self.shape, dtype=np.float64)
        for d in range(dim):
            total_outflow += np.maximum(0, J_R[d]) + np.maximum(0, J_L[d])

        max_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_out / (total_outflow + 1e-9))

        # Apply limiter only to positive (outgoing) flux
        J_R_lim = np.zeros_like(J_R)
        J_L_lim = np.zeros_like(J_L)
        J_diff_R_scaled = np.zeros_like(J_diff_R)
        for d in range(dim):
            J_R_lim[d] = np.where(J_R[d] > 0, J_R[d] * scale, J_R[d])
            J_L_lim[d] = np.where(J_L[d] > 0, J_L[d] * scale, J_L[d])
            J_diff_R_scaled[d] = np.where(J_diff_R[d] > 0, J_diff_R[d] * scale, J_diff_R[d])

        # Dissipation
        D = np.zeros(self.shape, dtype=np.float64)
        for d in range(dim):
            D += (np.abs(J_R_lim[d]) + np.abs(J_L_lim[d])) * self.Delta_tau

        # STEP 4: Resource update (matching det_v6_3)
        dF = np.zeros(self.shape, dtype=np.float64)
        for d in range(dim):
            transfer_R = J_R_lim[d] * self.Delta_tau
            transfer_L = J_L_lim[d] * self.Delta_tau
            outflow = transfer_R + transfer_L
            # Inflow: from left neighbor going right, from right neighbor going left
            inflow = self._roll(transfer_R, 1, d) + self._roll(transfer_L, -1, d)
            dF += inflow - outflow

        F_old = self.F.copy()
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)

        # STEP 5: Grace injection (VI.5 - matching det_v6_3)
        # I_g = D * w / w_sum where w = a * need
        if p.grace_enabled:
            n = np.maximum(0, p.F_MIN_grace - self.F)
            w = self.a * n

            # Local sum of weights
            w_sum = w.copy()
            for d in range(dim):
                w_sum += self._roll(w, -1, d) + self._roll(w, 1, d)
            w_sum = np.maximum(w_sum, 1e-12)

            I_g = D * w / w_sum
            self.F += I_g
            self.total_grace += np.sum(I_g)

        # STEP 6: Momentum update (IV.4 with v6.3 gravity coupling)
        if p.momentum_enabled:
            for d in range(dim):
                Delta_tau_R = 0.5 * (self.Delta_tau +
                                     self._roll(self.Delta_tau, -1, d))
                decay = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_R)

                # Diffusion drive (use scaled J_diff)
                dpi_diff = p.alpha_pi * J_diff_R_scaled[d] * Delta_tau_R

                # Gravity coupling (v6.3: beta_g)
                if p.gravity_enabled:
                    g_bond = 0.5 * (self.g[d] + self._roll(self.g[d], -1, d))
                    dpi_grav = p.beta_g * g_bond * Delta_tau_R
                else:
                    dpi_grav = 0

                self.pi[d] = decay * self.pi[d] + dpi_diff + dpi_grav

        # STEP 7: Structure update
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)

        # STEP 8: Agency update (v6.4)
        if p.agency_enabled:
            # Structural ceiling
            a_max = 1.0 / (1.0 + p.lambda_a * self.q**2)

            # Local average presence
            P_local = self.P.copy()
            for d in range(dim):
                P_local += self._roll(self.P, -1, d) + self._roll(self.P, 1, d)
            P_local /= (1 + 2 * dim)

            # Average coherence
            C_sum = np.zeros(self.shape)
            for d in range(dim):
                C_sum += self.C[d] + self._roll(self.C[d], 1, d)
            C_avg = C_sum / (2 * dim)

            # Coherence-gated relational drive
            gamma = p.gamma_a_max * (C_avg ** p.gamma_a_power)
            delta_a = gamma * (self.P - P_local)

            # Unified update
            self.a = self.a + p.beta_a * (a_max - self.a) + delta_a
            self.a = np.clip(self.a, 0.0, a_max)

        # STEP 9: Coherence dynamics
        if p.coherence_enabled:
            for d in range(dim):
                self.C[d] = np.clip(
                    self.C[d] + p.alpha_C * np.abs(J_diff_R[d]) * self.Delta_tau
                    - p.lambda_C * self.C[d] * self.Delta_tau,
                    p.C_init, 1.0
                )

        self._clip()
        self.time += dt
        self.step_count += 1

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def total_mass(self) -> float:
        """Total resource in lattice."""
        return float(np.sum(self.F))

    def total_q(self) -> float:
        """Total structure in lattice."""
        return float(np.sum(self.q))

    def potential_energy(self) -> float:
        """Gravitational potential energy."""
        return float(np.sum(self.F * self.Phi))

    def center_of_mass(self) -> Tuple:
        """Center of mass position."""
        total = np.sum(self.F) + 1e-9
        if self.dim == 1:
            x = np.arange(self.N)
            return (float(np.sum(x * self.F) / total),)
        elif self.dim == 2:
            y, x = np.mgrid[0:self.N, 0:self.N]
            return (float(np.sum(y * self.F) / total),
                    float(np.sum(x * self.F) / total))
        else:
            z, y, x = np.mgrid[0:self.N, 0:self.N, 0:self.N]
            return (float(np.sum(z * self.F) / total),
                    float(np.sum(y * self.F) / total),
                    float(np.sum(x * self.F) / total))

    def separation(self) -> float:
        """Distance between two largest blobs (1D/2D only)."""
        if self.dim > 2:
            return 0.0  # Not implemented for 3D

        # Try scipy first, fall back to simple peak finding
        try:
            from scipy.ndimage import label
            return self._separation_scipy(label)
        except ImportError:
            return self._separation_simple()

    def _separation_scipy(self, label_func) -> float:
        """Separation using scipy's connected component labeling."""
        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = label_func(above)

        if num < 2:
            return 0.0

        # Find centers of mass for each blob
        blobs = []
        for i in range(1, num + 1):
            mask = labeled == i
            mass = np.sum(self.F[mask])
            if mass < 0.1:
                continue

            if self.dim == 1:
                x = np.arange(self.N)
                cx = np.sum(x[mask] * self.F[mask]) / mass
                blobs.append({'x': cx, 'mass': mass})
            else:
                y, x = np.mgrid[0:self.N, 0:self.N]
                cx = np.sum(x[mask] * self.F[mask]) / mass
                cy = np.sum(y[mask] * self.F[mask]) / mass
                blobs.append({'x': cx, 'y': cy, 'mass': mass})

        if len(blobs) < 2:
            return 0.0

        blobs.sort(key=lambda b: -b['mass'])

        if self.dim == 1:
            dx = blobs[1]['x'] - blobs[0]['x']
            if dx > self.N/2: dx -= self.N
            if dx < -self.N/2: dx += self.N
            return abs(dx)
        else:
            dx = blobs[1]['x'] - blobs[0]['x']
            dy = blobs[1]['y'] - blobs[0]['y']
            if dx > self.N/2: dx -= self.N
            if dx < -self.N/2: dx += self.N
            if dy > self.N/2: dy -= self.N
            if dy < -self.N/2: dy += self.N
            return float(np.sqrt(dx**2 + dy**2))

    def _separation_simple(self) -> float:
        """Simple separation using peak finding (no scipy)."""
        if self.dim == 1:
            # Find two highest peaks
            threshold = np.max(self.F) * 0.3
            above_idx = np.where(self.F > threshold)[0]

            if len(above_idx) < 2:
                return 0.0

            # Find peak 1 (max)
            peak1_idx = np.argmax(self.F)
            peak1_mass = self.F[peak1_idx]

            # Find peak 2 (max away from peak 1)
            # Mask out region around peak 1
            mask = np.ones(self.N, dtype=bool)
            exclude_radius = self.N // 10
            for i in range(-exclude_radius, exclude_radius + 1):
                idx = (peak1_idx + i) % self.N
                mask[idx] = False

            masked_F = self.F.copy()
            masked_F[~mask] = 0

            peak2_idx = np.argmax(masked_F)
            peak2_mass = masked_F[peak2_idx]

            if peak2_mass < self.p.F_VAC * 5:
                return 0.0

            # Compute separation (periodic)
            dx = peak2_idx - peak1_idx
            if dx > self.N/2: dx -= self.N
            if dx < -self.N/2: dx += self.N
            return abs(dx)
        else:
            # 2D: Simple max-finding approach
            flat_idx = np.argmax(self.F)
            peak1_y, peak1_x = np.unravel_index(flat_idx, self.F.shape)

            # Mask out region around peak 1
            mask = np.ones_like(self.F, dtype=bool)
            exclude_radius = self.N // 10
            for dy in range(-exclude_radius, exclude_radius + 1):
                for dx in range(-exclude_radius, exclude_radius + 1):
                    y = (peak1_y + dy) % self.N
                    x = (peak1_x + dx) % self.N
                    mask[y, x] = False

            masked_F = self.F.copy()
            masked_F[~mask] = 0

            flat_idx2 = np.argmax(masked_F)
            peak2_y, peak2_x = np.unravel_index(flat_idx2, self.F.shape)

            if masked_F[peak2_y, peak2_x] < self.p.F_VAC * 5:
                return 0.0

            dx = peak2_x - peak1_x
            dy = peak2_y - peak1_y
            if dx > self.N/2: dx -= self.N
            if dx < -self.N/2: dx += self.N
            if dy > self.N/2: dy -= self.N
            if dy < -self.N/2: dy += self.N

            return float(np.sqrt(dx**2 + dy**2))

    def render_ascii(self, field: str = 'F', width: int = 60) -> str:
        """Render field as ASCII art."""
        if field == 'F':
            data = self.F
        elif field == 'q':
            data = self.q
        elif field == 'a':
            data = self.a
        elif field == 'P':
            data = self.P
        else:
            data = self.F

        # For 1D, create a horizontal bar
        if self.dim == 1:
            # Resample to width
            indices = np.linspace(0, self.N - 1, width).astype(int)
            values = data[indices]

            # Normalize
            vmin, vmax = np.min(values), np.max(values)
            if vmax > vmin:
                norm = (values - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(values)

            # ASCII characters by density
            chars = " ░▒▓█"
            lines = []
            for row in range(5):
                threshold = 1.0 - (row + 0.5) / 5
                line = ""
                for v in norm:
                    if v >= threshold:
                        idx = min(int(v * 5), 4)
                        line += chars[idx]
                    else:
                        line += " "
                lines.append(line)

            return "\n".join(lines)

        elif self.dim == 2:
            # Resample to width x height (half height for aspect)
            height = width // 2
            xi = np.linspace(0, self.N - 1, width).astype(int)
            yi = np.linspace(0, self.N - 1, height).astype(int)
            values = data[np.ix_(yi, xi)]

            # Normalize
            vmin, vmax = np.min(values), np.max(values)
            if vmax > vmin:
                norm = (values - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(values)

            chars = " ░▒▓█"
            lines = []
            for row in range(height):
                line = ""
                for col in range(width):
                    v = norm[row, col]
                    idx = min(int(v * 5), 4)
                    line += chars[idx]
                lines.append(line)

            return "\n".join(lines)

        else:
            # 3D: show middle slice
            mid = self.N // 2
            slice_data = data[mid, :, :]
            height = width // 2
            xi = np.linspace(0, self.N - 1, width).astype(int)
            yi = np.linspace(0, self.N - 1, height).astype(int)
            values = slice_data[np.ix_(yi, xi)]

            vmin, vmax = np.min(values), np.max(values)
            if vmax > vmin:
                norm = (values - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(values)

            chars = " ░▒▓█"
            lines = [f"[Z-slice at {mid}]"]
            for row in range(height):
                line = ""
                for col in range(width):
                    v = norm[row, col]
                    idx = min(int(v * 5), 4)
                    line += chars[idx]
                lines.append(line)

            return "\n".join(lines)


# =============================================================================
# Global Lattice Registry
# =============================================================================

_lattices: Dict[int, DETLattice] = {}
_next_id: int = 1


def lattice_create(dim: int = 1, N: int = 100, **params) -> int:
    """Create a new lattice and return its ID."""
    global _next_id

    p = LatticeParams(dim=dim, N=N)
    for key, value in params.items():
        if hasattr(p, key):
            setattr(p, key, value)

    lattice = DETLattice(p)
    lattice_id = _next_id
    _lattices[lattice_id] = lattice
    _next_id += 1

    return lattice_id


def lattice_get(lattice_id: int) -> Optional[DETLattice]:
    """Get lattice by ID."""
    return _lattices.get(lattice_id)


def lattice_destroy(lattice_id: int) -> bool:
    """Destroy a lattice."""
    if lattice_id in _lattices:
        del _lattices[lattice_id]
        return True
    return False


def lattice_list() -> List[int]:
    """List all lattice IDs."""
    return list(_lattices.keys())
