"""
DET v6.3 Unified Collider - PyTorch Accelerated
================================================

GPU-accelerated implementation supporting 1D, 2D, and 3D simulations.
Uses PyTorch tensors and FFT for efficient computation.

Features:
- Unified architecture for all dimensions
- GPU acceleration via PyTorch
- Grace v6.4 antisymmetric edge flux
- Lattice correction factor eta
- Beta_g gravity-momentum coupling
- Angular momentum support (2D/3D)

Reference: DET Theory Card v6.3
"""

import torch
import torch.fft
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass
class DETParamsTorch:
    """Unified DET v6.3 Parameters for PyTorch backend"""
    N: int = 64
    dim: int = 3
    dt: float = 0.02
    device: str = "cpu"

    # Core physics
    F_VAC: float = 0.01
    F_MIN: float = 0.0
    C_init: float = 0.3

    # Diffusive flux
    diff_enabled: bool = True

    # Momentum (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.12
    lambda_pi: float = 0.01
    mu_pi: float = 0.35
    pi_max: float = 5.0

    # Angular Momentum (IV.5)
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

    # Gravity (V)
    gravity_enabled: bool = True
    alpha_grav: float = 0.02
    kappa_grav: float = 10.0
    mu_grav: float = 2.0
    beta_g: float = 10.0  # v6.3: gravity-momentum coupling

    # Boundary / Grace v6.4 (VI)
    boundary_enabled: bool = True
    grace_enabled: bool = True
    eta_g: float = 0.5
    beta_grace: float = 0.4
    C_quantum: float = 0.85
    R_grace: int = 2
    F_MIN_grace: float = 0.05

    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.02

    # Agency
    # Agency dynamics - v6.4 update
    agency_dynamic: bool = True
    lambda_a: float = 30.0      # Structural ceiling coupling
    beta_a: float = 0.2         # Relaxation rate toward ceiling
    gamma_a_max: float = 0.15   # Max relational drive strength
    gamma_a_power: float = 2.0  # Coherence gating exponent

    # Coherence dynamics
    coherence_dynamic: bool = True
    alpha_C: float = 0.04
    lambda_C: float = 0.002

    # Sigma dynamics
    sigma_dynamic: bool = True

    # Numerical
    outflow_limit: float = 0.2

    # v6.3: Option B - Coherence-weighted load
    # H_i = Σ_{j ∈ N_R(i)} √C_ij * σ_ij
    coherence_weighted_H: bool = False


def compute_lattice_correction(N: int, dim: int) -> float:
    """
    Derivable lattice renormalization constant eta.
    Based on research in roadmap_to_6_3/lattice_correction_study.
    """
    if dim == 1:
        if N <= 64: return 0.92
        if N <= 128: return 0.96
        return 0.98
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


class DETColliderTorch:
    """
    DET v6.3 Unified Collider - PyTorch Accelerated

    Supports 1D, 2D, and 3D simulations with GPU acceleration.

    Key features:
    - FFT-based gravity solver
    - Grace v6.4 antisymmetric edge flux
    - Beta_g gravity-momentum coupling
    - Angular momentum on plaquettes (2D/3D)
    - Lattice correction factor
    """

    def __init__(self, params: DETParamsTorch):
        self.p = params
        self.device = torch.device(params.device)
        N = params.N
        dim = params.dim
        shape = tuple([N] * dim)

        # Lattice correction
        self.eta = compute_lattice_correction(N, dim)

        # Per-node state
        self.F = torch.full(shape, params.F_VAC, device=self.device, dtype=torch.float64)
        self.q = torch.zeros(shape, device=self.device, dtype=torch.float64)
        self.a = torch.ones(shape, device=self.device, dtype=torch.float64)
        self.sigma = torch.ones(shape, device=self.device, dtype=torch.float64)
        self.theta = torch.rand(shape, device=self.device, dtype=torch.float64) * 2 * np.pi

        # Per-bond momentum (one per dimension)
        self.pi = torch.zeros([dim] + list(shape), device=self.device, dtype=torch.float64)

        # Per-bond coherence (one per dimension)
        self.C = torch.full([dim] + list(shape), params.C_init, device=self.device, dtype=torch.float64)

        # Angular momentum on plaquettes (for dim >= 2)
        if dim >= 2:
            # Number of plaquettes: dim*(dim-1)/2
            num_plaquettes = dim * (dim - 1) // 2
            self.L = torch.zeros([num_plaquettes] + list(shape), device=self.device, dtype=torch.float64)
        else:
            self.L = None

        # Gravity fields
        self.Phi = torch.zeros(shape, device=self.device, dtype=torch.float64)
        self.g = torch.zeros([dim] + list(shape), device=self.device, dtype=torch.float64)

        # Presence and proper time
        self.P = torch.ones(shape, device=self.device, dtype=torch.float64)
        self.Delta_tau = torch.ones(shape, device=self.device, dtype=torch.float64) * params.dt
        self.accumulated_proper_time = torch.zeros(shape, device=self.device, dtype=torch.float64)

        # Diagnostics
        self.step_count = 0
        self.time = 0.0
        self.total_grace_injected = 0.0
        self.last_grace = torch.zeros(shape, device=self.device, dtype=torch.float64)

        self._setup_fft_kernels()

    def _setup_fft_kernels(self):
        """Precompute FFT kernels for Helmholtz and Poisson solvers."""
        N = self.p.N
        dim = self.p.dim

        # Create frequency grid
        freqs = [torch.fft.fftfreq(N, device=self.device, dtype=torch.float64) * N for _ in range(dim)]
        grids = torch.meshgrid(*freqs, indexing='ij')

        # Discrete Laplacian in Fourier space: L_k = -4 * sum(sin^2(pi * k_i / N))
        self.L_k = torch.zeros([N]*dim, device=self.device, dtype=torch.float64)
        for g in grids:
            self.L_k -= 4 * torch.sin(torch.pi * g / N)**2

        # Helmholtz kernel: H_k = L_k - alpha_grav
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[torch.abs(self.H_k) < 1e-12] = 1e-12

        # Poisson kernel (with singularity handling)
        self.L_k_poisson = self.L_k.clone()
        idx = tuple([0]*dim)
        self.L_k_poisson[idx] = 1.0

    def _solve_gravity(self):
        """Solve gravitational fields from structure q."""
        if not self.p.gravity_enabled:
            self.g.zero_()
            return

        # 1. Helmholtz baseline: (L - alpha)b = -alpha * q
        q_k = torch.fft.fftn(self.q)
        b_k = -self.p.alpha_grav * q_k / self.H_k
        b = torch.real(torch.fft.ifftn(b_k))

        # 2. Relative source: rho = q - b
        rho = self.q - b

        # 3. Poisson potential with lattice correction
        # Sign: kappa*rho / L_k with L_k < 0 gives Phi < 0 (attractive gravity)
        kappa_eff = self.p.kappa_grav * self.eta
        rho_k = torch.fft.fftn(rho)
        idx = tuple([0]*self.p.dim)
        rho_k[idx] = 0
        Phi_k = kappa_eff * rho_k / self.L_k_poisson  # Removed minus for attraction
        Phi_k[idx] = 0
        self.Phi = torch.real(torch.fft.ifftn(Phi_k))

        # 4. Gradient for g
        for d in range(self.p.dim):
            self.g[d] = -0.5 * (torch.roll(self.Phi, shifts=-1, dims=d) -
                                torch.roll(self.Phi, shifts=1, dims=d))

    def _compute_grace(self, D: torch.Tensor) -> torch.Tensor:
        """Compute grace injection (simplified v6.2 style)."""
        p = self.p
        n = torch.clamp(p.F_MIN_grace - self.F, min=0.0)
        w = self.a * n

        # Local sum
        w_sum = w.clone()
        for d in range(p.dim):
            for shift in [-1, 1]:
                w_sum = w_sum + torch.roll(w, shifts=shift, dims=d)
        w_sum = w_sum + 1e-12

        I_g = D * w / w_sum
        return I_g

    def add_packet(self, pos: Tuple, mass: float, width: float,
                   momentum: Optional[Tuple] = None, initial_q: float = 0.0,
                   initial_spin: float = 0.0):
        """Add a Gaussian resource packet with optional momentum and spin."""
        N = self.p.N
        dim = self.p.dim
        coords = torch.meshgrid(*[torch.arange(N, device=self.device, dtype=torch.float64)
                                  for _ in range(dim)], indexing='ij')

        r2 = torch.zeros_like(self.F)
        for d in range(dim):
            diff = (coords[d] - pos[d] + N/2) % N - N/2
            r2 = r2 + diff**2

        envelope = torch.exp(-0.5 * r2 / width**2)
        self.F = self.F + mass * envelope

        # Add coherence
        for d in range(dim):
            self.C[d] = torch.clamp(self.C[d] + 0.5 * envelope, self.p.C_init, 1.0)

        if momentum is not None:
            for d in range(dim):
                if d < len(momentum):
                    self.pi[d] = self.pi[d] + momentum[d] * envelope

        if initial_q > 0:
            self.q = self.q + initial_q * envelope

        if initial_spin != 0 and self.L is not None:
            # Add to first plaquette (XY plane in 3D, only plaquette in 2D)
            self.L[0] = self.L[0] + initial_spin * envelope

        self._clip()

    def add_spin(self, pos: Tuple, spin: float, width: float):
        """Add angular momentum to a region."""
        if self.L is None:
            return

        N = self.p.N
        dim = self.p.dim
        coords = torch.meshgrid(*[torch.arange(N, device=self.device, dtype=torch.float64)
                                  for _ in range(dim)], indexing='ij')

        r2 = torch.zeros_like(self.F)
        for d in range(dim):
            diff = (coords[d] - pos[d] + N/2) % N - N/2
            r2 = r2 + diff**2

        envelope = torch.exp(-0.5 * r2 / width**2)
        self.L[0] = self.L[0] + spin * envelope
        self._clip()

    def _clip(self):
        """Enforce physical bounds."""
        p = self.p
        self.F = torch.clamp(self.F, p.F_MIN, 1000)
        self.q = torch.clamp(self.q, 0, 1)
        self.a = torch.clamp(self.a, 0, 1)
        self.pi = torch.clamp(self.pi, -p.pi_max, p.pi_max)
        if self.L is not None:
            self.L = torch.clamp(self.L, -p.L_max, p.L_max)

    def _compute_coherence_weighted_H(self) -> torch.Tensor:
        """Compute Option B coherence-weighted load.

        H_i = Σ_{j ∈ N_R(i)} √C_ij * σ_ij

        Sum over 2*dim neighbors.
        """
        dim = self.p.dim
        H = torch.zeros_like(self.F)

        for d in range(dim):
            # Positive direction: C[d] is coherence on bond (i, i+d)
            sqrt_C_p = torch.sqrt(self.C[d])
            sigma_p = 0.5 * (self.sigma + torch.roll(self.sigma, shifts=-1, dims=d))
            H = H + sqrt_C_p * sigma_p

            # Negative direction: C[d] at (i-1) is coherence on bond (i-1, i)
            sqrt_C_m = torch.sqrt(torch.roll(self.C[d], shifts=1, dims=d))
            sigma_m = 0.5 * (self.sigma + torch.roll(self.sigma, shifts=1, dims=d))
            H = H + sqrt_C_m * sigma_m

        return H

    def step(self):
        """Execute one canonical DET update step."""
        p = self.p
        dim = p.dim
        dk = p.dt

        # STEP 0: Gravity
        self._solve_gravity()

        # STEP 1: Presence
        # Option B: Coherence-weighted load H_i = Σ_{j} √C_ij * σ_ij
        if p.coherence_weighted_H:
            H = self._compute_coherence_weighted_H()
        else:
            H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        self.accumulated_proper_time = self.accumulated_proper_time + self.Delta_tau

        # STEP 2: Compute fluxes
        total_div = torch.zeros_like(self.F)
        J_diff_all = torch.zeros_like(self.pi)

        for d in range(dim):
            F_j = torch.roll(self.F, shifts=-1, dims=d)
            a_j = torch.roll(self.a, shifts=-1, dims=d)

            # Agency gate
            g_a = torch.sqrt(torch.clamp(self.a * a_j, min=0.0))

            # Diffusive flux
            if p.diff_enabled:
                sqrt_C = torch.sqrt(self.C[d])
                classical = self.F - F_j
                J_diff = g_a * self.sigma * (self.C[d] + 1e-4) * classical
                J_diff_all[d] = J_diff
                total_div = total_div + J_diff
                total_div = total_div - torch.roll(J_diff, shifts=1, dims=d)

            # Momentum flux
            if p.momentum_enabled:
                F_avg = 0.5 * (self.F + F_j)
                J_mom = p.mu_pi * self.sigma * self.pi[d] * F_avg
                total_div = total_div + J_mom
                total_div = total_div - torch.roll(J_mom, shifts=1, dims=d)

            # Floor flux
            if p.floor_enabled:
                s = torch.clamp((self.F - p.F_core) / p.F_core, min=0.0)**p.floor_power
                s_j = torch.roll(s, shifts=-1, dims=d)
                J_floor = p.eta_floor * self.sigma * (s + s_j) * (self.F - F_j)
                total_div = total_div + J_floor
                total_div = total_div - torch.roll(J_floor, shifts=1, dims=d)

            # Gravity flux
            if p.gravity_enabled:
                g_bond = 0.5 * (self.g[d] + torch.roll(self.g[d], shifts=-1, dims=d))
                F_avg = 0.5 * (self.F + F_j)
                J_grav = p.mu_grav * self.sigma * g_bond * F_avg
                total_div = total_div + J_grav
                total_div = total_div - torch.roll(J_grav, shifts=1, dims=d)

        # Angular momentum flux (2D/3D only)
        if p.angular_momentum_enabled and self.L is not None and dim >= 2:
            # For 2D: L[0] is L_Z
            # For 3D: L[0] is L_XY, L[1] is L_YZ, L[2] is L_XZ
            F_avg_0 = 0.5 * (self.F + torch.roll(self.F, shifts=-1, dims=0))
            F_avg_1 = 0.5 * (self.F + torch.roll(self.F, shifts=-1, dims=1))

            # Rotational flux from L[0] (XY plaquette in 3D, Z in 2D)
            J_rot_0 = p.mu_L * self.sigma * F_avg_0 * (self.L[0] - torch.roll(self.L[0], shifts=1, dims=1))
            J_rot_1 = -p.mu_L * self.sigma * F_avg_1 * (self.L[0] - torch.roll(self.L[0], shifts=1, dims=0))

            total_div = total_div + J_rot_0 - torch.roll(J_rot_0, shifts=1, dims=0)
            total_div = total_div + J_rot_1 - torch.roll(J_rot_1, shifts=1, dims=1)

        # STEP 3: Dissipation (for grace)
        D = torch.abs(total_div) * self.Delta_tau

        # STEP 4: Update F
        dF = -total_div * self.Delta_tau
        self.F = torch.clamp(self.F + dF, p.F_MIN, 1000)

        # STEP 5: Grace Injection
        if p.boundary_enabled and p.grace_enabled:
            I_g = self._compute_grace(D)
            self.F = self.F + I_g
            self.last_grace = I_g.clone()
            self.total_grace_injected += torch.sum(I_g).item()

        # STEP 6: Momentum update with gravity coupling
        if p.momentum_enabled:
            for d in range(dim):
                Delta_tau_bond = 0.5 * (self.Delta_tau + torch.roll(self.Delta_tau, shifts=-1, dims=d))
                decay = torch.clamp(1.0 - p.lambda_pi * Delta_tau_bond, min=0.0)

                dpi_diff = p.alpha_pi * J_diff_all[d] * Delta_tau_bond

                if p.gravity_enabled:
                    g_bond = 0.5 * (self.g[d] + torch.roll(self.g[d], shifts=-1, dims=d))
                    dpi_grav = p.beta_g * g_bond * Delta_tau_bond
                else:
                    dpi_grav = 0

                self.pi[d] = decay * self.pi[d] + dpi_diff + dpi_grav

        # STEP 7: Angular momentum update
        if p.angular_momentum_enabled and self.L is not None and dim >= 2:
            Delta_tau_plaq = 0.25 * (self.Delta_tau +
                                     torch.roll(self.Delta_tau, shifts=-1, dims=0) +
                                     torch.roll(self.Delta_tau, shifts=-1, dims=1) +
                                     torch.roll(torch.roll(self.Delta_tau, shifts=-1, dims=0), shifts=-1, dims=1))

            # Curl of momentum for L[0]
            curl = (self.pi[0] + torch.roll(self.pi[1], shifts=-1, dims=0) -
                    torch.roll(self.pi[0], shifts=-1, dims=1) - self.pi[1])

            decay = torch.clamp(1.0 - p.lambda_L * Delta_tau_plaq, min=0.0)
            self.L[0] = decay * self.L[0] + p.alpha_L * curl * Delta_tau_plaq

        # STEP 8: Structure update
        if p.q_enabled:
            self.q = torch.clamp(self.q + p.alpha_q * torch.clamp(-dF, min=0.0), 0, 1)

        # STEP 9: Agency update
        if p.agency_dynamic:
            # v6.4 Agency Law: Structural Ceiling + Relational Drive

            # Step 1: Structural ceiling (matter law)
            a_max = 1.0 / (1.0 + p.lambda_a * self.q**2)

            # Step 2: Relational drive (life law)
            # Compute local average presence
            P_local = self.P.clone()
            for d in range(dim):
                P_local = P_local + torch.roll(self.P, 1, dims=d) + torch.roll(self.P, -1, dims=d)
            P_local = P_local / (1 + 2 * dim)

            # Average coherence at each node
            C_sum = torch.zeros_like(self.F)
            for d in range(dim):
                C_sum = C_sum + self.C[d] + torch.roll(self.C[d], 1, dims=d)
            C_avg = C_sum / (2 * dim)

            # Coherence-gated drive: γ(C) = γ_max * C^n
            gamma = p.gamma_a_max * (C_avg ** p.gamma_a_power)

            # Relational drive
            delta_a_drive = gamma * (self.P - P_local)

            # Unified update
            self.a = self.a + p.beta_a * (a_max - self.a) + delta_a_drive
            self.a = torch.clamp(self.a, 0.0, a_max)

        # STEP 10: Coherence dynamics
        if p.coherence_dynamic:
            for d in range(dim):
                self.C[d] = torch.clamp(
                    self.C[d] + p.alpha_C * torch.abs(J_diff_all[d]) * self.Delta_tau
                    - p.lambda_C * self.C[d] * self.Delta_tau,
                    p.C_init, 1.0
                )

        self._clip()
        self.time += dk
        self.step_count += 1

    # ==================== DIAGNOSTICS ====================

    def total_mass(self) -> float:
        return torch.sum(self.F).item()

    def total_q(self) -> float:
        return torch.sum(self.q).item()

    def total_angular_momentum(self) -> float:
        if self.L is None:
            return 0.0
        return torch.sum(self.L).item()

    def potential_energy(self) -> float:
        return torch.sum(self.F * self.Phi).item()

    def center_of_mass(self) -> Tuple:
        N = self.p.N
        dim = self.p.dim
        coords = torch.meshgrid(*[torch.arange(N, device=self.device, dtype=torch.float64)
                                  for _ in range(dim)], indexing='ij')
        total = torch.sum(self.F) + 1e-9
        com = tuple(float(torch.sum(coords[d] * self.F) / total) for d in range(dim))
        return com

    def to_numpy(self, field: str = 'F') -> np.ndarray:
        """Convert a field to numpy array."""
        return getattr(self, field).cpu().numpy()


class DETRaytracer:
    """
    DET v6.3 Raytracer - Volumetric Rendering and State Probing
    Uses ray-marching to visualize the DET lattice.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def render_volume(self, field: torch.Tensor,
                      resolution: Tuple[int, int] = (128, 128),
                      num_steps: int = 64,
                      step_size: float = 0.5,
                      camera_distance: float = 2.0) -> torch.Tensor:
        """
        Render a 3D field using ray-marching.

        field: 3D tensor [N, N, N]
        resolution: (width, height) of output image
        num_steps: number of ray march steps
        step_size: step size in grid units
        camera_distance: distance multiplier for camera position
        """
        if field.dim() != 3:
            raise ValueError("Raytracer only supports 3D fields")

        N = field.shape[0]
        W, H = resolution

        # Generate rays in NDC space
        x = torch.linspace(-1, 1, W, device=self.device, dtype=torch.float64)
        y = torch.linspace(-1, 1, H, device=self.device, dtype=torch.float64)
        gx, gy = torch.meshgrid(x, y, indexing='ij')

        # Ray origins (camera looking at center from +Z)
        ray_origins = torch.zeros((W, H, 3), device=self.device, dtype=torch.float64)
        ray_origins[..., 2] = -N * camera_distance

        # Ray directions (toward center)
        ray_dirs = torch.stack([gx * 0.5, gy * 0.5, torch.ones_like(gx)], dim=-1)
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)

        # Ray-marching
        accumulated = torch.zeros((W, H), device=self.device, dtype=torch.float64)
        transmittance = torch.ones((W, H), device=self.device, dtype=torch.float64)

        for step in range(num_steps):
            t = step * step_size
            current_pos = ray_origins + ray_dirs * t

            # Map to grid coordinates
            grid_pos = current_pos + N / 2
            ix = grid_pos[..., 0].long()
            iy = grid_pos[..., 1].long()
            iz = grid_pos[..., 2].long()

            # Bounds check
            mask = ((ix >= 0) & (ix < N) &
                    (iy >= 0) & (iy < N) &
                    (iz >= 0) & (iz < N))

            # Sample field
            density = torch.zeros((W, H), device=self.device, dtype=torch.float64)
            if torch.any(mask):
                ix_valid = torch.clamp(ix, 0, N-1)
                iy_valid = torch.clamp(iy, 0, N-1)
                iz_valid = torch.clamp(iz, 0, N-1)
                density = field[ix_valid, iy_valid, iz_valid] * mask.float()

            # Accumulate with absorption
            absorption = density * step_size * 0.1
            accumulated = accumulated + transmittance * density * step_size
            transmittance = transmittance * torch.exp(-absorption)

        return accumulated

    def probe_ray(self, field: torch.Tensor, origin: torch.Tensor,
                  direction: torch.Tensor, max_dist: float,
                  step_size: float = 0.1) -> torch.Tensor:
        """Probe the field along a single ray and return the profile."""
        steps = int(max_dist / step_size)
        profile = []
        N = field.shape[0]

        direction = direction / torch.norm(direction)

        for i in range(steps):
            pos = origin + direction * (i * step_size)
            grid_pos = (pos + N/2).long()

            if all(0 <= grid_pos[d] < N for d in range(len(grid_pos))):
                if field.dim() == 3:
                    val = field[grid_pos[0], grid_pos[1], grid_pos[2]].item()
                elif field.dim() == 2:
                    val = field[grid_pos[0], grid_pos[1]].item()
                else:
                    val = field[grid_pos[0]].item()
                profile.append(val)
            else:
                profile.append(0.0)

        return torch.tensor(profile, device=self.device)

    def render_slice(self, field: torch.Tensor, axis: int = 2,
                     position: Optional[int] = None) -> torch.Tensor:
        """Extract a 2D slice from a 3D field."""
        if field.dim() != 3:
            raise ValueError("Slice rendering only supports 3D fields")

        N = field.shape[0]
        if position is None:
            position = N // 2

        if axis == 0:
            return field[position, :, :]
        elif axis == 1:
            return field[:, position, :]
        else:
            return field[:, :, position]


# ============================================================
# TEST SUITE
# ============================================================

def test_pytorch_gravity_vacuum(verbose: bool = True) -> bool:
    """Test gravity in vacuum."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: PyTorch Gravity in Vacuum")
        print("="*60)

    params = DETParamsTorch(N=32, dim=3, gravity_enabled=True, q_enabled=False, boundary_enabled=False)
    sim = DETColliderTorch(params)

    for _ in range(100):
        sim.step()

    max_g = torch.max(torch.abs(sim.g)).item()
    max_Phi = torch.max(torch.abs(sim.Phi)).item()

    passed = max_g < 1e-10 and max_Phi < 1e-10

    if verbose:
        print(f"  Max |g|: {max_g:.2e}")
        print(f"  Max |Phi|: {max_Phi:.2e}")
        print(f"  Lattice correction eta: {sim.eta:.3f}")
        print(f"  {'PASSED' if passed else 'FAILED'}")

    return passed


def test_pytorch_mass_conservation(verbose: bool = True) -> bool:
    """Test mass conservation."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: PyTorch Mass Conservation")
        print("="*60)

    params = DETParamsTorch(N=32, dim=3, gravity_enabled=True, boundary_enabled=True)
    sim = DETColliderTorch(params)
    sim.add_packet((16, 16, 16), mass=10.0, width=3.0, momentum=(0.2, 0.2, 0.2))

    initial_mass = sim.total_mass()

    for _ in range(200):
        sim.step()

    final_mass = sim.total_mass()
    grace_added = sim.total_grace_injected
    effective_drift = abs(final_mass - initial_mass - grace_added) / initial_mass

    passed = effective_drift < 0.05

    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Grace added: {grace_added:.4f}")
        print(f"  Effective drift: {effective_drift*100:.4f}%")
        print(f"  {'PASSED' if passed else 'FAILED'}")

    return passed


def test_pytorch_binding(verbose: bool = True) -> bool:
    """Test gravitational binding."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: PyTorch Gravitational Binding")
        print("="*60)

    params = DETParamsTorch(N=32, dim=3, gravity_enabled=True, q_enabled=True, boundary_enabled=True)
    sim = DETColliderTorch(params)

    sim.add_packet((16, 16, 10), mass=8.0, width=2.5, momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((16, 16, 22), mass=8.0, width=2.5, momentum=(0, 0, -0.1), initial_q=0.3)

    # Run and check if gravity fields are non-zero
    for t in range(100):
        sim.step()
        if verbose and t % 20 == 0:
            max_g = torch.max(torch.abs(sim.g)).item()
            print(f"  t={t}: max|g|={max_g:.4f}, PE={sim.potential_energy():.3f}")

    max_g = torch.max(torch.abs(sim.g)).item()
    passed = max_g > 0.01

    if verbose:
        print(f"  Final max|g|: {max_g:.4f}")
        print(f"  {'PASSED' if passed else 'FAILED'}")

    return passed


def test_pytorch_raytracer(verbose: bool = True) -> bool:
    """Test raytracer."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: PyTorch Raytracer")
        print("="*60)

    tracer = DETRaytracer()

    # Create test field
    field = torch.zeros((32, 32, 32))
    field[14:18, 14:18, 14:18] = 1.0  # Small cube in center

    img = tracer.render_volume(field, resolution=(64, 64))

    passed = img.shape == (64, 64) and torch.max(img).item() > 0

    if verbose:
        print(f"  Rendered image shape: {img.shape}")
        print(f"  Max density: {torch.max(img).item():.4f}")
        print(f"  {'PASSED' if passed else 'FAILED'}")

    return passed


def test_pytorch_option_b_coherence_weighted_H(verbose: bool = True) -> bool:
    """Test Option B: Coherence-weighted load H_i = Σ √C_ij σ_ij."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: PyTorch Option B - Coherence-Weighted Load")
        print("="*60)

    params = DETParamsTorch(
        N=32, dim=3,
        gravity_enabled=True, q_enabled=True, boundary_enabled=True,
        coherence_weighted_H=True  # Enable Option B
    )
    sim = DETColliderTorch(params)

    sim.add_packet((16, 16, 10), mass=8.0, width=2.5, momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((16, 16, 22), mass=8.0, width=2.5, momentum=(0, 0, -0.1), initial_q=0.3)

    for t in range(100):
        sim.step()
        if verbose and t % 20 == 0:
            H_mean = sim._compute_coherence_weighted_H().mean().item()
            print(f"  t={t}: H_mean={H_mean:.4f}, PE={sim.potential_energy():.3f}")

    max_g = torch.max(torch.abs(sim.g)).item()
    passed = max_g > 0.01

    if verbose:
        print(f"  Final max|g|: {max_g:.4f}")
        print(f"  Option B {'PASSED' if passed else 'FAILED'}")

    return passed


def run_pytorch_test_suite(include_option_b: bool = False):
    """Run PyTorch test suite."""
    print("="*70)
    print("DET v6.3 PYTORCH COLLIDER - TEST SUITE")
    print("="*70)

    results = {}

    results['vacuum_gravity'] = test_pytorch_gravity_vacuum(verbose=True)
    results['mass_conservation'] = test_pytorch_mass_conservation(verbose=True)
    results['binding'] = test_pytorch_binding(verbose=True)
    results['raytracer'] = test_pytorch_raytracer(verbose=True)

    if include_option_b:
        results['option_b_coherence_weighted_H'] = test_pytorch_option_b_coherence_weighted_H(verbose=True)

    print("\n" + "="*70)
    print("PYTORCH TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return results


if __name__ == "__main__":
    run_pytorch_test_suite()
