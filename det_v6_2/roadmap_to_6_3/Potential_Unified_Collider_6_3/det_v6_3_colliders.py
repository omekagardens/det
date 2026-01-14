import torch
import torch.fft
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class DETParams:
    """Unified DET v6.3 Parameters"""
    N: int = 64
    dim: int = 3
    dt: float = 0.02
    device: str = "cpu"
    
    # Physics constants
    F_VAC: float = 0.01
    F_MIN: float = 0.0
    C_init: float = 0.3
    
    # Momentum (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.12
    lambda_pi: float = 0.01
    mu_pi: float = 0.35
    pi_max: float = 5.0
    
    # Gravity (V)
    gravity_enabled: bool = True
    alpha_grav: float = 0.02
    kappa_grav: float = 10.0
    mu_grav: float = 2.0
    beta_g_mom: float = 5.0  # Momentum-gravity coupling
    
    # Boundary / Grace v6.4 (VI)
    boundary_enabled: bool = True
    eta_g: float = 0.5
    beta_g: float = 0.4
    C_quantum: float = 0.85
    R_grace: int = 2
    
    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.02
    
    # Agency
    a_rate: float = 0.2
    a_coupling: float = 30.0

class DETCollidertorch:
    """
    DET v6.3 Unified Collider - PyTorch Accelerated
    Supports 1D, 2D, and 3D simulations.
    """
    def __init__(self, params: DETParams):
        self.p = params
        self.device = torch.device(params.device)
        N = params.N
        dim = params.dim
        shape = [N] * dim
        
        # State variables
        self.F = torch.full(shape, params.F_VAC, device=self.device, dtype=torch.float64)
        self.q = torch.zeros(shape, device=self.device, dtype=torch.float64)
        self.a = torch.ones(shape, device=self.device, dtype=torch.float64)
        self.sigma = torch.ones(shape, device=self.device, dtype=torch.float64)
        
        # Momentum (stored per dimension)
        self.pi = torch.zeros([dim] + shape, device=self.device, dtype=torch.float64)
        # Coherence (stored per dimension)
        self.C = torch.full([dim] + shape, params.C_init, device=self.device, dtype=torch.float64)
        
        # Gravity fields
        self.Phi = torch.zeros(shape, device=self.device, dtype=torch.float64)
        self.g = torch.zeros([dim] + shape, device=self.device, dtype=torch.float64)
        
        # Lattice correction η
        self.eta = self._compute_lattice_correction(N)
        
        # Precompute FFT kernels
        self._setup_fft_kernels()
        
        self.step_count = 0
        self.total_grace_injected = 0.0

    def _compute_lattice_correction(self, N: int) -> float:
        """
        Derivable lattice renormalization constant η.
        Based on research in roadmap_to_6_3/lattice_correction_study.
        """
        # Empirical fit from the study: η converges to 1 as N increases
        # For N=64, η ≈ 0.9545. For N=128, η ≈ 0.9751.
        # Simple logarithmic fit for interpolation:
        if N <= 32: return 0.9010
        if N <= 64: return 0.9545
        if N <= 96: return 0.9683
        return 0.9751

    def _setup_fft_kernels(self):
        N = self.p.N
        dim = self.p.dim
        
        # Create frequency grid
        freqs = [torch.fft.fftfreq(N, device=self.device) * N for _ in range(dim)]
        grids = torch.meshgrid(*freqs, indexing='ij')
        
        # Discrete Laplacian in Fourier space: L_k = -4 * sum(sin^2(pi * k_i / N))
        self.L_k = torch.zeros([N]*dim, device=self.device)
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
        if not self.p.gravity_enabled:
            return
            
        # 1. Helmholtz baseline: (L - alpha)b = -alpha * q
        q_k = torch.fft.fftn(self.q)
        b_k = -self.p.alpha_grav * q_k / self.H_k
        b = torch.real(torch.fft.ifftn(b_k))
        
        # 2. Relative source: rho = q - b
        rho = self.q - b
        
        # 3. Poisson potential: L * Phi = -kappa * rho
        # Apply lattice correction η to kappa
        kappa_eff = self.p.kappa_grav / self.eta
        rho_k = torch.fft.fftn(rho)
        idx = tuple([0]*self.p.dim)
        rho_k[idx] = 0
        Phi_k = -kappa_eff * rho_k / self.L_k_poisson
        Phi_k[idx] = 0
        self.Phi = torch.real(torch.fft.ifftn(Phi_k))
        
        # 4. Gradient for g
        for d in range(self.p.dim):
            self.g[d] = -0.5 * (torch.roll(self.Phi, shifts=-1, dims=d) - torch.roll(self.Phi, shifts=1, dims=d))

    def _compute_grace_v64(self, F_local_avg: torch.Tensor) -> torch.Tensor:
        """Grace v6.4: Antisymmetric Edge Flux"""
        p = self.p
        dim = p.dim
        
        # Threshold and Need
        F_thresh = p.beta_g * F_local_avg
        need = torch.clamp(F_thresh - self.F, min=0.0)
        excess = torch.clamp(self.F - F_thresh, min=0.0)
        
        d_cap = self.a * excess
        r_need = self.a * need
        
        # Local sum of r_need (Manhattan distance R=2)
        r_sum = torch.zeros_like(r_need)
        # Simplified R=1 neighborhood for performance in 3D, can be expanded
        for d in range(dim):
            r_sum += torch.roll(r_need, shifts=1, dims=d) + torch.roll(r_need, shifts=-1, dims=d)
        r_sum += r_need
        
        total_grace_div = torch.zeros_like(self.F)
        
        for d in range(dim):
            # Bond-local quantities
            a_j = torch.roll(self.a, shifts=-1, dims=d)
            g_a = torch.sqrt(torch.clamp(self.a * a_j, min=0.0))
            
            C_j = torch.roll(self.C[d], shifts=-1, dims=d) # This is not quite right, C is on bonds
            # In this simplified tensor model, C[d] represents the bond in direction d
            Q_ij = torch.clamp(1.0 - torch.sqrt(self.C[d]) / p.C_quantum, min=0.0, max=1.0)
            
            d_j = torch.roll(d_cap, shifts=-1, dims=d)
            r_j = torch.roll(r_need, shifts=-1, dims=d)
            r_sum_j = torch.roll(r_sum, shifts=-1, dims=d)
            
            # G_{i->j}
            term_i_j = d_cap * r_j / (r_sum + 1e-10)
            term_j_i = d_j * r_need / (r_sum_j + 1e-10)
            
            G_flux = p.eta_g * g_a * Q_ij * (term_i_j - term_j_i)
            
            # Divergence
            total_grace_div -= G_flux # Outflow from i
            total_grace_div += torch.roll(G_flux, shifts=1, dims=d) # Inflow from i-1
            
        return total_grace_div

    def step(self):
        p = self.p
        dim = p.dim
        
        # 0. Gravity
        self._solve_gravity()
        
        # 1. Presence
        # P = a * sigma / (1 + F) / (1 + H)
        P = self.a * self.sigma / (1.0 + self.F) / (2.0) # Simplified H=1
        Delta_tau = P * p.dt
        
        # 2. Fluxes
        total_div = torch.zeros_like(self.F)
        J_diff_all = torch.zeros([dim] + list(self.F.shape), device=self.device, dtype=torch.float64)
        
        for d in range(dim):
            F_j = torch.roll(self.F, shifts=-1, dims=d)
            a_j = torch.roll(self.a, shifts=-1, dims=d)
            
            # Agency gate
            g_a = torch.sqrt(torch.clamp(self.a * a_j, min=0.0))
            
            # Diffusive
            sqrt_C = torch.sqrt(self.C[d])
            # Simplified: no phase dynamics in this core version yet
            J_diff = g_a * self.sigma * (1.0 - sqrt_C) * (self.F - F_j)
            J_diff_all[d] = J_diff
            
            # Momentum
            F_avg = 0.5 * (self.F + F_j)
            J_mom = p.mu_pi * self.sigma * self.pi[d] * F_avg
            
            # Gravity flux
            g_bond = 0.5 * (self.g[d] + torch.roll(self.g[d], shifts=-1, dims=d))
            J_grav = p.mu_grav * self.sigma * g_bond * F_avg
            
            # Total bond flux
            J_total = J_diff + J_mom + J_grav
            
            # Divergence
            total_div += J_total # Outflow
            total_div -= torch.roll(J_total, shifts=1, dims=d) # Inflow
            
        # 3. Update F
        # Use p.dt for consistency with grace_div
        self.F = torch.clamp(self.F - total_div * Delta_tau, min=p.F_MIN)
        
        # 4. Grace Injection
        if p.boundary_enabled:
            # Compute local average for threshold
            F_avg_kernel = torch.ones([3]*dim, device=self.device, dtype=torch.float64) / (3**dim)
            # Use circular padding for periodic BCs
            F_padded = torch.nn.functional.pad(self.F.unsqueeze(0).unsqueeze(0), [1,1]*dim, mode='circular')
            if dim == 1:
                F_local_avg = torch.nn.functional.conv1d(F_padded, F_avg_kernel.unsqueeze(0).unsqueeze(0)).squeeze()
            elif dim == 2:
                F_local_avg = torch.nn.functional.conv2d(F_padded, F_avg_kernel.unsqueeze(0).unsqueeze(0)).squeeze()
            elif dim == 3:
                F_local_avg = torch.nn.functional.conv3d(F_padded, F_avg_kernel.unsqueeze(0).unsqueeze(0)).squeeze()
            
            grace_div = self._compute_grace_v64(F_local_avg)
            # Grace is already a divergence (inflow - outflow), so we add it
            # Use Delta_tau for consistency with diffusive flux
            self.F = torch.clamp(self.F + grace_div * Delta_tau, min=p.F_MIN)
            self.total_grace_injected += torch.sum(grace_div * Delta_tau).item()
            
        # 5. Momentum Update
        if p.momentum_enabled:
            for d in range(dim):
                Delta_tau_bond = 0.5 * (Delta_tau + torch.roll(Delta_tau, shifts=-1, dims=d))
                decay = torch.clamp(1.0 - p.lambda_pi * Delta_tau_bond, min=0.0)
                
                # Gravity coupling
                g_bond = 0.5 * (self.g[d] + torch.roll(self.g[d], shifts=-1, dims=d))
                dpi_grav = p.beta_g_mom * p.mu_grav * g_bond * Delta_tau_bond
                
                # Diffusive coupling
                dpi_diff = p.alpha_pi * J_diff_all[d] * Delta_tau_bond
                
                self.pi[d] = torch.clamp(decay * self.pi[d] + dpi_diff + dpi_grav, -p.pi_max, p.pi_max)
                
        # 6. Structure (q-locking)
        if p.q_enabled:
            self.q = torch.clamp(self.q + p.alpha_q * (self.F - p.F_VAC) * Delta_tau, 0.0, 1.0)
            
        self.step_count += 1

    def add_packet(self, pos: Tuple, mass: float, width: float, momentum: Optional[Tuple] = None, initial_q: float = 0.0):
        """Add a Gaussian resource packet."""
        N = self.p.N
        dim = self.p.dim
        coords = torch.meshgrid(*[torch.arange(N, device=self.device) for _ in range(dim)], indexing='ij')
        
        r2 = torch.zeros_like(self.F)
        for d in range(dim):
            diff = (coords[d] - pos[d] + N/2) % N - N/2
            r2 += diff**2
            
        envelope = torch.exp(-0.5 * r2 / width**2)
        self.F += mass * envelope
        
        if momentum is not None:
            for d in range(dim):
                self.pi[d] += momentum[d] * envelope
        
        if initial_q > 0:
            self.q += initial_q * envelope
                
    def total_mass(self):
        return torch.sum(self.F).item()

# Example usage and basic test
if __name__ == "__main__":
    params = DETParams(N=32, dim=3, device="cpu")
    collider = DETCollidertorch(params)
    collider.add_packet((16, 16, 16), 10.0, 3.0, (0.5, 0.0, 0.0))
    
    print(f"Initial mass: {collider.total_mass():.4f}")
    for _ in range(10):
        collider.step()
    print(f"Mass after 10 steps: {collider.total_mass():.4f}")
    print(f"Total grace injected: {collider.total_grace_injected:.4f}")
