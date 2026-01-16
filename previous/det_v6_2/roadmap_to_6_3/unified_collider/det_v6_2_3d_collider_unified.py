"""
DET v6.2 3D Collider - Unified Implementation
=============================================

Complete implementation with all DET modules:
- Gravity module (Section V): Helmholtz baseline, Poisson potential, gravitational flux
- Boundary operators (Section VI): Grace injection, bond healing
- Agency-gated diffusion (IV.2)
- Presence-clocked transport (III.1)
- Momentum dynamics (IV.4)
- Angular momentum dynamics (IV.5)
- Floor repulsion (IV.6)

Reference: DET Theory Card v6.2
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import label
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


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
    
    # Agency dynamics (VI.2B)
    agency_dynamic: bool = True
    a_coupling: float = 30.0
    a_rate: float = 0.2
    
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


class DETCollider3DUnified:
    """
    DET v6.2 3D Collider - Unified with Gravity and Boundary Operators
    """
    
    def __init__(self, params: Optional[DETParams3D] = None):
        self.p = params or DETParams3D()
        N = self.p.N
        
        # Per-node state
        self.F = np.ones((N, N, N), dtype=np.float64) * self.p.F_VAC
        self.q = np.zeros((N, N, N), dtype=np.float64)
        self.a = np.ones((N, N, N), dtype=np.float64)
        
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
        
        # STEP 1: Presence
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        Delta_tau_Xp = 0.5 * (self.Delta_tau + Xp(self.Delta_tau))
        Delta_tau_Yp = 0.5 * (self.Delta_tau + Yp(self.Delta_tau))
        Delta_tau_Zp = 0.5 * (self.Delta_tau + Zp(self.Delta_tau))
        
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
        
        # STEP 5: Grace Injection (VI.5)
        if p.boundary_enabled and p.grace_enabled:
            I_g = self.compute_grace_injection(D)
            self.F = self.F + I_g
            self.last_grace_injection = I_g.copy()
            self.total_grace_injected += np.sum(I_g)
        else:
            self.last_grace_injection = np.zeros((N, N, N))
        
        # STEP 6: Momentum update with gravity
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
        
        # STEP 7: Angular momentum update (IV.5)
        if p.angular_momentum_enabled:
            Delta_tau_XY = 0.25 * (self.Delta_tau + Xp(self.Delta_tau) + 
                                   Yp(self.Delta_tau) + Xp(Yp(self.Delta_tau)))
            Delta_tau_YZ = 0.25 * (self.Delta_tau + Yp(self.Delta_tau) + 
                                   Zp(self.Delta_tau) + Yp(Zp(self.Delta_tau)))
            Delta_tau_XZ = 0.25 * (self.Delta_tau + Xp(self.Delta_tau) + 
                                   Zp(self.Delta_tau) + Xp(Zp(self.Delta_tau)))
            
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
        
        self._clip()
        self.time += dk
        self.step_count += 1
    
    def total_mass(self) -> float:
        return float(np.sum(self.F))
    
    def total_q(self) -> float:
        return float(np.sum(self.q))
    
    def potential_energy(self) -> float:
        return float(np.sum(self.F * self.Phi))
    
    def center_of_mass(self) -> Tuple[float, float, float]:
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        total = np.sum(self.F) + 1e-9
        return (float(np.sum(x * self.F) / total), 
                float(np.sum(y * self.F) / total), 
                float(np.sum(z * self.F) / total))
    
    def separation(self) -> float:
        """Find separation between two largest peaks."""
        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = label(above)
        
        if num < 2:
            return 0.0
        
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        
        coms = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) == 0:
                continue
            weights = self.F[mask]
            total_mass = np.sum(weights)
            if total_mass < 0.1:
                continue
            cx = np.sum(x[mask] * weights) / total_mass
            cy = np.sum(y[mask] * weights) / total_mass
            cz = np.sum(z[mask] * weights) / total_mass
            coms.append({'x': cx, 'y': cy, 'z': cz, 'mass': total_mass})
        
        coms.sort(key=lambda c: -c['mass'])
        
        if len(coms) < 2:
            return 0.0
        
        dx = coms[1]['x'] - coms[0]['x']
        dy = coms[1]['y'] - coms[0]['y']
        dz = coms[1]['z'] - coms[0]['z']
        
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        if dz > N/2: dz -= N
        if dz < -N/2: dz += N
        
        return np.sqrt(dx**2 + dy**2 + dz**2)


# ============================================================
# FULL TEST SUITE
# ============================================================

def test_gravity_vacuum(verbose: bool = True) -> bool:
    """Gravity has no effect in vacuum (q=0)"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravity in Vacuum")
        print("="*60)
    
    params = DETParams3D(N=16, gravity_enabled=True, q_enabled=False, boundary_enabled=False)
    sim = DETCollider3DUnified(params)
    
    for _ in range(200):
        sim.step()
    
    max_g = np.max(np.abs(sim.gx)) + np.max(np.abs(sim.gy)) + np.max(np.abs(sim.gz))
    max_Phi = np.max(np.abs(sim.Phi))
    
    passed = max_g < 1e-10 and max_Phi < 1e-10
    
    if verbose:
        print(f"  Max |g|: {max_g:.2e}")
        print(f"  Max |Φ|: {max_Phi:.2e}")
        print(f"  Vacuum gravity {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F7_mass_conservation(verbose: bool = True) -> bool:
    """F7: Mass conservation with gravity + boundary"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F7: Mass Conservation (Gravity + Boundary)")
        print("="*60)
    
    params = DETParams3D(N=24, F_MIN=0.0, gravity_enabled=True, boundary_enabled=True)
    sim = DETCollider3DUnified(params)
    sim.add_packet((8, 8, 8), mass=10.0, width=3.0, momentum=(0.2, 0.2, 0.2))
    sim.add_packet((16, 16, 16), mass=10.0, width=3.0, momentum=(-0.2, -0.2, -0.2))
    
    initial_mass = sim.total_mass()
    
    for t in range(500):
        sim.step()
    
    final_mass = sim.total_mass()
    grace_added = sim.total_grace_injected
    effective_drift = abs(final_mass - initial_mass - grace_added) / initial_mass
    
    passed = effective_drift < 0.10
    
    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Grace added: {grace_added:.4f}")
        print(f"  Effective drift: {effective_drift*100:.4f}%")
        print(f"  F7 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F6_gravitational_binding(verbose: bool = True) -> Dict:
    """F6: Gravitational binding"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F6: Gravitational Binding (3D)")
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
    
    sim = DETCollider3DUnified(params)
    
    initial_sep = 12
    center = params.N // 2
    sim.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, -0.1), initial_q=0.3)
    
    rec = {'t': [], 'sep': [], 'PE': []}
    
    for t in range(1500):
        sep = sim.separation()
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
    
    rec['passed'] = sep_decreased or bound_state
    rec['initial_sep'] = initial_sep_m
    rec['final_sep'] = final_sep
    rec['min_sep'] = min_sep
    
    if verbose:
        print(f"\n  Initial sep: {initial_sep_m:.1f}, Final: {final_sep:.1f}, Min: {min_sep:.1f}")
        print(f"  F6 {'PASSED' if rec['passed'] else 'FAILED'}")
    
    return rec


def test_F2_grace_coercion(verbose: bool = True) -> bool:
    """F2: Grace doesn't go to a=0 nodes"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F2: Grace Coercion (a=0 blocks grace)")
        print("="*60)
    
    params = DETParams3D(
        N=24, boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.15, a_rate=0.0, gravity_enabled=True
    )
    sim = DETCollider3DUnified(params)
    
    sim.add_packet((12, 12, 6), mass=3.0, width=2.0, momentum=(0, 0, 0.3))
    sim.add_packet((12, 12, 18), mass=3.0, width=2.0, momentum=(0, 0, -0.3))
    
    sz, sy, sx = 12, 12, 12
    sim.a[sz, sy, sx] = 0.0
    sim.F[sz, sy, sx] = 0.01
    
    for _ in range(200):
        sim.step()
    
    sentinel_grace = sim.last_grace_injection[sz, sy, sx]
    
    passed = sentinel_grace == 0.0
    
    if verbose:
        print(f"  Sentinel a = {sim.a[sz, sy, sx]:.4f}")
        print(f"  Sentinel grace = {sentinel_grace:.2e}")
        print(f"  F2 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F3_boundary_redundancy(verbose: bool = True) -> bool:
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
        sim = DETCollider3DUnified(params)
        sim.add_packet((12, 12, 6), mass=2.0, width=2.0, momentum=(0, 0, 0.3))
        sim.add_packet((12, 12, 18), mass=2.0, width=2.0, momentum=(0, 0, -0.3))
        
        for _ in range(300):
            sim.step()
        
        return np.mean(sim.F[10:14, 10:14, 10:14]), sim.total_grace_injected
    
    F_off, grace_off = run_scenario(False)
    F_on, grace_on = run_scenario(True)
    
    passed = grace_on > grace_off + 0.001
    
    if verbose:
        print(f"  OFF: F={F_off:.4f}, grace={grace_off:.4f}")
        print(f"  ON:  F={F_on:.4f}, grace={grace_on:.4f}")
        print(f"  F3 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F8_vacuum_momentum(verbose: bool = True) -> bool:
    """F8: Momentum doesn't push vacuum"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F8: Vacuum Momentum")
        print("="*60)
    
    params = DETParams3D(
        N=16, momentum_enabled=True, q_enabled=False, floor_enabled=False,
        F_MIN=0.0, gravity_enabled=False, boundary_enabled=False
    )
    sim = DETCollider3DUnified(params)
    sim.F = np.ones_like(sim.F) * params.F_VAC
    sim.pi_X = np.ones_like(sim.pi_X) * 1.0
    
    initial_mass = sim.total_mass()
    
    for _ in range(200):
        sim.step()
    
    final_mass = sim.total_mass()
    drift = abs(final_mass - initial_mass) / initial_mass
    
    passed = drift < 0.01
    
    if verbose:
        print(f"  Mass drift: {drift*100:.4f}%")
        print(f"  F8 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F9_symmetry_drift(verbose: bool = True) -> bool:
    """F9: Symmetric IC doesn't drift"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F9: Symmetry Drift")
        print("="*60)
    
    params = DETParams3D(N=20, momentum_enabled=False, gravity_enabled=False, boundary_enabled=False)
    sim = DETCollider3DUnified(params)
    
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
    
    if verbose:
        print(f"  Max COM drift: {max_drift:.4f} cells")
        print(f"  F9 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def run_full_test_suite():
    """Run complete 3D test suite."""
    print("="*70)
    print("DET v6.2 3D COLLIDER UNIFIED - FULL TEST SUITE")
    print("Gravity + Boundary Operators")
    print("="*70)
    
    results = {}
    
    results['vacuum_gravity'] = test_gravity_vacuum(verbose=True)
    results['F7'] = test_F7_mass_conservation(verbose=True)
    results['F6'] = test_F6_gravitational_binding(verbose=True)
    results['F2'] = test_F2_grace_coercion(verbose=True)
    results['F3'] = test_F3_boundary_redundancy(verbose=True)
    results['F8'] = test_F8_vacuum_momentum(verbose=True)
    results['F9'] = test_F9_symmetry_drift(verbose=True)
    
    print("\n" + "="*70)
    print("3D SUITE SUMMARY")
    print("="*70)
    print(f"  Vacuum gravity: {'PASS' if results['vacuum_gravity'] else 'FAIL'}")
    print(f"  F7 (Mass conservation): {'PASS' if results['F7'] else 'FAIL'}")
    print(f"  F6 (Gravitational binding): {'PASS' if results['F6']['passed'] else 'FAIL'}")
    print(f"  F2 (Grace coercion): {'PASS' if results['F2'] else 'FAIL'}")
    print(f"  F3 (Boundary redundancy): {'PASS' if results['F3'] else 'FAIL'}")
    print(f"  F8 (Vacuum momentum): {'PASS' if results['F8'] else 'FAIL'}")
    print(f"  F9 (Symmetry drift): {'PASS' if results['F9'] else 'FAIL'}")
    
    all_passed = (results['vacuum_gravity'] and results['F7'] and 
                  results['F6']['passed'] and results['F2'] and 
                  results['F3'] and results['F8'] and results['F9'])
    print(f"\n  OVERALL: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    
    return results


if __name__ == "__main__":
    run_full_test_suite()
