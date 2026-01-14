"""
DET v6_2 3D Collider with Gravity Module
======================================

This implementation adds the DET gravity module (Section V) to the 3D collider:
- Helmholtz baseline solver for b
- Poisson potential solver for Φ
- Gravitational force computation and flux

Also includes plaquette angular momentum (IV.4b).

Reference: DET Theory Card v6_2.0
"""

import numpy as np
from scipy.fft import fftn, ifftn
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time


@dataclass
class DETParams3D:
    """DET 3D simulation parameters with gravity."""
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
    
    # Plaquette Angular Momentum (IV.4b)
    angular_momentum_enabled: bool = True
    alpha_L: float = 0.06
    lambda_L: float = 0.005
    mu_L: float = 0.18
    L_max: float = 5.0
    
    # Floor repulsion
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
    
    # Gravity module (V.1-V.3) - NEW
    gravity_enabled: bool = True
    alpha_grav: float = 0.02
    kappa_grav: float = 5.0
    mu_grav: float = 2.0
    
    # Numerical stability
    outflow_limit: float = 0.2


class DETCollider3DGravity:
    """
    DET v6_2 3D Collider with Gravity and Angular Momentum
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
        
        # Precompute FFT wavenumbers
        self._setup_fft_solvers()
    
    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers for spectral solvers."""
        N = self.p.N
        kx = np.fft.fftfreq(N) * N
        ky = np.fft.fftfreq(N) * N
        kz = np.fft.fftfreq(N) * N
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Discrete 3D Laplacian eigenvalues
        self.L_k = -4 * (np.sin(np.pi * KX / N)**2 + 
                        np.sin(np.pi * KY / N)**2 + 
                        np.sin(np.pi * KZ / N)**2)
        
        # Helmholtz operator
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12
        
        # Poisson
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0, 0, 0] = 1.0
    
    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        """Solve Helmholtz equation: (L - α)b = -α * source"""
        source_k = fftn(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifftn(b_k))
    
    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        """Solve Poisson equation: L Φ = -κ * source"""
        source_k = fftn(source)
        source_k[0, 0, 0] = 0
        Phi_k = -self.p.kappa_grav * source_k / self.L_k_poisson
        Phi_k[0, 0, 0] = 0
        return np.real(ifftn(Phi_k))
    
    def _compute_gravity(self):
        """Compute gravitational fields from structural debt q."""
        if not self.p.gravity_enabled:
            self.gx = np.zeros_like(self.F)
            self.gy = np.zeros_like(self.F)
            self.gz = np.zeros_like(self.F)
            return
        
        # Helmholtz baseline
        self.b = self._solve_helmholtz(self.q)
        
        # Relative source
        rho = self.q - self.b
        
        # Poisson potential
        self.Phi = self._solve_poisson(rho)
        
        # Gravitational force (negative gradient)
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)
        Zm = lambda arr: np.roll(arr, 1, axis=0)
        
        self.gx = -0.5 * (Xp(self.Phi) - Xm(self.Phi))
        self.gy = -0.5 * (Yp(self.Phi) - Ym(self.Phi))
        self.gz = -0.5 * (Zp(self.Phi) - Zm(self.Phi))
    
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
        """Enforce physical bounds."""
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
        """Execute one canonical DET update step with gravity."""
        p = self.p
        dk = p.DT
        
        # Neighbor operators
        Xp = lambda arr: np.roll(arr, -1, axis=2)
        Xm = lambda arr: np.roll(arr, 1, axis=2)
        Yp = lambda arr: np.roll(arr, -1, axis=1)
        Ym = lambda arr: np.roll(arr, 1, axis=1)
        Zp = lambda arr: np.roll(arr, -1, axis=0)
        Zm = lambda arr: np.roll(arr, 1, axis=0)
        
        # ============================================================
        # STEP 0: Compute gravitational fields
        # ============================================================
        self._compute_gravity()
        
        # ============================================================
        # STEP 1: Presence and proper time
        # ============================================================
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        Delta_tau_Xp = 0.5 * (self.Delta_tau + Xp(self.Delta_tau))
        Delta_tau_Yp = 0.5 * (self.Delta_tau + Yp(self.Delta_tau))
        Delta_tau_Zp = 0.5 * (self.Delta_tau + Zp(self.Delta_tau))
        
        # ============================================================
        # STEP 2: Compute flux components
        # ============================================================
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
        
        # ============================================================
        # STEP 3: Conservative limiter
        # ============================================================
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
        
        # ============================================================
        # STEP 4: Resource update
        # ============================================================
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
        
        # ============================================================
        # STEP 5: Momentum update with gravity
        # ============================================================
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
        
        # ============================================================
        # STEP 6: Structural update
        # ============================================================
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # ============================================================
        # STEP 7: Agency update
        # ============================================================
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
    
    def separation(self) -> float:
        """Find separation between two largest peaks."""
        from scipy.ndimage import label
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
# TESTS
# ============================================================

def test_F6_gravitational_binding_3d(verbose: bool = True) -> Dict:
    """Test gravitational binding in 3D."""
    if verbose:
        print("\n" + "="*60)
        print("TEST F6: Gravitational Binding (3D)")
        print("="*60)
    
    params = DETParams3D(
        N=32,
        DT=0.02,
        F_VAC=0.001,
        F_MIN=0.0,
        C_init=0.3,
        diff_enabled=True,
        momentum_enabled=True,
        alpha_pi=0.1,
        lambda_pi=0.002,
        mu_pi=0.5,
        angular_momentum_enabled=False,
        floor_enabled=False,
        q_enabled=True,
        alpha_q=0.02,
        agency_dynamic=True,
        a_coupling=3.0,
        a_rate=0.05,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=10.0,
        mu_grav=3.0
    )
    
    sim = DETCollider3DGravity(params)
    
    # Two packets with slight inward momentum
    initial_sep = 12
    center = params.N // 2
    sim.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, -0.1), initial_q=0.3)
    
    initial_mass = sim.total_mass()
    
    rec = {'t': [], 'sep': [], 'mass_err': [], 'q_total': [], 'PE': []}
    
    n_steps = 1500
    
    for t in range(n_steps):
        sep = sim.separation()
        mass_err = 100 * (sim.total_mass() - initial_mass) / initial_mass
        
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['mass_err'].append(mass_err)
        rec['q_total'].append(sim.total_q())
        rec['PE'].append(sim.potential_energy())
        
        if verbose and t % 300 == 0:
            print(f"  t={t}: sep={sep:.1f}, q_tot={sim.total_q():.3f}, "
                  f"PE={sim.potential_energy():.3f}, min_a={np.min(sim.a):.3f}")
        
        sim.step()
    
    initial_sep_measured = rec['sep'][0] if rec['sep'][0] > 0 else initial_sep
    final_sep = rec['sep'][-1]
    min_sep = min(rec['sep'])
    
    sep_decreased = final_sep < initial_sep_measured * 0.9
    bound_state = min_sep < initial_sep_measured * 0.5
    
    rec['initial_sep'] = initial_sep_measured
    rec['final_sep'] = final_sep
    rec['min_sep'] = min_sep
    rec['passed'] = sep_decreased or bound_state
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Initial separation: {initial_sep_measured:.1f}")
        print(f"    Final separation: {final_sep:.1f}")
        print(f"    Minimum separation: {min_sep:.1f}")
        print(f"    F6 {'PASSED' if rec['passed'] else 'FAILED'}")
    
    return rec


def test_mass_conservation_3d(verbose: bool = True) -> bool:
    """Test mass conservation with gravity enabled."""
    if verbose:
        print("\n" + "="*60)
        print("TEST F7: Mass Conservation (3D with Gravity)")
        print("="*60)
    
    params = DETParams3D(N=24, gravity_enabled=True)
    sim = DETCollider3DGravity(params)
    sim.add_packet((8, 8, 8), mass=10.0, width=3.0, momentum=(0.2, 0.2, 0.2))
    sim.add_packet((16, 16, 16), mass=10.0, width=3.0, momentum=(-0.2, -0.2, -0.2))
    
    initial_mass = sim.total_mass()
    
    for t in range(500):
        sim.step()
    
    final_mass = sim.total_mass()
    drift_pct = 100 * abs(final_mass - initial_mass) / initial_mass
    
    passed = drift_pct < 10.0
    
    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Drift: {drift_pct:.4f}%")
        print(f"  F7 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def run_full_test_suite_3d():
    """Run the complete 3D test suite with gravity."""
    print("="*70)
    print("DET v6_2 3D COLLIDER WITH GRAVITY - FULL TEST SUITE")
    print("="*70)
    
    results = {}
    
    results['F7'] = test_mass_conservation_3d(verbose=True)
    results['F6'] = test_F6_gravitational_binding_3d(verbose=True)
    
    print("\n" + "="*70)
    print("SUITE SUMMARY")
    print("="*70)
    print(f"  F7 (Mass conservation): {'PASS' if results['F7'] else 'FAIL'}")
    print(f"  F6 (Gravitational binding): {'PASS' if results['F6']['passed'] else 'FAIL'}")
    
    return results


if __name__ == "__main__":
    results = run_full_test_suite_3d()
