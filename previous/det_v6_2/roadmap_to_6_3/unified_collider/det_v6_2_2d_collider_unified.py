"""
DET v6.2 2D Collider - Unified Implementation
=============================================

Complete implementation with all DET modules:
- Gravity module (Section V): Helmholtz baseline, Poisson potential, gravitational flux
- Boundary operators (Section VI): Grace injection, bond healing
- Agency-gated diffusion (IV.2)
- Presence-clocked transport (III.1)
- Momentum dynamics (IV.4)
- Floor repulsion (IV.6)

Reference: DET Theory Card v6.2
"""

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.ndimage import label
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class DETParams2D:
    """DET 2D simulation parameters - complete."""
    N: int = 100
    DT: float = 0.015
    F_VAC: float = 0.01
    F_MIN: float = 0.0
    R: int = 5
    
    # Coherence
    C_init: float = 0.3
    
    # Momentum module (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.08
    lambda_pi: float = 0.015
    mu_pi: float = 0.25
    pi_max: float = 2.5
    
    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.015
    
    # Agency dynamics (VI.2B)
    a_coupling: float = 30.0
    a_rate: float = 0.2
    
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
    
    # Boundary operators (VI)
    boundary_enabled: bool = True
    grace_enabled: bool = True
    F_MIN_grace: float = 0.05
    healing_enabled: bool = False
    eta_heal: float = 0.03
    R_boundary: int = 3
    
    # Numerical stability
    outflow_limit: float = 0.20


def periodic_local_sum_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs)."""
    result = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
    return result


class DETCollider2DUnified:
    """
    DET v6.2 2D Collider - Unified with Gravity and Boundary Operators
    """
    
    def __init__(self, params: Optional[DETParams2D] = None):
        self.p = params or DETParams2D()
        N = self.p.N
        
        # Per-node state
        self.F = np.ones((N, N)) * self.p.F_VAC
        self.q = np.zeros((N, N))
        self.a = np.ones((N, N))
        
        # Per-bond state
        self.pi_E = np.zeros((N, N))
        self.pi_S = np.zeros((N, N))
        self.C_E = np.ones((N, N)) * self.p.C_init
        self.C_S = np.ones((N, N)) * self.p.C_init
        self.sigma = np.ones((N, N))
        
        # Gravity fields
        self.b = np.zeros((N, N))
        self.Phi = np.zeros((N, N))
        self.gx = np.zeros((N, N))
        self.gy = np.zeros((N, N))
        
        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N))
        self.Delta_tau = np.ones((N, N)) * self.p.DT
        
        # Boundary operator diagnostics
        self.last_grace_injection = np.zeros((N, N))
        self.last_healing_E = np.zeros((N, N))
        self.last_healing_S = np.zeros((N, N))
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
        Phi_k = -self.p.kappa_grav * source_k / self.L_k_poisson
        Phi_k[0, 0] = 0
        return np.real(ifft2(Phi_k))
    
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
                   initial_q: float = 0.0):
        """Add a Gaussian resource packet."""
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
        
        self._clip()
    
    def _clip(self):
        self.F = np.clip(self.F, self.p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_E = np.clip(self.pi_E, -self.p.pi_max, self.p.pi_max)
        self.pi_S = np.clip(self.pi_S, -self.p.pi_max, self.p.pi_max)
    
    def step(self):
        """Execute one canonical DET update step."""
        p = self.p
        N = p.N
        dk = p.DT
        
        E = lambda x: np.roll(x, -1, axis=1)
        W = lambda x: np.roll(x, 1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)
        Nb = lambda x: np.roll(x, 1, axis=0)
        
        # STEP 0: Gravity
        self._compute_gravity()
        
        # STEP 1: Presence
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        Delta_tau_E = 0.5 * (self.Delta_tau + E(self.Delta_tau))
        Delta_tau_S = 0.5 * (self.Delta_tau + S(self.Delta_tau))
        
        # STEP 2: Flow computation
        classical_E = self.F - E(self.F)
        classical_W = self.F - W(self.F)
        classical_S = self.F - S(self.F)
        classical_N = self.F - Nb(self.F)
        
        sqrt_C_E = np.sqrt(self.C_E)
        sqrt_C_S = np.sqrt(self.C_S)
        sqrt_C_W = np.sqrt(W(self.C_E))
        sqrt_C_N = np.sqrt(Nb(self.C_S))
        
        drive_E = (1 - sqrt_C_E) * classical_E
        drive_W = (1 - sqrt_C_W) * classical_W
        drive_S = (1 - sqrt_C_S) * classical_S
        drive_N = (1 - sqrt_C_N) * classical_N
        
        g_E = np.sqrt(self.a * E(self.a))
        g_W = np.sqrt(self.a * W(self.a))
        g_S = np.sqrt(self.a * S(self.a))
        g_N = np.sqrt(self.a * Nb(self.a))
        
        cond_E = self.sigma * (self.C_E + 1e-4)
        cond_W = self.sigma * (W(self.C_E) + 1e-4)
        cond_S = self.sigma * (self.C_S + 1e-4)
        cond_N = self.sigma * (Nb(self.C_S) + 1e-4)
        
        J_diff_E = g_E * cond_E * drive_E
        J_diff_W = g_W * cond_W * drive_W
        J_diff_S = g_S * cond_S * drive_S
        J_diff_N = g_N * cond_N * drive_N
        
        # Momentum flux
        if p.momentum_enabled:
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_W = 0.5 * (self.F + W(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            F_avg_N = 0.5 * (self.F + Nb(self.F))
            
            J_mom_E = p.mu_pi * self.sigma * self.pi_E * F_avg_E
            J_mom_W = -p.mu_pi * self.sigma * W(self.pi_E) * F_avg_W
            J_mom_S = p.mu_pi * self.sigma * self.pi_S * F_avg_S
            J_mom_N = -p.mu_pi * self.sigma * Nb(self.pi_S) * F_avg_N
        else:
            J_mom_E = J_mom_W = J_mom_S = J_mom_N = 0
        
        # Floor flux
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_E = p.eta_floor * self.sigma * (s + E(s)) * classical_E
            J_floor_W = p.eta_floor * self.sigma * (s + W(s)) * classical_W
            J_floor_S = p.eta_floor * self.sigma * (s + S(s)) * classical_S
            J_floor_N = p.eta_floor * self.sigma * (s + Nb(s)) * classical_N
        else:
            J_floor_E = J_floor_W = J_floor_S = J_floor_N = 0
        
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
            
            J_grav_E = p.mu_grav * self.sigma * gx_bond_E * F_avg_E
            J_grav_W = p.mu_grav * self.sigma * gx_bond_W * F_avg_W
            J_grav_S = p.mu_grav * self.sigma * gy_bond_S * F_avg_S
            J_grav_N = p.mu_grav * self.sigma * gy_bond_N * F_avg_N
        else:
            J_grav_E = J_grav_W = J_grav_S = J_grav_N = 0
        
        # Total flux
        J_E = J_diff_E + J_mom_E + J_floor_E + J_grav_E
        J_W = J_diff_W + J_mom_W + J_floor_W + J_grav_W
        J_S = J_diff_S + J_mom_S + J_floor_S + J_grav_S
        J_N = J_diff_N + J_mom_N + J_floor_N + J_grav_N
        
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
        
        # STEP 6: Momentum update with gravity
        if p.momentum_enabled:
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)
            
            dpi_diff_E = p.alpha_pi * J_diff_E_scaled * Delta_tau_E
            dpi_diff_S = p.alpha_pi * J_diff_S_scaled * Delta_tau_S
            
            if p.gravity_enabled:
                gx_bond_E = 0.5 * (self.gx + E(self.gx))
                gy_bond_S = 0.5 * (self.gy + S(self.gy))
                dpi_grav_E = 5.0 * p.mu_grav * gx_bond_E * Delta_tau_E
                dpi_grav_S = 5.0 * p.mu_grav * gy_bond_S * Delta_tau_S
            else:
                dpi_grav_E = dpi_grav_S = 0
            
            self.pi_E = decay_E * self.pi_E + dpi_diff_E + dpi_grav_E
            self.pi_S = decay_S * self.pi_S + dpi_diff_S + dpi_grav_S
        
        # STEP 7: Structure update
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # STEP 8: Bond Healing
        if p.boundary_enabled and p.healing_enabled:
            dC_heal_E, dC_heal_S = self.compute_bond_healing(D)
            self.C_E = np.clip(self.C_E + dC_heal_E, p.C_init, 1.0)
            self.C_S = np.clip(self.C_S + dC_heal_S, p.C_init, 1.0)
            self.last_healing_E = dC_heal_E.copy()
            self.last_healing_S = dC_heal_S.copy()
        else:
            self.last_healing_E = np.zeros((N, N))
            self.last_healing_S = np.zeros((N, N))
        
        # STEP 9: Agency update
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        self._clip()
        self.time += dk
        self.step_count += 1
    
    def total_mass(self) -> float:
        return float(np.sum(self.F))
    
    def center_of_mass(self) -> Tuple[float, float]:
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        total = np.sum(self.F) + 1e-9
        cx = float(np.sum(x * self.F) / total)
        cy = float(np.sum(y * self.F) / total)
        return (cy, cx)
    
    def total_q(self) -> float:
        return float(np.sum(self.q))
    
    def potential_energy(self) -> float:
        return float(np.sum(self.F * self.Phi))
    
    def separation(self) -> float:
        """Find separation between two largest peaks."""
        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = label(above)
        
        if num < 2:
            return 0.0
        
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        
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
            coms.append({'x': cx, 'y': cy, 'mass': total_mass})
        
        coms.sort(key=lambda c: -c['mass'])
        
        if len(coms) < 2:
            return 0.0
        
        dx = coms[1]['x'] - coms[0]['x']
        dy = coms[1]['y'] - coms[0]['y']
        
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        
        return np.sqrt(dx**2 + dy**2)


# ============================================================
# FULL TEST SUITE
# ============================================================

def test_gravity_vacuum(verbose: bool = True) -> bool:
    """Gravity has no effect in vacuum (q=0)"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravity in Vacuum")
        print("="*60)
    
    params = DETParams2D(gravity_enabled=True, q_enabled=False, boundary_enabled=False)
    sim = DETCollider2DUnified(params)
    
    for _ in range(300):
        sim.step()
    
    max_g = np.max(np.abs(sim.gx)) + np.max(np.abs(sim.gy))
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
    
    params = DETParams2D(F_MIN=0.0, gravity_enabled=True, boundary_enabled=True)
    sim = DETCollider2DUnified(params)
    sim.add_packet((30, 30), mass=10.0, width=5.0, momentum=(0.3, 0.3))
    sim.add_packet((70, 70), mass=10.0, width=5.0, momentum=(-0.3, -0.3))
    
    initial_mass = sim.total_mass()
    
    for t in range(1000):
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
        print("TEST F6: Gravitational Binding (2D)")
        print("="*60)
    
    params = DETParams2D(
        N=100, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        C_init=0.5,
        momentum_enabled=True, alpha_pi=0.1, lambda_pi=0.002, mu_pi=0.5,
        q_enabled=True, alpha_q=0.02,
        a_coupling=3.0, a_rate=0.05,
        floor_enabled=False,
        gravity_enabled=True, alpha_grav=0.01, kappa_grav=10.0, mu_grav=3.0,
        boundary_enabled=True, grace_enabled=True
    )
    
    sim = DETCollider2DUnified(params)
    
    initial_sep = 40
    center = params.N // 2
    sim.add_packet((center, center - initial_sep//2), mass=8.0, width=5.0, 
                   momentum=(0, 0.1), initial_q=0.3)
    sim.add_packet((center, center + initial_sep//2), mass=8.0, width=5.0, 
                   momentum=(0, -0.1), initial_q=0.3)
    
    rec = {'t': [], 'sep': [], 'PE': []}
    
    for t in range(2000):
        sep = sim.separation()
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['PE'].append(sim.potential_energy())
        
        if verbose and t % 400 == 0:
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
    
    params = DETParams2D(
        N=60, boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.15, a_rate=0.0, gravity_enabled=True
    )
    sim = DETCollider2DUnified(params)
    
    sim.add_packet((30, 15), mass=3.0, width=3.0, momentum=(0, 0.3))
    sim.add_packet((30, 45), mass=3.0, width=3.0, momentum=(0, -0.3))
    
    sentinel_y, sentinel_x = 30, 30
    sim.a[sentinel_y, sentinel_x] = 0.0
    sim.F[sentinel_y, sentinel_x] = 0.01
    
    for _ in range(200):
        sim.step()
    
    sentinel_grace = sim.last_grace_injection[sentinel_y, sentinel_x]
    
    passed = sentinel_grace == 0.0
    
    if verbose:
        print(f"  Sentinel a = {sim.a[sentinel_y, sentinel_x]:.4f}")
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
        params = DETParams2D(
            N=60, F_VAC=0.02,
            boundary_enabled=boundary_on, grace_enabled=True, F_MIN_grace=0.15,
            gravity_enabled=True
        )
        sim = DETCollider2DUnified(params)
        sim.add_packet((30, 15), mass=2.0, width=2.5, momentum=(0, 0.3))
        sim.add_packet((30, 45), mass=2.0, width=2.5, momentum=(0, -0.3))
        
        for _ in range(300):
            sim.step()
        
        return np.mean(sim.F[25:35, 25:35]), sim.total_grace_injected
    
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
    
    params = DETParams2D(
        momentum_enabled=True, q_enabled=False, floor_enabled=False,
        F_MIN=0.0, gravity_enabled=False, boundary_enabled=False
    )
    sim = DETCollider2DUnified(params)
    sim.F = np.ones_like(sim.F) * params.F_VAC
    sim.pi_E = np.ones_like(sim.pi_E) * 1.0
    sim.pi_S = np.ones_like(sim.pi_S) * 1.0
    
    initial_mass = sim.total_mass()
    
    for _ in range(300):
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
    
    params = DETParams2D(momentum_enabled=False, gravity_enabled=False, boundary_enabled=False)
    sim = DETCollider2DUnified(params)
    
    N = params.N
    sim.add_packet((N//2, N//4), mass=5.0, width=4.0, momentum=(0, 0))
    sim.add_packet((N//2, 3*N//4), mass=5.0, width=4.0, momentum=(0, 0))
    
    initial_com = sim.center_of_mass()
    
    max_drift = 0
    for _ in range(500):
        com = sim.center_of_mass()
        drift = np.sqrt((com[0] - initial_com[0])**2 + (com[1] - initial_com[1])**2)
        max_drift = max(max_drift, drift)
        sim.step()
    
    passed = max_drift < 1.0
    
    if verbose:
        print(f"  Max COM drift: {max_drift:.4f} cells")
        print(f"  F9 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def run_full_test_suite():
    """Run complete 2D test suite."""
    print("="*70)
    print("DET v6.2 2D COLLIDER UNIFIED - FULL TEST SUITE")
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
    print("2D SUITE SUMMARY")
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
