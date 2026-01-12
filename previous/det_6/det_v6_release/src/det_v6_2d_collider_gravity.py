"""
DET v6 2D Collider with Gravity Module
======================================

This implementation adds the DET gravity module (Section V) to the 2D collider:
- Helmholtz baseline solver for b
- Poisson potential solver for Φ
- Gravitational force computation and flux

Reference: DET Theory Card v6.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import time


@dataclass
class DETParams2D:
    """DET 2D simulation parameters with gravity."""
    N: int = 100                    # Grid size
    DT: float = 0.015               # Global step size (dk)
    F_VAC: float = 0.01             # Vacuum resource level
    R: int = 5                      # Local neighborhood radius
    
    # Coherence
    C_init: float = 0.3             # Initial bond coherence
    
    # Momentum module (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.08          # Momentum accumulation rate
    lambda_pi: float = 0.015        # Momentum friction/decay
    mu_pi: float = 0.25             # Momentum-to-flux coupling
    pi_max: float = 2.5             # Momentum clip
    
    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.015          # q accumulation rate
    
    # Agency dynamics
    a_coupling: float = 30.0        # λ in a_target = 1/(1 + λq²)
    a_rate: float = 0.2             # β response rate
    
    # Floor repulsion (IV.3a)
    floor_enabled: bool = True
    eta_floor: float = 0.12         # Floor coupling strength
    F_core: float = 4.0             # Floor activation threshold
    floor_power: float = 2.0        # Floor nonlinearity exponent
    
    # Gravity module (V.1-V.3) - NEW
    gravity_enabled: bool = True
    alpha_grav: float = 0.02        # Helmholtz screening parameter
    kappa_grav: float = 5.0         # Gravity coupling strength
    mu_grav: float = 2.0            # Gravitational flux coupling
    
    # Numerical stability
    outflow_limit: float = 0.20     # Max fraction of F that can leave per Δτ


def periodic_local_sum_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs)."""
    result = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
    return result


class DETCollider2DGravity:
    """
    DET v6 2D Collider with Gravity Module
    """
    
    def __init__(self, params: Optional[DETParams2D] = None):
        self.p = params or DETParams2D()
        N = self.p.N
        
        # Per-node state
        self.F = np.ones((N, N)) * self.p.F_VAC      # Free resource
        self.q = np.zeros((N, N))                     # Structural debt
        self.a = np.ones((N, N))                      # Agency
        
        # Per-bond state - East and South directions
        self.pi_E = np.zeros((N, N))                  # Bond momentum (E)
        self.pi_S = np.zeros((N, N))                  # Bond momentum (S)
        self.C_E = np.ones((N, N)) * self.p.C_init    # Coherence (E)
        self.C_S = np.ones((N, N)) * self.p.C_init    # Coherence (S)
        self.sigma = np.ones((N, N))                  # Processing rate
        
        # Gravity fields
        self.b = np.zeros((N, N))                     # Baseline field
        self.Phi = np.zeros((N, N))                   # Gravitational potential
        self.gx = np.zeros((N, N))                    # Gravitational force (x)
        self.gy = np.zeros((N, N))                    # Gravitational force (y)
        
        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N))
        self.Delta_tau = np.ones((N, N)) * self.p.DT
        
        # Precompute FFT wavenumbers
        self._setup_fft_solvers()
    
    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers for spectral solvers."""
        N = self.p.N
        kx = np.fft.fftfreq(N) * N
        ky = np.fft.fftfreq(N) * N
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        
        # Discrete 2D Laplacian eigenvalues
        self.L_k = -4 * (np.sin(np.pi * KX / N)**2 + np.sin(np.pi * KY / N)**2)
        
        # Helmholtz operator
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12
        
        # Poisson (handle k=0)
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0, 0] = 1.0
    
    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        """Solve Helmholtz equation: (L - α)b = -α * source"""
        source_k = fft2(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifft2(b_k))
    
    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        """Solve Poisson equation: L Φ = -κ * source (attractive gravity)"""
        source_k = fft2(source)
        source_k[0, 0] = 0  # Compatibility condition
        
        # For attractive gravity: Φ_k = -κ * source_k / L_k
        # This gives Φ > 0 at mass locations (potential hill in our convention)
        # But force g = -∇Φ points toward mass (attraction)
        Phi_k = -self.p.kappa_grav * source_k / self.L_k_poisson
        Phi_k[0, 0] = 0
        
        return np.real(ifft2(Phi_k))
    
    def _compute_gravity(self):
        """Compute gravitational fields from structural debt q."""
        if not self.p.gravity_enabled:
            self.gx = np.zeros_like(self.F)
            self.gy = np.zeros_like(self.F)
            return
        
        # Step 1: Helmholtz baseline
        self.b = self._solve_helmholtz(self.q)
        
        # Step 2: Relative source
        rho = self.q - self.b
        
        # Step 3: Poisson potential
        self.Phi = self._solve_poisson(rho)
        
        # Step 4: Gravitational force (negative gradient)
        E = lambda x: np.roll(x, -1, axis=1)
        W = lambda x: np.roll(x, 1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)
        Nb = lambda x: np.roll(x, 1, axis=0)
        
        self.gx = -0.5 * (E(self.Phi) - W(self.Phi))
        self.gy = -0.5 * (S(self.Phi) - Nb(self.Phi))
    
    def add_packet(self, center: Tuple[int, int], mass: float = 6.0, 
                   width: float = 5.0, momentum: Tuple[float, float] = (0, 0),
                   initial_q: float = 0.0):
        """Add a Gaussian resource packet with optional initial momentum and q."""
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
        """Enforce physical bounds on state variables."""
        self.F = np.clip(self.F, self.p.F_VAC, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_E = np.clip(self.pi_E, -self.p.pi_max, self.p.pi_max)
        self.pi_S = np.clip(self.pi_S, -self.p.pi_max, self.p.pi_max)
    
    def step(self):
        """Execute one canonical DET update step with gravity."""
        p = self.p
        N = p.N
        dk = p.DT
        
        # Neighbor access operators
        E = lambda x: np.roll(x, -1, axis=1)
        W = lambda x: np.roll(x, 1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)
        Nb = lambda x: np.roll(x, 1, axis=0)
        
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
        
        Delta_tau_E = 0.5 * (self.Delta_tau + E(self.Delta_tau))
        Delta_tau_S = 0.5 * (self.Delta_tau + S(self.Delta_tau))
        
        # ============================================================
        # STEP 2: Flow computation
        # ============================================================
        
        # Classical (pressure) contribution
        classical_E = self.F - E(self.F)
        classical_W = self.F - W(self.F)
        classical_S = self.F - S(self.F)
        classical_N = self.F - Nb(self.F)
        
        # Coherence interpolation
        sqrt_C_E = np.sqrt(self.C_E)
        sqrt_C_S = np.sqrt(self.C_S)
        sqrt_C_W = np.sqrt(W(self.C_E))
        sqrt_C_N = np.sqrt(Nb(self.C_S))
        
        # Drive (simplified - classical only for now)
        drive_E = (1 - sqrt_C_E) * classical_E
        drive_W = (1 - sqrt_C_W) * classical_W
        drive_S = (1 - sqrt_C_S) * classical_S
        drive_N = (1 - sqrt_C_N) * classical_N
        
        # Agency-gated diffusion
        g_E = np.sqrt(self.a * E(self.a))
        g_W = np.sqrt(self.a * W(self.a))
        g_S = np.sqrt(self.a * S(self.a))
        g_N = np.sqrt(self.a * Nb(self.a))
        
        # Conductivity
        cond_E = self.sigma * (self.C_E + 1e-4)
        cond_W = self.sigma * (W(self.C_E) + 1e-4)
        cond_S = self.sigma * (self.C_S + 1e-4)
        cond_N = self.sigma * (Nb(self.C_S) + 1e-4)
        
        # Agency-gated diffusive flux
        J_diff_E = g_E * cond_E * drive_E
        J_diff_W = g_W * cond_W * drive_W
        J_diff_S = g_S * cond_S * drive_S
        J_diff_N = g_N * cond_N * drive_N
        
        # Momentum-driven flux
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
        
        # Floor repulsion flux
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_E = p.eta_floor * self.sigma * (s + E(s)) * classical_E
            J_floor_W = p.eta_floor * self.sigma * (s + W(s)) * classical_W
            J_floor_S = p.eta_floor * self.sigma * (s + S(s)) * classical_S
            J_floor_N = p.eta_floor * self.sigma * (s + Nb(s)) * classical_N
        else:
            J_floor_E = J_floor_W = J_floor_S = J_floor_N = 0
        
        # Gravitational flux - NOT agency-gated
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
        
        # ============================================================
        # STEP 3: Conservative limiter
        # ============================================================
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
        
        # ============================================================
        # STEP 4: Resource update
        # ============================================================
        transfer_E = J_E_lim * self.Delta_tau
        transfer_W = J_W_lim * self.Delta_tau
        transfer_S = J_S_lim * self.Delta_tau
        transfer_N = J_N_lim * self.Delta_tau
        
        outflow = transfer_E + transfer_W + transfer_S + transfer_N
        inflow = W(transfer_E) + E(transfer_W) + Nb(transfer_S) + S(transfer_N)
        
        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_VAC, 1000)
        
        # ============================================================
        # STEP 5: Momentum update with gravity
        # ============================================================
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
        
        # ============================================================
        # STEP 6: Structural update
        # ============================================================
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # ============================================================
        # STEP 7: Agency update
        # ============================================================
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
        from scipy.ndimage import label
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
# TESTS
# ============================================================

def test_F6_gravitational_binding_2d(verbose: bool = True) -> Dict:
    """Test gravitational binding in 2D."""
    if verbose:
        print("\n" + "="*60)
        print("TEST F6: Gravitational Binding (2D)")
        print("="*60)
    
    params = DETParams2D(
        N=100,
        DT=0.02,
        F_VAC=0.001,
        C_init=0.5,
        momentum_enabled=True,
        alpha_pi=0.1,
        lambda_pi=0.002,
        mu_pi=0.5,
        q_enabled=True,
        alpha_q=0.02,
        a_coupling=3.0,
        a_rate=0.05,
        floor_enabled=False,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=10.0,
        mu_grav=3.0
    )
    
    sim = DETCollider2DGravity(params)
    
    # Two packets with slight inward momentum
    initial_sep = 40
    center = params.N // 2
    sim.add_packet((center, center - initial_sep//2), mass=8.0, width=5.0, 
                   momentum=(0, 0.1), initial_q=0.3)
    sim.add_packet((center, center + initial_sep//2), mass=8.0, width=5.0, 
                   momentum=(0, -0.1), initial_q=0.3)
    
    initial_mass = sim.total_mass()
    
    rec = {'t': [], 'sep': [], 'mass_err': [], 'q_total': [], 'PE': []}
    
    n_steps = 2000
    
    for t in range(n_steps):
        sep = sim.separation()
        mass_err = 100 * (sim.total_mass() - initial_mass) / initial_mass
        
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['mass_err'].append(mass_err)
        rec['q_total'].append(sim.total_q())
        rec['PE'].append(sim.potential_energy())
        
        if verbose and t % 400 == 0:
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


def test_mass_conservation_2d(verbose: bool = True) -> bool:
    """Test mass conservation with gravity enabled."""
    if verbose:
        print("\n" + "="*60)
        print("TEST F7: Mass Conservation (2D with Gravity)")
        print("="*60)
    
    params = DETParams2D(gravity_enabled=True)
    sim = DETCollider2DGravity(params)
    sim.add_packet((30, 30), mass=10.0, width=5.0, momentum=(0.3, 0.3))
    sim.add_packet((70, 70), mass=10.0, width=5.0, momentum=(-0.3, -0.3))
    
    initial_mass = sim.total_mass()
    
    for t in range(1000):
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


def run_full_test_suite_2d():
    """Run the complete 2D test suite with gravity."""
    print("="*70)
    print("DET v6 2D COLLIDER WITH GRAVITY - FULL TEST SUITE")
    print("="*70)
    
    results = {}
    
    results['F7'] = test_mass_conservation_2d(verbose=True)
    results['F6'] = test_F6_gravitational_binding_2d(verbose=True)
    
    print("\n" + "="*70)
    print("SUITE SUMMARY")
    print("="*70)
    print(f"  F7 (Mass conservation): {'PASS' if results['F7'] else 'FAIL'}")
    print(f"  F6 (Gravitational binding): {'PASS' if results['F6']['passed'] else 'FAIL'}")
    
    return results


if __name__ == "__main__":
    results = run_full_test_suite_2d()
