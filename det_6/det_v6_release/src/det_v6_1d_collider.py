"""
DET v6 1D Collider - Minimal Reference Implementation
======================================================

This is a minimal 1D implementation of the DET v6 theory card for testing
fundamental dynamics in the simplest possible setting.

Key features:
- Agency-gated diffusion (IV.2)
- Presence-clocked transport (III.1)
- Momentum dynamics (IV.4, optional)
- Floor repulsion (IV.6, optional)
- Canonical q-locking (Appendix B)
- Target-tracking agency update (VI.2B)

Reference: DET Theory Card v6.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class DETParams1D:
    """DET 1D simulation parameters."""
    N: int = 200                    # Grid size
    DT: float = 0.02                # Global step size (dk)
    F_VAC: float = 0.01             # Vacuum resource level
    F_MIN: float = 0.0              # True minimum (0 for conservation tests)
    
    # Coherence
    C_init: float = 0.3             # Initial bond coherence
    
    # Momentum module (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.10          # Momentum accumulation rate
    lambda_pi: float = 0.02         # Momentum friction/decay
    mu_pi: float = 0.30             # Momentum-to-flux coupling
    pi_max: float = 3.0             # Momentum clip
    
    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.015          # q accumulation rate
    
    # Agency dynamics (target-tracking variant, VI.2B)
    a_coupling: float = 30.0        # λ in a_target = 1/(1 + λq²)
    a_rate: float = 0.2             # β response rate
    
    # Floor repulsion (IV.6)
    floor_enabled: bool = True
    eta_floor: float = 0.15         # Floor coupling strength
    F_core: float = 5.0             # Floor activation threshold
    floor_power: float = 2.0        # Floor nonlinearity exponent
    
    # Numerical stability
    outflow_limit: float = 0.25     # Max fraction of F that can leave per Δτ


class DETCollider1D:
    """
    DET v6 1D Collider - Minimal Reference Implementation
    """
    
    def __init__(self, params: Optional[DETParams1D] = None):
        self.p = params or DETParams1D()
        N = self.p.N
        
        # Per-node state (II.1)
        self.F = np.ones(N) * self.p.F_VAC      # Free resource
        self.q = np.zeros(N)                     # Structural debt
        self.a = np.ones(N)                      # Agency
        
        # Per-bond state (II.2) - Right direction only (1D)
        self.pi_R = np.zeros(N)                  # Bond momentum (Right)
        self.C_R = np.ones(N) * self.p.C_init    # Coherence (Right)
        self.sigma = np.ones(N)                  # Processing rate
        
        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones(N)                      # Presence (cached)
        self.Delta_tau = np.ones(N) * self.p.DT  # Proper time step (cached)
        
    def add_packet(self, center: int, mass: float = 5.0, 
                   width: float = 5.0, momentum: float = 0):
        """Add a Gaussian resource packet with optional initial momentum."""
        N = self.p.N
        x = np.arange(N)
        r2 = (x - center)**2
        envelope = np.exp(-0.5 * r2 / width**2)
        
        self.F += mass * envelope
        self.C_R = np.clip(self.C_R + 0.5 * envelope, self.p.C_init, 1.0)
        
        if momentum != 0:
            mom_env = np.exp(-0.5 * r2 / (width * 2)**2)
            self.pi_R += momentum * mom_env
        
        self._clip()
    
    def _clip(self):
        """Enforce physical bounds on state variables."""
        self.F = np.clip(self.F, self.p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_R = np.clip(self.pi_R, -self.p.pi_max, self.p.pi_max)
    
    def step(self):
        """Execute one canonical DET update step."""
        p = self.p
        N = p.N
        dk = p.DT
        
        # Neighbor access operators (periodic BCs)
        R = lambda x: np.roll(x, -1)   # Right neighbor
        L = lambda x: np.roll(x, 1)    # Left neighbor
        
        # ============================================================
        # STEP 1: Presence and proper time (III.1)
        # Using H_i = σ_i (degenerate option A)
        # ============================================================
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        # Bond-local time steps (IV.4)
        Delta_tau_R = 0.5 * (self.Delta_tau + R(self.Delta_tau))
        
        # ============================================================
        # STEP 2: Flow computation
        # ============================================================
        
        # Classical (pressure) contribution
        classical_R = self.F - R(self.F)
        classical_L = self.F - L(self.F)
        
        # Coherence interpolation
        sqrt_C_R = np.sqrt(self.C_R)
        sqrt_C_L = np.sqrt(L(self.C_R))
        
        # Combined drive (classical only in 1D for simplicity)
        drive_R = (1 - sqrt_C_R) * classical_R
        drive_L = (1 - sqrt_C_L) * classical_L
        
        # Agency-gated diffusion (IV.2)
        g_R = np.sqrt(self.a * R(self.a))
        g_L = np.sqrt(self.a * L(self.a))
        
        # Conductivity
        cond_R = self.sigma * (self.C_R + 1e-4)
        cond_L = self.sigma * (L(self.C_R) + 1e-4)
        
        # Agency-gated diffusive flux
        J_diff_R = g_R * cond_R * drive_R
        J_diff_L = g_L * cond_L * drive_L
        
        # Momentum-driven flux (IV.4) - NOT agency-gated
        if p.momentum_enabled:
            F_avg_R = 0.5 * (self.F + R(self.F))
            F_avg_L = 0.5 * (self.F + L(self.F))
            
            J_mom_R = p.mu_pi * self.sigma * self.pi_R * F_avg_R
            J_mom_L = -p.mu_pi * self.sigma * L(self.pi_R) * F_avg_L
        else:
            J_mom_R = J_mom_L = 0
        
        # Floor repulsion flux (IV.6) - NOT agency-gated
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_R = p.eta_floor * self.sigma * (s + R(s)) * classical_R
            J_floor_L = p.eta_floor * self.sigma * (s + L(s)) * classical_L
        else:
            J_floor_R = J_floor_L = 0
        
        # Total flux per direction
        J_R = J_diff_R + J_mom_R + J_floor_R
        J_L = J_diff_L + J_mom_L + J_floor_L
        
        # ============================================================
        # STEP 3: Presence-clocked conservative limiter
        # ============================================================
        total_outflow = np.maximum(0, J_R) + np.maximum(0, J_L)
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))
        
        J_R_lim = np.where(J_R > 0, J_R * scale, J_R)
        J_L_lim = np.where(J_L > 0, J_L * scale, J_L)
        
        J_diff_R_scaled = np.where(J_diff_R > 0, J_diff_R * scale, J_diff_R)
        
        # ============================================================
        # STEP 4: Resource update (IV.7)
        # ============================================================
        transfer_R = J_R_lim * self.Delta_tau
        transfer_L = J_L_lim * self.Delta_tau
        
        outflow = transfer_R + transfer_L
        inflow = L(transfer_R) + R(transfer_L)
        
        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)
        
        # ============================================================
        # STEP 5: Momentum update (IV.4)
        # ============================================================
        if p.momentum_enabled:
            decay_R = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_R)
            self.pi_R = decay_R * self.pi_R + p.alpha_pi * J_diff_R_scaled * Delta_tau_R
        
        # ============================================================
        # STEP 6: Structural update (Canonical q-locking)
        # ============================================================
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # ============================================================
        # STEP 7: Agency update (VI.2B - target-tracking)
        # ============================================================
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        self._clip()
        self.time += dk
        self.step_count += 1
    
    def total_mass(self) -> float:
        return float(np.sum(self.F))
    
    def center_of_mass(self) -> float:
        N = self.p.N
        x = np.arange(N)
        total = np.sum(self.F) + 1e-9
        return float(np.sum(x * self.F) / total)
    
    def separation(self) -> float:
        """Find separation between two largest peaks."""
        from scipy.ndimage import label
        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = label(above)
        
        if num < 2:
            return 0.0
        
        N = self.p.N
        x = np.arange(N)
        
        coms = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) == 0:
                continue
            weights = self.F[mask]
            total_mass = np.sum(weights)
            if total_mass < 0.1:
                continue
            com = np.sum(x[mask] * weights) / total_mass
            coms.append({'x': com, 'mass': total_mass})
        
        coms.sort(key=lambda c: -c['mass'])
        
        if len(coms) < 2:
            return 0.0
        
        dx = coms[1]['x'] - coms[0]['x']
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        
        return abs(dx)


# ============================================================
# FALSIFIER TESTS
# ============================================================

def test_F7_mass_conservation(verbose: bool = True) -> bool:
    """
    Falsifier F7: Mass Non-Conservation
    Total mass should not drift by more than 10% over 1000 steps.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F7: Mass Conservation")
        print("="*60)
    
    params = DETParams1D(F_MIN=0.0)  # No vacuum clamp
    sim = DETCollider1D(params)
    sim.add_packet(50, mass=10.0, width=5.0, momentum=0.5)
    sim.add_packet(150, mass=10.0, width=5.0, momentum=-0.5)
    
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


def test_F8_vacuum_momentum(verbose: bool = True) -> bool:
    """
    Falsifier F8: Momentum Pushes Vacuum
    Momentum in vacuum should not produce sustained transport.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F8: Vacuum Momentum")
        print("="*60)
    
    params = DETParams1D(
        momentum_enabled=True,
        q_enabled=False,
        floor_enabled=False,
        F_MIN=0.0
    )
    
    sim = DETCollider1D(params)
    sim.F = np.ones_like(sim.F) * params.F_VAC
    sim.pi_R = np.ones_like(sim.pi_R) * 1.0  # Nonzero momentum
    
    initial_mass = sim.total_mass()
    
    max_J_mom = 0
    for t in range(500):
        R = lambda x: np.roll(x, -1)
        F_avg_R = 0.5 * (sim.F + R(sim.F))
        J_mom_R = params.mu_pi * sim.sigma * sim.pi_R * F_avg_R
        max_J_mom = max(max_J_mom, np.max(np.abs(J_mom_R)))
        sim.step()
    
    final_mass = sim.total_mass()
    drift = abs(final_mass - initial_mass) / initial_mass
    
    passed = max_J_mom < 0.01 and drift < 0.01
    
    if verbose:
        print(f"  Max |J_mom| observed: {max_J_mom:.6f}")
        print(f"  Mass drift: {drift*100:.4f}%")
        print(f"  F8 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F9_symmetry_drift(verbose: bool = True) -> bool:
    """
    Falsifier F9: Spontaneous Drift from Symmetric Rest
    Symmetric initial conditions should not develop COM drift.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F9: Symmetry Drift")
        print("="*60)
    
    params = DETParams1D(momentum_enabled=False)
    sim = DETCollider1D(params)
    
    N = params.N
    sim.add_packet(N//4, mass=5.0, width=4.0, momentum=0)
    sim.add_packet(3*N//4, mass=5.0, width=4.0, momentum=0)
    
    initial_com = sim.center_of_mass()
    
    com_drift = []
    for t in range(1000):
        com = sim.center_of_mass()
        drift = abs(com - initial_com)
        com_drift.append(drift)
        sim.step()
    
    max_drift = max(com_drift)
    passed = max_drift < 1.0
    
    if verbose:
        print(f"  Initial COM: {initial_com:.2f}")
        print(f"  Max COM drift: {max_drift:.4f} cells")
        print(f"  F9 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def run_collision_test(verbose: bool = True) -> Dict:
    """Run a standard 1D collision test."""
    if verbose:
        print("\n" + "="*60)
        print("1D COLLISION TEST")
        print("="*60)
    
    params = DETParams1D()
    sim = DETCollider1D(params)
    
    sim.add_packet(50, mass=8.0, width=5.0, momentum=0.6)
    sim.add_packet(150, mass=8.0, width=5.0, momentum=-0.6)
    
    initial_mass = sim.total_mass()
    
    rec = {'t': [], 'sep': [], 'mass_err': [], 'q_max': [], 'min_a': []}
    
    for t in range(3000):
        sep = sim.separation()
        mass_err = 100 * (sim.total_mass() - initial_mass) / initial_mass
        
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['mass_err'].append(mass_err)
        rec['q_max'].append(np.max(sim.q))
        rec['min_a'].append(np.min(sim.a))
        
        if verbose and t % 500 == 0:
            print(f"  t={t}: sep={sep:.1f}, mass_err={mass_err:+.3f}%, "
                  f"q_max={np.max(sim.q):.3f}, min_a={np.min(sim.a):.3f}")
        
        sim.step()
    
    rec['min_sep'] = min(rec['sep'])
    rec['collision'] = rec['min_sep'] < 5
    rec['final_mass_err'] = rec['mass_err'][-1]
    
    if verbose:
        print(f"\n  Collision: {'YES' if rec['collision'] else 'NO'}")
        print(f"  Min separation: {rec['min_sep']:.1f}")
        print(f"  Final mass error: {rec['final_mass_err']:+.3f}%")
    
    return rec


def run_full_test_suite():
    """Run the complete 1D DET falsifier test suite."""
    print("="*70)
    print("DET v6 1D COLLIDER - FULL TEST SUITE")
    print("="*70)
    
    results = {}
    
    results['collision'] = run_collision_test(verbose=True)
    results['F7'] = test_F7_mass_conservation(verbose=True)
    results['F8'] = test_F8_vacuum_momentum(verbose=True)
    results['F9'] = test_F9_symmetry_drift(verbose=True)
    
    print("\n" + "="*70)
    print("SUITE SUMMARY")
    print("="*70)
    print(f"  Collision test: {'PASS' if results['collision']['collision'] else 'FAIL'}")
    print(f"  F7 (Mass conservation): {'PASS' if results['F7'] else 'FAIL'}")
    print(f"  F8 (Vacuum momentum): {'PASS' if results['F8'] else 'FAIL'}")
    print(f"  F9 (Symmetry drift): {'PASS' if results['F9'] else 'FAIL'}")
    
    return results


if __name__ == "__main__":
    run_full_test_suite()
