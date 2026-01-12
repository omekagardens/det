"""
DET v5 2D Collider - Theory-Faithful Implementation
====================================================

Key corrections from previous version:
1. Agency-gated diffusion: J_diff *= sqrt(a_i * a_j) per DET IV.2
2. Presence-clocked transport: F updates use Δτ, not global dt
3. Outflow limiter references Δτ, not dt
4. Full falsifier test suite included

Reference: DET Theory Card 4.2/5.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time


@dataclass
class DETParams:
    """DET simulation parameters with documented defaults."""
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
    
    # Agency dynamics (target-tracking variant, VI)
    a_coupling: float = 30.0        # λ in a_target = 1/(1 + λq²)
    a_rate: float = 0.2             # β response rate
    
    # Floor repulsion (IV.3a)
    floor_enabled: bool = True
    eta_floor: float = 0.12         # Floor coupling strength
    F_core: float = 4.0             # Floor activation threshold
    floor_power: float = 2.0        # Floor nonlinearity exponent
    
    # Numerical stability
    outflow_limit: float = 0.20     # Max fraction of F that can leave per Δτ
    
    # Phase dynamics (V.0)
    phase_enabled: bool = True
    omega_0: float = 0.0            # Base phase frequency
    gamma_0: float = 0.1            # Phase coupling gain


def periodic_local_sum_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs)."""
    result = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
    return result


class DETCollider2D:
    """
    DET v5 2D Collider - Theory-Faithful Implementation
    
    Implements the canonical DET update loop with:
    - Agency-gated diffusive transport (IV.2)
    - Presence-clocked time evolution (III.1)
    - Momentum dynamics (IV.4, optional)
    - Floor repulsion (IV.3a, optional)
    - Canonical q-locking (Appendix B)
    - Target-tracking agency update (VI)
    """
    
    def __init__(self, params: Optional[DETParams] = None):
        self.p = params or DETParams()
        N = self.p.N
        
        # Per-node state (II.1)
        self.F = np.ones((N, N)) * self.p.F_VAC      # Free resource
        self.q = np.zeros((N, N))                     # Structural debt
        self.theta = np.zeros((N, N))                 # Phase
        self.a = np.ones((N, N))                      # Agency
        
        # Per-bond state (II.2) - East and South directions
        self.pi_E = np.zeros((N, N))                  # Bond momentum (E)
        self.pi_S = np.zeros((N, N))                  # Bond momentum (S)
        self.C_E = np.ones((N, N)) * self.p.C_init    # Coherence (E)
        self.C_S = np.ones((N, N)) * self.p.C_init    # Coherence (S)
        self.sigma = np.ones((N, N))                  # Processing rate
        
        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N))                      # Presence (cached)
        self.Delta_tau = np.ones((N, N)) * self.p.DT  # Proper time step (cached)
        
    def add_packet(self, center: Tuple[int, int], mass: float = 6.0, 
                   width: float = 5.0, momentum: Tuple[float, float] = (0, 0)):
        """Add a Gaussian resource packet with optional initial momentum."""
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
        
        self._clip()
    
    def _clip(self):
        """Enforce physical bounds on state variables."""
        self.F = np.clip(self.F, self.p.F_VAC, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_E = np.clip(self.pi_E, -self.p.pi_max, self.p.pi_max)
        self.pi_S = np.clip(self.pi_S, -self.p.pi_max, self.p.pi_max)
    
    def step(self):
        """
        Execute one canonical DET update step.
        
        Update ordering per Section X:
        1. Compute H_i, P_i, Δτ_i
        2. Compute ψ_i, then all J components
        3. Compute dissipation D_i
        4. (Boundary operators - not implemented in closed system)
        5. Update F
        5a. Update momentum π (if enabled)
        6. Update structure q
        7. Update agency a
        8. Update phase θ (if enabled)
        """
        p = self.p
        N = p.N
        dk = p.DT  # Global step size
        
        # Neighbor access operators (periodic BCs)
        E = lambda x: np.roll(x, -1, axis=1)   # East neighbor
        W = lambda x: np.roll(x, 1, axis=1)    # West neighbor
        S = lambda x: np.roll(x, -1, axis=0)   # South neighbor
        Nb = lambda x: np.roll(x, 1, axis=0)   # North neighbor
        
        # ============================================================
        # STEP 1: Presence and proper time (III.1)
        # P_i = a_i * σ_i / (1 + F_i^op) / (1 + H_i)
        # Using H_i = σ_i (degenerate option A) for simplicity
        # ============================================================
        H = self.sigma  # Coordination load (Option A)
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        # Bond-local time steps (IV.4)
        Delta_tau_E = 0.5 * (self.Delta_tau + E(self.Delta_tau))
        Delta_tau_S = 0.5 * (self.Delta_tau + S(self.Delta_tau))
        
        # ============================================================
        # STEP 2: Flow computation
        # ============================================================
        
        # Local wavefunction (IV.1)
        F_local = periodic_local_sum_2d(self.F, p.R) + 1e-9
        amp = np.sqrt(np.clip(self.F / F_local, 0, 1))
        psi = amp * np.exp(1j * self.theta)
        
        # Quantum (phase) contribution to flow
        quantum_E = np.imag(np.conj(psi) * E(psi))
        quantum_W = np.imag(np.conj(psi) * W(psi))
        quantum_S = np.imag(np.conj(psi) * S(psi))
        quantum_N = np.imag(np.conj(psi) * Nb(psi))
        
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
        
        # Combined drive (quantum-classical interpolated)
        drive_E = sqrt_C_E * quantum_E + (1 - sqrt_C_E) * classical_E
        drive_W = sqrt_C_W * quantum_W + (1 - sqrt_C_W) * classical_W
        drive_S = sqrt_C_S * quantum_S + (1 - sqrt_C_S) * classical_S
        drive_N = sqrt_C_N * quantum_N + (1 - sqrt_C_N) * classical_N
        
        # *** CRITICAL FIX: Agency-gated diffusion (IV.2) ***
        # g^{(a)}_{ij} = sqrt(a_i * a_j)
        g_E = np.sqrt(self.a * E(self.a))
        g_W = np.sqrt(self.a * W(self.a))
        g_S = np.sqrt(self.a * S(self.a))
        g_N = np.sqrt(self.a * Nb(self.a))
        
        # Conductivity includes σ and C
        cond_E = self.sigma * (self.C_E + 1e-4)
        cond_W = self.sigma * (W(self.C_E) + 1e-4)
        cond_S = self.sigma * (self.C_S + 1e-4)
        cond_N = self.sigma * (Nb(self.C_S) + 1e-4)
        
        # Agency-gated diffusive flux
        J_diff_E = g_E * cond_E * drive_E
        J_diff_W = g_W * cond_W * drive_W
        J_diff_S = g_S * cond_S * drive_S
        J_diff_N = g_N * cond_N * drive_N
        
        # Momentum-driven flux (IV.4) - NOT agency-gated
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
        
        # Floor repulsion flux (IV.3a) - NOT agency-gated
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_E = p.eta_floor * self.sigma * (s + E(s)) * classical_E
            J_floor_W = p.eta_floor * self.sigma * (s + W(s)) * classical_W
            J_floor_S = p.eta_floor * self.sigma * (s + S(s)) * classical_S
            J_floor_N = p.eta_floor * self.sigma * (s + Nb(s)) * classical_N
        else:
            J_floor_E = J_floor_W = J_floor_S = J_floor_N = 0
        
        # Total flux per direction
        J_E = J_diff_E + J_mom_E + J_floor_E
        J_W = J_diff_W + J_mom_W + J_floor_W
        J_S = J_diff_S + J_mom_S + J_floor_S
        J_N = J_diff_N + J_mom_N + J_floor_N
        
        # ============================================================
        # *** CRITICAL FIX: Presence-clocked conservative limiter ***
        # Outflow is limited relative to F/Δτ, not F/dt
        # ============================================================
        total_outflow = (np.maximum(0, J_E) + np.maximum(0, J_W) + 
                         np.maximum(0, J_S) + np.maximum(0, J_N))
        # Use Δτ (presence-scaled) for the limit
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))
        
        # Apply limiter to outgoing fluxes only
        J_E_lim = np.where(J_E > 0, J_E * scale, J_E)
        J_W_lim = np.where(J_W > 0, J_W * scale, J_W)
        J_S_lim = np.where(J_S > 0, J_S * scale, J_S)
        J_N_lim = np.where(J_N > 0, J_N * scale, J_N)
        
        # Track scaled diffusive flux for momentum update
        J_diff_E_scaled = np.where(J_diff_E > 0, J_diff_E * scale, J_diff_E)
        J_diff_S_scaled = np.where(J_diff_S > 0, J_diff_S * scale, J_diff_S)
        
        # ============================================================
        # STEP 3: Dissipation (V.2)
        # D_i = Σ|J_{i→j}| * Δτ_i
        # ============================================================
        D = (np.abs(J_E_lim) + np.abs(J_W_lim) + 
             np.abs(J_S_lim) + np.abs(J_N_lim)) * self.Delta_tau
        
        # ============================================================
        # STEP 5: Resource update (IV.3)
        # *** CRITICAL FIX: Use Δτ for transport ***
        # F_i^+ = F_i - Σ J_{i→j} * Δτ_i + I_{g→i}
        # 
        # Sender-clocked transport for conservation:
        # - Transfer_{i→j} = J_{i→j} * Δτ_i
        # - This same amount leaves i and arrives at j
        # ============================================================
        
        # Compute transfer amounts (sender pays from their clock)
        transfer_E = J_E_lim * self.Delta_tau  # from i to East neighbor
        transfer_W = J_W_lim * self.Delta_tau  # from i to West neighbor
        transfer_S = J_S_lim * self.Delta_tau  # from i to South neighbor
        transfer_N = J_N_lim * self.Delta_tau  # from i to North neighbor
        
        # Outflow from i (sum of what i sends to all neighbors)
        outflow = transfer_E + transfer_W + transfer_S + transfer_N
        
        # Inflow to i (receive what neighbors sent, using THEIR transfer amounts)
        # W(transfer_E) = what West neighbor sent East (to us)
        # E(transfer_W) = what East neighbor sent West (to us)
        # etc.
        inflow = W(transfer_E) + E(transfer_W) + Nb(transfer_S) + S(transfer_N)
        
        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_VAC, 1000)
        
        # ============================================================
        # STEP 5a: Momentum update (IV.4)
        # π^+ = (1 - λ_π Δτ)π + α_π J^{diff} Δτ
        # ============================================================
        if p.momentum_enabled:
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)
            self.pi_E = decay_E * self.pi_E + p.alpha_pi * J_diff_E_scaled * Delta_tau_E
            self.pi_S = decay_S * self.pi_S + p.alpha_pi * J_diff_S_scaled * Delta_tau_S
        
        # ============================================================
        # STEP 6: Structural update (Appendix B - canonical q-locking)
        # q^+ = clip(q + α_q * max(0, -ΔF), 0, 1)
        # ============================================================
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # ============================================================
        # STEP 7: Agency update (VI - target-tracking variant)
        # a^+ = a + β(a_target - a), a_target = 1/(1 + λq²)
        # ============================================================
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        # ============================================================
        # STEP 8: Phase update (V.0)
        # θ^+ = θ + ω_0 Δτ + coupling term
        # ============================================================
        if p.phase_enabled:
            # Base evolution
            self.theta = self.theta + p.omega_0 * self.Delta_tau
            
            # Coupling term (optional)
            if p.gamma_0 > 0:
                d_E = np.angle(np.exp(1j * (E(self.theta) - self.theta)))
                d_W = np.angle(np.exp(1j * (W(self.theta) - self.theta)))
                d_S = np.angle(np.exp(1j * (S(self.theta) - self.theta)))
                d_N = np.angle(np.exp(1j * (Nb(self.theta) - self.theta)))
                
                coupling = (p.gamma_0 * self.sigma * 
                           (sqrt_C_E * g_E * d_E + sqrt_C_W * g_W * d_W +
                            sqrt_C_S * g_S * d_S + sqrt_C_N * g_N * d_N))
                self.theta = self.theta + coupling * self.Delta_tau
            
            self.theta = np.mod(self.theta, 2 * np.pi)
        
        # ============================================================
        # Auxiliary updates (coherence, sigma - phenomenological)
        # ============================================================
        # Simple coherence dynamics (placeholder for v5 proper treatment)
        self.C_E = np.clip(self.C_E + 0.05 * np.abs(J_E_lim) * self.Delta_tau 
                          - 0.002 * self.C_E * self.Delta_tau, p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.05 * np.abs(J_S_lim) * self.Delta_tau 
                          - 0.002 * self.C_S * self.Delta_tau, p.C_init, 1.0)
        
        # Processing rate adaptation
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_E_lim) + np.abs(J_S_lim))
        
        self._clip()
        self.time += dk
        self.step_count += 1
    
    def separation(self) -> Tuple[float, int]:
        """Compute separation between two largest blobs."""
        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = ndimage.label(above)
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
            com_x = np.sum(x[mask] * weights) / total_mass
            com_y = np.sum(y[mask] * weights) / total_mass
            coms.append({'x': com_x, 'y': com_y, 'mass': total_mass})
        
        coms.sort(key=lambda c: -c['mass'])
        
        if len(coms) < 2:
            return 0.0, len(coms)
        
        dx = coms[1]['x'] - coms[0]['x']
        dy = coms[1]['y'] - coms[0]['y']
        
        # Handle periodic wrapping
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        
        return np.sqrt(dx**2 + dy**2), len(coms)
    
    def center_of_mass(self) -> Tuple[float, float]:
        """Compute global center of mass (for symmetry tests)."""
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        total = np.sum(self.F) + 1e-9
        com_x = np.sum(x * self.F) / total
        com_y = np.sum(y * self.F) / total
        return com_x, com_y
    
    def peak_count(self, threshold_ratio: float = 0.3) -> int:
        """Count local maxima above threshold."""
        from scipy.ndimage import maximum_filter
        threshold = self.p.F_VAC + threshold_ratio * (np.max(self.F) - self.p.F_VAC)
        local_max = (self.F == maximum_filter(self.F, size=3)) & (self.F > threshold)
        return np.sum(local_max)


# ============================================================
# FALSIFIER TEST SUITE
# ============================================================

def run_collision_test(params: DETParams, steps: int = 12000, 
                       verbose: bool = True) -> Dict:
    """Run a standard collision test and return diagnostics."""
    sim = DETCollider2D(params)
    sim.add_packet((50, 30), mass=6.0, width=5.0, momentum=(0, 0.5))
    sim.add_packet((50, 70), mass=6.0, width=5.0, momentum=(0, -0.5))
    
    initial_F = np.sum(sim.F)
    
    rec = {
        't': [], 'sep': [], 'mass_err': [], 'blobs': [], 
        'q_max': [], 'min_a': [], 'peaks': [], 'P_min': []
    }
    snapshots = []
    
    for t in range(steps):
        sep, num = sim.separation()
        mass_err = 100 * (np.sum(sim.F) - initial_F) / initial_F
        
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['mass_err'].append(mass_err)
        rec['blobs'].append(num)
        rec['q_max'].append(np.max(sim.q))
        rec['min_a'].append(np.min(sim.a))
        rec['peaks'].append(sim.peak_count())
        rec['P_min'].append(np.min(sim.P))
        
        if t in [0, steps//6, 2*steps//6, 3*steps//6, 4*steps//6, 5*steps//6]:
            snapshots.append((t, sim.F.copy(), sim.q.copy(), sim.a.copy()))
        
        if verbose and t % 2000 == 0:
            print(f"  t={t}: sep={sep:.1f}, blobs={num}, mass_err={mass_err:+.3f}%, "
                  f"q_max={np.max(sim.q):.3f}, min_a={np.min(sim.a):.3f}")
        
        sim.step()
    
    rec['snapshots'] = snapshots
    rec['min_sep'] = min(rec['sep'])
    rec['final_mass_err'] = rec['mass_err'][-1]
    rec['collision'] = rec['min_sep'] < 5
    
    return rec


def test_F8_vacuum_momentum(verbose: bool = True) -> bool:
    """
    Falsifier F8: Momentum pushes vacuum
    
    With momentum module enabled, initializing π≠0 on bonds in a region 
    with negligible F≈0 should produce NO sustained transport.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F8: Vacuum Momentum Falsifier")
        print("="*60)
    
    params = DETParams(
        momentum_enabled=True,
        q_enabled=False,
        floor_enabled=False
    )
    
    sim = DETCollider2D(params)
    # Initialize with near-vacuum F but nonzero momentum
    sim.F = np.ones_like(sim.F) * params.F_VAC
    sim.pi_E = np.ones_like(sim.pi_E) * 1.0  # Nonzero momentum
    sim.pi_S = np.ones_like(sim.pi_S) * 1.0
    
    initial_F = np.sum(sim.F)
    
    # Run for a while
    max_J_mom = 0
    for t in range(500):
        # Compute momentum flux manually to check
        E = lambda x: np.roll(x, -1, axis=1)
        F_avg_E = 0.5 * (sim.F + E(sim.F))
        J_mom_E = params.mu_pi * sim.sigma * sim.pi_E * F_avg_E
        max_J_mom = max(max_J_mom, np.max(np.abs(J_mom_E)))
        sim.step()
    
    final_F = np.sum(sim.F)
    drift = abs(final_F - initial_F) / initial_F
    
    # F8 passes if momentum-driven flux is negligible in vacuum
    # (F_avg ≈ F_VAC is tiny, so J_mom should be tiny)
    passed = max_J_mom < 0.01 and drift < 0.01
    
    if verbose:
        print(f"  Max |J_mom| observed: {max_J_mom:.6f}")
        print(f"  Mass drift: {drift*100:.4f}%")
        print(f"  F8 {'PASSED' if passed else 'FAILED'}: " + 
              ("Momentum flux vanishes in vacuum" if passed else 
               "Momentum incorrectly pushes vacuum!"))
    
    return passed


def test_F9_symmetry_drift(verbose: bool = True) -> bool:
    """
    Falsifier F9: Spontaneous drift from symmetric rest
    
    With symmetric initial conditions and π=0, the system should NOT
    develop persistent net drift (COM should stay fixed).
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F9: Symmetry Drift Falsifier")
        print("="*60)
    
    params = DETParams(momentum_enabled=False)  # No momentum to ensure symmetry
    sim = DETCollider2D(params)
    
    # Symmetric setup: two identical packets, symmetric positions
    N = params.N
    sim.add_packet((N//2, N//4), mass=5.0, width=4.0, momentum=(0, 0))
    sim.add_packet((N//2, 3*N//4), mass=5.0, width=4.0, momentum=(0, 0))
    
    initial_com = sim.center_of_mass()
    
    com_drift = []
    for t in range(2000):
        com = sim.center_of_mass()
        drift = np.sqrt((com[0] - initial_com[0])**2 + (com[1] - initial_com[1])**2)
        com_drift.append(drift)
        sim.step()
    
    max_drift = max(com_drift)
    passed = max_drift < 1.0  # Less than 1 cell drift
    
    if verbose:
        print(f"  Initial COM: ({initial_com[0]:.2f}, {initial_com[1]:.2f})")
        print(f"  Max COM drift: {max_drift:.4f} cells")
        print(f"  F9 {'PASSED' if passed else 'FAILED'}: " +
              ("Symmetry preserved" if passed else "Spontaneous symmetry breaking!"))
    
    return passed


def run_ablation_matrix(verbose: bool = True) -> Dict:
    """
    Ablation test: Compare behavior with different module combinations.
    
    Runs 4 configurations:
    (i)   No momentum, no floor
    (ii)  Momentum only
    (iii) Floor only
    (iv)  Both (full model)
    """
    if verbose:
        print("\n" + "="*60)
        print("ABLATION MATRIX: Module Combinations")
        print("="*60)
    
    configs = [
        ("No mom, no floor", DETParams(momentum_enabled=False, floor_enabled=False)),
        ("Momentum only", DETParams(momentum_enabled=True, floor_enabled=False)),
        ("Floor only", DETParams(momentum_enabled=False, floor_enabled=True)),
        ("Both (full)", DETParams(momentum_enabled=True, floor_enabled=True)),
    ]
    
    results = {}
    for name, params in configs:
        if verbose:
            print(f"\n  Running: {name}")
        rec = run_collision_test(params, steps=8000, verbose=False)
        results[name] = {
            'min_sep': rec['min_sep'],
            'collision': rec['collision'],
            'final_blobs': rec['blobs'][-1],
            'mass_err': rec['final_mass_err'],
            'max_q': max(rec['q_max']),
            'min_a': min(rec['min_a']),
            'final_peaks': rec['peaks'][-1]
        }
        if verbose:
            r = results[name]
            print(f"    min_sep={r['min_sep']:.1f}, collision={r['collision']}, "
                  f"blobs={r['final_blobs']}, peaks={r['final_peaks']}, "
                  f"mass_err={r['mass_err']:+.2f}%")
    
    return results


def run_q_sensitivity_scan(verbose: bool = True) -> Dict:
    """
    Parameter sensitivity: Scan alpha_q and a_coupling.
    
    Looking for regimes where:
    - q doesn't instantly saturate
    - Collision still occurs
    - Capture vs merge discrimination
    """
    if verbose:
        print("\n" + "="*60)
        print("Q-LOCKING SENSITIVITY SCAN")
        print("="*60)
    
    alpha_q_vals = [0.005, 0.010, 0.015, 0.025]
    a_coupling_vals = [10.0, 20.0, 30.0, 50.0]
    
    results = {}
    for alpha_q in alpha_q_vals:
        for a_coupling in a_coupling_vals:
            params = DETParams(alpha_q=alpha_q, a_coupling=a_coupling)
            key = f"αq={alpha_q}, λ={a_coupling}"
            
            if verbose:
                print(f"\n  {key}")
            
            rec = run_collision_test(params, steps=6000, verbose=False)
            results[key] = {
                'min_sep': rec['min_sep'],
                'collision': rec['collision'],
                'max_q': max(rec['q_max']),
                'min_a': min(rec['min_a']),
                'mass_err': rec['final_mass_err'],
                'final_peaks': rec['peaks'][-1]
            }
            
            if verbose:
                r = results[key]
                print(f"    min_sep={r['min_sep']:.1f}, max_q={r['max_q']:.3f}, "
                      f"min_a={r['min_a']:.3f}, peaks={r['final_peaks']}")
    
    return results


def run_full_test_suite():
    """Run the complete DET falsifier test suite."""
    print("="*70)
    print("DET v5 2D COLLIDER - FULL TEST SUITE")
    print("Theory-Faithful Implementation with Agency-Gated Transport")
    print("="*70)
    
    # Main collision test
    print("\n" + "="*60)
    print("MAIN COLLISION TEST")
    print("="*60)
    
    params = DETParams()
    print(f"\nParameters (reporting per card requirements):")
    print(f"  DT={params.DT}, N={params.N}, R={params.R}")
    print(f"  Momentum: α_π={params.alpha_pi}, λ_π={params.lambda_pi}, "
          f"μ_π={params.mu_pi}, π_max={params.pi_max}")
    print(f"  Floor: η_f={params.eta_floor}, F_core={params.F_core}, p={params.floor_power}")
    print(f"  Q-lock: α_q={params.alpha_q}")
    print(f"  Agency: β={params.a_rate}, λ={params.a_coupling}")
    
    rec = run_collision_test(params, steps=12000, verbose=True)
    
    print(f"\n{'='*60}")
    print("MAIN TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Collision occurred: {'YES' if rec['collision'] else 'NO'}")
    print(f"  Min separation: {rec['min_sep']:.1f}")
    print(f"  Final blob count: {rec['blobs'][-1]}")
    print(f"  Final peak count: {rec['peaks'][-1]}")
    print(f"  Mass conservation error: {rec['final_mass_err']:+.3f}%")
    print(f"  Max q reached: {max(rec['q_max']):.4f}")
    print(f"  Min agency reached: {min(rec['min_a']):.4f}")
    print(f"  Min presence reached: {min(rec['P_min']):.6f}")
    
    # Falsifier tests
    f8_pass = test_F8_vacuum_momentum(verbose=True)
    f9_pass = test_F9_symmetry_drift(verbose=True)
    
    # Ablation matrix
    ablation = run_ablation_matrix(verbose=True)
    
    # Q sensitivity (abbreviated)
    print("\n" + "="*60)
    print("Q-SENSITIVITY SCAN (abbreviated)")
    print("="*60)
    q_scan = run_q_sensitivity_scan(verbose=True)
    
    # Summary
    print("\n" + "="*70)
    print("SUITE SUMMARY")
    print("="*70)
    print(f"  F8 (Vacuum momentum): {'PASS' if f8_pass else 'FAIL'}")
    print(f"  F9 (Symmetry drift): {'PASS' if f9_pass else 'FAIL'}")
    print(f"  Main collision: {'PASS' if rec['collision'] else 'FAIL'}")
    print(f"  Mass conservation (<10%): {'PASS' if abs(rec['final_mass_err']) < 10 else 'FAIL'}")
    
    # Binding assessment
    binding_achieved = rec['collision'] and rec['blobs'][-1] == 1
    print(f"\n  Binding (F6-style): {'Merge achieved' if binding_achieved else 'No stable merge'}")
    
    return {
        'main': rec,
        'f8': f8_pass,
        'f9': f9_pass,
        'ablation': ablation,
        'q_scan': q_scan
    }


def create_visualization(rec: Dict, filename: str = 'det_v5_2d_fixed.png'):
    """Create visualization of collision results."""
    fig = plt.figure(figsize=(18, 14))
    
    snapshots = rec['snapshots']
    
    # Row 1: F snapshots
    for i, (t, F, q, a) in enumerate(snapshots):
        ax = fig.add_subplot(4, 6, i + 1)
        im = ax.imshow(F, cmap='plasma', vmin=0)
        ax.set_title(f'F t={t}', fontsize=9)
        ax.axis('off')
    
    # Row 2: Agency snapshots
    for i, (t, F, q, a) in enumerate(snapshots):
        ax = fig.add_subplot(4, 6, i + 7)
        ax.imshow(a, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'Agency t={t}', fontsize=9)
        ax.axis('off')
    
    # Row 3: Time series
    ax = fig.add_subplot(4, 3, 7)
    ax.plot(rec['t'], rec['sep'], 'b-', lw=1.5)
    ax.axhline(5, color='g', ls='--', alpha=0.5, label='Collision threshold')
    ax.fill_between(rec['t'], 0, rec['sep'], 
                    where=[s < 10 for s in rec['sep']], alpha=0.2, color='green')
    ax.set_xlabel('Step')
    ax.set_ylabel('Separation')
    ax.set_title('Inter-body Separation')
    ax.legend(fontsize=8)
    
    ax = fig.add_subplot(4, 3, 8)
    ax.plot(rec['t'], rec['min_a'], 'g-', lw=1.5, label='min(a)')
    ax.plot(rec['t'], rec['q_max'], 'r-', lw=1.5, label='max(q)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Agency & Structure Evolution')
    ax.legend(fontsize=8)
    
    ax = fig.add_subplot(4, 3, 9)
    ax.plot(rec['t'], rec['mass_err'], 'm-', lw=1.5)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.axhline(10, color='r', ls='--', alpha=0.3, label='F7 threshold')
    ax.axhline(-10, color='r', ls='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mass Error %')
    ax.set_title('Mass Conservation (F7)')
    ax.legend(fontsize=8)
    
    # Row 4: Additional diagnostics
    ax = fig.add_subplot(4, 3, 10)
    ax.plot(rec['t'], rec['blobs'], 'k-', lw=1, label='Blobs')
    ax.plot(rec['t'], rec['peaks'], 'b--', lw=1, label='Peaks')
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title('Blob & Peak Count')
    ax.legend(fontsize=8)
    
    ax = fig.add_subplot(4, 3, 11)
    ax.plot(rec['t'], rec['P_min'], 'purple', lw=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('min(P)')
    ax.set_title('Minimum Presence (Clock Freeze)')
    ax.set_yscale('log')
    
    ax = fig.add_subplot(4, 3, 12)
    ax.axis('off')
    summary = f"""
DET v5 2D Collider - FIXED
==========================
Theory-faithful implementation with:
✓ Agency-gated diffusion (IV.2)
✓ Presence-clocked transport (Δτ)
✓ Conservative limiter (Appendix N)

Results:
  Collision: {'YES' if rec['collision'] else 'NO'}
  Min separation: {rec['min_sep']:.1f}
  Final blobs: {rec['blobs'][-1]}
  Mass error: {rec['final_mass_err']:+.3f}%

Mechanism (now faithful):
  q accumulates on compression
  High q → low a (target-tracking)
  Low a → diffusion GATED (g_ij)
  Floor/momentum → binding forces
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DET v5 2D Collider - Theory-Faithful Implementation', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")


if __name__ == "__main__":
    start = time.time()
    
    results = run_full_test_suite()
    
    # Create visualization
    create_visualization(results['main'], './det_v5_2d_fixed.png')
    
    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s")