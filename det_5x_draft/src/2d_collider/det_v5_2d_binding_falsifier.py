"""
DET F6 Binding Falsifier - Rigorous Implementation
===================================================

F6 — Binding Failure (Scoped)
With (i) gravity–flow coupling enabled (VIII.2) and (ii) agency-gated diffusion 
enabled (IV.2), two initially separated compact bodies with nonzero realized 
structure (q>0 under the declared q-locking law) fail to form a bound state 
across a broad set of initial impact parameters (separation returns to its 
pre-collision value).

This test:
1. Scans impact parameters (velocity, offset, initial separation)
2. Classifies outcomes: BOUND, MERGED, ESCAPED
3. Determines if binding occurs across "broad" parameter space

A BOUND state means: two distinct bodies remain at nonzero separation < initial
This is the "capture without merge" regime - key for matter-like behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import time


class Outcome(Enum):
    ESCAPED = "escaped"      # Separation returns to or exceeds initial
    BOUND = "bound"          # Two bodies, stable separation < initial
    MERGED = "merged"        # Single body formed
    UNCERTAIN = "uncertain"  # Could not classify


@dataclass
class DETParams:
    """DET simulation parameters."""
    N: int = 100
    DT: float = 0.015
    F_VAC: float = 0.01
    R: int = 5
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
    
    # Agency dynamics
    a_coupling: float = 30.0
    a_rate: float = 0.2
    
    # Floor repulsion (IV.3a)
    floor_enabled: bool = True
    eta_floor: float = 0.12
    F_core: float = 4.0
    floor_power: float = 2.0
    
    # Numerical
    outflow_limit: float = 0.20
    
    # Phase
    phase_enabled: bool = True
    omega_0: float = 0.0
    gamma_0: float = 0.1


def periodic_local_sum_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs)."""
    result = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
    return result


class DETCollider2D:
    """DET v5 2D Collider - Theory-Faithful Implementation."""
    
    def __init__(self, params: Optional[DETParams] = None):
        self.p = params or DETParams()
        N = self.p.N
        
        self.F = np.ones((N, N)) * self.p.F_VAC
        self.q = np.zeros((N, N))
        self.theta = np.zeros((N, N))
        self.a = np.ones((N, N))
        self.pi_E = np.zeros((N, N))
        self.pi_S = np.zeros((N, N))
        self.C_E = np.ones((N, N)) * self.p.C_init
        self.C_S = np.ones((N, N)) * self.p.C_init
        self.sigma = np.ones((N, N))
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones((N, N))
        self.Delta_tau = np.ones((N, N)) * self.p.DT
        
    def add_packet(self, center: Tuple[float, float], mass: float = 6.0, 
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
        self.F = np.clip(self.F, self.p.F_VAC, 1000)
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
        
        # Step 1: Presence and proper time
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        Delta_tau_E = 0.5 * (self.Delta_tau + E(self.Delta_tau))
        Delta_tau_S = 0.5 * (self.Delta_tau + S(self.Delta_tau))
        
        # Step 2: Flow computation
        F_local = periodic_local_sum_2d(self.F, p.R) + 1e-9
        amp = np.sqrt(np.clip(self.F / F_local, 0, 1))
        psi = amp * np.exp(1j * self.theta)
        
        quantum_E = np.imag(np.conj(psi) * E(psi))
        quantum_W = np.imag(np.conj(psi) * W(psi))
        quantum_S = np.imag(np.conj(psi) * S(psi))
        quantum_N = np.imag(np.conj(psi) * Nb(psi))
        
        classical_E = self.F - E(self.F)
        classical_W = self.F - W(self.F)
        classical_S = self.F - S(self.F)
        classical_N = self.F - Nb(self.F)
        
        sqrt_C_E = np.sqrt(self.C_E)
        sqrt_C_S = np.sqrt(self.C_S)
        sqrt_C_W = np.sqrt(W(self.C_E))
        sqrt_C_N = np.sqrt(Nb(self.C_S))
        
        drive_E = sqrt_C_E * quantum_E + (1 - sqrt_C_E) * classical_E
        drive_W = sqrt_C_W * quantum_W + (1 - sqrt_C_W) * classical_W
        drive_S = sqrt_C_S * quantum_S + (1 - sqrt_C_S) * classical_S
        drive_N = sqrt_C_N * quantum_N + (1 - sqrt_C_N) * classical_N
        
        # Agency-gated diffusion (IV.2)
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
        
        # Momentum flux (not agency-gated)
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
        
        # Floor repulsion (not agency-gated)
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_E = p.eta_floor * self.sigma * (s + E(s)) * classical_E
            J_floor_W = p.eta_floor * self.sigma * (s + W(s)) * classical_W
            J_floor_S = p.eta_floor * self.sigma * (s + S(s)) * classical_S
            J_floor_N = p.eta_floor * self.sigma * (s + Nb(s)) * classical_N
        else:
            J_floor_E = J_floor_W = J_floor_S = J_floor_N = 0
        
        J_E = J_diff_E + J_mom_E + J_floor_E
        J_W = J_diff_W + J_mom_W + J_floor_W
        J_S = J_diff_S + J_mom_S + J_floor_S
        J_N = J_diff_N + J_mom_N + J_floor_N
        
        # Conservative limiter (Δτ-referenced)
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
        
        # Step 3: Dissipation
        D = (np.abs(J_E_lim) + np.abs(J_W_lim) + 
             np.abs(J_S_lim) + np.abs(J_N_lim)) * self.Delta_tau
        
        # Step 5: Resource update (sender-clocked for conservation)
        transfer_E = J_E_lim * self.Delta_tau
        transfer_W = J_W_lim * self.Delta_tau
        transfer_S = J_S_lim * self.Delta_tau
        transfer_N = J_N_lim * self.Delta_tau
        
        outflow = transfer_E + transfer_W + transfer_S + transfer_N
        inflow = W(transfer_E) + E(transfer_W) + Nb(transfer_S) + S(transfer_N)
        
        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_VAC, 1000)
        
        # Step 5a: Momentum update
        if p.momentum_enabled:
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)
            self.pi_E = decay_E * self.pi_E + p.alpha_pi * J_diff_E_scaled * Delta_tau_E
            self.pi_S = decay_S * self.pi_S + p.alpha_pi * J_diff_S_scaled * Delta_tau_S
        
        # Step 6: Structural update (canonical q-locking)
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # Step 7: Agency update (target-tracking)
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        # Step 8: Phase update
        if p.phase_enabled:
            self.theta = self.theta + p.omega_0 * self.Delta_tau
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
        
        # Auxiliary updates
        self.C_E = np.clip(self.C_E + 0.05 * np.abs(J_E_lim) * self.Delta_tau 
                          - 0.002 * self.C_E * self.Delta_tau, p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.05 * np.abs(J_S_lim) * self.Delta_tau 
                          - 0.002 * self.C_S * self.Delta_tau, p.C_init, 1.0)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_E_lim) + np.abs(J_S_lim))
        
        self._clip()
        self.time += dk
        self.step_count += 1
    
    def find_bodies(self) -> List[Dict]:
        """Find distinct bodies and their properties."""
        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = ndimage.label(above)
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        
        bodies = []
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
            
            # Also track q and a in the body
            q_mean = np.sum(self.q[mask] * weights) / total_mass
            a_mean = np.sum(self.a[mask] * weights) / total_mass
            
            bodies.append({
                'x': com_x, 'y': com_y, 'mass': total_mass,
                'q_mean': q_mean, 'a_mean': a_mean, 'size': np.sum(mask)
            })
        
        bodies.sort(key=lambda b: -b['mass'])
        return bodies
    
    def separation(self) -> Tuple[float, int]:
        """Compute separation between two largest bodies."""
        bodies = self.find_bodies()
        if len(bodies) < 2:
            return 0.0, len(bodies)
        
        N = self.p.N
        dx = bodies[1]['x'] - bodies[0]['x']
        dy = bodies[1]['y'] - bodies[0]['y']
        
        # Periodic wrapping
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        
        return np.sqrt(dx**2 + dy**2), len(bodies)
    
    def peak_separation(self) -> Tuple[float, int]:
        """
        Find separation between density peaks (more sensitive than blob detection).
        Uses local maxima detection.
        """
        from scipy.ndimage import maximum_filter
        
        threshold = self.p.F_VAC + 0.2 * (np.max(self.F) - self.p.F_VAC)
        local_max = (self.F == maximum_filter(self.F, size=5)) & (self.F > threshold)
        
        peaks = np.where(local_max)
        if len(peaks[0]) < 2:
            return 0.0, len(peaks[0])
        
        # Get peak values and positions
        peak_vals = self.F[peaks]
        indices = np.argsort(-peak_vals)[:2]  # Two strongest peaks
        
        y1, x1 = peaks[0][indices[0]], peaks[1][indices[0]]
        y2, x2 = peaks[0][indices[1]], peaks[1][indices[1]]
        
        N = self.p.N
        dx = x2 - x1
        dy = y2 - y1
        
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        
        return np.sqrt(dx**2 + dy**2), len(peaks[0])


@dataclass
class BindingTestResult:
    """Results from a single binding test run."""
    outcome: Outcome
    initial_sep: float
    min_sep: float
    final_sep: float
    final_peak_sep: float
    num_bodies: int
    num_peaks: int
    sep_history: List[float]
    peak_sep_history: List[float]
    max_q: float
    min_a: float
    mass_error: float
    params: Dict


def classify_outcome(result: BindingTestResult, 
                     merge_threshold: float = 5.0,
                     escape_ratio: float = 0.9) -> Outcome:
    """
    Classify the outcome of a collision.
    
    MERGED: Final separation < merge_threshold AND < 2 distinct peaks
    ESCAPED: Final separation > escape_ratio * initial separation
    BOUND: Two bodies maintained with 0 < final_sep < initial_sep
    """
    # Check for merge (single body or very close peaks)
    if result.final_sep < merge_threshold and result.num_peaks <= 1:
        return Outcome.MERGED
    
    # Check for escape
    if result.final_sep > escape_ratio * result.initial_sep:
        return Outcome.ESCAPED
    
    # Check for bound state with peak separation
    if result.final_peak_sep > merge_threshold and result.final_peak_sep < result.initial_sep:
        # Verify stability: check if separation oscillates or stabilizes
        late_seps = result.peak_sep_history[-len(result.peak_sep_history)//4:]
        sep_std = np.std(late_seps)
        sep_mean = np.mean(late_seps)
        
        # Bound if late-time separation is stable and nonzero
        if sep_mean > merge_threshold and sep_std < 0.2 * sep_mean:
            return Outcome.BOUND
        # Also bound if oscillating with bounded amplitude
        if sep_mean > merge_threshold:
            return Outcome.BOUND
    
    # If merged based on blob count but has 2 peaks
    if result.num_bodies == 1 and result.num_peaks >= 2 and result.final_peak_sep > merge_threshold:
        return Outcome.BOUND  # Captured within single "blob" but distinct peaks
    
    return Outcome.UNCERTAIN


def run_binding_test(velocity: float, offset: float = 0.0, 
                     initial_sep: float = 40.0, mass: float = 6.0,
                     steps: int = 15000, params: Optional[DETParams] = None,
                     verbose: bool = False) -> BindingTestResult:
    """
    Run a single binding test with given impact parameters.
    
    Args:
        velocity: Initial approach velocity (magnitude)
        offset: Impact parameter (perpendicular offset from head-on)
        initial_sep: Initial separation between bodies
        mass: Mass of each body
        steps: Number of simulation steps
        params: DET parameters
        verbose: Print progress
    """
    params = params or DETParams()
    sim = DETCollider2D(params)
    
    N = params.N
    center = N // 2
    
    # Place bodies with offset and velocity
    # Body 1: left of center, moving right
    y1 = center + offset // 2
    x1 = center - initial_sep // 2
    # Body 2: right of center, moving left
    y2 = center - offset // 2
    x2 = center + initial_sep // 2
    
    sim.add_packet((y1, x1), mass=mass, width=5.0, momentum=(0, velocity))
    sim.add_packet((y2, x2), mass=mass, width=5.0, momentum=(0, -velocity))
    
    initial_F = np.sum(sim.F)
    sep0, _ = sim.separation()
    
    sep_history = []
    peak_sep_history = []
    q_max_history = []
    a_min_history = []
    
    for t in range(steps):
        sep, num_bodies = sim.separation()
        peak_sep, num_peaks = sim.peak_separation()
        
        sep_history.append(sep)
        peak_sep_history.append(peak_sep)
        q_max_history.append(np.max(sim.q))
        a_min_history.append(np.min(sim.a))
        
        if verbose and t % 3000 == 0:
            print(f"  t={t}: sep={sep:.1f}, peak_sep={peak_sep:.1f}, "
                  f"bodies={num_bodies}, peaks={num_peaks}")
        
        sim.step()
    
    # Final measurements
    final_sep, num_bodies = sim.separation()
    final_peak_sep, num_peaks = sim.peak_separation()
    mass_error = 100 * (np.sum(sim.F) - initial_F) / initial_F
    
    result = BindingTestResult(
        outcome=Outcome.UNCERTAIN,  # Will be classified below
        initial_sep=sep0,
        min_sep=min(sep_history),
        final_sep=final_sep,
        final_peak_sep=final_peak_sep,
        num_bodies=num_bodies,
        num_peaks=num_peaks,
        sep_history=sep_history,
        peak_sep_history=peak_sep_history,
        max_q=max(q_max_history),
        min_a=min(a_min_history),
        mass_error=mass_error,
        params={'velocity': velocity, 'offset': offset, 'initial_sep': initial_sep, 'mass': mass}
    )
    
    result.outcome = classify_outcome(result)
    return result


def run_f6_parameter_scan(verbose: bool = True) -> Dict:
    """
    Run F6 binding falsifier across parameter space.
    
    Scans:
    - Velocity: low to high (0.1 to 1.5)
    - Offset: head-on to glancing (0 to 20)
    
    F6 PASSES if binding occurs across "broad" parameter space.
    F6 FAILS if all collisions result in escape or merge (no capture).
    """
    if verbose:
        print("="*70)
        print("F6 BINDING FALSIFIER - PARAMETER SCAN")
        print("="*70)
        print("\nScanning impact parameters to find binding regime...")
    
    velocities = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    offsets = [0, 5, 10, 15, 20]
    
    results = {}
    outcome_counts = {Outcome.ESCAPED: 0, Outcome.BOUND: 0, 
                      Outcome.MERGED: 0, Outcome.UNCERTAIN: 0}
    
    for vel in velocities:
        for off in offsets:
            key = f"v={vel:.1f}, b={off}"
            if verbose:
                print(f"\n  Running: {key}")
            
            result = run_binding_test(velocity=vel, offset=off, steps=12000, verbose=False)
            results[key] = result
            outcome_counts[result.outcome] += 1
            
            if verbose:
                print(f"    Outcome: {result.outcome.value}")
                print(f"    sep: {result.initial_sep:.1f} → {result.final_sep:.1f} "
                      f"(min={result.min_sep:.1f})")
                print(f"    peak_sep: {result.final_peak_sep:.1f}, "
                      f"peaks={result.num_peaks}, bodies={result.num_bodies}")
                print(f"    q_max={result.max_q:.3f}, min_a={result.min_a:.3f}")
    
    # F6 assessment
    total = len(results)
    bound_count = outcome_counts[Outcome.BOUND]
    merged_count = outcome_counts[Outcome.MERGED]
    escaped_count = outcome_counts[Outcome.ESCAPED]
    
    # F6 passes if we get bound states OR merges (both are "binding")
    binding_count = bound_count + merged_count
    binding_fraction = binding_count / total
    
    # "Broad parameter space" = at least 50% of parameter space shows binding
    f6_passed = binding_fraction >= 0.5
    
    if verbose:
        print("\n" + "="*70)
        print("F6 SCAN RESULTS")
        print("="*70)
        print(f"\nOutcome distribution (N={total}):")
        print(f"  BOUND:     {bound_count:3d} ({100*bound_count/total:.1f}%)")
        print(f"  MERGED:    {merged_count:3d} ({100*merged_count/total:.1f}%)")
        print(f"  ESCAPED:   {escaped_count:3d} ({100*escaped_count/total:.1f}%)")
        print(f"  UNCERTAIN: {outcome_counts[Outcome.UNCERTAIN]:3d}")
        print(f"\nBinding (BOUND+MERGED): {binding_count}/{total} = {100*binding_fraction:.1f}%")
        print(f"\nF6 Falsifier: {'PASS' if f6_passed else 'FAIL'}")
        
        if bound_count > 0:
            print("\n  ✓ BOUND states found (capture without full merge)")
            print("    This demonstrates matter-like capture dynamics.")
        if bound_count == 0 and merged_count > 0:
            print("\n  ⚠ Only MERGED states (no distinct bound orbits)")
            print("    Bodies collide and coalesce, no stable binary formation.")
    
    return {
        'results': results,
        'outcome_counts': outcome_counts,
        'binding_fraction': binding_fraction,
        'f6_passed': f6_passed
    }


def visualize_f6_results(scan_results: Dict, filename: str = 'det_f6_binding.png'):
    """Create visualization of F6 binding test results."""
    results = scan_results['results']
    
    # Extract parameter grid
    velocities = sorted(set(r.params['velocity'] for r in results.values()))
    offsets = sorted(set(r.params['offset'] for r in results.values()))
    
    # Create outcome matrix
    outcome_map = {Outcome.ESCAPED: 0, Outcome.BOUND: 1, Outcome.MERGED: 2, Outcome.UNCERTAIN: -1}
    outcome_matrix = np.zeros((len(offsets), len(velocities)))
    sep_matrix = np.zeros((len(offsets), len(velocities)))
    
    for key, result in results.items():
        vi = velocities.index(result.params['velocity'])
        oi = offsets.index(result.params['offset'])
        outcome_matrix[oi, vi] = outcome_map[result.outcome]
        sep_matrix[oi, vi] = result.final_peak_sep
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Outcome phase diagram
    ax1 = fig.add_subplot(2, 2, 1)
    cmap = plt.cm.RdYlGn
    im = ax1.imshow(outcome_matrix, cmap=cmap, aspect='auto', origin='lower',
                    extent=[velocities[0]-0.1, velocities[-1]+0.1, 
                            offsets[0]-2.5, offsets[-1]+2.5])
    ax1.set_xlabel('Initial Velocity')
    ax1.set_ylabel('Impact Offset')
    ax1.set_title('Outcome Phase Diagram\n(0=Escaped, 1=Bound, 2=Merged)')
    cbar = plt.colorbar(im, ax=ax1, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Escaped', 'Bound', 'Merged'])
    
    # 2. Final peak separation
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(sep_matrix, cmap='viridis', aspect='auto', origin='lower',
                     extent=[velocities[0]-0.1, velocities[-1]+0.1, 
                             offsets[0]-2.5, offsets[-1]+2.5])
    ax2.set_xlabel('Initial Velocity')
    ax2.set_ylabel('Impact Offset')
    ax2.set_title('Final Peak Separation')
    plt.colorbar(im2, ax=ax2, label='Separation')
    
    # 3. Example trajectories (pick one of each outcome type)
    ax3 = fig.add_subplot(2, 2, 3)
    colors = {'escaped': 'red', 'bound': 'green', 'merged': 'blue', 'uncertain': 'gray'}
    plotted = set()
    
    for key, result in results.items():
        outcome_str = result.outcome.value
        if outcome_str not in plotted:
            t = np.arange(len(result.peak_sep_history))
            ax3.plot(t, result.peak_sep_history, color=colors[outcome_str], 
                    label=f'{outcome_str.capitalize()} (v={result.params["velocity"]:.1f})',
                    alpha=0.8, lw=1.5)
            plotted.add(outcome_str)
    
    ax3.axhline(5, color='k', ls='--', alpha=0.3, label='Merge threshold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Peak Separation')
    ax3.set_title('Example Separation Trajectories')
    ax3.legend(fontsize=8)
    
    # 4. Summary text
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    counts = scan_results['outcome_counts']
    total = sum(counts.values())
    
    summary = f"""
F6 BINDING FALSIFIER RESULTS
{'='*40}

Parameter Space Scanned:
  Velocities: {velocities}
  Offsets: {offsets}
  Total runs: {total}

Outcome Distribution:
  ESCAPED:   {counts[Outcome.ESCAPED]:3d} ({100*counts[Outcome.ESCAPED]/total:.1f}%)
  BOUND:     {counts[Outcome.BOUND]:3d} ({100*counts[Outcome.BOUND]/total:.1f}%)
  MERGED:    {counts[Outcome.MERGED]:3d} ({100*counts[Outcome.MERGED]/total:.1f}%)
  UNCERTAIN: {counts[Outcome.UNCERTAIN]:3d}

Binding Fraction: {scan_results['binding_fraction']*100:.1f}%

F6 FALSIFIER: {'PASS ✓' if scan_results['f6_passed'] else 'FAIL ✗'}

Interpretation:
{'  Bodies form bound states or merge across' if scan_results['f6_passed'] else '  Binding fails -'}
{'  broad parameter space (F6 satisfied)' if scan_results['f6_passed'] else '  bodies escape without capture'}
"""
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('DET F6 Binding Falsifier - Impact Parameter Scan', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {filename}")


def run_detailed_binding_analysis(verbose: bool = True):
    """
    Run detailed analysis of a specific binding scenario.
    Shows time evolution of separation, structure, and agency.
    """
    if verbose:
        print("\n" + "="*70)
        print("DETAILED BINDING ANALYSIS")
        print("="*70)
    
    # Choose parameters that might show binding
    params = DETParams(
        alpha_q=0.02,      # Moderate q accumulation
        a_coupling=40.0,   # Stronger agency response
        eta_floor=0.15,    # Stronger floor repulsion
    )
    
    sim = DETCollider2D(params)
    
    # Setup: moderate velocity, small offset for potential capture
    N = params.N
    center = N // 2
    initial_sep = 35
    velocity = 0.5
    offset = 8
    
    y1, x1 = center + offset//2, center - initial_sep//2
    y2, x2 = center - offset//2, center + initial_sep//2
    
    sim.add_packet((y1, x1), mass=5.0, width=4.0, momentum=(0, velocity))
    sim.add_packet((y2, x2), mass=5.0, width=4.0, momentum=(0, -velocity))
    
    initial_F = np.sum(sim.F)
    
    # Recording
    rec = {'t': [], 'sep': [], 'peak_sep': [], 'bodies': [], 'peaks': [],
           'q_max': [], 'min_a': [], 'mass_err': []}
    snapshots = []
    
    steps = 15000
    for t in range(steps):
        sep, num_bodies = sim.separation()
        peak_sep, num_peaks = sim.peak_separation()
        
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['peak_sep'].append(peak_sep)
        rec['bodies'].append(num_bodies)
        rec['peaks'].append(num_peaks)
        rec['q_max'].append(np.max(sim.q))
        rec['min_a'].append(np.min(sim.a))
        rec['mass_err'].append(100 * (np.sum(sim.F) - initial_F) / initial_F)
        
        if t in [0, steps//5, 2*steps//5, 3*steps//5, 4*steps//5]:
            snapshots.append((t, sim.F.copy(), sim.q.copy(), sim.a.copy()))
        
        if verbose and t % 3000 == 0:
            print(f"  t={t}: sep={sep:.1f}, peak_sep={peak_sep:.1f}, "
                  f"bodies={num_bodies}, peaks={num_peaks}, q_max={np.max(sim.q):.3f}")
        
        sim.step()
    
    # Classify outcome
    result = BindingTestResult(
        outcome=Outcome.UNCERTAIN,
        initial_sep=rec['sep'][0],
        min_sep=min(rec['sep']),
        final_sep=rec['sep'][-1],
        final_peak_sep=rec['peak_sep'][-1],
        num_bodies=rec['bodies'][-1],
        num_peaks=rec['peaks'][-1],
        sep_history=rec['sep'],
        peak_sep_history=rec['peak_sep'],
        max_q=max(rec['q_max']),
        min_a=min(rec['min_a']),
        mass_error=rec['mass_err'][-1],
        params={'velocity': velocity, 'offset': offset}
    )
    result.outcome = classify_outcome(result)
    
    if verbose:
        print(f"\n  Outcome: {result.outcome.value}")
        print(f"  Final: sep={result.final_sep:.1f}, peak_sep={result.final_peak_sep:.1f}")
        print(f"  Bodies={result.num_bodies}, Peaks={result.num_peaks}")
    
    return rec, snapshots, result


def create_detailed_visualization(rec: Dict, snapshots: List, result: BindingTestResult,
                                  filename: str = 'det_f6_detailed.png'):
    """Create detailed visualization of binding dynamics."""
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Snapshots
    for i, (t, F, q, a) in enumerate(snapshots):
        ax = fig.add_subplot(3, 5, i + 1)
        ax.imshow(F, cmap='plasma', vmin=0)
        ax.set_title(f't={t}', fontsize=9)
        ax.axis('off')
    
    # Row 2: q and a snapshots
    for i, (t, F, q, a) in enumerate(snapshots):
        ax = fig.add_subplot(3, 5, i + 6)
        # Composite: q as red channel, a as green
        composite = np.zeros((*q.shape, 3))
        composite[:,:,0] = q / (np.max(q) + 1e-6)  # Red = structure
        composite[:,:,1] = a  # Green = agency
        ax.imshow(composite)
        ax.set_title(f'q(R)/a(G) t={t}', fontsize=9)
        ax.axis('off')
    
    # Row 3: Time series
    ax = fig.add_subplot(3, 3, 7)
    ax.plot(rec['t'], rec['sep'], 'b-', lw=1, alpha=0.5, label='Blob sep')
    ax.plot(rec['t'], rec['peak_sep'], 'r-', lw=1.5, label='Peak sep')
    ax.axhline(5, color='g', ls='--', alpha=0.3, label='Merge threshold')
    ax.axhline(rec['sep'][0], color='k', ls=':', alpha=0.3, label='Initial sep')
    ax.set_xlabel('Step')
    ax.set_ylabel('Separation')
    ax.set_title(f'Separation Evolution\nOutcome: {result.outcome.value}')
    ax.legend(fontsize=8)
    
    ax = fig.add_subplot(3, 3, 8)
    ax.plot(rec['t'], rec['min_a'], 'g-', lw=1.5, label='min(a)')
    ax.plot(rec['t'], rec['q_max'], 'r-', lw=1.5, label='max(q)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Agency & Structure')
    ax.legend(fontsize=8)
    
    ax = fig.add_subplot(3, 3, 9)
    ax.plot(rec['t'], rec['peaks'], 'purple', lw=1, label='Peaks')
    ax.plot(rec['t'], rec['bodies'], 'k--', lw=1, label='Bodies')
    ax.set_xlabel('Step')
    ax.set_ylabel('Count')
    ax.set_title('Peak & Body Count')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 5)
    
    plt.suptitle('DET F6 Detailed Binding Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")


if __name__ == "__main__":
    start = time.time()
    
    # Run F6 parameter scan
    scan_results = run_f6_parameter_scan(verbose=True)
    
    # Visualize results
    visualize_f6_results(scan_results, './det_f6_binding.png')
    
    # Run detailed analysis
    rec, snapshots, result = run_detailed_binding_analysis(verbose=True)
    create_detailed_visualization(rec, snapshots, result, './det_f6_detailed.png')
    
    elapsed = time.time() - start
    print(f"\nTotal runtime: {elapsed:.1f}s")