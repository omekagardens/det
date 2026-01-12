"""
DET v5 Momentum Collider
========================

Implements the proposed per-bond momentum extension (Section IV.4) and tests:
1. Momentum enables separated dipoles to collide
2. Momentum conservation (with λ_π = 0)
3. Collision dynamics and q-locking mass generation
4. Inertialess limit recovery (λ_π → ∞)

New state variable:
    π_{ij} ∈ ℝ  (directed momentum on bond i↔j, with π_{ij} = -π_{ji})

Momentum dynamics:
    π_{ij}^+ = π_{ij} + α_π J_{i→j} Δτ - λ_π π_{ij} Δτ

Momentum-driven flow:
    J^{(mom)}_{i→j} = μ_π σ_{ij} π_{ij}

Total flow:
    J = J^{(diff)} + J^{(grav)} + J^{(mom)} + J^{(floor)}
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class MomentumParams:
    """Parameters for momentum module (IV.4)"""
    alpha_pi: float = 0.05      # Momentum accumulation rate
    lambda_pi: float = 0.005    # Momentum decay rate (friction)
    mu_pi: float = 0.5          # Momentum-flow coupling strength
    pi_max: float = 2.0         # Stability bound

@dataclass 
class DETParams:
    """Full DET v5 parameters"""
    # Domain
    N: int = 400
    DT: float = 0.02
    
    # Resource
    F_VAC: float = 0.01
    
    # Coherence
    C_MIN: float = 0.05
    R: int = 10
    
    # Phase (extension)
    NU: float = 0.1
    
    # Coherence dynamics (extension)
    ALPHA_C: float = 0.05
    GAMMA_C: float = 2.0
    LAMBDA_C: float = 0.002
    
    # q-locking (canonical B.3)
    ALPHA_Q: float = 0.02
    
    # Momentum module (IV.4)
    momentum: MomentumParams = None
    
    # Dipole structure
    dipole_sep: float = 15.0
    width: float = 5.0
    amplitude: float = 6.0
    
    def __post_init__(self):
        if self.momentum is None:
            self.momentum = MomentumParams()


class DETStateV5:
    """State variables for DET v5 with momentum"""
    
    def __init__(self, N: int, C_MIN: float = 0.05, F_VAC: float = 0.01):
        self.N = N
        self.x = np.arange(N)
        
        # Per-node variables (II.1)
        self.F = np.ones(N) * F_VAC          # Resource
        self.theta = np.zeros(N)              # Phase
        self.q = np.zeros(N)                  # Structural debt
        
        # Per-bond variables (II.2)
        self.C_right = np.ones(N) * C_MIN     # Coherence i→i+1
        self.C_left = np.ones(N) * C_MIN      # Coherence i→i-1
        self.sigma = np.ones(N)               # Conductivity
        
        # NEW: Per-bond momentum (IV.4)
        # π_right[i] = momentum on bond i→i+1 (rightward positive)
        # Antisymmetry: π_{i,i+1} = -π_{i+1,i} is implicit
        self.pi_right = np.zeros(N)
        
    def add_gaussian(self, pos: float, amplitude: float, width: float,
                     C_init: float = 0.7):
        """Add Gaussian resource peak"""
        dx = self.x - pos
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        envelope = np.exp(-0.5 * (dx / width)**2)
        
        self.F += amplitude * envelope
        self.C_right += C_init * envelope
        self.C_left += C_init * envelope
        
    def add_dipole(self, center: float, params: DETParams, C_init: float = 0.7):
        """Add dipole (two peaks) at center"""
        for offset in [-params.dipole_sep/2, params.dipole_sep/2]:
            self.add_gaussian(center + offset, params.amplitude,
                            params.width, C_init)
    
    def add_momentum(self, center: float, momentum: float, width: float = 25.0):
        """Add momentum impulse centered at position"""
        dx = self.x - center
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        envelope = np.exp(-0.5 * (dx / width)**2)
        self.pi_right += momentum * envelope
        
    def clip_all(self, C_MIN: float, pi_max: float):
        """Ensure all variables stay in valid ranges"""
        self.C_right = np.clip(self.C_right, C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, C_MIN, 1.0)
        self.pi_right = np.clip(self.pi_right, -pi_max, pi_max)
        
    def get_separation(self) -> float:
        """Compute separation between left and right halves"""
        N = self.N
        left_F = self.F[:N//2]
        right_F = self.F[N//2:]
        left_com = np.sum(self.x[:N//2] * left_F) / (np.sum(left_F) + 1e-9)
        right_com = np.sum(self.x[N//2:] * right_F) / (np.sum(right_F) + 1e-9)
        return right_com - left_com
    
    def get_total_momentum(self) -> float:
        """Total momentum (weighted by local F for physical meaning)"""
        return np.sum(self.pi_right)
    
    def get_weighted_momentum(self) -> float:
        """F-weighted momentum"""
        return np.sum(self.pi_right * self.F)


def local_sum(x: np.ndarray, radius: int) -> np.ndarray:
    """Local sum for normalization"""
    kernel = np.ones(2 * radius + 1)
    return np.convolve(x, kernel, mode='same')


def count_peaks(F: np.ndarray, F_VAC: float = 0.01, threshold: float = 0.3) -> int:
    """Count resource peaks"""
    data = np.maximum(F - F_VAC, 0)
    data_max = maximum_filter(data, 7)
    maxima = (data == data_max) & (data > threshold)
    return np.sum(maxima)


def det_step_v5(state: DETStateV5, params: DETParams, 
                with_q_locking: bool = True,
                with_momentum: bool = True) -> Dict:
    """
    Single DET v5 update step with momentum.
    
    Flow decomposition (IV.2a):
        J = J^{(diff)} + J^{(mom)}
        
    (Gravity and floor not implemented in this test)
    """
    N = state.N
    dt = params.DT
    mom = params.momentum
    
    idx = np.arange(N)
    ip1 = (idx + 1) % N
    im1 = (idx - 1) % N
    
    # === STEP A: Wavefunction (IV.1) ===
    F_local_sum = local_sum(state.F, params.R) + 1e-9
    amp = np.sqrt(np.clip(state.F / F_local_sum, 0, 1))
    psi = amp * np.exp(1j * state.theta)
    
    # === STEP B: Phase evolution (extension) ===
    d_fwd = np.angle(np.exp(1j * (state.theta[ip1] - state.theta[idx])))
    d_bwd = np.angle(np.exp(1j * (state.theta[im1] - state.theta[idx])))
    state.theta = state.theta + params.NU * (d_fwd + d_bwd) * dt
    state.theta = np.mod(state.theta, 2*np.pi)
    
    # === STEP C: Diffusive Flow J^{(diff)} (IV.2) ===
    quantum_R = np.imag(np.conj(psi[idx]) * psi[ip1])
    quantum_L = np.imag(np.conj(psi[idx]) * psi[im1])
    classical_R = state.F[idx] - state.F[ip1]
    classical_L = state.F[idx] - state.F[im1]
    
    sqrt_C_R = np.sqrt(state.C_right)
    sqrt_C_L = np.sqrt(state.C_left)
    
    J_diff_R = state.sigma * (state.C_right + 1e-4) * (
        sqrt_C_R * quantum_R + (1 - sqrt_C_R) * classical_R
    )
    J_diff_L = state.sigma * (state.C_left + 1e-4) * (
        sqrt_C_L * quantum_L + (1 - sqrt_C_L) * classical_L
    )
    
    # === STEP D: Momentum-Driven Flow J^{(mom)} (IV.4) ===
    if with_momentum:
        J_mom_R = mom.mu_pi * state.sigma * state.pi_right
        # Leftward bond uses antisymmetric momentum from neighbor
        J_mom_L = -mom.mu_pi * state.sigma * np.roll(state.pi_right, 1)
    else:
        J_mom_R = np.zeros(N)
        J_mom_L = np.zeros(N)
    
    # === STEP E: Total Flow (IV.2a) ===
    J_right = J_diff_R + J_mom_R
    J_left = J_diff_L + J_mom_L
    
    # === STEP F: Conservative Limiting (Appendix N) ===
    max_allowed = 0.25 * state.F / dt
    total_out = np.abs(J_right) + np.abs(J_left)
    scale = np.minimum(1.0, max_allowed / (total_out + 1e-9))
    J_right = np.clip(J_right * scale, -5, 5)
    J_left = np.clip(J_left * scale, -5, 5)
    
    # === STEP G: Resource Update (IV.3) ===
    dF = (J_right[im1] + J_left[ip1]) - (J_right + J_left)
    F_new = np.clip(state.F + dF * dt, params.F_VAC, 100)
    
    # === STEP H: q-locking (B.3 Step 6) ===
    if with_q_locking:
        delta_F = F_new - state.F
        state.q = np.clip(state.q + params.ALPHA_Q * np.maximum(0, -delta_F), 0, 1)
    
    state.F = F_new
    
    # === STEP I: Momentum Update (IV.4) ===
    if with_momentum:
        # π accumulates from net flow, decays with friction
        state.pi_right = (state.pi_right 
                         + mom.alpha_pi * J_right * dt 
                         - mom.lambda_pi * state.pi_right * dt)
        state.pi_right = np.clip(state.pi_right, -mom.pi_max, mom.pi_max)
    
    # === STEP J: Coherence Update (extension) ===
    compression = np.maximum(0, dF)
    alpha = params.ALPHA_C * (1.0 + 20.0 * compression)
    
    J_R_align = np.maximum(0, J_right) + params.GAMMA_C * np.maximum(0, np.roll(J_right, 1))
    J_L_align = np.maximum(0, J_left) + params.GAMMA_C * np.maximum(0, np.roll(J_left, -1))
    
    state.C_right += (alpha * J_R_align - params.LAMBDA_C * state.C_right) * dt
    state.C_left += (alpha * J_L_align - params.LAMBDA_C * state.C_left) * dt
    
    state.clip_all(params.C_MIN, mom.pi_max)
    
    state.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_right) + np.abs(J_left))
    
    return {
        'J_right': J_right,
        'J_left': J_left,
        'J_diff_R': J_diff_R,
        'J_mom_R': J_mom_R,
        'dF': dF
    }


def run_collision_v5(initial_separation: float = 100,
                     initial_momentum: float = 0.3,
                     params: Optional[DETParams] = None,
                     steps: int = 30000,
                     with_momentum: bool = True,
                     with_q_locking: bool = True,
                     snapshot_times: Optional[List[float]] = None) -> Dict:
    """
    Run DET v5 collision with momentum.
    """
    if params is None:
        params = DETParams()
    
    if snapshot_times is None:
        snapshot_times = [0, 50, 100, 200, 400, 600, 800, 1000, 1500]
    
    # Initialize state
    state = DETStateV5(params.N, params.C_MIN, params.F_VAC)
    
    # Create two dipoles
    center = params.N // 2
    left_center = center - initial_separation / 2
    right_center = center + initial_separation / 2
    
    state.add_dipole(left_center, params)
    state.add_dipole(right_center, params)
    
    # Add initial momentum (toward each other)
    if with_momentum and initial_momentum > 0:
        state.add_momentum(left_center, initial_momentum)      # Left moves right
        state.add_momentum(right_center, -initial_momentum)    # Right moves left
    
    state.clip_all(params.C_MIN, params.momentum.pi_max)
    
    # Tracking
    traces = {
        'peaks': [],
        'separation': [],
        'total_pi': [],
        'weighted_pi': [],
        'total_F': [],
        'q_max': [],
        'q_sum': []
    }
    snapshots = []
    
    for step in range(steps):
        t = step * params.DT
        
        # Track
        traces['peaks'].append(count_peaks(state.F, params.F_VAC))
        traces['separation'].append(state.get_separation())
        traces['total_pi'].append(state.get_total_momentum())
        traces['weighted_pi'].append(state.get_weighted_momentum())
        traces['total_F'].append(np.sum(state.F))
        traces['q_max'].append(np.max(state.q))
        traces['q_sum'].append(np.sum(state.q))
        
        # Snapshots
        if any(abs(t - st) < params.DT/2 for st in snapshot_times):
            snapshots.append({
                't': t,
                'F': state.F.copy(),
                'q': state.q.copy(),
                'pi': state.pi_right.copy(),
                'peaks': traces['peaks'][-1],
                'sep': traces['separation'][-1]
            })
        
        # Step
        det_step_v5(state, params, with_q_locking, with_momentum)
    
    # Final snapshot
    snapshots.append({
        't': step * params.DT,
        'F': state.F.copy(),
        'q': state.q.copy(),
        'pi': state.pi_right.copy(),
        'peaks': traces['peaks'][-1],
        'sep': traces['separation'][-1]
    })
    
    # Convert traces to arrays
    for key in traces:
        traces[key] = np.array(traces[key])
    
    return {
        'traces': traces,
        'snapshots': snapshots,
        'params': params,
        'initial_separation': initial_separation,
        'initial_momentum': initial_momentum,
        'with_momentum': with_momentum,
        'with_q_locking': with_q_locking
    }


if __name__ == "__main__":
    print("=" * 70)
    print("DET v5 MOMENTUM COLLIDER TEST")
    print("=" * 70)
    print("\nTesting per-bond momentum (Section IV.4 proposal)")
    print("=" * 70)
    
    # Run tests
    results = {}
    
    # Test 1: No momentum (baseline)
    print("\n[1] No momentum (baseline)...")
    results['no_mom'] = run_collision_v5(
        initial_momentum=0.0, with_momentum=False, steps=25000
    )
    r = results['no_mom']
    print(f"    Initial sep: {r['traces']['separation'][0]:.1f}")
    print(f"    Final sep: {r['traces']['separation'][-1]:.1f}")
    print(f"    Min sep: {np.min(r['traces']['separation']):.1f}")
    
    # Test 2: With momentum p=0.3
    print("\n[2] With momentum p=0.3...")
    results['p03'] = run_collision_v5(
        initial_momentum=0.3, with_momentum=True, steps=25000
    )
    r = results['p03']
    print(f"    Initial sep: {r['traces']['separation'][0]:.1f}")
    print(f"    Final sep: {r['traces']['separation'][-1]:.1f}")
    print(f"    Min sep: {np.min(r['traces']['separation']):.1f}")
    
    # Test 3: With momentum p=0.5
    print("\n[3] With momentum p=0.5...")
    results['p05'] = run_collision_v5(
        initial_momentum=0.5, with_momentum=True, steps=25000
    )
    r = results['p05']
    print(f"    Initial sep: {r['traces']['separation'][0]:.1f}")
    print(f"    Final sep: {r['traces']['separation'][-1]:.1f}")
    print(f"    Min sep: {np.min(r['traces']['separation']):.1f}")
    collision = np.min(r['traces']['separation']) < 50
    print(f"    COLLISION: {'YES!' if collision else 'No'}")
    
    # Test 4: Momentum with q-locking
    print("\n[4] Momentum p=0.5 with q-locking...")
    results['p05_q'] = run_collision_v5(
        initial_momentum=0.5, with_momentum=True, 
        with_q_locking=True, steps=25000
    )
    r = results['p05_q']
    print(f"    Initial sep: {r['traces']['separation'][0]:.1f}")
    print(f"    Min sep: {np.min(r['traces']['separation']):.1f}")
    print(f"    Final q_max: {r['traces']['q_max'][-1]:.4f}")
    print(f"    Final q_sum: {r['traces']['q_sum'][-1]:.2f}")
    
    # Test 5: Momentum conservation (λ_π = 0)
    print("\n[5] Momentum conservation test (λ_π = 0)...")
    params_conserve = DETParams()
    params_conserve.momentum.lambda_pi = 0.0
    results['conserve'] = run_collision_v5(
        initial_momentum=0.3, params=params_conserve,
        with_momentum=True, steps=15000
    )
    r = results['conserve']
    pi_init = r['traces']['total_pi'][0]
    pi_final = r['traces']['total_pi'][-1]
    pi_drift = abs(pi_final - pi_init) / (abs(pi_init) + 1e-9)
    print(f"    Initial Π: {pi_init:.4f}")
    print(f"    Final Π: {pi_final:.4f}")
    print(f"    Drift: {pi_drift*100:.2f}%")
    print(f"    CONSERVED: {'YES' if pi_drift < 0.1 else 'NO'}")
    
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION...")
    print("=" * 70)
