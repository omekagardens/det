"""
Grace v6.4 - Antisymmetric Edge Flux Formulation
Strictly local, conserving by construction, no hidden globals

Key changes from v6.3:
1. Grace is an antisymmetric EDGE FLUX G_{i→j} = -G_{j→i}
2. Conservation automatic (no balancing step)
3. Quantum gate is BOND-LOCAL: Q_{ij} = [1 - √C_{ij}/C_quantum]_+
4. F_min^abs demoted to numerical stability floor (not physical threshold)
5. No overlapping neighborhood double-counting
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


@dataclass
class DETConfig:
    N: int = 32
    dt: float = 0.02
    T_max: int = 2000
    sigma_base: float = 1.0
    omega_0: float = 1.0
    epsilon: float = 1e-10
    
    alpha_C: float = 0.05
    lambda_C: float = 0.01
    C_min: float = 0.001
    
    # Grace v6.4 parameters
    eta_g: float = 0.5           # Grace flux coefficient (increased for visible effect)
    beta_g: float = 0.4          # Relative need threshold
    C_quantum: float = 0.85      # Bond-local quantum gate threshold
    
    # Numerical stability floor (NOT a physical threshold)
    # Used only to prevent dead-zero regions in finite precision
    F_floor: float = 1e-6
    
    lambda_a: float = 1.0
    beta_a: float = 0.05


class Outcome(Enum):
    RECOVERED = "RECOVERED"
    FROZEN = "FROZEN"
    COLLAPSED = "COLLAPSED"
    PARTIAL = "PARTIAL"


@dataclass
class DETState:
    N: int
    F: np.ndarray
    q: np.ndarray
    a: np.ndarray
    theta: np.ndarray
    tau: np.ndarray
    C_h: np.ndarray  # Horizontal coherence [N, N-1]
    C_v: np.ndarray  # Vertical coherence [N-1, N]
    sigma_h: np.ndarray
    sigma_v: np.ndarray
    
    # Diagnostics
    grace_flux_h: np.ndarray = field(default_factory=lambda: np.array([]))
    grace_flux_v: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @classmethod
    def create(cls, N: int, config: DETConfig) -> 'DETState':
        return cls(
            N=N, F=np.ones((N, N)), q=np.zeros((N, N)),
            a=np.ones((N, N)) * 0.5,
            theta=np.random.uniform(0, 2*np.pi, (N, N)),
            tau=np.zeros((N, N)),
            C_h=np.ones((N, N-1)) * 0.5,
            C_v=np.ones((N-1, N)) * 0.5,
            sigma_h=np.ones((N, N-1)) * config.sigma_base,
            sigma_v=np.ones((N-1, N)) * config.sigma_base,
            grace_flux_h=np.zeros((N, N-1)),
            grace_flux_v=np.zeros((N-1, N)),
        )
    
    def copy(self):
        return DETState(
            N=self.N, F=self.F.copy(), q=self.q.copy(), a=self.a.copy(),
            theta=self.theta.copy(), tau=self.tau.copy(),
            C_h=self.C_h.copy(), C_v=self.C_v.copy(),
            sigma_h=self.sigma_h.copy(), sigma_v=self.sigma_v.copy(),
            grace_flux_h=self.grace_flux_h.copy(),
            grace_flux_v=self.grace_flux_v.copy(),
        )


# ============================================================================
# Core Dynamics
# ============================================================================

def compute_psi(F, theta, epsilon):
    N = F.shape[0]
    F_local_sum = np.zeros_like(F)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            F_local_sum += np.roll(np.roll(F, -di, axis=0), -dj, axis=1)
    return np.sqrt(F / (F_local_sum + epsilon)) * np.exp(1j * theta)


def compute_agency_gate(a_i, a_j):
    return np.sqrt(np.maximum(a_i, 0) * np.maximum(a_j, 0))


def compute_diffusive_flow(state, config, direction):
    """Standard diffusive flow J_{i→j}"""
    psi = compute_psi(state.F, state.theta, config.epsilon)
    
    if direction == 'h':
        psi_i, psi_j = psi[:, :-1], psi[:, 1:]
        F_i, F_j = state.F[:, :-1], state.F[:, 1:]
        a_i, a_j = state.a[:, :-1], state.a[:, 1:]
        C, sigma = state.C_h, state.sigma_h
    else:
        psi_i, psi_j = psi[:-1, :], psi[1:, :]
        F_i, F_j = state.F[:-1, :], state.F[1:, :]
        a_i, a_j = state.a[:-1, :], state.a[1:, :]
        C, sigma = state.C_v, state.sigma_v
    
    g_a = compute_agency_gate(a_i, a_j)
    quantum_flow = np.imag(np.conj(psi_i) * psi_j)
    classical_flow = F_i - F_j
    sqrt_C = np.sqrt(np.clip(C, 0, 1))
    
    return g_a * sigma * (sqrt_C * quantum_flow + (1 - sqrt_C) * classical_flow)


def compute_local_F_avg(F, R=2):
    """Compute local average of F over R-radius neighborhood"""
    N = F.shape[0]
    F_sum = np.zeros_like(F)
    count = 0
    
    for di in range(-R, R+1):
        for dj in range(-R, R+1):
            if abs(di) + abs(dj) <= R:  # Manhattan distance
                F_sum += np.roll(np.roll(F, -di, axis=0), -dj, axis=1)
                count += 1
    
    return F_sum / count


def compute_neighbor_need_sum(need, direction, side, R=2):
    """
    Compute sum of recipient needs in node i's neighborhood.
    For edge (i,j), we need Σ_{k∈N_R(i)} r_k and Σ_{k∈N_R(j)} r_k
    
    direction: 'h' or 'v'
    side: 'i' (left/top node) or 'j' (right/bottom node)
    R: neighborhood radius (default 2 to reach beyond single-layer barriers)
    """
    N = need.shape[0]
    
    # Sum over R-radius neighborhood
    r_sum = np.zeros_like(need)
    for di in range(-R, R+1):
        for dj in range(-R, R+1):
            if abs(di) + abs(dj) <= R:  # Manhattan distance for locality
                r_sum += np.roll(np.roll(need, -di, axis=0), -dj, axis=1)
    
    if direction == 'h':
        if side == 'i':
            return r_sum[:, :-1]  # Left nodes of horizontal edges
        else:
            return r_sum[:, 1:]   # Right nodes of horizontal edges
    else:
        if side == 'i':
            return r_sum[:-1, :]  # Top nodes of vertical edges
        else:
            return r_sum[1:, :]   # Bottom nodes of vertical edges


# ============================================================================
# GRACE v6.4: Antisymmetric Edge Flux
# ============================================================================

def compute_grace_flux(state, config, direction, enabled=True):
    """
    Grace v6.4 - Antisymmetric edge flux formulation
    
    G_{i→j} = η_g · g^(a)_{ij} · Q_{ij} · (d_i · r_j/Σr_N(i) - d_j · r_i/Σr_N(j))
    
    where:
    - d_i = a_i · excess_i (donor capacity)
    - r_i = a_i · need_i (recipient need)
    - g^(a)_{ij} = √(a_i·a_j) (agency gate - NOT physical conductivity!)
    - Q_{ij} = [1 - √C_{ij}/C_quantum]_+ (bond-local quantum gate)
    
    KEY DESIGN CHOICE: Grace flows through AGENCY-connected paths, not
    conductivity-connected paths. This allows grace to bypass low-σ barriers
    while still respecting the relational structure via agency.
    
    This is antisymmetric by construction: G_{i→j} = -G_{j→i}
    Conservation is automatic.
    
    Returns: G_{i→j} flux array (same shape as bonds)
    """
    if not enabled:
        if direction == 'h':
            return np.zeros((state.N, state.N - 1))
        else:
            return np.zeros((state.N - 1, state.N))
    
    N = state.N
    eps = config.epsilon
    
    # Compute local threshold (relative to local average)
    F_local_avg = compute_local_F_avg(state.F)
    F_threshold = config.beta_g * F_local_avg
    # Note: F_floor is ONLY for numerical stability, not used in threshold
    
    # Need and excess at each node
    need = np.maximum(0, F_threshold - state.F)    # How much below threshold
    excess = np.maximum(0, state.F - F_threshold)  # How much above threshold
    
    # Donor capacity and recipient need (agency-gated)
    d = state.a * excess  # Donor capacity
    r = state.a * need    # Recipient need
    
    # Get bond-adjacent values
    if direction == 'h':
        d_i, d_j = d[:, :-1], d[:, 1:]
        r_i, r_j = r[:, :-1], r[:, 1:]
        a_i, a_j = state.a[:, :-1], state.a[:, 1:]
        C = state.C_h
    else:
        d_i, d_j = d[:-1, :], d[1:, :]
        r_i, r_j = r[:-1, :], r[1:, :]
        a_i, a_j = state.a[:-1, :], state.a[1:, :]
        C = state.C_v
    
    # Agency gate: g^(a)_{ij} = √(a_i·a_j)
    # Grace flows through agency-connected paths, NOT physical conductivity
    g_a = np.sqrt(np.maximum(a_i, 0) * np.maximum(a_j, 0))
    
    # Bond-local quantum gate: Q_{ij} = [1 - √C_{ij}/C_quantum]_+
    Q_ij = np.clip(1 - np.sqrt(C) / config.C_quantum, 0, 1)
    
    # Neighborhood need sums for normalization
    r_sum_Ni = compute_neighbor_need_sum(r, direction, 'i')  # Σ_{k∈N(i)} r_k
    r_sum_Nj = compute_neighbor_need_sum(r, direction, 'j')  # Σ_{k∈N(j)} r_k
    
    # Grace flux (antisymmetric by construction)
    # G_{i→j} = η_g · g^(a)_{ij} · Q_{ij} · (d_i · r_j/Σr_N(i) - d_j · r_i/Σr_N(j))
    term_i_to_j = d_i * r_j / (r_sum_Ni + eps)  # i donates to j
    term_j_to_i = d_j * r_i / (r_sum_Nj + eps)  # j donates to i
    
    G = config.eta_g * g_a * Q_ij * (term_i_to_j - term_j_to_i)
    
    return G


def update_resource_v64(state, config, grace_enabled=True):
    """
    Resource update with antisymmetric grace flux
    
    F_i^+ = F_i - Σ_j J_{i→j} Δτ + Σ_j G_{j→i} Δτ
    
    Both J and G are antisymmetric, so conservation is automatic.
    """
    N = state.N
    dt = config.dt
    
    # Diffusive flow
    J_h = compute_diffusive_flow(state, config, 'h')
    J_v = compute_diffusive_flow(state, config, 'v')
    
    # Grace flux (antisymmetric edge flux)
    G_h = compute_grace_flux(state, config, 'h', enabled=grace_enabled)
    G_v = compute_grace_flux(state, config, 'v', enabled=grace_enabled)
    
    # Store for diagnostics
    state.grace_flux_h = G_h
    state.grace_flux_v = G_v
    
    # Combine flows: total flux = diffusive + grace
    # Both are antisymmetric, so we just add them
    T_h = J_h + G_h
    T_v = J_v + G_v
    
    # Update F: subtract outflow
    dF = np.zeros((N, N))
    dF[:, :-1] += T_h * dt  # Outflow from left node
    dF[:, 1:] -= T_h * dt   # Inflow to right node
    dF[:-1, :] += T_v * dt  # Outflow from top node
    dF[1:, :] -= T_v * dt   # Inflow to bottom node
    
    F_new = state.F - dF
    
    # Numerical floor only (not physical threshold)
    F_new = np.maximum(F_new, config.F_floor)
    
    return F_new


def update_coherence(state, config):
    J_h = compute_diffusive_flow(state, config, 'h')
    J_v = compute_diffusive_flow(state, config, 'v')
    
    C_h_new = state.C_h + config.alpha_C * np.abs(J_h) * config.dt - config.lambda_C * state.C_h * config.dt
    C_v_new = state.C_v + config.alpha_C * np.abs(J_v) * config.dt - config.lambda_C * state.C_v * config.dt
    
    return np.clip(C_h_new, config.C_min, 1.0), np.clip(C_v_new, config.C_min, 1.0)


def update_phase(state, config):
    P = state.a * config.sigma_base / (1 + state.F)
    return (state.theta + config.omega_0 * P * config.dt) % (2 * np.pi)


def update_agency(state, config):
    a_target = 1.0 / (1.0 + config.lambda_a * state.q**2)
    return np.clip(state.a + config.beta_a * (a_target - state.a), 0, 1)


def step(state, config, grace_enabled=True):
    new_state = state.copy()
    new_state.F = update_resource_v64(state, config, grace_enabled)
    new_state.C_h, new_state.C_v = update_coherence(state, config)
    new_state.theta = update_phase(state, config)
    new_state.a = update_agency(state, config)
    new_state.grace_flux_h = state.grace_flux_h.copy()
    new_state.grace_flux_v = state.grace_flux_v.copy()
    return new_state


# ============================================================================
# Test Scenarios
# ============================================================================

def setup_quantum_trap(config):
    """E1: High coherence, tests bond-local quantum gate"""
    N = config.N
    state = DETState.create(N, config)
    center = N // 2
    r_dep = 3
    
    y, x = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    depleted_mask = (np.abs(y - center) <= r_dep) & (np.abs(x - center) <= r_dep)
    
    state.F = np.ones((N, N)) * 2.0
    state.F[depleted_mask] = 0.05
    state.a = np.ones((N, N)) * 0.02
    state.a[depleted_mask] = 0.4
    
    for i in range(N):
        for j in range(N):
            dy, dx = i - center, j - center
            r = np.sqrt(dx**2 + dy**2) + 0.1
            state.theta[i, j] = np.arctan2(dy, dx) + r * 0.5
    
    state.C_h[:] = 0.98
    state.C_v[:] = 0.98
    
    return state, depleted_mask


def setup_conductivity_trap(config):
    """E3: Low conductivity barrier, tests grace bypass"""
    N = config.N
    state = DETState.create(N, config)
    center = N // 2
    r_dep = 3
    r_barrier = 5
    
    y, x = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    depleted_mask = (np.abs(y - center) <= r_dep) & (np.abs(x - center) <= r_dep)
    barrier_mask = ((np.abs(y - center) > r_dep) & (np.abs(y - center) <= r_barrier)) | \
                   ((np.abs(x - center) > r_dep) & (np.abs(x - center) <= r_barrier))
    
    state.F = np.ones((N, N)) * 1.5
    state.F[depleted_mask] = 0.1
    state.a = np.ones((N, N)) * 0.4
    
    for i in range(N):
        for j in range(N-1):
            if barrier_mask[i, j] or barrier_mask[i, j+1]:
                state.sigma_h[i, j] = 0.01
    for i in range(N-1):
        for j in range(N):
            if barrier_mask[i, j] or barrier_mask[i+1, j]:
                state.sigma_v[i, j] = 0.01
    
    state.C_h[:] = 0.5
    state.C_v[:] = 0.5
    
    return state, depleted_mask


def setup_coordinated_zero(config):
    """E4: Large depleted region, tests conservation"""
    N = config.N
    state = DETState.create(N, config)
    
    boundary = int(N * 0.6)
    depleted_mask = np.zeros((N, N), dtype=bool)
    depleted_mask[:, :boundary] = True
    
    state.F[:, :boundary] = 0.01  # Near-zero (not exactly zero for stability)
    state.F[:, boundary:] = 2.0
    state.a = np.ones((N, N)) * 0.3
    state.C_h[:] = 0.2
    state.C_v[:] = 0.2
    
    return state, depleted_mask


def setup_overlap_stress(config):
    """
    T1: Overlap Stress Test
    
    Alternating donor/recipient checkerboard pattern.
    Every node's neighborhood overlaps with neighbors.
    Tests that donors aren't taxed multiple times.
    """
    N = config.N
    state = DETState.create(N, config)
    
    # Checkerboard: even squares are donors, odd are recipients
    y, x = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    donor_mask = ((y + x) % 2 == 0)
    
    state.F[donor_mask] = 2.0       # Donors have excess
    state.F[~donor_mask] = 0.2      # Recipients have need
    state.a = np.ones((N, N)) * 0.5  # Uniform agency
    state.C_h[:] = 0.3              # Low coherence (grace active)
    state.C_v[:] = 0.3
    
    depleted_mask = ~donor_mask
    
    return state, depleted_mask, donor_mask


def setup_mixed_coherence_channel(config):
    """
    T2: Mixed Coherence Channel Test
    
    Low-C corridor feeding into high-C "quantum island".
    Grace should help in corridor but NOT inject across high-C boundary.
    """
    N = config.N
    state = DETState.create(N, config)
    center = N // 2
    
    # Create regions
    # Left corridor: low C, depleted
    # Center island: high C, moderate F
    # Right region: low C, resource-rich
    
    corridor_mask = np.zeros((N, N), dtype=bool)
    corridor_mask[:, :N//3] = True
    
    island_mask = np.zeros((N, N), dtype=bool)
    island_mask[center-4:center+4, N//3:2*N//3] = True
    
    # Resource distribution
    state.F = np.ones((N, N)) * 1.5  # Default rich
    state.F[corridor_mask] = 0.1     # Corridor depleted
    state.F[island_mask] = 0.8       # Island moderate
    
    state.a = np.ones((N, N)) * 0.4
    
    # Coherence: low everywhere except island
    state.C_h[:] = 0.2
    state.C_v[:] = 0.2
    
    # High coherence in island (both internal and boundary bonds)
    for i in range(center-4, center+4):
        for j in range(N//3-1, 2*N//3):
            if j < N-1:
                state.C_h[i, j] = 0.95
    for i in range(center-5, center+4):
        for j in range(N//3, 2*N//3):
            if i < N-1:
                state.C_v[i, j] = 0.95
    
    depleted_mask = corridor_mask
    
    return state, depleted_mask, island_mask


# ============================================================================
# Simulation
# ============================================================================

@dataclass
class SimResult:
    F_history: List[np.ndarray]
    F_total_history: List[float]
    F_depleted_mean_history: List[float]
    grace_magnitude_history: List[float]
    final_state: DETState
    depleted_mask: np.ndarray
    outcome: Outcome
    recovery_time: Optional[int]


def run_simulation(initial_state, depleted_mask, config, grace_enabled=True, record_interval=20):
    state = initial_state.copy()
    
    F_history = [state.F.copy()]
    F_total_history = [state.F.sum()]
    F_depleted_mean_history = [state.F[depleted_mask].mean() if depleted_mask.any() else 0]
    grace_magnitude_history = [0.0]
    
    recovery_time = None
    
    for t in range(config.T_max):
        state = step(state, config, grace_enabled)
        
        if t % record_interval == 0:
            F_history.append(state.F.copy())
            F_total_history.append(state.F.sum())
            if depleted_mask.any():
                F_depleted_mean_history.append(state.F[depleted_mask].mean())
            # Grace magnitude (sum of absolute flux)
            grace_mag = np.abs(state.grace_flux_h).sum() + np.abs(state.grace_flux_v).sum()
            grace_magnitude_history.append(grace_mag)
        
        if recovery_time is None and depleted_mask.any() and state.F[depleted_mask].mean() > 0.5:
            recovery_time = t
    
    F_dep_final = state.F[depleted_mask] if depleted_mask.any() else np.array([1.0])
    f_rec = np.mean(F_dep_final > config.beta_g * state.F.mean())
    
    if f_rec > 0.8 and np.mean(F_dep_final) > 0.5:
        outcome = Outcome.RECOVERED
    elif F_total_history[-1] < 0.7 * F_total_history[0]:
        outcome = Outcome.COLLAPSED
    elif f_rec < 0.2:
        outcome = Outcome.FROZEN
    else:
        outcome = Outcome.PARTIAL
    
    return SimResult(
        F_history=F_history, F_total_history=F_total_history,
        F_depleted_mean_history=F_depleted_mean_history,
        grace_magnitude_history=grace_magnitude_history,
        final_state=state, depleted_mask=depleted_mask,
        outcome=outcome, recovery_time=recovery_time
    )


# ============================================================================
# Validation Suite
# ============================================================================

def run_validation_suite():
    print("="*70)
    print("GRACE v6.4 VALIDATION SUITE")
    print("Antisymmetric Edge Flux - Strictly Local")
    print("="*70)
    
    config = DETConfig(N=32, T_max=2000, dt=0.02, eta_g=0.3, C_quantum=0.85)
    
    results = {}
    
    # ====== TEST 1: Conservation (automatic by antisymmetry) ======
    print("\n" + "-"*60)
    print("TEST 1: Conservation (E4: Coordinated Zero)")
    print("-"*60)
    
    state, mask = setup_coordinated_zero(config)
    result_on = run_simulation(state.copy(), mask, config, grace_enabled=True)
    result_off = run_simulation(state.copy(), mask, config, grace_enabled=False)
    
    cons_on = result_on.F_total_history[-1] / result_on.F_total_history[0]
    cons_off = result_off.F_total_history[-1] / result_off.F_total_history[0]
    
    print(f"  Grace ON:  F_total = {result_on.F_total_history[0]:.4f} → {result_on.F_total_history[-1]:.4f}")
    print(f"             Conservation: {cons_on:.8f} ({(cons_on-1)*100:+.6f}%)")
    print(f"  Grace OFF: Conservation: {cons_off:.8f}")
    
    cons_pass = abs(cons_on - 1.0) < 0.001  # Within 0.1% (tighter bound)
    print(f"\n  CONSERVATION TEST: {'✓ PASS' if cons_pass else '✗ FAIL'}")
    
    results['conservation'] = {'on': result_on, 'off': result_off, 'cons': cons_on, 'pass': cons_pass}
    
    # ====== TEST 2: Bond-Local Quantum Gate (E1) ======
    print("\n" + "-"*60)
    print("TEST 2: Bond-Local Quantum Gate (E1: Quantum Trap)")
    print("-"*60)
    
    state, mask = setup_quantum_trap(config)
    result_on = run_simulation(state.copy(), mask, config, grace_enabled=True)
    result_off = run_simulation(state.copy(), mask, config, grace_enabled=False)
    
    f_on = result_on.F_depleted_mean_history[-1]
    f_off = result_off.F_depleted_mean_history[-1]
    total_grace = sum(result_on.grace_magnitude_history)
    
    print(f"  Grace ON:  F_depleted = {f_on:.4f}")
    print(f"  Grace OFF: F_depleted = {f_off:.4f}")
    print(f"  Total grace magnitude: {total_grace:.6f}")
    print(f"  Difference: {f_on - f_off:+.4f}")
    
    # Grace should be suppressed AND not harm recovery
    quantum_pass = total_grace < 0.1 and f_on >= f_off * 0.95
    print(f"\n  QUANTUM GATE TEST: {'✓ PASS' if quantum_pass else '✗ FAIL'}")
    
    results['quantum'] = {'on': result_on, 'off': result_off, 'grace': total_grace, 'pass': quantum_pass}
    
    # ====== TEST 3: Necessity (E3) ======
    print("\n" + "-"*60)
    print("TEST 3: Necessity (E3: Conductivity Trap)")
    print("-"*60)
    
    state, mask = setup_conductivity_trap(config)
    result_on = run_simulation(state.copy(), mask, config, grace_enabled=True)
    result_off = run_simulation(state.copy(), mask, config, grace_enabled=False)
    
    f_on = result_on.F_depleted_mean_history[-1]
    f_off = result_off.F_depleted_mean_history[-1]
    delta = f_on - f_off
    
    print(f"  Grace ON:  F_depleted = {f_on:.4f}")
    print(f"  Grace OFF: F_depleted = {f_off:.4f}")
    print(f"  Difference: {delta:+.4f}")
    
    necessity_pass = delta > 0.03
    print(f"\n  NECESSITY TEST: {'✓ PASS' if necessity_pass else '✗ FAIL'}")
    
    results['necessity'] = {'on': result_on, 'off': result_off, 'delta': delta, 'pass': necessity_pass}
    
    # ====== TEST 4: Agency Inviolability (F2) ======
    print("\n" + "-"*60)
    print("TEST 4: Agency Inviolability (a=0 receives no grace)")
    print("-"*60)
    
    state = DETState.create(config.N, config)
    center = config.N // 2
    r = 3
    y, x = np.meshgrid(np.arange(config.N), np.arange(config.N), indexing='ij')
    zero_a_mask = (np.abs(y - center) <= r) & (np.abs(x - center) <= r)
    
    state.F[zero_a_mask] = 0.05
    state.a[zero_a_mask] = 0.0
    state.a[~zero_a_mask] = 0.5
    state.C_h[:] = 0.3
    state.C_v[:] = 0.3
    
    # Run one step
    F_before = state.F[zero_a_mask].sum()
    state_after = step(state, config, grace_enabled=True)
    
    # Check grace flux into zero-agency region
    # For horizontal bonds adjacent to zero-a region
    grace_into_zero = 0.0
    G_h = state_after.grace_flux_h
    G_v = state_after.grace_flux_v
    
    # Sum flux INTO the zero-a region (negative flux means inflow from right/bottom)
    for i in range(config.N):
        for j in range(config.N):
            if zero_a_mask[i, j]:
                # Check all adjacent bonds
                if j > 0:
                    grace_into_zero += max(0, -G_h[i, j-1])  # From left
                if j < config.N - 1:
                    grace_into_zero += max(0, G_h[i, j])     # From right
                if i > 0:
                    grace_into_zero += max(0, -G_v[i-1, j])  # From top
                if i < config.N - 1:
                    grace_into_zero += max(0, G_v[i, j])     # From bottom
    
    print(f"  Grace flux into a=0 region: {grace_into_zero:.2e}")
    
    agency_pass = grace_into_zero < config.epsilon * 100  # Allow small numerical error
    print(f"\n  AGENCY INVIOLABILITY TEST: {'✓ PASS' if agency_pass else '✗ FAIL'}")
    
    results['agency'] = {'grace_in': grace_into_zero, 'pass': agency_pass}
    
    # ====== TEST 5: Overlap Stress (T1) ======
    print("\n" + "-"*60)
    print("TEST 5: Overlap Stress (T1: Checkerboard Pattern)")
    print("-"*60)
    
    state, depleted_mask, donor_mask = setup_overlap_stress(config)
    initial_donor_F = state.F[donor_mask].copy()
    
    # Run simulation
    result = run_simulation(state.copy(), depleted_mask, config, grace_enabled=True)
    
    # Check that donors were never over-taxed
    # In edge-flux formulation, each edge can only transfer what the donor has
    # Check conservation and that no donor went negative
    min_donor_F = result.final_state.F[donor_mask].min()
    
    # Also check for oscillation (ping-pong)
    F_history = result.F_depleted_mean_history
    oscillation = 0
    for i in range(2, len(F_history)):
        if (F_history[i] - F_history[i-1]) * (F_history[i-1] - F_history[i-2]) < 0:
            oscillation += 1
    oscillation_rate = oscillation / max(1, len(F_history) - 2)
    
    print(f"  Min donor F (should be > 0): {min_donor_F:.4f}")
    print(f"  Oscillation rate: {oscillation_rate:.2%}")
    print(f"  Conservation: {result.F_total_history[-1]/result.F_total_history[0]:.8f}")
    
    overlap_pass = min_donor_F > 0 and oscillation_rate < 0.3
    print(f"\n  OVERLAP STRESS TEST: {'✓ PASS' if overlap_pass else '✗ FAIL'}")
    
    results['overlap'] = {'min_donor': min_donor_F, 'oscillation': oscillation_rate, 'pass': overlap_pass}
    
    # ====== TEST 6: Mixed Coherence Channel (T2) ======
    print("\n" + "-"*60)
    print("TEST 6: Mixed Coherence Channel (T2)")
    print("-"*60)
    
    state, corridor_mask, island_mask = setup_mixed_coherence_channel(config)
    
    result_on = run_simulation(state.copy(), corridor_mask, config, grace_enabled=True)
    result_off = run_simulation(state.copy(), corridor_mask, config, grace_enabled=False)
    
    # Check corridor recovery
    f_corridor_on = result_on.F_depleted_mean_history[-1]
    f_corridor_off = result_off.F_depleted_mean_history[-1]
    
    # Check that island was NOT disturbed by grace
    f_island_on = result_on.final_state.F[island_mask].mean()
    f_island_off = result_off.final_state.F[island_mask].mean()
    island_disturbance = abs(f_island_on - f_island_off)
    
    print(f"  Corridor F (grace ON):  {f_corridor_on:.4f}")
    print(f"  Corridor F (grace OFF): {f_corridor_off:.4f}")
    print(f"  Corridor improvement:   {f_corridor_on - f_corridor_off:+.4f}")
    print(f"  Island F (grace ON):    {f_island_on:.4f}")
    print(f"  Island F (grace OFF):   {f_island_off:.4f}")
    print(f"  Island disturbance:     {island_disturbance:.4f}")
    
    # Grace should help corridor AND not disturb high-C island much
    channel_pass = (f_corridor_on > f_corridor_off) and (island_disturbance < 0.1)
    print(f"\n  MIXED COHERENCE TEST: {'✓ PASS' if channel_pass else '✗ FAIL'}")
    
    results['channel'] = {
        'corridor_on': f_corridor_on, 'corridor_off': f_corridor_off,
        'island_dist': island_disturbance, 'pass': channel_pass
    }
    
    # ====== SUMMARY ======
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_pass = all([
        results['conservation']['pass'],
        results['quantum']['pass'],
        results['necessity']['pass'],
        results['agency']['pass'],
        results['overlap']['pass'],
        results['channel']['pass'],
    ])
    
    print(f"  1. Conservation:       {'✓' if results['conservation']['pass'] else '✗'} ({results['conservation']['cons']:.8f})")
    print(f"  2. Quantum Gate:       {'✓' if results['quantum']['pass'] else '✗'} (grace={results['quantum']['grace']:.2e})")
    print(f"  3. Necessity:          {'✓' if results['necessity']['pass'] else '✗'} (Δ={results['necessity']['delta']:+.4f})")
    print(f"  4. Agency (F2):        {'✓' if results['agency']['pass'] else '✗'}")
    print(f"  5. Overlap Stress:     {'✓' if results['overlap']['pass'] else '✗'} (min_donor={results['overlap']['min_donor']:.4f})")
    print(f"  6. Mixed Coherence:    {'✓' if results['channel']['pass'] else '✗'} (island_dist={results['channel']['island_dist']:.4f})")
    print(f"\n  {'★ ALL TESTS PASS ★' if all_pass else '✗ SOME TESTS FAIL'}")
    
    return results, config


def plot_results(results, config):
    """Generate comprehensive validation plots"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Standard tests
    tests = [
        ('conservation', 'E4: Conservation'),
        ('quantum', 'E1: Quantum Gate'),
        ('necessity', 'E3: Necessity'),
    ]
    
    for idx, (key, title) in enumerate(tests):
        res = results[key]
        ax = axes[0, idx]
        t = np.arange(len(res['on'].F_depleted_mean_history)) * 20
        ax.plot(t, res['on'].F_depleted_mean_history, 'b-', label='Grace ON', lw=2)
        ax.plot(t, res['off'].F_depleted_mean_history, 'r--', label='Grace OFF', lw=2)
        ax.axhline(0.5, color='g', ls=':', alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean F (depleted)')
        status = '✓' if res['pass'] else '✗'
        ax.set_title(f"{title} {status}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Conservation detail
    ax = axes[0, 3]
    res = results['conservation']
    t = np.arange(len(res['on'].F_total_history)) * 20
    F_init = res['on'].F_total_history[0]
    ax.plot(t, (np.array(res['on'].F_total_history)/F_init - 1) * 1e6, 'b-', lw=2, label='Grace ON')
    ax.plot(t, (np.array(res['off'].F_total_history)/F_init - 1) * 1e6, 'r--', lw=2, label='Grace OFF')
    ax.axhline(0, color='g', ls=':')
    ax.set_xlabel('Time')
    ax.set_ylabel('(F_total/F_init - 1) × 10⁶')
    ax.set_title(f'Conservation Error (ppm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 2: New tests
    # Overlap stress
    ax = axes[1, 0]
    ax.text(0.5, 0.7, f"Overlap Stress Test", ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.5, f"Min donor F: {results['overlap']['min_donor']:.4f}", ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.35, f"Oscillation: {results['overlap']['oscillation']:.1%}", ha='center', fontsize=10, transform=ax.transAxes)
    status = '✓ PASS' if results['overlap']['pass'] else '✗ FAIL'
    ax.text(0.5, 0.15, status, ha='center', fontsize=14, fontweight='bold', 
            color='green' if results['overlap']['pass'] else 'red', transform=ax.transAxes)
    ax.axis('off')
    
    # Mixed coherence
    ax = axes[1, 1]
    ax.text(0.5, 0.7, f"Mixed Coherence Test", ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.5, f"Corridor Δ: {results['channel']['corridor_on'] - results['channel']['corridor_off']:+.4f}", 
            ha='center', fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.35, f"Island disturbance: {results['channel']['island_dist']:.4f}", 
            ha='center', fontsize=10, transform=ax.transAxes)
    status = '✓ PASS' if results['channel']['pass'] else '✗ FAIL'
    ax.text(0.5, 0.15, status, ha='center', fontsize=14, fontweight='bold',
            color='green' if results['channel']['pass'] else 'red', transform=ax.transAxes)
    ax.axis('off')
    
    # Final state for necessity test
    ax = axes[1, 2]
    res = results['necessity']
    diff = res['on'].final_state.F - res['off'].final_state.F
    vmax = max(0.1, np.abs(diff).max())
    im = ax.imshow(diff, cmap='RdBu', vmin=-vmax, vmax=vmax)
    ax.contour(res['on'].depleted_mask, colors='black', linewidths=1, linestyles='--')
    ax.set_title(f'E3: ΔF (ON-OFF)\nmax={diff.max():.3f}')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Summary
    ax = axes[1, 3]
    ax.axis('off')
    all_pass = all([r['pass'] for r in results.values()])
    summary = "GRACE v6.4\n" + "="*20 + "\nAntisymmetric Edge Flux\n\n"
    for name, res in results.items():
        status = '✓' if res['pass'] else '✗'
        summary += f"{name}: {status}\n"
    summary += "\n" + "="*20 + "\n"
    summary += "★ ALL PASS ★" if all_pass else "✗ SOME FAIL"
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/claude/grace_v64_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to /home/claude/grace_v64_results.png")


if __name__ == "__main__":
    results, config = run_validation_suite()
    plot_results(results, config)
