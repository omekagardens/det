"""
DET v6.x Concurrent Regimes & Partial Observability Simulation
================================================================

Implements the proposed extension spec for:
  - K-regime ("Kingdom/Utopia"): high coherence, low structural debt, high stable presence
  - W-regime ("World"): lower coherence, higher structural debt, noisier presence
  - Observability Gate: local measurement channel for cross-regime perception
  - Attunement Feedback: optional coherence reinforcement from mutual observation

Uses the DET v6.3 1D Collider as the simulation engine.

Reference: DET v6.x Extension Spec: Concurrent Regimes & Partial Observability
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from det_v6_3_1d_collider import DETCollider1D, DETParams1D


# ============================================================
# REGIME & OBSERVABILITY PARAMETERS
# ============================================================

@dataclass
class RegimeParams:
    """Parameters for the Concurrent Regimes & Observability extension.
    
    All weights are Bucket B/C parameters per the spec.
    """
    # --- Regime Index K_i weights ---
    w_C: float = 0.35       # weight for mean coherence
    w_a: float = 0.25       # weight for agency
    w_P: float = 0.15       # weight for normalized presence
    w_q: float = 0.25       # weight for structural debt (subtracted)
    
    # --- Observability Gate exponents ---
    alpha_obs: float = 1.0   # agency exponent in O_i
    beta_obs: float = 1.0    # coherence exponent in O_i
    gamma_obs: float = 1.0   # (1-q) exponent in O_i
    
    # --- Attunement feedback ---
    attunement_enabled: bool = False
    eta_attune: float = 0.01  # attunement coupling strength
    
    # --- Numerical ---
    epsilon: float = 1e-8     # regularization constant


# ============================================================
# REGIME DIAGNOSTICS RECORD
# ============================================================

@dataclass
class RegimeDiagnostics:
    """Snapshot of all regime and observability fields at one timestep."""
    step: int
    K: np.ndarray           # Regime index per node
    O: np.ndarray           # Observability gate per node
    Xi: np.ndarray          # Raw structuredness per node
    Xi_seen: np.ndarray     # Perceived structuredness per node
    mean_K: float
    mean_O: float
    mean_Xi_seen: float


# ============================================================
# CONCURRENT REGIMES SIMULATOR
# ============================================================

class DETRegimeSimulator:
    """
    Simulates concurrent K/W regimes and partial observability
    within the DET framework.
    
    Key principles:
      - All readouts are strictly local (neighborhood only)
      - No hidden globals or nonlocal coupling
      - Observability is gated by the observer's own state
      - Attunement feedback is optional and agency-gated
    """
    
    def __init__(self, collider: DETCollider1D, regime_params: Optional[RegimeParams] = None):
        self.sim = collider
        self.rp = regime_params or RegimeParams()
        self.N = collider.p.N
        
        # Per-node fields
        self.K = np.zeros(self.N)         # Regime index
        self.O = np.zeros(self.N)         # Observability gate
        self.Xi = np.zeros(self.N)        # Raw structuredness
        self.Xi_seen = np.zeros(self.N)   # Perceived structuredness
        
        # Phase angles (complex wavefunction) for signal computation
        self.theta = np.random.uniform(0, 2*np.pi, self.N)
        
        # Diagnostics history
        self.history: List[RegimeDiagnostics] = []
    
    def compute_regime_index(self) -> np.ndarray:
        """
        Compute the local Regime Index K_i for each node.
        
        K_i = clip(w_C * C̄_i + w_a * a_i + w_P * P̃_i - w_q * q_i, 0, 1)
        
        Where:
          C̄_i = mean coherence over neighbors
          P̃_i = P_i / (mean P over neighbors + epsilon)
          
        K_i ≈ 1 → Kingdom-like
        K_i ≈ 0 → World-like
        """
        rp = self.rp
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        
        # Mean coherence over neighbors (left and right bonds)
        C_bar = 0.5 * (self.sim.C_R + L(self.sim.C_R))
        
        # Normalized presence
        P = self.sim.P
        P_neighbors = 0.5 * (R(P) + L(P))
        P_tilde = P / (P_neighbors + rp.epsilon)
        P_tilde = np.clip(P_tilde, 0, 3.0)  # cap to avoid extreme values
        
        # Regime index
        K = rp.w_C * C_bar + rp.w_a * self.sim.a + rp.w_P * P_tilde - rp.w_q * self.sim.q
        self.K = np.clip(K, 0, 1)
        return self.K
    
    def compute_structuredness(self) -> np.ndarray:
        """
        Compute the local phase-coherence structuredness Ξ_i.
        
        S_i = Σ_{j∈N(i)} √C_ij * ψ_j  (complex sum)
        Ξ_i = |S_i| / (Σ √C_ij * |ψ_j| + ε)
        
        Ξ_i ≈ 1 → neighborhood is phase-aligned (structured signal)
        Ξ_i ≈ 0 → neighborhood is scrambled (noise)
        """
        rp = self.rp
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        
        # Complex wavefunction amplitudes
        F = self.sim.F
        amp = np.sqrt(F / (F + rp.epsilon))
        psi = amp * np.exp(1j * self.theta)
        
        # Incoming complex sum from neighbors
        sqrt_C_R = np.sqrt(self.sim.C_R)
        sqrt_C_L = np.sqrt(L(self.sim.C_R))
        
        S = sqrt_C_R * R(psi) + sqrt_C_L * L(psi)
        
        # Denominator: sum of absolute contributions
        denom = sqrt_C_R * np.abs(R(psi)) + sqrt_C_L * np.abs(L(psi)) + rp.epsilon
        
        Xi = np.abs(S) / denom
        self.Xi = np.clip(Xi, 0, 1)
        return self.Xi
    
    def compute_observability_gate(self) -> np.ndarray:
        """
        Compute the Observability Gate O_i.
        
        O_i = clip((a_i)^α * (C̄_i)^β * (1-q_i)^γ, 0, 1)
        
        High a, high C, low q → high observability
        """
        rp = self.rp
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        
        C_bar = 0.5 * (self.sim.C_R + L(self.sim.C_R))
        
        O = (self.sim.a ** rp.alpha_obs) * \
            (C_bar ** rp.beta_obs) * \
            ((1.0 - self.sim.q) ** rp.gamma_obs)
        
        self.O = np.clip(O, 0, 1)
        return self.O
    
    def compute_perceived_structuredness(self) -> np.ndarray:
        """
        Compute what each node "can see": Ξ_i^(seen) = O_i * Ξ_i
        
        In W-regime, Ξ^seen is small even if K-region is adjacent.
        As node transitions, Ξ^seen rises.
        """
        self.Xi_seen = self.O * self.Xi
        return self.Xi_seen
    
    def apply_attunement_feedback(self):
        """
        Optional: Observation affects coherence (attunement).
        
        ΔC_ij^(attune) = η_O * √(O_i * O_j) * (Ξ_i^seen * Ξ_j^seen) * Δτ_ij
        
        Safety: gated by √(O_i * O_j) which includes agency.
        """
        if not self.rp.attunement_enabled:
            return
        
        rp = self.rp
        R = lambda x: np.roll(x, -1)
        
        # Mutual observability on right bond
        O_mutual = np.sqrt(self.O * R(self.O))
        
        # Mutual perceived structuredness
        Xi_mutual = self.Xi_seen * R(self.Xi_seen)
        
        # Bond-local proper time
        Delta_tau_R = 0.5 * (self.sim.Delta_tau + R(self.sim.Delta_tau))
        
        # Attunement increment
        dC_attune = rp.eta_attune * O_mutual * Xi_mutual * Delta_tau_R
        
        # Apply to coherence
        self.sim.C_R = np.clip(self.sim.C_R + dC_attune, 0, 1)
    
    def update_phases(self):
        """
        Update phase angles based on local dynamics.
        Phases evolve based on local resource and momentum.
        """
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        
        # Phase evolution: driven by local F gradient and momentum
        dtheta = 0.1 * (R(self.sim.F) - L(self.sim.F)) / (self.sim.F + self.rp.epsilon)
        if self.sim.p.momentum_enabled:
            dtheta += 0.05 * self.sim.pi_R
        
        self.theta = (self.theta + dtheta * self.sim.p.DT) % (2 * np.pi)
    
    def step(self, step_num: int) -> RegimeDiagnostics:
        """
        Execute one full step:
        1. Run the underlying DET collider step
        2. Update phase angles
        3. Compute all regime/observability readouts
        4. Optionally apply attunement feedback
        """
        # Step the underlying physics
        self.sim.step()
        
        # Update phases
        self.update_phases()
        
        # Compute all readouts
        self.compute_regime_index()
        self.compute_structuredness()
        self.compute_observability_gate()
        self.compute_perceived_structuredness()
        
        # Optional feedback
        self.apply_attunement_feedback()
        
        # Record diagnostics
        diag = RegimeDiagnostics(
            step=step_num,
            K=self.K.copy(),
            O=self.O.copy(),
            Xi=self.Xi.copy(),
            Xi_seen=self.Xi_seen.copy(),
            mean_K=float(np.mean(self.K)),
            mean_O=float(np.mean(self.O)),
            mean_Xi_seen=float(np.mean(self.Xi_seen))
        )
        self.history.append(diag)
        
        return diag


# ============================================================
# INITIALIZATION HELPERS
# ============================================================

def create_k_region(collider: DETCollider1D, center: int, width: int,
                    C_level: float = 0.9, q_level: float = 0.02, F_level: float = 2.0):
    """Initialize a Kingdom-like region: high C, low q, moderate F."""
    N = collider.p.N
    x = np.arange(N)
    dist = np.minimum(np.abs(x - center), N - np.abs(x - center))
    mask = dist < width
    
    # Set high coherence in region
    collider.C_R[mask] = np.maximum(collider.C_R[mask], C_level)
    # Set low structural debt
    collider.q[mask] = np.minimum(collider.q[mask], q_level)
    # Add moderate resource
    envelope = np.exp(-0.5 * (dist / (width * 0.5))**2)
    collider.F += F_level * envelope * mask
    # Ensure agency is high
    collider.a[mask] = np.maximum(collider.a[mask], 0.95)


def create_w_region(collider: DETCollider1D, center: int, width: int,
                    C_level: float = 0.15, q_level: float = 0.5, F_level: float = 0.5):
    """Initialize a World-like region: low C, high q, lower F."""
    N = collider.p.N
    x = np.arange(N)
    dist = np.minimum(np.abs(x - center), N - np.abs(x - center))
    mask = dist < width
    
    # Set low coherence
    collider.C_R[mask] = np.minimum(collider.C_R[mask], C_level)
    # Set high structural debt
    collider.q[mask] = np.maximum(collider.q[mask], q_level)
    # Add some resource
    envelope = np.exp(-0.5 * (dist / (width * 0.5))**2)
    collider.F += F_level * envelope * mask


# ============================================================
# EXPERIMENT E1: ADJACENT REGIMES IN ONE GRAPH
# ============================================================

def experiment_E1_adjacent_regimes(output_dir: str = "/home/ubuntu/det_regime_results"):
    """
    E1: Adjacent Regimes in One Graph
    
    Initialize a connected lattice with two regions:
      Region A (left): higher initial C, lower q  → K-regime
      Region B (right): lower C, higher q          → W-regime
    
    Let normal DET updates run and measure:
      K_i(t), O_i(t), Ξ_i^seen(t) near the boundary
    
    Prediction: nodes in B near boundary show increasing Ξ^seen
    only if they locally transition (C rises, q falls).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT E1: Adjacent Regimes in One Graph")
    print("="*70)
    
    N = 200
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0,
        C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.01,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
        boundary_enabled=True, grace_enabled=True
    )
    
    collider = DETCollider1D(params)
    
    # Region A (K-regime): left half
    create_k_region(collider, center=50, width=40, C_level=0.9, q_level=0.02, F_level=3.0)
    
    # Region B (W-regime): right half
    create_w_region(collider, center=150, width=40, C_level=0.12, q_level=0.55, F_level=0.3)
    
    # Regime simulator
    regime_params = RegimeParams(
        w_C=0.35, w_a=0.25, w_P=0.15, w_q=0.25,
        alpha_obs=1.0, beta_obs=1.0, gamma_obs=1.0,
        attunement_enabled=False
    )
    regime_sim = DETRegimeSimulator(collider, regime_params)
    
    # Run simulation
    n_steps = 2000
    
    # Track boundary nodes
    boundary_idx = 100  # boundary between regions
    K_node_left = 80    # deep in K-region
    W_node_right = 120  # in W-region near boundary
    W_node_deep = 160   # deep in W-region
    
    times = []
    K_at_boundary = []
    K_at_K = []
    K_at_W_near = []
    K_at_W_deep = []
    O_at_boundary = []
    O_at_K = []
    O_at_W_near = []
    O_at_W_deep = []
    Xi_seen_boundary = []
    Xi_seen_K = []
    Xi_seen_W_near = []
    Xi_seen_W_deep = []
    
    # Spatial snapshots
    snapshot_steps = [0, 200, 500, 1000, 1500, 1999]
    snapshots = []
    
    print(f"\nRunning {n_steps} steps...")
    
    for t in range(n_steps):
        diag = regime_sim.step(t)
        
        times.append(t)
        K_at_boundary.append(diag.K[boundary_idx])
        K_at_K.append(diag.K[K_node_left])
        K_at_W_near.append(diag.K[W_node_right])
        K_at_W_deep.append(diag.K[W_node_deep])
        
        O_at_boundary.append(diag.O[boundary_idx])
        O_at_K.append(diag.O[K_node_left])
        O_at_W_near.append(diag.O[W_node_right])
        O_at_W_deep.append(diag.O[W_node_deep])
        
        Xi_seen_boundary.append(diag.Xi_seen[boundary_idx])
        Xi_seen_K.append(diag.Xi_seen[K_node_left])
        Xi_seen_W_near.append(diag.Xi_seen[W_node_right])
        Xi_seen_W_deep.append(diag.Xi_seen[W_node_deep])
        
        if t in snapshot_steps:
            snapshots.append({
                'step': t,
                'K': diag.K.copy(),
                'O': diag.O.copy(),
                'Xi_seen': diag.Xi_seen.copy(),
                'C': collider.C_R.copy(),
                'q': collider.q.copy(),
                'F': collider.F.copy()
            })
            print(f"  Step {t}: mean K={diag.mean_K:.3f}, mean O={diag.mean_O:.3f}, "
                  f"mean Ξ^seen={diag.mean_Xi_seen:.3f}")
    
    # === PLOT 1: Time Evolution at Key Nodes ===
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("E1: Regime & Observability Evolution at Key Nodes", fontsize=14, fontweight='bold')
    
    ax = axes[0]
    ax.plot(times, K_at_K, 'b-', linewidth=1.5, label=f'K-region (node {K_node_left})')
    ax.plot(times, K_at_boundary, 'g-', linewidth=1.5, label=f'Boundary (node {boundary_idx})')
    ax.plot(times, K_at_W_near, 'orange', linewidth=1.5, label=f'W-near (node {W_node_right})')
    ax.plot(times, K_at_W_deep, 'r-', linewidth=1.5, label=f'W-deep (node {W_node_deep})')
    ax.set_ylabel('K_i (Regime Index)')
    ax.set_title('Regime Index Over Time')
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax = axes[1]
    ax.plot(times, O_at_K, 'b-', linewidth=1.5, label=f'K-region')
    ax.plot(times, O_at_boundary, 'g-', linewidth=1.5, label=f'Boundary')
    ax.plot(times, O_at_W_near, 'orange', linewidth=1.5, label=f'W-near')
    ax.plot(times, O_at_W_deep, 'r-', linewidth=1.5, label=f'W-deep')
    ax.set_ylabel('O_i (Observability)')
    ax.set_title('Observability Gate Over Time')
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(times, Xi_seen_K, 'b-', linewidth=1.5, label=f'K-region')
    ax.plot(times, Xi_seen_boundary, 'g-', linewidth=1.5, label=f'Boundary')
    ax.plot(times, Xi_seen_W_near, 'orange', linewidth=1.5, label=f'W-near')
    ax.plot(times, Xi_seen_W_deep, 'r-', linewidth=1.5, label=f'W-deep')
    ax.set_xlabel('Step')
    ax.set_ylabel('Ξ_i^seen')
    ax.set_title('Perceived Structuredness Over Time')
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/E1_time_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/E1_time_evolution.png")
    
    # === PLOT 2: Spatial Snapshots ===
    n_snaps = len(snapshots)
    fig, axes = plt.subplots(n_snaps, 3, figsize=(16, 3 * n_snaps))
    fig.suptitle("E1: Spatial Profiles at Key Timesteps", fontsize=14, fontweight='bold')
    
    x = np.arange(N)
    for i, snap in enumerate(snapshots):
        ax = axes[i, 0]
        ax.fill_between(x, snap['K'], alpha=0.7, color='blue')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel(f"Step {snap['step']}\nK_i")
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.set_title('Regime Index K_i')
        
        ax = axes[i, 1]
        ax.fill_between(x, snap['O'], alpha=0.7, color='green')
        ax.set_ylabel('O_i')
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.set_title('Observability Gate O_i')
        
        ax = axes[i, 2]
        ax.fill_between(x, snap['Xi_seen'], alpha=0.7, color='purple')
        ax.set_ylabel('Ξ^seen')
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.set_title('Perceived Structuredness Ξ^seen')
    
    axes[-1, 0].set_xlabel('Position')
    axes[-1, 1].set_xlabel('Position')
    axes[-1, 2].set_xlabel('Position')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/E1_spatial_snapshots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/E1_spatial_snapshots.png")
    
    return regime_sim


# ============================================================
# EXPERIMENT E2: ASYMMETRY TEST
# ============================================================

def experiment_E2_asymmetry_test(output_dir: str = "/home/ubuntu/det_regime_results"):
    """
    E2: "Kingdom can see world" asymmetry test
    
    Pick representative nodes:
      i in A (K-like)
      j in B (W-like)
    
    Compare Ξ_i^seen vs Ξ_j^seen when both observe across the same boundary.
    
    Prediction: Ξ_i^seen >> Ξ_j^seen until j transitions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT E2: Asymmetric Observability Test")
    print("="*70)
    
    N = 200
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0,
        C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.01,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
        boundary_enabled=True, grace_enabled=True
    )
    
    collider = DETCollider1D(params)
    
    # Sharp boundary at center
    boundary = N // 2
    
    # K-region: left side
    for i in range(boundary):
        collider.C_R[i] = 0.92
        collider.q[i] = 0.02
        collider.F[i] = 2.5
        collider.a[i] = 0.98
    
    # W-region: right side
    for i in range(boundary, N):
        collider.C_R[i] = 0.10
        collider.q[i] = 0.60
        collider.F[i] = 0.2
        collider.a[i] = 0.5
    
    regime_params = RegimeParams(
        w_C=0.35, w_a=0.25, w_P=0.15, w_q=0.25,
        alpha_obs=1.0, beta_obs=1.0, gamma_obs=1.0,
        attunement_enabled=False
    )
    regime_sim = DETRegimeSimulator(collider, regime_params)
    
    # Observer nodes equidistant from boundary
    K_observer = boundary - 5   # 5 nodes into K-region
    W_observer = boundary + 5   # 5 nodes into W-region
    
    n_steps = 2000
    times = []
    Xi_seen_K_obs = []
    Xi_seen_W_obs = []
    O_K_obs = []
    O_W_obs = []
    K_K_obs = []
    K_W_obs = []
    ratio_Xi = []
    
    print(f"\nK-observer at node {K_observer}, W-observer at node {W_observer}")
    print(f"Running {n_steps} steps...")
    
    for t in range(n_steps):
        diag = regime_sim.step(t)
        
        times.append(t)
        Xi_seen_K_obs.append(diag.Xi_seen[K_observer])
        Xi_seen_W_obs.append(diag.Xi_seen[W_observer])
        O_K_obs.append(diag.O[K_observer])
        O_W_obs.append(diag.O[W_observer])
        K_K_obs.append(diag.K[K_observer])
        K_W_obs.append(diag.K[W_observer])
        
        # Asymmetry ratio
        r = diag.Xi_seen[K_observer] / (diag.Xi_seen[W_observer] + 1e-10)
        ratio_Xi.append(min(r, 100))  # cap for plotting
    
    # === PLOT: Asymmetry ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("E2: Asymmetric Observability — K-observer vs W-observer", 
                 fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(times, Xi_seen_K_obs, 'b-', linewidth=1.5, label=f'K-observer (node {K_observer})')
    ax.plot(times, Xi_seen_W_obs, 'r-', linewidth=1.5, label=f'W-observer (node {W_observer})')
    ax.set_ylabel('Ξ^seen')
    ax.set_title('Perceived Structuredness: K vs W Observer')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.semilogy(times, np.array(ratio_Xi) + 0.01, 'purple', linewidth=1.5)
    ax.set_ylabel('Ξ^seen_K / Ξ^seen_W')
    ax.set_title('Asymmetry Ratio (log scale)')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Symmetry line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(times, O_K_obs, 'b-', linewidth=1.5, label='O (K-observer)')
    ax.plot(times, O_W_obs, 'r-', linewidth=1.5, label='O (W-observer)')
    ax.set_xlabel('Step')
    ax.set_ylabel('O_i')
    ax.set_title('Observability Gate Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(times, K_K_obs, 'b-', linewidth=1.5, label='K (K-observer)')
    ax.plot(times, K_W_obs, 'r-', linewidth=1.5, label='K (W-observer)')
    ax.set_xlabel('Step')
    ax.set_ylabel('K_i')
    ax.set_title('Regime Index Comparison')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/E2_asymmetry_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/E2_asymmetry_test.png")
    
    # Print summary
    mean_Xi_K = np.mean(Xi_seen_K_obs[-500:])
    mean_Xi_W = np.mean(Xi_seen_W_obs[-500:])
    print(f"\n  Late-time mean Ξ^seen (K-observer): {mean_Xi_K:.4f}")
    print(f"  Late-time mean Ξ^seen (W-observer): {mean_Xi_W:.4f}")
    print(f"  Asymmetry ratio: {mean_Xi_K / (mean_Xi_W + 1e-10):.2f}x")
    
    return regime_sim


# ============================================================
# EXPERIMENT E3: ATTUNEMENT FEEDBACK
# ============================================================

def experiment_E3_attunement_feedback(output_dir: str = "/home/ubuntu/det_regime_results"):
    """
    E3: Attunement Feedback Test
    
    Compare dynamics with and without the attunement feedback term.
    
    With attunement: mutually capable neighbors stabilize coherence faster.
    Without: baseline DET dynamics only.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT E3: Attunement Feedback Comparison")
    print("="*70)
    
    results = {}
    
    for label, attune_enabled, eta in [
        ("No Attunement", False, 0.0),
        ("Mild Attunement (η=0.02)", True, 0.02),
        ("Strong Attunement (η=0.10)", True, 0.10)
    ]:
        print(f"\n--- {label} ---")
        
        N = 200
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0,
            C_init=0.3,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.01,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
            boundary_enabled=True, grace_enabled=True
        )
        
        collider = DETCollider1D(params)
        
        # Same initial conditions
        np.random.seed(42)
        create_k_region(collider, center=50, width=40, C_level=0.9, q_level=0.02, F_level=3.0)
        create_w_region(collider, center=150, width=40, C_level=0.12, q_level=0.55, F_level=0.3)
        
        regime_params = RegimeParams(
            w_C=0.35, w_a=0.25, w_P=0.15, w_q=0.25,
            attunement_enabled=attune_enabled,
            eta_attune=eta
        )
        regime_sim = DETRegimeSimulator(collider, regime_params)
        
        n_steps = 2000
        times = []
        mean_K_vals = []
        mean_C_vals = []
        boundary_Xi_seen = []
        
        boundary_node = 100
        
        for t in range(n_steps):
            diag = regime_sim.step(t)
            times.append(t)
            mean_K_vals.append(diag.mean_K)
            mean_C_vals.append(float(np.mean(collider.C_R)))
            boundary_Xi_seen.append(diag.Xi_seen[boundary_node])
        
        results[label] = {
            'times': times,
            'mean_K': mean_K_vals,
            'mean_C': mean_C_vals,
            'boundary_Xi_seen': boundary_Xi_seen
        }
        
        print(f"  Final mean K: {mean_K_vals[-1]:.4f}")
        print(f"  Final mean C: {mean_C_vals[-1]:.4f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("E3: Effect of Attunement Feedback on Regime Dynamics", 
                 fontsize=14, fontweight='bold')
    
    colors = {'No Attunement': 'red', 'Mild Attunement (η=0.02)': 'blue', 
              'Strong Attunement (η=0.10)': 'green'}
    
    for label, data in results.items():
        axes[0].plot(data['times'], data['mean_K'], color=colors[label], 
                    linewidth=1.5, label=label)
        axes[1].plot(data['times'], data['mean_C'], color=colors[label], 
                    linewidth=1.5, label=label)
        axes[2].plot(data['times'], data['boundary_Xi_seen'], color=colors[label], 
                    linewidth=1.5, label=label, alpha=0.7)
    
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Mean K_i')
    axes[0].set_title('System-Wide Regime Index')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Mean C')
    axes[1].set_title('System-Wide Coherence')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Ξ^seen at boundary')
    axes[2].set_title('Boundary Perceived Structuredness')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/E3_attunement_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_dir}/E3_attunement_comparison.png")
    
    return results


# ============================================================
# EXPERIMENT F: FALSIFIER TESTS
# ============================================================

def experiment_falsifiers(output_dir: str = "/home/ubuntu/det_regime_results"):
    """
    Run all four falsifier tests from the spec:
    
    F_KO1: No Hidden Globals
    F_KO2: No Coercion (a=0 → O=0)
    F_KO3: Continuity
    F_KO4: Asymmetry arises locally
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("FALSIFIER TESTS")
    print("="*70)
    
    results = {}
    
    # --- F_KO1: No Hidden Globals ---
    print("\n--- F_KO1: No Hidden Globals ---")
    print("  Test: Replacing far-away K-region with random data cannot change O/Ξ in W-region")
    
    N = 200
    params = DETParams1D(N=N, DT=0.02, F_VAC=0.01, C_init=0.3,
                         momentum_enabled=True, q_enabled=True,
                         gravity_enabled=False,  # disable gravity to isolate locality
                         boundary_enabled=False)
    
    # Run A: with K-region far away (nodes 10-30)
    collider_A = DETCollider1D(params)
    create_k_region(collider_A, center=20, width=10, C_level=0.9, q_level=0.02, F_level=3.0)
    create_w_region(collider_A, center=180, width=10, C_level=0.12, q_level=0.55, F_level=0.3)
    
    # Use fixed seed for phase initialization
    np.random.seed(42)
    regime_A = DETRegimeSimulator(collider_A, RegimeParams())
    
    # Run B: K-region replaced with random (same W-region)
    collider_B = DETCollider1D(params)
    # Randomize the far-away region differently
    for i in range(10, 30):
        collider_B.C_R[i] = 0.5  # just different from K-region
        collider_B.q[i] = 0.3
        collider_B.F[i] = 1.0
    create_w_region(collider_B, center=180, width=10, C_level=0.12, q_level=0.55, F_level=0.3)
    
    # Use SAME seed for phase initialization so Xi comparison is fair
    np.random.seed(42)
    regime_B = DETRegimeSimulator(collider_B, RegimeParams())
    
    # Run only 10 steps — short enough that signals cannot propagate across the gap
    # (with DT=0.02 and local-only updates, 10 steps affects ~10 neighbors max)
    for t in range(10):
        diag_A = regime_A.step(t)
        diag_B = regime_B.step(t)
    
    # Compare O and Xi_seen in the W-region (nodes 170-190) — far from K-region
    w_slice = slice(170, 190)
    O_diff = np.max(np.abs(diag_A.O[w_slice] - diag_B.O[w_slice]))
    Xi_diff = np.max(np.abs(diag_A.Xi_seen[w_slice] - diag_B.Xi_seen[w_slice]))
    
    fko1_pass = O_diff < 0.01 and Xi_diff < 0.01
    print(f"  Max |ΔO| in W-region: {O_diff:.6f}")
    print(f"  Max |ΔΞ^seen| in W-region: {Xi_diff:.6f}")
    print(f"  F_KO1: {'PASS' if fko1_pass else 'FAIL'}")
    results['F_KO1'] = fko1_pass
    
    # --- F_KO2: No Coercion ---
    print("\n--- F_KO2: No Coercion ---")
    print("  Test: If a_i=0, then O_i=0 and Ξ_i^seen=0 always")
    
    collider_C = DETCollider1D(DETParams1D(N=50, DT=0.02, C_init=0.5,
                                            boundary_enabled=False, gravity_enabled=False))
    collider_C.a[:] = 0.0  # zero agency everywhere
    collider_C.F[:] = 5.0  # plenty of resource
    collider_C.C_R[:] = 0.9  # high coherence
    
    regime_C = DETRegimeSimulator(collider_C, RegimeParams())
    
    for t in range(100):
        # Force a=0 after each collider step (collider's agency update would
        # restore a via the ceiling law; we override to test the O_i gate)
        diag_C = regime_C.step(t)
        collider_C.a[:] = 0.0
        # Recompute O with forced a=0
        regime_C.compute_observability_gate()
        regime_C.compute_perceived_structuredness()
        diag_C.O = regime_C.O.copy()
        diag_C.Xi_seen = regime_C.Xi_seen.copy()
    
    max_O = np.max(diag_C.O)
    max_Xi_seen = np.max(diag_C.Xi_seen)
    
    fko2_pass = max_O < 1e-10 and max_Xi_seen < 1e-10
    print(f"  Max O with a=0: {max_O:.2e}")
    print(f"  Max Ξ^seen with a=0: {max_Xi_seen:.2e}")
    print(f"  F_KO2: {'PASS' if fko2_pass else 'FAIL'}")
    results['F_KO2'] = fko2_pass
    
    # --- F_KO3: Continuity ---
    print("\n--- F_KO3: Continuity ---")
    print("  Test: Scanning coherence smoothly changes Ξ^seen — no discontinuous jumps")
    
    C_values = np.linspace(0.05, 0.95, 40)
    Xi_seen_values = []
    
    for C_val in C_values:
        np.random.seed(123)  # Same seed for each run to isolate C effect
        collider_D = DETCollider1D(DETParams1D(N=30, DT=0.02, C_init=C_val,
                                                boundary_enabled=False, gravity_enabled=False))
        collider_D.F[:] = 2.0
        collider_D.q[:] = 0.1
        np.random.seed(123)  # Reset seed for phase init
        regime_D = DETRegimeSimulator(collider_D, RegimeParams())
        
        # Run a few steps to stabilize
        for t in range(30):
            diag_D = regime_D.step(t)
        
        Xi_seen_values.append(float(np.mean(diag_D.Xi_seen)))
    
    # Check for continuity: max jump between adjacent C values
    Xi_arr = np.array(Xi_seen_values)
    max_jump = np.max(np.abs(np.diff(Xi_arr)))
    
    fko3_pass = max_jump < 0.20  # no large discontinuities (phase noise adds ~0.05)
    print(f"  Max jump in Ξ^seen over C scan: {max_jump:.4f}")
    print(f"  F_KO3: {'PASS' if fko3_pass else 'FAIL'}")
    results['F_KO3'] = fko3_pass
    
    # --- F_KO4: Asymmetry arises locally ---
    print("\n--- F_KO4: Asymmetry arises locally ---")
    print("  Test: Identical local states → identical O and observation capacity")
    
    collider_E = DETCollider1D(DETParams1D(N=100, DT=0.02, C_init=0.5,
                                            boundary_enabled=False, gravity_enabled=False))
    collider_E.F[:] = 2.0
    collider_E.q[:] = 0.2
    collider_E.a[:] = 0.8
    
    regime_E = DETRegimeSimulator(collider_E, RegimeParams())
    
    for t in range(50):
        diag_E = regime_E.step(t)
    
    # All nodes should have nearly identical O (modulo periodic boundary effects)
    O_std = np.std(diag_E.O)
    O_range = np.max(diag_E.O) - np.min(diag_E.O)
    
    fko4_pass = O_range < 0.01
    print(f"  O range across identical nodes: {O_range:.6f}")
    print(f"  O std: {O_std:.6f}")
    print(f"  F_KO4: {'PASS' if fko4_pass else 'FAIL'}")
    results['F_KO4'] = fko4_pass
    
    # === Summary ===
    print("\n" + "="*70)
    print("FALSIFIER SUMMARY")
    print("="*70)
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name}: {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'} ({sum(results.values())}/{len(results)})")
    
    # Save continuity plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(C_values, Xi_seen_values, 'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Initial Coherence C')
    ax.set_ylabel('Mean Ξ^seen')
    ax.set_title('F_KO3 Continuity Test: Ξ^seen vs Coherence')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/falsifier_continuity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_dir}/falsifier_continuity.png")
    
    return results


# ============================================================
# EXPERIMENT E4: DECAY-REGIME COUPLING
# ============================================================

def experiment_E4_decay_regime_coupling(output_dir: str = "/home/ubuntu/det_regime_results"):
    """
    E4: Coupling between radioactive decay and regime dynamics.
    
    Shows that:
    1. K-regime nodes are decay-resistant (low instability)
    2. W-regime nodes are decay-prone (high instability)
    3. Decay events push nodes further into W-regime
    4. The regime index K_i is functionally the anti-instability score
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("EXPERIMENT E4: Decay-Regime Coupling")
    print("="*70)
    
    # Import decay module
    from det_radioactive_decay import DETDecaySimulator, DecayParams, create_nucleus_cluster
    
    N = 300
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0,
        C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.015,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
        boundary_enabled=True, grace_enabled=True
    )
    
    collider = DETCollider1D(params)
    
    # K-region on left with nuclei
    create_k_region(collider, center=75, width=50, C_level=0.85, q_level=0.05, F_level=2.0)
    
    # W-region on right with nuclei
    create_w_region(collider, center=225, width=50, C_level=0.15, q_level=0.45, F_level=0.5)
    
    # Place identical nuclei in BOTH regions
    n_nuclei_per_region = 5
    K_positions = [50, 60, 70, 80, 90]
    W_positions = [210, 220, 230, 240, 250]
    
    for pos in K_positions + W_positions:
        create_nucleus_cluster(collider, pos, mass=8.0, q_level=0.55, width=2.0)
    
    # Set up both simulators on the same collider
    decay_params = DecayParams(
        w_q=1.0, w_grad=0.5, w_C=0.3,
        s_crit=0.60,
        T_eff_window=15, T_eff_floor=0.002,
        energy_release_fraction=0.35,
        q_reduction=0.20,
        coherence_damage_radius=5,
        coherence_damage_factor=0.15
    )
    
    regime_params = RegimeParams(
        w_C=0.35, w_a=0.25, w_P=0.15, w_q=0.25,
        attunement_enabled=False
    )
    
    decay_sim = DETDecaySimulator(collider, decay_params)
    regime_sim = DETRegimeSimulator(collider, regime_params)
    
    # Run simulation
    n_steps = 2000
    times = []
    K_decays = []
    W_decays = []
    mean_K_K_region = []
    mean_K_W_region = []
    
    K_total = 0
    W_total = 0
    
    K_slice = slice(30, 110)
    W_slice = slice(190, 270)
    
    print(f"\nRunning {n_steps} steps with both decay and regime dynamics...")
    
    for t in range(n_steps):
        # Step the decay simulator (which steps the collider)
        events = decay_sim.step(t)
        
        # Compute regime readouts (without stepping collider again)
        regime_sim.update_phases()
        regime_sim.compute_regime_index()
        regime_sim.compute_structuredness()
        regime_sim.compute_observability_gate()
        regime_sim.compute_perceived_structuredness()
        
        for ev in events:
            if ev.position in range(30, 110):
                K_total += 1
                print(f"  K-REGION DECAY at step {ev.step}, pos={ev.position}")
            elif ev.position in range(190, 270):
                W_total += 1
                print(f"  W-REGION DECAY at step {ev.step}, pos={ev.position}")
        
        times.append(t)
        K_decays.append(K_total)
        W_decays.append(W_total)
        mean_K_K_region.append(float(np.mean(regime_sim.K[K_slice])))
        mean_K_W_region.append(float(np.mean(regime_sim.K[W_slice])))
    
    print(f"\n  Total K-region decays: {K_total}")
    print(f"  Total W-region decays: {W_total}")
    print(f"  Ratio (W/K): {W_total / max(K_total, 1):.1f}x")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("E4: Decay-Regime Coupling — K-regime Resists Decay", 
                 fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(times, K_decays, 'b-', linewidth=2, label='K-region decays')
    ax.plot(times, W_decays, 'r-', linewidth=2, label='W-region decays')
    ax.set_ylabel('Cumulative Decays')
    ax.set_title('Decay Events by Region')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(times, mean_K_K_region, 'b-', linewidth=1.5, label='K-region mean K_i')
    ax.plot(times, mean_K_W_region, 'r-', linewidth=1.5, label='W-region mean K_i')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Mean K_i')
    ax.set_title('Regime Index by Region')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final spatial profiles
    ax = axes[1, 0]
    x = np.arange(N)
    ax.fill_between(x, regime_sim.K, alpha=0.7, color='blue', label='K_i')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('K_i')
    ax.set_title('Final Regime Index Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.fill_between(x, decay_sim.instability_scores, alpha=0.7, color='red', label='s_i')
    ax.axhline(y=decay_params.s_crit, color='black', linestyle='--', alpha=0.5, label='s_crit')
    ax.set_xlabel('Position')
    ax.set_ylabel('s_i')
    ax.set_title('Final Instability Score Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/E4_decay_regime_coupling.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/E4_decay_regime_coupling.png")
    
    return {'K_decays': K_total, 'W_decays': W_total}


# ============================================================
# MAIN RUNNER
# ============================================================

def run_all_experiments():
    """Run all regime and observability experiments."""
    output_dir = "/home/ubuntu/det_regime_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("DET v6.x CONCURRENT REGIMES & PARTIAL OBSERVABILITY")
    print("SIMULATION SUITE")
    print("="*70)
    
    results = {}
    
    # E1: Adjacent Regimes
    results['E1'] = experiment_E1_adjacent_regimes(output_dir)
    
    # E2: Asymmetry Test
    results['E2'] = experiment_E2_asymmetry_test(output_dir)
    
    # E3: Attunement Feedback
    results['E3'] = experiment_E3_attunement_feedback(output_dir)
    
    # Falsifier Tests
    results['falsifiers'] = experiment_falsifiers(output_dir)
    
    # E4: Decay-Regime Coupling
    results['E4'] = experiment_E4_decay_regime_coupling(output_dir)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}/")
    
    return results


if __name__ == "__main__":
    run_all_experiments()
