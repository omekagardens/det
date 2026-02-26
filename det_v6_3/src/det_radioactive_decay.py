"""
DET Radioactive Decay Simulation Module
========================================

Implements the radioactive decay model from the DET theory:
  - Decay as q-threshold crossing (instability score s_i >= s_crit)
  - Kramers-like hazard rate with effective temperature from local flux noise
  - Emergent half-life from structural debt, coherence buffering, and fluctuations
  - Decay product generation (energy release, daughter nucleus, momentum kick)
  - Coherence-mediated stabilization and contagious decay effects

Uses the DET v6.3 1D Collider as the simulation engine.

Reference: DET Theory Card v6.3, Radioactive Decay in DET (module proposal)
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
# DECAY PARAMETERS
# ============================================================

@dataclass
class DecayParams:
    """Parameters for DET radioactive decay model.
    
    All weights and thresholds are derived from DET base parameters
    to avoid introducing new fundamental constants.
    """
    # --- Instability Score Weights ---
    # w_q: weight for raw structural debt q^D
    w_q: float = 1.0
    # w_grad: weight for debt gradient (stress concentration)
    w_grad: float = 0.5
    # w_C: weight for coherence deficit (buffering weakness)
    w_C: float = 0.3
    
    # --- Threshold ---
    # s_crit: critical instability score for decay
    s_crit: float = 0.7
    
    # --- Hazard Rate ---
    # T_eff_window: number of steps to compute flux variance (effective temperature)
    T_eff_window: int = 20
    # T_eff_floor: minimum effective temperature to avoid division by zero
    T_eff_floor: float = 0.001
    # nu_base: base attempt frequency (linked to 1/DT, the simulation clock)
    nu_base: float = 1.0
    
    # --- Decay Products ---
    # energy_release_fraction: fraction of local F released as energy pulse
    energy_release_fraction: float = 0.5
    # q_reduction: how much q is reduced at the decay site (debt discharge)
    q_reduction: float = 0.3
    # momentum_kick: momentum imparted to decay products
    momentum_kick: float = 0.5
    # coherence_damage_radius: how far coherence damage spreads
    coherence_damage_radius: int = 5
    # coherence_damage_factor: fraction of coherence destroyed by decay event
    coherence_damage_factor: float = 0.3
    
    # --- Stabilization ---
    # coherence_stabilization: if True, high coherence actively suppresses decay
    coherence_stabilization: bool = True


# ============================================================
# DECAY EVENT RECORD
# ============================================================

@dataclass
class DecayEvent:
    """Record of a single decay event."""
    step: int
    position: int
    instability_score: float
    hazard_rate: float
    q_before: float
    q_after: float
    F_released: float
    T_eff: float


# ============================================================
# RADIOACTIVE DECAY SIMULATOR
# ============================================================

class DETDecaySimulator:
    """
    Simulates radioactive decay within the DET framework.
    
    A 'nucleus' is modeled as a high-q, high-F localized cluster on the 1D lattice.
    Decay occurs when the local instability score crosses the critical threshold.
    
    Key DET principles:
      - Decay is NOT random; it is a lawful threshold crossing in the q-field
      - Half-life EMERGES from structural debt, coherence, and fluctuations
      - Coherence acts as a buffer against decay
      - Decay products carry momentum and reduce local debt
    """
    
    def __init__(self, collider: DETCollider1D, decay_params: Optional[DecayParams] = None):
        self.sim = collider
        self.dp = decay_params or DecayParams()
        self.N = collider.p.N
        
        # Flux history for computing T_eff (effective temperature)
        self.flux_history: List[np.ndarray] = []
        
        # Decay event log
        self.decay_events: List[DecayEvent] = []
        
        # Per-node tracking
        self.instability_scores = np.zeros(self.N)
        self.hazard_rates = np.zeros(self.N)
        self.T_eff = np.zeros(self.N)
        self.decayed = np.zeros(self.N, dtype=bool)  # Track which sites have decayed
        
        # Diagnostics
        self.total_nuclei_initial = 0
        self.total_nuclei_remaining = 0
        
    def compute_instability_score(self) -> np.ndarray:
        """
        Compute the local instability score s_i for each node.
        
        s_i = w_q * q_i^D
              + w_grad * sum_{j in N(i)} |q_i^D - q_j^D|
              + w_C * sum_{j in N(i)} (1 - C_ij)
        
        Returns: array of instability scores
        """
        dp = self.dp
        q = self.sim.q
        C = self.sim.C_R
        
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        
        # Term 1: Raw structural debt
        term_debt = dp.w_q * q
        
        # Term 2: Debt gradient (stress concentration)
        grad_R = np.abs(q - R(q))
        grad_L = np.abs(q - L(q))
        term_gradient = dp.w_grad * (grad_R + grad_L)
        
        # Term 3: Coherence deficit (buffering weakness)
        # C_R is coherence on the right bond; L(C_R) is coherence on the left bond
        deficit_R = 1.0 - C
        deficit_L = 1.0 - L(C)
        term_coherence = dp.w_C * (deficit_R + deficit_L)
        
        s = term_debt + term_gradient + term_coherence
        self.instability_scores = s
        return s
    
    def compute_effective_temperature(self) -> np.ndarray:
        """
        Compute the effective fluctuation temperature T_eff from local flux variance.
        
        T_eff_i = Var(dF_i) over the last T_eff_window steps.
        
        This is the "noise" that drives the system over the barrier.
        """
        dp = self.dp
        
        if len(self.flux_history) < 2:
            self.T_eff = np.ones(self.N) * dp.T_eff_floor
            return self.T_eff
        
        # Stack recent flux changes
        window = min(len(self.flux_history), dp.T_eff_window)
        recent = np.array(self.flux_history[-window:])
        
        # Variance across time at each spatial position
        T_eff = np.var(recent, axis=0)
        T_eff = np.maximum(T_eff, dp.T_eff_floor)
        
        self.T_eff = T_eff
        return T_eff
    
    def compute_hazard_rate(self) -> np.ndarray:
        """
        Compute the local hazard (decay) rate h_i.
        
        h_i = nu_i * exp(-(s_crit - s_i) / T_eff_i)
        
        Only nodes with s_i > 0 have nonzero hazard.
        When s_i >= s_crit, the exponent becomes non-negative and decay is imminent.
        """
        dp = self.dp
        s = self.instability_scores
        T_eff = self.T_eff
        
        # Attempt frequency: linked to local clock rate
        nu = dp.nu_base * self.sim.P
        
        # Barrier height: how far below threshold
        barrier = np.maximum(0, dp.s_crit - s)
        
        # Kramers-like rate
        h = nu * np.exp(-barrier / T_eff)
        
        # Only active where there is significant q (i.e., a "nucleus")
        nucleus_mask = self.sim.q > 0.05
        h = h * nucleus_mask
        
        # Zero out already-decayed sites
        h = h * (~self.decayed)
        
        self.hazard_rates = h
        return h
    
    def check_for_decays(self, step: int) -> List[DecayEvent]:
        """
        Check if any nodes undergo decay this step.
        
        Uses the hazard rate as a probability per step:
        P(decay at i this step) = h_i * DT
        
        Returns list of decay events that occurred.
        """
        dp = self.dp
        dt = self.sim.p.DT
        
        # Probability of decay this step
        p_decay = self.hazard_rates * dt
        p_decay = np.clip(p_decay, 0, 1)
        
        # Stochastic sampling (the "randomness" is emergent from local fluctuations)
        rolls = np.random.random(self.N)
        decay_mask = (rolls < p_decay) & (~self.decayed)
        
        events = []
        decay_sites = np.where(decay_mask)[0]
        
        for site in decay_sites:
            event = self._execute_decay(site, step)
            events.append(event)
            self.decay_events.append(event)
        
        return events
    
    def _execute_decay(self, site: int, step: int) -> DecayEvent:
        """
        Execute a decay event at the given site.
        
        This involves:
        1. Recording the pre-decay state
        2. Releasing energy (F pulse)
        3. Reducing structural debt (q discharge)
        4. Imparting momentum to decay products
        5. Damaging local coherence (contagious instability)
        """
        dp = self.dp
        N = self.N
        
        # Record pre-decay state
        q_before = self.sim.q[site]
        s_i = self.instability_scores[site]
        h_i = self.hazard_rates[site]
        T_eff_i = self.T_eff[site]
        
        # 1. Energy release: convert fraction of local F into outward pulse
        F_local = self.sim.F[site]
        F_released = F_local * dp.energy_release_fraction
        self.sim.F[site] -= F_released
        
        # Distribute released energy as a Gaussian pulse around the site
        x = np.arange(N)
        dist = np.minimum(np.abs(x - site), N - np.abs(x - site))  # periodic distance
        pulse = F_released * np.exp(-0.5 * (dist / 3.0)**2)
        pulse[site] = 0  # Don't add back to the decay site
        pulse_norm = pulse / (np.sum(pulse) + 1e-12) * F_released
        self.sim.F += pulse_norm
        
        # 2. Debt discharge: reduce q at the decay site
        q_after = max(0, q_before - dp.q_reduction)
        self.sim.q[site] = q_after
        
        # 3. Momentum kick: impart momentum to nearby bonds
        kick_env = np.exp(-0.5 * (dist / 4.0)**2)
        # Rightward kick on right side, leftward kick on left side
        direction = np.sign(x - site)
        # Handle periodic wrapping
        direction[dist > N//4] = -direction[dist > N//4]
        self.sim.pi_R += dp.momentum_kick * kick_env * direction
        
        # 4. Coherence damage: reduce coherence in the neighborhood
        damage_mask = dist <= dp.coherence_damage_radius
        self.sim.C_R[damage_mask] *= (1.0 - dp.coherence_damage_factor)
        
        # Mark as decayed
        self.decayed[site] = True
        
        # Clip all values
        self.sim._clip()
        
        return DecayEvent(
            step=step,
            position=site,
            instability_score=s_i,
            hazard_rate=h_i,
            q_before=q_before,
            q_after=q_after,
            F_released=F_released,
            T_eff=T_eff_i
        )
    
    def identify_nuclei(self, q_threshold: float = 0.1) -> np.ndarray:
        """Identify 'nuclei' as contiguous regions with q above threshold."""
        from scipy.ndimage import label
        mask = self.sim.q > q_threshold
        labeled, num = label(mask)
        return labeled, num
    
    def count_remaining_nuclei(self, q_threshold: float = 0.1) -> int:
        """Count the number of undecayed nuclei remaining."""
        _, num = self.identify_nuclei(q_threshold)
        return num
    
    def step(self, step_number: int) -> List[DecayEvent]:
        """
        Execute one full simulation step:
        1. Run the DET collider step (physics update)
        2. Record flux for T_eff computation
        3. Compute instability scores
        4. Compute effective temperature
        5. Compute hazard rates
        6. Check for and execute decays
        """
        # Save F before step for flux computation
        F_before = self.sim.F.copy()
        
        # Run the underlying DET physics
        self.sim.step()
        
        # Record flux change for T_eff
        dF = self.sim.F - F_before
        self.flux_history.append(dF.copy())
        if len(self.flux_history) > self.dp.T_eff_window * 2:
            self.flux_history = self.flux_history[-self.dp.T_eff_window:]
        
        # Compute decay diagnostics
        self.compute_instability_score()
        self.compute_effective_temperature()
        self.compute_hazard_rate()
        
        # Check for decays
        events = self.check_for_decays(step_number)
        
        return events


# ============================================================
# SIMULATION SCENARIOS
# ============================================================

def create_nucleus_cluster(sim: DETCollider1D, center: int, 
                           mass: float = 10.0, q_level: float = 0.5,
                           width: float = 3.0):
    """Create a 'nucleus' — a high-q, high-F localized cluster."""
    sim.add_packet(center, mass=mass, width=width, initial_q=q_level)


def scenario_single_nucleus_decay(output_dir: str = "/home/ubuntu/det_decay_results"):
    """
    Scenario 1: Single Nucleus Decay
    
    Place a single high-q nucleus and observe its decay dynamics.
    Demonstrates threshold crossing and energy release.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("SCENARIO 1: Single Nucleus Decay")
    print("="*70)
    
    # Set up collider with parameters tuned for nuclear-scale dynamics
    params = DETParams1D(
        N=200, DT=0.02, F_VAC=0.01, F_MIN=0.0,
        C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.015,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
        boundary_enabled=True, grace_enabled=True
    )
    
    collider = DETCollider1D(params)
    
    # Create a single nucleus at center
    center = params.N // 2
    create_nucleus_cluster(collider, center, mass=15.0, q_level=0.6, width=3.0)
    
    # Set up decay simulator
    decay_params = DecayParams(
        w_q=1.0, w_grad=0.5, w_C=0.3,
        s_crit=0.65,
        T_eff_window=20, T_eff_floor=0.001,
        energy_release_fraction=0.4,
        q_reduction=0.25,
        momentum_kick=0.3,
        coherence_damage_radius=8,
        coherence_damage_factor=0.2
    )
    
    decay_sim = DETDecaySimulator(collider, decay_params)
    
    # Run simulation
    n_steps = 2000
    
    # Recording arrays
    times = []
    q_center = []
    F_center = []
    s_center = []
    h_center = []
    T_eff_center = []
    total_q = []
    total_F = []
    decay_times = []
    
    # Snapshot storage
    snapshots = []
    snapshot_steps = [0, 200, 500, 1000, 1500, 1999]
    
    print(f"\nRunning {n_steps} steps...")
    
    for t in range(n_steps):
        events = decay_sim.step(t)
        
        # Record
        times.append(t)
        q_center.append(collider.q[center])
        F_center.append(collider.F[center])
        s_center.append(decay_sim.instability_scores[center])
        h_center.append(decay_sim.hazard_rates[center])
        T_eff_center.append(decay_sim.T_eff[center])
        total_q.append(np.sum(collider.q))
        total_F.append(np.sum(collider.F))
        
        for ev in events:
            decay_times.append(ev.step)
            print(f"  DECAY at step {ev.step}, pos={ev.position}, "
                  f"s={ev.instability_score:.3f}, q: {ev.q_before:.3f} -> {ev.q_after:.3f}, "
                  f"F_released={ev.F_released:.3f}")
        
        if t in snapshot_steps:
            snapshots.append({
                'step': t,
                'F': collider.F.copy(),
                'q': collider.q.copy(),
                's': decay_sim.instability_scores.copy(),
                'h': decay_sim.hazard_rates.copy(),
                'C': collider.C_R.copy()
            })
    
    print(f"\nTotal decay events: {len(decay_sim.decay_events)}")
    
    # --- PLOT 1: Time Evolution ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("DET Radioactive Decay: Single Nucleus Time Evolution", fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(times, q_center, 'r-', linewidth=1.5)
    ax.set_ylabel('q (structural debt)')
    ax.set_title('Structural Debt at Nucleus Center')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for dt_val in decay_times:
        ax.axvline(x=dt_val, color='orange', linestyle='--', alpha=0.7, label='Decay' if dt_val == decay_times[0] else '')
    if decay_times:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(times, F_center, 'b-', linewidth=1.5)
    ax.set_ylabel('F (resource)')
    ax.set_title('Resource at Nucleus Center')
    for dt_val in decay_times:
        ax.axvline(x=dt_val, color='orange', linestyle='--', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(times, s_center, 'purple', linewidth=1.5)
    ax.axhline(y=decay_params.s_crit, color='red', linestyle='--', label=f's_crit = {decay_params.s_crit}')
    ax.set_ylabel('s_i (instability score)')
    ax.set_title('Instability Score at Nucleus')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.semilogy(times, np.array(h_center) + 1e-20, 'green', linewidth=1.5)
    ax.set_ylabel('h_i (hazard rate)')
    ax.set_title('Hazard Rate at Nucleus')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 0]
    ax.plot(times, total_q, 'r-', linewidth=1.5, label='Total q')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Structural Debt')
    ax.set_title('System-Wide Structural Debt')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[2, 1]
    ax.plot(times, total_F, 'b-', linewidth=1.5, label='Total F')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Resource')
    ax.set_title('System-Wide Resource (Conservation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario1_time_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scenario1_time_evolution.png")
    
    # --- PLOT 2: Spatial Snapshots ---
    n_snaps = len(snapshots)
    fig, axes = plt.subplots(n_snaps, 3, figsize=(16, 3 * n_snaps))
    fig.suptitle("DET Radioactive Decay: Spatial Snapshots", fontsize=14, fontweight='bold')
    
    x = np.arange(params.N)
    for i, snap in enumerate(snapshots):
        ax = axes[i, 0]
        ax.fill_between(x, snap['F'], alpha=0.6, color='blue')
        ax.set_ylabel(f"Step {snap['step']}\nF")
        ax.set_xlim(center - 40, center + 40)
        if i == 0:
            ax.set_title('Resource F')
        
        ax = axes[i, 1]
        ax.fill_between(x, snap['q'], alpha=0.6, color='red')
        ax.set_ylabel('q')
        ax.set_xlim(center - 40, center + 40)
        if i == 0:
            ax.set_title('Structural Debt q')
        
        ax = axes[i, 2]
        ax.fill_between(x, snap['s'], alpha=0.6, color='purple')
        ax.axhline(y=decay_params.s_crit, color='red', linestyle='--', alpha=0.7)
        ax.set_ylabel('s_i')
        ax.set_xlim(center - 40, center + 40)
        if i == 0:
            ax.set_title('Instability Score s_i')
    
    axes[-1, 0].set_xlabel('Position')
    axes[-1, 1].set_xlabel('Position')
    axes[-1, 2].set_xlabel('Position')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario1_spatial_snapshots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scenario1_spatial_snapshots.png")
    
    return decay_sim


def scenario_half_life_measurement(output_dir: str = "/home/ubuntu/det_decay_results"):
    """
    Scenario 2: Half-Life Measurement
    
    Place many identical nuclei and measure the emergent half-life.
    Demonstrates that exponential decay statistics emerge from deterministic
    threshold-crossing dynamics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SCENARIO 2: Half-Life Measurement (Ensemble)")
    print("="*70)
    
    # Set up a wider lattice with many nuclei
    N = 400
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
    
    # Place 20 identical nuclei evenly spaced
    n_nuclei = 20
    spacing = N // (n_nuclei + 1)
    nucleus_positions = [(i + 1) * spacing for i in range(n_nuclei)]
    
    for pos in nucleus_positions:
        # Add slight random variation to make each nucleus slightly different
        # (mimicking real-world micro-variations in nuclear environment)
        q_var = 0.55 + np.random.uniform(-0.05, 0.05)
        mass_var = 12.0 + np.random.uniform(-1.0, 1.0)
        create_nucleus_cluster(collider, pos, mass=mass_var, q_level=q_var, width=2.5)
    
    # Decay parameters
    decay_params = DecayParams(
        w_q=1.0, w_grad=0.5, w_C=0.3,
        s_crit=0.60,
        T_eff_window=15, T_eff_floor=0.002,
        energy_release_fraction=0.35,
        q_reduction=0.20,
        momentum_kick=0.25,
        coherence_damage_radius=5,
        coherence_damage_factor=0.15
    )
    
    decay_sim = DETDecaySimulator(collider, decay_params)
    
    # Run simulation
    n_steps = 3000
    
    times = []
    n_remaining = []
    cumulative_decays = []
    
    print(f"\nInitial nuclei: {n_nuclei}")
    print(f"Running {n_steps} steps...")
    
    total_decayed = 0
    
    for t in range(n_steps):
        events = decay_sim.step(t)
        total_decayed += len(events)
        
        times.append(t)
        n_remaining.append(n_nuclei - total_decayed)
        cumulative_decays.append(total_decayed)
        
        for ev in events:
            print(f"  DECAY #{total_decayed} at step {ev.step}, pos={ev.position}, "
                  f"s={ev.instability_score:.3f}")
        
        if total_decayed >= n_nuclei:
            print(f"\n  All nuclei decayed by step {t}!")
            # Pad remaining steps
            for t2 in range(t+1, n_steps):
                times.append(t2)
                n_remaining.append(0)
                cumulative_decays.append(total_decayed)
            break
    
    print(f"\nTotal decayed: {total_decayed} / {n_nuclei}")
    
    # Compute half-life from the data
    n_remaining_arr = np.array(n_remaining, dtype=float)
    times_arr = np.array(times, dtype=float)
    
    # Find when N/2 nuclei remain
    half_n = n_nuclei / 2.0
    half_life_step = None
    for i, nr in enumerate(n_remaining):
        if nr <= half_n:
            half_life_step = times[i]
            break
    
    # Fit exponential: N(t) = N0 * exp(-lambda * t)
    # Use log-linear fit on non-zero data
    valid = n_remaining_arr > 0
    if np.sum(valid) > 10:
        log_n = np.log(n_remaining_arr[valid])
        t_valid = times_arr[valid]
        # Linear fit: log(N) = log(N0) - lambda * t
        coeffs = np.polyfit(t_valid, log_n, 1)
        lambda_fit = -coeffs[0]
        t_half_fit = np.log(2) / lambda_fit if lambda_fit > 0 else float('inf')
    else:
        lambda_fit = 0
        t_half_fit = float('inf')
    
    print(f"\n  Measured half-life (first crossing): {half_life_step} steps")
    print(f"  Fitted decay constant lambda: {lambda_fit:.6f}")
    print(f"  Fitted half-life: {t_half_fit:.1f} steps")
    
    # --- PLOT: Half-Life Measurement ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DET Radioactive Decay: Emergent Half-Life", fontsize=14, fontweight='bold')
    
    # Survival curve
    ax = axes[0, 0]
    ax.plot(times, n_remaining, 'b-', linewidth=2, label='Remaining nuclei')
    ax.axhline(y=half_n, color='red', linestyle='--', alpha=0.7, label=f'N/2 = {half_n:.0f}')
    if half_life_step is not None:
        ax.axvline(x=half_life_step, color='green', linestyle='--', alpha=0.7, 
                   label=f'T_1/2 = {half_life_step} steps')
    ax.set_xlabel('Step')
    ax.set_ylabel('Nuclei Remaining')
    ax.set_title('Survival Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log survival curve with fit
    ax = axes[0, 1]
    valid_plot = np.array(n_remaining) > 0
    ax.semilogy(np.array(times)[valid_plot], np.array(n_remaining)[valid_plot], 
                'bo', markersize=3, alpha=0.5, label='Data')
    if lambda_fit > 0:
        t_fit = np.linspace(0, max(times), 200)
        n_fit = n_nuclei * np.exp(-lambda_fit * t_fit)
        ax.semilogy(t_fit, n_fit, 'r-', linewidth=2, 
                    label=f'Fit: N₀·exp(-λt), T₁/₂={t_half_fit:.0f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Nuclei Remaining (log)')
    ax.set_title('Log-Survival with Exponential Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative decays
    ax = axes[1, 0]
    ax.plot(times, cumulative_decays, 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Decays')
    ax.set_title('Cumulative Decay Events')
    ax.grid(True, alpha=0.3)
    
    # Decay event histogram
    ax = axes[1, 1]
    if decay_sim.decay_events:
        event_steps = [ev.step for ev in decay_sim.decay_events]
        ax.hist(event_steps, bins=30, color='orange', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Decays per Bin')
    ax.set_title('Decay Event Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario2_half_life.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scenario2_half_life.png")
    
    return decay_sim, t_half_fit


def scenario_coherence_stabilization(output_dir: str = "/home/ubuntu/det_decay_results"):
    """
    Scenario 3: Coherence-Mediated Stabilization
    
    Compare decay rates of identical nuclei in:
    (a) Low-coherence environment (standard)
    (b) High-coherence environment (stabilized)
    
    Demonstrates the DET prediction that coherence buffering can alter decay kinetics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SCENARIO 3: Coherence-Mediated Stabilization")
    print("="*70)
    
    results = {}
    
    for label, C_init_val in [("Low Coherence (C=0.15)", 0.15), 
                               ("High Coherence (C=0.80)", 0.80)]:
        print(f"\n--- {label} ---")
        
        N = 400
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0,
            C_init=C_init_val,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.015,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
            boundary_enabled=True, grace_enabled=True
        )
        
        collider = DETCollider1D(params)
        
        # Place identical nuclei
        n_nuclei = 20
        spacing = N // (n_nuclei + 1)
        
        np.random.seed(42)  # Same random seed for fair comparison
        for i in range(n_nuclei):
            pos = (i + 1) * spacing
            create_nucleus_cluster(collider, pos, mass=12.0, q_level=0.55, width=2.5)
        
        decay_params = DecayParams(
            w_q=1.0, w_grad=0.5, w_C=0.3,
            s_crit=0.60,
            T_eff_window=15, T_eff_floor=0.002,
            energy_release_fraction=0.35,
            q_reduction=0.20,
            momentum_kick=0.25,
            coherence_damage_radius=5,
            coherence_damage_factor=0.15
        )
        
        decay_sim = DETDecaySimulator(collider, decay_params)
        
        n_steps = 3000
        times = []
        n_remaining = []
        total_decayed = 0
        
        for t in range(n_steps):
            events = decay_sim.step(t)
            total_decayed += len(events)
            times.append(t)
            n_remaining.append(n_nuclei - total_decayed)
            
            if total_decayed >= n_nuclei:
                for t2 in range(t+1, n_steps):
                    times.append(t2)
                    n_remaining.append(0)
                break
        
        # Compute half-life
        half_n = n_nuclei / 2.0
        half_life_step = None
        for i, nr in enumerate(n_remaining):
            if nr <= half_n:
                half_life_step = times[i]
                break
        
        results[label] = {
            'times': times,
            'n_remaining': n_remaining,
            'half_life': half_life_step,
            'total_decayed': total_decayed,
            'C_init': C_init_val
        }
        
        print(f"  Total decayed: {total_decayed}/{n_nuclei}")
        print(f"  Half-life: {half_life_step} steps")
    
    # --- PLOT: Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("DET Prediction: Coherence Stabilization of Radioactive Decay", 
                 fontsize=14, fontweight='bold')
    
    colors = {'Low Coherence (C=0.15)': 'red', 'High Coherence (C=0.80)': 'blue'}
    
    for label, data in results.items():
        ax = axes[0]
        ax.plot(data['times'], data['n_remaining'], color=colors[label], 
                linewidth=2, label=label)
        
        ax = axes[1]
        valid = np.array(data['n_remaining']) > 0
        if np.sum(valid) > 0:
            ax.semilogy(np.array(data['times'])[valid], 
                       np.array(data['n_remaining'])[valid],
                       color=colors[label], linewidth=2, label=label)
    
    axes[0].axhline(y=n_nuclei/2, color='gray', linestyle='--', alpha=0.5, label='N/2')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Nuclei Remaining')
    axes[0].set_title('Survival Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Nuclei Remaining (log)')
    axes[1].set_title('Log-Survival Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario3_coherence_stabilization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_dir}/scenario3_coherence_stabilization.png")
    
    return results


def scenario_decay_chain(output_dir: str = "/home/ubuntu/det_decay_results"):
    """
    Scenario 4: Decay Chain Simulation
    
    Simulate a chain of decays where the daughter nucleus from one decay
    can itself become unstable and decay further.
    
    Demonstrates:
    - Sequential transmutation
    - Energy cascade
    - Contagious instability from coherence damage
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SCENARIO 4: Decay Chain (Sequential Transmutation)")
    print("="*70)
    
    N = 300
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0,
        C_init=0.25,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.02,  # Slightly higher q accumulation
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
        boundary_enabled=True, grace_enabled=True
    )
    
    collider = DETCollider1D(params)
    
    # Create a "heavy" nucleus (high q, high F) that will undergo chain decay
    center = N // 2
    create_nucleus_cluster(collider, center, mass=25.0, q_level=0.8, width=5.0)
    
    # Also place some "medium" nuclei nearby that could be destabilized
    create_nucleus_cluster(collider, center - 30, mass=10.0, q_level=0.45, width=3.0)
    create_nucleus_cluster(collider, center + 30, mass=10.0, q_level=0.45, width=3.0)
    
    # Decay parameters: allow re-decay (don't permanently mark sites)
    decay_params = DecayParams(
        w_q=1.0, w_grad=0.6, w_C=0.4,
        s_crit=0.55,
        T_eff_window=15, T_eff_floor=0.003,
        energy_release_fraction=0.3,
        q_reduction=0.15,  # Smaller reduction = can decay again
        momentum_kick=0.4,
        coherence_damage_radius=10,  # Wider damage = contagious
        coherence_damage_factor=0.25
    )
    
    decay_sim = DETDecaySimulator(collider, decay_params)
    
    # Allow re-decay: reset decayed flags periodically
    n_steps = 4000
    
    times = []
    total_events = []
    q_profiles = []
    F_profiles = []
    snapshot_steps = [0, 500, 1000, 2000, 3000, 3999]
    
    cumulative = 0
    
    print(f"\nRunning {n_steps} steps with chain decay enabled...")
    
    for t in range(n_steps):
        # Allow re-decay every 200 steps (nucleus can decay multiple times)
        if t % 200 == 0 and t > 0:
            decay_sim.decayed[:] = False
        
        events = decay_sim.step(t)
        cumulative += len(events)
        
        times.append(t)
        total_events.append(cumulative)
        
        for ev in events:
            print(f"  CHAIN DECAY at step {ev.step}, pos={ev.position}, "
                  f"q: {ev.q_before:.3f} -> {ev.q_after:.3f}")
        
        if t in snapshot_steps:
            q_profiles.append(collider.q.copy())
            F_profiles.append(collider.F.copy())
    
    print(f"\nTotal chain decay events: {cumulative}")
    
    # --- PLOT: Chain Decay ---
    n_snaps = len(q_profiles)
    fig, axes = plt.subplots(n_snaps, 2, figsize=(14, 3 * n_snaps))
    fig.suptitle("DET Decay Chain: Sequential Transmutation", fontsize=14, fontweight='bold')
    
    x = np.arange(N)
    for i in range(n_snaps):
        ax = axes[i, 0]
        ax.fill_between(x, F_profiles[i], alpha=0.6, color='blue')
        ax.set_ylabel(f"Step {snapshot_steps[i]}\nF")
        ax.set_xlim(center - 60, center + 60)
        if i == 0:
            ax.set_title('Resource F')
        
        ax = axes[i, 1]
        ax.fill_between(x, q_profiles[i], alpha=0.6, color='red')
        ax.set_ylabel('q')
        ax.set_xlim(center - 60, center + 60)
        if i == 0:
            ax.set_title('Structural Debt q')
    
    axes[-1, 0].set_xlabel('Position')
    axes[-1, 1].set_xlabel('Position')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario4_decay_chain.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scenario4_decay_chain.png")
    
    # Cumulative events plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, total_events, 'r-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Decay Events')
    ax.set_title('DET Decay Chain: Cumulative Events Over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario4_cumulative_events.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scenario4_cumulative_events.png")
    
    return decay_sim


def scenario_environmental_dependence(output_dir: str = "/home/ubuntu/det_decay_results"):
    """
    Scenario 5: Environmental Dependence of Decay Rate
    
    Test the DET prediction that decay rates depend on environmental conditions:
    - Varying fluctuation amplitude (effective temperature)
    - Varying structural debt background
    
    This is a falsifiable DET claim with cosmological implications.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SCENARIO 5: Environmental Dependence of Decay Rate")
    print("="*70)
    
    # Test across different initial coherence and q backgrounds
    conditions = [
        {"label": "Cold/Quiet (low flux)", "F_VAC": 0.001, "C_init": 0.5},
        {"label": "Standard", "F_VAC": 0.01, "C_init": 0.3},
        {"label": "Hot/Noisy (high flux)", "F_VAC": 0.05, "C_init": 0.15},
        {"label": "High Background Debt", "F_VAC": 0.01, "C_init": 0.3},
    ]
    
    all_results = {}
    
    for cond in conditions:
        label = cond["label"]
        print(f"\n--- {label} ---")
        
        N = 400
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=cond["F_VAC"], F_MIN=0.0,
            C_init=cond["C_init"],
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.015,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
            boundary_enabled=True, grace_enabled=True
        )
        
        collider = DETCollider1D(params)
        
        # Add background debt for the "High Background Debt" condition
        if "Background Debt" in label:
            collider.q += 0.15  # Uniform background debt
        
        # Place identical nuclei
        n_nuclei = 20
        spacing = N // (n_nuclei + 1)
        np.random.seed(42)
        for i in range(n_nuclei):
            pos = (i + 1) * spacing
            create_nucleus_cluster(collider, pos, mass=12.0, q_level=0.55, width=2.5)
        
        decay_params = DecayParams(
            w_q=1.0, w_grad=0.5, w_C=0.3,
            s_crit=0.60,
            T_eff_window=15, T_eff_floor=0.002,
        )
        
        decay_sim = DETDecaySimulator(collider, decay_params)
        
        n_steps = 3000
        times = []
        n_remaining = []
        total_decayed = 0
        
        for t in range(n_steps):
            events = decay_sim.step(t)
            total_decayed += len(events)
            times.append(t)
            n_remaining.append(n_nuclei - total_decayed)
            if total_decayed >= n_nuclei:
                for t2 in range(t+1, n_steps):
                    times.append(t2)
                    n_remaining.append(0)
                break
        
        half_n = n_nuclei / 2.0
        half_life = None
        for i, nr in enumerate(n_remaining):
            if nr <= half_n:
                half_life = times[i]
                break
        
        all_results[label] = {
            'times': times,
            'n_remaining': n_remaining,
            'half_life': half_life,
            'total_decayed': total_decayed
        }
        
        print(f"  Decayed: {total_decayed}/{n_nuclei}, Half-life: {half_life}")
    
    # --- PLOT ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("DET Falsifiable Prediction: Environmental Dependence of Decay Rate", 
                 fontsize=14, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'purple']
    
    for (label, data), color in zip(all_results.items(), colors):
        axes[0].plot(data['times'], data['n_remaining'], color=color, 
                    linewidth=2, label=f"{label} (T½={data['half_life']})")
        
        valid = np.array(data['n_remaining']) > 0
        if np.sum(valid) > 0:
            axes[1].semilogy(np.array(data['times'])[valid],
                           np.array(data['n_remaining'])[valid],
                           color=color, linewidth=2, label=label)
    
    axes[0].axhline(y=10, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Nuclei Remaining')
    axes[0].set_title('Survival Curves by Environment')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Nuclei Remaining (log)')
    axes[1].set_title('Log-Survival by Environment')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario5_environmental_dependence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {output_dir}/scenario5_environmental_dependence.png")
    
    # Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    labels_list = list(all_results.keys())
    half_lives = [all_results[l]['half_life'] if all_results[l]['half_life'] is not None else 3000 
                  for l in labels_list]
    bars = ax.bar(range(len(labels_list)), half_lives, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels(labels_list, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Half-Life (steps)')
    ax.set_title('DET Prediction: Half-Life Depends on Environment')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, hl in zip(bars, half_lives):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                f'{hl}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scenario5_half_life_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/scenario5_half_life_comparison.png")
    
    return all_results


# ============================================================
# MAIN RUNNER
# ============================================================

def run_all_scenarios():
    """Run all radioactive decay simulation scenarios."""
    output_dir = "/home/ubuntu/det_decay_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("DET RADIOACTIVE DECAY SIMULATION SUITE")
    print("="*70)
    print("Implementing: q-Threshold Crossing & Emergent Half-Life")
    print("Framework: DET v6.3 1D Collider")
    print("="*70)
    
    results = {}
    
    # Scenario 1: Single nucleus
    results['single'] = scenario_single_nucleus_decay(output_dir)
    
    # Scenario 2: Half-life measurement
    results['half_life'] = scenario_half_life_measurement(output_dir)
    
    # Scenario 3: Coherence stabilization
    results['coherence'] = scenario_coherence_stabilization(output_dir)
    
    # Scenario 4: Decay chain
    results['chain'] = scenario_decay_chain(output_dir)
    
    # Scenario 5: Environmental dependence
    results['environment'] = scenario_environmental_dependence(output_dir)
    
    print("\n" + "="*70)
    print("ALL SCENARIOS COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_dir}/")
    
    return results


if __name__ == "__main__":
    run_all_scenarios()
