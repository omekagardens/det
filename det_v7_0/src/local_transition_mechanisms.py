"""
DET Local Transition Mechanisms: W → K via Strictly-Local Operations
=====================================================================

DET is strictly local — no global state, no nonlocal coupling. So the
question "what lifts a W-regime into K?" must be answered using only
DET's own local operators:

  L1. GRACE FLUX — The boundary operator injects F into depleted nodes,
      gated by agency. Does this alone raise K?

  L2. BOND HEALING — Agency-gated coherence repair on bonds with
      dissipation. Does healing C locally push K up?

  L3. ATTUNEMENT FEEDBACK — Mutual observation reinforces C on bonds.
      Does attunement from a nearby K-region lift adjacent W-nodes?

  L4. SELECTIVE OBSERVATION — A W-node that "chooses" to observe
      (orient toward) K-regime neighbors. Modeled as biased attunement
      where the node preferentially attunes to high-K neighbors.

  L5. OBSERVATION LIMITING — A W-node that limits its observation of
      other W-regime nodes (reduces attunement to low-K neighbors).
      Does cutting W-W feedback loops accelerate transition?

  L6. K-REGIME PROXIMITY — A W-region adjacent to a K-region. Does
      the K-region's natural dynamics (high C, low q, grace) "leak"
      across the boundary and lift the W-region?

  L7. COMBINED STRATEGY — Grace + Healing + Selective Observation.
      What is the optimal combination?

All mechanisms are strictly local per the DET theory card.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from det_v6_3_1d_collider import DETCollider1D, DETParams1D
from det_concurrent_regimes import (
    DETRegimeSimulator, RegimeParams, RegimeDiagnostics,
    create_k_region, create_w_region
)

OUTPUT_DIR = "/home/ubuntu/det_app_results/local_mechanisms"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SELECTIVE OBSERVATION SIMULATOR (extends DETRegimeSimulator)
# ============================================================

class SelectiveObservationSimulator(DETRegimeSimulator):
    """
    Extended regime simulator that models SELECTIVE OBSERVATION:
    
    A node can bias its attunement toward high-K neighbors (choosing
    to observe the K-regime) and/or away from low-K neighbors
    (limiting observation of the W-regime).
    
    This is modeled as a modified attunement feedback where the
    coupling strength depends on the neighbor's regime index.
    
    Strictly local: each node only sees its immediate neighbors.
    """
    
    def __init__(self, collider, regime_params=None,
                 selective_mode='none',
                 selectivity_strength=1.0,
                 selective_mask=None):
        """
        selective_mode:
          'none'     — standard attunement (no bias)
          'seek_K'   — preferentially attune to high-K neighbors
          'avoid_W'  — reduce attunement to low-K neighbors
          'both'     — seek K AND avoid W simultaneously
        
        selectivity_strength: how strongly the bias operates (0=no bias, 1=full)
        selective_mask: boolean array, True for nodes that use selective observation
        """
        super().__init__(collider, regime_params)
        self.selective_mode = selective_mode
        self.selectivity_strength = selectivity_strength
        if selective_mask is None:
            self.selective_mask = np.ones(self.N, dtype=bool)
        else:
            self.selective_mask = selective_mask
    
    def apply_attunement_feedback(self):
        """
        Modified attunement with selective observation.
        
        Standard: ΔC_ij = η * √(O_i * O_j) * (Ξ_i^seen * Ξ_j^seen) * Δτ_ij
        
        Selective: ΔC_ij = η * √(O_i * O_j) * (Ξ_i^seen * Ξ_j^seen) * Δτ_ij * w_ij
        
        where w_ij is a weighting factor based on the neighbor's K:
          seek_K:  w_ij = K_j^s  (high-K neighbors get more attunement)
          avoid_W: w_ij = 1 - (1-K_j)^s  (low-K neighbors get less)
          both:    w_ij = K_j^s * [1 - (1-K_j)^s]  (combined)
        """
        if not self.rp.attunement_enabled:
            return
        
        rp = self.rp
        R = lambda x: np.roll(x, -1)
        s = self.selectivity_strength
        
        # Mutual observability on right bond
        O_mutual = np.sqrt(self.O * R(self.O))
        
        # Mutual perceived structuredness
        Xi_mutual = self.Xi_seen * R(self.Xi_seen)
        
        # Bond-local proper time
        Delta_tau_R = 0.5 * (self.sim.Delta_tau + R(self.sim.Delta_tau))
        
        # Base attunement increment
        dC_base = rp.eta_attune * O_mutual * Xi_mutual * Delta_tau_R
        
        # Compute selective weighting
        K_neighbor = R(self.K)  # right neighbor's K
        
        if self.selective_mode == 'none':
            w = np.ones(self.N)
        elif self.selective_mode == 'seek_K':
            # Upweight high-K neighbors: w = K_j^s (0 for W, 1 for K)
            w = np.where(self.selective_mask, K_neighbor ** s, 1.0)
        elif self.selective_mode == 'avoid_W':
            # Downweight low-K neighbors: w = 1 - (1-K_j)^s
            w = np.where(self.selective_mask, 1.0 - (1.0 - K_neighbor) ** s, 1.0)
        elif self.selective_mode == 'both':
            # Combined: strongly prefer K, strongly avoid W
            w_seek = K_neighbor ** s
            w_avoid = 1.0 - (1.0 - K_neighbor) ** s
            w = np.where(self.selective_mask, w_seek * w_avoid, 1.0)
        else:
            w = np.ones(self.N)
        
        # Apply weighted attunement
        dC_attune = dC_base * w
        self.sim.C_R = np.clip(self.sim.C_R + dC_attune, 0, 1)


# ============================================================
# HELPER: Create a mixed K/W lattice
# ============================================================

def create_mixed_lattice(N=200, K_center=50, K_width=30,
                         W_C=0.15, W_q=0.50, W_F=0.5,
                         K_C=0.90, K_q=0.02, K_F=3.0,
                         seed=42):
    """Create a lattice with a K-region on the left and W-region everywhere else."""
    np.random.seed(seed)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=W_C,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.003,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.03
    )
    collider = DETCollider1D(params)
    
    # Set W-regime everywhere
    collider.C_R[:] = W_C
    collider.q[:] = W_q
    collider.F[:] = W_F
    collider.a[:] = 0.5
    
    # Set K-region
    x = np.arange(N)
    dist = np.abs(x - K_center)
    K_mask = dist < K_width
    collider.C_R[K_mask] = K_C
    collider.q[K_mask] = K_q
    collider.F[K_mask] = K_F
    collider.a[K_mask] = 0.95
    
    return collider, params, K_mask


# ============================================================
# L1: GRACE FLUX ALONE
# ============================================================

def experiment_L1_grace_flux():
    """
    L1: Can grace flux alone lift a W-regime into K?
    
    Grace injects F into depleted nodes. F affects Presence (P),
    which feeds into K_i. But grace does NOT directly change C or q.
    
    Compare: grace enabled vs disabled.
    """
    print("\n" + "="*70)
    print("L1: Grace Flux Alone — Can It Lift W→K?")
    print("="*70)
    
    N = 200
    conditions = {
        'No Grace': {'grace': False, 'healing': False},
        'Grace Only': {'grace': True, 'healing': False},
        'Grace + Healing': {'grace': True, 'healing': True},
    }
    
    results = {}
    n_steps = 3000
    
    for name, cfg in conditions.items():
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.15,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False,
            boundary_enabled=True,
            grace_enabled=cfg['grace'],
            F_MIN_grace=0.05,
            healing_enabled=cfg['healing'],
            eta_heal=0.03
        )
        collider = DETCollider1D(params)
        collider.C_R[:] = 0.15
        collider.q[:] = 0.50
        collider.F[:] = 0.5
        collider.a[:] = 0.5
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams())
        
        K_trace = []
        C_trace = []
        q_trace = []
        F_trace = []
        
        for t in range(n_steps):
            diag = regime.step(t)
            K_trace.append(float(np.mean(diag.K)))
            C_trace.append(float(np.mean(collider.C_R)))
            q_trace.append(float(np.mean(collider.q)))
            F_trace.append(float(np.mean(collider.F)))
        
        results[name] = {
            'K': np.array(K_trace),
            'C': np.array(C_trace),
            'q': np.array(q_trace),
            'F': np.array(F_trace)
        }
        print(f"  {name:20s}: final K={K_trace[-1]:.4f}, C={C_trace[-1]:.4f}, "
              f"q={q_trace[-1]:.4f}, F={F_trace[-1]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("L1: Grace Flux Alone — Can It Lift W→K?", fontsize=14, fontweight='bold')
    
    colors = {'No Grace': 'red', 'Grace Only': 'blue', 'Grace + Healing': 'green'}
    
    for name, data in results.items():
        c = colors[name]
        axes[0,0].plot(data['K'], color=c, linewidth=2, label=name)
        axes[0,1].plot(data['C'], color=c, linewidth=2, label=name)
        axes[1,0].plot(data['q'], color=c, linewidth=2, label=name)
        axes[1,1].plot(data['F'], color=c, linewidth=2, label=name)
    
    for ax, title, ylabel in zip(axes.flat,
        ['Regime Index K', 'Coherence C', 'Structural Debt q', 'Resource F'],
        ['K_i', 'C', 'q', 'F']):
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[0,0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/L1_grace_flux.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/L1_grace_flux.png")
    return results


# ============================================================
# L2: K-REGIME PROXIMITY — DOES K "LEAK" INTO W?
# ============================================================

def experiment_L2_proximity():
    """
    L2: K-Regime Proximity
    
    Place a K-region next to a W-region. Does the K-region's natural
    dynamics (high C, low q, grace, healing) leak across the boundary
    and lift the adjacent W-nodes?
    
    Measure K(x) profile over time to see if the K-boundary advances.
    """
    print("\n" + "="*70)
    print("L2: K-Regime Proximity — Does K Leak Into W?")
    print("="*70)
    
    N = 200
    K_center = 50
    K_width = 30
    
    configs = {
        'No boundary ops': {'grace': False, 'healing': False},
        'Grace only': {'grace': True, 'healing': False},
        'Grace + Healing': {'grace': True, 'healing': True},
    }
    
    results = {}
    n_steps = 3000
    snapshot_steps = [0, 500, 1000, 2000, 3000]
    
    for name, cfg in configs.items():
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.15,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False,
            boundary_enabled=True,
            grace_enabled=cfg['grace'],
            F_MIN_grace=0.05,
            healing_enabled=cfg['healing'],
            eta_heal=0.03
        )
        collider = DETCollider1D(params)
        
        # W-regime everywhere
        collider.C_R[:] = 0.15
        collider.q[:] = 0.50
        collider.F[:] = 0.5
        collider.a[:] = 0.5
        
        # K-region
        x = np.arange(N)
        K_mask = np.abs(x - K_center) < K_width
        collider.C_R[K_mask] = 0.90
        collider.q[K_mask] = 0.02
        collider.F[K_mask] = 3.0
        collider.a[K_mask] = 0.95
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams(
            attunement_enabled=True, eta_attune=0.05
        ))
        
        snapshots = {}
        K_trace_W = []  # mean K in W-region only
        C_trace_W = []
        
        W_mask = ~K_mask
        
        for t in range(n_steps + 1):
            diag = regime.step(t)
            K_trace_W.append(float(np.mean(diag.K[W_mask])))
            C_trace_W.append(float(np.mean(collider.C_R[W_mask])))
            
            if t in snapshot_steps:
                snapshots[t] = {
                    'K': diag.K.copy(),
                    'C': collider.C_R.copy(),
                    'q': collider.q.copy(),
                    'O': diag.O.copy()
                }
        
        results[name] = {
            'K_W': np.array(K_trace_W),
            'C_W': np.array(C_trace_W),
            'snapshots': snapshots
        }
        print(f"  {name:20s}: W-region final K={K_trace_W[-1]:.4f}, C={C_trace_W[-1]:.4f}")
    
    # Plot
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("L2: K-Regime Proximity — Does K Leak Into W?", fontsize=14, fontweight='bold')
    
    colors = {'No boundary ops': 'red', 'Grace only': 'blue', 'Grace + Healing': 'green'}
    
    # Panel 1: K in W-region over time
    ax = fig.add_subplot(gs[0, 0])
    for name, data in results.items():
        ax.plot(data['K_W'], color=colors[name], linewidth=2, label=name)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean K (W-region)')
    ax.set_title('K in W-Region Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: C in W-region over time
    ax = fig.add_subplot(gs[0, 1])
    for name, data in results.items():
        ax.plot(data['C_W'], color=colors[name], linewidth=2, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean C (W-region)')
    ax.set_title('C in W-Region Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panels 3-8: Spatial snapshots for Grace + Healing
    best = results['Grace + Healing']
    for idx, t in enumerate(snapshot_steps):
        if t not in best['snapshots']:
            continue
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        snap = best['snapshots'][t]
        ax.plot(snap['K'], 'g-', linewidth=1.5, label='K')
        ax.plot(snap['C'], 'b-', linewidth=1.5, label='C')
        ax.plot(snap['q'], 'r-', linewidth=1.5, label='q')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.axvspan(K_center - K_width, K_center + K_width, alpha=0.1, color='green')
        ax.set_xlabel('Node')
        ax.set_ylabel('Value')
        ax.set_title(f'Spatial Profile at t={t}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.savefig(f"{OUTPUT_DIR}/L2_proximity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/L2_proximity.png")
    return results


# ============================================================
# L3: SELECTIVE OBSERVATION — CHOOSING TO OBSERVE K
# ============================================================

def experiment_L3_selective_observation():
    """
    L3: Selective Observation
    
    A W-node that "chooses" to observe K-regime neighbors gets
    preferential attunement from those neighbors. Compare:
    
    a) No attunement (baseline)
    b) Standard attunement (unbiased)
    c) Seek-K attunement (bias toward high-K neighbors)
    d) Avoid-W attunement (reduce coupling to low-K neighbors)
    e) Both (seek K + avoid W)
    
    Setup: K-region on left, W-region on right. W-nodes near the
    boundary use selective observation.
    """
    print("\n" + "="*70)
    print("L3: Selective Observation — Choosing to Observe K")
    print("="*70)
    
    N = 200
    K_center = 50
    K_width = 30
    
    modes = {
        'No Attunement': ('none', 0.0, False),
        'Standard Attunement': ('none', 0.0, True),
        'Seek K': ('seek_K', 2.0, True),
        'Avoid W': ('avoid_W', 2.0, True),
        'Seek K + Avoid W': ('both', 2.0, True),
    }
    
    results = {}
    n_steps = 3000
    
    for name, (mode, strength, attune_on) in modes.items():
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.15,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False,
            boundary_enabled=True, grace_enabled=True,
            F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.03
        )
        collider = DETCollider1D(params)
        
        # W-regime everywhere
        collider.C_R[:] = 0.15
        collider.q[:] = 0.50
        collider.F[:] = 0.5
        collider.a[:] = 0.5
        
        # K-region
        x = np.arange(N)
        K_mask = np.abs(x - K_center) < K_width
        collider.C_R[K_mask] = 0.90
        collider.q[K_mask] = 0.02
        collider.F[K_mask] = 3.0
        collider.a[K_mask] = 0.95
        
        # Selective observation mask: W-nodes only
        W_mask = ~K_mask
        
        np.random.seed(42)
        rp = RegimeParams(attunement_enabled=attune_on, eta_attune=0.10)
        regime = SelectiveObservationSimulator(
            collider, rp,
            selective_mode=mode,
            selectivity_strength=strength,
            selective_mask=W_mask
        )
        
        K_trace_W = []
        C_trace_W = []
        K_trace_boundary = []  # nodes near the K/W boundary
        
        # Boundary zone: W-nodes within 10 of K-region edge
        boundary_mask = W_mask & (np.abs(x - K_center) < K_width + 10)
        
        for t in range(n_steps):
            diag = regime.step(t)
            K_trace_W.append(float(np.mean(diag.K[W_mask])))
            C_trace_W.append(float(np.mean(collider.C_R[W_mask])))
            if np.any(boundary_mask):
                K_trace_boundary.append(float(np.mean(diag.K[boundary_mask])))
            else:
                K_trace_boundary.append(0.0)
        
        results[name] = {
            'K_W': np.array(K_trace_W),
            'C_W': np.array(C_trace_W),
            'K_boundary': np.array(K_trace_boundary),
            'final_K_profile': diag.K.copy(),
            'final_C_profile': collider.C_R.copy()
        }
        print(f"  {name:25s}: W-region K={K_trace_W[-1]:.4f}, "
              f"boundary K={K_trace_boundary[-1]:.4f}, C_W={C_trace_W[-1]:.4f}")
    
    # Plot
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("L3: Selective Observation — Choosing to Observe K",
                 fontsize=14, fontweight='bold')
    
    colors = {
        'No Attunement': 'gray',
        'Standard Attunement': 'blue',
        'Seek K': 'green',
        'Avoid W': 'orange',
        'Seek K + Avoid W': 'red',
    }
    
    # Panel 1: K in W-region
    ax = fig.add_subplot(gs[0, 0])
    for name, data in results.items():
        ax.plot(data['K_W'], color=colors[name], linewidth=2, label=name)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean K (W-region)')
    ax.set_title('K in W-Region Over Time')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: K at boundary
    ax = fig.add_subplot(gs[0, 1])
    for name, data in results.items():
        ax.plot(data['K_boundary'], color=colors[name], linewidth=2, label=name)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean K (boundary zone)')
    ax.set_title('K at K/W Boundary Over Time')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: C in W-region
    ax = fig.add_subplot(gs[1, 0])
    for name, data in results.items():
        ax.plot(data['C_W'], color=colors[name], linewidth=2, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean C (W-region)')
    ax.set_title('Coherence in W-Region Over Time')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 4: Final K profiles
    ax = fig.add_subplot(gs[1, 1])
    for name, data in results.items():
        ax.plot(data['final_K_profile'], color=colors[name], linewidth=1.5, label=name, alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(K_center - K_width, K_center + K_width, alpha=0.1, color='green')
    ax.set_xlabel('Node')
    ax.set_ylabel('K_i')
    ax.set_title('Final K Profile (t=3000)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Final C profiles
    ax = fig.add_subplot(gs[2, 0])
    for name, data in results.items():
        ax.plot(data['final_C_profile'], color=colors[name], linewidth=1.5, label=name, alpha=0.8)
    ax.axvspan(K_center - K_width, K_center + K_width, alpha=0.1, color='green')
    ax.set_xlabel('Node')
    ax.set_ylabel('C')
    ax.set_title('Final Coherence Profile (t=3000)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Summary comparison
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')
    summary = "SELECTIVE OBSERVATION SUMMARY\n\n"
    for name, data in results.items():
        delta_K = data['K_W'][-1] - data['K_W'][0]
        delta_C = data['C_W'][-1] - data['C_W'][0]
        summary += f"{name:25s}: ΔK={delta_K:+.4f}, ΔC={delta_C:+.4f}\n"
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(f"{OUTPUT_DIR}/L3_selective_observation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/L3_selective_observation.png")
    return results


# ============================================================
# L4: PURE W-REGIME — INTERNAL SELECTIVE OBSERVATION
# ============================================================

def experiment_L4_pure_W_internal():
    """
    L4: Pure W-Regime Internal Dynamics
    
    No K-region at all. Can a subset of W-nodes that "choose" to
    observe each other more carefully (higher mutual attunement)
    bootstrap themselves into K?
    
    This tests whether agency-driven selective observation within
    a purely W-regime system can create a local K-island.
    """
    print("\n" + "="*70)
    print("L4: Pure W-Regime — Can Internal Observation Bootstrap K?")
    print("="*70)
    
    N = 200
    
    # A "community" of nodes that choose to observe each other
    community_center = 100
    community_width = 20
    x = np.arange(N)
    community_mask = np.abs(x - community_center) < community_width
    
    configs = {
        'No attunement': (False, 0.0),
        'Low attunement (η=0.05)': (True, 0.05),
        'Medium attunement (η=0.20)': (True, 0.20),
        'High attunement (η=0.50)': (True, 0.50),
        'Very high attunement (η=1.0)': (True, 1.0),
    }
    
    results = {}
    n_steps = 5000
    
    for name, (attune_on, eta) in configs.items():
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.15,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False,
            boundary_enabled=True, grace_enabled=True,
            F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.03
        )
        collider = DETCollider1D(params)
        
        # Pure W-regime everywhere
        collider.C_R[:] = 0.15
        collider.q[:] = 0.50
        collider.F[:] = 0.5
        collider.a[:] = 0.5
        
        # Community has slightly higher agency (they "choose" to engage)
        collider.a[community_mask] = 0.8
        
        np.random.seed(42)
        rp = RegimeParams(attunement_enabled=attune_on, eta_attune=eta)
        
        # Community nodes use selective observation (seek each other)
        regime = SelectiveObservationSimulator(
            collider, rp,
            selective_mode='seek_K' if attune_on else 'none',
            selectivity_strength=2.0,
            selective_mask=community_mask
        )
        
        K_community = []
        K_outside = []
        C_community = []
        
        outside_mask = ~community_mask
        
        for t in range(n_steps):
            diag = regime.step(t)
            K_community.append(float(np.mean(diag.K[community_mask])))
            K_outside.append(float(np.mean(diag.K[outside_mask])))
            C_community.append(float(np.mean(collider.C_R[community_mask])))
        
        results[name] = {
            'K_community': np.array(K_community),
            'K_outside': np.array(K_outside),
            'C_community': np.array(C_community),
            'final_K': diag.K.copy(),
            'final_C': collider.C_R.copy()
        }
        print(f"  {name:35s}: community K={K_community[-1]:.4f}, "
              f"outside K={K_outside[-1]:.4f}, community C={C_community[-1]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("L4: Pure W-Regime — Can Internal Observation Bootstrap K?",
                 fontsize=14, fontweight='bold')
    
    cmap = plt.cm.viridis
    
    ax = axes[0, 0]
    for i, (name, data) in enumerate(results.items()):
        color = cmap(i / max(1, len(results) - 1))
        ax.plot(data['K_community'], color=color, linewidth=2, label=name)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean K (community)')
    ax.set_title('K in Community Over Time')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        color = cmap(i / max(1, len(results) - 1))
        ax.plot(data['C_community'], color=color, linewidth=2, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean C (community)')
    ax.set_title('Coherence in Community Over Time')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for i, (name, data) in enumerate(results.items()):
        color = cmap(i / max(1, len(results) - 1))
        ax.plot(data['final_K'], color=color, linewidth=1.5, alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.axvspan(community_center - community_width, community_center + community_width,
               alpha=0.1, color='blue', label='Community')
    ax.set_xlabel('Node')
    ax.set_ylabel('K_i')
    ax.set_title('Final K Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for i, (name, data) in enumerate(results.items()):
        color = cmap(i / max(1, len(results) - 1))
        ax.plot(data['final_C'], color=color, linewidth=1.5, alpha=0.8)
    ax.axvspan(community_center - community_width, community_center + community_width,
               alpha=0.1, color='blue', label='Community')
    ax.set_xlabel('Node')
    ax.set_ylabel('C')
    ax.set_title('Final Coherence Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/L4_pure_W_internal.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/L4_pure_W_internal.png")
    return results


# ============================================================
# L5: MECHANISM DECOMPOSITION — WHAT MATTERS MOST?
# ============================================================

def experiment_L5_mechanism_decomposition():
    """
    L5: Mechanism Decomposition
    
    Systematically test every combination of local mechanisms to
    determine which ones matter most for W→K transition near a
    K-region boundary.
    
    Mechanisms:
      G = Grace flux
      H = Bond healing
      A = Attunement feedback
      S = Selective observation (seek K)
    """
    print("\n" + "="*70)
    print("L5: Mechanism Decomposition — What Matters Most?")
    print("="*70)
    
    N = 200
    K_center = 50
    K_width = 30
    x = np.arange(N)
    K_mask = np.abs(x - K_center) < K_width
    W_mask = ~K_mask
    
    # All 16 combinations of 4 binary mechanisms
    mechanisms = {
        '----': (False, False, False, False),
        'G---': (True, False, False, False),
        '-H--': (False, True, False, False),
        '--A-': (False, False, True, False),
        '---S': (False, False, False, True),
        'GH--': (True, True, False, False),
        'G-A-': (True, False, True, False),
        'G--S': (True, False, False, True),
        '-HA-': (False, True, True, False),
        '-H-S': (False, True, False, True),
        '--AS': (False, False, True, True),
        'GHA-': (True, True, True, False),
        'GH-S': (True, True, False, True),
        'G-AS': (True, False, True, True),
        '-HAS': (False, True, True, True),
        'GHAS': (True, True, True, True),
    }
    
    results = {}
    n_steps = 3000
    
    for name, (grace, healing, attune, selective) in mechanisms.items():
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.15,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False,
            boundary_enabled=True,
            grace_enabled=grace,
            F_MIN_grace=0.05,
            healing_enabled=healing,
            eta_heal=0.03
        )
        collider = DETCollider1D(params)
        
        collider.C_R[:] = 0.15
        collider.q[:] = 0.50
        collider.F[:] = 0.5
        collider.a[:] = 0.5
        
        collider.C_R[K_mask] = 0.90
        collider.q[K_mask] = 0.02
        collider.F[K_mask] = 3.0
        collider.a[K_mask] = 0.95
        
        np.random.seed(42)
        rp = RegimeParams(attunement_enabled=attune, eta_attune=0.10)
        
        if selective:
            regime = SelectiveObservationSimulator(
                collider, rp,
                selective_mode='seek_K',
                selectivity_strength=2.0,
                selective_mask=W_mask
            )
        else:
            regime = SelectiveObservationSimulator(
                collider, rp,
                selective_mode='none',
                selectivity_strength=0.0,
                selective_mask=W_mask
            )
        
        K_W_final = 0
        for t in range(n_steps):
            diag = regime.step(t)
        
        K_W_final = float(np.mean(diag.K[W_mask]))
        C_W_final = float(np.mean(collider.C_R[W_mask]))
        
        results[name] = {
            'K_W': K_W_final,
            'C_W': C_W_final,
            'grace': grace,
            'healing': healing,
            'attune': attune,
            'selective': selective
        }
    
    # Sort by K_W
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['K_W'])
    
    print(f"\n  {'Config':>6} | {'K_W':>8} | {'C_W':>8} | {'G':>3} | {'H':>3} | {'A':>3} | {'S':>3}")
    print("  " + "-"*50)
    for name, data in sorted_results:
        print(f"  {name:>6} | {data['K_W']:8.4f} | {data['C_W']:8.4f} | "
              f"{'Y' if data['grace'] else '-':>3} | {'Y' if data['healing'] else '-':>3} | "
              f"{'Y' if data['attune'] else '-':>3} | {'Y' if data['selective'] else '-':>3}")
    
    # Plot: bar chart of K_W for all combinations
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("L5: Mechanism Decomposition — What Matters Most?",
                 fontsize=14, fontweight='bold')
    
    names = [x[0] for x in sorted_results]
    K_vals = [x[1]['K_W'] for x in sorted_results]
    C_vals = [x[1]['C_W'] for x in sorted_results]
    
    bar_colors = []
    for n, d in sorted_results:
        if d['grace'] and d['healing'] and d['attune'] and d['selective']:
            bar_colors.append('gold')
        elif d['healing']:
            bar_colors.append('green')
        elif d['attune'] or d['selective']:
            bar_colors.append('blue')
        elif d['grace']:
            bar_colors.append('orange')
        else:
            bar_colors.append('gray')
    
    ax = axes[0]
    bars = ax.barh(range(len(names)), K_vals, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontfamily='monospace', fontsize=9)
    ax.set_xlabel('Final K in W-Region')
    ax.set_title('W-Region K by Mechanism Combination')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    ax = axes[1]
    bars = ax.barh(range(len(names)), C_vals, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontfamily='monospace', fontsize=9)
    ax.set_xlabel('Final C in W-Region')
    ax.set_title('W-Region C by Mechanism Combination')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/L5_decomposition.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/L5_decomposition.png")
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("DET LOCAL TRANSITION MECHANISMS: W → K")
    print("="*70)
    
    r1 = experiment_L1_grace_flux()
    r2 = experiment_L2_proximity()
    r3 = experiment_L3_selective_observation()
    r4 = experiment_L4_pure_W_internal()
    r5 = experiment_L5_mechanism_decomposition()
    
    print("\n" + "="*70)
    print("ALL LOCAL MECHANISM EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
