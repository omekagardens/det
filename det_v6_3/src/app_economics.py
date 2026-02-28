"""
DET Application: Economics & Market Dynamics
==============================================

Models economic networks using DET primitives to demonstrate:
  C1. Market Regime Classification — mapping financial markets to K/W regimes
      using coherence (trust/institutional strength), structural debt (systemic
      risk), and agency (market participation).
  C2. Contagion vs Resilience — how shocks propagate differently through
      K-regime (well-regulated) vs W-regime (fragile) market sectors.
  C3. Regulatory Coherence Engineering — modeling the effect of regulation
      as coherence injection, showing how it shifts markets from W to K regime.
  C4. Wealth/Resource Flow — how resource (F) flows between K and W sectors
      and the role of observability in information asymmetry.

DET Primitive → Economic Interpretation:
  F_i     → Capital/wealth at institution i
  C_ij    → Trust/institutional bond strength between i and j
  q_i     → Systemic risk / toxic debt at institution i
  a_i     → Market participation / agency of actor i
  K_i     → Market health index (K≈1: stable, K≈0: fragile)
  O_i     → Information transparency / market visibility
  Ξ^seen  → Perceived market signal quality
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from det_v6_3_1d_collider import DETCollider1D, DETParams1D
from det_concurrent_regimes import (
    DETRegimeSimulator, RegimeParams,
    create_k_region, create_w_region
)

OUTPUT_DIR = "/home/ubuntu/det_app_results/economics"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# C1: MARKET REGIME CLASSIFICATION
# ============================================================

def experiment_C1_market_regimes():
    """
    C1: Market Regime Classification
    
    Model a 300-node economic network with three sectors:
      - Sector A (nodes 0-99): Well-regulated (high C, low q) → K-regime
      - Sector B (nodes 100-199): Mixed/emerging (moderate C, moderate q)
      - Sector C (nodes 200-299): Shadow/fragile (low C, high q) → W-regime
    
    Run the DET dynamics and classify each sector using K_i, O_i, and Ξ^seen.
    """
    print("\n" + "="*70)
    print("C1: Market Regime Classification")
    print("="*70)
    
    N = 300
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.003,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    np.random.seed(42)
    collider = DETCollider1D(params)
    
    # Sector A: Well-regulated (K-regime)
    for i in range(0, 100):
        collider.C_R[i] = 0.85 + np.random.uniform(-0.05, 0.05)
        collider.q[i] = 0.05 + np.random.uniform(0, 0.03)
        collider.F[i] = 3.0 + np.random.uniform(-0.5, 0.5)
        collider.a[i] = 0.90 + np.random.uniform(-0.05, 0.05)
    
    # Sector B: Mixed/emerging
    for i in range(100, 200):
        collider.C_R[i] = 0.45 + np.random.uniform(-0.10, 0.10)
        collider.q[i] = 0.25 + np.random.uniform(-0.05, 0.10)
        collider.F[i] = 1.5 + np.random.uniform(-0.5, 0.5)
        collider.a[i] = 0.70 + np.random.uniform(-0.10, 0.10)
    
    # Sector C: Shadow/fragile (W-regime)
    for i in range(200, 300):
        collider.C_R[i] = 0.12 + np.random.uniform(-0.05, 0.05)
        collider.q[i] = 0.55 + np.random.uniform(-0.05, 0.15)
        collider.F[i] = 0.5 + np.random.uniform(-0.2, 0.3)
        collider.a[i] = 0.40 + np.random.uniform(-0.15, 0.15)
    
    np.random.seed(42)
    regime = DETRegimeSimulator(collider, RegimeParams())
    
    # Run simulation
    n_steps = 1000
    sector_A = slice(0, 100)
    sector_B = slice(100, 200)
    sector_C = slice(200, 300)
    
    times = []
    metrics = {s: {'K': [], 'O': [], 'Xi_seen': [], 'F': [], 'q': []}
               for s in ['Sector A (Regulated)', 'Sector B (Emerging)', 'Sector C (Shadow)']}
    sector_slices = {
        'Sector A (Regulated)': sector_A,
        'Sector B (Emerging)': sector_B,
        'Sector C (Shadow)': sector_C
    }
    
    for t in range(n_steps):
        diag = regime.step(t)
        times.append(t)
        
        for name, sl in sector_slices.items():
            metrics[name]['K'].append(float(np.mean(diag.K[sl])))
            metrics[name]['O'].append(float(np.mean(diag.O[sl])))
            metrics[name]['Xi_seen'].append(float(np.mean(diag.Xi_seen[sl])))
            metrics[name]['F'].append(float(np.mean(collider.F[sl])))
            metrics[name]['q'].append(float(np.mean(collider.q[sl])))
    
    # Print final state
    for name in metrics:
        m = metrics[name]
        print(f"\n  {name}:")
        print(f"    K̄={m['K'][-1]:.3f}, Ō={m['O'][-1]:.3f}, Ξ̄^seen={m['Xi_seen'][-1]:.4f}")
        print(f"    F̄={m['F'][-1]:.3f}, q̄={m['q'][-1]:.3f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("C1: Market Regime Classification — Three Economic Sectors",
                 fontsize=14, fontweight='bold')
    
    colors = {'Sector A (Regulated)': 'blue',
              'Sector B (Emerging)': 'orange',
              'Sector C (Shadow)': 'red'}
    
    # K_i over time
    ax = axes[0, 0]
    for name, m in metrics.items():
        ax.plot(times, m['K'], color=colors[name], linewidth=2, label=name)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Mean K_i (Market Health)')
    ax.set_title('Regime Index')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # O_i over time
    ax = axes[0, 1]
    for name, m in metrics.items():
        ax.plot(times, m['O'], color=colors[name], linewidth=2, label=name)
    ax.set_ylabel('Mean O_i (Transparency)')
    ax.set_title('Information Transparency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Ξ^seen over time
    ax = axes[0, 2]
    for name, m in metrics.items():
        ax.plot(times, m['Xi_seen'], color=colors[name], linewidth=2, label=name)
    ax.set_ylabel('Mean Ξ^seen')
    ax.set_title('Perceived Signal Quality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Capital (F) over time
    ax = axes[1, 0]
    for name, m in metrics.items():
        ax.plot(times, m['F'], color=colors[name], linewidth=2, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean F (Capital)')
    ax.set_title('Capital Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Systemic risk (q) over time
    ax = axes[1, 1]
    for name, m in metrics.items():
        ax.plot(times, m['q'], color=colors[name], linewidth=2, label=name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean q (Systemic Risk)')
    ax.set_title('Systemic Risk Accumulation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Final spatial profile
    ax = axes[1, 2]
    x = np.arange(N)
    final_K = regime.K.copy()
    ax.fill_between(x[:100], final_K[:100], alpha=0.6, color='blue', label='Regulated')
    ax.fill_between(x[100:200], final_K[100:200], alpha=0.6, color='orange', label='Emerging')
    ax.fill_between(x[200:], final_K[200:], alpha=0.6, color='red', label='Shadow')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Institution Index')
    ax.set_ylabel('K_i')
    ax.set_title('Final Market Health Profile')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/C1_market_regimes.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    return metrics


# ============================================================
# C2: CONTAGION VS RESILIENCE
# ============================================================

def experiment_C2_contagion():
    """
    C2: Contagion vs Resilience
    
    Inject a "financial shock" (sudden capital destruction + debt spike)
    into both a K-regime sector and a W-regime sector. Measure:
      - Shock propagation distance
      - Recovery time
      - Cascade (how many additional institutions fail)
    """
    print("\n" + "="*70)
    print("C2: Contagion vs Resilience — Financial Shock Propagation")
    print("="*70)
    
    N = 300
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.003,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    np.random.seed(42)
    collider = DETCollider1D(params)
    
    # K-sector (left): well-regulated
    create_k_region(collider, center=75, width=60, C_level=0.88, q_level=0.03, F_level=3.0)
    # W-sector (right): fragile
    create_w_region(collider, center=225, width=60, C_level=0.12, q_level=0.55, F_level=0.3)
    
    np.random.seed(42)
    regime = DETRegimeSimulator(collider, RegimeParams())
    
    # Warm up
    for t in range(200):
        regime.step(t)
    
    # Record pre-shock state
    pre_K = regime.K.copy()
    pre_F = collider.F.copy()
    
    # Inject shock at center of each sector
    k_shock_center = 75
    w_shock_center = 225
    shock_width = 5
    
    for center in [k_shock_center, w_shock_center]:
        for i in range(center - shock_width, center + shock_width):
            if 0 <= i < N:
                collider.F[i] = max(0, collider.F[i] - 5.0)  # capital destruction
                collider.q[i] = min(1.0, collider.q[i] + 0.5)  # debt spike
                collider.C_R[i] *= 0.3  # trust collapse
    
    print(f"  Shock injected at nodes {k_shock_center} (K-sector) and {w_shock_center} (W-sector)")
    
    # Track post-shock dynamics
    n_post = 1500
    times = []
    
    k_sector = slice(15, 135)
    w_sector = slice(165, 285)
    
    k_mean_K = []
    w_mean_K = []
    k_mean_F = []
    w_mean_F = []
    k_failed = []  # nodes with K < 0.1
    w_failed = []
    
    # Track spatial propagation
    k_shock_reach = []
    w_shock_reach = []
    
    for t in range(n_post):
        diag = regime.step(200 + t)
        times.append(t)
        
        k_mean_K.append(float(np.mean(diag.K[k_sector])))
        w_mean_K.append(float(np.mean(diag.K[w_sector])))
        k_mean_F.append(float(np.mean(collider.F[k_sector])))
        w_mean_F.append(float(np.mean(collider.F[w_sector])))
        
        # Count failed institutions
        k_failed.append(int(np.sum(diag.K[k_sector] < 0.15)))
        w_failed.append(int(np.sum(diag.K[w_sector] < 0.05)))
        
        # Measure shock propagation: how far from center has K dropped > 20%
        k_drop = pre_K[k_sector] - diag.K[k_sector]
        w_drop = pre_K[w_sector] - diag.K[w_sector]
        
        k_affected = np.where(k_drop > 0.1)[0]
        w_affected = np.where(w_drop > 0.02)[0]
        
        k_reach = int(np.max(np.abs(k_affected - (k_shock_center - 15)))) if len(k_affected) > 0 else 0
        w_reach = int(np.max(np.abs(w_affected - (w_shock_center - 165)))) if len(w_affected) > 0 else 0
        
        k_shock_reach.append(k_reach)
        w_shock_reach.append(w_reach)
    
    print(f"\n  K-sector: max failed institutions = {max(k_failed)}/120")
    print(f"  W-sector: max failed institutions = {max(w_failed)}/120")
    print(f"  K-sector: max shock reach = {max(k_shock_reach)} nodes")
    print(f"  W-sector: max shock reach = {max(w_shock_reach)} nodes")
    print(f"  K-sector final K̄ = {k_mean_K[-1]:.3f}")
    print(f"  W-sector final K̄ = {w_mean_K[-1]:.3f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("C2: Financial Contagion — K-Sector Resilience vs W-Sector Cascade",
                 fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(times, k_mean_K, 'b-', linewidth=2, label='K-sector (Regulated)')
    ax.plot(times, w_mean_K, 'r-', linewidth=2, label='W-sector (Shadow)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5, label='Shock')
    ax.set_ylabel('Mean K_i')
    ax.set_title('Market Health After Shock')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(times, k_failed, 'b-', linewidth=2, label='K-sector failed')
    ax.plot(times, w_failed, 'r-', linewidth=2, label='W-sector failed')
    ax.set_ylabel('Failed Institutions')
    ax.set_title('Institutional Failures (Cascade)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(times, k_shock_reach, 'b-', linewidth=2, label='K-sector reach')
    ax.plot(times, w_shock_reach, 'r-', linewidth=2, label='W-sector reach')
    ax.set_xlabel('Steps After Shock')
    ax.set_ylabel('Shock Reach (nodes)')
    ax.set_title('Contagion Propagation Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(times, k_mean_F, 'b-', linewidth=2, label='K-sector capital')
    ax.plot(times, w_mean_F, 'r-', linewidth=2, label='W-sector capital')
    ax.set_xlabel('Steps After Shock')
    ax.set_ylabel('Mean F (Capital)')
    ax.set_title('Capital Recovery After Shock')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/C2_contagion.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return {
        'k_max_failed': max(k_failed),
        'w_max_failed': max(w_failed),
        'k_max_reach': max(k_shock_reach),
        'w_max_reach': max(w_shock_reach)
    }


# ============================================================
# C3: REGULATORY COHERENCE ENGINEERING
# ============================================================

def experiment_C3_regulation():
    """
    C3: Regulatory Coherence Engineering
    
    Start with a W-regime (fragile) market. Apply "regulation" modeled as
    gradual coherence injection at different rates. Measure:
      - Time to reach K-regime (K > 0.5)
      - Effect on systemic risk (q)
      - Effect on capital stability (F variance)
    """
    print("\n" + "="*70)
    print("C3: Regulatory Coherence Engineering")
    print("="*70)
    
    N = 100
    base_params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.2,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.003,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    regulation_rates = [0.0, 0.001, 0.003, 0.005, 0.01]
    results = {}
    
    for rate in regulation_rates:
        np.random.seed(42)
        collider = DETCollider1D(base_params)
        
        # Start as W-regime
        collider.C_R[:] = 0.15
        collider.q[:] = 0.50
        collider.F[:] = 0.8
        collider.a[:] = 0.5
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams())
        
        n_steps = 2000
        times = []
        mean_K = []
        mean_C = []
        mean_q = []
        F_std = []
        
        transition_step = None
        
        for t in range(n_steps):
            # Apply regulation: gradual C injection AND q reduction
            # Real regulation both builds trust (C↑) and reduces risk (q↓)
            if rate > 0:
                collider.C_R[:] = np.clip(collider.C_R + rate, 0, 1)
                collider.q[:] = np.clip(collider.q - rate * 0.5, 0, 1)
            
            diag = regime.step(t)
            times.append(t)
            mean_K.append(float(np.mean(diag.K)))
            mean_C.append(float(np.mean(collider.C_R)))
            mean_q.append(float(np.mean(collider.q)))
            F_std.append(float(np.std(collider.F)))
            
            if transition_step is None and np.mean(diag.K) > 0.5:
                transition_step = t
        
        label = f"Rate={rate}" if rate > 0 else "No Regulation"
        results[label] = {
            'times': times,
            'mean_K': mean_K,
            'mean_C': mean_C,
            'mean_q': mean_q,
            'F_std': F_std,
            'transition_step': transition_step,
            'rate': rate
        }
        
        print(f"  {label}: transition at step {transition_step or 'NEVER'}, "
              f"final K̄={mean_K[-1]:.3f}, final q̄={mean_q[-1]:.3f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("C3: Regulatory Coherence Engineering — From W-Regime to K-Regime",
                 fontsize=14, fontweight='bold')
    
    cmap = plt.cm.viridis
    n_rates = len(results)
    
    ax = axes[0, 0]
    for i, (label, data) in enumerate(results.items()):
        color = cmap(i / max(1, n_rates - 1))
        ax.plot(data['times'], data['mean_K'], color=color, linewidth=2, label=label)
        if data['transition_step']:
            ax.axvline(x=data['transition_step'], color=color, linestyle=':', alpha=0.4)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='K/W boundary')
    ax.set_ylabel('Mean K_i')
    ax.set_title('Market Health Under Regulation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for i, (label, data) in enumerate(results.items()):
        color = cmap(i / max(1, n_rates - 1))
        ax.plot(data['times'], data['mean_C'], color=color, linewidth=2, label=label)
    ax.set_ylabel('Mean C (Trust/Institutional Strength)')
    ax.set_title('Coherence Growth Under Regulation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for i, (label, data) in enumerate(results.items()):
        color = cmap(i / max(1, n_rates - 1))
        ax.plot(data['times'], data['mean_q'], color=color, linewidth=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean q (Systemic Risk)')
    ax.set_title('Systemic Risk Reduction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Transition time vs regulation rate
    ax = axes[1, 1]
    rates_with_transition = [(data['rate'], data['transition_step'])
                              for data in results.values()
                              if data['transition_step'] is not None]
    if rates_with_transition:
        r, t = zip(*rates_with_transition)
        ax.plot(r, t, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel('Regulation Rate (ΔC per step)')
        ax.set_ylabel('Steps to K-Regime Transition')
        ax.set_title('Regulation Efficiency')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No transitions achieved', ha='center', va='center',
                fontsize=14, transform=ax.transAxes)
        ax.set_title('Regulation Efficiency')
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/C3_regulation.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return results


# ============================================================
# C4: INFORMATION ASYMMETRY & WEALTH FLOW
# ============================================================

def experiment_C4_information_asymmetry():
    """
    C4: Information Asymmetry & Wealth Flow
    
    Model how information transparency (O_i) and signal quality (Ξ^seen)
    create advantages for K-regime actors over W-regime actors.
    
    K-regime actors can "see" market structure clearly and make better
    decisions (modeled as resource-efficient flow). W-regime actors
    operate blind, leading to resource waste.
    """
    print("\n" + "="*70)
    print("C4: Information Asymmetry & Wealth Flow")
    print("="*70)
    
    N = 200
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.003,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    np.random.seed(42)
    collider = DETCollider1D(params)
    
    # K-sector: left half (well-informed)
    create_k_region(collider, center=50, width=40, C_level=0.88, q_level=0.03, F_level=2.0)
    # W-sector: right half (poorly informed)
    create_w_region(collider, center=150, width=40, C_level=0.12, q_level=0.55, F_level=2.0)
    # Note: SAME initial capital (F=2.0) to both sectors
    
    # Enable attunement so K-sector's mutual observability compounds advantage
    np.random.seed(42)
    regime = DETRegimeSimulator(collider, RegimeParams(
        attunement_enabled=True, eta_attune=0.05
    ))
    
    k_sector = slice(10, 90)
    w_sector = slice(110, 190)
    
    n_steps = 2000
    times = []
    k_total_F = []
    w_total_F = []
    k_mean_O = []
    w_mean_O = []
    k_mean_Xi = []
    w_mean_Xi = []
    k_gini = []
    w_gini = []
    
    def compute_gini(arr):
        """Compute Gini coefficient for inequality measurement."""
        arr = np.sort(arr)
        n = len(arr)
        if n == 0 or np.sum(arr) == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))
    
    for t in range(n_steps):
        diag = regime.step(t)
        times.append(t)
        
        k_total_F.append(float(np.sum(collider.F[k_sector])))
        w_total_F.append(float(np.sum(collider.F[w_sector])))
        k_mean_O.append(float(np.mean(diag.O[k_sector])))
        w_mean_O.append(float(np.mean(diag.O[w_sector])))
        k_mean_Xi.append(float(np.mean(diag.Xi_seen[k_sector])))
        w_mean_Xi.append(float(np.mean(diag.Xi_seen[w_sector])))
        
        k_gini.append(compute_gini(collider.F[k_sector]))
        w_gini.append(compute_gini(collider.F[w_sector]))
    
    print(f"  Initial total capital: K={k_total_F[0]:.1f}, W={w_total_F[0]:.1f}")
    print(f"  Final total capital:   K={k_total_F[-1]:.1f}, W={w_total_F[-1]:.1f}")
    print(f"  K-sector capital change: {((k_total_F[-1]/k_total_F[0])-1)*100:+.1f}%")
    print(f"  W-sector capital change: {((w_total_F[-1]/w_total_F[0])-1)*100:+.1f}%")
    print(f"  Final Gini: K={k_gini[-1]:.3f}, W={w_gini[-1]:.3f}")
    print(f"  Final mean O: K={k_mean_O[-1]:.3f}, W={w_mean_O[-1]:.4f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("C4: Information Asymmetry & Wealth Flow",
                 fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(times, k_total_F, 'b-', linewidth=2, label='K-sector (Informed)')
    ax.plot(times, w_total_F, 'r-', linewidth=2, label='W-sector (Blind)')
    ax.set_ylabel('Total Capital (F)')
    ax.set_title('Capital Accumulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(times, k_mean_O, 'b-', linewidth=2, label='K-sector O')
    ax.plot(times, w_mean_O, 'r-', linewidth=2, label='W-sector O')
    ax.set_ylabel('Mean O_i (Transparency)')
    ax.set_title('Information Transparency Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.plot(times, k_mean_Xi, 'b-', linewidth=2, label='K-sector Ξ^seen')
    ax.plot(times, w_mean_Xi, 'r-', linewidth=2, label='W-sector Ξ^seen')
    ax.set_ylabel('Mean Ξ^seen')
    ax.set_title('Market Signal Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(times, k_gini, 'b-', linewidth=2, label='K-sector Gini')
    ax.plot(times, w_gini, 'r-', linewidth=2, label='W-sector Gini')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Wealth Inequality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Capital ratio
    ax = axes[1, 1]
    ratio = [k / max(w, 0.01) for k, w in zip(k_total_F, w_total_F)]
    ax.plot(times, ratio, 'purple', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Parity')
    ax.set_xlabel('Step')
    ax.set_ylabel('K/W Capital Ratio')
    ax.set_title('Wealth Gap Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final wealth distribution
    ax = axes[1, 2]
    k_F_final = np.sort(collider.F[k_sector])[::-1]
    w_F_final = np.sort(collider.F[w_sector])[::-1]
    ax.plot(np.arange(len(k_F_final)), k_F_final, 'b-', linewidth=2, label='K-sector')
    ax.plot(np.arange(len(w_F_final)), w_F_final, 'r-', linewidth=2, label='W-sector')
    ax.set_xlabel('Institution Rank')
    ax.set_ylabel('Capital F')
    ax.set_title('Final Wealth Distribution (Ranked)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/C4_information_asymmetry.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return {
        'k_final_F': k_total_F[-1],
        'w_final_F': w_total_F[-1],
        'k_gini': k_gini[-1],
        'w_gini': w_gini[-1]
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("DET APPLICATION: ECONOMICS & MARKET DYNAMICS")
    print("="*70)
    
    r1 = experiment_C1_market_regimes()
    r2 = experiment_C2_contagion()
    r3 = experiment_C3_regulation()
    r4 = experiment_C4_information_asymmetry()
    
    print("\n" + "="*70)
    print("ALL ECONOMICS EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
