"""
DET Application: Materials Science & Stability Engineering
============================================================

Models material lattices using DET primitives to demonstrate:
  B1. Self-Healing Dynamics — K-regime materials autonomously repair damage
      via coherence diffusion and attunement feedback.
  B2. Coherence-Engineered Stability — artificially boosting C in a material
      shifts it from W-regime to K-regime, dramatically reducing decay/failure.
  B3. Stress-Strain Response — K vs W materials under increasing external load
      (modeled as q-injection) show fundamentally different failure modes.

All dynamics use the DET 1D collider as the substrate.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from det_v6_3_1d_collider import DETCollider1D, DETParams1D
from det_concurrent_regimes import (
    DETRegimeSimulator, RegimeParams,
    create_k_region, create_w_region
)

OUTPUT_DIR = "/home/ubuntu/det_app_results/materials"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# B1: SELF-HEALING DYNAMICS
# ============================================================

def experiment_B1_self_healing():
    """
    B1: Self-Healing Dynamics
    
    Create a K-regime material lattice. Introduce a "crack" (a contiguous
    region of destroyed bonds: C=0, q=1, F=0). Run with attunement feedback
    enabled. Measure:
      - Crack width over time
      - Coherence recovery at the crack site
      - Comparison: healing WITH vs WITHOUT attunement
    """
    print("\n" + "="*70)
    print("B1: Self-Healing Dynamics")
    print("="*70)
    
    N = 200
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.5,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.005,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    results = {}
    
    for label, attune_enabled, eta in [("No Attunement", False, 0.0),
                                         ("Mild Attunement (η=0.05)", True, 0.05),
                                         ("Strong Attunement (η=0.15)", True, 0.15)]:
        np.random.seed(42)
        collider = DETCollider1D(params)
        
        # Create a uniform K-regime material
        collider.C_R[:] = 0.90
        collider.q[:] = 0.02
        collider.a[:] = 0.95
        collider.F[:] = 3.0
        
        rp = RegimeParams(attunement_enabled=attune_enabled, eta_attune=eta)
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, rp)
        
        # Warm up for 100 steps
        for t in range(100):
            regime.step(t)
        
        # Record pre-crack state
        pre_crack_C = collider.C_R.copy()
        pre_crack_K = regime.K.copy()
        
        # Introduce a "crack" at nodes 95-105
        crack_start, crack_end = 95, 105
        for i in range(crack_start, crack_end):
            collider.C_R[i] = 0.0
            if i > 0:
                collider.C_R[i-1] = 0.0
            collider.q[i] = 0.9
            collider.F[i] = 0.01
        
        # Track healing
        n_heal_steps = 1500
        times = []
        crack_width = []  # number of nodes with C < 0.3
        mean_C_crack = []
        mean_K_crack = []
        mean_K_bulk = []
        
        crack_region = slice(crack_start - 5, crack_end + 5)
        bulk_region_L = slice(20, 80)
        bulk_region_R = slice(120, 180)
        
        for t in range(n_heal_steps):
            diag = regime.step(100 + t)
            times.append(t)
            
            # Measure crack width
            low_C = np.sum(collider.C_R[crack_region] < 0.3)
            crack_width.append(int(low_C))
            mean_C_crack.append(float(np.mean(collider.C_R[crack_region])))
            mean_K_crack.append(float(np.mean(diag.K[crack_region])))
            mean_K_bulk.append(float(np.mean(
                np.concatenate([diag.K[bulk_region_L], diag.K[bulk_region_R]]))))
        
        results[label] = {
            'times': times,
            'crack_width': crack_width,
            'mean_C_crack': mean_C_crack,
            'mean_K_crack': mean_K_crack,
            'mean_K_bulk': mean_K_bulk,
            'final_C': collider.C_R.copy(),
            'final_K': regime.K.copy()
        }
        
        print(f"\n  {label}:")
        print(f"    Initial crack width: {crack_width[0]} low-C nodes")
        print(f"    Final crack width:   {crack_width[-1]} low-C nodes")
        print(f"    Mean C at crack (final): {mean_C_crack[-1]:.3f}")
        print(f"    Mean K at crack (final): {mean_K_crack[-1]:.3f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("B1: Self-Healing Dynamics — Coherence Recovery After Crack",
                 fontsize=14, fontweight='bold')
    
    colors = {'No Attunement': 'gray',
              'Mild Attunement (η=0.05)': 'blue',
              'Strong Attunement (η=0.15)': 'green'}
    
    # Crack width over time
    ax = axes[0, 0]
    for label, data in results.items():
        ax.plot(data['times'], data['crack_width'], color=colors[label],
                linewidth=2, label=label)
    ax.set_ylabel('Low-C Nodes in Crack Region')
    ax.set_xlabel('Steps After Crack')
    ax.set_title('Crack Width Over Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Mean C at crack site
    ax = axes[0, 1]
    for label, data in results.items():
        ax.plot(data['times'], data['mean_C_crack'], color=colors[label],
                linewidth=2, label=label)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Pre-crack C')
    ax.set_ylabel('Mean Coherence at Crack')
    ax.set_xlabel('Steps After Crack')
    ax.set_title('Coherence Recovery at Crack Site')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Mean K at crack vs bulk
    ax = axes[1, 0]
    for label, data in results.items():
        ax.plot(data['times'], data['mean_K_crack'], color=colors[label],
                linewidth=2, linestyle='-', label=f'{label} (crack)')
        ax.plot(data['times'], data['mean_K_bulk'], color=colors[label],
                linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Mean K_i')
    ax.set_xlabel('Steps After Crack')
    ax.set_title('Regime Index: Crack vs Bulk (dashed)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Final spatial C profile
    ax = axes[1, 1]
    x = np.arange(N)
    for label, data in results.items():
        ax.plot(x[70:130], data['final_C'][70:130], color=colors[label],
                linewidth=2, label=label)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Original C')
    ax.axvspan(95, 105, alpha=0.1, color='red', label='Crack site')
    ax.set_xlabel('Position')
    ax.set_ylabel('Coherence C')
    ax.set_title('Final Coherence Profile Near Crack')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/B1_self_healing.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    return results


# ============================================================
# B2: COHERENCE-ENGINEERED STABILITY
# ============================================================

def experiment_B2_coherence_engineering():
    """
    B2: Coherence-Engineered Stability
    
    Start with a uniform W-regime material. Progressively "engineer" its
    coherence upward (simulating a manufacturing process). Measure:
      - Regime index K_i as a function of engineered C
      - Decay rate (from instability score) as a function of C
      - Critical C threshold where the material transitions from W to K
    """
    print("\n" + "="*70)
    print("B2: Coherence-Engineered Stability Curve")
    print("="*70)
    
    N = 100
    C_levels = np.linspace(0.05, 0.95, 30)
    
    mean_K_values = []
    mean_s_values = []  # instability score
    mean_O_values = []
    decay_counts = []
    
    for C_val in C_levels:
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=C_val,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.005,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False, boundary_enabled=False
        )
        collider = DETCollider1D(params)
        collider.C_R[:] = C_val
        collider.q[:] = max(0.01, 0.6 - 0.6 * C_val)  # q inversely related to C
        collider.F[:] = 1.0 + 2.0 * C_val
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams())
        
        # Run 500 steps
        for t in range(500):
            diag = regime.step(t)
        
        # Compute instability score (from decay module logic)
        s = 0.4 * collider.q + 0.35 * (1.0 - collider.C_R) + 0.25 * (1.0 / (collider.F + 0.1))
        
        # Count "decay events" (nodes where s > 0.6)
        n_decays = int(np.sum(s > 0.6))
        
        mean_K_values.append(float(np.mean(diag.K)))
        mean_s_values.append(float(np.mean(s)))
        mean_O_values.append(float(np.mean(diag.O)))
        decay_counts.append(n_decays)
    
    # Find critical C threshold (K crosses 0.5)
    K_arr = np.array(mean_K_values)
    crossings = np.where(np.diff(np.sign(K_arr - 0.5)))[0]
    C_critical = float(C_levels[crossings[0]]) if len(crossings) > 0 else None
    
    print(f"  Critical C threshold (K=0.5): {C_critical:.3f}" if C_critical else "  No clear transition found")
    print(f"  At C=0.1: K̄={mean_K_values[1]:.3f}, s̄={mean_s_values[1]:.3f}, decays={decay_counts[1]}")
    print(f"  At C=0.5: K̄={mean_K_values[15]:.3f}, s̄={mean_s_values[15]:.3f}, decays={decay_counts[15]}")
    print(f"  At C=0.9: K̄={mean_K_values[28]:.3f}, s̄={mean_s_values[28]:.3f}, decays={decay_counts[28]}")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("B2: Coherence-Engineered Stability — Manufacturing Curve",
                 fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(C_levels, mean_K_values, 'b-o', linewidth=2, markersize=4)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='K/W boundary')
    if C_critical:
        ax.axvline(x=C_critical, color='green', linestyle=':', alpha=0.7, label=f'C_crit={C_critical:.2f}')
    ax.fill_between(C_levels, 0, 0.5, alpha=0.05, color='red')
    ax.fill_between(C_levels, 0.5, 1.0, alpha=0.05, color='blue')
    ax.text(0.15, 0.25, 'W-REGIME\n(Unstable)', ha='center', fontsize=11, color='red', alpha=0.7)
    ax.text(0.80, 0.75, 'K-REGIME\n(Stable)', ha='center', fontsize=11, color='blue', alpha=0.7)
    ax.set_xlabel('Engineered Coherence C')
    ax.set_ylabel('Mean K_i')
    ax.set_title('Regime Index vs Coherence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(C_levels, mean_s_values, 'r-o', linewidth=2, markersize=4)
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Decay threshold')
    ax.set_xlabel('Engineered Coherence C')
    ax.set_ylabel('Mean Instability Score s')
    ax.set_title('Instability Score vs Coherence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.bar(C_levels, decay_counts, width=0.025, color='red', alpha=0.7)
    ax.set_xlabel('Engineered Coherence C')
    ax.set_ylabel('Nodes with s > 0.6')
    ax.set_title('Decay-Prone Nodes vs Coherence')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(C_levels, mean_O_values, 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Engineered Coherence C')
    ax.set_ylabel('Mean O_i')
    ax.set_title('Observability vs Coherence')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/B2_coherence_engineering.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    return {
        'C_levels': C_levels.tolist(),
        'mean_K': mean_K_values,
        'mean_s': mean_s_values,
        'C_critical': C_critical
    }


# ============================================================
# B3: STRESS-STRAIN RESPONSE
# ============================================================

def experiment_B3_stress_strain():
    """
    B3: Stress-Strain Response Under External Load
    
    Model external mechanical stress as a gradual injection of structural
    debt (q) into the lattice. Compare K-regime vs W-regime materials:
      - Elastic regime: material absorbs stress, K_i stays high
      - Yield point: K_i begins to drop
      - Fracture: K_i collapses, coherence shatters
    """
    print("\n" + "="*70)
    print("B3: Stress-Strain Response (K vs W Materials)")
    print("="*70)
    
    N = 150
    base_params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.5,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.002,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    results = {}
    
    for label, C_init, q_init in [("K-Material (C=0.92)", 0.92, 0.02),
                                    ("Mixed-Material (C=0.55)", 0.55, 0.20),
                                    ("W-Material (C=0.15)", 0.15, 0.50)]:
        np.random.seed(42)
        collider = DETCollider1D(base_params)
        collider.C_R[:] = C_init
        collider.q[:] = q_init
        collider.a[:] = 0.9
        collider.F[:] = 2.0
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams())
        
        # Warm up
        for t in range(100):
            regime.step(t)
        
        # Apply increasing stress (q injection) over 2000 steps
        n_stress_steps = 2000
        stress_rate = np.linspace(0.0, 0.005, n_stress_steps)  # increasing stress
        
        times = []
        applied_stress = []  # cumulative q injected
        mean_K = []
        mean_C = []
        mean_q = []
        min_K = []
        fracture_detected = False
        fracture_step = None
        cumulative_stress = 0.0
        
        for t in range(n_stress_steps):
            # Apply stress: inject q uniformly
            dq = stress_rate[t] * base_params.DT
            collider.q[:] = np.clip(collider.q + dq, 0, 1)
            cumulative_stress += dq * N
            
            diag = regime.step(100 + t)
            
            times.append(t)
            applied_stress.append(cumulative_stress / N)  # per-node average
            mean_K.append(float(np.mean(diag.K)))
            mean_C.append(float(np.mean(collider.C_R)))
            mean_q.append(float(np.mean(collider.q)))
            min_K.append(float(np.min(diag.K)))
            
            # Detect fracture: when mean K drops below 0.1
            if not fracture_detected and np.mean(diag.K) < 0.1:
                fracture_detected = True
                fracture_step = t
        
        results[label] = {
            'times': times,
            'applied_stress': applied_stress,
            'mean_K': mean_K,
            'mean_C': mean_C,
            'mean_q': mean_q,
            'min_K': min_K,
            'fracture_step': fracture_step
        }
        
        print(f"\n  {label}:")
        print(f"    Initial K̄={mean_K[0]:.3f}, Final K̄={mean_K[-1]:.3f}")
        if fracture_step:
            print(f"    Fracture at step {fracture_step} (stress={applied_stress[fracture_step]:.4f})")
        else:
            print(f"    No fracture detected (material survived)")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("B3: Stress-Strain Response — K-Material vs W-Material",
                 fontsize=14, fontweight='bold')
    
    colors = {'K-Material (C=0.92)': 'blue',
              'Mixed-Material (C=0.55)': 'orange',
              'W-Material (C=0.15)': 'red'}
    
    # Stress-strain curve (stress vs K)
    ax = axes[0, 0]
    for label, data in results.items():
        ax.plot(data['applied_stress'], data['mean_K'], color=colors[label],
                linewidth=2, label=label)
        if data['fracture_step']:
            s = data['applied_stress'][data['fracture_step']]
            ax.axvline(x=s, color=colors[label], linestyle=':', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Applied Stress (cumulative q per node)')
    ax.set_ylabel('Mean K_i (Structural Integrity)')
    ax.set_title('Stress-Strain Curve')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # K over time
    ax = axes[0, 1]
    for label, data in results.items():
        ax.plot(data['times'], data['mean_K'], color=colors[label],
                linewidth=2, label=label)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean K_i')
    ax.set_title('Regime Index Under Increasing Load')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Coherence degradation
    ax = axes[1, 0]
    for label, data in results.items():
        ax.plot(data['times'], data['mean_C'], color=colors[label],
                linewidth=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Coherence C')
    ax.set_title('Coherence Degradation Under Stress')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # q accumulation
    ax = axes[1, 1]
    for label, data in results.items():
        ax.plot(data['times'], data['mean_q'], color=colors[label],
                linewidth=2, label=label)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='q=0.5')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Structural Debt q')
    ax.set_title('Debt Accumulation Under Stress')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/B3_stress_strain.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("DET APPLICATION: MATERIALS SCIENCE & STABILITY ENGINEERING")
    print("="*70)
    
    r1 = experiment_B1_self_healing()
    r2 = experiment_B2_coherence_engineering()
    r3 = experiment_B3_stress_strain()
    
    print("\n" + "="*70)
    print("ALL MATERIALS EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
