"""
DET Application: Computing & Information Processing
=====================================================

Models a network of computational nodes using DET primitives.
Demonstrates:
  A1. Coherence-Routed Information Transfer — signals propagate faster and
      with higher fidelity through high-C (K-regime) channels.
  A2. Fault Tolerance via Regime Resilience — K-regime clusters self-heal
      after node failures; W-regime clusters cascade.
  A3. Observability-Gated Error Detection — only high-O nodes can detect
      and correct errors in their neighborhood.

All dynamics use the DET 1D collider as the substrate.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import List, Tuple
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from det_v6_3_1d_collider import DETCollider1D, DETParams1D
from det_concurrent_regimes import (
    DETRegimeSimulator, RegimeParams,
    create_k_region, create_w_region
)

OUTPUT_DIR = "/home/ubuntu/det_app_results/computing"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# A1: COHERENCE-ROUTED INFORMATION TRANSFER
# ============================================================

def experiment_A1_coherence_routing():
    """
    A1: Coherence-Routed Information Transfer
    
    Inject a structured "signal packet" (a localized F-pulse with coherent
    phase) at the center of the lattice.  Measure how far and how cleanly
    the packet propagates through:
      (a) a high-C (K-regime) channel
      (b) a low-C (W-regime) channel
    
    Metrics:
      - Signal reach: how many nodes away the pulse is detectable
      - Fidelity: Ξ^seen at the receiving end
      - Latency: steps until the signal arrives at a target node
    """
    print("\n" + "="*70)
    print("A1: Coherence-Routed Information Transfer")
    print("="*70)
    
    N = 300
    base_params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.005,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    results = {}
    
    for label, C_level, q_level in [("K-channel (high C)", 0.92, 0.02),
                                      ("W-channel (low C)", 0.12, 0.50)]:
        np.random.seed(42)
        collider = DETCollider1D(base_params)
        
        # Set uniform channel properties
        collider.C_R[:] = C_level
        collider.q[:] = q_level
        collider.a[:] = 0.8
        collider.F[:] = 1.0
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams())
        
        # Inject signal pulse at node 150 (center)
        pulse_center = 150
        pulse_width = 5
        for i in range(pulse_center - pulse_width, pulse_center + pulse_width):
            collider.F[i] += 10.0  # strong F-pulse
            regime.theta[i] = 0.0  # coherent phase
        
        # Track signal propagation
        n_steps = 500
        F_history = np.zeros((n_steps, N))
        Xi_seen_history = np.zeros((n_steps, N))
        K_history = np.zeros((n_steps, N))
        
        for t in range(n_steps):
            diag = regime.step(t)
            F_history[t] = collider.F.copy()
            Xi_seen_history[t] = diag.Xi_seen.copy()
            K_history[t] = diag.K.copy()
        
        # Measure signal at target nodes
        targets = [155, 160, 170, 180, 190, 200]
        arrival_times = {}
        peak_fidelity = {}
        
        for tgt in targets:
            # Signal arrival: first time Ξ^seen exceeds 0.05 (structured signal detected)
            xi_trace = Xi_seen_history[:, tgt]
            arrived = np.where(xi_trace > 0.05)[0]
            arrival_times[tgt] = int(arrived[0]) if len(arrived) > 0 else n_steps
            peak_fidelity[tgt] = float(np.max(xi_trace))
        
        results[label] = {
            'F_history': F_history,
            'Xi_seen_history': Xi_seen_history,
            'K_history': K_history,
            'arrival_times': arrival_times,
            'peak_fidelity': peak_fidelity
        }
        
        print(f"\n  {label}:")
        for tgt in targets:
            print(f"    Node {tgt} (dist={tgt-pulse_center}): "
                  f"arrival={arrival_times[tgt]} steps, "
                  f"peak Ξ^seen={peak_fidelity[tgt]:.4f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("A1: Coherence-Routed Information Transfer", fontsize=14, fontweight='bold')
    
    for row, (label, data) in enumerate(results.items()):
        # F propagation heatmap
        ax = axes[row, 0]
        im = ax.imshow(data['F_history'][:300, 100:220].T, aspect='auto',
                       cmap='hot', origin='lower', extent=[0, 300, 100, 220])
        ax.set_ylabel('Position')
        ax.set_title(f'{label}\nResource F Propagation')
        if row == 1:
            ax.set_xlabel('Step')
        plt.colorbar(im, ax=ax, label='F')
        ax.axhline(y=150, color='cyan', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Ξ^seen heatmap
        ax = axes[row, 1]
        im = ax.imshow(data['Xi_seen_history'][:300, 100:220].T, aspect='auto',
                       cmap='viridis', origin='lower', extent=[0, 300, 100, 220],
                       vmin=0, vmax=0.8)
        ax.set_ylabel('Position')
        ax.set_title(f'Perceived Structuredness Ξ^seen')
        if row == 1:
            ax.set_xlabel('Step')
        plt.colorbar(im, ax=ax, label='Ξ^seen')
        
        # Arrival time and fidelity bar chart
        ax = axes[row, 2]
        targets = list(data['arrival_times'].keys())
        distances = [t - 150 for t in targets]
        arrivals = [data['arrival_times'][t] for t in targets]
        fidelities = [data['peak_fidelity'][t] for t in targets]
        
        ax2 = ax.twinx()
        bars1 = ax.bar([d - 1.5 for d in distances], arrivals, width=3,
                       color='steelblue', alpha=0.7, label='Arrival (steps)')
        bars2 = ax2.bar([d + 1.5 for d in distances], fidelities, width=3,
                        color='coral', alpha=0.7, label='Peak Ξ^seen')
        ax.set_xlabel('Distance from source')
        ax.set_ylabel('Arrival Time (steps)', color='steelblue')
        ax2.set_ylabel('Peak Ξ^seen', color='coral')
        ax.set_title('Signal Reach & Fidelity')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/A1_coherence_routing.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    return results


# ============================================================
# A2: FAULT TOLERANCE VIA REGIME RESILIENCE
# ============================================================

def experiment_A2_fault_tolerance():
    """
    A2: Fault Tolerance via Regime Resilience
    
    Create a lattice with a K-regime cluster and a W-regime cluster.
    At step 200, "kill" 10% of nodes in each cluster (set F=0, a=0, q=1).
    
    Measure:
      - Recovery time: how quickly the cluster restores mean K_i
      - Cascade depth: how many additional nodes degrade after the fault
      - Final health: mean K_i after 2000 steps
    """
    print("\n" + "="*70)
    print("A2: Fault Tolerance via Regime Resilience")
    print("="*70)
    
    N = 300
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.005,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    np.random.seed(42)
    collider = DETCollider1D(params)
    
    # K-cluster: nodes 30-120
    create_k_region(collider, center=75, width=45, C_level=0.92, q_level=0.02, F_level=3.0)
    # W-cluster: nodes 180-270
    create_w_region(collider, center=225, width=45, C_level=0.12, q_level=0.55, F_level=0.3)
    
    np.random.seed(42)
    regime = DETRegimeSimulator(collider, RegimeParams())
    
    k_slice = slice(30, 120)
    w_slice = slice(180, 270)
    
    n_steps = 2000
    fault_step = 200
    
    # Track metrics
    times = []
    k_mean_K = []
    w_mean_K = []
    k_mean_F = []
    w_mean_F = []
    k_degraded_count = []
    w_degraded_count = []
    
    # Snapshot pre-fault K_i
    pre_fault_K_k = None
    pre_fault_K_w = None
    
    for t in range(n_steps):
        diag = regime.step(t)
        
        if t == fault_step - 1:
            pre_fault_K_k = diag.K[k_slice].copy()
            pre_fault_K_w = diag.K[w_slice].copy()
        
        if t == fault_step:
            # Kill 10% of nodes in each cluster
            np.random.seed(99)
            k_nodes = np.arange(30, 120)
            w_nodes = np.arange(180, 270)
            
            k_kill = np.random.choice(k_nodes, size=9, replace=False)
            w_kill = np.random.choice(w_nodes, size=9, replace=False)
            
            for idx in k_kill:
                collider.F[idx] = 0.0
                collider.a[idx] = 0.0
                collider.q[idx] = 1.0
                collider.C_R[idx] = 0.0
                if idx > 0:
                    collider.C_R[idx-1] = 0.0
            
            for idx in w_kill:
                collider.F[idx] = 0.0
                collider.a[idx] = 0.0
                collider.q[idx] = 1.0
                collider.C_R[idx] = 0.0
                if idx > 0:
                    collider.C_R[idx-1] = 0.0
            
            print(f"  FAULT INJECTED at step {t}: killed {len(k_kill)} K-nodes, {len(w_kill)} W-nodes")
        
        times.append(t)
        k_mean_K.append(float(np.mean(diag.K[k_slice])))
        w_mean_K.append(float(np.mean(diag.K[w_slice])))
        k_mean_F.append(float(np.mean(collider.F[k_slice])))
        w_mean_F.append(float(np.mean(collider.F[w_slice])))
        
        # Count degraded nodes (K_i < 0.3 in K-cluster, K_i < 0.05 in W-cluster)
        k_degraded_count.append(int(np.sum(diag.K[k_slice] < 0.3)))
        w_degraded_count.append(int(np.sum(diag.K[w_slice] < 0.05)))
    
    # Compute recovery metrics
    k_pre = np.mean(pre_fault_K_k)
    w_pre = np.mean(pre_fault_K_w)
    
    k_post = k_mean_K[-1]
    w_post = w_mean_K[-1]
    
    k_recovery_pct = (k_post / k_pre) * 100 if k_pre > 0 else 0
    w_recovery_pct = (w_post / w_pre) * 100 if w_pre > 0 else 0
    
    # Max cascade depth
    k_max_degraded = max(k_degraded_count[fault_step:])
    w_max_degraded = max(w_degraded_count[fault_step:])
    
    print(f"\n  K-cluster: pre-fault K̄={k_pre:.3f}, post-fault K̄={k_post:.3f}, "
          f"recovery={k_recovery_pct:.1f}%")
    print(f"  W-cluster: pre-fault K̄={w_pre:.3f}, post-fault K̄={w_post:.3f}, "
          f"recovery={w_recovery_pct:.1f}%")
    print(f"  K-cluster max degraded nodes: {k_max_degraded}/90")
    print(f"  W-cluster max degraded nodes: {w_max_degraded}/90")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("A2: Fault Tolerance — K-regime Self-Heals, W-regime Cascades",
                 fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(times, k_mean_K, 'b-', linewidth=1.5, label='K-cluster mean K_i')
    ax.plot(times, w_mean_K, 'r-', linewidth=1.5, label='W-cluster mean K_i')
    ax.axvline(x=fault_step, color='black', linestyle='--', alpha=0.7, label='Fault injected')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Mean K_i')
    ax.set_title('Regime Index Recovery After Fault')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(times, k_mean_F, 'b-', linewidth=1.5, label='K-cluster mean F')
    ax.plot(times, w_mean_F, 'r-', linewidth=1.5, label='W-cluster mean F')
    ax.axvline(x=fault_step, color='black', linestyle='--', alpha=0.7, label='Fault injected')
    ax.set_ylabel('Mean F (Resource)')
    ax.set_title('Resource Recovery After Fault')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(times, k_degraded_count, 'b-', linewidth=1.5, label='K-cluster degraded')
    ax.plot(times, w_degraded_count, 'r-', linewidth=1.5, label='W-cluster degraded')
    ax.axvline(x=fault_step, color='black', linestyle='--', alpha=0.7, label='Fault injected')
    ax.set_xlabel('Step')
    ax.set_ylabel('Degraded Node Count')
    ax.set_title('Cascade Depth (Degraded Nodes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final spatial profile
    ax = axes[1, 1]
    final_K = regime.K.copy()
    x = np.arange(N)
    ax.fill_between(x[k_slice], final_K[k_slice], alpha=0.6, color='blue', label='K-cluster')
    ax.fill_between(x[w_slice], final_K[w_slice], alpha=0.6, color='red', label='W-cluster')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('K_i')
    ax.set_title('Final Regime Profile (t=2000)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/A2_fault_tolerance.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    return {
        'k_recovery_pct': k_recovery_pct,
        'w_recovery_pct': w_recovery_pct,
        'k_max_degraded': k_max_degraded,
        'w_max_degraded': w_max_degraded
    }


# ============================================================
# A3: OBSERVABILITY-GATED ERROR DETECTION
# ============================================================

def experiment_A3_error_detection():
    """
    A3: Observability-Gated Error Detection
    
    Inject random "bit-flip" errors (phase inversions) into a lattice.
    Measure how effectively nodes in K vs W regimes can detect these errors
    using their Ξ^seen readout.
    
    A node "detects" an error if its Ξ^seen drops below a threshold
    within 10 steps of the error injection.
    """
    print("\n" + "="*70)
    print("A3: Observability-Gated Error Detection")
    print("="*70)
    
    N = 200
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.3,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.005,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    np.random.seed(42)
    collider = DETCollider1D(params)
    
    # K-region: left half
    create_k_region(collider, center=50, width=40, C_level=0.92, q_level=0.02, F_level=3.0)
    # W-region: right half
    create_w_region(collider, center=150, width=40, C_level=0.12, q_level=0.55, F_level=0.3)
    
    np.random.seed(42)
    regime = DETRegimeSimulator(collider, RegimeParams())
    
    # Warm up
    for t in range(200):
        regime.step(t)
    
    # Record baseline Ξ^seen
    baseline_Xi = regime.Xi_seen.copy()
    baseline_O = regime.O.copy()
    
    # Inject errors at random positions in both regions
    np.random.seed(123)
    k_error_nodes = np.random.choice(range(20, 80), size=10, replace=False)
    w_error_nodes = np.random.choice(range(120, 180), size=10, replace=False)
    
    error_nodes = np.concatenate([k_error_nodes, w_error_nodes])
    
    # Inject phase flips
    for idx in error_nodes:
        regime.theta[idx] = (regime.theta[idx] + np.pi) % (2 * np.pi)
    
    print(f"  Injected {len(error_nodes)} phase-flip errors")
    print(f"    K-region errors at: {sorted(k_error_nodes)}")
    print(f"    W-region errors at: {sorted(w_error_nodes)}")
    
    # Run 50 steps and track detection
    detection_window = 50
    Xi_seen_traces = {n: [] for n in error_nodes}
    O_traces = {n: [] for n in error_nodes}
    
    # Also track neighbors
    neighbor_detection = {n: [] for n in error_nodes}
    
    for t in range(detection_window):
        diag = regime.step(200 + t)
        for n in error_nodes:
            Xi_seen_traces[n].append(diag.Xi_seen[n])
            O_traces[n].append(diag.O[n])
            # Check if any neighbor detected the anomaly
            neighbors = [max(0, n-1), min(N-1, n+1)]
            neighbor_Xi = np.mean([diag.Xi_seen[nb] for nb in neighbors])
            neighbor_detection[n].append(neighbor_Xi)
    
    # Compute detection metrics
    k_detected = 0
    w_detected = 0
    k_detection_times = []
    w_detection_times = []
    
    # Detection: K-region nodes have high baseline Ξ^seen, so a phase flip
    # causes a measurable drop. W-region nodes have near-zero Ξ^seen baseline,
    # so they cannot detect anything — the error is invisible.
    # Use relative threshold: 20% drop from baseline
    
    for n in k_error_nodes:
        baseline_val = baseline_Xi[n]
        if baseline_val < 0.01:
            continue  # can't detect from zero baseline
        for t, xi in enumerate(Xi_seen_traces[n]):
            if xi < baseline_val * 0.8:  # 20% drop
                k_detected += 1
                k_detection_times.append(t)
                break
    
    for n in w_error_nodes:
        baseline_val = baseline_Xi[n]
        if baseline_val < 0.01:
            continue  # can't detect from zero baseline
        for t, xi in enumerate(Xi_seen_traces[n]):
            if xi < baseline_val * 0.8:
                w_detected += 1
                w_detection_times.append(t)
                break
    
    print(f"\n  K-region: {k_detected}/{len(k_error_nodes)} errors detected")
    if k_detection_times:
        print(f"    Mean detection time: {np.mean(k_detection_times):.1f} steps")
    print(f"  W-region: {w_detected}/{len(w_error_nodes)} errors detected")
    if w_detection_times:
        print(f"    Mean detection time: {np.mean(w_detection_times):.1f} steps")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("A3: Observability-Gated Error Detection",
                 fontsize=14, fontweight='bold')
    
    # Ξ^seen traces for K-region errors
    ax = axes[0, 0]
    for n in k_error_nodes:
        ax.plot(Xi_seen_traces[n], 'b-', alpha=0.4, linewidth=1)
    ax.set_ylabel('Ξ^seen')
    ax.set_title('K-Region Error Nodes: Ξ^seen After Error')
    ax.set_xlabel('Steps after error')
    ax.grid(True, alpha=0.3)
    
    # Ξ^seen traces for W-region errors
    ax = axes[0, 1]
    for n in w_error_nodes:
        ax.plot(Xi_seen_traces[n], 'r-', alpha=0.4, linewidth=1)
    ax.set_ylabel('Ξ^seen')
    ax.set_title('W-Region Error Nodes: Ξ^seen After Error')
    ax.set_xlabel('Steps after error')
    ax.grid(True, alpha=0.3)
    
    # O comparison
    ax = axes[1, 0]
    k_mean_O = np.mean([O_traces[n] for n in k_error_nodes], axis=0)
    w_mean_O = np.mean([O_traces[n] for n in w_error_nodes], axis=0)
    ax.plot(k_mean_O, 'b-', linewidth=2, label='K-region mean O')
    ax.plot(w_mean_O, 'r-', linewidth=2, label='W-region mean O')
    ax.set_ylabel('O_i (Observability)')
    ax.set_xlabel('Steps after error')
    ax.set_title('Observability Gate: K vs W Error Nodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Detection summary bar chart
    ax = axes[1, 1]
    categories = ['K-Region', 'W-Region']
    detected = [k_detected, w_detected]
    total = [len(k_error_nodes), len(w_error_nodes)]
    undetected = [t - d for t, d in zip(total, detected)]
    
    x_pos = np.arange(len(categories))
    ax.bar(x_pos, detected, width=0.4, color=['blue', 'red'], alpha=0.7, label='Detected')
    ax.bar(x_pos, undetected, width=0.4, bottom=detected, color=['lightblue', 'lightsalmon'],
           alpha=0.7, label='Undetected')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Error Count')
    ax.set_title('Error Detection Rate by Regime')
    ax.legend()
    
    # Add mean O annotation
    mean_O_k = float(np.mean(baseline_O[20:80]))
    mean_O_w = float(np.mean(baseline_O[120:180]))
    ax.annotate(f'Mean O={mean_O_k:.2f}', xy=(0, total[0] + 0.3), ha='center', fontsize=10)
    ax.annotate(f'Mean O={mean_O_w:.3f}', xy=(1, total[1] + 0.3), ha='center', fontsize=10)
    
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/A3_error_detection.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    return {
        'k_detected': k_detected,
        'w_detected': w_detected,
        'k_total': len(k_error_nodes),
        'w_total': len(w_error_nodes),
        'mean_O_k': mean_O_k,
        'mean_O_w': mean_O_w
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("DET APPLICATION: COMPUTING & INFORMATION PROCESSING")
    print("="*70)
    
    r1 = experiment_A1_coherence_routing()
    r2 = experiment_A2_fault_tolerance()
    r3 = experiment_A3_error_detection()
    
    print("\n" + "="*70)
    print("ALL COMPUTING EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
