"""
Diagnostic: Why can't local mechanisms lift W→K?

The core issue: attunement feedback is ΔC = η * √(O_i*O_j) * (Ξ_i^seen * Ξ_j^seen) * Δτ
But in W-regime: O_i = a^α * C^β * (1-q)^γ ≈ 0.5^1 * 0.15^1 * 0.5^1 = 0.0375
And Ξ^seen = O * Ξ, so Ξ^seen ≈ 0.04 * Ξ

The product √(O_i*O_j) * Ξ_i^seen * Ξ_j^seen ≈ 0.04 * 0.04^2 ≈ 6e-5
Even with η=1.0, ΔC ≈ 6e-5 * Δτ ≈ 6e-5 * 0.02 ≈ 1.2e-6 per step
Over 5000 steps: total ΔC ≈ 0.006 — negligible.

The question becomes: what LOCAL mechanism in DET CAN change C and q?

Let's trace through the collider step function:
1. C changes via: bond healing (dC_heal = η_heal * g_R * room * D * Δτ)
   - g_R = √(a_i * a_j) — agency gate
   - room = 1 - C_R — room to grow
   - D = dissipation = |J_R| + |J_L| — needs FLOW
   - Δτ — proper time

2. q changes via: q-locking (dq = α_q * max(0, -ΔF))
   - q only INCREASES (from resource loss)
   - q NEVER decreases in the standard model!

THIS IS THE KEY INSIGHT: q has no decay term. Once q is high, it stays high.
The only way q decreases is if it was never high to begin with.

So the real question is: what can REDUCE q? In the current model, nothing.
But grace flux can inject F, which prevents further q accumulation.
And bond healing can raise C, which requires dissipation (flow).

Let's test with modified dynamics:
1. What if q had a slow natural decay? (q → q * (1 - λ_q * Δτ))
2. What if grace could heal q directly? (Δq = -η_q * grace_flux)
3. What if high-agency nodes could locally reduce their own q?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from det_v6_3_1d_collider import DETCollider1D, DETParams1D
from det_concurrent_regimes import DETRegimeSimulator, RegimeParams

OUTPUT_DIR = "/home/ubuntu/det_app_results/local_mechanisms"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def experiment_D1_diagnose_bottleneck():
    """
    D1: Diagnose exactly which variable is the bottleneck.
    
    Start a W-regime node and trace all the intermediate quantities
    that feed into K, O, attunement, healing, and grace.
    """
    print("\n" + "="*70)
    print("D1: Bottleneck Diagnosis — Why Can't W Escape?")
    print("="*70)
    
    N = 50
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
    collider.C_R[:] = 0.15
    collider.q[:] = 0.50
    collider.F[:] = 0.5
    collider.a[:] = 0.8  # high agency
    
    np.random.seed(42)
    regime = DETRegimeSimulator(collider, RegimeParams(
        attunement_enabled=True, eta_attune=1.0  # maximum attunement
    ))
    
    # Trace node 25
    node = 25
    traces = {k: [] for k in ['K', 'O', 'Xi', 'Xi_seen', 'C', 'q', 'a', 'F', 'P',
                                'grace', 'healing', 'dissipation', 'a_max']}
    
    for t in range(2000):
        diag = regime.step(t)
        
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        
        traces['K'].append(float(diag.K[node]))
        traces['O'].append(float(diag.O[node]))
        traces['Xi'].append(float(diag.Xi[node]))
        traces['Xi_seen'].append(float(diag.Xi_seen[node]))
        traces['C'].append(float(collider.C_R[node]))
        traces['q'].append(float(collider.q[node]))
        traces['a'].append(float(collider.a[node]))
        traces['F'].append(float(collider.F[node]))
        traces['P'].append(float(collider.P[node]))
        traces['grace'].append(float(collider.last_grace_injection[node]))
        traces['healing'].append(float(collider.last_healing[node]))
        
        # Dissipation
        D = np.abs(collider.pi_R)  # proxy for flow
        traces['dissipation'].append(float(D[node]))
        
        # Agency ceiling
        a_max = 1.0 / (1.0 + params.lambda_a * collider.q[node]**2)
        traces['a_max'].append(a_max)
    
    # Print final state
    print(f"  Node {node} final state:")
    for k, v in traces.items():
        print(f"    {k:15s}: {v[-1]:.6f}")
    
    # Compute the attunement increment
    O_i = traces['O'][-1]
    Xi_seen = traces['Xi_seen'][-1]
    dC_attune = 1.0 * O_i * Xi_seen**2 * 0.02  # η * O * Ξ² * Δτ
    print(f"\n  Attunement increment per step: {dC_attune:.2e}")
    print(f"  Steps needed for ΔC=0.1: {0.1/max(dC_attune, 1e-20):.0f}")
    
    # THE KEY: a_max with q=0.5
    a_max_at_q05 = 1.0 / (1.0 + 30.0 * 0.5**2)
    print(f"\n  Agency ceiling at q=0.5: a_max = {a_max_at_q05:.4f}")
    print(f"  Agency ceiling at q=0.3: a_max = {1.0/(1+30*0.3**2):.4f}")
    print(f"  Agency ceiling at q=0.1: a_max = {1.0/(1+30*0.1**2):.4f}")
    print(f"  Agency ceiling at q=0.0: a_max = {1.0/(1+30*0.0**2):.4f}")
    
    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("D1: Bottleneck Diagnosis — Why Can't W Escape?",
                 fontsize=14, fontweight='bold')
    
    plot_items = [
        ('K', 'Regime Index K', 'green'),
        ('O', 'Observability O', 'blue'),
        ('Xi_seen', 'Perceived Structuredness Ξ^seen', 'purple'),
        ('C', 'Coherence C', 'cyan'),
        ('q', 'Structural Debt q', 'red'),
        ('a', 'Agency a', 'orange'),
        ('F', 'Resource F', 'brown'),
        ('grace', 'Grace Injection', 'gold'),
        ('healing', 'Bond Healing ΔC', 'lime'),
    ]
    
    for ax, (key, title, color) in zip(axes.flat, plot_items):
        ax.plot(traces[key], color=color, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Step')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/D1_bottleneck.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/D1_bottleneck.png")
    
    return traces


def experiment_D2_q_decay_hypothesis():
    """
    D2: What if q had a natural decay mechanism?
    
    In the current model, q only increases. But what if:
    - Grace could reduce q (grace heals damage)
    - High agency could slowly reduce q (active repair)
    - Natural q decay existed (time heals all wounds)
    
    Test each hypothesis by manually applying q reduction
    after each step and measuring the effect on K.
    """
    print("\n" + "="*70)
    print("D2: q-Decay Hypothesis — What If q Could Decrease?")
    print("="*70)
    
    N = 100
    n_steps = 3000
    
    # K-region adjacent to W-region
    K_center = 25
    K_width = 15
    x = np.arange(N)
    K_mask = np.abs(x - K_center) < K_width
    W_mask = ~K_mask
    
    hypotheses = {
        'No q-decay (baseline)': {'mode': 'none'},
        'Slow natural q-decay (λ=0.0001)': {'mode': 'natural', 'rate': 0.0001},
        'Agency-gated q-decay (λ=0.001)': {'mode': 'agency', 'rate': 0.001},
        'Grace-driven q-repair': {'mode': 'grace', 'rate': 0.01},
        'C-gated q-repair (λ=0.001)': {'mode': 'coherence', 'rate': 0.001},
    }
    
    results = {}
    
    for name, cfg in hypotheses.items():
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
        
        collider.C_R[:] = 0.15; collider.q[:] = 0.50
        collider.F[:] = 0.5; collider.a[:] = 0.5
        
        collider.C_R[K_mask] = 0.90; collider.q[K_mask] = 0.02
        collider.F[K_mask] = 3.0; collider.a[K_mask] = 0.95
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams(
            attunement_enabled=True, eta_attune=0.10
        ))
        
        K_trace = []
        q_trace = []
        C_trace = []
        
        for t in range(n_steps):
            diag = regime.step(t)
            
            # Apply q-decay hypothesis (only to W-region)
            if cfg['mode'] == 'natural':
                collider.q[W_mask] *= (1 - cfg['rate'])
            elif cfg['mode'] == 'agency':
                # Agency-gated: high-a nodes repair faster
                dq = cfg['rate'] * collider.a[W_mask] * collider.Delta_tau[W_mask]
                collider.q[W_mask] = np.clip(collider.q[W_mask] - dq, 0, 1)
            elif cfg['mode'] == 'grace':
                # Grace-driven: grace injection also heals q
                grace = collider.last_grace_injection[W_mask]
                dq = cfg['rate'] * grace
                collider.q[W_mask] = np.clip(collider.q[W_mask] - dq, 0, 1)
            elif cfg['mode'] == 'coherence':
                # C-gated: coherence enables q repair
                C_avg = 0.5 * (collider.C_R[W_mask] + np.roll(collider.C_R, 1)[W_mask])
                dq = cfg['rate'] * C_avg * collider.a[W_mask] * collider.Delta_tau[W_mask]
                collider.q[W_mask] = np.clip(collider.q[W_mask] - dq, 0, 1)
            
            K_trace.append(float(np.mean(diag.K[W_mask])))
            q_trace.append(float(np.mean(collider.q[W_mask])))
            C_trace.append(float(np.mean(collider.C_R[W_mask])))
        
        results[name] = {
            'K': np.array(K_trace),
            'q': np.array(q_trace),
            'C': np.array(C_trace),
            'final_K_profile': diag.K.copy(),
            'final_q_profile': collider.q.copy(),
            'final_C_profile': collider.C_R.copy()
        }
        print(f"  {name:40s}: K={K_trace[-1]:.4f}, q={q_trace[-1]:.4f}, C={C_trace[-1]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("D2: q-Decay Hypothesis — What If q Could Decrease?",
                 fontsize=14, fontweight='bold')
    
    cmap = plt.cm.viridis
    
    for i, (name, data) in enumerate(results.items()):
        color = cmap(i / max(1, len(results) - 1))
        axes[0,0].plot(data['K'], color=color, linewidth=2, label=name)
        axes[0,1].plot(data['q'], color=color, linewidth=2, label=name)
        axes[0,2].plot(data['C'], color=color, linewidth=2, label=name)
    
    axes[0,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[0,0].set_title('K in W-Region'); axes[0,0].set_ylabel('K')
    axes[0,1].set_title('q in W-Region'); axes[0,1].set_ylabel('q')
    axes[0,2].set_title('C in W-Region'); axes[0,2].set_ylabel('C')
    
    for ax in axes[0]:
        ax.set_xlabel('Step')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
    
    # Spatial profiles for best hypothesis
    best_name = max(results.keys(), key=lambda k: results[k]['K'][-1])
    best = results[best_name]
    
    axes[1,0].plot(best['final_K_profile'], 'g-', linewidth=2)
    axes[1,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1,0].axvspan(K_center-K_width, K_center+K_width, alpha=0.1, color='green')
    axes[1,0].set_title(f'Final K Profile ({best_name})')
    axes[1,0].set_xlabel('Node')
    
    axes[1,1].plot(best['final_q_profile'], 'r-', linewidth=2)
    axes[1,1].axvspan(K_center-K_width, K_center+K_width, alpha=0.1, color='green')
    axes[1,1].set_title(f'Final q Profile ({best_name})')
    axes[1,1].set_xlabel('Node')
    
    axes[1,2].plot(best['final_C_profile'], 'b-', linewidth=2)
    axes[1,2].axvspan(K_center-K_width, K_center+K_width, alpha=0.1, color='green')
    axes[1,2].set_title(f'Final C Profile ({best_name})')
    axes[1,2].set_xlabel('Node')
    
    for ax in axes[1]:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/D2_q_decay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/D2_q_decay.png")
    return results


def experiment_D3_enhanced_healing():
    """
    D3: Enhanced Bond Healing
    
    The current healing formula: dC = η_heal * √(a_i*a_j) * (1-C) * D * Δτ
    
    In W-regime, D (dissipation) is very low because flow is low.
    What if we enhance healing via:
    1. Higher η_heal
    2. Grace-driven healing (grace injection also heals bonds)
    3. Agency-driven healing (high-a nodes can heal bonds without dissipation)
    """
    print("\n" + "="*70)
    print("D3: Enhanced Bond Healing — Can Stronger Healing Break Through?")
    print("="*70)
    
    N = 100
    n_steps = 3000
    K_center = 25; K_width = 15
    x = np.arange(N)
    K_mask = np.abs(x - K_center) < K_width
    W_mask = ~K_mask
    
    configs = {
        'Baseline (η=0.03)': {'eta_heal': 0.03, 'extra_heal': 0.0, 'q_decay': 0.0},
        'Strong healing (η=0.30)': {'eta_heal': 0.30, 'extra_heal': 0.0, 'q_decay': 0.0},
        'Very strong healing (η=1.0)': {'eta_heal': 1.0, 'extra_heal': 0.0, 'q_decay': 0.0},
        'Strong heal + q-decay': {'eta_heal': 0.30, 'extra_heal': 0.0, 'q_decay': 0.0005},
        'Strong heal + agency heal + q-decay': {'eta_heal': 0.30, 'extra_heal': 0.01, 'q_decay': 0.0005},
    }
    
    results = {}
    
    for name, cfg in configs.items():
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.15,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False,
            boundary_enabled=True, grace_enabled=True,
            F_MIN_grace=0.05, healing_enabled=True, eta_heal=cfg['eta_heal']
        )
        collider = DETCollider1D(params)
        
        collider.C_R[:] = 0.15; collider.q[:] = 0.50
        collider.F[:] = 0.5; collider.a[:] = 0.5
        
        collider.C_R[K_mask] = 0.90; collider.q[K_mask] = 0.02
        collider.F[K_mask] = 3.0; collider.a[K_mask] = 0.95
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams(
            attunement_enabled=True, eta_attune=0.10
        ))
        
        K_trace = []
        C_trace = []
        q_trace = []
        
        for t in range(n_steps):
            diag = regime.step(t)
            
            # Extra agency-driven healing
            if cfg['extra_heal'] > 0:
                dC_extra = cfg['extra_heal'] * collider.a[W_mask] * (1 - collider.C_R[W_mask]) * collider.Delta_tau[W_mask]
                collider.C_R[W_mask] = np.clip(collider.C_R[W_mask] + dC_extra, 0, 1)
            
            # q-decay
            if cfg['q_decay'] > 0:
                collider.q[W_mask] *= (1 - cfg['q_decay'])
            
            K_trace.append(float(np.mean(diag.K[W_mask])))
            C_trace.append(float(np.mean(collider.C_R[W_mask])))
            q_trace.append(float(np.mean(collider.q[W_mask])))
        
        results[name] = {
            'K': np.array(K_trace),
            'C': np.array(C_trace),
            'q': np.array(q_trace)
        }
        print(f"  {name:45s}: K={K_trace[-1]:.4f}, C={C_trace[-1]:.4f}, q={q_trace[-1]:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("D3: Enhanced Bond Healing — Can Stronger Healing Break Through?",
                 fontsize=14, fontweight='bold')
    
    cmap = plt.cm.plasma
    for i, (name, data) in enumerate(results.items()):
        color = cmap(i / max(1, len(results) - 1))
        axes[0].plot(data['K'], color=color, linewidth=2, label=name)
        axes[1].plot(data['C'], color=color, linewidth=2, label=name)
        axes[2].plot(data['q'], color=color, linewidth=2, label=name)
    
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('K in W-Region'); axes[0].set_ylabel('K')
    axes[1].set_title('C in W-Region'); axes[1].set_ylabel('C')
    axes[2].set_title('q in W-Region'); axes[2].set_ylabel('q')
    
    for ax in axes:
        ax.set_xlabel('Step')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/D3_enhanced_healing.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/D3_enhanced_healing.png")
    return results


if __name__ == "__main__":
    print("="*70)
    print("DIAGNOSTIC: WHY CAN'T LOCAL MECHANISMS LIFT W→K?")
    print("="*70)
    
    d1 = experiment_D1_diagnose_bottleneck()
    d2 = experiment_D2_q_decay_hypothesis()
    d3 = experiment_D3_enhanced_healing()
    
    print("\n" + "="*70)
    print("ALL DIAGNOSTICS COMPLETE")
    print("="*70)
