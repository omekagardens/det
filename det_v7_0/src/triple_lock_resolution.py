"""
DET v6.5: Triple Lock Resolution Experiment
=============================================

Re-runs the W→K local transition experiments from the previous analysis
with the Jubilee/Forgiveness operator enabled, to verify that the
"Triple Lock" (q-ratchet, observability gate, stagnation loop) is broken.

Experiments:
  R1: Jubilee-only (no healing, no grace) — does q-decay alone break the lock?
  R2: Jubilee + healing — does coherence propagation + q-decay create a cascade?
  R3: Jubilee + healing + grace — full operator complement
  R4: Dose-response curve — Jubilee rate sweep
  R5: Long-run convergence — does the W-region fully transition to K?
  R6: Comparison: all 16 mechanism combos from original study, now with Jubilee

Output: Plots and diagnostics to /home/ubuntu/det_app_results/triple_lock/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from det_v6_3_1d_collider import DETCollider1D, DETParams1D
from det_concurrent_regimes import DETRegimeSimulator, RegimeParams

OUT = "/home/ubuntu/det_app_results/triple_lock"
os.makedirs(OUT, exist_ok=True)


def setup_KW_system(params, N=200, K_center=50, K_width=30):
    """Create a K-region adjacent to a W-region."""
    sim = DETCollider1D(params)
    x = np.arange(N)
    K_mask = np.abs(x - K_center) < K_width
    W_mask = ~K_mask

    # K-region: healthy
    sim.C_R[K_mask] = 0.85
    sim.q[K_mask] = 0.02
    sim.q_D[K_mask] = 0.02
    sim.q_I[K_mask] = 0.0
    sim.F[K_mask] = 3.0
    sim.a[K_mask] = 0.95

    # W-region: damaged
    sim.C_R[W_mask] = 0.15
    sim.q[W_mask] = 0.50
    sim.q_D[W_mask] = 0.50
    sim.q_I[W_mask] = 0.0
    sim.F[W_mask] = 0.5
    sim.a[W_mask] = 0.5

    # Gradient at boundary
    for i in range(K_center - K_width - 5, K_center - K_width + 5):
        if 0 <= i < N:
            frac = (i - (K_center - K_width - 5)) / 10.0
            sim.C_R[i] = 0.15 * (1 - frac) + 0.85 * frac
            sim.q_D[i] = 0.50 * (1 - frac) + 0.02 * frac
            sim.q[i] = sim.q_D[i]
            sim.F[i] = 0.5 * (1 - frac) + 3.0 * frac
    for i in range(K_center + K_width - 5, K_center + K_width + 5):
        if 0 <= i < N:
            frac = (i - (K_center + K_width - 5)) / 10.0
            sim.C_R[i] = 0.85 * (1 - frac) + 0.15 * frac
            sim.q_D[i] = 0.02 * (1 - frac) + 0.50 * frac
            sim.q[i] = sim.q_D[i]
            sim.F[i] = 3.0 * (1 - frac) + 0.5 * frac

    return sim, K_mask, W_mask


def run_experiment(sim, K_mask, W_mask, steps=20000, label=""):
    """Run simulation and track K, q_D, C, a in W-region."""
    regime = DETRegimeSimulator(sim, RegimeParams(
        attunement_enabled=True, eta_attune=0.10
    ))

    trace = {'t': [], 'K_W': [], 'qD_W': [], 'C_W': [], 'a_W': [],
             'K_K': [], 'jubilee_total': []}

    for t in range(steps):
        diag = regime.step(t)
        if t % 100 == 0:
            trace['t'].append(t)
            trace['K_W'].append(float(np.mean(diag.K[W_mask])))
            trace['qD_W'].append(float(np.mean(sim.q_D[W_mask])))
            trace['C_W'].append(float(np.mean(sim.C_R[W_mask])))
            trace['a_W'].append(float(np.mean(sim.a[W_mask])))
            trace['K_K'].append(float(np.mean(diag.K[K_mask])))
            trace['jubilee_total'].append(sim.total_jubilee)

    return trace


# ============================================================
# R1: Jubilee-only (no healing, no grace)
# ============================================================
print("=" * 60)
print("R1: Jubilee-only (no healing, no grace)")
print("=" * 60)

N = 200
results_R1 = {}

for jub_on in [False, True]:
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, C_init=0.15,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.001,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=False,  # No grace, no healing
        jubilee_enabled=jub_on, delta_q=0.10, n_q=1, D_0=0.01
    )
    sim, K_mask, W_mask = setup_KW_system(params, N)
    label = "Jubilee ON" if jub_on else "Jubilee OFF"
    trace = run_experiment(sim, K_mask, W_mask, steps=20000, label=label)
    results_R1[label] = trace
    print(f"  {label}: final K_W={trace['K_W'][-1]:.4f}, qD_W={trace['qD_W'][-1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("R1: Jubilee-Only (No Healing, No Grace)", fontsize=14, fontweight='bold')

for label, trace in results_R1.items():
    ls = '-' if 'ON' in label else '--'
    axes[0, 0].plot(trace['t'], trace['K_W'], ls, label=label)
    axes[0, 1].plot(trace['t'], trace['qD_W'], ls, label=label)
    axes[1, 0].plot(trace['t'], trace['C_W'], ls, label=label)
    axes[1, 1].plot(trace['t'], trace['a_W'], ls, label=label)

for ax, title in zip(axes.flat, ['K (Regime Index) in W', 'q_D (Damage Debt) in W',
                                   'C (Coherence) in W', 'a (Agency) in W']):
    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/R1_jubilee_only.png", dpi=150)
plt.close()


# ============================================================
# R2: Jubilee + Healing
# ============================================================
print("\n" + "=" * 60)
print("R2: Jubilee + Healing")
print("=" * 60)

results_R2 = {}

for config_name, jub, heal in [("Neither", False, False), ("Healing only", False, True),
                                 ("Jubilee only", True, False), ("Jubilee + Healing", True, True)]:
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, C_init=0.15,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.001,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True, grace_enabled=True, F_MIN_grace=0.05,
        healing_enabled=heal, eta_heal=0.10,
        jubilee_enabled=jub, delta_q=0.10, n_q=1, D_0=0.01
    )
    sim, K_mask, W_mask = setup_KW_system(params, N)
    trace = run_experiment(sim, K_mask, W_mask, steps=20000)
    results_R2[config_name] = trace
    print(f"  {config_name}: final K_W={trace['K_W'][-1]:.4f}, qD_W={trace['qD_W'][-1]:.4f}, "
          f"C_W={trace['C_W'][-1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("R2: Jubilee + Healing Combinations", fontsize=14, fontweight='bold')

styles = {'Neither': ('--', 'gray'), 'Healing only': ('-.', 'blue'),
          'Jubilee only': (':', 'orange'), 'Jubilee + Healing': ('-', 'green')}

for label, trace in results_R2.items():
    ls, color = styles[label]
    axes[0, 0].plot(trace['t'], trace['K_W'], ls, color=color, label=label)
    axes[0, 1].plot(trace['t'], trace['qD_W'], ls, color=color, label=label)
    axes[1, 0].plot(trace['t'], trace['C_W'], ls, color=color, label=label)
    axes[1, 1].plot(trace['t'], trace['a_W'], ls, color=color, label=label)

for ax, title in zip(axes.flat, ['K (Regime Index) in W', 'q_D (Damage Debt) in W',
                                   'C (Coherence) in W', 'a (Agency) in W']):
    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/R2_jubilee_healing.png", dpi=150)
plt.close()


# ============================================================
# R3: Full operator complement
# ============================================================
print("\n" + "=" * 60)
print("R3: Full Operator Complement (Jubilee + Healing + Grace)")
print("=" * 60)

results_R3 = {}

for config_name, jub in [("v6.3 (no Jubilee)", False), ("v6.5 (with Jubilee)", True)]:
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, C_init=0.15,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.001,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True, grace_enabled=True, F_MIN_grace=0.05,
        healing_enabled=True, eta_heal=0.10,
        jubilee_enabled=jub, delta_q=0.10, n_q=1, D_0=0.01
    )
    sim, K_mask, W_mask = setup_KW_system(params, N)
    trace = run_experiment(sim, K_mask, W_mask, steps=20000)
    results_R3[config_name] = trace
    print(f"  {config_name}: final K_W={trace['K_W'][-1]:.4f}, qD_W={trace['qD_W'][-1]:.4f}, "
          f"C_W={trace['C_W'][-1]:.4f}, a_W={trace['a_W'][-1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("R3: v6.3 vs v6.5 — Full Operator Complement", fontsize=14, fontweight='bold')

for label, trace in results_R3.items():
    ls = '-' if 'v6.5' in label else '--'
    color = 'green' if 'v6.5' in label else 'red'
    axes[0, 0].plot(trace['t'], trace['K_W'], ls, color=color, label=label, linewidth=2)
    axes[0, 1].plot(trace['t'], trace['qD_W'], ls, color=color, label=label, linewidth=2)
    axes[1, 0].plot(trace['t'], trace['C_W'], ls, color=color, label=label, linewidth=2)
    axes[1, 1].plot(trace['t'], trace['a_W'], ls, color=color, label=label, linewidth=2)

for ax, title in zip(axes.flat, ['K (Regime Index) in W', 'q_D (Damage Debt) in W',
                                   'C (Coherence) in W', 'a (Agency) in W']):
    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/R3_full_complement.png", dpi=150)
plt.close()


# ============================================================
# R4: Dose-response curve (Jubilee rate sweep)
# ============================================================
print("\n" + "=" * 60)
print("R4: Dose-Response Curve (delta_q sweep)")
print("=" * 60)

delta_q_values = [0.0, 0.001, 0.005, 0.01, 0.05, 0.10, 0.50, 1.0, 5.0]
final_K_W = []
final_qD_W = []

for dq in delta_q_values:
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, C_init=0.15,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.001,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True, grace_enabled=True, F_MIN_grace=0.05,
        healing_enabled=True, eta_heal=0.10,
        jubilee_enabled=(dq > 0), delta_q=max(dq, 0.001), n_q=1, D_0=0.01
    )
    sim, K_mask, W_mask = setup_KW_system(params, N)
    regime = DETRegimeSimulator(sim, RegimeParams(attunement_enabled=True, eta_attune=0.10))

    for t in range(20000):
        diag = regime.step(t)

    K_W = float(np.mean(diag.K[W_mask]))
    qD_W = float(np.mean(sim.q_D[W_mask]))
    final_K_W.append(K_W)
    final_qD_W.append(qD_W)
    print(f"  delta_q={dq:.3f}: K_W={K_W:.4f}, qD_W={qD_W:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("R4: Dose-Response Curve (Jubilee Rate Sweep)", fontsize=14, fontweight='bold')

ax1.semilogx([max(d, 1e-4) for d in delta_q_values], final_K_W, 'o-', color='green', linewidth=2)
ax1.set_xlabel('δ_q (Jubilee Rate)')
ax1.set_ylabel('Final K in W-region')
ax1.set_title('Regime Index vs Jubilee Rate')
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='K=0.5 threshold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.semilogx([max(d, 1e-4) for d in delta_q_values], final_qD_W, 'o-', color='orange', linewidth=2)
ax2.set_xlabel('δ_q (Jubilee Rate)')
ax2.set_ylabel('Final q_D in W-region')
ax2.set_title('Damage Debt vs Jubilee Rate')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/R4_dose_response.png", dpi=150)
plt.close()


# ============================================================
# R5: Long-run convergence
# ============================================================
print("\n" + "=" * 60)
print("R5: Long-Run Convergence (50,000 steps)")
print("=" * 60)

np.random.seed(42)
params = DETParams1D(
    N=N, DT=0.02, F_VAC=0.01, C_init=0.15,
    momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
    q_enabled=True, alpha_q=0.001,
    lambda_a=30.0, beta_a=0.2,
    floor_enabled=True, eta_floor=0.15, F_core=5.0,
    gravity_enabled=False,
    boundary_enabled=True, grace_enabled=True, F_MIN_grace=0.05,
    healing_enabled=True, eta_heal=0.10,
    jubilee_enabled=True, delta_q=1.0, n_q=1, D_0=0.01
)
sim, K_mask, W_mask = setup_KW_system(params, N)
trace_R5 = run_experiment(sim, K_mask, W_mask, steps=50000)

print(f"  Final K_W={trace_R5['K_W'][-1]:.4f}, qD_W={trace_R5['qD_W'][-1]:.4f}, "
      f"C_W={trace_R5['C_W'][-1]:.4f}, a_W={trace_R5['a_W'][-1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("R5: Long-Run Convergence (50,000 steps, δ_q=1.0)", fontsize=14, fontweight='bold')

axes[0, 0].plot(trace_R5['t'], trace_R5['K_W'], '-', color='green', linewidth=2)
axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_title('K (Regime Index) in W')

axes[0, 1].plot(trace_R5['t'], trace_R5['qD_W'], '-', color='orange', linewidth=2)
axes[0, 1].set_title('q_D (Damage Debt) in W')

axes[1, 0].plot(trace_R5['t'], trace_R5['C_W'], '-', color='blue', linewidth=2)
axes[1, 0].set_title('C (Coherence) in W')

axes[1, 1].plot(trace_R5['t'], trace_R5['a_W'], '-', color='purple', linewidth=2)
axes[1, 1].set_title('a (Agency) in W')

for ax in axes.flat:
    ax.set_xlabel('Step')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/R5_long_run.png", dpi=150)
plt.close()


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("TRIPLE LOCK RESOLUTION SUMMARY")
print("=" * 60)
print(f"  R1 (Jubilee-only):     K_W OFF={results_R1['Jubilee OFF']['K_W'][-1]:.4f}, "
      f"ON={results_R1['Jubilee ON']['K_W'][-1]:.4f}")
print(f"  R2 (Best combo):       K_W={results_R2['Jubilee + Healing']['K_W'][-1]:.4f}")
print(f"  R3 (v6.3 vs v6.5):    v6.3={results_R3['v6.3 (no Jubilee)']['K_W'][-1]:.4f}, "
      f"v6.5={results_R3['v6.5 (with Jubilee)']['K_W'][-1]:.4f}")
print(f"  R4 (Best dose):        K_W={max(final_K_W):.4f} at δ_q={delta_q_values[final_K_W.index(max(final_K_W))]}")
print(f"  R5 (Long-run):         K_W={trace_R5['K_W'][-1]:.4f}, qD_W={trace_R5['qD_W'][-1]:.4f}")
print(f"\n  All plots saved to {OUT}/")
