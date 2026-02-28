"""
Generate Triple Lock resolution plots from the q_D sweep data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = "/home/ubuntu/det_app_results/triple_lock"
os.makedirs(OUT, exist_ok=True)

# Data from the q_D sweep (10,000 steps, delta_q=1.0, n_q=1, D_0=0.01)
qD_init = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
a_max_W = [0.7692, 0.5970, 0.4545, 0.3478, 0.2703, 0.2139, 0.1724, 0.1413, 0.1176]
final_qD = [0.0409, 0.0821, 0.1378, 0.1997, 0.2543, 0.3014, 0.3462, 0.3901, 0.4336]
delta_qD = [0.0591, 0.0679, 0.0622, 0.0503, 0.0457, 0.0486, 0.0538, 0.0599, 0.0664]
final_K_W = [0.4886, 0.4402, 0.3822, 0.3256, 0.2809, 0.2503, 0.2259, 0.2051, 0.1867]
total_jub = [5.0787, 3.7439, 2.4779, 1.6171, 1.1353, 0.8712, 0.7148, 0.6137, 0.5452]

# Baseline (no Jubilee): K_W ≈ 0.107 for all q_D levels
baseline_K = 0.107

# Plot 1: Triple Lock Resolution Overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DET v6.5: Triple Lock Resolution — Jubilee Operator Impact",
             fontsize=14, fontweight='bold')

# q_D reduction
ax = axes[0, 0]
ax.bar(range(len(qD_init)), qD_init, alpha=0.3, color='red', label='Initial q_D')
ax.bar(range(len(qD_init)), final_qD, alpha=0.7, color='green', label='Final q_D')
ax.set_xticks(range(len(qD_init)))
ax.set_xticklabels([f'{q:.2f}' for q in qD_init])
ax.set_xlabel('Initial q_D')
ax.set_ylabel('q_D')
ax.set_title('Damage Debt Reduction by Jubilee')
ax.legend()
ax.grid(True, alpha=0.3)

# K_W improvement
ax = axes[0, 1]
ax.plot(qD_init, final_K_W, 'o-', color='green', linewidth=2, markersize=8, label='With Jubilee')
ax.axhline(y=baseline_K, color='red', linestyle='--', linewidth=2, label=f'Without Jubilee (K≈{baseline_K})')
ax.axhline(y=0.5, color='blue', linestyle=':', alpha=0.5, label='K=0.5 (K-regime threshold)')
ax.set_xlabel('Initial q_D')
ax.set_ylabel('Final K in W-boundary')
ax.set_title('Regime Index Improvement')
ax.legend()
ax.grid(True, alpha=0.3)

# Percentage reduction
ax = axes[1, 0]
pct_reduction = [d/q * 100 for d, q in zip(delta_qD, qD_init)]
ax.bar(range(len(qD_init)), pct_reduction, color='teal', alpha=0.7)
ax.set_xticks(range(len(qD_init)))
ax.set_xticklabels([f'{q:.2f}' for q in qD_init])
ax.set_xlabel('Initial q_D')
ax.set_ylabel('% q_D Reduction')
ax.set_title('Percentage Damage Healed')
ax.grid(True, alpha=0.3)

# Total Jubilee applied
ax = axes[1, 1]
ax.plot(qD_init, total_jub, 's-', color='purple', linewidth=2, markersize=8)
ax.set_xlabel('Initial q_D')
ax.set_ylabel('Total Jubilee Applied')
ax.set_title('Jubilee Activity vs Initial Damage')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/R_triple_lock_resolution.png", dpi=150)
plt.close()

# Plot 2: The Cascade Mechanism
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle("The Jubilee Cascade: How the Triple Lock Breaks",
             fontsize=14, fontweight='bold')

# Show the feedback loop: lower q_D → higher a_max → more flow → more Jubilee
final_a_max = [1.0 / (1.0 + 30.0 * q**2) for q in final_qD]

ax.plot(qD_init, a_max_W, 'o--', color='red', linewidth=2, label='Initial a_max (locked)')
ax.plot(qD_init, final_a_max, 's-', color='green', linewidth=2, label='Final a_max (after Jubilee)')
ax.fill_between(qD_init, a_max_W, final_a_max, alpha=0.2, color='green')
ax.set_xlabel('Initial q_D', fontsize=12)
ax.set_ylabel('Agency Ceiling (a_max)', fontsize=12)
ax.set_title('Agency Ceiling Recovery via Jubilee', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/R_cascade_mechanism.png", dpi=150)
plt.close()

# Plot 3: K improvement factor
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
improvement_factor = [k / baseline_K for k in final_K_W]
ax.bar(range(len(qD_init)), improvement_factor, color='green', alpha=0.7)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (no Jubilee)')
ax.set_xticks(range(len(qD_init)))
ax.set_xticklabels([f'{q:.2f}' for q in qD_init])
ax.set_xlabel('Initial q_D', fontsize=12)
ax.set_ylabel('K Improvement Factor (vs baseline)', fontsize=12)
ax.set_title('DET v6.5 Jubilee: Regime Index Improvement Factor', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/R_improvement_factor.png", dpi=150)
plt.close()

print("All Triple Lock resolution plots saved.")
print(f"Output directory: {OUT}")
