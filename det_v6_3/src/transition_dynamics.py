"""
DET Regime Transition Dynamics: W → K
=======================================

High-resolution characterization of the W-to-K regime transition.

Questions addressed:
  T1. DRIVEN TRANSITION SHAPE — When coherence is injected at a constant rate,
      does K_i rise linearly, exponentially, or with a sigmoidal phase-change?
  T2. DERIVATIVE ANALYSIS — Where is dK/dt maximal? Is there an inflection
      point (acceleration → deceleration)? A singularity-like spike?
  T3. SPONTANEOUS TRANSITION — Can a W-regime spontaneously cross to K
      without external driving? If so, what triggers it?
  T4. HYSTERESIS — Is the W→K transition reversible? Does K→W follow the
      same path, or is there hysteresis (different forward/reverse paths)?
  T5. MULTI-VARIABLE PHASE PORTRAIT — Map the transition in (C, q, K) space
      to reveal the full topology of the transition surface.
  T6. CRITICAL SLOWING DOWN — Near the transition point, does the system
      exhibit critical slowing (longer relaxation times)?

All dynamics use the DET 1D collider + concurrent regimes simulator.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from det_v6_3_1d_collider import DETCollider1D, DETParams1D
from det_concurrent_regimes import (
    DETRegimeSimulator, RegimeParams,
    create_k_region, create_w_region
)

OUTPUT_DIR = "/home/ubuntu/det_app_results/transition"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FITTING MODELS
# ============================================================

def linear_model(t, a, b):
    return a * t + b

def exponential_model(t, a, b, c):
    return a * (1 - np.exp(-b * t)) + c

def sigmoid_model(t, K_max, t_mid, steepness, K_min):
    """Logistic sigmoid — the classic phase-transition shape."""
    return K_min + (K_max - K_min) / (1.0 + np.exp(-steepness * (t - t_mid)))

def power_law_model(t, a, n, c):
    return a * (t ** n) + c

def tanh_model(t, K_max, t_mid, steepness, K_min):
    """Hyperbolic tangent — another phase-transition shape."""
    return K_min + (K_max - K_min) * 0.5 * (1 + np.tanh(steepness * (t - t_mid)))


# ============================================================
# T1: DRIVEN TRANSITION SHAPE
# ============================================================

def experiment_T1_driven_transition():
    """
    T1: Driven Transition Shape
    
    Start with a pure W-regime lattice. Inject coherence at a constant rate
    (simulating external driving). Record K(t) at very high time resolution.
    Fit multiple models (linear, exponential, sigmoid, power-law) to determine
    the best description of the transition.
    """
    print("\n" + "="*70)
    print("T1: Driven Transition Shape — High-Resolution K(t) Curve")
    print("="*70)
    
    N = 100
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.1,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.003,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    # Multiple driving rates to see if the shape changes
    drive_rates = [0.0005, 0.001, 0.002, 0.004, 0.008]
    all_results = {}
    
    for rate in drive_rates:
        np.random.seed(42)
        collider = DETCollider1D(params)
        
        # Pure W-regime start
        collider.C_R[:] = 0.10
        collider.q[:] = 0.55
        collider.F[:] = 0.5
        collider.a[:] = 0.50
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams())
        
        n_steps = 3000
        times = []
        K_trace = []
        C_trace = []
        q_trace = []
        O_trace = []
        
        for t in range(n_steps):
            # Drive: inject coherence AND reduce q
            collider.C_R[:] = np.clip(collider.C_R + rate, 0, 1)
            collider.q[:] = np.clip(collider.q - rate * 0.5, 0, 1)
            
            diag = regime.step(t)
            times.append(t)
            K_trace.append(float(np.mean(diag.K)))
            C_trace.append(float(np.mean(collider.C_R)))
            q_trace.append(float(np.mean(collider.q)))
            O_trace.append(float(np.mean(diag.O)))
        
        all_results[rate] = {
            'times': np.array(times),
            'K': np.array(K_trace),
            'C': np.array(C_trace),
            'q': np.array(q_trace),
            'O': np.array(O_trace)
        }
        
        # Find transition point (K crosses 0.5)
        K_arr = np.array(K_trace)
        cross = np.where(K_arr > 0.5)[0]
        t_cross = int(cross[0]) if len(cross) > 0 else None
        
        print(f"  Rate={rate:.4f}: transition at step {t_cross or 'NEVER'}, "
              f"final K={K_trace[-1]:.3f}")
    
    # === FIT MODELS to the middle rate ===
    ref_rate = 0.002
    data = all_results[ref_rate]
    t_data = data['times'].astype(float)
    K_data = data['K']
    
    # Only fit the transition region (where K is changing)
    active = np.where((K_data > 0.05) & (K_data < 0.72))[0]
    if len(active) > 20:
        t_fit = t_data[active]
        K_fit = K_data[active]
        t_fit_norm = t_fit - t_fit[0]  # normalize to start at 0
        
        fits = {}
        
        # Linear fit
        try:
            popt, _ = curve_fit(linear_model, t_fit_norm, K_fit)
            K_pred = linear_model(t_fit_norm, *popt)
            residual = np.mean((K_fit - K_pred)**2)
            fits['Linear'] = {'params': popt, 'residual': residual, 'pred': K_pred}
        except Exception:
            pass
        
        # Exponential fit
        try:
            popt, _ = curve_fit(exponential_model, t_fit_norm, K_fit,
                               p0=[0.7, 0.005, 0.05], maxfev=10000)
            K_pred = exponential_model(t_fit_norm, *popt)
            residual = np.mean((K_fit - K_pred)**2)
            fits['Exponential'] = {'params': popt, 'residual': residual, 'pred': K_pred}
        except Exception:
            pass
        
        # Sigmoid fit
        try:
            t_mid_guess = t_fit_norm[len(t_fit_norm)//2]
            popt, _ = curve_fit(sigmoid_model, t_fit_norm, K_fit,
                               p0=[0.75, t_mid_guess, 0.01, 0.05], maxfev=10000)
            K_pred = sigmoid_model(t_fit_norm, *popt)
            residual = np.mean((K_fit - K_pred)**2)
            fits['Sigmoid'] = {'params': popt, 'residual': residual, 'pred': K_pred}
        except Exception:
            pass
        
        # Power-law fit
        try:
            popt, _ = curve_fit(power_law_model, t_fit_norm + 1, K_fit,
                               p0=[0.001, 1.0, 0.05], maxfev=10000)
            K_pred = power_law_model(t_fit_norm + 1, *popt)
            residual = np.mean((K_fit - K_pred)**2)
            fits['Power Law'] = {'params': popt, 'residual': residual, 'pred': K_pred}
        except Exception:
            pass
        
        print(f"\n  Model fits (rate={ref_rate}):")
        for name, f in sorted(fits.items(), key=lambda x: x[1]['residual']):
            print(f"    {name:15s}: MSE = {f['residual']:.6f}")
            if name == 'Sigmoid':
                print(f"      K_max={f['params'][0]:.3f}, t_mid={f['params'][1]:.1f}, "
                      f"steepness={f['params'][2]:.5f}, K_min={f['params'][3]:.3f}")
            elif name == 'Power Law':
                print(f"      exponent n={f['params'][1]:.3f}")
    else:
        fits = {}
        t_fit_norm = np.array([])
        K_fit = np.array([])
    
    # === PLOT ===
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("T1: W→K Driven Transition — Shape Analysis",
                 fontsize=16, fontweight='bold')
    
    # Panel 1: K(t) for all driving rates
    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.plasma
    for i, (rate, data) in enumerate(all_results.items()):
        color = cmap(i / max(1, len(all_results) - 1))
        ax.plot(data['times'], data['K'], color=color, linewidth=2,
                label=f'rate={rate}')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='K/W boundary')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean K_i')
    ax.set_title('K(t) at Different Driving Rates')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: C(t) and q(t) for reference rate
    ax = fig.add_subplot(gs[0, 1])
    ref = all_results[ref_rate]
    ax.plot(ref['times'], ref['C'], 'b-', linewidth=2, label='C (coherence)')
    ax.plot(ref['times'], ref['q'], 'r-', linewidth=2, label='q (debt)')
    ax.plot(ref['times'], ref['K'], 'g-', linewidth=2, label='K (regime)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title(f'C, q, K Evolution (rate={ref_rate})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Model fits comparison
    ax = fig.add_subplot(gs[0, 2])
    if len(t_fit_norm) > 0:
        ax.scatter(t_fit_norm[::5], K_fit[::5], s=10, color='black', alpha=0.5, label='Data')
        fit_colors = {'Linear': 'red', 'Exponential': 'blue', 'Sigmoid': 'green', 'Power Law': 'orange'}
        for name, f in fits.items():
            ax.plot(t_fit_norm, f['pred'], color=fit_colors.get(name, 'gray'),
                    linewidth=2, label=f"{name} (MSE={f['residual']:.5f})")
        ax.set_xlabel('Steps (normalized)')
        ax.set_ylabel('K_i')
        ax.set_title('Model Fits to Transition Region')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # Panel 4: dK/dt (first derivative)
    ax = fig.add_subplot(gs[1, 0])
    for i, (rate, data) in enumerate(all_results.items()):
        color = cmap(i / max(1, len(all_results) - 1))
        K_smooth = savgol_filter(data['K'], min(51, len(data['K'])//10*2+1), 3)
        dKdt = np.gradient(K_smooth, data['times'])
        ax.plot(data['times'], dKdt, color=color, linewidth=1.5, label=f'rate={rate}')
    ax.set_xlabel('Step')
    ax.set_ylabel('dK/dt')
    ax.set_title('First Derivative: Rate of K Change')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 5: d²K/dt² (second derivative — acceleration)
    ax = fig.add_subplot(gs[1, 1])
    for i, (rate, data) in enumerate(all_results.items()):
        color = cmap(i / max(1, len(all_results) - 1))
        K_smooth = savgol_filter(data['K'], min(101, len(data['K'])//10*2+1), 3)
        d2Kdt2 = np.gradient(np.gradient(K_smooth, data['times']), data['times'])
        ax.plot(data['times'], d2Kdt2, color=color, linewidth=1.5, label=f'rate={rate}')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('d²K/dt²')
    ax.set_title('Second Derivative: Acceleration of Transition')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 6: K vs C (parametric — is the relationship linear or nonlinear?)
    ax = fig.add_subplot(gs[1, 2])
    for i, (rate, data) in enumerate(all_results.items()):
        color = cmap(i / max(1, len(all_results) - 1))
        ax.plot(data['C'], data['K'], color=color, linewidth=2, label=f'rate={rate}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='K=C (linear)')
    ax.set_xlabel('Mean Coherence C')
    ax.set_ylabel('Mean K_i')
    ax.set_title('K vs C: Parametric Transition Curve')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 7: O(t) — observability during transition
    ax = fig.add_subplot(gs[2, 0])
    for i, (rate, data) in enumerate(all_results.items()):
        color = cmap(i / max(1, len(all_results) - 1))
        ax.plot(data['times'], data['O'], color=color, linewidth=2, label=f'rate={rate}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean O_i')
    ax.set_title('Observability During Transition')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 8: dK/dC — how much K changes per unit C
    ax = fig.add_subplot(gs[2, 1])
    for i, (rate, data) in enumerate(all_results.items()):
        color = cmap(i / max(1, len(all_results) - 1))
        C_smooth = savgol_filter(data['C'], min(51, len(data['C'])//10*2+1), 3)
        K_smooth = savgol_filter(data['K'], min(51, len(data['K'])//10*2+1), 3)
        dC = np.gradient(C_smooth)
        dK = np.gradient(K_smooth)
        dKdC = dK / (dC + 1e-10)
        # Smooth again for display
        dKdC_smooth = savgol_filter(dKdC, min(101, len(dKdC)//10*2+1), 3)
        ax.plot(data['C'], dKdC_smooth, color=color, linewidth=1.5, label=f'rate={rate}')
    ax.set_xlabel('Mean Coherence C')
    ax.set_ylabel('dK/dC (Susceptibility)')
    ax.set_title('Transition Susceptibility')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Panel 9: Summary text
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    summary_lines = ["TRANSITION SHAPE SUMMARY", ""]
    if fits:
        best = min(fits.items(), key=lambda x: x[1]['residual'])
        summary_lines.append(f"Best fit model: {best[0]}")
        summary_lines.append(f"  MSE = {best[1]['residual']:.6f}")
        if best[0] == 'Sigmoid':
            p = best[1]['params']
            summary_lines.append(f"  K_max={p[0]:.3f}, t_mid={p[1]:.0f}")
            summary_lines.append(f"  steepness={p[2]:.5f}")
        summary_lines.append("")
        for name, f in sorted(fits.items(), key=lambda x: x[1]['residual']):
            summary_lines.append(f"{name}: MSE={f['residual']:.6f}")
    ax.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(f"{OUTPUT_DIR}/T1_driven_transition.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/T1_driven_transition.png")
    
    return all_results, fits


# ============================================================
# T2: HYSTERESIS — FORWARD AND REVERSE TRANSITION
# ============================================================

def experiment_T2_hysteresis():
    """
    T2: Hysteresis Test
    
    Drive the system W→K (forward), then reverse the driving (K→W).
    Does the reverse path follow the same curve? Or is there hysteresis
    (the system "remembers" its K-state and resists returning to W)?
    """
    print("\n" + "="*70)
    print("T2: Hysteresis — Forward (W→K) vs Reverse (K→W) Transition")
    print("="*70)
    
    N = 100
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.1,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.003,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False, boundary_enabled=False
    )
    
    rate = 0.002
    
    np.random.seed(42)
    collider = DETCollider1D(params)
    collider.C_R[:] = 0.10
    collider.q[:] = 0.55
    collider.F[:] = 0.5
    collider.a[:] = 0.50
    
    np.random.seed(42)
    regime = DETRegimeSimulator(collider, RegimeParams())
    
    # Phase 1: Forward drive (W→K)
    n_forward = 2000
    fwd_times = []
    fwd_K = []
    fwd_C = []
    fwd_q = []
    
    for t in range(n_forward):
        collider.C_R[:] = np.clip(collider.C_R + rate, 0, 1)
        collider.q[:] = np.clip(collider.q - rate * 0.5, 0, 1)
        diag = regime.step(t)
        fwd_times.append(t)
        fwd_K.append(float(np.mean(diag.K)))
        fwd_C.append(float(np.mean(collider.C_R)))
        fwd_q.append(float(np.mean(collider.q)))
    
    print(f"  Forward: K went from {fwd_K[0]:.3f} to {fwd_K[-1]:.3f}")
    
    # Phase 2: Reverse drive (K→W)
    n_reverse = 2000
    rev_times = []
    rev_K = []
    rev_C = []
    rev_q = []
    
    for t in range(n_reverse):
        collider.C_R[:] = np.clip(collider.C_R - rate, 0, 1)
        collider.q[:] = np.clip(collider.q + rate * 0.5, 0, 1)
        diag = regime.step(n_forward + t)
        rev_times.append(n_forward + t)
        rev_K.append(float(np.mean(diag.K)))
        rev_C.append(float(np.mean(collider.C_R)))
        rev_q.append(float(np.mean(collider.q)))
    
    print(f"  Reverse: K went from {rev_K[0]:.3f} to {rev_K[-1]:.3f}")
    
    # Compute hysteresis area
    # Interpolate both curves onto same C grid
    fwd_C_arr = np.array(fwd_C)
    fwd_K_arr = np.array(fwd_K)
    rev_C_arr = np.array(rev_C)
    rev_K_arr = np.array(rev_K)
    
    C_common = np.linspace(max(fwd_C_arr.min(), rev_C_arr.min()),
                           min(fwd_C_arr.max(), rev_C_arr.max()), 200)
    fwd_K_interp = np.interp(C_common, fwd_C_arr, fwd_K_arr)
    rev_K_interp = np.interp(C_common, rev_C_arr[::-1], rev_K_arr[::-1])
    
    hysteresis_area = float(np.trapz(np.abs(fwd_K_interp - rev_K_interp), C_common))
    print(f"  Hysteresis area: {hysteresis_area:.4f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("T2: Hysteresis — Forward (W→K) vs Reverse (K→W)",
                 fontsize=14, fontweight='bold')
    
    # K vs C parametric
    ax = axes[0]
    ax.plot(fwd_C, fwd_K, 'b-', linewidth=2.5, label='Forward (W→K)', alpha=0.8)
    ax.plot(rev_C, rev_K, 'r-', linewidth=2.5, label='Reverse (K→W)', alpha=0.8)
    ax.fill_between(C_common, fwd_K_interp, rev_K_interp, alpha=0.15, color='purple',
                    label=f'Hysteresis area={hysteresis_area:.3f}')
    ax.set_xlabel('Mean Coherence C')
    ax.set_ylabel('Mean K_i')
    ax.set_title('Hysteresis Loop: K vs C')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # K vs time
    ax = axes[1]
    all_times = fwd_times + rev_times
    all_K = fwd_K + rev_K
    ax.plot(all_times, all_K, 'purple', linewidth=2)
    ax.axvline(x=n_forward, color='black', linestyle=':', label='Drive reversal')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(fwd_times, fwd_K, alpha=0.1, color='blue')
    ax.fill_between(rev_times, rev_K, alpha=0.1, color='red')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean K_i')
    ax.set_title('K(t): Forward Then Reverse')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # C and q vs time
    ax = axes[2]
    all_C = fwd_C + rev_C
    all_q = fwd_q + rev_q
    ax.plot(all_times, all_C, 'b-', linewidth=2, label='C')
    ax.plot(all_times, all_q, 'r-', linewidth=2, label='q')
    ax.plot(all_times, all_K, 'g-', linewidth=2, label='K')
    ax.axvline(x=n_forward, color='black', linestyle=':', label='Drive reversal')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('C, q, K During Full Cycle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/T2_hysteresis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/T2_hysteresis.png")
    
    return hysteresis_area


# ============================================================
# T3: CRITICAL SLOWING DOWN
# ============================================================

def experiment_T3_critical_slowing():
    """
    T3: Critical Slowing Down
    
    Near a phase transition, systems typically exhibit "critical slowing down":
    perturbations take longer to relax. We test this by:
    1. Setting the system at various C values (below, at, above transition)
    2. Applying a small perturbation (q pulse)
    3. Measuring the relaxation time back to equilibrium
    """
    print("\n" + "="*70)
    print("T3: Critical Slowing Down Near Transition")
    print("="*70)
    
    N = 100
    C_values = np.linspace(0.15, 0.90, 25)
    relaxation_times = []
    equilibrium_K = []
    
    for C_val in C_values:
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=C_val,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False, boundary_enabled=False
        )
        collider = DETCollider1D(params)
        collider.C_R[:] = C_val
        collider.q[:] = max(0.01, 0.6 - 0.6 * C_val)
        collider.F[:] = 1.0 + 2.0 * C_val
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams())
        
        # Equilibrate
        for t in range(500):
            diag = regime.step(t)
        
        K_eq = float(np.mean(diag.K))
        equilibrium_K.append(K_eq)
        
        # Apply perturbation: q pulse
        collider.q[:] = np.clip(collider.q + 0.15, 0, 1)
        
        # Measure relaxation
        K_perturbed = []
        for t in range(1000):
            diag = regime.step(500 + t)
            K_perturbed.append(float(np.mean(diag.K)))
        
        # Relaxation time: time to recover to within 5% of equilibrium
        K_perturbed = np.array(K_perturbed)
        threshold = K_eq * 0.95 if K_eq > 0.1 else K_eq + 0.01
        recovered = np.where(K_perturbed >= threshold)[0]
        relax_time = int(recovered[0]) if len(recovered) > 0 else 1000
        relaxation_times.append(relax_time)
    
    # Find the C value with maximum relaxation time (critical point)
    max_relax_idx = np.argmax(relaxation_times)
    C_critical = C_values[max_relax_idx]
    
    print(f"  Peak relaxation time: {relaxation_times[max_relax_idx]} steps at C={C_critical:.3f}")
    print(f"  K at critical C: {equilibrium_K[max_relax_idx]:.3f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("T3: Critical Slowing Down Near W→K Transition",
                 fontsize=14, fontweight='bold')
    
    ax = axes[0]
    ax.plot(C_values, relaxation_times, 'ro-', linewidth=2, markersize=6)
    ax.axvline(x=C_critical, color='green', linestyle=':', linewidth=2,
               label=f'C_crit={C_critical:.2f}')
    ax.set_xlabel('Coherence C')
    ax.set_ylabel('Relaxation Time (steps)')
    ax.set_title('Relaxation Time vs Coherence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(C_values, equilibrium_K, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=C_critical, color='green', linestyle=':', linewidth=2)
    ax.set_xlabel('Coherence C')
    ax.set_ylabel('Equilibrium K_i')
    ax.set_title('Equilibrium K vs Coherence')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(equilibrium_K, relaxation_times, 'mo-', linewidth=2, markersize=6)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='K=0.5')
    ax.set_xlabel('Equilibrium K_i')
    ax.set_ylabel('Relaxation Time (steps)')
    ax.set_title('Relaxation Time vs Regime Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/T3_critical_slowing.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/T3_critical_slowing.png")
    
    return C_critical, relaxation_times


# ============================================================
# T4: PHASE PORTRAIT IN (C, q, K) SPACE
# ============================================================

def experiment_T4_phase_portrait():
    """
    T4: Phase Portrait
    
    Map the transition surface in (C, q) → K space by sweeping both
    C and q independently. This reveals the full topology of the
    transition boundary.
    """
    print("\n" + "="*70)
    print("T4: Phase Portrait — Transition Surface in (C, q, K) Space")
    print("="*70)
    
    N = 50  # smaller for speed
    n_C = 30
    n_q = 30
    C_range = np.linspace(0.05, 0.95, n_C)
    q_range = np.linspace(0.05, 0.95, n_q)
    
    K_surface = np.zeros((n_q, n_C))
    O_surface = np.zeros((n_q, n_C))
    
    for i, q_val in enumerate(q_range):
        for j, C_val in enumerate(C_range):
            np.random.seed(42)
            params = DETParams1D(
                N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=C_val,
                momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
                q_enabled=True, alpha_q=0.001,
                lambda_a=30.0, beta_a=0.2,
                floor_enabled=True, eta_floor=0.15, F_core=5.0,
                gravity_enabled=False, boundary_enabled=False
            )
            collider = DETCollider1D(params)
            collider.C_R[:] = C_val
            collider.q[:] = q_val
            collider.F[:] = 1.0 + 2.0 * C_val
            
            np.random.seed(42)
            regime = DETRegimeSimulator(collider, RegimeParams())
            
            # Run to equilibrium
            for t in range(300):
                diag = regime.step(t)
            
            K_surface[i, j] = float(np.mean(diag.K))
            O_surface[i, j] = float(np.mean(diag.O))
    
    # Find the K=0.5 contour (transition boundary)
    print(f"  K range: [{K_surface.min():.3f}, {K_surface.max():.3f}]")
    
    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("T4: Phase Portrait — Transition Surface in (C, q) Space",
                 fontsize=14, fontweight='bold')
    
    # K surface
    ax = axes[0]
    im = ax.imshow(K_surface, extent=[C_range[0], C_range[-1], q_range[-1], q_range[0]],
                   aspect='auto', cmap='RdYlBu', vmin=0, vmax=1)
    cs = ax.contour(C_range, q_range, K_surface, levels=[0.5], colors='black', linewidths=2)
    ax.clabel(cs, fmt='K=0.5')
    ax.set_xlabel('Coherence C')
    ax.set_ylabel('Structural Debt q')
    ax.set_title('Regime Index K_i(C, q)')
    plt.colorbar(im, ax=ax, label='K_i')
    
    # O surface
    ax = axes[1]
    im = ax.imshow(O_surface, extent=[C_range[0], C_range[-1], q_range[-1], q_range[0]],
                   aspect='auto', cmap='viridis', vmin=0, vmax=1)
    cs = ax.contour(C_range, q_range, O_surface, levels=[0.3, 0.5, 0.7],
                    colors='white', linewidths=1)
    ax.clabel(cs, fmt='%.1f')
    ax.set_xlabel('Coherence C')
    ax.set_ylabel('Structural Debt q')
    ax.set_title('Observability O_i(C, q)')
    plt.colorbar(im, ax=ax, label='O_i')
    
    # Transition boundary with gradient arrows
    ax = axes[2]
    ax.contourf(C_range, q_range, K_surface, levels=20, cmap='RdYlBu', alpha=0.7)
    ax.contour(C_range, q_range, K_surface, levels=[0.5], colors='black', linewidths=3)
    
    # Gradient arrows showing dK direction
    dKdC = np.gradient(K_surface, axis=1)
    dKdq = np.gradient(K_surface, axis=0)
    skip = 3
    ax.quiver(C_range[::skip], q_range[::skip],
              dKdC[::skip, ::skip], -dKdq[::skip, ::skip],
              alpha=0.5, scale=15, color='black')
    
    ax.set_xlabel('Coherence C')
    ax.set_ylabel('Structural Debt q')
    ax.set_title('Transition Boundary + Gradient Field')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/T4_phase_portrait.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/T4_phase_portrait.png")
    
    return K_surface, O_surface


# ============================================================
# T5: SPONTANEOUS TRANSITION (ATTUNEMENT-DRIVEN)
# ============================================================

def experiment_T5_spontaneous():
    """
    T5: Spontaneous Transition
    
    Can a W-regime system spontaneously transition to K without external
    coherence injection? Test with attunement feedback enabled — where
    mutual observation reinforces coherence.
    
    Start at various C values near the boundary and see if attunement
    can push the system over.
    """
    print("\n" + "="*70)
    print("T5: Spontaneous Transition via Attunement Feedback")
    print("="*70)
    
    N = 100
    C_starts = [0.30, 0.40, 0.50, 0.60, 0.70]
    eta_values = [0.0, 0.05, 0.10, 0.20]
    
    results = {}
    
    for eta in eta_values:
        for C_start in C_starts:
            np.random.seed(42)
            params = DETParams1D(
                N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=C_start,
                momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
                q_enabled=True, alpha_q=0.003,
                lambda_a=30.0, beta_a=0.2,
                floor_enabled=True, eta_floor=0.15, F_core=5.0,
                gravity_enabled=False, boundary_enabled=False
            )
            collider = DETCollider1D(params)
            collider.C_R[:] = C_start
            collider.q[:] = max(0.01, 0.5 - 0.5 * C_start)
            collider.F[:] = 1.0 + 1.5 * C_start
            collider.a[:] = 0.7
            
            np.random.seed(42)
            rp = RegimeParams(attunement_enabled=(eta > 0), eta_attune=eta)
            regime = DETRegimeSimulator(collider, rp)
            
            n_steps = 3000
            K_trace = []
            C_trace = []
            
            for t in range(n_steps):
                diag = regime.step(t)
                K_trace.append(float(np.mean(diag.K)))
                C_trace.append(float(np.mean(collider.C_R)))
            
            key = (eta, C_start)
            results[key] = {
                'K': np.array(K_trace),
                'C': np.array(C_trace),
                'final_K': K_trace[-1],
                'final_C': C_trace[-1]
            }
    
    # Print summary
    print(f"\n  {'η_attune':>10} | {'C_start':>8} | {'Final K':>8} | {'Final C':>8} | {'Crossed K=0.5':>14}")
    print("  " + "-"*60)
    for (eta, C_start), data in results.items():
        crossed = "YES" if data['final_K'] > 0.5 else "no"
        print(f"  {eta:10.2f} | {C_start:8.2f} | {data['final_K']:8.3f} | {data['final_C']:8.3f} | {crossed:>14}")
    
    # === PLOT ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("T5: Spontaneous Transition via Attunement Feedback",
                 fontsize=14, fontweight='bold')
    
    cmap_C = plt.cm.viridis
    
    for idx, eta in enumerate(eta_values):
        ax = axes[idx // 2, idx % 2]
        for i, C_start in enumerate(C_starts):
            color = cmap_C(i / max(1, len(C_starts) - 1))
            data = results[(eta, C_start)]
            ax.plot(range(len(data['K'])), data['K'], color=color, linewidth=2,
                    label=f'C₀={C_start:.2f}')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean K_i')
        ax.set_title(f'η_attune = {eta:.2f}' + (' (no attunement)' if eta == 0 else ''))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.0)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/T5_spontaneous.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/T5_spontaneous.png")
    
    return results


# ============================================================
# T6: TRANSITION UNIVERSALITY — RATE-INDEPENDENT SHAPE
# ============================================================

def experiment_T6_universality():
    """
    T6: Transition Universality
    
    If the transition is a true phase change, its SHAPE should be
    rate-independent when plotted against the control parameter (C)
    rather than time. All curves K(C) should collapse onto a single
    universal curve regardless of driving rate.
    """
    print("\n" + "="*70)
    print("T6: Universality — Rate-Independent Transition Shape")
    print("="*70)
    
    N = 100
    drive_rates = [0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01]
    
    results = {}
    
    for rate in drive_rates:
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.1,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False, boundary_enabled=False
        )
        collider = DETCollider1D(params)
        collider.C_R[:] = 0.10
        collider.q[:] = 0.55
        collider.F[:] = 0.5
        collider.a[:] = 0.50
        
        np.random.seed(42)
        regime = DETRegimeSimulator(collider, RegimeParams())
        
        n_steps = int(min(10000, 1.2 / rate))  # enough to reach C≈1
        K_trace = []
        C_trace = []
        
        for t in range(n_steps):
            collider.C_R[:] = np.clip(collider.C_R + rate, 0, 1)
            collider.q[:] = np.clip(collider.q - rate * 0.5, 0, 1)
            diag = regime.step(t)
            K_trace.append(float(np.mean(diag.K)))
            C_trace.append(float(np.mean(collider.C_R)))
        
        results[rate] = {
            'K': np.array(K_trace),
            'C': np.array(C_trace)
        }
    
    # Compute collapse quality: variance of K at each C value
    C_grid = np.linspace(0.15, 0.85, 100)
    K_at_C = []
    for rate, data in results.items():
        K_interp = np.interp(C_grid, data['C'], data['K'])
        K_at_C.append(K_interp)
    K_at_C = np.array(K_at_C)
    K_variance = np.var(K_at_C, axis=0)
    mean_variance = float(np.mean(K_variance))
    max_variance = float(np.max(K_variance))
    
    print(f"  Collapse quality: mean variance = {mean_variance:.6f}, max = {max_variance:.6f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("T6: Universality — Do All Driving Rates Collapse to One Curve?",
                 fontsize=14, fontweight='bold')
    
    cmap = plt.cm.plasma
    
    # K vs C (should collapse)
    ax = axes[0]
    for i, (rate, data) in enumerate(results.items()):
        color = cmap(i / max(1, len(results) - 1))
        ax.plot(data['C'], data['K'], color=color, linewidth=2, label=f'rate={rate}')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Coherence C')
    ax.set_ylabel('Mean K_i')
    ax.set_title('K(C): Universal Collapse?')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Variance at each C
    ax = axes[1]
    ax.plot(C_grid, K_variance, 'r-', linewidth=2)
    ax.fill_between(C_grid, 0, K_variance, alpha=0.2, color='red')
    ax.set_xlabel('Coherence C')
    ax.set_ylabel('Var(K) across rates')
    ax.set_title(f'Collapse Variance (mean={mean_variance:.5f})')
    ax.grid(True, alpha=0.3)
    
    # Mean ± std band
    ax = axes[2]
    K_mean = np.mean(K_at_C, axis=0)
    K_std = np.std(K_at_C, axis=0)
    ax.plot(C_grid, K_mean, 'b-', linewidth=3, label='Mean K(C)')
    ax.fill_between(C_grid, K_mean - K_std, K_mean + K_std, alpha=0.2, color='blue',
                    label='±1 std')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Coherence C')
    ax.set_ylabel('K_i')
    ax.set_title('Universal Transition Curve ± Spread')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/T6_universality.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/T6_universality.png")
    
    return mean_variance, results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*70)
    print("DET REGIME TRANSITION DYNAMICS: W → K")
    print("="*70)
    
    r1_results, r1_fits = experiment_T1_driven_transition()
    r2_hysteresis = experiment_T2_hysteresis()
    r3_C_crit, r3_relax = experiment_T3_critical_slowing()
    r4_K_surface, r4_O_surface = experiment_T4_phase_portrait()
    r5_results = experiment_T5_spontaneous()
    r6_variance, r6_results = experiment_T6_universality()
    
    print("\n" + "="*70)
    print("ALL TRANSITION DYNAMICS EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*70)
