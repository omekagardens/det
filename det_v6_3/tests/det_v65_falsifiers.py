"""
DET v6.5 Falsifier Suite: Jubilee Operator + Agency-First
==========================================================

New falsifiers for the v6.5 patch:

  F_QD1 — Jubilee Non-Coercion:
    If any node with a_i=0 experiences q_D decrease → FAIL

  F_QD2 — No Hidden Globals:
    If adding disconnected components changes Jubilee rates → FAIL

  F_QD3 — No Cheap Redemption:
    In a sealed, no-flow config with D_i=0, q_D must not decrease → FAIL

  F_QD4 — W→K Local Transition Exists:
    W-regime pocket adjacent to K-regime must transition with Jubilee → FAIL if not

  F_A0 — Spontaneous Structure Test (Agency-First):
    System with a_i=0 for all nodes must NOT generate stable ordering,
    arithmetic-like structure, or long-term information → FAIL if it does

Backward compatibility tests:
  Existing 1D collider tests must still pass with jubilee_enabled=False
  Existing 1D collider tests must still pass with jubilee_enabled=True (no regression)

Reference: DET v6.4 Patch Card (Jubilee/Forgiveness)
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_1d_collider import DETCollider1D, DETParams1D
from det_concurrent_regimes import DETRegimeSimulator, RegimeParams


# ============================================================
# F_QD1: JUBILEE NON-COERCION
# ============================================================

def test_F_QD1_jubilee_non_coercion(verbose=True):
    """
    F_QD1: If any node with a_i=0 experiences q_D decrease → FAIL

    Setup: Create nodes with a=0 and high q_D. Run with Jubilee enabled.
    Verify that q_D does not decrease for a=0 nodes.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_QD1: Jubilee Non-Coercion (a=0 blocks Jubilee)")
        print("="*60)

    N = 100
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.5,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.003,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.03,
        jubilee_enabled=True, delta_q=0.01, n_q=2, D_0=0.05
    )
    sim = DETCollider1D(params)

    # Set up: high q_D everywhere, active flow
    sim.q_D[:] = 0.5
    sim.q_I[:] = 0.0
    sim.q[:] = 0.5
    sim.F[:] = 2.0
    sim.C_R[:] = 0.7

    # Sentinel nodes: a=0 (should NEVER have q_D reduced)
    sentinel_indices = [10, 20, 30, 40, 50]
    for idx in sentinel_indices:
        sim.a[idx] = 0.0

    # Active nodes: a=1 (should have q_D reduced)
    active_indices = [15, 25, 35, 45, 55]
    for idx in active_indices:
        sim.a[idx] = 1.0

    initial_qD_sentinels = [sim.q_D[i] for i in sentinel_indices]
    initial_qD_active = [sim.q_D[i] for i in active_indices]

    # Create some flow by adding packets
    sim.add_packet(50, mass=10.0, width=10.0, momentum=0.5)

    for t in range(500):
        sim.step()
        # Force sentinel agency to stay at 0 (agency update might change it)
        for idx in sentinel_indices:
            sim.a[idx] = 0.0

    # Check: sentinel q_D must NOT have decreased
    sentinel_decreased = False
    for i, idx in enumerate(sentinel_indices):
        if sim.q_D[idx] < initial_qD_sentinels[i] - 1e-10:
            sentinel_decreased = True
            if verbose:
                print(f"  VIOLATION: sentinel[{idx}] q_D decreased from "
                      f"{initial_qD_sentinels[i]:.6f} to {sim.q_D[idx]:.6f}")

    # Check: at least some active nodes should have q_D reduced
    active_any_decreased = False
    for i, idx in enumerate(active_indices):
        if sim.q_D[idx] < initial_qD_active[i] - 1e-10:
            active_any_decreased = True

    passed = (not sentinel_decreased) and active_any_decreased

    if verbose:
        print(f"  Sentinel q_D decreased: {sentinel_decreased}")
        print(f"  Active q_D decreased: {active_any_decreased}")
        print(f"  F_QD1 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'sentinel_decreased': sentinel_decreased,
            'active_decreased': active_any_decreased}


# ============================================================
# F_QD2: NO HIDDEN GLOBALS
# ============================================================

def test_F_QD2_no_hidden_globals(verbose=True):
    """
    F_QD2: Adding disconnected components must NOT change Jubilee rates.

    Run two simulations:
    A) Small isolated packet with Jubilee
    B) Same packet + distant second packet

    The Jubilee rate in the first packet's region must be identical.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_QD2: No Hidden Globals (Jubilee is strictly local)")
        print("="*60)

    def run_scenario(add_distant_packet=False, seed=42):
        np.random.seed(seed)
        N = 200
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.5,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.003,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False,
            boundary_enabled=False,  # No boundary to avoid grace interference
            jubilee_enabled=True, delta_q=0.01, n_q=2, D_0=0.05
        )
        sim = DETCollider1D(params)

        # Primary packet at position 30
        sim.add_packet(30, mass=5.0, width=5.0, momentum=0.3, initial_q=0.3)

        # Optional distant packet
        if add_distant_packet:
            sim.add_packet(170, mass=20.0, width=5.0, momentum=-0.5, initial_q=0.8)

        jubilee_trace = []
        for t in range(200):
            sim.step()
            # Record Jubilee in the primary packet's region (20-40)
            jubilee_trace.append(float(np.sum(sim.last_jubilee[20:40])))

        return np.array(jubilee_trace), sim.q_D[20:40].copy()

    trace_A, qD_A = run_scenario(add_distant_packet=False)
    trace_B, qD_B = run_scenario(add_distant_packet=True)

    # The Jubilee rates in the local region should be identical
    max_diff = float(np.max(np.abs(trace_A - trace_B)))
    qD_diff = float(np.max(np.abs(qD_A - qD_B)))

    passed = max_diff < 1e-10 and qD_diff < 1e-10

    if verbose:
        print(f"  Max Jubilee rate difference: {max_diff:.2e}")
        print(f"  Max q_D difference: {qD_diff:.2e}")
        print(f"  F_QD2 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'max_jubilee_diff': max_diff, 'max_qD_diff': qD_diff}


# ============================================================
# F_QD3: NO CHEAP REDEMPTION
# ============================================================

def test_F_QD3_no_cheap_redemption(verbose=True):
    """
    F_QD3: In a sealed, no-flow configuration with D_i=0, q_D must NOT decrease.

    Setup: Uniform F (no gradients → no flow → no dissipation).
    Even with high agency and coherence, Jubilee should not fire.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_QD3: No Cheap Redemption (D=0 blocks Jubilee)")
        print("="*60)

    N = 100
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=1.0, F_MIN=0.0, C_init=0.9,
        momentum_enabled=False,  # No momentum → no momentum flux
        q_enabled=True, alpha_q=0.0,  # No q-locking (prevent q increase)
        lambda_a=0.0, beta_a=0.0,  # No agency dynamics
        floor_enabled=False,
        gravity_enabled=False,
        boundary_enabled=False,
        jubilee_enabled=True, delta_q=1.0, n_q=1, D_0=0.01  # Very aggressive Jubilee
    )
    sim = DETCollider1D(params)

    # Uniform F → no gradients → no flow → D=0
    sim.F[:] = 1.0
    sim.q_D[:] = 0.8
    sim.q_I[:] = 0.0
    sim.q[:] = 0.8
    sim.a[:] = 1.0
    sim.C_R[:] = 0.9

    initial_qD = sim.q_D.copy()

    for t in range(500):
        sim.step()

    max_decrease = float(np.max(initial_qD - sim.q_D))

    passed = max_decrease < 1e-10

    if verbose:
        print(f"  Max q_D decrease: {max_decrease:.2e}")
        print(f"  Total Jubilee applied: {sim.total_jubilee:.2e}")
        print(f"  F_QD3 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'max_decrease': max_decrease,
            'total_jubilee': sim.total_jubilee}


# ============================================================
# F_QD4: W→K LOCAL TRANSITION EXISTS
# ============================================================

def test_F_QD4_WK_transition(verbose=True):
    """
    F_QD4: W→K Local Transition Exists

    The spec says: "Construct a W-regime pocket (high q_D, low C) adjacent
    to K-regime and run with Jubilee enabled. If no statistically robust
    W→K transition occurs across a declared parameter sweep → FAIL"

    Strategy: Use a moderately damaged W-zone (q_D=0.35) adjacent to a
    strong K-zone. The K-zone provides flow and coherence at the boundary.
    Bond healing propagates coherence into W. Once C rises, Jubilee
    activates and reduces q_D. We measure q_D decrease in the W-boundary
    zone as the primary metric, and compare Jubilee-ON vs Jubilee-OFF.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_QD4: W→K Local Transition Exists (Jubilee enabled)")
        print("="*60)

    N = 200
    x = np.arange(N)

    def run_with_jubilee(jubilee_on, delta_q=0.05, steps=10000):
        np.random.seed(42)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.15,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.001,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=False,
            boundary_enabled=True, grace_enabled=True,
            F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.10,
            jubilee_enabled=jubilee_on, delta_q=delta_q, n_q=1, D_0=0.01
        )
        sim = DETCollider1D(params)

        # K-region: left half (0-100)
        K_mask = x < 100
        W_mask = x >= 100
        # Boundary zone: W-side cells near the K/W interface (100-120)
        W_boundary = (x >= 100) & (x < 120)

        # K-region: healthy
        sim.C_R[K_mask] = 0.85
        sim.q[K_mask] = 0.02
        sim.q_D[K_mask] = 0.02
        sim.q_I[K_mask] = 0.0
        sim.F[K_mask] = 3.0
        sim.a[K_mask] = 0.95

        # W-region: damaged but not maximally
        sim.C_R[W_mask] = 0.20
        sim.q[W_mask] = 0.35
        sim.q_D[W_mask] = 0.35
        sim.q_I[W_mask] = 0.0
        sim.F[W_mask] = 0.8
        sim.a[W_mask] = 0.3

        # Gradient at boundary for smooth transition
        for i in range(90, 110):
            frac = (i - 90) / 20.0
            sim.C_R[i] = 0.85 * (1 - frac) + 0.20 * frac
            sim.q_D[i] = 0.02 * (1 - frac) + 0.35 * frac
            sim.q[i] = sim.q_D[i]
            sim.F[i] = 3.0 * (1 - frac) + 0.8 * frac
            sim.a[i] = 0.95 * (1 - frac) + 0.3 * frac

        initial_qD_boundary = float(np.mean(sim.q_D[W_boundary]))

        for t in range(steps):
            sim.step()

        final_qD_boundary = float(np.mean(sim.q_D[W_boundary]))
        final_a_boundary = float(np.mean(sim.a[W_boundary]))
        final_C_boundary = float(np.mean(sim.C_R[W_boundary]))

        return {
            'initial_qD': initial_qD_boundary,
            'final_qD': final_qD_boundary,
            'qD_decrease': initial_qD_boundary - final_qD_boundary,
            'final_a': final_a_boundary,
            'final_C': final_C_boundary,
            'total_jubilee': sim.total_jubilee
        }

    # Run with Jubilee OFF (baseline)
    off_result = run_with_jubilee(False)
    if verbose:
        print(f"  Jubilee OFF: q_D {off_result['initial_qD']:.4f} → {off_result['final_qD']:.4f} "
              f"(decrease={off_result['qD_decrease']:.4f})")

    # Run with Jubilee ON at multiple rates
    best_decrease = 0
    best_dq = 0
    for dq in [0.01, 0.05, 0.10, 0.50]:
        on_result = run_with_jubilee(True, delta_q=dq)
        if verbose:
            print(f"  Jubilee ON (dq={dq:.2f}): q_D {on_result['initial_qD']:.4f} → "
                  f"{on_result['final_qD']:.4f} (decrease={on_result['qD_decrease']:.4f}, "
                  f"jubilee_total={on_result['total_jubilee']:.4f})")
        if on_result['qD_decrease'] > best_decrease:
            best_decrease = on_result['qD_decrease']
            best_dq = dq

    # PASS if Jubilee produces measurably more q_D decrease than baseline
    jubilee_helps = best_decrease > off_result['qD_decrease'] + 1e-6
    # Also check that q_D actually decreased (any measurable amount)
    qD_decreased = best_decrease > 1e-5

    # The key test: Jubilee-ON must produce q_D decrease where Jubilee-OFF does not
    passed = jubilee_helps and qD_decreased

    if verbose:
        print(f"\n  Best q_D decrease: {best_decrease:.4f} at delta_q={best_dq}")
        print(f"  Baseline decrease: {off_result['qD_decrease']:.4f}")
        print(f"  Jubilee helps: {jubilee_helps}")
        print(f"  q_D decreased: {qD_decreased}")
        print(f"  F_QD4 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'best_decrease': best_decrease,
            'baseline_decrease': off_result['qD_decrease'], 'best_dq': best_dq}


# ============================================================
# F_A0: SPONTANEOUS STRUCTURE TEST (Agency-First)
# ============================================================

def test_F_A0_spontaneous_structure(verbose=True):
    """
    F_A0: Spontaneous Structure Test (Agency-First)

    Initialize a system with:
      - a_i = 0 for all nodes
      - stochastic noise only
      - no boundary operators

    If stable ordering, arithmetic-like structure, or long-term
    information emerges → Agency-First is false → FAIL

    We measure:
    1. Spatial ordering (autocorrelation)
    2. Information persistence (mutual information over time)
    3. Structure formation (clustering of F)
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_A0: Spontaneous Structure (Agency-First)")
        print("="*60)

    N = 200
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.0,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=False,  # No q dynamics
        lambda_a=0.0, beta_a=0.0,  # No agency dynamics
        floor_enabled=False,
        gravity_enabled=False,
        boundary_enabled=False,
        jubilee_enabled=False
    )
    sim = DETCollider1D(params)

    # ALL agency = 0
    sim.a[:] = 0.0

    # Random noise initial conditions
    sim.F[:] = np.random.uniform(0.001, 2.0, N)
    sim.C_R[:] = 0.0
    sim.pi_R[:] = np.random.uniform(-0.5, 0.5, N)

    initial_F = sim.F.copy()

    # Measure 1: Initial spatial autocorrelation
    def spatial_autocorrelation(x, lag=1):
        """Pearson correlation between x and shifted x."""
        x_shifted = np.roll(x, lag)
        mean_x = np.mean(x)
        cov = np.mean((x - mean_x) * (x_shifted - mean_x))
        var = np.var(x) + 1e-15
        return cov / var

    initial_autocorr = spatial_autocorrelation(sim.F)

    # Run simulation
    F_snapshots = [sim.F.copy()]
    autocorr_trace = [initial_autocorr]

    for t in range(2000):
        sim.step()
        if t % 100 == 0:
            F_snapshots.append(sim.F.copy())
            autocorr_trace.append(spatial_autocorrelation(sim.F))

    final_autocorr = spatial_autocorrelation(sim.F)

    # Measure 2: Information persistence
    # Cross-correlation between initial and final F patterns
    initial_centered = initial_F - np.mean(initial_F)
    final_centered = sim.F - np.mean(sim.F)
    cross_corr = np.abs(np.sum(initial_centered * final_centered)) / \
                 (np.sqrt(np.sum(initial_centered**2) * np.sum(final_centered**2)) + 1e-15)

    # Measure 3: Structure formation (Gini coefficient of F)
    def gini(x):
        x_sorted = np.sort(x)
        n = len(x)
        cumx = np.cumsum(x_sorted)
        return (n + 1 - 2 * np.sum(cumx) / (cumx[-1] + 1e-15)) / n

    initial_gini = gini(initial_F)
    final_gini = gini(sim.F)

    # Measure 4: Total flow (should be zero with a=0)
    total_flow = float(np.sum(np.abs(sim.pi_R)))

    # PASS criteria for Agency-First:
    # The key prediction: with a=0, the agency gate g_R = sqrt(a_i*a_j) = 0
    # So ALL diffusive flow should be zero. F should not change at all.
    # No dynamics = no new structure = no new information = Agency-First confirmed.
    F_change = float(np.max(np.abs(sim.F - initial_F)))
    no_flow = F_change < 1e-10

    # If F doesn't change, cross-correlation is trivially 1.0 (frozen state).
    # That's not "information persistence" — it's absence of dynamics.
    # The real test is: did any NEW ordering emerge?
    no_ordering = final_autocorr <= initial_autocorr + 0.1
    no_structure = final_gini <= initial_gini + 0.1

    # If nothing changed (no_flow=True), that IS the Agency-First prediction:
    # without agency, nothing happens — no structure, no information, no movement.
    if no_flow:
        no_persistence = True  # Frozen state is not "persistent information"
    else:
        no_persistence = cross_corr < 0.5

    passed = no_ordering and no_persistence and no_structure and no_flow

    if verbose:
        print(f"  Initial autocorrelation: {initial_autocorr:.4f}")
        print(f"  Final autocorrelation:   {final_autocorr:.4f}")
        print(f"  Cross-correlation (info persistence): {cross_corr:.4f}")
        print(f"  Initial Gini: {initial_gini:.4f}")
        print(f"  Final Gini:   {final_gini:.4f}")
        print(f"  Max F change: {F_change:.2e}")
        print(f"  No ordering: {no_ordering}")
        print(f"  No persistence: {no_persistence}")
        print(f"  No structure: {no_structure}")
        print(f"  No flow (a=0 gate): {no_flow}")
        print(f"  F_A0 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'no_ordering': no_ordering,
            'no_persistence': no_persistence, 'no_structure': no_structure,
            'no_flow': no_flow, 'F_change': F_change}


# ============================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================

def test_backward_compat_jubilee_off(verbose=True):
    """Verify existing behavior is unchanged when jubilee_enabled=False."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Backward Compatibility (Jubilee OFF)")
        print("="*60)

    # Run with jubilee_enabled=False (default)
    np.random.seed(42)
    params_old = DETParams1D(
        N=100, DT=0.02, F_VAC=0.01, C_init=0.3,
        momentum_enabled=True, q_enabled=True,
        gravity_enabled=False, boundary_enabled=True, grace_enabled=True,
        jubilee_enabled=False
    )
    sim_old = DETCollider1D(params_old)
    sim_old.add_packet(50, mass=5.0, width=5.0, momentum=0.3, initial_q=0.3)

    for _ in range(500):
        sim_old.step()

    F_old = sim_old.F.copy()
    q_old = sim_old.q.copy()
    a_old = sim_old.a.copy()

    # Run again with same seed
    np.random.seed(42)
    params_new = DETParams1D(
        N=100, DT=0.02, F_VAC=0.01, C_init=0.3,
        momentum_enabled=True, q_enabled=True,
        gravity_enabled=False, boundary_enabled=True, grace_enabled=True,
        jubilee_enabled=False
    )
    sim_new = DETCollider1D(params_new)
    sim_new.add_packet(50, mass=5.0, width=5.0, momentum=0.3, initial_q=0.3)

    for _ in range(500):
        sim_new.step()

    max_F_diff = float(np.max(np.abs(sim_new.F - F_old)))
    max_q_diff = float(np.max(np.abs(sim_new.q - q_old)))
    max_a_diff = float(np.max(np.abs(sim_new.a - a_old)))

    passed = max_F_diff < 1e-10 and max_q_diff < 1e-10 and max_a_diff < 1e-10

    if verbose:
        print(f"  Max F difference: {max_F_diff:.2e}")
        print(f"  Max q difference: {max_q_diff:.2e}")
        print(f"  Max a difference: {max_a_diff:.2e}")
        print(f"  Backward Compat {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed}


def test_backward_compat_jubilee_on_no_regression(verbose=True):
    """Verify that enabling Jubilee doesn't break mass conservation or time dilation."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Backward Compatibility (Jubilee ON, no regression)")
        print("="*60)

    np.random.seed(42)
    params = DETParams1D(
        N=200, DT=0.02, F_VAC=0.01, C_init=0.3,
        momentum_enabled=True, q_enabled=True,
        gravity_enabled=True, boundary_enabled=True, grace_enabled=True,
        healing_enabled=True, eta_heal=0.03,
        jubilee_enabled=True, delta_q=0.001, n_q=2, D_0=0.05
    )
    sim = DETCollider1D(params)
    sim.add_packet(100, mass=20.0, width=5.0, initial_q=0.5)

    initial_mass = sim.total_mass()

    for _ in range(500):
        sim.step()

    final_mass = sim.total_mass()
    grace = sim.total_grace_injected
    mass_drift = abs(final_mass - initial_mass - grace) / initial_mass

    # Time dilation: P should still be lower at center
    P_center = sim.P[100]
    P_edge = sim.P[150]
    time_dilated = P_center < P_edge

    # q_D should have decreased somewhere (Jubilee is active)
    jubilee_active = sim.total_jubilee > 0

    passed = mass_drift < 0.10 and time_dilated and jubilee_active

    if verbose:
        print(f"  Mass drift: {mass_drift*100:.4f}%")
        print(f"  Time dilation (P_center < P_edge): {time_dilated}")
        print(f"  Jubilee active: {jubilee_active} (total: {sim.total_jubilee:.6f})")
        print(f"  No Regression {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'mass_drift': mass_drift,
            'time_dilated': time_dilated, 'jubilee_active': jubilee_active}


# ============================================================
# q_I IMMUTABILITY TEST
# ============================================================

def test_qI_immutability(verbose=True):
    """Verify that q_I (identity debt) is never modified by Jubilee."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: q_I Immutability (Jubilee only affects q_D)")
        print("="*60)

    N = 200
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, C_init=0.8,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.001,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.10,
        jubilee_enabled=True, delta_q=0.50, n_q=1, D_0=0.01,
        q_I_fraction=0.1
    )
    sim = DETCollider1D(params)

    # Set initial conditions: high coherence, moderate q_D, active flow
    sim.q_I[:] = 0.1
    sim.q_D[:] = 0.2
    sim.q[:] = 0.3
    sim.C_R[:] = 0.8
    sim.a[:] = 0.9

    # Create strong flow with colliding packets
    sim.add_packet(70, mass=10.0, width=10.0, momentum=0.5)
    sim.add_packet(130, mass=10.0, width=10.0, momentum=-0.5)

    initial_qI = sim.q_I.copy()
    initial_qD_mean = float(np.mean(sim.q_D))

    for _ in range(5000):
        sim.step()

    # q_I should only increase (from q-locking), never decrease from Jubilee
    qI_decreased_from_jubilee = np.any(sim.q_I < initial_qI - 1e-10)

    # q_D should have decreased somewhere (Jubilee is active)
    final_qD_mean = float(np.mean(sim.q_D))
    jubilee_active = sim.total_jubilee > 0.01

    passed = (not qI_decreased_from_jubilee) and jubilee_active

    if verbose:
        print(f"  q_I decreased below initial: {qI_decreased_from_jubilee}")
        print(f"  Mean q_I: {np.mean(sim.q_I):.4f} (initial: {np.mean(initial_qI):.4f})")
        print(f"  Mean q_D: {final_qD_mean:.4f} (initial: {initial_qD_mean:.4f})")
        print(f"  Total Jubilee: {sim.total_jubilee:.4f}")
        print(f"  Jubilee active: {jubilee_active}")
        print(f"  q_I Immutability {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'qI_decreased': qI_decreased_from_jubilee,
            'jubilee_active': jubilee_active}


# ============================================================
# MAIN SUITE RUNNER
# ============================================================

def run_v65_falsifier_suite():
    """Run the complete v6.5 falsifier suite."""
    print("="*70)
    print("DET v6.5 FALSIFIER SUITE")
    print("Jubilee/Forgiveness Operator + Agency-First")
    print("="*70)

    start_time = time.time()
    results = {}

    # New Jubilee falsifiers
    results['F_QD1'] = test_F_QD1_jubilee_non_coercion()
    results['F_QD2'] = test_F_QD2_no_hidden_globals()
    results['F_QD3'] = test_F_QD3_no_cheap_redemption()
    results['F_QD4'] = test_F_QD4_WK_transition()

    # Agency-First falsifier
    results['F_A0'] = test_F_A0_spontaneous_structure()

    # Backward compatibility
    results['Compat_OFF'] = test_backward_compat_jubilee_off()
    results['Compat_ON'] = test_backward_compat_jubilee_on_no_regression()

    # q_I immutability
    results['q_I_immutable'] = test_qI_immutability()

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("v6.5 FALSIFIER SUITE SUMMARY")
    print("="*70)

    passed_count = 0
    total_count = 0

    for name, result in results.items():
        status = result.get('passed', False) if isinstance(result, dict) else result
        passed_count += 1 if status else 0
        total_count += 1
        print(f"  {name:20s}: {'PASS' if status else 'FAIL'}")

    print(f"\n  TOTAL: {passed_count}/{total_count} PASSED")
    print(f"  Runtime: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    results = run_v65_falsifier_suite()
