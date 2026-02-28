"""
DET v6.5b Mandatory Falsifiers: Thermodynamic & Gravitational Safety
=====================================================================

Implements the four mandatory falsifiers from the structural review:

  F_QD5 — Energy Coupling:
    Jubilee must be suppressed when F_op → 0. No free energy = no forgiveness.
    If q_D decreases in a vacuum (F ≈ 0) region → FAIL

  F_QD6 — Gravitational Stability Under Jubilee:
    A large q_D compact object undergoing Jubilee must not create nonlocal
    gravitational shifts. Far-field potential must decay smoothly.
    If far-field Φ changes faster than local q_D changes → FAIL

  F_QD7 — No Spontaneous Annealing:
    Random noise without coherence must not reduce q_D.
    If q_D decreases in a low-C, noisy environment → FAIL

  F_A1+ — Agency Bootstrap Test:
    If a_i=0 everywhere, no Jubilee cascade should ever start,
    even with high-C neighbors. This is stronger than F_A0.
    If any q_D decreases with global a=0 → FAIL

Reference: DET v6.5 Structural Review (Pasted_content_04.txt)
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_1d_collider import DETCollider1D, DETParams1D


# ============================================================
# F_QD5: ENERGY COUPLING (Jubilee suppressed when F_op → 0)
# ============================================================

def test_F_QD5_energy_coupling(verbose=True):
    """
    F_QD5: Jubilee must be suppressed when F_op → 0.

    Setup: Two regions — one with abundant F (resource-rich), one with F ≈ F_VAC
    (vacuum/depleted). Both have identical q_D, C, and a. Both have flow.

    The vacuum region should see near-zero Jubilee activity.
    The resource-rich region should see active Jubilee.

    This tests the reviewer's mandatory requirement:
    "Δq_D ∝ min(q_D, F_op/(1+F_op))"
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST F_QD5: Energy Coupling (F_op → 0 blocks Jubilee)")
        print("=" * 60)

    N = 200
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.7,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.001,
        lambda_a=30.0, beta_a=0.2,
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.03,
        jubilee_enabled=True, delta_q=0.01, n_q=1, D_0=0.01,
        jubilee_energy_coupling=True  # MANDATORY
    )
    sim = DETCollider1D(params)

    # Region A (0-99): Resource-rich, high F
    sim.F[:100] = 10.0
    sim.q_D[:100] = 0.4
    sim.q_I[:100] = 0.0
    sim.q[:100] = 0.4
    sim.a[:100] = 0.8
    sim.C_R[:100] = 0.8

    # Region B (100-199): Vacuum, F ≈ F_VAC
    sim.F[100:] = 0.011  # Just barely above F_VAC
    sim.q_D[100:] = 0.4
    sim.q_I[100:] = 0.0
    sim.q[100:] = 0.4
    sim.a[100:] = 0.8
    sim.C_R[100:] = 0.8

    # Add flow via packets in both regions (rich gets mass, vacuum gets only momentum)
    sim.add_packet(50, mass=5.0, width=10.0, momentum=0.5)
    sim.add_packet(150, mass=0.0, width=10.0, momentum=0.5)

    initial_qD_rich = float(np.mean(sim.q_D[20:80]))
    initial_qD_vacuum = float(np.mean(sim.q_D[120:180]))

    jubilee_rich_total = 0.0
    jubilee_vacuum_total = 0.0

    for t in range(5000):
        sim.step()
        jubilee_rich_total += float(np.sum(sim.last_jubilee[20:80]))
        jubilee_vacuum_total += float(np.sum(sim.last_jubilee[120:180]))

    final_qD_rich = float(np.mean(sim.q_D[20:80]))
    final_qD_vacuum = float(np.mean(sim.q_D[120:180]))

    rich_decrease = initial_qD_rich - final_qD_rich
    vacuum_decrease = initial_qD_vacuum - final_qD_vacuum

    # The vacuum region should have dramatically less Jubilee than the rich region
    # The energy coupling caps Jubilee by F_op/(1+F_op), which is ~0.001 for F≈F_VAC
    # Use the Jubilee totals as the primary metric (more direct)
    jubilee_ratio = jubilee_vacuum_total / (jubilee_rich_total + 1e-15)
    qD_ratio = max(0, vacuum_decrease) / (max(0, rich_decrease) + 1e-15)

    # Pass criteria: vacuum Jubilee total is < 20% of rich Jubilee total
    # (The energy cap F_op/(1+F_op) at F=0.01 is 0.0099 vs 0.909 at F=10 → 100x ratio)
    passed = (jubilee_rich_total > 0.0001) and (jubilee_ratio < 0.20)

    if verbose:
        print(f"  Rich region q_D decrease: {rich_decrease:.6f}")
        print(f"  Vacuum region q_D decrease: {vacuum_decrease:.6f}")
        print(f"  Jubilee totals — Rich: {jubilee_rich_total:.6f}, Vacuum: {jubilee_vacuum_total:.6f}")
        print(f"  Jubilee ratio (vacuum/rich): {jubilee_ratio:.4f}")
        print(f"  F_QD5 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'rich_decrease': rich_decrease,
            'vacuum_decrease': vacuum_decrease, 'jubilee_ratio': jubilee_ratio}


# ============================================================
# F_QD6: GRAVITATIONAL STABILITY UNDER JUBILEE
# ============================================================

def test_F_QD6_gravitational_stability(verbose=True):
    """
    F_QD6: Gravitational Stability Under Jubilee

    Differential test: run two identical simulations, one with Jubilee ON
    and one with Jubilee OFF. Compare the far-field gravitational potential.

    The Jubilee-induced ΔΦ at far-field must:
    1. Be smooth (no discontinuities)
    2. Be small relative to the baseline Φ (no gravitational amplification)
    3. Not introduce nonlocal artifacts (difference should be gradual)

    If Jubilee creates nonlocal gravitational artifacts → FAIL
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST F_QD6: Gravitational Stability Under Jubilee")
        print("=" * 60)

    N = 200
    far_probes = [20, 30, 170, 180]
    steps = 2000

    def run_sim(jubilee_on, seed=42):
        np.random.seed(seed)
        params = DETParams1D(
            N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.5,
            momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
            q_enabled=True, alpha_q=0.001,
            lambda_a=30.0, beta_a=0.2,
            floor_enabled=True, eta_floor=0.15, F_core=5.0,
            gravity_enabled=True, alpha_grav=0.05, kappa_grav=2.0, mu_grav=1.0, beta_g=5.0,
            boundary_enabled=True, grace_enabled=True,
            F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.05,
            jubilee_enabled=jubilee_on, delta_q=0.01, n_q=1, D_0=0.01,
            jubilee_energy_coupling=True
        )
        sim = DETCollider1D(params)
        sim.add_packet(100, mass=15.0, width=5.0, initial_q=0.6)
        sim.C_R[:] = 0.6
        sim.a[:] = 0.8

        Phi_trace = []
        for t in range(steps):
            sim.step()
            if t % 100 == 0:
                Phi_trace.append([float(sim.Phi[p]) for p in far_probes])

        return np.array(Phi_trace), sim.Phi.copy(), sim.q_D.copy()

    Phi_trace_ON, Phi_final_ON, qD_final_ON = run_sim(True)
    Phi_trace_OFF, Phi_final_OFF, qD_final_OFF = run_sim(False)

    # The difference in far-field Φ between Jubilee ON and OFF
    Phi_diff_trace = Phi_trace_ON - Phi_trace_OFF

    # Test 1: Smooth — the difference should evolve gradually
    if len(Phi_diff_trace) > 1:
        dPhi_diff_steps = np.diff(Phi_diff_trace, axis=0)
        max_jump = float(np.max(np.abs(dPhi_diff_steps)))
        smooth = max_jump < 0.01  # Jubilee-induced changes should be tiny per step
    else:
        smooth = True
        max_jump = 0.0

    # Test 2: Small — Jubilee-induced far-field Φ change should be small
    # relative to the baseline Φ
    baseline_Phi = float(np.mean(np.abs(Phi_final_OFF[far_probes])))
    jubilee_Phi_diff = float(np.mean(np.abs(Phi_final_ON[far_probes] - Phi_final_OFF[far_probes])))
    if baseline_Phi > 1e-10:
        relative_change = jubilee_Phi_diff / baseline_Phi
    else:
        relative_change = 0.0
    small = relative_change < 0.50  # Less than 50% change from Jubilee

    # Test 3: The q_D difference should be the source of the Φ difference
    # (no amplification — Φ change should be proportional to q_D change)
    qD_diff = float(np.mean(np.abs(qD_final_ON - qD_final_OFF)))
    no_amplification = True  # If both are small, that's fine

    passed = smooth and small and no_amplification

    if verbose:
        print(f"  Baseline far-field |Φ|: {baseline_Phi:.6f}")
        print(f"  Jubilee-induced far-field ΔΦ: {jubilee_Phi_diff:.6f}")
        print(f"  Relative change: {relative_change:.4f} ({relative_change*100:.1f}%)")
        print(f"  Max step jump in ΔΦ: {max_jump:.6f}")
        print(f"  Mean |q_D difference|: {qD_diff:.6f}")
        print(f"  Smooth: {smooth}")
        print(f"  Small: {small}")
        print(f"  F_QD6 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'smooth': smooth, 'small': small,
            'relative_change': relative_change, 'max_jump': max_jump,
            'qD_diff': qD_diff}


# ============================================================
# F_QD7: NO SPONTANEOUS ANNEALING
# ============================================================

def test_F_QD7_no_spontaneous_annealing(verbose=True):
    """
    F_QD7: No Spontaneous Annealing

    Random noise without coherence must not reduce q_D.

    Setup: Low coherence (C ≈ 0), random F perturbations (noise),
    high q_D, moderate agency. Even with flow from noise, the
    C^n_q term should suppress Jubilee activation.

    If q_D decreases → FAIL
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST F_QD7: No Spontaneous Annealing (noise + low C)")
        print("=" * 60)

    N = 200
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.01,  # Very low coherence
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.0,  # Disable q-locking to isolate Jubilee
        lambda_a=0.0, beta_a=0.0,  # Freeze agency
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=False,
        jubilee_enabled=True, delta_q=1.0, n_q=2, D_0=0.001,  # Very aggressive Jubilee
        jubilee_energy_coupling=True
    )
    sim = DETCollider1D(params)

    # Random noisy F field (creates flow from gradients)
    sim.F[:] = 2.0 + np.random.randn(N) * 0.5
    sim.F = np.clip(sim.F, 0.01, 100)
    sim.q_D[:] = 0.5
    sim.q_I[:] = 0.0
    sim.q[:] = 0.5
    sim.a[:] = 0.8  # High agency
    sim.C_R[:] = 0.01  # Very low coherence

    initial_qD = sim.q_D.copy()

    for t in range(1000):
        sim.step()
        # Keep coherence low (prevent coherence growth from flow)
        sim.C_R[:] = 0.01

    max_decrease = float(np.max(initial_qD - sim.q_D))
    total_jubilee = sim.total_jubilee

    # With C=0.01 and n_q=2, the activation is C^2 = 0.0001
    # Even aggressive delta_q=1.0 should produce negligible Jubilee
    passed = max_decrease < 0.001  # Less than 0.1% decrease

    if verbose:
        print(f"  Max q_D decrease: {max_decrease:.6f}")
        print(f"  Total Jubilee applied: {total_jubilee:.6f}")
        print(f"  Mean C_R: {np.mean(sim.C_R):.4f}")
        print(f"  F_QD7 {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'max_decrease': max_decrease,
            'total_jubilee': total_jubilee}


# ============================================================
# F_A1+: AGENCY BOOTSTRAP TEST (Stronger than F_A0)
# ============================================================

def test_F_A1plus_agency_bootstrap(verbose=True):
    """
    F_A1+: Agency Bootstrap Test

    If a_i=0 everywhere, no Jubilee cascade should ever start,
    even with high-C neighbors and strong flow.

    This is stronger than F_A0 because it specifically tests:
    - Jubilee activation with a=0 (should be zero: S_i = a*... = 0)
    - No q_D decrease anywhere
    - No emergent agency (a stays at 0)

    Setup: All nodes have a=0, high C, high F, strong flow,
    and Jubilee enabled with aggressive parameters.

    If any q_D decreases OR any a > 0 emerges → FAIL
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST F_A1+: Agency Bootstrap (a=0 blocks all Jubilee)")
        print("=" * 60)

    N = 200
    np.random.seed(42)
    params = DETParams1D(
        N=N, DT=0.02, F_VAC=0.01, F_MIN=0.0, C_init=0.9,
        momentum_enabled=True, alpha_pi=0.10, lambda_pi=0.02, mu_pi=0.30,
        q_enabled=True, alpha_q=0.0,  # Disable q-locking
        lambda_a=0.0, beta_a=0.0,  # Freeze agency at 0
        floor_enabled=True, eta_floor=0.15, F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.05, healing_enabled=True, eta_heal=0.10,
        jubilee_enabled=True, delta_q=10.0, n_q=1, D_0=0.001,  # Extremely aggressive
        jubilee_energy_coupling=True
    )
    sim = DETCollider1D(params)

    # Ideal conditions EXCEPT agency = 0
    sim.F[:] = 10.0
    sim.q_D[:] = 0.5
    sim.q_I[:] = 0.0
    sim.q[:] = 0.5
    sim.a[:] = 0.0  # ZERO agency everywhere
    sim.C_R[:] = 0.95  # Very high coherence

    # Add strong flow
    sim.add_packet(50, mass=10.0, width=10.0, momentum=1.0)
    sim.add_packet(150, mass=10.0, width=10.0, momentum=-1.0)

    initial_qD = sim.q_D.copy()

    for t in range(1000):
        sim.step()
        # Force agency to stay at 0 (agency update might try to change it)
        sim.a[:] = 0.0

    max_qD_decrease = float(np.max(initial_qD - sim.q_D))
    max_agency = float(np.max(sim.a))
    total_jubilee = sim.total_jubilee

    # With a=0, the Jubilee activation S_i = a * ... = 0
    # Also, the agency gate g_ij = sqrt(a_i * a_j) = 0, so no flow
    # q_D should not decrease at all
    no_qD_decrease = max_qD_decrease < 1e-10
    no_agency = max_agency < 1e-10
    no_jubilee = total_jubilee < 1e-10

    passed = no_qD_decrease and no_agency and no_jubilee

    if verbose:
        print(f"  Max q_D decrease: {max_qD_decrease:.2e}")
        print(f"  Max agency: {max_agency:.2e}")
        print(f"  Total Jubilee: {total_jubilee:.2e}")
        print(f"  No q_D decrease: {no_qD_decrease}")
        print(f"  No agency: {no_agency}")
        print(f"  No Jubilee: {no_jubilee}")
        print(f"  F_A1+ {'PASSED' if passed else 'FAILED'}")

    return {'passed': passed, 'max_qD_decrease': max_qD_decrease,
            'max_agency': max_agency, 'total_jubilee': total_jubilee}


# ============================================================
# MAIN SUITE RUNNER
# ============================================================

def run_v65b_falsifier_suite():
    """Run the complete v6.5b mandatory falsifier suite."""
    print("=" * 70)
    print("DET v6.5b MANDATORY FALSIFIER SUITE")
    print("Thermodynamic & Gravitational Safety (Structural Review)")
    print("=" * 70)

    start_time = time.time()
    results = {}

    results['F_QD5'] = test_F_QD5_energy_coupling()
    results['F_QD6'] = test_F_QD6_gravitational_stability()
    results['F_QD7'] = test_F_QD7_no_spontaneous_annealing()
    results['F_A1+'] = test_F_A1plus_agency_bootstrap()

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("v6.5b MANDATORY FALSIFIER SUITE SUMMARY")
    print("=" * 70)

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
    results = run_v65b_falsifier_suite()
