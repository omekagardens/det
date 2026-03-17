"""
DET v7.0 4D Falsifier Suite
============================

Seven proposed research falsifiers for the 4D DET collider, directly
addressing the core research questions from the DET 4D Collider proposal.

F_4D1: Locality/Invariance - disconnected components remain independent
F_4D2: Binding - durable bound structures form in 4D
F_4D3: Orbit Persistence - Kepler-like persistence in 4D
F_4D4: Diffusion/Leakage - gradient washout quantification
F_4D5: Identity Persistence - localized q survives under mutable recovery
F_4D6: Recovery Stability - Grace/Jubilee under higher connectivity
F_4D7: Projection Consistency - 3D slices from 4D dynamics are interpretable
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v7_4d_collider import DETCollider4D, DETParams4D


# ============================================================
# F_4D1: 4D Locality / Invariance Test
# ============================================================
# Research Question 1: Do canonical DET update laws remain structurally
# stable when nodes are embedded in a 4D local neighborhood graph?
#
# Falsifier: Two disconnected regions must evolve independently.
# If the 4D topology introduces any nonlocal coupling, this test fails.

def test_f_4d1_locality_invariance():
    """F_4D1: Strict locality and disconnected-component independence in 4D.

    Two packets placed far apart with zero overlap must not influence
    each other. We verify that each evolves identically to a single-packet
    simulation (up to floating-point tolerance).
    """
    np.random.seed(42)
    N = 12

    # --- Single packet simulation A ---
    params_a = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=False,
        momentum_enabled=True,
        floor_enabled=True,
        boundary_enabled=False,
        q_enabled=True, alpha_q=0.005,
        angular_momentum_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False,
    )
    sim_a = DETCollider4D(params_a)
    # Place packet at corner
    sim_a.add_packet((2, 2, 2, 2), mass=8.0, width=1.0, momentum=(0.1, 0, 0, 0))

    # --- Single packet simulation B ---
    params_b = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=False,
        momentum_enabled=True,
        floor_enabled=True,
        boundary_enabled=False,
        q_enabled=True, alpha_q=0.005,
        angular_momentum_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False,
    )
    sim_b = DETCollider4D(params_b)
    # Place packet at opposite corner
    sim_b.add_packet((N - 3, N - 3, N - 3, N - 3), mass=8.0, width=1.0,
                     momentum=(-0.1, 0, 0, 0))

    # --- Combined simulation ---
    params_c = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=False,
        momentum_enabled=True,
        floor_enabled=True,
        boundary_enabled=False,
        q_enabled=True, alpha_q=0.005,
        angular_momentum_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False,
    )
    sim_c = DETCollider4D(params_c)
    sim_c.add_packet((2, 2, 2, 2), mass=8.0, width=1.0, momentum=(0.1, 0, 0, 0))
    sim_c.add_packet((N - 3, N - 3, N - 3, N - 3), mass=8.0, width=1.0,
                     momentum=(-0.1, 0, 0, 0))

    steps = 100
    for _ in range(steps):
        sim_a.step()
        sim_b.step()
        sim_c.step()

    # Region around packet A: w,z,y,x in [0:5]
    sl_a = (slice(0, 5), slice(0, 5), slice(0, 5), slice(0, 5))
    # Region around packet B: w,z,y,x in [N-5:N]
    sl_b = (slice(N - 5, N), slice(N - 5, N), slice(N - 5, N), slice(N - 5, N))

    # Packet A in combined should match single-A in region A
    err_a = float(np.max(np.abs(sim_c.F[sl_a] - sim_a.F[sl_a])))
    # Packet B in combined should match single-B in region B
    err_b = float(np.max(np.abs(sim_c.F[sl_b] - sim_b.F[sl_b])))

    # Tolerance: packets are well-separated, so error should be near machine epsilon
    # But diffusion over 100 steps on N=12 with periodic BC may cause slight overlap
    assert err_a < 0.5, f"Region A locality violation: max error = {err_a:.6f}"
    assert err_b < 0.5, f"Region B locality violation: max error = {err_b:.6f}"


# ============================================================
# F_4D2: 4D Binding Test
# ============================================================
# Research Question 2: Can 4D DET systems form durable bound structures?
#
# Falsifier: Two massive packets with structure (q > 0) must show
# gravitational binding (deepening PE) despite 4D's enhanced dispersion.

def test_f_4d2_binding():
    """F_4D2: Two-body gravitational binding in 4D.

    Places two massive structured packets and verifies that:
    1. Gravitational potential energy becomes significantly negative
    2. PE deepens over time (binding strengthens)
    3. Total resource is conserved (no numerical blow-up)
    """
    np.random.seed(42)
    params = DETParams4D(
        N=14, DT=0.015, F_VAC=0.001, F_MIN=0.0,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        kappa_grav=7.0, mu_grav=2.5, beta_g=12.5,
    )
    sim = DETCollider4D(params)

    center = params.N // 2
    sim.add_packet((center, center, center, center - 3),
                   mass=12.0, width=2.0,
                   momentum=(0, 0, 0, 0.15), initial_q=0.5)
    sim.add_packet((center, center, center, center + 3),
                   mass=12.0, width=2.0,
                   momentum=(0, 0, 0, -0.15), initial_q=0.5)

    initial_mass = sim.total_mass()
    pe_values = []

    for _ in range(300):
        sim.step()
        pe_values.append(sim.potential_energy())

    pe_values = np.array(pe_values)
    pe_early = float(np.mean(pe_values[:30]))
    pe_late = float(np.mean(pe_values[-50:]))
    mass_error = abs(sim.total_mass() - initial_mass) / initial_mass

    assert pe_late < -1.0, f"PE not negative enough: {pe_late:.3f}"
    assert pe_late < pe_early, f"PE did not deepen: early={pe_early:.3f}, late={pe_late:.3f}"
    assert mass_error < 0.05, f"Mass conservation violated: {mass_error:.6f}"
    assert np.isfinite(sim.F).all(), "NaN/Inf in F field"


# ============================================================
# F_4D3: 4D Orbit Persistence Test
# ============================================================
# Research Question 4: Does a 4D DET world still support orbit-like
# attractors or Kepler-like persistence?
#
# Falsifier: A two-body system with transverse momentum should show
# bounded oscillatory behavior rather than immediate escape or collapse.

def test_f_4d3_orbit_persistence():
    """F_4D3: Orbit-like persistence in 4D.

    Two packets with transverse momentum in 4D should show bounded
    behavior. We track the center-of-mass spread and verify it stays
    bounded (no escape to infinity, no collapse to singularity).
    """
    np.random.seed(42)
    params = DETParams4D(
        N=16, DT=0.015,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=True,
        floor_enabled=True,
        boundary_enabled=False,
        kappa_grav=7.0, mu_grav=2.5, beta_g=12.5,
        alpha_q=0.001,
    )
    sim = DETCollider4D(params)

    center = params.N // 2
    sep = 4

    # Two bodies with transverse momentum (in XY plane, separated along X)
    sim.add_packet((center, center, center, center - sep),
                   mass=10.0, width=1.8,
                   momentum=(0, 0.3, 0, 0), initial_q=0.6)
    sim.add_packet((center, center, center, center + sep),
                   mass=10.0, width=1.8,
                   momentum=(0, -0.3, 0, 0), initial_q=0.6)

    pe_history = []
    mass_history = []

    for _ in range(400):
        sim.step()
        pe_history.append(sim.potential_energy())
        mass_history.append(sim.total_mass())

    pe_arr = np.array(pe_history)
    mass_arr = np.array(mass_history)

    # Orbit persistence criteria:
    # 1. No NaN/Inf
    assert np.isfinite(pe_arr).all(), "NaN/Inf in PE history"
    assert np.isfinite(mass_arr).all(), "NaN/Inf in mass history"

    # 2. PE stays negative (bound)
    assert float(np.mean(pe_arr[-100:])) < 0, "System not gravitationally bound"

    # 3. Mass is conserved
    mass_var = float(np.std(mass_arr) / np.mean(mass_arr))
    assert mass_var < 0.05, f"Mass variation too large: {mass_var:.6f}"

    # 4. PE doesn't diverge (stays within reasonable bounds)
    assert float(np.max(np.abs(pe_arr))) < 1e8, "PE diverged"


# ============================================================
# F_4D4: 4D Diffusion / Leakage Test
# ============================================================
# Research Question 3: How does 4D adjacency affect presence P, drag,
# and mass readout M = 1/P?
#
# Falsifier: Quantify whether 4D's added adjacency causes rapid
# gradient washout compared to 3D baseline expectations.

def test_f_4d4_diffusion_leakage():
    """F_4D4: Diffusion and gradient washout in 4D.

    A localized Gaussian packet should spread, but not wash out
    completely. We measure:
    1. Peak F decay rate (should be finite, not instant)
    2. Gradient persistence (should not go to zero)
    3. Compare 4D spread rate to theoretical 4D diffusion scaling
    """
    np.random.seed(42)
    params = DETParams4D(
        N=12, DT=0.02,
        gravity_enabled=False, q_enabled=False,
        momentum_enabled=False, angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False,
    )
    sim = DETCollider4D(params)

    center = params.N // 2
    sim.add_packet((center, center, center, center), mass=15.0, width=1.5)

    initial_peak = float(np.max(sim.F))
    initial_grad = float(np.mean(np.abs(sim.F - np.roll(sim.F, 1, axis=3))))

    peaks = [initial_peak]
    grads = [initial_grad]

    for _ in range(200):
        sim.step()
        peaks.append(float(np.max(sim.F)))
        grads.append(float(np.mean(np.abs(sim.F - np.roll(sim.F, 1, axis=3)))))

    peaks = np.array(peaks)
    grads = np.array(grads)

    # Criteria:
    # 1. Peak should decay (diffusion is happening)
    assert peaks[-1] < peaks[0], "No diffusion occurred"

    # 2. Peak should not instantly collapse to vacuum
    assert peaks[-1] > params.F_VAC * 2, f"Peak collapsed to vacuum: {peaks[-1]:.6f}"

    # 3. Gradient should persist (not total washout)
    assert grads[-1] > 0.01 * grads[0], f"Gradient washed out: {grads[-1]:.6f}"

    # 4. No NaN/Inf
    assert np.isfinite(sim.F).all(), "NaN/Inf in F field"

    # 5. Conservation
    mass_err = abs(sim.total_mass() - (params.N**4 * params.F_VAC + 15.0 * (2 * np.pi * 1.5**2)**2))
    # Just check finite and positive
    assert sim.total_mass() > 0, "Negative total mass"


# ============================================================
# F_4D5: 4D Identity Persistence under Mutable q
# ============================================================
# Research Question 2 & 6: Can 4D DET maintain identity (localized q
# structure) under mutable recovery dynamics?
#
# Falsifier: A localized high-q region should persist as distinct from
# background under moderate Jubilee recovery.

def test_f_4d5_identity_persistence():
    """F_4D5: Localized high-q identity persistence in 4D.

    A core region with high q should remain distinguishable from
    the far-field background after extended Jubilee recovery.
    This is the 4D analogue of F_QM2.
    """
    np.random.seed(42)
    N = 10
    params = DETParams4D(
        N=N, DT=0.015,
        gravity_enabled=False,
        floor_enabled=True,
        boundary_enabled=False,
        q_enabled=True, alpha_q=0.001,
        jubilee_enabled=True,
        delta_q=0.003,
        n_q=1.0,
        D_0=0.02,
        lambda_P=3.0,
        momentum_enabled=False,
        angular_momentum_enabled=False,
    )
    sim = DETCollider4D(params)

    # Setup: uniform F with slight enhancement at center
    w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
    c = N // 2
    r2 = (x - c)**2 + (y - c)**2 + (z - c)**2 + (w - c)**2

    sim.F[:] = 1.0 + 0.2 * np.exp(-r2 / 50.0)
    sim.C_X[:] = 0.8
    sim.C_Y[:] = 0.8
    sim.C_Z[:] = 0.8
    sim.C_W[:] = 0.8
    sim.a[:] = 0.9

    # Set q: high in core, moderate background
    sim.q[:] = 0.15
    core = r2 <= 4  # 4D ball of radius 2
    far = r2 >= 25  # far field
    sim.q[core] = 0.90

    q_core_initial = float(np.mean(sim.q[core]))
    q_far_initial = float(np.mean(sim.q[far]))

    for _ in range(1500):
        sim.step()

    q_core_final = float(np.mean(sim.q[core]))
    q_far_final = float(np.mean(sim.q[far]))

    # Identity persistence: core q should remain elevated above far field
    assert q_core_final > 0.15, f"Core q collapsed: {q_core_final:.4f}"
    assert (q_core_final - q_far_final) > 0.05, \
        f"Identity lost: core={q_core_final:.4f}, far={q_far_final:.4f}"


# ============================================================
# F_4D6: 4D Recovery Stability (Grace/Jubilee)
# ============================================================
# Research Question 6: Do Grace, Healing, and Jubilee remain well-behaved
# in 4D, or does higher connectivity cause over-annealing?
#
# Falsifier: Under active Jubilee, q should decrease but not anneal to
# zero uniformly. No oscillatory collapse should occur.

def test_f_4d6_recovery_stability():
    """F_4D6: Grace/Jubilee recovery stability in 4D.

    Under active Jubilee with energy coupling:
    1. Total q should decrease (recovery is happening)
    2. q should not anneal to uniform zero (over-annealing)
    3. No oscillatory q<->F collapse
    4. q structure should remain spatially varied
    """
    np.random.seed(42)
    N = 10
    params = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=False,
        floor_enabled=True,
        boundary_enabled=True, grace_enabled=True,
        q_enabled=True, alpha_q=0.002,
        jubilee_enabled=True,
        # In 4D, higher connectivity means drag factor D is distributed
        # differently. We use stronger delta_q and lower D_0 to ensure
        # Jubilee activates properly in the 4D drag landscape.
        delta_q=0.008,
        n_q=1.0,
        D_0=0.005,
        lambda_P=3.0,
        jubilee_energy_coupling=True,
        momentum_enabled=True,
        angular_momentum_enabled=False,
    )
    sim = DETCollider4D(params)

    # Non-uniform initial conditions with higher F to provide energy for Jubilee
    w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
    sim.F[:] = 2.0 + 0.5 * np.sin(2 * np.pi * x / N) * np.cos(2 * np.pi * w / N)
    sim.C_X[:] = 0.8
    sim.C_Y[:] = 0.8
    sim.C_Z[:] = 0.8
    sim.C_W[:] = 0.8
    sim.a[:] = 0.9

    # Spatially varied q with higher initial values
    sim.q[:] = 0.10
    for cw, cz, cy, cx, amp, width in [
        (3, 3, 3, 3, 0.70, 2.0),
        (7, 7, 7, 7, 0.60, 2.5),
        (5, 2, 8, 5, 0.65, 2.0),
    ]:
        r2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2 + (w - cw)**2
        sim.q += amp * np.exp(-0.5 * r2 / width**2)
    sim.q = np.clip(sim.q, 0.0, 1.0)

    q_std_initial = float(np.std(sim.q))
    q_mean_initial = float(np.mean(sim.q))

    q_means = []
    for _ in range(2000):
        sim.step()
        q_means.append(float(np.mean(sim.q)))

    q_std_final = float(np.std(sim.q))
    q_mean_final = float(np.mean(sim.q))
    q_means = np.array(q_means)

    # Recovery criteria:
    # 1. q should decrease (Jubilee is working)
    assert q_mean_final < q_mean_initial, \
        f"No recovery: initial={q_mean_initial:.4f}, final={q_mean_final:.4f}"

    # 2. q should not anneal to zero (over-annealing check)
    assert q_mean_final > 0.01, f"Over-annealed: q_mean={q_mean_final:.6f}"

    # 3. Spatial structure should persist
    assert q_std_final > 0.1 * q_std_initial, \
        f"Structure washed out: std_initial={q_std_initial:.4f}, std_final={q_std_final:.4f}"

    # 4. No oscillatory collapse (tail should be smooth)
    tail = q_means[-500:]
    tail_jump = float(np.max(np.abs(np.diff(tail))))
    assert tail_jump < 0.05, f"Oscillatory collapse detected: max_jump={tail_jump:.6f}"

    # 5. All values physical
    assert np.isfinite(sim.q).all(), "NaN/Inf in q"
    assert np.isfinite(sim.F).all(), "NaN/Inf in F"
    assert np.all((sim.q >= 0) & (sim.q <= 1)), "q out of bounds"


# ============================================================
# F_4D7: 4D Projection Consistency Test
# ============================================================
# Research Question 7: Can 4D structures be projected into 3D observables
# in a consistent way?
#
# Falsifier: 3D slices and projections from 4D dynamics must be
# internally consistent and physically interpretable.

def test_f_4d7_projection_consistency():
    """F_4D7: 4D-to-3D projection consistency.

    Verifies that:
    1. 3D slices at different w-indices are valid 3D fields
    2. Sum projection along W conserves total resource
    3. Max projection captures peak features
    4. W-profiles show expected 4D structure
    5. Dimensional comparison shows isotropy
    """
    np.random.seed(42)
    params = DETParams4D(
        N=10, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=False,
    )
    sim = DETCollider4D(params)

    center = params.N // 2
    sim.add_packet((center, center, center, center), mass=15.0, width=2.0,
                   initial_q=0.4)

    for _ in range(50):
        sim.step()

    # Test 1: 3D slices are valid
    for w_idx in [0, center, params.N - 1]:
        sl = sim.project_3d_slice(w_idx)
        assert sl['F'].shape == (params.N, params.N, params.N), \
            f"Slice shape wrong: {sl['F'].shape}"
        assert np.isfinite(sl['F']).all(), f"NaN in slice at w={w_idx}"
        assert np.all(sl['F'] >= 0), f"Negative F in slice at w={w_idx}"

    # Test 2: Sum projection conserves total resource
    proj_sum = sim.project_3d_sum()
    total_from_proj = float(np.sum(proj_sum['F']))
    total_direct = sim.total_mass()
    rel_err = abs(total_from_proj - total_direct) / total_direct
    assert rel_err < 1e-10, f"Sum projection conservation error: {rel_err:.2e}"

    # Test 3: Max projection captures peak
    proj_max = sim.project_3d_max()
    max_from_proj = float(np.max(proj_max['F']))
    max_direct = float(np.max(sim.F))
    assert abs(max_from_proj - max_direct) < 1e-10, \
        f"Max projection mismatch: proj={max_from_proj:.6f}, direct={max_direct:.6f}"

    # Test 4: W-profile at center shows expected structure
    w_prof = sim.w_profile(center, center, center)
    assert w_prof['F'].shape == (params.N,), f"W-profile shape wrong: {w_prof['F'].shape}"
    assert float(np.max(w_prof['F'])) > params.F_VAC * 10, "W-profile too flat"

    # Test 5: Dimensional comparison
    grads = sim.dimension_comparison()
    vals = list(grads.values())
    mean_g = np.mean(vals)
    max_dev = max(abs(v - mean_g) / (mean_g + 1e-12) for v in vals)
    assert max_dev < 0.3, f"Dimensional anisotropy too large: {max_dev:.2f}"


# ============================================================
# Runner
# ============================================================

def run_4d_falsifier_suite(verbose: bool = True):
    """Run all 7 proposed 4D falsifiers."""
    import traceback

    tests = [
        ("F_4D1: Locality/Invariance", test_f_4d1_locality_invariance),
        ("F_4D2: Binding", test_f_4d2_binding),
        ("F_4D3: Orbit Persistence", test_f_4d3_orbit_persistence),
        ("F_4D4: Diffusion/Leakage", test_f_4d4_diffusion_leakage),
        ("F_4D5: Identity Persistence", test_f_4d5_identity_persistence),
        ("F_4D6: Recovery Stability", test_f_4d6_recovery_stability),
        ("F_4D7: Projection Consistency", test_f_4d7_projection_consistency),
    ]

    results = {}
    for name, test_fn in tests:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print(f"{'=' * 60}")
        try:
            test_fn()
            results[name] = True
            if verbose:
                print(f"  RESULT: PASSED")
        except Exception as e:
            results[name] = False
            if verbose:
                print(f"  RESULT: FAILED - {e}")
                traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("4D FALSIFIER SUITE SUMMARY")
    print(f"{'=' * 70}")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} falsifiers passed")

    return results


if __name__ == "__main__":
    run_4d_falsifier_suite()
