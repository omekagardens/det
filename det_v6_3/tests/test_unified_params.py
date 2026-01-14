"""
Test DET with Unified Parameter Schema
======================================

Verify that the unified parameter derivation:
1. Matches legacy parameter values exactly
2. Can be used in simulations without changing physics

Note: The comprehensive falsifiers (det_comprehensive_falsifiers.py) use
custom-tuned parameters for each test scenario. This test verifies that
the unified schema produces correct default values.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_unified_params import DETUnifiedParams
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def test_parameter_equivalence():
    """Verify unified params match legacy defaults exactly."""
    print("Testing: Parameter Equivalence")
    print("-" * 50)

    unified = DETUnifiedParams()

    # Legacy values from det_v6_3_3d_collider.py defaults
    legacy = {
        'alpha_pi': 0.12,
        'lambda_pi': 0.008,
        'mu_pi': 0.35,
        'alpha_L': 0.06,
        'lambda_L': 0.005,
        'mu_L': 0.18,
        'eta_floor': 0.12,
        'F_core': 5.0,
        'floor_power': 2.0,
        'alpha_q': 0.012,
        'beta_a': 0.2,
        'gamma_a_max': 0.15,
        'gamma_a_power': 2.0,
        'alpha_C': 0.04,
        'lambda_C': 0.002,
        'C_init': 0.15,
        'alpha_grav': 0.02,
        'kappa_grav': 5.0,
        'mu_grav': 2.0,
        'beta_g': 10.0,
    }

    all_match = True
    mismatches = []

    for name, leg_val in legacy.items():
        uni_val = getattr(unified, name)
        match = abs(leg_val - uni_val) < 0.001
        if not match:
            mismatches.append((name, leg_val, uni_val))
            all_match = False

    if all_match:
        print("  All 20 parameters match legacy defaults")
    else:
        for name, leg, uni in mismatches:
            print(f"  MISMATCH: {name} legacy={leg} unified={uni}")

    return all_match


def test_legacy_conversion():
    """Test conversion to legacy DETParams3D format."""
    print("\nTesting: Legacy Conversion")
    print("-" * 50)

    unified = DETUnifiedParams(N=32, gravity_enabled=True)
    legacy = unified.to_legacy_params()

    # Verify it's the right type
    is_correct_type = isinstance(legacy, DETParams3D)

    # Verify key values transferred
    values_match = (
        legacy.N == 32 and
        legacy.gravity_enabled == True and
        abs(legacy.alpha_pi - 0.12) < 0.001 and
        abs(legacy.kappa_grav - 5.0) < 0.001
    )

    passed = is_correct_type and values_match
    print(f"  Type correct: {is_correct_type}")
    print(f"  Values transferred: {values_match}")

    return passed


def test_simulation_runs():
    """Test that unified params can run a simulation."""
    print("\nTesting: Simulation Execution")
    print("-" * 50)

    unified = DETUnifiedParams(N=24)
    legacy = unified.to_legacy_params()
    sim = DETCollider3D(legacy)

    sim.add_packet((12, 12, 12), mass=10.0, width=2.0)

    initial_mass = np.sum(sim.F)

    # Run 100 steps
    for _ in range(100):
        sim.step()

    final_mass = np.sum(sim.F)
    drift = abs(final_mass - initial_mass) / initial_mass

    passed = drift < 0.01  # Less than 1% mass drift
    print(f"  Initial mass: {initial_mass:.2f}")
    print(f"  Final mass: {final_mass:.2f}")
    print(f"  Drift: {drift*100:.4f}%")

    return passed


def test_locality():
    """Test locality with unified params."""
    print("\nTesting: Locality (F1)")
    print("-" * 50)

    params = DETUnifiedParams(N=32)
    legacy = params.to_legacy_params()
    sim = DETCollider3D(legacy)

    # Add pulse at center
    sim.add_packet((16, 16, 16), mass=10.0, width=1.5)

    initial_F_far = sim.F[16, 16, 26].copy()  # 10 cells away

    # One step
    sim.step()

    final_F_far = sim.F[16, 16, 26]

    # Should not have propagated 10 cells in 1 step
    propagated = abs(final_F_far - initial_F_far) > 0.001

    print(f"  Far cell change: {abs(final_F_far - initial_F_far):.6f}")
    print(f"  Locality preserved: {not propagated}")
    return not propagated


def test_mass_conservation():
    """Test mass conservation with unified params."""
    print("\nTesting: Mass Conservation (F7)")
    print("-" * 50)

    params = DETUnifiedParams(N=32, boundary_enabled=False)
    legacy = params.to_legacy_params()
    sim = DETCollider3D(legacy)

    sim.add_packet((16, 16, 16), mass=50.0, width=3.0)

    initial_mass = np.sum(sim.F)

    for _ in range(500):
        sim.step()

    final_mass = np.sum(sim.F)
    drift = abs(final_mass - initial_mass) / initial_mass

    print(f"  Initial: {initial_mass:.4f}, Final: {final_mass:.4f}")
    print(f"  Drift: {drift*100:.4f}%")
    return drift < 0.001


def test_time_dilation():
    """Test gravitational time dilation with unified params."""
    print("\nTesting: Time Dilation (F_GTD1)")
    print("-" * 50)

    params = DETUnifiedParams(N=32, gravity_enabled=True)
    legacy = params.to_legacy_params()
    sim = DETCollider3D(legacy)

    center = 16
    sim.add_packet((center, center, center), mass=50.0, width=2.5, initial_q=0.8)

    for _ in range(200):
        sim.step()

    # Check presence at center vs edge
    P_center = sim.P[center, center, center]
    P_edge = sim.P[center, center, center + 10]

    dilation = P_center < P_edge  # Should tick slower at center (more F)

    print(f"  P_center: {P_center:.4f}, P_edge: {P_edge:.4f}")
    print(f"  Time dilation correct: {dilation}")
    return dilation


def run_all_tests():
    """Run all unified parameter tests."""
    print("=" * 70)
    print("UNIFIED PARAMETER SCHEMA TESTS")
    print("=" * 70)
    print("\nNote: Physics falsifiers (F6, F_L3) use custom-tuned parameters.")
    print("See det_comprehensive_falsifiers.py for full physics tests.\n")

    results = {}

    results['Param_Equivalence'] = test_parameter_equivalence()
    results['Legacy_Conversion'] = test_legacy_conversion()
    results['Simulation_Runs'] = test_simulation_runs()
    results['F1_Locality'] = test_locality()
    results['F7_MassConservation'] = test_mass_conservation()
    results['F_GTD1_TimeDilation'] = test_time_dilation()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("-" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED with unified parameters!")
    else:
        print("✗ Some tests failed")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
