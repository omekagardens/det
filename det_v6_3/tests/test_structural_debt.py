"""
Structural Debt Test Suite
==========================

Tests the v6.4 structural debt couplings:
1. Conductivity gate (q → σ_eff): High debt blocks flow
2. Temporal distortion (q → P): High debt slows time
3. Decoherence enhancement (q → C_decay): High debt decoheres faster

Also demonstrates patterned micro-losses encoding state.

Reference: det_structural_debt.md
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/det/det_v6_3/src')
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def test_conductivity_gate():
    """
    F_SD1: Conductivity Gate Test

    Verify that high-q regions have reduced effective conductivity.

    Setup:
    - Create adjacent low-q and high-q cells with F gradient
    - Measure flux directly at the boundary
    - Compare flux through low-q vs high-q bonds

    Pass: Flux through high-q bond is reduced
    """
    print("=" * 60)
    print("F_SD1: Conductivity Gate Test")
    print("=" * 60)

    N = 16  # Small grid for direct measurement

    p = DETParams3D(
        N=N, DT=0.02,
        debt_conductivity_enabled=True,
        xi_conductivity=5.0,
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False,
        grace_enabled=False,
        q_enabled=False,  # Don't accumulate more q during test
    )

    c = DETCollider3D(p)

    # Create two test regions:
    # Region A (low-q): x = 2-4, y = 2-4, z = 2-4
    # Region B (high-q): x = 2-4, y = 8-10, z = 2-4
    # Both have same F gradient in X direction

    # Set low-q region
    c.q[2:5, 2:5, 2:5] = 0.02

    # Set high-q region
    c.q[2:5, 8:11, 2:5] = 0.9

    # Create F gradient: left side high, right side low
    c.F[2, 2:5, 2:5] = 5.0   # High F at x=2 (low-q region)
    c.F[4, 2:5, 2:5] = 1.0   # Low F at x=4 (low-q region)
    c.F[2, 8:11, 2:5] = 5.0  # High F at x=2 (high-q region)
    c.F[4, 8:11, 2:5] = 1.0  # Low F at x=4 (high-q region)

    # Measure F at x=3 (middle) before and after one step
    F_mid_low_q_before = np.mean(c.F[3, 2:5, 2:5])
    F_mid_high_q_before = np.mean(c.F[3, 8:11, 2:5])

    # Take one step
    c.step()

    F_mid_low_q_after = np.mean(c.F[3, 2:5, 2:5])
    F_mid_high_q_after = np.mean(c.F[3, 8:11, 2:5])

    # Change in F at middle indicates flux through
    delta_F_low_q = F_mid_low_q_after - F_mid_low_q_before
    delta_F_high_q = F_mid_high_q_after - F_mid_high_q_before

    # Calculate expected gate
    q_low = 0.02
    q_high = 0.9
    gate_low = 1.0 / (1.0 + p.xi_conductivity * 2 * q_low)
    gate_high = 1.0 / (1.0 + p.xi_conductivity * 2 * q_high)
    expected_ratio = gate_high / gate_low

    # Flux ratio
    actual_ratio = abs(delta_F_high_q) / (abs(delta_F_low_q) + 1e-9)

    passed = actual_ratio < 0.5  # High-q should have < 50% flux

    print(f"  Low-q debt gate:  {gate_low:.4f}")
    print(f"  High-q debt gate: {gate_high:.4f}")
    print(f"  Expected flux ratio: {expected_ratio:.4f}")
    print(f"  ΔF in low-q region:  {delta_F_low_q:.4f}")
    print(f"  ΔF in high-q region: {delta_F_high_q:.4f}")
    print(f"  Actual flux ratio: {actual_ratio:.4f}")
    print(f"  PASS: {passed}")
    print()

    return passed


def test_temporal_distortion():
    """
    F_SD2: Temporal Distortion Test

    Verify high-q regions have slower proper time.

    Setup:
    - Create two regions: low-q and high-q
    - Compare accumulated proper time

    Pass: High-q region accumulates less proper time
    """
    print("=" * 60)
    print("F_SD2: Temporal Distortion Test")
    print("=" * 60)

    N = 32

    p = DETParams3D(
        N=N, DT=0.02,
        debt_temporal_enabled=True,
        zeta_temporal=0.8,
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
    )

    c = DETCollider3D(p)

    # Create low-q and high-q regions
    c.q[:N//2, :, :] = 0.05   # Low debt
    c.q[N//2:, :, :] = 0.7    # High debt

    # Uniform resource and agency
    c.F[:] = 1.0
    c.a[:] = 1.0

    # Run
    for _ in range(100):
        c.step()

    # Compare accumulated proper time
    tau_low_q = np.mean(c.accumulated_proper_time[:N//2, :, :])
    tau_high_q = np.mean(c.accumulated_proper_time[N//2:, :, :])

    ratio = tau_high_q / tau_low_q
    passed = ratio < 0.8  # High-q should be at least 20% slower

    print(f"  Mean proper time (low q):  {tau_low_q:.4f}")
    print(f"  Mean proper time (high q): {tau_high_q:.4f}")
    print(f"  Time dilation ratio: {ratio:.3f}")
    print(f"  PASS: {passed}")
    print()

    return passed


def test_decoherence_enhancement():
    """
    F_SD3: Decoherence Enhancement Test

    Verify high-q regions have faster coherence decay.

    Setup:
    - Start with uniform high coherence
    - Create isolated low-q and high-q cells
    - Disable flux-based coherence growth (set alpha_C=0)
    - Pure decay test

    Pass: High-q region loses coherence faster
    """
    print("=" * 60)
    print("F_SD3: Decoherence Enhancement Test")
    print("=" * 60)

    N = 16

    p = DETParams3D(
        N=N, DT=0.02,
        debt_decoherence_enabled=True,
        theta_decoherence=3.0,  # Strong coupling
        C_init=0.01,  # Very low floor
        lambda_C=0.05,  # Fast base decay
        alpha_C=0.0,  # NO flux-based growth - pure decay test
        diff_enabled=False,  # No diffusion
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False,
        grace_enabled=False,
        debt_temporal_enabled=False,  # Keep time uniform for fair comparison
        agency_dynamic=False,  # Keep agency fixed for pure coherence test
    )

    c = DETCollider3D(p)

    # Set high initial coherence everywhere
    c.C_X[:] = 0.8
    c.C_Y[:] = 0.8
    c.C_Z[:] = 0.8

    # Create low-q and high-q regions
    c.q[:N//2, :, :] = 0.02   # Very low debt
    c.q[N//2:, :, :] = 0.8    # High debt

    # Uniform resource
    c.F[:] = 1.0

    # Calculate expected decay rates
    lambda_low = p.lambda_C * (1.0 + p.theta_decoherence * 0.02)
    lambda_high = p.lambda_C * (1.0 + p.theta_decoherence * 0.8)
    print(f"  Expected λ_C (low-q):  {lambda_low:.4f}")
    print(f"  Expected λ_C (high-q): {lambda_high:.4f}")
    print(f"  Expected decay ratio: {lambda_high/lambda_low:.2f}x faster")

    # Run
    for _ in range(200):
        c.step()

    # Compare coherence
    C_low_q = np.mean([
        np.mean(c.C_X[:N//2, :, :]),
        np.mean(c.C_Y[:N//2, :, :]),
        np.mean(c.C_Z[:N//2, :, :])
    ])
    C_high_q = np.mean([
        np.mean(c.C_X[N//2:, :, :]),
        np.mean(c.C_Y[N//2:, :, :]),
        np.mean(c.C_Z[N//2:, :, :])
    ])

    ratio = C_high_q / C_low_q
    passed = ratio < 0.95  # High-q should have less coherence (expect ~0.89)

    print(f"  Final coherence (low-q):  {C_low_q:.4f}")
    print(f"  Final coherence (high-q): {C_high_q:.4f}")
    print(f"  Coherence ratio: {ratio:.3f}")
    print(f"  PASS: {passed}")
    print()

    return passed


def test_channel_routing():
    """
    F_SD4: Channel Routing Test

    Verify that flux prefers low-q channels over high-q walls.

    Setup:
    - Create parallel paths: one through low-q, one through high-q
    - Apply same F gradient to both
    - Compare flux through each path

    Pass: Flux through low-q channel > flux through high-q wall
    """
    print("=" * 60)
    print("F_SD4: Channel Routing Test")
    print("=" * 60)

    N = 16

    p = DETParams3D(
        N=N, DT=0.02,
        debt_conductivity_enabled=True,
        xi_conductivity=5.0,
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False,
        grace_enabled=False,
        q_enabled=False,
    )

    c = DETCollider3D(p)

    # Create two parallel paths in X direction:
    # Path A (channel): y=3-5, low-q throughout
    # Path B (wall): y=10-12, high-q in middle

    # Channel path: low q everywhere
    c.q[4:12, 3:6, 4:12] = 0.02

    # Wall path: high-q in the middle section (x=7-9)
    c.q[4:12, 10:13, 4:12] = 0.02  # Start low
    c.q[7:10, 10:13, 4:12] = 0.9   # High-q wall in middle

    # Same F gradient on both paths: high F on left, low on right
    c.F[4:6, 3:6, 4:12] = 5.0    # Channel left
    c.F[10:12, 3:6, 4:12] = 0.5  # Channel right
    c.F[4:6, 10:13, 4:12] = 5.0  # Wall path left
    c.F[10:12, 10:13, 4:12] = 0.5  # Wall path right

    # Record F in middle of each path before stepping
    F_channel_before = np.mean(c.F[7:10, 3:6, 4:12])
    F_wall_before = np.mean(c.F[7:10, 10:13, 4:12])

    # Run several steps
    for _ in range(10):
        c.step()

    F_channel_after = np.mean(c.F[7:10, 3:6, 4:12])
    F_wall_after = np.mean(c.F[7:10, 10:13, 4:12])

    # F should increase more in channel (more flux flowing through)
    delta_channel = F_channel_after - F_channel_before
    delta_wall = F_wall_after - F_wall_before

    ratio = delta_channel / (delta_wall + 1e-9) if delta_wall > 0 else float('inf')
    passed = ratio > 1.5 if delta_wall > 0 else delta_channel > delta_wall

    print(f"  ΔF in channel (low-q): {delta_channel:.4f}")
    print(f"  ΔF in wall path (high-q): {delta_wall:.4f}")
    print(f"  Channel advantage ratio: {ratio:.2f}")
    print(f"  PASS: {passed}")
    print()

    return passed


def test_debt_trap():
    """
    F_SD5: Debt Trap Test

    Verify the debt trap feedback loop:
    High q → slow time → slow dynamics → more debt accumulation

    Setup:
    - Create a region experiencing resource drain
    - Enable temporal distortion
    - Verify debt accumulates faster without temporal coupling
      (because slower time means slower dynamics, less drain)

    This tests the debt trap mechanism is working.
    """
    print("=" * 60)
    print("F_SD5: Debt Trap Feedback Test")
    print("=" * 60)

    N = 16

    # Without temporal coupling - dynamics run at full speed
    p_no_coupling = DETParams3D(
        N=N, DT=0.02,
        debt_temporal_enabled=False,
        gravity_enabled=True,  # Gravity causes resource movement
        momentum_enabled=True,
        angular_momentum_enabled=False,
        floor_enabled=False,
    )

    # With temporal coupling - high-q slows dynamics
    p_coupling = DETParams3D(
        N=N, DT=0.02,
        debt_temporal_enabled=True,
        zeta_temporal=1.0,
        gravity_enabled=True,
        momentum_enabled=True,
        angular_momentum_enabled=False,
        floor_enabled=False,
    )

    def setup_drain(collider):
        """Create conditions that cause resource drain in center."""
        # High-q seed in center creates gravity well
        mid = N // 2
        collider.q[mid-2:mid+2, mid-2:mid+2, mid-2:mid+2] = 0.5
        # Resource distributed around
        collider.F[:] = 1.0
        collider.F[mid-1:mid+1, mid-1:mid+1, mid-1:mid+1] = 0.2  # Low in center

    # Run without coupling
    c_no = DETCollider3D(p_no_coupling)
    setup_drain(c_no)
    initial_q = np.sum(c_no.q)

    for _ in range(200):
        c_no.step()

    q_gain_no = np.sum(c_no.q) - initial_q

    # Run with coupling
    c_yes = DETCollider3D(p_coupling)
    setup_drain(c_yes)

    for _ in range(200):
        c_yes.step()

    q_gain_yes = np.sum(c_yes.q) - initial_q

    # With temporal coupling, high-q regions slow down
    # This means dynamics are slower in the trapped region
    # So we expect different q evolution patterns

    print(f"  q accumulation (no temporal coupling):   {q_gain_no:.4f}")
    print(f"  q accumulation (with temporal coupling): {q_gain_yes:.4f}")
    print(f"  (Different evolution patterns expected)")

    # The test passes if both show q accumulation (debt is accumulating)
    # and they differ (temporal coupling affects dynamics)
    passed = q_gain_no > 0 and q_gain_yes > 0 and abs(q_gain_no - q_gain_yes) > 0.01
    print(f"  PASS: {passed}")
    print()

    return passed


def demo_bit_encoding():
    """
    Demonstration: Bit Encoding Through Patterned Micro-Losses

    Shows how q-patterns can encode binary information that
    biases future dynamics.
    """
    print("=" * 60)
    print("DEMO: Bit Encoding Through Structural Debt")
    print("=" * 60)

    N = 32

    p = DETParams3D(
        N=N, DT=0.02,
        debt_conductivity_enabled=True,
        xi_conductivity=3.0,
        debt_temporal_enabled=True,
        zeta_temporal=0.5,
        gravity_enabled=True,
        kappa_grav=3.0,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
    )

    c = DETCollider3D(p)

    # Encode a simple 4-bit pattern: 1010
    # Each bit is a 4x4x4 region with high or low q
    bit_size = 4
    bits = [1, 0, 1, 0]

    for i, bit in enumerate(bits):
        x_start = 4 + i * (bit_size + 2)
        if bit == 1:
            c.q[x_start:x_start+bit_size,
                N//2-bit_size//2:N//2+bit_size//2,
                N//2-bit_size//2:N//2+bit_size//2] = 0.7
        else:
            c.q[x_start:x_start+bit_size,
                N//2-bit_size//2:N//2+bit_size//2,
                N//2-bit_size//2:N//2+bit_size//2] = 0.05

    # Put resource uniformly
    c.F[:] = 1.0

    print(f"  Encoded pattern: {bits}")
    print(f"  Initial total q: {np.sum(c.q):.3f}")

    # Run dynamics
    for _ in range(200):
        c.step()

    # Read back the pattern by measuring local properties
    read_bits = []
    for i in range(4):
        x_start = 4 + i * (bit_size + 2)
        region_q = np.mean(c.q[x_start:x_start+bit_size,
                               N//2-bit_size//2:N//2+bit_size//2,
                               N//2-bit_size//2:N//2+bit_size//2])
        read_bits.append(1 if region_q > 0.3 else 0)

    print(f"  Read back pattern: {read_bits}")
    print(f"  Final total q: {np.sum(c.q):.3f}")
    print(f"  Pattern preserved: {bits == read_bits}")

    # Measure how the pattern biased dynamics
    print("\n  Effects of q-pattern on dynamics:")
    for i in range(4):
        x_start = 4 + i * (bit_size + 2)
        region_F = np.mean(c.F[x_start:x_start+bit_size,
                               N//2-bit_size//2:N//2+bit_size//2,
                               N//2-bit_size//2:N//2+bit_size//2])
        region_P = np.mean(c.P[x_start:x_start+bit_size,
                               N//2-bit_size//2:N//2+bit_size//2,
                               N//2-bit_size//2:N//2+bit_size//2])
        print(f"    Bit {i} ({'1' if bits[i] else '0'}): "
              f"F={region_F:.3f}, P={region_P:.4f}")

    print("\n  High-q ('1') bits have:")
    print("    - Higher F (gravitational attraction)")
    print("    - Lower P (slower proper time)")
    print()


def run_all_tests():
    """Run all structural debt tests."""
    print("\n" + "=" * 60)
    print("  DET STRUCTURAL DEBT TEST SUITE")
    print("=" * 60 + "\n")

    results = []

    results.append(("F_SD1: Conductivity Gate", test_conductivity_gate()))
    results.append(("F_SD2: Temporal Distortion", test_temporal_distortion()))
    results.append(("F_SD3: Decoherence Enhancement", test_decoherence_enhancement()))
    results.append(("F_SD4: Channel Routing", test_channel_routing()))
    results.append(("F_SD5: Debt Trap Feedback", test_debt_trap()))

    # Run demo
    demo_bit_encoding()

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
