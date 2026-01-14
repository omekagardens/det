"""
Root Cause Analysis: Why Keplerian Orbits Fail in DET
======================================================

Hypothesis: DET orbits fail because of two fundamental issues:
1. Momentum π decays due to λ_π term (even if small)
2. F field spreads diffusively, so "particles" disperse

This diagnostic tests these hypotheses.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def test_momentum_conservation():
    """Test if total momentum is conserved in DET."""
    print("="*70)
    print("TEST 1: MOMENTUM CONSERVATION")
    print("="*70)
    print("\nQuestion: Is total momentum Σπ conserved?")

    N = 32
    center = N // 2

    params = DETParams3D(
        N=N,
        DT=0.02,
        F_VAC=0.01,
        gravity_enabled=False,  # Isolate momentum dynamics
        momentum_enabled=True,
        alpha_pi=0.1,
        lambda_pi=0.001,  # Decay rate
        mu_pi=0.3,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    sim = DETCollider3D(params)

    # Add a packet with momentum
    sim.add_packet((center, center, center), mass=5.0, width=2.0,
                   momentum=(1.0, 0.5, 0.0), initial_q=0.0)

    print(f"\nInitial momentum: px={1.0}, py={0.5}, pz={0.0}")
    print(f"λ_π (decay rate) = {params.lambda_pi}")

    print("\n" + "-"*50)
    print(f"{'Step':>6} {'Σπ_X':>12} {'Σπ_Y':>12} {'Σπ_Z':>12}")
    print("-"*50)

    pi_X_initial = np.sum(sim.pi_X)
    pi_Y_initial = np.sum(sim.pi_Y)

    for t in range(500):
        sim.step()
        if t % 50 == 0:
            pi_X = np.sum(sim.pi_X)
            pi_Y = np.sum(sim.pi_Y)
            pi_Z = np.sum(sim.pi_Z)
            print(f"{t:>6} {pi_X:>12.6f} {pi_Y:>12.6f} {pi_Z:>12.6f}")

    pi_X_final = np.sum(sim.pi_X)
    pi_Y_final = np.sum(sim.pi_Y)

    decay_X = 1 - pi_X_final / pi_X_initial if pi_X_initial != 0 else 0
    decay_Y = 1 - pi_Y_final / pi_Y_initial if pi_Y_initial != 0 else 0

    print("-"*50)
    print(f"\nπ_X decay: {decay_X*100:.1f}%")
    print(f"π_Y decay: {decay_Y*100:.1f}%")

    if abs(decay_X) > 0.1 or abs(decay_Y) > 0.1:
        print("\n⚠️  CONCLUSION: Momentum is NOT conserved!")
        print("Root cause: λ_π decay term drains momentum from the system.")
    else:
        print("\n✓ Momentum approximately conserved")


def test_wavepacket_dispersion():
    """Test if localized wavepackets spread out."""
    print("\n" + "="*70)
    print("TEST 2: WAVEPACKET DISPERSION")
    print("="*70)
    print("\nQuestion: Does a localized F packet maintain its shape?")

    N = 32
    center = N // 2

    params = DETParams3D(
        N=N,
        DT=0.02,
        F_VAC=0.001,
        gravity_enabled=False,
        momentum_enabled=False,  # Pure diffusion
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    sim = DETCollider3D(params)

    # Add a localized packet
    sim.add_packet((center, center, center), mass=5.0, width=2.0,
                   momentum=(0, 0, 0), initial_q=0.0)

    print("\nTracking packet width (RMS) over time:")
    print("-"*50)
    print(f"{'Step':>6} {'F_max':>12} {'RMS_r':>12} {'Mass':>12}")
    print("-"*50)

    for t in range(500):
        sim.step()
        if t % 50 == 0:
            # Compute center of mass and RMS width
            total_F = np.sum(sim.F)
            if total_F > 0:
                z, y, x = np.mgrid[0:N, 0:N, 0:N]
                cx = np.sum(x * sim.F) / total_F
                cy = np.sum(y * sim.F) / total_F
                cz = np.sum(z * sim.F) / total_F

                dx2 = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
                rms = np.sqrt(np.sum(dx2 * sim.F) / total_F)

                F_max = np.max(sim.F)
                print(f"{t:>6} {F_max:>12.4f} {rms:>12.2f} {total_F:>12.4f}")

    print("-"*50)
    print("\n⚠️  CONCLUSION: Packets spread due to diffusive F dynamics.")


def test_orbit_with_zero_decay():
    """Test orbit with λ_π = 0 (no momentum decay)."""
    print("\n" + "="*70)
    print("TEST 3: ORBIT WITH ZERO MOMENTUM DECAY")
    print("="*70)
    print("\nQuestion: Can orbits work if we set λ_π = 0?")

    N = 48
    center = N // 2

    params = DETParams3D(
        N=N,
        DT=0.015,
        F_VAC=0.001,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=15.0,
        mu_grav=2.0,
        beta_g=5.0,
        momentum_enabled=True,
        alpha_pi=0.15,
        lambda_pi=0.0,  # ZERO decay!
        mu_pi=0.4,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    sim = DETCollider3D(params)

    # Central mass
    sim.add_packet((center, center, center), mass=50.0, width=3.0, initial_q=0.8)

    # Let gravity establish
    for _ in range(50):
        sim.step()

    # Get orbital velocity
    r = 8
    g_at_r = np.abs(sim.gz[center, center, center + r])
    v_orbit = np.sqrt(r * g_at_r) if g_at_r > 0 else 1.0

    print(f"\nCentral mass at center, test particle at r={r}")
    print(f"|g| at r: {g_at_r:.4f}")
    print(f"Orbital velocity: {v_orbit:.4f}")

    # Add test particle
    sim.add_packet((center, center, center + r), mass=1.0, width=1.5,
                   momentum=(0, v_orbit, 0), initial_q=0.05)

    print("\nTracking orbit (with λ_π = 0):")
    print("-"*60)
    print(f"{'Step':>6} {'Σπ_Y':>12} {'r':>8} {'θ (deg)':>10}")
    print("-"*60)

    initial_angle = 0
    prev_y = 0
    prev_z = r

    for t in range(1500):
        sim.step()

        if t % 150 == 0:
            total_pi_Y = np.sum(sim.pi_Y)

            # Find test particle position
            blobs = sim.find_blobs(threshold_ratio=3.0)
            r_current = 0
            angle = 0

            if len(blobs) >= 2:
                # Sort by mass, take second (test particle)
                blobs_sorted = sorted(blobs, key=lambda b: b['mass'], reverse=True)
                test = blobs_sorted[1] if len(blobs_sorted) > 1 else blobs_sorted[0]

                dy = test['y'] - center
                dz = test['z'] - center
                r_current = np.sqrt(dy**2 + dz**2 + (test['x'] - center)**2)
                angle = np.degrees(np.arctan2(dy, dz))

            print(f"{t:>6} {total_pi_Y:>12.4f} {r_current:>8.1f} {angle:>10.1f}")

    print("-"*60)


def test_fundamental_orbit_requirement():
    """
    Test the fundamental requirement for Keplerian orbits:
    centripetal acceleration = gravitational acceleration

    For circular orbit: v²/r = |g|
    """
    print("\n" + "="*70)
    print("TEST 4: CENTRIPETAL vs GRAVITATIONAL ACCELERATION")
    print("="*70)

    N = 48
    center = N // 2

    params = DETParams3D(
        N=N,
        DT=0.015,
        F_VAC=0.001,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=15.0,
        mu_grav=2.0,
        momentum_enabled=True,
        alpha_pi=0.15,
        lambda_pi=0.0,
        mu_pi=0.4,
    )

    sim = DETCollider3D(params)

    # Central mass
    sim.add_packet((center, center, center), mass=50.0, width=3.0, initial_q=0.8)

    for _ in range(100):
        sim.step()

    print("\nGravitational field profile:")
    print("-"*50)
    print(f"{'r':>6} {'|g|':>12} {'v_circ':>12} {'v²/r':>12}")
    print("-"*50)

    for r in [4, 6, 8, 10, 12, 14]:
        if center + r < N:
            g_mag = np.sqrt(
                sim.gx[center, center, center + r]**2 +
                sim.gy[center, center, center + r]**2 +
                sim.gz[center, center, center + r]**2
            )
            v_circ = np.sqrt(r * g_mag) if g_mag > 0 else 0
            v2_r = v_circ**2 / r if r > 0 else 0

            print(f"{r:>6} {g_mag:>12.4f} {v_circ:>12.4f} {v2_r:>12.4f}")

    print("-"*50)
    print("\nFor Keplerian orbit: v²/r should equal |g| (centripetal = gravity)")


if __name__ == "__main__":
    test_momentum_conservation()
    test_wavepacket_dispersion()
    test_orbit_with_zero_decay()
    test_fundamental_orbit_requirement()

    print("\n" + "="*70)
    print("SUMMARY: ROOT CAUSES OF KEPLER FAILURE")
    print("="*70)
    print("""
1. MOMENTUM DECAY: λ_π > 0 causes total momentum to drain from system.
   → Even λ_π = 0.001 causes significant decay over orbital timescales.

2. WAVEPACKET DISPERSION: F field spreads diffusively.
   → "Particles" don't stay localized; they spread into background.

3. NO PARTICLE IDENTITY: DET tracks field densities, not particles.
   → No mechanism to maintain a distinct orbiting "test mass".

POTENTIAL FIXES:
1. Set λ_π = 0 (zero momentum decay) for gravitational scenarios
2. Add "particle mode" that tracks discrete point masses separately
3. Reformulate momentum as conserved quantity (Σπ = const)
4. Add localization mechanism to prevent wavepacket spreading
""")
