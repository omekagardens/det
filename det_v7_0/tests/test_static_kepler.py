"""
Test Kepler's Law with STATIC DET Gravity Field
================================================

Key change: DON'T evolve the DET field during orbit.
Just use the static Φ field established initially.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def get_gravity_interpolated(sim, x, y, z):
    """Get interpolated gravity at position."""
    N = sim.p.N

    x = x % N
    y = y % N
    z = z % N

    ix = int(x)
    iy = int(y)
    iz = int(z)

    fx = x - ix
    fy = y - iy
    fz = z - iz

    ix1 = (ix + 1) % N
    iy1 = (iy + 1) % N
    iz1 = (iz + 1) % N

    def interp(field):
        c000 = field[iz, iy, ix]
        c001 = field[iz, iy, ix1]
        c010 = field[iz, iy1, ix]
        c011 = field[iz, iy1, ix1]
        c100 = field[iz1, iy, ix]
        c101 = field[iz1, iy, ix1]
        c110 = field[iz1, iy1, ix]
        c111 = field[iz1, iy1, ix1]

        c00 = c000 * (1 - fx) + c001 * fx
        c01 = c010 * (1 - fx) + c011 * fx
        c10 = c100 * (1 - fx) + c101 * fx
        c11 = c110 * (1 - fx) + c111 * fx

        c0 = c00 * (1 - fy) + c01 * fy
        c1 = c10 * (1 - fy) + c11 * fy

        return c0 * (1 - fz) + c1 * fz

    return interp(sim.gx), interp(sim.gy), interp(sim.gz)


def test_kepler_static_field():
    """Test Kepler with static (frozen) DET gravity field."""

    print("="*70)
    print("KEPLER TEST WITH STATIC DET GRAVITY FIELD")
    print("="*70)

    N = 64
    center = N // 2

    # Create DET simulation
    params = DETParams3D(
        N=N,
        DT=0.01,
        F_VAC=0.001,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=20.0,
        mu_grav=2.0,
        q_enabled=True,
        alpha_q=0.0,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    sim = DETCollider3D(params)

    # Add massive central body
    sim.add_packet((center, center, center), mass=100.0, width=3.0,
                   momentum=(0, 0, 0), initial_q=0.9)

    # Establish gravity field
    print("\nEstablishing gravity field...")
    for _ in range(200):
        sim.step()

    # FREEZE the fields by storing them
    gx_frozen = sim.gx.copy()
    gy_frozen = sim.gy.copy()
    gz_frozen = sim.gz.copy()
    Phi_frozen = sim.Phi.copy()

    # Check gravity profile
    print("\nStatic gravitational field profile:")
    print("-"*60)
    print(f"{'r':>6} {'|g|':>12} {'|g|*r²':>12} {'Φ':>12}")
    print("-"*60)

    g_r2_values = []
    for r in [4, 6, 8, 10, 12, 14, 16]:
        if center + r < N:
            g_mag = np.sqrt(
                gx_frozen[center, center, center + r]**2 +
                gy_frozen[center, center, center + r]**2 +
                gz_frozen[center, center, center + r]**2
            )
            g_r2 = g_mag * r**2
            g_r2_values.append(g_r2)
            phi = Phi_frozen[center, center, center + r]
            print(f"{r:>6} {g_mag:>12.4f} {g_r2:>12.4f} {phi:>12.4f}")

    print("-"*60)
    mean_gr2 = np.mean(g_r2_values)
    std_gr2 = np.std(g_r2_values)
    cv_gr2 = std_gr2 / mean_gr2 if mean_gr2 > 0 else 0
    print(f"Mean |g|*r² = {mean_gr2:.4f}, CV = {cv_gr2:.3f}")

    if cv_gr2 > 0.3:
        print("⚠️  Gravity deviates from 1/r² - this will affect Kepler's law")

    # Test orbits at different radii
    print("\n" + "="*70)
    print("ORBIT TESTS")
    print("="*70)

    results = []

    for r_orbit in [6, 8, 10, 12]:
        print(f"\n--- Testing orbit at r = {r_orbit} ---")

        # Get circular velocity
        g_at_r = np.sqrt(
            gx_frozen[center, center, center + r_orbit]**2 +
            gy_frozen[center, center, center + r_orbit]**2 +
            gz_frozen[center, center, center + r_orbit]**2
        )
        v_circ = np.sqrt(r_orbit * g_at_r) if g_at_r > 0 else 1.0

        print(f"|g| at r={r_orbit}: {g_at_r:.4f}, v_circ = {v_circ:.4f}")

        # Initialize particle: position offset in Z, velocity tangential in Y
        px, py, pz = float(center), float(center), float(center + r_orbit)
        vx, vy, vz = 0.0, v_circ, 0.0

        # Initial angular momentum L = r × (m*v)
        # For position (0, 0, r) and velocity (0, v, 0):
        # L_x = 0*0 - r*v = -r*v (rotating about X axis)
        m = 1.0  # test particle mass
        L_initial = m * r_orbit * v_circ

        print(f"Initial L_x = {-L_initial:.4f}")

        # Leapfrog integration with FROZEN gravity field
        dt = 0.01
        max_steps = 5000

        angles = []
        radii = []
        L_values = []

        prev_angle = 0
        total_angle = 0

        for t in range(max_steps):
            # Get gravity at current position (from frozen field)
            ix, iy, iz = int(px) % N, int(py) % N, int(pz) % N

            # Simple nearest-neighbor gravity
            ax = gx_frozen[iz, iy, ix]
            ay = gy_frozen[iz, iy, ix]
            az = gz_frozen[iz, iy, ix]

            # Leapfrog: half-step velocity
            vx += 0.5 * ax * dt
            vy += 0.5 * ay * dt
            vz += 0.5 * az * dt

            # Full-step position
            px += vx * dt
            py += vy * dt
            pz += vz * dt

            # Periodic boundaries
            px = px % N
            py = py % N
            pz = pz % N

            # Get gravity at new position
            ix, iy, iz = int(px) % N, int(py) % N, int(pz) % N
            ax = gx_frozen[iz, iy, ix]
            ay = gy_frozen[iz, iy, ix]
            az = gz_frozen[iz, iy, ix]

            # Leapfrog: second half-step velocity
            vx += 0.5 * ax * dt
            vy += 0.5 * ay * dt
            vz += 0.5 * az * dt

            # Compute diagnostics
            rx = px - center
            ry = py - center
            rz = pz - center

            # Handle periodic wrapping
            if rx > N/2: rx -= N
            if rx < -N/2: rx += N
            if ry > N/2: ry -= N
            if ry < -N/2: ry += N
            if rz > N/2: rz -= N
            if rz < -N/2: rz += N

            r_current = np.sqrt(rx**2 + ry**2 + rz**2)
            radii.append(r_current)

            # Angular momentum L_x = m*(ry*vz - rz*vy)
            L_x = m * (ry * vz - rz * vy)
            L_values.append(L_x)

            # Angle in YZ plane
            angle = np.arctan2(ry, rz)
            d_angle = angle - prev_angle
            if d_angle > np.pi: d_angle -= 2*np.pi
            if d_angle < -np.pi: d_angle += 2*np.pi
            total_angle += d_angle
            prev_angle = angle
            angles.append(total_angle)

            # Progress report
            if t % 1000 == 0 and t > 0:
                orbits = abs(total_angle) / (2 * np.pi)
                L_ratio = abs(L_x / L_initial) if L_initial != 0 else 0
                print(f"  t={t}: r={r_current:.2f}, orbits={orbits:.2f}, L_ratio={L_ratio:.4f}")

            # Check for escape
            if r_current > N/2 - 2:
                print(f"  ⚠️ Particle escaped at t={t}")
                break

        # Analyze results
        num_orbits = abs(total_angle) / (2 * np.pi)

        if num_orbits > 0.5:
            period = len(angles) * dt / num_orbits
            t2_r3 = (period ** 2) / (r_orbit ** 3)
        else:
            period = 0
            t2_r3 = 0

        mean_r = np.mean(radii)
        ecc = (np.max(radii) - np.min(radii)) / (np.max(radii) + np.min(radii)) if (np.max(radii) + np.min(radii)) > 0 else 0

        # L conservation
        L_final = L_values[-1] if L_values else 0
        L_conservation = abs(L_final / L_initial) if L_initial != 0 else 0

        print(f"\nResults for r={r_orbit}:")
        print(f"  Orbits: {num_orbits:.2f}")
        print(f"  Period: {period:.2f}")
        print(f"  Eccentricity: {ecc:.4f}")
        print(f"  T²/r³: {t2_r3:.4f}")
        print(f"  L conservation: {L_conservation:.4f}")

        results.append({
            'r': r_orbit,
            'orbits': num_orbits,
            'period': period,
            'ecc': ecc,
            't2_r3': t2_r3,
            'L_conservation': L_conservation
        })

    # Final analysis
    print("\n" + "="*70)
    print("KEPLER'S THIRD LAW ANALYSIS")
    print("="*70)

    print("\n" + "-"*70)
    print(f"{'r':>6} {'Orbits':>10} {'Period':>10} {'Ecc':>10} {'T²/r³':>12} {'L_cons':>10}")
    print("-"*70)

    t2_r3_values = []
    for res in results:
        print(f"{res['r']:>6} {res['orbits']:>10.2f} {res['period']:>10.2f} "
              f"{res['ecc']:>10.4f} {res['t2_r3']:>12.4f} {res['L_conservation']:>10.4f}")
        if res['orbits'] >= 1.0:
            t2_r3_values.append(res['t2_r3'])

    if len(t2_r3_values) >= 2:
        mean_ratio = np.mean(t2_r3_values)
        std_ratio = np.std(t2_r3_values)
        cv = std_ratio / mean_ratio if mean_ratio > 0 else float('inf')

        print("-"*70)
        print(f"\nMean T²/r³ = {mean_ratio:.4f}")
        print(f"Std dev = {std_ratio:.4f}")
        print(f"CV (σ/μ) = {cv:.4f}")

        if cv < 0.20:
            print("\n✓ KEPLER'S THIRD LAW SATISFIED!")
        else:
            print(f"\n✗ Kepler's Law not satisfied (CV = {cv*100:.1f}% > 20%)")
    else:
        print("\n⚠️ Not enough complete orbits for Kepler analysis")

    print("\n" + "="*70)


if __name__ == "__main__":
    test_kepler_static_field()
