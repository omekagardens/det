"""
Test Kepler with Trilinear Interpolation for Gravity
=====================================================

The escape issue may be due to discontinuous gravity from nearest-neighbor lookup.
Use smooth trilinear interpolation instead.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def trilinear_interp(field, x, y, z, N):
    """Trilinear interpolation for smooth gravity lookup."""
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


def run_orbit_interpolated(r_orbit, gx, gy, gz, center, N, dt=0.005, max_steps=15000):
    """Run orbit with trilinear interpolated gravity."""

    # Get circular velocity
    g_at_r = np.sqrt(
        trilinear_interp(gx, center, center, center + r_orbit, N)**2 +
        trilinear_interp(gy, center, center, center + r_orbit, N)**2 +
        trilinear_interp(gz, center, center, center + r_orbit, N)**2
    )
    v_circ = np.sqrt(r_orbit * g_at_r) if g_at_r > 0 else 1.0

    # Initialize
    px, py, pz = float(center), float(center), float(center + r_orbit)
    vx, vy, vz = 0.0, v_circ, 0.0

    radii = []
    angles = []
    energies = []
    prev_angle = 0
    total_angle = 0

    for t in range(max_steps):
        # Interpolated gravity
        ax = trilinear_interp(gx, px, py, pz, N)
        ay = trilinear_interp(gy, px, py, pz, N)
        az = trilinear_interp(gz, px, py, pz, N)

        # Leapfrog
        vx += 0.5 * ax * dt
        vy += 0.5 * ay * dt
        vz += 0.5 * az * dt

        px += vx * dt
        py += vy * dt
        pz += vz * dt

        # Wrap
        px = px % N
        py = py % N
        pz = pz % N

        ax = trilinear_interp(gx, px, py, pz, N)
        ay = trilinear_interp(gy, px, py, pz, N)
        az = trilinear_interp(gz, px, py, pz, N)

        vx += 0.5 * ax * dt
        vy += 0.5 * ay * dt
        vz += 0.5 * az * dt

        # Diagnostics
        rx = px - center
        ry = py - center
        rz = pz - center
        if rx > N/2: rx -= N
        if rx < -N/2: rx += N
        if ry > N/2: ry -= N
        if ry < -N/2: ry += N
        if rz > N/2: rz -= N
        if rz < -N/2: rz += N

        r_current = np.sqrt(rx**2 + ry**2 + rz**2)
        radii.append(r_current)

        # Energy (KE only, for tracking)
        KE = 0.5 * (vx**2 + vy**2 + vz**2)
        energies.append(KE)

        # Angle
        angle = np.arctan2(ry, rz)
        d_angle = angle - prev_angle
        if d_angle > np.pi: d_angle -= 2*np.pi
        if d_angle < -np.pi: d_angle += 2*np.pi
        total_angle += d_angle
        prev_angle = angle
        angles.append(total_angle)

        # Escape check
        if r_current > N/2 - 2:
            break

        # Collision check
        if r_current < 2:
            break

        # Enough orbits
        if abs(total_angle) > 10 * np.pi:
            break

    num_orbits = abs(total_angle) / (2 * np.pi)
    period = len(angles) * dt / num_orbits if num_orbits > 0.5 else 0
    t2_r3 = (period ** 2) / (r_orbit ** 3) if period > 0 else 0
    ecc = (np.max(radii) - np.min(radii)) / (np.max(radii) + np.min(radii)) if radii else 1

    return {
        'r': r_orbit,
        'v_circ': v_circ,
        'g_at_r': g_at_r,
        'orbits': num_orbits,
        'period': period,
        'ecc': ecc,
        't2_r3': t2_r3,
        'r_final': radii[-1] if radii else 0,
        'escaped': radii[-1] > N/2 - 2 if radii else True
    }


def main():
    print("="*70)
    print("KEPLER TEST WITH TRILINEAR INTERPOLATED GRAVITY")
    print("="*70)

    N = 64
    center = N // 2

    params = DETParams3D(
        N=N,
        DT=0.01,
        F_VAC=0.0001,
        gravity_enabled=True,
        alpha_grav=0.001,
        kappa_grav=25.0,  # Stronger coupling
        mu_grav=2.0,
        q_enabled=True,
        alpha_q=0.0,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    sim = DETCollider3D(params)

    # Central mass (larger, more concentrated)
    sim.add_packet((center, center, center), mass=100.0, width=1.5,
                   momentum=(0, 0, 0), initial_q=0.98)

    print("\nEstablishing gravity field...")
    for _ in range(400):
        sim.step()

    gx = sim.gx.copy()
    gy = sim.gy.copy()
    gz = sim.gz.copy()

    # Check field
    print("\nGravity profile:")
    print("-"*50)
    for r in [4, 6, 8, 10, 12, 14, 16]:
        if center + r < N - 2:
            g = np.sqrt(
                trilinear_interp(gx, center, center, center + r, N)**2 +
                trilinear_interp(gy, center, center, center + r, N)**2 +
                trilinear_interp(gz, center, center, center + r, N)**2
            )
            gr2 = g * r**2
            print(f"r={r:>3}: |g|={g:.4f}, |g|*r²={gr2:.2f}")

    # Test orbits
    print("\n" + "="*70)
    print("ORBIT TESTS (dt=0.005, trilinear interpolation)")
    print("="*70)

    results = []

    for r_orbit in [6, 8, 10, 12, 14]:
        print(f"\nr = {r_orbit}:")
        result = run_orbit_interpolated(r_orbit, gx, gy, gz, center, N, dt=0.005)
        results.append(result)

        status = "ESCAPED" if result['escaped'] else f"r_final={result['r_final']:.1f}"
        print(f"  v_circ={result['v_circ']:.3f}, orbits={result['orbits']:.2f}, ecc={result['ecc']:.3f}, {status}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    t2_r3_values = []
    print(f"\n{'r':>4} {'orbits':>8} {'ecc':>8} {'T²/r³':>12}")
    print("-"*40)
    for res in results:
        print(f"{res['r']:>4} {res['orbits']:>8.2f} {res['ecc']:>8.3f} {res['t2_r3']:>12.4f}")
        if res['orbits'] >= 1.0 and not res['escaped']:
            t2_r3_values.append(res['t2_r3'])

    if len(t2_r3_values) >= 2:
        mean_ratio = np.mean(t2_r3_values)
        cv = np.std(t2_r3_values) / mean_ratio if mean_ratio > 0 else 0
        print(f"\nT²/r³ mean={mean_ratio:.4f}, CV={cv*100:.1f}%")

        if cv < 0.20:
            print("✓ KEPLER SATISFIED")
        else:
            print("✗ KEPLER NOT SATISFIED")
    else:
        stable_count = sum(1 for r in results if not r['escaped'] and r['orbits'] >= 0.5)
        print(f"\n⚠️ Only {stable_count} stable orbits found")

        # Diagnostic: why escaping?
        print("\nDiagnostic: Checking orbit stability...")
        print("\nThe escape suggests v_circ > actual circular velocity.")
        print("This happens when gravity is WEAKER than expected at larger r.")
        print("\nFor DET with |g| ∝ r^-1.84 instead of r^-2:")
        print("- Gravity weakens faster than 1/r²")
        print("- Particles at larger r have too much speed")
        print("- They spiral outward")

        # Try with velocity correction
        print("\n" + "="*70)
        print("VELOCITY-CORRECTED TEST")
        print("="*70)
        print("Adjusting for non-1/r² gravity: v = 0.85 * sqrt(r * |g|)")

        results2 = []
        for r_orbit in [8, 10, 12]:
            g_at_r = np.sqrt(
                trilinear_interp(gx, center, center, center + r_orbit, N)**2 +
                trilinear_interp(gy, center, center, center + r_orbit, N)**2 +
                trilinear_interp(gz, center, center, center + r_orbit, N)**2
            )
            v_corrected = 0.85 * np.sqrt(r_orbit * g_at_r) if g_at_r > 0 else 1.0

            px, py, pz = float(center), float(center), float(center + r_orbit)
            vx, vy, vz = 0.0, v_corrected, 0.0

            radii = []
            total_angle = 0
            prev_angle = 0
            dt = 0.005

            for t in range(20000):
                ax = trilinear_interp(gx, px, py, pz, N)
                ay = trilinear_interp(gy, px, py, pz, N)
                az = trilinear_interp(gz, px, py, pz, N)

                vx += 0.5 * ax * dt
                vy += 0.5 * ay * dt
                vz += 0.5 * az * dt
                px += vx * dt
                py += vy * dt
                pz += vz * dt
                px, py, pz = px % N, py % N, pz % N

                ax = trilinear_interp(gx, px, py, pz, N)
                ay = trilinear_interp(gy, px, py, pz, N)
                az = trilinear_interp(gz, px, py, pz, N)

                vx += 0.5 * ax * dt
                vy += 0.5 * ay * dt
                vz += 0.5 * az * dt

                rx, ry, rz = px - center, py - center, pz - center
                if rx > N/2: rx -= N
                if rx < -N/2: rx += N
                if ry > N/2: ry -= N
                if ry < -N/2: ry += N
                if rz > N/2: rz -= N
                if rz < -N/2: rz += N

                r = np.sqrt(rx**2 + ry**2 + rz**2)
                radii.append(r)

                angle = np.arctan2(ry, rz)
                d_angle = angle - prev_angle
                if d_angle > np.pi: d_angle -= 2*np.pi
                if d_angle < -np.pi: d_angle += 2*np.pi
                total_angle += d_angle
                prev_angle = angle

                if r > N/2 - 2 or r < 2:
                    break
                if abs(total_angle) > 12 * np.pi:
                    break

            num_orbits = abs(total_angle) / (2 * np.pi)
            period = t * dt / num_orbits if num_orbits > 0 else 0
            t2_r3 = period**2 / r_orbit**3 if period > 0 else 0
            ecc = (max(radii) - min(radii)) / (max(radii) + min(radii)) if radii else 1

            print(f"r={r_orbit}: orbits={num_orbits:.2f}, ecc={ecc:.3f}, T²/r³={t2_r3:.4f}")
            if num_orbits >= 1:
                results2.append({'r': r_orbit, 't2_r3': t2_r3})

        if len(results2) >= 2:
            t2_vals = [x['t2_r3'] for x in results2]
            mean_t2 = np.mean(t2_vals)
            cv_t2 = np.std(t2_vals) / mean_t2 if mean_t2 > 0 else 0
            print(f"\nT²/r³ mean={mean_t2:.4f}, CV={cv_t2*100:.1f}%")


if __name__ == "__main__":
    main()
