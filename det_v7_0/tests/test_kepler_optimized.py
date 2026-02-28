"""
Test Kepler's Law with Optimized DET Parameters
================================================

Using α_grav = 0.001 which gives |g| ∝ r^-1.84 (close to 1/r²)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def run_orbit_test(r_orbit, sim, gx_frozen, gy_frozen, gz_frozen, center, N, dt=0.01):
    """Run a single orbit test and return results."""

    # Get circular velocity from gravity at orbit radius
    g_at_r = np.sqrt(
        gx_frozen[center, center, center + r_orbit]**2 +
        gy_frozen[center, center, center + r_orbit]**2 +
        gz_frozen[center, center, center + r_orbit]**2
    )
    v_circ = np.sqrt(r_orbit * g_at_r) if g_at_r > 0 else 1.0

    # Initialize particle
    px, py, pz = float(center), float(center), float(center + r_orbit)
    vx, vy, vz = 0.0, v_circ, 0.0

    m = 1.0
    L_initial = m * r_orbit * v_circ

    # Track orbit
    max_steps = 8000
    angles = []
    radii = []

    prev_angle = 0
    total_angle = 0

    for t in range(max_steps):
        # Get gravity (nearest neighbor from frozen field)
        ix, iy, iz = int(px) % N, int(py) % N, int(pz) % N

        ax = gx_frozen[iz, iy, ix]
        ay = gy_frozen[iz, iy, ix]
        az = gz_frozen[iz, iy, ix]

        # Leapfrog integration
        vx += 0.5 * ax * dt
        vy += 0.5 * ay * dt
        vz += 0.5 * az * dt

        px += vx * dt
        py += vy * dt
        pz += vz * dt

        px = px % N
        py = py % N
        pz = pz % N

        ix, iy, iz = int(px) % N, int(py) % N, int(pz) % N
        ax = gx_frozen[iz, iy, ix]
        ay = gy_frozen[iz, iy, ix]
        az = gz_frozen[iz, iy, ix]

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

        # Track angle
        angle = np.arctan2(ry, rz)
        d_angle = angle - prev_angle
        if d_angle > np.pi: d_angle -= 2*np.pi
        if d_angle < -np.pi: d_angle += 2*np.pi
        total_angle += d_angle
        prev_angle = angle
        angles.append(total_angle)

        # Check escape
        if r_current > N/2 - 2:
            break

        # Check for enough orbits
        if abs(total_angle) > 6 * np.pi:  # 3 orbits
            break

    # Results
    num_orbits = abs(total_angle) / (2 * np.pi)

    if num_orbits > 0.5:
        period = len(angles) * dt / num_orbits
        t2_r3 = (period ** 2) / (r_orbit ** 3)
    else:
        period = 0
        t2_r3 = 0

    ecc = (np.max(radii) - np.min(radii)) / (np.max(radii) + np.min(radii)) if radii else 1.0

    return {
        'r': r_orbit,
        'v_circ': v_circ,
        'g_at_r': g_at_r,
        'orbits': num_orbits,
        'period': period,
        'ecc': ecc,
        't2_r3': t2_r3,
        'escaped': r_current > N/2 - 2 if radii else True
    }


def main():
    print("="*70)
    print("KEPLER TEST WITH OPTIMIZED DET PARAMETERS")
    print("="*70)
    print("\nUsing α_grav = 0.001 (gives |g| ∝ r^-1.84)")

    N = 64
    center = N // 2

    params = DETParams3D(
        N=N,
        DT=0.01,
        F_VAC=0.0001,  # Very low vacuum
        gravity_enabled=True,
        alpha_grav=0.001,  # Optimized for ~1/r²
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

    # Central mass
    sim.add_packet((center, center, center), mass=80.0, width=2.0,
                   momentum=(0, 0, 0), initial_q=0.95)

    # Establish and freeze gravity
    print("\nEstablishing gravity field...")
    for _ in range(300):
        sim.step()

    gx_frozen = sim.gx.copy()
    gy_frozen = sim.gy.copy()
    gz_frozen = sim.gz.copy()

    # Check gravity profile
    print("\nGravity profile (optimized α_grav = 0.001):")
    print("-"*50)
    print(f"{'r':>6} {'|g|':>12} {'v_circ':>12}")
    print("-"*50)

    for r in [6, 8, 10, 12, 14]:
        if center + r < N - 2:
            g = np.sqrt(
                gx_frozen[center, center, center + r]**2 +
                gy_frozen[center, center, center + r]**2 +
                gz_frozen[center, center, center + r]**2
            )
            v = np.sqrt(r * g) if g > 0 else 0
            print(f"{r:>6} {g:>12.4f} {v:>12.4f}")

    # Test orbits
    print("\n" + "="*70)
    print("ORBIT TESTS")
    print("="*70)

    results = []
    dt = 0.008  # Smaller timestep for better accuracy

    for r_orbit in [6, 8, 10, 12, 14]:
        print(f"\nTesting r = {r_orbit}...")
        result = run_orbit_test(r_orbit, sim, gx_frozen, gy_frozen, gz_frozen, center, N, dt)
        results.append(result)

        status = "ESCAPED" if result['escaped'] else "OK"
        print(f"  |g| = {result['g_at_r']:.4f}, v_circ = {result['v_circ']:.4f}")
        print(f"  Orbits: {result['orbits']:.2f}, Ecc: {result['ecc']:.4f}, Status: {status}")
        if result['orbits'] >= 1:
            print(f"  Period: {result['period']:.2f}, T²/r³: {result['t2_r3']:.4f}")

    # Kepler analysis
    print("\n" + "="*70)
    print("KEPLER'S THIRD LAW ANALYSIS")
    print("="*70)

    print("\n" + "-"*70)
    print(f"{'r':>6} {'|g|':>10} {'v_circ':>10} {'Orbits':>8} {'Period':>10} {'T²/r³':>12} {'Status':>8}")
    print("-"*70)

    t2_r3_values = []
    for res in results:
        status = "ESCAPED" if res['escaped'] else "OK"
        print(f"{res['r']:>6} {res['g_at_r']:>10.4f} {res['v_circ']:>10.4f} "
              f"{res['orbits']:>8.2f} {res['period']:>10.2f} {res['t2_r3']:>12.4f} {status:>8}")

        if res['orbits'] >= 1.0 and not res['escaped']:
            t2_r3_values.append(res['t2_r3'])

    if len(t2_r3_values) >= 2:
        mean_ratio = np.mean(t2_r3_values)
        std_ratio = np.std(t2_r3_values)
        cv = std_ratio / mean_ratio if mean_ratio > 0 else float('inf')

        print("-"*70)
        print(f"\nMean T²/r³ = {mean_ratio:.4f}")
        print(f"Std dev = {std_ratio:.4f}")
        print(f"CV (σ/μ) = {cv:.4f} ({cv*100:.1f}%)")

        if cv < 0.20:
            print("\n" + "="*70)
            print("✓ KEPLER'S THIRD LAW SATISFIED!")
            print("  T²/r³ is constant within 20% tolerance")
            print("="*70)
        else:
            print(f"\n✗ T²/r³ varies by {cv*100:.1f}% (threshold: 20%)")
    else:
        print(f"\n⚠️ Only {len(t2_r3_values)} complete orbits - need at least 2")

        if all(r['escaped'] for r in results):
            print("\nAll particles escaped! Possible issues:")
            print("1. Circular velocity estimate too high (causing outward spiral)")
            print("2. Gravity too weak at large r")
            print("3. Timestep too large")

            # Try with lower velocity
            print("\n" + "="*70)
            print("RETRY WITH REDUCED VELOCITY (0.7x)")
            print("="*70)

            results2 = []
            for r_orbit in [8, 10, 12]:
                # Manually reduce velocity
                g_at_r = np.sqrt(
                    gx_frozen[center, center, center + r_orbit]**2 +
                    gy_frozen[center, center, center + r_orbit]**2 +
                    gz_frozen[center, center, center + r_orbit]**2
                )
                v_circ = 0.7 * np.sqrt(r_orbit * g_at_r) if g_at_r > 0 else 1.0

                # Run custom orbit
                px, py, pz = float(center), float(center), float(center + r_orbit)
                vx, vy, vz = 0.0, v_circ, 0.0

                radii = []
                angles = []
                prev_angle = 0
                total_angle = 0

                for t in range(10000):
                    ix, iy, iz = int(px) % N, int(py) % N, int(pz) % N
                    ax = gx_frozen[iz, iy, ix]
                    ay = gy_frozen[iz, iy, ix]
                    az = gz_frozen[iz, iy, ix]

                    vx += 0.5 * ax * dt
                    vy += 0.5 * ay * dt
                    vz += 0.5 * az * dt

                    px += vx * dt
                    py += vy * dt
                    pz += vz * dt

                    px = px % N
                    py = py % N
                    pz = pz % N

                    ix, iy, iz = int(px) % N, int(py) % N, int(pz) % N
                    ax = gx_frozen[iz, iy, ix]
                    ay = gy_frozen[iz, iy, ix]
                    az = gz_frozen[iz, iy, ix]

                    vx += 0.5 * ax * dt
                    vy += 0.5 * ay * dt
                    vz += 0.5 * az * dt

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

                    angle = np.arctan2(ry, rz)
                    d_angle = angle - prev_angle
                    if d_angle > np.pi: d_angle -= 2*np.pi
                    if d_angle < -np.pi: d_angle += 2*np.pi
                    total_angle += d_angle
                    prev_angle = angle
                    angles.append(total_angle)

                    if r_current > N/2 - 2 or r_current < 2:
                        break
                    if abs(total_angle) > 8 * np.pi:
                        break

                num_orbits = abs(total_angle) / (2 * np.pi)
                period = len(angles) * dt / num_orbits if num_orbits > 0 else 0
                t2_r3 = (period ** 2) / (r_orbit ** 3) if num_orbits > 0 else 0
                ecc = (np.max(radii) - np.min(radii)) / (np.max(radii) + np.min(radii)) if radii else 1

                print(f"r={r_orbit}: orbits={num_orbits:.2f}, ecc={ecc:.3f}, T²/r³={t2_r3:.4f}")
                if num_orbits >= 1:
                    results2.append({'r': r_orbit, 't2_r3': t2_r3})

            if len(results2) >= 2:
                t2_r3_vals = [r['t2_r3'] for r in results2]
                mean_r = np.mean(t2_r3_vals)
                std_r = np.std(t2_r3_vals)
                cv_r = std_r / mean_r if mean_r > 0 else 0
                print(f"\nWith reduced velocity: CV = {cv_r*100:.1f}%")


if __name__ == "__main__":
    main()
