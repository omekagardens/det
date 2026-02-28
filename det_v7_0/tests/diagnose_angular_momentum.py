"""
Diagnose Angular Momentum Conservation in DET Orbital Dynamics
===============================================================

Key Question: Why does angular momentum decay during orbits?

DET has two types of angular momentum:
1. Particle angular momentum: L_particle = r × p (Newtonian)
2. Plaquette angular momentum: L_ijkl on lattice faces

The problem: DET's plaquette L may not conserve Newtonian particle L.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def compute_particle_angular_momentum(sim, center):
    """
    Compute total Newtonian angular momentum L = Σ r × (F * v)
    where v is approximated from momentum field π.
    """
    N = sim.p.N

    # Create position grids relative to center
    z, y, x = np.ogrid[:N, :N, :N]
    rx = x - center
    ry = y - center
    rz = z - center

    # Handle periodic boundaries
    rx = np.where(rx > N/2, rx - N, rx)
    rx = np.where(rx < -N/2, rx + N, rx)
    ry = np.where(ry > N/2, ry - N, ry)
    ry = np.where(ry < -N/2, ry + N, ry)
    rz = np.where(rz > N/2, rz - N, rz)
    rz = np.where(rz < -N/2, rz + N, rz)

    # Velocity from momentum (approximate: v ≈ π / σ)
    eps = 1e-10
    vx = sim.pi_X / (sim.sigma + eps)
    vy = sim.pi_Y / (sim.sigma + eps)
    vz = sim.pi_Z / (sim.sigma + eps)

    # L = r × (F * v) = F * (r × v)
    Lx = sim.F * (ry * vz - rz * vy)
    Ly = sim.F * (rz * vx - rx * vz)
    Lz = sim.F * (rx * vy - ry * vx)

    return np.sum(Lx), np.sum(Ly), np.sum(Lz)


def compute_blob_angular_momentum(blobs, center):
    """Compute angular momentum from blob positions and momenta."""
    Lx = Ly = Lz = 0

    for blob in blobs:
        rx = blob['x'] - center
        ry = blob['y'] - center
        rz = blob['z'] - center

        px = blob.get('px', 0)
        py = blob.get('py', 0)
        pz = blob.get('pz', 0)

        m = blob['mass']

        # L = r × p = m * r × v
        Lx += m * (ry * pz - rz * py)
        Ly += m * (rz * px - rx * pz)
        Lz += m * (rx * py - ry * px)

    return Lx, Ly, Lz


def run_orbital_diagnostics():
    """Run detailed diagnostics on angular momentum during orbital motion."""

    print("=" * 70)
    print("ANGULAR MOMENTUM CONSERVATION DIAGNOSTIC")
    print("=" * 70)

    N = 64
    center = N // 2

    # Test both with and without plaquette angular momentum
    for ang_mom_enabled in [False, True]:

        print(f"\n{'='*60}")
        print(f"Test: angular_momentum_enabled = {ang_mom_enabled}")
        print("="*60)

        params = DETParams3D(
            N=N,
            DT=0.02,
            F_VAC=0.001,
            F_MIN=0.0,

            # Gravity
            gravity_enabled=True,
            alpha_grav=0.01,  # Lower for closer to 1/r²
            kappa_grav=15.0,
            mu_grav=2.0,
            beta_g=5.0,

            # Momentum - key for orbital dynamics
            momentum_enabled=True,
            alpha_pi=0.15,
            lambda_pi=0.001,  # Very low decay
            mu_pi=0.4,
            pi_max=10.0,

            # Angular momentum
            angular_momentum_enabled=ang_mom_enabled,
            alpha_L=0.08,
            lambda_L=0.001,
            mu_L=0.2,

            # Structure
            q_enabled=True,
            alpha_q=0.01,

            # Disable interference
            floor_enabled=False,
            boundary_enabled=False,
            agency_dynamic=False,
            sigma_dynamic=False,
            coherence_dynamic=False
        )

        sim = DETCollider3D(params)

        # Central mass
        central_mass = 60.0
        central_q = 0.85
        sim.add_packet((center, center, center), mass=central_mass, width=3.0,
                       momentum=(0, 0, 0), initial_q=central_q)

        # Let field establish
        for _ in range(50):
            sim.step()

        # Measure gravitational field for velocity estimate
        r_orbit = 10
        g_at_r = np.sqrt(
            sim.gx[center, center, center + r_orbit]**2 +
            sim.gy[center, center, center + r_orbit]**2 +
            sim.gz[center, center, center + r_orbit]**2
        )
        v_orbital = np.sqrt(r_orbit * g_at_r) if g_at_r > 0 else 1.0

        print(f"\nOrbital radius: {r_orbit}")
        print(f"|g| at radius: {g_at_r:.4f}")
        print(f"Orbital velocity: {v_orbital:.4f}")

        # Add orbiting test particle
        test_mass = 1.0
        sim.add_packet(
            (center, center, center + r_orbit),
            mass=test_mass,
            width=1.5,
            momentum=(0, v_orbital, 0),  # Tangential velocity in Y
            initial_q=0.05
        )

        # Track angular momentum
        L_initial = compute_particle_angular_momentum(sim, center)
        L_initial_mag = np.sqrt(L_initial[0]**2 + L_initial[1]**2 + L_initial[2]**2)

        print(f"\nInitial L = ({L_initial[0]:.2f}, {L_initial[1]:.2f}, {L_initial[2]:.2f})")
        print(f"Initial |L| = {L_initial_mag:.2f}")

        # Track over time
        print("\n" + "-"*60)
        print(f"{'Step':>6} {'|L|':>12} {'L_ratio':>10} {'r':>8} {'v':>8}")
        print("-"*60)

        L_history = []
        r_history = []

        for t in range(2000):
            sim.step()

            if t % 200 == 0:
                L_current = compute_particle_angular_momentum(sim, center)
                L_mag = np.sqrt(L_current[0]**2 + L_current[1]**2 + L_current[2]**2)
                L_ratio = L_mag / L_initial_mag if L_initial_mag > 0 else 0

                L_history.append(L_mag)

                # Find blobs to get current radius
                blobs = sim.find_blobs(threshold_ratio=3.0)

                r_current = 0
                v_current = 0
                if len(blobs) >= 2:
                    # Find test particle (smaller mass)
                    blobs_sorted = sorted(blobs, key=lambda b: b['mass'], reverse=True)
                    if len(blobs_sorted) >= 2:
                        test = blobs_sorted[1]
                        dx = test['x'] - center
                        dy = test['y'] - center
                        dz = test['z'] - center
                        r_current = np.sqrt(dx**2 + dy**2 + dz**2)
                        r_history.append(r_current)

                        # Estimate velocity from momentum
                        v_current = np.sqrt(test.get('px', 0)**2 +
                                          test.get('py', 0)**2 +
                                          test.get('pz', 0)**2)

                print(f"{t:>6} {L_mag:>12.2f} {L_ratio:>10.4f} {r_current:>8.1f} {v_current:>8.2f}")

        # Summary
        L_final = L_history[-1] if L_history else 0
        L_decay = 1 - (L_final / L_initial_mag) if L_initial_mag > 0 else 0

        print("-"*60)
        print(f"\nFinal |L| = {L_final:.2f}")
        print(f"L decay = {L_decay*100:.1f}%")

        if L_decay > 0.5:
            print("⚠️  SEVERE: Angular momentum NOT conserved!")
        elif L_decay > 0.1:
            print("⚠️  WARNING: Significant angular momentum loss")
        else:
            print("✓ Angular momentum approximately conserved")


def analyze_momentum_dissipation():
    """Analyze where momentum is being dissipated."""

    print("\n" + "="*70)
    print("MOMENTUM DISSIPATION ANALYSIS")
    print("="*70)

    N = 48
    center = N // 2

    params = DETParams3D(
        N=N,
        DT=0.02,
        F_VAC=0.001,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=10.0,
        mu_grav=2.0,
        momentum_enabled=True,
        alpha_pi=0.15,
        lambda_pi=0.001,
        mu_pi=0.4,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    sim = DETCollider3D(params)

    # Central mass
    sim.add_packet((center, center, center), mass=50.0, width=3.0, initial_q=0.8)

    for _ in range(30):
        sim.step()

    # Add orbiting particle
    r = 8
    g_at_r = np.abs(sim.gz[center, center, center + r])
    v = np.sqrt(r * g_at_r) if g_at_r > 0 else 1.0

    sim.add_packet((center, center, center + r), mass=1.0, width=1.5,
                   momentum=(0, v, 0), initial_q=0.05)

    print(f"\nInitial setup: r={r}, v={v:.3f}")

    # Track momentum fields
    print("\nMomentum field evolution:")
    print("-"*60)
    print(f"{'Step':>6} {'Σπ_X':>12} {'Σπ_Y':>12} {'Σπ_Z':>12} {'ΣF':>12}")
    print("-"*60)

    for t in range(500):
        sim.step()

        if t % 50 == 0:
            total_pi_x = np.sum(sim.pi_X)
            total_pi_y = np.sum(sim.pi_Y)
            total_pi_z = np.sum(sim.pi_Z)
            total_F = np.sum(sim.F)

            print(f"{t:>6} {total_pi_x:>12.4f} {total_pi_y:>12.4f} {total_pi_z:>12.4f} {total_F:>12.2f}")

    print("-"*60)
    print("\nNote: π decay indicates momentum is being dissipated.")
    print("This is due to lambda_pi decay term in π update.")


if __name__ == "__main__":
    run_orbital_diagnostics()
    analyze_momentum_dissipation()
