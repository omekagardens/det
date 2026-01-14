#!/usr/bin/env python3
"""
Kepler Live Test
================

Run DET simulations to verify Kepler's Third Law emergence.

This test actually runs the DET particle tracker to verify that
T² ∝ r³ emerges from the DET gravity module.

Usage:
    python kepler_live_test.py --radii 6 8 10 12 --orbits 3

Reference: DET Theory Card v6.3, Appendix D
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D
from det_particle_tracker import ParticleTracker, ParticleTrackerParams, Particle


@dataclass
class OrbitResult:
    """Result from a single orbit test."""
    radius: float
    period: float
    eccentricity: float
    T2_over_r3: float
    num_orbits: float
    stable: bool
    energy_error: float
    angular_momentum_error: float


def setup_central_mass(N: int = 64, mass: float = 100.0, q: float = 0.9) -> DETCollider3D:
    """
    Set up a DET simulation with a central gravitating mass.

    Parameters
    ----------
    N : int
        Grid size
    mass : float
        Central mass (F units)
    q : float
        Structural debt (q) of central mass

    Returns
    -------
    DETCollider3D
        Configured collider with central mass
    """
    params = DETParams3D(
        N=N,
        DT=0.02,
        F_VAC=0.001,
        F_MIN=0.0,
        # Enable gravity
        gravity_enabled=True,
        alpha_grav=0.02,
        kappa_grav=5.0,
        mu_grav=2.0,
        # Enable structure
        q_enabled=True,
        alpha_q=0.012,
        # Disable dynamics we don't need
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=True,
        boundary_enabled=False,
        grace_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False,
        coherence_dynamic=False,
    )

    sim = DETCollider3D(params)

    # Add central mass
    center = N // 2
    sim.add_packet((center, center, center), mass=mass, width=2.0, initial_q=q)

    # Let the field settle
    for _ in range(50):
        sim.step()

    return sim


def measure_gravity_field(sim: DETCollider3D) -> Tuple[np.ndarray, np.ndarray]:
    """
    Measure the gravitational field at various radii.

    Returns
    -------
    radii : np.ndarray
        Radii at which field was measured (cells)
    g_magnitudes : np.ndarray
        Gravitational field magnitude at each radius
    """
    center = sim.p.N // 2
    radii = []
    g_mags = []

    # Compute gradient of potential
    Phi = sim.Phi
    gx = -np.gradient(Phi, axis=2)
    gy = -np.gradient(Phi, axis=1)
    gz = -np.gradient(Phi, axis=0)

    # Sample at various radii along x-axis
    for r in range(2, center - 2):
        x = center + r
        g = np.sqrt(gx[center, center, x]**2 +
                   gy[center, center, x]**2 +
                   gz[center, center, x]**2)
        radii.append(r)
        g_mags.append(g)

    return np.array(radii), np.array(g_mags)


def compute_circular_velocity(sim: DETCollider3D, radius: float) -> float:
    """
    Compute the circular orbital velocity at given radius.

    For a circular orbit: v = sqrt(r * |g(r)|)
    """
    center = sim.p.N // 2

    # Get gravitational field at radius
    Phi = sim.Phi
    gx = -np.gradient(Phi, axis=2)

    # Sample at radius along x-axis
    r_int = int(radius)
    x = center + r_int
    g_r = abs(gx[center, center, x])

    return np.sqrt(radius * g_r)


def run_orbit_test(sim: DETCollider3D, radius: float, num_orbits: int = 3,
                   verbose: bool = False) -> OrbitResult:
    """
    Run an orbit test at given radius.

    Parameters
    ----------
    sim : DETCollider3D
        Simulation with established gravity field
    radius : float
        Orbital radius in cells
    num_orbits : int
        Number of complete orbits to track
    verbose : bool
        Print progress

    Returns
    -------
    OrbitResult
        Results of the orbit test
    """
    center = sim.p.N // 2

    # Compute circular velocity
    v_circ = compute_circular_velocity(sim, radius)

    # Set up particle tracker
    params = ParticleTrackerParams(dt=0.01, integrator='leapfrog')
    tracker = ParticleTracker(sim, params)

    # Initial position and velocity (circular orbit in x-y plane)
    x0, y0, z0 = center + radius, float(center), float(center)
    vx0, vy0, vz0 = 0.0, v_circ, 0.0  # Perpendicular to radius

    tracker.add_particle(x0, y0, z0, vx0, vy0, vz0, mass=1.0)

    # Track orbit
    positions_x = [x0]
    positions_y = [y0]
    velocities_y = [vy0]
    times = [0.0]

    # Estimate period
    estimated_period = 2 * np.pi * radius / v_circ if v_circ > 0 else 1000

    # Track for several orbits
    max_steps = int(num_orbits * estimated_period / params.dt * 2)  # 2x safety margin
    max_steps = min(max_steps, 50000)  # Cap at 50k steps

    prev_angle = 0
    total_angle = 0

    for step in range(max_steps):
        tracker.step()

        p = tracker.particles[0]
        positions_x.append(p.x)
        positions_y.append(p.y)
        velocities_y.append(p.vy)
        times.append(tracker.time)

        # Track angular progress
        dx = p.x - center
        dy = p.y - center
        angle = np.arctan2(dy, dx)

        d_angle = angle - prev_angle
        if d_angle > np.pi:
            d_angle -= 2 * np.pi
        elif d_angle < -np.pi:
            d_angle += 2 * np.pi

        total_angle += d_angle
        prev_angle = angle

        # Stop after enough orbits
        if abs(total_angle) >= 2 * np.pi * num_orbits:
            break

        if verbose and step % 1000 == 0:
            orbits_done = abs(total_angle) / (2 * np.pi)
            r_current = np.sqrt(dx**2 + dy**2)
            print(f"  Step {step}: orbits={orbits_done:.2f}, r={r_current:.2f}")

    # Analyze results
    positions_x = np.array(positions_x)
    positions_y = np.array(positions_y)
    times = np.array(times)

    # Compute orbital parameters
    dx = positions_x - center
    dy = positions_y - center
    r = np.sqrt(dx**2 + dy**2)

    r_min = r.min()
    r_max = r.max()
    eccentricity = (r_max - r_min) / (r_max + r_min) if (r_max + r_min) > 0 else 1.0

    # Compute period from angle tracking
    orbits_completed = abs(total_angle) / (2 * np.pi)
    if orbits_completed > 0.1:
        period = times[-1] / orbits_completed
    else:
        period = 0

    # Compute Kepler ratio
    if period > 0 and radius > 0:
        T2_over_r3 = period**2 / radius**3
    else:
        T2_over_r3 = 0

    # Energy conservation (simplified - just kinetic in y)
    KE_initial = 0.5 * velocities_y[0]**2
    KE_final = 0.5 * velocities_y[-1]**2
    energy_error = abs(KE_final - KE_initial) / KE_initial if KE_initial > 0 else 0

    # Angular momentum conservation (simplified)
    L_initial = positions_x[0] * velocities_y[0]
    L_final = positions_x[-1] * velocities_y[-1]
    L_error = abs(L_final - L_initial) / abs(L_initial) if L_initial != 0 else 0

    # Stability check
    stable = eccentricity < 0.1 and orbits_completed >= num_orbits * 0.5

    return OrbitResult(
        radius=radius,
        period=period,
        eccentricity=eccentricity,
        T2_over_r3=T2_over_r3,
        num_orbits=orbits_completed,
        stable=stable,
        energy_error=energy_error,
        angular_momentum_error=L_error
    )


def run_kepler_test(radii: List[float] = None, num_orbits: int = 3,
                    N: int = 64, verbose: bool = True) -> Dict:
    """
    Run full Kepler test across multiple radii.

    Parameters
    ----------
    radii : List[float]
        Orbital radii to test (cells)
    num_orbits : int
        Number of orbits per radius
    N : int
        Grid size
    verbose : bool
        Print progress

    Returns
    -------
    Dict
        Test results including Kepler verification
    """
    if radii is None:
        radii = [6, 8, 10, 12, 14]

    if verbose:
        print("=" * 60)
        print("KEPLER LIVE TEST")
        print("=" * 60)
        print(f"Grid size: {N}")
        print(f"Radii: {radii}")
        print(f"Orbits per radius: {num_orbits}")
        print()

    # Set up central mass
    if verbose:
        print("Setting up central mass...")

    sim = setup_central_mass(N=N, mass=100.0, q=0.9)

    if verbose:
        print("  Done. Measuring gravity field...")

    r_field, g_field = measure_gravity_field(sim)

    if verbose:
        print(f"  g(r=10) = {g_field[8]:.4f}")
        print()

    # Run orbit tests
    results = []
    for radius in radii:
        if verbose:
            print(f"Testing r = {radius}...")

        result = run_orbit_test(sim, radius, num_orbits, verbose=False)
        results.append(result)

        if verbose:
            print(f"  Period: {result.period:.2f}")
            print(f"  Eccentricity: {result.eccentricity:.4f}")
            print(f"  T²/r³: {result.T2_over_r3:.4f}")
            print(f"  Orbits completed: {result.num_orbits:.2f}")
            print(f"  Stable: {result.stable}")
            print()

    # Analyze Kepler's Law
    T2_r3_values = [r.T2_over_r3 for r in results if r.T2_over_r3 > 0]

    if T2_r3_values:
        mean_ratio = np.mean(T2_r3_values)
        std_ratio = np.std(T2_r3_values)
        cv = std_ratio / mean_ratio if mean_ratio > 0 else 1.0
        kepler_satisfied = cv < 0.05  # 5% coefficient of variation
    else:
        mean_ratio = 0
        std_ratio = 0
        cv = 1.0
        kepler_satisfied = False

    if verbose:
        print("=" * 60)
        print("KEPLER ANALYSIS")
        print("=" * 60)
        print(f"T²/r³ values: {[f'{v:.4f}' for v in T2_r3_values]}")
        print(f"Mean: {mean_ratio:.4f}")
        print(f"Std:  {std_ratio:.4f}")
        print(f"CV:   {cv*100:.2f}%")
        print()
        print(f"KEPLER'S THIRD LAW: {'SATISFIED' if kepler_satisfied else 'NOT SATISFIED'}")
        print("=" * 60)

    return {
        'radii': radii,
        'results': results,
        'T2_r3_values': T2_r3_values,
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'cv': cv,
        'kepler_satisfied': kepler_satisfied
    }


def main():
    parser = argparse.ArgumentParser(description='Kepler Live Test')
    parser.add_argument('--radii', nargs='+', type=float, default=[6, 8, 10, 12, 14],
                        help='Orbital radii to test (cells)')
    parser.add_argument('--orbits', type=int, default=3,
                        help='Number of orbits per radius')
    parser.add_argument('--grid', '-N', type=int, default=64,
                        help='Grid size')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    results = run_kepler_test(
        radii=args.radii,
        num_orbits=args.orbits,
        N=args.grid,
        verbose=not args.quiet
    )

    return 0 if results['kepler_satisfied'] else 1


if __name__ == "__main__":
    sys.exit(main())
