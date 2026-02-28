"""
DET v6.3 Standard Candle Test: Keplerian Orbits
=================================================

HIGH PRIORITY FALSIFIABILITY TEST

Purpose: Prove DET is a physical theory, not just a simulator.
Test: Does Kepler's Third Law (T² ∝ r³) emerge NATURALLY from the
      geometry of the Φ field, without parameter tuning?

Success Criteria:
- T²/r³ should be approximately constant across different orbital radii
- This must emerge from DET's Poisson-based gravity, not from tuning μ_g or β_g

Reference: DET Theory Card v6.3, Section V (Gravity Module)
"""

import numpy as np
from scipy.fft import fftn, ifftn
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


@dataclass
class OrbitalResult:
    """Results from a single orbital measurement."""
    radius: float           # Semi-major axis (initial orbital radius)
    period: float           # Measured orbital period (in simulation steps)
    num_orbits: float       # Number of complete orbits observed
    eccentricity: float     # Orbital eccentricity (0 = circular)
    t2_over_r3: float       # T²/r³ ratio
    stable: bool            # Did the orbit remain stable?


class KeplerianOrbitTest:
    """
    Test Kepler's Third Law emergence in DET.

    Key insight: In Newtonian gravity with Φ ∝ 1/r potential,
    circular orbital velocity v = √(GM/r), so period T = 2πr/v = 2π√(r³/GM)
    Therefore T² = (4π²/GM) * r³, giving T²/r³ = constant.

    For DET to be physically grounded, this must emerge from the
    Poisson equation L*Φ = -κ*ρ without parameter tuning.
    """

    def __init__(self, grid_size: int = 64, central_mass: float = 50.0,
                 central_q: float = 0.8, verbose: bool = True):
        """
        Initialize test with a central massive body.

        Args:
            grid_size: Simulation grid size
            central_mass: Mass (F) of central body
            central_q: Structure parameter of central body (sources gravity)
            verbose: Print progress
        """
        self.N = grid_size
        self.central_mass = central_mass
        self.central_q = central_q
        self.verbose = verbose
        self.center = grid_size // 2

    def estimate_orbital_velocity(self, radius: float, sim: DETCollider3D) -> float:
        """
        Estimate circular orbital velocity from the gravitational field.

        For circular orbit: v² = r * |g| where g is gravitational acceleration.
        """
        # Sample gravitational field at the orbital radius
        center = self.center

        # Get g at (center, center, center + radius)
        r_int = int(round(radius))
        if center + r_int >= self.N:
            r_int = self.N - center - 1

        g_sample = np.sqrt(
            sim.gx[center, center, center + r_int]**2 +
            sim.gy[center, center, center + r_int]**2 +
            sim.gz[center, center, center + r_int]**2
        )

        # v = sqrt(r * |g|) for circular orbit
        v_orbital = np.sqrt(radius * g_sample) if g_sample > 0 else 0.1

        return v_orbital

    def measure_orbital_period(self, radius: float, max_steps: int = 10000,
                               velocity_factor: float = 1.0) -> OrbitalResult:
        """
        Measure orbital period for a test particle at given radius.

        Args:
            radius: Initial orbital radius from center
            max_steps: Maximum simulation steps
            velocity_factor: Multiplier for estimated orbital velocity

        Returns:
            OrbitalResult with measured period and diagnostics
        """
        if self.verbose:
            print(f"\n  Measuring orbit at r = {radius:.1f}...")

        # Create simulation with standard parameters (NO TUNING)
        params = DETParams3D(
            N=self.N,
            DT=0.015,
            F_VAC=0.001,
            F_MIN=0.0,

            # Standard gravity parameters - NOT TUNED for Kepler
            gravity_enabled=True,
            alpha_grav=0.02,
            kappa_grav=10.0,
            mu_grav=2.0,
            beta_g=5.0,

            # Enable necessary physics
            momentum_enabled=True,
            alpha_pi=0.1,
            lambda_pi=0.001,  # Very low decay for stable orbits
            mu_pi=0.5,

            q_enabled=True,
            alpha_q=0.01,

            # Disable features that might interfere
            angular_momentum_enabled=False,
            floor_enabled=False,
            boundary_enabled=False,
            agency_dynamic=False,
            sigma_dynamic=False,
            coherence_dynamic=False
        )

        sim = DETCollider3D(params)
        center = self.center

        # Place massive central body with structure q
        sim.add_packet(
            (center, center, center),
            mass=self.central_mass,
            width=3.0,
            momentum=(0, 0, 0),
            initial_q=self.central_q
        )

        # Let gravity field establish
        for _ in range(50):
            sim.step()

        # Estimate orbital velocity from gravitational field
        v_est = self.estimate_orbital_velocity(radius, sim)
        v_orbital = v_est * velocity_factor

        if self.verbose:
            print(f"    Estimated v_orbital = {v_est:.4f}, using v = {v_orbital:.4f}")

        # Place test particle with tangential velocity (in Y direction, offset in Z)
        test_mass = 0.5  # Small test mass
        r_int = int(round(radius))

        sim.add_packet(
            (center, center, center + r_int),
            mass=test_mass,
            width=1.5,
            momentum=(0, v_orbital, 0),  # Tangential velocity
            initial_q=0.05
        )

        # Track orbital motion
        angles = []
        distances = []
        times = []

        initial_angle = 0
        prev_angle = 0
        total_angle = 0

        for t in range(max_steps):
            sim.step()

            # Find test particle position (look for secondary blob)
            blobs = sim.find_blobs(threshold_ratio=5.0)

            if len(blobs) >= 2:
                # Central body is usually blob 0 (most massive)
                # Test particle is usually blob 1
                central = blobs[0]
                test = blobs[1] if blobs[1]['mass'] < blobs[0]['mass'] else blobs[0]
                if blobs[1]['mass'] < blobs[0]['mass']:
                    test = blobs[1]
                else:
                    test = blobs[1]  # Take second blob anyway

                # Compute position relative to center
                dx = test['x'] - central['x']
                dy = test['y'] - central['y']
                dz = test['z'] - central['z']

                # Handle periodic boundaries
                if dx > self.N/2: dx -= self.N
                if dx < -self.N/2: dx += self.N
                if dy > self.N/2: dy -= self.N
                if dy < -self.N/2: dy += self.N
                if dz > self.N/2: dz -= self.N
                if dz < -self.N/2: dz += self.N

                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                # Compute angle in XY plane (for orbits in XY plane)
                # Since we started with velocity in Y and offset in Z,
                # track angle in YZ plane
                angle = np.arctan2(dy, dz)

                # Track cumulative angle
                d_angle = angle - prev_angle
                if d_angle > np.pi: d_angle -= 2*np.pi
                if d_angle < -np.pi: d_angle += 2*np.pi
                total_angle += d_angle
                prev_angle = angle

                angles.append(total_angle)
                distances.append(distance)
                times.append(t)

                # Check for completed orbits
                num_orbits = abs(total_angle) / (2 * np.pi)

                if self.verbose and t % 1000 == 0:
                    print(f"    t={t}: r={distance:.1f}, orbits={num_orbits:.2f}")

                # Stop if we have enough data (at least 2 orbits)
                if num_orbits >= 2.5:
                    break
            else:
                # Lost track of blobs - orbit may have destabilized
                if t > 500:
                    break

        # Analyze results
        if len(angles) < 100:
            return OrbitalResult(
                radius=radius,
                period=0,
                num_orbits=0,
                eccentricity=1.0,
                t2_over_r3=0,
                stable=False
            )

        angles = np.array(angles)
        distances = np.array(distances)
        times = np.array(times)

        num_orbits = abs(angles[-1]) / (2 * np.pi)

        if num_orbits < 0.5:
            return OrbitalResult(
                radius=radius,
                period=0,
                num_orbits=num_orbits,
                eccentricity=1.0,
                t2_over_r3=0,
                stable=False
            )

        # Estimate period from total angle traversed
        period = len(times) / num_orbits

        # Compute eccentricity from distance variation
        r_mean = np.mean(distances)
        r_min = np.min(distances)
        r_max = np.max(distances)
        eccentricity = (r_max - r_min) / (r_max + r_min) if (r_max + r_min) > 0 else 0

        # Kepler's ratio
        t2_over_r3 = (period ** 2) / (radius ** 3) if radius > 0 else 0

        # Orbit is stable if eccentricity is low and we completed orbits
        stable = eccentricity < 0.5 and num_orbits >= 1.0

        result = OrbitalResult(
            radius=radius,
            period=period,
            num_orbits=num_orbits,
            eccentricity=eccentricity,
            t2_over_r3=t2_over_r3,
            stable=stable
        )

        if self.verbose:
            print(f"    Period = {period:.1f} steps, orbits = {num_orbits:.2f}")
            print(f"    Eccentricity = {eccentricity:.3f}, T²/r³ = {t2_over_r3:.4f}")
            print(f"    Stable: {stable}")

        return result

    def run_kepler_test(self, radii: List[float] = None,
                        max_steps: int = 15000) -> Dict:
        """
        Run Kepler's Third Law test at multiple orbital radii.

        Args:
            radii: List of orbital radii to test
            max_steps: Maximum steps per orbit

        Returns:
            Dictionary with results and analysis
        """
        if radii is None:
            radii = [8.0, 10.0, 12.0, 15.0, 18.0]

        print("\n" + "="*70)
        print("DET v6.3 STANDARD CANDLE TEST: KEPLER'S THIRD LAW")
        print("="*70)
        print("\nTest: Does T² ∝ r³ emerge naturally from DET gravity?")
        print(f"Central mass: {self.central_mass}, Central q: {self.central_q}")
        print(f"Grid size: {self.N}x{self.N}x{self.N}")
        print("\nNote: Using STANDARD parameters - NO TUNING for Kepler!")

        results = []

        for r in radii:
            # Try different velocity factors to find stable orbit
            for v_factor in [1.0, 0.8, 1.2, 0.6, 1.5]:
                result = self.measure_orbital_period(r, max_steps, v_factor)
                if result.stable and result.num_orbits >= 1.0:
                    results.append(result)
                    break
            else:
                # No stable orbit found, record best attempt
                result = self.measure_orbital_period(r, max_steps, 1.0)
                results.append(result)

        # Analyze Kepler's Law
        print("\n" + "="*70)
        print("KEPLER'S THIRD LAW ANALYSIS")
        print("="*70)

        stable_results = [r for r in results if r.stable]

        print(f"\nStable orbits: {len(stable_results)}/{len(results)}")
        print("\nResults table:")
        print("-" * 60)
        print(f"{'Radius':>8} {'Period':>10} {'Orbits':>8} {'Ecc':>8} {'T²/r³':>12} {'Stable':>8}")
        print("-" * 60)

        for r in results:
            stable_str = "YES" if r.stable else "NO"
            print(f"{r.radius:>8.1f} {r.period:>10.1f} {r.num_orbits:>8.2f} "
                  f"{r.eccentricity:>8.3f} {r.t2_over_r3:>12.4f} {stable_str:>8}")

        # Compute Kepler ratio statistics for stable orbits
        if len(stable_results) >= 2:
            t2_r3_values = [r.t2_over_r3 for r in stable_results]
            mean_ratio = np.mean(t2_r3_values)
            std_ratio = np.std(t2_r3_values)
            cv = std_ratio / mean_ratio if mean_ratio > 0 else float('inf')

            print("\n" + "-" * 60)
            print("T²/r³ Statistics (stable orbits only):")
            print(f"  Mean T²/r³ = {mean_ratio:.4f}")
            print(f"  Std dev    = {std_ratio:.4f}")
            print(f"  CV (σ/μ)   = {cv:.4f}")

            # Kepler's law is satisfied if CV < 0.2 (20% variation)
            kepler_satisfied = cv < 0.20

            print("\n" + "="*70)
            if kepler_satisfied:
                print("RESULT: KEPLER'S THIRD LAW EMERGES NATURALLY!")
                print(f"T²/r³ is constant within {cv*100:.1f}% variation")
                print("DET gravity produces physically correct orbital mechanics.")
            else:
                print("RESULT: KEPLER'S THIRD LAW NOT SATISFIED")
                print(f"T²/r³ varies by {cv*100:.1f}% (threshold: 20%)")
                print("Further investigation needed.")
            print("="*70)

            return {
                'results': results,
                'stable_results': stable_results,
                'mean_t2_r3': mean_ratio,
                'std_t2_r3': std_ratio,
                'cv': cv,
                'kepler_satisfied': kepler_satisfied,
                'num_stable': len(stable_results)
            }
        else:
            print("\n" + "="*70)
            print("RESULT: INSUFFICIENT STABLE ORBITS")
            print("Need at least 2 stable orbits to test Kepler's law.")
            print("="*70)

            return {
                'results': results,
                'stable_results': stable_results,
                'kepler_satisfied': False,
                'num_stable': len(stable_results)
            }


def run_standard_candle_test():
    """Run the full Kepler standard candle test."""

    # Test with default grid
    test = KeplerianOrbitTest(
        grid_size=64,
        central_mass=80.0,
        central_q=0.9,
        verbose=True
    )

    # Test at multiple radii
    radii = [8.0, 10.0, 12.0, 14.0, 16.0]

    results = test.run_kepler_test(radii=radii, max_steps=20000)

    return results


def quick_gravity_field_check():
    """Quick check of the gravitational field profile."""
    print("\n" + "="*70)
    print("GRAVITY FIELD PROFILE CHECK")
    print("="*70)

    params = DETParams3D(
        N=64,
        gravity_enabled=True,
        alpha_grav=0.02,
        kappa_grav=10.0,
        q_enabled=True,
        boundary_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False
    )

    sim = DETCollider3D(params)
    center = params.N // 2

    # Add central mass
    sim.add_packet((center, center, center), mass=50.0, width=3.0, initial_q=0.8)

    # Let field establish
    for _ in range(100):
        sim.step()

    # Sample |g| at different radii
    print("\nGravitational field |g| vs radius:")
    print("-" * 40)
    print(f"{'Radius':>8} {'|g|':>12} {'|g|*r²':>12} {'Expected 1/r²':>15}")
    print("-" * 40)

    g_r2_values = []

    for r in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
        if center + r < params.N:
            g_mag = np.sqrt(
                sim.gx[center, center, center + r]**2 +
                sim.gy[center, center, center + r]**2 +
                sim.gz[center, center, center + r]**2
            )
            g_r2 = g_mag * r**2
            g_r2_values.append(g_r2)
            print(f"{r:>8} {g_mag:>12.6f} {g_r2:>12.4f} {'~const if 1/r²':>15}")

    if len(g_r2_values) > 2:
        mean_gr2 = np.mean(g_r2_values)
        std_gr2 = np.std(g_r2_values)
        cv = std_gr2 / mean_gr2 if mean_gr2 > 0 else float('inf')

        print("-" * 40)
        print(f"Mean |g|*r² = {mean_gr2:.4f}, CV = {cv:.3f}")

        if cv < 0.3:
            print("✓ Gravity follows ~1/r² law (good for Kepler!)")
        else:
            print("✗ Gravity deviates from 1/r² law")


if __name__ == "__main__":
    # First check the gravity field profile
    quick_gravity_field_check()

    # Then run the full Kepler test
    results = run_standard_candle_test()
