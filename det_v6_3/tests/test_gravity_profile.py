"""
Test DET Gravity Profile vs 1/r²
================================

Question: Can we tune DET parameters to get 1/r² gravity?

Key parameters:
- alpha_grav: Helmholtz screening (lower = less screening = closer to 1/r²)
- kappa_grav: Poisson coupling strength
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def test_gravity_profile(alpha_grav, kappa_grav, title=""):
    """Test gravity profile with given parameters."""

    N = 64
    center = N // 2

    params = DETParams3D(
        N=N,
        DT=0.01,
        F_VAC=0.001,
        gravity_enabled=True,
        alpha_grav=alpha_grav,
        kappa_grav=kappa_grav,
        mu_grav=2.0,
        q_enabled=True,
        alpha_q=0.0,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    sim = DETCollider3D(params)

    # Add point-like mass
    sim.add_packet((center, center, center), mass=50.0, width=2.0,
                   momentum=(0, 0, 0), initial_q=0.9)

    # Establish gravity
    for _ in range(200):
        sim.step()

    # Profile
    radii = [3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
    g_values = []
    g_r2_values = []

    for r in radii:
        if center + r < N - 2:
            g_mag = np.sqrt(
                sim.gx[center, center, center + r]**2 +
                sim.gy[center, center, center + r]**2 +
                sim.gz[center, center, center + r]**2
            )
            g_values.append(g_mag)
            g_r2_values.append(g_mag * r**2)

    # Compute CV for |g|*r² (should be ~0 for 1/r²)
    mean_gr2 = np.mean(g_r2_values)
    std_gr2 = np.std(g_r2_values)
    cv = std_gr2 / mean_gr2 if mean_gr2 > 0 else float('inf')

    return radii[:len(g_values)], g_values, g_r2_values, cv


def main():
    print("="*70)
    print("DET GRAVITY PROFILE ANALYSIS")
    print("="*70)
    print("\nGoal: Find parameters that give 1/r² gravity (|g|*r² = const)")

    # Test different alpha_grav values
    alpha_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    kappa = 20.0

    print("\n" + "-"*70)
    print(f"Testing α_grav values (κ = {kappa})")
    print("-"*70)

    results = []
    for alpha in alpha_values:
        radii, g_vals, gr2_vals, cv = test_gravity_profile(alpha, kappa)
        results.append((alpha, cv, radii, g_vals, gr2_vals))
        print(f"α_grav = {alpha:.3f}: CV(|g|*r²) = {cv:.4f}")

    # Find best alpha
    best = min(results, key=lambda x: x[1])
    print(f"\nBest α_grav = {best[0]:.3f} with CV = {best[1]:.4f}")

    # Show detailed profile for best
    print("\n" + "-"*70)
    print(f"Detailed profile for α_grav = {best[0]}")
    print("-"*70)
    print(f"{'r':>6} {'|g|':>12} {'|g|*r²':>12}")
    print("-"*40)
    for i, r in enumerate(best[2]):
        print(f"{r:>6} {best[3][i]:>12.4f} {best[4][i]:>12.4f}")

    # Test with very low alpha
    print("\n" + "="*70)
    print("TESTING WITH MINIMAL SCREENING (α_grav = 0.0001)")
    print("="*70)

    radii, g_vals, gr2_vals, cv = test_gravity_profile(0.0001, kappa)
    print(f"\nCV(|g|*r²) = {cv:.4f}")

    print("\n" + "-"*40)
    print(f"{'r':>6} {'|g|':>12} {'|g|*r²':>12}")
    print("-"*40)
    for i, r in enumerate(radii):
        print(f"{r:>6} {g_vals[i]:>12.4f} {gr2_vals[i]:>12.4f}")

    # Analyze the power law
    print("\n" + "="*70)
    print("POWER LAW ANALYSIS")
    print("="*70)

    # Fit |g| = A * r^n to find n
    radii, g_vals, _, cv = test_gravity_profile(0.001, kappa)

    log_r = np.log(radii)
    log_g = np.log(g_vals)

    # Linear regression: log(g) = log(A) + n*log(r)
    n, log_A = np.polyfit(log_r, log_g, 1)
    A = np.exp(log_A)

    print(f"\nFitted power law: |g| = {A:.4f} * r^{n:.4f}")
    print(f"Expected for 1/r²: |g| ∝ r^-2")
    print(f"\nExponent n = {n:.4f} (should be -2.0 for 1/r²)")

    if abs(n + 2) < 0.3:
        print("✓ Close to 1/r² law")
    else:
        print(f"✗ Deviates from 1/r² by {abs(n + 2):.2f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
The DET gravity field with Helmholtz screening does NOT produce pure 1/r².

Root cause: The baseline field b in DET (screened Poisson) modifies the
effective source. The Helmholtz equation:

    (L - α)b = -α*q

creates an exponentially-screened baseline, so:

    ρ = q - b

is a modified source that doesn't equal a point mass.

IMPLICATIONS FOR KEPLER:
- DET gravity law is NOT Newtonian 1/r²
- Kepler's Third Law (T² ∝ r³) relies on 1/r² gravity
- DET will NOT naturally reproduce Kepler's law without modification

POTENTIAL FIXES:
1. Bypass Helmholtz baseline: Set b = 0 (no screening)
2. Use direct Poisson: L*Φ = -κ*q (no baseline subtraction)
3. Accept non-Keplerian gravity as a DET prediction
""")


if __name__ == "__main__":
    main()
