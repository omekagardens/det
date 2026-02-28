"""
Test DET gravity profile behavior.

This module provides:
- a reusable helper (`run_gravity_profile`) for exploratory scripts, and
- a pytest smoke test ensuring the profile routine runs deterministically.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def run_gravity_profile(alpha_grav: float, kappa_grav: float):
    """Return sampled |g| and |g|*r^2 for a central packet."""
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
        boundary_enabled=False,
    )

    sim = DETCollider3D(params)
    sim.add_packet((center, center, center), mass=50.0, width=2.0, momentum=(0, 0, 0), initial_q=0.9)

    for _ in range(200):
        sim.step()

    radii = [3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
    g_values = []
    g_r2_values = []

    for r in radii:
        if center + r < N - 2:
            g_mag = np.sqrt(
                sim.gx[center, center, center + r] ** 2
                + sim.gy[center, center, center + r] ** 2
                + sim.gz[center, center, center + r] ** 2
            )
            g_values.append(g_mag)
            g_r2_values.append(g_mag * r**2)

    mean_gr2 = np.mean(g_r2_values)
    std_gr2 = np.std(g_r2_values)
    cv = std_gr2 / mean_gr2 if mean_gr2 > 0 else float("inf")

    return radii[: len(g_values)], np.array(g_values), np.array(g_r2_values), cv


def test_gravity_profile_smoke():
    """Profile routine should return finite outputs."""
    radii, g_vals, gr2_vals, cv = run_gravity_profile(alpha_grav=0.01, kappa_grav=20.0)

    assert len(radii) > 3
    assert g_vals.shape[0] == len(radii)
    assert gr2_vals.shape[0] == len(radii)
    assert np.all(np.isfinite(g_vals))
    assert np.all(np.isfinite(gr2_vals))
    assert np.isfinite(cv)


def main():
    """CLI helper for manual inspection."""
    alpha_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    kappa = 20.0

    print("=" * 70)
    print("DET GRAVITY PROFILE ANALYSIS")
    print("=" * 70)

    best_alpha = None
    best_cv = float("inf")

    for alpha in alpha_values:
        _, _, _, cv = run_gravity_profile(alpha_grav=alpha, kappa_grav=kappa)
        print(f"alpha_grav={alpha:.4f} -> CV(|g|*r^2)={cv:.4f}")
        if cv < best_cv:
            best_cv = cv
            best_alpha = alpha

    print(f"Best alpha_grav={best_alpha} with CV={best_cv:.4f}")


if __name__ == "__main__":
    main()
