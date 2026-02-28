"""F_GTD5' - Drag-inclusive clock ratio consistency."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_1d_collider import DETCollider1D, DETParams1D


def test_gtd5_prime_drag_clock_ratio():
    lambda_dp = 4.0
    lambda_ip = 2.5
    params = DETParams1D(
        N=120,
        DT=0.02,
        gravity_enabled=False,
        momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False,
        q_enabled=False,
        lambda_DP=lambda_dp,
        lambda_IP=lambda_ip,
        gamma_v=1.0,
    )
    sim = DETCollider1D(params)

    # Two regions with same base state but different debt composition.
    sim.F[:] = 1.0
    sim.sigma[:] = 1.0
    sim.a[:] = 1.0
    sim.C_R[:] = 0.6

    region_a = slice(20, 50)   # low drag
    region_b = slice(70, 100)  # high drag

    sim.q_I[region_a] = 0.10
    sim.q_D[region_a] = 0.10
    sim.q_I[region_b] = 0.70
    sim.q_D[region_b] = 0.20
    sim.q = np.clip(sim.q_I + sim.q_D, 0, 1)

    # Single step computes P and Δτ.
    sim.step()

    ratio_measured = float(np.mean(sim.P[region_a]) / np.mean(sim.P[region_b]))

    D_a = 1.0 / (1.0 + lambda_dp * 0.10 + lambda_ip * 0.10)
    D_b = 1.0 / (1.0 + lambda_dp * 0.20 + lambda_ip * 0.70)
    ratio_expected = D_a / D_b

    rel_err = abs(ratio_measured - ratio_expected) / ratio_expected
    assert rel_err < 0.02, f"clock ratio mismatch: measured={ratio_measured}, expected={ratio_expected}"

