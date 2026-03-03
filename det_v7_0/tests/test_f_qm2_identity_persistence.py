"""F_QM2 - Localized high-q persistence under moderate recovery."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_2d_collider import DETCollider2D, DETParams2D


def test_f_qm2_identity_persistence():
    np.random.seed(42)
    params = DETParams2D(
        N=64,
        DT=0.015,
        gravity_enabled=False,
        floor_enabled=True,
        boundary_enabled=False,
        q_enabled=True,
        alpha_q=0.001,
        jubilee_enabled=True,
        delta_q=0.003,
        n_q=1.0,
        D_0=0.02,
        lambda_P=3.0,
    )
    sim = DETCollider2D(params)

    y, x = np.mgrid[0 : params.N, 0 : params.N]
    c = params.N // 2
    r2 = (x - c) ** 2 + (y - c) ** 2

    sim.F[:] = 1.0 + 0.2 * np.exp(-r2 / 200.0)
    sim.C_E[:] = 0.8
    sim.C_S[:] = 0.8
    sim.a[:] = 0.9

    sim.q[:] = 0.18
    core = r2 <= 16
    far = r2 >= 324
    sim.q[core] = 0.92

    for _ in range(2500):
        sim.step()

    q_core = float(np.mean(sim.q[core]))
    q_far = float(np.mean(sim.q[far]))

    assert q_core > 0.25
    assert (q_core - q_far) > 0.08
