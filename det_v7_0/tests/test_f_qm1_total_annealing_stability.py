"""F_QM1 - Total Annealing Stability for unified mutable-q branch."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_1d_collider import DETCollider1D, DETParams1D


def test_f_qm1_total_annealing_stability():
    np.random.seed(42)
    params = DETParams1D(
        N=128,
        DT=0.02,
        gravity_enabled=False,
        floor_enabled=True,
        boundary_enabled=False,
        q_enabled=True,
        alpha_q=0.001,
        jubilee_enabled=True,
        delta_q=0.004,
        n_q=1.0,
        D_0=0.02,
        lambda_P=3.0,
    )
    sim = DETCollider1D(params)

    x = np.arange(params.N)
    sim.F[:] = 1.2 + 0.3 * np.sin(2 * np.pi * x / params.N)
    sim.C_R[:] = 0.75
    sim.a[:] = 0.9

    sim.q[:] = 0.05
    for c, amp, w in [(24, 0.65, 6.0), (64, 0.55, 7.0), (98, 0.60, 5.5)]:
        sim.q += amp * np.exp(-0.5 * ((x - c) / w) ** 2)
    sim.q = np.clip(sim.q, 0.0, 1.0)

    q_std_initial = float(np.std(sim.q))
    q_mean_initial = float(np.mean(sim.q))

    for _ in range(3000):
        sim.step()

    q_std_final = float(np.std(sim.q))
    q_mean_final = float(np.mean(sim.q))

    # Must not globally anneal to near-uniform vacuum under mild Jubilee.
    assert q_std_final > 0.2 * q_std_initial
    assert q_mean_final > 0.02
    assert q_mean_final < q_mean_initial
