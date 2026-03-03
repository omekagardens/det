"""F_QM4 - No runaway q<->F oscillatory collapse."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_1d_collider import DETCollider1D, DETParams1D


def test_f_qm4_no_oscillatory_collapse():
    np.random.seed(42)
    params = DETParams1D(
        N=160,
        DT=0.02,
        gravity_enabled=False,
        momentum_enabled=True,
        floor_enabled=True,
        boundary_enabled=False,
        q_enabled=True,
        alpha_q=0.004,
        jubilee_enabled=True,
        delta_q=0.010,
        n_q=1.0,
        D_0=0.01,
        lambda_P=3.5,
    )
    sim = DETCollider1D(params)

    x = np.arange(params.N)
    sim.F[:] = 1.0 + 0.4 * np.sin(2 * np.pi * x / params.N)
    sim.q[:] = np.clip(0.35 + 0.25 * np.cos(2 * np.pi * x / params.N), 0.0, 1.0)
    sim.C_R[:] = 0.7
    sim.a[:] = 0.85

    q_mean = []
    F_mean = []
    for _ in range(4000):
        sim.step()
        q_mean.append(float(np.mean(sim.q)))
        F_mean.append(float(np.mean(sim.F)))

    q_mean = np.asarray(q_mean)
    F_mean = np.asarray(F_mean)

    assert np.isfinite(sim.q).all()
    assert np.isfinite(sim.F).all()
    assert np.isfinite(q_mean).all()
    assert np.isfinite(F_mean).all()
    assert np.all((sim.q >= 0.0) & (sim.q <= 1.0))

    tail = q_mean[-800:]
    tail_jump = float(np.max(np.abs(np.diff(tail))))
    assert tail_jump < 0.08
