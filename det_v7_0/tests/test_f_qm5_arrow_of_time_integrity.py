"""F_QM5 - Arrow-of-time integrity under mutable-q recovery."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_1d_collider import DETCollider1D, DETParams1D


def _shannon_entropy(field: np.ndarray) -> float:
    total = float(np.sum(field))
    if total <= 0:
        return 0.0
    p = np.clip(field / total, 1e-15, None)
    return float(-np.sum(p * np.log(p)))


def test_f_qm5_arrow_of_time_integrity():
    np.random.seed(42)
    params = DETParams1D(
        N=180,
        DT=0.02,
        gravity_enabled=False,
        momentum_enabled=True,
        floor_enabled=False,
        boundary_enabled=False,
        q_enabled=True,
        alpha_q=0.002,
        jubilee_enabled=True,
        delta_q=0.003,
        n_q=1.0,
        D_0=0.02,
        lambda_P=3.0,
    )
    sim = DETCollider1D(params)

    # Non-uniform initial condition to test entropy direction.
    x = np.arange(params.N)
    sim.F[:] = 1.0 + 0.8 * np.exp(-0.5 * ((x - 40) / 9.0) ** 2) + 0.6 * np.exp(-0.5 * ((x - 120) / 7.0) ** 2)
    sim.q[:] = np.clip(0.2 + 0.25 * np.sin(2 * np.pi * x / params.N), 0.0, 1.0)
    sim.C_R[:] = 0.65
    sim.a[:] = 0.9

    ent = []
    for _ in range(2500):
        sim.step()
        ent.append(_shannon_entropy(sim.F))

    ent = np.asarray(ent)
    slope = float(np.polyfit(np.arange(ent.size), ent, 1)[0])

    # Statistical arrow: no sustained entropy reversal under recovery dynamics.
    assert np.isfinite(ent).all()
    assert ent[-1] >= 0.98 * ent[0]
    assert slope >= -1e-5
