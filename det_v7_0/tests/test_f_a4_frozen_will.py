"""F_A4 - Frozen Will persistence under extreme drag."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_1d_collider import DETCollider1D, DETParams1D


def test_f_a4_frozen_will_long_run():
    params = DETParams1D(
        N=32,
        DT=0.02,
        momentum_enabled=False,
        floor_enabled=False,
        gravity_enabled=False,
        boundary_enabled=False,
        q_enabled=False,
        coherence_weighted_H=False,
        lambda_P=25.0,
        beta_a=0.1,
        a0=1.0,
        epsilon_a=0.0,
    )
    sim = DETCollider1D(params)

    # Extreme debt -> extreme drag, but agency remains primitive.
    sim.q[:] = 1.0
    sim.a[:] = 1.0
    baseline = sim.a.copy()

    max_dev = 0.0
    for step in range(100_000):
        sim.step()
        if step % 5000 == 0:
            max_dev = max(max_dev, float(np.max(np.abs(sim.a - baseline))))
            if max_dev > 0.01:
                break

    assert max_dev <= 0.01
