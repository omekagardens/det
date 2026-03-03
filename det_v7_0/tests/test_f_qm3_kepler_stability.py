"""F_QM3 - Kepler/Bound-orbit stability under mutable-q dynamics."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_2d_collider import DETCollider2D, DETParams2D


def test_f_qm3_kepler_stability():
    np.random.seed(42)
    params = DETParams2D(
        N=80,
        DT=0.015,
        gravity_enabled=True,
        alpha_grav=0.02,
        kappa_grav=7.0,
        mu_grav=2.5,
        momentum_enabled=True,
        floor_enabled=False,
        boundary_enabled=False,
        q_enabled=True,
        alpha_q=0.001,
        jubilee_enabled=True,
        delta_q=0.002,
        n_q=1.0,
        D_0=0.03,
        lambda_P=3.0,
    )
    sim = DETCollider2D(params)

    center = params.N // 2
    initial_sep = 20

    sim.add_packet((center, center - initial_sep // 2), mass=9.0, width=3.0, momentum=(0.40, 0.0), initial_q=0.8)
    sim.add_packet((center, center + initial_sep // 2), mass=9.0, width=3.0, momentum=(-0.40, 0.0), initial_q=0.8)

    seps = []
    for _ in range(2200):
        seps.append(sim.separation())
        sim.step()

    seps = np.asarray([s for s in seps if s > 0])
    assert seps.size > 200

    # Stable bound behavior: not immediate escape, no collapse to NaN.
    assert np.isfinite(seps).all()
    assert float(np.max(seps)) < 45.0
    assert float(np.percentile(seps, 95)) < 40.0
    assert float(np.percentile(seps, 5)) > 2.0
