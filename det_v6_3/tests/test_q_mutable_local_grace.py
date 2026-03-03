"""
Exploratory tests for mutable-q behavior in DET v6.3.

This module evaluates a local-only extension where structural debt q can decrease
through two strictly local channels:
1) local positive resource balance (dF > 0), and
2) local grace injection I_g.

The canonical v6.3 default remains immutable-q under loss (locking only).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def _run_steps(sim: DETCollider3D, steps: int = 120):
    for _ in range(steps):
        sim.step()


def test_q_is_immutable_by_default():
    """Baseline: canonical q-locking only should not decrease total q."""
    p = DETParams3D(
        N=18,
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        agency_dynamic=False,
        boundary_enabled=True,
        grace_enabled=True,
        q_enabled=True,
        alpha_q=0.02,
        q_mutable_local_enabled=False,
    )
    sim = DETCollider3D(p)

    sim.q[8:10, 8:10, 8:10] = 0.6
    sim.F[:] = 0.06
    sim.F[8:10, 8:10, 8:10] = 0.005

    q0 = float(np.sum(sim.q))
    _run_steps(sim, steps=80)
    q1 = float(np.sum(sim.q))

    # With canonical locking-only law, q should not decrease in aggregate.
    assert q1 >= q0 - 1e-9


def test_q_mutable_with_local_resource_and_grace():
    """Exploration mode: q can decrease under local resource inflow and grace."""
    p = DETParams3D(
        N=20,
        DT=0.02,
        F_VAC=0.01,
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        agency_dynamic=False,
        boundary_enabled=True,
        grace_enabled=True,
        F_MIN_grace=0.12,
        R_boundary=1,
        q_enabled=True,
        alpha_q=0.02,
        q_mutable_local_enabled=True,
        # Tuned so this scenario enters a net debt-relief regime (exploration target).
        alpha_q_local_resource_relief=1.00,
        alpha_q_grace_relief=0.25,
    )
    sim = DETCollider3D(p)

    # Debt-rich, resource-poor center; resource-rich shell nearby to drive local inflow.
    c = p.N // 2
    sim.q[c-2:c+2, c-2:c+2, c-2:c+2] = 0.75
    sim.F[:] = 0.02
    sim.F[c-2:c+2, c-2:c+2, c-2:c+2] = 0.005
    sim.F[c-4:c+4, c-4:c+4, c-4:c+4] += 0.12

    q0_total = float(np.sum(sim.q))
    q0_center = float(np.mean(sim.q[c-1:c+1, c-1:c+1, c-1:c+1]))

    _run_steps(sim, steps=120)

    q1_total = float(np.sum(sim.q))
    q1_center = float(np.mean(sim.q[c-1:c+1, c-1:c+1, c-1:c+1]))

    relief_local = float(np.sum(sim.last_q_relief_local))
    relief_grace = float(np.sum(sim.last_q_relief_grace))

    assert relief_local > 0.0
    assert relief_grace > 0.0
    assert q1_total < q0_total
    assert q1_center < q0_center


def test_mutable_q_changes_outcome_vs_immutable_control():
    """Sanity check: mutable-q run diverges from immutable control."""

    base = dict(
        N=20,
        DT=0.02,
        gravity_enabled=True,
        boundary_enabled=True,
        grace_enabled=True,
        F_MIN_grace=0.10,
        q_enabled=True,
        alpha_q=0.015,
    )

    p_immutable = DETParams3D(
        **base,
        q_mutable_local_enabled=False,
    )

    p_mutable = DETParams3D(
        **base,
        q_mutable_local_enabled=True,
        alpha_q_local_resource_relief=0.03,
        alpha_q_grace_relief=0.20,
    )

    s0 = DETCollider3D(p_immutable)
    s1 = DETCollider3D(p_mutable)

    for sim in (s0, s1):
        sim.add_packet((10, 10, 6), mass=2.2, width=2.0, momentum=(0, 0, 0.15), initial_q=0.45)
        sim.add_packet((10, 10, 14), mass=2.2, width=2.0, momentum=(0, 0, -0.15), initial_q=0.45)
        sim.F[9:12, 9:12, 9:12] = 0.01

    _run_steps(s0, 150)
    _run_steps(s1, 150)

    # Mutable-q should end with less total structural debt under same initial conditions.
    assert np.sum(s1.q) < np.sum(s0.q)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
