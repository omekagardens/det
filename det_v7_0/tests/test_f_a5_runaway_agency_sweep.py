"""F_A5 - Runaway agency stability sweep (no structural ceiling)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_2d_collider import DETCollider2D, DETParams2D


def run_sweep_case(beta_a: float):
    params = DETParams2D(
        N=40,
        DT=0.015,
        gravity_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False,
        q_enabled=True,
        alpha_qD=0.0,
        lambda_IP=2.0,
        lambda_DP=3.0,
        beta_a=beta_a,
        a0=1.0,
        epsilon_a=0.0,
    )
    sim = DETCollider2D(params)

    # Strong coherence and presence gradients.
    y, x = np.mgrid[0 : params.N, 0 : params.N]
    center = params.N // 2
    r2 = (x - center) ** 2 + (y - center) ** 2
    sim.F[:] = 0.2 + 4.0 * np.exp(-r2 / 40.0)
    sim.q_I[:] = 0.2 * (r2 > 80)
    sim.q_D[:] = 0.2 * (r2 <= 80)
    sim.q = np.clip(sim.q_I + sim.q_D, 0, 1)
    sim.C_E[:] = 0.9
    sim.C_S[:] = 0.9
    sim.a[:] = 0.6

    mean_trace = []
    for _ in range(1500):
        sim.step()
        mean_trace.append(float(np.mean(sim.a)))

    mean_trace = np.array(mean_trace)
    tail = mean_trace[-300:]
    tail_jump = float(np.max(np.abs(np.diff(tail))))
    bounded = bool(np.all((sim.a >= 0.0) & (sim.a <= 1.0)))
    finite = bool(np.isfinite(sim.a).all() and np.isfinite(sim.P).all())
    stable_tail = tail_jump < 0.2
    return bounded and finite and stable_tail


def test_f_a5_runaway_agency_sweep():
    beta_sweep = [0.1, 0.2, 0.4, 0.8, 1.2, 2.0]
    outcomes = [run_sweep_case(beta) for beta in beta_sweep]
    assert all(outcomes), f"Failed sweep cases: {beta_sweep}"

