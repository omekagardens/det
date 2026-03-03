#!/usr/bin/env python3
"""Grace/Jubilee interaction checks for mutable-q DET v7.

This test isolates an energy-starved regime and verifies:
1) Energy-coupled Jubilee is blocked when F_op ~ 0 and Grace is off.
2) Grace can lift local F above F_VAC, enabling some Jubilee under energy coupling.
3) Disabling energy coupling permits Jubilee even when F_op remains ~0.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_1d_collider import DETParams1D, DETCollider1D


def _run_starved_scenario(*, grace_enabled: bool, jubilee_energy_coupling: bool, steps: int = 3000) -> dict:
    np.random.seed(7)

    n = 180
    x = np.arange(n)
    params = DETParams1D(
        N=n,
        DT=0.02,
        F_VAC=0.06,
        F_MIN=0.0,
        C_init=0.9,
        momentum_enabled=True,
        alpha_pi=0.08,
        lambda_pi=0.01,
        mu_pi=0.35,
        q_enabled=True,
        alpha_q=0.0005,
        beta_a=0.2,
        floor_enabled=True,
        eta_floor=0.15,
        F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True,
        grace_enabled=grace_enabled,
        F_MIN_grace=0.30,
        healing_enabled=False,
        jubilee_enabled=True,
        delta_q=0.12,
        n_q=1,
        D_0=0.01,
        jubilee_energy_coupling=jubilee_energy_coupling,
        outflow_limit=0.35,
    )
    sim = DETCollider1D(params)

    # Start in a high-q, low-F, high-C regime with persistent local dissipation.
    sim.F[:] = 0.01
    sim.q[:] = 0.75
    sim.a[:] = 0.95
    sim.C_R[:] = 0.92
    sim.pi_R[:] = 1.0 * np.sin(2.0 * np.pi * x / 12.0)

    q_initial = float(np.mean(sim.q))

    for _ in range(steps):
        sim.step()

    f_op = np.maximum(sim.F - params.F_VAC, 0.0)

    return {
        "q_drop": q_initial - float(np.mean(sim.q)),
        "total_jubilee": float(sim.total_jubilee),
        "total_grace": float(sim.total_grace_injected),
        "mean_f": float(np.mean(sim.F)),
        "mean_f_op": float(np.mean(f_op)),
    }


def test_grace_unlocks_energy_coupled_jubilee_in_starved_regime() -> None:
    no_grace_energy_on = _run_starved_scenario(grace_enabled=False, jubilee_energy_coupling=True)
    grace_energy_on = _run_starved_scenario(grace_enabled=True, jubilee_energy_coupling=True)
    no_grace_energy_off = _run_starved_scenario(grace_enabled=False, jubilee_energy_coupling=False)

    # With coupling ON and no Grace, F_op stays zero and Jubilee is blocked.
    assert no_grace_energy_on["mean_f_op"] < 1e-12
    assert no_grace_energy_on["total_jubilee"] < 1e-12

    # Grace raises F and F_op enough to permit at least some Jubilee.
    assert grace_energy_on["total_grace"] > 0.1
    assert grace_energy_on["mean_f"] > no_grace_energy_on["mean_f"]
    assert grace_energy_on["mean_f_op"] > 1e-4
    assert grace_energy_on["total_jubilee"] > 1e-4

    # With coupling OFF, Jubilee proceeds even without operational free resource.
    assert no_grace_energy_off["mean_f_op"] < 1e-12
    assert no_grace_energy_off["total_jubilee"] > 0.05
    assert no_grace_energy_off["q_drop"] > 5e-4
