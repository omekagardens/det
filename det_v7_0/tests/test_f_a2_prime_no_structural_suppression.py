"""F_A2' - No structural agency suppression under debt drag."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def test_f_a2_prime_no_structural_suppression():
    params = DETParams3D(
        N=20,
        DT=0.02,
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False,
        q_enabled=False,
        coherence_dynamic=False,
        sigma_dynamic=False,
        lambda_IP=6.0,
        lambda_DP=6.0,
        beta_a=0.15,
        a0=1.0,
        epsilon_a=0.0,
    )
    sim = DETCollider3D(params)

    # Start from high agency everywhere.
    sim.a[:] = 1.0
    sim.F[:] = 1.0
    sim.sigma[:] = 1.0

    c = params.N // 2
    core = np.s_[c - 2 : c + 2, c - 2 : c + 2, c - 2 : c + 2]
    shell = np.s_[c - 6 : c - 3, c - 6 : c - 3, c - 6 : c - 3]

    # Inject high immutable debt in the core.
    sim.q_I[core] = 0.95
    sim.q_D[core] = 0.0
    sim.q = np.clip(sim.q_I + sim.q_D, 0, 1)

    for _ in range(600):
        sim.step()

    a_core = float(np.mean(sim.a[core]))
    a_shell = float(np.mean(sim.a[shell]))
    P_core = float(np.mean(sim.P[core]))
    P_shell = float(np.mean(sim.P[shell]))

    # Agency must stay high despite debt; presence must show drag.
    assert a_core >= 0.95
    assert a_shell >= 0.95
    assert P_core < 0.6 * P_shell

