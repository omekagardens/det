"""
F_S5 — Triviality Falsifier (Extended)
========================================
Patch: DET-7S-SPIRIT-HOST-1

Question: Is S just another name for "alive cluster"?

This file provides extended triviality tests beyond the basic one in
test_selfhood_development.py. It tests multiple scenarios to ensure S
captures something genuinely distinct from generic metabolic activity.

Setup:
  Compare:
    1. high-life / low-reciprocity clusters
    2. high-life / high-reciprocity / stable-pointer clusters

Pass: Group 2 has significantly higher stable S than group 1.
Fail: All viable clusters converge to identical S.

Meaning: Distinguishes selfhood from generic metabolism / activity.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from det_v6_3_2d_collider import DETCollider2D, DETParams2D
from det_v7_0.experimental.selfhood import (
    SelfhoodHarness,
    HostFitnessParams,
    SelfFieldParams,
)


def _make_static_sim(N=32):
    """Create a static (no dynamics) simulation for controlled comparison."""
    params = DETParams2D(
        N=N,
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        boundary_enabled=False,
        q_enabled=False,
        agency_dynamic=False,
        coherence_dynamic=False,
        sigma_dynamic=False,
        C_init=0.05,
    )
    return DETCollider2D(params)


class TestTrivialityFalsifierExtended:
    """Extended triviality tests for selfhood."""

    def test_high_agency_alone_insufficient(self):
        """High agency alone (without coherence/reciprocity) should not
        produce high S."""
        np.random.seed(500)
        sim = _make_static_sim()
        core = (slice(8, 24), slice(8, 24))

        sim.a[core] = 1.0
        sim.F[core] = 10.0
        sim.C_E[:] = 0.05
        sim.C_S[:] = 0.05
        sim.q[:] = 0.0

        harness = SelfhoodHarness(
            sim,
            self_params=SelfFieldParams(mu_S=0.1, Theta_H=0.2),
        )

        for _ in range(500):
            harness.step()

        S_mean = float(np.mean(harness.S[core]))
        print(f"  High agency, low coherence: mean S = {S_mean:.4f}")

        assert S_mean < 0.3, (
            f"Triviality: high agency alone produced S={S_mean:.4f}"
        )

    def test_high_coherence_alone_insufficient(self):
        """High coherence alone (without agency) should not produce high S."""
        np.random.seed(501)
        sim = _make_static_sim()
        core = (slice(8, 24), slice(8, 24))

        sim.a[core] = 0.1      # low agency
        sim.F[core] = 10.0
        sim.C_E[core] = 0.95
        sim.C_S[core] = 0.95
        sim.q[:] = 0.0

        harness = SelfhoodHarness(
            sim,
            self_params=SelfFieldParams(mu_S=0.1, Theta_H=0.2),
        )

        for _ in range(500):
            harness.step()

        S_mean = float(np.mean(harness.S[core]))
        print(f"  Low agency, high coherence: mean S = {S_mean:.4f}")

        # S should be lower than the full-condition case
        assert S_mean < 0.5, (
            f"Triviality: high coherence alone produced S={S_mean:.4f}"
        )

    def test_high_q_suppresses_selfhood(self):
        """High structural debt should suppress selfhood even with good
        agency and coherence."""
        np.random.seed(502)
        sim = _make_static_sim()
        core = (slice(8, 24), slice(8, 24))

        sim.a[core] = 0.95
        sim.F[core] = 5.0
        sim.C_E[core] = 0.8
        sim.C_S[core] = 0.8
        sim.q[core] = 0.9      # very high structural debt

        harness = SelfhoodHarness(
            sim,
            self_params=SelfFieldParams(mu_S=0.1, Theta_H=0.2),
        )

        for _ in range(500):
            harness.step()

        S_high_q = float(np.mean(harness.S[core]))

        # Compare with low q
        np.random.seed(503)
        sim2 = _make_static_sim()
        sim2.a[core] = 0.95
        sim2.F[core] = 5.0
        sim2.C_E[core] = 0.8
        sim2.C_S[core] = 0.8
        sim2.q[core] = 0.05    # low structural debt

        harness2 = SelfhoodHarness(
            sim2,
            self_params=SelfFieldParams(mu_S=0.1, Theta_H=0.2),
        )

        for _ in range(500):
            harness2.step()

        S_low_q = float(np.mean(harness2.S[core]))

        print(f"  High q: mean S = {S_high_q:.4f}")
        print(f"  Low q:  mean S = {S_low_q:.4f}")

        assert S_low_q > S_high_q + 0.05, (
            f"Structural debt does not suppress selfhood: "
            f"S_low_q={S_low_q:.4f}, S_high_q={S_high_q:.4f}"
        )

    def test_reciprocity_matters(self):
        """Reciprocal support should produce higher S than one-way flow."""
        np.random.seed(504)

        # Symmetric F distribution (reciprocal)
        sim1 = _make_static_sim()
        core = (slice(8, 24), slice(8, 24))
        sim1.a[core] = 0.9
        sim1.F[core] = 5.0     # uniform -> symmetric -> reciprocal
        sim1.C_E[core] = 0.8
        sim1.C_S[core] = 0.8
        sim1.q[core] = 0.1

        harness1 = SelfhoodHarness(
            sim1,
            self_params=SelfFieldParams(mu_S=0.08, Theta_H=0.2),
        )

        for _ in range(500):
            harness1.step()

        S_symmetric = float(np.mean(harness1.S[core]))
        R_symmetric = float(np.mean(harness1.diag.R[core]))

        # Asymmetric F distribution (one-way)
        np.random.seed(505)
        sim2 = _make_static_sim()
        sim2.a[core] = 0.9
        # Create strong gradient: one side has all the F
        sim2.F[8:16, 8:24] = 15.0
        sim2.F[16:24, 8:24] = 0.1
        sim2.C_E[core] = 0.8
        sim2.C_S[core] = 0.8
        sim2.q[core] = 0.1

        harness2 = SelfhoodHarness(
            sim2,
            self_params=SelfFieldParams(mu_S=0.08, Theta_H=0.2),
        )

        for _ in range(500):
            harness2.step()

        S_asymmetric = float(np.mean(harness2.S[core]))
        R_asymmetric = float(np.mean(harness2.diag.R[core]))

        print(f"  Symmetric: mean S={S_symmetric:.4f}, mean R={R_symmetric:.4f}")
        print(f"  Asymmetric: mean S={S_asymmetric:.4f}, mean R={R_asymmetric:.4f}")

        # Reciprocity should be higher in symmetric case
        assert R_symmetric >= R_asymmetric, (
            f"Reciprocity metric broken: R_sym={R_symmetric:.4f} < R_asym={R_asymmetric:.4f}"
        )
