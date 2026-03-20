"""
F_S4 — Developmental Emergence Test
=====================================
Patch: DET-7S-SPIRIT-HOST-1

Question: Can selfhood appear gradually after agency is already present?

Setup:
  - Initialize high a, weak pointer stability, weak reciprocity
  - Slowly increase coherence and reciprocity over time

Expected signature:
  - a exists early
  - H_host rises later
  - S crosses threshold only after maturation

Pass condition:
  There exists a delayed onset window: t_{S up} - t_{a up} > T_min

Fail:
  S trivially rises immediately whenever a is nonzero.

Also includes F_S5 — Triviality Falsifier.
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


class TestDevelopmentalEmergence:
    """F_S4: Selfhood must emerge gradually, not instantly with agency."""

    def test_delayed_onset_with_maturation(self):
        """S must show delayed onset relative to agency when developmental
        maturity is enabled."""
        np.random.seed(100)

        params = DETParams2D(
            N=32,
            gravity_enabled=False,
            momentum_enabled=False,
            angular_momentum_enabled=False,
            boundary_enabled=False,
            q_enabled=False,
            agency_dynamic=False,
            coherence_dynamic=False,
            sigma_dynamic=False,
            C_init=0.1,
        )
        sim = DETCollider2D(params)

        # Start with high agency but poor everything else
        core = (slice(10, 22), slice(10, 22))
        sim.a[core] = 0.95
        sim.F[core] = 0.5
        sim.C_E[:] = 0.1
        sim.C_S[:] = 0.1
        sim.q[:] = 0.5

        host_params = HostFitnessParams(
            developmental_enabled=True,
            mu_D=0.003,
            lambda_D=0.001,
            w_D=1.5,
        )
        self_params = SelfFieldParams(
            mu_S=0.05,
            Theta_H=0.10,   # lower threshold to allow S to rise
            k_H=10.0,
        )

        harness = SelfhoodHarness(sim, host_params=host_params, self_params=self_params)

        # Track when agency is high vs when S rises
        a_high_step = 0  # agency is high from step 0
        S_threshold = 0.1
        S_onset_step = None

        total_steps = 3000
        history_S = []
        history_H = []
        history_D = []

        for t in range(total_steps):
            # Gradually improve conditions (simulating development)
            if t > 200:
                ramp = min(1.0, (t - 200) / 1000.0)
                sim.C_E[core] = 0.1 + 0.7 * ramp
                sim.C_S[core] = 0.1 + 0.7 * ramp
                sim.q[core] = 0.5 - 0.4 * ramp
                sim.F[core] = 0.5 + 4.0 * ramp

            harness.step()

            mean_S = float(np.mean(harness.S[core]))
            mean_H = float(np.mean(harness.diag.H_host[core]))
            mean_D = float(np.mean(harness.D_mature[core]))

            history_S.append(mean_S)
            history_H.append(mean_H)
            history_D.append(mean_D)

            if S_onset_step is None and mean_S > S_threshold:
                S_onset_step = t

        print(f"  Agency high from step: {a_high_step}")
        print(f"  S onset step (>{S_threshold}): {S_onset_step}")
        print(f"  Final mean S: {history_S[-1]:.4f}")
        print(f"  Final mean H_host: {history_H[-1]:.4f}")
        print(f"  Final mean D_mature: {history_D[-1]:.4f}")

        # Pass condition: delayed onset
        T_min = 50  # minimum delay in steps
        assert S_onset_step is not None, "S never rose above threshold"
        assert S_onset_step > T_min, (
            f"S rose too quickly: onset at step {S_onset_step} < T_min={T_min}. "
            f"Selfhood should not appear instantly with agency."
        )

    def test_no_selfhood_without_development(self):
        """Without developmental maturity, S should still require host fitness,
        but may rise faster than with maturation enabled."""
        np.random.seed(101)

        params = DETParams2D(
            N=32,
            gravity_enabled=False,
            momentum_enabled=False,
            angular_momentum_enabled=False,
            boundary_enabled=False,
            q_enabled=False,
            agency_dynamic=False,
            coherence_dynamic=False,
            sigma_dynamic=False,
            C_init=0.1,
        )
        sim = DETCollider2D(params)

        core = (slice(10, 22), slice(10, 22))
        sim.a[core] = 0.95
        sim.F[core] = 5.0
        sim.C_E[core] = 0.8
        sim.C_S[core] = 0.8
        sim.q[core] = 0.1

        # No developmental gating
        host_params = HostFitnessParams(developmental_enabled=False)
        self_params = SelfFieldParams(mu_S=0.05, Theta_H=0.25)

        harness = SelfhoodHarness(sim, host_params=host_params, self_params=self_params)

        S_onset_no_dev = None
        for t in range(500):
            harness.step()
            if S_onset_no_dev is None and float(np.mean(harness.S[core])) > 0.1:
                S_onset_no_dev = t

        # With developmental gating
        np.random.seed(101)
        sim2 = DETCollider2D(params)
        sim2.a[core] = 0.95
        sim2.F[core] = 5.0
        sim2.C_E[core] = 0.8
        sim2.C_S[core] = 0.8
        sim2.q[core] = 0.1

        host_params2 = HostFitnessParams(
            developmental_enabled=True, mu_D=0.003, lambda_D=0.001, w_D=1.5,
        )
        harness2 = SelfhoodHarness(sim2, host_params=host_params2, self_params=self_params)

        S_onset_with_dev = None
        for t in range(500):
            harness2.step()
            if S_onset_with_dev is None and float(np.mean(harness2.S[core])) > 0.1:
                S_onset_with_dev = t

        print(f"  S onset without development: step {S_onset_no_dev}")
        print(f"  S onset with development: step {S_onset_with_dev}")

        # Development should delay onset
        if S_onset_no_dev is not None and S_onset_with_dev is not None:
            assert S_onset_with_dev > S_onset_no_dev, (
                f"Developmental gating did not delay S onset: "
                f"no_dev={S_onset_no_dev}, with_dev={S_onset_with_dev}"
            )


class TestTrivialityFalsifier:
    """F_S5: S must not be just another name for 'alive cluster'.

    Compare:
      1. high-life / low-reciprocity clusters
      2. high-life / high-reciprocity / stable-pointer clusters

    Pass: Group 2 has significantly higher stable S than group 1.
    Fail: All viable clusters converge to identical S.
    """

    def test_selfhood_not_trivially_alive(self):
        """S must distinguish between generic activity and mature selfhood."""
        np.random.seed(200)

        # --- Group 1: High life, low reciprocity ---
        params1 = DETParams2D(
            N=32,
            gravity_enabled=False,
            momentum_enabled=False,
            angular_momentum_enabled=False,
            boundary_enabled=False,
            q_enabled=False,
            agency_dynamic=False,
            coherence_dynamic=False,
            sigma_dynamic=False,
            C_init=0.1,
        )
        sim1 = DETCollider2D(params1)

        core = (slice(8, 24), slice(8, 24))

        # High agency and F, but low coherence (poor reciprocity)
        sim1.a[core] = 0.95
        sim1.F[core] = 5.0
        sim1.C_E[core] = 0.15   # low coherence -> low reciprocity
        sim1.C_S[core] = 0.15
        sim1.q[core] = 0.1

        harness1 = SelfhoodHarness(
            sim1,
            host_params=HostFitnessParams(),
            self_params=SelfFieldParams(mu_S=0.05, Theta_H=0.10),
        )

        for _ in range(500):
            harness1.step()

        S_group1 = float(np.mean(harness1.S[core]))

        # --- Group 2: High life, high reciprocity, stable pointers ---
        np.random.seed(201)
        sim2 = DETCollider2D(params1)

        sim2.a[core] = 0.95
        sim2.F[core] = 5.0
        sim2.C_E[core] = 0.85   # high coherence -> high reciprocity
        sim2.C_S[core] = 0.85
        sim2.q[core] = 0.05     # low debt -> stable pointers

        harness2 = SelfhoodHarness(
            sim2,
            host_params=HostFitnessParams(),
            self_params=SelfFieldParams(mu_S=0.05, Theta_H=0.10),
        )

        for _ in range(500):
            harness2.step()

        S_group2 = float(np.mean(harness2.S[core]))

        delta_min = 0.1
        delta = S_group2 - S_group1

        print(f"  Group 1 (low reciprocity) mean S: {S_group1:.4f}")
        print(f"  Group 2 (high reciprocity) mean S: {S_group2:.4f}")
        print(f"  Delta: {delta:.4f} (need > {delta_min})")

        assert delta > delta_min, (
            f"Triviality: S does not distinguish selfhood from generic activity. "
            f"S_group2 - S_group1 = {delta:.4f} < {delta_min}"
        )
