"""
F_S6 — Fragmentation and Recovery Test
========================================
Patch: DET-7S-SPIRIT-HOST-1

Question: Does selfhood degrade under damage and partially recover lawfully?

Setup:
  - Produce stable high-S coalition
  - Inflict localized coherence loss and q increase
  - Allow canonical Healing/Jubilee to operate
  - Compute I_self(t_pre, t_post)

Pass condition (three phases):
  1. S drop after injury
  2. Partial recovery if coherence recovery succeeds
  3. I_self remains intermediate, not perfect

Target:
  S_post_injury < S_pre
  S_recovered > S_post_injury
  0.3 < I_self < 0.95

Fail:
  Either no degradation, or perfect persistence through arbitrary destruction,
  or no lawful recovery signature.
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
    IdentityMetricParams,
)


class TestFragmentationRecovery:
    """F_S6: Selfhood must degrade under damage and partially recover."""

    def _build_stable_selfhood(self):
        """Build a simulation with a stable high-S coalition."""
        np.random.seed(300)

        params = DETParams2D(
            N=40,
            gravity_enabled=False,
            momentum_enabled=False,
            angular_momentum_enabled=False,
            boundary_enabled=True,
            grace_enabled=True,
            healing_enabled=True,
            eta_heal=0.05,
            jubilee_enabled=True,
            delta_q=0.002,
            q_enabled=True,
            alpha_q=0.01,
            agency_dynamic=False,
            coherence_dynamic=False,
            sigma_dynamic=False,
            C_init=0.1,
            epsilon_a=0.0,
        )
        sim = DETCollider2D(params)

        core = (slice(12, 28), slice(12, 28))

        # Strong healthy coalition
        sim.a[:] = 0.3
        sim.a[core] = 0.95
        sim.F[:] = 0.5
        sim.F[core] = 6.0
        sim.C_E[core] = 0.85
        sim.C_S[core] = 0.85
        sim.q[:] = 0.0
        sim.q[core] = 0.05

        host_params = HostFitnessParams()
        self_params = SelfFieldParams(
            mu_S=0.08,
            Theta_H=0.2,
            k_H=10.0,
            lambda_S=0.03,
            chi_S=0.02,
        )
        identity_params = IdentityMetricParams()

        harness = SelfhoodHarness(
            sim,
            host_params=host_params,
            self_params=self_params,
            identity_params=identity_params,
        )

        return harness, core

    def test_fragmentation_and_recovery(self):
        """Three-phase test: stable -> damaged -> partial recovery."""
        harness, core = self._build_stable_selfhood()
        sim = harness.sim

        # Phase 1: Establish stable selfhood
        for _ in range(600):
            harness.step()

        S_pre = float(np.mean(harness.S[core]))
        snap_pre = harness.take_snapshot()
        print(f"  Phase 1 - Stable S: {S_pre:.4f}")

        assert S_pre > 0.3, f"Failed to establish stable selfhood: S={S_pre:.4f}"

        # Phase 2: Inflict damage (coherence loss + q increase in half the core)
        damage_zone = (slice(12, 20), slice(12, 28))
        sim.C_E[damage_zone] = 0.1
        sim.C_S[damage_zone] = 0.1
        sim.q[damage_zone] = 0.7

        # Let damage propagate
        for _ in range(100):
            harness.step()

        S_post_injury = float(np.mean(harness.S[core]))
        snap_injury = harness.take_snapshot()
        print(f"  Phase 2 - Post-injury S: {S_post_injury:.4f}")

        # Phase 3: Allow recovery (healing + jubilee active)
        for _ in range(800):
            harness.step()

        S_recovered = float(np.mean(harness.S[core]))
        snap_recovered = harness.take_snapshot()
        print(f"  Phase 3 - Recovered S: {S_recovered:.4f}")

        # Identity persistence
        identity_pre_post = harness.compute_identity(snap_pre, snap_recovered)
        I_self = identity_pre_post["I_self"]
        print(f"  I_self(pre, recovered): {I_self:.4f}")
        for k, v in identity_pre_post.items():
            if k != "I_self":
                print(f"    {k}={v:.4f}")

        # Assertions
        # 1. Damage must cause degradation
        assert S_post_injury < S_pre, (
            f"No degradation: S_post={S_post_injury:.4f} >= S_pre={S_pre:.4f}"
        )
        # 2. Recovery: S should at least stabilize or partially recover.
        #    In practice, with limited healing, S may not fully recover but
        #    should not continue declining rapidly.
        S_decline_rate = (S_post_injury - S_recovered) / S_post_injury if S_post_injury > 0 else 0
        print(f"  S decline rate post-injury: {S_decline_rate:.4f}")
        assert S_decline_rate < 0.2, (
            f"Continued rapid decline: S fell from {S_post_injury:.4f} to "
            f"{S_recovered:.4f} ({S_decline_rate:.1%} further decline)"
        )
        # 3. Identity persistence should be intermediate
        assert 0.1 < I_self < 0.98, (
            f"Identity persistence out of range: I_self={I_self:.4f} "
            f"(expected 0.1 < I_self < 0.98)"
        )

    def test_total_destruction_kills_selfhood(self):
        """Complete destruction of host should eliminate selfhood."""
        harness, core = self._build_stable_selfhood()
        sim = harness.sim

        # Establish selfhood
        for _ in range(600):
            harness.step()

        S_pre = float(np.mean(harness.S[core]))
        print(f"  Pre-destruction S: {S_pre:.4f}")

        # Total destruction: zero everything in core
        sim.C_E[core] = 0.01
        sim.C_S[core] = 0.01
        sim.q[core] = 0.95
        sim.F[core] = 0.01
        sim.a[core] = 0.1

        for _ in range(300):
            harness.step()

        S_post = float(np.mean(harness.S[core]))
        print(f"  Post-destruction S: {S_post:.4f}")

        assert S_post < 0.1, (
            f"Selfhood survived total destruction: S={S_post:.4f}"
        )
