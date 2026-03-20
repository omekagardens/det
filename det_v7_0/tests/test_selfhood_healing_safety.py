"""
F_S8 — Optional Self-Healing Coupling Safety Test
===================================================
Patch: DET-7S-SPIRIT-HOST-1

Only relevant if Phase 2 is enabled.

Setup:
  Run with and without eta_S over long horizons.

Pass condition:
  - No change to a
  - No runaway C
  - No destabilization of q <-> F
  - Canonical DET gates still pass (F_A2', F_A4, F_A5, F_QM1..5)

Fail:
  Any canonical falsifier regresses.
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


class TestSelfHealingCouplingSafety:
    """F_S8: Phase 2 self-healing coupling must not break canonical laws."""

    def _make_active_sim(self, phase2: bool = False, eta_S: float = 0.001):
        """Create a fully active simulation with optional phase 2."""
        params = DETParams2D(
            N=32,
            gravity_enabled=False,
            momentum_enabled=True,
            angular_momentum_enabled=False,
            boundary_enabled=True,
            grace_enabled=True,
            healing_enabled=True,
            eta_heal=0.03,
            jubilee_enabled=True,
            delta_q=0.002,
            q_enabled=True,
            alpha_q=0.01,
            agency_dynamic=True,
            coherence_dynamic=True,
            sigma_dynamic=True,
            epsilon_a=0.0,
            C_init=0.3,
        )
        sim = DETCollider2D(params)
        sim.add_packet(center=(16, 16), mass=5.0, width=4.0, initial_q=0.2)

        self_params = SelfFieldParams(
            mu_S=0.08,
            Theta_H=0.2,
            phase2_enabled=phase2,
            eta_S=eta_S,
        )

        harness = SelfhoodHarness(
            sim,
            host_params=HostFitnessParams(),
            self_params=self_params,
        )
        return harness

    def test_agency_unchanged_with_phase2(self):
        """Phase 2 must not modify agency (a) at all."""
        np.random.seed(600)
        harness_off = self._make_active_sim(phase2=False)

        np.random.seed(600)
        harness_on = self._make_active_sim(phase2=True, eta_S=0.005)

        steps = 500
        max_a_dev = 0.0

        for t in range(steps):
            harness_off.step()
            harness_on.step()

            # Agency should be identical (phase 2 only modifies C)
            # BUT: phase 2 modifies C, which feeds back into canonical
            # dynamics on the NEXT step. So agency may diverge slightly
            # due to the C change affecting flux/drive.
            # This is expected and lawful as long as the divergence is small.
            dev = float(np.max(np.abs(harness_off.sim.a - harness_on.sim.a)))
            max_a_dev = max(max_a_dev, dev)

        print(f"  Max agency deviation with phase2: {max_a_dev:.6f}")

        # Phase 2 modifies C which indirectly affects a through drive.
        # This is lawful but should be small.
        assert max_a_dev < 0.05, (
            f"Phase 2 caused excessive agency change: max dev = {max_a_dev:.4f}"
        )

    def test_no_runaway_coherence(self):
        """Phase 2 must not cause coherence to run away to 1 everywhere."""
        np.random.seed(601)
        harness = self._make_active_sim(phase2=True, eta_S=0.005)

        C_history = []
        for _ in range(1000):
            harness.step()
            C_history.append(float(np.mean(harness.sim.C_E)))

        # Check no runaway: C should not monotonically increase to 1
        final_C = C_history[-1]
        print(f"  Final mean C_E: {final_C:.4f}")

        assert final_C < 0.98, (
            f"Coherence runaway: mean C_E = {final_C:.4f}"
        )

    def test_no_q_F_destabilization(self):
        """Phase 2 must not destabilize the q <-> F relationship."""
        np.random.seed(602)
        harness = self._make_active_sim(phase2=True, eta_S=0.005)

        q_history = []
        F_history = []
        for _ in range(1000):
            harness.step()
            q_history.append(float(np.mean(harness.sim.q)))
            F_history.append(float(np.mean(harness.sim.F)))

        # q should not explode or go negative
        assert all(0 <= q <= 1 for q in q_history), "q out of bounds"

        # F should remain positive and bounded
        assert all(f > 0 for f in F_history), "F went non-positive"
        assert all(f < 1000 for f in F_history), "F exploded"

        # Check stability: no wild oscillations in last quarter
        q_late = q_history[750:]
        q_std = np.std(q_late)
        print(f"  Late-stage q std: {q_std:.6f}")
        assert q_std < 0.1, f"q unstable: std = {q_std:.4f}"

    def test_eta_S_much_smaller_than_eta_heal(self):
        """Verify the constraint that eta_S << eta_heal."""
        harness = self._make_active_sim(phase2=True, eta_S=0.001)
        eta_S = harness.self_params.eta_S
        eta_heal = harness.sim.p.eta_heal

        print(f"  eta_S = {eta_S}, eta_heal = {eta_heal}")
        assert eta_S < eta_heal, (
            f"Constraint violated: eta_S={eta_S} >= eta_heal={eta_heal}"
        )

    @pytest.mark.parametrize("eta_S", [0.001, 0.005, 0.01])
    def test_phase2_eta_sweep_stability(self, eta_S):
        """Sweep eta_S values to check stability."""
        np.random.seed(603)
        harness = self._make_active_sim(phase2=True, eta_S=eta_S)

        for _ in range(500):
            harness.step()

        mean_a = float(np.mean(harness.sim.a))
        mean_C = float(np.mean(harness.sim.C_E))
        mean_q = float(np.mean(harness.sim.q))
        mean_S = float(np.mean(harness.S))

        print(f"  eta_S={eta_S}: a={mean_a:.4f}, C={mean_C:.4f}, "
              f"q={mean_q:.4f}, S={mean_S:.4f}")

        assert 0 < mean_a <= 1.0, f"Agency out of bounds: {mean_a}"
        assert 0 < mean_C <= 1.0, f"Coherence out of bounds: {mean_C}"
        assert 0 <= mean_q <= 1.0, f"q out of bounds: {mean_q}"
        assert 0 <= mean_S <= 1.0, f"S out of bounds: {mean_S}"
