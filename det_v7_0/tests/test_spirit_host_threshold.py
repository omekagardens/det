"""
F_S1 — Ghost Insertion Falsifier
=================================
Patch: DET-7S-SPIRIT-HOST-1

Question: Can S rise without lawful host formation?

Setup:
  - 2D lattice
  - Random low-coherence clusters
  - Keep C_bar, R, M_ptr below threshold everywhere
  - Sweep mu_S

Pass condition:
  max_i S_i < 0.1 for all runs where H^host_i < Theta_H - delta

Fail:
  Any stable high-S region appears below host threshold.

Meaning:
  Rejects hidden spirit insertion.

Also includes F_S2 — Agency Override Falsifier (co-located for threshold tests).
"""

import sys
import os
import numpy as np
import pytest

# Ensure collider is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from det_v6_3_2d_collider import DETCollider2D, DETParams2D
from det_v7_0.experimental.selfhood import (
    SelfhoodHarness,
    HostFitnessParams,
    SelfFieldParams,
)


class TestGhostInsertionFalsifier:
    """F_S1: S must not rise without lawful host formation."""

    def _make_hostile_sim(self, mu_S: float = 0.05):
        """Create a simulation where host conditions are deliberately poor.

        Low coherence, no reciprocity, high structural debt -> H_host should
        stay well below threshold, and S should never rise.
        """
        np.random.seed(42)

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
            C_init=0.05,       # very low coherence
            lambda_P=3.0,
        )
        sim = DETCollider2D(params)

        # Set hostile conditions: low coherence, high q, low agency
        sim.C_E[:] = 0.05
        sim.C_S[:] = 0.05
        sim.q[:] = 0.8        # high structural debt
        sim.a[:] = 0.3        # low agency
        sim.F[:] = 0.02       # near-vacuum resource

        self_params = SelfFieldParams(
            mu_S=mu_S,
            Theta_H=0.3,
            k_H=10.0,
            lambda_S=0.03,
            chi_S=0.02,
        )
        host_params = HostFitnessParams()

        harness = SelfhoodHarness(sim, host_params=host_params, self_params=self_params)
        return harness

    def test_ghost_insertion_default_mu(self):
        """S must stay below 0.1 when host conditions are hostile (default mu_S)."""
        harness = self._make_hostile_sim(mu_S=0.05)

        for _ in range(500):
            harness.step()

        max_S = float(np.max(harness.S))
        max_H = float(np.max(harness.diag.H_host))

        print(f"  max(S) = {max_S:.6f}, max(H_host) = {max_H:.6f}")
        assert max_S < 0.1, (
            f"Ghost insertion detected: max(S)={max_S:.4f} without lawful host "
            f"(max H_host={max_H:.4f})"
        )

    @pytest.mark.parametrize("mu_S", [0.01, 0.05, 0.1, 0.5, 1.0])
    def test_ghost_insertion_mu_sweep(self, mu_S):
        """S must stay below 0.1 across mu_S sweep when host is hostile."""
        harness = self._make_hostile_sim(mu_S=mu_S)

        for _ in range(500):
            harness.step()

        max_S = float(np.max(harness.S))
        max_H = float(np.max(harness.diag.H_host))
        delta = 0.05

        print(f"  mu_S={mu_S}: max(S)={max_S:.6f}, max(H_host)={max_H:.6f}")

        # Only enforce if host fitness is genuinely below threshold
        if max_H < (harness.self_params.Theta_H - delta):
            assert max_S < 0.1, (
                f"Ghost insertion at mu_S={mu_S}: max(S)={max_S:.4f} "
                f"with H_host={max_H:.4f} < Theta_H-delta"
            )

    def test_host_fitness_stays_low_in_hostile_conditions(self):
        """Verify that H_host is genuinely low in our hostile setup."""
        harness = self._make_hostile_sim()

        for _ in range(200):
            harness.step()

        max_H = float(np.max(harness.diag.H_host))
        print(f"  max(H_host) = {max_H:.6f} (should be << Theta_H=0.3)")
        assert max_H < 0.2, f"Host fitness unexpectedly high: {max_H:.4f}"


class TestAgencyOverrideFalsifier:
    """F_S2: The speculative layer must not modify agency.

    Setup: duplicate deterministic run.
      Run A: canonical DET only
      Run B: canonical DET + speculative readout layer

    Pass: a_i^(A)(t) == a_i^(B)(t) for all i, t to tolerance < 1e-12
    """

    def _make_deterministic_sim(self):
        """Create a deterministic simulation with dynamics enabled."""
        params = DETParams2D(
            N=32,
            gravity_enabled=False,
            momentum_enabled=True,
            angular_momentum_enabled=False,
            boundary_enabled=True,
            grace_enabled=True,
            healing_enabled=True,
            eta_heal=0.03,
            q_enabled=True,
            jubilee_enabled=True,
            agency_dynamic=True,
            coherence_dynamic=True,
            sigma_dynamic=True,
            epsilon_a=0.0,  # no noise -> deterministic
        )
        return params

    def test_agency_override_phase1(self):
        """Phase 1 readout must not alter agency at all."""
        np.random.seed(123)
        params_A = self._make_deterministic_sim()
        sim_A = DETCollider2D(params_A)
        # Add a packet to create interesting dynamics
        sim_A.add_packet(center=(16, 16), mass=5.0, width=4.0, initial_q=0.3)

        np.random.seed(123)
        params_B = self._make_deterministic_sim()
        sim_B = DETCollider2D(params_B)
        sim_B.add_packet(center=(16, 16), mass=5.0, width=4.0, initial_q=0.3)

        # Harness for run B (phase 1 = readout only)
        harness_B = SelfhoodHarness(
            sim_B,
            self_params=SelfFieldParams(mu_S=0.1, phase2_enabled=False),
        )

        steps = 300
        max_deviation = 0.0

        for t in range(steps):
            sim_A.step()
            harness_B.step()

            dev = float(np.max(np.abs(sim_A.a - sim_B.a)))
            max_deviation = max(max_deviation, dev)

            if dev > 1e-12:
                pytest.fail(
                    f"Agency divergence at step {t}: max|a_A - a_B| = {dev:.2e}"
                )

        print(f"  Max agency deviation over {steps} steps: {max_deviation:.2e}")
        assert max_deviation < 1e-12, (
            f"Agency override detected: max deviation = {max_deviation:.2e}"
        )

    def test_all_fields_match_phase1(self):
        """All canonical fields (F, q, C_E, C_S, P) must match in phase 1."""
        np.random.seed(456)
        params_A = self._make_deterministic_sim()
        sim_A = DETCollider2D(params_A)
        sim_A.add_packet(center=(16, 16), mass=5.0, width=4.0, initial_q=0.2)

        np.random.seed(456)
        params_B = self._make_deterministic_sim()
        sim_B = DETCollider2D(params_B)
        sim_B.add_packet(center=(16, 16), mass=5.0, width=4.0, initial_q=0.2)

        harness_B = SelfhoodHarness(
            sim_B,
            self_params=SelfFieldParams(mu_S=0.2, phase2_enabled=False),
        )

        for t in range(200):
            sim_A.step()
            harness_B.step()

        fields = {
            "a": (sim_A.a, sim_B.a),
            "F": (sim_A.F, sim_B.F),
            "q": (sim_A.q, sim_B.q),
            "C_E": (sim_A.C_E, sim_B.C_E),
            "C_S": (sim_A.C_S, sim_B.C_S),
            "P": (sim_A.P, sim_B.P),
        }

        for name, (fA, fB) in fields.items():
            dev = float(np.max(np.abs(fA - fB)))
            print(f"  {name}: max deviation = {dev:.2e}")
            assert dev < 1e-12, (
                f"Canonical field '{name}' diverged: max deviation = {dev:.2e}"
            )
