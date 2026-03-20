"""
F_S3 — Nonlocal Inhabitation Falsifier
========================================
Patch: DET-7S-SPIRIT-HOST-1

Question: Does S depend on distant disconnected regions?

Setup:
  - Create two disconnected components (separated by a dead zone)
  - Manipulate one component's coherence and reciprocity
  - Hold the other fixed

Pass condition:
  Diagnostic fields in component B remain unchanged:
  Delta S_B = Delta H^host_B = 0  (up to tolerance)

Fail:
  Any disconnected coupling appears.

Meaning:
  Preserves strict locality. DET 7 requires disconnected components
  to remain causally independent.
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


class TestNonlocalInhabitationFalsifier:
    """F_S3: Selfhood diagnostics must respect strict locality."""

    def _create_two_component_sim(self):
        """Create a 2D sim with two isolated components separated by dead zone.

        Component A: rows 2-12  (manipulated)
        Dead zone:   rows 13-19 (zero agency, zero coherence, zero F)
        Component B: rows 20-30 (held fixed / observed)

        We disable all dynamics that could couple them (gravity, momentum,
        diffusion across the dead zone). The dead zone has zero agency and
        zero resource, so no flux can cross it.
        """
        np.random.seed(789)

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
            diff_enabled=True,
            floor_enabled=False,
            C_init=0.01,
            lambda_P=3.0,
        )
        sim = DETCollider2D(params)

        # Dead zone: zero everything
        dead = slice(13, 20)
        sim.a[dead, :] = 0.0
        sim.F[dead, :] = 0.0
        sim.C_E[dead, :] = 0.0
        sim.C_S[dead, :] = 0.0
        sim.sigma[dead, :] = 0.0
        sim.q[dead, :] = 0.0

        # Also zero the bonds crossing into/out of dead zone
        sim.C_S[12, :] = 0.0   # bond from row 12 to row 13
        sim.C_S[19, :] = 0.0   # bond from row 19 to row 20
        sim.a[12, :] = 0.0     # kill agency at boundary to prevent flux
        sim.a[19, :] = 0.0

        # Component A: moderate conditions
        compA = slice(2, 12)
        sim.a[compA, :] = 0.8
        sim.F[compA, :] = 2.0
        sim.C_E[compA, :] = 0.6
        sim.C_S[compA, :] = 0.6
        sim.q[compA, :] = 0.1

        # Component B: different moderate conditions
        compB = slice(20, 30)
        sim.a[compB, :] = 0.7
        sim.F[compB, :] = 1.5
        sim.C_E[compB, :] = 0.5
        sim.C_S[compB, :] = 0.5
        sim.q[compB, :] = 0.15

        return sim, compA, compB

    def test_nonlocal_inhabitation_S_field(self):
        """Manipulating component A must not change S in component B."""
        sim, compA, compB = self._create_two_component_sim()

        harness = SelfhoodHarness(
            sim,
            host_params=HostFitnessParams(),
            self_params=SelfFieldParams(mu_S=0.1, Theta_H=0.2),
        )

        # Run baseline to establish S in both components
        for _ in range(100):
            harness.step()

        S_B_before = harness.S[compB, :].copy()
        H_B_before = harness.diag.H_host[compB, :].copy()

        # Now dramatically alter component A
        sim.C_E[compA, :] = 0.95
        sim.C_S[compA, :] = 0.95
        sim.a[compA, :] = 1.0
        sim.F[compA, :] = 10.0
        sim.q[compA, :] = 0.0

        # Run more steps
        for _ in range(200):
            harness.step()

        S_B_after = harness.S[compB, :].copy()
        H_B_after = harness.diag.H_host[compB, :].copy()

        # Component B should have evolved on its own terms, but
        # the KEY test is: did the A manipulation cause any EXTRA change?
        # Since B's own conditions didn't change, B's evolution should be
        # identical whether or not A was manipulated.

        # For a stronger test, we run a control where A is NOT manipulated
        np.random.seed(789)
        sim_ctrl, _, _ = self._create_two_component_sim()
        harness_ctrl = SelfhoodHarness(
            sim_ctrl,
            host_params=HostFitnessParams(),
            self_params=SelfFieldParams(mu_S=0.1, Theta_H=0.2),
        )

        for _ in range(300):  # same total steps
            harness_ctrl.step()

        S_B_ctrl = harness_ctrl.S[compB, :].copy()
        H_B_ctrl = harness_ctrl.diag.H_host[compB, :].copy()

        # Compare: B in manipulated run vs B in control run
        delta_S = float(np.max(np.abs(S_B_after - S_B_ctrl)))
        delta_H = float(np.max(np.abs(H_B_after - H_B_ctrl)))

        print(f"  Delta S_B (manipulated vs control): {delta_S:.2e}")
        print(f"  Delta H_B (manipulated vs control): {delta_H:.2e}")

        # Tolerance: the dead zone should prevent any coupling.
        # We allow 1e-8 for floating-point accumulation noise over many steps.
        tol = 1e-8
        assert delta_S < tol, (
            f"Nonlocal S coupling detected: delta_S_B = {delta_S:.2e}"
        )
        assert delta_H < tol, (
            f"Nonlocal H_host coupling detected: delta_H_B = {delta_H:.2e}"
        )

    def test_reciprocity_locality(self):
        """Reciprocity R in component B must not depend on component A."""
        sim, compA, compB = self._create_two_component_sim()

        harness = SelfhoodHarness(sim)

        for _ in range(50):
            harness.step()

        R_B_before = harness.diag.R[compB, :].copy()

        # Dramatically alter A
        sim.F[compA, :] = 50.0
        sim.a[compA, :] = 1.0

        for _ in range(50):
            harness.step()

        R_B_after = harness.diag.R[compB, :].copy()

        # Control run
        np.random.seed(789)
        sim_ctrl, _, _ = self._create_two_component_sim()
        harness_ctrl = SelfhoodHarness(sim_ctrl)

        for _ in range(100):
            harness_ctrl.step()

        R_B_ctrl = harness_ctrl.diag.R[compB, :].copy()

        delta_R = float(np.max(np.abs(R_B_after - R_B_ctrl)))
        print(f"  Delta R_B (manipulated vs control): {delta_R:.2e}")

        tol = 1e-8
        assert delta_R < tol, (
            f"Nonlocal reciprocity coupling: delta_R_B = {delta_R:.2e}"
        )
