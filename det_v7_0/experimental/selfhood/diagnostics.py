"""
Selfhood Diagnostics Harness for DET v7.

Wraps a DETCollider2D instance and computes all speculative selfhood
observables (R, H_host, S, I_self, D_mature) as post-step diagnostics.

Phase 1: readout-only. No canonical law is modified.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .host_fitness import (
    HostFitnessParams,
    compute_neighborhood_averages_2d,
    compute_reciprocity_2d,
    compute_pointer_stability_proxy,
    compute_host_fitness,
    update_developmental_maturity,
)
from .self_field import (
    SelfFieldParams,
    update_self_field,
    compute_self_assisted_healing_2d,
)
from .identity_metrics import (
    IdentityMetricParams,
    Snapshot,
    compute_identity_persistence,
)


@dataclass
class SelfhoodDiagnostics:
    """Container for all speculative selfhood fields."""
    R: np.ndarray              # reciprocity
    H_host: np.ndarray         # host fitness
    S: np.ndarray              # self-coherence occupancy
    D_mature: np.ndarray       # developmental maturity
    M_ptr: np.ndarray          # pointer stability proxy
    C_bar: np.ndarray          # local average coherence
    a_bar: np.ndarray          # local average agency
    P_bar: np.ndarray          # local average presence
    q_bar: np.ndarray          # local average structural debt


class SelfhoodHarness:
    """Wraps a DETCollider2D and computes selfhood diagnostics after each step.

    Usage:
        sim = DETCollider2D(params)
        harness = SelfhoodHarness(sim)
        for _ in range(steps):
            harness.step()
        print(harness.diag.S.mean())
    """

    def __init__(
        self,
        collider,
        host_params: HostFitnessParams | None = None,
        self_params: SelfFieldParams | None = None,
        identity_params: IdentityMetricParams | None = None,
    ):
        self.sim = collider
        self.host_params = host_params or HostFitnessParams()
        self.self_params = self_params or SelfFieldParams()
        self.identity_params = identity_params or IdentityMetricParams()

        N = self.sim.p.N
        shape = (N, N)

        # Initialize speculative fields
        self.S = np.zeros(shape, dtype=np.float64)
        self.D_mature = np.zeros(shape, dtype=np.float64)

        # Latest diagnostics
        self.diag: SelfhoodDiagnostics | None = None

        # Snapshot history for identity persistence
        self._snapshots: list[tuple[float, Snapshot]] = []

    def step(self):
        """Run one canonical DET step, then compute selfhood diagnostics."""
        # === CANONICAL DET STEP (unmodified) ===
        self.sim.step()

        # === SPECULATIVE DIAGNOSTICS (readout-only, after step 15) ===
        self._compute_diagnostics()

    def _compute_diagnostics(self):
        """Compute all speculative selfhood fields from current collider state."""
        sim = self.sim

        # 1. Neighborhood averages
        avgs = compute_neighborhood_averages_2d(
            a=sim.a, P=sim.P, C_E=sim.C_E, C_S=sim.C_S, q=sim.q,
        )

        # 2. Reciprocity
        R = compute_reciprocity_2d(
            C_E=sim.C_E, C_S=sim.C_S, F=sim.F, a=sim.a,
            epsilon=self.host_params.epsilon,
        )

        # 3. Pointer stability proxy
        M_ptr = compute_pointer_stability_proxy(
            q=sim.q, C_E=sim.C_E, C_S=sim.C_S,
        )

        # 4. Developmental maturity (if enabled)
        if self.host_params.developmental_enabled:
            self.D_mature = update_developmental_maturity(
                D_mature=self.D_mature,
                C_bar=avgs["C_bar"],
                R=R,
                P_bar=avgs["P_bar"],
                params=self.host_params,
            )

        # 5. Host fitness
        H_host = compute_host_fitness(
            a_bar=avgs["a_bar"],
            P_bar=avgs["P_bar"],
            C_bar=avgs["C_bar"],
            q_bar=avgs["q_bar"],
            R=R,
            M_ptr=M_ptr,
            params=self.host_params,
            D_mature=self.D_mature if self.host_params.developmental_enabled else None,
        )

        # 6. Self-coherence occupancy
        self.S = update_self_field(
            S=self.S,
            H_host=H_host,
            C_bar=avgs["C_bar"],
            q_bar=avgs["q_bar"],
            params=self.self_params,
        )

        # 7. Phase 2: optional self-assisted healing (disabled by default)
        if self.self_params.phase2_enabled:
            D = 1.0 / (1.0 + sim.p.lambda_P * sim.q)
            dC_E, dC_S = compute_self_assisted_healing_2d(
                S=self.S,
                a=sim.a,
                C_E=sim.C_E,
                C_S=sim.C_S,
                D=D,
                Delta_tau=sim.Delta_tau,
                D_mature=self.D_mature if self.host_params.developmental_enabled else None,
                params=self.self_params,
            )
            sim.C_E = np.clip(sim.C_E + dC_E, sim.p.C_init, 1.0)
            sim.C_S = np.clip(sim.C_S + dC_S, sim.p.C_init, 1.0)

        # Store diagnostics
        self.diag = SelfhoodDiagnostics(
            R=R,
            H_host=H_host,
            S=self.S.copy(),
            D_mature=self.D_mature.copy(),
            M_ptr=M_ptr,
            C_bar=avgs["C_bar"],
            a_bar=avgs["a_bar"],
            P_bar=avgs["P_bar"],
            q_bar=avgs["q_bar"],
        )

    def take_snapshot(self) -> Snapshot:
        """Take an identity snapshot of the current state."""
        if self.diag is None:
            self._compute_diagnostics()
        snap = Snapshot(
            S=self.S.copy(),
            C_E=self.sim.C_E.copy(),
            C_S=self.sim.C_S.copy(),
            C_bar=self.diag.C_bar.copy(),
            M_ptr=self.diag.M_ptr.copy(),
            time=self.sim.time,
        )
        self._snapshots.append((self.sim.time, snap))
        return snap

    def compute_identity(self, snap1: Snapshot, snap2: Snapshot) -> dict:
        """Compute identity persistence between two snapshots."""
        return compute_identity_persistence(snap1, snap2, self.identity_params)
