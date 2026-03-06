"""
DET-C1 stage simulation runner (no plotting dependency).

Runs a connected two-regime scenario plus a disconnected control:
- Reports regime-level P_eff, W, R and path-level Gamma, V, nonverbal accuracy.
- Keeps all dynamics local by reading from the 3D collider state.
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_consciousness_c1 import ConsciousnessParamsC1, DETConsciousnessC1
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def _build_masks(N: int) -> Tuple[np.ndarray, np.ndarray]:
    _, _, x = np.indices((N, N, N))
    mask_a = x <= (N // 2 - 2)
    mask_b = x >= (N // 2 + 1)
    return mask_a, mask_b


def _initialize_sim(N: int = 20) -> DETCollider3D:
    p = DETParams3D(
        N=N,
        DT=0.02,
        F_VAC=0.01,
        gravity_enabled=True,
        momentum_enabled=True,
        angular_momentum_enabled=False,
        floor_enabled=True,
        boundary_enabled=True,
        grace_enabled=True,
        F_MIN_grace=0.10,
        q_enabled=True,
        alpha_q=0.015,
        q_mutable_local_enabled=True,
        alpha_q_local_resource_relief=0.08,
        alpha_q_grace_relief=0.25,
        agency_dynamic=True,
        debt_conductivity_enabled=True,
        xi_conductivity=3.0,
        debt_temporal_enabled=True,
        zeta_temporal=0.6,
        debt_decoherence_enabled=True,
        theta_decoherence=1.2,
    )
    sim = DETCollider3D(p)

    c = N // 2
    sim.add_packet((c, c, c - 4), mass=4.0, width=2.2, momentum=(0, 0, 0.10), initial_q=0.55)
    sim.add_packet((c, c, c + 4), mass=4.0, width=2.2, momentum=(0, 0, -0.10), initial_q=0.55)

    # Two semi-distinct regimes with a corridor in the middle.
    mask_a, mask_b = _build_masks(N)
    sim.C_X[mask_a] = 0.80
    sim.C_Y[mask_a] = 0.80
    sim.C_Z[mask_a] = 0.80
    sim.C_X[mask_b] = 0.55
    sim.C_Y[mask_b] = 0.55
    sim.C_Z[mask_b] = 0.55

    sim.q[mask_a] = np.minimum(sim.q[mask_a], 0.25)
    sim.q[mask_b] = np.maximum(sim.q[mask_b], 0.40)
    return sim


def _format_stage_row(stage: str, state_a, state_b, path):
    return (
        f"{stage:>12} | "
        f"U_A={state_a.U:.2f} U_B={state_b.U:.2f} | "
        f"P_eff_A={state_a.P_eff:.4f} P_eff_B={state_b.P_eff:.4f} | "
        f"W_A={state_a.W:.4f} W_B={state_b.W:.4f} | "
        f"Gamma={path.Gamma:.5f} V={path.V:.5f} Acc={path.nonverbal_accuracy:.5f}"
    )


def run_det_c1_stage_simulation(seed: int = 7, stage_steps: int = 120) -> Dict[str, float]:
    np.random.seed(seed)
    sim = _initialize_sim(N=20)
    mask_a, mask_b = _build_masks(sim.p.N)

    module = DETConsciousnessC1(
        sim,
        ConsciousnessParamsC1(
            alpha_U=1.0,
            beta_U=1.0,
            V0=1.0,
            path_presence_min=0.01,
            path_coherence_min=0.05,
            periodic_paths=False,
        ),
    )

    stages = [
        ("Fragmented", 0.18, 0.16),
        ("Transition", 0.40, 0.38),
        ("Unified", 0.68, 0.65),
        ("Utopic-ish", 0.90, 0.88),
    ]

    print("\nDET-C1 Connected Scenario")
    print("=" * 110)
    for stage_name, u_a, u_b in stages:
        for _ in range(stage_steps):
            sim.step()

        state_a = module.compute_regime_state("A", mask_a, U=u_a)
        state_b = module.compute_regime_state("B", mask_b, U=u_b)
        path = module.compute_path_state("A", "B", mask_a, mask_b, U_a=u_a, U_b=u_b)
        print(_format_stage_row(stage_name, state_a, state_b, path))

    # Disconnected control for locality claim (C4-like check).
    sim_control = _initialize_sim(N=20)
    mask_a_c, mask_b_c = _build_masks(sim_control.p.N)
    module_control = DETConsciousnessC1(
        sim_control,
        ConsciousnessParamsC1(
            path_presence_min=0.01,
            path_coherence_min=0.05,
            periodic_paths=False,
        ),
    )
    mid_bond_x = sim_control.p.N // 2 - 1
    sim_control.C_X[:, :, mid_bond_x] = 0.0

    for _ in range(stage_steps):
        sim_control.step()
        sim_control.C_X[:, :, mid_bond_x] = 0.0

    disconnected = module_control.compute_path_state(
        "A", "B", mask_a_c, mask_b_c, U_a=0.9, U_b=0.9
    )

    print("\nDET-C1 Disconnected Control")
    print("=" * 110)
    print(
        f"path_exists={disconnected.path_exists} "
        f"Gamma={disconnected.Gamma:.6f} "
        f"Acc={disconnected.nonverbal_accuracy:.6f}"
    )

    return {
        "connected_final_gamma": float(path.Gamma),
        "connected_final_accuracy": float(path.nonverbal_accuracy),
        "disconnected_gamma": float(disconnected.Gamma),
        "disconnected_accuracy": float(disconnected.nonverbal_accuracy),
    }


if __name__ == "__main__":
    results = run_det_c1_stage_simulation(seed=7, stage_steps=120)
    print("\nSummary")
    print("=" * 110)
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
