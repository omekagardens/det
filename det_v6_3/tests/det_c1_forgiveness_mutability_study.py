"""
DET-C1 study: conscious integration x forgiveness (q-mutable) interaction.

Research question:
How much additional utopic-trajectory uplift is gained when conscious
integration (higher U) is combined with forgiveness/q-debt reduction?
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_consciousness_c1 import ConsciousnessParamsC1, DETConsciousnessC1
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


@dataclass(frozen=True)
class StudyCondition:
    name: str
    q_mutable_local_enabled: bool
    alpha_q_local_resource_relief: float
    alpha_q_grace_relief: float


STAGES: List[Tuple[str, float, float]] = [
    ("Fragmented", 0.18, 0.16),
    ("Transition", 0.40, 0.38),
    ("Unified", 0.68, 0.65),
    ("Utopic-ish", 0.90, 0.88),
    ("Full", 1.00, 1.00),
]

METRICS = ["P_eff_A", "P_eff_B", "W_A", "W_B", "Gamma", "V", "Acc", "q_A", "q_B", "q_mean"]


def _build_masks(N: int):
    _, _, x = np.indices((N, N, N))
    mask_a = x <= (N // 2 - 2)
    mask_b = x >= (N // 2 + 1)
    return mask_a, mask_b


def _initialize_sim(condition: StudyCondition, seed: int, N: int = 20) -> Tuple[DETCollider3D, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
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
        q_mutable_local_enabled=condition.q_mutable_local_enabled,
        alpha_q_local_resource_relief=condition.alpha_q_local_resource_relief,
        alpha_q_grace_relief=condition.alpha_q_grace_relief,
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

    mask_a, mask_b = _build_masks(N)
    sim.C_X[mask_a] = 0.80
    sim.C_Y[mask_a] = 0.80
    sim.C_Z[mask_a] = 0.80
    sim.C_X[mask_b] = 0.55
    sim.C_Y[mask_b] = 0.55
    sim.C_Z[mask_b] = 0.55

    sim.q[mask_a] = np.minimum(sim.q[mask_a], 0.25)
    sim.q[mask_b] = np.maximum(sim.q[mask_b], 0.40)

    # Small seed-specific perturbation to test robustness.
    sim.F = np.clip(sim.F + rng.normal(0.0, 0.0015, sim.F.shape), p.F_MIN, 1000.0)
    sim.q = np.clip(sim.q + rng.normal(0.0, 0.010, sim.q.shape), 0.0, 1.0)
    sim.C_X = np.clip(sim.C_X + rng.normal(0.0, 0.010, sim.C_X.shape), p.C_init, 1.0)
    sim.C_Y = np.clip(sim.C_Y + rng.normal(0.0, 0.010, sim.C_Y.shape), p.C_init, 1.0)
    sim.C_Z = np.clip(sim.C_Z + rng.normal(0.0, 0.010, sim.C_Z.shape), p.C_init, 1.0)

    return sim, mask_a, mask_b


def _empty_stage_metrics() -> Dict[str, List[float]]:
    return {k: [] for k in METRICS}


def run_condition(condition: StudyCondition, seeds: int = 8, stage_steps: int = 120):
    per_stage: Dict[str, Dict[str, List[float]]] = {name: _empty_stage_metrics() for name, _, _ in STAGES}

    for seed in range(seeds):
        sim, mask_a, mask_b = _initialize_sim(condition, seed=seed, N=20)
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

        for stage_name, u_a, u_b in STAGES:
            for _ in range(stage_steps):
                sim.step()

            state_a = module.compute_regime_state("A", mask_a, U=u_a)
            state_b = module.compute_regime_state("B", mask_b, U=u_b)
            path = module.compute_path_state("A", "B", mask_a, mask_b, U_a=u_a, U_b=u_b)

            stage_store = per_stage[stage_name]
            stage_store["P_eff_A"].append(state_a.P_eff)
            stage_store["P_eff_B"].append(state_b.P_eff)
            stage_store["W_A"].append(state_a.W)
            stage_store["W_B"].append(state_b.W)
            stage_store["Gamma"].append(path.Gamma)
            stage_store["V"].append(path.V)
            stage_store["Acc"].append(path.nonverbal_accuracy)
            stage_store["q_A"].append(float(np.mean(sim.q[mask_a])))
            stage_store["q_B"].append(float(np.mean(sim.q[mask_b])))
            stage_store["q_mean"].append(float(np.mean(sim.q)))

    # Mean/std summary.
    stats = {}
    for stage_name, metric_map in per_stage.items():
        stats[stage_name] = {}
        for metric_name, values in metric_map.items():
            arr = np.asarray(values, dtype=float)
            stats[stage_name][metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
    return stats


def interaction_effect(stats_imm, stats_mut, metric: str, higher_is_better: bool = True):
    """
    Difference-in-differences interaction:
      (Full_mut - Frag_mut) - (Full_imm - Frag_imm)
    """
    sign = 1.0 if higher_is_better else -1.0
    full_mut = stats_mut["Full"][metric]["mean"]
    frag_mut = stats_mut["Fragmented"][metric]["mean"]
    full_imm = stats_imm["Full"][metric]["mean"]
    frag_imm = stats_imm["Fragmented"][metric]["mean"]
    return sign * ((full_mut - frag_mut) - (full_imm - frag_imm))


def _print_condition_table(name: str, stats):
    print(f"\n{name}")
    header = (
        f"{'Stage':>11} | {'P_A':>8} {'P_B':>8} {'W_A':>8} {'W_B':>8} "
        f"{'Gamma':>9} {'V':>8} {'Acc':>8} {'q_mean':>8}"
    )
    print(header)
    print("-" * len(header))
    for stage_name, _, _ in STAGES:
        s = stats[stage_name]
        print(
            f"{stage_name:>11} | "
            f"{s['P_eff_A']['mean']:.6f} {s['P_eff_B']['mean']:.6f} "
            f"{s['W_A']['mean']:.6f} {s['W_B']['mean']:.6f} "
            f"{s['Gamma']['mean']:.6f} {s['V']['mean']:.6f} "
            f"{s['Acc']['mean']:.6f} {s['q_mean']['mean']:.6f}"
        )


def run_study(seeds: int = 8, stage_steps: int = 120):
    conditions = [
        StudyCondition("immutable", False, 0.0, 0.0),
        StudyCondition("mutable_moderate", True, 0.08, 0.25),
        StudyCondition("mutable_strong", True, 0.16, 0.35),
    ]

    all_stats = {}
    for cond in conditions:
        all_stats[cond.name] = run_condition(cond, seeds=seeds, stage_steps=stage_steps)
        _print_condition_table(cond.name, all_stats[cond.name])

    print("\nInteraction Effects vs immutable (positive = stronger utopic slope)")
    print("Metric         moderate        strong")
    print("-------------------------------------------")
    better = {
        "P_eff_A": True,
        "P_eff_B": True,
        "W_A": False,
        "W_B": False,
        "Gamma": True,
        "V": False,
        "Acc": True,
        "q_mean": False,
    }
    for metric, high_good in better.items():
        did_mod = interaction_effect(all_stats["immutable"], all_stats["mutable_moderate"], metric, high_good)
        did_str = interaction_effect(all_stats["immutable"], all_stats["mutable_strong"], metric, high_good)
        print(f"{metric:12} {did_mod:>11.6f} {did_str:>11.6f}")

    return all_stats


if __name__ == "__main__":
    run_study(seeds=8, stage_steps=120)
