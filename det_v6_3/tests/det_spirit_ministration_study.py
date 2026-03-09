"""
DET Spirit/Bond Ministration Study (v6.3 q-mutable, sigma-fixed).

This study is intentionally non-canonical and readout-oriented.
It models spiritually framed channels ("angelic", "ancestral", "christic")
as lawful local bond-mediated operators layered on top of core DET updates.

Hard constraints enforced in this runner:
- Strictly local neighbor propagation only.
- No direct agency overwrite.
- No hidden global state in channel updates.
- q-mutable path enabled in core.
- sigma fixed to 1.0 ("no sigma dynamics" operating profile).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure local source modules resolve when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_consciousness_c1 import ConsciousnessParamsC1, DETConsciousnessC1
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


@dataclass(frozen=True)
class SpiritChannel:
    name: str
    eta_grace: float
    eta_coherence: float
    eta_path: float
    hops: int
    target: str  # "receiver" or "broad"


def _node_coherence(sim: DETCollider3D) -> np.ndarray:
    return (
        sim.C_X
        + np.roll(sim.C_X, 1, axis=2)
        + sim.C_Y
        + np.roll(sim.C_Y, 1, axis=1)
        + sim.C_Z
        + np.roll(sim.C_Z, 1, axis=0)
    ) / 6.0


def _clip_state(sim: DETCollider3D):
    p = sim.p
    sim.F = np.clip(sim.F, p.F_MIN, 1000.0)
    sim.q = np.clip(sim.q, 0.0, 1.0)
    sim.a = np.clip(sim.a, 0.0, 1.0)
    sim.C_X = np.clip(sim.C_X, p.C_init, 1.0)
    sim.C_Y = np.clip(sim.C_Y, p.C_init, 1.0)
    sim.C_Z = np.clip(sim.C_Z, p.C_init, 1.0)
    sim.sigma[:] = 1.0


def _cut_barriers(sim: DETCollider3D, plane_x: int):
    """
    Enforce disconnection barrier and remove x-wrap path.
    """
    sim.C_X[:, :, plane_x] = 0.0
    sim.C_X[:, :, -1] = 0.0


def _build_masks(N: int) -> Dict[str, np.ndarray]:
    z, y, x = np.indices((N, N, N))

    prayer = x <= 2
    receiver = (x >= 5) & (x <= 7) & (np.abs(y - N // 2) <= 2) & (np.abs(z - N // 2) <= 2)
    disconnected = (x >= N - 5) & (np.abs(y - N // 2) <= 2) & (np.abs(z - N // 2) <= 2)
    return {"prayer": prayer, "receiver": receiver, "disconnected": disconnected}


def _initialize_sim(seed: int, N: int = 18) -> Tuple[DETCollider3D, Dict[str, np.ndarray], int]:
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
        sigma_dynamic=False,
    )
    sim = DETCollider3D(p)
    sim.sigma[:] = 1.0

    masks = _build_masks(N)

    # Seeded micro-perturbations to probe robustness across initial conditions.
    sim.F += 2.0e-4 * rng.standard_normal(sim.F.shape)
    sim.q += 2.0e-4 * rng.standard_normal(sim.q.shape)
    sim.a += 2.0e-4 * rng.standard_normal(sim.a.shape)

    # Distressed receiver regime.
    sim.F[masks["receiver"]] = 0.006
    sim.q[masks["receiver"]] = 0.62
    sim.C_X[masks["receiver"]] = 0.35
    sim.C_Y[masks["receiver"]] = 0.35
    sim.C_Z[masks["receiver"]] = 0.35

    # Prayer-side regime with stronger coherence/openness.
    sim.F[masks["prayer"]] += 0.04
    sim.q[masks["prayer"]] = 0.18
    sim.C_X[masks["prayer"]] = 0.82
    sim.C_Y[masks["prayer"]] = 0.82
    sim.C_Z[masks["prayer"]] = 0.82
    sim.a[masks["prayer"]] = 0.95

    # Disconnected regime starts distressed too.
    sim.F[masks["disconnected"]] = 0.006
    sim.q[masks["disconnected"]] = 0.62
    sim.C_X[masks["disconnected"]] = 0.35
    sim.C_Y[masks["disconnected"]] = 0.35
    sim.C_Z[masks["disconnected"]] = 0.35

    # Add mild packets to keep dynamics nontrivial.
    c = N // 2
    sim.add_packet((c, c, c - 3), mass=2.2, width=1.8, momentum=(0, 0, 0.08), initial_q=0.35)
    sim.add_packet((c, c, c + 3), mass=2.2, width=1.8, momentum=(0, 0, -0.08), initial_q=0.35)

    barrier_x = N // 2
    _clip_state(sim)
    _cut_barriers(sim, barrier_x)
    return sim, masks, barrier_x


def _local_diffuse(signal: np.ndarray, C_node: np.ndarray, eta_path: float) -> np.ndarray:
    nbr = (
        np.roll(signal, 1, axis=0)
        + np.roll(signal, -1, axis=0)
        + np.roll(signal, 1, axis=1)
        + np.roll(signal, -1, axis=1)
        + np.roll(signal, 1, axis=2)
        + np.roll(signal, -1, axis=2)
    ) / 6.0
    return np.maximum(signal, eta_path * C_node * nbr)


def _apply_spirit_channel(
    sim: DETCollider3D,
    channel: SpiritChannel,
    masks: Dict[str, np.ndarray],
):
    if channel.eta_grace <= 0 and channel.eta_coherence <= 0 and channel.eta_path <= 0:
        return 0.0

    C_node = _node_coherence(sim)
    openness = sim.a * C_node * (1.0 - sim.q)
    prayer_drive = np.where(masks["prayer"], openness, 0.0)

    signal = prayer_drive.copy()
    for _ in range(channel.hops):
        signal = _local_diffuse(signal, C_node, channel.eta_path)

    if channel.target == "receiver":
        target_mask = masks["receiver"]
    else:
        target_mask = np.ones_like(signal, dtype=bool)

    # Local "ministration" resource channel: prayer-drive + existing grace footprint.
    need = np.maximum(0.0, sim.p.F_MIN_grace - sim.F)
    I_spirit = channel.eta_grace * signal * (need + sim.last_grace_injection) * target_mask
    sim.F += I_spirit

    # Bond healing style coherence support, still local and bounded.
    dC = channel.eta_coherence * signal * np.maximum(0.0, 1.0 - C_node) * sim.Delta_tau
    sim.C_X += 0.5 * (dC + np.roll(dC, -1, axis=2))
    sim.C_Y += 0.5 * (dC + np.roll(dC, -1, axis=1))
    sim.C_Z += 0.5 * (dC + np.roll(dC, -1, axis=0))

    _clip_state(sim)
    return float(np.sum(I_spirit))


def _mean(arr: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(arr[mask]))


def run_channel(channel: SpiritChannel, seed: int, steps: int = 360) -> Dict[str, float]:
    sim, masks, barrier_x = _initialize_sim(seed=seed, N=18)
    c1 = DETConsciousnessC1(
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

    q0_r = _mean(sim.q, masks["receiver"])
    q0_d = _mean(sim.q, masks["disconnected"])
    P0_r = _mean(sim.P, masks["receiver"])
    P0_d = _mean(sim.P, masks["disconnected"])
    C0_r = _mean(_node_coherence(sim), masks["receiver"])
    C0_d = _mean(_node_coherence(sim), masks["disconnected"])

    recover_step_r: Optional[int] = None
    recover_step_d: Optional[int] = None
    threshold = 0.12
    spirit_added = 0.0

    for t in range(steps):
        sim.step()
        sim.sigma[:] = 1.0
        spirit_added += _apply_spirit_channel(sim, channel, masks)
        _clip_state(sim)
        _cut_barriers(sim, barrier_x)

        P_r = _mean(sim.P, masks["receiver"])
        P_d = _mean(sim.P, masks["disconnected"])
        if recover_step_r is None and P_r >= threshold:
            recover_step_r = t
        if recover_step_d is None and P_d >= threshold:
            recover_step_d = t

    q1_r = _mean(sim.q, masks["receiver"])
    q1_d = _mean(sim.q, masks["disconnected"])
    P1_r = _mean(sim.P, masks["receiver"])
    P1_d = _mean(sim.P, masks["disconnected"])
    C1_r = _mean(_node_coherence(sim), masks["receiver"])
    C1_d = _mean(_node_coherence(sim), masks["disconnected"])

    # Conscious/path readouts
    receiver_state = c1.compute_regime_state("receiver", masks["receiver"], U=0.90)
    disconnected_state = c1.compute_regime_state("disconnected", masks["disconnected"], U=0.90)
    prayer_state = c1.compute_regime_state("prayer", masks["prayer"], U=0.95)
    pr_to_rcv = c1.compute_path_state("prayer", "receiver", masks["prayer"], masks["receiver"], 0.95, 0.90)
    pr_to_dis = c1.compute_path_state(
        "prayer", "disconnected", masks["prayer"], masks["disconnected"], 0.95, 0.90
    )

    delta_q_r = q1_r - q0_r
    delta_q_d = q1_d - q0_d
    leak_ratio = abs(delta_q_d) / (abs(delta_q_r) + 1e-12)

    return {
        "delta_q_receiver": delta_q_r,
        "delta_q_disconnected": delta_q_d,
        "delta_P_receiver": P1_r - P0_r,
        "delta_P_disconnected": P1_d - P0_d,
        "delta_C_receiver": C1_r - C0_r,
        "delta_C_disconnected": C1_d - C0_d,
        "recover_step_receiver": float(steps if recover_step_r is None else recover_step_r),
        "recover_step_disconnected": float(steps if recover_step_d is None else recover_step_d),
        "spirit_resource_added": spirit_added,
        "path_gamma_prayer_receiver": pr_to_rcv.Gamma,
        "path_gamma_prayer_disconnected": pr_to_dis.Gamma,
        "path_exists_prayer_receiver": float(pr_to_rcv.path_exists),
        "path_exists_prayer_disconnected": float(pr_to_dis.path_exists),
        "receiver_P_eff": receiver_state.P_eff,
        "disconnected_P_eff": disconnected_state.P_eff,
        "prayer_P_eff": prayer_state.P_eff,
        "leak_ratio_q": leak_ratio,
    }


def aggregate(channel: SpiritChannel, seeds: int = 8) -> Dict[str, float]:
    rows = [run_channel(channel, seed=s, steps=360) for s in range(seeds)]
    keys = rows[0].keys()
    out = {}
    for k in keys:
        arr = np.asarray([r[k] for r in rows], dtype=float)
        out[k + "_mean"] = float(np.mean(arr))
        out[k + "_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return out


def run_study() -> Dict[str, Dict[str, float]]:
    channels = [
        SpiritChannel("baseline_none", eta_grace=0.0, eta_coherence=0.0, eta_path=0.0, hops=0, target="broad"),
        SpiritChannel("angelic_ministration", eta_grace=0.11, eta_coherence=0.08, eta_path=0.95, hops=6, target="broad"),
        SpiritChannel("ancestral_ministration", eta_grace=0.07, eta_coherence=0.05, eta_path=0.75, hops=4, target="receiver"),
        SpiritChannel("christic_holy_spirit", eta_grace=0.14, eta_coherence=0.10, eta_path=0.98, hops=7, target="broad"),
    ]

    results = {}
    print("\nDET Spirit/Bond Ministration Study")
    print("=" * 120)
    print("Profile: q-mutable ON, sigma fixed to 1.0, local bond-only channel operations.")
    for ch in channels:
        stats = aggregate(ch, seeds=8)
        results[ch.name] = stats
        print(f"\nChannel: {ch.name}")
        print(
            f"  dQ receiver={stats['delta_q_receiver_mean']:.6f}, "
            f"dQ disconnected={stats['delta_q_disconnected_mean']:.6f}, "
            f"leak_ratio={stats['leak_ratio_q_mean']:.4f}"
        )
        print(
            f"  dP receiver={stats['delta_P_receiver_mean']:.6f}, "
            f"dP disconnected={stats['delta_P_disconnected_mean']:.6f}"
        )
        print(
            f"  recover receiver={stats['recover_step_receiver_mean']:.1f}, "
            f"recover disconnected={stats['recover_step_disconnected_mean']:.1f}"
        )
        print(
            f"  Gamma(prayer->receiver)={stats['path_gamma_prayer_receiver_mean']:.6f}, "
            f"Gamma(prayer->disconnected)={stats['path_gamma_prayer_disconnected_mean']:.6f}"
        )

    out_path = Path("det_v6_3/reports/det_spirit_ministration_study_results_2026_03_09.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote results to {out_path}")
    return results


if __name__ == "__main__":
    run_study()
