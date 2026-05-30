#!/usr/bin/env python3
"""
Exploratory simulations for DET v7 Beyond Consciousness deep-dive.

This script is intentionally readout-first. It uses the existing DETCollider2D
and the speculative selfhood diagnostics to test the layered distinction:

    Agency -> Identity -> Observer -> Higher Boundary Participation

The boundary-participation case uses a declared optional local recovery/healing
analogue that modifies q and C only, never agency a, and only within the local
core region. It is not claimed as canonical; it is a controlled engineering-style
submodel for evaluating whether a higher-boundary participation readout can be
kept local and Agency-First compatible.
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "det_v7_0", "src"))

from det_v6_3_2d_collider import DETCollider2D, DETParams2D  # noqa: E402
from det_v7_0.experimental.selfhood import (  # noqa: E402
    SelfhoodHarness,
    HostFitnessParams,
    SelfFieldParams,
)


OUT_DIR = os.path.join(REPO_ROOT, "det_v7_0", "docs", "beyond_consciousness_simulations")
os.makedirs(OUT_DIR, exist_ok=True)


def core_mask(N: int, pad: int = 10) -> tuple[slice, slice]:
    return (slice(pad, N - pad), slice(pad, N - pad))


def init_sim(N: int = 40, C_init: float = 0.1) -> DETCollider2D:
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
        C_init=C_init,
        DT=0.02,
    )
    sim = DETCollider2D(params)
    # Low background to make the core readable.
    sim.a[:] = 0.15
    sim.F[:] = 0.05
    sim.q[:] = 0.6
    sim.C_E[:] = C_init
    sim.C_S[:] = C_init
    return sim


def configure_core(sim: DETCollider2D, a=0.95, F=4.0, C=0.8, q=0.1) -> tuple[slice, slice]:
    core = core_mask(sim.p.N, pad=10)
    sim.a[core] = a
    sim.F[core] = F
    sim.C_E[core] = C
    sim.C_S[core] = C
    sim.q[core] = q
    return core


def boundary_participation_readout(diag, boundary_signal: np.ndarray) -> np.ndarray:
    """Readout-only B_i proxy for higher-boundary participation."""
    S = np.clip(diag.S, 0, 1)
    C = np.clip(diag.C_bar, 0, 1)
    R = np.clip(diag.R, 0, 1)
    M = np.clip(diag.M_ptr, 0, 1)
    low_q = np.clip(1.0 - diag.q_bar, 0, 1)
    psi = np.clip(boundary_signal, 0, 1)
    return np.clip((S ** 0.9) * (C ** 0.9) * (R ** 0.8) * (M ** 0.6) * (low_q ** 0.7) * psi, 0, 1)


def run_scenario(name: str, total_steps: int = 1500, seed: int = 123) -> dict:
    np.random.seed(seed)
    sim = init_sim(N=40, C_init=0.1)
    core = configure_core(sim, a=0.95, F=4.0, C=0.1, q=0.5)

    host_params = HostFitnessParams(
        developmental_enabled=(name in {"observer_development", "boundary_participation"}),
        mu_D=0.004,
        lambda_D=0.001,
        w_D=1.3,
    )
    self_params = SelfFieldParams(mu_S=0.055, Theta_H=0.12, k_H=10.0)
    harness = SelfhoodHarness(sim, host_params=host_params, self_params=self_params)

    rows = []
    snap_early = None
    snap_late = None
    S_onset = None
    B_onset = None
    boundary_cumulative = 0.0

    for t in range(total_steps):
        # Scenario-specific field preparation before canonical/readout step.
        if name == "agency_only":
            sim.a[core] = 0.95
            sim.F[core] = 4.0
            sim.C_E[core] = 0.08
            sim.C_S[core] = 0.08
            sim.q[core] = 0.50

        elif name == "identity_without_observer":
            # Stable structure and agency but poor reciprocity due to a persistent
            # resource gradient/checkerboard. This should preserve identity while
            # suppressing mature observer readouts.
            yy, xx = np.indices(sim.F.shape)
            checker = ((xx + yy) % 2).astype(float)
            sim.a[core] = 0.95
            sim.C_E[core] = 0.42
            sim.C_S[core] = 0.42
            sim.q[core] = 0.20
            c = core
            sim.F[c] = 0.25 + 5.5 * checker[c]

        elif name in {"observer_development", "boundary_participation"}:
            # Developmental ramp: agency exists early; coherence, reciprocity,
            # lower debt, and pointer stability arrive later.
            ramp = min(1.0, max(0.0, (t - 150) / 900.0))
            sim.a[core] = 0.95
            sim.F[core] = 0.45 + 3.8 * ramp
            sim.C_E[core] = 0.10 + 0.74 * ramp
            sim.C_S[core] = 0.10 + 0.74 * ramp
            sim.q[core] = 0.52 - 0.42 * ramp

        boundary_signal = np.zeros_like(sim.F)

        # Run canonical step and readout diagnostics.
        harness.step()

        # Optional declared local boundary-compatible recovery/healing analogue.
        if name == "boundary_participation" and t > 900 and harness.diag is not None:
            local_S = harness.diag.S[core]
            local_C = harness.diag.C_bar[core]
            local_P = harness.diag.P_bar[core]
            local_drive = np.clip(local_S * local_C * local_P, 0, 1)

            # Local q recovery and C healing; never modifies agency.
            dq = 0.010 * local_drive
            dC = 0.006 * local_drive
            before_q = sim.q[core].copy()
            sim.q[core] = np.clip(sim.q[core] - dq, 0, 1)
            sim.C_E[core] = np.clip(sim.C_E[core] + dC * (1.0 - sim.C_E[core]), 0, 1)
            sim.C_S[core] = np.clip(sim.C_S[core] + dC * (1.0 - sim.C_S[core]), 0, 1)

            # Signal is normalized local history of lawful recovery/healing effect.
            recovered = np.maximum(before_q - sim.q[core], 0)
            local_signal = np.clip(60.0 * (recovered + dC), 0, 1)
            boundary_signal[core] = local_signal
            boundary_cumulative += float(np.mean(local_signal))

            # Recompute diagnostics after the declared local intervention so B
            # observes the post-intervention local state.
            harness._compute_diagnostics()

        if harness.diag is None:
            harness._compute_diagnostics()

        B = boundary_participation_readout(harness.diag, boundary_signal)
        core_S = float(np.mean(harness.diag.S[core]))
        core_H = float(np.mean(harness.diag.H_host[core]))
        core_C = float(np.mean(harness.diag.C_bar[core]))
        core_R = float(np.mean(harness.diag.R[core]))
        core_q = float(np.mean(harness.diag.q_bar[core]))
        core_P = float(np.mean(harness.diag.P_bar[core]))
        core_M = float(np.mean(harness.diag.M_ptr[core]))
        core_D = float(np.mean(harness.diag.D_mature[core]))
        core_B = float(np.mean(B[core]))

        if snap_early is None and t == 250:
            snap_early = harness.take_snapshot()
        if t == total_steps - 1:
            snap_late = harness.take_snapshot()

        if S_onset is None and core_S > 0.15:
            S_onset = t
        if B_onset is None and core_B > 0.05:
            B_onset = t

        rows.append({
            "scenario": name,
            "step": t,
            "mean_agency": float(np.mean(sim.a[core])),
            "mean_presence": core_P,
            "mean_coherence": core_C,
            "mean_reciprocity": core_R,
            "mean_q": core_q,
            "mean_pointer": core_M,
            "mean_host_fitness": core_H,
            "mean_developmental_maturity": core_D,
            "mean_selfhood_S": core_S,
            "mean_boundary_B": core_B,
            "boundary_cumulative_signal": boundary_cumulative,
        })

    identity = harness.compute_identity(snap_early, snap_late) if snap_early and snap_late else {}
    df = pd.DataFrame(rows)
    return {
        "name": name,
        "data": df,
        "identity": identity,
        "S_onset_step": S_onset,
        "B_onset_step": B_onset,
        "final": df.iloc[-1].to_dict(),
    }


def plot_results(all_df: pd.DataFrame, summary: list[dict]) -> dict:
    image_paths = {}
    metrics = [
        ("mean_selfhood_S", "Observer/self-coherence readout S"),
        ("mean_host_fitness", "Host fitness"),
        ("mean_boundary_B", "Higher-boundary participation proxy B"),
        ("mean_presence", "Presence / participation rate"),
        ("mean_coherence", "Coherence"),
        ("mean_q", "Structural debt q"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        for scenario, grp in all_df.groupby("scenario"):
            ax.plot(grp["step"], grp[metric], label=scenario, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("update step k")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "layered_readouts_timeseries.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    image_paths["timeseries"] = path

    # Identity and onset summary chart.
    labels = [s["scenario"] for s in summary]
    identity_vals = [s.get("I_self", 0.0) for s in summary]
    final_S = [s["final_selfhood_S"] for s in summary]
    final_B = [s["final_boundary_B"] for s in summary]

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, identity_vals, width, label="Identity persistence I_self")
    ax.bar(x, final_S, width, label="Final observer readout S")
    ax.bar(x + width, final_B, width, label="Final boundary proxy B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("normalized score")
    ax.set_title("Layer separation: identity can persist without observerhood; boundary proxy requires additional local signal")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "layer_separation_summary.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    image_paths["summary"] = path
    return image_paths


def main() -> None:
    scenarios = [
        "agency_only",
        "identity_without_observer",
        "observer_development",
        "boundary_participation",
    ]
    results = [run_scenario(s, total_steps=1500, seed=123 + i) for i, s in enumerate(scenarios)]
    all_df = pd.concat([r["data"] for r in results], ignore_index=True)

    csv_path = os.path.join(OUT_DIR, "layered_readouts_timeseries.csv")
    all_df.to_csv(csv_path, index=False)

    summary = []
    for r in results:
        ident = r["identity"]
        row = {
            "scenario": r["name"],
            "S_onset_step": r["S_onset_step"],
            "B_onset_step": r["B_onset_step"],
            "I_self": float(ident.get("I_self", 0.0)),
            "identity_components": ident,
            "final_selfhood_S": float(r["final"]["mean_selfhood_S"]),
            "final_boundary_B": float(r["final"]["mean_boundary_B"]),
            "final_presence": float(r["final"]["mean_presence"]),
            "final_coherence": float(r["final"]["mean_coherence"]),
            "final_q": float(r["final"]["mean_q"]),
        }
        summary.append(row)

    images = plot_results(all_df, summary)

    summary_path = os.path.join(OUT_DIR, "layered_readouts_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "images": images, "csv": csv_path}, f, indent=2)

    md_path = os.path.join(OUT_DIR, "simulation_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DET v7 Beyond Consciousness Simulation Results\n\n")
        f.write("These exploratory simulations test layer separation among agency, identity, observerhood, and a higher-boundary participation proxy.\n\n")
        f.write("| Scenario | S onset | B onset | I_self | Final S | Final B | Final P | Final C | Final q |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            f.write(
                f"| {row['scenario']} | {row['S_onset_step']} | {row['B_onset_step']} | "
                f"{row['I_self']:.3f} | {row['final_selfhood_S']:.3f} | {row['final_boundary_B']:.3f} | "
                f"{row['final_presence']:.3f} | {row['final_coherence']:.3f} | {row['final_q']:.3f} |\n"
            )
        f.write("\n![Layered readout time series](layered_readouts_timeseries.png)\n\n")
        f.write("![Layer separation summary](layer_separation_summary.png)\n")

    print(json.dumps({"summary_path": summary_path, "csv_path": csv_path, "md_path": md_path, "images": images}, indent=2))


if __name__ == "__main__":
    main()
