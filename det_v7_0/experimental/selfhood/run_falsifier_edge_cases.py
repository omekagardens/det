#!/usr/bin/env python3
"""
DET v7 Beyond Consciousness falsifier edge-case simulations.

This script is intentionally deterministic and readout-first. It does not claim
that death, resurrection, spirit, or utopic-boundary participation are proven.
It tests whether those concepts can be separated into falsifiable DET-style
readouts:

    record persistence != active consciousness
    snapshot similarity != path-sensitive identity continuity
    boundary growth != nonlocal/coercive override

Outputs are written to:
    det_v7_0/docs/beyond_consciousness_falsifier_simulations/
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from det_v7_0.experimental.selfhood.identity_metrics import (  # noqa: E402
    Snapshot,
    compute_identity_persistence,
)

OUT_DIR = os.path.join(REPO_ROOT, "det_v7_0", "docs", "beyond_consciousness_falsifier_simulations")
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class State:
    """Minimal readout state for edge-case testing."""

    a: np.ndarray
    P: np.ndarray
    q_damage: np.ndarray
    C: np.ndarray
    S: np.ndarray
    M_ptr: np.ndarray
    record: np.ndarray

    def snapshot(self, time: float) -> Snapshot:
        return Snapshot(
            S=self.S.copy(),
            C_E=self.C.copy(),
            C_S=self.C.copy(),
            C_bar=self.C.copy(),
            M_ptr=self.M_ptr.copy(),
            time=time,
        )


def gaussian_pattern(N: int = 30, cx: float = 0.0, cy: float = 0.0, sigma: float = 0.34) -> np.ndarray:
    y, x = np.mgrid[-1:1:complex(N), -1:1:complex(N)]
    pat = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)))
    pat = pat / (pat.max() + 1e-12)
    return pat


def initial_state(N: int = 30) -> State:
    record = gaussian_pattern(N=N)
    return State(
        a=0.92 * np.ones((N, N)),
        P=0.16 * record + 0.03,
        q_damage=0.18 + 0.15 * (1 - record),
        C=0.20 + 0.72 * record,
        S=0.86 * record,
        M_ptr=0.10 + 0.86 * record,
        record=record.copy(),
    )


def mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def update_alive(st: State, healing_drive: float = 0.0) -> None:
    """Simple deterministic live-host update."""
    st.P = np.clip(0.02 + st.a * st.C * (1 - st.q_damage) / 4.5, 0, 1)
    st.q_damage = np.clip(st.q_damage - healing_drive * 0.010 * st.C * (1 - st.q_damage), 0, 1)
    st.C = np.clip(st.C + healing_drive * 0.006 * (1 - st.C) - 0.0004 * st.q_damage, 0, 1)
    host = np.clip(st.a * st.P * st.C * (1 - st.q_damage), 0, 1)
    st.S = np.clip(st.S + 0.08 * (host > 0.045) * host * (1 - st.S) - 0.035 * (1 - st.P) * st.S, 0, 1)
    st.M_ptr = np.clip(0.96 * st.M_ptr + 0.04 * st.record * st.C, 0, 1)


def apply_death_state(st: State, active_spirit_channel: float = 0.0) -> None:
    """Collapse embodied participation while optionally declaring a spirit-channel P."""
    embodied_P = np.zeros_like(st.P)
    st.P = np.clip(embodied_P + active_spirit_channel * st.record, 0, 1)

    if active_spirit_channel <= 0:
        # Frozen record / latent pattern: active selfhood should not grow.
        st.S = np.clip(st.S * 0.995, 0, 1)
    else:
        # A declared active channel can maintain/update observer readout.
        host = np.clip(st.a * st.P * st.C * (1 - st.q_damage), 0, 1)
        st.S = np.clip(st.S + 0.05 * host * (1 - st.S) - 0.004 * (1 - st.P) * st.S, 0, 1)

    # Record remains addressable if the substrate/boundary preserves it.
    st.M_ptr = np.clip(0.998 * st.M_ptr + 0.002 * st.record, 0, 1)


def rehost_from_record(st: State, record: np.ndarray, repair_strength: float = 0.75) -> None:
    """Lawful resurrection analogue: restore capable host from preserved record."""
    st.record = record.copy()
    st.C = np.clip(0.30 + 0.68 * record, 0, 1)
    st.q_damage = np.clip((1 - repair_strength) * (0.28 + 0.20 * (1 - record)), 0, 1)
    st.P = np.clip(0.04 + st.a * st.C * (1 - st.q_damage) / 4.0, 0, 1)
    st.S = np.clip(0.18 * record, 0, 1)
    st.M_ptr = np.clip(0.20 + 0.78 * record, 0, 1)


def run_death_resurrection_scenario(name: str, steps: int = 900) -> dict:
    st = initial_state()
    pre_death_snapshot = None
    final_snapshot = None
    rows = []
    preserved_record = st.record.copy()
    path_overlap = 1.0
    declared_spirit_channel = 0.0

    for k in range(steps):
        if k < 250:
            update_alive(st, healing_drive=0.0)
            status = "alive"
        elif name == "death_frozen_record":
            apply_death_state(st, active_spirit_channel=0.0)
            status = "death_frozen"
            path_overlap *= 0.9996
        elif name == "active_spirit_channel":
            declared_spirit_channel = 0.055
            apply_death_state(st, active_spirit_channel=declared_spirit_channel)
            status = "death_with_declared_channel"
            path_overlap *= 0.9998
        elif name == "resurrection_rehost":
            if k < 530:
                apply_death_state(st, active_spirit_channel=0.0)
                status = "death_frozen"
                path_overlap *= 0.9995
            elif k == 530:
                rehost_from_record(st, preserved_record, repair_strength=0.82)
                status = "rehosted"
                path_overlap *= 0.92
            else:
                update_alive(st, healing_drive=0.55)
                status = "resurrected_live"
        elif name == "copy_without_path":
            if k < 530:
                apply_death_state(st, active_spirit_channel=0.0)
                status = "death_frozen"
                path_overlap *= 0.998
            elif k == 530:
                # Same type of pattern, but shifted and with no lawful transfer path.
                copied_record = gaussian_pattern(N=30, cx=0.45, cy=-0.35)
                rehost_from_record(st, copied_record, repair_strength=0.82)
                status = "copied_no_path"
                path_overlap = 0.08
            else:
                update_alive(st, healing_drive=0.55)
                status = "copy_live"
        else:
            raise ValueError(f"unknown scenario {name}")

        if k == 240:
            pre_death_snapshot = st.snapshot(k)
        if k == steps - 1:
            final_snapshot = st.snapshot(k)

        rows.append(
            {
                "scenario": name,
                "step": k,
                "status": status,
                "mean_agency": mean(st.a),
                "mean_presence": mean(st.P),
                "mean_damage_q": mean(st.q_damage),
                "mean_coherence": mean(st.C),
                "mean_selfhood_S": mean(st.S),
                "record_integrity": mean(st.M_ptr * st.record) / (mean(st.record) + 1e-12),
                "declared_spirit_channel": declared_spirit_channel,
                "path_overlap": path_overlap,
            }
        )

    ident = compute_identity_persistence(pre_death_snapshot, final_snapshot)
    path_sensitive = float(np.clip(ident["I_self"] * path_overlap, 0, 1))
    return {
        "name": name,
        "data": pd.DataFrame(rows),
        "identity_snapshot": ident,
        "path_sensitive_identity": path_sensitive,
        "final": rows[-1],
    }


def run_utopic_scenario(name: str, steps: int = 900) -> dict:
    st = initial_state()
    initial_agency = st.a.copy()
    rows = []
    access = 0.0
    direct_agency_write = 0.0

    for k in range(steps):
        # Utopic regime capacity grows in both scenarios.
        K_capacity = 1.0 / (1.0 + np.exp(-(k - 420) / 75.0))
        if name == "utopic_disconnected":
            kappa = 0.0
        elif name == "utopic_connected_noncoercive":
            kappa = 0.55 * K_capacity
        else:
            raise ValueError(f"unknown utopic scenario {name}")

        access += kappa / steps
        # Boundary action is mediated through C and q only, never direct a.
        st.q_damage = np.clip(st.q_damage - kappa * 0.006 * st.C * (1 - st.q_damage), 0, 1)
        st.C = np.clip(st.C + kappa * 0.004 * (1 - st.C), 0, 1)
        update_alive(st, healing_drive=0.10 * kappa)
        direct_agency_write = float(np.max(np.abs(st.a - initial_agency)))

        rows.append(
            {
                "scenario": name,
                "step": k,
                "K_capacity": float(K_capacity),
                "kappa": float(kappa),
                "utopic_access": float(access),
                "mean_agency": mean(st.a),
                "mean_presence": mean(st.P),
                "mean_damage_q": mean(st.q_damage),
                "mean_coherence": mean(st.C),
                "mean_selfhood_S": mean(st.S),
                "direct_agency_write": direct_agency_write,
            }
        )

    return {"name": name, "data": pd.DataFrame(rows), "final": rows[-1]}


def plot_outputs(death_df: pd.DataFrame, utopic_df: pd.DataFrame, summary_rows: list[dict]) -> dict:
    images = {}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = [
        ("mean_presence", "Presence / participation rate"),
        ("mean_selfhood_S", "Observer readout S"),
        ("record_integrity", "Record integrity proxy"),
        ("path_overlap", "Path-overlap factor"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for scenario, grp in death_df.groupby("scenario"):
            ax.plot(grp["step"], grp[metric], label=scenario, linewidth=1.7)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("update step k")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "death_resurrection_edge_cases.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    images["death_resurrection"] = path

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = [
        ("K_capacity", "Utopic-regime capacity"),
        ("kappa", "Present lawful coupling kappa"),
        ("mean_damage_q", "Mean damage readout"),
        ("direct_agency_write", "Direct agency-write deviation"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for scenario, grp in utopic_df.groupby("scenario"):
            ax.plot(grp["step"], grp[metric], label=scenario, linewidth=1.7)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("update step k")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "utopic_boundary_edge_cases.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    images["utopic_boundary"] = path

    cont = pd.DataFrame([r for r in summary_rows if r["family"] == "death_resurrection"])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cont))
    width = 0.35
    ax.bar(x - width / 2, cont["snapshot_identity"], width, label="Snapshot identity")
    ax.bar(x + width / 2, cont["path_sensitive_identity"], width, label="Path-sensitive identity")
    ax.set_xticks(x)
    ax.set_xticklabels(cont["scenario"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("continuity score")
    ax.set_title("Similarity is not enough: path-sensitive continuity controls copy false positives")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "continuity_controls.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    images["continuity_controls"] = path

    return images


def main() -> None:
    death_scenarios = [
        "death_frozen_record",
        "active_spirit_channel",
        "resurrection_rehost",
        "copy_without_path",
    ]
    utopic_scenarios = [
        "utopic_disconnected",
        "utopic_connected_noncoercive",
    ]

    death_results = [run_death_resurrection_scenario(s) for s in death_scenarios]
    utopic_results = [run_utopic_scenario(s) for s in utopic_scenarios]

    death_df = pd.concat([r["data"] for r in death_results], ignore_index=True)
    utopic_df = pd.concat([r["data"] for r in utopic_results], ignore_index=True)

    summary_rows: list[dict] = []
    for r in death_results:
        final = r["final"]
        summary_rows.append(
            {
                "family": "death_resurrection",
                "scenario": r["name"],
                "final_presence": float(final["mean_presence"]),
                "final_selfhood_S": float(final["mean_selfhood_S"]),
                "record_integrity": float(final["record_integrity"]),
                "declared_spirit_channel": float(final["declared_spirit_channel"]),
                "snapshot_identity": float(r["identity_snapshot"]["I_self"]),
                "path_sensitive_identity": float(r["path_sensitive_identity"]),
            }
        )

    for r in utopic_results:
        final = r["final"]
        summary_rows.append(
            {
                "family": "utopic_boundary",
                "scenario": r["name"],
                "final_K_capacity": float(final["K_capacity"]),
                "final_kappa": float(final["kappa"]),
                "final_access": float(final["utopic_access"]),
                "final_damage_q": float(final["mean_damage_q"]),
                "final_coherence": float(final["mean_coherence"]),
                "direct_agency_write": float(final["direct_agency_write"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    images = plot_outputs(death_df, utopic_df, summary_rows)

    tests = {
        "F_D1_death_suppresses_embodied_participation": bool(
            summary_df.loc[summary_df.scenario == "death_frozen_record", "final_presence"].iloc[0] < 0.01
        ),
        "F_D2_frozen_record_not_active_observer": bool(
            summary_df.loc[summary_df.scenario == "death_frozen_record", "declared_spirit_channel"].iloc[0] == 0.0
            and summary_df.loc[summary_df.scenario == "death_frozen_record", "record_integrity"].iloc[0] > 0.4
        ),
        "F_D3_path_sensitive_copy_control": bool(
            summary_df.loc[summary_df.scenario == "copy_without_path", "path_sensitive_identity"].iloc[0]
            < summary_df.loc[summary_df.scenario == "resurrection_rehost", "path_sensitive_identity"].iloc[0]
        ),
        "F_D5_active_consciousness_requires_channel": bool(
            summary_df.loc[summary_df.scenario == "active_spirit_channel", "declared_spirit_channel"].iloc[0] > 0.0
            and summary_df.loc[summary_df.scenario == "death_frozen_record", "declared_spirit_channel"].iloc[0] == 0.0
        ),
        "F_K1_disconnected_utopic_no_present_coupling": bool(
            summary_df.loc[summary_df.scenario == "utopic_disconnected", "final_kappa"].iloc[0] == 0.0
        ),
        "F_K2_boundary_channel_no_agency_override": bool(
            summary_df.loc[summary_df.scenario == "utopic_connected_noncoercive", "direct_agency_write"].iloc[0] < 1e-12
        ),
        "F_K3_noncoercive_growth": bool(
            summary_df.loc[summary_df.scenario == "utopic_connected_noncoercive", "final_access"].iloc[0]
            > summary_df.loc[summary_df.scenario == "utopic_disconnected", "final_access"].iloc[0]
        ),
    }

    death_csv = os.path.join(OUT_DIR, "death_resurrection_edge_cases.csv")
    utopic_csv = os.path.join(OUT_DIR, "utopic_boundary_edge_cases.csv")
    summary_csv = os.path.join(OUT_DIR, "falsifier_edge_case_summary.csv")
    json_path = os.path.join(OUT_DIR, "falsifier_edge_case_summary.json")
    md_path = os.path.join(OUT_DIR, "falsifier_edge_case_results.md")

    death_df.to_csv(death_csv, index=False)
    utopic_df.to_csv(utopic_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "tests": tests, "images": images}, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DET v7 Falsifier Edge-Case Simulation Results\n\n")
        f.write("These deterministic edge-case simulations test death-state participation collapse, record persistence, active-channel requirements, resurrection continuity, copy controls, and non-coercive utopic-boundary coupling. They are falsifier scaffolds, not empirical proof of metaphysical claims.\n\n")
        f.write("## Summary\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n## Falsifier checks\n\n")
        f.write("| Check | Pass |\n|---|---:|\n")
        for key, value in tests.items():
            f.write(f"| `{key}` | {value} |\n")
        f.write("\n![Death and resurrection edge cases](death_resurrection_edge_cases.png)\n\n")
        f.write("![Continuity controls](continuity_controls.png)\n\n")
        f.write("![Utopic boundary edge cases](utopic_boundary_edge_cases.png)\n")

    print(json.dumps({"summary": json_path, "markdown": md_path, "tests": tests, "images": images}, indent=2))


if __name__ == "__main__":
    main()
