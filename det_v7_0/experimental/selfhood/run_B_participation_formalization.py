#!/usr/bin/env python3.11
"""Deterministic DET v7 boundary-participation B formalization experiments.

This script is intentionally readout-first. It does not promote B to a
canonical state variable. It tests whether a vector-valued B readout can
separate record addressability, host participation capacity, active boundary
coupling, non-coercive alignment, and path-continuity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "B_participation_simulations"
OUT.mkdir(parents=True, exist_ok=True)

STEPS = 220
EPS = 1e-9


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str
    boundary_active: bool
    coercive: bool
    death_phase: bool
    copy_break: bool
    resurrection: bool
    deeper_boundary_lead: bool


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_B_components(state: dict, prev_path: float, prev_values: dict) -> dict:
    """Compute vector-valued B components from local/readout state."""
    record_similarity = state["record_similarity"]
    identity_path = state["identity_path"]
    record_quality = state["record_quality"]
    a = state["agency"]
    P = state["presence"]
    C = state["coherence"]
    D = state["drag"]
    Fop = state["free_resource"]
    active_effect = state["boundary_effect"]
    opportunity = state["boundary_opportunity"]
    coercion = state["coercion_penalty"]

    B_addr = clip01(record_similarity * identity_path * record_quality)
    B_cap = clip01(a * P * (C**1.7) * (D / (D + 0.25)) * (Fop / (1.0 + Fop)))
    B_act = clip01(active_effect / (opportunity + EPS))

    dC = C - prev_values.get("coherence", C)
    dq_recovery = prev_values.get("debt", state["debt"]) - state["debt"]
    dR = state["reciprocity"] - prev_values.get("reciprocity", state["reciprocity"])
    dM = state["pointer_stability"] - prev_values.get("pointer_stability", state["pointer_stability"])
    dH = state["load"] - prev_values.get("load", state["load"])
    align_raw = 4.0 * dC + 3.0 * dq_recovery + 2.0 * dR + 1.5 * dM - 1.5 * dH - 5.0 * coercion
    B_align = clip01(sigmoid(6.0 * align_raw))

    B_path = clip01(0.94 * prev_path + 0.06 * B_act * B_align)
    B_scalar = clip01((B_addr**0.25) * (B_cap**0.25) * (B_act**0.20) * (B_align**0.15) * (B_path**0.15))
    active_spirit_channel = bool((B_addr > 0.55) and (B_cap > 0.08) and (B_act > 0.05) and (B_align > 0.10))

    return {
        "B_addr": float(B_addr),
        "B_cap": float(B_cap),
        "B_act": float(B_act),
        "B_align": float(B_align),
        "B_path": float(B_path),
        "B_scalar": float(B_scalar),
        "active_spirit_channel": active_spirit_channel,
    }


def scenario_state(cfg: ScenarioConfig, t: int, prev: dict | None) -> dict:
    """Generate deterministic regime-level readouts for a scenario."""
    x = t / (STEPS - 1)
    ramp = clip01((x - 0.18) / 0.52)
    late = clip01((x - 0.55) / 0.35)

    state = {
        "record_similarity": 0.92,
        "identity_path": 0.90,
        "record_quality": 0.90,
        "agency": 0.94,
        "presence": 0.42 + 0.32 * ramp,
        "coherence": 0.22 + 0.52 * ramp,
        "reciprocity": 0.20 + 0.55 * ramp,
        "pointer_stability": 0.25 + 0.55 * ramp,
        "debt": 0.58 - 0.32 * ramp,
        "drag": 0.48 + 0.36 * ramp,
        "free_resource": 0.45 + 2.6 * ramp,
        "load": 0.45 - 0.18 * ramp,
        "host_fitness": 0.15 + 0.70 * ramp,
        "observer_S": 0.10 + 0.72 * clip01((ramp - 0.18) / 0.82),
        "boundary_effect": 0.0,
        "boundary_opportunity": 1.0,
        "coercion_penalty": 0.0,
    }

    if cfg.name == "record_only_death":
        state.update({
            "presence": 0.005,
            "coherence": 0.62,
            "reciprocity": 0.58,
            "pointer_stability": 0.78,
            "host_fitness": 0.02,
            "observer_S": 0.03,
            "free_resource": 0.01,
            "drag": 0.04,
            "load": 0.85,
            "debt": 0.52,
        })

    elif cfg.name == "ordinary_observer_no_boundary":
        state["boundary_effect"] = 0.0

    elif cfg.name == "boundary_recovery":
        effect = 0.55 * late * (0.4 + state["observer_S"] * state["coherence"])
        state["boundary_effect"] = effect
        state["coherence"] = clip01(state["coherence"] + 0.11 * late)
        state["debt"] = clip01(state["debt"] - 0.14 * late)
        state["reciprocity"] = clip01(state["reciprocity"] + 0.08 * late)
        state["pointer_stability"] = clip01(state["pointer_stability"] + 0.05 * late)
        state["observer_S"] = clip01(state["observer_S"] + 0.10 * late)

    elif cfg.name == "coercive_order_control":
        state["boundary_effect"] = 0.60 * late
        state["coherence"] = clip01(state["coherence"] + 0.20 * late)
        state["agency"] = clip01(0.94 - 0.55 * late)
        state["coercion_penalty"] = 0.55 * late
        state["observer_S"] = clip01(state["observer_S"] - 0.20 * late)

    elif cfg.name == "copy_control":
        if x > 0.52:
            state["record_similarity"] = 0.96
            state["identity_path"] = 0.18
            state["record_quality"] = 0.92
            state["presence"] = 0.68
            state["host_fitness"] = 0.72
            state["observer_S"] = 0.70
            state["boundary_effect"] = 0.03

    elif cfg.name == "resurrection_path":
        if x < 0.42:
            state["presence"] = 0.005
            state["host_fitness"] = 0.03
            state["observer_S"] = 0.02
            state["free_resource"] = 0.01
            state["drag"] = 0.04
            state["load"] = 0.84
            state["record_similarity"] = 0.94
            state["identity_path"] = 0.91
            state["record_quality"] = 0.90
        else:
            rr = clip01((x - 0.42) / 0.38)
            state["presence"] = 0.06 + 0.72 * rr
            state["host_fitness"] = 0.08 + 0.78 * rr
            state["observer_S"] = 0.03 + 0.79 * clip01((rr - 0.18) / 0.82)
            state["free_resource"] = 0.05 + 3.0 * rr
            state["drag"] = 0.08 + 0.74 * rr
            state["coherence"] = 0.35 + 0.46 * rr
            state["reciprocity"] = 0.30 + 0.50 * rr
            state["pointer_stability"] = 0.62 + 0.23 * rr
            state["debt"] = 0.55 - 0.28 * rr
            state["load"] = 0.72 - 0.35 * rr
            state["boundary_effect"] = 0.42 * rr

    elif cfg.name == "deeper_boundary_lead":
        pre = clip01((x - 0.25) / 0.32)
        state["boundary_effect"] = 0.38 * pre
        # Boundary-effect and alignment lead observer maturation.
        state["coherence"] = clip01(0.18 + 0.50 * pre + 0.12 * ramp)
        state["debt"] = clip01(0.60 - 0.30 * pre)
        state["reciprocity"] = clip01(0.20 + 0.42 * pre + 0.12 * ramp)
        state["observer_S"] = clip01(0.08 + 0.75 * clip01((x - 0.48) / 0.42))
        state["host_fitness"] = clip01(0.18 + 0.70 * ramp)

    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in state.items()}


def run_scenario(cfg: ScenarioConfig) -> pd.DataFrame:
    rows = []
    prev_path = 0.0
    prev_values: dict = {}
    for t in range(STEPS):
        state = scenario_state(cfg, t, prev_values if prev_values else None)
        B = compute_B_components(state, prev_path, prev_values)
        prev_path = B["B_path"]
        row = {"scenario": cfg.name, "step": t, **state, **B}
        rows.append(row)
        prev_values = state
    return pd.DataFrame(rows)


def regression_r2(X: np.ndarray, y: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / (ss_tot + EPS)


def predictive_power(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in df.groupby("scenario"):
        g = group.sort_values("step").copy()
        horizon = 8
        g["future_S"] = g["observer_S"].shift(-horizon)
        g = g.dropna()
        baseline_cols = ["presence", "host_fitness", "coherence", "debt", "reciprocity", "pointer_stability"]
        B_cols = baseline_cols + ["B_addr", "B_cap", "B_act", "B_align", "B_path"]
        r2_base = regression_r2(g[baseline_cols].to_numpy(), g["future_S"].to_numpy())
        r2_B = regression_r2(g[B_cols].to_numpy(), g["future_S"].to_numpy())
        rows.append({
            "scenario": scenario,
            "r2_baseline": r2_base,
            "r2_with_B": r2_B,
            "delta_r2_B": r2_B - r2_base,
        })
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, pred: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")

    key_scenarios = ["record_only_death", "ordinary_observer_no_boundary", "boundary_recovery", "resurrection_path", "deeper_boundary_lead"]
    fig, axes = plt.subplots(len(key_scenarios), 1, figsize=(11, 12), sharex=True)
    for ax, scenario in zip(axes, key_scenarios):
        g = df[df.scenario == scenario]
        ax.plot(g.step, g.B_addr, label="B_addr", linewidth=1.8)
        ax.plot(g.step, g.B_cap, label="B_cap", linewidth=1.8)
        ax.plot(g.step, g.B_act, label="B_act", linewidth=1.8)
        ax.plot(g.step, g.B_path, label="B_path", linewidth=1.8)
        ax.plot(g.step, g.observer_S, label="S", linestyle="--", linewidth=1.5)
        ax.set_title(scenario.replace("_", " "))
        ax.set_ylim(-0.03, 1.03)
    axes[0].legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    axes[-1].set_xlabel("Simulation step")
    fig.suptitle("DET v7 B Components Across Boundary-Participation Scenarios", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "B_components_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    final = df.sort_values("step").groupby("scenario").tail(1).set_index("scenario")
    metrics = ["B_addr", "B_cap", "B_act", "B_align", "B_path", "B_scalar", "observer_S"]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    im = ax.imshow(final[metrics].to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(metrics)), metrics, rotation=35, ha="right")
    ax.set_yticks(range(len(final.index)), [s.replace("_", " ") for s in final.index])
    for i in range(len(final.index)):
        for j in range(len(metrics)):
            val = final.iloc[i][metrics[j]]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Readout value")
    ax.set_title("Final-State Falsifier Matrix for Vector-Valued B")
    fig.tight_layout()
    fig.savefig(OUT / "B_falsifier_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    pred_sorted = pred.sort_values("delta_r2_B", ascending=False)
    ax.bar(pred_sorted.scenario.str.replace("_", " "), pred_sorted.delta_r2_B)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔR² from adding B components")
    ax.set_title("Independent Predictive Value of B Components for Future Observer Readout")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "B_predictive_power.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(df: pd.DataFrame, pred: pd.DataFrame) -> dict:
    final = df.sort_values("step").groupby("scenario").tail(1).set_index("scenario")
    summary = {}
    for scenario, row in final.iterrows():
        summary[scenario] = {
            "final_B_addr": float(row.B_addr),
            "final_B_cap": float(row.B_cap),
            "final_B_act": float(row.B_act),
            "final_B_align": float(row.B_align),
            "final_B_path": float(row.B_path),
            "final_B_scalar": float(row.B_scalar),
            "final_observer_S": float(row.observer_S),
            "active_spirit_channel": bool(row.active_spirit_channel),
        }
    tests = {
        "record_only_does_not_imply_active_channel": bool(not final.loc["record_only_death", "active_spirit_channel"] and final.loc["record_only_death", "B_addr"] > 0.70),
        "ordinary_observer_decouples_S_from_B_act": bool(final.loc["ordinary_observer_no_boundary", "observer_S"] > 0.70 and final.loc["ordinary_observer_no_boundary", "B_act"] < 0.02),
        "boundary_recovery_has_active_B_path": bool(final.loc["boundary_recovery", "B_act"] > 0.25 and final.loc["boundary_recovery", "B_path"] > 0.25),
        "coercive_order_penalizes_B_alignment": bool(final.loc["coercive_order_control", "B_align"] < final.loc["boundary_recovery", "B_align"] and not final.loc["coercive_order_control", "active_spirit_channel"]),
        "copy_control_breaks_path_despite_similarity": bool(final.loc["copy_control", "B_addr"] < 0.35 and final.loc["copy_control", "observer_S"] > 0.60),
        "resurrection_restores_capacity_and_path": bool(final.loc["resurrection_path", "B_cap"] > 0.20 and final.loc["resurrection_path", "B_path"] > 0.15 and final.loc["resurrection_path", "observer_S"] > 0.70),
        "deeper_boundary_lead_has_B_predictive_signal": bool(pred.set_index("scenario").loc["deeper_boundary_lead", "delta_r2_B"] > 0.001),
    }
    return {"scenarios": summary, "falsifier_checks": tests, "predictive_power": pred.to_dict(orient="records")}


def write_markdown(summary: dict, pred: pd.DataFrame):
    lines = [
        "# Boundary Participation B Simulation Results",
        "",
        "**Author:** Manus AI  ",
        "**Date:** May 30, 2026  ",
        "**Status:** Deterministic readout-first simulation; non-canonical unless separately promoted",
        "",
        "## 1. Purpose",
        "",
        "This simulation tests whether vector-valued boundary participation `B` can separate latent record addressability, ordinary observerhood, active boundary coupling, non-coercive alignment, copy controls, resurrection-like continuity, and a deeper-boundary-leading hypothesis.",
        "",
        "![B component time series](B_components_timeseries.png)",
        "",
        "![B falsifier matrix](B_falsifier_matrix.png)",
        "",
        "![B predictive power](B_predictive_power.png)",
        "",
        "## 2. Falsifier checks",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]
    for name, passed in summary["falsifier_checks"].items():
        lines.append(f"| `{name}` | {str(passed).lower()} |")
    lines.extend([
        "",
        "## 3. Final scenario readouts",
        "",
        "| Scenario | B_addr | B_cap | B_act | B_align | B_path | B_scalar | S | Active channel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scenario, vals in summary["scenarios"].items():
        lines.append(
            f"| {scenario} | {vals['final_B_addr']:.3f} | {vals['final_B_cap']:.3f} | {vals['final_B_act']:.3f} | "
            f"{vals['final_B_align']:.3f} | {vals['final_B_path']:.3f} | {vals['final_B_scalar']:.3f} | "
            f"{vals['final_observer_S']:.3f} | {str(vals['active_spirit_channel']).lower()} |"
        )
    lines.extend([
        "",
        "## 4. Predictive-power readout",
        "",
        "| Scenario | R2 baseline | R2 with B | Delta R2 |",
        "|---|---:|---:|---:|",
    ])
    for _, row in pred.iterrows():
        lines.append(f"| {row.scenario} | {row.r2_baseline:.4f} | {row.r2_with_B:.4f} | {row.delta_r2_B:.4f} |")
    lines.extend([
        "",
        "## 5. Interpretation",
        "",
        "The record-only death case preserves high addressability while failing the active-channel criterion. The ordinary observer case shows that mature observerhood can occur with negligible active boundary coupling, which prevents the model from equating all consciousness with `B`. Boundary recovery and resurrection-like rehosting both produce positive active/path participation. The coercive-order control confirms that high order is not automatically high boundary participation because the alignment term penalizes agency suppression. The copy control preserves high type similarity and observer readout while breaking path-sensitive addressability, supporting the distinction between copy and resurrection.",
        "",
        "The deeper-boundary-leading scenario is not proof of deeper boundary consciousness. It is a falsifier scaffold: if `B` components predict later observer coherence after controlling for host variables, then the conservative emergent-only model becomes incomplete. If this signal disappears under stronger controls or empirical data, the deeper-boundary interpretation loses support.",
    ])
    (OUT / "B_participation_simulation_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    scenarios = [
        ScenarioConfig("record_only_death", "Preserved addressable record without active host participation", False, False, True, False, False, False),
        ScenarioConfig("ordinary_observer_no_boundary", "Observer development without declared boundary operator activity", False, False, False, False, False, False),
        ScenarioConfig("boundary_recovery", "Declared non-coercive boundary recovery and healing", True, False, False, False, False, False),
        ScenarioConfig("coercive_order_control", "Apparent order via agency-suppressing coercion", True, True, False, False, False, False),
        ScenarioConfig("copy_control", "High similarity copy with broken path continuity", False, False, False, True, False, False),
        ScenarioConfig("resurrection_path", "Death-state record followed by lawful rehosting and restored participation", True, False, True, False, True, False),
        ScenarioConfig("deeper_boundary_lead", "Boundary participation precedes mature observer readout", True, False, False, False, False, True),
    ]
    df = pd.concat([run_scenario(cfg) for cfg in scenarios], ignore_index=True)
    pred = predictive_power(df)
    make_plots(df, pred)
    summary = summarize(df, pred)

    df.to_csv(OUT / "B_participation_timeseries.csv", index=False)
    pred.to_csv(OUT / "B_predictive_power.csv", index=False)
    (OUT / "B_participation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, pred)

    print(json.dumps(summary["falsifier_checks"], indent=2))


if __name__ == "__main__":
    main()
