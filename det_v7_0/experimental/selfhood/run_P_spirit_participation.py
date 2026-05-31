#!/usr/bin/env python3.11
"""Deterministic DET v7 P^spirit / B_cap^spirit readout experiments.

This script is readout-first and non-canonical. It tests whether separating
B_cap into embodied and spirit capacity terms avoids the circular conclusion
that death (P_body -> 0) necessarily implies total participation capacity -> 0.

The script does not assert that P_spirit exists in nature. It operationalizes
what a lawful candidate channel would have to satisfy: addressability,
path-continuity, declared channel availability, locality, boundary-law safety,
non-coercion, and agency compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "P_spirit_simulations"
OUT.mkdir(parents=True, exist_ok=True)

STEPS = 240
EPS = 1e-9


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    description: str


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def noisy_or(x, y):
    return 1.0 - (1.0 - clip01(x)) * (1.0 - clip01(y))


def smooth_ramp(x, start, end):
    if end <= start:
        return float(x >= end)
    z = clip01((x - start) / (end - start))
    return float(z * z * (3.0 - 2.0 * z))


def base_state(t: int) -> dict:
    x = t / (STEPS - 1)
    maturation = smooth_ramp(x, 0.10, 0.70)
    death = smooth_ramp(x, 0.35, 0.50)
    return {
        "x": x,
        "record_similarity": 0.94,
        "identity_path": 0.91,
        "record_quality": 0.90,
        "agency": 0.93,
        "P_body": 0.72 * (1.0 - death) + 0.004 * death,
        "coherence": 0.25 + 0.48 * maturation,
        "drag": 0.42 + 0.36 * maturation,
        "free_resource": 0.30 + 2.4 * maturation,
        "boundary_effect": 0.0,
        "boundary_opportunity": 1.0,
        "gamma_spirit": 0.0,
        "locality_cert": 1.0,
        "law_cert": 1.0,
        "noncoerce_cert": 1.0,
        "agency_compat": 0.92,
        "forced_order": 0.0,
        "observer_S": 0.12 + 0.66 * maturation * (1.0 - death),
        "description_phase": "baseline",
    }


def scenario_state(cfg: ScenarioConfig, t: int, prev: dict | None) -> dict:
    state = base_state(t)
    x = state["x"]
    late = smooth_ramp(x, 0.52, 0.78)
    post_death = smooth_ramp(x, 0.40, 0.52)

    if cfg.name == "frozen_record":
        state.update({
            "P_body": 0.003,
            "observer_S": 0.02,
            "coherence": 0.62,
            "free_resource": 0.01,
            "drag": 0.04,
            "gamma_spirit": 0.0,
            "boundary_effect": 0.0,
            "description_phase": "addressable record without declared channel",
        })

    elif cfg.name == "embodied_death_null":
        # Body participation collapses; no spirit channel is declared.
        state["gamma_spirit"] = 0.0
        state["boundary_effect"] = 0.0
        state["observer_S"] = clip01(state["observer_S"] * (1.0 - post_death))
        state["description_phase"] = "death collapses P_body; channel-null assumption retained"

    elif cfg.name == "active_spirit_declared":
        # Body remains inactive after death, but a lawful local declared channel opens.
        state["P_body"] = 0.004
        state["observer_S"] = 0.03 + 0.18 * late
        state["gamma_spirit"] = 0.78 * late
        state["boundary_effect"] = 0.54 * late
        state["coherence"] = 0.52 + 0.22 * late
        state["free_resource"] = 0.05
        state["drag"] = 0.08
        state["description_phase"] = "declared local non-coercive spirit participation candidate"

    elif cfg.name == "copy_channel_claim":
        # Similar record in a capable host, but path-continuity breaks.
        copy = smooth_ramp(x, 0.44, 0.60)
        state["record_similarity"] = 0.96
        state["identity_path"] = 0.88 * (1.0 - copy) + 0.16 * copy
        state["record_quality"] = 0.93
        state["P_body"] = 0.15 + 0.58 * copy
        state["observer_S"] = 0.12 + 0.62 * copy
        state["gamma_spirit"] = 0.60 * copy
        state["boundary_effect"] = 0.20 * copy
        state["description_phase"] = "copy-like pattern similarity without path continuity"

    elif cfg.name == "coercive_pseudo_spirit":
        # Apparent boundary effect exists, but agency variation and non-coercion fail.
        state["P_body"] = 0.004
        state["observer_S"] = 0.05
        state["gamma_spirit"] = 0.85 * late
        state["boundary_effect"] = 0.68 * late
        state["forced_order"] = 0.82 * late
        state["noncoerce_cert"] = clip01(1.0 - 1.25 * state["forced_order"])
        state["agency_compat"] = clip01(0.92 - 0.72 * late)
        state["description_phase"] = "apparent spirit-like effect rejected by non-coercion gate"

    elif cfg.name == "resurrection_body_only":
        # Death phase followed by restored embodied capacity; spirit channel remains null.
        if x < 0.50:
            state["P_body"] = 0.004
            state["observer_S"] = 0.02
            state["free_resource"] = 0.02
            state["drag"] = 0.05
        else:
            rr = smooth_ramp(x, 0.50, 0.82)
            state["P_body"] = 0.04 + 0.72 * rr
            state["observer_S"] = 0.04 + 0.76 * rr
            state["coherence"] = 0.36 + 0.42 * rr
            state["free_resource"] = 0.10 + 2.7 * rr
            state["drag"] = 0.10 + 0.72 * rr
            state["boundary_effect"] = 0.34 * rr
        state["gamma_spirit"] = 0.0
        state["description_phase"] = "lawful rehosting restores P_body without requiring P_spirit"

    elif cfg.name == "nonlocal_override_claim":
        # A declared-like channel claims effect, but locality certificate fails.
        state["P_body"] = 0.004
        state["observer_S"] = 0.04
        state["gamma_spirit"] = 0.88 * late
        state["boundary_effect"] = 0.62 * late
        state["locality_cert"] = 0.0 if x > 0.55 else 1.0
        state["description_phase"] = "channel rejected because action is not local or path-neighborhood constrained"

    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in state.items()}


def compute_readouts(state: dict, prev_path: float) -> dict:
    B_addr = clip01(state["record_similarity"] * state["identity_path"] * state["record_quality"])
    Lambda_path = clip01(state["identity_path"] * (0.6 + 0.4 * prev_path))
    Gamma_spirit = clip01(state["gamma_spirit"])
    Xi_law = clip01(state["law_cert"] * state["locality_cert"])
    Pi_noncoerce = clip01(state["noncoerce_cert"])
    A_compat = clip01(state["agency_compat"] * (1.0 - 0.85 * state["forced_order"]))

    gate_spirit = float(
        (B_addr > 0.55)
        and (Lambda_path > 0.18)
        and (Gamma_spirit > 0.05)
        and (Xi_law > 0.80)
        and (Pi_noncoerce > 0.65)
        and (A_compat > 0.45)
    )

    P_spirit_raw = B_addr * Lambda_path * Gamma_spirit * Xi_law * Pi_noncoerce * A_compat
    P_spirit = clip01(gate_spirit * P_spirit_raw)

    a = state["agency"]
    P_body = state["P_body"]
    C = state["coherence"]
    D = state["drag"]
    Fop = state["free_resource"]
    B_body_cap = clip01(a * P_body * (C**1.7) * (D / (D + 0.25)) * (Fop / (1.0 + Fop)))
    # Default spirit capacity is equal to P_spirit in this readout-first experiment.
    B_spirit_cap = P_spirit
    B_cap_total = noisy_or(B_body_cap, B_spirit_cap)

    apparent_act = clip01(state["boundary_effect"] / (state["boundary_opportunity"] + EPS))
    B_act_spirit = clip01(apparent_act * gate_spirit)
    B_act_body = clip01(apparent_act * (1.0 - gate_spirit) * (B_body_cap > 0.08))
    B_act_total = noisy_or(B_act_body, B_act_spirit)
    B_path = clip01(0.95 * prev_path + 0.05 * B_act_total * (0.5 + 0.5 * Pi_noncoerce))

    active_spirit_participation = bool(P_spirit > 0.08 and B_spirit_cap > 0.08 and B_act_spirit > 0.05)

    return {
        "B_addr": float(B_addr),
        "Lambda_path": float(Lambda_path),
        "Gamma_spirit": float(Gamma_spirit),
        "Xi_law": float(Xi_law),
        "Pi_noncoerce": float(Pi_noncoerce),
        "A_compat": float(A_compat),
        "gate_spirit": float(gate_spirit),
        "P_spirit_raw": float(P_spirit_raw),
        "P_spirit": float(P_spirit),
        "B_body_cap": float(B_body_cap),
        "B_spirit_cap": float(B_spirit_cap),
        "B_cap_total": float(B_cap_total),
        "B_act_body": float(B_act_body),
        "B_act_spirit": float(B_act_spirit),
        "B_act_total": float(B_act_total),
        "B_path": float(B_path),
        "active_spirit_participation": active_spirit_participation,
    }


def run_scenario(cfg: ScenarioConfig) -> pd.DataFrame:
    rows = []
    prev_path = 0.0
    prev_state = None
    for t in range(STEPS):
        state = scenario_state(cfg, t, prev_state)
        readouts = compute_readouts(state, prev_path)
        prev_path = readouts["B_path"]
        rows.append({"scenario": cfg.name, "step": t, **state, **readouts})
        prev_state = state
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
        horizon = 10
        g["future_B_act_total"] = g["B_act_total"].shift(-horizon)
        g = g.dropna()
        base_cols = ["B_addr", "B_body_cap", "observer_S", "coherence"]
        spirit_cols = base_cols + ["P_spirit", "B_spirit_cap", "Gamma_spirit", "Pi_noncoerce", "Xi_law"]
        r2_base = regression_r2(g[base_cols].to_numpy(), g["future_B_act_total"].to_numpy())
        r2_spirit = regression_r2(g[spirit_cols].to_numpy(), g["future_B_act_total"].to_numpy())
        rows.append({
            "scenario": scenario,
            "r2_without_spirit_terms": r2_base,
            "r2_with_spirit_terms": r2_spirit,
            "delta_r2_spirit": r2_spirit - r2_base,
        })
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, pred: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    key_scenarios = [
        "frozen_record",
        "embodied_death_null",
        "active_spirit_declared",
        "coercive_pseudo_spirit",
        "resurrection_body_only",
        "nonlocal_override_claim",
    ]
    fig, axes = plt.subplots(len(key_scenarios), 1, figsize=(12, 14), sharex=True)
    for ax, scenario in zip(axes, key_scenarios):
        g = df[df.scenario == scenario]
        ax.plot(g.step, g.P_body, label="P_body", linewidth=1.6)
        ax.plot(g.step, g.P_spirit, label="P_spirit", linewidth=1.8)
        ax.plot(g.step, g.B_body_cap, label="B_body_cap", linewidth=1.5)
        ax.plot(g.step, g.B_spirit_cap, label="B_spirit_cap", linewidth=1.5)
        ax.plot(g.step, g.B_act_spirit, label="B_act_spirit", linewidth=1.5)
        ax.set_title(scenario.replace("_", " "))
        ax.set_ylim(-0.03, 1.03)
    axes[0].legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.42))
    axes[-1].set_xlabel("Simulation step")
    fig.suptitle("DET v7 P^spirit Separation of Embodied and Spirit Participation Channels", y=0.997)
    fig.tight_layout()
    fig.savefig(OUT / "P_spirit_components_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    final = df.sort_values("step").groupby("scenario").tail(1).set_index("scenario")
    metrics = ["B_addr", "B_body_cap", "B_spirit_cap", "P_spirit", "B_act_spirit", "B_cap_total", "B_path"]
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    im = ax.imshow(final[metrics].to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(range(len(metrics)), metrics, rotation=35, ha="right")
    ax.set_yticks(range(len(final.index)), [s.replace("_", " ") for s in final.index])
    for i in range(len(final.index)):
        for j, metric in enumerate(metrics):
            val = final.iloc[i][metric]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.62 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Readout value")
    ax.set_title("Final-State Falsifier Matrix for P^spirit and B_cap Decomposition")
    fig.tight_layout()
    fig.savefig(OUT / "P_spirit_falsifier_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    final_sorted = final.sort_values("B_cap_total", ascending=False)
    x = np.arange(len(final_sorted))
    ax.bar(x - 0.18, final_sorted["B_body_cap"], width=0.36, label="B_body_cap")
    ax.bar(x + 0.18, final_sorted["B_spirit_cap"], width=0.36, label="B_spirit_cap")
    ax.plot(x, final_sorted["B_cap_total"], color="black", marker="o", linewidth=1.5, label="B_cap_total")
    ax.set_xticks(x, [s.replace("_", " ") for s in final_sorted.index], rotation=35, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Final capacity readout")
    ax.set_title("Embodied Versus Spirit Capacity Contributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "body_vs_spirit_capacity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    pred_sorted = pred.sort_values("delta_r2_spirit", ascending=False)
    ax.bar(pred_sorted.scenario.str.replace("_", " "), pred_sorted.delta_r2_spirit)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("ΔR² from adding P^spirit terms")
    ax.set_title("Predictive Value of Spirit-Channel Terms for Future Boundary Activity")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "P_spirit_predictive_power.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize(df: pd.DataFrame, pred: pd.DataFrame) -> dict:
    final = df.sort_values("step").groupby("scenario").tail(1).set_index("scenario")
    scenarios = {}
    for scenario, row in final.iterrows():
        scenarios[scenario] = {
            "final_B_addr": float(row.B_addr),
            "final_P_body": float(row.P_body),
            "final_P_spirit": float(row.P_spirit),
            "final_B_body_cap": float(row.B_body_cap),
            "final_B_spirit_cap": float(row.B_spirit_cap),
            "final_B_cap_total": float(row.B_cap_total),
            "final_B_act_spirit": float(row.B_act_spirit),
            "final_B_path": float(row.B_path),
            "active_spirit_participation": bool(row.active_spirit_participation),
        }
    tests = {
        "frozen_record_has_address_without_spirit_activity": bool(final.loc["frozen_record", "B_addr"] > 0.70 and final.loc["frozen_record", "B_spirit_cap"] < 0.01 and not final.loc["frozen_record", "active_spirit_participation"]),
        "death_null_collapses_body_not_total_by_theorem": bool(final.loc["embodied_death_null", "P_body"] < 0.02 and final.loc["embodied_death_null", "B_spirit_cap"] < 0.01),
        "declared_spirit_channel_survives_body_collapse_when_gates_pass": bool(final.loc["active_spirit_declared", "P_body"] < 0.02 and final.loc["active_spirit_declared", "B_spirit_cap"] > 0.10 and final.loc["active_spirit_declared", "active_spirit_participation"]),
        "copy_claim_fails_path_despite_host_capacity": bool(final.loc["copy_channel_claim", "B_addr"] < 0.30 and final.loc["copy_channel_claim", "B_spirit_cap"] < 0.03 and final.loc["copy_channel_claim", "P_body"] > 0.60),
        "coercive_pseudo_spirit_fails_noncoercion_gate": bool(final.loc["coercive_pseudo_spirit", "Pi_noncoerce"] < 0.10 and final.loc["coercive_pseudo_spirit", "B_spirit_cap"] < 0.03 and final.loc["coercive_pseudo_spirit", "B_act_spirit"] < 0.03),
        "resurrection_can_restore_body_capacity_without_spirit_channel": bool(final.loc["resurrection_body_only", "B_body_cap"] > 0.20 and final.loc["resurrection_body_only", "B_spirit_cap"] < 0.01 and final.loc["resurrection_body_only", "B_cap_total"] > 0.20),
        "nonlocal_override_rejected_even_with_channel_claim": bool(final.loc["nonlocal_override_claim", "Xi_law"] < 0.10 and final.loc["nonlocal_override_claim", "B_spirit_cap"] < 0.01),
    }
    return {"scenarios": scenarios, "falsifier_checks": tests, "predictive_power": pred.to_dict(orient="records")}


def write_markdown(summary: dict) -> None:
    lines = [
        "# P^spirit Participation Simulation Results",
        "",
        "**Author:** Manus AI  ",
        "**Date:** May 31, 2026  ",
        "**Status:** Deterministic readout-first simulation; non-canonical unless separately promoted",
        "",
        "## 1. Purpose",
        "",
        "This simulation tests whether the DET v7 `B_cap` term can be decomposed into embodied and spirit capacity channels without assuming that death automatically collapses all possible participation. The experiment distinguishes frozen record, embodied death under a channel-null assumption, a declared lawful spirit-channel hypothesis, copy controls, coercive pseudo-spirit controls, resurrection through body capacity, and nonlocal override claims.",
        "",
        "![P spirit component time series](P_spirit_components_timeseries.png)",
        "",
        "![P spirit falsifier matrix](P_spirit_falsifier_matrix.png)",
        "",
        "![Body versus spirit capacity](body_vs_spirit_capacity.png)",
        "",
        "![P spirit predictive power](P_spirit_predictive_power.png)",
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
        "| Scenario | B_addr | P_body | P_spirit | B_body_cap | B_spirit_cap | B_cap_total | B_act_spirit | B_path | Active spirit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scenario, vals in summary["scenarios"].items():
        lines.append(
            f"| {scenario.replace('_', ' ')} | "
            f"{vals['final_B_addr']:.3f} | "
            f"{vals['final_P_body']:.3f} | "
            f"{vals['final_P_spirit']:.3f} | "
            f"{vals['final_B_body_cap']:.3f} | "
            f"{vals['final_B_spirit_cap']:.3f} | "
            f"{vals['final_B_cap_total']:.3f} | "
            f"{vals['final_B_act_spirit']:.3f} | "
            f"{vals['final_B_path']:.3f} | "
            f"{str(vals['active_spirit_participation']).lower()} |"
        )
    lines.extend([
        "",
        "## 4. Interpretation",
        "",
        "The key result is that `B_addr > 0` is not sufficient for active spirit participation. Frozen record and conservative death-null scenarios preserve addressability while leaving `P_spirit`, `B_spirit_cap`, and `B_act_spirit` near zero. The declared spirit-channel scenario shows the logical form of a non-circular positive case: embodied participation remains near zero, but `P_spirit` becomes positive only when addressability, path-continuity, declared channel availability, locality, lawfulness, non-coercion, and agency compatibility all pass.",
        "",
        "Copy, coercion, and nonlocal override controls demonstrate why `P_spirit` must be gate-constrained. Pattern similarity alone cannot create path-continuity; apparent high boundary effect cannot count as lawful spirit participation if agency variation is suppressed; and channel claims fail when locality is absent. Resurrection is separated from spirit participation because it restores `B_body_cap` through a capable host while keeping `B_spirit_cap` at zero under the conservative channel-null setting.",
        "",
    ])
    (OUT / "P_spirit_participation_results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    scenarios = [
        ScenarioConfig("frozen_record", "Addressable record with no embodied or spirit channel."),
        ScenarioConfig("embodied_death_null", "Embodied participation collapses; undeclared spirit channel remains zero."),
        ScenarioConfig("active_spirit_declared", "Positive non-embodied channel when all law gates pass."),
        ScenarioConfig("copy_channel_claim", "High pattern similarity but broken path-continuity."),
        ScenarioConfig("coercive_pseudo_spirit", "Apparent activity rejected by non-coercion and agency gates."),
        ScenarioConfig("resurrection_body_only", "Restored body capacity without a spirit channel."),
        ScenarioConfig("nonlocal_override_claim", "Channel claim rejected by locality/law gate."),
    ]
    df = pd.concat([run_scenario(cfg) for cfg in scenarios], ignore_index=True)
    pred = predictive_power(df)
    make_plots(df, pred)
    summary = summarize(df, pred)
    df.to_csv(OUT / "P_spirit_timeseries.csv", index=False)
    final_rows = []
    for scenario, vals in summary["scenarios"].items():
        final_rows.append({"scenario": scenario, **vals})
    pd.DataFrame(final_rows).to_csv(OUT / "P_spirit_summary.csv", index=False)
    pred.to_csv(OUT / "P_spirit_predictive_power.csv", index=False)
    (OUT / "P_spirit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary)
    print(json.dumps(summary["falsifier_checks"], indent=2))


if __name__ == "__main__":
    main()
