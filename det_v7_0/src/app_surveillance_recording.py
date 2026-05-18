"""
DET application-layer model: digital surveillance versus natural structural recording.

This script does not modify canonical DET v7 laws. It defines a readout/application
model for a digital recording layer d_i(t) that sits above the canonical local
state variables F, q, a, C, H, and P. The model is intended to support the report
`det_v7_0/reports/det_surveillance_recording_kingdom_regimes_2026_05_18.md`.

Key interpretation:
- Natural recording is represented by canonical structural history q and pointer
  topology. Digital recording is a derivative readout stock d; it can improve
  coordination only through lawful local readout effects and cannot become a
  truer substrate than matter/history itself.
- Regime quality K and observability O follow the concurrent-regimes extension.
- Net digital surveillance value is positive only when recording is lawful,
  local, agency-gated, high-fidelity, and coherence-preserving.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SurveillanceParams:
    """Parameters for the DET surveillance application layer."""

    # Regime/readout coefficients. These mirror the concurrent-regimes spec.
    w_C: float = 0.35
    w_a: float = 0.25
    w_P: float = 0.15
    w_q: float = 0.25
    alpha_obs: float = 1.0
    beta_obs: float = 1.0
    gamma_obs: float = 1.0

    # Canonical-style presence parameters.
    lambda_P: float = 1.0
    gamma_v: float = 1.0
    sigma: float = 1.0

    # Application-layer surveillance parameters.
    record_intensity: float = 0.65
    fidelity: float = 0.85
    decay_digital: float = 0.01
    resource_cost: float = 0.018
    harm_prevention_gain: float = 0.020
    load_cost: float = 0.030
    decoherence_cost: float = 0.025
    false_record_cost: float = 0.020
    accountability_gain: float = 0.035
    coordination_gain: float = 0.025
    restoration_gain: float = 0.020
    alpha_q: float = 0.045
    eta_C_pos: float = 0.020
    eta_C_neg: float = 0.030
    eta_H: float = 0.040
    epsilon: float = 1e-8


@dataclass
class Scenario:
    """Initial and policy values for one deterministic scenario."""

    name: str
    C: float
    q: float
    a: float
    F: float
    H: float
    lawfulness: float
    record_intensity: float
    fidelity: float


SCENARIOS = [
    Scenario(
        name="Kingdom-aligned transparent recording",
        C=0.92,
        q=0.04,
        a=0.96,
        F=1.60,
        H=0.08,
        lawfulness=0.95,
        record_intensity=0.45,
        fidelity=0.94,
    ),
    Scenario(
        name="Mixed civic audit regime",
        C=0.62,
        q=0.22,
        a=0.78,
        F=1.15,
        H=0.22,
        lawfulness=0.62,
        record_intensity=0.62,
        fidelity=0.82,
    ),
    Scenario(
        name="World/coercive panoptic regime",
        C=0.28,
        q=0.58,
        a=0.70,
        F=0.90,
        H=0.45,
        lawfulness=0.18,
        record_intensity=0.86,
        fidelity=0.66,
    ),
]


def clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def presence(a: float, C: float, q: float, F: float, H: float, p: SurveillanceParams) -> float:
    """Canonical-style presence readout P = P_base / (1 + lambda_P q)."""
    # C is not in the canonical P expression, but is included indirectly in the
    # application state because high C reduces H and supports lawful readout.
    del C
    p_base = a * p.sigma * (1.0 / (1.0 + F)) * (1.0 / (1.0 + H)) * (1.0 / p.gamma_v)
    drag = 1.0 / (1.0 + p.lambda_P * q)
    return float(p_base * drag)


def regime_index(C: float, a: float, q: float, P: float, P_ref: float, p: SurveillanceParams) -> float:
    """K = clip(w_C*C + w_a*a + w_P*P_tilde - w_q*q, 0, 1)."""
    P_tilde = np.clip(P / (P_ref + p.epsilon), 0.0, 3.0)
    return float(clip01(p.w_C * C + p.w_a * a + p.w_P * P_tilde - p.w_q * q))


def observability(C: float, a: float, q: float, p: SurveillanceParams) -> float:
    """O = clip(a^alpha C^beta (1-q)^gamma, 0, 1)."""
    return float(clip01((a ** p.alpha_obs) * (C ** p.beta_obs) * ((1.0 - q) ** p.gamma_obs)))


def step_state(state: dict[str, float], p: SurveillanceParams) -> dict[str, float]:
    """Advance one application-layer step.

    The canonical q update is represented by q += alpha_q * max(0, -dF).
    Digital memory d is an ordinary derivative stock, not a core DET field.
    """
    C = state["C"]
    q = state["q"]
    a = state["a"]
    F = state["F"]
    H = state["H"]
    L = state["lawfulness"]
    r = state["record_intensity"]
    v = state["fidelity"]
    d = state.get("digital_record", 0.0)

    P = presence(a, C, q, F, H, p)
    P_ref = state.get("P_ref", P)
    K = regime_index(C, a, q, P, P_ref, p)
    O = observability(C, a, q, p)

    # Benefits are readout-mediated: accountability, coordination, and restorative
    # targeting are high only when recording is lawful and readable in the local regime.
    accountability = p.accountability_gain * r * v * L * K
    coordination = p.coordination_gain * r * v * L * O
    restoration = p.restoration_gain * r * v * L * K * O
    benefit = accountability + coordination + restoration

    # Costs arise through lawful DET channels: increased H, resource expenditure,
    # decoherence, and false-record pressure. Agency is not directly overwritten.
    coercion = 1.0 - L
    surveillance_load = p.load_cost * r * coercion * (1.0 + q) * (1.0 - 0.5 * K)
    resource_cost = p.resource_cost * r * (0.6 + 0.4 * coercion)
    decoherence = p.decoherence_cost * r * coercion * (1.0 - C) * (1.0 - K)
    false_record = p.false_record_cost * r * (1.0 - v * K) * (0.5 + coercion)
    cost = surveillance_load + resource_cost + decoherence + false_record

    net = benefit - cost

    # State updates.
    dF = p.harm_prevention_gain * r * v * L * K - resource_cost - surveillance_load
    F_next = max(0.0, F + dF)
    dq = p.alpha_q * max(0.0, -dF)
    q_next = float(clip01(q + dq))

    dC = p.eta_C_pos * r * v * L * K * O - p.eta_C_neg * r * coercion * (1.0 - O)
    C_next = float(clip01(C + dC))

    dH = p.eta_H * r * coercion - 0.015 * r * v * L * K
    H_next = max(0.0, H + dH)

    digital_record_next = max(0.0, (1.0 - p.decay_digital) * d + r * v)

    next_state = {
        **state,
        "C": C_next,
        "q": q_next,
        "F": F_next,
        "H": H_next,
        "digital_record": digital_record_next,
        "P": P,
        "K": K,
        "O": O,
        "benefit": benefit,
        "cost": cost,
        "net": net,
        "dF": dF,
        "dq": dq,
        "dC": dC,
        "dH": dH,
    }
    return next_state


def simulate_scenario(scenario: Scenario, p: SurveillanceParams, steps: int = 160) -> pd.DataFrame:
    P0 = presence(scenario.a, scenario.C, scenario.q, scenario.F, scenario.H, p)
    state = {
        **asdict(scenario),
        "digital_record": 0.0,
        "P_ref": P0,
    }
    rows = []
    for t in range(steps):
        state = step_state(state, p)
        rows.append({"t": t, **state})
    return pd.DataFrame(rows)


def net_value_grid(p: SurveillanceParams, n: int = 101) -> pd.DataFrame:
    """Evaluate one-step net value over regime K and lawfulness L.

    This abstract grid holds r and v fixed and expresses the threshold structure
    implied by the benefit/cost functional.
    """
    rows = []
    for K in np.linspace(0.0, 1.0, n):
        for L in np.linspace(0.0, 1.0, n):
            r = p.record_intensity
            v = p.fidelity
            # Let observability rise with K in the grid, but keep it distinct so
            # low-lawfulness, high-K abuse remains costly.
            O = np.clip(0.15 + 0.80 * K, 0.0, 1.0)
            accountability = p.accountability_gain * r * v * L * K
            coordination = p.coordination_gain * r * v * L * O
            restoration = p.restoration_gain * r * v * L * K * O
            benefit = accountability + coordination + restoration
            coercion = 1.0 - L
            surveillance_load = p.load_cost * r * coercion * (1.0 + (1.0 - K)) * (1.0 - 0.5 * K)
            resource_cost = p.resource_cost * r * (0.6 + 0.4 * coercion)
            decoherence = p.decoherence_cost * r * coercion * (1.0 - K) ** 2
            false_record = p.false_record_cost * r * (1.0 - v * K) * (0.5 + coercion)
            cost = surveillance_load + resource_cost + decoherence + false_record
            rows.append({"K": K, "lawfulness": L, "net": benefit - cost, "benefit": benefit, "cost": cost})
    return pd.DataFrame(rows)


def write_outputs(output_dir: Path) -> None:
    p = SurveillanceParams()
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "parameters.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(p), f, indent=2)

    scenario_frames = []
    summaries = []
    for scenario in SCENARIOS:
        df = simulate_scenario(scenario, p)
        scenario_frames.append(df)
        final = df.iloc[-1]
        summaries.append({
            "scenario": scenario.name,
            "mean_net": float(df["net"].mean()),
            "final_K": float(final["K"]),
            "final_O": float(final["O"]),
            "final_q": float(final["q"]),
            "final_C": float(final["C"]),
            "final_H": float(final["H"]),
            "final_digital_record": float(final["digital_record"]),
        })
    all_scenarios = pd.concat(scenario_frames, ignore_index=True)
    all_scenarios.to_csv(output_dir / "scenario_timeseries.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "scenario_summary.csv", index=False)

    grid = net_value_grid(p)
    grid.to_csv(output_dir / "net_value_grid.csv", index=False)

    # Heatmap: net digital recording value by K and lawfulness.
    pivot = grid.pivot(index="lawfulness", columns="K", values="net")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="coolwarm",
        vmin=-max(abs(grid["net"].min()), abs(grid["net"].max())),
        vmax=max(abs(grid["net"].min()), abs(grid["net"].max())),
    )
    ax.contour(
        np.linspace(0, 1, pivot.shape[1]),
        np.linspace(0, 1, pivot.shape[0]),
        pivot.values,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )
    ax.set_xlabel("Regime index K (World-like → Kingdom-like)")
    ax.set_ylabel("Lawfulness / agency-gated consent L")
    ax.set_title("DET Digital Recording Net Value Threshold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Net value per step")
    fig.tight_layout()
    fig.savefig(output_dir / "net_value_heatmap.png", dpi=180)
    plt.close(fig)

    # Scenario trajectories.
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharex=True)
    for name, group in all_scenarios.groupby("name"):
        axes[0, 0].plot(group["t"], group["K"], label=name)
        axes[0, 1].plot(group["t"], group["net"], label=name)
        axes[1, 0].plot(group["t"], group["q"], label=name)
        axes[1, 1].plot(group["t"], group["C"], label=name)
    axes[0, 0].set_ylabel("K")
    axes[0, 0].set_title("Regime index")
    axes[0, 1].set_ylabel("Net")
    axes[0, 1].set_title("Digital-recording net value")
    axes[1, 0].set_ylabel("q")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_title("Natural structural history/debt")
    axes[1, 1].set_ylabel("C")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_title("Coherence")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "scenario_trajectories.png", dpi=180)
    plt.close(fig)

    # Human-readable Markdown summary table for direct inclusion.
    with (output_dir / "scenario_summary.md").open("w", encoding="utf-8") as f:
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")


if __name__ == "__main__":
    write_outputs(Path("/home/ubuntu/det/det_v7_0/reports/surveillance_recording_model"))
