"""
DET application-layer model: digital surveillance versus natural structural recording.

This script does not modify canonical DET v7 laws. It defines a readout/application
model for digital recording layers that sit above the canonical local state
variables F, q, a, C, H, and P. The model supports the report
`det_v7_0/reports/det_surveillance_recording_kingdom_regimes_2026_05_18.md`.

Key interpretation:
- Natural recording is represented by canonical structural history q and pointer
  topology. Digital recording is a derivative readout stock d; it can improve
  coordination only through lawful local readout effects and cannot become a
  truer substrate than matter/history itself.
- Regime quality K and observability O follow the concurrent-regimes extension.
- Net digital surveillance value is positive only when recording is lawful,
  local, agency-gated, high-fidelity, and coherence-preserving.
- Dual-regime operation is modeled as two simultaneous readout layers, such as
  an oppressive work regime and a Kingdom-like present/future regime, with
  lawful bleed-over terms. Artificial observers consolidate derivative records;
  they are not granted ontological priority or direct agency-write authority.
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

    # Single-regime surveillance parameters.
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

    # Dual-regime and now/future Kingdom parameters.
    grace_coherence_gain: float = 0.018
    grace_load_relief: float = 0.028
    grace_resource_gain: float = 0.014
    bleed_work_to_kingdom: float = 0.040
    bleed_kingdom_to_work: float = 0.032
    future_tension_cost: float = 0.012

    # Artificial observer / AI consolidation parameters.
    ai_audit_gain: float = 0.028
    ai_scale_cost: float = 0.026
    ai_false_amplification: float = 0.035
    ai_context_loss_cost: float = 0.020
    ai_min_governance_for_help: float = 0.50

    epsilon: float = 1e-8


@dataclass
class Scenario:
    """Initial and policy values for one deterministic single-regime scenario."""

    name: str
    C: float
    q: float
    a: float
    F: float
    H: float
    lawfulness: float
    record_intensity: float
    fidelity: float


@dataclass
class DualRegimeScenario:
    """Initial and policy values for a dual-regime scenario.

    The same embodied node is exposed to a work/institutional recording regime and
    a Kingdom-like present/future readout regime. The artificial observer is a
    derivative consolidation layer over the two streams.
    """

    name: str
    C: float
    q: float
    a: float
    F: float
    H: float
    work_lawfulness: float
    work_record_intensity: float
    work_fidelity: float
    kingdom_lawfulness: float
    kingdom_record_intensity: float
    kingdom_fidelity: float
    kingdom_overlap: float
    ai_consolidation: float
    ai_governance: float
    ai_fidelity: float
    ai_opacity: float


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


DUAL_REGIME_SCENARIOS = [
    DualRegimeScenario(
        name="Oppressive work recording with weak Kingdom channel",
        C=0.58,
        q=0.25,
        a=0.82,
        F=1.05,
        H=0.30,
        work_lawfulness=0.12,
        work_record_intensity=0.90,
        work_fidelity=0.80,
        kingdom_lawfulness=0.88,
        kingdom_record_intensity=0.28,
        kingdom_fidelity=0.92,
        kingdom_overlap=0.25,
        ai_consolidation=0.72,
        ai_governance=0.24,
        ai_fidelity=0.82,
        ai_opacity=0.74,
    ),
    DualRegimeScenario(
        name="Oppressive work recording with Kingdom counter-witness",
        C=0.58,
        q=0.25,
        a=0.82,
        F=1.05,
        H=0.30,
        work_lawfulness=0.12,
        work_record_intensity=0.90,
        work_fidelity=0.80,
        kingdom_lawfulness=0.94,
        kingdom_record_intensity=0.58,
        kingdom_fidelity=0.95,
        kingdom_overlap=0.58,
        ai_consolidation=0.60,
        ai_governance=0.86,
        ai_fidelity=0.90,
        ai_opacity=0.18,
    ),
    DualRegimeScenario(
        name="Transparent work regime absorbing Kingdom bleed-over",
        C=0.70,
        q=0.18,
        a=0.84,
        F=1.20,
        H=0.22,
        work_lawfulness=0.78,
        work_record_intensity=0.55,
        work_fidelity=0.86,
        kingdom_lawfulness=0.92,
        kingdom_record_intensity=0.52,
        kingdom_fidelity=0.94,
        kingdom_overlap=0.50,
        ai_consolidation=0.50,
        ai_governance=0.76,
        ai_fidelity=0.88,
        ai_opacity=0.22,
    ),
    DualRegimeScenario(
        name="AI panopticon consolidation across dual regimes",
        C=0.45,
        q=0.35,
        a=0.78,
        F=0.96,
        H=0.42,
        work_lawfulness=0.10,
        work_record_intensity=0.96,
        work_fidelity=0.72,
        kingdom_lawfulness=0.86,
        kingdom_record_intensity=0.42,
        kingdom_fidelity=0.90,
        kingdom_overlap=0.45,
        ai_consolidation=0.95,
        ai_governance=0.10,
        ai_fidelity=0.76,
        ai_opacity=0.90,
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


def single_regime_terms(
    r: float,
    v: float,
    L: float,
    K: float,
    O: float,
    C: float,
    q: float,
    p: SurveillanceParams,
) -> dict[str, float]:
    """Return benefit/cost terms for one derivative recording stream."""
    accountability = p.accountability_gain * r * v * L * K
    coordination = p.coordination_gain * r * v * L * O
    restoration = p.restoration_gain * r * v * L * K * O
    benefit = accountability + coordination + restoration

    coercion = 1.0 - L
    surveillance_load = p.load_cost * r * coercion * (1.0 + q) * (1.0 - 0.5 * K)
    resource_cost = p.resource_cost * r * (0.6 + 0.4 * coercion)
    decoherence = p.decoherence_cost * r * coercion * (1.0 - C) * (1.0 - K)
    false_record = p.false_record_cost * r * (1.0 - v * K) * (0.5 + coercion)
    cost = surveillance_load + resource_cost + decoherence + false_record

    return {
        "accountability": accountability,
        "coordination": coordination,
        "restoration": restoration,
        "benefit": benefit,
        "surveillance_load": surveillance_load,
        "resource_cost": resource_cost,
        "decoherence": decoherence,
        "false_record": false_record,
        "cost": cost,
    }


def step_state(state: dict[str, float], p: SurveillanceParams) -> dict[str, float]:
    """Advance one single-regime application-layer step.

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
    terms = single_regime_terms(r, v, L, K, O, C, q, p)

    benefit = terms["benefit"]
    cost = terms["cost"]
    net = benefit - cost

    # State updates.
    dF = p.harm_prevention_gain * r * v * L * K - terms["resource_cost"] - terms["surveillance_load"]
    F_next = max(0.0, F + dF)
    dq = p.alpha_q * max(0.0, -dF)
    q_next = float(clip01(q + dq))

    coercion = 1.0 - L
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


def step_dual_regime_state(state: dict[str, float], p: SurveillanceParams) -> dict[str, float]:
    """Advance one dual-regime step with work, Kingdom, and AI readout layers.

    Strict DET guardrails:
    - work, Kingdom, and AI records are derivative readouts rather than replacements
      for natural q/pointer history;
    - AI consolidation cannot directly write agency a;
    - cross-regime bleed-over changes local F, H, C, and q only through explicit,
      bounded application-layer channels.
    """
    C = state["C"]
    q = state["q"]
    a = state["a"]
    F = state["F"]
    H = state["H"]
    L_w = state["work_lawfulness"]
    r_w = state["work_record_intensity"]
    v_w = state["work_fidelity"]
    L_k = state["kingdom_lawfulness"]
    r_k = state["kingdom_record_intensity"]
    v_k = state["kingdom_fidelity"]
    phi = state["kingdom_overlap"]
    A = state["ai_consolidation"]
    G = state["ai_governance"]
    v_ai = state["ai_fidelity"]
    opacity = state["ai_opacity"]
    d_work = state.get("digital_work_record", 0.0)
    d_kingdom = state.get("digital_kingdom_record", 0.0)
    d_ai = state.get("ai_consolidated_record", 0.0)

    P = presence(a, C, q, F, H, p)
    P_ref = state.get("P_ref", P)
    K = regime_index(C, a, q, P, P_ref, p)
    O = observability(C, a, q, p)

    work = single_regime_terms(r_w, v_w, L_w, K, O, C, q, p)
    kingdom = single_regime_terms(r_k, v_k, L_k, K, O, C, q, p)

    total_stream = r_w * v_w + r_k * v_k
    L_mix = (r_w * v_w * L_w + r_k * v_k * L_k) / (total_stream + p.epsilon)

    # Kingdom now/future overlap. phi is not a hidden global force; it gates how
    # much high-coherence future regime can be locally expressed in the present.
    kingdom_presence_lift = phi * r_k * v_k * L_k * K * O
    grace_coherence = p.grace_coherence_gain * kingdom_presence_lift * (1.0 - C)
    grace_relief = p.grace_load_relief * kingdom_presence_lift * (1.0 + q)
    grace_resource = p.grace_resource_gain * kingdom_presence_lift

    # Bleed-over: oppressive work records follow the agent into other contexts;
    # Kingdom counter-witness can also bleed back into work as audit/restoration.
    work_to_kingdom_bleed = (
        p.bleed_work_to_kingdom
        * r_w
        * (1.0 - L_w)
        * (1.0 + A * (1.0 - G + opacity) / 2.0)
        * (1.0 - C)
        * (1.0 - 0.5 * O)
    )
    kingdom_to_work_bleed = (
        p.bleed_kingdom_to_work
        * r_k
        * v_k
        * L_k
        * phi
        * K
        * O
        * (0.5 + 0.5 * G)
    )

    # Artificial observer terms. AI helps only when consolidated records remain
    # governed, transparent, and context-preserving; otherwise consolidation
    # amplifies false records and cross-context pressure.
    ai_audit = p.ai_audit_gain * A * G * v_ai * L_mix * K * O
    ai_scale = p.ai_scale_cost * A * total_stream * (1.0 - G + opacity) * (1.0 + q) * (1.0 - 0.35 * K)
    ai_false = p.ai_false_amplification * A * total_stream * (1.0 - v_ai * K) * (1.0 - G * L_mix)
    ai_context_loss = p.ai_context_loss_cost * A * opacity * r_w * (1.0 - L_w) * (1.0 - O)
    ai_net = ai_audit - ai_scale - ai_false - ai_context_loss

    # The future/now Kingdom gradient is beneficial only under lawful Kingdom
    # readout. If the present is highly incoherent and AI/work layers are opaque,
    # the gap itself adds tension because future truth is locally misread.
    future_tension = p.future_tension_cost * phi * ((1.0 - C) + q) * (1.0 - L_k * K) * (0.5 + 0.5 * opacity * A)

    benefit = work["benefit"] + kingdom["benefit"] + ai_audit + grace_resource + grace_relief + kingdom_to_work_bleed
    cost = work["cost"] + kingdom["cost"] + ai_scale + ai_false + ai_context_loss + work_to_kingdom_bleed + future_tension
    net = benefit - cost

    # State updates through lawful DET channels.
    dF = (
        p.harm_prevention_gain * (r_w * v_w * L_w + r_k * v_k * L_k) * K
        + grace_resource
        + ai_audit
        - work["resource_cost"]
        - kingdom["resource_cost"]
        - work["surveillance_load"]
        - ai_scale
    )
    F_next = max(0.0, F + dF)
    dq = p.alpha_q * max(0.0, -dF) + 0.20 * work_to_kingdom_bleed + 0.10 * ai_false
    q_next = float(clip01(q + dq))

    dC = (
        p.eta_C_pos * (r_w * v_w * L_w + r_k * v_k * L_k) * K * O
        + grace_coherence
        + 0.25 * kingdom_to_work_bleed
        - p.eta_C_neg * r_w * (1.0 - L_w) * (1.0 - O)
        - work_to_kingdom_bleed
        - 0.25 * ai_context_loss
        - 0.20 * future_tension
    )
    C_next = float(clip01(C + dC))

    dH = (
        p.eta_H * r_w * (1.0 - L_w) * (1.0 + A * (1.0 - G + opacity) / 2.0)
        + 0.50 * ai_scale
        + future_tension
        - grace_relief
        - kingdom_to_work_bleed
        - 0.30 * ai_audit
    )
    H_next = max(0.0, H + dH)

    digital_work_next = max(0.0, (1.0 - p.decay_digital) * d_work + r_w * v_w)
    digital_kingdom_next = max(0.0, (1.0 - p.decay_digital) * d_kingdom + r_k * v_k)
    ai_next = max(0.0, (1.0 - p.decay_digital) * d_ai + A * v_ai * total_stream)

    next_state = {
        **state,
        "C": C_next,
        "q": q_next,
        "F": F_next,
        "H": H_next,
        "digital_work_record": digital_work_next,
        "digital_kingdom_record": digital_kingdom_next,
        "ai_consolidated_record": ai_next,
        "P": P,
        "K": K,
        "O": O,
        "work_benefit": work["benefit"],
        "work_cost": work["cost"],
        "kingdom_benefit": kingdom["benefit"],
        "kingdom_cost": kingdom["cost"],
        "kingdom_presence_lift": kingdom_presence_lift,
        "grace_coherence": grace_coherence,
        "grace_relief": grace_relief,
        "work_to_kingdom_bleed": work_to_kingdom_bleed,
        "kingdom_to_work_bleed": kingdom_to_work_bleed,
        "ai_audit": ai_audit,
        "ai_scale": ai_scale,
        "ai_false": ai_false,
        "ai_context_loss": ai_context_loss,
        "ai_net": ai_net,
        "future_tension": future_tension,
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


def simulate_dual_regime_scenario(
    scenario: DualRegimeScenario,
    p: SurveillanceParams,
    steps: int = 160,
) -> pd.DataFrame:
    P0 = presence(scenario.a, scenario.C, scenario.q, scenario.F, scenario.H, p)
    state = {
        **asdict(scenario),
        "digital_work_record": 0.0,
        "digital_kingdom_record": 0.0,
        "ai_consolidated_record": 0.0,
        "P_ref": P0,
    }
    rows = []
    for t in range(steps):
        state = step_dual_regime_state(state, p)
        rows.append({"t": t, **state})
    return pd.DataFrame(rows)


def net_value_grid(p: SurveillanceParams, n: int = 101) -> pd.DataFrame:
    """Evaluate one-step net value over regime K and lawfulness L."""
    rows = []
    for K in np.linspace(0.0, 1.0, n):
        for L in np.linspace(0.0, 1.0, n):
            r = p.record_intensity
            v = p.fidelity
            # Let observability rise with K in the grid, but keep it distinct so
            # low-lawfulness, high-K abuse remains costly.
            O = np.clip(0.15 + 0.80 * K, 0.0, 1.0)
            terms = single_regime_terms(r, v, L, K, O, C=K, q=(1.0 - K), p=p)
            rows.append({"K": K, "lawfulness": L, "net": terms["benefit"] - terms["cost"], "benefit": terms["benefit"], "cost": terms["cost"]})
    return pd.DataFrame(rows)


def dual_bleed_grid(p: SurveillanceParams, n: int = 81) -> pd.DataFrame:
    """One-step net value grid for work lawfulness and AI governance.

    This grid holds Kingdom readout present but not dominant, so it emphasizes
    how oppressive work recording and AI consolidation can either bleed into or
    be corrected by lawful counter-witness.
    """
    rows = []
    base = DualRegimeScenario(
        name="grid",
        C=0.58,
        q=0.25,
        a=0.82,
        F=1.05,
        H=0.30,
        work_lawfulness=0.0,
        work_record_intensity=0.90,
        work_fidelity=0.80,
        kingdom_lawfulness=0.92,
        kingdom_record_intensity=0.48,
        kingdom_fidelity=0.94,
        kingdom_overlap=0.45,
        ai_consolidation=0.75,
        ai_governance=0.0,
        ai_fidelity=0.84,
        ai_opacity=0.60,
    )
    for L_w in np.linspace(0.0, 1.0, n):
        for G in np.linspace(0.0, 1.0, n):
            scenario = DualRegimeScenario(**{**asdict(base), "work_lawfulness": float(L_w), "ai_governance": float(G)})
            df = simulate_dual_regime_scenario(scenario, p, steps=1)
            row = df.iloc[-1]
            rows.append({
                "work_lawfulness": L_w,
                "ai_governance": G,
                "net": float(row["net"]),
                "work_to_kingdom_bleed": float(row["work_to_kingdom_bleed"]),
                "kingdom_to_work_bleed": float(row["kingdom_to_work_bleed"]),
                "ai_net": float(row["ai_net"]),
            })
    return pd.DataFrame(rows)


def ai_consolidation_grid(p: SurveillanceParams, n: int = 81) -> pd.DataFrame:
    """One-step grid over AI consolidation and AI governance."""
    rows = []
    base = DualRegimeScenario(
        name="ai_grid",
        C=0.52,
        q=0.30,
        a=0.80,
        F=1.00,
        H=0.35,
        work_lawfulness=0.16,
        work_record_intensity=0.92,
        work_fidelity=0.78,
        kingdom_lawfulness=0.90,
        kingdom_record_intensity=0.50,
        kingdom_fidelity=0.93,
        kingdom_overlap=0.50,
        ai_consolidation=0.0,
        ai_governance=0.0,
        ai_fidelity=0.82,
        ai_opacity=0.70,
    )
    for A in np.linspace(0.0, 1.0, n):
        for G in np.linspace(0.0, 1.0, n):
            # More governance also lowers opacity in the sensitivity surface.
            scenario = DualRegimeScenario(**{
                **asdict(base),
                "ai_consolidation": float(A),
                "ai_governance": float(G),
                "ai_opacity": float(0.85 * (1.0 - G) + 0.10),
            })
            df = simulate_dual_regime_scenario(scenario, p, steps=1)
            row = df.iloc[-1]
            rows.append({
                "ai_consolidation": A,
                "ai_governance": G,
                "net": float(row["net"]),
                "ai_net": float(row["ai_net"]),
                "work_to_kingdom_bleed": float(row["work_to_kingdom_bleed"]),
            })
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

    # Single-regime scenario trajectories.
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

    # Dual-regime scenarios.
    dual_frames = []
    dual_summaries = []
    for scenario in DUAL_REGIME_SCENARIOS:
        df = simulate_dual_regime_scenario(scenario, p)
        dual_frames.append(df)
        final = df.iloc[-1]
        dual_summaries.append({
            "scenario": scenario.name,
            "mean_net": float(df["net"].mean()),
            "mean_work_to_kingdom_bleed": float(df["work_to_kingdom_bleed"].mean()),
            "mean_kingdom_to_work_bleed": float(df["kingdom_to_work_bleed"].mean()),
            "mean_ai_net": float(df["ai_net"].mean()),
            "final_K": float(final["K"]),
            "final_O": float(final["O"]),
            "final_q": float(final["q"]),
            "final_C": float(final["C"]),
            "final_H": float(final["H"]),
            "final_ai_record": float(final["ai_consolidated_record"]),
        })
    all_dual = pd.concat(dual_frames, ignore_index=True)
    all_dual.to_csv(output_dir / "dual_regime_timeseries.csv", index=False)
    dual_summary_df = pd.DataFrame(dual_summaries)
    dual_summary_df.to_csv(output_dir / "dual_regime_summary.csv", index=False)

    # Human-readable Markdown summary tables for direct inclusion.
    with (output_dir / "scenario_summary.md").open("w", encoding="utf-8") as f:
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    with (output_dir / "dual_regime_summary.md").open("w", encoding="utf-8") as f:
        f.write(dual_summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    # Dual-regime trajectory plot.
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 11.0), sharex=True)
    for name, group in all_dual.groupby("name"):
        axes[0, 0].plot(group["t"], group["K"], label=name)
        axes[0, 1].plot(group["t"], group["net"], label=name)
        axes[1, 0].plot(group["t"], group["work_to_kingdom_bleed"], label=name)
        axes[1, 1].plot(group["t"], group["kingdom_to_work_bleed"], label=name)
        axes[2, 0].plot(group["t"], group["ai_net"], label=name)
        axes[2, 1].plot(group["t"], group["H"], label=name)
    axes[0, 0].set_title("Regime index under dual operation")
    axes[0, 1].set_title("Dual-regime net value")
    axes[1, 0].set_title("Oppressive work → Kingdom bleed")
    axes[1, 1].set_title("Kingdom counter-witness → work bleed")
    axes[2, 0].set_title("AI consolidation net effect")
    axes[2, 1].set_title("Coordination load H")
    axes[2, 0].set_xlabel("Step")
    axes[2, 1].set_xlabel("Step")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "dual_regime_trajectories.png", dpi=180)
    plt.close(fig)

    # Work lawfulness vs AI governance heatmap.
    dual_grid = dual_bleed_grid(p)
    dual_grid.to_csv(output_dir / "dual_bleed_grid.csv", index=False)
    pivot = dual_grid.pivot(index="ai_governance", columns="work_lawfulness", values="net")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    max_abs = max(abs(dual_grid["net"].min()), abs(dual_grid["net"].max()))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    ax.contour(
        np.linspace(0, 1, pivot.shape[1]),
        np.linspace(0, 1, pivot.shape[0]),
        pivot.values,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )
    ax.set_xlabel("Work-regime lawfulness")
    ax.set_ylabel("AI governance / auditability")
    ax.set_title("Dual-Regime Bleed-Over Net Threshold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("One-step net value")
    fig.tight_layout()
    fig.savefig(output_dir / "dual_bleed_heatmap.png", dpi=180)
    plt.close(fig)

    # AI consolidation vs governance heatmap.
    ai_grid = ai_consolidation_grid(p)
    ai_grid.to_csv(output_dir / "ai_consolidation_grid.csv", index=False)
    pivot = ai_grid.pivot(index="ai_governance", columns="ai_consolidation", values="ai_net")
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    max_abs = max(abs(ai_grid["ai_net"].min()), abs(ai_grid["ai_net"].max()))
    im = ax.imshow(
        pivot.values,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    ax.contour(
        np.linspace(0, 1, pivot.shape[1]),
        np.linspace(0, 1, pivot.shape[0]),
        pivot.values,
        levels=[0.0],
        colors="black",
        linewidths=1.5,
    )
    ax.set_xlabel("AI consolidation strength")
    ax.set_ylabel("AI governance / transparency")
    ax.set_title("Artificial Observer Consolidation: Help vs Harm")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("AI net effect")
    fig.tight_layout()
    fig.savefig(output_dir / "ai_consolidation_heatmap.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    write_outputs(Path("/home/ubuntu/det/det_v7_0/reports/surveillance_recording_model"))
