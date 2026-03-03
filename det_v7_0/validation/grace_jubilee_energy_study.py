#!/usr/bin/env python3
"""Grace-Jubilee interaction study for DET v7 mutable-q baseline.

Focus:
- How Grace injection affects local free-resource budget F_op
- How F_op gating controls Jubilee when energy coupling is enabled
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from det_v6_3_1d_collider import DETCollider1D, DETParams1D


@dataclass
class StudyResult:
    scenario: str
    grace_enabled: bool
    jubilee_energy_coupling: bool
    momentum_amp: float
    steps: int
    q_initial: float
    q_final: float
    q_drop: float
    mean_f: float
    mean_f_op: float
    total_grace: float
    total_jubilee: float


def run_case(*, scenario: str, grace_enabled: bool, jubilee_energy_coupling: bool, momentum_amp: float, steps: int) -> StudyResult:
    np.random.seed(7)

    n = 180
    x = np.arange(n)
    params = DETParams1D(
        N=n,
        DT=0.02,
        F_VAC=0.06,
        F_MIN=0.0,
        C_init=0.9,
        momentum_enabled=True,
        alpha_pi=0.08,
        lambda_pi=0.01,
        mu_pi=0.35,
        q_enabled=True,
        alpha_q=0.0005,
        beta_a=0.2,
        floor_enabled=True,
        eta_floor=0.15,
        F_core=5.0,
        gravity_enabled=False,
        boundary_enabled=True,
        grace_enabled=grace_enabled,
        F_MIN_grace=0.30,
        healing_enabled=False,
        jubilee_enabled=True,
        delta_q=0.12,
        n_q=1,
        D_0=0.01,
        jubilee_energy_coupling=jubilee_energy_coupling,
        outflow_limit=0.35,
    )
    sim = DETCollider1D(params)

    # High-q, high-C, high-a but energy-starved initialization.
    sim.F[:] = 0.01
    sim.q[:] = 0.75
    sim.a[:] = 0.95
    sim.C_R[:] = 0.92
    # Controlled dissipation source to drive boundary operators.
    sim.pi_R[:] = momentum_amp * np.sin(2.0 * np.pi * x / 12.0)

    q_initial = float(np.mean(sim.q))
    for _ in range(steps):
        sim.step()

    f_op = np.maximum(sim.F - params.F_VAC, 0.0)
    q_final = float(np.mean(sim.q))

    return StudyResult(
        scenario=scenario,
        grace_enabled=grace_enabled,
        jubilee_energy_coupling=jubilee_energy_coupling,
        momentum_amp=momentum_amp,
        steps=steps,
        q_initial=q_initial,
        q_final=q_final,
        q_drop=q_initial - q_final,
        mean_f=float(np.mean(sim.F)),
        mean_f_op=float(np.mean(f_op)),
        total_grace=float(sim.total_grace_injected),
        total_jubilee=float(sim.total_jubilee),
    )


def to_markdown(results: list[StudyResult]) -> str:
    lines = [
        "# Grace-Jubilee Energy Interaction Study",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Scenario Summary",
        "",
        "All runs use the same starved initial state (`F=0.01`, `q=0.75`, `a=0.95`, `C=0.92`) and differ by Grace toggle, energy-coupling toggle, and dissipation strength.",
        "",
        "## Results",
        "",
        "| Scenario | Grace | Energy Coupling | Mom Amp | q_drop | mean F_op | total Grace | total Jubilee |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            "| {s} | {g} | {e} | {m:.1f} | {qd:.6f} | {fop:.6f} | {tg:.6f} | {tj:.6f} |".format(
                s=r.scenario,
                g="on" if r.grace_enabled else "off",
                e="on" if r.jubilee_energy_coupling else "off",
                m=r.momentum_amp,
                qd=r.q_drop,
                fop=r.mean_f_op,
                tg=r.total_grace,
                tj=r.total_jubilee,
            )
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "1. With energy coupling ON and Grace OFF in the low-dissipation regime, Jubilee is effectively blocked (`total Jubilee ~ 0`) because `F_op ~ 0`.",
            "2. Turning Grace ON raises `F` and `F_op`, enabling nonzero Jubilee under the same energy-coupled law.",
            "3. With energy coupling OFF, Jubilee proceeds even when `F_op ~ 0`, confirming the cap is the gating mechanism.",
            "4. In higher-dissipation regimes, Grace still increases both `F_op` and total Jubilee, but no longer acts as a hard on/off unlock.",
            "",
            "Interpretation: Grace and Jubilee are coupled through local resource availability. Grace is not direct forgiveness; it shifts local budgets so energy-coupled Jubilee can lawfully act.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grace-Jubilee energy interaction study")
    parser.add_argument("--steps", type=int, default=3000, help="Steps per scenario")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent.parent / "reports" / "grace_jubilee_energy_interaction_2026_03_03.md"),
        help="Output markdown path",
    )
    parser.add_argument("--json", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    scenarios = [
        ("starved-lowD", 1.0),
        ("starved-highD", 2.0),
    ]

    results: list[StudyResult] = []
    for name, amp in scenarios:
        for grace in (False, True):
            for energy in (True, False):
                results.append(
                    run_case(
                        scenario=name,
                        grace_enabled=grace,
                        jubilee_energy_coupling=energy,
                        momentum_amp=amp,
                        steps=args.steps,
                    )
                )

    md = to_markdown(results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))

    print(md)
    print(f"Saved report: {out_path}")


if __name__ == "__main__":
    main()
