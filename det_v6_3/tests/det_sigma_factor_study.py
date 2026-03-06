"""
DET v6.3 sigma-factor study.

Goal:
1) Quantify how important sigma is in collider dynamics.
2) Test whether sigma can be moved out of core and derived from local primitives.

Modes compared:
- core_dynamic: current 3D core sigma law (sigma_dynamic=True)
- frozen: sigma held at 1.0 (sigma_dynamic=False)
- external_local_derived: sigma_dynamic=False + external local policy update
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_consciousness_c1 import ConsciousnessParamsC1, DETConsciousnessC1
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


@dataclass(frozen=True)
class SigmaMode:
    name: str
    use_core_dynamic: bool
    use_external_policy: bool


def _neighbor_ops():
    Xp = lambda arr: np.roll(arr, -1, axis=2)
    Xm = lambda arr: np.roll(arr, 1, axis=2)
    Yp = lambda arr: np.roll(arr, -1, axis=1)
    Ym = lambda arr: np.roll(arr, 1, axis=1)
    Zp = lambda arr: np.roll(arr, -1, axis=0)
    Zm = lambda arr: np.roll(arr, 1, axis=0)
    return Xp, Xm, Yp, Ym, Zp, Zm


def derive_sigma_local(sim: DETCollider3D) -> np.ndarray:
    """
    External sigma policy derived from local primitives only.

    Uses: a, q, local coherence, and local F-gradient magnitude.
    """
    Xp, Xm, Yp, Ym, Zp, Zm = _neighbor_ops()

    C_avg = (
        sim.C_X
        + Xm(sim.C_X)
        + sim.C_Y
        + Ym(sim.C_Y)
        + sim.C_Z
        + Zm(sim.C_Z)
    ) / 6.0

    gradF = (
        np.abs(sim.F - Xp(sim.F))
        + np.abs(sim.F - Xm(sim.F))
        + np.abs(sim.F - Yp(sim.F))
        + np.abs(sim.F - Ym(sim.F))
        + np.abs(sim.F - Zp(sim.F))
        + np.abs(sim.F - Zm(sim.F))
    ) / 6.0
    grad_term = gradF / (1.0 + gradF)

    sigma = 0.75 + 0.20 * sim.a + 0.25 * C_avg + 0.15 * grad_term - 0.15 * sim.q
    return np.clip(sigma, 0.2, 2.5)


def apply_mode_sigma(sim: DETCollider3D, mode: SigmaMode):
    if mode.use_external_policy:
        sim.sigma[:] = derive_sigma_local(sim)
    elif not mode.use_core_dynamic:
        sim.sigma[:] = 1.0


def make_params(mode: SigmaMode, N: int) -> DETParams3D:
    return DETParams3D(
        N=N,
        DT=0.02,
        F_VAC=0.01,
        gravity_enabled=True,
        momentum_enabled=True,
        angular_momentum_enabled=True,
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
        sigma_dynamic=mode.use_core_dynamic,
    )


def _step(sim: DETCollider3D, mode: SigmaMode):
    sim.step()
    apply_mode_sigma(sim, mode)


def scenario_binding(mode: SigmaMode, seed: int = 11) -> Dict[str, float]:
    np.random.seed(seed)
    params = make_params(mode, N=30)
    params.angular_momentum_enabled = False
    params.floor_enabled = False
    sim = DETCollider3D(params)
    apply_mode_sigma(sim, mode)

    c = params.N // 2
    init_sep = 10
    sim.add_packet((c, c, c - init_sep // 2), mass=6.0, width=2.2, momentum=(0, 0, 0.08), initial_q=0.4)
    sim.add_packet((c, c, c + init_sep // 2), mass=6.0, width=2.2, momentum=(0, 0, -0.08), initial_q=0.4)

    seps = []
    for _ in range(1000):
        sep, _ = sim.separation()
        seps.append(sep)
        _step(sim, mode)

    seps = np.asarray(seps, dtype=float)
    valid = seps[seps > 0]
    final_sep = float(valid[-1]) if len(valid) else float(seps[-1])
    min_sep = float(np.min(valid)) if len(valid) else float(np.min(seps))
    passed = (final_sep < init_sep * 0.9) or (min_sep < init_sep * 0.6)
    return {"passed": float(passed), "final_sep": final_sep, "min_sep": min_sep}


def scenario_mass_conservation(mode: SigmaMode, seed: int = 12) -> Dict[str, float]:
    np.random.seed(seed)
    params = make_params(mode, N=24)
    sim = DETCollider3D(params)
    apply_mode_sigma(sim, mode)

    sim.add_packet((8, 8, 8), mass=8.0, width=2.5, momentum=(0.15, 0.15, 0.15), initial_q=0.35)
    sim.add_packet((16, 16, 16), mass=8.0, width=2.5, momentum=(-0.15, -0.15, -0.15), initial_q=0.35)

    m0 = sim.total_mass()
    for _ in range(500):
        _step(sim, mode)
    m1 = sim.total_mass()
    drift = abs((m1 - m0) - sim.total_grace_injected) / (m0 + 1e-12)
    return {"passed": float(drift < 0.12), "effective_drift": float(drift)}


def scenario_time_dilation(mode: SigmaMode, seed: int = 13) -> Dict[str, float]:
    np.random.seed(seed)
    params = make_params(mode, N=28)
    params.angular_momentum_enabled = False
    sim = DETCollider3D(params)
    apply_mode_sigma(sim, mode)

    c = params.N // 2
    sim.add_packet((c, c, c), mass=20.0, width=2.8, initial_q=0.5)
    for _ in range(400):
        _step(sim, mode)

    tau_center = float(sim.accumulated_proper_time[c, c, c])
    tau_edge = float(sim.accumulated_proper_time[c, c, c + 9])
    dilation = tau_edge / (tau_center + 1e-12)
    return {"passed": float(dilation > 1.01), "dilation_factor": dilation}


def scenario_longrun_stability(mode: SigmaMode, seed: int = 14) -> Dict[str, float]:
    np.random.seed(seed)
    params = make_params(mode, N=24)
    params.angular_momentum_enabled = False
    sim = DETCollider3D(params)
    apply_mode_sigma(sim, mode)

    c = params.N // 2
    sim.add_packet((c, c, c - 4), mass=3.5, width=2.0, momentum=(0, 0, 0.10), initial_q=0.45)
    sim.add_packet((c, c, c + 4), mass=3.5, width=2.0, momentum=(0, 0, -0.10), initial_q=0.45)

    ok = True
    for _ in range(4000):
        _step(sim, mode)
        if not np.isfinite(sim.F).all() or not np.isfinite(sim.P).all() or not np.isfinite(sim.sigma).all():
            ok = False
            break

    return {
        "passed": float(ok),
        "F_max": float(np.max(sim.F)),
        "P_mean": float(np.mean(sim.P)),
        "sigma_mean": float(np.mean(sim.sigma)),
        "q_mean": float(np.mean(sim.q)),
    }


def scenario_c1_trajectory(mode: SigmaMode, seed: int = 15) -> Dict[str, float]:
    np.random.seed(seed)
    params = make_params(mode, N=20)
    params.angular_momentum_enabled = False
    sim = DETCollider3D(params)
    apply_mode_sigma(sim, mode)

    c = params.N // 2
    sim.add_packet((c, c, c - 4), mass=4.0, width=2.2, momentum=(0, 0, 0.10), initial_q=0.55)
    sim.add_packet((c, c, c + 4), mass=4.0, width=2.2, momentum=(0, 0, -0.10), initial_q=0.55)

    _, _, x = np.indices((params.N, params.N, params.N))
    mask_a = x <= (params.N // 2 - 2)
    mask_b = x >= (params.N // 2 + 1)

    # Regime split for interpretable trajectory readouts.
    sim.C_X[mask_a] = 0.80
    sim.C_Y[mask_a] = 0.80
    sim.C_Z[mask_a] = 0.80
    sim.C_X[mask_b] = 0.55
    sim.C_Y[mask_b] = 0.55
    sim.C_Z[mask_b] = 0.55
    sim.q[mask_a] = np.minimum(sim.q[mask_a], 0.25)
    sim.q[mask_b] = np.maximum(sim.q[mask_b], 0.40)

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
        ("Full", 1.00, 1.00),
    ]

    stage_metrics = {}
    for stage, u_a, u_b in stages:
        for _ in range(120):
            _step(sim, mode)
        a = module.compute_regime_state("A", mask_a, U=u_a)
        b = module.compute_regime_state("B", mask_b, U=u_b)
        p = module.compute_path_state("A", "B", mask_a, mask_b, U_a=u_a, U_b=u_b)
        stage_metrics[stage] = {
            "P_eff_A": a.P_eff,
            "P_eff_B": b.P_eff,
            "W_A": a.W,
            "W_B": b.W,
            "Gamma": p.Gamma,
            "Acc": p.nonverbal_accuracy,
        }

    frag = stage_metrics["Fragmented"]
    full = stage_metrics["Full"]
    return {
        "dP_eff_A": float(full["P_eff_A"] - frag["P_eff_A"]),
        "dP_eff_B": float(full["P_eff_B"] - frag["P_eff_B"]),
        "dGamma": float(full["Gamma"] - frag["Gamma"]),
        "dAcc": float(full["Acc"] - frag["Acc"]),
        "dW_A": float(full["W_A"] - frag["W_A"]),
        "dW_B": float(full["W_B"] - frag["W_B"]),
    }


def bench_step_time(mode: SigmaMode, seed: int = 17) -> float:
    np.random.seed(seed)
    params = make_params(mode, N=24)
    params.angular_momentum_enabled = False
    sim = DETCollider3D(params)
    c = params.N // 2
    sim.add_packet((c, c, c), mass=3.0, width=2.0, initial_q=0.35)
    apply_mode_sigma(sim, mode)

    # Warmup
    for _ in range(30):
        _step(sim, mode)
    t0 = time.perf_counter()
    n = 300
    for _ in range(n):
        _step(sim, mode)
    return (time.perf_counter() - t0) / n


def run_sigma_study():
    modes = [
        SigmaMode("core_dynamic", use_core_dynamic=True, use_external_policy=False),
        SigmaMode("frozen", use_core_dynamic=False, use_external_policy=False),
        SigmaMode("external_local_derived", use_core_dynamic=False, use_external_policy=True),
    ]

    print("\nDET v6.3 Sigma Factor Study")
    print("=" * 120)
    print("All runs include q-mutable exploration path in core parameters.")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mode in modes:
        mode_results: Dict[str, Dict[str, float]] = {}
        mode_results["binding"] = scenario_binding(mode)
        mode_results["mass"] = scenario_mass_conservation(mode)
        mode_results["time_dilation"] = scenario_time_dilation(mode)
        mode_results["longrun"] = scenario_longrun_stability(mode)
        mode_results["c1"] = scenario_c1_trajectory(mode)
        mode_results["perf"] = {"step_time_s": bench_step_time(mode)}
        results[mode.name] = mode_results

        print(f"\nMode: {mode.name}")
        print("-" * 120)
        print(
            f"binding(pass={int(mode_results['binding']['passed'])}, "
            f"min_sep={mode_results['binding']['min_sep']:.3f}, "
            f"final_sep={mode_results['binding']['final_sep']:.3f})"
        )
        print(
            f"mass(pass={int(mode_results['mass']['passed'])}, "
            f"effective_drift={mode_results['mass']['effective_drift']:.6f})"
        )
        print(
            f"time_dilation(pass={int(mode_results['time_dilation']['passed'])}, "
            f"dilation_factor={mode_results['time_dilation']['dilation_factor']:.6f})"
        )
        print(
            f"longrun(pass={int(mode_results['longrun']['passed'])}, "
            f"F_max={mode_results['longrun']['F_max']:.4f}, "
            f"P_mean={mode_results['longrun']['P_mean']:.6f}, "
            f"sigma_mean={mode_results['longrun']['sigma_mean']:.6f}, "
            f"q_mean={mode_results['longrun']['q_mean']:.6f})"
        )
        print(
            f"C1 deltas: dP_A={mode_results['c1']['dP_eff_A']:.6f}, "
            f"dP_B={mode_results['c1']['dP_eff_B']:.6f}, "
            f"dGamma={mode_results['c1']['dGamma']:.6f}, "
            f"dAcc={mode_results['c1']['dAcc']:.6f}, "
            f"dW_A={mode_results['c1']['dW_A']:.6f}, "
            f"dW_B={mode_results['c1']['dW_B']:.6f}"
        )
        print(f"perf step_time={mode_results['perf']['step_time_s']:.6f}s")

    # Relative comparisons vs core_dynamic baseline.
    base = results["core_dynamic"]
    print("\nRelative to core_dynamic")
    print("-" * 120)
    for mode_name in ["frozen", "external_local_derived"]:
        cur = results[mode_name]
        print(
            f"{mode_name}: "
            f"dilation_delta={cur['time_dilation']['dilation_factor'] - base['time_dilation']['dilation_factor']:.6f}, "
            f"dP_B_delta={cur['c1']['dP_eff_B'] - base['c1']['dP_eff_B']:.6f}, "
            f"dGamma_delta={cur['c1']['dGamma'] - base['c1']['dGamma']:.6f}, "
            f"drift_delta={cur['mass']['effective_drift'] - base['mass']['effective_drift']:.6f}, "
            f"step_time_ratio={cur['perf']['step_time_s'] / (base['perf']['step_time_s'] + 1e-12):.3f}"
        )

    return results


if __name__ == "__main__":
    run_sigma_study()
