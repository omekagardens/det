"""
Deep sigma study for DET v6.3 q-mutable branch.

Compares:
1) core sigma behavior
2) sigma-removed behavior (sigma forced to 1.0 in 3D collider)

Coverage:
- Comprehensive falsifiers (v6.3)
- v6.5 falsifier suite
- v6.5b falsifier suite
- Quantum-classical analysis
- Gravity calibrations (G extraction, lensing, cosmological, black hole)
- SI unit/direct classical mapping tests
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "calibration"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


@dataclass(frozen=True)
class SigmaStudyMode:
    name: str
    sigma_removed: bool


class SigmaRemovedContext:
    """
    Force sigma=1.0 in DETCollider3D by patching __init__ and step.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._orig_init = None
        self._orig_step = None

    def __enter__(self):
        if not self.enabled:
            return self

        self._orig_init = DETCollider3D.__init__
        self._orig_step = DETCollider3D.step

        def init_wrapped(this, *args, **kwargs):
            self._orig_init(this, *args, **kwargs)
            this.p.sigma_dynamic = False
            this.sigma[:] = 1.0

        def step_wrapped(this, *args, **kwargs):
            this.sigma[:] = 1.0
            out = self._orig_step(this, *args, **kwargs)
            this.sigma[:] = 1.0
            return out

        DETCollider3D.__init__ = init_wrapped
        DETCollider3D.step = step_wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        DETCollider3D.__init__ = self._orig_init
        DETCollider3D.step = self._orig_step
        return False


def _safe_call(name: str, fn: Callable[[], Any], silence: bool = True) -> Tuple[bool, Any, float, Optional[str]]:
    t0 = time.perf_counter()
    try:
        if silence:
            with contextlib.redirect_stdout(io.StringIO()):
                out = fn()
        else:
            out = fn()
        elapsed = time.perf_counter() - t0
        return True, out, elapsed, None
    except Exception as exc:  # pragma: no cover - diagnostic path
        elapsed = time.perf_counter() - t0
        return False, None, elapsed, f"{type(exc).__name__}: {exc}"


def sigma_activity_probe(mode: SigmaStudyMode, steps: int = 800) -> Dict[str, float]:
    """
    Measure how much sigma actually moves under dynamic mode.
    """
    p = DETParams3D(
        N=24,
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
        sigma_dynamic=not mode.sigma_removed,
    )
    sim = DETCollider3D(p)
    c = p.N // 2
    sim.add_packet((c, c, c - 4), mass=4.5, width=2.2, momentum=(0, 0, 0.15), initial_q=0.5)
    sim.add_packet((c, c, c + 4), mass=4.5, width=2.2, momentum=(0, 0, -0.15), initial_q=0.5)
    if mode.sigma_removed:
        sim.sigma[:] = 1.0

    sigmas = []
    for _ in range(steps):
        if mode.sigma_removed:
            sim.sigma[:] = 1.0
        sim.step()
        if mode.sigma_removed:
            sim.sigma[:] = 1.0
        sigmas.append(float(np.mean(sim.sigma)))

    arr = np.asarray(sigmas, dtype=float)
    return {
        "sigma_mean_mean": float(np.mean(arr)),
        "sigma_mean_std_over_time": float(np.std(arr)),
        "sigma_min": float(np.min(arr)),
        "sigma_max": float(np.max(arr)),
    }


def _count_passed(results: Dict[str, Any]) -> Tuple[int, int]:
    passed = 0
    total = 0
    for _, value in results.items():
        if isinstance(value, dict) and "passed" in value:
            ok = bool(value["passed"])
        else:
            ok = bool(value)
        passed += int(ok)
        total += 1
    return passed, total


def run_mode(mode: SigmaStudyMode) -> Dict[str, Any]:
    # Delayed imports to ensure patched class methods are active.
    import det_comprehensive_falsifiers as comp
    import det_v65_falsifiers as v65
    import det_v65b_falsifiers as v65b
    import test_si_units as si_map

    from extract_g_calibration import run_g_calibration
    from gravitational_lensing import run_lensing_analysis
    from cosmological_scaling import run_cosmological_analysis
    from black_hole_thermodynamics import run_black_hole_analysis
    from quantum_classical_transition import run_quantum_classical_analysis

    mode_out: Dict[str, Any] = {"mode": mode.name, "sigma_removed": mode.sigma_removed}

    with SigmaRemovedContext(enabled=mode.sigma_removed):
        ok, out, elapsed, err = _safe_call(
            "falsifier_comprehensive",
            lambda: comp.run_comprehensive_test_suite(include_option_b=False),
            silence=True,
        )
        mode_out["falsifier_comprehensive"] = {
            "ok": ok,
            "runtime_s": elapsed,
            "error": err,
        }
        if ok:
            p, t = _count_passed(out)
            mode_out["falsifier_comprehensive"]["passed"] = p
            mode_out["falsifier_comprehensive"]["total"] = t

        ok, out, elapsed, err = _safe_call(
            "falsifier_v65",
            lambda: v65.run_v65_falsifier_suite(),
            silence=True,
        )
        mode_out["falsifier_v65"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            p, t = _count_passed(out)
            mode_out["falsifier_v65"]["passed"] = p
            mode_out["falsifier_v65"]["total"] = t

        ok, out, elapsed, err = _safe_call(
            "falsifier_v65b",
            lambda: v65b.run_v65b_falsifier_suite(),
            silence=True,
        )
        mode_out["falsifier_v65b"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            p, t = _count_passed(out)
            mode_out["falsifier_v65b"]["passed"] = p
            mode_out["falsifier_v65b"]["total"] = t

        ok, out, elapsed, err = _safe_call("si_units", lambda: si_map.run_all_tests(), silence=True)
        mode_out["si_units"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            mode_out["si_units"]["passed"] = bool(out)

        ok, out, elapsed, err = _safe_call(
            "qm_transition",
            lambda: run_quantum_classical_analysis(
                grid_size=16, C_init_values=[0.2, 0.5, 0.8], verbose=False
            ),
            silence=True,
        )
        mode_out["qm_transition"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            rs = out["regime_summary"]
            mode_out["qm_transition"]["quantum_count"] = int(rs["quantum_count"])
            mode_out["qm_transition"]["classical_count"] = int(rs["classical_count"])
            mode_out["qm_transition"]["transition_count"] = int(rs["transition_count"])

        ok, out, elapsed, err = _safe_call(
            "g_calibration",
            lambda: run_g_calibration(grid_size=24, kappa=5.0, verbose=False),
            silence=True,
        )
        mode_out["g_calibration"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            mode_out["g_calibration"]["calibration_passed"] = bool(out.calibration_passed)
            mode_out["g_calibration"]["G_orbital_mean"] = float(out.G_orbital_mean)
            mode_out["g_calibration"]["G_potential_mean"] = float(out.G_potential_mean)
            mode_out["g_calibration"]["G_theoretical"] = float(out.G_theoretical)

        ok, out, elapsed, err = _safe_call(
            "lensing",
            lambda: run_lensing_analysis(grid_size=24, mass=35.0, verbose=False),
            silence=True,
        )
        mode_out["lensing"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            mode_out["lensing"]["mean_relative_error"] = float(np.mean(out.relative_errors))
            mode_out["lensing"]["einstein_radius"] = float(out.einstein_radius)

        ok, out, elapsed, err = _safe_call(
            "cosmology",
            lambda: run_cosmological_analysis(
                grid_size=24, kappa=5.0, growth_steps=120, verbose=False
            ),
            silence=True,
        )
        mode_out["cosmology"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            mode_out["cosmology"]["growth_rate"] = float(out.growth.growth_rate)
            mode_out["cosmology"]["spectral_index"] = float(out.power_spectrum.spectral_index)
            mode_out["cosmology"]["correlation_length"] = float(out.correlation.correlation_length)

        ok, out, elapsed, err = _safe_call(
            "black_hole",
            lambda: run_black_hole_analysis(
                grid_size=24, kappa=5.0, masses=[20.0, 30.0, 45.0], verbose=False
            ),
            silence=True,
        )
        mode_out["black_hole"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            cmp = out["comparison"]
            mode_out["black_hole"]["agreement_temperature"] = float(cmp.agreement_temperature)
            mode_out["black_hole"]["agreement_entropy"] = float(cmp.agreement_entropy)
            mode_out["black_hole"]["T_exponent_det"] = float(cmp.T_exponent_det)
            mode_out["black_hole"]["entropy_exponent_det"] = float(cmp.entropy_exponent_det)

        ok, out, elapsed, err = _safe_call(
            "sigma_activity",
            lambda: sigma_activity_probe(mode, steps=800),
            silence=True,
        )
        mode_out["sigma_activity"] = {"ok": ok, "runtime_s": elapsed, "error": err}
        if ok:
            mode_out["sigma_activity"].update(out)

    return mode_out


def print_summary(results: Dict[str, Dict[str, Any]]):
    print("\nDET Sigma Deep Study Summary")
    print("=" * 120)
    for mode_name, out in results.items():
        print(f"\nMode: {mode_name} (sigma_removed={out['sigma_removed']})")
        for key in [
            "falsifier_comprehensive",
            "falsifier_v65",
            "falsifier_v65b",
            "si_units",
            "qm_transition",
            "g_calibration",
            "lensing",
            "cosmology",
            "black_hole",
            "sigma_activity",
        ]:
            item = out[key]
            if not item["ok"]:
                print(f"  {key:24s} ERROR: {item['error']} ({item['runtime_s']:.1f}s)")
                continue
            line = f"  {key:24s} ok ({item['runtime_s']:.1f}s)"
            if "passed" in item and "total" in item:
                line += f" pass={item['passed']}/{item['total']}"
            elif "passed" in item and isinstance(item["passed"], bool):
                line += f" pass={int(item['passed'])}"
            print(line)

    # Basic deltas for quick read.
    if "core" in results and "sigma_removed" in results:
        core = results["core"]
        rem = results["sigma_removed"]
        print("\nCore vs Sigma-Removed Deltas")
        print("-" * 120)
        if core["g_calibration"]["ok"] and rem["g_calibration"]["ok"]:
            d = rem["g_calibration"]["G_orbital_mean"] - core["g_calibration"]["G_orbital_mean"]
            print(f"  delta G_orbital_mean: {d:+.6f}")
        if core["lensing"]["ok"] and rem["lensing"]["ok"]:
            d = rem["lensing"]["mean_relative_error"] - core["lensing"]["mean_relative_error"]
            print(f"  delta lensing mean relative error: {d:+.6f}")
        if core["cosmology"]["ok"] and rem["cosmology"]["ok"]:
            d = rem["cosmology"]["growth_rate"] - core["cosmology"]["growth_rate"]
            print(f"  delta cosmology growth_rate: {d:+.6f}")
        if core["black_hole"]["ok"] and rem["black_hole"]["ok"]:
            d = rem["black_hole"]["agreement_temperature"] - core["black_hole"]["agreement_temperature"]
            print(f"  delta BH agreement_temperature: {d:+.6f}")


def main():
    modes = [
        SigmaStudyMode(name="core", sigma_removed=False),
        SigmaStudyMode(name="sigma_removed", sigma_removed=True),
    ]
    results = {}
    for mode in modes:
        results[mode.name] = run_mode(mode)

    print_summary(results)

    out_path = Path("det_v6_3/reports/det_sigma_deep_study_results_2026_03_06.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote raw results to {out_path}")


if __name__ == "__main__":
    main()
