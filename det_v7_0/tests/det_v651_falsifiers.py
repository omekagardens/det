"""DET v6.5.1/v7 mandatory falsifier runner."""

import importlib
import traceback


TEST_TARGETS = [
    ("F_A2'", "test_f_a2_prime_no_structural_suppression", "test_f_a2_prime_no_structural_suppression"),
    ("F_A4", "test_f_a4_frozen_will", "test_f_a4_frozen_will_long_run"),
    ("F_A5", "test_f_a5_runaway_agency_sweep", "test_f_a5_runaway_agency_sweep"),
    ("F_GTD5'", "test_gtd5_prime_drag_clock_ratio", "test_gtd5_prime_drag_clock_ratio"),
    ("F_BH-Drag-3D", "test_bh_drag_scaling_3d", "test_bh_drag_scaling_3d"),
]


def run_v651_falsifier_suite(verbose: bool = True):
    results = {}
    for name, module_name, fn_name in TEST_TARGETS:
        try:
            module = importlib.import_module(module_name)
            getattr(module, fn_name)()
            results[name] = {"passed": True}
            if verbose:
                print(f"{name:15s}: PASS")
        except Exception as exc:  # pragma: no cover - diagnostic path
            results[name] = {"passed": False, "error": str(exc)}
            if verbose:
                print(f"{name:15s}: FAIL")
                traceback.print_exc()

    passed = sum(1 for r in results.values() if r.get("passed"))
    total = len(results)
    if verbose:
        print(f"\nSummary: {passed}/{total} mandatory falsifiers passed")
    return results


if __name__ == "__main__":
    run_v651_falsifier_suite(verbose=True)

