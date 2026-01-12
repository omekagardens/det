"""
DET v6 Comprehensive Test Suite
================================

This script runs all falsifier tests across 1D, 2D, and 3D colliders
and generates a consolidated report.

Reference: DET Theory Card v6.0
"""

import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_1d_tests():
    """Run 1D collider tests."""
    print("\n" + "="*70)
    print("1D COLLIDER TESTS")
    print("="*70)
    
    from det_v6_1d_collider import (
        test_F7_mass_conservation,
        test_F8_vacuum_momentum,
        test_F9_symmetry_drift,
        run_collision_test
    )
    
    results = {
        'collision': run_collision_test(verbose=True),
        'F7': test_F7_mass_conservation(verbose=True),
        'F8': test_F8_vacuum_momentum(verbose=True),
        'F9': test_F9_symmetry_drift(verbose=True)
    }
    
    return results


def run_2d_tests():
    """Run 2D collider tests."""
    print("\n" + "="*70)
    print("2D COLLIDER TESTS")
    print("="*70)
    
    from det_v6_2d_collider import (
        test_F8_vacuum_momentum,
        test_F9_symmetry_drift,
        run_collision_test,
        DETParams
    )
    
    # Run main collision test
    params = DETParams()
    print(f"\nParameters:")
    print(f"  DT={params.DT}, N={params.N}, R={params.R}")
    print(f"  Momentum: α_π={params.alpha_pi}, λ_π={params.lambda_pi}")
    
    results = {
        'collision': run_collision_test(params, steps=6000, verbose=True),
        'F8': test_F8_vacuum_momentum(verbose=True),
        'F9': test_F9_symmetry_drift(verbose=True)
    }
    
    return results


def run_3d_tests():
    """Run 3D collider tests."""
    print("\n" + "="*70)
    print("3D COLLIDER TESTS")
    print("="*70)
    
    from det_v6_3d_collider import (
        test_F_L1_rotational_flux_conservation,
        test_F_L2_vacuum_spin_no_transport,
        DETParams3D
    )
    
    results = {
        'F_L1': test_F_L1_rotational_flux_conservation(verbose=True),
        'F_L2': test_F_L2_vacuum_spin_no_transport(verbose=True)
    }
    
    return results


def generate_report(results_1d, results_2d, results_3d, runtime):
    """Generate a comprehensive test report."""
    
    report = []
    report.append("# DET v6 Comprehensive Test Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Runtime:** {runtime:.1f} seconds")
    
    report.append("\n## Summary")
    report.append("\n| Dimension | Test | Result |")
    report.append("|-----------|------|--------|")
    
    # 1D results
    report.append(f"| 1D | Collision | {'PASS' if results_1d['collision']['collision'] else 'FAIL'} |")
    report.append(f"| 1D | F7 (Mass Conservation) | {'PASS' if results_1d['F7'] else 'FAIL'} |")
    report.append(f"| 1D | F8 (Vacuum Momentum) | {'PASS' if results_1d['F8'] else 'FAIL'} |")
    report.append(f"| 1D | F9 (Symmetry Drift) | {'PASS' if results_1d['F9'] else 'FAIL'} |")
    
    # 2D results
    report.append(f"| 2D | Collision | {'PASS' if results_2d['collision']['collision'] else 'FAIL'} |")
    report.append(f"| 2D | F8 (Vacuum Momentum) | {'PASS' if results_2d['F8'] else 'FAIL'} |")
    report.append(f"| 2D | F9 (Symmetry Drift) | {'PASS' if results_2d['F9'] else 'FAIL'} |")
    
    # 3D results
    report.append(f"| 3D | F_L1 (Rotational Conservation) | {'PASS' if results_3d['F_L1']['passed'] else 'FAIL'} |")
    report.append(f"| 3D | F_L2 (Vacuum Spin) | {'PASS' if results_3d['F_L2']['passed'] else 'FAIL'} |")
    
    # Detailed results
    report.append("\n## Detailed Results")
    
    report.append("\n### 1D Collider")
    report.append(f"\n**Collision Test:**")
    report.append(f"- Min separation: {results_1d['collision']['min_sep']:.1f}")
    report.append(f"- Final mass error: {results_1d['collision']['final_mass_err']:+.3f}%")
    
    report.append("\n### 2D Collider")
    report.append(f"\n**Collision Test:**")
    report.append(f"- Min separation: {results_2d['collision']['min_sep']:.1f}")
    report.append(f"- Final mass error: {results_2d['collision']['final_mass_err']:+.3f}%")
    report.append(f"- Max q reached: {max(results_2d['collision']['q_max']):.4f}")
    report.append(f"- Min agency reached: {min(results_2d['collision']['min_a']):.4f}")
    
    report.append("\n### 3D Collider")
    report.append(f"\n**F_L1 (Rotational Flux Conservation):**")
    report.append(f"- Mass error: {results_3d['F_L1']['mass_err']:.2e}")
    report.append(f"- Max COM drift: {results_3d['F_L1']['max_drift']:.6f} cells")
    
    report.append(f"\n**F_L2 (Vacuum Spin No Transport):**")
    report.append(f"- Scaling OK: {results_3d['F_L2']['scaling_ok']}")
    report.append(f"- Mass conservation OK: {results_3d['F_L2']['mass_ok']}")
    
    # Falsifier coverage
    report.append("\n## Falsifier Coverage")
    report.append("\n| ID | Description | Tested | Status |")
    report.append("|:---|:---|:---:|:---:|")
    report.append("| F1 | Locality Violation | No | - |")
    report.append("| F2 | Coercion | No | - |")
    report.append("| F3 | Boundary Redundancy | No | - |")
    report.append("| F4 | No Regime Transition | No | - |")
    report.append("| F5 | Hidden Global Aggregates | No | - |")
    report.append("| F6 | Binding Failure | Partial | - |")
    report.append(f"| F7 | Mass Non-Conservation | Yes | {'PASS' if results_1d['F7'] else 'FAIL'} |")
    report.append(f"| F8 | Momentum Pushes Vacuum | Yes | {'PASS' if results_1d['F8'] and results_2d['F8'] else 'FAIL'} |")
    report.append(f"| F9 | Spontaneous Drift | Yes | {'PASS' if results_1d['F9'] and results_2d['F9'] else 'FAIL'} |")
    report.append("| F10 | Regime Discontinuity | No | - |")
    report.append(f"| F_L1 | Rotational Conservation | Yes | {'PASS' if results_3d['F_L1']['passed'] else 'FAIL'} |")
    report.append(f"| F_L2 | Vacuum Spin Transport | Yes | {'PASS' if results_3d['F_L2']['passed'] else 'FAIL'} |")
    report.append("| F_L3 | Orbital Capture Failure | No | - |")
    
    return "\n".join(report)


def main():
    """Run all tests and generate report."""
    print("="*70)
    print("DET v6 COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    start_time = time.time()
    
    # Run tests
    results_1d = run_1d_tests()
    results_2d = run_2d_tests()
    results_3d = run_3d_tests()
    
    runtime = time.time() - start_time
    
    # Generate report
    report = generate_report(results_1d, results_2d, results_3d, runtime)
    
    # Save report
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    report_path = os.path.join(results_dir, 'test_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n\nReport saved to: {report_path}")
    print(f"Total runtime: {runtime:.1f}s")
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    all_passed = (
        results_1d['collision']['collision'] and
        results_1d['F7'] and results_1d['F8'] and results_1d['F9'] and
        results_2d['collision']['collision'] and
        results_2d['F8'] and results_2d['F9'] and
        results_3d['F_L1']['passed'] and results_3d['F_L2']['passed']
    )
    
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - See report for details")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
