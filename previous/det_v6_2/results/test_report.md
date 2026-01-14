# DET v6 Comprehensive Test Report

**Generated:** 2026-01-11 14:10:47
**Total Runtime:** 59.5 seconds

## Summary

| Dimension | Test | Result |
|-----------|------|--------|
| 1D | Collision | FAIL |
| 1D | F7 (Mass Conservation) | PASS |
| 1D | F8 (Vacuum Momentum) | PASS |
| 1D | F9 (Symmetry Drift) | PASS |
| 2D | Collision | PASS |
| 2D | F8 (Vacuum Momentum) | PASS |
| 2D | F9 (Symmetry Drift) | PASS |
| 3D | F_L1 (Rotational Conservation) | PASS |
| 3D | F_L2 (Vacuum Spin) | PASS |

## Detailed Results

### 1D Collider

**Collision Test:**
- Min separation: 96.9
- Final mass error: +0.000%

### 2D Collider

**Collision Test:**
- Min separation: 0.0
- Final mass error: +0.132%
- Max q reached: 0.0327
- Min agency reached: 0.9690

### 3D Collider

**F_L1 (Rotational Flux Conservation):**
- Mass error: 1.20e-16
- Max COM drift: 0.002804 cells

**F_L2 (Vacuum Spin No Transport):**
- Scaling OK: True
- Mass conservation OK: True

## Falsifier Coverage

| ID | Description | Tested | Status |
|:---|:---|:---:|:---:|
| F1 | Locality Violation | No | - |
| F2 | Coercion | No | - |
| F3 | Boundary Redundancy | No | - |
| F4 | No Regime Transition | No | - |
| F5 | Hidden Global Aggregates | No | - |
| F6 | Binding Failure | Partial | - |
| F7 | Mass Non-Conservation | Yes | PASS |
| F8 | Momentum Pushes Vacuum | Yes | PASS |
| F9 | Spontaneous Drift | Yes | PASS |
| F10 | Regime Discontinuity | No | - |
| F_L1 | Rotational Conservation | Yes | PASS |
| F_L2 | Vacuum Spin Transport | Yes | PASS |
| F_L3 | Orbital Capture Failure | No | - |