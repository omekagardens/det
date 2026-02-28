# DET v7 Refactor Report

**Date:** 2026-02-28  
**Branch intent:** Agency-First invariance + structural drag baseline

## 1. Scope Completed
- Created new model family directory: `det_v7_0/` (copied from `det_v6_3/` then refactored).
- Added canonical `det_theory_card_6_5_1.md` to both:
  - `det_v6_3/docs/`
  - `det_v7_0/docs/`
- Marked `det_theory_card_6_3.md` and `det_theory_card_6_5.md` as legacy/superseded.
- Added migration/deprecation map:
  - `det_v6_3/docs/v7_migration_deprecation_map.md`
  - `det_v7_0/docs/v7_migration_deprecation_map.md`

## 2. Core Dynamics Refactor (`det_v7_0/src`)
Updated colliders:
- `det_v6_3_1d_collider.py`
- `det_v6_3_2d_collider.py`
- `det_v6_3_3d_collider.py`
- `det_v6_3_collider_torch.py`

Canonical law changes integrated:
- Debt decomposition with retained `q_I`, `q_D`, and compatibility `q` readout.
- Presence drag:
  - `D = 1/(1 + lambda_DP*q_D + lambda_IP*q_I)`
  - `P = base_presence * D`
- Agency update uses:
  - `a^+ = clip(a + beta_a*(a0-a) + gamma(C)*(P-P_neighbors) + xi, 0, 1)`
- Structural ceilings removed from canonical update path.
- Jubilee constrained to `q_D` only.
- Legacy `q` write compatibility added (mapped into `(q_I, q_D)`).

## 3. Architecture Separation
Added explicit canonical core modules in `det_v7_0/core`:
- `presence.py`
- `gravity.py`
- `agency.py`
- `structure.py`
- `flow.py`
- `boundary.py`
- `update_loop.py`

## 4. Falsifier/Test Updates (`det_v7_0/tests`)
Added mandatory v6.5.1/v7 falsifiers:
- `test_f_a2_prime_no_structural_suppression.py`
- `test_f_a4_frozen_will.py`
- `test_f_a5_runaway_agency_sweep.py`
- `test_gtd5_prime_drag_clock_ratio.py`
- `test_bh_drag_scaling_3d.py`
- Runner: `det_v651_falsifiers.py`

## 5. Calibration/EM
- Updated `det_v7_0/calibration/black_hole_thermodynamics.py` for drag-inclusive presence semantics.
- Added legacy notice to `det_v7_0/calibration/quantum_classical_transition.py`.
- Updated EM module helper for drag-aware presence coupling:
  - `det_v7_0/src/det_em/det_em_v6_3.py`

## 6. Validation Execution Status
### Completed
- Syntax validation via `python3 -m py_compile` / `compileall` across changed `src`, `core`, `tests`, `calibration`.

### Blocked
Runtime falsifiers/tests/colliders could not be executed in this environment because required Python dependencies are unavailable and network package install is blocked:
- missing `numpy`/`scipy`/`pytest`
- `pip install` fails due offline index access

## 7. Risk Notes
- Legacy calibration modules and narrative docs are retained for traceability; several still contain pre-v6.5.1 ceiling language and are explicitly marked historical in migration docs.
- Canonical claims should be tied to v7 falsifier outputs once dependencies are available and runs are executed.
- Additional legacy references to `lambda_a` remain in non-canonical application/demo scripts and historical falsifier suites under `det_v7_0/src` and `det_v7_0/tests`; these are now explicitly categorized in `v7_migration_deprecation_map.md`.
