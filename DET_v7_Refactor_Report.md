# DET v7 Refactor Report

**Date:** 2026-02-28  
**Branch intent:** Agency-First invariance + structural drag baseline

## 1. Scope Completed
- Created new model family directory: `det_v7_0/` (copied from `det_v6_3/` then refactored).
- Added unified canonical model card:
  - `det_v7_0/docs/det_model_v7_0.md`
  - pointer in legacy tree: `det_v6_3/docs/det_model_v7_0.md`
- Added `det_theory_card_6_5_1.md` patch card to both trees (now legacy patch reference):
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
- Updated `det_v7_0/calibration/quantum_classical_transition.py` to drag-era agency analysis (no structural ceiling).
- Updated EM module helper for drag-aware presence coupling:
  - `det_v7_0/src/det_em/det_em_v6_3.py`

## 6. Validation Execution Status
### Completed
- Installed runtime dependencies in `.venv`: `numpy`, `scipy`, `pytest`, `matplotlib`, `pandas`, `torch`.
- Full test suite:
  - `pytest det_v7_0/tests -q` -> **199 passed**, 0 failed.
- Mandatory v6.5.1/v7 falsifiers:
  - `python det_v7_0/tests/det_v651_falsifiers.py` -> **5/5 passed**.
- Legacy/extended falsifiers:
  - `python det_v7_0/tests/det_v65_falsifiers.py` -> **8/8 passed**.
  - `python det_v7_0/tests/det_v65b_falsifiers.py` -> **4/4 passed**.
  - `python det_v7_0/tests/det_comprehensive_falsifiers.py` -> **15/15 passed**.
- Validation harness:
  - `python det_v7_0/validation/det_validation_harness.py --all` -> **6/6 passed**.
- Additional validation scripts:
  - `gps_realistic_test.py` -> PASS
  - `hafele_keating_test.py` -> PASS
  - `gps_data_loader.py` + `bell_data_loader.py` -> executed successfully
- Collider/EM smoke:
  - 1D/2D/3D NumPy colliders, Torch collider, and EM module stepped successfully.
- Syntax validation:
  - `python -m compileall -q det_v7_0/src det_v7_0/core det_v7_0/tests det_v7_0/calibration det_v7_0/validation` -> PASS

## 7. Risk Notes
- `kepler_live_test.py` with full default live-orbit settings was computationally long in this environment and was manually interrupted during the largest radius case.
- A reduced live run (`--radii 6 8 --orbits 1 --grid 48`) completed and reported `KEPLER'S THIRD LAW: NOT SATISFIED` (CV 13.07%).
- The canonical repository validation gate remains the deterministic harness/falsifier suites above, all of which passed.
- Consistency audit document added:
  - `det_v7_0/docs/v7_consistency_review_2026_02_28.md`
