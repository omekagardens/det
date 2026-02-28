# DET v7 Migration and Deprecation Map

**Date:** 2026-02-28  
**Scope reviewed:** `/docs`, `/src`, `/tests`, `/validation`, `/reports`

## 1. Canonical Docs
- `det_theory_card_6_5_1.md`: **Canonical baseline** for Agency-First + structural drag.
- `det_theory_card_6_3.md`: **Legacy/superseded** for core law; keep as historical reference.
- `det_theory_card_6_5.md`: **Legacy/superseded** by v6.5.1 card.

## 2. Docs to Treat as Historical/Contextual (Not Canonical Dynamics)
The following are retained as domain/application narrative and should not be used as canonical core laws:
- `det_aging_as_structural_debt.md`
- `det_debt_aging_spirit_synthesis.md`
- `det_spirit_debt_applications.md`
- `det_structural_debt.md`
- `det_structural_debt_applications.md`
- `det_resurrection_dual_mode.md`
- `theological_implications_of_det_6_3.md`
- `afterlife_and_spirit_agency_in_det.md`
- `coexisting_kingdom_in_det.md`
- `merging_worlds_technology.md`

These documents include pre-v6.5.1 agency-ceiling language and should be read as non-canonical unless explicitly updated.

## 3. Core Source Status
- `det_v6_3_1d_collider.py` in `det_v7_0/src`: **Updated to v6.5.1/v7 canonical laws**.
- `det_v6_3_2d_collider.py` in `det_v7_0/src`: **Updated to v6.5.1/v7 canonical laws**.
- `det_v6_3_3d_collider.py` in `det_v7_0/src`: **Updated to v6.5.1/v7 canonical laws**.
- `det_v6_3_collider_torch.py` in `det_v7_0/src`: **Updated to v6.5.1/v7 canonical laws**.
- `det_em/det_em_v6_3.py` in `det_v7_0/src`: updated with drag-aware presence helper.

## 4. New Core Architecture Layer (v7)
Added under `det_v7_0/core`:
- `presence.py`
- `gravity.py`
- `agency.py`
- `structure.py`
- `flow.py`
- `boundary.py`
- `update_loop.py`

This separates canonical equations from calibration/readout modules.

## 5. Test/Falsifier Status
New mandatory v6.5.1/v7 falsifier tests were added under `det_v7_0/tests`:
- `test_f_a2_prime_no_structural_suppression.py`
- `test_f_a4_frozen_will.py`
- `test_f_a5_runaway_agency_sweep.py`
- `test_gtd5_prime_drag_clock_ratio.py`
- `test_bh_drag_scaling_3d.py`
- Runner: `det_v651_falsifiers.py`

Legacy `det_v6_3/tests` suites remain for historical continuity and compatibility checks.

## 6. Validation and Reports
- Validation harnesses under `/validation` and legacy reports under `/reports` are retained as historical outputs.
- For canonical claims, use the new falsifier set above plus calibration outputs generated from `det_v7_0` colliders.

