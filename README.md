# Deep Existence Theory (DET) Repository

## Active Model Families
- `det_v6_3/`: Legacy baseline and historical artifacts.
- `det_v7_0/`: Refactored Agency-First + structural-drag baseline (mutable combined `q` canonical path).

## Canonical Model Card
- `det_v7_0/docs/det_theory_card_7_0.md`

## v7 Applications Review
- `det_v7_0/docs/det_v7_applications_review.md`
- `det_v7_0/docs/det_v7_battery_storage_recovery_review.md`
- `det_v7_0/reports/grace_jubilee_energy_interaction_2026_03_03.md`

Legacy cards `det_theory_card_6_3.md`, `det_theory_card_6_5.md`, and
`det_theory_card_6_5_1.md` are retained for historical traceability and are
explicitly marked as superseded.

## v7 Core Architecture
`det_v7_0/core` provides canonical law modules:
- `presence.py`
- `gravity.py`
- `agency.py`
- `structure.py`
- `flow.py`
- `boundary.py`
- `update_loop.py`

## Mandatory v7 Falsifiers
Implemented in `det_v7_0/tests`:
- `test_f_a2_prime_no_structural_suppression.py`
- `test_f_a4_frozen_will.py`
- `test_f_a5_runaway_agency_sweep.py`
- `test_f_qm1_total_annealing_stability.py`
- `test_f_qm2_identity_persistence.py`
- `test_f_qm3_kepler_stability.py`
- `test_f_qm4_no_oscillatory_collapse.py`
- `test_f_qm5_arrow_of_time_integrity.py`
- `test_gtd5_prime_drag_clock_ratio.py`
- `test_bh_drag_scaling_3d.py`
- runner: `det_v651_falsifiers.py`

## Notes
- Calibration and validation modules remain readout layers.
- Legacy reports and docs are preserved; use `v7_migration_deprecation_map.md` for status.
