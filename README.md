# Deep Existence Theory (DET) Repository

## Active Model Families
- `det_v6_3/`: Legacy baseline and historical artifacts.
- `det_v7_0/`: Refactored Agency-First + structural-drag baseline.

## Canonical Model Card
- `det_v7_0/docs/det_model_v7_0.md`

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

## Mandatory v6.5.1/v7 Falsifiers
Implemented in `det_v7_0/tests`:
- `test_f_a2_prime_no_structural_suppression.py`
- `test_f_a4_frozen_will.py`
- `test_f_a5_runaway_agency_sweep.py`
- `test_gtd5_prime_drag_clock_ratio.py`
- `test_bh_drag_scaling_3d.py`
- runner: `det_v651_falsifiers.py`

## Notes
- Calibration and validation modules remain readout layers.
- Legacy reports and docs are preserved; use `v7_migration_deprecation_map.md` for status.
