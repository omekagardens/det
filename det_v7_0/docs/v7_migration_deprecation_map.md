# DET v7 Migration and Deprecation Map (Mutable Combined-`q`)

**Date:** March 3, 2026
**Scope reviewed:** `/docs`, `/core`, `/src`, `/tests`, `/validation`, `/calibration`

## 1) Canonical Documentation

- **Canonical card:** `det_v7_0/docs/det_theory_card_7_0.md`
- **Alias redirect:** `det_v7_0/docs/det_model_v7_0.md`

Legacy historical cards (superseded for active core law):
- `det_theory_card_6_3.md`
- `det_theory_card_6_5.md`
- `det_theory_card_6_5_1.md`

## 2) Canonical Core Law Status

Active v7 canonical model now uses:
- single mutable `q in [0,1]`
- drag coupling `lambda_P`
- structure accumulation `alpha_q`
- lawful local recovery/Jubilee on total `q`
- no structural agency ceiling

## 3) Removed from Canonical Path

- dual debt state (`q_I`, `q_D`)
- dual drag couplings (`lambda_IP`, `lambda_DP`)
- split accumulation (`alpha_qI`, `alpha_qD`)
- `lambda_a`-driven structural ceiling logic (`a_max` path)

## 4) Codebase Consistency Summary

### 4.1 Core/Source
`det_v7_0/core` and `det_v7_0/src` colliders are aligned with mutable combined-`q` law:
- 1D, 2D, 3D, Torch colliders updated
- EM helper updated to unified drag input
- unified parameter schema updated to `lambda_P` + `alpha_q`

### 4.2 Calibration
Readout modules remain active and aligned with unified `q`:
- `black_hole_thermodynamics.py`
- `quantum_classical_transition.py`
- gravity/lensing/rotation/cosmology calibration modules

### 4.3 Tests and Falsifiers
Mandatory gates include:
- `F_A2'`, `F_A4`, `F_A5`
- `F_QM1`..`F_QM5`
- `F_GTD5'`
- `F_BH-Drag-3D`

Regression/historical suites retained:
- `det_v65_falsifiers.py`
- `det_v65b_falsifiers.py`
- `det_comprehensive_falsifiers.py`

## 5) Validation/Physics Readout Status

Validation harness (`det_validation_harness.py --all`) remains passing with gravity/Kepler/Bell checks.
Bell and gravity-focused pytest slices remain passing under unified mutable-`q` implementation.

## 6) Documentation Classification

### Canonical operational docs
- `det_theory_card_7_0.md`
- `det_v7_applications_review.md`
- `det_v7_battery_storage_recovery_review.md`

### Historical/contextual docs (non-canonical dynamics)
Narrative/theological/legacy-interpretive documents remain archived for traceability and must not override canonical equations.

## 7) Migration Rule for Legacy Saved States

If legacy split states are encountered:
- map `q <- clip(q_I + q_D, 0, 1)`
- drop split channels in canonical runtime state

## 8) Acceptance Gate Snapshot (March 3, 2026)

- mandatory mutable-`q` falsifiers: pass
- legacy/comprehensive falsifiers: pass
- full `pytest det_v7_0/tests -q`: pass
- validation harness: pass

Repository is aligned with mutable combined-`q` DET v7 canonical model.
