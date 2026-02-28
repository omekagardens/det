# DET v7 Consistency Review (2026-02-28)

## Scope Reviewed
- `det_v7_0/core`
- `det_v7_0/src`
- `det_v7_0/calibration`
- `det_v7_0/tests`
- `det_v7_0/validation`
- `det_v7_0/docs` and mirrored legacy docs in `det_v6_3/docs`

## Canonical Reference Decision
- Canonical model definition is now centralized in:
  - `det_v7_0/docs/det_model_v7_0.md`
- Legacy cards are retained but explicitly marked superseded.

## Module Consistency Summary
- Core/colliders: consistent with v7 agency-first + drag path.
- Boundary semantics: Jubilee is q_D-only and agency-safe in canonical path.
- Calibration:
  - `black_hole_thermodynamics.py` aligned with v7 drag-era behavior and passing canonical BH falsifier.
  - `quantum_classical_transition.py` updated from ceiling-era logic to drag-era agency analysis.
- Tests/falsifiers:
  - Canonical v7 falsifier suite present and passing.
  - Legacy falsifier suites retained for regression archaeology.

## Document Consistency Summary
- Unified canonical card added: `det_model_v7_0.md`.
- README and migration maps now point to unified card.
- `det_theory_card_6_3.md`, `det_theory_card_6_5.md`, `det_theory_card_6_5_1.md` marked as legacy/superseded.

## Remaining Legacy/Contextual Docs
The following remain intentionally historical or application-narrative and are not canonical law definitions:
- theology/afterlife/coexisting/merging-worlds documents
- structural debt application narratives
- historical analyses/reports generated against prior version labels

These remain for traceability and should not override `det_model_v7_0.md`.

## Validation Snapshot
- `pytest det_v7_0/tests -q`: pass
- `det_v651_falsifiers.py`: pass (all mandatory)
- `det_v65_falsifiers.py`: pass
- `det_v65b_falsifiers.py`: pass
- `det_comprehensive_falsifiers.py`: pass
- `det_validation_harness.py --all`: pass

## Conclusion
Repository-level canonical dynamics and validation gates are consistent with DET v7 under a single unified model card, with legacy artifacts explicitly classified and non-canonical.
