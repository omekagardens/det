# DET v7 Consistency Review (Updated 2026-03-03)

## Scope Reviewed
- `det_v7_0/core`
- `det_v7_0/src`
- `det_v7_0/calibration`
- `det_v7_0/tests`
- `det_v7_0/validation`
- `det_v7_0/docs`

## Canonical Decision

Canonical model reference is:
- `det_v7_0/docs/det_theory_card_7_0.md`

Model status:
- v7 is now canonical with **mutable combined `q`**.
- dual-debt split (`q_I/q_D`) is removed from active canonical code path.

## Module Consistency Summary

- Core/colliders: aligned with unified `q`, `lambda_P`, and no structural agency ceiling.
- Boundary semantics: Jubilee reduces total `q` under local energy/gating constraints and remains agency-safe.
- Calibration:
  - `black_hole_thermodynamics.py` aligned to unified drag model.
  - `quantum_classical_transition.py` aligned to unified drag/agency analysis.
- Tests/falsifiers:
  - mandatory mutable-`q` gate set present and passing,
  - legacy falsifier suites retained for regression archaeology.

## Test and Validation Snapshot (March 3, 2026)

- `det_v651_falsifiers.py`: 10/10 passed
- `det_v65_falsifiers.py`: 8/8 passed
- `det_v65b_falsifiers.py`: 4/4 passed
- `det_comprehensive_falsifiers.py`: 15/15 passed
- `pytest det_v7_0/tests -q`: 204 passed
- Bell/gravity-focused pytest slice: 50 passed
- `det_validation_harness.py --all`: 6/6 passed

No failing assertions were observed. Existing warnings are from legacy-style tests returning values instead of assertions.

## Documentation Consistency Summary

- Canonical model card and v7 application docs now reflect unified mutable `q` semantics.
- Root README points to the canonical unified card.
- v6.x cards remain available as historical references only.

## Conclusion

Repository-level canonical dynamics, falsifiers, colliders, calibration, and validation harnesses are consistent with DET v7 mutable combined-`q` baseline as of March 3, 2026.
