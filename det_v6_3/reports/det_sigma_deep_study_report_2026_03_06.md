# DET v6.3 Deep Sigma Study Report

**Date:** 2026-03-06  
**Branch:** `codex/v6-3-q-mutable-local-grace-exploration`  
**Runner:** `det_v6_3/tests/det_sigma_deep_study.py`  
**Raw output:** `det_v6_3/reports/det_sigma_deep_study_results_2026_03_06.json`

## 1) Objective

Evaluate whether processing factor `sigma` is structurally necessary in core dynamics, using broad coverage:

- gravity-facing calibrations,
- quantum-classical analysis,
- full falsifier harnesses,
- direct SI/classical unit mapping.

The decision target was binary:

- keep `sigma` as a core evolving field, or
- remove it from core (equivalent study mode: force `sigma=1` each step in 3D).

## 2) Modes Compared

1. **core**
   - existing behavior, including 3D `sigma_dynamic` law.
2. **sigma_removed**
   - hard override in 3D: `sigma=1.0` before and after each step.

Both modes used the same q-mutable branch settings.

## 3) Coverage Executed

For each mode:

1. `det_comprehensive_falsifiers.py` (`run_comprehensive_test_suite`)
2. `det_v65_falsifiers.py` (`run_v65_falsifier_suite`)
3. `det_v65b_falsifiers.py` (`run_v65b_falsifier_suite`)
4. `test_si_units.py` (`run_all_tests`) for direct classical/SI mapping
5. `run_quantum_classical_analysis` (grid 16, coherence sweep)
6. `run_g_calibration` (grid 24)
7. `run_lensing_analysis` (grid 24)
8. `run_cosmological_analysis` (grid 24, 120 growth steps)
9. `run_black_hole_analysis` (grid 24, masses 20/30/45)
10. sigma activity probe (measure actual sigma movement under core mode)

## 4) Key Results

## 4.1 Falsifiers

Identical pass counts in both modes:

- v6.3 comprehensive: `15/15`
- v6.5 suite: `8/8`
- v6.5b suite: `4/4`

No falsifier regressions under sigma removal.

## 4.2 Gravity / QM / Cosmology / BH / SI

All pipelines executed successfully in both modes.  
Measured deltas (sigma_removed - core):

- `delta G_orbital_mean = +0.000000`
- `delta lensing mean relative error = +0.000000`
- `delta cosmology growth_rate = +0.000000`
- `delta BH agreement_temperature = +0.000000`
- Quantum-classical regime summary identical (`1 quantum / 0 classical / 2 transition`)
- SI/direct unit map test passed in both modes.

Within this study resolution, these modules were invariant to sigma removal.

## 4.3 Sigma dynamics activity

Core mode sigma activity probe:

- `sigma_mean_mean = 1.004555`
- `sigma_mean_std_over_time = 0.002221`
- `sigma_min = 1.002000`
- `sigma_max = 1.008854`

Interpretation: in tested 3D dynamics, sigma moved only ~0.2%-0.9% around 1.0.

## 5) Interpretation

1. **Empirical role in current codepath is weak.**  
   Sigma evolution is numerically active but very small in amplitude.

2. **Broad behavior is preserved when sigma is removed from core evolution.**  
   Across all executed falsifiers and calibration/readout pipelines, outputs were effectively unchanged at reported precision.

3. **Clarity/simplicity argument is supported by evidence.**  
   Given negligible movement and no observed regressions in this deep sweep, sigma appears to function as near-constant scaling noise rather than a critical independent primitive.

## 6) Answer to Decision Question

Based on this deep study, there is strong evidence that `sigma` is **not required as an independently evolving core state variable** in the current v6.3 q-mutable branch behavior.

If the project goal is maximal clarity and minimal primitive set, the data supports removing core sigma dynamics and treating sigma as fixed/derived at use sites.

## 7) Practical Caveat

This conclusion is strictly about this codebase and tested scenarios.  
If future submodels rely on stronger sigma excursions, they should re-introduce it explicitly as an opt-in extension, not a hidden baseline dependency.
