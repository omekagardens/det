# Deep Existence Theory (DET) Unified Model v7.0

**Status:** Canonical model card for DET v7.0  
**Date:** 2026-02-28  
**Supersedes:** `det_theory_card_6_3.md`, `det_theory_card_6_5.md`, `det_theory_card_6_5_1.md` as canonical references.

## 1. Scope
This document is the single authoritative model definition for DET v7.0.
All core-dynamics claims, implementation behavior, and falsification gates must be consistent with this card.

## 2. Non-Negotiable Invariants
- Agency-First (A0): agency is primitive and cannot be directly suppressed by structural debt.
- Strict locality: no hidden global normalization, no disconnected-component coupling, no implicit shared mutable state.
- Boundary lawfulness: boundary operators are local; they do not directly modify agency.
- Deterministic evolution under fixed seed.

## 3. Canonical State
Per node `i`:
- `F_i`: resource field
- `q_I,i in [0,1]`: identity debt (immutable by default)
- `q_D,i in [0,1]`: dissipative debt (recoverable)
- `q_i = clip(q_I,i + q_D,i, 0, 1)` (compatibility readout only)
- `a_i in [0,1]`: agency
- `sigma_i > 0`: local conductivity/processing
- `P_i`: presence
- `Delta_tau_i`: proper-time increment

Bond/plaquette-local states (unchanged in ontology): momentum, angular momentum, coherence.

## 4. Canonical Laws
### 4.1 Presence and Drag
- `D_i = 1 / (1 + lambda_DP*q_D,i + lambda_IP*q_I,i)`
- `P_i = (a_i * sigma_i / ((1+F_i)*(1+H_i)*gamma_v)) * D_i`
- `M_i = 1 / P_i` (readout)

Debt affects expression through `P` via `D`, never by directly capping `a`.

### 4.2 Agency Update (No Structural Ceiling)
`a_i+ = clip(a_i + beta_a*(a0 - a_i) + gamma(C_i)*(P_i - mean_local(P)) + xi_i, 0, 1)`

Where:
- `gamma(C) = gamma_max * C^n`, `n >= 2`
- `xi_i` optional bounded persistence noise

### 4.3 Structure Update
- Canonical damage accrual: `q_D+ = clip(q_D + alpha_qD*max(0, -DeltaF), 0, 1)`
- Default identity update: `q_I+ = q_I`
- Any nontrivial `q_I` update must be declared as an explicit submodel.

### 4.4 Jubilee Semantics
Jubilee reduces `q_D` only; it never modifies `a`.
Interpretation: drag removal / clock-rate recovery.

## 5. Canonical Closed Update Order
1. Solve Helmholtz baseline `b`
2. Solve gravitational potential `Phi`
3. Compute `P` (including drag)
4. Compute `Delta_tau`
5. Compute fluxes: diffusive, momentum, rotational, floor, gravitational
6. Apply conservative limiter
7. Update `F`
8. Update momentum `pi`
9. Update angular momentum `L`
10. Update structure (`q_D`, optional explicit `q_I` law)
11. Update agency
12. Update coherence
13. Update pointer/diagnostic records

No out-of-band core dynamics are allowed.

## 6. Boundary Operator Rules
- Strictly local operations.
- Edge operators antisymmetric where applicable.
- No direct write to agency.
- No hidden global state injection.
- Jubilee gates are local and debt-targeted (`q_D` only).

## 7. Parameters
### Active canonical additions
- `lambda_DP`, `lambda_IP`, `alpha_qD`, `a0`, `gamma_v`
- Optional numerical: `epsilon_a`

### Deprecated from canonical path
- `lambda_a` (ceiling coupling)
- Any `a_max(q)` structural ceiling rule

Compatibility aliases may remain in parameter dataclasses as no-ops for backward input compatibility.

## 8. Architecture Separation
Required separation:
- Core engine: `det_v7_0/core`
- Runtime/model implementations: `det_v7_0/src`
- Calibration/readout: `det_v7_0/calibration`
- Falsifiers/tests: `det_v7_0/tests`
- Validation harnesses: `det_v7_0/validation`

Calibration modules are readout layers and may not alter core update laws.

## 9. Mandatory Falsification Gates (v7 Canonical)
- `F_A2'` No structural agency suppression
- `F_A4` Frozen-will persistence
- `F_A5` Runaway-agency stability sweep
- `F_GTD5'` Drag-inclusive clock ratio
- `F_BH-Drag-3D` 3D BH drag thermodynamic scaling

Canonization requires all gates passing under declared thresholds.

## 10. Validation Policy
Mandatory physical-validation harnesses remain required (Kepler, GPS/clock, Bell, lensing/cosmology where applicable).
If alternate laws are studied, they must be explicit submodels and cannot silently replace canonical behavior.

## 11. Migration and Backward Compatibility
- Legacy state default: `q_I <- q`, `q_D <- 0`
- Keep `q = clip(q_I + q_D, 0, 1)` for compatibility readout.
- Legacy docs/cards are historical references; this card governs current canonical behavior.

## 12. Acceptance Statement
DET v7.0 is accepted only when:
- canonical falsifier suite passes,
- strict locality is preserved,
- agency remains irreducible in canonical dynamics,
- drag-mediated time-law behavior is stable and reproducible.

## 13. Final Principle
History slows participation-time.  
History does not delete will.
