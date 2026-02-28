# Deep Existence Theory (DET) v6.5.1

**Patch ID:** `DET-6.5.1-A0-DRAG`  
**Status:** Experimental baseline (recommended replacement for v6.4 ceiling law)  
**Date:** 2026-02-28

## 0. Why v6.5.1
v6.4 introduced direct structural suppression of agency via `a_max(q)`. That violates Agency-First invariance (A0).  
v6.5.1 removes structural agency ceilings and moves debt effects into presence/clock-rate through structural drag.

## 1. Ontology and Invariants
- Primitive fields per node: information continuity, agency, movement participation.
- Agency A0 is irreducible and inviolable.
- No operator may directly suppress `a` from debt.
- Boundary operators must never directly write agency.
- Strict locality is mandatory: no hidden global normalization or disconnected-component coupling.

## 2. Canonical State Variables
Per node (`i`):
- `F_i`: resource state
- `q_I,i in [0,1]`: identity debt (immutable by default)
- `q_D,i in [0,1]`: dissipative damage debt (recoverable)
- `q_i = clip(q_I,i + q_D,i, 0, 1)` (legacy readout only)
- `a_i in [0,1]`: agency
- `sigma_i > 0`: processing/conductivity
- `P_i`: presence
- `Delta_tau_i`: proper-time increment

Per bond/plaquette: momentum, coherence, angular momentum (unchanged structurally from v6.3).

## 3. Canonical Laws
### 3.1 Presence With Structural Drag
\[
P_i = \left(\frac{a_i\,\sigma_i}{(1+F_i)(1+H_i)\,\gamma_v}\right) D_i
\]
\[
D_i = \frac{1}{1 + \lambda_{DP} q^D_i + \lambda_{IP} q^I_i}
\]

Mass readout remains:
\[
M_i = P_i^{-1}
\]

### 3.2 Agency Update (No Structural Ceiling)
\[
a_i^+ = \mathrm{clip}\Big(a_i + \beta_a(a_0-a_i) + \gamma(C_i)(P_i-\bar P_{\mathcal N(i)}) + \xi_i,\,0,1\Big)
\]
\[
\gamma(C)=\gamma_{\max} C^n,\quad n\ge 2
\]

Key rule: debt can affect agency only indirectly via `P`, never by direct ceiling/cap.

### 3.3 Structure Update
Default canonical damage accrual:
\[
(q_D)_i^+ = \mathrm{clip}\left((q_D)_i + \alpha_{qD}\max(0,-\Delta F_i),\,0,1\right)
\]

Default identity law:
\[
(q_I)_i^+ = (q_I)_i
\]

Any nontrivial `q_I` law must be explicitly declared as a selectable submodel.

### 3.4 Jubilee Semantics
Jubilee applies to `q_D` only:
- Old wording: “restores agency” (deprecated)
- Canonical wording: **removes drag / restores clock-rate participation**

## 4. Canonical Update Order (Closed, Deterministic)
1. Solve Helmholtz baseline `b`
2. Solve gravitational potential `Phi`
3. Compute presence `P` (including drag)
4. Compute `Delta_tau`
5. Compute flux components: diffusive, momentum, rotational, floor, gravitational
6. Apply conservative limiter
7. Update `F`
8. Update momentum `pi`
9. Update angular momentum `L`
10. Update structure (`q_D`, optional explicit `q_I` law)
11. Update agency
12. Update coherence
13. Update pointer/diagnostic records

No out-of-band dynamics are allowed.

## 5. Parameter Changes
### Added (physical/B-bucket)
- `lambda_DP`
- `lambda_IP`
- `alpha_qD`
- `a0`

### Optional numerical (C-bucket)
- `epsilon_a` with clipped noise: `xi ~ clip(N(0,epsilon_a), -epsilon_a, +epsilon_a)`

### Deprecated
- `lambda_a` as structural agency coupling
- any canonical `a_max(q)` path
- F_A1 “zombie ceiling” interpretation

## 6. Mandatory Falsifiers (v6.5.1)
- `F_A2'`: no structural agency suppression
- `F_A4`: frozen will persistence (100k+ horizon)
- `F_A5`: runaway agency stability across `beta_a` sweep
- `F_GTD5'`: drag-inclusive clock ratio
- `F_BH-Drag-3D`: 3D black-hole thermodynamic scaling

Existing validation suites (Kepler, Bell, GPS, lensing, cosmology) remain required.

## 7. Calibration Layer Constraint
Calibration modules are readouts only. They must not:
- inject new core dynamics,
- alter canonical update order,
- introduce hidden globals.

## 8. Migration Rules
- Replace scalar `q` internals with `(q_I, q_D)`.
- Legacy state migration default:
  - `q_I <- q`
  - `q_D <- 0`
- Keep `q = clip(q_I + q_D, 0, 1)` only as compatibility readout.

## 9. Acceptance Gate
v6.5.1 baseline is accepted when all pass:
- `F_A2'`
- `F_A4`
- `F_A5`
- `F_GTD5'`
- `F_BH-Drag-3D`

If BH scaling is not yet validated in 3D, black-hole thermodynamic claims remain provisional.

## 10. Final Principle
History slows participation-time.  
History does not delete will.

