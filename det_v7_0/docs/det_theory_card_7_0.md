# Deep Existence Theory (DET) v7.0

**Unified Canonical Theory Card (Mutable Combined-`q` Baseline)**

**Release Date:** March 3, 2026
**Canonical status:** Active canonical model for `det_v7_0`
**Patch lineage:** includes `DET-7X-Q-UNIFIED` mutable-`q` consolidation

---

## Changelog Lineage

### v6.3 Baseline
1. Established strictly local relational update dynamics.
2. Stabilized collider families (1D/2D/3D/Torch) and baseline falsifier programs.
3. Formalized gravity, flow decomposition, and readout-layer calibration structure.

### v6.4 Historical (Superseded)
1. Introduced structural agency ceiling (`a_max`) in canonical path.
2. Retained only for historical traceability and regression archaeology.

### v6.5 Historical (Superseded)
1. Added stronger recovery/Jubilee machinery and expanded debt semantics.
2. Introduced split debt (`q_I`, `q_D`) as an intermediate architecture.

### v6.5.1 Patch Layer
1. Removed structural agency ceiling from canonical path.
2. Enforced Agency-First invariance (`A0`) as a hard rule.
3. Moved debt effects to drag/presence clock-rate expression.
4. Added upgraded falsifiers: `F_A2'`, `F_A4`, `F_A5`, `F_GTD5'`, `F_BH-Drag-3D`.

### v7.0 Mutable Combined-`q` Consolidation (Current Canonical)
1. Removed dual-debt split from active canonical engine.
2. Unified structural debt as a single mutable local field `q in [0,1]`.
3. Unified drag coupling as `lambda_P`.
4. Standardized structure update as `alpha_q` accumulation plus lawful local recovery.
5. Added mutable-`q` stability falsifiers: `F_QM1`..`F_QM5`.

---

## 0) Scope Axiom

DET v7 is a closed, deterministic, strictly local relational dynamics. Each node evolves through local interaction of:
- resource/field participation,
- retained structural history,
- irreducible agency,
- movement/event progression.

No global hidden normalizers, no nonlocal coupling, and no direct structural suppression of agency are allowed.

---

## 1) Ontology and Invariants (Non-Negotiable)

### 1.1 Strict Locality
All sums/averages and operators are neighborhood-local, bond-local, or plaquette-local.
Disconnected components must remain causally independent.

### 1.2 Agency-First Invariance (`A0`)
Agency is primitive and inviolable:
- no direct debt-to-agency suppression,
- no structural ceiling on `a`,
- no boundary operator direct writes to `a`.

### 1.3 Structural History Expression
History affects expression through participation/clock-rate drag via `P` and `Delta_tau`.
It does not delete will.

### 1.4 Falsifiability First
All major claims require explicit falsifiers with declared thresholds.
Calibration modules remain readout layers only.

### 1.5 Boundary-Agent Continuity (Carried from v6.3)
DET v7 retains the boundary-agent layer.
It acts only through explicit local operators (Grace, Healing, Jubilee), with no direct write to agency and no hidden nonlocal state.
Boundary action is lawful local dynamics, not arbitrary override.

---

## 2) Canonical State Variables

### 2.1 Node-Local Variables (`i`)
- `F_i`: local resource field
- `q_i in [0,1]`: total structural debt (mutable)
- `a_i in [0,1]`: agency
- `sigma_i > 0`: local conductivity/processing factor
- `H_i >= 0`: local coordination load
- `P_i > 0`: local presence (effective participation rate)
- `Delta_tau_i`: local proper-time increment
- `C_i in [0,1]`: coherence factor

### 2.2 Bond/Plaquette Variables
- local momentum channels (`pi`-sector)
- local rotational channels (`L`-sector)
- local boundary exchange channels

### 2.3 Derived Readouts
- mass readout: `M_i = 1 / P_i`
- gravity source readout: `rho_i = q_i - b_i`

---

## 3) Canonical Laws

### 3.1 Presence and Drag
Base presence:

\[
P_{i}^{base} = a_i\,\sigma_i\,\frac{1}{1+F_i}\,\frac{1}{1+H_i}\,\frac{1}{\gamma_v}
\]

Unified drag multiplier:

\[
D_i = \frac{1}{1 + \lambda_P q_i}
\]

Canonical presence:

\[
P_i = P_i^{base}\,D_i
\]

Proper-time update:

\[
\Delta\tau_i = P_i\,\Delta k
\]

Mass readout:

\[
M_i \equiv P_i^{-1}
\]

### 3.2 Structure Accumulation (Loss Locking)

\[
dq_{lock,i} = \alpha_q\max(0,-\Delta F_i)
\]
\[
q_i^{+} = \mathrm{clip}(q_i + dq_{lock,i}, 0, 1)
\]

### 3.3 Structure Recovery / Jubilee
Recovery is energy-coupled and local:

\[
S_i = a_i\,C_i^{n_q}\,\frac{D_i}{D_i + D_0}
\]
\[
dq_{jub,i} = \delta_q\,S_i\,\Delta\tau_i
\]
\[
F_{op,i} = \max(F_i - F_{VAC}, 0)
\]
\[
energy\_cap_i = \frac{F_{op,i}}{1 + F_{op,i}}
\]
\[
q_i^{+} = \mathrm{clip}\left(q_i - \min(dq_{jub,i}, energy\_cap_i, q_i), 0, 1\right)
\]

In DET v7, there is no separate global Hamiltonian energy state in the canonical card.
The operative local energy-like budget for Jubilee is the free-resource readout `F_{op,i}` derived from `F_i` above the vacuum floor `F_{VAC}`.

Jubilee never modifies `a` directly.

### 3.4 Agency Update (No Ceiling)

\[
a_i^{+} = \mathrm{clip}\left(
 a_i + \beta_a(a_0-a_i) + \gamma(C_i)(P_i-\bar P_{\mathcal{N}(i)}) + \xi_i,
 0, 1
\right)
\]

\[
\gamma(C)=\gamma_{max} C^n,\quad n\ge2
\]

Debt affects agency only indirectly through `P`.

### 3.5 Gravity Source and Potential
Helmholtz baseline:

\[
(\mathcal{L}-\alpha_{grav})b = -\alpha_{grav}q
\]

Source and potential:

\[
\rho = q-b
\]

Poisson-like local solve and local gradient readout define gravity field.

### 3.6 Boundary Operators (Canonical v7)
Local dissipation proxy from limited flux:

\[
D_i = \sum_{j \in \mathcal{N}(i)} |J_{i\to j}|\,\Delta\tau_i
\]

Grace injection (v6.2-style local node operator used in canonical colliders):

\[
n_i = \max(0, F_{MIN}^{grace} - F_i),\quad w_i = a_i n_i
\]
\[
I_{g\to i} = D_i \frac{w_i}{\sum_{k \in \mathcal{N}_R(i)} w_k + \varepsilon}
\]

Properties:
- strictly local neighborhood normalization,
- agency-gated (`a_i=0 \Rightarrow I_{g\to i}=0`),
- modifies `F`, never modifies `a` directly.

Bond healing (if enabled):

\[
\Delta C^{heal}_{ij} = \eta_{heal}\,g^{(a)}_{ij}\,(1-C_{ij})\,\bar D_{ij}\,\Delta\tau_{ij},
\quad g^{(a)}_{ij}=\sqrt{a_i a_j}
\]

Jubilee remains the `q`-recovery operator in Section 3.3, energy-coupled through `F_{op}`.

---

## 4) Canonical Update Order (Must Preserve)

1. Solve baseline field `b` (Helmholtz)
2. Solve gravitational potential `Phi`
3. Compute presence `P` (including drag)
4. Compute `Delta_tau`
5. Compute flux components (diffusive, momentum, rotational, floor, gravitational)
6. Apply conservative limiter
7. Update `F` from local flux divergence
8. Apply boundary Grace operator to `F` (if enabled)
9. Update momentum `pi`
10. Update angular momentum `L`
11. Update structure `q` from loss locking
12. Apply boundary Healing/Jubilee operators (if enabled)
13. Update agency `a`
14. Update coherence `C`
15. Update pointer/record diagnostics

No out-of-band core-law insertion is allowed.

---

## 5) Boundary Operator Rules

1. Boundary operators must be strictly local.
2. Operators may not inject hidden global state.
3. Operators may not directly modify agency.
4. Grace acts on `F` via local depletion need and local dissipation budget.
5. Healing acts on bond coherence `C_ij` only.
6. Jubilee acts on `q` only, with local gating and optional energy coupling cap.
7. Any alternate edge-flux Grace law must be explicitly declared as a selectable submodel.

---

## 6) Parameter Schema

### 6.1 Active Canonical Parameters
- `lambda_P`: drag coupling on total `q`
- `alpha_q`: debt accumulation rate from loss
- `delta_q`: Jubilee recovery coupling
- `n_q`: Jubilee coherence exponent
- `D_0`: Jubilee drag-activation scale
- `F_VAC`: vacuum floor for operational free-resource readout `F_op`
- `F_MIN_grace`: Grace depletion threshold
- `R_boundary`: Grace local neighborhood radius
- `eta_heal`: bond-healing coupling (if enabled)
- `boundary_enabled`, `grace_enabled`, `healing_enabled`, `jubilee_enabled`: boundary operator toggles
- `jubilee_energy_coupling`: enforce `F_op` energy cap on Jubilee
- `beta_a`: agency relaxation rate
- `a0`: agency attractor (default `1.0`)
- `gamma_max`, `n`: coherence gate parameters
- `gamma_v`: velocity/time scaling factor
- optional `epsilon_a`: persistence noise amplitude

### 6.2 Optional Noise Model

\[
\xi_i \sim \mathrm{clip}(\mathcal{N}(0,\epsilon_a),-\epsilon_a,+\epsilon_a)
\]

Default: `epsilon_a = 0` unless explicitly enabled.

### 6.3 Deprecated/Removed from Canonical Path
- `lambda_a`
- `a_max(q)` computations
- dual-`q` (`q_I`, `q_D`) core semantics
- dual drag couplings (`lambda_IP`, `lambda_DP`)
- split accumulation (`alpha_qI`, `alpha_qD`)

---

## 7) Mandatory Falsifiers and Validation Gates

### 7.1 Agency and Stability Gates
- `F_A2'`: no structural agency suppression
- `F_A4`: frozen will persistence under extreme drag
- `F_A5`: runaway-agency stability across `beta_a` sweep

### 7.2 Mutable-`q` Specific Gates
- `F_QM1`: total annealing stability
- `F_QM2`: identity persistence under moderate recovery
- `F_QM3`: Kepler stability under mutable `q`
- `F_QM4`: no runaway oscillatory `q <-> F` collapse
- `F_QM5`: arrow-of-time integrity under recovery dynamics

### 7.3 Time/Gravity/Thermodynamics Gates
- `F_GTD5'`: drag-inclusive clock ratio
- `F_BH-Drag-3D`: 3D BH scaling gate for thermodynamic claims

### 7.4 External Readout Validation
Must remain passing in validation harness and dedicated tests:
- GPS/rocket/lab gravitational time checks
- Kepler readout consistency
- Bell/CHSH operational envelope
- lensing/rotation/G readouts

---

## 8) Test Harness and Determinism Requirements

1. Deterministic seeds for all long-run falsifiers.
2. Long-run tests (including 100k+ horizons where configured) support checkpoint sampling.
3. Early-failure triggers must terminate unstable trajectories quickly.
4. Acceptance thresholds must be explicit in test definitions.

---

## 9) Calibration Layer Separation

Calibration modules remain readout-only:
- `extract_g_calibration.py`
- `galaxy_rotation_curves.py`
- `gravitational_lensing.py`
- `cosmological_scaling.py`
- `black_hole_thermodynamics.py`
- `quantum_classical_transition.py`

They may not alter core update equations, inject hidden globals, or replace canonical laws.

---

## 10) Migration and Compatibility Notes

### 10.1 State Migration
- New canonical state uses scalar `q` only.
- Legacy split states map by default: `q <- clip(q_I + q_D, 0, 1)`.

### 10.2 Parameter Migration
- Replace `lambda_IP/lambda_DP` with `lambda_P`.
- Replace `alpha_qI/alpha_qD` with `alpha_q`.
- Remove `lambda_a` and all structural-ceiling paths.

### 10.3 Legacy Cards
`det_theory_card_6_3.md`, `det_theory_card_6_5.md`, and `det_theory_card_6_5_1.md` remain historical references only.

---

## 11) Verification Snapshot (March 3, 2026)

Executed against repository state on `codex/det-v7-refactor`:
- `det_v651_falsifiers.py`: 10/10 pass
- `det_v65_falsifiers.py`: 8/8 pass
- `det_v65b_falsifiers.py`: 4/4 pass
- `det_comprehensive_falsifiers.py`: 15/15 pass
- `det_validation_harness.py --all`: 6/6 pass
- Bell/gravity-focused pytest slice: 50 passed
- full `pytest det_v7_0/tests -q`: 204 passed

Warnings remain in legacy-style tests returning non-`None`; no failing assertions.

Focused boundary-operator study artifact:
- `det_v7_0/reports/grace_jubilee_energy_interaction_2026_03_03.md`

---

## 12) Open Research Questions

1. What recovery rate asymmetries best prevent over-annealing while preserving locality?
2. Which observables best separate drag-from-`q` vs slowing-from-`F/H` in physical inference tasks?
3. How robust is 3D BH scaling across broader geometry and mass sweeps?
4. Which control policies use mutable-`q` recovery most effectively without instability?

---

## Final Principle

History shapes participation-time through lawful local dynamics.
History does not delete will.

Locality is inviolable.
Agency is irreducible.
Physics emerges from closed, local, falsifiable updates.
