# DET v7 Review: Battery Storage Recovery and Faster Charging

**Date:** 2026-02-28  
**Intent:** Deep technical review of how DET v7 can model and improve battery charging speed by reducing recoverable historical damage while preserving safety and life  
**Canonical dependency:** `det_v7_0/docs/det_theory_card_7_0.md`

---

## 1. Executive Thesis

DET v7 gives a useful battery-control framing:

1. Separate degradation into:
   - `q_I`: irreversible historical damage (cannot be "forgiven")
   - `q_D`: recoverable drag (can be reduced with lawful recovery operations)
2. Map charging performance to presence:
   - `P` as local effective charge-acceptance rate
3. Treat restoration as drag reduction:
   - reduce `q_D` to increase present-moment throughput

This turns fast charging from a one-dimensional current maximization problem into a state-managed control problem:
maximize charge acceptance now while minimizing future irreversible damage growth.

---

## 2. Why Battery Systems Need This Split

Conventional BMS logic often lumps degradation effects into a few coarse state variables (SOC, SOH, temperature, resistance). DET v7 suggests a sharper split:

1. **Irreversible pathways** align with `q_I`:
   - permanent loss of lithium inventory
   - active material loss and electrode cracking
   - persistent impedance growth that does not recover after rest/conditioning
2. **Recoverable pathways** align with `q_D`:
   - transient concentration polarization
   - temporary transport bottlenecks
   - short-horizon kinetic/thermal stress signatures that relax with protocol shaping

Practical implication:
not all "aged behavior" is fixed. Some is drag that can be reduced quickly, improving immediate charging performance.

---

## 3. DET v7 Variable Mapping for Batteries

### 3.1 Core mapping

1. `F_i`: local electrochemical free-energy/availability state at region `i`
2. `sigma_i`: effective local transport capability (ionic + electronic pathways)
3. `H_i`: local coordination burden from gradients and heterogeneity
4. `a_i`: local control receptivity / actuation headroom (not directly suppressed by debt in canonical law)
5. `q_I,i`: irreversible aging debt
6. `q_D,i`: recoverable operational debt
7. `P_i`: local charge-acceptance tempo
8. `D_i`: structural drag from debt decomposition

Canonical relation:

\[
P_i = \left(\frac{a_i \sigma_i}{(1+F_i)(1+H_i)\gamma_v}\right)\cdot
\frac{1}{1+\lambda_{DP}q_{D,i}+\lambda_{IP}q_{I,i}}
\]

Interpretation:
for fixed chemistry and thermal bounds, faster safe charging comes from increasing the base term and reducing drag, especially the recoverable portion (`q_D`).

### 3.2 Practical observables for estimation

Possible measurement proxies (cell or segment level):

1. pulse-response resistance and relaxation curves
2. incremental capacity and differential voltage signatures
3. rest-voltage recovery shape
4. thermal gradients and hotspot persistence
5. impedance-spectrum features by frequency band

These proxies can estimate latent `q_I` and `q_D` without requiring direct physical observability of DET fields.

---

## 4. What "Forgiving History" Means in Battery Terms

In v7 language:

1. You do not erase `q_I`.
2. You can lower `q_D` via lawful local recovery operations.
3. Lower `q_D` increases `D`, which increases `P`, improving immediate charging throughput.

Battery translation:
recovery windows should be designed as targeted drag-removal interventions, not as attempts to reverse irreversible wear.

---

## 5. Candidate Recovery Operators (Battery Jubilee Analogues)

These are model-level categories, not immediate deployment recipes.

1. **Rest-based relaxation windows**
   - short holds to dissipate concentration and thermal gradients
   - expected primary effect: `q_D` down, `q_I` unchanged
2. **Pulse-shaped current profiles**
   - alternating high/low segments to avoid sustained high-stress regimes
   - expected effect: reduced accumulation of transient drag and lower irreversible side reaction pressure
3. **Thermal homogenization windows**
   - reduce spatial heterogeneity before and during high-rate segments
   - expected effect: lower `H_i` and reduced localized `q_D` spikes
4. **Adaptive balancing at module level**
   - cell-level redistribution to prevent worst-cell drag domination
   - expected effect: pack-level `P` improves via bottleneck relief
5. **Recovery micro-cycles**
   - intentionally inserted low-stress segments to recover transport kinetics
   - expected effect: improve near-term acceptance without masking long-term `q_I` trend

Safety constraint:
all operators must remain under chemistry-specific voltage, temperature, and current limits.

---

## 6. Fast-Charge Control Policy Under DET v7

### 6.1 Objective

At each control step, maximize usable charging progress while constraining risk:

1. maximize `sum_i P_i`
2. minimize growth rate of `q_I`
3. stabilize or reduce `q_D` when practical
4. enforce hard safety bounds

### 6.2 Suggested control architecture

1. **Estimator layer**
   - infer `(q_I, q_D, H, sigma)` from telemetry
2. **Predictive policy layer**
   - select current/thermal/balancing actions
3. **Recovery scheduler**
   - trigger Jubilee-like windows when marginal throughput gain is favorable
4. **Safety governor**
   - override with strict thermal/voltage/current constraints

### 6.3 Charge profile concept

1. Early phase: exploit high `P` regions with controlled high current.
2. Mid phase: pulse/relax to prevent `q_D` runaway in stressed regions.
3. Late phase: tighten constraints to avoid irreversible damage acceleration.
4. Completion: optional short recovery phase to leave cell in lower-drag state for next session.

---

## 7. Simulation and Validation Program

### 7.1 New submodel proposal

Create explicit submodel family:

- `DET-BAT-v7` (readout/control layer over canonical v7 dynamics)
- no changes to canonical core laws
- battery assumptions encoded in configuration and measurement adapters

### 7.2 Suggested repository additions

1. `det_v7_0/validation/det_battery_recovery_harness.py`
2. `det_v7_0/tests/test_battery_qd_recovery_gain.py`
3. `det_v7_0/tests/test_battery_qi_nonrecoverability.py`
4. `det_v7_0/tests/test_battery_fastcharge_drag_policy.py`
5. `det_v7_0/reports/det_v7_battery_recovery_baseline.md`

### 7.3 Falsifiers for battery application claims

1. **F_BAT1: Recoverable-drag gain**
   - claim: lowering estimated `q_D` increases near-term charge acceptance at fixed safety bounds
   - pass: statistically significant reduction in time-to-target-SOC for matched cohorts
2. **F_BAT2: Irreversible debt non-forgiveness**
   - claim: interventions cannot materially reverse `q_I`
   - pass: long-horizon irreversible indicators do not show fake recovery artifacts
3. **F_BAT3: Throughput/life tradeoff improvement**
   - claim: DET policy improves charging time without increasing irreversible fade rate beyond threshold
   - pass: better Pareto frontier than baseline CC-CV policy
4. **F_BAT4: Pack bottleneck relief**
   - claim: module-aware recovery reduces worst-cell drag dominance
   - pass: improved pack-level utilization and reduced thermal spread
5. **F_BAT5: Safety invariance**
   - claim: recovery policy never violates hard safety envelopes
   - pass: zero envelope violations over stress campaign

---

## 8. Experimental Design Recommendations

### 8.1 Data campaign phases

1. **Phase A: Observatory build**
   - instrument high-rate and rest transitions
   - derive candidate `q_I/q_D` estimators
2. **Phase B: Identification**
   - fit estimator and validate against held-out cycle data
3. **Phase C: Policy simulation**
   - compare DET-informed policy vs baseline CC-CV and heuristic adaptive charging
4. **Phase D: Controlled pilot**
   - bench-level deployment with strict safety supervision

### 8.2 Core metrics

1. time-to-80% SOC and time-to-90% SOC
2. peak and integrated thermal stress
3. coulombic efficiency and round-trip efficiency
4. irreversible capacity fade per equivalent full cycle
5. impedance growth trajectory
6. cell imbalance and pack utilization spread

---

## 9. Practical Product Opportunities

1. **DET-aware BMS firmware mode**
   - optional policy layer for adaptive fast charging and recovery scheduling
2. **Fleet-level battery operations platform**
   - charger dispatch based on estimated `(q_I, q_D)` states and service urgency
3. **Second-life battery triage**
   - distinguish high-`q_D` recoverable assets from high-`q_I` exhausted assets
4. **Warranty analytics**
   - clearer attribution: irreversible wear vs recoverable operating drag
5. **Charging infrastructure coordination**
   - station-side policy that allocates power where marginal drag-removal benefit is highest

---

## 10. Risks and Failure Modes

1. **State-estimation ambiguity**
   - poor separation of `q_I` and `q_D` can create false recovery expectations
2. **Overfitting to one chemistry**
   - policy may not transfer across NMC/LFP/solid-state variants without recalibration
3. **Control-induced oscillation**
   - aggressive pulse/recovery switching can create thermal or voltage oscillations
4. **Apparent short-term wins with long-term harm**
   - throughput gains might hide accelerated irreversible damage if monitoring is weak
5. **Safety model mismatch**
   - invalid constraints can produce hazardous operating commands

Mitigation:
require conservative safety governors, chemistry-specific calibration, and long-horizon validation gates.

---

## 11. Integration with Existing DET v7 Docs

1. Canonical law: `det_theory_card_7_0.md`
2. Broad applications map: `det_v7_applications_review.md`
3. Structural debt context: `det_structural_debt.md`, `det_structural_debt_applications.md`
4. v7 migration and consistency context: `v7_migration_deprecation_map.md`, `v7_consistency_review_2026_02_28.md`

This battery review is an application-layer proposal and does not alter canonical v7 equations.

---

## 12. Bottom Line

If the estimator can reliably separate irreversible (`q_I`) from recoverable (`q_D`) debt, DET v7 offers a direct path to faster practical charging:

1. reduce recoverable drag before/during charge,
2. preserve safety envelopes,
3. avoid pretending irreversible history can be erased.

The engineering target is not "undo all aging."
The target is "recover present-moment performance without increasing irreversible damage slope."
