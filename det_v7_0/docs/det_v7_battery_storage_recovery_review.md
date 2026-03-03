# DET v7 Review: Battery Storage Recovery and Faster Charging (Mutable Combined-`q`)

**Date:** March 3, 2026
**Intent:** Deep technical review of how canonical DET v7 can improve charge acceptance by lawful reduction of accumulated structural drag `q`
**Canonical dependency:** `det_v7_0/docs/det_theory_card_7_0.md`

---

## 1) Executive Thesis

Under canonical v7, battery performance loss is modeled as local drag carried by a single mutable debt state `q`.

Core implications:
1. Agency/control intent is not structurally suppressed.
2. Throughput degradation appears via reduced `P` (effective participation/charge acceptance).
3. Recovery operations can reduce `q` locally when energy and gating allow.
4. Faster practical charging comes from managing `q` dynamics, not only increasing current.

---

## 2) DET Mapping for Battery Systems

### 2.1 Variable map
- `F_i`: local electrochemical free-energy/availability state
- `sigma_i`: local transport capability (ionic/electronic)
- `H_i`: local coordination burden (gradients, heterogeneity)
- `q_i`: local accumulated structural drag/debt
- `P_i`: local charge-acceptance tempo
- `a_i`: local control receptivity (not directly debt-suppressed)

Canonical law:

\[
P_i = \left(\frac{a_i\sigma_i}{(1+F_i)(1+H_i)\gamma_v}\right)\cdot\frac{1}{1+\lambda_P q_i}
\]

Meaning: for fixed chemistry and safety limits, lowering local `q` raises effective near-term charging throughput.

### 2.2 Observable proxies for estimating `q`
- pulse-response resistance and relaxation
- rest-voltage recovery curvature
- incremental capacity / differential voltage signatures
- thermal gradient persistence
- impedance spectrum features

These provide estimator inputs for latent local `q` fields.

---

## 3) What "Forgiving History" Means Now

In mutable combined-`q` v7:
1. history can be rewritten only through lawful local energetic recovery,
2. no direct agency modification is permitted,
3. recovery must remain bounded by local energy-cap and safety constraints.

Battery interpretation: use recovery windows to reduce reversible drag burden in practice, while accepting that repeated stress can re-accumulate `q`.

---

## 4) Candidate Battery Recovery Operators (Jubilee Analogues)

1. **Pulse-relax windows:** reduce local transport congestion and thermal gradients.
2. **Thermal equalization intervals:** reduce hotspot-driven `q` accumulation.
3. **Adaptive balancing episodes:** relieve worst-cell bottlenecks and pack-level drag concentration.
4. **Recovery micro-cycles:** brief low-stress phases to lower near-term drag before next high-power burst.
5. **Load-aware tapering:** enforce asymmetric rates to avoid oscillatory `q <-> F` instability.

All operators must obey chemistry-specific voltage/current/temperature envelopes.

---

## 5) Fast-Charge Control Policy Framing

### Objective at each control step
1. maximize usable charging progress (`sum P_i`),
2. constrain `q` growth,
3. trigger local recovery only when net throughput benefit is positive,
4. preserve hard safety constraints.

### Suggested architecture
1. **Estimator layer:** infer `(q, H, sigma, F)` from telemetry.
2. **Predictive policy layer:** choose current/thermal/balancing actions.
3. **Recovery scheduler:** apply Jubilee-like windows under local gating.
4. **Safety governor:** override any unsafe command.

---

## 6) Falsifiers for Battery Application Claims

1. **F_BAT1 Recoverable throughput gain**
- Claim: reducing estimated `q` lowers time-to-target-SOC under fixed safety bounds.

2. **F_BAT2 No free recovery**
- Claim: without sufficient `F_op` and gating, policy cannot force large `q` reduction.

3. **F_BAT3 Tradeoff improvement**
- Claim: DET policy improves charge-time vs degradation Pareto frontier compared with baseline CC-CV.

4. **F_BAT4 Bottleneck relief**
- Claim: pack-level control reduces worst-cell drag dominance and thermal spread.

5. **F_BAT5 Safety invariance**
- Claim: policy keeps zero hard-envelope violations across stress campaigns.

---

## 7) Experimental Program

### Phase A: Observatory
- collect high-rate, rest, and thermal transition data,
- train/validate `q` estimators.

### Phase B: Identification
- fit local DET readout model parameters (`lambda_P`, `alpha_q`, recovery couplings).

### Phase C: Closed-loop simulation
- compare DET-informed policy vs CC-CV and heuristic adaptive baselines.

### Phase D: Bench pilot
- deploy supervised experiments with conservative safety governors.

Core metrics:
- time-to-80% and time-to-90% SOC,
- thermal stress integrals,
- coulombic/round-trip efficiency,
- long-horizon fade and impedance growth,
- module imbalance spread.

---

## 8) Product Opportunities

1. DET-aware BMS fast-charge mode.
2. Fleet charger dispatch based on estimated local `q` state.
3. Second-life triage by recoverability profile and projected drag response.
4. Warranty analytics separating stress-induced vs recoverable behavior.

---

## 9) Risks and Mitigations

Risks:
1. estimator ambiguity for latent `q`,
2. chemistry transfer failure,
3. oscillatory control under aggressive recovery,
4. short-term throughput gains masking long-term damage,
5. safety-model mismatch.

Mitigations:
- conservative safety governors,
- chemistry-specific calibration,
- long-horizon falsifier gates,
- continuous drift monitoring.

---

## Bottom Line

Canonical mutable-`q` DET v7 gives a practical fast-charge strategy: treat historical load as local drag, reduce that drag lawfully and safely, and recover present-moment performance without violating locality or agency-first invariance.
