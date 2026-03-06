# DET v6.3 Sigma Factor Study Report

**Date:** 2026-03-06  
**Branch:** `codex/v6-3-q-mutable-local-grace-exploration`  
**Runner:** `det_v6_3/tests/det_sigma_factor_study.py`

## 1) Objective

Assess the practical role of processing factor `sigma` in v6.3 and answer:

1. What is `sigma` functionally in the implementation?
2. Does it have a real local update law?
3. Can `sigma` be moved out of core and derived from other local primitives in colliders/sims/falsifiers?

## 2) Code Audit Findings

### 2.1 Where `sigma` is used

Across colliders, `sigma` appears in:

- Presence law: `P = a * sigma / (1+F) / (1+H)` (plus optional factors in 3D debt-temporal mode).
- Diffusive, momentum, floor, gravity, and rotational flux multipliers.
- Option-B load via bond-averaged `sigma`.

### 2.2 Update law status by collider

- **1D collider:** `sigma` is initialized to ones and **has no dynamic update step**.
  - This means 1D already treats `sigma` as effectively fixed/exogenous.
- **2D collider:** has local update when `sigma_dynamic=True`:
  - `sigma <- 1 + 0.1 * log(1 + J_mag)`
- **3D collider:** same local update pattern when `sigma_dynamic=True`:
  - `sigma <- 1 + 0.1 * log(1 + J_mag)`

Interpretation: v6.3 has mixed sigma semantics (static in 1D, dynamic in 2D/3D).

## 3) Study Design

All modes used q-mutable exploration parameters:

- `q_mutable_local_enabled=True`
- `alpha_q_local_resource_relief=0.08`
- `alpha_q_grace_relief=0.25`

Sigma modes compared:

1. `core_dynamic`
   - 3D core law active (`sigma_dynamic=True`)
2. `frozen`
   - `sigma_dynamic=False`, force `sigma=1.0`
3. `external_local_derived`
   - `sigma_dynamic=False`
   - external per-step local policy:
     - derived from `a`, `q`, local coherence, and local `|grad F|`
     - no nonlocal/global inputs

Scenarios measured:

- Binding proxy
- Mass conservation proxy
- Time dilation
- 4000-step stability
- DET-C1 utopic trajectory deltas (Fragmented -> Full): `dP_A`, `dP_B`, `dGamma`, `dAcc`, `dW_A`, `dW_B`
- Step-time benchmark

## 4) Results

## 4.1 Mode outputs

### core_dynamic

- `mass drift`: `0.050998` (pass)
- `time dilation factor`: `199.784`
- `longrun`: pass; `sigma_mean=1.003072`
- DET-C1 deltas:
  - `dP_A=+0.108660`
  - `dP_B=+0.003514`
  - `dGamma=+0.107825`
  - `dAcc=+0.093370`
  - `dW_A=-0.327001`
  - `dW_B=-0.275010`
- step time: `0.007397 s`

### frozen

- `mass drift`: `0.052605` (pass)
- `time dilation factor`: `218.181`
- `longrun`: pass; `sigma_mean=1.000000`
- DET-C1 deltas:
  - `dP_A=+0.112072`
  - `dP_B=+0.002845`
  - `dGamma=+0.107773`
  - `dAcc=+0.093332`
  - `dW_A=-0.325878`
  - `dW_B=-0.274337`
- step time: `0.007356 s`

### external_local_derived

- `mass drift`: `0.060402` (pass but worst of three)
- `time dilation factor`: `224.113`
- `longrun`: pass; `sigma_mean=0.994245`
- DET-C1 deltas:
  - `dP_A=+0.086799`
  - `dP_B=+0.001870`
  - `dGamma=+0.107154`
  - `dAcc=+0.092858`
  - `dW_A=-0.329741`
  - `dW_B=-0.274506`
- step time: `0.007466 s`

### Relative to core_dynamic

- `frozen`:
  - `dP_B delta = -0.000668`
  - `dGamma delta = -0.000052`
  - `mass drift delta = +0.001606`
  - runtime ratio `0.994x`
- `external_local_derived`:
  - `dP_B delta = -0.001644`
  - `dGamma delta = -0.000671`
  - `mass drift delta = +0.009404`
  - runtime ratio `1.009x`

## 5) Interpretation

## 5.1 What sigma is in practice

In v6.3 implementation, `sigma` acts as a **local throughput/conductivity modulator** that scales transport and appears in presence numerator while also contributing to load.

## 5.2 Importance

- Changing sigma mode does alter outcomes, but in this study differences were moderate.
- The frozen mode stayed close to core_dynamic on key DET-C1 trajectory metrics.
- A naive external-derived sigma policy remained viable (stable, local, passing basic checks) but degraded mass drift and slightly reduced utopic-trajectory lift.

## 5.3 Can sigma move out of core?

**Yes, technically feasible**, with caveats:

- 1D already effectively does this (no dynamic sigma update in core step).
- 3D/2D can run with `sigma_dynamic=False` and external local policy assignment.
- But policy quality matters: external sigma must be carefully calibrated or regressions appear.

## 6) Recommendation

Adopt a policy-architecture migration instead of hard deletion:

1. Keep `sigma` field in state for compatibility.
2. Move sigma evolution into a pluggable local policy layer (`sigma_policy`).
3. Provide at least:
   - `identity` policy (`sigma=1`)
   - `core_flux` policy (current `1 + 0.1 log(1+J_mag)`)
   - `primitive_derived` policy (local-only fallback)
4. Run full falsifier battery under each declared policy profile before promoting one to default.

## 7) Direct answer to your question

- `sigma` is currently a local processing/transport factor, but not consistently fundamental across colliders.
- It can be moved out of core and derived on demand from local primitives/policies.
- Doing so is plausible for colliders/sims/falsifiers, but only if we formalize sigma-policy contracts and recalibrate to avoid subtle drift/regression.
