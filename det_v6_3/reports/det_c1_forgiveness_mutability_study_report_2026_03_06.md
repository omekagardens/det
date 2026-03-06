# DET-C1 Forgiveness Interaction Study (v6.3 q-mutable branch)

**Date:** 2026-03-06  
**Branch:** `codex/v6-3-q-mutable-local-grace-exploration`  
**Study runner:** `det_v6_3/tests/det_c1_forgiveness_mutability_study.py`

## 1) Research Question

How much additional utopic-trajectory gain appears when:

1. more stimuli-bond processing is moved from hidden partitioning into conscious integration (higher `U`), and  
2. forgiveness/q-debt reduction (mutable `q`) is enabled?

## 2) Experimental Design

Factorial design:

- **Conscious integration trajectory (within run):** Fragmented -> Transition -> Unified -> Utopic-ish -> Full
- **Forgiveness condition (between runs):**
  - `immutable`: `q_mutable_local_enabled=False`
  - `mutable_moderate`: `alpha_q_local_resource_relief=0.08`, `alpha_q_grace_relief=0.25`
  - `mutable_strong`: `alpha_q_local_resource_relief=0.16`, `alpha_q_grace_relief=0.35`

Common settings:

- q-enabled DET v6.3 3D collider with structural debt couplings active.
- 8 seed-perturbed runs per condition.
- 120 steps per stage.

Readouts:

- Regime effectiveness: `P_eff_A`, `P_eff_B`
- Symbolic dependence: `W_A`, `W_B`, `V`
- Inter-regime communication: `Gamma`, `Acc`
- Debt state: `q_mean`

## 3) Key Stage Means (from study runner)

### 3.1 Immutable

- Fragmented: `P_A=0.484380`, `P_B=0.062585`, `Gamma=0.021936`, `Acc=0.021465`, `q_mean=0.193222`
- Full: `P_A=0.593779`, `P_B=0.064727`, `Gamma=0.129088`, `Acc=0.114327`, `q_mean=0.197479`

### 3.2 Mutable (strong)

- Fragmented: `P_A=0.485674`, `P_B=0.063653`, `Gamma=0.022031`, `Acc=0.021556`, `q_mean=0.189122`
- Full: `P_A=0.592703`, `P_B=0.070674`, `Gamma=0.129788`, `Acc=0.114875`, `q_mean=0.180329`

## 4) Effect Decomposition

Using:

- **Integration-only:** `Full_immutable - Fragmented_immutable`
- **Forgiveness-only (early):** `Fragmented_mutable_strong - Fragmented_immutable`
- **Combined:** `Full_mutable_strong - Fragmented_immutable`
- **Additive forgiveness at full integration:** `Full_mutable_strong - Full_immutable`

### 4.1 High-is-better metrics

- `P_eff_A`
  - Integration-only: `+0.109399`
  - Forgiveness-only (early): `+0.001294`
  - Additive forgiveness at full: `-0.001076` (near saturation / slightly negative)
- `P_eff_B`
  - Integration-only: `+0.002142`
  - Forgiveness-only (early): `+0.001068`
  - Additive forgiveness at full: `+0.005947` (major extra lift in debt-heavier regime B)
- `Gamma`
  - Integration-only: `+0.107152`
  - Forgiveness-only (early): `+0.000095`
  - Additive forgiveness at full: `+0.000700`
- `Acc`
  - Integration-only: `+0.092862`
  - Forgiveness-only (early): `+0.000091`
  - Additive forgiveness at full: `+0.000548`

### 4.2 Low-is-better metrics

- `W_A`
  - Integration-only: `-0.327555`
  - Additive forgiveness at full: `+0.001597` (slight regression vs full immutable)
- `W_B`
  - Integration-only: `-0.274439`
  - Additive forgiveness at full: `-0.001667` (further improvement)
- `V`
  - Integration-only: `-0.092862`
  - Additive forgiveness at full: `-0.000548` (further improvement)
- `q_mean`
  - Integration-only: `+0.004257` (debt drift upward without forgiveness)
  - Forgiveness-only (early): `-0.004100`
  - Additive forgiveness at full: `-0.017150` (strong debt suppression at full integration)

## 5) Interaction (Difference-in-Differences)

Reported by runner as:

- Positive values mean forgiveness strengthens the utopic slope from Fragmented -> Full.

Strong mutable vs immutable:

- `P_eff_B`: `+0.004878` (strong positive interaction)
- `Gamma`: `+0.000605` (small positive interaction)
- `Acc`: `+0.000458` (small positive interaction)
- `q_mean` (lower is better): `+0.013049` equivalent improvement in debt slope
- `P_eff_A`: `-0.002370` (slight negative interaction; A already near saturation)

## 6) Interpretation

### 6.1 Direct answer

Moving stimuli-bond processing into conscious integration (`U` up) is the primary driver of utopic-trajectory improvement in this setup.  
Adding forgiveness/q mutability provides **additional uplift**, but that uplift is **selective**:

- Strong for debt-heavier / lower-performing regime segments (here, regime B and global debt).
- Modest for already high-performing coherent segments (here, regime A is near saturation, so extra forgiveness adds little and can slightly reduce A-only gains).

### 6.2 Practical meaning

- **Integration alone:** big gains in communication and conscious performance.
- **Integration + forgiveness:** further reduces structural debt and improves weaker regimes, making overall trajectory more inclusive and less debt-bound.
- **Best interpretation:** forgiveness acts as a downstream accelerator of conscious integration quality, mostly where debt is still constraining expression.

## 7) Conclusion

In this v6.3 q-mutable study, the combined pathway:

1. increase conscious integration (`U`) and  
2. apply lawful local forgiveness (`q` mutability)

improves utopic trajectory more than integration alone at the whole-system level, chiefly by lifting debt-limited regime segments and reducing aggregate debt burden.
