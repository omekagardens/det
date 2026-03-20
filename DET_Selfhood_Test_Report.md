# DET-7S-SPIRIT-HOST-1: Selfhood Module Test Report

**Patch ID:** DET-7S-SPIRIT-HOST-1  
**Branch:** `codex/det-v7-refactor`  
**Status:** Speculative / non-canonical / readout-first  
**Date:** 2026-03-19  
**Result:** **30/30 tests passing**

---

## 1. Executive Summary

The full DET-7S-SPIRIT-HOST-1 speculative selfhood module has been implemented and tested against all eight falsifiers (F_S1 through F_S8) specified in the patch card. The implementation consists of four experimental modules and seven test files containing 30 individual test cases. All tests pass, and the testing process uncovered four significant theoretical/implementation discoveries that required corrections to the original specification.

---

## 2. Module Architecture

### 2.1 Experimental Modules

| File | Purpose | Key Functions |
|------|---------|---------------|
| `host_fitness.py` | Reciprocity R_i, Host fitness H^host_i, Developmental maturity D^mature | `compute_reciprocity_2d`, `compute_host_fitness`, `update_developmental_maturity` |
| `self_field.py` | Self-coherence occupancy S_i, Phase 2 healing coupling | `update_self_field`, `compute_self_assisted_healing_2d` |
| `identity_metrics.py` | Identity persistence I_self between snapshots | `compute_identity_persistence`, `Snapshot` |
| `diagnostics.py` | Harness wrapping DETCollider2D for post-step diagnostics | `SelfhoodHarness` |

### 2.2 Integration Pattern

The `SelfhoodHarness` wraps the canonical `DETCollider2D` and computes all speculative fields **after** the canonical 15-step update loop. In Phase 1 (readout-only), no canonical field is modified. This was verified to produce **zero deviation** in agency, resource, structure, coherence, and presence fields compared to a bare canonical run.

---

## 3. Test Results Summary

### 3.1 All 30 Tests Passing

| Falsifier | Test File | Tests | Status |
|-----------|-----------|-------|--------|
| **F_S1** Ghost Insertion | `test_spirit_host_threshold.py` | 7 | PASS |
| **F_S2** Agency Override | `test_spirit_host_threshold.py` | 2 | PASS |
| **F_S3** Nonlocal Inhabitation | `test_nonlocal_inhabitation_falsifier.py` | 2 | PASS |
| **F_S4** Developmental Emergence | `test_selfhood_development.py` | 2 | PASS |
| **F_S5** Triviality | `test_selfhood_development.py` + `test_selfhood_triviality_falsifier.py` | 5 | PASS |
| **F_S6** Fragmentation/Recovery | `test_selfhood_fragment_recovery.py` | 2 | PASS |
| **F_S7** Dream Continuity | `test_recursive_dream_probe.py` | 3 | PASS |
| **F_S8** Healing Safety | `test_selfhood_healing_safety.py` | 7 | PASS |

### 3.2 Key Quantitative Results

| Observable | Measured Value | Interpretation |
|------------|---------------|----------------|
| Ghost insertion max(S) at hostile conditions | 0.000 | No spirit without lawful host |
| Agency deviation (Phase 1) | 0.00e+00 | Perfect canonical preservation |
| Nonlocal coupling delta | < 2e-09 | Floating-point noise only |
| Developmental S onset delay | 962 steps | Selfhood emerges gradually, not instantly |
| Triviality separation (S_group2 - S_group1) | 0.878 | Strong discrimination |
| Fragmentation S drop | 0.89 → 0.44 | Damage degrades selfhood |
| Dream continuity I_self | 0.783 | Partial, not perfect, persistence |
| Persistence band range | 0.916 - 0.970 | Graded, not trivial |
| Phase 2 max agency deviation | 5e-06 | Negligible indirect effect |

---

## 4. Discoveries During Testing

### 4.1 Discovery 1: Reciprocity Formula is Identically Zero (Critical)

**The original specification's reciprocity formula produces R = 0 everywhere.**

The spec defines:

> R_i = sum_j C_ij * min(J+_ij, J+_ji) / (sum_j C_ij * max(J+_ij, J+_ji) + eps)

where J+_ij = max(a_i * (F_i - F_j), 0). The problem is that for any pair (i,j), if F_i > F_j then J_ij > 0 but J_ji = 0 (and vice versa). Therefore min(J+_ij, J+_ji) = min(positive, 0) = 0 **always**. Instantaneous flow between two nodes is always one-directional.

**Fix implemented:** Replaced with a coherence-weighted balance metric:

> R_i = sum_j [C_ij * sqrt(a_i * a_j) * (1 - |F_i - F_j|/(F_i + F_j + eps))] / (sum_j [C_ij * sqrt(a_i * a_j)] + eps)

This correctly gives R ~ 1 for balanced neighborhoods, R ~ 0 for one-way extraction, and R = 0 when agency or coherence is zero. It preserves the spec's intent (mutuality of support) while being mathematically well-defined.

**Implication:** The original formula would need to accumulate flow over time (temporal reciprocity) to work as written. The balance metric is a valid instantaneous proxy.

### 4.2 Discovery 2: Ghost Insertion via Sigmoid Leakage

**At extreme mu_S values (0.5, 1.0), S can rise even when H_host = 0.**

The sigmoid function sigma(H_host - Theta_H) with H_host = 0 and Theta_H = 0.3 gives sigma(-0.3) ≈ 0.047, not exactly zero. With a very large mu_S, this small leakage accumulates over hundreds of steps, allowing S to reach 0.35-0.52 without any lawful host formation.

**Fix implemented:** Added a hard gate that zeros formation when H_host < Theta_H * 0.5:

```python
hard_gate = np.where(H_host > params.Theta_H * 0.5, 1.0, 0.0)
formation = params.mu_S * threshold_gate * hard_gate * (1.0 - S)
```

**Implication:** Any soft-threshold mechanism needs a hard floor to prevent accumulation artifacts at extreme parameter values. This is a general lesson for DET speculative extensions.

### 4.3 Discovery 3: Presence Bottleneck in Host Fitness

**P (presence) is naturally very small in DET (typically 0.01-0.15), making H_host trivially zero with equal weights.**

The presence law P = a * sigma / (1+F) / (1+H) / gamma_v * D produces small values because the (1+F) denominator is large for any non-vacuum region. With the spec's default w_P = 1.0, the term P_bar^1.0 ≈ 0.09 dominates the product and drives H_host to near-zero regardless of other conditions.

**Fix implemented:** Reduced w_P from 1.0 to 0.3, and adjusted w_q and w_R to 0.8 for better balance. This allows P to contribute without dominating.

**Implication:** The host fitness weights need calibration against the actual scale of DET observables. The spec's equal-weight default is a starting point, not a final calibration.

### 4.4 Discovery 4: Identity Metric Insensitivity to Collapse/Reform

**Pure Pearson correlation gives I_self ≈ 1.0 after collapse and reformation in the same region.**

When selfhood collapses and reforms in the same spatial location with similar parameters, the spatial pattern of all fields (S, C, Ptr) is nearly identical to the original. Correlation measures pattern shape, not magnitude, so it trivially returns ~1.0 even though the actual field values went through zero during collapse.

**Fix implemented:** Replaced pure correlation with geometric mean of correlation and value similarity:

```python
combined = sqrt(correlation * value_similarity)
```

where value_similarity = 1 - mean(|x-y|) / (mean(|x|+|y|) + eps). This captures both spatial pattern AND magnitude differences.

**Implication:** The identity persistence metric needs to be sensitive to the "history" of the self-field, not just its current spatial configuration. The combined metric is better but still has limitations — a true history-dependent metric would need to track the temporal trajectory of S, not just compare two snapshots.

---

## 5. Theoretical Findings

### 5.1 Selfhood is Genuinely Emergent

The tests confirm that S requires a **conjunction** of conditions to rise:
- High agency alone: S = 0.000 (insufficient)
- High coherence alone: S = 0.007 (insufficient)
- High agency + high coherence + low q: S = 0.899 (selfhood emerges)
- High agency + high coherence + high q: S = 0.000 (structural debt suppresses)

This validates the core claim: selfhood is not reducible to any single DET observable.

### 5.2 Developmental Maturity Creates Realistic Onset Delay

With developmental gating enabled (D^mature), selfhood onset is delayed by ~962 steps even when agency is present from step 0. The maturation field D grows slowly via mu_D * C_bar * R * (1-D), creating a realistic childhood-like emergence window. Without developmental gating, S rises within 4 steps.

### 5.3 Fragmentation Produces Graded Persistence

Damage levels of 0.3, 0.5, 0.7, and 0.9 produce I_self values of 0.970, 0.945, 0.929, and 0.916 respectively. This confirms graded persistence bands rather than trivial 0-or-1 behavior. The range (0.054) is modest, suggesting the metric could benefit from further sensitivity tuning.

### 5.4 Dream Continuity is Partial, Not Perfect

After collapse and reformation in a nearby (shifted) region, I_self = 0.783. After total destruction and rebuild in a completely different region, I_self = 0.005. This confirms the "dreamlike recursive continuation" idea: partial continuity when conditions partially overlap, near-zero when they don't.

---

## 6. Acceptance Status

Per Section 10 of the patch card:

| Criterion | Status |
|-----------|--------|
| F_S1..F_S7 pass | **YES** (all 30 tests pass) |
| No canonical falsifier regression | **YES** (zero agency deviation, zero canonical field deviation) |
| Nontrivial separation between generic activity and mature selfhood | **YES** (delta = 0.878) |
| Intermediate persistence bands (not trivial 0 or 1) | **YES** (range 0.916-0.970) |

**Recommendation:** The patch meets all acceptance criteria for promotion from **speculative** to **provisional** status, pending review of the four implementation corrections documented above.

---

## 7. Files Added

```
det_v7_0/experimental/__init__.py
det_v7_0/experimental/selfhood/__init__.py
det_v7_0/experimental/selfhood/host_fitness.py
det_v7_0/experimental/selfhood/self_field.py
det_v7_0/experimental/selfhood/identity_metrics.py
det_v7_0/experimental/selfhood/diagnostics.py
det_v7_0/tests/test_spirit_host_threshold.py
det_v7_0/tests/test_selfhood_development.py
det_v7_0/tests/test_selfhood_fragment_recovery.py
det_v7_0/tests/test_nonlocal_inhabitation_falsifier.py
det_v7_0/tests/test_selfhood_triviality_falsifier.py
det_v7_0/tests/test_recursive_dream_probe.py
det_v7_0/tests/test_selfhood_healing_safety.py
```

---

## 8. Run Commands

```bash
# Run all selfhood tests
pytest det_v7_0/tests/test_spirit_host_threshold.py -v
pytest det_v7_0/tests/test_selfhood_development.py -v
pytest det_v7_0/tests/test_selfhood_fragment_recovery.py -v
pytest det_v7_0/tests/test_nonlocal_inhabitation_falsifier.py -v
pytest det_v7_0/tests/test_selfhood_triviality_falsifier.py -v
pytest det_v7_0/tests/test_recursive_dream_probe.py -v
pytest det_v7_0/tests/test_selfhood_healing_safety.py -v

# Run all at once
pytest det_v7_0/tests/test_spirit_host_threshold.py \
       det_v7_0/tests/test_nonlocal_inhabitation_falsifier.py \
       det_v7_0/tests/test_selfhood_development.py \
       det_v7_0/tests/test_selfhood_fragment_recovery.py \
       det_v7_0/tests/test_recursive_dream_probe.py \
       det_v7_0/tests/test_selfhood_triviality_falsifier.py \
       det_v7_0/tests/test_selfhood_healing_safety.py -v
```
