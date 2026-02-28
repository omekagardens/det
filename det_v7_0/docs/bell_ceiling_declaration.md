# DET Bell/CHSH Ceiling Declaration

**Document Type:** Theoretical Commitment
**Version:** 1.0
**Date:** 2026-01-14

---

## 1. Declaration

**DET declares an OPERATIONAL CEILING for Bell/CHSH violation.**

The maximum CHSH value S achievable in DET depends on measurable system parameters, not on fundamental limits of the theory.

---

## 2. The Two Claim Types (Explained)

### A. Fundamental Ceiling (NOT chosen)

A fundamental ceiling would mean:
- DET predicts S ≤ S_max^DET even under ideal conditions
- S_max^DET ≈ 2.4 (85% of Tsirelson bound)
- This would be a hard prediction, falsifiable by any clean experiment showing S > S_max^DET

**Why we reject this:**
- No clear physical principle in DET that would impose a fundamental limit below 2√2
- The retrocausal reconciliation algorithm CAN produce S = 2√2 with C = 1
- Claiming a fundamental limit would require explaining WHY C cannot reach 1

### B. Operational Ceiling (CHOSEN)

An operational ceiling means:
- DET predicts S depends on coherence C, detector efficiency η, and other measurable parameters
- The formula: S(C) = 2√2 × C (for ideal detectors)
- The observed ~85% is explained by: C ≈ 0.85 in typical experimental conditions

**Why we choose this:**
- Matches DET's structure: coherence C is a fundamental state variable
- Provides testable predictions: S should correlate with measured coherence/visibility
- Consistent with experimental trends: lower-quality experiments show lower S

---

## 3. The Operational Formula

### 3.1 Basic Form

For a pair of entangled particles with coherence C:

```
S_max(C) = 2√2 × C
```

where:
- S is the CHSH parameter: S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
- E(α,β) is the correlation function
- C ∈ [0,1] is the coherence of the entangled pair

### 3.2 With Detector Effects

Including detector efficiency η and dark count rate d:

```
S_observed(C, η, d) = S_max(C) × visibility(η, d)
```

where visibility depends on detection loopholes.

### 3.3 DET Correlation Function

In the retrocausal reconciliation model:

```
E(α, β) = -C × cos(α - β)
```

This gives:
- S = 2√2 × C at optimal angles (α=0, α'=π/2, β=π/4, β'=3π/4)
- Classical limit (C→0): S → 0
- Quantum limit (C→1): S → 2√2

---

## 4. Falsification Criteria

### 4.1 What Would Falsify This Claim

The operational ceiling claim is falsified if:

1. **Wrong trend:** S does NOT correlate with measured coherence/visibility
   - Test: Plot S vs C across multiple experiments
   - Falsifier: Significant anti-correlation or no correlation

2. **Wrong functional form:** S(C) ≠ 2√2 × C
   - Test: Fit S vs C data to various functional forms
   - Falsifier: Better fit to different formula

3. **Systematic under-prediction:** DET always predicts S lower than observed
   - Test: Compare DET predictions to observed S across many experiments
   - Falsifier: Consistent positive residuals (observed > predicted)

### 4.2 What Would NOT Falsify This Claim

- Experiments achieving S > 2.4: This just means their C > 0.85
- Experiments with S < 2: This could mean low C, high noise, or detection issues
- Variation in S across experiments: Expected from variation in C

---

## 5. Testable Predictions

### 5.1 Correlation Tests

**Prediction 1:** S should increase monotonically with measured visibility V.

If V = 2C - 1 (standard visibility formula), then:
```
S = √2 × (V + 1) = √2 × V + √2
```

**Prediction 2:** Perfect visibility (V=1) should give S = 2√2 ≈ 2.828.

**Prediction 3:** For V < 1/√2 ≈ 0.707, classical bound should be violated (S > 2).

### 5.2 Degradation Tests

**Prediction 4:** Adding decoherence (noise, path length, temperature) should reduce S proportionally to the reduction in visibility.

**Prediction 5:** The relationship S(V) should be linear with slope √2.

### 5.3 Specific Numerical Predictions

| Visibility V | Coherence C | DET Prediction S |
|--------------|-------------|------------------|
| 0.70 | 0.85 | 2.40 |
| 0.80 | 0.90 | 2.55 |
| 0.90 | 0.95 | 2.69 |
| 0.95 | 0.975 | 2.76 |
| 0.99 | 0.995 | 2.81 |
| 1.00 | 1.00 | 2.83 |

---

## 6. Open Questions

### 6.1 What Determines C?

DET currently treats C as a given parameter. Future work should explain:
- How C emerges from source preparation
- Why typical experiments have C ≈ 0.85
- What physical processes reduce C below 1

### 6.2 The "Free Choice" Question

In the retrocausal model, measurement settings are boundary conditions.
Does this affect the notion of "free choice" required for Bell tests?

**Current answer:** No, because:
- Marginal distributions are independent of distant settings (no-signaling)
- Experimenters can freely choose settings
- Reconciliation doesn't violate causality

---

## 7. Experimental Test Protocol

To test the DET operational ceiling:

### Step 1: Gather Data
- Collect Bell test results from published experiments
- Record: S value, visibility/fidelity, error bars, experimental conditions

### Step 2: Extract Coherence
- Compute C from reported visibility: C = (V + 1)/2
- Or fit from correlation function if available

### Step 3: Compare
- Plot observed S vs DET prediction S_DET = 2√2 × C
- Compute residuals and check for systematic bias

### Step 4: Verdict
- If residuals are unbiased and small → operational ceiling supported
- If residuals are systematically positive → DET under-predicts
- If residuals show no S-C correlation → operational ceiling falsified

---

## 8. Commitment Statement

**We commit to:**

1. Using the operational ceiling claim (not fundamental) for all DET predictions
2. Testing against published Bell data as it becomes available
3. Revising the formula if systematic deviations are found
4. Publishing updates to this declaration if the claim type changes

**Signed:** DET Validation Harness v1.0

---

## Appendix: Derivation of S(C) = 2√2 × C

From the DET retrocausal correlation function:

```
E(α, β) = -C × cos(α - β)
```

The CHSH combination at optimal angles:
```
S = E(0, π/4) - E(0, 3π/4) + E(π/2, π/4) + E(π/2, 3π/4)
  = -C[cos(-π/4) - cos(-3π/4) + cos(π/4) + cos(-π/4)]
  = -C[√2/2 + √2/2 + √2/2 + √2/2]
  = -C × 2√2
```

Taking absolute value: |S| = 2√2 × C

QED.
