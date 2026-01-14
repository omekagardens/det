# DET Readout Specification v1.0

**Purpose:** Define exactly what DET predicts for clock rate, time dilation, and gravitational redshift—with explicit formulas suitable for testing against GPS, rocket experiments, and lab height-difference measurements.

**Claim Type:** This document specifies the DET measurement mapping used for falsification testing.

---

## 1. Core Readout Quantities

### 1.1 Presence (P) — Local Clock Rate

**Definition:** The probability of advancing proper time at node i in one discrete step.

```
P_i = a_i × σ_i × (1 + F_i)^(-1) × (1 + H_i)^(-1)
```

**Components:**
| Symbol | Name | Range | Physical Meaning |
|--------|------|-------|------------------|
| a_i | Agency | [0,1] | Intrinsic participation capacity |
| σ_i | Processing rate | > 0 | Local interaction strength (typically = 1) |
| F_i | Resource | ≥ 0 | Local mass/energy concentration |
| H_i | Coordination load | > 0 | Relational overhead (= σ_i or Σ√C_ij·σ_ij) |

**DET Clock Law:** Presence P is the *proper time increment per coordinate step*:
```
dτ_i/dk = P_i
```

### 1.2 Clock Rate Ratio (Time Dilation)

For two clocks at different locations (different F values), with equal agency and processing:

```
P_A / P_B = (1 + F_B) / (1 + F_A)
```

**Key insight:** Higher F (more resource concentration) → lower P (slower clock).

### 1.3 Resource Field (F) — Gravitational Potential Proxy

F is the local energy/mass density. In the presence of gravity:
- F accumulates in potential wells via gravitational flux
- F_∞ represents the "reference" F at infinity (typically = F_VAC ≈ 0.01)

**Gravitational F-accumulation:** The gravitational flux drives F toward potential minima:
```
J^(grav)_{i→j} = μ_g × σ_{ij} × (F_i + F_j)/2 × (Φ_i - Φ_j)
```

---

## 2. Mapping to Standard Physics

### 2.1 Gravitational Redshift

**Standard GR prediction:**
```
(f_received / f_emitted) = √(g_tt(receiver) / g_tt(emitter)) ≈ 1 + ΔΦ/c²
```
where ΔΦ = Φ_receiver - Φ_emitter (gravitational potential difference).

**DET prediction (clock rate ratio):**
```
P_low / P_high = (1 + F_high) / (1 + F_low)
```

**Mapping:** In steady-state with F proportional to potential depth:
```
F ∝ -Φ/c²
```
So a clock in a deeper potential well (more negative Φ, larger F) runs slower.

### 2.2 GPS Clock Correction

GPS satellites experience two competing effects:
1. **Gravitational:** Clocks at altitude run FASTER (less F, higher P)
2. **Kinematic:** Moving clocks run SLOWER (velocity time dilation)

**DET Gravitational Component:**
The fractional frequency shift due to gravity:
```
Δf/f = (P_sat - P_ground) / P_ground ≈ (F_ground - F_sat) / (1 + F_ground)
```

For weak fields (F << 1):
```
Δf/f ≈ F_ground - F_sat = ΔF
```

**Calibration to SI:** Using DET SI unit conversion:
```
F_SI = F_DET × (Φ_scale / c²)
```
where Φ_scale depends on the chosen unit system.

### 2.3 Laboratory Height Difference

For small height differences Δh in uniform gravity g:
```
ΔΦ = g × Δh
ΔF ≈ (g × Δh) / c²

Δf/f = ΔF ≈ g × Δh / c²  ≈  (9.8 × Δh) / (3×10^8)² ≈ 1.1×10^(-16) × Δh[m]
```

**DET must reproduce:** ~1.1×10^(-16) per meter of height.

---

## 3. Reference Clock Choice

### 3.1 What Counts as "Infinity" (P_∞)

The reference clock P_∞ is defined as the clock rate at a location where:
- F = F_VAC (vacuum background level)
- a = 1 (full agency)
- σ = σ_0 (baseline processing rate)
- H = H_0 (baseline coordination load)

```
P_∞ = a_∞ × σ_∞ / (1 + F_VAC) / (1 + H_∞) = 1 / (1 + F_VAC) / 2  [for default params]
```

### 3.2 Operational Definition

For any measurement:
1. **Identify reference point:** Choose a location far from gravitating masses (F ≈ F_VAC)
2. **Measure F at test location:** From simulation or field mapping
3. **Compute clock ratio:** P_test / P_ref = (1 + F_ref) / (1 + F_test)

---

## 4. Validation Targets

### 4.1 GPS Clock Test (G1)

**Observables:**
- Satellite clock offset (broadcast vs measured)
- Periodic variations from orbital eccentricity
- Position-dependent corrections

**DET Predictions:**
| Quantity | Expected Value | Formula |
|----------|----------------|---------|
| Mean offset | +38 μs/day | From ΔF between orbit and ground |
| Eccentricity term | ±few ns | From periodic F variation in elliptical orbit |
| Sign | Positive (sat runs fast) | F_ground > F_orbit |

### 4.2 Rocket Redshift Test (G2)

**Setup:** Frequency comparison between ground and ascending/descending rocket.

**DET Prediction:**
```
(f_rocket - f_ground) / f_ground = (F_ground - F_rocket) / (1 + F_ground)
```

Must match the standard result: Δf/f = gh/c² to within experimental uncertainty.

### 4.3 Lab Height Test (G3)

**Setup:** Two optical clocks separated by height Δh.

**DET Prediction:**
```
Δf/f = g×Δh/c² × (1 + correction_terms)
```

Correction terms from DET lattice discretization should be << 1%.

### 4.4 Bell/CHSH Test (B1)

**Two claim types:**

**A. Fundamental Ceiling:**
- DET predicts: S_max^DET ≈ 2.4 (85% of Tsirelson bound 2√2 ≈ 2.83)
- Falsifier: Clean S > 2.4 in well-controlled experiment
- Formula: S = ΣE(a_i, b_j) where E uses reconciliation algorithm

**B. Operational Ceiling:**
- DET predicts: S depends on measured coherence C, detector efficiency η
- S(C, η) = S_max × visibility_factor(C, η)
- Falsifier: Wrong trend S vs (C, η)

**Current implementation uses Operational Ceiling (B):**
```
E(α, β) = -C × cos(α - β)

S_max = 2√2 × C  for ideal coherence

With C = 0.85: S ≈ 2.4
```

---

## 5. Calibration Protocol

### 5.1 From Lattice to SI

1. **Choose scale:** Pick length a [m/cell] appropriate for the test
   - GPS: a ≈ 10⁴ m/cell (Earth radius / N)
   - Lab: a ≈ 0.1 m/cell

2. **Derive time scale:** τ₀ = a/c

3. **Derive mass scale:** m₀ = G_eff × a × c²/G where G_eff = ηκ/(4π)

4. **Convert F to potential:**
   ```
   Φ[m²/s²] = -F × c² × calibration_factor
   ```

5. **Convert P to clock rate:**
   ```
   (proper time rate) = P × (coordinate time rate)
   ```

### 5.2 Required Calibrations

| Test | Calibration | How Determined |
|------|-------------|----------------|
| GPS | κ (Poisson coupling) | Match mean clock offset |
| Rocket | F_scale | Match peak redshift |
| Lab | F_scale | Match fractional shift/meter |
| Bell | C_quantum (coherence threshold) | Match observed visibility |

---

## 6. Implementation Notes

### 6.1 F Initialization for Earth Tests

For Earth-surface tests, initialize F field as:
```
F(r) = F_VAC + κ × M_Earth / (4π × r)  [in lattice units]
```

Or solve Poisson with q = 1 at Earth's center.

### 6.2 Steady-State Requirement

Time dilation tests require F to reach steady-state:
- Run simulation until max|dF/dt| < threshold
- Or use analytical steady-state F(r) = F_VAC + A/r

### 6.3 Coordinate System

- **Lattice coordinates:** Integer (i, j, k) cell indices
- **Physical coordinates:** x = i×a, y = j×a, z = k×a
- **Spherical mapping:** r = √(x² + y² + z²), convert to/from (r, θ, φ)

---

## 7. Pass/Fail Criteria

### 7.1 Sign Test (Critical)
- Clock in deeper potential well must tick slower
- ∂P/∂F < 0 is required by formula
- If wrong sign → theory falsified

### 7.2 Magnitude Test (Calibrated)
- After one-parameter calibration, must match observed effects to within:
  - GPS: ±10% of mean offset
  - Rocket: ±reported uncertainty
  - Lab: ±5% of fractional shift

### 7.3 Functional Form Test (Discriminating)
- Periodic variations (GPS eccentricity) must match functional form
- Height dependence must be linear in weak-field limit
- Cannot be achieved with "fudge factors" without changing physics

---

## 8. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-14 | Initial specification |

---

*This document is part of the DET Validation Harness.*
