# Deep Existence Theory (DET)
## Constants, Calibration, and Parameter Options

**Document role:**  
This file catalogs *how constants may enter DET*, what can be *derived*, what must be *chosen*, and which quantities are *environmental*.  
It does **not** introduce new primitives and does **not** alter the core DET equations.

DET remains explicitly a **resource–clock–constraint theory**.  
This document exists to prevent parameter sprawl, symbol drift, and hidden assumptions.

---

## 0. Guiding Principles

1. **Core DET contains no physical constants**
   - No \(c\), no \(G\), no \(\hbar\)
   - Only resource, clocks, flow, coherence, constraints

2. **Known constants may appear only as:**
   - calibration targets,
   - unit conversions,
   - emergent mappings,
   - or optional falsifiable hypotheses

3. **Every parameter must be classified as one of:**
   - Gauge / unit choice  
   - Emergent constant  
   - Environmental / material parameter  
   - Genuinely new DET constant  

If a symbol does not fit cleanly, it should not be in the core.

---

## 1. Gauge Choices (Not Physical Parameters)

These quantities can always be fixed by convention and should **never be treated as free constants**.

| Symbol | Meaning | Status |
|------|--------|-------|
| \(k\) | Event ordering index | Gauge (simulation only) |
| \(k_i\) | Local event counter | Gauge |
| \(P_0, M_0\) | Reference clock/mass | Gauge fixing |
| \(F_*\) | Resource unit | Unit choice |
| \(F_{\Psi,*}\) | Coherence unit | Unit choice |
| \(\omega_0\) | Phase scaling | Unit choice |

**Rule:**  
Gauge quantities may be set to 1 without loss of generality.

---

## 2. Emergent Physical Constants (Calibration Targets)

These are **not introduced** by DET but may be *identified* with known constants when mapping to physics.

### 2.1 Speed of Light \(c\)

**DET quantity:**  
\[
c_* = \frac{\bar L}{\bar T_{\text{hop}}}
\]

- Emerges as a **stable propagation fixed point**
- Selected dynamically via coherence survival
- No assumption of universality required in core DET

**Calibration option:**
- Choose units such that \(c_* = c\)
- Or leave \(c_*\) dimensionless in simulations

**Status:** emergent, calibratable

---

### 2.2 Gravitational Constant \(G\)

**DET relation (continuum limit):**
\[
G = \frac{\kappa c_*^4}{4\pi \bar\sigma}
\]

- \(G\) is **not primitive**
- Depends on network coupling \(\kappa\) and average connectivity

**Calibration option:**
- Fix \(\kappa\) by matching observed \(G\) for a chosen graph class
- Or leave \(\kappa\) in network units for abstract simulations

**Status:** derived mapping

---

### 2.3 Planck Constant \(\hbar\)

**DET appearance:**
- Enters only when writing Schrödinger-form equations
- Converts phase/time units into action units

**Interpretation:**
- \(\hbar\) is a **unit-conversion constant**
- Not a new dynamical parameter

**Calibration option:**
- Set \(\omega_0 = 1\) and recover \(\hbar\) when mapping to SI
- Or absorb \(\hbar\) into rescaled time/phase units

**Status:** conversion constant

---

## 3. Environmental / Material Parameters

These are expected to vary by system, medium, or experiment.  
They are **not universal** and should be labeled as such.

| Symbol | Meaning |
|------|--------|
| \(\gamma\) | Resource dissipation |
| \(\lambda_{\text{env}}\) | Environmental decoherence |
| \(\eta_{ij}\) | Transfer efficiency |
| \(L_{ij}\) | Latency geometry |
| \(\sigma_{ij}\) | Edge conductivity |
| \(s_i\) | Measurement sink strength |

**Rule:**  
Environmental parameters must never be advertised as fundamental constants.

---

## 4. Candidate DET-Specific Constants (Critical Audit)

After removing gauge and environmental quantities, the remaining candidates are:

### 4.1 Clock-Load Coupling (\(\beta\))

Appears in:
\[
P_i = \frac{1}{1 + \beta F_i/F_*}
\]

- Controls how resource load slows clocks
- Governs emergence of mass
- Likely **order-1** in natural units

**Status:** core DET parameter

---

### 4.2 Speed-Selection Penalty (\(\alpha\))

Appears in decoherence:
\[
\lambda_{ij} \supset \alpha\left(\frac{v_{ij}-c_*}{c_*}\right)^2
\]

- Enforces stability of \(c_*\)
- Acts like a Lyapunov curvature
- Can often be absorbed into nondimensionalization

**Status:** core or semi-core (can be fixed to 1)

---

### 4.3 Intrinsic Decoherence Floor (\(\lambda_0\)) — OPTIONAL

Appears as:
\[
\lambda_{ij} = \lambda_0 + \lambda_{\text{env}} + \dots
\]

Two consistent stances:

#### Conservative stance
- Set \(\lambda_0 = 0\)
- All decoherence is environmental
- DET matches standard QM exactly in vacuum

#### Bold stance
- \(\lambda_0 > 0\)
- Predicts eventual decay of Bell correlations
- Provides a clean falsification condition

**Status:** optional flagship parameter

---

## 5. Parameter Reduction Strategy

To avoid over-parameterization, DET should be presented with:

### Minimal Universal Set (Recommended)
- \(\beta\): clock-load strength
- \(\alpha\): propagation stability (or set \(\alpha=1\))
- \(\lambda_0\): **either 0 or single new constant**

Everything else is:
- unit choice,
- environmental,
- or calibration.

---

## 6. How to Use Known Constants (Practically)

### Simulation-first work
- Set \(c_* = 1\), \(F_* = 1\), \(\omega_0 = 1\)
- Work entirely in dimensionless units

### Physics-mapping work
- Fix \(c_* \rightarrow c\)
- Fix \(\kappa\) via \(G\)
- Insert \(\hbar\) only when expressing Schrödinger form

### Experimental prediction work
- Keep \(\lambda_0\) explicit (if nonzero)
- Quote bounds, not exact values
- Emphasize falsification conditions

---

## 7. Explicit Non-Claims

DET does **not** claim:

- That \(c, G, \hbar\) are fundamental primitives
- That their numerical values are derived from first principles (yet)
- That the Unified Mathematical Formulation is unique
- That all parameters are universal

---

## 8. Summary Table

| Quantity | Role | Core? |
|------|------|------|
| \(c_*\) | Emergent speed | No |
| \(G\) | Mapping constant | No |
| \(\hbar\) | Unit conversion | No |
| \(\beta\) | Clock-load law | Yes |
| \(\alpha\) | Stability curvature | Maybe |
| \(\lambda_0\) | Vacuum decoherence | Optional |
| \(\gamma,\lambda_{\text{env}},\eta\) | Environment | No |

---

## 9. Status

This document exists to:
- keep DET conceptually clean,
- prevent parameter creep,
- and make explicit what is assumed vs derived.

All **core claims of DET remain independent of this file**.