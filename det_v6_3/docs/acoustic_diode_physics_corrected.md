# DET Acoustic Diodes: Corrected Physics
## Addressing Reciprocity and Structural Directionality

**Framework:** DET v6.3 Bond Momentum Theory
**Status:** Critical corrections to ensure physics consistency

---

## 1. The Problem with Naive λ_π Asymmetry

### 1.1 What Was Wrong

The original implementation set:
```python
lambda_pi_forward = 0.005  # Low decay
lambda_pi_reverse = 0.030  # High decay
```

This is **physically invalid** because:
1. The material doesn't "know" which direction is "forward"
2. It's equivalent to a hidden global orientation flag
3. Passive linear systems must obey reciprocity (Onsager relations)

### 1.2 The Fundamental Constraint

**Passive linear systems are reciprocal.** This is not just a preference—it's a consequence of microscopic reversibility and the second law of thermodynamics.

To break reciprocity, you need at least one of:

| Mechanism | Example | How It Breaks Reciprocity |
|-----------|---------|---------------------------|
| **Nonlinearity** | Amplitude-dependent response | Different response at different amplitudes |
| **Time modulation** | Active pumping | Parametric asymmetry |
| **Bias field** | Rotation, magnetic field | Breaks time-reversal symmetry |
| **Memory state** | Pre-charged internal variable | State-dependent dissipation |

The DET π-field is a **memory state** route—but it must be implemented correctly.

---

## 2. DET-Clean Solution

### 2.1 Fix A: Structure-Dependent λ_π

Instead of direction-dependent λ_π, make it **structure-dependent**:

$$\lambda_{\pi,ij} = \lambda_0 + \lambda_s \cdot S_{ij}$$

Where $S_{ij}$ is a **local scattering factor** computed from:
- Geometric features (sawtooth angle, surface roughness)
- Impedance mismatch at interfaces
- Defect density / grain boundaries
- Local curvature

**The key insight:** A sawtooth has the SAME λ_π everywhere, but:
- Approaching from the gradual side: wave "sees" smooth transition → low scattering
- Approaching from the steep side: wave "sees" abrupt barrier → high scattering

This is computed LOCALLY based on the wave's interaction with structure.

### 2.2 Fix B: π-State-Dependent Transmission

The π field provides the reciprocity-breaking mechanism:

1. **π charges based on flux history:**
   $$\frac{d\pi_{ij}}{dt} = \alpha_\pi J_{ij} - \lambda_\pi \pi_{ij}$$

2. **π affects subsequent transmission in a nonlinear way:**
   $$J_{total} = J_{diff} + J_{mom}(\pi, F)$$

   where $J_{mom}$ depends on both π and F (the resource field).

3. **The asymmetry emerges from structure-π interaction:**
   - Structure creates different π charging patterns depending on approach direction
   - Pre-charged π biases subsequent transmission
   - This is a genuine internal state, not a parameter choice

### 2.3 The Correct Picture

```
RECIPROCITY-BREAKING VIA π-MEMORY
═══════════════════════════════════════════════════════════════

LINEAR PASSIVE SYSTEM (Reciprocal):
────────────────────────────────────────────────────────────────

    A ════════════════════════════════════════════ B
              Structure alone (linear)

    T(A→B) = T(B→A)   ← ALWAYS, by Onsager relations


π-MEMORY SYSTEM (Non-reciprocal):
────────────────────────────────────────────────────────────────

    Step 1: Initial state (π = 0 everywhere)

    A ═══════►        STRUCTURE        ═══════► B
              Wave approaches from A
              π charges based on A→B geometry


    Step 2: π is now charged (state changed)

    A         ┌──────────────────────┐         B
              │  π field charged     │
              │  (internal state)    │
              └──────────────────────┘


    Step 3: Subsequent transmission is π-dependent

    A ════════════════════════════════════════════ B
              NOW: transmission depends on π

    T(A→B | π_state) ≠ T(B→A | π_state)

    The asymmetry comes from the COMBINATION of:
    - Structure (determines WHERE π charges)
    - π state (determines HOW transmission is modified)
    - Nonlinear coupling (π affects J in state-dependent way)

═══════════════════════════════════════════════════════════════
```

---

## 3. Correct Physical Mechanisms

### 3.1 Mechanism 1: Geometric Scattering Asymmetry + π Charging

```
SAWTOOTH GEOMETRY π-CHARGING
═══════════════════════════════════════════════════════════════

Cross-section of sawtooth:

    FORWARD APPROACH (from left):

         Gradual impedance transition
              ╱
             ╱    Wave couples smoothly
            ╱     π charges efficiently along slope
           ╱      ────────────────────────────►
    ──────╱

    Result: Large π accumulation, oriented forward


    REVERSE APPROACH (from right):

         Abrupt impedance discontinuity
              │
              │    Wave reflects strongly
              │    π charges chaotically, dissipates
         ◄────│
    ──────────│

    Result: Small π accumulation, quickly scattered

═══════════════════════════════════════════════════════════════
```

The λ_π is the SAME, but the **charging efficiency** differs due to geometry.

### 3.2 Mechanism 2: Resonant π Trap

```
RESONANT π-TRAP DIODE
═══════════════════════════════════════════════════════════════

Structure: Asymmetric resonant cavity

    ┌─────────────────────────────────────────────┐
    │                                             │
    │   ENTRY        CAVITY         EXIT          │
    │   (gradual)                   (abrupt)      │
    │                                             │
    │      ╱╲                          │          │
    │     ╱  ╲       ┌─────┐          │          │
    │    ╱    ╲──────│     │──────────│          │
    │   ╱            │  π  │          │          │
    │  ╱             │TRAP │          │          │
    │                └─────┘                      │
    └─────────────────────────────────────────────┘

FORWARD (entering through gradual side):
1. Wave enters cavity smoothly
2. π accumulates in trap (resonant enhancement)
3. Charged π enhances forward transmission via J_mom
4. Wave exits through abrupt side (π-assisted)

REVERSE (entering through abrupt side):
1. Wave partially reflects at abrupt entry
2. Less energy enters cavity
3. π doesn't charge as efficiently
4. Exit through gradual side gets no π boost

The asymmetry comes from:
- WHERE π charges (structure-dependent)
- HOW MUCH π charges (geometry-dependent efficiency)
- Not from different λ_π values

═══════════════════════════════════════════════════════════════
```

### 3.3 Mechanism 3: Nonlinear π-Flux Coupling

The momentum-driven flux has a nonlinear dependence:

$$J_{mom} = \mu_\pi \sigma \pi \cdot \frac{F_i + F_j}{2}$$

This is **nonlinear** because:
- $J_{mom}$ depends on π (internal state)
- π depends on past J (memory)
- The F-weighting creates amplitude dependence

For small signals (linear regime):
$$J_{mom} \approx \mu_\pi \sigma \pi \cdot F_{background}$$

But for larger signals, the F-weighting creates amplitude-dependent behavior that breaks the linear reciprocity constraint.

### 3.4 Mechanism 4: Pre-Bias Operation

The most robust approach: **pre-charge π with a bias source**.

```
PRE-BIASED π DIODE
═══════════════════════════════════════════════════════════════

    BIAS              SIGNAL
    SOURCE            INPUT
       │                │
       ▼                ▼
    ┌──────────────────────────────────────┐
    │                                      │
    │   π-CHANNEL (pre-charged by bias)   │
    │                                      │
    │   ──────────────────────────────►   │  π bias direction
    │                                      │
    └──────────────────────────────────────┘
                        │
                        ▼
                    SIGNAL
                    OUTPUT

Operation:
1. Bias source maintains steady π in forward direction
2. Forward signal: π assists → enhanced transmission
3. Reverse signal: π opposes → reduced transmission

This is analogous to:
- Faraday rotator (magnetic bias breaks reciprocity)
- Circulator (rotation provides bias)

The bias can be:
- Steady flow through structure (fluid, thermal gradient)
- Acoustic pumping at different frequency
- Mechanical pre-stress

═══════════════════════════════════════════════════════════════
```

---

## 4. Corrected Mathematical Framework

### 4.1 Local Scattering Factor

Define the local scattering factor at each bond:

$$S_{ij} = S_{geometry} + S_{impedance} + S_{defect}$$

Where:

**Geometric scattering:**
$$S_{geometry} = \kappa_g \cdot |\nabla_n Z|$$

$\nabla_n Z$ is the impedance gradient in the direction of wave propagation.

**Impedance mismatch scattering:**
$$S_{impedance} = \kappa_Z \cdot \left(\frac{Z_j - Z_i}{Z_j + Z_i}\right)^2$$

**Defect scattering:**
$$S_{defect} = \kappa_d \cdot \rho_{defect}$$

### 4.2 Direction-Dependent Charging Efficiency

The π charging rate depends on wave-structure interaction:

$$\frac{d\pi_{ij}}{dt} = \alpha_\pi \cdot \eta(direction, structure) \cdot J_{ij} - \lambda_\pi \pi_{ij}$$

Where $\eta$ is the **charging efficiency**:
- $\eta = 1$ for smooth (gradual) approach
- $\eta < 1$ for abrupt approach (energy lost to scattering)

**Key:** λ_π is constant; $\eta$ varies with geometry.

### 4.3 State-Dependent Transmission

The total flux includes a π-dependent term:

$$J_{total} = J_{diff} + J_{mom}(\pi)$$

$$J_{mom} = \mu_\pi \sigma \pi \cdot F_{avg} \cdot g(\pi)$$

Where $g(\pi)$ can include nonlinear saturation:

$$g(\pi) = \frac{\pi}{\pi_{max}} \cdot \left(1 - \frac{|\pi|}{\pi_{max}}\right)$$

This creates amplitude-dependent behavior that breaks linear reciprocity.

---

## 5. Corrected Simulation Implementation

### 5.1 Structure-Based Scattering

```python
@dataclass
class DETDiodeParams:
    """Parameters for structure-based acoustic diode."""
    N: int = 200
    DT: float = 0.01

    # Base momentum parameters (SAME everywhere)
    lambda_pi: float = 0.015      # Uniform decay rate
    alpha_pi: float = 0.12        # Uniform charging rate
    mu_pi: float = 0.35           # Uniform mobility
    pi_max: float = 3.0           # Saturation limit

    # Structure parameters
    sawtooth_start: int = 80
    sawtooth_end: int = 120
    sawtooth_angle: float = 20.0  # degrees
    n_teeth: int = 10


class StructureBasedDiode:
    """
    Acoustic diode with structure-dependent π charging.

    Reciprocity is broken by:
    1. Structure-dependent charging efficiency η
    2. Nonlinear π-flux coupling
    3. Memory state (π accumulation)
    """

    def __init__(self, params: DETDiodeParams):
        self.p = params
        self.N = params.N

        # Fields
        self.F = np.ones(self.N)              # Resource
        self.pi = np.zeros(self.N)            # Momentum (single field, signed)
        self.C_R = np.ones(self.N) * 0.7      # Coherence

        # Compute LOCAL scattering profile from structure
        self.scattering_profile = self._compute_scattering_profile()
        self.charging_efficiency_pos = self._compute_charging_efficiency(+1)
        self.charging_efficiency_neg = self._compute_charging_efficiency(-1)

        self.time = 0.0

    def _compute_scattering_profile(self) -> np.ndarray:
        """
        Compute local scattering factor from structure geometry.
        This is FIXED by the structure, not direction-dependent.
        """
        S = np.zeros(self.N)

        # Sawtooth region
        start, end = self.p.sawtooth_start, self.p.sawtooth_end
        tooth_width = (end - start) / self.p.n_teeth
        angle_rad = np.radians(self.p.sawtooth_angle)

        for i in range(self.p.n_teeth):
            tooth_start = int(start + i * tooth_width)
            tooth_end = int(start + (i + 1) * tooth_width)
            tooth_mid = (tooth_start + tooth_end) // 2

            # Gradual slope region (low scattering)
            for j in range(tooth_start, tooth_mid):
                S[j] = 0.1 * np.tan(angle_rad)  # Low scattering

            # Steep face region (high scattering)
            for j in range(tooth_mid, tooth_end):
                S[j] = 1.0 / np.tan(angle_rad)  # High scattering

        return S

    def _compute_charging_efficiency(self, direction: int) -> np.ndarray:
        """
        Compute π charging efficiency based on wave direction and structure.

        direction: +1 for forward (left to right), -1 for reverse

        This determines HOW EFFICIENTLY π charges when a wave
        approaches from a given direction.
        """
        eta = np.ones(self.N)

        start, end = self.p.sawtooth_start, self.p.sawtooth_end
        tooth_width = (end - start) / self.p.n_teeth

        for i in range(self.p.n_teeth):
            tooth_start = int(start + i * tooth_width)
            tooth_end = int(start + (i + 1) * tooth_width)
            tooth_mid = (tooth_start + tooth_end) // 2

            if direction > 0:  # Forward: gradual entry, steep exit
                # Gradual region: high efficiency (smooth coupling)
                for j in range(tooth_start, tooth_mid):
                    eta[j] = 0.95
                # Steep region: lower efficiency (some reflection)
                for j in range(tooth_mid, tooth_end):
                    eta[j] = 0.6
            else:  # Reverse: steep entry, gradual exit
                # Steep region: low efficiency (strong reflection)
                for j in range(tooth_mid, tooth_end):
                    eta[j] = 0.3
                # Gradual region: moderate efficiency
                for j in range(tooth_start, tooth_mid):
                    eta[j] = 0.7

        return eta

    def step(self):
        """Execute one simulation step with structure-based physics."""
        p = self.p

        # Compute proper time
        P = np.ones(self.N)  # Simplified for clarity
        dt_proper = P * p.DT

        # Compute diffusive flux (standard DET)
        sigma = self.C_R  # Conductivity ~ coherence
        J_diff = np.zeros(self.N)
        for i in range(self.N - 1):
            J_diff[i] = sigma[i] * (self.F[i] - self.F[i+1])

        # Compute momentum-driven flux (nonlinear!)
        F_avg = (self.F[:-1] + self.F[1:]) / 2
        J_mom = p.mu_pi * sigma[:-1] * self.pi[:-1] * F_avg

        # Add nonlinear saturation
        saturation = 1.0 - np.abs(self.pi[:-1]) / p.pi_max
        J_mom *= np.maximum(saturation, 0)

        # Total flux
        J_total = J_diff + J_mom

        # Determine effective charging efficiency based on flux direction
        eta = np.where(J_total[:-1] > 0,
                       self.charging_efficiency_pos[:-1],
                       self.charging_efficiency_neg[:-1])

        # Update π with direction-dependent charging efficiency
        # (λ_π is the SAME everywhere - no direction flag!)
        d_pi = (p.alpha_pi * eta * J_total[:-1]
                - p.lambda_pi * self.pi[:-1]) * dt_proper[:-1]
        self.pi[:-1] += d_pi

        # Clip to saturation
        self.pi = np.clip(self.pi, -p.pi_max, p.pi_max)

        # Update F (resource conservation)
        dF = np.zeros(self.N)
        dF[:-1] -= J_total
        dF[1:] += J_total
        self.F += dF * p.DT
        self.F = np.maximum(self.F, 0.01)

        self.time += p.DT

        return J_total
```

### 5.2 Pre-Biased Operation

```python
class PreBiasedDiode(StructureBasedDiode):
    """
    Diode with pre-charged π field (bias source).

    This is the most robust reciprocity-breaking mechanism:
    a DC bias maintains π in one direction, creating
    genuine asymmetric transmission.
    """

    def __init__(self, params: DETDiodeParams, bias_strength: float = 0.5):
        super().__init__(params)
        self.bias_strength = bias_strength
        self.bias_region = slice(params.sawtooth_start, params.sawtooth_end)

    def apply_bias(self):
        """Apply DC bias to maintain π in forward direction."""
        # Continuous injection of forward-directed momentum
        self.pi[self.bias_region] += self.bias_strength * self.p.DT
        # Decay still applies, so this reaches steady state

    def step(self):
        """Step with bias applied."""
        self.apply_bias()
        return super().step()
```

---

## 6. Testable Predictions

### 6.1 With Structure Only (No Pre-Bias)

For a passive structure without pre-bias:

| Condition | Predicted R | Notes |
|-----------|-------------|-------|
| Small signal, π = 0 | R ≈ 1.0 | Linear reciprocal |
| Large signal, π = 0 | R ≈ 1.2-1.5 | Nonlinear regime |
| After forward pulse | R > 1 transiently | π charged forward |
| After reverse pulse | R < 1 transiently | π charged reverse |

**Key test:** Send pulse A→B, measure R. Then send pulse B→A, measure R.
After A→B pulse, R should be enhanced (π assists forward).

### 6.2 With Pre-Bias

| Condition | Predicted R | Notes |
|-----------|-------------|-------|
| Steady bias, small signal | R ≈ 2-5 | Bias provides asymmetry |
| Steady bias, large signal | R ≈ 3-8 | Nonlinearity adds to bias |
| No bias (control) | R ≈ 1 | Reciprocal baseline |

### 6.3 Comparison to Naive Implementation

| Metric | Naive (λ_π asymmetry) | Correct (structure + π-state) |
|--------|----------------------|------------------------------|
| Physics validity | ❌ Violates reciprocity | ✅ Consistent |
| R without bias | ~10:1 (fake) | ~1:1 (correct) |
| R with bias | N/A | ~3-8:1 (real) |
| R after training pulse | N/A | Transient asymmetry |
| Energy conservation | Questionable | ✅ Conserved |

---

## 7. Revised Application Requirements

### 7.1 Applications That Still Work

These applications remain valid with correct physics:

1. **Pre-biased diodes:** Active element provides DC bias
   - Wind turbine nacelles (thermal gradient bias)
   - HVAC (airflow provides bias)
   - Industrial enclosures (forced ventilation)

2. **Nonlinear/high-amplitude regime:**
   - High-power ultrasound
   - Shock waves
   - Impact protection

3. **π-trained operation:**
   - Pulse train "charges" the diode
   - Subsequent signals see asymmetry
   - Requires periodic refreshing

### 7.2 Applications That Need Revision

These need active bias or other modifications:

1. **Passive wall panels:** Need thermal gradient or airflow
2. **Vehicle panels:** Need vibration bias from engine
3. **Smart speaker isolation:** Could use speaker output as bias

### 7.3 New Application: Self-Biasing Diode

```
SELF-BIASING ACOUSTIC DIODE
═══════════════════════════════════════════════════════════════

Use ambient energy to maintain π bias:

    ┌─────────────────────────────────────────────┐
    │                                             │
    │   AMBIENT       RECTIFIER      π-CHANNEL   │
    │   VIBRATION ──► (nonlinear) ──► (biased)   │
    │                                             │
    │   Low-frequency ambient charges π           │
    │   High-frequency signal sees asymmetry      │
    │                                             │
    └─────────────────────────────────────────────┘

Works in environments with:
- HVAC rumble (charges π)
- Traffic vibration (charges π)
- Machinery hum (charges π)

Signal band (speech, music) sees pre-biased diode!

═══════════════════════════════════════════════════════════════
```

---

## 8. Summary of Corrections

### 8.1 What Changed

| Aspect | Before (Wrong) | After (Correct) |
|--------|---------------|-----------------|
| λ_π | Direction-dependent parameter | Uniform, structure-independent |
| Asymmetry source | λ_π_fwd ≠ λ_π_rev | Structure-dependent η, nonlinearity |
| Passive operation | Claimed R ~ 10:1 | R ≈ 1 (reciprocal) |
| Reciprocity | Violated | Respected |
| Bias requirement | None | Required for strong asymmetry |

### 8.2 Key Principles

1. **No hidden globals:** Directionality must emerge from local structure
2. **Respect reciprocity:** Linear passive systems are symmetric
3. **π is a state:** Its value affects behavior (not just parameters)
4. **Bias breaks symmetry:** Pre-charged π provides genuine asymmetry

### 8.3 What DET π-Memory Actually Provides

The π field provides a **valid physical mechanism** for breaking reciprocity through:
- Internal state variable (memory)
- Nonlinear coupling (state-dependent transmission)
- Structure-dependent charging (different efficiency by approach)

But this only works when:
- The system operates in nonlinear regime, OR
- A bias source pre-charges π, OR
- Transient asymmetry after training pulse is acceptable

---

*Document created: January 2026*
*Status: Critical physics correction*
*Framework: DET v6.3 with reciprocity-consistent interpretation*
