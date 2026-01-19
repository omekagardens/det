# Acoustic Diodes via DET Momentum Dynamics: A Deep Design Study

**Framework:** DET v6.3 Bond Momentum (Section IV.4)
**Application:** Non-reciprocal acoustic wave transmission

---

## 1. Physical Foundation: Mapping Acoustics to DET

### 1.1 The Core Correspondence

| Acoustic Quantity | DET Variable | Physical Basis |
|-------------------|--------------|----------------|
| Acoustic pressure | F (resource) | Energy density field |
| Particle velocity | J (flux) | Transport rate |
| Phonon momentum | π (bond momentum) | Directed transport memory |
| Acoustic impedance | σ (conductivity) | Transport coefficient |
| Attenuation | λ_π (decay rate) | Energy dissipation |
| Wave coherence | C (coherence) | Phase correlation |

### 1.2 Why DET Captures Acoustic Rectification

Traditional acoustics assumes **linear reciprocity**: if sound travels from A→B, it travels equally from B→A. This follows from the symmetry of the wave equation.

DET breaks this symmetry through the **momentum memory field π**:

```
Standard Wave Equation (reciprocal):
    ∂²p/∂t² = c² ∇²p

DET Acoustic Model (non-reciprocal):
    ∂²F/∂t² = c² ∇²F + μ_π ∇·(π F)
                        └── Momentum-driven term breaks symmetry
```

The π field accumulates from past flux and drives continued transport, creating **history-dependent, directional behavior**.

### 1.3 The Rectification Mechanism

**Forward propagation (+x):**
1. Wave creates flux J > 0
2. Flux charges momentum: dπ/dt = α_π J - λ_π^(+) π
3. Low λ_π^(+) → momentum persists
4. Persistent π drives additional forward flux
5. **Result:** Enhanced transmission

**Reverse propagation (-x):**
1. Wave creates flux J < 0
2. Flux charges reverse momentum
3. High λ_π^(-) → momentum dissipates quickly
4. Little momentum-driven boost
5. **Result:** Attenuated transmission

**Rectification Ratio:**
$$R = \frac{T_{forward}}{T_{reverse}} \approx \frac{\lambda_\pi^{(-)}}{\lambda_\pi^{(+)}}$$

---

## 2. Acoustic Diode Architectures

### 2.1 Architecture A: Asymmetric Phononic Crystal

**Concept:** Use geometric asymmetry in a phononic crystal to create direction-dependent scattering.

```
ASYMMETRIC PHONONIC CRYSTAL (Top View)
═══════════════════════════════════════════════════════════════

Direction →  (Forward: High transmission)

    ╱│    ╱│    ╱│    ╱│    ╱│    ╱│    ╱│    ╱│
   ╱ │   ╱ │   ╱ │   ╱ │   ╱ │   ╱ │   ╱ │   ╱ │
  ╱  │  ╱  │  ╱  │  ╱  │  ╱  │  ╱  │  ╱  │  ╱  │
 ────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴────

Direction ←  (Reverse: Low transmission)

    │╲    │╲    │╲    │╲    │╲    │╲    │╲    │╲
    │ ╲   │ ╲   │ ╲   │ ╲   │ ╲   │ ╲   │ ╲   │ ╲
    │  ╲  │  ╲  │  ╲  │  ╲  │  ╲  │  ╲  │  ╲  │  ╲
 ────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴────

═══════════════════════════════════════════════════════════════
```

**DET Mapping:**
- Gradual slope (forward): Adiabatic impedance matching → low scattering → low λ_π
- Steep face (reverse): Impedance mismatch → strong scattering → high λ_π

**Design Parameters:**
| Parameter | Symbol | Typical Value | Effect |
|-----------|--------|---------------|--------|
| Tooth angle | θ | 15-30° | Steeper = more asymmetry |
| Period | Λ | λ_acoustic/4 | Resonant enhancement |
| Tooth height | h | Λ/2 | Scattering strength |
| Number of periods | N | 10-50 | Rectification builds up |

**DET Parameters:**
```python
# Forward direction (along gradual slope)
lambda_pi_forward = 0.005  # Low decay - momentum persists
alpha_pi = 0.15            # Standard charging

# Reverse direction (against steep face)
lambda_pi_reverse = 0.040  # High decay - momentum scattered

# Rectification ratio
R_geometry = lambda_pi_reverse / lambda_pi_forward  # = 8:1
```

### 2.2 Architecture B: Graded Impedance Stack

**Concept:** Create a stack of layers with monotonically varying acoustic impedance.

```
GRADED IMPEDANCE ACOUSTIC DIODE (Cross-section)
═══════════════════════════════════════════════════════════════

         Z₁        Z₂        Z₃        Z₄        Z₅
      (low Z)                                  (high Z)
    ┌─────────┬─────────┬─────────┬─────────┬─────────┐
    │░░░░░░░░░│▒▒▒▒▒▒▒▒▒│▓▓▓▓▓▓▓▓▓│█████████│█████████│
    │░░░░░░░░░│▒▒▒▒▒▒▒▒▒│▓▓▓▓▓▓▓▓▓│█████████│█████████│
    │░░░░░░░░░│▒▒▒▒▒▒▒▒▒│▓▓▓▓▓▓▓▓▓│█████████│█████████│
    │░░░░░░░░░│▒▒▒▒▒▒▒▒▒│▓▓▓▓▓▓▓▓▓│█████████│█████████│
    └─────────┴─────────┴─────────┴─────────┴─────────┘

    ══════════════════════════════════════════════════
         Z = ρc increases monotonically →
    ══════════════════════════════════════════════════

Forward (Low Z → High Z): Gradual impedance increase
    • Each interface: small reflection
    • Phonons thermalize gradually
    • Momentum transfers adiabatically
    • λ_π effective: LOW

Reverse (High Z → Low Z): Abrupt impedance decrease
    • Each interface: large reflection
    • Strong backscattering
    • Momentum rapidly randomized
    • λ_π effective: HIGH

═══════════════════════════════════════════════════════════════
```

**Physics:**

At each interface, reflection coefficient:
$$r = \frac{Z_2 - Z_1}{Z_2 + Z_1}$$

**Forward (Z increasing):** Z₂ > Z₁ → r > 0 (small positive)
- Transmitted wave slightly phase-shifted
- Momentum gradually transferred to heavier medium
- Coherent addition of transmitted waves

**Reverse (Z decreasing):** Z₂ < Z₁ → r < 0 (negative)
- Strong reflection at each interface
- Multiple scattering randomizes momentum
- Destructive interference

**Material Stack Example:**
| Layer | Material | Density (kg/m³) | Sound Speed (m/s) | Z (MRayl) |
|-------|----------|-----------------|-------------------|-----------|
| 1 | Silicone rubber | 1100 | 1000 | 1.1 |
| 2 | PMMA | 1180 | 2750 | 3.2 |
| 3 | Aluminum | 2700 | 6420 | 17.3 |
| 4 | Steel | 7800 | 5960 | 46.5 |
| 5 | Tungsten | 19300 | 5220 | 100.7 |

**Impedance ratios:**
- Layer 1→2: Z₂/Z₁ = 2.9
- Layer 2→3: Z₂/Z₁ = 5.4
- Layer 3→4: Z₂/Z₁ = 2.7
- Layer 4→5: Z₂/Z₁ = 2.2

Total impedance ratio: Z₅/Z₁ = 91.5 (excellent asymmetry)

### 2.3 Architecture C: Nonlinear Acoustic Diode

**Concept:** Use amplitude-dependent nonlinearity to create rectification.

```
NONLINEAR ACOUSTIC DIODE
═══════════════════════════════════════════════════════════════

    ┌────────────────────────────────────────────────────┐
    │                                                    │
    │   Linear      Nonlinear        Linear              │
    │   Medium      Element          Medium              │
    │              ┌──────┐                              │
    │   ░░░░░░░░░░░│██████│░░░░░░░░░░░░░░░░░░░          │
    │   ░░░░░░░░░░░│██████│░░░░░░░░░░░░░░░░░░░          │
    │   ░░░░░░░░░░░│██████│░░░░░░░░░░░░░░░░░░░          │
    │              └──────┘                              │
    │                 │                                  │
    │                 ▼                                  │
    │         Asymmetric stiffness:                      │
    │         K(+) ≠ K(-)                                │
    │                                                    │
    └────────────────────────────────────────────────────┘

Forward compression → Stiff → High transmission
Reverse compression → Soft → Low transmission (absorbed)

═══════════════════════════════════════════════════════════════
```

**DET Mapping:**

The nonlinear element has direction-dependent properties:

$$\sigma(J) = \begin{cases} \sigma_+ & J > 0 \text{ (forward)} \\ \sigma_- & J < 0 \text{ (reverse)} \end{cases}$$

where σ₊ > σ₋.

**In DET terms:**
- Forward: High σ → high flux → high momentum charging
- Reverse: Low σ → low flux → minimal momentum buildup

**Nonlinear Element Options:**
1. **Granular chain with asymmetric contacts**
2. **Pre-stressed membrane**
3. **Buckling beam array**
4. **Magnetic-mechanical coupling**

### 2.4 Architecture D: Active Acoustic Diode (π-Feedback Enhanced)

**Concept:** Use the DET π-feedback mechanism to create active amplification in the forward direction.

```
ACTIVE π-FEEDBACK ACOUSTIC DIODE
═══════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   INPUT    MOMENTUM      FEEDBACK      MOMENTUM  OUTPUT │
    │    ───►    CHARGING      AMPLIFIER     CHANNEL    ───►  │
    │            ZONE          (F > F_crit)   ZONE            │
    │                                                         │
    │   ┌────┐   ┌────────┐   ┌────────┐   ┌────────┐  ┌────┐│
    │   │    │   │ α_π ↑  │   │ G = λ_π│   │ μ_π ↑  │  │    ││
    │   │ IN │──►│ charges│──►│ ───────│──►│ drives │──►│OUT ││
    │   │    │   │   π    │   │  λ_eff │   │  flux  │  │    ││
    │   └────┘   └────────┘   └────────┘   └────────┘  └────┘│
    │                              │                          │
    │                              │ F_avg controlled         │
    │                              │ to set gain              │
    │                              ▼                          │
    │                    ┌──────────────────┐                 │
    │                    │  GAIN CONTROL    │                 │
    │                    │  (Stress/Temp)   │                 │
    │                    └──────────────────┘                 │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

    Forward: Input → π charges → Amplified by feedback → High output
    Reverse: Input → π dies quickly → No amplification → Low output

═══════════════════════════════════════════════════════════════
```

**Key Equations:**

Gain in forward direction:
$$G_{forward} = \frac{\lambda_\pi}{\lambda_\pi - \alpha_\pi \mu_\pi \sigma F_{avg}}$$

For F_avg approaching F_crit:
$$G_{forward} \to \infty$$

Reverse direction (high λ_π):
$$G_{reverse} \approx 1 \text{ (no amplification)}$$

**Effective Rectification:**
$$R_{active} = G_{forward} \times R_{passive}$$

With G = 5x and R_passive = 8:1:
$$R_{active} = 5 \times 8 = 40:1$$

---

## 3. Frequency-Dependent Analysis

### 3.1 Frequency Scaling of Rectification

The rectification ratio depends on frequency through the momentum dynamics:

**Momentum charging time:**
$$\tau_{charge} = \frac{1}{\alpha_\pi \omega}$$

**Momentum decay time:**
$$\tau_{decay} = \frac{1}{\lambda_\pi}$$

**Optimal frequency range:**
$$\frac{1}{\tau_{decay}} < \omega < \frac{1}{\tau_{charge}}$$

$$\lambda_\pi < \omega < \alpha_\pi^{-1}$$

With α_π = 0.15 and λ_π = 0.005:
$$0.005 < \omega < 6.7$$

In physical units (assuming DET time unit = 1 μs):
$$5 \text{ kHz} < f < 1 \text{ MHz}$$

### 3.2 Frequency-Dependent Rectification Curve

```
RECTIFICATION vs FREQUENCY
═══════════════════════════════════════════════════════════════

    R (rectification ratio)
    │
 10 ┤                    ┌─────────────────┐
    │                   ╱                   ╲
  8 ┤                  ╱                     ╲
    │                 ╱                       ╲
  6 ┤                ╱         OPTIMAL         ╲
    │               ╱          BAND             ╲
  4 ┤              ╱                             ╲
    │             ╱                               ╲
  2 ┤────────────╱                                 ╲──────────
    │   LOW f                                        HIGH f
  1 ┼────────────┼─────────────┼─────────────┼────────────────
    │          f_low        f_opt          f_high
    │
    └─────────────────────────────────────────────────────────►
                                                         f (Hz)

    f_low  = λ_π / (2π) ≈ λ_π × 160 kHz  (if DET unit = 1 μs)
    f_high = 1 / (α_π × 2π) ≈ 1 MHz
    f_opt  = √(f_low × f_high) ≈ 400 kHz

═══════════════════════════════════════════════════════════════
```

**Physical Interpretation:**

- **Low frequency (f < f_low):**
  - Wave period >> momentum decay time
  - Momentum decays between cycles
  - No buildup → R ≈ 1

- **Optimal frequency (f_low < f < f_high):**
  - Momentum charges and persists across cycles
  - Maximum asymmetry effect
  - R → R_max

- **High frequency (f > f_high):**
  - Wave period << charging time
  - Momentum can't respond fast enough
  - Averages out → R ≈ 1

### 3.3 Bandwidth Optimization

**For broadband operation:** Use multiple cascaded stages with different resonances.

```
BROADBAND ACOUSTIC DIODE (Cascaded Design)
═══════════════════════════════════════════════════════════════

    ┌──────────────┬──────────────┬──────────────┐
    │   STAGE 1    │   STAGE 2    │   STAGE 3    │
    │  f₁ = 50kHz  │  f₂ = 200kHz │  f₃ = 800kHz │
    │   R₁ = 5     │   R₂ = 5     │   R₃ = 5     │
    └──────────────┴──────────────┴──────────────┘

    Combined: R_total = R₁ × R₂ × R₃ = 125
    Bandwidth: 30 kHz - 1.2 MHz

═══════════════════════════════════════════════════════════════
```

---

## 4. Detailed Design: 100 kHz Ultrasonic Diode

### 4.1 Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Center frequency | 100 kHz | Ultrasonic range |
| Bandwidth | 50-200 kHz | 2 octaves |
| Rectification ratio | >10:1 | Forward/reverse power |
| Insertion loss (forward) | <3 dB | Minimal attenuation |
| Insertion loss (reverse) | >13 dB | Strong attenuation |
| Size | 50 mm × 50 mm × 20 mm | Compact |

### 4.2 Architecture Selection

For 100 kHz, wavelength in typical solids:
- Aluminum: λ = 64 mm
- PMMA: λ = 27 mm
- Water: λ = 15 mm

**Selected: Graded Impedance Stack + Asymmetric Scatterers**

### 4.3 Layer Design

```
100 kHz ULTRASONIC DIODE LAYER STRUCTURE
═══════════════════════════════════════════════════════════════

         ┌─────────────────────────────────────────────────┐
         │                                                 │
    ───► │  MATCHING   GRADED STACK    SAWTOOTH   OUTPUT  │ ───►
         │   LAYER                      ARRAY              │
         │                                                 │
         │  ┌─────┐ ┌───┬───┬───┬───┐ ┌─────────┐ ┌─────┐ │
         │  │     │ │   │   │   │   │ │╱│╱│╱│╱│╱│ │     │ │
         │  │ L1  │ │L2 │L3 │L4 │L5 │ │╱│╱│╱│╱│╱│ │ L6  │ │
         │  │     │ │   │   │   │   │ │╱│╱│╱│╱│╱│ │     │ │
         │  └─────┘ └───┴───┴───┴───┘ └─────────┘ └─────┘ │
         │                                                 │
         └─────────────────────────────────────────────────┘

Layer specifications:
═══════════════════════════════════════════════════════════════
Layer  Material        Thickness   Z (MRayl)   Purpose
═══════════════════════════════════════════════════════════════
L1     Silicone gel    2.0 mm      1.0         Input matching
L2     LDPE            1.5 mm      1.8         Gradient start
L3     HDPE            1.5 mm      2.4         Gradient
L4     PMMA            2.0 mm      3.2         Gradient
L5     Aluminum        3.0 mm      17.3        Gradient end
L6     Steel           2.0 mm      46.5        Output/momentum channel
═══════════════════════════════════════════════════════════════

Sawtooth array:
- Material: Aluminum
- Tooth angle: 20°
- Tooth pitch: 1.5 mm (λ/4 at 100 kHz)
- Number of teeth: 30
- Total length: 45 mm

═══════════════════════════════════════════════════════════════
```

### 4.4 DET Parameter Mapping

**For this design:**

```python
# Physical parameters
f_center = 100e3  # Hz
rho_steel = 7800  # kg/m³
c_steel = 5960    # m/s
wavelength = c_steel / f_center  # 59.6 mm

# DET time unit calibration
# Choose: 1 DET time unit = 1 microsecond
dt_physical = 1e-6  # seconds
omega_DET = 2 * np.pi * f_center * dt_physical  # = 0.628

# DET parameters for sawtooth section
lambda_pi_forward = 0.003   # Very low decay (momentum persists)
lambda_pi_reverse = 0.030   # High decay (momentum scatters)
alpha_pi = 0.12             # Charging rate
mu_pi = 0.40                # Momentum mobility

# Predicted rectification
R_sawtooth = lambda_pi_reverse / lambda_pi_forward  # = 10:1

# F_avg in steel (normalized)
F_avg = 0.5  # Mid-range

# Check stability for active enhancement
F_crit = lambda_pi_forward / (alpha_pi * mu_pi * 1.0)  # = 0.0625
# F_avg > F_crit: UNSTABLE → Can use as amplifier!

# If we use active enhancement:
lambda_eff = lambda_pi_forward - alpha_pi * mu_pi * F_avg
# lambda_eff = 0.003 - 0.12 * 0.40 * 0.5 = 0.003 - 0.024 = -0.021
# Negative! System will amplify in forward direction.

# For stable operation, reduce F_avg:
F_stable = 0.04
lambda_eff_stable = 0.003 - 0.12 * 0.40 * 0.04  # = 0.0011
G_forward = lambda_pi_forward / lambda_eff_stable  # = 2.7x

# Total rectification with active enhancement
R_total = R_sawtooth * G_forward  # = 10 * 2.7 = 27:1
```

### 4.5 Performance Predictions

| Metric | Passive Only | With π-Feedback | Notes |
|--------|--------------|-----------------|-------|
| Forward transmission | -2 dB | -1 dB | Enhanced by feedback |
| Reverse transmission | -12 dB | -15 dB | Blocked |
| Rectification ratio | 10:1 | 27:1 | 14 dB improvement |
| Bandwidth | 50-200 kHz | 50-200 kHz | Same |
| Power handling | >1 W | <0.5 W | Feedback limits power |

---

## 5. Advanced Concepts

### 5.1 Acoustic Transistor (Three-Terminal Device)

**Concept:** Use a control port to modulate the rectification.

```
ACOUSTIC TRANSISTOR (DET-Based)
═══════════════════════════════════════════════════════════════

                      CONTROL (Gate)
                          │
                          ▼
                    ┌──────────┐
                    │ F_avg    │
                    │ modulator│
                    └────┬─────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │   SOURCE    ┌──────┴──────┐    DRAIN    │
    │    ───►     │  π-CHANNEL  │     ───►    │
    │             │  G = f(F)   │             │
    │             └─────────────┘             │
    │                                         │
    └─────────────────────────────────────────┘

Control signal modulates F_avg:
- F_avg ↑ → Gain ↑ → More transmission
- F_avg ↓ → Gain ↓ → Less transmission

Acoustic equivalent of a MOSFET!

═══════════════════════════════════════════════════════════════
```

**Control mechanisms:**
1. **Thermal:** Temperature changes elastic modulus (F)
2. **Mechanical:** Applied stress changes F
3. **Electromagnetic:** Magnetostriction or piezoelectric coupling
4. **Optical:** Photothermal effect

### 5.2 Acoustic Logic Gates

Using acoustic transistors, build logic:

```
ACOUSTIC AND GATE
═══════════════════════════════════════════════════════════════

    INPUT A ───┐
               ├──► ACOUSTIC ──► OUTPUT
    INPUT B ───┘    TRANSISTOR

    Truth table:
    A  B  │ OUT
    ──────┼────
    0  0  │  0   (Both low F → low gain → no output)
    0  1  │  0
    1  0  │  0
    1  1  │  1   (Both high F → high gain → output)

═══════════════════════════════════════════════════════════════
```

### 5.3 Acoustic Circulator (Three-Port Device)

```
ACOUSTIC CIRCULATOR
═══════════════════════════════════════════════════════════════

                    PORT 2
                      ▲
                      │
                ┌─────┴─────┐
                │           │
    PORT 1 ────►│  ROTATING │────► PORT 3
                │  π-FIELD  │
                │           │
                └───────────┘

    Circulation pattern:
    1 → 2 → 3 → 1 → ...

    Each path has asymmetric λ_π configured for:
    - 1→2: Low λ_π (high transmission)
    - 2→1: High λ_π (blocked)
    - etc.

═══════════════════════════════════════════════════════════════
```

---

## 6. Fabrication Approaches

### 6.1 Sawtooth Scatterers

**Method 1: Micromachining**
- Material: Silicon or aluminum
- Process: DRIE (Deep Reactive Ion Etching) or micromilling
- Resolution: 10-100 μm features
- Frequency range: 100 kHz - 10 MHz

**Method 2: 3D Printing**
- Material: Metal (SLM) or polymer (SLA)
- Process: Additive manufacturing
- Resolution: 50-200 μm
- Frequency range: 20-500 kHz

**Method 3: Molding**
- Material: Silicone or epoxy
- Process: Cast from machined master
- Resolution: 10-50 μm
- Frequency range: 50 kHz - 2 MHz

### 6.2 Graded Impedance Stack

**Method: Layer bonding**
1. Prepare individual layers (cut, polish)
2. Apply thin coupling layer (epoxy, gel)
3. Stack and compress
4. Cure if needed
5. Final machining to size

**Critical:** Minimize interface reflections with good acoustic coupling.

### 6.3 Active Enhancement (Feedback Control)

**F_avg modulation methods:**

1. **Static stress:**
   - Bolt torque on stack
   - Simple but not tunable

2. **Piezoelectric:**
   - PZT actuator applies controlled stress
   - Bandwidth: DC - 100 kHz
   - Voltage controlled

3. **Magnetostrictive:**
   - Terfenol-D or Galfenol element
   - Current controlled
   - Bandwidth: DC - 10 kHz

---

## 7. Applications

### 7.1 Ultrasonic Isolation

**Problem:** Ultrasonic sensors pick up reflections from their own emissions.

**Solution:** Place acoustic diode between transducer and environment.
- Forward: Emission passes through
- Reverse: Echoes don't return to transducer
- Result: Clean one-way sensing

### 7.2 Acoustic Energy Harvesting

**Problem:** Vibration energy harvesters work bidirectionally, limiting efficiency.

**Solution:** Acoustic diode + resonant cavity
- Vibrations enter from any direction
- Diode rectifies flow into cavity
- Cavity accumulates energy
- One-way valve prevents backflow

```
ACOUSTIC ENERGY HARVESTER
═══════════════════════════════════════════════════════════════

    AMBIENT        ACOUSTIC       RESONANT        ENERGY
    VIBRATION  ──►   DIODE    ──►  CAVITY    ──►  HARVESTER
                      │              │
                      │              └── Accumulates energy
                      └── Blocks backflow

═══════════════════════════════════════════════════════════════
```

### 7.3 Noise Control

**Problem:** Sound transmission through walls/barriers is bidirectional.

**Solution:** Asymmetric barrier with acoustic diode
- One direction: Normal transmission (allow speech)
- Reverse direction: Blocked (prevent noise ingress)

### 7.4 Medical Ultrasound

**Problem:** Imaging artifacts from multiple reflections.

**Solution:** Diode at transducer face
- Outgoing pulses: Pass through
- Incoming echoes: Pass through (diode is direction-selective, not amplitude-selective)
- Internal reflections: Blocked by diode

---

## 8. Summary: DET Design Principles for Acoustic Diodes

### 8.1 Key Parameters

| DET Parameter | Effect on Acoustic Diode | Optimization |
|---------------|-------------------------|--------------|
| λ_π(forward) | Memory persistence | Minimize for high R |
| λ_π(reverse) | Scattering rate | Maximize for high R |
| α_π | Charging sensitivity | Match to frequency |
| μ_π | Momentum-flux coupling | Increase for gain |
| F_avg | Operating point | Control for gain/stability |
| C | Wave coherence | High for wave-like transport |

### 8.2 Design Checklist

1. **Choose architecture** based on frequency and application
2. **Calculate λ_π ratio** for required rectification
3. **Verify frequency range** is in optimal band
4. **Check F_crit** if using active enhancement
5. **Select materials** for impedance matching
6. **Design geometry** for target λ_π asymmetry
7. **Validate with DET simulation** before fabrication

### 8.3 Fundamental Limits

**Maximum passive rectification:**
$$R_{max} \approx \frac{\lambda_\pi^{max}}{\lambda_\pi^{min}} \approx 10-20$$

Limited by achievable λ_π asymmetry in real materials.

**Maximum active rectification:**
$$R_{active} = R_{passive} \times G_{feedback}$$

Limited by stability (F < F_crit) and nonlinear effects.

**Bandwidth-rectification tradeoff:**
Higher R requires more resonant structures → narrower bandwidth.

---

## Appendix A: DET Acoustic Diode Simulation Code

See `det_v6_3/tests/test_acoustic_diode.py` for implementation.

---

*Document created: January 2026*
*Framework: DET v6.3 Momentum Dynamics*
*Application: Non-reciprocal Acoustic Wave Transmission*
