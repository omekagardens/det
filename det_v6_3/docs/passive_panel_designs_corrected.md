# Passive Panel Designs: Physics-Corrected Approach
## Self-Biasing and Ambient-Driven Acoustic Diodes

**Framework:** DET v6.3 with reciprocity-consistent interpretation
**Status:** Updated designs respecting Onsager reciprocity constraints

---

## 1. The Challenge: Passive ≠ Parameter Asymmetry

### 1.1 What We Can't Do

A truly passive, linear system **cannot** have direction-dependent transmission:

```
FORBIDDEN (violates Onsager):
════════════════════════════════════════════════════════════════════

    PASSIVE LINEAR PANEL
    ┌────────────────────────────────────────┐
    │                                        │
    │   λ_π(→) = 0.005  ← WRONG!            │
    │   λ_π(←) = 0.050  ← Can't do this     │
    │                                        │
    └────────────────────────────────────────┘

    This would require the material to "know" which way is forward
    without any external reference or internal state.

════════════════════════════════════════════════════════════════════
```

### 1.2 What We Can Do

Non-reciprocity requires breaking time-reversal symmetry through:

1. **Self-biasing** - Ambient energy maintains π state
2. **Thermal gradient** - Temperature difference creates flow
3. **Airflow coupling** - Ventilation provides bias
4. **Nonlinear operation** - High amplitude regime
5. **Periodic training** - Pulsed refresh of π state

---

## 2. Self-Biasing Panel Design

### 2.1 Core Concept

Use **low-frequency ambient energy** to charge π, creating a bias that affects **higher-frequency signals**.

```
SELF-BIASING ACOUSTIC PANEL
════════════════════════════════════════════════════════════════════

    AMBIENT                    SIGNAL
    (low freq)                 (speech, music)
        │                          │
        ▼                          ▼
    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    │   ┌──────────────┐    ┌──────────────┐    ┌────────────┐  │
    │   │ LOW-PASS     │    │ π-CHARGING   │    │ SIGNAL     │  │
    │   │ RESONATOR    │───►│ STRUCTURE    │───►│ CHANNEL    │  │
    │   │ (collects    │    │ (sawtooth)   │    │ (sees bias)│  │
    │   │  LF energy)  │    │              │    │            │  │
    │   └──────────────┘    └──────────────┘    └────────────┘  │
    │                                                            │
    │   Ambient rumble        π charges         Signal sees      │
    │   (HVAC, traffic,       in forward        pre-biased       │
    │    footsteps)           direction         channel          │
    │                                                            │
    └────────────────────────────────────────────────────────────┘

FREQUENCY SEPARATION:
- Ambient/bias band: 10-100 Hz (building rumble, HVAC)
- Signal band: 200 Hz - 8 kHz (speech, music)

The π field integrates LF energy → creates DC bias → affects HF transmission

════════════════════════════════════════════════════════════════════
```

### 2.2 Design Parameters

| Parameter | Value | Function |
|-----------|-------|----------|
| LF resonator frequency | 20-50 Hz | Captures building rumble |
| LF resonator Q | 2-5 | Broadband collection |
| Structure region | Sawtooth, 8-12 teeth | Directional π charging |
| Signal band | 200 Hz - 8 kHz | Protected from structure resonance |
| λ_π | 0.01-0.02 | Slow decay retains bias |
| Expected R | 1.5-3.0 | With typical ambient levels |

### 2.3 Physical Implementation

```
SELF-BIASING PANEL CROSS-SECTION
════════════════════════════════════════════════════════════════════

    NOISE SIDE                              QUIET SIDE
        │                                        │
        ▼                                        ▼
    ┌───────┬─────────────────────────────┬───────┐
    │ FACE  │                             │ FACE  │
    │ SHEET │    INTERNAL STRUCTURE       │ SHEET │
    │       │                             │       │
    │   ░   │  ╔══════════════════════╗   │   ░   │
    │   ░   │  ║ ╱╲    ╱╲    ╱╲    ╱╲ ║   │   ░   │
    │   ░   │  ║╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲║   │   ░   │
    │   ░   │  ║    ╲╱    ╲╱    ╲╱    ║   │   ░   │
    │   ░   │  ╚══════════════════════╝   │   ░   │
    │   ░   │       SAWTOOTH CORE         │   ░   │
    │   ░   │                             │   ░   │
    │   ░   │  ┌──────────────────────┐   │   ░   │
    │   ░   │  │    LF RESONATOR      │   │   ░   │
    │   ░   │  │    (tuned mass)      │   │   ░   │
    │   ░   │  └──────────────────────┘   │   ░   │
    │       │                             │       │
    └───────┴─────────────────────────────┴───────┘

    Face sheets: Cork or dense fiber (impedance matching)
    Sawtooth core: Bamboo or oriented fiber composite
    LF resonator: Mass-spring system tuned to building rumble

════════════════════════════════════════════════════════════════════
```

### 2.4 Materials (Renewable, Low-Cost)

| Component | Material Options | Est. Cost/m² |
|-----------|-----------------|--------------|
| Face sheets | Cork (3mm), hemp board | $8-15 |
| Sawtooth core | Bamboo strips, corrugated cardboard | $5-12 |
| LF resonator | Sand bag + rubber springs | $3-8 |
| Bonding | Natural latex, starch paste | $2-5 |
| **Total** | | **$18-40/m²** |

---

## 3. Thermal Gradient Panel

### 3.1 Concept

Indoor-outdoor temperature difference creates natural convection bias.

```
THERMAL GRADIENT ACOUSTIC PANEL
════════════════════════════════════════════════════════════════════

    OUTDOOR              PANEL              INDOOR
    (cold)                                  (warm)
      │                                        │
      ▼                                        ▼
    ┌─────┬────────────────────────────┬─────┐
    │     │    POROUS STRUCTURE        │     │
    │ T₁  │    with π-channels         │ T₂  │
    │     │                            │     │
    │     │  ╔════════════════════╗    │     │
    │ 5°C │  ║ Thermal convection ║    │22°C │
    │     │  ║ creates airflow    ║    │     │
    │     │  ║ ───────────────►   ║    │     │
    │     │  ║ (outward in winter)║    │     │
    │     │  ╚════════════════════╝    │     │
    │     │                            │     │
    └─────┴────────────────────────────┴─────┘

    ΔT = 17°C → Natural convection velocity ~0.1-0.5 m/s
    Airflow charges π in consistent direction
    Sound traveling against flow sees π opposition

════════════════════════════════════════════════════════════════════
```

### 3.2 Physics

Thermal convection velocity:
$$v_{conv} \approx \sqrt{\frac{g \beta \Delta T L}{\rho c_p}}$$

For ΔT = 15°C, L = 10cm panel:
$$v_{conv} \approx 0.2 \text{ m/s}$$

This velocity biases the π field:
$$\pi_{bias} = \frac{\alpha_\pi \rho v_{conv}}{\lambda_\pi}$$

### 3.3 Design for Different Climates

| Climate | ΔT (typical) | Bias Strength | Expected R |
|---------|--------------|---------------|------------|
| Cold winter | 20-30°C | Strong | 2.5-4.0 |
| Mild | 10-15°C | Moderate | 1.5-2.5 |
| Hot summer (AC) | 5-15°C | Moderate | 1.5-2.5 |
| Tropical (no AC) | 0-5°C | Weak | 1.0-1.3 |

**Note:** Works best in climates with significant indoor-outdoor temperature difference.

### 3.4 Panel Structure

```
THERMAL-BIASED PANEL LAYERS
════════════════════════════════════════════════════════════════════

    OUTDOOR                                     INDOOR
       │                                           │
       ▼                                           ▼
    ┌──────┬──────────┬────────────┬──────────┬──────┐
    │POROUS│ CHANNEL  │ SAWTOOTH   │ CHANNEL  │POROUS│
    │FACE  │ PLENUM   │ STRUCTURE  │ PLENUM   │FACE  │
    │      │          │            │          │      │
    │  ▓   │    ←     │ ╱╲╱╲╱╲╱╲   │    ←     │  ▓   │
    │  ▓   │  (air)   │ ╲╱╲╱╲╱╲╱   │  (air)   │  ▓   │
    │  ▓   │    ←     │            │    ←     │  ▓   │
    │      │          │            │          │      │
    └──────┴──────────┴────────────┴──────────┴──────┘
       │                                           │
       └───────────── Airflow direction ───────────┘
                    (warm to cold)

    - Porous faces allow slow airflow
    - Channel plenums distribute flow
    - Sawtooth charges π from consistent flow direction
    - Sound from outdoor meets π opposition

════════════════════════════════════════════════════════════════════
```

---

## 4. HVAC-Coupled Panel (Ventilated Barriers)

### 4.1 Concept

Use building ventilation as π bias source.

```
HVAC-INTEGRATED ACOUSTIC PANEL
════════════════════════════════════════════════════════════════════

    SUPPLY AIR                          ROOM
    (from HVAC)
        │
        ▼
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   ┌───────────────────────────────────────┐    │
    │   │         VENTILATION PLENUM             │    │
    │   │  ═══════════════════════════════════►  │    │
    │   └─────────────────┬─────────────────────┘    │
    │                     │                          │
    │                     ▼                          │
    │   ┌─────────────────────────────────────┐      │
    │   │        π-CHARGING STRUCTURE          │      │
    │   │        (sawtooth array)              │      │
    │   │     ╱│╱│╱│╱│╱│╱│╱│╱│╱│╱│╱│╱│        │      │
    │   │    │╱│╱│╱│╱│╱│╱│╱│╱│╱│╱│╱│╱│        │      │
    │   └─────────────────────────────────────┘      │
    │                     │                          │
    │                     ▼                          │
    │   ┌─────────────────────────────────────┐      │
    │   │         DISTRIBUTION GRILLE          │      │
    │   │         (to room)                    │      │
    │   └─────────────────────────────────────┘      │
    │                                                 │
    └─────────────────────────────────────────────────┘

    - HVAC airflow (0.5-2 m/s) provides strong bias
    - Acoustic path through structure sees pre-biased π
    - Fresh air delivered, noise blocked

════════════════════════════════════════════════════════════════════
```

### 4.2 Airflow Requirements

| Application | Airflow Rate | Velocity | Expected R |
|-------------|--------------|----------|------------|
| Office | 10-20 CFM/person | 0.5-1 m/s | 2-3 |
| Hospital | 15-25 CFM/person | 1-2 m/s | 3-5 |
| Lab/Cleanroom | 20-60 CFM/person | 2-5 m/s | 5-10 |

### 4.3 Practical Design

```
HVAC-INTEGRATED WALL SECTION
════════════════════════════════════════════════════════════════════

    CORRIDOR                                 OFFICE
    (noise source)                           (quiet zone)
         │                                        │
         ▼                                        ▼
    ┌─────────┬────────────────────────────┬─────────┐
    │ SOLID   │    VENTILATION CHANNEL     │ SOLID   │
    │ WALL    │                            │ WALL    │
    │         │  ┌──────────────────────┐  │         │
    │    ▓    │  │ SUPPLY AIR ═════════►│  │    ▓    │
    │    ▓    │  │                      │  │    ▓    │
    │    ▓    │  │    SAWTOOTH CORE     │  │    ▓    │
    │    ▓    │  │   ╱╲╱╲╱╲╱╲╱╲╱╲╱╲    │  │    ▓    │
    │    ▓    │  │                      │  │    ▓    │
    │    ▓    │  │ ══════════════════►  │  │    ▓    │
    │    ▓    │  └──────────────────────┘  │    ▓    │
    │         │                            │         │
    └─────────┴────────────────────────────┴─────────┘

    - Air enters from HVAC, exits through grille to office
    - Sound from corridor must travel against airflow
    - π bias from airflow opposes sound transmission

════════════════════════════════════════════════════════════════════
```

---

## 5. Nonlinear/High-Amplitude Panels

### 5.1 Concept

For high-SPL environments, operate in nonlinear regime where saturation creates asymmetry.

```
NONLINEAR ACOUSTIC PANEL (High SPL)
════════════════════════════════════════════════════════════════════

    HIGH-NOISE                              PROTECTED
    ENVIRONMENT                             ZONE
    (>90 dB SPL)
         │                                       │
         ▼                                       ▼
    ┌─────────────────────────────────────────────────┐
    │                                                 │
    │   ┌─────────────────────────────────────────┐  │
    │   │     NONLINEAR ELEMENT ARRAY             │  │
    │   │                                         │  │
    │   │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐        │  │
    │   │  │ ◊ │ │ ◊ │ │ ◊ │ │ ◊ │ │ ◊ │        │  │
    │   │  │   │ │   │ │   │ │   │ │   │        │  │
    │   │  └───┘ └───┘ └───┘ └───┘ └───┘        │  │
    │   │     Buckling beams / snap-through      │  │
    │   │                                         │  │
    │   └─────────────────────────────────────────┘  │
    │                                                 │
    │   High amplitude: Forward compression stiff    │
    │                   Reverse compression soft     │
    │                                                 │
    └─────────────────────────────────────────────────┘

    Nonlinearity creates amplitude-dependent transmission
    Works without external bias, but only at high SPL

════════════════════════════════════════════════════════════════════
```

### 5.2 Nonlinear Elements

| Element Type | Mechanism | Threshold SPL | R at High SPL |
|--------------|-----------|---------------|---------------|
| Buckling beams | Snap-through | ~85 dB | 2-4 |
| Prestressed membranes | Stiffening | ~80 dB | 1.5-3 |
| Granular contacts | Hertzian | ~75 dB | 1.5-2.5 |
| Perforated plates | Jet formation | ~90 dB | 2-5 |

### 5.3 Applications

Best suited for:
- Industrial machinery enclosures
- Airport/highway barriers
- Concert venues
- Generator housings

**Not suitable for:**
- Quiet office environments
- Residential (SPL too low)
- Libraries, hospitals

---

## 6. Training-Pulse Panels (Periodic Refresh)

### 6.1 Concept

Periodically send "training pulses" to maintain π bias.

```
TRAINING-PULSE ACOUSTIC PANEL
════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   ┌──────────┐                         ┌──────────┐        │
    │   │ TRAINING │                         │ SIGNAL   │        │
    │   │ PULSE    │──────► π-STRUCTURE ────►│ PATH     │        │
    │   │ SOURCE   │       (holds bias)      │          │        │
    │   │ (LF)     │                         │ (audio)  │        │
    │   └──────────┘                         └──────────┘        │
    │        │                                                    │
    │        └──── Periodic pulse (every 10-60 sec)              │
    │              Frequency: 20-50 Hz                            │
    │              Duration: 0.5-2 sec                            │
    │              Amplitude: 80-90 dB                            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    Between training pulses:
    - π decays with time constant τ = 1/λ_π ≈ 50-100 sec
    - R decreases from ~3 to ~1.3 over decay period
    - Training pulse restores R

════════════════════════════════════════════════════════════════════
```

### 6.2 Training Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| Pulse frequency | 20-50 Hz | Below speech band |
| Pulse duration | 0.5-2 sec | Enough to charge π |
| Pulse amplitude | 80-90 dB | Strong enough to drive structure |
| Repeat interval | 30-120 sec | Before π decays significantly |
| λ_π of structure | 0.005-0.015 | Slower = longer hold time |

### 6.3 Implementation Options

1. **Embedded speaker:**
   - Small LF driver in panel
   - Controlled by timer circuit
   - Power: ~1W average

2. **External source:**
   - Subwoofer in room
   - Synchronized with other panels
   - Can use existing audio system

3. **Mechanical oscillator:**
   - Motor-driven mass
   - Very low frequency (5-20 Hz)
   - No electronics in panel itself

---

## 7. Comparison of Approaches

### 7.1 Performance Summary

| Approach | Power Required | Expected R | Best Application |
|----------|---------------|------------|------------------|
| Self-biasing (ambient) | 0W | 1.5-3 | Buildings with rumble |
| Thermal gradient | 0W | 1.5-4 | Exterior walls |
| HVAC-coupled | 0W (uses existing) | 2-5 | Ventilated spaces |
| Nonlinear | 0W | 2-4 | High-SPL industrial |
| Training pulse | 0.5-2W | 2-4 | Any, with power |

### 7.2 Decision Matrix

```
PANEL SELECTION GUIDE
════════════════════════════════════════════════════════════════════

                        Ambient    Thermal   HVAC    Non-    Training
                        Rumble     Gradient  Coupled linear   Pulse
────────────────────────────────────────────────────────────────────
Office building           ✓✓         ○        ✓✓       ✗        ✓
Residential              ✓          ✓✓        ○        ✗        ○
Industrial               ✓✓          ○        ✓       ✓✓        ✓
Hospital                 ✓           ○        ✓✓       ✗        ✓✓
Airport/Highway          ✓           ○        ✗       ✓✓        ✓
Data center              ✓✓          ✓        ✓✓       ✗        ○

✓✓ = Excellent fit
✓  = Good fit
○  = May work
✗  = Not suitable

════════════════════════════════════════════════════════════════════
```

---

## 8. Updated Material Specifications

### 8.1 Self-Biasing Panel Bill of Materials

For 1 m² panel:

| Component | Material | Quantity | Est. Cost |
|-----------|----------|----------|-----------|
| Face sheets | Cork board 5mm | 2 m² | $16 |
| Core structure | Bamboo sawtooth (8 teeth) | 1 set | $8 |
| LF resonator | Sand (500g) + rubber mounts | 1 set | $5 |
| Bonding | Natural latex adhesive | 200ml | $4 |
| Frame | Reclaimed wood | 3m linear | $6 |
| **Total** | | | **$39/m²** |

### 8.2 Thermal Gradient Panel Bill of Materials

For 1 m² panel:

| Component | Material | Quantity | Est. Cost |
|-----------|----------|----------|-----------|
| Porous faces | Open-cell cork 10mm | 2 m² | $24 |
| Air channels | Cardboard honeycomb | 0.5 m² | $4 |
| Core structure | Oriented hemp fiber | 0.3 m² | $8 |
| Gaskets | Natural rubber strip | 4m | $6 |
| Frame | Bamboo or wood | 3m linear | $8 |
| **Total** | | | **$50/m²** |

### 8.3 HVAC-Coupled Panel Bill of Materials

For 1 m² panel:

| Component | Material | Quantity | Est. Cost |
|-----------|----------|----------|-----------|
| Plenum box | Sheet metal (recycled) | 0.3 m² | $12 |
| Core structure | 3D-printed sawtooth or bamboo | 1 set | $15 |
| Grille | Perforated metal | 0.2 m² | $8 |
| Gaskets | Silicone or rubber | 2m | $4 |
| Mounting hardware | Steel brackets | 4 pcs | $6 |
| **Total** | | | **$45/m²** |

---

## 9. Experimental Validation Protocol

### 9.1 Self-Biasing Validation

```
SELF-BIASING TEST PROTOCOL
════════════════════════════════════════════════════════════════════

Equipment needed:
- Panel under test
- Shaker or subwoofer (for controlled LF excitation)
- Speaker (signal source, 500 Hz - 4 kHz)
- Microphone (measurement)
- Spectrum analyzer app

Protocol:

1. BASELINE (no bias):
   - Panel in quiet environment (no LF)
   - Measure forward transmission: speaker on noise side, mic on quiet side
   - Measure reverse transmission: swap positions
   - Calculate R_baseline (should be ~1.0)

2. WITH AMBIENT BIAS:
   - Apply 30 Hz tone at 75-85 dB for 30 seconds (simulates rumble)
   - Immediately measure forward and reverse transmission
   - Calculate R_biased (should be > 1.5)

3. DECAY TEST:
   - Stop LF excitation
   - Measure R at t = 0, 30, 60, 120 seconds
   - Plot R vs time to determine decay constant

Pass criteria:
- R_baseline: 0.8 - 1.2 (reciprocal)
- R_biased (at t=0): > 1.5
- Decay time constant: > 30 seconds

════════════════════════════════════════════════════════════════════
```

### 9.2 Thermal Gradient Validation

```
THERMAL GRADIENT TEST PROTOCOL
════════════════════════════════════════════════════════════════════

Equipment needed:
- Panel under test with thermal chambers on each side
- Heat source (heat lamp, hot plate)
- Thermometers (2)
- Speaker and microphone
- Smoke pencil or flow visualization

Protocol:

1. ISOTHERMAL BASELINE:
   - Both sides at same temperature (22°C)
   - Measure forward and reverse transmission
   - Calculate R_isothermal (should be ~1.0)

2. ESTABLISH GRADIENT:
   - Heat one side to 35-40°C
   - Keep other side at 20-22°C
   - Wait 10 minutes for convection to establish
   - Verify airflow with smoke pencil

3. MEASURE WITH GRADIENT:
   - Measure transmission (speaker on hot side → cold side)
   - Measure transmission (speaker on cold side → hot side)
   - Calculate R_gradient

4. REVERSE GRADIENT:
   - Swap hot and cold sides
   - Repeat measurements
   - R should invert (forward becomes reverse)

Pass criteria:
- R_isothermal: 0.8 - 1.2
- R_gradient (ΔT = 15°C): > 1.5
- R inverts when gradient inverts

════════════════════════════════════════════════════════════════════
```

---

## 10. Installation Guidelines

### 10.1 Self-Biasing Panel Installation

```
INSTALLATION FOR MAXIMUM SELF-BIASING
════════════════════════════════════════════════════════════════════

GOOD: Panel receives building vibration

    ┌─────────────────────────────────┐
    │         FLOOR/CEILING           │
    │                                 │
    │    ══════════════════════       │ ← Rigid connection
    │    │     PANEL        │        │    transfers vibration
    │    │                  │        │
    │    │   LF resonator   │        │
    │    │   inside panel   │        │
    │    │                  │        │
    │    ══════════════════════       │
    │                                 │
    └─────────────────────────────────┘


BAD: Panel isolated from building vibration

    ┌─────────────────────────────────┐
    │         FLOOR/CEILING           │
    │                                 │
    │    ~~~~ rubber mounts ~~~~      │ ← Isolation blocks
    │    │     PANEL        │        │    vibration input
    │    │                  │        │
    │    │   LF resonator   │        │
    │    │   (no input!)    │        │
    │    │                  │        │
    │    ~~~~ rubber mounts ~~~~      │
    │                                 │
    └─────────────────────────────────┘

    Vibration isolation defeats self-biasing mechanism!

════════════════════════════════════════════════════════════════════
```

### 10.2 Orientation Matters

```
CORRECT ORIENTATION
════════════════════════════════════════════════════════════════════

    NOISE SIDE              QUIET SIDE
         │                       │
         │    Sawtooth teeth     │
         │    point toward       │
         │    noise source       │
         │                       │
         ▼         ╱╲            ▼
    ═══════════════╱  ╲══════════════
                  ╱    ╲
                 ╱      ╲
                ╱        ╲
    ═══════════╱          ╲══════════
              ╱            ╲
             ╱              ╲
            ╱────────────────╲
           │                  │
           ▼                  ▼
    Forward sound       Reverse sound
    (gradual entry)     (steep entry)
    → HIGH transmission → LOW transmission

    The gradual slope should face the direction you want to
    ALLOW sound from. Steep face blocks reverse direction.

════════════════════════════════════════════════════════════════════
```

---

## 11. Summary: What Actually Works

### 11.1 Physics-Compliant Approaches

| Approach | Mechanism | Physics Validity |
|----------|-----------|------------------|
| Self-biasing | Ambient LF charges π | ✓ Valid (π is real state) |
| Thermal gradient | Convection provides flow bias | ✓ Valid (physical flow) |
| HVAC-coupled | Airflow charges π | ✓ Valid (physical flow) |
| Nonlinear elements | Amplitude-dependent response | ✓ Valid (nonlinearity) |
| Training pulses | Periodic π refresh | ✓ Valid (driven system) |

### 11.2 What Doesn't Work

| Approach | Why It Fails |
|----------|-------------|
| λ_π(→) ≠ λ_π(←) | Material can't "know" direction |
| Pure geometry (linear) | Onsager reciprocity applies |
| Static impedance gradient | Linear, passive → reciprocal |
| "One-way valve" claims | Need bias/nonlinearity |

### 11.3 Key Design Principle

**Every DET acoustic diode needs a reciprocity-breaking mechanism:**

1. **Internal state (π)** that is **charged by external source**
2. **Nonlinear coupling** between π and transmission
3. **Bias source**: ambient, thermal, airflow, or active

Without these, the system is reciprocal regardless of structure.

---

*Document created: January 2026*
*Status: Physics-corrected designs*
*Framework: DET v6.3 with Onsager-compliant interpretation*
