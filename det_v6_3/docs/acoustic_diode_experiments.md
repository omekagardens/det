# Physical Experiments for DET Acoustic Diodes
## Low-Cost, Renewable Materials Approach

**Framework:** DET v6.3 Bond Momentum Theory
**Goal:** Validate acoustic rectification using sustainable, accessible materials

---

## Executive Summary

This document provides detailed experimental protocols for testing DET-predicted acoustic diode behavior using:
- **Low-cost materials** (under $50 total for most experiments)
- **Renewable/sustainable resources** (wood, cork, natural fibers, biopolymers)
- **Simple fabrication** (hand tools, basic workshop equipment)
- **Accessible measurement** (smartphone apps, DIY sensors, affordable equipment)

---

## 1. Core Physics to Validate

### 1.1 The DET Prediction

According to DET, materials with **asymmetric momentum decay (λ_π)** will exhibit:
- **Forward direction:** Sound transmits efficiently (low λ_π → momentum persists)
- **Reverse direction:** Sound is attenuated (high λ_π → momentum scatters)

### 1.2 Physical Mechanisms Creating λ_π Asymmetry

| Mechanism | Physical Basis | Material Examples |
|-----------|---------------|-------------------|
| Geometric ratchet | Sawtooth scatters differently each way | Carved wood, corrugated cardboard |
| Fiber alignment | Grain direction affects wave propagation | Bamboo, wood, natural fiber mats |
| Density gradient | Impedance mismatch varies with direction | Layered cork, graded particle boards |
| Viscoelastic asymmetry | Damping differs by direction | Pre-stressed rubber, oriented polymers |

### 1.3 Key Measurements

1. **Transmission ratio:** T_forward / T_reverse
2. **Insertion loss:** dB attenuation in each direction
3. **Frequency dependence:** Optimal frequency for rectification
4. **Temporal persistence:** How long does asymmetry persist after pulse?

---

## 2. Renewable Material Options

### 2.1 Material Comparison Table

| Material | Cost | Renewability | Workability | Acoustic Properties | Best For |
|----------|------|--------------|-------------|---------------------|----------|
| Bamboo | Very Low | Excellent | Easy (saw, drill) | Strong anisotropy | Fiber-aligned diode |
| Cork | Low | Excellent | Easy | Low impedance, damping | Gradient stacks |
| Wood (softwood) | Very Low | Good | Easy | Moderate anisotropy | Sawtooth structures |
| Cardboard | Very Low | Excellent | Very Easy | Tunable density | Quick prototypes |
| Natural rubber | Low | Good | Moderate | High damping | Viscoelastic designs |
| Hemp/Jute fiber | Very Low | Excellent | Easy | Directional | Fiber composites |
| Agar/Gelatin | Very Low | Excellent | Easy | Tunable, hydrogel | Gradient media |
| Mycelium | Very Low | Excellent | Requires growth | Controllable structure | Grown diodes |
| Clay/Pottery | Very Low | Excellent | Moderate | High impedance | Hard scatterers |
| Sand/Gravel | Very Low | Good | Easy | Granular | Ratchet channels |

### 2.2 Recommended Material Combinations

**Experiment A - Wood Grain Diode:**
- Primary: Bamboo or pine (strong grain)
- Secondary: Cork (low impedance matching)
- Cost: ~$10

**Experiment B - Corrugated Ratchet:**
- Primary: Cardboard (corrugated)
- Secondary: Thin cardboard sheets
- Cost: ~$2

**Experiment C - Fiber Composite:**
- Primary: Hemp fiber mat
- Secondary: Natural latex or starch binder
- Cost: ~$15

**Experiment D - Hydrogel Gradient:**
- Primary: Agar or gelatin
- Secondary: Water, salt (density modifier)
- Cost: ~$8

**Experiment E - Granular Ratchet:**
- Primary: Sand, fine gravel
- Secondary: Cardboard channel
- Cost: ~$5

---

## 3. Experiment A: Bamboo Grain Acoustic Diode

### 3.1 Concept

Bamboo has strong fiber alignment creating natural acoustic anisotropy:
- **With grain:** Sound travels efficiently along fibers
- **Against grain:** Sound scatters at fiber boundaries

By cutting bamboo at an angle, we create direction-dependent transmission.

### 3.2 Materials

| Item | Quantity | Source | Est. Cost |
|------|----------|--------|-----------|
| Bamboo pole (3-5 cm diameter) | 1 meter | Garden center | $3-5 |
| Cork sheet (3mm thick) | 30x30 cm | Craft store | $4 |
| Wood glue | Small bottle | Hardware store | $3 |
| Sandpaper (120, 220 grit) | 2 sheets each | Hardware store | $2 |
| **Total** | | | **~$12-14** |

### 3.3 Tools Required

- Hand saw or hacksaw
- Ruler, protractor
- Clamps (2-4)
- Drill (optional, for mounting holes)

### 3.4 Fabrication Steps

```
BAMBOO DIODE CONSTRUCTION
═══════════════════════════════════════════════════════════════

Step 1: Cut bamboo segments
────────────────────────────────────────────────────────────────

Cut 8-10 segments, each 1-2 cm thick, at a 20° angle to the axis.

                    ╱ ─ ─ ─ ─ ─ ─ ─ ╱
                   ╱               ╱│
                  ╱               ╱ │
                 ╱ ─ ─ ─ ─ ─ ─ ─ ╱  │
    Bamboo      │    SEGMENT    │   │
    fibers  →   │    (angled)   │   │
                │               │  ╱
                 ╲ ─ ─ ─ ─ ─ ─ ─╲╱
                  ╲     20°      ╲
                   ╲              ╲

Step 2: Sand cut faces smooth
────────────────────────────────────────────────────────────────

Use 120 grit, then 220 grit. Goal: flat, smooth surfaces for good
acoustic coupling.


Step 3: Stack segments with consistent orientation
────────────────────────────────────────────────────────────────

All angled cuts should face the same direction:

    FORWARD →

    ╱│╱│╱│╱│╱│╱│╱│╱│╱│╱│
    │╱│╱│╱│╱│╱│╱│╱│╱│╱│╱
    ╱│╱│╱│╱│╱│╱│╱│╱│╱│╱│

    ← REVERSE (blocked)


Step 4: Add cork matching layers
────────────────────────────────────────────────────────────────

Cut cork circles matching bamboo diameter.
Place at input and output for impedance matching.

    [CORK] [BAMBOO STACK] [CORK]


Step 5: Glue and clamp assembly
────────────────────────────────────────────────────────────────

Apply thin layer of wood glue between each segment.
Clamp gently for 24 hours.

═══════════════════════════════════════════════════════════════
```

### 3.5 Test Protocol

**Equipment needed:**
- Speaker or buzzer (phone speaker works)
- Microphone (phone, laptop mic, or dedicated)
- Audio recording app with spectrum analyzer
- Optional: oscilloscope app

**Procedure:**

1. **Setup:**
   - Place diode between speaker and microphone
   - Distance: speaker → diode → microphone (10cm each)
   - Isolate from table vibrations (foam padding underneath)

2. **Forward test:**
   - Orient diode with gradual slope facing speaker
   - Play test tone (1 kHz, 5 kHz, 10 kHz sweeps)
   - Record received amplitude at microphone

3. **Reverse test:**
   - Rotate diode 180°
   - Repeat same tones
   - Record amplitudes

4. **Rectification calculation:**
   ```
   R = A_forward / A_reverse
   R_dB = 20 × log10(R)
   ```

5. **Frequency sweep:**
   - Test 100 Hz to 20 kHz
   - Record R at each frequency
   - Identify optimal frequency range

### 3.6 Expected Results

| Frequency | Predicted R | Notes |
|-----------|-------------|-------|
| 100-500 Hz | 1.0-1.2 | Too low, wavelength >> structure |
| 1-5 kHz | 1.3-2.0 | Moderate effect |
| 5-15 kHz | 1.5-3.0 | Best range for bamboo grain |
| >15 kHz | Decreasing | Attenuation dominates |

**Success criterion:** R > 1.3 at any frequency validates DET prediction.

---

## 4. Experiment B: Corrugated Cardboard Ratchet

### 4.1 Concept

Corrugated cardboard provides pre-made sawtooth structure:

```
CARDBOARD CORRUGATION STRUCTURE
═══════════════════════════════════════════════════════════════

Cross-section of corrugated cardboard:

    ─────────────────────────────────────────  Outer liner
       ╱╲    ╱╲    ╱╲    ╱╲    ╱╲    ╱╲
      ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲       Fluting
     ╱    ╲╱    ╲╱    ╲╱    ╲╱    ╲╱    ╲
    ─────────────────────────────────────────  Inner liner

The triangular channels create direction-dependent acoustic paths.

═══════════════════════════════════════════════════════════════
```

### 4.2 Materials

| Item | Quantity | Source | Est. Cost |
|------|----------|--------|-----------|
| Corrugated cardboard | 1 large box | Recycled | $0 |
| Thin cardboard sheets | 4-6 | Recycled | $0 |
| White glue (PVA) | Small bottle | Dollar store | $1 |
| Rubber bands or tape | Several | Any | $1 |
| **Total** | | | **~$2** |

### 4.3 Fabrication Steps

```
CORRUGATED RATCHET CONSTRUCTION
═══════════════════════════════════════════════════════════════

Step 1: Cut corrugated strips
────────────────────────────────────────────────────────────────

Cut strips 5 cm wide, 15-20 cm long.
Cut PERPENDICULAR to the corrugation direction so channels are
aligned along the strip length.


Step 2: Stack multiple layers
────────────────────────────────────────────────────────────────

Stack 3-5 layers with corrugations aligned in same direction.

    Layer 5:  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲
    Layer 4:  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲
    Layer 3:  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲
    Layer 2:  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲
    Layer 1:  ╱╲╱╲╱╲╱╲╱╲╱╲╱╲

Sound traveling with the "grain" of the corrugation vs against it
will experience different scattering.


Step 3: Seal edges
────────────────────────────────────────────────────────────────

Use tape or glue to seal the top/bottom/sides.
Leave only the channel ends open.


Step 4: Create test housing
────────────────────────────────────────────────────────────────

Place the stack in a cardboard tube or box to isolate it.

    ┌─────────────────────────────────────┐
    │                                     │
    │   [CORRUGATED STACK INSIDE]         │
    │                                     │
    └─────────────────────────────────────┘
         ↑                            ↑
      Input                       Output

═══════════════════════════════════════════════════════════════
```

### 4.4 Test Protocol

Same as Experiment A, but optimize for lower frequencies (corrugation period ~4mm → optimal around 1-5 kHz).

### 4.5 Variations

**Variation B1: Double corrugation**
- Use double-wall cardboard (two corrugated layers)
- Expected: Higher rectification

**Variation B2: Angled corrugation**
- Stack corrugated layers at 15° angle between layers
- Creates helical path, enhancing asymmetry

**Variation B3: Graded flute size**
- Stack different cardboard types (fine, medium, coarse fluting)
- Creates impedance gradient

---

## 5. Experiment C: Hemp Fiber Composite Diode

### 5.1 Concept

Natural fibers (hemp, jute, sisal) have strong directional properties:
- **Parallel to fiber:** Fast sound propagation
- **Perpendicular:** Slower, more scattering

Creating a composite with oriented fibers makes an acoustic diode.

### 5.2 Materials

| Item | Quantity | Source | Est. Cost |
|------|----------|--------|-----------|
| Hemp fiber or jute rope | 50g | Craft store, garden center | $5 |
| Cornstarch or flour | 100g | Grocery store | $1 |
| Water | 500ml | Tap | $0 |
| Cooking pot | 1 | Kitchen | $0 |
| Mold (small box, 10x10x3 cm) | 1 | Cardboard | $0 |
| Plastic wrap | 1 roll | Grocery store | $2 |
| **Total** | | | **~$8** |

### 5.3 Fabrication Steps

```
HEMP FIBER COMPOSITE DIODE
═══════════════════════════════════════════════════════════════

Step 1: Prepare fiber
────────────────────────────────────────────────────────────────

If using rope: Unravel into individual fibers.
If using raw hemp: Comb to align fibers.

Goal: Parallel fiber bundle, ~5cm diameter, 15cm long.


Step 2: Make starch binder
────────────────────────────────────────────────────────────────

Mix 2 tablespoons cornstarch with 1 cup cold water.
Heat while stirring until thick paste forms.
Let cool to room temperature.


Step 3: Impregnate fibers
────────────────────────────────────────────────────────────────

Dip fiber bundle in starch paste.
Work paste into fibers.
Gently squeeze excess (keep fibers aligned!).


Step 4: Orient fibers at angle in mold
────────────────────────────────────────────────────────────────

Line mold with plastic wrap.
Place fiber bundle at 20-30° angle.

    TOP VIEW OF MOLD:

    ┌────────────────────────┐
    │  ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲   │
    │   ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲  │
    │    ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ │  ← Fibers at angle
    │     ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲ ╲│
    └────────────────────────┘

         FORWARD →


Step 5: Dry and cure
────────────────────────────────────────────────────────────────

Dry at room temperature 24-48 hours.
Or dry in oven at 60°C for 4-6 hours.

Final product: Rigid composite with angled fiber orientation.


Step 6: Trim to final shape
────────────────────────────────────────────────────────────────

Cut edges clean with sharp knife.
Sand surfaces smooth if needed.

═══════════════════════════════════════════════════════════════
```

### 5.4 Expected Results

Fiber composites typically show:
- Rectification ratio: 1.5-3.0
- Optimal frequency: 2-10 kHz
- Good durability

---

## 6. Experiment D: Agar Hydrogel Gradient

### 6.1 Concept

Agar gel with varying concentration creates density gradient:
- Low concentration: Low impedance
- High concentration: Higher impedance

Casting a gradient creates asymmetric transmission.

### 6.2 Materials

| Item | Quantity | Source | Est. Cost |
|------|----------|--------|-----------|
| Agar powder | 25g | Grocery store (Asian section) | $4 |
| Salt (NaCl) | 50g | Grocery store | $0.50 |
| Water | 1 liter | Tap | $0 |
| Tall narrow container (test tube or bottle) | 1 | Kitchen/lab | $2 |
| Thermometer | 1 | Kitchen | $0 |
| **Total** | | | **~$6.50** |

### 6.3 Fabrication Steps

```
AGAR GRADIENT HYDROGEL DIODE
═══════════════════════════════════════════════════════════════

Step 1: Prepare agar solutions at different concentrations
────────────────────────────────────────────────────────────────

Solution A (low concentration):
- 1g agar + 0g salt in 100ml water
- Impedance: ~1.5 MRayl

Solution B (medium):
- 2g agar + 5g salt in 100ml water
- Impedance: ~1.8 MRayl

Solution C (high concentration):
- 4g agar + 15g salt in 100ml water
- Impedance: ~2.2 MRayl

Heat each to 85°C until dissolved. Keep warm.


Step 2: Layer casting (bottom-up gradient)
────────────────────────────────────────────────────────────────

Container orientation: INPUT at bottom, OUTPUT at top.

1. Pour Solution A (2cm layer), let cool to gel (~40°C, ~5 min)
2. Pour Solution B on top (2cm layer), let gel
3. Pour Solution C on top (2cm layer), let gel

    ┌─────────────┐
    │ ░░░░░░░░░░░ │ ← Solution C (high Z)
    │ ░░░░░░░░░░░ │
    ├─────────────┤
    │ ▒▒▒▒▒▒▒▒▒▒▒ │ ← Solution B (medium Z)
    │ ▒▒▒▒▒▒▒▒▒▒▒ │
    ├─────────────┤
    │ ▓▓▓▓▓▓▓▓▓▓▓ │ ← Solution A (low Z)
    │ ▓▓▓▓▓▓▓▓▓▓▓ │
    └─────────────┘


Step 3: Full cure
────────────────────────────────────────────────────────────────

Refrigerate 2-4 hours for complete gelation.


Step 4: Test in container (water coupling recommended)
────────────────────────────────────────────────────────────────

Keep gel in container for testing.
Use water between transducer and gel for acoustic coupling.

═══════════════════════════════════════════════════════════════
```

### 6.4 Physics Notes

**Forward (Low Z → High Z):**
- Gradual impedance increase
- Small reflections at each interface
- Most energy transmitted

**Reverse (High Z → Low Z):**
- Each interface has strong reflection
- Multiple reflections randomize momentum
- Energy scattered back

### 6.5 Variations

**D1: Continuous gradient**
- Mix solutions while pouring
- Creates smooth gradient
- Better impedance matching

**D2: Gelatin alternative**
- Use food-grade gelatin instead of agar
- Similar properties, different texture
- Add glycerin for flexibility

---

## 7. Experiment E: Granular Ratchet Channel

### 7.1 Concept

A channel filled with asymmetric granular structures creates a mechanical ratchet for acoustic waves.

### 7.2 Materials

| Item | Quantity | Source | Est. Cost |
|------|----------|--------|-----------|
| Fine sand | 500g | Hardware store, beach | $2 |
| Small gravel (2-5mm) | 500g | Hardware store, aquarium | $3 |
| Cardboard tubes | 2-3 | Paper towel rolls | $0 |
| Thin cardboard for baffles | 1 sheet | Recycled | $0 |
| Hot glue or tape | As needed | Craft store | $2 |
| **Total** | | | **~$7** |

### 7.3 Fabrication Steps

```
GRANULAR RATCHET CHANNEL
═══════════════════════════════════════════════════════════════

Step 1: Create angled baffles
────────────────────────────────────────────────────────────────

Cut cardboard circles that fit inside the tube.
Cut at angle to create wedge shape.

    Side view of baffles:

    ──╲──────────────╲──────────────╲──────────────╲──
       ╲              ╲              ╲              ╲
        ╲              ╲              ╲              ╲
         ╲              ╲              ╲              ╲
    ──────╲──────────────╲──────────────╲──────────────╲

    All angled the same way (e.g., 30° from perpendicular)


Step 2: Install baffles in tube
────────────────────────────────────────────────────────────────

Glue baffles every 2-3 cm along tube.
Ensure all angles point same direction.


Step 3: Fill chambers with granular material
────────────────────────────────────────────────────────────────

Option A: Uniform fill
- Fill all chambers with same sand/gravel mix

Option B: Graded fill
- First chambers: Fine sand (low impedance)
- Middle chambers: Mixed sand/gravel
- Last chambers: Coarse gravel (high impedance)

    ┌─╲────────╲────────╲────────╲────────╲─┐
    │  ╲ fine  ╲  mix   ╲  mix   ╲coarse ╲  │
    │   ╲ sand ╲        ╲        ╲gravel ╲  │
    │    ╲     ╲        ╲        ╲       ╲  │
    └─────╲──────╲────────╲────────╲───────╲┘


Step 4: Seal ends with mesh
────────────────────────────────────────────────────────────────

Use fabric or mesh to retain particles but allow sound.

═══════════════════════════════════════════════════════════════
```

### 7.4 Physics

- **Forward:** Sound pushes particles against gradual slope (efficient)
- **Reverse:** Sound hits steep baffle faces (scattering)
- Granular contacts provide additional nonlinearity

---

## 8. Measurement Methods

### 8.1 Budget Tier: Smartphone-Based (~$0)

**Apps needed:**
- Frequency generator app (e.g., "Tone Generator")
- Spectrum analyzer app (e.g., "Spectroid", "phyphox")
- Sound level meter app

**Procedure:**
1. Generate test tone with one phone
2. Record with second phone (or same phone's mic if far enough)
3. Measure peak amplitude at test frequency
4. Compare forward vs reverse

**Limitations:**
- Frequency range: ~100 Hz - 15 kHz
- Amplitude accuracy: ±3-5 dB
- Phase information: Not available

### 8.2 Intermediate Tier: USB Audio Interface (~$30-50)

**Equipment:**
- USB audio interface (e.g., Behringer UCA202, ~$30)
- Small speaker or piezo buzzer (~$5)
- Electret microphone (~$2)
- Laptop with Audacity (free software)

**Benefits:**
- Better frequency response
- Quantitative measurements
- Save data for analysis

### 8.3 Advanced Tier: Dedicated Equipment (~$100-300)

**Equipment:**
- Function generator (used: ~$50-100)
- Oscilloscope (used: ~$50-100, or USB scope ~$100)
- Piezoelectric transducers (~$10-20)
- Small power amplifier (~$20)

**Benefits:**
- Precise frequency control
- Time-domain measurements
- Higher frequencies (ultrasonic possible)

### 8.4 Measurement Protocol Template

```
ACOUSTIC DIODE MEASUREMENT PROTOCOL
═══════════════════════════════════════════════════════════════

DATE: _______________
SAMPLE: _______________
OPERATOR: _______________

SETUP:
- Speaker/transducer type: _______________
- Microphone/receiver type: _______________
- Distance speaker-sample: _______ cm
- Distance sample-mic: _______ cm
- Temperature: _______ °C
- Humidity: _______ %

MEASUREMENTS:

Frequency (Hz) | Forward (dB) | Reverse (dB) | R = Fwd/Rev | R (dB)
───────────────┼──────────────┼──────────────┼─────────────┼────────
     100       │              │              │             │
     200       │              │              │             │
     500       │              │              │             │
    1000       │              │              │             │
    2000       │              │              │             │
    5000       │              │              │             │
   10000       │              │              │             │
   15000       │              │              │             │
   20000       │              │              │             │

NOTES:

_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

CONCLUSION:
- Maximum rectification: R = _______ at f = _______ Hz
- Optimal frequency band: _______ to _______ Hz
- DET prediction validated? YES / NO / PARTIAL

═══════════════════════════════════════════════════════════════
```

---

## 9. Data Analysis

### 9.1 Calculating Rectification Ratio

```python
# Python analysis script (requires numpy, matplotlib)
import numpy as np
import matplotlib.pyplot as plt

# Input your measurements
frequencies = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]  # Hz
forward_dB = [...]  # Fill in your measurements
reverse_dB = [...]  # Fill in your measurements

# Calculate rectification
R_dB = np.array(forward_dB) - np.array(reverse_dB)
R_linear = 10 ** (R_dB / 20)

# Find optimal frequency
idx_max = np.argmax(R_dB)
f_optimal = frequencies[idx_max]
R_max = R_linear[idx_max]

print(f"Maximum rectification: R = {R_max:.2f} at f = {f_optimal} Hz")
print(f"Rectification in dB: {R_dB[idx_max]:.1f} dB")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogx(frequencies, R_dB, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Rectification (dB)', fontsize=12)
plt.title('Acoustic Diode: Rectification vs Frequency', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('rectification_curve.png', dpi=150)
plt.show()
```

### 9.2 Comparing to DET Predictions

From DET theory, optimal frequency range:
$$f_{optimal} \approx \sqrt{\lambda_\pi \times (1/\alpha_\pi)} \times f_{scale}$$

For bamboo with estimated parameters:
- λ_π ≈ 0.01 (relatively low decay along grain)
- α_π ≈ 0.1 (moderate charging)
- f_scale ≈ 10 kHz (structural resonance)

Predicted optimal: ~3-5 kHz

Compare your measured optimal frequency to this prediction.

---

## 10. Troubleshooting

### 10.1 Common Issues

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| R ≈ 1 at all frequencies | Poor acoustic coupling | Add gel/oil at interfaces |
| R < 1 (reverse better than forward) | Sample installed backwards | Rotate 180° |
| High attenuation both directions | Material too lossy | Try stiffer material |
| Resonance peaks | Standing waves in sample | Change sample length |
| Inconsistent results | Vibration pickup | Better isolation, move outdoors |
| No signal at high frequency | Microphone rolloff | Use piezo sensor |

### 10.2 Improving Rectification

1. **Stack multiple diode sections** in series
   - Each section adds ~1.5-2x to total R
   - N sections: R_total ≈ R_single^N

2. **Add impedance matching layers**
   - Cork or rubber at input/output
   - Reduces reflection losses

3. **Increase geometric asymmetry**
   - Steeper angles in sawtooth structures
   - More pronounced fiber alignment

4. **Test at resonant frequencies**
   - Find sample's natural resonance
   - Rectification often peaks there

---

## 11. Safety Notes

### 11.1 General Safety

- **Ear protection:** When testing at high volumes (>85 dB)
- **Eye protection:** When cutting/sanding materials
- **Ventilation:** When using glues or heating materials
- **Hot materials:** Agar/starch solutions can cause burns

### 11.2 Material-Specific

- **Bamboo:** Splinters possible; wear gloves when handling fresh cuts
- **Agar:** Food-safe; dispose in compost
- **Cork:** Generally safe; low dust
- **Cardboard:** Minimal hazards; recycle when done
- **Natural fibers:** May cause irritation; work in ventilated area

---

## 12. Experiment Progression

### 12.1 Recommended Order

1. **Start with Experiment B** (Corrugated cardboard)
   - Quickest to build (~30 minutes)
   - No cost (recycled materials)
   - Good for learning measurement technique

2. **Then Experiment A** (Bamboo)
   - More robust diode
   - Better rectification expected
   - Reusable for multiple tests

3. **Then Experiment E** (Granular ratchet)
   - Different physics (particle mechanics)
   - Good for comparing mechanisms

4. **Advanced: Experiment D** (Hydrogel gradient)
   - Requires more careful preparation
   - Demonstrates impedance gradient concept

5. **Advanced: Experiment C** (Fiber composite)
   - Most involved fabrication
   - Best durability and performance

### 12.2 Validation Checklist

For each experiment, verify:

- [ ] Forward transmission > reverse transmission (R > 1)
- [ ] Effect is reproducible (test 3+ times)
- [ ] Effect persists when sample is moved/remounted
- [ ] Effect scales with sample thickness (more sections → higher R)
- [ ] Effect has frequency dependence (optimal band exists)
- [ ] Control test: symmetric sample shows R ≈ 1

---

## 13. Extensions and Advanced Experiments

### 13.1 Multi-Stage Cascade

Build 2-3 diode sections in series:

```
CASCADED ACOUSTIC DIODE
═══════════════════════════════════════════════════════════════

    ┌──────────┐    ┌──────────┐    ┌──────────┐
───►│ DIODE 1  │───►│ DIODE 2  │───►│ DIODE 3  │───►
    │  R = 2   │    │  R = 2   │    │  R = 2   │
    └──────────┘    └──────────┘    └──────────┘

    Total rectification: R_total = 2 × 2 × 2 = 8

═══════════════════════════════════════════════════════════════
```

### 13.2 Active Enhancement (π-Feedback Analog)

Add controlled pre-stress to the sample:
- Compression increases effective F (in DET terms)
- Higher F → Higher gain in forward direction

**Method:**
1. Mount sample in vise or clamp
2. Apply controlled pressure
3. Measure R at different stress levels
4. Plot R vs. applied pressure

DET predicts: R should increase with stress (until instability threshold)

### 13.3 Temperature Dependence

DET predicts temperature affects λ_π (higher T → higher decay rate):

**Procedure:**
1. Measure R at room temperature (~20°C)
2. Warm sample (hair dryer) to ~40°C, measure R
3. Cool sample (refrigerator) to ~5°C, measure R

**Expected:** Higher temperature → lower R (momentum decays faster)

### 13.4 Mycelium-Grown Diode (Advanced/Long-term)

Grow a diode structure using mushroom mycelium:

1. Create mold with asymmetric internal structure
2. Inoculate with oyster mushroom spawn
3. Grow 2-4 weeks
4. Heat-kill and dry
5. Test acoustic properties

**Advantage:** Fully renewable, carbon-negative material

---

## 14. Reporting Results

### 14.1 Documentation Template

```
ACOUSTIC DIODE EXPERIMENT REPORT
═══════════════════════════════════════════════════════════════

EXPERIMENT: [A/B/C/D/E]
DATE: _______________
MATERIALS USED:
_______________________________________________________________

FABRICATION NOTES:
_______________________________________________________________
_______________________________________________________________

MEASUREMENTS:
(Attach data table and frequency plot)

KEY RESULTS:
- Maximum R: _______ at f = _______ Hz
- Optimal frequency band: _______ to _______ Hz
- Total insertion loss (forward): _______ dB

COMPARISON TO DET PREDICTION:
- Predicted R: _______
- Predicted optimal f: _______
- Agreement: GOOD / MODERATE / POOR

OBSERVATIONS:
_______________________________________________________________
_______________________________________________________________

PHOTOS:
(Attach photos of sample, setup, and any interesting features)

═══════════════════════════════════════════════════════════════
```

### 14.2 Sharing Results

Consider sharing successful experiments:
- GitHub repository issues/discussions
- Maker community forums
- Science education platforms
- Social media with #DETacoustics #AcousticDiode

---

## 15. Summary: Quick Reference

### 15.1 Minimum Viable Experiment

**Cardboard Ratchet (Experiment B):**
- Materials: Corrugated cardboard, tape ($0-2)
- Time: 30 minutes
- Equipment: 2 smartphones
- Expected R: 1.2-1.5

### 15.2 Best Performance Experiment

**Bamboo Grain Diode (Experiment A):**
- Materials: Bamboo, cork, glue ($12-15)
- Time: 2-3 hours (plus 24h cure)
- Equipment: USB audio interface recommended
- Expected R: 1.5-3.0

### 15.3 Key Success Factors

1. **Good acoustic coupling** between components
2. **Consistent orientation** of asymmetric elements
3. **Multiple sections** for cumulative effect
4. **Frequency optimization** for your specific design
5. **Careful measurement** with multiple trials

---

## Appendix: Supplier Suggestions

### Natural Materials
- **Bamboo:** Local garden centers, craft stores, or cut from garden
- **Cork:** Craft stores, wine shops (for corks), Amazon
- **Hemp fiber:** Craft stores, online sustainable material suppliers
- **Jute:** Garden centers (as twine), craft stores

### Binding Agents
- **Cornstarch:** Any grocery store
- **Agar:** Asian grocery stores, Amazon
- **Wood glue:** Hardware stores, craft stores
- **Natural latex:** Art supply stores, online

### Tools
- **Saws:** Harbor Freight (budget), local hardware
- **Sandpaper:** Any hardware store
- **Clamps:** Harbor Freight, hardware stores

### Measurement
- **USB audio interfaces:** Behringer UCA202 (~$30), Focusrite Scarlett (~$100)
- **Piezo transducers:** Amazon, electronics suppliers (~$5-10 for pack)
- **Used oscilloscopes:** eBay, local ham radio clubs

---

*Document created: January 2026*
*Framework: DET v6.3 Momentum Dynamics*
*Application: Low-Cost Acoustic Diode Experimentation*
