# DET Canonical Collider Results

## Executive Summary

**DET dipoles undergo MERGER when overlapping, producing structural debt (mass).**

This is consistent with canonical DET 4.2 and represents genuine collision physics.

---

## Experimental Setup

- **Flow equation:** `J = σ[√C Im(ψ*ψ') + (1-√C)(F_i - F_j)]` (canonical, no gravity)
- **q-locking:** `q += α_q × max(0, -ΔF)` (canonical B.3 Step 6)
- **Extensions:** Flow-driven coherence, phase diffusion (declared)
- **Ablations:** Agency fixed at a=1

---

## Key Results

### 1. DET "Solitons" are Dipoles

Single peaks are **unstable**. The natural stable structure is a **2-peak dipole**.
- Separation emerges from quantum-classical flow balance
- ~40 grid units equilibrium separation (parameter-dependent)

### 2. Phase Gradients Don't Create Momentum

DET has no inertia. Phase affects flow direction but doesn't create persistent motion.
To study collisions, dipoles must start in interaction range (overlapping).

### 3. Collision Outcomes Depend on Overlap

| Overlap | Initial | Final | Outcome |
|---------|---------|-------|---------|
| 0 | 4 | 5 | Fragmentation |
| 10 | 3 | **2** | **MERGER** |
| 15 | 3 | **2** | **MERGER** |
| 20 | 1 | 2 | Fragmentation |
| 25 | 2 | 2 | Stable |

### 4. Merger Creates Mass (q)

During merger (overlap=10):
- q_max increases from 0 → 0.16
- q localizes at collision site
- This is **DET mass generation from collision**

Timeline:
```
t=0:    3 peaks, q=0.000  (initial)
t=50:   1 peak,  q=0.027  (momentary fusion)
t=750:  2 peaks, q=0.157  (stable dipole + mass)
```

---

## Physics Interpretation

### What Happens During Merger

1. **Overlapping dipoles create flow conflict**
   - Quantum currents oppose each other
   - Classical gradients are steep

2. **Flow resolves by redistributing F**
   - Peaks merge temporarily
   - q-locking captures the "lost" F as structure

3. **New equilibrium emerges**
   - Single dipole (2 peaks)
   - Structural debt (q > 0) remains at collision site

### DET Analogy to Particle Physics

| DET Concept | Physics Analogue |
|-------------|------------------|
| Dipole | Particle (bound state) |
| F (resource) | Energy |
| q (structural debt) | Rest mass |
| Merger | Inelastic collision |
| q accumulation | Mass generation |

---

## Consistency with DET 4.2

✓ **Canonical flow equation** (no gravity term)  
✓ **Canonical q-locking** (Appendix B.3)  
✓ **Local dynamics only**  
✓ **No hidden globals**  

Extensions (declared):
- Flow-driven coherence dynamics (v5 preview)
- Phase diffusion

---

## Falsifiability

These results would be **falsified** if:

1. Mergers occur without q accumulation
2. q accumulates in isolated (non-colliding) dipoles
3. Final states depend on global information
4. Results change when domain size increases (locality violation)

---

## Next Steps

1. **Vary q-locking strength** to see q accumulation scaling
2. **Test multi-body collisions** (3+ dipoles)
3. **Compute emergent gravity Φ** around merged dipoles
4. **Check if high-q regions behave as black hole seeds** (a→0, P→0)
5. **2D/3D extensions** for realistic particle physics
