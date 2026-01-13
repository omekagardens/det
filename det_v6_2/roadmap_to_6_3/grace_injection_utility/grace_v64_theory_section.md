# Grace Injection v6.4: Antisymmetric Edge Flux Formulation

## Summary

Grace v6.4 implements grace injection as an **antisymmetric edge flux**, ensuring:
- **Strict locality**: No hidden globals, no balancing steps
- **Automatic conservation**: Antisymmetry guarantees $\sum G_{i\to j} = 0$
- **No double-counting**: Each edge transfers once, no overlapping pools
- **Bond-local quantum gate**: Suppresses grace on high-coherence bonds

**All six validation tests pass:**

| Test | Description | Result |
|:-----|:------------|:-------|
| Conservation | E4: Coordinated Zero | ✓ (0.022%) |
| Quantum Gate | E1: High-C trap | ✓ (0.0 grace) |
| Necessity | E3: Conductivity trap | ✓ (Δ=+0.17) |
| Agency (F2) | Zero-a receives nothing | ✓ |
| Overlap Stress | T1: Checkerboard | ✓ (no double-tax) |
| Mixed Coherence | T2: Channel test | ✓ (island undisturbed) |

---

## Theory Card Section (VI.5)

### VI.5 Grace Injection (Antisymmetric Edge Flux, v6.4)

**Purpose:** Enable recovery from depletion while preserving strict locality and automatic conservation.

**Threshold (relative, R-local):**
$$F_{\text{thresh},i} = \beta_g \cdot \langle F \rangle_{\mathcal{N}_R(i)}$$

**Need and Excess:**
$$n_i = [F_{\text{thresh},i} - F_i]_+, \qquad e_i = [F_i - F_{\text{thresh},i}]_+$$

**Donor capacity and recipient need (agency-gated):**
$$d_i = a_i \cdot e_i, \qquad r_i = a_i \cdot n_i$$

**Agency gate (grace flows through agency-connected paths):**
$$g^{(a)}_{ij} = \sqrt{a_i \cdot a_j}$$

**Bond-local quantum gate:**
$$Q_{ij} = \left[1 - \frac{\sqrt{C_{ij}}}{C_{\text{quantum}}}\right]_+$$

**Grace flux (antisymmetric by construction):**
$$\boxed{
G_{i\to j} = \eta_g \cdot g^{(a)}_{ij} \cdot Q_{ij} \cdot \left(
  d_i \cdot \frac{r_j}{\sum_{k\in\mathcal{N}_R(i)} r_k + \varepsilon}
  - d_j \cdot \frac{r_i}{\sum_{k\in\mathcal{N}_R(j)} r_k + \varepsilon}
\right)
}$$

**Resource update (combines diffusive and grace flux):**
$$F_i^+ = F_i - \sum_{j\in\mathcal{N}(i)} \left(J_{i\to j} + G_{i\to j}\right) \Delta\tau_i$$

**Parameters:**

| Parameter | Default | Description |
|:----------|:--------|:------------|
| $\eta_g$ | 0.5 | Grace flux coefficient |
| $\beta_g$ | 0.4 | Relative need threshold |
| $C_{\text{quantum}}$ | 0.85 | Quantum gate threshold |
| $R$ | 2 | Neighborhood radius |

**Numerical stability floor:** $F_{\text{floor}} = 10^{-6}$ (tooling constant, not physical threshold)

---

## Key Design Decisions

### 1. Antisymmetric Edge Flux (not node-level injection)

**Previous approach (v6.3):** Pool dissipation over neighborhood, distribute to needy nodes.
- Problem: Required global balancing step to ensure conservation
- Problem: Overlapping neighborhoods caused double-counting

**Current approach (v6.4):** Grace is a bond flux $G_{i\to j} = -G_{j\to i}$.
- Conservation automatic by antisymmetry
- No overlapping pools, no double-counting
- Each bond transfers independently

### 2. Agency Gate (not physical conductivity)

**Previous approach:** Grace flux $\propto \sigma_{ij}$ (physical conductivity).
- Problem: Grace throttled by same barriers it's meant to bypass

**Current approach:** Grace flux $\propto g^{(a)}_{ij} = \sqrt{a_i a_j}$.
- Grace flows through agency-connected paths
- Bypasses low-$\sigma$ barriers when agency is present
- Consistent with boundary operator semantics (grace respects agency, not physics)

### 3. Bond-Local Quantum Gate

**Previous approach:** Node-level gate $Q_i$ based on average coherence.
- Problem: Mixed-coherence neighborhoods could leak grace into high-C regions

**Current approach:** Bond-local gate $Q_{ij} = [1 - \sqrt{C_{ij}}/C_{\text{quantum}}]_+$.
- Each bond individually gated
- High-C bonds block grace even if node average is moderate
- Prevents interference with quantum recovery mechanisms

### 4. R-Neighborhood for Need Sums

The normalization $\sum_{k\in\mathcal{N}_R(i)} r_k$ uses $R=2$ (Manhattan distance) to:
- Reach beyond single-layer barriers
- Maintain strict locality (explicit bounded neighborhood)
- Match the scale of grace redistribution to physical scenarios

---

## Validation Details

### Test 1: Conservation (E4)

**Setup:** 60% of grid depleted (F=0.01), 40% resource-rich (F=2.0)

**Results:**
- Grace ON: F_total = 838.08 → 838.27 (conservation: 1.00022)
- Grace OFF: F_total = 838.08 → 838.27 (conservation: 1.00022)

**Analysis:** Conservation within 0.022% — purely numerical (finite timestep).
The antisymmetric flux structure guarantees exact conservation in the continuous limit.

### Test 2: Bond-Local Quantum Gate (E1)

**Setup:** Central depleted region, very high coherence (C=0.98), spiral phases

**Results:**
- Total grace injected: 0.000000
- Recovery identical with/without grace (F=1.246)

**Analysis:** The bond-local gate $Q_{ij} = [1 - \sqrt{0.98}/0.85]_+ = 0$ completely suppresses grace.
Quantum recovery proceeds via phase-mediated flow without interference.

### Test 3: Necessity (E3)

**Setup:** Central depleted region, low-$\sigma$ barrier (0.01), normal outer region

**Results:**
- Grace ON: F_depleted = 0.371
- Grace OFF: F_depleted = 0.201
- Improvement: +85%

**Analysis:** Diffusion is throttled by low-$\sigma$ barrier, but grace flows through agency-connected paths, enabling recovery. This demonstrates necessity: grace enables outcomes diffusion cannot.

### Test 4: Agency Inviolability (F2)

**Setup:** Central region with $a_i = 0$, depleted (F=0.05)

**Results:**
- Grace flux into zero-agency region: 0.00

**Analysis:** The agency gate $g^{(a)}_{ij} = \sqrt{0 \cdot a_j} = 0$ blocks all grace to zero-agency nodes.
Falsifier F2 (Coercion) is satisfied by construction.

### Test 5: Overlap Stress (T1)

**Setup:** Checkerboard pattern — alternating donors (F=2.0) and recipients (F=0.2)

**Results:**
- Min donor F after simulation: 0.953 (never overtaxed)
- Oscillation rate: 1% (stable)
- Conservation: 1.00000000 (perfect)

**Analysis:** Edge-flux formulation prevents double-counting. Each donor gives through its edges only, not through overlapping pools.

### Test 6: Mixed Coherence Channel (T2)

**Setup:** Low-C corridor → high-C island → resource-rich region

**Results:**
- Corridor improvement: +0.0001 (grace helps slightly)
- Island disturbance: 0.0000 (grace blocked by high-C boundary)

**Analysis:** Grace flows through low-C corridor bonds but is blocked at the high-C island boundary by the bond-local quantum gate.

---

## Updated Falsifiers

| ID | Description | Test |
|:---|:------------|:-----|
| F_G1 | Grace Creates Mass | $\sum F_i$ increases by >0.1% per step |
| F_G2 | Non-Local Grace | Grace depends on state outside $\mathcal{N}_R(i)$ |
| F_G3 | Coerced Grace | Node with $a_i=0$ receives $G_{j\to i}>0$ |
| F_G4 | Grace Redundancy | No scenario shows necessity (checked via E3) |
| F_G5 | Quantum Harm | Grace reduces recovery in high-C regime (checked via E1) |
| F_G6 | Double Counting | Donor taxed more than $d_i$ in one step (checked via T1) |
| F_G7 | High-C Leakage | Grace flows across high-C boundary (checked via T2) |

---

## Comparison with Previous Versions

| Property | v6.2 | v6.3 | v6.4 |
|:---------|:-----|:-----|:-----|
| Conservation | ✗ (16% violation) | ✓ (with global balance) | ✓ (by construction) |
| Locality | ✗ (F_min global) | ✓ (relative threshold) | ✓ (edge flux) |
| No double-counting | ? | ✗ (overlapping pools) | ✓ (edge-local) |
| Quantum gate | ✗ | ✓ (node-level) | ✓ (bond-local) |
| Necessity | ✓ | ✓ | ✓ |
| Agency (F2) | ✓ | ✓ | ✓ |

---

## Open Questions

1. **Gravity interaction:** Should grace be suppressed in deep potential wells?
   - Proposed: Add gravity gate $G^{(\Phi)}_i = [1 - \gamma_\Phi |\nabla\Phi|]_+$
   - Status: Not yet implemented/tested

2. **Optimal parameters:** Current defaults work for test scenarios; may need tuning for specific applications.

3. **3D validation:** All tests run on 2D grids; 3D behavior should be verified.

4. **Long-time stability:** 2000-step simulations pass; longer runs may reveal edge cases.

---

## Files

| File | Description |
|:-----|:------------|
| `grace_v64_edge_flux.py` | Complete implementation with validation suite |
| `grace_v64_results.png` | Validation plots |
| `grace_v64_theory_section.md` | This document |
