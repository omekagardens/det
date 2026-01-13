# Grace Necessity Test Protocol

## Objective

Determine whether grace injection is a **necessary** boundary operator (enables otherwise-impossible recovery) or **redundant** (merely accelerates diffusion).

**Falsifier target:** F3 (Boundary Redundancy) and proposed F_G5

---

## 1. Theoretical Analysis: When Grace Should Matter

### 1.1 Diffusion Failure Modes

Diffusion can fail to rescue a depleted node when:

1. **Phase-locked flow (quantum regime)**
   - High coherence $C_{ij} \to 1$
   - Flow dominated by $\text{Im}(\psi_i^* \psi_j)$
   - Phase gradient points away from depleted node
   - Resource flows "wrong direction" despite pressure gradient

2. **Local isolation**
   - Low conductivity $\sigma_{ij}$ to neighbors
   - Diffusion rate $\propto \sigma_{ij}$ is too slow
   - Node starves before diffusion catches up

3. **Coordinated depletion**
   - Multiple adjacent nodes depleted simultaneously
   - No local gradient to drive flow
   - Everyone is poor, no one can give

4. **Agency asymmetry**
   - Depleted node has moderate $a_i$
   - Neighbors have low $a_j$
   - Bond gate $g^{(a)}_{ij} = \sqrt{a_i a_j}$ is low
   - But grace weight $w_i = a_i \cdot n_i$ is moderate

### 1.2 Grace Success Conditions

Grace succeeds where diffusion fails when:

- **Neighborhood activity exists** ($D_{\text{pool}} > 0$)
- **Depleted node is open** ($a_i > 0$)
- **Need-targeting matters** (phase or gradient doesn't favor node)

**Critical test regime:** High coherence + wrong phase alignment + neighborhood activity

---

## 2. Test Protocol

### 2.1 Experimental Setup

**Grid:** 2D lattice, $N \times N$ (suggest $N = 32$ for tractability)

**Two identical systems:**
- System A: Grace enabled ($\eta_g > 0$)
- System B: Grace disabled ($\eta_g = 0$)

**Initialization (near-freeze but not dead):**
```
Global:
  - Mean agency: ⟨a⟩ = 0.3 (low but nonzero)
  - Mean resource: ⟨F⟩ = 1.0
  - Mean coherence: ⟨C⟩ = 0.8 (quantum-dominated regime)

Depleted region (central 5×5 patch):
  - F_i = 0.1 (near-zero resource)
  - a_i = 0.4 (moderate agency — can receive)
  - θ_i = uniform random (no phase alignment with neighbors)

Surrounding region:
  - F_i = 1.5 (resource-rich)
  - a_i = 0.25 (slightly lower agency)
  - θ_i = coherent wave pattern (phase gradient points outward)
```

**Key design choice:** Phase gradient points *away* from depleted region. In the quantum regime, this makes diffusion flow outward, exacerbating depletion.

### 2.2 Measurements

Run both systems for $T = 1000$ steps. Record:

| Metric | Definition |
|:---|:---|
| Recovery time $t_{\text{rec}}$ | First step where depleted region $\langle F \rangle > 0.5$ |
| Recovery fraction $f_{\text{rec}}$ | Fraction of depleted nodes with $F_i > F_{\min}$ at $t = T$ |
| Final coherence $\langle C \rangle_{\text{final}}$ | Mean coherence at $t = T$ |
| Total resource $\sum F_i(T)$ | Conservation check |
| Outcome class | Recovered / Frozen / Collapsed |

**Outcome classification:**
- **Recovered:** $f_{\text{rec}} > 0.9$ and $\langle C \rangle > 0.5$
- **Frozen:** $f_{\text{rec}} < 0.1$ and system stable (no drift)
- **Collapsed:** Total resource $\sum F_i$ drops by > 20%

### 2.3 Success Criteria

| Result | Interpretation |
|:---|:---|
| A: Recovered, B: Recovered | Grace redundant (both recover) |
| A: Recovered, B: Frozen | **Grace necessary** (enables recovery) |
| A: Recovered, B: Collapsed | Grace prevents collapse |
| A: Frozen, B: Frozen | Grace insufficient (need stronger intervention) |
| A: Frozen, B: Recovered | Grace harmful?! (investigate) |

**Primary success:** A recovers, B does not. This demonstrates grace is not boundary-redundant.

---

## 3. Scenario Variants

### 3.1 Scenario α: Phase-Locked Depletion (Primary)

As described above. Tests grace vs. quantum-regime diffusion.

### 3.2 Scenario β: Isolated Node

**Setup:**
- Single depleted node at center
- All bonds to neighbors have $\sigma_{ij} = 0.1$ (low conductivity)
- Neighbors are resource-rich with normal conductivity among themselves

**Prediction:** Diffusion too slow; grace (if pooled from neighborhood) can rescue.

### 3.3 Scenario γ: Coordinated Depletion

**Setup:**
- Large depleted region (half the grid)
- No internal gradients
- Boundary with resource-rich region

**Prediction:** Diffusion works at boundary; grace may accelerate interior recovery.

### 3.4 Scenario δ: Agency Desert

**Setup:**
- Depleted region has $a_i = 0.5$
- Surrounding region has $a_j = 0.05$ (very low agency)
- Bond gates $g^{(a)}_{ij} \approx 0.16$ (suppressed)

**Prediction:** Diffusion suppressed by low neighbor agency. Grace weight $w_i = a_i \cdot n_i$ still moderate.

**This tests whether grace can route around low-agency neighbors via the dissipation pool.**

---

## 4. Controls and Sanity Checks

### 4.1 Positive Control (Grace Should Work)

- Initialize with depleted region, favorable conditions
- Grace enabled
- Verify recovery occurs

### 4.2 Negative Control (Grace Should Fail)

- Initialize with $a_i = 0$ in depleted region
- Grace enabled
- Verify $I_{g \to i} = 0$ (no coercion)

### 4.3 Conservation Check

- Track $\sum F_i$ over time
- With proposed revision: $\Delta \sum F_i \leq \eta_g \sum D_i$ per step
- Flag any violation

### 4.4 Locality Check

- Add disconnected subgraph (separate component)
- Verify dynamics in main graph unchanged
- Verify no grace flows between components

---

## 5. Parameter Sensitivity

Run primary scenario (α) across parameter grid:

| Parameter | Range | Purpose |
|:---|:---|:---|
| $\eta_g$ | [0.01, 0.1, 0.5, 1.0] | Grace strength |
| $\beta_g$ | [0.1, 0.3, 0.5] | Need threshold |
| $\langle C \rangle_{\text{init}}$ | [0.2, 0.5, 0.8] | Classical vs quantum regime |
| $\langle a \rangle_{\text{init}}$ | [0.1, 0.3, 0.5] | Agency level |

**Key question:** Is there a phase boundary where grace transitions from redundant to necessary?

---

## 6. Predictions

Based on theoretical analysis:

### 6.1 Grace Necessary Regime

- High coherence ($C > 0.7$)
- Phase misalignment (gradient away from depleted region)
- Moderate agency in depleted region ($a \in [0.2, 0.6]$)
- Active neighborhood ($D_{\text{pool}} > 0$)

**Prediction:** Grace-enabled recovers; grace-disabled freezes or collapses.

### 6.2 Grace Redundant Regime

- Low coherence ($C < 0.3$) — classical diffusion dominates
- Phase aligned or irrelevant
- High agency everywhere

**Prediction:** Both systems recover; grace only accelerates.

### 6.3 Grace Insufficient Regime

- Very low agency everywhere ($a < 0.1$)
- No neighborhood activity ($D_{\text{pool}} \approx 0$)

**Prediction:** Neither system recovers. Grace cannot create activity from nothing.

---

## 7. Implementation Notes

### 7.1 Phase Initialization

To create "wrong" phase alignment:

```python
# Depleted region: random phases
theta_depleted = np.random.uniform(0, 2*np.pi, size=(5,5))

# Surrounding region: coherent outward wave
# Phase increases radially from center
def outward_phase(x, y, center):
    dx, dy = x - center[0], y - center[1]
    return np.arctan2(dy, dx)  # Radial phase

theta_surround = outward_phase(X, Y, center=(N//2, N//2))
```

This creates Im(ψ_i* ψ_j) > 0 for bonds pointing outward → flow away from center.

### 7.2 Grace Tracking

Add diagnostic outputs:
```python
grace_total = sum(I_g_to_i for all i)
grace_received = {i: I_g_to_i for i in depleted_region}
D_pool_avg = mean(D_pool_i for i in depleted_region)
```

### 7.3 Outcome Detection

```python
def classify_outcome(F_depleted, F_total_init, F_total_final, C_final):
    f_rec = mean(F_depleted > F_min) 
    if f_rec > 0.9 and mean(C_final) > 0.5:
        return "RECOVERED"
    elif f_rec < 0.1 and abs(F_total_final - F_total_init) < 0.1:
        return "FROZEN"
    elif F_total_final < 0.8 * F_total_init:
        return "COLLAPSED"
    else:
        return "PARTIAL"
```

---

## 8. Expected Outcome Matrix

| Scenario | Coherence | Phase | Grace A | No-Grace B | Conclusion |
|:---|:---|:---|:---|:---|:---|
| α (primary) | High | Misaligned | Recovered | Frozen | **Necessary** |
| β (isolated) | Medium | Any | Recovered | Slow/Partial | Accelerant |
| γ (coordinated) | Medium | N/A | Recovered | Boundary only | Necessary for interior |
| δ (agency desert) | Medium | Aligned | Recovered | Partial | Necessary |

If Scenario α shows A: Recovered, B: Frozen → **Grace is a justified operator, not redundant.**

---

## 9. Failure Analysis

If test fails to show necessity:

1. **Grace too weak:** Increase $\eta_g$, verify grace actually injects meaningful amounts
2. **Diffusion stronger than expected:** Classical regime dominates; try higher $C$
3. **Phase not actually misaligned:** Verify Im(ψ_i* ψ_j) has correct sign
4. **Agency too low:** Depleted nodes can't receive anything; increase $a_i$
5. **Setup error:** Verify both systems start from identical states

---

## 10. Deliverables

After running test suite:

1. **Outcome table:** System A vs B for each scenario
2. **Time series plots:** $\langle F \rangle_{\text{depleted}}$ over time
3. **Phase diagram:** Recovery fraction vs ($C$, $\eta_g$)
4. **Conservation audit:** $\sum F_i$ drift over time
5. **Conclusion:** Grace necessary / redundant / insufficient with parameter bounds
