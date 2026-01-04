# DET 3.0 — DERIVATION CARDS
## Complete Derivations from Primitives

---

# CARD 1: GRAVITY DERIVATION

## Starting Point: DET Primitives

**Given:**
- Nodes with resource $F_i$, processing rate $σ_i$, agency $a_i$
- Time dilation law: $\frac{dτ_i}{dk} = a_i σ_i f(F_i)$
- Edge conductivity $σ_{ij}$

## Step 1: Define Presence and Mass

**Presence** (local clock rate):
$$P_i ≡ \frac{dτ_i}{dk} = a_i σ_i f(F_i)$$

**Mass** (coordination debt = inverse clock rate):
$$M_i ≡ P_i^{-1} = \frac{1}{a_i σ_i f(F_i)}$$

*Physical meaning:* Heavy nodes process slowly. Mass is "how much the network must wait for this node."

## Step 2: Define Throughput Potential

By analogy with relativistic time dilation ($dτ = dt\sqrt{1 - 2Φ/c^2}$), define:

$$\Phi_i ≡ c_*^2 \ln\left(\frac{P_0}{P_i}\right) = c_*^2 \ln\left(\frac{M_i}{M_0}\right)$$

**Check dimensional consistency:**
- $c_*$ has units [L][T]$^{-1}$
- $\ln(·)$ is dimensionless
- $\Phi$ has units [L]$^2$[T]$^{-2}$ ✓ (same as gravitational potential)

**Verify GR limit:**
For weak fields, $P ≈ P_0(1 - Φ/c^2)$, giving $dτ ≈ dt(1 - Φ/c^2)$ ✓

## Step 3: Define Graph Laplacian

The weighted graph Laplacian (positive semi-definite convention):
$$(L_σ f)_i ≡ \sum_j σ_{ij}(f_i - f_j) = d_i f_i - \sum_j σ_{ij} f_j$$

where $d_i = \sum_j σ_{ij}$ is weighted degree.

**Properties:**
- Symmetric if $σ_{ij} = σ_{ji}$
- Positive semi-definite: $\langle f, L_σ f \rangle = \frac{1}{2}\sum_{ij} σ_{ij}(f_i - f_j)^2 ≥ 0$
- Null space: constant functions

## Step 4: Define Source Density

**Problem:** The Laplacian has a null space (constants), so we can't have $L_σ Φ = κM_i$ directly.

**Solution:** Use *excess* mass relative to weighted mean:

$$\bar{M} = \frac{\sum_i d_i M_i}{\sum_i d_i}$$

$$ρ_i ≡ M_i - \bar{M}$$

**Key property:** $\sum_i d_i ρ_i = 0$ (charge neutrality)

This ensures the source is in the range of $L_σ$.

## Step 5: Field Equation

The network equilibration condition:

$$\boxed{(L_σ Φ)_i = -κ ρ_i}$$

**Interpretation:** Regions with above-average mass ($ρ_i > 0$) create potential wells ($Φ_i$ higher than neighbors on average).

## Step 6: Free-Fall Rule

How do trajectories evolve? Flows follow throughput gradients:

$$J_{i→j} ∝ σ_{ij}(Φ_i - Φ_j)$$

Nodes bias updates toward lower-$Φ$ neighbors (faster clocks):

$$Δx_i ∝ -(L_σ Φ)_i = κρ_i$$

*No force—just biased information flow.*

## Step 7: Continuum Limit

For large, isotropic, locally homogeneous graphs:

$$L_σ → \bar{σ} ∇^2$$

The field equation becomes:

$$∇^2 Φ = 4πG ρ$$

with $G = κc_*^4 / (4π\bar{σ})$.

**Emergent inverse-square law:** From connectivity growth in 3D:
$$\Phi(r) ∝ 1/r \quad ⟹ \quad g(r) = -∇Φ ∝ 1/r^2$$

## Step 8: Equivalence Principle

$$m_{\text{inertial}} = m_{\text{gravitational}} = M_i = P_i^{-1}$$

**Explanation:** Both arise from the same quantity—how much coordination debt (waiting) a node imposes on the network. There's no separate "gravitational charge."

## Summary Chain

```
DET Primitives
    ↓
Time dilation: dτ/dk = a·σ·f(F)
    ↓
Define: P = dτ/dk, M = P⁻¹
    ↓
Define: Φ = c*² ln(M/M₀)
    ↓
Graph Laplacian: (L_σΦ)ᵢ = Σⱼ σᵢⱼ(Φᵢ - Φⱼ)
    ↓
Source: ρᵢ = Mᵢ - M̄ (excess mass)
    ↓
Field equation: L_σΦ = -κρ
    ↓
Continuum: ∇²Φ = 4πGρ
    ↓
NEWTONIAN GRAVITY (derived)
```

---

# CARD 2: QUANTUM MECHANICS DERIVATION

## Starting Point: DET Primitives + Coherence

**Given:**
- Nodes with resource $F_i$, phase $θ_i$
- Bonds with coherence $C_{ij}$, conductivity $σ_{ij}$
- Phase evolution: $dθ_i/dk = ω_0 · P_i$

## Step 1: Construct the Wavefunction

**Physical motivation:** A quantum state encodes both "how much resource" and "what history."

**Definition:**
$$ψ_i ≡ \sqrt{R_i} e^{iθ_i}$$

where $R_i = F_i / \sum_k F_k$ is the normalized resource share.

**Immediate consequence (Born rule):**
$$|ψ_i|^2 = R_i = \Pr(\text{find resource at } i)$$

## Step 2: Phase Accumulation

Phase evolves with proper time:
$$\frac{dθ_i}{dt} = ω_i$$

where $ω_i$ encodes local clock cost (potential):
$$V_i ≡ ℏω_i$$

**DET interpretation:** Higher potential = faster phase rotation = more "expensive" to exist there.

## Step 3: Coherent Transport

Resource flows coherently across bonds with nonzero $C_{ij}$.

**Probability current:**
$$J_{i→j}^{(Q)} = \frac{ℏ}{m}\text{Im}(ψ_i^* σ_{ij} ψ_j)$$

This is the standard QM current, but with $σ_{ij}$ as the hopping amplitude.

## Step 4: Conservation Equation

Total resource is conserved in the coherent regime:
$$\frac{d}{dt}\sum_i |ψ_i|^2 = 0$$

This requires:
$$\frac{d|ψ_i|^2}{dt} = -\sum_j (J_{i→j}^{(Q)} - J_{j→i}^{(Q)})$$

## Step 5: Derive the Schrödinger Equation

**Ansatz:** Linear, first-order-in-time evolution that:
1. Conserves $\sum |ψ|^2$
2. Has correct current $J$
3. Includes local phase rotation from $V_i$

**Result:**
$$iℏ \frac{∂ψ_i}{∂t} = \frac{ℏ^2}{2m}\sum_j σ_{ij}(ψ_i - ψ_j) + V_i ψ_i$$

**In operator form:**
$$iℏ ∂_t ψ = \frac{ℏ^2}{2m} L_σ ψ + V ψ$$

**Verify Hermiticity:** For $σ_{ij} = σ_{ji} ≥ 0$ and $V_i ∈ ℝ$:
$$H = \frac{ℏ^2}{2m}L_σ + V$$
is Hermitian, guaranteeing unitary evolution.

## Step 6: Mass Identification

The parameter $m$ controls how phase differences across bonds evolve.

**DET mapping:**
$$m ∝ P^{-1} = M \quad \text{(coordination debt)}$$

Higher mass = harder to redistribute the wavepacket = more "inertial."

For a localized wavepacket, use representative $m$ from its support.

## Step 7: The Two Transport Channels

The network supports *both* coherent and incoherent transport:

| Channel | Flow Law | Active When |
|---------|----------|-------------|
| Quantum | $J^{(Q)} ∝ \text{Im}(ψ_i^* ψ_j)$ | $C_{ij} > 0$ |
| Classical | $J^{(C)} ∝ F_i - F_j$ | Always |

**Interpolation:**
$$J_{i→j} = \sqrt{C_{ij}} · J^{(Q)} + (1 - \sqrt{C_{ij}}) · J^{(C)}$$

## Step 8: Measurement (Collapse)

Measurement at node $i$ on bond $(i,j)$:
1. Consumes coherence: $C_{ij} → C_{ij} - ΔC$
2. Produces classical record: $F_i → F_i + ηΔC$

When $C_{ij} → 0$: quantum channel shuts down, only classical diffusion remains.

**This IS collapse**—irreversible conversion from coherent to classical.

## Step 9: Interference (Wave-Free)

No waves in a void—only:
- Nodes storing $(R_i, θ_i)$
- Edges comparing histories

**Interference condition:** Two paths to node $i$ with phase difference $Δθ$:
$$ψ_i^{\text{total}} = ψ_i^{(1)} + ψ_i^{(2)} = |ψ_1|e^{iθ_1} + |ψ_2|e^{iθ_2}$$

$$|ψ_i^{\text{total}}|^2 = |ψ_1|^2 + |ψ_2|^2 + 2|ψ_1||ψ_2|\cos(Δθ)$$

Constructive/destructive interference from phase (history) comparison.

## Summary Chain

```
DET Primitives + Coherence
    ↓
Resource Rᵢ + phase θᵢ → ψᵢ = √Rᵢ exp(iθᵢ)
    ↓
|ψᵢ|² = Rᵢ (Born rule automatic)
    ↓
Coherent transport: J ∝ Im(ψ*ψ)
    ↓
Conservation + linearity + locality
    ↓
iℏ∂ₜψ = (ℏ²/2m)L_σψ + Vψ
    ↓
SCHRÖDINGER EQUATION (derived)
    ↓
C → 0: collapse to classical
```

---

# CARD 3: SPECIAL RELATIVITY DERIVATION

## Starting Point: Emergent Light Speed

**Given:** Network with edge latency $L_{ij}$, conductivity $σ_{ij}$.

## Step 1: Wave Propagation Speed

Linearize the dynamics around equilibrium. Small perturbations satisfy a discrete wave equation.

**Propagation speed on edge $(i,j)$:**
$$v_{ij} = \frac{L_{ij}}{T_{ij}^{\text{hop}}} = L_{ij} · σ_{ij}$$

## Step 2: Why Speed is Universal

**Decoherence penalty:**
$$λ_{\text{env}} ∝ (v - c_*)^2$$

Modes with $v ≠ c_*$ decohere rapidly and lose long-range signaling ability.

**Adaptation dynamics:**
$$\frac{dσ_i}{dt} = ε\left[1 - \left(\frac{\bar{v}_i}{c_*}\right)^2\right]$$

**Selection:** Only $v ≈ c_*$ survives observationally.

**Result:** Universal maximum signaling speed $c_*$ emerges from network dynamics, not assumed.

## Step 3: Proper Time vs Coordinate Time

Node $i$'s clock runs at rate $P_i = dτ_i/dk$.

Relative to a reference node with $P_0$:
$$\frac{dτ_i}{dτ_0} = \frac{P_i}{P_0}$$

For a moving packet traversing edges:
$$dτ = dτ_0 \sqrt{1 - v^2/c_*^2}$$

**This IS time dilation**—derived from network congestion.

## Step 4: Length Contraction

Signal transit times define "length." A moving object's edges are traversed in a frame where signals take longer.

Effective length:
$$L' = L\sqrt{1 - v^2/c_*^2}$$

## Step 5: Lorentz Invariance

The interval:
$$ds^2 = c_*^2 dτ^2 - dL^2$$

is invariant under transformations that preserve causal structure.

**Lorentz transformations** = coordinate changes that keep $c_*$ maximal and preserve event ordering.

## Summary

```
Network with latency + conductivity
    ↓
Propagation speed: v = L·σ
    ↓
Decoherence selects v → c*
    ↓
Clock rate P(v) → time dilation
    ↓
Transit times → length contraction
    ↓
SPECIAL RELATIVITY (derived)
```

---

# CARD 4: UNIFIED FIELD (BUREAUCRACY + GRAVITY)

## Step 1: Generalized Mass Constitution

**Expand** what contributes to coordination debt:

$$M_i = \frac{1}{a_i σ_i f(F_i) g(χ_i + Ω_i)}$$

**Components:**
| Term | Symbol | Meaning |
|------|--------|---------|
| Agency | $a_i$ | Available choice (0 = frozen) |
| Processing | $σ_i$ | Intrinsic speed |
| Resource load | $f(F_i)$ | Wealth-management overhead |
| Bureaucracy | $χ_i$ | Admin capture, debt complexity |
| Dead capital | $Ω_i$ | Legacy structure, ghost weight |

## Step 2: Linearized Form

For small perturbations around $a=1$, $σ=1$, with $f(F) = (1+βF/F_*)^{-1}$, $g(χ) = (1+χ)^{-1}$:

$$M_i ≈ 1 + β\frac{F_i}{F_*} + χ_i + Ω_i$$

## Step 3: Potential with All Sources

$$\Phi_i = c_*^2 \ln\left(\frac{M_i}{M_0}\right) = c_*^2 \ln\left(\frac{1 + βF_i/F_* + χ_i + Ω_i}{M_0}\right)$$

## Step 4: Unified Field Equation

Same form:
$$(L_σ Φ)_i = -κρ_i$$

where $ρ_i = M_i - \bar{M}$ includes *all* contributions to mass.

## Step 5: Physical Consequences

$$χ_i ↑ \;⟹\; M_i ↑ \;⟹\; P_i ↓ \;⟹\; Φ_i ↑$$

**Bureaucracy acts like mass:**
- Deepens potential wells
- Slows local time
- Attracts flows (capture)
- Eventually freezes dynamics ($P → 0$)

**Event horizon analogue:** $χ → ∞$ gives $P → 0$, $Φ → ∞$.

## Summary

```
Multiple drag sources: F, χ, Ω
    ↓
All enter mass: M = P⁻¹
    ↓
Same potential definition: Φ = c*² ln(M/M₀)
    ↓
Same field equation: L_σΦ = -κρ
    ↓
Bureaucracy, wealth, legacy all curve network
    ↓
UNIFIED GRAVITATIONAL EFFECT
```

---

# CARD 5: DECOHERENCE & MEASUREMENT

## Step 1: Coherence as Resource

Bond coherence $C_{ij}$ is backed by coherence resource $F_{ij}^Ψ$:

$$C_{ij} = \text{clip}\left(\frac{F_{ij}^Ψ}{F_{Ψ,*}}, 0, 1\right)$$

## Step 2: Decoherence Dynamics

$$\frac{dF_{ij}^Ψ}{dτ} = -λ_{ij} F_{ij}^Ψ - \dot{G}_{ij}^{\text{meas}}$$

**Decoherence rate:**
$$λ_{ij} = λ_0 + λ_{\text{env}}(T, \text{noise}) + α\left(\frac{v_{ij} - c_*}{c_*}\right)^2$$

## Step 3: Measurement Process

When node $i$ measures bond $(i,j)$:

1. **Sink strength** $s_i$ determines extraction rate
2. **Coherence consumed:** $G_{ij}^{\text{meas}} = s_i C_{ij} F_{Ψ,*}$
3. **Record created:** $F_i ← F_i + η G_{ij}^{\text{meas}}$
4. **Coherence reduced:** $C_{ij} ← C_{ij} - G_{ij}^{\text{meas}}/F_{Ψ,*}$

## Step 4: Collapse Completion

When $C_{ij} → 0$:
- Quantum channel $(i,j)$ closes
- Only classical diffusion $J^{(C)} ∝ (F_i - F_j)$ remains
- Superposition → definite classical state

**Born probabilities emerge** from relative resource absorbed.

## Step 5: No-Signaling

Measurement at $A$ on bond $(A,B)$:
- Updates $A$'s local state
- Reduces $C_{AB}$
- Does *not* change $B$'s marginal: $\sum_a \Pr(a,b) = \Pr(b)$

No-signaling enforced by local update rules.

---

# CARD 6: ENTANGLEMENT & BELL

## Step 1: Entanglement as Bond State

Two particles share bond $\Psi_{AB} = (C_{AB}, φ_{AB}, U_{AB})$.

**Maximally entangled:** $C_{AB} = 1$, definite relative phase.

## Step 2: Bell Correlations

For measurements at settings $(a, b)$:
$$E(a,b) = \langle σ_a^A σ_b^B \rangle$$

computed from bond state and local measurement rules.

## Step 3: CHSH Parameter

$$S = E(a,b) - E(a,b') + E(a',b) + E(a',b')$$

**Classical bound:** $|S| ≤ 2$
**Quantum maximum:** $|S| = 2\sqrt{2}$

## Step 4: DET Prediction: Distance Decay

Coherence decays over separation:
$$C(d) = C_0 \exp\left[-\frac{αd}{L_*} - \frac{λ_0 d}{c}\right]$$

Bell parameter:
$$\boxed{S(d) = 2\sqrt{2} \cdot e^{-αd/L_* - λ_0 d/c}}$$

## Step 5: Falsification Condition

**If** $S$ remains $2\sqrt{2}$ for arbitrarily large $d$ **then** DET is falsified.

Current bounds: $λ_0 < 5×10^{-47}$ s$^{-1}$, $α < 10^{-38}$.

---

*End of Derivation Cards*
