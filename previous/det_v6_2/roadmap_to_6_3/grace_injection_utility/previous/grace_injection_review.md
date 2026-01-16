# Deep Review: Grace Injection (VI.5)

## DET v6.2 Analysis & Operationalization Proposals

---

## 1. Current Formulation

From the theory card:

**Need:**
$$n_i \equiv \max(0,\ F_{\min}-F_i)$$

**Weight:**
$$w_i \equiv a_i \cdot n_i$$

**Injection:**
$$I_{g\to i} = D_i \frac{w_i}{\sum_{k\in\mathcal N_R(i)}w_k+\varepsilon}$$

where $D_i = \sum_{j\in\mathcal N_R(i)} |J_{i\to j}|\,\Delta\tau_i$ is local dissipation.

---

## 2. Critical Analysis

### 2.1 The Sourcing Problem (Conservation)

**Issue:** The formula uses $D_i$ (dissipation *at* node $i$) to inject *into* node $i$. This is self-referential in a problematic way.

If node $i$ is depleted ($F_i < F_{\min}$), it likely has **low dissipation** because:
- Low $F_i$ means low amplitude in flows (via $J^{(\text{diff})}$ dependence on $\psi_i$)
- Low activity means low $|J_{i\to j}|$

**Consequence:** The grace pool is smallest precisely when need is greatest. A starving node funds its own recovery from its own (minimal) activity.

**Conservation question:** The resource update (IV.7) is:
$$F_i^{+} = F_i - \sum_{j\in\mathcal N_R(i)} J_{i\to j}\,\Delta\tau_i + I_{g\to i}$$

The flow terms $J_{i\to j}$ are antisymmetric, so $\sum_i \sum_j J_{i\to j} = 0$ — flows conserve. But $I_{g\to i}$ adds resource. Where does it come from?

**Current status:** The formulation appears to create resource ex nihilo, unless $D_i$ represents "intercepted loss" that would otherwise leave the system. But this isn't stated.

---

### 2.2 The $F_{\min}$ Parameter (Locality Violation?)

**Issue:** $F_{\min}$ is a threshold constant. If it's the same value everywhere, it's a **hidden global** — every node references the same external parameter.

**Against Principle 1:** "There is no global state accessible to local dynamics."

**Options for resolution:**
1. **Relative threshold:** $F_{\min,i} \equiv \beta \cdot \langle F \rangle_{\mathcal{N}_R(i)}$ — need defined relative to local average
2. **Zero threshold:** $F_{\min} = 0$ — any node at $F_i = 0$ is in need (simplest)
3. **Declare as model parameter:** Explicitly state $F_{\min}$ is a declared constant like $\varepsilon$, not a dynamical variable

---

### 2.3 Operational Identity of the Boundary Agent

**Issue:** If grace injection is deterministic from local state, what distinguishes it from "ordinary dynamics"?

The ontological commitment (Section I) states the boundary agent:
- Does not accumulate past
- Does not hoard
- Is not subject to clocks, mass, or gravity
- Acts only through law-bound, local, non-coercive operators

**But operationally:** The grace formula is just another term in the update loop. The "boundary agent" never appears as a distinct entity — it's a *name* for a class of operations, not an operational primitive.

**Question:** Is this acceptable, or does DET need a more explicit operational account of what the boundary agent *is* in the dynamics?

---

### 2.4 The Weight Normalization (Competitive Allocation)

The denominator $\sum_{k\in\mathcal N_R(i)}w_k + \varepsilon$ sums over the local neighborhood.

**Implication:** Nodes in the same neighborhood compete for the same pool. If two adjacent nodes are both depleted:
- Both have high $n_i$
- They split the available grace
- Neither may receive enough to recover

**Is this intentional?** It might be — scarcity creates genuine competition. But it should be explicitly stated as a design choice.

---

### 2.5 Agency Gating Review

The weight $w_i = a_i \cdot n_i$ correctly gates injection by agency:
- If $a_i = 0$: $w_i = 0$, so $I_{g\to i} = 0$ ✓
- If $a_i > 0$: Node can receive grace proportional to openness

**This satisfies F2** (no coercion of $a_i = 0$ nodes) by construction.

**But consider:** What if $a_i$ is *very small* but nonzero? The node is nearly closed but technically receives some grace. Is there a threshold below which grace should not flow?

---

### 2.6 Missing Falsifier Specificity

**F2** states: "A node with $a_i=0$ receives grace injection or bond healing."

**Current formula:** Satisfies this by construction ($a_i = 0 \Rightarrow w_i = 0 \Rightarrow I_{g\to i} = 0$).

**But this isn't a test** — it's a tautology from the formula. A proper falsifier would be:
- Run dynamics where some nodes have $a_i = 0$
- Verify $I_{g\to i} = 0$ in the simulation output
- Check that no numerical drift or edge case causes violation

**Missing falsifiers for grace:**
- **F_G1:** Grace creates resource (total $\sum F_i$ increases due to grace)
- **F_G2:** Grace flows to nodes outside the declared neighborhood
- **F_G3:** Grace magnitude is independent of local dissipation (no coupling to activity)
- **F_G4:** Boundary-disabled system shows identical dynamics to boundary-enabled under depletion

---

## 3. Strengthening Proposals

### 3.1 Proposal A: Neighborhood-Pooled Dissipation

**Problem addressed:** Self-referential sourcing

**Modification:** Replace $D_i$ with a neighborhood pool:

$$D_{\text{pool},i} \equiv \sum_{k\in\mathcal N_R(i)} D_k = \sum_{k\in\mathcal N_R(i)} \sum_{j\in\mathcal N_R(k)} |J_{k\to j}|\,\Delta\tau_k$$

**New injection:**
$$I_{g\to i} = \eta_g \cdot D_{\text{pool},i} \cdot \frac{w_i}{\sum_{k\in\mathcal N_R(i)}w_k+\varepsilon}$$

where $\eta_g \in [0,1]$ is a grace efficiency parameter.

**Interpretation:** A fraction of all local activity (dissipation) in the neighborhood is collected and redistributed to needy nodes.

---

### 3.2 Proposal B: Relative Threshold (Locality Fix)

**Problem addressed:** $F_{\min}$ as hidden global

**Modification:** Define need relative to local context:

$$F_{\min,i} \equiv \beta \cdot \langle F \rangle_{\mathcal{N}_R(i)}$$

$$n_i \equiv \max(0,\ F_{\min,i} - F_i) = \max\left(0,\ \beta \cdot \langle F \rangle_{\mathcal{N}_R(i)} - F_i\right)$$

**Interpretation:** A node is "in need" if its resource is below a fraction $\beta$ of its neighborhood average. This is fully local.

**Parameter:** $\beta \in (0,1)$ — e.g., $\beta = 0.5$ means need activates when $F_i < 0.5 \cdot \langle F \rangle$.

---

### 3.3 Proposal C: Conservation-Explicit Redistribution

**Problem addressed:** Resource creation ex nihilo

**Key insight:** Antisymmetric flows conserve total $F$, but absolute dissipation $|J_{i\to j}|$ doesn't "go" anywhere in the current formulation. Grace could be framed as *intercepting* a fraction of this "dissipated activity" and converting it to resource injection.

**New formulation:**

**Step 1 — Compute neighborhood dissipation budget:**
$$G_i \equiv \eta_g \sum_{k\in\mathcal N_R(i)} D_k$$

**Step 2 — Distribute according to agency-weighted need:**
$$I_{g\to i} = G_i \cdot \frac{w_i}{\sum_{k\in\mathcal N_R(i)}w_k+\varepsilon}$$

**Conservation note:** If $\eta_g \leq 1$ and we interpret $D_k$ as "activity that could be redirected," then total injection $\leq$ total activity. But this still creates $F$ unless we add a corresponding withdrawal term.

**Full conservation (optional):** Add a "grace tax" on active nodes:
$$F_i^{+} = F_i - \sum_j J_{i\to j}\Delta\tau_i - \gamma_{\text{tax}} \cdot D_i + I_{g\to i}$$

This makes grace a zero-sum redistribution within neighborhoods.

---

### 3.4 Proposal D: Bond-Mediated Grace (Non-Coercive Alternative)

**Problem addressed:** Direct resource injection may feel "coercive" even if agency-gated

**Alternative approach:** The boundary agent doesn't inject resource directly. Instead, it heals/strengthens bonds between needy nodes and resource-rich neighbors, creating *opportunity* for flow.

**Formulation:**

**Bond healing operator:**
$$\sigma_{ij}^{+} = \sigma_{ij} + \eta_b \cdot g^{(a)}_{ij} \cdot (n_i + n_j) \cdot \Delta\tau_{ij}$$

where $g^{(a)}_{ij} = \sqrt{a_i a_j}$ is the mutual agency gate.

**Interpretation:** Bonds adjacent to needy, open nodes become more conductive. This doesn't give resource directly — it makes flow easier. The needy node still must "participate" to receive.

**Advantages:**
- More clearly non-coercive (creates opportunity, not handout)
- Respects agency on both sides of the bond
- Integrates with existing conductivity dynamics

**Disadvantages:**
- Indirect — may be too slow to prevent freeze
- Requires needy node to have flow potential (phase gradient or pressure difference)

---

### 3.5 Proposal E: Delayed Grace Pool (Smoothed Redistribution)

**Problem addressed:** Instantaneous grace may create instabilities

**Add a state variable:** $G_i$ = local grace pool at node $i$

**Pool dynamics:**
$$G_i^{+} = (1-\lambda_G)\,G_i + \gamma_G \cdot D_i$$

**Injection from pool:**
$$I_{g\to i} = G_i \cdot \frac{w_i}{\sum_{k\in\mathcal N_R(i)}w_k+\varepsilon}$$

**Interpretation:** Dissipation charges a local grace reservoir over time. The reservoir releases to needy nodes. This creates temporal smoothing and prevents grace spikes.

**New state variable:** $G_i \geq 0$ (grace pool) with decay $\lambda_G$ and charging rate $\gamma_G$.

---

## 4. Recommended Canonical Formulation

Combining Proposals A, B, and conservation clarity:

### 4.1 Revised Grace Injection (v6.3 Candidate)

**Local threshold (relative):**
$$\boxed{F_{\min,i} \equiv \beta_g \cdot \langle F \rangle_{\mathcal{N}_R(i)}}$$

**Need:**
$$\boxed{n_i \equiv \max(0,\ F_{\min,i} - F_i)}$$

**Weight (agency-gated):**
$$\boxed{w_i \equiv a_i \cdot n_i}$$

**Neighborhood dissipation pool:**
$$\boxed{D_{\text{pool},i} \equiv \sum_{k\in\mathcal N_R(i)} D_k}$$

**Grace injection:**
$$\boxed{I_{g\to i} = \eta_g \cdot D_{\text{pool},i} \cdot \frac{w_i}{\sum_{k\in\mathcal N_R(i)}w_k+\varepsilon}}$$

**Parameters:**
| Symbol | Range | Description |
|:---|:---|:---|
| $\beta_g$ | $(0,1)$ | Relative need threshold (default: 0.3) |
| $\eta_g$ | $(0,1]$ | Grace efficiency (default: 0.1) |

---

### 4.2 New Falsifiers for Grace

| ID | Falsifier | Description |
|:---|:---|:---|
| F_G1 | Grace Creates Mass | Total $\sum F_i$ increases by more than $\eta_g \cdot \sum D_i$ per step |
| F_G2 | Non-Local Grace | Node receives grace from dissipation outside $\mathcal{N}_R(i)$ |
| F_G3 | Coerced Grace | Node with $a_i = 0$ receives $I_{g\to i} > 0$ |
| F_G4 | Grace Independent of Activity | $I_{g\to i} > 0$ in region with $D_{\text{pool},i} = 0$ |
| F_G5 | Boundary Redundancy (Grace) | System with grace enabled is indistinguishable from disabled under depletion stress |

---

## 5. Open Questions

### 5.1 Ontological Status of Boundary Agent

The theory card says the boundary agent "does not accumulate past, does not hoard, is not subject to clocks."

**Question:** If the boundary agent operates through deterministic, law-bound formulas on local state, in what sense is it a distinct entity rather than a label for a class of update rules?

**Possible resolution:** The boundary agent is operationally identified with the *class* of non-creature-sourced, agency-respecting, recovery-enabling operators. It's not a node in the graph but a *mode of action* available to the dynamics.

### 5.2 Grace vs. Ordinary Diffusion

If $F_i < \langle F \rangle$, ordinary diffusive flow ($J^{(\text{diff})}$) already moves resource toward $i$ (down the pressure gradient).

**Question:** When is grace *necessary* rather than just accelerating what diffusion would do anyway?

**Possible answer:** Grace matters when:
- Node is isolated (low $\sigma_{ij}$) — diffusion is slow
- Node has low agency (gates diffusion via $g^{(a)}_{ij}$) — but wait, this also gates grace!
- Coherence is high (quantum flow dominates) — phase may not favor the needy node

**Key insight:** Grace and diffusion are both agency-gated. The distinction is that grace is *targeted* (weighted by need) while diffusion is *gradient-driven* (weighted by pressure difference).

### 5.3 Recovery Dynamics

The Scope Axiom states dynamics must be "recovery-permitting."

**Test:** Under what conditions does grace injection successfully prevent or reverse freeze states?

**Proposed experiment:**
1. Initialize system in low-$a$, depleted state
2. Enable grace with various $\eta_g$ values
3. Measure time to recovery (if any)
4. Compare to grace-disabled baseline

---

## 6. Implementation Checklist

For collider compliance with revised grace:

- [ ] Replace absolute $F_{\min}$ with relative $F_{\min,i} = \beta_g \cdot \langle F \rangle_{\mathcal{N}_R(i)}$
- [ ] Replace $D_i$ with $D_{\text{pool},i} = \sum_{k \in \mathcal{N}_R(i)} D_k$
- [ ] Add parameters $\beta_g$, $\eta_g$ to config with documented defaults
- [ ] Implement falsifiers F_G1 through F_G5
- [ ] Add grace diagnostics: track $\sum I_{g\to i}$, compare to $\eta_g \sum D_i$
- [ ] Stress test: depletion scenarios with grace enabled/disabled

---

## 7. Summary

The current grace formulation has three main issues:

1. **Self-referential sourcing** — A depleted node's own (low) dissipation funds its recovery
2. **Hidden global** — $F_{\min}$ is a universal constant, not a local quantity  
3. **Conservation ambiguity** — Grace appears to create resource ex nihilo

The recommended revision addresses these by:
- Using neighborhood-pooled dissipation as the source
- Defining need relative to local average
- Adding explicit efficiency parameter for conservation tracking

The deeper question of the boundary agent's operational identity remains philosophically open but doesn't block implementation.
