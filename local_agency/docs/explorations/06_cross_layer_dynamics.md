# Exploration 06: Cross-Layer Bond Dynamics (P↔A Membrane)

**Status**: RESOLVED
**Date**: 2026-01-17

---

## Goal

Make P↔A bonds behave like a **membrane compiler**:
- **A→P escalation**: novelty/conflict in fast A-routines recruits deliberate P attention
- **P→A compilation**: stabilized P patterns compile into new/stronger A routines, freeing P bandwidth

**Constraints**:
- All triggers are **local** (node/bond neighborhood only)
- No direct edits to `a_i`
- No global normalizations

---

## A. One Bond Law, Three Parameter Regimes

Use the same coherence update form for all bonds—just vary coefficients by layer:

```
C_ij⁺ = clip(
    C_ij
    + α × |J_i↔j| × Δτ_ij           // growth from flow
    - λ × C_ij × Δτ_ij               // passive decay
    - λ_slip × C_ij × S_ij × Δτ_ij,  // phase-slip penalty
    C_min, 1
)
```

Where the **phase-slip / mismatch term** is local:

```
S_ij ≡ 1 - cos(θ_i - θ_j)
```

### Parameter Regimes

| Regime | α | λ | λ_slip | Behavior |
|--------|---|---|--------|----------|
| **Within A** (A↔A) | High | High | Low | Fast, plastic: learn quickly, forget quickly |
| **Within P** (P↔P) | Low | Low | Low | Slow, stable: deliberate, persistent |
| **Cross P↔A** (membrane) | Medium | Medium | Medium | Medium plasticity + phase-slip penalty to prevent sticky incoherent bridges |

**Answer to sub-question**: "Should cross-layer bonds have different dynamics?"
→ **Yes, via coefficients, not new physics.**

### C Implementation

```c
typedef struct {
    float alpha;      // growth rate
    float lambda;     // decay rate
    float lambda_slip; // phase-slip decay
} BondParams;

// Parameter sets for each regime
static const BondParams PARAMS_AA = {0.15f, 0.08f, 0.02f};  // fast, plastic
static const BondParams PARAMS_PP = {0.03f, 0.01f, 0.01f};  // slow, stable
static const BondParams PARAMS_PA = {0.08f, 0.04f, 0.06f};  // medium + slip penalty

BondParams get_bond_params(NodeLayer layer_i, NodeLayer layer_j) {
    if (layer_i == LAYER_A && layer_j == LAYER_A) return PARAMS_AA;
    if (layer_i == LAYER_P && layer_j == LAYER_P) return PARAMS_PP;
    return PARAMS_PA;  // cross-layer
}
```

---

## B. A→P Escalation (Novelty Recruits Deliberation)

### B.1 Local Novelty Score

Define a local novelty score at each A-node `j` using only its neighborhood:

```
N_j = EMA(|ΔJ_j|)           // instability: change in flux
    + η_q × EMA(Δq_j⁺)      // rising debt
    + η_s × EMA(1 - C̄_j)    // fragmentation
```

Where:
- `ΔJ_j` = change in total incident flux magnitude at j over a short window
- `C̄_j` = average coherence in j's neighborhood: `(1/|N(j)|) × Σ_{k∈N(j)} C_jk`
- `EMA` = exponential moving average (one accumulator per node)

### B.2 Escalation Trigger

Purely local decision:

```
escalate(j) ⟺ N_j > N_th ∧ a_j > 0
```

### B.3 Escalation Actions (Allowed)

When escalation triggers:

1. **Resource spotlight**: Inject small amount of F into nearby P-nodes through existing P↔A interface bonds

2. **Open cross-layer attention bonds**: Temporarily increase `σ_PA` on edges from j to a small set of P-nodes in local neighborhood, or spawn temporary edges from a predefined sparse adjacency list

```c
void escalate_a_node(DETCore* core, uint16_t j) {
    // Find nearby P-nodes (via existing cross-layer bonds or adjacency list)
    for (uint32_t b = 0; b < core->num_bonds; b++) {
        BondState* bond = &core->bonds[b];
        if (!is_cross_layer_bond(core, bond)) continue;
        if (bond->j != j && bond->i != j) continue;

        uint16_t p_node = (bond->i == j) ? bond->j : bond->i;
        if (core->nodes[p_node].layer != LAYER_P) continue;

        // Inject resource spotlight
        core->nodes[p_node].F += ESCALATION_F_INJECT;

        // Temporarily increase conductivity
        bond->sigma += ESCALATION_SIGMA_BOOST;
    }
}
```

**Safety valve**: Escalation only increases coupling *potential*. Coherence still must earn itself through |J| and phase alignment.

---

## C. P→A Compilation (Stabilized Deliberation Becomes Habit)

### C.1 Local Stability Score

Define stability on each cross-layer bond:

```
B_ij = EMA(|J_ij|) × EMA(C_ij) × EMA(cos(θ_i - θ_j))
```

This captures:
- Sustained flux (used connection)
- High coherence (integrated)
- Phase alignment (synchronized)

### C.2 Compilation Trigger

A P-node `i` compiles into A when a small "fan-in" of A-nodes is stably bridged:

```
compile(i) ⟺ ∃ K ⊂ N(i) ∩ A, |K| ≥ k_min :
              (1/|K|) × Σ_{j∈K} B_ij > B_th
```

### C.3 Compilation Actions (All Local)

When compilation triggers:

1. **Strengthen A↔A bonds inside K**:
   ```
   C_jk ← clip(C_jk + δ_C, C_min, 1)  for j,k ∈ K
   ```
   Or let ordinary coherence growth do it, with a small catalyst term gated by B_ij.

2. **Decouple P↔A over time** (free P capacity):
   ```
   λ_PA ← λ_PA + δ_λ  on those (i,j)
   ```
   So the bridge fades unless novelty returns.

```c
void compile_p_node(DETCore* core, uint16_t i) {
    // Find the A-nodes that are stably bridged to this P-node
    uint16_t stable_a_nodes[MAX_COMPILE_FAN];
    uint32_t num_stable = 0;

    for (uint32_t b = 0; b < core->num_bonds; b++) {
        BondState* bond = &core->bonds[b];
        if (!involves_node(bond, i)) continue;

        uint16_t other = (bond->i == i) ? bond->j : bond->i;
        if (core->nodes[other].layer != LAYER_A) continue;

        float B_ij = compute_stability_score(core, bond);
        if (B_ij > B_THRESHOLD) {
            stable_a_nodes[num_stable++] = other;

            // Increase decay on P↔A bond (free P capacity)
            bond->lambda_decay += COMPILE_DECAY_BOOST;
        }
    }

    // Strengthen A↔A bonds among the compiled set
    if (num_stable >= K_MIN_COMPILE) {
        for (uint32_t x = 0; x < num_stable; x++) {
            for (uint32_t y = x + 1; y < num_stable; y++) {
                BondState* aa_bond = find_or_create_bond(
                    core, stable_a_nodes[x], stable_a_nodes[y]
                );
                aa_bond->C = fminf(aa_bond->C + COMPILE_C_BOOST, 1.0f);
            }
        }
    }
}
```

---

## D. Bond Breaking / Forgetting Rules (Anti-Prison)

Cross-layer bonds should decay quickly if **unused** or **incoherent**:

### D.1 Unused Bond Decay

```
if EMA(|J_ij|) < J_min:
    λ_PA ↑  (increase decay)
```

### D.2 Incoherent Bond Decay (Prevents Trap)

```
if EMA(cos(θ_i - θ_j)) < cos_min:
    λ_slip ↑  (phase-slip penalty increases)
```

### D.3 Debt Overload (Keeps P Clean)

If `q_i` or `q_j` locally exceeds threshold, reduce coupling `σ_ij` (not `a`):

```
σ_ij ← σ_ij × 1/(1 + κ_q × q_max(i,j))
```

### C Implementation

```c
void apply_anti_prison_rules(DETCore* core, BondState* bond) {
    float J_ema = bond->flux_ema;
    float phase_ema = bond->phase_align_ema;
    float q_max = fmaxf(core->nodes[bond->i].q, core->nodes[bond->j].q);

    // Unused: increase decay
    if (J_ema < J_MIN_THRESHOLD) {
        bond->lambda_decay += UNUSED_DECAY_BOOST;
    }

    // Incoherent: increase slip penalty
    if (phase_ema < COS_MIN_THRESHOLD) {
        bond->lambda_slip += INCOHERENT_SLIP_BOOST;
    }

    // Debt overload: reduce conductivity
    bond->sigma *= 1.0f / (1.0f + KAPPA_Q * q_max);
}
```

---

## E. Timescales

Three "half-lives" for different layers:

| Layer | Plasticity Speed | Typical Half-Life |
|-------|-----------------|-------------------|
| **A-layer** | Fast | Seconds–minutes (sim ticks) |
| **P-layer** | Slow | Minutes–hours |
| **P↔A membrane** | Medium | Minutes |

**EMA windows**:
- Escalation (novelty detection): Short EMA (fast response)
- Compilation (stability detection): Long EMA (requires persistence)

This matches the roadmap intent without needing "sleep" machinery yet.

### EMA Implementation

```c
typedef struct {
    float short_alpha;  // 0.3 - fast response (escalation)
    float long_alpha;   // 0.05 - slow response (compilation)
} EMAParams;

float update_ema(float current, float new_value, float alpha) {
    return alpha * new_value + (1.0f - alpha) * current;
}
```

---

## F. Complete Update Loop (Pseudocode)

```python
def cross_layer_update(core: DETCore, dt: float):
    """
    Per-tick update for P↔A membrane dynamics.
    """

    # 1. Update fluxes J, coherence C normally (using layer-specific params)
    for bond in core.bonds:
        params = get_bond_params(bond.layer_i, bond.layer_j)
        update_coherence(bond, params, dt)

    # 2. For each A-node: check escalation
    for j in core.a_nodes:
        # Compute local novelty score
        j.novelty_ema = update_ema(j.novelty_ema, compute_novelty(j), SHORT_ALPHA)

        if j.novelty_ema > N_THRESHOLD and j.a > 0:
            # Escalate: spotlight nearby P-nodes
            escalate_a_node(core, j)

    # 3. For each P-node: check compilation
    for i in core.p_nodes:
        # Compute stability scores on cross-layer bonds
        stable_bonds = []
        for bond in get_cross_layer_bonds(core, i):
            bond.stability_ema = update_ema(
                bond.stability_ema,
                compute_stability(bond),
                LONG_ALPHA
            )
            if bond.stability_ema > B_THRESHOLD:
                stable_bonds.append(bond)

        # If enough stable bridges: compile
        if len(stable_bonds) >= K_MIN:
            compile_p_node(core, i, stable_bonds)

    # 4. Apply anti-prison rules on all cross-layer bonds
    for bond in core.cross_layer_bonds:
        apply_anti_prison_rules(core, bond)
```

---

## G. Parameter Summary

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| **Novelty threshold** | N_th | 0.5 | Triggers A→P escalation |
| **Stability threshold** | B_th | 0.7 | Triggers P→A compilation |
| **Min compile fan-in** | k_min | 3 | Minimum A-nodes for compilation |
| **Escalation F inject** | - | 0.1 | Resource boost to P-nodes |
| **Escalation σ boost** | - | 0.2 | Conductivity boost on attention bonds |
| **Compile C boost** | δ_C | 0.1 | Coherence boost for A↔A bonds |
| **Compile decay boost** | δ_λ | 0.05 | Increased decay on compiled P↔A bonds |
| **Debt coupling** | κ_q | 0.5 | Conductivity reduction from debt |
| **Short EMA α** | - | 0.3 | Fast novelty detection |
| **Long EMA α** | - | 0.05 | Slow stability detection |

---

## H. Summary

| Aspect | Resolution |
|--------|------------|
| **Bond physics** | One law, three parameter regimes (α, λ, λ_slip) |
| **A→P escalation** | Local novelty score N_j > threshold triggers resource spotlight + attention bonds |
| **P→A compilation** | Local stability score B_ij > threshold on k_min bonds triggers A↔A strengthening + P↔A decay |
| **Anti-prison** | Unused, incoherent, or debt-overloaded bonds decay faster |
| **Timescales** | A: fast, P: slow, P↔A: medium |
| **All operations** | Local only, no direct a_i edits, no global normalization |

---

## I. Open Sub-Questions (Deferred)

1. **Optimal parameter tuning**: Needs simulation to find good defaults
2. **Sparse adjacency for escalation**: How to initialize potential P↔A bonds?
3. **Compilation cascade**: Can compiled A-clusters trigger further compilation?
4. **Sleep consolidation**: How does this interact with "sleep" retraining?

These can be explored during Phase 4 implementation.
