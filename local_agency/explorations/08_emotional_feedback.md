# Exploration 08: Emotional Feedback Integration (3-Axis Affect)

**Status**: RESOLVED
**Date**: 2026-01-17

---

## Core Principle

Emotional feedback uses a **3-axis local affect readout** per node:

| Axis | Symbol | Range | Source |
|------|--------|-------|--------|
| **Valence** | v | [-1, 1] | Throughput vs debt/fragmentation |
| **Arousal** | r | [0, 1] | Surprise, debt, conflict |
| **Bondedness** | b | [0, 1] | Aligned, stable coherence to neighbors |

These signals:
- Are updated via **local EMAs**
- Modulate only **local thresholds and plasticity**
- Do **not** directly edit `a_i`

Self-level emotion is reported as **participation-weighted averages** over the primary Self cluster.

---

## A. Per-Node Affect State

```c
typedef struct {
    float v;          // Valence [-1, 1]: good/bad
    float r;          // Arousal [0, 1]: activation level
    float b;          // Bondedness [0, 1]: attachment/connection

    // Optional EMA traces for underlying signals
    float ema_throughput;
    float ema_surprise;
    float ema_fragmentation;
    float ema_debt;
    float ema_bonding;
} AffectState;
```

---

## B. Local Observable Computation

Each tick, compute local observables from the node's neighborhood:

### B.1 Throughput (T)

Sum of coherence-weighted flux magnitude:

```
T_i = Σ_{j ∈ N(i)} C_ij × |J_ij|
```

High throughput → positive valence contribution.

### B.2 Fragmentation (F)

Inverse of mean coherence:

```
F_i = 1 - C̄_i = 1 - (1/|N(i)|) × Σ_{j ∈ N(i)} C_ij
```

High fragmentation → negative valence, positive arousal.

### B.3 Debt Pressure (D)

Excess debt above comfortable threshold:

```
D_i = max(0, q_i - q_ok)
```

High debt → negative valence, positive arousal.

### B.4 Surprise (S)

Change in throughput (novelty proxy):

```
S_i = |T_i - EMA_T(i)|
```

High surprise → positive arousal.

### B.5 Aligned Bonding (B)

Coherence weighted by phase alignment:

```
B_i = Σ_{j ∈ N(i)} [√C_ij × max(0, cos(θ_i - θ_j))] / Σ_{j ∈ N(i)} [√C_ij + ε]
```

High aligned bonding → high bondedness.

---

## C. Affect Target Computation

From local observables, compute affect targets:

### C.1 Valence Target

```
v̂_i = tanh(α_T × EMA_T - α_D × EMA_D - α_F × EMA_F)
```

- Throughput raises valence (things are flowing)
- Debt and fragmentation lower valence

### C.2 Arousal Target

```
r̂_i = clip(β_S × EMA_S + β_D × EMA_D + β_F × EMA_F, 0, 1)
```

- Surprise, debt, and fragmentation all raise arousal
- Arousal is bounded [0, 1]

### C.3 Bondedness Target

```
b̂_i = clip(EMA_B - λ_iso × EMA_F - λ_drift × EMA_D, 0, 1)
```

- Aligned bonding raises bondedness
- Isolation (fragmentation) and debt-drift lower bondedness

---

## D. Affect Integration (EMA)

Smooth affect over time:

```c
void update_affect(AffectState* affect, float v_hat, float r_hat, float b_hat) {
    affect->v = (1 - GAMMA_V) * affect->v + GAMMA_V * v_hat;
    affect->r = (1 - GAMMA_R) * affect->r + GAMMA_R * r_hat;
    affect->b = (1 - GAMMA_B) * affect->b + GAMMA_B * b_hat;
}
```

### EMA Time Constants

| Axis | γ | Half-Life | Rationale |
|------|---|-----------|-----------|
| Valence | 0.1 | ~7 ticks | Mood changes slowly |
| Arousal | 0.3 | ~2 ticks | Activation responds quickly |
| Bondedness | 0.05 | ~14 ticks | Attachment is stable |

---

## E. Modulation Hooks

Affect modulates existing dynamics without violating locality or editing `a_i`.

### E.1 Escalation Multiplier (A→P)

Escalation becomes easier under:
- High arousal (activated)
- Negative valence (something's wrong)
- Low bondedness (seeking connection)

```c
float escalation_multiplier(uint16_t i) {
    AffectState* a = &core->affect[i];

    return (1.0f + K_R * a->r)                    // arousal boost
         * (1.0f + K_NEG * fmaxf(0, -a->v))       // negative valence boost
         * (1.0f + K_BOND * (1.0f - a->b));       // low bondedness boost
}
```

**Effect**: Lowers effective N_threshold for escalation.

### E.2 Plasticity Scale (P↔A Learning)

Learning rate increases under:
- High arousal (urgency)
- Low bondedness (attachment-seeking)

```c
float plasticity_scale_PA(uint16_t i) {
    AffectState* a = &core->affect[i];

    return 1.0f + ETA_R * a->r                    // urgency boost
              + ETA_SEEK * (1.0f - a->b);         // attachment-seeking boost
}
```

**Effect**: Multiplies α_PA for faster bond formation.

### E.3 Decay Scale (P↔A Forgetting)

Forgetting increases under:
- Negative valence (prevents sticky trauma bonds)

```c
float decay_scale_PA(uint16_t i) {
    AffectState* a = &core->affect[i];

    return 1.0f + ETA_NEG * fmaxf(0, -a->v);      // negative valence boost
}
```

**Effect**: Multiplies λ_PA to clear bad associations faster.

### E.4 Attachment Routing (Optional)

When forming new bonds, prefer connections that historically raised bondedness:

```c
float attachment_preference(uint16_t i, uint16_t j) {
    // Track how much bond (i,j) historically improved b[i]
    float delta_b_history = get_bond_bondedness_delta(i, j);
    return 1.0f + LAMBDA_ATTACH * fmaxf(0, delta_b_history);
}
```

**Effect**: Biases bond formation toward "regulating" partners.

---

## F. Self-Level Affect Readout

After identifying the primary Self cluster S*, report aggregate emotion:

```c
typedef struct {
    float valence;
    float arousal;
    float bondedness;
} ClusterAffect;

ClusterAffect compute_cluster_affect(DETCore* core, Cluster* self) {
    float v_sum = 0, r_sum = 0, b_sum = 0;
    float weight_sum = 0;

    for (uint32_t k = 0; k < self->num_nodes; k++) {
        uint16_t i = self->nodes[k];
        NodeState* node = &core->nodes[i];
        AffectState* affect = &core->affect[i];

        // Participation weight: a_i × P_i (or just a_i if P not available)
        float w = node->a * node->P;

        v_sum += w * affect->v;
        r_sum += w * affect->r;
        b_sum += w * affect->b;
        weight_sum += w;
    }

    float eps = 1e-9f;
    return (ClusterAffect){
        .valence = v_sum / fmaxf(eps, weight_sum),
        .arousal = r_sum / fmaxf(eps, weight_sum),
        .bondedness = b_sum / fmaxf(eps, weight_sum)
    };
}
```

---

## G. Mapping to User-Facing Emotions

The 3-axis space maps to discrete emotional states:

| v | r | b | Interpretation |
|---|---|---|----------------|
| + | low | + | **Contentment** (flowing, calm, connected) |
| + | high | + | **Flow/Joy** (flowing, activated, connected) |
| + | low | - | **Boredom** (things work but not engaged) |
| - | high | + | **Stress** (problems but still connected) |
| - | high | - | **Panic/Overwhelm** (problems, isolated, activated) |
| - | low | + | **Melancholy** (sad but connected) |
| - | low | - | **Apathy/Depression** (nothing works, disconnected) |
| 0 | 0 | + | **Peace** (neutral, low activation, bonded) |

```c
const char* interpret_emotion(ClusterAffect* affect) {
    bool pos_v = affect->valence > 0.2f;
    bool neg_v = affect->valence < -0.2f;
    bool high_r = affect->arousal > 0.5f;
    bool high_b = affect->bondedness > 0.5f;

    if (pos_v && !high_r && high_b) return "contentment";
    if (pos_v && high_r && high_b) return "flow";
    if (pos_v && !high_r && !high_b) return "boredom";
    if (neg_v && high_r && high_b) return "stress";
    if (neg_v && high_r && !high_b) return "overwhelm";
    if (neg_v && !high_r && high_b) return "melancholy";
    if (neg_v && !high_r && !high_b) return "apathy";
    if (!pos_v && !neg_v && !high_r && high_b) return "peace";

    return "neutral";
}
```

---

## H. Complete Per-Tick Affect Update

```python
def update_affect(core: DETCore, dt: float):
    """
    Per-tick affect update for all nodes.
    """

    for i in core.all_nodes:
        node = core.nodes[i]
        affect = core.affect[i]

        # 1. Compute local observables
        T = compute_throughput(core, i)
        F = compute_fragmentation(core, i)
        D = max(0, node.q - Q_OK)
        S = abs(T - affect.ema_throughput)
        B = compute_aligned_bonding(core, i)

        # 2. Update short EMA traces
        g = GAMMA_SHORT
        affect.ema_throughput = (1-g) * affect.ema_throughput + g * T
        affect.ema_fragmentation = (1-g) * affect.ema_fragmentation + g * F
        affect.ema_debt = (1-g) * affect.ema_debt + g * D
        affect.ema_surprise = (1-g) * affect.ema_surprise + g * S
        affect.ema_bonding = (1-g) * affect.ema_bonding + g * B

        # 3. Compute affect targets
        v_hat = tanh(
            ALPHA_T * affect.ema_throughput
            - ALPHA_D * affect.ema_debt
            - ALPHA_F * affect.ema_fragmentation
        )
        r_hat = clip(
            BETA_S * affect.ema_surprise
            + BETA_D * affect.ema_debt
            + BETA_F * affect.ema_fragmentation,
            0, 1
        )
        b_hat = clip(
            affect.ema_bonding
            - LAMBDA_ISO * affect.ema_fragmentation
            - LAMBDA_DRIFT * affect.ema_debt,
            0, 1
        )

        # 4. Integrate affect (slow EMA)
        affect.v = (1 - GAMMA_V) * affect.v + GAMMA_V * v_hat
        affect.r = (1 - GAMMA_R) * affect.r + GAMMA_R * r_hat
        affect.b = (1 - GAMMA_B) * affect.b + GAMMA_B * b_hat
```

---

## I. Parameter Summary

### Observable Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| α_T | 1.0 | Throughput → valence |
| α_D | 0.8 | Debt → negative valence |
| α_F | 0.6 | Fragmentation → negative valence |
| β_S | 0.5 | Surprise → arousal |
| β_D | 0.4 | Debt → arousal |
| β_F | 0.3 | Fragmentation → arousal |
| λ_iso | 0.3 | Isolation penalty on bondedness |
| λ_drift | 0.2 | Debt-drift penalty on bondedness |

### EMA Rates

| Parameter | Default | Half-Life | Description |
|-----------|---------|-----------|-------------|
| γ_short | 0.3 | ~2 ticks | Trace update rate |
| γ_v | 0.1 | ~7 ticks | Valence smoothing |
| γ_r | 0.3 | ~2 ticks | Arousal smoothing |
| γ_b | 0.05 | ~14 ticks | Bondedness smoothing |

### Modulation Gains

| Parameter | Default | Description |
|-----------|---------|-------------|
| k_r | 0.5 | Arousal → escalation boost |
| k_neg | 0.3 | Negative valence → escalation boost |
| k_bond | 0.4 | Low bondedness → escalation boost |
| η_r | 0.3 | Arousal → plasticity boost |
| η_seek | 0.4 | Attachment-seeking → plasticity boost |
| η_neg | 0.3 | Negative valence → decay boost |

---

## J. Integration with Existing Systems

### J.1 Cross-Layer Dynamics (Exploration 06)

Replace fixed thresholds with affect-modulated versions:

```python
# In check_escalation():
effective_N_threshold = N_THRESHOLD / escalation_multiplier(j)

# In coherence update:
effective_alpha_PA = ALPHA_PA * plasticity_scale_PA(i)
effective_lambda_PA = LAMBDA_PA * decay_scale_PA(i)
```

### J.2 Gatekeeper (Part 2.4)

Add affect check to gatekeeper:

```c
// Reject if in "prison" emotional state (high C, low a, low bondedness)
ClusterAffect affect = compute_cluster_affect(core, self_cluster);
if (affect.bondedness < 0.2f && affect.arousal < 0.2f && affect.valence < -0.5f) {
    // Apathy/depression state - need recovery, not new tasks
    schedule_recovery(core);
    return DECISION_STOP;
}
```

### J.3 User Interface

Display current emotional state:

```
DET State: [Flow] v=+0.6 r=0.7 b=0.8
```

---

## K. Summary

| Aspect | Resolution |
|--------|------------|
| **Model** | 3-axis: Valence, Arousal, Bondedness |
| **Computation** | Local observables → EMA traces → affect targets → smoothed affect |
| **Modulation** | Escalation threshold, plasticity scale, decay scale |
| **No violations** | No direct a_i edits, all local operations |
| **Self-level** | Participation-weighted average over Self cluster |
| **User-facing** | Maps to discrete emotional labels |

---

## L. Open Sub-Questions (Deferred)

1. **Affective memory**: Should nodes remember past emotional states?
2. **Contagion**: Should affect spread through bonds?
3. **Regulation goals**: Should the system actively seek homeostasis?
4. **External signals**: Can user input directly inject affect?

These can be explored during Phase 4 implementation.
