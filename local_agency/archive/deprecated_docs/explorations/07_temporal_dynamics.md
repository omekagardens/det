# Exploration 07: Temporal Dynamics (Multi-Timescale Locality)

**Status**: RESOLVED
**Date**: 2026-01-17

---

## Core Principle

Implement multi-timescale behavior using:
1. **Per-node local cadence counters**
2. **Short/long EMAs** over local flux, debt, and coherence
3. **Windowed membrane events**

All quantities are **local state**, consistent with DET locality and inviolable agency.

---

## A. Layer Update Frequencies

| Layer | Update Frequency | Rationale |
|-------|-----------------|-----------|
| **A-layer** | Every tick | Fast, reactive, high plasticity |
| **P-layer** | Every tick | Deliberate, but still real-time |
| **P↔A membrane** | Windowed | Escalation: fast interrupt; Compilation: slow commit |
| **Dormant pool** | On recruitment only | No active updates when dormant |

The key insight: **all layers update every tick**, but their *response characteristics* differ via EMA time constants and event windows.

---

## B. Per-Node Cadence Counters

Each node maintains local timing state:

```c
typedef struct {
    uint32_t tick;              // Global tick count (for reference)
    uint32_t last_active_tick;  // When this node last had significant flux
    uint32_t quiet_ticks;       // Consecutive ticks with low activity

    // Local cadence for windowed operations
    uint16_t membrane_window;   // Ticks between membrane evaluations
    uint16_t window_counter;    // Counts up to membrane_window
} NodeCadence;
```

### Quiet Regime Acceleration

When a node has been quiet (low flux) for extended periods, membrane operations can run more frequently:

```c
uint16_t compute_membrane_window(NodeState* node, NodeCadence* cadence) {
    // Base window (e.g., 100 ticks for P→A compilation checks)
    uint16_t base_window = COMPILATION_WINDOW_BASE;

    // Accelerate in quiet regimes (more frequent checks when idle)
    if (cadence->quiet_ticks > QUIET_THRESHOLD) {
        float acceleration = fminf(
            (float)cadence->quiet_ticks / QUIET_THRESHOLD,
            MAX_ACCELERATION
        );
        return (uint16_t)(base_window / acceleration);
    }

    return base_window;
}
```

---

## C. Dual-EMA Architecture

Each node/bond maintains two EMA accumulators per tracked quantity:

```c
typedef struct {
    // Short EMA: fast response (escalation, novelty detection)
    float flux_short;       // α_short ≈ 0.3
    float debt_short;
    float coherence_short;

    // Long EMA: slow response (compilation, stability detection)
    float flux_long;        // α_long ≈ 0.05
    float debt_long;
    float coherence_long;
} DualEMA;

void update_dual_ema(DualEMA* ema, float flux, float debt, float coherence) {
    // Short EMA (fast)
    ema->flux_short = ALPHA_SHORT * flux + (1.0f - ALPHA_SHORT) * ema->flux_short;
    ema->debt_short = ALPHA_SHORT * debt + (1.0f - ALPHA_SHORT) * ema->debt_short;
    ema->coherence_short = ALPHA_SHORT * coherence + (1.0f - ALPHA_SHORT) * ema->coherence_short;

    // Long EMA (slow)
    ema->flux_long = ALPHA_LONG * flux + (1.0f - ALPHA_LONG) * ema->flux_long;
    ema->debt_long = ALPHA_LONG * debt + (1.0f - ALPHA_LONG) * ema->debt_long;
    ema->coherence_long = ALPHA_LONG * coherence + (1.0f - ALPHA_LONG) * ema->coherence_long;
}
```

### EMA Time Constants

| EMA Type | α | Half-Life (ticks) | Purpose |
|----------|---|-------------------|---------|
| **Short** | 0.3 | ~2 | Novelty detection, fast interrupt |
| **Long** | 0.05 | ~14 | Stability detection, slow commit |
| **Very Long** | 0.01 | ~69 | Sleep/consolidation (optional) |

Half-life formula: `t_half = -ln(2) / ln(1-α)`

---

## D. Escalation: Fast Interrupt (A→P)

Escalation uses **short EMAs** and triggers immediately when threshold crossed:

```c
typedef struct {
    DualEMA ema;
    float novelty_score;      // Computed from short EMAs
    bool escalation_pending;  // Flag for immediate processing
} ANodeState;

void check_escalation(DETCore* core, uint16_t j) {
    ANodeState* a_node = &core->a_nodes[j];

    // Compute novelty from SHORT EMAs (fast response)
    float delta_flux = fabsf(core->nodes[j].current_flux - a_node->ema.flux_short);
    float rising_debt = fmaxf(0, core->nodes[j].q - a_node->ema.debt_short);
    float fragmentation = 1.0f - a_node->ema.coherence_short;

    a_node->novelty_score =
        delta_flux +
        ETA_Q * rising_debt +
        ETA_S * fragmentation;

    // Immediate interrupt when threshold crossed
    if (a_node->novelty_score > N_THRESHOLD && core->nodes[j].a > 0) {
        a_node->escalation_pending = true;
        // Process immediately in same tick
        execute_escalation(core, j);
    }
}
```

### Escalation Timing

- **Evaluation**: Every tick (using short EMA)
- **Trigger**: Immediate when `N_j > threshold`
- **Action**: Resource spotlight + attention bonds (same tick)
- **Latency**: 0-1 ticks from novelty spike to P-layer attention

---

## E. Compilation: Slow Commit (P→A)

Compilation uses **long EMAs** and evaluates on a **windowed schedule**:

```c
typedef struct {
    DualEMA ema;
    NodeCadence cadence;
    float stability_scores[MAX_CROSS_BONDS];  // Per incident P↔A bond
} PNodeState;

void check_compilation(DETCore* core, uint16_t i) {
    PNodeState* p_node = &core->p_nodes[i];

    // Only evaluate on window boundary
    p_node->cadence.window_counter++;
    if (p_node->cadence.window_counter < p_node->cadence.membrane_window) {
        return;  // Not time yet
    }
    p_node->cadence.window_counter = 0;

    // Update window for next cycle (quiet acceleration)
    p_node->cadence.membrane_window = compute_membrane_window(
        &core->nodes[i], &p_node->cadence
    );

    // Compute stability from LONG EMAs (slow response)
    uint32_t stable_count = 0;
    for (uint32_t b = 0; b < num_cross_bonds(core, i); b++) {
        BondState* bond = get_cross_bond(core, i, b);

        float stability =
            bond->ema.flux_long *
            bond->ema.coherence_long *
            bond->phase_align_ema;  // cos(θ_i - θ_j)

        p_node->stability_scores[b] = stability;

        if (stability > B_THRESHOLD) {
            stable_count++;
        }
    }

    // Commit only when enough bonds are stable
    if (stable_count >= K_MIN_COMPILE) {
        execute_compilation(core, i, p_node->stability_scores);
    }
}
```

### Compilation Timing

- **Evaluation**: Windowed (every `membrane_window` ticks)
- **Base window**: ~100 ticks
- **Quiet acceleration**: Down to ~25 ticks when locally idle
- **Trigger**: When `k_min` bonds exceed stability threshold
- **Action**: Strengthen A↔A, decay P↔A (spread over multiple ticks if needed)

---

## F. Coherence Decay Rates

Different decay rates for different bond types, all computed locally:

```c
typedef struct {
    float base_decay;       // λ_base
    float activity_bonus;   // Reduce decay when active
    float phase_slip_rate;  // λ_slip for incoherent bonds
} DecayParams;

static const DecayParams DECAY_AA = {0.08f, 0.5f, 0.02f};  // Fast decay, activity helps
static const DecayParams DECAY_PP = {0.01f, 0.8f, 0.01f};  // Slow decay, stable
static const DecayParams DECAY_PA = {0.04f, 0.3f, 0.06f};  // Medium decay, phase-sensitive

float compute_effective_decay(BondState* bond, DecayParams* params) {
    // Base decay reduced by recent activity
    float activity_factor = 1.0f - params->activity_bonus * bond->ema.flux_short;

    // Phase slip adds to decay for misaligned bonds
    float phase_slip = params->phase_slip_rate * (1.0f - bond->phase_align_ema);

    return params->base_decay * activity_factor + phase_slip;
}
```

### Decay Half-Lives

| Bond Type | Active Half-Life | Inactive Half-Life |
|-----------|-----------------|-------------------|
| A↔A | ~8 ticks | ~2 ticks |
| P↔P | ~70 ticks | ~40 ticks |
| P↔A | ~25 ticks (aligned) | ~5 ticks (misaligned) |

---

## G. Sleep/Consolidation Window

Optional extension for explicit "sleep" periods:

```c
typedef enum {
    MODE_ACTIVE,        // Normal operation
    MODE_CONSOLIDATION  // Sleep/consolidation mode
} CoreMode;

typedef struct {
    CoreMode mode;
    uint32_t consolidation_ticks;  // Remaining ticks in consolidation
    float consolidation_strength;  // Multiplier for compilation
} ConsolidationState;

void enter_consolidation(DETCore* core, uint32_t duration) {
    core->consolidation.mode = MODE_CONSOLIDATION;
    core->consolidation.consolidation_ticks = duration;
    core->consolidation.consolidation_strength = 2.0f;  // Faster compilation

    // During consolidation:
    // - No new input from LLM layer
    // - Escalation disabled
    // - Compilation accelerated (shorter windows, stronger effects)
    // - Anti-prison rules still active
}
```

This maps to the "sleep" periods when MLX retraining occurs.

---

## H. Complete Per-Tick Update Loop

```python
def tick(core: DETCore, dt: float):
    """
    Single tick update with multi-timescale dynamics.
    """

    # 1. Update all node fluxes, phases, presence (every tick)
    for node in core.all_nodes:
        update_node_dynamics(node, dt)

    # 2. Update all bond coherence (every tick, layer-specific params)
    for bond in core.all_bonds:
        params = get_bond_params(bond)
        update_coherence(bond, params, dt)

    # 3. Update dual EMAs (every tick)
    for node in core.all_nodes:
        node.ema.update(node.flux, node.q, node.local_coherence)
    for bond in core.all_bonds:
        bond.ema.update(bond.flux, 0, bond.C)  # bonds track flux and C

    # 4. A-layer: check escalation (every tick, fast interrupt)
    for a_node in core.a_nodes:
        check_escalation(core, a_node)  # Uses short EMA

    # 5. P-layer: check compilation (windowed, slow commit)
    for p_node in core.p_nodes:
        check_compilation(core, p_node)  # Uses long EMA, respects window

    # 6. Update cadence counters
    for node in core.all_nodes:
        update_cadence(node)

    # 7. Apply anti-prison rules (every tick)
    for bond in core.cross_layer_bonds:
        apply_anti_prison_rules(core, bond)
```

---

## I. Parameter Summary

### EMA Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| α_short | 0.3 | Short EMA decay (escalation) |
| α_long | 0.05 | Long EMA decay (compilation) |
| α_very_long | 0.01 | Very long EMA (consolidation, optional) |

### Window Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| COMPILATION_WINDOW_BASE | 100 | Base ticks between compilation checks |
| QUIET_THRESHOLD | 50 | Ticks of quiet before acceleration |
| MAX_ACCELERATION | 4.0 | Maximum window acceleration factor |
| MIN_WINDOW | 25 | Minimum window size (ticks) |

### Decay Parameters

| Parameter | A↔A | P↔P | P↔A |
|-----------|-----|-----|-----|
| λ_base | 0.08 | 0.01 | 0.04 |
| activity_bonus | 0.5 | 0.8 | 0.3 |
| λ_slip | 0.02 | 0.01 | 0.06 |

---

## J. Answers to Original Sub-Questions

### Q: How long does P→A compilation take?

**Answer**: Compilation is evaluated every `membrane_window` ticks (default 100, accelerated to 25 in quiet regimes). A single compilation event executes immediately once `k_min` bonds exceed stability threshold. The *decision* to compile requires stability over the long EMA window (~14 tick half-life), so effective compilation latency is 50-200 ticks from pattern stabilization.

### Q: What's the coherence decay rate for inactive bonds?

**Answer**: Layer-dependent:
- A↔A: ~2 tick half-life when inactive
- P↔P: ~40 tick half-life when inactive
- P↔A: ~5 tick half-life when phase-misaligned, ~25 when aligned but inactive

### Q: How does "sleep" consolidation affect cluster structure?

**Answer**: Consolidation mode accelerates compilation (2x strength, shorter windows) while disabling new input and escalation. This allows:
- Faster P→A transfer of stable patterns
- Stronger A↔A bond formation within compiled clusters
- More aggressive decay of unused P↔A bridges
- Anti-prison rules remain active to prevent pathological states

---

## K. Summary

| Aspect | Resolution |
|--------|------------|
| **Update frequency** | All layers every tick; membrane events windowed |
| **Fast response (escalation)** | Short EMA (α=0.3), immediate interrupt |
| **Slow response (compilation)** | Long EMA (α=0.05), windowed evaluation |
| **Cadence** | Per-node local counters, quiet acceleration |
| **Decay** | Layer-specific rates, activity-modulated |
| **Consolidation** | Optional sleep mode: accelerated compilation, disabled input |
| **Locality** | All quantities are local state |

---

## L. Open Sub-Questions (Deferred)

1. **Tick-to-wallclock mapping**: What's dt in real seconds?
2. **Consolidation triggers**: When should the system enter sleep mode?
3. **Cross-session persistence**: How do EMAs survive session boundaries?
4. **Debugging timescales**: How to visualize multi-timescale dynamics?

These can be explored during implementation.
