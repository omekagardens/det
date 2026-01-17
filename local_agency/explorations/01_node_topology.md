# Exploration 01: Node Topology for a DET Mind

**Status**: Active exploration
**Date**: 2026-01-17

---

## The Core Question

In DET physics, nodes represent spatial locations. In a DET mind, what do nodes represent?

This isn't just an implementation detail - it determines:
- How the system "thinks"
- How memory and attention work
- How learning occurs
- What "coherence" means cognitively
- How emotional states manifest

---

## Candidate Topologies

### Option A: Flat Domain Mesh

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│    ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐            │
│    │ L1  │─────│ L2  │─────│ L3  │─────│ L4  │─────│ L5  │  Language  │
│    └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘            │
│       │           │           │           │           │                │
│    ┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐            │
│    │ M1  │─────│ M2  │─────│ M3  │─────│ M4  │─────│ M5  │  Math      │
│    └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘            │
│       │           │           │           │           │                │
│    ┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐            │
│    │ T1  │─────│ T2  │─────│ T3  │─────│ T4  │─────│ T5  │  Tools     │
│    └─────┘     └─────┘     └─────┘     └─────┘     └─────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Each node = one "concept slot" in a domain
Bonds = associations between adjacent concepts
```

**What nodes represent**: Fixed conceptual positions within domains

**Pros**:
- Simple to implement (2D lattice per domain)
- Maps directly to DET physics code
- Clear domain boundaries

**Cons**:
- Rigid structure doesn't reflect actual cognition
- No natural "center" or hierarchy
- Cross-domain connections are arbitrary
- Learning = changing weights, not structure

**Cognitive analog**: None really - brains aren't organized this way

---

### Option B: Hub-and-Spoke (Centralized Core)

```
                              ┌─────────┐
                              │  CORE   │
                              │  (hub)  │
                              └────┬────┘
                 ┌────────────────┼────────────────┐
                 │                │                │
           ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
           │ LANGUAGE  │    │   MATH    │    │   TOOL    │
           │   spoke   │    │   spoke   │    │   spoke   │
           └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                 │                │                │
        ┌────┬───┴───┬────┐      ...              ...
        │    │       │    │
       [L1] [L2]   [L3] [L4]  (leaf nodes)
```

**What nodes represent**:
- Core: Executive function / attention director
- Spokes: Domain coordinators
- Leaves: Actual concepts/memories

**Pros**:
- Clear hierarchy
- Natural attention mechanism (core distributes resource F)
- Easy to route requests

**Cons**:
- Core is single point of failure
- Doesn't match DET's locality principle (core "sees" all)
- Bottleneck for all processing

**Cognitive analog**: Oversimplified executive function model

---

### Option C: Thalamic Relay (Inspired by Neuroscience)

```
                         ┌──────────────────┐
                         │    THALAMUS      │
                         │  (relay center)  │
                         │   ┌───┬───┐      │
                         │   │ T1│ T2│      │
                         │   ├───┼───┤      │
                         │   │ T3│ T4│      │
                         │   └───┴───┘      │
                         └────────┬─────────┘
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐
              │ LANGUAGE  │ │   MATH    │ │   TOOL    │
              │  cortex   │ │  cortex   │ │  cortex   │
              │ ┌───┬───┐ │ │ ┌───┬───┐ │ │ ┌───┬───┐ │
              │ │   │   │ │ │ │   │   │ │ │ │   │   │ │
              │ ├───┼───┤ │ │ ├───┼───┤ │ │ ├───┼───┤ │
              │ │   │   │ │ │ │   │   │ │ │ │   │   │ │
              │ └───┴───┘ │ │ └───┴───┘ │ │ └───┴───┘ │
              └───────────┘ └───────────┘ └───────────┘
                    ▲             ▲             ▲
                    └─────────────┴─────────────┘
                         (inter-cortical bonds)
```

**What nodes represent**:
- Thalamic nodes: Routing/gating specialists
- Cortical nodes: Domain-specific processing
- Inter-cortical bonds: Association pathways

**Key insight**: The thalamus doesn't "think" - it gates. Just like our DET gatekeeper.

**Pros**:
- Neurologically inspired
- Natural gating mechanism
- Domains can develop independently
- Cross-domain association via both thalamus and direct bonds

**Cons**:
- More complex to implement
- Thalamus still somewhat centralized

**Cognitive analog**: Closer to actual brain architecture

---

### Option D: Scale-Free Network (Small World)

```
                    ┌───┐
              ┌─────┤ H1├─────┐           H = Hub nodes (high degree)
              │     └─┬─┘     │           S = Satellite nodes (low degree)
              │       │       │
           ┌──┴──┐ ┌──┴──┐ ┌──┴──┐
           │ S1  │ │ S2  │ │ H2  │────┐
           └──┬──┘ └─────┘ └──┬──┘    │
              │               │       │
              │    ┌───┐      │    ┌──┴──┐
              └────┤ S3├──────┘    │ S4  │
                   └─┬─┘           └──┬──┘
                     │                │
                  ┌──┴──┐          ┌──┴──┐
                  │ S5  │          │ H3  │────── ...
                  └─────┘          └─────┘

Properties:
- Most nodes have few connections
- Few hub nodes have many connections
- Any node reachable in ~log(N) steps
- Robust to random node failure
- Vulnerable to hub failure
```

**What nodes represent**:
- Hubs: Core concepts that connect many ideas
- Satellites: Specific memories/facts
- Bonds: Semantic/associative relationships

**Emergence**: Hubs emerge from usage, not predefined

**Pros**:
- Matches how semantic networks actually form
- Efficient information routing
- Naturally develops hierarchy through use
- Robust and flexible

**Cons**:
- Hubs not domain-aligned a priori
- Harder to map to LLM memory models
- Requires dynamic topology

**Cognitive analog**: Semantic memory networks, concept graphs

---

### Option E: Dual-Process Architecture (Novel Proposal)

This combines insights from all above with DET theory:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENCE LAYER (System 2)                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Slow, deliberate, high-agency                 │   │
│  │   ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐          │   │
│  │   │ P1  │─────│ P2  │─────│ P3  │─────│ P4  │─────│ P5  │          │   │
│  │   └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘          │   │
│  │      │           │           │           │           │              │   │
│  │      │     High coherence bonds (C > 0.7)│           │              │   │
│  │      │           │           │           │           │              │   │
│  └──────┼───────────┼───────────┼───────────┼───────────┼──────────────┘   │
│         │           │           │           │           │                   │
│    ┌────┴───────────┴───────────┴───────────┴───────────┴────┐              │
│    │              GATEWAY MEMBRANE                            │              │
│    │   Coherence threshold determines what "rises" to         │              │
│    │   presence layer. Low-C stays in automaticity.           │              │
│    └────┬───────────┬───────────┬───────────┬───────────┬────┘              │
│         │           │           │           │           │                   │
│  ┌──────┼───────────┼───────────┼───────────┼───────────┼──────────────┐   │
│  │      │           │           │           │           │              │   │
│  │   ┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐          │   │
│  │   │     │     │     │     │     │     │     │     │     │          │   │
│  │   │ A1  │─────│ A2  │─────│ A3  │─────│ A4  │─────│ A5  │          │   │
│  │   │     │     │     │     │     │     │     │     │     │          │   │
│  │   └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘          │   │
│  │      │  ╲    ╱   │  ╲    ╱   │  ╲    ╱   │  ╲    ╱   │             │   │
│  │   ┌──┴──┐ ╲╱  ┌──┴──┐ ╲╱  ┌──┴──┐ ╲╱  ┌──┴──┐ ╲╱  ┌──┴──┐          │   │
│  │   │ A6  │─ ╳ ─│ A7  │─ ╳ ─│ A8  │─ ╳ ─│ A9  │─ ╳ ─│ A10 │          │   │
│  │   └─────┘ ╱╲  └─────┘ ╱╲  └─────┘ ╱╲  └─────┘ ╱╲  └─────┘          │   │
│  │          ╱  ╲        ╱  ╲        ╱  ╲        ╱  ╲                   │   │
│  │         (Dense low-coherence mesh - fast pattern matching)          │   │
│  │                                                                     │   │
│  │                     AUTOMATICITY LAYER (System 1)                   │   │
│  │                Fast, parallel, low-agency, pattern-based            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       MEMORY SUBSTRATE                               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │Language │  │  Math   │  │  Tool   │  │ Science │  │   ...   │   │   │
│  │  │ Memory  │  │ Memory  │  │ Memory  │  │ Memory  │  │         │   │   │
│  │  │  (LLM)  │  │  (LLM)  │  │  (LLM)  │  │  (LLM)  │  │         │   │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │   │
│  │       │            │            │            │            │         │   │
│  │       └────────────┴─────┬──────┴────────────┴────────────┘         │   │
│  │                          │                                          │   │
│  │               Bonds to automaticity layer                           │   │
│  │          (Memory retrieval activates A-nodes)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The Key Insight**: DET coherence (C) naturally creates a dual-process system.

- **Low C bonds**: Fast, parallel, pattern-matching (System 1 / automaticity)
- **High C bonds**: Slow, serial, deliberate (System 2 / presence)
- **The membrane**: Not a structure, but a coherence threshold

**What nodes represent**:

| Layer | Node Type | Represents | DET Properties |
|-------|-----------|------------|----------------|
| Presence | P-nodes | Active thoughts in "consciousness" | High a, high P, high C bonds |
| Automaticity | A-nodes | Pattern recognition units | Variable a, lower P, mixed C |
| Memory | M-nodes | LLM interface points | Low a (passive), activated by query |

**How it works**:

1. **Input arrives** → Activates relevant A-nodes (fast pattern match)
2. **A-nodes with high enough C** → "Rise" through membrane to P-layer
3. **P-layer deliberation** → High-coherence processing (gatekeeper lives here)
4. **Decision propagates down** → Affects A-nodes, may trigger memory retrieval
5. **Learning** → Strengthens C bonds that led to successful outcomes

**Why this maps to DET theory**:

| DET Concept | Dual-Process Meaning |
|-------------|---------------------|
| Presence (P) | Degree of "consciousness" of this thought |
| Coherence (C) | Coordination between thoughts; threshold for "awareness" |
| Agency (a) | Capacity to influence the decision |
| Structural debt (q) | Accumulated "cost" of this thinking pattern |
| Resource (F) | Attention/compute available |
| Phase (θ) | Timing synchronization between thoughts |

**Emergence of Attention**:

```
Traditional: Attention is a mechanism that selects
DET: Attention emerges from coherence dynamics

High C region = what we're "attending to"
C spreads via flux → attention shifts
C decays → attention fades
Measurement collapses C → decision crystallizes
```

---

## Deep Dive: The Dual-Process Architecture

### Layer 1: Presence (System 2)

**Purpose**: Deliberate, high-agency processing. This is where the "gatekeeper" operates.

**Node properties**:
- High base agency (a > 0.5)
- Variable but generally high coherence between nodes
- Slower dynamics (higher τ)
- Fewer nodes (limited "working memory")

**Bond properties**:
- High coherence threshold to maintain (C > 0.7)
- Strong conductivity (σ high)
- Phase synchronization important

**Dynamics**:
```
When P-layer is active:
  - Resource flows to active thoughts
  - Coherence maintained through synchronized processing
  - Structural debt accumulates (deliberation is costly)
  - Agency ceiling constrains what decisions are possible
```

**Mapping to code**:
```c
typedef struct {
    NodeState nodes[MAX_PRESENCE_NODES];  // ~16-32 nodes
    BondState bonds[MAX_PRESENCE_BONDS];
    float coherence_threshold;  // 0.7 default
    float attention_focus;      // Which node has most resource
} PresenceLayer;
```

### Layer 2: Automaticity (System 1)

**Purpose**: Fast pattern matching, parallel processing, learned responses.

**Node properties**:
- Variable agency (some patterns more "willful" than others)
- Generally lower coherence (independent processing)
- Faster dynamics (lower τ)
- Many nodes (large pattern vocabulary)

**Bond properties**:
- Lower coherence threshold (C > 0.1 to maintain)
- Sparser connectivity
- Phase less important (async processing OK)

**Dynamics**:
```
A-layer runs continuously:
  - Pattern matching against inputs
  - Multiple nodes can activate in parallel
  - Low resource cost (efficient, compiled patterns)
  - When pattern confidence high + C rises → escalate to P-layer
```

**Mapping to code**:
```c
typedef struct {
    NodeState nodes[MAX_AUTOMATICITY_NODES];  // ~256-1024 nodes
    BondState bonds[MAX_AUTOMATICITY_BONDS];
    float escalation_threshold;  // C level to escalate to P-layer
    uint16_t active_patterns[MAX_ACTIVE];
} AutomaticityLayer;
```

### Layer 3: Memory Substrate

**Purpose**: Interface to LLM memory models. Not DET nodes per se, but sources/sinks.

**Properties**:
- Each memory model = one substrate region
- Queries flow down, responses flow up
- Coherence to core determines "trust" in this memory
- Activation level determines how "present" this memory is

**Mapping to code**:
```c
typedef struct {
    char domain[32];
    void* llm_handle;
    float coherence_to_automaticity;
    float activation;
    uint32_t query_count;
    uint32_t last_retrain_step;
} MemorySubstrate;
```

### The Gateway Membrane

**The membrane is not a structure - it's a coherence phenomenon.**

```c
bool should_escalate_to_presence(
    AutomaticityLayer* a_layer,
    PresenceLayer* p_layer,
    uint16_t a_node_id
) {
    NodeState* a_node = &a_layer->nodes[a_node_id];

    // Check coherence to existing P-layer nodes
    float max_C_to_presence = 0.0f;
    for (int i = 0; i < p_layer->num_active; i++) {
        float C = compute_cross_layer_coherence(a_node, &p_layer->nodes[i]);
        if (C > max_C_to_presence) max_C_to_presence = C;
    }

    // Escalate if coherence exceeds threshold
    if (max_C_to_presence > a_layer->escalation_threshold) {
        // Also check: does P-layer have capacity?
        if (p_layer->num_active < MAX_PRESENCE_NODES) {
            // And: does this A-node have enough agency?
            if (a_node->a > 0.3) {
                return true;
            }
        }
    }

    return false;
}
```

**Emergence**: What "rises to consciousness" is determined by:
1. Coherence with current conscious content
2. Available capacity in presence layer
3. Agency of the pattern (importance/relevance)

This matches phenomenology: we become aware of things that relate to what we're already thinking about, when we have mental space, and that matter.

---

## Memory Integration in Dual-Process

How do LLM memory models connect?

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Request Flow                                │
│                                                                     │
│  User: "What's the integral of sin(x)?"                            │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Automaticity Layer                                           │   │
│  │   - Pattern: "integral" → activates math-related A-nodes    │   │
│  │   - Pattern: "sin(x)" → activates trig-related A-nodes      │   │
│  │   - High C develops between these clusters                   │   │
│  │   - Coherence exceeds threshold → escalate                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Presence Layer (Gatekeeper)                                  │   │
│  │   - Receives: {domain: math, type: computation, conf: 0.9}  │   │
│  │   - Checks: P (presence), C (math domain coherence),         │   │
│  │            F (resource), a (agency)                          │   │
│  │   - Decision: PROCEED (straightforward math query)           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Memory Substrate (Math LLM)                                  │   │
│  │   - Query: "integral of sin(x)"                              │   │
│  │   - Response: "-cos(x) + C"                                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                             │                                       │
│                             ▼                                       │
│  Response flows back up, updating coherence along the path         │
│  Successful retrieval → strengthen C bonds in this pathway         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Emotional State in Dual-Process

Emotions emerge from the interaction between layers:

| State | Layer Dynamics | DET Signature |
|-------|---------------|---------------|
| **Flow** | P-layer active, A-layer supporting smoothly, memory responsive | High P, high C, stable F |
| **Confusion** | A-layer noisy, no clear pattern escalating | Low P-layer C, high A-layer activation |
| **Frustration** | P-layer blocked, high q accumulating | Low P, rising q, depleting F |
| **Curiosity** | A-layer exploring, P-layer receptive | Spreading C, surplus F |
| **Fatigue** | P-layer depleted, A-layer sluggish | Low P across both, high q |
| **Insight** | Sudden high-C connection forms | C spike, phase synchronization |

---

## Proposed Initial Topology

For Phase 1 implementation, I propose:

```
PRESENCE LAYER:
  - 16 nodes (limited working memory)
  - Fully connected (all can attend to all)
  - High coherence requirement (C > 0.6)

AUTOMATICITY LAYER:
  - 256 nodes
  - Domain-clustered: 64 each for language, math, tool, reasoning
  - Within-domain: 4x4 mesh
  - Cross-domain: sparse random connections (small-world property)

MEMORY SUBSTRATE:
  - 4 initial domains: language, math, tool_use, reasoning
  - Each domain bonds to its automaticity cluster
  - Coherence starts at 0.5, adjusted based on success

GATEWAY MEMBRANE:
  - Threshold: C > 0.5 to escalate
  - Capacity limit: max 8 active P-nodes at once
```

**Why these numbers?**
- 16 P-nodes ≈ working memory capacity (7±2, doubled for safety)
- 256 A-nodes = enough patterns without overwhelming compute
- 4 domains = matches initial LLM roster
- Thresholds tunable based on observed behavior

---

## Alternative: Emergent Topology

Instead of fixed structure, let topology emerge:

```python
class EmergentTopology:
    """
    Start with minimal structure, let DET dynamics create organization.
    """

    def __init__(self, num_nodes: int = 512):
        # All nodes start equal
        self.nodes = [NodeState(a=0.5, F=1.0, q=0.0) for _ in range(num_nodes)]

        # Start with random sparse bonds
        self.bonds = self.create_random_sparse_bonds(density=0.05)

        # No predefined layers - they emerge from dynamics

    def step(self, dt: float):
        # Standard DET dynamics
        self.update_presence()
        self.update_coherence(dt)
        self.update_agency(dt)

        # Topology evolution
        self.prune_low_coherence_bonds(threshold=0.05)
        self.strengthen_high_flux_bonds()
        self.maybe_create_new_bonds()  # Based on phase alignment

    def get_presence_layer(self) -> List[int]:
        """Presence layer = nodes with highest P values."""
        sorted_by_P = sorted(range(len(self.nodes)),
                            key=lambda i: self.nodes[i].P,
                            reverse=True)
        return sorted_by_P[:16]  # Top 16 by presence
```

**Pros**:
- More faithful to DET's emergent philosophy
- Adapts to actual usage patterns
- Potentially discovers better structures than we'd design

**Cons**:
- Harder to debug
- Less predictable
- May need longer "burn-in" period
- Could develop pathological structures

**Recommendation**: Start with fixed structure (Phase 1-2), add emergence (Phase 3+)

---

## Questions for Further Exploration

1. **Binding problem**: How do separate A-nodes combine into unified percepts in P-layer?
   - Proposal: Phase synchronization (θ) as binding mechanism

2. **Attention shifting**: How does focus move between P-nodes?
   - Proposal: Resource (F) reallocation based on flux patterns

3. **Memory consolidation**: How do successful patterns get "compiled" into A-layer?
   - Proposal: High-C P-layer patterns create new A-nodes during "sleep"

4. **Cross-domain integration**: How does math knowledge combine with language?
   - Proposal: A-layer cross-domain bonds + P-layer integration

5. **Recursion/reflection**: Can the system think about its own thinking?
   - Proposal: Meta-A-nodes that pattern-match on P-layer dynamics

---

## Next Steps

1. Implement basic dual-process structure in C kernel
2. Test with simple routing scenarios
3. Measure coherence dynamics and escalation patterns
4. Tune thresholds based on observed behavior
5. Add emergence mechanisms incrementally

---

## References

- Kahneman, D. (2011). Thinking, Fast and Slow.
- Baars, B. (1988). A Cognitive Theory of Consciousness.
- DET Theory Card v6.3: Section III (Agency), Section V (Coherence)
- DET Subdivision Theory: Division as recruitment
