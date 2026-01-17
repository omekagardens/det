# Exploration 05: LLM-to-DET Interface (Port + Stimulus + Recruitment)

**Status**: RESOLVED
**Date**: 2026-01-17

---

## The Core Principle

LLM output must map to:
1. A **request packet** (discrete intent + domain + constraints), and
2. A **stimulus field** (soft activation cues)

...delivered only through **local interface bonds** to a small "sensory membrane" of DET nodes.

**LLM must NOT**:
- Directly set `a_i`, `q_i`, `C_ij`, or "self-cluster membership"
- Inject global normalizations or global control signals

This preserves the **inverted paradigm**: DET is the mind, LLM is the tool/memory.

---

## A. The Interface Object Model

### A.1 DetIntentPacket (Phase 1 Ready)

Minimal schema that the LLM emits:

```json
{
  "domain": "math|language|tool_use|science|...",
  "intent": "answer|plan|execute|learn|summarize|reflect|debug|...",
  "complexity": 0.0-1.0,
  "risk": 0.0-1.0,
  "tool_suggestions": ["bash", "mlx_train", "search_docs", "..."],
  "constraints": ["no_network", "read_only", "..."],
  "tags": ["algebra", "integration", "swiftui", "..."],
  "text": "cleaned user request"
}
```

This is **intent classification formalized as a contract**.

### A.2 StimulusVector (Later Phases)

Optional payload for richer activation:
- An embedding vector (from the LLM), or
- A quantized code (VQ) index list

**Key**: The DET core doesn't "understand embeddings"; it only accepts local stimuli.

---

## B. The Sensory Membrane

Create a small set of **interface nodes** in the DET graph (16–64 nodes), each representing a "port":

```
PORT NODES (Sensory Membrane)
├── PORT_INTENT_answer
├── PORT_INTENT_plan
├── PORT_INTENT_execute
├── PORT_INTENT_learn
├── PORT_INTENT_summarize
├── PORT_INTENT_reflect
├── PORT_INTENT_debug
│
├── PORT_DOMAIN_math
├── PORT_DOMAIN_language
├── PORT_DOMAIN_tool_use
├── PORT_DOMAIN_science
│
├── PORT_RISK          (continuous 0-1)
├── PORT_COMPLEXITY    (continuous 0-1)
│
└── PORT_TAG_* (top-k tags, optional)
```

**These nodes are NOT the Self** — they are like sensory neurons at the boundary.

### B.1 The Only Allowed Coupling

When a request arrives:

1. **Inject resource F** into the relevant port nodes (local):
   - Higher for higher confidence, lower for uncertainty

2. **Open temporary bonds** from ports to candidate A-layer clusters (also local):
   - Think of this as "spotlighting" parts of the automaticity layer

**Crucially**:
- We're not setting `C_ij` directly
- We're creating a **localized condition** where normal DET dynamics can strengthen bonds if coherent and useful

### B.2 Practical Mapping Rule (Phase 1)

```python
def packet_to_stimulus(packet: DetIntentPacket) -> PortStimulus:
    """Map LLM packet to port activations."""

    activations = {}

    # Intent activates a small fixed subset of port nodes
    activations[f"PORT_INTENT_{packet.intent}"] = 1.0

    # Domain activates the domain gateway nodes
    activations[f"PORT_DOMAIN_{packet.domain}"] = 1.0

    # Complexity/risk set magnitude of injection
    activations["PORT_COMPLEXITY"] = packet.complexity
    activations["PORT_RISK"] = packet.risk

    # Optional: top-k tags
    for tag in packet.tags[:3]:  # limit to top 3
        activations[f"PORT_TAG_{tag}"] = 0.5

    return PortStimulus(activations)
```

This immediately plugs into the gatekeeper logic.

---

## C. Three Interface Levels (Evolution Path)

### Level 1 (Phase 1): Discrete Intent → Port Injection

```
LLM does: classify + tag + rewrite request into "det-friendly" text
DET does: gate, recruit, respond
```

| Pros | Cons |
|------|------|
| Simplest | Coarse routing |
| Robust | Less nuance |
| Debuggable | Limited expressiveness |

**This is the starting point.**

### Level 2 (Phase 2-3): Tag Vocabulary → Sparse Node Routing

Add a fixed **tag lexicon** (512–4096 tags).

Map tags to A-layer subdomains via static lookup:
- `algebra` → math cluster nodes
- `bash` → tool_use cluster nodes
- `swiftui` → language + tool_use intersection

This is a clean intermediate between hard-coded intent and learned embeddings.

### Level 3 (Phase 4+): Learned Projection (Embedding → Local Stimulus Field)

Train a small projector `g_φ` that maps an LLM embedding `e` to a sparse activation over port/gateway nodes:

```
s = g_φ(e)  →  ΔF_p ∝ s_p
```

**Locality safeguard**: The projector may only distribute activation across the membrane/gateway neighborhood; no direct writes deeper in the graph.

**Training signal** — use DET outcomes:
- Did the gatekeeper accept?
- Did coherence in the target domain increase?
- Did the task complete with low debt growth?

The mapping learns what the DET core **actually benefits from**, not what "sounds right."

---

## D. The Drop-In Interface Algorithm

```python
def process_llm_request(
    det_core: DETCore,
    llm: LLMLayer,
    user_input: str,
    max_retries: int = 5
) -> Response:
    """
    Main interface loop between LLM and DET core.
    """

    for retry in range(max_retries):
        # 1. LLM → IntentPacket
        packet = llm.classify_and_package(user_input, retry_hint=retry)

        # 2. Packet → PortStimulus
        stimulus = packet_to_stimulus(packet)

        # 3. Inject F into port nodes
        for port_id, activation in stimulus.activations.items():
            port_node = det_core.get_port_node(port_id)
            port_node.F += ETA * activation

        # 4. Create temporary bonds from ports to candidate gateways
        #    (domain cluster heads) with low initial C
        temp_bonds = det_core.create_temporary_interface_bonds(
            stimulus.active_ports,
            target_domains=[packet.domain],
            initial_C=0.1
        )

        # 5. Run N DET steps (a short "attention window")
        for _ in range(ATTENTION_STEPS):
            det_core.step(dt=0.02)

        # 6. Run gatekeeper evaluation
        decision = det_core.evaluate_request(
            tokens=packet.to_tokens(),
            target_domain=packet.domain,
            retry_count=retry
        )

        # 7. Handle decision
        if decision == DET_PROCEED:
            # Execute the request
            result = execute_request(det_core, llm, packet)

            # Clean up temporary bonds (some may have strengthened)
            det_core.finalize_interface_bonds(temp_bonds)

            return result

        elif decision == DET_RETRY:
            # LLM reformulates packet; loop continues
            user_input = llm.reformulate_request(
                original=user_input,
                packet=packet,
                feedback="DET core requested simpler formulation"
            )
            det_core.cleanup_interface_bonds(temp_bonds)
            continue

        elif decision == DET_STOP:
            det_core.cleanup_interface_bonds(temp_bonds)
            return Response(
                success=False,
                message="Request declined by DET core (agency/resource constraints)"
            )

        elif decision == DET_ESCALATE:
            det_core.cleanup_interface_bonds(temp_bonds)
            return escalate_to_external_llm(user_input, packet)

    return Response(success=False, message="Max retries exceeded")
```

---

## E. C Kernel Data Structures

### Port Node Types

```c
typedef enum {
    PORT_TYPE_INTENT,      // answer, plan, execute, learn, etc.
    PORT_TYPE_DOMAIN,      // math, language, tool_use, science
    PORT_TYPE_CONTINUOUS,  // risk, complexity (0-1 values)
    PORT_TYPE_TAG          // optional semantic tags
} PortType;

typedef struct {
    uint16_t node_id;      // index in DET node array
    PortType type;
    char name[32];         // "intent_answer", "domain_math", etc.
    uint8_t target_domain; // which domain cluster this port feeds
} PortNode;
```

### Interface Bond Management

```c
typedef struct {
    uint16_t port_node;    // source (port)
    uint16_t target_node;  // destination (gateway/A-layer)
    float initial_C;       // starting coherence (low, e.g., 0.1)
    float current_C;       // current coherence after DET steps
    bool is_temporary;     // marked for cleanup after request
    uint32_t created_at;   // tick when created
} InterfaceBond;

#define MAX_INTERFACE_BONDS 256

typedef struct {
    PortNode ports[64];
    uint32_t num_ports;

    InterfaceBond interface_bonds[MAX_INTERFACE_BONDS];
    uint32_t num_interface_bonds;

    float eta;             // injection magnitude parameter
    uint32_t attention_steps; // DET steps per attention window
} InterfaceLayer;
```

### Stimulus Injection

```c
void inject_stimulus(
    DETCore* core,
    InterfaceLayer* interface,
    const PortStimulus* stimulus
) {
    for (uint32_t i = 0; i < stimulus->num_activations; i++) {
        uint16_t port_idx = stimulus->port_indices[i];
        float activation = stimulus->activations[i];

        // Get the actual DET node for this port
        uint16_t node_id = interface->ports[port_idx].node_id;

        // Inject resource (local operation)
        core->nodes[node_id].F += interface->eta * activation;
    }
}
```

---

## F. Key Insight

> **LLM outputs map to port-node resource injections and temporary local interface bonds that bias recruitment; the DET core's own dynamics decide which dormant/automatic nodes join the active cluster and what bonds strengthen.**

This is the **DET-native, non-coercive interface**.

---

## G. Summary: Resolution

| Aspect | Decision |
|--------|----------|
| **What LLM emits** | `DetIntentPacket` (structured, not thoughts) |
| **Where it goes** | Sensory membrane (16-64 port nodes) |
| **What it does** | Resource injection + temporary bonds |
| **What it can't do** | Directly set a, q, C, or cluster membership |
| **Evolution path** | Intent → Tags → Learned projection |
| **Training signal** | DET outcomes (acceptance, coherence, debt) |

---

## H. Open Sub-Questions (Deferred)

1. **Optimal port count**: 16, 32, or 64? Needs simulation.
2. **Attention window length**: How many DET steps per request?
3. **Bond finalization criteria**: When does a temporary bond become permanent?
4. **Embedding projector architecture**: MLP? Sparse attention? VQ?

These can be explored during implementation.
