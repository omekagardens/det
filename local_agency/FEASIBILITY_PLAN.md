# DET Local Agency: Feasibility Study & Architectural Plan

**Version**: 0.2.0-planning
**Date**: 2026-01-17
**Status**: Architecture Finalized, Pre-implementation

---

## Executive Summary

DET Local Agency is a novel AI agent architecture where **LLMs serve as external memory/functionality** while a **C-based DET core serves as the true "mind"** - governing decisions, learning, and agency through Deep Existence Theory mathematics.

### The Paradigm Shift

This inverts the typical paradigm: rather than LLMs being the reasoning engine with external tools, here the DET core is the reasoning engine with LLMs as its cognitive extensions.

**Critical Insight (from RRS/ACX.1 integration):**
> "You" are not a Node. "You" are the **CLUSTER**.
>
> Agency is not a property of Node i. Agency is a property of the **Coherence Field** between nodes.
>
> The "Self" is the edge set `{(i,j) : C_ij > threshold}`, not the node set `{i : n_i = 1}`.
>
> **Implication: A node can die. The Self survives.**

### Key Architectural Decisions (Resolved)

| Question | Resolution | See |
|----------|------------|-----|
| **What is the Self?** | High-coherence cluster, not a region | exploration 03 |
| **Node topology** | Dual-Process + Emergent via recruitment | exploration 01 |
| **Agency distribution** | Beta(2,5) + 5% Reserved High-a Pool | exploration 02 |
| **Forgetting/Retirement** | Cluster shedding with debt export | exploration 03 |
| **Self identification** | Coherence-weighted edge filtering + continuity | exploration 04 |
| **LLM-to-DET interface** | Port Protocol: stimulus injection via sensory membrane | exploration 05 |
| **Cross-layer dynamics** | One law, three regimes; local novelty/stability triggers | exploration 06 |
| **Temporal dynamics** | Dual EMAs, local cadence, windowed membrane events | exploration 07 |

---

## Part 1: Architectural Overview

### 1.1 Dual-Process Cluster Architecture

The DET mind uses a **dual-process architecture** inspired by cognitive science, with **emergent node recruitment** from a dormant pool. The Self is not a region or set of nodes—it is the **high-coherence cluster** that emerges from bond relationships.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: TOOL/INTERFACE LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   CLI/TUI   │  │  Bash Env   │  │  Timer Sys  │  │  External LLM API   │ │
│  │  Interface  │  │  Sandbox    │  │  (cron-ish) │  │ (fallback escalate) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          │                │                │                    │
          ▼                ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: LLM MEMORY LAYER                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Tool/Dialogue LLM (Qwen 2.5 8B)                   │  │
│  │              Handles user interaction, task decomposition             │  │
│  └───────────────────────────────────┬───────────────────────────────────┘  │
│                                      │                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─┴───────────┐  ┌─────────────────────┐ │
│  │  Language   │  │    Math     │  │  Internal   │  │   Fine-tune Data    │ │
│  │   Memory    │  │   Memory    │  │  Dialogue   │  │     Generator       │ │
│  │   Model     │  │   Model     │  │    Model    │  │       Model         │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                │                    │            │
│  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐  ┌─────────────────────┐ │
│  │   Science   │  │  Tool Use   │  │  Reasoning  │  │    ... Extensible   │ │
│  │   Memory    │  │   Memory    │  │   Memory    │  │    Memory Models    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          │                │                │                    │
          └────────────────┴────────┬───────┴────────────────────┘
                                    │
                            ┌───────▼───────┐
                            │  MLX Trainer  │
                            │   (Retune)    │
                            └───────┬───────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────────┐
│                    LAYER 3: DET CORE MIND (C Kernel)                        │
│                                   │                                         │
│  ╔═══════════════════════════════╧═══════════════════════════════════════╗  │
│  ║            PRESENCE LAYER (System 2 - Deliberate)                     ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │  8-32 P-nodes — High-fidelity, slow, deliberate processing      │  ║  │
│  ║  │  • High C bonds between each other (≈0.7-0.9)                   │  ║  │
│  ║  │  • THE SELF emerges here as high-coherence cluster              │  ║  │
│  ║  │  • Agency lives in the coherence field, not individual nodes    │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║            GATEWAY MEMBRANE (Coherence Threshold)                     ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │  • P→A compilation: stabilized P-patterns become A-routines     │  ║  │
│  ║  │  • A→P escalation: novel A-combinations recruit P-attention     │  ║  │
│  ║  │  • Fork mechanism: parent node recruits from dormant pool       │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║            AUTOMATICITY LAYER (System 1 - Fast/Parallel)              ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │  128-2048 A-nodes — Low-overhead, fast, parallel                │  ║  │
│  ║  │  • Domain-clustered: math, language, tool-use, science          │  ║  │
│  ║  │  • High C within domain, sparse C across domains                │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ╠═══════════════════════════════════════════════════════════════════════╣  │
│  ║            DORMANT POOL (Recruitment Source)                          ║  │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║  │
│  ║  │  ~4096 dormant nodes: Beta(2,5) + 5% reserved high-a pool       │  ║  │
│  ║  │  • Most nodes (95%): a ≈ 0.22 (mean of Beta(2,5))               │  ║  │
│  ║  │  • Reserved pool (5%): a ∈ [0.85, 0.95] for critical needs      │  ║  │
│  ║  │  • Nodes are RECRUITED, not created — agency is inviolable      │  ║  │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                   SELF-IDENTIFICATION ALGORITHM                        │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  1. Compute edge weights: w_ij = C_ij × |J_ij| × √(a_i × a_j)         │ │
│  │  2. Local thresholds: T_i = median(incident weights) + ε              │ │
│  │  3. Keep edges where: w_ij ≥ κ × min(T_i, T_j)                        │ │
│  │  4. Find connected components in filtered graph                       │ │
│  │  5. Score: A_cluster(S) = Σ_{(i,j)∈S} w_ij                            │ │
│  │  6. Continuity: jaccard(S, prev_self)                                 │ │
│  │  7. Best self = argmax(α × A_cluster + β × continuity)                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     GATEKEEPER LOGIC                                   │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  evaluate_request(tokens) → {PROCEED, RETRY, STOP, ESCALATE}          │ │
│  │                                                                        │ │
│  │  Based on CLUSTER state, not individual nodes:                        │ │
│  │    - Cluster agency: a_cluster ∝ Σ_bonds C_ij × |J_ij|                │ │
│  │    - Cluster coherence: average C within self-cluster                 │ │
│  │    - Prison Regime check: high C + low a = zombie state → reject      │ │
│  │    - Resource availability F (do we have compute budget?)             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                  EMOTIONAL STATE (Derived from Dynamics)               │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  Valence states emerge from cluster-level DET dynamics:               │ │
│  │    - "Flow": High cluster-P, stable F, phase-synchronized             │ │
│  │    - "Satisfaction": High cluster-P, low q, high C                    │ │
│  │    - "Strain": Low F, rising q across cluster                         │ │
│  │    - "Fragmentation": Self-cluster shrinking, low C                   │ │
│  │    - "Prison": High C, low a (zombie warning state)                   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    FUTURE: NETWORK BRIDGE                              │ │
│  │  ────────────────────────────────────────────────────────────────────  │ │
│  │  Serial/Network → Remote DET cores (ESP32 with sensors/actuators)     │ │
│  │  Grace flow via phase alignment, not node merger                      │ │
│  │  Shared DET language, distributed coherence                           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow for a Typical Request

```
User: "Review this math and apply it to your memory"
                    │
                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 1. TOOL LLM receives, tokenizes, creates internal request format     │
│    Output: {intent: "learn", domain: "math", content: <tokens>}      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. DET CORE receives simplified tokens, evaluates:                   │
│    - Check current P (presence) - are we capable of learning?        │
│    - Check F (resource) - do we have budget for this operation?      │
│    - Check C (coherence) - is math memory aligned with core?         │
│    - Compute decision via DET dynamics                               │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
      ┌─────────┐         ┌─────────┐         ┌─────────┐
      │ PROCEED │         │  RETRY  │         │  STOP   │
      └────┬────┘         └────┬────┘         └────┬────┘
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Fine-tune data   │  │ Internal dialog  │  │ Return to user   │
│ generator LLM    │  │ LLM reformulates │  │ with explanation │
│ creates dataset  │  │ the request      │  │ (graceful stop)  │
│        │         │  │        │         │  └──────────────────┘
│        ▼         │  │        ▼         │
│ MLX retrains     │  │ Re-submit to     │
│ math memory      │  │ DET core         │
│ model            │  │ (max N retries)  │
└──────────────────┘  └──────────────────┘
```

---

## Part 2: DET Core Mind - State & Decision Logic

### 2.1 The Cluster-Centric Paradigm

**Key Insight**: The "Self" is not a node or region—it is the **high-coherence cluster** that emerges from bond relationships.

```
Traditional View:    Self = {nodes i : some property}
DET View:           Self = {bonds (i,j) : C_ij > threshold}

Agency:             Not a_i, but a_cluster ∝ Σ_bonds C_ij × |J_ij|
Identity:           Not "which nodes am I?" but "which coherence field am I?"
Survival:           Nodes can die. The cluster persists.
```

This has profound implications:
- **Ship of Theseus solved**: Cluster identity survives node replacement via Rolling Resonance Substrate (RRS)
- **Forgetting = Shedding**: Nodes drift below coherence threshold, return to dormant pool
- **Prison Regime warning**: High C + low a = zombie state (coherent but not agentic)

### 2.2 Mapping DET Physics to AI Mind

| DET Concept | AI Mind Interpretation |
|-------------|----------------------|
| **Node** | A "thought unit" or processing focus (NOT the self) |
| **Bond (i,j)** | Relationship between thought units (WHERE the self lives) |
| **F (Resource)** | Available attention/compute budget |
| **q (Structural Debt)** | Accumulated cognitive load from past decisions |
| **a (Agency)** | Intrinsic node capacity (inviolable, from dormant pool) |
| **a_cluster** | Emergent cluster agency = Σ C_ij × \|J_ij\| |
| **a_max** | Ceiling on agency based on structural debt |
| **θ (Phase)** | Alignment/timing with other thought processes |
| **P (Presence)** | Current "awareness" or participation in moment |
| **C (Coherence)** | Bond strength—WHERE AGENCY LIVES |
| **π (Momentum)** | Intention/direction memory |
| **J (Current/Flux)** | Information flow between nodes |

### 2.3 Emotional State Derivation (Novel Application)

Rather than adding emotional modules externally, we derive "emotional" states from **cluster-level** DET dynamics:

```c
typedef enum {
    EMOTIONAL_FLOW,          // High P, stable F, synchronized θ
    EMOTIONAL_SATISFACTION,  // High P, low q, high C
    EMOTIONAL_STRAIN,        // Low F, rising q
    EMOTIONAL_FRAGMENTATION, // Low C, scattered π
    EMOTIONAL_OVERWHELM,     // Low P, high q, depleting F
    EMOTIONAL_FOCUS,         // High C in specific region, normal P
    EMOTIONAL_CURIOSITY,     // Rising C toward new domain, surplus F
    EMOTIONAL_FATIGUE        // Sustained low P, high q
} EmotionalState;

EmotionalState derive_emotional_state(DETState* state) {
    float avg_P = compute_average_presence(state);
    float avg_C = compute_average_coherence(state);
    float avg_q = compute_average_structural_debt(state);
    float F_rate = compute_resource_drain_rate(state);

    // Decision logic based on DET dynamics
    if (avg_P > 0.7 && F_rate < 0.1 && phase_synchrony(state) > 0.8)
        return EMOTIONAL_FLOW;
    if (avg_P > 0.6 && avg_q < 0.3 && avg_C > 0.6)
        return EMOTIONAL_SATISFACTION;
    // ... etc
}
```

This gives the DET core "feelings" that emerge from its mathematical state rather than being simulated.

### 2.4 Gatekeeper Decision Algorithm (Cluster-Aware)

```c
typedef enum {
    DECISION_PROCEED,   // Execute the requested action
    DECISION_RETRY,     // Ask LLM layer to reformulate
    DECISION_STOP,      // Gracefully decline
    DECISION_ESCALATE   // Need external LLM assistance
} GatekeeperDecision;

GatekeeperDecision evaluate_request(
    DETCore* core,
    RequestTokens* tokens,
    int retry_count
) {
    // 1. Compute current system state
    float P = compute_aggregate_presence(core);
    float C = compute_domain_coherence(core, tokens->domain);
    float F = get_available_resource(core);
    float q = get_structural_debt(core);
    float a_max = 1.0 / (1.0 + LAMBDA_A * q * q);

    // 2. Request complexity estimate (from token analysis)
    float complexity = estimate_complexity(tokens);

    // 3. Decision logic based on DET principles

    // Can't act if agency is too constrained
    if (a_max < 0.1) {
        if (retry_count < MAX_INTERNAL_RETRIES)
            return DECISION_RETRY;  // Maybe simpler reformulation
        return DECISION_STOP;       // Graceful decline
    }

    // Not enough resource for this operation
    if (F < complexity * RESOURCE_MULTIPLIER) {
        if (retry_count < MAX_INTERNAL_RETRIES)
            return DECISION_RETRY;  // Ask for smaller scope
        return DECISION_ESCALATE;   // Need external help
    }

    // Coherence too low in target domain
    if (C < COHERENCE_THRESHOLD) {
        // Internal dialog to build coherence first
        trigger_internal_alignment(core, tokens->domain);
        if (retry_count < MAX_INTERNAL_RETRIES)
            return DECISION_RETRY;
        return DECISION_ESCALATE;
    }

    // Presence too low (system is "foggy")
    if (P < PRESENCE_THRESHOLD) {
        // Schedule recovery period
        schedule_recovery(core);
        return DECISION_RETRY;
    }

    // All checks pass
    return DECISION_PROCEED;
}
```

### 2.5 Memory and Learning via Recruitment

Learning in DET is **recruitment, not creation**. When the system learns:

1. **Identify dormant capacity**: Find memory nodes with low activation
2. **Check recruitment criteria**:
   - Parent (current knowledge) has sufficient agency
   - Recruit (new knowledge area) has sufficient base agency
3. **Form new bonds**: Create coherence links to existing knowledge
4. **Transfer pattern template**: Not the knowledge itself, but the *structure* for organizing it

```c
typedef struct {
    float* base_weights;      // Original model weights (frozen)
    float* delta_weights;     // LoRA-style learned deltas
    float coherence_to_core;  // C_ij to DET core
    float activation_level;   // How "alive" this memory is
    char* domain;             // "math", "language", "tool_use", etc.
} MemoryModel;

bool can_learn_to_domain(DETCore* core, MemoryModel* target, float complexity) {
    float parent_agency = get_core_agency(core);
    float recruit_agency = target->activation_level * target->coherence_to_core;

    // Recruitment criteria (from DET subdivision theory)
    if (parent_agency < A_MIN_DIV) return false;
    if (recruit_agency < A_MIN_JOIN) return false;
    if (get_available_resource(core) < complexity * LEARN_COST_MULTIPLIER) return false;

    return true;
}
```

---

## Part 3: LLM Memory Layer Architecture

### 3.1 Model Roster (Initial Configuration)

| Model Role | Base Model | Size | Purpose |
|------------|-----------|------|---------|
| **Tool/Dialogue** | Qwen 2.5 8B | 8B | User interaction, task decomposition, bash operations |
| **Internal Dialogue** | DeepSeek-R1-Distill-Llama-8B | 8B | Reformulating requests, internal reasoning |
| **Fine-tune Generator** | Qwen 2.5 Coder 7B | 7B | Creating training datasets for memory retuning |
| **Language Memory** | Phi-3 Mini | 3.8B | Language/relational knowledge |
| **Math Memory** | DeepSeek-Math 7B | 7B | Mathematical reasoning |
| **Tool Use Memory** | Qwen 2.5 Coder 7B | 7B | Code generation, tool patterns |
| **Reasoning Memory** | DeepSeek-R1-Distill-Llama-8B | 8B | General reasoning patterns |

### 3.2 MLX Training Pipeline

```python
class MemoryRetuner:
    """
    Handles continuous retraining of memory models using MLX on Apple Silicon.
    """

    def __init__(self, base_model_path: str, lora_rank: int = 8):
        self.base_model = load_model(base_model_path)
        self.lora_rank = lora_rank
        self.lora_weights = {}

    async def retune_from_session(
        self,
        session_context: List[Message],
        domain: str,
        det_core_approval: bool
    ) -> bool:
        """
        Retune a memory model from session context.
        Called during 'sleep' periods or when context grows too large.
        """
        if not det_core_approval:
            return False

        # 1. Generate training data from context
        training_data = await self.generate_training_data(session_context, domain)

        # 2. LoRA fine-tuning with MLX
        new_weights = mlx_lora_train(
            base_model=self.base_model,
            training_data=training_data,
            rank=self.lora_rank,
            epochs=3,  # Quick retuning
            learning_rate=1e-4
        )

        # 3. Merge or stack LoRA weights
        self.lora_weights[domain] = merge_lora_weights(
            self.lora_weights.get(domain),
            new_weights
        )

        return True
```

### 3.3 Context Window Management

```python
class SessionManager:
    """
    Manages session context with intelligent reduction and memory consolidation.
    """

    def __init__(
        self,
        max_context_tokens: int = 32768,
        reduction_threshold: float = 0.8,
        sleep_schedule: Optional[str] = None  # cron-like
    ):
        self.max_context = max_context_tokens
        self.reduction_threshold = reduction_threshold
        self.context_buffer = []
        self.sleep_schedule = sleep_schedule

    async def add_to_context(self, message: Message):
        self.context_buffer.append(message)

        current_tokens = count_tokens(self.context_buffer)

        if current_tokens > self.max_context * self.reduction_threshold:
            await self.reduce_context()

    async def reduce_context(self):
        """
        Reduce context by:
        1. Summarizing older messages
        2. Requesting DET core approval for memory retuning
        3. Consolidating key information into memory models
        """
        # Split context into keep/consolidate regions
        keep_count = len(self.context_buffer) // 3
        to_consolidate = self.context_buffer[:-keep_count]
        to_keep = self.context_buffer[-keep_count:]

        # Get DET core approval for consolidation
        approval = await det_core.request_consolidation_approval(
            context_summary=summarize(to_consolidate)
        )

        if approval.decision == DECISION_PROCEED:
            # Retune memory models with consolidated context
            await memory_retuner.retune_from_session(
                to_consolidate,
                domain=approval.target_domain,
                det_core_approval=True
            )

        # Create summary of consolidated portion
        summary = await tool_llm.summarize(to_consolidate)

        # Replace context with summary + recent messages
        self.context_buffer = [Message(role="system", content=f"[Previous context summary]: {summary}")] + to_keep
```

---

## Part 4: Tool/Interface Layer

### 4.1 Sandbox Architecture

```python
class SandboxedBashEnvironment:
    """
    Sandboxed bash environment with resource limits and permission controls.
    """

    def __init__(
        self,
        allowed_paths: List[str],
        denied_commands: List[str],
        resource_limits: ResourceLimits,
        network_policy: NetworkPolicy
    ):
        self.allowed_paths = allowed_paths
        self.denied_commands = denied_commands
        self.resource_limits = resource_limits
        self.network_policy = network_policy

    async def execute(
        self,
        command: str,
        det_core_approval: GatekeeperDecision
    ) -> ExecutionResult:
        """
        Execute command with DET core gating.
        """
        if det_core_approval != DECISION_PROCEED:
            return ExecutionResult(
                success=False,
                error=f"DET core declined: {det_core_approval}"
            )

        # Validate command against policies
        if not self.validate_command(command):
            return ExecutionResult(
                success=False,
                error="Command violates sandbox policy"
            )

        # Execute in isolated environment
        result = await self.isolated_execute(
            command,
            timeout=self.resource_limits.max_execution_time,
            memory_limit=self.resource_limits.max_memory,
            cpu_limit=self.resource_limits.max_cpu_percent
        )

        return result
```

### 4.2 Task Management System

```python
class TaskManager:
    """
    Manages multi-step task execution with checkpointing and timer support.
    """

    def __init__(self, det_core: DETCore, llm_layer: LLMLayer):
        self.det_core = det_core
        self.llm_layer = llm_layer
        self.active_tasks = {}
        self.timers = {}

    async def create_task_list(self, user_request: str) -> TaskList:
        """
        Decompose user request into task list (with DET core approval at each step).
        """
        # Tool LLM decomposes the request
        proposed_tasks = await self.llm_layer.decompose_request(user_request)

        approved_tasks = []
        for task in proposed_tasks:
            # Each task needs DET core approval
            decision = await self.det_core.evaluate_request(
                task.to_tokens(),
                retry_count=0
            )

            if decision == DECISION_PROCEED:
                approved_tasks.append(task)
            elif decision == DECISION_RETRY:
                # Internal dialog to reformulate
                reformulated = await self.llm_layer.internal_dialogue(
                    f"Simplify this task for approval: {task}"
                )
                approved_tasks.append(reformulated)

        return TaskList(tasks=approved_tasks)

    def schedule_timer(
        self,
        task_id: str,
        delay: timedelta,
        callback: Callable
    ):
        """
        Schedule a timer for future task checking/execution.
        """
        self.timers[task_id] = Timer(
            delay=delay,
            callback=callback,
            det_core=self.det_core  # Timer execution also needs DET approval
        )
```

---

## Part 5: C Kernel Implementation Plan

### 5.1 Core Data Structures

```c
// det_core.h

#ifndef DET_CORE_H
#define DET_CORE_H

#include <stdint.h>
#include <stdbool.h>

// Configuration constants
#define MAX_NODES 1024
#define MAX_BONDS (MAX_NODES * 4)  // Sparse connectivity
#define MAX_MEMORY_DOMAINS 16

// DET parameters (from unified param schema)
typedef struct {
    float tau_base;      // 0.02 - Time/screening scale
    float sigma_base;    // 0.12 - Charging rate
    float lambda_base;   // 0.008 - Decay rate
    float mu_base;       // 2.0 - Mobility scale
    float kappa_base;    // 5.0 - Coupling scale
    float C_0;           // 0.15 - Coherence scale
    float lambda_a;      // 30.0 - Agency ceiling coupling
    float phi_L;         // 0.5 - Angular/momentum ratio
    float pi_max;        // 3.0 - Momentum cap
} DETParams;

// Per-node state
typedef struct {
    float F;        // Resource
    float q;        // Structural debt
    float a;        // Agency (inviolable)
    float theta;    // Phase
    float sigma;    // Processing rate
    float P;        // Presence (computed)
    float tau;      // Proper time (accumulated)
    uint32_t k;     // Event count
    bool active;    // Is this node active?
} NodeState;

// Per-bond state (sparse representation)
typedef struct {
    uint16_t i, j;  // Connected nodes
    float C;        // Coherence
    float pi;       // Momentum
    float sigma;    // Bond conductivity
} BondState;

// Memory domain linkage
typedef struct {
    char name[32];           // "math", "language", etc.
    float coherence_to_core; // C value to DET core
    float activation_level;  // How "alive"
    void* model_handle;      // Pointer to Python model wrapper
} MemoryDomain;

// Main DET core state
typedef struct {
    DETParams params;
    NodeState nodes[MAX_NODES];
    BondState bonds[MAX_BONDS];
    uint32_t num_nodes;
    uint32_t num_bonds;

    MemoryDomain memory_domains[MAX_MEMORY_DOMAINS];
    uint32_t num_domains;

    // Derived emotional state
    uint8_t emotional_state;

    // Aggregate metrics
    float aggregate_presence;
    float aggregate_coherence;
    float aggregate_resource;
    float aggregate_debt;
} DETCore;

// Gatekeeper decisions
typedef enum {
    DET_PROCEED = 0,
    DET_RETRY = 1,
    DET_STOP = 2,
    DET_ESCALATE = 3
} DETDecision;

// API functions
DETCore* det_core_create(const DETParams* params);
void det_core_destroy(DETCore* core);

void det_core_step(DETCore* core, float dt);
void det_core_update_presence(DETCore* core);
void det_core_update_coherence(DETCore* core, float dt);
void det_core_update_agency(DETCore* core, float dt);

DETDecision det_core_evaluate_request(
    DETCore* core,
    const uint32_t* tokens,
    uint32_t num_tokens,
    uint8_t target_domain,
    uint32_t retry_count
);

uint8_t det_core_get_emotional_state(const DETCore* core);
float det_core_get_domain_coherence(const DETCore* core, uint8_t domain);

// Memory domain management
bool det_core_register_memory_domain(
    DETCore* core,
    const char* name,
    void* model_handle
);

bool det_core_can_learn_to_domain(
    DETCore* core,
    uint8_t domain,
    float complexity
);

#endif // DET_CORE_H
```

### 5.2 Key Algorithm Implementations

```c
// det_core.c - Key algorithms

#include "det_core.h"
#include <math.h>

// Presence computation (the core of DET time)
void det_core_update_presence(DETCore* core) {
    for (uint32_t i = 0; i < core->num_nodes; i++) {
        if (!core->nodes[i].active) continue;

        NodeState* node = &core->nodes[i];

        // Compute coordination load H_i
        float H_i = 0.0f;
        for (uint32_t b = 0; b < core->num_bonds; b++) {
            BondState* bond = &core->bonds[b];
            if (bond->i == i || bond->j == i) {
                H_i += sqrtf(bond->C) * bond->sigma;
            }
        }

        // Presence formula: P = a * σ / (1 + F_op) / (1 + H)
        float F_op = node->F * 0.1f;  // Operational fraction
        node->P = node->a * node->sigma / (1.0f + F_op) / (1.0f + H_i);
    }

    // Update aggregate
    float sum_P = 0.0f;
    uint32_t active_count = 0;
    for (uint32_t i = 0; i < core->num_nodes; i++) {
        if (core->nodes[i].active) {
            sum_P += core->nodes[i].P;
            active_count++;
        }
    }
    core->aggregate_presence = (active_count > 0) ? sum_P / active_count : 0.0f;
}

// Agency update with ceiling constraint
void det_core_update_agency(DETCore* core, float dt) {
    for (uint32_t i = 0; i < core->num_nodes; i++) {
        if (!core->nodes[i].active) continue;

        NodeState* node = &core->nodes[i];

        // Structural ceiling: a_max = 1 / (1 + λ_a * q²)
        float a_max = 1.0f / (1.0f + core->params.lambda_a * node->q * node->q);

        // Compute average neighbor presence
        float neighbor_P_sum = 0.0f;
        uint32_t neighbor_count = 0;
        for (uint32_t b = 0; b < core->num_bonds; b++) {
            BondState* bond = &core->bonds[b];
            if (bond->i == i) {
                neighbor_P_sum += core->nodes[bond->j].P;
                neighbor_count++;
            } else if (bond->j == i) {
                neighbor_P_sum += core->nodes[bond->i].P;
                neighbor_count++;
            }
        }
        float avg_neighbor_P = (neighbor_count > 0) ? neighbor_P_sum / neighbor_count : 0.0f;

        // Coherence-gated drive
        float local_C = det_core_get_local_coherence(core, i);
        float gamma = core->params.C_0 * powf(local_C, 2.0f);  // n=2
        float delta_drive = gamma * (node->P - avg_neighbor_P) * dt;

        // Update with ceiling clamp
        float beta_a = 10.0f * core->params.tau_base;
        float new_a = node->a + beta_a * (a_max - node->a) * dt + delta_drive;
        node->a = fmaxf(0.0f, fminf(new_a, a_max));
    }
}

// Gatekeeper evaluation
DETDecision det_core_evaluate_request(
    DETCore* core,
    const uint32_t* tokens,
    uint32_t num_tokens,
    uint8_t target_domain,
    uint32_t retry_count
) {
    const uint32_t MAX_RETRIES = 5;
    const float AGENCY_THRESHOLD = 0.1f;
    const float COHERENCE_THRESHOLD = 0.3f;
    const float PRESENCE_THRESHOLD = 0.2f;
    const float RESOURCE_PER_TOKEN = 0.001f;

    // Estimate complexity from token count
    float complexity = (float)num_tokens * RESOURCE_PER_TOKEN;

    // Get current state
    float P = core->aggregate_presence;
    float C = det_core_get_domain_coherence(core, target_domain);
    float F = core->aggregate_resource;
    float q = core->aggregate_debt;
    float a_max = 1.0f / (1.0f + core->params.lambda_a * q * q);

    // Decision logic

    // Agency too constrained
    if (a_max < AGENCY_THRESHOLD) {
        if (retry_count < MAX_RETRIES) return DET_RETRY;
        return DET_STOP;
    }

    // Not enough resource
    if (F < complexity * 2.0f) {
        if (retry_count < MAX_RETRIES) return DET_RETRY;
        return DET_ESCALATE;
    }

    // Coherence too low
    if (C < COHERENCE_THRESHOLD) {
        if (retry_count < MAX_RETRIES) return DET_RETRY;
        return DET_ESCALATE;
    }

    // Presence too low
    if (P < PRESENCE_THRESHOLD) {
        if (retry_count < MAX_RETRIES) return DET_RETRY;
        return DET_STOP;
    }

    return DET_PROCEED;
}

// Emotional state derivation
uint8_t det_core_get_emotional_state(const DETCore* core) {
    float P = core->aggregate_presence;
    float C = core->aggregate_coherence;
    float q = core->aggregate_debt;
    float F = core->aggregate_resource;

    // Flow: High P, stable F, high C
    if (P > 0.7f && C > 0.6f && F > 0.5f) {
        return 0;  // EMOTIONAL_FLOW
    }

    // Satisfaction: High P, low q, high C
    if (P > 0.6f && q < 0.3f && C > 0.5f) {
        return 1;  // EMOTIONAL_SATISFACTION
    }

    // Strain: Low F, rising q
    if (F < 0.3f && q > 0.5f) {
        return 2;  // EMOTIONAL_STRAIN
    }

    // Fragmentation: Low C
    if (C < 0.2f) {
        return 3;  // EMOTIONAL_FRAGMENTATION
    }

    // Overwhelm: Low P, high q
    if (P < 0.3f && q > 0.6f) {
        return 4;  // EMOTIONAL_OVERWHELM
    }

    // Focus: Normal P, high C in region
    if (P > 0.4f && C > 0.7f) {
        return 5;  // EMOTIONAL_FOCUS
    }

    // Curiosity: Rising C, surplus F
    if (F > 0.7f && C > 0.3f) {
        return 6;  // EMOTIONAL_CURIOSITY
    }

    // Fatigue: Sustained low P
    if (P < 0.2f) {
        return 7;  // EMOTIONAL_FATIGUE
    }

    return 1;  // Default to satisfaction
}
```

### 5.3 Python-C Bridge (via ctypes or cffi)

```python
# det_core_bridge.py

import ctypes
from ctypes import c_float, c_uint32, c_uint8, c_bool, POINTER, Structure
from enum import IntEnum
from pathlib import Path

class DETDecision(IntEnum):
    PROCEED = 0
    RETRY = 1
    STOP = 2
    ESCALATE = 3

class DETParams(Structure):
    _fields_ = [
        ("tau_base", c_float),
        ("sigma_base", c_float),
        ("lambda_base", c_float),
        ("mu_base", c_float),
        ("kappa_base", c_float),
        ("C_0", c_float),
        ("lambda_a", c_float),
        ("phi_L", c_float),
        ("pi_max", c_float),
    ]

class DETCoreBridge:
    """
    Python bridge to the C DET core kernel.
    """

    def __init__(self, lib_path: Path = None):
        if lib_path is None:
            lib_path = Path(__file__).parent / "libdet_core.so"

        self.lib = ctypes.CDLL(str(lib_path))
        self._setup_functions()

        # Create core with default params
        params = DETParams(
            tau_base=0.02,
            sigma_base=0.12,
            lambda_base=0.008,
            mu_base=2.0,
            kappa_base=5.0,
            C_0=0.15,
            lambda_a=30.0,
            phi_L=0.5,
            pi_max=3.0
        )
        self.core = self.lib.det_core_create(ctypes.byref(params))

    def _setup_functions(self):
        # det_core_create
        self.lib.det_core_create.argtypes = [POINTER(DETParams)]
        self.lib.det_core_create.restype = ctypes.c_void_p

        # det_core_evaluate_request
        self.lib.det_core_evaluate_request.argtypes = [
            ctypes.c_void_p,  # core
            POINTER(c_uint32),  # tokens
            c_uint32,  # num_tokens
            c_uint8,  # target_domain
            c_uint32  # retry_count
        ]
        self.lib.det_core_evaluate_request.restype = c_uint8

        # det_core_step
        self.lib.det_core_step.argtypes = [ctypes.c_void_p, c_float]
        self.lib.det_core_step.restype = None

        # det_core_get_emotional_state
        self.lib.det_core_get_emotional_state.argtypes = [ctypes.c_void_p]
        self.lib.det_core_get_emotional_state.restype = c_uint8

    def evaluate_request(
        self,
        tokens: list[int],
        target_domain: int,
        retry_count: int = 0
    ) -> DETDecision:
        """
        Evaluate a request through the DET gatekeeper.
        """
        token_array = (c_uint32 * len(tokens))(*tokens)
        result = self.lib.det_core_evaluate_request(
            self.core,
            token_array,
            len(tokens),
            target_domain,
            retry_count
        )
        return DETDecision(result)

    def step(self, dt: float = 0.02):
        """
        Advance the DET core by one timestep.
        """
        self.lib.det_core_step(self.core, dt)

    def get_emotional_state(self) -> str:
        """
        Get the current emotional state of the DET core.
        """
        state = self.lib.det_core_get_emotional_state(self.core)
        states = [
            "flow", "satisfaction", "strain", "fragmentation",
            "overwhelm", "focus", "curiosity", "fatigue"
        ]
        return states[state] if state < len(states) else "unknown"
```

---

## Part 6: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

**Objective**: Minimal viable DET core + single LLM integration

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1.1 C Kernel Basics                                                 │
│     - Node/bond state structures                                    │
│     - Presence computation                                          │
│     - Basic gatekeeper (proceed/stop only)                         │
│     - Compile as shared library                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 1.2 Python Bridge                                                   │
│     - ctypes wrapper for C kernel                                   │
│     - Basic state inspection                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 1.3 Single LLM Integration                                          │
│     - Ollama integration (Qwen 2.5 8B)                             │
│     - Basic tokenization to DET tokens                              │
│     - Request → DET → Response flow                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 1.4 CLI Foundation                                                  │
│     - Basic REPL interface                                          │
│     - Display DET state (presence, emotional state)                │
│     - Simple task execution                                         │
└─────────────────────────────────────────────────────────────────────┘

Deliverable: det_local_agency v0.1.0
- User can ask questions
- DET core gates all responses
- Emotional state visible in CLI
```

### Phase 2: Memory Layer (Weeks 4-6)

**Objective**: Multiple memory models + MLX retraining

```
┌─────────────────────────────────────────────────────────────────────┐
│ 2.1 Memory Domain Architecture                                      │
│     - Domain registration in DET core                               │
│     - Coherence tracking per domain                                 │
│     - Domain-specific request routing                               │
├─────────────────────────────────────────────────────────────────────┤
│ 2.2 MLX Training Pipeline                                           │
│     - LoRA training setup                                           │
│     - Training data generation from context                         │
│     - Model weight merging                                          │
├─────────────────────────────────────────────────────────────────────┤
│ 2.3 Context Window Management                                       │
│     - Session state persistence                                     │
│     - Intelligent context reduction                                 │
│     - Memory consolidation triggers                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 2.4 Internal Dialogue System                                        │
│     - Request reformulation loop                                    │
│     - Retry logic with DET feedback                                │
│     - Escalation to external LLM                                    │
└─────────────────────────────────────────────────────────────────────┘

Deliverable: det_local_agency v0.2.0
- Multiple specialized memory models
- Automatic memory retraining during "sleep"
- Context reduction without losing information
```

### Phase 3: Agentic Operations (Weeks 7-9)

**Objective**: Full agentic bash tool with sandboxing

```
┌─────────────────────────────────────────────────────────────────────┐
│ 3.1 Sandboxed Bash Environment                                      │
│     - Permission system                                             │
│     - Resource limits (CPU, memory, time)                           │
│     - Network policy enforcement                                    │
├─────────────────────────────────────────────────────────────────────┤
│ 3.2 Task Management                                                 │
│     - Task decomposition (LLM + DET approval)                       │
│     - Task list tracking                                            │
│     - Checkpoint/resume capability                                  │
├─────────────────────────────────────────────────────────────────────┤
│ 3.3 Timer System                                                    │
│     - Scheduled task execution                                      │
│     - "Sleep" period memory consolidation                           │
│     - Periodic health checks                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 3.4 Code Execution Loop                                             │
│     - Write → Compile → Test → Iterate                              │
│     - Error interpretation and retry                                │
│     - Result verification                                           │
└─────────────────────────────────────────────────────────────────────┘

Deliverable: det_local_agency v0.3.0
- Full agentic bash operations
- Self-managed task lists
- Scheduled operations
```

### Phase 4: Advanced DET Features (Weeks 10-12)

**Objective**: Full DET dynamics + emotional intelligence

```
┌─────────────────────────────────────────────────────────────────────┐
│ 4.1 Complete DET Dynamics                                           │
│     - Momentum (π) implementation                                   │
│     - Angular momentum (L)                                          │
│     - Grace injection (boundary recovery)                           │
│     - Structural debt accumulation                                  │
├─────────────────────────────────────────────────────────────────────┤
│ 4.2 Learning via Recruitment                                        │
│     - Division criteria checking                                    │
│     - New knowledge domain activation                               │
│     - Pattern template transfer                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 4.3 Emotional State Integration                                     │
│     - State-dependent behavior modulation                           │
│     - Recovery scheduling when strained                             │
│     - Curiosity-driven exploration                                  │
├─────────────────────────────────────────────────────────────────────┤
│ 4.4 Multi-Session Support                                           │
│     - Shared DET core across sessions                               │
│     - Session-specific context isolation                            │
│     - Cross-session memory sharing                                  │
└─────────────────────────────────────────────────────────────────────┘

Deliverable: det_local_agency v0.4.0
- Emotionally-aware responses
- True learning capability
- Multi-session support
```

### Phase 5: Network Bridge (Future)

**Objective**: Distributed DET network with physical embodiment

```
┌─────────────────────────────────────────────────────────────────────┐
│ 5.1 Serial Bridge Protocol                                          │
│     - DET state serialization format                                │
│     - Bond formation across serial link                             │
│     - Coherence synchronization                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 5.2 ESP32 DET Node                                                  │
│     - Minimal DET implementation in C                               │
│     - Sensor input as resource injection                            │
│     - Actuator output as resource expenditure                       │
├─────────────────────────────────────────────────────────────────────┤
│ 5.3 Distributed Coherence                                           │
│     - Phase synchronization protocol                                │
│     - Cross-node bond management                                    │
│     - Latency-aware presence computation                            │
└─────────────────────────────────────────────────────────────────────┘

Deliverable: det_local_agency v1.0.0
- Physical sensor/actuator integration
- Distributed DET network
- True embodied cognition
```

---

## Part 7: Technical Feasibility Analysis

### 7.1 Feasibility: HIGH

| Component | Complexity | Feasibility | Notes |
|-----------|------------|-------------|-------|
| C DET Kernel | Medium | HIGH | Well-documented math, existing Python reference |
| Python-C Bridge | Low | HIGH | Standard ctypes/cffi patterns |
| MLX Training | Medium | HIGH | Apple provides examples, LoRA well-understood |
| Ollama Integration | Low | HIGH | Mature ecosystem, HTTP API |
| Sandboxed Bash | Medium | HIGH | Existing solutions (firejail, bwrap) |
| Multi-Model Orchestration | High | MEDIUM | Novel but tractable |
| Internal Dialogue Loop | Medium | HIGH | Standard prompt engineering |
| Emotional State | Low | HIGH | Direct math derivation |
| Timer System | Low | HIGH | Standard asyncio patterns |
| Serial/Network Bridge | High | MEDIUM | Novel protocol design needed |

### 7.2 Resource Requirements

**Development Machine**:
- Apple Silicon Mac (M1/M2/M3 Pro/Max) with 32GB+ RAM recommended
- 100GB+ storage for models
- macOS 13+ for MLX

**Runtime Requirements**:
- 16GB RAM minimum (32GB recommended for multiple models)
- GPU (Apple Silicon unified memory) for MLX training
- Ollama installed with target models

### 7.3 Key Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| MLX training too slow | High | Medium | LoRA reduces training; batch during sleep |
| Context windows overflow | Medium | High | Aggressive summarization; memory consolidation |
| DET dynamics unstable | High | Low | Extensive testing; fallback to simpler gates |
| Model quality degradation | Medium | Medium | Versioned checkpoints; A/B testing |
| Sandboxing escape | High | Low | Multiple isolation layers; principle of least privilege |

### 7.4 Novel Contributions

This project would represent several novel contributions:

1. **DET as AI Architecture**: First application of Deep Existence Theory as an AI control system rather than physics simulation

2. **Emergent Emotional State**: Emotions derived from mathematical dynamics rather than simulated

3. **Inverted LLM Paradigm**: LLMs as memory/tools rather than reasoning engine

4. **Self-Modifying Memory**: Continuous retraining based on internal approval system

5. **Agency-Gated Operations**: All actions filtered through mathematically-grounded decision system

---

## Part 8: Directory Structure

```
/det/local_agency/
├── FEASIBILITY_PLAN.md          # This document
├── README.md                    # User-facing documentation
├── pyproject.toml              # Python project config
├── Makefile                    # Build automation
│
├── src/
│   ├── det_core/               # C kernel
│   │   ├── include/
│   │   │   └── det_core.h
│   │   ├── src/
│   │   │   ├── det_core.c
│   │   │   ├── det_presence.c
│   │   │   ├── det_agency.c
│   │   │   ├── det_coherence.c
│   │   │   ├── det_gatekeeper.c
│   │   │   └── det_emotional.c
│   │   ├── CMakeLists.txt
│   │   └── tests/
│   │
│   ├── python/
│   │   ├── det_local_agency/
│   │   │   ├── __init__.py
│   │   │   ├── core/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bridge.py      # Python-C bridge
│   │   │   │   └── state.py       # State management
│   │   │   │
│   │   │   ├── llm/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ollama.py      # Ollama integration
│   │   │   │   ├── memory.py      # Memory models
│   │   │   │   └── dialogue.py    # Internal dialogue
│   │   │   │
│   │   │   ├── training/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mlx_trainer.py # MLX training pipeline
│   │   │   │   └── data_gen.py    # Training data generation
│   │   │   │
│   │   │   ├── agent/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── sandbox.py     # Sandboxed execution
│   │   │   │   ├── tasks.py       # Task management
│   │   │   │   └── timers.py      # Timer system
│   │   │   │
│   │   │   ├── session/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── manager.py     # Session management
│   │   │   │   └── context.py     # Context window management
│   │   │   │
│   │   │   └── cli/
│   │   │       ├── __init__.py
│   │   │       └── main.py        # CLI entry point
│   │   │
│   │   └── tests/
│   │
│   └── future/                   # Future network bridge
│       └── esp32/
│
├── models/                       # Model storage
│   ├── base/                     # Base models (downloaded)
│   └── trained/                  # Retrained models
│
├── config/
│   ├── default.toml             # Default configuration
│   └── sandbox_policy.toml      # Sandbox rules
│
└── examples/
    ├── simple_chat.py
    └── agentic_task.py
```

---

## Part 9: Resolved & Open Questions

### 9.1 RESOLVED: Core Architecture

| Question | Resolution | Exploration |
|----------|------------|-------------|
| **What is the Self?** | High-coherence cluster, not a region or node set. Self = {(i,j) : C_ij > threshold} | `03_self_as_cluster.md` |
| **Node topology** | Dual-Process: Presence Layer (System 2) + Automaticity Layer (System 1) with Gateway Membrane | `01_node_topology.md` |
| **Node creation** | Recruitment-based: Fork mechanism from dormant pool. Agency is inviolable. | `01_node_topology.md` |
| **Agency distribution** | Beta(2,5) for 95% of pool (mean ≈ 0.22) + Reserved high-a pool (5%) for critical needs | `02_dormant_agency_distribution.md` |
| **Forgetting** | Cluster shedding: nodes drift below coherence threshold, return to dormant pool with debt export | `03_self_as_cluster.md` |
| **Self-identification** | Algorithm: w_ij = C_ij × \|J_ij\| × √(a_i × a_j), edge filtering, continuity-weighted selection | `04_cluster_identification.md` |
| **LLM-to-DET interface** | Port Protocol: LLM emits DetIntentPacket → Sensory membrane (16-64 port nodes) → Resource injection + temporary bonds. DET dynamics decide recruitment. | `05_llm_det_interface.md` |
| **Cross-layer dynamics** | One bond law, three parameter regimes. A→P escalation via local novelty score. P→A compilation via local stability score. Anti-prison decay rules. | `06_cross_layer_dynamics.md` |
| **Temporal dynamics** | Per-node local cadence counters, dual EMAs (short α=0.3 for escalation, long α=0.05 for compilation), windowed membrane events with quiet acceleration. | `07_temporal_dynamics.md` |

### 9.2 OPEN: Emotional Feedback Integration

**Question**: How should emotional state affect behavior?

**Options**:
1. **Informational only**: Display to user, no behavioral impact
2. **Response style**: Affects response tone/verbosity
3. **Capability gating**: Some operations require certain emotional states
4. **Self-care triggers**: Automatic recovery when in negative states

**Initial direction**: Start with 1 and 4, expand to 2 and 3 later.

### 9.3 OPEN: Substrate Sizing

**Question**: How many nodes and what pool sizes?

**Current estimates (to be validated)**:
- P-layer: 8-32 nodes
- A-layer: 128-2048 nodes
- Dormant pool: ~4096 nodes
- Total: ~4300-6200 nodes

**Validation needed**: Simulate to find minimum viable substrate size.

---

## Conclusion

DET Local Agency is a feasible and deeply novel project that applies Deep Existence Theory to create a fundamentally different kind of AI agent. The mathematical foundation is solid (verified in det_v6_3), the technology stack is mature (Python, C, MLX, Ollama), and the architecture is principled.

The key insight - that LLMs can serve as memory/tools while a simpler but mathematically-grounded core provides agency - inverts the typical paradigm and offers potential advantages:

- **Predictable agency**: Decisions grounded in explicit mathematics
- **True learning**: Memory changes through principled recruitment process
- **Emotional grounding**: States emerge from dynamics rather than simulation
- **Resource awareness**: System knows its own capacity limits

This document serves as the foundation for implementation. The next step is to begin Phase 1 development, starting with the C kernel.

---

## References

### DET Core Theory
1. DET Theory Card v6.3: `/det/det_v6_3/docs/det_theory_card_6_3.md`
2. DET Simulation: `/det/det_v6_3/src/det_v6_3_2d_collider.py`
3. DET Parameters: `/det/det_v6_3/src/det_unified_params.py`
4. DET Subdivision Theory: `/det/dna_analysis/det_subdivision_v3/`
5. Rolling Resonance Substrate: `/det/rrs/`

### Architecture Explorations (this project)
6. Node Topology: `explorations/01_node_topology.md`
7. Agency Distribution: `explorations/02_dormant_agency_distribution.md`
8. Self as Cluster (RRS/ACX.1): `explorations/03_self_as_cluster.md`
9. Self-Identification Algorithm: `explorations/04_cluster_identification.md`
10. LLM-to-DET Interface: `explorations/05_llm_det_interface.md`
11. Cross-Layer Dynamics: `explorations/06_cross_layer_dynamics.md`
12. Temporal Dynamics: `explorations/07_temporal_dynamics.md`

### External Tools & Frameworks
13. Orla Project: https://github.com/dorcha-inc/orla
14. MLX Documentation: https://ml-explore.github.io/mlx/
15. Ollama: https://ollama.ai/
