# DET Local Agency - Roadmap v2

## Vision

A fully DET-native operating system where **everything is an Existence-Lang creature** executed via the EIS substrate. The only non-EL code is the substrate layer itself (C/Metal for performance).

```
┌─────────────────────────────────────────────────┐
│                 User Terminal                    │
└─────────────────────┬───────────────────────────┘
                      │ (terminal port)
┌─────────────────────▼───────────────────────────┐
│           TerminalCreature.ex                    │
└─────────────────────┬───────────────────────────┘
                      │ bond
┌─────────────────────▼───────────────────────────┐
│            LLMCreature.ex                        │
├──────────┬──────────┴──────────┬────────────────┤
│   bond   │       bond          │     bond       │
▼          ▼                     ▼                ▼
Memory    Tool               Reasoner         Planner
.ex       .ex                  .ex              .ex
└──────────┴──────────┬──────────┴────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              EIS Interpreter                     │
│    (executes bytecode, manages phases)           │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           Substrate Layer (C/Metal)              │
│  - Primitives (llm_call, exec, file_*, etc.)    │
│  - State management (F, a, q, bonds)            │
│  - GPU acceleration (Metal compute shaders)      │
└─────────────────────────────────────────────────┘
```

---

## Completed Phases

### Phase 1-13: Foundation ✅
- C kernel with DET physics
- Python bindings
- Existence-Lang parser and compiler
- EIS bytecode format
- Substrate v2 with phase-aware execution
- DET-OS bootstrap and runtime
- Creature loading system (JIT + bytecode)

### Phase 14: GPU Backend ✅
- Metal compute shaders for parallel execution
- 1,000+ ticks/sec with 10,000 nodes
- Python/C/Metal integration complete
- GPU commands in det_os_boot REPL

### Phase 15: EIS Interpreter ✅
- Python interpreter executes EIS bytecode
- Full phase semantics (READ→PROPOSE→CHOOSE→COMMIT)
- 46 opcodes implemented
- Register file with scalar/ref/token registers

### Phase 16: Kernel Runtime ✅
- Message dispatcher
- Kernel execution with F cost tracking
- CreatureRunner for creature lifecycle
- Bond-based messaging

### Phase 17: Substrate Primitives ✅
- 22+ primitives implemented
- LLM, file I/O, shell execution, time, random, terminal
- Primitive registry with cost model
- Agency-gated execution

### Phase 18: Pure EL Creatures ✅
- LLMCreature.ex, ToolCreature.ex, MemoryCreature.ex
- ReasonerCreature.ex, PlannerCreature.ex
- CalculatorCreature.ex (math expression evaluator)
- All creature logic in Existence-Lang

### Phase 19: Terminal Creature ✅
- TerminalCreature.ex with 9 kernels
- Terminal primitives (read, write, prompt, color)
- Minimal Python bootstrap (det_os_boot.py)
- Bond-based command dispatch

### Phase 20: Full Integration ✅
- **20.1** Python wrapper deprecation (archived to /archive)
- **20.2** Performance optimization (25% faster parser, 6.8x faster load)
- **20.3** GPU integration in det_os_boot.py
- **20.4** Project cleanup (removed 300KB deprecated code)

---

## Current Phase

### Phase 20.5: Native DET Collider
**Goal**: DET physics experimentation using native Existence-Lang, not Python wrappers

The collider implements DET v6.3 physics as pure Existence-Lang creatures on a lattice substrate. This validates that the substrate's core physics (F, a, q, bonds, phases) can express the full DET theory.

#### Design Philosophy

The substrate already has the physics primitives:
- **Nodes** are creatures with F (resource), a (agency), q (structure)
- **Bonds** have C (coherence), and can carry momentum π
- **Phases** (READ→PROPOSE→CHOOSE→COMMIT) map to physics timesteps
- **GPU** accelerates parallel node/bond updates

What we need to add:
1. **Lattice topology** - Regular grid bond patterns (1D chain, 2D grid, 3D cubic)
2. **FFT primitives** - For Helmholtz/Poisson gravity solvers (must be C/Metal)
3. **Neighbor access** - Efficient access to adjacent nodes on lattice
4. **Physics kernels** - Pure EL implementations of DET v6.3 operators

```
┌─────────────────────────────────────────────────────────────┐
│                    ColliderCreature.ex                       │
│  Orchestrates physics: Init, Step, Query, Render            │
└──────────────────────────┬──────────────────────────────────┘
                           │ bonds to lattice
┌──────────────────────────▼──────────────────────────────────┐
│                    Lattice of Nodes                          │
│  Each node: F, a, q, σ, P, Δτ                               │
│  Each bond: C, π (momentum)                                  │
│  Topology: 1D chain, 2D grid, or 3D cubic                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Substrate + GPU                           │
│  FFT solver, parallel phase execution, Metal shaders        │
└─────────────────────────────────────────────────────────────┘
```

#### 20.5.1 Lattice Substrate Extensions
- [ ] `lattice_create(dim, N)` - Create 1D/2D/3D periodic lattice
- [ ] `lattice_get_neighbor(node, direction)` - Get adjacent node ref
- [ ] `lattice_get_bond(node, direction)` - Get bond to neighbor
- [ ] Automatic bond topology for regular grids
- [ ] Lattice-aware GPU dispatch (coalesced memory access)

#### 20.5.2 FFT Gravity Primitives
- [ ] `fft_forward(field)` - Forward FFT on lattice field
- [ ] `fft_inverse(field_k)` - Inverse FFT
- [ ] `gravity_solve(q_field)` - Helmholtz + Poisson solver
- [ ] Metal Performance Shaders integration for GPU FFT
- [ ] Lattice correction factor η by dimension/size

#### 20.5.3 Physics Operators in Existence-Lang

**Presence Kernel** (physics.ex):
```existence
kernel ComputePresence {
    // P = a·σ / (1 + F) / (1 + H)
    // Δτ = P · dt
    phase READ {
        my_F ::= witness(self.F);
        my_a ::= witness(self.a);
        my_sigma ::= witness(self.sigma);
    }
    phase PROPOSE {
        proposal UpdatePresence {
            score = 1.0;
            effect {
                H = my_sigma;  // or coherence-weighted Option B
                P = my_a * my_sigma / (1.0 + my_F) / (1.0 + H);
                self.P = P;
                self.Delta_tau = P * DT;
            }
        }
    }
    // ... CHOOSE, COMMIT
}
```

**Flow Kernel** - Diffusive, momentum, floor, gravity flux
**Momentum Kernel** - β_g gravity coupling, decay
**Structure Kernel** - q accumulation from outflow
**Agency Kernel** - v6.4 structural ceiling + relational drive
**Grace Kernel** - Boundary injection for depleted nodes

#### 20.5.4 ColliderCreature.ex
- [ ] Init kernel: Create lattice, set parameters
- [ ] AddPacket kernel: Inject Gaussian resource distribution
- [ ] Step kernel: Execute one/N physics timesteps
- [ ] Query kernel: Total mass, separation, potential energy
- [ ] Render kernel: Generate ASCII field visualization

#### 20.5.5 Visualization (Temporary Python)
- [ ] ASCII art renderer for terminal (via terminal primitives)
- [ ] Python matplotlib hook for development visualization
- [ ] Future: Native DET visualization creature

#### 20.5.6 Validation Tests
- [ ] Vacuum stability (q=0 → no gravity)
- [ ] Mass conservation (with grace accounting)
- [ ] Gravitational binding (two-body inspiral)
- [ ] Time dilation (P decreases with F)
- [ ] Compare results with det_v6_3 Python colliders

---

## Current Architecture

### Entry Point
```bash
python det_os_boot.py [--gpu] [-v]
```

### Core Modules
```
det/
├── lang/           # Existence-Lang compiler
│   ├── parser.py          # Recursive descent parser
│   ├── tokens.py          # Lexer/tokenizer
│   ├── ast_nodes.py       # AST definitions
│   ├── eis_compiler.py    # AST → EIS bytecode
│   ├── bytecode_cache.py  # .exb caching
│   └── excompile.py       # CLI compiler
│
├── eis/            # EIS Virtual Machine
│   ├── vm.py              # Interpreter
│   ├── creature_runner.py # Creature execution
│   ├── primitives.py      # Primitive registry
│   ├── registers.py       # Register file
│   ├── phases.py          # Phase controller
│   ├── memory.py          # Trace store
│   └── encoding.py        # Opcode definitions
│
├── os/             # DET-OS Kernel
│   ├── kernel.py          # Core kernel
│   └── creatures/         # Built-in creatures
│
└── metal.py        # Metal GPU backend
```

### Existence-Lang Creatures
```
src/existence/
├── terminal.ex     # REPL interface
├── llm.ex          # LLM reasoning
├── tool.ex         # Shell execution
├── memory.ex       # Memory storage
├── reasoner.ex     # Chain-of-thought
├── planner.ex      # Task planning
├── calculator.ex   # Math evaluation
├── kernel.ex       # Core kernel library
├── physics.ex      # DET physics operators (presence, flow, agency)
└── collider.ex     # Native DET collider (Phase 20.5)
```

---

## Next Phases

### Phase 21: LLM Integration Enhancement ✅ COMPLETE
**Goal**: Better LLM coordination and multi-model support

#### 21.1 LLM Creature Enhancement ✅
- [x] Multi-model support in LLMCreature.ex
- [x] Temperature modulation by agency/arousal
- [x] Token budget management per-creature
- [x] Streaming response support

#### 21.2 Conversation Management ✅
- [x] ConversationCreature.ex for multi-turn context
- [x] Memory-backed context window (via StoreMemory/RecallMemory kernels)
- [x] Automatic context summarization when budget exceeded (Summarize kernel)
- [x] DET state explanation and translation (ExplainDET kernel)
- [x] Integrated reasoning/planning via Reasoner/Planner bonds
- [x] Built-in DET command help system

#### 21.3 Model Routing (Partial - in LLMCreature)
- [x] Domain-aware model selection (code→coder, math→math-model)
- [ ] Fallback chains when primary model unavailable
- [ ] Cost tracking per model

### Phase 22: System Integration
**Goal**: Full system access and tool execution

#### 22.1 Enhanced Tool Creature
- [ ] Multi-step command execution
- [ ] Working directory management
- [ ] Environment variable handling
- [ ] Async command execution

#### 22.2 File System Creature
- [ ] FileCreature.ex for file operations
- [ ] Watch/monitor file changes
- [ ] Project structure understanding

#### 22.3 Git Integration
- [ ] GitCreature.ex for version control
- [ ] Commit, branch, diff operations
- [ ] Conflict resolution assistance

### Phase 23: Networking
**Goal**: Distributed creature execution

#### 23.1 Network Primitives
- [ ] `net_connect(host, port)` - Establish connection
- [ ] `net_send(channel, data)` - Send message
- [ ] `net_receive(channel)` - Receive message
- [ ] `net_discover()` - Find peers

#### 23.2 Remote Bonds
- [ ] Bond channels over TCP/WebSocket
- [ ] Serialization of bond messages
- [ ] Reconnection handling

#### 23.3 Distributed Execution
- [ ] Remote creature spawning
- [ ] Load balancing across nodes
- [ ] Consensus for shared state

### Phase 24: Persistence
**Goal**: State persistence and recovery

#### 24.1 Creature Serialization
- [ ] Save creature state (F, a, q, registers)
- [ ] Save bond topology
- [ ] Save message queues

#### 24.2 Checkpoint/Restore
- [ ] Periodic checkpoints
- [ ] Crash recovery
- [ ] State rollback

#### 24.3 Memory-Mapped Substrate
- [ ] mmap for node/bond arrays
- [ ] Shared memory between processes
- [ ] Persistent trace store

### Phase 25: Security
**Goal**: Capability-based access control

#### 25.1 Resource Quotas
- [ ] F budget per creature
- [ ] Primitive call limits
- [ ] Memory limits

#### 25.2 Capability System
- [ ] Capabilities granted via bonds
- [ ] Capability delegation
- [ ] Revocation

#### 25.3 Sandboxed Primitives
- [ ] File access restrictions
- [ ] Network restrictions
- [ ] Process isolation

### Phase 26: Native Model Inference (Ollama Replacement)
**Goal**: DET-native LLM inference - load GGUF models directly, execute as creatures, replace Ollama dependency

**Key Insight**: Model inference becomes DET-native while avoiding the trap of implementing matmul/attention as bond diffusion (beautiful but slow). Instead, treat matmul/attention as effect kernels whose output is deterministic and auditable, while DET controls when/why they run and how results are committed.

**Critical Design**: The sampler is where you "stay DET" - `det_choose_token()` is the ur-choice that gets committed as past trace/witness.

#### 26.1 Foundation Primitives (M1)
**Goal**: Basic tensor operations in substrate (C + Metal)

- [ ] Memory-mapped tensor loading (`tensor_mmap`)
- [ ] Basic matmul (CPU fallback with BLAS)
- [ ] Metal matmul shader (GPU-accelerated)
- [ ] RMSNorm primitive
- [ ] SiLU activation primitive
- [ ] Softmax with optional temperature
- [ ] Element-wise ops (add, mul, div_scalar)

**Performance Notes** (from substrate_lattice.c learnings):
- Keep buffers persistent (reuse across tokens)
- Avoid per-step allocations
- Profile hot paths early

#### 26.2 GGUF Model Loading (M2)
**Goal**: Parse GGUF format, extract weights

- [ ] GGUF header parser (metadata extraction)
- [ ] Weight tensor extraction
- [ ] Quantization support (Q4_K_M minimum, Q8_0)
- [ ] Memory-mapped weights (don't load full model into RAM)
- [ ] ModelLoaderCreature.ex

**Interim**: Use ollama.cpp as library for GGUF parsing during development

#### 26.3 Inference Pipeline (M3)
**Goal**: Single token generation via creatures

- [ ] EmbeddingCreature.ex (token → embedding)
- [ ] TransformerLayerCreature.ex (coordinates attn + FFN)
- [ ] AttentionHeadCreature.ex (attention as auditable effect)
- [ ] FFNCreature.ex (feed-forward computation)
- [ ] SamplerCreature.ex with `det_choose_token()`

**Architecture Decision**: Per-layer creatures (not per-head) for simplicity, with attention heads as internal structure.

#### 26.4 KV Cache (M4)
**Goal**: Efficient multi-token generation

- [ ] `kv_cache_create(layers, max_len, d)` primitive
- [ ] `kv_cache_append(cache, k, v)` primitive
- [ ] `kv_cache_slice(cache, start, end)` for sliding window
- [ ] Cache management in ModelCreature
- [ ] Context window handling (truncation, summarization trigger)

**Performance Critical**: KV cache correctness + efficiency is where decode speed lives.

#### 26.5 DET-Native Sampler
**Goal**: The sacred DET integration point

- [ ] `det_choose_token(logits, temperature, top_p, top_k)` primitive
- [ ] DET agency modulates temperature (higher a → lower temp)
- [ ] DET arousal modulates variability
- [ ] Choice committed as witness/trace
- [ ] Reproducible via deterministic RNG from trace state
- [ ] Repetition penalty via context tokens

**Key**: Even when using external sampling, wrap it in DET choose semantics.

#### 26.6 Truthfulness Weighting
**Goal**: Compute reliability score T for each output

- [ ] TruthfulnessEvaluator creature
- [ ] Per-token truthfulness from layer states
- [ ] Composite score from: debt (q), agency (a), attention entropy, bond coherence
- [ ] Truthfulness vector output (factual_grounding, logical_coherence, etc.)
- [ ] Calibration infrastructure

**Formula**: `T = w_debt/(1+q) + w_agency*a + w_entropy*(1-H/H_max) + w_coherence*C`

#### 26.7 Metal Performance Shaders (M5)
**Goal**: GPU-accelerated inference

- [ ] Metal attention kernel (Q·K^T/√d_k, softmax, V multiply)
- [ ] Metal matmul for large weight matrices
- [ ] Batch prefill (prompt ingestion)
- [ ] Speculative decoding exploration

**Apply learnings from substrate_lattice Metal shaders**.

#### 26.8 Integration (M6)
**Goal**: Drop-in replacement for Ollama

- [ ] LLMCreature.ex compatibility layer (unchanged interface)
- [ ] Model switching support (load/unload)
- [ ] Streaming token output
- [ ] Graceful fallback to Ollama if native fails
- [ ] Deprecate Ollama primitives (phase out `llm_call_v2` HTTP calls)

#### 26.9 Success Criteria

**MVP**:
- [ ] Load and run phi-2 (2.7B params) or qwen2-0.5B
- [ ] Generate coherent text (same quality as Ollama)
- [ ] Track F expenditure per generation
- [ ] Integrate with existing LLMCreature.ex

**Full Success**:
- [ ] Support llama-architecture models up to 7B
- [ ] Performance within 2x of Ollama throughput
- [ ] Quantization support (Q4_K_M, Q8_0)
- [ ] Streaming token generation
- [ ] Full DET physics integration (P-based scheduling)
- [ ] Truthfulness weighting on all outputs
- [ ] Deprecate Ollama dependency

#### 26.10 DET Physics Mapping

| Transformer Concept | DET Concept |
|---------------------|-------------|
| Attention weight | Bond coherence (auditable, not simulated) |
| Softmax temperature | Agency-modulated decisiveness |
| Token sampling | Ur-choice (committed as witness) |
| FLOPS (matmul) | F expenditure |
| Model weights | Creature structure (q) |
| KV cache | Creature memory state |

#### 26.11 Anti-Hallucination Mechanisms

| Pathology | DET Mechanism |
|-----------|---------------|
| Reward hacking | F expenditure tracks real compute |
| False confidence | Agency from structure, not assertion |
| Ungrounded claims | Debt (q) accumulation |
| Hidden attention | Attention is auditable effect |
| Post-hoc justification | Atomic commits, no retroactive changes |

---

## Design Principles

1. **DET First**: All behavior emerges from DET physics (F, a, P, bonds)
2. **Existence-Lang Native**: All creature logic in EL, not Python
3. **Substrate is Infrastructure**: Only primitives and execution in Python/C/Metal
4. **Bonds are Communication**: No direct method calls between creatures
5. **Phases are Atomic**: READ→PROPOSE→CHOOSE→COMMIT is the execution unit
6. **Resources are Real**: Every action costs F, tracked honestly
7. **GPU When Needed**: Use Metal for large-scale parallel execution
8. **No Wrapper Creatures**: Physics/colliders are native EL on substrate, not Python wraps

---

## Quick Start

```bash
# Navigate to project
cd local_agency/src/python

# Run DET-OS
python det_os_boot.py

# Run with GPU acceleration
python det_os_boot.py --gpu

# Run with verbose output
python det_os_boot.py -v
```

### REPL Commands
```
help              Show commands
status            Show creature status
list              List available creatures
load <name>       Load a creature
use <name>        Load and bond creature
bond <a> <b>      Bond two creatures
calc <expr>       Calculator shortcut
gpu               Show GPU status
gpu benchmark     Run GPU benchmark
quit              Exit
```

---

*Last Updated: 2026-01-24* | *Phase 20.5 - Native DET Collider*
