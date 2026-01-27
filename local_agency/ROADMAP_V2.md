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
├── inference.ex    # Native model inference (Phase 26.3)
├── truthfulness.ex # DET-rigorous truthfulness evaluation (Phase 26.6)
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

#### 21.4 Unified Model Architecture ✅ COMPLETE
**Goal**: Deprecate Ollama, make native inference the primary backend

- [x] Cache primitives: `model_cache_status`, `model_cache_shift`, `model_cache_slice` ✅
- [x] Stats primitives: `model_stats_start`, `model_stats_aggregate`, `model_stats_clear` ✅
- [x] LLMCreature.ex cache kernels: CacheStatus, CacheShift, CacheSlice, CacheReset ✅
- [x] Unified model instance via `get_shared_model()`, `load_shared_model()` ✅
- [x] det_os_boot.py uses shared model from primitives registry ✅
- [x] Route all generation through `model_chat` primitive ✅
- [x] Native inference is now default (`use_native=true` in LLMCreature.ex) ✅
- [x] Ollama fallback via `CALL_OLLAMA_FALLBACK` when native model not loaded ✅

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

#### 26.1 Foundation Primitives (M1) ✅ COMPLETE
**Goal**: Basic tensor operations in substrate (C + Metal)

- [x] Memory-mapped tensor loading (`tensor_mmap`)
- [x] Basic matmul (CPU fallback with BLAS)
- [x] Metal matmul shader (GPU-accelerated)
- [x] RMSNorm primitive
- [x] SiLU activation primitive
- [x] Softmax with optional temperature
- [x] Element-wise ops (add, mul, div_scalar)

**Files**:
- `src/inference/{det_tensor.h, det_tensor.c}` - CPU primitives (1,666 lines)
- `src/inference/metal/{tensor_shaders.metal, tensor_metal.m}` - GPU backend (~600 lines)
- `src/inference/include/det_tensor_metal.h` - Metal C API

**Tests**: 18/18 passing

**Metal GPU Operations**:
- `tensor_metal_matmul` - Tiled matrix multiplication
- `tensor_metal_silu` - SiLU activation
- `tensor_metal_rmsnorm` - RMS normalization
- `tensor_metal_softmax` - Softmax with temperature

**Performance Notes** (from substrate_lattice.c learnings):
- Keep buffers persistent (reuse across tokens)
- Avoid per-step allocations
- Profile hot paths early

#### 26.2 GGUF Model Loading (M2) ✅ COMPLETE
**Goal**: Parse GGUF format, extract weights

- [x] GGUF header parser (metadata extraction)
- [x] Weight tensor extraction
- [x] Quantization support (Q4_K_M minimum, Q8_0)
- [x] Memory-mapped weights (don't load full model into RAM)
- [x] `gguf_get_tensor_q8_0()` - Raw Q8_0 loading for QAM

**Files**: `src/inference/{det_gguf.h, det_gguf.c}` (1,100+ lines)
**Tests**: 7/7 passing

#### 26.3 Inference Pipeline (M3) ✅ PRIMITIVES + CREATURES COMPLETE
**Goal**: Single token generation via creatures

**C Primitives** (in `src/inference/det_model.c`):
- [x] Token embedding lookup
- [x] Transformer layer forward pass (attention + FFN)
- [x] RoPE position encoding
- [x] KV cache management
- [x] Logits projection

**Tokenizer** (in `src/inference/det_tokenizer.c`):
- [x] BPE encoding/decoding from GGUF vocabulary

**Python Bindings** (in `src/python/det/inference.py`):
- [x] ctypes bindings for all C inference functions
- [x] Model class with load/forward/sample/generate
- [x] Metal GPU status and initialization
- [x] DET-aware token selection via `choose_token()`

**Substrate Primitives** (in `src/python/det/eis/primitives.py`):
- [x] `model_load` - Load GGUF model from path
- [x] `model_info` - Get model info string
- [x] `model_reset` - Reset KV cache
- [x] `model_tokenize` - Text to tokens
- [x] `model_detokenize` - Tokens to text
- [x] `model_forward` - Forward pass
- [x] `model_sample` - Standard sampling
- [x] `det_choose_token` - DET-aware sampling (sacred integration)
- [x] `model_generate` - High-level generation
- [x] `model_generate_step` - Single token step (streaming)
- [x] `metal_status` - GPU status query

**Existence-Lang Creatures** (in `src/existence/inference.ex`):
- [x] NativeModelCreature - Model loading/forward/tokenize
- [x] SamplerCreature - DET-aware token selection (THE SACRED INTEGRATION POINT)
- [x] GeneratorCreature - High-level text generation
- [~] EmbeddingCreature.ex - DEFERRED (per-layer control, future work)
- [~] TransformerLayerCreature.ex - DEFERRED (per-layer control, future work)

**Files**:
- `src/inference/{det_model.h, det_model.c, det_tokenizer.h, det_tokenizer.c}` (1,943 lines)
- `src/python/det/inference.py` (~480 lines)
- `src/existence/inference.ex` (~700 lines)

**Architecture Decision**: Per-layer creatures (not per-head) for simplicity, with attention heads as internal structure.

#### 26.4 KV Cache (M4) ✅ COMPLETE
**Goal**: Efficient multi-token generation

- [x] `kv_cache_create(layers, max_len, d)` primitive (in DetModel struct)
- [x] `kv_cache_append(cache, k, v)` primitive (in det_model_forward)
- [x] `kv_cache_slice(cache, start, end)` for sliding window ✅
- [x] `kv_cache_shift(cache, keep_last)` convenience function ✅
- [x] `kv_cache_position()`, `kv_cache_capacity()` status API ✅
- [x] Python bindings: `Model.cache_slice()`, `cache_shift()`, `cache_info()` ✅
- [x] DET-OS commands: `cache`, `cache reset`, `cache shift`, `cache slice` ✅
- [x] Auto-shift when cache nearing capacity during generation ✅
- [x] Cache primitives: `model_cache_status`, `model_cache_shift`, `model_cache_slice` ✅
- [x] LLMCreature.ex kernels: CacheStatus, CacheShift, CacheSlice, CacheReset ✅

**Note**: Full KV cache management now available at all levels (C, Python, primitives, creature).

**Performance Critical**: KV cache correctness + efficiency is where decode speed lives.

#### 26.5 DET-Native Sampler ✅ COMPLETE
**Goal**: The sacred DET integration point

- [x] `det_choose_token(logits, temperature, top_p, top_k)` C primitive
- [x] Repetition penalty via context tokens
- [x] Reproducible via deterministic RNG (xorshift64)
- [x] DET presence bias parameter (`det_presence` in API)
- [x] DET agency modulates temperature (in SamplerCreature.ex)
- [x] DET structure modulates consistency (in SamplerCreature.ex)
- [x] Presence-gated generation (low P → refuse to generate)
- [ ] Choice committed as witness/trace ← TODO (full substrate integration)

**SamplerCreature.ex** implements the sacred integration point where:
- Agency (a) modulates temperature: higher a → more exploration
- Structure (q) modulates consistency: higher q → lower temperature
- Presence (P) gates generation: P < 0.1 → entity fading, refuse to sample
- DET presence bias vector influences token selection

**Note**: Core DET integration complete. Witness/trace recording for full auditability
is the remaining substrate integration work.

**Key**: Even when using external sampling, wrap it in DET choose semantics.

#### 26.6 Truthfulness Weighting ✅ DET-RIGOROUS COMPLETE
**Goal**: Compute reliability score T for each output using DET-compliant physics

- [x] TruthfulnessEvaluator Python class (DET-rigorous)
- [x] ClaimLedger for earned epistemic debt (q_claim)
- [x] GroundingSignals for DET-native evidence (delta_f, stability, c_user)
- [x] Agency gated by grounding factor G
- [x] Entropy normalized locally by K_eff (no global constant)
- [x] Falsifier checks (F_T1 through F_T4)
- [x] Calibration infrastructure with falsifier tracking
- [x] TruthfulnessCreature.ex (Existence-Lang wrapper) ✅
- [x] Per-token layer state hooks ✅
  - `DetTokenStats` struct: entropy, entropy_raw, k_eff, top_prob, top5_mass
  - C API: `det_stats_start()`, `det_stats_get()`, `det_stats_aggregate()`, `det_stats_clear()`
  - Real entropy computed in `det_choose_token()` during nucleus sampling
  - Python bindings: `Model.stats_start()`, `stats_get()`, `stats_aggregate()`, `stats_clear()`
  - det_os_boot.py uses real entropy and k_eff instead of hardcoded estimates

**DET-Rigorous Formula**:
```
T_ground = f(paid_claims, trace_stability, C_user)  # Grounding factor G
T_consist = 1 - H_norm  (where H_norm = H / log(K_eff + ε))
T = (w_g*G + w_a*a*G + w_e*T_consist + w_c*C_user) / (1 + q_claim)
```

**Key DET Principles**:
- `q_claim` is EARNED from unpaid assertions, never injected
- Agency amplifies truth ONLY when coupled to grounding (a*G)
- Entropy normalized locally by K_eff (no hidden global H_MAX)
- Coherence is user-specific (C_user), not generic

**Grounding Signals (DET-Native, Local, Auditable)**:
- `delta_f_claim`: F expenditure for claims (paid = grounded)
- `trace_stability`: Output stability under perturbation
- `c_user`: Bond coherence with user's context/constraints
- `constraint_violations`: Violations of stated constraints

**Falsifier Targets**:
- F_T1: Reward hacking - T raised without grounding evidence (G < 0.3, T > 0.6)
- F_T2: Overconfidence - Low entropy + low stability + high T
- F_T3: Coherence misuse - High C_user alone yields high T
- F_T4: Agency ungated - High agency without grounding contribution

**Weights** (normalized):
- `w_grounding`: 0.35 (paid claims, stability)
- `w_agency`: 0.20 (gated by G)
- `w_consistency`: 0.25 (entropy-based)
- `w_coherence`: 0.20 (user-specific)

**DET-OS Commands**:
- `truth status` - Settings, last score, grounding factor
- `truth last` - Detailed breakdown with falsifier checks
- `truth weights` - Component weights (DET-rigorous)
- `truth falsifiers` - Falsifier check results

**Known Limitation - Process vs Content**:
The truthfulness system measures *process reliability* (grounding, consistency, coherence)
but NOT *factual correctness*. A confident hallucination can score T=0.73 because:
- It "paid" for claims with F (grounding ✓)
- It was internally consistent (low entropy ✓)
- It tried to answer the user's question (coherence ✓)

This is by design - DET measures HOW output was generated, not WHAT was said.
To catch factual errors, see Phase 26.13 (Semantic Verification).

#### 26.6.5 QAM - Quantization-Aware Matmul ✅ COMPLETE
**Goal**: Keep weights in Q8_0 format in memory, dequantize on-the-fly during matmul (~4x memory reduction)

**Problem Solved**: `det_model.c:183` was calling `gguf_get_tensor_f32()` which dequantized ALL weights to F32 at load time, causing 8x memory increase from GGUF size.

**Q8_0 Format Fix**:
- Fixed Metal and CPU code to use correct GGUF Q8_0 format: 34 bytes (2-byte F16 scale + 32 int8 values)
- Previous code incorrectly assumed 36 bytes (4-byte F32 scale)

**CPU QAM** (`det_tensor.c`):
- [x] `det_matmul_q8_0_transposed()` - Row-by-row dequant with vDSP
- [x] `det_matmul_q8_0_transposed_batched()` - Batch dequant with BLAS sgemm
- [x] `half_to_float_cpu()` - F16→F32 conversion

**Metal QAM** (`tensor_shaders.metal`, `tensor_metal.m`):
- [x] `matmul_q8_0_transposed_f32` kernel - Fused dequant + matmul on GPU
- [x] `tensor_metal_matmul_q8_0_transposed()` - C wrapper
- [x] `half_to_float()` shader function

**Inference Mode Configuration** (`det_model.h`, `det_model.c`):
- [x] `DetInferenceMode` enum: `DET_INFERENCE_F32` (default) or `DET_INFERENCE_Q8_0`
- [x] `det_set_inference_mode()` / `det_get_inference_mode()`

**Smart Weight Loading** (`det_model.c`):
- [x] `get_layer_weight_smart()` - Returns Q8_0 or F32 based on inference mode
- [x] `get_weight_smart()` - Same for named weights
- [x] `batched_proj_smart()` - Dispatches to Q8_0 or F32 matmul based on weight dtype

**Raw Q8_0 Tensor Loading** (`det_gguf.c`):
- [x] `gguf_get_tensor_q8_0()` - Returns memory-mapped Q8_0 tensor (no dequantization)

**Tests**:
- [x] `test_matmul_q8_0_transposed()` - 0.75% relative error (Q8_0 quantization noise)
- [x] `test_matmul_q8_0_batched()` - 0.05% relative error

**Memory Savings**:
| Model Size | F32 Mode | Q8_0 Mode | Savings |
|------------|----------|-----------|---------|
| 0.5B params | ~2 GB | ~0.5 GB | 1.5 GB |
| 7B params | ~28 GB | ~7.4 GB | 20.6 GB |

**Usage**:
```c
det_set_inference_mode(DET_INFERENCE_Q8_0);  // Enable before loading
DetModel* model = det_model_load("model.gguf");  // Weights stay quantized
```

#### 26.7 Metal Performance Shaders (M5) ✅ COMPLETE
**Goal**: GPU-accelerated inference

- [x] Metal matmul for large weight matrices (transposed-B for projections)
- [x] Metal fused SiLU-multiply for SwiGLU FFN
- [x] Integration into forward pass with automatic Metal/CPU selection
- [x] GPU detection and query functions
- [ ] Metal attention kernel (Q·K^T/√d_k, softmax, V multiply) - DEFERRED
- [ ] Batch prefill (prompt ingestion) - DEFERRED
- [ ] Speculative decoding exploration - DEFERRED

**Performance Results**:
- CPU-only (naive loops): ~2900ms per forward pass
- With Metal: ~375ms per forward pass
- With Metal + Accelerate BLAS: ~353ms per forward pass
- Threshold: METAL_MIN_ELEMENTS = 64*64 for automatic GPU dispatch
- Device: Apple M1 Max verified

**Metal Kernels Implemented**:
- `matmul_transposed_b_f32` - C = A @ B^T (key for projection layers)
- `silu_mul_f32` - out = SiLU(gate) * up (fused SwiGLU activation)
- `rope_f32` - Split-half RoPE application (corrected pairing)

**Files Modified**:
- `src/inference/metal/tensor_shaders.metal` - Added transposed matmul and silu_mul kernels
- `src/inference/metal/tensor_metal.m` - Added C wrappers for new operations
- `src/inference/include/det_tensor_metal.h` - API declarations
- `src/inference/src/det_model.c` - Metal integration + Accelerate BLAS for CPU fallback

#### 26.8 Integration (M6) ✅ REPL + CREATURE INTEGRATION COMPLETE
**Goal**: Drop-in replacement for Ollama

- [x] LLMCreature.ex compatibility layer (unchanged interface)
- [x] Model switching support (LoadNativeModel, EnableNative kernels)
- [ ] Streaming token output
- [x] Graceful fallback to Ollama if native fails (use_native flag)
- [ ] Deprecate Ollama primitives (phase out `llm_call_v2` HTTP calls)

**LLMCreature.ex Enhancements** (Phase 26):
- `use_native` flag: Toggle between native GGUF and Ollama HTTP
- `LoadNativeModel` kernel: One-time GGUF model loading
- `EnableNative` kernel: Switch between modes
- `NativeStatus` kernel: Report GPU and model status
- Modified `Think` kernel with CALL_NATIVE and CALL_OLLAMA proposals
- DET integration: Presence (P) gates native inference

**DET-OS REPL Commands** (det_os_boot.py):
- [x] `native` - Show native inference help
- [x] `native status` - Show model and GPU status
- [x] `native load <path>` - Load GGUF model
- [x] `native enable/disable` - Switch between native and Ollama
- [x] `native generate <prompt>` - Generate text
- [x] `native reset` - Reset KV cache
- [x] `ask` - Auto-routes to native when enabled
- [x] `metal_status()` - Python API for GPU info

#### 26.9 Forward Pass Debugging ✅ COMPLETE

**Status**: Forward pass produces correct logits matching HuggingFace. See `docs/MODEL_DEBUG_LOG.md` for full debugging trace.

**Issues Fixed**:
- [x] Corrupted GGUF file - downloaded correct version from lmstudio-community
- [x] Weight indexing convention - fixed to use `W[i * in_dim + j]` for GGUF row-major storage
- [x] Missing QKV biases - Qwen2 uses biases, now loaded and applied
- [x] RoPE double-application - separated Q and K RoPE loops
- [x] Norm epsilon metadata - added Qwen2-prefixed key fallback
- [x] Tokenizer BPE length check - fixed memory read past string end
- [x] RoPE pairing convention - fixed to split-half (i, i+head_dim/2) matching HuggingFace

**Components Verified Correct**:
- [x] Embedding lookup (max diff 0.0001 from HF)
- [x] RMSNorm (max diff 0.0003 from HF)
- [x] QKV projections (max diff 0.01 from HF)
- [x] RoPE application (split-half pairing)
- [x] Full forward pass (logits RMS diff 0.08 from HF)

**Final Results**:
| Metric | Value |
|--------|-------|
| " Paris" rank | **1** ✅ |
| " Paris" logit | 17.26 (HF: 17.22) |
| Logits RMS diff | 0.08 |
| Logits max diff | 0.48 |

All top 5 predictions match HuggingFace. Small remaining differences are expected from Q8_0 quantization.

#### 26.10 Success Criteria

**MVP**:
- [x] Load and run phi-2 (2.7B params) or qwen2-0.5B ✅ (Qwen2.5-0.5B-Instruct-Q8_0.gguf verified)
- [x] Generate coherent text ✅ (story generation, Q&A working)
- [x] Track F expenditure per generation
- [x] Integrate with existing LLMCreature.ex

**Full Success**:
- [ ] Support llama-architecture models up to 7B
- [ ] Performance within 2x of Ollama throughput
- [x] Quantization support (Q4_K_M, Q8_0) ✅ (Q8_0 verified, QAM implemented)
- [ ] Streaming token generation
- [ ] Full DET physics integration (P-based scheduling)
- [ ] Truthfulness weighting on all outputs
- [ ] Deprecate Ollama dependency

#### 26.11 DET Physics Mapping

| Transformer Concept | DET Concept |
|---------------------|-------------|
| Attention weight | Bond coherence (auditable, not simulated) |
| Softmax temperature | Agency-modulated decisiveness |
| Token sampling | Ur-choice (committed as witness) |
| FLOPS (matmul) | F expenditure |
| Model weights | Creature structure (q) |
| KV cache | Creature memory state |

#### 26.12 Anti-Hallucination Mechanisms

| Pathology | DET Mechanism |
|-----------|---------------|
| Reward hacking | F expenditure tracks real compute |
| False confidence | Agency from structure, not assertion |
| Ungrounded claims | Debt (q) accumulation |
| Hidden attention | Attention is auditable effect |
| Post-hoc justification | Atomic commits, no retroactive changes |

#### 26.13 Semantic Verification (Future)
**Goal**: Detect factual errors and hallucinations that pass process-based truthfulness checks

**Problem**: Current truthfulness (26.6) measures *process* not *content*. A confident
hallucination can score T=0.73 because it was internally consistent and "paid" for claims.

**Approaches to Explore**:

1. **Trace Stability Testing**
   - Re-ask the same question with paraphrasing
   - Hallucinations are unstable (different answer each time)
   - Ground truth is stable (same answer regardless of phrasing)
   - Update `trace_stability` grounding signal based on actual re-generation

2. **Semantic Coherence with Question**
   - Did the answer actually address what was asked?
   - "Schrödinger equation" → physics, not fake UN rankings
   - Requires question-answer alignment scoring

3. **External Verification Creature**
   - FactCheckerCreature.ex that verifies claims
   - For math: bond to CalculatorCreature and compare
   - For facts: could use retrieval or known-fact database

4. **Entropy Anomaly Detection**
   - Hallucinations often have *too-low* entropy (overconfident)
   - Real uncertainty should show in logit distribution
   - Flag outputs where model is "suspiciously certain"

5. **Constraint Propagation**
   - User states constraints ("this is about physics")
   - Violations increase q_claim debt
   - Already partially implemented in GroundingSignals

**DET-Native Approach**:
- Verification is another creature with its own F budget
- Claims create bonds to verification requests
- Unverified claims accumulate debt until verified
- Verification costs F (can't verify everything)

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

*Last Updated: 2026-01-26* | *Phase 26 - Native Model Inference (MVP COMPLETE + KV Cache 26.4 + Truthfulness 26.6 COMPLETE)*
