# DET Local Agency - Development Log

**Project**: DET Local Agency
**Start Date**: 2026-01-17
**Current Phase**: Phase 14 Complete (GPU Metal Backend)

---

## Phase 1: Foundation (Completed)

### 1.1 C Kernel Basics ✅
- **Status**: Complete
- **Files Created**:
  - `src/det_core/include/det_core.h` - Full API header (370 lines)
  - `src/det_core/src/det_core.c` - Implementation (~900 lines)
  - `src/det_core/CMakeLists.txt` - Build system
  - `src/det_core/tests/test_det_core.c` - Test suite
- **Tests**: 22/22 passing
- **Features Implemented**:
  - Node/bond state structures
  - Dual-layer architecture (P-layer: 16 nodes, A-layer: 256 nodes)
  - Dormant pool (3760 nodes) with Beta(2,5) agency distribution
  - Presence computation with coordination load
  - Coherence dynamics with layer-specific parameters
  - 3-axis affect (Valence/Arousal/Bondedness)
  - Dual EMAs for temporal dynamics
  - Self-cluster identification algorithm
  - Gatekeeper with prison regime detection
  - Port protocol for LLM interface
  - Node recruitment/retirement

### 1.2 Python-C Bridge ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/__init__.py`
  - `src/python/det/core.py` - ctypes bindings
  - `src/python/test_det.py` - Python test suite
- **Tests**: 11/11 passing
- **Features**:
  - Full ctypes wrapper for all C functions
  - Pythonic interface with properties
  - State inspection utilities

### 1.3 Ollama Integration ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/llm.py` - LLM interface
- **Features**:
  - `OllamaClient` for API communication
  - `DetIntentPacket` port protocol
  - Intent/domain classification
  - Complexity and risk assessment
  - Affect-modulated temperature

### 1.4 CLI REPL ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/cli.py` - CLI interface
  - `src/python/det_cli.py` - Entry point
- **Features**:
  - Interactive REPL
  - Affect visualization
  - `/state`, `/affect`, `/clear`, `/help`, `/quit` commands
  - Real-time DET state display

### Phase 1 Artifacts
- **Virtual Environment**: `.venv/`
- **Requirements**: `requirements.txt`
- **Build Output**: `src/det_core/build/libdet_core.dylib`

---

## Phase 2: Memory Layer (In Progress)

### 2.1 Memory Domain Architecture ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/memory.py`
- **Features**:
  - `MemoryDomain` enum (8 domains: GENERAL, MATH, LANGUAGE, TOOL_USE, SCIENCE, CODE, REASONING, DIALOGUE)
  - `DomainConfig` with model associations
  - `MemoryEntry` with DET state tracking
  - `DomainRouter` with keyword-based routing
  - `MemoryManager` with persistence
  - Coherence-tracked domain routing
  - Memory consolidation for training data

### 2.2 MLX Training Pipeline ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/training.py`
  - `src/python/test_training.py`
- **Features**:
  - `TrainingConfig`: Configurable LoRA parameters (rank, alpha, dropout, layers)
  - `TrainingExample`: Training data representation with chat format conversion
  - `TrainingDataGenerator`: Generate Q&A pairs from memories and context
    - Domain-specific templates (MATH, CODE, REASONING, etc.)
    - Q&A extraction from content
    - Conversation context extraction
    - JSONL dataset export
  - `LoRATrainer`: MLX-based LoRA fine-tuning
    - Model loading with LoRA layers
    - Training with checkpointing
    - Adapter management (save, load, list, delete)
  - `MemoryRetuner`: High-level integration
    - DET core approval checking before training
    - Memory-to-training-data conversion
    - Session consolidation support
- **Tests**: 22/22 passing (`test_training.py`)

### 2.3 Context Window Management ✅
- **Status**: Complete (in memory.py)
- **Features**:
  - `ContextWindow` class
  - Token budget management
  - Automatic context reduction
  - Memory storage on reduction

### 2.4 Internal Dialogue System ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/dialogue.py`
- **Features**:
  - `InternalDialogue` with reformulation strategies
  - Strategy selection based on affect/coherence
  - Multi-turn internal thinking (`/think` command)
  - Escalation handling
  - Dialogue history and summaries

### Phase 2 Integration ✅
- **Status**: Complete
- **Files Updated**:
  - `src/python/det/__init__.py` - Exports new modules
  - `src/python/det/cli.py` - Integrated memory and dialogue
- **Tests**: 10/10 passing (`test_phase2.py`)
- **New CLI Commands**:
  - `/memory` - Show memory domain statistics
  - `/store <text>` - Store a memory
  - `/recall <query>` - Recall memories
  - `/think <topic>` - Internal thinking

---

## Phase 3: Agentic Operations (Completed)

### 3.1 Sandboxed Bash Environment ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/sandbox.py`
- **Features**:
  - `CommandAnalyzer` with risk assessment (SAFE → CRITICAL)
  - `BashSandbox` with permission levels
  - Resource limits (CPU, memory, time, output)
  - Network policy enforcement
  - Path whitelisting/blacklisting
  - DET integration for affect-aware approval
  - `FileOperations` for safe file access

### 3.2 Task Management ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/tasks.py`
- **Features**:
  - `TaskManager` with persistence
  - `Task` and `TaskStep` structures
  - LLM-based task decomposition
  - Checkpoint/resume capability
  - Priority levels
  - DET affect tracking at creation

### 3.3 Timer System ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/timer.py`
- **Features**:
  - `TimerSystem` with threaded execution
  - Schedule types: ONCE, INTERVAL, DAILY
  - Built-in callbacks: health_check, memory_consolidation, det_maintenance
  - Event persistence
  - Pause/resume/cancel events

### 3.4 Code Execution Loop ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/executor.py`
- **Features**:
  - `CodeExecutor` with Write→Compile→Test→Iterate cycle
  - `ErrorInterpreter` with pattern-based suggestions
  - `LanguageRunner` supporting Python, JS, TS, C, C++, Rust, Go, Bash
  - Automatic code fixing with LLM
  - Session management

### Phase 3 Integration ✅
- **Tests**: 17/17 passing (`test_phase3.py`)
- **Version**: 0.3.0

---

## Phase 4: Advanced DET Features (Completed)

### 4.1 Complete DET Dynamics ✅
- **Status**: Complete
- **Files Updated**:
  - `src/det_core/include/det_core.h` - New API functions
  - `src/det_core/src/det_core.c` - Extended dynamics (~200 lines)
  - `src/python/det/core.py` - Python bindings
- **Features**:
  - **Bond Momentum (π)**: Directional memory of information flow
  - **Angular Momentum (L)**: Phase rotation tendency with coupling
  - **Grace Injection**: Boundary recovery mechanism for resource replenishment
  - **Structural Debt Updates**: Enhanced debt accumulation and decay
  - **Phase Velocity (dθ/dt)**: Rate of phase change tracking

### 4.2 Learning via Recruitment ✅
- **Status**: Complete
- **Features**:
  - `can_learn(complexity, domain)`: Check recruitment feasibility
  - `learning_capacity()`: Compute available learning capacity
  - `activate_domain(name, num_nodes, coherence)`: Recruit dormant nodes
  - `transfer_pattern(source, target, strength)`: Cross-domain learning
  - Dormant node recruitment with coherence initialization
  - Division criteria based on self-cluster state

### 4.3 Emotional State Integration ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/emotional.py`
- **Features**:
  - `EmotionalMode`: EXPLORATION, FOCUSED, RECOVERY, VIGILANT, NEUTRAL
  - `BehaviorModulation`: State-dependent parameters
    - LLM temperature multiplier
    - Risk tolerance
    - Exploration rate
    - Retry patience
    - Complexity threshold
  - `RecoveryState`: Strain tracking and recovery scheduling
  - Mode determination from V/A/B affect
  - Curiosity spike detection
  - Grace injection during recovery

### 4.4 Multi-Session Support ✅
- **Status**: Complete
- **Features**:
  - **State Serialization**: Binary save/load of full DET core state
  - `save_state() / load_state()`: In-memory serialization
  - `save_to_file() / load_from_file()`: File persistence
  - `MultiSessionManager`: Session lifecycle management
  - `SessionContext`: Per-session state tracking
  - Cross-session memory via shared DET core
  - Session topic tracking

### Phase 4 Integration ✅
- **Tests**: 25/25 passing (`test_phase4.py`)
- **Version**: 0.4.1

---

## Phase 5: Production Readiness (In Progress)

### 5.1 Multi-LLM Routing ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/routing.py`
  - `src/python/test_phase5.py`
- **Features**:
  - `ModelConfig`: Model configuration with domain mapping
    - Name, display name, domains list
    - Priority for routing decisions
    - Default temperature and system prompts
    - Performance hints (is_fast, supports_code, supports_math)
  - `ModelPool`: Model pool with health monitoring
    - Model registration and client management
    - Periodic health checking (threaded)
    - Latency tracking
    - Graceful degradation on failures
  - `LLMRouter`: Domain-aware request routing
    - Automatic domain detection from text
    - Priority-based model selection
    - Fallback to general model
    - DET affect-modulated temperature
  - `MultiModelInterface`: High-level pipeline
    - DET gatekeeper integration
    - Full request processing
  - Default models configured:
    - `general`: Llama 3.2 3B (GENERAL, DIALOGUE, LANGUAGE)
    - `math`: DeepSeek Math 7B (MATH)
    - `code`: Qwen 2.5 Coder 7B (CODE, TOOL_USE)
    - `reasoning`: DeepSeek R1 8B (REASONING, SCIENCE)
- **Tests**: 28/28 passing (`test_phase5.py`)

### 5.2 Sleep/Consolidation Cycle ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/consolidation.py`
- **Features**:
  - `IdleDetector`: Activity tracking and idle detection
    - Configurable idle threshold
    - Idle duration tracking
  - `ConsolidationConfig`: Cycle configuration
    - Timing (idle threshold, interval, max duration)
    - Training settings (min examples, max domains)
    - DET integration (valence gating, grace injection)
  - `ConsolidationCycle`: Cycle state tracking
    - Phases: MEMORY_SCAN → DATA_GENERATION → MODEL_TRAINING → GRACE_INJECTION → VERIFICATION
    - Domain and job tracking
  - `ConsolidationManager`: Full lifecycle management
    - Automatic idle monitoring (threaded)
    - Scheduled consolidation via TimerSystem
    - MLX training integration via MemoryRetuner
    - DET affect-gated training (positive valence required)
    - Grace injection during recovery
  - `setup_consolidation()`: Convenience setup function
- **Tests**: 18 new tests (46 total Phase 5)

### 5.3 Network Integration (Preliminary) ✅
- **Status**: Complete (Foundation)
- **Files Created**:
  - `src/python/det/network.py`
- **Features**:
  - **Protocol Definitions**:
    - `MessageType`: Binary message types (HEARTBEAT, STATE_UPDATE, AFFECT_UPDATE, STIMULUS_INJECT, GRACE_INJECT, etc.)
    - `NodeType`: Node types (ESP32, RASPBERRY_PI, PYTHON_AGENT, SENSOR, ACTUATOR)
    - `NodeStatus`: Connection states (CONNECTING, CONNECTED, DISCONNECTED, ERROR, SLEEPING)
    - `DETMessage`: Binary protocol with magic bytes, checksums, serialization/deserialization
    - `NodeInfo`: Node metadata with capabilities and DET assignments
  - **Abstract Interfaces**:
    - `Transport`: Abstract transport interface (connect, send, receive)
    - `ExternalNode`: Abstract interface for external DET nodes (state sync, messaging)
  - **Stub Implementations**:
    - `StubTransport`: Testing transport with message injection
    - `StubExternalNode`: Simulated external node for development
  - **Network Registry**:
    - `NetworkRegistry`: Node registration, discovery, and management
    - State broadcasting to all connected nodes
    - Message handler registration
    - Status reporting
  - **Future Integration Points**:
    - `SerialTransport`: Placeholder for ESP32/serial communication
    - `ESP32Node`: Placeholder for ESP32 hardware node
  - `create_stub_network()`: Convenience function for testing
- **Tests**: 29 new tests (75 total Phase 5)
- **Note**: This is a preliminary/foundation implementation. Full ESP32/serial integration deferred to future module.

### Phase 5 Integration ✅
- **Tests**: 75/75 passing (`test_phase5.py`)
- **Version**: 0.5.2
- **Total Tests**: 182/182 (22 C + 11 Bridge + 10 Phase2 + 22 Phase2.2 + 17 Phase3 + 25 Phase4 + 75 Phase5)

---

## Phase 6: Development Tools (In Progress)

### 6.1 CLI Test Harness ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/harness.py`
  - `src/python/test_phase6.py`
- **Features**:
  - **HarnessController**: Programmatic control for debugging
    - Resource injection (F, q, agency)
    - Bond manipulation (create, destroy, set coherence)
    - Time controls (pause, resume, step, speed)
    - State inspection (nodes, bonds, aggregates, affect)
    - Snapshot/restore for state comparison
    - Event logging with callbacks
    - Watchers for conditional triggers
  - **HarnessCLI**: Interactive cmd-based CLI
    - Status commands (status, node, bond, self, affect, emotional)
    - Injection commands (inject_f, inject_q, inject_all, set_agency)
    - Bond commands (create_bond, destroy_bond, set_coherence)
    - Time commands (step, pause, resume, speed, run, stop)
    - Snapshot commands (snapshot, restore, snapshots, delete_snapshot)
    - Event commands (events, clear_events)
  - `create_harness()`: Convenience factory function
  - `run_harness_cli()`: Launch interactive CLI
- **Tests**: 35/35 passing (`test_phase6.py`)
- **Note**: Foundation for future webapp-based visualization

### 6.2 Web Visualization ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/webapp/__init__.py`
  - `src/python/det/webapp/server.py`
  - `src/python/det/webapp/api.py`
  - `src/python/det/webapp/templates/index.html`
- **Features**:
  - **FastAPI Server**: Async web server with WebSocket support
    - REST API for status, nodes, bonds, control
    - WebSocket endpoint for real-time state updates
    - Lifespan management for background tasks
  - **DETStateAPI**: Clean API wrapper for frontend
    - Full state retrieval (nodes, bonds, aggregates, affect)
    - Visualization data with 3D positions and colors
    - Control methods (step, pause, resume, speed)
    - Injection and snapshot methods
  - **3D Visualization**: Three.js-based interactive viewer
    - Nodes as spheres with affect-based coloring
    - Bonds as lines with coherence-based opacity
    - Self-cluster highlighting
    - Orbit controls for navigation
  - **Dashboard UI**: Real-time stats and controls
    - Status cards (presence, coherence, resource, debt)
    - Affect bars (valence, arousal, bondedness)
    - Time controls (step, pause, speed slider)
    - Node list and event log
  - `create_app()`: FastAPI app factory
  - `run_server()`: Convenience launcher (uvicorn)
- **Tests**: 11 new tests (46 total Phase 6)
- **Dependencies**: fastapi, uvicorn, websockets, jinja2

### 6.3 Advanced Interactive Probing ✅
- **Status**: Complete
- **Files Modified**:
  - `src/python/det/harness.py` - Added advanced probing methods
  - `src/python/test_phase6.py` - Added 11 new tests
- **Features**:
  - **Escalation Control**:
    - `trigger_escalation(node)`: Force escalation on a node
  - **Grace Management**:
    - `inject_grace(node, amount)`: Inject grace into a node
    - `inject_grace_all(amount)`: Inject grace into all nodes that need it
    - `get_total_grace_needed()`: Get total grace deficit across all nodes
  - **Learning & Domains**:
    - `get_learning_capacity()`: Get available agency for recruitment
    - `can_learn(complexity, domain)`: Check if learning is possible
    - `activate_domain(name, num_nodes, coherence)`: Activate a new domain
    - `transfer_pattern(source, target, strength)`: Transfer pattern between domains
  - **Gatekeeper**:
    - `evaluate_request(tokens, domain, retry_count)`: Evaluate request through gatekeeper
  - **CLI Commands**:
    - `escalate <node>` - Trigger escalation
    - `grace <node> <amount>` - Inject grace
    - `grace_all <amount>` - Inject grace to all needing nodes
    - `grace_needed` - Show total grace needed
    - `learning [complexity]` - Show learning capacity
    - `activate_domain <name> <n> [c]` - Activate domain
    - `transfer <src> <tgt> [s]` - Transfer pattern
    - `gatekeeper <tokens...>` - Evaluate request
- **Tests**: 11 new tests (57 total Phase 6)

### 6.4 Metrics and Logging ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/metrics.py` - Metrics collector and profiler
  - `setup_det.py` - Automated setup script
  - `GETTING_STARTED.md` - Getting started documentation
- **Files Modified**:
  - `src/python/det/webapp/server.py` - Added metrics API endpoints
  - `src/python/test_phase6.py` - Added 17 new tests
- **Features**:
  - **MetricsCollector**:
    - Sample DET state (P, C, F, q, V/A/B, tick times)
    - Rolling window storage with configurable limits
    - Timeline data retrieval for any field
    - Dashboard with current values and trends
    - Statistical summary (min, max, mean, stdev)
  - **DET Event Logging**:
    - Escalation, compilation, recruitment events
    - Bond formed/broken events
    - Prison regime detection
    - Grace injection, domain activation
    - Gatekeeper decisions
    - Event callbacks and filtering
  - **Performance Profiler**:
    - Tick timing (avg, min, max, p50, p95)
    - Step-level timing within ticks
    - Memory usage tracking
    - Full profiling report
  - **Webapp API Endpoints**:
    - `GET /api/metrics/dashboard` - Dashboard data
    - `GET /api/metrics/samples` - Recent samples
    - `GET /api/metrics/timeline/{field}` - Timeline data
    - `GET /api/metrics/events` - Event log
    - `GET /api/metrics/statistics` - Statistical summary
    - `GET /api/metrics/profiling` - Performance profiling
  - **Setup Script** (`setup_det.py`):
    - System requirements check
    - Virtual environment creation
    - Python dependencies installation
    - C kernel build
    - Ollama model downloads
    - Verification tests
  - **Getting Started Guide** (`GETTING_STARTED.md`):
    - Prerequisites and installation
    - Automated and manual setup
    - Running CLI and web visualization
    - Configuration options
    - API reference
    - Troubleshooting
- **Tests**: 17 new tests (74 total Phase 6)

### Phase 6 Integration ✅
- **Tests**: 74/74 passing (`test_phase6.py`)
- **Version**: 0.6.4
- **Total Tests**: 256/256 (22 C + 11 Bridge + 10 Phase2 + 22 Phase2.2 + 17 Phase3 + 25 Phase4 + 75 Phase5 + 74 Phase6)

---

## Phase 7: Existence-Lang (Complete)

### 7.1 Language Core ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/lang/tokens.py` - Lexer with ~60 token types
  - `src/python/det/lang/ast_nodes.py` - AST node definitions
  - `src/python/det/lang/parser.py` - Recursive descent parser
  - `src/python/det/lang/semantic.py` - Semantic analysis
  - `src/python/det/lang/transpiler.py` - Python transpiler
  - `src/python/det/lang/runtime.py` - Runtime support
  - `src/python/det/lang/errors.py` - Error handling
  - `src/python/det/lang/repl.py` - Interactive REPL
- **Features**:
  - Four equalities: `:=` (alias), `==` (measure), `=` (reconcile), `≡` (covenant)
  - Temporal semantics: `if_past`, `repeat_past`, `forecast`
  - Creature/kernel/presence structure
  - Phase blocks: READ, PROPOSE, CHOOSE, COMMIT
  - Witness binding with `::=`

### 7.2 Standard Library ✅
- **Files Created**:
  - `src/python/det/lang/stdlib/primitives.py` - Transfer, Diffuse, Distinct, Compare
  - `src/python/det/lang/stdlib/arithmetic.py` - AddSigned, SubSigned, MulByPastToken
  - `src/python/det/lang/stdlib/grace.py` - GraceOffer, GraceFlow
- **Tests**: 36/36 passing (`test_phase7.py`)

---

## Phase 8: EIS VM (Complete)

### 8.1 Existence Instruction Set ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/eis/vm.py` - Virtual machine (~80 opcodes)
  - `src/python/det/eis/assembler.py` - Assembly language
  - `src/python/det/eis/encoding.py` - Value encoding
  - `src/python/det/eis/memory.py` - Memory management
  - `src/python/det/eis/phases.py` - Execution phases
  - `src/python/det/eis/registers.py` - Register definitions
  - `src/python/det/eis/types.py` - Type system

### 8.2 Compiler ✅
- **Files Created**:
  - `src/python/det/lang/eis_compiler.py` - Existence-Lang → EIS bytecode
  - `src/python/det/lang/eis_native.py` - Native code generation

### 8.3 Native Compiler ✅
- **Features**:
  - ARM64 code generation
  - x86_64 code generation
  - IR optimization passes
  - JIT compilation support
- **Tests**: 50/50 passing (`test_phase8.py`)

---

## Phase 9: DET-OS (Complete)

### 9.1 Kernel Core ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/os/kernel.py` - Kernel core
  - `src/python/det/os/scheduler.py` - Presence-based scheduling
  - `src/python/det/os/allocator.py` - F-conserving memory
  - `src/python/det/os/gatekeeper.py` - Agency-based access control
  - `src/python/det/os/ipc.py` - Bond-based IPC
  - `src/python/det/os/creature.py` - Creature model

### 9.2 Existence Kernel ✅
- **Files Created**:
  - `src/existence/kernel.ex` - Kernel in Existence-Lang
  - `src/existence/physics.ex` - DET physics layer
  - `src/python/det/os/existence/bootstrap.py` - Boot sequence
  - `src/python/det/os/existence/runtime.py` - Runtime execution
- **Tests**: All passing (`test_phase9.py`, `test_phase9_5.py`)

---

## Phase 10: Substrate v2 & Physics Layer (Complete)

### 10.1 DET-Aware Substrate ✅
- **Status**: Complete
- **Files Created**:
  - `src/substrate/include/eis_substrate_v2.h` - ~50 opcodes
  - `src/substrate/include/substrate_types.h` - Type definitions
  - `src/substrate/include/effect_table.h` - Effect table
  - `src/substrate/src/eis_substrate_v2.c` - Implementation (~1300 lines)
  - `src/substrate/tests/test_substrate_v2.c` - Test suite
- **Key Features**:
  - Phase-based execution: READ → PROPOSE → CHOOSE → COMMIT
  - Proposal system with scoring and effects
  - Effect table: XFER_F, DIFFUSE, SET_F, ADD_F, etc.
  - Typed references: NodeRef, BondRef, PropRef, ChoiceRef
  - SoA memory layout for GPU readiness
- **Tests**: 31/31 passing

### 10.2 Physics Layer ✅
- **Files Created**:
  - `src/existence/physics.ex` - DET physics in Existence-Lang
  - `src/python/det/os/existence/physics_bridge.py` - Python bridge to substrate
- **Physics Kernels**:
  - `Transfer` - Antisymmetric resource movement
  - `Diffuse` - Symmetric flux exchange
  - `Compare` - Trace measurement
  - `Distinct` - Create distinction (ur-choice)
  - `Reconcile` - Attempted unification
  - `GraceFlow` - Complete grace protocol
- **Architecture**:
  ```
  physics.ex (fundamental laws)
      ↓ imported by
  kernel.ex (OS services)
      ↓ bridges via
  physics_bridge.py (Python)
      ↓ executes on
  substrate v2 (C)
  ```

### 10.3 Project Cleanup ✅
- **Removed**: Empty directories (config, examples, models, visualization, etc.)
- **Reorganized**: Moved explorations to `docs/explorations/`
- **Tests**: All passing (87 substrate + 37 det_core + 36 phase7 + 50 phase8 + phase9 + phase9.5)

---

## Phase 11: Substrate v2 DET-Honesty Improvements (Complete)

### 11.1 Phase Legality Enforcement ✅
- **Status**: Complete
- **Change**: Instructions now validated against current phase
  - Loads (LDN, LDB, LDNB) only in READ phase
  - Proposals (PROP_NEW, PROP_SCORE) only in PROPOSE phase
  - Stores (STN, STB) only in COMMIT phase
  - CHOOSE only in CHOOSE phase
- **Benefit**: Prevents "present" leaking via same-tick writes + reads

### 11.2 Deterministic RNG from Trace ✅
- **Status**: Complete
- **Change**: Random seed derived from trace state, not global RNG
  - `seed = hash(r, k, tick, lane_id)` for node lanes
  - Uses MurmurHash3 finalizer for fast mixing
- **Benefit**: Reproducibility and parallel safety

### 11.3 Bond Ownership (Effect Deduplication) ✅
- **Status**: Complete
- **Change**: Added `LaneOwnershipMode` (NONE, NODE, BOND)
  - NODE mode: Only lane with `lane_id == min(src, dst)` can emit XFER_F
  - BOND mode: Only lane with `lane_id == bond_id` can emit bond effects
- **Benefit**: Prevents double-counting in parallel execution

### 11.4 Native-Endian Bytecode & Predecoding ✅
- **Status**: Complete
- **Change**:
  - Added `substrate_decode_le` / `substrate_encode_le` for little-endian
  - Added `substrate_predecode()` to pre-parse entire program
  - PC is instruction index in predecoded mode (faster lookup)
- **Benefit**: Eliminates decode overhead from hot loop

### 11.5 Threaded Dispatch (Computed Goto) ✅
- **Status**: Complete
- **Change**: Added `substrate_run_fast()` using GCC/Clang computed goto
  - Dispatch table with labels for each opcode
  - `DISPATCH_NEXT()` macro jumps directly to next handler
  - Fallback to `substrate_run()` for non-GCC compilers
- **Benefit**: 2-3x faster dispatch vs switch statement

### Phase 11 Test Summary
- **New Tests**: 9 additional tests
- **Total**: 40/40 passing for substrate v2

---

## Phase 12: Compiler v2 Update (Complete)

### 12.1 EIS Encoding v2 Opcodes ✅
- **Status**: Complete
- **Files Modified**:
  - `src/python/det/eis/encoding.py` - Added v2 opcodes
- **New Constants**:
  - Phase control: `V2_PHASE_R` (0x04), `V2_PHASE_P` (0x05), `V2_PHASE_C` (0x06), `V2_PHASE_X` (0x07)
  - Proposals: `V2_PROP_NEW` (0x50), `V2_PROP_SCORE` (0x51), `V2_PROP_EFFECT` (0x52), `V2_PROP_ARG` (0x53), `V2_PROP_END` (0x54)
  - Choose/Commit: `V2_CHOOSE` (0x60), `V2_COMMIT` (0x61), `V2_WITNESS` (0x62)
  - Stores: `V2_STN` (0x70), `V2_STB` (0x71), `V2_STT` (0x72)
  - Typed loads: `V2_LDN` (0x10), `V2_LDB` (0x11), `V2_LDI` (0x13), `V2_LDI_F` (0x14)
  - Register ops: `V2_MOV` (0x20), `V2_MOVR` (0x21), `V2_MOVT` (0x22), `V2_TSET` (0x23), `V2_TGET` (0x24)
  - Comparison: `V2_CMP` (0x40), `V2_CMPE` (0x41), `V2_TEQ` (0x42), `V2_TNE` (0x43)
  - I/O: `V2_IN` (0x80), `V2_OUT` (0x81), `V2_EMIT` (0x82), `V2_POLL` (0x83)
  - System: `V2_RAND` (0xF0), `V2_SEED` (0xF1), `V2_LANE` (0xF2), `V2_TIME` (0xF3), `V2_DEBUG` (0xFE)

### 12.2 Phase-Aware Code Generation ✅
- **Status**: Complete
- **Files Modified**:
  - `src/python/det/lang/eis_compiler.py` - Added v2 mode
- **Changes**:
  - Added `use_v2` parameter to `EISCompiler` class
  - New `_compile_phase_block()` method emits `V2_PHASE_*` opcodes
  - Updated `_compile_kernel()` to use phase-aware compilation
  - Added `_compile_choice()` and `_compile_commit()` for v2 opcodes
  - Added `_emit_raw()` helper for raw opcode emission

### 12.3 Proposal Compilation v2 ✅
- **Status**: Complete
- **Changes**:
  - Updated `_compile_proposal()` for v2 mode
  - New sequence: `V2_PROP_NEW` → `V2_PROP_SCORE` → `V2_PROP_EFFECT` → `V2_PROP_ARG` → `V2_PROP_END`
  - Added `_compile_proposal_effect_v2()` for effect encoding
  - Effect types: TRANSFER (1), STORE (2), KERNEL_BASE (0x10+)

### 12.4 V2 Compiler Tests ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/test_compiler_v2.py` - V2 compiler test suite
- **Tests**:
  - Opcode value verification (matches C substrate)
  - Phase opcode mapping
  - Compiler mode flags (v1/v2)
  - Raw instruction emission
  - Bytecode encoding/decoding roundtrip
  - Full compilation flow integration
- **Total**: 11/11 tests passing

### Phase 12 Summary
- **V2 opcodes**: Match C substrate `eis_substrate_v2.h` exactly
- **Compilation modes**: V1 (default) maintains compatibility, V2 for substrate v2
- **Phase-aware**: Emits `V2_PHASE_R/P/C/X` at phase boundaries
- **Proposal sequence**: Full v2 proposal instruction sequence support

---

## Current Test Summary

| Component | Tests | Status |
|-----------|-------|--------|
| Compiler v2 | 11/11 | ✅ |
| Substrate v2 | 40/40 | ✅ |
| Substrate v1 | 56/56 | ✅ |
| Metal Backend | 9/9 | ✅ |
| DET Core | 37/37 | ✅ |
| Phase 7 (Lang) | 36/36 | ✅ |
| Phase 8 (EIS) | 50/50 | ✅ |
| Phase 9 (OS) | All | ✅ |
| Phase 9.5 (Kernel) | All | ✅ |
| Phase 14 (Metal) | 8/8 | ✅ |

---

## Phase 13: Full Existence-Lang Compilation (Complete)

### 13.1 Parser Enhancements ✅
- **Status**: Complete
- **Files Modified**:
  - `src/python/det/lang/parser.py` - Extended parser
  - `src/python/det/lang/tokens.py` - New tokens
  - `src/python/det/lang/ast_nodes.py` - New AST nodes
- **Features**:
  - `CONTEXTUAL_KEYWORDS` class constant for keywords usable as identifiers
  - `_check_name_like()` and `_expect_name()` helpers
  - `AS` keyword for loop variable binding (`repeat_past(N) as i`)
  - Array literals (`[a, b, c]`)
  - Bitwise OR operator (`|`)
  - Statements inside proposal blocks
  - Contextual keywords: `KERNEL`, `PRESENCE`, `CREATURES`, `CREATURE`, `INIT`, `CHANNEL`, etc.

### 13.2 physics.ex Parsing ✅
- **Status**: Complete
- **Result**: Parses 13 kernels successfully
  - Transfer, Diffuse, Compare, Distinct, Reconcile
  - ComputePresence, CoherenceDecay, CoherenceStrengthen
  - GraceOffer, GraceAccept, GraceFlow
  - Add, Multiply
- **Limitation**: Compilation hits register allocation limits

### 13.3 kernel.ex Parsing ✅
- **Status**: Complete
- **Result**: Parses 2 declarations successfully
  - Creature: KernelCreature (with 10 nested kernels)
  - Presence: DET_OS
- **Nested Kernels**: Schedule, Allocate, Free, Send, Receive, Gate, Grace, Spawn, Kill, Tick

### 13.4 End-to-End Integration ✅
- **Status**: Complete
- **Verified**:
  - Source → Parser → AST → Compiler → EIS v2 bytecode pipeline works
  - V2 phase opcodes (V2_PHASE_P, V2_PROP_NEW) appear in bytecode
  - Substrate v2 tests: 40/40 passing
  - Compiler v2 tests: 11/11 passing
  - Phase 7 tests: 36/36 passing

### 13.5 Register Allocation Fix ✅
- **Issue**: Original allocator had only 16 scalar, 8 ref, 8 token registers
- **Solution**: Enhanced register allocator with:
  - Expanded to 32 registers per class (matches instruction encoding 5-bit limit)
  - Expanded substrate to 64/32/32 registers for future use
  - Scope-based register lifetime tracking
  - Automatic register freeing at phase/proposal boundaries
  - Spill slot infrastructure (for future overflow handling)
  - Class-local register indices for proper instruction encoding
- **Result**: physics.ex (13 kernels) and kernel.ex (1 creature) now compile successfully

### Phase 13 Summary
- Both physics.ex and kernel.ex parse and compile successfully
- 13 physics kernels: Transfer, Diffuse, Compare, Distinct, Reconcile, etc.
- Kernel creature: KernelCreature with nested OS kernels
- Foundation ready for GPU backend and hardware acceleration

---

## Phase 14: GPU Backend with Metal Compute Shaders (Complete)

### 14.1 Metal Shader Core ✅
- **Status**: Complete
- **Files Created**:
  - `src/substrate/metal/substrate_shaders.metal` - Metal compute kernels (~800 lines)
- **Kernels**:
  - `init_lanes` - Initialize lane state and registers
  - `phase_read` - READ phase execution (load trace values)
  - `phase_propose` - PROPOSE phase execution (emit proposals)
  - `phase_choose` - CHOOSE phase execution (deterministic selection)
  - `phase_commit` - COMMIT phase execution (apply effects)
  - `compute_presence` - Derived value computation
- **Features**:
  - All 40+ opcodes implemented in Metal shading language
  - Atomic operations for parallel-safe effects
  - MurmurHash3-based deterministic RNG
  - SoA buffer layout for coalesced access

### 14.2 Objective-C Bridge ✅
- **Status**: Complete
- **Files Created**:
  - `src/substrate/metal/metal_backend.m` - Objective-C implementation (~700 lines)
- **Features**:
  - `SubstrateMetal` class for GPU resource management
  - Pipeline state creation for all phase kernels
  - Buffer allocation with shared storage mode
  - State upload/download methods
  - Phase and tick execution

### 14.3 C API Header ✅
- **Status**: Complete
- **Files Created**:
  - `src/substrate/include/substrate_metal.h` - C API header (~150 lines)
- **API Functions**:
  - Lifecycle: `sub_metal_create()`, `sub_metal_destroy()`
  - State transfer: `sub_metal_upload_nodes()`, `sub_metal_download_nodes()`, etc.
  - Execution: `sub_metal_execute_phase()`, `sub_metal_execute_tick()`
  - Configuration: `sub_metal_set_lane_mode()`, `sub_metal_set_seed()`
  - Query: `sub_metal_device_name()`, `sub_metal_memory_usage()`

### 14.4 C Wrapper ✅
- **Status**: Complete
- **Files Created**:
  - `src/substrate/src/substrate_metal.c` - Stub for non-Apple platforms
- **Features**:
  - Stub implementations return `SUB_METAL_ERR_NO_DEVICE`
  - Allows cross-platform compilation

### 14.5 Build System ✅
- **Status**: Complete
- **Files Created**:
  - `src/substrate/metal/CMakeLists.txt` - Metal build rules
- **Files Modified**:
  - `src/substrate/CMakeLists.txt` - Added Metal subdirectory
- **Build Features**:
  - Shader compilation with `xcrun metal`
  - Library linking with `xcrun metallib`
  - ARC-enabled Objective-C compilation
  - Automatic metallib deployment

### 14.6 Python Bindings ✅
- **Status**: Complete
- **Files Created**:
  - `src/python/det/metal.py` - Python ctypes bindings (~500 lines)
- **Classes**:
  - `MetalBackend` - High-level GPU interface
  - `NodeArraysHelper` - Helper for node state management
  - `BondArraysHelper` - Helper for bond state management
- **Features**:
  - Automatic library discovery
  - Type-safe ctypes signatures
  - Convenient helper classes for state management
  - Availability checking

### 14.7 Test Suite ✅
- **Status**: Complete
- **Files Created**:
  - `src/substrate/tests/test_metal.c` - C test suite
  - `src/python/test_phase14.py` - Python test suite
- **Tests**:
  - Availability detection
  - Context creation/destruction
  - Node upload/download roundtrip
  - Bond upload/download roundtrip
  - Program upload
  - Single tick execution
  - Multiple tick execution
  - Performance benchmarking

### Phase 14 Summary
- **Architecture**: Metal compute shaders executing EIS substrate on GPU
- **Phase Mapping**: Each DET phase maps to a GPU kernel dispatch
- **Memory Layout**: SoA buffers for coalesced GPU memory access
- **Atomics**: Effect application uses Metal atomics for parallel safety
- **Performance Target**: ~20x speedup over CPU for large graphs
- **Version**: 0.14.0

---

## DET-OS CLI Integration (2026-01-24)

### LLM as Creature ✅
- **Files Created**:
  - `src/python/det_os_cli.py` - DET-OS CLI with LLM as creature
  - `src/python/det/os/creatures/base.py` - CreatureWrapper base class
  - `src/python/det/os/creatures/memory.py` - MemoryCreature for store/recall
  - `src/python/test_det_os_cli.py` - Integration test suite (13 tests)

- **Architecture**:
  ```
  User Input
      ↓
  LLMCreature (F, a, bonds)
      ↓ bond
  MemoryCreature (stores/recalls)
      ↓
  DET-OS Kernel (kernel.ex)
      ↓
  Substrate v2 (C/Metal)
  ```

- **Features**:
  - LLMCreature with resource (F) depletion per token
  - Agency (a) modulates LLM temperature
  - Creature-to-creature bonding via channels
  - MemoryCreature with store/recall protocol
  - Bond-based IPC with coherence tracking
  - Commands: /store, /recall, /memory, /state, /bonds, /inject

### Bond Coherence Stability Fix ✅
- **Issue**: Bond coherence was decaying too aggressively during idle time
  - Coherence dropped from 1.0 to 0.0 after ~10 seconds of user inactivity
  - Caused /store and /recall commands to fail
- **Fix**: Perfect bonds (coherence >= 0.99) no longer decay from idleness
  - Imperfect bonds only decay after 30 seconds (not 1 second)
  - Decay rate reduced 10x for stability

### Enhanced Memory System ✅
- **Memory Types**:
  - `fact` - Factual information (names, dates, technical details)
  - `preference` - User preferences and likes/dislikes
  - `instruction` - Standing instructions (highest priority)
  - `context` - Conversation context
  - `episode` - Episode summaries

- **Features**:
  - Importance scoring (1-10 scale)
  - Type-weighted recall (instructions prioritized)
  - LLM-driven memory extraction (extracts facts/preferences from conversation)
  - Recency and access-count boosting
  - Instructions never pruned

- **New Commands**:
  - `/store [type:importance] text` - Store with type and importance
  - `/instruct <text>` - Quick store as instruction (importance=9)
  - `/memories [type]` - List memories, optionally filtered by type

- **Tests**: 16/16 passing (3 new enhanced memory tests)

### Existence-Lang Creature Types ✅
- **File Created**: `src/existence/creatures.ex` (~900 lines)
- **Architecture**: Creatures defined in Existence-Lang, Python wrappers for CLI

**MemoryCreature** (creatures.ex + memory.py):
- Store/recall memories via bond IPC protocol
- Memory types: fact, preference, instruction, context, episode
- Type-weighted recall with importance scoring

**ToolCreature** (creatures.ex + tool.py):
- Execute commands in sandboxed environment
- Risk analysis: safe/moderate/high/critical
- Agency thresholds: a >= 0.3 (safe) to a >= 0.9 (critical)
- Resource cost: base + CPU + memory usage

**ReasonerCreature** (creatures.ex + reasoner.py):
- Chain-of-thought reasoning generation
- Agency-based depth: a=0.7 -> max 7 steps
- Step types: analyze, infer, conclude
- Optional external reasoning function (LLM)

**PlannerCreature** (creatures.ex + planner.py):
- Task decomposition with dependencies
- Action types: think, execute, store, recall, reason
- Dependency graph generation
- Constraint-aware planning

- **Tests**: 23/23 passing (7 new creature tests)

### Creature Loading System ✅
- **Files Created**:
  - `src/python/det/os/creatures/loader.py` (~630 lines)

- **Architecture**:
  - `CreatureLoader` - Central loader for managing creature lifecycle
  - `LoadMode` - Enum: BUILTIN, JIT, BYTECODE
  - `CreatureSpec` - Metadata for loadable creatures
  - `LoadedCreature` - Registry entry for loaded creatures

- **Loading Methods**:
  1. **Built-in** (`/load memory`): Python wrapper creatures
  2. **JIT** (`/load /path/to/custom.ex`): Compile .ex at runtime
  3. **Bytecode** (`/load /path/to/creature.exb`): Pre-compiled bytecode

- **Bytecode Format (.exb)**:
  - Header: MAGIC (DETC) + VERSION + FLAGS + METADATA_LEN + CODE_LEN
  - Metadata: JSON blob with creature spec
  - Code: EIS bytecode instructions
  - Supports caching of JIT-compiled creatures

- **CLI Commands**:
  - `/creatures` - List available and loaded creatures
  - `/load <name|path>` - Load by name or .ex/.exb path
  - `/unload <name>` - Unload a creature
  - `/bond <name>` - Bond loaded creature with LLM
  - `/compile <path>` - Compile .ex to .exb bytecode

- **Features**:
  - Automatic creature discovery from search paths
  - Source hash-based cache invalidation
  - Creature registry with load mode tracking
  - Bidirectional bond creation

### Phase 15: EIS Interpreter & Kernel Execution ✅
- **Status**: Complete
- **Files Created/Modified**:
  - `src/python/det/eis/creature_runner.py` (NEW - ~300 lines)
  - `src/python/det/lang/eis_compiler.py` (Modified - kernel compilation)

- **EIS VM** (already existed in `det/eis/`):
  - Full instruction decoder for 4-byte EIS instructions
  - Register file: R0-R63 (scalar), H0-H31 (ref), T0-T31 (token)
  - Phase execution: READ → PROPOSE → CHOOSE → COMMIT
  - TraceStore for node/bond state

- **Kernel Compilation**:
  - Compiler now generates bytecode for nested kernels in creatures
  - `CompiledCreature.kernels` dict holds kernel bytecode
  - Kernel phases compile to proper EIS opcodes

- **Creature Runner**:
  - `CreatureRunner` - Executes compiled creatures via EIS VM
  - `spawn()` - Create creature instance from compiled code
  - `bond()` - Create bond between creatures
  - `send()/receive()` - Message passing via bonds
  - `invoke_kernel()` - Execute kernel bytecode
  - `process_messages()` - Dispatch "invoke" messages to kernels

- **Bond Protocol**:
  ```python
  # Send kernel invocation
  runner.send(cid1, cid2, {
      "type": "invoke",
      "kernel": "Greet",
      "inputs": {"name": "World"}
  })

  # Receive result
  response = runner.receive(cid1, cid2)
  # {"type": "result", "kernel": "Greet", "success": True, ...}
  ```

- **Tests**: 27/27 passing (existing tests still work)

---

## Phase 17: Substrate Primitives ✅

**Status**: Complete
**Date**: 2026-01-24

### 17.1 Primitive Interface ✅
- Added `PRIMITIVE` token type to lexer
- Added `PrimitiveCallExpr` AST node
- Parser handles `primitive("name", arg1, arg2, ...)` syntax
- Compiler emits `V2_PRIM` instruction with name ID and argument count

### 17.2 Core Primitives ✅
Implemented 17 primitives in `det/eis/primitives.py`:

| Primitive | Base Cost | Min Agency | Description |
|-----------|-----------|------------|-------------|
| `llm_call` | 1.0 | 0.3 | Call LLM with prompt |
| `llm_chat` | 1.5 | 0.3 | Chat with message history |
| `exec` | 0.5 | 0.5 | Execute shell command |
| `exec_safe` | 0.2 | 0.3 | Safe read-only commands |
| `file_read` | 0.1 | 0.2 | Read file contents |
| `file_write` | 0.2 | 0.5 | Write to file |
| `file_exists` | 0.01 | 0.1 | Check file exists |
| `file_list` | 0.05 | 0.2 | List directory |
| `now` | 0.001 | 0.0 | Unix timestamp |
| `now_iso` | 0.001 | 0.0 | ISO time string |
| `sleep` | 0.01 | 0.1 | Sleep milliseconds |
| `random` | 0.001 | 0.0 | Random [0,1) |
| `random_int` | 0.001 | 0.0 | Random integer |
| `random_seed` | 0.001 | 0.0 | Set random seed |
| `print` | 0.001 | 0.0 | Debug output |
| `log` | 0.001 | 0.0 | Log with level |
| `hash_sha256` | 0.01 | 0.0 | SHA256 hash |

### 17.3 Primitive Registry ✅
- `PrimitiveRegistry` class manages primitives
- F/agency checking before execution
- Cost tracking (base + per-unit)
- Call history for debugging

### 17.4 VM Integration ✅
- Added `V2_PRIM` opcode (0x84)
- VM dispatches to primitive registry
- F deducted on successful call
- Result returned in destination register

### 17.5 Creature Runner Integration ✅
- `call_primitive()` method on CreatureRunner
- Bond-based primitive calling via `{"type": "primitive", ...}` messages

### Files Created/Modified
- `det/lang/tokens.py` - Added PRIMITIVE token
- `det/lang/ast_nodes.py` - Added PrimitiveCallExpr
- `det/lang/parser.py` - Parse primitive() calls
- `det/lang/eis_compiler.py` - Compile to V2_PRIM
- `det/eis/encoding.py` - Added V2_PRIM opcode
- `det/eis/primitives.py` - NEW: Full primitive system
- `det/eis/vm.py` - Execute V2_PRIM instructions
- `det/eis/creature_runner.py` - Primitive integration
- `det/eis/__init__.py` - Export primitive classes

---

## Phase 18: Pure EL Creatures ✅

**Status**: Complete
**Date**: 2026-01-24

### 18.1 LLMCreature.ex ✅
- Think kernel using `primitive("llm_call", prompt)`
- Chat kernel for multi-turn conversations
- Temperature modulation by agency
- Token cost accounting in F

### 18.2 ToolCreature.ex ✅
- ExecSafe kernel using `primitive("exec_safe", command)`
- Exec kernel for full command execution (agency-gated)
- FileRead/FileWrite kernels using file primitives
- Agency-gated execution levels

### 18.3 MemoryCreature.ex ✅
- Store kernel with importance-weighted cost
- Recall kernel with matching
- Prune kernel for memory management
- Pure EL (uses `primitive("now")` for timestamps)

### 18.4 ReasonerCreature.ex ✅
- Reason kernel with chain-of-thought
- Optional LLM assistance based on agency
- Analyze kernel for statement analysis
- Configurable reasoning depth

### 18.5 PlannerCreature.ex ✅
- Plan kernel for task planning
- Decompose kernel for subtask generation
- Estimate kernel for resource estimation
- Active plan tracking

### Compilation Results
| Creature | Kernels | Init (bytes) | Kernel Code (bytes) |
|----------|---------|--------------|---------------------|
| LLMCreature | 3 | 84 | 340 |
| ToolCreature | 5 | 108 | 820 |
| MemoryCreature | 4 | 168 | 1008 |
| ReasonerCreature | 3 | 96 | 540 |
| PlannerCreature | 5 | 104 | 904 |

### Files Created
- `src/existence/llm.ex` - LLM creature (~150 lines)
- `src/existence/tool.ex` - Tool creature (~250 lines)
- `src/existence/memory.ex` - Memory creature (~250 lines)
- `src/existence/reasoner.ex` - Reasoner creature (~180 lines)
- `src/existence/planner.ex` - Planner creature (~250 lines)

---

## Phase 19: Terminal Creature 🔄

**Status**: In Progress
**Date**: 2026-01-24

### 19.1 Terminal Primitives ✅
Added 5 terminal port primitives to the substrate:
- `terminal_read()` - Read line of user input
- `terminal_write(msg)` - Write output to terminal
- `terminal_prompt(prompt)` - Display prompt and read input
- `terminal_clear()` - Clear terminal screen
- `terminal_color(color)` - Set terminal text color (reset, red, green, yellow, blue, cyan, etc.)

### 19.2 TerminalCreature.ex ✅
Created the CLI as a pure Existence-Lang creature with 9 kernels:
- **Init**: Display welcome message and setup
- **ReadInput**: Read user input via terminal_prompt primitive
- **WriteOutput**: Write output with color support
- **Dispatch**: Route commands to LLM or Tool creatures
- **Help**: Display help information
- **Status**: Show creature state (F, agency, stats)
- **Quit**: Gracefully stop the terminal
- **BondLLM**: Create bond to LLM creature
- **BondTool**: Create bond to Tool creature

### 19.3 Bootstrap Loader ✅
Created minimal Python bootstrap (`det_os_boot.py`):
- Loads and compiles TerminalCreature from .ex source
- Spawns creature instance via CreatureRunner
- Provides REPL loop for command dispatch
- Command-line options for verbose mode and custom creatures

### Compilation Results
| Creature | Kernels | Init (bytes) | Total Kernel (bytes) |
|----------|---------|--------------|----------------------|
| TerminalCreature | 9 | 112 | 1,660 |

### Files Created/Modified
- `src/existence/terminal.ex` - Terminal creature (~450 lines)
- `src/python/det_os_boot.py` - Bootstrap loader (~220 lines)
- `src/python/det/eis/primitives.py` - Added terminal primitives
- `src/python/det/lang/eis_compiler.py` - Added primitive ID mappings
- `src/python/det/eis/vm.py` - Added primitive ID mappings

### Remaining Work
- [x] Proper input/output port mapping in invoke_kernel ✅
- [ ] Bond-based dispatch to LLM/Tool creatures
- [ ] Command history and line editing

### 19.4 I/O Port Mapping ✅
**Date**: 2026-01-24

Updated compiler and runtime to properly track and use kernel port information:

**Compiler Changes (`eis_compiler.py`)**:
- Added `CompiledPort` dataclass with name, direction, type, and register index
- `CompiledKernel` now includes `ports: List[CompiledPort]`
- Port allocation captures direction (in/out) and register assignment

**Runtime Changes (`creature_runner.py`)**:
- Added `KernelPort` dataclass for runtime port representation
- `invoke_kernel` now maps input values to correct registers
- Output values are collected from correct registers by port name

**Example port mapping for WriteOutput kernel**:
```
  in    message: TokenReg @ reg 0
  in    color: TokenReg @ reg 1
  out   success: Register @ reg 0
```

### 19.5 Bond-Based Dispatch ✅
**Date**: 2026-01-24

Implemented proper DET bond-based dispatch between creatures:

**Architecture**:
```
TerminalCreature <--bond--> LLMCreature (channel 0)
TerminalCreature <--bond--> ToolCreature (channel 1)
```

**Bootstrap Process (`det_os_boot.py`)**:
1. Compile all three creatures (terminal.ex, llm.ex, tool.ex)
2. Spawn creature instances with initial F/a values
3. Create bonds between Terminal and LLM/Tool
4. Run REPL with bond-based command dispatch

**Message Flow**:
```
User Input -> Terminal -> [bond message] -> LLM/Tool
                      <- [bond response] <-
                 -> Display Output
```

**Commands**:
- `ask <query>` - Sends via bond to LLMCreature
- `run <cmd>` - Sends via bond to ToolCreature (safe mode)
- `exec <cmd>` - Sends via bond to ToolCreature (full mode)
- `status` - Shows creature F, messages sent/received
- `bonds` - Shows bond connections

**Test Results**:
```
Terminal: F=100.0, msgs_sent=4, msgs_recv=4
LLM:      F=100.0, msgs_sent=1, msgs_recv=1
Tool:     F=49.3,  msgs_sent=3, msgs_recv=3
```

**Files Modified**:
- `src/python/det_os_boot.py` - Rewrote as DETRuntime class with bond dispatch (~400 lines)

---

## Phase 20: Full Integration 🔄

**Status**: In Progress
**Date**: 2026-01-24

### 20.1 Deprecate Python Wrappers ✅
Updated CreatureLoader to prefer pure Existence-Lang creatures over Python wrappers:

**Changes to `det/os/creatures/loader.py`:**
- `load()` now searches for .ex files BEFORE checking built-in wrappers
- Added `force_builtin=True` parameter to force deprecated Python wrappers
- Added `EL_CREATURES` mapping for name-to-file resolution
- `list_available()` now shows creature type (existence-lang vs builtin-deprecated)
- Added deprecation warnings when Python wrappers are used

**Available EL Creatures:**
| Creature | File | Has Deprecated Wrapper |
|----------|------|------------------------|
| llm | llm.ex | No |
| tool | tool.ex | Yes |
| memory | memory.ex | Yes |
| reasoner | reasoner.ex | Yes |
| planner | planner.ex | Yes |
| terminal | terminal.ex | No |

### Files Modified
- `src/python/det/os/creatures/loader.py` - Prefer EL over Python wrappers

### 20.2 Performance Optimization ✅

Profiled and optimized the Existence-Lang compilation pipeline.

**Problem Analysis:**
Profiling revealed the tokenizer and parser were the main bottlenecks:
- `len(self.source)` called 833K times in tokenizer (never changes, should be cached)
- `peek()` called 525K times, `advance()` called 308K times
- Parser's `current()`, `check()`, `match()` calling `len()` on every invocation

**Tokenizer Optimizations (`det/lang/tokens.py`):**
- Added `__slots__` to Lexer class
- Cached `self.source_len = len(source)` in `__init__`
- Optimized `skip_whitespace()` with inlined loop (avoids method calls)
- Optimized `read_identifier()` with direct string slicing
- Optimized `read_number()` with direct string access

**Parser Optimizations (`det/lang/parser.py`):**
- Added `__slots__` to Parser class
- Cached `self.tokens_len = len(self.tokens)` in `__init__`
- Cached `self.eof_token` to avoid repeated construction
- Optimized `current()`, `peek()`, `advance()`, `check()`, `match()` to use cached values
- Eliminated redundant method calls in hot paths

**Performance Results:**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Tokenization (terminal.ex) | 6.79ms | 5.06ms | 25% faster |
| Parse time (all .ex files) | 57.10ms | 51.34ms | 10% faster |
| Total compile time | 62.50ms | 56.81ms | 9% faster |
| Rate | 57,438 lines/sec | 63,197 lines/sec | 10% faster |

**Files Modified:**
- `src/python/det/lang/tokens.py` - Tokenizer optimizations
- `src/python/det/lang/parser.py` - Parser optimizations

### 20.3 Bytecode Caching ✅

Implemented `.exb` bytecode cache for fast creature loading.

**File Format (.exb):**
- 32-byte header with magic, version, source mtime, size, CRC32 hash
- Pickled CompiledCreatureData payload

**New Files:**
- `det/lang/bytecode_cache.py` - BytecodeCache class (~250 lines)
- `det/lang/excompile.py` - CLI compiler tool

**Updated Files:**
- `det_os_boot.py` - Uses BytecodeCache for loading

**Performance Results:**

| Scenario | JIT | Cached | Speedup |
|----------|-----|--------|---------|
| Bootstrap (3 creatures) | 17.44ms | 2.55ms | **6.8x** |
| Single creature load | 7.75ms | 0.16ms | **48x** |

**CLI Commands:**
```bash
# Precompile all creatures
python -m det.lang.excompile --all

# Compile single file
python -m det.lang.excompile terminal.ex

# Show bytecode info
python -m det.lang.excompile --info terminal.exb

# Clean cache
python -m det.lang.excompile --clean ../existence/
```

**REPL Commands:**
- `list` - Shows cache status for each creature
- `recompile <name>` - Force recompile from source
- `recompile all` - Recompile all creatures

### Remaining Work
- [ ] Consider removing Python wrappers entirely
- [x] GPU acceleration (Phase 14 - complete)
- [ ] Consider Cython/C interpreter for further speedup

---

## Phase 14: GPU Backend with Metal Compute Shaders ✅

### 14.1 Metal Compute Shaders ✅
- **Status**: Complete (verified 2026-01-24)
- **Files**:
  - `src/substrate/metal/substrate_shaders.metal` (1,138 lines)
  - `src/substrate/metal/metal_backend.m` (855 lines)
  - `src/substrate/include/substrate_metal.h` (C API header)
  - `src/substrate/src/substrate_metal.c` (C wrapper)
  - `src/python/det/metal.py` (~600 lines Python bindings)

### 14.2 Features Implemented
- **Phase-based kernel dispatch**: READ → PROPOSE → CHOOSE → COMMIT
- **Structure-of-Arrays layout**: GPU-optimized memory access
- **Lane ownership model**: Parallel-safe effect deduplication
- **Atomic operations**: XFER_F, DIFFUSE for parallel safety
- **Full instruction set**: 46 opcodes implemented in Metal

### 14.3 Performance Results

**C Test Suite**: 9/9 tests passing

| Configuration | Ticks/sec |
|--------------|-----------|
| 100 nodes, 200 bonds | 4,249 |
| 1,000 nodes, 2,000 bonds | 4,608 |
| 10,000 nodes, 20,000 bonds | 4,145 |

**Python Integration**:

| Configuration | Ticks/sec |
|--------------|-----------|
| 1,000 nodes, 2,000 bonds | 1,115 |
| 10,000 nodes, 20,000 bonds | 1,065 |

**Key Metrics**:
- Device: Apple M1 Max
- GPU Memory: ~61 MB allocated
- Near-constant tick rate regardless of graph size (parallel scaling)

### 14.4 Build Instructions
```bash
cd local_agency/src/substrate/build
cmake .. -DENABLE_METAL=ON
make -j4

# Run Metal tests
./metal/test_metal

# Python verification
cd ../../python
python -c "from det.metal import MetalBackend; print(f'Metal: {MetalBackend.is_available()}')"
```

### 14.5 Python API Usage
```python
from det.metal import MetalBackend, NodeArraysHelper, BondArraysHelper

# Create backend
metal = MetalBackend()
print(f"Device: {metal.device_name}")

# Create and initialize arrays
nodes = NodeArraysHelper(1000)
bonds = BondArraysHelper(2000)
for i in range(2000):
    bonds.connect(i, i % 1000, (i + 1) % 1000)

# Upload, execute, download
metal.upload_nodes(nodes.as_ctypes(), 1000)
metal.upload_bonds(bonds.as_ctypes(), 2000)
metal.execute_ticks(1000, num_ticks=100)
metal.synchronize()
metal.download_nodes(nodes.as_ctypes(), 1000)
```

### 14.6 DET-OS GPU Integration ✅
- **Status**: Complete (2026-01-24)
- **Files Modified**:
  - `det_os_boot.py` - Added GPU backend integration

**Features**:
- `--gpu` flag for GPU-enabled startup
- GPU commands in REPL:
  - `gpu` / `gpu status` - Show GPU status
  - `gpu enable` / `gpu disable` - Toggle GPU acceleration
  - `gpu tick [N]` - Execute N substrate ticks on GPU
  - `gpu benchmark [nodes] [bonds] [ticks]` - Run performance benchmark
- Automatic creature state sync between Python and GPU
- GPU status in `status` command output

**REPL Usage**:
```
det> gpu
GPU Status:
  Available: True
  Enabled: No
  Device: Apple M1 Max (ready)

det> gpu enable
GPU acceleration enabled

det> gpu tick 100
Executed 100 GPU ticks in 85.85ms

det> gpu benchmark 10000 20000 1000
GPU Benchmark
  Device: Apple M1 Max
  Configuration: 10000 nodes, 20000 bonds, 1000 ticks
  GPU Time: 957.25ms
  Rate: 1045 ticks/sec
```

**Architecture Notes**:
- GPU accelerates substrate-level operations (node/bond state updates)
- High-level message passing and primitives remain in Python
- Best for large-scale simulations (100+ nodes)
- For 3-4 creatures, Python VM is sufficient (GPU overhead not worthwhile)

---

## Next Steps

See `ROADMAP_V2.md` for detailed roadmap.

1. **Phase 20: Full Integration** (remaining):
   - [x] Performance profiling and optimization (20.2 complete)
   - [x] GPU acceleration (Phase 14 complete)
   - [ ] Remove deprecated Python wrappers (optional)
   - [ ] Consider Cython/C interpreter for further speedup

2. **Future Work**:
   - [ ] Integrate Metal backend into det_os_boot for GPU-accelerated creature execution
   - [ ] Cross-platform GPU support (Vulkan compute for Linux/Windows)
   - [ ] Remote presence networking

---

## Technical Notes

### Build Instructions
```bash
# Build C kernel
cd src/det_core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j4

# Run C tests
./det_core_test

# Setup Python
cd ../..
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Python tests
cd src/python
python test_det.py

# Run CLI
python det_cli.py --model llama3.2:3b
```

### Key Equations Implemented
- **Presence**: `P = a * σ / (1 + F_op) / (1 + H_i)`
- **Coherence**: `dC = α*J - λ*C - slip*C*S_ij`
- **Agency Ceiling**: `a_max = 1 / (1 + λ_a * q²)`
- **Prison Regime**: `C > 0.7 && a_max < 0.2 && bondedness < 0.3`

### Constants
- `DET_MAX_NODES`: 4096
- `DET_P_LAYER_SIZE`: 16
- `DET_A_LAYER_SIZE`: 256
- `DET_DORMANT_SIZE`: 3760
- `DET_MAX_PORTS`: 64

---

## Issues & Resolutions

| Date | Issue | Resolution |
|------|-------|------------|
| 2026-01-17 | Missing `stdio.h` in det_core.c | Added include |
| 2026-01-17 | Port nodes exceeding DET_MAX_NODES | Reduced DORMANT_SIZE from 3824 to 3760 |
| 2026-01-17 | Prison regime test failing | Fixed to use aggregate_debt for a_max calculation |
| 2026-01-17 | `requests` module not found | Created .venv with requirements.txt |
| 2026-01-17 | State serialization tick not preserved | Fixed save/load order to include num_active and correct tick positioning |

---

*Last Updated: 2026-01-24 (Phase 17: Substrate Primitives Complete)*
