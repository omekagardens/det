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
│  - Scheduling (presence-based)                   │
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

### Phase 14: GPU Backend (Planned) ⏸️
- Metal compute shaders for parallel execution
- Deferred until interpreter is complete

---

## Current Work: Full EIS Execution

### Phase 15: EIS Interpreter ✅
**Status**: Complete

Built Python interpreter that executes EIS bytecode with full phase semantics.

#### 15.1 Instruction Decoder ✅
- [x] Decode 4-byte EIS instructions
- [x] Extract opcode, operands, flags
- [x] Handle opcodes (LDI, ADD, SUB, MUL, DIV, CMP, PROP_*, etc.)

#### 15.2 Register File ✅
- [x] R0-R63: Scalar registers (float)
- [x] H0-H31: Reference registers (node/bond IDs)
- [x] T0-T31: Token registers (witness tokens)
- [x] Lane-based register isolation

#### 15.3 Phase Execution Engine ✅
- [x] READ phase: Load state, create witnesses
- [x] PROPOSE phase: Generate proposals with scores
- [x] CHOOSE phase: Deterministic selection
- [x] COMMIT phase: Apply effects, emit witnesses

#### 15.4 Memory Model ✅
- [x] Node state access (F, a, q, σ, P, τ)
- [x] Bond state access (C, flux, node_i, node_j)
- [x] TraceStore for state management

**Files**:
- `src/python/det/eis/vm.py` - Main VM (557 lines)
- `src/python/det/eis/registers.py` - Register file
- `src/python/det/eis/encoding.py` - Instruction decoder
- `src/python/det/eis/phases.py` - Phase execution
- `src/python/det/eis/memory.py` - Memory model

---

### Phase 16: Kernel Runtime 🔄
**Status**: In Progress

Integrate interpreter with creature/kernel dispatch.

#### 16.1 Message Dispatcher ✅
- [x] Parse bond messages
- [x] Match to kernel by name/type
- [x] Invoke kernel with inputs

#### 16.2 Kernel Execution ✅
- [x] Execute kernel bytecode
- [x] Track F cost
- [x] Send response via bond

#### 16.3 Proposal Mechanics
- [ ] Collect proposals from PROPOSE phase
- [ ] Score aggregation
- [ ] Deterministic choice (seeded RNG)
- [ ] Effect application

#### 16.4 Creature Runner ✅
- [x] Spawn creatures from compiled code
- [x] Manage creature state (F, a, q)
- [x] Bond creation and messaging
- [x] Kernel invocation via bonds

**Files**:
- `src/python/det/eis/creature_runner.py` - Creature execution layer (NEW)

---

### Phase 17: Substrate Primitives ✅
**Status**: Complete

Define and implement external I/O primitives callable from Existence-Lang.

#### 17.1 Primitive Interface ✅
- [x] Define `primitive` keyword in EL
- [x] `primitive("name", arg1, arg2, ...)` syntax
- [x] Compiler emits V2_PRIM instruction
- [x] VM dispatches to primitive registry

#### 17.2 Core Primitives ✅
17 primitives implemented:
- [x] `llm_call`, `llm_chat` - LLM interaction
- [x] `exec`, `exec_safe` - Shell execution
- [x] `file_read`, `file_write`, `file_exists`, `file_list` - File I/O
- [x] `now`, `now_iso`, `sleep` - Time operations
- [x] `random`, `random_int`, `random_seed` - Randomness
- [x] `print`, `log` - Debug output
- [x] `hash_sha256` - Cryptography

#### 17.3 Primitive Registry ✅
- [x] Python primitive implementations
- [x] Primitive cost model (base + per-unit F consumption)
- [x] Agency checking (min_agency per primitive)
- [x] Call history tracking

**Files**:
- `src/python/det/eis/primitives.py` - Full implementation (~550 lines)
- `src/python/det/lang/tokens.py` - PRIMITIVE token
- `src/python/det/lang/ast_nodes.py` - PrimitiveCallExpr
- `src/python/det/lang/parser.py` - Parse primitive calls
- `src/python/det/lang/eis_compiler.py` - Compile V2_PRIM
- `src/python/det/eis/encoding.py` - V2_PRIM opcode (0x84)
- `src/python/det/eis/vm.py` - Execute primitives

---

### Phase 18: Pure EL Creatures ✅
**Status**: Complete

Rewrite all creatures in pure Existence-Lang using primitives.

#### 18.1 LLMCreature.ex ✅
- [x] Think kernel using `primitive("llm_call", prompt)`
- [x] Chat kernel for multi-turn conversations
- [x] Temperature modulation by agency
- [x] Token cost accounting in F

#### 18.2 ToolCreature.ex ✅
- [x] ExecSafe kernel using `primitive("exec_safe", command)`
- [x] Exec kernel for full execution (agency-gated)
- [x] FileRead/FileWrite kernels
- [x] Agency-gated execution levels

#### 18.3 MemoryCreature.ex ✅
- [x] Store kernel with importance-weighted cost
- [x] Recall kernel with matching
- [x] Prune kernel for memory management
- [x] Uses `primitive("now")` for timestamps

#### 18.4 ReasonerCreature.ex ✅
- [x] Reason kernel with chain-of-thought
- [x] Optional LLM assistance based on agency
- [x] Analyze kernel for statement analysis

#### 18.5 PlannerCreature.ex ✅
- [x] Plan kernel for task planning
- [x] Decompose kernel for subtask generation
- [x] Estimate kernel for resource estimation

**Files**:
- `src/existence/llm.ex` - 150 lines, 3 kernels
- `src/existence/tool.ex` - 250 lines, 5 kernels
- `src/existence/memory.ex` - 250 lines, 4 kernels
- `src/existence/reasoner.ex` - 180 lines, 3 kernels
- `src/existence/planner.ex` - 250 lines, 5 kernels

---

### Phase 19: Terminal Creature ✅
**Status**: Complete

The CLI itself becomes an Existence-Lang creature.

#### 19.1 Terminal Primitives ✅
- [x] `terminal_read()` - Read line of user input
- [x] `terminal_write(msg)` - Write output (no newline)
- [x] `terminal_prompt(prompt)` - Prompt with prefix
- [x] `terminal_clear()` - Clear terminal screen
- [x] `terminal_color(color)` - Set text color

#### 19.2 TerminalCreature.ex ✅
- [x] Main creature with 9 kernels
- [x] Init, ReadInput, WriteOutput kernels
- [x] Dispatch kernel for command routing
- [x] Help, Status, Quit kernels
- [x] BondLLM, BondTool kernels for creature connectivity

#### 19.3 Bootstrap ✅
- [x] Minimal Python bootstrap loader (`det_os_boot.py`)
- [x] Load TerminalCreature from bytecode
- [x] REPL loop structure

#### 19.4 Full Integration ✅
- [x] Proper input/output port mapping (compiler records port info)
- [x] Bond-based command dispatch to LLM/Tool creatures
- [ ] Command history and editing (optional enhancement)

**Files**:
- `src/existence/terminal.ex` - 450 lines, 9 kernels
- `src/python/det_os_boot.py` - 220 lines
- `src/python/det/eis/primitives.py` - 5 terminal primitives

---

### Phase 20: Full Integration 🔄
**Status**: In Progress

Complete the DET-native operating system.

#### 20.1 Deprecate Python Wrappers ✅
- [x] Mark Python creature implementations as deprecated
- [x] Update CreatureLoader to prefer .ex files
- [x] Add deprecation warnings for built-in usage
- [x] Add force_builtin flag for backward compatibility
- [x] All creature logic now in Existence-Lang

#### 20.2 Performance Optimization
- [ ] Profile interpreter
- [ ] Hot path optimization
- [ ] Consider Cython/C interpreter

#### 20.3 GPU Acceleration (Phase 14 Revival)
- [ ] Metal compute shaders for interpreter
- [ ] Parallel creature execution
- [ ] Batch kernel dispatch

**Files Modified**:
- `det/os/creatures/loader.py` - Updated to prefer EL creatures

---

## Future Phases

### Phase 21: Networking
- Bond channels over network
- Distributed creature execution
- Consensus for shared state

### Phase 22: Persistence
- Creature state serialization
- Checkpoint/restore
- Memory-mapped substrate

### Phase 23: Security
- Capability-based access control
- Resource quotas
- Sandboxed primitives

---

## Design Principles

1. **DET First**: All behavior emerges from DET physics (F, a, P, bonds)
2. **Existence-Lang Native**: All creature logic in EL, not Python
3. **Substrate is Infrastructure**: Only primitives and execution in Python/C
4. **Bonds are Communication**: No direct method calls between creatures
5. **Phases are Atomic**: READ→PROPOSE→CHOOSE→COMMIT is the execution unit
6. **Resources are Real**: Every action costs F, tracked honestly

---

*Last Updated: 2026-01-24*
