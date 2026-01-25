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
└── physics.ex      # DET physics layer
```

---

## Next Phases

### Phase 21: LLM Integration Enhancement
**Goal**: Better LLM coordination and multi-model support

#### 21.1 LLM Creature Enhancement
- [ ] Multi-model support in LLMCreature.ex
- [ ] Temperature modulation by agency/arousal
- [ ] Token budget management per-creature
- [ ] Streaming response support

#### 21.2 Conversation Management
- [ ] ConversationCreature.ex for multi-turn context
- [ ] Memory-backed context window
- [ ] Automatic context summarization when budget exceeded

#### 21.3 Model Routing
- [ ] Domain-aware model selection (code→coder, math→math-model)
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

---

## Design Principles

1. **DET First**: All behavior emerges from DET physics (F, a, P, bonds)
2. **Existence-Lang Native**: All creature logic in EL, not Python
3. **Substrate is Infrastructure**: Only primitives and execution in Python/C
4. **Bonds are Communication**: No direct method calls between creatures
5. **Phases are Atomic**: READ→PROPOSE→CHOOSE→COMMIT is the execution unit
6. **Resources are Real**: Every action costs F, tracked honestly
7. **GPU When Needed**: Use Metal for large-scale parallel execution

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

*Last Updated: 2026-01-24*
