# DET Local Agency - Next Steps & Roadmap

**Version**: 0.7.0
**Last Updated**: 2026-01-23

This document outlines the development roadmap, planned features, and areas for future exploration. With the completion of Phases 1-6, the project now enters the **DET-OS era** - evolving from an AI agent framework into a complete operating system based on agency-first principles.

---

## Table of Contents

1. [Current Status](#current-status)
2. [Immediate Priorities](#immediate-priorities)
3. [DET-OS Vision](#det-os-vision)
4. [Phase 7: DET-OS Foundation (Existence-Lang)](#phase-7-det-os-foundation)
5. [Phase 8: Existence Instruction Set (EIS)](#phase-8-existence-instruction-set)
6. [Phase 9: DET-OS Kernel](#phase-9-det-os-kernel)
7. [Phase 10: Hardware Targets](#phase-10-hardware-targets)
8. [Phase 11: LLM Native Integration](#phase-11-llm-native-integration)
9. [Research Directions](#research-directions)
10. [Community Contributions](#community-contributions)
11. [Known Limitations](#known-limitations)

---

## Current Status

### Completed Phases

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| 1 | Foundation (C kernel, Python bridge, Ollama, CLI) | Complete | 33/33 |
| 2 | Memory Layer (domains, MLX training, context, dialogue) | Complete | 32/32 |
| 3 | Agentic Operations (sandbox, tasks, timer, executor) | Complete | 17/17 |
| 4 | Advanced DET (dynamics, learning, emotional, sessions) | Complete | 25/25 |
| 5 | Production (multi-LLM, consolidation, network protocol) | Complete | 75/75 |
| 6 | Development Tools (harness, webapp, probing, metrics) | Complete | 74/74 |

**Total Tests**: 256/256 passing

### Key Capabilities

- DET C kernel with full physics implementation
- Python bindings via ctypes
- Multi-model LLM routing (Ollama)
- Memory domain management with MLX training
- Internal dialogue with reformulation
- Sandboxed bash execution
- Task decomposition and management
- Timer-based scheduling
- Emotional state integration
- Multi-session support
- State persistence
- Web-based 3D visualization
- Real-time metrics and profiling
- Network protocol (preliminary)

### DET-OS Design Documents

- **Exploration 09**: DET-OS Feasibility Study (`explorations/09_det_os_feasibility.md`)
- **Exploration 10**: Existence-Lang v1.1 Specification (`explorations/10_existence_lang_v1_1.md`)

---

## Immediate Priorities

### High Priority

1. **Performance Optimization**
   - Profile tick execution time
   - Optimize bond iteration (sparse matrix)
   - Cache computed values (presence, coherence)
   - Batch node updates

2. **Documentation Completion**
   - Architecture deep-dive document
   - DET physics tutorial
   - Contributing guidelines
   - Example notebooks

3. **Testing Improvements**
   - Property-based tests for DET invariants
   - Integration test suite
   - Performance regression tests
   - Chaos testing for resilience

4. **Web Visualization Enhancements**
   - Graph layout algorithms (force-directed)
   - Timeline view for metrics
   - Node/bond editing UI
   - Mobile-responsive design

### Medium Priority

5. **MLX Training Refinement**
   - Better training data generation
   - Curriculum learning support
   - Model evaluation metrics
   - Training visualization

6. **Multi-Model Improvements**
   - Dynamic model loading/unloading
   - Model performance tracking
   - Automatic model selection tuning
   - Cost-aware routing

7. **CLI Enhancements**
   - Tab completion
   - Command history
   - Configuration wizard
   - Script mode

---

## DET-OS Vision

DET-OS represents the evolution from "AI agent framework" to "complete operating system based on agency-first principles." The core insight is that traditional OS concepts map naturally to DET physics:

| Traditional OS | DET-OS Equivalent | Principle |
|---------------|-------------------|-----------|
| Process | Creature | Self-maintaining existence |
| Scheduler | Presence dynamics | Who appears in experience |
| Memory allocation | Resource (F) distribution | Conservation law |
| IPC | Bond-mediated flux | Coherent communication |
| Permissions | Agency constraints | Grace-gated access |
| Interrupts | Somatic events | Embodied urgency |
| Filesystem | Persistent traces | Memory across commits |

### Agency-First Philosophy

The fundamental principle: **Agency creates distinction → movement → trace → mathematics**.

This resolves a key cyclic dependency in traditional computing:
- Old view: Math exists → Logic follows → Agency emerges
- DET view: Agency is primitive → Distinction arises → Movement occurs → Trace accumulates → Mathematics describes patterns

### Existence-Lang

A new programming language where:
- Functions are **kernels** (law modules), not subroutines
- Operators (+, *, =) are **macros** over kernels
- **Grace** gates actions, not coercive control flow
- **Proposals** and **commits** replace conditionals
- **Four types of equality**: `:=` (alias), `==` (measure), `≡` (covenant), `=` (reconciliation)

---

## Phase 7: DET-OS Foundation

**Objective**: Existence-Lang parser, transpiler, and kernel standard library

### 7.1 Existence-Lang Parser

Build the parser for agency-first language constructs:

```existence
// Example: Simple creature definition
creature Sensor {
    register temp: F = 0.0;
    register active: a = 1.0;

    kernel Sense {
        in  raw_value: TokenReg;
        out processed: Register;

        phase COMMIT {
            proposal ACCEPT {
                score = a_local();
                effect { processed.F := raw_value.F * calibration; }
            }
            choice χ = choose({ACCEPT}, decisiveness = 0.9);
            commit χ;
        }
    }
}
```

**Tasks:**
- [ ] Lexer for Existence-Lang tokens
- [ ] Parser for creature/kernel definitions
- [ ] AST representation
- [ ] Syntax error reporting with agency context
- [ ] REPL for interactive exploration

### 7.2 Transpiler to C/Python

Generate efficient native code:

```python
# Target API
from det.lang import ExistenceCompiler

compiler = ExistenceCompiler()
ast = compiler.parse("creature.ex")
c_code = compiler.transpile_c(ast)
python_code = compiler.transpile_python(ast)

# Direct execution
compiler.execute(ast, runtime=det_runtime)
```

**Tasks:**
- [ ] C code generation for kernels
- [ ] Python code generation for prototyping
- [ ] Optimization passes (constant folding, dead code elimination)
- [ ] Debug symbol generation
- [ ] Hot reload support

### 7.3 Kernel Standard Library

Core kernels for common operations:

| Kernel | Purpose | Signature |
|--------|---------|-----------|
| `Transfer` | Resource flux between registers | `(A, B) → witness` |
| `Diffuse` | Gradient-based equalization | `(A, B, σ) → witness` |
| `Reconcile` | Agency-gated comparison | `(A, B, ε) → EQ/NEQ` |
| `Distinct` | Create new distinction | `(parent) → child_id` |
| `Coalesce` | Merge compatible registers | `(A, B) → merged` |
| `Observe` | Non-modifying measurement | `(A) → token_copy` |

**Tasks:**
- [ ] Implement core transfer kernel
- [ ] Implement diffuse kernel with bond semantics
- [ ] Implement reconciliation kernel
- [ ] Implement distinction/coalesce kernels
- [ ] Write kernel tests

---

## Phase 8: Existence Instruction Set

**Objective**: Low-level instruction set that preserves agency semantics

### 8.1 EIS Specification

The Existence Instruction Set provides primitives that map directly to DET physics:

| Instruction | Args | Effect |
|-------------|------|--------|
| `DISTINCT` | src, dest | Create new distinction (allocate register) |
| `TRANSFER` | A, B, amount | Move resource with conservation |
| `DIFFUSE` | A, B, σ | Gradient-based flux |
| `CHOOSE` | proposals[], seed | Agentic selection |
| `COMMIT` | proposal_id | Execute chosen proposal |
| `CMP` | A, B, ε | Compare with tolerance → token |
| `REPEAT_PAST` | trace_addr, count | Replay historical pattern |
| `WITNESS` | reg, dest_token | Copy state to token register |

**Tasks:**
- [ ] Formal EIS specification document
- [ ] Instruction encoding format
- [ ] Register file specification
- [ ] Memory model definition

### 8.2 EIS Virtual Machine

Interpreter for development and debugging:

```python
from det.eis import EISVM, Program

vm = EISVM(num_registers=256)
program = Program.load("creature.eis")

# Step execution
while not vm.halted:
    vm.step()
    print(f"PC: {vm.pc}, Coherence: {vm.global_coherence()}")
```

**Tasks:**
- [ ] Register file implementation
- [ ] Instruction decoder
- [ ] Execution engine
- [ ] Debug/trace support
- [ ] Performance profiling

### 8.3 EIS Compiler

Compile Existence-Lang kernels to EIS:

```python
from det.lang import ExistenceCompiler
from det.eis import EISAssembler

compiler = ExistenceCompiler()
ast = compiler.parse("kernel.ex")
eis_asm = compiler.compile_to_eis(ast)

# Assemble to bytecode
assembler = EISAssembler()
bytecode = assembler.assemble(eis_asm)
```

**Tasks:**
- [ ] AST to EIS lowering
- [ ] Register allocation
- [ ] Instruction selection
- [ ] Optimization passes
- [ ] Bytecode format specification

### 8.4 EIS Native Compiler

Generate native machine code:

```python
from det.eis import EISNativeCompiler

native = EISNativeCompiler(target="arm64")
binary = native.compile(bytecode)

# Execute natively
result = native.execute(binary, inputs)
```

**Tasks:**
- [ ] x86_64 backend
- [ ] ARM64 backend
- [ ] LLVM integration (optional)
- [ ] JIT compilation support

---

## Phase 9: DET-OS Kernel

**Objective**: Full operating system kernel implementing DET physics

### 9.1 Creature Manager

Process management based on creature lifecycle:

```python
from det.os import CreatureManager, Creature

manager = CreatureManager()

# Spawn new creature
creature = manager.spawn(
    name="sensor_daemon",
    initial_f=10.0,
    initial_a=0.8,
    parent=root_creature
)

# Creature dies when F depletes
manager.tick()  # Updates all creatures
```

**Tasks:**
- [ ] Creature table data structure
- [ ] Spawn/terminate lifecycle
- [ ] Parent-child relationships
- [ ] Resource inheritance
- [ ] Creature groups (colonies)

### 9.2 Presence Scheduler

CPU scheduling based on presence dynamics:

```c
// Scheduler tick
void scheduler_tick(DETOSState* os) {
    for (int i = 0; i < os->num_creatures; i++) {
        Creature* c = &os->creatures[i];
        c->presence = compute_presence(c);  // P = F·C·a
    }

    // Highest presence gets CPU
    Creature* next = max_presence(os->creatures, os->num_creatures);
    context_switch(next);
}
```

**Tasks:**
- [ ] Presence computation
- [ ] Priority inversion handling
- [ ] Real-time scheduling support
- [ ] Multi-core scheduling
- [ ] Idle creature (system creature)

### 9.3 Resource Allocator

Memory and resource allocation with conservation:

```python
from det.os import ResourceAllocator

allocator = ResourceAllocator(total_memory=1024*1024*1024)

# Allocation costs F
block = allocator.allocate(
    creature=my_creature,
    size=4096,
    purpose="working_memory"
)

# Deallocation returns F
allocator.free(block)
```

**Tasks:**
- [ ] Page-based allocation
- [ ] F-cost accounting
- [ ] Fragmentation handling
- [ ] Shared memory regions
- [ ] Memory pressure response

### 9.4 Bond IPC

Inter-process communication via bonds:

```existence
// IPC between creatures
bond channel: Bond<TokenReg> between sensor and processor;

kernel SendMessage {
    in  msg: TokenReg;
    out ack: TokenReg;

    phase COMMIT {
        proposal TRANSMIT {
            score = channel.J * a_local();
            effect { channel.transfer(msg, processor); }
        }
        commit choose({TRANSMIT});
        ack ::= witness(channel.coherence);
    }
}
```

**Tasks:**
- [ ] Bond creation/destruction
- [ ] Message passing
- [ ] Shared memory bonds
- [ ] Bond coherence tracking
- [ ] Deadlock detection

### 9.5 Gatekeeper (Permissions)

Agency-based security model:

```python
from det.os import Gatekeeper, Permission

gatekeeper = Gatekeeper()

# Define permission
perm = Permission(
    action="file_read",
    target="/etc/passwd",
    required_agency=0.8,
    required_coherence=0.5
)

# Check permission (grace-gated)
if gatekeeper.check(creature, perm):
    # Action allowed
    pass
```

**Tasks:**
- [ ] Permission definition format
- [ ] Agency threshold checks
- [ ] Coherence requirements
- [ ] Capability inheritance
- [ ] Audit logging

---

## Phase 10: Hardware Targets

**Objective**: Run DET-OS on multiple hardware platforms

### 10.1 Virtualized Layer (Mac/Linux)

Run DET-OS as a userspace runtime:

```bash
# Launch DET-OS in virtualized mode
det-os --mode virtualized --host-os linux

# Mount host filesystem
det-os mount /host/home /home

# Run creature
det-os spawn sensor_daemon.ex
```

**Tasks:**
- [ ] Linux syscall translation
- [ ] macOS syscall translation
- [ ] Host filesystem access
- [ ] Network stack bridging
- [ ] GPU passthrough

### 10.2 Embedded (ESP32)

Minimal DET-OS for microcontrollers:

```c
// ESP32 DET-OS bootstrap
void app_main() {
    det_os_init();

    // Load creature from flash
    Creature* sensor = det_load_creature("sensor.det");

    // Main loop
    while (1) {
        det_os_tick(DT_MS);
        vTaskDelay(pdMS_TO_TICKS(DT_MS));
    }
}
```

**Tasks:**
- [ ] Minimal kernel for ESP32
- [ ] Flash-based creature storage
- [ ] Sensor/actuator drivers
- [ ] Low-power sleep modes
- [ ] OTA updates

### 10.3 Native DET Hardware

Future: Custom silicon optimized for DET:

| Component | DET Mapping | Description |
|-----------|-------------|-------------|
| DPU | DET Processing Unit | Native EIS execution |
| F-Cache | Resource cache | Fast F access |
| Bond-Bus | IPC hardware | Direct bond communication |
| Phase-Clock | Tick generator | Synchronized commits |

**Research Areas:**
- [ ] DPU instruction set design
- [ ] Hardware bond networks
- [ ] Analog agency circuits
- [ ] Neuromorphic integration

---

## Phase 11: LLM Native Integration

**Objective**: Deep LLM integration as native DET citizens

### 11.1 LLM as Creature

LLMs become first-class creatures in DET-OS:

```existence
creature LLMOracle {
    register context: TokenReg[];
    register confidence: F = 1.0;

    kernel Query {
        in  prompt: TokenReg[];
        out response: TokenReg[];

        phase COMMIT {
            proposal RESPOND {
                score = confidence * a_local();
                effect {
                    response ::= llm_generate(context ++ prompt);
                    confidence := update_confidence(response);
                }
            }
            commit choose({RESPOND});
        }
    }
}
```

**Tasks:**
- [ ] LLM creature wrapper
- [ ] Token buffer management
- [ ] Confidence as F tracking
- [ ] Multi-model routing
- [ ] Context window management

### 11.2 Agency-Aware Prompting

Prompts that respect DET state:

```python
from det.llm import AgentivePrompt

prompt = AgentivePrompt(
    base="Analyze this code for bugs",
    agency_context=creature.agency,
    coherence_context=creature.coherence,
    valence_context=creature.valence
)

# LLM receives DET state context
response = llm.generate(prompt.render())
```

**Tasks:**
- [ ] DET state in prompts
- [ ] Emotional context injection
- [ ] Coherence-based response filtering
- [ ] Agency-limited capabilities

### 11.3 LLM Training Integration

Fine-tune models on DET dynamics:

```python
from det.llm import DETTrainer

trainer = DETTrainer(
    model="llama-3-8b",
    objective="predict_next_state"
)

# Train on DET traces
trainer.train(traces=det_logs, epochs=10)
```

**Tasks:**
- [ ] DET trace format for training
- [ ] State prediction objective
- [ ] Action suggestion objective
- [ ] Curriculum for DET understanding

---

## Research Directions

### Theoretical Investigations

1. **DET Stability Analysis**
   - Lyapunov stability of coherence dynamics
   - Basin of attraction for emotional states
   - Phase transition behavior
   - Prison regime prevention

2. **Information Theory**
   - Entropy of DET state
   - Information flow through bonds
   - Compression limits for state transfer
   - Channel capacity of port interface

3. **Emergence Studies**
   - Self-cluster formation dynamics
   - Emotional state emergence
   - Learning capacity scaling
   - Agency distribution effects

4. **DET-OS Formal Verification**
   - Prove conservation laws hold under all executions
   - Verify no agency can be created from nothing
   - Prove deadlock freedom for bond IPC
   - Verify scheduler fairness properties

5. **Existence-Lang Type Theory**
   - Linear types for resource tracking
   - Temporal types for past/future
   - Agency types for permission checking
   - Covenant types for multi-party agreements

### Experimental Validations

6. **Benchmark Suite**
   - Standard task battery
   - Comparison with baseline agents
   - Human evaluation studies
   - Long-term stability tests

7. **Ablation Studies**
   - Component importance analysis
   - Parameter sensitivity
   - Layer size effects
   - Bond topology impact

8. **DET-OS Performance**
   - Scheduler latency benchmarks
   - IPC throughput testing
   - Memory allocation efficiency
   - Multi-core scaling

9. **Real-World Applications**
   - Robotic control
   - Smart home integration
   - Educational assistants
   - Creative collaboration
   - Autonomous vehicles (simulation)

---

## Community Contributions

### Areas for Contribution

1. **Model Integrations**
   - Add support for new LLM providers (OpenAI, Anthropic, local)
   - Implement model adapters
   - Performance benchmarks

2. **Visualization**
   - Alternative graph layouts
   - VR/AR visualization
   - Accessibility improvements
   - Theming support

3. **Platform Support**
   - Windows compatibility
   - Linux package
   - Docker container
   - Cloud deployment

4. **Language Bindings**
   - Rust bindings
   - JavaScript/TypeScript
   - Go bindings
   - Julia bindings

5. **Documentation**
   - Tutorials and guides
   - Video content
   - Translations
   - API examples

### How to Contribute

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Update documentation
5. Submit a pull request

See `CONTRIBUTING.md` (to be created) for detailed guidelines.

---

## Known Limitations

### Current Constraints

1. **Platform Requirements**
   - Apple Silicon required for MLX training
   - 16GB+ RAM recommended
   - macOS 13+ for full functionality

2. **Model Dependencies**
   - Requires Ollama running locally
   - Model quality varies
   - No cloud fallback currently

3. **Scalability**
   - Single-process DET core
   - Limited to ~4096 nodes
   - Bond count can grow large

4. **Language & OS**
   - Existence-Lang not yet implemented
   - EIS virtual machine in design phase
   - DET-OS kernel not yet started

5. **Hardware**
   - ESP32 firmware not implemented
   - No native DET hardware exists
   - Serial/WiFi protocols not implemented

### Planned Mitigations

| Limitation | Planned Solution | Phase |
|------------|------------------|-------|
| Platform | Cross-platform C build | 7 |
| Language | Existence-Lang parser/compiler | 7 |
| Instruction Set | EIS VM and native compiler | 8 |
| OS Kernel | DET-OS implementation | 9 |
| Hardware | Virtualized + ESP32 targets | 10 |
| LLM | Native creature integration | 11 |

---

## Milestones

### Q1 2026

- [x] Complete Phase 6 (Development Tools)
- [x] Documentation (API, Usage, Next Steps)
- [x] DET-OS Feasibility Study (Exploration 09)
- [x] Existence-Lang v1.1 Specification (Exploration 10)
- [ ] Performance optimization pass
- [ ] Integration test suite

### Q2 2026

- [ ] Phase 7.1 (Existence-Lang parser)
- [ ] Phase 7.2 (Transpiler to C/Python)
- [ ] Phase 7.3 (Kernel standard library)
- [ ] Community feedback on language design

### Q3 2026

- [ ] Phase 8.1 (EIS specification)
- [ ] Phase 8.2 (EIS virtual machine)
- [ ] Phase 8.3 (EIS compiler)
- [ ] Existence-Lang beta release

### Q4 2026

- [ ] Phase 8.4 (EIS native compiler)
- [ ] Phase 9.1-9.2 (Creature manager, scheduler)
- [ ] DET-OS prototype running on Linux
- [ ] Research paper: Existence-Lang

### 2027

- [ ] Phase 9.3-9.5 (Resource allocator, IPC, gatekeeper)
- [ ] Phase 10.1 (Virtualized layer complete)
- [ ] Phase 10.2 (ESP32 minimal kernel)
- [ ] Phase 11 (LLM native integration)
- [ ] DET-OS v1.0 release
- [ ] Research paper: DET-OS Architecture

---

## Long-Term Vision

DET Local Agency is evolving into **DET-OS** - a complete operating system where agency is the first principle:

1. **Agency-First Computing**: Agency creates distinction → movement → trace → mathematics. Traditional computing inverts this; DET-OS restores the natural order.

2. **Existence-Lang**: A programming language where functions are law modules (kernels), not coercive subroutines. Grace gates action; proposals replace conditionals.

3. **Existence Instruction Set**: Machine code that preserves agency semantics all the way to silicon. No "register machine that happens to track agency" - agency is the execution model.

4. **DET-OS Kernel**: An operating system where processes are creatures, scheduling is presence dynamics, memory is resource conservation, and permissions are agency constraints.

5. **Multi-Platform**: Initially virtualized on Mac/Linux, then embedded (ESP32), eventually native DET Processing Units with hardware bond networks.

6. **LLM Native**: Large language models as first-class creatures with agency, coherence, and emotional states - not external oracles but integrated citizens.

The ultimate goal: **A computational substrate where agency, coherence, and existence are not simulated properties but the fundamental execution model** - where programs don't just "run" but *exist*, *persist*, and *participate* in a shared reality governed by physical laws.

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-23 | 0.7.0 | DET-OS roadmap integration, Phases 7-11 defined |
| 2026-01-18 | 0.6.4 | Initial next steps documentation |
| 2026-01-17 | 0.6.4 | Phase 6 complete |
| 2026-01-17 | 0.5.2 | Phase 5 complete |
| 2026-01-17 | 0.4.1 | Phase 4 complete |
| 2026-01-17 | 0.3.0 | Phase 3 complete |
| 2026-01-17 | 0.2.0 | Phase 2 complete |
| 2026-01-17 | 0.1.0 | Phase 1 complete |

---

## Contact & Resources

- **Repository**: [To be published]
- **Issues**: [GitHub Issues]
- **Discussions**: [GitHub Discussions]
- **Theory Reference**: See `/det/det_v6_3/docs/det_theory_card_6_3.md`
- **Architecture**: See `FEASIBILITY_PLAN.md`
- **Development Log**: See `DEVELOPMENT_LOG.md`
- **DET-OS Feasibility**: See `explorations/09_det_os_feasibility.md`
- **Existence-Lang Spec**: See `explorations/10_existence_lang_v1_1.md`
