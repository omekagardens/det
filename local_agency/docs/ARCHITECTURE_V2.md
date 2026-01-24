# DET Local Agency: Architecture v2.1 - DET-Aware Substrate

**Version**: 2.1.0
**Date**: 2026-01-23
**Status**: Architecture Specification

---

## Executive Summary

This document defines the target architecture for DET-OS: **Existence-Lang First** with a **DET-Aware Substrate**.

### Key Principles

1. **Existence-Lang is the system language** - physics, kernel, applications
2. **Substrate is DET-aware** - phases, proposals, typed refs built in
3. **Substrate contains NO physics** - transfer, diffuse, coherence are in Existence-Lang
4. **Python is for bridges only** - LLM, web API, development tools
5. **GPU-ready design** - SIMD-friendly, no divergent branches, coalesced memory

### Architecture Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    Applications (Existence-Lang)                 │
│                  User creatures, services, tools                 │
├─────────────────────────────────────────────────────────────────┤
│                 DET-OS Kernel (Existence-Lang)                   │
│           kernel.ex: Schedule, Allocate, Gate, Grace, IPC        │
├─────────────────────────────────────────────────────────────────┤
│                 DET Physics (Existence-Lang)                     │
│      physics.ex: transfer, diffuse, presence, coherence, agency  │
│              Compiled to EIS bytecode, runs on substrate         │
├─────────────────────────────────────────────────────────────────┤
│                  EIS Substrate v2 (Minimal C)                    │
│     eis_substrate.c: DET-aware execution (phases, proposals)     │
│              NO DET PHYSICS - just execution substrate           │
├─────────────────────────────────────────────────────────────────┤
│                    Host OS / Hardware                            │
│              CPU, GPU, or future DET-native silicon              │
└─────────────────────────────────────────────────────────────────┘

Bridges (Python - external integration only):
├── LLM Integration (llm_bridge.py)
├── Web API (webapp/server.py)
├── Development Tools (harness, visualization)
└── Bootstrap Loader (bootstrap.py)
```

---

## Part 1: Layer Definitions

### Layer 0: EIS Substrate v2 (DET-Aware, GPU-Ready)

The substrate is the **only C code** in the system. It provides DET-aware execution primitives but contains **no DET physics**.

**What It Contains**:
- Phase-aware execution model (READ → PROPOSE → CHOOSE → COMMIT)
- Typed references (NodeRef, BondRef, FieldRef)
- Proposal buffers and selection
- Effect table for verified operations
- Minimal arithmetic (ADD, SUB, MUL, DIV, SQRT, MIN, MAX)
- Comparison (produces tokens)
- I/O primitives

**What It Does NOT Contain**:
- NO transfer dynamics
- NO diffuse computation
- NO presence calculation
- NO coherence update
- NO agency ceiling logic
- NO grace protocol
- NO scheduling decisions

**Register Model**:
```
Per-Lane Registers:
  R0-R15   : 16 scalar registers (F32)
  H0-H7    : 8 reference registers (NodeRef, BondRef, etc.)
  T0-T7    : 8 token registers (Tok32)
```

**Instruction Set** (~30 opcodes):
```
PHASE CONTROL:
  PHASE.R, PHASE.P, PHASE.C, PHASE.X, TICK, NOP, HALT, YIELD

TYPED LOADS (READ phase):
  LDN dst, nodeRef, field      ; Load node field
  LDB dst, bondRef, field      ; Load bond field
  LDNB dst, nodeRef, nbrIdx, field  ; Load neighbor field

ARITHMETIC:
  ADD, SUB, MUL, DIV, MAD
  SQRT, MIN, MAX, RELU, NEG, ABS

COMPARISON:
  CMP dstTok, src0, src1       ; Produces LT/EQ/GT token
  CMPE dstTok, src0, src1, eps ; Compare with epsilon

PROPOSALS (PROPOSE phase):
  PROP.NEW propRef
  PROP.SCORE propRef, scoreReg
  PROP.EFFECT propRef, effectId, args

CHOOSE/COMMIT:
  CHOOSE dstChoice, propList, decisiveness, seed
  COMMIT choiceRef
  WITNESS tokRef, witnessType

STORES (COMMIT phase only):
  STN nodeRef, field, src
  STB bondRef, field, src
  STT tokRef, srcTok

I/O:
  IN dst, channel
  OUT channel, src
  EMIT bufRef, byte

SYSTEM:
  RAND dst
  SEED src
  LANE dst
  TIME dst
```

**Effect Table** (verified operations):
```
EFFECT_XFER_F(src, dst, amount)  ; Antisymmetric transfer
EFFECT_DIFFUSE(bond, delta)      ; Symmetric flux
EFFECT_SET_F(node, value)        ; Direct set
EFFECT_ADD_F(node, delta)        ; Increment
EFFECT_SET_C(bond, value)        ; Set coherence
EFFECT_EMIT_TOK(tokRef, value)   ; Witness token
EFFECT_EMIT_BYTE(bufRef, byte)   ; Boundary output
```

**Memory Layout** (GPU-optimized, SoA):
```c
typedef struct {
    float* F;        // [num_nodes] Resource
    float* q;        // [num_nodes] Structural debt
    float* a;        // [num_nodes] Agency
    float* theta;    // [num_nodes] Phase angle
    float* sigma;    // [num_nodes] Processing rate
    float* P;        // [num_nodes] Presence
    uint32_t* k;     // [num_nodes] Event count
    uint32_t* flags; // [num_nodes] Status flags
} NodeArrays;

typedef struct {
    uint32_t* node_i;  // [num_bonds] First node
    uint32_t* node_j;  // [num_bonds] Second node
    float* C;          // [num_bonds] Coherence
    float* pi;         // [num_bonds] Momentum
    float* sigma;      // [num_bonds] Conductivity
} BondArrays;
```

**GPU Execution Model**:
```
Per-Tick Dispatch:
  Kernel 1: NODE_READ    (1 thread per node)
  Kernel 2: BOND_READ    (1 thread per bond)
  Kernel 3: PROPOSE      (all lanes)
  Kernel 4: CHOOSE       (all lanes)
  Kernel 5: BOND_COMMIT  (1 thread per bond, atomic)
  Kernel 6: NODE_COMMIT  (1 thread per node)
```

### Layer 1: DET Physics (Existence-Lang)

The core physics, written in Existence-Lang, compiled to EIS bytecode.

**File: `physics.ex`**

Contains:
- `kernel Transfer` - antisymmetric resource movement
- `kernel Diffuse` - symmetric flux exchange
- `kernel ComputePresence` - presence from agency and coordination
- `kernel UpdateCoherence` - phase alignment and coherence dynamics
- `kernel UpdateAgencyCeiling` - agency ceiling from structural debt
- `kernel GraceFlow` - bond-local grace protocol
- `kernel Distinct` - create two distinct identities
- `kernel Compare` - trace measurement

### Layer 2: DET-OS Kernel (Existence-Lang)

Uses Layer 1 physics to implement OS functions.

**File: `kernel.ex`**

Contains:
- `kernel Schedule` - presence-based scheduling
- `kernel Allocate` - resource allocation via transfer
- `kernel Gate` - agency-gated operations
- `kernel Grace` - grace distribution
- `kernel IPC` - inter-creature communication via bonds

### Layer 3: Applications (Existence-Lang)

User-space creatures that run on the kernel.

---

## Part 2: Compilation Targets

### Multiple Backends

```
                    ┌───────────────────────────────────────────┐
                    │         Existence-Lang Source             │
                    │   (physics.ex, kernel.ex, apps/*.ex)      │
                    └─────────────────────┬─────────────────────┘
                                          │
                              ┌───────────┴───────────┐
                              │     Compiler Front    │
                              │   (Parser, Analyzer)  │
                              └───────────┬───────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │   EIS Backend   │         │    C Backend    │         │   GPU Backend   │
    │   (bytecode)    │         │   (transpile)   │         │ (compute shader)│
    └────────┬────────┘         └────────┬────────┘         └────────┬────────┘
             │                           │                           │
             ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │   .eis files    │         │   .c/.h files   │         │  .glsl/.spv     │
    │   (portable)    │         │   (portable)    │         │   (GPU native)  │
    └────────┬────────┘         └────────┬────────┘         └────────┬────────┘
             │                           │                           │
             ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │  EIS Substrate  │         │   C Compiler    │         │  Vulkan/CUDA    │
    │  (interpreter)  │         │  (gcc/clang)    │         │  (high perf)    │
    └─────────────────┘         └─────────────────┘         └─────────────────┘
```

### Recommended Backend by Scenario

| Scenario | Backend | Reason |
|----------|---------|--------|
| Development/debugging | EIS bytecode | Easy inspection |
| Production (CPU) | C transpiler | Native performance |
| Embedded (ESP32) | C transpiler | Low overhead |
| Simulation (GPU) | Compute shader | Massive parallelism |
| DET hardware | EIS bytecode | Direct execution |

---

## Part 3: Key Design Decisions

### 3.1 Agency Gates Proposals, Not Memory

```
WRONG (traditional permission model):
  if (a > threshold) allow_memory_access()

RIGHT (DET model):
  score = sqrt(a_i * a_j) * Q_ij
  PROP.SCORE proposal, score
  CHOOSE selects based on score
  // Agency affects dynamics, not raw access
```

### 3.2 Phases Prevent Present-Time Branching

```
READ phase:    Load past trace → registers
PROPOSE phase: Emit proposals with scores (all evaluated, no branching)
CHOOSE phase:  Deterministic selection from local seed
COMMIT phase:  Apply effects, emit witnesses
```

No `if (x > 0) then ...` - only `if_past(token == GT) then ...`

### 3.3 Proposals Are Data, Not Control Flow

```existence
// Proposals are buffer entries, not branches
proposal XFER {
    score = sqrt(a_src * a_dst);
    effect = EFFECT_XFER_F(src, dst, amount);
}

proposal REFUSE {
    score = 1.0 - sqrt(a_src * a_dst);
    effect = EFFECT_NONE;
}

// CHOOSE selects one deterministically - no divergent branches
choice = CHOOSE(proposals, decisiveness, seed);
```

### 3.4 Conservation by Construction

Bond-lanes process each bond exactly once:
```
BOND_COMMIT kernel:
  // Compute G_ij once
  G_ij = accepted_ij - accepted_ji

  // Atomic antisymmetric update
  atomicAdd(F[i], -G_ij)
  atomicAdd(F[j], +G_ij)
```

### 3.5 Phase Alignment Without Transcendentals

Option: Store phase as unit vector, not angle:
```
// Instead of θ (requires COS):
theta: float

// Store (cos(θ), sin(θ)):
cos_theta: float
sin_theta: float

// Alignment = dot product:
align = cos_i * cos_j + sin_i * sin_j
// Just MUL + ADD, no COS
```

---

## Part 4: File Structure (Target)

```
local_agency/
├── docs/
│   ├── ARCHITECTURE_V2.md      # This document
│   ├── SUBSTRATE_SPEC.md       # Substrate specification
│   ├── ROADMAP_V2.md           # Migration roadmap
│   └── API.md
│
├── src/
│   ├── substrate/              # Layer 0: DET-Aware Substrate (C)
│   │   ├── include/
│   │   │   ├── eis_substrate.h
│   │   │   ├── substrate_types.h
│   │   │   └── effect_table.h
│   │   ├── src/
│   │   │   ├── eis_substrate.c
│   │   │   ├── phase_control.c
│   │   │   ├── proposal_buffer.c
│   │   │   ├── memory_soa.c
│   │   │   └── io_boundary.c
│   │   ├── gpu/
│   │   │   ├── substrate.comp   # Vulkan compute shader
│   │   │   └── substrate.cu     # CUDA version
│   │   ├── CMakeLists.txt
│   │   └── tests/
│   │
│   ├── existence/              # Layers 1-3: Existence-Lang
│   │   ├── physics.ex          # Layer 1: DET Physics
│   │   ├── kernel.ex           # Layer 2: DET-OS Kernel
│   │   ├── stdlib/             # Standard library
│   │   │   ├── arithmetic.ex
│   │   │   ├── collections.ex
│   │   │   └── io.ex
│   │   └── apps/               # Layer 3: Applications
│   │       └── shell.ex
│   │
│   ├── compiler/               # Existence-Lang Compiler
│   │   ├── python/
│   │   │   ├── lexer.py
│   │   │   ├── parser.py
│   │   │   ├── codegen_eis.py  # EIS bytecode
│   │   │   ├── codegen_c.py    # C transpiler
│   │   │   └── codegen_gpu.py  # Compute shader
│   │   └── bytecode/
│   │       ├── physics.eis
│   │       └── kernel.eis
│   │
│   └── bridges/                # Python Bridges
│       ├── __init__.py
│       ├── bootstrap.py
│       ├── llm_bridge.py
│       ├── web_bridge.py
│       └── dev_tools.py
│
└── tests/
    ├── test_substrate.c        # Substrate unit tests
    ├── test_physics.py         # Physics tests
    ├── test_kernel.py          # Kernel tests
    └── test_integration.py     # Full stack tests
```

---

## Part 5: Migration from Current State

### Current State

| Component | Language | Issue |
|-----------|----------|-------|
| det_core.c | C | DET physics in C (wrong layer) |
| eis_vm.c | C | Has DET opcodes mixed in |
| Python OS | Python | Core runtime in Python |

### Target State

| Component | Language | Status |
|-----------|----------|--------|
| eis_substrate.c | C | DET-aware but no physics |
| physics.ex | Existence | All DET physics |
| kernel.ex | Existence | All OS functions |
| bridges/*.py | Python | External I/O only |

### Migration Path

1. **Phase 1**: Rewrite substrate as DET-aware (phases, proposals, effects)
2. **Phase 2**: Implement physics.ex using substrate primitives
3. **Phase 3**: Update kernel.ex to import physics.ex
4. **Phase 4**: Refactor Python to bridges only
5. **Phase 5**: Deprecate det_core.c
6. **Phase 6**: Add GPU backend
7. **Phase 7**: Integration testing

---

## Part 6: Hardware Path

```
                    EIS Substrate v2
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │   CPU   │       │   GPU   │       │DET ASIC │
   │Interpret│       │ Compute │       │ Native  │
   │  or JIT │       │ Shaders │       │ Silicon │
   └─────────┘       └─────────┘       └─────────┘

Same bytecode (.eis) runs on all backends.
Same physics.ex, kernel.ex, applications.
Only substrate implementation changes.
```

### DET-Native Hardware Features

Future silicon can implement:
- Hardware proposal buffers
- Native choose/commit units
- Antisymmetric memory operations
- Hardware phase sequencing
- Direct EIS opcode execution

---

## References

1. Exploration 09: DET-OS Feasibility
2. Exploration 10: Existence-Lang v1.1
3. SUBSTRATE_SPEC.md - Detailed substrate specification
4. ROADMAP_V2.md - Migration timeline
5. User design document - Register model, instruction encoding, memory model
