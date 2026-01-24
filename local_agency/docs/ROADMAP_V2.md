# DET Local Agency: Roadmap v2.1 - DET-Aware Substrate Migration

**Version**: 2.1.0
**Date**: 2026-01-23
**Status**: Active Development

---

## Overview

This roadmap outlines the migration to **Architecture v2.1**: a DET-aware substrate with Existence-Lang first design. The substrate is GPU-ready with phase-aware execution, proposal buffers, and typed references.

---

## Current State Assessment

### What Exists Now

| Component | Location | Language | Status | Target |
|-----------|----------|----------|--------|--------|
| DET Core (physics) | `det_core/src/det_core.c` | C | Working | **DELETE** → physics.ex |
| EIS VM (v1) | `substrate/` | C | Working | **REWRITE** → v2 DET-aware |
| Existence-Lang Parser | `python/det/lang/` | Python | Working | Keep (compiler tool) |
| EIS Compiler | `python/det/lang/eis/` | Python | Working | Extend (new opcodes) |
| Kernel (Ex-Lang) | `python/det/os/existence/kernel.ex` | Existence | Partial | Complete |
| Runtime | `python/det/os/existence/runtime.py` | Python | Working | Refactor → bridge only |

### What Changes

1. **Substrate v1 → v2**: Pure execution → DET-aware (phases, proposals, typed refs)
2. **~80 opcodes → ~30 opcodes**: Minimal, GPU-friendly instruction set
3. **32+16+16 registers → 16+8+8**: Smaller, practical register file
4. **Generic LD/ST → Typed LDN/LDB/LDNB**: Phase-aware, locality-enforced
5. **No proposals → PROP.\*/CHOOSE/COMMIT**: First-class proposal mechanism
6. **AoS → SoA memory**: GPU-optimized structure-of-arrays layout

---

## Migration Phases

### Phase 0: Documentation ✓

**Status**: Complete

**Deliverables**:
- [x] ARCHITECTURE_V2.md (updated to v2.1)
- [x] SUBSTRATE_SPEC.md (new)
- [x] ROADMAP_V2.md (this document)

---

### Phase 1: Substrate v2 Implementation ✓

**Status**: Complete

**Goal**: Rewrite substrate as DET-aware, GPU-ready.

**Completed**:
- [x] Type definitions (substrate_types.h)
- [x] Effect table (effect_table.h)
- [x] Core instruction set (~50 opcodes)
- [x] VM structure with phase-based execution
- [x] Proposal system with scoring and effects
- [x] SoA memory layout for GPU readiness
- [x] Test suite: 31/31 passing

**Tasks**:

#### 1.1 Type Definitions

```c
// substrate_types.h

// Register file (per lane)
typedef struct {
    float scalars[16];      // R0-R15
    uint32_t refs[8];       // H0-H7 (NodeRef, BondRef, etc.)
    uint32_t tokens[8];     // T0-T7
} SubstrateRegs;

// Node state (SoA layout)
typedef struct {
    float* F;               // [num_nodes]
    float* q;               // [num_nodes]
    float* a;               // [num_nodes]
    float* theta;           // [num_nodes]
    float* cos_theta;       // [num_nodes] (optional: unit vector)
    float* sin_theta;       // [num_nodes] (optional: unit vector)
    float* sigma;           // [num_nodes]
    float* P;               // [num_nodes]
    uint32_t* k;            // [num_nodes]
    uint32_t* r;            // [num_nodes]
    uint32_t* flags;        // [num_nodes]
} NodeArrays;

// Bond state (SoA layout)
typedef struct {
    uint32_t* node_i;       // [num_bonds]
    uint32_t* node_j;       // [num_bonds]
    float* C;               // [num_bonds]
    float* pi;              // [num_bonds]
    float* sigma;           // [num_bonds]
    uint32_t* flags;        // [num_bonds]
} BondArrays;

// Proposal entry
typedef struct {
    float score;
    uint16_t effect_id;
    uint16_t arg_count;
    uint32_t args[4];       // Packed arguments
} Proposal;

// Execution phase
typedef enum {
    PHASE_READ = 0,
    PHASE_PROPOSE = 1,
    PHASE_CHOOSE = 2,
    PHASE_COMMIT = 3
} SubstratePhase;
```

#### 1.2 Effect Table

```c
// effect_table.h

typedef enum {
    EFFECT_NONE = 0x00,
    EFFECT_XFER_F = 0x01,       // Antisymmetric transfer
    EFFECT_DIFFUSE = 0x02,      // Symmetric flux
    EFFECT_SET_F = 0x03,        // Direct set
    EFFECT_ADD_F = 0x04,        // Increment
    EFFECT_SET_Q = 0x05,
    EFFECT_ADD_Q = 0x06,
    EFFECT_SET_C = 0x07,
    EFFECT_ADD_C = 0x08,
    EFFECT_SET_PI = 0x09,
    EFFECT_SET_THETA = 0x0A,
    EFFECT_EMIT_TOK = 0x10,
    EFFECT_EMIT_BYTE = 0x11,
} EffectId;

// Effect application (in COMMIT phase)
void effect_apply(SubstrateVM* vm, EffectId id, uint32_t* args);
```

#### 1.3 Core Instruction Set

```c
// eis_substrate.h

typedef enum {
    // Phase control (0x00-0x0F)
    OP_NOP = 0x00,
    OP_HALT = 0x01,
    OP_YIELD = 0x02,
    OP_TICK = 0x03,
    OP_PHASE_R = 0x04,      // Enter READ
    OP_PHASE_P = 0x05,      // Enter PROPOSE
    OP_PHASE_C = 0x06,      // Enter CHOOSE
    OP_PHASE_X = 0x07,      // Enter COMMIT

    // Typed loads (0x10-0x1F) - READ phase
    OP_LDN = 0x10,          // Load node field
    OP_LDB = 0x11,          // Load bond field
    OP_LDNB = 0x12,         // Load neighbor field
    OP_LDI = 0x13,          // Load immediate
    OP_LDI_F = 0x14,        // Load float immediate

    // Register ops (0x20-0x2F)
    OP_MOV = 0x20,
    OP_MOVR = 0x21,         // Ref to/from scalar
    OP_MOVT = 0x22,         // Token to/from scalar
    OP_TSET = 0x23,         // Set token
    OP_TGET = 0x24,         // Get token

    // Arithmetic (0x30-0x3F)
    OP_ADD = 0x30,
    OP_SUB = 0x31,
    OP_MUL = 0x32,
    OP_DIV = 0x33,
    OP_MAD = 0x34,          // Multiply-add
    OP_NEG = 0x35,
    OP_ABS = 0x36,
    OP_SQRT = 0x37,
    OP_MIN = 0x38,
    OP_MAX = 0x39,
    OP_RELU = 0x3A,
    OP_CLAMP = 0x3B,

    // Comparison (0x40-0x4F)
    OP_CMP = 0x40,          // Compare → token
    OP_CMPE = 0x41,         // Compare with epsilon
    OP_TEQ = 0x42,          // Token equal
    OP_TNE = 0x43,          // Token not equal

    // Proposals (0x50-0x5F) - PROPOSE phase
    OP_PROP_NEW = 0x50,
    OP_PROP_SCORE = 0x51,
    OP_PROP_EFFECT = 0x52,
    OP_PROP_END = 0x53,

    // Choose/Commit (0x60-0x6F)
    OP_CHOOSE = 0x60,
    OP_COMMIT = 0x61,
    OP_WITNESS = 0x62,

    // Stores (0x70-0x7F) - COMMIT phase
    OP_STN = 0x70,          // Store node field
    OP_STB = 0x71,          // Store bond field
    OP_STT = 0x72,          // Store token

    // I/O (0x80-0x8F)
    OP_IN = 0x80,
    OP_OUT = 0x81,
    OP_EMIT = 0x82,
    OP_POLL = 0x83,

    // System (0xF0-0xFF)
    OP_RAND = 0xF0,
    OP_SEED = 0xF1,
    OP_LANE = 0xF2,
    OP_TIME = 0xF3,
    OP_DEBUG = 0xFE,
    OP_INVALID = 0xFF,
} SubstrateOpcode;
```

#### 1.4 VM Structure

```c
// Substrate VM
typedef struct {
    // Per-lane state (current lane)
    SubstrateRegs regs;
    uint32_t lane_id;
    SubstratePhase phase;
    uint64_t seed;

    // Program
    const uint8_t* program;
    uint32_t program_size;
    uint32_t pc;

    // Trace memory (shared)
    NodeArrays* nodes;
    uint32_t num_nodes;
    BondArrays* bonds;
    uint32_t num_bonds;

    // Proposal buffers (per lane)
    Proposal* proposals;
    uint32_t num_proposals;
    uint32_t max_proposals;

    // Scratch memory
    float* scratch;
    uint32_t scratch_size;

    // I/O
    SubstrateIOChannel io[16];

    // State
    SubstrateState state;
    char error_msg[128];

    // Counters
    uint64_t tick;
    uint64_t instructions;
} SubstrateVM;
```

**Files Created**:
- `src/substrate/include/eis_substrate.h`
- `src/substrate/include/substrate_types.h`
- `src/substrate/include/effect_table.h`
- `src/substrate/src/eis_substrate.c`
- `src/substrate/src/phase_control.c`
- `src/substrate/src/proposal_buffer.c`
- `src/substrate/src/effect_apply.c`
- `src/substrate/src/memory_soa.c`
- `src/substrate/tests/test_substrate.c`

**Success Criteria**:
- [ ] Phase transitions work correctly
- [ ] Typed loads respect phase restrictions
- [ ] Proposal buffer operations work
- [ ] CHOOSE selects deterministically
- [ ] COMMIT applies effects correctly
- [ ] Conservation maintained for XFER_F

---

### Phase 2: Compiler Update

**Goal**: Update Existence-Lang compiler for v2 substrate.

**Tasks**:

#### 2.1 New Opcode Support

```python
# codegen_eis.py additions

# Phase control
def emit_phase_read(self):
    self.emit(OP_PHASE_R, 0, 0, 0, 0)

def emit_phase_propose(self):
    self.emit(OP_PHASE_P, 0, 0, 0, 0)

# Typed loads
def emit_ldn(self, dst, node_ref, field_id):
    self.emit(OP_LDN, dst, node_ref, field_id, 0)

def emit_ldb(self, dst, bond_ref, field_id):
    self.emit(OP_LDB, dst, bond_ref, field_id, 0)

# Proposals
def emit_prop_new(self, prop_ref):
    self.emit(OP_PROP_NEW, prop_ref, 0, 0, 0)

def emit_prop_score(self, prop_ref, score_reg):
    self.emit(OP_PROP_SCORE, prop_ref, score_reg, 0, 0)

def emit_prop_effect(self, prop_ref, effect_id, *args):
    self.emit(OP_PROP_EFFECT, prop_ref, effect_id, len(args), 0)
    for arg in args:
        self.emit_arg(arg)
```

#### 2.2 Kernel Compilation

Compile kernel syntax to proposal operations:

```existence
// Source
proposal XFER {
    score = sqrt(a_src * a_dst);
    effect {
        transfer(src, dst, amount);
    }
}

// Compiles to:
PROP.NEW H0
MUL R0, R1, R2          ; a_src * a_dst
SQRT R0, R0             ; sqrt
PROP.SCORE H0, R0
PROP.EFFECT H0, EFFECT_XFER_F, src, dst, amount
PROP.END H0
```

**Files Modified**:
- `src/compiler/python/codegen_eis.py`
- `src/compiler/python/semantic.py`

**Success Criteria**:
- [ ] All v2 opcodes emitted correctly
- [ ] Kernel blocks compile to proposals
- [ ] Phase blocks respected
- [ ] Effect arguments encoded correctly

---

### Phase 3: physics.ex Implementation ✓

**Status**: Complete

**Goal**: Implement DET physics in Existence-Lang using v2 substrate.

**Completed**:
- [x] Transfer kernel (antisymmetric resource movement)
- [x] Diffuse kernel (symmetric flux exchange)
- [x] Compare kernel (trace measurement)
- [x] Distinct kernel (create distinction)
- [x] Reconcile kernel (attempted unification)
- [x] GraceOffer/GraceAccept/GraceFlow kernels
- [x] ComputePresence kernel (P = F · C · a)
- [x] CoherenceDecay/Strengthen kernels
- [x] Physics bridge to substrate v2 (physics_bridge.py)

**Tasks**:

#### 3.1 Transfer Kernel

```existence
kernel Transfer {
    in  src: NodeRef;
    in  dst: NodeRef;
    in  amount: Register;
    out witness: TokenReg;

    phase READ {
        a_src ::= load(src, FIELD_A);
        a_dst ::= load(dst, FIELD_A);
        F_src ::= load(src, FIELD_F);
    }

    phase PROPOSE {
        proposal XFER {
            score = sqrt(a_src * a_dst);
            effect = EFFECT_XFER_F(src, dst, min(amount, F_src));
        }

        proposal REFUSE {
            score = 1.0 - sqrt(a_src * a_dst);
            effect = EFFECT_NONE;
        }
    }

    phase CHOOSE {
        choice ::= choose({XFER, REFUSE}, decisiveness=0.9, seed=local_seed());
    }

    phase COMMIT {
        commit(choice);
        witness ::= if_past(choice == XFER) then XFER_OK else XFER_REFUSED;
    }
}
```

#### 3.2 Diffuse Kernel

```existence
kernel Diffuse {
    in  bond: BondRef;
    out flux: Register;
    out witness: TokenReg;

    params { dt = 0.02; }

    phase READ {
        i ::= load(bond, FIELD_NODE_I);
        j ::= load(bond, FIELD_NODE_J);
        C ::= load(bond, FIELD_C);
        sigma ::= load(bond, FIELD_SIGMA);
        F_i ::= load_node(i, FIELD_F);
        F_j ::= load_node(j, FIELD_F);
    }

    phase PROPOSE {
        delta ::= sigma * (F_i - F_j) * dt;

        proposal FLOW {
            score = C;
            effect = EFFECT_DIFFUSE(bond, delta);
        }
    }

    phase CHOOSE {
        choice ::= choose({FLOW}, decisiveness=1.0, seed=local_seed());
    }

    phase COMMIT {
        commit(choice);
        flux ::= abs(delta);
        witness ::= DIFFUSE_OK;
    }
}
```

#### 3.3 ComputePresence Kernel

```existence
kernel ComputePresence {
    in  node: NodeRef;
    in  bonds: BondRef[];

    phase READ {
        a ::= load(node, FIELD_A);
        sigma ::= load(node, FIELD_SIGMA);
        F ::= load(node, FIELD_F);

        // Compute coordination load H
        H ::= 0.0;
        repeat_past(len(bonds)) {
            bond ::= bonds[cursor];
            C ::= load(bond, FIELD_C);
            s ::= load(bond, FIELD_SIGMA);
            H ::= H + sqrt(C) * s;
        }
    }

    phase PROPOSE {
        F_op ::= F * 0.1;
        P_new ::= a * sigma / (1.0 + F_op) / (1.0 + H);

        proposal COMPUTE {
            score = 1.0;
            effect = EFFECT_SET_F(node, FIELD_P, P_new);
        }
    }

    phase CHOOSE {
        commit(choose({COMPUTE}));
    }
}
```

#### 3.4 UpdateCoherence Kernel

```existence
kernel UpdateCoherence {
    in  bond: BondRef;

    params {
        alpha_C = 0.15;
        lambda_C = 0.02;
        slip_threshold = 0.3;
        dt = 0.02;
    }

    phase READ {
        i ::= load(bond, FIELD_NODE_I);
        j ::= load(bond, FIELD_NODE_J);
        C ::= load(bond, FIELD_C);
        pi ::= load(bond, FIELD_PI);

        // Phase alignment (using unit vectors)
        cos_i ::= load_node(i, FIELD_COS_THETA);
        sin_i ::= load_node(i, FIELD_SIN_THETA);
        cos_j ::= load_node(j, FIELD_COS_THETA);
        sin_j ::= load_node(j, FIELD_SIN_THETA);

        align ::= cos_i * cos_j + sin_i * sin_j;  // dot product = cos(θ_i - θ_j)
    }

    phase PROPOSE {
        slip ::= if_past(align < slip_threshold) then 1.0 else 0.0;
        dC ::= alpha_C * abs(pi) - lambda_C * C - slip * C;
        C_new ::= clamp(C + dC * dt, 0.0, 1.0);

        proposal UPDATE {
            score = 1.0;
            effect = EFFECT_SET_C(bond, C_new);
        }
    }

    phase CHOOSE {
        commit(choose({UPDATE}));
    }
}
```

#### 3.5 GraceFlow Kernel

```existence
kernel GraceFlow {
    in  bond: BondRef;

    params {
        beta_g = 0.8;
        eta_g = 0.5;
        C_quantum = 0.3;
    }

    phase READ {
        i ::= load(bond, FIELD_NODE_I);
        j ::= load(bond, FIELD_NODE_J);
        C ::= load(bond, FIELD_C);

        F_i ::= load_node(i, FIELD_F);
        F_j ::= load_node(j, FIELD_F);
        a_i ::= load_node(i, FIELD_A);
        a_j ::= load_node(j, FIELD_A);

        // Simplified local mean (full version uses neighbor aggregation)
        F_thresh_i ::= beta_g * F_i;
        F_thresh_j ::= beta_g * F_j;

        need_i ::= relu(F_thresh_i - F_i);
        excess_i ::= relu(F_i - F_thresh_i);
        need_j ::= relu(F_thresh_j - F_j);
        excess_j ::= relu(F_j - F_thresh_j);

        donor_cap_i ::= a_i * excess_i;
        recv_cap_i ::= a_i * need_i;
        donor_cap_j ::= a_j * excess_j;
        recv_cap_j ::= a_j * need_j;

        Q_ij ::= relu(1.0 - sqrt(C) / C_quantum);
        w_ij ::= sqrt(a_i * a_j) * Q_ij;
    }

    phase PROPOSE {
        offer_i_to_j ::= eta_g * donor_cap_i * w_ij * recv_cap_j / (w_ij * recv_cap_j + 0.001);
        offer_j_to_i ::= eta_g * donor_cap_j * w_ij * recv_cap_i / (w_ij * recv_cap_i + 0.001);
        G_ij ::= offer_i_to_j - offer_j_to_i;

        proposal GRACE {
            score = w_ij;
            effect = EFFECT_DIFFUSE(bond, G_ij);
        }
    }

    phase CHOOSE {
        commit(choose({GRACE}));
    }
}
```

**Files Created**:
- `src/existence/physics.ex`
- `src/compiler/bytecode/physics.eis`

**Success Criteria**:
- [ ] Transfer conserves F
- [ ] Diffuse is antisymmetric
- [ ] Presence formula correct
- [ ] Coherence stable
- [ ] Grace bounded

---

### Phase 4: kernel.ex Update

**Goal**: Update kernel.ex to import physics.ex.

**Tasks**:

```existence
// kernel.ex

import physics.{Transfer, Diffuse, ComputePresence, UpdateCoherence, GraceFlow}

kernel Schedule {
    // Use ComputePresence for scheduling
    ...
}

kernel Allocate {
    // Use Transfer for allocation
    ...
}

kernel Grace {
    // Use GraceFlow for grace distribution
    ...
}
```

**Files Modified**:
- `src/existence/kernel.ex`

---

### Phase 5: Python Bridge Refactor

**Goal**: Remove Python from core execution.

**Tasks**:

1. Create minimal bridges:
   - `bootstrap.py` - Load and run bytecode
   - `llm_bridge.py` - LLM I/O
   - `web_bridge.py` - HTTP interface

2. Delete Python OS modules:
   - ~~scheduler.py~~
   - ~~allocator.py~~
   - ~~gatekeeper.py~~
   - ~~ipc.py~~
   - ~~creature.py~~

---

### Phase 6: GPU Backend (Optional)

**Goal**: Generate compute shaders from Existence-Lang.

**Tasks**:

1. Create `codegen_gpu.py`
2. Generate GLSL/SPIR-V for Vulkan
3. Generate CUDA for NVIDIA

**GPU Dispatch Structure**:
```glsl
// node_read.comp
layout(local_size_x = 256) in;

void main() {
    uint lane = gl_GlobalInvocationID.x;
    if (lane >= num_nodes) return;

    // Load node fields
    float F = nodes_F[lane];
    float a = nodes_a[lane];
    // ... compute derived values
    // Store to scratch
    scratch_need[lane] = relu(F_thresh - F);
}
```

---

### Phase 7: Integration & Testing

**Goal**: Full stack validation.

**Tests**:
1. Phase transitions
2. Proposal selection determinism
3. Conservation laws
4. Coherence dynamics
5. Grace flow
6. GPU vs CPU equivalence

---

## Timeline Summary

| Phase | Description | Duration | Cumulative |
|-------|-------------|----------|------------|
| Phase 0 | Documentation | Done | Done |
| Phase 1 | Substrate v2 | 2-3 weeks | 2-3 weeks |
| Phase 2 | Compiler Update | 1-2 weeks | 3-5 weeks |
| Phase 3 | physics.ex | 2-3 weeks | 5-8 weeks |
| Phase 4 | kernel.ex | 1 week | 6-9 weeks |
| Phase 5 | Python Bridges | 1 week | 7-10 weeks |
| Phase 6 | GPU Backend | 2-3 weeks | 9-13 weeks |
| Phase 7 | Testing | 1-2 weeks | 10-15 weeks |

**Total: 10-15 weeks**

---

## Success Metrics

1. **Architecture**
   - [ ] Substrate is DET-aware (~30 opcodes)
   - [ ] No DET physics in C
   - [ ] No Python in core execution
   - [ ] GPU-ready memory layout

2. **Functionality**
   - [ ] All physics kernels work
   - [ ] Conservation maintained
   - [ ] Deterministic execution
   - [ ] Phase semantics enforced

3. **Performance**
   - [ ] CPU: Within 2x of old C
   - [ ] GPU: 10-100x speedup for large graphs

---

## References

1. ARCHITECTURE_V2.md
2. SUBSTRATE_SPEC.md
3. Exploration 09: DET-OS Feasibility
4. Exploration 10: Existence-Lang v1.1
5. User design document (register model, memory model, execution model)
