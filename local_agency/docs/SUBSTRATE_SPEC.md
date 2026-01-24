# EIS Substrate v2: DET-Aware, GPU-Ready Specification

**Version**: 2.0.0
**Date**: 2026-01-23
**Status**: Specification

---

## Executive Summary

The EIS Substrate is the minimal execution layer for DET-OS. It is **DET-aware** (phases, proposals, typed refs) but contains **no DET physics** (transfer, diffuse, coherence are in Existence-Lang).

Key design goals:
1. **Minimal**: ~30 core opcodes
2. **DET-Aware**: Phase model, proposals, typed references
3. **GPU-Ready**: SIMD-friendly, no divergent branches, coalesced memory
4. **Hardware Path**: Same bytecode runs on CPU, GPU, or future DET silicon

---

## Part 1: Register Model

### 1.1 Register File (Per Lane)

Each execution lane (node-lane or bond-lane) has:

```
SCALAR REGISTERS: R0-R15 (16 registers)
  - Type: F32 (32-bit float)
  - Purpose: Working values, intermediates
  - Ephemeral: Reset each tick

REFERENCE REGISTERS: H0-H7 (8 registers)
  - Type: 32-bit typed reference
  - Subtypes:
    - NodeRef: Node ID (0..MAX_NODES)
    - BondRef: Bond ID (0..MAX_BONDS)
    - FieldRef: Packed {kind:2, field_id:6, flags:8, offset:16}
    - PropRef: Proposal buffer index
    - BufRef: Boundary buffer handle

TOKEN REGISTERS: T0-T7 (8 registers)
  - Type: Tok32 (32-bit token)
  - Values: LT, EQ, GT, TRUE, FALSE, OK, FAIL, REFUSE, etc.
  - Purpose: Witness tokens, comparison results
```

### 1.2 Type Summary

| Type | Bits | Description |
|------|------|-------------|
| F32 | 32 | IEEE 754 float |
| I32 | 32 | Signed integer (loop counts, indices) |
| Tok32 | 32 | Token enum value |
| NodeRef | 32 | Node ID |
| BondRef | 32 | Bond ID or packed (i, j, layer) |
| FieldRef | 32 | Packed field descriptor |
| PropRef | 32 | Proposal buffer index |
| BufRef | 32 | Boundary buffer handle |

### 1.3 Register vs Trace

| Location | Purpose | Lifetime |
|----------|---------|----------|
| Registers | Computations, intermediates | One phase |
| Trace | Falsifiable truth (F, q, a, C, π, tokens) | Persistent |

**Rule**: Registers are ephemeral working space. Trace is canonical truth.

---

## Part 2: Memory Model

### 2.1 Memory Regions

```
┌─────────────────────────────────────────────────────────────┐
│                     TRACE STORE                             │
│  Authoritative, falsifiable state                           │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Node Arrays    │  │  Bond Arrays    │                   │
│  │  F[N], q[N]     │  │  C[B], π[B]     │                   │
│  │  a[N], θ[N]     │  │  σ[B], i[B]     │                   │
│  │  σ[N], P[N]     │  │  j[B], flags[B] │                   │
│  │  k[N], r[N]     │  │                 │                   │
│  └─────────────────┘  └─────────────────┘                   │
│  ┌─────────────────────────────────────┐                    │
│  │  Token Store (witness logs, tapes)  │                    │
│  └─────────────────────────────────────┘                    │
├─────────────────────────────────────────────────────────────┤
│                    SCRATCH MEMORY                           │
│  Ephemeral working storage                                  │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Per-Node       │  │  Per-Bond       │                   │
│  │  need[N]        │  │  offer_ij[B]    │                   │
│  │  excess[N]      │  │  weight[B]      │                   │
│  │  donor_cap[N]   │  │  flux[B]        │                   │
│  └─────────────────┘  └─────────────────┘                   │
├─────────────────────────────────────────────────────────────┤
│                   PROPOSAL BUFFERS                          │
│  Per-lane proposal staging                                  │
│  ┌─────────────────────────────────────┐                    │
│  │  Proposal = {score, effect_id, args}│                    │
│  │  Max 4-8 proposals per lane/tick    │                    │
│  └─────────────────────────────────────┘                    │
├─────────────────────────────────────────────────────────────┤
│                   BOUNDARY BUFFERS                          │
│  Readout-only by default                                    │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  Output Sinks   │  │  Input Sources  │                   │
│  │  stdout, files  │  │  sensors, stdin │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Layout (Structure-of-Arrays)

For GPU coalescing, all arrays are SoA:

```c
// Node state - Structure of Arrays
typedef struct {
    float* F;           // [num_nodes] Resource
    float* q;           // [num_nodes] Structural debt
    float* a;           // [num_nodes] Agency
    float* theta;       // [num_nodes] Phase angle
    float* sigma;       // [num_nodes] Processing rate
    float* P;           // [num_nodes] Presence (computed)
    float* tau;         // [num_nodes] Proper time
    uint32_t* k;        // [num_nodes] Event count
    uint32_t* r;        // [num_nodes] Reconciliation seed
    uint32_t* flags;    // [num_nodes] Status flags
} NodeArrays;

// Bond state - Structure of Arrays
typedef struct {
    uint32_t* node_i;   // [num_bonds] First node
    uint32_t* node_j;   // [num_bonds] Second node
    float* C;           // [num_bonds] Coherence
    float* pi;          // [num_bonds] Momentum
    float* sigma;       // [num_bonds] Conductivity
    uint32_t* flags;    // [num_bonds] Status flags
} BondArrays;
```

### 2.3 Access Rules

**Agency does NOT gate memory access. Agency gates proposal eligibility.**

| Operation | Phase | Rule |
|-----------|-------|------|
| Load node field | READ | Allowed for own node or neighbors |
| Load bond field | READ | Allowed for incident bonds |
| Write to trace | COMMIT only | Via verified effects |
| Write to scratch | Any | Lane-local only |

**Locality Enforcement**:
- Node-lane `i` may access: node `i`, bonds incident to `i`, neighbors via bonds
- Bond-lane `(i,j)` may access: nodes `i` and `j`, bond `(i,j)`

---

## Part 3: Phase Model

### 3.1 Canonical Tick Phases

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 0: READ                                              │
│  - Load trace fields into registers                         │
│  - Compute derived quantities                               │
│  - Produce measurement tokens                               │
│  - NO writes to trace                                       │
├─────────────────────────────────────────────────────────────┤
│  PHASE 1: PROPOSE                                           │
│  - Emit proposals with scores                               │
│  - Effects not applied yet (staged in buffers)              │
│  - All proposals evaluated (no branching)                   │
├─────────────────────────────────────────────────────────────┤
│  PHASE 2: CHOOSE                                            │
│  - Deterministic selection from proposals                   │
│  - Uses only past trace + local seed (k, r, θ)              │
│  - Never uses global state                                  │
├─────────────────────────────────────────────────────────────┤
│  PHASE 3: COMMIT                                            │
│  - Apply chosen effects                                     │
│  - All trace writes happen here                             │
│  - Emit witness tokens                                      │
│  - Antisymmetric effects applied once per bond              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Phase Enforcement

The substrate enforces phase semantics:
- `LDN`, `LDB`, `LDNB` only allowed in READ phase
- `PROP.*` only allowed in PROPOSE phase
- `CHOOSE` only allowed in CHOOSE phase
- `COMMIT`, `WITNESS` only allowed in COMMIT phase
- Arithmetic allowed in any phase (operates on registers only)

### 3.3 GPU Dispatch Model

```
Per-Tick GPU Dispatch:

  ┌──────────────────┐
  │ Kernel 1:        │  All node-lanes execute READ
  │ NODE_READ        │  1 thread per node
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │ Kernel 2:        │  All bond-lanes execute READ
  │ BOND_READ        │  1 thread per bond
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │ Kernel 3:        │  All lanes emit proposals
  │ PROPOSE          │  No branching, all proposals evaluated
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │ Kernel 4:        │  Deterministic selection
  │ CHOOSE           │  Can be fused with COMMIT
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │ Kernel 5:        │  Bond effects (antisymmetric, once)
  │ BOND_COMMIT      │  Atomic adds for conservation
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │ Kernel 6:        │  Node effects, witness tokens
  │ NODE_COMMIT      │
  └──────────────────┘
```

---

## Part 4: Instruction Set

### 4.1 Encoding Format

Fixed 32-bit encoding (RISC-style):

```
┌────────┬───────┬───────┬───────┬───────────┐
│ opcode │  dst  │ src0  │ src1  │    imm    │
│  8 bit │ 5 bit │ 5 bit │ 5 bit │   9 bit   │
└────────┴───────┴───────┴───────┴───────────┘

- Opcode: 256 possible (using ~30)
- dst/src: Register index (0-31, uses 16 scalar + 8 ref + 8 token)
- imm: 9-bit signed immediate (-256 to +255)

Extension word (optional, for large immediates):
┌────────────────────────────────────────────┐
│              32-bit extension              │
└────────────────────────────────────────────┘
```

### 4.2 Core Instruction Set (~30 opcodes)

#### Phase Control (0x00-0x0F)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x00 | NOP | - | No operation |
| 0x01 | HALT | - | Stop execution |
| 0x02 | YIELD | - | Yield to scheduler |
| 0x03 | TICK | - | Advance to next tick |
| 0x04 | PHASE.R | - | Enter READ phase |
| 0x05 | PHASE.P | - | Enter PROPOSE phase |
| 0x06 | PHASE.C | - | Enter CHOOSE phase |
| 0x07 | PHASE.X | - | Enter COMMIT phase |

#### Typed Loads (0x10-0x1F)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x10 | LDN | dst, nodeRef, field | Load node.field → dst |
| 0x11 | LDB | dst, bondRef, field | Load bond.field → dst |
| 0x12 | LDNB | dst, nodeRef, nbrIdx, field | Load neighbor field |
| 0x13 | LDI | dst, imm | Load immediate |
| 0x14 | LDI.F | dst, [ext] | Load float immediate (ext word) |

#### Register Ops (0x20-0x2F)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x20 | MOV | dst, src | Copy register |
| 0x21 | MOVR | dst, srcRef | Ref to scalar (as float) |
| 0x22 | MOVT | dst, srcTok | Token to scalar |
| 0x23 | TSET | dstTok, imm | Set token value |
| 0x24 | TGET | dst, srcTok | Get token as scalar |

#### Arithmetic (0x30-0x3F)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x30 | ADD | dst, src0, src1 | dst = src0 + src1 |
| 0x31 | SUB | dst, src0, src1 | dst = src0 - src1 |
| 0x32 | MUL | dst, src0, src1 | dst = src0 * src1 |
| 0x33 | DIV | dst, src0, src1 | dst = src0 / src1 (safe) |
| 0x34 | MAD | dst, src0, src1, src2 | dst = src0 * src1 + src2 |
| 0x35 | NEG | dst, src | dst = -src |
| 0x36 | ABS | dst, src | dst = |src| |
| 0x37 | SQRT | dst, src | dst = sqrt(max(0, src)) |
| 0x38 | MIN | dst, src0, src1 | dst = min(src0, src1) |
| 0x39 | MAX | dst, src0, src1 | dst = max(src0, src1) |
| 0x3A | RELU | dst, src | dst = max(0, src) |
| 0x3B | CLAMP | dst, src, lo, hi | dst = clamp(src, lo, hi) |

#### Comparison (0x40-0x4F)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x40 | CMP | dstTok, src0, src1 | Compare → LT/EQ/GT token |
| 0x41 | CMPE | dstTok, src0, src1, eps | Compare with epsilon |
| 0x42 | TEQ | dstTok, tok0, tok1 | Token equality |
| 0x43 | TNE | dstTok, tok0, tok1 | Token inequality |

#### Proposal Ops (0x50-0x5F)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x50 | PROP.NEW | propRef | Begin new proposal |
| 0x51 | PROP.SCORE | propRef, scoreReg | Set proposal score |
| 0x52 | PROP.EFFECT | propRef, effectId, args | Attach effect |
| 0x53 | PROP.END | propRef | Finalize proposal |

#### Choose/Commit (0x60-0x6F)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x60 | CHOOSE | dstChoice, propList, dec, seed | Select proposal |
| 0x61 | COMMIT | choiceRef | Apply chosen effect |
| 0x62 | WITNESS | tokRef, witnessType | Emit witness token |

#### Stores (0x70-0x7F) - COMMIT phase only

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x70 | STN | nodeRef, field, src | Store to node.field |
| 0x71 | STB | bondRef, field, src | Store to bond.field |
| 0x72 | STT | tokRef, srcTok | Store token |

#### I/O (0x80-0x8F)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0x80 | IN | dst, channel | Read from I/O channel |
| 0x81 | OUT | channel, src | Write to I/O channel |
| 0x82 | EMIT | bufRef, byte | Append byte to buffer |
| 0x83 | POLL | dstTok, channel | Check channel ready |

#### System (0xF0-0xFF)

| Opcode | Mnemonic | Encoding | Description |
|--------|----------|----------|-------------|
| 0xF0 | RAND | dst | Random [0,1) from lane seed |
| 0xF1 | SEED | src | Set lane seed |
| 0xF2 | LANE | dst | Get lane ID |
| 0xF3 | TIME | dst | Get tick count |
| 0xFE | DEBUG | imm | Debug breakpoint |
| 0xFF | INVALID | - | Invalid opcode |

---

## Part 5: Effect Table

Effects are verified, antisymmetric operations. They are NOT opcodes - they are data attached to proposals.

### 5.1 Effect Definitions

| Effect ID | Name | Args | Description |
|-----------|------|------|-------------|
| 0x01 | XFER_F | src_node, dst_node, amount | Transfer F (antisymmetric) |
| 0x02 | DIFFUSE | bond, delta | Symmetric flux on bond |
| 0x03 | SET_F | node, value | Set node.F directly |
| 0x04 | ADD_F | node, delta | Add to node.F |
| 0x05 | SET_Q | node, value | Set node.q |
| 0x06 | ADD_Q | node, delta | Add to node.q |
| 0x07 | SET_C | bond, value | Set bond.C |
| 0x08 | ADD_C | bond, delta | Add to bond.C |
| 0x09 | SET_PI | bond, value | Set bond.π |
| 0x0A | SET_THETA | node, value | Set node.θ |
| 0x10 | EMIT_TOK | tokRef, value | Emit witness token |
| 0x11 | EMIT_BYTE | bufRef, byte | Emit byte to boundary |

### 5.2 Conservation Guarantees

- `XFER_F`: Applied once per bond, atomically subtracts from src and adds to dst
- `DIFFUSE`: Computes delta once, applies antisymmetrically
- All effects are deterministic given the same inputs

---

## Part 6: Token Values

### 6.1 Standard Tokens

```c
// Comparison tokens
TOK_LT      = 0x0010    // Less than
TOK_EQ      = 0x0011    // Equal
TOK_GT      = 0x0012    // Greater than

// Boolean tokens
TOK_FALSE   = 0x0001
TOK_TRUE    = 0x0002

// Witness tokens
TOK_OK      = 0x0100    // Operation succeeded
TOK_FAIL    = 0x0101    // Operation failed
TOK_REFUSE  = 0x0102    // Agency refused
TOK_PARTIAL = 0x0103    // Partial success

// Transfer witnesses
TOK_XFER_OK      = 0x0200
TOK_XFER_REFUSED = 0x0201
TOK_XFER_PARTIAL = 0x0202

// Diffuse witnesses
TOK_DIFFUSE_OK   = 0x0210

// Grace witnesses
TOK_GRACE_OK     = 0x0220
TOK_GRACE_NONE   = 0x0221

// Special
TOK_VOID    = 0x0000    // No value
TOK_ERR     = 0xFFFF    // Error
```

---

## Part 7: Field IDs

### 7.1 Node Fields

| ID | Name | Type | Description |
|----|------|------|-------------|
| 0x00 | F | F32 | Resource |
| 0x01 | q | F32 | Structural debt |
| 0x02 | a | F32 | Agency |
| 0x03 | theta | F32 | Phase angle |
| 0x04 | sigma | F32 | Processing rate |
| 0x05 | P | F32 | Presence (computed) |
| 0x06 | tau | F32 | Proper time |
| 0x07 | k | I32 | Event count |
| 0x08 | r | I32 | Reconciliation seed |
| 0x09 | flags | I32 | Status flags |

### 7.2 Bond Fields

| ID | Name | Type | Description |
|----|------|------|-------------|
| 0x00 | C | F32 | Coherence |
| 0x01 | pi | F32 | Momentum |
| 0x02 | sigma | F32 | Conductivity |
| 0x03 | node_i | NodeRef | First node |
| 0x04 | node_j | NodeRef | Second node |
| 0x05 | flags | I32 | Status flags |

---

## Part 8: Phase Alignment Alternative

For efficient phase alignment without COS:

### 8.1 Unit Vector Representation

Store phase as unit vector instead of angle:

```
Traditional:     θ (angle)
Unit vector:     (cos_θ, sin_θ)

Alignment = cos(θ_i - θ_j)
          = cos_i * cos_j + sin_i * sin_j
          = dot product (just MUL + ADD)
```

### 8.2 Node Fields (Extended)

| ID | Name | Type | Description |
|----|------|------|-------------|
| 0x0A | cos_theta | F32 | cos(θ) |
| 0x0B | sin_theta | F32 | sin(θ) |

### 8.3 Alignment Computation

```
// Without COS instruction:
MUL tmp1, cos_i, cos_j      // cos_i * cos_j
MUL tmp2, sin_i, sin_j      // sin_i * sin_j
ADD align, tmp1, tmp2       // alignment = dot product
```

This uses 2x storage but avoids transcendentals entirely.

---

## Part 9: Implementation Notes

### 9.1 CPU Implementation

- Interpreter loop with switch dispatch
- Optional JIT compilation
- Per-lane state in thread-local storage

### 9.2 GPU Implementation

- Separate compute shaders per phase
- Atomic operations for commit
- Coalesced memory access via SoA layout
- Shared memory for per-block scratch

### 9.3 DET Hardware

- Direct execution of EIS opcodes
- Hardware proposal buffers
- Hardware choose/commit units
- Native antisymmetric memory operations

---

## References

1. Existence-Lang v1.1 Specification
2. DET Theory Card v6.3
3. ARCHITECTURE_V2.md
4. User design document (register model, instruction encoding, memory model, execution model)
