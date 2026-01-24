/**
 * EIS Substrate v2 - DET-Aware Execution Layer
 * =============================================
 *
 * Minimal execution substrate with DET-aware primitives:
 * - Phase-based execution (READ → PROPOSE → CHOOSE → COMMIT)
 * - Typed references (NodeRef, BondRef, FieldRef)
 * - Proposal buffers and deterministic selection
 * - Effect table for verified operations
 *
 * Contains NO DET physics - those are in physics.ex (Existence-Lang).
 *
 * GPU-ready design:
 * - Structure-of-Arrays memory layout
 * - No divergent branches (proposals are data)
 * - Phase-separated kernel dispatch
 */

#ifndef EIS_SUBSTRATE_V2_H
#define EIS_SUBSTRATE_V2_H

#include "substrate_types.h"
#include "effect_table.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * OPCODES (~30 core instructions)
 * ========================================================================== */

typedef enum {
    /* ----- Phase Control (0x00-0x0F) ----- */
    OP_NOP      = 0x00,     /* No operation */
    OP_HALT     = 0x01,     /* Stop execution */
    OP_YIELD    = 0x02,     /* Yield to scheduler */
    OP_TICK     = 0x03,     /* Advance to next tick */
    OP_PHASE_R  = 0x04,     /* Enter READ phase */
    OP_PHASE_P  = 0x05,     /* Enter PROPOSE phase */
    OP_PHASE_C  = 0x06,     /* Enter CHOOSE phase */
    OP_PHASE_X  = 0x07,     /* Enter COMMIT phase */

    /* ----- Typed Loads (0x10-0x1F) - READ phase ----- */
    OP_LDN      = 0x10,     /* Load node field: dst = nodes[ref].field */
    OP_LDB      = 0x11,     /* Load bond field: dst = bonds[ref].field */
    OP_LDNB     = 0x12,     /* Load neighbor field via bond */
    OP_LDI      = 0x13,     /* Load immediate: dst = imm */
    OP_LDI_F    = 0x14,     /* Load float immediate (uses ext word) */

    /* ----- Register Ops (0x20-0x2F) ----- */
    OP_MOV      = 0x20,     /* Copy: dst = src */
    OP_MOVR     = 0x21,     /* Ref to/from scalar */
    OP_MOVT     = 0x22,     /* Token to/from scalar */
    OP_TSET     = 0x23,     /* Set token: T[dst] = imm */
    OP_TGET     = 0x24,     /* Get token as scalar: dst = (float)T[src] */

    /* ----- Arithmetic (0x30-0x3F) ----- */
    OP_ADD      = 0x30,     /* dst = src0 + src1 */
    OP_SUB      = 0x31,     /* dst = src0 - src1 */
    OP_MUL      = 0x32,     /* dst = src0 * src1 */
    OP_DIV      = 0x33,     /* dst = src0 / src1 (safe: 0 if div-by-0) */
    OP_MAD      = 0x34,     /* dst = src0 * src1 + dst (fused multiply-add) */
    OP_NEG      = 0x35,     /* dst = -src0 */
    OP_ABS      = 0x36,     /* dst = |src0| */
    OP_SQRT     = 0x37,     /* dst = sqrt(max(0, src0)) */
    OP_MIN      = 0x38,     /* dst = min(src0, src1) */
    OP_MAX      = 0x39,     /* dst = max(src0, src1) */
    OP_RELU     = 0x3A,     /* dst = max(0, src0) */
    OP_CLAMP    = 0x3B,     /* dst = clamp(src0, 0, 1) - common case */

    /* ----- Comparison (0x40-0x4F) ----- */
    OP_CMP      = 0x40,     /* Compare: T[dst] = LT/EQ/GT */
    OP_CMPE     = 0x41,     /* Compare with epsilon */
    OP_TEQ      = 0x42,     /* Token equal: T[dst] = (T[src0] == T[src1]) */
    OP_TNE      = 0x43,     /* Token not equal */

    /* ----- Proposals (0x50-0x5F) - PROPOSE phase ----- */
    OP_PROP_NEW     = 0x50, /* Begin new proposal: H[dst] = new proposal */
    OP_PROP_SCORE   = 0x51, /* Set score: proposals[H[dst]].score = src0 */
    OP_PROP_EFFECT  = 0x52, /* Set effect: proposals[H[dst]].effect = ... */
    OP_PROP_ARG     = 0x53, /* Add argument to current proposal */
    OP_PROP_END     = 0x54, /* Finalize proposal */

    /* ----- Choose/Commit (0x60-0x6F) ----- */
    OP_CHOOSE   = 0x60,     /* Select proposal: H[dst] = chosen index */
    OP_COMMIT   = 0x61,     /* Apply chosen effect */
    OP_WITNESS  = 0x62,     /* Emit witness: T[dst] = witness token */

    /* ----- Stores (0x70-0x7F) - COMMIT phase only ----- */
    OP_STN      = 0x70,     /* Store node field: nodes[ref].field = src */
    OP_STB      = 0x71,     /* Store bond field: bonds[ref].field = src */
    OP_STT      = 0x72,     /* Store token: token_store[ref] = T[src] */

    /* ----- I/O (0x80-0x8F) ----- */
    OP_IN       = 0x80,     /* Read from channel: dst = io[imm] */
    OP_OUT      = 0x81,     /* Write to channel: io[imm] = src0 */
    OP_EMIT     = 0x82,     /* Emit byte to buffer: buf[H[dst]].append(src0) */
    OP_POLL     = 0x83,     /* Poll channel: T[dst] = io_ready(imm) */

    /* ----- System (0xF0-0xFF) ----- */
    OP_RAND     = 0xF0,     /* Random [0,1): dst = random() */
    OP_SEED     = 0xF1,     /* Set seed: seed = src0 */
    OP_LANE     = 0xF2,     /* Get lane ID: dst = lane_id */
    OP_TIME     = 0xF3,     /* Get tick: dst = tick */
    OP_DEBUG    = 0xFE,     /* Debug breakpoint */
    OP_INVALID  = 0xFF,     /* Invalid opcode */
} SubstrateOpcode;

/* ==========================================================================
 * SUBSTRATE VM
 * ========================================================================== */

typedef struct SubstrateVM {
    /* ----- Per-Lane State ----- */
    SubstrateRegs regs;         /* Register file */
    uint32_t lane_id;           /* Current lane ID */
    SubstratePhase phase;       /* Current execution phase */
    uint64_t seed;              /* Random seed for this lane */
    LaneOwnershipMode lane_mode; /* Effect deduplication mode */

    /* ----- Program ----- */
    const uint8_t* program;
    uint32_t program_size;
    uint32_t pc;                /* Program counter (byte offset for raw, instr index for predecoded) */

    /* ----- Predecoded Program (optional, faster execution) ----- */
    PredecodedProgram* predecoded;
    bool use_predecoded;        /* True to use predecoded path */

    /* ----- Trace Memory (Shared) ----- */
    NodeArrays* nodes;
    BondArrays* bonds;

    /* ----- Proposal Buffer ----- */
    ProposalBuffer prop_buf;

    /* ----- Scratch Memory ----- */
    float* scratch;
    uint32_t scratch_size;

    /* ----- I/O ----- */
    SubstrateIOChannel io[SUB_NUM_IO_CHANNELS];

    /* ----- Boundary Buffers ----- */
    BoundaryBuffer* boundaries;
    uint32_t num_boundaries;

    /* ----- Execution State ----- */
    SubstrateState state;
    char error_msg[128];

    /* ----- Counters ----- */
    uint64_t tick;
    uint64_t instructions;

    /* ----- Trace/Debug ----- */
    bool trace_enabled;
    void (*trace_fn)(void* ctx, const SubstrateInstr* instr, const SubstrateRegs* regs);
    void* trace_ctx;
} SubstrateVM;

/* ==========================================================================
 * API FUNCTIONS
 * ========================================================================== */

/* ----- Lifecycle ----- */

/** Create a new substrate VM */
SubstrateVM* substrate_create(void);

/** Destroy VM and free resources */
void substrate_destroy(SubstrateVM* vm);

/** Reset VM state (keep program and trace memory) */
void substrate_reset(SubstrateVM* vm);

/* ----- Memory Setup ----- */

/** Allocate node arrays */
bool substrate_alloc_nodes(SubstrateVM* vm, uint32_t num_nodes);

/** Allocate bond arrays */
bool substrate_alloc_bonds(SubstrateVM* vm, uint32_t num_bonds);

/** Allocate scratch memory */
bool substrate_alloc_scratch(SubstrateVM* vm, uint32_t size);

/** Add a boundary buffer */
uint32_t substrate_add_boundary(SubstrateVM* vm, uint32_t capacity, bool readonly);

/* ----- Program Loading ----- */

/** Load program into VM */
bool substrate_load_program(SubstrateVM* vm, const uint8_t* program, uint32_t size);

/* ----- Execution ----- */

/** Set current lane ID */
void substrate_set_lane(SubstrateVM* vm, uint32_t lane_id);

/** Set lane ownership mode for effect deduplication */
void substrate_set_lane_mode(SubstrateVM* vm, LaneOwnershipMode mode);

/** Execute one instruction */
SubstrateState substrate_step(SubstrateVM* vm);

/** Execute until halt/yield/error or max_steps */
SubstrateState substrate_run(SubstrateVM* vm, uint32_t max_steps);

/** Resume execution after yield */
SubstrateState substrate_resume(SubstrateVM* vm, uint32_t max_steps);

/** Fast execution using threaded dispatch (GCC/Clang computed goto) */
SubstrateState substrate_run_fast(SubstrateVM* vm, uint32_t max_steps);

/** Execute a complete tick (all phases) */
SubstrateState substrate_tick(SubstrateVM* vm);

/* ----- Phase Control ----- */

/** Set execution phase */
void substrate_set_phase(SubstrateVM* vm, SubstratePhase phase);

/** Get current phase */
SubstratePhase substrate_get_phase(const SubstrateVM* vm);

/* ----- Register Access ----- */

float substrate_get_scalar(const SubstrateVM* vm, uint8_t reg);
void substrate_set_scalar(SubstrateVM* vm, uint8_t reg, float value);

uint32_t substrate_get_ref(const SubstrateVM* vm, uint8_t reg);
void substrate_set_ref(SubstrateVM* vm, uint8_t reg, uint32_t value);

uint32_t substrate_get_token(const SubstrateVM* vm, uint8_t reg);
void substrate_set_token(SubstrateVM* vm, uint8_t reg, uint32_t value);

/* ----- Node Access ----- */

float substrate_node_get_f(const SubstrateVM* vm, uint32_t node_id, NodeFieldId field);
void substrate_node_set_f(SubstrateVM* vm, uint32_t node_id, NodeFieldId field, float value);

uint32_t substrate_node_get_i(const SubstrateVM* vm, uint32_t node_id, NodeFieldId field);
void substrate_node_set_i(SubstrateVM* vm, uint32_t node_id, NodeFieldId field, uint32_t value);

/* ----- Bond Access ----- */

float substrate_bond_get_f(const SubstrateVM* vm, uint32_t bond_id, BondFieldId field);
void substrate_bond_set_f(SubstrateVM* vm, uint32_t bond_id, BondFieldId field, float value);

uint32_t substrate_bond_get_i(const SubstrateVM* vm, uint32_t bond_id, BondFieldId field);
void substrate_bond_set_i(SubstrateVM* vm, uint32_t bond_id, BondFieldId field, uint32_t value);

/* ----- Proposal Operations ----- */

/** Begin a new proposal */
uint32_t substrate_prop_new(SubstrateVM* vm);

/** Set proposal score */
void substrate_prop_score(SubstrateVM* vm, uint32_t prop_idx, float score);

/** Set proposal effect */
void substrate_prop_effect(SubstrateVM* vm, uint32_t prop_idx, EffectId effect_id,
                           const uint32_t* args, uint32_t arg_count);

/** Choose from proposals (deterministic) */
uint32_t substrate_choose(SubstrateVM* vm, float decisiveness);

/** Commit chosen proposal */
uint32_t substrate_commit(SubstrateVM* vm);

/* ----- I/O ----- */

void substrate_io_configure(SubstrateVM* vm, uint8_t channel,
                            float (*read_fn)(void*),
                            void (*write_fn)(void*, float),
                            void* ctx);
void substrate_io_inject(SubstrateVM* vm, uint8_t channel, float value);
float substrate_io_read(const SubstrateVM* vm, uint8_t channel);

/* ----- Boundary Buffers ----- */

void substrate_boundary_emit_byte(SubstrateVM* vm, uint32_t buf_id, uint8_t byte);
uint32_t substrate_boundary_read(const SubstrateVM* vm, uint32_t buf_id,
                                  uint8_t* out, uint32_t max_len);

/* ----- Instruction Encode/Decode ----- */

SubstrateInstr substrate_decode(const uint8_t* data, uint32_t offset, uint32_t* consumed);
SubstrateInstr substrate_decode_le(const uint8_t* data, uint32_t offset, uint32_t* consumed);
uint32_t substrate_encode(const SubstrateInstr* instr, uint8_t* out);
uint32_t substrate_encode_le(const SubstrateInstr* instr, uint8_t* out);

/* ----- Predecoding ----- */

/** Predecode program into instruction array for fast dispatch */
bool substrate_predecode(SubstrateVM* vm);

/** Free predecoded program */
void substrate_free_predecoded(SubstrateVM* vm);

/** Enable/disable predecoded execution path */
void substrate_use_predecoded(SubstrateVM* vm, bool enable);

/* ----- Debug/Utility ----- */

const char* substrate_opcode_name(uint8_t opcode);
const char* substrate_token_name(uint32_t token);
const char* substrate_state_name(SubstrateState state);
const char* substrate_phase_name(SubstratePhase phase);

void substrate_set_trace(SubstrateVM* vm,
                          void (*fn)(void*, const SubstrateInstr*, const SubstrateRegs*),
                          void* ctx);

void substrate_get_stats(const SubstrateVM* vm, uint64_t* tick, uint64_t* instructions);

void substrate_disassemble(const uint8_t* program, uint32_t size,
                            char* out, uint32_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* EIS_SUBSTRATE_V2_H */
