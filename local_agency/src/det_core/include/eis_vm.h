/**
 * EIS Virtual Machine - C Implementation
 * =======================================
 *
 * Native C implementation of the Existence Instruction Set VM.
 * Executes EIS bytecode with proper phase semantics.
 *
 * Architecture:
 *   - Two-tier register file: scalar regs + trace regs
 *   - Fixed 32-bit instruction encoding (RISC-like)
 *   - Explicit tick phases: READ → PROPOSE → CHOOSE → COMMIT
 *   - Integration with DET kernel for trace memory
 *
 * Instruction Format (32-bit):
 *   [opcode:8][dst:5][src0:5][src1:5][imm:9]
 */

#ifndef EIS_VM_H
#define EIS_VM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "det_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * CONFIGURATION
 * ========================================================================== */

#define EIS_MAX_SCALAR_REGS   16    /* R0-R15 */
#define EIS_MAX_REF_REGS      8     /* H0-H7 */
#define EIS_MAX_TOKEN_REGS    8     /* T0-T7 */
#define EIS_MAX_PROPOSALS     32    /* Per-lane proposal buffer */
#define EIS_MAX_EFFECTS       16    /* Effects per proposal */
#define EIS_SCRATCH_SIZE      256   /* Scratch memory words */
#define EIS_MAX_PROGRAM_SIZE  65536 /* Max bytecode size */

/* ==========================================================================
 * OPCODES
 * ========================================================================== */

typedef enum {
    /* Phase Control (0x00-0x0F) */
    EIS_OP_NOP = 0x00,
    EIS_OP_PHASE = 0x01,
    EIS_OP_HALT = 0x02,
    EIS_OP_YIELD = 0x03,
    EIS_OP_FENCE = 0x04,
    EIS_OP_TICK = 0x05,

    /* Load Operations (0x10-0x1F) */
    EIS_OP_LDI = 0x10,
    EIS_OP_LDI_EXT = 0x11,
    EIS_OP_LDN = 0x12,
    EIS_OP_LDB = 0x13,
    EIS_OP_LDNB = 0x14,
    EIS_OP_LDT = 0x15,
    EIS_OP_LDR = 0x16,
    EIS_OP_LDBUF = 0x17,

    /* Store Operations (0x20-0x2F) */
    EIS_OP_ST_TOK = 0x20,
    EIS_OP_ST_NODE = 0x21,
    EIS_OP_ST_BOND = 0x22,
    EIS_OP_ST_BUF = 0x23,

    /* Arithmetic (0x30-0x3F) */
    EIS_OP_ADD = 0x30,
    EIS_OP_SUB = 0x31,
    EIS_OP_MUL = 0x32,
    EIS_OP_DIV = 0x33,
    EIS_OP_MAD = 0x34,
    EIS_OP_NEG = 0x35,
    EIS_OP_ABS = 0x36,
    EIS_OP_MOD = 0x37,

    /* Math Functions (0x40-0x4F) */
    EIS_OP_SQRT = 0x40,
    EIS_OP_EXP = 0x41,
    EIS_OP_LOG = 0x42,
    EIS_OP_SIN = 0x43,
    EIS_OP_COS = 0x44,
    EIS_OP_MIN = 0x45,
    EIS_OP_MAX = 0x46,
    EIS_OP_CLAMP = 0x47,
    EIS_OP_RELU = 0x48,
    EIS_OP_SIGMOID = 0x49,

    /* Comparison and Token Ops (0x50-0x5F) */
    EIS_OP_CMP = 0x50,
    EIS_OP_CMP_EPS = 0x51,
    EIS_OP_TEQ = 0x52,
    EIS_OP_TNE = 0x53,
    EIS_OP_TMOV = 0x54,
    EIS_OP_TSET = 0x55,
    EIS_OP_TGET = 0x56,

    /* Proposal Operations (0x60-0x6F) */
    EIS_OP_PROP_BEGIN = 0x60,
    EIS_OP_PROP_SCORE = 0x61,
    EIS_OP_PROP_EFFECT = 0x62,
    EIS_OP_PROP_END = 0x63,
    EIS_OP_PROP_LIST = 0x64,

    /* Choose/Commit (0x70-0x7F) */
    EIS_OP_CHOOSE = 0x70,
    EIS_OP_COMMIT = 0x71,
    EIS_OP_COMMIT_ALL = 0x72,
    EIS_OP_WITNESS = 0x73,
    EIS_OP_ABORT = 0x74,

    /* Reference/Handle Ops (0x80-0x8F) */
    EIS_OP_MKNODE = 0x80,
    EIS_OP_MKBOND = 0x81,
    EIS_OP_MKFIELD = 0x82,
    EIS_OP_MKPROP = 0x83,
    EIS_OP_MKBUF = 0x84,
    EIS_OP_GETSELF = 0x85,
    EIS_OP_GETBOND = 0x86,
    EIS_OP_NEIGHBOR = 0x87,

    /* Conservation Primitives (0x90-0x9F) */
    EIS_OP_XFER = 0x90,
    EIS_OP_DIFFUSE = 0x91,
    EIS_OP_DISTINCT = 0x92,
    EIS_OP_COALESCE = 0x93,

    /* Grace Protocol (0xA0-0xAF) */
    EIS_OP_GRACE_NEED = 0xA0,
    EIS_OP_GRACE_EXCESS = 0xA1,
    EIS_OP_GRACE_OFFER = 0xA2,
    EIS_OP_GRACE_ACCEPT = 0xA3,
    EIS_OP_GRACE_COMMIT = 0xA4,

    /* Branching (0xB0-0xBF) */
    EIS_OP_BR_TOK = 0xB0,
    EIS_OP_BR_PHASE = 0xB1,
    EIS_OP_CALL = 0xB2,
    EIS_OP_RET = 0xB3,

    /* System/Debug (0xF0-0xFF) */
    EIS_OP_DEBUG = 0xF0,
    EIS_OP_TRACE = 0xF1,
    EIS_OP_ASSERT = 0xF2,
    EIS_OP_PRINT = 0xF3,
    EIS_OP_STATS = 0xF4,
    EIS_OP_INVALID = 0xFF
} EIS_Opcode;

/* ==========================================================================
 * EXECUTION PHASES
 * ========================================================================== */

typedef enum {
    EIS_PHASE_IDLE = 0,
    EIS_PHASE_READ = 1,
    EIS_PHASE_PROPOSE = 2,
    EIS_PHASE_CHOOSE = 3,
    EIS_PHASE_COMMIT = 4
} EIS_Phase;

/* ==========================================================================
 * WITNESS TOKENS
 * ========================================================================== */

typedef enum {
    EIS_TOK_VOID = 0x0000,

    /* Comparison */
    EIS_TOK_LT = 0x0001,
    EIS_TOK_EQ = 0x0002,
    EIS_TOK_GT = 0x0003,

    /* Reconciliation */
    EIS_TOK_EQ_OK = 0x0010,
    EIS_TOK_EQ_FAIL = 0x0011,
    EIS_TOK_EQ_REFUSE = 0x0012,

    /* Transfer */
    EIS_TOK_XFER_OK = 0x0020,
    EIS_TOK_XFER_PARTIAL = 0x0021,
    EIS_TOK_XFER_BLOCKED = 0x0022,

    /* Write */
    EIS_TOK_WRITE_OK = 0x0030,
    EIS_TOK_WRITE_REFUSED = 0x0031,

    /* Proposal */
    EIS_TOK_PROP_ACCEPTED = 0x0040,
    EIS_TOK_PROP_REJECTED = 0x0041,

    /* Grace */
    EIS_TOK_GRACE_OFFERED = 0x0050,
    EIS_TOK_GRACE_ACCEPTED = 0x0051,
    EIS_TOK_GRACE_DECLINED = 0x0052,

    /* Boolean */
    EIS_TOK_TRUE = 0xFFFE,
    EIS_TOK_FALSE = 0xFFFF
} EIS_WitnessToken;

/* ==========================================================================
 * NODE FIELD IDs
 * ========================================================================== */

typedef enum {
    EIS_FIELD_F = 0,
    EIS_FIELD_Q = 1,
    EIS_FIELD_A = 2,
    EIS_FIELD_THETA = 3,
    EIS_FIELD_R = 4,
    EIS_FIELD_K = 5,
    EIS_FIELD_FLAGS = 6,
    EIS_FIELD_TOK0 = 8,
    EIS_FIELD_TOK1 = 9,
    EIS_FIELD_TOK2 = 10,
    EIS_FIELD_TOK3 = 11,
    EIS_FIELD_TOK4 = 12,
    EIS_FIELD_TOK5 = 13,
    EIS_FIELD_TOK6 = 14,
    EIS_FIELD_TOK7 = 15,
    EIS_FIELD_SIGMA = 16,
    EIS_FIELD_P = 17,
    EIS_FIELD_TAU = 18,
    EIS_FIELD_H = 19
} EIS_NodeField;

typedef enum {
    EIS_BOND_C = 0,
    EIS_BOND_PI = 1,
    EIS_BOND_SIGMA = 2,
    EIS_BOND_FLAGS = 3,
    EIS_BOND_TOK0 = 8,
    EIS_BOND_TOK1 = 9
} EIS_BondField;

/* ==========================================================================
 * EXECUTION STATE
 * ========================================================================== */

typedef enum {
    EIS_STATE_RUNNING = 0,
    EIS_STATE_HALTED = 1,
    EIS_STATE_YIELDED = 2,
    EIS_STATE_ERROR = 3,
    EIS_STATE_BREAKPOINT = 4
} EIS_ExecState;

/* ==========================================================================
 * EFFECT TYPES
 * ========================================================================== */

typedef enum {
    EIS_EFFECT_NOP = 0x00,
    EIS_EFFECT_XFER_F = 0x01,
    EIS_EFFECT_SET_TOKEN = 0x10,
    EIS_EFFECT_SET_NODE_FIELD = 0x20,
    EIS_EFFECT_SET_BOND_FIELD = 0x30,
    EIS_EFFECT_WITNESS = 0xFF
} EIS_EffectType;

/* ==========================================================================
 * DATA STRUCTURES
 * ========================================================================== */

/** Decoded instruction */
typedef struct {
    uint8_t opcode;
    uint8_t dst;     /* 5 bits */
    uint8_t src0;    /* 5 bits */
    uint8_t src1;    /* 5 bits */
    int16_t imm;     /* 9 bits sign-extended */
    uint32_t ext;    /* Extension word */
    bool has_ext;
} EIS_Instruction;

/** Effect to be applied in COMMIT */
typedef struct {
    EIS_EffectType type;
    uint16_t src_node;
    uint16_t dst_node;
    float amount;
    uint16_t target_ref;
    uint8_t field_id;
    int32_t value;
} EIS_Effect;

/** Proposal */
typedef struct {
    float score;
    EIS_Effect effects[EIS_MAX_EFFECTS];
    uint32_t num_effects;
    bool committed;
} EIS_Proposal;

/** Register file for one lane */
typedef struct {
    /* Scalar registers R0-R15 */
    float scalars[EIS_MAX_SCALAR_REGS];

    /* Reference registers H0-H7 (store node/bond IDs) */
    uint32_t refs[EIS_MAX_REF_REGS];
    uint8_t ref_types[EIS_MAX_REF_REGS];  /* 0=node, 1=bond, 2=void */

    /* Token registers T0-T7 */
    uint32_t tokens[EIS_MAX_TOKEN_REGS];

    /* Self reference */
    uint16_t self_node;
    uint32_t self_bond;  /* Packed (i, j, layer) */
} EIS_RegisterFile;

/** Proposal buffer */
typedef struct {
    EIS_Proposal proposals[EIS_MAX_PROPOSALS];
    uint32_t num_proposals;
    uint32_t choices[EIS_MAX_PROPOSALS];  /* choice_id -> proposal_index */
    uint32_t num_choices;
} EIS_ProposalBuffer;

/** Lane execution context */
typedef struct {
    uint32_t lane_id;
    uint8_t lane_type;  /* 0=node, 1=bond */

    /* Register file */
    EIS_RegisterFile regs;

    /* Proposal buffer */
    EIS_ProposalBuffer proposals;

    /* Scratch memory */
    float scratch[EIS_SCRATCH_SIZE];

    /* Program state */
    uint32_t pc;
    const uint8_t *program;
    uint32_t program_size;

    /* Execution state */
    EIS_ExecState state;
    char error_msg[128];

    /* Current proposal being built */
    int32_t current_proposal;  /* -1 if none */
} EIS_Lane;

/** Phase controller */
typedef struct {
    EIS_Phase current_phase;
    uint32_t tick;
    bool strict;  /* Enforce phase rules */
    uint32_t violation_count;
} EIS_PhaseController;

/** EIS Virtual Machine */
typedef struct {
    /* Phase control */
    EIS_PhaseController phases;

    /* Lanes */
    EIS_Lane *node_lanes;
    uint32_t num_node_lanes;
    EIS_Lane *bond_lanes;
    uint32_t num_bond_lanes;

    /* Connection to DET kernel */
    DETCore *det;

    /* Global state */
    uint32_t tick;
    bool halted;
    bool trace_execution;

    /* Kernel library (indexed by ID) */
    const uint8_t **kernels;
    uint32_t *kernel_sizes;
    uint32_t num_kernels;

    /* Statistics */
    uint64_t instructions_executed;
    uint64_t phases_completed;
} EIS_VM;

/* ==========================================================================
 * API FUNCTIONS
 * ========================================================================== */

/** Initialize VM with DET core connection */
EIS_VM* eis_vm_create(DETCore *det);

/** Destroy VM and free resources */
void eis_vm_destroy(EIS_VM *vm);

/** Create a node lane */
EIS_Lane* eis_vm_create_node_lane(EIS_VM *vm, uint16_t node_id,
                                   const uint8_t *program, uint32_t size);

/** Create a bond lane */
EIS_Lane* eis_vm_create_bond_lane(EIS_VM *vm, uint16_t node_i, uint16_t node_j,
                                   const uint8_t *program, uint32_t size);

/** Step one instruction in a lane */
EIS_ExecState eis_vm_step_lane(EIS_VM *vm, EIS_Lane *lane);

/** Run lane until halt/yield/error */
EIS_ExecState eis_vm_run_lane(EIS_VM *vm, EIS_Lane *lane, uint32_t max_steps);

/** Run a complete tick across all lanes */
void eis_vm_run_tick(EIS_VM *vm);

/** Register a kernel program */
uint32_t eis_vm_register_kernel(EIS_VM *vm, const uint8_t *program, uint32_t size);

/* ==========================================================================
 * INSTRUCTION DECODE/ENCODE
 * ========================================================================== */

/** Decode instruction from bytecode */
EIS_Instruction eis_decode_instruction(const uint8_t *data, uint32_t offset,
                                        uint32_t *consumed);

/** Encode instruction to bytecode */
uint32_t eis_encode_instruction(const EIS_Instruction *instr, uint8_t *out);

/* ==========================================================================
 * UTILITY FUNCTIONS
 * ========================================================================== */

/** Get opcode name string */
const char* eis_opcode_name(uint8_t opcode);

/** Get phase name string */
const char* eis_phase_name(EIS_Phase phase);

/** Get token name string */
const char* eis_token_name(uint32_t token);

/** Disassemble bytecode to text */
void eis_disassemble(const uint8_t *program, uint32_t size,
                     char *out, uint32_t out_size);

#ifdef __cplusplus
}
#endif

#endif /* EIS_VM_H */
