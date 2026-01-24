/**
 * EIS Substrate v2 - Type Definitions
 * ====================================
 *
 * DET-aware types for the minimal execution layer.
 * GPU-ready with Structure-of-Arrays layout.
 */

#ifndef SUBSTRATE_TYPES_H
#define SUBSTRATE_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * CONFIGURATION
 * ========================================================================== */

#define SUB_NUM_SCALAR_REGS     16      /* R0-R15 */
#define SUB_NUM_REF_REGS        8       /* H0-H7 */
#define SUB_NUM_TOKEN_REGS      8       /* T0-T7 */
#define SUB_NUM_IO_CHANNELS     16
#define SUB_MAX_PROPOSALS       8       /* Per lane */
#define SUB_MAX_EFFECT_ARGS     4
#define SUB_DEFAULT_SCRATCH     4096

/* ==========================================================================
 * REGISTER FILE (Per Lane)
 * ========================================================================== */

typedef struct {
    float scalars[SUB_NUM_SCALAR_REGS];     /* R0-R15: Working values */
    uint32_t refs[SUB_NUM_REF_REGS];        /* H0-H7: NodeRef, BondRef, etc. */
    uint32_t tokens[SUB_NUM_TOKEN_REGS];    /* T0-T7: Token values */
} SubstrateRegs;

/* ==========================================================================
 * REFERENCE TYPES
 * ========================================================================== */

/* Reference type tags (upper 4 bits of ref) */
typedef enum {
    REF_TYPE_NODE   = 0x0,      /* Node ID */
    REF_TYPE_BOND   = 0x1,      /* Bond ID */
    REF_TYPE_FIELD  = 0x2,      /* Packed field descriptor */
    REF_TYPE_PROP   = 0x3,      /* Proposal buffer index */
    REF_TYPE_BUF    = 0x4,      /* Boundary buffer handle */
    REF_TYPE_CHOICE = 0x5,      /* Choice result */
} RefType;

/* Pack/unpack reference */
#define REF_MAKE(type, id)      (((uint32_t)(type) << 28) | ((id) & 0x0FFFFFFF))
#define REF_TYPE(ref)           ((RefType)((ref) >> 28))
#define REF_ID(ref)             ((ref) & 0x0FFFFFFF)

/* ==========================================================================
 * NODE STATE (Structure-of-Arrays for GPU)
 * ========================================================================== */

typedef struct {
    /* Core DET state */
    float* F;           /* [num_nodes] Resource */
    float* q;           /* [num_nodes] Structural debt */
    float* a;           /* [num_nodes] Agency [0,1] */
    float* sigma;       /* [num_nodes] Processing rate */
    float* P;           /* [num_nodes] Presence (computed) */
    float* tau;         /* [num_nodes] Proper time */

    /* Phase (unit vector representation for GPU efficiency) */
    float* cos_theta;   /* [num_nodes] cos(θ) */
    float* sin_theta;   /* [num_nodes] sin(θ) */

    /* Counters and flags */
    uint32_t* k;        /* [num_nodes] Event count */
    uint32_t* r;        /* [num_nodes] Reconciliation seed */
    uint32_t* flags;    /* [num_nodes] Status flags */

    /* Allocation info */
    uint32_t num_nodes;
    uint32_t capacity;
} NodeArrays;

/* Node field IDs */
typedef enum {
    NODE_FIELD_F = 0x00,
    NODE_FIELD_Q = 0x01,
    NODE_FIELD_A = 0x02,
    NODE_FIELD_SIGMA = 0x03,
    NODE_FIELD_P = 0x04,
    NODE_FIELD_TAU = 0x05,
    NODE_FIELD_COS_THETA = 0x06,
    NODE_FIELD_SIN_THETA = 0x07,
    NODE_FIELD_K = 0x08,
    NODE_FIELD_R = 0x09,
    NODE_FIELD_FLAGS = 0x0A,
} NodeFieldId;

/* ==========================================================================
 * BOND STATE (Structure-of-Arrays for GPU)
 * ========================================================================== */

typedef struct {
    /* Endpoints */
    uint32_t* node_i;   /* [num_bonds] First node */
    uint32_t* node_j;   /* [num_bonds] Second node */

    /* Bond state */
    float* C;           /* [num_bonds] Coherence [0,1] */
    float* pi;          /* [num_bonds] Momentum */
    float* sigma;       /* [num_bonds] Conductivity */
    uint32_t* flags;    /* [num_bonds] Status flags */

    /* Allocation info */
    uint32_t num_bonds;
    uint32_t capacity;
} BondArrays;

/* Bond field IDs */
typedef enum {
    BOND_FIELD_NODE_I = 0x00,
    BOND_FIELD_NODE_J = 0x01,
    BOND_FIELD_C = 0x02,
    BOND_FIELD_PI = 0x03,
    BOND_FIELD_SIGMA = 0x04,
    BOND_FIELD_FLAGS = 0x05,
} BondFieldId;

/* ==========================================================================
 * EXECUTION PHASE
 * ========================================================================== */

typedef enum {
    PHASE_READ = 0,     /* Load past trace, compute derived values */
    PHASE_PROPOSE = 1,  /* Emit proposals with scores */
    PHASE_CHOOSE = 2,   /* Deterministic selection */
    PHASE_COMMIT = 3,   /* Apply effects, emit witnesses */
} SubstratePhase;

/* ==========================================================================
 * TOKEN VALUES
 * ========================================================================== */

typedef enum {
    /* Special */
    TOK_VOID    = 0x0000,
    TOK_ERR     = 0xFFFF,

    /* Boolean */
    TOK_FALSE   = 0x0001,
    TOK_TRUE    = 0x0002,

    /* Comparison */
    TOK_LT      = 0x0010,
    TOK_EQ      = 0x0011,
    TOK_GT      = 0x0012,

    /* Operation results */
    TOK_OK      = 0x0100,
    TOK_FAIL    = 0x0101,
    TOK_REFUSE  = 0x0102,
    TOK_PARTIAL = 0x0103,

    /* Transfer witnesses */
    TOK_XFER_OK      = 0x0200,
    TOK_XFER_REFUSED = 0x0201,
    TOK_XFER_PARTIAL = 0x0202,

    /* Diffuse witnesses */
    TOK_DIFFUSE_OK   = 0x0210,

    /* Grace witnesses */
    TOK_GRACE_OK     = 0x0220,
    TOK_GRACE_NONE   = 0x0221,

    /* Coherence witnesses */
    TOK_COV_ALIGNED  = 0x0230,
    TOK_COV_DRIFT    = 0x0231,
    TOK_COV_BROKEN   = 0x0232,
} TokenValue;

/* ==========================================================================
 * PROPOSAL
 * ========================================================================== */

typedef struct {
    float score;                        /* Selection weight */
    uint16_t effect_id;                 /* Effect to apply */
    uint16_t arg_count;                 /* Number of arguments */
    uint32_t args[SUB_MAX_EFFECT_ARGS]; /* Packed arguments */
    bool valid;                         /* Is this proposal active? */
} Proposal;

/* Proposal buffer (per lane) */
typedef struct {
    Proposal proposals[SUB_MAX_PROPOSALS];
    uint32_t count;
    uint32_t chosen;    /* Index of chosen proposal after CHOOSE */
} ProposalBuffer;

/* ==========================================================================
 * I/O CHANNEL
 * ========================================================================== */

typedef struct {
    float value;
    bool ready;
    bool enabled;

    /* Callbacks for external I/O */
    float (*read_fn)(void* ctx);
    void (*write_fn)(void* ctx, float value);
    void* user_ctx;
} SubstrateIOChannel;

/* ==========================================================================
 * BOUNDARY BUFFER
 * ========================================================================== */

typedef struct {
    uint8_t* data;
    uint32_t size;
    uint32_t capacity;
    bool readonly;      /* If true, can only read; if false, can append */
} BoundaryBuffer;

/* ==========================================================================
 * EXECUTION STATE
 * ========================================================================== */

typedef enum {
    SUB_STATE_RUNNING   = 0,
    SUB_STATE_HALTED    = 1,
    SUB_STATE_YIELDED   = 2,
    SUB_STATE_ERROR     = 3,
    SUB_STATE_WAITING   = 4,
    SUB_STATE_BREAKPOINT = 5,
} SubstrateState;

/* ==========================================================================
 * DECODED INSTRUCTION
 * ========================================================================== */

typedef struct {
    uint8_t opcode;
    uint8_t dst;        /* 5 bits: register index */
    uint8_t src0;       /* 5 bits: register or field */
    uint8_t src1;       /* 5 bits: register or field */
    int16_t imm;        /* 9 bits sign-extended */
    uint32_t ext;       /* Extension word */
    bool has_ext;
} SubstrateInstr;

#ifdef __cplusplus
}
#endif

#endif /* SUBSTRATE_TYPES_H */
