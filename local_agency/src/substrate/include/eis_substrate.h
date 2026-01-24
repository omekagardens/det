/**
 * EIS Substrate - Minimal Execution Layer
 * =======================================
 *
 * Pure instruction execution substrate with NO DET semantics.
 * DET physics are implemented in Existence-Lang on top of this.
 *
 * Architecture:
 *   - 32 scalar registers (R0-R31, floats)
 *   - 16 reference registers (H0-H15, uint32)
 *   - 16 token registers (T0-T15, uint32)
 *   - Linear memory array (configurable size)
 *   - 16 I/O channels
 *   - 32-bit instruction encoding: [opcode:8][dst:5][src0:5][src1:5][imm:9]
 *
 * This layer provides ONLY:
 *   - Arithmetic (add, sub, mul, div, etc.)
 *   - Math functions (sqrt, sin, cos, etc.)
 *   - Comparison (produces tokens)
 *   - Control flow (jump, call, ret)
 *   - Memory load/store
 *   - Raw I/O
 *
 * This layer does NOT provide:
 *   - Phase semantics (READ/PROPOSE/CHOOSE/COMMIT)
 *   - Coherence, presence, agency
 *   - Transfer, diffuse, grace
 *   - Proposals, witnesses
 *   - Any DET physics
 *
 * Those are implemented in physics.ex (Existence-Lang) on top of this.
 */

#ifndef EIS_SUBSTRATE_H
#define EIS_SUBSTRATE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * CONFIGURATION
 * ========================================================================== */

#define EIS_NUM_SCALAR_REGS     32      /* R0-R31 */
#define EIS_NUM_REF_REGS        16      /* H0-H15 */
#define EIS_NUM_TOKEN_REGS      16      /* T0-T15 */
#define EIS_NUM_IO_CHANNELS     16      /* I/O channels */
#define EIS_DEFAULT_MEMORY_SIZE 65536   /* 64K words default */
#define EIS_MAX_PROGRAM_SIZE    262144  /* 256K bytes */
#define EIS_CALL_STACK_SIZE     256     /* Max call depth */

/* ==========================================================================
 * OPCODES - Pure Execution Only
 * ========================================================================== */

typedef enum {
    /* Control (0x00-0x0F) */
    SUB_OP_NOP      = 0x00,     /* No operation */
    SUB_OP_HALT     = 0x01,     /* Stop execution */
    SUB_OP_YIELD    = 0x02,     /* Yield (return control, can resume) */
    SUB_OP_TICK     = 0x03,     /* Increment tick counter */
    SUB_OP_FENCE    = 0x04,     /* Memory barrier */
    SUB_OP_DEBUG    = 0x0F,     /* Breakpoint */

    /* Load Immediate (0x10-0x1F) */
    SUB_OP_LDI      = 0x10,     /* Load 9-bit signed immediate */
    SUB_OP_LDI_HI   = 0x11,     /* Load immediate high 16 bits */
    SUB_OP_LDI_LO   = 0x12,     /* Load immediate low 16 bits */
    SUB_OP_LDI_F    = 0x13,     /* Load immediate as float (next word) */

    /* Memory (0x20-0x2F) */
    SUB_OP_LD       = 0x20,     /* Load from memory: dst = mem[src0 + imm] */
    SUB_OP_ST       = 0x21,     /* Store to memory: mem[dst + imm] = src0 */
    SUB_OP_LDR      = 0x22,     /* Load from ref-indexed: dst = mem[H[src0] + imm] */
    SUB_OP_STR      = 0x23,     /* Store to ref-indexed: mem[H[dst] + imm] = src0 */
    SUB_OP_MOV      = 0x24,     /* Move: dst = src0 */
    SUB_OP_MOVR     = 0x25,     /* Move to/from ref register */
    SUB_OP_MOVT     = 0x26,     /* Move to/from token register */
    SUB_OP_PUSH     = 0x27,     /* Push to stack */
    SUB_OP_POP      = 0x28,     /* Pop from stack */

    /* Arithmetic (0x30-0x3F) */
    SUB_OP_ADD      = 0x30,     /* dst = src0 + src1 */
    SUB_OP_SUB      = 0x31,     /* dst = src0 - src1 */
    SUB_OP_MUL      = 0x32,     /* dst = src0 * src1 */
    SUB_OP_DIV      = 0x33,     /* dst = src0 / src1 (safe: 0 if div by 0) */
    SUB_OP_MOD      = 0x34,     /* dst = src0 % src1 */
    SUB_OP_NEG      = 0x35,     /* dst = -src0 */
    SUB_OP_ABS      = 0x36,     /* dst = |src0| */
    SUB_OP_ADDI     = 0x37,     /* dst = src0 + imm */
    SUB_OP_MULI     = 0x38,     /* dst = src0 * imm */
    SUB_OP_INC      = 0x39,     /* dst = dst + 1 */
    SUB_OP_DEC      = 0x3A,     /* dst = dst - 1 */

    /* Math Functions (0x40-0x4F) */
    SUB_OP_SQRT     = 0x40,     /* dst = sqrt(src0) */
    SUB_OP_EXP      = 0x41,     /* dst = exp(src0) */
    SUB_OP_LOG      = 0x42,     /* dst = log(src0) */
    SUB_OP_SIN      = 0x43,     /* dst = sin(src0) */
    SUB_OP_COS      = 0x44,     /* dst = cos(src0) */
    SUB_OP_TAN      = 0x45,     /* dst = tan(src0) */
    SUB_OP_ASIN     = 0x46,     /* dst = asin(src0) */
    SUB_OP_ACOS     = 0x47,     /* dst = acos(src0) */
    SUB_OP_ATAN     = 0x48,     /* dst = atan(src0) */
    SUB_OP_ATAN2    = 0x49,     /* dst = atan2(src0, src1) */
    SUB_OP_POW      = 0x4A,     /* dst = pow(src0, src1) */
    SUB_OP_FLOOR    = 0x4B,     /* dst = floor(src0) */
    SUB_OP_CEIL     = 0x4C,     /* dst = ceil(src0) */
    SUB_OP_ROUND    = 0x4D,     /* dst = round(src0) */
    SUB_OP_FMOD     = 0x4E,     /* dst = fmod(src0, src1) */

    /* Min/Max/Clamp (0x50-0x5F) */
    SUB_OP_MIN      = 0x50,     /* dst = min(src0, src1) */
    SUB_OP_MAX      = 0x51,     /* dst = max(src0, src1) */
    SUB_OP_CLAMP    = 0x52,     /* dst = clamp(src0, lo, hi) - uses imm for bounds index */
    SUB_OP_RELU     = 0x53,     /* dst = max(0, src0) */
    SUB_OP_SIGN     = 0x54,     /* dst = sign(src0) → -1, 0, or 1 */
    SUB_OP_LERP     = 0x55,     /* dst = src0 + t*(src1-src0), t in dst */

    /* Comparison (0x60-0x6F) - Produces token in T[dst-48] */
    SUB_OP_CMP      = 0x60,     /* Compare: T[dst] = LT/EQ/GT */
    SUB_OP_CMPI     = 0x61,     /* Compare with immediate */
    SUB_OP_TEQ      = 0x62,     /* Token equal: T[dst] = (T[src0] == T[src1]) */
    SUB_OP_TNE      = 0x63,     /* Token not equal */
    SUB_OP_TSET     = 0x64,     /* Set token: T[dst] = imm */
    SUB_OP_TGET     = 0x65,     /* Get token as float: dst = (float)T[src0] */
    SUB_OP_ISNAN    = 0x66,     /* T[dst] = isnan(src0) ? TRUE : FALSE */
    SUB_OP_ISINF    = 0x67,     /* T[dst] = isinf(src0) ? TRUE : FALSE */

    /* Bitwise (0x70-0x7F) - Interpret float bits as int */
    SUB_OP_AND      = 0x70,     /* dst = src0 & src1 (bitwise) */
    SUB_OP_OR       = 0x71,     /* dst = src0 | src1 */
    SUB_OP_XOR      = 0x72,     /* dst = src0 ^ src1 */
    SUB_OP_NOT      = 0x73,     /* dst = ~src0 */
    SUB_OP_SHL      = 0x74,     /* dst = src0 << imm */
    SUB_OP_SHR      = 0x75,     /* dst = src0 >> imm (unsigned) */
    SUB_OP_SAR      = 0x76,     /* dst = src0 >> imm (signed) */
    SUB_OP_FTOI     = 0x77,     /* Float to int */
    SUB_OP_ITOF     = 0x78,     /* Int to float */

    /* Reference Operations (0x80-0x8F) */
    SUB_OP_HREF     = 0x80,     /* Load ref: H[dst] = imm (or from src0) */
    SUB_OP_HGET     = 0x81,     /* Get ref as float: dst = (float)H[src0] */
    SUB_OP_HSET     = 0x82,     /* Set ref from float: H[dst] = (uint32)src0 */
    SUB_OP_HADD     = 0x83,     /* Add to ref: H[dst] += imm */
    SUB_OP_HMOV     = 0x84,     /* Move ref: H[dst] = H[src0] */

    /* Branching (0x90-0x9F) */
    SUB_OP_JMP      = 0x90,     /* Unconditional jump: PC += imm (sign-extended) */
    SUB_OP_JZ       = 0x91,     /* Jump if zero: if src0 == 0, PC += imm */
    SUB_OP_JNZ      = 0x92,     /* Jump if not zero: if src0 != 0, PC += imm */
    SUB_OP_JLT      = 0x93,     /* Jump if less than: if src0 < 0, PC += imm */
    SUB_OP_JGT      = 0x94,     /* Jump if greater than: if src0 > 0, PC += imm */
    SUB_OP_JTOK     = 0x95,     /* Jump on token: if T[src0] == imm, PC += ext */
    SUB_OP_CALL     = 0x96,     /* Call subroutine: push PC, PC = addr */
    SUB_OP_RET      = 0x97,     /* Return: pop PC */
    SUB_OP_JMPR     = 0x98,     /* Jump to ref: PC = H[src0] */

    /* I/O (0xA0-0xAF) */
    SUB_OP_IN       = 0xA0,     /* Read from channel: dst = io_read(imm) */
    SUB_OP_OUT      = 0xA1,     /* Write to channel: io_write(imm, src0) */
    SUB_OP_POLL     = 0xA2,     /* Poll channel: T[dst] = io_ready(imm) ? TRUE : FALSE */
    SUB_OP_INBLK    = 0xA3,     /* Block input: read multiple words */
    SUB_OP_OUTBLK   = 0xA4,     /* Block output: write multiple words */

    /* System (0xF0-0xFF) */
    SUB_OP_TRACE    = 0xF0,     /* Trace output (debug) */
    SUB_OP_PRINT    = 0xF1,     /* Print register (debug) */
    SUB_OP_TIME     = 0xF2,     /* Get tick counter */
    SUB_OP_RAND     = 0xF3,     /* Random number: dst = random [0,1) */
    SUB_OP_SEED     = 0xF4,     /* Set random seed */
    SUB_OP_INVALID  = 0xFF      /* Invalid opcode */
} SubstrateOpcode;

/* ==========================================================================
 * TOKEN VALUES
 * ========================================================================== */

typedef enum {
    SUB_TOK_VOID    = 0x0000,
    SUB_TOK_FALSE   = 0x0001,
    SUB_TOK_TRUE    = 0x0002,
    SUB_TOK_LT      = 0x0010,   /* Less than */
    SUB_TOK_EQ      = 0x0011,   /* Equal */
    SUB_TOK_GT      = 0x0012,   /* Greater than */
    SUB_TOK_NAN     = 0x0020,   /* Not a number */
    SUB_TOK_INF     = 0x0021,   /* Infinity */
    SUB_TOK_ERR     = 0xFFFF    /* Error */
} SubstrateToken;

/* ==========================================================================
 * EXECUTION STATE
 * ========================================================================== */

typedef enum {
    SUB_STATE_RUNNING   = 0,    /* Executing */
    SUB_STATE_HALTED    = 1,    /* Stopped (HALT) */
    SUB_STATE_YIELDED   = 2,    /* Paused (YIELD) */
    SUB_STATE_ERROR     = 3,    /* Error occurred */
    SUB_STATE_WAITING   = 4,    /* Waiting on I/O */
    SUB_STATE_BREAKPOINT = 5    /* Hit breakpoint */
} SubstrateState;

/* ==========================================================================
 * DATA STRUCTURES
 * ========================================================================== */

/** Decoded instruction */
typedef struct {
    uint8_t opcode;
    uint8_t dst;        /* 5 bits: 0-31 */
    uint8_t src0;       /* 5 bits: 0-31 */
    uint8_t src1;       /* 5 bits: 0-31 */
    int16_t imm;        /* 9 bits sign-extended */
    uint32_t ext;       /* Extension word (for LDI_F, JTOK, etc.) */
    bool has_ext;       /* Has extension word */
} SubstrateInstr;

/** I/O Channel */
typedef struct {
    float value;                /* Current value */
    bool ready;                 /* Data available */
    bool enabled;               /* Channel enabled */

    /* Callback for external I/O */
    float (*read_fn)(void* ctx);
    void (*write_fn)(void* ctx, float value);
    void* user_ctx;
} SubstrateIOChannel;

/** Register file */
typedef struct {
    float scalars[EIS_NUM_SCALAR_REGS];     /* R0-R31 */
    uint32_t refs[EIS_NUM_REF_REGS];        /* H0-H15 */
    uint32_t tokens[EIS_NUM_TOKEN_REGS];    /* T0-T15 */
} SubstrateRegs;

/** Call stack entry */
typedef struct {
    uint32_t return_addr;
} SubstrateStackFrame;

/** Substrate Virtual Machine */
typedef struct {
    /* Registers */
    SubstrateRegs regs;

    /* Program */
    const uint8_t* program;
    uint32_t program_size;
    uint32_t pc;                /* Program counter */

    /* Memory */
    float* memory;
    uint32_t memory_size;

    /* Call stack */
    SubstrateStackFrame call_stack[EIS_CALL_STACK_SIZE];
    uint32_t stack_ptr;         /* Stack pointer (grows down) */
    uint32_t call_depth;        /* Current call depth */

    /* I/O */
    SubstrateIOChannel io[EIS_NUM_IO_CHANNELS];

    /* State */
    SubstrateState state;
    char error_msg[128];

    /* Counters */
    uint64_t tick;
    uint64_t instructions_executed;

    /* Random state */
    uint64_t random_state;

    /* Trace/debug */
    bool trace_enabled;
    void (*trace_fn)(void* ctx, const SubstrateInstr* instr, const SubstrateRegs* regs);
    void* trace_ctx;
} SubstrateVM;

/* ==========================================================================
 * API FUNCTIONS
 * ========================================================================== */

/* ----- Lifecycle ----- */

/** Create a new substrate VM with given memory size */
SubstrateVM* substrate_create(uint32_t memory_size);

/** Destroy VM and free resources */
void substrate_destroy(SubstrateVM* vm);

/** Reset VM state (keep program loaded) */
void substrate_reset(SubstrateVM* vm);

/* ----- Program Loading ----- */

/** Load program into VM */
bool substrate_load_program(SubstrateVM* vm, const uint8_t* program, uint32_t size);

/* ----- Execution ----- */

/** Execute one instruction */
SubstrateState substrate_step(SubstrateVM* vm);

/** Execute until halt/yield/error or max_steps */
SubstrateState substrate_run(SubstrateVM* vm, uint32_t max_steps);

/** Resume execution after yield */
SubstrateState substrate_resume(SubstrateVM* vm, uint32_t max_steps);

/* ----- Register Access ----- */

/** Read scalar register */
float substrate_get_scalar(const SubstrateVM* vm, uint8_t reg);

/** Write scalar register */
void substrate_set_scalar(SubstrateVM* vm, uint8_t reg, float value);

/** Read reference register */
uint32_t substrate_get_ref(const SubstrateVM* vm, uint8_t reg);

/** Write reference register */
void substrate_set_ref(SubstrateVM* vm, uint8_t reg, uint32_t value);

/** Read token register */
uint32_t substrate_get_token(const SubstrateVM* vm, uint8_t reg);

/** Write token register */
void substrate_set_token(SubstrateVM* vm, uint8_t reg, uint32_t value);

/* ----- Memory Access ----- */

/** Read memory word */
float substrate_mem_read(const SubstrateVM* vm, uint32_t addr);

/** Write memory word */
void substrate_mem_write(SubstrateVM* vm, uint32_t addr, float value);

/** Read memory block */
void substrate_mem_read_block(const SubstrateVM* vm, uint32_t addr,
                               float* out, uint32_t count);

/** Write memory block */
void substrate_mem_write_block(SubstrateVM* vm, uint32_t addr,
                                const float* data, uint32_t count);

/* ----- I/O ----- */

/** Configure I/O channel with callbacks */
void substrate_io_configure(SubstrateVM* vm, uint8_t channel,
                            float (*read_fn)(void*),
                            void (*write_fn)(void*, float),
                            void* ctx);

/** Set I/O channel value (for external injection) */
void substrate_io_inject(SubstrateVM* vm, uint8_t channel, float value);

/** Read I/O channel value (for external consumption) */
float substrate_io_read(const SubstrateVM* vm, uint8_t channel);

/** Set I/O channel ready flag */
void substrate_io_set_ready(SubstrateVM* vm, uint8_t channel, bool ready);

/* ----- Instruction Encode/Decode ----- */

/** Decode instruction from bytecode */
SubstrateInstr substrate_decode(const uint8_t* data, uint32_t offset,
                                 uint32_t* consumed);

/** Encode instruction to bytecode */
uint32_t substrate_encode(const SubstrateInstr* instr, uint8_t* out);

/* ----- Debug/Utility ----- */

/** Get opcode name */
const char* substrate_opcode_name(uint8_t opcode);

/** Get token name */
const char* substrate_token_name(uint32_t token);

/** Get state name */
const char* substrate_state_name(SubstrateState state);

/** Set trace callback */
void substrate_set_trace(SubstrateVM* vm,
                          void (*fn)(void*, const SubstrateInstr*, const SubstrateRegs*),
                          void* ctx);

/** Disassemble program to text */
void substrate_disassemble(const uint8_t* program, uint32_t size,
                            char* out, uint32_t out_size);

/** Get statistics */
void substrate_get_stats(const SubstrateVM* vm,
                          uint64_t* tick, uint64_t* instructions);

#ifdef __cplusplus
}
#endif

#endif /* EIS_SUBSTRATE_H */
