/**
 * EIS Substrate - Minimal Execution Layer Implementation
 * ======================================================
 *
 * Pure instruction execution with NO DET semantics.
 */

#include "../include/eis_substrate.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ==========================================================================
 * INSTRUCTION DECODE/ENCODE
 * ========================================================================== */

SubstrateInstr substrate_decode(const uint8_t* data, uint32_t offset,
                                 uint32_t* consumed) {
    SubstrateInstr instr = {0};

    /* Read 32-bit word (big-endian) */
    uint32_t word = ((uint32_t)data[offset] << 24) |
                    ((uint32_t)data[offset + 1] << 16) |
                    ((uint32_t)data[offset + 2] << 8) |
                    ((uint32_t)data[offset + 3]);

    instr.opcode = (word >> 24) & 0xFF;
    instr.dst = (word >> 19) & 0x1F;
    instr.src0 = (word >> 14) & 0x1F;
    instr.src1 = (word >> 9) & 0x1F;
    instr.imm = word & 0x1FF;

    /* Sign-extend 9-bit immediate */
    if (instr.imm & 0x100) {
        instr.imm = instr.imm - 0x200;
    }

    *consumed = 4;

    /* Check if extension word needed */
    switch (instr.opcode) {
        case SUB_OP_LDI_F:
        case SUB_OP_JTOK:
        case SUB_OP_CALL:
            instr.has_ext = true;
            instr.ext = ((uint32_t)data[offset + 4] << 24) |
                        ((uint32_t)data[offset + 5] << 16) |
                        ((uint32_t)data[offset + 6] << 8) |
                        ((uint32_t)data[offset + 7]);
            *consumed = 8;
            break;
        default:
            instr.has_ext = false;
            break;
    }

    return instr;
}

uint32_t substrate_encode(const SubstrateInstr* instr, uint8_t* out) {
    uint32_t imm_bits = instr->imm & 0x1FF;
    uint32_t word = ((uint32_t)(instr->opcode & 0xFF) << 24) |
                    ((uint32_t)(instr->dst & 0x1F) << 19) |
                    ((uint32_t)(instr->src0 & 0x1F) << 14) |
                    ((uint32_t)(instr->src1 & 0x1F) << 9) |
                    (imm_bits & 0x1FF);

    out[0] = (word >> 24) & 0xFF;
    out[1] = (word >> 16) & 0xFF;
    out[2] = (word >> 8) & 0xFF;
    out[3] = word & 0xFF;

    if (instr->has_ext) {
        out[4] = (instr->ext >> 24) & 0xFF;
        out[5] = (instr->ext >> 16) & 0xFF;
        out[6] = (instr->ext >> 8) & 0xFF;
        out[7] = instr->ext & 0xFF;
        return 8;
    }

    return 4;
}

/* ==========================================================================
 * REGISTER ACCESS (INTERNAL)
 * ========================================================================== */

static inline float read_scalar(const SubstrateRegs* regs, uint8_t reg) {
    return (reg < EIS_NUM_SCALAR_REGS) ? regs->scalars[reg] : 0.0f;
}

static inline void write_scalar(SubstrateRegs* regs, uint8_t reg, float value) {
    if (reg < EIS_NUM_SCALAR_REGS) {
        regs->scalars[reg] = value;
    }
}

static inline uint32_t read_ref(const SubstrateRegs* regs, uint8_t reg) {
    return (reg < EIS_NUM_REF_REGS) ? regs->refs[reg] : 0;
}

static inline void write_ref(SubstrateRegs* regs, uint8_t reg, uint32_t value) {
    if (reg < EIS_NUM_REF_REGS) {
        regs->refs[reg] = value;
    }
}

static inline uint32_t read_token(const SubstrateRegs* regs, uint8_t reg) {
    return (reg < EIS_NUM_TOKEN_REGS) ? regs->tokens[reg] : SUB_TOK_VOID;
}

static inline void write_token(SubstrateRegs* regs, uint8_t reg, uint32_t value) {
    if (reg < EIS_NUM_TOKEN_REGS) {
        regs->tokens[reg] = value;
    }
}

/* ==========================================================================
 * RANDOM NUMBER GENERATOR (xorshift64)
 * ========================================================================== */

static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float random_float(SubstrateVM* vm) {
    uint64_t r = xorshift64(&vm->random_state);
    return (float)(r & 0xFFFFFFFF) / 4294967296.0f;
}

/* ==========================================================================
 * INSTRUCTION EXECUTION
 * ========================================================================== */

static void execute_instruction(SubstrateVM* vm, const SubstrateInstr* instr) {
    SubstrateRegs* regs = &vm->regs;
    float a, b, c;
    uint32_t addr;
    union { uint32_t u; float f; } conv;

    switch (instr->opcode) {

    /* === Control === */
    case SUB_OP_NOP:
        break;

    case SUB_OP_HALT:
        vm->state = SUB_STATE_HALTED;
        break;

    case SUB_OP_YIELD:
        vm->state = SUB_STATE_YIELDED;
        break;

    case SUB_OP_TICK:
        vm->tick++;
        break;

    case SUB_OP_FENCE:
        /* Memory barrier - no-op in single-threaded */
        break;

    case SUB_OP_DEBUG:
        vm->state = SUB_STATE_BREAKPOINT;
        break;

    /* === Load Immediate === */
    case SUB_OP_LDI:
        write_scalar(regs, instr->dst, (float)instr->imm);
        break;

    case SUB_OP_LDI_HI:
        conv.f = read_scalar(regs, instr->dst);
        conv.u = (conv.u & 0x0000FFFF) | ((uint32_t)(instr->imm & 0xFFFF) << 16);
        write_scalar(regs, instr->dst, conv.f);
        break;

    case SUB_OP_LDI_LO:
        conv.f = read_scalar(regs, instr->dst);
        conv.u = (conv.u & 0xFFFF0000) | ((uint32_t)(instr->imm & 0xFFFF));
        write_scalar(regs, instr->dst, conv.f);
        break;

    case SUB_OP_LDI_F:
        if (instr->has_ext) {
            conv.u = instr->ext;
            write_scalar(regs, instr->dst, conv.f);
        }
        break;

    /* === Memory === */
    case SUB_OP_LD:
        addr = (uint32_t)read_scalar(regs, instr->src0) + instr->imm;
        if (addr < vm->memory_size) {
            write_scalar(regs, instr->dst, vm->memory[addr]);
        }
        break;

    case SUB_OP_ST:
        addr = (uint32_t)read_scalar(regs, instr->dst) + instr->imm;
        if (addr < vm->memory_size) {
            vm->memory[addr] = read_scalar(regs, instr->src0);
        }
        break;

    case SUB_OP_LDR:
        addr = read_ref(regs, instr->src0) + instr->imm;
        if (addr < vm->memory_size) {
            write_scalar(regs, instr->dst, vm->memory[addr]);
        }
        break;

    case SUB_OP_STR:
        addr = read_ref(regs, instr->dst) + instr->imm;
        if (addr < vm->memory_size) {
            vm->memory[addr] = read_scalar(regs, instr->src0);
        }
        break;

    case SUB_OP_MOV:
        write_scalar(regs, instr->dst, read_scalar(regs, instr->src0));
        break;

    case SUB_OP_MOVR:
        /* Move between scalar and ref: dst <- H[src0] or H[dst] <- src0 */
        if (instr->imm == 0) {
            /* H[dst] = (uint32)src0 */
            write_ref(regs, instr->dst, (uint32_t)read_scalar(regs, instr->src0));
        } else {
            /* dst = (float)H[src0] */
            write_scalar(regs, instr->dst, (float)read_ref(regs, instr->src0));
        }
        break;

    case SUB_OP_MOVT:
        /* Move between scalar and token */
        if (instr->imm == 0) {
            /* T[dst] = (uint32)src0 */
            write_token(regs, instr->dst, (uint32_t)read_scalar(regs, instr->src0));
        } else {
            /* dst = (float)T[src0] */
            write_scalar(regs, instr->dst, (float)read_token(regs, instr->src0));
        }
        break;

    case SUB_OP_PUSH:
        if (vm->stack_ptr > 0) {
            vm->stack_ptr--;
            vm->memory[vm->stack_ptr] = read_scalar(regs, instr->src0);
        }
        break;

    case SUB_OP_POP:
        if (vm->stack_ptr < vm->memory_size) {
            write_scalar(regs, instr->dst, vm->memory[vm->stack_ptr]);
            vm->stack_ptr++;
        }
        break;

    /* === Arithmetic === */
    case SUB_OP_ADD:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, a + b);
        break;

    case SUB_OP_SUB:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, a - b);
        break;

    case SUB_OP_MUL:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, a * b);
        break;

    case SUB_OP_DIV:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, (b != 0.0f) ? (a / b) : 0.0f);
        break;

    case SUB_OP_MOD:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, (b != 0.0f) ? fmodf(a, b) : 0.0f);
        break;

    case SUB_OP_NEG:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, -a);
        break;

    case SUB_OP_ABS:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, fabsf(a));
        break;

    case SUB_OP_ADDI:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, a + (float)instr->imm);
        break;

    case SUB_OP_MULI:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, a * (float)instr->imm);
        break;

    case SUB_OP_INC:
        a = read_scalar(regs, instr->dst);
        write_scalar(regs, instr->dst, a + 1.0f);
        break;

    case SUB_OP_DEC:
        a = read_scalar(regs, instr->dst);
        write_scalar(regs, instr->dst, a - 1.0f);
        break;

    /* === Math Functions === */
    case SUB_OP_SQRT:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, sqrtf(fmaxf(0.0f, a)));
        break;

    case SUB_OP_EXP:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, expf(a));
        break;

    case SUB_OP_LOG:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, (a > 0.0f) ? logf(a) : -INFINITY);
        break;

    case SUB_OP_SIN:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, sinf(a));
        break;

    case SUB_OP_COS:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, cosf(a));
        break;

    case SUB_OP_TAN:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, tanf(a));
        break;

    case SUB_OP_ASIN:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, asinf(fmaxf(-1.0f, fminf(1.0f, a))));
        break;

    case SUB_OP_ACOS:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, acosf(fmaxf(-1.0f, fminf(1.0f, a))));
        break;

    case SUB_OP_ATAN:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, atanf(a));
        break;

    case SUB_OP_ATAN2:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, atan2f(a, b));
        break;

    case SUB_OP_POW:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, powf(a, b));
        break;

    case SUB_OP_FLOOR:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, floorf(a));
        break;

    case SUB_OP_CEIL:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, ceilf(a));
        break;

    case SUB_OP_ROUND:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, roundf(a));
        break;

    case SUB_OP_FMOD:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, (b != 0.0f) ? fmodf(a, b) : 0.0f);
        break;

    /* === Min/Max/Clamp === */
    case SUB_OP_MIN:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, fminf(a, b));
        break;

    case SUB_OP_MAX:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, fmaxf(a, b));
        break;

    case SUB_OP_CLAMP:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);  /* lo */
        c = vm->memory[instr->imm];          /* hi from memory */
        write_scalar(regs, instr->dst, fmaxf(b, fminf(c, a)));
        break;

    case SUB_OP_RELU:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, fmaxf(0.0f, a));
        break;

    case SUB_OP_SIGN:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f));
        break;

    case SUB_OP_LERP:
        a = read_scalar(regs, instr->src0);      /* start */
        b = read_scalar(regs, instr->src1);      /* end */
        c = read_scalar(regs, instr->dst);       /* t */
        write_scalar(regs, instr->dst, a + c * (b - a));
        break;

    /* === Comparison === */
    case SUB_OP_CMP:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        if (a < b) {
            write_token(regs, instr->dst, SUB_TOK_LT);
        } else if (a > b) {
            write_token(regs, instr->dst, SUB_TOK_GT);
        } else {
            write_token(regs, instr->dst, SUB_TOK_EQ);
        }
        break;

    case SUB_OP_CMPI:
        a = read_scalar(regs, instr->src0);
        b = (float)instr->imm;
        if (a < b) {
            write_token(regs, instr->dst, SUB_TOK_LT);
        } else if (a > b) {
            write_token(regs, instr->dst, SUB_TOK_GT);
        } else {
            write_token(regs, instr->dst, SUB_TOK_EQ);
        }
        break;

    case SUB_OP_TEQ:
        {
            uint32_t t0 = read_token(regs, instr->src0);
            uint32_t t1 = read_token(regs, instr->src1);
            write_token(regs, instr->dst, (t0 == t1) ? SUB_TOK_TRUE : SUB_TOK_FALSE);
        }
        break;

    case SUB_OP_TNE:
        {
            uint32_t t0 = read_token(regs, instr->src0);
            uint32_t t1 = read_token(regs, instr->src1);
            write_token(regs, instr->dst, (t0 != t1) ? SUB_TOK_TRUE : SUB_TOK_FALSE);
        }
        break;

    case SUB_OP_TSET:
        write_token(regs, instr->dst, (uint32_t)instr->imm);
        break;

    case SUB_OP_TGET:
        write_scalar(regs, instr->dst, (float)read_token(regs, instr->src0));
        break;

    case SUB_OP_ISNAN:
        a = read_scalar(regs, instr->src0);
        write_token(regs, instr->dst, isnan(a) ? SUB_TOK_TRUE : SUB_TOK_FALSE);
        break;

    case SUB_OP_ISINF:
        a = read_scalar(regs, instr->src0);
        write_token(regs, instr->dst, isinf(a) ? SUB_TOK_TRUE : SUB_TOK_FALSE);
        break;

    /* === Bitwise === */
    case SUB_OP_AND:
        conv.f = read_scalar(regs, instr->src0);
        {
            uint32_t u0 = conv.u;
            conv.f = read_scalar(regs, instr->src1);
            conv.u = u0 & conv.u;
            write_scalar(regs, instr->dst, conv.f);
        }
        break;

    case SUB_OP_OR:
        conv.f = read_scalar(regs, instr->src0);
        {
            uint32_t u0 = conv.u;
            conv.f = read_scalar(regs, instr->src1);
            conv.u = u0 | conv.u;
            write_scalar(regs, instr->dst, conv.f);
        }
        break;

    case SUB_OP_XOR:
        conv.f = read_scalar(regs, instr->src0);
        {
            uint32_t u0 = conv.u;
            conv.f = read_scalar(regs, instr->src1);
            conv.u = u0 ^ conv.u;
            write_scalar(regs, instr->dst, conv.f);
        }
        break;

    case SUB_OP_NOT:
        conv.f = read_scalar(regs, instr->src0);
        conv.u = ~conv.u;
        write_scalar(regs, instr->dst, conv.f);
        break;

    case SUB_OP_SHL:
        conv.f = read_scalar(regs, instr->src0);
        conv.u = conv.u << (instr->imm & 31);
        write_scalar(regs, instr->dst, conv.f);
        break;

    case SUB_OP_SHR:
        conv.f = read_scalar(regs, instr->src0);
        conv.u = conv.u >> (instr->imm & 31);
        write_scalar(regs, instr->dst, conv.f);
        break;

    case SUB_OP_SAR:
        conv.f = read_scalar(regs, instr->src0);
        {
            int32_t s = (int32_t)conv.u;
            s = s >> (instr->imm & 31);
            conv.u = (uint32_t)s;
        }
        write_scalar(regs, instr->dst, conv.f);
        break;

    case SUB_OP_FTOI:
        a = read_scalar(regs, instr->src0);
        conv.u = (uint32_t)(int32_t)a;
        write_scalar(regs, instr->dst, conv.f);
        break;

    case SUB_OP_ITOF:
        conv.f = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, (float)(int32_t)conv.u);
        break;

    /* === Reference Operations === */
    case SUB_OP_HREF:
        if (instr->src0 == 0) {
            write_ref(regs, instr->dst, (uint32_t)instr->imm);
        } else {
            write_ref(regs, instr->dst, (uint32_t)read_scalar(regs, instr->src0));
        }
        break;

    case SUB_OP_HGET:
        write_scalar(regs, instr->dst, (float)read_ref(regs, instr->src0));
        break;

    case SUB_OP_HSET:
        write_ref(regs, instr->dst, (uint32_t)read_scalar(regs, instr->src0));
        break;

    case SUB_OP_HADD:
        write_ref(regs, instr->dst, read_ref(regs, instr->dst) + instr->imm);
        break;

    case SUB_OP_HMOV:
        write_ref(regs, instr->dst, read_ref(regs, instr->src0));
        break;

    /* === Branching === */
    case SUB_OP_JMP:
        vm->pc += instr->imm * 4;  /* imm is in instructions, multiply by 4 for bytes */
        return;  /* Don't advance PC again */

    case SUB_OP_JZ:
        a = read_scalar(regs, instr->src0);
        if (a == 0.0f) {
            vm->pc += instr->imm * 4;
            return;
        }
        break;

    case SUB_OP_JNZ:
        a = read_scalar(regs, instr->src0);
        if (a != 0.0f) {
            vm->pc += instr->imm * 4;
            return;
        }
        break;

    case SUB_OP_JLT:
        a = read_scalar(regs, instr->src0);
        if (a < 0.0f) {
            vm->pc += instr->imm * 4;
            return;
        }
        break;

    case SUB_OP_JGT:
        a = read_scalar(regs, instr->src0);
        if (a > 0.0f) {
            vm->pc += instr->imm * 4;
            return;
        }
        break;

    case SUB_OP_JTOK:
        if (instr->has_ext) {
            uint32_t tok = read_token(regs, instr->src0);
            if (tok == (uint32_t)instr->imm) {
                vm->pc = instr->ext;
                return;
            }
        }
        break;

    case SUB_OP_CALL:
        if (vm->call_depth < EIS_CALL_STACK_SIZE && instr->has_ext) {
            vm->call_stack[vm->call_depth].return_addr = vm->pc + 8;  /* After this instruction */
            vm->call_depth++;
            vm->pc = instr->ext;
            return;
        } else {
            vm->state = SUB_STATE_ERROR;
            snprintf(vm->error_msg, sizeof(vm->error_msg), "Call stack overflow");
        }
        break;

    case SUB_OP_RET:
        if (vm->call_depth > 0) {
            vm->call_depth--;
            vm->pc = vm->call_stack[vm->call_depth].return_addr;
            return;
        } else {
            vm->state = SUB_STATE_HALTED;
        }
        break;

    case SUB_OP_JMPR:
        vm->pc = read_ref(regs, instr->src0);
        return;

    /* === I/O === */
    case SUB_OP_IN:
        {
            uint8_t ch = instr->imm & 0xF;
            if (vm->io[ch].read_fn) {
                write_scalar(regs, instr->dst, vm->io[ch].read_fn(vm->io[ch].user_ctx));
            } else {
                write_scalar(regs, instr->dst, vm->io[ch].value);
            }
        }
        break;

    case SUB_OP_OUT:
        {
            uint8_t ch = instr->imm & 0xF;
            a = read_scalar(regs, instr->src0);
            vm->io[ch].value = a;
            if (vm->io[ch].write_fn) {
                vm->io[ch].write_fn(vm->io[ch].user_ctx, a);
            }
        }
        break;

    case SUB_OP_POLL:
        {
            uint8_t ch = instr->imm & 0xF;
            write_token(regs, instr->dst, vm->io[ch].ready ? SUB_TOK_TRUE : SUB_TOK_FALSE);
        }
        break;

    case SUB_OP_INBLK:
        /* Block read: dst_addr, src_channel, count in imm */
        /* Simplified: not implemented yet */
        break;

    case SUB_OP_OUTBLK:
        /* Block write: similar */
        break;

    /* === System === */
    case SUB_OP_TRACE:
        if (vm->trace_enabled) {
            printf("TRACE: R%d = %f, imm=%d\n", instr->src0,
                   read_scalar(regs, instr->src0), instr->imm);
        }
        break;

    case SUB_OP_PRINT:
        printf("R%d = %f\n", instr->src0, read_scalar(regs, instr->src0));
        break;

    case SUB_OP_TIME:
        write_scalar(regs, instr->dst, (float)vm->tick);
        break;

    case SUB_OP_RAND:
        write_scalar(regs, instr->dst, random_float(vm));
        break;

    case SUB_OP_SEED:
        vm->random_state = (uint64_t)read_scalar(regs, instr->src0);
        if (vm->random_state == 0) vm->random_state = 1;
        break;

    default:
        snprintf(vm->error_msg, sizeof(vm->error_msg),
                 "Unknown opcode: 0x%02X at PC=0x%X", instr->opcode, vm->pc);
        vm->state = SUB_STATE_ERROR;
        break;
    }

    vm->instructions_executed++;
}

/* ==========================================================================
 * VM LIFECYCLE
 * ========================================================================== */

SubstrateVM* substrate_create(uint32_t memory_size) {
    SubstrateVM* vm = (SubstrateVM*)calloc(1, sizeof(SubstrateVM));
    if (!vm) return NULL;

    if (memory_size == 0) {
        memory_size = EIS_DEFAULT_MEMORY_SIZE;
    }

    vm->memory = (float*)calloc(memory_size, sizeof(float));
    if (!vm->memory) {
        free(vm);
        return NULL;
    }
    vm->memory_size = memory_size;

    /* Initialize stack pointer to top of memory */
    vm->stack_ptr = memory_size;

    /* Initialize random state */
    vm->random_state = 12345678901234567ULL;

    vm->state = SUB_STATE_HALTED;

    return vm;
}

void substrate_destroy(SubstrateVM* vm) {
    if (!vm) return;
    free(vm->memory);
    free(vm);
}

void substrate_reset(SubstrateVM* vm) {
    if (!vm) return;

    memset(&vm->regs, 0, sizeof(vm->regs));
    memset(vm->memory, 0, vm->memory_size * sizeof(float));
    memset(vm->call_stack, 0, sizeof(vm->call_stack));

    vm->pc = 0;
    vm->stack_ptr = vm->memory_size;
    vm->call_depth = 0;
    vm->state = SUB_STATE_RUNNING;
    vm->tick = 0;
    vm->instructions_executed = 0;
    vm->error_msg[0] = '\0';
}

bool substrate_load_program(SubstrateVM* vm, const uint8_t* program, uint32_t size) {
    if (!vm || !program || size == 0) return false;
    if (size > EIS_MAX_PROGRAM_SIZE) return false;

    vm->program = program;
    vm->program_size = size;
    vm->pc = 0;
    vm->state = SUB_STATE_RUNNING;

    return true;
}

/* ==========================================================================
 * EXECUTION
 * ========================================================================== */

SubstrateState substrate_step(SubstrateVM* vm) {
    if (!vm) return SUB_STATE_ERROR;
    if (vm->state != SUB_STATE_RUNNING) return vm->state;
    if (vm->pc >= vm->program_size) {
        vm->state = SUB_STATE_HALTED;
        return vm->state;
    }

    /* Decode */
    uint32_t consumed;
    SubstrateInstr instr = substrate_decode(vm->program, vm->pc, &consumed);

    /* Trace */
    if (vm->trace_fn) {
        vm->trace_fn(vm->trace_ctx, &instr, &vm->regs);
    }

    /* Execute */
    uint32_t old_pc = vm->pc;
    execute_instruction(vm, &instr);

    /* Advance PC if not a jump */
    if (vm->pc == old_pc) {
        vm->pc += consumed;
    }

    return vm->state;
}

SubstrateState substrate_run(SubstrateVM* vm, uint32_t max_steps) {
    uint32_t steps = 0;
    while (steps < max_steps && vm->state == SUB_STATE_RUNNING) {
        substrate_step(vm);
        steps++;
    }
    return vm->state;
}

SubstrateState substrate_resume(SubstrateVM* vm, uint32_t max_steps) {
    if (vm->state == SUB_STATE_YIELDED) {
        vm->state = SUB_STATE_RUNNING;
    }
    return substrate_run(vm, max_steps);
}

/* ==========================================================================
 * REGISTER ACCESS
 * ========================================================================== */

float substrate_get_scalar(const SubstrateVM* vm, uint8_t reg) {
    if (!vm || reg >= EIS_NUM_SCALAR_REGS) return 0.0f;
    return vm->regs.scalars[reg];
}

void substrate_set_scalar(SubstrateVM* vm, uint8_t reg, float value) {
    if (!vm || reg >= EIS_NUM_SCALAR_REGS) return;
    vm->regs.scalars[reg] = value;
}

uint32_t substrate_get_ref(const SubstrateVM* vm, uint8_t reg) {
    if (!vm || reg >= EIS_NUM_REF_REGS) return 0;
    return vm->regs.refs[reg];
}

void substrate_set_ref(SubstrateVM* vm, uint8_t reg, uint32_t value) {
    if (!vm || reg >= EIS_NUM_REF_REGS) return;
    vm->regs.refs[reg] = value;
}

uint32_t substrate_get_token(const SubstrateVM* vm, uint8_t reg) {
    if (!vm || reg >= EIS_NUM_TOKEN_REGS) return SUB_TOK_VOID;
    return vm->regs.tokens[reg];
}

void substrate_set_token(SubstrateVM* vm, uint8_t reg, uint32_t value) {
    if (!vm || reg >= EIS_NUM_TOKEN_REGS) return;
    vm->regs.tokens[reg] = value;
}

/* ==========================================================================
 * MEMORY ACCESS
 * ========================================================================== */

float substrate_mem_read(const SubstrateVM* vm, uint32_t addr) {
    if (!vm || addr >= vm->memory_size) return 0.0f;
    return vm->memory[addr];
}

void substrate_mem_write(SubstrateVM* vm, uint32_t addr, float value) {
    if (!vm || addr >= vm->memory_size) return;
    vm->memory[addr] = value;
}

void substrate_mem_read_block(const SubstrateVM* vm, uint32_t addr,
                               float* out, uint32_t count) {
    if (!vm || !out) return;
    for (uint32_t i = 0; i < count && (addr + i) < vm->memory_size; i++) {
        out[i] = vm->memory[addr + i];
    }
}

void substrate_mem_write_block(SubstrateVM* vm, uint32_t addr,
                                const float* data, uint32_t count) {
    if (!vm || !data) return;
    for (uint32_t i = 0; i < count && (addr + i) < vm->memory_size; i++) {
        vm->memory[addr + i] = data[i];
    }
}

/* ==========================================================================
 * I/O
 * ========================================================================== */

void substrate_io_configure(SubstrateVM* vm, uint8_t channel,
                            float (*read_fn)(void*),
                            void (*write_fn)(void*, float),
                            void* ctx) {
    if (!vm || channel >= EIS_NUM_IO_CHANNELS) return;
    vm->io[channel].read_fn = read_fn;
    vm->io[channel].write_fn = write_fn;
    vm->io[channel].user_ctx = ctx;
    vm->io[channel].enabled = true;
}

void substrate_io_inject(SubstrateVM* vm, uint8_t channel, float value) {
    if (!vm || channel >= EIS_NUM_IO_CHANNELS) return;
    vm->io[channel].value = value;
    vm->io[channel].ready = true;
}

float substrate_io_read(const SubstrateVM* vm, uint8_t channel) {
    if (!vm || channel >= EIS_NUM_IO_CHANNELS) return 0.0f;
    return vm->io[channel].value;
}

void substrate_io_set_ready(SubstrateVM* vm, uint8_t channel, bool ready) {
    if (!vm || channel >= EIS_NUM_IO_CHANNELS) return;
    vm->io[channel].ready = ready;
}

/* ==========================================================================
 * DEBUG/UTILITY
 * ========================================================================== */

const char* substrate_opcode_name(uint8_t opcode) {
    static const char* names[] = {
        [SUB_OP_NOP] = "NOP",
        [SUB_OP_HALT] = "HALT",
        [SUB_OP_YIELD] = "YIELD",
        [SUB_OP_TICK] = "TICK",
        [SUB_OP_DEBUG] = "DEBUG",
        [SUB_OP_LDI] = "LDI",
        [SUB_OP_LDI_F] = "LDI_F",
        [SUB_OP_LD] = "LD",
        [SUB_OP_ST] = "ST",
        [SUB_OP_MOV] = "MOV",
        [SUB_OP_ADD] = "ADD",
        [SUB_OP_SUB] = "SUB",
        [SUB_OP_MUL] = "MUL",
        [SUB_OP_DIV] = "DIV",
        [SUB_OP_NEG] = "NEG",
        [SUB_OP_ABS] = "ABS",
        [SUB_OP_SQRT] = "SQRT",
        [SUB_OP_SIN] = "SIN",
        [SUB_OP_COS] = "COS",
        [SUB_OP_MIN] = "MIN",
        [SUB_OP_MAX] = "MAX",
        [SUB_OP_RELU] = "RELU",
        [SUB_OP_CMP] = "CMP",
        [SUB_OP_TSET] = "TSET",
        [SUB_OP_JMP] = "JMP",
        [SUB_OP_JZ] = "JZ",
        [SUB_OP_JNZ] = "JNZ",
        [SUB_OP_CALL] = "CALL",
        [SUB_OP_RET] = "RET",
        [SUB_OP_IN] = "IN",
        [SUB_OP_OUT] = "OUT",
        [SUB_OP_TRACE] = "TRACE",
        [SUB_OP_RAND] = "RAND",
    };

    if (opcode < sizeof(names)/sizeof(names[0]) && names[opcode]) {
        return names[opcode];
    }
    return "???";
}

const char* substrate_token_name(uint32_t token) {
    switch (token) {
        case SUB_TOK_VOID: return "VOID";
        case SUB_TOK_FALSE: return "FALSE";
        case SUB_TOK_TRUE: return "TRUE";
        case SUB_TOK_LT: return "LT";
        case SUB_TOK_EQ: return "EQ";
        case SUB_TOK_GT: return "GT";
        case SUB_TOK_ERR: return "ERR";
        default: return "???";
    }
}

const char* substrate_state_name(SubstrateState state) {
    switch (state) {
        case SUB_STATE_RUNNING: return "RUNNING";
        case SUB_STATE_HALTED: return "HALTED";
        case SUB_STATE_YIELDED: return "YIELDED";
        case SUB_STATE_ERROR: return "ERROR";
        case SUB_STATE_WAITING: return "WAITING";
        case SUB_STATE_BREAKPOINT: return "BREAKPOINT";
        default: return "???";
    }
}

void substrate_set_trace(SubstrateVM* vm,
                          void (*fn)(void*, const SubstrateInstr*, const SubstrateRegs*),
                          void* ctx) {
    if (!vm) return;
    vm->trace_fn = fn;
    vm->trace_ctx = ctx;
    vm->trace_enabled = (fn != NULL);
}

void substrate_get_stats(const SubstrateVM* vm,
                          uint64_t* tick, uint64_t* instructions) {
    if (!vm) return;
    if (tick) *tick = vm->tick;
    if (instructions) *instructions = vm->instructions_executed;
}

void substrate_disassemble(const uint8_t* program, uint32_t size,
                            char* out, uint32_t out_size) {
    if (!program || !out || out_size == 0) return;

    uint32_t offset = 0;
    uint32_t written = 0;

    while (offset < size && written < out_size - 100) {
        uint32_t consumed;
        SubstrateInstr instr = substrate_decode(program, offset, &consumed);

        int n = snprintf(out + written, out_size - written,
                         "%04X: %s R%d, R%d, R%d, %d\n",
                         offset,
                         substrate_opcode_name(instr.opcode),
                         instr.dst, instr.src0, instr.src1, instr.imm);

        if (n > 0) written += n;
        offset += consumed;
    }
}
