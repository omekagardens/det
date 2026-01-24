/**
 * EIS Substrate v2 - Implementation
 * ==================================
 *
 * DET-aware execution layer with phase-based execution,
 * proposal buffers, and effect application.
 */

#include "../include/eis_substrate_v2.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ==========================================================================
 * INSTRUCTION DECODE/ENCODE
 * ========================================================================== */

SubstrateInstr substrate_decode(const uint8_t* data, uint32_t offset, uint32_t* consumed) {
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
        case OP_LDI_F:
        case OP_PROP_EFFECT:
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
                    (imm_bits);

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
 * REGISTER ACCESS (Internal)
 * ========================================================================== */

static inline float read_scalar(const SubstrateRegs* regs, uint8_t reg) {
    return (reg < SUB_NUM_SCALAR_REGS) ? regs->scalars[reg] : 0.0f;
}

static inline void write_scalar(SubstrateRegs* regs, uint8_t reg, float value) {
    if (reg < SUB_NUM_SCALAR_REGS) {
        regs->scalars[reg] = value;
    }
}

static inline uint32_t read_ref(const SubstrateRegs* regs, uint8_t reg) {
    return (reg < SUB_NUM_REF_REGS) ? regs->refs[reg] : 0;
}

static inline void write_ref(SubstrateRegs* regs, uint8_t reg, uint32_t value) {
    if (reg < SUB_NUM_REF_REGS) {
        regs->refs[reg] = value;
    }
}

static inline uint32_t read_token(const SubstrateRegs* regs, uint8_t reg) {
    return (reg < SUB_NUM_TOKEN_REGS) ? regs->tokens[reg] : TOK_VOID;
}

static inline void write_token(SubstrateRegs* regs, uint8_t reg, uint32_t value) {
    if (reg < SUB_NUM_TOKEN_REGS) {
        regs->tokens[reg] = value;
    }
}

/* ==========================================================================
 * NODE/BOND FIELD ACCESS
 * ========================================================================== */

float substrate_node_get_f(const SubstrateVM* vm, uint32_t node_id, NodeFieldId field) {
    if (!vm->nodes || node_id >= vm->nodes->num_nodes) return 0.0f;

    switch (field) {
        case NODE_FIELD_F: return vm->nodes->F[node_id];
        case NODE_FIELD_Q: return vm->nodes->q[node_id];
        case NODE_FIELD_A: return vm->nodes->a[node_id];
        case NODE_FIELD_SIGMA: return vm->nodes->sigma[node_id];
        case NODE_FIELD_P: return vm->nodes->P[node_id];
        case NODE_FIELD_TAU: return vm->nodes->tau[node_id];
        case NODE_FIELD_COS_THETA: return vm->nodes->cos_theta[node_id];
        case NODE_FIELD_SIN_THETA: return vm->nodes->sin_theta[node_id];
        default: return 0.0f;
    }
}

void substrate_node_set_f(SubstrateVM* vm, uint32_t node_id, NodeFieldId field, float value) {
    if (!vm->nodes || node_id >= vm->nodes->num_nodes) return;

    switch (field) {
        case NODE_FIELD_F: vm->nodes->F[node_id] = value; break;
        case NODE_FIELD_Q: vm->nodes->q[node_id] = value; break;
        case NODE_FIELD_A: vm->nodes->a[node_id] = fmaxf(0.0f, fminf(1.0f, value)); break;
        case NODE_FIELD_SIGMA: vm->nodes->sigma[node_id] = value; break;
        case NODE_FIELD_P: vm->nodes->P[node_id] = value; break;
        case NODE_FIELD_TAU: vm->nodes->tau[node_id] = value; break;
        case NODE_FIELD_COS_THETA: vm->nodes->cos_theta[node_id] = value; break;
        case NODE_FIELD_SIN_THETA: vm->nodes->sin_theta[node_id] = value; break;
        default: break;
    }
}

uint32_t substrate_node_get_i(const SubstrateVM* vm, uint32_t node_id, NodeFieldId field) {
    if (!vm->nodes || node_id >= vm->nodes->num_nodes) return 0;

    switch (field) {
        case NODE_FIELD_K: return vm->nodes->k[node_id];
        case NODE_FIELD_R: return vm->nodes->r[node_id];
        case NODE_FIELD_FLAGS: return vm->nodes->flags[node_id];
        default: return 0;
    }
}

void substrate_node_set_i(SubstrateVM* vm, uint32_t node_id, NodeFieldId field, uint32_t value) {
    if (!vm->nodes || node_id >= vm->nodes->num_nodes) return;

    switch (field) {
        case NODE_FIELD_K: vm->nodes->k[node_id] = value; break;
        case NODE_FIELD_R: vm->nodes->r[node_id] = value; break;
        case NODE_FIELD_FLAGS: vm->nodes->flags[node_id] = value; break;
        default: break;
    }
}

float substrate_bond_get_f(const SubstrateVM* vm, uint32_t bond_id, BondFieldId field) {
    if (!vm->bonds || bond_id >= vm->bonds->num_bonds) return 0.0f;

    switch (field) {
        case BOND_FIELD_C: return vm->bonds->C[bond_id];
        case BOND_FIELD_PI: return vm->bonds->pi[bond_id];
        case BOND_FIELD_SIGMA: return vm->bonds->sigma[bond_id];
        default: return 0.0f;
    }
}

void substrate_bond_set_f(SubstrateVM* vm, uint32_t bond_id, BondFieldId field, float value) {
    if (!vm->bonds || bond_id >= vm->bonds->num_bonds) return;

    switch (field) {
        case BOND_FIELD_C: vm->bonds->C[bond_id] = fmaxf(0.0f, fminf(1.0f, value)); break;
        case BOND_FIELD_PI: vm->bonds->pi[bond_id] = value; break;
        case BOND_FIELD_SIGMA: vm->bonds->sigma[bond_id] = value; break;
        default: break;
    }
}

uint32_t substrate_bond_get_i(const SubstrateVM* vm, uint32_t bond_id, BondFieldId field) {
    if (!vm->bonds || bond_id >= vm->bonds->num_bonds) return 0;

    switch (field) {
        case BOND_FIELD_NODE_I: return vm->bonds->node_i[bond_id];
        case BOND_FIELD_NODE_J: return vm->bonds->node_j[bond_id];
        case BOND_FIELD_FLAGS: return vm->bonds->flags[bond_id];
        default: return 0;
    }
}

void substrate_bond_set_i(SubstrateVM* vm, uint32_t bond_id, BondFieldId field, uint32_t value) {
    if (!vm->bonds || bond_id >= vm->bonds->num_bonds) return;

    switch (field) {
        case BOND_FIELD_NODE_I: vm->bonds->node_i[bond_id] = value; break;
        case BOND_FIELD_NODE_J: vm->bonds->node_j[bond_id] = value; break;
        case BOND_FIELD_FLAGS: vm->bonds->flags[bond_id] = value; break;
        default: break;
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
    uint64_t r = xorshift64(&vm->seed);
    return (float)(r & 0xFFFFFFFF) / 4294967296.0f;
}

/* ==========================================================================
 * PROPOSAL OPERATIONS
 * ========================================================================== */

uint32_t substrate_prop_new(SubstrateVM* vm) {
    if (vm->prop_buf.count >= SUB_MAX_PROPOSALS) {
        return 0xFFFFFFFF;  /* Buffer full */
    }

    uint32_t idx = vm->prop_buf.count++;
    Proposal* prop = &vm->prop_buf.proposals[idx];
    memset(prop, 0, sizeof(Proposal));
    prop->valid = true;
    prop->effect_id = EFFECT_NONE;
    prop->score = 0.0f;

    return idx;
}

void substrate_prop_score(SubstrateVM* vm, uint32_t prop_idx, float score) {
    if (prop_idx < vm->prop_buf.count) {
        vm->prop_buf.proposals[prop_idx].score = score;
    }
}

void substrate_prop_effect(SubstrateVM* vm, uint32_t prop_idx, EffectId effect_id,
                           const uint32_t* args, uint32_t arg_count) {
    if (prop_idx >= vm->prop_buf.count) return;

    Proposal* prop = &vm->prop_buf.proposals[prop_idx];
    prop->effect_id = effect_id;
    prop->arg_count = (arg_count > SUB_MAX_EFFECT_ARGS) ? SUB_MAX_EFFECT_ARGS : arg_count;

    for (uint32_t i = 0; i < prop->arg_count; i++) {
        prop->args[i] = args[i];
    }
}

uint32_t substrate_choose(SubstrateVM* vm, float decisiveness) {
    if (vm->prop_buf.count == 0) {
        vm->prop_buf.chosen = 0xFFFFFFFF;
        return 0xFFFFFFFF;
    }

    /* Find highest score and calculate total */
    float total_score = 0.0f;
    float max_score = -1.0f;
    uint32_t max_idx = 0;

    for (uint32_t i = 0; i < vm->prop_buf.count; i++) {
        if (vm->prop_buf.proposals[i].valid) {
            float score = vm->prop_buf.proposals[i].score;
            total_score += score;
            if (score > max_score) {
                max_score = score;
                max_idx = i;
            }
        }
    }

    if (total_score <= 0.0f) {
        /* No valid proposals, choose first */
        vm->prop_buf.chosen = 0;
        return 0;
    }

    /* High decisiveness: pick highest score deterministically */
    if (decisiveness >= 0.99f) {
        vm->prop_buf.chosen = max_idx;
        return max_idx;
    }

    /* Blend between random selection and deterministic max */
    float r = random_float(vm);

    /* If random roll beats decisiveness threshold, use weighted random */
    if (r > decisiveness) {
        float cumulative = 0.0f;
        float target = random_float(vm);
        for (uint32_t i = 0; i < vm->prop_buf.count; i++) {
            if (vm->prop_buf.proposals[i].valid) {
                cumulative += vm->prop_buf.proposals[i].score / total_score;
                if (target <= cumulative) {
                    vm->prop_buf.chosen = i;
                    return i;
                }
            }
        }
    }

    /* Otherwise use max */
    vm->prop_buf.chosen = max_idx;
    return max_idx;
}

uint32_t substrate_commit(SubstrateVM* vm) {
    if (vm->prop_buf.chosen >= vm->prop_buf.count) {
        return TOK_FAIL;
    }

    Proposal* chosen = &vm->prop_buf.proposals[vm->prop_buf.chosen];
    if (!chosen->valid) {
        return TOK_FAIL;
    }

    /* Apply the effect */
    uint32_t result = effect_apply(vm, chosen->effect_id, chosen->args);

    /* Clear proposal buffer for next tick */
    vm->prop_buf.count = 0;
    vm->prop_buf.chosen = 0xFFFFFFFF;

    return result;
}

/* ==========================================================================
 * EFFECT APPLICATION
 * ========================================================================== */

uint32_t effect_apply(SubstrateVM* vm, EffectId effect_id, const uint32_t* args) {
    switch (effect_id) {
        case EFFECT_NONE:
            return TOK_OK;

        case EFFECT_XFER_F: {
            /* Antisymmetric transfer: F[src] -= amount, F[dst] += amount */
            uint32_t src_node = args[0];
            uint32_t dst_node = args[1];
            float amount = effect_unpack_float(args[2]);

            if (!vm->nodes) return TOK_FAIL;
            if (src_node >= vm->nodes->num_nodes) return TOK_FAIL;
            if (dst_node >= vm->nodes->num_nodes) return TOK_FAIL;

            float available = vm->nodes->F[src_node];
            float actual = fminf(amount, available);
            if (actual < 0.0f) actual = 0.0f;

            vm->nodes->F[src_node] -= actual;
            vm->nodes->F[dst_node] += actual;

            return (actual > 0.0f) ? TOK_XFER_OK : TOK_XFER_PARTIAL;
        }

        case EFFECT_DIFFUSE: {
            /* Symmetric flux on bond: F[i] -= delta, F[j] += delta */
            uint32_t bond_id = args[0];
            float delta = effect_unpack_float(args[1]);

            if (!vm->bonds || !vm->nodes) return TOK_FAIL;
            if (bond_id >= vm->bonds->num_bonds) return TOK_FAIL;

            uint32_t i = vm->bonds->node_i[bond_id];
            uint32_t j = vm->bonds->node_j[bond_id];

            if (i >= vm->nodes->num_nodes) return TOK_FAIL;
            if (j >= vm->nodes->num_nodes) return TOK_FAIL;

            vm->nodes->F[i] -= delta;
            vm->nodes->F[j] += delta;

            return TOK_DIFFUSE_OK;
        }

        case EFFECT_SET_F: {
            uint32_t node_id = args[0];
            float value = effect_unpack_float(args[1]);
            substrate_node_set_f(vm, node_id, NODE_FIELD_F, value);
            return TOK_OK;
        }

        case EFFECT_ADD_F: {
            uint32_t node_id = args[0];
            float delta = effect_unpack_float(args[1]);
            float current = substrate_node_get_f(vm, node_id, NODE_FIELD_F);
            substrate_node_set_f(vm, node_id, NODE_FIELD_F, current + delta);
            return TOK_OK;
        }

        case EFFECT_SET_C: {
            uint32_t bond_id = args[0];
            float value = effect_unpack_float(args[1]);
            substrate_bond_set_f(vm, bond_id, BOND_FIELD_C, value);
            return TOK_OK;
        }

        case EFFECT_ADD_C: {
            uint32_t bond_id = args[0];
            float delta = effect_unpack_float(args[1]);
            float current = substrate_bond_get_f(vm, bond_id, BOND_FIELD_C);
            substrate_bond_set_f(vm, bond_id, BOND_FIELD_C, current + delta);
            return TOK_OK;
        }

        case EFFECT_SET_PI: {
            uint32_t bond_id = args[0];
            float value = effect_unpack_float(args[1]);
            substrate_bond_set_f(vm, bond_id, BOND_FIELD_PI, value);
            return TOK_OK;
        }

        case EFFECT_INC_K: {
            uint32_t node_id = args[0];
            uint32_t k = substrate_node_get_i(vm, node_id, NODE_FIELD_K);
            substrate_node_set_i(vm, node_id, NODE_FIELD_K, k + 1);
            return TOK_OK;
        }

        case EFFECT_EMIT_BYTE: {
            uint32_t buf_id = args[0];
            uint8_t byte = args[1] & 0xFF;
            substrate_boundary_emit_byte(vm, buf_id, byte);
            return TOK_OK;
        }

        default:
            return TOK_FAIL;
    }
}

/* ==========================================================================
 * INSTRUCTION EXECUTION
 * ========================================================================== */

static void execute_instruction(SubstrateVM* vm, const SubstrateInstr* instr) {
    SubstrateRegs* regs = &vm->regs;
    float a, b, c;
    union { uint32_t u; float f; } conv;

    switch (instr->opcode) {

    /* ===== Phase Control ===== */
    case OP_NOP:
        break;

    case OP_HALT:
        vm->state = SUB_STATE_HALTED;
        break;

    case OP_YIELD:
        vm->state = SUB_STATE_YIELDED;
        break;

    case OP_TICK:
        vm->tick++;
        /* Clear proposal buffer for new tick */
        vm->prop_buf.count = 0;
        vm->prop_buf.chosen = 0xFFFFFFFF;
        break;

    case OP_PHASE_R:
        vm->phase = PHASE_READ;
        break;

    case OP_PHASE_P:
        vm->phase = PHASE_PROPOSE;
        break;

    case OP_PHASE_C:
        vm->phase = PHASE_CHOOSE;
        break;

    case OP_PHASE_X:
        vm->phase = PHASE_COMMIT;
        break;

    /* ===== Typed Loads ===== */
    case OP_LDN: {
        /* Load node field: dst = nodes[H[src0]].field[src1] */
        uint32_t ref = read_ref(regs, instr->src0);
        uint32_t node_id = REF_ID(ref);
        NodeFieldId field = (NodeFieldId)instr->src1;
        float value = substrate_node_get_f(vm, node_id, field);
        write_scalar(regs, instr->dst, value);
        break;
    }

    case OP_LDB: {
        /* Load bond field: dst = bonds[H[src0]].field[src1] */
        uint32_t ref = read_ref(regs, instr->src0);
        uint32_t bond_id = REF_ID(ref);
        BondFieldId field = (BondFieldId)instr->src1;
        float value = substrate_bond_get_f(vm, bond_id, field);
        write_scalar(regs, instr->dst, value);
        break;
    }

    case OP_LDI:
        write_scalar(regs, instr->dst, (float)instr->imm);
        break;

    case OP_LDI_F:
        if (instr->has_ext) {
            conv.u = instr->ext;
            write_scalar(regs, instr->dst, conv.f);
        }
        break;

    /* ===== Register Ops ===== */
    case OP_MOV:
        write_scalar(regs, instr->dst, read_scalar(regs, instr->src0));
        break;

    case OP_MOVR:
        if (instr->imm == 0) {
            /* H[dst] = (uint32)R[src0] */
            write_ref(regs, instr->dst, (uint32_t)read_scalar(regs, instr->src0));
        } else {
            /* R[dst] = (float)H[src0] */
            write_scalar(regs, instr->dst, (float)read_ref(regs, instr->src0));
        }
        break;

    case OP_MOVT:
        if (instr->imm == 0) {
            /* T[dst] = (uint32)R[src0] */
            write_token(regs, instr->dst, (uint32_t)read_scalar(regs, instr->src0));
        } else {
            /* R[dst] = (float)T[src0] */
            write_scalar(regs, instr->dst, (float)read_token(regs, instr->src0));
        }
        break;

    case OP_TSET:
        write_token(regs, instr->dst, (uint32_t)instr->imm);
        break;

    case OP_TGET:
        write_scalar(regs, instr->dst, (float)read_token(regs, instr->src0));
        break;

    /* ===== Arithmetic ===== */
    case OP_ADD:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, a + b);
        break;

    case OP_SUB:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, a - b);
        break;

    case OP_MUL:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, a * b);
        break;

    case OP_DIV:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, (b != 0.0f) ? (a / b) : 0.0f);
        break;

    case OP_MAD:
        /* dst = src0 * src1 + dst */
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        c = read_scalar(regs, instr->dst);
        write_scalar(regs, instr->dst, a * b + c);
        break;

    case OP_NEG:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, -a);
        break;

    case OP_ABS:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, fabsf(a));
        break;

    case OP_SQRT:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, sqrtf(fmaxf(0.0f, a)));
        break;

    case OP_MIN:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, fminf(a, b));
        break;

    case OP_MAX:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        write_scalar(regs, instr->dst, fmaxf(a, b));
        break;

    case OP_RELU:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, fmaxf(0.0f, a));
        break;

    case OP_CLAMP:
        a = read_scalar(regs, instr->src0);
        write_scalar(regs, instr->dst, fmaxf(0.0f, fminf(1.0f, a)));
        break;

    /* ===== Comparison ===== */
    case OP_CMP:
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        if (a < b) {
            write_token(regs, instr->dst, TOK_LT);
        } else if (a > b) {
            write_token(regs, instr->dst, TOK_GT);
        } else {
            write_token(regs, instr->dst, TOK_EQ);
        }
        break;

    case OP_CMPE: {
        /* Compare with epsilon (from immediate or another register) */
        a = read_scalar(regs, instr->src0);
        b = read_scalar(regs, instr->src1);
        float eps = (float)instr->imm * 0.001f;  /* Scale immediate */
        if (fabsf(a - b) < eps) {
            write_token(regs, instr->dst, TOK_EQ);
        } else if (a < b) {
            write_token(regs, instr->dst, TOK_LT);
        } else {
            write_token(regs, instr->dst, TOK_GT);
        }
        break;
    }

    case OP_TEQ: {
        uint32_t t0 = read_token(regs, instr->src0);
        uint32_t t1 = read_token(regs, instr->src1);
        write_token(regs, instr->dst, (t0 == t1) ? TOK_TRUE : TOK_FALSE);
        break;
    }

    case OP_TNE: {
        uint32_t t0 = read_token(regs, instr->src0);
        uint32_t t1 = read_token(regs, instr->src1);
        write_token(regs, instr->dst, (t0 != t1) ? TOK_TRUE : TOK_FALSE);
        break;
    }

    /* ===== Proposals ===== */
    case OP_PROP_NEW: {
        uint32_t prop_idx = substrate_prop_new(vm);
        write_ref(regs, instr->dst, REF_MAKE(REF_TYPE_PROP, prop_idx));
        break;
    }

    case OP_PROP_SCORE: {
        uint32_t ref = read_ref(regs, instr->dst);
        uint32_t prop_idx = REF_ID(ref);
        float score = read_scalar(regs, instr->src0);
        substrate_prop_score(vm, prop_idx, score);
        break;
    }

    case OP_PROP_EFFECT: {
        uint32_t ref = read_ref(regs, instr->dst);
        uint32_t prop_idx = REF_ID(ref);
        EffectId effect_id = (EffectId)instr->src0;
        /* Args come from extension word and following instructions */
        if (instr->has_ext) {
            uint32_t args[1] = { instr->ext };
            substrate_prop_effect(vm, prop_idx, effect_id, args, 1);
        }
        break;
    }

    /* ===== Choose/Commit ===== */
    case OP_CHOOSE: {
        float decisiveness = read_scalar(regs, instr->src0);
        uint32_t choice = substrate_choose(vm, decisiveness);
        write_ref(regs, instr->dst, REF_MAKE(REF_TYPE_CHOICE, choice));
        break;
    }

    case OP_COMMIT: {
        uint32_t result = substrate_commit(vm);
        write_token(regs, instr->dst, result);
        break;
    }

    case OP_WITNESS:
        write_token(regs, instr->dst, (uint32_t)instr->imm);
        break;

    /* ===== Stores ===== */
    case OP_STN: {
        uint32_t ref = read_ref(regs, instr->dst);
        uint32_t node_id = REF_ID(ref);
        NodeFieldId field = (NodeFieldId)instr->src0;
        float value = read_scalar(regs, instr->src1);
        substrate_node_set_f(vm, node_id, field, value);
        break;
    }

    case OP_STB: {
        uint32_t ref = read_ref(regs, instr->dst);
        uint32_t bond_id = REF_ID(ref);
        BondFieldId field = (BondFieldId)instr->src0;
        float value = read_scalar(regs, instr->src1);
        substrate_bond_set_f(vm, bond_id, field, value);
        break;
    }

    /* ===== I/O ===== */
    case OP_IN: {
        uint8_t ch = instr->imm & 0xF;
        if (vm->io[ch].read_fn) {
            write_scalar(regs, instr->dst, vm->io[ch].read_fn(vm->io[ch].user_ctx));
        } else {
            write_scalar(regs, instr->dst, vm->io[ch].value);
        }
        break;
    }

    case OP_OUT: {
        uint8_t ch = instr->imm & 0xF;
        a = read_scalar(regs, instr->src0);
        vm->io[ch].value = a;
        if (vm->io[ch].write_fn) {
            vm->io[ch].write_fn(vm->io[ch].user_ctx, a);
        }
        break;
    }

    case OP_EMIT: {
        uint32_t ref = read_ref(regs, instr->dst);
        uint32_t buf_id = REF_ID(ref);
        uint8_t byte = (uint8_t)read_scalar(regs, instr->src0);
        substrate_boundary_emit_byte(vm, buf_id, byte);
        break;
    }

    /* ===== System ===== */
    case OP_RAND:
        write_scalar(regs, instr->dst, random_float(vm));
        break;

    case OP_SEED:
        vm->seed = (uint64_t)read_scalar(regs, instr->src0);
        if (vm->seed == 0) vm->seed = 1;
        break;

    case OP_LANE:
        write_scalar(regs, instr->dst, (float)vm->lane_id);
        break;

    case OP_TIME:
        write_scalar(regs, instr->dst, (float)vm->tick);
        break;

    case OP_DEBUG:
        vm->state = SUB_STATE_BREAKPOINT;
        break;

    default:
        snprintf(vm->error_msg, sizeof(vm->error_msg),
                 "Unknown opcode: 0x%02X at PC=0x%X", instr->opcode, vm->pc);
        vm->state = SUB_STATE_ERROR;
        break;
    }

    vm->instructions++;
}

/* ==========================================================================
 * VM LIFECYCLE
 * ========================================================================== */

SubstrateVM* substrate_create(void) {
    SubstrateVM* vm = (SubstrateVM*)calloc(1, sizeof(SubstrateVM));
    if (!vm) return NULL;

    vm->seed = 12345678901234567ULL;
    vm->state = SUB_STATE_HALTED;
    vm->phase = PHASE_READ;
    vm->prop_buf.chosen = 0xFFFFFFFF;

    return vm;
}

void substrate_destroy(SubstrateVM* vm) {
    if (!vm) return;

    /* Free node arrays */
    if (vm->nodes) {
        free(vm->nodes->F);
        free(vm->nodes->q);
        free(vm->nodes->a);
        free(vm->nodes->sigma);
        free(vm->nodes->P);
        free(vm->nodes->tau);
        free(vm->nodes->cos_theta);
        free(vm->nodes->sin_theta);
        free(vm->nodes->k);
        free(vm->nodes->r);
        free(vm->nodes->flags);
        free(vm->nodes);
    }

    /* Free bond arrays */
    if (vm->bonds) {
        free(vm->bonds->node_i);
        free(vm->bonds->node_j);
        free(vm->bonds->C);
        free(vm->bonds->pi);
        free(vm->bonds->sigma);
        free(vm->bonds->flags);
        free(vm->bonds);
    }

    /* Free scratch */
    free(vm->scratch);

    /* Free boundaries */
    if (vm->boundaries) {
        for (uint32_t i = 0; i < vm->num_boundaries; i++) {
            free(vm->boundaries[i].data);
        }
        free(vm->boundaries);
    }

    free(vm);
}

void substrate_reset(SubstrateVM* vm) {
    if (!vm) return;

    memset(&vm->regs, 0, sizeof(vm->regs));
    vm->pc = 0;
    vm->phase = PHASE_READ;
    vm->state = SUB_STATE_RUNNING;
    vm->tick = 0;
    vm->instructions = 0;
    vm->error_msg[0] = '\0';
    vm->prop_buf.count = 0;
    vm->prop_buf.chosen = 0xFFFFFFFF;
}

bool substrate_alloc_nodes(SubstrateVM* vm, uint32_t num_nodes) {
    if (vm->nodes) {
        /* Already allocated */
        return false;
    }

    vm->nodes = (NodeArrays*)calloc(1, sizeof(NodeArrays));
    if (!vm->nodes) return false;

    vm->nodes->F = (float*)calloc(num_nodes, sizeof(float));
    vm->nodes->q = (float*)calloc(num_nodes, sizeof(float));
    vm->nodes->a = (float*)calloc(num_nodes, sizeof(float));
    vm->nodes->sigma = (float*)calloc(num_nodes, sizeof(float));
    vm->nodes->P = (float*)calloc(num_nodes, sizeof(float));
    vm->nodes->tau = (float*)calloc(num_nodes, sizeof(float));
    vm->nodes->cos_theta = (float*)calloc(num_nodes, sizeof(float));
    vm->nodes->sin_theta = (float*)calloc(num_nodes, sizeof(float));
    vm->nodes->k = (uint32_t*)calloc(num_nodes, sizeof(uint32_t));
    vm->nodes->r = (uint32_t*)calloc(num_nodes, sizeof(uint32_t));
    vm->nodes->flags = (uint32_t*)calloc(num_nodes, sizeof(uint32_t));

    /* Initialize default values */
    for (uint32_t i = 0; i < num_nodes; i++) {
        vm->nodes->a[i] = 1.0f;         /* Full agency by default */
        vm->nodes->sigma[i] = 1.0f;     /* Default processing rate */
        vm->nodes->cos_theta[i] = 1.0f; /* Phase = 0 */
        vm->nodes->sin_theta[i] = 0.0f;
    }

    vm->nodes->num_nodes = num_nodes;
    vm->nodes->capacity = num_nodes;

    return true;
}

bool substrate_alloc_bonds(SubstrateVM* vm, uint32_t num_bonds) {
    if (vm->bonds) return false;

    vm->bonds = (BondArrays*)calloc(1, sizeof(BondArrays));
    if (!vm->bonds) return false;

    vm->bonds->node_i = (uint32_t*)calloc(num_bonds, sizeof(uint32_t));
    vm->bonds->node_j = (uint32_t*)calloc(num_bonds, sizeof(uint32_t));
    vm->bonds->C = (float*)calloc(num_bonds, sizeof(float));
    vm->bonds->pi = (float*)calloc(num_bonds, sizeof(float));
    vm->bonds->sigma = (float*)calloc(num_bonds, sizeof(float));
    vm->bonds->flags = (uint32_t*)calloc(num_bonds, sizeof(uint32_t));

    /* Initialize default values */
    for (uint32_t i = 0; i < num_bonds; i++) {
        vm->bonds->C[i] = 0.5f;         /* Default coherence */
        vm->bonds->sigma[i] = 1.0f;     /* Default conductivity */
    }

    vm->bonds->num_bonds = num_bonds;
    vm->bonds->capacity = num_bonds;

    return true;
}

bool substrate_alloc_scratch(SubstrateVM* vm, uint32_t size) {
    vm->scratch = (float*)calloc(size, sizeof(float));
    vm->scratch_size = size;
    return vm->scratch != NULL;
}

uint32_t substrate_add_boundary(SubstrateVM* vm, uint32_t capacity, bool readonly) {
    uint32_t idx = vm->num_boundaries;

    vm->boundaries = (BoundaryBuffer*)realloc(
        vm->boundaries,
        (vm->num_boundaries + 1) * sizeof(BoundaryBuffer)
    );

    BoundaryBuffer* buf = &vm->boundaries[idx];
    buf->data = (uint8_t*)calloc(capacity, 1);
    buf->size = 0;
    buf->capacity = capacity;
    buf->readonly = readonly;

    vm->num_boundaries++;

    return idx;
}

bool substrate_load_program(SubstrateVM* vm, const uint8_t* program, uint32_t size) {
    if (!vm || !program || size == 0) return false;

    vm->program = program;
    vm->program_size = size;
    vm->pc = 0;
    vm->state = SUB_STATE_RUNNING;

    return true;
}

/* ==========================================================================
 * EXECUTION
 * ========================================================================== */

void substrate_set_lane(SubstrateVM* vm, uint32_t lane_id) {
    vm->lane_id = lane_id;
}

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
    execute_instruction(vm, &instr);

    /* Advance PC */
    vm->pc += consumed;

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

void substrate_set_phase(SubstrateVM* vm, SubstratePhase phase) {
    vm->phase = phase;
}

SubstratePhase substrate_get_phase(const SubstrateVM* vm) {
    return vm->phase;
}

/* ==========================================================================
 * REGISTER ACCESS (Public)
 * ========================================================================== */

float substrate_get_scalar(const SubstrateVM* vm, uint8_t reg) {
    if (!vm || reg >= SUB_NUM_SCALAR_REGS) return 0.0f;
    return vm->regs.scalars[reg];
}

void substrate_set_scalar(SubstrateVM* vm, uint8_t reg, float value) {
    if (!vm || reg >= SUB_NUM_SCALAR_REGS) return;
    vm->regs.scalars[reg] = value;
}

uint32_t substrate_get_ref(const SubstrateVM* vm, uint8_t reg) {
    if (!vm || reg >= SUB_NUM_REF_REGS) return 0;
    return vm->regs.refs[reg];
}

void substrate_set_ref(SubstrateVM* vm, uint8_t reg, uint32_t value) {
    if (!vm || reg >= SUB_NUM_REF_REGS) return;
    vm->regs.refs[reg] = value;
}

uint32_t substrate_get_token(const SubstrateVM* vm, uint8_t reg) {
    if (!vm || reg >= SUB_NUM_TOKEN_REGS) return TOK_VOID;
    return vm->regs.tokens[reg];
}

void substrate_set_token(SubstrateVM* vm, uint8_t reg, uint32_t value) {
    if (!vm || reg >= SUB_NUM_TOKEN_REGS) return;
    vm->regs.tokens[reg] = value;
}

/* ==========================================================================
 * I/O
 * ========================================================================== */

void substrate_io_configure(SubstrateVM* vm, uint8_t channel,
                            float (*read_fn)(void*),
                            void (*write_fn)(void*, float),
                            void* ctx) {
    if (!vm || channel >= SUB_NUM_IO_CHANNELS) return;
    vm->io[channel].read_fn = read_fn;
    vm->io[channel].write_fn = write_fn;
    vm->io[channel].user_ctx = ctx;
    vm->io[channel].enabled = true;
}

void substrate_io_inject(SubstrateVM* vm, uint8_t channel, float value) {
    if (!vm || channel >= SUB_NUM_IO_CHANNELS) return;
    vm->io[channel].value = value;
    vm->io[channel].ready = true;
}

float substrate_io_read(const SubstrateVM* vm, uint8_t channel) {
    if (!vm || channel >= SUB_NUM_IO_CHANNELS) return 0.0f;
    return vm->io[channel].value;
}

/* ==========================================================================
 * BOUNDARY BUFFERS
 * ========================================================================== */

void substrate_boundary_emit_byte(SubstrateVM* vm, uint32_t buf_id, uint8_t byte) {
    if (!vm || buf_id >= vm->num_boundaries) return;
    BoundaryBuffer* buf = &vm->boundaries[buf_id];
    if (buf->readonly) return;
    if (buf->size >= buf->capacity) return;

    buf->data[buf->size++] = byte;
}

uint32_t substrate_boundary_read(const SubstrateVM* vm, uint32_t buf_id,
                                  uint8_t* out, uint32_t max_len) {
    if (!vm || buf_id >= vm->num_boundaries) return 0;
    const BoundaryBuffer* buf = &vm->boundaries[buf_id];

    uint32_t copy_len = (buf->size < max_len) ? buf->size : max_len;
    memcpy(out, buf->data, copy_len);
    return copy_len;
}

/* ==========================================================================
 * DEBUG/UTILITY
 * ========================================================================== */

const char* substrate_opcode_name(uint8_t opcode) {
    static const char* names[] = {
        [OP_NOP] = "NOP",
        [OP_HALT] = "HALT",
        [OP_YIELD] = "YIELD",
        [OP_TICK] = "TICK",
        [OP_PHASE_R] = "PHASE.R",
        [OP_PHASE_P] = "PHASE.P",
        [OP_PHASE_C] = "PHASE.C",
        [OP_PHASE_X] = "PHASE.X",
        [OP_LDN] = "LDN",
        [OP_LDB] = "LDB",
        [OP_LDI] = "LDI",
        [OP_MOV] = "MOV",
        [OP_ADD] = "ADD",
        [OP_SUB] = "SUB",
        [OP_MUL] = "MUL",
        [OP_DIV] = "DIV",
        [OP_SQRT] = "SQRT",
        [OP_MIN] = "MIN",
        [OP_MAX] = "MAX",
        [OP_RELU] = "RELU",
        [OP_CMP] = "CMP",
        [OP_PROP_NEW] = "PROP.NEW",
        [OP_PROP_SCORE] = "PROP.SCORE",
        [OP_PROP_EFFECT] = "PROP.EFFECT",
        [OP_CHOOSE] = "CHOOSE",
        [OP_COMMIT] = "COMMIT",
        [OP_WITNESS] = "WITNESS",
        [OP_STN] = "STN",
        [OP_STB] = "STB",
        [OP_IN] = "IN",
        [OP_OUT] = "OUT",
        [OP_RAND] = "RAND",
        [OP_TIME] = "TIME",
    };

    if (opcode < sizeof(names)/sizeof(names[0]) && names[opcode]) {
        return names[opcode];
    }
    return "???";
}

const char* substrate_token_name(uint32_t token) {
    switch (token) {
        case TOK_VOID: return "VOID";
        case TOK_FALSE: return "FALSE";
        case TOK_TRUE: return "TRUE";
        case TOK_LT: return "LT";
        case TOK_EQ: return "EQ";
        case TOK_GT: return "GT";
        case TOK_OK: return "OK";
        case TOK_FAIL: return "FAIL";
        case TOK_REFUSE: return "REFUSE";
        case TOK_XFER_OK: return "XFER_OK";
        case TOK_DIFFUSE_OK: return "DIFFUSE_OK";
        case TOK_ERR: return "ERR";
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

const char* substrate_phase_name(SubstratePhase phase) {
    switch (phase) {
        case PHASE_READ: return "READ";
        case PHASE_PROPOSE: return "PROPOSE";
        case PHASE_CHOOSE: return "CHOOSE";
        case PHASE_COMMIT: return "COMMIT";
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

void substrate_get_stats(const SubstrateVM* vm, uint64_t* tick, uint64_t* instructions) {
    if (!vm) return;
    if (tick) *tick = vm->tick;
    if (instructions) *instructions = vm->instructions;
}

SubstrateState substrate_tick(SubstrateVM* vm) {
    if (!vm) return SUB_STATE_ERROR;

    /* Execute all four phases in sequence */

    /* 1. READ phase - load past trace values */
    vm->phase = PHASE_READ;
    substrate_run(vm, 1000);
    if (vm->state != SUB_STATE_RUNNING && vm->state != SUB_STATE_YIELDED) {
        return vm->state;
    }

    /* 2. PROPOSE phase - emit proposals */
    vm->phase = PHASE_PROPOSE;
    vm->state = SUB_STATE_RUNNING;
    substrate_run(vm, 1000);
    if (vm->state != SUB_STATE_RUNNING && vm->state != SUB_STATE_YIELDED) {
        return vm->state;
    }

    /* 3. CHOOSE phase - deterministic selection */
    vm->phase = PHASE_CHOOSE;
    vm->state = SUB_STATE_RUNNING;
    substrate_run(vm, 100);
    if (vm->state != SUB_STATE_RUNNING && vm->state != SUB_STATE_YIELDED) {
        return vm->state;
    }

    /* 4. COMMIT phase - apply effects */
    vm->phase = PHASE_COMMIT;
    vm->state = SUB_STATE_RUNNING;
    substrate_run(vm, 1000);

    /* Advance tick counter */
    vm->tick++;
    vm->prop_buf.count = 0;
    vm->prop_buf.chosen = 0xFFFFFFFF;

    return vm->state;
}

/* ==========================================================================
 * EFFECT TABLE
 * ========================================================================== */

static const EffectDescriptor effect_descriptors[] = {
    { EFFECT_NONE, "NONE", 0, false, false, false },
    { EFFECT_XFER_F, "XFER_F", 3, true, true, false },
    { EFFECT_DIFFUSE, "DIFFUSE", 2, false, true, true },
    { EFFECT_SET_F, "SET_F", 2, false, true, false },
    { EFFECT_ADD_F, "ADD_F", 2, false, true, false },
    { EFFECT_SET_Q, "SET_Q", 2, false, true, false },
    { EFFECT_ADD_Q, "ADD_Q", 2, false, true, false },
    { EFFECT_SET_A, "SET_A", 2, false, true, false },
    { EFFECT_SET_SIGMA, "SET_SIGMA", 2, false, true, false },
    { EFFECT_SET_P, "SET_P", 2, false, true, false },
    { EFFECT_SET_THETA, "SET_THETA", 3, false, true, false },
    { EFFECT_INC_K, "INC_K", 1, false, true, false },
    { EFFECT_INC_TAU, "INC_TAU", 2, false, true, false },
    { EFFECT_SET_C, "SET_C", 2, false, false, true },
    { EFFECT_ADD_C, "ADD_C", 2, false, false, true },
    { EFFECT_SET_PI, "SET_PI", 2, false, false, true },
    { EFFECT_ADD_PI, "ADD_PI", 2, false, false, true },
    { EFFECT_SET_BOND_SIGMA, "SET_BOND_SIGMA", 2, false, false, true },
    { EFFECT_EMIT_TOK, "EMIT_TOK", 2, false, false, false },
    { EFFECT_EMIT_BYTE, "EMIT_BYTE", 2, false, false, false },
    { EFFECT_EMIT_FLOAT, "EMIT_FLOAT", 2, false, false, false },
    { EFFECT_SET_SEED, "SET_SEED", 1, false, false, false },
};

const EffectDescriptor EFFECT_TABLE[] = {
    { EFFECT_NONE, "NONE", 0, false, false, false },
};
const size_t EFFECT_TABLE_SIZE = sizeof(effect_descriptors) / sizeof(effect_descriptors[0]);

const EffectDescriptor* effect_get_descriptor(EffectId id) {
    for (size_t i = 0; i < sizeof(effect_descriptors) / sizeof(effect_descriptors[0]); i++) {
        if (effect_descriptors[i].id == id) {
            return &effect_descriptors[i];
        }
    }
    return NULL;
}

const char* effect_get_name(EffectId id) {
    const EffectDescriptor* desc = effect_get_descriptor(id);
    return desc ? desc->name : "UNKNOWN";
}

bool effect_validate(SubstrateVM* vm, EffectId effect_id, const uint32_t* args) {
    const EffectDescriptor* desc = effect_get_descriptor(effect_id);
    if (!desc) return false;

    /* Validate based on effect type */
    switch (effect_id) {
        case EFFECT_NONE:
            return true;

        case EFFECT_XFER_F:
        case EFFECT_SET_F:
        case EFFECT_ADD_F:
        case EFFECT_SET_A:
        case EFFECT_SET_SIGMA:
        case EFFECT_SET_P:
        case EFFECT_INC_K: {
            /* Validate node ID */
            uint32_t node_id = args[0];
            return vm->nodes && node_id < vm->nodes->num_nodes;
        }

        case EFFECT_DIFFUSE:
        case EFFECT_SET_C:
        case EFFECT_ADD_C:
        case EFFECT_SET_PI:
        case EFFECT_SET_BOND_SIGMA: {
            /* Validate bond ID */
            uint32_t bond_id = args[0];
            return vm->bonds && bond_id < vm->bonds->num_bonds;
        }

        default:
            return true;
    }
}

void substrate_disassemble(const uint8_t* program, uint32_t size,
                            char* out, uint32_t out_size) {
    if (!program || !out || out_size == 0) return;

    uint32_t offset = 0;
    uint32_t written = 0;

    while (offset < size && written < out_size - 1) {
        uint32_t consumed;
        SubstrateInstr instr = substrate_decode(program, offset, &consumed);

        int n = snprintf(out + written, out_size - written,
                         "0x%04X: %s R%d, R%d, R%d [imm=%d]\n",
                         offset,
                         substrate_opcode_name(instr.opcode),
                         instr.dst, instr.src0, instr.src1,
                         instr.imm);

        if (n < 0 || (uint32_t)n >= out_size - written) break;
        written += n;
        offset += consumed;
    }

    out[written] = '\0';
}
