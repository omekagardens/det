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

/**
 * Decode helper: extract fields from 32-bit word
 * Encoding: [opcode:8][dst:5][src0:5][src1:5][imm:9]
 */
static SubstrateInstr decode_word(uint32_t word, const uint8_t* ext_data, uint32_t* consumed) {
    SubstrateInstr instr = {0};

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
            if (ext_data) {
                /* Extension word follows in same endianness */
                instr.ext = ((uint32_t)ext_data[0] << 24) |
                            ((uint32_t)ext_data[1] << 16) |
                            ((uint32_t)ext_data[2] << 8) |
                            ((uint32_t)ext_data[3]);
            }
            *consumed = 8;
            break;
        default:
            instr.has_ext = false;
            break;
    }

    return instr;
}

/**
 * Decode instruction from big-endian bytecode (network order, portable)
 */
SubstrateInstr substrate_decode(const uint8_t* data, uint32_t offset, uint32_t* consumed) {
    /* Read 32-bit word (big-endian) */
    uint32_t word = ((uint32_t)data[offset] << 24) |
                    ((uint32_t)data[offset + 1] << 16) |
                    ((uint32_t)data[offset + 2] << 8) |
                    ((uint32_t)data[offset + 3]);

    return decode_word(word, data + offset + 4, consumed);
}

/**
 * Decode instruction from little-endian bytecode (native for most CPUs)
 * This is the fast path for x86/ARM.
 */
SubstrateInstr substrate_decode_le(const uint8_t* data, uint32_t offset, uint32_t* consumed) {
    /* Read 32-bit word (little-endian) */
    uint32_t word = ((uint32_t)data[offset + 3] << 24) |
                    ((uint32_t)data[offset + 2] << 16) |
                    ((uint32_t)data[offset + 1] << 8) |
                    ((uint32_t)data[offset]);

    SubstrateInstr instr = {0};

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
            /* Extension in little-endian */
            instr.ext = ((uint32_t)data[offset + 7] << 24) |
                        ((uint32_t)data[offset + 6] << 16) |
                        ((uint32_t)data[offset + 5] << 8) |
                        ((uint32_t)data[offset + 4]);
            *consumed = 8;
            break;
        default:
            instr.has_ext = false;
            break;
    }

    return instr;
}

/**
 * Encode instruction to big-endian bytecode (portable)
 */
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

/**
 * Encode instruction to little-endian bytecode (fast native)
 */
uint32_t substrate_encode_le(const SubstrateInstr* instr, uint8_t* out) {
    uint32_t imm_bits = instr->imm & 0x1FF;
    uint32_t word = ((uint32_t)(instr->opcode & 0xFF) << 24) |
                    ((uint32_t)(instr->dst & 0x1F) << 19) |
                    ((uint32_t)(instr->src0 & 0x1F) << 14) |
                    ((uint32_t)(instr->src1 & 0x1F) << 9) |
                    (imm_bits);

    /* Little-endian */
    out[0] = word & 0xFF;
    out[1] = (word >> 8) & 0xFF;
    out[2] = (word >> 16) & 0xFF;
    out[3] = (word >> 24) & 0xFF;

    if (instr->has_ext) {
        out[4] = instr->ext & 0xFF;
        out[5] = (instr->ext >> 8) & 0xFF;
        out[6] = (instr->ext >> 16) & 0xFF;
        out[7] = (instr->ext >> 24) & 0xFF;
        return 8;
    }

    return 4;
}

/* ==========================================================================
 * PREDECODING
 * ========================================================================== */

bool substrate_predecode(SubstrateVM* vm) {
    if (!vm || !vm->program || vm->program_size == 0) {
        return false;
    }

    /* Free existing predecoded program */
    substrate_free_predecoded(vm);

    /* Estimate instruction count (max = size/4) */
    uint32_t max_instrs = vm->program_size / 4;

    vm->predecoded = (PredecodedProgram*)calloc(1, sizeof(PredecodedProgram));
    if (!vm->predecoded) return false;

    vm->predecoded->instrs = (SubstrateInstr*)calloc(max_instrs, sizeof(SubstrateInstr));
    if (!vm->predecoded->instrs) {
        free(vm->predecoded);
        vm->predecoded = NULL;
        return false;
    }

    vm->predecoded->capacity = max_instrs;

    /* Decode all instructions */
    uint32_t offset = 0;
    while (offset < vm->program_size) {
        uint32_t consumed;
        SubstrateInstr instr = substrate_decode(vm->program, offset, &consumed);

        vm->predecoded->instrs[vm->predecoded->count++] = instr;
        offset += consumed;

        if (instr.opcode == OP_HALT) {
            break;  /* Stop at HALT */
        }
    }

    vm->use_predecoded = true;
    return true;
}

void substrate_free_predecoded(SubstrateVM* vm) {
    if (!vm || !vm->predecoded) return;

    free(vm->predecoded->instrs);
    free(vm->predecoded);
    vm->predecoded = NULL;
    vm->use_predecoded = false;
}

void substrate_use_predecoded(SubstrateVM* vm, bool enable) {
    if (!vm) return;
    vm->use_predecoded = enable && (vm->predecoded != NULL);
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
 * PHASE LEGALITY TABLE
 * ========================================================================== */

/**
 * Phase legality enforcement:
 * - READ phase: Loads only (LDN, LDB, LDNB, LDI, LDI_F)
 * - PROPOSE phase: Proposal ops only (PROP_NEW, PROP_SCORE, PROP_EFFECT, PROP_ARG, PROP_END)
 * - CHOOSE phase: CHOOSE only
 * - COMMIT phase: Stores and COMMIT only (STN, STB, STT, COMMIT)
 *
 * Arithmetic, register ops, and comparisons are allowed in all phases
 * (they operate on registers, not trace state).
 */

typedef enum {
    PHASE_MASK_READ    = (1 << PHASE_READ),
    PHASE_MASK_PROPOSE = (1 << PHASE_PROPOSE),
    PHASE_MASK_CHOOSE  = (1 << PHASE_CHOOSE),
    PHASE_MASK_COMMIT  = (1 << PHASE_COMMIT),
    PHASE_MASK_ALL     = 0x0F,
} PhaseMask;

static uint8_t opcode_phase_mask(uint8_t opcode) {
    switch (opcode) {
        /* Phase control - allowed anywhere */
        case OP_NOP:
        case OP_HALT:
        case OP_YIELD:
        case OP_TICK:
        case OP_PHASE_R:
        case OP_PHASE_P:
        case OP_PHASE_C:
        case OP_PHASE_X:
            return PHASE_MASK_ALL;

        /* Typed loads - READ phase only */
        case OP_LDN:
        case OP_LDB:
        case OP_LDNB:
            return PHASE_MASK_READ;

        /* Immediate loads - anywhere (no trace access) */
        case OP_LDI:
        case OP_LDI_F:
            return PHASE_MASK_ALL;

        /* Register ops - anywhere */
        case OP_MOV:
        case OP_MOVR:
        case OP_MOVT:
        case OP_TSET:
        case OP_TGET:
            return PHASE_MASK_ALL;

        /* Arithmetic - anywhere (pure register computation) */
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_MAD:
        case OP_NEG:
        case OP_ABS:
        case OP_SQRT:
        case OP_MIN:
        case OP_MAX:
        case OP_RELU:
        case OP_CLAMP:
            return PHASE_MASK_ALL;

        /* Comparison - anywhere */
        case OP_CMP:
        case OP_CMPE:
        case OP_TEQ:
        case OP_TNE:
            return PHASE_MASK_ALL;

        /* Proposals - PROPOSE phase only */
        case OP_PROP_NEW:
        case OP_PROP_SCORE:
        case OP_PROP_EFFECT:
        case OP_PROP_ARG:
        case OP_PROP_END:
            return PHASE_MASK_PROPOSE;

        /* Choose - CHOOSE phase only */
        case OP_CHOOSE:
            return PHASE_MASK_CHOOSE;

        /* Commit/Witness - COMMIT phase only */
        case OP_COMMIT:
        case OP_WITNESS:
            return PHASE_MASK_COMMIT;

        /* Stores - COMMIT phase only */
        case OP_STN:
        case OP_STB:
        case OP_STT:
            return PHASE_MASK_COMMIT;

        /* I/O - READ for input, COMMIT for output */
        case OP_IN:
        case OP_POLL:
            return PHASE_MASK_READ;

        case OP_OUT:
        case OP_EMIT:
            return PHASE_MASK_COMMIT;

        /* System - anywhere */
        case OP_RAND:
        case OP_SEED:
        case OP_LANE:
        case OP_TIME:
        case OP_DEBUG:
            return PHASE_MASK_ALL;

        default:
            return 0; /* Unknown opcode - disallowed */
    }
}

static bool opcode_allowed_in_phase(uint8_t opcode, SubstratePhase phase) {
    uint8_t mask = opcode_phase_mask(opcode);
    return (mask & (1 << phase)) != 0;
}

/* ==========================================================================
 * RANDOM NUMBER GENERATOR - Trace-derived (DET-clean)
 * ========================================================================== */

/**
 * MurmurHash3 32-bit finalizer - fast hash mixing
 * Used to derive deterministic seed from trace state.
 */
static uint32_t hash_mix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

/**
 * Combine multiple values into a hash.
 * seed = hash(r, k, tick, lane_id) for node lanes
 * seed = hash(r_i ^ r_j, bond_id, tick, lane_id) for bond lanes
 */
static uint32_t hash_combine(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t h = a;
    h = hash_mix32(h ^ b);
    h = hash_mix32(h ^ c);
    h = hash_mix32(h ^ d);
    return h;
}

/**
 * Derive lane-local seed from trace state.
 * This ensures reproducibility and parallel safety.
 */
static uint64_t derive_lane_seed(SubstrateVM* vm, uint32_t extra) {
    uint32_t r = 0, k = 0;

    /* Get trace values if we have a current node */
    if (vm->nodes && vm->lane_id < vm->nodes->num_nodes) {
        r = vm->nodes->r[vm->lane_id];
        k = vm->nodes->k[vm->lane_id];
    }

    uint32_t h = hash_combine(r, k, (uint32_t)vm->tick, vm->lane_id ^ extra);
    return (uint64_t)h | ((uint64_t)hash_mix32(h + 1) << 32);
}

static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/**
 * Get random float [0,1) using trace-derived seed.
 * The seed is updated per-call but derived from trace state.
 */
static float random_float(SubstrateVM* vm) {
    /* Derive seed from trace on first call of tick, then iterate */
    if (vm->seed == 0) {
        vm->seed = derive_lane_seed(vm, 0);
    }
    uint64_t r = xorshift64(&vm->seed);
    return (float)(r & 0xFFFFFFFF) / 4294967296.0f;
}

/**
 * Reset seed for new tick (will be re-derived on next random_float call)
 */
static void reset_lane_seed(SubstrateVM* vm) {
    vm->seed = 0;  /* Will be derived from trace on next use */
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

/**
 * Check if current lane owns this effect (for deduplication).
 * Returns true if the effect should be applied, false to skip.
 */
static bool effect_ownership_check(SubstrateVM* vm, EffectId effect_id, const uint32_t* args) {
    if (vm->lane_mode == LANE_OWNER_NONE) {
        return true;  /* No ownership enforcement */
    }

    switch (effect_id) {
        case EFFECT_XFER_F: {
            /* For node-lane model: only lane with lane_id == min(src, dst) owns */
            if (vm->lane_mode == LANE_OWNER_NODE) {
                uint32_t src_node = args[0];
                uint32_t dst_node = args[1];
                uint32_t owner = (src_node < dst_node) ? src_node : dst_node;
                return (vm->lane_id == owner);
            }
            return true;
        }

        case EFFECT_DIFFUSE:
        case EFFECT_SET_C:
        case EFFECT_ADD_C:
        case EFFECT_SET_PI:
        case EFFECT_ADD_PI:
        case EFFECT_SET_BOND_SIGMA: {
            /* For bond-lane model: only lane with lane_id == bond_id owns */
            if (vm->lane_mode == LANE_OWNER_BOND) {
                uint32_t bond_id = args[0];
                return (vm->lane_id == bond_id);
            }
            /* For node-lane model: only lane with lane_id == node_i owns */
            if (vm->lane_mode == LANE_OWNER_NODE && vm->bonds) {
                uint32_t bond_id = args[0];
                if (bond_id < vm->bonds->num_bonds) {
                    uint32_t owner = vm->bonds->node_i[bond_id];
                    return (vm->lane_id == owner);
                }
            }
            return true;
        }

        default:
            return true;  /* Other effects have no ownership requirements */
    }
}

uint32_t effect_apply(SubstrateVM* vm, EffectId effect_id, const uint32_t* args) {
    /* Ownership check - skip if this lane doesn't own the effect */
    if (!effect_ownership_check(vm, effect_id, args)) {
        return TOK_OK;  /* Silent skip - another lane will handle it */
    }

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

    /* Phase legality check - reject instructions not allowed in current phase */
    if (!opcode_allowed_in_phase(instr->opcode, vm->phase)) {
        snprintf(vm->error_msg, sizeof(vm->error_msg),
                 "Phase violation: %s (0x%02X) not allowed in %s phase at PC=0x%X",
                 substrate_opcode_name(instr->opcode), instr->opcode,
                 substrate_phase_name(vm->phase), vm->pc);
        vm->state = SUB_STATE_ERROR;
        /* Store phase violation token in T0 for inspection */
        write_token(regs, 0, TOK_PHASE_VIOLATION);
        return;
    }

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
        /* Reset seed so it's re-derived from trace */
        reset_lane_seed(vm);
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

    /* Free predecoded program */
    substrate_free_predecoded(vm);

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

void substrate_set_lane_mode(SubstrateVM* vm, LaneOwnershipMode mode) {
    vm->lane_mode = mode;
}

SubstrateState substrate_step(SubstrateVM* vm) {
    if (!vm) return SUB_STATE_ERROR;
    if (vm->state != SUB_STATE_RUNNING) return vm->state;

    SubstrateInstr instr;

    /* Use predecoded path if available (faster) */
    if (vm->use_predecoded && vm->predecoded) {
        if (vm->pc >= vm->predecoded->count) {
            vm->state = SUB_STATE_HALTED;
            return vm->state;
        }
        instr = vm->predecoded->instrs[vm->pc];
        vm->pc++;  /* PC is instruction index in predecoded mode */
    } else {
        /* Raw bytecode decode */
        if (vm->pc >= vm->program_size) {
            vm->state = SUB_STATE_HALTED;
            return vm->state;
        }

        uint32_t consumed;
        instr = substrate_decode(vm->program, vm->pc, &consumed);
        vm->pc += consumed;
    }

    /* Trace */
    if (vm->trace_fn) {
        vm->trace_fn(vm->trace_ctx, &instr, &vm->regs);
    }

    /* Execute */
    execute_instruction(vm, &instr);

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

/* ==========================================================================
 * THREADED DISPATCH (GCC/Clang computed goto)
 * ========================================================================== */

#if defined(__GNUC__) || defined(__clang__)

/**
 * Fast execution using computed goto (threaded dispatch).
 * This eliminates switch overhead by using a dispatch table.
 * Only available on GCC/Clang. For MSVC, falls back to substrate_run.
 */
SubstrateState substrate_run_fast(SubstrateVM* vm, uint32_t max_steps) {
    if (!vm || !vm->predecoded || !vm->use_predecoded) {
        /* Fall back to standard run if predecoded not available */
        return substrate_run(vm, max_steps);
    }

    if (vm->state != SUB_STATE_RUNNING) return vm->state;

    /* Local register cache for hot loop */
    SubstrateRegs* regs = &vm->regs;
    PredecodedProgram* prog = vm->predecoded;
    uint32_t pc = vm->pc;
    uint32_t steps = 0;

    /* Dispatch table - must match SubstrateOpcode enum */
    /* Default all entries to op_invalid, then set specific handlers */
    static const void* dispatch_table[256] = {
        [0 ... 255] = &&op_invalid,  /* Default */
    };

    /* Initialize handlers on first call (static ensures single init) */
    static bool table_initialized = false;
    if (!table_initialized) {
        ((void**)dispatch_table)[OP_NOP] = &&op_nop;
        ((void**)dispatch_table)[OP_HALT] = &&op_halt;
        ((void**)dispatch_table)[OP_YIELD] = &&op_yield;
        ((void**)dispatch_table)[OP_TICK] = &&op_tick;
        ((void**)dispatch_table)[OP_PHASE_R] = &&op_phase_r;
        ((void**)dispatch_table)[OP_PHASE_P] = &&op_phase_p;
        ((void**)dispatch_table)[OP_PHASE_C] = &&op_phase_c;
        ((void**)dispatch_table)[OP_PHASE_X] = &&op_phase_x;
        ((void**)dispatch_table)[OP_LDN] = &&op_ldn;
        ((void**)dispatch_table)[OP_LDB] = &&op_ldb;
        ((void**)dispatch_table)[OP_LDI] = &&op_ldi;
        ((void**)dispatch_table)[OP_LDI_F] = &&op_ldi_f;
        ((void**)dispatch_table)[OP_MOV] = &&op_mov;
        ((void**)dispatch_table)[OP_ADD] = &&op_add;
        ((void**)dispatch_table)[OP_SUB] = &&op_sub;
        ((void**)dispatch_table)[OP_MUL] = &&op_mul;
        ((void**)dispatch_table)[OP_DIV] = &&op_div;
        ((void**)dispatch_table)[OP_MAD] = &&op_mad;
        ((void**)dispatch_table)[OP_NEG] = &&op_neg;
        ((void**)dispatch_table)[OP_ABS] = &&op_abs;
        ((void**)dispatch_table)[OP_SQRT] = &&op_sqrt;
        ((void**)dispatch_table)[OP_MIN] = &&op_min;
        ((void**)dispatch_table)[OP_MAX] = &&op_max;
        ((void**)dispatch_table)[OP_RELU] = &&op_relu;
        ((void**)dispatch_table)[OP_CLAMP] = &&op_clamp;
        ((void**)dispatch_table)[OP_CMP] = &&op_cmp;
        ((void**)dispatch_table)[OP_PROP_NEW] = &&op_prop_new;
        ((void**)dispatch_table)[OP_PROP_SCORE] = &&op_prop_score;
        ((void**)dispatch_table)[OP_CHOOSE] = &&op_choose;
        ((void**)dispatch_table)[OP_COMMIT] = &&op_commit;
        ((void**)dispatch_table)[OP_WITNESS] = &&op_witness;
        ((void**)dispatch_table)[OP_STN] = &&op_stn;
        ((void**)dispatch_table)[OP_STB] = &&op_stb;
        ((void**)dispatch_table)[OP_IN] = &&op_in;
        ((void**)dispatch_table)[OP_OUT] = &&op_out;
        ((void**)dispatch_table)[OP_RAND] = &&op_rand;
        ((void**)dispatch_table)[OP_TIME] = &&op_time;
        ((void**)dispatch_table)[OP_LANE] = &&op_lane;
        table_initialized = true;
    }

    float a, b;
    union { uint32_t u; float f; } conv;
    const SubstrateInstr* instr;

    #define DISPATCH_NEXT() do { \
        if (++steps >= max_steps || pc >= prog->count || vm->state != SUB_STATE_RUNNING) { \
            goto done; \
        } \
        instr = &prog->instrs[pc++]; \
        goto *dispatch_table[instr->opcode]; \
    } while(0)

    /* Start dispatch */
    if (pc >= prog->count) goto done;
    instr = &prog->instrs[pc++];
    goto *dispatch_table[instr->opcode];

    /* ----- Handlers ----- */

op_nop:
    DISPATCH_NEXT();

op_halt:
    vm->state = SUB_STATE_HALTED;
    goto done;

op_yield:
    vm->state = SUB_STATE_YIELDED;
    goto done;

op_tick:
    vm->tick++;
    vm->prop_buf.count = 0;
    vm->prop_buf.chosen = 0xFFFFFFFF;
    reset_lane_seed(vm);
    DISPATCH_NEXT();

op_phase_r:
    vm->phase = PHASE_READ;
    DISPATCH_NEXT();

op_phase_p:
    vm->phase = PHASE_PROPOSE;
    DISPATCH_NEXT();

op_phase_c:
    vm->phase = PHASE_CHOOSE;
    DISPATCH_NEXT();

op_phase_x:
    vm->phase = PHASE_COMMIT;
    DISPATCH_NEXT();

op_ldn:
    if (vm->phase != PHASE_READ) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint32_t ref = regs->refs[instr->src0 & 0x7];
        uint32_t node_id = REF_ID(ref);
        regs->scalars[instr->dst & 0xF] = substrate_node_get_f(vm, node_id, (NodeFieldId)instr->src1);
    }
    DISPATCH_NEXT();

op_ldb:
    if (vm->phase != PHASE_READ) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint32_t ref = regs->refs[instr->src0 & 0x7];
        uint32_t bond_id = REF_ID(ref);
        regs->scalars[instr->dst & 0xF] = substrate_bond_get_f(vm, bond_id, (BondFieldId)instr->src1);
    }
    DISPATCH_NEXT();

op_ldi:
    regs->scalars[instr->dst & 0xF] = (float)instr->imm;
    DISPATCH_NEXT();

op_ldi_f:
    conv.u = instr->ext;
    regs->scalars[instr->dst & 0xF] = conv.f;
    DISPATCH_NEXT();

op_mov:
    regs->scalars[instr->dst & 0xF] = regs->scalars[instr->src0 & 0xF];
    DISPATCH_NEXT();

op_add:
    regs->scalars[instr->dst & 0xF] = regs->scalars[instr->src0 & 0xF] + regs->scalars[instr->src1 & 0xF];
    DISPATCH_NEXT();

op_sub:
    regs->scalars[instr->dst & 0xF] = regs->scalars[instr->src0 & 0xF] - regs->scalars[instr->src1 & 0xF];
    DISPATCH_NEXT();

op_mul:
    regs->scalars[instr->dst & 0xF] = regs->scalars[instr->src0 & 0xF] * regs->scalars[instr->src1 & 0xF];
    DISPATCH_NEXT();

op_div:
    b = regs->scalars[instr->src1 & 0xF];
    regs->scalars[instr->dst & 0xF] = (b != 0.0f) ? (regs->scalars[instr->src0 & 0xF] / b) : 0.0f;
    DISPATCH_NEXT();

op_mad:
    a = regs->scalars[instr->src0 & 0xF];
    b = regs->scalars[instr->src1 & 0xF];
    regs->scalars[instr->dst & 0xF] += a * b;
    DISPATCH_NEXT();

op_neg:
    regs->scalars[instr->dst & 0xF] = -regs->scalars[instr->src0 & 0xF];
    DISPATCH_NEXT();

op_abs:
    regs->scalars[instr->dst & 0xF] = fabsf(regs->scalars[instr->src0 & 0xF]);
    DISPATCH_NEXT();

op_sqrt:
    regs->scalars[instr->dst & 0xF] = sqrtf(fmaxf(0.0f, regs->scalars[instr->src0 & 0xF]));
    DISPATCH_NEXT();

op_min:
    regs->scalars[instr->dst & 0xF] = fminf(regs->scalars[instr->src0 & 0xF], regs->scalars[instr->src1 & 0xF]);
    DISPATCH_NEXT();

op_max:
    regs->scalars[instr->dst & 0xF] = fmaxf(regs->scalars[instr->src0 & 0xF], regs->scalars[instr->src1 & 0xF]);
    DISPATCH_NEXT();

op_relu:
    regs->scalars[instr->dst & 0xF] = fmaxf(0.0f, regs->scalars[instr->src0 & 0xF]);
    DISPATCH_NEXT();

op_clamp:
    regs->scalars[instr->dst & 0xF] = fmaxf(0.0f, fminf(1.0f, regs->scalars[instr->src0 & 0xF]));
    DISPATCH_NEXT();

op_cmp:
    a = regs->scalars[instr->src0 & 0xF];
    b = regs->scalars[instr->src1 & 0xF];
    regs->tokens[instr->dst & 0x7] = (a < b) ? TOK_LT : ((a > b) ? TOK_GT : TOK_EQ);
    DISPATCH_NEXT();

op_prop_new:
    if (vm->phase != PHASE_PROPOSE) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint32_t idx = substrate_prop_new(vm);
        regs->refs[instr->dst & 0x7] = REF_MAKE(REF_TYPE_PROP, idx);
    }
    DISPATCH_NEXT();

op_prop_score:
    if (vm->phase != PHASE_PROPOSE) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint32_t ref = regs->refs[instr->dst & 0x7];
        uint32_t idx = REF_ID(ref);
        substrate_prop_score(vm, idx, regs->scalars[instr->src0 & 0xF]);
    }
    DISPATCH_NEXT();

op_choose:
    if (vm->phase != PHASE_CHOOSE) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        float decisiveness = regs->scalars[instr->src0 & 0xF];
        uint32_t choice = substrate_choose(vm, decisiveness);
        regs->refs[instr->dst & 0x7] = REF_MAKE(REF_TYPE_CHOICE, choice);
    }
    DISPATCH_NEXT();

op_commit:
    if (vm->phase != PHASE_COMMIT) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint32_t result = substrate_commit(vm);
        regs->tokens[instr->dst & 0x7] = result;
    }
    DISPATCH_NEXT();

op_witness:
    if (vm->phase != PHASE_COMMIT) { vm->state = SUB_STATE_ERROR; goto done; }
    regs->tokens[instr->dst & 0x7] = (uint32_t)instr->imm;
    DISPATCH_NEXT();

op_stn:
    if (vm->phase != PHASE_COMMIT) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint32_t ref = regs->refs[instr->dst & 0x7];
        uint32_t node_id = REF_ID(ref);
        substrate_node_set_f(vm, node_id, (NodeFieldId)instr->src0, regs->scalars[instr->src1 & 0xF]);
    }
    DISPATCH_NEXT();

op_stb:
    if (vm->phase != PHASE_COMMIT) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint32_t ref = regs->refs[instr->dst & 0x7];
        uint32_t bond_id = REF_ID(ref);
        substrate_bond_set_f(vm, bond_id, (BondFieldId)instr->src0, regs->scalars[instr->src1 & 0xF]);
    }
    DISPATCH_NEXT();

op_in:
    if (vm->phase != PHASE_READ) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint8_t ch = instr->imm & 0xF;
        if (vm->io[ch].read_fn) {
            regs->scalars[instr->dst & 0xF] = vm->io[ch].read_fn(vm->io[ch].user_ctx);
        } else {
            regs->scalars[instr->dst & 0xF] = vm->io[ch].value;
        }
    }
    DISPATCH_NEXT();

op_out:
    if (vm->phase != PHASE_COMMIT) { vm->state = SUB_STATE_ERROR; goto done; }
    {
        uint8_t ch = instr->imm & 0xF;
        a = regs->scalars[instr->src0 & 0xF];
        vm->io[ch].value = a;
        if (vm->io[ch].write_fn) {
            vm->io[ch].write_fn(vm->io[ch].user_ctx, a);
        }
    }
    DISPATCH_NEXT();

op_rand:
    regs->scalars[instr->dst & 0xF] = random_float(vm);
    DISPATCH_NEXT();

op_time:
    regs->scalars[instr->dst & 0xF] = (float)vm->tick;
    DISPATCH_NEXT();

op_lane:
    regs->scalars[instr->dst & 0xF] = (float)vm->lane_id;
    DISPATCH_NEXT();

op_invalid:
    vm->state = SUB_STATE_ERROR;
    goto done;

done:
    vm->pc = pc - 1;  /* Adjust for pre-increment */
    vm->instructions += steps;

    #undef DISPATCH_NEXT
    return vm->state;
}

#else /* !__GNUC__ && !__clang__ */

/* Fallback for non-GCC/Clang compilers */
SubstrateState substrate_run_fast(SubstrateVM* vm, uint32_t max_steps) {
    return substrate_run(vm, max_steps);
}

#endif /* __GNUC__ || __clang__ */

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
        case TOK_XFER_PARTIAL: return "XFER_PARTIAL";
        case TOK_DIFFUSE_OK: return "DIFFUSE_OK";
        case TOK_GRACE_OK: return "GRACE_OK";
        case TOK_GRACE_NONE: return "GRACE_NONE";
        case TOK_PHASE_VIOLATION: return "PHASE_VIOLATION";
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

    /* Reset seed for trace-derived randomness */
    reset_lane_seed(vm);

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
