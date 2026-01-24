/**
 * EIS Virtual Machine - C Implementation
 * =======================================
 *
 * Native C implementation of the Existence Instruction Set VM.
 */

#include "../include/eis_vm.h"
#include "../include/det_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ==========================================================================
 * INSTRUCTION DECODE/ENCODE
 * ========================================================================== */

EIS_Instruction eis_decode_instruction(const uint8_t *data, uint32_t offset,
                                        uint32_t *consumed) {
    EIS_Instruction instr = {0};

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
        case EIS_OP_LDI_EXT:
        case EIS_OP_LDNB:
        case EIS_OP_CMP_EPS:
        case EIS_OP_PROP_EFFECT:
        case EIS_OP_CHOOSE:
        case EIS_OP_BR_TOK:
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

uint32_t eis_encode_instruction(const EIS_Instruction *instr, uint8_t *out) {
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
 * REGISTER ACCESS
 * ========================================================================== */

static float reg_read_scalar(EIS_RegisterFile *regs, uint8_t reg) {
    if (reg < EIS_MAX_SCALAR_REGS) {
        return regs->scalars[reg];
    }
    return 0.0f;
}

static void reg_write_scalar(EIS_RegisterFile *regs, uint8_t reg, float value) {
    if (reg < EIS_MAX_SCALAR_REGS) {
        regs->scalars[reg] = value;
    }
}

static uint32_t reg_read_ref(EIS_RegisterFile *regs, uint8_t reg) {
    if (reg >= 16 && reg < 24) {
        return regs->refs[reg - 16];
    }
    return 0;
}

static void reg_write_ref(EIS_RegisterFile *regs, uint8_t reg, uint32_t value, uint8_t type) {
    if (reg >= 16 && reg < 24) {
        regs->refs[reg - 16] = value;
        regs->ref_types[reg - 16] = type;
    }
}

static uint32_t reg_read_token(EIS_RegisterFile *regs, uint8_t reg) {
    if (reg >= 24 && reg < 32) {
        return regs->tokens[reg - 24];
    }
    return EIS_TOK_VOID;
}

static void reg_write_token(EIS_RegisterFile *regs, uint8_t reg, uint32_t value) {
    if (reg >= 24 && reg < 32) {
        regs->tokens[reg - 24] = value;
    }
}

/* Unified read: 0-15 scalar, 16-23 ref, 24-31 token */
static float reg_read(EIS_RegisterFile *regs, uint8_t reg) {
    if (reg < 16) {
        return regs->scalars[reg];
    } else if (reg < 24) {
        return (float)regs->refs[reg - 16];
    } else if (reg < 32) {
        return (float)regs->tokens[reg - 24];
    }
    return 0.0f;
}

static void reg_write(EIS_RegisterFile *regs, uint8_t reg, float value) {
    if (reg < 16) {
        regs->scalars[reg] = value;
    } else if (reg < 24) {
        regs->refs[reg - 16] = (uint32_t)value;
    } else if (reg < 32) {
        regs->tokens[reg - 24] = (uint32_t)value;
    }
}

/* ==========================================================================
 * PHASE CONTROL
 * ========================================================================== */

static bool phase_can_read_trace(EIS_PhaseController *phases) {
    return phases->current_phase >= EIS_PHASE_READ &&
           phases->current_phase <= EIS_PHASE_COMMIT;
}

static bool phase_can_write_trace(EIS_PhaseController *phases) {
    return phases->current_phase == EIS_PHASE_COMMIT;
}

static bool phase_can_propose(EIS_PhaseController *phases) {
    return phases->current_phase == EIS_PHASE_PROPOSE;
}

static bool phase_can_choose(EIS_PhaseController *phases) {
    return phases->current_phase == EIS_PHASE_CHOOSE;
}

static bool phase_can_commit(EIS_PhaseController *phases) {
    return phases->current_phase == EIS_PHASE_COMMIT;
}

static bool phase_check(EIS_PhaseController *phases, bool (*check_fn)(EIS_PhaseController*),
                        const char *op_name) {
    if (check_fn(phases)) {
        return true;
    }
    if (phases->strict) {
        phases->violation_count++;
        return false;
    }
    return true;  /* Non-strict: allow anyway */
}

/* ==========================================================================
 * DET INTEGRATION
 * ========================================================================== */

static float det_read_node_field(DETCore *det, uint16_t node_id, uint8_t field) {
    if (!det || node_id >= det->num_nodes) return 0.0f;

    DETNode *node = &det->nodes[node_id];
    switch (field) {
        case EIS_FIELD_F: return node->F;
        case EIS_FIELD_Q: return node->q;
        case EIS_FIELD_A: return node->a;
        case EIS_FIELD_THETA: return node->theta;
        case EIS_FIELD_SIGMA: return node->sigma;
        case EIS_FIELD_P: return node->P;
        case EIS_FIELD_TAU: return node->tau_accumulated;
        default: return 0.0f;
    }
}

static void det_write_node_field(DETCore *det, uint16_t node_id, uint8_t field, float value) {
    if (!det || node_id >= det->num_nodes) return;

    DETNode *node = &det->nodes[node_id];
    switch (field) {
        case EIS_FIELD_F: node->F = fmaxf(0.0f, value); break;
        case EIS_FIELD_Q: node->q = fmaxf(0.0f, value); break;
        case EIS_FIELD_A: node->a = fminf(1.0f, fmaxf(0.0f, value)); break;
        case EIS_FIELD_THETA: node->theta = fmodf(value, 2.0f * 3.14159265f); break;
        case EIS_FIELD_SIGMA: node->sigma = fmaxf(0.0f, value); break;
        default: break;
    }
}

static EIS_WitnessToken det_apply_xfer(DETCore *det, uint16_t src, uint16_t dst, float amount) {
    if (!det || src >= det->num_nodes || dst >= det->num_nodes) {
        return EIS_TOK_XFER_BLOCKED;
    }

    DETNode *src_node = &det->nodes[src];
    DETNode *dst_node = &det->nodes[dst];

    /* Clamp to available */
    float actual = fminf(amount, src_node->F);
    if (actual <= 0.0f) {
        return EIS_TOK_XFER_BLOCKED;
    }

    /* Antisymmetric update */
    src_node->F -= actual;
    dst_node->F += actual;

    return (actual < amount) ? EIS_TOK_XFER_PARTIAL : EIS_TOK_XFER_OK;
}

/* ==========================================================================
 * INSTRUCTION EXECUTION
 * ========================================================================== */

static void execute_instruction(EIS_VM *vm, EIS_Lane *lane, EIS_Instruction *instr) {
    EIS_RegisterFile *regs = &lane->regs;
    DETCore *det = vm->det;

    switch (instr->opcode) {

    /* === Phase Control === */
    case EIS_OP_NOP:
        break;

    case EIS_OP_PHASE:
        vm->phases.current_phase = (EIS_Phase)instr->imm;
        break;

    case EIS_OP_HALT:
        lane->state = EIS_STATE_HALTED;
        break;

    case EIS_OP_YIELD:
        lane->state = EIS_STATE_YIELDED;
        break;

    /* === Load Operations === */
    case EIS_OP_LDI:
        reg_write(regs, instr->dst, (float)instr->imm);
        break;

    case EIS_OP_LDI_EXT:
        if (instr->has_ext) {
            /* Interpret ext as float bits */
            union { uint32_t u; float f; } conv;
            conv.u = instr->ext;
            reg_write(regs, instr->dst, conv.f);
        }
        break;

    case EIS_OP_LDN:
        if (phase_check(&vm->phases, phase_can_read_trace, "LDN")) {
            uint16_t node_id;
            if (instr->src0 >= 16 && instr->src0 < 24) {
                node_id = (uint16_t)regs->refs[instr->src0 - 16];
            } else {
                node_id = (uint16_t)regs->scalars[instr->src0];
            }
            float value = det_read_node_field(det, node_id, (uint8_t)instr->imm);
            reg_write(regs, instr->dst, value);
        }
        break;

    case EIS_OP_LDT:
        {
            uint8_t tok_idx = (instr->src0 >= 24) ? instr->src0 - 24 : 0;
            reg_write(regs, instr->dst, (float)regs->tokens[tok_idx]);
        }
        break;

    /* === Store Operations === */
    case EIS_OP_ST_TOK:
        if (phase_check(&vm->phases, phase_can_commit, "ST_TOK")) {
            uint8_t tok_idx = (instr->dst >= 24) ? instr->dst - 24 : 0;
            regs->tokens[tok_idx] = (uint32_t)reg_read(regs, instr->src0);
        }
        break;

    case EIS_OP_ST_NODE:
        if (phase_check(&vm->phases, phase_can_commit, "ST_NODE")) {
            uint16_t node_id = regs->self_node;
            float value = reg_read(regs, instr->src0);
            det_write_node_field(det, node_id, (uint8_t)instr->imm, value);
        }
        break;

    /* === Arithmetic === */
    case EIS_OP_ADD:
        {
            float a = reg_read(regs, instr->src0);
            float b = reg_read(regs, instr->src1);
            reg_write(regs, instr->dst, a + b);
        }
        break;

    case EIS_OP_SUB:
        {
            float a = reg_read(regs, instr->src0);
            float b = reg_read(regs, instr->src1);
            reg_write(regs, instr->dst, a - b);
        }
        break;

    case EIS_OP_MUL:
        {
            float a = reg_read(regs, instr->src0);
            float b = reg_read(regs, instr->src1);
            reg_write(regs, instr->dst, a * b);
        }
        break;

    case EIS_OP_DIV:
        {
            float a = reg_read(regs, instr->src0);
            float b = reg_read(regs, instr->src1);
            reg_write(regs, instr->dst, (b != 0.0f) ? (a / b) : 0.0f);
        }
        break;

    case EIS_OP_NEG:
        {
            float a = reg_read(regs, instr->src0);
            reg_write(regs, instr->dst, -a);
        }
        break;

    case EIS_OP_ABS:
        {
            float a = reg_read(regs, instr->src0);
            reg_write(regs, instr->dst, fabsf(a));
        }
        break;

    /* === Math Functions === */
    case EIS_OP_SQRT:
        {
            float a = reg_read(regs, instr->src0);
            reg_write(regs, instr->dst, sqrtf(fmaxf(0.0f, a)));
        }
        break;

    case EIS_OP_SIN:
        {
            float a = reg_read(regs, instr->src0);
            reg_write(regs, instr->dst, sinf(a));
        }
        break;

    case EIS_OP_COS:
        {
            float a = reg_read(regs, instr->src0);
            reg_write(regs, instr->dst, cosf(a));
        }
        break;

    case EIS_OP_MIN:
        {
            float a = reg_read(regs, instr->src0);
            float b = reg_read(regs, instr->src1);
            reg_write(regs, instr->dst, fminf(a, b));
        }
        break;

    case EIS_OP_MAX:
        {
            float a = reg_read(regs, instr->src0);
            float b = reg_read(regs, instr->src1);
            reg_write(regs, instr->dst, fmaxf(a, b));
        }
        break;

    case EIS_OP_RELU:
        {
            float a = reg_read(regs, instr->src0);
            reg_write(regs, instr->dst, fmaxf(0.0f, a));
        }
        break;

    /* === Comparison === */
    case EIS_OP_CMP:
        {
            float a = reg_read(regs, instr->src0);
            float b = reg_read(regs, instr->src1);
            uint8_t tok_idx = (instr->dst >= 24) ? instr->dst - 24 : 0;
            if (a < b) {
                regs->tokens[tok_idx] = EIS_TOK_LT;
            } else if (a > b) {
                regs->tokens[tok_idx] = EIS_TOK_GT;
            } else {
                regs->tokens[tok_idx] = EIS_TOK_EQ;
            }
        }
        break;

    case EIS_OP_TSET:
        {
            uint8_t tok_idx = (instr->dst >= 24) ? instr->dst - 24 : 0;
            regs->tokens[tok_idx] = (uint32_t)instr->imm;
        }
        break;

    case EIS_OP_TMOV:
        {
            uint8_t src_idx = (instr->src0 >= 24) ? instr->src0 - 24 : 0;
            uint8_t dst_idx = (instr->dst >= 24) ? instr->dst - 24 : 0;
            regs->tokens[dst_idx] = regs->tokens[src_idx];
        }
        break;

    /* === Proposal Operations === */
    case EIS_OP_PROP_BEGIN:
        if (phase_check(&vm->phases, phase_can_propose, "PROP_BEGIN")) {
            if (lane->proposals.num_proposals < EIS_MAX_PROPOSALS) {
                lane->current_proposal = lane->proposals.num_proposals;
                EIS_Proposal *prop = &lane->proposals.proposals[lane->current_proposal];
                memset(prop, 0, sizeof(EIS_Proposal));
                lane->proposals.num_proposals++;
            }
        }
        break;

    case EIS_OP_PROP_SCORE:
        if (lane->current_proposal >= 0) {
            float score = reg_read(regs, instr->src0);
            lane->proposals.proposals[lane->current_proposal].score = score;
        }
        break;

    case EIS_OP_PROP_END:
        lane->current_proposal = -1;
        break;

    /* === Choose/Commit === */
    case EIS_OP_CHOOSE:
        if (phase_check(&vm->phases, phase_can_choose, "CHOOSE")) {
            /* Simple: choose highest score */
            if (lane->proposals.num_proposals > 0) {
                uint32_t best_idx = 0;
                float best_score = lane->proposals.proposals[0].score;
                for (uint32_t i = 1; i < lane->proposals.num_proposals; i++) {
                    if (lane->proposals.proposals[i].score > best_score) {
                        best_score = lane->proposals.proposals[i].score;
                        best_idx = i;
                    }
                }
                uint32_t choice_id = instr->dst;
                if (choice_id < EIS_MAX_PROPOSALS) {
                    lane->proposals.choices[choice_id] = best_idx;
                    lane->proposals.num_choices++;
                }
            }
        }
        break;

    case EIS_OP_COMMIT:
        if (phase_check(&vm->phases, phase_can_commit, "COMMIT")) {
            uint32_t choice_id = instr->src0;
            if (choice_id < lane->proposals.num_choices) {
                uint32_t prop_idx = lane->proposals.choices[choice_id];
                if (prop_idx < lane->proposals.num_proposals) {
                    EIS_Proposal *prop = &lane->proposals.proposals[prop_idx];
                    /* Apply effects */
                    for (uint32_t i = 0; i < prop->num_effects; i++) {
                        EIS_Effect *eff = &prop->effects[i];
                        switch (eff->type) {
                            case EIS_EFFECT_XFER_F:
                                det_apply_xfer(det, eff->src_node, eff->dst_node, eff->amount);
                                break;
                            case EIS_EFFECT_SET_NODE_FIELD:
                                det_write_node_field(det, eff->target_ref, eff->field_id,
                                                     (float)eff->value);
                                break;
                            default:
                                break;
                        }
                    }
                    prop->committed = true;
                }
            }
        }
        break;

    case EIS_OP_WITNESS:
        if (phase_check(&vm->phases, phase_can_commit, "WITNESS")) {
            uint8_t tok_idx = (instr->dst >= 24) ? instr->dst - 24 : 0;
            uint32_t value = (uint32_t)reg_read(regs, instr->src0);
            regs->tokens[tok_idx] = value;
            /* Could log to witness trace here */
        }
        break;

    /* === Reference Operations === */
    case EIS_OP_MKNODE:
        reg_write_ref(regs, instr->dst, (uint32_t)instr->imm, 0);  /* type 0 = node */
        break;

    case EIS_OP_GETSELF:
        if (lane->lane_type == 0) {  /* Node lane */
            reg_write_ref(regs, instr->dst, regs->self_node, 0);
        } else {  /* Bond lane */
            reg_write_ref(regs, instr->dst, regs->self_bond, 1);
        }
        break;

    /* === Conservation Primitives === */
    case EIS_OP_XFER:
        if (phase_check(&vm->phases, phase_can_commit, "XFER")) {
            float amount = reg_read(regs, instr->dst);
            uint16_t src = (uint16_t)reg_read_ref(regs, instr->src0);
            uint16_t dst = (uint16_t)reg_read_ref(regs, instr->src1);
            EIS_WitnessToken tok = det_apply_xfer(det, src, dst, amount);
            regs->tokens[0] = tok;  /* Result in T0 */
        }
        break;

    case EIS_OP_DIFFUSE:
        if (phase_check(&vm->phases, phase_can_commit, "DIFFUSE")) {
            float sigma = reg_read(regs, instr->dst);
            uint16_t node_a = (uint16_t)reg_read_ref(regs, instr->src0);
            uint16_t node_b = (uint16_t)reg_read_ref(regs, instr->src1);
            if (det && node_a < det->num_nodes && node_b < det->num_nodes) {
                float f_a = det->nodes[node_a].F;
                float f_b = det->nodes[node_b].F;
                float flux = sigma * (f_a - f_b) * 0.5f;
                det->nodes[node_a].F -= flux;
                det->nodes[node_b].F += flux;
            }
        }
        break;

    /* === Branching === */
    case EIS_OP_CALL:
        {
            uint32_t kernel_id = (uint32_t)instr->imm;
            if (kernel_id < vm->num_kernels && vm->kernels[kernel_id]) {
                /* Save return address (simplified - no call stack) */
                /* For now, just run kernel inline */
                /* TODO: Implement proper call stack */
            }
        }
        break;

    case EIS_OP_RET:
        lane->state = EIS_STATE_HALTED;
        break;

    /* === Debug === */
    case EIS_OP_DEBUG:
        lane->state = EIS_STATE_BREAKPOINT;
        break;

    case EIS_OP_TRACE:
        if (vm->trace_execution) {
            printf("TRACE[%u]: imm=%d\n", lane->lane_id, instr->imm);
        }
        break;

    default:
        snprintf(lane->error_msg, sizeof(lane->error_msg),
                 "Unknown opcode: 0x%02X", instr->opcode);
        lane->state = EIS_STATE_ERROR;
        break;
    }

    vm->instructions_executed++;
}

/* ==========================================================================
 * VM LIFECYCLE
 * ========================================================================== */

EIS_VM* eis_vm_create(DETCore *det) {
    EIS_VM *vm = (EIS_VM*)calloc(1, sizeof(EIS_VM));
    if (!vm) return NULL;

    vm->det = det;
    vm->phases.current_phase = EIS_PHASE_IDLE;
    vm->phases.strict = true;

    /* Allocate lane arrays */
    vm->node_lanes = (EIS_Lane*)calloc(DET_MAX_NODES, sizeof(EIS_Lane));
    vm->bond_lanes = (EIS_Lane*)calloc(DET_MAX_BONDS, sizeof(EIS_Lane));

    /* Kernel registry */
    vm->kernels = (const uint8_t**)calloc(256, sizeof(uint8_t*));
    vm->kernel_sizes = (uint32_t*)calloc(256, sizeof(uint32_t));

    return vm;
}

void eis_vm_destroy(EIS_VM *vm) {
    if (!vm) return;

    free(vm->node_lanes);
    free(vm->bond_lanes);
    free(vm->kernels);
    free(vm->kernel_sizes);
    free(vm);
}

EIS_Lane* eis_vm_create_node_lane(EIS_VM *vm, uint16_t node_id,
                                   const uint8_t *program, uint32_t size) {
    if (!vm || node_id >= DET_MAX_NODES) return NULL;

    EIS_Lane *lane = &vm->node_lanes[vm->num_node_lanes];
    memset(lane, 0, sizeof(EIS_Lane));

    lane->lane_id = node_id;
    lane->lane_type = 0;  /* Node */
    lane->regs.self_node = node_id;
    lane->program = program;
    lane->program_size = size;
    lane->state = EIS_STATE_RUNNING;
    lane->current_proposal = -1;

    vm->num_node_lanes++;
    return lane;
}

EIS_Lane* eis_vm_create_bond_lane(EIS_VM *vm, uint16_t node_i, uint16_t node_j,
                                   const uint8_t *program, uint32_t size) {
    if (!vm) return NULL;

    EIS_Lane *lane = &vm->bond_lanes[vm->num_bond_lanes];
    memset(lane, 0, sizeof(EIS_Lane));

    /* Pack bond ref: (i:12, j:12, layer:8) */
    lane->lane_id = ((uint32_t)node_i << 20) | ((uint32_t)node_j << 8);
    lane->lane_type = 1;  /* Bond */
    lane->regs.self_bond = lane->lane_id;
    lane->program = program;
    lane->program_size = size;
    lane->state = EIS_STATE_RUNNING;
    lane->current_proposal = -1;

    vm->num_bond_lanes++;
    return lane;
}

/* ==========================================================================
 * EXECUTION
 * ========================================================================== */

EIS_ExecState eis_vm_step_lane(EIS_VM *vm, EIS_Lane *lane) {
    if (!vm || !lane) return EIS_STATE_ERROR;
    if (lane->state != EIS_STATE_RUNNING) return lane->state;
    if (lane->pc >= lane->program_size) {
        lane->state = EIS_STATE_HALTED;
        return lane->state;
    }

    /* Decode instruction */
    uint32_t consumed;
    EIS_Instruction instr = eis_decode_instruction(lane->program, lane->pc, &consumed);

    /* Trace */
    if (vm->trace_execution) {
        printf("[%u] PC=%04X %s\n", lane->lane_id, lane->pc,
               eis_opcode_name(instr.opcode));
    }

    /* Execute */
    execute_instruction(vm, lane, &instr);
    lane->pc += consumed;

    return lane->state;
}

EIS_ExecState eis_vm_run_lane(EIS_VM *vm, EIS_Lane *lane, uint32_t max_steps) {
    uint32_t steps = 0;
    while (steps < max_steps) {
        EIS_ExecState state = eis_vm_step_lane(vm, lane);
        if (state != EIS_STATE_RUNNING) return state;
        steps++;
    }
    return lane->state;
}

void eis_vm_run_tick(EIS_VM *vm) {
    if (!vm) return;

    /* Begin tick */
    vm->phases.tick++;
    vm->tick++;

    /* Clear proposal buffers */
    for (uint32_t i = 0; i < vm->num_node_lanes; i++) {
        vm->node_lanes[i].proposals.num_proposals = 0;
        vm->node_lanes[i].proposals.num_choices = 0;
        vm->node_lanes[i].pc = 0;
        vm->node_lanes[i].state = EIS_STATE_RUNNING;
    }
    for (uint32_t i = 0; i < vm->num_bond_lanes; i++) {
        vm->bond_lanes[i].proposals.num_proposals = 0;
        vm->bond_lanes[i].proposals.num_choices = 0;
        vm->bond_lanes[i].pc = 0;
        vm->bond_lanes[i].state = EIS_STATE_RUNNING;
    }

    /* Run through phases */
    EIS_Phase phases[] = {EIS_PHASE_READ, EIS_PHASE_PROPOSE, EIS_PHASE_CHOOSE, EIS_PHASE_COMMIT};
    for (int p = 0; p < 4; p++) {
        vm->phases.current_phase = phases[p];

        /* Run node lanes */
        for (uint32_t i = 0; i < vm->num_node_lanes; i++) {
            eis_vm_run_lane(vm, &vm->node_lanes[i], 10000);
            vm->node_lanes[i].pc = 0;
            vm->node_lanes[i].state = EIS_STATE_RUNNING;
        }

        /* Run bond lanes */
        for (uint32_t i = 0; i < vm->num_bond_lanes; i++) {
            eis_vm_run_lane(vm, &vm->bond_lanes[i], 10000);
            vm->bond_lanes[i].pc = 0;
            vm->bond_lanes[i].state = EIS_STATE_RUNNING;
        }
    }

    vm->phases.current_phase = EIS_PHASE_IDLE;
    vm->phases_completed++;
}

uint32_t eis_vm_register_kernel(EIS_VM *vm, const uint8_t *program, uint32_t size) {
    if (!vm || vm->num_kernels >= 256) return 0xFFFFFFFF;

    uint32_t id = vm->num_kernels;
    vm->kernels[id] = program;
    vm->kernel_sizes[id] = size;
    vm->num_kernels++;
    return id;
}

/* ==========================================================================
 * UTILITY FUNCTIONS
 * ========================================================================== */

const char* eis_opcode_name(uint8_t opcode) {
    static const char* names[] = {
        [EIS_OP_NOP] = "NOP",
        [EIS_OP_PHASE] = "PHASE",
        [EIS_OP_HALT] = "HALT",
        [EIS_OP_YIELD] = "YIELD",
        [EIS_OP_LDI] = "LDI",
        [EIS_OP_LDI_EXT] = "LDI_EXT",
        [EIS_OP_LDN] = "LDN",
        [EIS_OP_LDT] = "LDT",
        [EIS_OP_ST_TOK] = "ST_TOK",
        [EIS_OP_ST_NODE] = "ST_NODE",
        [EIS_OP_ADD] = "ADD",
        [EIS_OP_SUB] = "SUB",
        [EIS_OP_MUL] = "MUL",
        [EIS_OP_DIV] = "DIV",
        [EIS_OP_NEG] = "NEG",
        [EIS_OP_ABS] = "ABS",
        [EIS_OP_SQRT] = "SQRT",
        [EIS_OP_SIN] = "SIN",
        [EIS_OP_COS] = "COS",
        [EIS_OP_MIN] = "MIN",
        [EIS_OP_MAX] = "MAX",
        [EIS_OP_RELU] = "RELU",
        [EIS_OP_CMP] = "CMP",
        [EIS_OP_TSET] = "TSET",
        [EIS_OP_TMOV] = "TMOV",
        [EIS_OP_PROP_BEGIN] = "PROP_BEGIN",
        [EIS_OP_PROP_SCORE] = "PROP_SCORE",
        [EIS_OP_PROP_END] = "PROP_END",
        [EIS_OP_CHOOSE] = "CHOOSE",
        [EIS_OP_COMMIT] = "COMMIT",
        [EIS_OP_WITNESS] = "WITNESS",
        [EIS_OP_MKNODE] = "MKNODE",
        [EIS_OP_GETSELF] = "GETSELF",
        [EIS_OP_XFER] = "XFER",
        [EIS_OP_DIFFUSE] = "DIFFUSE",
        [EIS_OP_CALL] = "CALL",
        [EIS_OP_RET] = "RET",
        [EIS_OP_DEBUG] = "DEBUG",
        [EIS_OP_TRACE] = "TRACE",
    };

    if (opcode < sizeof(names)/sizeof(names[0]) && names[opcode]) {
        return names[opcode];
    }
    return "???";
}

const char* eis_phase_name(EIS_Phase phase) {
    static const char* names[] = {
        [EIS_PHASE_IDLE] = "IDLE",
        [EIS_PHASE_READ] = "READ",
        [EIS_PHASE_PROPOSE] = "PROPOSE",
        [EIS_PHASE_CHOOSE] = "CHOOSE",
        [EIS_PHASE_COMMIT] = "COMMIT"
    };
    return (phase < 5) ? names[phase] : "???";
}

const char* eis_token_name(uint32_t token) {
    switch (token) {
        case EIS_TOK_VOID: return "VOID";
        case EIS_TOK_LT: return "LT";
        case EIS_TOK_EQ: return "EQ";
        case EIS_TOK_GT: return "GT";
        case EIS_TOK_XFER_OK: return "XFER_OK";
        case EIS_TOK_XFER_PARTIAL: return "XFER_PARTIAL";
        case EIS_TOK_XFER_BLOCKED: return "XFER_BLOCKED";
        case EIS_TOK_TRUE: return "TRUE";
        case EIS_TOK_FALSE: return "FALSE";
        default: return "???";
    }
}
