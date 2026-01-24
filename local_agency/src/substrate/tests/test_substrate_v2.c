/**
 * EIS Substrate v2 - Test Suite
 * =============================
 *
 * Tests for the DET-aware execution layer with phase-based execution,
 * proposal buffers, and effect application.
 */

#include "../include/eis_substrate_v2.h"
#include "../include/effect_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Test counters */
static int tests_run = 0;
static int tests_passed = 0;

/* Test macros */
#define TEST(name) \
    static void test_##name(void); \
    static void run_test_##name(void) { \
        tests_run++; \
        printf("  [TEST] %s... ", #name); \
        test_##name(); \
        tests_passed++; \
        printf("PASS\n"); \
    } \
    static void test_##name(void)

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            printf("FAIL\n"); \
            printf("    Assertion failed: %s\n", #cond); \
            printf("    At %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_NEAR(a, b, eps) ASSERT(fabsf((a) - (b)) < (eps))

/* ==========================================================================
 * TEST: VM Lifecycle
 * ========================================================================== */

TEST(vm_create_destroy) {
    SubstrateVM* vm = substrate_create();
    ASSERT(vm != NULL);
    ASSERT_EQ(vm->state, SUB_STATE_HALTED);
    ASSERT_EQ(vm->phase, PHASE_READ);
    substrate_destroy(vm);
}

TEST(vm_reset) {
    SubstrateVM* vm = substrate_create();
    vm->tick = 100;
    vm->instructions = 1000;
    substrate_reset(vm);
    ASSERT_EQ(vm->tick, 0);
    ASSERT_EQ(vm->instructions, 0);
    ASSERT_EQ(vm->state, SUB_STATE_RUNNING);
    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Memory Allocation
 * ========================================================================== */

TEST(alloc_nodes) {
    SubstrateVM* vm = substrate_create();
    ASSERT(substrate_alloc_nodes(vm, 10));
    ASSERT(vm->nodes != NULL);
    ASSERT_EQ(vm->nodes->num_nodes, 10);

    /* Check default values */
    ASSERT_NEAR(vm->nodes->a[0], 1.0f, 0.001f);  /* Full agency */
    ASSERT_NEAR(vm->nodes->sigma[0], 1.0f, 0.001f);
    ASSERT_NEAR(vm->nodes->cos_theta[0], 1.0f, 0.001f);
    ASSERT_NEAR(vm->nodes->sin_theta[0], 0.0f, 0.001f);

    substrate_destroy(vm);
}

TEST(alloc_bonds) {
    SubstrateVM* vm = substrate_create();
    ASSERT(substrate_alloc_bonds(vm, 5));
    ASSERT(vm->bonds != NULL);
    ASSERT_EQ(vm->bonds->num_bonds, 5);

    /* Check default values */
    ASSERT_NEAR(vm->bonds->C[0], 0.5f, 0.001f);  /* Default coherence */
    ASSERT_NEAR(vm->bonds->sigma[0], 1.0f, 0.001f);

    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Register Access
 * ========================================================================== */

TEST(scalar_registers) {
    SubstrateVM* vm = substrate_create();

    for (int i = 0; i < SUB_NUM_SCALAR_REGS; i++) {
        substrate_set_scalar(vm, i, (float)i * 1.5f);
    }

    for (int i = 0; i < SUB_NUM_SCALAR_REGS; i++) {
        ASSERT_NEAR(substrate_get_scalar(vm, i), (float)i * 1.5f, 0.001f);
    }

    substrate_destroy(vm);
}

TEST(ref_registers) {
    SubstrateVM* vm = substrate_create();

    substrate_set_ref(vm, 0, REF_MAKE(REF_TYPE_NODE, 42));
    uint32_t ref = substrate_get_ref(vm, 0);
    ASSERT_EQ(REF_TYPE(ref), REF_TYPE_NODE);
    ASSERT_EQ(REF_ID(ref), 42);

    substrate_set_ref(vm, 1, REF_MAKE(REF_TYPE_BOND, 7));
    ref = substrate_get_ref(vm, 1);
    ASSERT_EQ(REF_TYPE(ref), REF_TYPE_BOND);
    ASSERT_EQ(REF_ID(ref), 7);

    substrate_destroy(vm);
}

TEST(token_registers) {
    SubstrateVM* vm = substrate_create();

    substrate_set_token(vm, 0, TOK_OK);
    substrate_set_token(vm, 1, TOK_XFER_OK);
    substrate_set_token(vm, 2, TOK_LT);

    ASSERT_EQ(substrate_get_token(vm, 0), TOK_OK);
    ASSERT_EQ(substrate_get_token(vm, 1), TOK_XFER_OK);
    ASSERT_EQ(substrate_get_token(vm, 2), TOK_LT);

    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Node/Bond Field Access
 * ========================================================================== */

TEST(node_field_access) {
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 5);

    /* Set various fields */
    substrate_node_set_f(vm, 0, NODE_FIELD_F, 100.0f);
    substrate_node_set_f(vm, 0, NODE_FIELD_Q, 0.5f);
    substrate_node_set_f(vm, 0, NODE_FIELD_A, 0.8f);

    /* Read back */
    ASSERT_NEAR(substrate_node_get_f(vm, 0, NODE_FIELD_F), 100.0f, 0.001f);
    ASSERT_NEAR(substrate_node_get_f(vm, 0, NODE_FIELD_Q), 0.5f, 0.001f);
    ASSERT_NEAR(substrate_node_get_f(vm, 0, NODE_FIELD_A), 0.8f, 0.001f);

    /* Agency should be clamped to [0,1] */
    substrate_node_set_f(vm, 0, NODE_FIELD_A, 1.5f);
    ASSERT_NEAR(substrate_node_get_f(vm, 0, NODE_FIELD_A), 1.0f, 0.001f);

    substrate_node_set_f(vm, 0, NODE_FIELD_A, -0.5f);
    ASSERT_NEAR(substrate_node_get_f(vm, 0, NODE_FIELD_A), 0.0f, 0.001f);

    substrate_destroy(vm);
}

TEST(bond_field_access) {
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 5);
    substrate_alloc_bonds(vm, 3);

    /* Set bond endpoints */
    substrate_bond_set_i(vm, 0, BOND_FIELD_NODE_I, 0);
    substrate_bond_set_i(vm, 0, BOND_FIELD_NODE_J, 1);

    /* Set coherence */
    substrate_bond_set_f(vm, 0, BOND_FIELD_C, 0.9f);

    /* Read back */
    ASSERT_EQ(substrate_bond_get_i(vm, 0, BOND_FIELD_NODE_I), 0);
    ASSERT_EQ(substrate_bond_get_i(vm, 0, BOND_FIELD_NODE_J), 1);
    ASSERT_NEAR(substrate_bond_get_f(vm, 0, BOND_FIELD_C), 0.9f, 0.001f);

    /* Coherence should be clamped to [0,1] */
    substrate_bond_set_f(vm, 0, BOND_FIELD_C, 1.5f);
    ASSERT_NEAR(substrate_bond_get_f(vm, 0, BOND_FIELD_C), 1.0f, 0.001f);

    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Instruction Encode/Decode
 * ========================================================================== */

TEST(instruction_encode_decode) {
    SubstrateInstr instr = {0};
    instr.opcode = OP_ADD;
    instr.dst = 3;
    instr.src0 = 5;
    instr.src1 = 7;
    instr.imm = 0;
    instr.has_ext = false;

    uint8_t buffer[8];
    uint32_t size = substrate_encode(&instr, buffer);
    ASSERT_EQ(size, 4);

    uint32_t consumed;
    SubstrateInstr decoded = substrate_decode(buffer, 0, &consumed);
    ASSERT_EQ(consumed, 4);
    ASSERT_EQ(decoded.opcode, OP_ADD);
    ASSERT_EQ(decoded.dst, 3);
    ASSERT_EQ(decoded.src0, 5);
    ASSERT_EQ(decoded.src1, 7);
}

TEST(instruction_with_imm) {
    SubstrateInstr instr = {0};
    instr.opcode = OP_LDI;
    instr.dst = 2;
    instr.imm = -100;  /* Negative immediate */
    instr.has_ext = false;

    uint8_t buffer[8];
    uint32_t size = substrate_encode(&instr, buffer);
    ASSERT_EQ(size, 4);

    uint32_t consumed;
    SubstrateInstr decoded = substrate_decode(buffer, 0, &consumed);
    ASSERT_EQ(decoded.opcode, OP_LDI);
    ASSERT_EQ(decoded.dst, 2);
    ASSERT_EQ(decoded.imm, -100);
}

/* ==========================================================================
 * TEST: Simple Program Execution
 * ========================================================================== */

/* Helper to build program */
static uint8_t* build_program(SubstrateInstr* instrs, int count, uint32_t* out_size) {
    uint8_t* program = malloc(count * 8);  /* Max 8 bytes per instruction */
    uint32_t offset = 0;

    for (int i = 0; i < count; i++) {
        offset += substrate_encode(&instrs[i], program + offset);
    }

    *out_size = offset;
    return program;
}

TEST(execute_arithmetic) {
    SubstrateVM* vm = substrate_create();

    /* Program: R0 = 10, R1 = 3, R2 = R0 + R1, HALT */
    SubstrateInstr instrs[] = {
        { .opcode = OP_LDI, .dst = 0, .imm = 10 },
        { .opcode = OP_LDI, .dst = 1, .imm = 3 },
        { .opcode = OP_ADD, .dst = 2, .src0 = 0, .src1 = 1 },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 4, &size);

    substrate_load_program(vm, program, size);
    substrate_run(vm, 100);

    ASSERT_EQ(vm->state, SUB_STATE_HALTED);
    ASSERT_NEAR(substrate_get_scalar(vm, 0), 10.0f, 0.001f);
    ASSERT_NEAR(substrate_get_scalar(vm, 1), 3.0f, 0.001f);
    ASSERT_NEAR(substrate_get_scalar(vm, 2), 13.0f, 0.001f);

    free(program);
    substrate_destroy(vm);
}

TEST(execute_comparison) {
    SubstrateVM* vm = substrate_create();

    /* Program: R0 = 5, R1 = 10, CMP T0 = (R0 vs R1), HALT */
    SubstrateInstr instrs[] = {
        { .opcode = OP_LDI, .dst = 0, .imm = 5 },
        { .opcode = OP_LDI, .dst = 1, .imm = 10 },
        { .opcode = OP_CMP, .dst = 0, .src0 = 0, .src1 = 1 },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 4, &size);

    substrate_load_program(vm, program, size);
    substrate_run(vm, 100);

    ASSERT_EQ(vm->state, SUB_STATE_HALTED);
    ASSERT_EQ(substrate_get_token(vm, 0), TOK_LT);  /* 5 < 10 */

    free(program);
    substrate_destroy(vm);
}

TEST(execute_phases) {
    SubstrateVM* vm = substrate_create();

    /* Program: PHASE.R, PHASE.P, PHASE.C, PHASE.X, HALT */
    SubstrateInstr instrs[] = {
        { .opcode = OP_PHASE_R },
        { .opcode = OP_PHASE_P },
        { .opcode = OP_PHASE_C },
        { .opcode = OP_PHASE_X },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 5, &size);

    substrate_load_program(vm, program, size);

    /* Execute step by step, check phases */
    substrate_step(vm);
    ASSERT_EQ(vm->phase, PHASE_READ);

    substrate_step(vm);
    ASSERT_EQ(vm->phase, PHASE_PROPOSE);

    substrate_step(vm);
    ASSERT_EQ(vm->phase, PHASE_CHOOSE);

    substrate_step(vm);
    ASSERT_EQ(vm->phase, PHASE_COMMIT);

    substrate_step(vm);
    ASSERT_EQ(vm->state, SUB_STATE_HALTED);

    free(program);
    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Proposals
 * ========================================================================== */

TEST(proposal_create) {
    SubstrateVM* vm = substrate_create();

    uint32_t idx0 = substrate_prop_new(vm);
    uint32_t idx1 = substrate_prop_new(vm);
    uint32_t idx2 = substrate_prop_new(vm);

    ASSERT_EQ(idx0, 0);
    ASSERT_EQ(idx1, 1);
    ASSERT_EQ(idx2, 2);
    ASSERT_EQ(vm->prop_buf.count, 3);

    substrate_destroy(vm);
}

TEST(proposal_score) {
    SubstrateVM* vm = substrate_create();

    uint32_t idx = substrate_prop_new(vm);
    substrate_prop_score(vm, idx, 0.75f);

    ASSERT_NEAR(vm->prop_buf.proposals[idx].score, 0.75f, 0.001f);

    substrate_destroy(vm);
}

TEST(proposal_choose) {
    SubstrateVM* vm = substrate_create();

    /* Create proposals with different scores */
    uint32_t p0 = substrate_prop_new(vm);
    uint32_t p1 = substrate_prop_new(vm);
    uint32_t p2 = substrate_prop_new(vm);

    substrate_prop_score(vm, p0, 0.1f);
    substrate_prop_score(vm, p1, 0.8f);  /* Highest score */
    substrate_prop_score(vm, p2, 0.1f);

    /* High decisiveness should favor highest score */
    uint32_t choice = substrate_choose(vm, 0.99f);
    ASSERT_EQ(choice, 1);

    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Effects
 * ========================================================================== */

TEST(effect_xfer_f) {
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 5);

    /* Set initial resources */
    substrate_node_set_f(vm, 0, NODE_FIELD_F, 100.0f);
    substrate_node_set_f(vm, 1, NODE_FIELD_F, 50.0f);

    /* Apply transfer effect */
    uint32_t args[3] = {
        0,  /* src node */
        1,  /* dst node */
        effect_pack_float(30.0f)  /* amount */
    };
    uint32_t result = effect_apply(vm, EFFECT_XFER_F, args);

    ASSERT_EQ(result, TOK_XFER_OK);
    ASSERT_NEAR(substrate_node_get_f(vm, 0, NODE_FIELD_F), 70.0f, 0.001f);
    ASSERT_NEAR(substrate_node_get_f(vm, 1, NODE_FIELD_F), 80.0f, 0.001f);

    substrate_destroy(vm);
}

TEST(effect_diffuse) {
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 5);
    substrate_alloc_bonds(vm, 2);

    /* Set bond endpoints */
    substrate_bond_set_i(vm, 0, BOND_FIELD_NODE_I, 0);
    substrate_bond_set_i(vm, 0, BOND_FIELD_NODE_J, 1);

    /* Set initial resources */
    substrate_node_set_f(vm, 0, NODE_FIELD_F, 100.0f);
    substrate_node_set_f(vm, 1, NODE_FIELD_F, 50.0f);

    /* Apply diffuse effect (flux from node_i to node_j) */
    uint32_t args[2] = {
        0,  /* bond id */
        effect_pack_float(20.0f)  /* delta */
    };
    uint32_t result = effect_apply(vm, EFFECT_DIFFUSE, args);

    ASSERT_EQ(result, TOK_DIFFUSE_OK);
    ASSERT_NEAR(substrate_node_get_f(vm, 0, NODE_FIELD_F), 80.0f, 0.001f);
    ASSERT_NEAR(substrate_node_get_f(vm, 1, NODE_FIELD_F), 70.0f, 0.001f);

    substrate_destroy(vm);
}

TEST(effect_validation) {
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 5);

    /* Valid node ID */
    uint32_t args_valid[2] = { 3, effect_pack_float(10.0f) };
    ASSERT(effect_validate(vm, EFFECT_SET_F, args_valid));

    /* Invalid node ID (out of range) */
    uint32_t args_invalid[2] = { 100, effect_pack_float(10.0f) };
    ASSERT(!effect_validate(vm, EFFECT_SET_F, args_invalid));

    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Full Proposal-Choose-Commit Cycle
 * ========================================================================== */

TEST(full_proposal_cycle) {
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 3);

    /* Setup: node 0 has 100, node 1 has 0 */
    substrate_node_set_f(vm, 0, NODE_FIELD_F, 100.0f);
    substrate_node_set_f(vm, 1, NODE_FIELD_F, 0.0f);

    /* Create proposal: transfer 25 from node 0 to node 1 */
    uint32_t prop_idx = substrate_prop_new(vm);
    substrate_prop_score(vm, prop_idx, 1.0f);

    uint32_t args[3] = {
        0,  /* src */
        1,  /* dst */
        effect_pack_float(25.0f)
    };
    substrate_prop_effect(vm, prop_idx, EFFECT_XFER_F, args, 3);

    /* Choose (with high decisiveness) */
    uint32_t chosen = substrate_choose(vm, 1.0f);
    ASSERT_EQ(chosen, 0);

    /* Commit */
    uint32_t result = substrate_commit(vm);
    ASSERT_EQ(result, TOK_XFER_OK);

    /* Verify transfer happened */
    ASSERT_NEAR(substrate_node_get_f(vm, 0, NODE_FIELD_F), 75.0f, 0.001f);
    ASSERT_NEAR(substrate_node_get_f(vm, 1, NODE_FIELD_F), 25.0f, 0.001f);

    /* Proposal buffer should be cleared */
    ASSERT_EQ(vm->prop_buf.count, 0);

    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: I/O Channels
 * ========================================================================== */

TEST(io_inject_read) {
    SubstrateVM* vm = substrate_create();

    substrate_io_inject(vm, 0, 42.5f);
    substrate_io_inject(vm, 1, -10.0f);

    ASSERT_NEAR(substrate_io_read(vm, 0), 42.5f, 0.001f);
    ASSERT_NEAR(substrate_io_read(vm, 1), -10.0f, 0.001f);

    substrate_destroy(vm);
}

static float test_read_fn(void* ctx) {
    float* value = (float*)ctx;
    return *value;
}

static float test_write_value = 0.0f;
static void test_write_fn(void* ctx, float value) {
    (void)ctx;
    test_write_value = value;
}

TEST(io_callbacks) {
    SubstrateVM* vm = substrate_create();

    float read_value = 123.0f;
    substrate_io_configure(vm, 0, test_read_fn, test_write_fn, &read_value);

    /* Program with proper phases:
     * READ: IN(0) -> R0
     * PROPOSE: compute (arithmetic allowed anywhere)
     * COMMIT: OUT(0) = R0
     */
    SubstrateInstr instrs[] = {
        { .opcode = OP_PHASE_R },                        /* Enter READ phase */
        { .opcode = OP_IN, .dst = 0, .imm = 0 },         /* R0 = io[0] = 123 */
        { .opcode = OP_LDI, .dst = 1, .imm = 2 },        /* R1 = 2 */
        { .opcode = OP_MUL, .dst = 0, .src0 = 0, .src1 = 1 },  /* R0 = 123 * 2 = 246 */
        { .opcode = OP_PHASE_X },                        /* Enter COMMIT phase */
        { .opcode = OP_OUT, .src0 = 0, .imm = 0 },       /* io[0] = R0 */
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 7, &size);

    substrate_load_program(vm, program, size);
    substrate_run(vm, 100);

    ASSERT_NEAR(test_write_value, 246.0f, 0.001f);  /* 123 * 2 */

    free(program);
    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Boundary Buffers
 * ========================================================================== */

TEST(boundary_buffer) {
    SubstrateVM* vm = substrate_create();

    uint32_t buf_id = substrate_add_boundary(vm, 100, false);
    ASSERT_EQ(buf_id, 0);

    /* Emit bytes */
    substrate_boundary_emit_byte(vm, buf_id, 'H');
    substrate_boundary_emit_byte(vm, buf_id, 'i');
    substrate_boundary_emit_byte(vm, buf_id, '!');

    /* Read back */
    uint8_t out[10];
    uint32_t len = substrate_boundary_read(vm, buf_id, out, 10);
    ASSERT_EQ(len, 3);
    ASSERT_EQ(out[0], 'H');
    ASSERT_EQ(out[1], 'i');
    ASSERT_EQ(out[2], '!');

    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Debug/Utility Functions
 * ========================================================================== */

TEST(opcode_names) {
    ASSERT(strcmp(substrate_opcode_name(OP_NOP), "NOP") == 0);
    ASSERT(strcmp(substrate_opcode_name(OP_ADD), "ADD") == 0);
    ASSERT(strcmp(substrate_opcode_name(OP_PROP_NEW), "PROP.NEW") == 0);
    ASSERT(strcmp(substrate_opcode_name(OP_COMMIT), "COMMIT") == 0);
}

TEST(token_names) {
    ASSERT(strcmp(substrate_token_name(TOK_OK), "OK") == 0);
    ASSERT(strcmp(substrate_token_name(TOK_LT), "LT") == 0);
    ASSERT(strcmp(substrate_token_name(TOK_XFER_OK), "XFER_OK") == 0);
}

TEST(phase_names) {
    ASSERT(strcmp(substrate_phase_name(PHASE_READ), "READ") == 0);
    ASSERT(strcmp(substrate_phase_name(PHASE_PROPOSE), "PROPOSE") == 0);
    ASSERT(strcmp(substrate_phase_name(PHASE_CHOOSE), "CHOOSE") == 0);
    ASSERT(strcmp(substrate_phase_name(PHASE_COMMIT), "COMMIT") == 0);
}

TEST(effect_names) {
    ASSERT(strcmp(effect_get_name(EFFECT_NONE), "NONE") == 0);
    ASSERT(strcmp(effect_get_name(EFFECT_XFER_F), "XFER_F") == 0);
    ASSERT(strcmp(effect_get_name(EFFECT_DIFFUSE), "DIFFUSE") == 0);
}

TEST(effect_descriptors) {
    const EffectDescriptor* desc = effect_get_descriptor(EFFECT_XFER_F);
    ASSERT(desc != NULL);
    ASSERT_EQ(desc->arg_count, 3);
    ASSERT(desc->antisymmetric);
    ASSERT(desc->node_effect);
    ASSERT(!desc->bond_effect);

    desc = effect_get_descriptor(EFFECT_DIFFUSE);
    ASSERT(desc != NULL);
    ASSERT_EQ(desc->arg_count, 2);
    ASSERT(!desc->antisymmetric);
    ASSERT(desc->node_effect);
    ASSERT(desc->bond_effect);
}

/* ==========================================================================
 * TEST: Statistics
 * ========================================================================== */

TEST(stats) {
    SubstrateVM* vm = substrate_create();

    SubstrateInstr instrs[] = {
        { .opcode = OP_NOP },
        { .opcode = OP_NOP },
        { .opcode = OP_NOP },
        { .opcode = OP_TICK },
        { .opcode = OP_NOP },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 6, &size);

    substrate_load_program(vm, program, size);
    substrate_run(vm, 100);

    uint64_t tick, instructions;
    substrate_get_stats(vm, &tick, &instructions);

    ASSERT_EQ(tick, 1);
    ASSERT_EQ(instructions, 6);

    free(program);
    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Load Node/Bond Operations
 * ========================================================================== */

TEST(load_node_via_instruction) {
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 5);

    /* Set a node's F field */
    substrate_node_set_f(vm, 2, NODE_FIELD_F, 77.5f);

    /* Program: H0 = node ref 2, R0 = LDN(H0, F), HALT */
    vm->regs.refs[0] = REF_MAKE(REF_TYPE_NODE, 2);

    SubstrateInstr instrs[] = {
        { .opcode = OP_LDN, .dst = 0, .src0 = 0, .src1 = NODE_FIELD_F },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 2, &size);

    substrate_load_program(vm, program, size);
    substrate_run(vm, 100);

    ASSERT_NEAR(substrate_get_scalar(vm, 0), 77.5f, 0.001f);

    free(program);
    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Phase Legality Enforcement
 * ========================================================================== */

TEST(phase_violation_load_in_commit) {
    /* LDN should fail in COMMIT phase */
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 2);

    vm->regs.refs[0] = REF_MAKE(REF_TYPE_NODE, 0);

    SubstrateInstr instrs[] = {
        { .opcode = OP_PHASE_X },  /* Enter COMMIT phase */
        { .opcode = OP_LDN, .dst = 0, .src0 = 0, .src1 = NODE_FIELD_F },  /* Should fail */
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 3, &size);

    substrate_load_program(vm, program, size);
    substrate_run(vm, 100);

    /* Should be in error state due to phase violation */
    ASSERT_EQ(vm->state, SUB_STATE_ERROR);
    ASSERT_EQ(substrate_get_token(vm, 0), TOK_PHASE_VIOLATION);

    free(program);
    substrate_destroy(vm);
}

TEST(phase_violation_store_in_read) {
    /* STN should fail in READ phase */
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 2);

    vm->regs.refs[0] = REF_MAKE(REF_TYPE_NODE, 0);

    SubstrateInstr instrs[] = {
        { .opcode = OP_PHASE_R },  /* Enter READ phase (default, but explicit) */
        { .opcode = OP_STN, .dst = 0, .src0 = NODE_FIELD_F, .src1 = 1 },  /* Should fail */
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 3, &size);

    substrate_load_program(vm, program, size);
    substrate_run(vm, 100);

    ASSERT_EQ(vm->state, SUB_STATE_ERROR);

    free(program);
    substrate_destroy(vm);
}

TEST(phase_violation_proposal_in_read) {
    /* PROP_NEW should fail in READ phase */
    SubstrateVM* vm = substrate_create();

    SubstrateInstr instrs[] = {
        { .opcode = OP_PHASE_R },
        { .opcode = OP_PROP_NEW, .dst = 0 },  /* Should fail - only in PROPOSE */
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 3, &size);

    substrate_load_program(vm, program, size);
    substrate_run(vm, 100);

    ASSERT_EQ(vm->state, SUB_STATE_ERROR);

    free(program);
    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Deterministic RNG from Trace
 * ========================================================================== */

TEST(deterministic_rng_reproducibility) {
    /* Same trace state should produce same random sequence */
    SubstrateVM* vm1 = substrate_create();
    SubstrateVM* vm2 = substrate_create();

    substrate_alloc_nodes(vm1, 4);
    substrate_alloc_nodes(vm2, 4);

    /* Set identical trace state */
    vm1->nodes->r[0] = 12345;
    vm1->nodes->k[0] = 100;
    vm2->nodes->r[0] = 12345;
    vm2->nodes->k[0] = 100;

    substrate_set_lane(vm1, 0);
    substrate_set_lane(vm2, 0);

    /* Program: Generate 3 random numbers */
    SubstrateInstr instrs[] = {
        { .opcode = OP_TICK },       /* Reset seed from trace */
        { .opcode = OP_RAND, .dst = 0 },
        { .opcode = OP_RAND, .dst = 1 },
        { .opcode = OP_RAND, .dst = 2 },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 5, &size);

    substrate_load_program(vm1, program, size);
    substrate_load_program(vm2, program, size);

    substrate_run(vm1, 100);
    substrate_run(vm2, 100);

    /* Both VMs should produce identical random sequence */
    ASSERT_NEAR(substrate_get_scalar(vm1, 0), substrate_get_scalar(vm2, 0), 0.0001f);
    ASSERT_NEAR(substrate_get_scalar(vm1, 1), substrate_get_scalar(vm2, 1), 0.0001f);
    ASSERT_NEAR(substrate_get_scalar(vm1, 2), substrate_get_scalar(vm2, 2), 0.0001f);

    free(program);
    substrate_destroy(vm1);
    substrate_destroy(vm2);
}

TEST(different_lanes_different_rng) {
    /* Different lanes should produce different sequences */
    SubstrateVM* vm1 = substrate_create();
    SubstrateVM* vm2 = substrate_create();

    substrate_alloc_nodes(vm1, 4);
    substrate_alloc_nodes(vm2, 4);

    /* Same trace but different lane IDs */
    vm1->nodes->r[0] = vm1->nodes->r[1] = 12345;
    vm2->nodes->r[0] = vm2->nodes->r[1] = 12345;

    substrate_set_lane(vm1, 0);
    substrate_set_lane(vm2, 1);  /* Different lane */

    SubstrateInstr instrs[] = {
        { .opcode = OP_TICK },
        { .opcode = OP_RAND, .dst = 0 },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 3, &size);

    substrate_load_program(vm1, program, size);
    substrate_load_program(vm2, program, size);

    substrate_run(vm1, 100);
    substrate_run(vm2, 100);

    /* Different lanes should produce different values */
    float r1 = substrate_get_scalar(vm1, 0);
    float r2 = substrate_get_scalar(vm2, 0);
    ASSERT(fabsf(r1 - r2) > 0.001f);  /* Should be different */

    free(program);
    substrate_destroy(vm1);
    substrate_destroy(vm2);
}

/* ==========================================================================
 * TEST: Bond Ownership (Deduplication)
 * ========================================================================== */

TEST(bond_ownership_node_mode) {
    /* In NODE_LANE mode, only min(src, dst) lane should apply XFER_F */
    SubstrateVM* vm = substrate_create();
    substrate_alloc_nodes(vm, 4);

    /* Give node 0 some resource */
    vm->nodes->F[0] = 100.0f;

    /* Set to node-lane ownership mode */
    substrate_set_lane_mode(vm, LANE_OWNER_NODE);
    substrate_set_lane(vm, 1);  /* NOT the owner for transfer 0->2 (min=0) */

    /* Try to apply XFER_F effect directly */
    uint32_t args[3] = { 0, 2, effect_pack_float(10.0f) };  /* src=0, dst=2, amount=10 */
    uint32_t result = effect_apply(vm, EFFECT_XFER_F, args);

    /* Should be silently skipped (returns OK but no change) */
    ASSERT_EQ(result, TOK_OK);
    ASSERT_NEAR(vm->nodes->F[0], 100.0f, 0.001f);  /* Unchanged */

    /* Now try from correct owner (lane 0) */
    substrate_set_lane(vm, 0);
    result = effect_apply(vm, EFFECT_XFER_F, args);

    ASSERT_EQ(result, TOK_XFER_OK);
    ASSERT_NEAR(vm->nodes->F[0], 90.0f, 0.001f);  /* Reduced */
    ASSERT_NEAR(vm->nodes->F[2], 10.0f, 0.001f);  /* Increased */

    substrate_destroy(vm);
}

/* ==========================================================================
 * TEST: Predecoding
 * ========================================================================== */

TEST(predecode_program) {
    SubstrateVM* vm = substrate_create();

    SubstrateInstr instrs[] = {
        { .opcode = OP_LDI, .dst = 0, .imm = 10 },
        { .opcode = OP_LDI, .dst = 1, .imm = 20 },
        { .opcode = OP_ADD, .dst = 2, .src0 = 0, .src1 = 1 },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 4, &size);

    substrate_load_program(vm, program, size);

    /* Predecode the program */
    bool success = substrate_predecode(vm);
    ASSERT(success);
    ASSERT(vm->predecoded != NULL);
    ASSERT_EQ(vm->predecoded->count, 4);

    /* Run using predecoded path */
    substrate_run(vm, 100);

    ASSERT_NEAR(substrate_get_scalar(vm, 2), 30.0f, 0.001f);

    free(program);
    substrate_destroy(vm);
}

TEST(predecode_vs_raw_equivalence) {
    /* Predecoded and raw execution should produce same result */
    SubstrateVM* vm1 = substrate_create();
    SubstrateVM* vm2 = substrate_create();

    SubstrateInstr instrs[] = {
        { .opcode = OP_LDI, .dst = 0, .imm = 42 },
        { .opcode = OP_LDI, .dst = 1, .imm = 8 },
        { .opcode = OP_MUL, .dst = 2, .src0 = 0, .src1 = 1 },
        { .opcode = OP_LDI, .dst = 3, .imm = 6 },
        { .opcode = OP_ADD, .dst = 4, .src0 = 2, .src1 = 3 },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 6, &size);

    substrate_load_program(vm1, program, size);
    substrate_load_program(vm2, program, size);

    /* vm1: raw execution */
    substrate_use_predecoded(vm1, false);
    substrate_run(vm1, 100);

    /* vm2: predecoded execution */
    substrate_predecode(vm2);
    substrate_run(vm2, 100);

    /* Both should produce same result: 42*8+6 = 342 */
    ASSERT_NEAR(substrate_get_scalar(vm1, 4), 342.0f, 0.001f);
    ASSERT_NEAR(substrate_get_scalar(vm2, 4), 342.0f, 0.001f);

    free(program);
    substrate_destroy(vm1);
    substrate_destroy(vm2);
}

/* ==========================================================================
 * TEST: Threaded Dispatch (substrate_run_fast)
 * ========================================================================== */

TEST(threaded_dispatch_basic) {
    SubstrateVM* vm = substrate_create();

    SubstrateInstr instrs[] = {
        { .opcode = OP_LDI, .dst = 0, .imm = 100 },
        { .opcode = OP_LDI, .dst = 1, .imm = 23 },
        { .opcode = OP_SUB, .dst = 2, .src0 = 0, .src1 = 1 },
        { .opcode = OP_HALT },
    };

    uint32_t size;
    uint8_t* program = build_program(instrs, 4, &size);

    substrate_load_program(vm, program, size);
    substrate_predecode(vm);

    /* Use fast threaded dispatch */
    SubstrateState state = substrate_run_fast(vm, 100);

    ASSERT_EQ(state, SUB_STATE_HALTED);
    ASSERT_NEAR(substrate_get_scalar(vm, 2), 77.0f, 0.001f);

    free(program);
    substrate_destroy(vm);
}

/* ==========================================================================
 * MAIN
 * ========================================================================== */

int main(void) {
    printf("EIS Substrate v2 - Test Suite\n");
    printf("==============================\n\n");

    printf("VM Lifecycle:\n");
    run_test_vm_create_destroy();
    run_test_vm_reset();

    printf("\nMemory Allocation:\n");
    run_test_alloc_nodes();
    run_test_alloc_bonds();

    printf("\nRegister Access:\n");
    run_test_scalar_registers();
    run_test_ref_registers();
    run_test_token_registers();

    printf("\nNode/Bond Field Access:\n");
    run_test_node_field_access();
    run_test_bond_field_access();

    printf("\nInstruction Encode/Decode:\n");
    run_test_instruction_encode_decode();
    run_test_instruction_with_imm();

    printf("\nProgram Execution:\n");
    run_test_execute_arithmetic();
    run_test_execute_comparison();
    run_test_execute_phases();

    printf("\nProposals:\n");
    run_test_proposal_create();
    run_test_proposal_score();
    run_test_proposal_choose();

    printf("\nEffects:\n");
    run_test_effect_xfer_f();
    run_test_effect_diffuse();
    run_test_effect_validation();
    run_test_full_proposal_cycle();

    printf("\nI/O:\n");
    run_test_io_inject_read();
    run_test_io_callbacks();
    run_test_boundary_buffer();

    printf("\nDebug/Utility:\n");
    run_test_opcode_names();
    run_test_token_names();
    run_test_phase_names();
    run_test_effect_names();
    run_test_effect_descriptors();

    printf("\nStatistics:\n");
    run_test_stats();

    printf("\nLoad Operations:\n");
    run_test_load_node_via_instruction();

    printf("\nPhase Legality:\n");
    run_test_phase_violation_load_in_commit();
    run_test_phase_violation_store_in_read();
    run_test_phase_violation_proposal_in_read();

    printf("\nDeterministic RNG:\n");
    run_test_deterministic_rng_reproducibility();
    run_test_different_lanes_different_rng();

    printf("\nBond Ownership:\n");
    run_test_bond_ownership_node_mode();

    printf("\nPredecoding:\n");
    run_test_predecode_program();
    run_test_predecode_vs_raw_equivalence();

    printf("\nThreaded Dispatch:\n");
    run_test_threaded_dispatch_basic();

    printf("\n==============================\n");
    printf("Tests: %d/%d passed\n", tests_passed, tests_run);

    if (tests_passed == tests_run) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("Some tests FAILED!\n");
        return 1;
    }
}
