/**
 * EIS VM Test Suite
 * =================
 *
 * Tests for the native C implementation of the Existence Instruction Set VM.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "../include/eis_vm.h"
#include "../include/det_core.h"

/* ==========================================================================
 * TEST UTILITIES
 * ========================================================================== */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(expr, msg) do { \
    if (!(expr)) { \
        printf("  FAIL: %s\n", msg); \
        return 0; \
    } \
} while(0)

#define RUN_TEST(test_func) do { \
    tests_run++; \
    printf("Running %s... ", #test_func); \
    if (test_func()) { \
        tests_passed++; \
        printf("PASS\n"); \
    } else { \
        printf("FAILED\n"); \
    } \
} while(0)

/* Helper: encode a simple instruction */
static void encode_simple(uint8_t *out, uint8_t opcode, uint8_t dst,
                          uint8_t src0, uint8_t src1, int16_t imm) {
    uint32_t imm_bits = imm & 0x1FF;
    uint32_t word = ((uint32_t)(opcode & 0xFF) << 24) |
                    ((uint32_t)(dst & 0x1F) << 19) |
                    ((uint32_t)(src0 & 0x1F) << 14) |
                    ((uint32_t)(src1 & 0x1F) << 9) |
                    (imm_bits & 0x1FF);
    out[0] = (word >> 24) & 0xFF;
    out[1] = (word >> 16) & 0xFF;
    out[2] = (word >> 8) & 0xFF;
    out[3] = word & 0xFF;
}

/* ==========================================================================
 * INSTRUCTION ENCODING/DECODING TESTS
 * ========================================================================== */

static int test_decode_nop(void) {
    uint8_t program[4] = {0};
    encode_simple(program, EIS_OP_NOP, 0, 0, 0, 0);

    uint32_t consumed;
    EIS_Instruction instr = eis_decode_instruction(program, 0, &consumed);

    TEST_ASSERT(instr.opcode == EIS_OP_NOP, "opcode should be NOP");
    TEST_ASSERT(consumed == 4, "consumed should be 4");
    return 1;
}

static int test_decode_ldi(void) {
    uint8_t program[4] = {0};
    encode_simple(program, EIS_OP_LDI, 5, 0, 0, 42);

    uint32_t consumed;
    EIS_Instruction instr = eis_decode_instruction(program, 0, &consumed);

    TEST_ASSERT(instr.opcode == EIS_OP_LDI, "opcode should be LDI");
    TEST_ASSERT(instr.dst == 5, "dst should be 5");
    TEST_ASSERT(instr.imm == 42, "imm should be 42");
    return 1;
}

static int test_decode_negative_imm(void) {
    uint8_t program[4] = {0};
    encode_simple(program, EIS_OP_LDI, 3, 0, 0, -10);

    uint32_t consumed;
    EIS_Instruction instr = eis_decode_instruction(program, 0, &consumed);

    TEST_ASSERT(instr.opcode == EIS_OP_LDI, "opcode should be LDI");
    TEST_ASSERT(instr.dst == 3, "dst should be 3");
    TEST_ASSERT(instr.imm == -10, "imm should be -10");
    return 1;
}

static int test_encode_roundtrip(void) {
    EIS_Instruction original = {
        .opcode = EIS_OP_ADD,
        .dst = 7,
        .src0 = 3,
        .src1 = 5,
        .imm = 0,
        .has_ext = false
    };

    uint8_t buffer[8];
    uint32_t encoded = eis_encode_instruction(&original, buffer);
    TEST_ASSERT(encoded == 4, "should encode to 4 bytes");

    uint32_t consumed;
    EIS_Instruction decoded = eis_decode_instruction(buffer, 0, &consumed);

    TEST_ASSERT(decoded.opcode == original.opcode, "opcode roundtrip");
    TEST_ASSERT(decoded.dst == original.dst, "dst roundtrip");
    TEST_ASSERT(decoded.src0 == original.src0, "src0 roundtrip");
    TEST_ASSERT(decoded.src1 == original.src1, "src1 roundtrip");
    return 1;
}

/* ==========================================================================
 * VM LIFECYCLE TESTS
 * ========================================================================== */

static int test_vm_create_destroy(void) {
    DETCore *core = det_core_create();
    TEST_ASSERT(core != NULL, "core should be created");

    EIS_VM *vm = eis_vm_create(core);
    TEST_ASSERT(vm != NULL, "vm should be created");
    TEST_ASSERT(vm->det == core, "vm->det should point to core");
    TEST_ASSERT(vm->phases.current_phase == EIS_PHASE_IDLE, "initial phase should be IDLE");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

static int test_vm_create_lane(void) {
    DETCore *core = det_core_create();
    EIS_VM *vm = eis_vm_create(core);

    uint8_t program[8];
    encode_simple(program, EIS_OP_NOP, 0, 0, 0, 0);
    encode_simple(program + 4, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, 0, program, 8);
    TEST_ASSERT(lane != NULL, "lane should be created");
    TEST_ASSERT(lane->lane_type == 0, "lane type should be node (0)");
    TEST_ASSERT(lane->regs.self_node == 0, "self_node should be 0");
    TEST_ASSERT(lane->state == EIS_STATE_RUNNING, "initial state should be RUNNING");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * INSTRUCTION EXECUTION TESTS
 * ========================================================================== */

static int test_exec_ldi(void) {
    DETCore *core = det_core_create();
    EIS_VM *vm = eis_vm_create(core);

    /* LDI R5, 100 then HALT */
    uint8_t program[8];
    encode_simple(program, EIS_OP_LDI, 5, 0, 0, 100);
    encode_simple(program + 4, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, 0, program, 8);

    EIS_ExecState state = eis_vm_step_lane(vm, lane);
    TEST_ASSERT(state == EIS_STATE_RUNNING, "should still be running after LDI");
    TEST_ASSERT(fabsf(lane->regs.scalars[5] - 100.0f) < 0.001f, "R5 should be 100");

    state = eis_vm_step_lane(vm, lane);
    TEST_ASSERT(state == EIS_STATE_HALTED, "should be halted after HALT");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

static int test_exec_arithmetic(void) {
    DETCore *core = det_core_create();
    EIS_VM *vm = eis_vm_create(core);

    /* Load values, add, subtract, multiply */
    uint8_t program[24];
    encode_simple(program + 0, EIS_OP_LDI, 0, 0, 0, 10);   /* R0 = 10 */
    encode_simple(program + 4, EIS_OP_LDI, 1, 0, 0, 3);    /* R1 = 3 */
    encode_simple(program + 8, EIS_OP_ADD, 2, 0, 1, 0);    /* R2 = R0 + R1 = 13 */
    encode_simple(program + 12, EIS_OP_SUB, 3, 0, 1, 0);   /* R3 = R0 - R1 = 7 */
    encode_simple(program + 16, EIS_OP_MUL, 4, 0, 1, 0);   /* R4 = R0 * R1 = 30 */
    encode_simple(program + 20, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, 0, program, 24);
    eis_vm_run_lane(vm, lane, 100);

    TEST_ASSERT(fabsf(lane->regs.scalars[0] - 10.0f) < 0.001f, "R0 should be 10");
    TEST_ASSERT(fabsf(lane->regs.scalars[1] - 3.0f) < 0.001f, "R1 should be 3");
    TEST_ASSERT(fabsf(lane->regs.scalars[2] - 13.0f) < 0.001f, "R2 should be 13");
    TEST_ASSERT(fabsf(lane->regs.scalars[3] - 7.0f) < 0.001f, "R3 should be 7");
    TEST_ASSERT(fabsf(lane->regs.scalars[4] - 30.0f) < 0.001f, "R4 should be 30");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

static int test_exec_comparison(void) {
    DETCore *core = det_core_create();
    EIS_VM *vm = eis_vm_create(core);

    /* Compare: 10 < 20 -> T0 = LT */
    uint8_t program[16];
    encode_simple(program + 0, EIS_OP_LDI, 0, 0, 0, 10);    /* R0 = 10 */
    encode_simple(program + 4, EIS_OP_LDI, 1, 0, 0, 20);    /* R1 = 20 */
    encode_simple(program + 8, EIS_OP_CMP, 24, 0, 1, 0);    /* T0 = cmp(R0, R1) */
    encode_simple(program + 12, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, 0, program, 16);
    eis_vm_run_lane(vm, lane, 100);

    TEST_ASSERT(lane->regs.tokens[0] == EIS_TOK_LT, "T0 should be LT");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

static int test_exec_math_functions(void) {
    DETCore *core = det_core_create();
    EIS_VM *vm = eis_vm_create(core);

    /* SQRT, NEG, ABS */
    uint8_t program[20];
    encode_simple(program + 0, EIS_OP_LDI, 0, 0, 0, 16);    /* R0 = 16 */
    encode_simple(program + 4, EIS_OP_SQRT, 1, 0, 0, 0);    /* R1 = sqrt(R0) = 4 */
    encode_simple(program + 8, EIS_OP_NEG, 2, 1, 0, 0);     /* R2 = -R1 = -4 */
    encode_simple(program + 12, EIS_OP_ABS, 3, 2, 0, 0);    /* R3 = abs(R2) = 4 */
    encode_simple(program + 16, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, 0, program, 20);
    eis_vm_run_lane(vm, lane, 100);

    TEST_ASSERT(fabsf(lane->regs.scalars[1] - 4.0f) < 0.001f, "R1 should be 4");
    TEST_ASSERT(fabsf(lane->regs.scalars[2] - (-4.0f)) < 0.001f, "R2 should be -4");
    TEST_ASSERT(fabsf(lane->regs.scalars[3] - 4.0f) < 0.001f, "R3 should be 4");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * PHASE CONTROL TESTS
 * ========================================================================== */

static int test_phase_control(void) {
    DETCore *core = det_core_create();
    EIS_VM *vm = eis_vm_create(core);

    /* Set phase to READ */
    uint8_t program[8];
    encode_simple(program + 0, EIS_OP_PHASE, 0, 0, 0, EIS_PHASE_READ);
    encode_simple(program + 4, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, 0, program, 8);
    eis_vm_run_lane(vm, lane, 100);

    TEST_ASSERT(vm->phases.current_phase == EIS_PHASE_READ, "phase should be READ");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * DET INTEGRATION TESTS
 * ========================================================================== */

static int test_det_node_read(void) {
    DETCore *core = det_core_create();

    /* Use a P-layer node (indices 0-15 are pre-initialized and fit in 9-bit imm) */
    uint16_t node_id = 5;  /* P-layer node */
    core->nodes[node_id].F = 100.0f;
    core->nodes[node_id].sigma = 0.5f;

    EIS_VM *vm = eis_vm_create(core);
    vm->phases.strict = false;  /* Allow reads in any phase for testing */
    vm->phases.current_phase = EIS_PHASE_READ;

    /* LDN: Load node field */
    uint8_t program[16];
    encode_simple(program + 0, EIS_OP_MKNODE, 16, 0, 0, (int16_t)node_id);  /* H0 = node_id */
    encode_simple(program + 4, EIS_OP_LDN, 0, 16, 0, EIS_FIELD_F);          /* R0 = H0.F */
    encode_simple(program + 8, EIS_OP_LDN, 1, 16, 0, EIS_FIELD_SIGMA);      /* R1 = H0.sigma */
    encode_simple(program + 12, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, node_id, program, 16);
    eis_vm_run_lane(vm, lane, 100);

    TEST_ASSERT(fabsf(lane->regs.scalars[0] - 100.0f) < 0.001f, "R0 should be 100 (F)");
    TEST_ASSERT(fabsf(lane->regs.scalars[1] - 0.5f) < 0.001f, "R1 should be 0.5 (sigma)");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

static int test_det_node_write(void) {
    DETCore *core = det_core_create();

    /* Use a P-layer node (fits in 9-bit imm) */
    uint16_t node_id = 7;  /* P-layer node */
    core->nodes[node_id].F = 100.0f;

    EIS_VM *vm = eis_vm_create(core);
    vm->phases.strict = false;
    vm->phases.current_phase = EIS_PHASE_COMMIT;

    /* Store new value to node field */
    uint8_t program[12];
    encode_simple(program + 0, EIS_OP_LDI, 0, 0, 0, 50);              /* R0 = 50 */
    encode_simple(program + 4, EIS_OP_ST_NODE, 0, 0, 0, EIS_FIELD_F); /* self.F = R0 */
    encode_simple(program + 8, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, node_id, program, 12);
    eis_vm_run_lane(vm, lane, 100);

    TEST_ASSERT(fabsf(core->nodes[node_id].F - 50.0f) < 0.001f, "node F should be 50");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * PROPOSAL TESTS
 * ========================================================================== */

static int test_proposal_basic(void) {
    DETCore *core = det_core_create();
    EIS_VM *vm = eis_vm_create(core);
    vm->phases.strict = false;
    vm->phases.current_phase = EIS_PHASE_PROPOSE;

    /* Create a proposal with score */
    uint8_t program[20];
    encode_simple(program + 0, EIS_OP_PROP_BEGIN, 0, 0, 0, 0);   /* Begin proposal */
    encode_simple(program + 4, EIS_OP_LDI, 0, 0, 0, 100);        /* R0 = 100 (score) */
    encode_simple(program + 8, EIS_OP_PROP_SCORE, 0, 0, 0, 0);   /* Set score from R0 */
    encode_simple(program + 12, EIS_OP_PROP_END, 0, 0, 0, 0);    /* End proposal */
    encode_simple(program + 16, EIS_OP_HALT, 0, 0, 0, 0);

    EIS_Lane *lane = eis_vm_create_node_lane(vm, 0, program, 20);
    eis_vm_run_lane(vm, lane, 100);

    TEST_ASSERT(lane->proposals.num_proposals == 1, "should have 1 proposal");
    TEST_ASSERT(fabsf(lane->proposals.proposals[0].score - 100.0f) < 0.001f,
                "proposal score should be 100");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * UTILITY FUNCTION TESTS
 * ========================================================================== */

static int test_opcode_names(void) {
    TEST_ASSERT(strcmp(eis_opcode_name(EIS_OP_NOP), "NOP") == 0, "NOP name");
    TEST_ASSERT(strcmp(eis_opcode_name(EIS_OP_ADD), "ADD") == 0, "ADD name");
    TEST_ASSERT(strcmp(eis_opcode_name(EIS_OP_HALT), "HALT") == 0, "HALT name");
    TEST_ASSERT(strcmp(eis_opcode_name(EIS_OP_XFER), "XFER") == 0, "XFER name");
    return 1;
}

static int test_phase_names(void) {
    TEST_ASSERT(strcmp(eis_phase_name(EIS_PHASE_IDLE), "IDLE") == 0, "IDLE name");
    TEST_ASSERT(strcmp(eis_phase_name(EIS_PHASE_READ), "READ") == 0, "READ name");
    TEST_ASSERT(strcmp(eis_phase_name(EIS_PHASE_PROPOSE), "PROPOSE") == 0, "PROPOSE name");
    TEST_ASSERT(strcmp(eis_phase_name(EIS_PHASE_CHOOSE), "CHOOSE") == 0, "CHOOSE name");
    TEST_ASSERT(strcmp(eis_phase_name(EIS_PHASE_COMMIT), "COMMIT") == 0, "COMMIT name");
    return 1;
}

static int test_token_names(void) {
    TEST_ASSERT(strcmp(eis_token_name(EIS_TOK_VOID), "VOID") == 0, "VOID name");
    TEST_ASSERT(strcmp(eis_token_name(EIS_TOK_LT), "LT") == 0, "LT name");
    TEST_ASSERT(strcmp(eis_token_name(EIS_TOK_EQ), "EQ") == 0, "EQ name");
    TEST_ASSERT(strcmp(eis_token_name(EIS_TOK_GT), "GT") == 0, "GT name");
    TEST_ASSERT(strcmp(eis_token_name(EIS_TOK_XFER_OK), "XFER_OK") == 0, "XFER_OK name");
    return 1;
}

/* ==========================================================================
 * TICK EXECUTION TESTS
 * ========================================================================== */

static int test_run_tick(void) {
    DETCore *core = det_core_create();

    /* Recruit two nodes */
    int32_t node_a = det_core_recruit_node(core, DET_LAYER_A);
    int32_t node_b = det_core_recruit_node(core, DET_LAYER_A);
    TEST_ASSERT(node_a >= 0 && node_b >= 0, "should recruit nodes");

    core->nodes[node_a].F = 100.0f;
    core->nodes[node_b].F = 0.0f;

    EIS_VM *vm = eis_vm_create(core);

    /* Simple program: just halt */
    uint8_t program[4];
    encode_simple(program, EIS_OP_HALT, 0, 0, 0, 0);

    eis_vm_create_node_lane(vm, (uint16_t)node_a, program, 4);
    eis_vm_create_node_lane(vm, (uint16_t)node_b, program, 4);

    /* Run a tick */
    eis_vm_run_tick(vm);

    TEST_ASSERT(vm->tick == 1, "tick should be 1");
    TEST_ASSERT(vm->phases_completed == 1, "phases_completed should be 1");

    eis_vm_destroy(vm);
    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * MAIN
 * ========================================================================== */

int main(void) {
    printf("EIS VM Test Suite\n");
    printf("=================\n\n");

    /* Instruction encoding/decoding */
    printf("Instruction Encoding/Decoding:\n");
    RUN_TEST(test_decode_nop);
    RUN_TEST(test_decode_ldi);
    RUN_TEST(test_decode_negative_imm);
    RUN_TEST(test_encode_roundtrip);

    /* VM lifecycle */
    printf("\nVM Lifecycle:\n");
    RUN_TEST(test_vm_create_destroy);
    RUN_TEST(test_vm_create_lane);

    /* Instruction execution */
    printf("\nInstruction Execution:\n");
    RUN_TEST(test_exec_ldi);
    RUN_TEST(test_exec_arithmetic);
    RUN_TEST(test_exec_comparison);
    RUN_TEST(test_exec_math_functions);

    /* Phase control */
    printf("\nPhase Control:\n");
    RUN_TEST(test_phase_control);

    /* DET integration */
    printf("\nDET Integration:\n");
    RUN_TEST(test_det_node_read);
    RUN_TEST(test_det_node_write);

    /* Proposals */
    printf("\nProposals:\n");
    RUN_TEST(test_proposal_basic);

    /* Utility functions */
    printf("\nUtility Functions:\n");
    RUN_TEST(test_opcode_names);
    RUN_TEST(test_phase_names);
    RUN_TEST(test_token_names);

    /* Tick execution */
    printf("\nTick Execution:\n");
    RUN_TEST(test_run_tick);

    /* Summary */
    printf("\n=================\n");
    printf("Tests: %d/%d passed\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
