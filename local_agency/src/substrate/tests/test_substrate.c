/**
 * EIS Substrate Tests
 * ===================
 *
 * Tests for the minimal execution layer.
 * Verifies pure execution with NO DET semantics.
 */

#include "../include/eis_substrate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==========================================================================
 * TEST HELPERS
 * ========================================================================== */

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 0; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define FLOAT_EQ(a, b) (fabsf((a) - (b)) < 0.001f)

/* Encode a simple instruction */
static uint32_t encode_instr(uint8_t opcode, uint8_t dst, uint8_t src0, uint8_t src1, int16_t imm) {
    uint32_t imm_bits = imm & 0x1FF;
    return ((uint32_t)(opcode & 0xFF) << 24) |
           ((uint32_t)(dst & 0x1F) << 19) |
           ((uint32_t)(src0 & 0x1F) << 14) |
           ((uint32_t)(src1 & 0x1F) << 9) |
           (imm_bits);
}

/* Write instruction to buffer (big-endian) */
static void write_instr(uint8_t* buf, uint32_t offset, uint32_t instr) {
    buf[offset + 0] = (instr >> 24) & 0xFF;
    buf[offset + 1] = (instr >> 16) & 0xFF;
    buf[offset + 2] = (instr >> 8) & 0xFF;
    buf[offset + 3] = instr & 0xFF;
}

/* ==========================================================================
 * TESTS
 * ========================================================================== */

static int test_vm_lifecycle(void) {
    printf("test_vm_lifecycle:\n");

    /* Create VM */
    SubstrateVM* vm = substrate_create(1024);
    TEST_ASSERT(vm != NULL, "VM creation");

    /* Check initial state */
    TEST_ASSERT(vm->state == SUB_STATE_HALTED, "Initial state is HALTED");
    TEST_ASSERT(vm->memory_size == 1024, "Memory size correct");

    /* Reset */
    substrate_reset(vm);
    TEST_ASSERT(vm->state == SUB_STATE_RUNNING, "After reset, state is RUNNING");
    TEST_ASSERT(vm->pc == 0, "PC reset to 0");
    TEST_ASSERT(vm->tick == 0, "Tick counter reset");

    /* Destroy */
    substrate_destroy(vm);

    printf("  All lifecycle tests passed\n");
    return 1;
}

static int test_load_immediate(void) {
    printf("test_load_immediate:\n");

    SubstrateVM* vm = substrate_create(1024);
    TEST_ASSERT(vm != NULL, "VM creation");

    /* Program: LDI R0, 42; LDI R1, -10; HALT */
    uint8_t program[12];
    write_instr(program, 0, encode_instr(SUB_OP_LDI, 0, 0, 0, 42));
    write_instr(program, 4, encode_instr(SUB_OP_LDI, 1, 0, 0, -10 & 0x1FF));
    write_instr(program, 8, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    TEST_ASSERT(substrate_load_program(vm, program, sizeof(program)), "Program loaded");

    /* Execute */
    SubstrateState state = substrate_run(vm, 100);
    TEST_ASSERT(state == SUB_STATE_HALTED, "Program halted");

    /* Check registers */
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 0), 42.0f), "R0 = 42");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 1), -10.0f), "R1 = -10");

    substrate_destroy(vm);
    printf("  All LDI tests passed\n");
    return 1;
}

static int test_arithmetic(void) {
    printf("test_arithmetic:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * LDI R0, 10
     * LDI R1, 3
     * ADD R2, R0, R1   ; R2 = 13
     * SUB R3, R0, R1   ; R3 = 7
     * MUL R4, R0, R1   ; R4 = 30
     * DIV R5, R0, R1   ; R5 = 3.333...
     * NEG R6, R0       ; R6 = -10
     * ABS R7, R6       ; R7 = 10
     * HALT
     */
    uint8_t program[36];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 10));
    write_instr(program, 4,  encode_instr(SUB_OP_LDI, 1, 0, 0, 3));
    write_instr(program, 8,  encode_instr(SUB_OP_ADD, 2, 0, 1, 0));
    write_instr(program, 12, encode_instr(SUB_OP_SUB, 3, 0, 1, 0));
    write_instr(program, 16, encode_instr(SUB_OP_MUL, 4, 0, 1, 0));
    write_instr(program, 20, encode_instr(SUB_OP_DIV, 5, 0, 1, 0));
    write_instr(program, 24, encode_instr(SUB_OP_NEG, 6, 0, 0, 0));
    write_instr(program, 28, encode_instr(SUB_OP_ABS, 7, 6, 0, 0));
    write_instr(program, 32, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    SubstrateState state = substrate_run(vm, 100);
    TEST_ASSERT(state == SUB_STATE_HALTED, "Program halted");

    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 2), 13.0f), "ADD: R2 = 13");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 3), 7.0f), "SUB: R3 = 7");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 4), 30.0f), "MUL: R4 = 30");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 5), 10.0f / 3.0f), "DIV: R5 = 3.33...");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 6), -10.0f), "NEG: R6 = -10");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 7), 10.0f), "ABS: R7 = 10");

    substrate_destroy(vm);
    printf("  All arithmetic tests passed\n");
    return 1;
}

static int test_math_functions(void) {
    printf("test_math_functions:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * LDI R0, 4
     * SQRT R1, R0      ; R1 = 2
     * LDI R2, 0        ; R2 = 0 (for sin/cos)
     * SIN R3, R2       ; R3 = sin(0) = 0
     * COS R4, R2       ; R4 = cos(0) = 1
     * HALT
     */
    uint8_t program[24];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 4));
    write_instr(program, 4,  encode_instr(SUB_OP_SQRT, 1, 0, 0, 0));
    write_instr(program, 8,  encode_instr(SUB_OP_LDI, 2, 0, 0, 0));
    write_instr(program, 12, encode_instr(SUB_OP_SIN, 3, 2, 0, 0));
    write_instr(program, 16, encode_instr(SUB_OP_COS, 4, 2, 0, 0));
    write_instr(program, 20, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 1), 2.0f), "SQRT(4) = 2");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 3), 0.0f), "SIN(0) = 0");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 4), 1.0f), "COS(0) = 1");

    substrate_destroy(vm);
    printf("  All math function tests passed\n");
    return 1;
}

static int test_comparison(void) {
    printf("test_comparison:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * LDI R0, 5
     * LDI R1, 10
     * CMP T0, R0, R1   ; T0 = LT (5 < 10)
     * CMP T1, R1, R0   ; T1 = GT (10 > 5)
     * CMP T2, R0, R0   ; T2 = EQ (5 == 5)
     * HALT
     */
    uint8_t program[24];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 5));
    write_instr(program, 4,  encode_instr(SUB_OP_LDI, 1, 0, 0, 10));
    write_instr(program, 8,  encode_instr(SUB_OP_CMP, 0, 0, 1, 0));  /* T0 */
    write_instr(program, 12, encode_instr(SUB_OP_CMP, 1, 1, 0, 0));  /* T1 */
    write_instr(program, 16, encode_instr(SUB_OP_CMP, 2, 0, 0, 0));  /* T2 */
    write_instr(program, 20, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    TEST_ASSERT(substrate_get_token(vm, 0) == SUB_TOK_LT, "CMP: 5 < 10 -> LT");
    TEST_ASSERT(substrate_get_token(vm, 1) == SUB_TOK_GT, "CMP: 10 > 5 -> GT");
    TEST_ASSERT(substrate_get_token(vm, 2) == SUB_TOK_EQ, "CMP: 5 == 5 -> EQ");

    substrate_destroy(vm);
    printf("  All comparison tests passed\n");
    return 1;
}

static int test_branching(void) {
    printf("test_branching:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * LDI R0, 0        ; R0 = 0
     * LDI R1, 1        ; R1 = 1
     * JNZ R0, +2       ; Don't jump (R0 = 0)
     * ADD R0, R0, R1   ; R0 = 1
     * JNZ R0, +2       ; Jump (R0 = 1), skip next
     * ADD R0, R0, R1   ; (skipped)
     * ADD R0, R0, R1   ; R0 = 2
     * HALT
     */
    uint8_t program[32];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 0));
    write_instr(program, 4,  encode_instr(SUB_OP_LDI, 1, 0, 0, 1));
    write_instr(program, 8,  encode_instr(SUB_OP_JNZ, 0, 0, 0, 2));    /* Don't jump */
    write_instr(program, 12, encode_instr(SUB_OP_ADD, 0, 0, 1, 0));    /* R0 = 1 */
    write_instr(program, 16, encode_instr(SUB_OP_JNZ, 0, 0, 0, 2));    /* Jump over next */
    write_instr(program, 20, encode_instr(SUB_OP_ADD, 0, 0, 1, 0));    /* Skipped */
    write_instr(program, 24, encode_instr(SUB_OP_ADD, 0, 0, 1, 0));    /* R0 = 2 */
    write_instr(program, 28, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 0), 2.0f), "JNZ branching: R0 = 2");

    substrate_destroy(vm);
    printf("  All branching tests passed\n");
    return 1;
}

static int test_memory(void) {
    printf("test_memory:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * LDI R0, 42       ; Value to store
     * LDI R1, 100      ; Address
     * ST R1, R0, 0     ; mem[100] = 42
     * LDI R2, 0        ; Clear R2
     * LD R2, R1, 0     ; R2 = mem[100] = 42
     * ST R1, R0, 5     ; mem[105] = 42
     * LD R3, R1, 5     ; R3 = mem[105] = 42
     * HALT
     */
    uint8_t program[32];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 42));
    write_instr(program, 4,  encode_instr(SUB_OP_LDI, 1, 0, 0, 100));
    write_instr(program, 8,  encode_instr(SUB_OP_ST, 1, 0, 0, 0));
    write_instr(program, 12, encode_instr(SUB_OP_LDI, 2, 0, 0, 0));
    write_instr(program, 16, encode_instr(SUB_OP_LD, 2, 1, 0, 0));
    write_instr(program, 20, encode_instr(SUB_OP_ST, 1, 0, 0, 5));
    write_instr(program, 24, encode_instr(SUB_OP_LD, 3, 1, 0, 5));
    write_instr(program, 28, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 2), 42.0f), "LD: R2 = mem[100] = 42");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 3), 42.0f), "LD: R3 = mem[105] = 42");
    TEST_ASSERT(FLOAT_EQ(substrate_mem_read(vm, 100), 42.0f), "mem[100] = 42");
    TEST_ASSERT(FLOAT_EQ(substrate_mem_read(vm, 105), 42.0f), "mem[105] = 42");

    substrate_destroy(vm);
    printf("  All memory tests passed\n");
    return 1;
}

static int test_io(void) {
    printf("test_io:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Inject value into channel 0 */
    substrate_io_inject(vm, 0, 99.0f);

    /* Program:
     * IN R0, 0         ; R0 = io[0] = 99
     * LDI R1, 50
     * OUT 1, R1        ; io[1] = 50
     * HALT
     */
    uint8_t program[16];
    write_instr(program, 0,  encode_instr(SUB_OP_IN, 0, 0, 0, 0));    /* channel 0 */
    write_instr(program, 4,  encode_instr(SUB_OP_LDI, 1, 0, 0, 50));
    write_instr(program, 8,  encode_instr(SUB_OP_OUT, 0, 1, 0, 1));   /* channel 1 */
    write_instr(program, 12, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 0), 99.0f), "IN: R0 = io[0] = 99");
    TEST_ASSERT(FLOAT_EQ(substrate_io_read(vm, 1), 50.0f), "OUT: io[1] = 50");

    substrate_destroy(vm);
    printf("  All I/O tests passed\n");
    return 1;
}

static int test_tick_counter(void) {
    printf("test_tick_counter:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * TICK             ; tick++
     * TICK             ; tick++
     * TICK             ; tick++
     * TIME R0          ; R0 = tick = 3
     * HALT
     */
    uint8_t program[20];
    write_instr(program, 0,  encode_instr(SUB_OP_TICK, 0, 0, 0, 0));
    write_instr(program, 4,  encode_instr(SUB_OP_TICK, 0, 0, 0, 0));
    write_instr(program, 8,  encode_instr(SUB_OP_TICK, 0, 0, 0, 0));
    write_instr(program, 12, encode_instr(SUB_OP_TIME, 0, 0, 0, 0));
    write_instr(program, 16, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 0), 3.0f), "TIME: R0 = tick = 3");

    uint64_t tick, instr;
    substrate_get_stats(vm, &tick, &instr);
    TEST_ASSERT(tick == 3, "tick counter = 3");
    TEST_ASSERT(instr == 5, "5 instructions executed");

    substrate_destroy(vm);
    printf("  All tick counter tests passed\n");
    return 1;
}

static int test_yield_resume(void) {
    printf("test_yield_resume:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * LDI R0, 1
     * YIELD
     * LDI R0, 2
     * HALT
     */
    uint8_t program[16];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 1));
    write_instr(program, 4,  encode_instr(SUB_OP_YIELD, 0, 0, 0, 0));
    write_instr(program, 8,  encode_instr(SUB_OP_LDI, 0, 0, 0, 2));
    write_instr(program, 12, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));

    /* Run until yield */
    SubstrateState state = substrate_run(vm, 100);
    TEST_ASSERT(state == SUB_STATE_YIELDED, "First run yielded");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 0), 1.0f), "R0 = 1 after yield");

    /* Resume */
    state = substrate_resume(vm, 100);
    TEST_ASSERT(state == SUB_STATE_HALTED, "Resume then halted");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 0), 2.0f), "R0 = 2 after resume");

    substrate_destroy(vm);
    printf("  All yield/resume tests passed\n");
    return 1;
}

static int test_loop(void) {
    printf("test_loop:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program: Sum 1+2+3+4+5 = 15
     * LDI R0, 0        ; sum = 0
     * LDI R1, 1        ; i = 1
     * LDI R2, 6        ; limit = 6
     * loop:
     *   ADD R0, R0, R1 ; sum += i
     *   INC R1         ; i++
     *   SUB R3, R1, R2 ; R3 = i - limit
     *   JLT R3, -3     ; if i < limit, jump back 3 instructions
     * HALT
     */
    uint8_t program[32];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 0));     /* sum = 0 */
    write_instr(program, 4,  encode_instr(SUB_OP_LDI, 1, 0, 0, 1));     /* i = 1 */
    write_instr(program, 8,  encode_instr(SUB_OP_LDI, 2, 0, 0, 6));     /* limit = 6 */
    /* loop: */
    write_instr(program, 12, encode_instr(SUB_OP_ADD, 0, 0, 1, 0));     /* sum += i */
    write_instr(program, 16, encode_instr(SUB_OP_INC, 1, 0, 0, 0));     /* i++ */
    write_instr(program, 20, encode_instr(SUB_OP_SUB, 3, 1, 2, 0));     /* R3 = i - limit */
    write_instr(program, 24, encode_instr(SUB_OP_JLT, 0, 3, 0, -3 & 0x1FF));  /* if R3 < 0, loop */
    write_instr(program, 28, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    SubstrateState state = substrate_run(vm, 1000);

    TEST_ASSERT(state == SUB_STATE_HALTED, "Loop halted");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 0), 15.0f), "Sum 1..5 = 15");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 1), 6.0f), "i = 6 at end");

    substrate_destroy(vm);
    printf("  All loop tests passed\n");
    return 1;
}

static int test_reference_registers(void) {
    printf("test_reference_registers:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * HREF H0, 100     ; H0 = 100 (address)
     * LDI R0, 42
     * STR H0, R0, 0    ; mem[H0] = 42
     * LDR R1, H0, 0    ; R1 = mem[H0] = 42
     * HADD H0, 5       ; H0 = 105
     * STR H0, R0, 0    ; mem[105] = 42
     * HGET R2, H0      ; R2 = 105
     * HALT
     */
    uint8_t program[32];
    write_instr(program, 0,  encode_instr(SUB_OP_HREF, 0, 0, 0, 100));
    write_instr(program, 4,  encode_instr(SUB_OP_LDI, 0, 0, 0, 42));
    write_instr(program, 8,  encode_instr(SUB_OP_STR, 0, 0, 0, 0));    /* H0, R0 */
    write_instr(program, 12, encode_instr(SUB_OP_LDR, 1, 0, 0, 0));    /* R1, H0 */
    write_instr(program, 16, encode_instr(SUB_OP_HADD, 0, 0, 0, 5));
    write_instr(program, 20, encode_instr(SUB_OP_STR, 0, 0, 0, 0));
    write_instr(program, 24, encode_instr(SUB_OP_HGET, 2, 0, 0, 0));
    write_instr(program, 28, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 1), 42.0f), "LDR: R1 = 42");
    TEST_ASSERT(FLOAT_EQ(substrate_mem_read(vm, 100), 42.0f), "mem[100] = 42");
    TEST_ASSERT(FLOAT_EQ(substrate_mem_read(vm, 105), 42.0f), "mem[105] = 42");
    TEST_ASSERT(substrate_get_ref(vm, 0) == 105, "H0 = 105");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 2), 105.0f), "HGET: R2 = 105");

    substrate_destroy(vm);
    printf("  All reference register tests passed\n");
    return 1;
}

static int test_random(void) {
    printf("test_random:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * LDI R0, 42
     * SEED R0          ; Seed with 42
     * RAND R1          ; R1 = random [0,1)
     * RAND R2          ; R2 = random [0,1)
     * HALT
     */
    uint8_t program[20];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 42));
    write_instr(program, 4,  encode_instr(SUB_OP_SEED, 0, 0, 0, 0));
    write_instr(program, 8,  encode_instr(SUB_OP_RAND, 1, 0, 0, 0));
    write_instr(program, 12, encode_instr(SUB_OP_RAND, 2, 0, 0, 0));
    write_instr(program, 16, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    float r1 = substrate_get_scalar(vm, 1);
    float r2 = substrate_get_scalar(vm, 2);

    TEST_ASSERT(r1 >= 0.0f && r1 < 1.0f, "RAND: R1 in [0,1)");
    TEST_ASSERT(r2 >= 0.0f && r2 < 1.0f, "RAND: R2 in [0,1)");
    TEST_ASSERT(fabsf(r1 - r2) > 0.0001f, "RAND: R1 != R2");

    substrate_destroy(vm);
    printf("  All random tests passed\n");
    return 1;
}

static int test_min_max_clamp(void) {
    printf("test_min_max_clamp:\n");

    SubstrateVM* vm = substrate_create(1024);

    /* Program:
     * LDI R0, 10
     * LDI R1, 20
     * MIN R2, R0, R1   ; R2 = min(10, 20) = 10
     * MAX R3, R0, R1   ; R3 = max(10, 20) = 20
     * LDI R4, -5
     * RELU R5, R4      ; R5 = max(0, -5) = 0
     * RELU R6, R0      ; R6 = max(0, 10) = 10
     * HALT
     */
    uint8_t program[32];
    write_instr(program, 0,  encode_instr(SUB_OP_LDI, 0, 0, 0, 10));
    write_instr(program, 4,  encode_instr(SUB_OP_LDI, 1, 0, 0, 20));
    write_instr(program, 8,  encode_instr(SUB_OP_MIN, 2, 0, 1, 0));
    write_instr(program, 12, encode_instr(SUB_OP_MAX, 3, 0, 1, 0));
    write_instr(program, 16, encode_instr(SUB_OP_LDI, 4, 0, 0, -5 & 0x1FF));
    write_instr(program, 20, encode_instr(SUB_OP_RELU, 5, 4, 0, 0));
    write_instr(program, 24, encode_instr(SUB_OP_RELU, 6, 0, 0, 0));
    write_instr(program, 28, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    substrate_load_program(vm, program, sizeof(program));
    substrate_run(vm, 100);

    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 2), 10.0f), "MIN(10, 20) = 10");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 3), 20.0f), "MAX(10, 20) = 20");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 5), 0.0f), "RELU(-5) = 0");
    TEST_ASSERT(FLOAT_EQ(substrate_get_scalar(vm, 6), 10.0f), "RELU(10) = 10");

    substrate_destroy(vm);
    printf("  All min/max/clamp tests passed\n");
    return 1;
}

static int test_disassemble(void) {
    printf("test_disassemble:\n");

    uint8_t program[12];
    write_instr(program, 0, encode_instr(SUB_OP_LDI, 0, 0, 0, 42));
    write_instr(program, 4, encode_instr(SUB_OP_ADD, 2, 0, 1, 0));
    write_instr(program, 8, encode_instr(SUB_OP_HALT, 0, 0, 0, 0));

    char buf[1024];
    substrate_disassemble(program, sizeof(program), buf, sizeof(buf));

    TEST_ASSERT(strstr(buf, "LDI") != NULL, "Disassembly contains LDI");
    TEST_ASSERT(strstr(buf, "ADD") != NULL, "Disassembly contains ADD");
    TEST_ASSERT(strstr(buf, "HALT") != NULL, "Disassembly contains HALT");

    printf("  All disassemble tests passed\n");
    return 1;
}

/* ==========================================================================
 * MAIN
 * ========================================================================== */

int main(void) {
    printf("=================================\n");
    printf("EIS Substrate Tests\n");
    printf("NO DET semantics - pure execution\n");
    printf("=================================\n\n");

    test_vm_lifecycle();
    test_load_immediate();
    test_arithmetic();
    test_math_functions();
    test_comparison();
    test_branching();
    test_memory();
    test_io();
    test_tick_counter();
    test_yield_resume();
    test_loop();
    test_reference_registers();
    test_random();
    test_min_max_clamp();
    test_disassemble();

    printf("\n=================================\n");
    printf("Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("=================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
