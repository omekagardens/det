/**
 * Phase 14 Test: Metal GPU Backend
 * =================================
 *
 * C test suite for the Metal GPU backend.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../include/substrate_metal.h"
#include "../include/eis_substrate_v2.h"

/* Test utilities */
#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 0; \
    } \
} while(0)

#define TEST_PASS() do { \
    printf("  PASS\n"); \
    return 1; \
} while(0)

/* ==========================================================================
 * Helper Functions
 * ========================================================================== */

static NodeArrays* create_nodes(uint32_t num_nodes) {
    NodeArrays* nodes = (NodeArrays*)calloc(1, sizeof(NodeArrays));
    if (!nodes) return NULL;

    nodes->F = (float*)calloc(num_nodes, sizeof(float));
    nodes->q = (float*)calloc(num_nodes, sizeof(float));
    nodes->a = (float*)calloc(num_nodes, sizeof(float));
    nodes->sigma = (float*)calloc(num_nodes, sizeof(float));
    nodes->P = (float*)calloc(num_nodes, sizeof(float));
    nodes->tau = (float*)calloc(num_nodes, sizeof(float));
    nodes->cos_theta = (float*)calloc(num_nodes, sizeof(float));
    nodes->sin_theta = (float*)calloc(num_nodes, sizeof(float));
    nodes->k = (uint32_t*)calloc(num_nodes, sizeof(uint32_t));
    nodes->r = (uint32_t*)calloc(num_nodes, sizeof(uint32_t));
    nodes->flags = (uint32_t*)calloc(num_nodes, sizeof(uint32_t));

    nodes->num_nodes = num_nodes;
    nodes->capacity = num_nodes;

    /* Initialize defaults */
    for (uint32_t i = 0; i < num_nodes; i++) {
        nodes->a[i] = 1.0f;
        nodes->sigma[i] = 1.0f;
        nodes->cos_theta[i] = 1.0f;
        nodes->sin_theta[i] = 0.0f;
    }

    return nodes;
}

static void destroy_nodes(NodeArrays* nodes) {
    if (!nodes) return;
    free(nodes->F);
    free(nodes->q);
    free(nodes->a);
    free(nodes->sigma);
    free(nodes->P);
    free(nodes->tau);
    free(nodes->cos_theta);
    free(nodes->sin_theta);
    free(nodes->k);
    free(nodes->r);
    free(nodes->flags);
    free(nodes);
}

static BondArrays* create_bonds(uint32_t num_bonds) {
    BondArrays* bonds = (BondArrays*)calloc(1, sizeof(BondArrays));
    if (!bonds) return NULL;

    bonds->node_i = (uint32_t*)calloc(num_bonds, sizeof(uint32_t));
    bonds->node_j = (uint32_t*)calloc(num_bonds, sizeof(uint32_t));
    bonds->C = (float*)calloc(num_bonds, sizeof(float));
    bonds->pi = (float*)calloc(num_bonds, sizeof(float));
    bonds->sigma = (float*)calloc(num_bonds, sizeof(float));
    bonds->flags = (uint32_t*)calloc(num_bonds, sizeof(uint32_t));

    bonds->num_bonds = num_bonds;
    bonds->capacity = num_bonds;

    /* Initialize defaults */
    for (uint32_t i = 0; i < num_bonds; i++) {
        bonds->C[i] = 0.5f;
        bonds->sigma[i] = 1.0f;
    }

    return bonds;
}

static void destroy_bonds(BondArrays* bonds) {
    if (!bonds) return;
    free(bonds->node_i);
    free(bonds->node_j);
    free(bonds->C);
    free(bonds->pi);
    free(bonds->sigma);
    free(bonds->flags);
    free(bonds);
}

static PredecodedProgram* create_simple_program(void) {
    PredecodedProgram* prog = (PredecodedProgram*)calloc(1, sizeof(PredecodedProgram));
    if (!prog) return NULL;

    prog->capacity = 20;
    prog->instrs = (SubstrateInstr*)calloc(prog->capacity, sizeof(SubstrateInstr));

    /* Simple program: phase transitions + halt */
    prog->instrs[0].opcode = OP_PHASE_R;  /* READ */
    prog->instrs[1].opcode = OP_NOP;
    prog->instrs[2].opcode = OP_PHASE_P;  /* PROPOSE */
    prog->instrs[3].opcode = OP_NOP;
    prog->instrs[4].opcode = OP_PHASE_C;  /* CHOOSE */
    prog->instrs[5].opcode = OP_NOP;
    prog->instrs[6].opcode = OP_PHASE_X;  /* COMMIT */
    prog->instrs[7].opcode = OP_NOP;
    prog->instrs[8].opcode = OP_HALT;

    prog->count = 9;

    return prog;
}

static void destroy_program(PredecodedProgram* prog) {
    if (!prog) return;
    free(prog->instrs);
    free(prog);
}

/* ==========================================================================
 * Test Cases
 * ========================================================================== */

static int test_availability(void) {
    printf("\n=== Test: Metal Availability ===\n");

    int available = sub_metal_is_available();
    printf("  Metal available: %s\n", available ? "YES" : "NO");

    if (!available) {
        printf("  SKIP: Metal not available\n");
        return 1;  /* Skip is not a failure */
    }

    TEST_PASS();
}

static int test_create_destroy(void) {
    printf("\n=== Test: Create/Destroy Context ===\n");

    if (!sub_metal_is_available()) {
        printf("  SKIP: Metal not available\n");
        return 1;
    }

    SubstrateMetalHandle ctx = sub_metal_create();
    TEST_ASSERT(ctx != NULL, "Failed to create context");

    printf("  Device: %s\n", sub_metal_device_name(ctx));
    printf("  Memory: %zu bytes\n", sub_metal_memory_usage(ctx));

    sub_metal_destroy(ctx);

    TEST_PASS();
}

static int test_custom_config(void) {
    printf("\n=== Test: Custom Configuration ===\n");

    if (!sub_metal_is_available()) {
        printf("  SKIP: Metal not available\n");
        return 1;
    }

    SubstrateMetalConfig config = sub_metal_default_config();
    config.max_nodes = 1000;
    config.max_bonds = 2000;
    config.max_lanes = 1000;

    SubstrateMetalHandle ctx = sub_metal_create_with_config(&config);
    TEST_ASSERT(ctx != NULL, "Failed to create context with config");

    sub_metal_destroy(ctx);

    TEST_PASS();
}

static int test_upload_download_nodes(void) {
    printf("\n=== Test: Upload/Download Nodes ===\n");

    if (!sub_metal_is_available()) {
        printf("  SKIP: Metal not available\n");
        return 1;
    }

    SubstrateMetalHandle ctx = sub_metal_create();
    TEST_ASSERT(ctx != NULL, "Failed to create context");

    uint32_t num_nodes = 100;
    NodeArrays* nodes = create_nodes(num_nodes);
    TEST_ASSERT(nodes != NULL, "Failed to create nodes");

    /* Set test values */
    for (uint32_t i = 0; i < num_nodes; i++) {
        nodes->F[i] = (float)i * 0.5f;
        nodes->q[i] = (float)i * 0.1f;
    }

    /* Upload */
    int result = sub_metal_upload_nodes(ctx, nodes, num_nodes);
    TEST_ASSERT(result == SUB_METAL_OK, "Failed to upload nodes");

    /* Download to new arrays */
    NodeArrays* downloaded = create_nodes(num_nodes);

    result = sub_metal_download_nodes(ctx, downloaded, num_nodes);
    TEST_ASSERT(result == SUB_METAL_OK, "Failed to download nodes");

    /* Verify */
    int errors = 0;
    for (uint32_t i = 0; i < num_nodes; i++) {
        if (fabsf(downloaded->F[i] - nodes->F[i]) > 1e-6f) errors++;
        if (fabsf(downloaded->q[i] - nodes->q[i]) > 1e-6f) errors++;
    }

    TEST_ASSERT(errors == 0, "Data mismatch after download");

    destroy_nodes(nodes);
    destroy_nodes(downloaded);
    sub_metal_destroy(ctx);

    TEST_PASS();
}

static int test_upload_download_bonds(void) {
    printf("\n=== Test: Upload/Download Bonds ===\n");

    if (!sub_metal_is_available()) {
        printf("  SKIP: Metal not available\n");
        return 1;
    }

    SubstrateMetalHandle ctx = sub_metal_create();
    TEST_ASSERT(ctx != NULL, "Failed to create context");

    uint32_t num_bonds = 50;
    BondArrays* bonds = create_bonds(num_bonds);
    TEST_ASSERT(bonds != NULL, "Failed to create bonds");

    /* Set test values */
    for (uint32_t i = 0; i < num_bonds; i++) {
        bonds->node_i[i] = i;
        bonds->node_j[i] = (i + 1) % num_bonds;
        bonds->C[i] = 0.5f + (float)i / (2 * num_bonds);
    }

    /* Upload */
    int result = sub_metal_upload_bonds(ctx, bonds, num_bonds);
    TEST_ASSERT(result == SUB_METAL_OK, "Failed to upload bonds");

    /* Download */
    BondArrays* downloaded = create_bonds(num_bonds);

    result = sub_metal_download_bonds(ctx, downloaded, num_bonds);
    TEST_ASSERT(result == SUB_METAL_OK, "Failed to download bonds");

    /* Verify */
    int errors = 0;
    for (uint32_t i = 0; i < num_bonds; i++) {
        if (downloaded->node_i[i] != bonds->node_i[i]) errors++;
        if (downloaded->node_j[i] != bonds->node_j[i]) errors++;
        if (fabsf(downloaded->C[i] - bonds->C[i]) > 1e-6f) errors++;
    }

    TEST_ASSERT(errors == 0, "Data mismatch after download");

    destroy_bonds(bonds);
    destroy_bonds(downloaded);
    sub_metal_destroy(ctx);

    TEST_PASS();
}

static int test_upload_program(void) {
    printf("\n=== Test: Upload Program ===\n");

    if (!sub_metal_is_available()) {
        printf("  SKIP: Metal not available\n");
        return 1;
    }

    SubstrateMetalHandle ctx = sub_metal_create();
    TEST_ASSERT(ctx != NULL, "Failed to create context");

    PredecodedProgram* prog = create_simple_program();
    TEST_ASSERT(prog != NULL, "Failed to create program");

    int result = sub_metal_upload_program(ctx, prog);
    TEST_ASSERT(result == SUB_METAL_OK, "Failed to upload program");

    destroy_program(prog);
    sub_metal_destroy(ctx);

    TEST_PASS();
}

static int test_execute_tick(void) {
    printf("\n=== Test: Execute Tick ===\n");

    if (!sub_metal_is_available()) {
        printf("  SKIP: Metal not available\n");
        return 1;
    }

    SubstrateMetalHandle ctx = sub_metal_create();
    TEST_ASSERT(ctx != NULL, "Failed to create context");

    uint32_t num_nodes = 10;
    uint32_t num_lanes = 10;

    NodeArrays* nodes = create_nodes(num_nodes);
    PredecodedProgram* prog = create_simple_program();

    for (uint32_t i = 0; i < num_nodes; i++) {
        nodes->F[i] = 10.0f;
    }

    sub_metal_upload_nodes(ctx, nodes, num_nodes);
    sub_metal_upload_program(ctx, prog);

    int result = sub_metal_execute_tick(ctx, num_lanes);
    TEST_ASSERT(result == SUB_METAL_OK, "Failed to execute tick");

    sub_metal_synchronize(ctx);

    sub_metal_download_nodes(ctx, nodes, num_nodes);

    destroy_nodes(nodes);
    destroy_program(prog);
    sub_metal_destroy(ctx);

    TEST_PASS();
}

static int test_multiple_ticks(void) {
    printf("\n=== Test: Multiple Ticks ===\n");

    if (!sub_metal_is_available()) {
        printf("  SKIP: Metal not available\n");
        return 1;
    }

    SubstrateMetalHandle ctx = sub_metal_create();
    TEST_ASSERT(ctx != NULL, "Failed to create context");

    uint32_t num_nodes = 100;
    uint32_t num_lanes = 100;
    uint32_t num_ticks = 10;

    NodeArrays* nodes = create_nodes(num_nodes);
    PredecodedProgram* prog = create_simple_program();

    sub_metal_upload_nodes(ctx, nodes, num_nodes);
    sub_metal_upload_program(ctx, prog);

    int result = sub_metal_execute_ticks(ctx, num_lanes, num_ticks);
    TEST_ASSERT(result == SUB_METAL_OK, "Failed to execute ticks");

    sub_metal_synchronize(ctx);

    printf("  Executed %u ticks with %u lanes\n", num_ticks, num_lanes);

    destroy_nodes(nodes);
    destroy_program(prog);
    sub_metal_destroy(ctx);

    TEST_PASS();
}

static int test_performance(void) {
    printf("\n=== Test: Performance Benchmark ===\n");

    if (!sub_metal_is_available()) {
        printf("  SKIP: Metal not available\n");
        return 1;
    }

    SubstrateMetalHandle ctx = sub_metal_create();
    TEST_ASSERT(ctx != NULL, "Failed to create context");

    printf("  Device: %s\n", sub_metal_device_name(ctx));

    struct {
        uint32_t num_nodes;
        uint32_t num_bonds;
        uint32_t num_ticks;
    } configs[] = {
        { 100, 200, 100 },
        { 1000, 2000, 100 },
        { 10000, 20000, 100 },
    };

    for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); i++) {
        uint32_t num_nodes = configs[i].num_nodes;
        uint32_t num_bonds = configs[i].num_bonds;
        uint32_t num_ticks = configs[i].num_ticks;

        NodeArrays* nodes = create_nodes(num_nodes);
        BondArrays* bonds = create_bonds(num_bonds);
        PredecodedProgram* prog = create_simple_program();

        /* Ring topology */
        for (uint32_t j = 0; j < num_bonds; j++) {
            bonds->node_i[j] = j % num_nodes;
            bonds->node_j[j] = (j + 1) % num_nodes;
        }

        /* Initialize resource */
        for (uint32_t j = 0; j < num_nodes; j++) {
            nodes->F[j] = 10.0f;
        }

        sub_metal_upload_nodes(ctx, nodes, num_nodes);
        sub_metal_upload_bonds(ctx, bonds, num_bonds);
        sub_metal_upload_program(ctx, prog);

        clock_t start = clock();
        sub_metal_execute_ticks(ctx, num_nodes, num_ticks);
        sub_metal_synchronize(ctx);
        clock_t end = clock();

        double elapsed_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
        double rate = num_ticks * 1000.0 / elapsed_ms;

        printf("  Nodes: %5u, Bonds: %5u, Ticks: %u\n", num_nodes, num_bonds, num_ticks);
        printf("    Time: %.2f ms, Rate: %.1f ticks/sec\n", elapsed_ms, rate);

        destroy_nodes(nodes);
        destroy_bonds(bonds);
        destroy_program(prog);
    }

    sub_metal_destroy(ctx);

    TEST_PASS();
}

/* ==========================================================================
 * Main
 * ========================================================================== */

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("============================================================\n");
    printf("Phase 14: GPU Backend with Metal Compute Shaders\n");
    printf("============================================================\n");

    int passed = 0;
    int failed = 0;

    if (test_availability()) passed++; else failed++;
    if (test_create_destroy()) passed++; else failed++;
    if (test_custom_config()) passed++; else failed++;
    if (test_upload_download_nodes()) passed++; else failed++;
    if (test_upload_download_bonds()) passed++; else failed++;
    if (test_upload_program()) passed++; else failed++;
    if (test_execute_tick()) passed++; else failed++;
    if (test_multiple_ticks()) passed++; else failed++;
    if (test_performance()) passed++; else failed++;

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    printf("============================================================\n");

    return failed == 0 ? 0 : 1;
}
