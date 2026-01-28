/**
 * DET SSM Tests
 * ==============
 *
 * Unit tests for SSM (Mamba) operations.
 */

#include "det_ssm.h"
#include "det_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (fabsf((a) - (b)) > (eps)) { \
        fprintf(stderr, "FAIL: %s - expected %.6f, got %.6f (%s:%d)\n", \
                msg, (float)(b), (float)(a), __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

/* ==========================================================================
 * SSM CACHE TESTS
 * ========================================================================== */

static int test_ssm_cache_create_free(void) {
    printf("Testing SSM cache create/free...\n");

    /* Create cache with typical Mamba-130M dimensions */
    int32_t n_layers = 12;
    int32_t d_inner = 1024;
    int32_t d_state = 16;
    int32_t d_conv = 4;
    int32_t dt_rank = 64;  /* typical: ceil(d_model/16) where d_model=512 */
    int32_t max_seq_len = 256;

    DetSSMCache* cache = det_ssm_cache_create(n_layers, d_inner, d_state, d_conv, dt_rank, max_seq_len);
    ASSERT(cache != NULL, "Cache creation should succeed");
    ASSERT(cache->n_layers == n_layers, "Layer count should match");
    ASSERT(cache->d_inner == d_inner, "Inner dimension should match");
    ASSERT(cache->d_state == d_state, "State dimension should match");
    ASSERT(cache->d_conv == d_conv, "Conv width should match");
    ASSERT(cache->h != NULL, "Hidden state tensor should be allocated");
    ASSERT(cache->conv_state != NULL, "Conv state tensor should be allocated");
    ASSERT(cache->initialized == true, "Cache should be initialized");

    /* Verify dimensions */
    ASSERT(cache->h->shape[0] == n_layers, "h shape[0] should be n_layers");
    ASSERT(cache->h->shape[1] == d_inner, "h shape[1] should be d_inner");
    ASSERT(cache->h->shape[2] == d_state, "h shape[2] should be d_state");

    ASSERT(cache->conv_state->shape[0] == n_layers, "conv_state shape[0] should be n_layers");
    ASSERT(cache->conv_state->shape[1] == d_inner, "conv_state shape[1] should be d_inner");
    ASSERT(cache->conv_state->shape[2] == d_conv - 1, "conv_state shape[2] should be d_conv-1");

    det_ssm_cache_free(cache);
    printf("  PASS\n");
    return 0;
}

static int test_ssm_cache_reset(void) {
    printf("Testing SSM cache reset...\n");

    /* n_layers=4, d_inner=256, d_state=16, d_conv=4, dt_rank=8, max_seq=64 */
    DetSSMCache* cache = det_ssm_cache_create(4, 256, 16, 4, 8, 64);
    ASSERT(cache != NULL, "Cache creation should succeed");

    /* Fill with non-zero values */
    float* h_data = (float*)cache->h->data;
    float* conv_data = (float*)cache->conv_state->data;

    for (int i = 0; i < 4 * 256 * 16; i++) {
        h_data[i] = 1.0f;
    }
    for (int i = 0; i < 4 * 256 * 3; i++) {
        conv_data[i] = 1.0f;
    }

    /* Reset */
    det_ssm_cache_reset(cache);

    /* Verify zeros */
    for (int i = 0; i < 4 * 256 * 16; i++) {
        ASSERT_NEAR(h_data[i], 0.0f, 1e-6f, "h should be zero after reset");
    }
    for (int i = 0; i < 4 * 256 * 3; i++) {
        ASSERT_NEAR(conv_data[i], 0.0f, 1e-6f, "conv_state should be zero after reset");
    }

    det_ssm_cache_free(cache);
    printf("  PASS\n");
    return 0;
}

/* ==========================================================================
 * CAUSAL CONV1D TESTS
 * ========================================================================== */

static int test_conv1d_causal_basic(void) {
    printf("Testing causal conv1d basic...\n");

    /* Simple test: d_inner=2, d_conv=3, seq_len=4 */
    int32_t d_inner = 2;
    int32_t d_conv = 3;
    int32_t seq_len = 4;

    float input[8] = {1, 0, 2, 0, 3, 0, 4, 0};  /* [4, 2], only channel 0 has values */
    float weight[6] = {1, 0.5f, 0.25f, 1, 1, 1};  /* [2, 3] */
    float bias[2] = {0, 0};
    float output[8] = {0};
    float conv_state[4] = {0};  /* [2, 2] for d_conv-1=2 */

    int ret = det_conv1d_causal(output, input, weight, bias, conv_state, d_inner, d_conv, seq_len);
    ASSERT(ret == 0, "conv1d should succeed");

    /* Expected output for channel 0 with weights [1, 0.5, 0.25]:
     * t=0: 1*1 + 0.5*0 + 0.25*0 = 1
     * t=1: 1*2 + 0.5*1 + 0.25*0 = 2.5
     * t=2: 1*3 + 0.5*2 + 0.25*1 = 4.25
     * t=3: 1*4 + 0.5*3 + 0.25*2 = 6.0 */
    ASSERT_NEAR(output[0], 1.0f, 1e-5f, "output[0] should be 1.0");
    ASSERT_NEAR(output[2], 2.5f, 1e-5f, "output[2] should be 2.5");
    ASSERT_NEAR(output[4], 4.25f, 1e-5f, "output[4] should be 4.25");
    ASSERT_NEAR(output[6], 6.0f, 1e-5f, "output[6] should be 6.0");

    printf("  PASS\n");
    return 0;
}

/* ==========================================================================
 * SELECTIVE SCAN TESTS
 * ========================================================================== */

static int test_ssm_selective_scan_basic(void) {
    printf("Testing SSM selective scan basic...\n");

    /* Minimal test: d_inner=1, d_state=1, seq_len=3 */
    int32_t d_inner = 1;
    int32_t d_state = 1;
    int32_t seq_len = 3;

    float x[3] = {1.0f, 1.0f, 1.0f};
    float delta[3] = {0.1f, 0.1f, 0.1f};
    float A[1] = {-1.0f};  /* log(-A), so A = -exp(-1) = -0.368 */
    float B[3] = {1.0f, 1.0f, 1.0f};
    float C[3] = {1.0f, 1.0f, 1.0f};
    float D[1] = {0.5f};
    float h[1] = {0.0f};  /* Initial state */
    float y[3] = {0.0f, 0.0f, 0.0f};

    int ret = det_ssm_selective_scan(y, x, delta, A, NULL, B, C, D, h, d_inner, d_state, seq_len);
    ASSERT(ret == 0, "selective scan should succeed");

    /* With A_log = -1, A = -exp(-1) = -0.368
     * A_bar = exp(delta * A) = exp(0.1 * -0.368) = exp(-0.0368) = 0.964
     * B_bar = delta * B = 0.1 * 1 = 0.1
     *
     * t=0: h = 0.964*0 + 0.1*1 = 0.1, y = 1*0.1 + 0.5*1 = 0.6
     * t=1: h = 0.964*0.1 + 0.1*1 = 0.196, y = 1*0.196 + 0.5*1 = 0.696
     * t=2: h = 0.964*0.196 + 0.1*1 = 0.289, y = 1*0.289 + 0.5*1 = 0.789 */

    /* Output should have skip connection component */
    ASSERT(y[0] > 0.5f, "y[0] should include skip connection");
    ASSERT(y[1] > y[0], "y[1] should be larger than y[0] due to state accumulation");
    ASSERT(y[2] > y[1], "y[2] should be larger than y[1]");

    printf("  PASS\n");
    return 0;
}

/* ==========================================================================
 * SILU GATE TESTS
 * ========================================================================== */

static int test_silu_gate(void) {
    printf("Testing SiLU gate...\n");

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gate[4] = {0.0f, 1.0f, 2.0f, 3.0f};  /* Different gate values */
    float output[4] = {0};

    int ret = det_silu_gate(output, x, gate, 4);
    ASSERT(ret == 0, "silu_gate should succeed");

    /* SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0 */
    ASSERT_NEAR(output[0], 0.0f, 1e-5f, "x * SiLU(0) should be 0");

    /* SiLU(1) = 1 * sigmoid(1) = 1 * 0.731 = 0.731 */
    /* x[1] * SiLU(gate[1]) = 2.0 * 0.731 = 1.462 */
    ASSERT(output[1] > 1.0f && output[1] < 2.0f, "output[1] should be in range");

    /* Increasing gate should increase output (positive x) */
    ASSERT(output[2] > output[1], "output[2] > output[1]");
    ASSERT(output[3] > output[2], "output[3] > output[2]");

    printf("  PASS\n");
    return 0;
}

/* ==========================================================================
 * CONFIG TESTS
 * ========================================================================== */

static int test_ssm_default_config(void) {
    printf("Testing SSM default config...\n");

    DetSSMConfig config = det_ssm_default_config(768);

    ASSERT(config.d_model == 768, "d_model should be 768");
    ASSERT(config.d_inner == 1536, "d_inner should be 2*d_model = 1536");
    ASSERT(config.d_state == 16, "d_state should be 16");
    ASSERT(config.d_conv == 4, "d_conv should be 4");
    ASSERT(config.dt_rank == 48, "dt_rank should be ceil(768/16) = 48");
    ASSERT_NEAR(config.dt_min, 0.001f, 1e-6f, "dt_min should be 0.001");
    ASSERT_NEAR(config.dt_max, 0.1f, 1e-6f, "dt_max should be 0.1");

    printf("  PASS\n");
    return 0;
}

/* ==========================================================================
 * STATS TESTS
 * ========================================================================== */

static int test_ssm_stats(void) {
    printf("Testing SSM stats tracking...\n");

    /* Run a selective scan to populate stats */
    int32_t d_inner = 4;
    int32_t d_state = 2;
    int32_t seq_len = 2;

    float x[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    float delta[8] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    float A[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    float B[4] = {1, 1, 1, 1};
    float C[4] = {1, 1, 1, 1};
    float D[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float h[8] = {0};
    float y[8] = {0};

    det_ssm_selective_scan(y, x, delta, A, NULL, B, C, D, h, d_inner, d_state, seq_len);

    DetSSMLayerStats stats;
    det_ssm_get_stats(&stats);

    /* Stats should be populated */
    ASSERT(stats.state_delta > 0.0f, "state_delta should be positive after scan");

    printf("  PASS\n");
    return 0;
}

/* ==========================================================================
 * MAIN
 * ========================================================================== */

int main(void) {
    printf("\n=== DET SSM Tests ===\n\n");

    int failed = 0;

    failed += test_ssm_cache_create_free();
    failed += test_ssm_cache_reset();
    failed += test_conv1d_causal_basic();
    failed += test_ssm_selective_scan_basic();
    failed += test_silu_gate();
    failed += test_ssm_default_config();
    failed += test_ssm_stats();

    printf("\n");
    if (failed == 0) {
        printf("All SSM tests passed!\n");
    } else {
        printf("%d test(s) failed\n", failed);
    }

    return failed;
}
