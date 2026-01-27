/**
 * DET Tensor Primitives - Test Suite
 * ===================================
 *
 * Phase 26.1: Tests for foundation tensor operations.
 */

#include "../include/det_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [TEST] %s... ", name)
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

#define ASSERT_EQ(a, b) do { if ((a) != (b)) { FAIL("assertion failed"); return; } } while(0)
#define ASSERT_NEAR(a, b, tol) do { if (fabsf((a) - (b)) > (tol)) { \
    printf("FAIL: %.6f != %.6f (tol=%.6f)\n", (float)(a), (float)(b), (float)(tol)); \
    tests_failed++; return; } } while(0)

/* ==========================================================================
 * TENSOR LIFECYCLE TESTS
 * ========================================================================== */

void test_tensor_create(void) {
    TEST("tensor_create");

    int32_t shape[] = {4, 8};
    DetTensor* t = det_tensor_create(2, shape, DET_DTYPE_F32);

    if (!t) { FAIL("create returned NULL"); return; }
    if (t->ndim != 2) { FAIL("wrong ndim"); det_tensor_release(t); return; }
    if (t->shape[0] != 4 || t->shape[1] != 8) { FAIL("wrong shape"); det_tensor_release(t); return; }
    if (det_tensor_numel(t) != 32) { FAIL("wrong numel"); det_tensor_release(t); return; }
    if (!t->owns_data) { FAIL("should own data"); det_tensor_release(t); return; }

    det_tensor_release(t);
    PASS();
}

void test_tensor_clone(void) {
    TEST("tensor_clone");

    int32_t shape[] = {3, 4};
    DetTensor* src = det_tensor_create(2, shape, DET_DTYPE_F32);
    float* data = (float*)src->data;

    /* Fill with test data */
    for (int i = 0; i < 12; i++) {
        data[i] = (float)i;
    }

    DetTensor* dst = det_tensor_clone(src);

    if (!dst) { FAIL("clone returned NULL"); det_tensor_release(src); return; }

    /* Verify data was copied */
    float* dst_data = (float*)dst->data;
    for (int i = 0; i < 12; i++) {
        if (dst_data[i] != (float)i) {
            FAIL("data not copied correctly");
            det_tensor_release(src);
            det_tensor_release(dst);
            return;
        }
    }

    /* Verify modifying one doesn't affect the other */
    data[0] = 999.0f;
    if (dst_data[0] != 0.0f) {
        FAIL("tensors share data (should be independent)");
        det_tensor_release(src);
        det_tensor_release(dst);
        return;
    }

    det_tensor_release(src);
    det_tensor_release(dst);
    PASS();
}

void test_tensor_from_ptr(void) {
    TEST("tensor_from_ptr");

    float data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    int32_t shape[] = {3, 4};

    DetTensor* t = det_tensor_from_ptr(data, 2, shape, DET_DTYPE_F32);

    if (!t) { FAIL("from_ptr returned NULL"); return; }
    if (t->owns_data) { FAIL("should not own data"); det_tensor_release(t); return; }
    if (t->data != data) { FAIL("data pointer mismatch"); det_tensor_release(t); return; }

    det_tensor_release(t);
    PASS();
}

/* ==========================================================================
 * MATRIX OPERATION TESTS
 * ========================================================================== */

void test_matmul(void) {
    TEST("matmul");

    /* A: [2, 3], B: [3, 4], C: [2, 4] */
    int32_t shape_a[] = {2, 3};
    int32_t shape_b[] = {3, 4};
    int32_t shape_c[] = {2, 4};

    DetTensor* A = det_tensor_create(2, shape_a, DET_DTYPE_F32);
    DetTensor* B = det_tensor_create(2, shape_b, DET_DTYPE_F32);
    DetTensor* C = det_tensor_create(2, shape_c, DET_DTYPE_F32);

    /* A = [[1, 2, 3], [4, 5, 6]] */
    float* a = (float*)A->data;
    a[0] = 1; a[1] = 2; a[2] = 3;
    a[3] = 4; a[4] = 5; a[5] = 6;

    /* B = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]] */
    float* b = (float*)B->data;
    memset(b, 0, 12 * sizeof(float));
    b[0] = 1; b[5] = 1; b[10] = 1; b[11] = 1;

    int err = det_matmul(C, A, B);
    if (err != DET_TENSOR_OK) {
        FAIL("matmul failed");
        goto cleanup;
    }

    /* C should be [[1, 2, 3, 3], [4, 5, 6, 6]] */
    float* c = (float*)C->data;
    ASSERT_NEAR(c[0], 1.0f, 1e-5f);
    ASSERT_NEAR(c[1], 2.0f, 1e-5f);
    ASSERT_NEAR(c[2], 3.0f, 1e-5f);
    ASSERT_NEAR(c[3], 3.0f, 1e-5f);
    ASSERT_NEAR(c[4], 4.0f, 1e-5f);
    ASSERT_NEAR(c[5], 5.0f, 1e-5f);
    ASSERT_NEAR(c[6], 6.0f, 1e-5f);
    ASSERT_NEAR(c[7], 6.0f, 1e-5f);

    PASS();

cleanup:
    det_tensor_release(A);
    det_tensor_release(B);
    det_tensor_release(C);
}

void test_matvec(void) {
    TEST("matvec");

    /* A: [3, 4], x: [4], y: [3] */
    int32_t shape_a[] = {3, 4};
    int32_t shape_x[] = {4};
    int32_t shape_y[] = {3};

    DetTensor* A = det_tensor_create(2, shape_a, DET_DTYPE_F32);
    DetTensor* x = det_tensor_create(1, shape_x, DET_DTYPE_F32);
    DetTensor* y = det_tensor_create(1, shape_y, DET_DTYPE_F32);

    /* A = identity-like (first 3 columns) */
    float* a = (float*)A->data;
    memset(a, 0, 12 * sizeof(float));
    a[0] = 1; a[5] = 1; a[10] = 1;

    /* x = [1, 2, 3, 4] */
    float* xp = (float*)x->data;
    xp[0] = 1; xp[1] = 2; xp[2] = 3; xp[3] = 4;

    int err = det_matvec(y, A, x);
    if (err != DET_TENSOR_OK) {
        FAIL("matvec failed");
        goto cleanup;
    }

    /* y should be [1, 2, 3] */
    float* yp = (float*)y->data;
    ASSERT_NEAR(yp[0], 1.0f, 1e-5f);
    ASSERT_NEAR(yp[1], 2.0f, 1e-5f);
    ASSERT_NEAR(yp[2], 3.0f, 1e-5f);

    PASS();

cleanup:
    det_tensor_release(A);
    det_tensor_release(x);
    det_tensor_release(y);
}

/* ==========================================================================
 * ELEMENT-WISE OPERATION TESTS
 * ========================================================================== */

void test_add(void) {
    TEST("add");

    int32_t shape[] = {4};
    DetTensor* A = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* B = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* C = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* a = (float*)A->data;
    float* b = (float*)B->data;

    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
    b[0] = 5; b[1] = 6; b[2] = 7; b[3] = 8;

    int err = det_add(C, A, B);
    if (err != DET_TENSOR_OK) { FAIL("add failed"); goto cleanup; }

    float* c = (float*)C->data;
    ASSERT_NEAR(c[0], 6.0f, 1e-5f);
    ASSERT_NEAR(c[1], 8.0f, 1e-5f);
    ASSERT_NEAR(c[2], 10.0f, 1e-5f);
    ASSERT_NEAR(c[3], 12.0f, 1e-5f);

    PASS();

cleanup:
    det_tensor_release(A);
    det_tensor_release(B);
    det_tensor_release(C);
}

void test_mul(void) {
    TEST("mul");

    int32_t shape[] = {4};
    DetTensor* A = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* B = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* C = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* a = (float*)A->data;
    float* b = (float*)B->data;

    a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
    b[0] = 2; b[1] = 3; b[2] = 4; b[3] = 5;

    int err = det_mul(C, A, B);
    if (err != DET_TENSOR_OK) { FAIL("mul failed"); goto cleanup; }

    float* c = (float*)C->data;
    ASSERT_NEAR(c[0], 2.0f, 1e-5f);
    ASSERT_NEAR(c[1], 6.0f, 1e-5f);
    ASSERT_NEAR(c[2], 12.0f, 1e-5f);
    ASSERT_NEAR(c[3], 20.0f, 1e-5f);

    PASS();

cleanup:
    det_tensor_release(A);
    det_tensor_release(B);
    det_tensor_release(C);
}

void test_div_scalar(void) {
    TEST("div_scalar");

    int32_t shape[] = {4};
    DetTensor* A = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* B = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* a = (float*)A->data;
    a[0] = 2; a[1] = 4; a[2] = 6; a[3] = 8;

    int err = det_div_scalar(B, A, 2.0f);
    if (err != DET_TENSOR_OK) { FAIL("div_scalar failed"); goto cleanup; }

    float* b = (float*)B->data;
    ASSERT_NEAR(b[0], 1.0f, 1e-5f);
    ASSERT_NEAR(b[1], 2.0f, 1e-5f);
    ASSERT_NEAR(b[2], 3.0f, 1e-5f);
    ASSERT_NEAR(b[3], 4.0f, 1e-5f);

    PASS();

cleanup:
    det_tensor_release(A);
    det_tensor_release(B);
}

/* ==========================================================================
 * ACTIVATION FUNCTION TESTS
 * ========================================================================== */

void test_silu(void) {
    TEST("silu");

    int32_t shape[] = {4};
    DetTensor* x = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* y = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* xp = (float*)x->data;
    xp[0] = -2; xp[1] = -1; xp[2] = 0; xp[3] = 1;

    int err = det_silu(y, x);
    if (err != DET_TENSOR_OK) { FAIL("silu failed"); goto cleanup; }

    float* yp = (float*)y->data;

    /* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
    /* SiLU(-2) ≈ -0.2384, SiLU(-1) ≈ -0.2689, SiLU(0) = 0, SiLU(1) ≈ 0.7311 */
    ASSERT_NEAR(yp[0], -0.2384f, 0.001f);
    ASSERT_NEAR(yp[1], -0.2689f, 0.001f);
    ASSERT_NEAR(yp[2], 0.0f, 1e-5f);
    ASSERT_NEAR(yp[3], 0.7311f, 0.001f);

    PASS();

cleanup:
    det_tensor_release(x);
    det_tensor_release(y);
}

void test_relu(void) {
    TEST("relu");

    int32_t shape[] = {4};
    DetTensor* x = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* y = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* xp = (float*)x->data;
    xp[0] = -2; xp[1] = -1; xp[2] = 0; xp[3] = 1;

    int err = det_relu(y, x);
    if (err != DET_TENSOR_OK) { FAIL("relu failed"); goto cleanup; }

    float* yp = (float*)y->data;
    ASSERT_NEAR(yp[0], 0.0f, 1e-5f);
    ASSERT_NEAR(yp[1], 0.0f, 1e-5f);
    ASSERT_NEAR(yp[2], 0.0f, 1e-5f);
    ASSERT_NEAR(yp[3], 1.0f, 1e-5f);

    PASS();

cleanup:
    det_tensor_release(x);
    det_tensor_release(y);
}

/* ==========================================================================
 * NORMALIZATION TESTS
 * ========================================================================== */

void test_rmsnorm(void) {
    TEST("rmsnorm");

    int32_t shape_x[] = {4};
    int32_t shape_w[] = {4};

    DetTensor* x = det_tensor_create(1, shape_x, DET_DTYPE_F32);
    DetTensor* w = det_tensor_create(1, shape_w, DET_DTYPE_F32);
    DetTensor* y = det_tensor_create(1, shape_x, DET_DTYPE_F32);

    float* xp = (float*)x->data;
    float* wp = (float*)w->data;

    xp[0] = 1; xp[1] = 2; xp[2] = 3; xp[3] = 4;
    wp[0] = 1; wp[1] = 1; wp[2] = 1; wp[3] = 1;  /* Unit weight */

    int err = det_rmsnorm(y, x, w, 1e-6f);
    if (err != DET_TENSOR_OK) { FAIL("rmsnorm failed"); goto cleanup; }

    float* yp = (float*)y->data;

    /* RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386 */
    /* Normalized: [1, 2, 3, 4] / 2.7386 ≈ [0.365, 0.730, 1.095, 1.461] */
    float rms = sqrtf(7.5f);
    ASSERT_NEAR(yp[0], 1.0f / rms, 0.01f);
    ASSERT_NEAR(yp[1], 2.0f / rms, 0.01f);
    ASSERT_NEAR(yp[2], 3.0f / rms, 0.01f);
    ASSERT_NEAR(yp[3], 4.0f / rms, 0.01f);

    PASS();

cleanup:
    det_tensor_release(x);
    det_tensor_release(w);
    det_tensor_release(y);
}

/* ==========================================================================
 * SOFTMAX AND SAMPLING TESTS
 * ========================================================================== */

void test_softmax(void) {
    TEST("softmax");

    int32_t shape[] = {4};
    DetTensor* x = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* y = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* xp = (float*)x->data;
    xp[0] = 1; xp[1] = 2; xp[2] = 3; xp[3] = 4;

    int err = det_softmax(y, x, 1.0f);
    if (err != DET_TENSOR_OK) { FAIL("softmax failed"); goto cleanup; }

    float* yp = (float*)y->data;

    /* Check probabilities sum to 1 */
    float sum = yp[0] + yp[1] + yp[2] + yp[3];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    /* Check ordering (monotonically increasing) */
    if (yp[0] >= yp[1] || yp[1] >= yp[2] || yp[2] >= yp[3]) {
        FAIL("softmax output not monotonic");
        goto cleanup;
    }

    PASS();

cleanup:
    det_tensor_release(x);
    det_tensor_release(y);
}

void test_softmax_temperature(void) {
    TEST("softmax_temperature");

    int32_t shape[] = {4};
    DetTensor* x = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* y_low = det_tensor_create(1, shape, DET_DTYPE_F32);
    DetTensor* y_high = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* xp = (float*)x->data;
    xp[0] = 1; xp[1] = 2; xp[2] = 3; xp[3] = 4;

    /* Low temperature should make distribution more peaked */
    det_softmax(y_low, x, 0.5f);
    /* High temperature should make distribution more uniform */
    det_softmax(y_high, x, 2.0f);

    float* yl = (float*)y_low->data;
    float* yh = (float*)y_high->data;

    /* Low temp: highest prob should be higher */
    if (yl[3] <= yh[3]) {
        FAIL("low temp should increase peak");
        goto cleanup;
    }

    /* High temp: lowest prob should be higher */
    if (yh[0] <= yl[0]) {
        FAIL("high temp should increase low probs");
        goto cleanup;
    }

    PASS();

cleanup:
    det_tensor_release(x);
    det_tensor_release(y_low);
    det_tensor_release(y_high);
}

void test_sample_greedy(void) {
    TEST("sample_greedy");

    int32_t shape[] = {5};
    DetTensor* probs = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* p = (float*)probs->data;
    p[0] = 0.1f; p[1] = 0.2f; p[2] = 0.4f; p[3] = 0.2f; p[4] = 0.1f;

    int32_t idx = det_sample_greedy(probs);
    if (idx != 2) {
        FAIL("greedy should return index of max");
        det_tensor_release(probs);
        return;
    }

    det_tensor_release(probs);
    PASS();
}

void test_sample_top_k(void) {
    TEST("sample_top_k");

    int32_t shape[] = {10};
    DetTensor* probs = det_tensor_create(1, shape, DET_DTYPE_F32);

    float* p = (float*)probs->data;
    for (int i = 0; i < 10; i++) {
        p[i] = (float)(i + 1) / 55.0f;  /* Normalized */
    }

    /* Sample with top-k=3, should only sample from indices 7, 8, 9 */
    int counts[10] = {0};
    for (int trial = 0; trial < 100; trial++) {
        int32_t idx = det_sample_top_k(probs, 3, (uint64_t)(trial + 1));
        if (idx >= 0 && idx < 10) counts[idx]++;
    }

    /* Most samples should be from top 3 */
    int top3_count = counts[7] + counts[8] + counts[9];
    if (top3_count < 90) {
        FAIL("top-k should sample mostly from top k");
        det_tensor_release(probs);
        return;
    }

    det_tensor_release(probs);
    PASS();
}

/* ==========================================================================
 * ATTENTION TESTS
 * ========================================================================== */

void test_attention_scores(void) {
    TEST("attention_scores");

    /* Q: [1, 2, 4], K: [1, 3, 4] -> scores: [1, 2, 3] */
    int32_t shape_q[] = {1, 2, 4};
    int32_t shape_k[] = {1, 3, 4};
    int32_t shape_s[] = {1, 2, 3};

    DetTensor* Q = det_tensor_create(3, shape_q, DET_DTYPE_F32);
    DetTensor* K = det_tensor_create(3, shape_k, DET_DTYPE_F32);
    DetTensor* S = det_tensor_create(3, shape_s, DET_DTYPE_F32);

    /* Simple test: Q and K both have unit vectors */
    float* q = (float*)Q->data;
    float* k = (float*)K->data;
    memset(q, 0, 8 * sizeof(float));
    memset(k, 0, 12 * sizeof(float));

    /* Q[0,0] = [1,0,0,0], Q[0,1] = [0,1,0,0] */
    q[0] = 1.0f;
    q[5] = 1.0f;

    /* K[0,0] = [1,0,0,0], K[0,1] = [0,1,0,0], K[0,2] = [0,0,1,0] */
    k[0] = 1.0f;
    k[5] = 1.0f;
    k[10] = 1.0f;

    int err = det_attention_scores(S, Q, K);
    if (err != DET_TENSOR_OK) { FAIL("attention_scores failed"); goto cleanup; }

    float* s = (float*)S->data;

    /* Expected: S[0,i,j] = Q[0,i] · K[0,j] / sqrt(4) */
    /* S[0,0] = [1,0,0] / 2 = [0.5, 0, 0] */
    /* S[0,1] = [0,1,0] / 2 = [0, 0.5, 0] */
    ASSERT_NEAR(s[0], 0.5f, 1e-5f);  /* Q0 · K0 */
    ASSERT_NEAR(s[1], 0.0f, 1e-5f);  /* Q0 · K1 */
    ASSERT_NEAR(s[2], 0.0f, 1e-5f);  /* Q0 · K2 */
    ASSERT_NEAR(s[3], 0.0f, 1e-5f);  /* Q1 · K0 */
    ASSERT_NEAR(s[4], 0.5f, 1e-5f);  /* Q1 · K1 */
    ASSERT_NEAR(s[5], 0.0f, 1e-5f);  /* Q1 · K2 */

    PASS();

cleanup:
    det_tensor_release(Q);
    det_tensor_release(K);
    det_tensor_release(S);
}

void test_causal_mask(void) {
    TEST("causal_mask");

    int32_t shape[] = {3, 3};
    DetTensor* S = det_tensor_create(2, shape, DET_DTYPE_F32);

    /* Fill with ones */
    float* s = (float*)S->data;
    for (int i = 0; i < 9; i++) s[i] = 1.0f;

    int err = det_causal_mask(S);
    if (err != DET_TENSOR_OK) { FAIL("causal_mask failed"); goto cleanup; }

    /* Check lower triangle is preserved, upper triangle is -inf */
    ASSERT_NEAR(s[0], 1.0f, 1e-5f);      /* [0,0] */
    if (s[1] > -1e8f) { FAIL("[0,1] should be -inf"); goto cleanup; }
    if (s[2] > -1e8f) { FAIL("[0,2] should be -inf"); goto cleanup; }
    ASSERT_NEAR(s[3], 1.0f, 1e-5f);      /* [1,0] */
    ASSERT_NEAR(s[4], 1.0f, 1e-5f);      /* [1,1] */
    if (s[5] > -1e8f) { FAIL("[1,2] should be -inf"); goto cleanup; }
    ASSERT_NEAR(s[6], 1.0f, 1e-5f);      /* [2,0] */
    ASSERT_NEAR(s[7], 1.0f, 1e-5f);      /* [2,1] */
    ASSERT_NEAR(s[8], 1.0f, 1e-5f);      /* [2,2] */

    PASS();

cleanup:
    det_tensor_release(S);
}

/* ==========================================================================
 * Q8_0 QUANTIZATION-AWARE MATMUL TESTS
 * ========================================================================== */

/* Helper to convert float to F16 (half precision) */
static uint16_t float_to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);

    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FF;

    if (exp <= 0) {
        /* Underflow to zero */
        return sign;
    } else if (exp >= 31) {
        /* Overflow to inf */
        return sign | 0x7C00;
    }

    return sign | (exp << 10) | mant;
}

/* Helper to quantize a row of floats to Q8_0 format
 * Input: floats[K]
 * Output: q8_data (K/32 blocks of 34 bytes each)
 */
static void quantize_row_to_q8_0(const float* floats, uint8_t* q8_data, int K) {
    int num_blocks = K / 32;

    for (int blk = 0; blk < num_blocks; blk++) {
        const float* src = floats + blk * 32;
        uint8_t* dst = q8_data + blk * 34;

        /* Find max absolute value */
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float absval = fabsf(src[i]);
            if (absval > amax) amax = absval;
        }

        /* Compute scale */
        float scale = amax / 127.0f;
        uint16_t scale_h = float_to_half(scale);
        memcpy(dst, &scale_h, 2);

        /* Quantize values */
        int8_t* q = (int8_t*)(dst + 2);
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        for (int i = 0; i < 32; i++) {
            int32_t v = (int32_t)roundf(src[i] * inv_scale);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            q[i] = (int8_t)v;
        }
    }
}

void test_matmul_q8_0_transposed(void) {
    TEST("matmul_q8_0_transposed");

    /* Create test matrices:
     * X: [M, K] = [2, 64] (K must be multiple of 32)
     * W: [N, K] = [3, 64] quantized to Q8_0
     * Y: [M, N] = [2, 3]
     */
    int M = 2, N = 3, K = 64;

    /* Allocate F32 matrices */
    float* X = malloc(M * K * sizeof(float));
    float* W_f32 = malloc(N * K * sizeof(float));
    float* Y_f32 = malloc(M * N * sizeof(float));
    float* Y_q8 = malloc(M * N * sizeof(float));

    /* Q8_0 storage: N rows, each row has K/32 blocks of 34 bytes */
    int blocks_per_row = K / 32;
    uint8_t* W_q8 = malloc(N * blocks_per_row * 34);

    if (!X || !W_f32 || !Y_f32 || !Y_q8 || !W_q8) {
        FAIL("allocation failed");
        free(X); free(W_f32); free(Y_f32); free(Y_q8); free(W_q8);
        return;
    }

    /* Initialize with test data */
    for (int i = 0; i < M * K; i++) {
        X[i] = (float)(i % 10) * 0.1f - 0.5f;
    }
    for (int i = 0; i < N * K; i++) {
        W_f32[i] = (float)((i + 3) % 7) * 0.2f - 0.6f;
    }

    /* Quantize W to Q8_0 */
    for (int n = 0; n < N; n++) {
        quantize_row_to_q8_0(W_f32 + n * K, W_q8 + n * blocks_per_row * 34, K);
    }

    /* Compute reference Y_f32 = X @ W^T using F32 */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += X[m * K + k] * W_f32[n * K + k];
            }
            Y_f32[m * N + n] = sum;
        }
    }

    /* Compute Y_q8 = X @ W_q8^T using Q8_0 matmul */
    int err = det_matmul_q8_0_transposed(Y_q8, X, W_q8, M, N, K);
    if (err != DET_TENSOR_OK) {
        FAIL("det_matmul_q8_0_transposed failed");
        goto cleanup;
    }

    /* Compare results - allow for quantization error (typically <1% for Q8_0) */
    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err_val = fabsf(Y_f32[i] - Y_q8[i]);
        float rel_err = (fabsf(Y_f32[i]) > 1e-6f) ? err_val / fabsf(Y_f32[i]) : err_val;
        if (err_val > max_err) max_err = err_val;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    /* Q8_0 should have <5% relative error on average */
    if (max_rel_err > 0.1f) {  /* 10% tolerance for edge cases */
        printf("FAIL: max_err=%.6f, max_rel_err=%.2f%%\n", max_err, max_rel_err * 100.0f);
        tests_failed++;
        goto cleanup;
    }

    printf("(max_err=%.6f, rel=%.2f%%) ", max_err, max_rel_err * 100.0f);
    PASS();

cleanup:
    free(X);
    free(W_f32);
    free(Y_f32);
    free(Y_q8);
    free(W_q8);
}

void test_matmul_q8_0_batched(void) {
    TEST("matmul_q8_0_batched");

    /* Larger test for batched version */
    int M = 4, N = 128, K = 256;

    float* X = malloc(M * K * sizeof(float));
    float* W_f32 = malloc(N * K * sizeof(float));
    float* Y_f32 = malloc(M * N * sizeof(float));
    float* Y_q8 = malloc(M * N * sizeof(float));

    int blocks_per_row = K / 32;
    uint8_t* W_q8 = malloc(N * blocks_per_row * 34);

    if (!X || !W_f32 || !Y_f32 || !Y_q8 || !W_q8) {
        FAIL("allocation failed");
        free(X); free(W_f32); free(Y_f32); free(Y_q8); free(W_q8);
        return;
    }

    /* Initialize with deterministic data that avoids zero values */
    for (int i = 0; i < M * K; i++) {
        X[i] = 0.1f + 0.01f * (float)(i % 37);  /* Range: 0.1 to 0.46 */
    }
    for (int i = 0; i < N * K; i++) {
        W_f32[i] = 0.1f + 0.01f * (float)(i % 41);  /* Range: 0.1 to 0.50 */
    }

    /* Quantize W to Q8_0 */
    for (int n = 0; n < N; n++) {
        quantize_row_to_q8_0(W_f32 + n * K, W_q8 + n * blocks_per_row * 34, K);
    }

    /* Compute reference */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += X[m * K + k] * W_f32[n * K + k];
            }
            Y_f32[m * N + n] = sum;
        }
    }

    /* Compute using batched Q8_0 matmul */
    int err = det_matmul_q8_0_transposed_batched(Y_q8, X, W_q8, M, N, K);
    if (err != DET_TENSOR_OK) {
        FAIL("det_matmul_q8_0_transposed_batched failed");
        goto cleanup;
    }

    /* Compare with tolerance */
    float max_rel_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err_val = fabsf(Y_f32[i] - Y_q8[i]);
        float rel_err = (fabsf(Y_f32[i]) > 1e-6f) ? err_val / fabsf(Y_f32[i]) : err_val;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    if (max_rel_err > 0.1f) {
        printf("FAIL: max_rel_err=%.2f%%\n", max_rel_err * 100.0f);
        tests_failed++;
        goto cleanup;
    }

    printf("(rel_err=%.2f%%) ", max_rel_err * 100.0f);
    PASS();

cleanup:
    free(X);
    free(W_f32);
    free(Y_f32);
    free(Y_q8);
    free(W_q8);
}

/* ==========================================================================
 * WORKSPACE TESTS
 * ========================================================================== */

void test_workspace(void) {
    TEST("workspace");

    size_t sizes[] = {1024, 2048, 4096};
    DetTensorWorkspace* ws = det_workspace_create(sizes, 3);

    if (!ws) { FAIL("workspace create failed"); return; }

    /* Get a scratch buffer */
    int32_t shape[] = {64, 64};
    DetTensor* scratch = det_workspace_get_scratch(ws, 0, 2, shape, DET_DTYPE_F32);

    if (!scratch) { FAIL("get_scratch failed"); det_workspace_destroy(ws); return; }
    if (scratch->shape[0] != 64 || scratch->shape[1] != 64) {
        FAIL("wrong scratch shape");
        det_workspace_destroy(ws);
        return;
    }

    det_workspace_destroy(ws);
    PASS();
}

/* ==========================================================================
 * MAIN
 * ========================================================================== */

int main(void) {
    printf("============================================================\n");
    printf("Phase 26.1: DET Tensor Primitives - Test Suite\n");
    printf("============================================================\n\n");

    printf("Tensor Lifecycle:\n");
    test_tensor_create();
    test_tensor_clone();
    test_tensor_from_ptr();

    printf("\nMatrix Operations:\n");
    test_matmul();
    test_matvec();

    printf("\nElement-wise Operations:\n");
    test_add();
    test_mul();
    test_div_scalar();

    printf("\nActivation Functions:\n");
    test_silu();
    test_relu();

    printf("\nNormalization:\n");
    test_rmsnorm();

    printf("\nSoftmax and Sampling:\n");
    test_softmax();
    test_softmax_temperature();
    test_sample_greedy();
    test_sample_top_k();

    printf("\nAttention Operations:\n");
    test_attention_scores();
    test_causal_mask();

    printf("\nQ8_0 Quantization-Aware Matmul:\n");
    test_matmul_q8_0_transposed();
    test_matmul_q8_0_batched();

    printf("\nWorkspace:\n");
    test_workspace();

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
