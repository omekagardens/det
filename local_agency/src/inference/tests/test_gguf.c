/**
 * GGUF Loader Tests - Phase 26.2
 * ==============================
 */

#include "det_gguf.h"
#include "det_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    printf("  [TEST] %s... ", name); \
    fflush(stdout)

#define PASS() \
    do { printf("PASS\n"); tests_passed++; } while(0)

#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)

#define ASSERT(cond, msg) \
    if (!(cond)) { FAIL(msg); return; }

/* ==========================================================================
 * CREATE SYNTHETIC GGUF FILE
 * ========================================================================== */

/* Helper to write data to buffer */
typedef struct {
    uint8_t* data;
    size_t capacity;
    size_t size;
} Buffer;

static void buf_init(Buffer* b, size_t cap) {
    b->data = malloc(cap);
    b->capacity = cap;
    b->size = 0;
}

static void buf_free(Buffer* b) {
    free(b->data);
}

static void buf_write(Buffer* b, const void* data, size_t n) {
    if (b->size + n > b->capacity) {
        b->capacity = (b->capacity + n) * 2;
        b->data = realloc(b->data, b->capacity);
    }
    memcpy(b->data + b->size, data, n);
    b->size += n;
}

static void buf_write_u8(Buffer* b, uint8_t v) { buf_write(b, &v, 1); }
static void buf_write_u32(Buffer* b, uint32_t v) { buf_write(b, &v, 4); }
static void buf_write_u64(Buffer* b, uint64_t v) { buf_write(b, &v, 8); }
static void buf_write_f32(Buffer* b, float v) { buf_write(b, &v, 4); }

static void buf_write_string(Buffer* b, const char* s) {
    uint64_t len = strlen(s);
    buf_write_u64(b, len);
    buf_write(b, s, len);
}

/* Create a minimal valid GGUF file for testing */
static const char* create_test_gguf(size_t* out_size) {
    static char temp_path[] = "/tmp/det_test_XXXXXX.gguf";

    Buffer b;
    buf_init(&b, 4096);

    /* Header */
    buf_write_u32(&b, 0x46554747);  /* Magic "GGUF" */
    buf_write_u32(&b, 3);           /* Version 3 */
    buf_write_u64(&b, 2);           /* 2 tensors */
    buf_write_u64(&b, 3);           /* 3 metadata entries */

    /* Metadata */
    /* 1. general.architecture = "llama" */
    buf_write_string(&b, "general.architecture");
    buf_write_u32(&b, GGUF_TYPE_STRING);
    buf_write_string(&b, "llama");

    /* 2. llama.embedding_length = 128 */
    buf_write_string(&b, "llama.embedding_length");
    buf_write_u32(&b, GGUF_TYPE_UINT32);
    buf_write_u32(&b, 128);

    /* 3. llama.block_count = 4 */
    buf_write_string(&b, "llama.block_count");
    buf_write_u32(&b, GGUF_TYPE_UINT32);
    buf_write_u32(&b, 4);

    /* Tensor infos */
    /* 1. test_tensor: [4, 8] F32 */
    buf_write_string(&b, "test_tensor");
    buf_write_u32(&b, 2);           /* ndim */
    buf_write_u64(&b, 4);           /* shape[0] */
    buf_write_u64(&b, 8);           /* shape[1] */
    buf_write_u32(&b, GGUF_TENSOR_F32);  /* type */
    buf_write_u64(&b, 0);           /* offset */

    /* 2. bias_tensor: [8] F32 */
    buf_write_string(&b, "bias_tensor");
    buf_write_u32(&b, 1);           /* ndim */
    buf_write_u64(&b, 8);           /* shape[0] */
    buf_write_u32(&b, GGUF_TENSOR_F32);  /* type */
    buf_write_u64(&b, 4 * 8 * sizeof(float));  /* offset (after first tensor) */

    /* Align to 32 bytes */
    while (b.size % 32 != 0) {
        buf_write_u8(&b, 0);
    }

    /* Tensor data */
    /* test_tensor: 4x8 floats */
    for (int i = 0; i < 32; i++) {
        buf_write_f32(&b, (float)i * 0.1f);
    }

    /* bias_tensor: 8 floats */
    for (int i = 0; i < 8; i++) {
        buf_write_f32(&b, (float)i + 0.5f);
    }

    /* Write to temp file */
    int fd = mkstemps(temp_path, 5);
    if (fd < 0) {
        buf_free(&b);
        return NULL;
    }

    write(fd, b.data, b.size);
    close(fd);

    *out_size = b.size;
    buf_free(&b);

    /* Return allocated copy of path */
    char* path = malloc(strlen(temp_path) + 1);
    strcpy(path, temp_path);
    return path;
}

/* ==========================================================================
 * TESTS
 * ========================================================================== */

static const char* g_test_path = NULL;
static size_t g_test_size = 0;

static void test_gguf_open(void) {
    TEST("gguf_open");

    GgufContext* ctx = gguf_open(g_test_path);
    ASSERT(ctx != NULL, "Failed to open GGUF file");
    ASSERT(ctx->version == 3, "Wrong version");
    ASSERT(ctx->tensor_count == 2, "Wrong tensor count");
    ASSERT(ctx->metadata_count == 3, "Wrong metadata count");

    gguf_close(ctx);
    PASS();
}

static void test_gguf_metadata(void) {
    TEST("gguf_get_metadata");

    GgufContext* ctx = gguf_open(g_test_path);
    ASSERT(ctx != NULL, "Failed to open GGUF file");

    const char* arch = gguf_get_string(ctx, "general.architecture");
    ASSERT(arch != NULL, "Architecture not found");
    ASSERT(strcmp(arch, "llama") == 0, "Wrong architecture");

    uint32_t n_embd = gguf_get_u32(ctx, "llama.embedding_length", 0);
    ASSERT(n_embd == 128, "Wrong embedding length");

    uint32_t n_layer = gguf_get_u32(ctx, "llama.block_count", 0);
    ASSERT(n_layer == 4, "Wrong block count");

    gguf_close(ctx);
    PASS();
}

static void test_gguf_tensor_info(void) {
    TEST("gguf_get_tensor_info");

    GgufContext* ctx = gguf_open(g_test_path);
    ASSERT(ctx != NULL, "Failed to open GGUF file");

    const GgufTensorInfo* info = gguf_get_tensor_info(ctx, "test_tensor");
    ASSERT(info != NULL, "Tensor info not found");
    ASSERT(info->ndim == 2, "Wrong ndim");
    ASSERT(info->shape[0] == 4, "Wrong shape[0]");
    ASSERT(info->shape[1] == 8, "Wrong shape[1]");
    ASSERT(info->type == GGUF_TENSOR_F32, "Wrong type");

    const GgufTensorInfo* info2 = gguf_get_tensor_info(ctx, "bias_tensor");
    ASSERT(info2 != NULL, "Bias tensor info not found");
    ASSERT(info2->ndim == 1, "Wrong ndim for bias");
    ASSERT(info2->shape[0] == 8, "Wrong shape for bias");

    gguf_close(ctx);
    PASS();
}

static void test_gguf_get_tensor(void) {
    TEST("gguf_get_tensor");

    GgufContext* ctx = gguf_open(g_test_path);
    ASSERT(ctx != NULL, "Failed to open GGUF file");

    DetTensor* t = gguf_get_tensor(ctx, "test_tensor");
    ASSERT(t != NULL, "Failed to get tensor");
    ASSERT(t->ndim == 2, "Wrong ndim");
    ASSERT(t->shape[0] == 4, "Wrong shape[0]");
    ASSERT(t->shape[1] == 8, "Wrong shape[1]");
    ASSERT(t->dtype == DET_DTYPE_F32, "Wrong dtype");
    ASSERT(t->storage == DET_STORAGE_MMAP, "Wrong storage");

    /* Verify data */
    float* data = (float*)t->data;
    ASSERT(fabsf(data[0] - 0.0f) < 1e-5f, "Wrong data[0]");
    ASSERT(fabsf(data[1] - 0.1f) < 1e-5f, "Wrong data[1]");
    ASSERT(fabsf(data[31] - 3.1f) < 1e-5f, "Wrong data[31]");

    det_tensor_release(t);
    gguf_close(ctx);
    PASS();
}

static void test_gguf_model_params(void) {
    TEST("model parameters");

    GgufContext* ctx = gguf_open(g_test_path);
    ASSERT(ctx != NULL, "Failed to open GGUF file");

    /* Check convenience accessors */
    ASSERT(ctx->n_embd == 128, "Wrong n_embd");
    ASSERT(ctx->n_layer == 4, "Wrong n_layer");
    ASSERT(ctx->model_arch != NULL, "model_arch is NULL");
    ASSERT(strcmp(ctx->model_arch, "llama") == 0, "Wrong model_arch");

    /* Test architecture detection */
    DetModelArch arch = gguf_detect_arch(ctx);
    ASSERT(arch == DET_ARCH_LLAMA, "Wrong detected architecture");
    ASSERT(strcmp(det_arch_name(arch), "llama") == 0, "Wrong arch name");

    gguf_close(ctx);
    PASS();
}

static void test_gguf_type_utils(void) {
    TEST("type utilities");

    ASSERT(gguf_tensor_type_size(GGUF_TENSOR_F32) == 4, "Wrong F32 size");
    ASSERT(gguf_tensor_type_size(GGUF_TENSOR_F16) == 2, "Wrong F16 size");
    ASSERT(gguf_tensor_block_size(GGUF_TENSOR_Q8_0) == 32, "Wrong Q8_0 block size");
    ASSERT(gguf_tensor_block_size(GGUF_TENSOR_Q4_K) == 256, "Wrong Q4_K block size");

    ASSERT(gguf_type_to_det(GGUF_TENSOR_F32) == DET_DTYPE_F32, "Wrong F32 conversion");
    ASSERT(gguf_type_to_det(GGUF_TENSOR_F16) == DET_DTYPE_F16, "Wrong F16 conversion");
    ASSERT(gguf_type_to_det(GGUF_TENSOR_BF16) == DET_DTYPE_BF16, "Wrong BF16 conversion");

    ASSERT(strcmp(gguf_tensor_type_name(GGUF_TENSOR_F32), "F32") == 0, "Wrong F32 name");
    ASSERT(strcmp(gguf_tensor_type_name(GGUF_TENSOR_Q4_K), "Q4_K") == 0, "Wrong Q4_K name");

    PASS();
}

static void test_gguf_error_handling(void) {
    TEST("error handling");

    /* Test opening non-existent file */
    GgufContext* ctx = gguf_open("/nonexistent/path/model.gguf");
    ASSERT(ctx == NULL, "Should fail on non-existent file");

    /* Test error string */
    const char* err = gguf_strerror(GGUF_ERR_IO);
    ASSERT(err != NULL, "Error string is NULL");
    ASSERT(strlen(err) > 0, "Error string is empty");

    PASS();
}

/* ==========================================================================
 * MAIN
 * ========================================================================== */

int main(void) {
    printf("============================================================\n");
    printf("Phase 26.2: GGUF Loader - Test Suite\n");
    printf("============================================================\n\n");

    /* Create test GGUF file */
    g_test_path = create_test_gguf(&g_test_size);
    if (!g_test_path) {
        printf("ERROR: Failed to create test GGUF file\n");
        return 1;
    }
    printf("Created test GGUF: %s (%zu bytes)\n\n", g_test_path, g_test_size);

    printf("GGUF File Loading:\n");
    test_gguf_open();
    test_gguf_metadata();
    test_gguf_tensor_info();
    test_gguf_get_tensor();

    printf("\nModel Parameters:\n");
    test_gguf_model_params();

    printf("\nUtilities:\n");
    test_gguf_type_utils();
    test_gguf_error_handling();

    /* Cleanup */
    unlink(g_test_path);
    free((void*)g_test_path);

    printf("\n============================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
