/**
 * DET Tensor Primitives - Implementation
 * ======================================
 *
 * Phase 26.1: Foundation primitives for DET-native model inference.
 *
 * Uses Accelerate framework on macOS for BLAS operations.
 */

#include "../include/det_tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <dlfcn.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE 1
#endif

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ==========================================================================
 * DTYPE UTILITIES
 * ========================================================================== */

size_t det_dtype_size(DetDType dtype) {
    switch (dtype) {
        case DET_DTYPE_F32:   return sizeof(float);
        case DET_DTYPE_F16:   return 2;
        case DET_DTYPE_BF16:  return 2;
        case DET_DTYPE_Q8_0:  return 1;  /* Average, actual varies by block */
        case DET_DTYPE_Q4_K_M: return 1; /* Average */
        case DET_DTYPE_Q4_0:  return 1;  /* Average */
        case DET_DTYPE_I32:   return sizeof(int32_t);
        case DET_DTYPE_I16:   return sizeof(int16_t);
        default:              return 0;
    }
}

const char* det_dtype_name(DetDType dtype) {
    switch (dtype) {
        case DET_DTYPE_F32:    return "float32";
        case DET_DTYPE_F16:    return "float16";
        case DET_DTYPE_BF16:   return "bfloat16";
        case DET_DTYPE_Q8_0:   return "q8_0";
        case DET_DTYPE_Q4_K_M: return "q4_k_m";
        case DET_DTYPE_Q4_0:   return "q4_0";
        case DET_DTYPE_I32:    return "int32";
        case DET_DTYPE_I16:    return "int16";
        default:               return "unknown";
    }
}

const char* det_tensor_strerror(int err) {
    switch (err) {
        case DET_TENSOR_OK:        return "OK";
        case DET_TENSOR_ERR_ALLOC: return "Allocation failed";
        case DET_TENSOR_ERR_SHAPE: return "Shape mismatch";
        case DET_TENSOR_ERR_DTYPE: return "Unsupported dtype";
        case DET_TENSOR_ERR_IO:    return "I/O error";
        case DET_TENSOR_ERR_GPU:   return "GPU error";
        case DET_TENSOR_ERR_INVALID: return "Invalid argument";
        default:                   return "Unknown error";
    }
}

/* ==========================================================================
 * TENSOR LIFECYCLE
 * ========================================================================== */

static void compute_strides(DetTensor* t) {
    /* Row-major (C order) strides */
    t->stride[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; i--) {
        t->stride[i] = t->stride[i + 1] * t->shape[i + 1];
    }
}

DetTensor* det_tensor_create(int32_t ndim, const int32_t* shape, DetDType dtype) {
    if (ndim < 1 || ndim > DET_MAX_DIMS || !shape) return NULL;

    DetTensor* t = (DetTensor*)calloc(1, sizeof(DetTensor));
    if (!t) return NULL;

    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int32_t));
    t->dtype = dtype;
    t->storage = DET_STORAGE_CPU;
    t->owns_data = true;
    t->refcount = 1;
    t->scale = 1.0f;
    t->zero_point = 0;
    t->fd = -1;

    compute_strides(t);

    /* Calculate data size */
    size_t numel = det_tensor_numel(t);
    t->data_size = numel * det_dtype_size(dtype);

    /* Allocate aligned memory */
    #ifdef __APPLE__
    posix_memalign(&t->data, 64, t->data_size);  /* 64-byte alignment for SIMD */
    #else
    t->data = aligned_alloc(64, t->data_size);
    #endif

    if (!t->data) {
        free(t);
        return NULL;
    }

    memset(t->data, 0, t->data_size);
    return t;
}

DetTensor* det_tensor_from_ptr(void* data, int32_t ndim, const int32_t* shape, DetDType dtype) {
    if (!data || ndim < 1 || ndim > DET_MAX_DIMS || !shape) return NULL;

    DetTensor* t = (DetTensor*)calloc(1, sizeof(DetTensor));
    if (!t) return NULL;

    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int32_t));
    t->dtype = dtype;
    t->storage = DET_STORAGE_CPU;
    t->owns_data = false;  /* Does NOT own the data */
    t->refcount = 1;
    t->scale = 1.0f;
    t->zero_point = 0;
    t->fd = -1;
    t->data = data;

    compute_strides(t);
    t->data_size = det_tensor_numel(t) * det_dtype_size(dtype);

    return t;
}

DetTensor* det_tensor_view(DetTensor* src, int32_t offset, int32_t ndim, const int32_t* shape) {
    if (!src || !shape) return NULL;

    DetTensor* t = (DetTensor*)calloc(1, sizeof(DetTensor));
    if (!t) return NULL;

    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int32_t));
    t->dtype = src->dtype;
    t->storage = src->storage;
    t->owns_data = false;
    t->refcount = 1;
    t->scale = src->scale;
    t->zero_point = src->zero_point;
    t->fd = -1;

    compute_strides(t);

    /* Point to offset in source data */
    t->data = (char*)src->data + offset * det_dtype_size(src->dtype);
    t->data_size = det_tensor_numel(t) * det_dtype_size(t->dtype);

    /* Retain source to prevent premature free */
    det_tensor_retain(src);

    return t;
}

DetTensor* det_tensor_mmap(const char* path, size_t offset, int32_t ndim,
                            const int32_t* shape, DetDType dtype) {
    if (!path || !shape) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    DetTensor* t = (DetTensor*)calloc(1, sizeof(DetTensor));
    if (!t) {
        close(fd);
        return NULL;
    }

    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(int32_t));
    t->dtype = dtype;
    t->storage = DET_STORAGE_MMAP;
    t->owns_data = true;  /* Will munmap on release */
    t->refcount = 1;
    t->scale = 1.0f;
    t->zero_point = 0;
    t->fd = fd;
    t->file_offset = offset;

    compute_strides(t);
    t->data_size = det_tensor_numel(t) * det_dtype_size(dtype);

    /* Memory-map the region */
    size_t map_size = t->data_size + offset;
    void* mapped = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        free(t);
        return NULL;
    }

    t->data = (char*)mapped + offset;
    return t;
}

DetTensor* det_tensor_clone(const DetTensor* src) {
    if (!src) return NULL;

    DetTensor* dst = det_tensor_create(src->ndim, src->shape, src->dtype);
    if (!dst) return NULL;

    memcpy(dst->data, src->data, src->data_size);
    dst->scale = src->scale;
    dst->zero_point = src->zero_point;

    return dst;
}

void det_tensor_retain(DetTensor* t) {
    if (t) t->refcount++;
}

void det_tensor_release(DetTensor* t) {
    if (!t) return;

    t->refcount--;
    if (t->refcount > 0) return;

    if (t->owns_data && t->data) {
        if (t->storage == DET_STORAGE_MMAP) {
            /* Unmap the memory */
            void* base = (char*)t->data - t->file_offset;
            munmap(base, t->data_size + t->file_offset);
            if (t->fd >= 0) close(t->fd);
        } else if (t->storage == DET_STORAGE_CPU) {
            free(t->data);
        }
        /* GPU storage handled by Metal layer */
    }

    free(t);
}

bool det_tensor_is_contiguous(const DetTensor* t) {
    if (!t) return false;

    int32_t expected_stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->stride[i] != expected_stride) return false;
        expected_stride *= t->shape[i];
    }
    return true;
}

int det_tensor_copy(DetTensor* dst, const DetTensor* src) {
    if (!dst || !src) return DET_TENSOR_ERR_INVALID;
    if (dst->data_size != src->data_size) return DET_TENSOR_ERR_SHAPE;

    memcpy(dst->data, src->data, src->data_size);
    return DET_TENSOR_OK;
}

/* ==========================================================================
 * MATRIX OPERATIONS (BLAS)
 * ========================================================================== */

int det_matmul(DetTensor* C, const DetTensor* A, const DetTensor* B) {
    if (!C || !A || !B) return DET_TENSOR_ERR_INVALID;

    /* Currently only support float32 */
    if (A->dtype != DET_DTYPE_F32 || B->dtype != DET_DTYPE_F32 || C->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    /* A: [M, K], B: [K, N], C: [M, N] */
    if (A->ndim != 2 || B->ndim != 2 || C->ndim != 2) {
        return DET_TENSOR_ERR_SHAPE;
    }

    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];

    if (B->shape[0] != K || C->shape[0] != M || C->shape[1] != N) {
        return DET_TENSOR_ERR_SHAPE;
    }

#ifdef USE_ACCELERATE
    /* Use Accelerate BLAS (row-major) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,                           /* alpha */
                (const float*)A->data, K,       /* A, lda */
                (const float*)B->data, N,       /* B, ldb */
                0.0f,                           /* beta */
                (float*)C->data, N);            /* C, ldc */
#else
    /* Naive fallback */
    float* a = (float*)A->data;
    float* b = (float*)B->data;
    float* c = (float*)C->data;

    memset(c, 0, M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float aik = a[i * K + k];
            for (int j = 0; j < N; j++) {
                c[i * N + j] += aik * b[k * N + j];
            }
        }
    }
#endif

    return DET_TENSOR_OK;
}

int det_matvec(DetTensor* y, const DetTensor* A, const DetTensor* x) {
    if (!y || !A || !x) return DET_TENSOR_ERR_INVALID;

    if (A->dtype != DET_DTYPE_F32 || x->dtype != DET_DTYPE_F32 || y->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    /* A: [M, N], x: [N], y: [M] */
    if (A->ndim != 2 || x->ndim != 1 || y->ndim != 1) {
        return DET_TENSOR_ERR_SHAPE;
    }

    int M = A->shape[0];
    int N = A->shape[1];

    if (x->shape[0] != N || y->shape[0] != M) {
        return DET_TENSOR_ERR_SHAPE;
    }

#ifdef USE_ACCELERATE
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                M, N,
                1.0f,
                (const float*)A->data, N,
                (const float*)x->data, 1,
                0.0f,
                (float*)y->data, 1);
#else
    float* a = (float*)A->data;
    float* xp = (float*)x->data;
    float* yp = (float*)y->data;

    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += a[i * N + j] * xp[j];
        }
        yp[i] = sum;
    }
#endif

    return DET_TENSOR_OK;
}

/* ==========================================================================
 * ELEMENT-WISE OPERATIONS
 * ========================================================================== */

int det_add(DetTensor* C, const DetTensor* A, const DetTensor* B) {
    if (!C || !A || !B) return DET_TENSOR_ERR_INVALID;
    if (A->dtype != DET_DTYPE_F32 || B->dtype != DET_DTYPE_F32 || C->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    size_t n = det_tensor_numel(A);
    if (det_tensor_numel(B) != n || det_tensor_numel(C) != n) {
        return DET_TENSOR_ERR_SHAPE;
    }

    float* a = (float*)A->data;
    float* b = (float*)B->data;
    float* c = (float*)C->data;

#ifdef USE_ACCELERATE
    vDSP_vadd(a, 1, b, 1, c, 1, n);
#else
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
#endif

    return DET_TENSOR_OK;
}

int det_mul(DetTensor* C, const DetTensor* A, const DetTensor* B) {
    if (!C || !A || !B) return DET_TENSOR_ERR_INVALID;
    if (A->dtype != DET_DTYPE_F32 || B->dtype != DET_DTYPE_F32 || C->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    size_t n = det_tensor_numel(A);
    if (det_tensor_numel(B) != n || det_tensor_numel(C) != n) {
        return DET_TENSOR_ERR_SHAPE;
    }

    float* a = (float*)A->data;
    float* b = (float*)B->data;
    float* c = (float*)C->data;

#ifdef USE_ACCELERATE
    vDSP_vmul(a, 1, b, 1, c, 1, n);
#else
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
#endif

    return DET_TENSOR_OK;
}

int det_div_scalar(DetTensor* B, const DetTensor* A, float scalar) {
    if (!B || !A || scalar == 0.0f) return DET_TENSOR_ERR_INVALID;
    if (A->dtype != DET_DTYPE_F32 || B->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    size_t n = det_tensor_numel(A);
    if (det_tensor_numel(B) != n) return DET_TENSOR_ERR_SHAPE;

    float* a = (float*)A->data;
    float* b = (float*)B->data;

#ifdef USE_ACCELERATE
    vDSP_vsdiv(a, 1, &scalar, b, 1, n);
#else
    float inv = 1.0f / scalar;
    for (size_t i = 0; i < n; i++) {
        b[i] = a[i] * inv;
    }
#endif

    return DET_TENSOR_OK;
}

int det_mul_scalar(DetTensor* B, const DetTensor* A, float scalar) {
    if (!B || !A) return DET_TENSOR_ERR_INVALID;
    if (A->dtype != DET_DTYPE_F32 || B->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    size_t n = det_tensor_numel(A);
    if (det_tensor_numel(B) != n) return DET_TENSOR_ERR_SHAPE;

    float* a = (float*)A->data;
    float* b = (float*)B->data;

#ifdef USE_ACCELERATE
    vDSP_vsmul(a, 1, &scalar, b, 1, n);
#else
    for (size_t i = 0; i < n; i++) {
        b[i] = a[i] * scalar;
    }
#endif

    return DET_TENSOR_OK;
}

int det_mul_scalar_inplace(DetTensor* A, float scalar) {
    return det_mul_scalar(A, A, scalar);
}

/* ==========================================================================
 * ACTIVATION FUNCTIONS
 * ========================================================================== */

int det_silu(DetTensor* y, const DetTensor* x) {
    if (!y || !x) return DET_TENSOR_ERR_INVALID;
    if (x->dtype != DET_DTYPE_F32 || y->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    size_t n = det_tensor_numel(x);
    if (det_tensor_numel(y) != n) return DET_TENSOR_ERR_SHAPE;

    float* xp = (float*)x->data;
    float* yp = (float*)y->data;

    /* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
    for (size_t i = 0; i < n; i++) {
        float xi = xp[i];
        yp[i] = xi / (1.0f + expf(-xi));
    }

    return DET_TENSOR_OK;
}

int det_silu_inplace(DetTensor* x) {
    return det_silu(x, x);
}

int det_gelu(DetTensor* y, const DetTensor* x) {
    if (!y || !x) return DET_TENSOR_ERR_INVALID;
    if (x->dtype != DET_DTYPE_F32 || y->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    size_t n = det_tensor_numel(x);
    if (det_tensor_numel(y) != n) return DET_TENSOR_ERR_SHAPE;

    float* xp = (float*)x->data;
    float* yp = (float*)y->data;

    /* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    const float sqrt_2_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < n; i++) {
        float xi = xp[i];
        float inner = sqrt_2_pi * (xi + coeff * xi * xi * xi);
        yp[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }

    return DET_TENSOR_OK;
}

int det_relu(DetTensor* y, const DetTensor* x) {
    if (!y || !x) return DET_TENSOR_ERR_INVALID;
    if (x->dtype != DET_DTYPE_F32 || y->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    size_t n = det_tensor_numel(x);
    if (det_tensor_numel(y) != n) return DET_TENSOR_ERR_SHAPE;

    float* xp = (float*)x->data;
    float* yp = (float*)y->data;

#ifdef USE_ACCELERATE
    float zero = 0.0f;
    vDSP_vmax(xp, 1, &zero, 0, yp, 1, n);
#else
    for (size_t i = 0; i < n; i++) {
        yp[i] = xp[i] > 0.0f ? xp[i] : 0.0f;
    }
#endif

    return DET_TENSOR_OK;
}

/* ==========================================================================
 * NORMALIZATION
 * ========================================================================== */

int det_rmsnorm(DetTensor* y, const DetTensor* x, const DetTensor* weight, float eps) {
    if (!y || !x || !weight) return DET_TENSOR_ERR_INVALID;
    if (x->dtype != DET_DTYPE_F32 || y->dtype != DET_DTYPE_F32 || weight->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }

    /* x: [N, D] or [D], weight: [D], y: [N, D] or [D] */
    int32_t D = weight->shape[0];
    int32_t N = 1;

    if (x->ndim == 2) {
        N = x->shape[0];
        if (x->shape[1] != D) return DET_TENSOR_ERR_SHAPE;
    } else if (x->ndim == 1) {
        if (x->shape[0] != D) return DET_TENSOR_ERR_SHAPE;
    } else {
        return DET_TENSOR_ERR_SHAPE;
    }

    float* xp = (float*)x->data;
    float* yp = (float*)y->data;
    float* wp = (float*)weight->data;

    for (int32_t n = 0; n < N; n++) {
        float* x_row = xp + n * D;
        float* y_row = yp + n * D;

        /* Compute RMS */
        float sum_sq = 0.0f;
#ifdef USE_ACCELERATE
        vDSP_svesq(x_row, 1, &sum_sq, D);
#else
        for (int32_t d = 0; d < D; d++) {
            sum_sq += x_row[d] * x_row[d];
        }
#endif
        float rms = sqrtf(sum_sq / (float)D + eps);
        float inv_rms = 1.0f / rms;

        /* Normalize and scale by weight */
        for (int32_t d = 0; d < D; d++) {
            y_row[d] = x_row[d] * inv_rms * wp[d];
        }
    }

    return DET_TENSOR_OK;
}

/* ==========================================================================
 * SOFTMAX AND SAMPLING
 * ========================================================================== */

int det_softmax(DetTensor* y, const DetTensor* x, float temperature) {
    if (!y || !x) return DET_TENSOR_ERR_INVALID;
    if (x->dtype != DET_DTYPE_F32 || y->dtype != DET_DTYPE_F32) {
        return DET_TENSOR_ERR_DTYPE;
    }
    if (temperature <= 0.0f) temperature = 1.0f;

    size_t n = det_tensor_numel(x);
    if (det_tensor_numel(y) != n) return DET_TENSOR_ERR_SHAPE;

    float* xp = (float*)x->data;
    float* yp = (float*)y->data;

    /* Find max for numerical stability */
    float max_val = xp[0];
    for (size_t i = 1; i < n; i++) {
        if (xp[i] > max_val) max_val = xp[i];
    }

    /* Compute exp((x - max) / temperature) and sum */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float val = expf((xp[i] - max_val) / temperature);
        yp[i] = val;
        sum += val;
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; i++) {
        yp[i] *= inv_sum;
    }

    return DET_TENSOR_OK;
}

int det_softmax_inplace(DetTensor* x, float temperature) {
    return det_softmax(x, x, temperature);
}

/* Simple xorshift64 RNG */
static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float random_uniform(uint64_t* state) {
    return (float)(xorshift64(state) >> 11) / (float)(1ULL << 53);
}

int32_t det_sample_top_p(const DetTensor* probs, float p, uint64_t seed) {
    if (!probs || probs->dtype != DET_DTYPE_F32 || probs->ndim != 1) {
        return -1;
    }

    size_t vocab_size = probs->shape[0];
    float* prob = (float*)probs->data;

    /* Create index array and sort by probability (descending) */
    /* For simplicity, use selection approach instead of full sort */
    uint64_t rng_state = seed ? seed : 42;
    float threshold = random_uniform(&rng_state);

    /* Accumulate probability until we exceed threshold * p */
    float cumsum = 0.0f;
    float target = threshold;

    /* First, find indices that make up top-p mass */
    /* Simple O(n) approach: scan and accumulate */
    int32_t* indices = (int32_t*)malloc(vocab_size * sizeof(int32_t));
    float* sorted_probs = (float*)malloc(vocab_size * sizeof(float));
    if (!indices || !sorted_probs) {
        free(indices);
        free(sorted_probs);
        return det_sample_greedy(probs);  /* Fallback */
    }

    for (size_t i = 0; i < vocab_size; i++) {
        indices[i] = (int32_t)i;
        sorted_probs[i] = prob[i];
    }

    /* Partial sort to find top-p tokens */
    /* Insertion sort for simplicity (vocab usually ~32k-128k, top-p selects few) */
    for (size_t i = 1; i < vocab_size; i++) {
        int32_t idx = indices[i];
        float val = sorted_probs[i];
        size_t j = i;
        while (j > 0 && sorted_probs[j-1] < val) {
            sorted_probs[j] = sorted_probs[j-1];
            indices[j] = indices[j-1];
            j--;
        }
        sorted_probs[j] = val;
        indices[j] = idx;
    }

    /* Accumulate until we reach p */
    size_t cutoff = 0;
    cumsum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        cumsum += sorted_probs[i];
        cutoff = i + 1;
        if (cumsum >= p) break;
    }

    /* Renormalize truncated distribution */
    float renorm_sum = 0.0f;
    for (size_t i = 0; i < cutoff; i++) {
        renorm_sum += sorted_probs[i];
    }

    /* Sample from truncated distribution */
    target *= renorm_sum;
    cumsum = 0.0f;
    int32_t sampled = indices[0];

    for (size_t i = 0; i < cutoff; i++) {
        cumsum += sorted_probs[i];
        if (cumsum >= target) {
            sampled = indices[i];
            break;
        }
    }

    free(indices);
    free(sorted_probs);

    return sampled;
}

int32_t det_sample_top_k(const DetTensor* probs, int32_t k, uint64_t seed) {
    if (!probs || probs->dtype != DET_DTYPE_F32 || probs->ndim != 1) {
        return -1;
    }

    size_t vocab_size = probs->shape[0];
    float* prob = (float*)probs->data;

    if (k <= 0 || (size_t)k > vocab_size) k = (int32_t)vocab_size;

    uint64_t rng_state = seed ? seed : 42;

    /* Find top-k by partial sort */
    int32_t* top_indices = (int32_t*)malloc(k * sizeof(int32_t));
    float* top_probs = (float*)malloc(k * sizeof(float));
    if (!top_indices || !top_probs) {
        free(top_indices);
        free(top_probs);
        return det_sample_greedy(probs);
    }

    /* Initialize with first k elements */
    for (int32_t i = 0; i < k; i++) {
        top_indices[i] = i;
        top_probs[i] = prob[i];
    }

    /* Sort initial k (insertion sort) */
    for (int32_t i = 1; i < k; i++) {
        int32_t idx = top_indices[i];
        float val = top_probs[i];
        int32_t j = i;
        while (j > 0 && top_probs[j-1] < val) {
            top_probs[j] = top_probs[j-1];
            top_indices[j] = top_indices[j-1];
            j--;
        }
        top_probs[j] = val;
        top_indices[j] = idx;
    }

    /* Scan rest and insert if larger than min in top-k */
    for (size_t i = k; i < vocab_size; i++) {
        if (prob[i] > top_probs[k-1]) {
            /* Insert into sorted list */
            int32_t j = k - 1;
            while (j > 0 && top_probs[j-1] < prob[i]) {
                top_probs[j] = top_probs[j-1];
                top_indices[j] = top_indices[j-1];
                j--;
            }
            top_probs[j] = prob[i];
            top_indices[j] = (int32_t)i;
        }
    }

    /* Renormalize and sample */
    float sum = 0.0f;
    for (int32_t i = 0; i < k; i++) {
        sum += top_probs[i];
    }

    float target = random_uniform(&rng_state) * sum;
    float cumsum = 0.0f;
    int32_t sampled = top_indices[0];

    for (int32_t i = 0; i < k; i++) {
        cumsum += top_probs[i];
        if (cumsum >= target) {
            sampled = top_indices[i];
            break;
        }
    }

    free(top_indices);
    free(top_probs);

    return sampled;
}

int32_t det_sample_greedy(const DetTensor* probs) {
    if (!probs || probs->dtype != DET_DTYPE_F32 || probs->ndim != 1) {
        return -1;
    }

    size_t n = probs->shape[0];
    float* p = (float*)probs->data;

    int32_t max_idx = 0;
    float max_val = p[0];

    for (size_t i = 1; i < n; i++) {
        if (p[i] > max_val) {
            max_val = p[i];
            max_idx = (int32_t)i;
        }
    }

    return max_idx;
}

int det_repetition_penalty(DetTensor* logits, const int32_t* token_ids,
                            int32_t context_len, float penalty) {
    if (!logits || !token_ids || penalty <= 0.0f) return DET_TENSOR_ERR_INVALID;
    if (logits->dtype != DET_DTYPE_F32) return DET_TENSOR_ERR_DTYPE;

    float* lp = (float*)logits->data;
    size_t vocab_size = det_tensor_numel(logits);

    for (int32_t i = 0; i < context_len; i++) {
        int32_t token = token_ids[i];
        if (token >= 0 && (size_t)token < vocab_size) {
            /* If logit is positive, divide by penalty; if negative, multiply */
            if (lp[token] > 0) {
                lp[token] /= penalty;
            } else {
                lp[token] *= penalty;
            }
        }
    }

    return DET_TENSOR_OK;
}

/* ==========================================================================
 * ATTENTION OPERATIONS
 * ========================================================================== */

int det_attention_scores(DetTensor* scores, const DetTensor* Q, const DetTensor* K) {
    if (!scores || !Q || !K) return DET_TENSOR_ERR_INVALID;

    /* Q: [batch, seq_q, d_k], K: [batch, seq_k, d_k] */
    if (Q->ndim != 3 || K->ndim != 3 || scores->ndim != 3) {
        return DET_TENSOR_ERR_SHAPE;
    }

    int batch = Q->shape[0];
    int seq_q = Q->shape[1];
    int d_k = Q->shape[2];
    int seq_k = K->shape[1];

    if (K->shape[0] != batch || K->shape[2] != d_k) {
        return DET_TENSOR_ERR_SHAPE;
    }
    if (scores->shape[0] != batch || scores->shape[1] != seq_q || scores->shape[2] != seq_k) {
        return DET_TENSOR_ERR_SHAPE;
    }

    float* qp = (float*)Q->data;
    float* kp = (float*)K->data;
    float* sp = (float*)scores->data;

    float scale = 1.0f / sqrtf((float)d_k);

    /* For each batch, compute Q @ K^T */
    for (int b = 0; b < batch; b++) {
        float* q_batch = qp + b * seq_q * d_k;
        float* k_batch = kp + b * seq_k * d_k;
        float* s_batch = sp + b * seq_q * seq_k;

#ifdef USE_ACCELERATE
        /* Use BLAS: C = alpha * A @ B^T + beta * C */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_q, seq_k, d_k,
                    scale,
                    q_batch, d_k,
                    k_batch, d_k,
                    0.0f,
                    s_batch, seq_k);
#else
        for (int i = 0; i < seq_q; i++) {
            for (int j = 0; j < seq_k; j++) {
                float dot = 0.0f;
                for (int k = 0; k < d_k; k++) {
                    dot += q_batch[i * d_k + k] * k_batch[j * d_k + k];
                }
                s_batch[i * seq_k + j] = dot * scale;
            }
        }
#endif
    }

    return DET_TENSOR_OK;
}

int det_causal_mask(DetTensor* scores) {
    if (!scores || scores->dtype != DET_DTYPE_F32) return DET_TENSOR_ERR_INVALID;

    /* scores: [batch, seq_q, seq_k] or [seq_q, seq_k] */
    int batch = 1;
    int seq_q, seq_k;

    if (scores->ndim == 3) {
        batch = scores->shape[0];
        seq_q = scores->shape[1];
        seq_k = scores->shape[2];
    } else if (scores->ndim == 2) {
        seq_q = scores->shape[0];
        seq_k = scores->shape[1];
    } else {
        return DET_TENSOR_ERR_SHAPE;
    }

    float* sp = (float*)scores->data;
    float neg_inf = -1e9f;

    for (int b = 0; b < batch; b++) {
        float* s_batch = sp + b * seq_q * seq_k;
        for (int i = 0; i < seq_q; i++) {
            for (int j = i + 1; j < seq_k; j++) {
                s_batch[i * seq_k + j] = neg_inf;
            }
        }
    }

    return DET_TENSOR_OK;
}

/* ==========================================================================
 * WORKSPACE
 * ========================================================================== */

DetTensorWorkspace* det_workspace_create(const size_t* sizes, int num_buffers) {
    DetTensorWorkspace* ws = (DetTensorWorkspace*)calloc(1, sizeof(DetTensorWorkspace));
    if (!ws) return NULL;

    ws->num_scratch = num_buffers;

    for (int i = 0; i < num_buffers && i < 8; i++) {
        int32_t shape[1] = { (int32_t)(sizes[i] / sizeof(float)) };
        ws->scratch[i] = det_tensor_create(1, shape, DET_DTYPE_F32);
        ws->scratch_sizes[i] = sizes[i];
    }

    return ws;
}

void det_workspace_destroy(DetTensorWorkspace* ws) {
    if (!ws) return;

    for (int i = 0; i < ws->num_scratch; i++) {
        det_tensor_release(ws->scratch[i]);
    }

    free(ws);
}

DetTensor* det_workspace_get_scratch(DetTensorWorkspace* ws, int idx,
                                      int32_t ndim, const int32_t* shape, DetDType dtype) {
    if (!ws || idx < 0 || idx >= 8) return NULL;

    size_t needed = det_dtype_size(dtype);
    for (int i = 0; i < ndim; i++) {
        needed *= shape[i];
    }

    /* Reallocate if needed */
    if (!ws->scratch[idx] || ws->scratch_sizes[idx] < needed) {
        det_tensor_release(ws->scratch[idx]);
        ws->scratch[idx] = det_tensor_create(ndim, shape, dtype);
        ws->scratch_sizes[idx] = needed;
    } else {
        /* Reshape existing buffer */
        ws->scratch[idx]->ndim = ndim;
        memcpy(ws->scratch[idx]->shape, shape, ndim * sizeof(int32_t));
        ws->scratch[idx]->dtype = dtype;
        compute_strides(ws->scratch[idx]);
    }

    return ws->scratch[idx];
}

/* ==========================================================================
 * QUANTIZATION
 * ========================================================================== */

/* Q8_0 block structure: 1 float scale + 32 int8 values */
#define Q8_0_BLOCK_SIZE 32

/* Q4_0 block structure: 1 float16 scale + 16 bytes (32 4-bit values) */
#define Q4_0_BLOCK_SIZE 32

int det_dequantize(DetTensor* dst, const DetTensor* src) {
    if (!dst || !src) return DET_TENSOR_ERR_INVALID;
    if (dst->dtype != DET_DTYPE_F32) return DET_TENSOR_ERR_DTYPE;

    size_t n = det_tensor_numel(src);
    float* out = (float*)dst->data;

    switch (src->dtype) {
        case DET_DTYPE_F32:
            /* Already float, just copy */
            memcpy(out, src->data, n * sizeof(float));
            break;

        case DET_DTYPE_F16: {
            /* Convert half to float */
            const uint16_t* in = (const uint16_t*)src->data;
            for (size_t i = 0; i < n; i++) {
                /* IEEE 754 half to float conversion */
                uint32_t h = in[i];
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;

                if (exp == 0) {
                    /* Denormal or zero */
                    if (mant == 0) {
                        out[i] = 0.0f;
                    } else {
                        /* Denormal */
                        float f = (float)mant / 1024.0f;
                        f *= powf(2.0f, -14.0f);
                        out[i] = sign ? -f : f;
                    }
                } else if (exp == 31) {
                    /* Inf or NaN */
                    uint32_t f = sign | 0x7F800000 | (mant << 13);
                    memcpy(&out[i], &f, 4);
                } else {
                    /* Normal */
                    uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
                    memcpy(&out[i], &f, 4);
                }
            }
            break;
        }

        case DET_DTYPE_BF16: {
            /* bfloat16: just shift left by 16 to get float32 */
            const uint16_t* in = (const uint16_t*)src->data;
            for (size_t i = 0; i < n; i++) {
                uint32_t f = (uint32_t)in[i] << 16;
                memcpy(&out[i], &f, 4);
            }
            break;
        }

        case DET_DTYPE_Q8_0: {
            /* Q8_0: blocks of 32 int8 values with fp16 scale
             * Block layout: 2 bytes fp16 scale + 32 bytes int8 data = 34 bytes per block */
            const uint8_t* in = (const uint8_t*)src->data;
            size_t num_blocks = (n + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
            const size_t block_size_bytes = 2 + Q8_0_BLOCK_SIZE;  /* 2 byte scale + 32 int8 */

            for (size_t b = 0; b < num_blocks; b++) {
                /* Read scale (2 bytes as float16) */
                uint16_t scale_h;
                memcpy(&scale_h, in + b * block_size_bytes, 2);

                /* Convert half to float */
                float scale;
                {
                    uint32_t h = scale_h;
                    uint32_t sign = (h & 0x8000) << 16;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;

                    if (exp == 0) {
                        /* Subnormal or zero */
                        uint32_t f = sign;
                        memcpy(&scale, &f, 4);
                    } else if (exp == 31) {
                        /* Inf or NaN */
                        uint32_t f = sign | 0x7F800000 | (mant << 13);
                        memcpy(&scale, &f, 4);
                    } else {
                        /* Normal */
                        uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
                        memcpy(&scale, &f, 4);
                    }
                }

                /* Dequantize block */
                const int8_t* q = (const int8_t*)(in + b * block_size_bytes + 2);
                size_t base = b * Q8_0_BLOCK_SIZE;
                size_t count = (base + Q8_0_BLOCK_SIZE > n) ? (n - base) : Q8_0_BLOCK_SIZE;

                for (size_t i = 0; i < count; i++) {
                    out[base + i] = q[i] * scale;
                }
            }
            break;
        }

        case DET_DTYPE_Q4_0:
        case DET_DTYPE_Q4_K_M: {
            /* Q4_0: blocks of 32 4-bit values with float16 scale */
            const uint8_t* in = (const uint8_t*)src->data;
            size_t num_blocks = (n + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;

            for (size_t b = 0; b < num_blocks; b++) {
                /* Read scale (2 bytes as float16) */
                uint16_t scale_h;
                memcpy(&scale_h, in + b * (2 + Q4_0_BLOCK_SIZE / 2), 2);

                /* Convert half to float */
                float scale;
                {
                    uint32_t h = scale_h;
                    uint32_t sign = (h & 0x8000) << 16;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    if (exp == 0) {
                        scale = 0.0f;
                    } else {
                        uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
                        memcpy(&scale, &f, 4);
                    }
                }

                /* Dequantize block (2 values per byte) */
                const uint8_t* q = in + b * (2 + Q4_0_BLOCK_SIZE / 2) + 2;
                size_t base = b * Q4_0_BLOCK_SIZE;
                size_t count = (base + Q4_0_BLOCK_SIZE > n) ? (n - base) : Q4_0_BLOCK_SIZE;

                for (size_t i = 0; i < count; i += 2) {
                    uint8_t byte = q[i / 2];
                    /* Low nibble, then high nibble */
                    int8_t v0 = (byte & 0x0F) - 8;
                    int8_t v1 = ((byte >> 4) & 0x0F) - 8;

                    if (i < count) out[base + i] = v0 * scale;
                    if (i + 1 < count) out[base + i + 1] = v1 * scale;
                }
            }
            break;
        }

        default:
            return DET_TENSOR_ERR_DTYPE;
    }

    return DET_TENSOR_OK;
}

int det_quantize(DetTensor* dst, const DetTensor* src, DetDType target_dtype) {
    if (!dst || !src) return DET_TENSOR_ERR_INVALID;
    if (src->dtype != DET_DTYPE_F32) return DET_TENSOR_ERR_DTYPE;

    /* For now, only support F32 -> Q8_0 quantization */
    if (target_dtype != DET_DTYPE_Q8_0) {
        return DET_TENSOR_ERR_DTYPE;
    }

    size_t n = det_tensor_numel(src);
    const float* in = (const float*)src->data;
    uint8_t* out = (uint8_t*)dst->data;

    size_t num_blocks = (n + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;

    for (size_t b = 0; b < num_blocks; b++) {
        size_t base = b * Q8_0_BLOCK_SIZE;
        size_t count = (base + Q8_0_BLOCK_SIZE > n) ? (n - base) : Q8_0_BLOCK_SIZE;

        /* Find max absolute value in block */
        float amax = 0.0f;
        for (size_t i = 0; i < count; i++) {
            float abs_val = fabsf(in[base + i]);
            if (abs_val > amax) amax = abs_val;
        }

        /* Compute scale */
        float scale = amax / 127.0f;
        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        /* Write scale */
        memcpy(out + b * (4 + Q8_0_BLOCK_SIZE), &scale, 4);

        /* Quantize block */
        int8_t* q = (int8_t*)(out + b * (4 + Q8_0_BLOCK_SIZE) + 4);
        for (size_t i = 0; i < count; i++) {
            float v = in[base + i] * inv_scale;
            int32_t vi = (int32_t)roundf(v);
            if (vi > 127) vi = 127;
            if (vi < -128) vi = -128;
            q[i] = (int8_t)vi;
        }

        /* Zero-fill remainder of block */
        for (size_t i = count; i < Q8_0_BLOCK_SIZE; i++) {
            q[i] = 0;
        }
    }

    return DET_TENSOR_OK;
}

/* ==========================================================================
 * GLOBAL CONFIGURATION
 * ========================================================================== */

static bool g_use_gpu = true;

void det_tensor_set_use_gpu(bool use_gpu) {
    g_use_gpu = use_gpu;
}

bool det_tensor_gpu_available(void) {
#ifdef __APPLE__
    /* Check if Metal is available via dlopen */
    void* metal = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    if (metal) {
        dlclose(metal);
        return true;
    }
#endif
    return false;
}

const char* det_tensor_gpu_name(void) {
    /* TODO: Query actual GPU name via Metal */
    return "Apple Silicon GPU";
}
