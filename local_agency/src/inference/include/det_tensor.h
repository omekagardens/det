/**
 * DET Tensor Primitives - Phase 26.1 Foundation
 * ==============================================
 *
 * Tensor operations for DET-native model inference.
 * Provides CPU (BLAS) and GPU (Metal) implementations.
 *
 * Design Principles:
 * - Keep buffers persistent (reuse across tokens)
 * - Avoid per-call allocations
 * - Support quantized formats (Q4_K_M, Q8_0)
 * - Metal acceleration where beneficial
 */

#ifndef DET_TENSOR_H
#define DET_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * TENSOR DATA TYPES
 * ========================================================================== */

/** Tensor data type */
typedef enum {
    DET_DTYPE_F32 = 0,      /* 32-bit float */
    DET_DTYPE_F16 = 1,      /* 16-bit float (half) */
    DET_DTYPE_BF16 = 2,     /* bfloat16 */
    DET_DTYPE_Q8_0 = 3,     /* 8-bit quantized (block size 32) */
    DET_DTYPE_Q4_K_M = 4,   /* 4-bit quantized (k-quants) */
    DET_DTYPE_Q4_0 = 5,     /* 4-bit quantized (simple) */
    DET_DTYPE_I32 = 6,      /* 32-bit integer */
    DET_DTYPE_I16 = 7,      /* 16-bit integer */
} DetDType;

/** Storage location */
typedef enum {
    DET_STORAGE_CPU = 0,    /* CPU memory */
    DET_STORAGE_GPU = 1,    /* GPU memory (Metal buffer) */
    DET_STORAGE_MMAP = 2,   /* Memory-mapped file */
} DetStorage;

/* ==========================================================================
 * TENSOR STRUCTURE
 * ========================================================================== */

#define DET_MAX_DIMS 4

/**
 * Tensor - Multi-dimensional array
 *
 * Layout: Row-major (C order)
 * Strides are in elements, not bytes
 */
typedef struct DetTensor {
    /* Data pointer */
    void* data;             /* Raw data pointer */
    size_t data_size;       /* Total size in bytes */

    /* Shape and strides */
    int32_t ndim;           /* Number of dimensions */
    int32_t shape[DET_MAX_DIMS];    /* Dimension sizes */
    int32_t stride[DET_MAX_DIMS];   /* Strides in elements */

    /* Type and storage */
    DetDType dtype;         /* Data type */
    DetStorage storage;     /* Storage location */

    /* Memory management */
    bool owns_data;         /* True if tensor owns its data */
    int32_t refcount;       /* Reference count */

    /* For quantized types */
    float scale;            /* Quantization scale */
    int32_t zero_point;     /* Quantization zero point */

    /* For mmap */
    int fd;                 /* File descriptor for mmap */
    size_t file_offset;     /* Offset in mapped file */
} DetTensor;

/* ==========================================================================
 * TENSOR LIFECYCLE
 * ========================================================================== */

/** Create a new tensor with given shape and type */
DetTensor* det_tensor_create(int32_t ndim, const int32_t* shape, DetDType dtype);

/** Create tensor on GPU */
DetTensor* det_tensor_create_gpu(int32_t ndim, const int32_t* shape, DetDType dtype);

/** Create tensor view (shares data, doesn't own it) */
DetTensor* det_tensor_view(DetTensor* src, int32_t offset, int32_t ndim, const int32_t* shape);

/** Create tensor from existing data (doesn't copy, doesn't own) */
DetTensor* det_tensor_from_ptr(void* data, int32_t ndim, const int32_t* shape, DetDType dtype);

/** Create tensor from memory-mapped file */
DetTensor* det_tensor_mmap(const char* path, size_t offset, int32_t ndim,
                            const int32_t* shape, DetDType dtype);

/** Clone tensor (deep copy) */
DetTensor* det_tensor_clone(const DetTensor* src);

/** Increment reference count */
void det_tensor_retain(DetTensor* t);

/** Decrement reference count, free if zero */
void det_tensor_release(DetTensor* t);

/* ==========================================================================
 * TENSOR PROPERTIES
 * ========================================================================== */

/** Get number of elements */
static inline size_t det_tensor_numel(const DetTensor* t) {
    size_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

/** Get element size in bytes */
size_t det_dtype_size(DetDType dtype);

/** Get dtype name */
const char* det_dtype_name(DetDType dtype);

/** Check if tensor is contiguous */
bool det_tensor_is_contiguous(const DetTensor* t);

/** Get flat index from multi-dimensional indices */
static inline size_t det_tensor_index(const DetTensor* t, const int32_t* indices) {
    size_t idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        idx += indices[i] * t->stride[i];
    }
    return idx;
}

/* ==========================================================================
 * MEMORY TRANSFER
 * ========================================================================== */

/** Copy tensor to GPU */
int det_tensor_to_gpu(DetTensor* t);

/** Copy tensor to CPU */
int det_tensor_to_cpu(DetTensor* t);

/** Copy data between tensors */
int det_tensor_copy(DetTensor* dst, const DetTensor* src);

/* ==========================================================================
 * BASIC OPERATIONS
 * ========================================================================== */

/**
 * Matrix multiplication: C = A @ B
 *
 * A: [M, K]
 * B: [K, N]
 * C: [M, N]
 *
 * Uses Accelerate BLAS on macOS, OpenBLAS fallback elsewhere.
 * Automatically dequantizes quantized inputs.
 */
int det_matmul(DetTensor* C, const DetTensor* A, const DetTensor* B);

/**
 * Batched matrix multiplication: C = A @ B
 *
 * A: [batch, M, K]
 * B: [batch, K, N] or [K, N] (broadcast)
 * C: [batch, M, N]
 */
int det_matmul_batched(DetTensor* C, const DetTensor* A, const DetTensor* B);

/**
 * Matrix-vector multiplication: y = A @ x
 *
 * A: [M, N]
 * x: [N]
 * y: [M]
 */
int det_matvec(DetTensor* y, const DetTensor* A, const DetTensor* x);

/* ==========================================================================
 * ELEMENT-WISE OPERATIONS
 * ========================================================================== */

/** Element-wise addition: C = A + B */
int det_add(DetTensor* C, const DetTensor* A, const DetTensor* B);

/** Element-wise multiplication: C = A * B */
int det_mul(DetTensor* C, const DetTensor* A, const DetTensor* B);

/** Scalar division: B = A / scalar */
int det_div_scalar(DetTensor* B, const DetTensor* A, float scalar);

/** Scalar multiplication: B = A * scalar */
int det_mul_scalar(DetTensor* B, const DetTensor* A, float scalar);

/** In-place scalar multiplication: A *= scalar */
int det_mul_scalar_inplace(DetTensor* A, float scalar);

/** Fused multiply-add: D = A * B + C */
int det_fma(DetTensor* D, const DetTensor* A, const DetTensor* B, const DetTensor* C);

/* ==========================================================================
 * ACTIVATION FUNCTIONS
 * ========================================================================== */

/** SiLU (Swish) activation: y = x * sigmoid(x) */
int det_silu(DetTensor* y, const DetTensor* x);

/** SiLU in-place */
int det_silu_inplace(DetTensor* x);

/** GELU activation */
int det_gelu(DetTensor* y, const DetTensor* x);

/** ReLU activation */
int det_relu(DetTensor* y, const DetTensor* x);

/* ==========================================================================
 * NORMALIZATION
 * ========================================================================== */

/**
 * RMSNorm: y = (x / rms(x)) * weight
 *
 * x: [N, D] or [D]
 * weight: [D]
 * y: [N, D] or [D]
 * eps: small constant for numerical stability
 */
int det_rmsnorm(DetTensor* y, const DetTensor* x, const DetTensor* weight, float eps);

/**
 * LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
 */
int det_layernorm(DetTensor* y, const DetTensor* x,
                   const DetTensor* weight, const DetTensor* bias, float eps);

/* ==========================================================================
 * SOFTMAX AND SAMPLING
 * ========================================================================== */

/**
 * Softmax: y = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * x: [N] or [batch, N]
 * y: [N] or [batch, N]
 * temperature: divide logits by temperature before softmax (1.0 = no change)
 */
int det_softmax(DetTensor* y, const DetTensor* x, float temperature);

/**
 * Softmax in-place with temperature
 */
int det_softmax_inplace(DetTensor* x, float temperature);

/**
 * Top-p (nucleus) sampling
 *
 * probs: [vocab_size] probability distribution
 * p: cumulative probability threshold (0.0-1.0)
 * seed: random seed for reproducibility
 *
 * Returns: sampled token index
 */
int32_t det_sample_top_p(const DetTensor* probs, float p, uint64_t seed);

/**
 * Top-k sampling
 */
int32_t det_sample_top_k(const DetTensor* probs, int32_t k, uint64_t seed);

/**
 * Greedy sampling (argmax)
 */
int32_t det_sample_greedy(const DetTensor* probs);

/**
 * Apply repetition penalty to logits
 *
 * logits: [vocab_size] unnormalized log-probabilities
 * token_ids: [context_len] recent token IDs
 * penalty: repetition penalty factor (1.0 = no penalty)
 */
int det_repetition_penalty(DetTensor* logits, const int32_t* token_ids,
                            int32_t context_len, float penalty);

/* ==========================================================================
 * ATTENTION OPERATIONS
 * ========================================================================== */

/**
 * Scaled dot-product attention scores: scores = Q @ K^T / sqrt(d_k)
 *
 * Q: [batch, seq_q, d_k]
 * K: [batch, seq_k, d_k]
 * scores: [batch, seq_q, seq_k]
 */
int det_attention_scores(DetTensor* scores, const DetTensor* Q, const DetTensor* K);

/**
 * Apply causal mask to attention scores
 *
 * scores: [batch, seq_q, seq_k]
 * Fills upper triangle with -inf
 */
int det_causal_mask(DetTensor* scores);

/**
 * Full attention: output = softmax(Q @ K^T / sqrt(d_k)) @ V
 *
 * Q: [batch, seq_q, d_k]
 * K: [batch, seq_k, d_k]
 * V: [batch, seq_k, d_v]
 * output: [batch, seq_q, d_v]
 * is_causal: apply causal mask if true
 */
int det_attention(DetTensor* output, const DetTensor* Q, const DetTensor* K,
                   const DetTensor* V, bool is_causal);

/* ==========================================================================
 * ROPE (Rotary Position Embedding)
 * ========================================================================== */

/**
 * Apply rotary position embedding
 *
 * x: [batch, seq, n_heads, d_head]
 * pos: starting position
 * theta: base frequency (typically 10000.0)
 */
int det_rope(DetTensor* x, int32_t pos, float theta);

/**
 * Precompute RoPE sin/cos cache
 *
 * sin_cache: [max_seq, d_head/2]
 * cos_cache: [max_seq, d_head/2]
 */
int det_rope_cache_init(DetTensor* sin_cache, DetTensor* cos_cache,
                         int32_t max_seq, int32_t d_head, float theta);

/* ==========================================================================
 * QUANTIZATION
 * ========================================================================== */

/** Dequantize tensor to float32 */
int det_dequantize(DetTensor* dst, const DetTensor* src);

/** Quantize float32 tensor */
int det_quantize(DetTensor* dst, const DetTensor* src, DetDType target_dtype);

/* ==========================================================================
 * QUANTIZATION-AWARE MATMUL (QAM)
 * ========================================================================== */

/**
 * Q8_0 matmul with transposed weights: Y[M,N] = X[M,K] @ W_q8[N,K]^T
 *
 * Performs on-the-fly dequantization during matmul.
 * K must be a multiple of 32 (Q8_0 block size).
 *
 * @param Y Output matrix [M, N] float32
 * @param X Input matrix [M, K] float32
 * @param W_q8 Q8_0 quantized weight matrix [N, K] (raw bytes, 34 bytes per 32-element block)
 * @param M Number of rows in X and Y
 * @param N Number of columns in Y (and rows in W)
 * @param K Number of columns in X (and W), must be multiple of 32
 */
int det_matmul_q8_0_transposed(float* Y, const float* X, const uint8_t* W_q8,
                                int M, int N, int K);

/**
 * Batched Q8_0 matmul - more efficient for large N
 *
 * Dequantizes W in batches and uses BLAS sgemm for better performance.
 */
int det_matmul_q8_0_transposed_batched(float* Y, const float* X, const uint8_t* W_q8,
                                        int M, int N, int K);

/* ==========================================================================
 * WORKSPACE MANAGEMENT
 * ========================================================================== */

/**
 * Tensor workspace - Pre-allocated scratch buffers
 *
 * Avoids per-operation allocations for better performance.
 */
typedef struct DetTensorWorkspace {
    DetTensor* scratch[8];      /* Scratch tensors */
    size_t scratch_sizes[8];    /* Allocated sizes */
    int num_scratch;            /* Number of scratch buffers */

    /* GPU-specific */
    void* metal_queue;          /* MTLCommandQueue for GPU ops */
    void* metal_device;         /* MTLDevice */
} DetTensorWorkspace;

/** Create workspace with given scratch buffer sizes */
DetTensorWorkspace* det_workspace_create(const size_t* sizes, int num_buffers);

/** Destroy workspace */
void det_workspace_destroy(DetTensorWorkspace* ws);

/** Get scratch tensor (auto-resize if needed) */
DetTensor* det_workspace_get_scratch(DetTensorWorkspace* ws, int idx,
                                      int32_t ndim, const int32_t* shape, DetDType dtype);

/* ==========================================================================
 * ERROR HANDLING
 * ========================================================================== */

#define DET_TENSOR_OK           0
#define DET_TENSOR_ERR_ALLOC   -1
#define DET_TENSOR_ERR_SHAPE   -2
#define DET_TENSOR_ERR_DTYPE   -3
#define DET_TENSOR_ERR_IO      -4
#define DET_TENSOR_ERR_GPU     -5
#define DET_TENSOR_ERR_INVALID -6

/** Get error message for error code */
const char* det_tensor_strerror(int err);

/* ==========================================================================
 * GLOBAL CONFIGURATION
 * ========================================================================== */

/** Enable/disable GPU acceleration */
void det_tensor_set_use_gpu(bool use_gpu);

/** Check if GPU is available */
bool det_tensor_gpu_available(void);

/** Get GPU device name */
const char* det_tensor_gpu_name(void);

#ifdef __cplusplus
}
#endif

#endif /* DET_TENSOR_H */
