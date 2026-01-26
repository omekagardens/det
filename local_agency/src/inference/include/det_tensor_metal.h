/**
 * DET Tensor Metal Backend - Phase 26.1
 * ======================================
 *
 * GPU-accelerated tensor operations via Metal.
 */

#ifndef DET_TENSOR_METAL_H
#define DET_TENSOR_METAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize Metal backend
 *
 * Returns 0 on success, -1 on failure.
 */
int tensor_metal_init(void);

/**
 * Check if Metal GPU is available
 */
int tensor_metal_available(void);

/**
 * Get Metal device name
 */
const char* tensor_metal_device_name(void);

/**
 * GPU-accelerated matrix multiplication: C = A @ B
 *
 * A: [M, K]
 * B: [K, N]
 * C: [M, N]
 */
int tensor_metal_matmul(const float *A, const float *B, float *C,
                        uint32_t M, uint32_t N, uint32_t K);

/**
 * GPU-accelerated matrix multiplication with transposed B: C = A @ B^T
 *
 * A: [M, K]
 * B: [N, K] (stored as [N, K], treated as B^T)
 * C: [M, N]
 *
 * This is the key operation for projection layers where weights are [out, in].
 */
int tensor_metal_matmul_transposed_b(const float *A, const float *B, float *C,
                                      uint32_t M, uint32_t N, uint32_t K);

/**
 * GPU-accelerated SiLU activation
 */
int tensor_metal_silu(const float *x, float *y, uint32_t n);

/**
 * GPU-accelerated fused SiLU-multiply for SwiGLU: out = SiLU(gate) * up
 *
 * gate: [n] - gate projection output
 * up:   [n] - up projection output
 * out:  [n] - result
 */
int tensor_metal_silu_mul(const float *gate, const float *up, float *out, uint32_t n);

/**
 * GPU-accelerated RMSNorm
 */
int tensor_metal_rmsnorm(const float *x, const float *weight, float *y,
                          uint32_t rows, uint32_t dim, float eps);

/**
 * GPU-accelerated softmax
 */
int tensor_metal_softmax(const float *x, float *y,
                          uint32_t rows, uint32_t dim, float temp);

/**
 * GPU-accelerated attention scores: scores = Q @ K^T / sqrt(d_k)
 *
 * Q: [seq_q, d_k]
 * K: [seq_k, d_k]
 * scores: [seq_q, seq_k]
 */
int tensor_metal_attention_scores(const float *Q, const float *K, float *scores,
                                   uint32_t seq_q, uint32_t seq_k, uint32_t d_k);

/**
 * GPU-accelerated causal mask application
 *
 * Sets scores[i, j] = -1e9 for j > i
 */
int tensor_metal_causal_mask(float *scores, uint32_t seq_q, uint32_t seq_k);

/**
 * GPU-accelerated RoPE (Rotary Position Embedding)
 *
 * x: [seq, heads, head_dim] - modified in place
 * Uses split-half pairing: element[i] pairs with element[i + head_dim/2]
 */
int tensor_metal_rope(float *x, uint32_t seq, uint32_t heads,
                       uint32_t head_dim, uint32_t pos_offset, float theta);

/**
 * GPU-accelerated Q8_0 dequantization
 *
 * src: Q8_0 quantized data (36 bytes per block: 4-byte scale + 32 int8 values)
 * dst: Output float buffer (32 floats per block)
 * num_blocks: Number of Q8_0 blocks to dequantize
 */
int tensor_metal_dequantize_q8_0(const uint8_t *src, float *dst, uint32_t num_blocks);

#ifdef __cplusplus
}
#endif

#endif /* DET_TENSOR_METAL_H */
