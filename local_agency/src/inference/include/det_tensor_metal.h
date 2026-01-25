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
 * GPU-accelerated SiLU activation
 */
int tensor_metal_silu(const float *x, float *y, uint32_t n);

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

#ifdef __cplusplus
}
#endif

#endif /* DET_TENSOR_METAL_H */
