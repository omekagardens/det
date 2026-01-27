/**
 * DET Inference - Metal Compute Shaders
 * ======================================
 *
 * GPU-accelerated tensor operations for Phase 26.1
 */

#include <metal_stdlib>
using namespace metal;

/* ==========================================================================
 * MATRIX MULTIPLICATION
 * ========================================================================== */

/**
 * General matrix multiplication: C = A @ B
 *
 * A: [M, K]
 * B: [K, N]
 * C: [M, N]
 *
 * Uses tiled approach with threadgroup memory for better cache utilization.
 */

#define TILE_SIZE 16

kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    // Shared memory tiles
    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    float sum = 0.0f;

    // Loop over tiles
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < numTiles; t++) {
        // Load tiles into shared memory
        uint aCol = t * TILE_SIZE + tid.x;
        uint bRow = t * TILE_SIZE + tid.y;

        if (row < M && aCol < K) {
            As[tid.y][tid.x] = A[row * K + aCol];
        } else {
            As[tid.y][tid.x] = 0.0f;
        }

        if (bRow < K && col < N) {
            Bs[tid.y][tid.x] = B[bRow * N + col];
        } else {
            Bs[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[tid.y][k] * Bs[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Matrix multiplication with transposed B: C = A @ B^T
 *
 * A: [M, K]
 * B: [N, K]  (stored as [N, K], we treat it as B^T[K, N])
 * C: [M, N]
 *
 * This is the key kernel for projection operations where weights are [out, in].
 * C[i,j] = sum_k(A[i,k] * B[j,k])
 *
 * Simple non-tiled version for correctness.
 */
kernel void matmul_transposed_b_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[col * K + k];
    }
    C[row * N + col] = sum;
}

/**
 * Matrix-vector multiplication: y = A @ x
 *
 * A: [M, N]
 * x: [N]
 * y: [M]
 */
kernel void matvec_f32(
    device const float* A [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= M) return;

    float sum = 0.0f;
    device const float* row = A + gid * N;

    for (uint j = 0; j < N; j++) {
        sum += row[j] * x[j];
    }

    y[gid] = sum;
}

/* ==========================================================================
 * ELEMENT-WISE OPERATIONS
 * ========================================================================== */

kernel void add_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    C[gid] = A[gid] + B[gid];
}

kernel void mul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    C[gid] = A[gid] * B[gid];
}

kernel void mul_scalar_f32(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    B[gid] = A[gid] * scalar;
}

kernel void div_scalar_f32(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    B[gid] = A[gid] / scalar;
}

/* ==========================================================================
 * ACTIVATION FUNCTIONS
 * ========================================================================== */

kernel void silu_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float val = x[gid];
    y[gid] = val / (1.0f + exp(-val));
}

/**
 * Fused SiLU-multiply for SwiGLU FFN: out = SiLU(gate) * up
 *
 * gate: [n] - gate projection output
 * up:   [n] - up projection output
 * out:  [n] - result (can be same as gate for in-place)
 */
kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float g = gate[gid];
    float silu_g = g / (1.0f + exp(-g));
    out[gid] = silu_g * up[gid];
}

kernel void gelu_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    float val = x[gid];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x3 = val * val * val;
    float inner = 0.7978845608f * (val + 0.044715f * x3);
    y[gid] = 0.5f * val * (1.0f + tanh(inner));
}

kernel void relu_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    y[gid] = max(0.0f, x[gid]);
}

/* ==========================================================================
 * NORMALIZATION
 * ========================================================================== */

/**
 * RMS Normalization: y = x / rms(x) * weight
 *
 * Computed in two passes:
 * 1. Reduce to compute sum of squares
 * 2. Normalize and apply weight
 */
kernel void rmsnorm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    device const float* x_row = x + row * dim;
    device float* y_row = y + row * dim;

    // Compute sum of squares
    float ss = 0.0f;
    for (uint i = 0; i < dim; i++) {
        ss += x_row[i] * x_row[i];
    }

    // Compute scale
    float scale = 1.0f / sqrt(ss / float(dim) + eps);

    // Apply normalization and weight
    for (uint i = 0; i < dim; i++) {
        y_row[i] = x_row[i] * scale * weight[i];
    }
}

/* ==========================================================================
 * SOFTMAX
 * ========================================================================== */

/**
 * Softmax with temperature: y = softmax(x / temperature)
 *
 * Computed in three passes:
 * 1. Find max
 * 2. Compute exp(x - max) and sum
 * 3. Normalize
 */
kernel void softmax_f32(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    constant float& temperature [[buffer(3)]],
    uint row [[thread_position_in_grid]])
{
    device const float* x_row = x + row * dim;
    device float* y_row = y + row * dim;

    // Find max
    float max_val = x_row[0] / temperature;
    for (uint i = 1; i < dim; i++) {
        max_val = max(max_val, x_row[i] / temperature);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float val = exp(x_row[i] / temperature - max_val);
        y_row[i] = val;
        sum += val;
    }

    // Normalize
    for (uint i = 0; i < dim; i++) {
        y_row[i] /= sum;
    }
}

/* ==========================================================================
 * ATTENTION
 * ========================================================================== */

/**
 * Scaled dot-product attention scores: scores = Q @ K^T / sqrt(d_k)
 *
 * Q: [batch, seq_q, d_k]
 * K: [batch, seq_k, d_k]
 * scores: [batch, seq_q, seq_k]
 */
kernel void attention_scores_f32(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& seq_q [[buffer(4)]],
    constant uint& seq_k [[buffer(5)]],
    constant uint& d_k [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])  // (batch, seq_q, seq_k)
{
    uint b = gid.x;
    uint i = gid.y;
    uint j = gid.z;

    if (b >= batch || i >= seq_q || j >= seq_k) return;

    device const float* q = Q + b * seq_q * d_k + i * d_k;
    device const float* k = K + b * seq_k * d_k + j * d_k;

    float dot = 0.0f;
    for (uint d = 0; d < d_k; d++) {
        dot += q[d] * k[d];
    }

    float scale = 1.0f / sqrt(float(d_k));
    scores[b * seq_q * seq_k + i * seq_k + j] = dot * scale;
}

/**
 * Apply causal mask: scores[i, j] = -inf for j > i
 */
kernel void causal_mask_f32(
    device float* scores [[buffer(0)]],
    constant uint& seq_q [[buffer(1)]],
    constant uint& seq_k [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])  // (row, col)
{
    uint i = gid.x;
    uint j = gid.y;

    if (i >= seq_q || j >= seq_k) return;

    if (j > i) {
        scores[i * seq_k + j] = -1e9f;
    }
}

/* ==========================================================================
 * ROPE (Rotary Position Embedding)
 * ========================================================================== */

/**
 * Apply RoPE to a tensor (split-half pairing convention)
 *
 * x: [batch, seq, heads, head_dim]
 * Applies rotation based on position
 *
 * Uses HuggingFace-style split-half pairing:
 *   element[i] pairs with element[i + head_dim/2]
 * NOT consecutive pairing (i, i+1)
 */
kernel void rope_f32(
    device float* x [[buffer(0)]],
    constant uint& batch [[buffer(1)]],
    constant uint& seq [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& pos_offset [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])  // (batch, seq, head)
{
    uint b = gid.x;
    uint s = gid.y;
    uint h = gid.z;

    if (b >= batch || s >= seq || h >= heads) return;

    device float* head_ptr = x + b * seq * heads * head_dim +
                                 s * heads * head_dim +
                                 h * head_dim;

    uint pos = pos_offset + s;
    uint half_dim = head_dim / 2;

    // Split-half pairing: pair element[i] with element[i + half_dim]
    for (uint i = 0; i < half_dim; i++) {
        float freq = 1.0f / pow(theta, float(2 * i) / float(head_dim));
        float angle = float(pos) * freq;
        float cos_val = cos(angle);
        float sin_val = sin(angle);

        float x0 = head_ptr[i];
        float x1 = head_ptr[i + half_dim];
        head_ptr[i]            = x0 * cos_val - x1 * sin_val;
        head_ptr[i + half_dim] = x1 * cos_val + x0 * sin_val;
    }
}

/* ==========================================================================
 * QUANTIZATION HELPERS
 * ========================================================================== */

/**
 * Convert F16 (half) to F32 (float) - GGUF uses F16 scale in Q8_0
 */
inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        // Zero or denormal
        if (mant == 0) return 0.0f;
        // Denormal - rare, just return 0
        return 0.0f;
    } else if (exp == 31) {
        // Inf or NaN
        uint32_t f = sign | 0x7F800000 | (mant << 13);
        return as_type<float>(f);
    } else {
        // Normal
        uint32_t f = sign | ((exp + 112) << 23) | (mant << 13);
        return as_type<float>(f);
    }
}

/**
 * Dequantize Q8_0 block to F32
 *
 * GGUF Q8_0 block format: 2-byte F16 scale + 32 int8 values = 34 bytes per block
 */
kernel void dequantize_q8_0(
    device const uint8_t* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& num_blocks [[buffer(2)]],
    uint gid [[thread_position_in_grid]])  // block index
{
    if (gid >= num_blocks) return;

    // Each Q8_0 block is 2 bytes F16 scale + 32 bytes data = 34 bytes
    device const uint8_t* block = src + gid * 34;

    // Read scale (first 2 bytes as F16, convert to F32)
    uint16_t scale_h = *((device const uint16_t*)block);
    float scale = half_to_float(scale_h);

    // Dequantize 32 values
    device const int8_t* q = (device const int8_t*)(block + 2);
    device float* out = dst + gid * 32;

    for (uint i = 0; i < 32; i++) {
        out[i] = float(q[i]) * scale;
    }
}

/**
 * Fused Q8_0 dequantize + matmul with transposed B: C = A @ B_q8^T
 *
 * A: [M, K] float32
 * B_q8: [N, K] Q8_0 quantized (stored as N*K/32 blocks of 34 bytes each)
 * C: [M, N] float32
 *
 * Each row of B has K/32 Q8_0 blocks, each block is 34 bytes.
 */
kernel void matmul_q8_0_transposed_f32(
    device const float* A [[buffer(0)]],
    device const uint8_t* B_q8 [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // Output row (0..M-1)
    uint col = gid.x;  // Output column (0..N-1)

    if (row >= M || col >= N) return;

    // Pointers
    device const float* a_row = A + row * K;

    // B_q8 row layout: each row has K/32 blocks, each block is 34 bytes
    // Block j of row n starts at: n * (K/32 * 34) + j * 34
    uint num_blocks_per_row = K / 32;
    device const uint8_t* b_row = B_q8 + col * num_blocks_per_row * 34;

    float sum = 0.0f;

    // Process 32 elements at a time (one Q8_0 block)
    for (uint blk = 0; blk < num_blocks_per_row; blk++) {
        device const uint8_t* block = b_row + blk * 34;

        // Read F16 scale
        uint16_t scale_h = *((device const uint16_t*)block);
        float scale = half_to_float(scale_h);

        // Dequantize and accumulate
        device const int8_t* q = (device const int8_t*)(block + 2);
        uint base_k = blk * 32;

        for (uint i = 0; i < 32; i++) {
            float b_val = float(q[i]) * scale;
            sum += a_row[base_k + i] * b_val;
        }
    }

    C[row * N + col] = sum;
}
