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

/* ==========================================================================
 * SSM (MAMBA) KERNELS
 * ========================================================================== */

/**
 * SSM Selective Scan - Core Mamba recurrence
 *
 * Implements: h_new = A_bar * h_old + B_bar * x
 *             y = C * h + D * x
 *
 * Where A_bar = exp(delta * A), B_bar = delta * B
 *
 * This kernel processes one (i, j) state element per thread.
 * For batch/sequence processing, launch seq_len times or use parallel scan.
 *
 * x: [d_inner] input at current timestep
 * delta: [d_inner] discretization timestep
 * A: [d_inner, d_state] log-space A matrix (negative values)
 * B: [d_state] input projection at current timestep
 * C: [d_state] output projection at current timestep
 * D: [d_inner] skip connection
 * h: [d_inner, d_state] hidden state (updated in-place)
 * y: [d_inner] output (accumulated)
 */
kernel void ssm_scan_step_f32(
    device const float* x [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device const float* A [[buffer(2)]],
    device const float* B [[buffer(3)]],
    device const float* C [[buffer(4)]],
    device const float* D [[buffer(5)]],
    device float* h [[buffer(6)]],
    device float* y [[buffer(7)]],
    constant uint& d_inner [[buffer(8)]],
    constant uint& d_state [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])  // (i, j) = (inner_dim, state_dim)
{
    uint i = gid.x;  // Inner dimension index
    uint j = gid.y;  // State dimension index

    if (i >= d_inner || j >= d_state) return;

    // Read values
    float x_i = x[i];
    float dt = delta[i];
    float A_ij = -exp(A[i * d_state + j]);  // A stored as log(-A)
    float B_j = B[j];
    float C_j = C[j];

    // Discretization
    float A_bar = exp(dt * A_ij);
    float B_bar = dt * B_j;

    // State update: h_new = A_bar * h_old + B_bar * x
    uint h_idx = i * d_state + j;
    float h_old = h[h_idx];
    float h_new = A_bar * h_old + B_bar * x_i;
    h[h_idx] = h_new;

    // Output contribution (atomically add to y[i])
    // Note: For production, use threadgroup reduction instead of atomic
    float y_contrib = C_j * h_new;

    // Atomic add for thread safety (Metal 2.0+)
    // For better performance, use parallel reduction
    device atomic_float* y_atomic = (device atomic_float*)&y[i];
    atomic_fetch_add_explicit(y_atomic, y_contrib, memory_order_relaxed);
}

/**
 * SSM output with skip connection: y += D * x
 */
kernel void ssm_skip_add_f32(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* D [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    y[gid] += D[gid] * x[gid];
}

/**
 * Causal 1D convolution for SSM preprocessing
 *
 * out[t, c] = sum_{k=0}^{d_conv-1} w[c, k] * x[t-k, c]
 *
 * Handles boundary conditions and maintains state for autoregressive.
 *
 * x: [seq_len, d_inner] input
 * w: [d_inner, d_conv] weights (per-channel)
 * bias: [d_inner] bias (optional, can be nullptr check before launch)
 * conv_state: [d_inner, d_conv-1] state from previous tokens
 * out: [seq_len, d_inner] output
 */
kernel void conv1d_causal_f32(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* conv_state [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& d_inner [[buffer(6)]],
    constant uint& d_conv [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])  // (t, c) = (time, channel)
{
    uint t = gid.x;  // Time index
    uint c = gid.y;  // Channel index

    if (t >= seq_len || c >= d_inner) return;

    float sum = 0.0f;

    // Convolution
    for (uint k = 0; k < d_conv; k++) {
        int src_t = int(t) - int(k);
        float x_val;

        if (src_t >= 0) {
            x_val = x[src_t * d_inner + c];
        } else if (conv_state != nullptr) {
            // From state: index into [c, d_conv-1] buffer
            int state_idx = src_t + int(d_conv - 1);
            if (state_idx >= 0) {
                x_val = conv_state[c * (d_conv - 1) + state_idx];
            } else {
                x_val = 0.0f;
            }
        } else {
            x_val = 0.0f;
        }

        sum += w[c * d_conv + k] * x_val;
    }

    // Add bias
    if (bias != nullptr) {
        sum += bias[c];
    }

    out[t * d_inner + c] = sum;
}

/**
 * Update convolution state for next sequence
 *
 * Shifts the last (d_conv-1) values from x into conv_state
 */
kernel void conv1d_update_state_f32(
    device const float* x [[buffer(0)]],
    device float* conv_state [[buffer(1)]],
    constant uint& seq_len [[buffer(2)]],
    constant uint& d_inner [[buffer(3)]],
    constant uint& d_conv [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])  // (k, c) = (state_pos, channel)
{
    uint k = gid.x;  // Position in state buffer
    uint c = gid.y;  // Channel

    if (k >= d_conv - 1 || c >= d_inner) return;

    int src_t = int(seq_len) - int(d_conv - 1) + int(k);

    if (src_t >= 0) {
        conv_state[c * (d_conv - 1) + k] = x[src_t * d_inner + c];
    }
    // Otherwise keep existing state (shifted during next call)
}

/**
 * Fused SiLU + Conv1d output for SSM preprocessing
 *
 * out = SiLU(conv1d_output)
 */
kernel void conv1d_silu_f32(
    device float* out [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    float x = out[gid];
    out[gid] = x / (1.0f + exp(-x));
}

/**
 * SSM gated output: y_out = y_ssm * SiLU(z)
 *
 * y_ssm: [seq_len, d_inner] SSM output
 * z: [seq_len, d_inner] gate values
 * y_out: [seq_len, d_inner] gated output
 */
kernel void ssm_gate_output_f32(
    device const float* y_ssm [[buffer(0)]],
    device const float* z [[buffer(1)]],
    device float* y_out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float y = y_ssm[gid];
    float g = z[gid];
    float gate = g / (1.0f + exp(-g));  // SiLU
    y_out[gid] = y * gate;
}

/**
 * Parallel prefix scan for SSM (Blelloch scan)
 *
 * Used for batch processing multiple timesteps in parallel.
 * Computes: y[t] = a[t] * y[t-1] + b[t] for all t in parallel
 *
 * This is the up-sweep phase.
 */
kernel void ssm_parallel_scan_upsweep_f32(
    device float* a [[buffer(0)]],  // Multiplicative coefficients (A_bar)
    device float* b [[buffer(1)]],  // Additive coefficients (B_bar * x)
    constant uint& n [[buffer(2)]],  // Number of elements
    constant uint& stride [[buffer(3)]],  // Current stride (2^d)
    uint gid [[thread_position_in_grid]])
{
    uint idx = (gid + 1) * stride * 2 - 1;
    if (idx >= n) return;

    uint left_idx = idx - stride;

    // Combine: (a_right, b_right) * (a_left, b_left) = (a_right * a_left, a_right * b_left + b_right)
    float a_left = a[left_idx];
    float b_left = b[left_idx];
    float a_right = a[idx];
    float b_right = b[idx];

    a[idx] = a_right * a_left;
    b[idx] = a_right * b_left + b_right;
}

/**
 * Parallel prefix scan down-sweep phase
 */
kernel void ssm_parallel_scan_downsweep_f32(
    device float* a [[buffer(0)]],
    device float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint idx = (gid + 1) * stride * 2 - 1;
    if (idx >= n) return;

    uint left_idx = idx - stride;

    float a_left = a[left_idx];
    float b_left = b[left_idx];
    float a_right = a[idx];

    // Propagate: b_left = a_right * b_left + b_left_old (handled by storing temp)
    b[left_idx] = a_right * b_left + b[idx];
}

/* ==========================================================================
 * FUSED KERNELS (Phase 26.18)
 * ========================================================================== */

/**
 * Fused gate and up projection for SwiGLU FFN
 *
 * Computes both projections in a single kernel launch:
 *   gate[i] = x @ W_gate[i]^T  (for row i of output)
 *   up[i] = x @ W_up[i]^T      (for row i of output)
 *
 * x: [M, K] input
 * W_gate: [N, K] gate projection weights (transposed)
 * W_up: [N, K] up projection weights (transposed)
 * gate: [M, N] gate output
 * up: [M, N] up output
 *
 * Each thread computes one element of both gate and up outputs.
 */
kernel void fused_gate_up_proj_f32(
    device const float* x [[buffer(0)]],
    device const float* W_gate [[buffer(1)]],
    device const float* W_up [[buffer(2)]],
    device float* gate [[buffer(3)]],
    device float* up [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N dimension

    if (row >= M || col >= N) return;

    // Compute dot product for both gate and up projections
    float sum_gate = 0.0f;
    float sum_up = 0.0f;

    device const float* x_row = x + row * K;
    device const float* w_gate_row = W_gate + col * K;
    device const float* w_up_row = W_up + col * K;

    // Vectorized loop for better memory throughput
    uint k = 0;
    for (; k + 4 <= K; k += 4) {
        float4 xv = float4(x_row[k], x_row[k+1], x_row[k+2], x_row[k+3]);
        float4 wg = float4(w_gate_row[k], w_gate_row[k+1], w_gate_row[k+2], w_gate_row[k+3]);
        float4 wu = float4(w_up_row[k], w_up_row[k+1], w_up_row[k+2], w_up_row[k+3]);
        sum_gate += dot(xv, wg);
        sum_up += dot(xv, wu);
    }
    // Handle remainder
    for (; k < K; k++) {
        float xv = x_row[k];
        sum_gate += xv * w_gate_row[k];
        sum_up += xv * w_up_row[k];
    }

    gate[row * N + col] = sum_gate;
    up[row * N + col] = sum_up;
}

/**
 * Fused RMSNorm + matmul projection
 *
 * Combines normalization with projection:
 *   normed = x / sqrt(mean(x^2) + eps) * norm_weight
 *   out[row, col] = normed @ W[col]^T
 *
 * x: [M, K] input (will be normalized)
 * norm_weight: [K] normalization weights
 * W: [N, K] projection weights (transposed)
 * out: [M, N] output
 */
kernel void fused_rmsnorm_matmul_f32(
    device const float* x [[buffer(0)]],
    device const float* norm_weight [[buffer(1)]],
    device const float* W [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N dimension

    if (row >= M || col >= N) return;

    device const float* x_row = x + row * K;

    // Compute RMS for this row (each thread recomputes - could optimize with threadgroup)
    float sq_sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        float v = x_row[k];
        sq_sum += v * v;
    }
    float rms = sqrt(sq_sum / float(K) + eps);
    float inv_rms = 1.0f / rms;

    // Compute normalized dot product with weight column
    device const float* w_row = W + col * K;
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        float normed = x_row[k] * inv_rms * norm_weight[k];
        sum += normed * w_row[k];
    }

    out[row * N + col] = sum;
}

/**
 * Fused SwiGLU: out = SiLU(gate) * up followed by down projection
 *
 * This combines three operations:
 *   1. silu_gate = gate / (1 + exp(-gate))
 *   2. mul = silu_gate * up
 *   3. out[row, col] = mul @ W_down[col]^T
 *
 * gate: [M, N_ff] gate projection output
 * up: [M, N_ff] up projection output
 * W_down: [N_out, N_ff] down projection weights (transposed)
 * out: [M, N_out] final output
 */
kernel void fused_swiglu_down_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* W_down [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N_ff [[buffer(5)]],
    constant uint& N_out [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N_out dimension

    if (row >= M || col >= N_out) return;

    device const float* gate_row = gate + row * N_ff;
    device const float* up_row = up + row * N_ff;
    device const float* w_row = W_down + col * N_ff;

    float sum = 0.0f;
    for (uint k = 0; k < N_ff; k++) {
        float g = gate_row[k];
        float silu_g = g / (1.0f + exp(-g));  // SiLU activation
        float mul = silu_g * up_row[k];
        sum += mul * w_row[k];
    }

    out[row * N_out + col] = sum;
}

/**
 * Optimized matvec for single-token inference: y = x @ W^T
 *
 * When M=1, this is essentially N independent dot products.
 * Each thread computes one element of the output.
 *
 * x: [K] input vector
 * W: [N, K] weight matrix (transposed)
 * y: [N] output vector
 */
kernel void matvec_transposed_f32(
    device const float* x [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= N) return;

    device const float* w_row = W + gid * K;
    float sum = 0.0f;

    // Vectorized loop
    uint k = 0;
    for (; k + 4 <= K; k += 4) {
        float4 xv = float4(x[k], x[k+1], x[k+2], x[k+3]);
        float4 wv = float4(w_row[k], w_row[k+1], w_row[k+2], w_row[k+3]);
        sum += dot(xv, wv);
    }
    for (; k < K; k++) {
        sum += x[k] * w_row[k];
    }

    y[gid] = sum;
}

/* ==========================================================================
 * GPU-NATIVE ATTENTION KERNELS (Phase: GPU-Native Forward Pass)
 *
 * These kernels operate on GPU buffers directly, eliminating CPU-GPU transfers
 * during the attention computation. All intermediate data stays on GPU.
 * ========================================================================== */

/**
 * Fused attention scores with causal mask: scores = Q @ K^T / sqrt(d_k) with causal masking
 *
 * Computes scaled dot-product attention scores with causal mask applied inline.
 * For incremental generation, pos_offset gives the position of the first query token
 * so masking is applied correctly: scores[i,j] = -inf for j > pos_offset + i
 *
 * Q: [seq_q, d_k] query vectors (GPU buffer)
 * K: [seq_k, d_k] key vectors (GPU buffer)
 * scores: [seq_q, seq_k] output attention scores (GPU buffer)
 *
 * Each thread computes one element of the output scores matrix.
 */
kernel void attention_scores_causal_gpu(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& seq_q [[buffer(3)]],
    constant uint& seq_k [[buffer(4)]],
    constant uint& d_k [[buffer(5)]],
    constant uint& pos_offset [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;  // Query position (0 to seq_q-1)
    uint j = gid.x;  // Key position (0 to seq_k-1)

    if (i >= seq_q || j >= seq_k) return;

    // Causal masking: query at position (pos_offset + i) can only attend to keys at positions <= (pos_offset + i)
    // j is the absolute key position in the KV cache
    uint query_abs_pos = pos_offset + i;
    if (j > query_abs_pos) {
        scores[i * seq_k + j] = -1e9f;  // Masked out
        return;
    }

    // Compute Q[i] dot K[j]
    device const float* q = Q + i * d_k;
    device const float* k = K + j * d_k;

    float dot = 0.0f;
    for (uint d = 0; d < d_k; d++) {
        dot += q[d] * k[d];
    }

    // Scale by 1/sqrt(d_k)
    float scale = 1.0f / sqrt(float(d_k));
    scores[i * seq_k + j] = dot * scale;
}

/**
 * Row-wise softmax in-place on GPU buffer
 *
 * Applies softmax independently to each row of the input matrix.
 * Softmax is computed in a numerically stable way:
 *   softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * x: [rows, dim] input/output (modified in-place)
 *
 * Each thread processes one row.
 */
kernel void softmax_rows_gpu(
    device float* x [[buffer(0)]],
    constant uint& rows [[buffer(1)]],
    constant uint& dim [[buffer(2)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= rows) return;

    device float* row_ptr = x + row * dim;

    // Find max for numerical stability
    float max_val = row_ptr[0];
    for (uint j = 1; j < dim; j++) {
        max_val = max(max_val, row_ptr[j]);
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (uint j = 0; j < dim; j++) {
        float val = exp(row_ptr[j] - max_val);
        row_ptr[j] = val;
        sum += val;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint j = 0; j < dim; j++) {
        row_ptr[j] *= inv_sum;
    }
}

/**
 * Attention weighted sum: out = scores @ V
 *
 * Computes the final attention output by multiplying attention weights with values.
 *
 * scores: [seq_q, seq_k] attention weights (after softmax)
 * V: [seq_k, d_v] value vectors
 * out: [seq_q, d_v] output
 *
 * Each thread computes one element of the output.
 */
kernel void attention_weighted_sum_gpu(
    device const float* scores [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& seq_q [[buffer(3)]],
    constant uint& seq_k [[buffer(4)]],
    constant uint& d_v [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint i = gid.y;  // Query/output position
    uint d = gid.x;  // Value dimension

    if (i >= seq_q || d >= d_v) return;

    // out[i,d] = sum_j scores[i,j] * V[j,d]
    device const float* score_row = scores + i * seq_k;
    float sum = 0.0f;
    for (uint j = 0; j < seq_k; j++) {
        sum += score_row[j] * V[j * d_v + d];
    }
    out[i * d_v + d] = sum;
}

/**
 * RoPE (Rotary Position Embedding) in-place on GPU buffer
 *
 * Applies rotary position embedding to Q or K vectors.
 * Uses split-half pairing: element[i] pairs with element[i + head_dim/2]
 *
 * x: [seq, heads, head_dim] input/output (modified in-place)
 * pos_offset: starting position for the sequence
 * theta: RoPE base frequency (e.g., 10000.0)
 *
 * Grid: (seq, heads, 1) - each thread handles one head at one position
 */
kernel void rope_gpu(
    device float* x [[buffer(0)]],
    constant uint& seq [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& pos_offset [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint s = gid.y;  // Sequence position
    uint h = gid.x;  // Head index

    if (s >= seq || h >= heads) return;

    device float* head_ptr = x + s * heads * head_dim + h * head_dim;
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

/**
 * Copy K/V to cache at specified layer and position
 *
 * Copies new K or V vectors into the KV cache at the appropriate location.
 * cache: [n_layer, n_ctx, kv_dim] full cache buffer
 * src: [num_tokens, kv_dim] new K or V vectors to store
 *
 * layer_offset: starting offset in cache for this layer (layer * n_ctx * kv_dim)
 * pos: starting position in sequence
 * num_tokens: number of tokens to store
 */
kernel void kv_cache_store_gpu(
    device const float* src [[buffer(0)]],
    device float* cache [[buffer(1)]],
    constant uint& layer_offset [[buffer(2)]],
    constant uint& pos [[buffer(3)]],
    constant uint& num_tokens [[buffer(4)]],
    constant uint& kv_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint t = gid.y;  // Token index
    uint d = gid.x;  // Dimension

    if (t >= num_tokens || d >= kv_dim) return;

    uint src_idx = t * kv_dim + d;
    uint dst_idx = layer_offset + (pos + t) * kv_dim + d;
    cache[dst_idx] = src[src_idx];
}

/**
 * Add two GPU buffers element-wise (in-place variant): a = a + b
 *
 * Used for residual connections when data stays on GPU.
 */
kernel void add_inplace_gpu(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    a[gid] += b[gid];
}

/**
 * Scale GPU buffer in-place: x = x * scale
 *
 * Used for scaling attention outputs (e.g., differential attention lambda scaling).
 */
kernel void scale_inplace_gpu(
    device float* x [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    x[gid] *= scale;
}

// =============================================================================
// MULTI-HEAD ATTENTION WITH GQA SUPPORT
// These kernels handle all heads in parallel with proper GQA mapping
// =============================================================================

/**
 * Multi-head attention scores with causal masking and GQA
 *
 * Computes scores[t, h, s] = Q[t, h, :] dot K[s, kv_head(h), :] * scale
 * with causal masking where s > pos + t is masked to -inf
 *
 * Q: [num_tokens, n_head, head_dim] query vectors
 * K_cache: [seq_len, n_head_kv, head_dim] key cache for this layer
 * scores: [num_tokens, n_head, seq_len] output attention scores
 *
 * gqa_ratio: n_head / n_head_kv (how many Q heads share a KV head)
 *
 * Grid: (seq_len, n_head, num_tokens)
 */
kernel void attention_multihead_scores_gpu(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& num_tokens [[buffer(3)]],
    constant uint& n_head [[buffer(4)]],
    constant uint& n_head_kv [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    constant uint& pos_offset [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint s = gid.x;  // Key position in cache (0 to seq_len-1)
    uint h = gid.y;  // Query head (0 to n_head-1)
    uint t = gid.z;  // Query token (0 to num_tokens-1)

    if (s >= seq_len || h >= n_head || t >= num_tokens) return;

    // GQA: map query head to KV head
    uint gqa_ratio = n_head / n_head_kv;
    uint kv_head = h / gqa_ratio;

    // Causal mask: query at position (pos_offset + t) can only attend to keys at positions <= (pos_offset + t)
    uint query_abs_pos = pos_offset + t;
    uint score_idx = t * n_head * seq_len + h * seq_len + s;

    if (s > query_abs_pos) {
        scores[score_idx] = -1e9f;
        return;
    }

    // Compute Q[t,h,:] dot K[s,kv_head,:]
    device const float* q_ptr = Q + t * n_head * head_dim + h * head_dim;
    device const float* k_ptr = K_cache + s * n_head_kv * head_dim + kv_head * head_dim;

    float dot = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        dot += q_ptr[d] * k_ptr[d];
    }

    // Scale by 1/sqrt(head_dim)
    float scale = 1.0f / sqrt(float(head_dim));
    scores[score_idx] = dot * scale;
}

/**
 * Multi-head softmax with numerical stability
 *
 * Applies softmax to scores[t, h, :] for each token and head independently.
 * scores: [num_tokens, n_head, seq_len] (modified in-place)
 *
 * Grid: (n_head, num_tokens, 1) - one thread per (token, head) pair
 */
kernel void attention_multihead_softmax_gpu(
    device float* scores [[buffer(0)]],
    constant uint& num_tokens [[buffer(1)]],
    constant uint& n_head [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint h = gid.x;  // Head
    uint t = gid.y;  // Token

    if (h >= n_head || t >= num_tokens) return;

    device float* row = scores + t * n_head * seq_len + h * seq_len;

    // Find max for numerical stability
    float max_val = row[0];
    for (uint s = 1; s < seq_len; s++) {
        max_val = max(max_val, row[s]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint s = 0; s < seq_len; s++) {
        row[s] = exp(row[s] - max_val);
        sum += row[s];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint s = 0; s < seq_len; s++) {
        row[s] *= inv_sum;
    }
}

/**
 * Multi-head attention weighted sum with GQA
 *
 * Computes out[t, h, d] = sum_s scores[t, h, s] * V[s, kv_head(h), d]
 *
 * scores: [num_tokens, n_head, seq_len] attention weights
 * V_cache: [seq_len, n_head_kv, head_dim] value cache for this layer
 * out: [num_tokens, n_head, head_dim] attention output
 *
 * Grid: (head_dim, n_head, num_tokens)
 */
kernel void attention_multihead_weighted_sum_gpu(
    device const float* scores [[buffer(0)]],
    device const float* V_cache [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& num_tokens [[buffer(3)]],
    constant uint& n_head [[buffer(4)]],
    constant uint& n_head_kv [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d = gid.x;  // Output dimension within head
    uint h = gid.y;  // Head
    uint t = gid.z;  // Token

    if (d >= head_dim || h >= n_head || t >= num_tokens) return;

    // GQA: map query head to KV head
    uint gqa_ratio = n_head / n_head_kv;
    uint kv_head = h / gqa_ratio;

    device const float* score_row = scores + t * n_head * seq_len + h * seq_len;

    float sum = 0.0f;
    for (uint s = 0; s < seq_len; s++) {
        // V[s, kv_head, d]
        sum += score_row[s] * V_cache[s * n_head_kv * head_dim + kv_head * head_dim + d];
    }

    // out[t, h, d]
    out[t * n_head * head_dim + h * head_dim + d] = sum;
}

// =============================================================================
// DIFFERENTIAL ATTENTION (phi4flash)
// =============================================================================

/**
 * Differential attention scores computation
 *
 * Computes attention scores for both even (q1@k1) and odd (q2@k2) head groups
 * with causal masking and optional sliding window.
 *
 * Q: [num_tokens, n_head, head_dim] - query vectors
 * K_cache: [seq_len, n_head_kv, head_dim] - key cache
 * scores1: [num_tokens, half_heads, seq_len] - scores for even heads (q1@k1)
 * scores2: [num_tokens, half_heads, seq_len] - scores for odd heads (q2@k2)
 *
 * Grid: (seq_len, half_heads, num_tokens)
 */
kernel void diff_attn_scores_gpu(
    device const float* Q [[buffer(0)]],
    device const float* K_cache [[buffer(1)]],
    device float* scores1 [[buffer(2)]],
    device float* scores2 [[buffer(3)]],
    constant uint& num_tokens [[buffer(4)]],
    constant uint& n_head [[buffer(5)]],
    constant uint& n_head_kv [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& seq_len [[buffer(8)]],
    constant uint& pos_offset [[buffer(9)]],
    constant int& sliding_window [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint s = gid.x;   // Key position
    uint hh = gid.y;  // Half-head index (0 to half_heads-1)
    uint t = gid.z;   // Token index

    uint half_heads = n_head / 2;
    uint half_kv_heads = n_head_kv / 2;

    if (s >= seq_len || hh >= half_heads || t >= num_tokens) return;

    // Map half-head to actual head indices
    uint h1 = hh * 2;      // Even head (0, 2, 4, ...)
    uint h2 = hh * 2 + 1;  // Odd head (1, 3, 5, ...)

    // GQA mapping within each split group
    uint half_gqa = half_heads / half_kv_heads;
    uint kv_h1 = (hh / half_gqa) * 2;      // Even KV head
    uint kv_h2 = (hh / half_gqa) * 2 + 1;  // Odd KV head

    // Causal masking
    uint query_abs_pos = pos_offset + t;
    bool masked = (s > query_abs_pos);

    // Sliding window masking
    if (sliding_window > 0) {
        int window_start = (int)query_abs_pos - sliding_window + 1;
        if (window_start > 0 && s < (uint)window_start) {
            masked = true;
        }
    }

    uint scores1_idx = t * half_heads * seq_len + hh * seq_len + s;
    uint scores2_idx = t * half_heads * seq_len + hh * seq_len + s;

    if (masked) {
        scores1[scores1_idx] = -1e9f;
        scores2[scores2_idx] = -1e9f;
        return;
    }

    // Compute q1 @ k1 (even heads)
    device const float* q1 = Q + t * n_head * head_dim + h1 * head_dim;
    device const float* k1 = K_cache + s * n_head_kv * head_dim + kv_h1 * head_dim;
    float dot1 = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        dot1 += q1[d] * k1[d];
    }

    // Compute q2 @ k2 (odd heads)
    device const float* q2 = Q + t * n_head * head_dim + h2 * head_dim;
    device const float* k2 = K_cache + s * n_head_kv * head_dim + kv_h2 * head_dim;
    float dot2 = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        dot2 += q2[d] * k2[d];
    }

    float scale = 1.0f / sqrt(float(head_dim));
    scores1[scores1_idx] = dot1 * scale;
    scores2[scores2_idx] = dot2 * scale;
}

/**
 * Differential attention softmax for both score sets
 *
 * Grid: (half_heads, num_tokens, 1)
 */
kernel void diff_attn_softmax_gpu(
    device float* scores1 [[buffer(0)]],
    device float* scores2 [[buffer(1)]],
    constant uint& num_tokens [[buffer(2)]],
    constant uint& half_heads [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint hh = gid.x;
    uint t = gid.y;

    if (hh >= half_heads || t >= num_tokens) return;

    // Softmax for scores1
    device float* row1 = scores1 + t * half_heads * seq_len + hh * seq_len;
    float max1 = row1[0];
    for (uint s = 1; s < seq_len; s++) max1 = max(max1, row1[s]);
    float sum1 = 0.0f;
    for (uint s = 0; s < seq_len; s++) {
        row1[s] = exp(row1[s] - max1);
        sum1 += row1[s];
    }
    float inv1 = 1.0f / sum1;
    for (uint s = 0; s < seq_len; s++) row1[s] *= inv1;

    // Softmax for scores2
    device float* row2 = scores2 + t * half_heads * seq_len + hh * seq_len;
    float max2 = row2[0];
    for (uint s = 1; s < seq_len; s++) max2 = max(max2, row2[s]);
    float sum2 = 0.0f;
    for (uint s = 0; s < seq_len; s++) {
        row2[s] = exp(row2[s] - max2);
        sum2 += row2[s];
    }
    float inv2 = 1.0f / sum2;
    for (uint s = 0; s < seq_len; s++) row2[s] *= inv2;
}

/**
 * Differential attention weighted sum with differential formula
 *
 * Computes:
 *   attn1 = [scores1 @ v1, scores1 @ v2]  (2 * head_dim)
 *   attn2 = [scores2 @ v1, scores2 @ v2]  (2 * head_dim)
 *   diff = attn1 - lambda * attn2
 *   out = SubLN(diff) * output_scale
 *
 * Then interleaves back to original head order.
 *
 * Grid: (head_dim * 2, half_heads, num_tokens)
 */
kernel void diff_attn_weighted_sum_gpu(
    device const float* scores1 [[buffer(0)]],
    device const float* scores2 [[buffer(1)]],
    device const float* V_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const float* subln_weight [[buffer(4)]],
    constant uint& num_tokens [[buffer(5)]],
    constant uint& n_head [[buffer(6)]],
    constant uint& n_head_kv [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& seq_len [[buffer(9)]],
    constant float& lambda [[buffer(10)]],
    constant float& output_scale [[buffer(11)]],
    constant float& norm_eps [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d2 = gid.x;  // Dimension (0 to 2*head_dim-1)
    uint hh = gid.y;  // Half-head index
    uint t = gid.z;   // Token index

    uint half_heads = n_head / 2;
    uint half_kv_heads = n_head_kv / 2;
    uint double_head_dim = head_dim * 2;

    if (d2 >= double_head_dim || hh >= half_heads || t >= num_tokens) return;

    // Which V to use: first half uses v1 (even KV), second half uses v2 (odd KV)
    bool use_v2 = (d2 >= head_dim);
    uint d = use_v2 ? (d2 - head_dim) : d2;

    // GQA mapping
    uint half_gqa = half_heads / half_kv_heads;
    uint kv_h1 = (hh / half_gqa) * 2;      // Even KV head for v1
    uint kv_h2 = (hh / half_gqa) * 2 + 1;  // Odd KV head for v2
    uint kv_h = use_v2 ? kv_h2 : kv_h1;

    // Compute attn1 = scores1 @ V
    device const float* score_row1 = scores1 + t * half_heads * seq_len + hh * seq_len;
    float attn1 = 0.0f;
    for (uint s = 0; s < seq_len; s++) {
        attn1 += score_row1[s] * V_cache[s * n_head_kv * head_dim + kv_h * head_dim + d];
    }

    // Compute attn2 = scores2 @ V
    device const float* score_row2 = scores2 + t * half_heads * seq_len + hh * seq_len;
    float attn2 = 0.0f;
    for (uint s = 0; s < seq_len; s++) {
        attn2 += score_row2[s] * V_cache[s * n_head_kv * head_dim + kv_h * head_dim + d];
    }

    // Differential: diff = attn1 - lambda * attn2
    float diff = attn1 - lambda * attn2;

    // Apply SubLN weight (simplified - full SubLN requires computing RMS across 2*head_dim)
    // For now, just apply the weight
    if (subln_weight) {
        diff *= subln_weight[d2];
    }

    // Scale by output_scale = (1 - lambda_init)
    diff *= output_scale;

    // Map half-head back to original head order
    // hh corresponds to pair (h1=2*hh, h2=2*hh+1)
    // Output for d2 < head_dim goes to h1, d2 >= head_dim goes to h2
    uint h_out = use_v2 ? (hh * 2 + 1) : (hh * 2);

    // out[t, h_out, d]
    out[t * n_head * head_dim + h_out * head_dim + d] = diff;
}
