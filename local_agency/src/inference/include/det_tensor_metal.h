/**
 * DET Tensor Metal Backend - Phase 26.1
 * ======================================
 *
 * GPU-accelerated tensor operations via Metal.
 */

#ifndef DET_TENSOR_METAL_H
#define DET_TENSOR_METAL_H

#include <stdint.h>
#include <stddef.h>

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
 * GGUF Q8_0 format: 34 bytes per block (2-byte F16 scale + 32 int8 values)
 * src: Q8_0 quantized data
 * dst: Output float buffer (32 floats per block)
 * num_blocks: Number of Q8_0 blocks to dequantize
 */
int tensor_metal_dequantize_q8_0(const uint8_t *src, float *dst, uint32_t num_blocks);

/**
 * GPU-accelerated Q8_0 matmul with transposed B: C = A @ B_q8^T
 *
 * Fused dequantization + matmul for memory efficiency.
 *
 * A: [M, K] float32
 * B_q8: [N, K] Q8_0 quantized (K must be divisible by 32)
 * C: [M, N] float32
 */
int tensor_metal_matmul_q8_0_transposed(const float *A, const uint8_t *B_q8, float *C,
                                         uint32_t M, uint32_t N, uint32_t K);

/* ==========================================================================
 * PERSISTENT GPU BUFFERS (Phase 26.15)
 * ========================================================================== */

/**
 * Create a persistent GPU buffer from CPU data.
 *
 * The returned handle can be stored in DetTensor.metal_buffer and reused
 * for multiple operations without re-copying data.
 *
 * @param data Source data in CPU memory
 * @param size Size in bytes
 * @return Opaque buffer handle (MTLBuffer*), or NULL on failure
 */
void* tensor_metal_buffer_create(const void *data, size_t size);

/**
 * Create a persistent GPU buffer without initialization.
 *
 * Use for output buffers that will be written by GPU.
 *
 * @param size Size in bytes
 * @return Opaque buffer handle, or NULL on failure
 */
void* tensor_metal_buffer_create_empty(size_t size);

/**
 * Free a persistent GPU buffer.
 *
 * @param buffer Handle from tensor_metal_buffer_create
 */
void tensor_metal_buffer_free(void *buffer);

/**
 * Copy data from persistent GPU buffer back to CPU.
 *
 * @param buffer GPU buffer handle
 * @param dst Destination CPU memory
 * @param size Size in bytes to copy
 * @return 0 on success, -1 on failure
 */
int tensor_metal_buffer_read(void *buffer, void *dst, size_t size);

/**
 * Update data in persistent GPU buffer from CPU.
 *
 * @param buffer GPU buffer handle
 * @param src Source CPU memory
 * @param size Size in bytes to copy
 * @return 0 on success, -1 on failure
 */
int tensor_metal_buffer_write(void *buffer, const void *src, size_t size);

/**
 * Matrix multiply using persistent GPU buffers: C = A @ B^T
 *
 * All buffers must be persistent GPU buffers (from tensor_metal_buffer_create).
 * This avoids per-call buffer allocation and data copying.
 *
 * @param A_buf Persistent buffer for A [M, K]
 * @param B_buf Persistent buffer for B [N, K] (transposed)
 * @param C_buf Persistent buffer for output C [M, N]
 * @param M Rows of A and C
 * @param N Rows of B (cols of C)
 * @param K Cols of A and B
 * @return 0 on success, -1 on failure
 */
int tensor_metal_matmul_persistent(void *A_buf, void *B_buf, void *C_buf,
                                    uint32_t M, uint32_t N, uint32_t K);

/**
 * Q8_0 matmul using persistent GPU buffer for weights: C = A @ B_q8^T
 *
 * @param A Input activation (copied to GPU each call)
 * @param B_buf Persistent buffer for Q8_0 weights [N, K/32 blocks]
 * @param C Output (copied back from GPU)
 * @param M Rows of A and C
 * @param N Rows of B (output dimension)
 * @param K Cols of A, rows of B (K must be divisible by 32)
 * @return 0 on success, -1 on failure
 */
int tensor_metal_matmul_q8_0_persistent(const float *A, void *B_buf, float *C,
                                         uint32_t M, uint32_t N, uint32_t K);

/**
 * F32 matmul using persistent GPU buffer for weights: C = A @ B^T
 *
 * @param A Input activation (copied to GPU each call)
 * @param B_buf Persistent buffer for F32 weights [N, K]
 * @param C Output (copied back from GPU)
 * @param M Rows of A and C
 * @param N Rows of B (output dimension)
 * @param K Cols of A and B
 * @return 0 on success, -1 on failure
 */
int tensor_metal_matmul_f32_persistent(const float *A, void *B_buf, float *C,
                                        uint32_t M, uint32_t N, uint32_t K);

/**
 * Initialize scratch buffers for efficient inference.
 *
 * Pre-allocates GPU buffers for input activations and output results
 * to avoid per-call buffer allocation overhead.
 *
 * @param max_input_size Maximum input activation size in floats
 * @param max_output_size Maximum output size in floats
 * @return 0 on success, -1 on failure
 */
int tensor_metal_init_scratch(uint32_t max_input_size, uint32_t max_output_size);

/**
 * Free scratch buffers.
 */
void tensor_metal_free_scratch(void);

/* ==========================================================================
 * BATCH MODE - Accumulate operations in single command buffer (Phase 26.18)
 * ========================================================================== */

/**
 * Initialize persistent GPU buffers for forward pass.
 *
 * Pre-allocates GPU memory for all intermediate tensors to avoid
 * CPU-GPU transfers during the forward pass.
 *
 * @param max_seq Maximum sequence length
 * @param d_model Model dimension
 * @param max_intermediate Maximum intermediate dimension (e.g., FFN hidden)
 * @param vocab_size Vocabulary size for logits
 * @return 0 on success, -1 on failure
 */
int tensor_metal_init_forward_buffers(uint32_t max_seq, uint32_t d_model,
                                       uint32_t max_intermediate, uint32_t vocab_size);

/**
 * Free forward pass buffers.
 */
void tensor_metal_free_forward_buffers(void);

/**
 * Begin batch mode - start accumulating GPU operations.
 *
 * After calling this, GPU operations will be accumulated in a single
 * command buffer instead of being executed immediately. Call
 * tensor_metal_end_batch() to execute all accumulated operations.
 *
 * @return 0 on success, -1 on failure
 */
int tensor_metal_begin_batch(void);

/**
 * End batch mode - execute all accumulated operations.
 *
 * Commits the batch command buffer and waits for completion.
 *
 * @return Number of operations executed, or -1 on failure
 */
int tensor_metal_end_batch(void);

/**
 * Check if batch mode is currently active.
 *
 * @return 1 if active, 0 otherwise
 */
int tensor_metal_batch_active(void);

/**
 * Matmul with all data on GPU buffers: C = A @ B^T
 *
 * All buffers must be persistent GPU buffers. No CPU-GPU copies.
 * Compatible with batch mode.
 *
 * @param A_buf GPU buffer for A [M, K]
 * @param B_buf GPU buffer for B [N, K] (transposed)
 * @param C_buf GPU buffer for C [M, N]
 */
int tensor_metal_matmul_gpu_buffers(void *A_buf, void *B_buf, void *C_buf,
                                     uint32_t M, uint32_t N, uint32_t K);

/**
 * RMSNorm on GPU buffers.
 */
int tensor_metal_rmsnorm_gpu_buffers(void *x_buf, void *weight_buf, void *y_buf,
                                      uint32_t rows, uint32_t dim, float eps);

/**
 * SiLU multiply on GPU buffers: out = SiLU(gate) * up
 */
int tensor_metal_silu_mul_gpu_buffers(void *gate_buf, void *up_buf, void *out_buf, uint32_t n);

/**
 * Add on GPU buffers: out = a + b
 */
int tensor_metal_add_gpu_buffers(void *a_buf, void *b_buf, void *out_buf, uint32_t n);

/**
 * Copy CPU data to GPU buffer.
 */
int tensor_metal_upload_to_buffer(void *gpu_buf, const float *cpu_data, size_t size_bytes);

/**
 * Copy GPU buffer to CPU.
 */
int tensor_metal_download_from_buffer(void *gpu_buf, float *cpu_data, size_t size_bytes);

/**
 * Get forward pass GPU buffers.
 *
 * @param index 0 or 1 for ping-pong buffers
 */
void* tensor_metal_get_hidden_buffer(int index);
void* tensor_metal_get_intermediate_buffer(int index);
void* tensor_metal_get_logits_buffer(void);

/* ==========================================================================
 * FUSED KERNELS (Phase 26.18)
 * ========================================================================== */

/**
 * Fused gate + up projection for SwiGLU FFN.
 *
 * Computes both projections in single kernel launch:
 *   gate = x @ W_gate^T
 *   up = x @ W_up^T
 *
 * @param x Input [M, K]
 * @param W_gate_buf Persistent GPU buffer for gate weights [N, K]
 * @param W_up_buf Persistent GPU buffer for up weights [N, K]
 * @param gate Output gate [M, N]
 * @param up Output up [M, N]
 */
int tensor_metal_fused_gate_up_proj(const float *x, void *W_gate_buf, void *W_up_buf,
                                     float *gate, float *up,
                                     uint32_t M, uint32_t N, uint32_t K);

/**
 * Fused SwiGLU + down projection.
 *
 * Computes: out = (SiLU(gate) * up) @ W_down^T
 *
 * @param gate Gate projection output [M, N_ff]
 * @param up Up projection output [M, N_ff]
 * @param W_down_buf Persistent GPU buffer for down weights [N_out, N_ff]
 * @param out Output [M, N_out]
 */
int tensor_metal_fused_swiglu_down(const float *gate, const float *up, void *W_down_buf,
                                    float *out,
                                    uint32_t M, uint32_t N_ff, uint32_t N_out);

/**
 * Optimized matvec for single-token inference: y = x @ W^T
 *
 * When M=1, this is more efficient than general matmul.
 *
 * @param x Input vector [K]
 * @param W_buf Persistent GPU buffer for weights [N, K]
 * @param y Output vector [N]
 */
int tensor_metal_matvec_f32(const float *x, void *W_buf, float *y,
                             uint32_t N, uint32_t K);

/**
 * Complete GPU-native SwiGLU FFN - no intermediate CPU copies
 *
 * Computes: hidden_out += SiLU(hidden @ W1^T) * (hidden @ W3^T) @ W2^T
 * All operations execute in a single command buffer with data staying on GPU.
 *
 * @param hidden_in Input hidden state [M, n_embd]
 * @param hidden_out Output hidden state [M, n_embd] (residual added in-place)
 * @param W1_buf Persistent GPU buffer for gate weights [n_ff, n_embd]
 * @param W3_buf Persistent GPU buffer for up weights [n_ff, n_embd]
 * @param W2_buf Persistent GPU buffer for down weights [n_embd, n_ff]
 * @param M Number of tokens
 * @param n_ff FFN intermediate dimension
 * @param n_embd Model embedding dimension
 */
int tensor_metal_ffn_swiglu_complete(const float *hidden_in, float *hidden_out,
                                      void *W1_buf, void *W3_buf, void *W2_buf,
                                      uint32_t M, uint32_t n_ff, uint32_t n_embd);

/* ==========================================================================
 * SSM (MAMBA) GPU OPERATIONS
 * ========================================================================== */

/**
 * GPU-accelerated SSM selective scan step
 *
 * Computes one step of the SSM recurrence:
 *   h_new = exp(delta * A) * h_old + delta * B * x
 *   y = C * h_new + D * x
 *
 * x: [d_inner] input
 * delta: [d_inner] time step
 * A: [d_inner, d_state] log-space A matrix
 * B: [d_state] input projection
 * C: [d_state] output projection
 * D: [d_inner] skip connection
 * h: [d_inner, d_state] hidden state (updated in-place)
 * y: [d_inner] output
 */
int tensor_metal_ssm_scan_step(const float *x, const float *delta,
                                const float *A, const float *B,
                                const float *C, const float *D,
                                float *h, float *y,
                                uint32_t d_inner, uint32_t d_state);

/**
 * GPU-accelerated causal 1D convolution
 *
 * x: [seq_len, d_inner] input
 * w: [d_inner, d_conv] weights
 * bias: [d_inner] bias (can be NULL)
 * conv_state: [d_inner, d_conv-1] state (updated in-place)
 * out: [seq_len, d_inner] output
 */
int tensor_metal_conv1d_causal(const float *x, const float *w, const float *bias,
                                float *conv_state, float *out,
                                uint32_t seq_len, uint32_t d_inner, uint32_t d_conv);

/**
 * GPU-accelerated SSM gated output: y_out = y_ssm * SiLU(z)
 *
 * y_ssm: [n] SSM output
 * z: [n] gate values
 * y_out: [n] gated output
 */
int tensor_metal_ssm_gate(const float *y_ssm, const float *z, float *y_out, uint32_t n);

/**
 * GPU-accelerated parallel prefix scan for batch SSM
 *
 * Processes multiple timesteps in parallel using work-efficient scan.
 *
 * a: [seq_len, d_inner, d_state] A_bar coefficients
 * b: [seq_len, d_inner, d_state] B_bar * x coefficients
 * y: [seq_len, d_inner] output
 * h_init: [d_inner, d_state] initial hidden state
 * h_final: [d_inner, d_state] final hidden state (output)
 */
int tensor_metal_ssm_parallel_scan(const float *a, const float *b,
                                    const float *C, const float *D,
                                    const float *x, float *y,
                                    float *h_init, float *h_final,
                                    uint32_t seq_len, uint32_t d_inner, uint32_t d_state);

/* ==========================================================================
 * GPU-NATIVE ATTENTION OPERATIONS (GPU-Native Forward Pass)
 *
 * These functions operate on GPU buffers directly, keeping all intermediate
 * data on GPU to eliminate CPU-GPU transfer overhead during attention.
 * ========================================================================== */

/**
 * Fused attention scores with causal mask on GPU buffers.
 *
 * Computes: scores[i,j] = Q[i] @ K[j]^T / sqrt(d_k), masked for j > pos_offset+i
 *
 * @param Q_buf GPU buffer for Q [seq_q, d_k]
 * @param K_buf GPU buffer for K [seq_k, d_k]
 * @param scores_buf GPU buffer for scores [seq_q, seq_k]
 * @param seq_q Number of query positions
 * @param seq_k Number of key positions (KV cache length)
 * @param d_k Head dimension
 * @param pos_offset Absolute position of first query token (for causal masking)
 * @return 0 on success, -1 on failure
 */
int tensor_metal_attention_scores_causal_gpu(
    void* Q_buf, void* K_buf, void* scores_buf,
    uint32_t seq_q, uint32_t seq_k, uint32_t d_k, uint32_t pos_offset);

/**
 * Row-wise softmax in-place on GPU buffer.
 *
 * @param x_buf GPU buffer [rows, dim] - modified in-place
 * @param rows Number of rows
 * @param dim Dimension of each row
 * @return 0 on success, -1 on failure
 */
int tensor_metal_softmax_rows_gpu(void* x_buf, uint32_t rows, uint32_t dim);

/**
 * Attention weighted sum on GPU buffers: out = scores @ V
 *
 * @param scores_buf GPU buffer for attention weights [seq_q, seq_k]
 * @param V_buf GPU buffer for values [seq_k, d_v]
 * @param out_buf GPU buffer for output [seq_q, d_v]
 * @param seq_q Number of query positions
 * @param seq_k Number of key/value positions
 * @param d_v Value dimension (usually same as head dimension)
 * @return 0 on success, -1 on failure
 */
int tensor_metal_attention_weighted_sum_gpu(
    void* scores_buf, void* V_buf, void* out_buf,
    uint32_t seq_q, uint32_t seq_k, uint32_t d_v);

/**
 * RoPE (Rotary Position Embedding) in-place on GPU buffer.
 *
 * Applies rotary position embedding using split-half pairing convention.
 *
 * @param x_buf GPU buffer [seq, heads, head_dim] - modified in-place
 * @param seq Sequence length
 * @param heads Number of attention heads
 * @param head_dim Dimension per head
 * @param pos_offset Starting absolute position
 * @param theta RoPE frequency base (e.g., 10000.0)
 * @return 0 on success, -1 on failure
 */
int tensor_metal_rope_gpu(void* x_buf, uint32_t seq, uint32_t heads,
                          uint32_t head_dim, uint32_t pos_offset, float theta);

/**
 * Add two GPU buffers in-place: a = a + b
 *
 * Used for residual connections when data stays on GPU.
 *
 * @param a_buf GPU buffer (input and output)
 * @param b_buf GPU buffer (added to a)
 * @param n Number of elements
 * @return 0 on success, -1 on failure
 */
int tensor_metal_add_inplace_gpu(void* a_buf, void* b_buf, uint32_t n);

/**
 * Scale GPU buffer in-place: x = x * scale
 *
 * @param x_buf GPU buffer (modified in-place)
 * @param scale Scale factor
 * @param n Number of elements
 * @return 0 on success, -1 on failure
 */
int tensor_metal_scale_inplace_gpu(void* x_buf, float scale, uint32_t n);

/* ==========================================================================
 * MULTI-HEAD ATTENTION WITH GQA (GPU-Native)
 *
 * Complete multi-head attention computation on GPU buffers.
 * Handles Grouped Query Attention (GQA) where multiple Q heads share KV heads.
 * ========================================================================== */

/**
 * Multi-head attention computation on GPU.
 *
 * Computes complete attention for all heads in parallel:
 * 1. scores[t,h,s] = Q[t,h,:] dot K_cache[s,kv_h,:] / sqrt(head_dim) with causal mask
 * 2. scores = softmax(scores, dim=-1)
 * 3. out[t,h,:] = sum_s scores[t,h,s] * V_cache[s,kv_h,:]
 *
 * Handles GQA: kv_h = h / (n_head / n_head_kv)
 *
 * @param q_cpu CPU buffer with Q vectors [num_tokens, n_head * head_dim]
 * @param k_cache_cpu CPU buffer with K cache [seq_len, n_head_kv * head_dim]
 * @param v_cache_cpu CPU buffer with V cache [seq_len, n_head_kv * head_dim]
 * @param out_cpu CPU buffer for output [num_tokens, n_head * head_dim]
 * @param num_tokens Number of query tokens (typically 1 for generation)
 * @param seq_len Total sequence length (pos + num_tokens)
 * @param n_head Number of query heads
 * @param n_head_kv Number of KV heads (n_head_kv <= n_head)
 * @param head_dim Dimension per head
 * @param pos_offset Starting position for causal mask
 * @return 0 on success, -1 on failure
 */
int tensor_metal_attention_multihead(
    const float* q_cpu, const float* k_cache_cpu, const float* v_cache_cpu,
    float* out_cpu,
    uint32_t num_tokens, uint32_t seq_len,
    uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
    uint32_t pos_offset);

/**
 * Multi-head attention with all data already on GPU buffers.
 *
 * Same as tensor_metal_attention_multihead but uses GPU buffers directly.
 * More efficient when data is already on GPU (avoids upload/download).
 *
 * @param q_buf GPU buffer with Q vectors [num_tokens, n_head * head_dim]
 * @param k_cache_buf GPU buffer with K cache [seq_len, n_head_kv * head_dim]
 * @param v_cache_buf GPU buffer with V cache [seq_len, n_head_kv * head_dim]
 * @param out_buf GPU buffer for output [num_tokens, n_head * head_dim]
 * @param scores_buf GPU scratch buffer [num_tokens, n_head, seq_len]
 * @param num_tokens Number of query tokens
 * @param seq_len Total sequence length
 * @param n_head Number of query heads
 * @param n_head_kv Number of KV heads
 * @param head_dim Dimension per head
 * @param pos_offset Starting position for causal mask
 * @return 0 on success, -1 on failure
 */
int tensor_metal_attention_multihead_gpu(
    void* q_buf, void* k_cache_buf, void* v_cache_buf,
    void* out_buf, void* scores_buf,
    uint32_t num_tokens, uint32_t seq_len,
    uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
    uint32_t pos_offset);

/* ==========================================================================
 * DIFFERENTIAL ATTENTION (phi4flash GPU)
 *
 * GPU-accelerated differential attention with 4-way attention computation.
 * ========================================================================== */

/**
 * GPU differential attention computation.
 *
 * Computes phi4flash differential attention:
 * 1. Split heads into even (q1,k1,v1) and odd (q2,k2,v2) groups
 * 2. Compute 4 attention results: q1@k1@v1, q1@k1@v2, q2@k2@v1, q2@k2@v2
 * 3. Apply lambda scaling: diff = attn1 - lambda * attn2
 * 4. Apply SubLN and output scaling
 * 5. Interleave back to original head order
 *
 * @param q_cpu CPU buffer with Q vectors [num_tokens, n_head * head_dim]
 * @param k_cache_cpu CPU buffer with K cache [seq_len, n_head_kv * head_dim]
 * @param v_cache_cpu CPU buffer with V cache [seq_len, n_head_kv * head_dim]
 * @param out_cpu CPU buffer for output [num_tokens, n_head * head_dim]
 * @param subln_weight SubLN weight [2 * head_dim] or NULL
 * @param num_tokens Number of query tokens
 * @param seq_len Total sequence length
 * @param n_head Number of query heads
 * @param n_head_kv Number of KV heads
 * @param head_dim Dimension per head
 * @param pos_offset Starting position for causal mask
 * @param sliding_window Sliding window size (0 for no window)
 * @param lambda Lambda value for differential scaling
 * @param output_scale Output scale (1 - lambda_init)
 * @param norm_eps Normalization epsilon for SubLN
 * @return 0 on success, -1 on failure
 */
int tensor_metal_diff_attention(
    const float* q_cpu, const float* k_cache_cpu, const float* v_cache_cpu,
    float* out_cpu, const float* subln_weight,
    uint32_t num_tokens, uint32_t seq_len,
    uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
    uint32_t pos_offset, int32_t sliding_window,
    float lambda, float output_scale, float norm_eps);

/* ==========================================================================
 * GPU KV CACHE (GPU-Native Forward Pass)
 *
 * Keeps KV cache entirely on GPU to eliminate read/write transfers.
 * ========================================================================== */

/**
 * Initialize GPU KV cache.
 *
 * Allocates GPU buffers for K and V caches with dimensions:
 * K: [n_layer, n_ctx, n_head_kv * head_dim]
 * V: [n_layer, n_ctx, n_head_kv * head_dim]
 *
 * @param n_layer Number of transformer layers
 * @param n_ctx Maximum context length
 * @param n_head_kv Number of KV heads
 * @param head_dim Dimension per head
 * @return 0 on success, -1 on failure
 */
int tensor_metal_kv_cache_init(uint32_t n_layer, uint32_t n_ctx,
                               uint32_t n_head_kv, uint32_t head_dim);

/**
 * Store K/V vectors into GPU KV cache.
 *
 * @param k_buf GPU buffer with K vectors [num_tokens, kv_dim]
 * @param v_buf GPU buffer with V vectors [num_tokens, kv_dim]
 * @param layer Layer index (0 to n_layer-1)
 * @param pos Starting position in sequence
 * @param num_tokens Number of tokens to store
 * @param kv_dim KV dimension (n_head_kv * head_dim)
 * @return 0 on success, -1 on failure
 */
int tensor_metal_kv_cache_store(void* k_buf, void* v_buf,
                                uint32_t layer, uint32_t pos,
                                uint32_t num_tokens, uint32_t kv_dim);

/**
 * Get pointer to K cache GPU buffer for a layer.
 *
 * Returns a buffer starting at the layer's K cache.
 * Caller can use offset to access specific positions.
 *
 * @param layer Layer index
 * @return GPU buffer handle, or NULL if not initialized
 */
void* tensor_metal_kv_cache_get_k(uint32_t layer);

/**
 * Get pointer to V cache GPU buffer for a layer.
 *
 * @param layer Layer index
 * @return GPU buffer handle, or NULL if not initialized
 */
void* tensor_metal_kv_cache_get_v(uint32_t layer);

/**
 * Get current sequence length in KV cache.
 *
 * @return Current number of tokens in cache
 */
uint32_t tensor_metal_kv_cache_seq_len(void);

/**
 * Set current sequence length in KV cache.
 *
 * @param seq_len New sequence length
 */
void tensor_metal_kv_cache_set_seq_len(uint32_t seq_len);

/**
 * Free GPU KV cache.
 */
void tensor_metal_kv_cache_free(void);

/* ==========================================================================
 * COMPLETE GPU-NATIVE ATTENTION LAYER
 *
 * Single function that runs entire attention layer on GPU without CPU copies.
 * ========================================================================== */

/**
 * Complete GPU-native attention layer.
 *
 * Computes full attention layer with all intermediate data on GPU:
 * 1. RMSNorm input
 * 2. Q/K/V projections
 * 3. RoPE on Q/K (if enabled)
 * 4. Store K/V to cache
 * 5. Attention scores with causal mask
 * 6. Softmax
 * 7. Attention output (scores @ V)
 * 8. Output projection
 *
 * @param hidden_in_buf Input hidden state [M, n_embd] (GPU buffer)
 * @param hidden_out_buf Output hidden state [M, n_embd] (GPU buffer)
 * @param wq_buf Q projection weights [n_embd, n_embd] (GPU buffer)
 * @param wk_buf K projection weights [kv_dim, n_embd] (GPU buffer)
 * @param wv_buf V projection weights [kv_dim, n_embd] (GPU buffer)
 * @param wo_buf Output projection weights [n_embd, n_embd] (GPU buffer)
 * @param norm_buf Attention norm weights [n_embd] (GPU buffer)
 * @param layer_idx Layer index for KV cache
 * @param pos Current position in sequence
 * @param M Number of tokens
 * @param n_embd Model dimension
 * @param n_head Number of Q heads
 * @param n_head_kv Number of KV heads
 * @param norm_eps Normalization epsilon
 * @param rope_theta RoPE frequency base
 * @param use_rope Whether to apply RoPE
 * @return 0 on success, -1 on failure
 */
int tensor_metal_attention_layer_complete(
    void* hidden_in_buf, void* hidden_out_buf,
    void* wq_buf, void* wk_buf, void* wv_buf, void* wo_buf,
    void* norm_buf,
    uint32_t layer_idx,
    uint32_t pos,
    uint32_t M,
    uint32_t n_embd, uint32_t n_head, uint32_t n_head_kv,
    float norm_eps, float rope_theta,
    int use_rope);

/**
 * Initialize buffers for GPU-native forward pass.
 *
 * Allocates all persistent GPU buffers needed for forward pass:
 * - Hidden state buffers (ping-pong)
 * - Intermediate projection buffers
 * - Attention scratch buffers
 * - KV cache
 *
 * @param n_layer Number of layers
 * @param n_ctx Maximum context length
 * @param n_embd Model dimension
 * @param n_ff FFN intermediate dimension
 * @param n_head Number of Q heads
 * @param n_head_kv Number of KV heads
 * @param n_vocab Vocabulary size
 * @return 0 on success, -1 on failure
 */
int tensor_metal_init_gpu_forward(uint32_t n_layer, uint32_t n_ctx,
                                   uint32_t n_embd, uint32_t n_ff,
                                   uint32_t n_head, uint32_t n_head_kv,
                                   uint32_t n_vocab);

/**
 * Free GPU-native forward pass resources.
 */
void tensor_metal_free_gpu_forward(void);

/**
 * Check if GPU-native forward pass is initialized.
 *
 * @return 1 if initialized, 0 otherwise
 */
int tensor_metal_gpu_forward_available(void);

/* ==========================================================================
 * PERSISTENT GPU HIDDEN STATE - Zero-copy forward pass
 *
 * These functions enable keeping hidden state on GPU throughout the entire
 * forward pass, eliminating per-layer CPU-GPU transfers during generation.
 * ========================================================================== */

/**
 * Get persistent GPU hidden state buffer.
 * @return GPU buffer handle, or NULL if not initialized
 */
void* tensor_metal_get_gpu_hidden(void);

/**
 * Get persistent GPU residual buffer.
 */
void* tensor_metal_get_gpu_residual(void);

/**
 * Get persistent GPU normed buffer (post-normalization hidden).
 */
void* tensor_metal_get_gpu_normed(void);

/**
 * Upload hidden state from CPU to persistent GPU hidden buffer.
 *
 * @param src CPU float array
 * @param num_elements Number of floats
 * @return 0 on success
 */
int tensor_metal_upload_hidden(const float* src, uint32_t num_elements);

/**
 * Upload normalized hidden state to persistent GPU normed buffer.
 * Used for persistent FFN which reads from gpuNormed.
 *
 * @param src CPU float array (post-norm hidden state)
 * @param num_elements Number of floats
 * @return 0 on success
 */
int tensor_metal_upload_normed(const float* src, uint32_t num_elements);

/**
 * Download from persistent GPU hidden buffer to CPU.
 *
 * @param dst Pre-allocated CPU float array
 * @param num_elements Number of floats
 * @return 0 on success
 */
int tensor_metal_download_hidden(float* dst, uint32_t num_elements);

/**
 * GPU-to-GPU copy: residual = hidden (for residual connections).
 * No CPU involvement.
 */
int tensor_metal_copy_hidden_to_residual(uint32_t num_elements);

/**
 * GPU in-place add: hidden += residual
 * No CPU involvement.
 */
int tensor_metal_add_residual_to_hidden(uint32_t num_elements);

/**
 * RMSNorm on persistent buffers: normed = rmsnorm(hidden, weights)
 * Reads from gpuHidden, writes to gpuNormed.
 */
int tensor_metal_rmsnorm_hidden_to_normed(void* weights_buf,
                                           uint32_t M, uint32_t n_embd, float eps);

/**
 * Matmul from normed to hidden: hidden = normed @ W^T
 * Uses gpuNormed as input, gpuHidden as output.
 */
int tensor_metal_matmul_normed_to_hidden(void* W_buf,
                                          uint32_t M, uint32_t N, uint32_t K);

/**
 * Complete SwiGLU FFN on persistent GPU buffers.
 *
 * Reads from gpuNormed, writes to gpuHidden.
 * Uses gpuFFNGate and gpuFFNUp as internal scratch.
 * Does NOT add residual.
 *
 * hidden_out = SiLU(normed @ W1^T) * (normed @ W3^T) @ W2^T
 */
int tensor_metal_ffn_swiglu_gpu_persistent(void* W1_buf, void* W3_buf, void* W2_buf,
                                            uint32_t M, uint32_t n_ff, uint32_t n_embd);

/**
 * Complete FFN with residual in SINGLE command buffer.
 *
 * Combines 6 operations with no intermediate sync:
 * 1. Copy gpuHidden to gpuResidual
 * 2. FFN gate projection: normed @ W1^T
 * 3. FFN up projection: normed @ W3^T
 * 4. SiLU multiply: SiLU(gate) * up
 * 5. FFN down projection: result @ W2^T
 * 6. Residual add: hidden += residual
 *
 * Requires: gpuHidden has pre-FFN hidden state, gpuNormed has normalized hidden
 * Output: gpuHidden contains FFN output with residual added
 */
int tensor_metal_ffn_swiglu_with_residual(void* W1_buf, void* W3_buf, void* W2_buf,
                                           uint32_t M, uint32_t n_ff, uint32_t n_embd);

/**
 * Complete FFN with norm AND residual in SINGLE command buffer (7 ops).
 *
 * This is the most optimized FFN path:
 * 1. Copy gpuHidden to gpuResidual
 * 2. RMSNorm gpuHidden to gpuNormed
 * 3. FFN gate projection
 * 4. FFN up projection
 * 5. SiLU multiply
 * 6. FFN down projection
 * 7. Residual add
 *
 * Requires: gpuHidden has pre-FFN hidden state, norm weights on GPU
 * Output: gpuHidden contains FFN output with residual added
 * Only 1 upload + 1 download per layer (no separate norm upload needed)
 */
int tensor_metal_ffn_complete_with_norm(void* norm_buf,
                                         void* W1_buf, void* W3_buf, void* W2_buf,
                                         uint32_t M, uint32_t n_ff, uint32_t n_embd,
                                         float eps);

/**
 * Output projection to CPU logits: logits = hidden @ W^T
 * Reads from gpuHidden, downloads result to CPU.
 */
int tensor_metal_output_proj_download(float* logits_out, void* W_buf,
                                       uint32_t M, uint32_t n_vocab, uint32_t n_embd);

#ifdef __cplusplus
}
#endif

#endif /* DET_TENSOR_METAL_H */
