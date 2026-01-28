/**
 * DET SSM (State Space Model) - Mamba Support
 * ============================================
 *
 * DET-native implementation of Mamba/SSM architecture.
 * Maps SSM hidden states to DET physics quantities.
 *
 * DET-SSM Conceptual Mapping:
 * - h_k hidden state -> Node state (F, a, q, tau)
 * - A matrix (decay) -> Presence formula
 * - B*x_k input -> Bond flux
 * - C readout -> Witness token
 * - D skip connection -> Direct substrate coupling
 * - delta_t discretization -> Proper time step
 */

#ifndef DET_SSM_H
#define DET_SSM_H

#include "det_tensor.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * SSM CONFIGURATION
 * ========================================================================== */

/**
 * SSM layer configuration (Mamba architecture)
 */
typedef struct {
    int32_t d_model;    /* Model dimension */
    int32_t d_inner;    /* Inner (expanded) dimension, typically 2*d_model */
    int32_t d_state;    /* SSM state dimension (typically 16) */
    int32_t d_conv;     /* Convolution kernel width (typically 4) */
    int32_t dt_rank;    /* Rank of delta_t projection */
    float dt_min;       /* Minimum delta_t */
    float dt_max;       /* Maximum delta_t */
    float dt_init;      /* Initial delta_t */
    float dt_scale;     /* Delta_t scaling factor */
} DetSSMConfig;

/**
 * Default SSM configuration for Mamba-130M
 */
static inline DetSSMConfig det_ssm_default_config(int32_t d_model) {
    return (DetSSMConfig){
        .d_model = d_model,
        .d_inner = d_model * 2,
        .d_state = 16,
        .d_conv = 4,
        .dt_rank = (d_model + 15) / 16,  /* ceil(d_model / 16) */
        .dt_min = 0.001f,
        .dt_max = 0.1f,
        .dt_init = 0.001f,
        .dt_scale = 1.0f,
    };
}

/* ==========================================================================
 * SSM LAYER WEIGHTS
 * ========================================================================== */

/**
 * SSM layer weights (Mamba architecture)
 *
 * Weight naming follows GGUF/llama.cpp convention:
 *   blk.%d.ssm_in.weight    -> ssm_in_proj
 *   blk.%d.ssm_conv1d.*     -> ssm_conv1d_weight/bias
 *   blk.%d.ssm_x.weight     -> ssm_x_proj
 *   blk.%d.ssm_dt.weight    -> ssm_dt_proj
 *   blk.%d.ssm_dt.bias      -> ssm_dt_proj_bias
 *   blk.%d.ssm_A_log        -> ssm_A_log
 *   blk.%d.ssm_D            -> ssm_D
 *   blk.%d.ssm_out.weight   -> ssm_out_proj
 */
typedef struct {
    DetTensor* ssm_in_proj;       /* [d_model, 2*d_inner] input projection */
    DetTensor* ssm_conv1d_weight; /* [d_inner, 1, d_conv] causal conv kernel */
    DetTensor* ssm_conv1d_bias;   /* [d_inner] conv bias */
    DetTensor* ssm_x_proj;        /* [d_inner, dt_rank + 2*d_state] x->delta,B,C */
    DetTensor* ssm_dt_proj;       /* [dt_rank, d_inner] delta projection */
    DetTensor* ssm_dt_proj_bias;  /* [d_inner] delta bias */
    DetTensor* ssm_A_log;         /* [d_inner, d_state] log of A matrix (negative) */
    DetTensor* ssm_D;             /* [d_inner] skip connection */
    DetTensor* ssm_out_proj;      /* [d_inner, d_model] output projection */
    DetTensor* ssm_norm;          /* [d_model] pre-SSM normalization */
    float* ssm_A_negexp;          /* Precomputed -exp(A_log) [d_inner * d_state] */
} DetSSMWeights;

/* ==========================================================================
 * SSM STATE CACHE
 * ========================================================================== */

/**
 * SSM state cache for efficient autoregressive generation
 *
 * Unlike KV cache which grows with sequence length, SSM state
 * is fixed size - this is the key advantage of SSM models.
 */
typedef struct {
    DetTensor* h;           /* [n_layer, d_inner, d_state] hidden state */
    DetTensor* conv_state;  /* [n_layer, d_inner, d_conv-1] conv history */
    int32_t n_layers;       /* Number of SSM layers */
    int32_t d_inner;        /* Inner dimension */
    int32_t d_state;        /* State dimension */
    int32_t d_conv;         /* Convolution width */
    int32_t dt_rank;        /* Delta projection rank */
    bool initialized;       /* Whether cache has been initialized */

    /* Pre-allocated workspace buffers for det_ssm_forward() */
    /* Avoids malloc/free per call - significant performance gain */
    int32_t workspace_seq_len;  /* Max seq_len workspace is sized for */
    float* xz;              /* [seq_len, 2*d_inner] */
    float* x_conv;          /* [seq_len, d_inner] */
    float* x_ssm;           /* [seq_len, d_inner] */
    float* x_proj;          /* [seq_len, dt_rank + 2*d_state] */
    float* delta;           /* [seq_len, d_inner] */
    float* y;               /* [seq_len, d_inner] */
    float* y_gated;         /* [seq_len, d_inner] */
} DetSSMCache;

/**
 * Create SSM cache with pre-allocated workspace
 *
 * @param n_layers Number of SSM layers
 * @param d_inner Inner (expanded) dimension
 * @param d_state State dimension
 * @param d_conv Convolution kernel width
 * @param dt_rank Delta projection rank
 * @param max_seq_len Maximum sequence length for workspace
 */
DetSSMCache* det_ssm_cache_create(int32_t n_layers, int32_t d_inner,
                                   int32_t d_state, int32_t d_conv,
                                   int32_t dt_rank, int32_t max_seq_len);

/**
 * Free SSM cache
 */
void det_ssm_cache_free(DetSSMCache* cache);

/**
 * Reset SSM cache (for new sequence)
 */
void det_ssm_cache_reset(DetSSMCache* cache);

/* ==========================================================================
 * SSM FORWARD PASS
 * ========================================================================== */

/**
 * Full SSM layer forward pass
 *
 * Computes the complete Mamba layer:
 *   1. Input projection: x -> (z, x_proj)
 *   2. Causal convolution on x_proj
 *   3. SiLU activation
 *   4. SSM selective scan
 *   5. Gated output: y * SiLU(z)
 *   6. Output projection
 *
 * @param output Output tensor [seq_len, d_model]
 * @param input Input tensor [seq_len, d_model]
 * @param weights SSM layer weights
 * @param config SSM configuration
 * @param cache SSM state cache (for autoregressive generation)
 * @param layer_idx Layer index (for cache indexing)
 * @param ssm_output_pre_gate Optional: stores SSM output BEFORE gating [seq_len, d_inner]
 *                            Used by SambaY to cache for cross-decoder. Can be NULL.
 * @return 0 on success, negative on error
 */
int det_ssm_forward(DetTensor* output,
                    const DetTensor* input,
                    const DetSSMWeights* weights,
                    const DetSSMConfig* config,
                    DetSSMCache* cache,
                    int32_t layer_idx,
                    float* ssm_output_pre_gate);

/**
 * SSM selective scan (core recurrence)
 *
 * Implements the discretized state space model:
 *   h_k = A_bar * h_{k-1} + B_bar * x_k
 *   y_k = C * h_k + D * x_k
 *
 * Where A_bar = exp(delta * A), B_bar = delta * B
 *
 * @param y Output [seq_len, d_inner]
 * @param x Input after conv [seq_len, d_inner]
 * @param delta Time step [seq_len, d_inner]
 * @param A_log A matrix [d_inner, d_state] (log-space, negative) - used if A_negexp is NULL
 * @param A_negexp Precomputed -exp(A_log) [d_inner, d_state] - major speedup, can be NULL
 * @param B B projection [seq_len, d_state]
 * @param C C projection [seq_len, d_state]
 * @param D Skip connection [d_inner]
 * @param h Hidden state [d_inner, d_state] (updated in place)
 * @param d_inner Inner dimension
 * @param d_state State dimension
 * @param seq_len Sequence length
 * @return 0 on success, negative on error
 */
int det_ssm_selective_scan(float* y,
                           const float* x,
                           const float* delta,
                           const float* A_log,
                           const float* A_negexp,
                           const float* B,
                           const float* C,
                           const float* D,
                           float* h,
                           int32_t d_inner,
                           int32_t d_state,
                           int32_t seq_len);

/**
 * Causal 1D convolution with state
 *
 * Computes causal conv1d: out[t] = sum_{k=0}^{d_conv-1} w[k] * x[t-k]
 * Maintains state for autoregressive generation.
 *
 * @param output Output [seq_len, d_inner]
 * @param input Input [seq_len, d_inner]
 * @param weight Conv weights [d_inner, d_conv]
 * @param bias Conv bias [d_inner] (can be NULL)
 * @param conv_state State [d_inner, d_conv-1] (updated in place)
 * @param d_inner Inner dimension
 * @param d_conv Convolution kernel width
 * @param seq_len Sequence length
 * @return 0 on success, negative on error
 */
int det_conv1d_causal(float* output,
                      const float* input,
                      const float* weight,
                      const float* bias,
                      float* conv_state,
                      int32_t d_inner,
                      int32_t d_conv,
                      int32_t seq_len);

/**
 * SiLU gating operation: out = x * SiLU(gate)
 *
 * @param output Output [n]
 * @param x Input values [n]
 * @param gate Gate values [n]
 * @param n Number of elements
 * @return 0 on success
 */
int det_silu_gate(float* output,
                  const float* x,
                  const float* gate,
                  int32_t n);

/* ==========================================================================
 * GMU (GATED MEMORY UNIT) - SambaY Architecture
 * ========================================================================== */

/**
 * GMU forward pass
 *
 * Gated Memory Unit reuses the hidden state from a previous SSM layer
 * instead of computing full attention/SSM. This is the key efficiency
 * innovation in SambaY architecture.
 *
 * GMU computes: out = swiglu(x, cached_ssm_output)
 *             = SiLU(W_gate @ x) * (W_up @ cached_ssm_output)
 *
 * @param output Output [seq_len, d_model]
 * @param input Input hidden states [seq_len, d_model]
 * @param cached_ssm Cached SSM output from final Mamba layer [seq_len, d_inner]
 * @param w_gate Gate projection weights [d_inner, d_model]
 * @param w_up Up projection weights [d_inner, d_inner]
 * @param w_down Down projection weights [d_model, d_inner]
 * @param d_model Model dimension
 * @param d_inner Inner dimension
 * @param seq_len Sequence length
 * @return 0 on success, negative on error
 */
int det_gmu_forward(float* output,
                    const float* input,
                    const float* cached_ssm,
                    const float* w_gate,
                    const float* w_up,
                    const float* w_down,
                    int32_t d_model,
                    int32_t d_inner,
                    int32_t seq_len);

/**
 * Simplified GMU using swiglu directly on cached state
 *
 * This version matches phi4flash's yoco_cross behavior:
 *   out = out_proj(swiglu(in_proj(x), cached_ssm))
 *
 * @param output Output [seq_len, d_model]
 * @param input Input [seq_len, d_model]
 * @param cached_ssm Cached SSM output [seq_len, d_inner]
 * @param in_proj Input projection [d_inner, d_model]
 * @param out_proj Output projection [d_model, d_inner]
 * @param d_model Model dimension
 * @param d_inner Inner dimension
 * @param seq_len Sequence length
 * @return 0 on success
 */
int det_gmu_swiglu(float* output,
                   const float* input,
                   const float* cached_ssm,
                   const float* in_proj,
                   const float* out_proj,
                   int32_t d_model,
                   int32_t d_inner,
                   int32_t seq_len);

/* ==========================================================================
 * SLIDING WINDOW ATTENTION HELPERS
 * ========================================================================== */

/**
 * Apply sliding window mask to attention scores
 *
 * Sets scores[i, j] = -inf for |i - j| > window_size
 *
 * @param scores Attention scores [seq_q, seq_k]
 * @param seq_q Query sequence length
 * @param seq_k Key sequence length
 * @param window_size Sliding window size
 * @param pos_offset Position offset for autoregressive generation
 * @return 0 on success
 */
int det_sliding_window_mask(float* scores,
                            int32_t seq_q,
                            int32_t seq_k,
                            int32_t window_size,
                            int32_t pos_offset);

/* ==========================================================================
 * DET PHYSICS INTEGRATION
 * ========================================================================== */

/**
 * SSM layer statistics for DET truthfulness tracking
 */
typedef struct {
    float state_norm;      /* ||h|| - magnitude of hidden state */
    float state_delta;     /* ||h_new - h_old|| - state change rate */
    float gate_activation; /* Mean SiLU(z) gate activation */
    float skip_ratio;      /* Ratio of D*x to total output */
} DetSSMLayerStats;

/**
 * Get statistics from last SSM forward pass
 *
 * These map to DET physics quantities:
 * - state_norm -> Resource (F) indicator
 * - state_delta -> Agency (a) measure
 * - gate_activation -> Presence strength
 */
void det_ssm_get_stats(DetSSMLayerStats* stats);

/* ==========================================================================
 * UTILITY FUNCTIONS
 * ========================================================================== */

/**
 * Check if layer is SSM (vs attention) based on weight patterns
 *
 * @param gguf GGUF context
 * @param layer_idx Layer index
 * @return true if layer has SSM weights
 */
bool det_is_ssm_layer(void* gguf, int32_t layer_idx);

/**
 * Get SSM config from GGUF metadata
 *
 * Reads Mamba-specific parameters:
 * - mamba.d_inner
 * - mamba.d_state
 * - mamba.d_conv
 * - mamba.dt_rank
 */
DetSSMConfig det_ssm_config_from_gguf(void* gguf);

#ifdef __cplusplus
}
#endif

#endif /* DET_SSM_H */
