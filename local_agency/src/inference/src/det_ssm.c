/**
 * DET SSM (State Space Model) - Implementation
 * =============================================
 *
 * CPU implementation of Mamba/SSM architecture.
 * Maps SSM hidden states to DET physics quantities.
 */

#include "det_ssm.h"
#include "det_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE 1
#endif

/* ARM NEON SIMD for element-wise operations */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

/* Metal GPU support (Phase 26.16) */
#ifdef DET_USE_METAL
#include "det_tensor_metal.h"
extern int g_metal_available;
#endif

/* Debug flag: print intermediate values for layer 0 SSM */
#define DEBUG_SSM_LAYER0 0

/* SSM timing debug (set to 1 to enable detailed SSM timing) */
#define DEBUG_SSM_TIMING 0

#if DEBUG_SSM_TIMING
#include <sys/time.h>
static double ssm_get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
static double t_in_proj = 0, t_conv1d = 0, t_silu = 0, t_x_proj = 0;
static double t_dt_proj = 0, t_selective_scan = 0, t_gate = 0, t_out_proj = 0;
static int ssm_timing_count = 0;
#endif

#if DEBUG_SSM_LAYER0
static void debug_ssm_rms(const char* label, const float* data, int n, int show_first) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += data[i] * data[i];
    }
    float rms = sqrtf(sum_sq / n);
    if (show_first && n >= 4) {
        fprintf(stderr, "    SSM %s: rms=%.6f [0:4]=[%.6f, %.6f, %.6f, %.6f]\n",
                label, rms, data[0], data[1], data[2], data[3]);
    } else {
        fprintf(stderr, "    SSM %s: rms=%.6f\n", label, rms);
    }
}
#endif

/* ==========================================================================
 * INTERNAL STATE FOR STATS TRACKING
 * ========================================================================== */

static DetSSMLayerStats g_last_ssm_stats = {0};

/* ==========================================================================
 * SSM CACHE MANAGEMENT
 * ========================================================================== */

DetSSMCache* det_ssm_cache_create(int32_t n_layers, int32_t d_inner,
                                   int32_t d_state, int32_t d_conv,
                                   int32_t dt_rank, int32_t max_seq_len) {
    DetSSMCache* cache = calloc(1, sizeof(DetSSMCache));
    if (!cache) return NULL;

    cache->n_layers = n_layers;
    cache->d_inner = d_inner;
    cache->d_state = d_state;
    cache->d_conv = d_conv;
    cache->dt_rank = dt_rank;

    /* Allocate hidden state: [n_layers, d_inner, d_state] */
    int32_t h_shape[3] = {n_layers, d_inner, d_state};
    cache->h = det_tensor_create(3, h_shape, DET_DTYPE_F32);
    if (!cache->h) {
        free(cache);
        return NULL;
    }

    /* Allocate conv state: [n_layers, d_inner, d_conv-1] */
    int32_t conv_shape[3] = {n_layers, d_inner, d_conv - 1};
    cache->conv_state = det_tensor_create(3, conv_shape, DET_DTYPE_F32);
    if (!cache->conv_state) {
        det_tensor_release(cache->h);
        free(cache);
        return NULL;
    }

    /* Allocate workspace buffers (avoids malloc/free per forward call) */
    cache->workspace_seq_len = max_seq_len > 0 ? max_seq_len : 1;
    int32_t proj_dim = dt_rank + 2 * d_state;

    cache->xz = malloc(cache->workspace_seq_len * 2 * d_inner * sizeof(float));
    cache->x_conv = malloc(cache->workspace_seq_len * d_inner * sizeof(float));
    cache->x_ssm = malloc(cache->workspace_seq_len * d_inner * sizeof(float));
    cache->x_proj = malloc(cache->workspace_seq_len * proj_dim * sizeof(float));
    cache->delta = malloc(cache->workspace_seq_len * d_inner * sizeof(float));
    cache->y = malloc(cache->workspace_seq_len * d_inner * sizeof(float));
    cache->y_gated = malloc(cache->workspace_seq_len * d_inner * sizeof(float));

    if (!cache->xz || !cache->x_conv || !cache->x_ssm || !cache->x_proj ||
        !cache->delta || !cache->y || !cache->y_gated) {
        det_ssm_cache_free(cache);
        return NULL;
    }

    /* Initialize to zero */
    det_ssm_cache_reset(cache);
    cache->initialized = true;

    return cache;
}

void det_ssm_cache_free(DetSSMCache* cache) {
    if (!cache) return;
    det_tensor_release(cache->h);
    det_tensor_release(cache->conv_state);
    /* Free workspace buffers */
    free(cache->xz);
    free(cache->x_conv);
    free(cache->x_ssm);
    free(cache->x_proj);
    free(cache->delta);
    free(cache->y);
    free(cache->y_gated);
    free(cache);
}

void det_ssm_cache_reset(DetSSMCache* cache) {
    if (!cache) return;

    if (cache->h && cache->h->data) {
        memset(cache->h->data, 0, cache->h->data_size);
    }
    if (cache->conv_state && cache->conv_state->data) {
        memset(cache->conv_state->data, 0, cache->conv_state->data_size);
    }
}

/* ==========================================================================
 * CORE SSM OPERATIONS
 * ========================================================================== */

/**
 * SiLU (Swish) activation: x * sigmoid(x)
 */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * Softplus: log(1 + exp(x))
 * Used for delta_t to ensure positivity
 */
static inline float softplus(float x) {
    if (x > 20.0f) return x;  /* Avoid overflow */
    return logf(1.0f + expf(x));
}

#ifdef USE_NEON
/**
 * Fast NEON sigmoid approximation using polynomial
 * Good accuracy for x in [-8, 8], saturates to 0/1 outside
 */
static inline float32x4_t neon_sigmoid_approx(float32x4_t x) {
    /* Clamp to reasonable range to avoid overflow */
    float32x4_t min_val = vdupq_n_f32(-8.0f);
    float32x4_t max_val = vdupq_n_f32(8.0f);
    x = vmaxq_f32(x, min_val);
    x = vminq_f32(x, max_val);

    /* Polynomial approximation: sigmoid(x) ≈ 0.5 + 0.25*x - 0.0078125*x^3
     * Accurate to ~1% for |x| < 5 */
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t quarter = vdupq_n_f32(0.25f);
    float32x4_t c3 = vdupq_n_f32(-0.0078125f);

    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);

    float32x4_t result = vfmaq_f32(half, quarter, x);  /* 0.5 + 0.25*x */
    result = vfmaq_f32(result, c3, x3);  /* + c3*x^3 */

    /* Clamp to [0, 1] */
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t one = vdupq_n_f32(1.0f);
    result = vmaxq_f32(result, zero);
    result = vminq_f32(result, one);

    return result;
}

/**
 * NEON SiLU: x * sigmoid(x)
 */
static inline float32x4_t neon_silu(float32x4_t x) {
    return vmulq_f32(x, neon_sigmoid_approx(x));
}

/**
 * Apply SiLU in-place using NEON (4 elements at a time)
 */
static void neon_silu_inplace(float* data, int32_t n) {
    int32_t i = 0;

    /* NEON vectorized path */
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        float32x4_t result = neon_silu(v);
        vst1q_f32(data + i, result);
    }

    /* Scalar remainder */
    for (; i < n; i++) {
        data[i] = silu(data[i]);
    }
}

/**
 * Apply SiLU gating: out[i] = y[i] * silu(z[i])
 */
static void neon_silu_gate(float* out, const float* y, const float* z, int32_t n) {
    int32_t i = 0;

    /* NEON vectorized path */
    for (; i + 3 < n; i += 4) {
        float32x4_t v_y = vld1q_f32(y + i);
        float32x4_t v_z = vld1q_f32(z + i);
        float32x4_t gate = neon_silu(v_z);
        float32x4_t result = vmulq_f32(v_y, gate);
        vst1q_f32(out + i, result);
    }

    /* Scalar remainder */
    for (; i < n; i++) {
        out[i] = y[i] * silu(z[i]);
    }
}

/**
 * Apply softplus with bias: out[i] = softplus(delta[i] + bias[i])
 * NEON-optimized using fast exp approximation
 */
static void neon_softplus_bias(float* delta, const float* bias, int32_t n) {
    /* Use scalar path - softplus needs accurate exp which is hard to vectorize well */
    for (int32_t i = 0; i < n; i++) {
        float val = delta[i];
        if (bias) val += bias[i];
        delta[i] = softplus(val);
    }
}
#endif /* USE_NEON */

int det_silu_gate(float* output,
                  const float* x,
                  const float* gate,
                  int32_t n) {
    if (!output || !x || !gate) return -1;

    for (int32_t i = 0; i < n; i++) {
        output[i] = x[i] * silu(gate[i]);
    }
    return 0;
}

int det_conv1d_causal(float* output,
                      const float* input,
                      const float* weight,
                      const float* bias,
                      float* conv_state,
                      int32_t d_inner,
                      int32_t d_conv,
                      int32_t seq_len) {
    if (!output || !input || !weight) return -1;

    /* For each channel */
    for (int32_t c = 0; c < d_inner; c++) {
        const float* w = weight + c * d_conv;

        /* Process each position */
        for (int32_t t = 0; t < seq_len; t++) {
            float sum = 0.0f;

            /* Convolve: out[t] = sum_{k=0}^{d_conv-1} w[k] * x[t - (d_conv-1) + k]
             *
             * PyTorch conv1d with causal (left) padding of d_conv-1:
             *   out[t] = sum_k weight[k] * padded_input[t + k]
             *   where padded_input[i] = 0 for i < d_conv-1
             *
             * This maps to accessing input at position: t + k - (d_conv - 1)
             * which equals: t - d_conv + 1 + k
             */
            for (int32_t k = 0; k < d_conv; k++) {
                int32_t src_t = t - (d_conv - 1) + k;  /* Fixed indexing */

                float x_val;
                if (src_t >= 0 && src_t < seq_len) {
                    /* Within current sequence */
                    x_val = input[src_t * d_inner + c];
                } else if (conv_state && src_t >= -(d_conv - 1) && src_t < 0) {
                    /* From conv state (previous tokens) */
                    int32_t state_idx = src_t + (d_conv - 1);  /* Map -3..-1 to 0..2 */
                    x_val = conv_state[c * (d_conv - 1) + state_idx];
                } else {
                    x_val = 0.0f;  /* Zero padding */
                }

                sum += w[k] * x_val;
            }

            /* Add bias */
            if (bias) {
                sum += bias[c];
            }

            output[t * d_inner + c] = sum;
        }

        /* Update conv state with last (d_conv-1) values.
         * State holds the last (d_conv-1) tokens for the next forward pass.
         *
         * Algorithm is safe because we process k in increasing order:
         * - When reading from input (src_t >= 0), no aliasing issue
         * - When shifting from old state, old_idx = seq_len + k > k (for seq_len > 0),
         *   so we never read from a position we've already written in this pass.
         */
        if (conv_state && seq_len > 0) {
            /* For short sequences, we need to shift from old state first.
             * Use a temporary buffer to avoid potential aliasing. */
            float temp_state[8];  /* d_conv-1 is typically 3, max 7 for safety */
            int32_t state_size = d_conv - 1;

            if (seq_len < state_size && state_size <= 8) {
                /* Copy old state to temp to avoid self-overwrite */
                for (int32_t k = 0; k < state_size; k++) {
                    temp_state[k] = conv_state[c * state_size + k];
                }
                for (int32_t k = 0; k < state_size; k++) {
                    int32_t src_t = seq_len - state_size + k;
                    if (src_t >= 0) {
                        conv_state[c * state_size + k] = input[src_t * d_inner + c];
                    } else {
                        int32_t old_idx = src_t + state_size;
                        if (old_idx >= 0) {
                            conv_state[c * state_size + k] = temp_state[old_idx];
                        }
                    }
                }
            } else {
                /* seq_len >= state_size: all values come from input, no aliasing */
                for (int32_t k = 0; k < state_size; k++) {
                    int32_t src_t = seq_len - state_size + k;
                    conv_state[c * state_size + k] = input[src_t * d_inner + c];
                }
            }
        }
    }

    return 0;
}

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
                           int32_t seq_len) {
    if (!y || !x || !delta || !B || !C || !h) return -1;
    if (!A_log && !A_negexp) return -1;  /* Need at least one A source */

    /*
     * Core SSM recurrence for each time step:
     *   A_bar = exp(delta * A)  -- A is stored as log(-A), so A_bar = exp(-delta * exp(A_log))
     *   B_bar = delta * B
     *   h_new = A_bar * h + B_bar * x
     *   y = C * h_new + D * x
     *
     * When A_negexp is precomputed, we skip the -exp(A_log) computation per element,
     * which is a significant speedup (expf is expensive).
     *
     * Note: The inner d_inner loop is parallelizable over i since each channel has
     * independent state h[i,:], but for single-token generation the dispatch overhead
     * would exceed the benefit.
     */
    for (int32_t t = 0; t < seq_len; t++) {
        const float* x_t = x + t * d_inner;
        const float* delta_t = delta + t * d_inner;
        const float* B_t = B + t * d_state;
        const float* C_t = C + t * d_state;
        float* y_t = y + t * d_inner;

        /* Process each channel (d_inner loop is parallelizable over i since each
         * channel has independent state h[i,:], but for single-token generation
         * the overhead would exceed benefit) */
        for (int32_t i = 0; i < d_inner; i++) {
            float dt = delta_t[i];
            float x_i = x_t[i];
            float y_i = 0.0f;

            /* Skip connection */
            if (D) {
                y_i = D[i] * x_i;
            }

            /* State update for each state dimension */
            float* h_i = h + i * d_state;
            for (int32_t j = 0; j < d_state; j++) {
                float h_old = h_i[j];

                float A_ij;
                if (A_negexp) {
                    A_ij = A_negexp[i * d_state + j];
                } else {
                    A_ij = -expf(A_log[i * d_state + j]);
                }

                float A_bar = expf(dt * A_ij);
                float h_new = A_bar * h_old + (dt * B_t[j]) * x_i;

                if (!isfinite(h_new)) h_new = 0.0f;
                h_i[j] = h_new;

                float y_contrib = C_t[j] * h_new;
                if (isfinite(y_contrib)) {
                    y_i += y_contrib;
                }
            }

            y_t[i] = y_i;
        }
    }

    return 0;
}

int det_ssm_forward(DetTensor* output,
                    const DetTensor* input,
                    const DetSSMWeights* weights,
                    const DetSSMConfig* config,
                    DetSSMCache* cache,
                    int32_t layer_idx,
                    float* ssm_output_pre_gate,
                    const float* cached_ssm_input) {
    if (!output || !input || !weights || !config) return -1;

    int32_t seq_len = input->shape[0];
    int32_t d_model = config->d_model;
    int32_t d_inner = config->d_inner;
    int32_t d_state = config->d_state;
    int32_t d_conv = config->d_conv;
    int32_t dt_rank = config->dt_rank;

    /* Detect if this is a cross-decoder SSM (smaller in_proj)
     * Cross-decoder SSM: in_proj is [d_inner, d_model]
     * Standard SSM: in_proj is [2*d_inner, d_model]
     */
    bool is_cross_decoder_ssm = false;
    if (weights->ssm_in_proj) {
        int32_t proj_size = (int32_t)weights->ssm_in_proj->shape[1];  /* Output dim */
        if (proj_size == d_inner) {
            is_cross_decoder_ssm = true;
        }
    }

    /*
     * YOCO Cross-decoder fast path:
     * If this is a cross-decoder SSM AND we have cached SSM output, use simplified swiglu path:
     *   out = out_proj(swiglu(in_proj(x), cached_ssm))
     * This skips conv1d and selective scan entirely.
     */
    if (is_cross_decoder_ssm && cached_ssm_input != NULL) {
        const float* in_proj = weights->ssm_in_proj ?
            (const float*)weights->ssm_in_proj->data : NULL;
        const float* out_proj = weights->ssm_out_proj ?
            (const float*)weights->ssm_out_proj->data : NULL;

        if (in_proj && out_proj) {
            int ret = det_gmu_swiglu((float*)output->data,
                                      (const float*)input->data,
                                      cached_ssm_input,
                                      in_proj,
                                      out_proj,
                                      d_model,
                                      d_inner,
                                      seq_len);
            /* For YOCO, the cached value passes through unchanged */
            if (ssm_output_pre_gate) {
                memcpy(ssm_output_pre_gate, cached_ssm_input, seq_len * d_inner * sizeof(float));
            }
            return ret;
        }
    }

    /* Use pre-allocated workspace from cache if available and large enough */
    float* xz;
    float* x_conv;
    float* x_ssm;
    float* x_proj;
    float* delta;
    float* y;
    float* y_gated;
    bool use_workspace = cache && cache->workspace_seq_len >= seq_len;

    if (use_workspace) {
        /* Use pre-allocated buffers from cache */
        xz = cache->xz;
        x_conv = cache->x_conv;
        x_ssm = cache->x_ssm;
        x_proj = cache->x_proj;
        delta = cache->delta;
        y = cache->y;
        y_gated = cache->y_gated;
    } else {
        /* Fallback to per-call allocation (for no-cache case or oversized seq) */
        xz = malloc(seq_len * 2 * d_inner * sizeof(float));
        x_conv = malloc(seq_len * d_inner * sizeof(float));
        x_ssm = malloc(seq_len * d_inner * sizeof(float));
        x_proj = malloc(seq_len * (dt_rank + 2 * d_state) * sizeof(float));
        delta = malloc(seq_len * d_inner * sizeof(float));
        y = malloc(seq_len * d_inner * sizeof(float));
        y_gated = malloc(seq_len * d_inner * sizeof(float));

        if (!xz || !x_conv || !x_ssm || !x_proj || !delta || !y || !y_gated) {
            free(xz); free(x_conv); free(x_ssm); free(x_proj);
            free(delta); free(y); free(y_gated);
            return -1;
        }
    }

    /* B and C are accessed as pointer views into x_proj (no copy needed) */
    float* B = NULL;
    float* C = NULL;

    const float* input_data = (const float*)input->data;
    float* output_data = (float*)output->data;

#if DEBUG_SSM_TIMING
    double t0, t1;
    t0 = ssm_get_time_ms();
#endif

    /* 1. Input projection:
     * Standard SSM: xz = input @ ssm_in_proj.T, where W is [2*d_inner, d_model]
     * Cross-decoder SSM: x = input @ ssm_in_proj.T, where W is [d_inner, d_model]
     *                    gate z is initialized to 1.0 (no gating)
     */
    if (!weights->ssm_in_proj) {
        /* No projection weight - zero-fill to prevent undefined behavior */
        memset(xz, 0, seq_len * 2 * d_inner * sizeof(float));
        memset(x_conv, 0, seq_len * d_inner * sizeof(float));
    } else {
        const float* W = (const float*)weights->ssm_in_proj->data;

        if (is_cross_decoder_ssm) {
            /* Cross-decoder: project to x only, set z=1 for no gating */
#ifdef USE_ACCELERATE
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, d_inner, d_model,
                        1.0f, input_data, d_model,
                        W, d_model,
                        0.0f, x_conv, d_inner);
#else
            for (int32_t t = 0; t < seq_len; t++) {
                for (int32_t j = 0; j < d_inner; j++) {
                    float sum = 0.0f;
                    for (int32_t k = 0; k < d_model; k++) {
                        sum += input_data[t * d_model + k] * W[j * d_model + k];
                    }
                    x_conv[t * d_inner + j] = sum;
                }
            }
#endif
            /* Set z (gate) to 1.0 - will become SiLU(1.0) = 0.731 as gate */
            for (int32_t i = 0; i < seq_len * d_inner; i++) {
                xz[seq_len * d_inner + i] = 1.0f;  /* Second half of xz is gate */
            }
        } else {
            /* Standard SSM: project to xz (x and gate z) */
            int in_proj_done = 0;
#ifdef DET_USE_METAL
            /* Use persistent GPU buffer if available (Phase 26.16) */
            if (weights->ssm_in_proj->metal_buffer && g_metal_available) {
                if (weights->ssm_in_proj->dtype == DET_DTYPE_Q8_0) {
                    in_proj_done = (tensor_metal_matmul_q8_0_persistent(
                        input_data, weights->ssm_in_proj->metal_buffer, xz,
                        seq_len, 2 * d_inner, d_model) == 0);
                } else if (weights->ssm_in_proj->dtype == DET_DTYPE_F32) {
                    in_proj_done = (tensor_metal_matmul_f32_persistent(
                        input_data, weights->ssm_in_proj->metal_buffer, xz,
                        seq_len, 2 * d_inner, d_model) == 0);
                }
#if DEBUG_SSM_TIMING
                static int debug_inproj_once = 0;
                if (!debug_inproj_once) {
                    fprintf(stderr, "SSM in_proj: GPU path %s (dtype=%d, buf=%p)\n",
                            in_proj_done ? "SUCCESS" : "FAILED",
                            weights->ssm_in_proj->dtype, weights->ssm_in_proj->metal_buffer);
                    debug_inproj_once = 1;
                }
#endif
            }
#endif
            if (!in_proj_done) {
#ifdef USE_ACCELERATE
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            seq_len, 2 * d_inner, d_model,
                            1.0f, input_data, d_model,
                            W, d_model,
                            0.0f, xz, 2 * d_inner);
#else
                for (int32_t t = 0; t < seq_len; t++) {
                    for (int32_t j = 0; j < 2 * d_inner; j++) {
                        float sum = 0.0f;
                        for (int32_t k = 0; k < d_model; k++) {
                            sum += input_data[t * d_model + k] * W[j * d_model + k];
                        }
                        xz[t * 2 * d_inner + j] = sum;
                    }
                }
#endif
            }
            /* Extract x from xz (first d_inner elements per timestep) */
            for (int32_t t = 0; t < seq_len; t++) {
                for (int32_t i = 0; i < d_inner; i++) {
                    x_conv[t * d_inner + i] = xz[t * 2 * d_inner + i];
                }
            }
        }
    }

#if DEBUG_SSM_LAYER0
    if (layer_idx == 0) {
        debug_ssm_rms("in_proj xz", xz, seq_len * 2 * d_inner, 0);
        debug_ssm_rms("in_proj x", x_conv, seq_len * d_inner, 1);
        float* z_start = xz + d_inner;  /* z is at offset d_inner in xz row 0 */
        debug_ssm_rms("in_proj z (row0)", z_start, d_inner, 1);
    }
#endif

#if DEBUG_SSM_TIMING
    t1 = ssm_get_time_ms();
    t_in_proj += t1 - t0;
    t0 = t1;
#endif

    /* 2. Causal convolution */
    float* conv_state = NULL;
    if (cache && cache->conv_state) {
        conv_state = (float*)cache->conv_state->data + layer_idx * d_inner * (d_conv - 1);
    }

    const float* conv_weight = weights->ssm_conv1d_weight ?
        (const float*)weights->ssm_conv1d_weight->data : NULL;
    const float* conv_bias = weights->ssm_conv1d_bias ?
        (const float*)weights->ssm_conv1d_bias->data : NULL;

    if (conv_weight) {
        det_conv1d_causal(x_ssm, x_conv, conv_weight, conv_bias,
                          conv_state, d_inner, d_conv, seq_len);
    } else {
        memcpy(x_ssm, x_conv, seq_len * d_inner * sizeof(float));
    }

#if DEBUG_SSM_TIMING
    t1 = ssm_get_time_ms();
    t_conv1d += t1 - t0;
    t0 = t1;
#endif

    /* 3. SiLU activation */
#if DEBUG_SSM_LAYER0
    if (layer_idx == 0) {
        debug_ssm_rms("conv1d x_conv", x_ssm, seq_len * d_inner, 1);
    }
#endif

#ifdef USE_NEON
    neon_silu_inplace(x_ssm, seq_len * d_inner);
#else
    for (int32_t i = 0; i < seq_len * d_inner; i++) {
        x_ssm[i] = silu(x_ssm[i]);
    }
#endif

#if DEBUG_SSM_LAYER0
    if (layer_idx == 0) {
        debug_ssm_rms("silu x_ssm", x_ssm, seq_len * d_inner, 1);
    }
#endif

#if DEBUG_SSM_TIMING
    t1 = ssm_get_time_ms();
    t_silu += t1 - t0;
    t0 = t1;
#endif

    /* 4. Project x to get delta, B, C */
    if (weights->ssm_x_proj) {
        const float* W = (const float*)weights->ssm_x_proj->data;
        int32_t proj_dim = dt_rank + 2 * d_state;

#ifdef USE_ACCELERATE
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, proj_dim, d_inner,
                    1.0f, x_ssm, d_inner,
                    W, d_inner,
                    0.0f, x_proj, proj_dim);
#else
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t j = 0; j < proj_dim; j++) {
                float sum = 0.0f;
                for (int32_t k = 0; k < d_inner; k++) {
                    sum += x_ssm[t * d_inner + k] * W[j * d_inner + k];
                }
                x_proj[t * proj_dim + j] = sum;
            }
        }
#endif
    }

    /* Extract delta (dt_rank), B (d_state), C (d_state) from x_proj
     * x_proj layout per timestep: [dt_rank | d_state (B) | d_state (C)]
     *
     * For B and C, we use a transposed view to avoid copies.
     * dt_proj needs extraction because its layout differs from what we need.
     */
    float* dt_proj = malloc(seq_len * dt_rank * sizeof(float));
    int32_t proj_dim = dt_rank + 2 * d_state;

    /* Allocate B/C in the layout expected by selective_scan: [seq_len, d_state] */
    float* B_buf = malloc(seq_len * d_state * sizeof(float));
    float* C_buf = malloc(seq_len * d_state * sizeof(float));
    if (!dt_proj || !B_buf || !C_buf) {
        free(dt_proj); free(B_buf); free(C_buf);
        if (!use_workspace) {
            free(xz); free(x_conv); free(x_ssm); free(x_proj);
            free(delta); free(y); free(y_gated);
        }
        return -1;
    }
    B = B_buf;
    C = C_buf;

    for (int32_t t = 0; t < seq_len; t++) {
        const float* proj_t = x_proj + t * proj_dim;
        memcpy(dt_proj + t * dt_rank, proj_t, dt_rank * sizeof(float));
        memcpy(B + t * d_state, proj_t + dt_rank, d_state * sizeof(float));
        memcpy(C + t * d_state, proj_t + dt_rank + d_state, d_state * sizeof(float));
    }

#if DEBUG_SSM_LAYER0
    if (layer_idx == 0) {
        debug_ssm_rms("x_proj dt_raw", dt_proj, seq_len * dt_rank, 0);
        debug_ssm_rms("x_proj B", B, seq_len * d_state, 0);
        debug_ssm_rms("x_proj C", C, seq_len * d_state, 0);
    }
#endif

#if DEBUG_SSM_TIMING
    t1 = ssm_get_time_ms();
    t_x_proj += t1 - t0;
    t0 = t1;
#endif

    /* 5. Project delta_t: delta = softplus(dt_proj @ ssm_dt_proj.T + bias) */
    if (weights->ssm_dt_proj) {
        const float* W = (const float*)weights->ssm_dt_proj->data;
        const float* bias = weights->ssm_dt_proj_bias ?
            (const float*)weights->ssm_dt_proj_bias->data : NULL;

#ifdef USE_ACCELERATE
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, d_inner, dt_rank,
                    1.0f, dt_proj, dt_rank,
                    W, dt_rank,
                    0.0f, delta, d_inner);
#else
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t j = 0; j < d_inner; j++) {
                float sum = 0.0f;
                for (int32_t k = 0; k < dt_rank; k++) {
                    sum += dt_proj[t * dt_rank + k] * W[j * dt_rank + k];
                }
                delta[t * d_inner + j] = sum;
            }
        }
#endif

        /* Add bias and apply softplus */
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t i = 0; i < d_inner; i++) {
                float val = delta[t * d_inner + i];
                if (bias) val += bias[i];
                delta[t * d_inner + i] = softplus(val);
            }
        }
    }
    free(dt_proj);

#if DEBUG_SSM_LAYER0
    if (layer_idx == 0) {
        debug_ssm_rms("dt_proj delta", delta, seq_len * d_inner, 1);
    }
#endif

#if DEBUG_SSM_TIMING
    t1 = ssm_get_time_ms();
    t_dt_proj += t1 - t0;
    t0 = t1;
#endif

    /* 6. SSM selective scan */
    float* h_state = NULL;
    if (cache && cache->h) {
        h_state = (float*)cache->h->data + layer_idx * d_inner * d_state;
    } else {
        h_state = calloc(d_inner * d_state, sizeof(float));
    }

    const float* A = weights->ssm_A_log ? (const float*)weights->ssm_A_log->data : NULL;
    const float* D_skip = weights->ssm_D ? (const float*)weights->ssm_D->data : NULL;

    if (A || weights->ssm_A_negexp) {
        det_ssm_selective_scan(y, x_ssm, delta, A, weights->ssm_A_negexp,
                               B, C, D_skip,
                               h_state, d_inner, d_state, seq_len);
    }

#if DEBUG_SSM_LAYER0
    if (layer_idx == 0) {
        debug_ssm_rms("scan y", y, seq_len * d_inner, 1);
    }
#endif

#if DEBUG_SSM_TIMING
    t1 = ssm_get_time_ms();
    t_selective_scan += t1 - t0;
    t0 = t1;
#endif

    if (!cache) {
        free(h_state);
    }

    /* Compute state norm for stats */
    if (cache && cache->h) {
        float* h = (float*)cache->h->data + layer_idx * d_inner * d_state;
        float norm_sq = 0.0f;
        for (int32_t i = 0; i < d_inner * d_state; i++) {
            norm_sq += h[i] * h[i];
        }
        g_last_ssm_stats.state_norm = sqrtf(norm_sq);
    }

    /* Copy SSM output BEFORE gating if requested (for SambaY caching) */
    if (ssm_output_pre_gate) {
        memcpy(ssm_output_pre_gate, y, seq_len * d_inner * sizeof(float));
    }

    /* 7. Gated output: y_gated = y * SiLU(z) */
    float gate_sum = 0.0f;
    if (is_cross_decoder_ssm) {
        /* Cross-decoder: no gating (z was set to 1.0, SiLU(1.0) = 0.731) */
        float gate = silu(1.0f);
        for (int32_t i = 0; i < seq_len * d_inner; i++) {
            y_gated[i] = y[i] * gate;
            gate_sum += gate;
        }
    } else {
        /* Extract z values and apply NEON-optimized gating */
        float* z_vals = malloc(seq_len * d_inner * sizeof(float));
        if (z_vals) {
            for (int32_t t = 0; t < seq_len; t++) {
                for (int32_t i = 0; i < d_inner; i++) {
                    z_vals[t * d_inner + i] = xz[t * 2 * d_inner + d_inner + i];
                }
            }
#ifdef USE_NEON
            neon_silu_gate(y_gated, y, z_vals, seq_len * d_inner);
#else
            for (int32_t i = 0; i < seq_len * d_inner; i++) {
                float gate = silu(z_vals[i]);
                y_gated[i] = y[i] * gate;
            }
#endif
            /* Compute gate sum for stats */
            for (int32_t i = 0; i < seq_len * d_inner; i++) {
                gate_sum += silu(z_vals[i]);
            }
            free(z_vals);
        } else {
            /* Fallback: original scalar path */
            for (int32_t t = 0; t < seq_len; t++) {
                for (int32_t i = 0; i < d_inner; i++) {
                    float z_val = xz[t * 2 * d_inner + d_inner + i];
                    float gate = silu(z_val);
                    y_gated[t * d_inner + i] = y[t * d_inner + i] * gate;
                    gate_sum += gate;
                }
            }
        }
    }
    g_last_ssm_stats.gate_activation = gate_sum / (seq_len * d_inner);

#if DEBUG_SSM_LAYER0
    if (layer_idx == 0) {
        debug_ssm_rms("gate y_gated", y_gated, seq_len * d_inner, 1);
    }
#endif

#if DEBUG_SSM_TIMING
    t1 = ssm_get_time_ms();
    t_gate += t1 - t0;
    t0 = t1;
#endif

    /* 8. Output projection: output = y_gated @ ssm_out_proj.T */
    if (weights->ssm_out_proj) {
        int out_proj_done = 0;
#ifdef DET_USE_METAL
        /* Use persistent GPU buffer if available (Phase 26.16) */
        if (weights->ssm_out_proj->metal_buffer && g_metal_available) {
            if (weights->ssm_out_proj->dtype == DET_DTYPE_Q8_0) {
                out_proj_done = (tensor_metal_matmul_q8_0_persistent(
                    y_gated, weights->ssm_out_proj->metal_buffer, output_data,
                    seq_len, d_model, d_inner) == 0);
            } else if (weights->ssm_out_proj->dtype == DET_DTYPE_F32) {
                out_proj_done = (tensor_metal_matmul_f32_persistent(
                    y_gated, weights->ssm_out_proj->metal_buffer, output_data,
                    seq_len, d_model, d_inner) == 0);
            }
#if DEBUG_SSM_TIMING
            static int debug_outproj_once = 0;
            if (!debug_outproj_once) {
                fprintf(stderr, "SSM out_proj: GPU path %s (dtype=%d, buf=%p)\n",
                        out_proj_done ? "SUCCESS" : "FAILED",
                        weights->ssm_out_proj->dtype, weights->ssm_out_proj->metal_buffer);
                debug_outproj_once = 1;
            }
#endif
        }
#endif
        if (!out_proj_done) {
            const float* W = (const float*)weights->ssm_out_proj->data;
#ifdef USE_ACCELERATE
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, d_model, d_inner,
                        1.0f, y_gated, d_inner,
                        W, d_inner,
                        0.0f, output_data, d_model);
#else
            for (int32_t t = 0; t < seq_len; t++) {
                for (int32_t j = 0; j < d_model; j++) {
                    float sum = 0.0f;
                    for (int32_t k = 0; k < d_inner; k++) {
                        sum += y_gated[t * d_inner + k] * W[j * d_inner + k];
                    }
                    output_data[t * d_model + j] = sum;
                }
            }
#endif
        }
    }

#if DEBUG_SSM_TIMING
    t1 = ssm_get_time_ms();
    t_out_proj += t1 - t0;
    ssm_timing_count++;

    /* Print summary every 16 calls (once per full layer pass) */
    if (ssm_timing_count % 16 == 0) {
        double total = t_in_proj + t_conv1d + t_silu + t_x_proj + t_dt_proj +
                       t_selective_scan + t_gate + t_out_proj;
        fprintf(stderr, "\nSSM Timing (16 layers, %d calls):\n", ssm_timing_count);
        fprintf(stderr, "  in_proj:  %6.1fms (%.1f%%)\n", t_in_proj, 100*t_in_proj/total);
        fprintf(stderr, "  conv1d:   %6.1fms (%.1f%%)\n", t_conv1d, 100*t_conv1d/total);
        fprintf(stderr, "  silu:     %6.1fms (%.1f%%)\n", t_silu, 100*t_silu/total);
        fprintf(stderr, "  x_proj:   %6.1fms (%.1f%%)\n", t_x_proj, 100*t_x_proj/total);
        fprintf(stderr, "  dt_proj:  %6.1fms (%.1f%%)\n", t_dt_proj, 100*t_dt_proj/total);
        fprintf(stderr, "  scan:     %6.1fms (%.1f%%)\n", t_selective_scan, 100*t_selective_scan/total);
        fprintf(stderr, "  gate:     %6.1fms (%.1f%%)\n", t_gate, 100*t_gate/total);
        fprintf(stderr, "  out_proj: %6.1fms (%.1f%%)\n", t_out_proj, 100*t_out_proj/total);
        fprintf(stderr, "  TOTAL:    %6.1fms\n", total);
        /* Reset counters */
        t_in_proj = t_conv1d = t_silu = t_x_proj = t_dt_proj = 0;
        t_selective_scan = t_gate = t_out_proj = 0;
    }
#endif

    /* Compute skip ratio for stats */
    if (D_skip) {
        float skip_sum = 0.0f, total_sum = 0.0f;
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t i = 0; i < d_inner; i++) {
                float x_i = x_ssm[t * d_inner + i];
                skip_sum += fabsf(D_skip[i] * x_i);
                total_sum += fabsf(y[t * d_inner + i]);
            }
        }
        g_last_ssm_stats.skip_ratio = (total_sum > 0) ? skip_sum / total_sum : 0.0f;
    }

    /* Cleanup */
    free(B);  /* B_buf */
    free(C);  /* C_buf */
    if (!use_workspace) {
        /* Only free if we allocated per-call */
        free(xz);
        free(x_conv);
        free(x_ssm);
        free(x_proj);
        free(delta);
        free(y);
        free(y_gated);
    }

    return 0;
}

/* ==========================================================================
 * DET PHYSICS INTEGRATION
 * ========================================================================== */

void det_ssm_get_stats(DetSSMLayerStats* stats) {
    if (stats) {
        *stats = g_last_ssm_stats;
    }
}

/* ==========================================================================
 * UTILITY FUNCTIONS
 * ========================================================================== */

bool det_is_ssm_layer(void* gguf_ptr, int32_t layer_idx) {
    GgufContext* gguf = (GgufContext*)gguf_ptr;
    if (!gguf) return false;

    /* Check for SSM-specific tensors */
    char tensor_name[256];

    /* Check for ssm_in.weight - unique to SSM layers */
    snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ssm_in.weight", layer_idx);
    if (gguf_get_tensor_info(gguf, tensor_name) != NULL) {
        return true;
    }

    /* Also check alternative naming */
    snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ssm_conv1d.weight", layer_idx);
    if (gguf_get_tensor_info(gguf, tensor_name) != NULL) {
        return true;
    }

    return false;
}

DetSSMConfig det_ssm_config_from_gguf(void* gguf_ptr) {
    GgufContext* gguf = (GgufContext*)gguf_ptr;
    DetSSMConfig config = {0};

    if (!gguf) return config;

    /* Try mamba-specific metadata first */
    config.d_model = gguf_get_u32(gguf, "mamba.embedding_length", 0);
    if (config.d_model == 0) {
        config.d_model = gguf->n_embd;
    }

    config.d_inner = gguf_get_u32(gguf, "mamba.d_inner", 0);
    if (config.d_inner == 0) {
        config.d_inner = config.d_model * 2;  /* Default expansion factor */
    }

    config.d_state = gguf_get_u32(gguf, "mamba.d_state", 16);
    config.d_conv = gguf_get_u32(gguf, "mamba.d_conv", 4);

    config.dt_rank = gguf_get_u32(gguf, "mamba.dt_rank", 0);
    if (config.dt_rank == 0) {
        config.dt_rank = (config.d_model + 15) / 16;  /* ceil(d_model / 16) */
    }

    config.dt_min = gguf_get_f32(gguf, "mamba.dt_min", 0.001f);
    config.dt_max = gguf_get_f32(gguf, "mamba.dt_max", 0.1f);
    config.dt_init = gguf_get_f32(gguf, "mamba.dt_init", 0.001f);
    config.dt_scale = gguf_get_f32(gguf, "mamba.dt_scale", 1.0f);

    return config;
}

/* ==========================================================================
 * GMU (GATED MEMORY UNIT) IMPLEMENTATION
 * ========================================================================== */

int det_gmu_forward(float* output,
                    const float* input,
                    const float* cached_ssm,
                    const float* w_gate,
                    const float* w_up,
                    const float* w_down,
                    int32_t d_model,
                    int32_t d_inner,
                    int32_t seq_len) {
    if (!output || !input || !cached_ssm || !w_gate || !w_down) return -1;

    /* Allocate intermediate buffers */
    float* gate = malloc(seq_len * d_inner * sizeof(float));
    float* up = malloc(seq_len * d_inner * sizeof(float));
    float* hidden = malloc(seq_len * d_inner * sizeof(float));

    if (!gate || !up || !hidden) {
        free(gate); free(up); free(hidden);
        return -1;
    }

    /* 1. Gate projection: gate = input @ w_gate.T */
#ifdef USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, d_inner, d_model,
                1.0f, input, d_model,
                w_gate, d_model,
                0.0f, gate, d_inner);
#else
    for (int32_t t = 0; t < seq_len; t++) {
        for (int32_t j = 0; j < d_inner; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < d_model; k++) {
                sum += input[t * d_model + k] * w_gate[j * d_model + k];
            }
            gate[t * d_inner + j] = sum;
        }
    }
#endif

    /* 2. Up projection on cached SSM (if w_up provided) or use directly */
    if (w_up) {
#ifdef USE_ACCELERATE
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, d_inner, d_inner,
                    1.0f, cached_ssm, d_inner,
                    w_up, d_inner,
                    0.0f, up, d_inner);
#else
        for (int32_t t = 0; t < seq_len; t++) {
            for (int32_t j = 0; j < d_inner; j++) {
                float sum = 0.0f;
                for (int32_t k = 0; k < d_inner; k++) {
                    sum += cached_ssm[t * d_inner + k] * w_up[j * d_inner + k];
                }
                up[t * d_inner + j] = sum;
            }
        }
#endif
    } else {
        memcpy(up, cached_ssm, seq_len * d_inner * sizeof(float));
    }

    /* 3. SwiGLU: hidden = SiLU(gate) * up */
#ifdef USE_NEON
    neon_silu_gate(hidden, up, gate, seq_len * d_inner);
#else
    for (int32_t i = 0; i < seq_len * d_inner; i++) {
        float g = gate[i];
        hidden[i] = (g / (1.0f + expf(-g))) * up[i];  /* SiLU(gate) * up */
    }
#endif

    /* 4. Down projection: output = hidden @ w_down.T */
#ifdef USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, d_model, d_inner,
                1.0f, hidden, d_inner,
                w_down, d_inner,
                0.0f, output, d_model);
#else
    for (int32_t t = 0; t < seq_len; t++) {
        for (int32_t j = 0; j < d_model; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < d_inner; k++) {
                sum += hidden[t * d_inner + k] * w_down[j * d_inner + k];
            }
            output[t * d_model + j] = sum;
        }
    }
#endif

    free(gate);
    free(up);
    free(hidden);

    return 0;
}

int det_gmu_swiglu(float* output,
                   const float* input,
                   const float* cached_ssm,
                   const float* in_proj,
                   const float* out_proj,
                   int32_t d_model,
                   int32_t d_inner,
                   int32_t seq_len) {
    if (!output || !input || !cached_ssm || !in_proj || !out_proj) return -1;

    /* Allocate intermediate buffer */
    float* proj = malloc(seq_len * d_inner * sizeof(float));
    float* hidden = malloc(seq_len * d_inner * sizeof(float));

    if (!proj || !hidden) {
        free(proj); free(hidden);
        return -1;
    }

    /* 1. Input projection: proj = input @ in_proj.T */
#ifdef USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, d_inner, d_model,
                1.0f, input, d_model,
                in_proj, d_model,
                0.0f, proj, d_inner);
#else
    for (int32_t t = 0; t < seq_len; t++) {
        for (int32_t j = 0; j < d_inner; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < d_model; k++) {
                sum += input[t * d_model + k] * in_proj[j * d_model + k];
            }
            proj[t * d_inner + j] = sum;
        }
    }
#endif

    /* 2. SwiGLU with cached SSM: hidden = SiLU(proj) * cached_ssm */
#ifdef USE_NEON
    neon_silu_gate(hidden, cached_ssm, proj, seq_len * d_inner);
#else
    for (int32_t i = 0; i < seq_len * d_inner; i++) {
        float g = proj[i];
        hidden[i] = (g / (1.0f + expf(-g))) * cached_ssm[i];
    }
#endif

    /* 3. Output projection: output = hidden @ out_proj.T */
#ifdef USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, d_model, d_inner,
                1.0f, hidden, d_inner,
                out_proj, d_inner,
                0.0f, output, d_model);
#else
    for (int32_t t = 0; t < seq_len; t++) {
        for (int32_t j = 0; j < d_model; j++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < d_inner; k++) {
                sum += hidden[t * d_inner + k] * out_proj[j * d_inner + k];
            }
            output[t * d_model + j] = sum;
        }
    }
#endif

    free(proj);
    free(hidden);

    return 0;
}

/* ==========================================================================
 * SLIDING WINDOW ATTENTION
 * ========================================================================== */

int det_sliding_window_mask(float* scores,
                            int32_t seq_q,
                            int32_t seq_k,
                            int32_t window_size,
                            int32_t pos_offset) {
    if (!scores || window_size <= 0) return -1;

    /*
     * Apply sliding window + causal mask.
     *
     * Query positions: [pos_offset, pos_offset + seq_q - 1]
     * Key positions: We assume keys are the full prefix [0, seq_k - 1]
     *   (In streaming with KV cache, seq_k = total cached + current)
     *
     * For query at absolute position q_pos, can attend to key at position k_pos where:
     *   - Causal: k_pos <= q_pos
     *   - Sliding window: q_pos - k_pos < window_size
     */
    for (int32_t i = 0; i < seq_q; i++) {
        int32_t q_pos = pos_offset + i;  /* Absolute query position */

        for (int32_t j = 0; j < seq_k; j++) {
            int32_t k_pos = j;  /* Key position (keys are full prefix) */

            /* Causal: can't attend to future (k_pos > q_pos) */
            if (k_pos > q_pos) {
                scores[i * seq_k + j] = -1e9f;
                continue;
            }

            /* Sliding window: can't attend beyond window_size tokens back */
            if (q_pos - k_pos >= window_size) {
                scores[i * seq_k + j] = -1e9f;
            }
        }
    }

    return 0;
}
