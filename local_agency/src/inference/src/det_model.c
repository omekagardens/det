/**
 * DET Model Inference - Implementation
 * =====================================
 *
 * LLM forward pass through transformer layers.
 * Supports Metal GPU acceleration when available.
 * Uses Apple Accelerate BLAS for optimized CPU operations.
 */

#include "det_model.h"
#include "det_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE 1
#endif

#ifdef DET_USE_METAL
#include "det_tensor_metal.h"
#endif

/* Metal acceleration state */
static int g_metal_initialized = 0;
static int g_metal_available = 0;

/* Inference mode configuration */
static DetInferenceMode g_inference_mode = DET_INFERENCE_F32;

/* Minimum matrix size to use Metal (avoid overhead for small matrices)
 * For thin matrices (small T), GPU data transfer overhead dominates.
 * CPU BLAS (Accelerate) is faster for autoregressive generation. */
#define METAL_MIN_ELEMENTS (2048 * 2048)  /* ~4M elements minimum for Metal */

/* ==========================================================================
 * TIMING DEBUG
 * ========================================================================== */

#include <sys/time.h>
static int g_debug_timing = 0;  /* Set to 1 for timing output */

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void det_enable_timing(int enable) {
    g_debug_timing = enable;
}

/* ==========================================================================
 * INFERENCE MODE CONFIGURATION
 * ========================================================================== */

void det_set_inference_mode(DetInferenceMode mode) {
    g_inference_mode = mode;
}

DetInferenceMode det_get_inference_mode(void) {
    return g_inference_mode;
}

/* ==========================================================================
 * INTERNAL HELPERS
 * ========================================================================== */

/* xorshift64 PRNG for reproducible sampling */
static uint64_t g_rng_state = 0x853c49e6748fea9bULL;

static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float random_float(uint64_t* state) {
    return (float)(xorshift64(state) >> 11) / (float)(1ULL << 53);
}

/* ==========================================================================
 * METAL GPU HELPERS
 * ========================================================================== */

/* Initialize Metal backend (called once) */
static void init_metal_if_available(void) {
    if (g_metal_initialized) return;
    g_metal_initialized = 1;

#ifdef DET_USE_METAL
    if (tensor_metal_init() == 0) {
        g_metal_available = 1;
        printf("Metal GPU: %s\n", tensor_metal_device_name());
    } else {
        g_metal_available = 0;
        printf("Metal GPU: not available, using CPU\n");
    }
#else
    g_metal_available = 0;
#endif
}

/* Check if Metal should be used for given matrix size */
static inline int should_use_metal(int M, int N, int K) {
    if (!g_metal_available) return 0;
    /* Only use Metal for large enough matrices to amortize copy overhead */
    return (M * N >= METAL_MIN_ELEMENTS) || (M * K >= METAL_MIN_ELEMENTS);
}

/**
 * Matrix multiplication with Metal acceleration
 * C[M,N] = A[M,K] @ B[K,N]
 *
 * Falls back to CPU for small matrices or if Metal unavailable.
 */
static void matmul_f32(float* C, const float* A, const float* B,
                       int M, int N, int K) {
#ifdef DET_USE_METAL
    if (should_use_metal(M, N, K)) {
        if (tensor_metal_matmul(A, B, C, M, N, K) == 0) {
            return;  /* Metal succeeded */
        }
        /* Fall through to CPU on Metal failure */
    }
#endif

    /* CPU fallback: straightforward triple loop */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Matrix-vector multiplication: y[M] = A[M,K] @ x[K]
 * (A is stored as [M,K], we compute y = A @ x)
 */
static void matvec_f32(float* y, const float* A, const float* x,
                       int M, int K) {
    /* For now, use CPU. Metal matvec has too much overhead for small vectors. */
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < K; j++) {
            sum += A[i * K + j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * Batched output projection: out[T,N] = hidden[T,K] @ W[N,K]^T
 * This is for large vocab projections where Metal helps significantly.
 *
 * W is stored as [N,K] (row-major), we need hidden @ W^T
 * Which is: out[t,n] = sum_k(hidden[t,k] * W[n,k])
 *
 * Uses Metal GPU acceleration when available, falls back to Accelerate BLAS.
 */
static void batched_proj_f32(float* out, const float* hidden, const float* W,
                             int T, int K, int N) {
#ifdef DET_USE_METAL
    /* Use Metal for large matrices where GPU overhead is amortized */
    if (g_metal_available && (T * N >= METAL_MIN_ELEMENTS)) {
        double t0 = g_debug_timing ? get_time_ms() : 0;
        if (tensor_metal_matmul_transposed_b(hidden, W, out, T, N, K) == 0) {
            if (g_debug_timing && N > 100000) {
                printf("    Metal matmul [%d,%d,%d]: %.1fms\n", T, K, N, get_time_ms() - t0);
            }
            return;  /* Metal succeeded */
        }
        /* Fall through to CPU on failure */
    }
#endif

#ifdef USE_ACCELERATE
    /* Use Accelerate BLAS: C = A @ B^T
     * A is hidden[T,K], B is W[N,K], Result is out[T,N] */
    double t0 = g_debug_timing ? get_time_ms() : 0;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                T, N, K,
                1.0f,           /* alpha */
                hidden, K,      /* A[T,K], lda=K */
                W, K,           /* B[N,K], ldb=K (transposed) */
                0.0f,           /* beta */
                out, N);        /* C[T,N], ldc=N */
    if (g_debug_timing && N > 100000) {
        printf("    BLAS matmul [%d,%d,%d]: %.1fms\n", T, K, N, get_time_ms() - t0);
    }
#else
    /* CPU fallback: compute row by row */
    for (int t = 0; t < T; t++) {
        const float* h = hidden + t * K;
        float* o = out + t * N;
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            const float* w_row = W + n * K;
            for (int k = 0; k < K; k++) {
                sum += h[k] * w_row[k];
            }
            o[n] = sum;
        }
    }
#endif
}

/**
 * Smart projection dispatch: handles both F32 and Q8_0 weights.
 *
 * out[T,N] = hidden[T,K] @ W[N,K]^T
 *
 * If W is Q8_0, uses quantization-aware matmul.
 * If W is F32, uses standard batched_proj_f32.
 */
static void batched_proj_smart(float* out, const float* hidden,
                                const DetTensor* W, int T, int K, int N) {
    if (!W) return;

    if (W->dtype == DET_DTYPE_Q8_0) {
        /* Q8_0 quantization-aware matmul */
        const uint8_t* W_q8 = (const uint8_t*)W->data;

#ifdef DET_USE_METAL
        /* Try Metal Q8_0 matmul for large matrices */
        if (g_metal_available && (T * N >= METAL_MIN_ELEMENTS)) {
            if (tensor_metal_matmul_q8_0_transposed(hidden, W_q8, out, T, N, K) == 0) {
                return;  /* Metal succeeded */
            }
            /* Fall through to CPU on failure */
        }
#endif

        /* CPU Q8_0 matmul with batched dequantization */
        det_matmul_q8_0_transposed_batched(out, hidden, W_q8, T, N, K);
    } else {
        /* F32 weights - use standard projection */
        batched_proj_f32(out, hidden, (const float*)W->data, T, K, N);
    }
}

/* ==========================================================================
 * MODEL LOADING
 * ========================================================================== */

/* Get layer weight by name pattern (dequantized to F32) */
static DetTensor* get_layer_weight(GgufContext* gguf, int layer, const char* name) {
    char full_name[256];
    snprintf(full_name, sizeof(full_name), "blk.%d.%s.weight", layer, name);
    return gguf_get_tensor_f32(gguf, full_name);
}

/* Get layer bias by name pattern */
static DetTensor* get_layer_bias(GgufContext* gguf, int layer, const char* name) {
    char full_name[256];
    snprintf(full_name, sizeof(full_name), "blk.%d.%s.bias", layer, name);
    return gguf_get_tensor_f32(gguf, full_name);
}

/* Get layer parameter by name (no .weight/.bias suffix) */
static DetTensor* get_layer_param(GgufContext* gguf, int layer, const char* name) {
    char full_name[256];
    snprintf(full_name, sizeof(full_name), "blk.%d.%s", layer, name);
    return gguf_get_tensor_f32(gguf, full_name);
}

/**
 * Smart weight loading based on inference mode.
 *
 * In Q8_0 mode: Returns Q8_0 tensor if source is Q8_0 (no dequantization)
 * In F32 mode:  Always returns dequantized F32 tensor
 */
static DetTensor* get_layer_weight_smart(GgufContext* gguf, int layer, const char* name) {
    char full_name[256];
    snprintf(full_name, sizeof(full_name), "blk.%d.%s.weight", layer, name);

    /* Check if we should use Q8_0 mode */
    if (g_inference_mode == DET_INFERENCE_Q8_0) {
        /* Try to get Q8_0 tensor first */
        const GgufTensorInfo* info = gguf_get_tensor_info(gguf, full_name);
        if (info && info->type == GGUF_TENSOR_Q8_0) {
            DetTensor* t = gguf_get_tensor_q8_0(gguf, full_name);
            if (t) return t;
        }
    }

    /* Fall back to F32 */
    return gguf_get_tensor_f32(gguf, full_name);
}

/* Get weight by full name with smart mode */
static DetTensor* get_weight_smart(GgufContext* gguf, const char* full_name) {
    if (g_inference_mode == DET_INFERENCE_Q8_0) {
        const GgufTensorInfo* info = gguf_get_tensor_info(gguf, full_name);
        if (info && info->type == GGUF_TENSOR_Q8_0) {
            DetTensor* t = gguf_get_tensor_q8_0(gguf, full_name);
            if (t) return t;
        }
    }
    return gguf_get_tensor_f32(gguf, full_name);
}

/**
 * Split fused QKV tensor into separate Q, K, V tensors.
 * Fused layout: [n_embd, q_dim + kv_dim + kv_dim] where:
 *   q_dim = n_head * head_dim = n_embd
 *   kv_dim = n_head_kv * head_dim
 *
 * Returns newly allocated tensors via out_q, out_k, out_v.
 */
static int split_fused_qkv(DetTensor* fused, int n_embd, int n_head, int n_head_kv,
                           DetTensor** out_q, DetTensor** out_k, DetTensor** out_v) {
    if (!fused || fused->ndim != 2) return -1;

    int head_dim = n_embd / n_head;
    int q_dim = n_embd;                    /* n_head * head_dim */
    int kv_dim = n_head_kv * head_dim;

    /* Verify fused tensor dimensions */
    int fused_total = (int)fused->shape[1];
    int expected = q_dim + kv_dim + kv_dim;
    if (fused_total != expected) {
        fprintf(stderr, "Fused QKV size mismatch: got %d, expected %d\n", fused_total, expected);
        return -1;
    }

    /* Create output tensors */
    int32_t q_shape[2] = {(int32_t)fused->shape[0], q_dim};
    int32_t kv_shape[2] = {(int32_t)fused->shape[0], kv_dim};

    *out_q = det_tensor_create(2, q_shape, DET_DTYPE_F32);
    *out_k = det_tensor_create(2, kv_shape, DET_DTYPE_F32);
    *out_v = det_tensor_create(2, kv_shape, DET_DTYPE_F32);

    if (!*out_q || !*out_k || !*out_v) {
        det_tensor_release(*out_q);
        det_tensor_release(*out_k);
        det_tensor_release(*out_v);
        return -1;
    }

    /* Copy data: fused is [n_embd, q+k+v], each row has Q then K then V */
    float* fused_data = (float*)fused->data;
    float* q_data = (float*)(*out_q)->data;
    float* k_data = (float*)(*out_k)->data;
    float* v_data = (float*)(*out_v)->data;

    int rows = (int)fused->shape[0];
    for (int row = 0; row < rows; row++) {
        float* src = fused_data + row * fused_total;
        memcpy(q_data + row * q_dim, src, q_dim * sizeof(float));
        memcpy(k_data + row * kv_dim, src + q_dim, kv_dim * sizeof(float));
        memcpy(v_data + row * kv_dim, src + q_dim + kv_dim, kv_dim * sizeof(float));
    }

    return 0;
}

/**
 * Load QKV weights - handles both separate and fused layouts.
 * Phi3/Phi4 use fused attn_qkv, other models use separate attn_q/k/v.
 *
 * For SambaY cross-decoder attention, Wqkv only contains Q (K/V from cached SSM).
 */
static void load_qkv_weights(GgufContext* gguf, int layer, DetModel* model, DetLayerWeights* lw) {
    /* Try separate weights first (most common) */
    lw->wq = get_layer_weight_smart(gguf, layer, "attn_q");
    lw->wk = get_layer_weight_smart(gguf, layer, "attn_k");
    lw->wv = get_layer_weight_smart(gguf, layer, "attn_v");

    /* If we got separate weights, we're done */
    if (lw->wq && lw->wk && lw->wv) return;

    /* Try fused QKV (Phi3/Phi4 style) - force F32 for splitting */
    char full_name[256];
    snprintf(full_name, sizeof(full_name), "blk.%d.attn_qkv.weight", layer);
    DetTensor* fused = gguf_get_tensor_f32(gguf, full_name);

    if (fused) {
        int n_embd = model->config.n_embd;
        int head_dim = n_embd / model->config.n_head;
        int expected_qkv = n_embd + 2 * (model->config.n_head_kv * head_dim);

        /* Check if this is Q-only (cross-decoder) or full QKV (self-decoder) */
        if (fused->shape[1] == (uint64_t)n_embd) {
            /* Q-only: cross-decoder attention
             * K/V will come from cached SSM output at runtime
             * Store as wq and leave wk/wv NULL */
            lw->wq = fused;
            lw->wk = NULL;
            lw->wv = NULL;
            lw->is_cross_decoder = true;  /* Mark for runtime handling */
            return;
        } else if (fused->shape[1] == (uint64_t)expected_qkv) {
            /* Full QKV: self-decoder attention */
            if (split_fused_qkv(fused, n_embd, model->config.n_head,
                               model->config.n_head_kv, &lw->wq, &lw->wk, &lw->wv) == 0) {
                det_tensor_release(fused);
                return;
            }
        } else {
            fprintf(stderr, "Unexpected QKV size for layer %d: got %d, expected %d or %d\n",
                    layer, (int)fused->shape[1], n_embd, expected_qkv);
        }
        det_tensor_release(fused);
    }
}

DetModel* det_model_load(const char* path) {
    /* Initialize Metal GPU if available */
    init_metal_if_available();

    /* Open GGUF file */
    GgufContext* gguf = gguf_open(path);
    if (!gguf) {
        fprintf(stderr, "Failed to open model: %s\n", path);
        return NULL;
    }

    DetModel* model = calloc(1, sizeof(DetModel));
    if (!model) {
        gguf_close(gguf);
        return NULL;
    }

    model->gguf = gguf;

    /* Extract configuration from GGUF metadata */
    model->config.arch = gguf_detect_arch(gguf);
    model->config.n_vocab = gguf->n_vocab;
    model->config.n_ctx = gguf->n_ctx;
    model->config.n_embd = gguf->n_embd;
    model->config.n_head = gguf->n_head;
    model->config.n_head_kv = gguf->n_head_kv;
    model->config.n_layer = gguf->n_layer;
    model->config.n_ff = gguf->n_ff;
    model->config.rope_freq_base = gguf->rope_freq_base;
    model->config.rope_freq_scale = gguf->rope_freq_scale;

    /* Calculate derived values */
    int head_dim = model->config.n_embd / model->config.n_head;
    model->config.n_rot = head_dim;

    /* Get normalization epsilon based on architecture */
    model->config.norm_eps = gguf_get_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 0.0f);
    if (model->config.norm_eps == 0.0f) {
        model->config.norm_eps = gguf_get_f32(gguf, "qwen2.attention.layer_norm_rms_epsilon", 0.0f);
    }
    if (model->config.norm_eps == 0.0f) {
        model->config.norm_eps = gguf_get_f32(gguf, "qwen3.attention.layer_norm_rms_epsilon", 0.0f);
    }
    if (model->config.norm_eps == 0.0f) {
        model->config.norm_eps = gguf_get_f32(gguf, "phi3.attention.layer_norm_rms_epsilon", 0.0f);
    }
    if (model->config.norm_eps == 0.0f) {
        model->config.norm_eps = 1e-6f;  /* Default fallback */
    }

    /* Validate config */
    if (model->config.n_vocab == 0 || model->config.n_embd == 0 ||
        model->config.n_layer == 0) {
        fprintf(stderr, "Invalid model configuration\n");
        det_model_free(model);
        return NULL;
    }

    /* Load embedding weights (dequantized to F32 for computation) */
    model->weights.tok_embd = gguf_get_tensor_f32(gguf, "token_embd.weight");
    if (!model->weights.tok_embd) {
        /* Try alternative name */
        model->weights.tok_embd = gguf_get_tensor_f32(gguf, "model.embed_tokens.weight");
    }

    /* Load output weights (dequantized) */
    model->weights.output_norm = gguf_get_tensor_f32(gguf, "output_norm.weight");
    if (!model->weights.output_norm) {
        model->weights.output_norm = gguf_get_tensor_f32(gguf, "model.norm.weight");
    }

    /* Output projection - can be Q8_0 in QAM mode (this is often the largest weight) */
    model->weights.output = get_weight_smart(gguf, "output.weight");
    if (!model->weights.output) {
        /* May share weights with embedding */
        model->weights.output = model->weights.tok_embd;
    }

    /* Allocate layer weights */
    model->weights.n_layers = model->config.n_layer;
    model->weights.layers = calloc(model->config.n_layer, sizeof(DetLayerWeights));
    if (!model->weights.layers) {
        det_model_free(model);
        return NULL;
    }

    /* Load layer weights
     * Use smart loading for projection weights (can be Q8_0 in QAM mode)
     * Always F32 for norms (small tensors, not worth quantizing) */
    model->n_attn_layers = 0;
    model->n_ssm_layers = 0;

    for (int i = 0; i < model->config.n_layer; i++) {
        DetLayerWeights* layer = &model->weights.layers[i];

        /* Check if this is an SSM layer (Mamba) or attention layer */
        layer->is_ssm_layer = det_is_ssm_layer(gguf, i);

        if (layer->is_ssm_layer) {
            /* Load SSM weights (Mamba layer) */
            model->n_ssm_layers++;

            /* SSM norm (pre-layer norm) */
            layer->ssm_norm = get_layer_weight(gguf, i, "ssm_norm");
            if (!layer->ssm_norm) {
                layer->ssm_norm = get_layer_weight(gguf, i, "attn_norm");
            }

            /* SSM projections */
            layer->ssm_in_proj = get_layer_weight_smart(gguf, i, "ssm_in");
            layer->ssm_conv1d_weight = get_layer_weight(gguf, i, "ssm_conv1d");
            layer->ssm_conv1d_bias = get_layer_bias(gguf, i, "ssm_conv1d");
            layer->ssm_x_proj = get_layer_weight_smart(gguf, i, "ssm_x");
            layer->ssm_dt_proj = get_layer_weight_smart(gguf, i, "ssm_dt");
            layer->ssm_dt_proj_bias = get_layer_bias(gguf, i, "ssm_dt");
            layer->ssm_A_log = get_layer_weight(gguf, i, "ssm_A_log");
            layer->ssm_D = get_layer_weight(gguf, i, "ssm_D");
            layer->ssm_out_proj = get_layer_weight_smart(gguf, i, "ssm_out");

            /* Precompute A_negexp = -exp(A_log) for major speedup in selective scan */
            layer->ssm_A_negexp = NULL;
            if (layer->ssm_A_log) {
                int64_t A_size = layer->ssm_A_log->shape[0] * layer->ssm_A_log->shape[1];
                layer->ssm_A_negexp = malloc(A_size * sizeof(float));
                if (layer->ssm_A_negexp) {
                    const float* A_log = (const float*)layer->ssm_A_log->data;
                    for (int64_t j = 0; j < A_size; j++) {
                        layer->ssm_A_negexp[j] = -expf(A_log[j]);
                    }
                }
            }

            /* FFN projections (SSM layers may also have FFN) */
            layer->ffn_norm = get_layer_weight(gguf, i, "ffn_norm");
            layer->w1 = get_layer_weight_smart(gguf, i, "ffn_gate");
            layer->w2 = get_layer_weight_smart(gguf, i, "ffn_down");
            layer->w3 = get_layer_weight_smart(gguf, i, "ffn_up");
        } else {
            /* Load attention weights (transformer layer) */
            model->n_attn_layers++;

            /* Normalization weights - always F32 */
            layer->attn_norm = get_layer_weight(gguf, i, "attn_norm");
            layer->ffn_norm = get_layer_weight(gguf, i, "ffn_norm");

            /* Attention projections - handles both separate and fused QKV */
            load_qkv_weights(gguf, i, model, layer);
            layer->wo = get_layer_weight_smart(gguf, i, "attn_output");

            /* QKV biases (Qwen2 uses these) - always F32 */
            layer->bq = get_layer_bias(gguf, i, "attn_q");
            layer->bk = get_layer_bias(gguf, i, "attn_k");
            layer->bv = get_layer_bias(gguf, i, "attn_v");

            /* QK-Norm weights (Qwen3 uses these) - always F32 */
            layer->q_norm = get_layer_weight(gguf, i, "attn_q_norm");
            layer->k_norm = get_layer_weight(gguf, i, "attn_k_norm");

            /* Differential attention weights (phi4flash) - always F32
             * Lambda params don't have .weight suffix in GGUF
             * SubLN has .weight suffix */
            layer->diff_lambda_q1 = get_layer_param(gguf, i, "diff_lambda_q1");
            layer->diff_lambda_k1 = get_layer_param(gguf, i, "diff_lambda_k1");
            layer->diff_lambda_q2 = get_layer_param(gguf, i, "diff_lambda_q2");
            layer->diff_lambda_k2 = get_layer_param(gguf, i, "diff_lambda_k2");
            layer->diff_subln = get_layer_weight(gguf, i, "diff_subln");
            layer->use_diff_attn = (layer->diff_lambda_q1 != NULL);

            /* FFN projections - can be Q8_0 in QAM mode */
            layer->w1 = get_layer_weight_smart(gguf, i, "ffn_gate");
            layer->w2 = get_layer_weight_smart(gguf, i, "ffn_down");
            layer->w3 = get_layer_weight_smart(gguf, i, "ffn_up");
        }
    }

    /* Set up SSM-specific structures if needed */
    model->has_ssm_layers = (model->n_ssm_layers > 0);
    if (model->has_ssm_layers) {
        /* Get SSM config from GGUF */
        model->ssm_config = det_ssm_config_from_gguf(gguf);

        /* Create SSM cache with workspace for typical generation lengths */
        int32_t max_workspace_seq = 2048;  /* Prompt + some generation */
        model->ssm_cache = det_ssm_cache_create(
            model->n_ssm_layers,
            model->ssm_config.d_inner,
            model->ssm_config.d_state,
            model->ssm_config.d_conv,
            model->ssm_config.dt_rank,
            max_workspace_seq
        );
    }

    /* Detect SambaY architecture (phi4flash, etc.) */
    if (model->config.arch == DET_ARCH_PHI4FLASH ||
        model->config.arch == DET_ARCH_SAMBAY) {
        model->config.is_sambay = true;

        /* Get sliding window size from GGUF metadata */
        model->config.sliding_window = gguf_get_u32(gguf, "phi4.sliding_window", 512);
        if (model->config.sliding_window == 512) {
            model->config.sliding_window = gguf_get_u32(gguf, "attention.sliding_window", 512);
        }

        /* Get mb_per_layer (Mamba blocks per layer, 2 = every other layer uses Mamba) */
        model->config.mb_per_layer = gguf_get_u32(gguf, "phi4.mb_per_layer", 2);

        /* Configure layer types for SambaY pattern:
         * The model has alternating Mamba/Attention throughout all 32 layers.
         * Self-decoder (layers 0-15): standard Mamba/Attention
         * Cross-decoder (layers 16-31): still alternating, but attention uses Q-only
         *   with K/V derived from cached SSM output (GMU pattern)
         *
         * Layer type is already detected from tensors during loading:
         * - is_ssm_layer: set by det_is_ssm_layer() based on presence of ssm_in.weight
         * - is_cross_decoder: set by load_qkv_weights() based on QKV tensor shape
         *
         * Here we configure sliding window and mark second-half layers for cross-decoder.
         */
        int half_layers = model->config.n_layer / 2;
        for (int i = 0; i < model->config.n_layer; i++) {
            DetLayerWeights* layer = &model->weights.layers[i];

            /* Keep is_ssm_layer as detected from tensor presence */
            /* Keep is_cross_decoder as detected from QKV shape */

            if (i < half_layers) {
                /* Self-decoder: use sliding window for attention layers */
                if (!layer->is_ssm_layer) {
                    layer->use_sliding_window = true;
                    layer->sliding_window_size = model->config.sliding_window;
                }
                layer->use_gmu = false;
            } else {
                /* Cross-decoder: attention layers use GMU pattern (Q-only + cached SSM) */
                if (!layer->is_ssm_layer && layer->is_cross_decoder) {
                    layer->use_gmu = true;
                    layer->use_sliding_window = false;
                }
            }
        }

        printf("  SambaY architecture: %d self-decoder layers, %d cross-decoder layers\n",
               half_layers, model->config.n_layer - half_layers);
        printf("  Sliding window: %d, Mamba every %d layers\n",
               model->config.sliding_window, model->config.mb_per_layer);
    }

    /* Create tokenizer */
    model->tokenizer = det_tokenizer_from_gguf(gguf);

    /* Allocate KV cache */
    int32_t kv_shape[4] = {
        model->config.n_layer,
        model->config.n_ctx,
        model->config.n_head_kv,
        head_dim
    };
    model->kv_cache.k = det_tensor_create(4, kv_shape, DET_DTYPE_F32);
    model->kv_cache.v = det_tensor_create(4, kv_shape, DET_DTYPE_F32);
    model->kv_cache.capacity = model->config.n_ctx;
    model->kv_cache.seq_len = 0;

    /* Create workspace for intermediate tensors */
    size_t workspace_sizes[8] = {
        model->config.n_embd * sizeof(float),                 /* hidden state */
        model->config.n_embd * sizeof(float),                 /* residual */
        model->config.n_embd * sizeof(float) * 4,             /* attention scratch */
        model->config.n_ff * sizeof(float),                   /* FFN scratch */
        model->config.n_vocab * sizeof(float),                /* logits */
        model->config.n_ctx * model->config.n_ctx * sizeof(float),  /* attention scores */
        0, 0
    };
    model->workspace = det_workspace_create(workspace_sizes, 6);

    /* Pre-allocate scratch buffers for forward pass (avoid malloc per call) */
    int n_ctx = model->config.n_ctx;
    int kv_dim = model->config.n_head_kv * head_dim;
    model->scratch.hidden = malloc(n_ctx * model->config.n_embd * sizeof(float));
    model->scratch.residual = malloc(n_ctx * model->config.n_embd * sizeof(float));
    model->scratch.q = malloc(n_ctx * model->config.n_embd * sizeof(float));
    model->scratch.k = malloc(n_ctx * kv_dim * sizeof(float));
    model->scratch.v = malloc(n_ctx * kv_dim * sizeof(float));
    model->scratch.att = malloc(n_ctx * n_ctx * sizeof(float));
    model->scratch.ffn_gate = malloc(n_ctx * model->config.n_ff * sizeof(float));
    model->scratch.ffn_up = malloc(n_ctx * model->config.n_ff * sizeof(float));
    model->scratch.temp = malloc(n_ctx * model->config.n_embd * sizeof(float));

    /* SambaY scratch buffer for cached SSM output (used by GMU) */
    int d_inner = model->has_ssm_layers ? model->ssm_config.d_inner : model->config.n_embd * 2;
    model->scratch.cached_ssm = malloc(n_ctx * d_inner * sizeof(float));

    /* Count differential attention layers */
    int n_diff_attn = 0;
    for (int i = 0; i < model->config.n_layer; i++) {
        if (model->weights.layers[i].use_diff_attn) {
            n_diff_attn++;
        }
    }

    printf("Loaded model: %s\n", det_arch_name(model->config.arch));
    printf("  Layers: %d, Embedding: %d, Heads: %d, Vocab: %d\n",
           model->config.n_layer, model->config.n_embd,
           model->config.n_head, model->config.n_vocab);
    if (model->has_ssm_layers) {
        printf("  Layer types: %d attention, %d SSM (Mamba)\n",
               model->n_attn_layers, model->n_ssm_layers);
    }
    if (n_diff_attn > 0) {
        printf("  Differential attention: %d layers\n", n_diff_attn);
    }
    printf("  Inference mode: %s\n",
           g_inference_mode == DET_INFERENCE_Q8_0 ? "Q8_0 (QAM)" : "F32");

    return model;
}

void det_model_free(DetModel* model) {
    if (!model) return;

    /* Free layer weights */
    if (model->weights.layers) {
        for (int i = 0; i < model->weights.n_layers; i++) {
            DetLayerWeights* layer = &model->weights.layers[i];

            /* Attention weights */
            det_tensor_release(layer->attn_norm);
            det_tensor_release(layer->wq);
            det_tensor_release(layer->wk);
            det_tensor_release(layer->wv);
            det_tensor_release(layer->wo);
            det_tensor_release(layer->bq);
            det_tensor_release(layer->bk);
            det_tensor_release(layer->bv);
            det_tensor_release(layer->q_norm);
            det_tensor_release(layer->k_norm);

            /* Differential attention weights */
            det_tensor_release(layer->diff_lambda_q1);
            det_tensor_release(layer->diff_lambda_k1);
            det_tensor_release(layer->diff_lambda_q2);
            det_tensor_release(layer->diff_lambda_k2);
            det_tensor_release(layer->diff_subln);

            /* FFN weights */
            det_tensor_release(layer->ffn_norm);
            det_tensor_release(layer->w1);
            det_tensor_release(layer->w2);
            det_tensor_release(layer->w3);

            /* SSM weights (Mamba) */
            det_tensor_release(layer->ssm_in_proj);
            det_tensor_release(layer->ssm_conv1d_weight);
            det_tensor_release(layer->ssm_conv1d_bias);
            det_tensor_release(layer->ssm_x_proj);
            det_tensor_release(layer->ssm_dt_proj);
            det_tensor_release(layer->ssm_dt_proj_bias);
            det_tensor_release(layer->ssm_A_log);
            free(layer->ssm_A_negexp);
            det_tensor_release(layer->ssm_D);
            det_tensor_release(layer->ssm_out_proj);
            det_tensor_release(layer->ssm_norm);
        }
        free(model->weights.layers);
    }

    /* Free other weights */
    det_tensor_release(model->weights.tok_embd);
    det_tensor_release(model->weights.output_norm);
    if (model->weights.output != model->weights.tok_embd) {
        det_tensor_release(model->weights.output);
    }

    /* Free KV cache */
    det_tensor_release(model->kv_cache.k);
    det_tensor_release(model->kv_cache.v);

    /* Free SSM cache */
    if (model->ssm_cache) {
        det_ssm_cache_free(model->ssm_cache);
    }

    /* Free scratch buffers */
    free(model->scratch.hidden);
    free(model->scratch.residual);
    free(model->scratch.q);
    free(model->scratch.k);
    free(model->scratch.v);
    free(model->scratch.att);
    free(model->scratch.ffn_gate);
    free(model->scratch.ffn_up);
    free(model->scratch.temp);
    free(model->scratch.cached_ssm);

    /* Free workspace */
    det_workspace_destroy(model->workspace);

    /* Free tokenizer */
    det_tokenizer_free(model->tokenizer);

    /* Free RoPE cache */
    det_tensor_release(model->rope_sin);
    det_tensor_release(model->rope_cos);

    /* Close GGUF */
    gguf_close(model->gguf);

    free(model);
}

void det_model_reset(DetModel* model) {
    if (!model) return;

    /* Reset KV cache (attention layers) */
    model->kv_cache.seq_len = 0;

    /* Reset SSM cache (Mamba layers) */
    if (model->ssm_cache) {
        det_ssm_cache_reset(model->ssm_cache);
    }
}

/* ==========================================================================
 * KV CACHE MANAGEMENT
 * ========================================================================== */

int32_t det_kv_cache_position(const DetModel* model) {
    if (!model) return 0;
    return model->kv_cache.seq_len;
}

int32_t det_kv_cache_capacity(const DetModel* model) {
    if (!model) return 0;
    return model->kv_cache.capacity;
}

int det_kv_cache_slice(DetModel* model, int32_t start, int32_t end) {
    if (!model) return -1;

    int32_t seq_len = model->kv_cache.seq_len;

    /* Validate bounds */
    if (start < 0 || end < 0 || start > end || end > seq_len) {
        fprintf(stderr, "kv_cache_slice: invalid range [%d, %d) for seq_len=%d\n",
                start, end, seq_len);
        return -1;
    }

    /* Nothing to do if slicing from beginning */
    if (start == 0) {
        model->kv_cache.seq_len = end;
        return 0;
    }

    /* Shift cache data left */
    const DetModelConfig* cfg = &model->config;
    int head_dim = cfg->n_embd / cfg->n_head;
    int kv_dim = cfg->n_head_kv * head_dim;
    int32_t new_len = end - start;

    float* k_cache = (float*)model->kv_cache.k->data;
    float* v_cache = (float*)model->kv_cache.v->data;

    /* Shift each layer's cache */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        size_t layer_offset = layer * cfg->n_ctx * kv_dim;

        /* Move [start, end) to [0, new_len) */
        memmove(k_cache + layer_offset,
                k_cache + layer_offset + start * kv_dim,
                new_len * kv_dim * sizeof(float));

        memmove(v_cache + layer_offset,
                v_cache + layer_offset + start * kv_dim,
                new_len * kv_dim * sizeof(float));
    }

    model->kv_cache.seq_len = new_len;
    return 0;
}

int det_kv_cache_shift(DetModel* model, int32_t keep_last) {
    if (!model) return -1;

    int32_t seq_len = model->kv_cache.seq_len;

    if (keep_last <= 0) {
        /* Reset cache */
        model->kv_cache.seq_len = 0;
        return 0;
    }

    if (keep_last >= seq_len) {
        /* Nothing to shift - keep everything */
        return 0;
    }

    /* Shift to keep last N tokens */
    return det_kv_cache_slice(model, seq_len - keep_last, seq_len);
}

/* ==========================================================================
 * FORWARD PASS
 * ========================================================================== */

DetTensor* det_model_forward(DetModel* model,
                             const int32_t* tokens, int32_t num_tokens,
                             DetTensor* logits) {
    if (!model || !tokens || num_tokens <= 0) return NULL;

    double t_start = 0, t_end = 0;
    if (g_debug_timing) t_start = get_time_ms();

    const DetModelConfig* cfg = &model->config;
    int head_dim = cfg->n_embd / cfg->n_head;
    int kv_dim = cfg->n_head_kv * head_dim;
    int pos = model->kv_cache.seq_len;

    /* Check context limit */
    if (pos + num_tokens > cfg->n_ctx) {
        fprintf(stderr, "Context length exceeded\n");
        return NULL;
    }

    /* Allocate output logits if not provided */
    if (!logits) {
        int32_t shape[2] = { num_tokens, cfg->n_vocab };
        logits = det_workspace_get_scratch(model->workspace, 4, 2, shape, DET_DTYPE_F32);
    }

    /* Use pre-allocated scratch buffers (avoid malloc/free per forward pass) */
    float* hidden = model->scratch.hidden;
    float* residual = model->scratch.residual;
    float* q = model->scratch.q;
    float* k = model->scratch.k;
    float* v = model->scratch.v;
    float* att = model->scratch.att;
    float* ffn_gate = model->scratch.ffn_gate;
    float* ffn_up = model->scratch.ffn_up;

    /* Get embedding weights */
    float* embd_data = (float*)model->weights.tok_embd->data;

    /* Token embedding lookup */
    for (int t = 0; t < num_tokens; t++) {
        int32_t token = tokens[t];
        if (token < 0 || token >= cfg->n_vocab) {
            token = 0;  /* Fallback to token 0 */
        }
        memcpy(hidden + t * cfg->n_embd,
               embd_data + token * cfg->n_embd,
               cfg->n_embd * sizeof(float));
    }

    double t_layers_start = g_debug_timing ? get_time_ms() : 0;
    double t_attn_total = 0, t_ffn_total = 0;

    /* SSM layer counter for cache indexing */
    int ssm_layer_idx = 0;

    /* Process each layer */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        DetLayerWeights* lw = &model->weights.layers[layer];

        double t_layer_start = g_debug_timing ? get_time_ms() : 0;

        /* Save residual */
        memcpy(residual, hidden, num_tokens * cfg->n_embd * sizeof(float));

        /* Dispatch based on layer type */
        if (lw->is_ssm_layer) {
            /* ============================================================
             * SSM LAYER (Mamba) FORWARD PASS
             * ============================================================ */

            /* SSM pre-norm (RMSNorm) */
            DetTensor* norm = lw->ssm_norm ? lw->ssm_norm : lw->attn_norm;
            if (norm) {
                float* norm_weight = (float*)norm->data;
                for (int t = 0; t < num_tokens; t++) {
                    float* h = hidden + t * cfg->n_embd;
                    float ss = 0.0f;
                    for (int i = 0; i < cfg->n_embd; i++) {
                        ss += h[i] * h[i];
                    }
                    float scale_val = 1.0f / sqrtf(ss / cfg->n_embd + cfg->norm_eps);
                    for (int i = 0; i < cfg->n_embd; i++) {
                        h[i] = h[i] * scale_val * norm_weight[i];
                    }
                }
            }

            /* Detect cross-decoder SSM (YOCO-cross) based on in_proj size
             * Cross-decoder SSM layers have smaller in_proj [d_inner, d_model]
             * vs standard SSM with [2*d_inner, d_model].
             * These layers use simplified SwiGLU with cached SSM values.
             */
            bool is_yoco_cross_ssm = false;
            if (cfg->is_sambay && lw->ssm_in_proj) {
                int d_inner = model->ssm_config.d_inner;
                if ((int32_t)lw->ssm_in_proj->shape[1] == d_inner) {
                    is_yoco_cross_ssm = true;
                }
            }

            if (is_yoco_cross_ssm && model->scratch.cached_ssm) {
                /* YOCO-cross SSM: Use simplified SwiGLU path
                 * out = out_proj(SwiGLU(in_proj(x), cached_ssm))
                 * This is the same as GMU attention layers. */
                const float* in_proj_data = lw->ssm_in_proj ?
                    (const float*)lw->ssm_in_proj->data : NULL;
                const float* out_proj_data = lw->ssm_out_proj ?
                    (const float*)lw->ssm_out_proj->data : NULL;

                if (in_proj_data && out_proj_data) {
                    det_gmu_swiglu(model->scratch.temp, hidden, model->scratch.cached_ssm,
                                   in_proj_data, out_proj_data,
                                   cfg->n_embd, model->ssm_config.d_inner, num_tokens);
                }
                ssm_layer_idx++;  /* Still count as SSM layer for stats */
            } else {
                /* Standard SSM forward pass */
                DetTensor input_tensor = {
                    .data = hidden,
                    .ndim = 2,
                    .shape = {num_tokens, cfg->n_embd, 0, 0},
                    .dtype = DET_DTYPE_F32,
                    .owns_data = false
                };
                DetTensor output_tensor = {
                    .data = model->scratch.temp,
                    .ndim = 2,
                    .shape = {num_tokens, cfg->n_embd, 0, 0},
                    .dtype = DET_DTYPE_F32,
                    .owns_data = false
                };

                /* Build SSM weights struct */
                DetSSMWeights ssm_weights = {
                    .ssm_in_proj = lw->ssm_in_proj,
                    .ssm_conv1d_weight = lw->ssm_conv1d_weight,
                    .ssm_conv1d_bias = lw->ssm_conv1d_bias,
                    .ssm_x_proj = lw->ssm_x_proj,
                    .ssm_dt_proj = lw->ssm_dt_proj,
                    .ssm_dt_proj_bias = lw->ssm_dt_proj_bias,
                    .ssm_A_log = lw->ssm_A_log,
                    .ssm_D = lw->ssm_D,
                    .ssm_out_proj = lw->ssm_out_proj,
                    .ssm_norm = lw->ssm_norm,
                    .ssm_A_negexp = lw->ssm_A_negexp
                };

                /* For SambaY, the last SSM layer in self-decoder caches its output
                 * BEFORE gating for use by cross-decoder GMU layers. */
                float* ssm_cache_buf = NULL;
                if (cfg->is_sambay) {
                    int half_layers = cfg->n_layer / 2;
                    int last_ssm_layer = (half_layers - 1) & ~1;  /* Last even layer in self-decoder */
                    if (layer == last_ssm_layer) {
                        ssm_cache_buf = model->scratch.cached_ssm;
                    }
                }

                /* Run SSM forward pass */
                det_ssm_forward(&output_tensor, &input_tensor, &ssm_weights,
                               &model->ssm_config, model->ssm_cache, ssm_layer_idx,
                               ssm_cache_buf);
                ssm_layer_idx++;
            }

            /* Copy output to hidden */
            memcpy(hidden, model->scratch.temp, num_tokens * cfg->n_embd * sizeof(float));

            /* Add residual */
            for (int i = 0; i < num_tokens * cfg->n_embd; i++) {
                hidden[i] += residual[i];
            }

            /* SSM layers may also have FFN - process it if present */
            if (lw->ffn_norm && (lw->w2 || lw->w3)) {
                /* Save residual for FFN */
                memcpy(residual, hidden, num_tokens * cfg->n_embd * sizeof(float));

                /* FFN pre-norm */
                float* norm_weight = (float*)lw->ffn_norm->data;
                for (int t = 0; t < num_tokens; t++) {
                    float* h = hidden + t * cfg->n_embd;
                    float ss = 0.0f;
                    for (int i = 0; i < cfg->n_embd; i++) {
                        ss += h[i] * h[i];
                    }
                    float scale_val = 1.0f / sqrtf(ss / cfg->n_embd + cfg->norm_eps);
                    for (int i = 0; i < cfg->n_embd; i++) {
                        h[i] = h[i] * scale_val * norm_weight[i];
                    }
                }

                /* Process FFN (same as attention layers) */
                if (lw->w1 && lw->w2 && lw->w3) {
                    /* SwiGLU variant */
                    batched_proj_smart(ffn_gate, hidden, lw->w1, num_tokens, cfg->n_embd, cfg->n_ff);
                    batched_proj_smart(ffn_up, hidden, lw->w3, num_tokens, cfg->n_embd, cfg->n_ff);
                    int ffn_total = num_tokens * cfg->n_ff;
                    for (int i = 0; i < ffn_total; i++) {
                        float g = ffn_gate[i];
                        ffn_gate[i] = (g / (1.0f + expf(-g))) * ffn_up[i];
                    }
                    batched_proj_smart(hidden, ffn_gate, lw->w2, num_tokens, cfg->n_ff, cfg->n_embd);
                } else if (lw->w2 && lw->w3) {
                    /* Simple MLP */
                    batched_proj_smart(ffn_up, hidden, lw->w3, num_tokens, cfg->n_embd, cfg->n_ff);
                    int ffn_total = num_tokens * cfg->n_ff;
                    for (int i = 0; i < ffn_total; i++) {
                        float x = ffn_up[i];
                        float x3 = x * x * x;
                        ffn_up[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
                    }
                    batched_proj_smart(hidden, ffn_up, lw->w2, num_tokens, cfg->n_ff, cfg->n_embd);
                }

                /* Add FFN residual */
                for (int i = 0; i < num_tokens * cfg->n_embd; i++) {
                    hidden[i] += residual[i];
                }
            }

            if (g_debug_timing) {
                double t_layer_end = get_time_ms();
                t_attn_total += (t_layer_end - t_layer_start);
            }

            continue;  /* Skip to next layer */
        }

        /* ============================================================
         * GMU LAYER (SambaY Cross-Decoder)
         * ============================================================ */
        if (lw->use_gmu && cfg->is_sambay) {
            /* GMU reuses cached SSM output instead of full attention/SSM.
             * This is the key efficiency of SambaY cross-decoder.
             *
             * GMU computation: out = out_proj(SwiGLU(in_proj(x), cached_ssm))
             */

            /* Pre-norm */
            DetTensor* norm = lw->attn_norm ? lw->attn_norm : lw->ffn_norm;
            if (norm) {
                float* norm_weight = (float*)norm->data;
                for (int t = 0; t < num_tokens; t++) {
                    float* h = hidden + t * cfg->n_embd;
                    float ss = 0.0f;
                    for (int i = 0; i < cfg->n_embd; i++) {
                        ss += h[i] * h[i];
                    }
                    float scale_val = 1.0f / sqrtf(ss / cfg->n_embd + cfg->norm_eps);
                    for (int i = 0; i < cfg->n_embd; i++) {
                        h[i] = h[i] * scale_val * norm_weight[i];
                    }
                }
            }

            /* GMU using cached SSM output
             * If SSM in_proj and out_proj exist, use them for GMU
             * Otherwise fallback to attention projections */
            float* in_proj_data = NULL;
            float* out_proj_data = NULL;
            int d_inner = model->ssm_config.d_inner;

            if (lw->ssm_in_proj && lw->ssm_out_proj) {
                in_proj_data = (float*)lw->ssm_in_proj->data;
                out_proj_data = (float*)lw->ssm_out_proj->data;
            } else if (lw->wq && lw->wo) {
                /* Fallback: use attention projections */
                in_proj_data = (float*)lw->wq->data;
                out_proj_data = (float*)lw->wo->data;
                d_inner = cfg->n_embd;
            }

            if (in_proj_data && out_proj_data && model->scratch.cached_ssm) {
                /* Run GMU */
                det_gmu_swiglu(model->scratch.temp, hidden, model->scratch.cached_ssm,
                               in_proj_data, out_proj_data,
                               cfg->n_embd, d_inner, num_tokens);

                /* Copy output to hidden */
                memcpy(hidden, model->scratch.temp, num_tokens * cfg->n_embd * sizeof(float));
            }

            /* Add residual */
            for (int i = 0; i < num_tokens * cfg->n_embd; i++) {
                hidden[i] += residual[i];
            }

            /* GMU layers also have FFN */
            if (lw->ffn_norm && lw->w2) {
                /* Save residual for FFN */
                memcpy(residual, hidden, num_tokens * cfg->n_embd * sizeof(float));

                /* FFN pre-norm */
                float* norm_weight = (float*)lw->ffn_norm->data;
                for (int t = 0; t < num_tokens; t++) {
                    float* h = hidden + t * cfg->n_embd;
                    float ss = 0.0f;
                    for (int i = 0; i < cfg->n_embd; i++) {
                        ss += h[i] * h[i];
                    }
                    float scale_val = 1.0f / sqrtf(ss / cfg->n_embd + cfg->norm_eps);
                    for (int i = 0; i < cfg->n_embd; i++) {
                        h[i] = h[i] * scale_val * norm_weight[i];
                    }
                }

                /* FFN (SwiGLU or simple) */
                if (lw->w1 && lw->w2 && lw->w3) {
                    batched_proj_smart(ffn_gate, hidden, lw->w1, num_tokens, cfg->n_embd, cfg->n_ff);
                    batched_proj_smart(ffn_up, hidden, lw->w3, num_tokens, cfg->n_embd, cfg->n_ff);
                    int ffn_total = num_tokens * cfg->n_ff;
                    for (int i = 0; i < ffn_total; i++) {
                        float g = ffn_gate[i];
                        ffn_gate[i] = (g / (1.0f + expf(-g))) * ffn_up[i];
                    }
                    batched_proj_smart(hidden, ffn_gate, lw->w2, num_tokens, cfg->n_ff, cfg->n_embd);
                } else if (lw->w2 && lw->w3) {
                    batched_proj_smart(ffn_up, hidden, lw->w3, num_tokens, cfg->n_embd, cfg->n_ff);
                    int ffn_total = num_tokens * cfg->n_ff;
                    for (int i = 0; i < ffn_total; i++) {
                        float x = ffn_up[i];
                        float x3 = x * x * x;
                        ffn_up[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
                    }
                    batched_proj_smart(hidden, ffn_up, lw->w2, num_tokens, cfg->n_ff, cfg->n_embd);
                }

                /* Add FFN residual */
                for (int i = 0; i < num_tokens * cfg->n_embd; i++) {
                    hidden[i] += residual[i];
                }
            }

            if (g_debug_timing) {
                double t_layer_end = get_time_ms();
                t_attn_total += (t_layer_end - t_layer_start);
            }

            continue;  /* Skip to next layer */
        }

        /* ============================================================
         * ATTENTION LAYER FORWARD PASS
         * ============================================================ */

        /* Skip layer if weights not loaded */
        if (!lw->attn_norm || !lw->wq || !lw->wk || !lw->wv || !lw->wo) {
            continue;
        }

        /* Attention pre-norm (RMSNorm) */
        if (lw->attn_norm) {
            float* norm_weight = (float*)lw->attn_norm->data;
            for (int t = 0; t < num_tokens; t++) {
                float* h = hidden + t * cfg->n_embd;
                float ss = 0.0f;
                for (int i = 0; i < cfg->n_embd; i++) {
                    ss += h[i] * h[i];
                }
                float scale = 1.0f / sqrtf(ss / cfg->n_embd + cfg->norm_eps);
                for (int i = 0; i < cfg->n_embd; i++) {
                    h[i] = h[i] * scale * norm_weight[i];
                }
            }
        }

        /* QKV projections
         * GGUF stores weights as [out_features, in_features] with ne[0]=in_features
         * For y = x @ W^T: y[i] = sum_j(x[j] * W[i * in_dim + j])
         *
         * Uses smart projection that handles both F32 and Q8_0 weights.
         */
        if (lw->wq) {
            /* QKV biases (Qwen2 uses these) */
            float* bq = lw->bq ? (float*)lw->bq->data : NULL;
            float* bk = lw->bk ? (float*)lw->bk->data : NULL;
            float* bv = lw->bv ? (float*)lw->bv->data : NULL;

            /* Batched Q projection: q[T, n_embd] = hidden[T, n_embd] @ Wq^T */
            batched_proj_smart(q, hidden, lw->wq, num_tokens, cfg->n_embd, cfg->n_embd);

            /* Batched K projection: k[T, kv_dim] = hidden[T, n_embd] @ Wk^T */
            batched_proj_smart(k, hidden, lw->wk, num_tokens, cfg->n_embd, kv_dim);

            /* Batched V projection: v[T, kv_dim] = hidden[T, n_embd] @ Wv^T */
            batched_proj_smart(v, hidden, lw->wv, num_tokens, cfg->n_embd, kv_dim);

            /* Add biases if present */
            if (bq) {
                for (int t = 0; t < num_tokens; t++) {
                    for (int i = 0; i < cfg->n_embd; i++) {
                        q[t * cfg->n_embd + i] += bq[i];
                    }
                }
            }
            if (bk) {
                for (int t = 0; t < num_tokens; t++) {
                    for (int i = 0; i < kv_dim; i++) {
                        k[t * kv_dim + i] += bk[i];
                    }
                }
            }
            if (bv) {
                for (int t = 0; t < num_tokens; t++) {
                    for (int i = 0; i < kv_dim; i++) {
                        v[t * kv_dim + i] += bv[i];
                    }
                }
            }

            /* QK-Norm: Apply RMSNorm to each head's Q and K vectors (Qwen3) */
            if (lw->q_norm && lw->k_norm) {
                float* q_norm_w = (float*)lw->q_norm->data;
                float* k_norm_w = (float*)lw->k_norm->data;

                /* Q-Norm: normalize each head's query vector */
                for (int t = 0; t < num_tokens; t++) {
                    for (int h = 0; h < cfg->n_head; h++) {
                        float* head_q = q + t * cfg->n_embd + h * head_dim;
                        /* RMSNorm for this head */
                        float sq_sum = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            sq_sum += head_q[d] * head_q[d];
                        }
                        float rms = sqrtf(sq_sum / head_dim + cfg->norm_eps);
                        for (int d = 0; d < head_dim; d++) {
                            head_q[d] = (head_q[d] / rms) * q_norm_w[d];
                        }
                    }
                }

                /* K-Norm: normalize each KV head's key vector */
                for (int t = 0; t < num_tokens; t++) {
                    for (int h = 0; h < cfg->n_head_kv; h++) {
                        float* head_k = k + t * kv_dim + h * head_dim;
                        /* RMSNorm for this head */
                        float sq_sum = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            sq_sum += head_k[d] * head_k[d];
                        }
                        float rms = sqrtf(sq_sum / head_dim + cfg->norm_eps);
                        for (int d = 0; d < head_dim; d++) {
                            head_k[d] = (head_k[d] / rms) * k_norm_w[d];
                        }
                    }
                }
            }

            /* Apply RoPE to Q and K */
            for (int t = 0; t < num_tokens; t++) {
                float* qt = q + t * cfg->n_embd;
                float* kt = k + t * kv_dim;

                /* Apply RoPE to Q heads
                 * HF-style: pairs element i with element i+half_dim
                 * cos/sin are computed for first half then repeated */
                int half_dim = head_dim / 2;
                for (int head = 0; head < cfg->n_head; head++) {
                    float* qh = qt + head * head_dim;
                    /* Compute rotated values in-place
                     * x_new[i] = x[i] * cos - x[i+half] * sin  (first half)
                     * x_new[i+half] = x[i+half] * cos + x[i] * sin (second half) */
                    for (int i = 0; i < half_dim; i++) {
                        float freq = 1.0f / powf(cfg->rope_freq_base, (float)(2 * i) / head_dim);
                        float angle = (pos + t) * freq;
                        float cos_val = cosf(angle);
                        float sin_val = sinf(angle);
                        float x0 = qh[i];
                        float x1 = qh[i + half_dim];
                        qh[i]            = x0 * cos_val - x1 * sin_val;
                        qh[i + half_dim] = x1 * cos_val + x0 * sin_val;
                    }
                }
                /* Apply RoPE to K heads (once per KV head) */
                for (int head = 0; head < cfg->n_head_kv; head++) {
                    float* kh = kt + head * head_dim;
                    for (int i = 0; i < half_dim; i++) {
                        float freq = 1.0f / powf(cfg->rope_freq_base, (float)(2 * i) / head_dim);
                        float angle = (pos + t) * freq;
                        float cos_val = cosf(angle);
                        float sin_val = sinf(angle);
                        float x0 = kh[i];
                        float x1 = kh[i + half_dim];
                        kh[i]            = x0 * cos_val - x1 * sin_val;
                        kh[i + half_dim] = x1 * cos_val + x0 * sin_val;
                    }
                }
            }
        }

        /* Store K,V in cache */
        float* k_cache = (float*)model->kv_cache.k->data;
        float* v_cache = (float*)model->kv_cache.v->data;
        size_t layer_offset = layer * cfg->n_ctx * kv_dim;

        for (int t = 0; t < num_tokens; t++) {
            memcpy(k_cache + layer_offset + (pos + t) * kv_dim,
                   k + t * kv_dim, kv_dim * sizeof(float));
            memcpy(v_cache + layer_offset + (pos + t) * kv_dim,
                   v + t * kv_dim, kv_dim * sizeof(float));
        }

        /* Attention computation */
        int seq_len = pos + num_tokens;

        if (lw->use_diff_attn) {
            /* ==========================================================
             * DIFFERENTIAL ATTENTION (phi4flash)
             *
             * Computes standard attention for each head, then pairs
             * heads (0,1), (2,3), etc. and applies:
             *   out = (attn_even - λ * attn_odd) * (1 - λ_init)
             *
             * λ = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
             * λ_init = 0.8 - 0.6 * exp(-0.3 * (layer + 1))
             * ========================================================== */
            float scale = 1.0f / sqrtf((float)head_dim);

            /* Compute lambda_init: λ_init = 0.8 - 0.6 * exp(-0.3 * (layer + 1)) */
            float lambda_init = 0.8f - 0.6f * expf(-0.3f * (layer + 1));
            float output_scale = 1.0f - lambda_init;

            /* Get lambda weights */
            float* lambda_q1 = lw->diff_lambda_q1 ? (float*)lw->diff_lambda_q1->data : NULL;
            float* lambda_k1 = lw->diff_lambda_k1 ? (float*)lw->diff_lambda_k1->data : NULL;
            float* lambda_q2 = lw->diff_lambda_q2 ? (float*)lw->diff_lambda_q2->data : NULL;
            float* lambda_k2 = lw->diff_lambda_k2 ? (float*)lw->diff_lambda_k2->data : NULL;
            float* subln_weight = lw->diff_subln ? (float*)lw->diff_subln->data : NULL;

            /* Compute lambda once (shared across all heads):
             * λ = exp(sum(λ_q1 * λ_k1)) - exp(sum(λ_q2 * λ_k2)) + λ_init */
            float lambda = lambda_init;
            if (lambda_q1 && lambda_k1 && lambda_q2 && lambda_k2) {
                float dot1 = 0.0f, dot2 = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot1 += lambda_q1[d] * lambda_k1[d];
                    dot2 += lambda_q2[d] * lambda_k2[d];
                }
                lambda = expf(dot1) - expf(dot2) + lambda_init;
            }

            for (int t = 0; t < num_tokens; t++) {
                float* qt = q + t * cfg->n_embd;
                float* out = hidden + t * cfg->n_embd;
                memset(out, 0, cfg->n_embd * sizeof(float));

                /* First pass: compute standard attention for each head */
                for (int h = 0; h < cfg->n_head; h++) {
                    int kv_head = h / (cfg->n_head / cfg->n_head_kv);
                    float* qh = qt + h * head_dim;
                    float* oh = out + h * head_dim;

                    /* Compute attention scores */
                    for (int s = 0; s < seq_len; s++) {
                        float* kh = k_cache + layer_offset + s * kv_dim + kv_head * head_dim;
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += qh[d] * kh[d];
                        }
                        att[s] = score * scale;
                    }

                    /* Causal mask */
                    for (int s = pos + t + 1; s < seq_len; s++) {
                        att[s] = -1e9f;
                    }

                    /* Sliding window mask */
                    if (lw->use_sliding_window && lw->sliding_window_size > 0) {
                        int curr_pos = pos + t;
                        int window_start = curr_pos - lw->sliding_window_size + 1;
                        if (window_start > 0) {
                            for (int s = 0; s < window_start; s++) {
                                att[s] = -1e9f;
                            }
                        }
                    }

                    /* Softmax */
                    float max_val = att[0];
                    for (int s = 1; s < seq_len; s++) {
                        if (att[s] > max_val) max_val = att[s];
                    }
                    float sum = 0.0f;
                    for (int s = 0; s < seq_len; s++) {
                        att[s] = expf(att[s] - max_val);
                        sum += att[s];
                    }
                    for (int s = 0; s < seq_len; s++) {
                        att[s] /= sum;
                    }

                    /* Apply attention to values */
                    for (int d = 0; d < head_dim; d++) {
                        float val = 0.0f;
                        for (int s = 0; s < seq_len; s++) {
                            float* vh = v_cache + layer_offset + s * kv_dim + kv_head * head_dim;
                            val += att[s] * vh[d];
                        }
                        oh[d] = val;
                    }
                }

                /* Second pass: apply differential formula to head pairs
                 * out[h1] = (out[h1] - lambda * out[h2]) * (1 - lambda_init)
                 * out[h2] = out[h2] (unchanged, will be used by next layers)
                 *
                 * Actually, differential attention combines pairs into single output:
                 * out_pair = (attn_even - lambda * attn_odd) * scale
                 * This reduces 40 heads to 40 heads with differential mixing */
                for (int h = 0; h < cfg->n_head; h += 2) {
                    float* oh1 = out + h * head_dim;       /* Even head */
                    float* oh2 = out + (h + 1) * head_dim; /* Odd head */

                    /* Apply differential formula in place */
                    for (int d = 0; d < head_dim; d++) {
                        float v1 = oh1[d];
                        float v2 = oh2[d];
                        oh1[d] = (v1 - lambda * v2) * output_scale;
                        oh2[d] = (v1 - lambda * v2) * output_scale; /* Same as h1 for now */
                    }

                    /* Apply SubLN to each head */
                    if (subln_weight) {
                        /* SubLN on oh1 */
                        float sq_sum = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            sq_sum += oh1[d] * oh1[d];
                        }
                        float rms = sqrtf(sq_sum / head_dim + cfg->norm_eps);
                        for (int d = 0; d < head_dim; d++) {
                            oh1[d] = (oh1[d] / rms) * subln_weight[d];
                        }

                        /* SubLN on oh2 (same values as oh1) */
                        sq_sum = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            sq_sum += oh2[d] * oh2[d];
                        }
                        rms = sqrtf(sq_sum / head_dim + cfg->norm_eps);
                        for (int d = 0; d < head_dim; d++) {
                            oh2[d] = (oh2[d] / rms) * subln_weight[d];
                        }
                    }
                }
            }
        } else {
            /* ==========================================================
             * STANDARD ATTENTION
             * ========================================================== */
            float scale = 1.0f / sqrtf((float)head_dim);

            for (int t = 0; t < num_tokens; t++) {
                float* qt = q + t * cfg->n_embd;
                float* out = hidden + t * cfg->n_embd;
                memset(out, 0, cfg->n_embd * sizeof(float));

                /* For each head */
                for (int h = 0; h < cfg->n_head; h++) {
                    int kv_head = h / (cfg->n_head / cfg->n_head_kv);  /* GQA mapping */
                    float* qh = qt + h * head_dim;

                    /* Compute attention scores */
                    for (int s = 0; s < seq_len; s++) {
                        float* kh = k_cache + layer_offset + s * kv_dim + kv_head * head_dim;
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += qh[d] * kh[d];
                        }
                        att[s] = score * scale;
                    }

                    /* Causal mask */
                    for (int s = pos + t + 1; s < seq_len; s++) {
                        att[s] = -1e9f;
                    }

                    /* Sliding window mask (SambaY) */
                    if (lw->use_sliding_window && lw->sliding_window_size > 0) {
                        int curr_pos = pos + t;
                        int window_start = curr_pos - lw->sliding_window_size + 1;
                        if (window_start > 0) {
                            for (int s = 0; s < window_start; s++) {
                                att[s] = -1e9f;
                            }
                        }
                    }

                    /* Softmax */
                    float max_val = att[0];
                    for (int s = 1; s < seq_len; s++) {
                        if (att[s] > max_val) max_val = att[s];
                    }
                    float sum = 0.0f;
                    for (int s = 0; s < seq_len; s++) {
                        att[s] = expf(att[s] - max_val);
                        sum += att[s];
                    }
                    for (int s = 0; s < seq_len; s++) {
                        att[s] /= sum;
                    }

                    /* Apply attention to values */
                    float* oh = out + h * head_dim;
                    for (int s = 0; s < seq_len; s++) {
                        float* vh = v_cache + layer_offset + s * kv_dim + kv_head * head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            oh[d] += att[s] * vh[d];
                        }
                    }
                }
            }
        }

        /* Output projection: out = attn @ Wo^T
         * Wo shape: [n_embd out, n_embd in] */
        if (lw->wo) {
            /* Use pre-allocated temp buffer for in-place projection */
            float* temp = model->scratch.temp;
            memcpy(temp, hidden, num_tokens * cfg->n_embd * sizeof(float));
            batched_proj_smart(hidden, temp, lw->wo, num_tokens, cfg->n_embd, cfg->n_embd);
        }

        /* Add residual */
        for (int i = 0; i < num_tokens * cfg->n_embd; i++) {
            hidden[i] += residual[i];
        }

        double t_attn_end = g_debug_timing ? get_time_ms() : 0;
        if (g_debug_timing) t_attn_total += (t_attn_end - t_layer_start);

        /* Save residual for FFN */
        memcpy(residual, hidden, num_tokens * cfg->n_embd * sizeof(float));

        double t_ffn_start = g_debug_timing ? get_time_ms() : 0;

        /* FFN pre-norm */
        if (lw->ffn_norm) {
            float* norm_weight = (float*)lw->ffn_norm->data;
            for (int t = 0; t < num_tokens; t++) {
                float* h = hidden + t * cfg->n_embd;
                float ss = 0.0f;
                for (int i = 0; i < cfg->n_embd; i++) {
                    ss += h[i] * h[i];
                }
                float scale = 1.0f / sqrtf(ss / cfg->n_embd + cfg->norm_eps);
                for (int i = 0; i < cfg->n_embd; i++) {
                    h[i] = h[i] * scale * norm_weight[i];
                }
            }
        }

        /* FFN - two variants supported:
         *
         * 1. SwiGLU (LLaMA/Qwen): gate + up + down
         *    W1 (gate) shape: [n_ff out, n_embd in]
         *    W2 (down) shape: [n_embd out, n_ff in]
         *    W3 (up) shape: [n_ff out, n_embd in]
         *    output = down(SiLU(gate(x)) * up(x))
         *
         * 2. Simple MLP (Phi): up + down with GELU
         *    W3 (up) shape: [n_ff out, n_embd in]
         *    W2 (down) shape: [n_embd out, n_ff in]
         *    output = down(GELU(up(x)))
         *
         * Uses smart projection for both F32 and Q8_0 weights.
         */
        if (lw->w1 && lw->w2 && lw->w3) {
            /* SwiGLU variant (has gate projection) */
            /* Batched gate projection: ffn_gate[T,n_ff] = hidden[T,n_embd] @ W1^T */
            batched_proj_smart(ffn_gate, hidden, lw->w1, num_tokens, cfg->n_embd, cfg->n_ff);

            /* Batched up projection: ffn_up[T,n_ff] = hidden[T,n_embd] @ W3^T */
            batched_proj_smart(ffn_up, hidden, lw->w3, num_tokens, cfg->n_embd, cfg->n_ff);

            /* SiLU(gate) * up - fused operation */
            {
                int ffn_total = num_tokens * cfg->n_ff;
                int use_cpu = 1;
#ifdef DET_USE_METAL
                if (g_metal_available && ffn_total >= 4096) {
                    if (tensor_metal_silu_mul(ffn_gate, ffn_up, ffn_gate, ffn_total) == 0) {
                        use_cpu = 0;
                    }
                }
#endif
                if (use_cpu) {
                    for (int i = 0; i < ffn_total; i++) {
                        float g = ffn_gate[i];
                        ffn_gate[i] = (g / (1.0f + expf(-g))) * ffn_up[i];
                    }
                }
            }

            /* Batched down projection: hidden[T,n_embd] = ffn_gate[T,n_ff] @ W2^T */
            batched_proj_smart(hidden, ffn_gate, lw->w2, num_tokens, cfg->n_ff, cfg->n_embd);
        } else if (lw->w2 && lw->w3) {
            /* Simple MLP variant (Phi - no gate, uses GELU) */
            /* Batched up projection: ffn_up[T,n_ff] = hidden[T,n_embd] @ W3^T */
            batched_proj_smart(ffn_up, hidden, lw->w3, num_tokens, cfg->n_embd, cfg->n_ff);

            /* GELU activation */
            {
                int ffn_total = num_tokens * cfg->n_ff;
                for (int i = 0; i < ffn_total; i++) {
                    float x = ffn_up[i];
                    /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
                    float x3 = x * x * x;
                    ffn_up[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
                }
            }

            /* Batched down projection: hidden[T,n_embd] = ffn_up[T,n_ff] @ W2^T */
            batched_proj_smart(hidden, ffn_up, lw->w2, num_tokens, cfg->n_ff, cfg->n_embd);
        }

        /* Add FFN residual */
        for (int i = 0; i < num_tokens * cfg->n_embd; i++) {
            hidden[i] += residual[i];
        }

        if (g_debug_timing) t_ffn_total += (get_time_ms() - t_ffn_start);
    }

    if (g_debug_timing) {
        printf("    Attention total: %.1fms (%.1fms/layer)\n",
               t_attn_total, t_attn_total / cfg->n_layer);
        printf("    FFN total: %.1fms (%.1fms/layer)\n",
               t_ffn_total, t_ffn_total / cfg->n_layer);
    }

    /* Final norm */
    if (model->weights.output_norm) {
        float* norm_weight = (float*)model->weights.output_norm->data;
        for (int t = 0; t < num_tokens; t++) {
            float* h = hidden + t * cfg->n_embd;
            float ss = 0.0f;
            for (int i = 0; i < cfg->n_embd; i++) {
                ss += h[i] * h[i];
            }
            float scale = 1.0f / sqrtf(ss / cfg->n_embd + cfg->norm_eps);
            for (int i = 0; i < cfg->n_embd; i++) {
                h[i] = h[i] * scale * norm_weight[i];
            }
        }
    }

    if (g_debug_timing) {
        double t_layers_end = get_time_ms();
        printf("  Layers: %.1fms\n", t_layers_end - t_layers_start);
    }

    double t_output_start = g_debug_timing ? get_time_ms() : 0;

    /* Output projection to logits
     * For tied embeddings, weight shape is (vocab_size, n_embd) stored row-major
     * logits[T, vocab] = hidden[T, embd] @ W[vocab, embd]^T
     *
     * This is typically the largest single operation (vocab can be 150K+).
     * Uses smart projection for both F32 and Q8_0 output weights.
     */
    float* logits_data = (float*)logits->data;
    if (model->weights.output) {
        batched_proj_smart(logits_data, hidden, model->weights.output,
                           num_tokens, cfg->n_embd, cfg->n_vocab);
    }

    if (g_debug_timing) {
        double t_output_end = get_time_ms();
        printf("  Output proj: %.1fms\n", t_output_end - t_output_start);
        printf("  Total forward: %.1fms\n", t_output_end - t_start);
    }

    /* Update cache position */
    model->kv_cache.seq_len = pos + num_tokens;

    /* No cleanup needed - using pre-allocated scratch buffers */
    return logits;
}

/* ==========================================================================
 * SAMPLING
 * ========================================================================== */

int32_t det_model_sample(DetModel* model, const DetTensor* logits,
                         float temperature, float top_p, int32_t top_k) {
    DetSamplingParams params = {
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repetition_penalty = 1.0f,
        .seed = 0,
    };
    return det_model_sample_ex(model, logits, &params, NULL, 0);
}

int32_t det_model_sample_ex(DetModel* model, const DetTensor* logits,
                            const DetSamplingParams* params,
                            const int32_t* context, int32_t context_len) {
    if (!model || !logits || !params) return -1;

    int32_t vocab_size = model->config.n_vocab;
    float* probs = malloc(vocab_size * sizeof(float));
    if (!probs) return -1;

    /* Get last token's logits */
    int32_t num_tokens = logits->shape[0];
    const float* last_logits = (const float*)logits->data +
                               (num_tokens - 1) * vocab_size;

    /* Copy logits */
    memcpy(probs, last_logits, vocab_size * sizeof(float));

    /* Apply repetition penalty */
    if (params->repetition_penalty != 1.0f && context && context_len > 0) {
        for (int i = 0; i < context_len; i++) {
            int32_t token = context[i];
            if (token >= 0 && token < vocab_size) {
                if (probs[token] > 0) {
                    probs[token] /= params->repetition_penalty;
                } else {
                    probs[token] *= params->repetition_penalty;
                }
            }
        }
    }

    /* Sample token */
    int32_t token = det_choose_token(model, probs, vocab_size,
                                     params->temperature, params->top_p,
                                     NULL, params->seed);

    free(probs);
    return token;
}

/* Forward declaration for stats recording (Phase 26.6) */
static void det_stats_record(DetModel* model, int32_t token_id,
                             const float* probs, int32_t k_eff, float entropy);

/* Struct for sorting (probability, token_index) pairs */
typedef struct {
    float prob;
    int32_t index;
} ProbIndexPair;

/* Comparator for descending sort by probability */
static int prob_index_cmp_desc(const void* a, const void* b) {
    float pa = ((const ProbIndexPair*)a)->prob;
    float pb = ((const ProbIndexPair*)b)->prob;
    return (pa < pb) - (pa > pb);  /* descending */
}

int32_t det_choose_token(DetModel* model,
                         const float* logits, int32_t vocab_size,
                         float temperature, float top_p,
                         float* det_presence,
                         uint64_t seed) {
    (void)model;

    /* Initialize RNG */
    uint64_t rng_state = (seed != 0) ? seed : g_rng_state;

    /* Allocate workspace */
    float* probs = malloc(vocab_size * sizeof(float));
    int32_t* indices = malloc(vocab_size * sizeof(int32_t));
    if (!probs || !indices) {
        free(probs);
        free(indices);
        return 0;
    }

    /* Copy logits and apply temperature */
    if (temperature > 0.0f) {
        for (int i = 0; i < vocab_size; i++) {
            probs[i] = logits[i] / temperature;
        }
    } else {
        /* Greedy: find argmax */
        int32_t best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        free(probs);
        free(indices);
        return best;
    }

    /* Apply DET presence bias (if provided) */
    if (det_presence) {
        for (int i = 0; i < vocab_size; i++) {
            probs[i] += det_presence[i];
        }
    }

    /* Softmax */
    float max_val = probs[0];
    for (int i = 1; i < vocab_size; i++) {
        if (probs[i] > max_val) max_val = probs[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(probs[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
        indices[i] = i;
    }

    /* Sort by probability (descending) for top-p using O(n log n) quicksort */
    {
        ProbIndexPair* pairs = malloc(vocab_size * sizeof(ProbIndexPair));
        if (pairs) {
            for (int i = 0; i < vocab_size; i++) {
                pairs[i].prob = probs[i];
                pairs[i].index = indices[i];
            }
            qsort(pairs, vocab_size, sizeof(ProbIndexPair), prob_index_cmp_desc);
            for (int i = 0; i < vocab_size; i++) {
                probs[i] = pairs[i].prob;
                indices[i] = pairs[i].index;
            }
            free(pairs);
        }
    }

    /* Top-p (nucleus) sampling */
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    /* Renormalize */
    sum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        sum += probs[i];
    }
    for (int i = 0; i < cutoff; i++) {
        probs[i] /= sum;
    }

    /* Compute entropy before sampling (for truthfulness) */
    float entropy = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        if (probs[i] > 1e-10f) {
            entropy -= probs[i] * logf(probs[i]);
        }
    }

    /* Sample from distribution */
    float r = random_float(&rng_state);
    cumsum = 0.0f;
    int32_t sampled = indices[0];
    for (int i = 0; i < cutoff; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            sampled = indices[i];
            break;
        }
    }

    /* Record stats if model has stats buffer (Phase 26.6) */
    if (model && model->gen_stats.stats) {
        det_stats_record(model, sampled, probs, cutoff, entropy);
    }

    /* Update global RNG state */
    if (seed == 0) {
        g_rng_state = rng_state;
    }

    free(probs);
    free(indices);
    return sampled;
}

/* ==========================================================================
 * TEXT GENERATION
 * ========================================================================== */

char* det_model_generate(DetModel* model,
                         const char* prompt,
                         int32_t max_tokens,
                         float temperature,
                         float top_p,
                         DetGenerateCallback callback,
                         void* user_data) {
    if (!model || !prompt) return NULL;

    /* Tokenize prompt */
    int32_t* tokens = malloc((max_tokens + 1024) * sizeof(int32_t));
    if (!tokens) return NULL;

    int32_t num_tokens = det_tokenize(model->tokenizer, prompt,
                                      tokens, 1024);
    if (num_tokens <= 0) {
        free(tokens);
        return NULL;
    }

    /* Process prompt */
    DetTensor* logits = det_model_forward(model, tokens, num_tokens, NULL);
    if (!logits) {
        free(tokens);
        return NULL;
    }

    /* Generate tokens */
    int32_t generated = 0;
    int32_t eos_id = det_eos_token(model->tokenizer);

    while (generated < max_tokens) {
        /* Sample next token */
        int32_t next_token = det_model_sample(model, logits,
                                              temperature, top_p, 0);

        if (next_token == eos_id || next_token < 0) {
            break;
        }

        tokens[num_tokens + generated] = next_token;
        generated++;

        /* Callback for streaming */
        if (callback) {
            const char* token_text = det_token_to_text(model->tokenizer, next_token);
            callback(token_text, next_token, user_data);
        }

        /* Forward next token */
        logits = det_model_forward(model, &next_token, 1, NULL);
        if (!logits) break;
    }

    /* Detokenize result */
    int32_t result_len = (num_tokens + generated) * 16;  /* Estimate */
    char* result = malloc(result_len);
    if (result) {
        det_detokenize(model->tokenizer,
                       tokens + num_tokens, generated,
                       result, result_len);
    }

    free(tokens);
    return result;
}

/* ==========================================================================
 * UTILITIES
 * ========================================================================== */

static char g_model_info[256];

const char* det_model_info(const DetModel* model) {
    if (!model) return "No model loaded";

    snprintf(g_model_info, sizeof(g_model_info),
             "%s: %d layers, %d embd, %d heads, %d vocab",
             det_arch_name(model->config.arch),
             model->config.n_layer,
             model->config.n_embd,
             model->config.n_head,
             model->config.n_vocab);

    return g_model_info;
}

size_t det_model_memory_usage(const DetModel* model) {
    if (!model) return 0;

    size_t total = sizeof(DetModel);

    /* KV cache */
    if (model->kv_cache.k) {
        total += model->kv_cache.k->data_size * 2;
    }

    /* Weights are memory-mapped, so minimal footprint */

    return total;
}

void det_model_print_config(const DetModel* model) {
    if (!model) return;

    printf("Model Configuration:\n");
    printf("  Architecture: %s\n", det_arch_name(model->config.arch));
    printf("  Vocabulary: %d\n", model->config.n_vocab);
    printf("  Context: %d\n", model->config.n_ctx);
    printf("  Embedding: %d\n", model->config.n_embd);
    printf("  Heads: %d (KV: %d)\n", model->config.n_head, model->config.n_head_kv);
    printf("  Layers: %d\n", model->config.n_layer);
    printf("  FFN: %d\n", model->config.n_ff);
    printf("  RoPE base: %.1f\n", model->config.rope_freq_base);
}

DetTokenizer* det_model_get_tokenizer(DetModel* model) {
    return model ? model->tokenizer : NULL;
}

const char* det_model_strerror(int err) {
    switch (err) {
        case DET_MODEL_OK:       return "OK";
        case DET_MODEL_ERR_IO:   return "I/O error";
        case DET_MODEL_ERR_GGUF: return "GGUF parse error";
        case DET_MODEL_ERR_ARCH: return "Unsupported architecture";
        case DET_MODEL_ERR_ALLOC: return "Allocation failed";
        case DET_MODEL_ERR_SHAPE: return "Shape mismatch";
        case DET_MODEL_ERR_INVALID: return "Invalid argument";
        default:                 return "Unknown error";
    }
}

/* ==========================================================================
 * METAL GPU QUERY
 * ========================================================================== */

int det_model_metal_available(void) {
    init_metal_if_available();
    return g_metal_available;
}

const char* det_model_metal_device(void) {
#ifdef DET_USE_METAL
    if (g_metal_available) {
        return tensor_metal_device_name();
    }
#endif
    return "CPU only";
}

/* ==========================================================================
 * PER-TOKEN STATS (Phase 26.6 - Truthfulness Hooks)
 * ========================================================================== */

void det_stats_start(DetModel* model, int32_t capacity) {
    if (!model || capacity <= 0) return;

    /* Free existing stats if any */
    det_stats_clear(model);

    /* Allocate stats buffer */
    model->gen_stats.stats = calloc(capacity, sizeof(DetTokenStats));
    model->gen_stats.count = 0;
    model->gen_stats.capacity = capacity;
}

DetTokenStats* det_stats_get(DetModel* model, int32_t* count) {
    if (!model || !model->gen_stats.stats) {
        if (count) *count = 0;
        return NULL;
    }
    if (count) *count = model->gen_stats.count;
    return model->gen_stats.stats;
}

void det_stats_aggregate(DetModel* model,
                         float* mean_entropy,
                         float* mean_k_eff,
                         float* min_entropy) {
    if (!model || !model->gen_stats.stats || model->gen_stats.count == 0) {
        if (mean_entropy) *mean_entropy = 0.0f;
        if (mean_k_eff) *mean_k_eff = 0.0f;
        if (min_entropy) *min_entropy = 0.0f;
        return;
    }

    float sum_entropy = 0.0f;
    float sum_k_eff = 0.0f;
    float min_ent = model->gen_stats.stats[0].entropy;

    for (int i = 0; i < model->gen_stats.count; i++) {
        DetTokenStats* s = &model->gen_stats.stats[i];
        sum_entropy += s->entropy;
        sum_k_eff += (float)s->k_eff;
        if (s->entropy < min_ent) min_ent = s->entropy;
    }

    int n = model->gen_stats.count;
    if (mean_entropy) *mean_entropy = sum_entropy / n;
    if (mean_k_eff) *mean_k_eff = sum_k_eff / n;
    if (min_entropy) *min_entropy = min_ent;
}

void det_stats_clear(DetModel* model) {
    if (!model) return;
    if (model->gen_stats.stats) {
        free(model->gen_stats.stats);
        model->gen_stats.stats = NULL;
    }
    model->gen_stats.count = 0;
    model->gen_stats.capacity = 0;
}

/**
 * Record stats for a sampled token (internal helper)
 *
 * Called from det_choose_token when model has stats enabled.
 */
static void det_stats_record(DetModel* model,
                             int32_t token_id,
                             const float* probs,
                             int32_t k_eff,
                             float entropy) {
    if (!model || !model->gen_stats.stats) return;
    if (model->gen_stats.count >= model->gen_stats.capacity) return;

    DetTokenStats* s = &model->gen_stats.stats[model->gen_stats.count];
    s->token_id = token_id;
    s->entropy = entropy;
    s->entropy_raw = entropy;  /* For now, same as entropy */
    s->k_eff = k_eff;

    /* Find top_prob (prob of selected token in sorted probs) */
    s->top_prob = probs[0];  /* Highest prob after sorting */

    /* Compute top5 mass */
    s->top5_mass = 0.0f;
    int top5 = (k_eff < 5) ? k_eff : 5;
    for (int i = 0; i < top5; i++) {
        s->top5_mass += probs[i];
    }

    model->gen_stats.count++;
}
