/**
 * DET Model Inference - Implementation
 * =====================================
 *
 * LLM forward pass through transformer layers.
 * Supports Metal GPU acceleration when available.
 */

#include "det_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef DET_USE_METAL
#include "det_tensor_metal.h"
#endif

/* Metal acceleration state */
static int g_metal_initialized = 0;
static int g_metal_available = 0;

/* Minimum matrix size to use Metal (avoid overhead for small matrices) */
#define METAL_MIN_ELEMENTS (64 * 64)

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
 * Uses Metal GPU acceleration when available and matrices are large enough.
 */
static void batched_proj_f32(float* out, const float* hidden, const float* W,
                             int T, int K, int N) {
#ifdef DET_USE_METAL
    /* Use Metal for large matrices where GPU overhead is amortized */
    if (g_metal_available && (T * N >= METAL_MIN_ELEMENTS || N >= 4096)) {
        if (tensor_metal_matmul_transposed_b(hidden, W, out, T, N, K) == 0) {
            return;  /* Metal succeeded */
        }
        /* Fall through to CPU on failure */
    }
#endif

    /* CPU: compute row by row (more cache-friendly for W access) */
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
        /* Try Qwen2 format */
        model->config.norm_eps = gguf_get_f32(gguf, "qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
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

    model->weights.output = gguf_get_tensor_f32(gguf, "output.weight");
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

    /* Load layer weights */
    for (int i = 0; i < model->config.n_layer; i++) {
        DetLayerWeights* layer = &model->weights.layers[i];

        layer->attn_norm = get_layer_weight(gguf, i, "attn_norm");
        layer->wq = get_layer_weight(gguf, i, "attn_q");
        layer->wk = get_layer_weight(gguf, i, "attn_k");
        layer->wv = get_layer_weight(gguf, i, "attn_v");
        layer->wo = get_layer_weight(gguf, i, "attn_output");

        /* QKV biases (Qwen2 uses these) */
        layer->bq = get_layer_bias(gguf, i, "attn_q");
        layer->bk = get_layer_bias(gguf, i, "attn_k");
        layer->bv = get_layer_bias(gguf, i, "attn_v");

        layer->ffn_norm = get_layer_weight(gguf, i, "ffn_norm");
        layer->w1 = get_layer_weight(gguf, i, "ffn_gate");
        layer->w2 = get_layer_weight(gguf, i, "ffn_down");
        layer->w3 = get_layer_weight(gguf, i, "ffn_up");
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

    printf("Loaded model: %s\n", det_arch_name(model->config.arch));
    printf("  Layers: %d, Embedding: %d, Heads: %d, Vocab: %d\n",
           model->config.n_layer, model->config.n_embd,
           model->config.n_head, model->config.n_vocab);

    return model;
}

void det_model_free(DetModel* model) {
    if (!model) return;

    /* Free layer weights */
    if (model->weights.layers) {
        for (int i = 0; i < model->weights.n_layers; i++) {
            DetLayerWeights* layer = &model->weights.layers[i];
            det_tensor_release(layer->attn_norm);
            det_tensor_release(layer->wq);
            det_tensor_release(layer->wk);
            det_tensor_release(layer->wv);
            det_tensor_release(layer->wo);
            det_tensor_release(layer->bq);
            det_tensor_release(layer->bk);
            det_tensor_release(layer->bv);
            det_tensor_release(layer->ffn_norm);
            det_tensor_release(layer->w1);
            det_tensor_release(layer->w2);
            det_tensor_release(layer->w3);
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
    model->kv_cache.seq_len = 0;
}

/* ==========================================================================
 * FORWARD PASS
 * ========================================================================== */

DetTensor* det_model_forward(DetModel* model,
                             const int32_t* tokens, int32_t num_tokens,
                             DetTensor* logits) {
    if (!model || !tokens || num_tokens <= 0) return NULL;

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

    /* Allocate working buffers */
    float* hidden = malloc(num_tokens * cfg->n_embd * sizeof(float));
    float* residual = malloc(num_tokens * cfg->n_embd * sizeof(float));
    float* q = malloc(num_tokens * cfg->n_embd * sizeof(float));
    float* k = malloc(num_tokens * kv_dim * sizeof(float));
    float* v = malloc(num_tokens * kv_dim * sizeof(float));
    float* att = malloc(num_tokens * cfg->n_ctx * sizeof(float));
    float* ffn_gate = malloc(num_tokens * cfg->n_ff * sizeof(float));
    float* ffn_up = malloc(num_tokens * cfg->n_ff * sizeof(float));

    if (!hidden || !residual || !q || !k || !v || !att || !ffn_gate || !ffn_up) {
        free(hidden); free(residual); free(q); free(k); free(v);
        free(att); free(ffn_gate); free(ffn_up);
        return NULL;
    }

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

    /* Process each layer */
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        DetLayerWeights* lw = &model->weights.layers[layer];

        /* Skip layer if weights not loaded */
        if (!lw->attn_norm || !lw->wq || !lw->wk || !lw->wv || !lw->wo) {
            continue;
        }

        /* Save residual */
        memcpy(residual, hidden, num_tokens * cfg->n_embd * sizeof(float));

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
         * Uses batched projection for better cache utilization.
         */
        if (lw->wq && lw->wq->dtype == DET_DTYPE_F32) {
            float* wq = (float*)lw->wq->data;
            float* wk = (float*)lw->wk->data;
            float* wv = (float*)lw->wv->data;
            /* QKV biases (Qwen2 uses these) */
            float* bq = lw->bq ? (float*)lw->bq->data : NULL;
            float* bk = lw->bk ? (float*)lw->bk->data : NULL;
            float* bv = lw->bv ? (float*)lw->bv->data : NULL;

            /* Batched Q projection: q[T, n_embd] = hidden[T, n_embd] @ Wq^T */
            batched_proj_f32(q, hidden, wq, num_tokens, cfg->n_embd, cfg->n_embd);

            /* Batched K projection: k[T, kv_dim] = hidden[T, n_embd] @ Wk^T */
            batched_proj_f32(k, hidden, wk, num_tokens, cfg->n_embd, kv_dim);

            /* Batched V projection: v[T, kv_dim] = hidden[T, n_embd] @ Wv^T */
            batched_proj_f32(v, hidden, wv, num_tokens, cfg->n_embd, kv_dim);

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

        /* Attention computation (simplified) */
        float scale = 1.0f / sqrtf((float)head_dim);
        int seq_len = pos + num_tokens;

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

        /* Output projection: out = attn @ Wo^T
         * Wo shape: [n_embd out, n_embd in] */
        if (lw->wo && lw->wo->dtype == DET_DTYPE_F32) {
            float* wo = (float*)lw->wo->data;
            for (int t = 0; t < num_tokens; t++) {
                float* h = hidden + t * cfg->n_embd;
                float temp[4096];  /* Assume n_embd <= 4096 */
                memcpy(temp, h, cfg->n_embd * sizeof(float));

                for (int i = 0; i < cfg->n_embd; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < cfg->n_embd; j++) {
                        sum += temp[j] * wo[i * cfg->n_embd + j];
                    }
                    h[i] = sum;
                }
            }
        }

        /* Add residual */
        for (int i = 0; i < num_tokens * cfg->n_embd; i++) {
            hidden[i] += residual[i];
        }

        /* Save residual for FFN */
        memcpy(residual, hidden, num_tokens * cfg->n_embd * sizeof(float));

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

        /* FFN (SwiGLU)
         * W1 (gate) shape: [n_ff out, n_embd in]
         * W2 (down) shape: [n_embd out, n_ff in]
         * W3 (up) shape: [n_ff out, n_embd in]
         *
         * Uses batched projection for better cache/Metal utilization.
         */
        if (lw->w1 && lw->w2 && lw->w3 &&
            lw->w1->dtype == DET_DTYPE_F32) {
            float* w1 = (float*)lw->w1->data;  /* gate */
            float* w2 = (float*)lw->w2->data;  /* down */
            float* w3 = (float*)lw->w3->data;  /* up */

            /* Batched gate projection: ffn_gate[T,n_ff] = hidden[T,n_embd] @ W1^T */
            batched_proj_f32(ffn_gate, hidden, w1, num_tokens, cfg->n_embd, cfg->n_ff);

            /* Batched up projection: ffn_up[T,n_ff] = hidden[T,n_embd] @ W3^T */
            batched_proj_f32(ffn_up, hidden, w3, num_tokens, cfg->n_embd, cfg->n_ff);

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
            batched_proj_f32(hidden, ffn_gate, w2, num_tokens, cfg->n_ff, cfg->n_embd);
        }

        /* Add FFN residual */
        for (int i = 0; i < num_tokens * cfg->n_embd; i++) {
            hidden[i] += residual[i];
        }
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

    /* Output projection to logits
     * For tied embeddings, weight shape is (vocab_size, n_embd) stored row-major
     * logits[T, vocab] = hidden[T, embd] @ W[vocab, embd]^T
     *
     * This is typically the largest single operation (vocab can be 150K+).
     */
    float* logits_data = (float*)logits->data;
    if (model->weights.output && model->weights.output->dtype == DET_DTYPE_F32) {
        float* wo = (float*)model->weights.output->data;
        /* Use batched projection for potential Metal acceleration */
        batched_proj_f32(logits_data, hidden, wo,
                         num_tokens, cfg->n_embd, cfg->n_vocab);
    }

    /* Update cache position */
    model->kv_cache.seq_len = pos + num_tokens;

    /* Cleanup */
    free(hidden);
    free(residual);
    free(q);
    free(k);
    free(v);
    free(att);
    free(ffn_gate);
    free(ffn_up);

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

    /* Sort by probability (descending) for top-p */
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs[j] > probs[i]) {
                float tmp_p = probs[i]; probs[i] = probs[j]; probs[j] = tmp_p;
                int32_t tmp_i = indices[i]; indices[i] = indices[j]; indices[j] = tmp_i;
            }
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
