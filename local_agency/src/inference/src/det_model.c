/**
 * DET Model Inference - Implementation
 * =====================================
 *
 * LLM forward pass through transformer layers.
 */

#include "det_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* RoPE application */
static void apply_rope(float* q, float* k, int head_dim, int pos, float theta) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        /* Apply rotation to q */
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        /* Apply rotation to k */
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i]     = k0 * cos_val - k1 * sin_val;
        k[i + 1] = k0 * sin_val + k1 * cos_val;
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
         */
        if (lw->wq && lw->wq->dtype == DET_DTYPE_F32) {
            float* wq = (float*)lw->wq->data;
            float* wk = (float*)lw->wk->data;
            float* wv = (float*)lw->wv->data;
            /* QKV biases (Qwen2 uses these) */
            float* bq = lw->bq ? (float*)lw->bq->data : NULL;
            float* bk = lw->bk ? (float*)lw->bk->data : NULL;
            float* bv = lw->bv ? (float*)lw->bv->data : NULL;

            for (int t = 0; t < num_tokens; t++) {
                float* h = hidden + t * cfg->n_embd;
                float* qt = q + t * cfg->n_embd;
                float* kt = k + t * kv_dim;
                float* vt = v + t * kv_dim;

                /* Q = h @ Wq^T + bq
                 * Wq shape: [n_embd out, n_embd in] */
                for (int i = 0; i < cfg->n_embd; i++) {
                    float sum = bq ? bq[i] : 0.0f;
                    for (int j = 0; j < cfg->n_embd; j++) {
                        sum += h[j] * wq[i * cfg->n_embd + j];
                    }
                    qt[i] = sum;
                }

                /* K = h @ Wk^T + bk
                 * Wk shape: [kv_dim out, n_embd in] */
                for (int i = 0; i < kv_dim; i++) {
                    float sum = bk ? bk[i] : 0.0f;
                    for (int j = 0; j < cfg->n_embd; j++) {
                        sum += h[j] * wk[i * cfg->n_embd + j];
                    }
                    kt[i] = sum;
                }

                /* V = h @ Wv^T + bv
                 * Wv shape: [kv_dim out, n_embd in] */
                for (int i = 0; i < kv_dim; i++) {
                    float sum = bv ? bv[i] : 0.0f;
                    for (int j = 0; j < cfg->n_embd; j++) {
                        sum += h[j] * wv[i * cfg->n_embd + j];
                    }
                    vt[i] = sum;
                }

                /* Apply RoPE to Q heads */
                for (int head = 0; head < cfg->n_head; head++) {
                    float* qh = qt + head * head_dim;
                    for (int i = 0; i < head_dim; i += 2) {
                        float freq = 1.0f / powf(cfg->rope_freq_base, (float)i / head_dim);
                        float angle = (pos + t) * freq;
                        float cos_val = cosf(angle);
                        float sin_val = sinf(angle);
                        float q0 = qh[i];
                        float q1 = qh[i + 1];
                        qh[i]     = q0 * cos_val - q1 * sin_val;
                        qh[i + 1] = q0 * sin_val + q1 * cos_val;
                    }
                }
                /* Apply RoPE to K heads (once per KV head) */
                for (int head = 0; head < cfg->n_head_kv; head++) {
                    float* kh = kt + head * head_dim;
                    for (int i = 0; i < head_dim; i += 2) {
                        float freq = 1.0f / powf(cfg->rope_freq_base, (float)i / head_dim);
                        float angle = (pos + t) * freq;
                        float cos_val = cosf(angle);
                        float sin_val = sinf(angle);
                        float k0 = kh[i];
                        float k1 = kh[i + 1];
                        kh[i]     = k0 * cos_val - k1 * sin_val;
                        kh[i + 1] = k0 * sin_val + k1 * cos_val;
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
         */
        if (lw->w1 && lw->w2 && lw->w3 &&
            lw->w1->dtype == DET_DTYPE_F32) {
            float* w1 = (float*)lw->w1->data;  /* gate */
            float* w2 = (float*)lw->w2->data;  /* down */
            float* w3 = (float*)lw->w3->data;  /* up */

            for (int t = 0; t < num_tokens; t++) {
                float* h = hidden + t * cfg->n_embd;
                float* gate = ffn_gate + t * cfg->n_ff;
                float* up = ffn_up + t * cfg->n_ff;

                /* gate = h @ W1^T */
                for (int i = 0; i < cfg->n_ff; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < cfg->n_embd; j++) {
                        sum += h[j] * w1[i * cfg->n_embd + j];
                    }
                    gate[i] = sum;
                }

                /* up = h @ W3^T */
                for (int i = 0; i < cfg->n_ff; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < cfg->n_embd; j++) {
                        sum += h[j] * w3[i * cfg->n_embd + j];
                    }
                    up[i] = sum;
                }

                /* SiLU(gate) * up */
                for (int i = 0; i < cfg->n_ff; i++) {
                    float g = gate[i];
                    gate[i] = (g / (1.0f + expf(-g))) * up[i];
                }

                /* h = gate @ W2^T */
                for (int i = 0; i < cfg->n_embd; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < cfg->n_ff; j++) {
                        sum += gate[j] * w2[i * cfg->n_ff + j];
                    }
                    h[i] = sum;
                }
            }
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
     * logits[i] = sum_j(hidden[j] * weight[i][j]) = sum_j(hidden[j] * weight[i * n_embd + j]) */
    float* logits_data = (float*)logits->data;
    if (model->weights.output && model->weights.output->dtype == DET_DTYPE_F32) {
        float* wo = (float*)model->weights.output->data;
        for (int t = 0; t < num_tokens; t++) {
            float* h = hidden + t * cfg->n_embd;
            float* l = logits_data + t * cfg->n_vocab;
            for (int i = 0; i < cfg->n_vocab; i++) {
                float sum = 0.0f;
                for (int j = 0; j < cfg->n_embd; j++) {
                    sum += h[j] * wo[i * cfg->n_embd + j];  /* Note: row-major (vocab, embd) */
                }
                l[i] = sum;
            }
        }
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
