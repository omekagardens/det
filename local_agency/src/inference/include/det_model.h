/**
 * DET Model Inference - Phase 26.4
 * =================================
 *
 * LLM inference pipeline for DET-native model execution.
 * Supports LLaMA-family architectures (LLaMA, Qwen2, Mistral, etc.)
 *
 * Key Features:
 * - Memory-mapped weights from GGUF files
 * - KV cache for efficient token generation
 * - DET-aware sampling (det_choose_token)
 */

#ifndef DET_MODEL_H
#define DET_MODEL_H

#include "det_tensor.h"
#include "det_gguf.h"
#include "det_tokenizer.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * INFERENCE MODE CONFIGURATION
 * ========================================================================== */

/**
 * Inference mode determines how weights are stored and processed.
 *
 * DET_INFERENCE_F32:  (Default) Dequantize all weights to F32 at load time.
 *                     Higher memory usage (~8x for Q8_0 models).
 *                     Fastest computation.
 *
 * DET_INFERENCE_Q8_0: Keep Q8_0 weights quantized in memory.
 *                     Lower memory usage (~4x savings).
 *                     On-the-fly dequantization during matmul (QAM).
 */
typedef enum {
    DET_INFERENCE_F32 = 0,   /* Dequantize at load time (current behavior) */
    DET_INFERENCE_Q8_0 = 1,  /* Keep quantized, dequant during matmul (QAM) */
} DetInferenceMode;

/**
 * Set global inference mode for weight loading
 *
 * Must be called before det_model_load() to take effect.
 */
void det_set_inference_mode(DetInferenceMode mode);

/**
 * Get current inference mode
 */
DetInferenceMode det_get_inference_mode(void);

/* ==========================================================================
 * MODEL CONFIGURATION
 * ========================================================================== */

/** Model hyperparameters */
typedef struct {
    int32_t n_vocab;        /* Vocabulary size */
    int32_t n_ctx;          /* Maximum context length */
    int32_t n_embd;         /* Embedding dimension */
    int32_t n_head;         /* Number of attention heads */
    int32_t n_head_kv;      /* Number of KV heads (for GQA) */
    int32_t n_layer;        /* Number of transformer layers */
    int32_t n_ff;           /* Feed-forward hidden dimension */
    int32_t n_rot;          /* Number of rotary dimensions */

    float rope_freq_base;   /* RoPE frequency base */
    float rope_freq_scale;  /* RoPE frequency scale */
    float norm_eps;         /* Normalization epsilon */

    DetModelArch arch;      /* Model architecture type */
} DetModelConfig;

/* ==========================================================================
 * MODEL WEIGHTS
 * ========================================================================== */

/** Transformer layer weights */
typedef struct {
    /* Attention */
    DetTensor* wq;          /* Query projection [n_embd, n_embd] */
    DetTensor* wk;          /* Key projection [n_embd, n_head_kv * head_dim] */
    DetTensor* wv;          /* Value projection [n_embd, n_head_kv * head_dim] */
    DetTensor* wo;          /* Output projection [n_embd, n_embd] */

    /* Attention biases (Qwen2 uses these) */
    DetTensor* bq;          /* Query bias [n_embd] */
    DetTensor* bk;          /* Key bias [n_head_kv * head_dim] */
    DetTensor* bv;          /* Value bias [n_head_kv * head_dim] */

    /* Attention normalization */
    DetTensor* attn_norm;   /* Pre-attention RMSNorm weights */

    /* Feed-forward */
    DetTensor* w1;          /* FFN gate projection [n_embd, n_ff] */
    DetTensor* w2;          /* FFN down projection [n_ff, n_embd] */
    DetTensor* w3;          /* FFN up projection [n_embd, n_ff] */

    /* FFN normalization */
    DetTensor* ffn_norm;    /* Pre-FFN RMSNorm weights */
} DetLayerWeights;

/** Model weights */
typedef struct {
    DetTensor* tok_embd;        /* Token embeddings [n_vocab, n_embd] */
    DetTensor* output_norm;     /* Final RMSNorm weights [n_embd] */
    DetTensor* output;          /* Output projection [n_embd, n_vocab] */

    DetLayerWeights* layers;    /* Layer weights array */
    int32_t n_layers;
} DetModelWeights;

/* ==========================================================================
 * KV CACHE
 * ========================================================================== */

/** Key-Value cache for efficient generation */
typedef struct {
    DetTensor* k;           /* Key cache [n_layer, n_ctx, n_head_kv, head_dim] */
    DetTensor* v;           /* Value cache [n_layer, n_ctx, n_head_kv, head_dim] */
    int32_t seq_len;        /* Current sequence length */
    int32_t capacity;       /* Maximum sequence length */
} DetKVCache;

/* ==========================================================================
 * MODEL CONTEXT
 * ========================================================================== */

/** Pre-allocated scratch buffers for forward pass */
typedef struct {
    float* hidden;      /* [n_ctx, n_embd] */
    float* residual;    /* [n_ctx, n_embd] */
    float* q;           /* [n_ctx, n_embd] */
    float* k;           /* [n_ctx, kv_dim] */
    float* v;           /* [n_ctx, kv_dim] */
    float* att;         /* [n_ctx, n_ctx] */
    float* ffn_gate;    /* [n_ctx, n_ff] */
    float* ffn_up;      /* [n_ctx, n_ff] */
    float* temp;        /* [n_ctx, n_embd] for output projection */
} DetScratchBuffers;

/* ==========================================================================
 * PER-TOKEN STATS (Phase 26.6 - Truthfulness Hooks)
 * ========================================================================== */

/**
 * Per-token statistics for truthfulness evaluation
 *
 * Captured during sampling to provide real entropy/k_eff values
 * instead of hardcoded estimates.
 */
typedef struct {
    float entropy;          /* Logit distribution entropy (after temperature) */
    float entropy_raw;      /* Raw entropy (before temperature) */
    int32_t k_eff;          /* Effective candidates (nucleus set size) */
    float top_prob;         /* Probability of selected token */
    float top5_mass;        /* Total probability mass in top 5 tokens */
    int32_t token_id;       /* The selected token */
} DetTokenStats;

/**
 * Buffer for per-generation token stats
 */
typedef struct {
    DetTokenStats* stats;   /* Array of per-token stats */
    int32_t count;          /* Number of tokens generated */
    int32_t capacity;       /* Maximum capacity */
} DetGenerationStats;

/* ==========================================================================
 * MODEL CONTEXT
 * ========================================================================== */

/** Model inference context */
typedef struct DetModel {
    /* Configuration */
    DetModelConfig config;

    /* Weights (memory-mapped from GGUF) */
    DetModelWeights weights;

    /* KV cache */
    DetKVCache kv_cache;

    /* Pre-allocated scratch buffers (avoid malloc in forward pass) */
    DetScratchBuffers scratch;

    /* Tokenizer */
    DetTokenizer* tokenizer;

    /* GGUF context (keeps file mapped) */
    GgufContext* gguf;

    /* Workspace for intermediate tensors */
    DetTensorWorkspace* workspace;

    /* RoPE sin/cos cache */
    DetTensor* rope_sin;
    DetTensor* rope_cos;

    /* Per-token stats for truthfulness (Phase 26.6) */
    DetGenerationStats gen_stats;
} DetModel;

/* ==========================================================================
 * MODEL LIFECYCLE
 * ========================================================================== */

/**
 * Load model from GGUF file
 *
 * Memory-maps weights for efficient loading.
 * Returns NULL on error.
 */
DetModel* det_model_load(const char* path);

/**
 * Free model and all resources
 */
void det_model_free(DetModel* model);

/**
 * Reset KV cache (for new conversation)
 */
void det_model_reset(DetModel* model);

/* ==========================================================================
 * INFERENCE
 * ========================================================================== */

/**
 * Compute logits for a sequence of tokens
 *
 * tokens: Input token IDs
 * num_tokens: Number of tokens
 * logits: Output logits tensor [num_tokens, vocab_size] (if NULL, uses internal buffer)
 *
 * Returns: Pointer to logits tensor, or NULL on error
 */
DetTensor* det_model_forward(DetModel* model,
                             const int32_t* tokens, int32_t num_tokens,
                             DetTensor* logits);

/**
 * Generate next token given logits
 *
 * Uses DET-aware sampling (sacred point for DET integration)
 */
int32_t det_model_sample(DetModel* model, const DetTensor* logits,
                         float temperature, float top_p, int32_t top_k);

/**
 * Generate text from prompt
 *
 * prompt: Input prompt text
 * max_tokens: Maximum tokens to generate
 * temperature: Sampling temperature (0.0 = greedy)
 * top_p: Top-p (nucleus) sampling threshold
 * callback: Called for each generated token (can be NULL)
 * user_data: Passed to callback
 *
 * Returns: Allocated string with generated text (caller must free)
 */
typedef void (*DetGenerateCallback)(const char* text, int32_t token_id, void* user_data);

char* det_model_generate(DetModel* model,
                         const char* prompt,
                         int32_t max_tokens,
                         float temperature,
                         float top_p,
                         DetGenerateCallback callback,
                         void* user_data);

/* ==========================================================================
 * SAMPLING PARAMETERS
 * ========================================================================== */

/** Sampling configuration */
typedef struct {
    float temperature;      /* Temperature (1.0 = normal, 0.0 = greedy) */
    float top_p;            /* Top-p (nucleus) sampling */
    int32_t top_k;          /* Top-k sampling (0 = disabled) */
    float repetition_penalty;   /* Repetition penalty (1.0 = none) */
    float presence_penalty;     /* Presence penalty */
    float frequency_penalty;    /* Frequency penalty */
    uint64_t seed;          /* Random seed (0 = random) */
} DetSamplingParams;

/** Default sampling parameters */
static inline DetSamplingParams det_default_sampling(void) {
    return (DetSamplingParams){
        .temperature = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
        .repetition_penalty = 1.1f,
        .presence_penalty = 0.0f,
        .frequency_penalty = 0.0f,
        .seed = 0,
    };
}

/**
 * Sample with full parameters
 */
int32_t det_model_sample_ex(DetModel* model, const DetTensor* logits,
                            const DetSamplingParams* params,
                            const int32_t* context, int32_t context_len);

/* ==========================================================================
 * DET INTEGRATION
 * ========================================================================== */

/**
 * DET-aware token selection
 *
 * This is the sacred integration point where DET physics
 * can influence token selection through presence (P) values.
 *
 * In full DET mode, this function consults the substrate
 * to bias sampling toward tokens aligned with agent state.
 */
int32_t det_choose_token(DetModel* model,
                         const float* logits, int32_t vocab_size,
                         float temperature, float top_p,
                         float* det_presence,  /* Optional DET presence values */
                         uint64_t seed);

/* ==========================================================================
 * PER-TOKEN STATS API (Phase 26.6 - Truthfulness Hooks)
 * ========================================================================== */

/**
 * Start collecting token stats for a generation
 *
 * capacity: Maximum tokens to track (stats buffer size)
 */
void det_stats_start(DetModel* model, int32_t capacity);

/**
 * Get collected token stats
 *
 * Returns array of DetTokenStats, sets count to number of tokens.
 * Returns NULL if stats not started.
 */
DetTokenStats* det_stats_get(DetModel* model, int32_t* count);

/**
 * Get aggregated stats for generation
 *
 * mean_entropy: Average entropy across all tokens
 * mean_k_eff: Average effective candidates
 * min_entropy: Minimum entropy (most confident token)
 */
void det_stats_aggregate(DetModel* model,
                         float* mean_entropy,
                         float* mean_k_eff,
                         float* min_entropy);

/**
 * Clear stats buffer (call before new generation)
 */
void det_stats_clear(DetModel* model);

/* ==========================================================================
 * UTILITIES
 * ========================================================================== */

/**
 * Get model info string
 */
const char* det_model_info(const DetModel* model);

/**
 * Get model memory usage in bytes
 */
size_t det_model_memory_usage(const DetModel* model);

/**
 * Print model configuration
 */
void det_model_print_config(const DetModel* model);

/**
 * Get tokenizer from model (for Python bindings)
 */
DetTokenizer* det_model_get_tokenizer(DetModel* model);

/* ==========================================================================
 * ERROR CODES
 * ========================================================================== */

#define DET_MODEL_OK          0
#define DET_MODEL_ERR_IO     -1
#define DET_MODEL_ERR_GGUF   -2
#define DET_MODEL_ERR_ARCH   -3
#define DET_MODEL_ERR_ALLOC  -4
#define DET_MODEL_ERR_SHAPE  -5
#define DET_MODEL_ERR_INVALID -6

/**
 * Get error message
 */
const char* det_model_strerror(int err);

/**
 * Enable/disable timing debug output
 */
void det_enable_timing(int enable);

/* ==========================================================================
 * METAL GPU ACCELERATION
 * ========================================================================== */

/**
 * Check if Metal GPU acceleration is available
 *
 * Returns 1 if Metal is available and initialized, 0 otherwise.
 */
int det_model_metal_available(void);

/**
 * Get Metal device name
 *
 * Returns the name of the Metal GPU device, or "CPU only" if not available.
 */
const char* det_model_metal_device(void);

#ifdef __cplusplus
}
#endif

#endif /* DET_MODEL_H */
