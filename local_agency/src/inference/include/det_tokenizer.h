/**
 * DET Tokenizer - Phase 26.3
 * ==========================
 *
 * BPE/SentencePiece tokenizer for DET inference.
 * Loads vocabulary from GGUF model files.
 *
 * Supports:
 * - BPE (Byte-Pair Encoding) used by LLaMA, Qwen, etc.
 * - Pre-tokenization rules for different model families
 * - Special tokens (BOS, EOS, PAD, etc.)
 */

#ifndef DET_TOKENIZER_H
#define DET_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * TOKENIZER TYPES
 * ========================================================================== */

/** Tokenizer model type */
typedef enum {
    DET_TOKENIZER_BPE = 0,      /* Byte-Pair Encoding */
    DET_TOKENIZER_UNIGRAM = 1,  /* Unigram (SentencePiece) */
    DET_TOKENIZER_WORD = 2,     /* Word-level */
} DetTokenizerType;

/** Special token IDs */
typedef struct {
    int32_t bos_id;     /* Beginning of sequence */
    int32_t eos_id;     /* End of sequence */
    int32_t pad_id;     /* Padding */
    int32_t unk_id;     /* Unknown token */
    int32_t nl_id;      /* Newline */
} DetSpecialTokens;

/** Vocabulary entry */
typedef struct {
    char* text;         /* Token text (UTF-8) */
    float score;        /* Token score/probability */
    int32_t type;       /* Token type (normal, special, byte, etc.) */
} DetVocabEntry;

/** BPE merge rule */
typedef struct {
    int32_t left;       /* Left token ID */
    int32_t right;      /* Right token ID */
    int32_t result;     /* Merged token ID */
} DetBPEMerge;

/** BPE merge lookup entry (for O(log n) binary search) */
typedef struct {
    uint64_t key;       /* (left << 32) | right - for sorting/search */
    int32_t result;     /* Merged token ID */
    int32_t priority;   /* Merge priority (lower = higher priority) */
} DetMergeLookup;

/** Added token entry (for special tokens like <|end|>) */
typedef struct {
    const char* text;   /* Token text (points to vocab entry) */
    int32_t id;         /* Token ID */
    int32_t len;        /* Pre-computed strlen for efficiency */
} DetAddedToken;

/** Trie node for added token matching */
typedef struct DetTrieNode {
    struct DetTrieNode* children[256];  /* Child nodes by byte */
    int32_t token_id;                   /* Token ID if terminal (-1 otherwise) */
} DetTrieNode;

/** Tokenizer context */
typedef struct DetTokenizer {
    /* Vocabulary */
    DetVocabEntry* vocab;
    int32_t vocab_size;

    /* Type */
    DetTokenizerType type;

    /* Special tokens */
    DetSpecialTokens special;

    /* BPE merge rules */
    DetBPEMerge* merges;
    int32_t num_merges;

    /* Optimized merge lookup (sorted by key for binary search) */
    DetMergeLookup* merge_lookup;   /* Sorted array for O(log n) merge search */
    int32_t merge_lookup_count;     /* Number of valid entries */

    /* Lookup tables for fast encoding */
    void* token_to_id;      /* Hash table: text -> id */
    int32_t* byte_token_ids;/* Precomputed byte fallback IDs (256 entries) */

    /* Added tokens (special tokens matched before BPE) */
    DetAddedToken* added_tokens;    /* Array of added tokens, sorted by length desc */
    int32_t num_added_tokens;       /* Number of added tokens */
    DetTrieNode* added_tokens_trie; /* Trie for O(k) added token matching */

    /* Configuration */
    bool add_bos;           /* Add BOS token at start */
    bool add_eos;           /* Add EOS token at end */
    const char* pre_tokenizer;  /* Pre-tokenization pattern */
} DetTokenizer;

/* ==========================================================================
 * TOKENIZER LIFECYCLE
 * ========================================================================== */

/**
 * Create tokenizer from GGUF model file
 *
 * Extracts vocabulary and merge rules from GGUF metadata.
 * Returns NULL on error.
 */
#include "det_gguf.h"
DetTokenizer* det_tokenizer_from_gguf(GgufContext* gguf);

/**
 * Create tokenizer from vocabulary file
 *
 * Supports JSON vocab files and tokenizer.model (SentencePiece).
 */
DetTokenizer* det_tokenizer_load(const char* path);

/**
 * Destroy tokenizer and free resources
 */
void det_tokenizer_free(DetTokenizer* tok);

/* ==========================================================================
 * ENCODING (Text -> Tokens)
 * ========================================================================== */

/**
 * Encode text to token IDs
 *
 * text: UTF-8 input text
 * tokens: Output array (caller-allocated)
 * max_tokens: Maximum number of tokens to output
 *
 * Returns: Number of tokens written, or negative error code
 */
int32_t det_tokenize(const DetTokenizer* tok, const char* text,
                     int32_t* tokens, int32_t max_tokens);

/**
 * Encode with options
 *
 * add_bos: Prepend BOS token
 * add_eos: Append EOS token
 */
int32_t det_tokenize_ex(const DetTokenizer* tok, const char* text,
                        int32_t* tokens, int32_t max_tokens,
                        bool add_bos, bool add_eos);

/**
 * Count tokens in text (without allocating)
 */
int32_t det_token_count(const DetTokenizer* tok, const char* text);

/* ==========================================================================
 * DECODING (Tokens -> Text)
 * ========================================================================== */

/**
 * Decode single token to text
 *
 * Returns: Token text (valid until tokenizer is freed)
 */
const char* det_token_to_text(const DetTokenizer* tok, int32_t token_id);

/**
 * Decode single token to text with BPE decoding (Ġ→space, Ċ→newline)
 *
 * Returns: Decoded token text (valid until next call - uses static buffer)
 */
const char* det_token_to_text_decoded(const DetTokenizer* tok, int32_t token_id);

/**
 * Decode token sequence to text
 *
 * tokens: Input token IDs
 * num_tokens: Number of tokens
 * text: Output buffer (caller-allocated)
 * max_len: Maximum output length
 *
 * Returns: Number of bytes written (excluding null terminator)
 */
int32_t det_detokenize(const DetTokenizer* tok,
                       const int32_t* tokens, int32_t num_tokens,
                       char* text, int32_t max_len);

/**
 * Decode incrementally (for streaming)
 *
 * Handles partial multi-byte characters properly.
 * prev_token: Previous token (for merge handling), or -1
 */
int32_t det_detokenize_incremental(const DetTokenizer* tok,
                                    int32_t token_id, int32_t prev_token,
                                    char* text, int32_t max_len);

/* ==========================================================================
 * SPECIAL TOKENS
 * ========================================================================== */

/**
 * Get special token ID
 *
 * name: Token name ("bos", "eos", "pad", "unk", etc.)
 * Returns: Token ID or -1 if not found
 */
int32_t det_get_special_token(const DetTokenizer* tok, const char* name);

/**
 * Check if token is special
 */
bool det_is_special_token(const DetTokenizer* tok, int32_t token_id);

/**
 * Get BOS token ID
 */
static inline int32_t det_bos_token(const DetTokenizer* tok) {
    return tok ? tok->special.bos_id : -1;
}

/**
 * Get EOS token ID
 */
static inline int32_t det_eos_token(const DetTokenizer* tok) {
    return tok ? tok->special.eos_id : -1;
}

/* ==========================================================================
 * UTILITIES
 * ========================================================================== */

/**
 * Get vocabulary size
 */
static inline int32_t det_vocab_size(const DetTokenizer* tok) {
    return tok ? tok->vocab_size : 0;
}

/**
 * Print tokenization for debugging
 */
void det_print_tokens(const DetTokenizer* tok,
                      const int32_t* tokens, int32_t num_tokens);

/* Non-inline wrappers for ctypes/FFI binding */
int32_t det_bos_token_export(const DetTokenizer* tok);
int32_t det_eos_token_export(const DetTokenizer* tok);
int32_t det_vocab_size_export(const DetTokenizer* tok);

/* ==========================================================================
 * ERROR CODES
 * ========================================================================== */

#define DET_TOK_OK           0
#define DET_TOK_ERR_ALLOC   -1
#define DET_TOK_ERR_INVALID -2
#define DET_TOK_ERR_ENCODING -3
#define DET_TOK_ERR_OVERFLOW -4
#define DET_TOK_ERR_IO      -5

/**
 * Get error message
 */
const char* det_tokenizer_strerror(int err);

#ifdef __cplusplus
}
#endif

#endif /* DET_TOKENIZER_H */
