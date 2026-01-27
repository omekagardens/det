/**
 * DET Tokenizer - Implementation
 * ==============================
 */

#include "det_tokenizer.h"
#include "det_gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ==========================================================================
 * HASH TABLE FOR TOKEN LOOKUP
 * ========================================================================== */

#define HASH_TABLE_SIZE 65536

typedef struct HashEntry {
    char* key;
    int32_t value;
    struct HashEntry* next;
} HashEntry;

typedef struct {
    HashEntry** buckets;
    int size;
} HashTable;

static uint32_t hash_string(const char* s) {
    uint32_t h = 5381;
    while (*s) {
        h = ((h << 5) + h) ^ (uint8_t)*s++;
    }
    return h;
}

static HashTable* hash_create(int size) {
    HashTable* ht = calloc(1, sizeof(HashTable));
    if (!ht) return NULL;
    ht->buckets = calloc(size, sizeof(HashEntry*));
    if (!ht->buckets) {
        free(ht);
        return NULL;
    }
    ht->size = size;
    return ht;
}

static void hash_free(HashTable* ht) {
    if (!ht) return;
    for (int i = 0; i < ht->size; i++) {
        HashEntry* e = ht->buckets[i];
        while (e) {
            HashEntry* next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }
    free(ht->buckets);
    free(ht);
}

static void hash_insert(HashTable* ht, const char* key, int32_t value) {
    uint32_t idx = hash_string(key) % ht->size;
    HashEntry* e = malloc(sizeof(HashEntry));
    if (!e) return;
    e->key = strdup(key);
    e->value = value;
    e->next = ht->buckets[idx];
    ht->buckets[idx] = e;
}

static int32_t hash_get(HashTable* ht, const char* key, int32_t default_val) {
    uint32_t idx = hash_string(key) % ht->size;
    HashEntry* e = ht->buckets[idx];
    while (e) {
        if (strcmp(e->key, key) == 0) {
            return e->value;
        }
        e = e->next;
    }
    return default_val;
}

/* ==========================================================================
 * TOKENIZER LIFECYCLE
 * ========================================================================== */

DetTokenizer* det_tokenizer_from_gguf(GgufContext* gguf) {
    if (!gguf) return NULL;

    DetTokenizer* tok = calloc(1, sizeof(DetTokenizer));
    if (!tok) return NULL;

    /* Get vocabulary from GGUF metadata */
    uint64_t vocab_count = 0;
    const char** tokens = gguf_get_string_array(gguf, "tokenizer.ggml.tokens", &vocab_count);

    if (!tokens || vocab_count == 0) {
        /* Try alternative key */
        tokens = gguf_get_string_array(gguf, "tokenizer.tokens", &vocab_count);
    }

    if (!tokens || vocab_count == 0) {
        free(tok);
        return NULL;
    }

    tok->vocab_size = (int32_t)vocab_count;
    tok->vocab = calloc(vocab_count, sizeof(DetVocabEntry));
    if (!tok->vocab) {
        free((void*)tokens);
        free(tok);
        return NULL;
    }

    /* Copy token strings */
    for (uint64_t i = 0; i < vocab_count; i++) {
        tok->vocab[i].text = strdup(tokens[i]);
        tok->vocab[i].score = 0.0f;
        tok->vocab[i].type = 0;
    }
    free((void*)tokens);

    /* Get scores if available */
    const GgufValue* scores_val = gguf_get_metadata(gguf, "tokenizer.ggml.scores");
    if (scores_val && scores_val->type == GGUF_TYPE_ARRAY &&
        scores_val->arr.elem_type == GGUF_TYPE_FLOAT32) {
        float* scores = (float*)scores_val->arr.data;
        for (uint64_t i = 0; i < vocab_count && i < scores_val->arr.count; i++) {
            tok->vocab[i].score = scores[i];
        }
    }

    /* Get token types if available */
    const GgufValue* types_val = gguf_get_metadata(gguf, "tokenizer.ggml.token_type");
    if (types_val && types_val->type == GGUF_TYPE_ARRAY &&
        types_val->arr.elem_type == GGUF_TYPE_INT32) {
        int32_t* types = (int32_t*)types_val->arr.data;
        for (uint64_t i = 0; i < vocab_count && i < types_val->arr.count; i++) {
            tok->vocab[i].type = types[i];
        }
    }

    /* Build token -> ID lookup table */
    tok->token_to_id = hash_create(HASH_TABLE_SIZE);
    if (tok->token_to_id) {
        for (int32_t i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i].text) {
                hash_insert(tok->token_to_id, tok->vocab[i].text, i);
            }
        }
    }

    /* Get special token IDs */
    tok->special.bos_id = gguf_get_u32(gguf, "tokenizer.ggml.bos_token_id", 1);
    tok->special.eos_id = gguf_get_u32(gguf, "tokenizer.ggml.eos_token_id", 2);
    tok->special.pad_id = gguf_get_u32(gguf, "tokenizer.ggml.padding_token_id", 0);
    tok->special.unk_id = gguf_get_u32(gguf, "tokenizer.ggml.unknown_token_id", 0);

    /* Find newline token */
    tok->special.nl_id = -1;
    for (int32_t i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i].text && strcmp(tok->vocab[i].text, "\n") == 0) {
            tok->special.nl_id = i;
            break;
        }
    }

    /* Get BPE merges if available */
    uint64_t merge_count = 0;
    const char** merges = gguf_get_string_array(gguf, "tokenizer.ggml.merges", &merge_count);
    if (merges && merge_count > 0) {
        tok->merges = calloc(merge_count, sizeof(DetBPEMerge));
        tok->num_merges = (int32_t)merge_count;

        /* Parse merge rules: "token1 token2" -> result */
        for (uint64_t i = 0; i < merge_count; i++) {
            const char* merge = merges[i];
            char left[256], right[256];

            /* Split on space */
            const char* space = strchr(merge, ' ');
            if (space) {
                size_t left_len = space - merge;
                if (left_len < sizeof(left)) {
                    strncpy(left, merge, left_len);
                    left[left_len] = '\0';
                    strncpy(right, space + 1, sizeof(right) - 1);
                    right[sizeof(right) - 1] = '\0';

                    tok->merges[i].left = hash_get(tok->token_to_id, left, -1);
                    tok->merges[i].right = hash_get(tok->token_to_id, right, -1);

                    /* Result is the merged string */
                    char merged[512];
                    snprintf(merged, sizeof(merged), "%s%s", left, right);
                    tok->merges[i].result = hash_get(tok->token_to_id, merged, -1);
                }
            }
        }
        free((void*)merges);
    }

    /* Determine tokenizer type */
    const char* model_type = gguf_get_string(gguf, "tokenizer.ggml.model");
    if (model_type) {
        if (strcmp(model_type, "gpt2") == 0 || strcmp(model_type, "llama") == 0) {
            tok->type = DET_TOKENIZER_BPE;
        } else if (strcmp(model_type, "unigram") == 0) {
            tok->type = DET_TOKENIZER_UNIGRAM;
        }
    } else {
        tok->type = DET_TOKENIZER_BPE;  /* Default */
    }

    /* Default settings */
    tok->add_bos = gguf_get_u32(gguf, "tokenizer.ggml.add_bos_token", 1) != 0;
    tok->add_eos = false;

    return tok;
}

DetTokenizer* det_tokenizer_load(const char* path) {
    /* Try loading as GGUF file first */
    GgufContext* gguf = gguf_open(path);
    if (gguf) {
        DetTokenizer* tok = det_tokenizer_from_gguf(gguf);
        gguf_close(gguf);
        return tok;
    }

    /* TODO: Support JSON vocab files */
    return NULL;
}

void det_tokenizer_free(DetTokenizer* tok) {
    if (!tok) return;

    /* Free vocabulary */
    if (tok->vocab) {
        for (int32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i].text);
        }
        free(tok->vocab);
    }

    /* Free merges */
    free(tok->merges);

    /* Free hash table */
    hash_free(tok->token_to_id);

    free(tok);
}

/* ==========================================================================
 * ENCODING
 * ========================================================================== */

/* Simple greedy BPE tokenization */
static int32_t tokenize_bpe(const DetTokenizer* tok, const char* text,
                            int32_t* tokens, int32_t max_tokens) {
    if (!tok || !text || !tokens) return DET_TOK_ERR_INVALID;

    int32_t num_tokens = 0;
    const char* p = text;

    /* First pass: split into initial tokens (characters/bytes) */
    int32_t* work_tokens = malloc(strlen(text) * 4 * sizeof(int32_t));
    if (!work_tokens) return DET_TOK_ERR_ALLOC;

    int32_t work_count = 0;

    while (*p) {
        /* Try to find longest matching token */
        int32_t best_id = -1;
        int best_len = 0;

        /* GPT-2 style: spaces are represented as 'Ġ' (U+0120 = 0xC4 0xA0) prefix */
        bool space_prefix = (*p == ' ');
        const char* search_start = space_prefix ? (p + 1) : p;

        /* Try different lengths, longest first */
        size_t search_len = strlen(search_start);
        for (int len = 16; len >= 1; len--) {
            /* Proper length check - don't try lengths beyond the string */
            if ((size_t)len > search_len) continue;

            char buf[32];
            int buf_pos = 0;

            /* Add Ġ prefix for space-prefixed tokens */
            if (space_prefix && len <= 14) {
                buf[buf_pos++] = 0xC4;  /* UTF-8 for U+0120 */
                buf[buf_pos++] = 0xA0;
            }

            if (buf_pos + len >= (int)sizeof(buf)) continue;
            strncpy(buf + buf_pos, search_start, len);
            buf[buf_pos + len] = '\0';

            int32_t id = hash_get(tok->token_to_id, buf, -1);
            if (id >= 0) {
                best_id = id;
                best_len = len + (space_prefix ? 1 : 0);  /* Include space in consumed length */
                break;
            }
        }

        /* If space-prefixed lookup failed, try without prefix */
        if (best_id < 0 && space_prefix) {
            size_t p_len = strlen(p);
            for (int len = 16; len >= 1; len--) {
                /* Proper length check */
                if ((size_t)len > p_len) continue;

                char buf[32];
                if (len >= (int)sizeof(buf)) continue;
                strncpy(buf, p, len);
                buf[len] = '\0';

                int32_t id = hash_get(tok->token_to_id, buf, -1);
                if (id >= 0) {
                    best_id = id;
                    best_len = len;
                    break;
                }
            }
        }

        if (best_id >= 0) {
            work_tokens[work_count++] = best_id;
            p += best_len;
        } else {
            /* Unknown character - use byte fallback or unknown token */
            char byte_token[8];
            snprintf(byte_token, sizeof(byte_token), "<0x%02X>", (uint8_t)*p);
            int32_t byte_id = hash_get(tok->token_to_id, byte_token, tok->special.unk_id);
            work_tokens[work_count++] = byte_id;
            p++;
        }
    }

    /* BPE merge pass */
    if (tok->merges && tok->num_merges > 0) {
        bool changed = true;
        while (changed && work_count > 1) {
            changed = false;

            /* Find best merge (lowest merge index = highest priority) */
            int best_pos = -1;
            int best_merge = tok->num_merges;  /* Start with worst priority */

            for (int i = 0; i < work_count - 1; i++) {
                int32_t left = work_tokens[i];
                int32_t right = work_tokens[i + 1];

                /* Find merge rule */
                for (int m = 0; m < tok->num_merges && m < best_merge; m++) {
                    if (tok->merges[m].left == left &&
                        tok->merges[m].right == right &&
                        tok->merges[m].result >= 0) {
                        best_pos = i;
                        best_merge = m;
                        break;
                    }
                }
            }

            /* Apply best merge */
            if (best_pos >= 0 && best_merge < tok->num_merges) {
                work_tokens[best_pos] = tok->merges[best_merge].result;
                /* Shift remaining tokens */
                for (int i = best_pos + 1; i < work_count - 1; i++) {
                    work_tokens[i] = work_tokens[i + 1];
                }
                work_count--;
                changed = true;
            }
        }
    }

    /* Copy to output */
    num_tokens = (work_count < max_tokens) ? work_count : max_tokens;
    memcpy(tokens, work_tokens, num_tokens * sizeof(int32_t));

    free(work_tokens);
    return num_tokens;
}

int32_t det_tokenize(const DetTokenizer* tok, const char* text,
                     int32_t* tokens, int32_t max_tokens) {
    return det_tokenize_ex(tok, text, tokens, max_tokens, tok->add_bos, tok->add_eos);
}

int32_t det_tokenize_ex(const DetTokenizer* tok, const char* text,
                        int32_t* tokens, int32_t max_tokens,
                        bool add_bos, bool add_eos) {
    if (!tok || !text || !tokens || max_tokens <= 0) {
        return DET_TOK_ERR_INVALID;
    }

    int32_t pos = 0;

    /* Add BOS if requested */
    if (add_bos && tok->special.bos_id >= 0) {
        if (pos >= max_tokens) return DET_TOK_ERR_OVERFLOW;
        tokens[pos++] = tok->special.bos_id;
    }

    /* Tokenize text */
    int32_t text_tokens = tokenize_bpe(tok, text, tokens + pos, max_tokens - pos);
    if (text_tokens < 0) return text_tokens;
    pos += text_tokens;

    /* Add EOS if requested */
    if (add_eos && tok->special.eos_id >= 0) {
        if (pos >= max_tokens) return DET_TOK_ERR_OVERFLOW;
        tokens[pos++] = tok->special.eos_id;
    }

    return pos;
}

int32_t det_token_count(const DetTokenizer* tok, const char* text) {
    if (!tok || !text) return 0;

    /* Temporary buffer for counting */
    int32_t* temp = malloc(strlen(text) * 4 * sizeof(int32_t));
    if (!temp) return 0;

    int32_t count = tokenize_bpe(tok, text, temp, (int32_t)(strlen(text) * 4));

    free(temp);
    return (count > 0) ? count : 0;
}

/* ==========================================================================
 * DECODING
 * ========================================================================== */

const char* det_token_to_text(const DetTokenizer* tok, int32_t token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }
    return tok->vocab[token_id].text ? tok->vocab[token_id].text : "";
}

/**
 * Decode BPE token text, converting Ġ (U+0120) back to space
 * Returns number of bytes written (not including null terminator)
 */
static int32_t decode_bpe_text(const char* src, char* dst, int32_t max_len);

/* Static buffer for decoded token text (for streaming) */
static char g_decoded_token_buf[256];

const char* det_token_to_text_decoded(const DetTokenizer* tok, int32_t token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }
    const char* raw = tok->vocab[token_id].text;
    if (!raw) return "";

    /* Decode BPE text (convert Ġ to space, etc.) */
    decode_bpe_text(raw, g_decoded_token_buf, sizeof(g_decoded_token_buf));
    return g_decoded_token_buf;
}

/**
 * Decode BPE token text, converting Ġ (U+0120) back to space
 * Returns number of bytes written (not including null terminator)
 */
static int32_t decode_bpe_text(const char* src, char* dst, int32_t max_len) {
    int32_t pos = 0;
    const uint8_t* s = (const uint8_t*)src;

    while (*s && pos < max_len - 1) {
        /* Check for Ġ (U+0120 = UTF-8: 0xC4 0xA0) */
        if (s[0] == 0xC4 && s[1] == 0xA0) {
            dst[pos++] = ' ';
            s += 2;
        }
        /* Check for Ċ (U+010A = UTF-8: 0xC4 0x8A) - newline in some tokenizers */
        else if (s[0] == 0xC4 && s[1] == 0x8A) {
            dst[pos++] = '\n';
            s += 2;
        }
        /* Check for ĉ (U+0109 = UTF-8: 0xC4 0x89) - tab in some tokenizers */
        else if (s[0] == 0xC4 && s[1] == 0x89) {
            dst[pos++] = '\t';
            s += 2;
        }
        else {
            dst[pos++] = *s++;
        }
    }

    dst[pos] = '\0';
    return pos;
}

int32_t det_detokenize(const DetTokenizer* tok,
                       const int32_t* tokens, int32_t num_tokens,
                       char* text, int32_t max_len) {
    if (!tok || !tokens || !text || max_len <= 0) {
        return DET_TOK_ERR_INVALID;
    }

    int32_t pos = 0;
    text[0] = '\0';

    for (int32_t i = 0; i < num_tokens; i++) {
        const char* token_text = det_token_to_text(tok, tokens[i]);

        /* Decode BPE text (convert Ġ to space, etc.) */
        char decoded[256];
        int32_t len = decode_bpe_text(token_text, decoded, sizeof(decoded));

        if (pos + len >= max_len) {
            break;
        }

        memcpy(text + pos, decoded, len);
        pos += len;
    }

    text[pos] = '\0';
    return pos;
}

int32_t det_detokenize_incremental(const DetTokenizer* tok,
                                    int32_t token_id, int32_t prev_token,
                                    char* text, int32_t max_len) {
    (void)prev_token;  /* Not used in simple implementation */

    if (!tok || !text || max_len <= 0) {
        return DET_TOK_ERR_INVALID;
    }

    const char* token_text = det_token_to_text(tok, token_id);

    /* Decode BPE text (convert Ġ to space, etc.) */
    return decode_bpe_text(token_text, text, max_len);
}

/* ==========================================================================
 * SPECIAL TOKENS
 * ========================================================================== */

int32_t det_get_special_token(const DetTokenizer* tok, const char* name) {
    if (!tok || !name) return -1;

    if (strcmp(name, "bos") == 0 || strcmp(name, "<s>") == 0) {
        return tok->special.bos_id;
    }
    if (strcmp(name, "eos") == 0 || strcmp(name, "</s>") == 0) {
        return tok->special.eos_id;
    }
    if (strcmp(name, "pad") == 0 || strcmp(name, "<pad>") == 0) {
        return tok->special.pad_id;
    }
    if (strcmp(name, "unk") == 0 || strcmp(name, "<unk>") == 0) {
        return tok->special.unk_id;
    }

    /* Try looking up in vocabulary */
    if (tok->token_to_id) {
        return hash_get(tok->token_to_id, name, -1);
    }

    return -1;
}

bool det_is_special_token(const DetTokenizer* tok, int32_t token_id) {
    if (!tok || token_id < 0) return false;

    return (token_id == tok->special.bos_id ||
            token_id == tok->special.eos_id ||
            token_id == tok->special.pad_id ||
            token_id == tok->special.unk_id);
}

/* ==========================================================================
 * UTILITIES
 * ========================================================================== */

void det_print_tokens(const DetTokenizer* tok,
                      const int32_t* tokens, int32_t num_tokens) {
    if (!tok || !tokens) return;

    printf("[");
    for (int32_t i = 0; i < num_tokens; i++) {
        if (i > 0) printf(", ");
        const char* text = det_token_to_text(tok, tokens[i]);
        printf("%d:'%s'", tokens[i], text);
    }
    printf("]\n");
}

/* ==========================================================================
 * ERROR HANDLING
 * ========================================================================== */

const char* det_tokenizer_strerror(int err) {
    switch (err) {
        case DET_TOK_OK:          return "OK";
        case DET_TOK_ERR_ALLOC:   return "Allocation failed";
        case DET_TOK_ERR_INVALID: return "Invalid argument";
        case DET_TOK_ERR_ENCODING: return "Encoding error";
        case DET_TOK_ERR_OVERFLOW: return "Buffer overflow";
        case DET_TOK_ERR_IO:      return "I/O error";
        default:                  return "Unknown error";
    }
}

/* ==========================================================================
 * NON-INLINE WRAPPERS FOR CTYPES BINDING
 * ========================================================================== */

int32_t det_bos_token_export(const DetTokenizer* tok) {
    return tok ? tok->special.bos_id : -1;
}

int32_t det_eos_token_export(const DetTokenizer* tok) {
    return tok ? tok->special.eos_id : -1;
}

int32_t det_vocab_size_export(const DetTokenizer* tok) {
    return tok ? tok->vocab_size : 0;
}
