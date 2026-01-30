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
#include <limits.h>

/* Token type constants (from GGUF/llama.cpp) */
#define TOKEN_TYPE_NORMAL       0
#define TOKEN_TYPE_UNKNOWN      1
#define TOKEN_TYPE_CONTROL      2
#define TOKEN_TYPE_USER_DEFINED 3
#define TOKEN_TYPE_UNUSED       4
#define TOKEN_TYPE_BYTE         5

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
 * MERGE LOOKUP TABLE (Binary Search Optimization)
 * ========================================================================== */

static inline uint64_t make_merge_key(int32_t left, int32_t right) {
    return ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
}

static int compare_merge_lookup(const void* a, const void* b) {
    const DetMergeLookup* ma = (const DetMergeLookup*)a;
    const DetMergeLookup* mb = (const DetMergeLookup*)b;
    if (ma->key < mb->key) return -1;
    if (ma->key > mb->key) return 1;
    /* Same key - use priority as tiebreaker (lower priority wins) */
    return ma->priority - mb->priority;
}

/**
 * Binary search for merge with given (left, right) pair
 * Returns merge result token ID, or -1 if no merge exists
 * Also returns priority via out_priority if not NULL
 */
static int32_t find_merge_binary(const DetTokenizer* tok, int32_t left, int32_t right,
                                  int* out_priority) {
    if (!tok->merge_lookup || tok->merge_lookup_count == 0) return -1;

    uint64_t key = make_merge_key(left, right);

    /* Binary search */
    int lo = 0, hi = tok->merge_lookup_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        uint64_t mid_key = tok->merge_lookup[mid].key;

        if (mid_key == key) {
            if (out_priority) *out_priority = tok->merge_lookup[mid].priority;
            return tok->merge_lookup[mid].result;
        } else if (mid_key < key) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

/* ==========================================================================
 * TRIE FOR ADDED TOKEN MATCHING
 * ========================================================================== */

static DetTrieNode* trie_create_node(void) {
    DetTrieNode* node = calloc(1, sizeof(DetTrieNode));
    if (node) node->token_id = -1;
    return node;
}

static void trie_free(DetTrieNode* node) {
    if (!node) return;
    for (int i = 0; i < 256; i++) {
        if (node->children[i]) {
            trie_free(node->children[i]);
        }
    }
    free(node);
}

static void trie_insert(DetTrieNode* root, const char* text, int32_t token_id) {
    if (!root || !text) return;

    DetTrieNode* node = root;
    const uint8_t* p = (const uint8_t*)text;

    while (*p) {
        if (!node->children[*p]) {
            node->children[*p] = trie_create_node();
            if (!node->children[*p]) return;  /* Allocation failed */
        }
        node = node->children[*p];
        p++;
    }
    node->token_id = token_id;
}

/**
 * Find longest matching added token at current position
 * Returns token ID and match length, or -1 if no match
 */
static int32_t trie_find_longest(const DetTrieNode* root, const char* text,
                                  const char* text_end, int32_t* out_len) {
    if (!root || !text) return -1;

    int32_t best_id = -1;
    int32_t best_len = 0;
    int32_t current_len = 0;

    const DetTrieNode* node = root;
    const uint8_t* p = (const uint8_t*)text;

    while ((const char*)p < text_end && node) {
        node = node->children[*p];
        if (!node) break;

        current_len++;
        if (node->token_id >= 0) {
            best_id = node->token_id;
            best_len = current_len;
        }
        p++;
    }

    if (out_len) *out_len = best_len;
    return best_id;
}

/* ==========================================================================
 * PRECOMPUTED BYTE TOKENS
 * ========================================================================== */

static void precompute_byte_tokens(DetTokenizer* tok) {
    tok->byte_token_ids = malloc(256 * sizeof(int32_t));
    if (!tok->byte_token_ids) return;

    char buf[16];
    for (int i = 0; i < 256; i++) {
        snprintf(buf, sizeof(buf), "<0x%02X>", i);
        tok->byte_token_ids[i] = hash_get(tok->token_to_id, buf, tok->special.unk_id);
    }
}

/* ==========================================================================
 * ADDED TOKEN HELPERS
 * ========================================================================== */

/* Check if token is an "added token" that should be matched before BPE */
static bool is_added_token(const DetVocabEntry* entry) {
    if (!entry || !entry->text) return false;

    /* Type 3 = USER_DEFINED (added tokens) */
    if (entry->type == TOKEN_TYPE_USER_DEFINED) return true;

    /* Also match <|...|> pattern (common for chat tokens) */
    const char* text = entry->text;
    size_t len = strlen(text);
    if (len >= 4 && text[0] == '<' && text[1] == '|' &&
        text[len-2] == '|' && text[len-1] == '>') {
        return true;
    }

    return false;
}

/* Comparison function for sorting added tokens by length (descending) */
static int compare_added_tokens(const void* a, const void* b) {
    const DetAddedToken* ta = (const DetAddedToken*)a;
    const DetAddedToken* tb = (const DetAddedToken*)b;
    return tb->len - ta->len;  /* Descending order */
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
        tok->merge_lookup = calloc(merge_count, sizeof(DetMergeLookup));
        tok->num_merges = (int32_t)merge_count;

        int32_t valid_merges = 0;

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

                    int32_t left_id = hash_get(tok->token_to_id, left, -1);
                    int32_t right_id = hash_get(tok->token_to_id, right, -1);

                    tok->merges[i].left = left_id;
                    tok->merges[i].right = right_id;

                    /* Result is the merged string */
                    char merged[512];
                    snprintf(merged, sizeof(merged), "%s%s", left, right);
                    int32_t result_id = hash_get(tok->token_to_id, merged, -1);
                    tok->merges[i].result = result_id;

                    /* Build lookup table entry (only for valid merges) */
                    if (left_id >= 0 && right_id >= 0 && result_id >= 0 && tok->merge_lookup) {
                        tok->merge_lookup[valid_merges].key = make_merge_key(left_id, right_id);
                        tok->merge_lookup[valid_merges].result = result_id;
                        tok->merge_lookup[valid_merges].priority = (int32_t)i;
                        valid_merges++;
                    }
                }
            }
        }
        free((void*)merges);

        /* Sort merge lookup table for binary search */
        if (tok->merge_lookup && valid_merges > 0) {
            tok->merge_lookup_count = valid_merges;
            qsort(tok->merge_lookup, valid_merges, sizeof(DetMergeLookup), compare_merge_lookup);
            fprintf(stderr, "Tokenizer: built merge lookup table with %d entries\n", valid_merges);
        }
    }

    /* Precompute byte fallback tokens */
    precompute_byte_tokens(tok);

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

    /* Build added tokens list for special token handling */
    int32_t added_count = 0;
    for (int32_t i = 0; i < tok->vocab_size; i++) {
        if (is_added_token(&tok->vocab[i])) {
            added_count++;
        }
    }

    if (added_count > 0) {
        tok->added_tokens = calloc(added_count, sizeof(DetAddedToken));
        tok->added_tokens_trie = trie_create_node();  /* Build trie for O(k) lookup */

        if (tok->added_tokens) {
            int32_t j = 0;
            for (int32_t i = 0; i < tok->vocab_size; i++) {
                if (is_added_token(&tok->vocab[i])) {
                    tok->added_tokens[j].text = tok->vocab[i].text;
                    tok->added_tokens[j].id = i;
                    tok->added_tokens[j].len = (int32_t)strlen(tok->vocab[i].text);

                    /* Insert into trie */
                    if (tok->added_tokens_trie) {
                        trie_insert(tok->added_tokens_trie, tok->vocab[i].text, i);
                    }
                    j++;
                }
            }
            tok->num_added_tokens = added_count;

            /* Sort by length descending for longest-match-first (fallback) */
            qsort(tok->added_tokens, added_count, sizeof(DetAddedToken),
                  compare_added_tokens);

            /* Debug: print added tokens count */
            fprintf(stderr, "Tokenizer: found %d added tokens for special token handling\n",
                    added_count);
        }
    }

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
    free(tok->merge_lookup);

    /* Free added tokens */
    free(tok->added_tokens);
    trie_free(tok->added_tokens_trie);

    /* Free byte token IDs */
    free(tok->byte_token_ids);

    /* Free hash table */
    hash_free(tok->token_to_id);

    free(tok);
}

/* ==========================================================================
 * ENCODING
 * ========================================================================== */

/* Forward declaration */
static int32_t tokenize_bpe_segment(const DetTokenizer* tok, const char* text,
                                     int32_t text_len, int32_t* tokens, int32_t max_tokens);

/**
 * Preprocess text for tiktoken-style tokenizers (phi-4, GPT-4)
 *
 * Tiktoken uses regex-based splitting that handles whitespace differently:
 * - Consecutive spaces are preserved
 * - Leading spaces on words become part of the word token
 *
 * Returns: allocated processed text (caller must free), or NULL on error
 */
static char* preprocess_tiktoken(const char* text, size_t* out_len) {
    if (!text) return NULL;

    size_t len = strlen(text);
    /* Allocate with extra space for potential modifications */
    char* processed = malloc(len + 1);
    if (!processed) return NULL;

    /* For now, just copy - the main preprocessing is done in BPE segment
     * by properly handling the space prefix (Ġ) conversion */
    memcpy(processed, text, len + 1);
    if (out_len) *out_len = len;
    return processed;
}

/**
 * Tokenize with added token handling
 *
 * Scans for special/added tokens first (like <|end|>, <|user|>), splits
 * text at their boundaries, then tokenizes each segment with BPE.
 */
static int32_t tokenize_with_added_tokens(const DetTokenizer* tok, const char* text,
                                           int32_t* tokens, int32_t max_tokens) {
    if (!tok || !text || !tokens) return DET_TOK_ERR_INVALID;
    if (!tok->added_tokens || tok->num_added_tokens == 0) {
        /* No added tokens - fall through to BPE */
        return tokenize_bpe_segment(tok, text, (int32_t)strlen(text), tokens, max_tokens);
    }

    int32_t num_tokens = 0;
    const char* p = text;
    const char* text_end = text + strlen(text);

    while (p < text_end && num_tokens < max_tokens) {
        /* Try to match an added token at current position using trie (O(k)) */
        int32_t matched_id = -1;
        int32_t matched_len = 0;

        if (tok->added_tokens_trie) {
            /* Fast O(k) trie lookup - finds longest match */
            matched_id = trie_find_longest(tok->added_tokens_trie, p, text_end, &matched_len);
        } else {
            /* Fallback: scan through added tokens (sorted by length) */
            for (int32_t i = 0; i < tok->num_added_tokens; i++) {
                const DetAddedToken* at = &tok->added_tokens[i];
                if (p + at->len <= text_end &&
                    memcmp(p, at->text, at->len) == 0) {
                    matched_id = at->id;
                    matched_len = at->len;
                    break;  /* First match is longest due to sorting */
                }
            }
        }

        if (matched_id >= 0) {
            /* Found an added token - emit it directly */
            if (num_tokens >= max_tokens) return DET_TOK_ERR_OVERFLOW;
            tokens[num_tokens++] = matched_id;
            p += matched_len;
        } else {
            /* Find the next added token (or end of string) */
            const char* next_added = text_end;

            /* Scan forward to find where next added token starts */
            for (const char* scan = p + 1; scan < text_end; scan++) {
                int32_t scan_len = 0;
                if (tok->added_tokens_trie) {
                    if (trie_find_longest(tok->added_tokens_trie, scan, text_end, &scan_len) >= 0) {
                        next_added = scan;
                        break;
                    }
                } else {
                    /* Fallback: check each added token */
                    for (int32_t i = 0; i < tok->num_added_tokens; i++) {
                        const DetAddedToken* at = &tok->added_tokens[i];
                        if (scan + at->len <= text_end &&
                            memcmp(scan, at->text, at->len) == 0) {
                            next_added = scan;
                            goto found_next;
                        }
                    }
                }
            }
            found_next:;  /* Empty statement after label for C99 compatibility */

            /* Tokenize the segment before the next added token */
            int32_t segment_len = (int32_t)(next_added - p);
            if (segment_len > 0) {
                int32_t seg_tokens = tokenize_bpe_segment(tok, p, segment_len,
                                                          tokens + num_tokens,
                                                          max_tokens - num_tokens);
                if (seg_tokens < 0) return seg_tokens;
                num_tokens += seg_tokens;
            }
            p = next_added;
        }
    }

    return num_tokens;
}

/* Simple greedy BPE tokenization (for a text segment) */
static int32_t tokenize_bpe_segment(const DetTokenizer* tok, const char* text,
                                     int32_t text_len, int32_t* tokens, int32_t max_tokens) {
    if (!tok || !text || !tokens) return DET_TOK_ERR_INVALID;
    if (text_len <= 0) return 0;

    int32_t num_tokens = 0;
    const char* p = text;
    const char* text_end = text + text_len;

    /* First pass: split into initial tokens (characters/bytes) */
    int32_t* work_tokens = malloc(text_len * 4 * sizeof(int32_t));
    if (!work_tokens) return DET_TOK_ERR_ALLOC;

    int32_t work_count = 0;

    /* Track if we're at start of text (for tiktoken-style leading space handling) */
    bool at_start = true;

    while (p < text_end) {
        /* Try to find longest matching token */
        int32_t best_id = -1;
        int best_len = 0;

        /* GPT-2/tiktoken style: spaces become 'Ġ' (U+0120) prefix on following word
         * Exception: leading space at very start may be handled differently */
        bool space_prefix = (*p == ' ');

        /* For tiktoken: try space as standalone token first if at text start */
        if (space_prefix && at_start) {
            /* Try matching just the space character */
            int32_t space_id = hash_get(tok->token_to_id, " ", -1);
            if (space_id >= 0) {
                work_tokens[work_count++] = space_id;
                p++;
                continue;
            }
        }

        const char* search_start = space_prefix ? (p + 1) : p;
        at_start = false;  /* No longer at start after first char */

        /* Try different lengths, longest first */
        size_t search_len = (size_t)(text_end - search_start);
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
            size_t p_len = (size_t)(text_end - p);
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
            /* Unknown character - use precomputed byte fallback (no snprintf!) */
            int32_t byte_id;
            if (tok->byte_token_ids) {
                byte_id = tok->byte_token_ids[(uint8_t)*p];
            } else {
                /* Fallback: compute on the fly */
                char byte_token[8];
                snprintf(byte_token, sizeof(byte_token), "<0x%02X>", (uint8_t)*p);
                byte_id = hash_get(tok->token_to_id, byte_token, tok->special.unk_id);
            }
            work_tokens[work_count++] = byte_id;
            p++;
        }
    }

    /* BPE merge pass - use binary search for O(n log m) instead of O(n * m) */
    if (tok->merge_lookup && tok->merge_lookup_count > 0) {
        bool changed = true;
        while (changed && work_count > 1) {
            changed = false;

            /* Find best merge (lowest priority = highest precedence) */
            int best_pos = -1;
            int best_priority = INT32_MAX;
            int32_t best_result = -1;

            for (int i = 0; i < work_count - 1; i++) {
                int priority = 0;
                int32_t result = find_merge_binary(tok, work_tokens[i], work_tokens[i + 1], &priority);
                if (result >= 0 && priority < best_priority) {
                    best_pos = i;
                    best_priority = priority;
                    best_result = result;
                }
            }

            /* Apply best merge */
            if (best_pos >= 0 && best_result >= 0) {
                work_tokens[best_pos] = best_result;
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

    /* Tokenize text (with added token handling) */
    int32_t text_tokens = tokenize_with_added_tokens(tok, text, tokens + pos, max_tokens - pos);
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

    int32_t count = tokenize_with_added_tokens(tok, text, temp, (int32_t)(strlen(text) * 4));

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
