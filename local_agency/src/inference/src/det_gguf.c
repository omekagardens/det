/**
 * DET GGUF Model Loader - Phase 26.2 Implementation
 * ==================================================
 */

#include "det_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* ==========================================================================
 * INTERNAL HELPERS
 * ========================================================================== */

/* Read position tracking */
typedef struct {
    const uint8_t* data;
    const uint8_t* end;
    const uint8_t* pos;
} ReadCtx;

static bool read_init(ReadCtx* r, const void* data, size_t size) {
    r->data = data;
    r->end = (const uint8_t*)data + size;
    r->pos = data;
    return true;
}

static bool read_check(ReadCtx* r, size_t n) {
    return (r->pos + n <= r->end);
}

static uint8_t read_u8(ReadCtx* r) {
    if (!read_check(r, 1)) return 0;
    return *r->pos++;
}

static uint16_t read_u16(ReadCtx* r) {
    if (!read_check(r, 2)) return 0;
    uint16_t v;
    memcpy(&v, r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(ReadCtx* r) {
    if (!read_check(r, 4)) return 0;
    uint32_t v;
    memcpy(&v, r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(ReadCtx* r) {
    if (!read_check(r, 8)) return 0;
    uint64_t v;
    memcpy(&v, r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(ReadCtx* r) {
    if (!read_check(r, 4)) return 0.0f;
    float v;
    memcpy(&v, r->pos, 4);
    r->pos += 4;
    return v;
}

static double read_f64(ReadCtx* r) {
    if (!read_check(r, 8)) return 0.0;
    double v;
    memcpy(&v, r->pos, 8);
    r->pos += 8;
    return v;
}

static bool read_string(ReadCtx* r, GgufString* out) {
    out->len = read_u64(r);
    if (!read_check(r, out->len)) {
        out->data = NULL;
        return false;
    }
    /* Allocate and copy (add null terminator for convenience) */
    out->data = malloc(out->len + 1);
    if (!out->data) return false;
    memcpy(out->data, r->pos, out->len);
    out->data[out->len] = '\0';
    r->pos += out->len;
    return true;
}

static void free_string(GgufString* s) {
    if (s->data) {
        free(s->data);
        s->data = NULL;
    }
    s->len = 0;
}

/* Forward declaration */
static bool read_value(ReadCtx* r, GgufValue* out);

static bool read_array(ReadCtx* r, GgufValue* out) {
    out->arr.elem_type = (GgufType)read_u32(r);
    out->arr.count = read_u64(r);

    if (out->arr.count == 0) {
        out->arr.data = NULL;
        return true;
    }

    /* Allocate array based on element type */
    size_t elem_size;
    switch (out->arr.elem_type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            elem_size = 1;
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            elem_size = 2;
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            elem_size = 4;
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            elem_size = 8;
            break;
        case GGUF_TYPE_STRING:
            elem_size = sizeof(GgufString);
            break;
        default:
            return false;
    }

    out->arr.data = calloc(out->arr.count, elem_size);
    if (!out->arr.data) return false;

    /* Read elements */
    for (uint64_t i = 0; i < out->arr.count; i++) {
        switch (out->arr.elem_type) {
            case GGUF_TYPE_UINT8:
                ((uint8_t*)out->arr.data)[i] = read_u8(r);
                break;
            case GGUF_TYPE_INT8:
                ((int8_t*)out->arr.data)[i] = (int8_t)read_u8(r);
                break;
            case GGUF_TYPE_UINT16:
                ((uint16_t*)out->arr.data)[i] = read_u16(r);
                break;
            case GGUF_TYPE_INT16:
                ((int16_t*)out->arr.data)[i] = (int16_t)read_u16(r);
                break;
            case GGUF_TYPE_UINT32:
                ((uint32_t*)out->arr.data)[i] = read_u32(r);
                break;
            case GGUF_TYPE_INT32:
                ((int32_t*)out->arr.data)[i] = (int32_t)read_u32(r);
                break;
            case GGUF_TYPE_UINT64:
                ((uint64_t*)out->arr.data)[i] = read_u64(r);
                break;
            case GGUF_TYPE_INT64:
                ((int64_t*)out->arr.data)[i] = (int64_t)read_u64(r);
                break;
            case GGUF_TYPE_FLOAT32:
                ((float*)out->arr.data)[i] = read_f32(r);
                break;
            case GGUF_TYPE_FLOAT64:
                ((double*)out->arr.data)[i] = read_f64(r);
                break;
            case GGUF_TYPE_BOOL:
                ((bool*)out->arr.data)[i] = read_u8(r) != 0;
                break;
            case GGUF_TYPE_STRING:
                if (!read_string(r, &((GgufString*)out->arr.data)[i])) {
                    /* Cleanup on failure */
                    for (uint64_t j = 0; j < i; j++) {
                        free_string(&((GgufString*)out->arr.data)[j]);
                    }
                    free(out->arr.data);
                    return false;
                }
                break;
            default:
                free(out->arr.data);
                return false;
        }
    }

    return true;
}

static bool read_value(ReadCtx* r, GgufValue* out) {
    out->type = (GgufType)read_u32(r);

    switch (out->type) {
        case GGUF_TYPE_UINT8:
            out->u8 = read_u8(r);
            break;
        case GGUF_TYPE_INT8:
            out->i8 = (int8_t)read_u8(r);
            break;
        case GGUF_TYPE_UINT16:
            out->u16 = read_u16(r);
            break;
        case GGUF_TYPE_INT16:
            out->i16 = (int16_t)read_u16(r);
            break;
        case GGUF_TYPE_UINT32:
            out->u32 = read_u32(r);
            break;
        case GGUF_TYPE_INT32:
            out->i32 = (int32_t)read_u32(r);
            break;
        case GGUF_TYPE_UINT64:
            out->u64 = read_u64(r);
            break;
        case GGUF_TYPE_INT64:
            out->i64 = (int64_t)read_u64(r);
            break;
        case GGUF_TYPE_FLOAT32:
            out->f32 = read_f32(r);
            break;
        case GGUF_TYPE_FLOAT64:
            out->f64 = read_f64(r);
            break;
        case GGUF_TYPE_BOOL:
            out->b = read_u8(r) != 0;
            break;
        case GGUF_TYPE_STRING:
            if (!read_string(r, &out->str)) return false;
            break;
        case GGUF_TYPE_ARRAY:
            if (!read_array(r, out)) return false;
            break;
        default:
            return false;
    }

    return true;
}

static void free_value(GgufValue* v) {
    if (v->type == GGUF_TYPE_STRING) {
        free_string(&v->str);
    } else if (v->type == GGUF_TYPE_ARRAY) {
        if (v->arr.elem_type == GGUF_TYPE_STRING && v->arr.data) {
            for (uint64_t i = 0; i < v->arr.count; i++) {
                free_string(&((GgufString*)v->arr.data)[i]);
            }
        }
        free(v->arr.data);
    }
}

/* ==========================================================================
 * GGUF LOADING
 * ========================================================================== */

GgufContext* gguf_open(const char* path) {
    GgufContext* ctx = calloc(1, sizeof(GgufContext));
    if (!ctx) return NULL;

    /* Open file */
    ctx->fd = open(path, O_RDONLY);
    if (ctx->fd < 0) {
        free(ctx);
        return NULL;
    }

    /* Get file size */
    struct stat st;
    if (fstat(ctx->fd, &st) < 0) {
        close(ctx->fd);
        free(ctx);
        return NULL;
    }
    ctx->file_size = st.st_size;

    /* Memory-map the file */
    ctx->mapped_data = mmap(NULL, ctx->file_size, PROT_READ, MAP_PRIVATE, ctx->fd, 0);
    if (ctx->mapped_data == MAP_FAILED) {
        close(ctx->fd);
        free(ctx);
        return NULL;
    }

    /* Parse header */
    ReadCtx r;
    read_init(&r, ctx->mapped_data, ctx->file_size);

    /* Check magic */
    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        gguf_close(ctx);
        return NULL;
    }

    /* Read version */
    ctx->version = read_u32(&r);
    if (ctx->version < GGUF_VERSION_2 || ctx->version > GGUF_VERSION_3) {
        gguf_close(ctx);
        return NULL;
    }

    /* Read counts */
    ctx->tensor_count = read_u64(&r);
    ctx->metadata_count = read_u64(&r);

    /* Allocate metadata array */
    if (ctx->metadata_count > 0) {
        ctx->metadata = calloc(ctx->metadata_count, sizeof(GgufKV));
        if (!ctx->metadata) {
            gguf_close(ctx);
            return NULL;
        }

        /* Read metadata key-value pairs */
        for (uint64_t i = 0; i < ctx->metadata_count; i++) {
            if (!read_string(&r, &ctx->metadata[i].key)) {
                gguf_close(ctx);
                return NULL;
            }
            if (!read_value(&r, &ctx->metadata[i].value)) {
                gguf_close(ctx);
                return NULL;
            }
        }
    }

    /* Allocate tensor info array */
    if (ctx->tensor_count > 0) {
        ctx->tensors = calloc(ctx->tensor_count, sizeof(GgufTensorInfo));
        if (!ctx->tensors) {
            gguf_close(ctx);
            return NULL;
        }

        /* Read tensor infos */
        for (uint64_t i = 0; i < ctx->tensor_count; i++) {
            if (!read_string(&r, &ctx->tensors[i].name)) {
                gguf_close(ctx);
                return NULL;
            }
            ctx->tensors[i].ndim = read_u32(&r);
            if (ctx->tensors[i].ndim > DET_MAX_DIMS) {
                gguf_close(ctx);
                return NULL;
            }
            for (uint32_t d = 0; d < ctx->tensors[i].ndim; d++) {
                ctx->tensors[i].shape[d] = read_u64(&r);
            }
            ctx->tensors[i].type = (GgufTensorType)read_u32(&r);
            ctx->tensors[i].offset = read_u64(&r);
        }
    }

    /* Calculate data section offset (aligned to 32 bytes) */
    size_t header_size = r.pos - r.data;
    ctx->data_offset = (header_size + 31) & ~31;

    /* Extract common model parameters */
    ctx->model_arch = gguf_get_string(ctx, "general.architecture");
    ctx->n_vocab = gguf_get_u32(ctx, "llama.vocab_size", 0);
    if (ctx->n_vocab == 0) ctx->n_vocab = gguf_get_u32(ctx, "qwen2.vocab_size", 0);

    /* Fallback: get vocab size from tokenizer.ggml.tokens array length */
    if (ctx->n_vocab == 0) {
        const GgufValue* tokens_val = gguf_get_metadata(ctx, "tokenizer.ggml.tokens");
        if (tokens_val && tokens_val->type == GGUF_TYPE_ARRAY) {
            ctx->n_vocab = (uint32_t)tokens_val->arr.count;
        }
    }

    ctx->n_ctx = gguf_get_u32(ctx, "llama.context_length", 2048);
    if (ctx->n_ctx == 2048) ctx->n_ctx = gguf_get_u32(ctx, "qwen2.context_length", 2048);

    ctx->n_embd = gguf_get_u32(ctx, "llama.embedding_length", 0);
    if (ctx->n_embd == 0) ctx->n_embd = gguf_get_u32(ctx, "qwen2.embedding_length", 0);

    ctx->n_head = gguf_get_u32(ctx, "llama.attention.head_count", 0);
    if (ctx->n_head == 0) ctx->n_head = gguf_get_u32(ctx, "qwen2.attention.head_count", 0);

    ctx->n_head_kv = gguf_get_u32(ctx, "llama.attention.head_count_kv", ctx->n_head);
    if (ctx->n_head_kv == ctx->n_head) ctx->n_head_kv = gguf_get_u32(ctx, "qwen2.attention.head_count_kv", ctx->n_head);

    ctx->n_layer = gguf_get_u32(ctx, "llama.block_count", 0);
    if (ctx->n_layer == 0) ctx->n_layer = gguf_get_u32(ctx, "qwen2.block_count", 0);

    ctx->n_ff = gguf_get_u32(ctx, "llama.feed_forward_length", 0);
    if (ctx->n_ff == 0) ctx->n_ff = gguf_get_u32(ctx, "qwen2.feed_forward_length", 0);

    ctx->rope_freq_base = gguf_get_f32(ctx, "llama.rope.freq_base", 10000.0f);
    if (ctx->rope_freq_base == 10000.0f) ctx->rope_freq_base = gguf_get_f32(ctx, "qwen2.rope.freq_base", 10000.0f);

    ctx->rope_freq_scale = gguf_get_f32(ctx, "llama.rope.scale_linear", 1.0f);

    return ctx;
}

void gguf_close(GgufContext* ctx) {
    if (!ctx) return;

    /* Free metadata */
    if (ctx->metadata) {
        for (uint64_t i = 0; i < ctx->metadata_count; i++) {
            free_string(&ctx->metadata[i].key);
            free_value(&ctx->metadata[i].value);
        }
        free(ctx->metadata);
    }

    /* Free tensor infos */
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->tensor_count; i++) {
            free_string(&ctx->tensors[i].name);
        }
        free(ctx->tensors);
    }

    /* Unmap and close file */
    if (ctx->mapped_data && ctx->mapped_data != MAP_FAILED) {
        munmap(ctx->mapped_data, ctx->file_size);
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }

    free(ctx);
}

const GgufTensorInfo* gguf_get_tensor_info(const GgufContext* ctx, const char* name) {
    if (!ctx || !name) return NULL;

    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].name.data &&
            strcmp(ctx->tensors[i].name.data, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

const GgufTensorInfo* gguf_get_tensor_info_by_index(const GgufContext* ctx, uint64_t index) {
    if (!ctx || index >= ctx->tensor_count) return NULL;
    return &ctx->tensors[index];
}

DetTensor* gguf_get_tensor(GgufContext* ctx, const char* name) {
    if (!ctx || !name) return NULL;

    const GgufTensorInfo* info = gguf_get_tensor_info(ctx, name);
    if (!info) return NULL;

    /* Calculate data pointer */
    void* data = (uint8_t*)ctx->mapped_data + ctx->data_offset + info->offset;

    /* Create shape array (convert uint64_t to int32_t) */
    int32_t shape[DET_MAX_DIMS];
    for (uint32_t i = 0; i < info->ndim; i++) {
        shape[i] = (int32_t)info->shape[i];
    }

    /* Create tensor view */
    DetTensor* t = det_tensor_from_ptr(data, info->ndim, shape, gguf_type_to_det(info->type));
    if (t) {
        t->storage = DET_STORAGE_MMAP;
    }

    return t;
}

DetTensor* gguf_get_tensor_f32(GgufContext* ctx, const char* name) {
    DetTensor* src = gguf_get_tensor(ctx, name);
    if (!src) return NULL;

    /* If already F32, just clone */
    if (src->dtype == DET_DTYPE_F32) {
        DetTensor* dst = det_tensor_clone(src);
        det_tensor_release(src);
        return dst;
    }

    /* Create F32 tensor with same shape */
    DetTensor* dst = det_tensor_create(src->ndim, src->shape, DET_DTYPE_F32);
    if (!dst) {
        det_tensor_release(src);
        return NULL;
    }

    /* Dequantize */
    if (det_dequantize(dst, src) != DET_TENSOR_OK) {
        det_tensor_release(dst);
        det_tensor_release(src);
        return NULL;
    }

    det_tensor_release(src);
    return dst;
}

DetTensor* gguf_get_tensor_q8_0(GgufContext* ctx, const char* name) {
    if (!ctx || !name) return NULL;

    const GgufTensorInfo* info = gguf_get_tensor_info(ctx, name);
    if (!info) return NULL;

    /* Only works for Q8_0 tensors */
    if (info->type != GGUF_TENSOR_Q8_0) {
        return NULL;
    }

    /* Calculate data pointer */
    void* data = (uint8_t*)ctx->mapped_data + ctx->data_offset + info->offset;

    /* Create shape array (convert uint64_t to int32_t) */
    int32_t shape[DET_MAX_DIMS];
    for (uint32_t i = 0; i < info->ndim; i++) {
        shape[i] = (int32_t)info->shape[i];
    }

    /* Create tensor view - keeps data as Q8_0, no dequantization */
    DetTensor* t = det_tensor_from_ptr(data, info->ndim, shape, DET_DTYPE_Q8_0);
    if (t) {
        t->storage = DET_STORAGE_MMAP;

        /* Calculate actual data size for Q8_0:
         * Each row of a 2D tensor [N, K] has K/32 blocks of 34 bytes */
        if (info->ndim == 2) {
            int32_t N = shape[0];
            int32_t K = shape[1];
            int32_t blocks_per_row = (K + 31) / 32;
            t->data_size = N * blocks_per_row * 34;
        } else if (info->ndim == 1) {
            int32_t K = shape[0];
            int32_t num_blocks = (K + 31) / 32;
            t->data_size = num_blocks * 34;
        }
    }

    return t;
}

/* ==========================================================================
 * METADATA ACCESS
 * ========================================================================== */

const GgufValue* gguf_get_metadata(const GgufContext* ctx, const char* key) {
    if (!ctx || !key) return NULL;

    for (uint64_t i = 0; i < ctx->metadata_count; i++) {
        if (ctx->metadata[i].key.data &&
            strcmp(ctx->metadata[i].key.data, key) == 0) {
            return &ctx->metadata[i].value;
        }
    }
    return NULL;
}

const char* gguf_get_string(const GgufContext* ctx, const char* key) {
    const GgufValue* v = gguf_get_metadata(ctx, key);
    if (!v || v->type != GGUF_TYPE_STRING) return NULL;
    return v->str.data;
}

uint32_t gguf_get_u32(const GgufContext* ctx, const char* key, uint32_t default_val) {
    const GgufValue* v = gguf_get_metadata(ctx, key);
    if (!v) return default_val;

    switch (v->type) {
        case GGUF_TYPE_UINT8:  return v->u8;
        case GGUF_TYPE_INT8:   return (uint32_t)v->i8;
        case GGUF_TYPE_UINT16: return v->u16;
        case GGUF_TYPE_INT16:  return (uint32_t)v->i16;
        case GGUF_TYPE_UINT32: return v->u32;
        case GGUF_TYPE_INT32:  return (uint32_t)v->i32;
        case GGUF_TYPE_UINT64: return (uint32_t)v->u64;
        case GGUF_TYPE_INT64:  return (uint32_t)v->i64;
        default: return default_val;
    }
}

float gguf_get_f32(const GgufContext* ctx, const char* key, float default_val) {
    const GgufValue* v = gguf_get_metadata(ctx, key);
    if (!v) return default_val;

    switch (v->type) {
        case GGUF_TYPE_FLOAT32: return v->f32;
        case GGUF_TYPE_FLOAT64: return (float)v->f64;
        case GGUF_TYPE_UINT32:  return (float)v->u32;
        case GGUF_TYPE_INT32:   return (float)v->i32;
        default: return default_val;
    }
}

const char** gguf_get_string_array(const GgufContext* ctx, const char* key, uint64_t* count) {
    const GgufValue* v = gguf_get_metadata(ctx, key);
    if (!v || v->type != GGUF_TYPE_ARRAY || v->arr.elem_type != GGUF_TYPE_STRING) {
        if (count) *count = 0;
        return NULL;
    }

    if (count) *count = v->arr.count;

    /* Create array of string pointers */
    const char** arr = malloc(v->arr.count * sizeof(char*));
    if (!arr) return NULL;

    GgufString* strs = (GgufString*)v->arr.data;
    for (uint64_t i = 0; i < v->arr.count; i++) {
        arr[i] = strs[i].data;
    }

    return arr;
}

/* ==========================================================================
 * TENSOR TYPE UTILITIES
 * ========================================================================== */

/* Block size for quantized types */
static const uint32_t gguf_block_sizes[] = {
    [GGUF_TENSOR_F32]     = 1,
    [GGUF_TENSOR_F16]     = 1,
    [GGUF_TENSOR_Q4_0]    = 32,
    [GGUF_TENSOR_Q4_1]    = 32,
    [GGUF_TENSOR_Q5_0]    = 32,
    [GGUF_TENSOR_Q5_1]    = 32,
    [GGUF_TENSOR_Q8_0]    = 32,
    [GGUF_TENSOR_Q8_1]    = 32,
    [GGUF_TENSOR_Q2_K]    = 256,
    [GGUF_TENSOR_Q3_K]    = 256,
    [GGUF_TENSOR_Q4_K]    = 256,
    [GGUF_TENSOR_Q5_K]    = 256,
    [GGUF_TENSOR_Q6_K]    = 256,
    [GGUF_TENSOR_Q8_K]    = 256,
    [GGUF_TENSOR_BF16]    = 1,
    [GGUF_TENSOR_I8]      = 1,
    [GGUF_TENSOR_I16]     = 1,
    [GGUF_TENSOR_I32]     = 1,
    [GGUF_TENSOR_I64]     = 1,
    [GGUF_TENSOR_F64]     = 1,
};

/* Bytes per block for quantized types */
static const size_t gguf_type_sizes[] = {
    [GGUF_TENSOR_F32]     = 4,
    [GGUF_TENSOR_F16]     = 2,
    [GGUF_TENSOR_Q4_0]    = 2 + 16,   /* F16 scale + 32 4-bit values */
    [GGUF_TENSOR_Q4_1]    = 4 + 16,   /* scale + min + 32 4-bit values */
    [GGUF_TENSOR_Q5_0]    = 2 + 4 + 16,
    [GGUF_TENSOR_Q5_1]    = 4 + 4 + 16,
    [GGUF_TENSOR_Q8_0]    = 2 + 32,   /* F16 scale (2 bytes) + 32 int8 values = 34 bytes */
    [GGUF_TENSOR_Q8_1]    = 8 + 32,   /* scale + sum + 32 8-bit values */
    [GGUF_TENSOR_Q2_K]    = 256/4 + 256/16 + 2 + 2,
    [GGUF_TENSOR_Q3_K]    = 256/8*3 + 256/16 + 2,
    [GGUF_TENSOR_Q4_K]    = 2 + 2 + 12 + 256/2,
    [GGUF_TENSOR_Q5_K]    = 2 + 2 + 12 + 256/8 + 256/2,
    [GGUF_TENSOR_Q6_K]    = 256/2 + 256/4 + 256/16 + 2,
    [GGUF_TENSOR_Q8_K]    = 4 + 256 + 2*256/16,
    [GGUF_TENSOR_BF16]    = 2,
    [GGUF_TENSOR_I8]      = 1,
    [GGUF_TENSOR_I16]     = 2,
    [GGUF_TENSOR_I32]     = 4,
    [GGUF_TENSOR_I64]     = 8,
    [GGUF_TENSOR_F64]     = 8,
};

size_t gguf_tensor_type_size(GgufTensorType type) {
    if (type < 0 || type > GGUF_TENSOR_BF16) return 0;
    return gguf_type_sizes[type];
}

uint32_t gguf_tensor_block_size(GgufTensorType type) {
    if (type < 0 || type > GGUF_TENSOR_BF16) return 1;
    return gguf_block_sizes[type];
}

DetDType gguf_type_to_det(GgufTensorType type) {
    switch (type) {
        case GGUF_TENSOR_F32:  return DET_DTYPE_F32;
        case GGUF_TENSOR_F16:  return DET_DTYPE_F16;
        case GGUF_TENSOR_BF16: return DET_DTYPE_BF16;
        case GGUF_TENSOR_Q8_0:
        case GGUF_TENSOR_Q8_1:
        case GGUF_TENSOR_Q8_K: return DET_DTYPE_Q8_0;
        case GGUF_TENSOR_Q4_0:
        case GGUF_TENSOR_Q4_1:
        case GGUF_TENSOR_Q4_K: return DET_DTYPE_Q4_K_M;
        case GGUF_TENSOR_I32:  return DET_DTYPE_I32;
        case GGUF_TENSOR_I16:  return DET_DTYPE_I16;
        default:               return DET_DTYPE_F32;  /* Fallback */
    }
}

static const char* gguf_type_names[] = {
    [GGUF_TENSOR_F32]     = "F32",
    [GGUF_TENSOR_F16]     = "F16",
    [GGUF_TENSOR_Q4_0]    = "Q4_0",
    [GGUF_TENSOR_Q4_1]    = "Q4_1",
    [GGUF_TENSOR_Q5_0]    = "Q5_0",
    [GGUF_TENSOR_Q5_1]    = "Q5_1",
    [GGUF_TENSOR_Q8_0]    = "Q8_0",
    [GGUF_TENSOR_Q8_1]    = "Q8_1",
    [GGUF_TENSOR_Q2_K]    = "Q2_K",
    [GGUF_TENSOR_Q3_K]    = "Q3_K",
    [GGUF_TENSOR_Q4_K]    = "Q4_K",
    [GGUF_TENSOR_Q5_K]    = "Q5_K",
    [GGUF_TENSOR_Q6_K]    = "Q6_K",
    [GGUF_TENSOR_Q8_K]    = "Q8_K",
    [GGUF_TENSOR_BF16]    = "BF16",
    [GGUF_TENSOR_I8]      = "I8",
    [GGUF_TENSOR_I16]     = "I16",
    [GGUF_TENSOR_I32]     = "I32",
    [GGUF_TENSOR_I64]     = "I64",
    [GGUF_TENSOR_F64]     = "F64",
};

const char* gguf_tensor_type_name(GgufTensorType type) {
    if (type < 0 || type > GGUF_TENSOR_BF16) return "unknown";
    return gguf_type_names[type];
}

/* ==========================================================================
 * MODEL ARCHITECTURE
 * ========================================================================== */

DetModelArch gguf_detect_arch(const GgufContext* ctx) {
    if (!ctx || !ctx->model_arch) return DET_ARCH_UNKNOWN;

    if (strcmp(ctx->model_arch, "llama") == 0) return DET_ARCH_LLAMA;
    if (strcmp(ctx->model_arch, "qwen2") == 0) return DET_ARCH_QWEN2;
    if (strcmp(ctx->model_arch, "phi3") == 0) return DET_ARCH_PHI3;
    if (strcmp(ctx->model_arch, "gemma") == 0) return DET_ARCH_GEMMA;
    if (strcmp(ctx->model_arch, "mistral") == 0) return DET_ARCH_MISTRAL;

    return DET_ARCH_UNKNOWN;
}

const char* det_arch_name(DetModelArch arch) {
    switch (arch) {
        case DET_ARCH_LLAMA:   return "llama";
        case DET_ARCH_QWEN2:   return "qwen2";
        case DET_ARCH_PHI3:    return "phi3";
        case DET_ARCH_GEMMA:   return "gemma";
        case DET_ARCH_MISTRAL: return "mistral";
        default:               return "unknown";
    }
}

/* ==========================================================================
 * ERROR HANDLING
 * ========================================================================== */

static const char* gguf_error_strings[] = {
    "OK",
    "I/O error",
    "Invalid magic number (not a GGUF file)",
    "Unsupported GGUF version",
    "Parse error",
    "Tensor not found",
    "Type mismatch",
    "Allocation failed",
};

const char* gguf_strerror(int err) {
    if (err >= 0) return gguf_error_strings[0];
    int idx = -err;
    if (idx >= (int)(sizeof(gguf_error_strings)/sizeof(gguf_error_strings[0]))) {
        return "Unknown error";
    }
    return gguf_error_strings[idx];
}
