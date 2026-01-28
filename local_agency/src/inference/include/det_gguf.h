/**
 * DET GGUF Model Loader - Phase 26.2
 * ===================================
 *
 * Loads GGUF model files (llama.cpp format) for DET inference.
 * Supports memory-mapped loading for efficient large model handling.
 *
 * GGUF Format Overview:
 * - Magic: "GGUF" (4 bytes)
 * - Version: uint32_t (2 or 3)
 * - Tensor count: uint64_t
 * - Metadata KV count: uint64_t
 * - Metadata key-value pairs
 * - Tensor infos (name, dims, type, offset)
 * - Tensor data (aligned)
 */

#ifndef DET_GGUF_H
#define DET_GGUF_H

#include "det_tensor.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * GGUF TYPES
 * ========================================================================== */

/** GGUF magic number */
#define GGUF_MAGIC 0x46554747  /* "GGUF" */

/** GGUF version */
#define GGUF_VERSION_2 2
#define GGUF_VERSION_3 3

/** GGUF metadata value types */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GgufType;

/** GGUF tensor types (matches llama.cpp) */
typedef enum {
    GGUF_TENSOR_F32     = 0,
    GGUF_TENSOR_F16     = 1,
    GGUF_TENSOR_Q4_0    = 2,
    GGUF_TENSOR_Q4_1    = 3,
    GGUF_TENSOR_Q5_0    = 6,
    GGUF_TENSOR_Q5_1    = 7,
    GGUF_TENSOR_Q8_0    = 8,
    GGUF_TENSOR_Q8_1    = 9,
    GGUF_TENSOR_Q2_K    = 10,
    GGUF_TENSOR_Q3_K    = 11,
    GGUF_TENSOR_Q4_K    = 12,
    GGUF_TENSOR_Q5_K    = 13,
    GGUF_TENSOR_Q6_K    = 14,
    GGUF_TENSOR_Q8_K    = 15,
    GGUF_TENSOR_IQ2_XXS = 16,
    GGUF_TENSOR_IQ2_XS  = 17,
    GGUF_TENSOR_IQ3_XXS = 18,
    GGUF_TENSOR_IQ1_S   = 19,
    GGUF_TENSOR_IQ4_NL  = 20,
    GGUF_TENSOR_IQ3_S   = 21,
    GGUF_TENSOR_IQ2_S   = 22,
    GGUF_TENSOR_IQ4_XS  = 23,
    GGUF_TENSOR_I8      = 24,
    GGUF_TENSOR_I16     = 25,
    GGUF_TENSOR_I32     = 26,
    GGUF_TENSOR_I64     = 27,
    GGUF_TENSOR_F64     = 28,
    GGUF_TENSOR_BF16    = 29,
} GgufTensorType;

/* ==========================================================================
 * GGUF STRUCTURES
 * ========================================================================== */

/** GGUF string (length-prefixed) */
typedef struct {
    uint64_t len;
    char* data;           /* NOT null-terminated in file */
} GgufString;

/** GGUF metadata value */
typedef struct GgufValue {
    GgufType type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        uint64_t  u64;
        int64_t   i64;
        float     f32;
        double    f64;
        bool      b;
        GgufString str;
        struct {
            GgufType elem_type;
            uint64_t count;
            void* data;
        } arr;
    };
} GgufValue;

/** GGUF metadata key-value pair */
typedef struct {
    GgufString key;
    GgufValue value;
} GgufKV;

/** GGUF tensor info */
typedef struct {
    GgufString name;
    uint32_t ndim;
    uint64_t shape[DET_MAX_DIMS];
    GgufTensorType type;
    uint64_t offset;      /* Offset from start of tensor data section */
} GgufTensorInfo;

/** GGUF file context */
typedef struct {
    /* File mapping */
    int fd;
    void* mapped_data;
    size_t file_size;

    /* Header info */
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;

    /* Metadata */
    GgufKV* metadata;

    /* Tensor info */
    GgufTensorInfo* tensors;

    /* Data section offset */
    size_t data_offset;

    /* Convenience pointers for common metadata */
    const char* model_arch;       /* e.g., "llama", "qwen2" */
    uint32_t n_vocab;
    uint32_t n_ctx;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_layer;
    uint32_t n_ff;
    float rope_freq_base;
    float rope_freq_scale;
} GgufContext;

/* ==========================================================================
 * GGUF LOADING
 * ========================================================================== */

/**
 * Open a GGUF file for reading
 *
 * Memory-maps the file for efficient access to large models.
 * Returns NULL on error.
 */
GgufContext* gguf_open(const char* path);

/**
 * Close GGUF file and free resources
 */
void gguf_close(GgufContext* ctx);

/**
 * Get tensor info by name
 *
 * Returns NULL if tensor not found.
 */
const GgufTensorInfo* gguf_get_tensor_info(const GgufContext* ctx, const char* name);

/**
 * Get tensor info by index
 *
 * Returns NULL if index out of range.
 */
const GgufTensorInfo* gguf_get_tensor_info_by_index(const GgufContext* ctx, uint64_t index);

/**
 * Create a DetTensor view of GGUF tensor data
 *
 * The tensor shares memory with the GGUF file mapping.
 * Returns NULL on error.
 */
DetTensor* gguf_get_tensor(GgufContext* ctx, const char* name);

/**
 * Create a DetTensor copy (dequantized to F32)
 *
 * Useful for tensors that need modification.
 * Returns NULL on error.
 */
DetTensor* gguf_get_tensor_f32(GgufContext* ctx, const char* name);

/**
 * Get raw Q8_0 tensor without dequantization
 *
 * Returns a memory-mapped view of the Q8_0 data.
 * Only works for Q8_0 tensors, returns NULL otherwise.
 * The tensor's dtype will be DET_DTYPE_Q8_0.
 *
 * Use this for QAM (Quantization-Aware Matmul) to keep weights
 * quantized in memory and dequantize on-the-fly during matmul.
 */
DetTensor* gguf_get_tensor_q8_0(GgufContext* ctx, const char* name);

/* ==========================================================================
 * METADATA ACCESS
 * ========================================================================== */

/**
 * Get metadata value by key
 *
 * Returns NULL if key not found.
 */
const GgufValue* gguf_get_metadata(const GgufContext* ctx, const char* key);

/**
 * Get string metadata
 *
 * Returns NULL if key not found or not a string.
 */
const char* gguf_get_string(const GgufContext* ctx, const char* key);

/**
 * Get uint32 metadata
 *
 * Returns default_val if key not found or wrong type.
 */
uint32_t gguf_get_u32(const GgufContext* ctx, const char* key, uint32_t default_val);

/**
 * Get float metadata
 *
 * Returns default_val if key not found or wrong type.
 */
float gguf_get_f32(const GgufContext* ctx, const char* key, float default_val);

/**
 * Get string array metadata
 *
 * Returns array of strings with count elements.
 * Caller must free the returned array (not the strings).
 */
const char** gguf_get_string_array(const GgufContext* ctx, const char* key, uint64_t* count);

/* ==========================================================================
 * TENSOR TYPE UTILITIES
 * ========================================================================== */

/**
 * Get element size for GGUF tensor type
 *
 * For quantized types, returns block size.
 */
size_t gguf_tensor_type_size(GgufTensorType type);

/**
 * Get block size for GGUF tensor type
 *
 * Returns 1 for non-quantized types.
 */
uint32_t gguf_tensor_block_size(GgufTensorType type);

/**
 * Convert GGUF tensor type to DET dtype
 */
DetDType gguf_type_to_det(GgufTensorType type);

/**
 * Get tensor type name
 */
const char* gguf_tensor_type_name(GgufTensorType type);

/* ==========================================================================
 * MODEL ARCHITECTURE
 * ========================================================================== */

/** Model architecture type */
typedef enum {
    DET_ARCH_UNKNOWN = 0,
    DET_ARCH_LLAMA,
    DET_ARCH_QWEN2,
    DET_ARCH_QWEN3,
    DET_ARCH_PHI3,
    DET_ARCH_GEMMA,
    DET_ARCH_MISTRAL,
    /* SSM/Mamba architectures */
    DET_ARCH_MAMBA,         /* Pure Mamba (all SSM layers) */
    DET_ARCH_MAMBA2,        /* Mamba-2 architecture */
    DET_ARCH_JAMBA,         /* Hybrid Jamba (mixed attention+SSM) */
    DET_ARCH_ZAMBA,         /* Zamba variant */
    /* Hybrid SSM-Transformer architectures */
    DET_ARCH_PHI4FLASH,     /* Phi-4-mini-flash-reasoning (SambaY) */
    DET_ARCH_SAMBAY,        /* Generic SambaY hybrid architecture */
} DetModelArch;

/**
 * Detect model architecture from GGUF metadata
 */
DetModelArch gguf_detect_arch(const GgufContext* ctx);

/**
 * Get architecture name
 */
const char* det_arch_name(DetModelArch arch);

/* ==========================================================================
 * ERROR HANDLING
 * ========================================================================== */

#define GGUF_OK            0
#define GGUF_ERR_IO       -1
#define GGUF_ERR_MAGIC    -2
#define GGUF_ERR_VERSION  -3
#define GGUF_ERR_PARSE    -4
#define GGUF_ERR_NOTFOUND -5
#define GGUF_ERR_TYPE     -6
#define GGUF_ERR_ALLOC    -7

/**
 * Get last GGUF error message
 */
const char* gguf_strerror(int err);

#ifdef __cplusplus
}
#endif

#endif /* DET_GGUF_H */
