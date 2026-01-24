/**
 * EIS Substrate v2 - Effect Table
 * ================================
 *
 * Verified effects for the COMMIT phase.
 * Effects are data attached to proposals, not opcodes.
 */

#ifndef EFFECT_TABLE_H
#define EFFECT_TABLE_H

#include "substrate_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * EFFECT IDs
 * ========================================================================== */

typedef enum {
    /* No effect */
    EFFECT_NONE = 0x00,

    /* Resource transfer (antisymmetric) */
    EFFECT_XFER_F = 0x01,       /* Args: src_node, dst_node, amount */
                                /* F[src] -= amount; F[dst] += amount */

    /* Symmetric flux (bond-based) */
    EFFECT_DIFFUSE = 0x02,      /* Args: bond_id, delta */
                                /* F[i] -= delta; F[j] += delta */

    /* Direct field sets */
    EFFECT_SET_F = 0x03,        /* Args: node, value */
    EFFECT_ADD_F = 0x04,        /* Args: node, delta */
    EFFECT_SET_Q = 0x05,        /* Args: node, value */
    EFFECT_ADD_Q = 0x06,        /* Args: node, delta */
    EFFECT_SET_A = 0x07,        /* Args: node, value (clamped to [0,1]) */
    EFFECT_SET_SIGMA = 0x08,    /* Args: node, value */
    EFFECT_SET_P = 0x09,        /* Args: node, value */
    EFFECT_SET_THETA = 0x0A,    /* Args: node, cos, sin (unit vector) */
    EFFECT_INC_K = 0x0B,        /* Args: node (k += 1) */
    EFFECT_INC_TAU = 0x0C,      /* Args: node, delta (tau += delta) */

    /* Bond field sets */
    EFFECT_SET_C = 0x10,        /* Args: bond, value (clamped to [0,1]) */
    EFFECT_ADD_C = 0x11,        /* Args: bond, delta (clamped result) */
    EFFECT_SET_PI = 0x12,       /* Args: bond, value */
    EFFECT_ADD_PI = 0x13,       /* Args: bond, delta */
    EFFECT_SET_BOND_SIGMA = 0x14, /* Args: bond, value */

    /* Token emission */
    EFFECT_EMIT_TOK = 0x20,     /* Args: tok_slot, value */

    /* Boundary output */
    EFFECT_EMIT_BYTE = 0x30,    /* Args: buf_id, byte */
    EFFECT_EMIT_FLOAT = 0x31,   /* Args: buf_id, float_bits */

    /* System */
    EFFECT_SET_SEED = 0xF0,     /* Args: seed_value */
} EffectId;

/* ==========================================================================
 * EFFECT ARGUMENT PACKING
 * ========================================================================== */

/* Pack float into uint32 for effect args */
static inline uint32_t effect_pack_float(float f) {
    union { float f; uint32_t u; } conv;
    conv.f = f;
    return conv.u;
}

/* Unpack float from uint32 */
static inline float effect_unpack_float(uint32_t u) {
    union { float f; uint32_t u; } conv;
    conv.u = u;
    return conv.f;
}

/* ==========================================================================
 * EFFECT DESCRIPTORS
 * ========================================================================== */

typedef struct {
    EffectId id;
    const char* name;
    uint8_t arg_count;
    bool antisymmetric;     /* True if conservation-critical */
    bool node_effect;       /* True if affects node state */
    bool bond_effect;       /* True if affects bond state */
} EffectDescriptor;

/* Effect table (defined in effect_table.c) */
extern const EffectDescriptor EFFECT_TABLE[];
extern const size_t EFFECT_TABLE_SIZE;

/* Get effect descriptor */
const EffectDescriptor* effect_get_descriptor(EffectId id);

/* Get effect name */
const char* effect_get_name(EffectId id);

/* ==========================================================================
 * EFFECT APPLICATION (Forward declarations)
 * ========================================================================== */

/* Forward declare VM type */
struct SubstrateVM;

/**
 * Apply an effect to the trace store.
 * Called during COMMIT phase only.
 *
 * @param vm The substrate VM
 * @param effect_id The effect to apply
 * @param args Packed arguments (count depends on effect)
 * @return TOK_OK on success, error token otherwise
 */
uint32_t effect_apply(struct SubstrateVM* vm, EffectId effect_id, const uint32_t* args);

/**
 * Validate effect arguments before application.
 *
 * @param vm The substrate VM
 * @param effect_id The effect to validate
 * @param args Packed arguments
 * @return true if valid
 */
bool effect_validate(struct SubstrateVM* vm, EffectId effect_id, const uint32_t* args);

#ifdef __cplusplus
}
#endif

#endif /* EFFECT_TABLE_H */
