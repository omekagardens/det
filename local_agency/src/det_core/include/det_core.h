/**
 * DET Local Agency - C Kernel Header
 * ==================================
 *
 * Deep Existence Theory core implementation for AI mind substrate.
 *
 * Architecture: Dual-Process (P-layer + A-layer) with cluster-centric Self.
 * Key insight: "You are the CLUSTER" - agency lives in coherence field.
 *
 * Reference: FEASIBILITY_PLAN.md, explorations/*.md
 */

#ifndef DET_CORE_H
#define DET_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * CONFIGURATION CONSTANTS
 * ========================================================================== */

#define DET_MAX_NODES       4096    /* Total node capacity (P + A + dormant) */
#define DET_MAX_BONDS       16384   /* Sparse bond capacity */
#define DET_MAX_PORTS       64      /* LLM interface port nodes */
#define DET_MAX_DOMAINS     16      /* Memory domain types */

/* Layer sizes (Phase 1 defaults, tunable) */
#define DET_P_LAYER_SIZE    16      /* Presence layer nodes */
#define DET_A_LAYER_SIZE    256     /* Automaticity layer nodes */
#define DET_DORMANT_SIZE    3760    /* Dormant pool nodes (leaves room for ports) */

/* ==========================================================================
 * ENUMERATIONS
 * ========================================================================== */

/** Node layer classification */
typedef enum {
    DET_LAYER_DORMANT = 0,  /* Inactive, in dormant pool */
    DET_LAYER_A = 1,        /* Automaticity layer (System 1) */
    DET_LAYER_P = 2,        /* Presence layer (System 2) */
    DET_LAYER_PORT = 3      /* LLM interface port node */
} DETLayer;

/** Gatekeeper decision outcomes */
typedef enum {
    DET_DECISION_PROCEED = 0,   /* Execute the requested action */
    DET_DECISION_RETRY = 1,     /* Ask LLM to reformulate */
    DET_DECISION_STOP = 2,      /* Gracefully decline */
    DET_DECISION_ESCALATE = 3   /* Need external LLM assistance */
} DETDecision;

/** Emotional state interpretation */
typedef enum {
    DET_EMOTION_NEUTRAL = 0,
    DET_EMOTION_FLOW = 1,
    DET_EMOTION_CONTENTMENT = 2,
    DET_EMOTION_STRESS = 3,
    DET_EMOTION_OVERWHELM = 4,
    DET_EMOTION_APATHY = 5,
    DET_EMOTION_BOREDOM = 6,
    DET_EMOTION_PEACE = 7
} DETEmotion;

/* ==========================================================================
 * CORE DATA STRUCTURES
 * ========================================================================== */

/** DET physics parameters (from det_unified_params.py) */
typedef struct {
    float tau_base;         /* 0.02 - Time/screening scale */
    float sigma_base;       /* 0.12 - Charging rate */
    float lambda_base;      /* 0.008 - Decay rate */
    float mu_base;          /* 2.0 - Mobility scale */
    float kappa_base;       /* 5.0 - Coupling scale */
    float C_0;              /* 0.15 - Coherence scale */
    float lambda_a;         /* 30.0 - Agency ceiling coupling */
    float phi_L;            /* 0.5 - Angular/momentum ratio */
    float pi_max;           /* 3.0 - Momentum cap */

    /* Layer-specific bond parameters */
    float alpha_AA, lambda_AA, slip_AA;   /* A↔A: fast, plastic */
    float alpha_PP, lambda_PP, slip_PP;   /* P↔P: slow, stable */
    float alpha_PA, lambda_PA, slip_PA;   /* P↔A: medium, phase-sensitive */
} DETParams;

/** Per-node affect state (3-axis emotional feedback) */
typedef struct {
    float v;                /* Valence [-1, 1]: good/bad */
    float r;                /* Arousal [0, 1]: activation level */
    float b;                /* Bondedness [0, 1]: attachment */

    /* EMA traces for underlying signals */
    float ema_throughput;
    float ema_surprise;
    float ema_fragmentation;
    float ema_debt;
    float ema_bonding;
} DETAffect;

/** Per-node dual EMA state (temporal dynamics) */
typedef struct {
    float flux_short;       /* Short EMA of flux (α=0.3) */
    float flux_long;        /* Long EMA of flux (α=0.05) */
    float coherence_short;
    float coherence_long;
    float debt_short;
    float debt_long;
} DETDualEMA;

/** Per-node cadence state (temporal windowing) */
typedef struct {
    uint32_t last_active_tick;
    uint32_t quiet_ticks;
    uint16_t membrane_window;
    uint16_t window_counter;
} DETCadence;

/** Per-node state */
typedef struct {
    /* Core DET physics */
    float F;                /* Resource */
    float q;                /* Structural debt */
    float a;                /* Agency (inviolable intrinsic) */
    float theta;            /* Phase */
    float sigma;            /* Processing rate */
    float P;                /* Presence (computed) */
    float tau;              /* Proper time (accumulated) */

    /* Phase 4: Extended dynamics */
    float L;                /* Angular momentum (spin) */
    float dtheta_dt;        /* Phase velocity */
    float grace_buffer;     /* Grace injection buffer (boundary recovery) */

    /* Classification */
    DETLayer layer;
    uint8_t domain;         /* Memory domain (for A-layer) */
    bool active;            /* Is this node active? */

    /* Extensions */
    DETAffect affect;
    DETDualEMA ema;
    DETCadence cadence;

    /* Novelty/stability for membrane dynamics */
    float novelty_score;
    bool escalation_pending;
} DETNode;

/** Per-bond state */
typedef struct {
    uint16_t i, j;          /* Connected node indices */
    float C;                /* Coherence */
    float pi;               /* Momentum */
    float sigma;            /* Bond conductivity */

    /* EMA traces */
    float flux_ema;
    float phase_align_ema;
    float stability_ema;

    /* Decay modulation */
    float lambda_decay;
    float lambda_slip;

    /* Flags */
    bool is_temporary;      /* Marked for cleanup after request */
    bool is_cross_layer;    /* P↔A bond */
} DETBond;

/** Port node for LLM interface */
typedef struct {
    uint16_t node_id;       /* Index in node array */
    uint8_t port_type;      /* Intent, domain, risk, etc. */
    char name[32];          /* "intent_answer", "domain_math", etc. */
    uint8_t target_domain;  /* Which domain cluster this feeds */
} DETPort;

/** Memory domain linkage */
typedef struct {
    char name[32];          /* "math", "language", etc. */
    float coherence_to_core;
    float activation_level;
    void* model_handle;     /* Pointer to Python model wrapper */
} DETDomain;

/** Self-cluster identification result */
typedef struct {
    uint16_t* nodes;        /* Array of node indices in Self */
    uint32_t num_nodes;
    float cluster_agency;   /* Σ C_ij × |J_ij| */
    float continuity;       /* Jaccard with previous Self */

    /* Aggregate affect */
    float valence;
    float arousal;
    float bondedness;
} DETSelf;

/** Main DET core state */
typedef struct {
    DETParams params;

    /* Node storage */
    DETNode nodes[DET_MAX_NODES];
    uint32_t num_nodes;
    uint32_t num_active;

    /* Bond storage (sparse) */
    DETBond bonds[DET_MAX_BONDS];
    uint32_t num_bonds;

    /* Port nodes for LLM interface */
    DETPort ports[DET_MAX_PORTS];
    uint32_t num_ports;

    /* Memory domains */
    DETDomain domains[DET_MAX_DOMAINS];
    uint32_t num_domains;

    /* Self-cluster (current) */
    DETSelf self;
    uint16_t self_nodes_storage[DET_MAX_NODES]; /* Backing storage */

    /* Aggregate metrics (updated each tick) */
    float aggregate_presence;
    float aggregate_coherence;
    float aggregate_resource;
    float aggregate_debt;

    /* Tick counter */
    uint64_t tick;

    /* Emotional state interpretation */
    DETEmotion emotion;
} DETCore;

/* ==========================================================================
 * API FUNCTIONS
 * ========================================================================== */

/* ----- Lifecycle ----- */

/** Create and initialize a DET core with default parameters */
DETCore* det_core_create(void);

/** Create with custom parameters */
DETCore* det_core_create_with_params(const DETParams* params);

/** Destroy and free a DET core */
void det_core_destroy(DETCore* core);

/** Reset core to initial state (keep params) */
void det_core_reset(DETCore* core);

/* ----- Simulation ----- */

/** Advance the DET core by one timestep */
void det_core_step(DETCore* core, float dt);

/** Update presence for all nodes */
void det_core_update_presence(DETCore* core);

/** Update coherence for all bonds */
void det_core_update_coherence(DETCore* core, float dt);

/** Update agency ceilings */
void det_core_update_agency(DETCore* core, float dt);

/** Update affect (emotional feedback) */
void det_core_update_affect(DETCore* core, float dt);

/** Identify the Self cluster */
void det_core_identify_self(DETCore* core);

/* ----- Phase 4: Extended Dynamics ----- */

/** Update momentum (π) for all bonds */
void det_core_update_momentum(DETCore* core, float dt);

/** Update angular momentum (L) and phase velocity */
void det_core_update_angular_momentum(DETCore* core, float dt);

/** Update structural debt accumulation */
void det_core_update_debt(DETCore* core, float dt);

/** Inject grace to a node (boundary recovery) */
void det_core_inject_grace(DETCore* core, uint16_t node_id, float amount);

/** Process grace buffers (called during step) */
void det_core_process_grace(DETCore* core, float dt);

/** Check if node is at recovery boundary */
bool det_core_needs_grace(const DETCore* core, uint16_t node_id);

/** Get total grace needed across all nodes */
float det_core_total_grace_needed(const DETCore* core);

/* ----- Gatekeeper ----- */

/** Evaluate a request through the gatekeeper */
DETDecision det_core_evaluate_request(
    DETCore* core,
    const uint32_t* tokens,
    uint32_t num_tokens,
    uint8_t target_domain,
    uint32_t retry_count
);

/* ----- Port Interface (LLM) ----- */

/** Initialize port nodes for LLM interface */
void det_core_init_ports(DETCore* core);

/** Inject stimulus through port nodes */
void det_core_inject_stimulus(
    DETCore* core,
    const uint8_t* port_indices,
    const float* activations,
    uint32_t num_activations
);

/** Create temporary interface bonds */
void det_core_create_interface_bonds(
    DETCore* core,
    uint8_t target_domain,
    float initial_C
);

/** Clean up temporary interface bonds */
void det_core_cleanup_interface_bonds(DETCore* core);

/* ----- Memory Domains ----- */

/** Register a memory domain */
bool det_core_register_domain(
    DETCore* core,
    const char* name,
    void* model_handle
);

/** Get domain coherence */
float det_core_get_domain_coherence(const DETCore* core, uint8_t domain);

/* ----- Queries ----- */

/** Get current emotional state */
DETEmotion det_core_get_emotion(const DETCore* core);

/** Get emotional state as string */
const char* det_core_emotion_string(DETEmotion emotion);

/** Get Self cluster affect */
void det_core_get_self_affect(
    const DETCore* core,
    float* valence,
    float* arousal,
    float* bondedness
);

/** Get aggregate metrics */
void det_core_get_aggregates(
    const DETCore* core,
    float* presence,
    float* coherence,
    float* resource,
    float* debt
);

/* ----- Node/Bond Management ----- */

/** Recruit a node from dormant pool */
int32_t det_core_recruit_node(DETCore* core, DETLayer target_layer);

/** Return a node to dormant pool */
void det_core_retire_node(DETCore* core, uint16_t node_id);

/** Create a bond between two nodes */
int32_t det_core_create_bond(DETCore* core, uint16_t i, uint16_t j);

/** Find bond between two nodes (-1 if not found) */
int32_t det_core_find_bond(const DETCore* core, uint16_t i, uint16_t j);

/* ----- Phase 4: Learning via Recruitment ----- */

/** Check if learning/division is possible (recruitment criteria) */
bool det_core_can_learn(const DETCore* core, float complexity, uint8_t domain);

/** Activate a new domain with recruited nodes */
bool det_core_activate_domain(
    DETCore* core,
    const char* name,
    uint32_t num_nodes,
    float initial_coherence
);

/** Transfer pattern template from source to target domain */
bool det_core_transfer_pattern(
    DETCore* core,
    uint8_t source_domain,
    uint8_t target_domain,
    float transfer_strength
);

/** Get learning capacity (available agency for recruitment) */
float det_core_learning_capacity(const DETCore* core);

/* ----- Phase 4: Multi-Session Support ----- */

/** Save core state to buffer (returns bytes written, 0 on error) */
size_t det_core_save_state(const DETCore* core, void* buffer, size_t buffer_size);

/** Load core state from buffer (returns success) */
bool det_core_load_state(DETCore* core, const void* buffer, size_t data_size);

/** Get state size for serialization */
size_t det_core_state_size(const DETCore* core);

/* ----- Default Parameters ----- */

/** Get default parameters */
DETParams det_default_params(void);

#ifdef __cplusplus
}
#endif

#endif /* DET_CORE_H */
