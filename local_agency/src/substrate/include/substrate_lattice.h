/**
 * EIS Substrate v2 - Lattice Topology Extension
 * ==============================================
 *
 * The lattice is the fundamental computational space of DET.
 * Every creature lives on nodes, bonds connect them, and all
 * physics operates on the lattice.
 *
 * This extension adds:
 * - Coordinate systems (1D, 2D, 3D)
 * - Implicit neighbor relationships
 * - Periodic boundary conditions
 * - FFT-based gravity solving
 * - Bulk operations on neighborhoods
 *
 * Design:
 * - Lattice is a TOPOLOGY over existing NodeArrays/BondArrays
 * - Coordinates map to node IDs
 * - Bonds are auto-generated from lattice adjacency
 * - All DET physics kernels work on lattice-aware substrate
 */

#ifndef SUBSTRATE_LATTICE_H
#define SUBSTRATE_LATTICE_H

#include "substrate_types.h"
#include "eis_substrate_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * LATTICE CONFIGURATION
 * ========================================================================== */

#define LATTICE_MAX_DIM         3       /* Max dimensions (1D, 2D, 3D) */
#define LATTICE_MAX_SIZE        1024    /* Max size per dimension */

/** Boundary condition type */
typedef enum {
    BOUNDARY_PERIODIC = 0,      /* Wrap around (torus) */
    BOUNDARY_FIXED = 1,         /* Fixed boundary values */
    BOUNDARY_REFLECT = 2,       /* Reflecting boundary */
    BOUNDARY_OPEN = 3,          /* Open boundary (no wrap) */
} LatticeBoundary;

/** Lattice configuration */
typedef struct {
    uint32_t dim;               /* Dimensionality: 1, 2, or 3 */
    uint32_t shape[LATTICE_MAX_DIM];  /* Size in each dimension */
    LatticeBoundary boundary;   /* Boundary condition */
    float dx;                   /* Grid spacing (default 1.0) */
    float dt;                   /* Time step (default 0.01) */
} LatticeConfig;

/** Direction enum for neighbor access */
typedef enum {
    DIR_X_POS = 0,  /* +x direction */
    DIR_X_NEG = 1,  /* -x direction */
    DIR_Y_POS = 2,  /* +y direction (2D/3D) */
    DIR_Y_NEG = 3,  /* -y direction (2D/3D) */
    DIR_Z_POS = 4,  /* +z direction (3D) */
    DIR_Z_NEG = 5,  /* -z direction (3D) */
} LatticeDirection;

/* ==========================================================================
 * LATTICE PHYSICS PARAMETERS (DET v6.3)
 * ========================================================================== */

/** DET v6.3 physics parameters */
typedef struct {
    /* Flow parameters */
    float sigma;                /* Base conductivity (default 1.0) */
    float outflow_limit;        /* Max fraction of F that can leave per step (0.25) */
    float F_floor;              /* Minimum F for floor flux (0.001) */

    /* Gravity parameters */
    float kappa_grav;           /* Helmholtz screening (5.0) */
    float mu_grav;              /* Poisson source strength (2.0) */
    float beta_g;               /* Momentum-gravity coupling (10.0) */

    /* Momentum parameters */
    float alpha_pi;             /* Momentum amplification from flow (0.12) */
    float lambda_pi;            /* Momentum decay rate (0.008) */

    /* Coherence parameters */
    float alpha_C;              /* Coherence growth from flow (0.01) */
    float lambda_C;             /* Coherence decay rate (0.001) */
    float C_init;               /* Initial/minimum coherence (0.15) */

    /* Structure parameters */
    float alpha_q;              /* Outflow -> structure rate (0.05) */
    float gamma_q;              /* Structure decay rate (0.01) */

    /* Grace parameters */
    bool grace_enabled;         /* Enable grace injection */
    float F_MIN_grace;          /* Grace threshold (0.01) */

    /* Feature flags */
    bool gravity_enabled;       /* Enable gravitational flux */
    bool momentum_enabled;      /* Enable momentum dynamics */
    bool variable_dt;           /* Use variable timestep (Delta_tau) */
} LatticePhysicsParams;

/* ==========================================================================
 * LATTICE STRUCTURE
 * ========================================================================== */

/**
 * Lattice - Topology over NodeArrays/BondArrays
 *
 * The lattice owns its own NodeArrays and BondArrays.
 * Coordinates (x, y, z) map to linear node IDs.
 * Bonds are auto-generated for lattice adjacency.
 */
typedef struct Lattice {
    /* Configuration */
    LatticeConfig config;
    LatticePhysicsParams physics;

    /* Node/Bond arrays (owned by lattice) */
    NodeArrays nodes;
    BondArrays bonds;

    /* Derived geometry */
    uint32_t num_nodes;         /* Total nodes = product of shape */
    uint32_t num_bonds;         /* Total bonds = adjacency count */
    uint32_t num_neighbors;     /* Neighbors per node (2*dim for interior) */

    /* Neighbor lookup table
     * neighbor_offset[d] = offset to add for direction d
     * Handles periodic boundaries via lookup
     */
    int32_t* neighbor_offset;   /* [num_nodes * 2 * dim] */

    /* Bond lookup table
     * bond_index[node * 2 * dim + dir] = bond ID for that direction
     */
    uint32_t* bond_index;       /* [num_nodes * 2 * dim] */

    /* Bond metadata (Fix A6: store dimension at creation, not inferred) */
    uint8_t* bond_dim;          /* [num_bonds] dimension index for each bond */

    /* FFT workspace (for gravity solver) */
    float* fft_workspace;       /* Complex interleaved: [num_nodes * 2] */
    float* psi_field;           /* Helmholtz solution */
    float* phi_field;           /* Poisson solution (gravity potential) */

    /* Scratch arrays for physics (Fix B8: pre-allocated, not per-step) */
    float* Delta_tau;           /* [num_nodes] proper timestep */
    float* J_total;             /* [num_nodes * dim] total flux per direction */
    float* grad_phi;            /* [num_nodes * dim] gravity gradient */
    float* J_flux;              /* [num_bonds] per-bond flux (Fix B10) */
    float* scale;               /* [num_nodes] limiter scale factors */
    float* temp_weights;        /* [num_nodes] temp array for packet injection (Fix C12) */

    /* Statistics */
    uint64_t step_count;
    double total_mass_initial;
    double total_mass_current;

    /* State */
    bool initialized;
    uint32_t id;                /* Lattice handle for primitive calls */
} Lattice;

/* ==========================================================================
 * LATTICE LIFECYCLE API
 * ========================================================================== */

/** Create a new lattice with given configuration */
Lattice* lattice_create(const LatticeConfig* config);

/** Create with default physics parameters */
Lattice* lattice_create_default(uint32_t dim, uint32_t size);

/** Destroy lattice and free all resources */
void lattice_destroy(Lattice* lattice);

/** Reset lattice to initial state (keep topology) */
void lattice_reset(Lattice* lattice);

/* ==========================================================================
 * COORDINATE/INDEX CONVERSION
 * ========================================================================== */

/** Convert (x) to node ID (1D) */
static inline uint32_t lattice_coord_1d(const Lattice* L, int32_t x) {
    int32_t nx = (int32_t)L->config.shape[0];
    x = ((x % nx) + nx) % nx;  /* Periodic wrap */
    return (uint32_t)x;
}

/** Convert (x, y) to node ID (2D) */
static inline uint32_t lattice_coord_2d(const Lattice* L, int32_t x, int32_t y) {
    int32_t nx = (int32_t)L->config.shape[0];
    int32_t ny = (int32_t)L->config.shape[1];
    x = ((x % nx) + nx) % nx;
    y = ((y % ny) + ny) % ny;
    return (uint32_t)(y * nx + x);
}

/** Convert (x, y, z) to node ID (3D) */
static inline uint32_t lattice_coord_3d(const Lattice* L, int32_t x, int32_t y, int32_t z) {
    int32_t nx = (int32_t)L->config.shape[0];
    int32_t ny = (int32_t)L->config.shape[1];
    int32_t nz = (int32_t)L->config.shape[2];
    x = ((x % nx) + nx) % nx;
    y = ((y % ny) + ny) % ny;
    z = ((z % nz) + nz) % nz;
    return (uint32_t)(z * ny * nx + y * nx + x);
}

/** Convert node ID to coordinates */
void lattice_index_to_coord(const Lattice* L, uint32_t node_id,
                            int32_t* x, int32_t* y, int32_t* z);

/* ==========================================================================
 * NEIGHBOR ACCESS
 * ========================================================================== */

/** Get neighbor node ID in given direction */
uint32_t lattice_get_neighbor(const Lattice* L, uint32_t node_id, LatticeDirection dir);

/** Get bond ID connecting node to neighbor in direction */
uint32_t lattice_get_bond(const Lattice* L, uint32_t node_id, LatticeDirection dir);

/** Get number of neighbors for a node (handles boundaries) */
uint32_t lattice_neighbor_count(const Lattice* L, uint32_t node_id);

/* ==========================================================================
 * FIELD ACCESS (convenience wrappers)
 * ========================================================================== */

/** Get/Set node F (resource) */
static inline float lattice_get_F(const Lattice* L, uint32_t node_id) {
    return L->nodes.F[node_id];
}
static inline void lattice_set_F(Lattice* L, uint32_t node_id, float value) {
    L->nodes.F[node_id] = value;
}

/** Get/Set node q (structure) */
static inline float lattice_get_q(const Lattice* L, uint32_t node_id) {
    return L->nodes.q[node_id];
}
static inline void lattice_set_q(Lattice* L, uint32_t node_id, float value) {
    L->nodes.q[node_id] = value;
}

/** Get/Set node a (agency) */
static inline float lattice_get_a(const Lattice* L, uint32_t node_id) {
    return L->nodes.a[node_id];
}
static inline void lattice_set_a(Lattice* L, uint32_t node_id, float value) {
    L->nodes.a[node_id] = value;
}

/** Get/Set bond C (coherence) */
static inline float lattice_get_C(const Lattice* L, uint32_t bond_id) {
    return L->bonds.C[bond_id];
}
static inline void lattice_set_C(Lattice* L, uint32_t bond_id, float value) {
    L->bonds.C[bond_id] = value;
}

/** Get/Set bond pi (momentum) */
static inline float lattice_get_pi(const Lattice* L, uint32_t bond_id) {
    return L->bonds.pi[bond_id];
}
static inline void lattice_set_pi(Lattice* L, uint32_t bond_id, float value) {
    L->bonds.pi[bond_id] = value;
}

/* ==========================================================================
 * PACKET INJECTION
 * ========================================================================== */

/**
 * Add a Gaussian resource packet to the lattice
 *
 * @param L         Lattice
 * @param center    Center position array [dim]
 * @param mass      Total resource to inject
 * @param width     Gaussian width (sigma)
 * @param momentum  Momentum per direction [dim] (can be NULL)
 * @param initial_q Initial structure value [0,1]
 */
void lattice_add_packet(Lattice* L,
                        const float* center,
                        float mass,
                        float width,
                        const float* momentum,
                        float initial_q);

/* ==========================================================================
 * PHYSICS STEP
 * ========================================================================== */

/**
 * Execute one physics timestep
 *
 * Implements DET v6.3 flow:
 * 1. Compute presence: P = a*sigma / (1+F) / (1+H)
 * 2. Compute proper timestep: Delta_tau = P * dt
 * 3. Compute gravity: Helmholtz + Poisson
 * 4. Compute fluxes: diffusive + momentum + gravity + floor
 * 5. Apply limiter for mass conservation
 * 6. Update F, q, C, pi, a
 * 7. Grace injection for depleted nodes
 */
void lattice_step(Lattice* L);

/** Execute n physics timesteps */
void lattice_step_n(Lattice* L, uint32_t n);

/**
 * Fused step interface for batching (Fix D13)
 *
 * Executes n steps with minimal overhead, batching witness emissions.
 * Returns total flux magnitude for witnessable output.
 *
 * @param L           Lattice
 * @param n           Number of steps
 * @param out_fluxes  Optional: array to receive per-bond flux snapshots [num_bonds * n]
 * @param emit_every  Emit witness snapshot every N steps (0 = only final)
 * @return            Total flux magnitude over all steps
 */
float lattice_step_fused(Lattice* L, uint32_t n, float* out_fluxes, uint32_t emit_every);

/* ==========================================================================
 * FFT GRAVITY SOLVER
 * ========================================================================== */

/**
 * Solve gravity potential via FFT
 *
 * 1. Helmholtz: (nabla^2 - kappa^2) psi = q
 * 2. Poisson: nabla^2 phi = -mu * psi
 *
 * Uses vDSP/Accelerate on macOS, FFTW fallback on other platforms.
 */
void lattice_solve_gravity(Lattice* L);

/** Get gravity potential at node */
float lattice_get_phi(const Lattice* L, uint32_t node_id);

/** Get gravity gradient in direction */
float lattice_get_grad_phi(const Lattice* L, uint32_t node_id, LatticeDirection dir);

/* ==========================================================================
 * STATISTICS
 * ========================================================================== */

/** Get total mass (sum of F) */
float lattice_total_mass(const Lattice* L);

/** Get center of mass position */
void lattice_center_of_mass(const Lattice* L, float* com);

/** Get separation between two largest mass concentrations */
float lattice_separation(const Lattice* L);

/** Get total kinetic energy (from momentum) */
float lattice_kinetic_energy(const Lattice* L);

/** Get gravitational potential energy */
float lattice_potential_energy(const Lattice* L);

/** Get comprehensive statistics */
typedef struct {
    float total_mass;
    float total_structure;
    float total_momentum;
    float kinetic_energy;
    float potential_energy;
    float separation;
    float com[LATTICE_MAX_DIM];
    uint64_t step_count;
} LatticeStats;

void lattice_get_stats(const Lattice* L, LatticeStats* stats);

/* ==========================================================================
 * RENDERING
 * ========================================================================== */

/** Field to render */
typedef enum {
    RENDER_FIELD_F = 0,         /* Resource */
    RENDER_FIELD_Q = 1,         /* Structure */
    RENDER_FIELD_A = 2,         /* Agency */
    RENDER_FIELD_P = 3,         /* Presence */
    RENDER_FIELD_PHI = 4,       /* Gravity potential */
} RenderField;

/**
 * Render lattice field to ASCII art
 *
 * @param L         Lattice
 * @param field     Field to render
 * @param width     Output width in characters
 * @param out       Output buffer
 * @param out_size  Buffer size
 * @return Number of characters written
 */
uint32_t lattice_render(const Lattice* L, RenderField field,
                        uint32_t width, char* out, uint32_t out_size);

/* ==========================================================================
 * PARAMETER CONTROL
 * ========================================================================== */

/** Set physics parameter by name */
bool lattice_set_param(Lattice* L, const char* param_name, float value);

/** Get physics parameter by name */
float lattice_get_param(const Lattice* L, const char* param_name);

/** Set default v6.3 physics parameters */
void lattice_set_default_physics(LatticePhysicsParams* params);

/* ==========================================================================
 * LATTICE REGISTRY (for primitive calls)
 * Note: Registry is thread-safe via internal mutex (Fix C11)
 * ========================================================================== */

/** Initialize global lattice registry */
void lattice_registry_init(void);

/** Shutdown registry and destroy all lattices */
void lattice_registry_shutdown(void);

/** Register a new lattice, returns handle ID (thread-safe) */
uint32_t lattice_registry_add(Lattice* L);

/** Get lattice by handle ID (thread-safe) */
Lattice* lattice_registry_get(uint32_t id);

/** Remove lattice from registry (does not destroy) (thread-safe) */
void lattice_registry_remove(uint32_t id);

/* ==========================================================================
 * SUBSTRATE INTEGRATION
 * ========================================================================== */

/**
 * Integrate lattice with substrate VM
 *
 * Maps lattice nodes/bonds to VM's trace memory.
 * Allows EIS bytecode to operate on lattice directly.
 */
void lattice_bind_to_vm(Lattice* L, SubstrateVM* vm);

/**
 * Sync VM state back to lattice after execution
 */
void lattice_sync_from_vm(Lattice* L, SubstrateVM* vm);

#ifdef __cplusplus
}
#endif

#endif /* SUBSTRATE_LATTICE_H */
