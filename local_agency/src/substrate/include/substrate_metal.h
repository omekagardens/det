/**
 * EIS Substrate v2 - Metal GPU Backend API
 * =========================================
 *
 * GPU acceleration for the EIS Substrate using Metal compute shaders.
 * The substrate's Structure-of-Arrays layout and phase-based execution
 * model map directly to GPU parallel execution.
 *
 * Usage:
 *   1. Create a Metal context: sub_metal_create()
 *   2. Upload node/bond state: sub_metal_upload_nodes(), sub_metal_upload_bonds()
 *   3. Upload compiled program: sub_metal_upload_program()
 *   4. Execute phases: sub_metal_execute_tick() or individual phases
 *   5. Download results: sub_metal_download_nodes(), sub_metal_download_bonds()
 *   6. Destroy context: sub_metal_destroy()
 */

#ifndef SUBSTRATE_METAL_H
#define SUBSTRATE_METAL_H

#include "substrate_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * OPAQUE HANDLE
 * ========================================================================== */

/** Opaque handle to Metal backend context */
typedef struct SubstrateMetalContext* SubstrateMetalHandle;

/* ==========================================================================
 * CONFIGURATION
 * ========================================================================== */

/** Metal backend configuration */
typedef struct {
    uint32_t max_nodes;         /** Maximum number of nodes (default: 65536) */
    uint32_t max_bonds;         /** Maximum number of bonds (default: 131072) */
    uint32_t max_lanes;         /** Maximum number of lanes (default: 65536) */
    uint32_t max_proposals;     /** Max proposals per lane (default: 8) */
    uint32_t max_instructions;  /** Max program instructions (default: 4096) */
    bool enable_timestamps;     /** Enable GPU timestamps for profiling */
    bool prefer_discrete_gpu;   /** Prefer discrete GPU if available */
} SubstrateMetalConfig;

/** Default configuration */
SubstrateMetalConfig sub_metal_default_config(void);

/* ==========================================================================
 * LIFECYCLE
 * ========================================================================== */

/**
 * Check if Metal GPU is available on this system.
 * @return 1 if Metal is available, 0 otherwise
 */
int sub_metal_is_available(void);

/**
 * Create a Metal backend context with default configuration.
 * @return Handle to the context, or NULL on failure
 */
SubstrateMetalHandle sub_metal_create(void);

/**
 * Create a Metal backend context with custom configuration.
 * @param config Configuration options
 * @return Handle to the context, or NULL on failure
 */
SubstrateMetalHandle sub_metal_create_with_config(const SubstrateMetalConfig* config);

/**
 * Destroy a Metal backend context and free resources.
 * @param ctx Handle to destroy
 */
void sub_metal_destroy(SubstrateMetalHandle ctx);

/* ==========================================================================
 * STATE TRANSFER
 * ========================================================================== */

/**
 * Upload node state from CPU to GPU.
 * @param ctx Metal context
 * @param nodes Node arrays to upload
 * @param num_nodes Number of nodes (must not exceed max_nodes)
 * @return 0 on success, negative error code on failure
 */
int sub_metal_upload_nodes(SubstrateMetalHandle ctx, const NodeArrays* nodes, uint32_t num_nodes);

/**
 * Upload bond state from CPU to GPU.
 * @param ctx Metal context
 * @param bonds Bond arrays to upload
 * @param num_bonds Number of bonds (must not exceed max_bonds)
 * @return 0 on success, negative error code on failure
 */
int sub_metal_upload_bonds(SubstrateMetalHandle ctx, const BondArrays* bonds, uint32_t num_bonds);

/**
 * Upload compiled EIS program to GPU constant memory.
 * @param ctx Metal context
 * @param program Predecoded program
 * @return 0 on success, negative error code on failure
 */
int sub_metal_upload_program(SubstrateMetalHandle ctx, const PredecodedProgram* program);

/**
 * Download node state from GPU to CPU.
 * @param ctx Metal context
 * @param nodes Node arrays to fill (must be pre-allocated)
 * @param num_nodes Number of nodes to download
 * @return 0 on success, negative error code on failure
 */
int sub_metal_download_nodes(SubstrateMetalHandle ctx, NodeArrays* nodes, uint32_t num_nodes);

/**
 * Download bond state from GPU to CPU.
 * @param ctx Metal context
 * @param bonds Bond arrays to fill (must be pre-allocated)
 * @param num_bonds Number of bonds to download
 * @return 0 on success, negative error code on failure
 */
int sub_metal_download_bonds(SubstrateMetalHandle ctx, BondArrays* bonds, uint32_t num_bonds);

/* ==========================================================================
 * EXECUTION
 * ========================================================================== */

/**
 * Execute a single phase on the GPU.
 * @param ctx Metal context
 * @param phase Phase to execute (PHASE_READ, PHASE_PROPOSE, PHASE_CHOOSE, PHASE_COMMIT)
 * @param num_lanes Number of parallel lanes to execute
 * @return 0 on success, negative error code on failure
 */
int sub_metal_execute_phase(SubstrateMetalHandle ctx, SubstratePhase phase, uint32_t num_lanes);

/**
 * Execute one full tick (all 4 phases: READ → PROPOSE → CHOOSE → COMMIT).
 * @param ctx Metal context
 * @param num_lanes Number of parallel lanes to execute
 * @return 0 on success, negative error code on failure
 */
int sub_metal_execute_tick(SubstrateMetalHandle ctx, uint32_t num_lanes);

/**
 * Execute multiple ticks.
 * @param ctx Metal context
 * @param num_lanes Number of parallel lanes
 * @param num_ticks Number of ticks to execute
 * @return 0 on success, negative error code on failure
 */
int sub_metal_execute_ticks(SubstrateMetalHandle ctx, uint32_t num_lanes, uint32_t num_ticks);

/**
 * Wait for all GPU operations to complete.
 * @param ctx Metal context
 */
void sub_metal_synchronize(SubstrateMetalHandle ctx);

/* ==========================================================================
 * LANE CONFIGURATION
 * ========================================================================== */

/**
 * Set lane ownership mode for effect deduplication.
 * @param ctx Metal context
 * @param mode Ownership mode (LANE_OWNER_NONE, LANE_OWNER_NODE, LANE_OWNER_BOND)
 */
void sub_metal_set_lane_mode(SubstrateMetalHandle ctx, LaneOwnershipMode mode);

/**
 * Set random seed for all lanes.
 * @param ctx Metal context
 * @param seed Base seed (each lane derives its own from this + lane_id)
 */
void sub_metal_set_seed(SubstrateMetalHandle ctx, uint64_t seed);

/**
 * Set current tick number.
 * @param ctx Metal context
 * @param tick Tick number
 */
void sub_metal_set_tick(SubstrateMetalHandle ctx, uint64_t tick);

/* ==========================================================================
 * QUERY
 * ========================================================================== */

/**
 * Get the Metal device name.
 * @param ctx Metal context
 * @return Device name string (valid for lifetime of context)
 */
const char* sub_metal_device_name(SubstrateMetalHandle ctx);

/**
 * Get GPU memory usage in bytes.
 * @param ctx Metal context
 * @return Total allocated GPU memory
 */
size_t sub_metal_memory_usage(SubstrateMetalHandle ctx);

/**
 * Get execution statistics.
 * @param ctx Metal context
 * @param out_ticks Output: number of ticks executed
 * @param out_gpu_time_ns Output: total GPU time in nanoseconds
 */
void sub_metal_get_stats(SubstrateMetalHandle ctx, uint64_t* out_ticks, uint64_t* out_gpu_time_ns);

/**
 * Get the last error message.
 * @param ctx Metal context
 * @return Error message string (valid until next API call)
 */
const char* sub_metal_get_error(SubstrateMetalHandle ctx);

/* ==========================================================================
 * LATTICE PHYSICS (GPU-accelerated DET v6.3)
 * ========================================================================== */

#include "substrate_lattice.h"

/**
 * Upload lattice configuration to GPU.
 * @param ctx Metal context
 * @param config Lattice configuration
 * @return 0 on success, negative error code on failure
 */
int sub_metal_upload_lattice_config(SubstrateMetalHandle ctx, const LatticeConfig* config,
                                     const LatticePhysicsParams* physics);

/**
 * Execute one lattice physics step on GPU.
 * This runs all physics kernels: presence, flux, limiter, apply, momentum, etc.
 * @param ctx Metal context
 * @param L Lattice (for node/bond data)
 * @return 0 on success, negative error code on failure
 */
int sub_metal_lattice_step(SubstrateMetalHandle ctx, Lattice* L);

/**
 * Execute multiple lattice physics steps on GPU.
 * @param ctx Metal context
 * @param L Lattice
 * @param num_steps Number of steps to execute
 * @return 0 on success, negative error code on failure
 */
int sub_metal_lattice_step_n(SubstrateMetalHandle ctx, Lattice* L, uint32_t num_steps);

/**
 * Solve gravity potential on GPU using iterative method.
 * For large lattices, this is much faster than CPU.
 * @param ctx Metal context
 * @param L Lattice
 * @param iterations Number of Jacobi iterations
 * @return 0 on success, negative error code on failure
 */
int sub_metal_lattice_solve_gravity(SubstrateMetalHandle ctx, Lattice* L, uint32_t iterations);

/**
 * Compute lattice statistics on GPU.
 * @param ctx Metal context
 * @param L Lattice
 * @param stats Output statistics
 * @return 0 on success, negative error code on failure
 */
int sub_metal_lattice_get_stats(SubstrateMetalHandle ctx, const Lattice* L, LatticeStats* stats);

/* ==========================================================================
 * ERROR CODES
 * ========================================================================== */

#define SUB_METAL_OK            0
#define SUB_METAL_ERR_NO_DEVICE -1
#define SUB_METAL_ERR_ALLOC     -2
#define SUB_METAL_ERR_COMPILE   -3
#define SUB_METAL_ERR_UPLOAD    -4
#define SUB_METAL_ERR_EXECUTE   -5
#define SUB_METAL_ERR_DOWNLOAD  -6
#define SUB_METAL_ERR_INVALID   -7
#define SUB_METAL_ERR_OVERFLOW  -8

#ifdef __cplusplus
}
#endif

#endif /* SUBSTRATE_METAL_H */
