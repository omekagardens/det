/**
 * EIS Substrate v2 - Metal Backend C Wrapper
 * ==========================================
 *
 * Thin C wrapper for the Metal backend.
 * On Apple platforms, this links to the Objective-C implementation.
 * On other platforms, provides stub implementations that return errors.
 */

#include "../include/substrate_metal.h"
#include "../include/substrate_lattice.h"

#if !defined(__APPLE__)

/* ==========================================================================
 * STUB IMPLEMENTATION FOR NON-APPLE PLATFORMS
 * ========================================================================== */

SubstrateMetalConfig sub_metal_default_config(void) {
    SubstrateMetalConfig config = {0};
    config.max_nodes = 65536;
    config.max_bonds = 131072;
    config.max_lanes = 65536;
    config.max_proposals = SUB_MAX_PROPOSALS;
    config.max_instructions = 4096;
    config.enable_timestamps = false;
    config.prefer_discrete_gpu = false;
    return config;
}

int sub_metal_is_available(void) {
    return 0;  /* Metal not available on non-Apple platforms */
}

SubstrateMetalHandle sub_metal_create(void) {
    return NULL;
}

SubstrateMetalHandle sub_metal_create_with_config(const SubstrateMetalConfig* config) {
    (void)config;
    return NULL;
}

void sub_metal_destroy(SubstrateMetalHandle ctx) {
    (void)ctx;
}

int sub_metal_upload_nodes(SubstrateMetalHandle ctx, const NodeArrays* nodes, uint32_t num_nodes) {
    (void)ctx; (void)nodes; (void)num_nodes;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_upload_bonds(SubstrateMetalHandle ctx, const BondArrays* bonds, uint32_t num_bonds) {
    (void)ctx; (void)bonds; (void)num_bonds;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_upload_program(SubstrateMetalHandle ctx, const PredecodedProgram* program) {
    (void)ctx; (void)program;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_download_nodes(SubstrateMetalHandle ctx, NodeArrays* nodes, uint32_t num_nodes) {
    (void)ctx; (void)nodes; (void)num_nodes;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_download_bonds(SubstrateMetalHandle ctx, BondArrays* bonds, uint32_t num_bonds) {
    (void)ctx; (void)bonds; (void)num_bonds;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_execute_phase(SubstrateMetalHandle ctx, SubstratePhase phase, uint32_t num_lanes) {
    (void)ctx; (void)phase; (void)num_lanes;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_execute_tick(SubstrateMetalHandle ctx, uint32_t num_lanes) {
    (void)ctx; (void)num_lanes;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_execute_ticks(SubstrateMetalHandle ctx, uint32_t num_lanes, uint32_t num_ticks) {
    (void)ctx; (void)num_lanes; (void)num_ticks;
    return SUB_METAL_ERR_NO_DEVICE;
}

void sub_metal_synchronize(SubstrateMetalHandle ctx) {
    (void)ctx;
}

void sub_metal_set_lane_mode(SubstrateMetalHandle ctx, LaneOwnershipMode mode) {
    (void)ctx; (void)mode;
}

void sub_metal_set_seed(SubstrateMetalHandle ctx, uint64_t seed) {
    (void)ctx; (void)seed;
}

void sub_metal_set_tick(SubstrateMetalHandle ctx, uint64_t tick) {
    (void)ctx; (void)tick;
}

const char* sub_metal_device_name(SubstrateMetalHandle ctx) {
    (void)ctx;
    return "Metal not available";
}

size_t sub_metal_memory_usage(SubstrateMetalHandle ctx) {
    (void)ctx;
    return 0;
}

void sub_metal_get_stats(SubstrateMetalHandle ctx, uint64_t* out_ticks, uint64_t* out_gpu_time_ns) {
    (void)ctx;
    if (out_ticks) *out_ticks = 0;
    if (out_gpu_time_ns) *out_gpu_time_ns = 0;
}

const char* sub_metal_get_error(SubstrateMetalHandle ctx) {
    (void)ctx;
    return "Metal is only available on macOS/iOS";
}

/* Lattice physics stubs (Fix: add missing stubs for lattice API) */
int sub_metal_upload_lattice_config(SubstrateMetalHandle ctx, const LatticeConfig* config,
                                     const LatticePhysicsParams* physics) {
    (void)ctx; (void)config; (void)physics;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_lattice_step(SubstrateMetalHandle ctx, Lattice* L) {
    (void)ctx; (void)L;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_lattice_step_n(SubstrateMetalHandle ctx, Lattice* L, uint32_t num_steps) {
    (void)ctx; (void)L; (void)num_steps;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_lattice_solve_gravity(SubstrateMetalHandle ctx, Lattice* L, uint32_t iterations) {
    (void)ctx; (void)L; (void)iterations;
    return SUB_METAL_ERR_NO_DEVICE;
}

int sub_metal_lattice_get_stats(SubstrateMetalHandle ctx, const Lattice* L, LatticeStats* stats) {
    (void)ctx; (void)L; (void)stats;
    return SUB_METAL_ERR_NO_DEVICE;
}

#endif /* !__APPLE__ */
