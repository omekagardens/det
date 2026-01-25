/**
 * EIS Substrate v2 - Lattice Implementation
 * ==========================================
 *
 * C implementation of the DET lattice substrate.
 * This is the fundamental computational space for DET-OS.
 */

#include "../include/substrate_lattice.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* macOS Accelerate for FFT */
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_VDSP_FFT 1
#endif

/* ==========================================================================
 * LATTICE REGISTRY
 * ========================================================================== */

#define MAX_LATTICES 64

static Lattice* g_lattice_registry[MAX_LATTICES] = {0};
static uint32_t g_next_lattice_id = 1;
static bool g_registry_initialized = false;

void lattice_registry_init(void) {
    if (g_registry_initialized) return;
    memset(g_lattice_registry, 0, sizeof(g_lattice_registry));
    g_next_lattice_id = 1;
    g_registry_initialized = true;
}

void lattice_registry_shutdown(void) {
    for (int i = 0; i < MAX_LATTICES; i++) {
        if (g_lattice_registry[i]) {
            lattice_destroy(g_lattice_registry[i]);
            g_lattice_registry[i] = NULL;
        }
    }
    g_registry_initialized = false;
}

uint32_t lattice_registry_add(Lattice* L) {
    if (!g_registry_initialized) lattice_registry_init();

    for (int i = 0; i < MAX_LATTICES; i++) {
        if (g_lattice_registry[i] == NULL) {
            g_lattice_registry[i] = L;
            L->id = g_next_lattice_id++;
            return L->id;
        }
    }
    return 0;  /* Registry full */
}

Lattice* lattice_registry_get(uint32_t id) {
    for (int i = 0; i < MAX_LATTICES; i++) {
        if (g_lattice_registry[i] && g_lattice_registry[i]->id == id) {
            return g_lattice_registry[i];
        }
    }
    return NULL;
}

void lattice_registry_remove(uint32_t id) {
    for (int i = 0; i < MAX_LATTICES; i++) {
        if (g_lattice_registry[i] && g_lattice_registry[i]->id == id) {
            g_lattice_registry[i] = NULL;
            return;
        }
    }
}

/* ==========================================================================
 * DEFAULT PHYSICS PARAMETERS (DET v6.3 Theory Card VII.2)
 * ========================================================================== */

void lattice_set_default_physics(LatticePhysicsParams* p) {
    /* Flow */
    p->sigma = 1.0f;
    p->outflow_limit = 0.25f;
    p->F_floor = 0.001f;

    /* Gravity */
    p->kappa_grav = 5.0f;
    p->mu_grav = 2.0f;
    p->beta_g = 10.0f;  /* 5.0 * mu_grav */

    /* Momentum */
    p->alpha_pi = 0.12f;
    p->lambda_pi = 0.008f;

    /* Coherence */
    p->alpha_C = 0.01f;
    p->lambda_C = 0.001f;
    p->C_init = 0.15f;

    /* Structure */
    p->alpha_q = 0.05f;
    p->gamma_q = 0.01f;

    /* Grace */
    p->grace_enabled = true;
    p->F_MIN_grace = 0.01f;

    /* Features */
    p->gravity_enabled = true;
    p->momentum_enabled = true;
    p->variable_dt = true;
}

/* ==========================================================================
 * LATTICE CREATION
 * ========================================================================== */

/** Allocate node arrays */
static bool alloc_node_arrays(NodeArrays* na, uint32_t count) {
    na->F = (float*)calloc(count, sizeof(float));
    na->q = (float*)calloc(count, sizeof(float));
    na->a = (float*)calloc(count, sizeof(float));
    na->sigma = (float*)calloc(count, sizeof(float));
    na->P = (float*)calloc(count, sizeof(float));
    na->tau = (float*)calloc(count, sizeof(float));
    na->cos_theta = (float*)calloc(count, sizeof(float));
    na->sin_theta = (float*)calloc(count, sizeof(float));
    na->k = (uint32_t*)calloc(count, sizeof(uint32_t));
    na->r = (uint32_t*)calloc(count, sizeof(uint32_t));
    na->flags = (uint32_t*)calloc(count, sizeof(uint32_t));

    if (!na->F || !na->q || !na->a || !na->sigma || !na->P ||
        !na->tau || !na->cos_theta || !na->sin_theta ||
        !na->k || !na->r || !na->flags) {
        return false;
    }

    na->num_nodes = count;
    na->capacity = count;

    /* Initialize defaults */
    for (uint32_t i = 0; i < count; i++) {
        na->a[i] = 1.0f;       /* Full agency */
        na->sigma[i] = 1.0f;   /* Base processing rate */
    }

    return true;
}

/** Free node arrays */
static void free_node_arrays(NodeArrays* na) {
    free(na->F); na->F = NULL;
    free(na->q); na->q = NULL;
    free(na->a); na->a = NULL;
    free(na->sigma); na->sigma = NULL;
    free(na->P); na->P = NULL;
    free(na->tau); na->tau = NULL;
    free(na->cos_theta); na->cos_theta = NULL;
    free(na->sin_theta); na->sin_theta = NULL;
    free(na->k); na->k = NULL;
    free(na->r); na->r = NULL;
    free(na->flags); na->flags = NULL;
    na->num_nodes = 0;
    na->capacity = 0;
}

/** Allocate bond arrays */
static bool alloc_bond_arrays(BondArrays* ba, uint32_t count) {
    ba->node_i = (uint32_t*)calloc(count, sizeof(uint32_t));
    ba->node_j = (uint32_t*)calloc(count, sizeof(uint32_t));
    ba->C = (float*)calloc(count, sizeof(float));
    ba->pi = (float*)calloc(count, sizeof(float));
    ba->sigma = (float*)calloc(count, sizeof(float));
    ba->flags = (uint32_t*)calloc(count, sizeof(uint32_t));

    if (!ba->node_i || !ba->node_j || !ba->C || !ba->pi || !ba->sigma || !ba->flags) {
        return false;
    }

    ba->num_bonds = count;
    ba->capacity = count;
    return true;
}

/** Free bond arrays */
static void free_bond_arrays(BondArrays* ba) {
    free(ba->node_i); ba->node_i = NULL;
    free(ba->node_j); ba->node_j = NULL;
    free(ba->C); ba->C = NULL;
    free(ba->pi); ba->pi = NULL;
    free(ba->sigma); ba->sigma = NULL;
    free(ba->flags); ba->flags = NULL;
    ba->num_bonds = 0;
    ba->capacity = 0;
}

/** Generate lattice topology (bonds + neighbor tables) */
static bool generate_topology(Lattice* L) {
    uint32_t dim = L->config.dim;
    uint32_t num_nodes = L->num_nodes;
    uint32_t num_dirs = 2 * dim;  /* +x, -x, +y, -y, +z, -z */

    /* Allocate neighbor offset table */
    L->neighbor_offset = (int32_t*)calloc(num_nodes * num_dirs, sizeof(int32_t));
    L->bond_index = (uint32_t*)calloc(num_nodes * num_dirs, sizeof(uint32_t));

    if (!L->neighbor_offset || !L->bond_index) {
        return false;
    }

    /* Compute strides */
    int32_t stride[3] = {1, 0, 0};
    if (dim >= 2) stride[1] = (int32_t)L->config.shape[0];
    if (dim >= 3) stride[2] = (int32_t)(L->config.shape[0] * L->config.shape[1]);

    /* Count bonds (each bond counted once, in +x, +y, +z directions) */
    uint32_t bond_count = 0;
    for (uint32_t d = 0; d < dim; d++) {
        bond_count += num_nodes;  /* One bond per node per positive direction */
    }

    /* Allocate bonds */
    if (!alloc_bond_arrays(&L->bonds, bond_count)) {
        return false;
    }

    /* Initialize bonds with C_init */
    for (uint32_t i = 0; i < bond_count; i++) {
        L->bonds.C[i] = L->physics.C_init;
        L->bonds.sigma[i] = L->physics.sigma;
    }

    /* Generate neighbor offsets and bonds */
    uint32_t bond_idx = 0;

    for (uint32_t node_id = 0; node_id < num_nodes; node_id++) {
        /* Get coordinates */
        int32_t x, y, z;
        lattice_index_to_coord(L, node_id, &x, &y, &z);

        for (uint32_t d = 0; d < dim; d++) {
            int32_t shape_d = (int32_t)L->config.shape[d];

            /* Positive direction */
            int32_t coord_pos = (d == 0) ? x : (d == 1) ? y : z;
            int32_t next_pos = (coord_pos + 1) % shape_d;  /* Periodic wrap */
            int32_t offset_pos = (next_pos - coord_pos) * stride[d];
            /* Handle wrap-around */
            if (next_pos == 0) {
                offset_pos = -(shape_d - 1) * stride[d];
            }

            uint32_t dir_pos = d * 2;  /* DIR_X_POS=0, DIR_Y_POS=2, DIR_Z_POS=4 */
            L->neighbor_offset[node_id * num_dirs + dir_pos] = offset_pos;

            /* Create bond for positive direction */
            uint32_t neighbor_id = (uint32_t)((int32_t)node_id + offset_pos);
            L->bonds.node_i[bond_idx] = node_id;
            L->bonds.node_j[bond_idx] = neighbor_id;
            L->bond_index[node_id * num_dirs + dir_pos] = bond_idx;
            bond_idx++;

            /* Negative direction */
            int32_t prev_pos = (coord_pos - 1 + shape_d) % shape_d;
            int32_t offset_neg = (prev_pos - coord_pos) * stride[d];
            /* Handle wrap-around */
            if (prev_pos == shape_d - 1) {
                offset_neg = (shape_d - 1) * stride[d];
            }

            uint32_t dir_neg = d * 2 + 1;  /* DIR_X_NEG=1, DIR_Y_NEG=3, DIR_Z_NEG=5 */
            L->neighbor_offset[node_id * num_dirs + dir_neg] = offset_neg;

            /* Find bond for negative direction (created by neighbor's positive dir) */
            uint32_t neg_neighbor = (uint32_t)((int32_t)node_id + offset_neg);
            /* Bond was created when processing neg_neighbor's positive direction */
            L->bond_index[node_id * num_dirs + dir_neg] =
                L->bond_index[neg_neighbor * num_dirs + dir_pos];
        }
    }

    L->num_bonds = bond_idx;
    L->num_neighbors = num_dirs;
    return true;
}

Lattice* lattice_create(const LatticeConfig* config) {
    if (!config || config->dim < 1 || config->dim > 3) {
        return NULL;
    }

    Lattice* L = (Lattice*)calloc(1, sizeof(Lattice));
    if (!L) return NULL;

    /* Copy config */
    L->config = *config;
    if (L->config.dx <= 0) L->config.dx = 1.0f;
    if (L->config.dt <= 0) L->config.dt = 0.01f;

    /* Set default physics */
    lattice_set_default_physics(&L->physics);

    /* Calculate total nodes */
    L->num_nodes = 1;
    for (uint32_t d = 0; d < config->dim; d++) {
        L->num_nodes *= config->shape[d];
    }

    /* Allocate node arrays */
    if (!alloc_node_arrays(&L->nodes, L->num_nodes)) {
        lattice_destroy(L);
        return NULL;
    }

    /* Generate topology (bonds + neighbor tables) */
    if (!generate_topology(L)) {
        lattice_destroy(L);
        return NULL;
    }

    /* Allocate FFT workspace */
    L->fft_workspace = (float*)calloc(L->num_nodes * 2, sizeof(float));
    L->psi_field = (float*)calloc(L->num_nodes, sizeof(float));
    L->phi_field = (float*)calloc(L->num_nodes, sizeof(float));

    /* Allocate scratch arrays */
    L->Delta_tau = (float*)calloc(L->num_nodes, sizeof(float));
    L->J_total = (float*)calloc(L->num_nodes * config->dim, sizeof(float));
    L->grad_phi = (float*)calloc(L->num_nodes * config->dim, sizeof(float));

    if (!L->fft_workspace || !L->psi_field || !L->phi_field ||
        !L->Delta_tau || !L->J_total || !L->grad_phi) {
        lattice_destroy(L);
        return NULL;
    }

    /* Initialize Delta_tau to dt */
    for (uint32_t i = 0; i < L->num_nodes; i++) {
        L->Delta_tau[i] = L->config.dt;
    }

    L->initialized = true;
    L->step_count = 0;

    /* Register in global registry */
    lattice_registry_add(L);

    return L;
}

Lattice* lattice_create_default(uint32_t dim, uint32_t size) {
    LatticeConfig config = {0};
    config.dim = dim;
    for (uint32_t d = 0; d < dim && d < LATTICE_MAX_DIM; d++) {
        config.shape[d] = size;
    }
    config.boundary = BOUNDARY_PERIODIC;
    config.dx = 1.0f;
    config.dt = 0.01f;
    return lattice_create(&config);
}

void lattice_destroy(Lattice* L) {
    if (!L) return;

    lattice_registry_remove(L->id);

    free_node_arrays(&L->nodes);
    free_bond_arrays(&L->bonds);

    free(L->neighbor_offset);
    free(L->bond_index);
    free(L->fft_workspace);
    free(L->psi_field);
    free(L->phi_field);
    free(L->Delta_tau);
    free(L->J_total);
    free(L->grad_phi);

    free(L);
}

void lattice_reset(Lattice* L) {
    if (!L) return;

    /* Zero out node state (keep topology) */
    memset(L->nodes.F, 0, L->num_nodes * sizeof(float));
    memset(L->nodes.q, 0, L->num_nodes * sizeof(float));
    memset(L->nodes.P, 0, L->num_nodes * sizeof(float));
    memset(L->nodes.tau, 0, L->num_nodes * sizeof(float));

    /* Reset agency to 1.0 */
    for (uint32_t i = 0; i < L->num_nodes; i++) {
        L->nodes.a[i] = 1.0f;
        L->nodes.sigma[i] = 1.0f;
    }

    /* Reset bonds to C_init */
    for (uint32_t i = 0; i < L->num_bonds; i++) {
        L->bonds.C[i] = L->physics.C_init;
        L->bonds.pi[i] = 0.0f;
    }

    /* Reset scratch */
    for (uint32_t i = 0; i < L->num_nodes; i++) {
        L->Delta_tau[i] = L->config.dt;
    }

    L->step_count = 0;
}

/* ==========================================================================
 * COORDINATE CONVERSION
 * ========================================================================== */

void lattice_index_to_coord(const Lattice* L, uint32_t node_id,
                            int32_t* x, int32_t* y, int32_t* z) {
    uint32_t nx = L->config.shape[0];
    uint32_t ny = (L->config.dim >= 2) ? L->config.shape[1] : 1;

    *x = (int32_t)(node_id % nx);
    *y = (L->config.dim >= 2) ? (int32_t)((node_id / nx) % ny) : 0;
    *z = (L->config.dim >= 3) ? (int32_t)(node_id / (nx * ny)) : 0;
}

/* ==========================================================================
 * NEIGHBOR ACCESS
 * ========================================================================== */

uint32_t lattice_get_neighbor(const Lattice* L, uint32_t node_id, LatticeDirection dir) {
    if (!L || node_id >= L->num_nodes || dir >= L->num_neighbors) {
        return node_id;  /* Return self on error */
    }
    int32_t offset = L->neighbor_offset[node_id * L->num_neighbors + dir];
    return (uint32_t)((int32_t)node_id + offset);
}

uint32_t lattice_get_bond(const Lattice* L, uint32_t node_id, LatticeDirection dir) {
    if (!L || node_id >= L->num_nodes || dir >= L->num_neighbors) {
        return 0;
    }
    return L->bond_index[node_id * L->num_neighbors + dir];
}

uint32_t lattice_neighbor_count(const Lattice* L, uint32_t node_id) {
    (void)node_id;  /* All interior nodes have same count for periodic */
    return L ? L->num_neighbors : 0;
}

/* ==========================================================================
 * PACKET INJECTION
 * ========================================================================== */

void lattice_add_packet(Lattice* L,
                        const float* center,
                        float mass,
                        float width,
                        const float* momentum,
                        float initial_q) {
    if (!L || !center || mass <= 0 || width <= 0) return;

    uint32_t dim = L->config.dim;
    float width_sq = width * width;
    float total_weight = 0.0f;

    /* First pass: compute Gaussian weights */
    for (uint32_t i = 0; i < L->num_nodes; i++) {
        int32_t x, y, z;
        lattice_index_to_coord(L, i, &x, &y, &z);

        /* Compute distance squared with periodic wrapping */
        float dist_sq = 0.0f;
        float coords[3] = {(float)x, (float)y, (float)z};

        for (uint32_t d = 0; d < dim; d++) {
            float shape_d = (float)L->config.shape[d];
            float diff = coords[d] - center[d];

            /* Handle periodic boundary */
            if (diff > shape_d / 2) diff -= shape_d;
            if (diff < -shape_d / 2) diff += shape_d;

            dist_sq += diff * diff;
        }

        float weight = expf(-dist_sq / (2.0f * width_sq));
        L->Delta_tau[i] = weight;  /* Temporarily store weight */
        total_weight += weight;
    }

    /* Second pass: inject resource proportional to weight */
    if (total_weight > 0) {
        for (uint32_t i = 0; i < L->num_nodes; i++) {
            float weight = L->Delta_tau[i];
            float added_F = mass * weight / total_weight;
            L->nodes.F[i] += added_F;

            /* Set structure where resource added */
            if (added_F > 0.001f) {
                L->nodes.q[i] = fmaxf(L->nodes.q[i], initial_q * weight / total_weight * 10.0f);
                L->nodes.q[i] = fminf(L->nodes.q[i], 1.0f);
            }
        }
    }

    /* Set momentum on bonds if provided */
    if (momentum) {
        for (uint32_t i = 0; i < L->num_nodes; i++) {
            float weight = L->Delta_tau[i] / (total_weight + 1e-9f);
            if (weight > 0.01f) {
                for (uint32_t d = 0; d < dim; d++) {
                    uint32_t bond_pos = lattice_get_bond(L, i, (LatticeDirection)(d * 2));
                    L->bonds.pi[bond_pos] += momentum[d] * weight * mass;
                }
            }
        }
    }

    /* Reset Delta_tau to dt */
    for (uint32_t i = 0; i < L->num_nodes; i++) {
        L->Delta_tau[i] = L->config.dt;
    }

    /* Track initial mass */
    if (L->step_count == 0) {
        L->total_mass_initial = lattice_total_mass(L);
    }
}

/* ==========================================================================
 * FFT GRAVITY SOLVER
 * ========================================================================== */

#ifdef USE_VDSP_FFT

/** vDSP FFT-based gravity solver (macOS Accelerate) */
void lattice_solve_gravity(Lattice* L) {
    if (!L || !L->physics.gravity_enabled) return;

    uint32_t N = L->num_nodes;
    uint32_t dim = L->config.dim;
    float kappa = L->physics.kappa_grav;
    float mu = L->physics.mu_grav;
    float dx = L->config.dx;

    /* For 1D, use simple vDSP FFT */
    if (dim == 1) {
        uint32_t log2N = 0;
        uint32_t temp = N;
        while (temp > 1) { temp >>= 1; log2N++; }

        /* Setup FFT */
        FFTSetup fftSetup = vDSP_create_fftsetup(log2N, FFT_RADIX2);
        if (!fftSetup) return;

        /* Pack q field into split complex */
        DSPSplitComplex split;
        split.realp = L->fft_workspace;
        split.imagp = L->fft_workspace + N;

        for (uint32_t i = 0; i < N; i++) {
            split.realp[i] = L->nodes.q[i];
            split.imagp[i] = 0.0f;
        }

        /* Forward FFT */
        vDSP_fft_zip(fftSetup, &split, 1, log2N, FFT_FORWARD);

        /* Apply Helmholtz + Poisson kernel in k-space */
        for (uint32_t k = 0; k < N; k++) {
            /* Wave number */
            float kx = (k <= N/2) ? (float)k : (float)k - (float)N;
            kx *= 2.0f * M_PI / ((float)N * dx);

            /* Laplacian in k-space: -k^2 */
            float k_sq = kx * kx;

            /* Helmholtz: 1 / (k^2 + kappa^2) */
            float helm = 1.0f / (k_sq + kappa * kappa + 1e-10f);

            /* Poisson: -mu / k^2 (but we combine: -mu * helm / k^2) */
            float poisson = (k_sq > 1e-10f) ? (-mu / k_sq) : 0.0f;

            /* Combined kernel: psi * poisson */
            float kernel = helm * poisson;

            split.realp[k] *= kernel;
            split.imagp[k] *= kernel;
        }

        /* Inverse FFT */
        vDSP_fft_zip(fftSetup, &split, 1, log2N, FFT_INVERSE);

        /* Scale and copy to phi_field */
        float scale = 1.0f / (float)N;
        for (uint32_t i = 0; i < N; i++) {
            L->phi_field[i] = split.realp[i] * scale;
        }

        /* Compute gradient (central difference) */
        for (uint32_t i = 0; i < N; i++) {
            uint32_t ip = (i + 1) % N;
            uint32_t im = (i + N - 1) % N;
            L->grad_phi[i] = (L->phi_field[ip] - L->phi_field[im]) / (2.0f * dx);
        }

        vDSP_destroy_fftsetup(fftSetup);
    }
    /* 2D/3D would need vDSP_fft2d_zip or manual decomposition */
    else {
        /* Fallback: direct Laplacian solve (slow but works) */
        /* TODO: Implement 2D/3D FFT */

        /* For now, use iterative Jacobi relaxation */
        float* phi_new = L->fft_workspace;
        float* phi_old = L->phi_field;
        memset(phi_old, 0, N * sizeof(float));

        int iterations = 50;
        for (int iter = 0; iter < iterations; iter++) {
            for (uint32_t i = 0; i < N; i++) {
                float sum = 0.0f;
                uint32_t count = 0;

                for (uint32_t d = 0; d < dim; d++) {
                    uint32_t ip = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2));
                    uint32_t im = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2 + 1));
                    sum += phi_old[ip] + phi_old[im];
                    count += 2;
                }

                /* Poisson: nabla^2 phi = -mu * q */
                float source = -mu * L->nodes.q[i] * dx * dx;
                phi_new[i] = (sum + source) / (float)count;
            }

            /* Swap */
            float* temp = phi_old;
            phi_old = phi_new;
            phi_new = temp;
        }

        /* Copy final result */
        if (phi_old != L->phi_field) {
            memcpy(L->phi_field, phi_old, N * sizeof(float));
        }

        /* Compute gradients */
        for (uint32_t i = 0; i < N; i++) {
            for (uint32_t d = 0; d < dim; d++) {
                uint32_t ip = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2));
                uint32_t im = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2 + 1));
                L->grad_phi[i * dim + d] = (L->phi_field[ip] - L->phi_field[im]) / (2.0f * dx);
            }
        }
    }
}

#else
/* Non-Apple fallback using iterative solver */
void lattice_solve_gravity(Lattice* L) {
    /* Same iterative solver as above */
    /* TODO: Add FFTW support for non-Apple platforms */
}
#endif

float lattice_get_phi(const Lattice* L, uint32_t node_id) {
    return (L && node_id < L->num_nodes) ? L->phi_field[node_id] : 0.0f;
}

float lattice_get_grad_phi(const Lattice* L, uint32_t node_id, LatticeDirection dir) {
    if (!L || node_id >= L->num_nodes) return 0.0f;
    uint32_t d = dir / 2;  /* Convert direction to dimension */
    if (d >= L->config.dim) return 0.0f;
    return L->grad_phi[node_id * L->config.dim + d];
}

/* ==========================================================================
 * PHYSICS STEP (DET v6.3)
 * ========================================================================== */

void lattice_step(Lattice* L) {
    if (!L || !L->initialized) return;

    const LatticePhysicsParams* p = &L->physics;
    uint32_t N = L->num_nodes;
    uint32_t dim = L->config.dim;
    float dt = L->config.dt;
    float dx = L->config.dx;

    /* ========== STEP 1: Compute Presence and Proper Time ========== */
    if (p->variable_dt) {
        for (uint32_t i = 0; i < N; i++) {
            float F = L->nodes.F[i];
            float a = L->nodes.a[i];
            float sigma = L->nodes.sigma[i];
            float H = L->phi_field[i];  /* Local curvature */

            /* P = a * sigma / (1 + F) / (1 + |H|) */
            float P = a * sigma / (1.0f + F) / (1.0f + fabsf(H));
            L->nodes.P[i] = P;
            L->Delta_tau[i] = P * dt;
        }
    } else {
        for (uint32_t i = 0; i < N; i++) {
            L->nodes.P[i] = L->nodes.a[i] * L->nodes.sigma[i];
            L->Delta_tau[i] = dt;
        }
    }

    /* ========== STEP 2: Solve Gravity ========== */
    if (p->gravity_enabled) {
        lattice_solve_gravity(L);
    }

    /* ========== STEP 3: Compute Fluxes ========== */
    /* Allocate flux arrays */
    float* J_R = (float*)calloc(N * dim, sizeof(float));  /* +dir flux */
    float* J_L = (float*)calloc(N * dim, sizeof(float));  /* -dir flux */
    if (!J_R || !J_L) {
        free(J_R);
        free(J_L);
        return;
    }

    for (uint32_t i = 0; i < N; i++) {
        float F_i = L->nodes.F[i];

        for (uint32_t d = 0; d < dim; d++) {
            /* Get neighbors */
            uint32_t i_pos = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2));
            uint32_t i_neg = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2 + 1));
            uint32_t bond_pos = lattice_get_bond(L, i, (LatticeDirection)(d * 2));
            uint32_t bond_neg = lattice_get_bond(L, i, (LatticeDirection)(d * 2 + 1));

            float F_pos = L->nodes.F[i_pos];
            float F_neg = L->nodes.F[i_neg];
            float C_pos = L->bonds.C[bond_pos];
            float C_neg = L->bonds.C[bond_neg];
            float pi_pos = L->bonds.pi[bond_pos];
            float pi_neg = L->bonds.pi[bond_neg];

            /* Geometric factor */
            float g = 1.0f / (dx * dx);

            /* ---- Diffusive flux (to +d neighbor) ---- */
            float grad_R = F_i - F_pos;
            float sqrt_C_R = sqrtf(fmaxf(C_pos, 0.01f));
            float drive_R = (1.0f - sqrt_C_R) * grad_R;
            float cond_R = p->sigma * (C_pos + 1e-4f);
            float J_diff_R = g * cond_R * drive_R;

            /* ---- Momentum flux ---- */
            float J_mom_R = g * pi_pos;

            /* ---- Gravity flux ---- */
            float J_grav_R = 0.0f;
            if (p->gravity_enabled && dim == 1) {
                float grad_phi = L->grad_phi[i];
                J_grav_R = -g * F_i * grad_phi;
            } else if (p->gravity_enabled) {
                float grad_phi_d = L->grad_phi[i * dim + d];
                J_grav_R = -g * F_i * grad_phi_d;
            }

            /* ---- Floor flux ---- */
            float J_floor_R = 0.0f;
            if (F_i < p->F_floor) {
                J_floor_R = -g * (p->F_floor - F_i) * 0.1f;
            }

            /* Total flux (positive = i -> i_pos) */
            J_R[i * dim + d] = J_diff_R + J_mom_R + J_grav_R + J_floor_R;

            /* ---- Diffusive flux (to -d neighbor) ---- */
            float grad_L = F_i - F_neg;
            float sqrt_C_L = sqrtf(fmaxf(C_neg, 0.01f));
            float drive_L = (1.0f - sqrt_C_L) * grad_L;
            float cond_L = p->sigma * (C_neg + 1e-4f);
            float J_diff_L = g * cond_L * drive_L;

            float J_mom_L = -g * pi_neg;  /* Negative direction */

            float J_grav_L = 0.0f;
            if (p->gravity_enabled) {
                float grad_phi_d = (dim == 1) ? -L->grad_phi[i] :
                                                 -L->grad_phi[i * dim + d];
                J_grav_L = -g * F_i * grad_phi_d;
            }

            float J_floor_L = 0.0f;
            if (F_i < p->F_floor) {
                J_floor_L = -g * (p->F_floor - F_i) * 0.1f;
            }

            J_L[i * dim + d] = J_diff_L + J_mom_L + J_grav_L + J_floor_L;
        }
    }

    /* ========== STEP 4: Conservative Limiter ========== */
    float* scale = (float*)calloc(N, sizeof(float));
    if (!scale) {
        free(J_R);
        free(J_L);
        return;
    }

    for (uint32_t i = 0; i < N; i++) {
        float total_outflow = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            total_outflow += fmaxf(0.0f, J_R[i * dim + d]);
            total_outflow += fmaxf(0.0f, J_L[i * dim + d]);
        }

        float max_out = p->outflow_limit * L->nodes.F[i] / (L->Delta_tau[i] + 1e-9f);
        scale[i] = fminf(1.0f, max_out / (total_outflow + 1e-9f));
    }

    /* Apply limiter */
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            if (J_R[i * dim + d] > 0) J_R[i * dim + d] *= scale[i];
            if (J_L[i * dim + d] > 0) J_L[i * dim + d] *= scale[i];
        }
    }

    /* ========== STEP 5: Update F (apply flux divergence) ========== */
    float total_dissipation = 0.0f;

    for (uint32_t i = 0; i < N; i++) {
        float dt_i = L->Delta_tau[i];
        float dF = 0.0f;

        for (uint32_t d = 0; d < dim; d++) {
            /* Divergence: incoming - outgoing */
            uint32_t i_pos = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2));
            uint32_t i_neg = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2 + 1));

            /* Incoming from neighbors */
            float in_from_neg = fmaxf(0.0f, J_R[i_neg * dim + d]);  /* neg's +flux to us */
            float in_from_pos = fmaxf(0.0f, J_L[i_pos * dim + d]);  /* pos's -flux to us */

            /* Outgoing from us */
            float out_to_pos = fmaxf(0.0f, J_R[i * dim + d]);
            float out_to_neg = fmaxf(0.0f, J_L[i * dim + d]);

            dF += (in_from_neg + in_from_pos - out_to_pos - out_to_neg) * dt_i;
        }

        float F_new = L->nodes.F[i] + dF;
        if (F_new < 0) {
            total_dissipation += -F_new;
            F_new = 0.0f;
        }
        L->nodes.F[i] = F_new;
    }

    /* ========== STEP 6: Update q (structure) ========== */
    for (uint32_t i = 0; i < N; i++) {
        float dt_i = L->Delta_tau[i];
        float net_outflow = 0.0f;

        for (uint32_t d = 0; d < dim; d++) {
            net_outflow += fmaxf(0.0f, J_R[i * dim + d]);
            net_outflow += fmaxf(0.0f, J_L[i * dim + d]);
        }

        float dq = p->alpha_q * net_outflow * dt_i - p->gamma_q * L->nodes.q[i] * dt_i;
        L->nodes.q[i] = fmaxf(0.0f, fminf(1.0f, L->nodes.q[i] + dq));
    }

    /* ========== STEP 7: Update C and pi (bond dynamics) ========== */
    if (p->momentum_enabled) {
        for (uint32_t b = 0; b < L->num_bonds; b++) {
            uint32_t i = L->bonds.node_i[b];
            uint32_t j = L->bonds.node_j[b];
            float dt_avg = (L->Delta_tau[i] + L->Delta_tau[j]) / 2.0f;

            /* Find which direction this bond represents */
            /* (Simplified: assume bonds are ordered by direction) */
            uint32_t d = b / N;
            if (d >= dim) d = 0;

            float J_diff = J_R[i * dim + d];  /* Flux on this bond */
            float grad_phi_d = (dim == 1) ? L->grad_phi[i] : L->grad_phi[i * dim + d];

            /* Momentum update */
            float pi_old = L->bonds.pi[b];
            float pi_new = pi_old
                         + p->alpha_pi * J_diff * dt_avg
                         - p->lambda_pi * pi_old * dt_avg
                         + p->beta_g * grad_phi_d * dt_avg;
            L->bonds.pi[b] = pi_new;

            /* Coherence update */
            float C_old = L->bonds.C[b];
            float C_new = C_old
                        + p->alpha_C * fabsf(J_diff) * dt_avg
                        - p->lambda_C * C_old * dt_avg;
            L->bonds.C[b] = fmaxf(p->C_init, fminf(1.0f, C_new));
        }
    }

    /* ========== STEP 8: Update a (agency) ========== */
    for (uint32_t i = 0; i < N; i++) {
        float dt_i = L->Delta_tau[i];
        float q_i = L->nodes.q[i];
        float a_old = L->nodes.a[i];

        /* Structural ceiling: a <= 1 - q */
        float ceiling = 1.0f - q_i;

        /* Compute neighbor mean */
        float a_sum = 0.0f;
        uint32_t count = 0;
        for (uint32_t d = 0; d < dim; d++) {
            a_sum += L->nodes.a[lattice_get_neighbor(L, i, (LatticeDirection)(d * 2))];
            a_sum += L->nodes.a[lattice_get_neighbor(L, i, (LatticeDirection)(d * 2 + 1))];
            count += 2;
        }
        float a_mean = a_sum / (float)count;

        /* Target is min of ceiling and neighbor mean */
        float a_target = fminf(ceiling, a_mean);

        /* Relaxation */
        float kappa_a = 0.1f;
        float a_new = a_old + (a_target - a_old) * kappa_a * dt_i;
        L->nodes.a[i] = fmaxf(0.0f, fminf(1.0f, a_new));
    }

    /* ========== STEP 9: Grace injection ========== */
    if (p->grace_enabled && total_dissipation > 0) {
        float total_need = 0.0f;
        for (uint32_t i = 0; i < N; i++) {
            float need = fmaxf(0.0f, p->F_MIN_grace - L->nodes.F[i]);
            total_need += L->nodes.a[i] * need;
        }

        if (total_need > 0) {
            for (uint32_t i = 0; i < N; i++) {
                float need = fmaxf(0.0f, p->F_MIN_grace - L->nodes.F[i]);
                float w = L->nodes.a[i] * need;
                float grace = total_dissipation * w / total_need;
                L->nodes.F[i] += grace;
            }
        }
    }

    /* ========== STEP 10: Update proper time ========== */
    for (uint32_t i = 0; i < N; i++) {
        L->nodes.tau[i] += L->Delta_tau[i];
    }

    /* Cleanup */
    free(J_R);
    free(J_L);
    free(scale);

    L->step_count++;
    L->total_mass_current = lattice_total_mass(L);
}

void lattice_step_n(Lattice* L, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        lattice_step(L);
    }
}

/* ==========================================================================
 * STATISTICS
 * ========================================================================== */

float lattice_total_mass(const Lattice* L) {
    if (!L) return 0.0f;
    float total = 0.0f;
    for (uint32_t i = 0; i < L->num_nodes; i++) {
        total += L->nodes.F[i];
    }
    return total;
}

void lattice_center_of_mass(const Lattice* L, float* com) {
    if (!L || !com) return;

    memset(com, 0, sizeof(float) * LATTICE_MAX_DIM);
    float total_mass = 0.0f;

    for (uint32_t i = 0; i < L->num_nodes; i++) {
        float F = L->nodes.F[i];
        int32_t x, y, z;
        lattice_index_to_coord(L, i, &x, &y, &z);

        com[0] += (float)x * F;
        if (L->config.dim >= 2) com[1] += (float)y * F;
        if (L->config.dim >= 3) com[2] += (float)z * F;
        total_mass += F;
    }

    if (total_mass > 0) {
        for (uint32_t d = 0; d < L->config.dim; d++) {
            com[d] /= total_mass;
        }
    }
}

float lattice_separation(const Lattice* L) {
    if (!L || L->num_nodes < 2) return 0.0f;

    /* Find two largest mass concentrations */
    uint32_t max1_idx = 0, max2_idx = 1;
    float max1_F = L->nodes.F[0], max2_F = L->nodes.F[1];

    if (max2_F > max1_F) {
        uint32_t tmp_i = max1_idx; max1_idx = max2_idx; max2_idx = tmp_i;
        float tmp_f = max1_F; max1_F = max2_F; max2_F = tmp_f;
    }

    for (uint32_t i = 2; i < L->num_nodes; i++) {
        float F = L->nodes.F[i];
        if (F > max1_F) {
            max2_idx = max1_idx; max2_F = max1_F;
            max1_idx = i; max1_F = F;
        } else if (F > max2_F) {
            max2_idx = i; max2_F = F;
        }
    }

    /* Compute distance */
    int32_t x1, y1, z1, x2, y2, z2;
    lattice_index_to_coord(L, max1_idx, &x1, &y1, &z1);
    lattice_index_to_coord(L, max2_idx, &x2, &y2, &z2);

    float dx = (float)(x2 - x1);
    float dy = (float)(y2 - y1);
    float dz = (float)(z2 - z1);

    /* Handle periodic boundaries */
    if (fabsf(dx) > L->config.shape[0] / 2.0f) {
        dx = (dx > 0) ? dx - L->config.shape[0] : dx + L->config.shape[0];
    }
    if (L->config.dim >= 2 && fabsf(dy) > L->config.shape[1] / 2.0f) {
        dy = (dy > 0) ? dy - L->config.shape[1] : dy + L->config.shape[1];
    }
    if (L->config.dim >= 3 && fabsf(dz) > L->config.shape[2] / 2.0f) {
        dz = (dz > 0) ? dz - L->config.shape[2] : dz + L->config.shape[2];
    }

    return sqrtf(dx*dx + dy*dy + dz*dz) * L->config.dx;
}

float lattice_kinetic_energy(const Lattice* L) {
    if (!L) return 0.0f;
    float total = 0.0f;
    for (uint32_t b = 0; b < L->num_bonds; b++) {
        float pi = L->bonds.pi[b];
        total += 0.5f * pi * pi;
    }
    return total;
}

float lattice_potential_energy(const Lattice* L) {
    if (!L) return 0.0f;
    float total = 0.0f;
    for (uint32_t i = 0; i < L->num_nodes; i++) {
        total += L->nodes.F[i] * L->phi_field[i];
    }
    return 0.5f * total;  /* Factor of 1/2 to avoid double counting */
}

void lattice_get_stats(const Lattice* L, LatticeStats* stats) {
    if (!L || !stats) return;

    memset(stats, 0, sizeof(LatticeStats));
    stats->total_mass = lattice_total_mass(L);

    float total_q = 0.0f, total_pi = 0.0f;
    for (uint32_t i = 0; i < L->num_nodes; i++) {
        total_q += L->nodes.q[i];
    }
    for (uint32_t b = 0; b < L->num_bonds; b++) {
        total_pi += fabsf(L->bonds.pi[b]);
    }

    stats->total_structure = total_q;
    stats->total_momentum = total_pi;
    stats->kinetic_energy = lattice_kinetic_energy(L);
    stats->potential_energy = lattice_potential_energy(L);
    stats->separation = lattice_separation(L);
    lattice_center_of_mass(L, stats->com);
    stats->step_count = L->step_count;
}

/* ==========================================================================
 * RENDERING
 * ========================================================================== */

uint32_t lattice_render(const Lattice* L, RenderField field,
                        uint32_t width, char* out, uint32_t out_size) {
    if (!L || !out || out_size == 0) return 0;

    const char* chars = " .:;+=xX#@";
    int num_chars = 10;

    /* Select field array */
    const float* data = NULL;
    switch (field) {
        case RENDER_FIELD_F: data = L->nodes.F; break;
        case RENDER_FIELD_Q: data = L->nodes.q; break;
        case RENDER_FIELD_A: data = L->nodes.a; break;
        case RENDER_FIELD_P: data = L->nodes.P; break;
        case RENDER_FIELD_PHI: data = L->phi_field; break;
        default: data = L->nodes.F; break;
    }

    /* Find min/max */
    float min_val = data[0], max_val = data[0];
    for (uint32_t i = 1; i < L->num_nodes; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    float range = max_val - min_val;
    if (range < 1e-9f) range = 1.0f;

    uint32_t pos = 0;
    uint32_t dim = L->config.dim;

    if (dim == 1) {
        /* 1D: Single line */
        uint32_t nx = L->config.shape[0];
        float scale = (float)nx / (float)width;

        for (uint32_t w = 0; w < width && pos < out_size - 1; w++) {
            /* Average over nodes in this bin */
            uint32_t x_start = (uint32_t)(w * scale);
            uint32_t x_end = (uint32_t)((w + 1) * scale);
            if (x_end > nx) x_end = nx;
            if (x_start >= x_end) x_start = x_end - 1;

            float sum = 0.0f;
            for (uint32_t x = x_start; x < x_end; x++) {
                sum += data[x];
            }
            float avg = sum / (float)(x_end - x_start);

            int idx = (int)((avg - min_val) / range * (num_chars - 1));
            if (idx < 0) idx = 0;
            if (idx >= num_chars) idx = num_chars - 1;
            out[pos++] = chars[idx];
        }
    } else if (dim == 2) {
        /* 2D: Multiple lines */
        uint32_t nx = L->config.shape[0];
        uint32_t ny = L->config.shape[1];
        uint32_t height = width * ny / nx;
        if (height < 1) height = 1;

        float scale_x = (float)nx / (float)width;
        float scale_y = (float)ny / (float)height;

        for (uint32_t h = 0; h < height && pos < out_size - 2; h++) {
            for (uint32_t w = 0; w < width && pos < out_size - 2; w++) {
                uint32_t x = (uint32_t)(w * scale_x);
                uint32_t y = (uint32_t)(h * scale_y);
                if (x >= nx) x = nx - 1;
                if (y >= ny) y = ny - 1;

                float val = data[y * nx + x];
                int idx = (int)((val - min_val) / range * (num_chars - 1));
                if (idx < 0) idx = 0;
                if (idx >= num_chars) idx = num_chars - 1;
                out[pos++] = chars[idx];
            }
            out[pos++] = '\n';
        }
    }

    out[pos] = '\0';
    return pos;
}

/* ==========================================================================
 * PARAMETER CONTROL
 * ========================================================================== */

bool lattice_set_param(Lattice* L, const char* param_name, float value) {
    if (!L || !param_name) return false;

    LatticePhysicsParams* p = &L->physics;

    if (strcmp(param_name, "sigma") == 0) { p->sigma = value; return true; }
    if (strcmp(param_name, "outflow_limit") == 0) { p->outflow_limit = value; return true; }
    if (strcmp(param_name, "F_floor") == 0) { p->F_floor = value; return true; }
    if (strcmp(param_name, "kappa_grav") == 0) { p->kappa_grav = value; return true; }
    if (strcmp(param_name, "mu_grav") == 0) { p->mu_grav = value; return true; }
    if (strcmp(param_name, "beta_g") == 0) { p->beta_g = value; return true; }
    if (strcmp(param_name, "alpha_pi") == 0) { p->alpha_pi = value; return true; }
    if (strcmp(param_name, "lambda_pi") == 0) { p->lambda_pi = value; return true; }
    if (strcmp(param_name, "alpha_C") == 0) { p->alpha_C = value; return true; }
    if (strcmp(param_name, "lambda_C") == 0) { p->lambda_C = value; return true; }
    if (strcmp(param_name, "C_init") == 0) { p->C_init = value; return true; }
    if (strcmp(param_name, "alpha_q") == 0) { p->alpha_q = value; return true; }
    if (strcmp(param_name, "gamma_q") == 0) { p->gamma_q = value; return true; }
    if (strcmp(param_name, "F_MIN_grace") == 0) { p->F_MIN_grace = value; return true; }
    if (strcmp(param_name, "gravity_enabled") == 0) { p->gravity_enabled = value > 0.5f; return true; }
    if (strcmp(param_name, "momentum_enabled") == 0) { p->momentum_enabled = value > 0.5f; return true; }
    if (strcmp(param_name, "grace_enabled") == 0) { p->grace_enabled = value > 0.5f; return true; }
    if (strcmp(param_name, "variable_dt") == 0) { p->variable_dt = value > 0.5f; return true; }
    if (strcmp(param_name, "dt") == 0) { L->config.dt = value; return true; }
    if (strcmp(param_name, "dx") == 0) { L->config.dx = value; return true; }

    return false;
}

float lattice_get_param(const Lattice* L, const char* param_name) {
    if (!L || !param_name) return 0.0f;

    const LatticePhysicsParams* p = &L->physics;

    if (strcmp(param_name, "sigma") == 0) return p->sigma;
    if (strcmp(param_name, "outflow_limit") == 0) return p->outflow_limit;
    if (strcmp(param_name, "F_floor") == 0) return p->F_floor;
    if (strcmp(param_name, "kappa_grav") == 0) return p->kappa_grav;
    if (strcmp(param_name, "mu_grav") == 0) return p->mu_grav;
    if (strcmp(param_name, "beta_g") == 0) return p->beta_g;
    if (strcmp(param_name, "alpha_pi") == 0) return p->alpha_pi;
    if (strcmp(param_name, "lambda_pi") == 0) return p->lambda_pi;
    if (strcmp(param_name, "alpha_C") == 0) return p->alpha_C;
    if (strcmp(param_name, "lambda_C") == 0) return p->lambda_C;
    if (strcmp(param_name, "C_init") == 0) return p->C_init;
    if (strcmp(param_name, "alpha_q") == 0) return p->alpha_q;
    if (strcmp(param_name, "gamma_q") == 0) return p->gamma_q;
    if (strcmp(param_name, "F_MIN_grace") == 0) return p->F_MIN_grace;
    if (strcmp(param_name, "gravity_enabled") == 0) return p->gravity_enabled ? 1.0f : 0.0f;
    if (strcmp(param_name, "momentum_enabled") == 0) return p->momentum_enabled ? 1.0f : 0.0f;
    if (strcmp(param_name, "grace_enabled") == 0) return p->grace_enabled ? 1.0f : 0.0f;
    if (strcmp(param_name, "variable_dt") == 0) return p->variable_dt ? 1.0f : 0.0f;
    if (strcmp(param_name, "dt") == 0) return L->config.dt;
    if (strcmp(param_name, "dx") == 0) return L->config.dx;

    return 0.0f;
}

/* ==========================================================================
 * SUBSTRATE INTEGRATION
 * ========================================================================== */

void lattice_bind_to_vm(Lattice* L, SubstrateVM* vm) {
    if (!L || !vm) return;

    /* Point VM's node/bond arrays to lattice's arrays */
    vm->nodes = &L->nodes;
    vm->bonds = &L->bonds;
}

void lattice_sync_from_vm(Lattice* L, SubstrateVM* vm) {
    /* Currently no-op since we share pointers */
    /* Would be needed if VM had separate copies */
    (void)L;
    (void)vm;
}
