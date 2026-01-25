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
#include <os/lock.h>
#define USE_OS_LOCK 1
#else
#include <pthread.h>
#endif

/* ==========================================================================
 * LATTICE REGISTRY (Fix C11: Thread-safe via mutex)
 * ========================================================================== */

#define MAX_LATTICES 64

static Lattice* g_lattice_registry[MAX_LATTICES] = {0};
static uint32_t g_next_lattice_id = 1;
static bool g_registry_initialized = false;

#ifdef USE_OS_LOCK
static os_unfair_lock g_registry_lock = OS_UNFAIR_LOCK_INIT;
#define REGISTRY_LOCK()   os_unfair_lock_lock(&g_registry_lock)
#define REGISTRY_UNLOCK() os_unfair_lock_unlock(&g_registry_lock)
#else
static pthread_mutex_t g_registry_mutex = PTHREAD_MUTEX_INITIALIZER;
#define REGISTRY_LOCK()   pthread_mutex_lock(&g_registry_mutex)
#define REGISTRY_UNLOCK() pthread_mutex_unlock(&g_registry_mutex)
#endif

void lattice_registry_init(void) {
    REGISTRY_LOCK();
    if (g_registry_initialized) {
        REGISTRY_UNLOCK();
        return;
    }
    memset(g_lattice_registry, 0, sizeof(g_lattice_registry));
    g_next_lattice_id = 1;
    g_registry_initialized = true;
    REGISTRY_UNLOCK();
}

void lattice_registry_shutdown(void) {
    REGISTRY_LOCK();
    for (int i = 0; i < MAX_LATTICES; i++) {
        if (g_lattice_registry[i]) {
            /* Note: lattice_destroy also removes from registry, so just destroy */
            Lattice* L = g_lattice_registry[i];
            g_lattice_registry[i] = NULL;
            REGISTRY_UNLOCK();
            lattice_destroy(L);
            REGISTRY_LOCK();
        }
    }
    g_registry_initialized = false;
    REGISTRY_UNLOCK();
}

uint32_t lattice_registry_add(Lattice* L) {
    REGISTRY_LOCK();
    if (!g_registry_initialized) {
        memset(g_lattice_registry, 0, sizeof(g_lattice_registry));
        g_next_lattice_id = 1;
        g_registry_initialized = true;
    }

    for (int i = 0; i < MAX_LATTICES; i++) {
        if (g_lattice_registry[i] == NULL) {
            g_lattice_registry[i] = L;
            L->id = g_next_lattice_id++;
            REGISTRY_UNLOCK();
            return L->id;
        }
    }
    REGISTRY_UNLOCK();
    return 0;  /* Registry full */
}

Lattice* lattice_registry_get(uint32_t id) {
    REGISTRY_LOCK();
    for (int i = 0; i < MAX_LATTICES; i++) {
        if (g_lattice_registry[i] && g_lattice_registry[i]->id == id) {
            Lattice* L = g_lattice_registry[i];
            REGISTRY_UNLOCK();
            return L;
        }
    }
    REGISTRY_UNLOCK();
    return NULL;
}

void lattice_registry_remove(uint32_t id) {
    REGISTRY_LOCK();
    for (int i = 0; i < MAX_LATTICES; i++) {
        if (g_lattice_registry[i] && g_lattice_registry[i]->id == id) {
            g_lattice_registry[i] = NULL;
            REGISTRY_UNLOCK();
            return;
        }
    }
    REGISTRY_UNLOCK();
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

/** Allocate bond metadata (Fix A6: bond_dim stored at creation) */
static bool alloc_bond_metadata(Lattice* L, uint32_t count) {
    L->bond_dim = (uint8_t*)calloc(count, sizeof(uint8_t));
    return L->bond_dim != NULL;
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

    /* Allocate bond metadata (Fix A6) */
    if (!alloc_bond_metadata(L, bond_count)) {
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

            /* Create bond for positive direction (Fix A6: store dimension) */
            uint32_t neighbor_id = (uint32_t)((int32_t)node_id + offset_pos);
            L->bonds.node_i[bond_idx] = node_id;
            L->bonds.node_j[bond_idx] = neighbor_id;
            L->bond_dim[bond_idx] = (uint8_t)d;  /* Store actual dimension */
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

    /* Allocate scratch arrays (Fix B8: pre-allocate all scratch buffers) */
    L->Delta_tau = (float*)calloc(L->num_nodes, sizeof(float));
    L->J_total = (float*)calloc(L->num_nodes * config->dim, sizeof(float));
    L->grad_phi = (float*)calloc(L->num_nodes * config->dim, sizeof(float));
    L->J_flux = (float*)calloc(L->num_bonds, sizeof(float));    /* Fix B10: per-bond flux */
    L->scale = (float*)calloc(L->num_nodes, sizeof(float));     /* Pre-allocated limiter scale */
    L->temp_weights = (float*)calloc(L->num_nodes, sizeof(float)); /* Fix C12: separate temp array */

    if (!L->fft_workspace || !L->psi_field || !L->phi_field ||
        !L->Delta_tau || !L->J_total || !L->grad_phi ||
        !L->J_flux || !L->scale || !L->temp_weights) {
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
    free(L->bond_dim);       /* Fix A6 */
    free(L->fft_workspace);
    free(L->psi_field);
    free(L->phi_field);
    free(L->Delta_tau);
    free(L->J_total);
    free(L->grad_phi);
    free(L->J_flux);         /* Fix B8/B10 */
    free(L->scale);          /* Fix B8 */
    free(L->temp_weights);   /* Fix C12 */

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

    /* First pass: compute Gaussian weights (Fix C12: use temp_weights, not Delta_tau) */
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
        L->temp_weights[i] = weight;  /* Use dedicated temp array */
        total_weight += weight;
    }

    /* Second pass: inject resource proportional to weight */
    if (total_weight > 0) {
        for (uint32_t i = 0; i < L->num_nodes; i++) {
            float weight = L->temp_weights[i];
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
            float weight = L->temp_weights[i] / (total_weight + 1e-9f);
            if (weight > 0.01f) {
                for (uint32_t d = 0; d < dim; d++) {
                    uint32_t bond_pos = lattice_get_bond(L, i, (LatticeDirection)(d * 2));
                    L->bonds.pi[bond_pos] += momentum[d] * weight * mass;
                }
            }
        }
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
    /* 2D/3D: Iterative Jacobi relaxation (Fix A2: include kappa_grav) */
    else {
        /* Helmholtz + Poisson via Jacobi iteration
         * Solving: (nabla^2 - kappa^2) psi = q, then nabla^2 phi = -mu * psi
         * Combined: nabla^2 phi = -mu * q / (1 + kappa^2 * dx^2 / (2*dim))
         * Using modified Helmholtz-Poisson kernel to match 1D FFT equation
         */
        float* phi_new = L->fft_workspace;
        float* phi_old = L->phi_field;
        memset(phi_old, 0, N * sizeof(float));

        /* Precompute Helmholtz denominator: includes kappa^2 for screening */
        float kappa_sq_dx_sq = kappa * kappa * dx * dx;
        float denom = 2.0f * (float)dim + kappa_sq_dx_sq;

        int iterations = 100;  /* More iterations for better convergence */
        for (int iter = 0; iter < iterations; iter++) {
            for (uint32_t i = 0; i < N; i++) {
                float sum = 0.0f;

                for (uint32_t d = 0; d < dim; d++) {
                    uint32_t ip = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2));
                    uint32_t im = lattice_get_neighbor(L, i, (LatticeDirection)(d * 2 + 1));
                    sum += phi_old[ip] + phi_old[im];
                }

                /* Helmholtz+Poisson: (nabla^2 - kappa^2) phi = -mu * q
                 * Discretized: (sum - 2*dim*phi)/dx^2 - kappa^2*phi = -mu*q
                 * Rearranged: phi = (sum + mu*q*dx^2) / (2*dim + kappa^2*dx^2)
                 */
                float source = mu * L->nodes.q[i] * dx * dx;
                phi_new[i] = (sum + source) / denom;
            }

            /* Swap buffers */
            float* temp = phi_old;
            phi_old = phi_new;
            phi_new = temp;
        }

        /* Copy final result if needed */
        if (phi_old != L->phi_field) {
            memcpy(L->phi_field, phi_old, N * sizeof(float));
        }

        /* Compute gradients (central difference) */
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
    uint32_t B = L->num_bonds;
    uint32_t dim = L->config.dim;
    float dt = L->config.dt;
    float dx = L->config.dx;
    float g = 1.0f / (dx * dx);  /* Geometric factor (Fix B9: compute once) */

    /* ========== STEP 1: Solve Gravity FIRST (Fix A1: solve before P) ========== */
    if (p->gravity_enabled) {
        lattice_solve_gravity(L);
    }

    /* ========== STEP 2: Compute Presence and Proper Time (uses fresh phi) ========== */
    if (p->variable_dt) {
        for (uint32_t i = 0; i < N; i++) {
            float F = L->nodes.F[i];
            float a = L->nodes.a[i];
            float sigma = L->nodes.sigma[i];
            float H = L->phi_field[i];  /* Now using fresh curvature! */

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

    /* ========== STEP 3: Compute Per-Bond Fluxes (Fix A3, B10: one flux per bond) ========== */
    /* Using pre-allocated L->J_flux array (Fix B8) */
    float* J = L->J_flux;
    memset(J, 0, B * sizeof(float));

    /* Also store diffusive-only flux for momentum update (Fix A5) */
    float* J_diff_only = L->temp_weights;  /* Reuse temp array */
    memset(J_diff_only, 0, B * sizeof(float));

    for (uint32_t b = 0; b < B; b++) {
        uint32_t i = L->bonds.node_i[b];
        uint32_t j = L->bonds.node_j[b];
        uint8_t d = L->bond_dim[b];  /* Fix A6: use stored dimension */

        float F_i = L->nodes.F[i];
        float F_j = L->nodes.F[j];
        float C_b = L->bonds.C[b];
        float pi_b = L->bonds.pi[b];

        /* Fix B9: precompute sqrt_C */
        float sqrt_C = sqrtf(fmaxf(C_b, 0.01f));
        float cond = p->sigma * (C_b + 1e-4f);

        /* ---- Diffusive flux: i -> j (Fix A3: compute once per bond) ---- */
        float grad = F_i - F_j;
        float drive = (1.0f - sqrt_C) * grad;
        float J_diff = g * cond * drive;
        J_diff_only[b] = J_diff;  /* Store for momentum update (Fix A5) */

        /* ---- Momentum flux: positive pi means flow i->j ---- */
        float J_mom = g * pi_b;

        /* ---- Gravity flux (Fix A7: bond-centered using avg gradient) ---- */
        float J_grav = 0.0f;
        if (p->gravity_enabled) {
            float grad_phi_i = (dim == 1) ? L->grad_phi[i] : L->grad_phi[i * dim + d];
            float grad_phi_j = (dim == 1) ? L->grad_phi[j] : L->grad_phi[j * dim + d];
            /* Bond-centered gradient and resource */
            float grad_phi_bond = (grad_phi_i + grad_phi_j) * 0.5f;
            float F_bond = (F_i + F_j) * 0.5f;
            J_grav = -g * F_bond * grad_phi_bond;
        }

        /* ---- Floor flux (bidirectional stabilizer) ---- */
        float J_floor = 0.0f;
        if (F_i < p->F_floor && F_j > F_i) {
            J_floor = -g * (p->F_floor - F_i) * 0.1f;  /* Pull from j */
        } else if (F_j < p->F_floor && F_i > F_j) {
            J_floor = g * (p->F_floor - F_j) * 0.1f;   /* Push to j */
        }

        /* Total flux on bond: positive = i -> j (Fix A3: single value) */
        J[b] = J_diff + J_mom + J_grav + J_floor;
    }

    /* ========== STEP 4: Conservative Limiter (Fix A4: check both donor AND receiver) ========== */
    /* Using pre-allocated L->scale array (Fix B8) */
    float* scale = L->scale;

    /* First pass: compute donor-side limits */
    for (uint32_t i = 0; i < N; i++) {
        scale[i] = 1.0f;
    }

    /* Accumulate outflow per node */
    for (uint32_t b = 0; b < B; b++) {
        uint32_t i = L->bonds.node_i[b];
        uint32_t j = L->bonds.node_j[b];
        float flux = J[b];

        if (flux > 0) {
            /* Outflow from i */
            float max_out_i = p->outflow_limit * L->nodes.F[i] / (L->Delta_tau[i] + 1e-9f);
            float scale_i = fminf(1.0f, max_out_i / (flux + 1e-9f));
            scale[i] = fminf(scale[i], scale_i);
        } else if (flux < 0) {
            /* Outflow from j */
            float max_out_j = p->outflow_limit * L->nodes.F[j] / (L->Delta_tau[j] + 1e-9f);
            float scale_j = fminf(1.0f, max_out_j / (-flux + 1e-9f));
            scale[j] = fminf(scale[j], scale_j);
        }
    }

    /* Fix A4: Second pass - check receiver capacity (prevent overflow) */
    /* This is a simplified check; full receiver limiting would need iteration */
    /* For now, we allow receivers to accept any amount (DET allows unbounded F) */

    /* Apply limiter to fluxes */
    for (uint32_t b = 0; b < B; b++) {
        uint32_t i = L->bonds.node_i[b];
        uint32_t j = L->bonds.node_j[b];
        float flux = J[b];

        if (flux > 0) {
            J[b] *= scale[i];
            J_diff_only[b] *= scale[i];
        } else if (flux < 0) {
            J[b] *= scale[j];
            J_diff_only[b] *= scale[j];
        }
    }

    /* ========== STEP 5: Update F (apply flux divergence, Fix D14: antisymmetric) ========== */
    float total_dissipation = 0.0f;

    for (uint32_t b = 0; b < B; b++) {
        uint32_t i = L->bonds.node_i[b];
        uint32_t j = L->bonds.node_j[b];
        float flux = J[b];
        float dt_avg = (L->Delta_tau[i] + L->Delta_tau[j]) * 0.5f;
        float dF = flux * dt_avg;

        /* Antisymmetric update: what leaves i arrives at j (Fix A3, D14) */
        L->nodes.F[i] -= dF;
        L->nodes.F[j] += dF;
    }

    /* Clamp negative F and track dissipation */
    for (uint32_t i = 0; i < N; i++) {
        if (L->nodes.F[i] < 0) {
            total_dissipation += -L->nodes.F[i];
            L->nodes.F[i] = 0.0f;
        }
    }

    /* ========== STEP 6: Update q (structure) ========== */
    for (uint32_t b = 0; b < B; b++) {
        uint32_t i = L->bonds.node_i[b];
        uint32_t j = L->bonds.node_j[b];
        float flux = J[b];
        float dt_avg = (L->Delta_tau[i] + L->Delta_tau[j]) * 0.5f;

        /* Outflow builds structure at source */
        if (flux > 0) {
            L->nodes.q[i] += p->alpha_q * flux * dt_avg;
        } else {
            L->nodes.q[j] += p->alpha_q * (-flux) * dt_avg;
        }
    }

    /* Decay and clamp q */
    for (uint32_t i = 0; i < N; i++) {
        float dt_i = L->Delta_tau[i];
        L->nodes.q[i] -= p->gamma_q * L->nodes.q[i] * dt_i;
        L->nodes.q[i] = fmaxf(0.0f, fminf(1.0f, L->nodes.q[i]));
    }

    /* ========== STEP 7: Update C and pi (bond dynamics) ========== */
    if (p->momentum_enabled) {
        for (uint32_t b = 0; b < B; b++) {
            uint32_t i = L->bonds.node_i[b];
            uint32_t j = L->bonds.node_j[b];
            uint8_t d = L->bond_dim[b];  /* Fix A6: use stored dimension */
            float dt_avg = (L->Delta_tau[i] + L->Delta_tau[j]) * 0.5f;

            /* Fix A5: Use diffusive-only flux for momentum charging */
            float J_diff = J_diff_only[b];

            /* Bond-centered gravity gradient (Fix A7) */
            float grad_phi_i = (dim == 1) ? L->grad_phi[i] : L->grad_phi[i * dim + d];
            float grad_phi_j = (dim == 1) ? L->grad_phi[j] : L->grad_phi[j * dim + d];
            float grad_phi_bond = (grad_phi_i + grad_phi_j) * 0.5f;

            /* Momentum update */
            float pi_old = L->bonds.pi[b];
            float pi_new = pi_old
                         + p->alpha_pi * J_diff * dt_avg
                         - p->lambda_pi * pi_old * dt_avg
                         + p->beta_g * grad_phi_bond * dt_avg;
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

    L->step_count++;
    L->total_mass_current = lattice_total_mass(L);
}

void lattice_step_n(Lattice* L, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        lattice_step(L);
    }
}

/**
 * Fused step interface for batching (Fix D13)
 *
 * Executes n steps with minimal overhead, optionally recording flux snapshots.
 * Returns total flux magnitude for witnessable output.
 */
float lattice_step_fused(Lattice* L, uint32_t n, float* out_fluxes, uint32_t emit_every) {
    if (!L || !L->initialized || n == 0) return 0.0f;

    float total_flux_magnitude = 0.0f;
    uint32_t B = L->num_bonds;

    for (uint32_t step = 0; step < n; step++) {
        /* Execute physics step */
        lattice_step(L);

        /* Accumulate flux magnitude for witnessing */
        for (uint32_t b = 0; b < B; b++) {
            total_flux_magnitude += fabsf(L->J_flux[b]);
        }

        /* Emit flux snapshot if requested (Fix D14: stable, witnessable output) */
        if (out_fluxes && emit_every > 0 && ((step + 1) % emit_every == 0)) {
            uint32_t snapshot_idx = step / emit_every;
            float* snapshot = out_fluxes + snapshot_idx * B;

            /* Copy current flux array (already antisymmetric from per-bond computation) */
            memcpy(snapshot, L->J_flux, B * sizeof(float));
        }
    }

    return total_flux_magnitude;
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
