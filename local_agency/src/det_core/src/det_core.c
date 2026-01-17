/**
 * DET Local Agency - C Kernel Implementation
 * ==========================================
 *
 * Core DET dynamics: presence, coherence, agency, affect.
 * Cluster-centric Self identification.
 * Gatekeeper decision logic.
 */

#include "det_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ==========================================================================
 * INTERNAL HELPERS
 * ========================================================================== */

static inline float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline float maxf(float a, float b) {
    return a > b ? a : b;
}

static inline float minf(float a, float b) {
    return a < b ? a : b;
}

/** Initialize agency distribution: Beta(2,5) + 5% reserved high-a pool */
static void init_dormant_pool_agency(DETCore* core) {
    /* Simple approximation of Beta(2,5): mean ≈ 0.286 */
    /* Using linear combination for reproducibility without random */
    uint32_t dormant_start = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE;
    uint32_t reserved_count = DET_DORMANT_SIZE / 20;  /* 5% */
    uint32_t main_count = DET_DORMANT_SIZE - reserved_count;

    for (uint32_t i = 0; i < main_count; i++) {
        uint32_t idx = dormant_start + i;
        /* Deterministic "random" based on index */
        float t = (float)(i % 100) / 100.0f;
        /* Approximate Beta(2,5) CDF inverse */
        core->nodes[idx].a = 0.1f + 0.25f * t * (1.0f - 0.3f * t);
    }

    /* Reserved high-a pool */
    for (uint32_t i = 0; i < reserved_count; i++) {
        uint32_t idx = dormant_start + main_count + i;
        float t = (float)i / (float)reserved_count;
        core->nodes[idx].a = 0.85f + 0.10f * t;  /* [0.85, 0.95] */
    }
}

/** Get bond parameters based on layer types */
static void get_bond_params(
    const DETCore* core,
    DETLayer layer_i,
    DETLayer layer_j,
    float* alpha,
    float* lambda,
    float* slip
) {
    if (layer_i == DET_LAYER_A && layer_j == DET_LAYER_A) {
        *alpha = core->params.alpha_AA;
        *lambda = core->params.lambda_AA;
        *slip = core->params.slip_AA;
    } else if (layer_i == DET_LAYER_P && layer_j == DET_LAYER_P) {
        *alpha = core->params.alpha_PP;
        *lambda = core->params.lambda_PP;
        *slip = core->params.slip_PP;
    } else {
        /* Cross-layer or involving ports */
        *alpha = core->params.alpha_PA;
        *lambda = core->params.lambda_PA;
        *slip = core->params.slip_PA;
    }
}

/* ==========================================================================
 * DEFAULT PARAMETERS
 * ========================================================================== */

DETParams det_default_params(void) {
    DETParams p = {
        /* Core DET physics */
        .tau_base = 0.02f,
        .sigma_base = 0.12f,
        .lambda_base = 0.008f,
        .mu_base = 2.0f,
        .kappa_base = 5.0f,
        .C_0 = 0.15f,
        .lambda_a = 30.0f,
        .phi_L = 0.5f,
        .pi_max = 3.0f,

        /* A↔A: fast, plastic */
        .alpha_AA = 0.15f,
        .lambda_AA = 0.08f,
        .slip_AA = 0.02f,

        /* P↔P: slow, stable */
        .alpha_PP = 0.03f,
        .lambda_PP = 0.01f,
        .slip_PP = 0.01f,

        /* P↔A: medium, phase-sensitive */
        .alpha_PA = 0.08f,
        .lambda_PA = 0.04f,
        .slip_PA = 0.06f
    };
    return p;
}

/* ==========================================================================
 * LIFECYCLE
 * ========================================================================== */

DETCore* det_core_create(void) {
    DETParams params = det_default_params();
    return det_core_create_with_params(&params);
}

DETCore* det_core_create_with_params(const DETParams* params) {
    DETCore* core = (DETCore*)calloc(1, sizeof(DETCore));
    if (!core) return NULL;

    core->params = *params;

    /* Initialize nodes */
    core->num_nodes = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE + DET_DORMANT_SIZE;

    /* P-layer nodes (first DET_P_LAYER_SIZE) */
    for (uint32_t i = 0; i < DET_P_LAYER_SIZE; i++) {
        core->nodes[i].layer = DET_LAYER_P;
        core->nodes[i].active = true;
        core->nodes[i].a = 0.7f + 0.2f * ((float)i / DET_P_LAYER_SIZE);
        core->nodes[i].F = 1.0f;
        core->nodes[i].sigma = params->sigma_base;
        core->nodes[i].theta = 2.0f * 3.14159f * i / DET_P_LAYER_SIZE;
    }

    /* A-layer nodes */
    for (uint32_t i = 0; i < DET_A_LAYER_SIZE; i++) {
        uint32_t idx = DET_P_LAYER_SIZE + i;
        core->nodes[idx].layer = DET_LAYER_A;
        core->nodes[idx].active = true;
        core->nodes[idx].a = 0.3f + 0.2f * ((float)(i % 50) / 50.0f);
        core->nodes[idx].F = 0.5f;
        core->nodes[idx].sigma = params->sigma_base;
        core->nodes[idx].domain = i / (DET_A_LAYER_SIZE / 4);  /* 4 domains */
    }

    /* Dormant pool */
    for (uint32_t i = 0; i < DET_DORMANT_SIZE; i++) {
        uint32_t idx = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE + i;
        core->nodes[idx].layer = DET_LAYER_DORMANT;
        core->nodes[idx].active = false;
        core->nodes[idx].F = 0.0f;
        core->nodes[idx].sigma = 0.0f;
    }
    init_dormant_pool_agency(core);

    core->num_active = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE;

    /* Initialize some bonds within P-layer (ring topology) */
    for (uint32_t i = 0; i < DET_P_LAYER_SIZE; i++) {
        uint32_t j = (i + 1) % DET_P_LAYER_SIZE;
        int32_t b = det_core_create_bond(core, i, j);
        if (b >= 0) {
            core->bonds[b].C = 0.7f;  /* High initial coherence in P-layer */
        }
    }

    /* Initialize Self storage */
    core->self.nodes = core->self_nodes_storage;
    core->self.num_nodes = 0;

    /* Initialize ports */
    det_core_init_ports(core);

    return core;
}

void det_core_destroy(DETCore* core) {
    if (core) {
        free(core);
    }
}

void det_core_reset(DETCore* core) {
    if (!core) return;

    DETParams params = core->params;
    memset(core, 0, sizeof(DETCore));
    core->params = params;

    /* Re-initialize (simplified) */
    core->self.nodes = core->self_nodes_storage;
}

/* ==========================================================================
 * SIMULATION
 * ========================================================================== */

void det_core_update_presence(DETCore* core) {
    float sum_P = 0.0f;
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < core->num_nodes; i++) {
        DETNode* node = &core->nodes[i];
        if (!node->active) continue;

        /* Compute coordination load H_i from incident bonds */
        float H_i = 0.0f;
        for (uint32_t b = 0; b < core->num_bonds; b++) {
            DETBond* bond = &core->bonds[b];
            if (bond->i == i || bond->j == i) {
                H_i += sqrtf(bond->C) * bond->sigma;
            }
        }

        /* Presence formula: P = a * σ / (1 + F_op) / (1 + H) */
        float F_op = node->F * 0.1f;
        node->P = node->a * node->sigma / (1.0f + F_op) / (1.0f + H_i);
        node->P = clampf(node->P, 0.0f, 1.0f);

        sum_P += node->P;
        active_count++;
    }

    core->aggregate_presence = (active_count > 0) ? sum_P / active_count : 0.0f;
}

void det_core_update_coherence(DETCore* core, float dt) {
    float sum_C = 0.0f;
    uint32_t bond_count = 0;

    for (uint32_t b = 0; b < core->num_bonds; b++) {
        DETBond* bond = &core->bonds[b];
        DETNode* ni = &core->nodes[bond->i];
        DETNode* nj = &core->nodes[bond->j];

        if (!ni->active || !nj->active) continue;

        /* Get layer-specific parameters */
        float alpha, lambda, slip;
        get_bond_params(core, ni->layer, nj->layer, &alpha, &lambda, &slip);

        /* Apply affect modulation to plasticity */
        float plasticity_mod = 1.0f;
        if (bond->is_cross_layer) {
            /* Average affect-based plasticity scale */
            float r_avg = (ni->affect.r + nj->affect.r) / 2.0f;
            float b_avg = (ni->affect.b + nj->affect.b) / 2.0f;
            plasticity_mod = 1.0f + 0.3f * r_avg + 0.4f * (1.0f - b_avg);
        }

        /* Compute flux magnitude (simplified) */
        float J_mag = fabsf(ni->P - nj->P) * bond->sigma;

        /* Phase slip term */
        float phase_diff = ni->theta - nj->theta;
        float S_ij = 1.0f - cosf(phase_diff);

        /* Update phase alignment EMA */
        float phase_align = cosf(phase_diff);
        bond->phase_align_ema = 0.9f * bond->phase_align_ema + 0.1f * phase_align;

        /* Coherence update */
        float effective_alpha = alpha * plasticity_mod;
        float effective_lambda = lambda + bond->lambda_decay;
        float effective_slip = slip + bond->lambda_slip;

        float dC = effective_alpha * J_mag * dt
                 - effective_lambda * bond->C * dt
                 - effective_slip * bond->C * S_ij * dt;

        bond->C = clampf(bond->C + dC, 0.01f, 1.0f);

        /* Update flux EMA */
        bond->flux_ema = 0.9f * bond->flux_ema + 0.1f * J_mag;

        sum_C += bond->C;
        bond_count++;
    }

    core->aggregate_coherence = (bond_count > 0) ? sum_C / bond_count : 0.0f;
}

void det_core_update_agency(DETCore* core, float dt) {
    float sum_F = 0.0f, sum_q = 0.0f;
    uint32_t active_count = 0;

    for (uint32_t i = 0; i < core->num_nodes; i++) {
        DETNode* node = &core->nodes[i];
        if (!node->active) continue;

        /* Structural ceiling: a_max = 1 / (1 + λ_a * q²) */
        float a_max = 1.0f / (1.0f + core->params.lambda_a * node->q * node->q);

        /* Agency relaxation toward ceiling */
        float beta_a = 10.0f * core->params.tau_base;
        float new_a = node->a + beta_a * (a_max - node->a) * dt;

        /* Agency is inviolable - only ceiling constrains it */
        node->a = clampf(new_a, 0.01f, a_max);

        /* Slow debt decay */
        node->q *= (1.0f - 0.001f * dt);

        sum_F += node->F;
        sum_q += node->q;
        active_count++;
    }

    core->aggregate_resource = (active_count > 0) ? sum_F / active_count : 0.0f;
    core->aggregate_debt = (active_count > 0) ? sum_q / active_count : 0.0f;
}

void det_core_update_affect(DETCore* core, float dt) {
    const float Q_OK = 0.3f;
    const float GAMMA_SHORT = 0.3f;
    const float GAMMA_V = 0.1f;
    const float GAMMA_R = 0.3f;
    const float GAMMA_B = 0.05f;
    const float ALPHA_T = 1.0f, ALPHA_D = 0.8f, ALPHA_F = 0.6f;
    const float BETA_S = 0.5f, BETA_D = 0.4f, BETA_F = 0.3f;
    const float LAMBDA_ISO = 0.3f, LAMBDA_DRIFT = 0.2f;
    const float EPS = 1e-9f;

    for (uint32_t i = 0; i < core->num_nodes; i++) {
        DETNode* node = &core->nodes[i];
        if (!node->active) continue;

        DETAffect* affect = &node->affect;

        /* Compute local observables */
        float T = 0.0f;      /* Throughput */
        float C_sum = 0.0f;  /* Coherence sum */
        float B_num = 0.0f, B_den = EPS;  /* Aligned bonding */
        uint32_t neighbor_count = 0;

        for (uint32_t b = 0; b < core->num_bonds; b++) {
            DETBond* bond = &core->bonds[b];
            uint16_t other = 0;
            if (bond->i == i) other = bond->j;
            else if (bond->j == i) other = bond->i;
            else continue;

            DETNode* other_node = &core->nodes[other];
            if (!other_node->active) continue;

            float J_mag = fabsf(node->P - other_node->P) * bond->sigma;
            T += bond->C * J_mag;
            C_sum += bond->C;
            neighbor_count++;

            float w = sqrtf(bond->C) + EPS;
            float phase_align = maxf(0.0f, cosf(node->theta - other_node->theta));
            B_den += w;
            B_num += sqrtf(bond->C) * phase_align;
        }

        float F = (neighbor_count > 0) ? 1.0f - C_sum / neighbor_count : 1.0f;
        float D = maxf(0.0f, node->q - Q_OK);
        float S = fabsf(T - affect->ema_throughput);
        float B = B_num / B_den;

        /* Update short traces */
        affect->ema_throughput = (1.0f - GAMMA_SHORT) * affect->ema_throughput + GAMMA_SHORT * T;
        affect->ema_fragmentation = (1.0f - GAMMA_SHORT) * affect->ema_fragmentation + GAMMA_SHORT * F;
        affect->ema_debt = (1.0f - GAMMA_SHORT) * affect->ema_debt + GAMMA_SHORT * D;
        affect->ema_surprise = (1.0f - GAMMA_SHORT) * affect->ema_surprise + GAMMA_SHORT * S;
        affect->ema_bonding = (1.0f - GAMMA_SHORT) * affect->ema_bonding + GAMMA_SHORT * B;

        /* Compute affect targets */
        float v_hat = tanhf(ALPHA_T * affect->ema_throughput
                         - ALPHA_D * affect->ema_debt
                         - ALPHA_F * affect->ema_fragmentation);

        float r_hat = clampf(BETA_S * affect->ema_surprise
                           + BETA_D * affect->ema_debt
                           + BETA_F * affect->ema_fragmentation, 0.0f, 1.0f);

        float b_hat = clampf(affect->ema_bonding
                           - LAMBDA_ISO * affect->ema_fragmentation
                           - LAMBDA_DRIFT * affect->ema_debt, 0.0f, 1.0f);

        /* Integrate affect */
        affect->v = (1.0f - GAMMA_V) * affect->v + GAMMA_V * v_hat;
        affect->r = (1.0f - GAMMA_R) * affect->r + GAMMA_R * r_hat;
        affect->b = (1.0f - GAMMA_B) * affect->b + GAMMA_B * b_hat;
    }
}

void det_core_identify_self(DETCore* core) {
    const float KAPPA = 1.2f;
    const float ALPHA = 1.0f;
    const float BETA = 2.0f;
    const float EPS = 1e-9f;

    /* Step 1: Compute edge weights w_ij = C_ij × |J_ij| × √(a_i × a_j) */
    for (uint32_t b = 0; b < core->num_bonds; b++) {
        DETBond* bond = &core->bonds[b];
        DETNode* ni = &core->nodes[bond->i];
        DETNode* nj = &core->nodes[bond->j];

        if (!ni->active || !nj->active) {
            bond->stability_ema = 0.0f;
            continue;
        }

        float J_mag = fabsf(ni->P - nj->P) * bond->sigma;
        float w = bond->C * J_mag * sqrtf(ni->a * nj->a);
        bond->stability_ema = 0.95f * bond->stability_ema + 0.05f * w;
    }

    /* Step 2-3: Local threshold filtering (simplified) */
    /* For Phase 1, just select P-layer nodes with high local coherence */
    core->self.num_nodes = 0;
    core->self.cluster_agency = 0.0f;

    float v_sum = 0.0f, r_sum = 0.0f, b_sum = 0.0f, w_sum = 0.0f;

    for (uint32_t i = 0; i < DET_P_LAYER_SIZE; i++) {
        DETNode* node = &core->nodes[i];
        if (!node->active) continue;

        /* Check if node has sufficient local coherence */
        float local_C = 0.0f;
        uint32_t bond_count = 0;
        for (uint32_t b = 0; b < core->num_bonds; b++) {
            DETBond* bond = &core->bonds[b];
            if (bond->i == i || bond->j == i) {
                local_C += bond->C;
                bond_count++;
                core->self.cluster_agency += bond->stability_ema;
            }
        }

        if (bond_count > 0) {
            local_C /= bond_count;
        }

        /* Include in Self if coherence is sufficient */
        if (local_C > 0.3f) {
            core->self.nodes[core->self.num_nodes++] = i;

            /* Accumulate affect */
            float w = node->a * node->P;
            v_sum += w * node->affect.v;
            r_sum += w * node->affect.r;
            b_sum += w * node->affect.b;
            w_sum += w;
        }
    }

    /* Compute Self-level affect */
    if (w_sum > EPS) {
        core->self.valence = v_sum / w_sum;
        core->self.arousal = r_sum / w_sum;
        core->self.bondedness = b_sum / w_sum;
    }

    /* Determine emotional state */
    bool pos_v = core->self.valence > 0.2f;
    bool neg_v = core->self.valence < -0.2f;
    bool high_r = core->self.arousal > 0.5f;
    bool high_b = core->self.bondedness > 0.5f;

    if (pos_v && !high_r && high_b) core->emotion = DET_EMOTION_CONTENTMENT;
    else if (pos_v && high_r && high_b) core->emotion = DET_EMOTION_FLOW;
    else if (pos_v && !high_r && !high_b) core->emotion = DET_EMOTION_BOREDOM;
    else if (neg_v && high_r && high_b) core->emotion = DET_EMOTION_STRESS;
    else if (neg_v && high_r && !high_b) core->emotion = DET_EMOTION_OVERWHELM;
    else if (neg_v && !high_r && !high_b) core->emotion = DET_EMOTION_APATHY;
    else if (!pos_v && !neg_v && !high_r && high_b) core->emotion = DET_EMOTION_PEACE;
    else core->emotion = DET_EMOTION_NEUTRAL;
}

void det_core_step(DETCore* core, float dt) {
    core->tick++;

    /* Process grace buffers first */
    det_core_process_grace(core, dt);

    /* Core DET updates */
    det_core_update_presence(core);
    det_core_update_coherence(core, dt);
    det_core_update_agency(core, dt);

    /* Phase 4: Extended dynamics */
    det_core_update_momentum(core, dt);
    det_core_update_angular_momentum(core, dt);
    det_core_update_debt(core, dt);

    /* Emotional feedback */
    det_core_update_affect(core, dt);

    /* Periodic Self identification */
    if (core->tick % 10 == 0) {
        det_core_identify_self(core);
    }
}

/* ==========================================================================
 * GATEKEEPER
 * ========================================================================== */

DETDecision det_core_evaluate_request(
    DETCore* core,
    const uint32_t* tokens,
    uint32_t num_tokens,
    uint8_t target_domain,
    uint32_t retry_count
) {
    const uint32_t MAX_RETRIES = 5;
    const float AGENCY_THRESHOLD = 0.1f;
    const float COHERENCE_THRESHOLD = 0.3f;
    const float PRESENCE_THRESHOLD = 0.2f;
    const float RESOURCE_PER_TOKEN = 0.001f;

    /* Estimate complexity from token count */
    float complexity = (float)num_tokens * RESOURCE_PER_TOKEN;

    /* Get current state */
    float P = core->aggregate_presence;
    float C = det_core_get_domain_coherence(core, target_domain);
    float F = core->aggregate_resource;
    float q = core->aggregate_debt;
    float a_max = 1.0f / (1.0f + core->params.lambda_a * q * q);

    /* Apply affect modulation to thresholds */
    float esc_mult = (1.0f + 0.5f * core->self.arousal)
                   * (1.0f + 0.3f * maxf(0.0f, -core->self.valence))
                   * (1.0f + 0.4f * (1.0f - core->self.bondedness));

    float effective_agency_threshold = AGENCY_THRESHOLD / esc_mult;

    /* Prison regime check */
    if (C > 0.7f && a_max < 0.2f && core->self.bondedness < 0.3f) {
        /* High coherence + low agency + low bondedness = zombie state */
        return DET_DECISION_STOP;
    }

    /* Agency too constrained */
    if (a_max < effective_agency_threshold) {
        if (retry_count < MAX_RETRIES) return DET_DECISION_RETRY;
        return DET_DECISION_STOP;
    }

    /* Not enough resource */
    if (F < complexity * 2.0f) {
        if (retry_count < MAX_RETRIES) return DET_DECISION_RETRY;
        return DET_DECISION_ESCALATE;
    }

    /* Coherence too low in target domain */
    if (C < COHERENCE_THRESHOLD) {
        if (retry_count < MAX_RETRIES) return DET_DECISION_RETRY;
        return DET_DECISION_ESCALATE;
    }

    /* Presence too low (system is "foggy") */
    if (P < PRESENCE_THRESHOLD) {
        if (retry_count < MAX_RETRIES) return DET_DECISION_RETRY;
        return DET_DECISION_STOP;
    }

    return DET_DECISION_PROCEED;
}

/* ==========================================================================
 * PORT INTERFACE
 * ========================================================================== */

void det_core_init_ports(DETCore* core) {
    /* Create port nodes for LLM interface */
    const char* intent_names[] = {"answer", "plan", "execute", "learn", "debug"};
    const char* domain_names[] = {"math", "language", "tool_use", "science"};

    core->num_ports = 0;

    /* Intent ports */
    for (int i = 0; i < 5 && core->num_ports < DET_MAX_PORTS; i++) {
        DETPort* port = &core->ports[core->num_ports];
        port->node_id = core->num_nodes;
        port->port_type = 0;  /* Intent */
        snprintf(port->name, sizeof(port->name), "intent_%s", intent_names[i]);

        /* Initialize the actual node */
        DETNode* node = &core->nodes[core->num_nodes];
        node->layer = DET_LAYER_PORT;
        node->active = true;
        node->a = 0.5f;
        node->F = 0.0f;
        node->sigma = core->params.sigma_base;

        core->num_nodes++;
        core->num_ports++;
    }

    /* Domain ports */
    for (int i = 0; i < 4 && core->num_ports < DET_MAX_PORTS; i++) {
        DETPort* port = &core->ports[core->num_ports];
        port->node_id = core->num_nodes;
        port->port_type = 1;  /* Domain */
        port->target_domain = i;
        snprintf(port->name, sizeof(port->name), "domain_%s", domain_names[i]);

        DETNode* node = &core->nodes[core->num_nodes];
        node->layer = DET_LAYER_PORT;
        node->active = true;
        node->a = 0.5f;
        node->F = 0.0f;
        node->sigma = core->params.sigma_base;

        core->num_nodes++;
        core->num_ports++;
    }
}

void det_core_inject_stimulus(
    DETCore* core,
    const uint8_t* port_indices,
    const float* activations,
    uint32_t num_activations
) {
    const float ETA = 0.5f;

    for (uint32_t i = 0; i < num_activations; i++) {
        if (port_indices[i] >= core->num_ports) continue;

        uint16_t node_id = core->ports[port_indices[i]].node_id;
        core->nodes[node_id].F += ETA * activations[i];
    }
}

void det_core_create_interface_bonds(
    DETCore* core,
    uint8_t target_domain,
    float initial_C
) {
    /* Find domain-related port */
    int port_node = -1;
    for (uint32_t p = 0; p < core->num_ports; p++) {
        if (core->ports[p].port_type == 1 &&
            core->ports[p].target_domain == target_domain) {
            port_node = core->ports[p].node_id;
            break;
        }
    }

    if (port_node < 0) return;

    /* Create temporary bonds to A-layer nodes in target domain */
    uint32_t a_start = DET_P_LAYER_SIZE;
    uint32_t a_end = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE;

    for (uint32_t i = a_start; i < a_end && core->num_bonds < DET_MAX_BONDS; i++) {
        if (core->nodes[i].domain != target_domain) continue;

        int32_t b = det_core_find_bond(core, port_node, i);
        if (b < 0) {
            b = det_core_create_bond(core, port_node, i);
        }

        if (b >= 0) {
            core->bonds[b].C = initial_C;
            core->bonds[b].is_temporary = true;
        }
    }
}

void det_core_cleanup_interface_bonds(DETCore* core) {
    /* Remove or weaken temporary bonds */
    for (uint32_t b = 0; b < core->num_bonds; b++) {
        if (core->bonds[b].is_temporary) {
            /* If bond strengthened significantly, keep it */
            if (core->bonds[b].C > 0.5f) {
                core->bonds[b].is_temporary = false;
            } else {
                /* Mark for removal by zeroing coherence */
                core->bonds[b].C = 0.0f;
            }
        }
    }
}

/* ==========================================================================
 * MEMORY DOMAINS
 * ========================================================================== */

bool det_core_register_domain(
    DETCore* core,
    const char* name,
    void* model_handle
) {
    if (core->num_domains >= DET_MAX_DOMAINS) return false;

    DETDomain* domain = &core->domains[core->num_domains];
    strncpy(domain->name, name, sizeof(domain->name) - 1);
    domain->model_handle = model_handle;
    domain->coherence_to_core = 0.5f;
    domain->activation_level = 0.5f;

    core->num_domains++;
    return true;
}

float det_core_get_domain_coherence(const DETCore* core, uint8_t domain) {
    if (domain >= core->num_domains) {
        /* Default coherence for unregistered domains */
        return 0.5f;
    }

    /* Compute average coherence of A-layer nodes in this domain */
    float sum_C = 0.0f;
    uint32_t count = 0;

    uint32_t a_start = DET_P_LAYER_SIZE;
    uint32_t a_end = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE;

    for (uint32_t i = a_start; i < a_end; i++) {
        if (core->nodes[i].domain != domain) continue;
        if (!core->nodes[i].active) continue;

        for (uint32_t b = 0; b < core->num_bonds; b++) {
            if (core->bonds[b].i == i || core->bonds[b].j == i) {
                sum_C += core->bonds[b].C;
                count++;
            }
        }
    }

    return (count > 0) ? sum_C / count : 0.5f;
}

/* ==========================================================================
 * QUERIES
 * ========================================================================== */

DETEmotion det_core_get_emotion(const DETCore* core) {
    return core->emotion;
}

const char* det_core_emotion_string(DETEmotion emotion) {
    switch (emotion) {
        case DET_EMOTION_FLOW: return "flow";
        case DET_EMOTION_CONTENTMENT: return "contentment";
        case DET_EMOTION_STRESS: return "stress";
        case DET_EMOTION_OVERWHELM: return "overwhelm";
        case DET_EMOTION_APATHY: return "apathy";
        case DET_EMOTION_BOREDOM: return "boredom";
        case DET_EMOTION_PEACE: return "peace";
        default: return "neutral";
    }
}

void det_core_get_self_affect(
    const DETCore* core,
    float* valence,
    float* arousal,
    float* bondedness
) {
    if (valence) *valence = core->self.valence;
    if (arousal) *arousal = core->self.arousal;
    if (bondedness) *bondedness = core->self.bondedness;
}

void det_core_get_aggregates(
    const DETCore* core,
    float* presence,
    float* coherence,
    float* resource,
    float* debt
) {
    if (presence) *presence = core->aggregate_presence;
    if (coherence) *coherence = core->aggregate_coherence;
    if (resource) *resource = core->aggregate_resource;
    if (debt) *debt = core->aggregate_debt;
}

/* ==========================================================================
 * NODE/BOND MANAGEMENT
 * ========================================================================== */

int32_t det_core_recruit_node(DETCore* core, DETLayer target_layer) {
    /* Find a dormant node to recruit */
    uint32_t dormant_start = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE;

    for (uint32_t i = dormant_start; i < core->num_nodes; i++) {
        if (core->nodes[i].layer == DET_LAYER_DORMANT && !core->nodes[i].active) {
            core->nodes[i].layer = target_layer;
            core->nodes[i].active = true;
            core->nodes[i].F = 0.5f;
            core->nodes[i].sigma = core->params.sigma_base;
            core->nodes[i].q = 0.0f;
            core->num_active++;
            return i;
        }
    }

    return -1;  /* No dormant nodes available */
}

void det_core_retire_node(DETCore* core, uint16_t node_id) {
    if (node_id >= core->num_nodes) return;

    DETNode* node = &core->nodes[node_id];
    if (node->layer == DET_LAYER_DORMANT) return;

    /* Export debt to neighbors before retiring */
    for (uint32_t b = 0; b < core->num_bonds; b++) {
        DETBond* bond = &core->bonds[b];
        uint16_t other = 0;
        if (bond->i == node_id) other = bond->j;
        else if (bond->j == node_id) other = bond->i;
        else continue;

        /* Transfer some debt */
        core->nodes[other].q += node->q * 0.1f * bond->C;

        /* Weaken bond */
        bond->C *= 0.5f;
    }

    node->layer = DET_LAYER_DORMANT;
    node->active = false;
    node->F = 0.0f;
    node->q = 0.0f;  /* Debt exported */
    node->sigma = 0.0f;
    core->num_active--;
}

int32_t det_core_create_bond(DETCore* core, uint16_t i, uint16_t j) {
    if (core->num_bonds >= DET_MAX_BONDS) return -1;
    if (i == j) return -1;

    /* Check if bond already exists */
    int32_t existing = det_core_find_bond(core, i, j);
    if (existing >= 0) return existing;

    DETBond* bond = &core->bonds[core->num_bonds];
    bond->i = i;
    bond->j = j;
    bond->C = 0.1f;
    bond->pi = 0.0f;
    bond->sigma = core->params.sigma_base;
    bond->flux_ema = 0.0f;
    bond->phase_align_ema = 0.0f;
    bond->stability_ema = 0.0f;
    bond->lambda_decay = 0.0f;
    bond->lambda_slip = 0.0f;
    bond->is_temporary = false;

    /* Check if cross-layer */
    DETLayer li = core->nodes[i].layer;
    DETLayer lj = core->nodes[j].layer;
    bond->is_cross_layer = (li != lj) &&
                           ((li == DET_LAYER_P && lj == DET_LAYER_A) ||
                            (li == DET_LAYER_A && lj == DET_LAYER_P));

    return core->num_bonds++;
}

int32_t det_core_find_bond(const DETCore* core, uint16_t i, uint16_t j) {
    for (uint32_t b = 0; b < core->num_bonds; b++) {
        if ((core->bonds[b].i == i && core->bonds[b].j == j) ||
            (core->bonds[b].i == j && core->bonds[b].j == i)) {
            return b;
        }
    }
    return -1;
}

/* ==========================================================================
 * PHASE 4: EXTENDED DYNAMICS
 * ========================================================================== */

void det_core_update_momentum(DETCore* core, float dt) {
    /*
     * Bond momentum (π) captures the "intention" or directional memory
     * of information flow between nodes.
     *
     * dπ/dt = κ * J - λ_π * π
     *
     * where:
     *   J = flux (P_i - P_j) * σ_bond
     *   κ = coupling scale
     *   λ_π = momentum decay
     */
    const float LAMBDA_PI = 0.1f;  /* Momentum decay rate */

    for (uint32_t b = 0; b < core->num_bonds; b++) {
        DETBond* bond = &core->bonds[b];
        DETNode* ni = &core->nodes[bond->i];
        DETNode* nj = &core->nodes[bond->j];

        if (!ni->active || !nj->active) continue;

        /* Compute directed flux (signed) */
        float J = (ni->P - nj->P) * bond->sigma;

        /* Momentum update with decay */
        float dpi = core->params.kappa_base * J * dt
                  - LAMBDA_PI * bond->pi * dt;

        bond->pi = clampf(bond->pi + dpi, -core->params.pi_max, core->params.pi_max);
    }
}

void det_core_update_angular_momentum(DETCore* core, float dt) {
    /*
     * Angular momentum (L) represents "spin" - the tendency for a node's
     * phase to continue rotating.
     *
     * dθ/dt = (1/τ) * L / (1 + H)
     * dL/dt = Σ_j C_ij * sin(θ_j - θ_i) - λ_L * L
     *
     * This creates phase coupling between bonded nodes while preserving
     * individual phase momentum.
     */
    const float LAMBDA_L = 0.05f;  /* Angular momentum decay */

    for (uint32_t i = 0; i < core->num_nodes; i++) {
        DETNode* node = &core->nodes[i];
        if (!node->active) continue;

        /* Compute torque from neighbors */
        float torque = 0.0f;
        float H_i = 0.0f;

        for (uint32_t b = 0; b < core->num_bonds; b++) {
            DETBond* bond = &core->bonds[b];
            uint16_t other = 0;
            if (bond->i == i) other = bond->j;
            else if (bond->j == i) other = bond->i;
            else continue;

            DETNode* other_node = &core->nodes[other];
            if (!other_node->active) continue;

            /* Phase coupling torque */
            float phase_diff = other_node->theta - node->theta;
            torque += bond->C * sinf(phase_diff);

            /* Coordination load */
            H_i += sqrtf(bond->C) * bond->sigma;
        }

        /* Update angular momentum */
        float dL = torque * dt - LAMBDA_L * node->L * dt;
        node->L = clampf(node->L + dL, -2.0f, 2.0f);

        /* Update phase velocity */
        node->dtheta_dt = node->L / (core->params.tau_base * (1.0f + H_i));

        /* Update phase */
        node->theta += node->dtheta_dt * dt;

        /* Keep phase in [0, 2π] */
        while (node->theta > 6.28318f) node->theta -= 6.28318f;
        while (node->theta < 0.0f) node->theta += 6.28318f;
    }
}

void det_core_update_debt(DETCore* core, float dt) {
    /*
     * Structural debt (q) accumulates from:
     *   1. High resource expenditure (F usage)
     *   2. Phase misalignment with neighbors
     *   3. Low coherence in bonds
     *
     * dq/dt = (F_spent / F_max) + Σ_j (1 - cos(θ_i - θ_j)) * w_j - λ_q * q
     *
     * Debt constrains agency ceiling: a_max = 1 / (1 + λ_a * q²)
     */
    const float LAMBDA_Q = 0.002f;  /* Debt decay rate */
    const float DEBT_RATE = 0.01f;  /* Debt accumulation from activity */

    for (uint32_t i = 0; i < core->num_nodes; i++) {
        DETNode* node = &core->nodes[i];
        if (!node->active) continue;

        /* Debt from resource usage */
        float usage_debt = DEBT_RATE * maxf(0.0f, 1.0f - node->F) * dt;

        /* Debt from phase misalignment */
        float align_debt = 0.0f;
        float total_weight = 0.0f;

        for (uint32_t b = 0; b < core->num_bonds; b++) {
            DETBond* bond = &core->bonds[b];
            uint16_t other = 0;
            if (bond->i == i) other = bond->j;
            else if (bond->j == i) other = bond->i;
            else continue;

            DETNode* other_node = &core->nodes[other];
            if (!other_node->active) continue;

            float phase_diff = node->theta - other_node->theta;
            float misalign = 1.0f - cosf(phase_diff);
            align_debt += bond->C * misalign;
            total_weight += bond->C;
        }

        if (total_weight > 0.0f) {
            align_debt = 0.005f * (align_debt / total_weight) * dt;
        }

        /* Total debt update with decay */
        float dq = usage_debt + align_debt - LAMBDA_Q * node->q * dt;
        node->q = clampf(node->q + dq, 0.0f, 5.0f);
    }
}

void det_core_inject_grace(DETCore* core, uint16_t node_id, float amount) {
    if (node_id >= core->num_nodes) return;

    DETNode* node = &core->nodes[node_id];
    if (!node->active) return;

    /* Add to grace buffer for processing during step */
    node->grace_buffer += amount;
}

void det_core_process_grace(DETCore* core, float dt) {
    /*
     * Grace injection is boundary recovery - when a node is near failure
     * (low F, high q), grace provides relief.
     *
     * Grace effects:
     *   1. Replenishes resource F
     *   2. Reduces structural debt q
     *   3. Boosts coherence with neighbors (temporary)
     */
    const float GRACE_TO_F = 0.5f;    /* Fraction to resource */
    const float GRACE_TO_Q = 0.3f;    /* Fraction to debt reduction */
    const float GRACE_TO_C = 0.2f;    /* Fraction to coherence boost */

    for (uint32_t i = 0; i < core->num_nodes; i++) {
        DETNode* node = &core->nodes[i];
        if (!node->active || node->grace_buffer <= 0.0f) continue;

        float grace = node->grace_buffer;
        node->grace_buffer = 0.0f;

        /* Replenish resource */
        node->F = clampf(node->F + grace * GRACE_TO_F, 0.0f, 2.0f);

        /* Reduce debt */
        node->q = maxf(0.0f, node->q - grace * GRACE_TO_Q);

        /* Boost neighbor coherence */
        for (uint32_t b = 0; b < core->num_bonds; b++) {
            DETBond* bond = &core->bonds[b];
            if (bond->i != i && bond->j != i) continue;

            bond->C = clampf(bond->C + grace * GRACE_TO_C * 0.1f, 0.0f, 1.0f);
        }
    }
}

bool det_core_needs_grace(const DETCore* core, uint16_t node_id) {
    if (node_id >= core->num_nodes) return false;

    const DETNode* node = &core->nodes[node_id];
    if (!node->active) return false;

    /* Grace needed when F is low and q is high */
    return (node->F < 0.2f && node->q > 0.5f);
}

float det_core_total_grace_needed(const DETCore* core) {
    float total = 0.0f;

    for (uint32_t i = 0; i < core->num_nodes; i++) {
        const DETNode* node = &core->nodes[i];
        if (!node->active) continue;

        if (det_core_needs_grace(core, i)) {
            /* Grace needed proportional to deficit */
            total += (0.5f - node->F) + node->q * 0.5f;
        }
    }

    return total;
}

/* ==========================================================================
 * PHASE 4: LEARNING VIA RECRUITMENT
 * ========================================================================== */

bool det_core_can_learn(const DETCore* core, float complexity, uint8_t domain) {
    /*
     * Learning criteria (from DET subdivision theory):
     *   1. Self-cluster has sufficient agency
     *   2. Target domain has capacity
     *   3. Resource is available
     *   4. Not in prison regime
     */
    const float A_MIN_LEARN = 0.3f;     /* Minimum cluster agency */
    const float F_MIN_LEARN = 0.4f;     /* Minimum resource */
    const float LEARN_COST = 0.1f;      /* Resource cost per unit complexity */

    /* Check cluster agency */
    if (core->self.cluster_agency < A_MIN_LEARN) return false;

    /* Check resource */
    if (core->aggregate_resource < F_MIN_LEARN) return false;
    if (core->aggregate_resource < complexity * LEARN_COST) return false;

    /* Check not in prison regime */
    float a_max = 1.0f / (1.0f + core->params.lambda_a * core->aggregate_debt * core->aggregate_debt);
    if (core->aggregate_coherence > 0.7f && a_max < 0.2f && core->self.bondedness < 0.3f) {
        return false;  /* Prison regime */
    }

    /* Count available dormant nodes */
    uint32_t available = 0;
    uint32_t dormant_start = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE;
    for (uint32_t i = dormant_start; i < core->num_nodes; i++) {
        if (core->nodes[i].layer == DET_LAYER_DORMANT && !core->nodes[i].active) {
            available++;
        }
    }

    /* Need at least some dormant nodes for recruitment */
    return available >= 4;
}

bool det_core_activate_domain(
    DETCore* core,
    const char* name,
    uint32_t num_nodes,
    float initial_coherence
) {
    if (core->num_domains >= DET_MAX_DOMAINS) return false;
    if (num_nodes == 0 || num_nodes > 64) return false;

    /* Register the domain */
    uint8_t domain_id = core->num_domains;
    DETDomain* domain = &core->domains[domain_id];
    strncpy(domain->name, name, sizeof(domain->name) - 1);
    domain->model_handle = NULL;
    domain->coherence_to_core = initial_coherence;
    domain->activation_level = 0.1f;
    core->num_domains++;

    /* Recruit nodes for the domain */
    uint16_t recruited[64];
    uint32_t recruited_count = 0;

    for (uint32_t n = 0; n < num_nodes; n++) {
        int32_t node_id = det_core_recruit_node(core, DET_LAYER_A);
        if (node_id < 0) break;

        core->nodes[node_id].domain = domain_id;
        recruited[recruited_count++] = node_id;
    }

    /* Create bonds between recruited nodes (ring + some cross-links) */
    for (uint32_t i = 0; i < recruited_count; i++) {
        /* Ring bond */
        uint32_t j = (i + 1) % recruited_count;
        int32_t b = det_core_create_bond(core, recruited[i], recruited[j]);
        if (b >= 0) {
            core->bonds[b].C = initial_coherence;
        }

        /* Cross-link to P-layer */
        if (i < DET_P_LAYER_SIZE) {
            b = det_core_create_bond(core, recruited[i], i);
            if (b >= 0) {
                core->bonds[b].C = initial_coherence * 0.5f;
                core->bonds[b].is_cross_layer = true;
            }
        }
    }

    return recruited_count > 0;
}

bool det_core_transfer_pattern(
    DETCore* core,
    uint8_t source_domain,
    uint8_t target_domain,
    float transfer_strength
) {
    if (source_domain >= core->num_domains) return false;
    if (target_domain >= core->num_domains) return false;

    /*
     * Pattern transfer creates bonds between source and target domain nodes,
     * and copies the relative coherence structure.
     */
    uint32_t a_start = DET_P_LAYER_SIZE;
    uint32_t a_end = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE;

    /* Find nodes in each domain */
    uint16_t source_nodes[64], target_nodes[64];
    uint32_t src_count = 0, tgt_count = 0;

    for (uint32_t i = a_start; i < a_end && (src_count < 64 || tgt_count < 64); i++) {
        if (!core->nodes[i].active) continue;

        if (core->nodes[i].domain == source_domain && src_count < 64) {
            source_nodes[src_count++] = i;
        } else if (core->nodes[i].domain == target_domain && tgt_count < 64) {
            target_nodes[tgt_count++] = i;
        }
    }

    if (src_count == 0 || tgt_count == 0) return false;

    /* Create cross-domain bonds */
    uint32_t bonds_created = 0;
    uint32_t max_bonds = minf(src_count, tgt_count);

    for (uint32_t i = 0; i < max_bonds; i++) {
        int32_t b = det_core_create_bond(core, source_nodes[i], target_nodes[i]);
        if (b >= 0) {
            core->bonds[b].C = transfer_strength * 0.5f;
            bonds_created++;
        }
    }

    /* Update domain coherence */
    core->domains[target_domain].coherence_to_core =
        (core->domains[target_domain].coherence_to_core +
         core->domains[source_domain].coherence_to_core * transfer_strength) /
        (1.0f + transfer_strength);

    return bonds_created > 0;
}

float det_core_learning_capacity(const DETCore* core) {
    /*
     * Learning capacity is determined by:
     *   1. Available dormant nodes with high agency
     *   2. Current resource level
     *   3. Cluster agency (self-coherence)
     */
    float dormant_agency = 0.0f;
    uint32_t dormant_count = 0;
    uint32_t dormant_start = DET_P_LAYER_SIZE + DET_A_LAYER_SIZE;

    for (uint32_t i = dormant_start; i < core->num_nodes; i++) {
        if (core->nodes[i].layer == DET_LAYER_DORMANT && !core->nodes[i].active) {
            dormant_agency += core->nodes[i].a;
            dormant_count++;
        }
    }

    if (dormant_count == 0) return 0.0f;

    float avg_dormant_agency = dormant_agency / dormant_count;
    float resource_factor = clampf(core->aggregate_resource, 0.0f, 1.0f);
    float cluster_factor = clampf(core->self.cluster_agency, 0.0f, 1.0f);

    return avg_dormant_agency * resource_factor * cluster_factor * (float)dormant_count / 100.0f;
}

/* ==========================================================================
 * PHASE 4: MULTI-SESSION SUPPORT
 * ========================================================================== */

size_t det_core_state_size(const DETCore* core) {
    /* Compute serialized state size */
    size_t size = 0;

    /* Header */
    size += sizeof(uint32_t);  /* magic */
    size += sizeof(uint32_t);  /* version */

    /* Params */
    size += sizeof(DETParams);

    /* Counts */
    size += sizeof(uint32_t) * 5;  /* num_nodes, num_active, num_bonds, num_ports, num_domains */

    /* Tick */
    size += sizeof(uint64_t);

    /* All nodes (simplified - save all for correct restoration) */
    size += core->num_nodes * sizeof(DETNode);

    /* Bonds */
    size += core->num_bonds * sizeof(DETBond);

    /* Aggregates */
    size += sizeof(float) * 4;

    /* Self cluster info */
    size += sizeof(float) * 4 + sizeof(uint32_t);  /* affect + num_nodes */
    size += core->self.num_nodes * sizeof(uint16_t);

    return size;
}

size_t det_core_save_state(const DETCore* core, void* buffer, size_t buffer_size) {
    size_t needed = det_core_state_size(core);
    if (buffer_size < needed) return 0;

    uint8_t* ptr = (uint8_t*)buffer;

    /* Magic number */
    uint32_t magic = 0x44455443;  /* "DETC" */
    memcpy(ptr, &magic, sizeof(magic));
    ptr += sizeof(magic);

    /* Version */
    uint32_t version = 4;  /* Phase 4 format */
    memcpy(ptr, &version, sizeof(version));
    ptr += sizeof(version);

    /* Params */
    memcpy(ptr, &core->params, sizeof(DETParams));
    ptr += sizeof(DETParams);

    /* Counts */
    memcpy(ptr, &core->num_nodes, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &core->num_active, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &core->num_bonds, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &core->num_ports, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, &core->num_domains, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    /* Tick */
    memcpy(ptr, &core->tick, sizeof(uint64_t));
    ptr += sizeof(uint64_t);

    /* All nodes */
    memcpy(ptr, core->nodes, core->num_nodes * sizeof(DETNode));
    ptr += core->num_nodes * sizeof(DETNode);

    /* Bonds */
    memcpy(ptr, core->bonds, core->num_bonds * sizeof(DETBond));
    ptr += core->num_bonds * sizeof(DETBond);

    /* Aggregates */
    memcpy(ptr, &core->aggregate_presence, sizeof(float));
    ptr += sizeof(float);
    memcpy(ptr, &core->aggregate_coherence, sizeof(float));
    ptr += sizeof(float);
    memcpy(ptr, &core->aggregate_resource, sizeof(float));
    ptr += sizeof(float);
    memcpy(ptr, &core->aggregate_debt, sizeof(float));
    ptr += sizeof(float);

    /* Self cluster */
    memcpy(ptr, &core->self.cluster_agency, sizeof(float));
    ptr += sizeof(float);
    memcpy(ptr, &core->self.valence, sizeof(float));
    ptr += sizeof(float);
    memcpy(ptr, &core->self.arousal, sizeof(float));
    ptr += sizeof(float);
    memcpy(ptr, &core->self.bondedness, sizeof(float));
    ptr += sizeof(float);
    memcpy(ptr, &core->self.num_nodes, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(ptr, core->self.nodes, core->self.num_nodes * sizeof(uint16_t));
    ptr += core->self.num_nodes * sizeof(uint16_t);

    return ptr - (uint8_t*)buffer;
}

bool det_core_load_state(DETCore* core, const void* buffer, size_t data_size) {
    if (data_size < sizeof(uint32_t) * 2) return false;

    const uint8_t* ptr = (const uint8_t*)buffer;

    /* Check magic */
    uint32_t magic;
    memcpy(&magic, ptr, sizeof(magic));
    ptr += sizeof(magic);
    if (magic != 0x44455443) return false;  /* "DETC" */

    /* Check version */
    uint32_t version;
    memcpy(&version, ptr, sizeof(version));
    ptr += sizeof(version);
    if (version != 4) return false;  /* Only support Phase 4 format */

    /* Load params */
    memcpy(&core->params, ptr, sizeof(DETParams));
    ptr += sizeof(DETParams);

    /* Load counts */
    uint32_t saved_num_nodes;
    memcpy(&saved_num_nodes, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&core->num_active, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&core->num_bonds, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&core->num_ports, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);
    memcpy(&core->num_domains, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    /* Load tick */
    memcpy(&core->tick, ptr, sizeof(uint64_t));
    ptr += sizeof(uint64_t);

    /* Load nodes - only up to saved count */
    if (saved_num_nodes <= DET_MAX_NODES) {
        memcpy(core->nodes, ptr, saved_num_nodes * sizeof(DETNode));
        ptr += saved_num_nodes * sizeof(DETNode);
        core->num_nodes = saved_num_nodes;
    } else {
        return false;  /* Invalid saved state */
    }

    /* Load bonds */
    if (core->num_bonds <= DET_MAX_BONDS) {
        memcpy(core->bonds, ptr, core->num_bonds * sizeof(DETBond));
        ptr += core->num_bonds * sizeof(DETBond);
    } else {
        return false;
    }

    /* Load aggregates */
    memcpy(&core->aggregate_presence, ptr, sizeof(float));
    ptr += sizeof(float);
    memcpy(&core->aggregate_coherence, ptr, sizeof(float));
    ptr += sizeof(float);
    memcpy(&core->aggregate_resource, ptr, sizeof(float));
    ptr += sizeof(float);
    memcpy(&core->aggregate_debt, ptr, sizeof(float));
    ptr += sizeof(float);

    /* Load self cluster */
    memcpy(&core->self.cluster_agency, ptr, sizeof(float));
    ptr += sizeof(float);
    memcpy(&core->self.valence, ptr, sizeof(float));
    ptr += sizeof(float);
    memcpy(&core->self.arousal, ptr, sizeof(float));
    ptr += sizeof(float);
    memcpy(&core->self.bondedness, ptr, sizeof(float));
    ptr += sizeof(float);
    memcpy(&core->self.num_nodes, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    /* Ensure self.nodes points to storage */
    core->self.nodes = core->self_nodes_storage;
    if (core->self.num_nodes <= DET_MAX_NODES) {
        memcpy(core->self.nodes, ptr, core->self.num_nodes * sizeof(uint16_t));
    }

    return true;
}
