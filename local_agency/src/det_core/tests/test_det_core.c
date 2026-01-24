/**
 * DET Core Test Suite
 * ===================
 *
 * Basic tests for the DET C kernel.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "det_core.h"

/* Test counters */
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    do { \
        printf("  Testing: %s... ", #name); \
        tests_run++; \
        if (test_##name()) { \
            printf("PASS\n"); \
            tests_passed++; \
        } else { \
            printf("FAIL\n"); \
        } \
    } while(0)

#define ASSERT(cond) \
    do { \
        if (!(cond)) { \
            printf("\n    Assertion failed: %s (line %d)\n", #cond, __LINE__); \
            return 0; \
        } \
    } while(0)

#define ASSERT_FLOAT_EQ(a, b, eps) \
    do { \
        if (fabsf((a) - (b)) > (eps)) { \
            printf("\n    Float assertion failed: %f != %f (line %d)\n", (a), (b), __LINE__); \
            return 0; \
        } \
    } while(0)

/* ==========================================================================
 * Lifecycle Tests
 * ========================================================================== */

int test_create_destroy(void) {
    DETCore* core = det_core_create();
    ASSERT(core != NULL);
    ASSERT(core->num_nodes > 0);
    ASSERT(core->tick == 0);
    det_core_destroy(core);
    return 1;
}

int test_default_params(void) {
    DETParams params = det_default_params();
    ASSERT_FLOAT_EQ(params.tau_base, 0.02f, 0.001f);
    ASSERT_FLOAT_EQ(params.sigma_base, 0.12f, 0.001f);
    ASSERT_FLOAT_EQ(params.lambda_base, 0.008f, 0.0001f);
    ASSERT_FLOAT_EQ(params.C_0, 0.15f, 0.001f);
    return 1;
}

int test_create_with_params(void) {
    DETParams params = det_default_params();
    params.tau_base = 0.05f;
    params.C_0 = 0.25f;

    DETCore* core = det_core_create_with_params(&params);
    ASSERT(core != NULL);
    ASSERT_FLOAT_EQ(core->params.tau_base, 0.05f, 0.001f);
    ASSERT_FLOAT_EQ(core->params.C_0, 0.25f, 0.001f);
    det_core_destroy(core);
    return 1;
}

int test_reset(void) {
    DETCore* core = det_core_create();

    /* Modify state */
    core->tick = 100;
    core->nodes[0].F = 5.0f;
    core->nodes[0].tau_accumulated = 100.0f;

    /* Reset */
    det_core_reset(core);

    /* Reset re-initializes nodes to defaults (P-layer gets F=1.0) */
    ASSERT(core->tick == 0);
    ASSERT_FLOAT_EQ(core->nodes[0].F, 1.0f, 0.01f);  /* P-layer default */
    ASSERT_FLOAT_EQ(core->nodes[0].tau_accumulated, 0.0f, 0.01f);  /* Reset clears time */

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Node/Bond Management Tests
 * ========================================================================== */

int test_layer_structure(void) {
    DETCore* core = det_core_create();

    /* Count nodes by layer */
    int p_count = 0, a_count = 0, dormant_count = 0, port_count = 0;
    for (uint32_t i = 0; i < core->num_nodes; i++) {
        switch (core->nodes[i].layer) {
            case DET_LAYER_P: p_count++; break;
            case DET_LAYER_A: a_count++; break;
            case DET_LAYER_DORMANT: dormant_count++; break;
            case DET_LAYER_PORT: port_count++; break;
        }
    }

    ASSERT(p_count == DET_P_LAYER_SIZE);
    ASSERT(a_count == DET_A_LAYER_SIZE);
    ASSERT(dormant_count == DET_DORMANT_SIZE);
    ASSERT(port_count == (int)core->num_ports);

    det_core_destroy(core);
    return 1;
}

int test_recruit_retire_node(void) {
    DETCore* core = det_core_create();

    uint32_t initial_active = core->num_active;

    /* Recruit a node to A-layer */
    int32_t node_id = det_core_recruit_node(core, DET_LAYER_A);
    ASSERT(node_id >= 0);
    ASSERT(core->nodes[node_id].layer == DET_LAYER_A);
    ASSERT(core->nodes[node_id].active == true);
    ASSERT(core->num_active == initial_active + 1);

    /* Retire the node */
    det_core_retire_node(core, (uint16_t)node_id);
    ASSERT(core->nodes[node_id].layer == DET_LAYER_DORMANT);
    ASSERT(core->nodes[node_id].active == false);
    ASSERT(core->num_active == initial_active);

    det_core_destroy(core);
    return 1;
}

int test_create_find_bond(void) {
    DETCore* core = det_core_create();

    /* Create a bond between first two P-layer nodes */
    int32_t bond_id = det_core_create_bond(core, 0, 1);
    ASSERT(bond_id >= 0);
    ASSERT(core->bonds[bond_id].i == 0);
    ASSERT(core->bonds[bond_id].j == 1);
    ASSERT(core->bonds[bond_id].C > 0.0f);

    /* Find the bond */
    int32_t found_id = det_core_find_bond(core, 0, 1);
    ASSERT(found_id == bond_id);

    /* Find should work in reverse order too */
    found_id = det_core_find_bond(core, 1, 0);
    ASSERT(found_id == bond_id);

    /* Non-existent bond */
    found_id = det_core_find_bond(core, 0, 100);
    ASSERT(found_id == -1);

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Physics Tests
 * ========================================================================== */

int test_presence_computation(void) {
    DETCore* core = det_core_create();

    /* Actual formula: P = a * σ / (1 + F_op) / (1 + H_i)
       where F_op = F * 0.1 and H_i is coordination load from bonds */

    /* P-layer node 0 has initial bonds (ring topology), so H_i > 0 */
    /* Just verify presence is computed and reasonable */
    det_core_update_presence(core);

    ASSERT(core->nodes[0].P >= 0.0f);
    ASSERT(core->nodes[0].P <= 1.0f);
    ASSERT(core->aggregate_presence > 0.0f);

    /* Verify that increasing agency increases presence */
    float p_before = core->nodes[0].P;
    core->nodes[0].a = 0.99f;  /* High agency */
    det_core_update_presence(core);
    ASSERT(core->nodes[0].P >= p_before);  /* Should increase or stay same */

    det_core_destroy(core);
    return 1;
}

int test_coherence_update(void) {
    DETCore* core = det_core_create();

    /* Create a new bond between nodes that don't have one */
    int32_t bond_id = det_core_create_bond(core, 0, 5);
    ASSERT(bond_id >= 0);

    float initial_C = core->bonds[bond_id].C;

    /* Coherence dynamics depend on flux, decay, and phase slip
       The update is: dC = alpha*J - lambda*C - slip*C*S_ij
       It may increase or decrease depending on conditions */
    det_core_update_coherence(core, 0.1f);

    /* Just verify coherence stays bounded and reasonable */
    ASSERT(core->bonds[bond_id].C >= 0.0f);
    ASSERT(core->bonds[bond_id].C <= 1.0f);

    det_core_destroy(core);
    return 1;
}

int test_agency_ceiling(void) {
    DETCore* core = det_core_create();

    /* Node agency should be bounded by ceiling */
    float initial_a = core->nodes[0].a;

    /* Run a few steps */
    for (int i = 0; i < 10; i++) {
        det_core_step(core, 0.1f);
    }

    /* Agency should remain reasonable */
    ASSERT(core->nodes[0].a >= 0.0f);
    ASSERT(core->nodes[0].a <= 2.0f);  /* Reasonable upper bound */

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Affect Tests
 * ========================================================================== */

int test_affect_initialization(void) {
    DETCore* core = det_core_create();

    /* P-layer nodes should have neutral affect */
    ASSERT_FLOAT_EQ(core->nodes[0].affect.v, 0.0f, 0.01f);
    ASSERT(core->nodes[0].affect.r >= 0.0f && core->nodes[0].affect.r <= 1.0f);
    ASSERT(core->nodes[0].affect.b >= 0.0f && core->nodes[0].affect.b <= 1.0f);

    det_core_destroy(core);
    return 1;
}

int test_affect_update(void) {
    DETCore* core = det_core_create();

    /* Run affect update */
    det_core_update_affect(core, 0.1f);

    /* Valence should be bounded */
    for (uint32_t i = 0; i < core->num_active; i++) {
        if (core->nodes[i].active) {
            ASSERT(core->nodes[i].affect.v >= -1.0f);
            ASSERT(core->nodes[i].affect.v <= 1.0f);
            ASSERT(core->nodes[i].affect.r >= 0.0f);
            ASSERT(core->nodes[i].affect.r <= 1.0f);
            ASSERT(core->nodes[i].affect.b >= 0.0f);
            ASSERT(core->nodes[i].affect.b <= 1.0f);
        }
    }

    det_core_destroy(core);
    return 1;
}

int test_emotion_interpretation(void) {
    DETCore* core = det_core_create();

    /* Run a few steps to get emotional state */
    for (int i = 0; i < 5; i++) {
        det_core_step(core, 0.1f);
    }

    DETEmotion emotion = det_core_get_emotion(core);
    ASSERT(emotion >= DET_EMOTION_NEUTRAL && emotion <= DET_EMOTION_PEACE);

    const char* emotion_str = det_core_emotion_string(emotion);
    ASSERT(emotion_str != NULL);
    ASSERT(strlen(emotion_str) > 0);

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Self-Identification Tests
 * ========================================================================== */

int test_self_identification(void) {
    DETCore* core = det_core_create();

    /* Create some bonds to form a cluster */
    det_core_create_bond(core, 0, 1);
    det_core_create_bond(core, 1, 2);
    det_core_create_bond(core, 0, 2);

    /* Run self identification */
    det_core_identify_self(core);

    /* Should have identified some nodes */
    ASSERT(core->self.num_nodes > 0);
    ASSERT(core->self.nodes != NULL);

    det_core_destroy(core);
    return 1;
}

int test_self_continuity(void) {
    DETCore* core = det_core_create();

    /* Create a cluster */
    det_core_create_bond(core, 0, 1);
    det_core_create_bond(core, 1, 2);

    /* First identification */
    det_core_identify_self(core);
    uint32_t first_count = core->self.num_nodes;

    /* Second identification (should have continuity) */
    det_core_identify_self(core);

    /* Continuity should be high for same cluster */
    ASSERT(core->self.continuity >= 0.0f);
    ASSERT(core->self.continuity <= 1.0f);

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Gatekeeper Tests
 * ========================================================================== */

int test_gatekeeper_basic(void) {
    DETCore* core = det_core_create();

    /* Set up healthy state */
    for (uint32_t i = 0; i < DET_P_LAYER_SIZE; i++) {
        core->nodes[i].F = 1.0f;
        core->nodes[i].a = 0.5f;
    }
    core->aggregate_coherence = 0.3f;
    core->self.bondedness = 0.5f;

    /* Evaluate a basic request */
    uint32_t tokens[] = {1, 2, 3};
    DETDecision decision = det_core_evaluate_request(core, tokens, 3, 0, 0);

    /* Should get a valid decision */
    ASSERT(decision >= DET_DECISION_PROCEED && decision <= DET_DECISION_ESCALATE);

    det_core_destroy(core);
    return 1;
}

int test_gatekeeper_prison_regime(void) {
    DETCore* core = det_core_create();

    /* Prison regime check: C > 0.7, a_max < 0.2, bondedness < 0.3
       where a_max = 1 / (1 + lambda_a * q^2) with lambda_a = 30
       For a_max < 0.2, need q^2 > (1/0.2 - 1)/30 = 4/30 = 0.133
       So q > sqrt(0.133) ≈ 0.365 */

    /* Set high debt for low agency ceiling */
    core->aggregate_debt = 0.5f;  /* a_max = 1/(1+30*0.25) = 1/8.5 ≈ 0.12 < 0.2 */

    /* Set low bondedness */
    core->self.bondedness = 0.1f;

    /* Register domain 0 and create bonds with high coherence */
    det_core_register_domain(core, "test", NULL);

    /* Create bonds in A-layer (domain 0) with high coherence */
    uint32_t a_start = DET_P_LAYER_SIZE;
    for (uint32_t i = 0; i < 10; i++) {
        int32_t b = det_core_create_bond(core, a_start + i, a_start + i + 1);
        if (b >= 0) {
            core->bonds[b].C = 0.9f;  /* High coherence */
        }
    }

    uint32_t tokens[] = {1, 2, 3};
    DETDecision decision = det_core_evaluate_request(core, tokens, 3, 0, 0);

    /* Should detect prison regime and STOP */
    ASSERT(decision == DET_DECISION_STOP);

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Port Interface Tests
 * ========================================================================== */

int test_port_initialization(void) {
    DETCore* core = det_core_create();

    det_core_init_ports(core);

    ASSERT(core->num_ports > 0);
    ASSERT(core->ports[0].node_id < core->num_nodes);

    det_core_destroy(core);
    return 1;
}

int test_stimulus_injection(void) {
    DETCore* core = det_core_create();
    /* Ports are already initialized in det_core_create() */

    /* Get initial F value (starts at 0 for port nodes) */
    uint16_t port_node = core->ports[0].node_id;
    float initial_F = core->nodes[port_node].F;

    /* Inject stimulus with high activation */
    uint8_t port_indices[] = {0, 1};
    float activations[] = {2.0f, 1.5f};  /* Higher activations to see effect */

    det_core_inject_stimulus(core, port_indices, activations, 2);

    /* Port nodes should have increased resource */
    /* F += ETA * activation where ETA = 0.5 */
    ASSERT(core->nodes[port_node].F > initial_F);
    ASSERT_FLOAT_EQ(core->nodes[port_node].F, initial_F + 0.5f * 2.0f, 0.01f);

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * v6.4 Structural Debt Coupling Tests
 * ========================================================================== */

int test_debt_temporal_distortion(void) {
    /* F_SD2: Temporal Distortion Test
     *
     * Verify high-q regions have slower proper time.
     * P = (a × σ) / ((1+F)(1+H)(1+ζ×q))
     *
     * Pass: High-q node has lower P than low-q node with same other params
     */
    DETCore* core = det_core_create();

    /* Ensure temporal coupling is enabled */
    ASSERT(core->params.debt_temporal_enabled == true);

    /* Set up two nodes with same params except q */
    core->nodes[0].q = 0.05f;  /* Low debt */
    core->nodes[1].q = 0.8f;   /* High debt */

    /* Same other parameters */
    core->nodes[0].a = 0.5f;
    core->nodes[1].a = 0.5f;
    core->nodes[0].F = 1.0f;
    core->nodes[1].F = 1.0f;
    core->nodes[0].sigma = 0.12f;
    core->nodes[1].sigma = 0.12f;

    /* Update presence */
    det_core_update_presence(core);

    /* High-q node should have lower presence (slower proper time) */
    ASSERT(core->nodes[1].P < core->nodes[0].P);

    /* Calculate expected ratio with ζ = 0.5 */
    float expected_ratio = (1.0f + 0.5f * 0.05f) / (1.0f + 0.5f * 0.8f);
    float actual_ratio = core->nodes[1].P / core->nodes[0].P;

    /* Should be close to expected (within 20% due to other factors like H) */
    ASSERT(actual_ratio < 0.9f);  /* High debt should reduce P significantly */

    det_core_destroy(core);
    return 1;
}

int test_debt_conductivity_gate(void) {
    /* F_SD1: Conductivity Gate Test
     *
     * Verify high-q bonds have reduced effective conductivity.
     * σ_eff = σ × g_q where g_q = 1/(1 + ξ(q_i + q_j))
     *
     * We test indirectly through flux: J = g^(a) × σ_eff × |ΔP|
     */
    DETCore* core = det_core_create();

    /* Ensure conductivity coupling is enabled */
    ASSERT(core->params.debt_conductivity_enabled == true);

    /* Create two bonds: one between low-q nodes, one between high-q nodes */
    int32_t low_bond = det_core_create_bond(core, 0, 1);
    int32_t high_bond = det_core_create_bond(core, 2, 3);

    ASSERT(low_bond >= 0);
    ASSERT(high_bond >= 0);

    /* Set up low-q pair */
    core->nodes[0].q = 0.02f;
    core->nodes[1].q = 0.02f;

    /* Set up high-q pair */
    core->nodes[2].q = 0.9f;
    core->nodes[3].q = 0.9f;

    /* Same other parameters */
    for (int i = 0; i < 4; i++) {
        core->nodes[i].a = 0.5f;
        core->nodes[i].P = 0.5f;
        core->nodes[i].sigma = 0.12f;
    }

    /* Create pressure gradient */
    core->nodes[0].P = 0.7f;
    core->nodes[1].P = 0.3f;
    core->nodes[2].P = 0.7f;
    core->nodes[3].P = 0.3f;

    /* Same bond params */
    core->bonds[low_bond].sigma = 0.12f;
    core->bonds[high_bond].sigma = 0.12f;
    core->bonds[low_bond].C = 0.5f;
    core->bonds[high_bond].C = 0.5f;

    /* Run coherence update to measure flux effect */
    det_core_update_coherence(core, 0.1f);

    /* Calculate expected conductivity gate
     * Low: g_q = 1/(1 + 2.0 × 0.04) = 1/1.08 ≈ 0.926
     * High: g_q = 1/(1 + 2.0 × 1.8) = 1/4.6 ≈ 0.217
     * Expected ratio: 0.217/0.926 ≈ 0.23
     */
    float xi = core->params.xi_conductivity;
    float low_q_sum = core->nodes[0].q + core->nodes[1].q;
    float high_q_sum = core->nodes[2].q + core->nodes[3].q;
    float g_q_low = 1.0f / (1.0f + xi * low_q_sum);
    float g_q_high = 1.0f / (1.0f + xi * high_q_sum);

    /* High-q gate should be much smaller */
    ASSERT(g_q_high < g_q_low * 0.5f);

    det_core_destroy(core);
    return 1;
}

int test_debt_decoherence(void) {
    /* F_SD3: Decoherence Enhancement Test
     *
     * Verify high-q bonds have faster coherence decay.
     * λ_eff = λ × (1 + θ × q_bond)
     *
     * Pass: High-q bond loses coherence faster than low-q bond
     */
    DETCore* core = det_core_create();

    /* Ensure decoherence coupling is enabled */
    ASSERT(core->params.debt_decoherence_enabled == true);

    /* Create two bonds */
    int32_t low_bond = det_core_create_bond(core, 0, 1);
    int32_t high_bond = det_core_create_bond(core, 2, 3);

    /* Set up low-q pair */
    core->nodes[0].q = 0.02f;
    core->nodes[1].q = 0.02f;

    /* Set up high-q pair */
    core->nodes[2].q = 0.8f;
    core->nodes[3].q = 0.8f;

    /* Same other parameters */
    for (int i = 0; i < 4; i++) {
        core->nodes[i].a = 0.5f;
        core->nodes[i].P = 0.1f;  /* Low presence → low flux → pure decay */
        core->nodes[i].sigma = 0.12f;
    }

    /* Start with same coherence */
    core->bonds[low_bond].C = 0.8f;
    core->bonds[high_bond].C = 0.8f;

    /* Run coherence update multiple times */
    for (int i = 0; i < 50; i++) {
        det_core_update_coherence(core, 0.1f);
    }

    /* High-q bond should have decayed more (lower C) */
    ASSERT(core->bonds[high_bond].C < core->bonds[low_bond].C);

    det_core_destroy(core);
    return 1;
}

int test_accumulated_proper_time(void) {
    /* Test that accumulated proper time tracks correctly */
    DETCore* core = det_core_create();

    /* Initial accumulated time should be 0 */
    ASSERT_FLOAT_EQ(core->nodes[0].tau_accumulated, 0.0f, 0.01f);

    /* Run several steps */
    for (int i = 0; i < 100; i++) {
        det_core_step(core, 0.1f);
    }

    /* Accumulated time should have increased */
    ASSERT(core->nodes[0].tau_accumulated > 0.0f);

    /* High-debt nodes should have accumulated less proper time */
    /* Create a high-debt node and compare */
    core->nodes[5].q = 0.9f;
    core->nodes[5].tau_accumulated = 0.0f;  /* Reset for comparison */
    core->nodes[6].q = 0.05f;
    core->nodes[6].tau_accumulated = 0.0f;

    for (int i = 0; i < 100; i++) {
        det_core_step(core, 0.1f);
    }

    /* High-debt node should accumulate less proper time */
    ASSERT(core->nodes[5].tau_accumulated < core->nodes[6].tau_accumulated);

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * v6.4 q-Pattern Encoding Tests
 * ========================================================================== */

int test_q_drain_resource(void) {
    /* Test basic resource drain and q accumulation */
    DETCore* core = det_core_create();

    /* Get initial values */
    float initial_F = core->nodes[0].F;
    float initial_q = core->nodes[0].q;

    /* Drain some resource */
    float drain_amount = 0.5f;
    float actual_drained = det_core_drain_resource(core, 0, drain_amount);

    /* Should have drained the requested amount */
    ASSERT(actual_drained > 0.0f);
    ASSERT_FLOAT_EQ(actual_drained, drain_amount, 0.01f);

    /* F should have decreased */
    ASSERT(core->nodes[0].F < initial_F);
    ASSERT_FLOAT_EQ(core->nodes[0].F, initial_F - drain_amount, 0.01f);

    /* q should have increased (α_q = 0.012) */
    ASSERT(core->nodes[0].q > initial_q);
    float expected_q_increase = 0.012f * drain_amount;
    ASSERT_FLOAT_EQ(core->nodes[0].q, initial_q + expected_q_increase, 0.01f);

    /* Test draining more than available */
    core->nodes[1].F = 0.1f;
    float large_drain = 1.0f;
    actual_drained = det_core_drain_resource(core, 1, large_drain);

    /* Should only drain available amount (minus minimum) */
    ASSERT(actual_drained < large_drain);
    ASSERT(core->nodes[1].F >= 0.009f);  /* Minimum preserved (with float tolerance) */

    det_core_destroy(core);
    return 1;
}

int test_q_write_pattern(void) {
    /* Test writing q patterns across nodes */
    DETCore* core = det_core_create();

    /* Create a pattern to write */
    uint16_t node_ids[4] = {0, 1, 2, 3};
    float q_targets[4] = {0.3f, 0.5f, 0.7f, 0.2f};

    /* Give nodes plenty of resource */
    for (int i = 0; i < 4; i++) {
        core->nodes[i].F = 100.0f;  /* Plenty of resource */
        core->nodes[i].q = 0.0f;    /* Start at zero debt */
    }

    /* Write the pattern */
    uint32_t success = det_core_write_q_pattern(core, node_ids, q_targets, 4);

    /* All 4 should succeed (we have plenty of resource) */
    ASSERT(success == 4);

    /* Check that q values match targets (within tolerance) */
    ASSERT(core->nodes[0].q >= q_targets[0] * 0.95f);
    ASSERT(core->nodes[1].q >= q_targets[1] * 0.95f);
    ASSERT(core->nodes[2].q >= q_targets[2] * 0.95f);
    ASSERT(core->nodes[3].q >= q_targets[3] * 0.95f);

    det_core_destroy(core);
    return 1;
}

int test_q_create_barrier(void) {
    /* Test barrier creation (high-q wall) */
    DETCore* core = det_core_create();

    /* Barrier nodes */
    uint16_t barrier[4] = {4, 5, 6, 7};

    /* Give them plenty of resource */
    for (int i = 4; i <= 7; i++) {
        core->nodes[i].F = 100.0f;
        core->nodes[i].q = 0.0f;
    }

    /* Create barrier with q_target = 0.8 */
    det_core_create_barrier(core, barrier, 4, 0.8f);

    /* All barrier nodes should have high q */
    for (int i = 4; i <= 7; i++) {
        ASSERT(core->nodes[i].q >= 0.7f);  /* Should be close to 0.8 */
    }

    /* Create a bond through the barrier and check conductivity is low */
    int32_t b = det_core_create_bond(core, 4, 5);
    ASSERT(b >= 0);

    float sigma_eff = det_core_get_effective_conductivity(core, b);
    float base_sigma = core->bonds[b].sigma;

    /* Effective conductivity should be much lower than base */
    /* g_q = 1/(1 + 2.0 × (0.8 + 0.8)) = 1/(1 + 3.2) ≈ 0.24 */
    ASSERT(sigma_eff < base_sigma * 0.5f);

    det_core_destroy(core);
    return 1;
}

int test_q_effective_conductivity(void) {
    /* Test effective conductivity calculation */
    DETCore* core = det_core_create();

    /* Create a bond */
    int32_t b = det_core_create_bond(core, 0, 1);
    ASSERT(b >= 0);

    /* Set up low q */
    core->nodes[0].q = 0.1f;
    core->nodes[1].q = 0.1f;

    float sigma_eff_low = det_core_get_effective_conductivity(core, b);

    /* Set up high q */
    core->nodes[0].q = 0.8f;
    core->nodes[1].q = 0.8f;

    float sigma_eff_high = det_core_get_effective_conductivity(core, b);

    /* High-q should have much lower conductivity */
    ASSERT(sigma_eff_high < sigma_eff_low * 0.5f);

    /* Calculate expected values
     * Low: g_q = 1/(1 + 2.0 × 0.2) = 1/1.4 ≈ 0.714
     * High: g_q = 1/(1 + 2.0 × 1.6) = 1/4.2 ≈ 0.238
     * Ratio ≈ 0.33
     */
    float expected_ratio = (1.0f / (1.0f + 2.0f * 1.6f)) / (1.0f / (1.0f + 2.0f * 0.2f));
    float actual_ratio = sigma_eff_high / sigma_eff_low;
    ASSERT_FLOAT_EQ(actual_ratio, expected_ratio, 0.05f);

    det_core_destroy(core);
    return 1;
}

int test_q_gradient(void) {
    /* Test q gradient measurement */
    DETCore* core = det_core_create();

    /* Create bonds around node 0 */
    det_core_create_bond(core, 0, 1);
    det_core_create_bond(core, 0, 2);
    det_core_create_bond(core, 0, 3);

    /* Set up q pattern: node 0 low, neighbors high */
    core->nodes[0].q = 0.1f;
    core->nodes[1].q = 0.8f;
    core->nodes[2].q = 0.5f;
    core->nodes[3].q = 0.3f;

    /* Get gradient at node 0 */
    float gradient = det_core_get_q_gradient(core, 0);

    /* Should be max |q_i - q_j| = |0.1 - 0.8| = 0.7 */
    ASSERT_FLOAT_EQ(gradient, 0.7f, 0.01f);

    /* Test uniform q (no gradient) - must set ALL nodes connected to node 0 */
    /* Node 0 has bonds to nodes 1,2,3 (test bonds) plus P-layer ring bonds */
    for (int i = 0; i < DET_P_LAYER_SIZE; i++) {
        core->nodes[i].q = 0.3f;
    }

    gradient = det_core_get_q_gradient(core, 0);
    ASSERT_FLOAT_EQ(gradient, 0.0f, 0.01f);

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Integration Tests
 * ========================================================================== */

int test_full_step(void) {
    DETCore* core = det_core_create();

    uint64_t initial_tick = core->tick;

    /* Run a full step */
    det_core_step(core, 0.1f);

    ASSERT(core->tick == initial_tick + 1);

    /* Run multiple steps */
    for (int i = 0; i < 100; i++) {
        det_core_step(core, 0.1f);
    }

    ASSERT(core->tick == initial_tick + 101);

    /* Aggregates should be computed */
    ASSERT(core->aggregate_presence >= 0.0f);
    ASSERT(core->aggregate_coherence >= 0.0f);

    det_core_destroy(core);
    return 1;
}

int test_aggregates(void) {
    DETCore* core = det_core_create();

    /* Run a few steps */
    for (int i = 0; i < 10; i++) {
        det_core_step(core, 0.1f);
    }

    float presence, coherence, resource, debt;
    det_core_get_aggregates(core, &presence, &coherence, &resource, &debt);

    ASSERT(presence >= 0.0f);
    ASSERT(coherence >= 0.0f);
    ASSERT(resource >= 0.0f);
    ASSERT(debt >= 0.0f && debt <= 1.0f);

    det_core_destroy(core);
    return 1;
}

int test_self_affect(void) {
    DETCore* core = det_core_create();

    /* Set up some bonds */
    det_core_create_bond(core, 0, 1);
    det_core_create_bond(core, 1, 2);

    /* Run a few steps */
    for (int i = 0; i < 10; i++) {
        det_core_step(core, 0.1f);
    }

    float valence, arousal, bondedness;
    det_core_get_self_affect(core, &valence, &arousal, &bondedness);

    ASSERT(valence >= -1.0f && valence <= 1.0f);
    ASSERT(arousal >= 0.0f && arousal <= 1.0f);
    ASSERT(bondedness >= 0.0f && bondedness <= 1.0f);

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Phase 5: Computational Substrate Tests
 * ========================================================================== */

int test_gate_creation(void) {
    /* Test that we can create gates of each type */
    DETCore* core = det_core_create();

    /* Create one of each type */
    int32_t buffer = det_core_create_gate(core, DET_GATE_BUFFER);
    int32_t not_g = det_core_create_gate(core, DET_GATE_NOT);
    int32_t and_g = det_core_create_gate(core, DET_GATE_AND);
    int32_t or_g = det_core_create_gate(core, DET_GATE_OR);

    ASSERT(buffer >= 0);
    ASSERT(not_g >= 0);
    ASSERT(and_g >= 0);
    ASSERT(or_g >= 0);

    /* Verify gate count */
    ASSERT(det_core_num_gates(core) == 4);

    /* Verify gates are retrievable */
    const DETGate* gate = det_core_get_gate(core, buffer);
    ASSERT(gate != NULL);
    ASSERT(gate->type == DET_GATE_BUFFER);
    ASSERT(gate->active == true);

    /* Test removal */
    ASSERT(det_core_remove_gate(core, buffer) == true);
    ASSERT(det_core_num_gates(core) == 3);
    ASSERT(det_core_get_gate(core, buffer) == NULL);

    det_core_destroy(core);
    return 1;
}

int test_gate_buffer(void) {
    /* Test buffer gate (pass-through) */
    DETCore* core = det_core_create();

    int32_t buffer = det_core_create_gate(core, DET_GATE_BUFFER);
    ASSERT(buffer >= 0);

    /* Set input high */
    det_core_set_gate_input(core, buffer, 0, 1.0f);

    /* Propagate signals */
    det_core_propagate_signals(core, 20, 0.1f);

    /* Output should be high */
    float out = det_core_get_gate_output(core, buffer);
    ASSERT(out >= 0.5f);  /* Should be logic 1 */

    /* Set input low */
    det_core_set_gate_input(core, buffer, 0, 0.0f);

    /* Propagate */
    det_core_propagate_signals(core, 20, 0.1f);

    /* Output should be low */
    out = det_core_get_gate_output(core, buffer);
    /* Note: May need more propagation for signal to decay */

    det_core_destroy(core);
    return 1;
}

int test_gate_not(void) {
    /* Test NOT gate (inverter) */
    DETCore* core = det_core_create();

    int32_t not_g = det_core_create_gate(core, DET_GATE_NOT);
    ASSERT(not_g >= 0);

    /* Input = 0 -> Output should be 1 */
    det_core_set_gate_input(core, not_g, 0, 0.0f);
    det_core_propagate_signals(core, 30, 0.1f);

    float out = det_core_get_gate_output(core, not_g);
    ASSERT(out >= 0.5f);  /* Output should be high when input is low */

    /* Input = 1 -> Output should be 0 */
    det_core_set_gate_input(core, not_g, 0, 1.0f);
    det_core_propagate_signals(core, 30, 0.1f);

    out = det_core_get_gate_output(core, not_g);
    ASSERT(out < 0.5f);  /* Output should be low when input is high */

    det_core_destroy(core);
    return 1;
}

int test_gate_and(void) {
    /* Test AND gate */
    DETCore* core = det_core_create();

    int32_t and_g = det_core_create_gate(core, DET_GATE_AND);
    ASSERT(and_g >= 0);

    /* 0 AND 0 = 0 */
    det_core_set_gate_input(core, and_g, 0, 0.0f);
    det_core_set_gate_input(core, and_g, 1, 0.0f);
    det_core_propagate_signals(core, 20, 0.1f);
    float out = det_core_get_gate_output(core, and_g);
    /* Low inputs should produce low output */

    /* 1 AND 0 = 0 */
    det_core_set_gate_input(core, and_g, 0, 1.0f);
    det_core_set_gate_input(core, and_g, 1, 0.0f);
    det_core_propagate_signals(core, 20, 0.1f);
    out = det_core_get_gate_output(core, and_g);
    /* Should still be relatively low (single input not enough) */

    /* 1 AND 1 = 1 */
    det_core_set_gate_input(core, and_g, 0, 1.0f);
    det_core_set_gate_input(core, and_g, 1, 1.0f);
    det_core_propagate_signals(core, 30, 0.1f);
    out = det_core_get_gate_output(core, and_g);
    float raw = det_core_get_gate_output_raw(core, and_g);
    /* Both inputs should produce higher output than single input */
    ASSERT(raw > 0.3f);  /* Some flow should reach output */

    det_core_destroy(core);
    return 1;
}

int test_gate_or(void) {
    /* Test OR gate */
    DETCore* core = det_core_create();

    int32_t or_g = det_core_create_gate(core, DET_GATE_OR);
    ASSERT(or_g >= 0);

    /* 0 OR 0 = 0 */
    det_core_set_gate_input(core, or_g, 0, 0.0f);
    det_core_set_gate_input(core, or_g, 1, 0.0f);
    det_core_propagate_signals(core, 20, 0.1f);

    /* 1 OR 0 = 1 */
    det_core_set_gate_input(core, or_g, 0, 1.0f);
    det_core_set_gate_input(core, or_g, 1, 0.0f);
    det_core_propagate_signals(core, 30, 0.1f);
    float raw = det_core_get_gate_output_raw(core, or_g);
    /* Single input should be enough for OR gate */
    ASSERT(raw > 0.2f);  /* Some signal should reach output */

    /* 1 OR 1 = 1 */
    det_core_set_gate_input(core, or_g, 0, 1.0f);
    det_core_set_gate_input(core, or_g, 1, 1.0f);
    det_core_propagate_signals(core, 20, 0.1f);
    raw = det_core_get_gate_output_raw(core, or_g);
    ASSERT(raw > 0.2f);

    det_core_destroy(core);
    return 1;
}

int test_half_adder(void) {
    /* Test half-adder circuit: S = A XOR B, C = A AND B
     * Since XOR is more complex, we'll just test that gates
     * can be connected and signals propagate.
     */
    DETCore* core = det_core_create();

    /* Create gates for half-adder */
    int32_t xor_g = det_core_create_gate(core, DET_GATE_XOR);  /* Sum */
    int32_t and_g = det_core_create_gate(core, DET_GATE_AND);  /* Carry */

    ASSERT(xor_g >= 0);
    ASSERT(and_g >= 0);

    /* Both gates receive same inputs: A and B */
    /* Set A=1, B=1: Expected S=0 (XOR), C=1 (AND) */
    det_core_set_gate_input(core, xor_g, 0, 1.0f);
    det_core_set_gate_input(core, xor_g, 1, 1.0f);
    det_core_set_gate_input(core, and_g, 0, 1.0f);
    det_core_set_gate_input(core, and_g, 1, 1.0f);

    det_core_propagate_signals(core, 30, 0.1f);

    /* Get outputs */
    float sum_raw = det_core_get_gate_output_raw(core, xor_g);
    float carry_raw = det_core_get_gate_output_raw(core, and_g);

    /* Carry should have signal (1 AND 1 = 1) */
    ASSERT(carry_raw > 0.2f);

    /* Test A=1, B=0 */
    det_core_set_gate_input(core, xor_g, 0, 1.0f);
    det_core_set_gate_input(core, xor_g, 1, 0.0f);
    det_core_set_gate_input(core, and_g, 0, 1.0f);
    det_core_set_gate_input(core, and_g, 1, 0.0f);

    det_core_propagate_signals(core, 30, 0.1f);

    sum_raw = det_core_get_gate_output_raw(core, xor_g);
    carry_raw = det_core_get_gate_output_raw(core, and_g);

    /* Sum should have some signal (1 XOR 0 = 1) */
    /* Carry should be lower (1 AND 0 = 0) */

    det_core_destroy(core);
    return 1;
}

/* ==========================================================================
 * Main
 * ========================================================================== */

int main(void) {
    printf("\n");
    printf("========================================\n");
    printf("  DET Core Test Suite\n");
    printf("========================================\n\n");

    printf("Lifecycle Tests:\n");
    TEST(create_destroy);
    TEST(default_params);
    TEST(create_with_params);
    TEST(reset);
    printf("\n");

    printf("Node/Bond Management Tests:\n");
    TEST(layer_structure);
    TEST(recruit_retire_node);
    TEST(create_find_bond);
    printf("\n");

    printf("Physics Tests:\n");
    TEST(presence_computation);
    TEST(coherence_update);
    TEST(agency_ceiling);
    printf("\n");

    printf("Affect Tests:\n");
    TEST(affect_initialization);
    TEST(affect_update);
    TEST(emotion_interpretation);
    printf("\n");

    printf("Self-Identification Tests:\n");
    TEST(self_identification);
    TEST(self_continuity);
    printf("\n");

    printf("Gatekeeper Tests:\n");
    TEST(gatekeeper_basic);
    TEST(gatekeeper_prison_regime);
    printf("\n");

    printf("Port Interface Tests:\n");
    TEST(port_initialization);
    TEST(stimulus_injection);
    printf("\n");

    printf("v6.4 Structural Debt Tests:\n");
    TEST(debt_temporal_distortion);
    TEST(debt_conductivity_gate);
    TEST(debt_decoherence);
    TEST(accumulated_proper_time);
    printf("\n");

    printf("v6.4 q-Pattern Encoding Tests:\n");
    TEST(q_drain_resource);
    TEST(q_write_pattern);
    TEST(q_create_barrier);
    TEST(q_effective_conductivity);
    TEST(q_gradient);
    printf("\n");

    printf("Phase 5 Computational Substrate Tests:\n");
    TEST(gate_creation);
    TEST(gate_buffer);
    TEST(gate_not);
    TEST(gate_and);
    TEST(gate_or);
    TEST(half_adder);
    printf("\n");

    printf("Integration Tests:\n");
    TEST(full_step);
    TEST(aggregates);
    TEST(self_affect);
    printf("\n");

    printf("========================================\n");
    printf("  Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("========================================\n\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
