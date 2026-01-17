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

    /* Reset */
    det_core_reset(core);

    /* Reset zeroes the struct (except params), node values become 0 */
    ASSERT(core->tick == 0);
    ASSERT_FLOAT_EQ(core->nodes[0].F, 0.0f, 0.01f);

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
