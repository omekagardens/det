/**
 * DET Physics - Existence-Lang v1.1
 * ==================================
 *
 * The fundamental physics of Dynamic Existence Theory expressed in
 * agency-first semantics. This file contains the Tier-0 primitives
 * that all other computation builds upon.
 *
 * Hierarchy:
 *   physics.ex  ← You are here (fundamental laws)
 *   kernel.ex   ← Imports physics.ex (OS services)
 *   *.ex        ← User programs (import kernel.ex)
 *
 * These kernels map directly to substrate v2 effects:
 *   Transfer  → EFFECT_XFER_F (antisymmetric)
 *   Diffuse   → EFFECT_DIFFUSE (symmetric)
 *   Grace     → EFFECT_XFER_F + debt reduction
 *
 * On DET-native hardware, these become silicon operations.
 */

// =============================================================================
// TIER 0: FUNDAMENTAL PHYSICS KERNELS
// =============================================================================

/**
 * Transfer - Antisymmetric Resource Movement
 * ------------------------------------------
 *
 * The most fundamental physics operation. Resource F moves from source
 * to destination, conserved exactly. This is the substrate of all
 * economic and computational operations.
 *
 * Properties:
 *   - Conserved: F_src + F_dst = constant
 *   - Antisymmetric: Transfer(A,B,x) = -Transfer(B,A,x)
 *   - Witnessed: produces immutable token
 *   - Agency-gated: src.a weights willingness
 *
 * Substrate: EFFECT_XFER_F
 */
kernel Transfer {
    in  src: Register;          // Source node (loses F)
    in  dst: Register;          // Destination node (gains F)
    in  amount: float;          // Requested amount
    out actual: float;          // Amount actually transferred
    out witness: TokenReg;      // Immutable record

    phase READ {
        // Read past trace values
        src_F ::= witness(src.F);
        src_a ::= witness(src.a);
        dst_F ::= witness(dst.F);
    }

    phase PROPOSE {
        // Calculate available amount (cannot transfer more than exists)
        available := min(amount, src_F);

        // Agency gates willingness (but doesn't spend agency)
        willing := available * src_a;

        proposal TRANSFER_FULL {
            // Full transfer when source has enough
            score = if_past(available >= amount) then src_a else 0.0;
            effect {
                // Antisymmetric update (substrate guarantees conservation)
                src.F := src_F - amount;
                dst.F := dst_F + amount;
                actual := amount;
                witness ::= witness(XFER_OK, amount);
            }
        }

        proposal TRANSFER_PARTIAL {
            // Partial transfer when source has less than requested
            score = if_past(available < amount && available > 0) then src_a else 0.0;
            effect {
                src.F := src_F - available;
                dst.F := dst_F + available;
                actual := available;
                witness ::= witness(XFER_PARTIAL, available);
            }
        }

        proposal REFUSE {
            // Refuse when agency is too low or nothing available
            score = if_past(willing <= 0 || available <= 0) then 1.0 else 0.0;
            effect {
                actual := 0.0;
                witness ::= witness(XFER_REFUSE, 0.0);
            }
        }
    }

    phase CHOOSE {
        choice := choose({TRANSFER_FULL, TRANSFER_PARTIAL, REFUSE});
    }

    phase COMMIT {
        // Increment event counter on transfer
        if_past(actual > 0) {
            src.k := src.k + 1;
        }
        commit choice;
    }
}

/**
 * Diffuse - Symmetric Flux Exchange
 * ----------------------------------
 *
 * Resource flows across a bond based on gradient and conductivity.
 * Unlike Transfer, Diffuse is bidirectional and symmetric - it represents
 * the natural tendency of gradients to equalize.
 *
 * Properties:
 *   - Symmetric: flow direction determined by gradient
 *   - Bond-mediated: requires coherent bond
 *   - Conserved: F_i + F_j = constant
 *   - Conductivity-limited: σ_ij limits flow rate
 *
 * Substrate: EFFECT_DIFFUSE
 */
kernel Diffuse {
    in  bond: Register;         // Bond connecting two nodes
    in  sigma: float;           // Conductivity (optional override)
    out flux: float;            // Actual flux magnitude
    out witness: TokenReg;

    phase READ {
        // Read bond endpoints
        node_i ::= bond.node_i;
        node_j ::= bond.node_j;

        // Read node states
        F_i ::= witness(node_i.F);
        F_j ::= witness(node_j.F);
        a_i ::= witness(node_i.a);
        a_j ::= witness(node_j.a);

        // Read bond state
        C ::= witness(bond.C);
        bond_sigma ::= witness(bond.sigma);
    }

    phase PROPOSE {
        // Gradient drives flow (i → j if F_i > F_j)
        gradient := F_i - F_j;

        // Conductivity (use override or bond default)
        effective_sigma := if_past(sigma > 0) then sigma else bond_sigma;

        // Coherence gates flow
        flow_rate := effective_sigma * C;

        // Agency weights participation (geometric mean)
        agency_weight := sqrt(a_i * a_j);

        // Compute flux (positive = i→j, negative = j→i)
        raw_flux := gradient * flow_rate * agency_weight * 0.1;

        // Clamp to available resources
        if_past(raw_flux > 0) {
            actual_flux := min(raw_flux, F_i);
        } else {
            actual_flux := max(raw_flux, -F_j);
        }

        proposal DIFFUSE {
            score = if_past(abs(actual_flux) > 0.0001) then C else 0.0;
            effect {
                // Symmetric update
                node_i.F := F_i - actual_flux;
                node_j.F := F_j + actual_flux;
                flux := abs(actual_flux);
                witness ::= witness(DIFFUSE_OK, actual_flux);
            }
        }

        proposal NO_GRADIENT {
            score = if_past(abs(gradient) < 0.0001) then 1.0 else 0.0;
            effect {
                flux := 0.0;
                witness ::= witness(DIFFUSE_NONE, 0.0);
            }
        }

        proposal INCOHERENT {
            score = if_past(C < 0.01) then 1.0 else 0.0;
            effect {
                flux := 0.0;
                witness ::= witness(DIFFUSE_BLOCKED, C);
            }
        }
    }

    phase CHOOSE {
        choice := choose({DIFFUSE, NO_GRADIENT, INCOHERENT});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * Compare - Measurement of Past Traces
 * -------------------------------------
 *
 * Compares two past trace values. This is NOT a logical assertion -
 * it is the act of measuring distinction. Produces a witness token.
 *
 * Properties:
 *   - Reads only past traces (never present)
 *   - Produces immutable witness
 *   - Costs one event (k += 1)
 *   - Epsilon determines equivalence threshold
 *
 * Note: This is "==" in Existence-Lang (trace equality/measurement)
 */
kernel Compare {
    in  a: Register;
    in  b: Register;
    in  epsilon: float;         // Equivalence threshold
    out witness: TokenReg;

    phase READ {
        val_a ::= witness(a.F);
        val_b ::= witness(b.F);
    }

    phase PROPOSE {
        diff := abs(val_a - val_b);

        proposal EQUAL {
            score = if_past(diff < epsilon) then 1.0 else 0.0;
            effect {
                witness ::= witness(CMP_EQ, diff);
            }
        }

        proposal LESS_THAN {
            score = if_past(val_a < val_b - epsilon) then 1.0 else 0.0;
            effect {
                witness ::= witness(CMP_LT, diff);
            }
        }

        proposal GREATER_THAN {
            score = if_past(val_a > val_b + epsilon) then 1.0 else 0.0;
            effect {
                witness ::= witness(CMP_GT, diff);
            }
        }
    }

    phase CHOOSE {
        choice := choose({EQUAL, LESS_THAN, GREATER_THAN});
    }

    phase COMMIT {
        // Measurement costs one event
        a.k := a.k + 1;
        commit choice;
    }
}

/**
 * Distinct - Create Two Distinct Identities
 * ------------------------------------------
 *
 * The ur-choice. Creates two distinct trace identities from void.
 * This is the most fundamental act of agency - differentiation.
 *
 * Properties:
 *   - Creates distinction (not value)
 *   - No ordering implied
 *   - No arithmetic involved
 *   - Costs one event
 *
 * All numbers emerge from counting committed distinctions.
 */
kernel Distinct {
    out id_a: Register;
    out id_b: Register;
    out witness: TokenReg;

    phase COMMIT {
        proposal CREATE_DISTINCTION {
            score = 1.0;
            effect {
                // Allocate two new trace identities
                id_a := alloc_node();
                id_b := alloc_node();

                // They are distinct by construction
                id_a.distinct_from := id_b;
                id_b.distinct_from := id_a;

                // Initial state: no resource, full agency
                id_a.F := 0.0;
                id_a.a := 1.0;
                id_b.F := 0.0;
                id_b.a := 1.0;

                witness ::= witness(DISTINCT_OK, id_a, id_b);
            }
        }
        commit choose({CREATE_DISTINCTION});
    }
}

/**
 * Reconcile - Attempted Unification
 * ----------------------------------
 *
 * The "=" operator. An attempted act of unification, not a logical
 * assertion. May succeed, fail, or be refused.
 *
 * Properties:
 *   - Volitional (requires agency)
 *   - Costly (increments k)
 *   - Witnessed (leaves trace)
 *   - May fail (distinction may be unbridgeable)
 *
 * This is the inverse of Distinct.
 */
kernel Reconcile {
    in  target_a: Register;
    in  target_b: Register;
    out witness: TokenReg;

    phase READ {
        F_a ::= witness(target_a.F);
        F_b ::= witness(target_b.F);
        a_a ::= witness(target_a.a);
        a_b ::= witness(target_b.a);
    }

    phase PROPOSE {
        // Can only reconcile if both have agency
        can_attempt := a_a > 0.1 && a_b > 0.1;

        // Cost of reconciliation based on distinction magnitude
        diff := abs(F_a - F_b);
        reconcile_cost := diff * 0.1;

        // Check if both can pay the cost
        can_pay := F_a >= reconcile_cost && F_b >= reconcile_cost;

        proposal RECONCILE_SUCCESS {
            score = if_past(can_attempt && can_pay && diff < 1.0) then (a_a + a_b) / 2 else 0.0;
            effect {
                // Unify: average the resources
                avg := (F_a + F_b) / 2;
                target_a.F := avg;
                target_b.F := avg;

                // Both pay reconciliation cost
                target_a.F := target_a.F - reconcile_cost / 2;
                target_b.F := target_b.F - reconcile_cost / 2;

                witness ::= witness(EQ_OK);
            }
        }

        proposal RECONCILE_FAIL {
            // Distinction too large to bridge
            score = if_past(can_attempt && diff >= 1.0) then 1.0 else 0.0;
            effect {
                // Partial cost for failed attempt
                target_a.F := F_a - reconcile_cost / 4;
                target_b.F := F_b - reconcile_cost / 4;

                witness ::= witness(EQ_FAIL);
            }
        }

        proposal RECONCILE_REFUSE {
            // Agency refused
            score = if_past(!can_attempt) then 1.0 else 0.0;
            effect {
                witness ::= witness(EQ_REFUSE);
            }
        }
    }

    phase CHOOSE {
        choice := choose({RECONCILE_SUCCESS, RECONCILE_FAIL, RECONCILE_REFUSE});
    }

    phase COMMIT {
        // Reconciliation attempt costs one event
        target_a.k := target_a.k + 1;
        commit choice;
    }
}

// =============================================================================
// TIER 1: DERIVED PHYSICS
// =============================================================================

/**
 * Presence - Compute Presence Value
 * ----------------------------------
 *
 * P = F · C · a
 *
 * Presence determines the "right to appear" in experience.
 * Higher presence = higher scheduling priority.
 */
kernel ComputePresence {
    in  node: Register;
    out P: float;

    phase READ {
        F ::= witness(node.F);
        C ::= witness(node.C_self);
        a ::= witness(node.a);
    }

    phase COMMIT {
        proposal COMPUTE {
            score = 1.0;
            effect {
                P := F * C * a;
                node.P := P;
            }
        }
        commit choose({COMPUTE});
    }
}

/**
 * CoherenceDecay - Natural Bond Degradation
 * ------------------------------------------
 *
 * Bonds decay without use. This represents the "forgetting" of
 * relationships that aren't maintained.
 *
 * C_new = C_old × (1 - γ × dt)
 */
kernel CoherenceDecay {
    in  bond: Register;
    in  dt: float;              // Time delta
    in  gamma: float;           // Decay rate
    out C_new: float;

    phase READ {
        C_old ::= witness(bond.C);
    }

    phase COMMIT {
        proposal DECAY {
            score = 1.0;
            effect {
                decay := gamma * dt;
                C_new := C_old * (1.0 - decay);
                C_new := max(0.0, C_new);  // Clamp to [0,1]
                bond.C := C_new;
            }
        }
        commit choose({DECAY});
    }
}

/**
 * CoherenceStrengthen - Bond Strengthening via Use
 * -------------------------------------------------
 *
 * Successful communication strengthens bonds.
 */
kernel CoherenceStrengthen {
    in  bond: Register;
    in  amount: float;
    out C_new: float;

    phase READ {
        C_old ::= witness(bond.C);
    }

    phase COMMIT {
        proposal STRENGTHEN {
            score = 1.0;
            effect {
                C_new := min(1.0, C_old + amount);
                bond.C := C_new;
            }
        }
        commit choose({STRENGTHEN});
    }
}

// =============================================================================
// TIER 2: GRACE PROTOCOL
// =============================================================================

/**
 * GraceOffer - Donor-side Grace Offer
 * ------------------------------------
 *
 * Grace is the boundary protocol. It provides emergency resources
 * to struggling nodes from those with excess.
 *
 * Properties:
 *   - Agency-gated (not agency-spending)
 *   - Local (bond-mediated)
 *   - Conserved
 *   - Cannot coerce
 */
kernel GraceOffer {
    in  donor: Register;
    in  receiver: Register;
    in  bond: Register;
    in  threshold: float;       // Grace threshold (β_g)
    out offer: float;
    out witness: TokenReg;

    phase READ {
        donor_F ::= witness(donor.F);
        donor_a ::= witness(donor.a);
        receiver_F ::= witness(receiver.F);
        receiver_a ::= witness(receiver.a);
        C ::= witness(bond.C);
    }

    phase PROPOSE {
        // Calculate local mean (simplified: just receiver)
        local_mean := receiver_F;
        F_thresh := threshold * local_mean;

        // Donor's excess
        excess := relu(donor_F - F_thresh);

        // Receiver's need
        need := relu(F_thresh - receiver_F);

        // Quantum gating (high coherence blocks grace)
        Q := relu(1.0 - sqrt(C) / 0.9);

        // Weight by agency (geometric mean) and quantum gate
        w := sqrt(donor_a * receiver_a) * Q;

        // Calculate offer
        raw_offer := 0.1 * excess * w;

        proposal OFFER {
            score = if_past(raw_offer > 0.001 && need > 0.001) then w else 0.0;
            effect {
                offer := raw_offer;
                witness ::= witness(GRACE_OFFER, offer);
            }
        }

        proposal NO_EXCESS {
            score = if_past(excess <= 0.001) then 1.0 else 0.0;
            effect {
                offer := 0.0;
                witness ::= witness(GRACE_NONE_EXCESS);
            }
        }

        proposal NO_NEED {
            score = if_past(need <= 0.001) then 1.0 else 0.0;
            effect {
                offer := 0.0;
                witness ::= witness(GRACE_NONE_NEED);
            }
        }

        proposal BLOCKED_COHERENT {
            score = if_past(Q <= 0.01) then 1.0 else 0.0;
            effect {
                offer := 0.0;
                witness ::= witness(GRACE_BLOCKED, C);
            }
        }
    }

    phase CHOOSE {
        choice := choose({OFFER, NO_EXCESS, NO_NEED, BLOCKED_COHERENT});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * GraceAccept - Receiver-side Grace Acceptance
 * ---------------------------------------------
 *
 * Receiver accepts offered grace, bounded by need.
 */
kernel GraceAccept {
    in  receiver: Register;
    in  offer: float;
    in  need: float;
    out accepted: float;
    out witness: TokenReg;

    phase PROPOSE {
        // Accept up to need (no overfilling)
        actual := min(offer, need);

        proposal ACCEPT {
            score = if_past(actual > 0.001) then 1.0 else 0.0;
            effect {
                accepted := actual;
                witness ::= witness(GRACE_ACCEPT, accepted);
            }
        }

        proposal DECLINE {
            score = if_past(actual <= 0.001) then 1.0 else 0.0;
            effect {
                accepted := 0.0;
                witness ::= witness(GRACE_DECLINE);
            }
        }
    }

    phase CHOOSE {
        choice := choose({ACCEPT, DECLINE});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * GraceFlow - Complete Grace Protocol on Bond
 * --------------------------------------------
 *
 * Executes full grace protocol: offer, accept, transfer.
 * This is the high-level interface typically used.
 */
kernel GraceFlow {
    in  bond: Register;
    in  threshold: float;
    out flow_ij: float;         // Grace from i to j
    out flow_ji: float;         // Grace from j to i
    out net_flow: float;        // Net flow (i→j positive)
    out witness: TokenReg;

    phase READ {
        node_i ::= bond.node_i;
        node_j ::= bond.node_j;

        F_i ::= witness(node_i.F);
        F_j ::= witness(node_j.F);
        a_i ::= witness(node_i.a);
        a_j ::= witness(node_j.a);
        C ::= witness(bond.C);
    }

    phase PROPOSE {
        // Symmetric: compute offers in both directions
        local_mean := (F_i + F_j) / 2;
        F_thresh := threshold * local_mean;

        // i's offer to j
        excess_i := relu(F_i - F_thresh);
        need_j := relu(F_thresh - F_j);
        Q := relu(1.0 - sqrt(C) / 0.9);
        w := sqrt(a_i * a_j) * Q;
        offer_ij := 0.1 * excess_i * w * need_j / max(need_j + 0.001, 1.0);

        // j's offer to i
        excess_j := relu(F_j - F_thresh);
        need_i := relu(F_thresh - F_i);
        offer_ji := 0.1 * excess_j * w * need_i / max(need_i + 0.001, 1.0);

        // Acceptance (bounded by need)
        accepted_ij := min(offer_ij, need_j);
        accepted_ji := min(offer_ji, need_i);

        // Net antisymmetric flow
        G := accepted_ij - accepted_ji;

        proposal FLOW {
            score = if_past(abs(G) > 0.0001) then w else 0.0;
            effect {
                // Antisymmetric transfer
                node_i.F := F_i - G;
                node_j.F := F_j + G;

                // Grace reduces debt
                if_past(G > 0) {
                    node_j.q := max(0, node_j.q - G * 0.5);
                } else {
                    node_i.q := max(0, node_i.q + G * 0.5);
                }

                flow_ij := accepted_ij;
                flow_ji := accepted_ji;
                net_flow := G;
                witness ::= witness(GRACE_FLOW_OK, G);
            }
        }

        proposal NO_FLOW {
            score = if_past(abs(G) <= 0.0001) then 1.0 else 0.0;
            effect {
                flow_ij := 0.0;
                flow_ji := 0.0;
                net_flow := 0.0;
                witness ::= witness(GRACE_FLOW_NONE);
            }
        }
    }

    phase CHOOSE {
        choice := choose({FLOW, NO_FLOW});
    }

    phase COMMIT {
        commit choice;
    }
}

// =============================================================================
// TIER 3: ARITHMETIC FROM AGENCY (Derived, not primitive)
// =============================================================================

/**
 * Add - Addition via Transfer
 * ----------------------------
 *
 * Addition is the repetition of agency action.
 * A + B = transfer both to result.
 *
 * This is NOT a primitive - it's a macro over Transfer.
 */
kernel Add {
    in  src_a: Register;
    in  src_b: Register;
    out result: Register;
    out witness: TokenReg;

    phase COMMIT {
        proposal ADD {
            score = 1.0;
            effect {
                // Allocate result node
                result := alloc_node();

                // Transfer both sources to result
                Transfer(src_a, result, src_a.F, actual_a, w_a);
                Transfer(src_b, result, src_b.F, actual_b, w_b);

                // Result is the union of both
                witness ::= witness(ADD_OK, result.F);
            }
        }
        commit choose({ADD});
    }
}

/**
 * Multiply - Multiplication via Repeated Addition
 * -------------------------------------------------
 *
 * Multiplication is repeated transfer.
 * A × B = add A to result B times.
 *
 * Note: This is expensive for large B. In practice,
 * we use past token to bound iteration.
 */
kernel Multiply {
    in  base: Register;
    in  count_token: TokenReg;  // How many times (past token)
    out result: Register;
    out witness: TokenReg;

    phase READ {
        count ::= witness(count_token);
        base_F ::= witness(base.F);
    }

    phase COMMIT {
        proposal MULTIPLY {
            score = if_past(count > 0) then 1.0 else 0.0;
            effect {
                result := alloc_node();
                result.F := 0.0;

                // Repeated transfer (bounded by past token)
                portion := base_F / count;
                repeat_past(count) {
                    result.F := result.F + portion;
                }

                witness ::= witness(MUL_OK, result.F);
            }
        }

        proposal ZERO {
            score = if_past(count <= 0) then 1.0 else 0.0;
            effect {
                result := alloc_node();
                result.F := 0.0;
                witness ::= witness(MUL_ZERO);
            }
        }

        commit choose({MULTIPLY, ZERO});
    }
}

// =============================================================================
// CONSTANTS (Token Values for Witnesses)
// =============================================================================

// Transfer witnesses
const XFER_OK = 0x0200;
const XFER_PARTIAL = 0x0202;
const XFER_REFUSE = 0x0201;

// Diffuse witnesses
const DIFFUSE_OK = 0x0210;
const DIFFUSE_NONE = 0x0211;
const DIFFUSE_BLOCKED = 0x0212;

// Compare witnesses
const CMP_LT = 0x0010;
const CMP_EQ = 0x0011;
const CMP_GT = 0x0012;

// Distinct witnesses
const DISTINCT_OK = 0x0220;

// Reconcile witnesses
const EQ_OK = 0x0100;
const EQ_FAIL = 0x0101;
const EQ_REFUSE = 0x0102;

// Grace witnesses
const GRACE_OFFER = 0x0230;
const GRACE_ACCEPT = 0x0231;
const GRACE_DECLINE = 0x0232;
const GRACE_NONE_EXCESS = 0x0233;
const GRACE_NONE_NEED = 0x0234;
const GRACE_BLOCKED = 0x0235;
const GRACE_FLOW_OK = 0x0236;
const GRACE_FLOW_NONE = 0x0237;

// Arithmetic witnesses
const ADD_OK = 0x0300;
const MUL_OK = 0x0301;
const MUL_ZERO = 0x0302;

// =============================================================================
// TIER 4: DET v6.3 COLLIDER PHYSICS KERNELS
// =============================================================================
//
// These kernels implement the full DET v6.3/v6.4 physics theory for lattice
// collider simulations. They match the equations in det_theory_card_6_3.md
// and the verified lattice.py implementation.
//
// Key parameters (from Theory Card VII.2):
//   kappa_grav = 5.0    -- Helmholtz screening length
//   mu_grav = 2.0       -- Gravity potential scale
//   beta_g = 10.0       -- Momentum-gravity coupling (5.0 × μ_g)
//   lambda_pi = 0.008   -- Momentum decay rate
//   alpha_pi = 0.12     -- Momentum amplification
//   C_init = 0.15       -- Initial bond coherence

/**
 * ComputePresenceV63 - DET v6.3 Presence Formula
 * -----------------------------------------------
 *
 * P = a·σ / (1 + F) / (1 + H)
 * Δτ = P · dt
 *
 * where:
 *   a = agency [0,1]
 *   σ = processing rate (sigma)
 *   F = resource (proper time dilates with resource depth)
 *   H = local curvature (from gravity potential)
 *
 * This is the "tick rate" of a node - how much proper time passes
 * per coordinate time. High F or high H slows time (gravitational
 * time dilation).
 */
kernel ComputePresenceV63 {
    in  node: Register;
    in  dt: float;              // Coordinate timestep
    in  H_local: float;         // Local curvature (optional, default 0)
    out P: float;               // Presence value
    out Delta_tau: float;       // Proper time elapsed

    phase READ {
        F ::= witness(node.F);
        a ::= witness(node.a);
        sigma ::= witness(node.sigma);
    }

    phase PROPOSE {
        // Full v6.3 presence formula
        // P = aσ / (1+F) / (1+H)
        denom_F := 1.0 + F;
        denom_H := 1.0 + H_local;
        P_raw := a * sigma / denom_F / denom_H;

        // Proper time increment
        tau_inc := P_raw * dt;

        proposal COMPUTE_PRESENCE {
            score = 1.0;
            effect {
                P := P_raw;
                Delta_tau := tau_inc;
                node.P := P;
                node.tau := node.tau + tau_inc;
            }
        }
    }

    phase CHOOSE {
        choice := choose({COMPUTE_PRESENCE});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * DiffusiveFlux - DET v6.3 Diffusive Flow Component
 * --------------------------------------------------
 *
 * J_diff = g · σ · (C + ε) · (1 - √C) · ∇F
 *
 * where:
 *   g = geometric factor
 *   σ = conductivity
 *   C = bond coherence [0,1]
 *   ε = small regularizer (~1e-4)
 *   ∇F = resource gradient across bond
 *
 * High coherence (C→1) suppresses diffusion (quantum freezeout).
 * Low coherence (C→0) allows classical diffusion.
 */
kernel DiffusiveFlux {
    in  bond: Register;
    in  sigma: float;           // Conductivity
    in  g: float;               // Geometric factor (1/r² or constant)
    out flux: float;            // Flux magnitude i→j
    out witness: TokenReg;

    phase READ {
        node_i ::= bond.node_i;
        node_j ::= bond.node_j;

        F_i ::= witness(node_i.F);
        F_j ::= witness(node_j.F);
        C ::= witness(bond.C);
    }

    phase PROPOSE {
        // Gradient (i→j)
        grad := F_i - F_j;

        // Bond conductivity with regularizer
        eps := 0.0001;
        cond := sigma * (C + eps);

        // Quantum suppression: (1 - √C)
        sqrt_C := sqrt(max(C, 0.01));
        drive := (1.0 - sqrt_C) * grad;

        // Full flux
        J_diff := g * cond * drive;

        proposal COMPUTE_FLUX {
            score = 1.0;
            effect {
                flux := J_diff;
                witness ::= witness(FLUX_DIFF_OK, J_diff);
            }
        }
    }

    phase CHOOSE {
        choice := choose({COMPUTE_FLUX});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * MomentumFlux - DET v6.3 Momentum-Driven Flow
 * ---------------------------------------------
 *
 * J_mom = sign(π) · g · |π|
 *
 * where:
 *   π = momentum on bond (scalar in 1D, vector component in 2D/3D)
 *   g = geometric factor
 *
 * Momentum carries resource in the direction of motion,
 * independent of local gradient.
 */
kernel MomentumFlux {
    in  bond: Register;
    in  g: float;               // Geometric factor
    out flux: float;
    out witness: TokenReg;

    phase READ {
        pi ::= witness(bond.pi);
    }

    phase PROPOSE {
        J_mom := g * pi;

        proposal COMPUTE_MOM_FLUX {
            score = 1.0;
            effect {
                flux := J_mom;
                witness ::= witness(FLUX_MOM_OK, J_mom);
            }
        }
    }

    phase CHOOSE {
        choice := choose({COMPUTE_MOM_FLUX});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * GravityFlux - DET v6.3 Gravitational Flow
 * ------------------------------------------
 *
 * J_grav = -g · F · ∇Φ
 *
 * where:
 *   F = local resource (mass density)
 *   ∇Φ = gravity potential gradient
 *
 * Resource flows "downhill" in the gravitational potential.
 */
kernel GravityFlux {
    in  bond: Register;
    in  phi_i: float;           // Gravity potential at node i
    in  phi_j: float;           // Gravity potential at node j
    in  g: float;               // Geometric factor
    out flux: float;
    out witness: TokenReg;

    phase READ {
        node_i ::= bond.node_i;
        F_i ::= witness(node_i.F);
    }

    phase PROPOSE {
        // Potential gradient (i→j)
        grad_phi := phi_i - phi_j;

        // Gravity flux: F flows downhill
        J_grav := -g * F_i * grad_phi;

        proposal COMPUTE_GRAV_FLUX {
            score = 1.0;
            effect {
                flux := J_grav;
                witness ::= witness(FLUX_GRAV_OK, J_grav);
            }
        }
    }

    phase CHOOSE {
        choice := choose({COMPUTE_GRAV_FLUX});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * MomentumUpdate - DET v6.3 Momentum Dynamics
 * --------------------------------------------
 *
 * π_new = π_old + α_π · J_diff · Δτ - λ_π · π · Δτ + β_g · ∇Φ · Δτ
 *
 * where:
 *   α_π = momentum amplification from flow (0.12)
 *   λ_π = momentum decay rate (0.008)
 *   β_g = gravity-momentum coupling (10.0)
 *   J_diff = diffusive flux (drives momentum in flow direction)
 *   ∇Φ = gravity potential gradient (accelerates mass)
 *
 * This implements Newton-like dynamics: momentum accumulates from
 * gravity and decays from friction.
 */
kernel MomentumUpdate {
    in  bond: Register;
    in  J_diff: float;          // Diffusive flux on this bond
    in  grad_phi: float;        // Gravity potential gradient
    in  Delta_tau: float;       // Proper time step
    in  alpha_pi: float;        // Amplification (default 0.12)
    in  lambda_pi: float;       // Decay rate (default 0.008)
    in  beta_g: float;          // Gravity coupling (default 10.0)
    out pi_new: float;
    out witness: TokenReg;

    phase READ {
        pi_old ::= witness(bond.pi);
    }

    phase PROPOSE {
        // Flow contribution (diffusion drives momentum)
        flow_contrib := alpha_pi * J_diff * Delta_tau;

        // Decay (friction)
        decay := lambda_pi * pi_old * Delta_tau;

        // Gravity acceleration
        grav_accel := beta_g * grad_phi * Delta_tau;

        // Update momentum
        pi_updated := pi_old + flow_contrib - decay + grav_accel;

        proposal UPDATE_MOMENTUM {
            score = 1.0;
            effect {
                pi_new := pi_updated;
                bond.pi := pi_new;
                witness ::= witness(MOM_UPDATE_OK, pi_new);
            }
        }
    }

    phase CHOOSE {
        choice := choose({UPDATE_MOMENTUM});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * CoherenceUpdate - DET v6.3 Bond Coherence Dynamics
 * ---------------------------------------------------
 *
 * C_new = clamp(C_old + α_C · |J_diff| · Δτ - λ_C · C · Δτ, C_init, 1)
 *
 * where:
 *   α_C = coherence growth from flow
 *   λ_C = coherence decay rate
 *   C_init = minimum coherence (floor)
 *
 * Coherence increases with flow (entanglement from interaction)
 * and decays naturally (decoherence). Cannot drop below C_init.
 */
kernel CoherenceUpdate {
    in  bond: Register;
    in  J_diff: float;          // Diffusive flux magnitude
    in  Delta_tau: float;       // Proper time step
    in  alpha_C: float;         // Growth rate (default 0.01)
    in  lambda_C: float;        // Decay rate (default 0.001)
    in  C_init: float;          // Floor value (default 0.15)
    out C_new: float;
    out witness: TokenReg;

    phase READ {
        C_old ::= witness(bond.C);
    }

    phase PROPOSE {
        // Growth from flow
        growth := alpha_C * abs(J_diff) * Delta_tau;

        // Natural decay
        decay := lambda_C * C_old * Delta_tau;

        // Update with floor
        C_updated := C_old + growth - decay;
        C_clamped := max(C_init, min(1.0, C_updated));

        proposal UPDATE_COHERENCE {
            score = 1.0;
            effect {
                C_new := C_clamped;
                bond.C := C_new;
                witness ::= witness(COH_UPDATE_OK, C_new);
            }
        }
    }

    phase CHOOSE {
        choice := choose({UPDATE_COHERENCE});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * StructureUpdate - DET v6.3 Structural Debt Dynamics
 * ----------------------------------------------------
 *
 * q_new = clamp(q_old + dq_outflow - γ_q · q · Δτ, 0, 1)
 *
 * where:
 *   dq_outflow = structural debt from net outflow
 *   γ_q = structure decay rate
 *
 * Structure (q) accumulates when a node loses resource through outflow.
 * This represents "structural debt" - the node maintains relationships
 * even after losing the resource that built them.
 *
 * q > 0 enables gravitational mass (a node with structure sources gravity).
 */
kernel StructureUpdate {
    in  node: Register;
    in  net_outflow: float;     // Net outflow this timestep
    in  Delta_tau: float;       // Proper time step
    in  alpha_q: float;         // Outflow → structure rate
    in  gamma_q: float;         // Structure decay rate
    out q_new: float;
    out witness: TokenReg;

    phase READ {
        q_old ::= witness(node.q);
    }

    phase PROPOSE {
        // Only positive outflow generates structure
        dq_out := alpha_q * max(0.0, net_outflow) * Delta_tau;

        // Natural decay
        decay := gamma_q * q_old * Delta_tau;

        // Update with bounds
        q_updated := q_old + dq_out - decay;
        q_clamped := max(0.0, min(1.0, q_updated));

        proposal UPDATE_STRUCTURE {
            score = 1.0;
            effect {
                q_new := q_clamped;
                node.q := q_new;
                witness ::= witness(STRUCT_UPDATE_OK, q_new);
            }
        }
    }

    phase CHOOSE {
        choice := choose({UPDATE_STRUCTURE});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * AgencyUpdateV64 - DET v6.4 Agency with Structural Ceiling
 * ----------------------------------------------------------
 *
 * a_new = clamp(a_old + (a_tgt - a_old) · κ_a · Δτ, 0, 1)
 *
 * where:
 *   a_tgt = min(1 - q, a_mean_neighbors)
 *
 * Key v6.4 feature: structure (q) creates a ceiling on agency.
 * A node with high structural debt cannot have high agency.
 * This prevents "zombie" nodes that maintain structure without agency.
 *
 * Additionally, agency relaxes toward the mean of neighbors
 * (relational drive toward equilibrium).
 */
kernel AgencyUpdateV64 {
    in  node: Register;
    in  a_mean: float;          // Mean agency of neighbors
    in  Delta_tau: float;       // Proper time step
    in  kappa_a: float;         // Relaxation rate (default 0.1)
    out a_new: float;
    out witness: TokenReg;

    phase READ {
        a_old ::= witness(node.a);
        q ::= witness(node.q);
    }

    phase PROPOSE {
        // Structural ceiling: a ≤ 1 - q
        ceiling := 1.0 - q;

        // Target is min of ceiling and neighbor mean
        a_tgt := min(ceiling, a_mean);

        // Relaxation dynamics
        delta := (a_tgt - a_old) * kappa_a * Delta_tau;
        a_updated := a_old + delta;

        // Clamp to [0, 1]
        a_clamped := max(0.0, min(1.0, a_updated));

        proposal UPDATE_AGENCY {
            score = 1.0;
            effect {
                a_new := a_clamped;
                node.a := a_new;
                witness ::= witness(AGENCY_UPDATE_OK, a_new);
            }
        }
    }

    phase CHOOSE {
        choice := choose({UPDATE_AGENCY});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * GraceInjection - DET v6.3 Boundary Grace Protocol
 * --------------------------------------------------
 *
 * Injects emergency resource to depleted nodes using dissipation pool.
 *
 * n_i = max(0, F_MIN - F_i)     -- need
 * w_i = a_i · n_i               -- weighted need
 * I_g = D · w_i / Σw            -- grace injection from dissipation D
 *
 * Grace preserves mass conservation by using the dissipation pool
 * (resource lost to numerical errors, floor, etc.) as the source.
 */
kernel GraceInjection {
    in  node: Register;
    in  dissipation_pool: float;  // Total dissipation available
    in  total_weighted_need: float;  // Σw across all nodes
    in  F_MIN: float;            // Grace threshold
    out injected: float;
    out witness: TokenReg;

    phase READ {
        F ::= witness(node.F);
        a ::= witness(node.a);
    }

    phase PROPOSE {
        // Calculate need
        need := max(0.0, F_MIN - F);

        // Weighted need
        w := a * need;

        // Grace injection (share of dissipation pool)
        // Avoid division by zero
        w_total := max(total_weighted_need, 0.0001);
        I_g := dissipation_pool * w / w_total;

        proposal INJECT_GRACE {
            score = if_past(need > 0.0001) then a else 0.0;
            effect {
                node.F := F + I_g;
                // Grace reduces structural debt
                node.q := max(0.0, node.q - I_g * 0.1);
                injected := I_g;
                witness ::= witness(GRACE_INJECT_OK, I_g);
            }
        }

        proposal NO_NEED {
            score = if_past(need <= 0.0001) then 1.0 else 0.0;
            effect {
                injected := 0.0;
                witness ::= witness(GRACE_INJECT_NONE);
            }
        }
    }

    phase CHOOSE {
        choice := choose({INJECT_GRACE, NO_NEED});
    }

    phase COMMIT {
        commit choice;
    }
}

/**
 * ConservativeLimiter - DET v6.3 Flux Limiter
 * --------------------------------------------
 *
 * Limits outflow to prevent nodes from going negative.
 *
 * max_out = outflow_limit · F / Δτ
 * scale = min(1, max_out / total_outflow)
 *
 * Applied at BONDS (not nodes) to preserve mass conservation.
 * Only limits positive (outgoing) flux components.
 */
kernel ConservativeLimiter {
    in  node: Register;
    in  total_outflow: float;   // Sum of positive flux from this node
    in  outflow_limit: float;   // Fraction of F that can leave (default 0.25)
    out scale: float;           // Limiter scale factor [0, 1]
    out witness: TokenReg;

    phase READ {
        F ::= witness(node.F);
        Delta_tau ::= witness(node.Delta_tau);
    }

    phase PROPOSE {
        // Maximum outflow this timestep
        eps := 0.000001;
        max_out := outflow_limit * F / (Delta_tau + eps);

        // Scale factor (1.0 = no limiting needed)
        raw_scale := max_out / (total_outflow + eps);
        scale_clamped := min(1.0, raw_scale);

        proposal COMPUTE_LIMIT {
            score = 1.0;
            effect {
                scale := scale_clamped;
                witness ::= witness(LIMITER_OK, scale);
            }
        }
    }

    phase CHOOSE {
        choice := choose({COMPUTE_LIMIT});
    }

    phase COMMIT {
        commit choice;
    }
}

// =============================================================================
// COLLIDER PHYSICS CONSTANTS
// =============================================================================

// Flux witnesses
const FLUX_DIFF_OK = 0x0400;
const FLUX_MOM_OK = 0x0401;
const FLUX_GRAV_OK = 0x0402;

// Momentum witnesses
const MOM_UPDATE_OK = 0x0410;

// Coherence witnesses
const COH_UPDATE_OK = 0x0420;

// Structure witnesses
const STRUCT_UPDATE_OK = 0x0430;

// Agency witnesses
const AGENCY_UPDATE_OK = 0x0440;

// Grace injection witnesses
const GRACE_INJECT_OK = 0x0450;
const GRACE_INJECT_NONE = 0x0451;

// Limiter witnesses
const LIMITER_OK = 0x0460;
