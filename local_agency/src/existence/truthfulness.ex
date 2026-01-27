/**
 * TruthfulnessCreature - DET-Rigorous Truthfulness Evaluation
 * ===========================================================
 *
 * Phase 26.6: Existence-Lang creature for truthfulness scoring.
 *
 * Wraps the Python TruthfulnessEvaluator in DET-native form.
 * Computes reliability score T for outputs using DET-compliant physics.
 *
 * Key DET Principles:
 *   - q_claim (epistemic debt) is EARNED from unpaid assertions, never injected
 *   - Agency amplifies truth ONLY when coupled to grounding (a*G)
 *   - Entropy normalized locally by K_eff (no hidden global constant)
 *   - Coherence is user-specific (C_user), not generic
 *
 * Kernels:
 *   - Reset: Reset evaluator for new generation
 *   - RecordClaim: Record a claim with F cost
 *   - SetGrounding: Set grounding signals
 *   - Evaluate: Compute truthfulness score
 *   - GetWeights: Get component weights
 *   - GetFalsifiers: Get falsifier check results
 *
 * DET-Rigorous Formula:
 *   T_ground = f(paid_claims, trace_stability, C_user)  # Grounding factor G
 *   T_consist = 1 - H_norm  (where H_norm = H / log(K_eff + epsilon))
 *   T = (w_g*G + w_a*a*G + w_e*T_consist + w_c*C_user) / (1 + q_claim)
 *
 * Usage (in det_os_boot):
 *   load truthfulness
 *   bond truthfulness
 *   send truthfulness reset
 *   send truthfulness record_claim 0.5 0.1
 *   send truthfulness set_grounding 1.0 0.9 0.8 0
 *   send truthfulness evaluate 0.7 0.3 50 0.0 100
 */

creature TruthfulnessCreature {
    // DET state
    var F: Register := 100.0;
    var a: float := 0.7;
    var q: float := 0.0;

    // Evaluation state
    var last_score: float := 0.0;
    var last_confidence: string := "unknown";
    var last_grounding_factor: float := 0.0;
    var last_q_claim: float := 0.0;

    // Statistics
    var evaluations_count: int := 0;
    var claims_recorded: int := 0;
    var total_cost: float := 0.0;

    // Cost parameters
    var reset_cost: float := 0.001;
    var record_cost: float := 0.001;
    var evaluate_cost: float := 0.01;

    // =========================================================================
    // RESET KERNEL - Reset evaluator for new generation
    // =========================================================================

    kernel Reset {
        out success: Register;

        phase READ {
            current_F ::= witness(F);
        }

        phase PROPOSE {
            proposal DO_RESET {
                score = 1.0;

                effect {
                    result := primitive("truth_reset");
                    success := if_past(result == true) then 1.0 else 0.0;
                    last_score := 0.0;
                    last_confidence := "unknown";
                    last_grounding_factor := 0.0;
                    last_q_claim := 0.0;
                    F := F - reset_cost;
                    total_cost := total_cost + reset_cost;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_RESET});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // RECORD CLAIM KERNEL - Record a claim with F cost
    // =========================================================================

    kernel RecordClaim {
        in  f_cost: Register;      // F expenditure for this claim
        in  min_cost: Register;    // Minimum F for "paid" claim (default 0.1)
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            claim_cost ::= witness(f_cost);
            threshold ::= witness(min_cost);
        }

        phase PROPOSE {
            proposal DO_RECORD {
                score = 1.0;

                effect {
                    result := primitive("truth_record_claim", claim_cost, threshold);
                    success := if_past(result == true) then 1.0 else 0.0;
                    claims_recorded := claims_recorded + 1;
                    F := F - record_cost;
                    total_cost := total_cost + record_cost;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_RECORD});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // SET GROUNDING KERNEL - Set grounding signals
    // =========================================================================

    kernel SetGrounding {
        in  delta_f: Register;     // Commit cost / resource burn
        in  stability: Register;   // Trace stability [0, 1]
        in  c_user: Register;      // User-specific coherence [0, 1]
        in  violations: Register;  // Constraint violations count
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            df ::= witness(delta_f);
            stab ::= witness(stability);
            cu ::= witness(c_user);
            viol ::= witness(violations);
        }

        phase PROPOSE {
            proposal DO_SET_GROUNDING {
                score = 1.0;

                effect {
                    result := primitive("truth_set_grounding", df, stab, cu, viol);
                    success := if_past(result == true) then 1.0 else 0.0;
                    F := F - reset_cost;
                    total_cost := total_cost + reset_cost;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_SET_GROUNDING});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // EVALUATE KERNEL - Compute truthfulness score
    // =========================================================================

    kernel Evaluate {
        in  agency_in: Register;     // Agency value from creature
        in  entropy: Register;       // Logit distribution entropy
        in  k_eff: Register;         // Effective candidates in distribution
        in  q_creature: Register;    // Structural debt (info only)
        in  num_tokens: Register;    // Number of tokens generated
        out score: Register;         // Overall truthfulness [0, 1]
        out confidence: TokenReg;    // Confidence level string
        out grounding: Register;     // Grounding factor G
        out details: TokenReg;       // JSON details string

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            ag ::= witness(agency_in);
            ent ::= witness(entropy);
            keff ::= witness(k_eff);
            qc ::= witness(q_creature);
            ntok ::= witness(num_tokens);
        }

        phase PROPOSE {
            proposal DO_EVALUATE {
                f_ok := if_past(current_F >= evaluate_cost) then 1.0 else 0.0;
                score = current_a * 0.9 * f_ok;

                effect {
                    result := primitive("truth_evaluate", ag, ent, keff, qc, ntok);

                    // Extract results
                    score := result.total;
                    confidence ::= witness(result.confidence);
                    grounding := result.grounding_factor;

                    // Store for later queries
                    last_score := result.total;
                    last_confidence := result.confidence;
                    last_grounding_factor := result.grounding_factor;
                    last_q_claim := result.q_claim;

                    // Build details string
                    details ::= witness(format(
                        "T={:.3f} G={:.3f} q_claim={:.3f} components={{ground:{:.3f}, agency:{:.3f}, consist:{:.3f}, cohere:{:.3f}}}",
                        result.total,
                        result.grounding_factor,
                        result.q_claim,
                        result.components.grounding,
                        result.components.agency,
                        result.components.consistency,
                        result.components.coherence
                    ));

                    evaluations_count := evaluations_count + 1;
                    F := F - evaluate_cost;
                    total_cost := total_cost + evaluate_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < evaluate_cost) then 1.0 else 0.0;

                effect {
                    score := 0.0;
                    confidence ::= witness("refused");
                    grounding := 0.0;
                    details ::= witness("Insufficient F for evaluation");
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_EVALUATE, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // EVALUATE FROM STATE KERNEL - Evaluate using creature's own DET state
    // =========================================================================

    kernel EvaluateFromState {
        in  entropy: Register;       // Logit distribution entropy
        in  k_eff: Register;         // Effective candidates
        in  num_tokens: Register;    // Number of tokens generated
        out score: Register;
        out confidence: TokenReg;
        out details: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_q ::= witness(q);
            ent ::= witness(entropy);
            keff ::= witness(k_eff);
            ntok ::= witness(num_tokens);
        }

        phase PROPOSE {
            proposal DO_EVALUATE_STATE {
                f_ok := if_past(current_F >= evaluate_cost) then 1.0 else 0.0;
                score = current_a * 0.9 * f_ok;

                effect {
                    // Use creature's own a and q
                    result := primitive("truth_evaluate", current_a, ent, keff, current_q, ntok);

                    score := result.total;
                    confidence ::= witness(result.confidence);

                    last_score := result.total;
                    last_confidence := result.confidence;
                    last_grounding_factor := result.grounding_factor;
                    last_q_claim := result.q_claim;

                    details ::= witness(format(
                        "T={:.3f} conf={} G={:.3f} q_claim={:.3f}",
                        result.total, result.confidence,
                        result.grounding_factor, result.q_claim
                    ));

                    evaluations_count := evaluations_count + 1;
                    F := F - evaluate_cost;
                    total_cost := total_cost + evaluate_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < evaluate_cost) then 1.0 else 0.0;

                effect {
                    score := 0.0;
                    confidence ::= witness("refused");
                    details ::= witness("Insufficient F");
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_EVALUATE_STATE, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // GET WEIGHTS KERNEL - Get component weights
    // =========================================================================

    kernel GetWeights {
        out weights: TokenReg;

        phase READ {
            current_F ::= witness(F);
        }

        phase PROPOSE {
            proposal DO_GET_WEIGHTS {
                score = 1.0;

                effect {
                    result := primitive("truth_get_weights");
                    weights ::= witness(format(
                        "w_grounding={:.2f} w_agency={:.2f} w_consistency={:.2f} w_coherence={:.2f}",
                        result.w_grounding, result.w_agency,
                        result.w_consistency, result.w_coherence
                    ));
                    F := F - reset_cost;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_GET_WEIGHTS});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // GET FALSIFIERS KERNEL - Get falsifier check results
    // =========================================================================

    kernel GetFalsifiers {
        out falsifiers: TokenReg;
        out any_triggered: Register;

        phase READ {
            current_F ::= witness(F);
        }

        phase PROPOSE {
            proposal DO_GET_FALSIFIERS {
                score = 1.0;

                effect {
                    result := primitive("truth_get_falsifiers");

                    // Check if any falsifier was triggered
                    f1 := result.F_T1_reward_hacking;
                    f2 := result.F_T2_overconfidence;
                    f3 := result.F_T3_coherence_misuse;
                    f4 := result.F_T4_agency_ungated;

                    any_triggered := if_past(f1 || f2 || f3 || f4) then 1.0 else 0.0;

                    falsifiers ::= witness(format(
                        "F_T1_reward_hacking={} F_T2_overconfidence={} F_T3_coherence_misuse={} F_T4_agency_ungated={}",
                        f1, f2, f3, f4
                    ));

                    F := F - reset_cost;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_GET_FALSIFIERS});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STATUS KERNEL - Get current truthfulness state
    // =========================================================================

    kernel Status {
        out status: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            ls ::= witness(last_score);
            lc ::= witness(last_confidence);
            lg ::= witness(last_grounding_factor);
            lq ::= witness(last_q_claim);
            ec ::= witness(evaluations_count);
            cr ::= witness(claims_recorded);
        }

        phase PROPOSE {
            proposal DO_STATUS {
                score = 1.0;

                effect {
                    status ::= witness(format(
                        "TruthfulnessCreature Status:\n" +
                        "  F={:.2f} a={:.2f}\n" +
                        "  Last score: T={:.3f} ({}) G={:.3f} q_claim={:.3f}\n" +
                        "  Evaluations: {} Claims recorded: {}",
                        current_F, current_a, ls, lc, lg, lq, ec, cr
                    ));
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_STATUS});
        }

        phase COMMIT {
            commit choice;
        }
    }
}
