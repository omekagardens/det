/**
 * Reasoner Creature - Pure Existence-Lang Implementation
 * =======================================================
 *
 * Chain-of-thought reasoning using optional LLM assistance.
 * Can reason purely in EL or use llm_call for complex reasoning.
 *
 * Protocol (via bonds):
 *   REQUEST:  {"type": "reason", "question": str, "depth": int}
 *   RESPONSE: {"type": "reasoning", "steps": list, "conclusion": str}
 */

creature ReasonerCreature {
    // DET state
    var F: Register := 75.0;
    var a: float := 0.6;
    var q: float := 0.0;

    // Configuration
    var max_depth: int := 5;
    var use_llm: bool := true;

    // Reasoning state
    var current_depth: int := 0;
    var steps_completed: int := 0;

    // Cost constants
    var step_cost: float := 0.1;
    var llm_step_cost: float := 1.0;

    // Statistics
    var total_reasonings: int := 0;
    var total_steps: int := 0;
    var llm_assisted: int := 0;

    // =========================================================================
    // REASON KERNEL - Perform chain-of-thought reasoning
    // =========================================================================

    kernel Reason {
        in  question: TokenReg;
        in  max_steps: Register;
        out conclusion: TokenReg;
        out step_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            llm_enabled ::= witness(use_llm);
        }

        phase PROPOSE {
            // Estimate cost based on requested depth
            estimated_cost := max_steps * step_cost;
            llm_cost := max_steps * llm_step_cost;

            proposal REASON_WITH_LLM {
                // Use LLM for deep reasoning if agency is high enough
                score = if_past(llm_enabled == true &&
                               current_a >= 0.5 &&
                               current_F >= llm_cost)
                        then current_a * 0.9
                        else 0.0;

                effect {
                    // Construct reasoning prompt
                    result := primitive("llm_call", question);

                    conclusion ::= witness(result);
                    step_count := max_steps;

                    // Update statistics
                    total_reasonings := total_reasonings + 1;
                    total_steps := total_steps + max_steps;
                    llm_assisted := llm_assisted + 1;

                    F := F - llm_cost;
                }
            }

            proposal REASON_SIMPLE {
                // Simple reasoning without LLM
                score = if_past(current_F >= estimated_cost && current_a >= 0.3)
                        then current_a * 0.6
                        else 0.0;

                effect {
                    // For now, return a placeholder
                    // Full EL reasoning would involve step-by-step computation
                    conclusion ::= witness("Reasoning completed in EL");
                    step_count := max_steps;

                    total_reasonings := total_reasonings + 1;
                    total_steps := total_steps + max_steps;

                    F := F - estimated_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < estimated_cost) then 1.0 else 0.0;

                effect {
                    conclusion ::= witness("Insufficient F for reasoning");
                    step_count := 0.0;
                }
            }

            proposal REFUSE_LOW_AGENCY {
                score = if_past(current_a < 0.3) then 1.0 else 0.0;

                effect {
                    conclusion ::= witness("Agency too low for reasoning");
                    step_count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({REASON_WITH_LLM, REASON_SIMPLE, REFUSE_LOW_F, REFUSE_LOW_AGENCY},
                            decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // ANALYZE KERNEL - Analyze a statement for truthfulness
    // =========================================================================

    kernel Analyze {
        in  statement: TokenReg;
        out analysis: TokenReg;
        out confidence: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            llm_enabled ::= witness(use_llm);
        }

        phase PROPOSE {
            proposal ANALYZE_WITH_LLM {
                score = if_past(llm_enabled == true &&
                               current_a >= 0.4 &&
                               current_F >= llm_step_cost)
                        then current_a * 0.85
                        else 0.0;

                effect {
                    result := primitive("llm_call", statement);
                    analysis ::= witness(result);

                    // Confidence based on agency
                    confidence := current_a * 0.9;

                    F := F - llm_step_cost;
                }
            }

            proposal ANALYZE_SIMPLE {
                score = if_past(current_F >= step_cost)
                        then current_a * 0.5
                        else 0.0;

                effect {
                    analysis ::= witness("Analysis requires more context");
                    confidence := current_a * 0.3;

                    F := F - step_cost;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < step_cost) then 1.0 else 0.0;

                effect {
                    analysis ::= witness("Insufficient F");
                    confidence := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({ANALYZE_WITH_LLM, ANALYZE_SIMPLE, REFUSE},
                            decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STATUS KERNEL - Report creature status
    // =========================================================================

    kernel Status {
        out status: TokenReg;

        phase READ {
            f ::= witness(F);
            ag ::= witness(a);
            reasonings ::= witness(total_reasonings);
            steps ::= witness(total_steps);
            assisted ::= witness(llm_assisted);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    status ::= witness(f, ag, reasonings, steps, assisted);
                }
            }
        }

        phase CHOOSE {
            choice := choose({REPORT});
        }

        phase COMMIT {
            commit choice;
        }
    }
}
