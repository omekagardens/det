/**
 * LLM Creature - Pure Existence-Lang Implementation
 * ==================================================
 *
 * Handles LLM interaction using substrate primitives.
 * All external I/O goes through the primitive() call.
 *
 * Protocol (via bonds):
 *   REQUEST:  {"type": "think", "prompt": str}
 *   RESPONSE: {"type": "response", "text": str, "tokens": int}
 */

creature LLMCreature {
    // DET state
    var F: Register := 100.0;
    var a: float := 0.7;
    var q: float := 0.0;

    // Configuration
    var temperature: float := 0.7;
    var max_tokens: int := 512;

    // Statistics
    var calls_made: int := 0;
    var tokens_generated: int := 0;
    var total_cost: float := 0.0;

    // Cost constants (per token)
    var cost_per_token: float := 0.01;
    var base_call_cost: float := 1.0;

    // =========================================================================
    // THINK KERNEL - Generate response from prompt
    // =========================================================================

    kernel Think {
        in  prompt: TokenReg;
        out response: TokenReg;
        out token_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            // Estimate cost before calling
            estimated_cost := base_call_cost;

            proposal CALL_LLM {
                // Score based on agency and available F
                score = current_a * 0.9;

                effect {
                    // Use primitive to call LLM
                    result := primitive("llm_call", prompt);

                    // Store response
                    response ::= witness(result);

                    // Update statistics
                    calls_made := calls_made + 1;

                    // Deduct base cost (per-token cost handled by primitive)
                    F := F - base_call_cost;
                    total_cost := total_cost + base_call_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < base_call_cost) then 1.0 else 0.0;

                effect {
                    response ::= witness("Insufficient F for LLM call");
                    token_count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({CALL_LLM, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CHAT KERNEL - Multi-turn conversation
    // =========================================================================

    kernel Chat {
        in  messages: TokenReg;
        out response: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal CALL_CHAT {
                score = current_a * 0.85;

                effect {
                    // Use chat primitive for multi-turn
                    result := primitive("llm_chat", messages);
                    response ::= witness(result);

                    calls_made := calls_made + 1;
                    F := F - 1.5;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < 1.5) then 1.0 else 0.0;

                effect {
                    response ::= witness("Insufficient F");
                }
            }
        }

        phase CHOOSE {
            choice := choose({CALL_CHAT, REFUSE}, decisiveness = current_a);
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
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_calls ::= witness(calls_made);
            current_cost ::= witness(total_cost);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    status ::= witness(current_F, current_a, current_calls, current_cost);
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
