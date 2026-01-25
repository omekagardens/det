/**
 * Conversation Creature - DET Interface and Context Manager
 * ==========================================================
 *
 * Phase 21.2: The main interface between users and the DET system.
 * Maintains conversation context, integrates with memory/reasoner/planner,
 * and translates DET state into human-understandable explanations.
 *
 * Key Responsibilities:
 * 1. Multi-turn conversation context management
 * 2. Memory-backed context persistence
 * 3. Automatic context summarization when budget exceeded
 * 4. DET state explanation and translation
 * 5. Command help and guidance
 * 6. Delegation to specialized creatures (Reasoner, Planner)
 *
 * Protocol (via bonds):
 *   REQUEST:  {"type": "chat", "message": str}
 *   REQUEST:  {"type": "explain_det"}
 *   REQUEST:  {"type": "help", "topic": str}
 *   RESPONSE: {"type": "response", "text": str, "context_used": int}
 *
 * Bonds:
 *   - LLMCreature: For text generation
 *   - MemoryCreature: For context persistence
 *   - ReasonerCreature: For complex reasoning tasks
 *   - PlannerCreature: For multi-step planning
 */

creature ConversationCreature {
    // DET state
    var F: Register := 100.0;
    var a: float := 0.75;
    var q: float := 0.0;
    var arousal: float := 0.5;
    var bondedness: float := 0.6;

    // Context window management
    var max_context_tokens: int := 4096;
    var current_context_tokens: int := 0;
    var message_count: int := 0;
    var max_messages: int := 50;

    // Summarization thresholds
    var summarize_threshold: float := 0.8;  // Summarize at 80% capacity
    var summary_target_ratio: float := 0.3; // Compress to 30% of original

    // Bond references (set during Init)
    var llm_bond: Register := 0.0;
    var memory_bond: Register := 0.0;
    var reasoner_bond: Register := 0.0;
    var planner_bond: Register := 0.0;

    // Conversation state
    var conversation_id: Register := 0.0;
    var last_user_message: TokenReg := "";
    var last_response: TokenReg := "";
    var context_summary: TokenReg := "";

    // Cost constants
    var chat_cost: float := 1.0;
    var summarize_cost: float := 2.0;
    var memory_store_cost: float := 0.1;
    var memory_recall_cost: float := 0.05;
    var reason_cost: float := 1.5;
    var plan_cost: float := 2.0;

    // Statistics
    var total_messages: int := 0;
    var total_summaries: int := 0;
    var total_memory_stores: int := 0;
    var total_delegations: int := 0;

    // =========================================================================
    // DET COMMAND KNOWLEDGE - Built-in help system
    // =========================================================================

    // System prompt loaded via primitive (too long for inline string)
    var system_prompt_id: TokenReg := "det_conversation_system";

    // =========================================================================
    // CHAT KERNEL - Main conversation interface
    // =========================================================================

    kernel Chat {
        in  user_message: TokenReg;
        in  include_context: Register;  // 1.0 = use context, 0.0 = fresh
        out response: TokenReg;
        out tokens_used: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_arousal ::= witness(arousal);
            current_bondedness ::= witness(bondedness);
            current_tokens ::= witness(current_context_tokens);
            max_tokens ::= witness(max_context_tokens);
            current_summary ::= witness(context_summary);
            prompt_id ::= witness(system_prompt_id);
        }

        phase PROPOSE {
            // Check if we need to summarize first
            context_ratio := current_tokens / max_tokens;

            // Calculate effective temperature from DET state
            base_temp := 0.7;
            agency_mod := (current_a - 0.5) * 0.2;
            arousal_mod := (current_arousal - 0.5) * 0.15;
            bond_mod := (current_bondedness - 0.5) * 0.1;
            effective_temp := base_temp + agency_mod + arousal_mod - bond_mod;

            proposal CHAT_WITH_CONTEXT {
                // Normal chat with context
                score = if_past(current_F >= chat_cost)
                        then current_a * 0.9
                        else 0.0;

                effect {
                    // Load system prompt via primitive
                    sys_prompt := primitive("get_system_prompt", prompt_id);

                    // Build prompt with context
                    full_prompt := sys_prompt;
                    if_past(include_context > 0.5) {
                        full_prompt := primitive("concat", full_prompt, current_summary);
                    }
                    full_prompt := primitive("concat", full_prompt, user_message);

                    // Call LLM via primitive
                    result := primitive("llm_call_v2", "llama3.2:3b", full_prompt, effective_temp, 1024);

                    response ::= witness(result.text);
                    tokens_used := result.tokens;

                    // Update context tracking
                    current_context_tokens := current_tokens + result.tokens + 50;
                    message_count := message_count + 1;
                    total_messages := total_messages + 1;
                    last_user_message ::= witness(user_message);
                    last_response ::= witness(result.text);

                    F := F - chat_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < chat_cost) then 1.0 else 0.0;

                effect {
                    response ::= witness("Insufficient F for conversation.");
                    tokens_used := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({CHAT_WITH_CONTEXT, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // EXPLAIN_DET KERNEL - Translate DET state to human language
    // =========================================================================

    kernel ExplainDET {
        in  creature_name: TokenReg;
        in  creature_F: Register;
        in  creature_a: Register;
        in  creature_q: Register;
        out explanation: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal EXPLAIN {
                score = if_past(current_F >= 0.5) then current_a else 0.0;

                effect {
                    // Use primitive to format DET explanation
                    result := primitive("explain_det_state", creature_name, creature_F, creature_a, creature_q);
                    explanation ::= witness(result);
                    F := F - 0.5;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < 0.5) then 1.0 else 0.0;

                effect {
                    explanation ::= witness("Low resources for explanation");
                }
            }
        }

        phase CHOOSE {
            choice := choose({EXPLAIN, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // HELP KERNEL - Provide contextual help on DET commands
    // =========================================================================

    kernel Help {
        in  topic: TokenReg;
        out help_text: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            prompt_id ::= witness(system_prompt_id);
        }

        phase PROPOSE {
            proposal PROVIDE_HELP {
                score = if_past(current_F >= 0.5) then current_a else 0.0;

                effect {
                    // Load system prompt and build help query
                    sys_prompt := primitive("get_system_prompt", prompt_id);
                    help_prompt := primitive("concat", sys_prompt, topic);

                    result := primitive("llm_call_v2", "llama3.2:3b", help_prompt, 0.4, 300);
                    help_text ::= witness(result.text);

                    F := F - 0.5;
                }
            }

            proposal BASIC_HELP {
                score = if_past(current_F < 0.5) then 0.5 else 0.0;

                effect {
                    help_text ::= witness("Type help for commands.");
                }
            }
        }

        phase CHOOSE {
            choice := choose({PROVIDE_HELP, BASIC_HELP}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // REASON KERNEL - Delegate complex reasoning to ReasonerCreature
    // =========================================================================

    kernel Reason {
        in  question: TokenReg;
        in  depth: Register;
        out conclusion: TokenReg;
        out reasoning_steps: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            total_cost := reason_cost + (depth * 0.2);

            proposal DO_REASON {
                score = if_past(current_F >= total_cost && current_a >= 0.4)
                        then current_a * 0.85
                        else 0.0;

                effect {
                    // Use reasoning model with chain-of-thought via primitive
                    result := primitive("llm_reason", question, depth);
                    conclusion ::= witness(result.text);
                    reasoning_steps := depth;

                    total_delegations := total_delegations + 1;
                    F := F - total_cost;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < total_cost) then 1.0 else 0.0;

                effect {
                    conclusion ::= witness("Insufficient resources for reasoning");
                    reasoning_steps := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_REASON, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // PLAN KERNEL - Delegate planning to PlannerCreature
    // =========================================================================

    kernel Plan {
        in  task: TokenReg;
        in  max_steps: Register;
        out plan_text: TokenReg;
        out step_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            total_cost := plan_cost + (max_steps * 0.1);

            proposal DO_PLAN {
                score = if_past(current_F >= total_cost && current_a >= 0.5)
                        then current_a * 0.8
                        else 0.0;

                effect {
                    // Use planning primitive
                    result := primitive("llm_plan", task, max_steps);
                    plan_text ::= witness(result.text);
                    step_count := max_steps;

                    total_delegations := total_delegations + 1;
                    F := F - total_cost;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < total_cost) then 1.0 else 0.0;

                effect {
                    plan_text ::= witness("Insufficient resources for planning");
                    step_count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_PLAN, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STORE_MEMORY KERNEL - Store important context to memory
    // =========================================================================

    kernel StoreMemory {
        in  content: TokenReg;
        in  mem_type: Register;    // 0=FACT, 1=PREFERENCE, 2=INSTRUCTION, 3=CONTEXT, 4=EPISODE
        in  importance: Register;
        out success: Register;
        out memory_id: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            cost := memory_store_cost * (1.0 + importance / 5.0);

            proposal STORE {
                score = if_past(current_F >= cost) then current_a else 0.0;

                effect {
                    // Store via memory primitive (integrates with MemoryCreature)
                    content_hash := primitive("hash", content);
                    result := primitive("memory_store", content_hash, mem_type, importance);

                    memory_id := result.id;
                    total_memory_stores := total_memory_stores + 1;
                    success := 1.0;
                    F := F - cost;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < cost) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                    memory_id := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({STORE, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // RECALL_MEMORY KERNEL - Recall relevant memories for context
    // =========================================================================

    kernel RecallMemory {
        in  query: TokenReg;
        in  limit: Register;
        out memories: TokenReg;
        out recall_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal RECALL {
                score = if_past(current_F >= memory_recall_cost) then current_a else 0.0;

                effect {
                    // Recall via memory primitive
                    query_hash := primitive("hash", query);
                    result := primitive("memory_recall", query_hash, limit);

                    memories ::= witness(result.content);
                    recall_count := result.count;
                    F := F - memory_recall_cost;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < memory_recall_cost) then 1.0 else 0.0;

                effect {
                    memories ::= witness("");
                    recall_count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({RECALL, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // SUMMARIZE KERNEL - Manually trigger context summarization
    // =========================================================================

    kernel Summarize {
        out new_summary: TokenReg;
        out tokens_saved: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_summary ::= witness(context_summary);
            current_tokens ::= witness(current_context_tokens);
        }

        phase PROPOSE {
            proposal DO_SUMMARIZE {
                score = if_past(current_F >= summarize_cost && current_tokens > 100)
                        then current_a * 0.8
                        else 0.0;

                effect {
                    // Use summarize primitive
                    result := primitive("llm_summarize", current_summary);

                    new_summary ::= witness(result.text);
                    context_summary ::= witness(result.text);

                    old_tokens := current_tokens;
                    current_context_tokens := result.tokens;
                    tokens_saved := old_tokens - result.tokens;

                    total_summaries := total_summaries + 1;
                    F := F - summarize_cost;
                }
            }

            proposal SKIP {
                score = if_past(current_tokens <= 100) then 1.0 else 0.0;

                effect {
                    new_summary ::= witness(current_summary);
                    tokens_saved := 0.0;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < summarize_cost) then 1.0 else 0.0;

                effect {
                    new_summary ::= witness("");
                    tokens_saved := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_SUMMARIZE, SKIP, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CLEAR KERNEL - Clear conversation context
    // =========================================================================

    kernel Clear {
        out success: Register;
        out cleared_messages: Register;

        phase READ {
            current_count ::= witness(message_count);
        }

        phase PROPOSE {
            proposal DO_CLEAR {
                score = 1.0;

                effect {
                    cleared_messages := current_count;
                    message_count := 0;
                    current_context_tokens := 0;
                    context_summary ::= witness("");
                    last_user_message ::= witness("");
                    last_response ::= witness("");
                    success := 1.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_CLEAR});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STATUS KERNEL - Report conversation status
    // =========================================================================

    kernel Status {
        out status_F: Register;
        out status_a: Register;
        out status_messages: Register;
        out status_tokens: Register;
        out status_summaries: Register;
        out status_delegations: Register;

        phase READ {
            f ::= witness(F);
            ag ::= witness(a);
            msgs ::= witness(total_messages);
            toks ::= witness(current_context_tokens);
            sums ::= witness(total_summaries);
            dels ::= witness(total_delegations);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    status_F := f;
                    status_a := ag;
                    status_messages := msgs;
                    status_tokens := toks;
                    status_summaries := sums;
                    status_delegations := dels;
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
