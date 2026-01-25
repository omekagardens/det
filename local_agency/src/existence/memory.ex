/**
 * Memory Creature - Pure Existence-Lang Implementation
 * =====================================================
 *
 * Stores and retrieves memories using register arrays.
 * No external primitives needed - pure EL computation.
 *
 * Memory Types (importance weights):
 *   - FACT (0): weight 1.5
 *   - PREFERENCE (1): weight 1.3
 *   - INSTRUCTION (2): weight 2.0 (highest priority)
 *   - CONTEXT (3): weight 1.0
 *   - EPISODE (4): weight 0.8 (lowest priority)
 *
 * Protocol (via bonds):
 *   STORE:  {"type": "store", "content": str, "mem_type": int, "importance": int}
 *   RECALL: {"type": "recall", "query": str, "limit": int}
 *   RESPONSE: {"type": "memories", "results": list}
 */

creature MemoryCreature {
    // DET state
    var F: Register := 50.0;
    var a: float := 0.5;
    var q: float := 0.0;

    // Memory configuration
    var max_memories: int := 100;
    var memory_count: int := 0;

    // Memory storage (simple register arrays)
    // Each memory is stored as: content_hash, type, importance, timestamp, access_count
    var mem_hashes: Register := 0.0;
    var mem_types: Register := 0.0;
    var mem_importance: Register := 0.0;
    var mem_timestamps: Register := 0.0;
    var mem_access_counts: Register := 0.0;

    // Type weights for scoring
    var weight_fact: float := 1.5;
    var weight_preference: float := 1.3;
    var weight_instruction: float := 2.0;
    var weight_context: float := 1.0;
    var weight_episode: float := 0.8;

    // Cost constants
    var store_cost_base: float := 0.05;
    var recall_cost: float := 0.02;
    var prune_cost: float := 0.1;

    // Statistics
    var total_stored: int := 0;
    var total_recalled: int := 0;
    var total_pruned: int := 0;

    // =========================================================================
    // STORE KERNEL - Store a new memory
    // =========================================================================

    kernel Store {
        in  content_hash: Register;
        in  mem_type: Register;
        in  importance: Register;
        out success: Register;
        out memory_id: Register;

        phase READ {
            current_F ::= witness(F);
            current_count ::= witness(memory_count);
            max_count ::= witness(max_memories);
        }

        phase PROPOSE {
            // Cost scales with importance
            cost := store_cost_base * (1.0 + importance / 10.0);

            proposal STORE_MEMORY {
                score = if_past(current_F >= cost && current_count < max_count)
                        then a * 0.9
                        else 0.0;

                effect {
                    // Store memory data
                    mem_hashes := content_hash;
                    mem_types := mem_type;
                    mem_importance := importance;
                    mem_timestamps := primitive("now");
                    mem_access_counts := 0.0;

                    // Update counters
                    memory_count := memory_count + 1;
                    total_stored := total_stored + 1;
                    memory_id := memory_count;

                    // Deduct cost
                    F := F - cost;

                    success := 1.0;
                }
            }

            proposal STORE_AFTER_PRUNE {
                // If full but have F, prune then store
                score = if_past(current_count >= max_count && current_F >= cost + prune_cost)
                        then a * 0.5
                        else 0.0;

                effect {
                    // Prune lowest importance memory
                    total_pruned := total_pruned + 1;
                    memory_count := memory_count - 1;

                    // Now store new memory
                    mem_hashes := content_hash;
                    mem_types := mem_type;
                    mem_importance := importance;
                    mem_timestamps := primitive("now");
                    mem_access_counts := 0.0;

                    memory_count := memory_count + 1;
                    total_stored := total_stored + 1;
                    memory_id := memory_count;

                    F := F - cost - prune_cost;
                    success := 1.0;
                }
            }

            proposal REFUSE_FULL {
                score = if_past(current_count >= max_count && current_F < cost + prune_cost)
                        then 1.0
                        else 0.0;

                effect {
                    success := 0.0;
                    memory_id := 0.0;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < cost) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                    memory_id := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({STORE_MEMORY, STORE_AFTER_PRUNE, REFUSE_FULL, REFUSE_LOW_F},
                            decisiveness = a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // RECALL KERNEL - Retrieve memories matching query
    // =========================================================================

    kernel Recall {
        in  query_hash: Register;
        in  limit: Register;
        out result_count: Register;
        out best_match_id: Register;
        out best_match_score: Register;

        phase READ {
            current_F ::= witness(F);
            current_count ::= witness(memory_count);
        }

        phase PROPOSE {
            proposal DO_RECALL {
                score = if_past(current_F >= recall_cost && current_count > 0)
                        then a
                        else 0.0;

                effect {
                    // Simple matching: return count and best match
                    // Full scoring would require array iteration (future enhancement)
                    result_count := current_count;

                    // Get most recent memory as best match for now
                    best_match_id := current_count;

                    // Score based on recency and importance
                    best_match_score := mem_importance;

                    // Update access count
                    mem_access_counts := mem_access_counts + 1.0;

                    // Update statistics
                    total_recalled := total_recalled + 1;

                    // Deduct cost
                    F := F - recall_cost;
                }
            }

            proposal REFUSE_EMPTY {
                score = if_past(current_count == 0) then 1.0 else 0.0;

                effect {
                    result_count := 0.0;
                    best_match_id := 0.0;
                    best_match_score := 0.0;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < recall_cost) then 1.0 else 0.0;

                effect {
                    result_count := 0.0;
                    best_match_id := 0.0;
                    best_match_score := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_RECALL, REFUSE_EMPTY, REFUSE_LOW_F}, decisiveness = a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // PRUNE KERNEL - Remove lowest importance memories
    // =========================================================================

    kernel Prune {
        in  target_count: Register;
        out removed_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_count ::= witness(memory_count);
        }

        phase PROPOSE {
            // Calculate how many to remove
            to_remove := current_count - target_count;

            proposal DO_PRUNE {
                score = if_past(current_F >= prune_cost && to_remove > 0)
                        then a * 0.7
                        else 0.0;

                effect {
                    // Remove lowest importance memories
                    // Simplified: just decrement count
                    memory_count := target_count;
                    removed_count := to_remove;
                    total_pruned := total_pruned + to_remove;

                    F := F - prune_cost;
                }
            }

            proposal NO_PRUNE_NEEDED {
                score = if_past(to_remove <= 0) then 1.0 else 0.0;

                effect {
                    removed_count := 0.0;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < prune_cost) then 1.0 else 0.0;

                effect {
                    removed_count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_PRUNE, NO_PRUNE_NEEDED, REFUSE_LOW_F}, decisiveness = a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STATUS KERNEL - Report creature status
    // =========================================================================

    kernel Status {
        out current_F: Register;
        out current_a: Register;
        out current_count: Register;
        out stats_stored: Register;
        out stats_recalled: Register;
        out stats_pruned: Register;

        phase READ {
            f ::= witness(F);
            ag ::= witness(a);
            count ::= witness(memory_count);
            stored ::= witness(total_stored);
            recalled ::= witness(total_recalled);
            pruned ::= witness(total_pruned);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    current_F := f;
                    current_a := ag;
                    current_count := count;
                    stats_stored := stored;
                    stats_recalled := recalled;
                    stats_pruned := pruned;
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
