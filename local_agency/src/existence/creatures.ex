/**
 * DET-OS Creatures - Written in Existence-Lang
 * =============================================
 *
 * Standard creature types for the DET-OS ecosystem.
 * Each creature is a first-class entity with F, a, and bonds.
 *
 * Creature Types:
 *   - MemoryCreature: Store and recall memories via bonds
 *   - ToolCreature: Execute commands in sandboxed environment
 *   - ReasonerCreature: Chain-of-thought processing
 *   - PlannerCreature: Task decomposition and planning
 *
 * All creatures communicate via bond channels using the IPC protocol.
 */

import physics.{
    Transfer,
    Diffuse,
    ComputePresence,
    CoherenceDecay,
    CoherenceStrengthen
};

// =============================================================================
// MEMORY TYPES - Enum for memory classification
// =============================================================================

enum MemoryType {
    FACT := 0;          // Factual information
    PREFERENCE := 1;    // User preferences
    INSTRUCTION := 2;   // Standing instructions (highest priority)
    CONTEXT := 3;       // Conversation context
    EPISODE := 4;       // Episode summaries (lowest priority)
}

// Memory type weights for recall scoring
const MEMORY_WEIGHTS: float[5] := [1.5, 1.3, 2.0, 1.0, 0.8];

// =============================================================================
// MEMORY CREATURE - Store and Recall Memories
// =============================================================================

creature MemoryCreature {
    /**
     * Memory Creature stores and retrieves memories via bond channels.
     *
     * Protocol:
     *   STORE: {"type": "store", "content": str, "memory_type": str, "importance": int}
     *   RECALL: {"type": "recall", "query": str, "limit": int, "memory_types": list}
     *   RESPONSE: {"type": "response", "memories": list}
     *
     * Storage costs F proportional to content length and importance.
     * Recall costs fixed F per query.
     */

    // Identity
    var cid: int := 0;
    var name: string := "memory";
    var parent_cid: int := 0;

    // DET state
    var F: Register := 50.0;
    var a: float := 0.5;
    var q: float := 0.0;
    var C_self: float := 1.0;

    // Memory storage
    var memories: Register[1000];       // Memory entries
    var memory_count: int := 0;
    var max_memories: int := 1000;

    // Memory entry structure (packed into Register)
    // Each memory: [content_ptr, type, importance, timestamp, access_count, source_cid]

    // Cost constants
    const STORE_COST_PER_100_CHARS: float := 0.1;
    const RECALL_COST: float := 0.05;

    // Statistics
    var total_stored: int := 0;
    var total_recalled: int := 0;

    // =========================================================================
    // STORE KERNEL - Store a new memory
    // =========================================================================

    kernel Store {
        /**
         * Store a memory entry.
         * Cost scales with content length and importance.
         */

        in  content: TokenReg;          // Memory content (immutable)
        in  memory_type: int;           // MemoryType enum value
        in  importance: int;            // 1-10 importance score
        in  source: int;                // Source creature ID
        out success: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_count ::= witness(memory_count);
        }

        phase PROPOSE {
            // Calculate cost: base cost * importance factor
            content_len := len(content);
            base_cost := (content_len / 100.0) * STORE_COST_PER_100_CHARS;
            importance_factor := 0.5 + (importance / 20.0);
            cost := max(0.01, base_cost * importance_factor);

            proposal STORE_MEMORY {
                score = if_past(current_F >= cost && current_count < max_memories)
                        then a
                        else 0.0;
                effect {
                    // Deduct cost
                    F := F - cost;

                    // Store memory entry
                    idx := memory_count;
                    memories[idx] := create_memory_entry(
                        content,
                        memory_type,
                        importance,
                        now(),
                        0,  // access_count
                        source
                    );

                    memory_count := memory_count + 1;
                    total_stored := total_stored + 1;

                    success ::= witness(true, idx);
                }
            }

            proposal DENY_INSUFFICIENT_F {
                score = if_past(current_F < cost) then 1.0 else 0.0;
                effect {
                    success ::= witness(false, "Insufficient F");
                }
            }

            proposal DENY_FULL {
                score = if_past(current_count >= max_memories) then 1.0 else 0.0;
                effect {
                    // Prune before denying
                    prune_memories();
                    success ::= witness(false, "Memory full, pruned");
                }
            }
        }

        phase CHOOSE {
            choice := choose({STORE_MEMORY, DENY_INSUFFICIENT_F, DENY_FULL});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // RECALL KERNEL - Retrieve matching memories
    // =========================================================================

    kernel Recall {
        /**
         * Recall memories matching a query.
         * Scoring considers: keyword match, type weight, importance, recency, access count.
         */

        in  query: TokenReg;            // Query string
        in  limit: int;                 // Max results
        in  type_filter: int[];         // Optional type filter (empty = all)
        out results: TokenReg[];        // Matching memories

        phase READ {
            current_F ::= witness(F);
            all_memories ::= memories[0..memory_count];
        }

        phase PROPOSE {
            proposal RECALL_MEMORIES {
                score = if_past(current_F >= RECALL_COST) then a else 0.0;
                effect {
                    F := F - RECALL_COST;
                    total_recalled := total_recalled + 1;

                    // Score all memories
                    scored := [];
                    repeat_past(memory_count) as i {
                        mem := memories[i];
                        mem_type := get_memory_type(mem);

                        // Type filter check
                        if_past(len(type_filter) == 0 || mem_type in type_filter) {
                            score := compute_match_score(mem, query);
                            if_past(score > 0) {
                                scored.push((i, score));
                            }
                        }
                    }

                    // Sort by score descending
                    sorted := sort_descending(scored, by = score);

                    // Take top results
                    results := [];
                    count := min(limit, len(sorted));
                    repeat_past(count) as i {
                        idx := sorted[i].0;
                        mem := memories[idx];

                        // Increment access count
                        mem.access_count := mem.access_count + 1;

                        results.push(mem);
                    }
                }
            }

            proposal DENY_INSUFFICIENT_F {
                score = if_past(current_F < RECALL_COST) then 1.0 else 0.0;
                effect {
                    results := [];
                }
            }
        }

        phase CHOOSE {
            choice := choose({RECALL_MEMORIES, DENY_INSUFFICIENT_F});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // PROCESS MESSAGES KERNEL - Handle incoming IPC
    // =========================================================================

    kernel ProcessMessages {
        /**
         * Process incoming messages from all bonded creatures.
         * Dispatches to Store/Recall based on message type.
         */

        in  channels: Register[];       // All bond channels
        out processed: int;             // Number of messages processed

        phase READ {
            pending_messages ::= get_all_pending(channels);
        }

        phase PROPOSE {
            proposal PROCESS_ALL {
                score = if_past(len(pending_messages) > 0) then a else 0.0;
                effect {
                    processed := 0;

                    repeat_past(len(pending_messages)) as i {
                        msg := pending_messages[i];
                        sender := msg.sender;

                        if_past(msg.type == "store") {
                            Store(
                                msg.content,
                                msg.memory_type,
                                msg.importance,
                                sender,
                                result
                            );

                            // Send acknowledgment
                            send_to(sender, {
                                type: "store_ack",
                                success: result.success,
                                memory_type: msg.memory_type,
                                importance: msg.importance
                            });
                        }
                        else if_past(msg.type == "recall") {
                            Recall(
                                msg.query,
                                msg.limit,
                                msg.memory_types,
                                memories
                            );

                            // Send response
                            send_to(sender, {
                                type: "response",
                                query: msg.query,
                                count: len(memories),
                                memories: memories
                            });
                        }
                        else if_past(msg.type == "get_instructions") {
                            // Special: get all instruction memories
                            instructions := get_by_type(INSTRUCTION);
                            send_to(sender, {
                                type: "instructions",
                                count: len(instructions),
                                instructions: instructions
                            });
                        }

                        processed := processed + 1;
                    }
                }
            }

            proposal NO_MESSAGES {
                score = if_past(len(pending_messages) == 0) then 1.0 else 0.0;
                effect {
                    processed := 0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({PROCESS_ALL, NO_MESSAGES});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // HELPER: Compute match score
    // =========================================================================

    func compute_match_score(memory: Register, query: TokenReg) -> float {
        content := get_content(memory);
        mem_type := get_memory_type(memory);
        importance := get_importance(memory);
        timestamp := get_timestamp(memory);
        access_count := get_access_count(memory);

        // Keyword matching (simplified)
        query_words := tokenize(query);
        content_words := tokenize(content);
        matched := count_intersection(query_words, content_words);
        keyword_score := matched / len(query_words);

        // Type weight
        type_weight := MEMORY_WEIGHTS[mem_type];

        // Importance boost (normalize around 1.0)
        importance_boost := importance / 5.0;

        // Recency boost
        age_hours := (now() - timestamp) / 3600.0;
        recency_boost := 1.0 / (1.0 + age_hours * 0.05);

        // Access boost
        access_boost := 1.0 + (access_count * 0.1);

        return keyword_score * type_weight * importance_boost * recency_boost * access_boost;
    }

    // =========================================================================
    // HELPER: Prune low-value memories
    // =========================================================================

    func prune_memories() {
        // Never prune instructions
        instructions := [];
        others := [];

        repeat_past(memory_count) as i {
            mem := memories[i];
            if_past(get_memory_type(mem) == INSTRUCTION) {
                instructions.push(mem);
            } else {
                others.push(mem);
            }
        }

        // Sort others by value (relevance * access)
        sorted := sort_descending(others, by = value_score);

        // Keep instructions + top others
        keep_count := max_memories - len(instructions);
        kept := instructions ++ sorted[0..keep_count];

        // Rebuild memory array
        memory_count := len(kept);
        repeat_past(memory_count) as i {
            memories[i] := kept[i];
        }
    }

    // =========================================================================
    // AGENCY - Main behavior loop
    // =========================================================================

    agency {
        repeat_past(INFINITY) {
            // Process any pending messages
            channels := get_bonded_channels();
            ProcessMessages(channels, processed);

            // Yield for a tick
            yield();
        }
    }

    // =========================================================================
    // GRACE - Recovery when F depleted
    // =========================================================================

    grace {
        // Request grace injection
        request_grace(10.0);
    }
}

// =============================================================================
// TOOL CREATURE - Execute Commands in Sandbox
// =============================================================================

creature ToolCreature {
    /**
     * Tool Creature executes commands in a sandboxed environment.
     * All execution costs F proportional to resource usage.
     *
     * Protocol:
     *   EXECUTE: {"type": "execute", "command": str, "timeout": int}
     *   RESULT: {"type": "result", "success": bool, "output": str, "error": str}
     *
     * Security:
     *   - Commands are sandboxed with resource limits
     *   - Agency level determines allowed operations
     *   - All executions are witnessed (immutable audit trail)
     */

    // Identity
    var cid: int := 0;
    var name: string := "tool";
    var parent_cid: int := 0;

    // DET state
    var F: Register := 30.0;
    var a: float := 0.6;            // Moderate agency for tool execution
    var q: float := 0.0;
    var C_self: float := 1.0;

    // Execution state
    var current_command: TokenReg;
    var is_executing: bool := false;
    var execution_count: int := 0;

    // Resource limits
    var max_cpu_ms: int := 5000;     // 5 second CPU limit
    var max_memory_mb: int := 256;   // 256 MB memory limit
    var max_output_bytes: int := 65536;

    // Cost constants
    const BASE_EXEC_COST: float := 0.5;
    const CPU_COST_PER_SEC: float := 0.1;
    const MEMORY_COST_PER_MB: float := 0.01;

    // =========================================================================
    // EXECUTE KERNEL - Run a command
    // =========================================================================

    kernel Execute {
        /**
         * Execute a command in sandbox.
         * Costs F based on resource consumption.
         */

        in  command: TokenReg;          // Command to execute
        in  timeout_ms: int;            // Timeout in milliseconds
        in  requester: int;             // Requesting creature
        out result: TokenReg;           // Execution result

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            command_risk ::= analyze_command_risk(command);
        }

        phase PROPOSE {
            // Risk levels require different agency thresholds
            // SAFE: a >= 0.3, MODERATE: a >= 0.5, HIGH: a >= 0.7, CRITICAL: a >= 0.9
            risk_thresholds := [0.3, 0.5, 0.7, 0.9];
            required_a := risk_thresholds[command_risk];

            proposal EXECUTE_COMMAND {
                score = if_past(current_F >= BASE_EXEC_COST && current_a >= required_a)
                        then a
                        else 0.0;
                effect {
                    // Mark as executing
                    is_executing := true;
                    current_command := command;

                    // Witness the execution attempt
                    exec_witness ::= witness(command, requester, now());

                    // Actually execute in sandbox
                    sandbox_result := sandbox_execute(
                        command,
                        timeout_ms,
                        max_cpu_ms,
                        max_memory_mb,
                        max_output_bytes
                    );

                    // Calculate actual cost
                    cpu_used := sandbox_result.cpu_ms / 1000.0;
                    mem_used := sandbox_result.memory_mb;
                    actual_cost := BASE_EXEC_COST +
                                   (cpu_used * CPU_COST_PER_SEC) +
                                   (mem_used * MEMORY_COST_PER_MB);

                    // Deduct cost
                    F := F - actual_cost;

                    // Build result
                    result ::= witness(
                        sandbox_result.success,
                        sandbox_result.output,
                        sandbox_result.error,
                        actual_cost
                    );

                    is_executing := false;
                    execution_count := execution_count + 1;
                }
            }

            proposal DENY_INSUFFICIENT_F {
                score = if_past(current_F < BASE_EXEC_COST) then 1.0 else 0.0;
                effect {
                    result ::= witness(false, "", "Insufficient F for execution");
                }
            }

            proposal DENY_INSUFFICIENT_AGENCY {
                score = if_past(current_a < required_a) then 1.0 else 0.0;
                effect {
                    result ::= witness(
                        false, "",
                        "Insufficient agency for risk level " + command_risk
                    );
                    // Log security event
                    audit_log(requester, "EXECUTE", command, "DENIED:LOW_AGENCY");
                }
            }
        }

        phase CHOOSE {
            choice := choose({EXECUTE_COMMAND, DENY_INSUFFICIENT_F, DENY_INSUFFICIENT_AGENCY});
        }

        phase COMMIT {
            commit choice;
            // Record execution in trace
            exec_trace ::= witness(command, result, execution_count);
        }
    }

    // =========================================================================
    // PROCESS MESSAGES KERNEL - Handle incoming IPC
    // =========================================================================

    kernel ProcessMessages {
        in  channels: Register[];
        out processed: int;

        phase READ {
            pending_messages ::= get_all_pending(channels);
        }

        phase PROPOSE {
            proposal PROCESS_ALL {
                score = if_past(len(pending_messages) > 0 && !is_executing) then a else 0.0;
                effect {
                    processed := 0;

                    repeat_past(len(pending_messages)) as i {
                        msg := pending_messages[i];
                        sender := msg.sender;

                        if_past(msg.type == "execute") {
                            Execute(
                                msg.command,
                                msg.timeout || 5000,
                                sender,
                                result
                            );

                            // Send result back
                            send_to(sender, {
                                type: "result",
                                success: result.success,
                                output: result.output,
                                error: result.error,
                                cost: result.cost
                            });
                        }

                        processed := processed + 1;
                    }
                }
            }

            proposal BUSY {
                score = if_past(is_executing) then 1.0 else 0.0;
                effect {
                    // Queue messages for later
                    processed := 0;
                }
            }

            proposal NO_MESSAGES {
                score = if_past(len(pending_messages) == 0) then 1.0 else 0.0;
                effect {
                    processed := 0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({PROCESS_ALL, BUSY, NO_MESSAGES});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // HELPER: Analyze command risk level
    // =========================================================================

    func analyze_command_risk(command: TokenReg) -> int {
        // Returns 0=SAFE, 1=MODERATE, 2=HIGH, 3=CRITICAL
        cmd_lower := lowercase(command);

        // Critical: rm -rf, dd, mkfs, etc.
        if_past(contains(cmd_lower, "rm -rf") ||
                contains(cmd_lower, "dd if=") ||
                contains(cmd_lower, "mkfs")) {
            return 3;  // CRITICAL
        }

        // High: sudo, chmod, network commands
        if_past(contains(cmd_lower, "sudo") ||
                contains(cmd_lower, "chmod") ||
                contains(cmd_lower, "curl") ||
                contains(cmd_lower, "wget")) {
            return 2;  // HIGH
        }

        // Moderate: file operations
        if_past(contains(cmd_lower, "rm ") ||
                contains(cmd_lower, "mv ") ||
                contains(cmd_lower, "cp ")) {
            return 1;  // MODERATE
        }

        return 0;  // SAFE
    }

    // =========================================================================
    // AGENCY - Main behavior loop
    // =========================================================================

    agency {
        repeat_past(INFINITY) {
            channels := get_bonded_channels();
            ProcessMessages(channels, processed);
            yield();
        }
    }

    grace {
        request_grace(5.0);
    }
}

// =============================================================================
// REASONER CREATURE - Chain-of-Thought Processing
// =============================================================================

creature ReasonerCreature {
    /**
     * Reasoner Creature provides chain-of-thought processing.
     * It breaks down complex problems into reasoning steps.
     *
     * Protocol:
     *   REASON: {"type": "reason", "problem": str, "max_steps": int}
     *   CHAIN: {"type": "chain", "steps": list, "conclusion": str}
     *
     * Each reasoning step costs F. Agency affects reasoning depth.
     */

    // Identity
    var cid: int := 0;
    var name: string := "reasoner";
    var parent_cid: int := 0;

    // DET state
    var F: Register := 40.0;
    var a: float := 0.7;            // Higher agency for reasoning
    var q: float := 0.0;
    var C_self: float := 1.0;

    // Reasoning state
    var current_problem: TokenReg;
    var reasoning_chain: TokenReg[];
    var is_reasoning: bool := false;
    var total_reasoned: int := 0;

    // Cost constants
    const STEP_COST: float := 0.3;
    const BASE_COST: float := 0.2;

    // =========================================================================
    // REASON KERNEL - Generate reasoning chain
    // =========================================================================

    kernel Reason {
        /**
         * Generate a chain of reasoning steps for a problem.
         * More agency allows deeper reasoning chains.
         */

        in  problem: TokenReg;          // Problem to reason about
        in  max_steps: int;             // Maximum reasoning steps
        in  requester: int;             // Requesting creature
        out chain: TokenReg;            // Reasoning chain result

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            // Agency determines max depth: a=0.5 -> 3 steps, a=1.0 -> 10 steps
            agency_depth := floor(a * 10);
            actual_max := min(max_steps, agency_depth);
            estimated_cost := BASE_COST + (actual_max * STEP_COST);

            proposal GENERATE_CHAIN {
                score = if_past(current_F >= estimated_cost) then a else 0.0;
                effect {
                    is_reasoning := true;
                    current_problem := problem;
                    reasoning_chain := [];

                    // Generate reasoning steps
                    step_num := 0;
                    should_continue := true;

                    repeat_past(actual_max) as i {
                        if_past(should_continue && F >= STEP_COST) {
                            // Generate next reasoning step
                            step := generate_reasoning_step(
                                problem,
                                reasoning_chain,
                                step_num
                            );

                            // Deduct cost
                            F := F - STEP_COST;

                            // Add to chain
                            reasoning_chain.push(step);
                            step_num := step_num + 1;

                            // Check if we've reached conclusion
                            if_past(is_conclusion(step)) {
                                should_continue := false;
                            }
                        }
                    }

                    // Build result
                    chain ::= witness(
                        reasoning_chain,
                        extract_conclusion(reasoning_chain),
                        step_num
                    );

                    is_reasoning := false;
                    total_reasoned := total_reasoned + 1;
                }
            }

            proposal DENY_INSUFFICIENT_F {
                score = if_past(current_F < estimated_cost) then 1.0 else 0.0;
                effect {
                    chain ::= witness([], "Insufficient F for reasoning", 0);
                }
            }
        }

        phase CHOOSE {
            choice := choose({GENERATE_CHAIN, DENY_INSUFFICIENT_F});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // PROCESS MESSAGES KERNEL
    // =========================================================================

    kernel ProcessMessages {
        in  channels: Register[];
        out processed: int;

        phase READ {
            pending_messages ::= get_all_pending(channels);
        }

        phase PROPOSE {
            proposal PROCESS_ALL {
                score = if_past(len(pending_messages) > 0 && !is_reasoning) then a else 0.0;
                effect {
                    processed := 0;

                    repeat_past(len(pending_messages)) as i {
                        msg := pending_messages[i];
                        sender := msg.sender;

                        if_past(msg.type == "reason") {
                            Reason(
                                msg.problem,
                                msg.max_steps || 5,
                                sender,
                                result
                            );

                            send_to(sender, {
                                type: "chain",
                                steps: result.steps,
                                conclusion: result.conclusion,
                                step_count: result.count
                            });
                        }

                        processed := processed + 1;
                    }
                }
            }

            proposal BUSY {
                score = if_past(is_reasoning) then 1.0 else 0.0;
                effect { processed := 0; }
            }

            proposal NO_MESSAGES {
                score = if_past(len(pending_messages) == 0) then 1.0 else 0.0;
                effect { processed := 0; }
            }
        }

        phase CHOOSE {
            choice := choose({PROCESS_ALL, BUSY, NO_MESSAGES});
        }

        phase COMMIT {
            commit choice;
        }
    }

    agency {
        repeat_past(INFINITY) {
            channels := get_bonded_channels();
            ProcessMessages(channels, processed);
            yield();
        }
    }

    grace {
        request_grace(8.0);
    }
}

// =============================================================================
// PLANNER CREATURE - Task Decomposition
// =============================================================================

creature PlannerCreature {
    /**
     * Planner Creature decomposes complex tasks into steps.
     *
     * Protocol:
     *   PLAN: {"type": "plan", "task": str, "constraints": dict}
     *   STEPS: {"type": "steps", "plan": list, "dependencies": dict}
     *
     * Planning costs F proportional to task complexity.
     */

    // Identity
    var cid: int := 0;
    var name: string := "planner";
    var parent_cid: int := 0;

    // DET state
    var F: Register := 35.0;
    var a: float := 0.65;
    var q: float := 0.0;
    var C_self: float := 1.0;

    // Planning state
    var current_task: TokenReg;
    var current_plan: TokenReg[];
    var is_planning: bool := false;
    var total_planned: int := 0;

    // Cost constants
    const PLAN_BASE_COST: float := 0.5;
    const STEP_COST: float := 0.15;

    // =========================================================================
    // PLAN KERNEL - Decompose a task
    // =========================================================================

    kernel Plan {
        in  task: TokenReg;             // Task description
        in  constraints: TokenReg;      // Planning constraints
        in  requester: int;
        out plan: TokenReg;             // Resulting plan

        phase READ {
            current_F ::= witness(F);
            task_complexity ::= estimate_complexity(task);
        }

        phase PROPOSE {
            estimated_steps := task_complexity * 3;  // Rough estimate
            estimated_cost := PLAN_BASE_COST + (estimated_steps * STEP_COST);

            proposal GENERATE_PLAN {
                score = if_past(current_F >= estimated_cost) then a else 0.0;
                effect {
                    is_planning := true;
                    current_task := task;

                    // Decompose task into steps
                    steps := decompose_task(task, constraints);

                    // Build dependency graph
                    dependencies := build_dependencies(steps);

                    // Calculate actual cost
                    actual_cost := PLAN_BASE_COST + (len(steps) * STEP_COST);
                    F := F - actual_cost;

                    plan ::= witness(steps, dependencies, len(steps));

                    is_planning := false;
                    total_planned := total_planned + 1;
                }
            }

            proposal DENY_INSUFFICIENT_F {
                score = if_past(current_F < estimated_cost) then 1.0 else 0.0;
                effect {
                    plan ::= witness([], {}, 0);
                }
            }
        }

        phase CHOOSE {
            choice := choose({GENERATE_PLAN, DENY_INSUFFICIENT_F});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // PROCESS MESSAGES KERNEL
    // =========================================================================

    kernel ProcessMessages {
        in  channels: Register[];
        out processed: int;

        phase READ {
            pending_messages ::= get_all_pending(channels);
        }

        phase PROPOSE {
            proposal PROCESS_ALL {
                score = if_past(len(pending_messages) > 0 && !is_planning) then a else 0.0;
                effect {
                    processed := 0;

                    repeat_past(len(pending_messages)) as i {
                        msg := pending_messages[i];
                        sender := msg.sender;

                        if_past(msg.type == "plan") {
                            Plan(
                                msg.task,
                                msg.constraints || {},
                                sender,
                                result
                            );

                            send_to(sender, {
                                type: "steps",
                                plan: result.steps,
                                dependencies: result.dependencies,
                                step_count: result.count
                            });
                        }

                        processed := processed + 1;
                    }
                }
            }

            proposal BUSY {
                score = if_past(is_planning) then 1.0 else 0.0;
                effect { processed := 0; }
            }

            proposal NO_MESSAGES {
                score = if_past(len(pending_messages) == 0) then 1.0 else 0.0;
                effect { processed := 0; }
            }
        }

        phase CHOOSE {
            choice := choose({PROCESS_ALL, BUSY, NO_MESSAGES});
        }

        phase COMMIT {
            commit choice;
        }
    }

    agency {
        repeat_past(INFINITY) {
            channels := get_bonded_channels();
            ProcessMessages(channels, processed);
            yield();
        }
    }

    grace {
        request_grace(7.0);
    }
}

// =============================================================================
// PRESENCE - Creature Ecosystem Bootstrap
// =============================================================================

presence CreatureEcosystem {
    /**
     * Bootstrap a standard creature ecosystem.
     * Creates memory, tool, reasoner, and planner creatures.
     */

    creatures {
        memory: MemoryCreature;
        tool: ToolCreature;
        reasoner: ReasonerCreature;
        planner: PlannerCreature;
    }

    bonds {
        // Creatures can bond with each other as needed
        // Bonds are created dynamically at runtime
    }

    init {
        // Initialize each creature with appropriate F
        inject_F(memory, 50.0);
        inject_F(tool, 30.0);
        inject_F(reasoner, 40.0);
        inject_F(planner, 35.0);

        // Set initial states
        memory.state := RUNNING;
        tool.state := RUNNING;
        reasoner.state := RUNNING;
        planner.state := RUNNING;
    }
}
