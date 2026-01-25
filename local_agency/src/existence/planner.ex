/**
 * Planner Creature - Pure Existence-Lang Implementation
 * ======================================================
 *
 * Task decomposition and planning using optional LLM assistance.
 * Breaks complex tasks into steps with dependencies.
 *
 * Protocol (via bonds):
 *   REQUEST:  {"type": "plan", "task": str, "constraints": dict}
 *   RESPONSE: {"type": "plan", "steps": list, "dependencies": dict}
 */

creature PlannerCreature {
    // DET state
    var F: Register := 75.0;
    var a: float := 0.6;
    var q: float := 0.0;

    // Configuration
    var max_steps: int := 10;
    var use_llm: bool := true;

    // Planning state
    var current_plan_steps: int := 0;
    var active_plans: int := 0;

    // Cost constants
    var plan_base_cost: float := 0.5;
    var step_cost: float := 0.05;
    var llm_plan_cost: float := 2.0;

    // Statistics
    var total_plans: int := 0;
    var total_steps_generated: int := 0;
    var llm_plans: int := 0;

    // =========================================================================
    // PLAN KERNEL - Create a plan for a task
    // =========================================================================

    kernel Plan {
        in  task: TokenReg;
        in  max_plan_steps: Register;
        out plan: TokenReg;
        out step_count: Register;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            llm_enabled ::= witness(use_llm);
        }

        phase PROPOSE {
            // Cost calculation
            basic_cost := plan_base_cost + (max_plan_steps * step_cost);

            proposal PLAN_WITH_LLM {
                // Use LLM for complex planning
                score = if_past(llm_enabled == true &&
                               current_a >= 0.5 &&
                               current_F >= llm_plan_cost)
                        then current_a * 0.9
                        else 0.0;

                effect {
                    // Generate plan using LLM
                    result := primitive("llm_call", task);

                    plan ::= witness(result);
                    step_count := max_plan_steps;
                    success := 1.0;

                    // Update statistics
                    total_plans := total_plans + 1;
                    total_steps_generated := total_steps_generated + max_plan_steps;
                    llm_plans := llm_plans + 1;
                    active_plans := active_plans + 1;

                    F := F - llm_plan_cost;
                }
            }

            proposal PLAN_SIMPLE {
                // Generate simple sequential plan
                score = if_past(current_F >= basic_cost && current_a >= 0.3)
                        then current_a * 0.5
                        else 0.0;

                effect {
                    // Simple plan: sequential steps
                    plan ::= witness("Sequential plan generated");
                    step_count := max_plan_steps;
                    success := 1.0;

                    total_plans := total_plans + 1;
                    total_steps_generated := total_steps_generated + max_plan_steps;
                    active_plans := active_plans + 1;

                    F := F - basic_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < basic_cost) then 1.0 else 0.0;

                effect {
                    plan ::= witness("Insufficient F for planning");
                    step_count := 0.0;
                    success := 0.0;
                }
            }

            proposal REFUSE_LOW_AGENCY {
                score = if_past(current_a < 0.3) then 1.0 else 0.0;

                effect {
                    plan ::= witness("Agency too low for planning");
                    step_count := 0.0;
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({PLAN_WITH_LLM, PLAN_SIMPLE, REFUSE_LOW_F, REFUSE_LOW_AGENCY},
                            decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // DECOMPOSE KERNEL - Break a task into subtasks
    // =========================================================================

    kernel Decompose {
        in  task: TokenReg;
        in  target_subtasks: Register;
        out subtasks: TokenReg;
        out subtask_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            llm_enabled ::= witness(use_llm);
        }

        phase PROPOSE {
            cost := plan_base_cost + (target_subtasks * step_cost);

            proposal DECOMPOSE_WITH_LLM {
                score = if_past(llm_enabled == true &&
                               current_a >= 0.4 &&
                               current_F >= llm_plan_cost)
                        then current_a * 0.85
                        else 0.0;

                effect {
                    result := primitive("llm_call", task);
                    subtasks ::= witness(result);
                    subtask_count := target_subtasks;

                    F := F - llm_plan_cost;
                }
            }

            proposal DECOMPOSE_SIMPLE {
                score = if_past(current_F >= cost)
                        then current_a * 0.4
                        else 0.0;

                effect {
                    subtasks ::= witness("Task decomposed into steps");
                    subtask_count := target_subtasks;

                    F := F - cost;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < cost) then 1.0 else 0.0;

                effect {
                    subtasks ::= witness("Cannot decompose - insufficient F");
                    subtask_count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DECOMPOSE_WITH_LLM, DECOMPOSE_SIMPLE, REFUSE},
                            decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // ESTIMATE KERNEL - Estimate resources for a task
    // =========================================================================

    kernel Estimate {
        in  task: TokenReg;
        out f_required: Register;
        out complexity: Register;
        out risk: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal DO_ESTIMATE {
                score = if_past(current_F >= step_cost)
                        then current_a
                        else 0.0;

                effect {
                    // Heuristic estimation
                    // In production, this would analyze the task
                    f_required := 5.0;
                    complexity := 0.5;
                    risk := 0.3;

                    F := F - step_cost;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < step_cost) then 1.0 else 0.0;

                effect {
                    f_required := 0.0;
                    complexity := 0.0;
                    risk := 1.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_ESTIMATE, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // COMPLETE_PLAN KERNEL - Mark a plan as complete
    // =========================================================================

    kernel CompletePlan {
        in  plan_id: Register;
        out success: Register;

        phase READ {
            current_active ::= witness(active_plans);
        }

        phase PROPOSE {
            proposal COMPLETE {
                score = if_past(current_active > 0) then 1.0 else 0.0;

                effect {
                    active_plans := active_plans - 1;
                    success := 1.0;
                }
            }

            proposal NO_ACTIVE {
                score = if_past(current_active == 0) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({COMPLETE, NO_ACTIVE});
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
            plans ::= witness(total_plans);
            steps ::= witness(total_steps_generated);
            llm ::= witness(llm_plans);
            active ::= witness(active_plans);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    status ::= witness(f, ag, plans, steps, llm, active);
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
