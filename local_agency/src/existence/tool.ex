/**
 * Tool Creature - Pure Existence-Lang Implementation
 * ===================================================
 *
 * Executes shell commands using substrate primitives.
 * Agency gates execution - low agency = safe commands only.
 *
 * Protocol (via bonds):
 *   REQUEST:  {"type": "exec", "command": str, "safe": bool}
 *   RESPONSE: {"type": "result", "output": str, "exit_code": int}
 */

creature ToolCreature {
    // DET state
    var F: Register := 50.0;
    var a: float := 0.5;
    var q: float := 0.0;

    // Configuration
    var allow_unsafe: bool := false;
    var timeout_ms: int := 30000;

    // Statistics
    var commands_executed: int := 0;
    var safe_commands: int := 0;
    var unsafe_commands: int := 0;
    var failed_commands: int := 0;

    // Cost constants
    var safe_exec_cost: float := 0.2;
    var unsafe_exec_cost: float := 0.5;

    // Agency thresholds
    var min_agency_safe: float := 0.3;
    var min_agency_unsafe: float := 0.5;

    // =========================================================================
    // EXEC_SAFE KERNEL - Execute safe (read-only) command
    // =========================================================================

    kernel ExecSafe {
        in  command: TokenReg;
        out output: TokenReg;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal EXECUTE_SAFE {
                // Require minimum agency for safe commands
                score = if_past(current_a >= min_agency_safe && current_F >= safe_exec_cost)
                        then current_a
                        else 0.0;

                effect {
                    // Call exec_safe primitive (whitelisted commands only)
                    result := primitive("exec_safe", command);

                    output ::= witness(result);
                    success := 1.0;

                    // Update statistics
                    commands_executed := commands_executed + 1;
                    safe_commands := safe_commands + 1;

                    // Deduct cost
                    F := F - safe_exec_cost;
                }
            }

            proposal REFUSE_LOW_AGENCY {
                score = if_past(current_a < min_agency_safe) then 1.0 else 0.0;

                effect {
                    output ::= witness("Agency too low for command execution");
                    success := 0.0;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < safe_exec_cost) then 1.0 else 0.0;

                effect {
                    output ::= witness("Insufficient F");
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({EXECUTE_SAFE, REFUSE_LOW_AGENCY, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // EXEC KERNEL - Execute any command (requires high agency)
    // =========================================================================

    kernel Exec {
        in  command: TokenReg;
        out output: TokenReg;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            unsafe_allowed ::= witness(allow_unsafe);
        }

        phase PROPOSE {
            proposal EXECUTE_UNSAFE {
                // Require high agency and explicit permission
                score = if_past(current_a >= min_agency_unsafe &&
                               current_F >= unsafe_exec_cost &&
                               unsafe_allowed == true)
                        then current_a * 0.8
                        else 0.0;

                effect {
                    // Call full exec primitive
                    result := primitive("exec", command);

                    output ::= witness(result);
                    success := 1.0;

                    // Update statistics
                    commands_executed := commands_executed + 1;
                    unsafe_commands := unsafe_commands + 1;

                    // Higher cost for unsafe commands
                    F := F - unsafe_exec_cost;
                }
            }

            proposal FALLBACK_SAFE {
                // Try safe exec if unsafe not allowed
                score = if_past(current_a >= min_agency_safe &&
                               current_F >= safe_exec_cost &&
                               unsafe_allowed == false)
                        then current_a * 0.5
                        else 0.0;

                effect {
                    result := primitive("exec_safe", command);
                    output ::= witness(result);
                    success := 1.0;

                    commands_executed := commands_executed + 1;
                    safe_commands := safe_commands + 1;
                    F := F - safe_exec_cost;
                }
            }

            proposal REFUSE_PERMISSION {
                score = if_past(current_a < min_agency_unsafe && unsafe_allowed == true)
                        then 1.0 else 0.0;

                effect {
                    output ::= witness("Agency too low for unsafe execution");
                    success := 0.0;
                    failed_commands := failed_commands + 1;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < safe_exec_cost) then 1.0 else 0.0;

                effect {
                    output ::= witness("Insufficient F");
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({EXECUTE_UNSAFE, FALLBACK_SAFE, REFUSE_PERMISSION, REFUSE_LOW_F},
                            decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // FILE_READ KERNEL - Read file contents
    // =========================================================================

    kernel FileRead {
        in  path: TokenReg;
        out content: TokenReg;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal READ_FILE {
                score = if_past(current_a >= 0.2 && current_F >= 0.1)
                        then current_a
                        else 0.0;

                effect {
                    result := primitive("file_read", path);
                    content ::= witness(result);
                    success := 1.0;
                    F := F - 0.1;
                }
            }

            proposal REFUSE {
                score = if_past(current_a < 0.2 || current_F < 0.1) then 1.0 else 0.0;

                effect {
                    content ::= witness("");
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({READ_FILE, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // FILE_WRITE KERNEL - Write file contents
    // =========================================================================

    kernel FileWrite {
        in  path: TokenReg;
        in  content: TokenReg;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal WRITE_FILE {
                // Require higher agency for writes
                score = if_past(current_a >= 0.5 && current_F >= 0.2)
                        then current_a * 0.8
                        else 0.0;

                effect {
                    result := primitive("file_write", path, content);
                    success := 1.0;
                    F := F - 0.2;
                }
            }

            proposal REFUSE {
                score = if_past(current_a < 0.5 || current_F < 0.2) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({WRITE_FILE, REFUSE}, decisiveness = current_a);
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
            exec_count ::= witness(commands_executed);
            safe_count ::= witness(safe_commands);
            unsafe_count ::= witness(unsafe_commands);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    status ::= witness(current_F, current_a, exec_count, safe_count, unsafe_count);
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
