/**
 * Terminal Creature - Pure Existence-Lang Implementation
 * =======================================================
 *
 * The CLI itself as an Existence-Lang creature.
 * Handles user input, dispatches to bonded creatures, and displays output.
 *
 * This creature serves as the primary interface between human users
 * and the DET-native operating system.
 *
 * Protocol (via bonds):
 *   TO LLM:      {"type": "think", "prompt": str}
 *   TO TOOL:     {"type": "exec", "command": str}
 *   FROM:        {"type": "response", "result": str}
 */

creature TerminalCreature {
    // DET state
    var F: Register := 100.0;
    var a: float := 0.8;
    var q: float := 0.0;

    // Configuration
    var prompt_text: TokenReg := "det> ";
    var welcome_msg: TokenReg := "DET Local Agency - Existence-Lang Terminal";
    var running: bool := true;

    // Connected bonds
    var llm_bond: Register := 0.0;
    var tool_bond: Register := 0.0;
    var memory_bond: Register := 0.0;

    // Statistics
    var commands_processed: int := 0;
    var total_input_chars: int := 0;

    // Cost constants
    var read_cost: float := 0.01;
    var write_cost: float := 0.005;
    var dispatch_cost: float := 0.1;

    // =========================================================================
    // INIT KERNEL - Initialize terminal
    // =========================================================================

    kernel Init {
        out success: Register;

        phase READ {
            current_F ::= witness(F);
        }

        phase PROPOSE {
            proposal DO_INIT {
                score = 1.0;

                effect {
                    // Display welcome message
                    primitive("terminal_color", "cyan");
                    primitive("terminal_write", welcome_msg);
                    primitive("terminal_color", "reset");
                    primitive("terminal_write", "\n");
                    primitive("terminal_write", "Type 'help' for commands, 'quit' to exit.\n\n");

                    success := 1.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_INIT});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // READ_INPUT KERNEL - Read user input from terminal
    // =========================================================================

    kernel ReadInput {
        out input: TokenReg;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_running ::= witness(running);
        }

        phase PROPOSE {
            proposal DO_READ {
                score = if_past(current_F >= read_cost && current_running == true)
                        then 1.0
                        else 0.0;

                effect {
                    // Display prompt and read input
                    result := primitive("terminal_prompt", prompt_text);
                    input ::= witness(result);

                    // Update stats
                    commands_processed := commands_processed + 1;

                    F := F - read_cost;
                    success := 1.0;
                }
            }

            proposal REFUSE_NOT_RUNNING {
                score = if_past(current_running == false) then 1.0 else 0.0;

                effect {
                    input ::= witness("");
                    success := 0.0;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < read_cost) then 1.0 else 0.0;

                effect {
                    input ::= witness("");
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_READ, REFUSE_NOT_RUNNING, REFUSE_LOW_F});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // WRITE_OUTPUT KERNEL - Write output to terminal
    // =========================================================================

    kernel WriteOutput {
        in  message: TokenReg;
        in  color: TokenReg;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
        }

        phase PROPOSE {
            proposal DO_WRITE {
                score = if_past(current_F >= write_cost)
                        then a
                        else 0.0;

                effect {
                    // Set color if provided
                    primitive("terminal_color", color);

                    // Write message
                    primitive("terminal_write", message);
                    primitive("terminal_write", "\n");

                    // Reset color
                    primitive("terminal_color", "reset");

                    F := F - write_cost;
                    success := 1.0;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < write_cost) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_WRITE, REFUSE}, decisiveness = a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // DISPATCH KERNEL - Dispatch command to appropriate creature
    // =========================================================================

    kernel Dispatch {
        in  command: TokenReg;
        out response: TokenReg;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            llm ::= witness(llm_bond);
            tool ::= witness(tool_bond);
        }

        phase PROPOSE {
            proposal DISPATCH_TO_LLM {
                // Commands starting with 'ask', 'think', or general queries
                score = if_past(current_F >= dispatch_cost && current_a >= 0.3)
                        then current_a * 0.8
                        else 0.0;

                effect {
                    // For now, use LLM primitive directly
                    // In full implementation, this would send via bond
                    result := primitive("llm_call", command);
                    response ::= witness(result);

                    F := F - dispatch_cost;
                    success := 1.0;
                }
            }

            proposal DISPATCH_TO_TOOL {
                // Commands starting with 'exec', 'run', shell commands
                score = if_past(current_F >= dispatch_cost && current_a >= 0.5)
                        then current_a * 0.7
                        else 0.0;

                effect {
                    // Execute via tool primitive
                    result := primitive("exec_safe", command);
                    response ::= witness(result);

                    F := F - dispatch_cost;
                    success := 1.0;
                }
            }

            proposal HANDLE_BUILTIN {
                // Built-in commands: help, status, quit
                score = if_past(current_F >= read_cost)
                        then 0.9
                        else 0.0;

                effect {
                    // Handle built-in command
                    response ::= witness("Built-in command processed");
                    F := F - read_cost;
                    success := 1.0;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < read_cost) then 1.0 else 0.0;

                effect {
                    response ::= witness("Insufficient F");
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DISPATCH_TO_LLM, DISPATCH_TO_TOOL, HANDLE_BUILTIN, REFUSE},
                            decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // HELP KERNEL - Display help information
    // =========================================================================

    kernel Help {
        out help_text: TokenReg;

        phase READ {
            // No state needed
        }

        phase PROPOSE {
            proposal SHOW_HELP {
                score = 1.0;

                effect {
                    primitive("terminal_color", "yellow");
                    primitive("terminal_write", "\nDET Terminal Commands:\n");
                    primitive("terminal_color", "reset");
                    primitive("terminal_write", "  help     - Show this help\n");
                    primitive("terminal_write", "  status   - Show creature status\n");
                    primitive("terminal_write", "  ask <q>  - Ask the LLM a question\n");
                    primitive("terminal_write", "  run <c>  - Run a shell command (safe)\n");
                    primitive("terminal_write", "  quit     - Exit the terminal\n");
                    primitive("terminal_write", "\n");

                    help_text ::= witness("Help displayed");
                }
            }
        }

        phase CHOOSE {
            choice := choose({SHOW_HELP});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STATUS KERNEL - Display creature status
    // =========================================================================

    kernel Status {
        out status: TokenReg;

        phase READ {
            f ::= witness(F);
            ag ::= witness(a);
            cmds ::= witness(commands_processed);
            run ::= witness(running);
        }

        phase PROPOSE {
            proposal SHOW_STATUS {
                score = 1.0;

                effect {
                    primitive("terminal_color", "green");
                    primitive("terminal_write", "\nTerminal Creature Status:\n");
                    primitive("terminal_color", "reset");
                    primitive("terminal_write", "  F (resource):    ");
                    primitive("print", f);
                    primitive("terminal_write", "  a (agency):      ");
                    primitive("print", ag);
                    primitive("terminal_write", "  Commands run:    ");
                    primitive("print", cmds);
                    primitive("terminal_write", "  Running:         ");
                    primitive("print", run);
                    primitive("terminal_write", "\n");

                    status ::= witness(f, ag, cmds, run);
                }
            }
        }

        phase CHOOSE {
            choice := choose({SHOW_STATUS});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // QUIT KERNEL - Stop the terminal
    // =========================================================================

    kernel Quit {
        out success: Register;

        phase READ {
            current_running ::= witness(running);
        }

        phase PROPOSE {
            proposal DO_QUIT {
                score = if_past(current_running == true) then 1.0 else 0.0;

                effect {
                    primitive("terminal_color", "cyan");
                    primitive("terminal_write", "\nGoodbye from DET Terminal.\n");
                    primitive("terminal_color", "reset");

                    running := false;
                    success := 1.0;
                }
            }

            proposal ALREADY_STOPPED {
                score = if_past(current_running == false) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_QUIT, ALREADY_STOPPED});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // BOND_LLM KERNEL - Create bond to LLM creature
    // =========================================================================

    kernel BondLLM {
        in  target_id: Register;
        out bond_id: Register;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_llm ::= witness(llm_bond);
        }

        phase PROPOSE {
            proposal CREATE_BOND {
                score = if_past(current_F >= 1.0 && current_llm == 0.0)
                        then a
                        else 0.0;

                effect {
                    // In full implementation, this would create actual bond
                    llm_bond := target_id;
                    bond_id := target_id;
                    F := F - 1.0;
                    success := 1.0;
                }
            }

            proposal ALREADY_BONDED {
                score = if_past(current_llm > 0.0) then 0.8 else 0.0;

                effect {
                    bond_id := current_llm;
                    success := 1.0;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < 1.0) then 1.0 else 0.0;

                effect {
                    bond_id := 0.0;
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({CREATE_BOND, ALREADY_BONDED, REFUSE}, decisiveness = a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // BOND_TOOL KERNEL - Create bond to Tool creature
    // =========================================================================

    kernel BondTool {
        in  target_id: Register;
        out bond_id: Register;
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            current_tool ::= witness(tool_bond);
        }

        phase PROPOSE {
            proposal CREATE_BOND {
                score = if_past(current_F >= 1.0 && current_tool == 0.0)
                        then a
                        else 0.0;

                effect {
                    tool_bond := target_id;
                    bond_id := target_id;
                    F := F - 1.0;
                    success := 1.0;
                }
            }

            proposal ALREADY_BONDED {
                score = if_past(current_tool > 0.0) then 0.8 else 0.0;

                effect {
                    bond_id := current_tool;
                    success := 1.0;
                }
            }

            proposal REFUSE {
                score = if_past(current_F < 1.0) then 1.0 else 0.0;

                effect {
                    bond_id := 0.0;
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({CREATE_BOND, ALREADY_BONDED, REFUSE}, decisiveness = a);
        }

        phase COMMIT {
            commit choice;
        }
    }
}
