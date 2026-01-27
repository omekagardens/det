/**
 * LLM Creature - Pure Existence-Lang Implementation
 * ==================================================
 *
 * Handles LLM interaction using substrate primitives.
 * All external I/O goes through the primitive() call.
 *
 * Protocol (via bonds):
 *   REQUEST:  {"type": "think", "prompt": str, "model": str}
 *   RESPONSE: {"type": "response", "text": str, "tokens": int}
 *
 * Phase 21 Enhancements:
 *   - Multi-model support (model selection per call)
 *   - Token budget management (per-creature limits)
 *   - Temperature modulation by agency/arousal
 *   - Streaming callback support
 *
 * Phase 26 Enhancements:
 *   - Native inference support (GGUF models, no Ollama dependency)
 *   - DET-aware sampling via det_choose_token
 *   - GPU acceleration via Metal (macOS)
 *   - Switchable backend (use_native flag)
 */

creature LLMCreature {
    // DET state
    var F: Register := 100.0;
    var a: float := 0.7;
    var q: float := 0.0;
    var P: float := 1.0;             // Phase 26: Presence for DET sampling
    var arousal: float := 0.5;       // Phase 21: For temperature modulation
    var bondedness: float := 0.5;    // Phase 21: For temperature modulation

    // Model configuration (Phase 21: Multi-model support)
    var model: string := "llama3.2:3b";      // Default model (Ollama name)
    var model_reasoning: string := "deepseek-r1:1.5b";  // Reasoning specialist
    var model_coding: string := "qwen2.5-coder:1.5b";   // Code specialist
    var model_fast: string := "phi4-mini";              // Fast/efficient model

    // Phase 26: Native inference configuration
    var use_native: bool := false;           // Toggle: true=native, false=Ollama
    var native_model_path: string := "";     // Path to GGUF model file
    var native_model_loaded: bool := false;  // Is native model loaded?
    var native_model_info: string := "";     // Model info string

    // Temperature configuration
    var base_temperature: float := 0.7;
    var temperature_agency_weight: float := 0.2;    // How much agency affects temp
    var temperature_arousal_weight: float := 0.15;  // How much arousal affects temp

    // Generation limits
    var max_tokens: int := 512;

    // Token budget management (Phase 21)
    var token_budget: int := 10000;         // Total tokens allowed
    var tokens_remaining: int := 10000;     // Tokens left in budget
    var budget_period: float := 3600.0;     // Budget reset period (seconds)
    var budget_start_time: float := 0.0;    // When current budget period started

    // Statistics
    var calls_made: int := 0;
    var tokens_generated: int := 0;
    var total_cost: float := 0.0;

    // Cost constants (per token, varies by model)
    var cost_per_token: float := 0.01;
    var base_call_cost: float := 1.0;
    var native_load_cost: float := 0.5;     // Phase 26: Cost to load native model

    // =========================================================================
    // THINK KERNEL - Generate response from prompt (Phase 21 + Phase 26)
    // Supports both Ollama (HTTP) and native inference (GGUF)
    // =========================================================================

    kernel Think {
        in  prompt: TokenReg;
        in  intent: TokenReg;  // Optional: code, reason, quick, general, etc.
        out response: TokenReg;
        out token_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_P ::= witness(P);
            current_arousal ::= witness(arousal);
            current_bondedness ::= witness(bondedness);
            current_budget ::= witness(tokens_remaining);
            current_model ::= witness(model);
            current_model_coding ::= witness(model_coding);
            current_model_reasoning ::= witness(model_reasoning);
            current_model_fast ::= witness(model_fast);
            current_base_temp ::= witness(base_temperature);
            is_native ::= witness(use_native);
            native_loaded ::= witness(native_model_loaded);
        }

        phase PROPOSE {
            // Compute effective temperature from DET state (inline)
            // Higher agency -> more exploration, higher arousal -> more variability
            agency_mod := (current_a - 0.5) * temperature_agency_weight;
            arousal_mod := (current_arousal - 0.5) * temperature_arousal_weight;
            bond_mod := (current_bondedness - 0.5) * 0.1;
            effective_temp := current_base_temp + agency_mod + arousal_mod - bond_mod;

            // Select model based on intent (inline) - for Ollama
            selected_model := current_model;
            if_past(intent == "code") { selected_model := current_model_coding; }
            if_past(intent == "debug") { selected_model := current_model_coding; }
            if_past(intent == "reason") { selected_model := current_model_reasoning; }
            if_past(intent == "plan") { selected_model := current_model_reasoning; }
            if_past(intent == "math") { selected_model := current_model_reasoning; }
            if_past(intent == "quick") { selected_model := current_model_fast; }
            if_past(intent == "simple") { selected_model := current_model_fast; }

            estimated_tokens := max_tokens;

            // Phase 26: Native inference path (DET-aware)
            proposal CALL_NATIVE {
                native_ok := if_past(is_native == true) then 1.0 else 0.0;
                loaded_ok := if_past(native_loaded == true) then 1.0 else 0.0;
                budget_ok := if_past(current_budget >= estimated_tokens) then 1.0 else 0.0;
                f_ok := if_past(current_F >= base_call_cost) then 1.0 else 0.0;
                presence_ok := if_past(current_P >= 0.1) then 1.0 else 0.0;
                score = current_a * 0.95 * native_ok * loaded_ok * budget_ok * f_ok * presence_ok;

                effect {
                    // Use DET-native inference with presence-aware sampling
                    result := primitive("model_generate", prompt, max_tokens);
                    response ::= witness(result);

                    // Estimate tokens from response length
                    actual_tokens := len(result) / 4;
                    token_count := actual_tokens;

                    // Update budget
                    tokens_remaining := tokens_remaining - actual_tokens;
                    tokens_generated := tokens_generated + actual_tokens;
                    calls_made := calls_made + 1;
                    F := F - base_call_cost;
                    total_cost := total_cost + base_call_cost + (actual_tokens * cost_per_token);
                }
            }

            // Ollama path (original)
            proposal CALL_OLLAMA {
                ollama_ok := if_past(is_native != true) then 1.0 else 0.0;
                budget_ok := if_past(current_budget >= estimated_tokens) then 1.0 else 0.0;
                f_ok := if_past(current_F >= base_call_cost) then 1.0 else 0.0;
                score = current_a * 0.9 * ollama_ok * budget_ok * f_ok;

                effect {
                    result := primitive("llm_call_v2", selected_model, prompt, effective_temp, max_tokens);
                    response ::= witness(result.text);
                    actual_tokens := result.tokens;
                    token_count := actual_tokens;

                    // Update budget
                    tokens_remaining := tokens_remaining - actual_tokens;
                    tokens_generated := tokens_generated + actual_tokens;
                    calls_made := calls_made + 1;
                    F := F - base_call_cost;
                    total_cost := total_cost + base_call_cost + (actual_tokens * cost_per_token);
                }
            }

            proposal REFUSE_NATIVE_NOT_LOADED {
                native_on := if_past(is_native == true) then 1.0 else 0.0;
                not_loaded := if_past(native_loaded != true) then 1.0 else 0.0;
                score = native_on * not_loaded * 0.98;

                effect {
                    response ::= witness("Native model not loaded - call LoadNativeModel first");
                    token_count := 0.0;
                }
            }

            proposal REFUSE_LOW_PRESENCE {
                low_presence := if_past(current_P < 0.1) then 1.0 else 0.0;
                native_on := if_past(is_native == true) then 1.0 else 0.0;
                score = low_presence * native_on * 0.97;

                effect {
                    response ::= witness("Presence too low - entity fading");
                    token_count := 0.0;
                }
            }

            proposal REFUSE_LOW_BUDGET {
                score = if_past(current_budget < estimated_tokens) then 0.95 else 0.0;

                effect {
                    response ::= witness("Token budget exhausted - wait for reset");
                    token_count := 0.0;
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
            choice := choose({CALL_NATIVE, CALL_OLLAMA, REFUSE_NATIVE_NOT_LOADED, REFUSE_LOW_PRESENCE, REFUSE_LOW_BUDGET, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // THINK_STREAM KERNEL - Streaming response generation (Phase 21)
    // =========================================================================

    kernel ThinkStream {
        in  prompt: TokenReg;
        in  intent: TokenReg;
        in  on_chunk: TokenReg;          // Callback name for each chunk
        out response: TokenReg;
        out token_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_arousal ::= witness(arousal);
            current_bondedness ::= witness(bondedness);
            current_budget ::= witness(tokens_remaining);
            current_model ::= witness(model);
            current_model_coding ::= witness(model_coding);
            current_model_reasoning ::= witness(model_reasoning);
            current_model_fast ::= witness(model_fast);
            current_base_temp ::= witness(base_temperature);
        }

        phase PROPOSE {
            // Compute effective temperature (inline)
            agency_mod := (current_a - 0.5) * temperature_agency_weight;
            arousal_mod := (current_arousal - 0.5) * temperature_arousal_weight;
            bond_mod := (current_bondedness - 0.5) * 0.1;
            effective_temp := current_base_temp + agency_mod + arousal_mod - bond_mod;

            // Select model (inline)
            selected_model := current_model;
            if_past(intent == "code") { selected_model := current_model_coding; }
            if_past(intent == "reason") { selected_model := current_model_reasoning; }
            if_past(intent == "quick") { selected_model := current_model_fast; }

            proposal STREAM_LLM {
                budget_ok := if_past(current_budget >= max_tokens) then 1.0 else 0.0;
                f_ok := if_past(current_F >= base_call_cost) then 1.0 else 0.0;
                score = current_a * 0.85 * budget_ok * f_ok;

                effect {
                    result := primitive("llm_stream_v2", selected_model, prompt, effective_temp, max_tokens, on_chunk);
                    response ::= witness(result.text);
                    actual_tokens := result.tokens;
                    token_count := actual_tokens;

                    tokens_remaining := tokens_remaining - actual_tokens;
                    tokens_generated := tokens_generated + actual_tokens;
                    calls_made := calls_made + 1;
                    F := F - base_call_cost;
                    total_cost := total_cost + base_call_cost + (actual_tokens * cost_per_token);
                }
            }

            proposal REFUSE {
                budget_fail := if_past(current_budget < max_tokens) then 1.0 else 0.0;
                f_fail := if_past(current_F < base_call_cost) then 1.0 else 0.0;
                score = budget_fail * 0.95;

                effect {
                    response ::= witness("Cannot stream: insufficient resources");
                    token_count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({STREAM_LLM, REFUSE}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CHAT KERNEL - Multi-turn conversation (Phase 21 enhanced)
    // =========================================================================

    kernel Chat {
        in  messages: TokenReg;
        in  intent: TokenReg;
        out response: TokenReg;
        out token_count: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_arousal ::= witness(arousal);
            current_bondedness ::= witness(bondedness);
            current_budget ::= witness(tokens_remaining);
            current_model ::= witness(model);
            current_model_coding ::= witness(model_coding);
            current_model_reasoning ::= witness(model_reasoning);
            current_model_fast ::= witness(model_fast);
            current_base_temp ::= witness(base_temperature);
        }

        phase PROPOSE {
            // Compute effective temperature (inline)
            agency_mod := (current_a - 0.5) * temperature_agency_weight;
            arousal_mod := (current_arousal - 0.5) * temperature_arousal_weight;
            bond_mod := (current_bondedness - 0.5) * 0.1;
            effective_temp := current_base_temp + agency_mod + arousal_mod - bond_mod;

            // Select model (inline)
            selected_model := current_model;
            if_past(intent == "code") { selected_model := current_model_coding; }
            if_past(intent == "reason") { selected_model := current_model_reasoning; }
            if_past(intent == "quick") { selected_model := current_model_fast; }

            chat_cost := 1.5;

            proposal CALL_CHAT {
                budget_ok := if_past(current_budget >= max_tokens) then 1.0 else 0.0;
                f_ok := if_past(current_F >= chat_cost) then 1.0 else 0.0;
                score = current_a * 0.85 * budget_ok * f_ok;

                effect {
                    result := primitive("llm_chat_v2", selected_model, messages, effective_temp, max_tokens);
                    response ::= witness(result.text);
                    actual_tokens := result.tokens;
                    token_count := actual_tokens;

                    tokens_remaining := tokens_remaining - actual_tokens;
                    tokens_generated := tokens_generated + actual_tokens;
                    calls_made := calls_made + 1;
                    F := F - chat_cost;
                    total_cost := total_cost + chat_cost + (actual_tokens * cost_per_token);
                }
            }

            proposal REFUSE_BUDGET {
                score = if_past(current_budget < max_tokens) then 0.95 else 0.0;

                effect {
                    response ::= witness("Token budget exhausted");
                    token_count := 0.0;
                }
            }

            proposal REFUSE_F {
                score = if_past(current_F < chat_cost) then 1.0 else 0.0;

                effect {
                    response ::= witness("Insufficient F");
                    token_count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({CALL_CHAT, REFUSE_BUDGET, REFUSE_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // SET_MODEL KERNEL - Change the active model (Phase 21)
    // =========================================================================

    kernel SetModel {
        in  model_name: TokenReg;
        in  model_type: TokenReg;  // default, reasoning, coding, fast
        out success: Register;

        phase READ {
            current_model ::= witness(model);
        }

        phase PROPOSE {
            proposal SET_DEFAULT {
                score = if_past(model_type == "default") then 1.0 else 0.0;

                effect {
                    model := model_name;
                    success := 1.0;
                }
            }

            proposal SET_REASONING {
                score = if_past(model_type == "reasoning") then 1.0 else 0.0;

                effect {
                    model_reasoning := model_name;
                    success := 1.0;
                }
            }

            proposal SET_CODING {
                score = if_past(model_type == "coding") then 1.0 else 0.0;

                effect {
                    model_coding := model_name;
                    success := 1.0;
                }
            }

            proposal SET_FAST {
                score = if_past(model_type == "fast") then 1.0 else 0.0;

                effect {
                    model_fast := model_name;
                    success := 1.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({SET_DEFAULT, SET_REASONING, SET_CODING, SET_FAST});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // RESET_BUDGET KERNEL - Reset token budget (Phase 21)
    // =========================================================================

    kernel ResetBudget {
        in  new_budget: Register;  // 0 means use default token_budget
        out old_remaining: Register;
        out new_remaining: Register;

        phase READ {
            current_remaining ::= witness(tokens_remaining);
            default_budget ::= witness(token_budget);
        }

        phase PROPOSE {
            proposal RESET_DEFAULT {
                score = if_past(new_budget <= 0.0) then 1.0 else 0.0;

                effect {
                    old_remaining := current_remaining;
                    tokens_remaining := default_budget;
                    new_remaining := default_budget;
                    budget_start_time := primitive("get_time");
                }
            }

            proposal RESET_CUSTOM {
                score = if_past(new_budget > 0.0) then 1.0 else 0.0;

                effect {
                    old_remaining := current_remaining;
                    tokens_remaining := new_budget;
                    new_remaining := new_budget;
                    budget_start_time := primitive("get_time");
                }
            }
        }

        phase CHOOSE {
            choice := choose({RESET_DEFAULT, RESET_CUSTOM});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // SET_AFFECT KERNEL - Update affect state for temperature modulation (Phase 21)
    // =========================================================================

    kernel SetAffect {
        in  new_arousal: Register;    // -1 means don't change
        in  new_bondedness: Register;
        in  new_agency: Register;
        out success: Register;

        phase READ {
            current_arousal ::= witness(arousal);
            current_bondedness ::= witness(bondedness);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal UPDATE_AFFECT {
                score = 1.0;

                effect {
                    // Only update if value is valid (>= 0), clamp to [0,1]
                    if_past(new_arousal >= 0.0) {
                        clamped_ar := new_arousal;
                        if_past(clamped_ar > 1.0) { clamped_ar := 1.0; }
                        if_past(clamped_ar < 0.0) { clamped_ar := 0.0; }
                        arousal := clamped_ar;
                    }
                    if_past(new_bondedness >= 0.0) {
                        clamped_bd := new_bondedness;
                        if_past(clamped_bd > 1.0) { clamped_bd := 1.0; }
                        if_past(clamped_bd < 0.0) { clamped_bd := 0.0; }
                        bondedness := clamped_bd;
                    }
                    if_past(new_agency >= 0.0) {
                        clamped_ag := new_agency;
                        if_past(clamped_ag > 1.0) { clamped_ag := 1.0; }
                        if_past(clamped_ag < 0.0) { clamped_ag := 0.0; }
                        a := clamped_ag;
                    }
                    success := 1.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({UPDATE_AFFECT});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STATUS KERNEL - Report creature status (Phase 21 enhanced)
    // =========================================================================

    kernel Status {
        out status_F: Register;
        out status_a: Register;
        out status_arousal: Register;
        out status_bondedness: Register;
        out status_calls: Register;
        out status_cost: Register;
        out status_tokens: Register;
        out status_budget: Register;
        out status_temp: Register;
        out status_model: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_arousal ::= witness(arousal);
            current_bondedness ::= witness(bondedness);
            current_calls ::= witness(calls_made);
            current_cost ::= witness(total_cost);
            current_tokens ::= witness(tokens_generated);
            current_budget ::= witness(tokens_remaining);
            current_model ::= witness(model);
            current_base_temp ::= witness(base_temperature);
        }

        phase PROPOSE {
            // Compute effective temperature inline
            agency_mod := (current_a - 0.5) * temperature_agency_weight;
            arousal_mod := (current_arousal - 0.5) * temperature_arousal_weight;
            bond_mod := (current_bondedness - 0.5) * 0.1;
            effective_temp := current_base_temp + agency_mod + arousal_mod - bond_mod;

            proposal REPORT {
                score = 1.0;

                effect {
                    status_F := current_F;
                    status_a := current_a;
                    status_arousal := current_arousal;
                    status_bondedness := current_bondedness;
                    status_calls := current_calls;
                    status_cost := current_cost;
                    status_tokens := current_tokens;
                    status_budget := current_budget;
                    status_temp := effective_temp;
                    status_model ::= witness(current_model);
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

    // =========================================================================
    // LIST_MODELS KERNEL - List configured models (Phase 21)
    // =========================================================================

    kernel ListModels {
        out model_default: TokenReg;
        out model_reason: TokenReg;
        out model_code: TokenReg;
        out model_quick: TokenReg;

        phase READ {
            m_default ::= witness(model);
            m_reasoning ::= witness(model_reasoning);
            m_coding ::= witness(model_coding);
            m_fast ::= witness(model_fast);
        }

        phase PROPOSE {
            proposal LIST {
                score = 1.0;

                effect {
                    model_default ::= witness(m_default);
                    model_reason ::= witness(m_reasoning);
                    model_code ::= witness(m_coding);
                    model_quick ::= witness(m_fast);
                }
            }
        }

        phase CHOOSE {
            choice := choose({LIST});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CONFIGURE KERNEL - Bulk configuration update (Phase 21)
    // =========================================================================

    kernel Configure {
        in  new_temperature: Register;
        in  new_max_tokens: Register;
        in  new_token_budget: Register;
        in  new_model: TokenReg;
        in  new_model_reasoning: TokenReg;
        in  new_model_coding: TokenReg;
        in  new_model_fast: TokenReg;
        out success: Register;

        phase READ {
            current_temp ::= witness(base_temperature);
            current_max ::= witness(max_tokens);
            current_budget ::= witness(token_budget);
        }

        phase PROPOSE {
            proposal APPLY_CONFIG {
                score = 1.0;

                effect {
                    // Apply temperature if provided (>= 0)
                    if_past(new_temperature >= 0.0) {
                        clamped_temp := new_temperature;
                        if_past(clamped_temp > 2.0) { clamped_temp := 2.0; }
                        if_past(clamped_temp < 0.1) { clamped_temp := 0.1; }
                        base_temperature := clamped_temp;
                    }

                    // Apply max_tokens if provided (> 0)
                    if_past(new_max_tokens > 0.0) {
                        max_tokens := new_max_tokens;
                    }

                    // Apply token_budget if provided (> 0)
                    if_past(new_token_budget > 0.0) {
                        token_budget := new_token_budget;
                        tokens_remaining := new_token_budget;
                    }

                    // Apply model strings if non-empty
                    if_past(new_model != "") {
                        model := new_model;
                    }
                    if_past(new_model_reasoning != "") {
                        model_reasoning := new_model_reasoning;
                    }
                    if_past(new_model_coding != "") {
                        model_coding := new_model_coding;
                    }
                    if_past(new_model_fast != "") {
                        model_fast := new_model_fast;
                    }

                    success := 1.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({APPLY_CONFIG});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // LOAD_NATIVE_MODEL KERNEL - Load GGUF model for native inference (Phase 26)
    // =========================================================================

    kernel LoadNativeModel {
        in  model_path: TokenReg;    // Path to GGUF file
        out success: Register;
        out info: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            already_loaded ::= witness(native_model_loaded);
            current_path ::= witness(native_model_path);
        }

        phase PROPOSE {
            proposal LOAD_MODEL {
                f_ok := if_past(current_F >= native_load_cost) then 1.0 else 0.0;
                score = current_a * 0.9 * f_ok;

                effect {
                    result := primitive("model_load", model_path);
                    if_past(result == true) {
                        native_model_loaded := true;
                        native_model_path := model_path;
                        model_info := primitive("model_info");
                        native_model_info := model_info;
                        info ::= witness(model_info);
                        success := 1.0;
                        F := F - native_load_cost;
                        total_cost := total_cost + native_load_cost;
                    }
                    if_past(result != true) {
                        info ::= witness("Failed to load model - check path");
                        success := 0.0;
                    }
                }
            }

            proposal ALREADY_LOADED {
                same_path := if_past(model_path == current_path) then 1.0 else 0.0;
                is_loaded := if_past(already_loaded == true) then 1.0 else 0.0;
                score = same_path * is_loaded * 0.95;

                effect {
                    info ::= witness(native_model_info);
                    success := 1.0;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < native_load_cost) then 1.0 else 0.0;

                effect {
                    info ::= witness("Insufficient F to load model");
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({LOAD_MODEL, ALREADY_LOADED, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // ENABLE_NATIVE KERNEL - Switch to native inference mode (Phase 26)
    // =========================================================================

    kernel EnableNative {
        in  enable: Register;    // 1.0 = enable native, 0.0 = use Ollama
        out success: Register;
        out status: TokenReg;

        phase READ {
            current_native ::= witness(use_native);
            is_loaded ::= witness(native_model_loaded);
            current_path ::= witness(native_model_path);
        }

        phase PROPOSE {
            proposal ENABLE_NATIVE_MODE {
                wants_native := if_past(enable >= 1.0) then 1.0 else 0.0;
                model_ready := if_past(is_loaded == true) then 1.0 else 0.0;
                score = wants_native * model_ready;

                effect {
                    use_native := true;
                    success := 1.0;
                    status ::= witness("Native inference enabled");
                }
            }

            proposal WARN_NOT_LOADED {
                wants_native := if_past(enable >= 1.0) then 1.0 else 0.0;
                not_loaded := if_past(is_loaded != true) then 1.0 else 0.0;
                score = wants_native * not_loaded * 0.95;

                effect {
                    use_native := true;  // Set anyway, Think will handle error
                    success := 0.5;
                    status ::= witness("Warning: Native enabled but model not loaded");
                }
            }

            proposal DISABLE_NATIVE_MODE {
                wants_ollama := if_past(enable < 1.0) then 1.0 else 0.0;
                score = wants_ollama;

                effect {
                    use_native := false;
                    success := 1.0;
                    status ::= witness("Ollama mode enabled");
                }
            }
        }

        phase CHOOSE {
            choice := choose({ENABLE_NATIVE_MODE, WARN_NOT_LOADED, DISABLE_NATIVE_MODE});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // NATIVE_STATUS KERNEL - Report native inference status (Phase 26)
    // =========================================================================

    kernel NativeStatus {
        out is_native: Register;
        out is_loaded: Register;
        out model_path: TokenReg;
        out model_info: TokenReg;
        out gpu_status: TokenReg;

        phase READ {
            current_native ::= witness(use_native);
            current_loaded ::= witness(native_model_loaded);
            current_path ::= witness(native_model_path);
            current_info ::= witness(native_model_info);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    is_native := if_past(current_native == true) then 1.0 else 0.0;
                    is_loaded := if_past(current_loaded == true) then 1.0 else 0.0;
                    model_path ::= witness(current_path);
                    model_info ::= witness(current_info);

                    // Get GPU status
                    gpu_info := primitive("metal_status");
                    if_past(gpu_info.available == true) {
                        gpu_status ::= witness(gpu_info.device);
                    }
                    if_past(gpu_info.available != true) {
                        gpu_status ::= witness("CPU only");
                    }
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

    // =========================================================================
    // CACHE_STATUS KERNEL - Get KV cache status (Phase 26.4)
    // =========================================================================

    kernel CacheStatus {
        out position: Register;
        out capacity: Register;
        out usage: Register;
        out remaining: Register;

        phase READ {
            is_loaded ::= witness(native_model_loaded);
        }

        phase PROPOSE {
            proposal GET_STATUS {
                loaded_ok := if_past(is_loaded == true) then 1.0 else 0.0;
                score = loaded_ok;

                effect {
                    status := primitive("model_cache_status");
                    position := status.position;
                    capacity := status.capacity;
                    usage := status.usage;
                    remaining := status.remaining;
                }
            }

            proposal NOT_LOADED {
                not_loaded := if_past(is_loaded != true) then 1.0 else 0.0;
                score = not_loaded * 0.95;

                effect {
                    position := 0.0;
                    capacity := 0.0;
                    usage := 0.0;
                    remaining := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({GET_STATUS, NOT_LOADED});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CACHE_SHIFT KERNEL - Sliding window cache management (Phase 26.4)
    // =========================================================================

    kernel CacheShift {
        in  keep_last: Register;    // Number of tokens to keep
        out success: Register;
        out new_position: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            is_loaded ::= witness(native_model_loaded);
        }

        phase PROPOSE {
            shift_cost := 0.02;

            proposal DO_SHIFT {
                loaded_ok := if_past(is_loaded == true) then 1.0 else 0.0;
                f_ok := if_past(current_F >= shift_cost) then 1.0 else 0.0;
                score = current_a * 0.9 * loaded_ok * f_ok;

                effect {
                    result := primitive("model_cache_shift", keep_last);
                    success := if_past(result == true) then 1.0 else 0.0;

                    // Get new position
                    status := primitive("model_cache_status");
                    new_position := status.position;

                    F := F - shift_cost;
                }
            }

            proposal NOT_LOADED {
                not_loaded := if_past(is_loaded != true) then 1.0 else 0.0;
                score = not_loaded * 0.95;

                effect {
                    success := 0.0;
                    new_position := 0.0;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < shift_cost) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                    new_position := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_SHIFT, NOT_LOADED, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CACHE_SLICE KERNEL - Explicit cache range control (Phase 26.4)
    // =========================================================================

    kernel CacheSlice {
        in  start_pos: Register;    // First position to keep
        in  end_pos: Register;      // One past last position
        out success: Register;
        out new_position: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            is_loaded ::= witness(native_model_loaded);
        }

        phase PROPOSE {
            slice_cost := 0.02;

            proposal DO_SLICE {
                loaded_ok := if_past(is_loaded == true) then 1.0 else 0.0;
                f_ok := if_past(current_F >= slice_cost) then 1.0 else 0.0;
                score = current_a * 0.9 * loaded_ok * f_ok;

                effect {
                    result := primitive("model_cache_slice", start_pos, end_pos);
                    success := if_past(result == true) then 1.0 else 0.0;

                    // Get new position
                    status := primitive("model_cache_status");
                    new_position := status.position;

                    F := F - slice_cost;
                }
            }

            proposal NOT_LOADED {
                not_loaded := if_past(is_loaded != true) then 1.0 else 0.0;
                score = not_loaded * 0.95;

                effect {
                    success := 0.0;
                    new_position := 0.0;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < slice_cost) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                    new_position := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_SLICE, NOT_LOADED, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CACHE_RESET KERNEL - Reset cache for new conversation (Phase 26.4)
    // =========================================================================

    kernel CacheReset {
        out success: Register;

        phase READ {
            current_F ::= witness(F);
            is_loaded ::= witness(native_model_loaded);
        }

        phase PROPOSE {
            reset_cost := 0.01;

            proposal DO_RESET {
                loaded_ok := if_past(is_loaded == true) then 1.0 else 0.0;
                f_ok := if_past(current_F >= reset_cost) then 1.0 else 0.0;
                score = loaded_ok * f_ok;

                effect {
                    primitive("model_reset");
                    success := 1.0;
                    F := F - reset_cost;
                }
            }

            proposal NOT_LOADED {
                not_loaded := if_past(is_loaded != true) then 1.0 else 0.0;
                score = not_loaded * 0.95;

                effect {
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_RESET, NOT_LOADED});
        }

        phase COMMIT {
            commit choice;
        }
    }
}
