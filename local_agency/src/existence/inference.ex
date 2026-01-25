/**
 * DET Native Inference Creatures - Phase 26.3
 * ============================================
 *
 * Existence-Lang creatures for native LLM inference.
 * Replaces Ollama dependency with DET-native GGUF inference.
 *
 * Key Creatures:
 *   - NativeModelCreature: Model loading and forward pass
 *   - SamplerCreature: DET-aware token selection (sacred integration point)
 *   - GeneratorCreature: High-level text generation
 *
 * The SamplerCreature implements the SACRED DET INTEGRATION POINT where
 * DET physics (presence field) influences token selection.
 */

// =============================================================================
// NATIVE MODEL CREATURE
// =============================================================================

creature NativeModelCreature {
    // DET state
    var F: Register := 100.0;
    var a: float := 0.7;
    var q: float := 0.0;

    // Model state
    var model_loaded: bool := false;
    var model_path: string := "";
    var vocab_size: int := 0;
    var context_length: int := 2048;

    // Statistics
    var forward_calls: int := 0;
    var tokens_processed: int := 0;
    var total_cost: float := 0.0;

    // Cost parameters
    var load_cost: float := 0.5;
    var forward_cost_per_token: float := 0.01;

    // =========================================================================
    // LOAD KERNEL - Load GGUF model from path
    // =========================================================================

    kernel Load {
        in  path: TokenReg;
        out success: Register;
        out info: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            already_loaded ::= witness(model_loaded);
        }

        phase PROPOSE {
            proposal LOAD_MODEL {
                f_ok := if_past(current_F >= load_cost) then 1.0 else 0.0;
                score = current_a * 0.9 * f_ok;

                effect {
                    result := primitive("model_load", path);
                    if_past(result == true) {
                        model_loaded := true;
                        model_path := path;
                        info ::= witness(primitive("model_info"));
                        success := 1.0;
                        F := F - load_cost;
                        total_cost := total_cost + load_cost;
                    }
                    if_past(result != true) {
                        info ::= witness("Failed to load model");
                        success := 0.0;
                    }
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < load_cost) then 1.0 else 0.0;

                effect {
                    info ::= witness("Insufficient F for model load");
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({LOAD_MODEL, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // FORWARD KERNEL - Run forward pass on tokens
    // =========================================================================

    kernel Forward {
        in  tokens: TokenReg;  // List of token IDs
        out logits_ready: Register;
        out error: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            is_loaded ::= witness(model_loaded);
        }

        phase PROPOSE {
            num_tokens := len(tokens);
            call_cost := num_tokens * forward_cost_per_token;

            proposal RUN_FORWARD {
                f_ok := if_past(current_F >= call_cost) then 1.0 else 0.0;
                loaded_ok := if_past(is_loaded == true) then 1.0 else 0.0;
                score = current_a * 0.9 * f_ok * loaded_ok;

                effect {
                    result := primitive("model_forward", tokens);
                    logits_ready := 1.0;
                    error ::= witness("");
                    forward_calls := forward_calls + 1;
                    tokens_processed := tokens_processed + num_tokens;
                    F := F - call_cost;
                    total_cost := total_cost + call_cost;
                }
            }

            proposal REFUSE_NOT_LOADED {
                score = if_past(is_loaded != true) then 1.0 else 0.0;

                effect {
                    logits_ready := 0.0;
                    error ::= witness("No model loaded");
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < call_cost) then 0.95 else 0.0;

                effect {
                    logits_ready := 0.0;
                    error ::= witness("Insufficient F for forward pass");
                }
            }
        }

        phase CHOOSE {
            choice := choose({RUN_FORWARD, REFUSE_NOT_LOADED, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // TOKENIZE KERNEL - Convert text to tokens
    // =========================================================================

    kernel Tokenize {
        in  text: TokenReg;
        out tokens: TokenReg;  // List of token IDs
        out count: Register;

        phase READ {
            is_loaded ::= witness(model_loaded);
        }

        phase PROPOSE {
            proposal DO_TOKENIZE {
                loaded_ok := if_past(is_loaded == true) then 1.0 else 0.0;
                score = loaded_ok;

                effect {
                    result := primitive("model_tokenize", text);
                    tokens ::= witness(result);
                    count := len(result);
                }
            }

            proposal FAIL_NOT_LOADED {
                score = if_past(is_loaded != true) then 1.0 else 0.0;

                effect {
                    tokens ::= witness([]);
                    count := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_TOKENIZE, FAIL_NOT_LOADED});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // DETOKENIZE KERNEL - Convert tokens to text
    // =========================================================================

    kernel Detokenize {
        in  tokens: TokenReg;
        out text: TokenReg;

        phase READ {
            is_loaded ::= witness(model_loaded);
        }

        phase PROPOSE {
            proposal DO_DETOKENIZE {
                loaded_ok := if_past(is_loaded == true) then 1.0 else 0.0;
                score = loaded_ok;

                effect {
                    result := primitive("model_detokenize", tokens);
                    text ::= witness(result);
                }
            }

            proposal FAIL_NOT_LOADED {
                score = if_past(is_loaded != true) then 1.0 else 0.0;

                effect {
                    text ::= witness("");
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_DETOKENIZE, FAIL_NOT_LOADED});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // RESET KERNEL - Reset KV cache for new conversation
    // =========================================================================

    kernel Reset {
        out success: Register;

        phase READ {
            is_loaded ::= witness(model_loaded);
        }

        phase PROPOSE {
            proposal DO_RESET {
                loaded_ok := if_past(is_loaded == true) then 1.0 else 0.0;
                score = loaded_ok;

                effect {
                    primitive("model_reset");
                    success := 1.0;
                }
            }

            proposal SKIP_NOT_LOADED {
                score = if_past(is_loaded != true) then 1.0 else 0.0;

                effect {
                    success := 0.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_RESET, SKIP_NOT_LOADED});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STATUS KERNEL - Report model status
    // =========================================================================

    kernel Status {
        out is_loaded: Register;
        out path: TokenReg;
        out info: TokenReg;
        out calls: Register;
        out tokens: Register;
        out cost: Register;

        phase READ {
            current_loaded ::= witness(model_loaded);
            current_path ::= witness(model_path);
            current_calls ::= witness(forward_calls);
            current_tokens ::= witness(tokens_processed);
            current_cost ::= witness(total_cost);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    is_loaded := if_past(current_loaded == true) then 1.0 else 0.0;
                    path ::= witness(current_path);
                    info ::= witness(primitive("model_info"));
                    calls := current_calls;
                    tokens := current_tokens;
                    cost := current_cost;
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


// =============================================================================
// SAMPLER CREATURE - DET-AWARE TOKEN SELECTION (SACRED INTEGRATION POINT)
// =============================================================================

creature SamplerCreature {
    // DET state - this is where presence influences sampling
    var F: Register := 100.0;
    var a: float := 0.7;          // Agency affects decisiveness
    var q: float := 0.0;          // Structure affects consistency
    var P: float := 1.0;          // Presence - THE KEY INTEGRATION

    // Sampling parameters
    var base_temperature: float := 0.7;
    var top_p: float := 0.9;
    var top_k: int := 40;
    var repetition_penalty: float := 1.1;

    // DET modulation weights
    var presence_weight: float := 0.3;    // How much presence biases sampling
    var agency_temp_weight: float := 0.2;  // Agency -> temperature
    var structure_temp_weight: float := 0.1;  // Structure -> consistency

    // Statistics
    var samples_made: int := 0;
    var total_cost: float := 0.0;

    // Cost parameters
    var sample_cost: float := 0.01;

    // =========================================================================
    // SAMPLE KERNEL - Standard sampling (no DET bias)
    // =========================================================================

    kernel Sample {
        in  temperature: Register;
        in  nucleus_p: Register;
        out token_id: Register;
        out error: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            proposal DO_SAMPLE {
                f_ok := if_past(current_F >= sample_cost) then 1.0 else 0.0;
                score = current_a * f_ok;

                effect {
                    // Use internal logits from last forward pass
                    result := primitive("model_sample", [], temperature, nucleus_p);
                    token_id := result;
                    error ::= witness("");
                    samples_made := samples_made + 1;
                    F := F - sample_cost;
                    total_cost := total_cost + sample_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < sample_cost) then 1.0 else 0.0;

                effect {
                    token_id := -1.0;
                    error ::= witness("Insufficient F for sampling");
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_SAMPLE, REFUSE_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // DET_SAMPLE KERNEL - DET-AWARE SAMPLING (THE SACRED INTEGRATION POINT)
    // =========================================================================

    /**
     * This is THE KEY integration between DET physics and language model inference.
     *
     * The presence field (P) computed by the substrate influences which tokens
     * are more likely to be selected. High presence on certain semantic concepts
     * biases the model toward generating those concepts.
     *
     * This creates a feedback loop:
     *   1. Substrate computes presence field based on DET dynamics
     *   2. Presence biases token selection
     *   3. Generated tokens update the substrate state
     *   4. Loop continues
     *
     * The det_presence array maps vocab indices to presence biases.
     * In practice, this would be computed from semantic embeddings of
     * the substrate's current state.
     */

    kernel DetSample {
        in  det_presence: TokenReg;  // Presence bias per vocab token
        out token_id: Register;
        out token_text: TokenReg;
        out error: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_q ::= witness(q);
            current_P ::= witness(P);
            current_temp ::= witness(base_temperature);
            current_top_p ::= witness(top_p);
        }

        phase PROPOSE {
            // Compute DET-modulated temperature
            // Higher agency -> more exploration (higher temp)
            // Higher structure -> more consistency (lower temp)
            agency_mod := (current_a - 0.5) * agency_temp_weight;
            structure_mod := current_q * structure_temp_weight;
            effective_temp := current_temp + agency_mod - structure_mod;

            // Clamp temperature
            if_past(effective_temp < 0.1) { effective_temp := 0.1; }
            if_past(effective_temp > 2.0) { effective_temp := 2.0; }

            proposal DO_DET_SAMPLE {
                f_ok := if_past(current_F >= sample_cost) then 1.0 else 0.0;
                score = current_a * current_P * f_ok;

                effect {
                    // Call the sacred primitive
                    result := primitive("det_choose_token", [], effective_temp, current_top_p, det_presence);
                    token_id := result;
                    token_text ::= witness(primitive("model_token_text", result));
                    error ::= witness("");
                    samples_made := samples_made + 1;
                    F := F - sample_cost;
                    total_cost := total_cost + sample_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < sample_cost) then 1.0 else 0.0;

                effect {
                    token_id := -1.0;
                    token_text ::= witness("");
                    error ::= witness("Insufficient F for DET sampling");
                }
            }

            proposal REFUSE_LOW_PRESENCE {
                // If presence is near zero, entity is fading - refuse to generate
                score = if_past(current_P < 0.1) then 0.95 else 0.0;

                effect {
                    token_id := -1.0;
                    token_text ::= witness("");
                    error ::= witness("Presence too low - entity fading");
                }
            }
        }

        phase CHOOSE {
            // Decisiveness tied to agency * presence
            dec := current_a * current_P;
            choice := choose({DO_DET_SAMPLE, REFUSE_LOW_F, REFUSE_LOW_PRESENCE}, decisiveness = dec);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CONFIGURE KERNEL - Update sampling parameters
    // =========================================================================

    kernel Configure {
        in  new_temperature: Register;
        in  new_top_p: Register;
        in  new_top_k: Register;
        in  new_presence_weight: Register;
        out success: Register;

        phase READ {
            current_temp ::= witness(base_temperature);
            current_top_p ::= witness(top_p);
        }

        phase PROPOSE {
            proposal APPLY_CONFIG {
                score = 1.0;

                effect {
                    if_past(new_temperature >= 0.0) {
                        clamped := new_temperature;
                        if_past(clamped > 2.0) { clamped := 2.0; }
                        if_past(clamped < 0.1) { clamped := 0.1; }
                        base_temperature := clamped;
                    }
                    if_past(new_top_p >= 0.0) {
                        clamped := new_top_p;
                        if_past(clamped > 1.0) { clamped := 1.0; }
                        if_past(clamped < 0.1) { clamped := 0.1; }
                        top_p := clamped;
                    }
                    if_past(new_top_k >= 0.0) {
                        top_k := new_top_k;
                    }
                    if_past(new_presence_weight >= 0.0) {
                        clamped := new_presence_weight;
                        if_past(clamped > 1.0) { clamped := 1.0; }
                        presence_weight := clamped;
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
    // SET_DET_STATE KERNEL - Update DET state from substrate
    // =========================================================================

    kernel SetDetState {
        in  new_F: Register;
        in  new_a: Register;
        in  new_q: Register;
        in  new_P: Register;
        out success: Register;

        phase READ {
            // Nothing to read - just updating
        }

        phase PROPOSE {
            proposal UPDATE_STATE {
                score = 1.0;

                effect {
                    if_past(new_F >= 0.0) { F := new_F; }
                    if_past(new_a >= 0.0) {
                        clamped := new_a;
                        if_past(clamped > 1.0) { clamped := 1.0; }
                        a := clamped;
                    }
                    if_past(new_q >= 0.0) { q := new_q; }
                    if_past(new_P >= 0.0) { P := new_P; }
                    success := 1.0;
                }
            }
        }

        phase CHOOSE {
            choice := choose({UPDATE_STATE});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // STATUS KERNEL - Report sampler status
    // =========================================================================

    kernel Status {
        out status_F: Register;
        out status_a: Register;
        out status_q: Register;
        out status_P: Register;
        out status_temp: Register;
        out status_top_p: Register;
        out status_samples: Register;
        out status_cost: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_q ::= witness(q);
            current_P ::= witness(P);
            current_temp ::= witness(base_temperature);
            current_top_p ::= witness(top_p);
            current_samples ::= witness(samples_made);
            current_cost ::= witness(total_cost);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    status_F := current_F;
                    status_a := current_a;
                    status_q := current_q;
                    status_P := current_P;
                    status_temp := current_temp;
                    status_top_p := current_top_p;
                    status_samples := current_samples;
                    status_cost := current_cost;
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


// =============================================================================
// GENERATOR CREATURE - HIGH-LEVEL TEXT GENERATION
// =============================================================================

creature GeneratorCreature {
    // DET state
    var F: Register := 100.0;
    var a: float := 0.7;
    var q: float := 0.0;
    var P: float := 1.0;

    // Generation parameters
    var max_tokens: int := 256;
    var temperature: float := 0.7;
    var top_p: float := 0.9;

    // Bonded creatures (composition)
    var model_creature: bond := null;
    var sampler_creature: bond := null;

    // Statistics
    var generations: int := 0;
    var tokens_generated: int := 0;
    var total_cost: float := 0.0;

    // Cost
    var base_cost: float := 1.0;

    // =========================================================================
    // GENERATE KERNEL - Full text generation from prompt
    // =========================================================================

    kernel Generate {
        in  prompt: TokenReg;
        in  max_length: Register;
        out response: TokenReg;
        out token_count: Register;
        out error: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_P ::= witness(P);
            current_temp ::= witness(temperature);
            current_top_p ::= witness(top_p);
            current_max ::= witness(max_tokens);
        }

        phase PROPOSE {
            // Use provided max_length or default
            gen_max := if_past(max_length > 0.0) then max_length else current_max;

            proposal DO_GENERATE {
                f_ok := if_past(current_F >= base_cost) then 1.0 else 0.0;
                score = current_a * current_P * f_ok;

                effect {
                    // Use high-level generate primitive
                    result := primitive("model_generate", prompt, gen_max);
                    response ::= witness(result);

                    // Count tokens (approximate from length)
                    gen_tokens := len(result) / 4;  // Rough estimate
                    token_count := gen_tokens;
                    error ::= witness("");

                    // Update statistics
                    generations := generations + 1;
                    tokens_generated := tokens_generated + gen_tokens;
                    gen_cost := base_cost + (gen_tokens * 0.02);
                    F := F - gen_cost;
                    total_cost := total_cost + gen_cost;
                }
            }

            proposal REFUSE_LOW_F {
                score = if_past(current_F < base_cost) then 1.0 else 0.0;

                effect {
                    response ::= witness("");
                    token_count := 0.0;
                    error ::= witness("Insufficient F for generation");
                }
            }

            proposal REFUSE_LOW_PRESENCE {
                score = if_past(current_P < 0.1) then 0.9 else 0.0;

                effect {
                    response ::= witness("");
                    token_count := 0.0;
                    error ::= witness("Presence too low");
                }
            }
        }

        phase CHOOSE {
            dec := current_a * current_P;
            choice := choose({DO_GENERATE, REFUSE_LOW_F, REFUSE_LOW_PRESENCE}, decisiveness = dec);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // GENERATE_STEP KERNEL - Single token generation (for streaming)
    // =========================================================================

    kernel GenerateStep {
        in  tokens: TokenReg;      // Current token sequence
        in  det_presence: TokenReg; // DET presence bias (optional)
        out token_id: Register;
        out token_text: TokenReg;
        out is_eos: Register;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_P ::= witness(P);
            current_temp ::= witness(temperature);
            current_top_p ::= witness(top_p);
        }

        phase PROPOSE {
            step_cost := 0.05;

            proposal DO_STEP {
                f_ok := if_past(current_F >= step_cost) then 1.0 else 0.0;
                score = current_a * current_P * f_ok;

                effect {
                    result := primitive("model_generate_step", tokens, current_temp, current_top_p);
                    token_id := result.token_id;
                    token_text ::= witness(result.token_text);
                    is_eos := if_past(result.is_eos == true) then 1.0 else 0.0;

                    F := F - step_cost;
                    tokens_generated := tokens_generated + 1;
                    total_cost := total_cost + step_cost;
                }
            }

            proposal FAIL_LOW_F {
                score = if_past(current_F < step_cost) then 1.0 else 0.0;

                effect {
                    token_id := -1.0;
                    token_text ::= witness("");
                    is_eos := 1.0;  // Force stop on failure
                }
            }
        }

        phase CHOOSE {
            choice := choose({DO_STEP, FAIL_LOW_F}, decisiveness = current_a);
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // CONFIGURE KERNEL - Update generation parameters
    // =========================================================================

    kernel Configure {
        in  new_max_tokens: Register;
        in  new_temperature: Register;
        in  new_top_p: Register;
        out success: Register;

        phase READ {
            // Nothing to read
        }

        phase PROPOSE {
            proposal APPLY_CONFIG {
                score = 1.0;

                effect {
                    if_past(new_max_tokens > 0.0) {
                        max_tokens := new_max_tokens;
                    }
                    if_past(new_temperature >= 0.0) {
                        clamped := new_temperature;
                        if_past(clamped > 2.0) { clamped := 2.0; }
                        if_past(clamped < 0.1) { clamped := 0.1; }
                        temperature := clamped;
                    }
                    if_past(new_top_p >= 0.0) {
                        clamped := new_top_p;
                        if_past(clamped > 1.0) { clamped := 1.0; }
                        if_past(clamped < 0.1) { clamped := 0.1; }
                        top_p := clamped;
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
    // STATUS KERNEL - Report generator status
    // =========================================================================

    kernel Status {
        out status_F: Register;
        out status_a: Register;
        out status_P: Register;
        out status_generations: Register;
        out status_tokens: Register;
        out status_cost: Register;
        out status_gpu: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            current_P ::= witness(P);
            current_gen ::= witness(generations);
            current_tokens ::= witness(tokens_generated);
            current_cost ::= witness(total_cost);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;

                effect {
                    status_F := current_F;
                    status_a := current_a;
                    status_P := current_P;
                    status_generations := current_gen;
                    status_tokens := current_tokens;
                    status_cost := current_cost;

                    // Get GPU status
                    gpu_info := primitive("metal_status");
                    if_past(gpu_info.available == true) {
                        status_gpu ::= witness(gpu_info.device);
                    }
                    if_past(gpu_info.available != true) {
                        status_gpu ::= witness("CPU only");
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
}
