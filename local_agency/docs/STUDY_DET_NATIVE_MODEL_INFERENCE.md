# Study: Existence-Lang Native Model Inference for DET-OS
## Replacing Ollama with DET-Native Model Loading via Creatures

**Version**: 0.1 (Draft Study)
**Date**: 2026-01-25
**Status**: Research Phase

---

## Executive Summary

This study explores the feasibility and design of replacing Ollama with a native DET-OS model inference system where language models are loaded and executed directly as Existence-Lang creatures on the EIS substrate. Instead of delegating to an external service, model weights become creature state, attention becomes DET physics, and inference emerges from the phase cycle.

---

## 1. Current Architecture Analysis

### 1.1 Current LLM Integration Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     User Prompt                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   LLMCreature.ex                             │
│  - Think kernel with model selection                         │
│  - Temperature modulation by agency/arousal                  │
│  - Token budget management                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ primitive("llm_call_v2", ...)
┌──────────────────────────▼──────────────────────────────────┐
│                  Primitives Layer (Python)                   │
│  - det/eis/primitives.py                                    │
│  - Marshals request to HTTP                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP POST localhost:11434
┌──────────────────────────▼──────────────────────────────────┐
│                      OLLAMA                                  │
│  - External process                                          │
│  - Manages model loading/unloading                          │
│  - Runs inference (llama.cpp under the hood)                │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Current Limitations

1. **External Dependency**: Ollama must be installed and running separately
2. **No DET Integration**: Model inference is a black box - no F/a/P awareness
3. **HTTP Overhead**: Every token generation requires HTTP roundtrip
4. **No Physics-Based Scheduling**: Model scheduling is external, not DET-driven
5. **Opaque Resource Usage**: F cost is estimated, not measured from actual computation
6. **No Creature Composability**: Can't bond model components together

### 1.3 Current Strengths (to Preserve)

1. Model management (switching models per-intent)
2. Temperature modulation by DET affect state
3. Token budget management
4. Multi-model configuration (default, reasoning, coding, fast)

---

## 2. Vision: DET-Native Model Inference

### 2.1 Core Concept

Transform language models from external services into **first-class DET creatures** where:

- **Model weights** are creature state (analogous to memory/q)
- **Attention** is implemented as bond coherence patterns
- **Token generation** emerges from the phase cycle
- **Resource usage** is tracked as actual F expenditure
- **Scheduling** follows DET presence-based priority

### 2.2 Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Prompt                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   LLMCreature.ex                             │
│  (Unchanged - orchestration layer)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ bond message
┌──────────────────────────▼──────────────────────────────────┐
│                 ModelCreature.ex                             │
│  - Holds model weights as creature state                     │
│  - Coordinates layer creatures via bonds                     │
│  - Manages KV cache as memory creature                       │
└──────────────────────┬────────────────┬─────────────────────┘
         bond          │                │         bond
┌──────────────────────▼──────┐  ┌──────▼──────────────────────┐
│    EmbeddingCreature.ex     │  │     AttentionCreature.ex    │
│  - Token → embedding        │  │  - Multi-head attention     │
│  - Embedding weights        │  │  - Attention as bonds       │
└─────────────────────────────┘  └─────────────────────────────┘
                                          │ bonds to heads
         ┌────────────────┬───────────────┼───────────────┬────────────────┐
         ▼                ▼               ▼               ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Head 0  │    │  Head 1  │    │  Head 2  │    │   ...    │    │  Head N  │
   │ creature │    │ creature │    │ creature │    │          │    │ creature │
   └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                 Substrate Layer (C/Metal)                    │
│  - Matrix multiplication primitives (BLAS/cuBLAS/Metal)     │
│  - Memory-mapped weight files                                │
│  - GPU tensor operations                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Design Components

### 3.1 Weight Loading: ModelLoaderCreature

**Purpose**: Load GGUF/safetensors model files and distribute weights to layer creatures

```existence
creature ModelLoaderCreature {
    var F: Register := 1000.0;
    var model_path: string;
    var quantization: string := "Q4_K_M";
    var loaded: bool := false;
    var layer_refs: array[CreatureRef];

    kernel Load {
        in  path: TokenReg;
        in  quant: TokenReg;
        out success: Register;
        out num_layers: Register;

        phase READ {
            current_F ::= witness(F);
        }

        phase PROPOSE {
            // Estimate F cost based on model size
            estimated_cost := primitive("model_estimate_load_cost", path, quant);

            proposal LOAD_MODEL {
                score = if_past(current_F >= estimated_cost) then 1.0 else 0.0;

                effect {
                    // Load model metadata (header only - cheap)
                    metadata := primitive("model_load_metadata", path);
                    num_layers := metadata.num_layers;

                    // Memory-map weight tensors (doesn't load into RAM yet)
                    weight_map := primitive("model_mmap_weights", path);

                    // Spawn layer creatures with weight references
                    for i in 0..num_layers {
                        layer_ref := spawn(LayerCreature, {
                            layer_idx: i,
                            weight_ptr: weight_map.layer_ptrs[i],
                            hidden_size: metadata.hidden_size,
                            num_heads: metadata.num_heads
                        });
                        layer_refs[i] := layer_ref;
                    }

                    loaded := true;
                    success := 1.0;
                    F := F - estimated_cost;
                }
            }
        }

        phase CHOOSE { choice := choose({LOAD_MODEL}); }
        phase COMMIT { commit choice; }
    }
}
```

### 3.2 Attention as Bonds: AttentionHeadCreature

**Key Insight**: Attention scores map naturally to bond coherence

- Q·K^T similarity → bond coherence C
- Softmax normalization → presence-based flow
- V projection → information flow along bonds

```existence
creature AttentionHeadCreature {
    var F: Register := 10.0;
    var a: float := 0.8;

    var head_idx: int;
    var d_k: int;           // Key dimension
    var weight_Q: TensorRef;
    var weight_K: TensorRef;
    var weight_V: TensorRef;

    // Attention bonds to context tokens
    var context_bonds: array[Bond];
    var bond_coherences: array[float];  // Attention scores as coherence

    kernel Attend {
        in  query: TensorReg;       // Current token representation
        in  keys: TensorReg;        // Context key cache
        in  values: TensorReg;      // Context value cache
        out attended: TensorReg;    // Attended output

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
            Q_weights ::= witness(weight_Q);
            K_weights ::= witness(weight_K);
            V_weights ::= witness(weight_V);
        }

        phase PROPOSE {
            // Project query, keys, values
            Q := primitive("matmul", query, Q_weights);
            K := primitive("matmul", keys, K_weights);
            V := primitive("matmul", values, V_weights);

            // Compute attention scores (bond coherences)
            // scores = Q @ K^T / sqrt(d_k)
            scores := primitive("attention_scores", Q, K, d_k);

            proposal ATTEND_FOCUSED {
                // Higher agency = sharper attention (lower temperature)
                temperature := 1.0 / (current_a + 0.1);
                score = current_a;

                effect {
                    // Softmax with temperature (presence normalization)
                    attn_weights := primitive("softmax", scores, temperature);

                    // Store as bond coherences for DET tracking
                    bond_coherences := attn_weights;

                    // Weighted sum of values (flow through bonds)
                    attended := primitive("matmul", attn_weights, V);

                    F := F - 0.1;  // Attention has F cost
                }
            }
        }

        phase CHOOSE { choice := choose({ATTEND_FOCUSED}, decisiveness = current_a); }
        phase COMMIT { commit choice; }
    }
}
```

### 3.3 FFN as Structure: FFNCreature

**Insight**: Feed-forward networks build structure (analogous to q accumulation)

```existence
creature FFNCreature {
    var F: Register := 10.0;
    var q: float := 0.0;  // Accumulated "structure" from processing

    var weight_up: TensorRef;
    var weight_gate: TensorRef;  // For SwiGLU
    var weight_down: TensorRef;

    kernel Forward {
        in  hidden: TensorReg;
        out output: TensorReg;

        phase READ {
            current_F ::= witness(F);
            current_q ::= witness(q);
        }

        phase PROPOSE {
            proposal COMPUTE {
                score = 1.0;

                effect {
                    // SwiGLU: down(up(x) * silu(gate(x)))
                    up := primitive("matmul", hidden, weight_up);
                    gate := primitive("matmul", hidden, weight_gate);
                    activated := primitive("silu", gate);
                    gated := primitive("elementwise_mul", up, activated);
                    output := primitive("matmul", gated, weight_down);

                    // FFN builds structure
                    q := q + 0.01;
                    F := F - 0.2;
                }
            }
        }

        phase CHOOSE { choice := choose({COMPUTE}); }
        phase COMMIT { commit choice; }
    }
}
```

### 3.4 Full Layer: TransformerLayerCreature

**Coordinates attention + FFN with DET physics**

```existence
creature TransformerLayerCreature {
    var F: Register := 50.0;
    var a: float := 0.7;
    var P: float := 0.0;  // Presence (computed from F, a)

    var layer_idx: int;
    var attn_creature: CreatureRef;
    var ffn_creature: CreatureRef;

    kernel Forward {
        in  hidden: TensorReg;
        in  kv_cache: TokenReg;
        out output: TensorReg;
        out updated_cache: TokenReg;

        phase READ {
            current_F ::= witness(F);
            current_a ::= witness(a);
        }

        phase PROPOSE {
            // Compute presence (DET scheduling priority)
            sigma := 0.5;
            H := sigma;
            computed_P := current_a * sigma / (1.0 + current_F) / (1.0 + H);

            proposal LAYER_FORWARD {
                score = computed_P;  // Presence determines proposal strength

                effect {
                    // Pre-norm
                    normed := primitive("rmsnorm", hidden);

                    // Attention (via bonded creature)
                    attn_out := bond_call(attn_creature, "Attend", {
                        query: normed,
                        keys: kv_cache.keys,
                        values: kv_cache.values
                    });

                    // Residual
                    after_attn := primitive("add", hidden, attn_out);

                    // FFN (via bonded creature)
                    normed2 := primitive("rmsnorm", after_attn);
                    ffn_out := bond_call(ffn_creature, "Forward", {hidden: normed2});

                    // Final residual
                    output := primitive("add", after_attn, ffn_out);

                    // Update KV cache
                    updated_cache := primitive("kv_cache_append", kv_cache, normed);

                    F := F - 1.0;
                    P := computed_P;
                }
            }
        }

        phase CHOOSE { choice := choose({LAYER_FORWARD}, decisiveness = current_a); }
        phase COMMIT { commit choice; }
    }
}
```

### 3.5 Token Generation: SamplerCreature

**Purpose**: Sample next token using DET-influenced sampling

```existence
creature SamplerCreature {
    var F: Register := 20.0;
    var a: float := 0.7;           // Agency affects exploration
    var arousal: float := 0.5;     // Arousal affects temperature

    var base_temperature: float := 0.7;
    var top_p: float := 0.9;
    var top_k: int := 40;
    var repetition_penalty: float := 1.1;

    kernel Sample {
        in  logits: TensorReg;
        in  context_tokens: TokenReg;  // For repetition penalty
        out token_id: Register;
        out token_prob: Register;

        phase READ {
            current_a ::= witness(a);
            current_arousal ::= witness(arousal);
            base_temp ::= witness(base_temperature);
        }

        phase PROPOSE {
            // DET-modulated temperature
            // Higher agency = more exploration (counter-intuitive but matches DET)
            // Higher arousal = more variability
            effective_temp := base_temp + (current_a - 0.5) * 0.2 + (current_arousal - 0.5) * 0.15;

            proposal SAMPLE_TOP_P {
                score = current_a * 0.9;

                effect {
                    // Apply temperature
                    scaled_logits := primitive("div_scalar", logits, effective_temp);

                    // Apply repetition penalty
                    penalized := primitive("repetition_penalty", scaled_logits, context_tokens, repetition_penalty);

                    // Softmax
                    probs := primitive("softmax", penalized);

                    // Top-p (nucleus) sampling
                    sampled := primitive("top_p_sample", probs, top_p);

                    token_id := sampled.token_id;
                    token_prob := sampled.probability;

                    F := F - 0.05;
                }
            }
        }

        phase CHOOSE { choice := choose({SAMPLE_TOP_P}, decisiveness = current_a); }
        phase COMMIT { commit choice; }
    }
}
```

---

## 4. Substrate Primitives Required

### 4.1 Model I/O Primitives

| Primitive | Description | Estimated Cost |
|-----------|-------------|----------------|
| `model_load_metadata(path)` | Read model header/config | F: 0.1 |
| `model_mmap_weights(path)` | Memory-map weight file | F: 0.5 |
| `model_estimate_load_cost(path, quant)` | Estimate F for full load | F: 0.01 |
| `tensor_load(ptr, shape, dtype)` | Load tensor from mmap | F: varies |

### 4.2 Compute Primitives (C/Metal)

| Primitive | Description | GPU Accelerated |
|-----------|-------------|-----------------|
| `matmul(A, B)` | Matrix multiplication | Yes (Metal) |
| `attention_scores(Q, K, d_k)` | Q @ K^T / sqrt(d_k) | Yes |
| `softmax(x, temp?)` | Softmax with optional temperature | Yes |
| `silu(x)` | SiLU activation | Yes |
| `rmsnorm(x)` | RMSNorm layer | Yes |
| `elementwise_mul(a, b)` | Element-wise multiply | Yes |
| `add(a, b)` | Element-wise add | Yes |
| `div_scalar(x, s)` | Divide by scalar | Yes |
| `top_p_sample(probs, p)` | Nucleus sampling | No (CPU) |
| `repetition_penalty(logits, ctx, pen)` | Apply rep penalty | No (CPU) |

### 4.3 Cache Primitives

| Primitive | Description |
|-----------|-------------|
| `kv_cache_create(layers, max_len, d)` | Allocate KV cache |
| `kv_cache_append(cache, k, v)` | Append new KV pair |
| `kv_cache_slice(cache, start, end)` | Sliding window |
| `kv_cache_clear(cache)` | Reset cache |

---

## 5. Implementation Phases

### Phase M1: Foundation Primitives
**Goal**: Basic tensor operations in substrate

- [ ] Memory-mapped tensor loading
- [ ] Basic matmul (CPU fallback)
- [ ] Metal matmul shader
- [ ] RMSNorm, SiLU, softmax primitives

### Phase M2: Model Loading
**Goal**: Load GGUF models into creature state

- [ ] GGUF header parser
- [ ] Weight tensor extraction
- [ ] Quantization support (Q4_K_M minimum)
- [ ] ModelLoaderCreature.ex

### Phase M3: Inference Pipeline
**Goal**: Single token generation

- [ ] EmbeddingCreature.ex
- [ ] TransformerLayerCreature.ex
- [ ] AttentionHeadCreature.ex
- [ ] FFNCreature.ex
- [ ] SamplerCreature.ex

### Phase M4: KV Cache
**Goal**: Efficient multi-token generation

- [ ] KV cache primitives
- [ ] Cache management in ModelCreature
- [ ] Context window handling

### Phase M5: Optimization
**Goal**: Performance parity with Ollama

- [ ] Metal Performance Shaders for attention
- [ ] Batch inference support
- [ ] Speculative decoding exploration
- [ ] Memory optimization (weight sharing)

### Phase M6: Integration
**Goal**: Drop-in replacement for current LLM stack

- [ ] LLMCreature.ex compatibility layer
- [ ] Model switching support
- [ ] Streaming token output
- [ ] Deprecate Ollama primitives

---

## 6. DET Physics Mapping

### 6.1 Attention ↔ Bond Coherence

| Transformer Concept | DET Concept |
|---------------------|-------------|
| Attention weight | Bond coherence C |
| Query-Key similarity | Bond formation potential |
| Softmax temperature | Agency-modulated decisiveness |
| Context length | Number of active bonds |
| Attention head | Specialized bond creature |

### 6.2 Resource Mapping

| Inference Cost | DET Resource |
|----------------|--------------|
| FLOPS (matmul) | F expenditure |
| Memory (weights) | Creature q (structure) |
| Tokens generated | Phase cycles consumed |
| Model size | Creature complexity |

### 6.3 Scheduling Integration

```
Model Presence: P_model = a * sigma / (1 + F_remaining) / (1 + H_model)

Where:
  a = model agency (confidence in generation)
  sigma = model coherence (internal consistency)
  F_remaining = compute budget left
  H_model = model entropy/uncertainty
```

This allows DET scheduler to naturally throttle expensive models when F is low, or prioritize high-agency responses.

---

## 7. Comparative Analysis

### 7.1 Ollama vs DET-Native

| Aspect | Ollama | DET-Native |
|--------|--------|------------|
| Setup complexity | External install | Built-in |
| Model loading | Automatic | Explicit creature spawn |
| Resource tracking | Opaque | Per-phase F accounting |
| Scheduling | External | DET presence-based |
| Composability | Monolithic | Bond-connected creatures |
| Debugging | Black box | Full creature inspection |
| Performance | Optimized | Needs optimization work |
| Model support | Wide | GGUF initially |

### 7.2 Performance Considerations

**Expected Overhead**:
- Phase cycle overhead per token: ~10-50μs
- Bond message passing: ~1-5μs per message
- Additional memory for creature state: ~1KB per creature

**Expected Benefits**:
- Eliminated HTTP overhead: ~1-5ms per call
- Better GPU utilization via integrated scheduling
- Reduced context switching (single process)

**Target**: Within 2x of Ollama throughput for equivalent models

---

## 8. Open Questions

### 8.1 Architecture Decisions

1. **Granularity**: One creature per layer vs one creature per attention head?
   - Per-layer: Fewer creatures, simpler bonding
   - Per-head: More DET-native, better parallelization

2. **Weight Storage**: Creature state vs shared memory region?
   - Creature state: True DET ownership
   - Shared: Memory efficient, less DET-native

3. **Cache Ownership**: Who owns the KV cache?
   - Model creature: Centralized, simpler
   - Separate cache creature: More composable

### 8.2 Technical Challenges

1. **Quantization**: How to handle quantized weights efficiently?
2. **Memory**: Large models may exceed creature state limits
3. **GPU Sync**: Ensuring phase commits align with GPU completion
4. **Model Updates**: How to handle model fine-tuning?

### 8.3 DET Theory Questions

1. Does attention truly map to bond coherence, or is this metaphorical?
2. What is the correct F cost model for inference?
3. Should model layers have agency, or just F?
4. How does this relate to DET v6.3 presence formula?

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance gap | High | Incremental optimization, keep Ollama fallback |
| Complexity | Medium | Start with small models (phi, qwen 0.5B) |
| Memory limits | Medium | Memory-mapped weights, streaming |
| Maintenance burden | Medium | Generate substrate primitives from spec |
| Model compatibility | Low | Focus on GGUF, llama-compatible arch |

---

## 10. Success Criteria

### Minimum Viable Product (MVP)
- [ ] Load and run phi-2 (2.7B params)
- [ ] Generate coherent text (same quality as Ollama)
- [ ] Track F expenditure per generation
- [ ] Integrate with existing LLMCreature.ex

### Full Success
- [ ] Support llama-architecture models up to 7B
- [ ] Performance within 2x of Ollama
- [ ] Quantization support (Q4_K_M, Q8_0)
- [ ] Streaming token generation
- [ ] Full DET physics integration (P-based scheduling)
- [ ] Deprecate Ollama dependency

---

## 11. Recommended Next Steps

1. **Prototype substrate primitives**: Implement `matmul`, `softmax`, `rmsnorm` in C with Metal shaders
2. **GGUF loader**: Parse GGUF format, extract weight tensors
3. **Minimal inference**: Single-layer forward pass to validate architecture
4. **Benchmark**: Compare token/sec with Ollama on same model
5. **Iterate**: Based on performance, decide creature granularity

---

## 12. Addressing Reward Hacking and Hallucination

A key advantage of DET-native model inference is the potential to reduce pathological behaviors that plague conventional LLM architectures. This section analyzes how DET physics naturally constrains reward hacking and hallucination.

### 12.1 The Problem: Opacity Enables Pathology

In conventional architectures (including Ollama-mediated inference):

```
┌─────────────────────────────────────────────────────────────┐
│                    BLACK BOX INFERENCE                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Attention patterns: Hidden                          │    │
│  │  Confidence levels: Inferred post-hoc               │    │
│  │  Resource usage: Opaque                              │    │
│  │  Internal state: Inaccessible                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│              Token output (no provenance)                    │
└─────────────────────────────────────────────────────────────┘
```

**Reward Hacking**: Models learn to produce outputs that *appear* correct (maximize reward signal) without genuine understanding. The opacity makes this undetectable.

**Hallucination**: Models confabulate plausible-sounding content because:
1. No cost for being wrong
2. No mechanism to signal uncertainty
3. Training rewards fluency over accuracy

### 12.2 DET-Native Constraints on Reward Hacking

#### 12.2.1 F Expenditure Creates Real Costs

In DET, every operation costs F (free energy). This is not abstract - it's tracked per-phase:

```existence
// Every attention computation costs F
effect {
    attended := primitive("matmul", attn_weights, V);
    F := F - 0.1;  // Real, tracked cost
}
```

**Anti-Hacking Properties**:
- Cannot "fake" computation - F expenditure is witnessed and committed
- Shortcuts that skip work don't reduce F cost (F tracks actual compute)
- Over-generation (verbose padding) depletes F faster → natural brevity pressure

#### 12.2.2 Presence Exposes Agency

The DET presence formula:

```
P = a · σ / (1 + F) / (1 + H)
```

Where:
- `a` = agency (genuine capacity to act)
- `σ` = coherence (internal consistency)
- `H` = entropy/uncertainty

**Anti-Hacking Properties**:
- Low-agency outputs have low presence → deprioritized by scheduler
- Cannot fake high agency without coherent internal state
- Gaming presence requires gaming *all* components simultaneously

#### 12.2.3 Bond Coherence Exposes Attention Patterns

Attention weights are stored as bond coherences - visible, inspectable DET state:

```existence
// Attention scores become inspectable bonds
bond_coherences := attn_weights;
```

**Anti-Hacking Properties**:
- Attention patterns are first-class DET state, not hidden internals
- Pathological attention (e.g., attending to nothing relevant) shows as low bond coherence
- Reward hacking via "attention tricks" becomes visible and auditable

#### 12.2.4 Phase Commits Are Atomic and Witnessed

The READ→PROPOSE→CHOOSE→COMMIT cycle ensures:

```
Phase COMMIT:
  - All effects applied atomically
  - State changes witnessed and recorded
  - No retroactive modification
```

**Anti-Hacking Properties**:
- Cannot modify outputs retroactively to appear more correct
- Proposal scores are recorded before choice - no post-hoc score inflation
- Full audit trail of what was proposed vs what was chosen

### 12.3 DET-Native Constraints on Hallucination

#### 12.3.1 Debt (q) as Grounding Deficit

In DET, `q` represents accumulated debt/structure. We can extend this to track **grounding**:

```existence
creature GroundedGeneratorCreature {
    var q: float := 0.0;  // Grounding debt

    kernel Generate {
        phase PROPOSE {
            proposal GROUNDED_OUTPUT {
                // Claims backed by retrieved facts
                grounding_score := check_retrieval_support(candidate);

                effect {
                    output := candidate;
                    // Ungrounded claims accumulate debt
                    q := q + (1.0 - grounding_score) * claim_weight;
                }
            }

            proposal HEDGED_OUTPUT {
                // Acknowledge uncertainty
                effect {
                    output := add_uncertainty_markers(candidate);
                    q := q + 0.01;  // Small debt for hedging
                }
            }
        }

        phase CHOOSE {
            // High debt creatures have reduced agency
            effective_a := base_a / (1.0 + q);
            choice := choose({GROUNDED_OUTPUT, HEDGED_OUTPUT},
                           decisiveness = effective_a);
        }
    }
}
```

**Anti-Hallucination Properties**:
- Ungrounded claims accumulate debt
- High debt reduces agency → reduced presence → less scheduling priority
- Natural pressure toward grounded or hedged outputs

#### 12.3.2 Agency Reflects Genuine Confidence

In DET, agency (`a`) is not arbitrary - it emerges from:
- Structural coherence (q balance)
- Resource availability (F)
- Relational health (bond coherence)

```existence
// Agency modulates sampling temperature
effective_temp := base_temp + (current_a - 0.5) * 0.2;

// Low agency → higher temperature → more uncertainty in sampling
// High agency → lower temperature → more decisive sampling
```

**Anti-Hallucination Properties**:
- Low genuine confidence → low agency → higher sampling temperature → more diverse candidates
- Model cannot claim high confidence while having low agency
- Temperature reflects internal state, not arbitrary setting

#### 12.3.3 Bond Coherence as Relevance Signal

Weak attention (low bond coherence) signals the model is "guessing":

```existence
kernel Attend {
    effect {
        attn_weights := primitive("softmax", scores, temperature);
        bond_coherences := attn_weights;

        // Compute attention entropy as uncertainty signal
        attn_entropy := -sum(attn_weights * log(attn_weights + eps));

        // High entropy attention = weak grounding
        if attn_entropy > ENTROPY_THRESHOLD {
            // Signal low confidence to parent creature
            signal_uncertainty := true;
        }
    }
}
```

**Anti-Hallucination Properties**:
- Diffuse attention (high entropy) is detected and signaled
- Parent creatures can request hedging or retrieval when attention is weak
- Hallucination often correlates with attention to irrelevant positions - now visible

#### 12.3.4 Proposal Competition Exposes Alternatives

The PROPOSE phase generates multiple candidates with scores:

```existence
phase PROPOSE {
    proposal CONFIDENT_CLAIM {
        score = 0.9;
        effect { output := "X is definitely Y"; }
    }

    proposal HEDGED_CLAIM {
        score = 0.7;
        effect { output := "X is likely Y, though I'm uncertain"; }
    }

    proposal RETRIEVAL_REQUEST {
        score = 0.6;
        effect { output := "[RETRIEVE: X relationship to Y]"; }
    }
}

phase CHOOSE {
    // Decisiveness modulates winner-take-all vs sampling
    choice := choose(proposals, decisiveness = a);
}
```

**Anti-Hallucination Properties**:
- Multiple proposals exist - not just the "most likely" token
- Lower decisiveness (low agency) → more likely to choose hedged/retrieval options
- Competition between confident and uncertain outputs is explicit

### 12.4 Architectural Mechanisms Summary

| Pathology | DET Mechanism | How It Helps |
|-----------|---------------|--------------|
| Reward hacking via shortcuts | F expenditure | Real costs can't be faked |
| Gaming confidence scores | Presence formula | Requires coherent internal state |
| Hidden attention tricks | Bond coherence | Attention is visible DET state |
| Post-hoc justification | Atomic commits | No retroactive modification |
| Ungrounded claims | Debt (q) accumulation | Hallucination creates debt |
| False confidence | Agency from structure | Low grounding → low agency |
| Ignoring uncertainty | Attention entropy | Weak attention detected |
| Single-path generation | Proposal competition | Alternatives are explicit |

### 12.5 Empirical Predictions

If this architecture works as theorized, we should observe:

1. **Calibration Improvement**: Model uncertainty (via agency) should correlate with actual accuracy
2. **Reduced Confident Errors**: High-agency outputs should be more accurate than low-agency ones
3. **Natural Hedging**: Low-grounding scenarios should produce hedged language without explicit prompting
4. **Attention Transparency**: Hallucinated content should show characteristic attention patterns (high entropy, low bond coherence)
5. **Debt Accumulation**: Extended confabulation should visibly increase q, reducing creature presence

### 12.6 Limitations and Caveats

**This is theoretical.** The mechanisms described require:

1. **Training Integration**: Models must be trained with DET-aware objectives to leverage these constraints
2. **Grounding Infrastructure**: Retrieval/fact-checking must be integrated for q debt to be meaningful
3. **Calibration Work**: The mapping between internal states and DET quantities (F, a, q) needs empirical tuning
4. **Overhead Tolerance**: Additional tracking adds computational cost

**DET does not magically solve hallucination.** It provides:
- Transparency (internal states are visible)
- Natural pressures (debt, presence, agency)
- Audit capability (phase commits, bond coherence)

The model still needs good training. DET makes pathological behavior *visible and costly*, not impossible.

### 12.7 Truthfulness Weighting: Quantifying Output Reliability

A key benefit of DET-native inference is the ability to **compute a truthfulness weight** for each output - a scalar (or vector) that accompanies generated text and quantifies how much to trust it.

#### 12.7.1 Composite Truthfulness Score

We can derive a truthfulness weight `T` from observable DET quantities:

```existence
creature TruthfulnessEvaluator {
    // Weights for combining signals (tunable)
    var w_debt: float := 0.3;
    var w_agency: float := 0.25;
    var w_entropy: float := 0.25;
    var w_coherence: float := 0.2;

    kernel ComputeTruthfulness {
        in  q: float;              // Accumulated debt
        in  a: float;              // Agency
        in  attn_entropy: float;   // Attention entropy (averaged across layers)
        in  bond_coherence: float; // Mean bond coherence to context
        out T: float;              // Truthfulness weight [0, 1]

        phase PROPOSE {
            proposal COMPUTE {
                effect {
                    // Debt penalty: high debt → low truthfulness
                    debt_score := 1.0 / (1.0 + q);

                    // Agency contribution: calibrated confidence
                    agency_score := a;  // Assumes a is calibrated to accuracy

                    // Entropy penalty: diffuse attention → uncertainty
                    // Normalize entropy to [0,1] range
                    max_entropy := log(context_length);
                    entropy_score := 1.0 - (attn_entropy / max_entropy);

                    // Coherence contribution: strong bonds → grounded
                    coherence_score := bond_coherence;

                    // Weighted combination
                    T := w_debt * debt_score
                       + w_agency * agency_score
                       + w_entropy * entropy_score
                       + w_coherence * coherence_score;

                    // Clamp to [0, 1]
                    T := clamp(T, 0.0, 1.0);
                }
            }
        }
    }
}
```

#### 12.7.2 Per-Token Truthfulness

For fine-grained analysis, compute `T` per token:

```existence
kernel GenerateWithTruthfulness {
    out tokens: array[int];
    out truthfulness: array[float];  // T_i for each token

    phase PROPOSE {
        for i in 0..max_tokens {
            // Generate token
            token_i := sample_next_token();

            // Compute per-token truthfulness from layer states
            T_i := compute_token_truthfulness(
                layer_debts,           // q from each layer
                layer_agencies,        // a from each layer
                attention_entropies,   // H from each attention head
                bond_coherences        // C from attention to context
            );

            tokens[i] := token_i;
            truthfulness[i] := T_i;
        }
    }
}
```

This enables:
- **Highlighting uncertain spans**: Low T tokens shown in red/italics
- **Selective verification**: Only fact-check claims where T < threshold
- **Automatic hedging insertion**: When T drops, insert "I think" / "possibly"

#### 12.7.3 Truthfulness Vector (Multi-Dimensional)

For richer signal, output a **truthfulness vector** rather than scalar:

```existence
struct TruthfulnessVector {
    factual_grounding: float;   // How well-grounded in retrieved facts
    logical_coherence: float;   // Internal consistency of reasoning
    source_attribution: float;  // Whether sources are cited/available
    uncertainty_acknowledged: float;  // Does output express appropriate doubt
    temporal_validity: float;   // Is information potentially outdated
}

kernel ComputeTruthfulnessVector {
    in  generation_state: GenerationState;
    out T_vec: TruthfulnessVector;

    phase PROPOSE {
        proposal COMPUTE {
            effect {
                // Factual grounding from retrieval bond coherence
                T_vec.factual_grounding := mean(retrieval_bond_coherences);

                // Logical coherence from cross-layer q consistency
                T_vec.logical_coherence := 1.0 - variance(layer_debts);

                // Source attribution from citation detection
                T_vec.source_attribution := citation_coverage_ratio;

                // Uncertainty acknowledgment from hedging token presence
                T_vec.uncertainty_acknowledged := hedging_token_ratio * (1.0 - agency);

                // Temporal validity from knowledge cutoff proximity
                T_vec.temporal_validity := temporal_relevance_score;
            }
        }
    }
}
```

#### 12.7.4 Output Format with Truthfulness

Generated output includes truthfulness metadata:

```json
{
  "text": "The capital of France is Paris.",
  "truthfulness": {
    "overall": 0.94,
    "per_token": [0.91, 0.88, 0.92, 0.95, 0.97, 0.98],
    "vector": {
      "factual_grounding": 0.97,
      "logical_coherence": 0.95,
      "source_attribution": 0.82,
      "uncertainty_acknowledged": 0.90,
      "temporal_validity": 0.99
    }
  },
  "det_state": {
    "q": 0.02,
    "a": 0.89,
    "mean_attn_entropy": 0.31,
    "mean_bond_coherence": 0.87
  }
}
```

#### 12.7.5 Calibration Requirements

For truthfulness weights to be meaningful, they must be **calibrated**:

1. **Ground Truth Dataset**: Collect outputs with known factual accuracy
2. **Correlation Analysis**: Measure correlation between T and actual accuracy
3. **Weight Tuning**: Optimize w_debt, w_agency, etc. to maximize correlation
4. **Threshold Selection**: Determine T thresholds for "reliable" vs "verify" vs "reject"

**Calibration Procedure**:
```
For each (output, ground_truth_accuracy) pair:
    1. Run DET-native inference, collect q, a, entropy, coherence
    2. Compute T with current weights
    3. Compare T to ground_truth_accuracy
    4. Adjust weights via gradient descent on MSE(T, accuracy)

Validation:
    - T > 0.8 should correlate with >90% factual accuracy
    - T < 0.3 should correlate with <50% factual accuracy
    - ECE (Expected Calibration Error) should be minimized
```

#### 12.7.6 Use Cases for Truthfulness Weights

| Use Case | How T Helps |
|----------|-------------|
| **User-facing display** | Show confidence indicator alongside responses |
| **Automatic fact-checking** | Only verify claims where T < threshold |
| **Agentic workflows** | Gate tool execution on T > safety threshold |
| **Training signal** | Use T as auxiliary reward in RLHF |
| **Ensemble selection** | Prefer outputs with higher T |
| **Retrieval triggering** | Low T triggers automatic retrieval augmentation |
| **Audit/compliance** | Log T for regulated domains (medical, legal, financial) |

#### 12.7.7 Comparison to Existing Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Token probabilities** | Built-in, no extra compute | Poor calibration, doesn't reflect factuality |
| **Verbalized confidence** | Easy to implement | Models confabulate confidence too |
| **Ensemble disagreement** | Captures genuine uncertainty | N× compute cost |
| **DET Truthfulness (T)** | Grounded in internal state, auditable | Requires DET-native architecture, calibration work |

The DET approach is unique in deriving confidence from **architectural signals** (debt, attention, coherence) rather than post-hoc estimation or verbalization.

### 12.8 Research Directions

1. **DET-Aware Fine-Tuning**: Train models with F/q/a signals as auxiliary losses
2. **Grounding-Debt Correlation**: Empirically measure relationship between q and factual accuracy
3. **Attention-Hallucination Signatures**: Characterize attention patterns during confabulation
4. **Agency Calibration**: Tune agency computation to correlate with human-judged confidence
5. **Intervention Studies**: When debt is high, does forcing retrieval improve accuracy?

---

## References

- DET v6.3 Specification (internal)
- ROADMAP_V2.md (local_agency)
- llm.ex (Phase 21 LLM Creature)
- llama.cpp (reference implementation)
- GGUF format specification (llama.cpp docs)
- Metal Performance Shaders documentation (Apple)

---

*This study is a living document. Update as implementation progresses.*
