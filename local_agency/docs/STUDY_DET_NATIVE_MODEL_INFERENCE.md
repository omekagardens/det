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

## References

- DET v6.3 Specification (internal)
- ROADMAP_V2.md (local_agency)
- llm.ex (Phase 21 LLM Creature)
- llama.cpp (reference implementation)
- GGUF format specification (llama.cpp docs)
- Metal Performance Shaders documentation (Apple)

---

*This study is a living document. Update as implementation progresses.*
