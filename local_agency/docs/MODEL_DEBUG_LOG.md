# DET Native Model Inference - Debug Log

## Overview

This document traces the debugging process for Phase 26's native model inference implementation. The goal is to run GGUF models directly without Ollama, with `det_model.c` providing the forward pass.

## Test Configuration

- **Model**: Qwen2.5-0.5B-Instruct (Q8_0 quantization)
- **Architecture**: 24 layers, 896 embedding, 14 Q heads, 2 KV heads (GQA)
- **Test Prompt**: "The capital of France is"
- **Expected Output**: "Paris" should rank #1

---

## Issue Summary

The forward pass runs but produces incorrect logits. For "The capital of France is":
- **Expected**: "Paris" (token 12095) ranked #1 with logit ~17.2
- **Actual**: " " or "the" ranked #1, "Paris" ranked ~400-13000

---

## Root Causes Identified

### 1. Corrupted GGUF File (FIXED)

**Problem**: The original `qwen2-0_5b-instruct-q8_0.gguf` had corrupted QKV bias values.

**Evidence**:
```
GGUF bk[:5]: [12.5, -10.4375, 10.625, -3.28125, 1.1015625]
HF   bk[:5]: [-8.8125, -3.328125, -6.3125, 0.6171875, -0.17578125]
```

The values are completely different - not even similar in magnitude pattern.

**Fix**: Downloaded correct GGUF from `lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF`:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF",
    filename="Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    local_dir="models"
)
```

After fix, biases match exactly (max diff = 0.000000).

---

### 2. Weight Indexing Convention (FIXED)

**Problem**: Matrix multiplication used wrong index order for GGUF's storage convention.

**GGUF Convention**:
- Weights stored as `[out_features, in_features]` in row-major order
- `ne[0]` = columns (innermost dimension)
- `ne[1]` = rows
- `W[row, col] = data[row * num_cols + col] = data[row * ne[0] + col]`

**Original Code** (incorrect):
```c
sum += h[j] * wk[j * kv_dim + i];  // WRONG!
```

**Fixed Code**:
```c
sum += h[j] * wk[i * cfg->n_embd + j];  // CORRECT
```

**Fix Applied To**:
- Q projection: `wq[i * n_embd + j]`
- K projection: `wk[i * n_embd + j]`
- V projection: `wv[i * n_embd + j]`
- Output projection: `wo[i * n_embd + j]`
- FFN gate (w1): `w1[i * n_embd + j]`
- FFN up (w3): `w3[i * n_embd + j]`
- FFN down (w2): `w2[i * n_ff + j]`

---

### 3. Missing QKV Biases (FIXED)

**Problem**: Qwen2 uses biases for Q, K, V projections but they weren't being loaded or applied.

**Evidence**: Qwen2 GGUF contains 72 bias tensors:
```
blk.0.attn_q.bias: shape=[896]
blk.0.attn_k.bias: shape=[128]
blk.0.attn_v.bias: shape=[128]
...
```

**Fix**: Added bias loading and application:
```c
// Loading
layer->bq = get_layer_bias(gguf, i, "attn_q");
layer->bk = get_layer_bias(gguf, i, "attn_k");
layer->bv = get_layer_bias(gguf, i, "attn_v");

// Application
float sum = bq ? bq[i] : 0.0f;
for (int j = 0; j < cfg->n_embd; j++) {
    sum += h[j] * wq[i * cfg->n_embd + j];
}
qt[i] = sum;
```

---

### 4. RoPE Application (FIXED)

**Problem**: RoPE was applying to K vectors multiple times (once per Q head sharing that K head).

**Original** (incorrect):
```c
for (int head = 0; head < cfg->n_head; head++) {
    apply_rope(qt + head * head_dim,
               kt + (head / gqa_ratio) * head_dim,  // K modified 7x!
               head_dim, pos, theta);
}
```

**Fixed**: Separate loops for Q and K:
```c
// Apply RoPE to all Q heads
for (int head = 0; head < cfg->n_head; head++) {
    // Apply to qt + head * head_dim
}

// Apply RoPE to K heads (once per KV head)
for (int head = 0; head < cfg->n_head_kv; head++) {
    // Apply to kt + head * head_dim
}
```

---

### 5. Norm Epsilon Metadata Key (FIXED)

**Problem**: Code looked for `llama.attention.layer_norm_rms_epsilon` but Qwen2 uses `qwen2.attention.layer_norm_rms_epsilon`.

**Fix**:
```c
model->config.norm_eps = gguf_get_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 0.0f);
if (model->config.norm_eps == 0.0f) {
    model->config.norm_eps = gguf_get_f32(gguf, "qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
}
```

---

## Verification Results

### Components Verified Correct

| Component | Max Diff from HF | Status |
|-----------|-----------------|--------|
| Embedding lookup | 0.0001 | ✅ PASS |
| RMSNorm | 0.0003 | ✅ PASS |
| Q projection | 0.01 | ✅ PASS |
| K projection | 0.004 | ✅ PASS |
| V projection | 0.0002 | ✅ PASS |

### Comparison Values (Token 374 = "is")

**Embedding**:
```
Our:  [-0.00502968, 0.0152986, -0.00586796, -0.00041914, -0.00796366]
HF:   [-0.00512695, 0.01538086, -0.00585938, -0.00045204, -0.00805664]
```

**After RMSNorm**:
```
Our:  [0.01809783, 0.0521805, 0.01143682, -0.00271782, 0.02671472]
HF:   [0.01844361, 0.05244901, 0.01141747, -0.00293049, 0.02702044]
```

**Q Projection**:
```
Our:  [0.0350, 0.2099, -0.0472, -0.6649, -13.7138]
HF:   [0.0358, 0.2107, -0.0466, -0.6665, -13.7124]
```

---

### 6. Tokenizer Length Check Bug (FIXED)

**Problem**: The tokenizer's BPE loop checked `search_start[len - 1] == '\0'` which reads garbage memory when len > strlen(search_start).

**Evidence**: Token " is" at end of string was tokenized as 285 ("is") instead of 374 ("Ġis").

**Fix**: Changed to proper length check using `strlen()`:
```c
size_t search_len = strlen(search_start);
for (int len = 16; len >= 1; len--) {
    if ((size_t)len > search_len) continue;  // Proper bounds check
    ...
}
```

---

### 7. RoPE Pairing Convention (FIXED)

**Problem**: Our RoPE implementation paired element `i` with `i+1` (consecutive), but HF/Qwen2 pairs element `i` with `i+head_dim/2` (split-half).

**Evidence**:
```
HF rotate_half pairs: element[i] with element[i + head_dim/2]
Our C implementation: element[i] with element[i + 1]
```

**Fix**: Changed to split-half pairing like HF:
```c
int half_dim = head_dim / 2;
for (int i = 0; i < half_dim; i++) {
    float freq = 1.0f / powf(rope_base, (float)(2 * i) / head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);
    float x0 = qh[i];
    float x1 = qh[i + half_dim];
    qh[i]            = x0 * cos_val - x1 * sin_val;
    qh[i + half_dim] = x1 * cos_val + x0 * sin_val;
}
```

---

## Final Results

After all fixes, the model produces **correct predictions**:

| Metric | Before | After |
|--------|--------|-------|
| " Paris" rank | ~400 | **1** ✅ |
| " Paris" logit | ~8.9 | **17.26** |
| Logits RMS diff | 4.27 | **0.08** |
| Logits max diff | 15.28 | **0.48** |

### Top 5 Predictions Comparison

| Rank | HF (FP32) | DET (Q8_0) |
|------|-----------|------------|
| 1 | " Paris" (17.22) | " Paris" (17.26) ✅ |
| 2 | " ______" (16.32) | " ______" (16.29) ✅ |
| 3 | ":\n" (15.70) | ":\n" (15.69) ✅ |
| 4 | ":\n\n" (15.57) | ":\n\n" (15.66) ✅ |
| 5 | " __" (15.39) | " __" (15.33) ✅ |

The small remaining differences (~0.05 logit) are expected from Q8_0 quantization noise.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/inference/src/det_model.c` | Fixed weight indexing, added biases, fixed RoPE pairing convention |
| `src/inference/src/det_tokenizer.c` | Fixed BPE tokenizer length check bug |
| `src/inference/include/det_model.h` | (header already had bias fields) |
| `src/python/det/inference.py` | Added `forward_logits` and `forward_logits_2d` methods |

---

## Test Commands

```bash
# Rebuild library
cd src/inference/build && make -j4

# Run Python test
python3 << 'EOF'
import sys
sys.path.insert(0, 'src/python')
from det.inference import Model
import numpy as np

model = Model("models/Qwen2.5-0.5B-Instruct-Q8_0.gguf")
model.reset()
tokens = [785, 6722, 315, 9625, 374]  # "The capital of France is"
logits = model.forward_logits_2d(tokens)
last_logits = np.array(logits[-1])

# Top prediction should be " Paris" (token 12095)
top_idx = np.argmax(last_logits)
print(f"Top prediction: token {top_idx}, logit {last_logits[top_idx]:.4f}")
print(f"Paris rank: {np.sum(last_logits >= last_logits[12095])}")  # Should be 1
EOF
```

---

## Summary

The native DET model inference is now **working correctly**:

- ✅ Embedding lookup
- ✅ RMSNorm
- ✅ QKV projections with biases
- ✅ RoPE (split-half pairing)
- ✅ Attention (GQA with KV cache)
- ✅ FFN (SwiGLU)
- ✅ Output projection
- ✅ BPE tokenization

The small remaining logit differences (~0.05) between Q8_0 and FP32 are acceptable and expected from quantization.

---

*Last Updated: 2025-01-26*
