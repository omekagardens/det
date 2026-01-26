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

## Remaining Issue

Despite all fixes, the final logits don't match HF output:
- Our " Paris" rank: ~400, logit ~8.9
- HF " Paris" rank: 1, logit ~17.2

### Hypothesis: Error Accumulation

The QKV projections match within ~0.01 per element. Across 24 layers with attention, FFN, and residuals, these small errors compound.

Quantization noise (Q8_0 vs FP32) contributes ~0.01 per operation. Over thousands of operations:
- 24 layers × (attention + FFN) × residuals
- Attention softmax amplifies small differences
- Final logits span ~30 points, but "Paris" vs "the" is ~2 points apart

### Areas to Investigate

1. **Attention Score Computation**: Verify Q·K^T scaling and masking
2. **Softmax Numerical Stability**: Check max subtraction is correct
3. **Residual Connections**: Verify add vs overwrite
4. **FFN Activation**: SiLU implementation matches
5. **Dequantization**: Q8_0 block alignment and scale reading
6. **Multi-token Context**: KV cache population across prompt tokens

---

## Files Modified

| File | Changes |
|------|---------|
| `src/inference/src/det_model.c` | Fixed weight indexing, added biases, fixed RoPE |
| `src/inference/src/det_model.c` | Added norm_eps fallback for Qwen2 |
| `src/inference/include/det_model.h` | (header already had bias fields) |

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

model = Model("models/Qwen2.5-0.5B-Instruct-Q8_0.gguf")
tokens = model.tokenize("The capital of France is")
# Remove BOS token (HF doesn't add one)
tokens_no_bos = [t for t in tokens if t != 151643]
model.reset()
logits = model.forward(tokens_no_bos)
# Check predictions...
EOF
```

---

## Next Steps

1. **Add Debug Logging**: Print layer-by-layer hidden state RMS to trace divergence
2. **Compare Attention Scores**: Verify Q·K^T values match HF
3. **Test Single Layer**: Run just layer 0 and compare output
4. **Check Dequantization**: Verify Q8_0 block boundaries align correctly
5. **Consider FP16 GGUF**: Use `qwen2.5-0.5b-instruct-fp16.gguf` to eliminate quantization as variable

---

*Last Updated: 2025-01-26*
