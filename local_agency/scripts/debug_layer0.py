#!/usr/bin/env python3
"""
Detailed comparison of Layer 0 between DET and HuggingFace.
This traces each step: embedding -> norm -> QKV -> attention -> output.
"""

import sys
import os
import numpy as np

sys.path.insert(0, 'src/python')
os.chdir('/Volumes/AI_DATA/development/det_local_agency/det/local_agency')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def rms_diff(a, b):
    """Compute RMS difference between two arrays."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    return np.sqrt(np.mean((a - b) ** 2))

def max_diff(a, b):
    """Compute max absolute difference."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    return np.max(np.abs(a - b))

def rms_val(a):
    """Compute RMS of array."""
    a = np.asarray(a, dtype=np.float32).flatten()
    return np.sqrt(np.mean(a ** 2))

def compare(name, det, hf, threshold=0.1):
    """Compare arrays and print result."""
    det = np.asarray(det, dtype=np.float32)
    hf = np.asarray(hf, dtype=np.float32)

    if det.shape != hf.shape:
        print(f"  {name}: SHAPE MISMATCH det={det.shape} hf={hf.shape}")
        return False

    rms = rms_diff(det, hf)
    maxd = max_diff(det, hf)
    status = "✅" if rms < threshold else "❌"
    print(f"  {name}: RMS={rms:.6f}, Max={maxd:.6f} {status}")

    if rms >= threshold:
        # Show first few values
        det_flat = det.flatten()[:5]
        hf_flat = hf.flatten()[:5]
        print(f"    DET[:5]: {det_flat}")
        print(f"    HF[:5]:  {hf_flat}")

    return rms < threshold

def rms_norm(x, weight, eps=1e-6):
    """RMS normalization."""
    variance = np.mean(x * x, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight

def silu(x):
    """SiLU activation."""
    return x / (1 + np.exp(-x))

def main():
    prompt = "The capital of France is"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("=" * 70)
    print("Layer 0 Deep Dive Debug")
    print("=" * 70)
    print(f"Prompt: '{prompt}'")
    print()

    # Load HF model
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids
    tokens = input_ids[0].tolist()
    print(f"Tokens: {tokens}")

    # Extract layer 0 weights
    layer0 = model.model.layers[0]

    # Get weights as numpy
    embd_weight = model.model.embed_tokens.weight.detach().numpy()
    attn_norm_weight = layer0.input_layernorm.weight.detach().numpy()

    wq = layer0.self_attn.q_proj.weight.detach().numpy()  # [out, in]
    wk = layer0.self_attn.k_proj.weight.detach().numpy()  # [out, in]
    wv = layer0.self_attn.v_proj.weight.detach().numpy()  # [out, in]
    wo = layer0.self_attn.o_proj.weight.detach().numpy()  # [out, in]

    bq = layer0.self_attn.q_proj.bias.detach().numpy()
    bk = layer0.self_attn.k_proj.bias.detach().numpy()
    bv = layer0.self_attn.v_proj.bias.detach().numpy()

    # Model config
    n_embd = 896
    n_head = 14
    n_head_kv = 2
    head_dim = n_embd // n_head  # 64
    kv_dim = n_head_kv * head_dim  # 128
    norm_eps = 1e-6
    rope_base = 1000000.0

    print(f"\nConfig: n_embd={n_embd}, n_head={n_head}, n_head_kv={n_head_kv}, head_dim={head_dim}")
    print(f"Weight shapes: wq={wq.shape}, wk={wk.shape}, wv={wv.shape}, wo={wo.shape}")

    # === Step 1: Embedding Lookup ===
    print("\n--- Step 1: Embedding Lookup ---")

    # HF embedding
    with torch.no_grad():
        hf_embd = model.model.embed_tokens(input_ids).numpy()[0]  # [seq_len, n_embd]

    # Manual embedding
    det_embd = np.array([embd_weight[t] for t in tokens])

    compare("Embedding", det_embd, hf_embd, threshold=0.001)

    # === Step 2: Pre-Attention RMSNorm ===
    print("\n--- Step 2: Pre-Attention RMSNorm ---")

    # HF norm
    with torch.no_grad():
        hf_normed = layer0.input_layernorm(torch.tensor(hf_embd).unsqueeze(0)).numpy()[0]

    # Manual norm
    det_normed = rms_norm(det_embd, attn_norm_weight, norm_eps)

    compare("After RMSNorm", det_normed, hf_normed, threshold=0.01)

    # === Step 3: Q, K, V Projections ===
    print("\n--- Step 3: Q, K, V Projections ---")

    # HF QKV (for last token only)
    with torch.no_grad():
        hf_input = torch.tensor(hf_normed).unsqueeze(0)  # [1, seq, n_embd]
        hf_q = layer0.self_attn.q_proj(hf_input).numpy()[0]  # [seq, n_embd]
        hf_k = layer0.self_attn.k_proj(hf_input).numpy()[0]  # [seq, kv_dim]
        hf_v = layer0.self_attn.v_proj(hf_input).numpy()[0]  # [seq, kv_dim]

    # Manual QKV: y = x @ W^T + b
    det_q = det_normed @ wq.T + bq  # [seq, n_embd]
    det_k = det_normed @ wk.T + bk  # [seq, kv_dim]
    det_v = det_normed @ wv.T + bv  # [seq, kv_dim]

    compare("Q projection", det_q, hf_q, threshold=0.1)
    compare("K projection", det_k, hf_k, threshold=0.1)
    compare("V projection", det_v, hf_v, threshold=0.1)

    # === Step 4: RoPE ===
    print("\n--- Step 4: RoPE Application ---")

    def apply_rope_to_tensor(x, head_dim, seq_len, rope_base):
        """Apply RoPE to [seq_len, dim] tensor."""
        x_out = x.copy()
        num_heads = x.shape[1] // head_dim

        for t in range(seq_len):
            pos = t
            for h in range(num_heads):
                offset = h * head_dim
                for i in range(0, head_dim, 2):
                    freq = 1.0 / (rope_base ** (i / head_dim))
                    angle = pos * freq
                    cos_val = np.cos(angle)
                    sin_val = np.sin(angle)

                    x0 = x_out[t, offset + i]
                    x1 = x_out[t, offset + i + 1]
                    x_out[t, offset + i] = x0 * cos_val - x1 * sin_val
                    x_out[t, offset + i + 1] = x0 * sin_val + x1 * cos_val

        return x_out

    seq_len = len(tokens)

    # Apply RoPE to Q and K
    det_q_rope = apply_rope_to_tensor(det_q, head_dim, seq_len, rope_base)
    det_k_rope = apply_rope_to_tensor(det_k, head_dim, seq_len, rope_base)

    # Get HF RoPE output by running attention with rotation
    # We need to get this from the attention module
    # For simplicity, let's compare post-softmax attention output

    # === Step 5: Attention Scores ===
    print("\n--- Step 5: Attention Computation ---")

    # Reshape for attention: [seq, n_head, head_dim] for Q, [seq, n_head_kv, head_dim] for K,V
    det_q_heads = det_q_rope.reshape(seq_len, n_head, head_dim)
    det_k_heads = det_k_rope.reshape(seq_len, n_head_kv, head_dim)
    det_v_heads = det_v.reshape(seq_len, n_head_kv, head_dim)  # V doesn't get RoPE

    scale = 1.0 / np.sqrt(head_dim)
    gqa_ratio = n_head // n_head_kv  # 7

    # Compute attention for each head
    # Q: [seq, n_head, head_dim]
    # K: [seq, n_head_kv, head_dim]
    # Score: [n_head, seq, seq]

    det_attn_out = np.zeros((seq_len, n_embd))

    for h in range(n_head):
        kv_h = h // gqa_ratio

        # Q for this head: [seq, head_dim]
        q_h = det_q_heads[:, h, :]
        k_h = det_k_heads[:, kv_h, :]
        v_h = det_v_heads[:, kv_h, :]

        # Attention scores: [seq, seq]
        scores = q_h @ k_h.T * scale

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
        scores = scores + mask

        # Softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        # Apply to V: [seq, head_dim]
        out_h = attn_weights @ v_h

        det_attn_out[:, h * head_dim:(h + 1) * head_dim] = out_h

    # Output projection
    det_attn_proj = det_attn_out @ wo.T

    # Skip direct attention comparison - HF attention API is complex
    # Instead compare full layer output

    print(f"  Manual attention computed: det_attn_proj shape = {det_attn_proj.shape}")
    print(f"  det_attn_proj (last token)[:5]: {det_attn_proj[-1, :5]}")

    # === Step 6: Residual + Full Layer ===
    print("\n--- Step 6: Full Layer Output ---")

    # Add residual
    det_after_attn = det_embd + det_attn_proj

    # FFN norm
    ffn_norm_weight = layer0.post_attention_layernorm.weight.detach().numpy()
    det_ffn_normed = rms_norm(det_after_attn, ffn_norm_weight, norm_eps)

    # FFN weights
    w1 = layer0.mlp.gate_proj.weight.detach().numpy()  # [n_ff, n_embd]
    w2 = layer0.mlp.down_proj.weight.detach().numpy()  # [n_embd, n_ff]
    w3 = layer0.mlp.up_proj.weight.detach().numpy()    # [n_ff, n_embd]

    # FFN: SwiGLU
    gate = det_ffn_normed @ w1.T  # [seq, n_ff]
    up = det_ffn_normed @ w3.T    # [seq, n_ff]
    ffn_out = silu(gate) * up     # [seq, n_ff]
    det_ffn_proj = ffn_out @ w2.T # [seq, n_embd]

    # Final residual
    det_layer_out = det_after_attn + det_ffn_proj

    # HF full layer output - use output_hidden_states
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        # hidden_states[0] = embedding, hidden_states[1] = after layer 0, etc.
        hf_layer_out = outputs.hidden_states[1].numpy()[0]  # [seq, n_embd]

    print(f"  HF layer output shape: {hf_layer_out.shape}")
    compare("Full layer output", det_layer_out, hf_layer_out, threshold=1.0)

    print("\n--- Last token hidden state comparison ---")
    print(f"DET layer 0 output (last token)[:5]: {det_layer_out[-1, :5]}")
    print(f"HF  layer 0 output (last token)[:5]: {hf_layer_out[-1, :5]}")

    rms_det = rms_val(det_layer_out[-1])
    rms_hf = rms_val(hf_layer_out[-1])
    print(f"DET RMS: {rms_det:.6f}, HF RMS: {rms_hf:.6f}")

    print("\n" + "=" * 70)
    print("Debug complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
