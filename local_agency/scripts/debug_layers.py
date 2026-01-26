#!/usr/bin/env python3
"""
Layer-by-layer comparison of DET inference vs HuggingFace.
This script identifies exactly where divergence begins.
"""

import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, 'src/python')
os.chdir('/Volumes/AI_DATA/development/det_local_agency/det/local_agency')

# Try to import our model
try:
    from det.inference import Model
    HAS_DET = True
except ImportError as e:
    print(f"Warning: Could not import det.inference: {e}")
    HAS_DET = False

# Try to import HuggingFace
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_HF = True
except ImportError:
    print("Warning: transformers not available")
    HAS_HF = False

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

def load_hf_model():
    """Load HuggingFace model for comparison."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading HuggingFace model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()
    return model, tokenizer

def hook_factory(storage, name):
    """Create a forward hook that stores activations."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            storage[name] = output[0].detach().cpu().numpy()
        else:
            storage[name] = output.detach().cpu().numpy()
    return hook

def run_hf_with_hooks(model, tokenizer, prompt):
    """Run HF model and capture intermediate activations."""
    # Tokenize without BOS (Qwen2 doesn't use explicit BOS)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs.input_ids
    print(f"HF tokens: {input_ids.tolist()[0]}")

    activations = {}
    hooks = []

    # Hook the embedding output
    def emb_hook(module, input, output):
        activations['embedding'] = output.detach().cpu().numpy()
    hooks.append(model.model.embed_tokens.register_forward_hook(emb_hook))

    # Hook each layer's output
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(
            hook_factory(activations, f'layer_{i}_output')
        ))
        # Hook attention output before residual
        hooks.append(layer.self_attn.register_forward_hook(
            hook_factory(activations, f'layer_{i}_attn')
        ))
        # Hook MLP output before residual
        hooks.append(layer.mlp.register_forward_hook(
            hook_factory(activations, f'layer_{i}_mlp')
        ))
        # Hook input layer norm
        hooks.append(layer.input_layernorm.register_forward_hook(
            hook_factory(activations, f'layer_{i}_attn_norm')
        ))
        # Hook post-attention layer norm
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            hook_factory(activations, f'layer_{i}_ffn_norm')
        ))

    # Hook final norm
    hooks.append(model.model.norm.register_forward_hook(
        hook_factory(activations, 'final_norm')
    ))

    # Run forward
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Get logits
    activations['logits'] = outputs.logits.detach().cpu().numpy()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations, input_ids.tolist()[0]

def run_det_forward(model_path, tokens):
    """Run DET model forward pass."""
    if not HAS_DET:
        print("DET model not available")
        return None

    model = Model(model_path)
    model.reset()

    # Get logits for each token
    logits = model.forward(tokens)

    return {
        'logits': np.array(logits).reshape(len(tokens), -1)
    }

def compare_component(name, det_val, hf_val, threshold=0.1):
    """Compare a component between DET and HF."""
    if det_val is None or hf_val is None:
        print(f"  {name}: SKIP (missing data)")
        return None

    det_val = np.asarray(det_val, dtype=np.float32)
    hf_val = np.asarray(hf_val, dtype=np.float32)

    if det_val.shape != hf_val.shape:
        print(f"  {name}: SHAPE MISMATCH det={det_val.shape} hf={hf_val.shape}")
        return None

    rms = rms_diff(det_val, hf_val)
    maxd = max_diff(det_val, hf_val)
    det_rms = rms_val(det_val)
    hf_rms = rms_val(hf_val)

    status = "✅" if rms < threshold else "❌"
    print(f"  {name}: RMS_diff={rms:.6f}, Max_diff={maxd:.6f}, "
          f"DET_rms={det_rms:.4f}, HF_rms={hf_rms:.4f} {status}")

    return rms

def main():
    prompt = "The capital of France is"
    model_path = "models/Qwen2.5-0.5B-Instruct-Q8_0.gguf"

    print("=" * 70)
    print("DET Model Layer-by-Layer Debug")
    print("=" * 70)
    print(f"Prompt: '{prompt}'")
    print()

    # Run HuggingFace
    if HAS_HF:
        print("Running HuggingFace model...")
        hf_model, hf_tokenizer = load_hf_model()
        hf_activations, hf_tokens = run_hf_with_hooks(hf_model, hf_tokenizer, prompt)
        print(f"HF tokens: {hf_tokens}")

        # Get top predictions
        logits = hf_activations['logits'][0, -1, :]  # Last token's logits
        top_k = 10
        top_indices = np.argsort(logits)[-top_k:][::-1]
        print(f"\nHF Top {top_k} predictions for next token:")
        for i, idx in enumerate(top_indices):
            token_text = hf_tokenizer.decode([idx])
            print(f"  {i+1}. Token {idx} ({repr(token_text)}): logit={logits[idx]:.4f}")

        # Check Paris token
        paris_token = 12095  # " Paris" token
        paris_rank = np.sum(logits >= logits[paris_token])
        print(f"\n' Paris' (token {paris_token}) rank: {paris_rank}, logit: {logits[paris_token]:.4f}")
    else:
        hf_activations = None
        hf_tokens = None

    # Run DET
    if HAS_DET:
        print("\n" + "-" * 70)
        print("Running DET model...")

        det_model = Model(model_path)
        det_tokens = det_model.tokenize(prompt)
        print(f"DET tokens (raw): {det_tokens}")

        # Remove BOS if present (HF doesn't add one for Qwen2)
        bos_token = 151643
        det_tokens_no_bos = [t for t in det_tokens if t != bos_token]
        print(f"DET tokens (no BOS): {det_tokens_no_bos}")

        # Compare tokens
        if hf_tokens:
            if det_tokens_no_bos == hf_tokens:
                print("✅ Tokens match HF")
            else:
                print("❌ Token mismatch!")
                print(f"   DET: {det_tokens_no_bos}")
                print(f"   HF:  {hf_tokens}")

        det_model.reset()
        det_logits_2d = det_model.forward_logits_2d(det_tokens_no_bos)
        det_logits = np.array(det_logits_2d)

        # Shape is already [num_tokens, vocab_size]
        num_tokens = det_logits.shape[0]
        vocab_size = det_logits.shape[1]
        print(f"DET logits shape: {det_logits.shape}")

        # Get last token's logits
        det_last_logits = det_logits[-1, :]

        top_k = 10
        top_indices = np.argsort(det_last_logits)[-top_k:][::-1]
        print(f"\nDET Top {top_k} predictions for next token:")
        for i, idx in enumerate(top_indices):
            if HAS_HF:
                token_text = hf_tokenizer.decode([idx])
            else:
                token_text = f"token_{idx}"
            print(f"  {i+1}. Token {idx} ({repr(token_text)}): logit={det_last_logits[idx]:.4f}")

        # Check Paris token
        paris_token = 12095
        paris_rank = np.sum(det_last_logits >= det_last_logits[paris_token])
        print(f"\n' Paris' (token {paris_token}) rank: {paris_rank}, logit: {det_last_logits[paris_token]:.4f}")

        # Compare logits
        if hf_activations:
            print("\n" + "-" * 70)
            print("Comparing DET vs HuggingFace:")
            print()

            hf_logits = hf_activations['logits'][0]  # [seq_len, vocab]

            # Compare last token's logits
            hf_last = hf_logits[-1, :]
            det_last = det_last_logits

            print(f"Logits shape - DET: {det_logits.shape}, HF: {hf_logits.shape}")

            # Overall comparison
            rms = rms_diff(det_last, hf_last)
            maxd = max_diff(det_last, hf_last)
            print(f"Final logits: RMS_diff={rms:.4f}, Max_diff={maxd:.4f}")

            # Find largest differences
            diff = np.abs(det_last - hf_last)
            top_diff_idx = np.argsort(diff)[-5:][::-1]
            print("\nLargest logit differences:")
            for idx in top_diff_idx:
                if HAS_HF:
                    token_text = hf_tokenizer.decode([idx])
                else:
                    token_text = f"token_{idx}"
                print(f"  Token {idx} ({repr(token_text)}): "
                      f"DET={det_last[idx]:.4f}, HF={hf_last[idx]:.4f}, "
                      f"diff={diff[idx]:.4f}")

            # Compare embedding layer outputs
            print("\n" + "-" * 70)
            print("Per-token hidden state RMS progression through HF layers:")

            for key in ['embedding'] + [f'layer_{i}_output' for i in range(24)]:
                if key in hf_activations:
                    val = hf_activations[key][0]  # Remove batch dim
                    if key == 'embedding':
                        print(f"  {key}: rms={rms_val(val[-1]):.6f}")
                    else:
                        layer_num = int(key.split('_')[1])
                        print(f"  layer_{layer_num:2d}: rms={rms_val(val[-1]):.6f}")

    print("\n" + "=" * 70)
    print("Debug complete.")
    print("=" * 70)

if __name__ == "__main__":
    main()
