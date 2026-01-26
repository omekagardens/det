#!/usr/bin/env python3
"""
Compare DET C implementation (GGUF) vs HuggingFace directly.
"""

import sys
import os
import numpy as np
import ctypes

sys.path.insert(0, 'src/python')
os.chdir('/Volumes/AI_DATA/development/det_local_agency/det/local_agency')

from det.inference import Model, DetTensor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_tensor_data(tensor_ptr, shape):
    """Extract numpy array from DetTensor pointer."""
    tensor = tensor_ptr.contents
    total = 1
    for i in range(tensor.ndim):
        total *= tensor.shape[i]

    float_ptr = ctypes.cast(tensor.data, ctypes.POINTER(ctypes.c_float))
    data = np.array([float_ptr[i] for i in range(total)])
    return data.reshape(shape)

def main():
    prompt = "The capital of France is"
    model_path = "models/Qwen2.5-0.5B-Instruct-Q8_0.gguf"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("=" * 70)
    print("DET C Implementation vs HuggingFace Comparison")
    print("=" * 70)
    print(f"Prompt: '{prompt}'")
    print()

    # Load HF model
    print("Loading HuggingFace model...")
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    hf_model.eval()

    # Load DET model
    print("Loading DET model...")
    det_model = Model(model_path)

    # Tokenize
    hf_inputs = hf_tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    hf_tokens = hf_inputs.input_ids[0].tolist()
    print(f"HF tokens: {hf_tokens}")

    det_tokens = det_model.tokenize(prompt)
    # Remove BOS if present
    det_tokens_no_bos = [t for t in det_tokens if t != 151643]
    print(f"DET tokens: {det_tokens_no_bos}")

    if hf_tokens != det_tokens_no_bos:
        print("WARNING: Token mismatch!")

    # Get HF output
    print("\nRunning HF forward pass...")
    with torch.no_grad():
        hf_outputs = hf_model(hf_inputs.input_ids, output_hidden_states=True)
        hf_logits = hf_outputs.logits.numpy()[0]  # [seq, vocab]

    # Get DET output
    print("Running DET forward pass...")
    det_model.reset()
    det_logits_2d = det_model.forward_logits_2d(det_tokens_no_bos)
    det_logits = np.array(det_logits_2d)

    print(f"\nLogits shapes - HF: {hf_logits.shape}, DET: {det_logits.shape}")

    # Compare last token logits
    hf_last = hf_logits[-1]
    det_last = det_logits[-1]

    rms_diff = np.sqrt(np.mean((hf_last - det_last) ** 2))
    max_diff = np.max(np.abs(hf_last - det_last))
    print(f"\nFinal logits comparison:")
    print(f"  RMS diff: {rms_diff:.4f}")
    print(f"  Max diff: {max_diff:.4f}")

    # Find top predictions
    print("\nHF Top 5 predictions:")
    hf_top = np.argsort(hf_last)[-5:][::-1]
    for idx in hf_top:
        token_text = hf_tokenizer.decode([idx])
        print(f"  Token {idx} ({repr(token_text)}): logit={hf_last[idx]:.4f}")

    print("\nDET Top 5 predictions:")
    det_top = np.argsort(det_last)[-5:][::-1]
    for idx in det_top:
        token_text = hf_tokenizer.decode([idx])
        print(f"  Token {idx} ({repr(token_text)}): logit={det_last[idx]:.4f}")

    # Check Paris token
    paris_token = 12095
    hf_paris_rank = np.sum(hf_last >= hf_last[paris_token])
    det_paris_rank = np.sum(det_last >= det_last[paris_token])
    print(f"\n' Paris' token {paris_token}:")
    print(f"  HF:  rank={hf_paris_rank}, logit={hf_last[paris_token]:.4f}")
    print(f"  DET: rank={det_paris_rank}, logit={det_last[paris_token]:.4f}")

    # Compare hidden states if we can access them
    print("\n--- Hidden State Comparison ---")

    # Get HF hidden states at each layer
    hf_hidden = hf_outputs.hidden_states  # tuple of [1, seq, n_embd]

    print("HF layer-by-layer RMS (last token):")
    for i, h in enumerate(hf_hidden):
        h_np = h.numpy()[0, -1]  # Last token
        rms = np.sqrt(np.mean(h_np ** 2))
        print(f"  Layer {i:2d}: RMS={rms:.6f}")

    # What about comparing embedding lookup?
    print("\n--- Embedding Comparison ---")
    hf_embd = hf_hidden[0].numpy()[0]  # [seq, n_embd]
    print(f"HF embedding (token 0)[:5]: {hf_embd[0, :5]}")
    print(f"HF embedding (last token)[:5]: {hf_embd[-1, :5]}")

    # Get DET embedding directly
    # We can't easily get intermediate values from C, but we verified
    # earlier that embedding lookup matches

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()
