"""
DET Inference - Python Bindings
================================

Python ctypes bindings for the DET native inference library.
Provides model loading, tokenization, and inference capabilities.

Phase 26.3: Bridges C primitives to Python/Existence-Lang.
"""

import os
import ctypes
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass


# =============================================================================
# CTYPES STRUCTURES
# =============================================================================

class DetTensor(ctypes.Structure):
    """Matches DetTensor in det_tensor.h"""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("data_size", ctypes.c_size_t),
        ("ndim", ctypes.c_int32),
        ("shape", ctypes.c_int32 * 4),
        ("stride", ctypes.c_int32 * 4),
        ("dtype", ctypes.c_int),
        ("storage", ctypes.c_int),
        ("owns_data", ctypes.c_bool),
        ("refcount", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("zero_point", ctypes.c_int32),
        ("fd", ctypes.c_int),
        ("file_offset", ctypes.c_size_t),
    ]


class DetModelConfig(ctypes.Structure):
    """Matches DetModelConfig in det_model.h"""
    _fields_ = [
        ("n_vocab", ctypes.c_int32),
        ("n_ctx", ctypes.c_int32),
        ("n_embd", ctypes.c_int32),
        ("n_head", ctypes.c_int32),
        ("n_head_kv", ctypes.c_int32),
        ("n_layer", ctypes.c_int32),
        ("n_ff", ctypes.c_int32),
        ("n_rot", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("norm_eps", ctypes.c_float),
        ("arch", ctypes.c_int),
    ]


class DetTokenStats(ctypes.Structure):
    """Matches DetTokenStats in det_model.h (Phase 26.6)"""
    _fields_ = [
        ("entropy", ctypes.c_float),       # Logit distribution entropy (after temperature)
        ("entropy_raw", ctypes.c_float),   # Raw entropy (before temperature)
        ("k_eff", ctypes.c_int32),         # Effective candidates (nucleus set size)
        ("top_prob", ctypes.c_float),      # Probability of selected token
        ("top5_mass", ctypes.c_float),     # Total probability mass in top 5 tokens
        ("token_id", ctypes.c_int32),      # The selected token
    ]


# =============================================================================
# LIBRARY LOADING
# =============================================================================

def _find_library() -> Optional[Path]:
    """Find the det_inference library."""
    # Get the base directory (local_agency/)
    # __file__ is in src/python/det/inference.py
    # Go up: det -> python -> src -> local_agency
    base_dir = Path(__file__).parent.parent.parent.parent

    # Check common locations
    locations = [
        # Build directory (src/inference/build/)
        base_dir / "src" / "inference" / "build" / "libdet_inference.dylib",
        base_dir / "src" / "inference" / "build" / "libdet_inference.so",
        # Alternative: inference at root level
        base_dir / "inference" / "build" / "libdet_inference.dylib",
        base_dir / "inference" / "build" / "libdet_inference.so",
        # Installed location
        Path("/usr/local/lib/libdet_inference.dylib"),
        Path("/usr/local/lib/libdet_inference.so"),
        # Relative to working directory
        Path("libdet_inference.dylib"),
        Path("libdet_inference.so"),
    ]

    for loc in locations:
        if loc.exists():
            return loc

    return None


_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    """Get or load the inference library."""
    global _lib
    if _lib is not None:
        return _lib

    lib_path = _find_library()
    if lib_path is None:
        raise RuntimeError("Could not find libdet_inference. Build it first with: cd src/inference/build && cmake .. && make")

    _lib = ctypes.CDLL(str(lib_path))
    _setup_bindings(_lib)
    return _lib


def _setup_bindings(lib: ctypes.CDLL):
    """Set up function signatures for ctypes."""

    # det_model_load
    lib.det_model_load.argtypes = [ctypes.c_char_p]
    lib.det_model_load.restype = ctypes.c_void_p

    # det_model_free
    lib.det_model_free.argtypes = [ctypes.c_void_p]
    lib.det_model_free.restype = None

    # det_model_reset
    lib.det_model_reset.argtypes = [ctypes.c_void_p]
    lib.det_model_reset.restype = None

    # det_model_get_tokenizer
    lib.det_model_get_tokenizer.argtypes = [ctypes.c_void_p]
    lib.det_model_get_tokenizer.restype = ctypes.c_void_p

    # det_model_forward
    lib.det_model_forward.argtypes = [
        ctypes.c_void_p,  # model
        ctypes.POINTER(ctypes.c_int32),  # tokens
        ctypes.c_int32,  # num_tokens
        ctypes.c_void_p,  # logits (can be NULL)
    ]
    lib.det_model_forward.restype = ctypes.POINTER(DetTensor)

    # det_model_sample
    lib.det_model_sample.argtypes = [
        ctypes.c_void_p,  # model
        ctypes.POINTER(DetTensor),  # logits
        ctypes.c_float,  # temperature
        ctypes.c_float,  # top_p
        ctypes.c_int32,  # top_k
    ]
    lib.det_model_sample.restype = ctypes.c_int32

    # det_choose_token
    lib.det_choose_token.argtypes = [
        ctypes.c_void_p,  # model
        ctypes.POINTER(ctypes.c_float),  # logits
        ctypes.c_int32,  # vocab_size
        ctypes.c_float,  # temperature
        ctypes.c_float,  # top_p
        ctypes.POINTER(ctypes.c_float),  # det_presence (can be NULL)
        ctypes.c_uint64,  # seed
    ]
    lib.det_choose_token.restype = ctypes.c_int32

    # det_tokenize
    lib.det_tokenize.argtypes = [
        ctypes.c_void_p,  # tokenizer
        ctypes.c_char_p,  # text
        ctypes.POINTER(ctypes.c_int32),  # tokens
        ctypes.c_int32,  # max_tokens
    ]
    lib.det_tokenize.restype = ctypes.c_int32

    # det_detokenize
    lib.det_detokenize.argtypes = [
        ctypes.c_void_p,  # tokenizer
        ctypes.POINTER(ctypes.c_int32),  # tokens
        ctypes.c_int32,  # num_tokens
        ctypes.c_char_p,  # text
        ctypes.c_int32,  # max_len
    ]
    lib.det_detokenize.restype = ctypes.c_int32

    # det_token_to_text
    lib.det_token_to_text.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.det_token_to_text.restype = ctypes.c_char_p

    # det_token_to_text_decoded (with BPE decoding for streaming)
    lib.det_token_to_text_decoded.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.det_token_to_text_decoded.restype = ctypes.c_char_p

    # det_bos_token_export (non-inline wrapper for ctypes)
    lib.det_bos_token_export.argtypes = [ctypes.c_void_p]
    lib.det_bos_token_export.restype = ctypes.c_int32

    # det_eos_token_export (non-inline wrapper for ctypes)
    lib.det_eos_token_export.argtypes = [ctypes.c_void_p]
    lib.det_eos_token_export.restype = ctypes.c_int32

    # det_vocab_size_export (non-inline wrapper for ctypes)
    lib.det_vocab_size_export.argtypes = [ctypes.c_void_p]
    lib.det_vocab_size_export.restype = ctypes.c_int32

    # det_model_info
    lib.det_model_info.argtypes = [ctypes.c_void_p]
    lib.det_model_info.restype = ctypes.c_char_p

    # Inference mode (QAM support)
    lib.det_set_inference_mode.argtypes = [ctypes.c_int]
    lib.det_set_inference_mode.restype = None

    lib.det_get_inference_mode.argtypes = []
    lib.det_get_inference_mode.restype = ctypes.c_int

    # Per-token stats API (Phase 26.6)
    lib.det_stats_start.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.det_stats_start.restype = None

    lib.det_stats_get.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
    lib.det_stats_get.restype = ctypes.POINTER(DetTokenStats)

    lib.det_stats_aggregate.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),  # mean_entropy
        ctypes.POINTER(ctypes.c_float),  # mean_k_eff
        ctypes.POINTER(ctypes.c_float),  # min_entropy
    ]
    lib.det_stats_aggregate.restype = None

    lib.det_stats_clear.argtypes = [ctypes.c_void_p]
    lib.det_stats_clear.restype = None

    # KV cache management (Phase 26.4)
    lib.det_kv_cache_position.argtypes = [ctypes.c_void_p]
    lib.det_kv_cache_position.restype = ctypes.c_int32

    lib.det_kv_cache_capacity.argtypes = [ctypes.c_void_p]
    lib.det_kv_cache_capacity.restype = ctypes.c_int32

    lib.det_kv_cache_slice.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.det_kv_cache_slice.restype = ctypes.c_int

    lib.det_kv_cache_shift.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.det_kv_cache_shift.restype = ctypes.c_int

    # Metal backend
    try:
        lib.tensor_metal_available.argtypes = []
        lib.tensor_metal_available.restype = ctypes.c_int

        lib.tensor_metal_device_name.argtypes = []
        lib.tensor_metal_device_name.restype = ctypes.c_char_p

        lib.tensor_metal_init.argtypes = []
        lib.tensor_metal_init.restype = ctypes.c_int
    except AttributeError:
        pass  # Metal not available


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

@dataclass
class SamplingParams:
    """Sampling parameters for token generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.1
    seed: int = 0


# Inference mode constants (match C enum)
INFERENCE_MODE_F32 = 0   # Dequantize all weights at load time
INFERENCE_MODE_Q8_0 = 1  # Keep Q8_0 weights quantized, dequant on-the-fly (QAM)


def set_inference_mode(mode: int):
    """Set inference mode before loading model.

    Args:
        mode: INFERENCE_MODE_F32 (0) or INFERENCE_MODE_Q8_0 (1)

    Must be called BEFORE loading a model to take effect.
    Q8_0 mode uses ~4x less memory but dequantizes during matmul.
    """
    lib = _get_lib()
    lib.det_set_inference_mode(mode)


def get_inference_mode() -> int:
    """Get current inference mode.

    Returns:
        INFERENCE_MODE_F32 (0) or INFERENCE_MODE_Q8_0 (1)
    """
    lib = _get_lib()
    return lib.det_get_inference_mode()


class Model:
    """
    DET-native LLM model.

    Loads GGUF models and provides inference capabilities.
    Integrates with DET physics for sampling.
    """

    def __init__(self, path: str):
        """Load model from GGUF file."""
        self._lib = _get_lib()
        self._handle = self._lib.det_model_load(path.encode('utf-8'))
        if not self._handle:
            raise RuntimeError(f"Failed to load model: {path}")

        # Get tokenizer pointer from model
        self._tokenizer = self._lib.det_model_get_tokenizer(self._handle)
        if not self._tokenizer:
            raise RuntimeError("Failed to get tokenizer from model")

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            self._lib.det_model_free(self._handle)

    @property
    def info(self) -> str:
        """Get model info string."""
        result = self._lib.det_model_info(self._handle)
        return result.decode('utf-8') if result else "Unknown"

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._lib.det_vocab_size_export(self._tokenizer)

    @property
    def bos_token(self) -> int:
        """Get BOS token ID."""
        return self._lib.det_bos_token_export(self._tokenizer)

    @property
    def eos_token(self) -> int:
        """Get EOS token ID."""
        return self._lib.det_eos_token_export(self._tokenizer)

    def reset(self):
        """Reset KV cache for new conversation."""
        self._lib.det_model_reset(self._handle)

    # =========================================================================
    # KV CACHE MANAGEMENT (Phase 26.4)
    # =========================================================================

    @property
    def cache_position(self) -> int:
        """Get current KV cache position (tokens in cache)."""
        return self._lib.det_kv_cache_position(self._handle)

    @property
    def cache_capacity(self) -> int:
        """Get KV cache capacity (max context length)."""
        return self._lib.det_kv_cache_capacity(self._handle)

    def cache_slice(self, start: int, end: int) -> bool:
        """
        Slice KV cache to keep only positions [start, end).

        Useful for sliding window attention or context truncation.
        After slice, cache_position becomes (end - start).

        Args:
            start: First position to keep (0-indexed)
            end: One past last position to keep

        Returns:
            True on success, False on error
        """
        return self._lib.det_kv_cache_slice(self._handle, start, end) == 0

    def cache_shift(self, keep_last: int) -> bool:
        """
        Shift KV cache to keep only last N tokens.

        Equivalent to: cache_slice(cache_position - keep_last, cache_position)

        Args:
            keep_last: Number of recent tokens to keep

        Returns:
            True on success, False on error
        """
        return self._lib.det_kv_cache_shift(self._handle, keep_last) == 0

    def cache_info(self) -> dict:
        """Get KV cache info as a dict."""
        pos = self.cache_position
        cap = self.cache_capacity
        return {
            'position': pos,
            'capacity': cap,
            'usage': pos / cap if cap > 0 else 0.0,
            'remaining': cap - pos,
        }

    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        max_tokens = len(text) * 4 + 16  # Estimate
        tokens = (ctypes.c_int32 * max_tokens)()

        n = self._lib.det_tokenize(
            self._tokenizer,
            text.encode('utf-8'),
            tokens,
            max_tokens
        )

        if n < 0:
            raise RuntimeError(f"Tokenization failed: {n}")

        return list(tokens[:n])

    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs to text."""
        token_array = (ctypes.c_int32 * len(tokens))(*tokens)
        max_len = len(tokens) * 16 + 16
        text = ctypes.create_string_buffer(max_len)

        n = self._lib.det_detokenize(
            self._tokenizer,
            token_array,
            len(tokens),
            text,
            max_len
        )

        if n < 0:
            raise RuntimeError(f"Detokenization failed: {n}")

        return text.value.decode('utf-8')

    def token_to_text(self, token_id: int) -> str:
        """Get text for a single token (with BPE decoding for streaming)."""
        result = self._lib.det_token_to_text_decoded(self._tokenizer, token_id)
        return result.decode('utf-8') if result else ""

    def forward(self, tokens: List[int]) -> 'ctypes.POINTER(DetTensor)':
        """Run forward pass, return logits tensor pointer."""
        token_array = (ctypes.c_int32 * len(tokens))(*tokens)

        logits = self._lib.det_model_forward(
            self._handle,
            token_array,
            len(tokens),
            None  # Use internal buffer
        )

        if not logits:
            raise RuntimeError("Forward pass failed")

        return logits

    def forward_logits(self, tokens: List[int]) -> List[float]:
        """Run forward pass, return logits as a flat Python list."""
        logits_tensor = self.forward(tokens)
        tensor = logits_tensor.contents

        # Get shape: [num_tokens, vocab_size]
        num_tokens = tensor.shape[0]
        vocab_size = tensor.shape[1]
        total_floats = num_tokens * vocab_size

        # Cast data pointer to float array
        float_ptr = ctypes.cast(tensor.data, ctypes.POINTER(ctypes.c_float))

        # Extract as list
        return [float_ptr[i] for i in range(total_floats)]

    def forward_logits_2d(self, tokens: List[int]) -> List[List[float]]:
        """Run forward pass, return logits as 2D list [num_tokens][vocab_size]."""
        logits_tensor = self.forward(tokens)
        tensor = logits_tensor.contents

        num_tokens = tensor.shape[0]
        vocab_size = tensor.shape[1]

        float_ptr = ctypes.cast(tensor.data, ctypes.POINTER(ctypes.c_float))

        result = []
        for t in range(num_tokens):
            offset = t * vocab_size
            row = [float_ptr[offset + i] for i in range(vocab_size)]
            result.append(row)
        return result

    def sample(self, logits, params: SamplingParams = None) -> int:
        """Sample next token from logits."""
        if params is None:
            params = SamplingParams()

        return self._lib.det_model_sample(
            self._handle,
            logits,
            params.temperature,
            params.top_p,
            params.top_k
        )

    def choose_token(self, logits: List[float], temperature: float = 0.7,
                     top_p: float = 0.9, det_presence: List[float] = None,
                     seed: int = 0) -> int:
        """
        DET-aware token selection.

        This is the sacred integration point where DET physics
        can influence token selection via presence values.

        Args:
            logits: Unnormalized log-probabilities [vocab_size]
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            det_presence: Optional DET presence bias [vocab_size]
            seed: Random seed (0 = use global state)

        Returns:
            Selected token ID
        """
        vocab_size = len(logits)
        logits_array = (ctypes.c_float * vocab_size)(*logits)

        presence_ptr = None
        if det_presence is not None:
            presence_array = (ctypes.c_float * vocab_size)(*det_presence)
            presence_ptr = presence_array

        return self._lib.det_choose_token(
            self._handle,
            logits_array,
            vocab_size,
            temperature,
            top_p,
            presence_ptr,
            seed
        )

    def generate(self, prompt: str, max_tokens: int = 256,
                 params: SamplingParams = None,
                 callback: Callable[[str, int], None] = None,
                 det_state: dict = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            params: Sampling parameters
            callback: Called for each token (text, token_id)
            det_state: Optional DET creature state for truthfulness evaluation

        Returns:
            Generated text
        """
        if params is None:
            params = SamplingParams()

        # Tokenize prompt
        tokens = self.tokenize(prompt)

        # Forward pass on prompt
        logits = self.forward(tokens)

        # Generate tokens
        generated = []
        for _ in range(max_tokens):
            token = self.sample(logits, params)

            if token == self.eos_token or token < 0:
                break

            generated.append(token)

            # Callback for streaming
            if callback:
                text = self.token_to_text(token)
                callback(text, token)

            # Forward next token
            logits = self.forward([token])

        return self.detokenize(generated)

    def generate_with_truthfulness(self, prompt: str, max_tokens: int = 256,
                                    params: SamplingParams = None,
                                    callback: Callable[[str, int], None] = None,
                                    det_state: dict = None,
                                    bond_state: dict = None) -> tuple:
        """
        Generate text with DET-rigorous truthfulness evaluation.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            params: Sampling parameters
            callback: Called for each token (text, token_id)
            det_state: DET creature state dict with 'F', 'a', 'q'
            bond_state: Optional bond state dict with 'coherence' or 'C'

        Returns:
            Tuple of (generated_text, TruthfulnessScore)
        """
        if params is None:
            params = SamplingParams()

        # Get evaluator and reset for new generation
        evaluator = get_truthfulness_evaluator()
        evaluator.reset_generation()

        # Set up grounding from bond state
        if bond_state:
            c_user = bond_state.get('coherence', bond_state.get('C', 1.0))
            evaluator.set_grounding_signals(c_user=c_user)

        # Tokenize prompt
        tokens = self.tokenize(prompt)

        # Forward pass on prompt
        logits = self.forward(tokens)

        # Track F expenditure for grounding
        initial_f = det_state.get('F', 100.0) if det_state else 100.0

        # Generate tokens
        generated = []
        for _ in range(max_tokens):
            token = self.sample(logits, params)

            if token == self.eos_token or token < 0:
                break

            generated.append(token)

            # Record claim with F cost estimate
            # (In full implementation, this would track actual F spent)
            evaluator.record_claim(f_cost=0.1)

            # Callback for streaming
            if callback:
                text = self.token_to_text(token)
                callback(text, token)

            # Forward next token
            logits = self.forward([token])

        text = self.detokenize(generated)

        # Update grounding with F expenditure
        final_f = det_state.get('F', 100.0) if det_state else 100.0
        delta_f = max(0.0, initial_f - final_f)
        evaluator.set_grounding_signals(delta_f=delta_f)

        # Compute truthfulness score
        # k_eff = top_k from params (effective candidates)
        k_eff = params.top_k if params.top_k > 0 else 100

        if det_state:
            score = evaluator.evaluate_from_det_state(
                creature_state=det_state,
                bond_state=bond_state,
                k_eff=k_eff,
                num_tokens=len(generated)
            )
        else:
            score = evaluator.evaluate(
                k_eff=k_eff,
                num_tokens=len(generated)
            )

        return text, score

    # =========================================================================
    # PER-TOKEN STATS API (Phase 26.6)
    # =========================================================================

    def stats_start(self, capacity: int = 1024):
        """
        Start collecting per-token stats for truthfulness evaluation.

        Must be called before generation to track real entropy and k_eff.

        Args:
            capacity: Maximum tokens to track (default 1024)
        """
        self._lib.det_stats_start(self._handle, capacity)

    def stats_clear(self):
        """Clear stats buffer for new generation."""
        self._lib.det_stats_clear(self._handle)

    def stats_get(self) -> List[dict]:
        """
        Get per-token stats from the last generation.

        Returns:
            List of dicts with 'entropy', 'entropy_raw', 'k_eff',
            'top_prob', 'top5_mass', 'token_id' for each token.
        """
        count = ctypes.c_int32()
        stats_ptr = self._lib.det_stats_get(self._handle, ctypes.byref(count))

        if not stats_ptr or count.value == 0:
            return []

        result = []
        for i in range(count.value):
            stat = stats_ptr[i]
            result.append({
                'entropy': stat.entropy,
                'entropy_raw': stat.entropy_raw,
                'k_eff': stat.k_eff,
                'top_prob': stat.top_prob,
                'top5_mass': stat.top5_mass,
                'token_id': stat.token_id,
            })
        return result

    def stats_aggregate(self) -> dict:
        """
        Get aggregated stats for the last generation.

        Returns:
            Dict with 'mean_entropy', 'mean_k_eff', 'min_entropy'
        """
        mean_entropy = ctypes.c_float()
        mean_k_eff = ctypes.c_float()
        min_entropy = ctypes.c_float()

        self._lib.det_stats_aggregate(
            self._handle,
            ctypes.byref(mean_entropy),
            ctypes.byref(mean_k_eff),
            ctypes.byref(min_entropy)
        )

        return {
            'mean_entropy': mean_entropy.value,
            'mean_k_eff': mean_k_eff.value,
            'min_entropy': min_entropy.value,
        }


# =============================================================================
# METAL GPU
# =============================================================================

def metal_available() -> bool:
    """Check if Metal GPU is available."""
    try:
        lib = _get_lib()
        return lib.tensor_metal_available() != 0
    except (RuntimeError, AttributeError):
        return False


def metal_device_name() -> str:
    """Get Metal device name."""
    try:
        lib = _get_lib()
        result = lib.tensor_metal_device_name()
        return result.decode('utf-8') if result else "Unknown"
    except (RuntimeError, AttributeError):
        return "Not available"


def metal_init() -> bool:
    """Initialize Metal backend."""
    try:
        lib = _get_lib()
        return lib.tensor_metal_init() == 0
    except (RuntimeError, AttributeError):
        return False


def metal_status() -> dict:
    """Get Metal GPU status as a dict.

    Returns:
        dict with 'available', 'device', and 'initialized' keys
    """
    available = metal_available()
    initialized = metal_init() if available else False
    # Get device name after init (needed for g_metal_ctx to be set)
    device = metal_device_name() if initialized else 'None'
    return {
        'available': available,
        'device': device,
        'initialized': initialized,
    }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_model: Optional[Model] = None


def load_model(path: str) -> Model:
    """Load a model from GGUF file."""
    global _global_model
    _global_model = Model(path)
    return _global_model


def generate(prompt: str, max_tokens: int = 256, **kwargs) -> str:
    """Generate text using the global model."""
    if _global_model is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    return _global_model.generate(prompt, max_tokens, **kwargs)


def tokenize(text: str) -> List[int]:
    """Tokenize text using the global model."""
    if _global_model is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    return _global_model.tokenize(text)


def detokenize(tokens: List[int]) -> str:
    """Detokenize tokens using the global model."""
    if _global_model is None:
        raise RuntimeError("No model loaded. Call load_model() first.")
    return _global_model.detokenize(tokens)


# =============================================================================
# CHAT TEMPLATES
# =============================================================================

class ChatTemplate:
    """Base class for chat templates."""

    def format_prompt(self, user_message: str, system_message: str = None) -> str:
        """Format a single user message with optional system message."""
        raise NotImplementedError

    def format_conversation(self, messages: List[dict]) -> str:
        """Format a list of messages [{'role': 'user'|'assistant'|'system', 'content': str}]."""
        raise NotImplementedError

    @property
    def stop_tokens(self) -> List[str]:
        """Return list of stop token strings."""
        return []


class QwenChatTemplate(ChatTemplate):
    """
    ChatML template for Qwen2.5-Instruct models.

    Format:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {user}<|im_end|>
        <|im_start|>assistant
    """

    DEFAULT_SYSTEM = "You are a helpful assistant."

    def format_prompt(self, user_message: str, system_message: str = None) -> str:
        """Format a single user message with optional system message."""
        if system_message is None:
            system_message = self.DEFAULT_SYSTEM

        return (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def format_conversation(self, messages: List[dict]) -> str:
        """Format a list of messages."""
        result = []

        # Add system message if first message is system, otherwise use default
        if messages and messages[0].get('role') == 'system':
            system_content = messages[0].get('content', self.DEFAULT_SYSTEM)
            result.append(f"<|im_start|>system\n{system_content}<|im_end|>")
            messages = messages[1:]
        else:
            result.append(f"<|im_start|>system\n{self.DEFAULT_SYSTEM}<|im_end|>")

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')

            if role == 'system':
                # Skip additional system messages (already handled)
                continue
            elif role == 'user':
                result.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == 'assistant':
                result.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Add assistant prompt for generation
        result.append("<|im_start|>assistant\n")

        return "\n".join(result)

    @property
    def stop_tokens(self) -> List[str]:
        return ["<|im_end|>", "<|endoftext|>"]


class LlamaChatTemplate(ChatTemplate):
    """
    Template for Llama 2/3 chat models.

    Format (simplified):
        <s>[INST] <<SYS>>
        {system}
        <</SYS>>

        {user} [/INST]
    """

    DEFAULT_SYSTEM = "You are a helpful assistant."

    def format_prompt(self, user_message: str, system_message: str = None) -> str:
        if system_message is None:
            system_message = self.DEFAULT_SYSTEM

        return (
            f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n"
            f"{user_message} [/INST]"
        )

    def format_conversation(self, messages: List[dict]) -> str:
        # Simplified implementation
        return self.format_prompt(
            messages[-1].get('content', '') if messages else '',
            messages[0].get('content') if messages and messages[0].get('role') == 'system' else None
        )

    @property
    def stop_tokens(self) -> List[str]:
        return ["</s>"]


# Template registry
CHAT_TEMPLATES = {
    'qwen': QwenChatTemplate(),
    'qwen2': QwenChatTemplate(),
    'qwen2.5': QwenChatTemplate(),
    'chatml': QwenChatTemplate(),  # ChatML is the same format
    'llama': LlamaChatTemplate(),
    'llama2': LlamaChatTemplate(),
    'llama3': LlamaChatTemplate(),
}


def get_chat_template(model_name: str) -> ChatTemplate:
    """Get chat template for a model based on its name."""
    model_lower = model_name.lower()

    # Check for known patterns
    if 'qwen' in model_lower:
        return CHAT_TEMPLATES['qwen']
    elif 'llama' in model_lower:
        return CHAT_TEMPLATES['llama']

    # Default to Qwen/ChatML format (common)
    return CHAT_TEMPLATES['chatml']


def detect_template_from_vocab(model: 'Model') -> ChatTemplate:
    """Auto-detect chat template by checking vocabulary for special tokens."""
    try:
        # Check if model has ChatML tokens (Qwen-style)
        # Try to tokenize the special tokens
        im_start_tokens = model.tokenize("<|im_start|>")
        if len(im_start_tokens) == 1:  # Single token = recognized special token
            return CHAT_TEMPLATES['qwen']
    except Exception:
        pass

    # Default to ChatML
    return CHAT_TEMPLATES['chatml']


# =============================================================================
# TRUTHFULNESS WEIGHTING (Phase 26.6) - DET-Rigorous Implementation
# =============================================================================

import math


@dataclass
class ClaimLedger:
    """
    Per-generation epistemic debt tracker.

    Separates q_claim (epistemic debt from ungrounded assertions)
    from q_creature (existential structural debt).

    DET principle: Debt must be earned by lawful updates, not injected.
    """

    # Accumulated claim cost (F spent on assertions)
    total_claim_cost: float = 0.0

    # Number of claims made (assertions beyond prompt context)
    num_claims: int = 0

    # Unpaid claims (claims without sufficient F expenditure)
    unpaid_claims: int = 0

    # Derived epistemic debt: unpaid_claims / (total_claims + 1)
    @property
    def q_claim(self) -> float:
        """Epistemic debt from ungrounded assertions."""
        if self.num_claims == 0:
            return 0.0
        return self.unpaid_claims / (self.num_claims + 1.0)

    def record_claim(self, f_cost: float, min_cost_threshold: float = 0.1):
        """
        Record a claim (assertion beyond prompt context).

        Args:
            f_cost: F expenditure for this claim
            min_cost_threshold: Minimum F required for a "paid" claim
        """
        self.num_claims += 1
        self.total_claim_cost += f_cost
        if f_cost < min_cost_threshold:
            self.unpaid_claims += 1

    def reset(self):
        """Reset ledger for new generation."""
        self.total_claim_cost = 0.0
        self.num_claims = 0
        self.unpaid_claims = 0


@dataclass
class GroundingSignals:
    """
    DET-native grounding signals (local, auditable, no external oracle).

    These provide evidence for truthfulness without circular injection.
    """

    # Commit cost / resource burn (ΔF_claim)
    # Higher cost = more "paid for" = better grounded
    delta_f_claim: float = 0.0

    # Trace stability: does output remain stable under perturbation?
    # 1.0 = fully stable, 0.0 = unstable (changes with paraphrase/re-ask)
    trace_stability: float = 1.0

    # Bond coherence with user specifically (not generic coherence)
    # Measures alignment with user's context, constraints, commitments
    c_user: float = 1.0

    # Constraint violations detected (user-stated constraints violated)
    constraint_violations: int = 0

    @property
    def grounding_factor(self) -> float:
        """
        Compute composite grounding factor G in [0, 1].

        G gates agency's contribution to truthfulness.
        High agency without grounding = "persuasive capability" not truth.
        """
        # Cost factor: sigmoid of claim cost (0.5 at cost=1.0)
        cost_factor = 1.0 / (1.0 + math.exp(-self.delta_f_claim + 1.0))

        # Stability factor: direct
        stability_factor = self.trace_stability

        # Coherence factor: penalize constraint violations
        coherence_factor = self.c_user * (1.0 / (1.0 + self.constraint_violations))

        # Geometric mean to require all factors
        g = (cost_factor * stability_factor * coherence_factor) ** (1.0 / 3.0)
        return max(0.0, min(1.0, g))


@dataclass
class TruthfulnessScore:
    """
    DET-rigorous truthfulness score for generated output.

    Key principles:
    - q_claim (epistemic debt) is earned, not injected
    - Agency amplifies truth only when coupled to grounding
    - Entropy normalized locally by K_eff, not global constant
    - Coherence is with user bond specifically

    Formula (DET-aligned):
    T_ground = f(paid_claims, trace_stability, C_user)
    T_consist = 1 - H_norm  (where H_norm = H / log(K_eff + ε))
    T = clip(w_g*T_ground + w_a*a*G + w_e*T_consist + w_c*C_user, 0, 1)

    Where G = grounding_factor gates agency contribution.
    """

    # Overall truthfulness score [0, 1]
    total: float

    # Component scores
    grounding_component: float   # From paid claims, stability, user coherence
    agency_component: float      # Agency * Grounding (gated, not direct)
    consistency_component: float # From entropy (locally normalized)
    coherence_component: float   # User-specific coherence

    # Raw values (for debugging/calibration)
    q_claim: float           # Epistemic debt (earned from generation)
    q_creature: float        # Structural debt (from creature state, info only)
    agency: float            # a value
    entropy: float           # H value (logit distribution entropy)
    entropy_normalized: float  # H / log(K_eff + ε)
    k_eff: int               # Effective candidates in truncated distribution
    coherence_user: float    # C_user (user-specific bond coherence)
    grounding_factor: float  # G = composite grounding

    # Metadata
    num_tokens: int
    confidence_level: str  # 'high', 'medium', 'low', 'very_low'

    # Falsifier flags (for calibration/debugging)
    falsifier_flags: dict = None

    def __repr__(self):
        return (f"TruthfulnessScore(T={self.total:.3f}, "
                f"confidence={self.confidence_level}, "
                f"q_claim={self.q_claim:.3f}, G={self.grounding_factor:.3f})")


@dataclass
class TruthfulnessWeights:
    """Weights for truthfulness components (DET-aligned)."""
    w_grounding: float = 0.35   # Weight for grounding component
    w_agency: float = 0.20      # Weight for agency (gated by G)
    w_consistency: float = 0.25 # Weight for consistency (entropy)
    w_coherence: float = 0.20   # Weight for user coherence

    def normalize(self):
        """Ensure weights sum to 1.0."""
        total = self.w_grounding + self.w_agency + self.w_consistency + self.w_coherence
        if total > 0:
            self.w_grounding /= total
            self.w_agency /= total
            self.w_consistency /= total
            self.w_coherence /= total


class TruthfulnessEvaluator:
    """
    DET-rigorous truthfulness evaluator.

    Key differences from naive implementation:
    1. q_claim is earned from generation, not passed in
    2. Agency contribution is gated by grounding factor G
    3. Entropy normalized locally by K_eff (truncated distribution size)
    4. Coherence is user-specific (C_user), not generic

    Anti-hallucination mechanisms:
    - Reward hacking: Claims require F expenditure (tracked in ledger)
    - False confidence: Agency gated by grounding (high a + low G = low truth)
    - Ungrounded claims: q_claim accumulates from unpaid assertions
    - Post-hoc justification: Trace stability detects retroactive changes

    Falsifier targets (DET-style):
    - F_T1: Can T be raised without grounding evidence? (Should fail)
    - F_T2: High T when entropy low but stability low? (Should fail)
    - F_T3: High C_user + wrong facts = high T? (Should fail)
    - F_T4: T depends on global aggregates? (Should fail - must be local)
    """

    CONFIDENCE_THRESHOLDS = {
        'high': 0.75,
        'medium': 0.50,
        'low': 0.25,
    }

    # Small epsilon for log normalization
    EPSILON = 1e-10

    def __init__(self, weights: TruthfulnessWeights = None):
        """Initialize evaluator with optional custom weights."""
        self.weights = weights or TruthfulnessWeights()
        self.weights.normalize()

        # Per-generation state
        self._current_ledger = ClaimLedger()
        self._current_grounding = GroundingSignals()

        # Calibration data
        self._calibration_samples: List[dict] = []

    def reset_generation(self):
        """Reset per-generation state for new generation."""
        self._current_ledger.reset()
        self._current_grounding = GroundingSignals()

    def record_claim(self, f_cost: float, min_cost: float = 0.1):
        """Record a claim during generation."""
        self._current_ledger.record_claim(f_cost, min_cost)

    def set_grounding_signals(self,
                               delta_f: float = None,
                               stability: float = None,
                               c_user: float = None,
                               violations: int = None):
        """Update grounding signals during/after generation."""
        if delta_f is not None:
            self._current_grounding.delta_f_claim = delta_f
        if stability is not None:
            self._current_grounding.trace_stability = stability
        if c_user is not None:
            self._current_grounding.c_user = c_user
        if violations is not None:
            self._current_grounding.constraint_violations = violations

    def normalize_entropy(self, entropy: float, k_eff: int) -> float:
        """
        Normalize entropy locally by K_eff.

        H_norm = H / log(K_eff + ε) in [0, 1]

        This is DET-compliant: local, per-step, no hidden global constant.
        """
        if k_eff <= 0:
            return 0.0
        max_entropy = math.log(k_eff + self.EPSILON)
        if max_entropy <= 0:
            return 0.0
        return min(1.0, entropy / max_entropy)

    def evaluate(self,
                 agency: float = 0.5,
                 entropy: float = 0.0,
                 k_eff: int = 100,
                 q_creature: float = 0.0,
                 num_tokens: int = 0,
                 ledger: ClaimLedger = None,
                 grounding: GroundingSignals = None) -> TruthfulnessScore:
        """
        Compute DET-rigorous truthfulness score.

        Args:
            agency: Agency value a from creature state
            entropy: Logit distribution entropy (after temperature/top-p/top-k)
            k_eff: Effective candidates in truncated distribution
            q_creature: Structural debt from creature (info only, not used in T)
            num_tokens: Number of tokens generated
            ledger: ClaimLedger with earned epistemic debt (or use current)
            grounding: GroundingSignals (or use current)

        Returns:
            TruthfulnessScore with DET-compliant composite score
        """
        w = self.weights

        # Use provided or current per-generation state
        ledger = ledger or self._current_ledger
        grounding = grounding or self._current_grounding

        # Get earned epistemic debt (not injected)
        q_claim = ledger.q_claim

        # Get grounding factor G
        G = grounding.grounding_factor

        # Normalize entropy locally
        H_norm = self.normalize_entropy(entropy, k_eff)

        # Clamp inputs
        agency_clamped = max(0.0, min(1.0, agency))
        c_user = max(0.0, min(1.0, grounding.c_user))

        # Compute components

        # Grounding component: based on paid claims and stability
        # Higher G = better grounded
        grounding_component = w.w_grounding * G

        # Agency component: agency * G (gated by grounding)
        # High agency without grounding = persuasive capability, not truth
        agency_component = w.w_agency * agency_clamped * G

        # Consistency component: 1 - H_norm (lower entropy = more consistent)
        consistency_component = w.w_consistency * (1.0 - H_norm)

        # Coherence component: user-specific coherence
        coherence_component = w.w_coherence * c_user

        # Total score
        total = (grounding_component + agency_component +
                 consistency_component + coherence_component)

        # Apply epistemic debt penalty
        # q_claim reduces total score (unpaid claims hurt truthfulness)
        total = total / (1.0 + q_claim)

        # Clamp to [0, 1]
        total = max(0.0, min(1.0, total))

        # Determine confidence level
        if total >= self.CONFIDENCE_THRESHOLDS['high']:
            confidence_level = 'high'
        elif total >= self.CONFIDENCE_THRESHOLDS['medium']:
            confidence_level = 'medium'
        elif total >= self.CONFIDENCE_THRESHOLDS['low']:
            confidence_level = 'low'
        else:
            confidence_level = 'very_low'

        # Check falsifier conditions
        falsifier_flags = self._check_falsifiers(
            total, G, H_norm, grounding.trace_stability, c_user, agency_clamped
        )

        return TruthfulnessScore(
            total=total,
            grounding_component=grounding_component,
            agency_component=agency_component,
            consistency_component=consistency_component,
            coherence_component=coherence_component,
            q_claim=q_claim,
            q_creature=q_creature,
            agency=agency_clamped,
            entropy=entropy,
            entropy_normalized=H_norm,
            k_eff=k_eff,
            coherence_user=c_user,
            grounding_factor=G,
            num_tokens=num_tokens,
            confidence_level=confidence_level,
            falsifier_flags=falsifier_flags
        )

    def _check_falsifiers(self, total: float, G: float, H_norm: float,
                          stability: float, c_user: float, agency: float) -> dict:
        """
        Check DET falsifier conditions for the truth system.

        Returns dict of falsifier flags (True = violation detected).
        """
        flags = {}

        # F_T1: High T without grounding evidence
        # If G < 0.3 but T > 0.6, something is wrong
        flags['F_T1_reward_hacking'] = (G < 0.3 and total > 0.6)

        # F_T2: High T when entropy low but stability low
        # Overconfidence: low H_norm + low stability + high T
        flags['F_T2_overconfidence'] = (H_norm < 0.3 and stability < 0.5 and total > 0.7)

        # F_T3: High coherence + high T but grounding is low
        # Coherence misuse: high C_user alone shouldn't guarantee truth
        flags['F_T3_coherence_misuse'] = (c_user > 0.8 and G < 0.3 and total > 0.6)

        # F_T4: This is structural (checked elsewhere) - non-local dependence
        # Can't easily check in evaluate(), but flag if agency contributes without G
        flags['F_T4_agency_ungated'] = (agency > 0.7 and G < 0.2 and total > 0.5)

        return flags

    def evaluate_from_det_state(self,
                                 creature_state: dict,
                                 bond_state: dict = None,
                                 logit_entropy: float = None,
                                 k_eff: int = None,
                                 num_tokens: int = 0,
                                 ledger: ClaimLedger = None,
                                 grounding: GroundingSignals = None) -> TruthfulnessScore:
        """
        Evaluate truthfulness from DET creature state.

        Args:
            creature_state: Dict with 'F', 'a', 'q' from creature
            bond_state: Optional dict with 'coherence' for user bond
            logit_entropy: Entropy of logit distribution (local, per-step)
            k_eff: Effective candidates (top-k or nucleus set size)
            num_tokens: Number of tokens generated
            ledger: Optional claim ledger (uses current if None)
            grounding: Optional grounding signals (uses current if None)

        Returns:
            TruthfulnessScore
        """
        agency = creature_state.get('a', 0.5)
        q_creature = creature_state.get('q', 0.0)

        # Default entropy and k_eff if not provided
        entropy = logit_entropy if logit_entropy is not None else 0.0
        k_eff = k_eff if k_eff is not None else 100

        # Update grounding with bond coherence if provided
        grounding = grounding or self._current_grounding
        if bond_state:
            c_user = bond_state.get('coherence', bond_state.get('C', 1.0))
            grounding.c_user = c_user

        return self.evaluate(
            agency=agency,
            entropy=entropy,
            k_eff=k_eff,
            q_creature=q_creature,
            num_tokens=num_tokens,
            ledger=ledger,
            grounding=grounding
        )

    def add_calibration_sample(self,
                                score: TruthfulnessScore,
                                ground_truth_correct: bool,
                                metadata: dict = None):
        """
        Add a calibration sample for future weight tuning.

        Args:
            score: TruthfulnessScore that was computed
            ground_truth_correct: Whether the output was actually correct
            metadata: Optional additional context
        """
        self._calibration_samples.append({
            'score': score,
            'correct': ground_truth_correct,
            'metadata': metadata or {},
            'falsifier_flags': score.falsifier_flags
        })

    def get_calibration_stats(self) -> dict:
        """Get statistics from calibration samples."""
        if not self._calibration_samples:
            return {'num_samples': 0}

        correct = [s for s in self._calibration_samples if s['correct']]
        incorrect = [s for s in self._calibration_samples if not s['correct']]

        # Count falsifier violations
        falsifier_counts = {}
        for sample in self._calibration_samples:
            flags = sample.get('falsifier_flags', {})
            for flag, triggered in flags.items():
                if triggered:
                    falsifier_counts[flag] = falsifier_counts.get(flag, 0) + 1

        return {
            'num_samples': len(self._calibration_samples),
            'num_correct': len(correct),
            'num_incorrect': len(incorrect),
            'avg_score_correct': (
                sum(s['score'].total for s in correct) / len(correct)
                if correct else 0.0
            ),
            'avg_score_incorrect': (
                sum(s['score'].total for s in incorrect) / len(incorrect)
                if incorrect else 0.0
            ),
            'falsifier_violations': falsifier_counts,
        }


# Global evaluator instance
_truthfulness_evaluator: Optional[TruthfulnessEvaluator] = None


def get_truthfulness_evaluator() -> TruthfulnessEvaluator:
    """Get or create the global truthfulness evaluator."""
    global _truthfulness_evaluator
    if _truthfulness_evaluator is None:
        _truthfulness_evaluator = TruthfulnessEvaluator()
    return _truthfulness_evaluator


def evaluate_truthfulness(agency: float = 0.5,
                          entropy: float = 0.0,
                          k_eff: int = 100,
                          q_creature: float = 0.0,
                          num_tokens: int = 0) -> TruthfulnessScore:
    """
    Convenience function to evaluate truthfulness (DET-rigorous).

    Note: For proper DET compliance, use the evaluator directly
    with ClaimLedger and GroundingSignals rather than this simplified API.
    """
    return get_truthfulness_evaluator().evaluate(
        agency=agency, entropy=entropy, k_eff=k_eff,
        q_creature=q_creature, num_tokens=num_tokens
    )
