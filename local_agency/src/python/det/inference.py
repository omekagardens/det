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


# =============================================================================
# LIBRARY LOADING
# =============================================================================

def _find_library() -> Optional[Path]:
    """Find the det_inference library."""
    # Check common locations
    locations = [
        # Build directory
        Path(__file__).parent.parent.parent.parent / "inference" / "build" / "libdet_inference.dylib",
        Path(__file__).parent.parent.parent.parent / "inference" / "build" / "libdet_inference.so",
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

    # det_bos_token
    lib.det_bos_token.argtypes = [ctypes.c_void_p]
    lib.det_bos_token.restype = ctypes.c_int32

    # det_eos_token
    lib.det_eos_token.argtypes = [ctypes.c_void_p]
    lib.det_eos_token.restype = ctypes.c_int32

    # det_vocab_size
    lib.det_vocab_size.argtypes = [ctypes.c_void_p]
    lib.det_vocab_size.restype = ctypes.c_int32

    # det_model_info
    lib.det_model_info.argtypes = [ctypes.c_void_p]
    lib.det_model_info.restype = ctypes.c_char_p

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

        # Get tokenizer pointer (stored in model struct)
        # For now, we use the model handle to access tokenizer functions
        self._tokenizer = self._handle  # In C, model->tokenizer is accessed via model

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
        return self._lib.det_vocab_size(self._tokenizer)

    @property
    def bos_token(self) -> int:
        """Get BOS token ID."""
        return self._lib.det_bos_token(self._tokenizer)

    @property
    def eos_token(self) -> int:
        """Get EOS token ID."""
        return self._lib.det_eos_token(self._tokenizer)

    def reset(self):
        """Reset KV cache for new conversation."""
        self._lib.det_model_reset(self._handle)

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
        """Get text for a single token."""
        result = self._lib.det_token_to_text(self._tokenizer, token_id)
        return result.decode('utf-8') if result else ""

    def forward(self, tokens: List[int]) -> 'ctypes.POINTER(DetTensor)':
        """Run forward pass, return logits tensor."""
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
                 callback: Callable[[str, int], None] = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            params: Sampling parameters
            callback: Called for each token (text, token_id)

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
