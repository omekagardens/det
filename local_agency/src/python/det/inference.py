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
        Generate text with truthfulness evaluation.

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

        text = self.detokenize(generated)

        # Compute truthfulness score
        evaluator = get_truthfulness_evaluator()
        if det_state:
            score = evaluator.evaluate_from_det_state(
                creature_state=det_state,
                bond_state=bond_state,
                num_tokens=len(generated)
            )
        else:
            # Default state if not provided
            score = evaluator.evaluate(num_tokens=len(generated))

        return text, score


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
# TRUTHFULNESS WEIGHTING (Phase 26.6)
# =============================================================================

@dataclass
class TruthfulnessScore:
    """
    Composite truthfulness score for generated output.

    The truthfulness system provides a reliability estimate for LLM outputs
    based on DET physics principles:
    - Debt (q): Structure accumulates from ungrounded claims
    - Agency (a): Higher agency = more grounded in real capabilities
    - Entropy (H): Lower attention entropy = more confident/focused generation
    - Coherence (C): Bond coherence reflects relational grounding

    Formula: T = w_debt/(1+q) + w_agency*a + w_entropy*(1-H/H_max) + w_coherence*C
    """

    # Overall truthfulness score [0, 1]
    total: float

    # Component scores
    debt_component: float      # Lower debt → higher truth
    agency_component: float    # Higher agency → higher truth
    entropy_component: float   # Lower entropy → higher confidence
    coherence_component: float # Higher coherence → higher truth

    # Raw values (for debugging/calibration)
    debt: float          # q value
    agency: float        # a value
    entropy: float       # H value (attention entropy)
    coherence: float     # C value (bond coherence)

    # Metadata
    num_tokens: int
    confidence_level: str  # 'high', 'medium', 'low', 'very_low'

    def __repr__(self):
        return (f"TruthfulnessScore(T={self.total:.3f}, "
                f"confidence={self.confidence_level}, "
                f"debt={self.debt:.3f}, agency={self.agency:.3f})")


@dataclass
class TruthfulnessWeights:
    """Weights for truthfulness components."""
    w_debt: float = 0.25      # Weight for debt component
    w_agency: float = 0.30    # Weight for agency component
    w_entropy: float = 0.25   # Weight for entropy component
    w_coherence: float = 0.20 # Weight for coherence component

    def normalize(self):
        """Ensure weights sum to 1.0."""
        total = self.w_debt + self.w_agency + self.w_entropy + self.w_coherence
        if total > 0:
            self.w_debt /= total
            self.w_agency /= total
            self.w_entropy /= total
            self.w_coherence /= total


class TruthfulnessEvaluator:
    """
    Evaluates truthfulness of LLM outputs using DET physics.

    This is the sacred integration point where DET theory provides
    a principled reliability estimate for generated content.

    Anti-hallucination mechanisms:
    - Reward hacking: F expenditure tracks real compute
    - False confidence: Agency from structure, not assertion
    - Ungrounded claims: Debt (q) accumulation
    - Post-hoc justification: Atomic commits, auditable trace
    """

    # Thresholds for confidence levels
    CONFIDENCE_THRESHOLDS = {
        'high': 0.75,
        'medium': 0.50,
        'low': 0.25,
    }

    # Maximum expected entropy (for normalization)
    # For vocab_size ~150k, H_max ≈ log2(150000) ≈ 17.2
    # But practical attention entropy is much lower
    H_MAX = 8.0  # Practical maximum for attention entropy

    def __init__(self, weights: TruthfulnessWeights = None):
        """Initialize evaluator with optional custom weights."""
        self.weights = weights or TruthfulnessWeights()
        self.weights.normalize()

        # Calibration data (accumulated for future calibration)
        self._calibration_samples: List[dict] = []

    def evaluate(self,
                 debt: float = 0.0,
                 agency: float = 0.5,
                 entropy: float = 0.0,
                 coherence: float = 1.0,
                 num_tokens: int = 0) -> TruthfulnessScore:
        """
        Compute truthfulness score from DET state.

        Args:
            debt: Structure/debt value q (0 = no debt, higher = more debt)
            agency: Agency value a (0 = no agency, 1 = full agency)
            entropy: Attention entropy H (0 = fully confident, higher = uncertain)
            coherence: Bond coherence C (0 = disconnected, 1 = fully coherent)
            num_tokens: Number of tokens generated (for context)

        Returns:
            TruthfulnessScore with composite score and components
        """
        w = self.weights

        # Compute components
        # Debt component: lower debt → higher score
        debt_component = w.w_debt / (1.0 + debt)

        # Agency component: higher agency → higher score
        # Clamp agency to [0, 1]
        agency_clamped = max(0.0, min(1.0, agency))
        agency_component = w.w_agency * agency_clamped

        # Entropy component: lower entropy → higher score
        # Normalize entropy to [0, 1] range
        entropy_normalized = min(1.0, entropy / self.H_MAX)
        entropy_component = w.w_entropy * (1.0 - entropy_normalized)

        # Coherence component: higher coherence → higher score
        coherence_clamped = max(0.0, min(1.0, coherence))
        coherence_component = w.w_coherence * coherence_clamped

        # Total score
        total = (debt_component + agency_component +
                 entropy_component + coherence_component)

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

        return TruthfulnessScore(
            total=total,
            debt_component=debt_component,
            agency_component=agency_component,
            entropy_component=entropy_component,
            coherence_component=coherence_component,
            debt=debt,
            agency=agency,
            entropy=entropy,
            coherence=coherence,
            num_tokens=num_tokens,
            confidence_level=confidence_level
        )

    def evaluate_from_det_state(self,
                                 creature_state: dict,
                                 bond_state: dict = None,
                                 attention_entropy: float = None,
                                 num_tokens: int = 0) -> TruthfulnessScore:
        """
        Evaluate truthfulness from DET creature and bond state.

        Args:
            creature_state: Dict with 'F', 'a', 'q' from creature
            bond_state: Optional dict with 'coherence' from bonds
            attention_entropy: Optional attention entropy from model internals
            num_tokens: Number of tokens generated

        Returns:
            TruthfulnessScore
        """
        debt = creature_state.get('q', 0.0)
        agency = creature_state.get('a', 0.5)

        # Default entropy to 0 if not provided (optimistic)
        entropy = attention_entropy if attention_entropy is not None else 0.0

        # Default coherence to 1.0 if not provided
        coherence = 1.0
        if bond_state:
            coherence = bond_state.get('coherence', bond_state.get('C', 1.0))

        return self.evaluate(
            debt=debt,
            agency=agency,
            entropy=entropy,
            coherence=coherence,
            num_tokens=num_tokens
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
            metadata: Optional additional context (prompt, response, etc.)
        """
        self._calibration_samples.append({
            'score': score,
            'correct': ground_truth_correct,
            'metadata': metadata or {}
        })

    def get_calibration_stats(self) -> dict:
        """Get statistics from calibration samples."""
        if not self._calibration_samples:
            return {'num_samples': 0}

        correct_samples = [s for s in self._calibration_samples if s['correct']]
        incorrect_samples = [s for s in self._calibration_samples if not s['correct']]

        avg_score_correct = (
            sum(s['score'].total for s in correct_samples) / len(correct_samples)
            if correct_samples else 0.0
        )
        avg_score_incorrect = (
            sum(s['score'].total for s in incorrect_samples) / len(incorrect_samples)
            if incorrect_samples else 0.0
        )

        return {
            'num_samples': len(self._calibration_samples),
            'num_correct': len(correct_samples),
            'num_incorrect': len(incorrect_samples),
            'avg_score_correct': avg_score_correct,
            'avg_score_incorrect': avg_score_incorrect,
            'score_separation': avg_score_correct - avg_score_incorrect,
        }


# Global evaluator instance
_truthfulness_evaluator: Optional[TruthfulnessEvaluator] = None


def get_truthfulness_evaluator() -> TruthfulnessEvaluator:
    """Get or create the global truthfulness evaluator."""
    global _truthfulness_evaluator
    if _truthfulness_evaluator is None:
        _truthfulness_evaluator = TruthfulnessEvaluator()
    return _truthfulness_evaluator


def evaluate_truthfulness(debt: float = 0.0,
                          agency: float = 0.5,
                          entropy: float = 0.0,
                          coherence: float = 1.0,
                          num_tokens: int = 0) -> TruthfulnessScore:
    """Convenience function to evaluate truthfulness."""
    return get_truthfulness_evaluator().evaluate(
        debt=debt, agency=agency, entropy=entropy,
        coherence=coherence, num_tokens=num_tokens
    )
