"""
DET Multi-LLM Routing
=====================

Domain-aware routing to specialized LLM models.

Phase 5.1 Implementation.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path

from .llm import OllamaClient, DET_SYSTEM_PROMPT
from .memory import MemoryDomain


class ModelStatus(Enum):
    """Status of a model."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class ModelConfig:
    """Configuration for a specialized model."""
    name: str  # Ollama model name (e.g., "deepseek-coder:6.7b")
    display_name: str  # Human-readable name
    domains: List[MemoryDomain]  # Domains this model handles
    priority: int = 0  # Higher = preferred when multiple models match

    # Model parameters
    context_length: int = 4096
    default_temperature: float = 0.7
    system_prompt: Optional[str] = None

    # Performance hints
    is_fast: bool = False  # Quick responses
    is_accurate: bool = True  # Prioritize accuracy
    supports_code: bool = False
    supports_math: bool = False

    # Runtime state
    status: ModelStatus = ModelStatus.UNKNOWN
    last_check: float = 0.0
    avg_latency_ms: float = 0.0
    error_count: int = 0


# Default model configurations
DEFAULT_MODELS = {
    "general": ModelConfig(
        name="llama3.2:3b",
        display_name="Llama 3.2 3B",
        domains=[MemoryDomain.GENERAL, MemoryDomain.DIALOGUE, MemoryDomain.LANGUAGE],
        priority=0,
        is_fast=True,
    ),
    "math": ModelConfig(
        name="deepseek-math:7b",
        display_name="DeepSeek Math 7B",
        domains=[MemoryDomain.MATH],
        priority=10,
        supports_math=True,
        default_temperature=0.3,
        system_prompt="You are a mathematical reasoning assistant. Show your work step by step.",
    ),
    "code": ModelConfig(
        name="qwen2.5-coder:7b",
        display_name="Qwen 2.5 Coder 7B",
        domains=[MemoryDomain.CODE, MemoryDomain.TOOL_USE],
        priority=10,
        supports_code=True,
        default_temperature=0.4,
        system_prompt="You are a coding assistant. Write clean, efficient, well-documented code.",
    ),
    "reasoning": ModelConfig(
        name="deepseek-r1:8b",
        display_name="DeepSeek R1 8B",
        domains=[MemoryDomain.REASONING, MemoryDomain.SCIENCE],
        priority=10,
        is_accurate=True,
        default_temperature=0.5,
        system_prompt="You are a logical reasoning assistant. Think through problems step by step.",
    ),
}


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    model_config: ModelConfig
    domain: MemoryDomain
    confidence: float
    fallback_used: bool = False
    reason: str = ""


class ModelPool:
    """
    Manages a pool of LLM models with health checking.

    Provides:
    - Model registration and configuration
    - Periodic health checks
    - Latency tracking
    - Graceful degradation
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        health_check_interval: float = 60.0
    ):
        """
        Initialize the model pool.

        Args:
            ollama_url: Ollama API base URL.
            health_check_interval: Seconds between health checks.
        """
        self.ollama_url = ollama_url
        self.health_check_interval = health_check_interval

        self._models: Dict[str, ModelConfig] = {}
        self._clients: Dict[str, OllamaClient] = {}
        self._lock = threading.Lock()
        self._health_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()

    def register_model(self, key: str, config: ModelConfig):
        """
        Register a model configuration.

        Args:
            key: Unique key for this model.
            config: Model configuration.
        """
        with self._lock:
            self._models[key] = config
            self._clients[key] = OllamaClient(
                base_url=self.ollama_url,
                model=config.name,
            )

    def register_defaults(self):
        """Register default model configurations."""
        for key, config in DEFAULT_MODELS.items():
            self.register_model(key, config)

    def get_model(self, key: str) -> Optional[ModelConfig]:
        """Get a model configuration by key."""
        return self._models.get(key)

    def get_client(self, key: str) -> Optional[OllamaClient]:
        """Get a client for a model."""
        return self._clients.get(key)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models with status."""
        result = []
        for key, config in self._models.items():
            result.append({
                "key": key,
                "name": config.name,
                "display_name": config.display_name,
                "domains": [d.name for d in config.domains],
                "status": config.status.value,
                "priority": config.priority,
                "avg_latency_ms": config.avg_latency_ms,
            })
        return result

    def check_model_health(self, key: str) -> ModelStatus:
        """
        Check health of a specific model.

        Args:
            key: Model key.

        Returns:
            Model status after check.
        """
        config = self._models.get(key)
        client = self._clients.get(key)

        if not config or not client:
            return ModelStatus.UNKNOWN

        try:
            # Check if model is available
            available_models = client.list_models()

            if config.name in available_models:
                config.status = ModelStatus.AVAILABLE
                config.error_count = 0
            else:
                # Model not loaded, try to find a matching one
                matching = [m for m in available_models if m.startswith(config.name.split(":")[0])]
                if matching:
                    # Update to use available version
                    config.name = matching[0]
                    client.model = matching[0]
                    config.status = ModelStatus.AVAILABLE
                else:
                    config.status = ModelStatus.UNAVAILABLE

        except Exception as e:
            config.status = ModelStatus.ERROR
            config.error_count += 1

        config.last_check = time.time()
        return config.status

    def check_all_health(self):
        """Check health of all registered models."""
        for key in self._models:
            self.check_model_health(key)

    def start_health_monitoring(self):
        """Start background health monitoring thread."""
        if self._health_thread and self._health_thread.is_alive():
            return

        self._stop_health_check.clear()

        def monitor():
            while not self._stop_health_check.wait(self.health_check_interval):
                self.check_all_health()

        self._health_thread = threading.Thread(target=monitor, daemon=True)
        self._health_thread.start()

    def stop_health_monitoring(self):
        """Stop background health monitoring."""
        self._stop_health_check.set()
        if self._health_thread:
            self._health_thread.join(timeout=5.0)

    def get_available_models(self) -> List[str]:
        """Get list of available model keys."""
        return [
            key for key, config in self._models.items()
            if config.status == ModelStatus.AVAILABLE
        ]

    def get_models_for_domain(self, domain: MemoryDomain) -> List[str]:
        """
        Get models that handle a specific domain.

        Args:
            domain: Target domain.

        Returns:
            List of model keys sorted by priority (highest first).
        """
        matching = [
            (key, config) for key, config in self._models.items()
            if domain in config.domains and config.status == ModelStatus.AVAILABLE
        ]
        # Sort by priority (highest first)
        matching.sort(key=lambda x: x[1].priority, reverse=True)
        return [key for key, _ in matching]


class LLMRouter:
    """
    Routes requests to appropriate LLM models based on domain.

    Integrates with DET core and memory system for intelligent routing.
    """

    def __init__(
        self,
        core=None,
        model_pool: Optional[ModelPool] = None,
        ollama_url: str = "http://localhost:11434",
        fallback_model: str = "general"
    ):
        """
        Initialize the LLM router.

        Args:
            core: DETCore instance for affect-modulated routing.
            model_pool: ModelPool instance (created if None).
            ollama_url: Ollama API URL.
            fallback_model: Model key to use as fallback.
        """
        self.core = core
        self.fallback_model = fallback_model

        # Initialize model pool
        self.pool = model_pool or ModelPool(ollama_url=ollama_url)
        if not model_pool:
            self.pool.register_defaults()

        # Conversation contexts per model
        self._contexts: Dict[str, List[Dict[str, str]]] = {}

        # Routing callbacks
        self.on_route: Optional[Callable[[RoutingResult], None]] = None
        self.on_fallback: Optional[Callable[[str, str], None]] = None

    def initialize(self):
        """Initialize the router (check model availability)."""
        self.pool.check_all_health()
        self.pool.start_health_monitoring()

    def shutdown(self):
        """Shutdown the router."""
        self.pool.stop_health_monitoring()

    def route(
        self,
        text: str,
        domain: Optional[MemoryDomain] = None
    ) -> RoutingResult:
        """
        Route a request to the appropriate model.

        Args:
            text: Request text.
            domain: Optional explicit domain (auto-detected if None).

        Returns:
            RoutingResult with selected model.
        """
        # Auto-detect domain if not provided
        if domain is None:
            domain, confidence = self._detect_domain(text)
        else:
            confidence = 1.0

        # Get models for this domain
        model_keys = self.pool.get_models_for_domain(domain)

        fallback_used = False
        reason = f"Best match for domain {domain.name}"

        if model_keys:
            # Use the highest priority available model
            selected_key = model_keys[0]
        else:
            # Fall back to general model
            selected_key = self.fallback_model
            fallback_used = True
            reason = f"No specialized model available for {domain.name}, using fallback"

            if self.on_fallback:
                self.on_fallback(domain.name, selected_key)

        config = self.pool.get_model(selected_key)

        if not config:
            # Last resort: use first available model
            available = self.pool.get_available_models()
            if available:
                selected_key = available[0]
                config = self.pool.get_model(selected_key)
                fallback_used = True
                reason = "Emergency fallback to first available model"
            else:
                raise RuntimeError("No LLM models available")

        result = RoutingResult(
            model_config=config,
            domain=domain,
            confidence=confidence,
            fallback_used=fallback_used,
            reason=reason,
        )

        if self.on_route:
            self.on_route(result)

        return result

    def _detect_domain(self, text: str) -> tuple[MemoryDomain, float]:
        """
        Detect domain from text.

        Uses keyword matching similar to DomainRouter.
        """
        text_lower = text.lower()

        # Domain keywords
        domain_scores = {domain: 0.0 for domain in MemoryDomain}

        keywords = {
            MemoryDomain.MATH: [
                "calculate", "compute", "equation", "math", "sum", "integral",
                "integrate", "derivative", "algebra", "geometry", "statistics",
                "probability", "solve", "formula"
            ],
            MemoryDomain.CODE: [
                "code", "program", "function", "class", "debug", "compile",
                "python", "javascript", "rust", "implement", "algorithm"
            ],
            MemoryDomain.TOOL_USE: [
                "file", "directory", "command", "execute", "run", "shell",
                "bash", "script", "install", "download"
            ],
            MemoryDomain.SCIENCE: [
                "physics", "chemistry", "biology", "experiment", "theory",
                "research", "molecule", "atom", "energy", "force"
            ],
            MemoryDomain.REASONING: [
                "why", "because", "therefore", "logic", "reason", "analyze",
                "deduce", "infer", "conclude", "evaluate"
            ],
            MemoryDomain.LANGUAGE: [
                "write", "translate", "grammar", "sentence", "essay",
                "story", "poem", "spelling", "vocabulary"
            ],
            MemoryDomain.DIALOGUE: [
                "conversation", "chat", "discuss", "talk", "clarify"
            ],
        }

        for domain, kws in keywords.items():
            for kw in kws:
                if kw in text_lower:
                    domain_scores[domain] += 1.0

        # Find best domain
        best_domain = max(domain_scores, key=domain_scores.get)
        best_score = domain_scores[best_domain]

        if best_score == 0:
            return MemoryDomain.GENERAL, 0.5

        # Normalize confidence
        total = sum(domain_scores.values())
        confidence = best_score / total if total > 0 else 0.5

        return best_domain, min(confidence, 0.95)

    def generate(
        self,
        text: str,
        domain: Optional[MemoryDomain] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the appropriate model.

        Args:
            text: User input text.
            domain: Optional explicit domain.
            temperature: Override temperature (uses model default if None).
            max_tokens: Maximum tokens to generate.
            system_prompt: Override system prompt.

        Returns:
            Response dictionary with 'response', 'model', 'domain', etc.
        """
        # Route to model
        routing = self.route(text, domain)
        config = routing.model_config
        client = self.pool.get_client(self._get_model_key(config))

        if not client:
            return {
                "error": "No client available",
                "response": "[Error: No LLM client available]"
            }

        # Apply DET affect modulation if core is available
        if self.core:
            valence, arousal, bondedness = self.core.get_self_affect()
            # Modulate temperature based on affect
            affect_temp = 0.1 * arousal - 0.05 * bondedness
        else:
            affect_temp = 0.0

        # Determine temperature
        if temperature is None:
            temperature = config.default_temperature
        temperature = max(0.1, min(temperature + affect_temp, 1.5))

        # Get or create context
        model_key = self._get_model_key(config)
        if model_key not in self._contexts:
            self._contexts[model_key] = []

        # Build messages
        sys_prompt = system_prompt or config.system_prompt or DET_SYSTEM_PROMPT

        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(self._contexts[model_key][-10:])  # Last 10 turns
        messages.append({"role": "user", "content": text})

        # Generate
        start_time = time.time()

        try:
            response = client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Update latency tracking
            config.avg_latency_ms = (
                config.avg_latency_ms * 0.9 + latency_ms * 0.1
            )

            assistant_content = response.get("message", {}).get("content", "")

            # Store in context
            self._contexts[model_key].append({"role": "user", "content": text})
            self._contexts[model_key].append({"role": "assistant", "content": assistant_content})

            # Trim context if too long
            if len(self._contexts[model_key]) > 20:
                self._contexts[model_key] = self._contexts[model_key][-20:]

            return {
                "response": assistant_content,
                "model": config.name,
                "model_display": config.display_name,
                "domain": routing.domain.name,
                "routing_confidence": routing.confidence,
                "fallback_used": routing.fallback_used,
                "routing_reason": routing.reason,
                "temperature": temperature,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            config.error_count += 1
            return {
                "error": str(e),
                "response": f"[Error from {config.display_name}: {e}]",
                "model": config.name,
                "domain": routing.domain.name,
            }

    def _get_model_key(self, config: ModelConfig) -> str:
        """Get the pool key for a model config."""
        for key, cfg in self.pool._models.items():
            if cfg is config:
                return key
        return self.fallback_model

    def clear_context(self, model_key: Optional[str] = None):
        """
        Clear conversation context.

        Args:
            model_key: Specific model key (clears all if None).
        """
        if model_key:
            self._contexts.pop(model_key, None)
        else:
            self._contexts.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "models": self.pool.list_models(),
            "available_count": len(self.pool.get_available_models()),
            "fallback_model": self.fallback_model,
            "has_core": self.core is not None,
            "context_sizes": {
                key: len(msgs) for key, msgs in self._contexts.items()
            },
        }


class MultiModelInterface:
    """
    High-level interface for multi-model LLM interaction with DET integration.

    Provides the complete pipeline:
    - Domain detection
    - Model routing
    - DET gatekeeper integration
    - Response generation
    """

    def __init__(
        self,
        core=None,
        router: Optional[LLMRouter] = None,
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize the multi-model interface.

        Args:
            core: DETCore instance.
            router: LLMRouter instance (created if None).
            ollama_url: Ollama API URL.
        """
        self.core = core
        self.router = router or LLMRouter(core=core, ollama_url=ollama_url)

    def initialize(self):
        """Initialize the interface."""
        self.router.initialize()

    def shutdown(self):
        """Shutdown the interface."""
        self.router.shutdown()

    def process(
        self,
        user_input: str,
        domain: Optional[MemoryDomain] = None
    ) -> Dict[str, Any]:
        """
        Process a user request through the full pipeline.

        Args:
            user_input: User input text.
            domain: Optional explicit domain.

        Returns:
            Response dictionary with full results.
        """
        from .core import DETDecision
        from .llm import DetIntentPacket, IntentType, DomainType

        result = {
            "input": user_input,
            "timestamp": time.time(),
        }

        # Route to get domain
        routing = self.router.route(user_input, domain)
        result["routing"] = {
            "model": routing.model_config.name,
            "domain": routing.domain.name,
            "confidence": routing.confidence,
            "fallback": routing.fallback_used,
        }

        # DET gatekeeper if core available
        if self.core:
            # Create simple intent packet
            tokens = [hash(w) % 65536 for w in user_input.split()[:50]]

            # Inject stimulus
            domain_idx = routing.domain.value
            self.core.inject_stimulus([domain_idx], [routing.confidence])

            # Step simulation
            self.core.step(0.1)

            # Evaluate
            decision = self.core.evaluate_request(
                tokens,
                target_domain=domain_idx,
                retry_count=0
            )

            result["det"] = {
                "decision": decision.name,
                "affect": dict(zip(
                    ["valence", "arousal", "bondedness"],
                    self.core.get_self_affect()
                )),
                "emotion": self.core.get_emotion_string(),
            }

            # Handle decisions
            if decision == DETDecision.STOP:
                result["response"] = "[DET: Request declined - system in protective state]"
                result["stopped"] = True
                return result

            if decision == DETDecision.ESCALATE:
                result["escalated"] = True
                # Continue but note escalation

        # Generate response
        gen_result = self.router.generate(user_input, routing.domain)
        result.update(gen_result)

        return result

    def get_status(self) -> Dict[str, Any]:
        """Get interface status."""
        status = self.router.get_status()
        status["has_core"] = self.core is not None

        if self.core:
            status["det_state"] = {
                "affect": dict(zip(
                    ["valence", "arousal", "bondedness"],
                    self.core.get_self_affect()
                )),
                "emotion": self.core.get_emotion_string(),
            }

        return status
