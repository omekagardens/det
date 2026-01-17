"""
DET LLM Interface - Ollama Integration
======================================

Connects the DET core to Ollama-hosted LLMs via the Port Protocol.
"""

import json
import requests
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import IntEnum


class IntentType(IntEnum):
    """Intent classification for LLM outputs."""
    ANSWER = 0
    PLAN = 1
    EXECUTE = 2
    LEARN = 3
    DEBUG = 4


class DomainType(IntEnum):
    """Domain classification."""
    MATH = 0
    LANGUAGE = 1
    TOOL_USE = 2
    SCIENCE = 3


@dataclass
class DetIntentPacket:
    """
    Port Protocol packet from LLM to DET core.

    This is the structured output format that the LLM produces,
    which gets translated into DET stimulus.
    """
    intent: IntentType
    confidence: float
    domain: DomainType
    complexity: float
    tokens: List[int] = field(default_factory=list)

    # Risk assessment
    risk_external: float = 0.0  # Risk of external effects
    risk_irreversible: float = 0.0  # Risk of irreversible actions

    # Metadata
    raw_text: str = ""
    requires_tool: bool = False
    tool_name: Optional[str] = None

    def to_port_activations(self) -> tuple[List[int], List[float]]:
        """
        Convert to port indices and activations for DET injection.

        Returns:
            Tuple of (port_indices, activations).
        """
        # Intent ports are 0-4
        # Domain ports are 5-8
        indices = [int(self.intent), 5 + int(self.domain)]
        activations = [self.confidence, 1.0 - self.complexity * 0.5]
        return indices, activations


class OllamaClient:
    """
    Client for Ollama API.

    Provides structured LLM inference with DET-compatible outputs.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        timeout: float = 60.0
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL.
            model: Model name to use.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=5.0
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        """List available models."""
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=10.0
            )
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except requests.RequestException:
            pass
        return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        context: Optional[List[int]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt.
            system: Optional system prompt.
            context: Optional context tokens from previous response.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: Whether to stream the response.

        Returns:
            Response dictionary with 'response', 'context', etc.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        if system:
            payload["system"] = system

        if context:
            payload["context"] = context

        response = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Chat with the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Response dictionary.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        response = self._session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


# System prompt for DET-aware responses
DET_SYSTEM_PROMPT = """You are an AI assistant integrated with a DET (Deep Existence Theory) core.

When responding, you should consider:
1. The intent of the request (answer, plan, execute, learn, debug)
2. The domain (math, language, tool_use, science)
3. The complexity and risk involved

You have access to tools for file operations, web searches, and code execution.
Before taking actions, consider whether they are reversible and safe.

Respond naturally, but be mindful that your outputs influence an underlying
cognitive substrate that tracks coherence, agency, and emotional state."""


class DETLLMInterface:
    """
    High-level interface between DET core and LLM.

    Handles the full request→DET→LLM→response flow with affect modulation.
    """

    def __init__(
        self,
        core,  # DETCore instance
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b"
    ):
        """
        Initialize the DET-LLM interface.

        Args:
            core: DETCore instance.
            ollama_url: Ollama API URL.
            model: Model to use.
        """
        self.core = core
        self.client = OllamaClient(base_url=ollama_url, model=model)
        self.context: Optional[List[int]] = None
        self.conversation_history: List[Dict[str, str]] = []

    def is_ready(self) -> bool:
        """Check if the system is ready."""
        return self.client.is_available()

    def classify_intent(self, text: str) -> tuple[IntentType, float]:
        """
        Classify the intent of a request.

        Simple keyword-based classification for Phase 1.
        Can be replaced with LLM-based classification later.

        Returns:
            Tuple of (intent, confidence).
        """
        text_lower = text.lower()

        # Execution keywords
        exec_keywords = ["run", "execute", "do", "create", "make", "write", "delete"]
        if any(kw in text_lower for kw in exec_keywords):
            return IntentType.EXECUTE, 0.8

        # Planning keywords
        plan_keywords = ["plan", "how to", "steps", "approach", "strategy"]
        if any(kw in text_lower for kw in plan_keywords):
            return IntentType.PLAN, 0.7

        # Learning keywords
        learn_keywords = ["explain", "teach", "learn", "understand", "why"]
        if any(kw in text_lower for kw in learn_keywords):
            return IntentType.LEARN, 0.7

        # Debug keywords
        debug_keywords = ["debug", "fix", "error", "problem", "issue", "wrong"]
        if any(kw in text_lower for kw in debug_keywords):
            return IntentType.DEBUG, 0.7

        # Default to answer
        return IntentType.ANSWER, 0.6

    def classify_domain(self, text: str) -> tuple[DomainType, float]:
        """
        Classify the domain of a request.

        Simple keyword-based classification for Phase 1.

        Returns:
            Tuple of (domain, confidence).
        """
        text_lower = text.lower()

        # Math keywords
        math_keywords = ["calculate", "math", "equation", "number", "compute", "sum"]
        if any(kw in text_lower for kw in math_keywords):
            return DomainType.MATH, 0.8

        # Tool use keywords
        tool_keywords = ["file", "run", "command", "execute", "script", "code"]
        if any(kw in text_lower for kw in tool_keywords):
            return DomainType.TOOL_USE, 0.7

        # Science keywords
        science_keywords = ["science", "physics", "chemistry", "biology", "experiment"]
        if any(kw in text_lower for kw in science_keywords):
            return DomainType.SCIENCE, 0.7

        # Default to language
        return DomainType.LANGUAGE, 0.6

    def estimate_complexity(self, text: str) -> float:
        """
        Estimate the complexity of a request.

        Simple heuristics for Phase 1.

        Returns:
            Complexity score from 0.0 to 1.0.
        """
        # Word count
        words = len(text.split())
        word_complexity = min(words / 100.0, 1.0)

        # Question complexity
        question_marks = text.count("?")
        question_complexity = min(question_marks * 0.2, 0.5)

        # Code indicators
        code_indicators = ["```", "def ", "class ", "function", "import"]
        code_complexity = 0.3 if any(ind in text for ind in code_indicators) else 0.0

        return min(word_complexity + question_complexity + code_complexity, 1.0)

    def assess_risk(self, text: str) -> tuple[float, float]:
        """
        Assess risk of a request.

        Returns:
            Tuple of (external_risk, irreversibility_risk).
        """
        text_lower = text.lower()

        # External risk keywords
        external_keywords = ["send", "post", "publish", "upload", "email", "network"]
        external_risk = 0.5 if any(kw in text_lower for kw in external_keywords) else 0.1

        # Irreversibility keywords
        irreversible_keywords = ["delete", "remove", "drop", "destroy", "overwrite", "format"]
        irreversible_risk = 0.7 if any(kw in text_lower for kw in irreversible_keywords) else 0.1

        return external_risk, irreversible_risk

    def create_intent_packet(self, text: str) -> DetIntentPacket:
        """
        Create a DET intent packet from user input.

        Args:
            text: User input text.

        Returns:
            DetIntentPacket for DET injection.
        """
        intent, intent_conf = self.classify_intent(text)
        domain, domain_conf = self.classify_domain(text)
        complexity = self.estimate_complexity(text)
        ext_risk, irr_risk = self.assess_risk(text)

        # Simple tokenization (word-based for now)
        tokens = [hash(word) % 65536 for word in text.split()[:50]]

        return DetIntentPacket(
            intent=intent,
            confidence=intent_conf,
            domain=domain,
            complexity=complexity,
            tokens=tokens,
            risk_external=ext_risk,
            risk_irreversible=irr_risk,
            raw_text=text
        )

    def process_request(
        self,
        user_input: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Process a user request through the full DET→LLM pipeline.

        This is the main entry point for the system.

        Args:
            user_input: User's input text.
            max_retries: Maximum gatekeeper retries.

        Returns:
            Response dictionary with 'response', 'decision', 'affect', etc.
        """
        from .core import DETDecision

        # Create intent packet
        packet = self.create_intent_packet(user_input)

        # Inject into DET core
        indices, activations = packet.to_port_activations()
        self.core.inject_stimulus(indices, activations)

        # Create interface bonds
        self.core.create_interface_bonds(int(packet.domain))

        # Run DET simulation step
        self.core.step(0.1)

        # Evaluate through gatekeeper
        decision = self.core.evaluate_request(
            packet.tokens,
            target_domain=int(packet.domain),
            retry_count=0
        )

        result = {
            "decision": decision,
            "intent": packet.intent.name,
            "domain": packet.domain.name,
            "complexity": packet.complexity,
            "risk": {
                "external": packet.risk_external,
                "irreversible": packet.risk_irreversible,
            },
            "affect": dict(zip(
                ["valence", "arousal", "bondedness"],
                self.core.get_self_affect()
            )),
            "emotion": self.core.get_emotion_string(),
        }

        # Handle gatekeeper decisions
        if decision == DETDecision.STOP:
            result["response"] = "[DET: Request declined - system in protective state]"
            result["stopped"] = True

        elif decision == DETDecision.ESCALATE:
            result["response"] = "[DET: Request needs external assistance]"
            result["escalated"] = True

        elif decision == DETDecision.RETRY:
            # Could reformulate, for now just proceed with caution
            result["retried"] = True
            decision = DETDecision.PROCEED

        if decision == DETDecision.PROCEED:
            # Generate LLM response
            try:
                # Add affect modulation to temperature
                valence, arousal, bondedness = self.core.get_self_affect()
                temperature = 0.7 + 0.2 * arousal - 0.1 * bondedness

                # Build messages for chat
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })

                messages = [{"role": "system", "content": DET_SYSTEM_PROMPT}]
                messages.extend(self.conversation_history[-10:])  # Keep last 10

                llm_response = self.client.chat(
                    messages=messages,
                    temperature=max(0.1, min(temperature, 1.0))
                )

                assistant_message = llm_response.get("message", {}).get("content", "")
                result["response"] = assistant_message

                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })

            except requests.RequestException as e:
                result["response"] = f"[LLM Error: {e}]"
                result["error"] = str(e)

        # Clean up interface bonds
        self.core.cleanup_interface_bonds()

        # Update aggregates
        presence, coherence, resource, debt = self.core.get_aggregates()
        result["aggregates"] = {
            "presence": presence,
            "coherence": coherence,
            "resource": resource,
            "debt": debt,
        }

        return result

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        self.context = None
