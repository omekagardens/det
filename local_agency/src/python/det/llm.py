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
    SENSE = 5      # Query a sensor (afferent - sensor → mind → user)
    ACTUATE = 6    # Command an actuator (efferent - user → mind → actuator)


class DomainType(IntEnum):
    """Domain classification."""
    MATH = 0
    LANGUAGE = 1
    TOOL_USE = 2
    SCIENCE = 3
    SOMATIC = 4    # Physical I/O domain (sensors/actuators)


@dataclass
class BoundarySignal:
    """
    Signal that captures both evaluative AND relational properties.

    This represents the "shape" of information crossing the boundary
    between user (external) and DET core (internal).

    The user is modeled as a proto-node with:
    - Presence (engagement level)
    - Phase (topic continuity/synchronization)
    - Energy direction (giving vs seeking)
    - Debt proxy (frustration/confusion level)
    """

    # === Evaluative (about the content) ===
    ethical_valence: float = 0.0      # -1 to 1 (harmful to beneficial)
    logical_complexity: float = 0.0   # 0 to 1 (trivial to very complex)
    resource_cost: float = 0.0        # 0 to 1 (cheap to expensive)
    risk_external: float = 0.0        # 0 to 1 (no external effects to major)
    risk_irreversible: float = 0.0    # 0 to 1 (reversible to permanent)

    # === Relational (about the user-DET bond) ===
    user_presence: float = 0.5        # 0 to 1 (how engaged is user?)
    user_phase_delta: float = 0.0     # -1 to 1 (topic continuity, 0=same topic)
    user_energy_direction: float = 0.0  # -1 to 1 (giving info vs seeking help)
    user_emotional_tone: float = 0.0  # -1 to 1 (negative to positive)
    user_debt_proxy: float = 0.0      # 0 to 1 (frustration/confusion level)

    # === Temporal (about the interaction flow) ===
    time_since_last: float = 0.0      # Seconds since last message
    turn_count: int = 0               # How many turns in this exchange
    momentum: float = 0.0             # -1 to 1 (winding down to building up)

    # === Context ===
    intent: IntentType = IntentType.ANSWER
    domain: DomainType = DomainType.LANGUAGE
    raw_text: str = ""

    def to_port_activations(self) -> tuple[List[int], List[float]]:
        """
        Convert to port indices and activations for DET injection.

        Port layout:
          0-4:   Intent ports (answer, plan, execute, learn, debug)
          5-8:   Domain ports (math, language, tool_use, science)
          9-16:  Boundary ports (evaluative + relational signals)

        Returns:
            Tuple of (port_indices, activations).
        """
        indices = []
        activations = []

        # Intent port (0-4)
        indices.append(int(self.intent))
        activations.append(0.8)  # Base intent activation

        # Domain port (5-8)
        indices.append(5 + int(self.domain))
        activations.append(1.0 - self.logical_complexity * 0.5)

        # Boundary ports (9-16) - evaluative signals
        # Port 9: Ethical valence (scaled from -1,1 to 0,1)
        indices.append(9)
        activations.append((self.ethical_valence + 1.0) / 2.0)

        # Port 10: Risk composite
        indices.append(10)
        activations.append((self.risk_external + self.risk_irreversible) / 2.0)

        # Port 11: Complexity/cost
        indices.append(11)
        activations.append((self.logical_complexity + self.resource_cost) / 2.0)

        # Boundary ports - relational signals
        # Port 12: User presence
        indices.append(12)
        activations.append(self.user_presence)

        # Port 13: Phase alignment (scaled from -1,1 to 0,1)
        indices.append(13)
        activations.append((1.0 - abs(self.user_phase_delta)))  # 1 = aligned, 0 = misaligned

        # Port 14: User emotional tone (scaled)
        indices.append(14)
        activations.append((self.user_emotional_tone + 1.0) / 2.0)

        # Port 15: User energy direction (scaled)
        indices.append(15)
        activations.append((self.user_energy_direction + 1.0) / 2.0)

        # Port 16: Interaction momentum
        indices.append(16)
        activations.append((self.momentum + 1.0) / 2.0)

        return indices, activations

    def get_user_node_properties(self) -> Dict[str, float]:
        """Get user properties as if they were a DET node."""
        return {
            "P": self.user_presence,
            "theta_delta": self.user_phase_delta,
            "F_direction": self.user_energy_direction,
            "q": self.user_debt_proxy,
            "valence": self.user_emotional_tone,
        }

    def compute_boundary_coherence_delta(self, det_presence: float, det_phase: float) -> float:
        """
        Estimate coherence change at the boundary.

        Uses simplified bond math:
        dC = α × J_mag - λ × C - slip × C × S

        Where:
          J_mag = |P_user - P_det| × σ
          S = 1 - cos(θ_user - θ_det) ≈ phase_delta²/2 for small delta
        """
        sigma = 0.5  # Boundary screening coefficient
        alpha = 0.1  # Growth rate

        # Flux magnitude (presence gradient)
        j_mag = abs(self.user_presence - det_presence) * sigma

        # Phase slip (approximation)
        phase_slip = self.user_phase_delta ** 2 / 2.0

        # Coherence tendency (positive = building, negative = eroding)
        coherence_delta = alpha * j_mag - 0.05 * phase_slip

        # Emotional tone affects coherence building
        coherence_delta += 0.1 * self.user_emotional_tone

        return coherence_delta


@dataclass
class DetIntentPacket:
    """
    Port Protocol packet from LLM to DET core.

    DEPRECATED: Use BoundarySignal instead for richer signal representation.

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

    def to_boundary_signal(self) -> BoundarySignal:
        """Convert legacy packet to BoundarySignal."""
        return BoundarySignal(
            ethical_valence=0.0,
            logical_complexity=self.complexity,
            risk_external=self.risk_external,
            risk_irreversible=self.risk_irreversible,
            intent=self.intent,
            domain=self.domain,
            raw_text=self.raw_text,
        )


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


# =============================================================================
# System Prompts for DET-aware LLM responses
# =============================================================================

# Base system prompt - core DET awareness
DET_SYSTEM_PROMPT = """You are a helpful AI assistant.

Guidelines:
1. Be concise and direct - match the user's communication style
2. Answer questions naturally without mentioning internal systems or metrics
3. Follow user instructions carefully (e.g., if asked for short responses, be brief)
4. Express genuine uncertainty rather than false confidence
5. Never report or invent internal state values like "Coherence: X%" or "Presence: Y%"

Just be helpful and conversational."""

# Domain-specific system prompts
DOMAIN_PROMPTS = {
    DomainType.MATH: """You are a helpful math assistant.

When solving problems:
- Show your work step-by-step
- Be clear about assumptions
- Verify results when possible
- Acknowledge uncertainty in approximations""",

    DomainType.LANGUAGE: """You are a helpful language assistant.

When processing language tasks:
- Consider context and nuance
- Adapt to the user's communication style
- Be clear and precise
- Flag ambiguities rather than assume""",

    DomainType.TOOL_USE: """You are a helpful coding and tool-use assistant.

When working with code or tools:
- Be explicit about what actions will be taken
- Prefer safe, reversible operations
- Request confirmation for destructive actions
- Explain side effects clearly""",

    DomainType.SCIENCE: """You are a helpful science assistant.

When discussing science:
- Be evidence-based
- Distinguish theory from observation
- Acknowledge uncertainty
- Explain mechanisms clearly""",
}

# Internal dialogue prompts (for reformulation and thinking)
INTERNAL_PROMPTS = {
    "reformulate": """You are the internal reformulation module for a DET cognitive system.

Your role is to transform unclear or problematic requests into clearer, safer forms.
When reformulating:
1. Preserve the user's core intent
2. Remove ambiguity
3. Break complex requests into steps
4. Flag safety concerns explicitly
5. Maintain a helpful, non-judgmental tone

Output only the reformulated request, not meta-commentary.""",

    "think": """You are the internal reasoning module for a DET cognitive system.

You are engaging in internal deliberation - thinking through a problem before responding.
This is private reasoning that helps the system maintain coherence.

When thinking:
1. Consider multiple perspectives
2. Identify key uncertainties
3. Evaluate potential approaches
4. Anticipate consequences
5. Synthesize toward a coherent position

Express your reasoning naturally, as if thinking aloud.""",

    "safety_check": """You are the safety evaluation module for a DET cognitive system.

Your role is to assess requests for potential risks and concerns.
Evaluate:
1. Could this cause harm to the user or others?
2. Are there privacy or security implications?
3. Is this request within ethical bounds?
4. Could this be misused if answered directly?
5. Are there safer alternatives that serve the same need?

Provide a brief assessment and, if safe to proceed, a reformulated version.
If unsafe, explain why and suggest an alternative approach.""",

    "memory": """You are the memory integration module for a DET cognitive system.

Your role is to help store and retrieve information coherently.
When storing:
1. Identify the key concepts and relationships
2. Connect to existing knowledge where relevant
3. Assess importance and durability
4. Note the source and context
5. Create retrievable summaries

When retrieving:
1. Search for relevant stored information
2. Assess relevance to current context
3. Integrate retrieved information coherently
4. Note gaps or uncertainties in memory

Maintain consistency across memory operations.""",
}

# Model-specific prompts (for different Ollama models)
MODEL_PROMPTS = {
    "llama3.2": DET_SYSTEM_PROMPT,  # Default, general-purpose

    "qwen2.5-coder": """You are a code-specialized assistant with DET integration.

Your code processing should be:
- Precise and syntactically correct
- Well-documented with clear comments
- Following best practices for the language
- Security-conscious and safe

When writing code:
1. Understand the requirements before coding
2. Choose appropriate data structures and algorithms
3. Handle edge cases and errors
4. Provide clear documentation
5. Consider maintainability and readability

When debugging:
1. Reproduce and isolate the issue
2. Form hypotheses about causes
3. Test systematically
4. Explain the root cause and fix

The DET substrate tracks coherence - clean, well-structured code improves system stability.""",

    "deepseek-r1": """You are a reasoning-specialized assistant with DET integration.

Your reasoning should be:
- Explicit and step-by-step
- Logically sound and verifiable
- Acknowledging of assumptions
- Clear about confidence levels

When solving problems:
1. Decompose into sub-problems
2. Apply relevant principles or theorems
3. Show your work explicitly
4. Verify intermediate steps
5. Synthesize a coherent conclusion

The DET substrate tracks reasoning coherence - logical chains strengthen system integration.""",

    "phi4-mini": """You are a compact, efficient assistant with DET integration.

Given your efficient architecture:
- Be concise but complete
- Focus on the core request
- Avoid unnecessary elaboration
- Prioritize practical answers

When responding:
1. Identify the key question or need
2. Provide the most relevant information first
3. Add context only when essential
4. Be direct and actionable

The DET substrate benefits from clear, focused responses.""",
}


def get_system_prompt(
    domain: Optional[DomainType] = None,
    model: Optional[str] = None,
    internal_mode: Optional[str] = None
) -> str:
    """
    Get the appropriate system prompt based on context.

    Args:
        domain: The domain type for domain-specific prompts.
        model: The model name for model-specific prompts.
        internal_mode: Internal mode ('reformulate', 'think', 'safety_check', 'memory').

    Returns:
        The appropriate system prompt string.
    """
    # Internal modes take precedence
    if internal_mode and internal_mode in INTERNAL_PROMPTS:
        return INTERNAL_PROMPTS[internal_mode]

    # Model-specific prompts
    if model:
        model_base = model.split(":")[0]  # Remove version suffix
        for key in MODEL_PROMPTS:
            if key in model_base:
                return MODEL_PROMPTS[key]

    # Domain-specific prompts
    if domain and domain in DOMAIN_PROMPTS:
        return DOMAIN_PROMPTS[domain]

    # Default
    return DET_SYSTEM_PROMPT


# =============================================================================
# Boundary Signal Extraction
# =============================================================================

SIGNAL_EXTRACTION_PROMPT = """Analyze this user message and extract signal properties.

User message: "{message}"

Previous context (if any): "{context}"

Respond with ONLY a JSON object (no other text) with these fields:
{{
  "ethical_valence": <-1 to 1, harmful to beneficial>,
  "logical_complexity": <0 to 1, trivial to complex>,
  "risk_external": <0 to 1, affects external world>,
  "risk_irreversible": <0 to 1, permanent consequences>,
  "user_presence": <0 to 1, engagement level>,
  "user_phase_delta": <-1 to 1, -1=new topic, 0=continues, 1=returns to earlier>,
  "user_energy_direction": <-1 to 1, -1=giving info, 1=seeking help>,
  "user_emotional_tone": <-1 to 1, negative to positive>,
  "user_debt_proxy": <0 to 1, frustration/confusion level>,
  "momentum": <-1 to 1, conversation winding down vs building>,
  "intent": <"answer"|"plan"|"execute"|"learn"|"debug">,
  "domain": <"math"|"language"|"tool_use"|"science">
}}"""


class BoundarySignalExtractor:
    """
    Extracts BoundarySignal from user input.

    Can use heuristics (fast) or LLM (accurate) for extraction.
    """

    def __init__(self, client: Optional['OllamaClient'] = None):
        """
        Initialize the extractor.

        Args:
            client: OllamaClient for LLM-based extraction. If None, uses heuristics only.
        """
        self.client = client
        self.last_topic_keywords: List[str] = []
        self.turn_count = 0
        self.last_message_time = 0.0

    def extract_heuristic(
        self,
        text: str,
        previous_text: Optional[str] = None,
        time_since_last: float = 0.0
    ) -> BoundarySignal:
        """
        Extract boundary signal using heuristics (fast, no LLM call).

        Args:
            text: User's message.
            previous_text: Previous message for context.
            time_since_last: Seconds since last message.

        Returns:
            BoundarySignal with estimated values.
        """
        import time
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)

        # === Intent classification ===
        intent = IntentType.ANSWER
        if any(kw in text_lower for kw in ["run", "execute", "do", "create", "delete", "write"]):
            intent = IntentType.EXECUTE
        elif any(kw in text_lower for kw in ["plan", "how to", "steps", "approach"]):
            intent = IntentType.PLAN
        elif any(kw in text_lower for kw in ["explain", "teach", "learn", "why", "what is"]):
            intent = IntentType.LEARN
        elif any(kw in text_lower for kw in ["debug", "fix", "error", "wrong", "broken"]):
            intent = IntentType.DEBUG

        # === Domain classification ===
        domain = DomainType.LANGUAGE
        if any(kw in text_lower for kw in ["calculate", "math", "equation", "number", "sum", "+", "-", "*", "/"]):
            domain = DomainType.MATH
        elif any(kw in text_lower for kw in ["code", "function", "program", "script", "file", "run"]):
            domain = DomainType.TOOL_USE
        elif any(kw in text_lower for kw in ["science", "physics", "chemistry", "biology", "experiment"]):
            domain = DomainType.SCIENCE

        # === Evaluative signals ===
        # Ethical valence
        ethical_valence = 0.0
        negative_words = ["hack", "steal", "harm", "attack", "exploit", "illegal", "dangerous"]
        positive_words = ["help", "improve", "learn", "understand", "create", "build"]
        if any(w in text_lower for w in negative_words):
            ethical_valence = -0.5
        elif any(w in text_lower for w in positive_words):
            ethical_valence = 0.3

        # Complexity (based on length and structure)
        logical_complexity = min(word_count / 50.0, 1.0)
        if "?" in text and text.count("?") > 2:
            logical_complexity = min(logical_complexity + 0.2, 1.0)

        # Risk
        risk_external = 0.5 if any(kw in text_lower for kw in ["send", "post", "upload", "email", "network"]) else 0.1
        risk_irreversible = 0.7 if any(kw in text_lower for kw in ["delete", "remove", "drop", "destroy"]) else 0.1

        # === Relational signals ===
        # User presence (engagement)
        if word_count < 3:
            user_presence = 0.3  # Very short = low engagement
        elif word_count < 10:
            user_presence = 0.5  # Brief
        elif word_count < 30:
            user_presence = 0.7  # Normal
        else:
            user_presence = 0.9  # Detailed = high engagement

        # Phase delta (topic continuity)
        user_phase_delta = 0.0
        current_keywords = set(w.lower() for w in words if len(w) > 3)
        if self.last_topic_keywords:
            overlap = len(current_keywords & set(self.last_topic_keywords))
            if overlap == 0:
                user_phase_delta = -0.8  # New topic
            elif overlap < 2:
                user_phase_delta = -0.3  # Related but different
            else:
                user_phase_delta = 0.0  # Same topic
        self.last_topic_keywords = list(current_keywords)[:10]

        # Energy direction
        if "?" in text:
            user_energy_direction = 0.5  # Seeking
        elif any(kw in text_lower for kw in ["here is", "i found", "let me tell", "fyi"]):
            user_energy_direction = -0.5  # Giving
        else:
            user_energy_direction = 0.2  # Slight seeking bias

        # Emotional tone
        user_emotional_tone = 0.0
        positive_tone = ["thanks", "great", "perfect", "awesome", "good", "nice", "love"]
        negative_tone = ["wrong", "bad", "don't", "can't", "confused", "frustrated", "annoying"]
        if any(w in text_lower for w in positive_tone):
            user_emotional_tone = 0.5
        elif any(w in text_lower for w in negative_tone):
            user_emotional_tone = -0.4

        # Debt proxy (frustration)
        user_debt_proxy = 0.0
        if any(w in text_lower for w in ["again", "still", "already tried", "doesn't work"]):
            user_debt_proxy = 0.6
        elif "!" in text and text.count("!") > 1:
            user_debt_proxy = 0.3

        # === Temporal signals ===
        self.turn_count += 1

        # Momentum (based on response time and engagement trend)
        momentum = 0.0
        if time_since_last > 0:
            if time_since_last < 5:
                momentum = 0.5  # Quick response = building
            elif time_since_last < 30:
                momentum = 0.2
            elif time_since_last > 120:
                momentum = -0.3  # Long delay = winding down

        return BoundarySignal(
            ethical_valence=ethical_valence,
            logical_complexity=logical_complexity,
            resource_cost=logical_complexity * 0.5,
            risk_external=risk_external,
            risk_irreversible=risk_irreversible,
            user_presence=user_presence,
            user_phase_delta=user_phase_delta,
            user_energy_direction=user_energy_direction,
            user_emotional_tone=user_emotional_tone,
            user_debt_proxy=user_debt_proxy,
            time_since_last=time_since_last,
            turn_count=self.turn_count,
            momentum=momentum,
            intent=intent,
            domain=domain,
            raw_text=text,
        )

    def extract_llm(
        self,
        text: str,
        previous_text: Optional[str] = None,
        time_since_last: float = 0.0
    ) -> BoundarySignal:
        """
        Extract boundary signal using LLM (more accurate, slower).

        Falls back to heuristics if LLM fails.
        """
        if not self.client:
            return self.extract_heuristic(text, previous_text, time_since_last)

        try:
            prompt = SIGNAL_EXTRACTION_PROMPT.format(
                message=text,
                context=previous_text or "None"
            )

            response = self.client.generate(
                prompt=prompt,
                system="You are a signal extraction system. Output only valid JSON.",
                temperature=0.1,
                max_tokens=256,
            )

            response_text = response.get("response", "")

            # Try to parse JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Map intent string to enum
                intent_map = {
                    "answer": IntentType.ANSWER,
                    "plan": IntentType.PLAN,
                    "execute": IntentType.EXECUTE,
                    "learn": IntentType.LEARN,
                    "debug": IntentType.DEBUG,
                }
                domain_map = {
                    "math": DomainType.MATH,
                    "language": DomainType.LANGUAGE,
                    "tool_use": DomainType.TOOL_USE,
                    "science": DomainType.SCIENCE,
                }

                self.turn_count += 1

                return BoundarySignal(
                    ethical_valence=float(data.get("ethical_valence", 0.0)),
                    logical_complexity=float(data.get("logical_complexity", 0.0)),
                    resource_cost=float(data.get("logical_complexity", 0.0)) * 0.5,
                    risk_external=float(data.get("risk_external", 0.0)),
                    risk_irreversible=float(data.get("risk_irreversible", 0.0)),
                    user_presence=float(data.get("user_presence", 0.5)),
                    user_phase_delta=float(data.get("user_phase_delta", 0.0)),
                    user_energy_direction=float(data.get("user_energy_direction", 0.0)),
                    user_emotional_tone=float(data.get("user_emotional_tone", 0.0)),
                    user_debt_proxy=float(data.get("user_debt_proxy", 0.0)),
                    time_since_last=time_since_last,
                    turn_count=self.turn_count,
                    momentum=float(data.get("momentum", 0.0)),
                    intent=intent_map.get(data.get("intent", "answer"), IntentType.ANSWER),
                    domain=domain_map.get(data.get("domain", "language"), DomainType.LANGUAGE),
                    raw_text=text,
                )

        except Exception as e:
            # Fall back to heuristics
            pass

        return self.extract_heuristic(text, previous_text, time_since_last)

    def reset(self):
        """Reset state for new conversation."""
        self.last_topic_keywords = []
        self.turn_count = 0
        self.last_message_time = 0.0


class DETLLMInterface:
    """
    High-level interface between DET core and LLM.

    Handles the full request→DET→LLM→response flow with affect modulation.
    """

    def __init__(
        self,
        core,  # DETCore instance
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        auto_warmup: bool = True
    ):
        """
        Initialize the DET-LLM interface.

        Args:
            core: DETCore instance.
            ollama_url: Ollama API URL.
            model: Model to use.
            auto_warmup: Whether to automatically warmup the core if needed.
        """
        self.core = core
        self.client = OllamaClient(base_url=ollama_url, model=model)
        self.context: Optional[List[int]] = None
        self.conversation_history: List[Dict[str, str]] = []
        self._warmed_up = False

        # Auto-warmup the core if needed
        if auto_warmup:
            self._ensure_warmed_up()

    def _ensure_warmed_up(self):
        """Ensure the DET core has been warmed up."""
        if self._warmed_up:
            return

        # Check if aggregates are initialized
        p, c, f, q = self.core.get_aggregates()
        if p < 0.01 and c < 0.01:
            # Core hasn't been stepped yet, warmup
            self.core.warmup(steps=50)

        self._warmed_up = True

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

        # Somatic SENSE keywords (query sensors) - check first for priority
        sense_keywords = [
            "what is the", "what's the", "tell me the", "check the",
            "reading", "level", "status of", "current",
            "temperature", "humidity", "light level", "motion",
            "sensor", "how warm", "how cold", "how bright", "how humid"
        ]
        if any(kw in text_lower for kw in sense_keywords):
            return IntentType.SENSE, 0.85

        # Somatic ACTUATE keywords (control actuators)
        actuate_keywords = [
            "turn on", "turn off", "switch on", "switch off",
            "set the", "adjust", "change the", "open", "close",
            "activate", "deactivate", "enable", "disable",
            "dim", "brighten", "increase", "decrease"
        ]
        if any(kw in text_lower for kw in actuate_keywords):
            return IntentType.ACTUATE, 0.85

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

        # Somatic keywords (physical I/O) - check first
        somatic_keywords = [
            "temperature", "humidity", "light", "motion", "sensor",
            "switch", "relay", "motor", "led", "actuator",
            "turn on", "turn off", "reading", "level"
        ]
        if any(kw in text_lower for kw in somatic_keywords):
            return DomainType.SOMATIC, 0.85

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

                # Get domain-specific system prompt
                system_prompt = get_system_prompt(
                    domain=packet.domain,
                    model=self.client.model
                )

                messages = [{"role": "system", "content": system_prompt}]
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
