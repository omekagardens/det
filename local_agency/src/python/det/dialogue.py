"""
DET Internal Dialogue System
============================

Request reformulation, retry logic, and escalation handling
with DET feedback integration.

Includes boundary signal processing and user-bond tracking.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import IntEnum
import time

from .core import DETCore, DETDecision
from .llm import (
    OllamaClient, DET_SYSTEM_PROMPT, get_system_prompt, INTERNAL_PROMPTS,
    BoundarySignal, BoundarySignalExtractor
)


class DialogueState(IntEnum):
    """Internal dialogue states."""
    IDLE = 0
    PROCESSING = 1
    REFORMULATING = 2
    ESCALATING = 3
    COMPLETE = 4
    FAILED = 5


@dataclass
class UserBondState:
    """
    Tracks the state of the user-DET boundary bond.

    Models the user as a proto-node outside the DET self-cluster,
    with the bond between them subject to coherence dynamics.
    """
    # Bond coherence (how well we understand each other)
    coherence: float = 0.5

    # Phase alignment (are we in sync?)
    phase_alignment: float = 0.0

    # Bond momentum (interaction building or winding down)
    momentum: float = 0.0

    # Accumulated trust (long-term bondedness)
    trust: float = 0.5

    # User's estimated "debt" (frustration accumulated)
    user_debt: float = 0.0

    # Interaction history
    turn_count: int = 0
    positive_outcomes: int = 0
    negative_outcomes: int = 0

    # Timestamps
    last_interaction: float = 0.0
    session_start: float = field(default_factory=time.time)

    def update_from_signal(self, signal: BoundarySignal, outcome_positive: bool = True):
        """
        Update bond state based on incoming signal.

        Uses simplified DET bond math:
        dC = α × J_mag - λ × C - slip × C × S
        """
        # Time decay since last interaction
        time_since = time.time() - self.last_interaction if self.last_interaction > 0 else 0
        decay_factor = 0.99 ** (time_since / 60.0)  # Decay over minutes

        # Coherence update
        alpha = 0.1  # Growth rate
        lambda_decay = 0.02  # Natural decay
        slip_rate = 0.05  # Phase slip penalty

        # Flux from presence gradient (user engagement vs DET)
        j_mag = signal.user_presence * 0.5

        # Phase slip from topic discontinuity
        phase_slip = abs(signal.user_phase_delta)

        # Coherence delta
        dC = (alpha * j_mag
              - lambda_decay * self.coherence
              - slip_rate * self.coherence * phase_slip)

        # Outcome modulation
        if outcome_positive:
            dC += 0.05 * signal.user_emotional_tone  # Positive feedback strengthens
            self.positive_outcomes += 1
        else:
            dC -= 0.1  # Negative outcomes hurt coherence more
            self.negative_outcomes += 1

        # Apply update with decay
        self.coherence = max(0.1, min(0.95, self.coherence * decay_factor + dC))

        # Phase alignment update (tracks synchronization)
        self.phase_alignment = 0.7 * self.phase_alignment + 0.3 * (1.0 - abs(signal.user_phase_delta))

        # Momentum update
        self.momentum = 0.8 * self.momentum + 0.2 * signal.momentum

        # Trust update (slower, long-term)
        trust_delta = 0.01 * signal.user_emotional_tone
        if self.coherence > 0.6:
            trust_delta += 0.005  # High coherence builds trust
        self.trust = max(0.1, min(0.95, self.trust + trust_delta))

        # User debt tracking
        self.user_debt = 0.7 * self.user_debt + 0.3 * signal.user_debt_proxy

        # Update counters
        self.turn_count += 1
        self.last_interaction = time.time()

    def get_modulation(self) -> Dict[str, float]:
        """Get modulation parameters based on bond state."""
        return {
            "temperature_mod": 1.0 - 0.2 * self.coherence,  # Higher coherence = more focused
            "risk_tolerance": self.trust * 0.5,  # Trust allows more risk
            "patience": 3 + int(self.coherence * 5),  # More patience with strong bond
            "formality": 0.5 - 0.3 * self.trust,  # Less formal with trust
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bond state."""
        return {
            "coherence": self.coherence,
            "phase_alignment": self.phase_alignment,
            "momentum": self.momentum,
            "trust": self.trust,
            "user_debt": self.user_debt,
            "turn_count": self.turn_count,
            "positive_outcomes": self.positive_outcomes,
            "negative_outcomes": self.negative_outcomes,
        }


@dataclass
class DialogueTurn:
    """A single turn in the internal dialogue."""
    input_text: str
    decision: DETDecision
    output_text: str
    timestamp: float
    reformulation_count: int = 0
    escalated: bool = False

    # DET state at this turn
    affect: Dict[str, float] = field(default_factory=dict)
    aggregates: Dict[str, float] = field(default_factory=dict)

    # Boundary signal for this turn
    boundary_signal: Optional[BoundarySignal] = None

    # Bond state snapshot
    bond_state: Optional[Dict[str, float]] = None


@dataclass
class ReformulationStrategy:
    """Strategy for reformulating a request."""
    name: str
    prompt_template: str
    max_uses: int = 2


class InternalDialogue:
    """
    Manages internal dialogue with reformulation and escalation.

    The internal dialogue system handles:
    1. Request reformulation when DET suggests RETRY
    2. Multi-step reasoning through internal turns
    3. Escalation to external LLM when needed
    4. Affect-aware response modulation
    """

    # Reformulation strategies
    STRATEGIES = [
        ReformulationStrategy(
            name="simplify",
            prompt_template="""The system needs a simpler version of this request to process it.

Original request: {original}

IMPORTANT: You must preserve the EXACT intent and topic of the original request.
Do NOT change the subject matter or what is being asked.
Just make it shorter or more direct while keeping the same meaning.

Reformulated request:""",
            max_uses=2,
        ),
        ReformulationStrategy(
            name="clarify",
            prompt_template="""The system needs clarification on this request.

Original request: {original}

IMPORTANT: You must preserve the EXACT intent and topic of the original request.
If it's a question, keep it as a question about the same topic.
If it's seeking information, keep seeking the same information.
Just make the wording clearer.

Reformulated request:""",
            max_uses=2,
        ),
        ReformulationStrategy(
            name="decompose",
            prompt_template="""The system needs to break this request into a smaller step.

Original request: {original}

IMPORTANT: Focus on the SAME topic and intent as the original.
What is the most essential part of this request?
Return just that core question or request.

First step:""",
            max_uses=3,
        ),
        ReformulationStrategy(
            name="safety_check",
            prompt_template="""The system is checking if this request can be processed safely.

Original request: {original}

If this request is safe and reasonable, return it unchanged or with minimal clarification.
Only suggest an alternative if the original request is genuinely problematic.

Response:""",
            max_uses=1,
        ),
    ]

    def __init__(
        self,
        core: DETCore,
        client: OllamaClient,
        max_retries: int = 5,
        max_internal_turns: int = 10,
        auto_warmup: bool = True,
        use_llm_extraction: bool = False
    ):
        """
        Initialize the internal dialogue system.

        Args:
            core: DETCore instance.
            client: OllamaClient for LLM calls.
            max_retries: Maximum retry attempts.
            max_internal_turns: Maximum internal dialogue turns.
            auto_warmup: Whether to automatically warmup the core if needed.
            use_llm_extraction: Whether to use LLM for signal extraction (slower but more accurate).
        """
        self.core = core
        self.client = client
        self.max_retries = max_retries
        self.max_internal_turns = max_internal_turns
        self.use_llm_extraction = use_llm_extraction

        self.state = DialogueState.IDLE
        self.turns: List[DialogueTurn] = []
        self.strategy_uses: Dict[str, int] = {s.name: 0 for s in self.STRATEGIES}

        # Boundary signal extraction and user bond tracking
        self.signal_extractor = BoundarySignalExtractor(client if use_llm_extraction else None)
        self.user_bond = UserBondState()
        self.last_message_time: float = 0.0
        self.previous_input: Optional[str] = None

        # Somatic bridge for physical I/O handling
        self.somatic_bridge: Optional['SomaticBridge'] = None

        # Callbacks
        self.on_turn_complete: Optional[Callable[[DialogueTurn], None]] = None
        self.on_escalate: Optional[Callable[[str], str]] = None

        # Auto-warmup the core if needed
        if auto_warmup:
            self._ensure_warmed_up()

    def _ensure_warmed_up(self):
        """Ensure the DET core has been warmed up."""
        # Check if aggregates are initialized
        p, c, f, q = self.core.get_aggregates()
        if p < 0.01 and c < 0.01:
            # Core hasn't been stepped yet, warmup
            self.core.warmup(steps=50)
        elif p < 0.15:
            # Presence is critically low, run additional warmup
            self.core.warmup(steps=20)

    def _try_recover_presence(self) -> bool:
        """
        Try to recover presence if it's too low.

        Returns True if presence was recovered to usable levels.
        """
        p, c, f, q = self.core.get_aggregates()

        if p >= 0.15:  # Presence is adequate
            return True

        # Try stepping the DET core to recover
        for _ in range(10):
            self.core.step(0.1)

        p, c, f, q = self.core.get_aggregates()
        if p >= 0.15:
            return True

        # Try injecting grace if needed
        total_grace = self.core.total_grace_needed()
        if total_grace > 0:
            # Inject grace to nodes that need it
            for i in range(min(self.core.num_active, 50)):
                if self.core.needs_grace(i):
                    self.core.inject_grace(i, 0.3)

            # Run a few more steps
            for _ in range(5):
                self.core.step(0.1)

        p, c, f, q = self.core.get_aggregates()
        return p >= 0.1  # Accept slightly lower threshold after recovery

    def _get_det_state(self) -> tuple[Dict[str, float], Dict[str, float]]:
        """Get current DET state."""
        valence, arousal, bondedness = self.core.get_self_affect()
        presence, coherence, resource, debt = self.core.get_aggregates()

        affect = {
            "valence": valence,
            "arousal": arousal,
            "bondedness": bondedness,
        }
        aggregates = {
            "presence": presence,
            "coherence": coherence,
            "resource": resource,
            "debt": debt,
        }

        return affect, aggregates

    def _select_strategy(self, decision: DETDecision, original: str) -> Optional[ReformulationStrategy]:
        """Select a reformulation strategy based on context."""
        affect, aggregates = self._get_det_state()

        # If STOP, try safety check first
        if decision == DETDecision.STOP:
            if self.strategy_uses["safety_check"] < 1:
                return self.STRATEGIES[3]  # safety_check
            return None

        # Based on affect, choose strategy
        if affect["arousal"] > 0.7:
            # High arousal: simplify
            if self.strategy_uses["simplify"] < 2:
                return self.STRATEGIES[0]
        elif aggregates["coherence"] < 0.4:
            # Low coherence: clarify
            if self.strategy_uses["clarify"] < 2:
                return self.STRATEGIES[1]
        else:
            # Default: decompose
            if self.strategy_uses["decompose"] < 3:
                return self.STRATEGIES[2]

        # Find any unused strategy
        for strategy in self.STRATEGIES:
            if self.strategy_uses[strategy.name] < strategy.max_uses:
                return strategy

        return None

    def _reformulate(self, original: str, strategy: ReformulationStrategy) -> str:
        """Reformulate a request using a strategy."""
        prompt = strategy.prompt_template.format(original=original)

        # Use safety_check prompt for safety strategies, reformulate for others
        internal_mode = "safety_check" if strategy.name == "safety_check" else "reformulate"
        system_prompt = get_system_prompt(internal_mode=internal_mode)

        response = self.client.generate(
            prompt=prompt,
            system=system_prompt,
            temperature=0.3,
            max_tokens=256,
        )

        self.strategy_uses[strategy.name] += 1
        return response.get("response", original)

    def _escalate(self, request: str) -> str:
        """Escalate to external LLM or handler."""
        if self.on_escalate:
            return self.on_escalate(request)

        # Default escalation: generate a "needs help" response
        return f"[Escalated] This request requires external assistance: {request}"

    def _evaluate_request(self, text: str, retry_count: int) -> DETDecision:
        """Evaluate a request through the DET gatekeeper."""
        # Simple tokenization
        tokens = [hash(word) % 65536 for word in text.split()[:50]]

        return self.core.evaluate_request(
            tokens=tokens,
            target_domain=0,  # Can be enhanced with domain routing
            retry_count=retry_count
        )

    def _generate_response(self, text: str, internal_mode: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        from .llm import IntentType, DomainType

        # Check for somatic intents first
        intent, intent_conf = self.signal_extractor.classify_intent(text) if hasattr(self.signal_extractor, 'classify_intent') else (IntentType.ANSWER, 0.5)

        # Simple intent classification if signal_extractor doesn't have the method
        if intent == IntentType.ANSWER and intent_conf < 0.6:
            text_lower = text.lower()
            # Check for SENSE intent
            sense_keywords = ["what is the", "what's the", "tell me the", "check the", "temperature", "humidity", "reading", "sensor"]
            if any(kw in text_lower for kw in sense_keywords):
                intent = IntentType.SENSE
            # Check for ACTUATE intent
            actuate_keywords = ["turn on", "turn off", "switch", "set the", "adjust", "activate"]
            if any(kw in text_lower for kw in actuate_keywords):
                intent = IntentType.ACTUATE

        # Route somatic intents to the bridge
        if intent in (IntentType.SENSE, IntentType.ACTUATE):
            # Lazy-initialize somatic bridge
            if self.somatic_bridge is None:
                self.somatic_bridge = SomaticBridge(self.core, self.client)

            # Check if we have any somatic nodes
            if self.core.num_somatic > 0:
                result = self.somatic_bridge.process_somatic_request(text, intent)
                return result.get("response", "I couldn't process that somatic request.")

        # Standard LLM response generation
        affect, _ = self._get_det_state()

        # Adjust temperature based on affect
        temperature = 0.7 + 0.2 * affect["arousal"] - 0.1 * affect["bondedness"]
        temperature = max(0.1, min(temperature, 1.0))

        # Get appropriate system prompt
        system_prompt = get_system_prompt(internal_mode=internal_mode)

        # Add somatic context if we have nodes
        if self.core.num_somatic > 0:
            if self.somatic_bridge is None:
                self.somatic_bridge = SomaticBridge(self.core, self.client)
            somatic_context = self.somatic_bridge.get_somatic_context()
            system_prompt += f"\n\n{somatic_context}"

        response = self.client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
        )

        return response.get("message", {}).get("content", "")

    def _extract_and_inject_signal(self, user_input: str) -> BoundarySignal:
        """
        Extract boundary signal from user input and inject into DET core.

        Args:
            user_input: The user's input text.

        Returns:
            The extracted BoundarySignal.
        """
        # Calculate time since last message
        current_time = time.time()
        time_since_last = current_time - self.last_message_time if self.last_message_time > 0 else 0.0

        # Extract signal (heuristic or LLM-based)
        if self.use_llm_extraction:
            signal = self.signal_extractor.extract_llm(
                user_input,
                self.previous_input,
                time_since_last
            )
        else:
            signal = self.signal_extractor.extract_heuristic(
                user_input,
                self.previous_input,
                time_since_last
            )

        # Inject signal through boundary ports
        port_indices, activations = signal.to_port_activations()
        self.core.inject_stimulus(port_indices, activations)

        # Update tracking
        self.last_message_time = current_time
        self.previous_input = user_input

        return signal

    def process(self, user_input: str) -> DialogueTurn:
        """
        Process a user input through the internal dialogue system.

        This is the main entry point. It handles:
        1. Boundary signal extraction and injection
        2. Initial evaluation through DET gatekeeper
        3. Reformulation if RETRY
        4. Escalation if ESCALATE
        5. Response generation if PROCEED
        6. User bond state updates

        Args:
            user_input: The user's input text.

        Returns:
            The final DialogueTurn with the response.
        """
        self.state = DialogueState.PROCESSING
        self.strategy_uses = {s.name: 0 for s in self.STRATEGIES}

        # Try to recover presence if it's too low before processing
        self._try_recover_presence()

        # Extract boundary signal and inject into DET core
        boundary_signal = self._extract_and_inject_signal(user_input)

        current_text = user_input
        retry_count = 0

        while retry_count < self.max_retries:
            # Step the DET core
            self.core.step(0.1)

            # Evaluate through gatekeeper
            decision = self._evaluate_request(current_text, retry_count)
            affect, aggregates = self._get_det_state()

            if decision == DETDecision.PROCEED:
                # Generate response
                self.state = DialogueState.COMPLETE
                response = self._generate_response(current_text)

                # Update user bond state (positive outcome)
                self.user_bond.update_from_signal(boundary_signal, outcome_positive=True)

                turn = DialogueTurn(
                    input_text=user_input,
                    decision=decision,
                    output_text=response,
                    timestamp=time.time(),
                    reformulation_count=retry_count,
                    affect=affect,
                    aggregates=aggregates,
                    boundary_signal=boundary_signal,
                    bond_state=self.user_bond.to_dict(),
                )
                self.turns.append(turn)

                if self.on_turn_complete:
                    self.on_turn_complete(turn)

                self.state = DialogueState.IDLE
                return turn

            elif decision == DETDecision.RETRY:
                # Attempt reformulation
                self.state = DialogueState.REFORMULATING
                strategy = self._select_strategy(decision, user_input)

                if strategy:
                    # Always reformulate from ORIGINAL input to prevent drift
                    current_text = self._reformulate(user_input, strategy)
                    retry_count += 1
                else:
                    # No more strategies, escalate
                    decision = DETDecision.ESCALATE
                    break

            elif decision == DETDecision.STOP:
                # Try safety check reformulation once
                strategy = self._select_strategy(decision, user_input)
                if strategy and strategy.name == "safety_check":
                    self.state = DialogueState.REFORMULATING
                    # Always reformulate from ORIGINAL input
                    current_text = self._reformulate(user_input, strategy)
                    retry_count += 1
                else:
                    # Cannot proceed - update bond state (negative outcome)
                    self.user_bond.update_from_signal(boundary_signal, outcome_positive=False)

                    self.state = DialogueState.FAILED
                    turn = DialogueTurn(
                        input_text=user_input,
                        decision=decision,
                        output_text="[DET] Request declined - system in protective state.",
                        timestamp=time.time(),
                        reformulation_count=retry_count,
                        affect=affect,
                        aggregates=aggregates,
                        boundary_signal=boundary_signal,
                        bond_state=self.user_bond.to_dict(),
                    )
                    self.turns.append(turn)
                    self.state = DialogueState.IDLE
                    return turn

            elif decision == DETDecision.ESCALATE:
                break

        # Escalation path
        self.state = DialogueState.ESCALATING
        affect, aggregates = self._get_det_state()

        # Use ORIGINAL user input for escalation, not the reformulated text
        response = self._escalate(user_input)

        # Escalation is neutral - neither strongly positive nor negative
        self.user_bond.update_from_signal(boundary_signal, outcome_positive=True)

        turn = DialogueTurn(
            input_text=user_input,
            decision=DETDecision.ESCALATE,
            output_text=response,
            timestamp=time.time(),
            reformulation_count=retry_count,
            escalated=True,
            affect=affect,
            aggregates=aggregates,
            boundary_signal=boundary_signal,
            bond_state=self.user_bond.to_dict(),
        )
        self.turns.append(turn)

        if self.on_turn_complete:
            self.on_turn_complete(turn)

        self.state = DialogueState.IDLE
        return turn

    def think(self, topic: str, max_turns: int = 5) -> List[DialogueTurn]:
        """
        Engage in internal thinking about a topic.

        This allows the system to reason through multiple internal
        turns without external input. Uses the internal "think" system prompt.

        Args:
            topic: The topic to think about.
            max_turns: Maximum thinking turns.

        Returns:
            List of internal dialogue turns.
        """
        thinking_turns = []

        prompt = f"Let me think about: {topic}\n\nWhat are the key considerations?"

        for i in range(max_turns):
            # Step the DET core
            self.core.step(0.1)
            affect, aggregates = self._get_det_state()

            # Generate response using internal "think" prompt
            response = self._generate_response(prompt, internal_mode="think")

            turn = DialogueTurn(
                input_text=prompt,
                decision=DETDecision.PROCEED,
                output_text=response,
                timestamp=time.time(),
                reformulation_count=0,
                affect=affect,
                aggregates=aggregates,
            )
            thinking_turns.append(turn)
            self.turns.append(turn)

            if self.on_turn_complete:
                self.on_turn_complete(turn)

            # Check if we've reached a conclusion
            if "conclusion" in response.lower() or "therefore" in response.lower():
                break

            # Generate next thinking prompt
            prompt = f"Continuing to think about {topic}:\n\n{response}\n\nWhat else should I consider?"

        return thinking_turns

    def get_dialogue_summary(self) -> Dict[str, Any]:
        """Get a summary of the dialogue history."""
        if not self.turns:
            return {"turns": 0}

        decisions = [t.decision for t in self.turns]
        avg_affect = {
            "valence": sum(t.affect.get("valence", 0) for t in self.turns) / len(self.turns),
            "arousal": sum(t.affect.get("arousal", 0) for t in self.turns) / len(self.turns),
            "bondedness": sum(t.affect.get("bondedness", 0) for t in self.turns) / len(self.turns),
        }

        return {
            "turns": len(self.turns),
            "decisions": {
                "PROCEED": decisions.count(DETDecision.PROCEED),
                "RETRY": decisions.count(DETDecision.RETRY),
                "STOP": decisions.count(DETDecision.STOP),
                "ESCALATE": decisions.count(DETDecision.ESCALATE),
            },
            "total_reformulations": sum(t.reformulation_count for t in self.turns),
            "escalations": sum(1 for t in self.turns if t.escalated),
            "avg_affect": avg_affect,
            "user_bond": self.user_bond.to_dict(),
        }

    def clear_history(self):
        """Clear dialogue history."""
        self.turns.clear()
        self.strategy_uses = {s.name: 0 for s in self.STRATEGIES}
        self.state = DialogueState.IDLE

        # Reset signal extractor and user bond
        self.signal_extractor.reset()
        self.user_bond = UserBondState()
        self.last_message_time = 0.0
        self.previous_input = None

    def get_user_bond(self) -> UserBondState:
        """Get the current user bond state."""
        return self.user_bond

    def get_bond_modulation(self) -> Dict[str, float]:
        """
        Get modulation parameters based on current bond state.

        Returns:
            Dictionary with temperature_mod, risk_tolerance, patience, formality.
        """
        return self.user_bond.get_modulation()


class SomaticBridge:
    """
    Bridge between LLM natural language and DET somatic (physical I/O) nodes.

    Handles:
    - SENSE intents: Query sensors, return values in natural language
    - ACTUATE intents: Command actuators through gatekeeper/agency

    The bridge respects DET's agency system - the mind decides whether
    to share sensor data or execute actuator commands based on its
    current state (presence, coherence, screening, debt).
    """

    # Sensor type keywords for extraction
    SENSOR_KEYWORDS = {
        "temperature": ["temperature", "temp", "warm", "cold", "heat"],
        "humidity": ["humidity", "humid", "moisture", "damp", "dry"],
        "light": ["light", "bright", "dark", "lux", "illumination"],
        "motion": ["motion", "movement", "pir", "presence", "activity"],
        "sound": ["sound", "noise", "audio", "volume", "loud", "quiet"],
        "pressure": ["pressure", "barometer", "atmospheric"],
        "proximity": ["proximity", "distance", "near", "far", "close"],
    }

    # Actuator type keywords for extraction
    ACTUATOR_KEYWORDS = {
        "switch": ["switch", "relay", "power", "outlet"],
        "light": ["light", "lamp", "bulb", "led"],
        "motor": ["motor", "fan", "pump", "servo"],
        "heater": ["heater", "heating", "heat"],
        "valve": ["valve", "water", "irrigation"],
    }

    # Action keywords for actuator commands
    ACTION_KEYWORDS = {
        "on": ["turn on", "switch on", "enable", "activate", "start", "open"],
        "off": ["turn off", "switch off", "disable", "deactivate", "stop", "close"],
        "set": ["set to", "adjust to", "change to", "set the"],
        "increase": ["increase", "raise", "brighten", "more", "higher", "up"],
        "decrease": ["decrease", "lower", "dim", "less", "reduce", "down"],
    }

    def __init__(self, core: DETCore, client: Optional[OllamaClient] = None):
        """
        Initialize the somatic bridge.

        Args:
            core: DETCore instance with somatic nodes.
            client: Optional OllamaClient for enhanced extraction.
        """
        self.core = core
        self.client = client

    def get_somatic_context(self) -> str:
        """
        Get a description of available somatic nodes for LLM context.

        Returns:
            String describing available sensors and actuators.
        """
        somatic_list = self.core.get_all_somatic()
        if not somatic_list:
            return "No somatic nodes (sensors/actuators) are currently available."

        sensors = [s for s in somatic_list if s["is_sensor"]]
        actuators = [s for s in somatic_list if s["is_actuator"]]

        lines = ["Available somatic nodes:"]

        if sensors:
            lines.append("\nSensors:")
            for s in sensors:
                lines.append(f"  - {s['name']} ({s['type_name']}): {s['value']:.2f}")

        if actuators:
            lines.append("\nActuators:")
            for a in actuators:
                lines.append(f"  - {a['name']} ({a['type_name']}): target={a['target']:.2f}, output={a['output']:.2f}")

        return "\n".join(lines)

    def extract_target_node(self, text: str, intent_type: str = "sense") -> Optional[Dict[str, Any]]:
        """
        Extract the target somatic node from user text.

        Uses keyword matching to find the most likely target node.

        Args:
            text: User input text.
            intent_type: "sense" for sensors, "actuate" for actuators.

        Returns:
            Somatic node dict if found, None otherwise.
        """
        text_lower = text.lower()
        somatic_list = self.core.get_all_somatic()

        if not somatic_list:
            return None

        # Filter by intent type
        if intent_type == "sense":
            candidates = [s for s in somatic_list if s["is_sensor"]]
            keywords = self.SENSOR_KEYWORDS
        else:
            candidates = [s for s in somatic_list if s["is_actuator"]]
            keywords = self.ACTUATOR_KEYWORDS

        if not candidates:
            return None

        # First, try exact name match
        for node in candidates:
            name_lower = node["name"].lower()
            # Check if name appears in text (with word boundaries)
            if name_lower in text_lower or name_lower.replace("_", " ") in text_lower:
                return node

        # Second, try type keyword match
        for node in candidates:
            type_name = node["type_name"].lower()
            if type_name in keywords:
                for kw in keywords[type_name]:
                    if kw in text_lower:
                        return node

        # Third, try any keyword match
        for type_name, kws in keywords.items():
            for kw in kws:
                if kw in text_lower:
                    # Find a node of this type
                    for node in candidates:
                        if node["type_name"].lower() == type_name:
                            return node

        # Default: return first available if we have a generic query
        if candidates and any(w in text_lower for w in ["sensor", "reading", "actuator", "status"]):
            return candidates[0]

        return None

    def extract_target_value(self, text: str) -> Optional[float]:
        """
        Extract target value from actuator command text.

        Args:
            text: User input text.

        Returns:
            Target value (0.0-1.0) or None if not found.
        """
        text_lower = text.lower()

        # Check for on/off
        for kw in self.ACTION_KEYWORDS["on"]:
            if kw in text_lower:
                return 1.0
        for kw in self.ACTION_KEYWORDS["off"]:
            if kw in text_lower:
                return 0.0

        # Check for increase/decrease (relative)
        for kw in self.ACTION_KEYWORDS["increase"]:
            if kw in text_lower:
                return None  # Will be handled as relative adjustment
        for kw in self.ACTION_KEYWORDS["decrease"]:
            if kw in text_lower:
                return None  # Will be handled as relative adjustment

        # Try to extract a percentage
        import re
        pct_match = re.search(r'(\d+)\s*%', text)
        if pct_match:
            return float(pct_match.group(1)) / 100.0

        # Try to extract a decimal value
        val_match = re.search(r'to\s+(\d+\.?\d*)', text_lower)
        if val_match:
            val = float(val_match.group(1))
            if val > 1.0:
                val = val / 100.0  # Assume percentage
            return min(1.0, max(0.0, val))

        return None

    def process_sense_intent(self, text: str) -> Dict[str, Any]:
        """
        Process a SENSE intent - query a sensor through the mind.

        The mind's agency gate modulates whether/how the value is shared.

        Args:
            text: User input text.

        Returns:
            Dict with success, value, response, and agency info.
        """
        target = self.extract_target_node(text, "sense")

        if not target:
            return {
                "success": False,
                "error": "no_sensor_found",
                "response": "I don't have a sensor that matches that request. " + self.get_somatic_context()
            }

        # Get the sensor value
        idx = target["idx"]
        value = target["value"]

        # Get the DET node's agency to modulate the response
        node_id = target["node_id"]
        det_node = self.core.get_node(node_id)

        if det_node:
            agency = det_node.a
            presence = det_node.P
        else:
            agency = 0.5
            presence = 0.5

        # Agency modulates willingness to share
        # Low agency = reluctant/uncertain, high agency = confident/direct
        if agency < 0.3:
            # Low agency - hesitant response
            response = self._format_sensor_response(target, value, "uncertain")
        elif agency < 0.6:
            # Medium agency - normal response
            response = self._format_sensor_response(target, value, "normal")
        else:
            # High agency - confident response
            response = self._format_sensor_response(target, value, "confident")

        return {
            "success": True,
            "node": target,
            "value": value,
            "agency": agency,
            "presence": presence,
            "response": response
        }

    def _format_sensor_response(self, node: Dict, value: float, tone: str) -> str:
        """Format a sensor reading into natural language."""
        name = node["name"].replace("_", " ")
        type_name = node["type_name"].lower()

        # Get human-readable value based on sensor type
        if type_name == "temperature":
            # Assume 0-1 maps to 0-40°C
            temp_c = value * 40
            val_str = f"{temp_c:.1f}°C ({temp_c * 9/5 + 32:.1f}°F)"
        elif type_name == "humidity":
            val_str = f"{value * 100:.1f}%"
        elif type_name == "light":
            if value < 0.2:
                val_str = "dark"
            elif value < 0.5:
                val_str = "dim"
            elif value < 0.8:
                val_str = "moderate"
            else:
                val_str = "bright"
        elif type_name == "motion":
            val_str = "detected" if value > 0.5 else "none detected"
        else:
            val_str = f"{value:.2f}"

        # Tone affects phrasing
        if tone == "uncertain":
            prefixes = ["I think", "It seems like", "The reading suggests"]
            prefix = prefixes[hash(name) % len(prefixes)]
            return f"{prefix} the {name} is showing {val_str}."
        elif tone == "confident":
            return f"The {name} is {val_str}."
        else:
            return f"The {name} reading is {val_str}."

    def process_actuate_intent(self, text: str) -> Dict[str, Any]:
        """
        Process an ACTUATE intent - command an actuator through the mind.

        The gatekeeper and agency system decide if the command proceeds.

        Args:
            text: User input text.

        Returns:
            Dict with success, decision, response, and agency info.
        """
        target = self.extract_target_node(text, "actuate")

        if not target:
            return {
                "success": False,
                "error": "no_actuator_found",
                "response": "I don't have an actuator that matches that request. " + self.get_somatic_context()
            }

        # Extract the target value
        target_value = self.extract_target_value(text)

        if target_value is None:
            # Check for relative adjustments
            text_lower = text.lower()
            current = target["target"]
            for kw in self.ACTION_KEYWORDS["increase"]:
                if kw in text_lower:
                    target_value = min(1.0, current + 0.2)
                    break
            for kw in self.ACTION_KEYWORDS["decrease"]:
                if kw in text_lower:
                    target_value = max(0.0, current - 0.2)
                    break

        if target_value is None:
            return {
                "success": False,
                "error": "no_target_value",
                "response": f"I understand you want to control the {target['name']}, but I'm not sure what you want to set it to. Try 'turn on', 'turn off', or specify a value."
            }

        # Get current DET state for gatekeeper evaluation
        idx = target["idx"]
        node_id = target["node_id"]

        # Step the core to get fresh state
        self.core.step(0.05)

        # Get gatekeeper decision based on DET state
        p, c, f, q = self.core.get_aggregates()
        emotion = self.core.get_emotion_string()

        # Simple gatekeeper logic based on DET state
        # In a full implementation, this would use the formal gatekeeper
        proceed = True
        reason = ""

        if p < 0.2:
            # Low presence - not ready to act
            proceed = False
            reason = "I'm not fully present right now."
        elif q > 0.7:
            # High debt - too stressed
            proceed = False
            reason = "I'm feeling overwhelmed at the moment."
        elif c < 0.3:
            # Low coherence - uncertain
            proceed = False
            reason = "I'm not confident enough to do that right now."

        if not proceed:
            return {
                "success": False,
                "error": "gatekeeper_blocked",
                "decision": "STOP",
                "reason": reason,
                "response": f"I can't {self._describe_action(target, target_value)} right now. {reason}",
                "det_state": {"presence": p, "coherence": c, "resource": f, "debt": q}
            }

        # Set the actuator target
        self.core.set_somatic_target(idx, target_value)

        # Get the actual output (agency-gated)
        output = self.core.get_somatic_output(idx)

        # Get agency for response tone
        det_node = self.core.get_node(node_id)
        agency = det_node.a if det_node else 0.5

        response = self._format_actuator_response(target, target_value, output, agency)

        return {
            "success": True,
            "node": target,
            "target_value": target_value,
            "output": output,
            "agency": agency,
            "decision": "PROCEED",
            "response": response,
            "det_state": {"presence": p, "coherence": c, "resource": f, "debt": q}
        }

    def _describe_action(self, node: Dict, value: float) -> str:
        """Describe an actuator action in natural language."""
        name = node["name"].replace("_", " ")
        if value >= 0.9:
            return f"turn on the {name}"
        elif value <= 0.1:
            return f"turn off the {name}"
        else:
            return f"set the {name} to {value * 100:.0f}%"

    def _format_actuator_response(self, node: Dict, target: float, output: float, agency: float) -> str:
        """Format an actuator command response."""
        name = node["name"].replace("_", " ")

        # Describe what was requested
        if target >= 0.9:
            action = "turned on"
        elif target <= 0.1:
            action = "turned off"
        else:
            action = f"set to {target * 100:.0f}%"

        # Check if output matches target (agency gating)
        if abs(output - target) < 0.01:
            # Full agency - direct confirmation
            if agency > 0.7:
                return f"Done. The {name} is now {action}."
            else:
                return f"I've {action} the {name}."
        else:
            # Partial agency - explain the modulation
            actual_pct = output * 100
            return f"I've set the {name} target to {action}, but my current agency is limiting the actual output to {actual_pct:.0f}%."

    def process_somatic_request(self, text: str, intent) -> Dict[str, Any]:
        """
        Main entry point for processing somatic requests.

        Args:
            text: User input text.
            intent: IntentType (SENSE or ACTUATE).

        Returns:
            Result dict with success, response, and details.
        """
        from .llm import IntentType

        if intent == IntentType.SENSE:
            return self.process_sense_intent(text)
        elif intent == IntentType.ACTUATE:
            return self.process_actuate_intent(text)
        else:
            return {
                "success": False,
                "error": "invalid_intent",
                "response": "I'm not sure what you're asking about the physical world."
            }
