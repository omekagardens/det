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
        affect, _ = self._get_det_state()

        # Adjust temperature based on affect
        temperature = 0.7 + 0.2 * affect["arousal"] - 0.1 * affect["bondedness"]
        temperature = max(0.1, min(temperature, 1.0))

        # Get appropriate system prompt
        system_prompt = get_system_prompt(internal_mode=internal_mode)

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
