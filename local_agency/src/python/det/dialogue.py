"""
DET Internal Dialogue System
============================

Request reformulation, retry logic, and escalation handling
with DET feedback integration.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import IntEnum
import time

from .core import DETCore, DETDecision
from .llm import OllamaClient, DET_SYSTEM_PROMPT


class DialogueState(IntEnum):
    """Internal dialogue states."""
    IDLE = 0
    PROCESSING = 1
    REFORMULATING = 2
    ESCALATING = 3
    COMPLETE = 4
    FAILED = 5


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
            prompt_template="""The previous request was too complex. Please simplify it.

Original request: {original}

Please reformulate this into a simpler, more focused request.""",
            max_uses=2,
        ),
        ReformulationStrategy(
            name="clarify",
            prompt_template="""The previous request was ambiguous. Please clarify it.

Original request: {original}

What specifically is being asked? Please reformulate with more clarity.""",
            max_uses=2,
        ),
        ReformulationStrategy(
            name="decompose",
            prompt_template="""The previous request requires multiple steps. Please break it down.

Original request: {original}

What is the first concrete step to address this request?""",
            max_uses=3,
        ),
        ReformulationStrategy(
            name="safety_check",
            prompt_template="""The system flagged potential concerns with this request.

Original request: {original}

Is this request safe to proceed with? If so, please rephrase it to address any concerns.
If not, explain what cannot be done and suggest an alternative.""",
            max_uses=1,
        ),
    ]

    def __init__(
        self,
        core: DETCore,
        client: OllamaClient,
        max_retries: int = 5,
        max_internal_turns: int = 10
    ):
        """
        Initialize the internal dialogue system.

        Args:
            core: DETCore instance.
            client: OllamaClient for LLM calls.
            max_retries: Maximum retry attempts.
            max_internal_turns: Maximum internal dialogue turns.
        """
        self.core = core
        self.client = client
        self.max_retries = max_retries
        self.max_internal_turns = max_internal_turns

        self.state = DialogueState.IDLE
        self.turns: List[DialogueTurn] = []
        self.strategy_uses: Dict[str, int] = {s.name: 0 for s in self.STRATEGIES}

        # Callbacks
        self.on_turn_complete: Optional[Callable[[DialogueTurn], None]] = None
        self.on_escalate: Optional[Callable[[str], str]] = None

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

        response = self.client.generate(
            prompt=prompt,
            system="You are a helpful assistant that reformulates requests to be clearer and more actionable.",
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

    def _generate_response(self, text: str) -> str:
        """Generate a response from the LLM."""
        affect, _ = self._get_det_state()

        # Adjust temperature based on affect
        temperature = 0.7 + 0.2 * affect["arousal"] - 0.1 * affect["bondedness"]
        temperature = max(0.1, min(temperature, 1.0))

        response = self.client.chat(
            messages=[
                {"role": "system", "content": DET_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
        )

        return response.get("message", {}).get("content", "")

    def process(self, user_input: str) -> DialogueTurn:
        """
        Process a user input through the internal dialogue system.

        This is the main entry point. It handles:
        1. Initial evaluation through DET gatekeeper
        2. Reformulation if RETRY
        3. Escalation if ESCALATE
        4. Response generation if PROCEED

        Args:
            user_input: The user's input text.

        Returns:
            The final DialogueTurn with the response.
        """
        self.state = DialogueState.PROCESSING
        self.strategy_uses = {s.name: 0 for s in self.STRATEGIES}

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

                turn = DialogueTurn(
                    input_text=user_input,
                    decision=decision,
                    output_text=response,
                    timestamp=time.time(),
                    reformulation_count=retry_count,
                    affect=affect,
                    aggregates=aggregates,
                )
                self.turns.append(turn)

                if self.on_turn_complete:
                    self.on_turn_complete(turn)

                self.state = DialogueState.IDLE
                return turn

            elif decision == DETDecision.RETRY:
                # Attempt reformulation
                self.state = DialogueState.REFORMULATING
                strategy = self._select_strategy(decision, current_text)

                if strategy:
                    current_text = self._reformulate(current_text, strategy)
                    retry_count += 1
                else:
                    # No more strategies, escalate
                    decision = DETDecision.ESCALATE
                    break

            elif decision == DETDecision.STOP:
                # Try safety check reformulation once
                strategy = self._select_strategy(decision, current_text)
                if strategy and strategy.name == "safety_check":
                    self.state = DialogueState.REFORMULATING
                    current_text = self._reformulate(current_text, strategy)
                    retry_count += 1
                else:
                    # Cannot proceed
                    self.state = DialogueState.FAILED
                    turn = DialogueTurn(
                        input_text=user_input,
                        decision=decision,
                        output_text="[DET] Request declined - system in protective state.",
                        timestamp=time.time(),
                        reformulation_count=retry_count,
                        affect=affect,
                        aggregates=aggregates,
                    )
                    self.turns.append(turn)
                    self.state = DialogueState.IDLE
                    return turn

            elif decision == DETDecision.ESCALATE:
                break

        # Escalation path
        self.state = DialogueState.ESCALATING
        affect, aggregates = self._get_det_state()

        response = self._escalate(current_text)

        turn = DialogueTurn(
            input_text=user_input,
            decision=DETDecision.ESCALATE,
            output_text=response,
            timestamp=time.time(),
            reformulation_count=retry_count,
            escalated=True,
            affect=affect,
            aggregates=aggregates,
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
        turns without external input.

        Args:
            topic: The topic to think about.
            max_turns: Maximum thinking turns.

        Returns:
            List of internal dialogue turns.
        """
        thinking_turns = []

        prompt = f"Let me think about: {topic}\n\nWhat are the key considerations?"

        for i in range(max_turns):
            turn = self.process(prompt)
            thinking_turns.append(turn)

            if turn.decision == DETDecision.STOP:
                break

            # Generate next thinking prompt
            prompt = f"Continuing to think about {topic}:\n\n{turn.output_text}\n\nWhat else should I consider?"

            # Check if we've reached a conclusion
            if "conclusion" in turn.output_text.lower() or "therefore" in turn.output_text.lower():
                break

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
        }

    def clear_history(self):
        """Clear dialogue history."""
        self.turns.clear()
        self.strategy_uses = {s.name: 0 for s in self.STRATEGIES}
        self.state = DialogueState.IDLE
