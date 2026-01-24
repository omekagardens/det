"""
Reasoner Creature
=================

A DET-OS creature that provides chain-of-thought reasoning.
Communicates with other creatures via bonds.

This Python wrapper interfaces with the ReasonerCreature defined in creatures.ex.

Reasoner messages:
    REASON: {"type": "reason", "problem": str, "max_steps": int}
    CHAIN: {"type": "chain", "steps": list, "conclusion": str}
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from .base import CreatureWrapper
from ..existence.runtime import ExistenceKernelRuntime, CreatureState


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_num: int
    thought: str
    reasoning_type: str  # 'analyze', 'infer', 'conclude', 'question'
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_num,
            "thought": self.thought,
            "type": self.reasoning_type,
            "confidence": self.confidence
        }


@dataclass
class ReasoningChain:
    """A complete chain of reasoning steps."""
    problem: str
    steps: List[ReasoningStep]
    conclusion: str
    total_cost: float
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "step_count": len(self.steps),
            "cost": self.total_cost,
            "elapsed_ms": self.elapsed_ms
        }


class ReasonerCreature(CreatureWrapper):
    """
    Reasoner Creature - provides chain-of-thought reasoning.

    Protocol:
        Other creatures send REASON messages via their bond.
        ReasonerCreature generates steps and sends CHAIN back.

    Reasoning costs F:
        - Base cost: 0.2 F
        - Per step: 0.3 F

    Agency affects reasoning depth:
        - a = 0.5 -> max 5 steps
        - a = 0.7 -> max 7 steps
        - a = 1.0 -> max 10 steps

    Example usage:
        llm.send_to(reasoner.cid, {
            "type": "reason",
            "problem": "Why is the sky blue?",
            "max_steps": 5
        })
        reasoner.process_messages()
        result = llm.receive_from(reasoner.cid)
    """

    # Cost constants
    BASE_COST = 0.2
    STEP_COST = 0.3

    def __init__(self, runtime: ExistenceKernelRuntime, cid: int,
                 reasoning_fn: Optional[Callable] = None):
        super().__init__(runtime, cid)
        self.is_reasoning = False
        self.total_reasoned = 0
        self.total_steps = 0
        self.total_cost = 0.0

        # External reasoning function (e.g., LLM call)
        # If None, uses simple template-based reasoning
        self.reasoning_fn = reasoning_fn

    @property
    def max_depth(self) -> int:
        """Maximum reasoning depth based on agency."""
        return max(1, int(self.a * 10))

    def can_reason(self, estimated_steps: int = 3) -> tuple:
        """Check if we can afford to reason."""
        estimated_cost = self.BASE_COST + (estimated_steps * self.STEP_COST)
        if self.F < estimated_cost:
            return False, f"Insufficient F (need {estimated_cost:.2f}, have {self.F:.2f})"
        return True, "OK"

    def reason(self, problem: str, max_steps: int = 5) -> ReasoningChain:
        """
        Generate a reasoning chain for a problem.
        """
        # Clamp to agency-based maximum
        actual_max = min(max_steps, self.max_depth)

        # Check resources
        can, reason = self.can_reason(actual_max)
        if not can:
            return ReasoningChain(
                problem=problem,
                steps=[],
                conclusion=f"Cannot reason: {reason}",
                total_cost=0,
                elapsed_ms=0
            )

        self.is_reasoning = True
        start_time = time.time()
        steps = []
        cost = self.BASE_COST
        self.F -= self.BASE_COST

        # Generate reasoning steps
        for i in range(actual_max):
            if self.F < self.STEP_COST:
                break

            # Generate step
            if self.reasoning_fn:
                # Use external reasoning (e.g., LLM)
                step = self._generate_step_external(problem, steps, i)
            else:
                # Use template-based reasoning
                step = self._generate_step_template(problem, steps, i)

            # Deduct cost
            self.F -= self.STEP_COST
            cost += self.STEP_COST

            steps.append(step)
            self.total_steps += 1

            # Check for conclusion
            if step.reasoning_type == 'conclude':
                break

        # Extract conclusion
        if steps and steps[-1].reasoning_type == 'conclude':
            conclusion = steps[-1].thought
        else:
            conclusion = self._synthesize_conclusion(steps)

        elapsed_ms = (time.time() - start_time) * 1000

        self.is_reasoning = False
        self.total_reasoned += 1
        self.total_cost += cost

        return ReasoningChain(
            problem=problem,
            steps=steps,
            conclusion=conclusion,
            total_cost=cost,
            elapsed_ms=elapsed_ms
        )

    def _generate_step_template(self, problem: str, prior_steps: List[ReasoningStep],
                                 step_num: int) -> ReasoningStep:
        """Generate a reasoning step using templates."""
        # Simple template-based reasoning
        if step_num == 0:
            return ReasoningStep(
                step_num=step_num,
                thought=f"Let me analyze the problem: {problem}",
                reasoning_type='analyze',
                confidence=0.9
            )
        elif step_num == 1:
            return ReasoningStep(
                step_num=step_num,
                thought="First, I'll identify the key components and relationships.",
                reasoning_type='analyze',
                confidence=0.85
            )
        elif step_num == len(prior_steps) and step_num >= 2:
            # Final step - conclude
            return ReasoningStep(
                step_num=step_num,
                thought="Based on the analysis, I can conclude the answer.",
                reasoning_type='conclude',
                confidence=0.8
            )
        else:
            return ReasoningStep(
                step_num=step_num,
                thought=f"Following from step {step_num}, I infer the next logical connection.",
                reasoning_type='infer',
                confidence=0.75
            )

    def _generate_step_external(self, problem: str, prior_steps: List[ReasoningStep],
                                  step_num: int) -> ReasoningStep:
        """Generate a reasoning step using external function (e.g., LLM)."""
        if not self.reasoning_fn:
            return self._generate_step_template(problem, prior_steps, step_num)

        # Build context from prior steps
        context = "\n".join([f"Step {s.step_num}: {s.thought}" for s in prior_steps])

        # Call external reasoning function
        prompt = f"""Problem: {problem}

Previous reasoning:
{context if context else "(Starting fresh)"}

Generate the next reasoning step. If you have enough information to conclude, say so.
Keep your response to 1-2 sentences."""

        try:
            thought = self.reasoning_fn(prompt)
            reasoning_type = 'conclude' if 'conclude' in thought.lower() or 'therefore' in thought.lower() else 'infer'

            return ReasoningStep(
                step_num=step_num,
                thought=thought,
                reasoning_type=reasoning_type,
                confidence=0.8
            )
        except Exception as e:
            return ReasoningStep(
                step_num=step_num,
                thought=f"Reasoning error: {e}",
                reasoning_type='error',
                confidence=0.1
            )

    def _synthesize_conclusion(self, steps: List[ReasoningStep]) -> str:
        """Synthesize a conclusion from the reasoning steps."""
        if not steps:
            return "No reasoning steps generated."

        # Simple synthesis - take the last substantive step
        for step in reversed(steps):
            if step.reasoning_type in ('infer', 'conclude'):
                return step.thought

        return steps[-1].thought if steps else "Unable to conclude."

    def process_messages(self):
        """Process incoming messages from all bonded creatures."""
        for peer_cid in list(self.bonds.keys()):
            messages = self.receive_all_from(peer_cid)

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")

                if msg_type == "reason":
                    problem = msg.get("problem", "")
                    max_steps = msg.get("max_steps", 5)

                    chain = self.reason(problem, max_steps)

                    # Send result back
                    self.send_to(peer_cid, {
                        "type": "chain",
                        "problem": chain.problem,
                        "steps": [s.to_dict() for s in chain.steps],
                        "conclusion": chain.conclusion,
                        "step_count": len(chain.steps),
                        "cost": chain.total_cost
                    })

    def get_stats(self) -> Dict[str, Any]:
        """Get reasoner creature statistics."""
        base = self.get_state_dict()
        base.update({
            "total_reasoned": self.total_reasoned,
            "total_steps": self.total_steps,
            "total_cost": round(self.total_cost, 2),
            "max_depth": self.max_depth,
            "is_reasoning": self.is_reasoning,
        })
        return base


def spawn_reasoner_creature(runtime: ExistenceKernelRuntime,
                            name: str = "reasoner",
                            initial_f: float = 40.0,
                            initial_a: float = 0.7,
                            reasoning_fn: Optional[Callable] = None) -> ReasonerCreature:
    """
    Spawn a new reasoner creature.
    Returns the ReasonerCreature wrapper.
    """
    cid = runtime.spawn(name, initial_f=initial_f, initial_a=initial_a)
    runtime.creatures[cid].state = CreatureState.RUNNING
    return ReasonerCreature(runtime, cid, reasoning_fn)
