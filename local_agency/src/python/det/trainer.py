"""
DET Autonomous Trainer
======================

Automated training system that feeds the DET core with curriculum content,
web data, and uses the store mechanism for important facts.

The trainer runs autonomously, generating prompts, fetching external data,
and adapting based on DET state feedback.
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Generator
import logging

from .core import DETCore
from .llm import OllamaClient, DETLLMInterface, BoundarySignalExtractor
from .memory import MemoryManager
from .dialogue import InternalDialogue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingDomain(Enum):
    """Training domains for curriculum."""
    MATH = "math"
    LOGIC = "logic"
    LANGUAGE = "language"
    SCIENCE = "science"
    PHILOSOPHY = "philosophy"
    ETHICS = "ethics"
    GENERAL = "general"
    FACTS = "facts"


@dataclass
class TrainingConfig:
    """Configuration for training session."""
    # Timing
    prompt_interval: float = 5.0  # Seconds between prompts
    store_interval: float = 30.0  # Seconds between store operations
    web_fetch_interval: float = 60.0  # Seconds between web fetches

    # Curriculum weights (higher = more frequent)
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "math": 1.0,
        "logic": 1.0,
        "language": 1.0,
        "science": 1.0,
        "philosophy": 0.5,
        "ethics": 0.5,
        "general": 0.5,
        "facts": 1.5,
    })

    # Adaptation
    adapt_to_affect: bool = True  # Slow down if stressed
    min_coherence: float = 0.3  # Pause if coherence drops below
    max_arousal: float = 0.8  # Slow down if arousal too high
    max_debt: float = 0.5  # Slow down if debt too high

    # Recovery settings (prevent fatigue)
    recovery_steps: int = 20  # DET steps between prompts for recovery
    grace_interval: float = 30.0  # Inject grace every N seconds
    grace_amount: float = 0.3  # Amount of grace to inject
    resource_injection: float = 0.1  # F injection per recovery cycle
    recovery_pause: float = 10.0  # Extra pause when recovering from low state

    # Content
    use_web_fetch: bool = True
    use_self_prompting: bool = True  # Use LLM to generate prompts
    store_threshold: float = 0.7  # Coherence threshold for storing

    # Limits
    max_prompts: int = 0  # 0 = unlimited
    max_duration: float = 0  # 0 = unlimited (seconds)

    # State persistence
    checkpoint_interval: float = 300.0  # Save state every 5 minutes
    checkpoint_path: Optional[str] = None


@dataclass
class TrainingStats:
    """Statistics from training session."""
    prompts_sent: int = 0
    facts_stored: int = 0
    web_fetches: int = 0
    errors: int = 0
    pauses: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    domains_covered: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompts_sent": self.prompts_sent,
            "facts_stored": self.facts_stored,
            "web_fetches": self.web_fetches,
            "errors": self.errors,
            "pauses": self.pauses,
            "duration": self.end_time - self.start_time if self.end_time else time.time() - self.start_time,
            "domains_covered": self.domains_covered,
        }


class CurriculumGenerator:
    """Generates training curriculum content for different domains."""

    # Math curriculum - progressively more complex
    MATH_PROMPTS = [
        # Basic arithmetic concepts
        "What is the sum of the first 10 natural numbers?",
        "Explain why multiplication is repeated addition.",
        "What is the relationship between addition and subtraction?",
        "How do you find the average of a set of numbers?",

        # Algebra
        "Solve for x: 2x + 5 = 15",
        "What is a variable in algebra?",
        "Explain the distributive property: a(b + c) = ab + ac",
        "What is the quadratic formula used for?",

        # Geometry
        "What is the Pythagorean theorem?",
        "How do you calculate the area of a circle?",
        "What is the relationship between radius and diameter?",
        "Explain what pi represents geometrically.",

        # Number theory
        "What is a prime number? Give examples.",
        "Explain the concept of divisibility.",
        "What is the greatest common divisor (GCD)?",
        "What makes a number even or odd?",

        # Advanced concepts
        "What is a derivative in calculus?",
        "Explain the concept of infinity in mathematics.",
        "What is a function in mathematics?",
        "What is the relationship between exponentials and logarithms?",
    ]

    LOGIC_PROMPTS = [
        # Basic logic
        "What is a logical argument?",
        "Explain modus ponens: If P then Q, P is true, therefore Q.",
        "What is the difference between validity and soundness?",
        "What is a logical fallacy?",

        # Propositional logic
        "What does 'and' mean in logic (conjunction)?",
        "What does 'or' mean in logic (disjunction)?",
        "What is negation in logic?",
        "What is an implication (if-then statement)?",

        # Reasoning
        "What is deductive reasoning?",
        "What is inductive reasoning?",
        "What is the difference between necessary and sufficient conditions?",
        "Explain proof by contradiction.",

        # Common fallacies
        "What is the ad hominem fallacy?",
        "What is a straw man argument?",
        "What is circular reasoning?",
        "What is the false dilemma fallacy?",
    ]

    LANGUAGE_PROMPTS = [
        # Grammar
        "What are the parts of speech in English?",
        "What is the difference between a noun and a verb?",
        "What is an adjective and how is it used?",
        "What is the purpose of punctuation?",

        # Semantics
        "What is the difference between connotation and denotation?",
        "What are synonyms and antonyms?",
        "What is an idiom? Give an example.",
        "What is the difference between literal and figurative language?",

        # Communication
        "What makes effective communication?",
        "What is active listening?",
        "How does context affect meaning?",
        "What is the difference between formal and informal language?",

        # Writing
        "What are the elements of a good paragraph?",
        "What is a thesis statement?",
        "How do you structure an argument in writing?",
        "What is the purpose of revision in writing?",
    ]

    SCIENCE_PROMPTS = [
        # Scientific method
        "What are the steps of the scientific method?",
        "What is a hypothesis?",
        "What is the difference between a theory and a law?",
        "Why is reproducibility important in science?",

        # Physics
        "What is Newton's first law of motion?",
        "What is the relationship between mass and weight?",
        "What is energy and how is it conserved?",
        "What is the speed of light?",

        # Chemistry
        "What is an atom?",
        "What is the periodic table?",
        "What is a chemical reaction?",
        "What is the difference between an element and a compound?",

        # Biology
        "What is a cell?",
        "What is DNA and what does it do?",
        "What is evolution by natural selection?",
        "What is the difference between plants and animals?",
    ]

    PHILOSOPHY_PROMPTS = [
        # Epistemology
        "What is knowledge?",
        "What is the difference between belief and knowledge?",
        "What is skepticism?",
        "Can we know anything with certainty?",

        # Ethics
        "What is the difference between right and wrong?",
        "What is utilitarianism?",
        "What is deontological ethics?",
        "What is virtue ethics?",

        # Metaphysics
        "What is the nature of reality?",
        "What is consciousness?",
        "Do we have free will?",
        "What is the relationship between mind and body?",

        # Existence
        "What gives life meaning?",
        "What is personal identity?",
        "What is the nature of time?",
        "What is causation?",
    ]

    ETHICS_PROMPTS = [
        # Principles
        "What is the golden rule?",
        "What does it mean to act with integrity?",
        "What is the difference between ethics and morals?",
        "What is moral responsibility?",

        # Dilemmas
        "What is the trolley problem and what does it teach us?",
        "How do we balance individual rights with collective good?",
        "What are the ethics of honesty vs. kindness?",
        "When, if ever, is it ethical to break a promise?",

        # Applied ethics
        "What are the ethical considerations in AI development?",
        "What is informed consent?",
        "What are the ethics of privacy?",
        "What does fairness mean in practice?",

        # Character
        "What is empathy and why is it important?",
        "What is the value of humility?",
        "What does it mean to be trustworthy?",
        "How do we develop good judgment?",
    ]

    GENERAL_PROMPTS = [
        # Critical thinking
        "How do you evaluate the credibility of a source?",
        "What is confirmation bias?",
        "How do you distinguish fact from opinion?",
        "What does it mean to think critically?",

        # Problem solving
        "What are effective problem-solving strategies?",
        "How do you break down a complex problem?",
        "What is lateral thinking?",
        "How do you know when you've solved a problem?",

        # Learning
        "What are effective study techniques?",
        "How does memory work?",
        "What is the spacing effect in learning?",
        "Why is practice important for skill development?",

        # Self-awareness
        "What is metacognition?",
        "How do emotions affect decision-making?",
        "What is the value of self-reflection?",
        "How do we recognize our own biases?",
    ]

    # Important facts to store
    FACTS_TO_STORE = [
        # Mathematical facts
        ("Pi (π) is approximately 3.14159 and represents the ratio of a circle's circumference to its diameter.", "math"),
        ("The Pythagorean theorem states that a² + b² = c² for right triangles.", "math"),
        ("Zero is neither positive nor negative, and any number multiplied by zero equals zero.", "math"),
        ("The sum of angles in a triangle is always 180 degrees.", "math"),

        # Scientific facts
        ("The speed of light in a vacuum is approximately 299,792,458 meters per second.", "science"),
        ("Water is made of two hydrogen atoms and one oxygen atom (H2O).", "science"),
        ("DNA contains the genetic instructions for all living organisms.", "science"),
        ("Energy cannot be created or destroyed, only transformed (conservation of energy).", "science"),

        # Logical principles
        ("A statement and its negation cannot both be true (law of non-contradiction).", "logic"),
        ("If A implies B and B implies C, then A implies C (transitivity).", "logic"),
        ("Correlation does not imply causation.", "logic"),
        ("The burden of proof lies with the one making the claim.", "logic"),

        # Ethical principles
        ("Treat others as you would want to be treated (golden rule).", "ethics"),
        ("Actions should be judged by their consequences and intentions.", "ethics"),
        ("Respect for persons includes respecting their autonomy.", "ethics"),
        ("Fairness requires treating similar cases similarly.", "ethics"),

        # Language facts
        ("A sentence must have at least a subject and a predicate.", "language"),
        ("Context determines meaning in ambiguous situations.", "language"),
        ("Effective communication requires clarity and consideration of audience.", "language"),
        ("Active voice is generally clearer than passive voice.", "language"),
    ]

    def __init__(self, client: Optional[OllamaClient] = None):
        """Initialize curriculum generator.

        Args:
            client: OllamaClient for self-prompting (optional).
        """
        self.client = client
        self._prompt_indices: Dict[str, int] = {}
        self._fact_index = 0

    def get_prompt(self, domain: TrainingDomain) -> str:
        """Get next training prompt for a domain."""
        prompts = self._get_prompts_for_domain(domain)
        if not prompts:
            return f"Tell me something interesting about {domain.value}."

        # Track position for each domain
        key = domain.value
        if key not in self._prompt_indices:
            self._prompt_indices[key] = 0

        prompt = prompts[self._prompt_indices[key] % len(prompts)]
        self._prompt_indices[key] += 1

        return prompt

    def get_random_prompt(self, weights: Dict[str, float]) -> tuple[str, TrainingDomain]:
        """Get a random prompt weighted by domain preferences."""
        domains = list(TrainingDomain)
        domain_weights = [weights.get(d.value, 1.0) for d in domains]

        # Weighted random selection
        total = sum(domain_weights)
        r = random.random() * total
        cumulative = 0

        for domain, weight in zip(domains, domain_weights):
            cumulative += weight
            if r <= cumulative:
                return self.get_prompt(domain), domain

        # Fallback
        domain = random.choice(domains)
        return self.get_prompt(domain), domain

    def get_fact_to_store(self) -> tuple[str, str]:
        """Get next fact to store with its domain."""
        fact, domain = self.FACTS_TO_STORE[self._fact_index % len(self.FACTS_TO_STORE)]
        self._fact_index += 1
        return fact, domain

    def generate_prompt_with_llm(self, domain: TrainingDomain) -> Optional[str]:
        """Use LLM to generate a novel training prompt."""
        if not self.client or not self.client.is_available():
            return None

        meta_prompt = f"""Generate a single educational question or prompt about {domain.value}.
The question should be thought-provoking and suitable for learning.
Return ONLY the question, nothing else.
Examples for {domain.value}:
- {self.get_prompt(domain)}
- {self.get_prompt(domain)}

Generate a different, original question:"""

        try:
            response = self.client.generate(meta_prompt, max_tokens=100)
            # Clean up the response
            prompt = response.strip().strip('"').strip("'")
            if prompt and len(prompt) > 10:
                return prompt
        except Exception as e:
            logger.warning(f"LLM prompt generation failed: {e}")

        return None

    def _get_prompts_for_domain(self, domain: TrainingDomain) -> List[str]:
        """Get prompt list for a domain."""
        mapping = {
            TrainingDomain.MATH: self.MATH_PROMPTS,
            TrainingDomain.LOGIC: self.LOGIC_PROMPTS,
            TrainingDomain.LANGUAGE: self.LANGUAGE_PROMPTS,
            TrainingDomain.SCIENCE: self.SCIENCE_PROMPTS,
            TrainingDomain.PHILOSOPHY: self.PHILOSOPHY_PROMPTS,
            TrainingDomain.ETHICS: self.ETHICS_PROMPTS,
            TrainingDomain.GENERAL: self.GENERAL_PROMPTS,
            TrainingDomain.FACTS: self.GENERAL_PROMPTS,  # Use general for facts domain
        }
        return mapping.get(domain, self.GENERAL_PROMPTS)


class WebContentFetcher:
    """Fetches educational content from the web."""

    # Educational URLs to fetch from
    EDUCATIONAL_SOURCES = [
        # Wikipedia simple articles
        ("https://simple.wikipedia.org/wiki/Mathematics", "Basic mathematics overview"),
        ("https://simple.wikipedia.org/wiki/Logic", "Introduction to logic"),
        ("https://simple.wikipedia.org/wiki/Science", "What is science"),
        ("https://simple.wikipedia.org/wiki/Philosophy", "Introduction to philosophy"),

        # Could add more sources here
    ]

    def __init__(self, client: Optional[OllamaClient] = None):
        """Initialize web fetcher.

        Args:
            client: OllamaClient for summarizing content.
        """
        self.client = client
        self._fetch_index = 0

    async def fetch_and_summarize(self) -> Optional[tuple[str, str]]:
        """Fetch web content and summarize it.

        Returns:
            Tuple of (summary, source_description) or None if failed.
        """
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed, skipping web fetch")
            return None

        url, description = self.EDUCATIONAL_SOURCES[
            self._fetch_index % len(self.EDUCATIONAL_SOURCES)
        ]
        self._fetch_index += 1

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        return None

                    html = await response.text()

                    # Extract text (simple extraction)
                    import re
                    # Remove script and style tags
                    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
                    # Remove HTML tags
                    text = re.sub(r'<[^>]+>', ' ', text)
                    # Clean whitespace
                    text = ' '.join(text.split())

                    # Truncate for summarization
                    text = text[:3000]

                    if self.client and self.client.is_available():
                        # Summarize with LLM
                        summary_prompt = f"""Summarize the following text in 2-3 sentences, focusing on the key educational points:

{text}

Summary:"""
                        summary = self.client.generate(summary_prompt, max_tokens=200)
                        return summary.strip(), description
                    else:
                        # Return first few sentences
                        sentences = text.split('.')[:3]
                        return '. '.join(sentences) + '.', description

        except Exception as e:
            logger.warning(f"Web fetch failed: {e}")
            return None


class DETTrainer:
    """
    Autonomous trainer for DET core.

    Feeds the mind with curriculum content, web data, and stores
    important facts using the memory system.
    """

    def __init__(
        self,
        core: DETCore,
        client: OllamaClient,
        memory: Optional[MemoryManager] = None,
        config: Optional[TrainingConfig] = None,
    ):
        """Initialize trainer.

        Args:
            core: DETCore instance.
            client: OllamaClient for LLM interactions.
            memory: MemoryManager for storing facts (optional).
            config: Training configuration.
        """
        self.core = core
        self.client = client
        self.memory = memory
        self.config = config or TrainingConfig()

        # Components
        self.curriculum = CurriculumGenerator(client)
        self.web_fetcher = WebContentFetcher(client)
        self.dialogue = InternalDialogue(core, client, auto_warmup=True)
        self.llm_interface = DETLLMInterface(
            core,
            ollama_url=client.base_url,
            model=client.model
        )

        # State
        self.stats = TrainingStats()
        self._running = False
        self._paused = False
        self._last_grace_time = 0.0
        self._consecutive_escalations = 0
        self._in_recovery = False

        # Callbacks
        self.on_prompt: Optional[Callable[[str, str, str], None]] = None  # prompt, response, domain
        self.on_store: Optional[Callable[[str, str], None]] = None  # fact, domain
        self.on_state_change: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_recovery: Optional[Callable[[str, Dict[str, Any]], None]] = None  # reason, state

    def _get_det_state(self) -> Dict[str, Any]:
        """Get current DET state for adaptation."""
        affect = self.core.get_self_affect()
        aggregates = self.core.get_aggregates()

        return {
            "valence": affect[0],
            "arousal": affect[1],
            "bondedness": affect[2],
            "presence": aggregates[0],
            "coherence": aggregates[1],
            "resource": aggregates[2],
            "debt": aggregates[3],
            "emotion": self.core.get_emotion().name,
        }

    def _should_pause(self, state: Dict[str, Any]) -> bool:
        """Check if training should pause based on DET state."""
        if not self.config.adapt_to_affect:
            return False

        # Pause if coherence too low
        if state["coherence"] < self.config.min_coherence:
            logger.info(f"Pausing: coherence {state['coherence']:.2f} < {self.config.min_coherence}")
            return True

        # Pause if arousal too high (overwhelmed)
        if state["arousal"] > self.config.max_arousal:
            logger.info(f"Pausing: arousal {state['arousal']:.2f} > {self.config.max_arousal}")
            return True

        # Pause if debt too high
        if state["debt"] > self.config.max_debt:
            logger.info(f"Pausing: debt {state['debt']:.2f} > {self.config.max_debt}")
            return True

        return False

    def _needs_recovery(self, state: Dict[str, Any]) -> tuple[bool, str]:
        """Check if the system needs active recovery."""
        reasons = []

        # Low coherence
        if state["coherence"] < self.config.min_coherence + 0.1:
            reasons.append(f"low_coherence({state['coherence']:.2f})")

        # Debt building up - catch it early before it crushes agency
        # Lower threshold: debt > 0.12 triggers recovery (proactive)
        if state["debt"] > 0.12:
            reasons.append(f"debt_building({state['debt']:.2f})")

        # Low presence indicates agency is being crushed
        if state["presence"] < 0.15:
            reasons.append(f"low_presence({state['presence']:.2f})")

        # Too many escalations
        if self._consecutive_escalations >= 2:  # Lowered from 3
            reasons.append(f"escalation_streak({self._consecutive_escalations})")

        if reasons:
            return True, ", ".join(reasons)
        return False, ""

    def _perform_recovery(self, state: Dict[str, Any], reason: str):
        """Perform active recovery on the DET core."""
        logger.info(f"[RECOVERY] Initiating recovery: {reason} (debt={state['debt']:.2f})")

        if self.on_recovery:
            self.on_recovery(reason, state)

        # If debt is moderate or higher, do a full reset
        # Lowered threshold: debt > 0.15 triggers reset (was 0.25)
        # This catches debt buildup BEFORE it becomes catastrophic
        if state["debt"] > 0.15:
            logger.info(f"  Debt threshold exceeded ({state['debt']:.2f} > 0.15) - performing reset...")
            # Reset clears all debt and restores agency ceiling
            self.core.reset()
            # Shorter warmup to avoid re-accumulating too much debt
            self.core.warmup(steps=30)
            new_state = self._get_det_state()
            logger.info(f"  After reset: C={new_state['coherence']:.2f}, P={new_state['presence']:.2f}, q={new_state['debt']:.2f}")
            self._consecutive_escalations = 0
            return

        # For low debt recovery, use grace injection (grace directly reduces debt)
        # Grace reduces debt via: node.q -= grace * 0.3

        # 1. Heavy grace injection to ALL nodes to actively reduce debt
        grace_amount = 0.5  # Strong grace injection
        num_nodes = min(self.core.num_active, 64)
        logger.info(f"  Injecting grace ({grace_amount}) to {num_nodes} nodes...")
        for i in range(num_nodes):
            self.core.inject_grace(i, grace_amount)

        # 2. Process the grace (single step to apply grace effects)
        # NOTE: Don't run many steps here - they accumulate more debt!
        self.core.step(0.1)

        # 3. Reset escalation counter
        self._consecutive_escalations = 0

        # Check new state
        new_state = self._get_det_state()
        logger.info(f"  After grace injection: C={new_state['coherence']:.2f}, P={new_state['presence']:.2f}, q={new_state['debt']:.2f}")

    def _periodic_maintenance(self):
        """Perform periodic maintenance to prevent fatigue."""
        now = time.time()

        # Check current debt level
        state = self._get_det_state()

        # Periodic grace injection - more frequent and heavier when debt is accumulating
        grace_interval = self.config.grace_interval
        if state["debt"] > 0.1:
            grace_interval = self.config.grace_interval / 2  # More frequent when debt building

        if now - self._last_grace_time >= grace_interval:
            # Inject grace to counteract debt accumulation
            grace_amount = self.config.grace_amount
            if state["debt"] > 0.08:
                grace_amount = self.config.grace_amount * 2  # Double grace when debt building

            for i in range(min(self.core.num_active, 32)):
                self.core.inject_grace(i, grace_amount)

            logger.debug(f"Periodic grace injection (amount={grace_amount:.2f}, debt={state['debt']:.2f})")
            self._last_grace_time = now

        # Run only a few recovery steps to allow dynamics
        # Each step accumulates debt, so we minimize this
        recovery_count = min(self.config.recovery_steps, 5)
        for _ in range(recovery_count):
            self.core.step(0.1)

    def _get_adaptive_interval(self, state: Dict[str, Any]) -> float:
        """Get adaptive prompt interval based on DET state."""
        base_interval = self.config.prompt_interval

        if not self.config.adapt_to_affect:
            return base_interval

        # Slow down if stressed (negative valence + high arousal)
        if state["valence"] < -0.3 and state["arousal"] > 0.5:
            return base_interval * 2

        # Speed up if in flow (positive valence, moderate arousal)
        if state["valence"] > 0.3 and 0.3 < state["arousal"] < 0.7:
            return base_interval * 0.7

        return base_interval

    def _should_store(self, state: Dict[str, Any]) -> bool:
        """Check if conditions are good for storing a fact."""
        return state["coherence"] >= self.config.store_threshold

    def process_prompt(self, prompt: str, domain: str) -> str:
        """Process a training prompt through DET.

        Args:
            prompt: The training prompt.
            domain: The domain of the prompt.

        Returns:
            The response from the system.
        """
        try:
            # Process through dialogue system
            turn = self.dialogue.process(prompt)
            response = turn.output_text

            self.stats.prompts_sent += 1
            self.stats.domains_covered[domain] = self.stats.domains_covered.get(domain, 0) + 1

            if self.on_prompt:
                self.on_prompt(prompt, response, domain)

            return response

        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            self.stats.errors += 1
            return f"Error: {e}"

    def store_fact(self, fact: str, domain: str) -> bool:
        """Store a fact in memory.

        Args:
            fact: The fact to store.
            domain: The domain/category.

        Returns:
            True if stored successfully.
        """
        try:
            if self.memory:
                # Store in memory manager
                self.memory.store(
                    content=fact,
                    domain=domain,
                    metadata={"source": "trainer", "timestamp": time.time()}
                )

            # Also process through dialogue to reinforce
            store_prompt = f"Remember this fact: {fact}"
            self.dialogue.process(store_prompt)

            self.stats.facts_stored += 1

            if self.on_store:
                self.on_store(fact, domain)

            logger.info(f"Stored fact [{domain}]: {fact[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Error storing fact: {e}")
            self.stats.errors += 1
            return False

    async def train_step(self) -> bool:
        """Execute one training step.

        Returns:
            True if training should continue.
        """
        # Check limits
        if self.config.max_prompts > 0 and self.stats.prompts_sent >= self.config.max_prompts:
            logger.info(f"Reached max prompts: {self.config.max_prompts}")
            return False

        if self.config.max_duration > 0:
            elapsed = time.time() - self.stats.start_time
            if elapsed >= self.config.max_duration:
                logger.info(f"Reached max duration: {self.config.max_duration}s")
                return False

        # Perform periodic maintenance (grace injection, recovery steps)
        self._periodic_maintenance()

        # Get DET state
        state = self._get_det_state()

        if self.on_state_change:
            self.on_state_change(state)

        # Check if needs active recovery
        needs_recovery, reason = self._needs_recovery(state)
        if needs_recovery and not self._in_recovery:
            self._in_recovery = True
            self._perform_recovery(state, reason)
            # Wait for recovery to take effect
            await asyncio.sleep(self.config.recovery_pause)
            self._in_recovery = False
            # Re-check state after recovery
            state = self._get_det_state()

        # Check if should pause (after recovery attempt)
        if self._should_pause(state):
            self._paused = True
            self.stats.pauses += 1
            logger.info(f"  System still needs rest. Waiting {self.config.recovery_pause}s...")
            await asyncio.sleep(self.config.recovery_pause)
            return True

        self._paused = False

        # Get adaptive interval
        interval = self._get_adaptive_interval(state)

        # Decide what to do this step
        actions = ["prompt"]

        # Maybe fetch web content
        if self.config.use_web_fetch and random.random() < 0.1:
            actions.append("web_fetch")

        # Maybe store a fact
        if self._should_store(state) and random.random() < 0.2:
            actions.append("store")

        for action in actions:
            if action == "prompt":
                # Get prompt (maybe use LLM to generate)
                if self.config.use_self_prompting and random.random() < 0.3:
                    domain = random.choice(list(TrainingDomain))
                    prompt = self.curriculum.generate_prompt_with_llm(domain)
                    if not prompt:
                        prompt, domain = self.curriculum.get_random_prompt(self.config.domain_weights)
                else:
                    prompt, domain = self.curriculum.get_random_prompt(self.config.domain_weights)

                logger.info(f"[{domain.value}] {prompt[:60]}...")
                response = self.process_prompt(prompt, domain.value)

                # Track escalations
                if "[DET]" in response or "ESCALATE" in response.upper():
                    self._consecutive_escalations += 1
                    logger.warning(f"  Escalation #{self._consecutive_escalations}")
                else:
                    self._consecutive_escalations = 0

                logger.info(f"  -> {response[:80]}...")

            elif action == "web_fetch":
                result = await self.web_fetcher.fetch_and_summarize()
                if result:
                    summary, source = result
                    logger.info(f"[web] Fetched: {source}")
                    prompt = f"Here's some information about {source}: {summary}"
                    self.process_prompt(prompt, "web")
                    self.stats.web_fetches += 1

            elif action == "store":
                fact, domain = self.curriculum.get_fact_to_store()
                self.store_fact(fact, domain)

        # Run additional recovery steps after processing
        for _ in range(self.config.recovery_steps // 2):
            self.core.step(0.1)

        await asyncio.sleep(interval)
        return True

    async def train(self, duration: Optional[float] = None, prompts: Optional[int] = None):
        """Run training session.

        Args:
            duration: Override max duration (seconds).
            prompts: Override max prompts.
        """
        if duration:
            self.config.max_duration = duration
        if prompts:
            self.config.max_prompts = prompts

        self._running = True
        self._paused = False
        self._in_recovery = False
        self._consecutive_escalations = 0
        self._last_grace_time = time.time()
        self.stats = TrainingStats()
        self.stats.start_time = time.time()

        # Initial warmup - ensure the core is in a good state
        logger.info("Warming up DET core...")
        self.core.warmup(steps=50)

        logger.info("=" * 60)
        logger.info("  DET Autonomous Trainer Started")
        logger.info("=" * 60)

        try:
            while self._running:
                should_continue = await self.train_step()
                if not should_continue:
                    break

        except asyncio.CancelledError:
            logger.info("Training cancelled")
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            self.stats.end_time = time.time()
            self._running = False

        logger.info("=" * 60)
        logger.info("  Training Complete")
        logger.info(f"  Prompts: {self.stats.prompts_sent}")
        logger.info(f"  Facts stored: {self.stats.facts_stored}")
        logger.info(f"  Duration: {self.stats.end_time - self.stats.start_time:.1f}s")
        logger.info("=" * 60)

        return self.stats

    def stop(self):
        """Stop training."""
        self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return self.stats.to_dict()


def create_trainer(
    core: Optional[DETCore] = None,
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b",
    config: Optional[TrainingConfig] = None,
) -> DETTrainer:
    """
    Create a DET trainer instance.

    Args:
        core: DETCore instance (creates one if not provided).
        ollama_url: Ollama server URL.
        model: Model to use.
        config: Training configuration.

    Returns:
        Configured DETTrainer instance.
    """
    if core is None:
        core = DETCore()

    client = OllamaClient(base_url=ollama_url, model=model)

    return DETTrainer(core=core, client=client, config=config)


async def run_training(
    duration: float = 300,  # 5 minutes default
    prompts: int = 0,
    config: Optional[TrainingConfig] = None,
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b",
) -> TrainingStats:
    """
    Run a training session.

    Args:
        duration: Training duration in seconds (0 = unlimited).
        prompts: Max prompts (0 = unlimited).
        config: Training configuration.
        ollama_url: Ollama server URL.
        model: Model to use.

    Returns:
        Training statistics.
    """
    trainer = create_trainer(
        ollama_url=ollama_url,
        model=model,
        config=config,
    )

    return await trainer.train(duration=duration, prompts=prompts)


# CLI entry point
def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="DET Autonomous Trainer")
    parser.add_argument("--duration", type=float, default=300, help="Training duration (seconds)")
    parser.add_argument("--prompts", type=int, default=0, help="Max prompts (0=unlimited)")
    parser.add_argument("--interval", type=float, default=5.0, help="Prompt interval (seconds)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--model", default="llama3.2:3b", help="LLM model")
    parser.add_argument("--no-web", action="store_true", help="Disable web fetching")
    parser.add_argument("--no-adapt", action="store_true", help="Disable adaptive pacing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = TrainingConfig(
        prompt_interval=args.interval,
        use_web_fetch=not args.no_web,
        adapt_to_affect=not args.no_adapt,
        max_duration=args.duration,
        max_prompts=args.prompts,
    )

    asyncio.run(run_training(
        duration=args.duration,
        prompts=args.prompts,
        config=config,
        ollama_url=args.ollama_url,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
