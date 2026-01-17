"""
DET Memory Domain Architecture
==============================

Memory domain management with coherence tracking and request routing.
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from enum import IntEnum
import time


class MemoryDomain(IntEnum):
    """Core memory domains."""
    GENERAL = 0
    MATH = 1
    LANGUAGE = 2
    TOOL_USE = 3
    SCIENCE = 4
    CODE = 5
    REASONING = 6
    DIALOGUE = 7


@dataclass
class DomainConfig:
    """Configuration for a memory domain."""
    domain_id: MemoryDomain
    name: str
    model_name: Optional[str] = None  # Ollama model for this domain
    priority: float = 1.0
    min_coherence: float = 0.3  # Minimum coherence to route to this domain
    keywords: List[str] = field(default_factory=list)

    # Training configuration
    lora_adapter_path: Optional[Path] = None
    training_samples: int = 0
    last_training: Optional[float] = None


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    domain: MemoryDomain
    timestamp: float
    embedding_hash: str
    importance: float = 0.5
    access_count: int = 0
    last_access: float = 0.0

    # DET-related
    coherence_at_storage: float = 0.0
    affect_at_storage: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "domain": self.domain,
            "timestamp": self.timestamp,
            "embedding_hash": self.embedding_hash,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "coherence_at_storage": self.coherence_at_storage,
            "affect_at_storage": self.affect_at_storage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            content=data["content"],
            domain=MemoryDomain(data["domain"]),
            timestamp=data["timestamp"],
            embedding_hash=data["embedding_hash"],
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_access=data.get("last_access", 0.0),
            coherence_at_storage=data.get("coherence_at_storage", 0.0),
            affect_at_storage=data.get("affect_at_storage", {}),
        )


class DomainRouter:
    """Routes requests to appropriate memory domains based on content analysis."""

    # Domain keyword mappings
    DOMAIN_KEYWORDS = {
        MemoryDomain.MATH: [
            "calculate", "compute", "equation", "formula", "math", "number",
            "algebra", "calculus", "geometry", "statistics", "probability",
            "sum", "product", "integral", "derivative", "matrix"
        ],
        MemoryDomain.CODE: [
            "code", "program", "function", "class", "variable", "debug",
            "compile", "syntax", "algorithm", "implement", "refactor",
            "python", "javascript", "rust", "c++", "java"
        ],
        MemoryDomain.TOOL_USE: [
            "file", "directory", "command", "execute", "run", "shell",
            "bash", "terminal", "script", "install", "download", "upload"
        ],
        MemoryDomain.SCIENCE: [
            "physics", "chemistry", "biology", "experiment", "hypothesis",
            "theory", "research", "scientific", "molecule", "atom", "energy"
        ],
        MemoryDomain.LANGUAGE: [
            "write", "translate", "grammar", "sentence", "word", "essay",
            "story", "poem", "language", "spelling", "vocabulary"
        ],
        MemoryDomain.REASONING: [
            "why", "because", "therefore", "logic", "reason", "conclude",
            "deduce", "infer", "analyze", "evaluate", "compare"
        ],
        MemoryDomain.DIALOGUE: [
            "conversation", "chat", "discuss", "talk", "say", "response",
            "question", "answer", "clarify", "explain"
        ],
    }

    def __init__(self):
        self._custom_rules: List[Callable[[str], Optional[MemoryDomain]]] = []

    def add_rule(self, rule: Callable[[str], Optional[MemoryDomain]]):
        """Add a custom routing rule."""
        self._custom_rules.append(rule)

    def route(self, text: str) -> tuple[MemoryDomain, float]:
        """
        Route text to the most appropriate domain.

        Returns:
            Tuple of (domain, confidence).
        """
        text_lower = text.lower()

        # Check custom rules first
        for rule in self._custom_rules:
            result = rule(text)
            if result is not None:
                return result, 0.9

        # Score each domain by keyword matches
        scores = {domain: 0.0 for domain in MemoryDomain}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[domain] += 1.0

        # Find best domain
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        if best_score == 0:
            return MemoryDomain.GENERAL, 0.5

        # Normalize confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5

        return best_domain, min(confidence, 0.95)


class MemoryManager:
    """
    Manages memory domains with DET coherence integration.

    Provides storage, retrieval, and consolidation of memories
    across multiple domains with coherence-based routing.
    """

    def __init__(self, core, storage_path: Optional[Path] = None):
        """
        Initialize the memory manager.

        Args:
            core: DETCore instance for coherence queries.
            storage_path: Path for persistent storage.
        """
        self.core = core
        self.storage_path = storage_path or Path.home() / ".det_agency" / "memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.router = DomainRouter()
        self.domains: Dict[MemoryDomain, DomainConfig] = {}
        self.memories: Dict[MemoryDomain, List[MemoryEntry]] = {
            domain: [] for domain in MemoryDomain
        }

        # Initialize default domains
        self._init_default_domains()

        # Load persisted memories
        self._load_memories()

    def _init_default_domains(self):
        """Initialize default domain configurations."""
        self.domains = {
            MemoryDomain.GENERAL: DomainConfig(
                domain_id=MemoryDomain.GENERAL,
                name="General",
                priority=0.5,
            ),
            MemoryDomain.MATH: DomainConfig(
                domain_id=MemoryDomain.MATH,
                name="Mathematics",
                priority=1.0,
                keywords=DomainRouter.DOMAIN_KEYWORDS[MemoryDomain.MATH],
            ),
            MemoryDomain.CODE: DomainConfig(
                domain_id=MemoryDomain.CODE,
                name="Code",
                priority=1.0,
                keywords=DomainRouter.DOMAIN_KEYWORDS[MemoryDomain.CODE],
            ),
            MemoryDomain.TOOL_USE: DomainConfig(
                domain_id=MemoryDomain.TOOL_USE,
                name="Tool Use",
                priority=0.9,
                keywords=DomainRouter.DOMAIN_KEYWORDS[MemoryDomain.TOOL_USE],
            ),
            MemoryDomain.SCIENCE: DomainConfig(
                domain_id=MemoryDomain.SCIENCE,
                name="Science",
                priority=0.8,
                keywords=DomainRouter.DOMAIN_KEYWORDS[MemoryDomain.SCIENCE],
            ),
            MemoryDomain.LANGUAGE: DomainConfig(
                domain_id=MemoryDomain.LANGUAGE,
                name="Language",
                priority=0.7,
                keywords=DomainRouter.DOMAIN_KEYWORDS[MemoryDomain.LANGUAGE],
            ),
            MemoryDomain.REASONING: DomainConfig(
                domain_id=MemoryDomain.REASONING,
                name="Reasoning",
                priority=0.9,
                keywords=DomainRouter.DOMAIN_KEYWORDS[MemoryDomain.REASONING],
            ),
            MemoryDomain.DIALOGUE: DomainConfig(
                domain_id=MemoryDomain.DIALOGUE,
                name="Dialogue",
                priority=0.6,
                keywords=DomainRouter.DOMAIN_KEYWORDS[MemoryDomain.DIALOGUE],
            ),
        }

        # Register domains in DET core
        for domain in self.domains.values():
            self.core.register_domain(domain.name)

    def _load_memories(self):
        """Load persisted memories from disk."""
        memory_file = self.storage_path / "memories.json"
        if memory_file.exists():
            try:
                with open(memory_file, "r") as f:
                    data = json.load(f)
                    for domain_str, entries in data.items():
                        domain = MemoryDomain(int(domain_str))
                        self.memories[domain] = [
                            MemoryEntry.from_dict(e) for e in entries
                        ]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass  # Start fresh if corrupted

    def _save_memories(self):
        """Persist memories to disk."""
        memory_file = self.storage_path / "memories.json"
        data = {
            str(domain.value): [e.to_dict() for e in entries]
            for domain, entries in self.memories.items()
        }
        with open(memory_file, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        """Compute a hash for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def store(
        self,
        content: str,
        domain: Optional[MemoryDomain] = None,
        importance: float = 0.5
    ) -> MemoryEntry:
        """
        Store a memory entry.

        Args:
            content: The content to store.
            domain: Target domain (auto-routed if None).
            importance: Importance score [0, 1].

        Returns:
            The created MemoryEntry.
        """
        # Auto-route if no domain specified
        if domain is None:
            domain, _ = self.router.route(content)

        # Get current DET state
        valence, arousal, bondedness = self.core.get_self_affect()
        coherence = self.core.get_domain_coherence(int(domain))

        entry = MemoryEntry(
            content=content,
            domain=domain,
            timestamp=time.time(),
            embedding_hash=self._compute_hash(content),
            importance=importance,
            coherence_at_storage=coherence,
            affect_at_storage={
                "valence": valence,
                "arousal": arousal,
                "bondedness": bondedness,
            }
        )

        # Check for duplicates
        for existing in self.memories[domain]:
            if existing.embedding_hash == entry.embedding_hash:
                # Update existing instead
                existing.access_count += 1
                existing.last_access = time.time()
                return existing

        self.memories[domain].append(entry)
        self._save_memories()

        return entry

    def retrieve(
        self,
        query: str,
        domain: Optional[MemoryDomain] = None,
        limit: int = 5,
        min_coherence: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query.
            domain: Target domain (searches all if None).
            limit: Maximum entries to return.
            min_coherence: Minimum domain coherence to include.

        Returns:
            List of matching MemoryEntry objects.
        """
        results = []
        query_lower = query.lower()

        # Determine domains to search
        if domain is not None:
            domains_to_search = [domain]
        else:
            # Search all domains with sufficient coherence
            domains_to_search = []
            for d in MemoryDomain:
                coherence = self.core.get_domain_coherence(int(d))
                if coherence >= min_coherence:
                    domains_to_search.append(d)

        # Score and collect entries
        scored_entries = []
        for d in domains_to_search:
            for entry in self.memories[d]:
                # Simple keyword matching (can be replaced with embeddings)
                score = sum(1 for word in query_lower.split()
                          if word in entry.content.lower())

                # Boost by importance and recency
                recency = 1.0 / (1.0 + (time.time() - entry.timestamp) / 86400)
                score = score * entry.importance * (1 + recency)

                if score > 0:
                    scored_entries.append((score, entry))

        # Sort by score and return top entries
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        results = [entry for _, entry in scored_entries[:limit]]

        # Update access counts
        for entry in results:
            entry.access_count += 1
            entry.last_access = time.time()

        return results

    def route_request(self, text: str) -> tuple[MemoryDomain, float, float]:
        """
        Route a request to the best domain considering DET coherence.

        Returns:
            Tuple of (domain, routing_confidence, domain_coherence).
        """
        # Get base routing
        domain, confidence = self.router.route(text)

        # Get domain coherence from DET core
        coherence = self.core.get_domain_coherence(int(domain))

        # Adjust confidence based on coherence
        adjusted_confidence = confidence * (0.5 + 0.5 * coherence)

        return domain, adjusted_confidence, coherence

    def get_domain_stats(self) -> Dict[str, Any]:
        """Get statistics about memory domains."""
        stats = {}
        for domain in MemoryDomain:
            entries = self.memories[domain]
            coherence = self.core.get_domain_coherence(int(domain))

            stats[domain.name] = {
                "entry_count": len(entries),
                "coherence": coherence,
                "total_accesses": sum(e.access_count for e in entries),
                "avg_importance": (
                    sum(e.importance for e in entries) / len(entries)
                    if entries else 0.0
                ),
            }

        return stats

    def consolidate(self, threshold_days: float = 7.0):
        """
        Consolidate old memories with low importance.

        Memories older than threshold with low access counts
        and importance are candidates for training data generation
        and then removal.

        Args:
            threshold_days: Age threshold for consolidation.
        """
        threshold_time = time.time() - (threshold_days * 86400)
        consolidated = []

        for domain in MemoryDomain:
            entries_to_keep = []
            entries_to_consolidate = []

            for entry in self.memories[domain]:
                # Keep if: recent, important, or frequently accessed
                if (entry.timestamp > threshold_time or
                    entry.importance > 0.7 or
                    entry.access_count > 5):
                    entries_to_keep.append(entry)
                else:
                    entries_to_consolidate.append(entry)

            self.memories[domain] = entries_to_keep
            consolidated.extend(entries_to_consolidate)

        self._save_memories()

        return consolidated

    def generate_training_data(self, domain: MemoryDomain) -> List[Dict[str, str]]:
        """
        Generate training data from memories in a domain.

        Returns:
            List of training examples in instruction-response format.
        """
        training_data = []

        for entry in self.memories[domain]:
            # Convert to instruction-response pairs
            # This is a simple heuristic; can be improved with LLM assistance
            if "?" in entry.content:
                # Looks like a Q&A pair
                parts = entry.content.split("?", 1)
                if len(parts) == 2:
                    training_data.append({
                        "instruction": parts[0].strip() + "?",
                        "response": parts[1].strip(),
                    })
            else:
                # Convert to a "recall" instruction
                summary = entry.content[:50] + "..."
                training_data.append({
                    "instruction": f"What do you know about: {summary}",
                    "response": entry.content,
                })

        return training_data


class ContextWindow:
    """
    Manages context window with intelligent reduction.

    Tracks conversation history and performs coherence-based
    reduction when the context grows too large.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        max_tokens: int = 4096,
        reduction_threshold: float = 0.8
    ):
        """
        Initialize the context window.

        Args:
            memory_manager: MemoryManager for storing reduced context.
            max_tokens: Maximum token budget.
            reduction_threshold: Trigger reduction at this % of max.
        """
        self.memory_manager = memory_manager
        self.max_tokens = max_tokens
        self.reduction_threshold = reduction_threshold

        self.messages: List[Dict[str, str]] = []
        self.token_count: int = 0

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text.split()) * 1.3  # ~1.3 tokens per word average

    def add_message(self, role: str, content: str):
        """Add a message to the context."""
        tokens = self._estimate_tokens(content)
        self.messages.append({"role": role, "content": content})
        self.token_count += tokens

        # Check if reduction is needed
        if self.token_count > self.max_tokens * self.reduction_threshold:
            self._reduce()

    def _reduce(self):
        """Reduce context by storing old messages to memory."""
        if len(self.messages) <= 4:
            return  # Keep at least a few messages

        # Store oldest messages to memory
        messages_to_store = self.messages[:-4]
        self.messages = self.messages[-4:]

        # Recount tokens
        self.token_count = sum(
            self._estimate_tokens(m["content"]) for m in self.messages
        )

        # Store to memory
        for msg in messages_to_store:
            content = f"[{msg['role']}] {msg['content']}"
            self.memory_manager.store(content, importance=0.3)

    def get_context(self) -> List[Dict[str, str]]:
        """Get current context messages."""
        return self.messages.copy()

    def clear(self):
        """Clear the context window."""
        # Store everything to memory first
        for msg in self.messages:
            content = f"[{msg['role']}] {msg['content']}"
            self.memory_manager.store(content, importance=0.2)

        self.messages = []
        self.token_count = 0
