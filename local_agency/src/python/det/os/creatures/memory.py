"""
Memory Creature
===============

A DET-OS creature that stores and retrieves memories.
Communicates with other creatures via bonds.

Memory Types:
    - fact: Factual information (e.g., "User's name is Sam")
    - preference: User preferences (e.g., "User prefers Python over JS")
    - instruction: Standing instructions (e.g., "Always be concise")
    - context: Conversation context (e.g., "Working on DET project")
    - episode: Conversation episodes (auto-stored summaries)

Memory messages:
    STORE: {"type": "store", "content": str, "memory_type": str,
            "importance": int, "metadata": dict}
    RECALL: {"type": "recall", "query": str, "limit": int,
             "memory_types": list}
    RESPONSE: {"type": "response", "memories": list, "query": str}
"""

import re
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set
from .base import CreatureWrapper
from ..existence.runtime import ExistenceKernelRuntime, CreatureState


class MemoryType(Enum):
    """Types of memories with different retrieval priorities."""
    FACT = "fact"              # Factual information about user/world
    PREFERENCE = "preference"  # User preferences and likes/dislikes
    INSTRUCTION = "instruction"  # Standing instructions to follow
    CONTEXT = "context"        # Current conversation context
    EPISODE = "episode"        # Conversation episode summaries


# Priority weights for memory types (higher = more important to recall)
MEMORY_TYPE_WEIGHTS = {
    MemoryType.INSTRUCTION: 2.0,  # Always prioritize instructions
    MemoryType.FACT: 1.5,         # Facts are important
    MemoryType.PREFERENCE: 1.3,   # Preferences matter
    MemoryType.CONTEXT: 1.0,      # Context is baseline
    MemoryType.EPISODE: 0.8,      # Episodes are background
}


@dataclass
class MemoryEntry:
    """A single memory entry with type and importance."""
    content: str
    memory_type: MemoryType = MemoryType.CONTEXT
    importance: int = 5  # 1-10 scale
    timestamp: float = field(default_factory=time.time)
    source_cid: int = 0
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def relevance(self) -> float:
        """Compute base relevance from type and importance."""
        type_weight = MEMORY_TYPE_WEIGHTS.get(self.memory_type, 1.0)
        return (self.importance / 10.0) * type_weight

    def matches(self, query: str, memory_types: Optional[Set[MemoryType]] = None) -> float:
        """
        Compute match score for a query.
        Uses keyword matching with type filtering and importance weighting.
        """
        # Filter by memory type if specified
        if memory_types and self.memory_type not in memory_types:
            return 0.0

        # Strip punctuation and lowercase
        query_clean = re.sub(r'[^\w\s]', '', query.lower())
        content_clean = re.sub(r'[^\w\s]', '', self.content.lower())

        query_words = set(query_clean.split())
        content_words = set(content_clean.split())

        if not query_words:
            return 0.0

        # Intersection over query size
        matched = len(query_words & content_words)
        keyword_score = matched / len(query_words)

        # Boost by importance and type weight
        importance_boost = self.importance / 5.0  # Normalize around 1.0
        type_weight = MEMORY_TYPE_WEIGHTS.get(self.memory_type, 1.0)

        # Recency boost (memories decay over hours)
        age_hours = (time.time() - self.timestamp) / 3600
        recency_boost = 1.0 / (1.0 + age_hours * 0.05)

        # Access boost (frequently accessed memories are more relevant)
        access_boost = 1.0 + (self.access_count * 0.1)

        return keyword_score * importance_boost * type_weight * recency_boost * access_boost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "source": self.source_cid,
            "access_count": self.access_count,
            "relevance": self.relevance,
        }


class MemoryCreature(CreatureWrapper):
    """
    Memory Creature - stores and retrieves memories via bonds.

    Protocol:
        Other creatures send STORE/RECALL messages via their bond.
        MemoryCreature processes messages and sends RESPONSE.

    Storage costs F:
        - Store: 0.1 F per 100 chars (scaled by importance)
        - Recall: 0.05 F per query

    Memory Types:
        - fact: Factual information
        - preference: User preferences
        - instruction: Standing instructions
        - context: Conversation context
        - episode: Episode summaries

    Example usage:
        # Store with type and importance
        llm.send_to(memory.cid, {
            "type": "store",
            "content": "User's name is Sam",
            "memory_type": "fact",
            "importance": 9
        })

        # Recall with type filter
        llm.send_to(memory.cid, {
            "type": "recall",
            "query": "name",
            "limit": 5,
            "memory_types": ["fact", "preference"]
        })
    """

    # Cost constants
    STORE_COST_PER_100_CHARS = 0.1
    RECALL_COST = 0.05

    def __init__(self, runtime: ExistenceKernelRuntime, cid: int):
        super().__init__(runtime, cid)
        self.memories: List[MemoryEntry] = []
        self.max_memories = 1000
        self.total_stored = 0
        self.total_recalled = 0

        # Stats by type
        self.stored_by_type: Dict[MemoryType, int] = {t: 0 for t in MemoryType}

    def store(self, content: str, source_cid: int = 0,
              memory_type: MemoryType = MemoryType.CONTEXT,
              importance: int = 5,
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a memory directly (internal use).
        Returns True if stored, False if insufficient F.
        """
        # Clamp importance to valid range
        importance = max(1, min(10, importance))

        # Calculate cost (higher importance = slightly higher cost)
        base_cost = len(content) / 100 * self.STORE_COST_PER_100_CHARS
        importance_factor = 0.5 + (importance / 20.0)  # 0.55 to 1.0
        cost = max(0.01, base_cost * importance_factor)

        if self.F < cost:
            return False

        # Deduct cost
        self.F -= cost

        # Create entry
        entry = MemoryEntry(
            content=content,
            memory_type=memory_type,
            importance=importance,
            source_cid=source_cid,
            metadata=metadata or {}
        )
        self.memories.append(entry)
        self.total_stored += 1
        self.stored_by_type[memory_type] += 1

        # Prune if over limit (remove oldest low-relevance)
        if len(self.memories) > self.max_memories:
            self._prune_memories()

        return True

    def recall(self, query: str, limit: int = 5,
               memory_types: Optional[List[MemoryType]] = None) -> List[MemoryEntry]:
        """
        Recall memories matching query (internal use).
        Returns list of matching memories.
        """
        if self.F < self.RECALL_COST:
            return []

        self.F -= self.RECALL_COST
        self.total_recalled += 1

        # Convert to set for efficient lookup
        type_filter = set(memory_types) if memory_types else None

        # Score all memories
        scored = [(m, m.matches(query, type_filter)) for m in self.memories]
        scored = [(m, s) for m, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top matches
        results = []
        for memory, score in scored[:limit]:
            memory.access_count += 1
            results.append(memory)

        return results

    def get_by_type(self, memory_type: MemoryType, limit: int = 10) -> List[MemoryEntry]:
        """Get memories of a specific type, sorted by importance."""
        typed = [m for m in self.memories if m.memory_type == memory_type]
        typed.sort(key=lambda m: (m.importance, -m.timestamp), reverse=True)
        return typed[:limit]

    def get_instructions(self) -> List[str]:
        """Get all instruction memories (for system prompt)."""
        instructions = self.get_by_type(MemoryType.INSTRUCTION, limit=20)
        return [m.content for m in instructions]

    def process_messages(self):
        """
        Process incoming messages from all bonded creatures.
        Send responses back through the bonds.
        """
        for peer_cid in list(self.bonds.keys()):
            messages = self.receive_all_from(peer_cid)

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")

                if msg_type == "store":
                    content = msg.get("content", "")
                    metadata = msg.get("metadata", {})

                    # Parse memory type
                    mem_type_str = msg.get("memory_type", "context")
                    try:
                        mem_type = MemoryType(mem_type_str)
                    except ValueError:
                        mem_type = MemoryType.CONTEXT

                    # Parse importance
                    importance = msg.get("importance", 5)

                    success = self.store(
                        content,
                        source_cid=peer_cid,
                        memory_type=mem_type,
                        importance=importance,
                        metadata=metadata
                    )

                    # Send acknowledgment
                    self.send_to(peer_cid, {
                        "type": "store_ack",
                        "success": success,
                        "memory_type": mem_type.value,
                        "importance": importance,
                        "content_preview": content[:50] if success else None
                    })

                elif msg_type == "recall":
                    query = msg.get("query", "")
                    limit = msg.get("limit", 5)

                    # Parse memory type filter
                    mem_types_str = msg.get("memory_types", None)
                    mem_types = None
                    if mem_types_str:
                        mem_types = []
                        for t in mem_types_str:
                            try:
                                mem_types.append(MemoryType(t))
                            except ValueError:
                                pass

                    memories = self.recall(query, limit, mem_types)

                    # Send response
                    self.send_to(peer_cid, {
                        "type": "response",
                        "query": query,
                        "count": len(memories),
                        "memories": [m.to_dict() for m in memories]
                    })

                elif msg_type == "get_instructions":
                    # Special message to get all standing instructions
                    instructions = self.get_instructions()
                    self.send_to(peer_cid, {
                        "type": "instructions",
                        "count": len(instructions),
                        "instructions": instructions
                    })

    def _prune_memories(self):
        """Remove oldest low-relevance memories when over limit."""
        # Never prune instructions - they're critical
        instructions = [m for m in self.memories if m.memory_type == MemoryType.INSTRUCTION]
        other = [m for m in self.memories if m.memory_type != MemoryType.INSTRUCTION]

        # Sort other memories by relevance score
        other.sort(
            key=lambda m: (m.relevance * (m.access_count + 1), m.timestamp),
            reverse=True
        )

        # Keep instructions + top other memories
        keep_count = self.max_memories - len(instructions)
        self.memories = instructions + other[:keep_count]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory creature statistics."""
        base = self.get_state_dict()

        # Count by type
        type_counts = {t.value: 0 for t in MemoryType}
        for m in self.memories:
            type_counts[m.memory_type.value] += 1

        # Average importance
        avg_importance = 0.0
        if self.memories:
            avg_importance = sum(m.importance for m in self.memories) / len(self.memories)

        base.update({
            "memory_count": len(self.memories),
            "total_stored": self.total_stored,
            "total_recalled": self.total_recalled,
            "max_memories": self.max_memories,
            "by_type": type_counts,
            "avg_importance": round(avg_importance, 1),
        })
        return base

    def list_memories(self, limit: int = 10, memory_type: Optional[MemoryType] = None) -> List[Dict]:
        """List recent memories for display."""
        mems = self.memories
        if memory_type:
            mems = [m for m in mems if m.memory_type == memory_type]

        # Sort by timestamp descending
        mems = sorted(mems, key=lambda m: m.timestamp, reverse=True)[:limit]
        return [m.to_dict() for m in mems]


def spawn_memory_creature(runtime: ExistenceKernelRuntime,
                          name: str = "memory",
                          initial_f: float = 50.0,
                          initial_a: float = 0.5) -> MemoryCreature:
    """
    Spawn a new memory creature.

    Returns the MemoryCreature wrapper.
    """
    from ..existence.bootstrap import DETOSBootstrap

    cid = runtime.spawn(name, initial_f=initial_f, initial_a=initial_a)
    runtime.creatures[cid].state = CreatureState.RUNNING

    return MemoryCreature(runtime, cid)


# Export memory types for CLI use
__all__ = ['MemoryCreature', 'MemoryEntry', 'MemoryType', 'spawn_memory_creature']
