"""
Memory Creature
===============

A DET-OS creature that stores and retrieves memories.
Communicates with other creatures via bonds.

Memory messages:
    STORE: {"type": "store", "content": str, "metadata": dict}
    RECALL: {"type": "recall", "query": str, "limit": int}
    RESPONSE: {"type": "response", "memories": list, "query": str}
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from .base import CreatureWrapper
from ..existence.runtime import ExistenceKernelRuntime, CreatureState


@dataclass
class MemoryEntry:
    """A single memory entry."""
    content: str
    timestamp: float = field(default_factory=time.time)
    source_cid: int = 0
    relevance: float = 1.0
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, query: str) -> float:
        """
        Compute match score for a query.
        Simple keyword matching with punctuation handling.
        """
        import re
        # Strip punctuation and lowercase
        query_clean = re.sub(r'[^\w\s]', '', query.lower())
        content_clean = re.sub(r'[^\w\s]', '', self.content.lower())

        query_words = set(query_clean.split())
        content_words = set(content_clean.split())

        if not query_words:
            return 0.0

        # Intersection over query size
        matched = len(query_words & content_words)
        score = matched / len(query_words)

        # Boost by relevance and recency
        age_hours = (time.time() - self.timestamp) / 3600
        recency_boost = 1.0 / (1.0 + age_hours * 0.1)

        return score * self.relevance * recency_boost


class MemoryCreature(CreatureWrapper):
    """
    Memory Creature - stores and retrieves memories via bonds.

    Protocol:
        Other creatures send STORE/RECALL messages via their bond.
        MemoryCreature processes messages and sends RESPONSE.

    Storage costs F:
        - Store: 0.1 F per 100 chars
        - Recall: 0.05 F per query

    Example usage:
        # From another creature
        llm.send_to(memory.cid, {"type": "store", "content": "Hello world"})
        llm.send_to(memory.cid, {"type": "recall", "query": "hello", "limit": 5})

        # Memory processes and responds
        memory.process_messages()

        # Other creature receives
        response = llm.receive_from(memory.cid)
        # {"type": "response", "memories": [...], "query": "hello"}
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

    def store(self, content: str, source_cid: int = 0,
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a memory directly (internal use).
        Returns True if stored, False if insufficient F.
        """
        # Calculate cost
        cost = len(content) / 100 * self.STORE_COST_PER_100_CHARS
        cost = max(0.01, cost)  # Minimum cost

        if self.F < cost:
            return False

        # Deduct cost
        self.F -= cost

        # Create entry
        entry = MemoryEntry(
            content=content,
            source_cid=source_cid,
            metadata=metadata or {}
        )
        self.memories.append(entry)
        self.total_stored += 1

        # Prune if over limit (remove oldest low-relevance)
        if len(self.memories) > self.max_memories:
            self._prune_memories()

        return True

    def recall(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """
        Recall memories matching query (internal use).
        Returns list of matching memories.
        """
        if self.F < self.RECALL_COST:
            return []

        self.F -= self.RECALL_COST
        self.total_recalled += 1

        # Score all memories
        scored = [(m, m.matches(query)) for m in self.memories]
        scored = [(m, s) for m, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top matches
        results = []
        for memory, score in scored[:limit]:
            memory.access_count += 1
            results.append(memory)

        return results

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
                    success = self.store(content, source_cid=peer_cid, metadata=metadata)

                    # Send acknowledgment
                    self.send_to(peer_cid, {
                        "type": "store_ack",
                        "success": success,
                        "content_preview": content[:50] if success else None
                    })

                elif msg_type == "recall":
                    query = msg.get("query", "")
                    limit = msg.get("limit", 5)
                    memories = self.recall(query, limit)

                    # Send response
                    self.send_to(peer_cid, {
                        "type": "response",
                        "query": query,
                        "count": len(memories),
                        "memories": [
                            {
                                "content": m.content,
                                "timestamp": m.timestamp,
                                "relevance": m.relevance,
                                "source": m.source_cid
                            }
                            for m in memories
                        ]
                    })

    def _prune_memories(self):
        """Remove oldest low-relevance memories when over limit."""
        # Sort by (relevance * access_count, timestamp)
        self.memories.sort(
            key=lambda m: (m.relevance * (m.access_count + 1), m.timestamp),
            reverse=True
        )
        # Keep top max_memories
        self.memories = self.memories[:self.max_memories]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory creature statistics."""
        base = self.get_state_dict()
        base.update({
            "memory_count": len(self.memories),
            "total_stored": self.total_stored,
            "total_recalled": self.total_recalled,
            "max_memories": self.max_memories,
        })
        return base


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
