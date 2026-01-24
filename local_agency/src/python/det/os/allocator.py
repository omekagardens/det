"""
DET-OS Resource Allocator
=========================

Resource allocation in DET-OS follows conservation laws. Unlike traditional
memory allocators:
    - Allocation costs F (resource)
    - Deallocation returns F (minus debt)
    - Total F in system is conserved
    - Fragmentation creates structural debt (q)
    - Grace injection can recover from debt

The allocator manages both memory and computational resources through the
unified lens of DET physics.

Memory Model:
    - Pages are quanta of memory
    - Each page has F cost and q debt
    - Shared pages bond creatures together
    - Page faults increase q (debt accumulation)
"""

from enum import Flag, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
import time


class AllocationFlags(Flag):
    """Allocation request flags."""
    NONE = 0
    ZERO = auto()        # Zero-initialize
    SHARED = auto()      # Shareable between creatures
    LOCKED = auto()      # Cannot be swapped
    EXECUTABLE = auto()  # Can contain code
    GRACEFUL = auto()    # Allow grace on failure


@dataclass
class MemoryBlock:
    """A block of allocated memory."""
    block_id: int
    owner: int           # Creature ID
    address: int         # Virtual address
    size: int            # Size in bytes
    flags: AllocationFlags

    # DET properties
    F_cost: float        # Resource cost to maintain
    q_debt: float        # Structural debt accumulated
    access_count: int = 0
    last_access: float = field(default_factory=time.time)

    # Sharing
    shared_with: Set[int] = field(default_factory=set)

    def access(self):
        """Record an access to this block."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class AllocationResult:
    """Result of an allocation request."""
    success: bool
    block: Optional[MemoryBlock]
    F_consumed: float
    reason: str


@dataclass
class ResourcePool:
    """Pool of available resources."""
    total_F: float           # Total resource in pool
    available_F: float       # Currently available
    total_memory: int        # Total memory bytes
    available_memory: int    # Currently available
    total_q: float = 0.0     # Total debt accumulated


class ResourceAllocator:
    """
    DET-OS resource allocator.

    Manages memory allocation with conservation semantics:
    - Allocation requires F (resource)
    - F is consumed proportional to size
    - Deallocation returns F (minus accumulated debt)
    - Fragmentation and thrashing increase q (debt)
    """

    def __init__(self,
                 total_memory: int = 1024 * 1024 * 1024,  # 1GB
                 total_F: float = 1000.0,
                 page_size: int = 4096,
                 F_per_page: float = 0.01):
        """
        Initialize resource allocator.

        Args:
            total_memory: Total memory in bytes
            total_F: Total resource pool
            page_size: Page size in bytes
            F_per_page: F cost per page
        """
        self.page_size = page_size
        self.F_per_page = F_per_page

        self.pool = ResourcePool(
            total_F=total_F,
            available_F=total_F,
            total_memory=total_memory,
            available_memory=total_memory
        )

        # Block tracking
        self.blocks: Dict[int, MemoryBlock] = {}
        self.next_block_id = 0
        self.next_address = 0x10000  # Start at 64KB

        # Free list (address, size)
        self.free_list: List[tuple] = [(0x10000, total_memory)]

        # Owner tracking
        self.owner_blocks: Dict[int, Set[int]] = {}  # creature_id -> block_ids

    def _pages_for_size(self, size: int) -> int:
        """Calculate pages needed for size."""
        return (size + self.page_size - 1) // self.page_size

    def _F_for_size(self, size: int) -> float:
        """Calculate F cost for size."""
        pages = self._pages_for_size(size)
        return pages * self.F_per_page

    def _find_free_block(self, size: int) -> Optional[tuple]:
        """Find a free block of sufficient size (first-fit)."""
        for i, (addr, block_size) in enumerate(self.free_list):
            if block_size >= size:
                return i, addr, block_size
        return None

    def allocate(self,
                 creature_id: int,
                 size: int,
                 flags: AllocationFlags = AllocationFlags.NONE,
                 creature_F: float = 0.0) -> AllocationResult:
        """
        Allocate memory for a creature.

        Args:
            creature_id: Owning creature ID
            size: Size in bytes
            flags: Allocation flags
            creature_F: Creature's available F (for cost check)

        Returns:
            AllocationResult with success/failure and block info
        """
        # Calculate cost
        F_cost = self._F_for_size(size)

        # Check if creature can afford it
        if creature_F < F_cost and not (flags & AllocationFlags.GRACEFUL):
            return AllocationResult(
                success=False,
                block=None,
                F_consumed=0.0,
                reason=f"Insufficient F: need {F_cost:.4f}, have {creature_F:.4f}"
            )

        # Check pool availability
        if self.pool.available_F < F_cost:
            return AllocationResult(
                success=False,
                block=None,
                F_consumed=0.0,
                reason=f"Pool depleted: need {F_cost:.4f}, pool has {self.pool.available_F:.4f}"
            )

        # Find free block
        result = self._find_free_block(size)
        if not result:
            # Fragmentation - increase debt
            self.pool.total_q += F_cost * 0.1
            return AllocationResult(
                success=False,
                block=None,
                F_consumed=0.0,
                reason="No contiguous block available (fragmentation)"
            )

        idx, addr, block_size = result

        # Allocate
        block_id = self.next_block_id
        self.next_block_id += 1

        block = MemoryBlock(
            block_id=block_id,
            owner=creature_id,
            address=addr,
            size=size,
            flags=flags,
            F_cost=F_cost,
            q_debt=0.0
        )

        # Update free list
        if block_size > size:
            # Shrink free block
            self.free_list[idx] = (addr + size, block_size - size)
        else:
            # Remove free block entirely
            self.free_list.pop(idx)

        # Update pool
        self.pool.available_F -= F_cost
        self.pool.available_memory -= size

        # Track block
        self.blocks[block_id] = block
        if creature_id not in self.owner_blocks:
            self.owner_blocks[creature_id] = set()
        self.owner_blocks[creature_id].add(block_id)

        return AllocationResult(
            success=True,
            block=block,
            F_consumed=F_cost,
            reason="Allocated"
        )

    def free(self, block_id: int) -> float:
        """
        Free a memory block.

        Returns F recovered (cost minus accumulated debt).
        """
        block = self.blocks.get(block_id)
        if not block:
            return 0.0

        # Calculate F to return (minus debt)
        F_return = max(0, block.F_cost - block.q_debt)

        # Return to free list (simple: just append, no coalescing)
        self.free_list.append((block.address, block.size))

        # Update pool
        self.pool.available_F += F_return
        self.pool.available_memory += block.size
        self.pool.total_q += block.q_debt

        # Remove tracking
        if block.owner in self.owner_blocks:
            self.owner_blocks[block.owner].discard(block_id)
        del self.blocks[block_id]

        return F_return

    def free_all_for_creature(self, creature_id: int) -> float:
        """Free all blocks owned by a creature."""
        if creature_id not in self.owner_blocks:
            return 0.0

        total_F = 0.0
        block_ids = list(self.owner_blocks[creature_id])
        for block_id in block_ids:
            total_F += self.free(block_id)

        return total_F

    def share_block(self, block_id: int, with_creature: int) -> bool:
        """Share a block with another creature."""
        block = self.blocks.get(block_id)
        if not block:
            return False

        if not (block.flags & AllocationFlags.SHARED):
            return False

        block.shared_with.add(with_creature)
        return True

    def access_block(self, block_id: int, is_write: bool = False) -> bool:
        """Record access to a block (for LRU, debt tracking)."""
        block = self.blocks.get(block_id)
        if not block:
            return False

        block.access()

        # Writes increase debt slightly (wear)
        if is_write:
            block.q_debt += 0.0001

        return True

    def get_stats(self) -> Dict:
        """Get allocator statistics."""
        return {
            "total_blocks": len(self.blocks),
            "total_F": self.pool.total_F,
            "available_F": self.pool.available_F,
            "used_F": self.pool.total_F - self.pool.available_F,
            "total_memory": self.pool.total_memory,
            "available_memory": self.pool.available_memory,
            "used_memory": self.pool.total_memory - self.pool.available_memory,
            "total_debt": self.pool.total_q,
            "fragmentation": len(self.free_list),
        }

    def defragment(self) -> float:
        """
        Attempt to defragment memory.

        Returns debt reduced by defragmentation.
        """
        # Simple: merge adjacent free blocks
        if len(self.free_list) < 2:
            return 0.0

        # Sort by address
        self.free_list.sort(key=lambda x: x[0])

        # Merge adjacent
        merged = []
        current_addr, current_size = self.free_list[0]

        for addr, size in self.free_list[1:]:
            if addr == current_addr + current_size:
                # Adjacent - merge
                current_size += size
            else:
                merged.append((current_addr, current_size))
                current_addr, current_size = addr, size

        merged.append((current_addr, current_size))
        self.free_list = merged

        # Reduce debt for defragmentation effort
        debt_reduced = min(0.1, self.pool.total_q * 0.1)
        self.pool.total_q -= debt_reduced

        return debt_reduced


__all__ = [
    'AllocationFlags', 'MemoryBlock', 'AllocationResult',
    'ResourcePool', 'ResourceAllocator'
]
