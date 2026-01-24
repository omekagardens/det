"""
DET-OS Bond-Based IPC
=====================

Inter-process communication in DET-OS uses bonds - coherent connections
between creatures. Unlike traditional IPC mechanisms:
    - Channels have coherence (C) that affects reliability
    - Message passing costs F (conserved)
    - High coherence = reliable delivery
    - Low coherence = message loss/corruption
    - Bonds strengthen with use, weaken with neglect

IPC Primitives:
    - BondChannel: Point-to-point coherent channel
    - MessageQueue: Async message buffer
    - SharedRegister: Bond-backed shared state

Key principle: Communication requires coherence. Incoherent creatures
cannot reliably exchange information.
"""

from enum import Enum, Flag, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generic, TypeVar
from collections import deque
import time
import random


class ChannelFlags(Flag):
    """Channel configuration flags."""
    NONE = 0
    RELIABLE = auto()     # Guarantee delivery (costs more F)
    ORDERED = auto()      # Maintain message order
    BROADCAST = auto()    # One-to-many
    SYNC = auto()         # Synchronous (blocking)
    BUFFERED = auto()     # Allow buffering


class IPCResult(Enum):
    """Result of IPC operation."""
    SUCCESS = auto()
    BLOCKED = auto()       # Would block
    NO_COHERENCE = auto()  # Channel too incoherent
    NO_RESOURCE = auto()   # Insufficient F
    CHANNEL_CLOSED = auto()
    QUEUE_FULL = auto()
    QUEUE_EMPTY = auto()
    TIMEOUT = auto()


@dataclass
class Message:
    """A message sent through a bond channel."""
    msg_id: int
    sender: int           # Creature ID
    receiver: int         # Creature ID (or -1 for broadcast)
    payload: Any          # Message content
    F_cost: float         # Resource cost
    timestamp: float = field(default_factory=time.time)
    coherence_at_send: float = 1.0  # Channel coherence when sent

    def size_estimate(self) -> int:
        """Estimate message size in bytes."""
        # Rough estimate based on payload
        if isinstance(self.payload, (bytes, bytearray)):
            return len(self.payload)
        elif isinstance(self.payload, str):
            return len(self.payload.encode())
        elif isinstance(self.payload, (int, float)):
            return 8
        elif isinstance(self.payload, dict):
            return 64 + len(str(self.payload))
        else:
            return 64


@dataclass
class MessageQueue:
    """Queue for async messages."""
    capacity: int = 64
    queue: deque = field(default_factory=deque)

    def push(self, msg: Message) -> IPCResult:
        """Push message to queue."""
        if len(self.queue) >= self.capacity:
            return IPCResult.QUEUE_FULL
        self.queue.append(msg)
        return IPCResult.SUCCESS

    def pop(self) -> tuple:
        """Pop message from queue. Returns (result, message)."""
        if not self.queue:
            return IPCResult.QUEUE_EMPTY, None
        return IPCResult.SUCCESS, self.queue.popleft()

    def peek(self) -> Optional[Message]:
        """Peek at front message without removing."""
        if self.queue:
            return self.queue[0]
        return None

    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def is_full(self) -> bool:
        return len(self.queue) >= self.capacity

    def size(self) -> int:
        return len(self.queue)


@dataclass
class BondChannel:
    """
    A coherent communication channel between creatures.

    The channel's coherence (C) determines reliability:
    - C ≈ 1.0: Perfect transmission
    - C ≈ 0.5: 50% chance of successful delivery
    - C ≈ 0.0: No communication possible

    Coherence dynamics:
    - Successful sends increase C
    - Failed sends/timeouts decrease C
    - Idle channels decay toward 0
    - Messages strengthen bonds
    """

    channel_id: int
    creature_a: int          # First endpoint creature ID
    creature_b: int          # Second endpoint creature ID
    flags: ChannelFlags = ChannelFlags.ORDERED

    # DET properties
    coherence: float = 0.5   # Channel coherence C ∈ [0, 1]
    sigma: float = 0.1       # Conductivity (bandwidth)

    # Buffers
    queue_a_to_b: MessageQueue = field(default_factory=MessageQueue)
    queue_b_to_a: MessageQueue = field(default_factory=MessageQueue)

    # Statistics
    messages_sent: int = 0
    messages_received: int = 0
    messages_lost: int = 0
    total_F_transferred: float = 0.0

    # Timing
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    # State
    closed: bool = False
    next_msg_id: int = 0

    def _get_queue(self, sender: int) -> MessageQueue:
        """Get the appropriate queue for sender."""
        if sender == self.creature_a:
            return self.queue_a_to_b
        else:
            return self.queue_b_to_a

    def _get_receiver(self, sender: int) -> int:
        """Get receiver for a given sender."""
        if sender == self.creature_a:
            return self.creature_b
        else:
            return self.creature_a

    def send(self, sender: int, payload: Any, F_available: float) -> tuple:
        """
        Send a message through the channel.

        Args:
            sender: Sender creature ID
            payload: Message content
            F_available: Sender's available F

        Returns:
            (IPCResult, F_consumed)
        """
        if self.closed:
            return IPCResult.CHANNEL_CLOSED, 0.0

        if sender not in (self.creature_a, self.creature_b):
            return IPCResult.NO_COHERENCE, 0.0

        # Calculate F cost (based on message size and coherence)
        msg_size = 64  # Base estimate
        if isinstance(payload, (bytes, bytearray)):
            msg_size = len(payload)

        # Cost increases with low coherence (retransmission overhead)
        coherence_factor = 1.0 / max(0.1, self.coherence)
        F_cost = (msg_size / 1024) * 0.01 * coherence_factor

        if F_available < F_cost:
            return IPCResult.NO_RESOURCE, 0.0

        # Check coherence threshold
        if self.coherence < 0.1 and not (self.flags & ChannelFlags.RELIABLE):
            return IPCResult.NO_COHERENCE, 0.0

        # Probabilistic delivery based on coherence
        delivery_prob = self.coherence if not (self.flags & ChannelFlags.RELIABLE) else 1.0

        if random.random() > delivery_prob:
            # Message lost due to incoherence
            self.messages_lost += 1
            self.coherence = max(0, self.coherence - 0.05)
            return IPCResult.NO_COHERENCE, F_cost * 0.5  # Partial cost for attempt

        # Create message
        msg = Message(
            msg_id=self.next_msg_id,
            sender=sender,
            receiver=self._get_receiver(sender),
            payload=payload,
            F_cost=F_cost,
            coherence_at_send=self.coherence
        )
        self.next_msg_id += 1

        # Push to queue
        queue = self._get_queue(sender)
        result = queue.push(msg)

        if result == IPCResult.SUCCESS:
            self.messages_sent += 1
            self.total_F_transferred += F_cost
            self.last_activity = time.time()
            # Successful send strengthens coherence
            self.coherence = min(1.0, self.coherence + 0.01)

        return result, F_cost

    def receive(self, receiver: int) -> tuple:
        """
        Receive a message from the channel.

        Args:
            receiver: Receiver creature ID

        Returns:
            (IPCResult, Message or None)
        """
        if self.closed:
            return IPCResult.CHANNEL_CLOSED, None

        if receiver not in (self.creature_a, self.creature_b):
            return IPCResult.NO_COHERENCE, None

        # Get queue where messages TO receiver are stored
        if receiver == self.creature_b:
            queue = self.queue_a_to_b
        else:
            queue = self.queue_b_to_a

        result, msg = queue.pop()

        if result == IPCResult.SUCCESS:
            self.messages_received += 1
            self.last_activity = time.time()
            # Successful receive strengthens coherence
            self.coherence = min(1.0, self.coherence + 0.01)

        return result, msg

    def tick(self, dt: float = 0.02):
        """Update channel for one tick."""
        if self.closed:
            return

        # Coherence decay when idle
        idle_time = time.time() - self.last_activity
        if idle_time > 1.0:
            decay = dt * 0.01 * idle_time
            self.coherence = max(0, self.coherence - decay)

        # Close channel if coherence too low
        if self.coherence <= 0:
            self.closed = True

    def close(self):
        """Close the channel."""
        self.closed = True


class IPCManager:
    """
    Manager for all IPC channels in the system.

    Tracks channels, handles creation/destruction, and provides
    routing for messages.
    """

    def __init__(self):
        self.channels: Dict[int, BondChannel] = {}
        self.next_channel_id = 0

        # Index: creature -> channels it's part of
        self.creature_channels: Dict[int, Set[int]] = {}

    def create_channel(self,
                       creature_a: int,
                       creature_b: int,
                       flags: ChannelFlags = ChannelFlags.ORDERED,
                       initial_coherence: float = 0.5) -> BondChannel:
        """Create a new channel between two creatures."""
        channel_id = self.next_channel_id
        self.next_channel_id += 1

        channel = BondChannel(
            channel_id=channel_id,
            creature_a=creature_a,
            creature_b=creature_b,
            flags=flags,
            coherence=initial_coherence
        )

        self.channels[channel_id] = channel

        # Update index
        for cid in (creature_a, creature_b):
            if cid not in self.creature_channels:
                self.creature_channels[cid] = set()
            self.creature_channels[cid].add(channel_id)

        return channel

    def get_channel(self, channel_id: int) -> Optional[BondChannel]:
        """Get channel by ID."""
        return self.channels.get(channel_id)

    def find_channel(self, creature_a: int, creature_b: int) -> Optional[BondChannel]:
        """Find channel between two creatures."""
        if creature_a not in self.creature_channels:
            return None

        for channel_id in self.creature_channels[creature_a]:
            channel = self.channels[channel_id]
            if creature_b in (channel.creature_a, channel.creature_b):
                return channel

        return None

    def close_channel(self, channel_id: int):
        """Close and remove a channel."""
        channel = self.channels.get(channel_id)
        if not channel:
            return

        channel.close()

        # Remove from index
        for cid in (channel.creature_a, channel.creature_b):
            if cid in self.creature_channels:
                self.creature_channels[cid].discard(channel_id)

        del self.channels[channel_id]

    def close_all_for_creature(self, creature_id: int):
        """Close all channels for a creature."""
        if creature_id not in self.creature_channels:
            return

        channel_ids = list(self.creature_channels[creature_id])
        for channel_id in channel_ids:
            self.close_channel(channel_id)

    def tick(self, dt: float = 0.02):
        """Update all channels."""
        closed = []
        for channel_id, channel in self.channels.items():
            channel.tick(dt)
            if channel.closed:
                closed.append(channel_id)

        for channel_id in closed:
            self.close_channel(channel_id)

    def get_stats(self) -> Dict:
        """Get IPC statistics."""
        total_sent = sum(c.messages_sent for c in self.channels.values())
        total_received = sum(c.messages_received for c in self.channels.values())
        total_lost = sum(c.messages_lost for c in self.channels.values())
        avg_coherence = (
            sum(c.coherence for c in self.channels.values()) / len(self.channels)
            if self.channels else 0.0
        )

        return {
            "num_channels": len(self.channels),
            "total_sent": total_sent,
            "total_received": total_received,
            "total_lost": total_lost,
            "avg_coherence": avg_coherence,
        }


__all__ = [
    'ChannelFlags', 'IPCResult', 'Message', 'MessageQueue',
    'BondChannel', 'IPCManager'
]
