"""
Base Creature Wrapper
=====================

Base class for creature wrappers that add behavior to RuntimeCreature.
"""

from typing import Optional, Dict, Any, List
from ..existence.runtime import ExistenceKernelRuntime, RuntimeCreature, CreatureState


class CreatureWrapper:
    """
    Base wrapper for DET-OS creatures.

    Provides common functionality for creatures:
    - Property access to DET state (F, a, presence)
    - Bond management
    - Message sending/receiving via bonds
    """

    def __init__(self, runtime: ExistenceKernelRuntime, cid: int):
        self.runtime = runtime
        self.cid = cid
        self.bonds: Dict[int, int] = {}  # peer_cid -> channel_id

    @property
    def creature(self) -> Optional[RuntimeCreature]:
        """Get the underlying runtime creature."""
        return self.runtime.creatures.get(self.cid)

    @property
    def F(self) -> float:
        """Current resource level."""
        c = self.creature
        return c.F if c else 0.0

    @F.setter
    def F(self, value: float):
        """Set resource level."""
        c = self.creature
        if c:
            c.F = value

    @property
    def a(self) -> float:
        """Current agency level."""
        c = self.creature
        return c.a if c else 0.0

    @property
    def presence(self) -> float:
        """Compute presence: P = F * a."""
        c = self.creature
        if c:
            return c.compute_presence()
        return 0.0

    @property
    def is_alive(self) -> bool:
        """Check if creature is still alive."""
        c = self.creature
        return c is not None and c.is_alive()

    @property
    def state(self) -> CreatureState:
        """Get creature state."""
        c = self.creature
        return c.state if c else CreatureState.DEAD

    def bond_with(self, other_cid: int, coherence: float = 0.8) -> int:
        """
        Create a bond with another creature.

        Returns the channel_id for this bond.
        """
        if other_cid in self.bonds:
            return self.bonds[other_cid]

        channel_id = self.runtime.create_channel(self.cid, other_cid, coherence)
        self.bonds[other_cid] = channel_id
        return channel_id

    def send_to(self, peer_cid: int, message: Any) -> bool:
        """Send a message to a bonded peer."""
        if peer_cid not in self.bonds:
            return False

        channel_id = self.bonds[peer_cid]
        return self.runtime.send(self.cid, channel_id, message)

    def receive_from(self, peer_cid: int) -> Optional[Any]:
        """Receive a message from a bonded peer."""
        if peer_cid not in self.bonds:
            return None

        channel_id = self.bonds[peer_cid]
        return self.runtime.receive(self.cid, channel_id)

    def receive_all_from(self, peer_cid: int) -> List[Any]:
        """Receive all pending messages from a bonded peer."""
        messages = []
        while True:
            msg = self.receive_from(peer_cid)
            if msg is None:
                break
            messages.append(msg)
        return messages

    def get_bond_coherence(self, peer_cid: int) -> float:
        """Get the coherence of a bond with a peer."""
        if peer_cid not in self.bonds:
            return 0.0

        channel_id = self.bonds[peer_cid]
        channel = self.runtime.channels.get(channel_id)
        return channel.coherence if channel else 0.0

    def inject_resource(self, amount: float):
        """Inject resource (grace) into the creature."""
        c = self.creature
        if c:
            c.F += amount

    def get_state_dict(self) -> Dict[str, Any]:
        """Get creature state as dictionary."""
        c = self.creature
        if not c:
            return {"status": "dead"}

        return {
            "cid": self.cid,
            "name": c.name,
            "F": c.F,
            "a": c.a,
            "presence": self.presence,
            "state": c.state.name,
            "bonds": list(self.bonds.keys()),
        }
