"""
DET Network Integration
=======================

Protocol and interfaces for distributed DET nodes.

Phase 5.3 - Preliminary Implementation.

This module provides the foundation for future network integration:
- Protocol definitions for DET node communication
- Abstract interfaces for external nodes (ESP32, etc.)
- Registry for managing distributed nodes
- Stub transports for serial/network communication

Full implementation deferred to future module.
"""

import json
import time
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import IntEnum
from pathlib import Path


# ============================================================================
# Protocol Definitions
# ============================================================================

class MessageType(IntEnum):
    """DET network message types."""
    # Core messages
    HEARTBEAT = 0x01
    ACK = 0x02
    NACK = 0x03

    # State sync
    STATE_REQUEST = 0x10
    STATE_RESPONSE = 0x11
    STATE_UPDATE = 0x12

    # DET dynamics
    PRESENCE_UPDATE = 0x20
    COHERENCE_UPDATE = 0x21
    AFFECT_UPDATE = 0x22
    BOND_UPDATE = 0x23

    # Stimulus/Response
    STIMULUS_INJECT = 0x30
    STIMULUS_RESPONSE = 0x31

    # Control
    NODE_REGISTER = 0x40
    NODE_DEREGISTER = 0x41
    NODE_CONFIG = 0x42

    # Grace/Recovery
    GRACE_REQUEST = 0x50
    GRACE_INJECT = 0x51


class NodeType(IntEnum):
    """Types of network nodes."""
    UNKNOWN = 0
    ESP32 = 1       # ESP32 microcontroller
    RASPBERRY_PI = 2
    PYTHON_AGENT = 3
    REMOTE_LLM = 4
    SENSOR = 5      # Generic sensor node
    ACTUATOR = 6    # Generic actuator node


class NodeStatus(IntEnum):
    """Status of a network node."""
    UNKNOWN = 0
    CONNECTING = 1
    CONNECTED = 2
    DISCONNECTED = 3
    ERROR = 4
    SLEEPING = 5


@dataclass
class DETMessage:
    """
    DET network protocol message.

    Binary format (for serial/embedded):
    - Header: 2 bytes magic (0xDE, 0x7A)
    - Length: 2 bytes (uint16, payload length)
    - Type: 1 byte (MessageType)
    - Sequence: 2 bytes (uint16)
    - Payload: variable
    - Checksum: 1 byte (XOR of all preceding bytes)
    """
    MAGIC = bytes([0xDE, 0x7A])

    msg_type: MessageType
    sequence: int = 0
    payload: bytes = b""
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        """Serialize to binary format."""
        header = self.MAGIC
        length = struct.pack("<H", len(self.payload))
        msg_type = struct.pack("B", self.msg_type)
        seq = struct.pack("<H", self.sequence)

        data = header + length + msg_type + seq + self.payload

        # XOR checksum
        checksum = 0
        for b in data:
            checksum ^= b

        return data + struct.pack("B", checksum)

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["DETMessage"]:
        """Deserialize from binary format."""
        if len(data) < 8:  # Minimum message size
            return None

        if data[:2] != cls.MAGIC:
            return None

        length = struct.unpack("<H", data[2:4])[0]
        msg_type = MessageType(data[4])
        sequence = struct.unpack("<H", data[5:7])[0]
        payload = data[7:7 + length]

        # Verify checksum
        expected_checksum = data[7 + length]
        checksum = 0
        for b in data[:7 + length]:
            checksum ^= b

        if checksum != expected_checksum:
            return None

        return cls(
            msg_type=msg_type,
            sequence=sequence,
            payload=payload,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.msg_type.name,
            "sequence": self.sequence,
            "payload": self.payload.hex(),
            "timestamp": self.timestamp,
        }


@dataclass
class NodeInfo:
    """Information about a network node."""
    node_id: str
    node_type: NodeType
    name: str = ""
    address: str = ""  # IP or serial port
    status: NodeStatus = NodeStatus.UNKNOWN

    # DET integration
    assigned_nodes: List[int] = field(default_factory=list)  # DET node indices
    domain: Optional[int] = None  # Memory domain if specialized

    # Connection stats
    last_seen: float = 0.0
    message_count: int = 0
    error_count: int = 0

    # Capabilities
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.name,
            "name": self.name,
            "address": self.address,
            "status": self.status.name,
            "assigned_nodes": self.assigned_nodes,
            "domain": self.domain,
            "last_seen": self.last_seen,
            "message_count": self.message_count,
            "capabilities": self.capabilities,
        }


# ============================================================================
# Abstract Interfaces
# ============================================================================

class Transport(ABC):
    """
    Abstract transport interface for network communication.

    Implementations can provide:
    - Serial port communication (for ESP32)
    - TCP/UDP sockets
    - WebSocket connections
    - etc.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close connection."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass

    @abstractmethod
    def send(self, data: bytes) -> bool:
        """Send raw bytes."""
        pass

    @abstractmethod
    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receive raw bytes with timeout."""
        pass


class ExternalNode(ABC):
    """
    Abstract interface for external DET nodes.

    External nodes participate in the DET dynamics as first-class
    cluster members, sharing presence, coherence, and affect.
    """

    @property
    @abstractmethod
    def node_id(self) -> str:
        """Unique node identifier."""
        pass

    @property
    @abstractmethod
    def node_info(self) -> NodeInfo:
        """Node information."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the node."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the node."""
        pass

    @abstractmethod
    def send_message(self, message: DETMessage) -> bool:
        """Send a DET message to the node."""
        pass

    @abstractmethod
    def receive_message(self, timeout: float = 1.0) -> Optional[DETMessage]:
        """Receive a DET message from the node."""
        pass

    @abstractmethod
    def update_state(self, presence: float, coherence: float, affect: tuple):
        """
        Update the node with current DET state.

        Args:
            presence: Current presence value.
            coherence: Current coherence value.
            affect: Tuple of (valence, arousal, bondedness).
        """
        pass

    @abstractmethod
    def get_state(self) -> Optional[Dict[str, float]]:
        """Get the node's current state."""
        pass


# ============================================================================
# Stub Implementations
# ============================================================================

class StubTransport(Transport):
    """
    Stub transport for testing and development.

    Does not perform actual I/O but logs operations.
    """

    def __init__(self, name: str = "stub"):
        self.name = name
        self._connected = False
        self._sent_messages: List[bytes] = []
        self._receive_queue: List[bytes] = []

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def send(self, data: bytes) -> bool:
        if not self._connected:
            return False
        self._sent_messages.append(data)
        return True

    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        if not self._connected:
            return None
        if self._receive_queue:
            return self._receive_queue.pop(0)
        return None

    def inject_receive(self, data: bytes):
        """Inject data into the receive queue (for testing)."""
        self._receive_queue.append(data)


class StubExternalNode(ExternalNode):
    """
    Stub external node for testing and development.

    Simulates an external node without actual hardware.
    """

    def __init__(
        self,
        node_id: str,
        node_type: NodeType = NodeType.PYTHON_AGENT,
        name: str = "stub_node"
    ):
        self._node_id = node_id
        self._info = NodeInfo(
            node_id=node_id,
            node_type=node_type,
            name=name,
            status=NodeStatus.DISCONNECTED,
            capabilities=["state_sync", "stimulus"],
        )
        self._transport = StubTransport()
        self._state = {
            "presence": 0.5,
            "coherence": 0.5,
            "valence": 0.0,
            "arousal": 0.3,
            "bondedness": 0.5,
        }
        self._sequence = 0

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def node_info(self) -> NodeInfo:
        return self._info

    def connect(self) -> bool:
        if self._transport.connect():
            self._info.status = NodeStatus.CONNECTED
            self._info.last_seen = time.time()
            return True
        return False

    def disconnect(self):
        self._transport.disconnect()
        self._info.status = NodeStatus.DISCONNECTED

    def send_message(self, message: DETMessage) -> bool:
        self._info.message_count += 1
        return self._transport.send(message.to_bytes())

    def receive_message(self, timeout: float = 1.0) -> Optional[DETMessage]:
        data = self._transport.receive(timeout)
        if data:
            return DETMessage.from_bytes(data)
        return None

    def update_state(self, presence: float, coherence: float, affect: tuple):
        self._state["presence"] = presence
        self._state["coherence"] = coherence
        self._state["valence"] = affect[0]
        self._state["arousal"] = affect[1]
        self._state["bondedness"] = affect[2]

        # Create and send state update message
        payload = struct.pack("<fffff",
            presence, coherence, affect[0], affect[1], affect[2]
        )
        msg = DETMessage(
            msg_type=MessageType.STATE_UPDATE,
            sequence=self._sequence,
            payload=payload,
        )
        self._sequence += 1
        self.send_message(msg)

    def get_state(self) -> Optional[Dict[str, float]]:
        return self._state.copy()


# ============================================================================
# Network Registry
# ============================================================================

class NetworkRegistry:
    """
    Registry for managing distributed DET nodes.

    Provides:
    - Node registration and discovery
    - State synchronization
    - Message routing
    """

    def __init__(
        self,
        core=None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize the network registry.

        Args:
            core: DETCore instance for state sync.
            storage_path: Path for persistent storage.
        """
        self.core = core
        self.storage_path = storage_path

        self._nodes: Dict[str, ExternalNode] = {}
        self._message_handlers: Dict[MessageType, List[Callable]] = {}

        # Callbacks
        self.on_node_connect: Optional[Callable[[NodeInfo], None]] = None
        self.on_node_disconnect: Optional[Callable[[NodeInfo], None]] = None
        self.on_message: Optional[Callable[[str, DETMessage], None]] = None

    def register_node(self, node: ExternalNode) -> bool:
        """
        Register an external node.

        Args:
            node: ExternalNode instance.

        Returns:
            True if registration succeeded.
        """
        if node.node_id in self._nodes:
            return False

        self._nodes[node.node_id] = node
        return True

    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node.

        Args:
            node_id: Node ID to remove.

        Returns:
            True if node was removed.
        """
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.disconnect()
            del self._nodes[node_id]

            if self.on_node_disconnect:
                self.on_node_disconnect(node.node_info)

            return True
        return False

    def get_node(self, node_id: str) -> Optional[ExternalNode]:
        """Get a registered node by ID."""
        return self._nodes.get(node_id)

    def list_nodes(self) -> List[NodeInfo]:
        """List all registered nodes."""
        return [n.node_info for n in self._nodes.values()]

    def connect_all(self) -> int:
        """
        Connect to all registered nodes.

        Returns:
            Number of successful connections.
        """
        connected = 0
        for node in self._nodes.values():
            if node.connect():
                connected += 1
                if self.on_node_connect:
                    self.on_node_connect(node.node_info)
        return connected

    def disconnect_all(self):
        """Disconnect from all nodes."""
        for node in self._nodes.values():
            node.disconnect()

    def broadcast_state(self):
        """
        Broadcast current DET state to all connected nodes.
        """
        if not self.core:
            return

        presence, coherence, _, _ = self.core.get_aggregates()
        affect = self.core.get_self_affect()

        for node in self._nodes.values():
            if node.node_info.status == NodeStatus.CONNECTED:
                node.update_state(presence, coherence, affect)

    def register_handler(
        self,
        msg_type: MessageType,
        handler: Callable[[str, DETMessage], None]
    ):
        """
        Register a message handler.

        Args:
            msg_type: Message type to handle.
            handler: Callable(node_id, message).
        """
        if msg_type not in self._message_handlers:
            self._message_handlers[msg_type] = []
        self._message_handlers[msg_type].append(handler)

    def poll_messages(self, timeout: float = 0.1):
        """
        Poll all nodes for messages.

        Args:
            timeout: Receive timeout per node.
        """
        for node_id, node in self._nodes.items():
            if node.node_info.status != NodeStatus.CONNECTED:
                continue

            msg = node.receive_message(timeout)
            if msg:
                # Invoke handlers
                handlers = self._message_handlers.get(msg.msg_type, [])
                for handler in handlers:
                    handler(node_id, msg)

                # General callback
                if self.on_message:
                    self.on_message(node_id, msg)

    def get_status(self) -> Dict[str, Any]:
        """Get registry status."""
        nodes = self.list_nodes()
        return {
            "total_nodes": len(nodes),
            "connected": sum(1 for n in nodes if n.status == NodeStatus.CONNECTED),
            "nodes": [n.to_dict() for n in nodes],
            "has_core": self.core is not None,
        }


# ============================================================================
# Future Integration Points
# ============================================================================

# These are placeholder classes that define the interface for future
# hardware-specific implementations.

class SerialTransport(Transport):
    """
    Serial port transport for ESP32 and other microcontrollers.

    TODO: Implement with pyserial when hardware is available.
    """

    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        raise NotImplementedError(
            "SerialTransport requires pyserial. "
            "Install with: pip install pyserial"
        )

    def connect(self) -> bool:
        raise NotImplementedError()

    def disconnect(self):
        raise NotImplementedError()

    def is_connected(self) -> bool:
        raise NotImplementedError()

    def send(self, data: bytes) -> bool:
        raise NotImplementedError()

    def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        raise NotImplementedError()


class ESP32Node(ExternalNode):
    """
    ESP32 external node implementation.

    TODO: Implement when ESP32 firmware is ready.
    """

    def __init__(self, port: str, node_id: Optional[str] = None):
        raise NotImplementedError(
            "ESP32Node requires hardware and firmware. "
            "See docs/esp32_setup.md for instructions."
        )

    @property
    def node_id(self) -> str:
        raise NotImplementedError()

    @property
    def node_info(self) -> NodeInfo:
        raise NotImplementedError()

    def connect(self) -> bool:
        raise NotImplementedError()

    def disconnect(self):
        raise NotImplementedError()

    def send_message(self, message: DETMessage) -> bool:
        raise NotImplementedError()

    def receive_message(self, timeout: float = 1.0) -> Optional[DETMessage]:
        raise NotImplementedError()

    def update_state(self, presence: float, coherence: float, affect: tuple):
        raise NotImplementedError()

    def get_state(self) -> Optional[Dict[str, float]]:
        raise NotImplementedError()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_stub_network(
    core=None,
    num_nodes: int = 2
) -> NetworkRegistry:
    """
    Create a stub network for testing.

    Args:
        core: DETCore instance.
        num_nodes: Number of stub nodes to create.

    Returns:
        Configured NetworkRegistry with stub nodes.
    """
    registry = NetworkRegistry(core=core)

    for i in range(num_nodes):
        node = StubExternalNode(
            node_id=f"stub_{i}",
            name=f"Stub Node {i}",
        )
        registry.register_node(node)

    return registry
