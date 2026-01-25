"""
Creature Runner - Execute Existence-Lang Creatures via EIS VM
==============================================================

Bridges the gap between:
- Compiled creatures (.ex -> bytecode)
- EIS VM execution
- Bond message dispatch

This is the core integration layer for DET-native creature execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import time

from .vm import EISVM, Lane, ExecutionState
from .types import NodeRef, NodeField, WitnessToken
from .encoding import decode_instruction
from .primitives import PrimitiveRegistry, get_registry, PrimitiveCall, PrimitiveResult


class CreatureState(Enum):
    """Creature lifecycle state."""
    SPAWNED = "spawned"
    RUNNING = "running"
    WAITING = "waiting"
    DEAD = "dead"


@dataclass
class KernelPort:
    """A kernel input or output port."""
    name: str
    direction: str  # "in" or "out"
    port_type: str  # "Register" or "TokenReg"
    reg_index: int  # Register index allocated by compiler


@dataclass
class CompiledCreatureData:
    """Compiled creature data for execution."""
    name: str
    init_code: bytes = b''
    kernels: Dict[str, bytes] = field(default_factory=dict)
    kernel_ports: Dict[str, List[KernelPort]] = field(default_factory=dict)  # kernel_name -> ports


@dataclass
class CreatureInstance:
    """A running creature instance."""
    cid: int
    name: str
    compiled: CompiledCreatureData

    # DET state
    F: float = 10.0
    a: float = 0.5
    q: float = 0.0

    # Execution state
    state: CreatureState = CreatureState.SPAWNED

    # Message queues (per bond)
    inbox: Dict[int, List[Dict]] = field(default_factory=dict)
    outbox: Dict[int, List[Dict]] = field(default_factory=dict)

    # Bonds: peer_cid -> channel_id
    bonds: Dict[int, int] = field(default_factory=dict)

    # Statistics
    kernels_executed: int = 0
    messages_received: int = 0
    messages_sent: int = 0


class CreatureRunner:
    """
    Executes compiled Existence-Lang creatures using the EIS VM.

    Responsibilities:
    - Manage creature instances
    - Dispatch bond messages to kernels
    - Execute kernel bytecode
    - Manage creature state (F, a, q)
    """

    def __init__(self, vm: Optional[EISVM] = None,
                 primitive_registry: Optional[PrimitiveRegistry] = None):
        self.vm = vm or EISVM()

        # Creature registry
        self.creatures: Dict[int, CreatureInstance] = {}
        self.next_cid: int = 1

        # Channel registry
        self.channels: Dict[int, 'Channel'] = {}
        self.next_channel_id: int = 0

        # Primitive registry (for external I/O)
        self.primitive_registry = primitive_registry or get_registry()

        # Global tick counter
        self.tick: int = 0

    def register_primitive(self, name: str, handler: Callable):
        """Register a primitive function callable from EL."""
        self.primitives[name] = handler

    def spawn(self, compiled: CompiledCreatureData,
              initial_f: float = 10.0,
              initial_a: float = 0.5) -> int:
        """
        Spawn a new creature instance.

        Returns the creature ID (cid).
        """
        cid = self.next_cid
        self.next_cid += 1

        instance = CreatureInstance(
            cid=cid,
            name=compiled.name,
            compiled=compiled,
            F=initial_f,
            a=initial_a,
            state=CreatureState.SPAWNED
        )

        self.creatures[cid] = instance

        # Create VM node lane for this creature
        self.vm.create_node_lane(cid, compiled.init_code)

        # Run initialization code
        if compiled.init_code:
            self._run_init(instance)

        instance.state = CreatureState.RUNNING
        return cid

    def _run_init(self, instance: CreatureInstance):
        """Run creature initialization code."""
        lane = self.vm.node_lanes.get(instance.cid)
        if lane:
            # Set initial state in trace
            node_ref = NodeRef(instance.cid)
            self.vm.trace.write_node(node_ref, NodeField.F, instance.F)
            self.vm.trace.write_node(node_ref, NodeField.A, instance.a)

            # Run init bytecode
            self.vm.run_lane(lane)

            # Read back any modified state
            instance.F = self.vm.trace.read_node(node_ref, NodeField.F)
            instance.a = self.vm.trace.read_node(node_ref, NodeField.A)

    def bond(self, cid_a: int, cid_b: int, coherence: float = 1.0) -> int:
        """
        Create a bond between two creatures.

        Returns the channel ID.
        """
        if cid_a not in self.creatures or cid_b not in self.creatures:
            raise ValueError("Invalid creature ID")

        channel_id = self.next_channel_id
        self.next_channel_id += 1

        channel = Channel(
            channel_id=channel_id,
            creature_a=cid_a,
            creature_b=cid_b,
            coherence=coherence
        )
        self.channels[channel_id] = channel

        # Register bond in both creatures
        self.creatures[cid_a].bonds[cid_b] = channel_id
        self.creatures[cid_a].inbox[cid_b] = []
        self.creatures[cid_a].outbox[cid_b] = []

        self.creatures[cid_b].bonds[cid_a] = channel_id
        self.creatures[cid_b].inbox[cid_a] = []
        self.creatures[cid_b].outbox[cid_a] = []

        return channel_id

    def send(self, from_cid: int, to_cid: int, message: Dict) -> bool:
        """
        Send a message from one creature to another via their bond.

        Returns True if sent successfully.
        """
        if from_cid not in self.creatures or to_cid not in self.creatures:
            return False

        sender = self.creatures[from_cid]
        receiver = self.creatures[to_cid]

        # Check bond exists
        if to_cid not in sender.bonds:
            return False

        channel_id = sender.bonds[to_cid]
        channel = self.channels.get(channel_id)
        if not channel:
            return False

        # Check coherence (message may be lost)
        import random
        if random.random() > channel.coherence:
            return False  # Message lost due to low coherence

        # Queue message
        receiver.inbox[from_cid].append(message)
        receiver.messages_received += 1
        sender.messages_sent += 1

        return True

    def receive(self, cid: int, from_cid: int) -> Optional[Dict]:
        """Receive a message from a specific peer."""
        if cid not in self.creatures:
            return None

        creature = self.creatures[cid]
        if from_cid not in creature.inbox:
            return None

        if creature.inbox[from_cid]:
            return creature.inbox[from_cid].pop(0)
        return None

    def receive_all(self, cid: int) -> List[tuple]:
        """Receive all pending messages. Returns [(from_cid, message), ...]"""
        if cid not in self.creatures:
            return []

        creature = self.creatures[cid]
        messages = []

        for from_cid, inbox in creature.inbox.items():
            while inbox:
                messages.append((from_cid, inbox.pop(0)))

        return messages

    def invoke_kernel(self, cid: int, kernel_name: str,
                      inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Invoke a kernel on a creature.

        Args:
            cid: Creature ID
            kernel_name: Name of kernel to invoke
            inputs: Input values for kernel ports

        Returns:
            Dict with outputs and execution info
        """
        if cid not in self.creatures:
            return {"error": "Invalid creature ID"}

        creature = self.creatures[cid]

        # Check kernel exists
        if kernel_name not in creature.compiled.kernels:
            return {"error": f"Unknown kernel: {kernel_name}"}

        kernel_code = creature.compiled.kernels[kernel_name]

        # Check F cost (base cost)
        base_cost = 0.1
        if creature.F < base_cost:
            return {"error": "Insufficient F", "F": creature.F}

        # Get or create lane
        lane = self.vm.node_lanes.get(cid)
        if not lane:
            lane = self.vm.create_node_lane(cid, kernel_code)
        else:
            # Update lane program to kernel code
            lane.program = kernel_code
            lane.pc = 0
            lane.state = ExecutionState.RUNNING

        # Set up inputs in registers using port mapping
        if inputs:
            ports = creature.compiled.kernel_ports.get(kernel_name, [])
            port_map = {p.name: p for p in ports if p.direction == "in"}

            for name, value in inputs.items():
                if name in port_map:
                    port = port_map[name]
                    reg_idx = port.reg_index
                    # Write to lane registers
                    if lane.registers:
                        if port.port_type == "Register":
                            # Scalar registers are at global indices 0-15
                            lane.registers.write(reg_idx, float(value) if isinstance(value, (int, float)) else 0.0)
                        elif port.port_type == "TokenReg":
                            # Token registers are at global indices 24-31
                            # The reg_index from compiler is class-local (0-7)
                            global_idx = 24 + reg_idx
                            lane.registers.write(global_idx, str(value) if value else "")

        # Sync creature state to VM trace
        node_ref = NodeRef(cid)
        self.vm.trace.write_node(node_ref, NodeField.F, creature.F)
        self.vm.trace.write_node(node_ref, NodeField.A, creature.a)

        # Execute kernel phases
        start_time = time.time()

        # Run through all phases
        self.vm.run_tick()

        elapsed_ms = (time.time() - start_time) * 1000

        # Read back state
        creature.F = self.vm.trace.read_node(node_ref, NodeField.F)
        creature.a = self.vm.trace.read_node(node_ref, NodeField.A)
        creature.kernels_executed += 1

        # Deduct base cost
        creature.F -= base_cost

        # Collect outputs from registers using port mapping
        outputs = {}
        ports = creature.compiled.kernel_ports.get(kernel_name, [])
        for port in ports:
            if port.direction == "out":
                if lane.registers:
                    if port.port_type == "Register":
                        # Scalar registers at global indices 0-15
                        outputs[port.name] = lane.registers.read(port.reg_index)
                    elif port.port_type == "TokenReg":
                        # Token registers at global indices 24-31
                        global_idx = 24 + port.reg_index
                        outputs[port.name] = lane.registers.read(global_idx)

        return {
            "success": True,
            "kernel": kernel_name,
            "outputs": outputs,
            "elapsed_ms": elapsed_ms,
            "F": creature.F
        }

    def call_primitive(self, cid: int, primitive_name: str,
                        args: List[Any] = None) -> Dict[str, Any]:
        """
        Call a primitive function on behalf of a creature.

        Args:
            cid: Creature ID
            primitive_name: Name of primitive to call
            args: Arguments for the primitive

        Returns:
            Dict with result and cost
        """
        if cid not in self.creatures:
            return {"error": "Invalid creature ID", "success": False}

        creature = self.creatures[cid]
        args = args or []

        # Call primitive with creature's F and agency
        call = self.primitive_registry.call(
            primitive_name,
            args,
            available_f=creature.F,
            agency=creature.a
        )

        # Deduct cost if successful
        if call.result_code == PrimitiveResult.OK:
            creature.F -= call.cost

        return {
            "success": call.result_code == PrimitiveResult.OK,
            "result": call.result,
            "result_code": call.result_code.name,
            "cost": call.cost,
            "elapsed_ms": call.elapsed_ms,
            "F": creature.F
        }

    def process_messages(self, cid: int):
        """
        Process pending messages for a creature.

        Messages with type "invoke" trigger kernel execution.
        Messages with type "primitive" trigger primitive calls.
        """
        if cid not in self.creatures:
            return

        creature = self.creatures[cid]
        messages = self.receive_all(cid)

        for from_cid, msg in messages:
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type")

            if msg_type == "invoke":
                # Invoke a kernel
                kernel = msg.get("kernel")
                inputs = msg.get("inputs", {})

                result = self.invoke_kernel(cid, kernel, inputs)

                # Send response back
                self.send(cid, from_cid, {
                    "type": "result",
                    "kernel": kernel,
                    **result
                })

            elif msg_type == "primitive":
                # Call a primitive
                name = msg.get("name")
                args = msg.get("args", [])

                result = self.call_primitive(cid, name, args)

                # Send response back
                self.send(cid, from_cid, {
                    "type": "primitive_result",
                    "name": name,
                    **result
                })

            elif msg_type == "ping":
                # Simple ping/pong
                self.send(cid, from_cid, {
                    "type": "pong",
                    "cid": cid,
                    "F": creature.F
                })

    def tick(self):
        """Advance the global tick and process all creatures."""
        self.tick += 1

        # Process messages for all creatures
        for cid in list(self.creatures.keys()):
            creature = self.creatures[cid]
            if creature.state == CreatureState.RUNNING:
                self.process_messages(cid)

    def get_creature_state(self, cid: int) -> Optional[Dict]:
        """Get creature state."""
        if cid not in self.creatures:
            return None

        c = self.creatures[cid]
        return {
            "cid": c.cid,
            "name": c.name,
            "F": c.F,
            "a": c.a,
            "state": c.state.value,
            "bonds": list(c.bonds.keys()),
            "kernels": list(c.compiled.kernels.keys()),
            "kernels_executed": c.kernels_executed,
            "messages_received": c.messages_received,
            "messages_sent": c.messages_sent
        }


@dataclass
class Channel:
    """A bond channel between two creatures."""
    channel_id: int
    creature_a: int
    creature_b: int
    coherence: float = 1.0
    last_activity: float = field(default_factory=time.time)


def compile_creature_for_runner(compiled_program) -> Dict[str, CompiledCreatureData]:
    """
    Convert compiled program to runner format.

    Takes output from EISCompiler and returns dict of CompiledCreatureData.
    """
    result = {}

    for name, compiled_creature in compiled_program.creatures.items():
        data = CompiledCreatureData(
            name=name,
            init_code=compiled_creature.init_code,
        )

        # Extract kernels and port information
        if hasattr(compiled_creature, 'kernels'):
            for kernel_name, compiled_kernel in compiled_creature.kernels.items():
                data.kernels[kernel_name] = compiled_kernel.code

                # Convert CompiledPort from compiler to KernelPort for runner
                ports = []
                if hasattr(compiled_kernel, 'ports') and compiled_kernel.ports:
                    for cp in compiled_kernel.ports:
                        ports.append(KernelPort(
                            name=cp.name,
                            direction=cp.direction,
                            port_type=cp.port_type,
                            reg_index=cp.reg_index
                        ))
                data.kernel_ports[kernel_name] = ports

        result[name] = data

    return result
