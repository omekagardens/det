"""
DET Test Harness
================

CLI-based debugging and probing tool for the DET core.

Phase 6.1/6.3 - Development Tools

Provides:
- Resource injection (F, q manipulation)
- Bond manipulation (create, destroy, modify C)
- Time controls (pause, step, speed)
- State inspection (nodes, bonds, self-cluster, affect)
- Event logging and replay
"""

import cmd
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from pathlib import Path
from datetime import datetime


class HarnessEventType(Enum):
    """Types of harness events for logging."""
    INJECT_F = "inject_f"
    INJECT_Q = "inject_q"
    SET_AGENCY = "set_agency"
    CREATE_BOND = "create_bond"
    DESTROY_BOND = "destroy_bond"
    SET_COHERENCE = "set_coherence"
    STEP = "step"
    PAUSE = "pause"
    RESUME = "resume"
    SPEED_CHANGE = "speed_change"
    SNAPSHOT = "snapshot"
    RESTORE = "restore"


@dataclass
class HarnessEvent:
    """A logged harness event."""
    event_type: HarnessEventType
    timestamp: float
    tick: int
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "tick": self.tick,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessEvent":
        return cls(
            event_type=HarnessEventType(data["type"]),
            timestamp=data["timestamp"],
            tick=data["tick"],
            params=data.get("params", {}),
        )


@dataclass
class Snapshot:
    """A snapshot of DET state for comparison/restore."""
    name: str
    timestamp: float
    tick: int
    state_data: bytes  # Serialized DET state
    metadata: Dict[str, Any] = field(default_factory=dict)


class HarnessController:
    """
    Controller for DET test harness operations.

    Provides programmatic access to all harness functionality.
    Can be used directly or through the CLI interface.
    """

    def __init__(self, core=None, storage_path: Optional[Path] = None, start_paused: bool = False):
        """
        Initialize the harness controller.

        Args:
            core: DETCore instance to control.
            storage_path: Path for storing logs, snapshots, etc.
            start_paused: Whether to start in paused state.
        """
        self.core = core
        self.storage_path = storage_path

        # Time control state
        self._paused = start_paused
        self._speed = 1.0  # 1.0 = normal, 2.0 = 2x, 0.5 = half
        self._step_mode = False
        self._steps_remaining = 0

        # Event logging
        self._events: List[HarnessEvent] = []
        self._event_callbacks: List[Callable[[HarnessEvent], None]] = []

        # Snapshots
        self._snapshots: Dict[str, Snapshot] = {}

        # Auto-run thread
        self._auto_run_thread: Optional[threading.Thread] = None
        self._auto_run_stop = threading.Event()

        # Watchers (callbacks when specific conditions met)
        self._watchers: Dict[str, Callable[[], bool]] = {}

        if storage_path:
            storage_path.mkdir(parents=True, exist_ok=True)

    def _log_event(self, event_type: HarnessEventType, **params):
        """Log a harness event."""
        tick = self.core.tick if self.core else 0
        event = HarnessEvent(
            event_type=event_type,
            timestamp=time.time(),
            tick=tick,
            params=params,
        )
        self._events.append(event)

        # Notify callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def add_event_callback(self, callback: Callable[[HarnessEvent], None]):
        """Add a callback for harness events."""
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: Callable[[HarnessEvent], None]):
        """Remove an event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    # =========================================================================
    # Resource Injection
    # =========================================================================

    def inject_f(self, node_index: int, amount: float) -> bool:
        """
        Inject resource F into a specific node.

        Args:
            node_index: Index of the node (in P-layer or A-layer).
            amount: Amount of F to inject (can be negative to drain).

        Returns:
            True if successful.
        """
        if not self.core:
            return False

        if node_index < 0 or node_index >= self.core.num_active:
            return False

        # Access the node directly via ctypes struct
        node = self.core._core.contents.nodes[node_index]
        current_f = node.F
        new_f = max(0.0, current_f + amount)
        node.F = new_f

        self._log_event(HarnessEventType.INJECT_F, node=node_index, amount=amount, new_value=new_f)
        return True

    def inject_q(self, node_index: int, amount: float) -> bool:
        """
        Inject structural debt q into a specific node.

        Args:
            node_index: Index of the node.
            amount: Amount of q to add (can be negative to reduce).

        Returns:
            True if successful.
        """
        if not self.core:
            return False

        if node_index < 0 or node_index >= self.core.num_active:
            return False

        node = self.core._core.contents.nodes[node_index]
        current_q = node.q
        new_q = max(0.0, current_q + amount)
        node.q = new_q

        self._log_event(HarnessEventType.INJECT_Q, node=node_index, amount=amount, new_value=new_q)
        return True

    def set_agency(self, node_index: int, value: float) -> bool:
        """
        Set agency for a node (use with caution - agency should be inviolable).

        Args:
            node_index: Index of the node.
            value: New agency value (0.0 to 1.0).

        Returns:
            True if successful.
        """
        if not self.core:
            return False

        if node_index < 0 or node_index >= self.core.num_active:
            return False

        value = max(0.0, min(1.0, value))
        node = self.core._core.contents.nodes[node_index]
        node.a = value

        self._log_event(HarnessEventType.SET_AGENCY, node=node_index, value=value)
        return True

    def inject_f_all(self, amount: float, layer: Optional[str] = None) -> int:
        """
        Inject F into all nodes (or a specific layer).

        Args:
            amount: Amount to inject per node.
            layer: "P" for P-layer only, "A" for A-layer only, None for all.

        Returns:
            Number of nodes affected.
        """
        if not self.core:
            return 0

        count = 0
        p_size = 16  # DET_P_LAYER_SIZE

        for i in range(self.core.num_active):
            in_p_layer = i < p_size

            if layer == "P" and not in_p_layer:
                continue
            if layer == "A" and in_p_layer:
                continue

            if self.inject_f(i, amount):
                count += 1

        return count

    # =========================================================================
    # Bond Manipulation
    # =========================================================================

    def create_bond(self, node_i: int, node_j: int, coherence: float = 0.5) -> bool:
        """
        Create a bond between two nodes.

        Args:
            node_i: First node index.
            node_j: Second node index.
            coherence: Initial coherence value.

        Returns:
            True if bond was created.
        """
        if not self.core:
            return False

        if node_i == node_j:
            return False

        if node_i < 0 or node_i >= self.core.num_active:
            return False
        if node_j < 0 or node_j >= self.core.num_active:
            return False

        coherence = max(0.0, min(1.0, coherence))

        # Use the existing C function to create the bond
        bond_idx = self.core._lib.det_core_create_bond(self.core._core, node_i, node_j)

        if bond_idx >= 0:
            # Set the initial coherence
            self.core._core.contents.bonds[bond_idx].C = coherence
            self._log_event(HarnessEventType.CREATE_BOND, node_i=node_i, node_j=node_j, coherence=coherence)
            return True

        return False

    def destroy_bond(self, node_i: int, node_j: int) -> bool:
        """
        Destroy a bond between two nodes.

        Args:
            node_i: First node index.
            node_j: Second node index.

        Returns:
            True if bond was destroyed.
        """
        if not self.core:
            return False

        # Find the bond
        bond_idx = self.core._lib.det_core_find_bond(self.core._core, node_i, node_j)
        if bond_idx < 0:
            return False

        # Mark the bond as inactive by setting coherence to 0
        # (Full removal would require shifting the array, so we just deactivate)
        self.core._core.contents.bonds[bond_idx].C = 0.0
        self._log_event(HarnessEventType.DESTROY_BOND, node_i=node_i, node_j=node_j)

        return True

    def set_coherence(self, node_i: int, node_j: int, coherence: float) -> bool:
        """
        Set coherence for an existing bond.

        Args:
            node_i: First node index.
            node_j: Second node index.
            coherence: New coherence value.

        Returns:
            True if successful.
        """
        if not self.core:
            return False

        coherence = max(0.0, min(1.0, coherence))

        # Find the bond
        bond_idx = self.core._lib.det_core_find_bond(self.core._core, node_i, node_j)
        if bond_idx < 0:
            return False

        self.core._core.contents.bonds[bond_idx].C = coherence
        self._log_event(HarnessEventType.SET_COHERENCE, node_i=node_i, node_j=node_j, coherence=coherence)

        return True

    # =========================================================================
    # Advanced Probing (Phase 6.3)
    # =========================================================================

    def trigger_escalation(self, node_index: int) -> bool:
        """
        Trigger escalation on a specific node.

        Args:
            node_index: Index of the node to escalate.

        Returns:
            True if escalation was triggered.
        """
        if not self.core:
            return False

        if node_index < 0 or node_index >= self.core.num_active:
            return False

        node = self.core._core.contents.nodes[node_index]
        node.escalation_pending = True
        node.novelty_score = 1.0  # High novelty triggers escalation

        self._log_event(HarnessEventType.STEP, action="escalation", node=node_index)
        return True

    def inject_grace(self, node_index: int, amount: float) -> bool:
        """
        Inject grace into a node (boundary recovery mechanism).

        Args:
            node_index: Index of the node.
            amount: Amount of grace to inject.

        Returns:
            True if successful.
        """
        if not self.core:
            return False

        if node_index < 0 or node_index >= self.core.num_active:
            return False

        # Use the C function for grace injection
        self.core._lib.det_core_inject_grace(self.core._core, node_index, amount)

        self._log_event(HarnessEventType.INJECT_F, action="grace", node=node_index, amount=amount)
        return True

    def inject_grace_all(self, amount: float) -> int:
        """
        Inject grace into all nodes that need it.

        Args:
            amount: Amount of grace per node.

        Returns:
            Number of nodes that received grace.
        """
        if not self.core:
            return 0

        count = 0
        for i in range(self.core.num_active):
            if self.core._lib.det_core_needs_grace(self.core._core, i):
                self.core._lib.det_core_inject_grace(self.core._core, i, amount)
                count += 1

        return count

    def activate_domain(self, name: str, num_nodes: int, initial_coherence: float = 0.3) -> bool:
        """
        Activate a new domain by recruiting dormant nodes.

        Args:
            name: Name for the domain.
            num_nodes: Number of nodes to recruit.
            initial_coherence: Initial coherence for domain bonds.

        Returns:
            True if domain was activated.
        """
        if not self.core:
            return False

        result = self.core._lib.det_core_activate_domain(
            self.core._core,
            name.encode('utf-8'),
            num_nodes,
            initial_coherence
        )

        if result:
            self._log_event(HarnessEventType.STEP, action="activate_domain", name=name, nodes=num_nodes)

        return result

    def transfer_pattern(self, source_domain: int, target_domain: int, strength: float = 0.5) -> bool:
        """
        Transfer pattern from source to target domain.

        Args:
            source_domain: Source domain index.
            target_domain: Target domain index.
            strength: Transfer strength (0-1).

        Returns:
            True if transfer was successful.
        """
        if not self.core:
            return False

        result = self.core._lib.det_core_transfer_pattern(
            self.core._core,
            source_domain,
            target_domain,
            strength
        )

        if result:
            self._log_event(HarnessEventType.STEP, action="transfer_pattern",
                          source=source_domain, target=target_domain, strength=strength)

        return result

    def get_learning_capacity(self) -> float:
        """
        Get current learning capacity (available agency for recruitment).

        Returns:
            Learning capacity value.
        """
        if not self.core:
            return 0.0

        return self.core._lib.det_core_learning_capacity(self.core._core)

    def can_learn(self, complexity: float, domain: int = 0) -> bool:
        """
        Check if learning/recruitment is possible.

        Args:
            complexity: Task complexity.
            domain: Target domain.

        Returns:
            True if learning is possible.
        """
        if not self.core:
            return False

        return self.core._lib.det_core_can_learn(self.core._core, complexity, domain)

    def get_total_grace_needed(self) -> float:
        """Get total grace needed across all nodes."""
        if not self.core:
            return 0.0

        return self.core._lib.det_core_total_grace_needed(self.core._core)

    def evaluate_request(self, tokens: List[int], domain: int = 0, retry_count: int = 0) -> str:
        """
        Evaluate a request through the gatekeeper.

        Args:
            tokens: Token IDs for the request.
            domain: Target domain.
            retry_count: Number of retries.

        Returns:
            Decision string: "PROCEED", "RETRY", "STOP", or "ESCALATE".
        """
        if not self.core:
            return "STOP"

        decision = self.core.evaluate_request(tokens, domain, retry_count)
        return decision.name

    # =========================================================================
    # Time Control
    # =========================================================================

    @property
    def paused(self) -> bool:
        """Check if simulation is paused."""
        return self._paused

    @property
    def speed(self) -> float:
        """Get current simulation speed multiplier."""
        return self._speed

    def pause(self):
        """Pause the simulation."""
        if not self._paused:
            self._paused = True
            self._log_event(HarnessEventType.PAUSE)

    def resume(self):
        """Resume the simulation."""
        if self._paused:
            self._paused = False
            self._log_event(HarnessEventType.RESUME)

    def toggle_pause(self):
        """Toggle pause state."""
        if self._paused:
            self.resume()
        else:
            self.pause()

    def set_speed(self, speed: float):
        """
        Set simulation speed multiplier.

        Args:
            speed: Speed multiplier (e.g., 0.5 for half speed, 2.0 for double).
        """
        speed = max(0.01, min(100.0, speed))
        old_speed = self._speed
        self._speed = speed
        self._log_event(HarnessEventType.SPEED_CHANGE, old_speed=old_speed, new_speed=speed)

    def step(self, dt: float = 0.1) -> bool:
        """
        Execute a single simulation step.

        Args:
            dt: Time delta for the step.

        Returns:
            True if step was executed.
        """
        if not self.core:
            return False

        self.core.step(dt)
        self._log_event(HarnessEventType.STEP, dt=dt)
        return True

    def step_n(self, n: int, dt: float = 0.1) -> int:
        """
        Execute N simulation steps.

        Args:
            n: Number of steps.
            dt: Time delta per step.

        Returns:
            Number of steps actually executed.
        """
        count = 0
        for _ in range(n):
            if self.step(dt):
                count += 1
        return count

    def run_until(self, condition: Callable[[], bool], max_steps: int = 1000, dt: float = 0.1) -> int:
        """
        Run simulation until a condition is met.

        Args:
            condition: Callable that returns True when condition is met.
            max_steps: Maximum steps to run.
            dt: Time delta per step.

        Returns:
            Number of steps executed.
        """
        count = 0
        while count < max_steps and not condition():
            self.step(dt)
            count += 1
        return count

    def start_auto_run(self, dt: float = 0.1, interval: float = 0.1):
        """
        Start automatic simulation running in background thread.

        Args:
            dt: Time delta per step.
            interval: Real-time interval between steps.
        """
        if self._auto_run_thread and self._auto_run_thread.is_alive():
            return

        self._auto_run_stop.clear()

        def run_loop():
            while not self._auto_run_stop.is_set():
                if not self._paused:
                    self.step(dt * self._speed)
                adjusted_interval = interval / self._speed if self._speed > 0 else interval
                time.sleep(adjusted_interval)

        self._auto_run_thread = threading.Thread(target=run_loop, daemon=True)
        self._auto_run_thread.start()

    def stop_auto_run(self):
        """Stop automatic simulation running."""
        self._auto_run_stop.set()
        if self._auto_run_thread:
            self._auto_run_thread.join(timeout=1.0)
            self._auto_run_thread = None

    # =========================================================================
    # State Inspection
    # =========================================================================

    def get_node_state(self, node_index: int) -> Optional[Dict[str, float]]:
        """
        Get full state of a specific node.

        Args:
            node_index: Index of the node.

        Returns:
            Dictionary with node state, or None if invalid.
        """
        if not self.core:
            return None

        if node_index < 0 or node_index >= self.core.num_active:
            return None

        node = self.core._core.contents.nodes[node_index]
        return {
            "index": node_index,
            "layer": "P" if node_index < 16 else "A",
            "F": node.F,
            "q": node.q,
            "a": node.a,
            "theta": node.theta,
            "sigma": node.sigma,
            "P": node.P,
            "L": node.L,
            "dtheta_dt": node.dtheta_dt,
            "grace_buffer": node.grace_buffer,
            "active": node.active,
        }

    def get_bond_state(self, node_i: int, node_j: int) -> Optional[Dict[str, Any]]:
        """
        Get state of a specific bond.

        Args:
            node_i: First node index.
            node_j: Second node index.

        Returns:
            Dictionary with bond state, or None if bond doesn't exist.
        """
        if not self.core:
            return None

        # Find the bond
        bond_idx = self.core._lib.det_core_find_bond(self.core._core, node_i, node_j)
        if bond_idx < 0:
            return None

        bond = self.core._core.contents.bonds[bond_idx]
        return {
            "node_i": bond.i,
            "node_j": bond.j,
            "coherence": bond.C,
            "pi": bond.pi,
            "sigma": bond.sigma,
            "is_temporary": bond.is_temporary,
            "is_cross_layer": bond.is_cross_layer,
        }

    def get_aggregates(self) -> Dict[str, float]:
        """Get aggregate DET state."""
        if not self.core:
            return {}

        p, c, f, q = self.core.get_aggregates()
        return {
            "presence": p,
            "coherence": c,
            "resource": f,
            "debt": q,
        }

    def get_affect(self) -> Dict[str, float]:
        """Get self-cluster affect state."""
        if not self.core:
            return {}

        v, a, b = self.core.get_self_affect()
        return {
            "valence": v,
            "arousal": a,
            "bondedness": b,
        }

    def get_self_cluster(self) -> List[int]:
        """Get indices of nodes in the self-cluster."""
        if not self.core:
            return []

        # Trigger self-identification
        self.core.identify_self()

        # Get self-cluster from the DET core struct
        self_struct = self.core._core.contents.self
        num_nodes = self_struct.num_nodes
        nodes = []
        for i in range(num_nodes):
            nodes.append(self_struct.nodes[i])
        return nodes

    def get_emotional_state(self) -> str:
        """Get current emotional state name."""
        if not self.core:
            return "unknown"

        return self.core.get_emotion().name

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current DET state."""
        if not self.core:
            return {}

        return {
            "tick": self.core.tick,
            "num_active": self.core.num_active,
            "num_bonds": self.core.num_bonds,
            "aggregates": self.get_aggregates(),
            "affect": self.get_affect(),
            "emotional_state": self.get_emotional_state(),
            "self_cluster_size": len(self.get_self_cluster()),
            "harness": {
                "paused": self._paused,
                "speed": self._speed,
                "event_count": len(self._events),
                "snapshot_count": len(self._snapshots),
            }
        }

    # =========================================================================
    # Snapshots
    # =========================================================================

    def take_snapshot(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Take a snapshot of current DET state.

        Args:
            name: Name for the snapshot.
            metadata: Optional metadata to attach.

        Returns:
            True if snapshot was taken.
        """
        if not self.core:
            return False

        state_data = self.core.save_state()

        snapshot = Snapshot(
            name=name,
            timestamp=time.time(),
            tick=self.core.tick,
            state_data=state_data,
            metadata=metadata or {},
        )

        self._snapshots[name] = snapshot
        self._log_event(HarnessEventType.SNAPSHOT, name=name)

        return True

    def restore_snapshot(self, name: str) -> bool:
        """
        Restore DET state from a snapshot.

        Args:
            name: Name of the snapshot to restore.

        Returns:
            True if restored successfully.
        """
        if not self.core:
            return False

        if name not in self._snapshots:
            return False

        snapshot = self._snapshots[name]
        self.core.load_state(snapshot.state_data)
        self._log_event(HarnessEventType.RESTORE, name=name)

        return True

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots."""
        return [
            {
                "name": s.name,
                "timestamp": s.timestamp,
                "tick": s.tick,
                "metadata": s.metadata,
            }
            for s in self._snapshots.values()
        ]

    def delete_snapshot(self, name: str) -> bool:
        """Delete a snapshot."""
        if name in self._snapshots:
            del self._snapshots[name]
            return True
        return False

    # =========================================================================
    # Event Log
    # =========================================================================

    def get_events(self, limit: Optional[int] = None, event_type: Optional[HarnessEventType] = None) -> List[Dict[str, Any]]:
        """
        Get logged events.

        Args:
            limit: Maximum number of events to return (most recent).
            event_type: Filter by event type.

        Returns:
            List of event dictionaries.
        """
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if limit:
            events = events[-limit:]

        return [e.to_dict() for e in events]

    def clear_events(self):
        """Clear the event log."""
        self._events.clear()

    def save_events(self, path: Path) -> bool:
        """Save events to a JSON file."""
        try:
            with open(path, "w") as f:
                json.dump([e.to_dict() for e in self._events], f, indent=2)
            return True
        except Exception:
            return False

    def load_events(self, path: Path) -> bool:
        """Load events from a JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._events = [HarnessEvent.from_dict(e) for e in data]
            return True
        except Exception:
            return False

    # =========================================================================
    # Watchers
    # =========================================================================

    def add_watcher(self, name: str, condition: Callable[[], bool], callback: Callable[[], None]):
        """
        Add a watcher that triggers when a condition is met.

        Args:
            name: Name for the watcher.
            condition: Callable that returns True when condition is met.
            callback: Callable to execute when condition is True.
        """
        self._watchers[name] = (condition, callback)

    def remove_watcher(self, name: str):
        """Remove a watcher."""
        if name in self._watchers:
            del self._watchers[name]

    def check_watchers(self):
        """Check all watchers and trigger callbacks if conditions are met."""
        triggered = []
        for name, (condition, callback) in self._watchers.items():
            try:
                if condition():
                    callback()
                    triggered.append(name)
            except Exception:
                pass
        return triggered


class HarnessCLI(cmd.Cmd):
    """
    Interactive CLI for the DET test harness.
    """

    intro = """
╔═══════════════════════════════════════════════════════════════╗
║           DET Test Harness - Interactive Debug CLI            ║
╠═══════════════════════════════════════════════════════════════╣
║  Type 'help' for available commands.                          ║
║  Type 'status' to see current DET state.                      ║
║  Type 'quit' to exit.                                         ║
╚═══════════════════════════════════════════════════════════════╝
"""
    prompt = "(harness) "

    def __init__(self, controller: HarnessController):
        super().__init__()
        self.controller = controller

    def _print_dict(self, d: Dict[str, Any], indent: int = 0):
        """Pretty print a dictionary."""
        prefix = "  " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                self._print_dict(v, indent + 1)
            elif isinstance(v, float):
                print(f"{prefix}{k}: {v:.4f}")
            else:
                print(f"{prefix}{k}: {v}")

    # -------------------------------------------------------------------------
    # Status Commands
    # -------------------------------------------------------------------------

    def do_status(self, arg):
        """Show current DET state summary."""
        summary = self.controller.get_summary()
        print("\n=== DET State Summary ===")
        self._print_dict(summary)
        print()

    def do_node(self, arg):
        """Show state of a specific node. Usage: node <index>"""
        try:
            index = int(arg)
            state = self.controller.get_node_state(index)
            if state:
                print(f"\n=== Node {index} State ===")
                self._print_dict(state)
            else:
                print(f"Invalid node index: {index}")
        except ValueError:
            print("Usage: node <index>")

    def do_bond(self, arg):
        """Show state of a specific bond. Usage: bond <i> <j>"""
        try:
            parts = arg.split()
            i, j = int(parts[0]), int(parts[1])
            state = self.controller.get_bond_state(i, j)
            if state:
                print(f"\n=== Bond ({i}, {j}) State ===")
                self._print_dict(state)
            else:
                print(f"No bond between {i} and {j}")
        except (ValueError, IndexError):
            print("Usage: bond <i> <j>")

    def do_self(self, arg):
        """Show self-cluster nodes."""
        cluster = self.controller.get_self_cluster()
        print(f"\nSelf-cluster: {cluster}")
        print(f"Size: {len(cluster)} nodes")

    def do_affect(self, arg):
        """Show current affect state (V/A/B)."""
        affect = self.controller.get_affect()
        print("\n=== Affect State ===")
        self._print_dict(affect)

    def do_emotional(self, arg):
        """Show current emotional state."""
        state = self.controller.get_emotional_state()
        print(f"\nEmotional state: {state}")

    # -------------------------------------------------------------------------
    # Injection Commands
    # -------------------------------------------------------------------------

    def do_inject_f(self, arg):
        """Inject resource F. Usage: inject_f <node> <amount>"""
        try:
            parts = arg.split()
            node = int(parts[0])
            amount = float(parts[1])
            if self.controller.inject_f(node, amount):
                print(f"Injected {amount} F into node {node}")
            else:
                print("Injection failed")
        except (ValueError, IndexError):
            print("Usage: inject_f <node> <amount>")

    def do_inject_q(self, arg):
        """Inject structural debt q. Usage: inject_q <node> <amount>"""
        try:
            parts = arg.split()
            node = int(parts[0])
            amount = float(parts[1])
            if self.controller.inject_q(node, amount):
                print(f"Injected {amount} q into node {node}")
            else:
                print("Injection failed")
        except (ValueError, IndexError):
            print("Usage: inject_q <node> <amount>")

    def do_inject_all(self, arg):
        """Inject F into all nodes. Usage: inject_all <amount> [P|A]"""
        try:
            parts = arg.split()
            amount = float(parts[0])
            layer = parts[1].upper() if len(parts) > 1 else None
            count = self.controller.inject_f_all(amount, layer)
            print(f"Injected {amount} F into {count} nodes")
        except (ValueError, IndexError):
            print("Usage: inject_all <amount> [P|A]")

    def do_set_agency(self, arg):
        """Set agency for a node. Usage: set_agency <node> <value>"""
        try:
            parts = arg.split()
            node = int(parts[0])
            value = float(parts[1])
            if self.controller.set_agency(node, value):
                print(f"Set agency of node {node} to {value}")
            else:
                print("Failed to set agency")
        except (ValueError, IndexError):
            print("Usage: set_agency <node> <value>")

    # -------------------------------------------------------------------------
    # Bond Commands
    # -------------------------------------------------------------------------

    def do_create_bond(self, arg):
        """Create a bond. Usage: create_bond <i> <j> [coherence]"""
        try:
            parts = arg.split()
            i, j = int(parts[0]), int(parts[1])
            c = float(parts[2]) if len(parts) > 2 else 0.5
            if self.controller.create_bond(i, j, c):
                print(f"Created bond ({i}, {j}) with C={c}")
            else:
                print("Failed to create bond")
        except (ValueError, IndexError):
            print("Usage: create_bond <i> <j> [coherence]")

    def do_destroy_bond(self, arg):
        """Destroy a bond. Usage: destroy_bond <i> <j>"""
        try:
            parts = arg.split()
            i, j = int(parts[0]), int(parts[1])
            if self.controller.destroy_bond(i, j):
                print(f"Destroyed bond ({i}, {j})")
            else:
                print("Bond not found or destruction failed")
        except (ValueError, IndexError):
            print("Usage: destroy_bond <i> <j>")

    def do_set_coherence(self, arg):
        """Set bond coherence. Usage: set_coherence <i> <j> <value>"""
        try:
            parts = arg.split()
            i, j = int(parts[0]), int(parts[1])
            c = float(parts[2])
            if self.controller.set_coherence(i, j, c):
                print(f"Set coherence of bond ({i}, {j}) to {c}")
            else:
                print("Failed to set coherence")
        except (ValueError, IndexError):
            print("Usage: set_coherence <i> <j> <value>")

    # -------------------------------------------------------------------------
    # Advanced Probing Commands (Phase 6.3)
    # -------------------------------------------------------------------------

    def do_escalate(self, arg):
        """Trigger escalation on a node. Usage: escalate <node>"""
        try:
            node = int(arg)
            if self.controller.trigger_escalation(node):
                print(f"Triggered escalation on node {node}")
            else:
                print("Escalation failed")
        except ValueError:
            print("Usage: escalate <node>")

    def do_grace(self, arg):
        """Inject grace into a node. Usage: grace <node> <amount>"""
        try:
            parts = arg.split()
            node = int(parts[0])
            amount = float(parts[1])
            if self.controller.inject_grace(node, amount):
                print(f"Injected {amount} grace into node {node}")
            else:
                print("Grace injection failed")
        except (ValueError, IndexError):
            print("Usage: grace <node> <amount>")

    def do_grace_all(self, arg):
        """Inject grace into all nodes that need it. Usage: grace_all <amount>"""
        try:
            amount = float(arg) if arg else 0.5
            count = self.controller.inject_grace_all(amount)
            print(f"Injected grace into {count} nodes")
        except ValueError:
            print("Usage: grace_all <amount>")

    def do_grace_needed(self, arg):
        """Show total grace needed across all nodes."""
        total = self.controller.get_total_grace_needed()
        print(f"Total grace needed: {total:.4f}")

    def do_learning(self, arg):
        """Show learning capacity and check if learning is possible."""
        capacity = self.controller.get_learning_capacity()
        print(f"Learning capacity: {capacity:.4f}")

        if arg:
            try:
                complexity = float(arg)
                can_learn = self.controller.can_learn(complexity)
                print(f"Can learn (complexity={complexity}): {can_learn}")
            except ValueError:
                print("Usage: learning [complexity]")

    def do_activate_domain(self, arg):
        """Activate a new domain. Usage: activate_domain <name> <num_nodes> [coherence]"""
        try:
            parts = arg.split()
            name = parts[0]
            num_nodes = int(parts[1])
            coherence = float(parts[2]) if len(parts) > 2 else 0.3
            if self.controller.activate_domain(name, num_nodes, coherence):
                print(f"Activated domain '{name}' with {num_nodes} nodes")
            else:
                print("Domain activation failed")
        except (ValueError, IndexError):
            print("Usage: activate_domain <name> <num_nodes> [coherence]")

    def do_transfer(self, arg):
        """Transfer pattern between domains. Usage: transfer <source> <target> [strength]"""
        try:
            parts = arg.split()
            source = int(parts[0])
            target = int(parts[1])
            strength = float(parts[2]) if len(parts) > 2 else 0.5
            if self.controller.transfer_pattern(source, target, strength):
                print(f"Transferred pattern from domain {source} to {target}")
            else:
                print("Pattern transfer failed")
        except (ValueError, IndexError):
            print("Usage: transfer <source> <target> [strength]")

    def do_gatekeeper(self, arg):
        """Evaluate request through gatekeeper. Usage: gatekeeper <token_ids...>"""
        try:
            tokens = [int(t) for t in arg.split()]
            if not tokens:
                print("Usage: gatekeeper <token_ids...>")
                return
            decision = self.controller.evaluate_request(tokens)
            print(f"Gatekeeper decision: {decision}")
        except ValueError:
            print("Usage: gatekeeper <token_ids...>")

    # -------------------------------------------------------------------------
    # Time Control Commands
    # -------------------------------------------------------------------------

    def do_step(self, arg):
        """Execute simulation steps. Usage: step [n] [dt]"""
        parts = arg.split() if arg else []
        n = int(parts[0]) if len(parts) > 0 else 1
        dt = float(parts[1]) if len(parts) > 1 else 0.1

        count = self.controller.step_n(n, dt)
        print(f"Executed {count} step(s) with dt={dt}")

    def do_pause(self, arg):
        """Pause the simulation."""
        self.controller.pause()
        print("Simulation paused")

    def do_resume(self, arg):
        """Resume the simulation."""
        self.controller.resume()
        print("Simulation resumed")

    def do_speed(self, arg):
        """Set simulation speed. Usage: speed <multiplier>"""
        if not arg:
            print(f"Current speed: {self.controller.speed}x")
            return

        try:
            speed = float(arg)
            self.controller.set_speed(speed)
            print(f"Speed set to {speed}x")
        except ValueError:
            print("Usage: speed <multiplier>")

    def do_run(self, arg):
        """Start auto-running simulation. Usage: run [dt] [interval]"""
        parts = arg.split() if arg else []
        dt = float(parts[0]) if len(parts) > 0 else 0.1
        interval = float(parts[1]) if len(parts) > 1 else 0.1

        self.controller.start_auto_run(dt, interval)
        print(f"Auto-run started (dt={dt}, interval={interval})")

    def do_stop(self, arg):
        """Stop auto-running simulation."""
        self.controller.stop_auto_run()
        print("Auto-run stopped")

    # -------------------------------------------------------------------------
    # Snapshot Commands
    # -------------------------------------------------------------------------

    def do_snapshot(self, arg):
        """Take a snapshot. Usage: snapshot <name>"""
        if not arg:
            print("Usage: snapshot <name>")
            return

        if self.controller.take_snapshot(arg):
            print(f"Snapshot '{arg}' taken")
        else:
            print("Failed to take snapshot")

    def do_restore(self, arg):
        """Restore a snapshot. Usage: restore <name>"""
        if not arg:
            print("Usage: restore <name>")
            return

        if self.controller.restore_snapshot(arg):
            print(f"Restored snapshot '{arg}'")
        else:
            print(f"Snapshot '{arg}' not found")

    def do_snapshots(self, arg):
        """List all snapshots."""
        snapshots = self.controller.list_snapshots()
        if not snapshots:
            print("No snapshots")
            return

        print("\n=== Snapshots ===")
        for s in snapshots:
            ts = datetime.fromtimestamp(s["timestamp"]).strftime("%H:%M:%S")
            print(f"  {s['name']}: tick={s['tick']} at {ts}")

    def do_delete_snapshot(self, arg):
        """Delete a snapshot. Usage: delete_snapshot <name>"""
        if not arg:
            print("Usage: delete_snapshot <name>")
            return

        if self.controller.delete_snapshot(arg):
            print(f"Deleted snapshot '{arg}'")
        else:
            print(f"Snapshot '{arg}' not found")

    # -------------------------------------------------------------------------
    # Event Log Commands
    # -------------------------------------------------------------------------

    def do_events(self, arg):
        """Show recent events. Usage: events [limit]"""
        limit = int(arg) if arg else 10
        events = self.controller.get_events(limit=limit)

        if not events:
            print("No events")
            return

        print(f"\n=== Last {len(events)} Events ===")
        for e in events:
            ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M:%S")
            print(f"  [{ts}] tick={e['tick']} {e['type']}: {e['params']}")

    def do_clear_events(self, arg):
        """Clear the event log."""
        self.controller.clear_events()
        print("Event log cleared")

    # -------------------------------------------------------------------------
    # Utility Commands
    # -------------------------------------------------------------------------

    def do_quit(self, arg):
        """Exit the harness CLI."""
        self.controller.stop_auto_run()
        print("Goodbye!")
        return True

    def do_exit(self, arg):
        """Exit the harness CLI."""
        return self.do_quit(arg)

    def do_help(self, arg):
        """Show help for commands."""
        if arg:
            super().do_help(arg)
        else:
            print("""
=== DET Test Harness Commands ===

STATUS:
  status          - Show DET state summary
  node <i>        - Show node state
  bond <i> <j>    - Show bond state
  self            - Show self-cluster nodes
  affect          - Show V/A/B affect
  emotional       - Show emotional state

INJECTION:
  inject_f <n> <amt>      - Inject F into node
  inject_q <n> <amt>      - Inject q into node
  inject_all <amt> [P|A]  - Inject F into all nodes
  set_agency <n> <val>    - Set node agency (caution!)

BONDS:
  create_bond <i> <j> [c] - Create bond with coherence c
  destroy_bond <i> <j>    - Destroy bond
  set_coherence <i> <j> <c> - Set bond coherence

TIME:
  step [n] [dt]   - Execute n steps
  pause           - Pause simulation
  resume          - Resume simulation
  speed [mult]    - Set/show speed multiplier
  run [dt] [int]  - Start auto-run
  stop            - Stop auto-run

ADVANCED PROBING (Phase 6.3):
  escalate <node>           - Trigger escalation on node
  grace <node> <amount>     - Inject grace into node
  grace_all <amount>        - Inject grace into all needing nodes
  grace_needed              - Show total grace needed
  learning [complexity]     - Show learning capacity (optionally check)
  activate_domain <name> <n> [c] - Activate domain with n nodes
  transfer <src> <tgt> [s]  - Transfer pattern between domains
  gatekeeper <tokens...>    - Evaluate request through gatekeeper

SNAPSHOTS:
  snapshot <name> - Take snapshot
  restore <name>  - Restore snapshot
  snapshots       - List snapshots
  delete_snapshot <name> - Delete snapshot

EVENTS:
  events [limit]  - Show recent events
  clear_events    - Clear event log

OTHER:
  help [cmd]      - Show help
  quit/exit       - Exit harness
""")


def create_harness(
    core=None,
    storage_path: Optional[Path] = None,
    start_paused: bool = False
) -> HarnessController:
    """
    Create a harness controller.

    Args:
        core: DETCore instance to control.
        storage_path: Path for storage.
        start_paused: Whether to start in paused state.

    Returns:
        Configured HarnessController.
    """
    return HarnessController(core=core, storage_path=storage_path, start_paused=start_paused)


def run_harness_cli(core=None, storage_path: Optional[Path] = None):
    """
    Run the interactive harness CLI.

    Args:
        core: DETCore instance to control.
        storage_path: Path for storage.
    """
    controller = create_harness(core=core, storage_path=storage_path)
    cli = HarnessCLI(controller)
    cli.cmdloop()
