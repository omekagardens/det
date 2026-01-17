"""
DET State API
=============

API wrapper for exposing DET state to the web interface.
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from ..core import DETCore
from ..harness import HarnessController


@dataclass
class DETStateAPI:
    """
    API for accessing and manipulating DET state.

    Provides a clean interface for the web frontend.
    """

    core: Optional[DETCore] = None
    harness: Optional[HarnessController] = None

    # State cache
    _last_update: float = 0.0
    _cache_duration: float = 0.05  # 50ms cache

    def get_status(self) -> Dict[str, Any]:
        """Get overall status."""
        if not self.core:
            return {"error": "No core available", "connected": False}

        return {
            "connected": True,
            "tick": self.core.tick,
            "num_active": self.core.num_active,
            "num_bonds": self.core.num_bonds,
            "paused": self.harness.paused if self.harness else False,
            "speed": self.harness.speed if self.harness else 1.0,
            "timestamp": time.time(),
        }

    def get_aggregates(self) -> Dict[str, float]:
        """Get aggregate metrics."""
        if not self.harness:
            return {}
        return self.harness.get_aggregates()

    def get_affect(self) -> Dict[str, float]:
        """Get affect state."""
        if not self.harness:
            return {}
        return self.harness.get_affect()

    def get_emotional_state(self) -> str:
        """Get emotional state name."""
        if not self.harness:
            return "unknown"
        return self.harness.get_emotional_state()

    def get_self_cluster(self) -> List[int]:
        """Get self-cluster node indices."""
        if not self.harness:
            return []
        return self.harness.get_self_cluster()

    def get_nodes(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        Get all node states.

        Args:
            include_inactive: Include dormant/inactive nodes.

        Returns:
            List of node state dictionaries.
        """
        if not self.core:
            return []

        nodes = []
        limit = self.core.num_nodes if include_inactive else self.core.num_active

        for i in range(limit):
            node = self.core._core.contents.nodes[i]
            if not include_inactive and not node.active:
                continue

            nodes.append({
                "index": i,
                "layer": self._get_layer_name(i),
                "F": round(node.F, 4),
                "q": round(node.q, 4),
                "a": round(node.a, 4),
                "theta": round(node.theta, 4),
                "sigma": round(node.sigma, 4),
                "P": round(node.P, 4),
                "L": round(node.L, 4),
                "active": node.active,
                "affect": {
                    "v": round(node.affect.v, 4),
                    "r": round(node.affect.r, 4),
                    "b": round(node.affect.b, 4),
                },
            })

        return nodes

    def get_node(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a specific node's state."""
        if not self.harness:
            return None
        return self.harness.get_node_state(index)

    def get_bonds(self, min_coherence: float = 0.01) -> List[Dict[str, Any]]:
        """
        Get all bond states.

        Args:
            min_coherence: Minimum coherence to include.

        Returns:
            List of bond state dictionaries.
        """
        if not self.core:
            return []

        bonds = []
        for i in range(self.core.num_bonds):
            bond = self.core._core.contents.bonds[i]
            if bond.C < min_coherence:
                continue

            bonds.append({
                "i": int(bond.i),
                "j": int(bond.j),
                "C": round(bond.C, 4),
                "pi": round(bond.pi, 4),
                "sigma": round(bond.sigma, 4),
                "is_cross_layer": bool(bond.is_cross_layer),
                "is_temporary": bool(bond.is_temporary),
            })

        return bonds

    def get_bond(self, i: int, j: int) -> Optional[Dict[str, Any]]:
        """Get a specific bond's state."""
        if not self.harness:
            return None
        return self.harness.get_bond_state(i, j)

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for initialization."""
        self_cluster = self.get_self_cluster()

        return {
            "status": self.get_status(),
            "aggregates": self.get_aggregates(),
            "affect": self.get_affect(),
            "emotional_state": self.get_emotional_state(),
            "self_cluster": self_cluster,
            "nodes": self.get_nodes(),
            "bonds": self.get_bonds(),
        }

    def get_visualization_data(self) -> Dict[str, Any]:
        """
        Get data optimized for 3D visualization.

        Returns node positions, colors, and bond connections.
        """
        if not self.core:
            return {"nodes": [], "bonds": [], "self_cluster": []}

        import math

        nodes = []
        self_cluster = set(self.get_self_cluster())

        for i in range(self.core.num_active):
            node = self.core._core.contents.nodes[i]

            # Position based on layer and index
            layer = self._get_layer_name(i)
            if layer == "P":
                # P-layer: inner ring
                angle = (i / 16) * 2 * math.pi
                radius = 2.0
                y = 0.0
            else:
                # A-layer: outer ring with domain clustering
                a_index = i - 16
                domain = node.domain
                angle = (a_index / 256) * 2 * math.pi + (domain * 0.5)
                radius = 4.0 + (domain % 4) * 0.5
                y = (domain // 4 - 1) * 1.0

            x = math.cos(angle) * radius
            z = math.sin(angle) * radius

            # Color based on affect
            v = (node.affect.v + 1) / 2  # Normalize to 0-1
            r_affect = node.affect.r
            b_affect = node.affect.b

            # RGB: valence affects red-green, arousal affects brightness
            brightness = 0.3 + r_affect * 0.7
            red = (1 - v) * brightness
            green = v * brightness
            blue = b_affect * brightness

            nodes.append({
                "id": i,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "size": round(0.1 + node.a * 0.3, 3),  # Size based on agency
                "color": {
                    "r": round(red, 3),
                    "g": round(green, 3),
                    "b": round(blue, 3),
                },
                "layer": layer,
                "in_self": i in self_cluster,
                "P": round(node.P, 3),
                "a": round(node.a, 3),
            })

        bonds = []
        for i in range(self.core.num_bonds):
            bond = self.core._core.contents.bonds[i]
            if bond.C < 0.05:  # Skip very weak bonds
                continue

            bonds.append({
                "source": int(bond.i),
                "target": int(bond.j),
                "strength": round(bond.C, 3),
                "is_cross_layer": bool(bond.is_cross_layer),
            })

        return {
            "nodes": nodes,
            "bonds": bonds,
            "self_cluster": list(self_cluster),
        }

    def _get_layer_name(self, index: int) -> str:
        """Get layer name for a node index."""
        if index < 16:
            return "P"
        elif index < 16 + 256:
            return "A"
        else:
            return "D"  # Dormant

    # Control methods

    def step(self, n: int = 1, dt: float = 0.1) -> Dict[str, Any]:
        """Execute simulation steps."""
        if not self.harness:
            return {"error": "No harness available"}

        count = self.harness.step_n(n, dt)
        return {
            "steps": count,
            "tick": self.core.tick if self.core else 0,
        }

    def pause(self) -> bool:
        """Pause simulation."""
        if self.harness:
            self.harness.pause()
            return True
        return False

    def resume(self) -> bool:
        """Resume simulation."""
        if self.harness:
            self.harness.resume()
            return True
        return False

    def set_speed(self, speed: float) -> float:
        """Set simulation speed."""
        if self.harness:
            self.harness.set_speed(speed)
            return self.harness.speed
        return 1.0

    def inject_f(self, node: int, amount: float) -> bool:
        """Inject resource into node."""
        if self.harness:
            return self.harness.inject_f(node, amount)
        return False

    def inject_q(self, node: int, amount: float) -> bool:
        """Inject debt into node."""
        if self.harness:
            return self.harness.inject_q(node, amount)
        return False

    def take_snapshot(self, name: str) -> bool:
        """Take state snapshot."""
        if self.harness:
            return self.harness.take_snapshot(name)
        return False

    def restore_snapshot(self, name: str) -> bool:
        """Restore state snapshot."""
        if self.harness:
            return self.harness.restore_snapshot(name)
        return False

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List available snapshots."""
        if self.harness:
            return self.harness.list_snapshots()
        return []

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events."""
        if self.harness:
            return self.harness.get_events(limit=limit)
        return []
