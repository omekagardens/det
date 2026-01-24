"""
DET-OS Gatekeeper - Agency-Based Security
==========================================

Security in DET-OS is based on agency constraints, not access control lists.
A creature's ability to perform actions depends on:
    - Its agency (a): intrinsic capability
    - Its coherence (C): relationship quality
    - Its resource (F): ability to pay cost
    - Target's agency: willingness to be acted upon

Key principles:
    - No action without agency cost
    - Grace gates access, not denial
    - Coherence enables trust
    - Conservation prevents privilege escalation

Permission Model:
    - Capabilities are agency-weighted tokens
    - Actions require meeting agency thresholds
    - Coherent creatures can share capabilities
    - Grace injection can temporarily elevate access
"""

from enum import Enum, Flag, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Callable
import time


class PermissionLevel(Enum):
    """Permission levels (agency thresholds)."""
    NONE = 0       # No access
    READ = 1       # Read-only (a >= 0.1)
    WRITE = 2      # Write access (a >= 0.3)
    EXECUTE = 3    # Execute code (a >= 0.5)
    ADMIN = 4      # System access (a >= 0.8)
    ROOT = 5       # Full access (a >= 0.95)


class AccessResult(Enum):
    """Result of access check."""
    GRANTED = auto()
    DENIED_NO_AGENCY = auto()      # Insufficient agency
    DENIED_NO_COHERENCE = auto()   # Not bonded/trusted
    DENIED_NO_RESOURCE = auto()    # Can't pay cost
    DENIED_NO_CAPABILITY = auto()  # Missing capability
    DENIED_TARGET_REFUSES = auto() # Target's agency blocks
    DENIED_POLICY = auto()         # Policy violation


@dataclass
class Permission:
    """A permission requirement."""
    action: str              # Action name ("file_read", "network", etc.)
    target: str              # Target resource
    level: PermissionLevel   # Required level
    min_agency: float = 0.0  # Minimum agency required
    min_coherence: float = 0.0  # Minimum coherence required
    F_cost: float = 0.0      # Resource cost to exercise

    def agency_threshold(self) -> float:
        """Get agency threshold for this permission level."""
        thresholds = {
            PermissionLevel.NONE: 0.0,
            PermissionLevel.READ: 0.1,
            PermissionLevel.WRITE: 0.3,
            PermissionLevel.EXECUTE: 0.5,
            PermissionLevel.ADMIN: 0.8,
            PermissionLevel.ROOT: 0.95,
        }
        return max(self.min_agency, thresholds.get(self.level, 0.0))


@dataclass
class Capability:
    """
    A capability token granting specific permissions.

    Capabilities are agency-weighted: they only work if the holder
    has sufficient agency to exercise them.
    """
    cap_id: int
    name: str
    permissions: List[Permission]
    owner: int               # Creature ID that owns this capability
    min_agency_to_use: float = 0.0
    transferable: bool = False
    expires_at: Optional[float] = None

    # Delegation chain
    delegated_from: Optional[int] = None
    delegation_depth: int = 0
    max_delegation_depth: int = 3

    def is_expired(self) -> bool:
        """Check if capability has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def can_delegate(self) -> bool:
        """Check if this capability can be delegated."""
        return self.transferable and self.delegation_depth < self.max_delegation_depth


class Gatekeeper:
    """
    DET-OS security gatekeeper.

    Evaluates access requests based on agency dynamics, not static ACLs.
    A creature must have sufficient agency, coherence, and resource to
    perform an action.
    """

    def __init__(self):
        self.capabilities: Dict[int, Capability] = {}
        self.next_cap_id = 0

        # Creature -> capabilities owned
        self.creature_caps: Dict[int, Set[int]] = {}

        # Policy callbacks
        self.policies: List[Callable] = []

        # Audit log
        self.audit_log: List[Dict] = []
        self.max_audit_entries = 10000

    def create_capability(self,
                          name: str,
                          owner: int,
                          permissions: List[Permission],
                          min_agency: float = 0.0,
                          transferable: bool = False,
                          ttl: Optional[float] = None) -> Capability:
        """Create a new capability."""
        cap_id = self.next_cap_id
        self.next_cap_id += 1

        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        cap = Capability(
            cap_id=cap_id,
            name=name,
            permissions=permissions,
            owner=owner,
            min_agency_to_use=min_agency,
            transferable=transferable,
            expires_at=expires_at
        )

        self.capabilities[cap_id] = cap

        if owner not in self.creature_caps:
            self.creature_caps[owner] = set()
        self.creature_caps[owner].add(cap_id)

        return cap

    def revoke_capability(self, cap_id: int):
        """Revoke a capability."""
        cap = self.capabilities.get(cap_id)
        if not cap:
            return

        if cap.owner in self.creature_caps:
            self.creature_caps[cap.owner].discard(cap_id)

        del self.capabilities[cap_id]

    def delegate_capability(self,
                            cap_id: int,
                            from_creature: int,
                            to_creature: int) -> Optional[Capability]:
        """Delegate a capability to another creature."""
        cap = self.capabilities.get(cap_id)
        if not cap:
            return None

        if cap.owner != from_creature:
            return None

        if not cap.can_delegate():
            return None

        # Create delegated copy
        new_cap = Capability(
            cap_id=self.next_cap_id,
            name=f"{cap.name}:delegated",
            permissions=cap.permissions.copy(),
            owner=to_creature,
            min_agency_to_use=cap.min_agency_to_use,
            transferable=cap.transferable,
            expires_at=cap.expires_at,
            delegated_from=from_creature,
            delegation_depth=cap.delegation_depth + 1,
            max_delegation_depth=cap.max_delegation_depth
        )
        self.next_cap_id += 1

        self.capabilities[new_cap.cap_id] = new_cap

        if to_creature not in self.creature_caps:
            self.creature_caps[to_creature] = set()
        self.creature_caps[to_creature].add(new_cap.cap_id)

        return new_cap

    def check_access(self,
                     creature_id: int,
                     creature_agency: float,
                     creature_coherence: float,
                     creature_F: float,
                     action: str,
                     target: str) -> tuple:
        """
        Check if a creature can perform an action on a target.

        Args:
            creature_id: Requesting creature
            creature_agency: Creature's agency (a)
            creature_coherence: Creature's coherence (C_self)
            creature_F: Creature's resource
            action: Action to perform
            target: Target resource

        Returns:
            (AccessResult, reason_string)
        """
        # Find applicable capabilities
        caps = self._find_capabilities(creature_id, action, target)

        if not caps:
            self._audit("DENIED", creature_id, action, target, "No capability")
            return AccessResult.DENIED_NO_CAPABILITY, "No capability for this action"

        # Find the best capability (lowest requirements)
        best_cap = None
        best_perm = None

        for cap in caps:
            if cap.is_expired():
                continue

            for perm in cap.permissions:
                if perm.action != action:
                    continue
                if perm.target != target and perm.target != "*":
                    continue

                # Check agency threshold
                threshold = max(perm.agency_threshold(), cap.min_agency_to_use)
                if creature_agency >= threshold:
                    if best_perm is None or threshold < best_perm.agency_threshold():
                        best_cap = cap
                        best_perm = perm

        if not best_perm:
            self._audit("DENIED", creature_id, action, target, "Insufficient agency")
            return AccessResult.DENIED_NO_AGENCY, "Insufficient agency for any capability"

        # Check coherence requirement
        if creature_coherence < best_perm.min_coherence:
            self._audit("DENIED", creature_id, action, target, "Insufficient coherence")
            return AccessResult.DENIED_NO_COHERENCE, \
                f"Coherence {creature_coherence:.2f} < {best_perm.min_coherence:.2f}"

        # Check resource cost
        if creature_F < best_perm.F_cost:
            self._audit("DENIED", creature_id, action, target, "Insufficient resource")
            return AccessResult.DENIED_NO_RESOURCE, \
                f"Resource {creature_F:.2f} < {best_perm.F_cost:.2f}"

        # Run custom policies
        for policy in self.policies:
            result = policy(creature_id, action, target, best_perm)
            if result != AccessResult.GRANTED:
                self._audit("DENIED", creature_id, action, target, f"Policy: {result}")
                return result, f"Policy denied: {result.name}"

        self._audit("GRANTED", creature_id, action, target, f"Via {best_cap.name}")
        return AccessResult.GRANTED, f"Granted via capability '{best_cap.name}'"

    def _find_capabilities(self, creature_id: int, action: str, target: str) -> List[Capability]:
        """Find capabilities that might grant this access."""
        if creature_id not in self.creature_caps:
            return []

        result = []
        for cap_id in self.creature_caps[creature_id]:
            cap = self.capabilities.get(cap_id)
            if not cap:
                continue

            for perm in cap.permissions:
                if perm.action == action or perm.action == "*":
                    if perm.target == target or perm.target == "*":
                        result.append(cap)
                        break

        return result

    def _audit(self, result: str, creature: int, action: str, target: str, reason: str):
        """Record audit entry."""
        entry = {
            "time": time.time(),
            "result": result,
            "creature": creature,
            "action": action,
            "target": target,
            "reason": reason
        }

        self.audit_log.append(entry)

        # Trim old entries
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries:]

    def add_policy(self, policy: Callable):
        """Add a custom policy check function."""
        self.policies.append(policy)

    def create_standard_capabilities(self, creature_id: int, level: PermissionLevel):
        """Create standard capabilities for a creature based on level."""
        if level.value >= PermissionLevel.READ.value:
            self.create_capability(
                name="read",
                owner=creature_id,
                permissions=[
                    Permission("read", "*", PermissionLevel.READ)
                ]
            )

        if level.value >= PermissionLevel.WRITE.value:
            self.create_capability(
                name="write",
                owner=creature_id,
                permissions=[
                    Permission("write", "*", PermissionLevel.WRITE)
                ]
            )

        if level.value >= PermissionLevel.EXECUTE.value:
            self.create_capability(
                name="execute",
                owner=creature_id,
                permissions=[
                    Permission("execute", "*", PermissionLevel.EXECUTE),
                    Permission("spawn", "*", PermissionLevel.EXECUTE),
                    Permission("channel", "*", PermissionLevel.EXECUTE),
                ]
            )

        if level.value >= PermissionLevel.ADMIN.value:
            self.create_capability(
                name="admin",
                owner=creature_id,
                permissions=[
                    Permission("admin", "*", PermissionLevel.ADMIN)
                ],
                transferable=True
            )

    def revoke_all_for_creature(self, creature_id: int):
        """Revoke all capabilities for a creature."""
        if creature_id not in self.creature_caps:
            return

        cap_ids = list(self.creature_caps[creature_id])
        for cap_id in cap_ids:
            self.revoke_capability(cap_id)

    def get_audit_log(self, count: int = 100) -> List[Dict]:
        """Get recent audit log entries."""
        return self.audit_log[-count:]


__all__ = [
    'PermissionLevel', 'AccessResult', 'Permission',
    'Capability', 'Gatekeeper'
]
