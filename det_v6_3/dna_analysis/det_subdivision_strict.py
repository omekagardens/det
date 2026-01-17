"""
DET Subdivision Theory v2: Strict-Core Compliant

This module implements agentic division following Deep Existence Theory (DET)
core principles, using DNA replication as the biological model.

KEY CORRECTIONS FROM v1:
1. NO NODE CREATION - Fixed substrate with dormant/active nodes
2. NO AGENCY COPYING - Mutual consent recruitment instead
3. LOCAL FORK MODEL - Division at specific bonds, not global low C
4. DERIVATIONAL COSTS - Resource/q emerge from existing laws
5. NO GLOBAL OPERATIONS - Everything is bond-local

DNA-GUIDED PRINCIPLES:
- Division is a propagating bond-rewire (zipper model)
- "Child" nodes already exist (dormant), recruitment not creation
- Pattern transfer (topology, phase) without agency copying
- Resource cost for breaking and forming bonds (ATP analog)
- Boundary as catalyst, not agency donor
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import math

# DET Constants
PHI = (1 + np.sqrt(5)) / 2


@dataclass
class DETNode:
    """
    A node in the DET substrate.

    In DET, nodes are not created or destroyed - they exist in the substrate
    and can be dormant (n=0) or participating (n=1).
    """
    id: int

    # Core DET properties (intrinsic, not settable from outside)
    a: float              # Agency - INVIOLABLE, cannot be set externally
    sigma: float = 1.0    # Processing rate / conductivity

    # Dynamic properties (can change through lawful dynamics)
    F: float = 1.0        # Resource
    q: float = 0.0        # Structural debt
    theta: float = 0.0    # Phase

    # Participation state
    n: int = 1            # 0 = dormant, 1 = participating

    # Bond coherences (to neighbor IDs)
    C_bonds: Dict[int, float] = field(default_factory=dict)

    # Records/traces (for pattern transfer, not agency)
    records: Dict[str, float] = field(default_factory=dict)

    @property
    def is_dormant(self) -> bool:
        return self.n == 0

    @property
    def is_active(self) -> bool:
        return self.n == 1

    @property
    def can_participate_in_division(self) -> bool:
        """Check if node meets minimum requirements for division involvement."""
        return self.a >= DET_DIVISION_PARAMS['a_min_division']

    @property
    def can_be_recruited(self) -> bool:
        """Check if dormant node can be recruited into participation."""
        return (self.is_dormant and
                self.a >= DET_DIVISION_PARAMS['a_min_join'] and
                self.F >= DET_DIVISION_PARAMS['F_min_join'])


# DET-compliant division parameters
DET_DIVISION_PARAMS = {
    # Agency thresholds (for gating, NOT for setting)
    'a_min_division': 0.2,    # Parent must have this agency to initiate
    'a_min_join': 0.1,        # Recruit must have this agency to join

    # Resource costs (paid locally)
    'kappa_break': 0.1,       # Cost to break a bond per unit C
    'kappa_form': 0.08,       # Cost to form a new bond per unit C_init

    # Coherence thresholds
    'C_fork_max': 0.3,        # Fork bond must be below this to break
    'C_template_min': 0.5,    # Template bonds must be above this
    'C_init': 0.15,           # Initial coherence for new bonds

    # Resource thresholds
    'F_min_division': 0.5,    # Minimum F to attempt division
    'F_min_join': 0.2,        # Minimum F for recruit

    # Locality
    'fork_radius': 1,         # Division only affects immediate bonds
}


@dataclass
class DETSubstrate:
    """
    The fixed DET substrate containing all nodes (active and dormant).

    In DET, the substrate is fixed. Nodes are not created or destroyed.
    Division changes participation states and bond topology.
    """
    nodes: Dict[int, DETNode] = field(default_factory=dict)
    bonds: Set[Tuple[int, int]] = field(default_factory=set)

    def get_neighbors(self, node_id: int, radius: int = 1) -> List[int]:
        """Get neighbor IDs within radius (locality constraint)."""
        neighbors = []
        for (i, j) in self.bonds:
            if i == node_id:
                neighbors.append(j)
            elif j == node_id:
                neighbors.append(i)
        return neighbors

    def get_dormant_neighbors(self, node_id: int) -> List[int]:
        """Get dormant nodes in local neighborhood that could be recruited."""
        neighbors = self.get_neighbors(node_id)
        return [n for n in neighbors if self.nodes[n].is_dormant]

    def get_active_neighbors(self, node_id: int) -> List[int]:
        """Get active nodes in local neighborhood."""
        neighbors = self.get_neighbors(node_id)
        return [n for n in neighbors if self.nodes[n].is_active]


class ForkState(Enum):
    """State of a division fork at a specific bond."""
    CLOSED = "closed"           # Bond intact, no division
    OPENING = "opening"         # Coherence reducing
    OPEN = "open"               # Bond broken, ready for recruitment
    RECRUITING = "recruiting"   # Finding dormant node
    REBONDING = "rebonding"     # Forming new bonds
    COMPLETE = "complete"       # Fork closed, division done at this site


@dataclass
class DivisionFork:
    """
    A local division fork at a specific bond.

    Division propagates through forks, one bond at a time (zipper model).
    This is entirely local - affects only the bond and its immediate neighbors.
    """
    parent_id: int
    fork_bond_neighbor: int
    state: ForkState = ForkState.CLOSED
    recruited_id: Optional[int] = None

    # Track costs paid (for debugging/analysis)
    F_spent_breaking: float = 0.0
    F_spent_forming: float = 0.0


class StrictCoreDivision:
    """
    DET-compliant division mechanism.

    Key principles:
    1. No node creation - recruit dormant nodes
    2. No agency copying - mutual consent gating only
    3. Local fork model - one bond at a time
    4. Derivational costs - F consumed, q emerges naturally
    5. Pattern transfer - topology/phase, not agency
    """

    def __init__(self, substrate: DETSubstrate):
        self.substrate = substrate
        self.active_forks: List[DivisionFork] = []
        self.division_history: List[Dict] = []

    def check_division_eligibility(self, node_id: int) -> Dict:
        """
        Check if a node can initiate division (local check only).

        Returns dict with eligibility status and reasons.
        """
        node = self.substrate.nodes[node_id]
        result = {
            'eligible': True,
            'reasons': [],
            'fork_candidates': []
        }

        # Check agency gate (NOT setting agency, just checking)
        if node.a < DET_DIVISION_PARAMS['a_min_division']:
            result['eligible'] = False
            result['reasons'].append(
                f"Agency {node.a:.3f} < min {DET_DIVISION_PARAMS['a_min_division']}"
            )

        # Check resource
        if node.F < DET_DIVISION_PARAMS['F_min_division']:
            result['eligible'] = False
            result['reasons'].append(
                f"Resource {node.F:.3f} < min {DET_DIVISION_PARAMS['F_min_division']}"
            )

        # Check for fork-able bonds (locally low C)
        for neighbor_id, C in node.C_bonds.items():
            if C < DET_DIVISION_PARAMS['C_fork_max']:
                result['fork_candidates'].append({
                    'neighbor': neighbor_id,
                    'coherence': C,
                    'breakable': True
                })

        if not result['fork_candidates']:
            result['eligible'] = False
            result['reasons'].append("No bonds with low enough coherence to fork")

        # Check for template bonds (need some high-C bonds to preserve pattern)
        template_bonds = [
            (n, C) for n, C in node.C_bonds.items()
            if C >= DET_DIVISION_PARAMS['C_template_min']
        ]
        if not template_bonds and len(node.C_bonds) > 1:
            result['reasons'].append("Warning: No high-coherence template bonds")

        # Check for recruitable dormant neighbors
        dormant = self.substrate.get_dormant_neighbors(node_id)
        recruitable = [
            d for d in dormant
            if self.substrate.nodes[d].can_be_recruited
        ]
        result['recruitable_dormant'] = recruitable

        if not recruitable:
            result['eligible'] = False
            result['reasons'].append("No recruitable dormant nodes in neighborhood")

        return result

    def initiate_fork(self, parent_id: int, fork_bond_neighbor: int) -> Optional[DivisionFork]:
        """
        Initiate a division fork at a specific bond.

        This is a LOCAL operation affecting only the parent-neighbor bond.
        """
        parent = self.substrate.nodes[parent_id]
        neighbor = self.substrate.nodes[fork_bond_neighbor]

        # Verify bond exists and is fork-able
        if fork_bond_neighbor not in parent.C_bonds:
            return None

        C = parent.C_bonds[fork_bond_neighbor]
        if C >= DET_DIVISION_PARAMS['C_fork_max']:
            return None  # Coherence too high to break

        # Create fork
        fork = DivisionFork(
            parent_id=parent_id,
            fork_bond_neighbor=fork_bond_neighbor,
            state=ForkState.OPENING
        )

        self.active_forks.append(fork)
        return fork

    def process_fork_step(self, fork: DivisionFork) -> bool:
        """
        Process one step of a division fork.

        Each step is LOCAL - only affects the fork bond and immediate neighborhood.

        Returns True if fork is complete, False if still processing.
        """
        parent = self.substrate.nodes[fork.parent_id]

        if fork.state == ForkState.OPENING:
            return self._process_opening(fork, parent)

        elif fork.state == ForkState.OPEN:
            return self._process_recruitment(fork, parent)

        elif fork.state == ForkState.RECRUITING:
            return self._process_rebonding(fork, parent)

        elif fork.state == ForkState.REBONDING:
            return self._finalize_fork(fork, parent)

        return fork.state == ForkState.COMPLETE

    def _process_opening(self, fork: DivisionFork, parent: DETNode) -> bool:
        """Break the fork bond (pay resource cost)."""
        neighbor_id = fork.fork_bond_neighbor
        C = parent.C_bonds.get(neighbor_id, 0)

        # Cost to break bond (like ATP for helicase)
        cost = DET_DIVISION_PARAMS['kappa_break'] * C
        if parent.F < cost:
            # Not enough resource - fork fails
            fork.state = ForkState.CLOSED
            return True

        # Pay the cost
        parent.F -= cost
        fork.F_spent_breaking += cost

        # q-locking happens naturally from F loss (existing DET mechanism)
        # No special q_inheritance needed

        # Break the bond (reduce coherence to near zero)
        parent.C_bonds[neighbor_id] = 0.01
        neighbor = self.substrate.nodes[neighbor_id]
        if fork.parent_id in neighbor.C_bonds:
            neighbor.C_bonds[fork.parent_id] = 0.01

        fork.state = ForkState.OPEN
        return False

    def _process_recruitment(self, fork: DivisionFork, parent: DETNode) -> bool:
        """Find and recruit a dormant node (mutual consent check)."""
        # Get recruitable dormant neighbors
        dormant = self.substrate.get_dormant_neighbors(fork.parent_id)

        for d_id in dormant:
            dormant_node = self.substrate.nodes[d_id]

            # MUTUAL CONSENT CHECK - both must be willing
            # Parent gate: already checked in eligibility
            # Recruit gate: check agency (NOT setting it!)
            if not dormant_node.can_be_recruited:
                continue

            # Recruit has sufficient intrinsic agency - recruitment possible
            fork.recruited_id = d_id
            fork.state = ForkState.RECRUITING
            return False

        # No suitable recruit found
        fork.state = ForkState.CLOSED
        return True

    def _process_rebonding(self, fork: DivisionFork, parent: DETNode) -> bool:
        """Form new bonds with recruited node (pay resource cost)."""
        if fork.recruited_id is None:
            fork.state = ForkState.CLOSED
            return True

        recruit = self.substrate.nodes[fork.recruited_id]

        # Activate the dormant node
        recruit.n = 1  # Now participating

        # Cost to form new bond
        C_init = DET_DIVISION_PARAMS['C_init']
        cost = DET_DIVISION_PARAMS['kappa_form'] * C_init

        if parent.F < cost:
            # Not enough resource - partial failure
            recruit.n = 0  # Return to dormant
            fork.recruited_id = None
            fork.state = ForkState.CLOSED
            return True

        # Pay cost and form bond
        parent.F -= cost
        fork.F_spent_forming += cost

        # Create bond between parent and recruit
        parent.C_bonds[fork.recruited_id] = C_init
        recruit.C_bonds[fork.parent_id] = C_init

        # Add bond to substrate
        self.substrate.bonds.add((min(fork.parent_id, fork.recruited_id),
                                   max(fork.parent_id, fork.recruited_id)))

        # PATTERN TRANSFER (not agency!)
        # Transfer topology hints, phase alignment, records
        self._transfer_pattern(parent, recruit)

        fork.state = ForkState.REBONDING
        return False

    def _transfer_pattern(self, parent: DETNode, recruit: DETNode):
        """
        Transfer pattern information (NOT agency) from parent to recruit.

        What transfers:
        - Phase alignment seed
        - Record traces
        - Bond topology hints (who to try connecting to)

        What does NOT transfer:
        - Agency a (inviolable)
        - Resource F (already has its own)
        - Structural debt q (emerges from own dynamics)
        """
        # Phase alignment (small nudge toward parent's phase)
        phase_seed = 0.1 * (parent.theta - recruit.theta)
        recruit.theta += phase_seed  # Small alignment, not forcing

        # Record traces (like epigenetic marks in DNA)
        # These are hints, not commands
        recruit.records['parent_id'] = parent.id
        recruit.records['pattern_seed'] = parent.records.get('pattern_seed', 0) + 1

        # Note: Agency is NEVER touched
        # recruit.a stays whatever it intrinsically was

    def _finalize_fork(self, fork: DivisionFork, parent: DETNode) -> bool:
        """Complete the fork and record the division."""
        fork.state = ForkState.COMPLETE

        # Record division event
        self.division_history.append({
            'parent_id': fork.parent_id,
            'recruited_id': fork.recruited_id,
            'fork_neighbor': fork.fork_bond_neighbor,
            'F_spent_breaking': fork.F_spent_breaking,
            'F_spent_forming': fork.F_spent_forming,
            'parent_agency': parent.a,
            'recruit_agency': self.substrate.nodes[fork.recruited_id].a if fork.recruited_id else None,
            # Note: agencies are just recorded, not related by copying
        })

        return True

    def run_division_attempt(self, node_id: int) -> Dict:
        """
        Attempt division for a node.

        This runs the full fork cycle for one eligible bond.
        All operations are local.
        """
        # Check eligibility
        eligibility = self.check_division_eligibility(node_id)
        if not eligibility['eligible']:
            return {
                'success': False,
                'reason': eligibility['reasons'],
                'recruited': None
            }

        # Pick first fork-able bond
        fork_candidate = eligibility['fork_candidates'][0]
        fork = self.initiate_fork(node_id, fork_candidate['neighbor'])

        if fork is None:
            return {
                'success': False,
                'reason': ['Failed to initiate fork'],
                'recruited': None
            }

        # Process fork to completion
        max_steps = 10
        for _ in range(max_steps):
            complete = self.process_fork_step(fork)
            if complete:
                break

        if fork.state == ForkState.COMPLETE and fork.recruited_id is not None:
            return {
                'success': True,
                'recruited_id': fork.recruited_id,
                'recruited_agency': self.substrate.nodes[fork.recruited_id].a,
                'F_spent': fork.F_spent_breaking + fork.F_spent_forming
            }
        else:
            return {
                'success': False,
                'reason': [f'Fork ended in state {fork.state}'],
                'recruited': None
            }


def demonstrate_strict_core_division():
    """Demonstrate DET-compliant division."""

    print("=" * 70)
    print("DET STRICT-CORE COMPLIANT DIVISION")
    print("Deep Existence Theory - Agentic Subdivision via Recruitment")
    print("=" * 70)

    # Create substrate with active and dormant nodes
    # NOTE: Nodes are NOT created during division - they pre-exist
    substrate = DETSubstrate()

    # Active parent node (high agency, sufficient resource)
    parent = DETNode(
        id=0,
        a=0.7,      # Intrinsic agency (cannot be changed)
        F=2.0,      # Sufficient resource
        q=0.2,      # Low structural debt
        n=1,        # Active
    )

    # Neighbor node (will have bond broken)
    neighbor = DETNode(
        id=1,
        a=0.5,
        F=1.0,
        n=1,
    )

    # Dormant nodes (potential recruits with their OWN intrinsic agency)
    dormant1 = DETNode(
        id=2,
        a=0.4,      # Has its own agency (NOT copied from parent!)
        F=0.5,
        n=0,        # Dormant
    )

    dormant2 = DETNode(
        id=3,
        a=0.05,     # Too low agency to be recruited
        F=0.3,
        n=0,
    )

    dormant3 = DETNode(
        id=4,
        a=0.3,
        F=0.1,      # Too low resource to join
        n=0,
    )

    # Add nodes to substrate
    for node in [parent, neighbor, dormant1, dormant2, dormant3]:
        substrate.nodes[node.id] = node

    # Create bonds
    # Parent has low-C bond to neighbor (fork-able)
    # and high-C bonds to dormant (template integrity)
    parent.C_bonds = {1: 0.2, 2: 0.6}  # Low C to 1, high C to 2
    neighbor.C_bonds = {0: 0.2}
    dormant1.C_bonds = {0: 0.6}

    substrate.bonds = {(0, 1), (0, 2)}

    print("\nINITIAL SUBSTRATE STATE:")
    print("-" * 40)
    for nid, node in substrate.nodes.items():
        status = "ACTIVE" if node.is_active else "DORMANT"
        print(f"  Node {nid}: a={node.a:.2f}, F={node.F:.2f}, n={node.n} ({status})")
        if node.C_bonds:
            bonds_str = ", ".join(f"{k}:C={v:.2f}" for k, v in node.C_bonds.items())
            print(f"           Bonds: {bonds_str}")

    # Run division
    division = StrictCoreDivision(substrate)

    print("\n" + "=" * 70)
    print("DIVISION ATTEMPT")
    print("=" * 70)

    # Check eligibility
    eligibility = division.check_division_eligibility(0)
    print(f"\nEligibility for node 0:")
    print(f"  Eligible: {eligibility['eligible']}")
    if eligibility['reasons']:
        print(f"  Notes: {eligibility['reasons']}")
    print(f"  Fork candidates: {eligibility['fork_candidates']}")
    print(f"  Recruitable dormant: {eligibility['recruitable_dormant']}")

    # Attempt division
    result = division.run_division_attempt(0)

    print(f"\nDivision result:")
    print(f"  Success: {result['success']}")
    if result['success']:
        print(f"  Recruited node: {result['recruited_id']}")
        print(f"  Recruit's intrinsic agency: {result['recruited_agency']:.3f}")
        print(f"  (Note: Agency was NOT copied - recruit had this agency all along)")
        print(f"  Resource spent: {result['F_spent']:.3f}")

    print("\n" + "=" * 70)
    print("FINAL SUBSTRATE STATE:")
    print("-" * 40)
    for nid, node in substrate.nodes.items():
        status = "ACTIVE" if node.is_active else "DORMANT"
        changed = " (NEWLY ACTIVE)" if nid == result.get('recruited_id') else ""
        print(f"  Node {nid}: a={node.a:.2f}, F={node.F:.2f}, n={node.n} ({status}){changed}")
        if node.C_bonds:
            bonds_str = ", ".join(f"{k}:C={v:.2f}" for k, v in node.C_bonds.items())
            print(f"           Bonds: {bonds_str}")
        if node.records:
            print(f"           Records: {node.records}")

    # Explain the key difference
    print("\n" + "=" * 70)
    print("KEY INSIGHT: RECRUITMENT VS TEMPLATING")
    print("=" * 70)
    print("""
    OLD (VIOLATED DET):
    - Create new node with new ID
    - Copy parent's agency to child
    - Agency "grows" through copying

    NEW (DET-COMPLIANT):
    - Recruit existing dormant node
    - Dormant node KEEPS its own intrinsic agency
    - Agency is a gate for consent, not a copyable quantity
    - What transfers: pattern (bonds, phase, records)
    - What does NOT transfer: agency

    The "child" was always there, dormant, with its own a.
    Division doesn't create agency - it activates participation.
    """)

    # Demonstrate why this matters
    print("\n" + "=" * 70)
    print("WHY THIS MATTERS FOR DET")
    print("=" * 70)
    print("""
    1. INVIOLABLE AGENCY PRESERVED:
       - No external force sets or copies agency
       - Each node's a is intrinsic and respected
       - Division requires mutual consent (both a's sufficient)

    2. FIXED SUBSTRATE PRESERVED:
       - No nodes created or destroyed
       - Topology changes through bond operations
       - Dormant nodes are "potential" waiting to participate

    3. LOCALITY PRESERVED:
       - Each fork step touches only immediate bonds
       - No global operations or caps
       - Division propagates like a zipper, one bond at a time

    4. DERIVATIONAL INTEGRITY:
       - Resource cost comes from breaking/forming bonds
       - q emerges from F dynamics (existing q-locking)
       - No ad-hoc inheritance factors needed

    5. COHERENCE MODEL FIXED:
       - Division requires LOCAL low C at fork bond
       - Template bonds stay HIGH C (preserve pattern)
       - Not "globally low C to divide"
    """)


if __name__ == "__main__":
    demonstrate_strict_core_division()
