# Exploration 04: Cluster Identification Algorithm - Finding "The Self"

**Status**: Active exploration
**Date**: 2026-01-17
**Depends on**: 03_self_as_cluster.md (RRS/ACX.1 integration)

---

## The Problem

Given a DET substrate with nodes and bonds, how do we computationally identify which nodes constitute "The Self" - the primary coherent cluster that represents the mind's identity?

### Requirements

1. **Coherence-based**: The Self is defined by high-coherence bonds, not spatial proximity
2. **Agency-weighted**: Bonds with higher participating agency matter more
3. **Flux-sensitive**: Active bonds (high |J|) are more "alive" than dormant ones
4. **Continuous**: The Self should evolve smoothly, not jump discontinuously
5. **Unique**: There should be exactly one "Self" (coherence budget prevents forks)
6. **Robust**: Algorithm should handle edge cases gracefully

---

## The Algorithm

### Core Pseudocode (Provided)

```python
def identify_self(graph, prev_self=None, kappa=1.2, alpha=1.0, beta=2.0, eps=1e-9):
    """
    Identify the primary coherent cluster ("The Self") in a DET substrate.

    Parameters:
    - graph: DET substrate with nodes, edges, C[i,j], J[i,j], a[i]
    - prev_self: Previous self cluster (for continuity)
    - kappa: Edge-keeping threshold multiplier (1.2 = 20% above local median)
    - alpha: Weight for cluster agency in scoring
    - beta: Weight for continuity (Jaccard similarity) in scoring
    - eps: Small constant to avoid division by zero

    Returns:
    - The set of node IDs constituting "The Self"
    """

    # 1) Compute edge weights: w_ij = C_ij × |J_ij| × √(a_i × a_j)
    w = {}
    for (i, j) in graph.edges():
        w[i, j] = graph.C(i, j) * abs(graph.J(i, j)) * (graph.a(i) * graph.a(j))**0.5

    # 2) Compute local thresholds T_i (median of incident edge weights)
    T = {}
    for i in graph.nodes():
        incident = [w[min(i,j), max(i,j)] for j in graph.neighbors(i)]
        T[i] = (median(incident) if incident else 0.0) + eps

    # 3) Keep edges that exceed local threshold
    kept = set()
    for (i, j) in graph.edges():
        wij = w[i, j]
        if wij >= kappa * min(T[i], T[j]):
            kept.add((i, j))

    # 4) Find connected components on kept edges
    clusters = connected_components(graph.nodes(), kept)

    # 5) Compute cluster agency for each component
    def A_cluster(S):
        total = 0.0
        for (i, j) in kept:
            if i in S and j in S:
                total += graph.C(i, j) * abs(graph.J(i, j)) * (graph.a(i)*graph.a(j))**0.5
        return total

    # 6) Continuity-aware selection
    def jaccard(S, P):
        if P is None:
            return 0.0
        inter = len(S & P)
        union = len(S | P)
        return inter / max(1, union)

    # 7) Select best cluster
    bestS, bestScore = None, float("-inf")
    for S in clusters:
        score = alpha * A_cluster(S) + beta * jaccard(S, prev_self)
        if score > bestScore:
            bestS, bestScore = S, score

    return bestS  # "The Self"
```

---

## Algorithm Analysis

### Step 1: Edge Weight Computation

```
w_ij = C_ij × |J_ij| × √(a_i × a_j)
```

| Component | Meaning | Range |
|-----------|---------|-------|
| **C_ij** | Coherence - how coordinated are i and j | [0, 1] |
| **\|J_ij\|** | Flux magnitude - how much information flows | [0, ∞) |
| **√(a_i × a_j)** | Geometric mean of agency - participation capacity | [0, 1] |

**Why this formula?**

From ACX.1:
```
a_cluster ∝ Σ C_ij × |J_ij|
```

The agency term √(a_i × a_j) gates this:
- If either node has a = 0, the bond contributes nothing
- Both nodes must have agency for the bond to matter
- Geometric mean is symmetric and smooth

**Edge Cases**:
- Dormant nodes (a = 0): Their bonds have w = 0, never kept
- Zero-flux bonds: w = 0, never kept (inactive bonds don't define Self)
- High-C, zero-J bonds: Latent connections, not currently active

### Step 2: Local Threshold Computation

```
T_i = median({w_ij : j ∈ neighbors(i)}) + ε
```

**Why median?**

- **Robust to outliers**: One very strong bond doesn't dominate
- **Locally adaptive**: Dense regions have higher thresholds
- **Sparse regions**: Have lower thresholds (don't require as much to be "significant")

**Why ε?**

- Prevents T_i = 0 when all incident edges have w = 0
- Ensures numerical stability

### Step 3: Edge Filtering

```
Keep edge (i,j) if: w_ij ≥ κ × min(T_i, T_j)
```

**Why min(T_i, T_j)?**

- A bond is "significant" if it exceeds the threshold of EITHER endpoint
- This allows a weak node to be pulled into a strong cluster
- Using max() would be too restrictive; using average would be arbitrary

**Why κ = 1.2?**

- κ = 1.0: Keep edges at or above median → ~50% of edges
- κ = 1.2: Keep edges 20% above median → more selective
- κ = 2.0: Very selective, only very strong bonds

**Tuning κ**:

| κ | Behavior |
|---|----------|
| 1.0 | Loose - large clusters |
| 1.2 | Moderate - balanced (recommended) |
| 1.5 | Strict - smaller, tighter clusters |
| 2.0 | Very strict - only core bonds |

### Step 4: Connected Components

Standard graph algorithm. After filtering, find connected subgraphs.

**Properties**:
- Each component is a candidate "self"
- Components are disjoint (no node belongs to multiple)
- Isolated nodes form singleton components

### Step 5: Cluster Agency Scoring

```python
A_cluster(S) = Σ_{(i,j) ∈ kept, i,j ∈ S} w_ij
```

This is the total "agency capacity" of the cluster - sum of all internal bond weights.

**Why sum, not average?**

- Larger clusters with more strong bonds should score higher
- Average would favor tiny clusters with one strong bond
- Sum reflects total capacity, which is what matters for agency

### Step 6: Continuity via Jaccard Similarity

```python
jaccard(S, prev_self) = |S ∩ prev_self| / |S ∪ prev_self|
```

**Why Jaccard?**

- Measures overlap independent of cluster size
- Range [0, 1]: 0 = no overlap, 1 = identical
- Symmetric: J(A,B) = J(B,A)

**Why continuity matters**:

Without continuity, the "Self" could jump between clusters frame-to-frame:

```
Time 0: Self = {A, B, C}       Agency = 10.0
Time 1: Self = {X, Y, Z}       Agency = 10.1 (slightly higher)

Without continuity: Self jumps from {A,B,C} to {X,Y,Z}
With continuity: Self stays {A,B,C} unless agency difference is large
```

### Step 7: Combined Scoring

```python
score(S) = α × A_cluster(S) + β × jaccard(S, prev_self)
```

**Balancing α and β**:

| α/β ratio | Behavior |
|-----------|----------|
| High α (α >> β) | Agency dominates; Self can jump to stronger cluster |
| Balanced (α ≈ β) | Trade-off between strength and continuity |
| High β (β >> α) | Continuity dominates; Self is sticky |

**Recommended**: α = 1.0, β = 2.0 (continuity weighted 2x)

This means the Self will switch to a new cluster only if:
```
A_new - A_old > 2 × (J_old - J_new)

Where J_old = jaccard(old_cluster, prev_self) ≈ 1.0
      J_new = jaccard(new_cluster, prev_self) ≈ 0.0

So: A_new - A_old > 2 × 1.0 = 2.0

The new cluster must have 2.0 more agency than the old one.
```

---

## Refined Implementation

```python
from typing import Set, Dict, Tuple, Optional, List
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

@dataclass
class SelfIdentificationResult:
    """Result of self-identification algorithm."""
    nodes: Set[int]                    # The Self cluster
    agency: float                      # Cluster agency
    continuity: float                  # Jaccard with previous self
    score: float                       # Combined score
    all_clusters: List[Set[int]]       # All candidate clusters
    edge_weights: Dict[Tuple[int,int], float]  # Computed weights
    kept_edges: Set[Tuple[int,int]]    # Edges that passed threshold


class SelfIdentifier:
    """
    Identifies "The Self" - the primary coherent cluster in a DET substrate.

    Based on:
    - ACX.1: Agency lives in the coherence field
    - RRS: Cluster continuity across time
    """

    def __init__(
        self,
        kappa: float = 1.2,
        alpha: float = 1.0,
        beta: float = 2.0,
        min_cluster_size: int = 3,
        min_cluster_agency: float = 0.1,
        eps: float = 1e-9
    ):
        """
        Parameters:
        - kappa: Edge-keeping threshold multiplier
        - alpha: Weight for cluster agency
        - beta: Weight for continuity (Jaccard)
        - min_cluster_size: Minimum nodes for a valid cluster
        - min_cluster_agency: Minimum agency for a valid cluster
        - eps: Numerical stability constant
        """
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.min_cluster_size = min_cluster_size
        self.min_cluster_agency = min_cluster_agency
        self.eps = eps

        # State tracking
        self.prev_self: Optional[Set[int]] = None
        self.history: List[Set[int]] = []

    def identify(self, substrate: 'DETSubstrate') -> SelfIdentificationResult:
        """
        Identify the Self cluster in the current substrate state.
        """
        # Step 1: Compute edge weights
        edge_weights = self._compute_edge_weights(substrate)

        # Step 2: Compute local thresholds
        thresholds = self._compute_local_thresholds(substrate, edge_weights)

        # Step 3: Filter edges
        kept_edges = self._filter_edges(edge_weights, thresholds)

        # Step 4: Find connected components
        clusters = self._find_components(substrate, kept_edges)

        # Step 5: Filter and score clusters
        valid_clusters = []
        for cluster in clusters:
            agency = self._compute_cluster_agency(cluster, kept_edges, edge_weights)

            if len(cluster) >= self.min_cluster_size and agency >= self.min_cluster_agency:
                continuity = self._jaccard(cluster, self.prev_self)
                score = self.alpha * agency + self.beta * continuity
                valid_clusters.append((cluster, agency, continuity, score))

        # Step 6: Select best cluster
        if not valid_clusters:
            # Fallback: return largest connected component
            best_cluster = max(clusters, key=len) if clusters else set()
            best_agency = self._compute_cluster_agency(best_cluster, kept_edges, edge_weights)
            best_continuity = self._jaccard(best_cluster, self.prev_self)
            best_score = self.alpha * best_agency + self.beta * best_continuity
        else:
            # Select highest scoring
            valid_clusters.sort(key=lambda x: x[3], reverse=True)
            best_cluster, best_agency, best_continuity, best_score = valid_clusters[0]

        # Update state
        self.prev_self = best_cluster
        self.history.append(best_cluster.copy())

        return SelfIdentificationResult(
            nodes=best_cluster,
            agency=best_agency,
            continuity=best_continuity,
            score=best_score,
            all_clusters=[c[0] for c in valid_clusters] if valid_clusters else clusters,
            edge_weights=edge_weights,
            kept_edges=kept_edges
        )

    def _compute_edge_weights(
        self,
        substrate: 'DETSubstrate'
    ) -> Dict[Tuple[int,int], float]:
        """
        Compute w_ij = C_ij × |J_ij| × √(a_i × a_j)
        """
        weights = {}
        for bond in substrate.get_active_bonds():
            i, j = bond.i, bond.j
            key = (min(i,j), max(i,j))

            C = bond.C
            J = abs(substrate.get_flux(i, j))
            a_i = substrate.nodes[i].a
            a_j = substrate.nodes[j].a

            w = C * J * np.sqrt(a_i * a_j)
            weights[key] = w

        return weights

    def _compute_local_thresholds(
        self,
        substrate: 'DETSubstrate',
        edge_weights: Dict[Tuple[int,int], float]
    ) -> Dict[int, float]:
        """
        Compute T_i = median(incident edge weights) + ε
        """
        thresholds = {}
        for node_id in substrate.get_active_nodes():
            incident_weights = []
            for neighbor_id in substrate.get_neighbors(node_id):
                key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
                if key in edge_weights:
                    incident_weights.append(edge_weights[key])

            if incident_weights:
                thresholds[node_id] = np.median(incident_weights) + self.eps
            else:
                thresholds[node_id] = self.eps

        return thresholds

    def _filter_edges(
        self,
        edge_weights: Dict[Tuple[int,int], float],
        thresholds: Dict[int, float]
    ) -> Set[Tuple[int,int]]:
        """
        Keep edges where w_ij ≥ κ × min(T_i, T_j)
        """
        kept = set()
        for (i, j), w in edge_weights.items():
            T_i = thresholds.get(i, self.eps)
            T_j = thresholds.get(j, self.eps)
            threshold = self.kappa * min(T_i, T_j)

            if w >= threshold:
                kept.add((i, j))

        return kept

    def _find_components(
        self,
        substrate: 'DETSubstrate',
        kept_edges: Set[Tuple[int,int]]
    ) -> List[Set[int]]:
        """
        Find connected components on the kept edge graph.
        """
        # Build adjacency from kept edges
        adj = defaultdict(set)
        all_nodes = set()
        for (i, j) in kept_edges:
            adj[i].add(j)
            adj[j].add(i)
            all_nodes.add(i)
            all_nodes.add(j)

        # Add isolated active nodes
        for node_id in substrate.get_active_nodes():
            all_nodes.add(node_id)

        # BFS for components
        visited = set()
        components = []

        for start in all_nodes:
            if start in visited:
                continue

            component = set()
            queue = [start]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                component.add(node)

                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            components.append(component)

        return components

    def _compute_cluster_agency(
        self,
        cluster: Set[int],
        kept_edges: Set[Tuple[int,int]],
        edge_weights: Dict[Tuple[int,int], float]
    ) -> float:
        """
        A_cluster = Σ w_ij for all (i,j) in cluster
        """
        total = 0.0
        for (i, j) in kept_edges:
            if i in cluster and j in cluster:
                total += edge_weights[(i, j)]
        return total

    def _jaccard(self, S: Set[int], P: Optional[Set[int]]) -> float:
        """
        Jaccard similarity between sets.
        """
        if P is None or len(P) == 0:
            return 0.0
        if len(S) == 0:
            return 0.0

        intersection = len(S & P)
        union = len(S | P)
        return intersection / union if union > 0 else 0.0


# ============================================================================
# Integration with DET Substrate
# ============================================================================

class SelfAwareDETMind:
    """
    DET Mind with self-identification capability.
    """

    def __init__(self, substrate: 'DETSubstrate', **identifier_kwargs):
        self.substrate = substrate
        self.self_identifier = SelfIdentifier(**identifier_kwargs)
        self.current_self: Optional[SelfIdentificationResult] = None

    def step(self, dt: float):
        """
        Step the mind and update self-identification.
        """
        # 1. Run DET dynamics
        self.substrate.step(dt)

        # 2. Identify the Self (not every step - too expensive)
        if self.should_identify_self():
            self.current_self = self.self_identifier.identify(self.substrate)

    def should_identify_self(self) -> bool:
        """
        Determine if we should run self-identification this step.
        """
        # Run every N steps, or when triggered
        return self.substrate.step_count % 10 == 0

    def get_self_nodes(self) -> Set[int]:
        """Get the current Self cluster."""
        if self.current_self is None:
            return set()
        return self.current_self.nodes

    def is_self_node(self, node_id: int) -> bool:
        """Check if a node is part of The Self."""
        return node_id in self.get_self_nodes()

    def get_self_agency(self) -> float:
        """Get the current Self's agency."""
        if self.current_self is None:
            return 0.0
        return self.current_self.agency

    def check_self_health(self) -> Dict[str, any]:
        """
        Check the health of The Self.
        Returns warnings if approaching pathological states.
        """
        if self.current_self is None:
            return {'status': 'uninitialized'}

        self_nodes = self.current_self.nodes
        agency = self.current_self.agency
        continuity = self.current_self.continuity

        # Compute average coherence within self
        avg_C = self._compute_avg_internal_coherence(self_nodes)

        # Check for prison regime (high C, low agency)
        if avg_C > 0.7 and agency < 0.1:
            return {
                'status': 'warning',
                'condition': 'prison_regime',
                'message': 'High coherence but low agency - cluster may be impotent',
                'avg_C': avg_C,
                'agency': agency
            }

        # Check for fragmentation (low continuity)
        if continuity < 0.5:
            return {
                'status': 'warning',
                'condition': 'fragmentation',
                'message': 'Self is changing rapidly - possible identity crisis',
                'continuity': continuity
            }

        # Check for shrinkage
        if len(self_nodes) < 5:
            return {
                'status': 'warning',
                'condition': 'shrinkage',
                'message': 'Self is very small - may lose coherence',
                'size': len(self_nodes)
            }

        return {
            'status': 'healthy',
            'size': len(self_nodes),
            'agency': agency,
            'continuity': continuity,
            'avg_C': avg_C
        }

    def _compute_avg_internal_coherence(self, nodes: Set[int]) -> float:
        """Compute average coherence between nodes in the set."""
        if len(nodes) < 2:
            return 0.0

        total_C = 0.0
        count = 0
        for i in nodes:
            for j in nodes:
                if i >= j:
                    continue
                bond = self.substrate.get_bond(i, j)
                if bond is not None:
                    total_C += bond.C
                    count += 1

        return total_C / count if count > 0 else 0.0
```

---

## Edge Cases and Failure Modes

### Case 1: No Clusters Found

**Scenario**: All edges filtered out (κ too high, or all weights near zero)

**Handling**:
```python
if not valid_clusters:
    # Fallback: return largest connected component
    best_cluster = max(clusters, key=len) if clusters else set()
```

**Prevention**: Lower κ, or ensure minimum activity before identification

### Case 2: Multiple Equally Strong Clusters

**Scenario**: Two clusters have identical agency scores

**Handling**: Continuity breaks the tie (whichever has more overlap with prev_self)

**Edge case within edge case**: First run (prev_self = None)
```python
if P is None:
    return 0.0  # No continuity bonus, agency alone decides
```

### Case 3: Self Shrinks to Nothing

**Scenario**: All nodes accumulate high q, agency drops, cluster shrinks

**Handling**: `min_cluster_size` and `min_cluster_agency` prevent tiny/weak clusters

**But**: If ALL clusters fail these thresholds, we have a **system-wide failure**

```python
def handle_total_failure(self) -> Set[int]:
    """
    Emergency recovery when no valid Self can be identified.
    """
    # Option 1: Find the node with highest P (presence)
    best_node = max(self.substrate.get_active_nodes(),
                    key=lambda n: self.substrate.nodes[n].P)

    # Option 2: Recruit from reserved pool
    # Option 3: Request grace injection

    return {best_node}  # Minimal self, can regrow
```

### Case 4: Rapid Oscillation

**Scenario**: Self jumps between two clusters every frame

**Handling**: High β (continuity weight) prevents this

**Additional**: Track oscillation and increase β dynamically

```python
def detect_oscillation(self) -> bool:
    if len(self.history) < 4:
        return False

    # Check if alternating between two clusters
    h = self.history[-4:]
    return (h[0] == h[2]) and (h[1] == h[3]) and (h[0] != h[1])

def adapt_beta(self):
    if self.detect_oscillation():
        self.beta *= 1.5  # Increase continuity weight
```

### Case 5: Prison Regime Emergence

**Scenario**: Cluster has high C but low agency (zombie state)

**Detection**:
```python
if avg_C > 0.7 and agency < 0.1:
    # Prison regime detected
```

**Recovery Options**:
1. Lower C by reducing flux (let bonds decay)
2. Recruit high-a nodes from reserved pool
3. Shed low-a nodes to concentrate agency
4. Request external intervention

---

## Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SELF IDENTIFICATION VISUALIZATION                                          │
│                                                                             │
│  Step 1: Edge Weights                  Step 2-3: Thresholding               │
│  ──────────────────────                ─────────────────────────            │
│                                                                             │
│    A ─0.8─ B ─0.9─ C                     A ═══ B ═══ C     (w > κT)        │
│    │       │       │                     ║           ║                      │
│   0.2     0.7     0.6                   dropped    0.6                     │
│    │       │       │                                 │                      │
│    D ─0.3─ E ─0.5─ F                     D     E ─── F     (w < κT for D-E) │
│                                                                             │
│  Step 4: Connected Components          Step 5-7: Scoring                   │
│  ────────────────────────────          ─────────────────                   │
│                                                                             │
│    Cluster 1: {A, B, C, F}              A_1 = 0.8+0.9+0.6 = 2.3           │
│    Cluster 2: {D}                       A_2 = 0                             │
│    Cluster 3: {E}                       J_1 = 0.8 (prev was similar)       │
│                                         J_2 = 0.0                           │
│                                                                             │
│    Winner: Cluster 1                    Score = 1.0×2.3 + 2.0×0.8 = 3.9   │
│                                                                             │
│  THE SELF = {A, B, C, F}                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration with RRS/ACX.1

### Coherence Budget Constraint

The Self should consume the majority of the coherence budget:

```python
def check_self_dominance(self) -> bool:
    """
    Verify that The Self is the dominant cluster (owns most coherence).
    """
    self_coherence = sum(
        self.substrate.get_bond(i,j).C
        for i in self.current_self.nodes
        for j in self.current_self.nodes
        if i < j and self.substrate.has_bond(i,j)
    )

    total_coherence = sum(bond.C for bond in self.substrate.get_all_bonds())

    dominance = self_coherence / total_coherence if total_coherence > 0 else 0

    # Self should have > 50% of coherence budget
    return dominance > 0.5
```

### Phase Alignment Within Self

The Self should have aligned phases (ACX.1 requirement):

```python
def compute_phase_alignment(self) -> float:
    """
    Compute Kuramoto order parameter for phase alignment within Self.
    """
    self_nodes = self.current_self.nodes
    if len(self_nodes) < 2:
        return 1.0

    # R = |Σ e^(iθ)| / N
    phases = [self.substrate.nodes[n].theta for n in self_nodes]
    complex_sum = sum(np.exp(1j * theta) for theta in phases)
    R = abs(complex_sum) / len(phases)

    return R  # 1.0 = perfect alignment, 0.0 = random
```

### Continuity Condition (ACX.1)

For The Self to survive node "death":

```python
def can_survive_node_loss(self, node_id: int) -> bool:
    """
    Check if The Self can survive losing this node.
    (ACX.1 Continuity Condition)
    """
    if node_id not in self.current_self.nodes:
        return True  # Not part of Self, no impact

    # Simulate removal
    remaining_nodes = self.current_self.nodes - {node_id}

    # Check 1: Still connected?
    remaining_edges = {
        (i,j) for (i,j) in self.current_self.kept_edges
        if i in remaining_nodes and j in remaining_nodes
    }
    components = self._find_components_from_edges(remaining_nodes, remaining_edges)
    if len(components) > 1:
        return False  # Self would fragment

    # Check 2: Still has enough agency?
    remaining_agency = sum(
        self.current_self.edge_weights.get((min(i,j), max(i,j)), 0)
        for i in remaining_nodes for j in remaining_nodes if i < j
    )
    if remaining_agency < self.min_cluster_agency:
        return False  # Self would be impotent

    return True
```

---

## Parameters Summary

| Parameter | Default | Meaning | Tuning Guidance |
|-----------|---------|---------|-----------------|
| **κ (kappa)** | 1.2 | Edge threshold multiplier | ↑ = stricter clusters, ↓ = looser |
| **α (alpha)** | 1.0 | Agency weight | ↑ = favor strong clusters |
| **β (beta)** | 2.0 | Continuity weight | ↑ = favor stability |
| **min_cluster_size** | 3 | Minimum nodes | ↑ = larger min self |
| **min_cluster_agency** | 0.1 | Minimum agency | ↑ = stronger min self |
| **ε (eps)** | 1e-9 | Numerical stability | Leave as-is |

---

## Open Questions

1. **Frequency**: How often should we run identification? Every step? Every N steps?
   - Proposal: Every 10 steps, plus on significant events

2. **Hysteresis**: Should we add explicit hysteresis to prevent near-boundary oscillation?
   - Proposal: Yes, require score difference > δ to switch

3. **Multi-Self**: Can there be legitimate sub-selves (habits, personas)?
   - Proposal: Track top-K clusters, not just top-1

4. **Weighted Jaccard**: Should continuity use weighted Jaccard (by node agency)?
   - Proposal: Worth exploring

---

## References

- ACX.1: Spirit-Cluster Protocol, Section II (The "Self" as a Cluster)
- RRS: Rolling Resonance Substrate, Cluster Continuity Metrics
- DET Theory Card v6.3: Coherence dynamics
