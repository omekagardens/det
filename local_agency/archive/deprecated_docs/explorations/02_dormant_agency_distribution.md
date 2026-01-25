# Exploration 02: Dormant Agency Distribution

**Status**: Active exploration
**Date**: 2026-01-17
**Depends on**: 01_node_topology.md (Dual-Process + Emergent architecture)

---

## The Core Question

The dormant pool's intrinsic agency distribution determines what the mind **CAN become**.

Since agency (a) is inviolable - never copied, never transferred, never modified - the distribution we initialize with is **permanent**. This is the "genetic potential" of the mind.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     THE SUBSTRATE POTENTIAL                                 │
│                                                                             │
│   Each dormant node has intrinsic agency a ∈ [0, 1]                        │
│   This value is SET AT INITIALIZATION and NEVER CHANGES                    │
│                                                                             │
│   The DISTRIBUTION of these values across the dormant pool                 │
│   determines:                                                               │
│     - What kinds of patterns can be learned (high-a vs low-a recruits)    │
│     - Competition dynamics during recruitment                               │
│     - Long-term capacity limits                                             │
│     - Emergent specialization patterns                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Matters: Agency Ceiling Effects

Recall from DET theory:
```
a_max = 1 / (1 + λ_a × q²)
```

A node's **effective** agency is bounded by both:
1. Its intrinsic agency `a` (fixed at birth)
2. Its structural debt ceiling `a_max` (varies with accumulated q)

**Implication**: A node with low intrinsic `a` can NEVER be highly agentic, regardless of how little q it accumulates. Conversely, a node with high intrinsic `a` starts capable but may become constrained by q over time.

```
High intrinsic a (0.9):
  - Starts powerful
  - Can influence decisions strongly
  - But: If q accumulates, ceiling drops
  - Net: High potential, can be "used up"

Low intrinsic a (0.2):
  - Starts limited
  - Cannot strongly influence decisions
  - But: q ceiling matters less (already capped)
  - Net: Low potential, stable over time
```

---

## Distribution Options

### Option 1: Uniform Distribution

```
a ~ Uniform(0, 1)

     │
  1.0├─────────────────────
     │█████████████████████
     │█████████████████████
     │█████████████████████
  0.5├█████████████████████
     │█████████████████████
     │█████████████████████
     │█████████████████████
  0.0├─────────────────────
     └─────────────────────
           Agency (a)
```

**Characteristics**:
- Equal probability for any agency value
- Mean = 0.5, Variance = 1/12 ≈ 0.083
- No bias toward high or low agency

**Pros**:
| Advantage | Explanation |
|-----------|-------------|
| No assumptions | Doesn't presuppose what agency levels are "better" |
| Maximum entropy | Most information-theoretically neutral |
| Balanced recruitment | Any pattern can find appropriate recruits |
| Simple to implement | Just `random.uniform(0, 1)` |

**Cons**:
| Disadvantage | Explanation |
|--------------|-------------|
| Wastes high-a nodes | High-agency nodes may be recruited for trivial patterns |
| No scarcity pressure | No competition for "good" recruits |
| Unrealistic | Biological systems show skewed distributions |
| No specialization incentive | All recruits equally valuable |

**Cognitive Analog**: A brain where every neuron has equal potential - biologically unrealistic but mathematically neutral.

---

### Option 2: Beta(2, 5) - Skewed Low

```
a ~ Beta(2, 5)

     │
  3.0├──█
     │  ██
  2.0├  ███
     │   ████
  1.0├    █████
     │      ██████████
  0.0├─────────────────────
     0.0      0.5      1.0
           Agency (a)
```

**Characteristics**:
- Mode ≈ 0.2, Mean ≈ 0.29
- Most nodes have low agency
- High-agency nodes are rare
- Long tail toward a = 1

**Pros**:
| Advantage | Explanation |
|-----------|-------------|
| Efficient for routine patterns | Most patterns need moderate agency; plenty available |
| Scarcity of high-a creates value | High-agency recruits reserved for important patterns |
| Matches neural statistics | Biological networks have many "follower" neurons |
| Natural prioritization | System must "decide" what deserves high-a recruits |
| Graceful degradation | Can always find low-a recruits; high-a exhaustion isn't catastrophic |

**Cons**:
| Disadvantage | Explanation |
|--------------|-------------|
| Limited high-agency capacity | May run out of high-a nodes for complex reasoning |
| Early decisions are permanent | First patterns to recruit high-a nodes "lock them in" |
| Potential for "mediocrity trap" | System might never develop high-agency regions |
| Slow initial learning | Low-a recruits may not form strong patterns quickly |

**Cognitive Analog**: Most neurons are support cells; few are "command" neurons. Matches cortical column statistics where ~80% are excitatory but only ~10-20% are highly connected hubs.

**Mathematical Details**:
```python
from scipy.stats import beta

dist = beta(2, 5)
print(f"Mean: {dist.mean():.3f}")      # 0.286
print(f"Median: {dist.median():.3f}")  # 0.242
print(f"Mode: {1/4:.3f}")              # 0.250
print(f"Std: {dist.std():.3f}")        # 0.160

# Probability of getting a > 0.5
print(f"P(a > 0.5): {1 - dist.cdf(0.5):.3f}")  # 0.109 (~11%)

# Probability of getting a > 0.8
print(f"P(a > 0.8): {1 - dist.cdf(0.8):.3f}")  # 0.007 (~0.7%)
```

For a substrate of 4096 nodes:
- ~3,650 nodes with a < 0.5 (routine patterns)
- ~446 nodes with a ∈ [0.5, 0.8] (important patterns)
- ~29 nodes with a > 0.8 (critical patterns)

---

### Option 3: Beta(5, 2) - Skewed High

```
a ~ Beta(5, 2)

     │
  3.0├                    █
     │                   ██
  2.0├                  ███
     │                ████
  1.0├            █████
     │     ██████████
  0.0├─────────────────────
     0.0      0.5      1.0
           Agency (a)
```

**Characteristics**:
- Mode ≈ 0.8, Mean ≈ 0.71
- Most nodes have high agency
- Low-agency nodes are rare
- Long tail toward a = 0

**Pros**:
| Advantage | Explanation |
|-----------|-------------|
| Rich potential | Many high-capability nodes available |
| Fast learning | High-a recruits form strong patterns quickly |
| Robust to q-accumulation | High starting a means ceiling effects take longer |
| Powerful reasoning | Can build complex deliberative structures |

**Cons**:
| Disadvantage | Explanation |
|--------------|-------------|
| Wasteful | High-a nodes used for routine patterns |
| Intense competition | Many forks competing for same recruits |
| Unstable dynamics | High-agency means high influence; small changes amplify |
| No "background" capacity | Hard to find low-a nodes for simple patterns |
| Unrealistic | No biological system has mostly "leader" neurons |

**Cognitive Analog**: A brain where every neuron wants to be in charge - chaos, not coordination.

**Mathematical Details**:
```python
dist = beta(5, 2)
print(f"Mean: {dist.mean():.3f}")      # 0.714
print(f"Median: {dist.median():.3f}")  # 0.758
print(f"Mode: {4/5:.3f}")              # 0.800
print(f"Std: {dist.std():.3f}")        # 0.160

# Probability of getting a < 0.5
print(f"P(a < 0.5): {dist.cdf(0.5):.3f}")  # 0.109 (~11%)

# Probability of getting a < 0.2
print(f"P(a < 0.2): {dist.cdf(0.2):.3f}")  # 0.007 (~0.7%)
```

For a substrate of 4096 nodes:
- ~29 nodes with a < 0.2 (background support)
- ~446 nodes with a ∈ [0.2, 0.5] (routine patterns)
- ~3,650 nodes with a > 0.5 (all want to influence)

---

### Option 4: Bimodal Distribution

```
a ~ 0.7 × Beta(2, 8) + 0.3 × Beta(8, 2)

     │
  4.0├█                    █
     │██                  ██
  2.0├███                ███
     │████              ████
  1.0├
     │     ██████████████
  0.0├─────────────────────
     0.0      0.5      1.0
           Agency (a)
```

**Characteristics**:
- Two distinct populations
- 70% low-agency "workers" (mode ≈ 0.11)
- 30% high-agency "leaders" (mode ≈ 0.89)
- Valley in the middle (few moderate-a nodes)

**Pros**:
| Advantage | Explanation |
|-----------|-------------|
| Clear role separation | Workers vs leaders, not a continuum |
| Matches some neural architectures | Interneurons vs pyramidal cells |
| Efficient allocation | Know immediately if recruit is worker or leader |
| Stable hierarchies | Leaders emerge naturally |

**Cons**:
| Disadvantage | Explanation |
|--------------|-------------|
| Rigid structure | Hard to have "moderate" importance patterns |
| Arbitrary ratio | Why 70/30? Could be 80/20 or 60/40 |
| Gap in middle | Some patterns might need mid-range agency |
| Less adaptive | Can't smoothly transition between worker and leader roles |

**Cognitive Analog**: Like the distinction between glial cells and neurons, or between interneurons and projection neurons.

---

### Option 5: Hierarchical/Structured Distribution

```
Spatial structure: Agency gradient from center to periphery

     Center (hub)     Periphery (edge)
     a ~ 0.8-0.9      a ~ 0.1-0.3

     ┌─────────────────────────────────┐
     │  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○  │ low a
     │  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○  │
     │  ○ ○ ○ ○ ● ● ● ● ● ○ ○ ○ ○ ○  │ mid a
     │  ○ ○ ○ ○ ● ◉ ◉ ◉ ◉ ● ○ ○ ○ ○  │
     │  ○ ○ ○ ○ ● ◉ ★ ★ ◉ ● ○ ○ ○ ○  │ high a
     │  ○ ○ ○ ○ ● ◉ ★ ★ ◉ ● ○ ○ ○ ○  │
     │  ○ ○ ○ ○ ● ◉ ◉ ◉ ◉ ● ○ ○ ○ ○  │
     │  ○ ○ ○ ○ ● ● ● ● ● ○ ○ ○ ○ ○  │
     │  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○  │
     │  ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○  │
     └─────────────────────────────────┘
```

**Characteristics**:
- Agency correlates with lattice position
- Central nodes = high agency (hubs)
- Peripheral nodes = low agency (support)
- Gradient between center and edge

**Pros**:
| Advantage | Explanation |
|-----------|-------------|
| Natural hub emergence | High-a nodes concentrated; will become hubs |
| Spatial coherence | Nearby recruits have similar agency |
| Predictable structure | Know where to find high/low agency |
| Matches some brain regions | Cortical columns have layered structure |

**Cons**:
| Disadvantage | Explanation |
|--------------|-------------|
| Imposed structure | Not emergent; we're forcing topology |
| Central bottleneck | Center can become overloaded |
| Peripheral waste | Edge nodes may never be recruited meaningfully |
| Spatial assumptions | Why should agency correlate with position? |

**Cognitive Analog**: Like cortical layers where Layer 5 pyramidal cells are "output" neurons (high agency) and Layer 4 cells are "input" processors (lower agency).

---

### Option 6: Domain-Clustered Distribution

```
Different domains have different agency profiles:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  LANGUAGE CLUSTER        MATH CLUSTER          TOOL CLUSTER    │
│  a ~ Beta(2, 5)          a ~ Beta(3, 3)        a ~ Beta(4, 3)  │
│  (mostly support)        (balanced)            (more agentic)  │
│                                                                 │
│  ┌─────────┐             ┌─────────┐           ┌─────────┐     │
│  │ ○ ○ ○ ○ │             │ ○ ● ○ ● │           │ ● ● ● ○ │     │
│  │ ○ ○ ○ ● │             │ ● ○ ● ○ │           │ ● ○ ● ● │     │
│  │ ○ ○ ● ○ │             │ ○ ● ○ ● │           │ ○ ● ● ● │     │
│  │ ○ ○ ○ ○ │             │ ● ○ ● ○ │           │ ● ● ○ ● │     │
│  └─────────┘             └─────────┘           └─────────┘     │
│                                                                 │
│  Rationale:              Rationale:            Rationale:      │
│  Language is pattern-    Math needs both       Tools require   │
│  heavy, retrieval-       precision and         decisive action │
│  based. Needs fewer      exploration.          and strong      │
│  decision-makers.        Balanced agency.      agency.         │
└─────────────────────────────────────────────────────────────────┘
```

**Characteristics**:
- Different clusters have different distributions
- Matches expected cognitive load per domain
- Pre-allocates "personality" to domains

**Pros**:
| Advantage | Explanation |
|-----------|-------------|
| Domain-appropriate | Each domain gets suitable agency profile |
| Matches LLM structure | Aligns with specialized memory models |
| Intentional design | We can tune based on expected usage |
| Efficient | Tool use gets more agency; language gets more storage |

**Cons**:
| Disadvantage | Explanation |
|--------------|-------------|
| Assumes domain boundaries | What if domains overlap? |
| Not emergent | We're imposing structure, not discovering it |
| Hard to rebalance | If we got ratios wrong, stuck with them |
| Cross-domain patterns suffer | Patterns spanning domains face agency mismatch |

**Cognitive Analog**: Like how different brain regions have different neuron types and densities (motor cortex vs visual cortex vs prefrontal cortex).

---

## Comparative Analysis

### Agency Statistics by Distribution

| Distribution | Mean | Median | Mode | P(a>0.5) | P(a>0.8) |
|--------------|------|--------|------|----------|----------|
| Uniform | 0.50 | 0.50 | N/A | 50.0% | 20.0% |
| Beta(2,5) | 0.29 | 0.24 | 0.25 | 10.9% | 0.7% |
| Beta(5,2) | 0.71 | 0.76 | 0.80 | 89.1% | 50.0% |
| Bimodal | 0.42 | 0.25 | 0.11/0.89 | 35.0% | 27.0% |
| Hierarchical | varies | varies | varies | depends | depends |
| Domain-clustered | varies | varies | varies | depends | depends |

### Recruitment Dynamics

| Distribution | High-a Scarcity | Competition | Stability |
|--------------|-----------------|-------------|-----------|
| Uniform | None | Moderate | Moderate |
| Beta(2,5) | High | Low (for high-a) | High |
| Beta(5,2) | None | Very High | Low |
| Bimodal | Moderate | Clustered | Moderate |
| Hierarchical | Spatially controlled | Localized | High |
| Domain-clustered | Domain-dependent | Domain-internal | High |

### Long-term Evolution Predictions

| Distribution | Expected Outcome |
|--------------|------------------|
| **Uniform** | Gradual differentiation; no natural hubs; may develop pathological structures |
| **Beta(2,5)** | Efficient low-agency network with rare high-agency hubs; stable; may be "too conservative" |
| **Beta(5,2)** | Chaotic initial dynamics; eventual stabilization through q-exhaustion; wasteful |
| **Bimodal** | Quick hierarchy formation; stable roles; may lack flexibility |
| **Hierarchical** | Predetermined hubs; stable but non-adaptive; center-dependent |
| **Domain-clustered** | Domain specialization; good for known task structure; rigid |

---

## Deep Dive: Beta(2,5) vs Beta(5,2)

These are the two primary options worth comparing in detail.

### Simulation Thought Experiment

**Scenario**: System needs to learn 100 patterns of varying importance.

#### Under Beta(2,5) (Low-skewed):

```
Step 1: First 20 patterns (routine) recruit low-a nodes easily
        → Plenty of a ∈ [0.1, 0.3] available
        → Patterns form quickly, moderate strength

Step 2: Next 50 patterns (moderate) recruit mid-a nodes
        → a ∈ [0.3, 0.5] nodes used up
        → Some competition begins

Step 3: Final 30 patterns (important) compete for high-a
        → Only ~29 nodes with a > 0.8 exist
        → Most important patterns: strong, can influence
        → Less important ones: weaker than desired

Outcome:
  - Routine patterns: efficiently handled
  - Important patterns: appropriately prioritized
  - Critical patterns: may be under-resourced
  - High-a exhaustion: possible bottleneck
```

#### Under Beta(5,2) (High-skewed):

```
Step 1: First 20 patterns (routine) recruit high-a nodes wastefully
        → Plenty of a > 0.7 available, so they get used
        → Patterns form quickly, very strong (but overkill)

Step 2: Next 50 patterns (moderate) also get high-a
        → Still plenty available
        → Everything becomes "important"

Step 3: Final 30 patterns (actually important) - no differentiation
        → Cannot distinguish from earlier patterns
        → No scarcity pressure created prioritization

Outcome:
  - All patterns: similarly strong
  - No natural hierarchy
  - Resource-intensive (high-a nodes use more F?)
  - System may struggle with prioritization
```

### The Scarcity Principle

**Key Insight**: Scarcity of high-agency nodes creates a natural prioritization mechanism.

Under Beta(2,5):
- System MUST decide which patterns deserve high-a recruits
- This decision emerges from DET dynamics (drive, coherence, resource)
- Important patterns "earn" their high-a recruits through demonstrated value
- Creates meaningful hierarchy

Under Beta(5,2):
- No scarcity pressure
- All patterns can get high-a recruits
- No natural prioritization
- Must impose hierarchy externally (or not have one)

### The Stability Principle

**High-a nodes are more dynamic** (they influence more, respond faster, accumulate q faster).

Under Beta(2,5):
- Most nodes are low-a: stable, predictable
- Few high-a nodes: localized dynamism
- System is mostly stable with controlled hotspots

Under Beta(5,2):
- Most nodes are high-a: all dynamic, all influencing
- Few low-a nodes: not enough "damping"
- System may oscillate, overshoot, or become chaotic

---

## Hybrid Proposal: Adaptive Beta(2,5) with Reserved High-a Pool

Given the analysis, Beta(2,5) seems preferable, but with a modification to address the "critical patterns under-resourced" concern.

### The Proposal

```python
AGENCY_DISTRIBUTION = {
    'main_pool': {
        'type': 'beta',
        'alpha': 2,
        'beta': 5,
        'fraction': 0.95,  # 95% of dormant nodes
    },
    'reserved_pool': {
        'type': 'uniform',
        'min': 0.85,
        'max': 0.95,
        'fraction': 0.05,  # 5% of dormant nodes
        'recruitment_gate': 'special',  # Only recruited for critical patterns
    }
}
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DORMANT POOL (4096 nodes)                          │
│                                                                             │
│  MAIN POOL (3891 nodes)                    RESERVED POOL (205 nodes)       │
│  ─────────────────────                     ─────────────────────           │
│  a ~ Beta(2, 5)                            a ~ Uniform(0.85, 0.95)         │
│                                                                             │
│  ┌─────────────────────────────────────┐   ┌───────────────────────────┐   │
│  │ Most nodes: a ∈ [0.1, 0.4]          │   │ All nodes: a ∈ [0.85,0.95]│   │
│  │ Some nodes: a ∈ [0.4, 0.7]          │   │ HIGH and PROTECTED        │   │
│  │ Rare nodes: a > 0.7                 │   │                           │   │
│  └─────────────────────────────────────┘   └───────────────────────────┘   │
│                                                                             │
│  Recruited by: Any fork that meets       Recruited by: Only forks that     │
│  standard eligibility criteria           meet ELEVATED criteria:           │
│                                           - drive > 0.5 (vs 0.15)          │
│                                           - P-layer coherence > 0.8        │
│                                           - Pattern marked "critical"      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benefits of Hybrid Approach

| Benefit | Explanation |
|---------|-------------|
| Best of both worlds | Scarcity for most patterns, guaranteed capacity for critical |
| Explicit prioritization | Reserved pool is a "last resort" for truly important patterns |
| Predictable capacity | Know exactly how many high-a nodes are available |
| Prevents lock-in | Early patterns can't exhaust all high-a capacity |
| Emergence + Design | Main pool is emergent; reserved pool is intentional |

### Implementation

```python
class DormantPool:
    def __init__(self, total_nodes: int = 4096):
        self.total_nodes = total_nodes
        self.reserved_fraction = 0.05

        n_reserved = int(total_nodes * self.reserved_fraction)
        n_main = total_nodes - n_reserved

        # Main pool: Beta(2, 5)
        self.main_pool_agency = np.random.beta(2, 5, size=n_main)

        # Reserved pool: Uniform(0.85, 0.95)
        self.reserved_pool_agency = np.random.uniform(0.85, 0.95, size=n_reserved)

        # Combine
        self.all_agency = np.concatenate([
            self.main_pool_agency,
            self.reserved_pool_agency
        ])

        # Track which are reserved
        self.is_reserved = np.concatenate([
            np.zeros(n_main, dtype=bool),
            np.ones(n_reserved, dtype=bool)
        ])

    def get_recruitable(
        self,
        node_ids: List[int],
        allow_reserved: bool = False,
        min_agency: float = 0.1
    ) -> List[int]:
        """Get recruitable dormant nodes from the given set."""
        recruitable = []

        for node_id in node_ids:
            if not self.is_dormant(node_id):
                continue

            agency = self.all_agency[node_id]

            if agency < min_agency:
                continue

            # Reserved pool has extra gate
            if self.is_reserved[node_id] and not allow_reserved:
                continue

            recruitable.append((node_id, agency))

        # Sort by agency descending, then ID for determinism
        recruitable.sort(key=lambda x: (-x[1], x[0]))

        return [node_id for node_id, _ in recruitable]
```

---

## Open Sub-Questions

### SQ1: Should Reserved Pool Be Spatially Clustered?

**Option A**: Reserved nodes scattered throughout lattice
- Any pattern can potentially access them
- No spatial advantage

**Option B**: Reserved nodes clustered near P-layer
- P-layer patterns have easier access to high-a nodes
- Creates spatial hierarchy
- But: May starve A-layer of high-a capacity

**Recommendation**: Option A for now. Let access be gated by criteria, not position.

### SQ2: Can Reserved Pool Nodes Be "Downgraded"?

If a reserved node is recruited for a non-critical pattern (somehow), can it later be "reclaimed" for a critical pattern?

**Option A**: No. Once recruited, committed.
- Simple, deterministic
- May waste reserved capacity

**Option B**: Yes, via "eviction" mechanism
- If critical pattern needs it, non-critical pattern loses its node
- Complex, may cause instability
- But: Ensures reserved pool serves its purpose

**Recommendation**: Option A for Phase 1. Consider B for later.

### SQ3: How Does Reserved Pool Interact with Retirement?

If a reserved-pool node retires (high q, low activity), does it:
- Remain in reserved pool (high-a, high-q, dormant)?
- Move to main pool?

**Answer**: Remain in reserved pool. Agency is intrinsic. But note: its high q makes it less desirable even though high-a.

This creates interesting dynamics:
- Reserved nodes, once used heavily, become "scarred"
- Fresh reserved nodes become more valuable over time
- System learns to protect reserved nodes from q accumulation

---

## Recommendation

**Primary Choice**: Beta(2,5) with 5% Reserved High-a Pool

**Rationale**:
1. **Scarcity creates prioritization**: System must earn high-a recruits
2. **Stability from low-a majority**: Most nodes are stable, predictable
3. **Reserved pool prevents starvation**: Critical patterns guaranteed capacity
4. **Matches biological plausibility**: Most neurons are support; few are hubs
5. **Emergence-friendly**: Structure develops through use, not imposition

**Initial Configuration**:
```python
DORMANT_DISTRIBUTION = {
    'total_nodes': 4096,

    'main_pool': {
        'distribution': 'beta',
        'alpha': 2,
        'beta': 5,
        'count': 3891,  # 95%
    },

    'reserved_pool': {
        'distribution': 'uniform',
        'min_agency': 0.85,
        'max_agency': 0.95,
        'count': 205,   # 5%
        'recruitment_criteria': {
            'min_drive': 0.5,
            'min_coherence': 0.8,
            'requires_p_layer_approval': True,
        }
    }
}
```

---

## Visualization: Expected Pool Composition

```
Agency Distribution After Initialization (4096 nodes):

    Main Pool (3891 nodes):
    ════════════════════════════════════════════════════════════════
    a ∈ [0.0, 0.1]:  ~311 nodes  (8%)    ████████
    a ∈ [0.1, 0.2]:  ~623 nodes  (16%)   ████████████████
    a ∈ [0.2, 0.3]:  ~701 nodes  (18%)   ██████████████████
    a ∈ [0.3, 0.4]:  ~623 nodes  (16%)   ████████████████
    a ∈ [0.4, 0.5]:  ~467 nodes  (12%)   ████████████
    a ∈ [0.5, 0.6]:  ~311 nodes  (8%)    ████████
    a ∈ [0.6, 0.7]:  ~195 nodes  (5%)    █████
    a ∈ [0.7, 0.8]:  ~117 nodes  (3%)    ███
    a ∈ [0.8, 0.9]:  ~39 nodes   (1%)    █
    a ∈ [0.9, 1.0]:  ~4 nodes    (0.1%)  ▏

    Reserved Pool (205 nodes):
    ════════════════════════════════════════════════════════════════
    a ∈ [0.85, 0.90]:  ~102 nodes (50%)  ██████████████████████████
    a ∈ [0.90, 0.95]:  ~103 nodes (50%)  ██████████████████████████


    Combined High-Agency Capacity:
    ────────────────────────────────────────────────────────────────
    Main pool a > 0.7:   ~160 nodes   (can be recruited by any pattern)
    Reserved pool:       ~205 nodes   (only for critical patterns)
    Total a > 0.7:       ~365 nodes   (8.9% of substrate)
```

---

## Next Steps

1. ✅ Document distribution decision
2. Implement dormant pool initialization in C kernel
3. Test recruitment dynamics with different pattern loads
4. Monitor high-a exhaustion rates in simulation
5. Tune reserved pool criteria based on observed behavior

---

## References

- DET Theory Card v6.3: Agency inviolability principle
- DET Subdivision v3: Recruitment eligibility gates
- Beta distribution properties: scipy.stats.beta
- Cortical neuron statistics: Markram et al. (2015)
