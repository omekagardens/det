# Deep Existence Theory (DET) v5  
**Strictly Local Relational Dynamics • Law-Bound Boundary Action • Past-Resolved Falsifiability**

DET v5 is an implementable theory of discrete, relational systems in which **time (presence), mass, gravity, and quantum-like behavior** arise from a closed update loop over **agents and bonds**, with **no global state available to local dynamics**. The model is designed to be **scientifically rigorous, falsifiable through past-resolved traces, and runnable as a simulation**.  [oai_citation:0‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

This repository is the **v5 research and implementation track**: code + experiments + documentation for validating (or falsifying) the canonical loop under strict locality constraints.

---

## What DET v5 claims (high level)

DET does *not* assume quantum postulates, relativistic axioms, or Standard Model structure as primitives. Instead, it proposes a local dynamical law set where:

- **Time** is an agent’s *rate of participation* in lawful events (“presence”).
- **Mass** is *coordination resistance* to participation (a readout from presence).
- **Gravity** is *memory imbalance* (structure retained as “debt”), sourced relative to a local baseline so uniform background does not gravitate.
- **Quantum-like behavior** appears as *local coherence in flow* on bonds.
- **Boundary action** (if enabled) is local, lawful, non-coercive, and cannot violate agency.

All of the above are meant to be tested in simulation as operational predictions, not treated as metaphors.  [oai_citation:1‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

---

## Foundational axiom (v5 scope)

**Present-moment participation requires three minimal capacities:**

1. **Information (I)** — pattern continuity (what can be carried forward)
2. **Agency (A)** — non-coercive choice (inviolable; cannot be directly modified by boundary operators)
3. **Movement (k)** — lawful eventing (time as realized update steps / event count)

DET treats this triad as the minimal structural pattern required for sustained “now”-participation in a lawful system. These are not moral assumptions; they are **operational constraints** required for simultaneously having:
- non-coercive agency,
- local time evolution,
- and recoverability (systems can heal rather than only freeze).  [oai_citation:2‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

---

## Core commitments (non-negotiables)

### 1) Strict locality / no hidden globals
- Every sum, average, and normalization is evaluated only on a **local node neighborhood** `𝒩_R(i)` or a **local bond neighborhood** `ℰ_R(i,j)`.
- Disconnected components cannot influence each other.
- No global normalizations, global averages, or shared hidden variables are allowed in the core dynamics.  [oai_citation:3‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

### 2) Closed update loop / implementability
DET is defined as a runnable, canonical update ordering (a single loop). Optional modules must be explicitly enabled and reported.

### 3) Law-bound boundary action (optional)
“Grace” is treated as **constrained, local, lawful operators** (e.g., injection/healing) that:
- are **non-coercive**,
- are **gated by agency** (if `a_i = 0`, boundary does not inject/heal),
- never directly modify agency (`a_i` is inviolable).  [oai_citation:4‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

### 4) Past-resolved falsifiability
The present moment is not directly falsifiable by design; DET is falsifiable through **past-resolved traces** (presence histories, flows, coherence histories, gravity readouts).

---

## What v5 is building (primitives → emergent readouts)

DET v5 uses only operational primitives:
- **Per-agent**: resource `F_i`, structural debt `q_i`, agency `a_i`, phase `θ_i`, local time/proper time `τ_i`, processing rate `σ_i`, event count `k_i`
- **Per-bond**: conductivity `σ_ij`, coherence `C_ij`  
(Additional optional modules may add extra state, but must remain local.)  [oai_citation:5‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

From these, v5 targets measurable readouts:
- **Presence / local clock rate** `P_i`
- **Mass** `M_i = P_i^{-1}`
- **Gravity potential** `Φ_i` sourced by baseline-referenced structure `ρ_i = q_i - b_i`
- **Transport / flow** decomposed into components (diffusive/phase, gravity drift, and optional constitutive terms)  [oai_citation:6‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

---

## Canonical dynamics (conceptual summary)

DET v5’s loop is centered on three interacting subsystems:

1. **Clock / presence law**  
Presence is the local clock rate and is reduced by coordination load, operational burden, and structural debt pressure through the dynamics.

2. **Conservative transport law**  
Flow is locally computed on bonds, antisymmetric pairwise, and conservative in closed systems (up to declared boundary operators).

3. **Structure formation (“q-locking”)**  
Structural debt `q_i` is not arbitrary: it evolves by a declared **q-locking law family**. Any published claim must specify the chosen q-locking rule and use it consistently across experiments.

v5 additionally emphasizes that **gravity is baseline-referenced** (contrast sources gravity, uniform background does not), to preserve a monopole term in 3D and enable stable far-field behavior.  [oai_citation:7‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

---

## Falsifiers (how this project can fail)

This project treats falsification as first-class. Examples of definitive failure modes include:

- **Locality violation**: embedding a subgraph into a larger graph changes its trajectory without a causal path.
- **Coercion**: any `a_i = 0` node receives boundary injection/healing.
- **Hidden globals**: any dependence on global aggregates or normalization outside `𝒩_R` / `ℰ_R`.
- **Binding failure (scoped)**: with gravity–flow coupling and agency-gated diffusion enabled, initially separated compact bodies fail to form bound states across broad initial conditions.
- **Mass non-conservation (scoped)**: closed-system mass drifts beyond tolerance under the conservative implementation rules.

(Full falsifier suite lives in the theory docs.)  [oai_citation:8‡README5_DRAFT.md](file-service://file-FCTZPrzQJ8341JMpHwk6kY)

---