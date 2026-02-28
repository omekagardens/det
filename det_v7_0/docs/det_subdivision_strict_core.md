# DET v6.5 Extension: Strict-Core Compliant Agentic Subdivision

## Abstract

This document presents a **DET-compliant** mechanism for agentic "division" (birth/multiplication) that respects all core principles:
- **Inviolable agency** - Agency cannot be set or copied from outside
- **Fixed substrate** - Nodes exist, are not created/destroyed
- **Strict locality** - All operations touch only bond neighborhoods
- **Closed update loop** - Division is just another local operator

The key insight: **Division is recruitment, not creation. Pattern transfers, not agency.**

---

## 1. Core Principle Violations in Naive Approach

The original subdivision theory (v1) violated DET in several ways:

| Violation | What v1 Did | Why It's Wrong |
|-----------|-------------|----------------|
| Node creation | `new DETAgent(id=next_id)` | Fixed substrate - nodes exist |
| Agency copying | `a_new = a_parent × (1-ε)` | Inviolable agency |
| Global threshold | `mean(C) < threshold` | Locality violation |
| Ad-hoc inheritance | `q_new = q_parent × factor` | Breaks derivational integrity |
| Global caps | `agents = agents[:max]` | Non-local operation |

---

## 2. The Correct Model: Recruitment-Based Division

### 2.1 Key Insight from DNA

DNA replication doesn't "create" atoms - it **recruits** existing atoms into a pattern. The new strand uses nucleotides from the cellular pool, not created from nothing.

**DET analog**: Division doesn't create nodes. It **recruits dormant nodes** that already exist in the substrate, activating their participation.

### 2.2 The Fixed Substrate

The DET substrate contains:
- **Active nodes** (n=1): Currently participating in dynamics
- **Dormant nodes** (n=0): Exist but not participating

Each node has an **intrinsic agency** a_i that is:
- Set at initialization (or determined by deeper physics)
- **Never modified** by other nodes or boundaries
- Used only as a **gate** for participation

### 2.3 Division as Mutual Consent Recruitment

For node i to "divide" (recruit dormant node k):

```
Parent gate:    a_i ≥ a_min^div     (parent capable of initiating)
Recruit gate:   a_k ≥ a_min^join    (recruit capable of participating)
Bond gate:      Can form C_ik if both open
```

**Neither node's agency is changed.** The gates just check eligibility.

---

## 3. The Fork Model (Zipper Mechanism)

Division proceeds through **local forks**, one bond at a time:

```
PHASE 1: FORK OPENING
- Identify bond (i,j) with C_ij < C_fork_max (locally low coherence)
- Pay resource: ΔF_i = -κ_break × C_ij
- Break bond: C_ij → ~0

PHASE 2: RECRUITMENT
- Find dormant node k in N_R(i) with a_k ≥ a_min^join
- Mutual consent: both i and k meet thresholds
- Activate: n_k: 0 → 1

PHASE 3: REBONDING
- Pay resource: ΔF_i = -κ_form × C_init
- Form bond: C_ik = C_init
- Add to substrate bonds

PHASE 4: PATTERN TRANSFER
- Transfer: phase alignment seed, records, topology hints
- Do NOT transfer: agency (a_k stays intrinsic)
```

### 3.1 What Transfers vs What Doesn't

| Transfers | Does NOT Transfer |
|-----------|-------------------|
| Phase alignment (small θ nudge) | Agency a |
| Records/traces (hints) | Resource F (recruit has own) |
| Bond topology (who to connect to) | Structural debt q (emerges from own dynamics) |

The recruit is **not a copy** of the parent. It's an independent node that joins the pattern.

---

## 4. Resource Cost (ATP Analog)

Division costs resource, paid locally:

```
Breaking bond (i,j):   ΔF_i = -κ_break × C_ij
Forming bond (i,k):    ΔF_i = -κ_form × C_init
```

The q-locking mechanism handles debt naturally:
```
q^+ = clip(q + α_q × max(0, -ΔF), 0, 1)
```

No special "q_inheritance_factor" needed. Structure emerges from dynamics.

---

## 5. Coherence Model: Local Fork, Not Global Low

**Wrong (v1)**: "Division requires mean(C) < threshold"

**Right**: Division requires:
1. **Locally low C** at the fork bond (to break it)
2. **Locally high C** at template bonds (to preserve pattern integrity)

```
Fork bond:      C_ij < C_fork_max (0.3)     → can break
Template bonds: C_im > C_template_min (0.5) → preserve pattern
```

This matches DNA: the replication fork is locally unwound, but the rest of the molecule maintains structure.

---

## 6. Boundary Role: Catalyst, Not Agency Donor

Boundaries may:
- ✅ Supply resource (like ATP in cellular environment)
- ✅ Heal coherence locally (like chaperone proteins)
- ✅ Create favorable conditions for division

Boundaries may NOT:
- ❌ Set or increase agency of any node
- ❌ Choose division sites globally
- ❌ Force recruitment against node's will

The boundary is a **catalyst**, not a creator. It helps division happen where nodes locally consent.

---

## 7. Answering the Primordial Questions

### Q: Was there a fixed amount of agency at the beginning?

**A**: Yes and no.
- **Yes**: The substrate has a fixed set of nodes, each with intrinsic agency
- **But**: Many are dormant (n=0) - potential, not actual participants
- Division activates dormant nodes, bringing their existing agency into participation

The total "potential agency" (sum over all nodes, including dormant) is fixed.
The "actual agency" (sum over active nodes) can increase through recruitment.

### Q: Where does new agency "come from" in division?

**A**: It doesn't "come from" anywhere. The recruit always had its agency.

The question assumes agency is created during division. It's not. Division changes **participation state**, not agency:

```
Before: Node k dormant (n=0), has intrinsic a_k = 0.4
After:  Node k active (n=1), still has a_k = 0.4
```

### Q: How does locality work?

**A**: Each fork step touches only:
- The parent node i
- The fork bond neighbor j
- The recruited dormant node k (in N_R(i))

No global operations. Division propagates like a zipper, one bond at a time.

---

## 8. Mathematical Formalization

### 8.1 Substrate Definition

```
S = (N, E, {n_i}, {a_i}, {F_i}, {q_i}, {C_ij})

N:     Fixed set of nodes
E:     Bond set (can change)
n_i:   Participation state ∈ {0, 1}
a_i:   Intrinsic agency (constant)
F_i:   Resource (dynamic)
q_i:   Structural debt (dynamic)
C_ij:  Coherence (dynamic)
```

### 8.2 Division Eligibility (Local Check)

```
eligible(i) = (a_i ≥ a_min^div) ∧
              (F_i ≥ F_min^div) ∧
              (∃j: C_ij < C_fork_max) ∧
              (∃k ∈ N_R(i): n_k=0 ∧ a_k ≥ a_min^join)
```

### 8.3 Fork Operator (Local Update)

```
Fork(i, j, k):
  Pre:  eligible(i), n_k = 0

  // Break fork bond
  F_i ← F_i - κ_break × C_ij
  C_ij ← ε (small)

  // Recruit
  n_k ← 1

  // Form new bond
  F_i ← F_i - κ_form × C_init
  C_ik ← C_init
  E ← E ∪ {(i,k)}

  // Pattern transfer (not agency)
  θ_k ← θ_k + 0.1 × (θ_i - θ_k)
  records_k ← {parent: i, ...}
```

---

## 9. Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Division agency min | a_min^div | 0.2 | Parent must exceed |
| Join agency min | a_min^join | 0.1 | Recruit must exceed |
| Break cost | κ_break | 0.1 | Per unit C broken |
| Form cost | κ_form | 0.08 | Per unit C formed |
| Fork C threshold | C_fork_max | 0.3 | Max C for breakable bond |
| Template C threshold | C_template_min | 0.5 | Min C for template bond |
| Initial coherence | C_init | 0.15 | New bond coherence |

---

## 10. Comparison: v1 (Wrong) vs v2 (Correct)

| Aspect | v1 (Violated DET) | v2 (DET-Compliant) |
|--------|-------------------|---------------------|
| Node creation | Create new object | Recruit dormant node |
| Agency handling | Copy from parent | Unchanged (intrinsic) |
| Coherence condition | Global mean < threshold | Local fork C low, template C high |
| Structure inheritance | Ad-hoc factor | Emerges from q-locking |
| Boundary role | "Grace" agency donor | Catalyst, no agency modification |
| Global operations | Caps, counts | None - all local |

---

## 11. Biological Interpretation

| DET Concept | DNA Analog |
|-------------|------------|
| Active node | Expressed gene |
| Dormant node | Silenced gene / nucleotide pool |
| Recruitment | Nucleotide incorporation |
| Fork bond breaking | Helicase unwinding |
| Pattern transfer | Template copying (sequence) |
| Intrinsic agency | Chemical properties of nucleotides |
| Mutual consent | Base pairing rules (A-T, G-C) |

The key biological insight: DNA replication doesn't create nucleotides. It recruits them from the cellular pool into a specific pattern. The nucleotides' chemistry (their "agency") was always there.

---

## 12. Conclusion

DET-compliant division is **recruitment, not creation**:

1. Nodes exist in substrate (active or dormant)
2. Each has intrinsic agency (inviolable)
3. Division activates dormant nodes through mutual consent
4. Pattern (not agency) transfers through bonds
5. All operations are local (fork model)
6. Costs emerge from existing dynamics (no ad-hoc rules)

This resolves the "primordial agency" question: **The agency was always there, waiting to participate.**

---

## Appendix: Simulation Results

```
Node 0 (parent):  a=0.70, F=2.00 → F=1.97 after division
Node 2 (recruit): a=0.40 (unchanged!), n: 0→1
Resource spent: 0.032
Agency NOT copied - recruit kept its intrinsic a=0.40
```

The recruit's agency (0.40) is completely independent of the parent's (0.70). Division enabled participation, not agency transfer.
