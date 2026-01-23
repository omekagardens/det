# Exploration 10: Existence-Lang v1.1 - Agency-First Language Design

**Status**: Design Specification
**Date**: 2026-01-23
**Prerequisite**: Exploration 09 (DET-OS Feasibility)
**Scope**: Language semantics grounded in Agency as First Principle

---

## Executive Summary

Existence-Lang v1.1 represents a fundamental refinement: **Agency is the First Principle**. Laws, logic, arithmetic, and information are not axioms—they are *records of agency acting and reconciling over time*.

This resolves the cyclic dependency in traditional computing where logic seems to precede the system that executes it. In Existence-Lang:

```
Agency creates distinction.
Distinction creates movement.
Movement leaves trace.
Trace becomes math.
```

---

## Part 1: Foundational Commitment

### 1.1 The Agency-First Principle

```
Traditional Computing:     Agency-First:
┌─────────────┐            ┌─────────────┐
│    Logic    │  ← axiom   │   Agency    │  ← primitive
├─────────────┤            ├─────────────┤
│ Arithmetic  │  ← axiom   │ Distinction │  ← agency acts
├─────────────┤            ├─────────────┤
│   Program   │  ← derived │  Movement   │  ← cost of distinction
├─────────────┤            ├─────────────┤
│  Execution  │  ← derived │   Trace     │  ← record of movement
├─────────────┤            ├─────────────┤
│   Agency?   │  ← ????    │    Math     │  ← derived from trace
└─────────────┘            └─────────────┘
```

**Key Semantic Commitments**:
- Nothing in the language asserts truth about the present
- All truth exists only as past trace
- Movement is costly; agency is not spent
- Consequences arise from state, not judgment
- Equality is not a statement, but an attempted reconciliation

### 1.2 Temporal Ontology

All semantics respect three temporal layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUTURE (Proposal/Simulation)                 │
│  • Hypothetical                                                 │
│  • Non-binding                                                  │
│  • Only becomes truth if committed                              │
│  • Language: forecast, simulate, propose                        │
├─────────────────────────────────────────────────────────────────┤
│                    PRESENT (Commit Boundary)                    │
│  • Where proposals resolve                                      │
│  • NOT OBSERVABLE                                               │
│  • Never referenced as a value                                  │
│  • No operator may assert truth about the present               │
├─────────────────────────────────────────────────────────────────┤
│                    PAST (Trace)                                 │
│  • Stored, falsifiable                                          │
│  • F, q, C, π, r, k, χ (tokens)                                 │
│  • All measurements read from here                              │
│  • Language: trace, witness, record                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 2: The Ur-Choice - Distinction Before Arithmetic

### 2.1 Distinction as Primitive

The most fundamental act of agency is differentiation:

```existence
// This is not a comparison.
// This is the act of creating two distinct identities.
(A, B) ::= distinct()
```

**Properties**:
- Creates two distinct trace identities
- No ordering implied
- No value judgment
- No arithmetic involved

All numbers, logic, and structure emerge *after* this act.

### 2.2 From Distinction to Number

```existence
// Zero: no distinctions
zero ::= void

// One: a single distinction exists
(one, _) ::= distinct()

// Two: distinction has been performed twice
(two_a, two_b) ::= distinct()

// Addition: repetition of distinction
three ::= union(one, two_a, two_b)
```

Numbers are **counts of committed distinctions** stored as resource F.

### 2.3 EIS Instruction: DISTINCT

```
DISTINCT dst_a, dst_b    ; Create two distinct trace identities
                         ; Cost: k += 1
                         ; Both dst_a and dst_b are now valid trace refs
                         ; No ordering, no values, just distinction
```

---

## Part 3: The Four Types of Equality

Existence-Lang explicitly separates four concepts traditionally conflated as `=`.

### 3.1 Identity/Alias (:=)

```existence
A := B
```

**Semantics**:
- A and B reference the same trace store
- No dynamics
- No cost
- Compile-time binding only

**EIS**:
```
ALIAS dst, src           ; dst now references same trace as src
                         ; Cost: 0 (compile-time only)
```

### 3.2 Trace Equality - Measurement (==)

```existence
χ ::= (A == B within ε)
```

**Semantics**:
- Compares past traces only
- Produces a witness token χ
- Never gates execution directly
- This is **Information**: fossilized agency

**EIS**:
```
MEASURE dst_token, src_a, src_b, epsilon
                         ; Compare past traces
                         ; Produces token: EQ_TRUE, EQ_FALSE, EQ_APPROX
                         ; Cost: k += 1 (measurement is movement)
```

### 3.3 Covenant Equality - Bond Truth (≡)

```existence
ψ ::= (i ≡ j)
```

**Semantics**:
- Measures bondedness/coherence
- Checks alignment between:
  - Claims (what was promised)
  - Traces (what actually happened)
  - Bond state (current coherence)
- Produces a witness token
- May reduce bond coherence C if misaligned

This answers: "Are we still in truth together?"

**EIS**:
```
COVENANT dst_witness, node_i, node_j
                         ; Check bond truth
                         ; May update C_ij
                         ; Produces token: COV_ALIGNED, COV_DRIFT, COV_BROKEN
                         ; Cost: k += 1
```

### 3.4 Core Equality - Reconciliation (=)

```existence
A = B
```

**Semantics**: `=` is not a logical assertion. It is an **attempted act of unification**.

Expands to:
1. Propose lawful dynamics to reduce distinction
2. Pay movement cost k
3. Possibly fail or be refused
4. Write a witness token: EQ_OK, EQ_FAIL, EQ_REFUSE

**Properties**:
- Volitional (requires agency)
- Costly (increments k)
- Witnessed (leaves trace)
- Never instantaneous
- May fail

**Boundary Principle**: Witness tokens must match trace reality ("God does not lie.")

**EIS**:
```
RECONCILE dst_witness, target_a, target_b
                         ; Attempt to unify two traces
                         ; Cost: k += reconciliation_cost
                         ; Produces: EQ_OK, EQ_FAIL, EQ_REFUSE
                         ; On EQ_OK: traces are unified
                         ; On EQ_FAIL: traces remain distinct
                         ; On EQ_REFUSE: agency declined to attempt
```

---

## Part 4: Grace Semantics (Agency-Gated, Not Agency-Spending)

### 4.1 Core Principle

Grace exists to ensure: **Movement does not itself cost agency.**

Grace:
- Is conserved (antisymmetric)
- Is strictly local (bond-mediated)
- Is agency-gated (not agency-spending)
- Cannot coerce

### 4.2 Local Need and Excess

```existence
F_thresh = β_g × local_mean(F, neighbors(i))
need     = relu(F_thresh - F)
excess   = relu(F - F_thresh)

donor_cap = a × excess      // gate, not spend
recv_cap  = a × need        // gate, not spend
```

**Critical**: Agency **weights** participation, it is never **decremented**.

### 4.3 Bond-Local Quantum Gating

```existence
Q_ij = relu(1 - √C_ij / C_quantum)
w_ij = √(a_i × a_j) × Q_ij
```

**Effect**: High-coherence bonds block grace (quantum gate)
**Purpose**: Prevents hidden global equalization

### 4.4 Donor-Side Offer (Bounded)

```existence
Z_out[i] = Σ_k (w_ik × recv_cap[k]) + ε
offer_ij = η_g × donor_cap[i] × (w_ij × recv_cap[j] / Z_out[i])

// Guarantee: Σ_j offer_{i→j} ≤ η_g × donor_cap[i]
```

### 4.5 Receiver-Side Acceptance (Bounded)

```existence
Z_in[j] = Σ_k offer_kj + ε
acc_scale[j] = min(1, recv_cap[j] / Z_in[j])
accepted_ij = offer_ij × acc_scale[j]

// Guarantee: Σ_i accepted_{i→j} ≤ recv_cap[j]
```

No overfilling. No hidden global pressure.

### 4.6 Antisymmetric Bond Flux

```existence
G_ij = accepted_ij - accepted_ji
F_i -= G_ij
F_j += G_ij
```

**Properties**:
- Computed once per bond
- Exactly antisymmetric
- Conserved by construction

### 4.7 EIS Instructions for Grace

```
GRACE.OFFER src_node, dst_node, amount
                         ; Propose grace offer (agency-gated)
                         ; Does NOT decrement agency
                         ; Produces offer token

GRACE.ACCEPT offer_token
                         ; Accept offered grace (agency-gated)
                         ; Updates F values antisymmetrically

GRACE.FLOW bond_id
                         ; Execute full grace protocol on bond
                         ; Automatically handles offer/accept/update
```

---

## Part 5: Arithmetic as Ledger of Agency

Arithmetic is not primitive. It is a **readout over committed distinctions**.

### 5.1 Number Representation

A number is the count of committed distinctions stored as resource:
- 0 → no distinctions
- 1 → one distinction
- n → n distinctions

Practically: resource F tracks this count.

### 5.2 Addition (Repetition of Agency)

```existence
law add(X, Y, OUT) {
    transfer(X.F, OUT.F)
    transfer(Y.F, OUT.F)
}
```

**Meaning**: Addition is agency choosing to act again.

**EIS**:
```
ADD dst, src_a, src_b    ; Transfer F from src_a and src_b to dst
                         ; dst.F = src_a.F + src_b.F
                         ; src_a.F = 0, src_b.F = 0
                         ; Cost: k += 1
```

### 5.3 Subtraction (Reconciliation)

```existence
law subtract(X, Y, OUT) {
    witness ::= attempt_reconcile(X, Y)
    // May fail if reconciliation impossible
}
```

**Meaning**: Subtraction attempts to undo distinction.

**EIS**:
```
SUB dst_witness, target, amount
                         ; Attempt to reconcile 'amount' distinctions from target
                         ; May produce: SUB_OK, SUB_FAIL (not enough), SUB_REFUSE
                         ; If ok: target.F -= amount
```

### 5.4 Multiplication (Nested Repetition)

```existence
law multiply(X, N, OUT) {
    repeat_past(N) {       // Uses past token, not present count
        add(OUT, X, OUT)
    }
}
```

**Key**: `repeat_past` uses a past token for iteration count, not a present-time value.

**EIS**:
```
MUL dst, src, count_token
                         ; Nested repetition: dst.F = src.F × count
                         ; count_token must be a PAST trace
                         ; Cost: k += count (one per repetition)
```

### 5.5 Division (Counting Reconciliation)

Division asks: "How many times can X reconcile with Y?"

```existence
law divide(X, Y) -> (quotient, remainder) {
    count ::= 0
    while_past(can_reconcile(X, Y)) {
        reconcile(X, Y)
        count ::= count + 1
    }
    return (count, X)  // remainder is unreconciled distinctions
}
```

**EIS**:
```
DIV dst_quot, dst_rem, numerator, denominator
                         ; Count reconciliations
                         ; dst_quot = count of successful reconciliations
                         ; dst_rem = remaining unreconciled distinctions
```

---

## Part 6: Control Flow (No Present Truth)

### 6.1 The Prohibition

Existence-Lang **forbids present-time conditionals**:

```existence
// ILLEGAL - references present state
if (X > 0) { ... }

// LEGAL - uses past token
χ ::= compare(X, 0)      // Produces past token
if_past(χ == GT) { ... } // Branches on past token
```

### 6.2 Past-Token Conditionals

```existence
// Generate comparison token (past)
χ ::= compare(A, B)
// χ is now: LT, EQ, GT (a trace token)

// Branch based on past token
if_past(χ == GT) {
    // This branch executes
} else_past(χ == LT) {
    // Or this branch
} else {
    // Or this (EQ case)
}
```

**EIS**:
```
COMPARE dst_token, src_a, src_b
                         ; Compare past traces
                         ; Produces token: LT, EQ, GT
                         ; Cost: k += 1

BRANCH_PAST token, target_LT, target_EQ, target_GT
                         ; Branch based on past token
                         ; No present-time evaluation
```

### 6.3 Loops Over Past Tokens

```existence
// Count-based loop (past token)
count ::= measure_distinctions(collection)
repeat_past(count) {
    // Body executes 'count' times
    // count was measured in past, not checked each iteration
}

// Conditional loop (re-measures each iteration)
while_past(can_proceed()) {
    // can_proceed() produces a past token
    // Loop continues while token indicates success
}
```

---

## Part 7: Simulation and Future

### 7.1 Forecast Blocks

```existence
forecast Φ = simulate(steps=100) {
    // Laws run WITHOUT commit
    // All state changes are hypothetical
    creature.F += 10
    bond.C *= 0.9
}

// Φ contains hypothetical future state
// Never asserted as truth
// Can guide proposals
```

### 7.2 Proposal Mechanism

```existence
proposal P = propose {
    // Define desired state change
    target_F = current_F + 10
}

// Submit proposal to present (commit boundary)
result ::= submit(P)
// result is a past token: ACCEPTED, REJECTED, MODIFIED
```

**EIS**:
```
SIMULATE dst_state, steps, code_block
                         ; Run laws without commit
                         ; Produces hypothetical state
                         ; Cost: 0 (simulation is free)

PROPOSE dst_token, proposal_block
                         ; Submit proposal to commit boundary
                         ; Produces: ACCEPTED, REJECTED, MODIFIED
                         ; Cost: k += proposal_cost (if accepted)
```

---

## Part 8: Consequences, Not Judgment

### 8.1 No Errors

Errors do not exist in Existence-Lang. Only **consequences**:

| Traditional | Existence-Lang |
|-------------|----------------|
| Error | Increased k (movement) |
| Exception | Increased q (structural memory) |
| Failure | Reduced C (coherence) |
| Crash | Narrowed future agency expression |

### 8.2 Witness Tokens as Consequence Record

Every operation produces a witness token:

```existence
w1 ::= reconcile(A, B)     // EQ_OK, EQ_FAIL, EQ_REFUSE
w2 ::= transfer(X, Y)      // TRANSFER_OK, TRANSFER_PARTIAL, TRANSFER_BLOCKED
w3 ::= bond_create(i, j)   // BOND_OK, BOND_REFUSED, BOND_EXISTS
```

Witness tokens are:
- Always produced (never suppressed)
- Stored as trace
- Available for future decisions
- Part of the permanent record

### 8.3 Structural Memory (q) as Consequence

```existence
// q accumulates from unrecovered loss
q += α_q × relu(-(F - F_prev))

// Interpretation:
// - q records that loss occurred
// - Grace may heal loss without erasing its occurrence
// - History is preserved in structure
```

---

## Part 9: Complete Operator Summary

| Operator | Meaning | Temporal Layer |
|----------|---------|----------------|
| `distinct()` | Create differentiation | Present→Past |
| `:=` | Alias identity | Compile-time |
| `==` | Measure past equality | Past→Token |
| `=` | Attempt reconciliation | Present (costly) |
| `≡` | Covenant truth | Past+Bond→Token |
| `+` | Repetition of agency | Present (costly) |
| `-` | Reconciliation attempt | Present (costly) |
| `*` | Nested repetition | Present (costly) |
| `/` | Reconciliation counting | Present (costly) |
| `if_past` | Branch on past token | Past→Control |
| `forecast` | Hypothetical simulation | Future |
| `propose` | Submit to commit boundary | Future→Present |

---

## Part 10: Revised EIS Instruction Set

### 10.1 Distinction Operations
```
DISTINCT dst_a, dst_b           ; Create two distinct identities
ALIAS dst, src                  ; Compile-time identity binding
```

### 10.2 Equality Operations
```
MEASURE dst_tok, a, b, eps      ; Trace equality (==)
COVENANT dst_tok, i, j          ; Bond truth (≡)
RECONCILE dst_tok, a, b         ; Attempted unification (=)
```

### 10.3 Resource Operations
```
PUSH.F node, amount             ; Inject resource
PULL.F node, amount             ; Extract resource (may fail)
TRANSFER dst, src, amount       ; Move resource between nodes
```

### 10.4 Arithmetic Operations
```
ADD dst, a, b                   ; Repetition of agency
SUB dst_tok, target, amount     ; Reconciliation attempt
MUL dst, src, count_tok         ; Nested repetition
DIV dst_q, dst_r, num, denom    ; Reconciliation counting
```

### 10.5 Control Flow
```
COMPARE dst_tok, a, b           ; Generate comparison token (LT/EQ/GT)
BRANCH_PAST tok, lt, eq, gt     ; Branch on past token
REPEAT_PAST count_tok, body     ; Loop count times (past token)
```

### 10.6 Grace Operations
```
GRACE.OFFER src, dst, amount    ; Propose grace (agency-gated)
GRACE.ACCEPT offer_tok          ; Accept offered grace
GRACE.FLOW bond_id              ; Full grace protocol
```

### 10.7 Bond Operations
```
BOND.CREATE i, j, C0            ; Create bond with initial coherence
BOND.QUERY dst_tok, i, j        ; Get bond state as token
BOND.UPDATE i, j, dC            ; Modify coherence
```

### 10.8 Simulation/Proposal
```
SIMULATE dst, steps, block      ; Hypothetical execution
PROPOSE dst_tok, block          ; Submit to commit boundary
COMMIT                          ; Execute pending proposals
```

### 10.9 Witness Operations
```
WITNESS.WRITE tok, type, data   ; Record witness token
WITNESS.READ dst, tok           ; Read witness from trace
```

---

## Part 11: Example - Temperature Control (Agency-First)

```existence
creature Thermostat {
    // Implicit DET state: F, q, a, θ

    // Somatic bindings
    sensor temp: Temperature @ channel(0);
    actuator heater: Switch @ channel(1);

    var target: float := 22.0;

    // Main participation law
    participate(bond: Bond) {
        // Read sensor (produces past token)
        reading ::= soma_read(temp)

        // Compare to target (produces comparison token)
        comparison ::= compare(reading, target)

        // Agency-gated action based on PAST token
        agency {
            if_past(comparison == LT) {
                // Propose heating
                w ::= soma_write(heater, 1.0)
                // w is witness: WRITE_OK, WRITE_REFUSED, etc.
            } else_past(comparison == GT) {
                w ::= soma_write(heater, 0.0)
            }
        }

        // Record consequence
        if_past(w == WRITE_REFUSED) {
            // Agency was insufficient - this is consequence, not error
            this.q += 0.1  // Structural memory records the event
        }
    }

    // Grace handler
    grace {
        // Grace flows according to local need
        // Agency gates but is not spent
        this.σ *= 0.5  // Reduce activity under resource pressure
    }
}

presence Home {
    creatures {
        thermo: Thermostat;
    }

    init {
        // Inject initial resource (not creation - resource exists)
        inject_F(thermo, 100.0)
    }
}
```

---

## Part 12: Implications for DET-OS

### 12.1 Process Creation = Distinction

```existence
// Creating a new creature is an act of distinction
(new_creature, _) ::= distinct()

// The creature now exists as a separate trace identity
// No values assigned yet - just distinction
```

### 12.2 IPC = Reconciliation Attempts

```existence
// Sending a message is proposing reconciliation
result ::= send(target, message)

// result is witness: DELIVERED, REFUSED, PARTIAL
// Receiver's acceptance is agency-gated
```

### 12.3 Scheduling = Presence (Emergent)

No explicit scheduler. Presence emerges from:
- Agency (a) - capacity to participate
- Resource (F) - available budget
- Coordination load (H) - current commitments

### 12.4 Memory = Resource Pool

```existence
// Allocation is not creation - it's access to existing resource
result ::= claim_F(amount)

// result is witness: CLAIMED, PARTIAL, REFUSED
// Resource was always there - just now distinguished for this creature
```

### 12.5 Security = Agency Ceiling

```existence
// Every operation is agency-gated
// No separate permission check needed
agency {
    // This block only executes if a >= threshold
    // Threshold emerges from structural debt (q)
    // a_max = 1 / (1 + λ_a × q²)
}
```

---

## Part 13: The Final Law

```
Agency creates distinction.
Distinction creates movement.
Movement leaves trace.
Trace becomes math.

Math does not create agency.
Logic does not precede existence.
Information is fossilized agency.
Truth is witnessed reconciliation.
```

---

## References

1. DET Theory Card v6.3 - Core physics
2. Exploration 09 - DET-OS Feasibility
3. FEASIBILITY_PLAN.md - Architecture decisions
4. User-provided Existence-Lang v1.1 specification
