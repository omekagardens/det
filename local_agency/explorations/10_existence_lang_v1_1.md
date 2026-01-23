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

## Part 13: Kernels - Functions as Law Modules

### 13.1 What a Function Is (and Is Not)

In Existence-Lang, a "function" is not a value-returning procedure. It is a **named local law kernel** that produces proposals, trace tokens, and conserved flux commits.

**NOT Allowed (Classic Function)**:
- `f(x) -> y` computed "instantly"
- Implicit mutation
- Hidden state
- Present-time truth

**Allowed (DET Kernel)**:
- **Ports**: Register references, bond handles, token handles
- **Phase**: Which step of the canonical loop it runs in
- **Proposal set**: What it can do
- **Commit rules**: How it writes traces/fluxes
- **Witness output**: Tokenized success/failure

Think "hardware block" or "local circuit," not "subroutine."

### 13.2 Kernel Syntax

```existence
kernel AddSigned {
    // Input ports (read-only references)
    in  xP: Register;
    in  xM: Register;
    in  yP: Register;
    in  yM: Register;

    // Output ports (write destinations)
    out oP: Register;
    out oM: Register;
    out w:  TokenReg;        // Witness token

    // Local constants only (no mutable state)
    params { J_step = 0.01; }

    // Phase declaration (when this kernel runs)
    phase COMMIT {
        // Proposal set (no if/else - all proposals evaluated)
        proposal RUN {
            score = 1.0;
            effect {
                Pump(J_step).move(xP, oP);
                Pump(J_step).move(yP, oP);
                Pump(J_step).move(xM, oM);
                Pump(J_step).move(yM, oM);
            }
        }

        // Choose from proposals (agency-weighted, deterministic from seed)
        choice χ = choose(
            {RUN},
            decisiveness = 1.0,
            seed = local_seed(k, r, θ)
        );

        // Commit chosen proposal
        commit χ;

        // Write witness token
        w ::= "OK";
    }
}
```

**Key Rules**:
- Kernels do NOT return values
- Kernels write ONLY through declared outputs
- Kernels may take multiple ticks (convergent operators)
- Correctness is signaled with witness tokens

### 13.3 Calling a Kernel: Wiring, Not Stack Frames

A "call" is local instantiation and wiring, not a stack frame:

```existence
// This places the kernel in the local update graph
call AddSigned(
    xP: Aplus,
    xM: Aminus,
    yP: Bplus,
    yM: Bminus,
    oP: Outplus,
    oM: Outminus,
    w:  Wtok
)
```

No stack. No global calls. Just wiring into the update schedule.

---

## Part 14: Operator Lowering - Macros Over Kernels

### 14.1 Operators Are Not Primitives

Operators like `+`, `*`, `<=`, `=` are NOT primitives. They are **macros** that expand into kernel instantiations.

```existence
// Operator definition mechanism (compile-time)
operator (X + Y) lowers_to AddSigned(...)
operator (X * N) lowers_to MulSmallInt(...)
operator (A = B) lowers_to Reconcile(...)
```

This prevents cyclicity: operators are syntax sugar over kernels already defined from physics primitives.

### 14.2 The `+=` Operator

```existence
// X += Y is AddSigned with output wired back to input
operator (X += Y) lowers_to AddSigned(
    xP: X+,
    xM: X-,
    yP: Y+,
    yM: Y-,
    oP: X+,    // Output back to input
    oM: X-,
    w:  _      // Witness discarded
)
```

This is legal because it's still flux-based, not instantaneous assignment.

### 14.3 The `=` Operator (Reconciliation)

The reconciliation equality is explicitly a kernel:

```existence
kernel Reconcile {
    inout A: Register;
    inout B: Register;
    out   w: TokenReg;
    params { eps = 1e-3; J_step = 0.01; }

    phase COMMIT {
        // Proposal to repair (reduce distinction)
        proposal REPAIR {
            score = sqrt(a_local());     // Willingness, NOT spend
            effect {
                Diffuse(σ = 1.0).step(A, B);
            }
        }

        // Proposal to refuse (agency declines)
        proposal REFUSE {
            score = 1.0 - sqrt(a_local());
            effect { /* no flux */ }
        }

        // Choose based on agency and coherence
        choice χ = choose(
            {REPAIR, REFUSE},
            decisiveness = g(a_local(), coherence_local()),
            seed = local_seed(k, r, θ)
        );
        commit χ;

        // Witness based on trace measurement
        w ::= cmp(A.F, B.F, eps);  // EQ/NEQ token
    }
}
```

Now `=` is never "truth." It's "attempt unification" with a witness.

---

## Part 15: Minimal Primitive Set

To support functional composition, the compiler/runtime provides these primitives:

### 15.1 Tier-0 Physics Primitives

```
transfer(src.F, dst.F, J_quanta)    ; Antisymmetric, clamped
diffuse(i.F, j.F, σ)                ; Antisymmetric flux
choose(proposals, decisiveness, seed) ; Tokenization
commit(choice)                       ; Write trace
repeat_past(N)                       ; Bounded schedule from past token
cmp(x, y, ε) → token                ; Trace measurement
```

**Everything else is kernels/macros built from these.**

### 15.2 Primitive Classification

| Primitive | What It Does | Cost |
|-----------|--------------|------|
| `transfer` | Move quanta between registers | k += 1 |
| `diffuse` | Bond-local flux exchange | k += 1 |
| `choose` | Select from proposal set | k += 0 (decision) |
| `commit` | Write chosen proposal to trace | k += proposal_cost |
| `repeat_past` | Bounded iteration (past token count) | k += N |
| `cmp` | Measure past traces | k += 1 |

---

## Part 16: Convergent Kernels - Operations Over Time

Many DET operations aren't instantaneous. Kernels declare completion via witness tokens.

### 16.1 Convergent Kernel Pattern

Example: "copy" finishes when |X−Y| < ε:

```existence
kernel CopyConverge {
    in  src: Register;
    in  dst: Register;
    out w:   TokenReg;
    params { eps = 1e-3; J_step = 0.01; }

    phase COMMIT {
        // Push quanta from src toward dst
        proposal PUSH {
            score = 1.0;
            effect {
                Pump(J_step).move(src, dst);
            }
        }

        choice χ = choose({PUSH}, decisiveness = 1.0, seed = local_seed(k, r, θ));
        commit χ;

        // Witness from past trace (convergence check)
        w ::= cmp(dst.F, src.F, eps);  // EQ when converged
    }
}
```

**No while loops**. The caller watches `w` to know when complete.

### 16.2 Caller Watches Witness

```existence
creature Copier {
    call CopyConverge(src: source_reg, dst: dest_reg, w: copy_witness)

    // React to witness in NEXT tick (not same tick!)
    participate(bond: Bond) {
        if_past(copy_witness == EQ) {
            // Copy complete - proceed with next operation
        }
    }
}
```

---

## Part 17: Pipeline Composition

### 17.1 The `pipe` Construct

Wire kernels together without control flow:

```existence
pipe {
    call AddSigned(
        xP: Aplus, xM: Aminus,
        yP: Bplus, yM: Bminus,
        oP: Tplus, oM: Tminus,
        w: W1
    )
    call Clamp0_1(
        in: Tplus/Tminus,
        out: Uplus/Uminus,
        w: W2
    )
}
```

These are multiple kernels scheduled each tick, not sequential execution.

### 17.2 Function-Like Syntax Sugar

For ergonomics, support "return-like" syntax:

```existence
(z, w) <~ add(x, y)
```

Lowers to:

```existence
call AddSigned(
    xP: x+, xM: x-,
    yP: y+, yM: y-,
    oP: z+, oM: z-,
    w: w
)
```

The `<~` operator means "wire outputs to these destinations."

---

## Part 18: The Anti-Drift Rule

### 18.1 The One Rule That Prevents Returning to "C"

**A kernel may not read a token it just wrote in the same tick to decide whether to run.**

Tokens only influence the **next** tick.

```existence
// ILLEGAL - same-tick read of own witness
phase COMMIT {
    w ::= cmp(A.F, B.F, eps);
    if_past(w == EQ) { ... }  // NO! w was just written!
}

// LEGAL - react in next tick
phase COMMIT {
    w ::= cmp(A.F, B.F, eps);
}
// In creature's participate(), which runs NEXT tick:
if_past(w == EQ) { ... }  // OK - reading from past
```

### 18.2 Why This Matters

This enforces "present unfalsifiable" in code, not just philosophy:
- No instantaneous feedback loops
- No hidden present-time branching
- All decisions flow from past trace
- Time moves forward, never sideways

---

## Part 19: Sub-Primitive Operator Library

### 19.1 `+` Over Signed Values

```existence
kernel AddSigned {
    // ... (as defined above)
}

operator (X + Y) lowers_to AddSigned(
    xP: X.positive, xM: X.negative,
    yP: Y.positive, yM: Y.negative,
    oP: _.positive, oM: _.negative,
    w: _
)
```

### 19.2 `*` Multiplication (Nested Repetition)

Multiplication requires `repeat_past`:

```existence
kernel MulByPastToken {
    in  base: Register;
    in  count: TokenReg;   // Must be past token (integer)
    out result: Register;
    out w: TokenReg;

    phase COMMIT {
        repeat_past(count) {
            // Each iteration is a separate tick
            call AddSigned(
                xP: result+, xM: result-,
                yP: base+, yM: base-,
                oP: result+, oM: result-,
                w: _
            )
        }
        w ::= "OK";
    }
}
```

`*` is only definable when:
- The multiplier is a past token integer, OR
- A dedicated multiplier module exists (log-domain or bitwise)

### 19.3 `<=` Converge-To-Target

```existence
kernel ConvergeTo {
    inout X: Register;
    in    target: Register;
    out   w: TokenReg;
    params { J_step = 0.01; eps = 1e-3; }

    phase COMMIT {
        proposal ADJUST {
            score = 1.0;
            effect {
                Diffuse(σ = 1.0).step(X, target);
            }
        }

        choice χ = choose({ADJUST}, decisiveness = 1.0, seed = local_seed(k, r, θ));
        commit χ;

        w ::= cmp(X.F, target.F, eps);
    }
}

operator (X <= target) lowers_to ConvergeTo(X: X, target: target, w: _)
```

---

## Part 20: The Final Law

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
5. User-provided kernel/function design (this section)
