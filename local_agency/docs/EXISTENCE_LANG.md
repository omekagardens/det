# Existence-Lang Reference

**Version**: 1.1
**Last Updated**: 2026-01-24

Existence-Lang is an agency-first programming language for DET-OS. All logic runs as creatures with phase-based execution.

---

## Core Philosophy

**Agency is the First Principle.** Laws, logic, and arithmetic are not axioms—they are records of agency acting over time.

```
Agency creates distinction.
Distinction creates movement.
Movement leaves trace.
Trace becomes math.
```

### Temporal Ontology

All semantics respect three layers:

| Layer | Description | Language |
|-------|-------------|----------|
| **FUTURE** | Proposals, hypothetical | `propose`, `forecast` |
| **PRESENT** | Commit boundary, not observable | Phase execution |
| **PAST** | Trace, stored, measurable | `witness`, `trace` |

---

## Creatures

Creatures are self-contained entities that communicate via bonds.

```existence
creature CalculatorCreature {
    var result: Register = 0.0;
    var last_expr: TokenReg = "";

    kernel Eval {
        in expr: TokenReg;
        out result: TokenReg;

        phase READ {
            input_expr ::= witness(expr);
        }

        phase PROPOSE {
            proposal EvalExpr {
                score = 1.0;
                effect {
                    result ::= primitive("eval_math", expr);
                }
            }
        }

        phase CHOOSE {
            choice := choose({EvalExpr});
        }

        phase COMMIT {
            commit choice;
        }
    }
}
```

### Creature Fields

| Field | Type | Description |
|-------|------|-------------|
| `F` | Register | Resource budget (0 to ∞) |
| `a` | Register | Agency level [0, 1] |
| `q` | Register | Structural debt (memory of loss) |
| Custom | Register/TokenReg | User-defined state |

---

## Types

### Register
Scalar numeric value (float).

```existence
var x: Register = 42.0;
var temperature: Register = 0.5;
```

### TokenReg
String or token value.

```existence
var message: TokenReg = "Hello";
var response: TokenReg = "";
```

### NodeRef / BondRef
References to nodes or bonds in the substrate.

```existence
var self_node: NodeRef = self;
var partner: BondRef = bond(0);
```

---

## Kernels

Kernels are named entry points for creature execution.

```existence
kernel ProcessMessage {
    in message: TokenReg;
    out response: TokenReg;

    phase READ { ... }
    phase PROPOSE { ... }
    phase CHOOSE { ... }
    phase COMMIT { ... }
}
```

### Kernel Parameters

| Direction | Description |
|-----------|-------------|
| `in` | Input parameter (read-only) |
| `out` | Output parameter (write in COMMIT) |
| `inout` | Bidirectional |

---

## Phases

Every kernel executes four phases in order:

### READ
Load state from trace. No mutations allowed.

```existence
phase READ {
    current_f ::= witness(self.F);
    input_val ::= witness(input);
}
```

### PROPOSE
Generate proposals with scores. Higher score = preferred.

```existence
phase PROPOSE {
    proposal OptionA {
        score = 0.8;
        effect {
            output = compute_a(input);
        }
    }

    proposal OptionB {
        score = 0.5;
        effect {
            output = compute_b(input);
        }
    }
}
```

### CHOOSE
Select among proposals deterministically.

```existence
phase CHOOSE {
    choice := choose({OptionA, OptionB});
}
```

### COMMIT
Apply effects from chosen proposal.

```existence
phase COMMIT {
    commit choice;
    witness(output);
}
```

---

## Operators

### Assignment Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `:=` | Alias (compile-time) | `A := B` |
| `::=` | Witness assignment | `x ::= witness(y)` |
| `=` | Value assignment | `x = y + 1` |

### Arithmetic

| Operator | Meaning | Cost |
|----------|---------|------|
| `+` | Addition | k += 1 |
| `-` | Subtraction | k += 1 |
| `*` | Multiplication | k += 1 |
| `/` | Division | k += 1 |
| `**` | Exponentiation | k += 1 |

### Comparison

```existence
// Comparison produces past token
χ ::= compare(A, B);  // LT, EQ, GT

if_past(χ == GT) {
    // A > B
} else_past(χ == LT) {
    // A < B
} else {
    // A == B
}
```

---

## Primitives

External operations with F cost tracking.

```existence
// Call a primitive
result ::= primitive("llm_call", prompt);
output ::= primitive("exec_safe", command);
content ::= primitive("file_read", path);
```

### Available Primitives

| Primitive | Description | Base Cost |
|-----------|-------------|-----------|
| `llm_call` | LLM completion | 1.0 + tokens |
| `llm_chat` | Multi-turn chat | 1.0 + tokens |
| `exec_safe` | Safe shell command | 0.5 |
| `exec` | Full shell (agency > 0.8) | 1.0 |
| `file_read` | Read file contents | 0.2 |
| `file_write` | Write file | 0.5 |
| `file_exists` | Check file exists | 0.1 |
| `file_list` | List directory | 0.2 |
| `now` | Current timestamp | 0.0 |
| `now_iso` | ISO timestamp | 0.0 |
| `sleep` | Pause execution | 0.0 |
| `random` | Random float [0,1] | 0.0 |
| `random_int` | Random integer | 0.0 |
| `print` | Debug output | 0.0 |
| `log` | Log message | 0.0 |
| `eval_math` | Evaluate math expression | 0.1 |
| `terminal_read` | Read user input | 0.0 |
| `terminal_write` | Write output | 0.0 |

---

## Bonds

Communication channels between creatures.

```existence
// In kernel, send via bond
bond_send(channel_id, message);

// Receive from bond
received ::= bond_receive(channel_id);
```

### Bond Properties

| Property | Type | Description |
|----------|------|-------------|
| `C` | Register | Coherence [0, 1] |
| `node_i` | NodeRef | First endpoint |
| `node_j` | NodeRef | Second endpoint |

---

## Witness Tokens

Operations produce witness tokens as consequence records:

| Token | Meaning |
|-------|---------|
| `OK` | Operation succeeded |
| `FAIL` | Operation failed |
| `PARTIAL` | Partial success |
| `BLOCKED` | Resource insufficient |
| `REFUSED` | Agency insufficient |
| `LT`, `EQ`, `GT` | Comparison result |

```existence
w ::= transfer(src, dst, amount);
if_past(w == OK) {
    // Transfer succeeded
} else {
    // Handle consequence
}
```

---

## Complete Example

```existence
/**
 * LLMCreature - Language Model Interface
 */
creature LLMCreature {
    var last_prompt: TokenReg = "";
    var last_response: TokenReg = "";

    kernel Think {
        in prompt: TokenReg;
        out response: TokenReg;

        phase READ {
            input_prompt ::= witness(prompt);
            available_f ::= witness(self.F);
        }

        phase PROPOSE {
            proposal CallLLM {
                score = 1.0;
                effect {
                    response ::= primitive("llm_call", prompt);
                    last_prompt = prompt;
                    last_response = response;
                }
            }
        }

        phase CHOOSE {
            choice := choose({CallLLM});
        }

        phase COMMIT {
            commit choice;
        }
    }
}
```

---

## File Extension

Existence-Lang source files use `.ex` extension.
Compiled bytecode uses `.exb` extension.

```bash
# Compile to bytecode
python -m det.lang.excompile creature.ex

# Run in DET-OS
python det_os_boot.py
det> load creature
det> bond creature
det> send creature kernel_name args
```

---

## Theoretical Foundation

For the complete theoretical specification including:
- Agency-first semantics
- The four types of equality (`:=`, `==`, `=`, `≡`)
- Temporal ontology details
- Ur-choice and distinction

See: `/archive/deprecated_docs/explorations/10_existence_lang_v1_1.md`

---

*Last Updated: 2026-01-24*
