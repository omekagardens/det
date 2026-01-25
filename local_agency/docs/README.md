# DET Local Agency Documentation

## Overview

DET Local Agency is a DET-native operating system where all logic runs as **Existence-Lang creatures** executed via the EIS (Existence-Informed Substrate) virtual machine.

## Core Documentation

| Document | Description |
|----------|-------------|
| [EXISTENCE_LANG.md](EXISTENCE_LANG.md) | Existence-Lang language reference |
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | System architecture with EIS VM and creature model |
| [SUBSTRATE_SPEC.md](SUBSTRATE_SPEC.md) | Substrate v2 specification (phases, effects, opcodes) |
| [ROADMAP_V2.md](ROADMAP_V2.md) | Project roadmap and completed phases |

## Root-Level Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](../GETTING_STARTED.md) | Quick start guide |
| [DEVELOPMENT_LOG.md](../DEVELOPMENT_LOG.md) | Development history |
| [FEASIBILITY_PLAN.md](../FEASIBILITY_PLAN.md) | Original technical specification |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 User Terminal                    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           Existence-Lang Creatures               │
│  terminal.ex, llm.ex, tool.ex, memory.ex, ...   │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              EIS Interpreter                     │
│    (READ → PROPOSE → CHOOSE → COMMIT)           │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           Substrate Layer (C/Metal)              │
│  Primitives, state management, GPU acceleration  │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd local_agency/src/python
python det_os_boot.py        # Standard mode
python det_os_boot.py --gpu  # GPU acceleration
python det_os_boot.py -v     # Verbose output
```

## Key Concepts

### Creatures
Self-contained entities written in Existence-Lang (.ex files) that communicate via bonds. Each creature has:
- **F** (Resource): Available computational budget
- **a** (Agency): Autonomy level [0,1]
- **Kernels**: Named entry points for execution

### Bonds
Communication channels between creatures. Messages flow through bonds, not direct method calls.

### Phases
Every tick executes four phases:
1. **READ**: Load state from trace
2. **PROPOSE**: Generate proposals with scores
3. **CHOOSE**: Select best proposal
4. **COMMIT**: Apply effects, emit witnesses

### Primitives
External I/O operations (LLM calls, file access, shell execution) with F cost tracking.

## Historical Documentation

Archived research and exploration documents are in `/archive/deprecated_docs/explorations/`:
- Node topology and layer dynamics
- Agency distribution models
- DET-OS feasibility studies
- Existence-Lang v1.1 specification

---

*Last Updated: 2026-01-24*
