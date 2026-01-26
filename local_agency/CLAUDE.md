# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DET Local Agency is a DET-native operating system where all logic runs as Existence-Lang creatures executed via the EIS (Existence-Informed Substrate) virtual machine. The only non-EL code is the substrate layer (C/Metal for performance) and Python for external integrations.

## Build & Run Commands

### Python Environment Setup
```bash
cd local_agency
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run DET-OS
```bash
cd local_agency/src/python
python det_os_boot.py          # Standard mode
python det_os_boot.py --gpu    # GPU acceleration
python det_os_boot.py -v       # Verbose output
```

### Build C Libraries

**EIS Substrate (src/substrate/):**
```bash
cd local_agency/src/substrate
mkdir -p build && cd build
cmake ..
make
./test_substrate_v2    # Run tests
```

**Inference Library (src/inference/):**
```bash
cd local_agency/src/inference
mkdir -p build && cd build
cmake ..
make
ctest                  # Run all tests
./test_tensor          # Tensor ops tests
./test_gguf            # GGUF loading tests
```

### Run Python Tests
```bash
cd local_agency/src/python
python test_det_os_cli.py
```

## Architecture

```
User Terminal
      │
      ▼
TerminalCreature.ex ──bond──► LLMCreature.ex
      │                              │
      ▼                        ┌─────┼─────┐
EIS Interpreter (Python)       ▼     ▼     ▼
      │                     Memory  Tool  Reasoner
      ▼                       .ex    .ex    .ex
Substrate Layer (C/Metal)
```

### Key Concepts

- **Creatures**: Self-contained entities in Existence-Lang (.ex files) with F (resource), a (agency), q (structure)
- **Bonds**: Communication channels between creatures - messages flow through bonds, not direct calls
- **Phases**: Every execution tick runs READ → PROPOSE → CHOOSE → COMMIT
- **Primitives**: Substrate-level operations (llm_call, exec, file_*, etc.) implemented in Python/C

### Module Structure

| Directory | Purpose |
|-----------|---------|
| `src/python/det_os_boot.py` | Main entry point |
| `src/python/det/lang/` | Existence-Lang compiler (parser, tokenizer, bytecode) |
| `src/python/det/eis/` | EIS virtual machine (vm.py, primitives.py, phases.py) |
| `src/python/det/os/` | DET-OS kernel and creature management |
| `src/python/det/metal.py` | Metal GPU backend |
| `src/python/det/inference.py` | Native model inference Python bindings |
| `src/existence/` | Creature definitions (.ex files) |
| `src/substrate/` | C substrate layer with CMake build |
| `src/inference/` | Native inference library (Phase 26) |

### Creature Files (src/existence/)

- `terminal.ex` - REPL interface with 9 kernels
- `llm.ex` - LLM reasoning with native inference support
- `tool.ex` - Safe shell command execution
- `memory.ex` - Persistent storage
- `reasoner.ex` - Chain-of-thought reasoning
- `planner.ex` - Task planning
- `calculator.ex` - Math expression evaluation
- `physics.ex` - DET physics operators (presence, flow, agency)
- `inference.ex` - Native model inference creatures

## Design Principles

1. **DET First**: All behavior emerges from DET physics (F, a, P, bonds)
2. **Existence-Lang Native**: All creature logic in EL, not Python
3. **Substrate is Infrastructure**: Only primitives and execution in Python/C/Metal
4. **Bonds are Communication**: No direct method calls between creatures
5. **Phases are Atomic**: READ→PROPOSE→CHOOSE→COMMIT is the execution unit
6. **Resources are Real**: Every action costs F, tracked honestly

## Dependencies

- Python 3.10+
- Ollama (for LLM models, optional if using native inference)
- macOS with Apple Silicon (for Metal GPU acceleration, optional)
- CMake 3.10+ (for building C libraries)

## Performance Notes

- **Precompiled bytecode is faster**: Creatures with precompiled `.exb` files load significantly faster than JIT compilation from `.ex` source. The bytecode cache is in `src/python/det/lang/bytecode_cache.py`.

## LLM Models

Download via Ollama:
```bash
ollama serve              # Start Ollama server (separate terminal)
ollama pull llama3.2:3b   # Download model
```

Or use native inference with GGUF files (Phase 26).
