# DET Local Agency - Getting Started

Welcome to DET Local Agency, a DET-native operating system where all logic runs as Existence-Lang creatures executed via the EIS virtual machine.

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** - For running local LLM models
- **macOS** (optional) - For GPU acceleration via Metal

### Installation

```bash
# Clone the repository
git clone https://github.com/omekagardens/det.git
cd det/local_agency

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama serve

# Download required model
ollama pull llama3.2:3b
```

### Run DET-OS

```bash
cd src/python
python det_os_boot.py
```

You'll see the DET-OS terminal:

```
Bootstrapping DET-OS...
Bootstrap complete.

============================================================
DET-OS Terminal (Bond-Based Dispatch)
============================================================
Type 'help' for commands, 'quit' to exit.

det>
```

### Command-Line Options

```bash
python det_os_boot.py          # Standard mode
python det_os_boot.py --gpu    # Enable GPU acceleration
python det_os_boot.py -v       # Verbose output
```

## Basic Usage

### Talking to the LLM

Type any message to send it to the LLM creature:

```
det> What is the capital of France?
[Sending to LLM via bond...]
Paris is the capital of France.
```

### Running Commands

Use `run` for safe shell commands:

```
det> run ls -la
[Sending to Tool via bond...]
total 208
drwxr-xr-x   7 user  staff    224 Jan 24 17:37 .
...
```

### Calculator

Quick math with `calc`:

```
det> calc 2 + 3 * 4
2 + 3 * 4 = 14

det> calc sqrt(144) + sin(0)
sqrt(144) + sin(0) = 12.0
```

### Creature Management

```
det> list                    # Show available creatures
det> load calculator         # Load a creature
det> use memory             # Load and bond in one step
det> bond calculator llm    # Bond two creatures together
det> status                 # Show creature status
```

### GPU Acceleration

```
det> gpu                     # Show GPU status
det> gpu enable              # Enable GPU
det> gpu benchmark 1000      # Run benchmark with 1000 nodes
det> gpu tick 100            # Execute 100 GPU ticks
```

## Architecture

```
User Terminal
      │
      ▼
TerminalCreature.ex ──bond──► LLMCreature.ex
      │                              │
      │                        ┌─────┼─────┐
      │                        ▼     ▼     ▼
      │                     Memory  Tool  Reasoner
      │                       .ex    .ex    .ex
      ▼
EIS Interpreter (Python)
      │
      ▼
Substrate Layer (C/Metal)
```

### Creatures

Self-contained entities in Existence-Lang (.ex files):
- **TerminalCreature** - REPL interface
- **LLMCreature** - LLM reasoning
- **ToolCreature** - Shell execution
- **MemoryCreature** - Persistent storage
- **CalculatorCreature** - Math evaluation

### Bonds

Communication channels between creatures. Messages flow through bonds, not direct calls.

### Phases

Every execution tick runs four phases:
1. **READ** - Load state from trace
2. **PROPOSE** - Generate proposals with scores
3. **CHOOSE** - Select best proposal
4. **COMMIT** - Apply effects

## Project Structure

```
local_agency/
├── src/
│   ├── python/
│   │   ├── det_os_boot.py      # Primary entry point
│   │   └── det/
│   │       ├── lang/           # Existence-Lang compiler
│   │       ├── eis/            # EIS virtual machine
│   │       ├── os/             # DET-OS kernel
│   │       └── metal.py        # GPU backend
│   ├── existence/              # Creature source files (.ex)
│   └── substrate/              # C/Metal substrate
├── docs/                       # Documentation
├── archive/                    # Deprecated code (historical)
├── GETTING_STARTED.md          # This file
├── DEVELOPMENT_LOG.md          # Development history
└── ROADMAP_V2.md               # Project roadmap
```

## Key Files

| File | Purpose |
|------|---------|
| `det_os_boot.py` | Main entry point |
| `det/lang/parser.py` | Existence-Lang parser |
| `det/eis/vm.py` | EIS bytecode interpreter |
| `det/eis/primitives.py` | Built-in primitives |
| `det/metal.py` | Metal GPU backend |
| `src/existence/*.ex` | Creature definitions |

## Troubleshooting

### "Ollama not running"

Start Ollama in a separate terminal:
```bash
ollama serve
```

### "Model not found"

Download the model:
```bash
ollama pull llama3.2:3b
```

### GPU Not Available

GPU requires macOS with Apple Silicon. Check status:
```
det> gpu
GPU Status:
  Available: True/False
```

### Slow Responses

- Use a smaller model: `ollama pull llama3.2:1b`
- Enable GPU acceleration: `python det_os_boot.py --gpu`

## Documentation

- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md)** - System architecture
- **[docs/SUBSTRATE_SPEC.md](docs/SUBSTRATE_SPEC.md)** - Substrate specification
- **[ROADMAP_V2.md](ROADMAP_V2.md)** - Project roadmap
- **[DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)** - Development history

## Next Steps

After getting started, explore:

1. **Load more creatures**: `use memory`, `use reasoner`, `use planner`
2. **Bond creatures together**: `bond llm memory`
3. **Check GPU performance**: `gpu benchmark`
4. **Read the roadmap**: Future phases include networking and persistence

---

*Last Updated: 2026-01-24*
