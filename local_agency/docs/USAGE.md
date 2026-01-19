# DET Local Agency - Usage Guide

**Version**: 0.6.4
**Last Updated**: 2026-01-18

This guide covers how to use DET Local Agency, from basic CLI interaction to advanced programmatic usage.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Interface](#cli-interface)
3. [Web Visualization](#web-visualization)
4. [Python API Usage](#python-api-usage)
5. [Test Harness](#test-harness)
6. [Memory System](#memory-system)
7. [Multi-Model Routing](#multi-model-routing)
8. [Task Automation](#task-automation)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3) for MLX training
- Ollama installed and running
- C compiler (clang/gcc)

### Automated Setup

```bash
cd local_agency
python setup_det.py
```

This script:
1. Checks system requirements
2. Creates virtual environment
3. Installs Python dependencies
4. Builds the C kernel
5. Downloads required Ollama models
6. Runs verification tests

### Manual Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Build C kernel
cd src/det_core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../../..

# Download models
ollama pull llama3.2:3b
ollama pull qwen2.5-coder:7b
```

### First Run

```bash
# Start Ollama (if not running)
ollama serve &

# Run the CLI
cd src/python
python det_cli.py
```

---

## CLI Interface

### Starting the CLI

```bash
# Default configuration
python det_cli.py

# Specify model
python det_cli.py --model mistral:7b

# Custom Ollama URL
python det_cli.py --url http://192.168.1.100:11434

# Disable features
python det_cli.py --no-state      # Hide DET state display
python det_cli.py --no-dialogue   # Disable internal dialogue

# Custom memory storage
python det_cli.py --storage /path/to/memory

# Show version
python det_cli.py --version
```

### CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/state` | Show detailed DET state | `/state` |
| `/inspect` | Full system inspection | `/inspect detailed` |
| `/grace` | Show/inject grace | `/grace 5 0.5` |
| `/affect` | Show affect visualization | `/affect` |
| `/memory` | Show memory statistics | `/memory` |
| `/store` | Store a memory | `/store This is important` |
| `/recall` | Search memories | `/recall python code` |
| `/think` | Internal thinking | `/think quantum mechanics` |
| `/webapp` | Launch web visualization | `/webapp 8420` |
| `/clear` | Clear conversation | `/clear` |
| `/help` | Show help | `/help` |
| `/quit` | Exit | `/quit` |

### Understanding the Display

After each response, the CLI shows DET state:

```
V[     ===  ] A[====      ] B[======    ] | P:0.45 C:0.62 F:0.78 q:0.12 | Emotion: contentment
```

- **V (Valence)**: Good/bad feeling (-1 to +1)
- **A (Arousal)**: Activation level (0 to 1)
- **B (Bondedness)**: Attachment/connection (0 to 1)
- **P**: Aggregate presence
- **C**: Aggregate coherence
- **F**: Available resource
- **q**: Structural debt
- **Emotion**: Derived emotional state

### Example Session

```
You> What is the capital of France?

[Processing...]
[Routed to language (conf: 0.85, coh: 0.72)]
[Decision: PROCEED, Reformulations: 0]

DET> The capital of France is Paris.

V[     ===  ] A[===       ] B[=====     ] | P:0.48 C:0.65 F:0.75 q:0.15 | Emotion: contentment

You> /think deeper about European geography

[Thinking about: deeper about European geography...]

Thought 1: Europe's geography is fascinating, with the Alps dividing...
Thought 2: The major rivers like the Rhine, Danube, and Seine...
Thought 3: Political boundaries have shifted throughout history...

[Completed 3 thinking turns]

You> /store Paris is also known as the City of Light

Stored to language domain (coherence: 0.68)

You> /recall light

Found 1 memories:
  1. [language] Paris is also known as the City of Light
```

---

## Web Visualization

### Launching the Web App

From the CLI:
```
You> /webapp 8420
Starting web visualization on http://127.0.0.1:8420
Opening browser...
```

Or programmatically:
```python
from det import create_harness, WEBAPP_AVAILABLE

if WEBAPP_AVAILABLE:
    from det.webapp import run_server

    core = DETCore()
    harness = create_harness(core=core)
    run_server(core=core, harness=harness, port=8420)
```

### Web Interface Features

#### 3D Visualization
- **Nodes**: Spheres colored by affect
  - Green = positive valence
  - Red = negative valence
  - Brightness = arousal
- **Bonds**: Lines between nodes
  - Thickness = coherence strength
  - Color = layer type
- **Self-cluster**: Highlighted with glow effect

#### Dashboard
- Real-time metrics (P, C, F, q)
- Affect bars (V, A, B)
- Emotional state indicator
- Event log

#### Controls
- **Play/Pause**: Toggle simulation
- **Step**: Single step forward
- **Speed**: 0.5x to 4x simulation speed
- **Snapshot**: Save/restore states

#### API Endpoints

Access raw data via REST API:
```bash
# Get full state
curl http://localhost:8420/api/state

# Get visualization data
curl http://localhost:8420/api/visualization

# Step simulation
curl -X POST http://localhost:8420/api/control/step -d '{"n": 10}'

# Get metrics
curl http://localhost:8420/api/metrics/dashboard
```

---

## Python API Usage

### Basic Usage

```python
from det import DETCore, DETLLMInterface, OllamaClient

# Create DET core
core = DETCore()
core.warmup(steps=50)

# Create LLM interface
interface = DETLLMInterface(core, model="llama3.2:3b")

# Process a request
result = interface.process_request("What is machine learning?")
print(f"Decision: {result['decision']}")
print(f"Response: {result['response']}")
```

### With Memory System

```python
from det import DETCore, MemoryManager, DomainRouter

core = DETCore()
core.warmup()

# Initialize memory
memory = MemoryManager(core)

# Store information
memory.store("Python is a programming language", importance=0.8)
memory.store("Machine learning uses statistical methods", importance=0.7)

# Retrieve
results = memory.retrieve("programming languages")
for entry in results:
    print(f"[{entry.domain.name}] {entry.content}")
```

### With Internal Dialogue

```python
from det import DETCore, OllamaClient, InternalDialogue

core = DETCore()
core.warmup()

client = OllamaClient(model="llama3.2:3b")
dialogue = InternalDialogue(core, client)

# Process with potential reformulation
turn = dialogue.process("Explain quantum entanglement simply")
print(f"Reformulations: {turn.reformulation_count}")
print(f"Response: {turn.output_text}")

# Multi-turn thinking
thoughts = dialogue.think("implications of quantum computing", max_turns=3)
for i, thought in enumerate(thoughts):
    print(f"Thought {i+1}: {thought.output_text[:100]}...")
```

### With Multi-Model Routing

```python
from det import DETCore, MultiModelInterface, DEFAULT_MODELS

core = DETCore()
core.warmup()

# Uses multiple specialized models
interface = MultiModelInterface(core)

# Math question routes to deepseek-math
result = interface.process("Solve: 2x + 5 = 13")

# Code question routes to qwen-coder
result = interface.process("Write a Python function to reverse a string")

# General question uses llama
result = interface.process("Tell me about the history of Rome")
```

### Emotional Integration

```python
from det import DETCore, EmotionalIntegration, EmotionalMode

core = DETCore()
core.warmup()

emotional = EmotionalIntegration(core)

# Check emotional state
mode = emotional.get_mode()
print(f"Current mode: {mode}")

# Get behavior modulation
mod = emotional.get_modulation()
print(f"Temperature multiplier: {mod.temperature_multiplier}")
print(f"Risk tolerance: {mod.risk_tolerance}")

# Schedule recovery if needed
if mode == EmotionalMode.RECOVERY:
    emotional.schedule_recovery()
```

### State Persistence

```python
from det import DETCore

core = DETCore()
core.warmup()

# Run some simulation
for _ in range(100):
    core.step()

# Save state
core.save_to_file("det_state.bin")

# Later, restore
core2 = DETCore()
core2.load_from_file("det_state.bin")
print(f"Restored at tick: {core2.tick}")
```

---

## Test Harness

### Interactive CLI Harness

```bash
python -c "from det import run_harness_cli; run_harness_cli()"
```

### Harness Commands

```
(harness) status          # Show current state
(harness) node 5          # Inspect node 5
(harness) bond 3 7        # Inspect bond between 3 and 7
(harness) self            # Show self-cluster
(harness) affect          # Show affect state

(harness) inject_f 5 1.0  # Inject F=1.0 to node 5
(harness) inject_q 5 0.5  # Inject q=0.5 to node 5
(harness) inject_all 0.5 0.1  # Inject F=0.5, q=0.1 to all

(harness) create_bond 3 7     # Create bond between 3 and 7
(harness) destroy_bond 3 7    # Destroy that bond
(harness) set_coherence 3 7 0.8  # Set coherence

(harness) step            # Single step
(harness) step 10         # 10 steps
(harness) pause           # Pause auto-run
(harness) resume          # Resume auto-run
(harness) speed 2.0       # Set 2x speed
(harness) run             # Start continuous run
(harness) stop            # Stop continuous run

(harness) snapshot before  # Save snapshot
(harness) restore before   # Restore snapshot
(harness) snapshots        # List snapshots

(harness) escalate 5       # Trigger escalation on node 5
(harness) grace 5 0.5      # Inject grace to node 5
(harness) grace_all 0.3    # Inject grace to all needing
(harness) learning         # Show learning capacity
(harness) gatekeeper 1 2 3 # Evaluate request tokens

(harness) quit
```

### Programmatic Harness

```python
from det import DETCore, create_harness

core = DETCore()
harness = create_harness(core=core, start_paused=True)

# Inject resources
harness.inject_f(5, 1.0)
harness.inject_q(5, 0.2)

# Step simulation
harness.step_n(10)

# Check state
print(harness.get_aggregates())
print(harness.get_affect())

# Take snapshot
harness.take_snapshot("experiment_1")

# Run more
harness.step_n(100)

# Compare
print("Before:", harness.list_snapshots()[0])
print("After:", harness.get_aggregates())

# Restore
harness.restore_snapshot("experiment_1")
```

---

## Memory System

### Memory Domains

| Domain | Purpose | Keywords |
|--------|---------|----------|
| GENERAL | General knowledge | (default) |
| MATH | Mathematical concepts | math, equation, calculate |
| LANGUAGE | Linguistics, grammar | language, word, grammar |
| TOOL_USE | Tool operations | tool, command, execute |
| SCIENCE | Scientific knowledge | science, physics, biology |
| CODE | Programming | code, function, class |
| REASONING | Logic, inference | reason, logic, because |
| DIALOGUE | Conversation | say, tell, ask |

### Memory Lifecycle

```python
from det import MemoryManager, MemoryDomain

memory = MemoryManager(core)

# 1. Store with auto-routing
entry = memory.store("The quadratic formula is x = (-b ± √(b²-4ac)) / 2a")
print(f"Routed to: {entry.domain.name}")  # → "math"

# 2. Store with explicit domain
entry = memory.store(
    "Always validate user input",
    domain=MemoryDomain.CODE,
    importance=0.9
)

# 3. Retrieve with relevance
results = memory.retrieve("quadratic", limit=5)
for r in results:
    print(f"Score: {r.access_count}, Content: {r.content[:50]}")

# 4. Domain-specific retrieval
results = memory.retrieve("formula", domain=MemoryDomain.MATH)
```

### Memory Consolidation

Memory can be consolidated during idle periods:

```python
from det import (
    DETCore, MemoryManager, TimerSystem,
    ConsolidationManager, setup_consolidation
)

core = DETCore()
memory = MemoryManager(core)
timer = TimerSystem(core)

# Setup automatic consolidation
consolidation = setup_consolidation(core, memory, timer)
consolidation.start()

# ... use the system ...

# Manually trigger consolidation
if consolidation.can_consolidate():
    consolidation.trigger_consolidation()
```

---

## Multi-Model Routing

### Default Model Configuration

```python
from det import DEFAULT_MODELS

for name, config in DEFAULT_MODELS.items():
    print(f"{name}: {config.name} → {config.domains}")
```

Output:
```
general: llama3.2:3b → ['general', 'dialogue', 'language']
math: deepseek-math:7b → ['math']
code: qwen2.5-coder:7b → ['code', 'tool_use']
reasoning: deepseek-r1:8b → ['reasoning', 'science']
```

### Custom Model Configuration

```python
from det import ModelConfig, ModelPool, LLMRouter

pool = ModelPool()

# Register custom models
pool.register(ModelConfig(
    name="codellama:34b",
    display_name="Code Llama 34B",
    domains=["code", "tool_use"],
    priority=10,  # Higher priority than default
    supports_code=True
))

pool.register(ModelConfig(
    name="mixtral:8x7b",
    display_name="Mixtral 8x7B",
    domains=["general", "reasoning"],
    priority=5,
    is_fast=False
))

# Use with router
router = LLMRouter(pool, core)
result = router.route("Write a sorting algorithm")
print(f"Routed to: {result.model.name}")
```

---

## Task Automation

### Creating Tasks

```python
from det import TaskManager, TaskPriority

manager = TaskManager(core, client)

# Create task from natural language
task = manager.create_task(
    "Analyze the project structure and create a summary document",
    priority=TaskPriority.HIGH
)

print(f"Task ID: {task.id}")
print(f"Steps: {len(task.steps)}")
for step in task.steps:
    print(f"  - {step.description}")
```

### Executing Tasks

```python
# Execute one step at a time
while task.status != TaskStatus.COMPLETED:
    result = manager.execute_step(task.id)
    print(f"Step result: {result['status']}")

    if result['status'] == 'failed':
        print(f"Error: {result['error']}")
        break
```

### Scheduled Tasks

```python
from det import TimerSystem, ScheduleType

timer = TimerSystem(core)

# Schedule one-time task
timer.schedule(
    callback=lambda: print("One-time task"),
    delay=60.0,  # 60 seconds
    schedule_type=ScheduleType.ONCE,
    name="reminder"
)

# Schedule recurring task
timer.schedule(
    callback=lambda: memory.consolidate(),
    delay=3600.0,  # Every hour
    schedule_type=ScheduleType.INTERVAL,
    name="consolidation"
)

timer.start()
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DET_OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `DET_DEFAULT_MODEL` | `llama3.2:3b` | Default LLM model |
| `DET_STORAGE_PATH` | `~/.det_agency` | Data storage path |
| `DET_LOG_LEVEL` | `INFO` | Logging level |

### Configuration File

Create `~/.det_agency/config.toml`:

```toml
[ollama]
url = "http://localhost:11434"
default_model = "llama3.2:3b"
timeout = 60.0

[memory]
storage_path = "~/.det_agency/memory"
max_entries_per_domain = 1000
consolidation_threshold = 0.8

[sandbox]
permission_level = "read"
timeout = 30.0
allowed_paths = ["/tmp", "~/projects"]

[det]
warmup_steps = 50
dt = 0.1
```

### DET Physics Parameters

For advanced tuning, modify DET physics:

```python
from det import DETParams, DETCore

params = DETParams()

# Time and screening
params.tau_base = 0.02      # Time scale
params.sigma_base = 0.12    # Charging rate

# Coherence dynamics
params.alpha_AA = 0.1       # A-layer coherence gain
params.lambda_AA = 0.05     # A-layer coherence decay
params.alpha_PP = 0.2       # P-layer coherence gain
params.lambda_PP = 0.02     # P-layer coherence decay

# Agency
params.lambda_a = 30.0      # Agency ceiling coupling

# Momentum
params.phi_L = 0.5          # Angular/momentum ratio
params.pi_max = 3.0         # Momentum cap

core = DETCore(params=params)
```

---

## Troubleshooting

### Common Issues

#### "Could not find det_core library"

The C kernel hasn't been built or is in an unexpected location.

```bash
# Rebuild the kernel
cd src/det_core
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Verify
ls -la libdet_core.dylib  # macOS
ls -la libdet_core.so     # Linux
```

#### "Ollama is not running"

```bash
# Start Ollama
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

#### "Model not found"

```bash
# List available models
ollama list

# Pull required model
ollama pull llama3.2:3b
```

#### "Import Error: fastapi"

Web visualization requires additional dependencies:

```bash
pip install fastapi uvicorn websockets jinja2
```

#### DET Core Shows Low Presence/Coherence

The system may need warmup:

```python
core = DETCore()
core.warmup(steps=100)  # More warmup steps
```

Or inject grace:
```python
total = core.total_grace_needed()
if total > 0:
    for i in range(core.num_active):
        if core.needs_grace(i):
            core.inject_grace(i, 0.5)
```

#### Memory System Not Persisting

Check storage path permissions:

```bash
ls -la ~/.det_agency/memory/
chmod 755 ~/.det_agency/memory/
```

#### Slow Response Times

1. Use a smaller model:
   ```bash
   python det_cli.py --model llama3.2:1b
   ```

2. Disable internal dialogue:
   ```bash
   python det_cli.py --no-dialogue
   ```

3. Check system resources:
   ```bash
   # CPU usage
   top -l 1 | grep -E "^CPU"

   # Memory
   vm_stat
   ```

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from det import DETCore
core = DETCore()
```

### Getting Help

- Check existing issues: https://github.com/anthropics/claude-code/issues
- Review the API documentation: `docs/API.md`
- Examine the development log: `DEVELOPMENT_LOG.md`

---

## Performance Tips

1. **Use warmup**: Always call `core.warmup()` after creation
2. **Batch operations**: Minimize individual API calls
3. **Use snapshots**: Save state before experiments
4. **Monitor metrics**: Use the metrics collector for profiling
5. **Right-size models**: Use smaller models for simple tasks
6. **Enable consolidation**: Let the system learn during idle time

---

## Security Considerations

1. **Sandbox permissions**: Default to READ-only, escalate as needed
2. **Path restrictions**: Whitelist allowed paths explicitly
3. **Network policy**: Control network access in sandbox
4. **Model trust**: Validate model outputs before execution
5. **Memory isolation**: Each session should have isolated context

---

## Next Steps

See [NEXT_STEPS.md](NEXT_STEPS.md) for the development roadmap and future features.
