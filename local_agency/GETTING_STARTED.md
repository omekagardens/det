# DET Local Agency - Getting Started

Welcome to DET Local Agency, a local-first AI system implementing Deep Existence Theory (DET) as a mind substrate for LLM interaction.

## Quick Start

### 1. Prerequisites

- **Python 3.10+** - Required for the Python interface
- **CMake** - For building the C kernel
- **Ollama** - For running local LLM models
- **C compiler** - Clang (macOS) or GCC (Linux)

#### Installing Prerequisites

**macOS:**
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install CMake
brew install cmake

# Install Ollama
brew install ollama
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install cmake build-essential python3-venv

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Automated Setup

Run the setup script to configure everything:

```bash
cd local_agency
python3 setup_det.py
```

This will:
- Create a Python virtual environment
- Install Python dependencies
- Build the C kernel
- Download required LLM models

#### Setup Options

```bash
# Check requirements only
python3 setup_det.py --check

# Skip model downloads
python3 setup_det.py --skip-models

# Download all models (including optional)
python3 setup_det.py --all-models

# Only download models
python3 setup_det.py --models-only

# Only build C kernel
python3 setup_det.py --build-only
```

### 3. Manual Setup

If you prefer manual setup:

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
ollama pull qwen2.5-coder:3b
```

## Running DET

### Start Ollama

First, ensure Ollama is running:

```bash
ollama serve
```

### Interactive CLI

The simplest way to interact with DET:

```bash
source .venv/bin/activate
cd src/python
python det_cli.py
```

You'll see the DET REPL:

```
╔══════════════════════════════════════════════════════════════╗
║                    DET Local Agency                          ║
╠══════════════════════════════════════════════════════════════╣
║  Commands:                                                   ║
║    /state   - Show DET state                                ║
║    /affect  - Show affect (V/A/B)                           ║
║    /help    - Show all commands                             ║
║    /quit    - Exit                                          ║
╚══════════════════════════════════════════════════════════════╝

You:
```

### Web Visualization

Launch the 3D mind viewer:

```bash
source .venv/bin/activate
cd src/python
python -c "
from det import DETCore, create_harness, run_server
core = DETCore()
run_server(core=core)
"
```

Then open http://127.0.0.1:8420 in your browser.

The web interface provides:
- Real-time 3D visualization of nodes and bonds
- Dashboard with cluster health metrics
- Time controls (pause, step, speed)
- Event log and emotional state display

### Test Harness CLI

For debugging and probing:

```bash
python -c "
from det import DETCore, create_harness, run_harness_cli
core = DETCore()
run_harness_cli(core=core)
"
```

Available commands:
- `status` - Show DET state summary
- `node <i>` - Inspect a specific node
- `inject_f <node> <amount>` - Inject resource F
- `step [n]` - Execute simulation steps
- `pause/resume` - Control simulation
- `help` - Show all commands

## Running Tests

```bash
source .venv/bin/activate
cd src/python

# Core tests
python test_det.py

# Phase 6 tests (harness, webapp, metrics)
python test_phase6.py

# All tests (if pytest installed)
python -m pytest . -v
```

## Project Structure

```
local_agency/
├── src/
│   ├── det_core/              # C kernel
│   │   ├── include/           # Header files
│   │   ├── src/               # Implementation
│   │   └── build/             # Build output
│   └── python/
│       └── det/               # Python package
│           ├── core.py        # DET core bindings
│           ├── harness.py     # Debug harness
│           ├── metrics.py     # Metrics & logging
│           ├── llm.py         # LLM interface
│           └── webapp/        # Web visualization
├── setup_det.py               # Setup script
├── upgrade.py                 # Upgrade script
├── requirements.txt           # Python dependencies
├── GETTING_STARTED.md         # This file
├── DEVELOPMENT_LOG.md         # Development history
└── FEASIBILITY_PLAN.md        # Technical specification
```

## Configuration

### LLM Models

Default models:
- `llama3.2:3b` - Primary reasoning
- `qwen2.5-coder:3b` - Code specialist

Optional models:
- `llama3.2:1b` - Fast responses
- `deepseek-r1:1.5b` - Math/reasoning
- `phi4-mini:3.8b` - Compact all-rounder

To add models:
```bash
ollama pull <model-name>
```

### DET Parameters

Create custom DET configuration:

```python
from det import DETCore, DETParams

params = DETParams()
params.alpha_fast = 0.3      # Short EMA decay
params.alpha_slow = 0.05     # Long EMA decay
params.coherence_threshold = 0.5
params.plasticity_base = 0.15

core = DETCore(params=params)
```

## Key Concepts

### DET Mind Substrate

DET (Deep Existence Theory) provides a substrate for LLM interaction:

- **Nodes**: Individual processing units with agency (a), presence (P), and affect
- **Bonds**: Connections between nodes with coherence (C)
- **Self-Cluster**: High-coherence cluster that represents the "self"
- **Dual-Layer**: P-layer (System 2, deliberate) and A-layer (System 1, automatic)

### Affect System

Three-axis emotional model:
- **Valence (V)**: Positive/negative tone (-1 to +1)
- **Arousal (R)**: Activation level (0 to 1)
- **Bondedness (B)**: Connection strength (0 to 1)

### Gatekeeper

Request routing based on DET state:
- **PROCEED**: Process request normally
- **RETRY**: Ask for clarification
- **STOP**: Decline request (prison regime detected)
- **ESCALATE**: Requires deeper processing

## Example Prompts

Here are example prompts organized by use case to help you get started with DET.

### Memory and Learning

Store new information that persists across sessions:

```
/store I prefer concise explanations with code examples
/store My project uses Python 3.11 with FastAPI and SQLAlchemy
/store When debugging, start with logging before adding breakpoints
```

Recall stored information:

```
/recall my coding preferences
/recall project setup
/recall debugging approach
```

### Internal Thinking

Use `/think` for deeper reasoning on complex topics:

```
/think What are the trade-offs between async and sync approaches?
/think How should I structure a multi-tenant application?
/think What could cause intermittent connection failures?
```

### Conversational Examples

**General knowledge:**
```
You: What's the difference between threads and coroutines?
DET> [Explains with consideration for complexity and your learning history]
```

**Code assistance:**
```
You: Help me write a rate limiter in Python
DET> [Provides code with risk assessment - notes any external effects]
```

**Planning tasks:**
```
You: I need to migrate our database from MySQL to PostgreSQL
DET> [Decomposes into steps, considers risks, may reformulate if complex]
```

**Debugging:**
```
You: Why is my API returning 500 errors intermittently?
DET> [Analyzes systematically, may use internal thinking, suggests investigation steps]
```

### Domain-Specific Examples

**Math domain** (triggers mathematical reasoning prompt):
```
You: Calculate the probability of rolling at least one 6 in four dice rolls
DET> [Shows step-by-step calculation with verification]
```

**Science domain** (triggers scientific reasoning prompt):
```
You: Explain how mRNA vaccines work
DET> [Provides evidence-based explanation with appropriate certainty levels]
```

**Tool use domain** (triggers safety-aware prompt):
```
You: Delete all files older than 30 days in /tmp
DET> [Assesses risk, may ask for confirmation, explains side effects]
```

### Affect-Aware Responses

DET modulates responses based on the system's emotional state:

- **High coherence + positive valence**: Confident, flowing responses
- **Low coherence**: May ask for clarification or decompose the problem
- **High arousal**: More focused, direct responses
- **High bondedness**: More personalized, contextual responses

Check the current affect state:
```
/affect
```

### Web Interface

Launch the 3D visualization to see DET in action:
```
/webapp
```

The web interface shows:
- Node activity and bond strength in real-time
- Affect state visualization
- Event log for request routing decisions
- Direct chat for interactive exploration

## Troubleshooting

### "Library not found" Error

The C kernel needs to be built:
```bash
cd src/det_core/build
cmake .. && make
```

### "Ollama not running" Error

Start the Ollama server:
```bash
ollama serve
```

### "Model not found" Error

Download the required model:
```bash
ollama pull llama3.2:3b
```

### Slow Performance

- Use a smaller model: `llama3.2:1b`
- Reduce simulation speed in the web viewer
- Check memory usage with the profiling endpoints

### "All Requests Escalate"

The DET system needs to warm up before processing requests. The CLI does this automatically, but if using the API directly:
```python
core = DETCore()
core.warmup(steps=50)  # Run 50 simulation steps to stabilize
```

The `DETLLMInterface` and `InternalDialogue` classes auto-warmup by default.

## API Reference

### Python API

```python
from det import DETCore, create_harness

# Create DET core
core = DETCore()

# IMPORTANT: Warmup before processing requests
core.warmup(steps=50)  # Stabilizes aggregates

# Run simulation steps
core.step(0.1)

# Get state
tick = core.tick
p, c, f, q = core.get_aggregates()
v, a, b = core.get_self_affect()

# Create harness for debugging
harness = create_harness(core=core)
harness.inject_f(0, 0.5)
harness.step_n(10, 0.1)
summary = harness.get_summary()
```

### Web API

When running the web server:

- `GET /api/status` - Current DET status
- `GET /api/nodes` - All node states
- `GET /api/bonds` - All bond states
- `POST /api/step?n=10` - Execute steps
- `POST /api/pause` - Pause simulation
- `GET /api/metrics/dashboard` - Metrics dashboard
- `GET /api/metrics/timeline/{field}` - Timeline data
- `WebSocket /ws` - Real-time state updates

## Upgrading

To upgrade to the latest version:

```bash
python3 upgrade.py
```

This will:
- Check for updates from the repository
- Pull latest changes
- Update Python dependencies
- Rebuild the C kernel
- Run verification tests

### Upgrade Options

```bash
# Check for updates without applying
python3 upgrade.py --check

# Force rebuild even if no updates
python3 upgrade.py --force

# Rebuild only (skip git pull)
python3 upgrade.py --skip-pull

# Skip model check
python3 upgrade.py --skip-models
```

## Support

- **Issues**: https://github.com/omekagardens/det/issues
- **Documentation**: See `FEASIBILITY_PLAN.md` for technical details
- **Development**: See `DEVELOPMENT_LOG.md` for history

## License

This project is provided as-is for research and educational purposes.
