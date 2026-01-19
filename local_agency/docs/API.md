# DET Local Agency - API Reference

**Version**: 0.6.4
**Last Updated**: 2026-01-18

This document provides comprehensive API documentation for the DET Local Agency system.

---

## Table of Contents

1. [Core Module](#core-module)
2. [LLM Interface](#llm-interface)
3. [Memory System](#memory-system)
4. [Internal Dialogue](#internal-dialogue)
5. [Multi-Model Routing](#multi-model-routing)
6. [Sandbox & Execution](#sandbox--execution)
7. [Task Management](#task-management)
8. [Timer System](#timer-system)
9. [Emotional Integration](#emotional-integration)
10. [Consolidation System](#consolidation-system)
11. [Network Protocol](#network-protocol)
12. [Test Harness](#test-harness)
13. [Web Application](#web-application)
14. [Metrics & Logging](#metrics--logging)

---

## Core Module

### `DETCore`

The main Python wrapper for the DET C kernel.

```python
from det import DETCore, DETParams, DETDecision, DETEmotion, DETLayer
```

#### Constructor

```python
DETCore(params: Optional[DETParams] = None, lib_path: Optional[str] = None)
```

**Parameters:**
- `params`: Custom DET physics parameters. Uses defaults if None.
- `lib_path`: Path to the shared library (auto-detected if None).

**Example:**
```python
# Create with defaults
core = DETCore()

# Create with custom parameters
params = DETParams()
params.tau_base = 0.02
params.sigma_base = 0.12
core = DETCore(params=params)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tick` | `int` | Current simulation tick |
| `num_nodes` | `int` | Total number of nodes |
| `num_active` | `int` | Number of active nodes |
| `num_bonds` | `int` | Number of bonds |
| `num_ports` | `int` | Number of port nodes |
| `params` | `DETParams` | Current physics parameters |

#### Simulation Methods

```python
# Advance simulation by one timestep
core.step(dt: float = 0.1)

# Reset to initial state
core.reset()

# Run warmup steps (recommended after creation)
core.warmup(steps: int = 50, dt: float = 0.1)

# Update individual components
core.update_presence()
core.update_coherence(dt: float = 0.1)
core.identify_self()
```

#### Gatekeeper

```python
decision = core.evaluate_request(
    tokens: List[int],       # Token IDs representing the request
    target_domain: int = 0,  # Target memory domain
    retry_count: int = 0     # Number of retries so far
) -> DETDecision
```

**Returns:** `DETDecision.PROCEED`, `RETRY`, `STOP`, or `ESCALATE`

#### Port Interface (LLM Communication)

```python
# Inject stimulus through port nodes
core.inject_stimulus(
    port_indices: List[int],  # Ports to activate
    activations: List[float]  # Activation values
)

# Create temporary interface bonds to a domain
core.create_interface_bonds(target_domain: int, initial_c: float = 0.3)

# Clean up temporary bonds
core.cleanup_interface_bonds()
```

#### Memory Domains

```python
# Register a new domain
core.register_domain(name: str) -> bool

# Get domain coherence
coherence = core.get_domain_coherence(domain: int) -> float
```

#### Queries

```python
# Get emotional state
emotion = core.get_emotion() -> DETEmotion
emotion_str = core.get_emotion_string() -> str

# Get Self cluster affect (valence, arousal, bondedness)
v, a, b = core.get_self_affect() -> Tuple[float, float, float]

# Get aggregate metrics (presence, coherence, resource, debt)
p, c, f, q = core.get_aggregates() -> Tuple[float, float, float, float]

# Get node/bond state
node = core.get_node(index: int) -> DETNode
bond = core.get_bond(index: int) -> DETBond
port = core.get_port(index: int) -> DETPort
```

#### Node/Bond Management

```python
# Recruit node from dormant pool
node_id = core.recruit_node(layer: DETLayer) -> int  # -1 if unavailable

# Return node to dormant pool
core.retire_node(node_id: int)

# Create bond between nodes
bond_id = core.create_bond(i: int, j: int) -> int  # -1 if failed

# Find existing bond
bond_id = core.find_bond(i: int, j: int) -> int  # -1 if not found
```

#### Grace Injection (Phase 4)

```python
# Inject grace for boundary recovery
core.inject_grace(node_id: int, amount: float)

# Check if node needs grace
needs = core.needs_grace(node_id: int) -> bool

# Get total grace needed across all nodes
total = core.total_grace_needed() -> float
```

#### Learning via Recruitment (Phase 4)

```python
# Check if learning is possible
can = core.can_learn(complexity: float, domain: int = 0) -> bool

# Activate a new domain with recruited nodes
success = core.activate_domain(
    name: str,
    num_nodes: int = 8,
    initial_coherence: float = 0.4
) -> bool

# Transfer pattern between domains
success = core.transfer_pattern(
    source_domain: int,
    target_domain: int,
    transfer_strength: float = 0.5
) -> bool

# Get available learning capacity
capacity = core.learning_capacity() -> float
```

#### State Persistence (Phase 4)

```python
# Save/load state as bytes
state_bytes = core.save_state() -> bytes
success = core.load_state(data: bytes) -> bool

# Save/load state from file
core.save_to_file(filepath: str)
success = core.load_from_file(filepath: str) -> bool
```

#### Inspection

```python
# Get full state snapshot
state = core.inspect(detailed: bool = False) -> dict
```

Returns a dictionary with:
- `tick`: Current tick
- `emotion`: Emotional state string
- `aggregates`: Dict with presence, coherence, resource, debt
- `self_affect`: Dict with valence, arousal, bondedness
- `grace`: Dict with nodes_needing, total_needed
- `counts`: Dict with nodes, active, bonds, ports, self_cluster
- `layers`: Dict with P, A, dormant, port counts
- `layer_health`: Averages for F and q per layer
- `domains`: Per-domain stats (count, coherence, avg_F)
- `self_cluster`: List of node IDs in self-cluster

### Enumerations

```python
class DETLayer(IntEnum):
    DORMANT = 0
    A = 1
    P = 2
    PORT = 3

class DETDecision(IntEnum):
    PROCEED = 0
    RETRY = 1
    STOP = 2
    ESCALATE = 3

class DETEmotion(IntEnum):
    NEUTRAL = 0
    FLOW = 1
    CONTENTMENT = 2
    STRESS = 3
    OVERWHELM = 4
    APATHY = 5
    BOREDOM = 6
    PEACE = 7
```

---

## LLM Interface

```python
from det import DETLLMInterface, OllamaClient, DetIntentPacket, IntentType, DomainType
```

### `OllamaClient`

HTTP client for Ollama API.

```python
client = OllamaClient(
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b",
    timeout: float = 60.0
)

# Check availability
available = client.is_available() -> bool

# Generate completion
response = client.generate(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> str

# Chat completion
response = client.chat(
    messages: List[dict],  # [{"role": "user", "content": "..."}]
    temperature: float = 0.7
) -> str
```

### `DETLLMInterface`

High-level interface combining DET core with LLM.

```python
interface = DETLLMInterface(
    core: DETCore,
    ollama_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b"
)

# Process a user request
result = interface.process_request(user_input: str) -> dict
```

Returns:
```python
{
    "intent": str,      # classify, generate, execute, etc.
    "domain": str,      # math, language, code, etc.
    "decision": DETDecision,
    "response": str,
    "affect": dict,
    "complexity": float,
    "risk": float
}
```

### `DetIntentPacket`

Port protocol packet for LLM-to-DET communication.

```python
packet = DetIntentPacket(
    intent: IntentType,
    domain: DomainType,
    complexity: float,  # 0.0-1.0
    risk: float,        # 0.0-1.0
    content_tokens: List[int]
)
```

---

## Memory System

```python
from det import MemoryManager, MemoryDomain, MemoryEntry, DomainRouter, ContextWindow
```

### `MemoryDomain` (Enum)

```python
class MemoryDomain(Enum):
    GENERAL = "general"
    MATH = "math"
    LANGUAGE = "language"
    TOOL_USE = "tool_use"
    SCIENCE = "science"
    CODE = "code"
    REASONING = "reasoning"
    DIALOGUE = "dialogue"
```

### `MemoryManager`

```python
manager = MemoryManager(
    core: DETCore,
    storage_path: Optional[Path] = None  # Default: ~/.det_agency/memory
)

# Store a memory
entry = manager.store(
    content: str,
    importance: float = 0.5,  # 0.0-1.0
    domain: Optional[MemoryDomain] = None  # Auto-detected if None
) -> MemoryEntry

# Retrieve memories
results = manager.retrieve(
    query: str,
    limit: int = 5,
    domain: Optional[MemoryDomain] = None
) -> List[MemoryEntry]

# Route request to domain
domain, confidence, coherence = manager.route_request(text: str) -> Tuple

# Get domain statistics
stats = manager.get_domain_stats() -> Dict[str, dict]

# Clear all memories
manager.clear()
```

### `ContextWindow`

Manages context with automatic reduction.

```python
window = ContextWindow(
    max_tokens: int = 32768,
    reduction_threshold: float = 0.8
)

# Add message
window.add_message(role: str, content: str)

# Get current context
messages = window.get_context() -> List[dict]

# Get token count
count = window.token_count() -> int

# Check if reduction needed
needs = window.needs_reduction() -> bool

# Reduce context (summarize older messages)
window.reduce()
```

---

## Internal Dialogue

```python
from det import InternalDialogue, DialogueTurn, DialogueState
```

### `InternalDialogue`

System for internal thinking and request reformulation.

```python
dialogue = InternalDialogue(
    core: DETCore,
    client: OllamaClient,
    max_reformulations: int = 3
)

# Process user input through dialogue system
turn = dialogue.process(user_input: str) -> DialogueTurn

# Multi-turn internal thinking
turns = dialogue.think(topic: str, max_turns: int = 3) -> List[DialogueTurn]

# Get dialogue history
history = dialogue.get_history() -> List[DialogueTurn]

# Get summary of recent dialogue
summary = dialogue.get_summary() -> str

# Clear history
dialogue.clear_history()
```

### `DialogueTurn`

```python
@dataclass
class DialogueTurn:
    input_text: str
    output_text: str
    decision: DETDecision
    reformulation_count: int
    affect_at_start: dict
    affect_at_end: dict
    timestamp: float
```

---

## Multi-Model Routing

```python
from det import (
    ModelConfig, ModelPool, ModelStatus, LLMRouter,
    RoutingResult, MultiModelInterface, DEFAULT_MODELS
)
```

### `ModelConfig`

```python
config = ModelConfig(
    name: str,              # Model identifier (e.g., "llama3.2:3b")
    display_name: str,      # Human-readable name
    domains: List[str],     # Domains this model handles
    priority: int = 0,      # Higher = preferred
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    is_fast: bool = False,
    supports_code: bool = False,
    supports_math: bool = False
)
```

### `ModelPool`

Manages multiple models with health monitoring.

```python
pool = ModelPool(base_url: str = "http://localhost:11434")

# Register a model
pool.register(config: ModelConfig)

# Get available model for domain
model = pool.get_model_for_domain(domain: str) -> Optional[ModelConfig]

# Get model health status
status = pool.get_status(model_name: str) -> ModelStatus

# Check all model health
pool.health_check()
```

### `LLMRouter`

Routes requests to appropriate models.

```python
router = LLMRouter(pool: ModelPool, core: DETCore)

# Route a request
result = router.route(text: str) -> RoutingResult
```

### `MultiModelInterface`

High-level interface with full pipeline.

```python
interface = MultiModelInterface(
    core: DETCore,
    base_url: str = "http://localhost:11434"
)

# Process request through full pipeline
result = interface.process(user_input: str) -> dict
```

### Default Models

```python
DEFAULT_MODELS = {
    "general": ModelConfig(name="llama3.2:3b", domains=["general", "dialogue", "language"]),
    "math": ModelConfig(name="deepseek-math:7b", domains=["math"]),
    "code": ModelConfig(name="qwen2.5-coder:7b", domains=["code", "tool_use"]),
    "reasoning": ModelConfig(name="deepseek-r1:8b", domains=["reasoning", "science"]),
}
```

---

## Sandbox & Execution

```python
from det import BashSandbox, FileOperations, CommandAnalyzer, RiskLevel, PermissionLevel
```

### `RiskLevel` (Enum)

```python
class RiskLevel(Enum):
    SAFE = 0        # Read-only, informational
    LOW = 1         # Local changes, reversible
    MEDIUM = 2      # System changes, needs review
    HIGH = 3        # Dangerous, requires approval
    CRITICAL = 4    # Potentially destructive
```

### `PermissionLevel` (Enum)

```python
class PermissionLevel(Enum):
    NONE = 0        # No permissions
    READ = 1        # Read-only access
    WRITE = 2       # File modifications
    EXECUTE = 3     # Command execution
    ADMIN = 4       # Full access
```

### `CommandAnalyzer`

```python
analyzer = CommandAnalyzer()

# Analyze command risk
risk = analyzer.analyze(command: str) -> RiskLevel

# Check if command is allowed
allowed = analyzer.is_allowed(command: str, level: PermissionLevel) -> bool
```

### `BashSandbox`

```python
sandbox = BashSandbox(
    core: DETCore,
    permission_level: PermissionLevel = PermissionLevel.READ,
    allowed_paths: Optional[List[str]] = None,
    timeout: float = 30.0,
    max_output: int = 10000
)

# Execute command
result = sandbox.execute(command: str) -> dict
# Returns: {"success": bool, "output": str, "error": str, "risk": RiskLevel}

# Execute with DET approval
result = sandbox.execute_with_approval(command: str) -> dict
```

### `FileOperations`

Safe file access operations.

```python
ops = FileOperations(sandbox: BashSandbox)

# Read file
content = ops.read(path: str) -> str

# Write file (requires WRITE permission)
success = ops.write(path: str, content: str) -> bool

# List directory
files = ops.list_dir(path: str) -> List[str]

# Check if path exists
exists = ops.exists(path: str) -> bool
```

---

## Task Management

```python
from det import TaskManager, Task, TaskStep, TaskStatus, TaskPriority
```

### `TaskPriority` (Enum)

```python
class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
```

### `TaskStatus` (Enum)

```python
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
```

### `TaskManager`

```python
manager = TaskManager(
    core: DETCore,
    client: OllamaClient,
    storage_path: Optional[Path] = None
)

# Create task from description (LLM decomposition)
task = manager.create_task(description: str, priority: TaskPriority = TaskPriority.NORMAL) -> Task

# Execute next step of a task
result = manager.execute_step(task_id: str) -> dict

# Get task status
task = manager.get_task(task_id: str) -> Optional[Task]

# List all tasks
tasks = manager.list_tasks(status: Optional[TaskStatus] = None) -> List[Task]

# Pause/resume task
manager.pause_task(task_id: str)
manager.resume_task(task_id: str)

# Cancel task
manager.cancel_task(task_id: str)
```

---

## Timer System

```python
from det import TimerSystem, ScheduledEvent, ScheduleType, setup_default_schedule
```

### `ScheduleType` (Enum)

```python
class ScheduleType(Enum):
    ONCE = "once"           # Single execution
    INTERVAL = "interval"   # Repeated at interval
    DAILY = "daily"         # Daily at specific time
```

### `TimerSystem`

```python
timer = TimerSystem(core: DETCore)

# Schedule an event
event_id = timer.schedule(
    callback: Callable,
    delay: float,  # Seconds
    schedule_type: ScheduleType = ScheduleType.ONCE,
    name: Optional[str] = None
) -> str

# Cancel event
timer.cancel(event_id: str)

# Pause/resume event
timer.pause(event_id: str)
timer.resume(event_id: str)

# List scheduled events
events = timer.list_events() -> List[ScheduledEvent]

# Start/stop timer thread
timer.start()
timer.stop()
```

### Built-in Callbacks

```python
# Setup default schedule (health check, memory consolidation, DET maintenance)
setup_default_schedule(timer: TimerSystem, core: DETCore, memory: MemoryManager)
```

---

## Emotional Integration

```python
from det import (
    EmotionalIntegration, EmotionalMode, BehaviorModulation,
    RecoveryState, MultiSessionManager, SessionContext
)
```

### `EmotionalMode` (Enum)

```python
class EmotionalMode(Enum):
    EXPLORATION = "exploration"  # High valence, high arousal
    FOCUSED = "focused"          # Positive valence, low arousal
    RECOVERY = "recovery"        # Low resources, needs rest
    VIGILANT = "vigilant"        # Negative valence, high arousal
    NEUTRAL = "neutral"          # Baseline state
```

### `EmotionalIntegration`

```python
integration = EmotionalIntegration(core: DETCore)

# Get current emotional mode
mode = integration.get_mode() -> EmotionalMode

# Get behavior modulation parameters
modulation = integration.get_modulation() -> BehaviorModulation

# Check for curiosity spike (high novelty)
curious = integration.detect_curiosity_spike() -> bool

# Schedule recovery if needed
integration.schedule_recovery()
```

### `BehaviorModulation`

```python
@dataclass
class BehaviorModulation:
    temperature_multiplier: float  # LLM temperature adjustment
    risk_tolerance: float          # Risk threshold adjustment
    exploration_rate: float        # Exploration vs exploitation
    retry_patience: int            # Max retries before stopping
    complexity_threshold: float    # Max complexity to attempt
```

### `MultiSessionManager`

```python
manager = MultiSessionManager(core: DETCore, storage_path: Optional[Path] = None)

# Create new session
session = manager.create_session(name: Optional[str] = None) -> SessionContext

# Get/switch sessions
session = manager.get_session(session_id: str) -> Optional[SessionContext]
manager.switch_session(session_id: str)

# List sessions
sessions = manager.list_sessions() -> List[SessionContext]

# Save/load sessions
manager.save_sessions()
manager.load_sessions()
```

---

## Consolidation System

```python
from det import (
    ConsolidationManager, ConsolidationConfig, ConsolidationState,
    ConsolidationPhase, ConsolidationCycle, IdleDetector, setup_consolidation
)
```

### `ConsolidationPhase` (Enum)

```python
class ConsolidationPhase(Enum):
    IDLE = "idle"
    MEMORY_SCAN = "memory_scan"
    DATA_GENERATION = "data_generation"
    MODEL_TRAINING = "model_training"
    GRACE_INJECTION = "grace_injection"
    VERIFICATION = "verification"
```

### `ConsolidationConfig`

```python
config = ConsolidationConfig(
    idle_threshold: float = 300.0,     # Seconds before consolidation
    min_interval: float = 3600.0,      # Min time between consolidations
    max_duration: float = 600.0,       # Max consolidation duration
    min_examples: int = 10,            # Min training examples required
    max_domains: int = 3,              # Max domains to train per cycle
    require_positive_valence: bool = True,  # Only train if mood is good
    grace_injection_amount: float = 0.5     # Grace during recovery
)
```

### `ConsolidationManager`

```python
manager = ConsolidationManager(
    core: DETCore,
    memory: MemoryManager,
    retuner: MemoryRetuner,
    timer: TimerSystem,
    config: Optional[ConsolidationConfig] = None
)

# Start/stop automatic consolidation
manager.start()
manager.stop()

# Manually trigger consolidation
success = manager.trigger_consolidation() -> bool

# Get current state
state = manager.get_state() -> ConsolidationState

# Get current cycle info
cycle = manager.get_current_cycle() -> Optional[ConsolidationCycle]
```

### Quick Setup

```python
# Setup consolidation with defaults
manager = setup_consolidation(
    core: DETCore,
    memory: MemoryManager,
    timer: TimerSystem
) -> ConsolidationManager
```

---

## Network Protocol

```python
from det import (
    MessageType, NodeType, NodeStatus, DETMessage, NodeInfo,
    Transport, ExternalNode, StubTransport, StubExternalNode,
    NetworkRegistry, create_stub_network
)
```

### `MessageType` (Enum)

```python
class MessageType(Enum):
    HEARTBEAT = 0x01
    STATE_UPDATE = 0x02
    AFFECT_UPDATE = 0x03
    STIMULUS_INJECT = 0x04
    GRACE_INJECT = 0x05
    BOND_UPDATE = 0x06
    SYNC_REQUEST = 0x07
    SYNC_RESPONSE = 0x08
```

### `NodeType` (Enum)

```python
class NodeType(Enum):
    ESP32 = "esp32"
    RASPBERRY_PI = "raspberry_pi"
    PYTHON_AGENT = "python_agent"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
```

### `DETMessage`

Binary protocol for network communication.

```python
message = DETMessage(
    msg_type: MessageType,
    source_id: int,
    target_id: int,
    payload: bytes
)

# Serialize/deserialize
data = message.serialize() -> bytes
message = DETMessage.deserialize(data: bytes) -> DETMessage
```

### `NetworkRegistry`

```python
registry = NetworkRegistry(core: DETCore)

# Register external node
registry.register_node(info: NodeInfo)

# Broadcast state to all nodes
registry.broadcast_state()

# Handle incoming message
registry.handle_message(message: DETMessage)

# Get network status
status = registry.get_status() -> dict
```

### Stub Implementations

For testing and development:

```python
# Create stub network for testing
registry, nodes = create_stub_network(core: DETCore, num_nodes: int = 3)
```

---

## Test Harness

```python
from det import (
    HarnessController, HarnessCLI, HarnessEvent, HarnessEventType,
    Snapshot, create_harness, run_harness_cli
)
```

### `HarnessController`

```python
harness = HarnessController(
    core: DETCore,
    storage_path: Optional[Path] = None,
    start_paused: bool = False
)

# Resource injection
harness.inject_f(node: int, amount: float) -> bool
harness.inject_q(node: int, amount: float) -> bool
harness.inject_all(f_amount: float, q_amount: float) -> int

# Agency manipulation
harness.set_agency(node: int, agency: float) -> bool

# Bond manipulation
harness.create_bond(i: int, j: int) -> int
harness.destroy_bond(i: int, j: int) -> bool
harness.set_coherence(i: int, j: int, coherence: float) -> bool

# Time controls
harness.pause()
harness.resume()
harness.step(dt: float = 0.1)
harness.step_n(n: int, dt: float = 0.1) -> int
harness.set_speed(speed: float)

# State inspection
harness.get_aggregates() -> dict
harness.get_affect() -> dict
harness.get_node_state(index: int) -> dict
harness.get_bond_state(i: int, j: int) -> dict
harness.get_self_cluster() -> List[int]
harness.get_emotional_state() -> str

# Snapshots
harness.take_snapshot(name: str) -> bool
harness.restore_snapshot(name: str) -> bool
harness.list_snapshots() -> List[dict]
harness.delete_snapshot(name: str) -> bool

# Event logging
harness.get_events(limit: int = 50) -> List[dict]
harness.clear_events()
harness.add_event_callback(callback: Callable[[HarnessEvent], None])

# Advanced probing (Phase 6.3)
harness.trigger_escalation(node: int)
harness.inject_grace(node: int, amount: float)
harness.inject_grace_all(amount: float) -> int
harness.get_total_grace_needed() -> float
harness.get_learning_capacity() -> float
harness.can_learn(complexity: float, domain: int = 0) -> bool
harness.activate_domain(name: str, num_nodes: int, coherence: float) -> bool
harness.transfer_pattern(source: int, target: int, strength: float) -> bool
harness.evaluate_request(tokens: List[int], domain: int, retry_count: int) -> str
```

### Quick Start

```python
# Create harness
harness = create_harness(core=None, start_paused=False)

# Run interactive CLI
run_harness_cli(harness)
```

---

## Web Application

```python
from det import create_app, run_server, DETStateAPI, WEBAPP_AVAILABLE
```

### Requirements

```bash
pip install fastapi uvicorn websockets jinja2
```

### `DETStateAPI`

Clean API wrapper for frontend.

```python
api = DETStateAPI(core=core, harness=harness)

# Status endpoints
api.get_status() -> dict
api.get_aggregates() -> dict
api.get_affect() -> dict
api.get_emotional_state() -> str
api.get_self_cluster() -> List[int]

# Node/Bond data
api.get_nodes(include_inactive=False) -> List[dict]
api.get_node(index: int) -> dict
api.get_bonds(min_coherence=0.01) -> List[dict]
api.get_bond(i: int, j: int) -> dict

# Full state
api.get_full_state() -> dict
api.get_visualization_data() -> dict

# Controls
api.step(n=1, dt=0.1) -> dict
api.pause() -> bool
api.resume() -> bool
api.set_speed(speed: float) -> float
api.inject_f(node: int, amount: float) -> bool
api.inject_q(node: int, amount: float) -> bool
api.take_snapshot(name: str) -> bool
api.restore_snapshot(name: str) -> bool
api.list_snapshots() -> List[dict]
api.get_events(limit=50) -> List[dict]
```

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Overall status |
| `/api/state` | GET | Full state |
| `/api/nodes` | GET | All nodes |
| `/api/nodes/{id}` | GET | Single node |
| `/api/bonds` | GET | All bonds |
| `/api/visualization` | GET | 3D visualization data |
| `/api/control/step` | POST | Execute steps |
| `/api/control/pause` | POST | Pause simulation |
| `/api/control/resume` | POST | Resume simulation |
| `/api/control/speed` | POST | Set speed |
| `/api/inject/f` | POST | Inject resource |
| `/api/inject/q` | POST | Inject debt |
| `/api/snapshots` | GET | List snapshots |
| `/api/snapshots` | POST | Take snapshot |
| `/api/snapshots/{name}` | PUT | Restore snapshot |
| `/api/events` | GET | Recent events |
| `/api/metrics/dashboard` | GET | Metrics dashboard |
| `/api/metrics/timeline/{field}` | GET | Timeline data |

### WebSocket

Connect to `/ws` for real-time state updates.

### Quick Start

```python
# Run webapp server
if WEBAPP_AVAILABLE:
    from det.webapp import run_server
    run_server(core=core, harness=harness, port=8420)
```

Or from CLI:
```bash
# From the REPL
/webapp 8420
```

---

## Metrics & Logging

```python
from det import (
    MetricsCollector, MetricsSample, DETEvent, DETEventType, Profiler,
    create_metrics_collector, create_profiler
)
```

### `DETEventType` (Enum)

```python
class DETEventType(Enum):
    ESCALATION = "escalation"
    COMPILATION = "compilation"
    RECRUITMENT = "recruitment"
    BOND_FORMED = "bond_formed"
    BOND_BROKEN = "bond_broken"
    PRISON_DETECTED = "prison_detected"
    GRACE_INJECTED = "grace_injected"
    DOMAIN_ACTIVATED = "domain_activated"
    GATEKEEPER_DECISION = "gatekeeper_decision"
```

### `MetricsCollector`

```python
collector = MetricsCollector(
    core: DETCore,
    max_samples: int = 1000,
    sample_interval: float = 0.1
)

# Start/stop collection
collector.start()
collector.stop()

# Manual sample
collector.sample()

# Get data
samples = collector.get_samples(limit=100) -> List[MetricsSample]
timeline = collector.get_timeline(field: str) -> List[Tuple[float, float]]
dashboard = collector.get_dashboard() -> dict
statistics = collector.get_statistics() -> dict

# Event logging
collector.log_event(event_type: DETEventType, **data)
events = collector.get_events(limit=50, event_type=None) -> List[DETEvent]
collector.add_event_callback(callback: Callable[[DETEvent], None])
```

### `Profiler`

```python
profiler = Profiler()

# Profile tick execution
profiler.start_tick()
profiler.end_tick()

# Profile individual steps
profiler.start_step(name: str)
profiler.end_step(name: str)

# Get report
report = profiler.get_report() -> dict
# Contains: tick_count, avg_tick_time, min_tick_time, max_tick_time,
#           p50, p95, memory_mb, step_timings
```

### Quick Start

```python
collector = create_metrics_collector(core)
profiler = create_profiler()

collector.start()

# ... run simulation ...

print(collector.get_dashboard())
print(profiler.get_report())
```

---

## Constants

```python
from det.core import (
    DET_MAX_NODES,      # 4096
    DET_MAX_BONDS,      # 16384
    DET_MAX_PORTS,      # 64
    DET_MAX_DOMAINS,    # 16
    DET_P_LAYER_SIZE,   # 16
    DET_A_LAYER_SIZE,   # 256
    DET_DORMANT_SIZE,   # 3760
)
```

---

## Error Handling

Most methods that can fail return success indicators:
- Boolean methods return `True`/`False`
- Index methods return `-1` on failure
- Optional returns are `None` on failure

Exceptions are raised for:
- Invalid parameters (ValueError)
- Missing library (FileNotFoundError)
- Core creation failure (RuntimeError)
- Index out of range (IndexError)

---

## Thread Safety

- The DET C kernel is **not thread-safe**
- Use locks when accessing core from multiple threads
- The `TimerSystem` and `ConsolidationManager` handle their own threading
- WebSocket broadcasts use async/await patterns

---

## Version History

| Version | Changes |
|---------|---------|
| 0.1.0 | Core DET kernel, Python bridge, Ollama integration, CLI |
| 0.2.0 | Memory domains, MLX training, context management, dialogue |
| 0.3.0 | Sandbox, tasks, timers, code execution |
| 0.4.0 | Extended dynamics, learning, emotional integration, sessions |
| 0.5.0 | Multi-LLM routing, consolidation, network protocol |
| 0.6.0 | Test harness, web visualization, metrics, logging |
| 0.6.4 | Advanced probing, metrics API, setup automation |
