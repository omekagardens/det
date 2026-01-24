"""
DET Local Agency - Python Interface
====================================

Python bindings for the DET C kernel using ctypes,
with Ollama LLM integration, memory management, internal dialogue,
and agentic operations (sandbox, tasks, timer, code execution).

Phase 4 adds: Extended DET dynamics, learning via recruitment,
emotional state integration, and multi-session support.

Phase 2.2 adds: MLX training pipeline for memory model retraining.

Phase 5.1 adds: Multi-LLM routing with domain-specialized models.

Phase 5.2 adds: Sleep/consolidation cycles with MLX training integration.

Phase 5.3 adds: Network protocol and interfaces for distributed DET nodes (preliminary).

Phase 6.1 adds: Test harness for DET debugging and probing.

Phase 6.2 adds: Web application for 3D visualization and real-time monitoring.

Phase 6.3 adds: Advanced interactive probing (escalation, grace, domains, gatekeeper).

Phase 6.4 adds: Metrics and logging (dashboard, event log, timeline, profiling).

Phase 7 adds: Existence-Lang programming language (agency-first language for DET-OS).
"""

from .core import DETCore, DETParams, DETDecision, DETEmotion, DETLayer, SomaticType, SomaticNode
from .harness import (
    HarnessController, HarnessCLI, HarnessEvent, HarnessEventType,
    Snapshot, create_harness, run_harness_cli
)

# Web app imports (optional - may not have FastAPI installed)
try:
    from .webapp import create_app, run_server, DETStateAPI
    WEBAPP_AVAILABLE = True
except ImportError:
    WEBAPP_AVAILABLE = False
    create_app = None
    run_server = None
    DETStateAPI = None
# LLM imports (optional - requires 'requests' package)
try:
    from .llm import DETLLMInterface, OllamaClient, DetIntentPacket, IntentType, DomainType
    from .memory import MemoryManager, MemoryDomain, MemoryEntry, DomainRouter, ContextWindow
    from .routing import (
        ModelConfig, ModelPool, ModelStatus, LLMRouter, RoutingResult, MultiModelInterface,
        DEFAULT_MODELS
    )
    from .dialogue import InternalDialogue, DialogueTurn, DialogueState, SomaticBridge
    from .sandbox import BashSandbox, FileOperations, CommandAnalyzer, RiskLevel, PermissionLevel
    from .tasks import TaskManager, Task, TaskStep, TaskStatus, TaskPriority
    from .timer import TimerSystem, ScheduledEvent, ScheduleType, setup_default_schedule
    from .executor import CodeExecutor, ExecutionSession, LanguageRunner, ErrorInterpreter
    from .emotional import (
        EmotionalIntegration, EmotionalMode, BehaviorModulation,
        RecoveryState, MultiSessionManager, SessionContext
    )
    from .training import (
        TrainingConfig, TrainingExample, TrainingJob, TrainingStatus,
        TrainingDataGenerator, LoRATrainer, MemoryRetuner, is_mlx_available
    )
    from .consolidation import (
        ConsolidationManager, ConsolidationConfig, ConsolidationState,
        ConsolidationPhase, ConsolidationCycle, IdleDetector, setup_consolidation
    )
    from .network import (
        MessageType, NodeType, NodeStatus, DETMessage, NodeInfo,
        Transport, ExternalNode, StubTransport, StubExternalNode,
        NetworkRegistry, create_stub_network
    )
    from .metrics import (
        MetricsCollector, MetricsSample, DETEvent, DETEventType, Profiler,
        create_metrics_collector, create_profiler
    )
    from .trainer import (
        DETTrainer, TrainingConfig as TrainerConfig, TrainingStats,
        TrainingDomain, CurriculumGenerator, WebContentFetcher,
        create_trainer, run_training
    )
    LLM_AVAILABLE = True
except ImportError:
    # LLM features not available without requests
    LLM_AVAILABLE = False
    DETLLMInterface = None
    OllamaClient = None
    DetIntentPacket = None
    IntentType = None
    DomainType = None
    MemoryManager = None
    MemoryDomain = None
    MemoryEntry = None
    DomainRouter = None
    ContextWindow = None
    ModelConfig = None
    ModelPool = None
    ModelStatus = None
    LLMRouter = None
    RoutingResult = None
    MultiModelInterface = None
    DEFAULT_MODELS = None
    InternalDialogue = None
    DialogueTurn = None
    DialogueState = None
    SomaticBridge = None
    BashSandbox = None
    FileOperations = None
    CommandAnalyzer = None
    RiskLevel = None
    PermissionLevel = None
    TaskManager = None
    Task = None
    TaskStep = None
    TaskStatus = None
    TaskPriority = None
    TimerSystem = None
    ScheduledEvent = None
    ScheduleType = None
    setup_default_schedule = None
    CodeExecutor = None
    ExecutionSession = None
    LanguageRunner = None
    ErrorInterpreter = None
    EmotionalIntegration = None
    EmotionalMode = None
    BehaviorModulation = None
    RecoveryState = None
    MultiSessionManager = None
    SessionContext = None
    TrainingConfig = None
    TrainingExample = None
    TrainingJob = None
    TrainingStatus = None
    TrainingDataGenerator = None
    LoRATrainer = None
    MemoryRetuner = None
    is_mlx_available = lambda: False
    ConsolidationManager = None
    ConsolidationConfig = None
    ConsolidationState = None
    ConsolidationPhase = None
    ConsolidationCycle = None
    IdleDetector = None
    setup_consolidation = None
    MessageType = None
    NodeType = None
    NodeStatus = None
    DETMessage = None
    NodeInfo = None
    Transport = None
    ExternalNode = None
    StubTransport = None
    StubExternalNode = None
    NetworkRegistry = None
    create_stub_network = None
    MetricsCollector = None
    MetricsSample = None
    DETEvent = None
    DETEventType = None
    Profiler = None
    create_metrics_collector = None
    create_profiler = None
    DETTrainer = None
    TrainerConfig = None
    TrainingStats = None
    TrainingDomain = None
    CurriculumGenerator = None
    WebContentFetcher = None
    create_trainer = None
    run_training = None

# Phase 7: Existence-Lang (optional - self-contained module)
try:
    from . import lang
    from .lang import (
        ExistenceRuntime, CreatureBase, KernelBase, Register, TokenReg,
        parse as parse_existence, transpile as transpile_existence
    )
    LANG_AVAILABLE = True
except ImportError:
    LANG_AVAILABLE = False
    lang = None
    ExistenceRuntime = None
    CreatureBase = None
    KernelBase = None
    Register = None
    TokenReg = None
    parse_existence = None
    transpile_existence = None

# Phase 9: DET-OS Kernel (optional - self-contained module)
try:
    from . import os as det_os
    DETOS_AVAILABLE = True
except ImportError:
    DETOS_AVAILABLE = False
    det_os = None

__version__ = "0.9.0"
__all__ = [
    # Core
    "DETCore", "DETParams", "DETDecision", "DETEmotion", "DETLayer",
    "SomaticType", "SomaticNode",
    # Harness (Phase 6.1)
    "HarnessController", "HarnessCLI", "HarnessEvent", "HarnessEventType",
    "Snapshot", "create_harness", "run_harness_cli",
    # Webapp (Phase 6.2)
    "create_app", "run_server", "DETStateAPI", "WEBAPP_AVAILABLE",
    # LLM
    "DETLLMInterface", "OllamaClient", "DetIntentPacket", "IntentType", "DomainType",
    # Memory
    "MemoryManager", "MemoryDomain", "MemoryEntry", "DomainRouter", "ContextWindow",
    # Routing (Phase 5.1)
    "ModelConfig", "ModelPool", "ModelStatus", "LLMRouter", "RoutingResult",
    "MultiModelInterface", "DEFAULT_MODELS",
    # Dialogue
    "InternalDialogue", "DialogueTurn", "DialogueState", "SomaticBridge",
    # Sandbox
    "BashSandbox", "FileOperations", "CommandAnalyzer", "RiskLevel", "PermissionLevel",
    # Tasks
    "TaskManager", "Task", "TaskStep", "TaskStatus", "TaskPriority",
    # Timer
    "TimerSystem", "ScheduledEvent", "ScheduleType", "setup_default_schedule",
    # Executor
    "CodeExecutor", "ExecutionSession", "LanguageRunner", "ErrorInterpreter",
    # Phase 4: Emotional Integration
    "EmotionalIntegration", "EmotionalMode", "BehaviorModulation",
    "RecoveryState", "MultiSessionManager", "SessionContext",
    # Phase 2.2: MLX Training
    "TrainingConfig", "TrainingExample", "TrainingJob", "TrainingStatus",
    "TrainingDataGenerator", "LoRATrainer", "MemoryRetuner", "is_mlx_available",
    # Phase 5.2: Sleep/Consolidation
    "ConsolidationManager", "ConsolidationConfig", "ConsolidationState",
    "ConsolidationPhase", "ConsolidationCycle", "IdleDetector", "setup_consolidation",
    # Phase 5.3: Network Integration (Preliminary)
    "MessageType", "NodeType", "NodeStatus", "DETMessage", "NodeInfo",
    "Transport", "ExternalNode", "StubTransport", "StubExternalNode",
    "NetworkRegistry", "create_stub_network",
    # Phase 6.4: Metrics and Logging
    "MetricsCollector", "MetricsSample", "DETEvent", "DETEventType", "Profiler",
    "create_metrics_collector", "create_profiler",
    # Trainer (Autonomous Training)
    "DETTrainer", "TrainerConfig", "TrainingStats", "TrainingDomain",
    "CurriculumGenerator", "WebContentFetcher", "create_trainer", "run_training",
    # Phase 7: Existence-Lang
    "lang", "LANG_AVAILABLE", "ExistenceRuntime", "CreatureBase", "KernelBase",
    "Register", "TokenReg", "parse_existence", "transpile_existence",
    # Phase 9: DET-OS Kernel
    "det_os", "DETOS_AVAILABLE",
    # Availability flags
    "LLM_AVAILABLE",
]
