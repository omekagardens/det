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
"""

from .core import DETCore, DETParams, DETDecision, DETEmotion, DETLayer
from .llm import DETLLMInterface, OllamaClient, DetIntentPacket, IntentType, DomainType
from .memory import MemoryManager, MemoryDomain, MemoryEntry, DomainRouter, ContextWindow
from .routing import (
    ModelConfig, ModelPool, ModelStatus, LLMRouter, RoutingResult, MultiModelInterface,
    DEFAULT_MODELS
)
from .dialogue import InternalDialogue, DialogueTurn, DialogueState
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

__version__ = "0.5.0"
__all__ = [
    # Core
    "DETCore", "DETParams", "DETDecision", "DETEmotion", "DETLayer",
    # LLM
    "DETLLMInterface", "OllamaClient", "DetIntentPacket", "IntentType", "DomainType",
    # Memory
    "MemoryManager", "MemoryDomain", "MemoryEntry", "DomainRouter", "ContextWindow",
    # Routing (Phase 5.1)
    "ModelConfig", "ModelPool", "ModelStatus", "LLMRouter", "RoutingResult",
    "MultiModelInterface", "DEFAULT_MODELS",
    # Dialogue
    "InternalDialogue", "DialogueTurn", "DialogueState",
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
]
