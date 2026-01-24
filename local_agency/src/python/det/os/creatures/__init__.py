"""
DET-OS Creatures
================

Standard creature implementations for the DET-OS.

Each creature is defined in Existence-Lang (creatures.ex) and has a
Python wrapper for CLI/API integration.

Creatures:
    MemoryCreature   - Store and recall memories via bonds
    ToolCreature     - Execute commands in sandboxed environment
    ReasonerCreature - Chain-of-thought reasoning
    PlannerCreature  - Task decomposition and planning
"""

from .base import CreatureWrapper
from .memory import MemoryCreature, MemoryType, MemoryEntry, spawn_memory_creature
from .tool import ToolCreature, ExecutionResult, spawn_tool_creature
from .reasoner import ReasonerCreature, ReasoningChain, ReasoningStep, spawn_reasoner_creature
from .planner import PlannerCreature, Plan, PlanStep, spawn_planner_creature

__all__ = [
    # Base
    'CreatureWrapper',

    # Memory
    'MemoryCreature',
    'MemoryType',
    'MemoryEntry',
    'spawn_memory_creature',

    # Tool
    'ToolCreature',
    'ExecutionResult',
    'spawn_tool_creature',

    # Reasoner
    'ReasonerCreature',
    'ReasoningChain',
    'ReasoningStep',
    'spawn_reasoner_creature',

    # Planner
    'PlannerCreature',
    'Plan',
    'PlanStep',
    'spawn_planner_creature',
]
