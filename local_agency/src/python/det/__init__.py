"""
DET Local Agency - Python Interface
====================================

DET-native operating system where all logic runs as Existence-Lang creatures
executed via the EIS (Existence-Informed Substrate) virtual machine.

Architecture:
    User Terminal
         |
    TerminalCreature.ex
         |
    LLMCreature.ex -- ToolCreature.ex -- MemoryCreature.ex
         |
    EIS Interpreter (det.eis)
         |
    Substrate Layer (C/Metal)

Primary Entry Point:
    python det_os_boot.py [--gpu] [-v]

Modules:
    det.lang  - Existence-Lang compiler (parser, AST, bytecode)
    det.eis   - EIS virtual machine (interpreter, phases, registers)
    det.os    - DET-OS kernel and creatures
    det.metal - Metal GPU backend (macOS)
"""

__version__ = "0.20.0"

# =============================================================================
# Core Modules (always available)
# =============================================================================

# Existence-Lang compiler
try:
    from . import lang
    from .lang import (
        ExistenceRuntime,
        parse as parse_existence,
    )
    LANG_AVAILABLE = True
except ImportError:
    LANG_AVAILABLE = False
    lang = None
    ExistenceRuntime = None
    parse_existence = None

# EIS Virtual Machine
try:
    from . import eis
    from .eis import (
        EISVM, Lane, ExecutionState,
        CreatureRunner, CompiledCreatureData, CreatureState,
        PrimitiveRegistry, get_registry,
    )
    EIS_AVAILABLE = True
except ImportError:
    EIS_AVAILABLE = False
    eis = None
    EISVM = None
    Lane = None
    ExecutionState = None
    CreatureRunner = None
    CompiledCreatureData = None
    CreatureState = None
    PrimitiveRegistry = None
    get_registry = None

# DET-OS Kernel
try:
    from . import os as det_os
    DETOS_AVAILABLE = True
except ImportError:
    DETOS_AVAILABLE = False
    det_os = None

# =============================================================================
# Optional Modules
# =============================================================================

# Metal GPU Backend (macOS only)
try:
    from .metal import (
        MetalBackend, NodeArraysHelper, BondArraysHelper,
        NodeArrays, BondArrays, PredecodedProgram, SubstrateInstr,
        PHASE_READ, PHASE_PROPOSE, PHASE_CHOOSE, PHASE_COMMIT,
        LANE_OWNER_NONE, LANE_OWNER_NODE, LANE_OWNER_BOND,
        benchmark_cpu_vs_gpu,
    )
    METAL_AVAILABLE = MetalBackend.is_available()
except ImportError:
    METAL_AVAILABLE = False
    MetalBackend = None
    NodeArraysHelper = None
    BondArraysHelper = None
    NodeArrays = None
    BondArrays = None
    PredecodedProgram = None
    SubstrateInstr = None
    PHASE_READ = None
    PHASE_PROPOSE = None
    PHASE_CHOOSE = None
    PHASE_COMMIT = None
    LANE_OWNER_NONE = None
    LANE_OWNER_NODE = None
    LANE_OWNER_BOND = None
    benchmark_cpu_vs_gpu = None

# LLM Interface (for primitives)
try:
    from .llm import OllamaClient, DETLLMInterface
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    OllamaClient = None
    DETLLMInterface = None

# Web App (optional visualization)
try:
    from .webapp import create_app, run_server
    WEBAPP_AVAILABLE = True
except ImportError:
    WEBAPP_AVAILABLE = False
    create_app = None
    run_server = None

# =============================================================================
# Utility modules (kept for compatibility but may be deprecated)
# =============================================================================

# Sandbox for tool execution
try:
    from .sandbox import BashSandbox, CommandAnalyzer, RiskLevel
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    BashSandbox = None
    CommandAnalyzer = None
    RiskLevel = None

# Network (future phase)
try:
    from .network import NetworkRegistry, DETMessage, NodeInfo
    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False
    NetworkRegistry = None
    DETMessage = None
    NodeInfo = None

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",

    # Availability flags
    "LANG_AVAILABLE",
    "EIS_AVAILABLE",
    "DETOS_AVAILABLE",
    "METAL_AVAILABLE",
    "LLM_AVAILABLE",
    "WEBAPP_AVAILABLE",
    "SANDBOX_AVAILABLE",
    "NETWORK_AVAILABLE",

    # Core modules
    "lang",
    "eis",
    "det_os",

    # Existence-Lang
    "ExistenceRuntime",
    "parse_existence",

    # EIS VM
    "EISVM",
    "Lane",
    "ExecutionState",
    "CreatureRunner",
    "CompiledCreatureData",
    "CreatureState",
    "PrimitiveRegistry",
    "get_registry",

    # Metal GPU
    "MetalBackend",
    "NodeArraysHelper",
    "BondArraysHelper",
    "NodeArrays",
    "BondArrays",
    "PHASE_READ",
    "PHASE_PROPOSE",
    "PHASE_CHOOSE",
    "PHASE_COMMIT",
    "benchmark_cpu_vs_gpu",

    # LLM
    "OllamaClient",
    "DETLLMInterface",

    # Web App
    "create_app",
    "run_server",

    # Sandbox
    "BashSandbox",
    "CommandAnalyzer",
    "RiskLevel",

    # Network
    "NetworkRegistry",
    "DETMessage",
    "NodeInfo",
]
