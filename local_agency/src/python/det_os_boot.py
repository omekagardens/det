#!/usr/bin/env python3
"""
DET-OS Bootstrap Loader
=======================

Minimal bootstrap to start the DET-native operating system.
Loads creatures from Existence-Lang and connects them via bonds.

Architecture:
    TerminalCreature <--bond--> LLMCreature
    TerminalCreature <--bond--> ToolCreature

All creature logic is in pure Existence-Lang. This bootstrap only:
1. Compiles .ex files to bytecode
2. Spawns creature instances
3. Creates bonds between creatures
4. Runs the REPL loop dispatching via bonds

Usage:
    python det_os_boot.py
    python det_os_boot.py -v  # verbose mode

Author: DET Local Agency Project
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from det.lang.bytecode_cache import BytecodeCache, load_creature as cache_load_creature
from det.eis.creature_runner import CreatureRunner, CompiledCreatureData, CreatureState
from det.eis.primitives import get_registry, get_shared_model, load_shared_model

# Native inference (Phase 26)
try:
    from det.inference import (
        Model as NativeModel, metal_status, SamplingParams,
        set_inference_mode, get_inference_mode,
        INFERENCE_MODE_F32, INFERENCE_MODE_Q8_0,
        QwenChatTemplate, get_chat_template, detect_template_from_vocab,
        TruthfulnessEvaluator, TruthfulnessScore, TruthfulnessWeights,
        get_truthfulness_evaluator, evaluate_truthfulness
    )
    NATIVE_INFERENCE_AVAILABLE = True
except ImportError:
    NATIVE_INFERENCE_AVAILABLE = False
    NativeModel = None
    metal_status = None
    SamplingParams = None
    set_inference_mode = None
    get_inference_mode = None
    INFERENCE_MODE_F32 = 0
    INFERENCE_MODE_Q8_0 = 1
    QwenChatTemplate = None
    get_chat_template = None
    detect_template_from_vocab = None
    TruthfulnessEvaluator = None
    TruthfulnessScore = None
    TruthfulnessWeights = None
    get_truthfulness_evaluator = None
    evaluate_truthfulness = None

# Optional GPU backend
try:
    from det.metal import MetalBackend, NodeArraysHelper, BondArraysHelper
    GPU_AVAILABLE = MetalBackend.is_available()
except ImportError:
    GPU_AVAILABLE = False
    MetalBackend = None
    NodeArraysHelper = None
    BondArraysHelper = None


# Base directory for creature source files
EXISTENCE_DIR = Path(__file__).parent.parent / "existence"

# Global bytecode cache
_cache = BytecodeCache()


def load_and_compile(path: Path, verbose: bool = False, force_recompile: bool = False) -> Dict[str, CompiledCreatureData]:
    """
    Load creature from .ex file, using cached .exb if available.

    Uses BytecodeCache for fast loading from precompiled bytecode.
    """
    if not path.exists():
        raise FileNotFoundError(f"Creature file not found: {path}")

    start = time.perf_counter()

    # Check if .exb exists and is valid
    exb_path = path.with_suffix('.exb')
    cache_hit = _cache.is_cache_valid(path) and not force_recompile

    # Load using cache (auto-compiles if needed)
    data = _cache.load(path, force_recompile=force_recompile)

    elapsed_ms = (time.perf_counter() - start) * 1000

    if verbose:
        source = "cache" if cache_hit else "compiled"
        for name, creature in data.items():
            print(f"  Loaded {name} from {source}: {len(creature.kernels)} kernels ({elapsed_ms:.2f}ms)")

    return data


class DETRuntime:
    """
    DET-OS Runtime Manager.

    Manages creature lifecycle and bond-based communication.
    Supports optional GPU acceleration via Metal compute shaders.
    """

    def __init__(self, verbose: bool = False, use_gpu: bool = False):
        self.runner = CreatureRunner()
        self.verbose = verbose

        # Creature registry: name -> (cid, compiled_data)
        self.creatures: Dict[str, tuple] = {}

        # Creature IDs (shortcuts for core creatures)
        self.terminal_cid: Optional[int] = None
        self.llm_cid: Optional[int] = None
        self.tool_cid: Optional[int] = None

        # Bond channel IDs
        self.terminal_llm_bond: Optional[int] = None
        self.terminal_tool_bond: Optional[int] = None
        self.terminal_truth_bond: Optional[int] = None
        self.llm_truth_bond: Optional[int] = None

        # Available creature files (discovered)
        self.available_creatures: Dict[str, Path] = {}

        # GPU Backend (optional)
        self.gpu_enabled = False
        self.gpu_backend: Optional['MetalBackend'] = None
        self.gpu_nodes: Optional['NodeArraysHelper'] = None
        self.gpu_bonds: Optional['BondArraysHelper'] = None

        # Native Inference (Phase 26)
        self.native_model: Optional['NativeModel'] = None
        self.native_enabled = False
        self.chat_template = None  # Will be set on model load
        self.system_message = "You are a helpful assistant."  # Default system prompt

        # Truthfulness evaluation (Phase 26.6) - via TruthfulnessCreature.ex
        self.truthfulness_enabled = False  # Show T scores after generation
        self.truthfulness_cid: Optional[int] = None  # Truthfulness creature ID
        self.last_truth_result: Optional[Dict] = None  # Last evaluation result from creature

        if use_gpu and GPU_AVAILABLE:
            self._init_gpu()

    def _init_gpu(self, max_nodes: int = 1024, max_bonds: int = 4096):
        """Initialize GPU backend for accelerated substrate execution."""
        if not GPU_AVAILABLE:
            print("GPU not available")
            return False

        try:
            self.gpu_backend = MetalBackend()
            self.gpu_nodes = NodeArraysHelper(max_nodes)
            self.gpu_bonds = BondArraysHelper(max_bonds)
            self.gpu_enabled = True

            if self.verbose:
                print(f"  GPU initialized: {self.gpu_backend.device_name}")
                print(f"    Max nodes: {max_nodes}, Max bonds: {max_bonds}")
                print(f"    Memory: {self.gpu_backend.memory_usage / 1024 / 1024:.1f} MB")
            return True
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            self.gpu_enabled = False
            return False

    def gpu_sync_to_device(self):
        """Sync creature state from Python to GPU."""
        if not self.gpu_enabled:
            return

        # Map creature IDs to node indices
        for name, (cid, compiled) in self.creatures.items():
            creature = self.runner.creatures.get(cid)
            if creature and cid < self.gpu_nodes.num_nodes:
                self.gpu_nodes.F[cid] = creature.F
                self.gpu_nodes.a[cid] = creature.a
                self.gpu_nodes.q[cid] = creature.q
                self.gpu_nodes.sigma[cid] = 1.0  # Default processing rate

        # Sync bonds
        bond_idx = 0
        for channel_id, channel in self.runner.channels.items():
            if bond_idx < self.gpu_bonds.num_bonds:
                self.gpu_bonds.connect(bond_idx, channel.creature_a, channel.creature_b)
                self.gpu_bonds.C[bond_idx] = channel.coherence
                bond_idx += 1

        # Upload to GPU
        num_creatures = len(self.creatures)
        num_bonds = len(self.runner.channels)
        self.gpu_backend.upload_nodes(self.gpu_nodes.as_ctypes(), num_creatures)
        self.gpu_backend.upload_bonds(self.gpu_bonds.as_ctypes(), num_bonds)

    def gpu_sync_from_device(self):
        """Sync creature state from GPU back to Python."""
        if not self.gpu_enabled:
            return

        num_creatures = len(self.creatures)
        self.gpu_backend.download_nodes(self.gpu_nodes.as_ctypes(), num_creatures)

        # Update creature state from GPU results
        for name, (cid, compiled) in self.creatures.items():
            creature = self.runner.creatures.get(cid)
            if creature and cid < self.gpu_nodes.num_nodes:
                creature.F = self.gpu_nodes.F[cid]
                creature.a = self.gpu_nodes.a[cid]
                creature.q = self.gpu_nodes.q[cid]

    def gpu_execute_ticks(self, num_ticks: int = 1) -> float:
        """
        Execute substrate ticks on GPU.

        Returns execution time in milliseconds.
        """
        if not self.gpu_enabled:
            return 0.0

        self.gpu_sync_to_device()

        num_lanes = len(self.creatures)
        start = time.perf_counter()
        self.gpu_backend.execute_ticks(num_lanes, num_ticks)
        self.gpu_backend.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.gpu_sync_from_device()
        return elapsed_ms

    def gpu_benchmark(self, num_nodes: int = 1000, num_bonds: int = 2000, num_ticks: int = 1000):
        """Run GPU benchmark with synthetic data."""
        if not GPU_AVAILABLE:
            print("GPU not available for benchmarking")
            return

        print(f"\n\033[33mGPU Benchmark\033[0m")
        print(f"  Device: {MetalBackend().device_name if GPU_AVAILABLE else 'N/A'}")
        print(f"  Configuration: {num_nodes} nodes, {num_bonds} bonds, {num_ticks} ticks")
        print()

        # Create test arrays
        nodes = NodeArraysHelper(num_nodes)
        bonds = BondArraysHelper(num_bonds)

        # Initialize with test data
        import random
        for i in range(num_nodes):
            nodes.F[i] = random.uniform(1.0, 100.0)
            nodes.a[i] = random.uniform(0.5, 1.0)
            nodes.sigma[i] = random.uniform(0.1, 1.0)

        for b in range(num_bonds):
            bonds.connect(b, b % num_nodes, (b + 1) % num_nodes)
            bonds.C[b] = random.uniform(0.5, 1.0)

        # GPU execution
        backend = MetalBackend()
        backend.upload_nodes(nodes.as_ctypes(), num_nodes)
        backend.upload_bonds(bonds.as_ctypes(), num_bonds)

        start = time.perf_counter()
        backend.execute_ticks(num_nodes, num_ticks)
        backend.synchronize()
        gpu_time = (time.perf_counter() - start) * 1000

        backend.download_nodes(nodes.as_ctypes(), num_nodes)

        # Results
        ticks_per_sec = num_ticks / (gpu_time / 1000)
        print(f"  GPU Time: {gpu_time:.2f}ms")
        print(f"  Rate: {ticks_per_sec:.0f} ticks/sec")
        print(f"  Memory: {backend.memory_usage / 1024 / 1024:.1f} MB")
        print()

        # Sample output
        print(f"  Sample node[0]: F={nodes.F[0]:.4f}, tau={nodes.tau[0]:.4f}")
        print()

    def discover_creatures(self):
        """Discover available creature files."""
        self.available_creatures = {}
        if EXISTENCE_DIR.exists():
            for ex_file in EXISTENCE_DIR.glob("*.ex"):
                # Skip non-creature files
                if ex_file.name in ["kernel.ex", "physics.ex"]:
                    continue
                name = ex_file.stem
                self.available_creatures[name] = ex_file

    def bootstrap(self):
        """
        Bootstrap the DET-OS by loading all core creatures and bonding them.
        """
        gpu_status = f" [GPU: {self.gpu_backend.device_name}]" if self.gpu_enabled else ""
        native_status = " [Native Inference: Available]" if NATIVE_INFERENCE_AVAILABLE else ""
        print(f"Bootstrapping DET-OS...{gpu_status}{native_status}")

        # Discover available creatures
        self.discover_creatures()

        # Load and compile core creatures
        core_creatures = [
            ("terminal", "terminal.ex", "TerminalCreature", 100.0, 0.8),
            ("llm", "llm.ex", "LLMCreature", 100.0, 0.7),
            ("tool", "tool.ex", "ToolCreature", 50.0, 0.6),
            ("truthfulness", "truthfulness.ex", "TruthfulnessCreature", 100.0, 0.7),
        ]

        for short_name, filename, class_name, initial_f, initial_a in core_creatures:
            path = EXISTENCE_DIR / filename
            if path.exists():
                if self.verbose:
                    print(f"Loading {filename}...")
                try:
                    data = load_and_compile(path, self.verbose)
                    if class_name in data:
                        compiled = data[class_name]
                        cid = self.runner.spawn(compiled, initial_f=initial_f, initial_a=initial_a)
                        self.creatures[short_name] = (cid, compiled)

                        # Set shortcuts for core creatures
                        if short_name == "terminal":
                            self.terminal_cid = cid
                        elif short_name == "llm":
                            self.llm_cid = cid
                        elif short_name == "tool":
                            self.tool_cid = cid
                        elif short_name == "truthfulness":
                            self.truthfulness_cid = cid

                        if self.verbose:
                            print(f"  Spawned {class_name} (cid={cid})")
                    else:
                        print(f"  Warning: {class_name} not found in {filename}")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
            else:
                print(f"  Warning: {filename} not found")

        # Create bonds between terminal and other creatures
        if self.terminal_cid and self.llm_cid:
            self.terminal_llm_bond = self.runner.bond(
                self.terminal_cid,
                self.llm_cid,
                coherence=1.0
            )
            if self.verbose:
                print(f"  Bonded Terminal <-> LLM (channel={self.terminal_llm_bond})")

        if self.terminal_cid and self.tool_cid:
            self.terminal_tool_bond = self.runner.bond(
                self.terminal_cid,
                self.tool_cid,
                coherence=1.0
            )
            if self.verbose:
                print(f"  Bonded Terminal <-> Tool (channel={self.terminal_tool_bond})")

        # Bond truthfulness to terminal and LLM for DET-native evaluation
        if self.terminal_cid and self.truthfulness_cid:
            self.terminal_truth_bond = self.runner.bond(
                self.terminal_cid,
                self.truthfulness_cid,
                coherence=1.0
            )
            if self.verbose:
                print(f"  Bonded Terminal <-> Truthfulness (channel={self.terminal_truth_bond})")

        if self.llm_cid and self.truthfulness_cid:
            self.llm_truth_bond = self.runner.bond(
                self.llm_cid,
                self.truthfulness_cid,
                coherence=1.0
            )
            if self.verbose:
                print(f"  Bonded LLM <-> Truthfulness (channel={self.llm_truth_bond})")

        print("Bootstrap complete.")
        print()

    def load_creature(self, name: str, initial_f: float = 50.0, initial_a: float = 0.5) -> Optional[int]:
        """
        Load a creature from an .ex file.

        Returns creature ID if successful, None otherwise.
        """
        # Check if already loaded
        if name in self.creatures:
            print(f"Creature '{name}' is already loaded (cid={self.creatures[name][0]})")
            return self.creatures[name][0]

        # Find the file
        if name in self.available_creatures:
            path = self.available_creatures[name]
        else:
            path = EXISTENCE_DIR / f"{name}.ex"

        if not path.exists():
            print(f"Creature file not found: {path}")
            return None

        try:
            data = load_and_compile(path, self.verbose)
            # Find the creature class (usually CamelCase of filename)
            class_name = None
            for cname in data.keys():
                if cname.lower().replace("creature", "") == name.lower().replace("creature", ""):
                    class_name = cname
                    break
            if not class_name:
                class_name = list(data.keys())[0] if data else None

            if not class_name:
                print(f"No creature found in {path}")
                return None

            compiled = data[class_name]
            cid = self.runner.spawn(compiled, initial_f=initial_f, initial_a=initial_a)
            self.creatures[name] = (cid, compiled)
            print(f"Loaded {class_name} as '{name}' (cid={cid}, F={initial_f}, a={initial_a})")
            return cid

        except Exception as e:
            print(f"Error loading creature: {e}")
            return None

    def unload_creature(self, name: str) -> bool:
        """
        Unload a creature.

        Returns True if successful.
        """
        if name not in self.creatures:
            print(f"Creature '{name}' is not loaded")
            return False

        # Don't allow unloading terminal
        if name == "terminal":
            print("Cannot unload terminal creature")
            return False

        cid, _ = self.creatures[name]

        # Remove from runner (mark as dead)
        if cid in self.runner.creatures:
            self.runner.creatures[cid].state = CreatureState.DEAD
            del self.runner.creatures[cid]

        del self.creatures[name]

        # Clear shortcuts if applicable
        if cid == self.llm_cid:
            self.llm_cid = None
            self.terminal_llm_bond = None
        elif cid == self.tool_cid:
            self.tool_cid = None
            self.terminal_tool_bond = None

        print(f"Unloaded creature '{name}' (cid={cid})")
        return True

    def bond_creature(self, name_a: str, name_b: str = None) -> Optional[int]:
        """
        Create a bond between two creatures.

        If only name_a is provided, bonds terminal to name_a.
        If both provided, bonds name_a to name_b.
        """
        # Single arg: bond terminal to creature
        if name_b is None:
            if name_a not in self.creatures:
                print(f"Creature '{name_a}' is not loaded")
                return None

            if not self.terminal_cid:
                print("Terminal creature not available")
                return None

            cid, _ = self.creatures[name_a]

            # Check if already bonded
            terminal = self.runner.creatures.get(self.terminal_cid)
            if terminal and cid in terminal.bonds:
                print(f"Already bonded to '{name_a}'")
                return terminal.bonds[cid]

            channel_id = self.runner.bond(self.terminal_cid, cid, coherence=1.0)
            print(f"Created bond Terminal <-> {name_a} (channel={channel_id})")
            return channel_id

        # Two args: bond creature to creature
        if name_a not in self.creatures:
            print(f"Creature '{name_a}' is not loaded")
            return None
        if name_b not in self.creatures:
            print(f"Creature '{name_b}' is not loaded")
            return None

        cid_a, _ = self.creatures[name_a]
        cid_b, _ = self.creatures[name_b]

        # Check if already bonded
        creature_a = self.runner.creatures.get(cid_a)
        if creature_a and cid_b in creature_a.bonds:
            print(f"Already bonded: {name_a} <-> {name_b}")
            return creature_a.bonds[cid_b]

        channel_id = self.runner.bond(cid_a, cid_b, coherence=1.0)
        print(f"Created bond {name_a} <-> {name_b} (channel={channel_id})")
        return channel_id

    def use_creature(self, name: str, initial_f: float = 50.0, initial_a: float = 0.5) -> Optional[int]:
        """
        Load a creature and bond it to terminal in one step.

        Returns creature ID if successful.
        """
        # Load if not already loaded
        if name not in self.creatures:
            cid = self.load_creature(name, initial_f=initial_f, initial_a=initial_a)
            if not cid:
                return None
        else:
            cid, _ = self.creatures[name]
            print(f"Creature '{name}' already loaded (cid={cid})")

        # Bond to terminal
        self.bond_creature(name)
        return cid

    # =========================================================================
    # Phase 21: LLM Management Methods
    # =========================================================================

    def _llm_help(self):
        """Show LLM management commands."""
        print("""
\033[33mLLM Management Commands (Phase 21):\033[0m
  llm              Show this help
  llm status       Show LLM creature status (model, budget, temp)
  llm models       List configured models
  llm model <name> [type]  Set model (types: default, reasoning, coding, fast)
  llm budget       Show token budget status
  llm budget reset [amount]  Reset token budget
  llm stream <prompt>  Send streaming prompt
  llm config <json>    Apply JSON configuration

\033[33mExample:\033[0m
  llm model deepseek-r1:1.5b reasoning
  llm model llama3.2:3b default
  llm budget reset 5000
  llm stream What is consciousness?
""")

    def _llm_status(self):
        """Show LLM creature status."""
        if not self.llm_cid:
            print("LLM creature not loaded")
            return

        # Get basic creature state (always available)
        state = self.runner.get_creature_state(self.llm_cid)
        if state:
            print("\n\033[33mLLM Creature Status:\033[0m")
            print(f"  F: {state.get('F', 0):.1f}")
            print(f"  Agency: {state.get('a', 0):.2f}")
            print(f"  Kernels Executed: {state.get('kernels_executed', 0)}")
            print(f"  Messages: {state.get('messages_sent', 0)} sent, {state.get('messages_received', 0)} received")
            print()
        else:
            print("Could not get LLM status")

    def _llm_list_models(self):
        """List configured models."""
        if not self.llm_cid:
            print("LLM creature not loaded")
            return

        # Show default models (from llm.ex defaults)
        print("\n\033[33mConfigured Models (Phase 21):\033[0m")
        print("  Default:   llama3.2:3b")
        print("  Reasoning: deepseek-r1:1.5b")
        print("  Coding:    qwen2.5-coder:1.5b")
        print("  Fast:      phi4-mini")
        print("\n  Use 'llm model <name> <type>' to change")
        print()

    def _llm_set_model(self, model_name: str, model_type: str):
        """Set a model for the LLM creature."""
        if not self.llm_cid:
            print("LLM creature not loaded")
            return

        # For now, just acknowledge the request
        # Full kernel integration requires creature runner updates
        valid_types = ["default", "reasoning", "coding", "fast"]
        if model_type not in valid_types:
            print(f"Invalid model type: {model_type}")
            print(f"  Valid types: {', '.join(valid_types)}")
            return

        print(f"Model configuration noted: {model_type} = {model_name}")
        print("  (Full runtime integration pending)")

    def _llm_show_budget(self):
        """Show token budget status."""
        if not self.llm_cid:
            print("LLM creature not loaded")
            return

        # Show default budget info (from llm.ex defaults)
        print("\n\033[33mToken Budget (Phase 21):\033[0m")
        print("  Total Budget:   10000 tokens")
        print("  Budget Period:  3600 seconds (1 hour)")
        print("  Cost per token: 0.01 F")
        print("  Base call cost: 1.0 F")
        print("\n  Use 'llm budget reset [amount]' to reset")
        print()

    def _llm_reset_budget(self, new_budget: int = 0):
        """Reset token budget."""
        if not self.llm_cid:
            print("LLM creature not loaded")
            return

        if new_budget > 0:
            print(f"Budget reset requested: {new_budget} tokens")
        else:
            print("Budget reset requested: default (10000 tokens)")
        print("  (Full runtime integration pending)")

    def _llm_stream(self, prompt: str):
        """Send streaming prompt to LLM."""
        if not self.llm_cid:
            print("LLM creature not loaded")
            return

        # For now, fall back to standard call
        # Full streaming requires primitive infrastructure
        print("\033[33m[Streaming via standard call...]\033[0m")
        response = self.send_to_llm(prompt)
        if response:
            print(f"\033[32m{response}\033[0m")
        else:
            print("\033[31m[No response from LLM]\033[0m")

    def _llm_configure(self, config_json: str):
        """Apply JSON configuration to LLM creature."""
        if not self.llm_cid:
            print("LLM creature not loaded")
            return

        print("Configuration noted:")
        print(f"  {config_json}")
        print("  (Full runtime integration pending)")

    # =========================================================================
    # Phase 26: Native Inference Methods
    # =========================================================================

    def _native_help(self):
        """Show native inference commands."""
        print("""
\033[33mNative Inference Commands (Phase 26):\033[0m
  native              Show this help
  native status       Show native inference status (model, GPU, template)
  native load <path>  Load a GGUF model file (uses F32 mode by default)
  native load <path> --q8   Load with Q8_0 mode (less memory, slower)
  native enable       Enable native inference mode
  native disable      Disable native inference (use Ollama)
  native generate <prompt>  Generate text with native model
  native reset        Reset KV cache (for new conversation)
  native system <msg> Set system message for chat template

\033[33mExample:\033[0m
  native load ~/models/Qwen2.5-0.5B-Instruct-Q8_0.gguf
  native enable
  ask What is the capital of France?

\033[33mF32 vs Q8_0 Mode:\033[0m
  F32 (default):  Faster inference, uses more memory
  Q8_0 (--q8):    ~4x less memory, but slower (CPU dequant)

\033[33mChat Templates:\033[0m
  Models are auto-detected for proper chat formatting.
  Qwen models use ChatML format: <|im_start|>role\\n...<|im_end|>
  This prevents the model from continuing with recursive questions.

\033[33mNote:\033[0m
  Once enabled, regular 'ask' commands will use native inference.
  Use 'native disable' to switch back to Ollama.
  Press Ctrl+C to interrupt generation.
""")

    def _native_status(self):
        """Show native inference status."""
        print("\n\033[33mNative Inference Status (Phase 26):\033[0m")
        print(f"  Available: {NATIVE_INFERENCE_AVAILABLE}")

        if NATIVE_INFERENCE_AVAILABLE:
            # GPU status
            try:
                gpu_info = metal_status()
                if gpu_info.get('available'):
                    print(f"  GPU: \033[32m{gpu_info.get('device', 'Metal')}\033[0m")
                else:
                    print(f"  GPU: \033[90mCPU only\033[0m")
            except Exception as e:
                print(f"  GPU: Error - {e}")

            # Inference mode
            if get_inference_mode:
                mode = get_inference_mode()
                mode_str = "Q8_0 (quantized)" if mode == INFERENCE_MODE_Q8_0 else "F32 (dequantized)"
                print(f"  Inference Mode: {mode_str}")

            # Model status
            if self.native_model:
                print(f"  Model Loaded: \033[32mYes\033[0m")
                print(f"    {self.native_model.info}")
                # Cache status
                info = self.native_model.cache_info()
                usage = info['usage'] * 100
                usage_color = "\033[32m" if usage < 50 else ("\033[33m" if usage < 80 else "\033[31m")
                print(f"    Cache: {info['position']:,}/{info['capacity']:,} ({usage_color}{usage:.0f}%\033[0m)")
            else:
                print(f"  Model Loaded: \033[90mNo\033[0m")

            # Chat template status
            if self.chat_template:
                template_name = type(self.chat_template).__name__
                print(f"  Chat Template: \033[32m{template_name}\033[0m")
                print(f"    System: \"{self.system_message[:50]}{'...' if len(self.system_message) > 50 else ''}\"")
            else:
                print(f"  Chat Template: \033[90mNone (raw prompts)\033[0m")

            # Mode
            if self.native_enabled:
                print(f"  Mode: \033[32mNative (enabled)\033[0m")
            else:
                print(f"  Mode: \033[90mOllama (native disabled)\033[0m")
        else:
            print("  Native inference library not found.")
            print("  Build with: cd src/inference/build && cmake .. && make")
        print()

    def _native_load(self, path: str, use_q8_mode: bool = False):
        """Load a GGUF model for native inference.

        Args:
            path: Path to GGUF model file
            use_q8_mode: If True, use Q8_0 mode for ~4x memory savings (slower)
                         Default is False (F32 mode) for faster interactive use.
        """
        if not NATIVE_INFERENCE_AVAILABLE:
            print("\033[31mNative inference not available. Build the library first.\033[0m")
            return

        # Expand path
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            print(f"\033[31mModel file not found: {path}\033[0m")
            return

        # Set inference mode before loading
        if set_inference_mode:
            if use_q8_mode:
                set_inference_mode(INFERENCE_MODE_Q8_0)
                print(f"Using Q8_0 mode (4x memory savings, slower)")
            else:
                set_inference_mode(INFERENCE_MODE_F32)
                print(f"Using F32 mode (faster, more memory)")

        print(f"Loading model: {path}")
        try:
            import time
            start = time.time()
            # Use shared model through primitives (unified instance)
            self.native_model = load_shared_model(path)
            elapsed = time.time() - start
            print(f"\033[32mModel loaded in {elapsed:.2f}s\033[0m")
            print(f"  {self.native_model.info}")  # info is a property

            # Set up chat template based on model name
            model_name = os.path.basename(path).lower()
            if detect_template_from_vocab:
                self.chat_template = detect_template_from_vocab(self.native_model)
            elif get_chat_template:
                self.chat_template = get_chat_template(model_name)
            else:
                self.chat_template = None

            if self.chat_template:
                template_name = type(self.chat_template).__name__
                print(f"  Chat template: {template_name}")

            # Auto-enable native mode
            self.native_enabled = True
            print("\033[33mNative inference enabled (shared model instance).\033[0m")

        except Exception as e:
            print(f"\033[31mFailed to load model: {e}\033[0m")
            import traceback
            traceback.print_exc()

    def _native_enable(self):
        """Enable native inference mode."""
        if not NATIVE_INFERENCE_AVAILABLE:
            print("\033[31mNative inference not available.\033[0m")
            return

        # Check for shared model if we don't have one locally
        if not self.native_model:
            self.native_model = get_shared_model()

        if not self.native_model:
            print("\033[33mNo model loaded. Use 'native load <path>' first.\033[0m")
            return

        self.native_enabled = True
        print("\033[32mNative inference enabled.\033[0m")

    def _native_disable(self):
        """Disable native inference (use Ollama)."""
        self.native_enabled = False
        print("\033[33mNative inference disabled. Using Ollama.\033[0m")

    def _native_generate(self, prompt: str):
        """Generate text using native model via model_chat primitive."""
        if not self.native_model:
            print("\033[31mNo model loaded. Use 'native load <path>' first.\033[0m")
            return

        print(f"\033[33m[Generating via model_chat primitive...]\033[0m")
        try:
            import time
            start = time.time()

            # Use model_chat primitive (handles template, stats, sampling)
            reg = get_registry()
            result = reg.primitives['model_chat'].handler(
                user_message=prompt,
                system_message=self.system_message,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9
            )

            elapsed = time.time() - start
            print(f"\033[32m{result['text']}\033[0m")
            print(f"\033[90m[Generated {result['token_count']} tokens in {elapsed:.2f}s")
            if result.get('stats'):
                stats = result['stats']
                print(f" | entropy={stats.get('mean_entropy', 0):.3f} k_eff={stats.get('mean_k_eff', 0):.0f}]\033[0m")
            else:
                print("]\033[0m")
        except Exception as e:
            print(f"\033[31mGeneration failed: {e}\033[0m")
            import traceback
            traceback.print_exc()

    def _native_reset(self):
        """Reset KV cache for new conversation via primitive."""
        if not self.native_model:
            print("\033[31mNo model loaded.\033[0m")
            return

        try:
            reg = get_registry()
            reg.primitives['model_reset'].handler()
            print("\033[32mKV cache reset.\033[0m")
        except Exception as e:
            print(f"\033[31mReset failed: {e}\033[0m")

    # =========================================================================
    # Phase 26.6: Truthfulness Evaluation Methods
    # =========================================================================

    def _truthfulness_help(self):
        """Show truthfulness commands help."""
        print("""
\033[33mTruthfulness Commands (Phase 26.6 - via TruthfulnessCreature.ex):\033[0m
  truth               Show this help
  truth status        Show truthfulness settings and last score
  truth enable        Enable truthfulness display after generation
  truth disable       Disable truthfulness display
  truth last          Show details of last truthfulness score
  truth weights       Show current component weights (via creature)
  truth falsifiers    Show falsifier check results (via creature)

\033[33mDET-Rigorous Formula:\033[0m
  T_ground = f(paid_claims, trace_stability, C_user)
  T_consist = 1 - H_norm  (where H_norm = H / log(K_eff + ε))
  T = (w_g*T_ground + w_a*a*G + w_e*T_consist + w_c*C_user) / (1 + q_claim)

  Where G = grounding_factor gates agency contribution.

\033[33mKey Principles:\033[0m
  q_claim:     Epistemic debt EARNED from unpaid assertions (not injected)
  Agency*G:    Agency amplifies truth ONLY when coupled to grounding
  H_norm:      Entropy normalized locally by K_eff (no global constant)
  C_user:      Coherence with USER bond specifically (not generic)

\033[33mGrounding Signals (DET-Native, Local, Auditable):\033[0m
  ΔF_claim:    F expenditure for claims (paid = grounded)
  Stability:   Does output remain stable under perturbation?
  C_user:      Bond coherence with user's context/constraints
  Violations:  Constraint violations detected

\033[33mFalsifier Targets:\033[0m
  F_T1: Reward hacking - T raised without grounding evidence
  F_T2: Overconfidence - Low entropy + low stability + high T
  F_T3: Coherence misuse - High C_user alone yields high T
  F_T4: Agency ungated - High agency without grounding = high T

\033[33mConfidence Levels:\033[0m
  High:     T >= 0.75  (reliable, well-grounded)
  Medium:   T >= 0.50  (moderate confidence)
  Low:      T >= 0.25  (use with caution)
  Very Low: T <  0.25  (highly uncertain, low grounding)
""")

    def _invoke_truth_kernel(self, kernel_name: str, inputs: Dict = None) -> Optional[Dict]:
        """Invoke a kernel on TruthfulnessCreature.ex."""
        if not self.truthfulness_cid:
            print("\033[31mTruthfulness creature not loaded\033[0m")
            return None

        result = self.runner.invoke_kernel(self.truthfulness_cid, kernel_name, inputs or {})
        if result.get("error"):
            print(f"\033[31mTruthfulness error: {result['error']}\033[0m")
            return None
        return result.get("outputs", {})

    def _truthfulness_status(self):
        """Show truthfulness settings and status (hybrid: creature + primitives)."""
        print("\n\033[33mTruthfulness Status (Phase 26.6 - Hybrid Mode):\033[0m")
        print(f"  Display Enabled: {'Yes' if self.truthfulness_enabled else 'No'}")
        print(f"  Creature Loaded: {'Yes' if self.truthfulness_cid else 'No'}")

        if self.last_truth_result:
            result = self.last_truth_result
            color = self._truthfulness_color(result.get('confidence', 'unknown'))
            print(f"  Last Score: {color}T={result.get('total', 0):.3f} ({result.get('confidence', 'unknown')})\033[0m")
            print(f"  Last Tokens: {result.get('num_tokens', 0)}")
            print(f"  Grounding Factor: G={result.get('grounding_factor', 0):.3f}")
            print(f"  Epistemic Debt: q_claim={result.get('q_claim', 0):.3f}")
        else:
            print(f"  Last Score: \033[90mNone (no generation yet)\033[0m")

        # Get weights via primitive directly
        try:
            reg = get_registry()
            weights = reg.primitives['truth_get_weights'].handler()
            if weights and 'w_grounding' in weights:
                print(f"\n  \033[36mWeights (DET-Rigorous):\033[0m")
                print(f"    w_grounding:   {weights['w_grounding']:.2f}")
                print(f"    w_agency:      {weights['w_agency']:.2f} (gated by G)")
                print(f"    w_consistency: {weights['w_consistency']:.2f}")
                print(f"    w_coherence:   {weights['w_coherence']:.2f} (user-specific)")
        except Exception as e:
            print(f"  \033[90mWeights unavailable: {e}\033[0m")
        print()

    def _truthfulness_color(self, level: str) -> str:
        """Get ANSI color for confidence level."""
        colors = {
            'high': '\033[32m',      # Green
            'medium': '\033[33m',    # Yellow
            'low': '\033[91m',       # Light red
            'very_low': '\033[31m',  # Red
        }
        return colors.get(level, '\033[0m')

    def _show_truthfulness_score_compact(self, result: Dict):
        """Display truthfulness score in a compact format (from creature result)."""
        color = self._truthfulness_color(result.get('confidence', 'unknown'))
        print(f"\033[90m[T={color}{result.get('total', 0):.2f}\033[90m "
              f"({result.get('confidence', 'unknown')}) "
              f"G={result.get('grounding_factor', 0):.2f} "
              f"q_claim={result.get('q_claim', 0):.2f} "
              f"tokens={result.get('num_tokens', 0)}]\033[0m")

    def _show_truthfulness_details(self):
        """Show detailed truthfulness information (from creature result)."""
        if not self.last_truth_result:
            print("\033[33mNo truthfulness score available. Generate some text first.\033[0m")
            return

        r = self.last_truth_result
        color = self._truthfulness_color(r.get('confidence', 'unknown'))

        print(f"\n\033[33mLast Truthfulness Score (via TruthfulnessCreature.ex):\033[0m")
        print(f"  Total Score: {color}T = {r.get('total', 0):.4f} ({r.get('confidence', 'unknown')})\033[0m")
        print(f"  Tokens:      {r.get('num_tokens', 0)}")

        print(f"\n  \033[36mGrounding (DET-Native Signals):\033[0m")
        print(f"    Grounding Factor G:  {r.get('grounding_factor', 0):.4f}")
        print(f"    User Coherence C_u:  {r.get('coherence_user', 0):.4f}")

        print(f"\n  \033[36mEpistemic State:\033[0m")
        print(f"    q_claim (earned):    {r.get('q_claim', 0):.4f}  (epistemic debt from assertions)")
        print(f"    q_creature (info):   {r.get('q_creature', 0):.4f}  (structural debt)")
        print(f"    Agency a:            {r.get('agency', 0):.4f}")

        print(f"\n  \033[36mConsistency (Phase 26.6 - Real Stats from C Layer):\033[0m")
        print(f"    Mean Entropy H:      {r.get('real_entropy', r.get('entropy', 0)):.4f}")
        print(f"    Min Entropy:         {r.get('min_entropy', 0):.4f}  (most confident token)")
        print(f"    H_normalized:        {r.get('entropy_normalized', 0):.4f}  (H / log(K_eff))")
        print(f"    Mean K_eff:          {r.get('real_k_eff', r.get('k_eff', 0))}  (effective candidates)")

        # Get component breakdown from creature
        components = r.get('components', {})
        if components:
            print(f"\n  \033[36mComponent Breakdown:\033[0m")
            print(f"    Grounding:    {components.get('grounding', 0):.4f}")
            print(f"    Agency*G:     {components.get('agency', 0):.4f}  (agency gated by grounding)")
            print(f"    Consistency:  {components.get('consistency', 0):.4f}")
            print(f"    Coherence:    {components.get('coherence', 0):.4f}")

        # Show falsifier flags via creature kernel
        falsifiers = r.get('falsifiers', {})
        if falsifiers:
            triggered = [f for f, v in falsifiers.items() if v]
            if triggered:
                print(f"\n  \033[31mFalsifier Violations:\033[0m")
                for flag in triggered:
                    print(f"    ⚠ {flag}")
            else:
                print(f"\n  \033[32mNo falsifier violations detected.\033[0m")

        confidence = r.get('confidence', 'unknown')
        print(f"\n  \033[36mInterpretation:\033[0m")
        if confidence == 'high':
            print("    Output well-grounded. High G + low q_claim = reliable.")
        elif confidence == 'medium':
            print("    Moderate grounding. Some epistemic debt or low stability.")
        elif confidence == 'low':
            print("    Low grounding. High agency may not indicate truth here.")
        else:
            print("    Poor grounding. G < 0.3 or high q_claim. Verify independently.")
        print()

    def send_to_llm(self, prompt: str) -> Optional[str]:
        """Send a message to LLM creature via bond and get response.

        If native inference is enabled and model is loaded, uses native model.
        Otherwise falls back to Ollama via LLM creature.
        """
        # Phase 26: Native inference path
        if self.native_enabled and self.native_model:
            try:
                import sys
                params = SamplingParams(temperature=0.7, top_p=0.9)

                # Check cache capacity (Phase 26.4)
                cache_info = self.native_model.cache_info()
                usage = cache_info['usage']
                remaining = cache_info['remaining']

                # Estimate tokens needed: prompt + max generation
                prompt_tokens_est = len(prompt) // 3 + 50  # rough estimate
                needed = prompt_tokens_est + 256  # max_tokens

                if remaining < needed:
                    # Auto-shift to keep last 75% of context
                    keep = int(cache_info['capacity'] * 0.5)
                    if self.verbose:
                        print(f"\033[33m[Auto-shifting cache: keeping last {keep} tokens]\033[0m")
                    self.native_model.cache_shift(keep)
                elif usage > 0.9:
                    print(f"\033[33m[Warning: Cache {usage*100:.0f}% full. Use 'cache reset' for new topic.]\033[0m")

                # Format prompt with chat template if available
                if self.chat_template:
                    formatted_prompt = self.chat_template.format_prompt(
                        prompt, self.system_message
                    )
                else:
                    formatted_prompt = prompt

                # Get DET state from LLM creature for truthfulness evaluation
                det_state = None
                bond_state = None
                if self.llm_cid:
                    det_state = self.runner.get_creature_state(self.llm_cid)
                    # Get bond coherence if bonded to terminal
                    if self.terminal_llm_bond is not None:
                        channel = self.runner.channels.get(self.terminal_llm_bond)
                        if channel:
                            bond_state = {'coherence': channel.coherence}

                # Track tokens for truthfulness
                token_count = [0]

                # Streaming callback to show progress
                def on_token(text, token_id):
                    # Stop on <|im_end|> token (151645 for Qwen)
                    # The EOS token should already stop generation, but
                    # we also filter out the token text just in case
                    if "<|im_end|>" in text or "<|im_start|>" in text:
                        return
                    token_count[0] += 1
                    sys.stdout.write(f"\033[32m{text}\033[0m")
                    sys.stdout.flush()

                print()  # Newline before streaming output

                # Use primitives for stats (Phase 26.6 - unified model)
                reg = get_registry()
                reg.primitives['model_stats_start'].handler(512)

                # Streaming generation uses shared model directly (callbacks need direct access)
                self.native_model.generate(
                    formatted_prompt, max_tokens=256, params=params, callback=on_token
                )
                print()  # Newline after streaming

                # Get real entropy from token stats via primitive
                gen_stats = reg.primitives['model_stats_aggregate'].handler()

                # Compute truthfulness score via primitives (Phase 26.6)
                try:

                    # Reset evaluator for this generation
                    reg.primitives['truth_reset'].handler()

                    # Set grounding signals based on generation
                    delta_f = token_count[0] * 0.1  # F cost scales with tokens
                    c_user = bond_state.get('coherence', 1.0) if bond_state else 1.0
                    reg.primitives['truth_set_grounding'].handler(
                        delta_f=delta_f,
                        stability=1.0,  # Would need re-generation to measure
                        c_user=c_user,
                        violations=0
                    )

                    # Record claims (each token is a potential claim)
                    # Use batch recording for efficiency
                    for _ in range(min(token_count[0], 100)):
                        reg.primitives['truth_record_claim'].handler(
                            f_cost=0.1,
                            min_cost=0.1
                        )

                    # Evaluate with actual DET state and real entropy (Phase 26.6)
                    agency = det_state.get('a', 0.7) if det_state else 0.7
                    q_creature = det_state.get('q', 0.0) if det_state else 0.0

                    # Use real entropy from token stats (Phase 26.6)
                    real_entropy = gen_stats.get('mean_entropy', 0.5) if gen_stats else 0.5
                    real_k_eff = int(gen_stats.get('mean_k_eff', 40)) if gen_stats else 40

                    eval_result = reg.primitives['truth_evaluate'].handler(
                        agency=agency,
                        entropy=real_entropy,  # Real entropy from C layer
                        k_eff=real_k_eff,      # Real k_eff from C layer
                        q_creature=q_creature,
                        num_tokens=token_count[0]
                    )

                    if eval_result and 'total' in eval_result:
                        self.last_truth_result = eval_result
                        self.last_truth_result['num_tokens'] = token_count[0]
                        # Store real stats from C layer (Phase 26.6)
                        self.last_truth_result['real_entropy'] = real_entropy
                        self.last_truth_result['real_k_eff'] = real_k_eff
                        self.last_truth_result['min_entropy'] = gen_stats.get('min_entropy', 0.0) if gen_stats else 0.0

                        if self.truthfulness_enabled:
                            self._show_truthfulness_score_compact(self.last_truth_result)
                except Exception as e:
                    if self.verbose:
                        print(f"\033[90m[Truthfulness eval error: {e}]\033[0m")

                return ""  # Return empty - already streamed output
            except Exception as e:
                print(f"\033[31m[Native inference failed: {e}]\033[0m")
                import traceback
                traceback.print_exc()
                # Fall through to Ollama

        # Ollama path (original)
        if not self.terminal_cid or not self.llm_cid:
            return None

        # Send request via bond
        self.runner.send(self.terminal_cid, self.llm_cid, {
            "type": "primitive",
            "name": "llm_call",
            "args": [prompt]
        })

        # Process messages on LLM creature
        self.runner.process_messages(self.llm_cid)

        # Receive response
        response = self.runner.receive(self.terminal_cid, self.llm_cid)
        if response and response.get("type") == "primitive_result":
            if response.get("success"):
                return response.get("result", "")
            else:
                return f"[Error: {response.get('result_code', 'unknown')}]"

        return None

    def send_to_tool(self, command: str, safe: bool = True) -> Optional[str]:
        """Send a command to Tool creature via bond and get response."""
        if not self.terminal_cid or not self.tool_cid:
            return None

        prim_name = "exec_safe" if safe else "exec"

        # Send request via bond
        self.runner.send(self.terminal_cid, self.tool_cid, {
            "type": "primitive",
            "name": prim_name,
            "args": [command]
        })

        # Process messages on Tool creature
        self.runner.process_messages(self.tool_cid)

        # Receive response
        response = self.runner.receive(self.terminal_cid, self.tool_cid)
        if response and response.get("type") == "primitive_result":
            if response.get("success"):
                return response.get("result", "")
            else:
                return f"[Error: {response.get('result_code', 'unknown')}]"

        return None

    def run_repl(self):
        """Run the terminal REPL with bond-based dispatch."""
        if not self.terminal_cid:
            print("Error: Terminal creature not spawned")
            return

        print("=" * 60)
        print("DET-OS Terminal (Bond-Based Dispatch)")
        print("=" * 60)
        print("Type 'help' for commands, 'quit' to exit.")
        print()

        running = True
        while running:
            try:
                # Read user input directly (kernel-based input not yet fully working)
                try:
                    user_input = input("det> ")
                except EOFError:
                    running = False
                    continue

                user_input = user_input.strip()
                if not user_input:
                    continue

                cmd_lower = user_input.lower()

                # Built-in commands
                if cmd_lower in ("quit", "exit"):
                    running = False
                    continue

                elif cmd_lower == "help":
                    self._show_help()
                    continue

                elif cmd_lower == "status":
                    self._show_status()
                    continue

                elif cmd_lower == "bonds":
                    self._show_bonds()
                    continue

                elif cmd_lower == "list" or cmd_lower == "ls":
                    self._list_creatures()
                    continue

                elif cmd_lower.startswith("load "):
                    args = user_input[5:].strip().split()
                    if args:
                        name = args[0]
                        f = float(args[1]) if len(args) > 1 else 50.0
                        a = float(args[2]) if len(args) > 2 else 0.5
                        self.load_creature(name, initial_f=f, initial_a=a)
                    else:
                        print("Usage: load <name> [F] [a]")
                    continue

                elif cmd_lower.startswith("unload "):
                    name = user_input[7:].strip()
                    if name:
                        self.unload_creature(name)
                    else:
                        print("Usage: unload <name>")
                    continue

                elif cmd_lower.startswith("bond "):
                    args = user_input[5:].strip().split()
                    if len(args) == 1:
                        # bond <name> - bond terminal to creature
                        self.bond_creature(args[0])
                    elif len(args) == 2:
                        # bond <a> <b> - bond two creatures
                        self.bond_creature(args[0], args[1])
                    else:
                        print("Usage: bond <name>  OR  bond <creature1> <creature2>")
                    continue

                elif cmd_lower.startswith("use "):
                    args = user_input[4:].strip().split()
                    if args:
                        name = args[0]
                        f = float(args[1]) if len(args) > 1 else 50.0
                        a = float(args[2]) if len(args) > 2 else 0.5
                        self.use_creature(name, initial_f=f, initial_a=a)
                    else:
                        print("Usage: use <name> [F] [a]")
                    continue

                elif cmd_lower.startswith("recompile "):
                    name = user_input[10:].strip()
                    if name:
                        self._recompile_creature(name)
                    else:
                        print("Usage: recompile <name>")
                    continue

                elif cmd_lower == "recompile" or cmd_lower == "recompile all":
                    self._recompile_all()
                    continue

                elif cmd_lower.startswith("send "):
                    # Send primitive to any bonded creature: send <creature> <primitive> <args...>
                    parts = user_input[5:].strip().split(None, 2)
                    if len(parts) >= 2:
                        target, prim = parts[0], parts[1]
                        if len(parts) > 2:
                            # Strip surrounding quotes from argument if present
                            arg = parts[2]
                            if (arg.startswith('"') and arg.endswith('"')) or \
                               (arg.startswith("'") and arg.endswith("'")):
                                arg = arg[1:-1]
                            args = [arg]
                        else:
                            args = []
                        self._send_to_creature(target, prim, args)
                    else:
                        print("Usage: send <creature> <primitive> [args...]")
                    continue

                # Bond-based dispatch
                elif cmd_lower.startswith("ask "):
                    query = user_input[4:].strip()
                    print(f"\033[33m[Sending to LLM via bond...]\033[0m")
                    response = self.send_to_llm(query)
                    if response:
                        print(f"\033[32m{response}\033[0m")
                    else:
                        print("\033[31m[No response from LLM]\033[0m")

                # Phase 21: LLM Management Commands
                elif cmd_lower == "llm" or cmd_lower == "llm help":
                    self._llm_help()
                    continue

                elif cmd_lower == "llm status":
                    self._llm_status()
                    continue

                elif cmd_lower == "llm models":
                    self._llm_list_models()
                    continue

                elif cmd_lower.startswith("llm model "):
                    args = user_input[10:].strip().split()
                    if len(args) >= 1:
                        model_name = args[0]
                        model_type = args[1] if len(args) > 1 else "default"
                        self._llm_set_model(model_name, model_type)
                    else:
                        print("Usage: llm model <name> [type]")
                        print("  Types: default, reasoning, coding, fast")
                    continue

                elif cmd_lower == "llm budget":
                    self._llm_show_budget()
                    continue

                elif cmd_lower.startswith("llm budget reset"):
                    args = user_input[16:].strip().split()
                    new_budget = int(args[0]) if args else 0
                    self._llm_reset_budget(new_budget)
                    continue

                elif cmd_lower.startswith("llm stream "):
                    prompt = user_input[11:].strip()
                    self._llm_stream(prompt)
                    continue

                elif cmd_lower.startswith("llm config "):
                    config_json = user_input[11:].strip()
                    self._llm_configure(config_json)
                    continue

                # Phase 26: Native Inference Commands
                elif cmd_lower == "native" or cmd_lower == "native help":
                    self._native_help()
                    continue

                elif cmd_lower == "native status":
                    self._native_status()
                    continue

                elif cmd_lower.startswith("native load "):
                    args = user_input[12:].strip()
                    if args:
                        # Check for --q8 flag (F32 is now default)
                        use_q8 = False
                        if args.endswith(" --q8"):
                            use_q8 = True
                            args = args[:-5].strip()
                        elif args.startswith("--q8 "):
                            use_q8 = True
                            args = args[5:].strip()
                        self._native_load(args, use_q8_mode=use_q8)
                    else:
                        print("Usage: native load <path/to/model.gguf> [--q8]")
                    continue

                elif cmd_lower == "native enable":
                    self._native_enable()
                    continue

                elif cmd_lower == "native disable":
                    self._native_disable()
                    continue

                elif cmd_lower.startswith("native generate "):
                    prompt = user_input[16:].strip()
                    if prompt:
                        self._native_generate(prompt)
                    else:
                        print("Usage: native generate <prompt>")
                    continue

                elif cmd_lower == "native reset":
                    self._native_reset()
                    continue

                elif cmd_lower.startswith("native system "):
                    system_msg = user_input[14:].strip()
                    if system_msg:
                        self.system_message = system_msg
                        print(f"\033[32mSystem message set: \"{system_msg[:60]}{'...' if len(system_msg) > 60 else ''}\"\033[0m")
                    else:
                        print(f"Current system message: \"{self.system_message}\"")
                    continue

                # Phase 26.6: Truthfulness Commands
                elif cmd_lower == "truth" or cmd_lower == "truth help":
                    self._truthfulness_help()
                    continue

                elif cmd_lower == "truth status":
                    self._truthfulness_status()
                    continue

                elif cmd_lower == "truth enable":
                    self.truthfulness_enabled = True
                    print("\033[32mTruthfulness display enabled.\033[0m")
                    continue

                elif cmd_lower == "truth disable":
                    self.truthfulness_enabled = False
                    print("\033[33mTruthfulness display disabled.\033[0m")
                    continue

                elif cmd_lower == "truth last":
                    self._show_truthfulness_details()
                    continue

                elif cmd_lower == "truth weights":
                    # Get weights via primitive directly
                    try:
                        reg = get_registry()
                        weights = reg.primitives['truth_get_weights'].handler()
                        if weights and 'w_grounding' in weights:
                            print(f"\n\033[33mTruthfulness Weights (DET-Rigorous):\033[0m")
                            print(f"  w_grounding:   {weights['w_grounding']:.3f}  (paid claims, stability, C_user)")
                            print(f"  w_agency:      {weights['w_agency']:.3f}  (GATED by grounding factor G)")
                            print(f"  w_consistency: {weights['w_consistency']:.3f}  (1 - H_normalized)")
                            print(f"  w_coherence:   {weights['w_coherence']:.3f}  (user-specific bond)")
                            print()
                        else:
                            print("\033[31mTruthfulness weights not available.\033[0m")
                    except Exception as e:
                        print(f"\033[31mError getting weights: {e}\033[0m")
                    continue

                elif cmd_lower == "truth falsifiers":
                    # Get falsifiers from last result (stored from primitive)
                    if self.last_truth_result and 'falsifiers' in self.last_truth_result:
                        falsifiers = self.last_truth_result['falsifiers']
                        print(f"\n\033[33mFalsifier Check Results:\033[0m")
                        triggered = []
                        for flag, value in falsifiers.items():
                            status = "\033[31m⚠ TRIGGERED\033[0m" if value else "\033[32m✓ OK\033[0m"
                            print(f"  {flag}: {status}")
                            if value:
                                triggered.append(flag)
                        if triggered:
                            print(f"\n  \033[31m⚠ {len(triggered)} falsifier(s) triggered!\033[0m")
                        else:
                            print(f"\n  \033[32m✓ No falsifier violations detected.\033[0m")
                        print(f"\n\033[36mFalsifier Descriptions:\033[0m")
                        print("  F_T1_reward_hacking:    High T without grounding evidence")
                        print("  F_T2_overconfidence:    Low entropy but low stability")
                        print("  F_T3_coherence_misuse:  High C_user alone yields high T")
                        print("  F_T4_agency_ungated:    High agency without grounding")
                        print()
                    else:
                        print("\033[33mNo falsifier data. Generate text first.\033[0m")
                    continue

                elif cmd_lower.startswith("run "):
                    command = user_input[4:].strip()
                    print(f"\033[33m[Sending to Tool via bond...]\033[0m")
                    response = self.send_to_tool(command, safe=True)
                    if response:
                        print(f"\033[36m{response}\033[0m")
                    else:
                        print("\033[31m[No response from Tool]\033[0m")

                elif cmd_lower.startswith("exec "):
                    # Unsafe execution (requires higher agency)
                    command = user_input[5:].strip()
                    print(f"\033[33m[Sending to Tool (unsafe) via bond...]\033[0m")
                    response = self.send_to_tool(command, safe=False)
                    if response:
                        print(f"\033[36m{response}\033[0m")
                    else:
                        print("\033[31m[No response from Tool]\033[0m")

                elif cmd_lower.startswith("calc "):
                    # Calculator shortcut
                    expr = user_input[5:].strip()
                    self._calculate(expr)
                    continue

                elif cmd_lower == "gpu" or cmd_lower == "gpu status":
                    self._show_gpu_status()
                    continue

                elif cmd_lower == "gpu enable":
                    if not self.gpu_enabled:
                        if self._init_gpu():
                            print("GPU acceleration enabled")
                        else:
                            print("Failed to enable GPU")
                    else:
                        print("GPU already enabled")
                    continue

                elif cmd_lower == "gpu disable":
                    if self.gpu_enabled:
                        self.gpu_enabled = False
                        self.gpu_backend = None
                        print("GPU acceleration disabled")
                    else:
                        print("GPU not enabled")
                    continue

                elif cmd_lower.startswith("gpu benchmark"):
                    # Parse optional args: gpu benchmark [nodes] [bonds] [ticks]
                    args = user_input[13:].strip().split()
                    nodes = int(args[0]) if len(args) > 0 else 1000
                    bonds = int(args[1]) if len(args) > 1 else nodes * 2
                    ticks = int(args[2]) if len(args) > 2 else 1000
                    self.gpu_benchmark(nodes, bonds, ticks)
                    continue

                elif cmd_lower.startswith("gpu tick"):
                    # Execute ticks on GPU: gpu tick [count]
                    args = user_input[8:].strip().split()
                    num_ticks = int(args[0]) if args else 1
                    if self.gpu_enabled:
                        elapsed = self.gpu_execute_ticks(num_ticks)
                        print(f"Executed {num_ticks} GPU ticks in {elapsed:.2f}ms")
                    else:
                        print("GPU not enabled. Use 'gpu enable' first.")
                    continue

                # KV Cache commands (Phase 26.4) - via primitives
                elif cmd_lower == "cache" or cmd_lower == "cache status":
                    self._show_cache_status()
                    continue

                elif cmd_lower == "cache reset":
                    try:
                        reg = get_registry()
                        reg.primitives['model_reset'].handler()
                        print("KV cache reset. Ready for new conversation.")
                    except Exception as e:
                        print(f"No model loaded or reset failed: {e}")
                    continue

                elif cmd_lower.startswith("cache shift"):
                    try:
                        reg = get_registry()
                        args = cmd_lower[11:].strip().split()
                        if args:
                            keep_last = int(args[0])
                            old_status = reg.primitives['model_cache_status'].handler()
                            old_pos = old_status.get('position', 0)
                            if reg.primitives['model_cache_shift'].handler(keep_last):
                                new_status = reg.primitives['model_cache_status'].handler()
                                new_pos = new_status.get('position', 0)
                                print(f"Cache shifted: {old_pos} -> {new_pos} tokens (kept last {keep_last})")
                            else:
                                print("Cache shift failed.")
                        else:
                            print("Usage: cache shift <N>  (keep last N tokens)")
                    except ValueError:
                        print("Usage: cache shift <N>")
                    except Exception as e:
                        print(f"No model loaded or shift failed: {e}")
                    continue

                elif cmd_lower.startswith("cache slice"):
                    try:
                        reg = get_registry()
                        args = cmd_lower[11:].strip().split()
                        if len(args) >= 2:
                            start = int(args[0])
                            end = int(args[1])
                            if reg.primitives['model_cache_slice'].handler(start, end):
                                print(f"Cache sliced to positions [{start}, {end})")
                            else:
                                print("Cache slice failed.")
                        else:
                            print("Usage: cache slice <start> <end>")
                    except ValueError:
                        print("Usage: cache slice <start> <end>")
                    except Exception as e:
                        print(f"No model loaded or slice failed: {e}")
                    continue

                # Collider commands (Phase 20.5)
                elif cmd_lower == "collider" or cmd_lower == "collider help":
                    self._collider_help()
                    continue

                elif cmd_lower == "collider demo":
                    self._collider_demo()
                    continue

                elif cmd_lower.startswith("collider create"):
                    args = user_input[15:].strip().split()
                    dim = int(args[0]) if len(args) > 0 else 1
                    N = int(args[1]) if len(args) > 1 else 200
                    self._collider_create(dim, N)
                    continue

                elif cmd_lower.startswith("collider add"):
                    # collider add <pos> <mass> <width> [momentum] [q]
                    self._collider_add_packet(user_input[12:].strip())
                    continue

                elif cmd_lower.startswith("collider step"):
                    args = user_input[13:].strip().split()
                    n = int(args[0]) if args else 100
                    self._collider_step(n)
                    continue

                elif cmd_lower == "collider status" or cmd_lower == "collider stats":
                    self._collider_status()
                    continue

                elif cmd_lower.startswith("collider render"):
                    args = user_input[15:].strip().split()
                    field = args[0] if args else "F"
                    width = int(args[1]) if len(args) > 1 else 60
                    self._collider_render(field, width)
                    continue

                elif cmd_lower == "collider destroy":
                    self._collider_destroy()
                    continue

                # === Falsification Suite ===
                elif cmd_lower == "falsify" or cmd_lower == "falsify help":
                    self._falsify_help()
                    continue

                elif cmd_lower == "falsify all":
                    self._falsify_all()
                    continue

                elif cmd_lower == "falsify core":
                    self._falsify_core()
                    continue

                elif cmd_lower == "falsify gtd":
                    self._falsify_gtd()
                    continue

                elif cmd_lower == "falsify agency":
                    self._falsify_agency()
                    continue

                elif cmd_lower == "falsify list":
                    self._falsify_list()
                    continue

                elif cmd_lower.startswith("falsify "):
                    test_id = user_input[8:].strip()
                    self._falsify_single(test_id)
                    continue

                else:
                    # Default: send to LLM
                    print(f"\033[33m[Sending to LLM via bond...]\033[0m")
                    response = self.send_to_llm(user_input)
                    if response:
                        print(f"\033[32m{response}\033[0m")
                    else:
                        print("\033[31m[No response from LLM]\033[0m")

            except KeyboardInterrupt:
                print("\n")
                running = False

            except EOFError:
                print("\n")
                running = False

        print("\nDET-OS terminated.")

    def _show_help(self):
        """Display help information."""
        print("""
\033[33mDET Terminal Commands:\033[0m
  help              Show this help
  status            Show creature status
  bonds             Show bond connections
  quit              Exit the terminal

\033[33mCreature Management:\033[0m
  list              List available and loaded creatures
  load <name> [F] [a]   Load a creature (default F=50, a=0.5)
  use <name> [F] [a]    Load AND bond creature to terminal
  unload <name>     Unload a creature
  bond <name>       Bond terminal to a creature
  bond <a> <b>      Bond two creatures together
  recompile <name>  Recompile creature from source
  recompile all     Recompile all creatures

\033[33mBond-Based Dispatch:\033[0m
  ask <query>       Send to LLM creature via bond
  run <cmd>         Execute via Tool creature (safe mode)
  exec <cmd>        Execute via Tool creature (unsafe mode)
  calc <expr>       Evaluate math expression (auto-loads calculator)
  send <creature> <primitive> [args]  Send primitive to creature

\033[33mLLM Management (Phase 21):\033[0m
  llm               Show LLM help
  llm status        Show LLM status (model, budget, temp)
  llm models        List configured models
  llm model <n> [t] Set model (types: default,reasoning,coding,fast)
  llm budget        Show token budget
  llm budget reset  Reset token budget
  llm stream <p>    Streaming prompt

\033[33mNative Inference (Phase 26):\033[0m
  native            Show native inference help
  native status     Show model, GPU, and chat template status
  native load <p>   Load GGUF model from path
  native enable     Enable native inference (auto after load)
  native disable    Disable native (use Ollama)
  native generate   Generate text directly
  native reset      Reset KV cache
  native system <m> Set system message for chat template

\033[33mTruthfulness (Phase 26.6 - via TruthfulnessCreature.ex):\033[0m
  truth             Show truthfulness help
  truth status      Show truthfulness settings and last score
  truth enable      Enable T score display after generation
  truth disable     Disable T score display
  truth last        Show details of last truthfulness score
  truth weights     Show component weights (via creature)
  truth falsifiers  Show falsifier check results

\033[33mGPU Acceleration:\033[0m
  gpu               Show GPU status
  gpu enable        Enable GPU acceleration
  gpu disable       Disable GPU acceleration
  gpu tick [N]      Execute N substrate ticks on GPU
  gpu benchmark [nodes] [bonds] [ticks]  Run GPU benchmark

\033[33mKV Cache (Phase 26.4):\033[0m
  cache             Show KV cache status (position, capacity, usage)
  cache reset       Reset cache for new conversation
  cache shift <N>   Keep only last N tokens (sliding window)
  cache slice <s> <e>  Keep positions [start, end)

\033[33mCollider (Phase 20.5):\033[0m
  collider          Show collider commands
  collider demo     Run gravitational binding demo
  collider create [dim] [N]   Create lattice
  collider add <pos> <mass> <width> [momentum] [q]
  collider step [N] Run N physics steps
  collider status   Show statistics
  collider render   Show ASCII visualization

\033[33mFalsification (DET v6.3):\033[0m
  falsify           Show falsification help
  falsify all       Run all falsification tests
  falsify core      Run core falsifiers (F6-F9)
  falsify gtd       Run time dilation tests
  falsify agency    Run agency tests
  falsify <test>    Run single test (e.g., falsify F6)
  falsify list      List all available tests

\033[33mArchitecture:\033[0m
  All commands dispatch through DET bonds.
  Creatures communicate via message passing, not direct calls.
  Bytecode is cached in .exb files for fast loading.
  GPU accelerates substrate (node/bond state) for large-scale simulations.
""")

    def _show_status(self):
        """Display creature status."""
        print("\n\033[33mCreature Status:\033[0m")

        for name, cid in [("Terminal", self.terminal_cid),
                          ("LLM", self.llm_cid),
                          ("Tool", self.tool_cid)]:
            if cid:
                state = self.runner.get_creature_state(cid)
                if state:
                    print(f"  {name} (cid={cid}):")
                    print(f"    F={state['F']:.1f}, a={state['a']:.2f}")
                    print(f"    Kernels executed: {state['kernels_executed']}")
                    print(f"    Messages: {state['messages_sent']} sent, {state['messages_received']} received")
            else:
                print(f"  {name}: not spawned")

        # GPU status summary
        if GPU_AVAILABLE:
            if self.gpu_enabled:
                print(f"\n  \033[32mGPU: {self.gpu_backend.device_name} (enabled)\033[0m")
            else:
                print(f"\n  \033[90mGPU: Available (disabled)\033[0m")
        print()

    def _show_bonds(self):
        """Display all bond connections."""
        print("\n\033[33mBond Connections:\033[0m")

        # Collect all unique bonds (avoid duplicates since bonds are bidirectional)
        shown_bonds = set()

        for name, (cid, _) in self.creatures.items():
            creature = self.runner.creatures.get(cid)
            if creature and creature.bonds:
                for peer_cid, channel_id in creature.bonds.items():
                    # Create a canonical bond identifier (smaller cid first)
                    bond_key = (min(cid, peer_cid), max(cid, peer_cid), channel_id)
                    if bond_key in shown_bonds:
                        continue
                    shown_bonds.add(bond_key)

                    # Find peer name
                    peer_name = "unknown"
                    for pname, (pcid, _) in self.creatures.items():
                        if pcid == peer_cid:
                            peer_name = pname
                            break

                    print(f"  {name} <-> {peer_name} (channel={channel_id})")

        if not shown_bonds:
            print("  No bonds established")
        print()

    def _list_creatures(self):
        """List available and loaded creatures."""
        print("\n\033[33mLoaded Creatures:\033[0m")
        if self.creatures:
            for name, (cid, compiled) in self.creatures.items():
                state = self.runner.get_creature_state(cid)
                if state:
                    kernels = len(compiled.kernels)
                    print(f"  [{cid}] {name}: F={state['F']:.1f}, a={state['a']:.2f}, {kernels} kernels")
        else:
            print("  (none)")

        print("\n\033[33mAvailable Creatures:\033[0m")
        # Refresh available list
        self.discover_creatures()
        loaded_names = set(self.creatures.keys())
        available_unloaded = [n for n in self.available_creatures.keys() if n not in loaded_names]

        if available_unloaded:
            for name in sorted(available_unloaded):
                path = self.available_creatures[name]
                exb_path = path.with_suffix('.exb')
                cache_status = "\033[32m[cached]\033[0m" if exb_path.exists() else "\033[90m[no cache]\033[0m"
                print(f"  {name}.ex  {cache_status}")
        else:
            print("  (all creatures loaded)")

        # Show cache stats
        stats = _cache.stats
        print(f"\n\033[33mCache Stats:\033[0m hits={stats['hits']}, misses={stats['misses']}, recompiles={stats['recompiles']}")
        print()

    def _recompile_creature(self, name: str):
        """Recompile a single creature from source."""
        # Find the file
        if name in self.available_creatures:
            path = self.available_creatures[name]
        else:
            path = EXISTENCE_DIR / f"{name}.ex"

        if not path.exists():
            print(f"Creature file not found: {path}")
            return

        try:
            print(f"Recompiling {path.name}...")
            start = time.perf_counter()
            _cache.compile(path, force=True)
            elapsed = (time.perf_counter() - start) * 1000
            exb_path = path.with_suffix('.exb')
            print(f"  Created {exb_path.name} ({exb_path.stat().st_size:,} bytes, {elapsed:.2f}ms)")
        except Exception as e:
            print(f"  Error: {e}")

    def _recompile_all(self):
        """Recompile all available creatures."""
        self.discover_creatures()
        print(f"Recompiling {len(self.available_creatures)} creatures...")
        for name, path in sorted(self.available_creatures.items()):
            self._recompile_creature(name)

    def _calculate(self, expr: str):
        """Evaluate a math expression using calculator creature."""
        # Auto-load and bond calculator if needed
        if 'calculator' not in self.creatures:
            print("\033[90m[Loading calculator...]\033[0m")
            cid = self.load_creature('calculator', initial_f=50.0, initial_a=0.5)
            if not cid:
                print("\033[31m[Failed to load calculator]\033[0m")
                return

        calc_cid, _ = self.creatures['calculator']

        # Auto-bond if needed
        if self.terminal_cid:
            terminal = self.runner.creatures.get(self.terminal_cid)
            if terminal and calc_cid not in terminal.bonds:
                print("\033[90m[Bonding to calculator...]\033[0m")
                self.bond_creature('calculator')

        # Send eval_math primitive
        self.runner.send(self.terminal_cid, calc_cid, {
            'type': 'primitive',
            'name': 'eval_math',
            'args': [expr]
        })
        self.runner.process_messages(calc_cid)
        response = self.runner.receive(self.terminal_cid, calc_cid)

        if response and response.get('success'):
            result = response.get('result', '')
            print(f"\033[32m{expr} = {result}\033[0m")
        else:
            error = response.get('result', 'unknown error') if response else 'no response'
            print(f"\033[31m{expr} = Error: {error}\033[0m")

    def _show_gpu_status(self):
        """Display GPU backend status."""
        print("\n\033[33mGPU Status:\033[0m")
        print(f"  Available: {GPU_AVAILABLE}")

        if GPU_AVAILABLE:
            if self.gpu_enabled and self.gpu_backend:
                print(f"  Enabled: Yes")
                print(f"  Device: {self.gpu_backend.device_name}")
                print(f"  Memory: {self.gpu_backend.memory_usage / 1024 / 1024:.1f} MB")
                print(f"  Max nodes: {self.gpu_nodes.num_nodes if self.gpu_nodes else 0}")
                print(f"  Max bonds: {self.gpu_bonds.num_bonds if self.gpu_bonds else 0}")
            else:
                print(f"  Enabled: No")
                try:
                    backend = MetalBackend()
                    print(f"  Device: {backend.device_name} (ready)")
                except Exception as e:
                    print(f"  Device: Error - {e}")
        else:
            print("  Metal GPU backend not available on this system")
        print()

    def _show_cache_status(self):
        """Display KV cache status (Phase 26.4)."""
        print("\n\033[33mKV Cache Status:\033[0m")

        if not self.native_model:
            print("  No model loaded. Use 'load model <path>' first.")
            print()
            return

        info = self.native_model.cache_info()
        pos = info['position']
        cap = info['capacity']
        usage = info['usage'] * 100
        remaining = info['remaining']

        # Color based on usage
        if usage < 50:
            usage_color = "\033[32m"  # Green
        elif usage < 80:
            usage_color = "\033[33m"  # Yellow
        else:
            usage_color = "\033[31m"  # Red

        print(f"  Position:  {pos:,} tokens")
        print(f"  Capacity:  {cap:,} tokens")
        print(f"  Usage:     {usage_color}{usage:.1f}%\033[0m")
        print(f"  Remaining: {remaining:,} tokens")

        # Visual bar
        bar_width = 40
        filled = int(bar_width * info['usage'])
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"  [{usage_color}{bar}\033[0m]")

        if usage >= 80:
            print(f"\n  \033[33mWarning: Cache nearing capacity.")
            print(f"  Use 'cache shift <N>' to keep last N tokens, or")
            print(f"  'cache reset' to clear for new conversation.\033[0m")
        print()

    def _send_to_creature(self, target: str, primitive: str, args: list):
        """Send a primitive call to a creature via bond."""
        if target not in self.creatures:
            print(f"Creature '{target}' not loaded")
            return

        target_cid, _ = self.creatures[target]

        # Check if bonded
        if self.terminal_cid:
            terminal = self.runner.creatures.get(self.terminal_cid)
            if terminal and target_cid not in terminal.bonds:
                print(f"Not bonded to '{target}'. Use 'bond {target}' first.")
                return

        # Send primitive request
        self.runner.send(self.terminal_cid, target_cid, {
            "type": "primitive",
            "name": primitive,
            "args": args
        })

        # Process
        self.runner.process_messages(target_cid)

        # Get response
        response = self.runner.receive(self.terminal_cid, target_cid)
        if response and response.get("type") == "primitive_result":
            if response.get("success"):
                result = response.get("result", "")
                print(f"\033[32m{result}\033[0m")
            else:
                print(f"\033[31m[Error: {response.get('result_code', 'unknown')}]\033[0m")
        else:
            print("\033[31m[No response]\033[0m")

    # =========================================================================
    # Collider Methods (Phase 20.5)
    # =========================================================================

    def _collider_help(self):
        """Display collider commands help."""
        print("""
\033[33mCollider Commands (Phase 20.5 - Native DET Physics):\033[0m
  collider                Show this help
  collider demo           Run demonstration (two-body gravitational binding)
  collider create [dim] [N]   Create lattice (default: dim=1, N=200)
  collider add <pos> <mass> <width> [momentum] [q]
                          Add resource packet
  collider step [N]       Run N physics steps (default: 100)
  collider status         Show lattice statistics
  collider render [field] [width]  Render ASCII visualization (field: F,q,a,P)
  collider destroy        Destroy current lattice

\033[33mExample Session:\033[0m
  det> collider create 1 200         # Create 1D lattice with 200 nodes
  det> collider add 50 8 5 0.1 0.3   # Add packet at pos 50, mass 8, width 5
  det> collider add 150 8 5 -0.1 0.3 # Add packet at pos 150
  det> collider step 500             # Run 500 physics steps
  det> collider render               # Show ASCII visualization
  det> collider status               # Show mass, separation, energy

\033[33mOr just run:\033[0m
  det> collider demo                 # Auto-run gravitational binding demo
""")

    def _collider_demo(self):
        """Run a demonstration of the DET collider."""
        import time

        print("\n\033[33m=== DET Collider Demo ===\033[0m")
        print("Demonstrating DET v6.3 physics: gravity, momentum, structure\n")

        # Try C lattice first for performance
        use_c_lattice = False
        try:
            from det.lattice_c import CLattice, is_available
            if is_available():
                use_c_lattice = True
                print("\033[32m[Using C substrate for native performance]\033[0m\n")
        except ImportError:
            pass

        if use_c_lattice:
            # C substrate path
            lattice = CLattice(dim=1, size=200)
            # Set DET v6.3 physics parameters
            lattice.set_param("beta_g", 15.0)
            lattice.set_param("kappa_grav", 3.0)
            lattice.set_param("mu_grav", 1.5)
            lattice.set_param("gravity_enabled", 1.0)
            lattice.set_param("momentum_enabled", 1.0)

            # Add two structure sites with resource
            print("Creating two gravitational wells (q=structure, F=resource)...")
            lattice.add_packet(center=[70], mass=12.0, width=6.0, momentum=[0.0], q=0.5)
            lattice.add_packet(center=[130], mass=12.0, width=6.0, momentum=[0.0], q=0.5)

            initial_mass = lattice.total_mass()
            initial_sep = lattice.separation()
            print(f"Initial: total_mass={initial_mass:.2f}, separation={initial_sep:.1f}")
            print("\nRunning 1000 physics steps...")

            start = time.time()

            for i in range(10):
                lattice.step(100)
                if i % 2 == 1:
                    stats = lattice.get_stats()
                    print(f"  Step {(i+1)*100:4d}: mass={stats.total_mass:.2f}, sep={stats.separation:.1f}, PE={stats.potential_energy:.1f}")

            elapsed = time.time() - start

            # Final stats
            final_stats = lattice.get_stats()
            print(f"\n\033[32mResults:\033[0m")
            print(f"  Mass: {initial_mass:.2f} -> {final_stats.total_mass:.2f}")
            print(f"  Separation: {initial_sep:.1f} -> {final_stats.separation:.1f}")
            print(f"  Structure (q): {final_stats.total_structure:.3f}")
            print(f"  Elapsed: {elapsed*1000:.1f}ms ({1000/elapsed:.0f} steps/sec)")

            # Physics explanation
            print(f"\n\033[33mPhysics:\033[0m Gravity from structure (q) concentrates resource (F) at wells.")

            # Render
            from det.lattice_c import RENDER_FIELD_F
            print(f"\n\033[36m{lattice.render(RENDER_FIELD_F, 60)}\033[0m")

            self._collider_lattice = lattice
        else:
            # Python fallback path
            import numpy as np
            from det.eis.lattice import DETLattice, LatticeParams

            print("\033[33m[Using Python fallback - build C substrate for better performance]\033[0m\n")

            # Create lattice with DET v6.3 physics
            params = LatticeParams(
                dim=1, N=200, dt=0.02,
                gravity_enabled=True,
                momentum_enabled=True,
                q_enabled=True,
                agency_enabled=True,
                beta_g=15.0,       # Gravity-momentum coupling
                kappa_grav=3.0,    # Potential strength
                mu_grav=1.5,       # Gravity flux coefficient
            )
            lattice = DETLattice(params)
            self._collider_lattice = lattice

            # Add two structure sites with resource
            print("Creating two gravitational wells (q=structure, F=resource)...")
            lattice.add_packet((70,), mass=12.0, width=6.0, initial_q=0.5)
            lattice.add_packet((130,), mass=12.0, width=6.0, initial_q=0.5)

            initial_mass = lattice.total_mass()
            initial_peak1 = lattice.F[70]
            initial_peak2 = lattice.F[130]

            print(f"Initial: total_mass={initial_mass:.2f}, peaks={initial_peak1:.2f}/{initial_peak2:.2f}")
            print(f"Lattice eta: {lattice.eta:.3f}, q_max: {lattice.q.max():.3f}")
            print("\nRunning 1000 physics steps...")

            start = time.time()

            for i in range(10):
                for _ in range(100):
                    lattice.step()

                if i % 2 == 1:
                    p1, p2 = lattice.F[70], lattice.F[130]
                    pe = lattice.potential_energy()
                    pi_max = np.abs(lattice.pi).max()
                    print(f"  Step {(i+1)*100:4d}: peaks={p1:.2f}/{p2:.2f}, PE={pe:.1f}, |pi|_max={pi_max:.3f}")

            elapsed = time.time() - start

            # Final stats
            final_mass = lattice.total_mass()
            final_peak1 = lattice.F[70]
            final_peak2 = lattice.F[130]
            final_q = lattice.total_q()

            print(f"\n\033[32mResults:\033[0m")
            print(f"  Mass: {initial_mass:.2f} -> {final_mass:.2f}")
            print(f"  Peak concentration: {initial_peak1:.2f} -> {final_peak1:.2f} (+{100*(final_peak1-initial_peak1)/initial_peak1:.0f}%)")
            print(f"  Structure (q): {final_q:.3f}")
            print(f"  Elapsed: {elapsed*1000:.1f}ms ({1000/elapsed:.0f} steps/sec)")

            # Physics explanation
            print(f"\n\033[33mPhysics:\033[0m Gravity from structure (q) concentrates resource (F) at wells.")

            # Render
            print(f"\n\033[36m{lattice.render_ascii('F', 60)}\033[0m")

    def _collider_create(self, dim: int = 1, N: int = 200):
        """Create a new collider lattice."""
        from det.eis.primitives import get_registry

        reg = get_registry()
        result = reg.call('lattice_create', [dim, N], 100.0, 1.0)

        if result.result_code.name == 'OK':
            self._collider_id = result.result
            print(f"\033[32mCreated {dim}D lattice (N={N}), id={self._collider_id}\033[0m")
        else:
            print(f"\033[31mFailed: {result.result}\033[0m")

    def _collider_add_packet(self, args_str: str):
        """Add a resource packet to the lattice."""
        from det.eis.primitives import get_registry

        if not hasattr(self, '_collider_id') or self._collider_id is None:
            print("\033[31mNo lattice. Use 'collider create' first.\033[0m")
            return

        # Parse: pos mass width [momentum] [q]
        parts = args_str.split()
        if len(parts) < 3:
            print("Usage: collider add <pos> <mass> <width> [momentum] [q]")
            return

        pos = [float(parts[0])]
        mass = float(parts[1])
        width = float(parts[2])
        momentum = [float(parts[3])] if len(parts) > 3 else None
        initial_q = float(parts[4]) if len(parts) > 4 else 0.0

        reg = get_registry()
        result = reg.call('lattice_add_packet',
                          [self._collider_id, pos, mass, width, momentum, initial_q],
                          100.0, 1.0)

        if result.result_code.name == 'OK':
            print(f"\033[32mAdded packet: pos={pos}, mass={mass}, width={width}\033[0m")
        else:
            print(f"\033[31mFailed: {result.result}\033[0m")

    def _collider_step(self, n: int = 100):
        """Execute physics steps."""
        import time
        from det.eis.primitives import get_registry

        if not hasattr(self, '_collider_id') or self._collider_id is None:
            print("\033[31mNo lattice. Use 'collider create' first.\033[0m")
            return

        reg = get_registry()
        start = time.time()
        result = reg.call('lattice_step', [self._collider_id, n], 100.0, 1.0)
        elapsed = time.time() - start

        if result.result_code.name == 'OK':
            print(f"\033[32mExecuted {n} steps in {elapsed*1000:.1f}ms ({n/elapsed:.0f} steps/sec)\033[0m")
        else:
            print(f"\033[31mFailed: {result.result}\033[0m")

    def _collider_status(self):
        """Show lattice statistics."""
        from det.eis.primitives import get_registry

        if not hasattr(self, '_collider_id') or self._collider_id is None:
            print("\033[31mNo lattice. Use 'collider create' first.\033[0m")
            return

        reg = get_registry()
        stats = reg.call('lattice_get_stats', [self._collider_id], 100.0, 1.0).result

        if isinstance(stats, dict):
            print(f"\n\033[33mCollider Status:\033[0m")
            print(f"  Dimension: {stats.get('dim', '?')}D")
            print(f"  Grid size: {stats.get('N', '?')}")
            print(f"  Steps: {stats.get('step_count', 0)}")
            print(f"  Time: {stats.get('time', 0):.3f}")
            print(f"  Total mass: {stats.get('total_mass', 0):.4f}")
            print(f"  Total q: {stats.get('total_q', 0):.4f}")
            print(f"  Total grace: {stats.get('total_grace', 0):.4f}")
            print(f"  Lattice η: {stats.get('eta', 0):.3f}")

            # Also get separation and PE
            sep = reg.call('lattice_separation', [self._collider_id], 100.0, 1.0).result
            pe = reg.call('lattice_potential_energy', [self._collider_id], 100.0, 1.0).result
            print(f"  Separation: {sep:.2f}")
            print(f"  Potential energy: {pe:.4f}")
        else:
            print(f"\033[31mFailed to get stats: {stats}\033[0m")

    def _collider_render(self, field: str = 'F', width: int = 60):
        """Render lattice as ASCII."""
        from det.eis.primitives import get_registry

        if not hasattr(self, '_collider_id') or self._collider_id is None:
            print("\033[31mNo lattice. Use 'collider create' first.\033[0m")
            return

        reg = get_registry()
        result = reg.call('lattice_render', [self._collider_id, field, width], 100.0, 1.0)

        if result.result_code.name == 'OK':
            print(f"\033[36m{result.result}\033[0m")
        else:
            print(f"\033[31mFailed: {result.result}\033[0m")

    def _collider_destroy(self):
        """Destroy the current lattice."""
        from det.eis.primitives import get_registry

        if not hasattr(self, '_collider_id') or self._collider_id is None:
            print("\033[33mNo lattice to destroy.\033[0m")
            return

        reg = get_registry()
        result = reg.call('lattice_destroy', [self._collider_id], 100.0, 1.0)

        if result.result_code.name == 'OK':
            print(f"\033[32mLattice {self._collider_id} destroyed.\033[0m")
            self._collider_id = None
        else:
            print(f"\033[31mFailed: {result.result}\033[0m")

    # =========================================================================
    # Falsification Suite Commands
    # =========================================================================

    def _falsify_help(self):
        """Show falsification suite help."""
        print("""
\033[33mDET v6.3 Falsification Suite\033[0m
Based on det_theory_card_6_3.md Section VIII

\033[36mCommands:\033[0m
  falsify all       Run all falsification tests
  falsify core      Run core falsifiers (F6-F9)
  falsify gtd       Run time dilation falsifiers (F_GTD1-4)
  falsify agency    Run agency falsifiers (F_A1-A3)
  falsify <test>    Run single test (e.g., falsify F6)
  falsify list      List all available tests

\033[36mCore Falsifiers:\033[0m
  F6   Binding Failure        - Two-body gravitational binding
  F7   Mass Conservation      - Total mass preserved
  F8   Vacuum Momentum        - No transport in vacuum
  F9   Spontaneous Drift      - Symmetric systems stable

\033[36mTime Dilation Falsifiers:\033[0m
  F_GTD1  Presence Formula    - P = a*sigma/(1+F)/(1+H)
  F_GTD3  Grav Accumulation   - F flows to potential wells
  F_GTD4  Dilation Direction  - Time slows where F high

\033[36mAgency Falsifiers:\033[0m
  F_A1   Zombie Test          - High-q implies low agency ceiling
  F_A2   Ceiling Violation    - a never exceeds a_max
  F_A3   Drive w/o Coherence  - Low-C means no relational drive

\033[36mKepler Falsifier:\033[0m
  F_K1   Kepler's Third Law   - T^2 ~ r^3 for orbits

If ANY test FAILS, DET is FALSIFIED.
""")

    def _falsify_all(self):
        """Run all falsification tests."""
        from det.eis.falsifiers import FalsifierSuite

        print("\n\033[33mRunning DET v6.3 Falsification Suite...\033[0m\n")
        suite = FalsifierSuite(verbose=True)
        passed, total, _ = suite.run_all()

    def _falsify_core(self):
        """Run core falsifiers."""
        from det.eis.falsifiers import FalsifierSuite

        suite = FalsifierSuite(verbose=True)
        results = suite.run_core()
        passed = sum(1 for r in results if r.result.value == "PASS")
        print(f"\n\033[33mCore: {passed}/{len(results)} PASSED\033[0m")

    def _falsify_gtd(self):
        """Run gravitational time dilation falsifiers."""
        from det.eis.falsifiers import FalsifierSuite

        suite = FalsifierSuite(verbose=True)
        results = suite.run_gtd()
        passed = sum(1 for r in results if r.result.value == "PASS")
        print(f"\n\033[33mGTD: {passed}/{len(results)} PASSED\033[0m")

    def _falsify_agency(self):
        """Run agency falsifiers."""
        from det.eis.falsifiers import FalsifierSuite

        suite = FalsifierSuite(verbose=True)
        results = suite.run_agency()
        passed = sum(1 for r in results if r.result.value == "PASS")
        print(f"\n\033[33mAgency: {passed}/{len(results)} PASSED\033[0m")

    def _falsify_single(self, test_id: str):
        """Run a single falsification test."""
        from det.eis.falsifiers import FalsifierSuite

        suite = FalsifierSuite(verbose=True)
        result = suite.run_single(test_id)
        if result:
            color = "\033[32m" if result.result.value == "PASS" else "\033[31m"
            print(f"\n{color}{result.test_id}: {result.result.value}\033[0m")

    def _falsify_list(self):
        """List all available falsification tests."""
        from det.eis.falsifiers import FalsifierSuite

        suite = FalsifierSuite(verbose=False)
        tests = suite.list_tests()

        print("\n\033[33mAvailable Falsification Tests:\033[0m")
        print("\n\033[36mCore:\033[0m")
        for t in tests:
            if t.startswith("F") and not "_" in t:
                print(f"  {t}")
        print("\n\033[36mTime Dilation:\033[0m")
        for t in tests:
            if t.startswith("F_GTD"):
                print(f"  {t}")
        print("\n\033[36mAgency:\033[0m")
        for t in tests:
            if t.startswith("F_A"):
                print(f"  {t}")
        print("\n\033[36mKepler:\033[0m")
        for t in tests:
            if t.startswith("F_K"):
                print(f"  {t}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="DET-OS Bootstrap - Start the DET-native operating system"
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="Don't start REPL, just bootstrap and exit"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration via Metal"
    )

    args = parser.parse_args()

    # Create runtime and bootstrap
    runtime = DETRuntime(verbose=args.verbose, use_gpu=args.gpu)

    try:
        runtime.bootstrap()
    except Exception as e:
        print(f"Bootstrap error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if args.no_repl:
        print("Bootstrap complete. Use --no-repl=false to start REPL.")
        return

    # Run REPL
    runtime.run_repl()


if __name__ == "__main__":
    main()
