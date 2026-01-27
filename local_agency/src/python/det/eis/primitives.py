"""
Substrate Primitives
====================

External I/O functions callable from Existence-Lang via the EIS VM.

Primitives are the ONLY way for EL creatures to interact with the outside world.
They form the boundary between the DET-native execution and the host system.

Primitive Categories:
    - LLM: llm_call, llm_chat
    - Shell: exec, exec_safe
    - File: file_read, file_write, file_exists
    - Time: now, sleep
    - Random: random, random_int, random_seed
    - Debug: print, log, trace

Cost Model:
    Each primitive has an associated F cost that is deducted before execution.
    If the creature has insufficient F, the primitive returns an error token.
"""

import os
import time
import random
import subprocess
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Tuple
from enum import IntEnum


class PrimitiveResult(IntEnum):
    """Result codes for primitive calls."""
    OK = 0x0100
    ERROR = 0x0101
    REFUSED = 0x0102      # Insufficient F or agency
    TIMEOUT = 0x0103
    NOT_FOUND = 0x0104
    PERMISSION = 0x0105   # Permission denied


@dataclass
class PrimitiveCall:
    """A primitive call with arguments and result."""
    name: str
    args: List[Any]
    result: Any = None
    result_code: PrimitiveResult = PrimitiveResult.OK
    cost: float = 0.0
    elapsed_ms: float = 0.0


@dataclass
class PrimitiveSpec:
    """Specification for a primitive function."""
    name: str
    handler: Callable
    base_cost: float              # Minimum F cost
    cost_per_unit: float = 0.0    # Additional cost (per char, per ms, etc.)
    min_agency: float = 0.0       # Minimum agency required
    description: str = ""
    arg_types: List[str] = field(default_factory=list)
    return_type: str = "any"


# =============================================================================
# TOKEN CHOICE TRACE SYSTEM (Phase 26.5)
# =============================================================================
# Records token choices as DET-compliant traces with full metadata for
# auditability. Each choice becomes a witness that can be replayed/verified.

class TokenChoiceWitness(IntEnum):
    """Witness tokens for token choice outcomes (matches substrate_types.h)."""
    CHOICE_OK        = 0x0300  # Token sampled successfully
    CHOICE_UNCERTAIN = 0x0301  # High entropy choice (H_norm > 0.7)
    CHOICE_CONFIDENT = 0x0302  # Low entropy choice (H_norm < 0.3)
    CHOICE_NARROW    = 0x0303  # Small k_eff (< 5, concentrated)
    CHOICE_BROAD     = 0x0304  # Large k_eff (> 50, diffuse)
    CHOICE_REFUSED   = 0x0305  # Choice refused (low presence/agency)


@dataclass
class TokenChoiceTrace:
    """
    A single token choice committed as a DET trace.

    This is the atomic unit of auditability for token generation.
    Each trace captures the full state at the moment of choice.
    """
    # Identity
    trace_id: int                    # Unique trace ID
    generation_id: int               # Which generation this belongs to
    position: int                    # Position in sequence (0-indexed)

    # The choice
    token_id: int                    # Selected token ID
    token_text: str                  # Decoded token text

    # Distribution state at choice
    entropy: float                   # H after temperature scaling
    entropy_raw: float               # H before temperature
    k_eff: int                       # Effective vocabulary size (nucleus)
    top_prob: float                  # P(chosen token)
    top5_mass: float                 # Sum of top-5 probabilities

    # DET state at choice
    agency: float                    # Creature agency when choice made
    presence: float                  # Creature presence when choice made
    temperature: float               # Sampling temperature used

    # Witness classification
    witness_token: int               # TokenChoiceWitness value

    # Timing
    timestamp: float                 # Unix timestamp of choice


@dataclass
class GenerationTrace:
    """
    A complete generation as a sequence of token choice traces.

    This is the full audit trail for a single generation request.
    """
    generation_id: int               # Unique generation ID
    prompt: str                      # Input prompt
    output: str                      # Generated output
    token_traces: List[TokenChoiceTrace] = field(default_factory=list)

    # Aggregate stats
    total_tokens: int = 0
    mean_entropy: float = 0.0
    mean_k_eff: float = 0.0
    min_entropy: float = 1.0
    max_entropy: float = 0.0

    # DET state
    agency_start: float = 0.7
    presence_start: float = 1.0
    f_cost: float = 0.0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0


class TraceLedger:
    """
    Ledger of all token choice traces for auditability.

    Maintains a complete record of every token choice, enabling:
    - Replay: Verify choices were deterministic given seed
    - Audit: Inspect distribution state at any choice
    - Analysis: Compute aggregate truthfulness metrics
    """

    def __init__(self, max_generations: int = 1000):
        self.generations: List[GenerationTrace] = []
        self.max_generations = max_generations
        self._next_trace_id = 0
        self._next_generation_id = 0
        self._current_generation: Optional[GenerationTrace] = None

    def start_generation(self, prompt: str, agency: float = 0.7,
                         presence: float = 1.0) -> int:
        """Start a new generation trace."""
        gen_id = self._next_generation_id
        self._next_generation_id += 1

        self._current_generation = GenerationTrace(
            generation_id=gen_id,
            prompt=prompt,
            output="",
            agency_start=agency,
            presence_start=presence,
            start_time=time.time()
        )
        return gen_id

    def record_choice(self, token_id: int, token_text: str,
                      entropy: float, entropy_raw: float, k_eff: int,
                      top_prob: float, top5_mass: float,
                      agency: float, presence: float, temperature: float) -> TokenChoiceTrace:
        """Record a single token choice as a trace."""
        if self._current_generation is None:
            # Auto-start if needed
            self.start_generation("", agency, presence)

        # Classify the choice
        h_norm = entropy / (max(1, k_eff) + 1e-6)  # Normalized entropy
        if h_norm > 0.7:
            witness = TokenChoiceWitness.CHOICE_UNCERTAIN
        elif h_norm < 0.3:
            witness = TokenChoiceWitness.CHOICE_CONFIDENT
        elif k_eff < 5:
            witness = TokenChoiceWitness.CHOICE_NARROW
        elif k_eff > 50:
            witness = TokenChoiceWitness.CHOICE_BROAD
        else:
            witness = TokenChoiceWitness.CHOICE_OK

        trace = TokenChoiceTrace(
            trace_id=self._next_trace_id,
            generation_id=self._current_generation.generation_id,
            position=len(self._current_generation.token_traces),
            token_id=token_id,
            token_text=token_text,
            entropy=entropy,
            entropy_raw=entropy_raw,
            k_eff=k_eff,
            top_prob=top_prob,
            top5_mass=top5_mass,
            agency=agency,
            presence=presence,
            temperature=temperature,
            witness_token=witness.value,
            timestamp=time.time()
        )

        self._next_trace_id += 1
        self._current_generation.token_traces.append(trace)
        self._current_generation.output += token_text

        return trace

    def end_generation(self, f_cost: float = 0.0) -> GenerationTrace:
        """Finalize and store the current generation trace."""
        if self._current_generation is None:
            raise RuntimeError("No generation in progress")

        gen = self._current_generation
        gen.end_time = time.time()
        gen.f_cost = f_cost
        gen.total_tokens = len(gen.token_traces)

        # Compute aggregates
        if gen.token_traces:
            entropies = [t.entropy for t in gen.token_traces]
            k_effs = [t.k_eff for t in gen.token_traces]
            gen.mean_entropy = sum(entropies) / len(entropies)
            gen.mean_k_eff = sum(k_effs) / len(k_effs)
            gen.min_entropy = min(entropies)
            gen.max_entropy = max(entropies)

        # Store and rotate if needed
        self.generations.append(gen)
        if len(self.generations) > self.max_generations:
            self.generations = self.generations[-self.max_generations:]

        self._current_generation = None
        return gen

    def get_generation(self, generation_id: int) -> Optional[GenerationTrace]:
        """Get a generation trace by ID."""
        for gen in self.generations:
            if gen.generation_id == generation_id:
                return gen
        return None

    def get_recent(self, count: int = 10) -> List[GenerationTrace]:
        """Get the most recent generation traces."""
        return self.generations[-count:]

    def get_trace(self, trace_id: int) -> Optional[TokenChoiceTrace]:
        """Get a specific token choice trace by ID."""
        for gen in self.generations:
            for trace in gen.token_traces:
                if trace.trace_id == trace_id:
                    return trace
        return None

    def clear(self):
        """Clear all traces."""
        self.generations = []
        self._current_generation = None

    def stats(self) -> dict:
        """Get summary statistics for the ledger."""
        if not self.generations:
            return {"generations": 0, "total_traces": 0}

        total_traces = sum(len(g.token_traces) for g in self.generations)
        all_entropies = [t.entropy for g in self.generations for t in g.token_traces]
        all_k_effs = [t.k_eff for g in self.generations for t in g.token_traces]

        # Count witness types
        witness_counts = {}
        for g in self.generations:
            for t in g.token_traces:
                w = t.witness_token
                witness_counts[w] = witness_counts.get(w, 0) + 1

        return {
            "generations": len(self.generations),
            "total_traces": total_traces,
            "mean_entropy": sum(all_entropies) / len(all_entropies) if all_entropies else 0,
            "mean_k_eff": sum(all_k_effs) / len(all_k_effs) if all_k_effs else 0,
            "witness_counts": witness_counts
        }


class PrimitiveRegistry:
    """
    Registry of all available primitives.

    The registry manages primitive registration, cost calculation,
    and execution with proper F/agency checking.
    """

    def __init__(self):
        self.primitives: Dict[str, PrimitiveSpec] = {}
        self.call_history: List[PrimitiveCall] = []
        self.total_cost: float = 0.0

        # Token choice trace ledger (Phase 26.5)
        self.trace_ledger: TraceLedger = TraceLedger()

        # Configuration
        self.ollama_url: str = "http://localhost:11434"
        self.ollama_model: str = "llama3.2:3b"
        self.exec_timeout: int = 30  # seconds
        self.max_file_size: int = 1024 * 1024  # 1MB

        # Register built-in primitives
        self._register_builtins()

    def _register_builtins(self):
        """Register all built-in primitives."""

        # === LLM Primitives ===
        self.register(PrimitiveSpec(
            name="llm_call",
            handler=self._llm_call,
            base_cost=1.0,
            cost_per_unit=0.01,  # per output token (estimated)
            min_agency=0.3,
            description="Call LLM with a prompt, returns response text",
            arg_types=["string"],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="llm_chat",
            handler=self._llm_chat,
            base_cost=1.5,
            cost_per_unit=0.01,
            min_agency=0.3,
            description="Chat with LLM using message history",
            arg_types=["list"],
            return_type="string"
        ))

        # === Shell Primitives ===
        self.register(PrimitiveSpec(
            name="exec",
            handler=self._exec,
            base_cost=0.5,
            cost_per_unit=0.001,  # per ms of execution
            min_agency=0.5,
            description="Execute shell command, returns output",
            arg_types=["string"],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="exec_safe",
            handler=self._exec_safe,
            base_cost=0.2,
            cost_per_unit=0.001,
            min_agency=0.3,
            description="Execute safe command (read-only), returns output",
            arg_types=["string"],
            return_type="string"
        ))

        # === File Primitives ===
        self.register(PrimitiveSpec(
            name="file_read",
            handler=self._file_read,
            base_cost=0.1,
            cost_per_unit=0.0001,  # per byte
            min_agency=0.2,
            description="Read file contents",
            arg_types=["string"],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="file_write",
            handler=self._file_write,
            base_cost=0.2,
            cost_per_unit=0.0001,
            min_agency=0.5,
            description="Write content to file",
            arg_types=["string", "string"],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="file_exists",
            handler=self._file_exists,
            base_cost=0.01,
            min_agency=0.1,
            description="Check if file exists",
            arg_types=["string"],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="file_list",
            handler=self._file_list,
            base_cost=0.05,
            min_agency=0.2,
            description="List files in directory",
            arg_types=["string"],
            return_type="list"
        ))

        # === Time Primitives ===
        self.register(PrimitiveSpec(
            name="now",
            handler=self._now,
            base_cost=0.001,
            min_agency=0.0,
            description="Get current Unix timestamp",
            arg_types=[],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="now_iso",
            handler=self._now_iso,
            base_cost=0.001,
            min_agency=0.0,
            description="Get current time as ISO string",
            arg_types=[],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="sleep",
            handler=self._sleep,
            base_cost=0.01,
            cost_per_unit=0.001,  # per ms
            min_agency=0.1,
            description="Sleep for milliseconds",
            arg_types=["float"],
            return_type="void"
        ))

        # === Random Primitives ===
        self.register(PrimitiveSpec(
            name="random",
            handler=self._random,
            base_cost=0.001,
            min_agency=0.0,
            description="Get random float [0, 1)",
            arg_types=[],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="random_int",
            handler=self._random_int,
            base_cost=0.001,
            min_agency=0.0,
            description="Get random integer in range [min, max]",
            arg_types=["int", "int"],
            return_type="int"
        ))

        self.register(PrimitiveSpec(
            name="random_seed",
            handler=self._random_seed,
            base_cost=0.001,
            min_agency=0.0,
            description="Set random seed for reproducibility",
            arg_types=["int"],
            return_type="void"
        ))

        # === Debug Primitives ===
        self.register(PrimitiveSpec(
            name="print",
            handler=self._print,
            base_cost=0.001,
            min_agency=0.0,
            description="Print message to console",
            arg_types=["string"],
            return_type="void"
        ))

        self.register(PrimitiveSpec(
            name="log",
            handler=self._log,
            base_cost=0.001,
            min_agency=0.0,
            description="Log message with level",
            arg_types=["string", "string"],
            return_type="void"
        ))

        # === Hash Primitives ===
        self.register(PrimitiveSpec(
            name="hash_sha256",
            handler=self._hash_sha256,
            base_cost=0.01,
            min_agency=0.0,
            description="Compute SHA256 hash of string",
            arg_types=["string"],
            return_type="string"
        ))

        # === Terminal Primitives (Phase 19) ===
        self.register(PrimitiveSpec(
            name="terminal_read",
            handler=self._terminal_read,
            base_cost=0.01,
            min_agency=0.1,
            description="Read line of user input from terminal",
            arg_types=[],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="terminal_write",
            handler=self._terminal_write,
            base_cost=0.001,
            min_agency=0.0,
            description="Write message to terminal output",
            arg_types=["string"],
            return_type="void"
        ))

        self.register(PrimitiveSpec(
            name="terminal_prompt",
            handler=self._terminal_prompt,
            base_cost=0.01,
            min_agency=0.1,
            description="Display prompt and read user input",
            arg_types=["string"],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="terminal_clear",
            handler=self._terminal_clear,
            base_cost=0.001,
            min_agency=0.0,
            description="Clear terminal screen",
            arg_types=[],
            return_type="void"
        ))

        self.register(PrimitiveSpec(
            name="terminal_color",
            handler=self._terminal_color,
            base_cost=0.001,
            min_agency=0.0,
            description="Set terminal text color (reset, red, green, yellow, blue, magenta, cyan)",
            arg_types=["string"],
            return_type="void"
        ))

        # === Math Primitives ===
        self.register(PrimitiveSpec(
            name="eval_math",
            handler=self._eval_math,
            base_cost=0.01,
            min_agency=0.0,
            description="Safely evaluate a math expression string",
            arg_types=["string"],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="math_pow",
            handler=self._math_pow,
            base_cost=0.001,
            min_agency=0.0,
            description="Compute base raised to power",
            arg_types=["float", "float"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="math_sqrt",
            handler=self._math_sqrt,
            base_cost=0.001,
            min_agency=0.0,
            description="Compute square root",
            arg_types=["float"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="math_sin",
            handler=self._math_sin,
            base_cost=0.001,
            min_agency=0.0,
            description="Compute sine (radians)",
            arg_types=["float"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="math_cos",
            handler=self._math_cos,
            base_cost=0.001,
            min_agency=0.0,
            description="Compute cosine (radians)",
            arg_types=["float"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="math_log",
            handler=self._math_log,
            base_cost=0.001,
            min_agency=0.0,
            description="Compute natural logarithm",
            arg_types=["float"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="math_abs",
            handler=self._math_abs,
            base_cost=0.001,
            min_agency=0.0,
            description="Compute absolute value",
            arg_types=["float"],
            return_type="float"
        ))

        # Register lattice primitives (Phase 20.5)
        self._register_lattice_primitives()

        # Register inference primitives (Phase 26.3)
        self._register_inference_primitives()

    def register(self, spec: PrimitiveSpec):
        """Register a primitive."""
        self.primitives[spec.name] = spec

    def get(self, name: str) -> Optional[PrimitiveSpec]:
        """Get primitive spec by name."""
        return self.primitives.get(name)

    def list_primitives(self) -> List[Dict[str, Any]]:
        """List all registered primitives."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "base_cost": spec.base_cost,
                "min_agency": spec.min_agency,
                "arg_types": spec.arg_types,
                "return_type": spec.return_type
            }
            for spec in self.primitives.values()
        ]

    def call(self, name: str, args: List[Any],
             available_f: float, agency: float) -> PrimitiveCall:
        """
        Call a primitive with F/agency checking.

        Args:
            name: Primitive name
            args: Arguments to pass
            available_f: Creature's available F
            agency: Creature's agency level

        Returns:
            PrimitiveCall with result and cost
        """
        call = PrimitiveCall(name=name, args=args)

        spec = self.primitives.get(name)
        if not spec:
            call.result_code = PrimitiveResult.NOT_FOUND
            call.result = f"Unknown primitive: {name}"
            return call

        # Check agency
        if agency < spec.min_agency:
            call.result_code = PrimitiveResult.REFUSED
            call.result = f"Insufficient agency: need {spec.min_agency}, have {agency}"
            return call

        # Check base cost
        if available_f < spec.base_cost:
            call.result_code = PrimitiveResult.REFUSED
            call.result = f"Insufficient F: need {spec.base_cost}, have {available_f}"
            return call

        # Execute
        start_time = time.time()
        try:
            result = spec.handler(*args)
            call.result = result
            call.result_code = PrimitiveResult.OK
        except TimeoutError:
            call.result_code = PrimitiveResult.TIMEOUT
            call.result = "Execution timed out"
        except PermissionError as e:
            call.result_code = PrimitiveResult.PERMISSION
            call.result = str(e)
        except Exception as e:
            call.result_code = PrimitiveResult.ERROR
            call.result = str(e)

        call.elapsed_ms = (time.time() - start_time) * 1000

        # Calculate cost
        call.cost = spec.base_cost
        if spec.cost_per_unit > 0:
            # Add variable cost based on result size or time
            if isinstance(call.result, str):
                call.cost += len(call.result) * spec.cost_per_unit
            elif call.elapsed_ms > 0:
                call.cost += call.elapsed_ms * spec.cost_per_unit

        # Track
        self.call_history.append(call)
        self.total_cost += call.cost

        return call

    # =========================================================================
    # Primitive Implementations
    # =========================================================================

    def _llm_call(self, prompt: str) -> str:
        """Call LLM with a single prompt."""
        try:
            import requests
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}")

    def _llm_chat(self, messages: List[Dict]) -> str:
        """Chat with LLM using message history."""
        try:
            import requests
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            raise RuntimeError(f"LLM chat failed: {e}")

    def _exec(self, command: str) -> str:
        """Execute shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.exec_timeout
            )
            if result.returncode != 0:
                return f"Error ({result.returncode}): {result.stderr}"
            return result.stdout
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {self.exec_timeout}s")
        except Exception as e:
            raise RuntimeError(f"Exec failed: {e}")

    def _exec_safe(self, command: str) -> str:
        """Execute safe (read-only) command."""
        # Whitelist of safe commands
        safe_prefixes = [
            "ls", "cat", "head", "tail", "wc", "grep", "find",
            "echo", "date", "pwd", "whoami", "hostname",
            "which", "type", "file", "stat"
        ]

        cmd_base = command.strip().split()[0] if command.strip() else ""
        if not any(cmd_base.startswith(prefix) for prefix in safe_prefixes):
            raise PermissionError(f"Command not in safe list: {cmd_base}")

        # Block dangerous patterns
        dangerous = ["rm ", "mv ", "cp ", ">", ">>", "|", ";", "&", "`", "$("]
        for pattern in dangerous:
            if pattern in command:
                raise PermissionError(f"Dangerous pattern in command: {pattern}")

        return self._exec(command)

    def _file_read(self, path: str) -> str:
        """Read file contents."""
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        size = os.path.getsize(path)
        if size > self.max_file_size:
            raise ValueError(f"File too large: {size} bytes (max {self.max_file_size})")

        with open(path, 'r') as f:
            return f.read()

    def _file_write(self, path: str, content: str) -> bool:
        """Write content to file."""
        path = os.path.expanduser(path)

        if len(content) > self.max_file_size:
            raise ValueError(f"Content too large: {len(content)} bytes")

        # Create directory if needed
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(path, 'w') as f:
            f.write(content)
        return True

    def _file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return os.path.exists(os.path.expanduser(path))

    def _file_list(self, path: str) -> List[str]:
        """List files in directory."""
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        if not os.path.isdir(path):
            raise ValueError(f"Not a directory: {path}")
        return os.listdir(path)

    def _now(self) -> float:
        """Get current Unix timestamp."""
        return time.time()

    def _now_iso(self) -> str:
        """Get current time as ISO string."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _sleep(self, ms: float) -> None:
        """Sleep for milliseconds."""
        time.sleep(ms / 1000.0)

    def _random(self) -> float:
        """Get random float [0, 1)."""
        return random.random()

    def _random_int(self, min_val: int, max_val: int) -> int:
        """Get random integer in range."""
        return random.randint(int(min_val), int(max_val))

    def _random_seed(self, seed: int) -> None:
        """Set random seed."""
        random.seed(int(seed))

    def _print(self, msg: str) -> None:
        """Print message."""
        print(f"[PRIMITIVE] {msg}")

    def _log(self, level: str, msg: str) -> None:
        """Log message with level."""
        print(f"[{level.upper()}] {msg}")

    def _hash_sha256(self, data: str) -> str:
        """Compute SHA256 hash."""
        return hashlib.sha256(data.encode()).hexdigest()

    # =========================================================================
    # Terminal Primitives (Phase 19)
    # =========================================================================

    def _terminal_read(self) -> str:
        """Read line of input from terminal."""
        try:
            return input()
        except EOFError:
            return ""

    def _terminal_write(self, msg: str) -> None:
        """Write message to terminal (no newline)."""
        import sys
        sys.stdout.write(str(msg))
        sys.stdout.flush()

    def _terminal_prompt(self, prompt: str) -> str:
        """Display prompt and read user input."""
        try:
            return input(prompt)
        except EOFError:
            return ""

    def _terminal_clear(self) -> None:
        """Clear terminal screen."""
        import sys
        if sys.platform == "win32":
            os.system("cls")
        else:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()

    def _terminal_color(self, color: str) -> None:
        """Set terminal text color."""
        import sys
        colors = {
            "reset": "\033[0m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "bold": "\033[1m",
            "dim": "\033[2m",
        }
        code = colors.get(color.lower(), colors["reset"])
        sys.stdout.write(code)
        sys.stdout.flush()

    # === Math Primitive Handlers ===

    def _eval_math(self, expr: str) -> str:
        """
        Safely evaluate a math expression.

        Supports: +, -, *, /, **, (), sqrt, sin, cos, tan, log, abs, pi, e
        """
        import math
        import re

        # Clean the expression
        expr = str(expr).strip()
        if not expr:
            return "Error: empty expression"

        # Define safe functions and constants
        safe_dict = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "abs": abs,
            "pow": pow,
            "pi": math.pi,
            "e": math.e,
            "floor": math.floor,
            "ceil": math.ceil,
            "round": round,
        }

        # Validate expression contains only allowed characters
        allowed = set("0123456789.+-*/() ,")
        allowed_words = set(safe_dict.keys())

        # Extract words from expression
        words = re.findall(r'[a-zA-Z_]+', expr)
        for word in words:
            if word not in allowed_words:
                return f"Error: unknown function '{word}'"

        # Check for dangerous patterns
        if "__" in expr or "import" in expr or "eval" in expr or "exec" in expr:
            return "Error: invalid expression"

        try:
            # Evaluate with restricted globals
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            # Format result
            if isinstance(result, float):
                if result == int(result):
                    return str(int(result))
                return f"{result:.10g}"
            return str(result)
        except ZeroDivisionError:
            return "Error: division by zero"
        except ValueError as e:
            return f"Error: {e}"
        except SyntaxError:
            return "Error: invalid syntax"
        except Exception as e:
            return f"Error: {e}"

    def _math_pow(self, base: float, exp: float) -> float:
        """Compute base raised to power."""
        return pow(float(base), float(exp))

    def _math_sqrt(self, value: float) -> float:
        """Compute square root."""
        import math
        return math.sqrt(float(value))

    def _math_sin(self, value: float) -> float:
        """Compute sine (radians)."""
        import math
        return math.sin(float(value))

    def _math_cos(self, value: float) -> float:
        """Compute cosine (radians)."""
        import math
        return math.cos(float(value))

    def _math_log(self, value: float) -> float:
        """Compute natural logarithm."""
        import math
        return math.log(float(value))

    def _math_abs(self, value: float) -> float:
        """Compute absolute value."""
        return abs(float(value))

    # =========================================================================
    # Lattice / Collider Primitives (Phase 20.5)
    # Now uses C substrate backend when available for performance
    # =========================================================================

    def _init_lattice_backend(self):
        """Initialize the lattice backend (C or Python fallback)."""
        if hasattr(self, '_lattice_backend_initialized'):
            return

        self._use_c_lattice = False
        self._c_lattices = {}  # Maps lattice_id -> CLattice instance

        try:
            from det.lattice_c import CLattice, is_available
            if is_available():
                self._use_c_lattice = True
                self._CLattice = CLattice
                print("[Primitives] Using C lattice substrate")
        except ImportError:
            pass

        if not self._use_c_lattice:
            print("[Primitives] Using Python lattice fallback")

        self._lattice_backend_initialized = True

    def _register_lattice_primitives(self):
        """Register lattice primitives for native DET collider."""

        self.register(PrimitiveSpec(
            name="lattice_create",
            handler=self._lattice_create,
            base_cost=0.1,
            min_agency=0.2,
            description="Create a DET lattice (dim, N, params...)",
            arg_types=["int", "int"],
            return_type="int"
        ))

        self.register(PrimitiveSpec(
            name="lattice_destroy",
            handler=self._lattice_destroy,
            base_cost=0.01,
            min_agency=0.1,
            description="Destroy a lattice by ID",
            arg_types=["int"],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="lattice_step",
            handler=self._lattice_step,
            base_cost=0.05,
            min_agency=0.1,
            description="Execute N physics steps on lattice",
            arg_types=["int", "int"],
            return_type="int"
        ))

        self.register(PrimitiveSpec(
            name="lattice_add_packet",
            handler=self._lattice_add_packet,
            base_cost=0.02,
            min_agency=0.1,
            description="Add resource packet (id, pos, mass, width, momentum, q)",
            arg_types=["int", "list", "float", "float"],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="lattice_total_mass",
            handler=self._lattice_total_mass,
            base_cost=0.01,
            min_agency=0.0,
            description="Get total mass in lattice",
            arg_types=["int"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="lattice_total_q",
            handler=self._lattice_total_q,
            base_cost=0.01,
            min_agency=0.0,
            description="Get total structure in lattice",
            arg_types=["int"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="lattice_separation",
            handler=self._lattice_separation,
            base_cost=0.02,
            min_agency=0.0,
            description="Get separation between two largest blobs",
            arg_types=["int"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="lattice_potential_energy",
            handler=self._lattice_potential_energy,
            base_cost=0.01,
            min_agency=0.0,
            description="Get gravitational potential energy",
            arg_types=["int"],
            return_type="float"
        ))

        self.register(PrimitiveSpec(
            name="lattice_center_of_mass",
            handler=self._lattice_center_of_mass,
            base_cost=0.01,
            min_agency=0.0,
            description="Get center of mass position",
            arg_types=["int"],
            return_type="list"
        ))

        self.register(PrimitiveSpec(
            name="lattice_render",
            handler=self._lattice_render,
            base_cost=0.02,
            min_agency=0.0,
            description="Render lattice field as ASCII art",
            arg_types=["int", "string", "int"],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="lattice_get_stats",
            handler=self._lattice_get_stats,
            base_cost=0.01,
            min_agency=0.0,
            description="Get lattice statistics",
            arg_types=["int"],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="lattice_set_param",
            handler=self._lattice_set_param,
            base_cost=0.01,
            min_agency=0.1,
            description="Set lattice parameter",
            arg_types=["int", "string", "float"],
            return_type="bool"
        ))

    def _lattice_create(self, dim: int = 1, N: int = 100, **kwargs) -> int:
        """Create a new DET lattice."""
        self._init_lattice_backend()

        if self._use_c_lattice:
            # Use C substrate
            lattice = self._CLattice(dim=int(dim), size=int(N), **kwargs)
            # Generate unique ID
            lattice_id = id(lattice) & 0xFFFFFF  # Truncate to reasonable ID
            self._c_lattices[lattice_id] = lattice
            return lattice_id
        else:
            # Fallback to Python
            from .lattice import lattice_create
            return lattice_create(int(dim), int(N), **kwargs)

    def _lattice_destroy(self, lattice_id: int) -> bool:
        """Destroy a lattice."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            del self._c_lattices[int(lattice_id)]
            return True
        else:
            from .lattice import lattice_destroy
            return lattice_destroy(int(lattice_id))

    def _lattice_step(self, lattice_id: int, num_steps: int = 1) -> int:
        """Execute physics steps on lattice."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            lattice = self._c_lattices[int(lattice_id)]
            lattice.step(int(num_steps))
            return lattice.get_stats().step_count
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            for _ in range(int(num_steps)):
                lattice.step()
            return lattice.step_count

    def _lattice_add_packet(self, lattice_id: int, pos: list,
                            mass: float = 5.0, width: float = 5.0,
                            momentum: list = None, initial_q: float = 0.0) -> bool:
        """Add resource packet to lattice."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            lattice = self._c_lattices[int(lattice_id)]
            center = list(pos) if pos else [lattice.size // 2] * lattice.dim
            mom = list(momentum) if momentum else None
            lattice.add_packet(center, float(mass), float(width), mom, float(initial_q))
            return True
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            center = tuple(pos) if pos else (lattice.N // 2,) * lattice.dim
            mom = tuple(momentum) if momentum else None
            lattice.add_packet(center, float(mass), float(width), mom, float(initial_q))
            return True

    def _lattice_total_mass(self, lattice_id: int) -> float:
        """Get total mass in lattice."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            return self._c_lattices[int(lattice_id)].total_mass()
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            return lattice.total_mass()

    def _lattice_total_q(self, lattice_id: int) -> float:
        """Get total structure in lattice."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            return self._c_lattices[int(lattice_id)].get_stats().total_structure
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            return lattice.total_q()

    def _lattice_separation(self, lattice_id: int) -> float:
        """Get separation between blobs."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            return self._c_lattices[int(lattice_id)].separation()
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            return lattice.separation()

    def _lattice_potential_energy(self, lattice_id: int) -> float:
        """Get potential energy."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            return self._c_lattices[int(lattice_id)].get_stats().potential_energy
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            return lattice.potential_energy()

    def _lattice_center_of_mass(self, lattice_id: int) -> list:
        """Get center of mass."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            return self._c_lattices[int(lattice_id)].get_stats().center_of_mass
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            return list(lattice.center_of_mass())

    def _lattice_render(self, lattice_id: int, field: str = "F", width: int = 60) -> str:
        """Render lattice as ASCII."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            # Map field name to render field constant
            from det.lattice_c import (RENDER_FIELD_F, RENDER_FIELD_Q,
                                        RENDER_FIELD_A, RENDER_FIELD_P)
            field_map = {
                'F': RENDER_FIELD_F, 'f': RENDER_FIELD_F,
                'Q': RENDER_FIELD_Q, 'q': RENDER_FIELD_Q,
                'A': RENDER_FIELD_A, 'a': RENDER_FIELD_A,
                'P': RENDER_FIELD_P, 'p': RENDER_FIELD_P,
            }
            render_field = field_map.get(field, RENDER_FIELD_F)
            return self._c_lattices[int(lattice_id)].render(render_field, int(width))
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            return lattice.render_ascii(field, int(width))

    def _lattice_get_stats(self, lattice_id: int) -> dict:
        """Get lattice statistics."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            lattice = self._c_lattices[int(lattice_id)]
            stats = lattice.get_stats()
            return {
                "dim": lattice.dim,
                "N": lattice.size,
                "step_count": stats.step_count,
                "time": stats.step_count * 0.02,  # Approximate from step count
                "total_mass": stats.total_mass,
                "total_q": stats.total_structure,
                "total_grace": 0.0,  # Not tracked in C substrate yet
                "eta": 1.0,  # Lattice correction
            }
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            return {
                "dim": lattice.dim,
                "N": lattice.N,
                "step_count": lattice.step_count,
                "time": lattice.time,
                "total_mass": lattice.total_mass(),
                "total_q": lattice.total_q(),
                "total_grace": lattice.total_grace,
                "eta": lattice.eta,
            }

    def _lattice_set_param(self, lattice_id: int, param: str, value: float) -> bool:
        """Set a lattice parameter."""
        self._init_lattice_backend()

        if self._use_c_lattice and int(lattice_id) in self._c_lattices:
            return self._c_lattices[int(lattice_id)].set_param(str(param), float(value))
        else:
            from .lattice import lattice_get
            lattice = lattice_get(int(lattice_id))
            if lattice is None:
                raise ValueError(f"Lattice not found: {lattice_id}")
            if hasattr(lattice.p, param):
                setattr(lattice.p, param, value)
                return True
            return False

    # =========================================================================
    # Inference Primitives (Phase 26.3)
    # Native LLM inference using DET inference library
    # =========================================================================

    def _init_inference_backend(self):
        """Initialize inference backend."""
        if hasattr(self, '_inference_backend_initialized'):
            return

        self._use_native_inference = False
        self._inference_model = None

        try:
            from det.inference import Model, metal_available, metal_init
            self._InferenceModel = Model
            self._metal_available = metal_available
            self._metal_init = metal_init
            self._use_native_inference = True
            print("[Primitives] Native inference library available")

            # Try to initialize Metal GPU
            if metal_available():
                if metal_init():
                    print("[Primitives] Metal GPU acceleration enabled")
        except ImportError as e:
            print(f"[Primitives] Native inference not available: {e}")

        self._inference_backend_initialized = True

    def _register_inference_primitives(self):
        """Register inference primitives for native LLM inference."""

        # === Model Loading ===
        self.register(PrimitiveSpec(
            name="model_load",
            handler=self._model_load,
            base_cost=0.5,
            min_agency=0.3,
            description="Load GGUF model from path",
            arg_types=["string"],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="model_info",
            handler=self._model_info,
            base_cost=0.01,
            min_agency=0.0,
            description="Get loaded model info",
            arg_types=[],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="model_reset",
            handler=self._model_reset,
            base_cost=0.01,
            min_agency=0.1,
            description="Reset model KV cache for new conversation",
            arg_types=[],
            return_type="void"
        ))

        # === KV Cache Management (Phase 26.4) ===
        self.register(PrimitiveSpec(
            name="model_cache_status",
            handler=self._model_cache_status,
            base_cost=0.001,
            min_agency=0.0,
            description="Get KV cache status (position, capacity, usage)",
            arg_types=[],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="model_cache_shift",
            handler=self._model_cache_shift,
            base_cost=0.02,
            min_agency=0.1,
            description="Shift cache to keep last N tokens (sliding window)",
            arg_types=["int"],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="model_cache_slice",
            handler=self._model_cache_slice,
            base_cost=0.02,
            min_agency=0.1,
            description="Slice cache to keep positions [start, end)",
            arg_types=["int", "int"],
            return_type="bool"
        ))

        # === Tokenization ===
        self.register(PrimitiveSpec(
            name="model_tokenize",
            handler=self._model_tokenize,
            base_cost=0.01,
            cost_per_unit=0.0001,  # per character
            min_agency=0.0,
            description="Convert text to token IDs",
            arg_types=["string"],
            return_type="list"
        ))

        self.register(PrimitiveSpec(
            name="model_detokenize",
            handler=self._model_detokenize,
            base_cost=0.01,
            cost_per_unit=0.0001,  # per token
            min_agency=0.0,
            description="Convert token IDs to text",
            arg_types=["list"],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="model_token_text",
            handler=self._model_token_text,
            base_cost=0.001,
            min_agency=0.0,
            description="Get text for a single token ID",
            arg_types=["int"],
            return_type="string"
        ))

        # === Inference ===
        self.register(PrimitiveSpec(
            name="model_forward",
            handler=self._model_forward,
            base_cost=0.1,
            cost_per_unit=0.01,  # per token
            min_agency=0.2,
            description="Run forward pass on tokens, returns logits",
            arg_types=["list"],
            return_type="list"
        ))

        self.register(PrimitiveSpec(
            name="model_sample",
            handler=self._model_sample,
            base_cost=0.01,
            min_agency=0.1,
            description="Sample next token from logits (temp, top_p)",
            arg_types=["list", "float", "float"],
            return_type="int"
        ))

        # === DET-Aware Sampling (Sacred Integration Point) ===
        self.register(PrimitiveSpec(
            name="det_choose_token",
            handler=self._det_choose_token,
            base_cost=0.02,
            min_agency=0.1,
            description="DET-aware token selection with presence bias",
            arg_types=["list", "float", "float", "list"],
            return_type="int"
        ))

        # === High-Level Generation ===
        self.register(PrimitiveSpec(
            name="model_generate",
            handler=self._model_generate,
            base_cost=1.0,
            cost_per_unit=0.02,  # per generated token
            min_agency=0.3,
            description="Generate text from prompt",
            arg_types=["string", "int"],
            return_type="string"
        ))

        self.register(PrimitiveSpec(
            name="model_chat",
            handler=self._model_chat,
            base_cost=1.0,
            cost_per_unit=0.02,
            min_agency=0.3,
            description="Chat generation with template, stats, and structured output",
            arg_types=["string", "string", "int", "float", "float"],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="model_generate_step",
            handler=self._model_generate_step,
            base_cost=0.15,
            min_agency=0.2,
            description="Generate single token (for streaming)",
            arg_types=["list", "float", "float"],
            return_type="dict"
        ))

        # === Per-Token Stats (Phase 26.6) ===
        self.register(PrimitiveSpec(
            name="model_stats_start",
            handler=self._model_stats_start,
            base_cost=0.001,
            min_agency=0.0,
            description="Start collecting per-token stats",
            arg_types=["int"],
            return_type="void"
        ))

        self.register(PrimitiveSpec(
            name="model_stats_aggregate",
            handler=self._model_stats_aggregate,
            base_cost=0.001,
            min_agency=0.0,
            description="Get aggregated stats (mean_entropy, mean_k_eff, min_entropy)",
            arg_types=[],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="model_stats_clear",
            handler=self._model_stats_clear,
            base_cost=0.001,
            min_agency=0.0,
            description="Clear stats buffer",
            arg_types=[],
            return_type="void"
        ))

        # === Token Choice Trace Primitives (Phase 26.5) ===
        self.register(PrimitiveSpec(
            name="trace_start_generation",
            handler=self._trace_start_generation,
            base_cost=0.001,
            min_agency=0.0,
            description="Start a new generation trace",
            arg_types=["string", "float", "float"],
            return_type="int"
        ))

        self.register(PrimitiveSpec(
            name="trace_record_choice",
            handler=self._trace_record_choice,
            base_cost=0.001,
            min_agency=0.0,
            description="Record a token choice as witness trace",
            arg_types=["int", "string", "float", "float", "int", "float", "float", "float", "float", "float"],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="trace_end_generation",
            handler=self._trace_end_generation,
            base_cost=0.001,
            min_agency=0.0,
            description="Finalize and store current generation trace",
            arg_types=["float"],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="trace_get_generation",
            handler=self._trace_get_generation,
            base_cost=0.001,
            min_agency=0.0,
            description="Get a generation trace by ID",
            arg_types=["int"],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="trace_get_recent",
            handler=self._trace_get_recent,
            base_cost=0.001,
            min_agency=0.0,
            description="Get recent generation traces",
            arg_types=["int"],
            return_type="list"
        ))

        self.register(PrimitiveSpec(
            name="trace_stats",
            handler=self._trace_stats,
            base_cost=0.001,
            min_agency=0.0,
            description="Get trace ledger statistics",
            arg_types=[],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="trace_clear",
            handler=self._trace_clear,
            base_cost=0.001,
            min_agency=0.0,
            description="Clear all traces",
            arg_types=[],
            return_type="void"
        ))

        # === GPU Status ===
        self.register(PrimitiveSpec(
            name="metal_status",
            handler=self._metal_status,
            base_cost=0.001,
            min_agency=0.0,
            description="Get Metal GPU status",
            arg_types=[],
            return_type="dict"
        ))

        # === Truthfulness Primitives (Phase 26.6) ===
        self.register(PrimitiveSpec(
            name="truth_reset",
            handler=self._truth_reset,
            base_cost=0.001,
            min_agency=0.0,
            description="Reset truthfulness evaluator for new generation",
            arg_types=[],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="truth_record_claim",
            handler=self._truth_record_claim,
            base_cost=0.001,
            min_agency=0.0,
            description="Record a claim with F cost",
            arg_types=["float", "float"],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="truth_set_grounding",
            handler=self._truth_set_grounding,
            base_cost=0.001,
            min_agency=0.0,
            description="Set grounding signals (delta_f, stability, c_user, violations)",
            arg_types=["float", "float", "float", "int"],
            return_type="bool"
        ))

        self.register(PrimitiveSpec(
            name="truth_evaluate",
            handler=self._truth_evaluate,
            base_cost=0.01,
            min_agency=0.0,
            description="Evaluate truthfulness from DET state",
            arg_types=["float", "float", "int", "float", "int"],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="truth_get_weights",
            handler=self._truth_get_weights,
            base_cost=0.001,
            min_agency=0.0,
            description="Get truthfulness component weights",
            arg_types=[],
            return_type="dict"
        ))

        self.register(PrimitiveSpec(
            name="truth_get_falsifiers",
            handler=self._truth_get_falsifiers,
            base_cost=0.001,
            min_agency=0.0,
            description="Get last falsifier check results",
            arg_types=[],
            return_type="dict"
        ))

    def _model_load(self, path: str) -> bool:
        """Load GGUF model from path."""
        self._init_inference_backend()

        if not self._use_native_inference:
            raise RuntimeError("Native inference not available")

        path = os.path.expanduser(str(path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        self._inference_model = self._InferenceModel(path)
        return True

    def _model_info(self) -> str:
        """Get loaded model info."""
        self._init_inference_backend()

        if self._inference_model is None:
            return "No model loaded"

        return self._inference_model.info

    def _model_reset(self) -> None:
        """Reset model KV cache."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        self._inference_model.reset()

    def _model_cache_status(self) -> dict:
        """Get KV cache status (Phase 26.4)."""
        self._init_inference_backend()

        if self._inference_model is None:
            return {
                "position": 0,
                "capacity": 0,
                "usage": 0.0,
                "remaining": 0,
                "loaded": False
            }

        info = self._inference_model.cache_info()
        info["loaded"] = True
        return info

    def _model_cache_shift(self, keep_last: int) -> bool:
        """Shift cache to keep last N tokens (Phase 26.4)."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        return self._inference_model.cache_shift(int(keep_last))

    def _model_cache_slice(self, start: int, end: int) -> bool:
        """Slice cache to keep positions [start, end) (Phase 26.4)."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        return self._inference_model.cache_slice(int(start), int(end))

    def _model_tokenize(self, text: str) -> list:
        """Tokenize text to token IDs."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        return self._inference_model.tokenize(str(text))

    def _model_detokenize(self, tokens: list) -> str:
        """Detokenize token IDs to text."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        return self._inference_model.detokenize([int(t) for t in tokens])

    def _model_token_text(self, token_id: int) -> str:
        """Get text for single token."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        return self._inference_model.token_to_text(int(token_id))

    def _model_forward(self, tokens: list) -> list:
        """Run forward pass, return logits."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        # Note: Forward returns a tensor pointer; we need to extract logits
        # For now, store internally and return token count processed
        token_list = [int(t) for t in tokens]
        self._last_logits = self._inference_model.forward(token_list)
        return [len(token_list)]  # Return count; logits stored internally

    def _model_sample(self, logits: list, temperature: float = 0.7,
                      top_p: float = 0.9) -> int:
        """Sample next token from logits."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        if not hasattr(self, '_last_logits') or self._last_logits is None:
            raise RuntimeError("No logits available - call model_forward first")

        from det.inference import SamplingParams
        params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p)
        )
        return self._inference_model.sample(self._last_logits, params)

    def _det_choose_token(self, logits: list, temperature: float = 0.7,
                          top_p: float = 0.9, det_presence: list = None) -> int:
        """
        DET-aware token selection.

        This is the SACRED INTEGRATION POINT where DET physics
        can influence token selection via presence values.

        The presence values act as a bias on the logits before sampling,
        allowing the substrate's computed presence field to guide generation.

        Args:
            logits: Raw logits from forward pass
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            det_presence: DET presence bias per token (optional)

        Returns:
            Selected token ID
        """
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        # Convert inputs
        logits_list = [float(x) for x in logits]
        presence_list = [float(x) for x in det_presence] if det_presence else None

        return self._inference_model.choose_token(
            logits_list,
            float(temperature),
            float(top_p),
            presence_list
        )

    def _model_generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text from prompt."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        return self._inference_model.generate(str(prompt), int(max_tokens))

    def _model_chat(self, user_message: str, system_message: str = "",
                    max_tokens: int = 256, temperature: float = 0.7,
                    top_p: float = 0.9, agency: float = 0.7,
                    presence: float = 1.0) -> dict:
        """
        Chat generation with template, stats, trace recording, and structured output.

        This is the recommended primitive for chat-style generation.
        Handles: chat template formatting, stats collection, trace recording, sampling params.

        Args:
            user_message: User's input message
            system_message: System prompt (empty = use default)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            agency: DET agency for trace recording
            presence: DET presence for trace recording

        Returns:
            dict with:
                - text: Generated response text
                - token_count: Number of tokens generated
                - stats: {mean_entropy, mean_k_eff, min_entropy}
                - trace: {generation_id, witness_counts}
        """
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        from det.inference import SamplingParams, get_chat_template, detect_template_from_vocab

        # Get chat template
        chat_template = detect_template_from_vocab(self._inference_model)
        if chat_template is None:
            chat_template = get_chat_template("qwen")  # Default to Qwen/ChatML

        # Format prompt with template
        if system_message:
            formatted_prompt = chat_template.format_prompt(str(user_message), str(system_message))
        else:
            formatted_prompt = chat_template.format_prompt(str(user_message))

        # Set up sampling params
        params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p)
        )

        # Start stats collection and trace (Phase 26.5)
        self._inference_model.stats_start(capacity=512)
        gen_id = self.trace_ledger.start_generation(
            user_message[:200],  # Truncate for storage
            float(agency),
            float(presence)
        )

        # Generate (non-streaming)
        token_texts = []
        def collect_tokens(text, token_id):
            token_texts.append((text, token_id))

        text = self._inference_model.generate(
            formatted_prompt,
            max_tokens=int(max_tokens),
            params=params,
            callback=collect_tokens
        )

        # Get per-token stats and record traces (Phase 26.5)
        token_stats = self._inference_model.stats_get()

        # Record each token choice as a trace
        for i, stat in enumerate(token_stats):
            token_text = token_texts[i][0] if i < len(token_texts) else ""
            self.trace_ledger.record_choice(
                token_id=stat['token_id'],
                token_text=token_text,
                entropy=stat['entropy'],
                entropy_raw=stat['entropy_raw'],
                k_eff=stat['k_eff'],
                top_prob=stat['top_prob'],
                top5_mass=stat['top5_mass'],
                agency=float(agency),
                presence=float(presence),
                temperature=float(temperature)
            )

        # End trace and get summary
        trace_summary = self.trace_ledger.end_generation(f_cost=1.0)

        # Get aggregate stats
        stats = self._inference_model.stats_aggregate()

        return {
            "text": text,
            "token_count": len(token_texts),
            "stats": stats,
            "trace": {
                "generation_id": trace_summary["generation_id"],
                "witness_counts": trace_summary["witness_counts"]
            }
        }

    def _model_generate_step(self, tokens: list, temperature: float = 0.7,
                             top_p: float = 0.9) -> dict:
        """
        Generate single token (for streaming/step-by-step generation).

        Returns dict with:
            - token_id: Generated token ID
            - token_text: Text for this token
            - is_eos: Whether this is end-of-sequence
        """
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        token_list = [int(t) for t in tokens]

        # Forward pass
        logits = self._inference_model.forward(token_list)

        # Sample
        from det.inference import SamplingParams
        params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p)
        )
        token_id = self._inference_model.sample(logits, params)

        # Check for EOS
        is_eos = token_id == self._inference_model.eos_token or token_id < 0

        return {
            "token_id": token_id,
            "token_text": self._inference_model.token_to_text(token_id) if not is_eos else "",
            "is_eos": is_eos
        }

    def _model_stats_start(self, capacity: int = 512) -> None:
        """Start collecting per-token stats (Phase 26.6)."""
        self._init_inference_backend()

        if self._inference_model is None:
            raise RuntimeError("No model loaded")

        self._inference_model.stats_start(int(capacity))

    def _model_stats_aggregate(self) -> dict:
        """Get aggregated stats from last generation (Phase 26.6)."""
        self._init_inference_backend()

        if self._inference_model is None:
            return {
                "mean_entropy": 0.0,
                "mean_k_eff": 0.0,
                "min_entropy": 0.0
            }

        return self._inference_model.stats_aggregate()

    def _model_stats_clear(self) -> None:
        """Clear stats buffer (Phase 26.6)."""
        self._init_inference_backend()

        if self._inference_model is None:
            return

        self._inference_model.stats_clear()

    def _metal_status(self) -> dict:
        """Get Metal GPU status."""
        self._init_inference_backend()

        if not self._use_native_inference:
            return {
                "available": False,
                "device": "None",
                "reason": "Native inference not loaded"
            }

        from det.inference import metal_available, metal_device_name
        available = metal_available()
        return {
            "available": available,
            "device": metal_device_name() if available else "None",
            "reason": "OK" if available else "Metal not available"
        }

    # =========================================================================
    # Truthfulness Primitives (Phase 26.6)
    # =========================================================================

    def _init_truthfulness(self):
        """Initialize truthfulness evaluator lazily."""
        if not hasattr(self, '_truthfulness_evaluator'):
            self._truthfulness_evaluator = None
            self._last_truth_score = None

        if self._truthfulness_evaluator is None:
            try:
                from det.inference import get_truthfulness_evaluator
                self._truthfulness_evaluator = get_truthfulness_evaluator()
            except ImportError:
                pass

    def _truth_reset(self) -> bool:
        """Reset truthfulness evaluator for new generation."""
        self._init_truthfulness()
        if self._truthfulness_evaluator is None:
            return False
        self._truthfulness_evaluator.reset_generation()
        self._last_truth_score = None
        return True

    def _truth_record_claim(self, f_cost: float, min_cost: float = 0.1) -> bool:
        """Record a claim with F cost during generation."""
        self._init_truthfulness()
        if self._truthfulness_evaluator is None:
            return False
        self._truthfulness_evaluator.record_claim(float(f_cost), float(min_cost))
        return True

    def _truth_set_grounding(self, delta_f: float = 0.0, stability: float = 1.0,
                              c_user: float = 1.0, violations: int = 0) -> bool:
        """Set grounding signals for truthfulness evaluation."""
        self._init_truthfulness()
        if self._truthfulness_evaluator is None:
            return False
        self._truthfulness_evaluator.set_grounding_signals(
            delta_f=float(delta_f),
            stability=float(stability),
            c_user=float(c_user),
            violations=int(violations)
        )
        return True

    def _truth_evaluate(self, agency: float = 0.5, entropy: float = 0.0,
                        k_eff: int = 100, q_creature: float = 0.0,
                        num_tokens: int = 0) -> dict:
        """
        Evaluate truthfulness from DET state.

        Returns dict with:
            - total: Overall truthfulness score [0, 1]
            - confidence: 'high', 'medium', 'low', 'very_low'
            - grounding_factor: G value
            - q_claim: Epistemic debt
            - components: Dict of component scores
            - falsifiers: Dict of falsifier flags
        """
        self._init_truthfulness()
        if self._truthfulness_evaluator is None:
            return {
                "total": 0.5,
                "confidence": "unknown",
                "grounding_factor": 0.0,
                "q_claim": 0.0,
                "components": {},
                "falsifiers": {},
                "error": "Truthfulness evaluator not available"
            }

        score = self._truthfulness_evaluator.evaluate(
            agency=float(agency),
            entropy=float(entropy),
            k_eff=int(k_eff),
            q_creature=float(q_creature),
            num_tokens=int(num_tokens)
        )

        self._last_truth_score = score

        return {
            "total": score.total,
            "confidence": score.confidence_level,
            "grounding_factor": score.grounding_factor,
            "q_claim": score.q_claim,
            "q_creature": score.q_creature,
            "agency": score.agency,
            "entropy": score.entropy,
            "entropy_normalized": score.entropy_normalized,
            "k_eff": score.k_eff,
            "coherence_user": score.coherence_user,
            "num_tokens": score.num_tokens,
            "components": {
                "grounding": score.grounding_component,
                "agency": score.agency_component,
                "consistency": score.consistency_component,
                "coherence": score.coherence_component
            },
            "falsifiers": score.falsifier_flags or {}
        }

    def _truth_get_weights(self) -> dict:
        """Get truthfulness component weights."""
        self._init_truthfulness()
        if self._truthfulness_evaluator is None:
            return {"error": "Truthfulness evaluator not available"}

        w = self._truthfulness_evaluator.weights
        return {
            "w_grounding": w.w_grounding,
            "w_agency": w.w_agency,
            "w_consistency": w.w_consistency,
            "w_coherence": w.w_coherence
        }

    def _truth_get_falsifiers(self) -> dict:
        """Get last falsifier check results."""
        self._init_truthfulness()
        if self._last_truth_score is None:
            return {"error": "No evaluation performed yet"}

        return self._last_truth_score.falsifier_flags or {}

    # =========================================================================
    # Token Choice Trace Primitives (Phase 26.5)
    # =========================================================================

    def _trace_start_generation(self, prompt: str, agency: float = 0.7,
                                 presence: float = 1.0) -> int:
        """Start a new generation trace."""
        return self.trace_ledger.start_generation(str(prompt), float(agency), float(presence))

    def _trace_record_choice(self, token_id: int, token_text: str,
                              entropy: float, entropy_raw: float, k_eff: int,
                              top_prob: float, top5_mass: float,
                              agency: float, presence: float, temperature: float) -> dict:
        """
        Record a token choice as a witness trace.

        This is the sacred integration point where token choices become
        DET-compliant witnesses committed to the trace ledger.
        """
        trace = self.trace_ledger.record_choice(
            token_id=int(token_id),
            token_text=str(token_text),
            entropy=float(entropy),
            entropy_raw=float(entropy_raw),
            k_eff=int(k_eff),
            top_prob=float(top_prob),
            top5_mass=float(top5_mass),
            agency=float(agency),
            presence=float(presence),
            temperature=float(temperature)
        )

        # Return witness info
        return {
            "trace_id": trace.trace_id,
            "witness_token": trace.witness_token,
            "witness_name": TokenChoiceWitness(trace.witness_token).name,
            "entropy": trace.entropy,
            "k_eff": trace.k_eff
        }

    def _trace_end_generation(self, f_cost: float = 0.0) -> dict:
        """Finalize and store current generation trace."""
        gen = self.trace_ledger.end_generation(float(f_cost))

        # Count witnesses by type
        witness_counts = {}
        for t in gen.token_traces:
            w = t.witness_token
            witness_counts[w] = witness_counts.get(w, 0) + 1

        return {
            "generation_id": gen.generation_id,
            "total_tokens": gen.total_tokens,
            "mean_entropy": gen.mean_entropy,
            "mean_k_eff": gen.mean_k_eff,
            "min_entropy": gen.min_entropy,
            "max_entropy": gen.max_entropy,
            "f_cost": gen.f_cost,
            "duration_ms": (gen.end_time - gen.start_time) * 1000,
            "witness_counts": witness_counts
        }

    def _trace_get_generation(self, generation_id: int) -> dict:
        """Get a generation trace by ID."""
        gen = self.trace_ledger.get_generation(int(generation_id))
        if gen is None:
            return {"error": f"Generation {generation_id} not found"}

        return {
            "generation_id": gen.generation_id,
            "prompt": gen.prompt[:100] + "..." if len(gen.prompt) > 100 else gen.prompt,
            "output": gen.output[:200] + "..." if len(gen.output) > 200 else gen.output,
            "total_tokens": gen.total_tokens,
            "mean_entropy": gen.mean_entropy,
            "mean_k_eff": gen.mean_k_eff,
            "f_cost": gen.f_cost,
            "traces": [
                {
                    "trace_id": t.trace_id,
                    "position": t.position,
                    "token_id": t.token_id,
                    "token_text": t.token_text,
                    "entropy": t.entropy,
                    "k_eff": t.k_eff,
                    "witness": TokenChoiceWitness(t.witness_token).name
                }
                for t in gen.token_traces[:50]  # Limit to first 50 for display
            ]
        }

    def _trace_get_recent(self, count: int = 10) -> list:
        """Get recent generation traces."""
        gens = self.trace_ledger.get_recent(int(count))
        return [
            {
                "generation_id": g.generation_id,
                "total_tokens": g.total_tokens,
                "mean_entropy": g.mean_entropy,
                "output_preview": g.output[:50] + "..." if len(g.output) > 50 else g.output
            }
            for g in gens
        ]

    def _trace_stats(self) -> dict:
        """Get trace ledger statistics."""
        return self.trace_ledger.stats()

    def _trace_clear(self) -> None:
        """Clear all traces."""
        self.trace_ledger.clear()


# Global registry instance
_registry: Optional[PrimitiveRegistry] = None


def get_registry() -> PrimitiveRegistry:
    """Get the global primitive registry."""
    global _registry
    if _registry is None:
        _registry = PrimitiveRegistry()
    return _registry


def get_shared_model():
    """
    Get the shared model instance from the primitive registry.

    This allows det_os_boot.py to use the same model as the primitives,
    enabling proper DET architecture while still allowing direct model
    access for streaming and stats collection.

    Returns:
        Model instance or None if not loaded
    """
    reg = get_registry()
    reg._init_inference_backend()
    return reg._inference_model


def load_shared_model(path: str):
    """
    Load a model through the primitive registry.

    This ensures the model is shared between det_os_boot.py and primitives.

    Args:
        path: Path to GGUF model file

    Returns:
        Model instance

    Raises:
        RuntimeError: If native inference not available
        FileNotFoundError: If model file not found
    """
    reg = get_registry()
    reg._model_load(path)
    return reg._inference_model


def call_primitive(name: str, args: List[Any],
                   available_f: float, agency: float) -> PrimitiveCall:
    """Convenience function to call a primitive."""
    return get_registry().call(name, args, available_f, agency)


def list_primitives() -> List[Dict[str, Any]]:
    """List all available primitives."""
    return get_registry().list_primitives()
