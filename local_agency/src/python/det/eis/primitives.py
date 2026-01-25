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


# Global registry instance
_registry: Optional[PrimitiveRegistry] = None


def get_registry() -> PrimitiveRegistry:
    """Get the global primitive registry."""
    global _registry
    if _registry is None:
        _registry = PrimitiveRegistry()
    return _registry


def call_primitive(name: str, args: List[Any],
                   available_f: float, agency: float) -> PrimitiveCall:
    """Convenience function to call a primitive."""
    return get_registry().call(name, args, available_f, agency)


def list_primitives() -> List[Dict[str, Any]]:
    """List all available primitives."""
    return get_registry().list_primitives()
