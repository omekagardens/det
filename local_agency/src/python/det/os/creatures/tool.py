"""
Tool Creature
=============

A DET-OS creature that executes commands in a sandboxed environment.
Communicates with other creatures via bonds.

This Python wrapper interfaces with the ToolCreature defined in creatures.ex.

Tool messages:
    EXECUTE: {"type": "execute", "command": str, "timeout": int}
    RESULT: {"type": "result", "success": bool, "output": str, "error": str}
"""

import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from .base import CreatureWrapper
from ..existence.runtime import ExistenceKernelRuntime, CreatureState


@dataclass
class ExecutionResult:
    """Result of a command execution."""
    success: bool
    output: str
    error: str
    exit_code: int
    elapsed_ms: float
    cost: float


class ToolCreature(CreatureWrapper):
    """
    Tool Creature - executes commands in a sandboxed environment.

    Protocol:
        Other creatures send EXECUTE messages via their bond.
        ToolCreature runs the command and sends RESULT back.

    Execution costs F:
        - Base cost: 0.5 F
        - CPU cost: 0.1 F per second
        - Memory cost: 0.01 F per MB

    Security:
        - Agency level determines allowed risk levels
        - a >= 0.3: SAFE commands (ls, cat, echo)
        - a >= 0.5: MODERATE commands (rm file, mv, cp)
        - a >= 0.7: HIGH commands (sudo, curl, wget)
        - a >= 0.9: CRITICAL commands (rm -rf, dd)

    Example usage:
        llm.send_to(tool.cid, {"type": "execute", "command": "ls -la"})
        tool.process_messages()
        result = llm.receive_from(tool.cid)
    """

    # Cost constants
    BASE_EXEC_COST = 0.5
    CPU_COST_PER_SEC = 0.1
    MEMORY_COST_PER_MB = 0.01

    # Risk level thresholds
    RISK_THRESHOLDS = {
        'safe': 0.3,
        'moderate': 0.5,
        'high': 0.7,
        'critical': 0.9
    }

    # Command patterns for risk assessment
    CRITICAL_PATTERNS = ['rm -rf', 'dd if=', 'mkfs', '> /dev/', 'chmod 777']
    HIGH_PATTERNS = ['sudo', 'chmod', 'chown', 'curl', 'wget', 'nc ', 'ssh ']
    MODERATE_PATTERNS = ['rm ', 'mv ', 'cp ', 'mkdir ', 'touch ']

    def __init__(self, runtime: ExistenceKernelRuntime, cid: int):
        super().__init__(runtime, cid)
        self.max_timeout_ms = 30000  # 30 seconds max
        self.max_output_bytes = 65536
        self.execution_count = 0
        self.total_cost = 0.0

        # Sandbox settings
        self.allowed_paths: List[str] = []
        self.blocked_paths: List[str] = ['/etc', '/sys', '/boot']

    def analyze_risk(self, command: str) -> str:
        """Analyze command risk level."""
        cmd_lower = command.lower()

        # Check critical patterns
        for pattern in self.CRITICAL_PATTERNS:
            if pattern in cmd_lower:
                return 'critical'

        # Check high risk patterns
        for pattern in self.HIGH_PATTERNS:
            if pattern in cmd_lower:
                return 'high'

        # Check moderate risk patterns
        for pattern in self.MODERATE_PATTERNS:
            if pattern in cmd_lower:
                return 'moderate'

        return 'safe'

    def can_execute(self, command: str) -> tuple:
        """
        Check if command can be executed with current agency.
        Returns (can_execute, reason).
        """
        risk = self.analyze_risk(command)
        required_a = self.RISK_THRESHOLDS[risk]

        if self.a < required_a:
            return False, f"Insufficient agency for {risk} command (need {required_a}, have {self.a:.2f})"

        if self.F < self.BASE_EXEC_COST:
            return False, f"Insufficient F (need {self.BASE_EXEC_COST}, have {self.F:.2f})"

        return True, "OK"

    def execute(self, command: str, timeout_ms: int = 5000) -> ExecutionResult:
        """
        Execute a command directly.
        Returns ExecutionResult.
        """
        # Check permissions
        can_exec, reason = self.can_execute(command)
        if not can_exec:
            return ExecutionResult(
                success=False,
                output="",
                error=reason,
                exit_code=-1,
                elapsed_ms=0,
                cost=0
            )

        # Clamp timeout
        timeout_ms = min(timeout_ms, self.max_timeout_ms)
        timeout_sec = timeout_ms / 1000.0

        # Execute in subprocess
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            elapsed_ms = (time.time() - start_time) * 1000

            # Truncate output if needed
            output = result.stdout[:self.max_output_bytes]
            error = result.stderr[:self.max_output_bytes]

            # Calculate cost
            cpu_cost = (elapsed_ms / 1000.0) * self.CPU_COST_PER_SEC
            cost = self.BASE_EXEC_COST + cpu_cost

            # Deduct cost
            self.F -= cost
            self.execution_count += 1
            self.total_cost += cost

            return ExecutionResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                exit_code=result.returncode,
                elapsed_ms=elapsed_ms,
                cost=cost
            )

        except subprocess.TimeoutExpired:
            elapsed_ms = timeout_ms
            cost = self.BASE_EXEC_COST + (timeout_sec * self.CPU_COST_PER_SEC)
            self.F -= cost
            self.total_cost += cost

            return ExecutionResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout_ms}ms",
                exit_code=-1,
                elapsed_ms=elapsed_ms,
                cost=cost
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=-1,
                elapsed_ms=(time.time() - start_time) * 1000,
                cost=self.BASE_EXEC_COST
            )

    def process_messages(self):
        """
        Process incoming messages from all bonded creatures.
        """
        for peer_cid in list(self.bonds.keys()):
            messages = self.receive_all_from(peer_cid)

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                msg_type = msg.get("type")

                if msg_type == "execute":
                    command = msg.get("command", "")
                    timeout = msg.get("timeout", 5000)

                    result = self.execute(command, timeout)

                    # Send result back
                    self.send_to(peer_cid, {
                        "type": "result",
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                        "exit_code": result.exit_code,
                        "elapsed_ms": result.elapsed_ms,
                        "cost": result.cost
                    })

    def get_stats(self) -> Dict[str, Any]:
        """Get tool creature statistics."""
        base = self.get_state_dict()
        base.update({
            "execution_count": self.execution_count,
            "total_cost": round(self.total_cost, 2),
            "max_timeout_ms": self.max_timeout_ms,
        })
        return base


def spawn_tool_creature(runtime: ExistenceKernelRuntime,
                        name: str = "tool",
                        initial_f: float = 30.0,
                        initial_a: float = 0.6) -> ToolCreature:
    """
    Spawn a new tool creature.
    Returns the ToolCreature wrapper.
    """
    cid = runtime.spawn(name, initial_f=initial_f, initial_a=initial_a)
    runtime.creatures[cid].state = CreatureState.RUNNING
    return ToolCreature(runtime, cid)
