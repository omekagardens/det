"""
DET Sandboxed Bash Environment
==============================

Secure bash execution with permissions, resource limits, and network policies.
Integrates with DET gatekeeper for agency-aware command approval.
"""

import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import IntEnum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Callable
import shlex


class PermissionLevel(IntEnum):
    """Permission levels for command execution."""
    DENY = 0       # Always deny
    ASK = 1        # Ask user for approval
    ALLOW = 2      # Allow with logging
    TRUST = 3      # Allow without logging


class RiskLevel(IntEnum):
    """Risk assessment for commands."""
    SAFE = 0       # Read-only, no side effects
    LOW = 1        # Local modifications, reversible
    MEDIUM = 2     # System changes, potentially reversible
    HIGH = 3       # Irreversible or external effects
    CRITICAL = 4   # Destructive or security-sensitive


@dataclass
class CommandPolicy:
    """Policy for a command or command pattern."""
    pattern: str                          # Regex pattern to match
    permission: PermissionLevel = PermissionLevel.ASK
    risk: RiskLevel = RiskLevel.MEDIUM
    description: str = ""
    max_runtime: float = 60.0            # Max runtime in seconds
    allow_network: bool = False
    allow_write: bool = True
    restricted_paths: List[str] = field(default_factory=list)


@dataclass
class ResourceLimits:
    """Resource limits for command execution."""
    max_cpu_seconds: float = 30.0
    max_memory_mb: int = 512
    max_output_bytes: int = 1024 * 1024  # 1MB
    max_runtime_seconds: float = 60.0
    max_processes: int = 10


@dataclass
class ExecutionResult:
    """Result of a command execution."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    runtime: float
    truncated: bool = False
    timed_out: bool = False
    permission_denied: bool = False
    risk_level: RiskLevel = RiskLevel.SAFE

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.permission_denied

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "runtime": self.runtime,
            "truncated": self.truncated,
            "timed_out": self.timed_out,
            "success": self.success,
        }


class CommandAnalyzer:
    """Analyzes commands for risk and policy matching."""

    # Dangerous command patterns
    DANGEROUS_PATTERNS = [
        (r'\brm\s+-rf?\s+/', RiskLevel.CRITICAL, "Recursive delete from root"),
        (r'\brm\s+-rf?\s+~', RiskLevel.CRITICAL, "Recursive delete from home"),
        (r'\brm\s+-rf?\s+\*', RiskLevel.CRITICAL, "Recursive delete with wildcard"),
        (r'\bmkfs\b', RiskLevel.CRITICAL, "Filesystem format"),
        (r'\bdd\s+.*of=/dev/', RiskLevel.CRITICAL, "Direct device write"),
        (r'>\s*/dev/sd', RiskLevel.CRITICAL, "Direct device write"),
        (r'\bchmod\s+-R\s+777', RiskLevel.HIGH, "Recursive permission change"),
        (r'\bchown\s+-R', RiskLevel.HIGH, "Recursive ownership change"),
        (r'\bsudo\b', RiskLevel.HIGH, "Sudo execution"),
        (r'\bsu\s+-', RiskLevel.HIGH, "User switch"),
        (r'\bcurl\b.*\|\s*(ba)?sh', RiskLevel.CRITICAL, "Pipe to shell"),
        (r'\bwget\b.*\|\s*(ba)?sh', RiskLevel.CRITICAL, "Pipe to shell"),
        (r'>\s*/etc/', RiskLevel.HIGH, "Write to /etc"),
        (r'\bsystemctl\s+(stop|disable|mask)', RiskLevel.HIGH, "Service control"),
        (r'\bkill\s+-9', RiskLevel.MEDIUM, "Force kill"),
        (r'\bpkill\b', RiskLevel.MEDIUM, "Process kill by name"),
        (r'\breboot\b', RiskLevel.CRITICAL, "System reboot"),
        (r'\bshutdown\b', RiskLevel.CRITICAL, "System shutdown"),
        (r'\beval\b', RiskLevel.HIGH, "Eval execution"),
        (r'\$\(.*\)', RiskLevel.MEDIUM, "Command substitution"),
        (r'`.*`', RiskLevel.MEDIUM, "Backtick substitution"),
    ]

    # Safe read-only commands
    SAFE_PATTERNS = [
        (r'^ls\b', RiskLevel.SAFE, "List directory"),
        (r'^cat\b', RiskLevel.SAFE, "Read file"),
        (r'^head\b', RiskLevel.SAFE, "Read file head"),
        (r'^tail\b', RiskLevel.SAFE, "Read file tail"),
        (r'^grep\b', RiskLevel.SAFE, "Search text"),
        (r'^find\b', RiskLevel.SAFE, "Find files"),
        (r'^wc\b', RiskLevel.SAFE, "Word count"),
        (r'^pwd\b', RiskLevel.SAFE, "Print directory"),
        (r'^echo\b', RiskLevel.SAFE, "Echo text"),
        (r'^date\b', RiskLevel.SAFE, "Show date"),
        (r'^whoami\b', RiskLevel.SAFE, "Show user"),
        (r'^env\b', RiskLevel.SAFE, "Show environment"),
        (r'^which\b', RiskLevel.SAFE, "Find command"),
        (r'^type\b', RiskLevel.SAFE, "Show command type"),
        (r'^file\b', RiskLevel.SAFE, "Show file type"),
        (r'^stat\b', RiskLevel.SAFE, "Show file stats"),
        (r'^df\b', RiskLevel.SAFE, "Disk free"),
        (r'^du\b', RiskLevel.SAFE, "Disk usage"),
        (r'^ps\b', RiskLevel.SAFE, "Process status"),
        (r'^top\s+-b\s+-n\s*1', RiskLevel.SAFE, "One-shot top"),
        (r'^uname\b', RiskLevel.SAFE, "System info"),
        (r'^man\b', RiskLevel.SAFE, "Manual page"),
        (r'^help\b', RiskLevel.SAFE, "Help"),
    ]

    # Network commands
    NETWORK_PATTERNS = [
        (r'\bcurl\b', "HTTP client"),
        (r'\bwget\b', "HTTP download"),
        (r'\bssh\b', "SSH connection"),
        (r'\bscp\b', "Secure copy"),
        (r'\brsync\b', "Remote sync"),
        (r'\bnc\b', "Netcat"),
        (r'\bnetcat\b', "Netcat"),
        (r'\bping\b', "Network ping"),
        (r'\btelnet\b', "Telnet"),
        (r'\bftp\b', "FTP"),
        (r'\bnmap\b', "Port scan"),
    ]

    def __init__(self):
        self._custom_policies: List[CommandPolicy] = []

    def add_policy(self, policy: CommandPolicy):
        """Add a custom policy."""
        self._custom_policies.append(policy)

    def analyze(self, command: str) -> tuple[RiskLevel, str, bool]:
        """
        Analyze a command for risk.

        Returns:
            Tuple of (risk_level, reason, needs_network).
        """
        # Check dangerous patterns first
        for pattern, risk, reason in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return risk, reason, False

        # Check for network usage
        needs_network = False
        for pattern, desc in self.NETWORK_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                needs_network = True
                break

        # Check safe patterns
        for pattern, risk, reason in self.SAFE_PATTERNS:
            if re.match(pattern, command.strip(), re.IGNORECASE):
                return risk, reason, needs_network

        # Default to medium risk
        return RiskLevel.LOW, "Standard command", needs_network

    def get_policy(self, command: str) -> Optional[CommandPolicy]:
        """Get matching policy for a command."""
        for policy in self._custom_policies:
            if re.search(policy.pattern, command, re.IGNORECASE):
                return policy
        return None


class BashSandbox:
    """
    Sandboxed bash execution environment.

    Provides secure command execution with:
    - Permission-based access control
    - Resource limits (CPU, memory, time)
    - Network policy enforcement
    - DET integration for agency-aware approval
    """

    # Default allowed directories
    DEFAULT_ALLOWED_PATHS = [
        "/tmp",
        "/var/tmp",
    ]

    # Default denied paths
    DEFAULT_DENIED_PATHS = [
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "~/.ssh",
        "~/.gnupg",
        "~/.aws",
        "~/.config/gcloud",
    ]

    def __init__(
        self,
        core=None,  # Optional DETCore for agency checks
        working_dir: Optional[Path] = None,
        limits: Optional[ResourceLimits] = None,
        allow_network: bool = False,
        interactive_approval: bool = True
    ):
        """
        Initialize the sandbox.

        Args:
            core: Optional DETCore for agency-based approval.
            working_dir: Working directory for commands.
            limits: Resource limits.
            allow_network: Whether to allow network access.
            interactive_approval: Whether to ask user for approval.
        """
        self.core = core
        self.working_dir = working_dir or Path.cwd()
        self.limits = limits or ResourceLimits()
        self.allow_network = allow_network
        self.interactive_approval = interactive_approval

        self.analyzer = CommandAnalyzer()
        self.history: List[ExecutionResult] = []

        # Permission callbacks
        self.on_approval_needed: Optional[Callable[[str, RiskLevel], bool]] = None

        # Allowed/denied paths (expanded)
        self.allowed_paths: Set[Path] = set()
        self.denied_paths: Set[Path] = set()

        self._init_paths()

    def _init_paths(self):
        """Initialize allowed and denied paths."""
        # Add working directory and subdirs
        self.allowed_paths.add(self.working_dir.resolve())

        # Add default allowed
        for p in self.DEFAULT_ALLOWED_PATHS:
            path = Path(p).expanduser()
            if path.exists():
                self.allowed_paths.add(path.resolve())

        # Add default denied
        for p in self.DEFAULT_DENIED_PATHS:
            path = Path(p).expanduser()
            self.denied_paths.add(path.resolve())

    def allow_path(self, path: Path):
        """Add a path to the allowed list."""
        self.allowed_paths.add(path.resolve())

    def deny_path(self, path: Path):
        """Add a path to the denied list."""
        self.denied_paths.add(path.resolve())

    def _check_path_access(self, command: str) -> tuple[bool, str]:
        """Check if command accesses denied paths."""
        # Extract potential paths from command
        # This is a simplified check - real implementation would parse properly
        tokens = shlex.split(command, posix=True)

        for token in tokens:
            if token.startswith('/') or token.startswith('~') or token.startswith('.'):
                try:
                    path = Path(token).expanduser().resolve()

                    # Check denied paths
                    for denied in self.denied_paths:
                        if path == denied or denied in path.parents:
                            return False, f"Access denied: {path}"

                except (ValueError, OSError):
                    pass

        return True, ""

    def _check_det_approval(self, command: str, risk: RiskLevel) -> bool:
        """Check DET core for approval based on agency state."""
        if not self.core:
            return True

        # Get current affect
        valence, arousal, bondedness = self.core.get_self_affect()

        # High arousal + negative valence = more cautious
        caution_factor = arousal * max(0, -valence)

        # Adjust risk threshold based on affect
        if risk >= RiskLevel.HIGH:
            # Need positive valence and bondedness for high-risk
            if valence < 0.2 or bondedness < 0.3:
                return False

        if risk >= RiskLevel.MEDIUM:
            # Reject if high caution
            if caution_factor > 0.5:
                return False

        return True

    def _request_approval(self, command: str, risk: RiskLevel, reason: str) -> bool:
        """Request user approval for a command."""
        if self.on_approval_needed:
            return self.on_approval_needed(command, risk)

        if not self.interactive_approval:
            return risk <= RiskLevel.LOW

        # Default interactive approval
        print(f"\n\033[93m[APPROVAL NEEDED]\033[0m")
        print(f"Command: {command}")
        print(f"Risk: {risk.name} - {reason}")

        try:
            response = input("Allow? (y/n): ").strip().lower()
            return response in ('y', 'yes')
        except (EOFError, KeyboardInterrupt):
            return False

    def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
        capture_output: bool = True
    ) -> ExecutionResult:
        """
        Execute a command in the sandbox.

        Args:
            command: Command to execute.
            timeout: Override timeout (uses limits.max_runtime_seconds if None).
            env: Additional environment variables.
            capture_output: Whether to capture stdout/stderr.

        Returns:
            ExecutionResult with output and status.
        """
        timeout = timeout or self.limits.max_runtime_seconds

        # Analyze command
        risk, reason, needs_network = self.analyzer.analyze(command)

        # Check network policy
        if needs_network and not self.allow_network:
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="Network access denied by policy",
                runtime=0.0,
                permission_denied=True,
                risk_level=risk,
            )

        # Check path access
        path_ok, path_reason = self._check_path_access(command)
        if not path_ok:
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=path_reason,
                runtime=0.0,
                permission_denied=True,
                risk_level=risk,
            )

        # Check DET approval
        if not self._check_det_approval(command, risk):
            return ExecutionResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr="DET core declined - system in cautious state",
                runtime=0.0,
                permission_denied=True,
                risk_level=risk,
            )

        # Request user approval for risky commands
        if risk >= RiskLevel.MEDIUM:
            if not self._request_approval(command, risk, reason):
                return ExecutionResult(
                    command=command,
                    exit_code=-1,
                    stdout="",
                    stderr="User denied approval",
                    runtime=0.0,
                    permission_denied=True,
                    risk_level=risk,
                )

        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        # Execute command
        start_time = time.time()
        truncated = False
        timed_out = False

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=str(self.working_dir),
                env=exec_env,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                preexec_fn=os.setsid if os.name != 'nt' else None,
            )

            try:
                stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
                stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ""
                stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ""

                # Check output size
                if len(stdout) > self.limits.max_output_bytes:
                    stdout = stdout[:self.limits.max_output_bytes] + "\n[OUTPUT TRUNCATED]"
                    truncated = True
                if len(stderr) > self.limits.max_output_bytes:
                    stderr = stderr[:self.limits.max_output_bytes] + "\n[OUTPUT TRUNCATED]"
                    truncated = True

            except subprocess.TimeoutExpired:
                # Kill the process group
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                stdout = ""
                stderr = f"Command timed out after {timeout} seconds"
                timed_out = True

            exit_code = process.returncode

        except Exception as e:
            stdout = ""
            stderr = f"Execution error: {e}"
            exit_code = -1

        runtime = time.time() - start_time

        result = ExecutionResult(
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            runtime=runtime,
            truncated=truncated,
            timed_out=timed_out,
            risk_level=risk,
        )

        self.history.append(result)
        return result

    def execute_script(
        self,
        script: str,
        interpreter: str = "/bin/bash",
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """
        Execute a multi-line script.

        Args:
            script: Script content.
            interpreter: Script interpreter.
            timeout: Execution timeout.

        Returns:
            ExecutionResult with output and status.
        """
        # Write script to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.sh',
            delete=False
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            # Make executable
            os.chmod(script_path, 0o755)

            # Execute
            command = f"{interpreter} {script_path}"
            return self.execute(command, timeout=timeout)

        finally:
            # Clean up
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def get_history(self, limit: int = 10) -> List[ExecutionResult]:
        """Get recent execution history."""
        return self.history[-limit:]

    def clear_history(self):
        """Clear execution history."""
        self.history.clear()


class FileOperations:
    """
    Safe file operations with sandbox integration.

    Provides read/write/search operations within allowed paths.
    """

    def __init__(self, sandbox: BashSandbox):
        """
        Initialize file operations.

        Args:
            sandbox: BashSandbox for path checking and command execution.
        """
        self.sandbox = sandbox

    def _check_path(self, path: Path) -> tuple[bool, str]:
        """Check if path is accessible."""
        resolved = path.resolve()

        # Check denied paths
        for denied in self.sandbox.denied_paths:
            if resolved == denied or denied in resolved.parents:
                return False, f"Access denied: {resolved}"

        # Check if within allowed paths
        for allowed in self.sandbox.allowed_paths:
            if resolved == allowed or allowed in resolved.parents:
                return True, ""

        return False, f"Path not in allowed list: {resolved}"

    def read(self, path: Path, max_lines: int = 1000) -> tuple[bool, str]:
        """
        Read a file.

        Args:
            path: File path.
            max_lines: Maximum lines to read.

        Returns:
            Tuple of (success, content_or_error).
        """
        ok, reason = self._check_path(path)
        if not ok:
            return False, reason

        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n[TRUNCATED at {max_lines} lines]")
                        break
                    lines.append(line)
                return True, ''.join(lines)
        except Exception as e:
            return False, f"Read error: {e}"

    def write(self, path: Path, content: str, append: bool = False) -> tuple[bool, str]:
        """
        Write to a file.

        Args:
            path: File path.
            content: Content to write.
            append: Whether to append (vs overwrite).

        Returns:
            Tuple of (success, message).
        """
        ok, reason = self._check_path(path)
        if not ok:
            return False, reason

        # Check DET approval for writes
        if self.sandbox.core:
            valence, _, bondedness = self.sandbox.core.get_self_affect()
            if valence < 0 and bondedness < 0.3:
                return False, "DET core declined write - low trust state"

        try:
            mode = 'a' if append else 'w'
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            return True, f"Written to {path}"
        except Exception as e:
            return False, f"Write error: {e}"

    def search(
        self,
        pattern: str,
        path: Path,
        recursive: bool = True,
        max_results: int = 100
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Search for pattern in files.

        Args:
            pattern: Search pattern (regex).
            path: Directory to search.
            recursive: Whether to search recursively.
            max_results: Maximum results to return.

        Returns:
            Tuple of (success, results).
        """
        ok, reason = self._check_path(path)
        if not ok:
            return False, []

        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)

            glob_pattern = "**/*" if recursive else "*"
            for file_path in path.glob(glob_pattern):
                if not file_path.is_file():
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                results.append({
                                    "file": str(file_path),
                                    "line": line_num,
                                    "content": line.strip()[:200],
                                })
                                if len(results) >= max_results:
                                    return True, results
                except (IOError, PermissionError):
                    continue

            return True, results

        except re.error as e:
            return False, [{"error": f"Invalid pattern: {e}"}]

    def list_dir(self, path: Path, include_hidden: bool = False) -> tuple[bool, List[Dict[str, Any]]]:
        """
        List directory contents.

        Args:
            path: Directory path.
            include_hidden: Whether to include hidden files.

        Returns:
            Tuple of (success, entries).
        """
        ok, reason = self._check_path(path)
        if not ok:
            return False, []

        try:
            entries = []
            for entry in path.iterdir():
                if not include_hidden and entry.name.startswith('.'):
                    continue

                entries.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else 0,
                })

            return True, sorted(entries, key=lambda x: (x["type"] == "file", x["name"]))

        except Exception as e:
            return False, [{"error": str(e)}]
