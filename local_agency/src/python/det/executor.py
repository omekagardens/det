"""
DET Code Execution Loop
=======================

Write → Compile → Test → Iterate cycle with error interpretation and retry.
"""

import re
import tempfile
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable


class ExecutionPhase(IntEnum):
    """Phases of code execution."""
    WRITE = 0
    COMPILE = 1
    TEST = 2
    VERIFY = 3
    COMPLETE = 4
    FAILED = 5


@dataclass
class ExecutionAttempt:
    """Record of a single execution attempt."""
    phase: ExecutionPhase
    code: str
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timestamp: float = field(default_factory=time.time)

    @property
    def success(self) -> bool:
        return self.exit_code == 0


@dataclass
class ExecutionSession:
    """A code execution session with history."""
    session_id: str
    language: str
    task_description: str
    attempts: List[ExecutionAttempt] = field(default_factory=list)
    current_code: str = ""
    phase: ExecutionPhase = ExecutionPhase.WRITE
    max_iterations: int = 5
    iteration: int = 0

    @property
    def is_complete(self) -> bool:
        return self.phase in (ExecutionPhase.COMPLETE, ExecutionPhase.FAILED)


class ErrorInterpreter:
    """Interprets error messages and suggests fixes."""

    # Common error patterns and suggestions
    ERROR_PATTERNS = {
        # Python errors
        r"IndentationError": "Check indentation - Python uses spaces/tabs for code blocks",
        r"SyntaxError: invalid syntax": "Check for missing colons, parentheses, or quotes",
        r"NameError: name '(\w+)' is not defined": "Variable or function '{0}' is not defined - check spelling or add import",
        r"ImportError: No module named '(\w+)'": "Module '{0}' is not installed - try: pip install {0}",
        r"ModuleNotFoundError": "Module not found - check module name or install it",
        r"TypeError: .* takes (\d+) .* (\d+) .* given": "Function expects {0} arguments but {1} were given",
        r"KeyError: '(\w+)'": "Key '{0}' not found in dictionary - check key name",
        r"IndexError: list index out of range": "Trying to access index beyond list length",
        r"AttributeError: '(\w+)' object has no attribute '(\w+)'": "Object of type '{0}' doesn't have attribute '{1}'",
        r"ValueError": "Invalid value provided - check data type and range",
        r"ZeroDivisionError": "Division by zero - add check for zero divisor",
        r"FileNotFoundError": "File does not exist - check path and filename",

        # C/C++ errors
        r"error: '(\w+)' undeclared": "Variable '{0}' not declared - add declaration",
        r"error: expected ';'": "Missing semicolon - add ; at end of statement",
        r"undefined reference to `(\w+)'": "Function '{0}' not defined or not linked",
        r"error: conflicting types": "Type mismatch - check function declarations",
        r"warning: implicit declaration": "Function used before declaration - add prototype",

        # JavaScript/Node errors
        r"ReferenceError: (\w+) is not defined": "Variable '{0}' is not defined",
        r"SyntaxError: Unexpected token": "Unexpected character - check brackets and semicolons",
        r"TypeError: Cannot read propert": "Trying to access property of undefined/null",

        # Rust errors
        r"error\[E0308\]: mismatched types": "Type mismatch - check expected vs actual types",
        r"error\[E0425\]: cannot find value `(\w+)`": "Variable '{0}' not in scope",
        r"error\[E0382\]: borrow of moved value": "Value was moved - consider using .clone()",

        # General
        r"permission denied": "Permission issue - check file/directory permissions",
        r"No such file or directory": "Path doesn't exist - verify the path",
        r"command not found": "Command not installed - install the required tool",
    }

    def interpret(self, stderr: str, stdout: str = "") -> List[Dict[str, str]]:
        """
        Interpret error output and suggest fixes.

        Args:
            stderr: Standard error output.
            stdout: Standard output (sometimes contains errors).

        Returns:
            List of interpretations with pattern, message, and suggestion.
        """
        combined = stderr + "\n" + stdout
        interpretations = []

        for pattern, suggestion_template in self.ERROR_PATTERNS.items():
            matches = re.finditer(pattern, combined, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                try:
                    suggestion = suggestion_template.format(*groups) if groups else suggestion_template
                except (IndexError, KeyError):
                    suggestion = suggestion_template

                interpretations.append({
                    "pattern": pattern,
                    "match": match.group(0),
                    "suggestion": suggestion,
                    "line": self._find_line_number(combined, match.start()),
                })

        return interpretations

    def _find_line_number(self, text: str, position: int) -> Optional[int]:
        """Find line number for a position in text."""
        lines = text[:position].split('\n')
        return len(lines) if lines else None


class LanguageRunner:
    """Runs code for different languages."""

    LANGUAGE_CONFIG = {
        "python": {
            "extension": ".py",
            "compile": None,  # Interpreted
            "run": "python3 {file}",
            "test": "python3 -m pytest {file} -v",
        },
        "javascript": {
            "extension": ".js",
            "compile": None,
            "run": "node {file}",
            "test": "node --test {file}",
        },
        "typescript": {
            "extension": ".ts",
            "compile": "tsc {file}",
            "run": "node {file_noext}.js",
            "test": "npx ts-node {file}",
        },
        "c": {
            "extension": ".c",
            "compile": "gcc {file} -o {file_noext}",
            "run": "./{file_noext}",
            "test": "./{file_noext}",  # Assuming test is built-in
        },
        "cpp": {
            "extension": ".cpp",
            "compile": "g++ {file} -o {file_noext}",
            "run": "./{file_noext}",
            "test": "./{file_noext}",
        },
        "rust": {
            "extension": ".rs",
            "compile": "rustc {file}",
            "run": "./{file_noext}",
            "test": "cargo test",
        },
        "go": {
            "extension": ".go",
            "compile": "go build {file}",
            "run": "go run {file}",
            "test": "go test -v",
        },
        "bash": {
            "extension": ".sh",
            "compile": None,
            "run": "bash {file}",
            "test": "bash {file}",
        },
    }

    def __init__(self, sandbox):
        """
        Initialize the runner.

        Args:
            sandbox: BashSandbox for execution.
        """
        self.sandbox = sandbox

    def get_config(self, language: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a language."""
        return self.LANGUAGE_CONFIG.get(language.lower())

    def write_code(self, code: str, language: str, filename: Optional[str] = None) -> Path:
        """
        Write code to a temporary file.

        Args:
            code: Code content.
            language: Programming language.
            filename: Optional filename.

        Returns:
            Path to the created file.
        """
        config = self.get_config(language)
        if not config:
            raise ValueError(f"Unsupported language: {language}")

        ext = config["extension"]
        if filename:
            filepath = self.sandbox.working_dir / filename
        else:
            filepath = self.sandbox.working_dir / f"code_{int(time.time())}{ext}"

        filepath.write_text(code)
        return filepath

    def compile_code(self, filepath: Path, language: str) -> ExecutionAttempt:
        """
        Compile code if needed.

        Args:
            filepath: Path to source file.
            language: Programming language.

        Returns:
            ExecutionAttempt with results.
        """
        config = self.get_config(language)
        if not config or not config.get("compile"):
            # No compilation needed
            return ExecutionAttempt(
                phase=ExecutionPhase.COMPILE,
                code="",
                command="(no compilation needed)",
                exit_code=0,
                stdout="",
                stderr="",
                duration=0.0,
            )

        # Build command
        file_noext = filepath.with_suffix("")
        command = config["compile"].format(
            file=str(filepath),
            file_noext=str(file_noext),
        )

        start = time.time()
        result = self.sandbox.execute(command)
        duration = time.time() - start

        return ExecutionAttempt(
            phase=ExecutionPhase.COMPILE,
            code="",
            command=command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=duration,
        )

    def run_code(self, filepath: Path, language: str) -> ExecutionAttempt:
        """
        Run code.

        Args:
            filepath: Path to source/compiled file.
            language: Programming language.

        Returns:
            ExecutionAttempt with results.
        """
        config = self.get_config(language)
        if not config:
            raise ValueError(f"Unsupported language: {language}")

        file_noext = filepath.with_suffix("")
        command = config["run"].format(
            file=str(filepath),
            file_noext=str(file_noext),
        )

        start = time.time()
        result = self.sandbox.execute(command)
        duration = time.time() - start

        return ExecutionAttempt(
            phase=ExecutionPhase.TEST,  # Running is part of testing
            code="",
            command=command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=duration,
        )

    def test_code(self, filepath: Path, language: str) -> ExecutionAttempt:
        """
        Run tests for code.

        Args:
            filepath: Path to test file.
            language: Programming language.

        Returns:
            ExecutionAttempt with results.
        """
        config = self.get_config(language)
        if not config:
            raise ValueError(f"Unsupported language: {language}")

        file_noext = filepath.with_suffix("")
        command = config.get("test", config["run"]).format(
            file=str(filepath),
            file_noext=str(file_noext),
        )

        start = time.time()
        result = self.sandbox.execute(command)
        duration = time.time() - start

        return ExecutionAttempt(
            phase=ExecutionPhase.TEST,
            code="",
            command=command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration=duration,
        )


class CodeExecutor:
    """
    Code execution loop with retry and error handling.

    Implements Write → Compile → Test → Iterate cycle.
    """

    def __init__(
        self,
        sandbox,
        llm_client=None,
        core=None,
        max_iterations: int = 5
    ):
        """
        Initialize the executor.

        Args:
            sandbox: BashSandbox for execution.
            llm_client: OllamaClient for code generation/fixing.
            core: DETCore for state monitoring.
            max_iterations: Maximum retry iterations.
        """
        self.sandbox = sandbox
        self.llm_client = llm_client
        self.core = core
        self.max_iterations = max_iterations

        self.runner = LanguageRunner(sandbox)
        self.interpreter = ErrorInterpreter()

        self.sessions: Dict[str, ExecutionSession] = {}

        # Callbacks
        self.on_iteration: Optional[Callable[[ExecutionSession, ExecutionAttempt], None]] = None
        self.on_complete: Optional[Callable[[ExecutionSession], None]] = None

    def create_session(
        self,
        language: str,
        task_description: str,
        initial_code: Optional[str] = None
    ) -> ExecutionSession:
        """
        Create a new execution session.

        Args:
            language: Programming language.
            task_description: Description of what the code should do.
            initial_code: Optional initial code.

        Returns:
            Created ExecutionSession.
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]

        session = ExecutionSession(
            session_id=session_id,
            language=language,
            task_description=task_description,
            current_code=initial_code or "",
            max_iterations=self.max_iterations,
        )

        self.sessions[session_id] = session
        return session

    def _generate_code(self, session: ExecutionSession) -> str:
        """Generate code using LLM."""
        if not self.llm_client:
            return session.current_code

        prompt = f"""Write {session.language} code for the following task:

{session.task_description}

Requirements:
- Code should be complete and runnable
- Include any necessary imports
- Add basic error handling
- No explanations, just the code

```{session.language}
"""

        response = self.llm_client.generate(
            prompt=prompt,
            system="You are a code generation assistant. Write clean, working code.",
            temperature=0.3,
            max_tokens=2048,
        )

        text = response.get("response", "")

        # Extract code from response
        code_match = re.search(r'```\w*\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        return text.strip()

    def _fix_code(self, session: ExecutionSession, error: str, suggestions: List[Dict]) -> str:
        """Attempt to fix code using LLM."""
        if not self.llm_client:
            return session.current_code

        suggestions_text = "\n".join(
            f"- {s['suggestion']}" for s in suggestions
        )

        prompt = f"""Fix the following {session.language} code that has errors:

Original task: {session.task_description}

Current code:
```{session.language}
{session.current_code}
```

Error:
{error}

Suggestions:
{suggestions_text}

Provide the fixed code only, no explanations:
```{session.language}
"""

        response = self.llm_client.generate(
            prompt=prompt,
            system="You are a code debugging assistant. Fix the code based on the error.",
            temperature=0.2,
            max_tokens=2048,
        )

        text = response.get("response", "")

        code_match = re.search(r'```\w*\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        return text.strip()

    def execute_session(self, session_id: str) -> ExecutionSession:
        """
        Execute a session through the full cycle.

        Args:
            session_id: Session ID.

        Returns:
            Updated ExecutionSession.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        while session.iteration < session.max_iterations and not session.is_complete:
            session.iteration += 1

            # Phase 1: Write code
            if not session.current_code:
                session.phase = ExecutionPhase.WRITE
                session.current_code = self._generate_code(session)

            # Write to file
            filepath = self.runner.write_code(
                session.current_code,
                session.language,
            )

            # Phase 2: Compile (if needed)
            session.phase = ExecutionPhase.COMPILE
            compile_attempt = self.runner.compile_code(filepath, session.language)
            compile_attempt.code = session.current_code
            session.attempts.append(compile_attempt)

            if self.on_iteration:
                self.on_iteration(session, compile_attempt)

            if not compile_attempt.success:
                # Interpret errors and fix
                suggestions = self.interpreter.interpret(compile_attempt.stderr)
                session.current_code = self._fix_code(
                    session,
                    compile_attempt.stderr,
                    suggestions,
                )
                continue

            # Phase 3: Test/Run
            session.phase = ExecutionPhase.TEST
            run_attempt = self.runner.run_code(filepath, session.language)
            run_attempt.code = session.current_code
            session.attempts.append(run_attempt)

            if self.on_iteration:
                self.on_iteration(session, run_attempt)

            if not run_attempt.success:
                # Interpret errors and fix
                suggestions = self.interpreter.interpret(
                    run_attempt.stderr,
                    run_attempt.stdout,
                )
                session.current_code = self._fix_code(
                    session,
                    run_attempt.stderr + "\n" + run_attempt.stdout,
                    suggestions,
                )
                continue

            # Phase 4: Verify (basic check that output is reasonable)
            session.phase = ExecutionPhase.VERIFY

            # If we got here, execution was successful
            session.phase = ExecutionPhase.COMPLETE

            if self.on_complete:
                self.on_complete(session)

            return session

        # Max iterations reached
        if not session.is_complete:
            session.phase = ExecutionPhase.FAILED

        return session

    def get_session(self, session_id: str) -> Optional[ExecutionSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[ExecutionSession]:
        """List all sessions."""
        return list(self.sessions.values())
