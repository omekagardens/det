#!/usr/bin/env python3
"""
DET Phase 3 Integration Tests
=============================

Tests for sandbox, tasks, timer, and code execution.
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det import (
    DETCore,
    BashSandbox, FileOperations, CommandAnalyzer, RiskLevel, PermissionLevel,
    TaskManager, Task, TaskStep, TaskStatus, TaskPriority,
    TimerSystem, ScheduledEvent, ScheduleType,
    CodeExecutor, ExecutionSession, LanguageRunner, ErrorInterpreter,
)


# ============================================================================
# Sandbox Tests
# ============================================================================

def test_command_analyzer():
    """Test command risk analysis."""
    print("  test_command_analyzer...", end=" ")

    analyzer = CommandAnalyzer()

    # Safe commands
    risk, reason, network = analyzer.analyze("ls -la")
    assert risk == RiskLevel.SAFE

    risk, reason, network = analyzer.analyze("cat /etc/hosts")
    assert risk == RiskLevel.SAFE

    # Dangerous commands
    risk, reason, network = analyzer.analyze("rm -rf /")
    assert risk == RiskLevel.CRITICAL

    risk, reason, network = analyzer.analyze("sudo apt install foo")
    assert risk == RiskLevel.HIGH

    # Network commands
    risk, reason, network = analyzer.analyze("curl https://example.com")
    assert network == True

    print("PASS")


def test_sandbox_init():
    """Test sandbox initialization."""
    print("  test_sandbox_init...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(
            working_dir=Path(tmpdir),
            allow_network=False,
            interactive_approval=False,
        )

        assert sandbox.working_dir == Path(tmpdir)
        assert not sandbox.allow_network

    print("PASS")


def test_sandbox_execute_safe():
    """Test sandbox execution of safe commands."""
    print("  test_sandbox_execute_safe...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(
            working_dir=Path(tmpdir),
            interactive_approval=False,
        )

        # Simple safe command
        result = sandbox.execute("echo hello")
        assert result.success
        assert "hello" in result.stdout

        # Another safe command
        result = sandbox.execute("pwd")
        assert result.success
        assert tmpdir in result.stdout

    print("PASS")


def test_sandbox_deny_network():
    """Test sandbox network denial."""
    print("  test_sandbox_deny_network...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(
            working_dir=Path(tmpdir),
            allow_network=False,
            interactive_approval=False,
        )

        # Network command should be denied
        result = sandbox.execute("curl https://example.com")
        assert result.permission_denied
        assert "Network access denied" in result.stderr

    print("PASS")


def test_file_operations():
    """Test file operations."""
    print("  test_file_operations...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(
            working_dir=Path(tmpdir),
            interactive_approval=False,
        )
        sandbox.allow_path(Path(tmpdir))

        files = FileOperations(sandbox)

        # Write file
        test_path = Path(tmpdir) / "test.txt"
        ok, msg = files.write(test_path, "Hello, World!")
        assert ok

        # Read file
        ok, content = files.read(test_path)
        assert ok
        assert "Hello, World!" in content

        # List directory
        ok, entries = files.list_dir(Path(tmpdir))
        assert ok
        assert any(e["name"] == "test.txt" for e in entries)

    print("PASS")


# ============================================================================
# Task Management Tests
# ============================================================================

def test_task_creation():
    """Test task creation."""
    print("  test_task_creation...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TaskManager(storage_path=Path(tmpdir))

        task = manager.create_task(
            title="Test Task",
            description="A test task",
            auto_decompose=False,
            priority=TaskPriority.HIGH,
        )

        assert task.title == "Test Task"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING

    print("PASS")


def test_task_steps():
    """Test adding and executing task steps."""
    print("  test_task_steps...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(
            working_dir=Path(tmpdir),
            interactive_approval=False,
        )

        manager = TaskManager(
            sandbox=sandbox,
            storage_path=Path(tmpdir) / "tasks",
        )

        task = manager.create_task(
            title="Multi-step Task",
            description="Task with steps",
            auto_decompose=False,
        )

        # Add steps
        manager.add_step(task.task_id, "Step 1: Echo hello", "echo hello")
        manager.add_step(task.task_id, "Step 2: List files", "ls")

        task = manager.get_task(task.task_id)
        assert len(task.steps) == 2

        # Execute first step
        step = manager.execute_step(task.task_id, 0)
        assert step.status == TaskStatus.COMPLETED

    print("PASS")


def test_task_persistence():
    """Test task persistence."""
    print("  test_task_persistence...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "tasks"

        # Create task in first manager
        manager1 = TaskManager(storage_path=storage)
        task = manager1.create_task(
            title="Persistent Task",
            description="Should persist",
        )
        task_id = task.task_id

        # Load in second manager
        manager2 = TaskManager(storage_path=storage)
        loaded = manager2.get_task(task_id)

        assert loaded is not None
        assert loaded.title == "Persistent Task"

    print("PASS")


# ============================================================================
# Timer System Tests
# ============================================================================

def test_timer_schedule_once():
    """Test one-time scheduling."""
    print("  test_timer_schedule_once...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        timer = TimerSystem(storage_path=Path(tmpdir))

        callback_called = [False]

        def test_callback(event, data):
            callback_called[0] = True

        timer.register_callback("test", test_callback)

        # Schedule for 0.1 seconds from now
        timer.schedule_once(
            name="Test Event",
            callback_name="test",
            run_at=time.time() + 0.1,
        )

        events = timer.list_events()
        assert len(events) == 1
        assert events[0].name == "Test Event"

    print("PASS")


def test_timer_schedule_interval():
    """Test interval scheduling."""
    print("  test_timer_schedule_interval...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        timer = TimerSystem(storage_path=Path(tmpdir))

        timer.schedule_interval(
            name="Recurring Event",
            callback_name="health_check",
            interval_seconds=60.0,
            max_runs=5,
        )

        events = timer.list_events()
        assert len(events) == 1
        assert events[0].schedule_type == ScheduleType.INTERVAL
        assert events[0].max_runs == 5

    print("PASS")


def test_timer_status():
    """Test timer status."""
    print("  test_timer_status...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        timer = TimerSystem(storage_path=Path(tmpdir))

        # Add some events
        timer.schedule_interval("Event 1", "health_check", 60.0)
        timer.schedule_interval("Event 2", "health_check", 120.0)

        status = timer.get_status()
        assert status["total_events"] == 2
        assert status["active"] == 2

    print("PASS")


# ============================================================================
# Code Execution Tests
# ============================================================================

def test_error_interpreter():
    """Test error interpretation."""
    print("  test_error_interpreter...", end=" ")

    interpreter = ErrorInterpreter()

    # Python errors
    results = interpreter.interpret("NameError: name 'foo' is not defined")
    assert len(results) > 0
    assert "foo" in results[0]["suggestion"]

    # Module not found
    results = interpreter.interpret("ImportError: No module named 'numpy'")
    assert len(results) > 0
    assert "numpy" in results[0]["suggestion"]

    print("PASS")


def test_language_runner_config():
    """Test language runner configuration."""
    print("  test_language_runner_config...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(working_dir=Path(tmpdir), interactive_approval=False)
        runner = LanguageRunner(sandbox)

        # Check Python config
        config = runner.get_config("python")
        assert config is not None
        assert config["extension"] == ".py"

        # Check C config
        config = runner.get_config("c")
        assert config is not None
        assert config["compile"] is not None

        # Unknown language
        config = runner.get_config("unknown_lang")
        assert config is None

    print("PASS")


def test_code_executor_session():
    """Test code execution session."""
    print("  test_code_executor_session...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(working_dir=Path(tmpdir), interactive_approval=False)
        executor = CodeExecutor(sandbox=sandbox, max_iterations=3)

        session = executor.create_session(
            language="python",
            task_description="Print hello world",
            initial_code='print("Hello, World!")',
        )

        assert session.language == "python"
        assert session.current_code == 'print("Hello, World!")'
        assert not session.is_complete

    print("PASS")


def test_code_executor_run():
    """Test code execution."""
    print("  test_code_executor_run...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(working_dir=Path(tmpdir), interactive_approval=False)
        sandbox.allow_path(Path(tmpdir))

        executor = CodeExecutor(sandbox=sandbox, max_iterations=3)

        session = executor.create_session(
            language="python",
            task_description="Print hello",
            initial_code='print("Hello from test!")',
        )

        result = executor.execute_session(session.session_id)

        # Check that we got some attempts
        assert len(result.attempts) > 0

        # If Python is available, it should succeed
        from det.executor import ExecutionPhase
        if result.phase == ExecutionPhase.COMPLETE:
            assert any("Hello from test!" in a.stdout for a in result.attempts)

    print("PASS")


# ============================================================================
# Integration Tests
# ============================================================================

def test_sandbox_det_integration():
    """Test sandbox with DET core integration."""
    print("  test_sandbox_det_integration...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = BashSandbox(
                core=core,
                working_dir=Path(tmpdir),
                interactive_approval=False,
            )

            # Run a safe command - should work
            result = sandbox.execute("echo test")
            assert result.success

    print("PASS")


def test_task_with_sandbox():
    """Test task execution with sandbox."""
    print("  test_task_with_sandbox...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = BashSandbox(working_dir=Path(tmpdir), interactive_approval=False)

        manager = TaskManager(
            sandbox=sandbox,
            storage_path=Path(tmpdir) / "tasks",
        )

        task = manager.create_task(
            title="Shell Task",
            description="Run shell commands",
            auto_decompose=False,
        )

        manager.add_step(task.task_id, "Echo", "echo 'task output'")

        result = manager.execute_task(task.task_id)
        assert result.status == TaskStatus.COMPLETED

    print("PASS")


# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run all Phase 3 tests."""
    print("\n========================================")
    print("  DET Phase 3 Test Suite")
    print("========================================\n")

    tests = [
        # Sandbox
        test_command_analyzer,
        test_sandbox_init,
        test_sandbox_execute_safe,
        test_sandbox_deny_network,
        test_file_operations,

        # Tasks
        test_task_creation,
        test_task_steps,
        test_task_persistence,

        # Timer
        test_timer_schedule_once,
        test_timer_schedule_interval,
        test_timer_status,

        # Executor
        test_error_interpreter,
        test_language_runner_config,
        test_code_executor_session,
        test_code_executor_run,

        # Integration
        test_sandbox_det_integration,
        test_task_with_sandbox,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n========================================")
    print(f"  Results: {passed}/{passed + failed} tests passed")
    print("========================================\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
