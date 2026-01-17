"""
DET Task Management System
==========================

Task decomposition, tracking, and checkpoint/resume capabilities.
Integrates with DET core for agency-aware task approval.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime


class TaskStatus(IntEnum):
    """Task execution status."""
    PENDING = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    FAILED = 3
    BLOCKED = 4
    CANCELLED = 5


class TaskPriority(IntEnum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class TaskStep:
    """A single step within a task."""
    step_id: str
    description: str
    command: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "command": self.command,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskStep":
        return cls(
            step_id=data["step_id"],
            description=data["description"],
            command=data.get("command"),
            status=TaskStatus(data.get("status", 0)),
            result=data.get("result"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class Task:
    """A task with steps and metadata."""
    task_id: str
    title: str
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    parent_id: Optional[str] = None  # For subtasks
    tags: List[str] = field(default_factory=list)

    # DET state at creation
    affect_at_creation: Dict[str, float] = field(default_factory=dict)

    # Checkpoint data
    checkpoint: Optional[Dict[str, Any]] = None

    @property
    def current_step_index(self) -> int:
        """Get index of current (in-progress or next pending) step."""
        for i, step in enumerate(self.steps):
            if step.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                return i
        return len(self.steps)

    @property
    def progress(self) -> float:
        """Get completion progress (0.0 to 1.0)."""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == TaskStatus.COMPLETED)
        return completed / len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "parent_id": self.parent_id,
            "tags": self.tags,
            "affect_at_creation": self.affect_at_creation,
            "checkpoint": self.checkpoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        return cls(
            task_id=data["task_id"],
            title=data["title"],
            description=data["description"],
            steps=[TaskStep.from_dict(s) for s in data.get("steps", [])],
            status=TaskStatus(data.get("status", 0)),
            priority=TaskPriority(data.get("priority", 1)),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
            affect_at_creation=data.get("affect_at_creation", {}),
            checkpoint=data.get("checkpoint"),
        )


class TaskDecomposer:
    """
    Decomposes complex tasks into steps using LLM.

    Integrates with DET core for approval of decomposition.
    """

    DECOMPOSITION_PROMPT = """Break down this task into concrete, executable steps.

Task: {task}

For each step, provide:
1. A clear description of what to do
2. The specific command to run (if applicable)

Format each step as:
STEP: <description>
COMMAND: <command or "none">

Be specific and actionable. Each step should be independently executable.
"""

    def __init__(self, llm_client=None, core=None):
        """
        Initialize the decomposer.

        Args:
            llm_client: OllamaClient for LLM calls.
            core: DETCore for approval.
        """
        self.llm_client = llm_client
        self.core = core

    def decompose(self, task_description: str) -> List[TaskStep]:
        """
        Decompose a task into steps.

        Args:
            task_description: The task to decompose.

        Returns:
            List of TaskStep objects.
        """
        if not self.llm_client:
            # Without LLM, create a single step
            return [TaskStep(
                step_id=str(uuid.uuid4())[:8],
                description=task_description,
            )]

        # Use LLM to decompose
        prompt = self.DECOMPOSITION_PROMPT.format(task=task_description)

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system="You are a task planning assistant. Break tasks into clear, executable steps.",
                temperature=0.3,
                max_tokens=1024,
            )

            text = response.get("response", "")
            return self._parse_steps(text)

        except Exception:
            # Fallback to single step
            return [TaskStep(
                step_id=str(uuid.uuid4())[:8],
                description=task_description,
            )]

    def _parse_steps(self, text: str) -> List[TaskStep]:
        """Parse LLM response into steps."""
        steps = []
        current_description = None
        current_command = None

        for line in text.split('\n'):
            line = line.strip()

            if line.upper().startswith('STEP:'):
                # Save previous step
                if current_description:
                    steps.append(TaskStep(
                        step_id=str(uuid.uuid4())[:8],
                        description=current_description,
                        command=current_command if current_command != "none" else None,
                    ))

                current_description = line[5:].strip()
                current_command = None

            elif line.upper().startswith('COMMAND:'):
                current_command = line[8:].strip()

        # Save last step
        if current_description:
            steps.append(TaskStep(
                step_id=str(uuid.uuid4())[:8],
                description=current_description,
                command=current_command if current_command != "none" else None,
            ))

        return steps if steps else [TaskStep(
            step_id=str(uuid.uuid4())[:8],
            description="Execute task",
        )]


class TaskManager:
    """
    Manages tasks with persistence and execution.

    Provides:
    - Task creation and decomposition
    - Progress tracking
    - Checkpoint/resume
    - DET integration for approval
    """

    def __init__(
        self,
        core=None,
        sandbox=None,
        storage_path: Optional[Path] = None,
        llm_client=None
    ):
        """
        Initialize the task manager.

        Args:
            core: DETCore for agency checks.
            sandbox: BashSandbox for command execution.
            storage_path: Path for task persistence.
            llm_client: OllamaClient for decomposition.
        """
        self.core = core
        self.sandbox = sandbox
        self.storage_path = storage_path or Path.home() / ".det_agency" / "tasks"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.decomposer = TaskDecomposer(llm_client, core)
        self.tasks: Dict[str, Task] = {}

        # Callbacks
        self.on_step_complete: Optional[Callable[[Task, TaskStep], None]] = None
        self.on_task_complete: Optional[Callable[[Task], None]] = None

        self._load_tasks()

    def _load_tasks(self):
        """Load persisted tasks."""
        tasks_file = self.storage_path / "tasks.json"
        if tasks_file.exists():
            try:
                with open(tasks_file, 'r') as f:
                    data = json.load(f)
                    for task_data in data.get("tasks", []):
                        task = Task.from_dict(task_data)
                        self.tasks[task.task_id] = task
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_tasks(self):
        """Persist tasks to disk."""
        tasks_file = self.storage_path / "tasks.json"
        data = {"tasks": [t.to_dict() for t in self.tasks.values()]}
        with open(tasks_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_task(
        self,
        title: str,
        description: str,
        auto_decompose: bool = True,
        priority: TaskPriority = TaskPriority.NORMAL,
        tags: Optional[List[str]] = None
    ) -> Task:
        """
        Create a new task.

        Args:
            title: Task title.
            description: Task description.
            auto_decompose: Whether to decompose into steps.
            priority: Task priority.
            tags: Optional tags.

        Returns:
            Created Task.
        """
        task_id = str(uuid.uuid4())[:8]

        # Get DET state
        affect = {}
        if self.core:
            v, a, b = self.core.get_self_affect()
            affect = {"valence": v, "arousal": a, "bondedness": b}

        # Decompose if requested
        steps = []
        if auto_decompose:
            steps = self.decomposer.decompose(description)

        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            steps=steps,
            priority=priority,
            tags=tags or [],
            affect_at_creation=affect,
        )

        self.tasks[task_id] = task
        self._save_tasks()

        return task

    def add_step(self, task_id: str, description: str, command: Optional[str] = None) -> Optional[TaskStep]:
        """Add a step to an existing task."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        step = TaskStep(
            step_id=str(uuid.uuid4())[:8],
            description=description,
            command=command,
        )

        task.steps.append(step)
        self._save_tasks()

        return step

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None
    ) -> List[Task]:
        """List tasks with optional filtering."""
        tasks = list(self.tasks.values())

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        if priority is not None:
            tasks = [t for t in tasks if t.priority == priority]

        # Sort by priority (descending) and creation time
        tasks.sort(key=lambda t: (-t.priority, t.created_at))

        return tasks

    def execute_step(self, task_id: str, step_index: Optional[int] = None) -> Optional[TaskStep]:
        """
        Execute a task step.

        Args:
            task_id: Task ID.
            step_index: Step index (uses current if None).

        Returns:
            Executed TaskStep or None.
        """
        task = self.tasks.get(task_id)
        if not task:
            return None

        # Get step
        if step_index is None:
            step_index = task.current_step_index

        if step_index >= len(task.steps):
            return None

        step = task.steps[step_index]

        # Update task status
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = time.time()

        # Update step status
        step.status = TaskStatus.IN_PROGRESS
        step.started_at = time.time()

        # Execute command if present
        if step.command and self.sandbox:
            result = self.sandbox.execute(step.command)

            if result.success:
                step.status = TaskStatus.COMPLETED
                step.result = result.stdout
            else:
                step.status = TaskStatus.FAILED
                step.error = result.stderr
        else:
            # Manual step - mark as completed
            step.status = TaskStatus.COMPLETED
            step.result = "Manual step completed"

        step.completed_at = time.time()

        # Check if task is complete
        if all(s.status == TaskStatus.COMPLETED for s in task.steps):
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            if self.on_task_complete:
                self.on_task_complete(task)
        elif any(s.status == TaskStatus.FAILED for s in task.steps):
            task.status = TaskStatus.FAILED

        # Callback
        if self.on_step_complete:
            self.on_step_complete(task, step)

        self._save_tasks()
        return step

    def execute_task(self, task_id: str, stop_on_failure: bool = True) -> Task:
        """
        Execute all steps of a task.

        Args:
            task_id: Task ID.
            stop_on_failure: Whether to stop on first failure.

        Returns:
            Updated Task.
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        for i in range(len(task.steps)):
            step = self.execute_step(task_id, i)

            if step and step.status == TaskStatus.FAILED and stop_on_failure:
                break

        return task

    def checkpoint(self, task_id: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a checkpoint for a task.

        Args:
            task_id: Task ID.
            data: Additional checkpoint data.

        Returns:
            Success status.
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.checkpoint = {
            "timestamp": time.time(),
            "step_index": task.current_step_index,
            "data": data or {},
        }

        self._save_tasks()
        return True

    def resume(self, task_id: str) -> Optional[Task]:
        """
        Resume a task from checkpoint.

        Args:
            task_id: Task ID.

        Returns:
            Resumed Task or None.
        """
        task = self.tasks.get(task_id)
        if not task or not task.checkpoint:
            return None

        # Reset status of failed/blocked steps after checkpoint
        checkpoint_index = task.checkpoint.get("step_index", 0)
        for i, step in enumerate(task.steps):
            if i >= checkpoint_index and step.status in (TaskStatus.FAILED, TaskStatus.BLOCKED):
                step.status = TaskStatus.PENDING
                step.error = None

        task.status = TaskStatus.IN_PROGRESS

        self._save_tasks()
        return task

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.CANCELLED

        # Cancel pending steps
        for step in task.steps:
            if step.status == TaskStatus.PENDING:
                step.status = TaskStatus.CANCELLED

        self._save_tasks()
        return True

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save_tasks()
            return True
        return False

    def get_summary(self) -> Dict[str, Any]:
        """Get task manager summary."""
        tasks = list(self.tasks.values())

        return {
            "total_tasks": len(tasks),
            "by_status": {
                "pending": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
                "in_progress": sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS),
                "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
                "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
                "cancelled": sum(1 for t in tasks if t.status == TaskStatus.CANCELLED),
            },
            "total_steps": sum(len(t.steps) for t in tasks),
            "completed_steps": sum(
                sum(1 for s in t.steps if s.status == TaskStatus.COMPLETED)
                for t in tasks
            ),
        }
