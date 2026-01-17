"""
DET Timer System
================

Scheduled task execution, sleep/consolidation cycles, and health monitoring.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import uuid


class ScheduleType(IntEnum):
    """Types of scheduled events."""
    ONCE = 0       # Run once at specified time
    INTERVAL = 1   # Run at regular intervals
    DAILY = 2      # Run daily at specified time
    CRON = 3       # Cron-style scheduling (simplified)


class EventStatus(IntEnum):
    """Scheduled event status."""
    ACTIVE = 0
    PAUSED = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4


@dataclass
class ScheduledEvent:
    """A scheduled event."""
    event_id: str
    name: str
    schedule_type: ScheduleType
    callback_name: str  # Name of registered callback
    data: Dict[str, Any] = field(default_factory=dict)

    # Timing
    interval_seconds: float = 0.0  # For INTERVAL type
    run_at_hour: int = 0           # For DAILY type (0-23)
    run_at_minute: int = 0         # For DAILY type (0-59)
    next_run: float = 0.0          # Unix timestamp

    # Status
    status: EventStatus = EventStatus.ACTIVE
    last_run: Optional[float] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

    # Limits
    max_runs: Optional[int] = None  # None = unlimited

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "schedule_type": self.schedule_type.value,
            "callback_name": self.callback_name,
            "data": self.data,
            "interval_seconds": self.interval_seconds,
            "run_at_hour": self.run_at_hour,
            "run_at_minute": self.run_at_minute,
            "next_run": self.next_run,
            "status": self.status.value,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "max_runs": self.max_runs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledEvent":
        return cls(
            event_id=data["event_id"],
            name=data["name"],
            schedule_type=ScheduleType(data["schedule_type"]),
            callback_name=data["callback_name"],
            data=data.get("data", {}),
            interval_seconds=data.get("interval_seconds", 0.0),
            run_at_hour=data.get("run_at_hour", 0),
            run_at_minute=data.get("run_at_minute", 0),
            next_run=data.get("next_run", 0.0),
            status=EventStatus(data.get("status", 0)),
            last_run=data.get("last_run"),
            run_count=data.get("run_count", 0),
            error_count=data.get("error_count", 0),
            last_error=data.get("last_error"),
            max_runs=data.get("max_runs"),
        )


class TimerSystem:
    """
    Timer system for scheduled execution.

    Provides:
    - One-time and recurring scheduled events
    - Sleep/consolidation cycles
    - Health monitoring
    - DET integration
    """

    # Default callbacks
    BUILTIN_CALLBACKS = {
        "health_check": "_health_check",
        "memory_consolidation": "_memory_consolidation",
        "det_maintenance": "_det_maintenance",
    }

    def __init__(
        self,
        core=None,
        memory_manager=None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize the timer system.

        Args:
            core: DETCore for state monitoring.
            memory_manager: MemoryManager for consolidation.
            storage_path: Path for persistence.
        """
        self.core = core
        self.memory_manager = memory_manager
        self.storage_path = storage_path or Path.home() / ".det_agency" / "timer"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.events: Dict[str, ScheduledEvent] = {}
        self.callbacks: Dict[str, Callable] = {}

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Register built-in callbacks
        self._register_builtins()

        # Load persisted events
        self._load_events()

    def _register_builtins(self):
        """Register built-in callbacks."""
        self.register_callback("health_check", self._health_check)
        self.register_callback("memory_consolidation", self._memory_consolidation)
        self.register_callback("det_maintenance", self._det_maintenance)

    def _load_events(self):
        """Load persisted events."""
        events_file = self.storage_path / "events.json"
        if events_file.exists():
            try:
                with open(events_file, 'r') as f:
                    data = json.load(f)
                    for event_data in data.get("events", []):
                        event = ScheduledEvent.from_dict(event_data)
                        self.events[event.event_id] = event
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_events(self):
        """Persist events to disk."""
        events_file = self.storage_path / "events.json"
        data = {"events": [e.to_dict() for e in self.events.values()]}
        with open(events_file, 'w') as f:
            json.dump(data, f, indent=2)

    def register_callback(self, name: str, callback: Callable):
        """
        Register a callback function.

        Args:
            name: Callback name.
            callback: Callable that takes (event, data) arguments.
        """
        self.callbacks[name] = callback

    def _calculate_next_run(self, event: ScheduledEvent) -> float:
        """Calculate next run time for an event."""
        now = time.time()

        if event.schedule_type == ScheduleType.ONCE:
            # Already set, don't change
            return event.next_run

        elif event.schedule_type == ScheduleType.INTERVAL:
            if event.last_run:
                return event.last_run + event.interval_seconds
            return now + event.interval_seconds

        elif event.schedule_type == ScheduleType.DAILY:
            # Calculate next occurrence of HH:MM
            dt = datetime.now()
            target = dt.replace(
                hour=event.run_at_hour,
                minute=event.run_at_minute,
                second=0,
                microsecond=0
            )

            if target.timestamp() <= now:
                # Already passed today, schedule for tomorrow
                target += timedelta(days=1)

            return target.timestamp()

        return now

    def schedule_once(
        self,
        name: str,
        callback_name: str,
        run_at: float,
        data: Optional[Dict[str, Any]] = None
    ) -> ScheduledEvent:
        """
        Schedule a one-time event.

        Args:
            name: Event name.
            callback_name: Name of registered callback.
            run_at: Unix timestamp to run at.
            data: Data to pass to callback.

        Returns:
            Created ScheduledEvent.
        """
        event = ScheduledEvent(
            event_id=str(uuid.uuid4())[:8],
            name=name,
            schedule_type=ScheduleType.ONCE,
            callback_name=callback_name,
            data=data or {},
            next_run=run_at,
            max_runs=1,
        )

        with self._lock:
            self.events[event.event_id] = event
            self._save_events()

        return event

    def schedule_interval(
        self,
        name: str,
        callback_name: str,
        interval_seconds: float,
        data: Optional[Dict[str, Any]] = None,
        max_runs: Optional[int] = None
    ) -> ScheduledEvent:
        """
        Schedule a recurring interval event.

        Args:
            name: Event name.
            callback_name: Name of registered callback.
            interval_seconds: Interval between runs.
            data: Data to pass to callback.
            max_runs: Maximum number of runs (None = unlimited).

        Returns:
            Created ScheduledEvent.
        """
        event = ScheduledEvent(
            event_id=str(uuid.uuid4())[:8],
            name=name,
            schedule_type=ScheduleType.INTERVAL,
            callback_name=callback_name,
            data=data or {},
            interval_seconds=interval_seconds,
            next_run=time.time() + interval_seconds,
            max_runs=max_runs,
        )

        with self._lock:
            self.events[event.event_id] = event
            self._save_events()

        return event

    def schedule_daily(
        self,
        name: str,
        callback_name: str,
        hour: int,
        minute: int = 0,
        data: Optional[Dict[str, Any]] = None
    ) -> ScheduledEvent:
        """
        Schedule a daily event.

        Args:
            name: Event name.
            callback_name: Name of registered callback.
            hour: Hour to run (0-23).
            minute: Minute to run (0-59).
            data: Data to pass to callback.

        Returns:
            Created ScheduledEvent.
        """
        event = ScheduledEvent(
            event_id=str(uuid.uuid4())[:8],
            name=name,
            schedule_type=ScheduleType.DAILY,
            callback_name=callback_name,
            data=data or {},
            run_at_hour=hour,
            run_at_minute=minute,
        )
        event.next_run = self._calculate_next_run(event)

        with self._lock:
            self.events[event.event_id] = event
            self._save_events()

        return event

    def cancel_event(self, event_id: str) -> bool:
        """Cancel a scheduled event."""
        with self._lock:
            if event_id in self.events:
                self.events[event_id].status = EventStatus.CANCELLED
                self._save_events()
                return True
        return False

    def pause_event(self, event_id: str) -> bool:
        """Pause a scheduled event."""
        with self._lock:
            if event_id in self.events:
                self.events[event_id].status = EventStatus.PAUSED
                self._save_events()
                return True
        return False

    def resume_event(self, event_id: str) -> bool:
        """Resume a paused event."""
        with self._lock:
            if event_id in self.events:
                event = self.events[event_id]
                if event.status == EventStatus.PAUSED:
                    event.status = EventStatus.ACTIVE
                    event.next_run = self._calculate_next_run(event)
                    self._save_events()
                    return True
        return False

    def _execute_event(self, event: ScheduledEvent):
        """Execute a scheduled event."""
        callback = self.callbacks.get(event.callback_name)
        if not callback:
            event.last_error = f"Callback not found: {event.callback_name}"
            event.error_count += 1
            return

        try:
            callback(event, event.data)
            event.run_count += 1
            event.last_run = time.time()
            event.last_error = None

            # Check if max runs reached
            if event.max_runs and event.run_count >= event.max_runs:
                event.status = EventStatus.COMPLETED
            else:
                # Schedule next run
                event.next_run = self._calculate_next_run(event)

        except Exception as e:
            event.last_error = str(e)
            event.error_count += 1

            # Mark as failed after too many errors
            if event.error_count >= 5:
                event.status = EventStatus.FAILED

    def _timer_loop(self):
        """Main timer loop."""
        while self._running:
            now = time.time()

            with self._lock:
                for event in self.events.values():
                    if event.status != EventStatus.ACTIVE:
                        continue

                    if event.next_run <= now:
                        self._execute_event(event)

                self._save_events()

            # Sleep for a short interval
            time.sleep(1.0)

    def start(self):
        """Start the timer system."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the timer system."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def list_events(self, status: Optional[EventStatus] = None) -> List[ScheduledEvent]:
        """List scheduled events."""
        with self._lock:
            events = list(self.events.values())

        if status is not None:
            events = [e for e in events if e.status == status]

        return sorted(events, key=lambda e: e.next_run)

    def get_status(self) -> Dict[str, Any]:
        """Get timer system status."""
        with self._lock:
            events = list(self.events.values())

        return {
            "running": self._running,
            "total_events": len(events),
            "active": sum(1 for e in events if e.status == EventStatus.ACTIVE),
            "paused": sum(1 for e in events if e.status == EventStatus.PAUSED),
            "completed": sum(1 for e in events if e.status == EventStatus.COMPLETED),
            "failed": sum(1 for e in events if e.status == EventStatus.FAILED),
            "total_runs": sum(e.run_count for e in events),
        }

    # Built-in callbacks

    def _health_check(self, event: ScheduledEvent, data: Dict[str, Any]):
        """Perform system health check."""
        health = {
            "timestamp": time.time(),
            "checks": {},
        }

        # Check DET core
        if self.core:
            try:
                state = self.core.inspect()
                health["checks"]["det_core"] = {
                    "status": "ok",
                    "tick": state["tick"],
                    "emotion": state["emotion"],
                    "presence": state["aggregates"]["presence"],
                }
            except Exception as e:
                health["checks"]["det_core"] = {"status": "error", "error": str(e)}
        else:
            health["checks"]["det_core"] = {"status": "not_configured"}

        # Check memory manager
        if self.memory_manager:
            try:
                stats = self.memory_manager.get_domain_stats()
                total_memories = sum(s["entry_count"] for s in stats.values())
                health["checks"]["memory"] = {
                    "status": "ok",
                    "total_memories": total_memories,
                    "domains": len(stats),
                }
            except Exception as e:
                health["checks"]["memory"] = {"status": "error", "error": str(e)}
        else:
            health["checks"]["memory"] = {"status": "not_configured"}

        # Log health check
        health_file = self.storage_path / "health.json"
        with open(health_file, 'w') as f:
            json.dump(health, f, indent=2)

    def _memory_consolidation(self, event: ScheduledEvent, data: Dict[str, Any]):
        """Perform memory consolidation (sleep cycle)."""
        if not self.memory_manager:
            return

        threshold_days = data.get("threshold_days", 7.0)
        consolidated = self.memory_manager.consolidate(threshold_days)

        # Log consolidation
        log = {
            "timestamp": time.time(),
            "consolidated_count": len(consolidated),
        }

        log_file = self.storage_path / "consolidation.json"
        with open(log_file, 'w') as f:
            json.dump(log, f, indent=2)

    def _det_maintenance(self, event: ScheduledEvent, data: Dict[str, Any]):
        """Perform DET core maintenance."""
        if not self.core:
            return

        # Run several simulation steps to let DET dynamics settle
        steps = data.get("steps", 10)
        dt = data.get("dt", 0.1)

        for _ in range(steps):
            self.core.step(dt)

        # Identify self
        self.core.identify_self()


def setup_default_schedule(timer: TimerSystem):
    """Set up default scheduled events."""
    # Health check every 5 minutes
    timer.schedule_interval(
        name="health_check",
        callback_name="health_check",
        interval_seconds=300,  # 5 minutes
    )

    # Memory consolidation daily at 3 AM
    timer.schedule_daily(
        name="nightly_consolidation",
        callback_name="memory_consolidation",
        hour=3,
        minute=0,
        data={"threshold_days": 7.0},
    )

    # DET maintenance every hour
    timer.schedule_interval(
        name="det_maintenance",
        callback_name="det_maintenance",
        interval_seconds=3600,  # 1 hour
        data={"steps": 5, "dt": 0.1},
    )
