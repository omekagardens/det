"""
DET Sleep/Consolidation System
==============================

Automated memory training during idle periods.

Phase 5.2 Implementation.
"""

import json
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from enum import Enum

from .timer import TimerSystem, ScheduledEvent, ScheduleType
from .training import (
    TrainingConfig, TrainingJob, TrainingStatus,
    MemoryRetuner, is_mlx_available
)
from .memory import MemoryDomain


class ConsolidationState(Enum):
    """State of the consolidation system."""
    IDLE = "idle"
    MONITORING = "monitoring"
    CONSOLIDATING = "consolidating"
    TRAINING = "training"
    RECOVERING = "recovering"


class ConsolidationPhase(Enum):
    """Phases of the consolidation cycle."""
    MEMORY_SCAN = "memory_scan"
    DATA_GENERATION = "data_generation"
    MODEL_TRAINING = "model_training"
    GRACE_INJECTION = "grace_injection"
    VERIFICATION = "verification"


@dataclass
class IdleDetector:
    """Detects idle periods for consolidation."""
    idle_threshold_seconds: float = 300.0  # 5 minutes
    last_activity: float = field(default_factory=time.time)
    activity_count: int = 0

    def record_activity(self):
        """Record user activity."""
        self.last_activity = time.time()
        self.activity_count += 1

    def is_idle(self) -> bool:
        """Check if system is idle."""
        return (time.time() - self.last_activity) > self.idle_threshold_seconds

    def idle_duration(self) -> float:
        """Get current idle duration in seconds."""
        return max(0, time.time() - self.last_activity)


@dataclass
class ConsolidationConfig:
    """Configuration for consolidation cycles."""
    # Timing
    idle_threshold_seconds: float = 300.0  # 5 min before starting
    min_consolidation_interval: float = 3600.0  # 1 hour between consolidations
    max_consolidation_duration: float = 1800.0  # 30 min max per cycle

    # Training
    training_config: Optional[TrainingConfig] = None
    min_examples_per_domain: int = 10
    max_domains_per_cycle: int = 3

    # DET integration
    require_positive_valence: bool = True
    max_strain_threshold: float = 0.7
    grace_injection_enabled: bool = True

    # Domains to consolidate (None = all)
    target_domains: Optional[List[MemoryDomain]] = None


@dataclass
class ConsolidationCycle:
    """Represents a consolidation cycle."""
    cycle_id: str
    started_at: float
    completed_at: Optional[float] = None
    state: ConsolidationState = ConsolidationState.CONSOLIDATING
    phase: ConsolidationPhase = ConsolidationPhase.MEMORY_SCAN
    domains_processed: List[str] = field(default_factory=list)
    training_jobs: List[str] = field(default_factory=list)
    memories_consolidated: int = 0
    examples_generated: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "state": self.state.value,
            "phase": self.phase.value,
            "domains_processed": self.domains_processed,
            "training_jobs": self.training_jobs,
            "memories_consolidated": self.memories_consolidated,
            "examples_generated": self.examples_generated,
            "error": self.error,
            "duration": (self.completed_at or time.time()) - self.started_at,
        }


class ConsolidationManager:
    """
    Manages sleep/consolidation cycles with MLX training integration.

    Provides:
    - Idle detection for automatic consolidation
    - Scheduled consolidation via TimerSystem
    - MLX-based memory retraining
    - DET affect-gated consolidation
    """

    def __init__(
        self,
        core=None,
        memory_manager=None,
        timer_system: Optional[TimerSystem] = None,
        retuner: Optional[MemoryRetuner] = None,
        config: Optional[ConsolidationConfig] = None,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize the consolidation manager.

        Args:
            core: DETCore instance.
            memory_manager: MemoryManager instance.
            timer_system: TimerSystem for scheduling.
            retuner: MemoryRetuner for training.
            config: Consolidation configuration.
            storage_path: Path for logs and state.
        """
        self.core = core
        self.memory_manager = memory_manager
        self.timer_system = timer_system
        self.config = config or ConsolidationConfig()
        self.storage_path = storage_path or Path.home() / ".det_agency" / "consolidation"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create retuner if not provided
        if retuner:
            self.retuner = retuner
        elif is_mlx_available() and core and memory_manager:
            training_config = self.config.training_config or TrainingConfig()
            self.retuner = MemoryRetuner(
                core=core,
                memory_manager=memory_manager,
                config=training_config
            )
        else:
            self.retuner = None

        # State
        self.idle_detector = IdleDetector(
            idle_threshold_seconds=self.config.idle_threshold_seconds
        )
        self._state = ConsolidationState.IDLE
        self._current_cycle: Optional[ConsolidationCycle] = None
        self._cycle_history: List[ConsolidationCycle] = []
        self._last_consolidation: float = 0.0

        # Threading
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        self._lock = threading.Lock()

        # Callbacks
        self.on_cycle_start: Optional[Callable[[ConsolidationCycle], None]] = None
        self.on_cycle_complete: Optional[Callable[[ConsolidationCycle], None]] = None
        self.on_training_complete: Optional[Callable[[TrainingJob], None]] = None

        # Register with timer system
        if self.timer_system:
            self._register_timer_callbacks()

        # Load history
        self._load_history()

    def _register_timer_callbacks(self):
        """Register callbacks with timer system."""
        self.timer_system.register_callback(
            "consolidation_cycle",
            self._timer_consolidation_callback
        )

    def _load_history(self):
        """Load consolidation history."""
        history_file = self.storage_path / "history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self._last_consolidation = data.get("last_consolidation", 0.0)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_history(self):
        """Save consolidation history."""
        history_file = self.storage_path / "history.json"
        data = {
            "last_consolidation": self._last_consolidation,
            "cycle_count": len(self._cycle_history),
        }
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)

    @property
    def state(self) -> ConsolidationState:
        """Get current consolidation state."""
        return self._state

    def record_activity(self):
        """Record user activity (resets idle timer)."""
        self.idle_detector.record_activity()

    def is_idle(self) -> bool:
        """Check if system is idle."""
        return self.idle_detector.is_idle()

    def can_consolidate(self) -> bool:
        """
        Check if consolidation can proceed.

        Considers:
        - Idle state
        - Time since last consolidation
        - DET affect state
        - MLX availability
        """
        # Check idle
        if not self.is_idle():
            return False

        # Check minimum interval
        time_since_last = time.time() - self._last_consolidation
        if time_since_last < self.config.min_consolidation_interval:
            return False

        # Check DET state
        if self.core and self.config.require_positive_valence:
            valence, arousal, bondedness = self.core.get_self_affect()
            if valence < 0:
                return False

        # Check if we have something to train
        if not self.memory_manager:
            return False

        # Check if MLX is available for training
        if not is_mlx_available():
            return False

        return True

    def start_monitoring(self):
        """Start idle monitoring for automatic consolidation."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_monitor.clear()
        self._state = ConsolidationState.MONITORING

        def monitor_loop():
            while not self._stop_monitor.wait(30.0):  # Check every 30 seconds
                if self.can_consolidate():
                    self.start_consolidation()

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop idle monitoring."""
        self._stop_monitor.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self._state = ConsolidationState.IDLE

    def start_consolidation(self) -> Optional[ConsolidationCycle]:
        """
        Start a consolidation cycle.

        Returns:
            ConsolidationCycle if started, None if cannot start.
        """
        with self._lock:
            if self._current_cycle:
                return None  # Already consolidating

            if not self.can_consolidate():
                return None

            import uuid
            cycle = ConsolidationCycle(
                cycle_id=str(uuid.uuid4())[:8],
                started_at=time.time(),
            )
            self._current_cycle = cycle
            self._state = ConsolidationState.CONSOLIDATING

        if self.on_cycle_start:
            self.on_cycle_start(cycle)

        # Run consolidation in thread
        thread = threading.Thread(
            target=self._run_consolidation,
            args=(cycle,),
            daemon=True
        )
        thread.start()

        return cycle

    def _run_consolidation(self, cycle: ConsolidationCycle):
        """Run the consolidation cycle."""
        try:
            # Phase 1: Memory scan
            cycle.phase = ConsolidationPhase.MEMORY_SCAN
            domains_to_process = self._scan_domains()

            if not domains_to_process:
                cycle.state = ConsolidationState.IDLE
                cycle.completed_at = time.time()
                return

            # Limit domains per cycle
            domains_to_process = domains_to_process[:self.config.max_domains_per_cycle]

            # Phase 2: Data generation (done by retuner)
            cycle.phase = ConsolidationPhase.DATA_GENERATION

            # Phase 3: Model training
            cycle.phase = ConsolidationPhase.MODEL_TRAINING
            self._state = ConsolidationState.TRAINING

            for domain in domains_to_process:
                if not self.retuner:
                    break

                # Check if we've exceeded max duration
                if (time.time() - cycle.started_at) > self.config.max_consolidation_duration:
                    break

                # Check DET state before training
                if self.core:
                    valence, _, _ = self.core.get_self_affect()
                    if valence < -0.3:
                        break  # Too strained

                # Start training
                job = self.retuner.retrain_from_memories(
                    domain=domain.name,
                    callback=self.on_training_complete
                )

                if job:
                    cycle.training_jobs.append(job.job_id)
                    cycle.domains_processed.append(domain.name)

            # Phase 4: Grace injection
            if self.config.grace_injection_enabled and self.core:
                cycle.phase = ConsolidationPhase.GRACE_INJECTION
                self._state = ConsolidationState.RECOVERING
                self._inject_recovery_grace()

            # Phase 5: Verification
            cycle.phase = ConsolidationPhase.VERIFICATION
            self._verify_consolidation(cycle)

            cycle.state = ConsolidationState.IDLE
            cycle.completed_at = time.time()

        except Exception as e:
            cycle.error = str(e)
            cycle.state = ConsolidationState.IDLE
            cycle.completed_at = time.time()

        finally:
            with self._lock:
                self._current_cycle = None
                self._last_consolidation = time.time()
                self._cycle_history.append(cycle)
                self._state = ConsolidationState.MONITORING if self._monitor_thread else ConsolidationState.IDLE

            self._save_history()
            self._save_cycle_log(cycle)

            if self.on_cycle_complete:
                self.on_cycle_complete(cycle)

    def _scan_domains(self) -> List[MemoryDomain]:
        """Scan for domains that need consolidation."""
        if not self.memory_manager:
            return []

        domains_to_process = []
        target_domains = self.config.target_domains or list(MemoryDomain)

        for domain in target_domains:
            memories = self.memory_manager.memories.get(domain, [])
            if len(memories) >= self.config.min_examples_per_domain:
                domains_to_process.append(domain)

        # Sort by memory count (most memories first)
        domains_to_process.sort(
            key=lambda d: len(self.memory_manager.memories.get(d, [])),
            reverse=True
        )

        return domains_to_process

    def _inject_recovery_grace(self):
        """Inject grace during recovery phase."""
        if not self.core:
            return

        # Use emotional integration if available
        try:
            from .emotional import EmotionalIntegration
            emotional = EmotionalIntegration(core=self.core)
            emotional.inject_grace_if_needed()
        except ImportError:
            # Fallback: inject grace manually
            for i in range(min(10, self.core.num_active)):
                if self.core.needs_grace(i):
                    self.core.inject_grace(i, 0.3)

    def _verify_consolidation(self, cycle: ConsolidationCycle):
        """Verify consolidation completed successfully."""
        # Check DET state is stable
        if self.core:
            for _ in range(5):
                self.core.step(0.1)

            valence, _, _ = self.core.get_self_affect()
            if valence < -0.2:
                cycle.error = "Post-consolidation valence negative"

    def _save_cycle_log(self, cycle: ConsolidationCycle):
        """Save cycle log to file."""
        log_file = self.storage_path / f"cycle_{cycle.cycle_id}.json"
        with open(log_file, 'w') as f:
            json.dump(cycle.to_dict(), f, indent=2)

    def _timer_consolidation_callback(self, event: ScheduledEvent, data: Dict[str, Any]):
        """Timer callback for scheduled consolidation."""
        # Force consolidation regardless of idle state
        self._last_consolidation = 0  # Reset to allow immediate consolidation

        # Temporarily mark as idle
        original_threshold = self.idle_detector.idle_threshold_seconds
        self.idle_detector.idle_threshold_seconds = 0
        self.idle_detector.last_activity = 0

        try:
            self.start_consolidation()
        finally:
            self.idle_detector.idle_threshold_seconds = original_threshold

    def schedule_consolidation(
        self,
        hour: int = 3,
        minute: int = 0
    ) -> Optional[ScheduledEvent]:
        """
        Schedule daily consolidation.

        Args:
            hour: Hour to run (0-23).
            minute: Minute to run (0-59).

        Returns:
            ScheduledEvent if scheduled.
        """
        if not self.timer_system:
            return None

        return self.timer_system.schedule_daily(
            name="nightly_consolidation",
            callback_name="consolidation_cycle",
            hour=hour,
            minute=minute,
            data={"force": True}
        )

    def schedule_interval_consolidation(
        self,
        interval_hours: float = 4.0
    ) -> Optional[ScheduledEvent]:
        """
        Schedule periodic consolidation.

        Args:
            interval_hours: Hours between consolidations.

        Returns:
            ScheduledEvent if scheduled.
        """
        if not self.timer_system:
            return None

        return self.timer_system.schedule_interval(
            name="periodic_consolidation",
            callback_name="consolidation_cycle",
            interval_seconds=interval_hours * 3600,
            data={"force": False}
        )

    def get_current_cycle(self) -> Optional[ConsolidationCycle]:
        """Get the current consolidation cycle."""
        return self._current_cycle

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get consolidation history."""
        return [c.to_dict() for c in self._cycle_history[-limit:]]

    def get_status(self) -> Dict[str, Any]:
        """Get consolidation manager status."""
        return {
            "state": self._state.value,
            "is_idle": self.is_idle(),
            "idle_duration": self.idle_detector.idle_duration(),
            "can_consolidate": self.can_consolidate(),
            "last_consolidation": self._last_consolidation,
            "time_since_last": time.time() - self._last_consolidation,
            "current_cycle": self._current_cycle.to_dict() if self._current_cycle else None,
            "cycle_count": len(self._cycle_history),
            "mlx_available": is_mlx_available(),
            "has_retuner": self.retuner is not None,
            "has_timer": self.timer_system is not None,
        }


def setup_consolidation(
    core=None,
    memory_manager=None,
    timer_system: Optional[TimerSystem] = None,
    config: Optional[ConsolidationConfig] = None,
    auto_start: bool = True
) -> ConsolidationManager:
    """
    Convenience function to set up consolidation with defaults.

    Args:
        core: DETCore instance.
        memory_manager: MemoryManager instance.
        timer_system: TimerSystem instance.
        config: Optional configuration.
        auto_start: Whether to start monitoring.

    Returns:
        Configured ConsolidationManager.
    """
    manager = ConsolidationManager(
        core=core,
        memory_manager=memory_manager,
        timer_system=timer_system,
        config=config,
    )

    if timer_system:
        # Schedule nightly consolidation at 3 AM
        manager.schedule_consolidation(hour=3, minute=0)

    if auto_start:
        manager.start_monitoring()

    return manager
