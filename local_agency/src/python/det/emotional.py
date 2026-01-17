"""
DET Emotional State Integration
===============================

State-dependent behavior modulation, recovery scheduling,
and curiosity-driven exploration.

Phase 4.3 Implementation.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Callable, Dict, List, Any
import time


class EmotionalMode(IntEnum):
    """Emotional processing modes."""
    EXPLORATION = 0    # High curiosity, seeking novelty
    FOCUSED = 1        # Deep engagement with task
    RECOVERY = 2       # Rest and consolidation
    VIGILANT = 3       # High arousal, cautious processing
    NEUTRAL = 4        # Balanced state


@dataclass
class BehaviorModulation:
    """Modulation parameters for behavior based on emotional state."""
    temperature_mult: float = 1.0       # LLM temperature multiplier
    risk_tolerance: float = 0.5         # Risk acceptance [0,1]
    exploration_rate: float = 0.3       # Novelty seeking [0,1]
    response_length_mult: float = 1.0   # Response length multiplier
    retry_patience: int = 3             # Max retries before giving up
    complexity_threshold: float = 0.5   # Max complexity to attempt
    context_depth: int = 10             # Context window depth


@dataclass
class RecoveryState:
    """State for recovery scheduling."""
    needs_recovery: bool = False
    recovery_urgency: float = 0.0       # [0,1] - how urgent
    last_recovery_time: float = 0.0
    recovery_duration: float = 0.0
    strain_accumulator: float = 0.0


class EmotionalIntegration:
    """
    Integrates DET emotional state with agent behavior.

    Provides:
    - State-dependent behavior modulation
    - Recovery scheduling when strained
    - Curiosity-driven exploration triggers
    """

    # Emotional state to mode mapping thresholds
    MODE_THRESHOLDS = {
        'curiosity_arousal': 0.6,
        'curiosity_valence': 0.2,
        'focus_valence': 0.3,
        'focus_bondedness': 0.5,
        'vigilance_arousal': 0.7,
        'vigilance_valence': -0.2,
        'recovery_valence': -0.3,
        'recovery_arousal': 0.3,
    }

    # Mode-specific behavior modulations
    MODE_MODULATIONS = {
        EmotionalMode.EXPLORATION: BehaviorModulation(
            temperature_mult=1.3,
            risk_tolerance=0.7,
            exploration_rate=0.7,
            response_length_mult=0.9,
            retry_patience=5,
            complexity_threshold=0.7,
            context_depth=15,
        ),
        EmotionalMode.FOCUSED: BehaviorModulation(
            temperature_mult=0.8,
            risk_tolerance=0.4,
            exploration_rate=0.2,
            response_length_mult=1.2,
            retry_patience=4,
            complexity_threshold=0.8,
            context_depth=20,
        ),
        EmotionalMode.RECOVERY: BehaviorModulation(
            temperature_mult=0.9,
            risk_tolerance=0.2,
            exploration_rate=0.1,
            response_length_mult=0.7,
            retry_patience=2,
            complexity_threshold=0.3,
            context_depth=5,
        ),
        EmotionalMode.VIGILANT: BehaviorModulation(
            temperature_mult=0.7,
            risk_tolerance=0.1,
            exploration_rate=0.1,
            response_length_mult=0.8,
            retry_patience=2,
            complexity_threshold=0.4,
            context_depth=8,
        ),
        EmotionalMode.NEUTRAL: BehaviorModulation(
            temperature_mult=1.0,
            risk_tolerance=0.5,
            exploration_rate=0.3,
            response_length_mult=1.0,
            retry_patience=3,
            complexity_threshold=0.5,
            context_depth=10,
        ),
    }

    def __init__(
        self,
        core=None,
        timer_system=None,
        recovery_threshold: float = 0.4,
        min_recovery_interval: float = 300.0  # 5 minutes
    ):
        """
        Initialize emotional integration.

        Args:
            core: DETCore instance for state monitoring.
            timer_system: TimerSystem for recovery scheduling.
            recovery_threshold: Valence threshold triggering recovery.
            min_recovery_interval: Minimum time between recoveries.
        """
        self.core = core
        self.timer_system = timer_system
        self.recovery_threshold = recovery_threshold
        self.min_recovery_interval = min_recovery_interval

        self.recovery = RecoveryState()
        self._current_mode = EmotionalMode.NEUTRAL
        self._mode_history: List[tuple] = []  # (timestamp, mode)

        # Callbacks
        self.on_mode_change: Optional[Callable[[EmotionalMode, EmotionalMode], None]] = None
        self.on_recovery_triggered: Optional[Callable[[float], None]] = None
        self.on_curiosity_spike: Optional[Callable[[float], None]] = None

        # Register with timer system if available
        if self.timer_system:
            self.timer_system.register_callback(
                "emotional_check",
                self._emotional_check_callback
            )

    @property
    def current_mode(self) -> EmotionalMode:
        """Get current emotional processing mode."""
        return self._current_mode

    @property
    def modulation(self) -> BehaviorModulation:
        """Get current behavior modulation parameters."""
        return self.MODE_MODULATIONS[self._current_mode]

    def update(self) -> EmotionalMode:
        """
        Update emotional state and return current mode.

        Should be called periodically or before processing requests.
        """
        if not self.core:
            return EmotionalMode.NEUTRAL

        # Get current affect from DET core
        valence, arousal, bondedness = self.core.get_self_affect()

        # Determine mode
        new_mode = self._determine_mode(valence, arousal, bondedness)

        # Track mode changes
        if new_mode != self._current_mode:
            old_mode = self._current_mode
            self._current_mode = new_mode
            self._mode_history.append((time.time(), new_mode))

            # Trim history
            if len(self._mode_history) > 100:
                self._mode_history = self._mode_history[-50:]

            # Notify callback
            if self.on_mode_change:
                self.on_mode_change(old_mode, new_mode)

        # Check recovery needs
        self._check_recovery(valence, arousal)

        # Check curiosity spikes
        self._check_curiosity(valence, arousal, bondedness)

        return self._current_mode

    def _determine_mode(
        self,
        valence: float,
        arousal: float,
        bondedness: float
    ) -> EmotionalMode:
        """Determine emotional mode from affect values."""
        th = self.MODE_THRESHOLDS

        # Recovery: negative valence, low arousal
        if valence < th['recovery_valence'] and arousal < th['recovery_arousal']:
            return EmotionalMode.RECOVERY

        # Vigilance: high arousal, negative valence
        if arousal > th['vigilance_arousal'] and valence < th['vigilance_valence']:
            return EmotionalMode.VIGILANT

        # Exploration: moderate-high arousal, positive valence
        if arousal > th['curiosity_arousal'] and valence > th['curiosity_valence']:
            return EmotionalMode.EXPLORATION

        # Focus: positive valence, high bondedness
        if valence > th['focus_valence'] and bondedness > th['focus_bondedness']:
            return EmotionalMode.FOCUSED

        return EmotionalMode.NEUTRAL

    def _check_recovery(self, valence: float, arousal: float):
        """Check if recovery is needed and schedule if necessary."""
        now = time.time()

        # Accumulate strain
        if valence < 0:
            self.recovery.strain_accumulator += abs(valence) * 0.1
        else:
            self.recovery.strain_accumulator *= 0.95  # Decay

        # Check if recovery needed
        needs_recovery = (
            valence < self.recovery_threshold or
            self.recovery.strain_accumulator > 1.0
        )

        self.recovery.needs_recovery = needs_recovery
        self.recovery.recovery_urgency = min(1.0, abs(valence) + self.recovery.strain_accumulator)

        # Schedule recovery if needed and enough time has passed
        if needs_recovery and self.timer_system:
            time_since_last = now - self.recovery.last_recovery_time
            if time_since_last > self.min_recovery_interval:
                self._schedule_recovery()

    def _schedule_recovery(self):
        """Schedule a recovery period."""
        if not self.timer_system:
            return

        # Calculate recovery duration based on urgency
        duration = 30.0 + 60.0 * self.recovery.recovery_urgency  # 30-90 seconds

        self.recovery.last_recovery_time = time.time()
        self.recovery.recovery_duration = duration

        # Notify callback
        if self.on_recovery_triggered:
            self.on_recovery_triggered(duration)

        # Schedule recovery completion
        self.timer_system.schedule_once(
            name="recovery_complete",
            callback_name="emotional_check",
            run_at=time.time() + duration,
            data={"action": "recovery_complete"}
        )

    def _check_curiosity(self, valence: float, arousal: float, bondedness: float):
        """Check for curiosity spikes that might trigger exploration."""
        if not self.core:
            return

        # Curiosity spike: high arousal + positive valence + low bondedness
        # (interested but not attached to current focus)
        curiosity_score = arousal * max(0, valence) * (1.0 - bondedness * 0.5)

        if curiosity_score > 0.4 and self.on_curiosity_spike:
            self.on_curiosity_spike(curiosity_score)

    def _emotional_check_callback(self, event, data: Dict[str, Any]):
        """Timer callback for emotional state checking."""
        action = data.get("action")

        if action == "recovery_complete":
            self.recovery.strain_accumulator *= 0.5  # Reduce strain after recovery
            self.recovery.needs_recovery = False

        # Update emotional state
        self.update()

    def get_llm_temperature(self, base_temperature: float = 0.7) -> float:
        """Get temperature adjusted for emotional state."""
        return base_temperature * self.modulation.temperature_mult

    def should_attempt_task(self, complexity: float) -> bool:
        """Check if a task should be attempted given current state."""
        return complexity <= self.modulation.complexity_threshold

    def get_max_retries(self) -> int:
        """Get maximum retry count for current state."""
        return self.modulation.retry_patience

    def get_status(self) -> Dict[str, Any]:
        """Get emotional integration status."""
        if self.core:
            valence, arousal, bondedness = self.core.get_self_affect()
        else:
            valence, arousal, bondedness = 0.0, 0.0, 0.0

        return {
            "mode": self._current_mode.name,
            "affect": {
                "valence": valence,
                "arousal": arousal,
                "bondedness": bondedness,
            },
            "recovery": {
                "needs_recovery": self.recovery.needs_recovery,
                "urgency": self.recovery.recovery_urgency,
                "strain": self.recovery.strain_accumulator,
            },
            "modulation": {
                "temperature_mult": self.modulation.temperature_mult,
                "risk_tolerance": self.modulation.risk_tolerance,
                "exploration_rate": self.modulation.exploration_rate,
                "complexity_threshold": self.modulation.complexity_threshold,
            },
            "mode_history_length": len(self._mode_history),
        }

    def inject_grace_if_needed(self):
        """
        Inject grace to nodes needing recovery.

        Call this during recovery periods to help restore the DET core.
        """
        if not self.core:
            return

        total_needed = self.core.total_grace_needed()
        if total_needed <= 0:
            return

        # Distribute grace to nodes that need it
        for i in range(self.core.num_nodes):
            if self.core.needs_grace(i):
                # Grace amount proportional to need
                node = self.core.get_node(i)
                amount = (0.5 - node.F) * 0.5 + node.q * 0.3
                self.core.inject_grace(i, max(0.1, amount))


@dataclass
class SessionContext:
    """Context for a user session."""
    session_id: str
    start_time: float = field(default_factory=time.time)
    message_count: int = 0
    emotional_mode_at_start: EmotionalMode = EmotionalMode.NEUTRAL
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "message_count": self.message_count,
            "emotional_mode_at_start": self.emotional_mode_at_start.name,
            "topics": self.topics,
            "duration": time.time() - self.start_time,
        }


class MultiSessionManager:
    """
    Manages multiple sessions with shared DET core.

    Phase 4.4: Multi-Session Support

    Provides:
    - Session-specific context isolation
    - Cross-session memory sharing via DET core
    - Session state persistence
    """

    def __init__(
        self,
        core=None,
        emotional: Optional[EmotionalIntegration] = None,
        state_dir: Optional[str] = None
    ):
        """
        Initialize multi-session manager.

        Args:
            core: Shared DETCore instance.
            emotional: EmotionalIntegration instance.
            state_dir: Directory for session state files.
        """
        self.core = core
        self.emotional = emotional
        self.state_dir = state_dir

        self.sessions: Dict[str, SessionContext] = {}
        self._active_session: Optional[str] = None

    @property
    def active_session(self) -> Optional[SessionContext]:
        """Get current active session."""
        if self._active_session and self._active_session in self.sessions:
            return self.sessions[self._active_session]
        return None

    def create_session(self, session_id: Optional[str] = None) -> SessionContext:
        """
        Create a new session.

        Args:
            session_id: Optional session ID (generated if not provided).

        Returns:
            Created SessionContext.
        """
        import uuid

        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        # Capture current emotional mode
        mode = EmotionalMode.NEUTRAL
        if self.emotional:
            mode = self.emotional.current_mode

        session = SessionContext(
            session_id=session_id,
            emotional_mode_at_start=mode,
        )

        self.sessions[session_id] = session
        self._active_session = session_id

        return session

    def switch_session(self, session_id: str) -> bool:
        """
        Switch to a different session.

        Args:
            session_id: Session ID to switch to.

        Returns:
            True if switch succeeded.
        """
        if session_id not in self.sessions:
            return False

        self._active_session = session_id
        return True

    def end_session(self, session_id: Optional[str] = None) -> bool:
        """
        End a session.

        Args:
            session_id: Session to end (active session if None).

        Returns:
            True if session was ended.
        """
        if session_id is None:
            session_id = self._active_session

        if session_id not in self.sessions:
            return False

        # Save session state if state_dir configured
        if self.state_dir:
            self._save_session_state(session_id)

        del self.sessions[session_id]

        if self._active_session == session_id:
            self._active_session = next(iter(self.sessions), None)

        return True

    def record_message(self, session_id: Optional[str] = None):
        """Record a message in the session."""
        if session_id is None:
            session_id = self._active_session

        if session_id and session_id in self.sessions:
            self.sessions[session_id].message_count += 1

    def add_topic(self, topic: str, session_id: Optional[str] = None):
        """Add a topic to the session."""
        if session_id is None:
            session_id = self._active_session

        if session_id and session_id in self.sessions:
            if topic not in self.sessions[session_id].topics:
                self.sessions[session_id].topics.append(topic)

    def save_core_state(self) -> bool:
        """Save DET core state to file."""
        if not self.core or not self.state_dir:
            return False

        import os
        os.makedirs(self.state_dir, exist_ok=True)

        filepath = os.path.join(self.state_dir, "core_state.det")
        self.core.save_to_file(filepath)
        return True

    def load_core_state(self) -> bool:
        """Load DET core state from file."""
        if not self.core or not self.state_dir:
            return False

        import os
        filepath = os.path.join(self.state_dir, "core_state.det")

        if not os.path.exists(filepath):
            return False

        return self.core.load_from_file(filepath)

    def _save_session_state(self, session_id: str):
        """Save session state to file."""
        if not self.state_dir:
            return

        import os
        import json

        os.makedirs(self.state_dir, exist_ok=True)

        session = self.sessions.get(session_id)
        if not session:
            return

        filepath = os.path.join(self.state_dir, f"session_{session_id}.json")
        with open(filepath, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "active_session": self._active_session,
            "session_count": len(self.sessions),
            "sessions": {
                sid: s.to_dict() for sid, s in self.sessions.items()
            },
            "has_core": self.core is not None,
            "has_emotional": self.emotional is not None,
            "state_dir": self.state_dir,
        }
