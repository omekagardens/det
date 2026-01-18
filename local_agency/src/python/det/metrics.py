"""
DET Metrics and Logging
=======================

Phase 6.4 - Development Tools

Provides:
- Cluster health metrics (P, C, q averages)
- Escalation/compilation event logging
- Emotional state timeline
- Performance profiling
"""

import time
import threading
import resource
import statistics
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from collections import deque
from datetime import datetime


class DETEventType(Enum):
    """Types of DET events to log."""
    ESCALATION = "escalation"
    COMPILATION = "compilation"
    RECRUITMENT = "recruitment"
    RETIREMENT = "retirement"
    BOND_FORMED = "bond_formed"
    BOND_BROKEN = "bond_broken"
    PRISON_DETECTED = "prison_detected"
    GRACE_INJECTED = "grace_injected"
    DOMAIN_ACTIVATED = "domain_activated"
    GATEKEEPER_DECISION = "gatekeeper_decision"


@dataclass
class DETEvent:
    """A logged DET event."""
    event_type: DETEventType
    timestamp: float
    tick: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "tick": self.tick,
            "details": self.details,
            "time_str": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S.%f")[:-3],
        }


@dataclass
class MetricsSample:
    """A single metrics sample."""
    timestamp: float
    tick: int
    presence: float
    coherence: float
    resource: float
    debt: float
    valence: float
    arousal: float
    bondedness: float
    num_active: int
    num_bonds: int
    self_cluster_size: int
    tick_time_ms: float = 0.0
    memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "tick": self.tick,
            "presence": self.presence,
            "coherence": self.coherence,
            "resource": self.resource,
            "debt": self.debt,
            "valence": self.valence,
            "arousal": self.arousal,
            "bondedness": self.bondedness,
            "num_active": self.num_active,
            "num_bonds": self.num_bonds,
            "self_cluster_size": self.self_cluster_size,
            "tick_time_ms": self.tick_time_ms,
            "memory_mb": self.memory_mb,
        }


class MetricsCollector:
    """
    Collects and stores DET metrics over time.

    Provides rolling windows for dashboard display and
    historical data for timeline visualization.
    """

    def __init__(self, max_samples: int = 1000, max_events: int = 500):
        """
        Initialize metrics collector.

        Args:
            max_samples: Maximum number of metric samples to retain.
            max_events: Maximum number of events to retain.
        """
        self._samples: deque = deque(maxlen=max_samples)
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()

        # Performance tracking
        self._last_tick_start: float = 0.0
        self._tick_times: deque = deque(maxlen=100)

        # Event callbacks
        self._event_callbacks: List[Callable[[DETEvent], None]] = []

    def sample(self, core) -> Optional[MetricsSample]:
        """
        Take a metrics sample from the DET core.

        Args:
            core: DETCore instance.

        Returns:
            MetricsSample or None if core is unavailable.
        """
        if not core:
            return None

        try:
            # Get aggregates
            p, c, f, q = core.get_aggregates()

            # Get affect
            v, a, b = core.get_self_affect()

            # Get self cluster size
            core.identify_self()
            self_size = core._core.contents.self.num_nodes

            # Get memory usage
            mem_info = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = mem_info.ru_maxrss / (1024 * 1024)  # Convert to MB

            # Calculate tick time
            tick_time_ms = 0.0
            if self._tick_times:
                tick_time_ms = statistics.mean(self._tick_times)

            sample = MetricsSample(
                timestamp=time.time(),
                tick=core.tick,
                presence=p,
                coherence=c,
                resource=f,
                debt=q,
                valence=v,
                arousal=a,
                bondedness=b,
                num_active=core.num_active,
                num_bonds=core.num_bonds,
                self_cluster_size=self_size,
                tick_time_ms=tick_time_ms,
                memory_mb=memory_mb,
            )

            with self._lock:
                self._samples.append(sample)

            return sample

        except Exception:
            return None

    def start_tick(self):
        """Mark the start of a simulation tick for timing."""
        self._last_tick_start = time.perf_counter()

    def end_tick(self):
        """Mark the end of a simulation tick and record timing."""
        if self._last_tick_start > 0:
            elapsed_ms = (time.perf_counter() - self._last_tick_start) * 1000
            self._tick_times.append(elapsed_ms)

    def log_event(self, event_type: DETEventType, tick: int, **details):
        """
        Log a DET event.

        Args:
            event_type: Type of event.
            tick: Current simulation tick.
            **details: Additional event details.
        """
        event = DETEvent(
            event_type=event_type,
            timestamp=time.time(),
            tick=tick,
            details=details,
        )

        with self._lock:
            self._events.append(event)

        # Notify callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception:
                pass

    def add_event_callback(self, callback: Callable[[DETEvent], None]):
        """Add a callback for DET events."""
        self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: Callable[[DETEvent], None]):
        """Remove an event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    def get_samples(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent metric samples.

        Args:
            limit: Maximum number of samples to return.

        Returns:
            List of sample dictionaries.
        """
        with self._lock:
            samples = list(self._samples)

        if limit:
            samples = samples[-limit:]

        return [s.to_dict() for s in samples]

    def get_events(self, limit: Optional[int] = None,
                   event_type: Optional[DETEventType] = None) -> List[Dict[str, Any]]:
        """
        Get recent events.

        Args:
            limit: Maximum number of events to return.
            event_type: Filter by event type.

        Returns:
            List of event dictionaries.
        """
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if limit:
            events = events[-limit:]

        return [e.to_dict() for e in events]

    def get_timeline(self, field: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get timeline data for a specific field.

        Args:
            field: Field name (valence, arousal, bondedness, presence, etc.)
            limit: Maximum number of points.

        Returns:
            List of {timestamp, tick, value} dicts.
        """
        with self._lock:
            samples = list(self._samples)

        if limit:
            samples = samples[-limit:]

        result = []
        for s in samples:
            value = getattr(s, field, None)
            if value is not None:
                result.append({
                    "timestamp": s.timestamp,
                    "tick": s.tick,
                    "value": value,
                })

        return result

    def get_dashboard(self) -> Dict[str, Any]:
        """
        Get dashboard data with current values and trends.

        Returns:
            Dashboard dictionary with current values and statistics.
        """
        with self._lock:
            samples = list(self._samples)
            events = list(self._events)

        if not samples:
            return {
                "current": None,
                "trends": {},
                "events_count": len(events),
                "recent_events": [],
            }

        current = samples[-1]

        # Calculate trends (comparing last 10 to previous 10)
        trends = {}
        if len(samples) >= 20:
            recent = samples[-10:]
            previous = samples[-20:-10]

            for field in ["presence", "coherence", "resource", "debt",
                          "valence", "arousal", "bondedness"]:
                recent_avg = statistics.mean(getattr(s, field) for s in recent)
                prev_avg = statistics.mean(getattr(s, field) for s in previous)

                if prev_avg != 0:
                    change = (recent_avg - prev_avg) / abs(prev_avg)
                else:
                    change = 0.0

                trends[field] = {
                    "current": recent_avg,
                    "previous": prev_avg,
                    "change": change,
                    "direction": "up" if change > 0.01 else "down" if change < -0.01 else "stable",
                }

        # Performance stats
        perf = {}
        if self._tick_times:
            perf = {
                "avg_tick_ms": statistics.mean(self._tick_times),
                "max_tick_ms": max(self._tick_times),
                "min_tick_ms": min(self._tick_times),
            }

        # Recent events (last 10)
        recent_events = [e.to_dict() for e in events[-10:]]

        return {
            "current": current.to_dict(),
            "trends": trends,
            "performance": perf,
            "events_count": len(events),
            "recent_events": recent_events,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistical summary of all collected data.

        Returns:
            Statistics dictionary.
        """
        with self._lock:
            samples = list(self._samples)
            events = list(self._events)

        if not samples:
            return {"sample_count": 0, "event_count": len(events)}

        stats = {
            "sample_count": len(samples),
            "event_count": len(events),
            "time_range": {
                "start": samples[0].timestamp,
                "end": samples[-1].timestamp,
                "duration_seconds": samples[-1].timestamp - samples[0].timestamp,
            },
            "tick_range": {
                "start": samples[0].tick,
                "end": samples[-1].tick,
            },
        }

        # Add field statistics
        for field in ["presence", "coherence", "resource", "debt",
                      "valence", "arousal", "bondedness"]:
            values = [getattr(s, field) for s in samples]
            stats[field] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

        # Event type counts
        event_counts = {}
        for e in events:
            t = e.event_type.value
            event_counts[t] = event_counts.get(t, 0) + 1
        stats["event_counts"] = event_counts

        return stats

    def clear(self):
        """Clear all collected data."""
        with self._lock:
            self._samples.clear()
            self._events.clear()
            self._tick_times.clear()


class Profiler:
    """
    Performance profiler for DET simulation.

    Tracks tick times, memory usage, and identifies bottlenecks.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize profiler.

        Args:
            window_size: Number of samples to keep for rolling statistics.
        """
        self._tick_times: deque = deque(maxlen=window_size)
        self._step_times: Dict[str, deque] = {}
        self._memory_samples: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

        self._current_tick_start: float = 0.0
        self._current_step_start: float = 0.0
        self._current_step_name: str = ""

    def start_tick(self):
        """Start timing a simulation tick."""
        self._current_tick_start = time.perf_counter()

    def end_tick(self):
        """End timing the current tick."""
        if self._current_tick_start > 0:
            elapsed = (time.perf_counter() - self._current_tick_start) * 1000
            with self._lock:
                self._tick_times.append(elapsed)

    def start_step(self, name: str):
        """Start timing a named step within a tick."""
        self._current_step_name = name
        self._current_step_start = time.perf_counter()

    def end_step(self):
        """End timing the current step."""
        if self._current_step_start > 0 and self._current_step_name:
            elapsed = (time.perf_counter() - self._current_step_start) * 1000
            with self._lock:
                if self._current_step_name not in self._step_times:
                    self._step_times[self._current_step_name] = deque(maxlen=100)
                self._step_times[self._current_step_name].append(elapsed)

    def sample_memory(self):
        """Sample current memory usage."""
        mem_info = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = mem_info.ru_maxrss / (1024 * 1024)
        with self._lock:
            self._memory_samples.append({
                "timestamp": time.time(),
                "memory_mb": memory_mb,
            })

    def get_tick_stats(self) -> Dict[str, float]:
        """Get tick timing statistics."""
        with self._lock:
            times = list(self._tick_times)

        if not times:
            return {"count": 0}

        return {
            "count": len(times),
            "avg_ms": statistics.mean(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "p50_ms": statistics.median(times),
            "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) >= 20 else max(times),
        }

    def get_step_stats(self) -> Dict[str, Dict[str, float]]:
        """Get per-step timing statistics."""
        with self._lock:
            step_data = {k: list(v) for k, v in self._step_times.items()}

        result = {}
        for name, times in step_data.items():
            if times:
                result[name] = {
                    "count": len(times),
                    "avg_ms": statistics.mean(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                }

        return result

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            samples = list(self._memory_samples)

        if not samples:
            return {"count": 0}

        values = [s["memory_mb"] for s in samples]

        return {
            "count": len(values),
            "current_mb": values[-1] if values else 0,
            "min_mb": min(values),
            "max_mb": max(values),
            "avg_mb": statistics.mean(values),
        }

    def get_report(self) -> Dict[str, Any]:
        """Get full profiling report."""
        return {
            "tick": self.get_tick_stats(),
            "steps": self.get_step_stats(),
            "memory": self.get_memory_stats(),
        }

    def clear(self):
        """Clear all profiling data."""
        with self._lock:
            self._tick_times.clear()
            self._step_times.clear()
            self._memory_samples.clear()


def create_metrics_collector(max_samples: int = 1000,
                             max_events: int = 500) -> MetricsCollector:
    """Create a metrics collector."""
    return MetricsCollector(max_samples=max_samples, max_events=max_events)


def create_profiler(window_size: int = 100) -> Profiler:
    """Create a profiler."""
    return Profiler(window_size=window_size)
