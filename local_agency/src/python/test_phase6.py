#!/usr/bin/env python3
"""
DET Phase 6 Tests
=================

Phase 6.1: Test harness for DET debugging and probing.
Phase 6.2: Web application for 3D visualization and real-time monitoring.
Phase 6.3: Advanced interactive probing (escalation, grace, domains, gatekeeper).
Phase 6.4: Metrics and logging (dashboard, event log, timeline, profiling).
"""

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det import (
    DETCore,
    HarnessController, HarnessCLI, HarnessEvent, HarnessEventType,
    Snapshot, create_harness, run_harness_cli,
    WEBAPP_AVAILABLE,
    # Phase 6.4: Metrics
    MetricsCollector, MetricsSample, DETEvent, DETEventType, Profiler,
    create_metrics_collector, create_profiler,
)

# Conditional webapp imports
if WEBAPP_AVAILABLE:
    from det import create_app, run_server, DETStateAPI
    from det.webapp.api import DETStateAPI as APIClass


# ============================================================================
# Harness Controller Tests
# ============================================================================

def test_harness_controller_init():
    """Test HarnessController initialization."""
    print("  test_harness_controller_init...", end=" ")

    controller = HarnessController()

    assert controller.core is None
    assert controller.paused is False
    assert controller.speed == 1.0

    print("PASS")


def test_harness_controller_with_core():
    """Test HarnessController with DET core."""
    print("  test_harness_controller_with_core...", end=" ")

    with DETCore() as core:
        controller = HarnessController(core=core)

        assert controller.core is core
        assert controller.paused is False

    print("PASS")


def test_harness_create_convenience():
    """Test create_harness convenience function."""
    print("  test_harness_create_convenience...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        assert controller.core is core
        assert isinstance(controller, HarnessController)

    print("PASS")


# ============================================================================
# Resource Injection Tests
# ============================================================================

def test_inject_f():
    """Test resource F injection."""
    print("  test_inject_f...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        # Run a few steps to initialize
        for _ in range(5):
            core.step(0.1)

        # Get initial state
        initial_state = controller.get_node_state(0)
        initial_f = initial_state["F"]

        # Inject F
        result = controller.inject_f(0, 0.5)

        assert result is True

        new_state = controller.get_node_state(0)
        assert new_state["F"] >= initial_f  # Should have increased

    print("PASS")


def test_inject_q():
    """Test structural debt q injection."""
    print("  test_inject_q...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(5):
            core.step(0.1)

        initial_state = controller.get_node_state(0)
        initial_q = initial_state["q"]

        result = controller.inject_q(0, 0.3)

        assert result is True

        new_state = controller.get_node_state(0)
        assert new_state["q"] >= initial_q

    print("PASS")


def test_set_agency():
    """Test agency setting."""
    print("  test_set_agency...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(5):
            core.step(0.1)

        result = controller.set_agency(0, 0.8)

        assert result is True

        state = controller.get_node_state(0)
        assert abs(state["a"] - 0.8) < 0.01

    print("PASS")


def test_inject_f_all():
    """Test F injection to all nodes."""
    print("  test_inject_f_all...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(5):
            core.step(0.1)

        count = controller.inject_f_all(0.1)

        assert count > 0
        assert count <= core.num_active

    print("PASS")


def test_inject_f_layer():
    """Test F injection to specific layer."""
    print("  test_inject_f_layer...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(5):
            core.step(0.1)

        p_count = controller.inject_f_all(0.1, layer="P")
        assert p_count <= 16  # P-layer size

    print("PASS")


def test_inject_invalid_node():
    """Test injection to invalid node index."""
    print("  test_inject_invalid_node...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        result = controller.inject_f(9999, 0.5)
        assert result is False

        result = controller.inject_f(-1, 0.5)
        assert result is False

    print("PASS")


# ============================================================================
# Bond Manipulation Tests
# ============================================================================

def test_create_bond():
    """Test bond creation."""
    print("  test_create_bond...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(5):
            core.step(0.1)

        # Try to create a new bond
        result = controller.create_bond(0, 5, 0.6)

        # May or may not succeed depending on existing bonds
        # Just verify the method works without error
        assert isinstance(result, bool)

    print("PASS")


def test_set_coherence():
    """Test bond coherence setting."""
    print("  test_set_coherence...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        # Find an existing bond
        if core.num_bonds > 0:
            bond = core._core.contents.bonds[0]
            i, j = bond.i, bond.j

            result = controller.set_coherence(i, j, 0.9)

            if result:
                state = controller.get_bond_state(i, j)
                assert state is not None
                assert abs(state["coherence"] - 0.9) < 0.01

    print("PASS")


def test_destroy_bond():
    """Test bond destruction."""
    print("  test_destroy_bond...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        # Find an existing bond
        if core.num_bonds > 0:
            bond = core._core.contents.bonds[0]
            i, j = bond.i, bond.j

            result = controller.destroy_bond(i, j)
            # Just verify it doesn't crash
            assert isinstance(result, bool)

    print("PASS")


# ============================================================================
# Time Control Tests
# ============================================================================

def test_pause_resume():
    """Test pause and resume."""
    print("  test_pause_resume...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        assert controller.paused is False

        controller.pause()
        assert controller.paused is True

        controller.resume()
        assert controller.paused is False

    print("PASS")


def test_toggle_pause():
    """Test toggle pause."""
    print("  test_toggle_pause...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        controller.toggle_pause()
        assert controller.paused is True

        controller.toggle_pause()
        assert controller.paused is False

    print("PASS")


def test_speed_control():
    """Test speed control."""
    print("  test_speed_control...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        assert controller.speed == 1.0

        controller.set_speed(2.0)
        assert controller.speed == 2.0

        controller.set_speed(0.5)
        assert controller.speed == 0.5

    print("PASS")


def test_step():
    """Test single step execution."""
    print("  test_step...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        initial_tick = core.tick

        result = controller.step(0.1)

        assert result is True
        assert core.tick == initial_tick + 1

    print("PASS")


def test_step_n():
    """Test multiple step execution."""
    print("  test_step_n...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        initial_tick = core.tick

        count = controller.step_n(10, 0.1)

        assert count == 10
        assert core.tick == initial_tick + 10

    print("PASS")


def test_run_until():
    """Test run until condition."""
    print("  test_run_until...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        initial_tick = core.tick

        # Run until tick reaches 5
        count = controller.run_until(
            lambda: core.tick >= initial_tick + 5,
            max_steps=100,
            dt=0.1
        )

        assert count == 5
        assert core.tick >= initial_tick + 5

    print("PASS")


# ============================================================================
# State Inspection Tests
# ============================================================================

def test_get_node_state():
    """Test node state retrieval."""
    print("  test_get_node_state...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(5):
            core.step(0.1)

        state = controller.get_node_state(0)

        assert state is not None
        assert "F" in state
        assert "q" in state
        assert "a" in state
        assert "P" in state
        assert "layer" in state

    print("PASS")


def test_get_bond_state():
    """Test bond state retrieval."""
    print("  test_get_bond_state...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        # Find an existing bond
        if core.num_bonds > 0:
            bond = core._core.contents.bonds[0]
            state = controller.get_bond_state(bond.i, bond.j)

            assert state is not None
            assert "coherence" in state
            assert "pi" in state

    print("PASS")


def test_get_aggregates():
    """Test aggregate retrieval."""
    print("  test_get_aggregates...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        agg = controller.get_aggregates()

        assert "presence" in agg
        assert "coherence" in agg
        assert "resource" in agg
        assert "debt" in agg

    print("PASS")


def test_get_affect():
    """Test affect retrieval."""
    print("  test_get_affect...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        affect = controller.get_affect()

        assert "valence" in affect
        assert "arousal" in affect
        assert "bondedness" in affect

    print("PASS")


def test_get_self_cluster():
    """Test self-cluster retrieval."""
    print("  test_get_self_cluster...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(20):
            core.step(0.1)

        cluster = controller.get_self_cluster()

        assert isinstance(cluster, list)

    print("PASS")


def test_get_summary():
    """Test summary retrieval."""
    print("  test_get_summary...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        summary = controller.get_summary()

        assert "tick" in summary
        assert "num_active" in summary
        assert "aggregates" in summary
        assert "affect" in summary
        assert "harness" in summary

    print("PASS")


# ============================================================================
# Snapshot Tests
# ============================================================================

def test_take_snapshot():
    """Test snapshot taking."""
    print("  test_take_snapshot...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        result = controller.take_snapshot("test_snap")

        assert result is True

        snapshots = controller.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["name"] == "test_snap"

    print("PASS")


def test_restore_snapshot():
    """Test snapshot restore."""
    print("  test_restore_snapshot...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        controller.take_snapshot("before")
        tick_before = core.tick

        for _ in range(20):
            core.step(0.1)

        tick_after = core.tick
        assert tick_after > tick_before

        result = controller.restore_snapshot("before")
        assert result is True

        # Tick should be restored
        assert core.tick == tick_before

    print("PASS")


def test_delete_snapshot():
    """Test snapshot deletion."""
    print("  test_delete_snapshot...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        controller.take_snapshot("temp")
        assert len(controller.list_snapshots()) == 1

        result = controller.delete_snapshot("temp")
        assert result is True
        assert len(controller.list_snapshots()) == 0

    print("PASS")


# ============================================================================
# Event Log Tests
# ============================================================================

def test_event_logging():
    """Test event logging."""
    print("  test_event_logging...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        controller.step(0.1)
        controller.inject_f(0, 0.1)
        controller.pause()
        controller.resume()

        events = controller.get_events()

        assert len(events) >= 3
        event_types = [e["type"] for e in events]
        assert "step" in event_types
        assert "inject_f" in event_types
        assert "pause" in event_types

    print("PASS")


def test_clear_events():
    """Test event clearing."""
    print("  test_clear_events...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        controller.step(0.1)
        controller.step(0.1)

        assert len(controller.get_events()) >= 2

        controller.clear_events()
        assert len(controller.get_events()) == 0

    print("PASS")


def test_event_callback():
    """Test event callbacks."""
    print("  test_event_callback...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        received = []

        def callback(event):
            received.append(event)

        controller.add_event_callback(callback)
        controller.step(0.1)

        assert len(received) == 1
        assert received[0].event_type == HarnessEventType.STEP

        controller.remove_event_callback(callback)
        controller.step(0.1)

        assert len(received) == 1  # No new events

    print("PASS")


# ============================================================================
# Watcher Tests
# ============================================================================

def test_add_watcher():
    """Test watcher addition."""
    print("  test_add_watcher...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        triggered = [False]

        def condition():
            return core.tick >= 5

        def callback():
            triggered[0] = True

        controller.add_watcher("test", condition, callback)

        for _ in range(10):
            controller.step(0.1)
            controller.check_watchers()

        assert triggered[0] is True

    print("PASS")


def test_remove_watcher():
    """Test watcher removal."""
    print("  test_remove_watcher...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        controller.add_watcher("test", lambda: True, lambda: None)
        controller.remove_watcher("test")

        # Should not crash
        controller.check_watchers()

    print("PASS")


# ============================================================================
# Advanced Probing Tests (Phase 6.3)
# ============================================================================

def test_trigger_escalation():
    """Test escalation triggering."""
    print("  test_trigger_escalation...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(5):
            core.step(0.1)

        result = controller.trigger_escalation(0)
        assert result is True

        # Check node has escalation pending
        node = core._core.contents.nodes[0]
        assert node.escalation_pending is True

    print("PASS")


def test_trigger_escalation_invalid():
    """Test escalation with invalid node."""
    print("  test_trigger_escalation_invalid...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        result = controller.trigger_escalation(9999)
        assert result is False

        result = controller.trigger_escalation(-1)
        assert result is False

    print("PASS")


def test_inject_grace():
    """Test grace injection."""
    print("  test_inject_grace...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(5):
            core.step(0.1)

        result = controller.inject_grace(0, 0.5)
        assert result is True

    print("PASS")


def test_inject_grace_all():
    """Test grace injection to all needing nodes."""
    print("  test_inject_grace_all...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        count = controller.inject_grace_all(0.5)
        # Count can be 0 if no nodes need grace
        assert isinstance(count, int)
        assert count >= 0

    print("PASS")


def test_get_total_grace_needed():
    """Test total grace needed retrieval."""
    print("  test_get_total_grace_needed...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        total = controller.get_total_grace_needed()
        assert isinstance(total, float)
        assert total >= 0.0

    print("PASS")


def test_get_learning_capacity():
    """Test learning capacity retrieval."""
    print("  test_get_learning_capacity...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        capacity = controller.get_learning_capacity()
        assert isinstance(capacity, float)

    print("PASS")


def test_can_learn():
    """Test learning capability check."""
    print("  test_can_learn...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        result = controller.can_learn(0.5)
        assert isinstance(result, bool)

    print("PASS")


def test_activate_domain():
    """Test domain activation."""
    print("  test_activate_domain...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        result = controller.activate_domain("test_domain", 4, 0.3)
        # May or may not succeed depending on available dormant nodes
        assert isinstance(result, bool)

    print("PASS")


def test_transfer_pattern():
    """Test pattern transfer."""
    print("  test_transfer_pattern...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        result = controller.transfer_pattern(0, 1, 0.5)
        # May or may not succeed depending on domains
        assert isinstance(result, bool)

    print("PASS")


def test_evaluate_request():
    """Test gatekeeper request evaluation."""
    print("  test_evaluate_request...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        for _ in range(10):
            core.step(0.1)

        decision = controller.evaluate_request([1, 2, 3])
        assert decision in ["PROCEED", "RETRY", "STOP", "ESCALATE"]

    print("PASS")


def test_advanced_probing_without_core():
    """Test advanced probing methods without core."""
    print("  test_advanced_probing_without_core...", end=" ")

    controller = create_harness()

    assert controller.trigger_escalation(0) is False
    assert controller.inject_grace(0, 0.5) is False
    assert controller.inject_grace_all(0.5) == 0
    assert controller.get_total_grace_needed() == 0.0
    assert controller.get_learning_capacity() == 0.0
    assert controller.can_learn(0.5) is False
    assert controller.activate_domain("test", 4) is False
    assert controller.transfer_pattern(0, 1) is False
    assert controller.evaluate_request([1, 2, 3]) == "STOP"

    print("PASS")


# ============================================================================
# CLI Tests
# ============================================================================

def test_harness_cli_init():
    """Test HarnessCLI initialization."""
    print("  test_harness_cli_init...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)
        cli = HarnessCLI(controller)

        assert cli.controller is controller

    print("PASS")


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_harness_flow():
    """Test full harness workflow."""
    print("  test_full_harness_flow...", end=" ")

    with DETCore() as core:
        controller = create_harness(core=core)

        # Run simulation
        controller.step_n(10, 0.1)

        # Take snapshot
        controller.take_snapshot("checkpoint1")

        # Inject resources
        controller.inject_f_all(0.1)

        # Modify state
        controller.set_agency(0, 0.7)

        # Run more
        controller.step_n(10, 0.1)

        # Check state
        summary = controller.get_summary()
        assert summary["tick"] == 20

        # Restore
        controller.restore_snapshot("checkpoint1")
        assert core.tick == 10

        # Check events
        events = controller.get_events()
        assert len(events) > 0

    print("PASS")


def test_harness_with_storage():
    """Test harness with storage path."""
    print("  test_harness_with_storage...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        with DETCore() as core:
            storage = Path(tmpdir) / "harness"
            controller = create_harness(core=core, storage_path=storage)

            assert storage.exists()

            # Generate some events
            controller.step_n(5, 0.1)

            # Save events
            events_file = storage / "events.json"
            controller.save_events(events_file)

            assert events_file.exists()

    print("PASS")


# ============================================================================
# Phase 6.2: Web Application Tests
# ============================================================================

def test_webapp_available():
    """Test that webapp is available."""
    print("  test_webapp_available...", end=" ")

    assert WEBAPP_AVAILABLE is True

    print("PASS")


def test_state_api_init():
    """Test DETStateAPI initialization."""
    print("  test_state_api_init...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    api = APIClass()
    assert api.core is None
    assert api.harness is None

    print("PASS")


def test_state_api_with_core():
    """Test DETStateAPI with DET core."""
    print("  test_state_api_with_core...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        api = APIClass(core=core, harness=harness)

        assert api.core is core
        assert api.harness is harness

    print("PASS")


def test_state_api_get_status():
    """Test status retrieval."""
    print("  test_state_api_get_status...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        api = APIClass(core=core, harness=harness)

        for _ in range(5):
            core.step(0.1)

        status = api.get_status()

        assert status["connected"] is True
        assert "tick" in status
        assert "num_active" in status

    print("PASS")


def test_state_api_get_nodes():
    """Test nodes retrieval."""
    print("  test_state_api_get_nodes...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        api = APIClass(core=core, harness=harness)

        for _ in range(5):
            core.step(0.1)

        nodes = api.get_nodes()

        assert len(nodes) > 0
        assert "index" in nodes[0]
        assert "layer" in nodes[0]
        assert "F" in nodes[0]

    print("PASS")


def test_state_api_get_bonds():
    """Test bonds retrieval."""
    print("  test_state_api_get_bonds...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        api = APIClass(core=core, harness=harness)

        for _ in range(10):
            core.step(0.1)

        bonds = api.get_bonds()

        assert isinstance(bonds, list)

    print("PASS")


def test_state_api_get_visualization_data():
    """Test visualization data retrieval."""
    print("  test_state_api_get_visualization_data...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        api = APIClass(core=core, harness=harness)

        for _ in range(10):
            core.step(0.1)

        data = api.get_visualization_data()

        assert "nodes" in data
        assert "bonds" in data
        assert "self_cluster" in data

        if len(data["nodes"]) > 0:
            node = data["nodes"][0]
            assert "x" in node
            assert "y" in node
            assert "z" in node
            assert "color" in node

    print("PASS")


def test_state_api_control():
    """Test control methods."""
    print("  test_state_api_control...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        api = APIClass(core=core, harness=harness)

        # Test step
        result = api.step(5, 0.1)
        assert result["steps"] == 5

        # Test pause
        assert api.pause() is True
        assert harness.paused is True

        # Test resume
        assert api.resume() is True
        assert harness.paused is False

        # Test speed
        speed = api.set_speed(2.0)
        assert speed == 2.0

    print("PASS")


def test_state_api_inject():
    """Test injection methods."""
    print("  test_state_api_inject...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        api = APIClass(core=core, harness=harness)

        for _ in range(5):
            core.step(0.1)

        # Test inject_f
        result = api.inject_f(0, 0.5)
        assert result is True

        # Test inject_q
        result = api.inject_q(0, 0.1)
        assert result is True

    print("PASS")


def test_create_app():
    """Test create_app function."""
    print("  test_create_app...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        app = create_app(core=core, harness=harness)

        assert app is not None
        assert hasattr(app, "routes")

    print("PASS")


def test_webapp_routes():
    """Test that webapp has expected routes."""
    print("  test_webapp_routes...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        app = create_app(core=core, harness=harness)

        routes = [route.path for route in app.routes]

        # Check API routes exist
        assert "/" in routes
        assert "/api/status" in routes
        assert "/api/nodes" in routes
        assert "/ws" in routes

    print("PASS")


# ============================================================================
# Phase 6.4: Metrics and Logging Tests
# ============================================================================

def test_metrics_collector_init():
    """Test MetricsCollector initialization."""
    print("  test_metrics_collector_init...", end=" ")

    collector = create_metrics_collector()

    assert collector is not None
    assert len(collector.get_samples()) == 0
    assert len(collector.get_events()) == 0

    print("PASS")


def test_metrics_sample():
    """Test metric sampling."""
    print("  test_metrics_sample...", end=" ")

    with DETCore() as core:
        collector = create_metrics_collector()

        for _ in range(10):
            core.step(0.1)

        sample = collector.sample(core)

        assert sample is not None
        assert sample.tick == core.tick
        assert isinstance(sample.presence, float)
        assert isinstance(sample.coherence, float)
        assert isinstance(sample.valence, float)
        assert isinstance(sample.arousal, float)
        assert isinstance(sample.bondedness, float)

    print("PASS")


def test_metrics_multiple_samples():
    """Test multiple metric samples."""
    print("  test_metrics_multiple_samples...", end=" ")

    with DETCore() as core:
        collector = create_metrics_collector()

        for _ in range(10):
            core.step(0.1)
            collector.sample(core)

        samples = collector.get_samples()
        assert len(samples) == 10

    print("PASS")


def test_metrics_event_logging():
    """Test DET event logging."""
    print("  test_metrics_event_logging...", end=" ")

    collector = create_metrics_collector()

    collector.log_event(DETEventType.ESCALATION, tick=1, node=5)
    collector.log_event(DETEventType.COMPILATION, tick=2, node=10)
    collector.log_event(DETEventType.BOND_FORMED, tick=3, node_i=1, node_j=5)

    events = collector.get_events()
    assert len(events) == 3

    assert events[0]["type"] == "escalation"
    assert events[1]["type"] == "compilation"
    assert events[2]["type"] == "bond_formed"

    print("PASS")


def test_metrics_event_filter():
    """Test event filtering by type."""
    print("  test_metrics_event_filter...", end=" ")

    collector = create_metrics_collector()

    collector.log_event(DETEventType.ESCALATION, tick=1)
    collector.log_event(DETEventType.COMPILATION, tick=2)
    collector.log_event(DETEventType.ESCALATION, tick=3)

    events = collector.get_events(event_type=DETEventType.ESCALATION)
    assert len(events) == 2

    print("PASS")


def test_metrics_event_callback():
    """Test event callbacks."""
    print("  test_metrics_event_callback...", end=" ")

    collector = create_metrics_collector()
    received = []

    def callback(event):
        received.append(event)

    collector.add_event_callback(callback)
    collector.log_event(DETEventType.ESCALATION, tick=1)

    assert len(received) == 1
    assert received[0].event_type == DETEventType.ESCALATION

    collector.remove_event_callback(callback)
    collector.log_event(DETEventType.ESCALATION, tick=2)

    assert len(received) == 1  # No new events

    print("PASS")


def test_metrics_timeline():
    """Test timeline data retrieval."""
    print("  test_metrics_timeline...", end=" ")

    with DETCore() as core:
        collector = create_metrics_collector()

        for _ in range(10):
            core.step(0.1)
            collector.sample(core)

        timeline = collector.get_timeline("valence")
        assert len(timeline) == 10

        for point in timeline:
            assert "timestamp" in point
            assert "tick" in point
            assert "value" in point

    print("PASS")


def test_metrics_dashboard():
    """Test dashboard data retrieval."""
    print("  test_metrics_dashboard...", end=" ")

    with DETCore() as core:
        collector = create_metrics_collector()

        for _ in range(25):
            core.step(0.1)
            collector.sample(core)

        collector.log_event(DETEventType.ESCALATION, tick=10)

        dashboard = collector.get_dashboard()

        assert "current" in dashboard
        assert "trends" in dashboard
        assert "events_count" in dashboard
        assert "recent_events" in dashboard

        assert dashboard["current"] is not None
        assert dashboard["events_count"] == 1

    print("PASS")


def test_metrics_statistics():
    """Test statistical summary."""
    print("  test_metrics_statistics...", end=" ")

    with DETCore() as core:
        collector = create_metrics_collector()

        for _ in range(20):
            core.step(0.1)
            collector.sample(core)

        stats = collector.get_statistics()

        assert "sample_count" in stats
        assert stats["sample_count"] == 20

        for field in ["presence", "coherence", "valence"]:
            assert field in stats
            assert "min" in stats[field]
            assert "max" in stats[field]
            assert "mean" in stats[field]

    print("PASS")


def test_profiler_init():
    """Test Profiler initialization."""
    print("  test_profiler_init...", end=" ")

    profiler = create_profiler()

    assert profiler is not None

    stats = profiler.get_tick_stats()
    assert stats["count"] == 0

    print("PASS")


def test_profiler_tick_timing():
    """Test tick timing."""
    print("  test_profiler_tick_timing...", end=" ")

    profiler = create_profiler()

    for _ in range(10):
        profiler.start_tick()
        time.sleep(0.001)  # Small delay
        profiler.end_tick()

    stats = profiler.get_tick_stats()

    assert stats["count"] == 10
    assert stats["avg_ms"] > 0
    assert stats["min_ms"] > 0
    assert stats["max_ms"] >= stats["min_ms"]

    print("PASS")


def test_profiler_step_timing():
    """Test step timing within ticks."""
    print("  test_profiler_step_timing...", end=" ")

    profiler = create_profiler()

    profiler.start_tick()
    profiler.start_step("dynamics")
    time.sleep(0.001)
    profiler.end_step()
    profiler.start_step("bonds")
    time.sleep(0.001)
    profiler.end_step()
    profiler.end_tick()

    step_stats = profiler.get_step_stats()

    assert "dynamics" in step_stats
    assert "bonds" in step_stats
    assert step_stats["dynamics"]["count"] == 1

    print("PASS")


def test_profiler_memory():
    """Test memory sampling."""
    print("  test_profiler_memory...", end=" ")

    profiler = create_profiler()

    profiler.sample_memory()

    stats = profiler.get_memory_stats()

    assert stats["count"] == 1
    assert stats["current_mb"] > 0

    print("PASS")


def test_profiler_report():
    """Test full profiling report."""
    print("  test_profiler_report...", end=" ")

    profiler = create_profiler()

    for _ in range(5):
        profiler.start_tick()
        profiler.start_step("test")
        profiler.end_step()
        profiler.end_tick()

    profiler.sample_memory()

    report = profiler.get_report()

    assert "tick" in report
    assert "steps" in report
    assert "memory" in report

    assert report["tick"]["count"] == 5

    print("PASS")


def test_metrics_clear():
    """Test clearing metrics data."""
    print("  test_metrics_clear...", end=" ")

    with DETCore() as core:
        collector = create_metrics_collector()

        for _ in range(5):
            core.step(0.1)
            collector.sample(core)

        collector.log_event(DETEventType.ESCALATION, tick=1)

        assert len(collector.get_samples()) == 5
        assert len(collector.get_events()) == 1

        collector.clear()

        assert len(collector.get_samples()) == 0
        assert len(collector.get_events()) == 0

    print("PASS")


def test_det_event_to_dict():
    """Test DETEvent serialization."""
    print("  test_det_event_to_dict...", end=" ")

    event = DETEvent(
        event_type=DETEventType.ESCALATION,
        timestamp=time.time(),
        tick=42,
        details={"node": 5, "novelty": 0.8},
    )

    d = event.to_dict()

    assert d["type"] == "escalation"
    assert d["tick"] == 42
    assert d["details"]["node"] == 5
    assert "time_str" in d

    print("PASS")


def test_webapp_metrics_routes():
    """Test that webapp has metrics routes."""
    print("  test_webapp_metrics_routes...", end=" ")

    if not WEBAPP_AVAILABLE:
        print("SKIP (webapp not available)")
        return

    with DETCore() as core:
        harness = create_harness(core=core)
        app = create_app(core=core, harness=harness)

        routes = [route.path for route in app.routes]

        # Check metrics routes exist
        assert "/api/metrics/dashboard" in routes
        assert "/api/metrics/samples" in routes
        assert "/api/metrics/statistics" in routes
        assert "/api/metrics/profiling" in routes

    print("PASS")


# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run all Phase 6 tests."""
    print("\n========================================")
    print("  DET Phase 6 Tests")
    print("========================================\n")

    tests = [
        # Harness Controller
        test_harness_controller_init,
        test_harness_controller_with_core,
        test_harness_create_convenience,

        # Resource Injection
        test_inject_f,
        test_inject_q,
        test_set_agency,
        test_inject_f_all,
        test_inject_f_layer,
        test_inject_invalid_node,

        # Bond Manipulation
        test_create_bond,
        test_set_coherence,
        test_destroy_bond,

        # Time Control
        test_pause_resume,
        test_toggle_pause,
        test_speed_control,
        test_step,
        test_step_n,
        test_run_until,

        # State Inspection
        test_get_node_state,
        test_get_bond_state,
        test_get_aggregates,
        test_get_affect,
        test_get_self_cluster,
        test_get_summary,

        # Snapshots
        test_take_snapshot,
        test_restore_snapshot,
        test_delete_snapshot,

        # Event Log
        test_event_logging,
        test_clear_events,
        test_event_callback,

        # Watchers
        test_add_watcher,
        test_remove_watcher,

        # Advanced Probing (Phase 6.3)
        test_trigger_escalation,
        test_trigger_escalation_invalid,
        test_inject_grace,
        test_inject_grace_all,
        test_get_total_grace_needed,
        test_get_learning_capacity,
        test_can_learn,
        test_activate_domain,
        test_transfer_pattern,
        test_evaluate_request,
        test_advanced_probing_without_core,

        # CLI
        test_harness_cli_init,

        # Integration
        test_full_harness_flow,
        test_harness_with_storage,

        # Phase 6.2: Webapp
        test_webapp_available,
        test_state_api_init,
        test_state_api_with_core,
        test_state_api_get_status,
        test_state_api_get_nodes,
        test_state_api_get_bonds,
        test_state_api_get_visualization_data,
        test_state_api_control,
        test_state_api_inject,
        test_create_app,
        test_webapp_routes,

        # Phase 6.4: Metrics and Logging
        test_metrics_collector_init,
        test_metrics_sample,
        test_metrics_multiple_samples,
        test_metrics_event_logging,
        test_metrics_event_filter,
        test_metrics_event_callback,
        test_metrics_timeline,
        test_metrics_dashboard,
        test_metrics_statistics,
        test_profiler_init,
        test_profiler_tick_timing,
        test_profiler_step_timing,
        test_profiler_memory,
        test_profiler_report,
        test_metrics_clear,
        test_det_event_to_dict,
        test_webapp_metrics_routes,
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
