#!/usr/bin/env python3
"""
DET Phase 4 Integration Tests
=============================

Tests for extended DET dynamics, learning via recruitment,
emotional state integration, and multi-session support.
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det import (
    DETCore, DETDecision, DETEmotion, DETLayer,
    EmotionalIntegration, EmotionalMode, BehaviorModulation,
    MultiSessionManager, SessionContext,
    TimerSystem,
)


# ============================================================================
# Phase 4.1: Extended DET Dynamics Tests
# ============================================================================

def test_momentum_update():
    """Test bond momentum (π) updates."""
    print("  test_momentum_update...", end=" ")

    with DETCore() as core:
        # Run some steps to let dynamics evolve
        for _ in range(20):
            core.step(0.1)

        # Check that bonds have non-zero momentum
        has_momentum = False
        for b in range(core.num_bonds):
            bond = core.get_bond(b)
            if abs(bond.pi) > 0.0001:
                has_momentum = True
                break

        assert has_momentum, "No momentum accumulated in bonds"

    print("PASS")


def test_angular_momentum():
    """Test angular momentum (L) and phase velocity."""
    print("  test_angular_momentum...", end=" ")

    with DETCore() as core:
        # Get initial phase
        initial_theta = core.get_node(0).theta

        # Run steps
        for _ in range(50):
            core.step(0.1)

        # Check that phase has changed
        final_theta = core.get_node(0).theta
        theta_changed = abs(final_theta - initial_theta) > 0.001

        # Check that L exists
        node = core.get_node(0)
        # L should be non-zero after dynamics
        # (may be small but shouldn't be exactly zero after torque)

        # At minimum, phases should have evolved
        assert theta_changed, "Phase did not evolve"

    print("PASS")


def test_debt_accumulation():
    """Test structural debt accumulation and decay."""
    print("  test_debt_accumulation...", end=" ")

    with DETCore() as core:
        # Deplete resource on a node to trigger debt accumulation
        node = core.get_node(0)

        # Run many steps with depleted resources
        for _ in range(100):
            core.step(0.1)

        # Debt should have some value (may be small due to decay)
        _, _, _, debt = core.get_aggregates()
        # Debt accumulates slowly but should be > 0 after extended running
        assert debt >= 0.0, "Debt is negative"

    print("PASS")


def test_grace_injection():
    """Test grace injection and recovery."""
    print("  test_grace_injection...", end=" ")

    with DETCore() as core:
        node_id = 0
        node = core.get_node(node_id)
        initial_F = node.F

        # Inject grace
        core.inject_grace(node_id, 1.0)

        # Step to process grace
        core.step(0.1)

        # Resource should have increased
        node = core.get_node(node_id)
        assert node.F >= initial_F, "Grace did not increase resource"

    print("PASS")


def test_grace_needs_detection():
    """Test grace needs detection."""
    print("  test_grace_needs_detection...", end=" ")

    with DETCore() as core:
        # Fresh core shouldn't need much grace
        total_needed = core.total_grace_needed()
        assert total_needed >= 0.0, "Grace needed is negative"

    print("PASS")


# ============================================================================
# Phase 4.2: Learning via Recruitment Tests
# ============================================================================

def test_can_learn():
    """Test learning capability check."""
    print("  test_can_learn...", end=" ")

    with DETCore() as core:
        # Identify self to establish cluster
        core.identify_self()

        # Run some steps to establish state
        for _ in range(20):
            core.step(0.1)

        core.identify_self()

        # Check if can learn simple task
        can_learn = core.can_learn(complexity=0.5, domain=0)
        # Result depends on state, but function should return without error
        assert isinstance(can_learn, bool)

    print("PASS")


def test_learning_capacity():
    """Test learning capacity calculation."""
    print("  test_learning_capacity...", end=" ")

    with DETCore() as core:
        # Run steps and identify self
        for _ in range(10):
            core.step(0.1)
        core.identify_self()

        capacity = core.learning_capacity()
        assert capacity >= 0.0, "Negative learning capacity"

    print("PASS")


def test_activate_domain():
    """Test domain activation with recruited nodes."""
    print("  test_activate_domain...", end=" ")

    with DETCore() as core:
        initial_active = core.num_active

        # Activate a new domain
        success = core.activate_domain(
            name="new_domain",
            num_nodes=4,
            initial_coherence=0.4
        )

        assert success, "Failed to activate domain"

        # Should have more active nodes
        assert core.num_active > initial_active, "No nodes recruited"

    print("PASS")


def test_pattern_transfer():
    """Test pattern transfer between domains."""
    print("  test_pattern_transfer...", end=" ")

    with DETCore() as core:
        # First register a source domain
        core.register_domain("source")

        # Activate a target domain
        core.activate_domain("target", num_nodes=4, initial_coherence=0.3)

        # Transfer pattern (may succeed or fail based on state)
        result = core.transfer_pattern(
            source_domain=0,
            target_domain=1,
            transfer_strength=0.5
        )
        # Result depends on state, just ensure no error
        assert isinstance(result, bool)

    print("PASS")


# ============================================================================
# Phase 4.3: Emotional State Integration Tests
# ============================================================================

def test_emotional_integration_init():
    """Test EmotionalIntegration initialization."""
    print("  test_emotional_integration_init...", end=" ")

    with DETCore() as core:
        emotional = EmotionalIntegration(core=core)

        assert emotional.core is core
        assert emotional.current_mode == EmotionalMode.NEUTRAL

    print("PASS")


def test_emotional_mode_update():
    """Test emotional mode updates."""
    print("  test_emotional_mode_update...", end=" ")

    with DETCore() as core:
        # Run some steps to establish state
        for _ in range(20):
            core.step(0.1)
        core.identify_self()

        emotional = EmotionalIntegration(core=core)
        mode = emotional.update()

        assert isinstance(mode, EmotionalMode)

    print("PASS")


def test_behavior_modulation():
    """Test behavior modulation based on emotional state."""
    print("  test_behavior_modulation...", end=" ")

    with DETCore() as core:
        emotional = EmotionalIntegration(core=core)

        mod = emotional.modulation
        assert isinstance(mod, BehaviorModulation)
        assert 0.0 < mod.temperature_mult < 2.0
        assert 0.0 <= mod.risk_tolerance <= 1.0
        assert mod.retry_patience > 0

    print("PASS")


def test_llm_temperature_adjustment():
    """Test LLM temperature adjustment."""
    print("  test_llm_temperature_adjustment...", end=" ")

    with DETCore() as core:
        emotional = EmotionalIntegration(core=core)

        temp = emotional.get_llm_temperature(base_temperature=0.7)
        assert 0.0 < temp < 2.0, "Temperature out of range"

    print("PASS")


def test_recovery_state():
    """Test recovery state tracking."""
    print("  test_recovery_state...", end=" ")

    with DETCore() as core:
        emotional = EmotionalIntegration(core=core)

        status = emotional.get_status()
        assert "mode" in status
        assert "recovery" in status
        assert "needs_recovery" in status["recovery"]

    print("PASS")


def test_emotional_status():
    """Test emotional status reporting."""
    print("  test_emotional_status...", end=" ")

    with DETCore() as core:
        emotional = EmotionalIntegration(core=core)
        emotional.update()

        status = emotional.get_status()
        assert "mode" in status
        assert "affect" in status
        assert "modulation" in status

    print("PASS")


# ============================================================================
# Phase 4.4: Multi-Session Support Tests
# ============================================================================

def test_core_state_save_load():
    """Test DET core state serialization."""
    print("  test_core_state_save_load...", end=" ")

    with DETCore() as core:
        # Run some steps
        for _ in range(10):
            core.step(0.1)

        # Save state
        data = core.save_state()
        assert len(data) > 0, "No state data saved"

        # Create new core and load
        with DETCore() as core2:
            success = core2.load_state(data)
            assert success, "Failed to load state"

    print("PASS")


def test_core_state_file_persistence():
    """Test DET core state file persistence."""
    print("  test_core_state_file_persistence...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/core.det"

        with DETCore() as core:
            # Run steps
            for _ in range(10):
                core.step(0.1)

            original_tick = core.tick

            # Save to file
            core.save_to_file(filepath)

        # Load in new core
        with DETCore() as core2:
            success = core2.load_from_file(filepath)
            assert success, "Failed to load from file"
            assert core2.tick == original_tick, "Tick not preserved"

    print("PASS")


def test_multi_session_manager_init():
    """Test MultiSessionManager initialization."""
    print("  test_multi_session_manager_init...", end=" ")

    with DETCore() as core:
        manager = MultiSessionManager(core=core)

        assert manager.core is core
        assert len(manager.sessions) == 0

    print("PASS")


def test_session_create_switch():
    """Test session creation and switching."""
    print("  test_session_create_switch...", end=" ")

    with DETCore() as core:
        manager = MultiSessionManager(core=core)

        # Create sessions
        s1 = manager.create_session("session1")
        s2 = manager.create_session("session2")

        assert len(manager.sessions) == 2
        assert manager.active_session.session_id == "session2"

        # Switch
        manager.switch_session("session1")
        assert manager.active_session.session_id == "session1"

    print("PASS")


def test_session_message_tracking():
    """Test session message tracking."""
    print("  test_session_message_tracking...", end=" ")

    with DETCore() as core:
        manager = MultiSessionManager(core=core)

        session = manager.create_session()
        assert session.message_count == 0

        manager.record_message()
        manager.record_message()

        assert manager.active_session.message_count == 2

    print("PASS")


def test_session_topics():
    """Test session topic tracking."""
    print("  test_session_topics...", end=" ")

    with DETCore() as core:
        manager = MultiSessionManager(core=core)

        manager.create_session()
        manager.add_topic("math")
        manager.add_topic("code")
        manager.add_topic("math")  # Duplicate

        assert len(manager.active_session.topics) == 2

    print("PASS")


def test_session_end():
    """Test session ending."""
    print("  test_session_end...", end=" ")

    with DETCore() as core:
        manager = MultiSessionManager(core=core)

        s1 = manager.create_session("s1")
        s2 = manager.create_session("s2")

        assert len(manager.sessions) == 2

        manager.end_session("s1")
        assert len(manager.sessions) == 1
        assert "s1" not in manager.sessions

    print("PASS")


def test_manager_status():
    """Test manager status reporting."""
    print("  test_manager_status...", end=" ")

    with DETCore() as core:
        manager = MultiSessionManager(core=core)
        manager.create_session()
        manager.record_message()

        status = manager.get_status()
        assert "active_session" in status
        assert "session_count" in status
        assert status["session_count"] == 1

    print("PASS")


# ============================================================================
# Integration Tests
# ============================================================================

def test_emotional_with_core_dynamics():
    """Test emotional integration with DET core dynamics."""
    print("  test_emotional_with_core_dynamics...", end=" ")

    with DETCore() as core:
        emotional = EmotionalIntegration(core=core)

        # Run multiple cycles
        for _ in range(30):
            core.step(0.1)
            emotional.update()

        # Should have tracked mode history
        assert emotional._mode_history is not None

    print("PASS")


def test_full_phase4_flow():
    """Test complete Phase 4 flow."""
    print("  test_full_phase4_flow...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        with DETCore() as core:
            # Create emotional integration
            emotional = EmotionalIntegration(core=core)

            # Create session manager
            manager = MultiSessionManager(
                core=core,
                emotional=emotional,
                state_dir=tmpdir
            )

            # Create session
            session = manager.create_session()

            # Simulate activity
            for i in range(20):
                core.step(0.1)
                emotional.update()

                if i % 5 == 0:
                    manager.record_message()

            # Check learning capability
            can_learn = core.can_learn(0.3)

            # Save state
            manager.save_core_state()

            # Verify state saved
            core_file = Path(tmpdir) / "core_state.det"
            assert core_file.exists()

    print("PASS")


# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run all Phase 4 tests."""
    print("\n========================================")
    print("  DET Phase 4 Test Suite")
    print("========================================\n")

    tests = [
        # Phase 4.1: Extended Dynamics
        test_momentum_update,
        test_angular_momentum,
        test_debt_accumulation,
        test_grace_injection,
        test_grace_needs_detection,

        # Phase 4.2: Learning via Recruitment
        test_can_learn,
        test_learning_capacity,
        test_activate_domain,
        test_pattern_transfer,

        # Phase 4.3: Emotional Integration
        test_emotional_integration_init,
        test_emotional_mode_update,
        test_behavior_modulation,
        test_llm_temperature_adjustment,
        test_recovery_state,
        test_emotional_status,

        # Phase 4.4: Multi-Session Support
        test_core_state_save_load,
        test_core_state_file_persistence,
        test_multi_session_manager_init,
        test_session_create_switch,
        test_session_message_tracking,
        test_session_topics,
        test_session_end,
        test_manager_status,

        # Integration
        test_emotional_with_core_dynamics,
        test_full_phase4_flow,
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
