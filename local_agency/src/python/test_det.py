#!/usr/bin/env python3
"""
DET Core Python Bindings Test
=============================

Tests for the Python-C bridge.
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from det import DETCore, DETDecision, DETEmotion, DETLayer


def test_create_destroy():
    """Test basic lifecycle."""
    print("  test_create_destroy...", end=" ")
    with DETCore() as core:
        assert core.tick == 0
        assert core.num_nodes > 0
        assert core.num_active > 0
    print("PASS")


def test_step():
    """Test simulation stepping."""
    print("  test_step...", end=" ")
    with DETCore() as core:
        initial_tick = core.tick
        core.step(0.1)
        assert core.tick == initial_tick + 1

        # Run multiple steps
        for _ in range(10):
            core.step(0.1)
        assert core.tick == initial_tick + 11
    print("PASS")


def test_aggregates():
    """Test aggregate queries."""
    print("  test_aggregates...", end=" ")
    with DETCore() as core:
        for _ in range(5):
            core.step(0.1)

        presence, coherence, resource, debt = core.get_aggregates()
        assert 0.0 <= presence <= 1.0
        assert 0.0 <= coherence <= 1.0
        assert resource >= 0.0
        assert 0.0 <= debt <= 1.0
    print("PASS")


def test_self_affect():
    """Test Self affect queries."""
    print("  test_self_affect...", end=" ")
    with DETCore() as core:
        for _ in range(5):
            core.step(0.1)

        valence, arousal, bondedness = core.get_self_affect()
        assert -1.0 <= valence <= 1.0
        assert 0.0 <= arousal <= 1.0
        assert 0.0 <= bondedness <= 1.0
    print("PASS")


def test_emotion():
    """Test emotional state."""
    print("  test_emotion...", end=" ")
    with DETCore() as core:
        for _ in range(5):
            core.step(0.1)

        emotion = core.get_emotion()
        assert isinstance(emotion, DETEmotion)

        emotion_str = core.get_emotion_string()
        assert len(emotion_str) > 0
    print("PASS")


def test_gatekeeper():
    """Test gatekeeper evaluation."""
    print("  test_gatekeeper...", end=" ")
    with DETCore() as core:
        # Run a few steps to establish state
        for _ in range(10):
            core.step(0.1)

        decision = core.evaluate_request([1, 2, 3], target_domain=0)
        assert isinstance(decision, DETDecision)
    print("PASS")


def test_stimulus_injection():
    """Test port stimulus injection."""
    print("  test_stimulus_injection...", end=" ")
    with DETCore() as core:
        # Get initial state
        port = core.get_port(0)
        node = core.get_node(port.node_id)
        initial_F = node.F

        # Inject stimulus
        core.inject_stimulus([0], [2.0])

        node = core.get_node(port.node_id)
        assert node.F > initial_F
    print("PASS")


def test_node_bond_access():
    """Test node and bond access."""
    print("  test_node_bond_access...", end=" ")
    with DETCore() as core:
        # Access P-layer node
        node = core.get_node(0)
        assert node.layer == DETLayer.P
        assert node.active

        # Access bond
        if core.num_bonds > 0:
            bond = core.get_bond(0)
            assert bond.C >= 0.0
    print("PASS")


def test_recruit_retire():
    """Test node recruitment and retirement."""
    print("  test_recruit_retire...", end=" ")
    with DETCore() as core:
        initial_active = core.num_active

        # Recruit a node
        node_id = core.recruit_node(DETLayer.A)
        if node_id >= 0:
            assert core.num_active == initial_active + 1

            node = core.get_node(node_id)
            assert node.active
            assert node.layer == DETLayer.A

            # Retire the node
            core.retire_node(node_id)
            assert core.num_active == initial_active
    print("PASS")


def test_inspect():
    """Test state inspection."""
    print("  test_inspect...", end=" ")
    with DETCore() as core:
        for _ in range(5):
            core.step(0.1)

        state = core.inspect()

        assert "tick" in state
        assert "emotion" in state
        assert "aggregates" in state
        assert "self_affect" in state
        assert "counts" in state
        assert "layers" in state

        assert state["layers"]["P"] == 16
        assert state["layers"]["A"] == 256
    print("PASS")


def test_repr():
    """Test string representation."""
    print("  test_repr...", end=" ")
    with DETCore() as core:
        s = repr(core)
        assert "DETCore" in s
        assert "tick=" in s
    print("PASS")


def run_tests():
    """Run all tests."""
    print("\n========================================")
    print("  DET Python Bindings Test Suite")
    print("========================================\n")

    tests = [
        test_create_destroy,
        test_step,
        test_aggregates,
        test_self_affect,
        test_emotion,
        test_gatekeeper,
        test_stimulus_injection,
        test_node_bond_access,
        test_recruit_retire,
        test_inspect,
        test_repr,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print("\n========================================")
    print(f"  Results: {passed}/{passed + failed} tests passed")
    print("========================================\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
