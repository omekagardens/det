#!/usr/bin/env python3
"""
DET Phase 5 Tests
=================

Phase 5.1: Multi-LLM routing with domain-specialized models.
Phase 5.2: Sleep/consolidation cycles with MLX training integration.
"""

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from det import (
    DETCore, MemoryDomain, MemoryManager, TimerSystem,
    ModelConfig, ModelPool, ModelStatus, LLMRouter, RoutingResult,
    MultiModelInterface, DEFAULT_MODELS,
    ConsolidationManager, ConsolidationConfig, ConsolidationState,
    ConsolidationPhase, ConsolidationCycle, IdleDetector, setup_consolidation,
    is_mlx_available,
)


# ============================================================================
# Model Configuration Tests
# ============================================================================

def test_model_config_creation():
    """Test ModelConfig creation."""
    print("  test_model_config_creation...", end=" ")

    config = ModelConfig(
        name="test-model:7b",
        display_name="Test Model 7B",
        domains=[MemoryDomain.MATH, MemoryDomain.SCIENCE],
        priority=10,
    )

    assert config.name == "test-model:7b"
    assert config.display_name == "Test Model 7B"
    assert MemoryDomain.MATH in config.domains
    assert config.priority == 10
    assert config.status == ModelStatus.UNKNOWN

    print("PASS")


def test_model_config_defaults():
    """Test ModelConfig default values."""
    print("  test_model_config_defaults...", end=" ")

    config = ModelConfig(
        name="test",
        display_name="Test",
        domains=[MemoryDomain.GENERAL],
    )

    assert config.context_length == 4096
    assert config.default_temperature == 0.7
    assert config.system_prompt is None
    assert config.is_fast is False
    assert config.is_accurate is True
    assert config.avg_latency_ms == 0.0

    print("PASS")


def test_model_status_enum():
    """Test ModelStatus enum values."""
    print("  test_model_status_enum...", end=" ")

    assert ModelStatus.UNKNOWN.value == "unknown"
    assert ModelStatus.AVAILABLE.value == "available"
    assert ModelStatus.UNAVAILABLE.value == "unavailable"
    assert ModelStatus.LOADING.value == "loading"
    assert ModelStatus.ERROR.value == "error"

    print("PASS")


def test_default_models():
    """Test DEFAULT_MODELS configurations."""
    print("  test_default_models...", end=" ")

    assert "general" in DEFAULT_MODELS
    assert "math" in DEFAULT_MODELS
    assert "code" in DEFAULT_MODELS
    assert "reasoning" in DEFAULT_MODELS

    # Check math model
    math = DEFAULT_MODELS["math"]
    assert MemoryDomain.MATH in math.domains
    assert math.supports_math is True
    assert math.default_temperature == 0.3

    # Check code model
    code = DEFAULT_MODELS["code"]
    assert MemoryDomain.CODE in code.domains
    assert code.supports_code is True

    print("PASS")


# ============================================================================
# Model Pool Tests
# ============================================================================

def test_model_pool_init():
    """Test ModelPool initialization."""
    print("  test_model_pool_init...", end=" ")

    pool = ModelPool()

    assert pool.ollama_url == "http://localhost:11434"
    assert pool.health_check_interval == 60.0

    print("PASS")


def test_model_pool_register():
    """Test model registration."""
    print("  test_model_pool_register...", end=" ")

    pool = ModelPool()

    config = ModelConfig(
        name="test-model",
        display_name="Test",
        domains=[MemoryDomain.GENERAL],
    )

    pool.register_model("test", config)

    assert pool.get_model("test") is config
    assert pool.get_client("test") is not None

    print("PASS")


def test_model_pool_register_defaults():
    """Test registering default models."""
    print("  test_model_pool_register_defaults...", end=" ")

    pool = ModelPool()
    pool.register_defaults()

    assert pool.get_model("general") is not None
    assert pool.get_model("math") is not None
    assert pool.get_model("code") is not None
    assert pool.get_model("reasoning") is not None

    print("PASS")


def test_model_pool_list_models():
    """Test listing registered models."""
    print("  test_model_pool_list_models...", end=" ")

    pool = ModelPool()
    pool.register_defaults()

    models = pool.list_models()

    assert len(models) >= 4
    assert all("key" in m for m in models)
    assert all("name" in m for m in models)
    assert all("domains" in m for m in models)

    print("PASS")


def test_model_pool_get_models_for_domain():
    """Test getting models for a domain."""
    print("  test_model_pool_get_models_for_domain...", end=" ")

    pool = ModelPool()
    pool.register_defaults()

    # Mark some models as available
    pool._models["math"].status = ModelStatus.AVAILABLE
    pool._models["general"].status = ModelStatus.AVAILABLE

    math_models = pool.get_models_for_domain(MemoryDomain.MATH)

    # Math model should be first (higher priority)
    assert "math" in math_models

    print("PASS")


# ============================================================================
# LLM Router Tests
# ============================================================================

def test_llm_router_init():
    """Test LLMRouter initialization."""
    print("  test_llm_router_init...", end=" ")

    router = LLMRouter()

    assert router.pool is not None
    assert router.fallback_model == "general"

    print("PASS")


def test_llm_router_with_core():
    """Test LLMRouter with DET core."""
    print("  test_llm_router_with_core...", end=" ")

    with DETCore() as core:
        router = LLMRouter(core=core)

        assert router.core is core

    print("PASS")


def test_llm_router_route_math():
    """Test routing math request."""
    print("  test_llm_router_route_math...", end=" ")

    router = LLMRouter()
    router.pool.register_defaults()

    # Mark math model as available
    router.pool._models["math"].status = ModelStatus.AVAILABLE
    router.pool._models["general"].status = ModelStatus.AVAILABLE

    result = router.route("Calculate the integral of x squared")

    assert isinstance(result, RoutingResult)
    assert result.domain == MemoryDomain.MATH

    print("PASS")


def test_llm_router_route_code():
    """Test routing code request."""
    print("  test_llm_router_route_code...", end=" ")

    router = LLMRouter()
    router.pool.register_defaults()

    # Mark code model as available
    router.pool._models["code"].status = ModelStatus.AVAILABLE
    router.pool._models["general"].status = ModelStatus.AVAILABLE

    result = router.route("Write a Python function to sort a list")

    assert result.domain == MemoryDomain.CODE

    print("PASS")


def test_llm_router_route_explicit_domain():
    """Test routing with explicit domain."""
    print("  test_llm_router_route_explicit_domain...", end=" ")

    router = LLMRouter()
    router.pool.register_defaults()
    router.pool._models["reasoning"].status = ModelStatus.AVAILABLE

    result = router.route("Hello world", domain=MemoryDomain.REASONING)

    assert result.domain == MemoryDomain.REASONING
    assert result.confidence == 1.0

    print("PASS")


def test_llm_router_fallback():
    """Test fallback when specialized model unavailable."""
    print("  test_llm_router_fallback...", end=" ")

    router = LLMRouter()
    router.pool.register_defaults()

    # Only mark general as available, leave math as UNKNOWN
    router.pool._models["general"].status = ModelStatus.AVAILABLE
    # Explicitly ensure math is not available
    router.pool._models["math"].status = ModelStatus.UNAVAILABLE

    # Force domain to MATH to test fallback
    result = router.route("Calculate the math equation", domain=MemoryDomain.MATH)

    # Should fall back to general model since math is unavailable
    assert result.fallback_used is True
    assert result.model_config.name == router.pool._models["general"].name

    print("PASS")


def test_llm_router_status():
    """Test router status reporting."""
    print("  test_llm_router_status...", end=" ")

    router = LLMRouter()
    router.pool.register_defaults()

    status = router.get_status()

    assert "models" in status
    assert "available_count" in status
    assert "fallback_model" in status
    assert "has_core" in status

    print("PASS")


def test_llm_router_clear_context():
    """Test clearing conversation context."""
    print("  test_llm_router_clear_context...", end=" ")

    router = LLMRouter()

    # Add some context
    router._contexts["test"] = [{"role": "user", "content": "hello"}]

    assert len(router._contexts) == 1

    router.clear_context()

    assert len(router._contexts) == 0

    print("PASS")


# ============================================================================
# Routing Result Tests
# ============================================================================

def test_routing_result_creation():
    """Test RoutingResult creation."""
    print("  test_routing_result_creation...", end=" ")

    config = ModelConfig(
        name="test",
        display_name="Test",
        domains=[MemoryDomain.MATH],
    )

    result = RoutingResult(
        model_config=config,
        domain=MemoryDomain.MATH,
        confidence=0.9,
        fallback_used=False,
        reason="Best match",
    )

    assert result.model_config is config
    assert result.domain == MemoryDomain.MATH
    assert result.confidence == 0.9
    assert result.fallback_used is False

    print("PASS")


# ============================================================================
# Multi-Model Interface Tests
# ============================================================================

def test_multi_model_interface_init():
    """Test MultiModelInterface initialization."""
    print("  test_multi_model_interface_init...", end=" ")

    interface = MultiModelInterface()

    assert interface.router is not None
    assert interface.core is None

    print("PASS")


def test_multi_model_interface_with_core():
    """Test MultiModelInterface with DET core."""
    print("  test_multi_model_interface_with_core...", end=" ")

    with DETCore() as core:
        interface = MultiModelInterface(core=core)

        assert interface.core is core
        assert interface.router.core is core

    print("PASS")


def test_multi_model_interface_status():
    """Test interface status reporting."""
    print("  test_multi_model_interface_status...", end=" ")

    with DETCore() as core:
        interface = MultiModelInterface(core=core)

        status = interface.get_status()

        assert "models" in status
        assert "has_core" in status
        assert status["has_core"] is True
        assert "det_state" in status

    print("PASS")


# ============================================================================
# Domain Detection Tests
# ============================================================================

def test_domain_detection_math():
    """Test domain detection for math."""
    print("  test_domain_detection_math...", end=" ")

    router = LLMRouter()

    domain, conf = router._detect_domain("Calculate the sum of 1 to 100")

    assert domain == MemoryDomain.MATH
    assert conf > 0.5

    print("PASS")


def test_domain_detection_code():
    """Test domain detection for code."""
    print("  test_domain_detection_code...", end=" ")

    router = LLMRouter()

    domain, conf = router._detect_domain("Write a Python function to reverse a string")

    assert domain == MemoryDomain.CODE
    assert conf > 0.5

    print("PASS")


def test_domain_detection_science():
    """Test domain detection for science."""
    print("  test_domain_detection_science...", end=" ")

    router = LLMRouter()

    domain, conf = router._detect_domain("Explain the physics of black holes")

    assert domain == MemoryDomain.SCIENCE
    assert conf > 0.5

    print("PASS")


def test_domain_detection_reasoning():
    """Test domain detection for reasoning."""
    print("  test_domain_detection_reasoning...", end=" ")

    router = LLMRouter()

    domain, conf = router._detect_domain("Why does water boil? Analyze the logic.")

    assert domain == MemoryDomain.REASONING
    assert conf > 0.3

    print("PASS")


def test_domain_detection_general():
    """Test domain detection falls back to general."""
    print("  test_domain_detection_general...", end=" ")

    router = LLMRouter()

    domain, conf = router._detect_domain("Hello, how are you?")

    assert domain == MemoryDomain.GENERAL
    assert conf == 0.5  # Default confidence for unknown

    print("PASS")


# ============================================================================
# Integration Tests
# ============================================================================

def test_routing_with_det_integration():
    """Test routing with DET core integration."""
    print("  test_routing_with_det_integration...", end=" ")

    with DETCore() as core:
        router = LLMRouter(core=core)
        router.pool.register_defaults()

        # Run some DET steps
        for _ in range(10):
            core.step(0.1)

        # Route a request
        router.pool._models["math"].status = ModelStatus.AVAILABLE
        result = router.route("Calculate the derivative")

        assert result.domain == MemoryDomain.MATH
        assert result.model_config is not None

    print("PASS")


def test_full_routing_flow():
    """Test full routing flow without actual LLM calls."""
    print("  test_full_routing_flow...", end=" ")

    with DETCore() as core:
        interface = MultiModelInterface(core=core)
        interface.router.pool.register_defaults()

        # Mark all models as available
        for model in interface.router.pool._models.values():
            model.status = ModelStatus.AVAILABLE

        # Test routing (not generation - that requires Ollama)
        routing = interface.router.route("Write Python code for sorting")

        assert routing.domain == MemoryDomain.CODE
        assert routing.model_config.supports_code is True

        routing2 = interface.router.route("Integrate x^2 from 0 to 1")

        assert routing2.domain == MemoryDomain.MATH
        assert routing2.model_config.supports_math is True

    print("PASS")


# ============================================================================
# Phase 5.2: Consolidation Tests
# ============================================================================

def test_idle_detector_init():
    """Test IdleDetector initialization."""
    print("  test_idle_detector_init...", end=" ")

    detector = IdleDetector()

    assert detector.idle_threshold_seconds == 300.0
    assert detector.activity_count == 0

    print("PASS")


def test_idle_detector_activity():
    """Test IdleDetector activity recording."""
    print("  test_idle_detector_activity...", end=" ")

    detector = IdleDetector(idle_threshold_seconds=1.0)

    detector.record_activity()
    assert detector.activity_count == 1
    assert not detector.is_idle()

    print("PASS")


def test_idle_detector_idle():
    """Test IdleDetector idle detection."""
    print("  test_idle_detector_idle...", end=" ")

    detector = IdleDetector(idle_threshold_seconds=0.1)

    detector.record_activity()
    time.sleep(0.15)

    assert detector.is_idle()
    assert detector.idle_duration() >= 0.1

    print("PASS")


def test_consolidation_config_defaults():
    """Test ConsolidationConfig default values."""
    print("  test_consolidation_config_defaults...", end=" ")

    config = ConsolidationConfig()

    assert config.idle_threshold_seconds == 300.0
    assert config.min_consolidation_interval == 3600.0
    assert config.max_consolidation_duration == 1800.0
    assert config.min_examples_per_domain == 10
    assert config.require_positive_valence is True

    print("PASS")


def test_consolidation_config_custom():
    """Test ConsolidationConfig with custom values."""
    print("  test_consolidation_config_custom...", end=" ")

    config = ConsolidationConfig(
        idle_threshold_seconds=60.0,
        min_consolidation_interval=1800.0,
        max_domains_per_cycle=5,
    )

    assert config.idle_threshold_seconds == 60.0
    assert config.min_consolidation_interval == 1800.0
    assert config.max_domains_per_cycle == 5

    print("PASS")


def test_consolidation_state_enum():
    """Test ConsolidationState enum values."""
    print("  test_consolidation_state_enum...", end=" ")

    assert ConsolidationState.IDLE.value == "idle"
    assert ConsolidationState.MONITORING.value == "monitoring"
    assert ConsolidationState.CONSOLIDATING.value == "consolidating"
    assert ConsolidationState.TRAINING.value == "training"
    assert ConsolidationState.RECOVERING.value == "recovering"

    print("PASS")


def test_consolidation_phase_enum():
    """Test ConsolidationPhase enum values."""
    print("  test_consolidation_phase_enum...", end=" ")

    assert ConsolidationPhase.MEMORY_SCAN.value == "memory_scan"
    assert ConsolidationPhase.DATA_GENERATION.value == "data_generation"
    assert ConsolidationPhase.MODEL_TRAINING.value == "model_training"
    assert ConsolidationPhase.GRACE_INJECTION.value == "grace_injection"
    assert ConsolidationPhase.VERIFICATION.value == "verification"

    print("PASS")


def test_consolidation_cycle_creation():
    """Test ConsolidationCycle creation."""
    print("  test_consolidation_cycle_creation...", end=" ")

    cycle = ConsolidationCycle(
        cycle_id="test-123",
        started_at=time.time(),
    )

    assert cycle.cycle_id == "test-123"
    assert cycle.state == ConsolidationState.CONSOLIDATING
    assert cycle.phase == ConsolidationPhase.MEMORY_SCAN
    assert len(cycle.domains_processed) == 0

    print("PASS")


def test_consolidation_cycle_to_dict():
    """Test ConsolidationCycle serialization."""
    print("  test_consolidation_cycle_to_dict...", end=" ")

    cycle = ConsolidationCycle(
        cycle_id="test-456",
        started_at=time.time(),
    )
    cycle.domains_processed.append("MATH")

    data = cycle.to_dict()

    assert data["cycle_id"] == "test-456"
    assert data["state"] == "consolidating"
    assert "MATH" in data["domains_processed"]
    assert "duration" in data

    print("PASS")


def test_consolidation_manager_init():
    """Test ConsolidationManager initialization."""
    print("  test_consolidation_manager_init...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConsolidationManager(storage_path=Path(tmpdir))

        assert manager.state == ConsolidationState.IDLE
        assert manager.core is None
        assert manager.memory_manager is None

    print("PASS")


def test_consolidation_manager_with_core():
    """Test ConsolidationManager with DET core."""
    print("  test_consolidation_manager_with_core...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConsolidationManager(
                core=core,
                storage_path=Path(tmpdir)
            )

            assert manager.core is core

    print("PASS")


def test_consolidation_manager_record_activity():
    """Test activity recording."""
    print("  test_consolidation_manager_record_activity...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConsolidationManager(storage_path=Path(tmpdir))

        manager.record_activity()

        assert not manager.is_idle()
        assert manager.idle_detector.activity_count == 1

    print("PASS")


def test_consolidation_manager_status():
    """Test status reporting."""
    print("  test_consolidation_manager_status...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConsolidationManager(storage_path=Path(tmpdir))

        status = manager.get_status()

        assert "state" in status
        assert "is_idle" in status
        assert "can_consolidate" in status
        assert "mlx_available" in status
        assert status["state"] == "idle"

    print("PASS")


def test_consolidation_manager_can_consolidate_no_memory():
    """Test can_consolidate without memory manager."""
    print("  test_consolidation_manager_can_consolidate_no_memory...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ConsolidationConfig(idle_threshold_seconds=0)
        manager = ConsolidationManager(
            config=config,
            storage_path=Path(tmpdir)
        )

        # Should be False without memory manager
        assert manager.can_consolidate() is False

    print("PASS")


def test_consolidation_manager_with_timer():
    """Test integration with timer system."""
    print("  test_consolidation_manager_with_timer...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            timer = TimerSystem(core=core, storage_path=Path(tmpdir) / "timer")
            manager = ConsolidationManager(
                core=core,
                timer_system=timer,
                storage_path=Path(tmpdir) / "consolidation"
            )

            assert manager.timer_system is timer

            # Schedule consolidation
            event = manager.schedule_consolidation(hour=3, minute=0)
            assert event is not None

    print("PASS")


def test_consolidation_manager_history():
    """Test consolidation history."""
    print("  test_consolidation_manager_history...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ConsolidationManager(storage_path=Path(tmpdir))

        history = manager.get_history()
        assert isinstance(history, list)
        assert len(history) == 0  # No history yet

    print("PASS")


def test_setup_consolidation():
    """Test setup_consolidation convenience function."""
    print("  test_setup_consolidation...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(core, storage_path=Path(tmpdir) / "memory")
            timer = TimerSystem(core=core, storage_path=Path(tmpdir) / "timer")

            manager = setup_consolidation(
                core=core,
                memory_manager=memory,
                timer_system=timer,
                auto_start=False  # Don't start monitoring in test
            )

            assert manager.core is core
            assert manager.memory_manager is memory
            assert manager.timer_system is timer

    print("PASS")


def test_consolidation_full_flow():
    """Test full consolidation flow (without actual training)."""
    print("  test_consolidation_full_flow...", end=" ")

    with DETCore() as core:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(core, storage_path=Path(tmpdir) / "memory")

            # Add some memories
            for i in range(15):
                memory.store(
                    f"Test memory {i}: What is {i}? Answer is {i}.",
                    domain=MemoryDomain.GENERAL,
                    importance=0.7
                )

            config = ConsolidationConfig(
                idle_threshold_seconds=0,  # No wait
                min_consolidation_interval=0,  # Allow immediate
                min_examples_per_domain=10,
            )

            manager = ConsolidationManager(
                core=core,
                memory_manager=memory,
                config=config,
                storage_path=Path(tmpdir) / "consolidation"
            )

            # Check idle detection works
            manager.idle_detector.last_activity = 0  # Force idle

            status = manager.get_status()
            assert status["is_idle"] is True

    print("PASS")


# ============================================================================
# Main
# ============================================================================

def run_tests():
    """Run all Phase 5 tests."""
    print("\n========================================")
    print("  DET Phase 5 Tests")
    print("========================================\n")

    tests = [
        # Phase 5.1: Model Configuration
        test_model_config_creation,
        test_model_config_defaults,
        test_model_status_enum,
        test_default_models,

        # Phase 5.1: Model Pool
        test_model_pool_init,
        test_model_pool_register,
        test_model_pool_register_defaults,
        test_model_pool_list_models,
        test_model_pool_get_models_for_domain,

        # Phase 5.1: LLM Router
        test_llm_router_init,
        test_llm_router_with_core,
        test_llm_router_route_math,
        test_llm_router_route_code,
        test_llm_router_route_explicit_domain,
        test_llm_router_fallback,
        test_llm_router_status,
        test_llm_router_clear_context,

        # Phase 5.1: Routing Result
        test_routing_result_creation,

        # Phase 5.1: Multi-Model Interface
        test_multi_model_interface_init,
        test_multi_model_interface_with_core,
        test_multi_model_interface_status,

        # Phase 5.1: Domain Detection
        test_domain_detection_math,
        test_domain_detection_code,
        test_domain_detection_science,
        test_domain_detection_reasoning,
        test_domain_detection_general,

        # Phase 5.1: Integration
        test_routing_with_det_integration,
        test_full_routing_flow,

        # Phase 5.2: Idle Detection
        test_idle_detector_init,
        test_idle_detector_activity,
        test_idle_detector_idle,

        # Phase 5.2: Consolidation Config
        test_consolidation_config_defaults,
        test_consolidation_config_custom,
        test_consolidation_state_enum,
        test_consolidation_phase_enum,

        # Phase 5.2: Consolidation Cycle
        test_consolidation_cycle_creation,
        test_consolidation_cycle_to_dict,

        # Phase 5.2: Consolidation Manager
        test_consolidation_manager_init,
        test_consolidation_manager_with_core,
        test_consolidation_manager_record_activity,
        test_consolidation_manager_status,
        test_consolidation_manager_can_consolidate_no_memory,
        test_consolidation_manager_with_timer,
        test_consolidation_manager_history,

        # Phase 5.2: Integration
        test_setup_consolidation,
        test_consolidation_full_flow,
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
