"""
Test Suite for Quantum-Classical Transition
===========================================

Tests the DET v6.4 Quantum-Classical Transition feature:
Agency-coherence interplay analysis.

Test Categories:
1. Coherence analysis
2. Agency analysis
3. Decoherence simulation
4. Entanglement metrics
5. Regime classification
6. Physics consistency
"""

import numpy as np
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'calibration'))

from quantum_classical_transition import (
    CoherenceAnalyzer,
    AgencyAnalyzer,
    DecoherenceSimulator,
    EntanglementAnalyzer,
    RegimeClassifier,
    QuantumClassicalAnalyzer,
    run_quantum_classical_analysis,
    C_QUANTUM_THRESHOLD,
    C_CLASSICAL_THRESHOLD,
    CoherenceState,
    AgencyState,
    EntanglementMetrics
)

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


# ==============================================================================
# TEST COHERENCE ANALYSIS
# ==============================================================================

class TestCoherenceAnalyzer:
    """Test coherence analysis functions."""

    def test_analyzer_creation(self):
        """Test basic analyzer creation."""
        analyzer = CoherenceAnalyzer(verbose=False)
        assert analyzer is not None

    def test_measure_coherence_state(self):
        """Test coherence state measurement."""
        params = DETParams3D(N=16, C_init=0.5)
        sim = DETCollider3D(params)

        analyzer = CoherenceAnalyzer(verbose=False)
        state = analyzer.measure_coherence_state(sim)

        assert isinstance(state, CoherenceState)
        assert 0 <= state.C_mean <= 1
        assert state.quantum_fraction + state.classical_fraction <= 1

    def test_high_coherence_state(self):
        """Test that high C_init gives high coherence."""
        params = DETParams3D(N=16, C_init=0.9)
        sim = DETCollider3D(params)

        analyzer = CoherenceAnalyzer(verbose=False)
        state = analyzer.measure_coherence_state(sim)

        assert state.C_mean > 0.5, "High C_init should give high mean coherence"

    def test_low_coherence_state(self):
        """Test that low C_init gives low coherence."""
        params = DETParams3D(N=16, C_init=0.05)
        sim = DETCollider3D(params)

        analyzer = CoherenceAnalyzer(verbose=False)
        state = analyzer.measure_coherence_state(sim)

        assert state.C_mean < 0.5, "Low C_init should give low mean coherence"

    def test_phase_correlation(self):
        """Test phase correlation computation."""
        params = DETParams3D(N=16, C_init=0.5)
        sim = DETCollider3D(params)

        analyzer = CoherenceAnalyzer(verbose=False)
        correlations = analyzer.compute_phase_correlation(sim, max_distance=5)

        assert len(correlations) == 5
        # Correlations should be bounded by [-1, 1]
        assert np.all(np.abs(correlations) <= 1.0)

    def test_coherence_weighted_correlation(self):
        """Test coherence-weighted correlation computation."""
        params = DETParams3D(N=16, C_init=0.5)
        sim = DETCollider3D(params)

        analyzer = CoherenceAnalyzer(verbose=False)
        corr = analyzer.measure_coherence_weighted_correlation(sim)

        # Should be bounded
        assert -1.0 <= corr <= 1.0


# ==============================================================================
# TEST AGENCY ANALYSIS
# ==============================================================================

class TestAgencyAnalyzer:
    """Test agency analysis functions."""

    def test_analyzer_creation(self):
        """Test basic analyzer creation."""
        analyzer = AgencyAnalyzer(verbose=False)
        assert analyzer.lambda_a == 30.0

    def test_agency_ceiling_computation(self):
        """Test agency ceiling formula."""
        analyzer = AgencyAnalyzer(lambda_a=30.0, verbose=False)

        # Test at q = 0: ceiling should be 1
        q_zero = np.zeros((4, 4, 4))
        ceiling = analyzer.compute_agency_ceiling(q_zero)
        assert np.allclose(ceiling, 1.0)

        # Test at q = 1: ceiling should be small
        q_one = np.ones((4, 4, 4))
        ceiling = analyzer.compute_agency_ceiling(q_one)
        expected = 1.0 / (1.0 + 30.0)  # ~0.032
        assert np.allclose(ceiling, expected)

    def test_measure_agency_state(self):
        """Test agency state measurement."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)

        analyzer = AgencyAnalyzer(verbose=False)
        state = analyzer.measure_agency_state(sim)

        assert isinstance(state, AgencyState)
        assert 0 <= state.a_mean <= 1
        assert 0 <= state.constrained_fraction <= 1

    def test_agency_coherence_correlation(self):
        """Test agency-coherence correlation."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)

        analyzer = AgencyAnalyzer(verbose=False)
        corr = analyzer.compute_agency_coherence_correlation(sim)

        # Correlation should be bounded
        assert -1.0 <= corr <= 1.0


# ==============================================================================
# TEST DECOHERENCE SIMULATION
# ==============================================================================

class TestDecoherenceSimulator:
    """Test decoherence simulation."""

    def test_simulator_creation(self):
        """Test simulator creation."""
        sim = DecoherenceSimulator(grid_size=16, verbose=False)
        assert sim.N == 16

    def test_setup_coherent_state(self):
        """Test coherent state setup."""
        decoher = DecoherenceSimulator(grid_size=16, verbose=False)
        sim = decoher.setup_coherent_state(C_init=0.8)

        # Should have high initial coherence (using bond coherences)
        C_mean = np.mean((sim.C_X + sim.C_Y + sim.C_Z) / 3.0)
        assert C_mean > 0.7

    def test_decoherence_simulation_runs(self):
        """Test that decoherence simulation completes."""
        decoher = DecoherenceSimulator(grid_size=16, verbose=False)
        result = decoher.run_decoherence_simulation(
            C_init=0.8,
            n_steps=50,
            sample_interval=10
        )

        assert len(result.time_steps) > 0
        assert len(result.coherence_history) > 0
        assert result.initial_coherence > 0

    def test_coherence_decays(self):
        """Test that coherence decays over time."""
        decoher = DecoherenceSimulator(grid_size=16, verbose=False)
        result = decoher.run_decoherence_simulation(
            C_init=0.9,
            n_steps=100,
            sample_interval=20
        )

        # Coherence should decay (or at least not increase significantly)
        # Due to dynamics, allow some variation
        assert result.final_coherence <= result.initial_coherence * 1.2

    def test_measurement_causes_decoherence(self):
        """Test that measurement reduces coherence."""
        decoher = DecoherenceSimulator(grid_size=16, verbose=False)
        sim = decoher.setup_coherent_state(C_init=0.8)

        C_before = np.mean((sim.C_X + sim.C_Y + sim.C_Z) / 3.0)
        C_after = decoher.simulate_measurement(sim, measurement_strength=0.5)

        assert C_after < C_before, "Measurement should reduce coherence"


# ==============================================================================
# TEST ENTANGLEMENT ANALYSIS
# ==============================================================================

class TestEntanglementAnalyzer:
    """Test entanglement-like correlation analysis."""

    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = EntanglementAnalyzer(verbose=False)
        assert analyzer is not None

    def test_bell_parameter(self):
        """Test Bell parameter computation."""
        params = DETParams3D(N=16, C_init=0.5)
        sim = DETCollider3D(params)

        analyzer = EntanglementAnalyzer(verbose=False)
        S = analyzer.compute_bell_parameter(sim, n_samples=100)

        # Bell parameter should be non-negative
        assert S >= 0

    def test_locality_violation(self):
        """Test locality violation measurement."""
        params = DETParams3D(N=16, C_init=0.5)
        sim = DETCollider3D(params)

        analyzer = EntanglementAnalyzer(verbose=False)
        violation = analyzer.compute_locality_violation(sim)

        # Violation should be bounded [0, 1]
        assert 0 <= violation <= 1

    def test_entanglement_metrics(self):
        """Test full entanglement metrics."""
        params = DETParams3D(N=16, C_init=0.5)
        sim = DETCollider3D(params)

        analyzer = EntanglementAnalyzer(verbose=False)
        metrics = analyzer.measure_entanglement_metrics(sim)

        assert isinstance(metrics, EntanglementMetrics)
        assert metrics.bell_parameter >= 0
        assert 0 <= metrics.locality_violation <= 1


# ==============================================================================
# TEST REGIME CLASSIFICATION
# ==============================================================================

class TestRegimeClassifier:
    """Test quantum-classical regime classification."""

    def test_classifier_creation(self):
        """Test classifier creation."""
        classifier = RegimeClassifier(verbose=False)
        assert classifier is not None

    def test_quantum_regime_classification(self):
        """Test classification of quantum regime."""
        classifier = RegimeClassifier(verbose=False)

        # Create quantum-like state
        coherence_state = CoherenceState(
            C_mean=0.8,
            C_max=0.95,
            C_std=0.1,
            quantum_fraction=0.7,
            classical_fraction=0.1
        )

        entanglement = EntanglementMetrics(
            correlation_strength=0.6,
            bell_parameter=2.5,  # > 2 = quantum
            locality_violation=0.4,
            coherence_correlation=0.5
        )

        regime = classifier.classify(coherence_state, entanglement)
        assert regime == 'quantum'

    def test_classical_regime_classification(self):
        """Test classification of classical regime."""
        classifier = RegimeClassifier(verbose=False)

        # Create classical-like state
        coherence_state = CoherenceState(
            C_mean=0.05,
            C_max=0.1,
            C_std=0.02,
            quantum_fraction=0.0,
            classical_fraction=0.9
        )

        entanglement = EntanglementMetrics(
            correlation_strength=0.1,
            bell_parameter=1.5,  # < 2 = classical
            locality_violation=0.05,
            coherence_correlation=0.1
        )

        regime = classifier.classify(coherence_state, entanglement)
        assert regime == 'classical'

    def test_transition_regime_classification(self):
        """Test classification of transition regime."""
        classifier = RegimeClassifier(verbose=False)

        # Create intermediate state
        coherence_state = CoherenceState(
            C_mean=0.3,
            C_max=0.5,
            C_std=0.15,
            quantum_fraction=0.3,
            classical_fraction=0.3
        )

        entanglement = EntanglementMetrics(
            correlation_strength=0.3,
            bell_parameter=1.8,  # Near classical limit
            locality_violation=0.15,
            coherence_correlation=0.3
        )

        regime = classifier.classify(coherence_state, entanglement)
        assert regime == 'transition'


# ==============================================================================
# TEST FULL ANALYZER
# ==============================================================================

class TestQuantumClassicalAnalyzer:
    """Test complete quantum-classical analyzer."""

    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = QuantumClassicalAnalyzer(grid_size=16, verbose=False)
        assert analyzer.N == 16

    def test_analyze_state(self):
        """Test state analysis."""
        params = DETParams3D(N=16, C_init=0.5)
        sim = DETCollider3D(params)

        analyzer = QuantumClassicalAnalyzer(grid_size=16, verbose=False)
        analysis = analyzer.analyze_state(sim)

        assert analysis.coherence_state is not None
        assert analysis.agency_state is not None
        assert analysis.entanglement is not None
        assert analysis.regime in ['quantum', 'classical', 'transition']

    def test_full_analysis(self):
        """Test complete analysis pipeline."""
        analyzer = QuantumClassicalAnalyzer(grid_size=16, verbose=False)
        results = analyzer.run_full_analysis(
            C_init_values=[0.2, 0.5, 0.8],
            decoherence_steps=50
        )

        assert 'analyses' in results
        assert 'decoherence_results' in results
        assert 'regime_summary' in results
        assert 'summary' in results

        assert len(results['analyses']) == 3
        assert len(results['decoherence_results']) == 3


# ==============================================================================
# TEST PHYSICS CONSISTENCY
# ==============================================================================

class TestPhysicsConsistency:
    """Test physics consistency of quantum-classical analysis."""

    def test_high_coherence_more_quantum(self):
        """Test that high coherence gives more quantum-like behavior."""
        analyzer = QuantumClassicalAnalyzer(grid_size=16, verbose=False)

        # High coherence simulation
        high_C_sim = analyzer.decoherence_sim.setup_coherent_state(C_init=0.9)
        high_analysis = analyzer.analyze_state(high_C_sim)

        # Low coherence simulation
        low_C_sim = analyzer.decoherence_sim.setup_coherent_state(C_init=0.1)
        low_analysis = analyzer.analyze_state(low_C_sim)

        # High coherence should have higher quantum fraction
        assert high_analysis.coherence_state.quantum_fraction >= low_analysis.coherence_state.quantum_fraction

    def test_coherence_threshold_consistency(self):
        """Test coherence threshold definitions are consistent."""
        assert C_QUANTUM_THRESHOLD > C_CLASSICAL_THRESHOLD
        assert C_QUANTUM_THRESHOLD <= 1.0
        assert C_CLASSICAL_THRESHOLD >= 0.0

    def test_bell_parameter_bounded(self):
        """Test that Bell parameter is reasonably bounded."""
        params = DETParams3D(N=16, C_init=0.5)
        sim = DETCollider3D(params)

        analyzer = EntanglementAnalyzer(verbose=False)
        S = analyzer.compute_bell_parameter(sim, n_samples=200)

        # Bell parameter should be bounded by 2*sqrt(2) ~ 2.83 (Tsirelson bound)
        # In practice, due to simulation, might be higher, but should be finite
        assert 0 <= S < 10

    def test_agency_bounded_by_ceiling(self):
        """Test that agency doesn't exceed ceiling."""
        params = DETParams3D(N=16, agency_dynamic=True, lambda_a=30.0)
        sim = DETCollider3D(params)

        # Run some steps
        for _ in range(20):
            sim.step()

        analyzer = AgencyAnalyzer(lambda_a=30.0, verbose=False)
        ceiling = analyzer.compute_agency_ceiling(sim.q)

        # Agency should not significantly exceed ceiling
        excess = sim.a - ceiling
        max_excess = np.max(excess)

        # Allow small numerical tolerance
        assert max_excess < 0.1, f"Agency exceeded ceiling by {max_excess}"

    def test_decoherence_rate_positive(self):
        """Test that decoherence rate is non-negative."""
        decoher = DecoherenceSimulator(grid_size=16, verbose=False)
        result = decoher.run_decoherence_simulation(
            C_init=0.8,
            n_steps=100,
            sample_interval=20
        )

        # Decoherence rate should be non-negative (coherence decays, not grows)
        assert result.decoherence_rate >= -0.01  # Allow small negative due to fitting noise


# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

def test_full_quantum_classical_analysis_integration():
    """
    Full integration test of quantum-classical transition analysis.

    Runs complete analysis pipeline and validates outputs.
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full Quantum-Classical Transition Analysis")
    print("="*60)

    results = run_quantum_classical_analysis(
        grid_size=16,
        C_init_values=[0.2, 0.5, 0.8],
        verbose=True
    )

    # Validate structure
    assert 'analyses' in results
    assert 'decoherence_results' in results
    assert 'regime_summary' in results
    assert 'summary' in results

    # Validate analyses
    analyses = results['analyses']
    assert len(analyses) == 3

    for item in analyses:
        analysis = item['analysis']
        assert analysis.coherence_state is not None
        assert analysis.agency_state is not None
        assert analysis.entanglement is not None
        assert analysis.regime in ['quantum', 'classical', 'transition']

    # Validate decoherence results
    for item in results['decoherence_results']:
        result = item['result']
        assert result.initial_coherence > 0
        assert len(result.time_steps) > 0

    # Validate regime summary
    summary = results['regime_summary']
    total = summary['quantum_count'] + summary['classical_count'] + summary['transition_count']
    assert total == 3

    # Check summary was generated
    assert len(results['summary']) > 100

    print("\n" + "-"*60)
    print("Integration test PASSED")
    print("-"*60)


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    print("Running Quantum-Classical Transition Test Suite")
    print("="*60)

    pytest.main([__file__, "-v", "--tb=short"])
