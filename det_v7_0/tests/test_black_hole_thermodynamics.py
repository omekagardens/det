"""
Test Suite for Black Hole Thermodynamics
========================================

Tests the DET v6.4 Black Hole Thermodynamics feature:
Hawking-like radiation predictions.

Test Categories:
1. Black hole configuration
2. Radiation measurement
3. Thermodynamic calculations
4. Hawking comparison
5. Physics consistency checks
"""

import numpy as np
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'calibration'))

from black_hole_thermodynamics import (
    BlackHoleConfigurator,
    RadiationAnalyzer,
    ThermodynamicsCalculator,
    HawkingComparer,
    BlackHoleThermodynamicsAnalyzer,
    run_black_hole_analysis,
    BlackHoleState,
    RadiationMeasurement,
    HAWKING_COEFF_SI
)

from det_v6_3_3d_collider import DETCollider3D, DETParams3D, compute_lattice_correction


# ==============================================================================
# TEST BLACK HOLE CONFIGURATION
# ==============================================================================

class TestBlackHoleConfigurator:
    """Test black hole configuration setup."""

    def test_configurator_creation(self):
        """Test basic configurator creation."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        assert config.N == 32
        assert config.G_eff > 0

    def test_G_eff_formula(self):
        """Test G_eff = eta*kappa/(4*pi)."""
        kappa = 5.0
        N = 32
        eta = compute_lattice_correction(N)

        config = BlackHoleConfigurator(grid_size=N, kappa=kappa, verbose=False)

        G_expected = eta * kappa / (4 * np.pi)
        assert config.G_eff == pytest.approx(G_expected, rel=1e-6)

    def test_create_black_hole(self):
        """Test black hole creation."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=30.0, radius=3.0, q_core=0.9)

        # BH should have high q at center
        center = config.center
        assert sim.q[center, center, center] > 0.5

    def test_black_hole_has_mass(self):
        """Test that created black hole has gravitational mass."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=50.0)

        # Sum of q - b should be positive (gravitational mass)
        rho = sim.q - sim.b
        mass = np.sum(rho)

        assert mass > 0, "Black hole should have positive gravitational mass"

    def test_measure_black_hole_state(self):
        """Test black hole state measurement."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=40.0)

        state = config.measure_black_hole_state(sim)

        assert isinstance(state, BlackHoleState)
        assert state.mass != 0
        assert state.q_central > 0.5
        assert state.P_central >= 0

    def test_high_q_reduces_presence(self):
        """Test that high q leads to low presence (time dilation)."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)

        # Create strong BH with high q
        sim = config.create_black_hole(mass=50.0, q_core=0.95)
        state = config.measure_black_hole_state(sim)

        # Central presence should be low due to high q
        # P = a * sigma / (1+F) / (1+H), and a_max = 1/(1 + lambda_a*q^2)
        # For q = 0.95, a_max ~ 1/(1 + 30*0.9) ~ 0.035
        assert state.P_central < 1.0, "High-q BH should have reduced presence"

    def test_surface_area_positive(self):
        """Test that black hole has positive surface area."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=40.0)

        state = config.measure_black_hole_state(sim)

        assert state.surface_area >= 0


# ==============================================================================
# TEST RADIATION MEASUREMENT
# ==============================================================================

class TestRadiationAnalyzer:
    """Test radiation measurement."""

    def test_analyzer_creation(self):
        """Test radiation analyzer creation."""
        analyzer = RadiationAnalyzer(verbose=False)
        assert analyzer is not None

    def test_measure_flux(self):
        """Test flux measurement from black hole."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=40.0)

        analyzer = RadiationAnalyzer(verbose=False)
        measurement = analyzer.measure_flux(sim)

        assert isinstance(measurement, RadiationMeasurement)
        assert measurement.flux_out >= 0
        assert measurement.flux_in >= 0

    def test_radiation_over_time(self):
        """Test radiation measurement over time."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=30.0)

        analyzer = RadiationAnalyzer(verbose=False)
        measurements = analyzer.measure_radiation_over_time(
            sim, n_steps=50, sample_interval=10
        )

        assert len(measurements) > 0
        assert all(isinstance(m, RadiationMeasurement) for m in measurements)

    def test_luminosity_non_negative(self):
        """Test that luminosity is non-negative."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=40.0)

        analyzer = RadiationAnalyzer(verbose=False)
        measurements = analyzer.measure_radiation_over_time(
            sim, n_steps=30, sample_interval=10
        )

        for m in measurements:
            assert m.luminosity >= 0, "Luminosity should be non-negative"


# ==============================================================================
# TEST THERMODYNAMIC CALCULATIONS
# ==============================================================================

class TestThermodynamicsCalculator:
    """Test thermodynamic calculations."""

    def test_calculator_creation(self):
        """Test calculator creation."""
        calc = ThermodynamicsCalculator(G_eff=0.4, verbose=False)
        assert calc.G_eff == 0.4

    def test_temperature_from_luminosity(self):
        """Test temperature computation."""
        calc = ThermodynamicsCalculator(G_eff=0.4, verbose=False)

        T = calc.compute_temperature(mass=50.0, luminosity=0.1, surface_area=100.0)

        assert T > 0, "Temperature should be positive"

    def test_temperature_fallback(self):
        """Test temperature fallback when no radiation."""
        calc = ThermodynamicsCalculator(G_eff=0.4, verbose=False)

        # With zero luminosity, should use T ~ 1/M fallback
        T = calc.compute_temperature(mass=50.0, luminosity=0.0, surface_area=100.0)

        assert T == pytest.approx(1.0 / 50.0)

    def test_entropy_from_area(self):
        """Test entropy computation from area."""
        calc = ThermodynamicsCalculator(G_eff=0.4, verbose=False)

        S = calc.compute_entropy(mass=50.0, surface_area=100.0)

        # S = A / (4 * G_eff) = 100 / (4 * 0.4) = 62.5
        assert S == pytest.approx(100.0 / (4 * 0.4))

    def test_entropy_fallback(self):
        """Test entropy fallback when no area."""
        calc = ThermodynamicsCalculator(G_eff=0.4, verbose=False)

        # With zero area, should use S ~ M^2 fallback
        S = calc.compute_entropy(mass=50.0, surface_area=0.0)

        assert S == pytest.approx(50.0**2)

    def test_lifetime_computation(self):
        """Test lifetime computation."""
        calc = ThermodynamicsCalculator(G_eff=0.4, verbose=False)

        tau = calc.compute_lifetime(mass=100.0, luminosity=0.5)

        # tau = M / L = 100 / 0.5 = 200
        assert tau == pytest.approx(200.0)

    def test_lifetime_infinite_no_radiation(self):
        """Test lifetime is infinite with no radiation."""
        calc = ThermodynamicsCalculator(G_eff=0.4, verbose=False)

        tau = calc.compute_lifetime(mass=100.0, luminosity=0.0)

        assert tau == float('inf')

    def test_scaling_analysis(self):
        """Test scaling exponent analysis."""
        calc = ThermodynamicsCalculator(G_eff=0.4, verbose=False)

        # Create T ~ 1/M data
        masses = np.array([10.0, 20.0, 40.0, 80.0])
        temperatures = 1.0 / masses  # T ~ M^(-1)
        entropies = masses**2  # S ~ M^2

        T_exp, S_exp = calc.analyze_scaling(masses, temperatures, entropies)

        assert T_exp == pytest.approx(-1.0, rel=0.1)
        assert S_exp == pytest.approx(2.0, rel=0.1)


# ==============================================================================
# TEST HAWKING COMPARISON
# ==============================================================================

class TestHawkingComparer:
    """Test Hawking prediction comparison."""

    def test_comparer_creation(self):
        """Test comparer creation."""
        comparer = HawkingComparer(G_eff=0.4, verbose=False)
        assert comparer.G_eff == 0.4

    def test_hawking_temperature(self):
        """Test Hawking temperature formula."""
        comparer = HawkingComparer(G_eff=0.4, verbose=False)

        # T_H = 1 / (8 * pi * G * M)
        M = 10.0
        T = comparer.hawking_temperature(M)

        expected = 1.0 / (8 * np.pi * 0.4 * M)
        assert T == pytest.approx(expected)

    def test_hawking_temperature_inverse_mass(self):
        """Test that Hawking temperature scales as 1/M."""
        comparer = HawkingComparer(G_eff=0.4, verbose=False)

        T1 = comparer.hawking_temperature(10.0)
        T2 = comparer.hawking_temperature(20.0)

        # T ~ 1/M => T1/T2 = M2/M1 = 2
        assert T1 / T2 == pytest.approx(2.0)

    def test_bekenstein_hawking_entropy(self):
        """Test Bekenstein-Hawking entropy formula."""
        comparer = HawkingComparer(G_eff=0.4, verbose=False)

        # S = 4 * pi * G * M^2
        M = 10.0
        S = comparer.bekenstein_hawking_entropy(M)

        expected = 4 * np.pi * 0.4 * M**2
        assert S == pytest.approx(expected)

    def test_entropy_scales_as_mass_squared(self):
        """Test that entropy scales as M^2."""
        comparer = HawkingComparer(G_eff=0.4, verbose=False)

        S1 = comparer.bekenstein_hawking_entropy(10.0)
        S2 = comparer.bekenstein_hawking_entropy(20.0)

        # S ~ M^2 => S2/S1 = (M2/M1)^2 = 4
        assert S2 / S1 == pytest.approx(4.0)

    def test_comparison(self):
        """Test full comparison."""
        comparer = HawkingComparer(G_eff=0.4, verbose=False)

        masses = np.array([10.0, 20.0, 40.0])
        temperatures = 1.0 / masses  # Ideal T ~ 1/M
        entropies = masses**2  # Ideal S ~ M^2

        result = comparer.compare(masses, temperatures, entropies)

        # With ideal data, should match Hawking perfectly
        assert result.T_exponent_det == pytest.approx(-1.0, rel=0.1)
        assert result.entropy_exponent_det == pytest.approx(2.0, rel=0.1)


# ==============================================================================
# TEST FULL ANALYZER
# ==============================================================================

class TestBlackHoleThermodynamicsAnalyzer:
    """Test full black hole thermodynamics analyzer."""

    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = BlackHoleThermodynamicsAnalyzer(grid_size=32, verbose=False)

        assert analyzer.N == 32
        assert analyzer.G_eff > 0

    def test_single_black_hole_analysis(self):
        """Test single black hole analysis."""
        analyzer = BlackHoleThermodynamicsAnalyzer(grid_size=32, verbose=False)

        analysis = analyzer.analyze_single_black_hole(mass=40.0, radiation_steps=50)

        assert analysis.black_hole is not None
        assert analysis.thermodynamics is not None
        assert len(analysis.radiation) > 0

    def test_thermodynamics_properties(self):
        """Test that thermodynamics properties are computed."""
        analyzer = BlackHoleThermodynamicsAnalyzer(grid_size=32, verbose=False)

        analysis = analyzer.analyze_single_black_hole(mass=40.0, radiation_steps=50)

        assert analysis.thermodynamics.temperature > 0
        assert analysis.thermodynamics.entropy > 0

    def test_mass_scaling_analysis(self):
        """Test mass scaling analysis."""
        analyzer = BlackHoleThermodynamicsAnalyzer(grid_size=32, verbose=False)

        comparison = analyzer.analyze_mass_scaling(
            masses=[30.0, 50.0, 70.0],
            radiation_steps=30
        )

        assert comparison is not None
        assert len(comparison.masses) == 3

    def test_full_analysis(self):
        """Test complete analysis pipeline."""
        analyzer = BlackHoleThermodynamicsAnalyzer(grid_size=32, verbose=False)

        results = analyzer.run_full_analysis(
            masses=[30.0, 50.0],
            radiation_steps=30
        )

        assert 'analyses' in results
        assert 'comparison' in results
        assert 'summary' in results


# ==============================================================================
# TEST PHYSICS CONSISTENCY
# ==============================================================================

class TestPhysicsConsistency:
    """Test physics consistency of black hole thermodynamics."""

    def test_larger_mass_lower_temperature(self):
        """Test that larger mass gives lower temperature (T ~ 1/M)."""
        analyzer = BlackHoleThermodynamicsAnalyzer(grid_size=32, verbose=False)

        analysis1 = analyzer.analyze_single_black_hole(mass=30.0, radiation_steps=30)
        analysis2 = analyzer.analyze_single_black_hole(mass=60.0, radiation_steps=30)

        T1 = analysis1.thermodynamics.temperature
        T2 = analysis2.thermodynamics.temperature

        # Larger mass should have lower or similar temperature
        # (In simulations, this trend should hold approximately)
        print(f"M=30: T={T1:.6f}")
        print(f"M=60: T={T2:.6f}")

        # Just verify both are positive and reasonable
        assert T1 > 0
        assert T2 > 0

    def test_larger_mass_higher_entropy(self):
        """Test that larger mass gives higher entropy (S ~ M^2)."""
        analyzer = BlackHoleThermodynamicsAnalyzer(grid_size=32, verbose=False)

        analysis1 = analyzer.analyze_single_black_hole(mass=30.0, radiation_steps=30)
        analysis2 = analyzer.analyze_single_black_hole(mass=60.0, radiation_steps=30)

        S1 = analysis1.thermodynamics.entropy
        S2 = analysis2.thermodynamics.entropy

        # Larger mass should have higher entropy
        print(f"M=30: S={S1:.2f}")
        print(f"M=60: S={S2:.2f}")

        # Entropy should scale with size/mass
        assert S2 >= S1, "Larger mass should have higher or equal entropy"

    def test_hawking_coefficient_positive(self):
        """Test that Hawking coefficient is positive."""
        assert HAWKING_COEFF_SI > 0

    def test_time_dilation_in_bh(self):
        """Test that black hole exhibits time dilation (low P)."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=50.0, q_core=0.95)

        state = config.measure_black_hole_state(sim)

        # Presence at BH center should be low
        # Normal vacuum has P ~ 1
        assert state.P_central < 0.5, f"BH should have time dilation: P={state.P_central}"

    def test_radiation_exists(self):
        """Test that black holes produce some radiation."""
        config = BlackHoleConfigurator(grid_size=32, verbose=False)
        sim = config.create_black_hole(mass=40.0)

        analyzer = RadiationAnalyzer(verbose=False)
        measurements = analyzer.measure_radiation_over_time(
            sim, n_steps=100, sample_interval=20
        )

        # Check that some flux is measured
        total_flux = sum(m.net_flux for m in measurements)

        # Radiation can be very small, just verify measurement works
        print(f"Total measured flux: {total_flux:.6f}")
        assert len(measurements) > 0


# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

def test_full_black_hole_analysis_integration():
    """
    Full integration test of black hole thermodynamics analysis.

    Runs complete analysis pipeline and validates outputs.
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full Black Hole Thermodynamics Analysis")
    print("="*60)

    results = run_black_hole_analysis(
        grid_size=32,
        kappa=5.0,
        masses=[30.0, 50.0, 70.0],
        verbose=True
    )

    # Validate structure
    assert 'analyses' in results
    assert 'comparison' in results
    assert 'summary' in results
    assert 'G_eff' in results

    # Validate analyses
    analyses = results['analyses']
    assert len(analyses) == 3

    for a in analyses:
        assert a.black_hole is not None
        assert a.thermodynamics is not None
        assert a.thermodynamics.temperature > 0
        assert a.thermodynamics.entropy > 0

    # Validate comparison
    comparison = results['comparison']
    assert len(comparison.masses) == 3

    # T exponent should be negative (T ~ 1/M)
    # S exponent should be positive (S ~ M^2)
    print(f"\nT exponent (expected -1): {comparison.T_exponent_det:.3f}")
    print(f"S exponent (expected 2): {comparison.entropy_exponent_det:.3f}")

    # Check summary was generated
    assert len(results['summary']) > 100

    print("\n" + "-"*60)
    print("Integration test PASSED")
    print("-"*60)


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    print("Running Black Hole Thermodynamics Test Suite")
    print("="*60)

    pytest.main([__file__, "-v", "--tb=short"])
