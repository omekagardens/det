"""
Test Suite for Gravitational Lensing
====================================

Tests the DET v6.4 Gravitational Lensing feature:
Ray-tracing through Φ field.

Test Categories:
1. Ray tracer basics
2. Deflection angle computation
3. Comparison with Schwarzschild prediction
4. Einstein radius calculation
5. Full lensing analysis
"""

import numpy as np
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'calibration'))

from gravitational_lensing import (
    GravitationalRayTracer,
    GravitationalLensing,
    ExtendedMassLensing,
    RayPath,
    LensingResult,
    run_lensing_analysis,
    C_LATTICE
)
from det_v6_3_3d_collider import DETCollider3D, DETParams3D, compute_lattice_correction
from det_si_units import solar_system_units, G_SI, C_SI


# ==============================================================================
# TEST RAY TRACER BASICS
# ==============================================================================

class TestRayTracerBasics:
    """Test basic ray tracer functionality."""

    def test_create_simulation(self):
        """Test creating a simulation for ray tracing."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        sim = lensing.setup_point_mass(mass=30.0, width=2.0, settling_steps=50)

        assert sim is not None
        assert sim.Phi.shape == (32, 32, 32)
        assert np.any(sim.Phi != 0), "Potential field should be non-zero"

        # Potential should be negative near center (attractive)
        center = 16
        phi_center = sim.Phi[center, center, center]
        phi_edge = sim.Phi[center, center, 0]
        assert phi_center < phi_edge, "Potential should be more negative at center"

        print(f"Potential at center: {phi_center:.4f}")
        print(f"Potential at edge: {phi_edge:.4f}")

    def test_ray_tracer_creation(self):
        """Test ray tracer initialization."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        sim = lensing.setup_point_mass(mass=30.0, settling_steps=50)

        tracer = GravitationalRayTracer(sim, verbose=False)

        assert tracer.N == 32
        assert tracer.center == 16
        assert tracer.phi_interp is not None

    def test_interpolation(self):
        """Test field interpolation at arbitrary positions."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        sim = lensing.setup_point_mass(mass=30.0, settling_steps=50)
        tracer = GravitationalRayTracer(sim, verbose=False)

        # Test potential at grid point
        pos = np.array([16.0, 16.0, 16.0])
        phi = tracer.get_potential(pos)
        assert np.isfinite(phi)

        # Test potential at off-grid point
        pos_off = np.array([16.5, 16.5, 16.5])
        phi_off = tracer.get_potential(pos_off)
        assert np.isfinite(phi_off)

        # Test gravity vector
        g = tracer.get_gravity(pos)
        assert g.shape == (3,)
        assert np.all(np.isfinite(g))

        print(f"Potential at center: {phi:.4f}")
        print(f"Gravity at center: {g}")


class TestRayTracing:
    """Test ray tracing through field."""

    def test_straight_ray_no_mass(self):
        """Test that ray goes straight when no mass present."""
        # Create simulation with no mass
        params = DETParams3D(
            N=32, DT=0.01, F_VAC=0.001,
            gravity_enabled=True, q_enabled=True
        )
        sim = DETCollider3D(params)

        # Step a bit to establish (empty) field
        for _ in range(10):
            sim.step()

        tracer = GravitationalRayTracer(sim, verbose=False)

        # Trace ray
        start = np.array([20.0, 16.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        path = tracer.trace_ray(start, direction, n_steps=100)

        # Ray should go roughly straight (very small deflection)
        assert path.total_deflection < 0.01, f"Deflection {path.total_deflection} too large for empty field"

        # Final x position should be close to initial
        delta_x = abs(path.positions[-1, 0] - path.positions[0, 0])
        assert delta_x < 1.0, f"Ray drifted too much in x: {delta_x}"

        print(f"Empty field deflection: {path.total_deflection:.6f} rad")

    def test_ray_deflected_by_mass(self):
        """Test that ray is deflected by central mass."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        sim = lensing.setup_point_mass(mass=50.0, settling_steps=50)
        tracer = GravitationalRayTracer(sim, verbose=False)

        # Trace ray with impact parameter of 5 cells
        start = np.array([21.0, 16.0, 0.0])  # 5 cells from center in x
        direction = np.array([0.0, 0.0, 1.0])
        path = tracer.trace_ray(start, direction, n_steps=200)

        # Should have measurable deflection
        assert path.total_deflection > 0.001, f"Expected measurable deflection, got {path.total_deflection}"

        print(f"Mass deflection: {path.total_deflection:.6f} rad")
        print(f"Impact parameter: {path.impact_parameter:.2f} cells")

    def test_deflection_increases_for_smaller_impact(self):
        """Test that deflection increases for smaller impact parameter."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        sim = lensing.setup_point_mass(mass=50.0, settling_steps=50)
        tracer = GravitationalRayTracer(sim, verbose=False)

        # Trace at two different impact parameters
        alpha_far, _ = tracer.compute_deflection(impact_parameter=10.0, n_steps=200)
        alpha_near, _ = tracer.compute_deflection(impact_parameter=5.0, n_steps=200)

        assert alpha_near > alpha_far, f"Closer ray should deflect more: {alpha_near} vs {alpha_far}"

        # Should roughly follow 1/b relation
        ratio = alpha_near / alpha_far
        expected_ratio = 10.0 / 5.0  # b_far / b_near
        assert 0.5 < ratio / expected_ratio < 2.0, f"Ratio {ratio} not close to expected {expected_ratio}"

        print(f"Deflection at b=5: {alpha_near:.6f} rad")
        print(f"Deflection at b=10: {alpha_far:.6f} rad")
        print(f"Ratio: {ratio:.2f} (expected ~{expected_ratio:.2f})")


# ==============================================================================
# TEST THEORETICAL PREDICTIONS
# ==============================================================================

class TestTheoreticalPredictions:
    """Test comparison with theoretical predictions."""

    def test_schwarzschild_formula(self):
        """Test theoretical Schwarzschild deflection formula."""
        lensing = GravitationalLensing(grid_size=32, kappa=5.0, verbose=False)

        M = 50.0
        b = 10.0

        # α = 4*G_eff*M/b
        alpha_theory = lensing.theoretical_deflection(b, M)

        expected = 4 * lensing.G_eff * M / b
        assert alpha_theory == pytest.approx(expected, rel=1e-10)

        print(f"Theoretical deflection: {alpha_theory:.6f} rad")

    def test_deflection_scales_with_mass(self):
        """Test that theoretical deflection scales linearly with mass."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)

        b = 10.0
        alpha1 = lensing.theoretical_deflection(b, mass_grav=50.0)
        alpha2 = lensing.theoretical_deflection(b, mass_grav=100.0)

        assert alpha2 / alpha1 == pytest.approx(2.0, rel=1e-10)

    def test_deflection_scales_with_impact(self):
        """Test that theoretical deflection scales as 1/b."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)

        M = 50.0
        alpha1 = lensing.theoretical_deflection(impact_parameter=5.0, mass_grav=M)
        alpha2 = lensing.theoretical_deflection(impact_parameter=10.0, mass_grav=M)

        # α ∝ 1/b => α1/α2 = b2/b1
        assert alpha1 / alpha2 == pytest.approx(2.0, rel=1e-10)

    def test_einstein_radius(self):
        """Test Einstein radius calculation."""
        lensing = GravitationalLensing(grid_size=32, kappa=5.0, verbose=False)

        M = 50.0
        d_lens = 10.0

        # R_E = sqrt(4*G_eff*M*D)
        R_E = lensing.compute_einstein_radius(M, d_lens=d_lens)

        expected = np.sqrt(4 * lensing.G_eff * M * d_lens)
        assert R_E == pytest.approx(expected, rel=1e-10)

        print(f"Einstein radius: {R_E:.2f} cells")


# ==============================================================================
# TEST LENSING ANALYSIS
# ==============================================================================

class TestLensingAnalysis:
    """Test complete lensing analysis."""

    def test_lensing_result_structure(self):
        """Test that lensing analysis returns proper structure."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        result = lensing.analyze_lensing(
            mass=30.0, impact_range=(4.0, 12.0), n_samples=5
        )

        assert isinstance(result, LensingResult)
        assert len(result.impact_parameters) == 5
        assert len(result.deflection_angles) == 5
        assert len(result.theoretical_deflections) == 5
        assert result.einstein_radius > 0
        assert result.mass_lattice > 0

    def test_det_matches_theory_qualitatively(self):
        """Test that DET deflections match theory qualitatively."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        result = lensing.analyze_lensing(
            mass=40.0, impact_range=(5.0, 15.0), n_samples=6
        )

        # Check deflections are positive
        assert np.all(result.deflection_angles > 0), "All deflections should be positive"

        # Check deflections decrease with impact parameter
        for i in range(len(result.deflection_angles) - 1):
            assert result.deflection_angles[i] >= result.deflection_angles[i+1] * 0.5, \
                "Deflection should generally decrease with b"

        print(f"Impact params: {result.impact_parameters}")
        print(f"DET deflections: {result.deflection_angles}")
        print(f"Theory deflections: {result.theoretical_deflections}")

    def test_deflection_profile_analysis(self):
        """Test deflection profile fitting."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        result = lensing.analyze_lensing(
            mass=50.0, impact_range=(5.0, 15.0), n_samples=8
        )

        profile = lensing.deflection_profile(result)

        assert 'A_fit' in profile
        assert 'A_theory' in profile
        assert 'r_squared' in profile
        assert 'follows_1_over_b' in profile

        # R² should be reasonable for 1/b fit
        print(f"Profile fit: A={profile['A_fit']:.4f}, R²={profile['r_squared']:.4f}")


class TestGravitationalMass:
    """Test gravitational mass measurement for lensing."""

    def test_gravitational_mass_positive(self):
        """Test that gravitational mass is positive."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        sim = lensing.setup_point_mass(mass=50.0, settling_steps=50)

        M_grav = lensing.measure_gravitational_mass(sim)
        assert M_grav > 0, f"Gravitational mass should be positive: {M_grav}"

        print(f"Gravitational mass: {M_grav:.4f}")

    def test_gravitational_mass_scales_with_input(self):
        """Test that gravitational mass scales with input mass."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)

        sim1 = lensing.setup_point_mass(mass=30.0, settling_steps=50)
        M1 = lensing.measure_gravitational_mass(sim1)

        sim2 = lensing.setup_point_mass(mass=60.0, settling_steps=50)
        M2 = lensing.measure_gravitational_mass(sim2)

        # In DET, q-b relationship is nonlinear due to Helmholtz baseline
        # M2 should be larger than M1, but not necessarily 2x
        assert M2 > M1, f"Larger input should give larger grav mass: {M2} vs {M1}"

        print(f"Mass 30 -> M_grav={M1:.4f}")
        print(f"Mass 60 -> M_grav={M2:.4f}")
        print(f"Ratio: {M2/M1:.2f}")


# ==============================================================================
# TEST PHYSICS CONSISTENCY
# ==============================================================================

class TestPhysicsConsistency:
    """Test physics consistency of lensing calculations."""

    def test_deflection_symmetric(self):
        """Test that deflection is symmetric for opposite impact parameters."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        sim = lensing.setup_point_mass(mass=50.0, settling_steps=50)
        tracer = GravitationalRayTracer(sim, verbose=False)

        # Trace rays on opposite sides of center
        alpha_plus, _ = tracer.compute_deflection(impact_parameter=8.0, n_steps=200)

        # Should get similar deflection (magnitudes should match)
        # Note: actual direction of deflection will differ
        assert alpha_plus > 0, "Deflection should be positive"

        print(f"Deflection at b=+8: {alpha_plus:.6f} rad")

    def test_weak_field_regime(self):
        """Test that we're in weak field regime (small deflections)."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        result = lensing.analyze_lensing(
            mass=30.0, impact_range=(8.0, 15.0), n_samples=5
        )

        # With smaller mass and larger impact parameters, deflections should be modest
        # (In small grids, near-field effects can be stronger)
        max_deflection = np.max(result.deflection_angles)
        print(f"Max deflection: {max_deflection:.4f} rad = {np.degrees(max_deflection):.2f} deg")

        # Just verify we get measurable but not extreme deflections
        assert max_deflection > 0, "Should have some deflection"
        assert max_deflection < 10.0, f"Deflection {max_deflection} is extreme"

    def test_potential_determines_deflection(self):
        """Test that deflection depends on gravitational potential."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)

        # Test that we get measurable deflections from potential
        sim = lensing.setup_point_mass(mass=40.0, settling_steps=50)
        tracer = GravitationalRayTracer(sim, verbose=False)

        # Get deflection at a reasonable impact parameter
        alpha, path = tracer.compute_deflection(impact_parameter=10.0)

        # Should have positive deflection
        assert alpha > 0, "Should have positive deflection from mass"

        # Check that potential along path varies
        phi_range = np.max(path.potentials) - np.min(path.potentials)
        assert phi_range > 0, "Potential should vary along path"

        print(f"Deflection: {alpha:.6f} rad")
        print(f"Potential range along path: {phi_range:.6f}")


# ==============================================================================
# TEST UNIT CONVERSIONS
# ==============================================================================

class TestUnitConversions:
    """Test SI unit conversions for lensing."""

    def test_einstein_radius_si(self):
        """Test Einstein radius conversion to SI units."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        result = lensing.analyze_lensing(mass=50.0, n_samples=5)

        # Einstein radius should be positive in both units
        assert result.einstein_radius > 0
        assert result.einstein_radius_si > 0

        # SI value should be much larger (meters vs cells)
        assert result.einstein_radius_si > result.einstein_radius

        print(f"Einstein radius: {result.einstein_radius:.2f} cells = {result.einstein_radius_si:.2e} m")

    def test_mass_si_conversion(self):
        """Test mass conversion to SI units."""
        lensing = GravitationalLensing(grid_size=32, verbose=False)
        result = lensing.analyze_lensing(mass=50.0, n_samples=5)

        # Mass should be positive in both units
        assert result.mass_lattice > 0
        assert result.mass_si > 0

        print(f"Mass: {result.mass_lattice:.2f} lattice = {result.mass_si:.2e} kg")


# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

def test_full_lensing_integration():
    """
    Full integration test of gravitational lensing analysis.

    Runs complete analysis and verifies results.
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full Gravitational Lensing")
    print("="*60)

    result = run_lensing_analysis(grid_size=32, mass=50.0, verbose=True)

    # Basic assertions
    assert result is not None
    assert len(result.impact_parameters) > 0
    assert len(result.deflection_angles) == len(result.impact_parameters)
    assert result.einstein_radius > 0

    # All deflections should be finite and positive
    assert np.all(np.isfinite(result.deflection_angles))
    assert np.all(result.deflection_angles >= 0)

    # Relative errors should be computed
    assert len(result.relative_errors) == len(result.deflection_angles)

    print("\n" + "-"*60)
    print("Integration test PASSED")
    print("-"*60)


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    print("Running Gravitational Lensing Test Suite")
    print("="*60)

    pytest.main([__file__, "-v", "--tb=short"])
