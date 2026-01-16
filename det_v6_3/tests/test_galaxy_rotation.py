"""
Test Suite for Galaxy Rotation Curves
=====================================

Tests the DET v6.4 Galaxy Rotation Curves feature:
Fit SPARC database observations.

Test Categories:
1. Data structures and sample data
2. Mass distribution models
3. DET rotation curve computation
4. Fitting and analysis
"""

import numpy as np
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'calibration'))

from galaxy_rotation_curves import (
    GalaxyObservation,
    RotationCurveFit,
    load_sample_galaxies,
    ExponentialDisk,
    NFWHalo,
    CombinedMassModel,
    DETRotationModel,
    RotationCurveAnalyzer,
    run_sparc_analysis,
    KPC, KMS, G_SI, M_SUN
)


# ==============================================================================
# TEST DATA STRUCTURES
# ==============================================================================

class TestGalaxyObservation:
    """Test GalaxyObservation data class."""

    def test_creation(self):
        """Test basic creation of galaxy observation."""
        galaxy = GalaxyObservation(
            name="TestGalaxy",
            radius_kpc=np.array([1.0, 2.0, 3.0]),
            v_obs=np.array([50.0, 80.0, 100.0]),
            v_err=np.array([5.0, 5.0, 5.0]),
            distance_mpc=10.0
        )

        assert galaxy.name == "TestGalaxy"
        assert galaxy.n_points == 3
        assert galaxy.r_max == 3.0

    def test_sample_galaxies_load(self):
        """Test loading sample SPARC galaxies."""
        galaxies = load_sample_galaxies()

        assert len(galaxies) > 0
        assert "NGC2403" in galaxies
        assert "UGC128" in galaxies
        assert "DDO154" in galaxies

        # Check NGC2403 properties
        ngc2403 = galaxies["NGC2403"]
        assert ngc2403.n_points > 10
        assert ngc2403.stellar_mass_msun > 1e9
        assert ngc2403.disk_scale_kpc > 0

        print(f"Loaded {len(galaxies)} sample galaxies")
        for name, gal in galaxies.items():
            print(f"  {name}: {gal.n_points} points, M*={gal.stellar_mass_msun:.1e} Msun")

    def test_sample_data_validity(self):
        """Verify sample data has valid values."""
        galaxies = load_sample_galaxies()

        for name, gal in galaxies.items():
            # Radii should be positive and increasing
            assert np.all(gal.radius_kpc > 0), f"{name}: radii should be positive"
            assert np.all(np.diff(gal.radius_kpc) > 0), f"{name}: radii should be increasing"

            # Velocities should be positive
            assert np.all(gal.v_obs > 0), f"{name}: velocities should be positive"

            # Errors should be positive
            assert np.all(gal.v_err > 0), f"{name}: errors should be positive"

            # Array lengths should match
            assert len(gal.radius_kpc) == len(gal.v_obs) == len(gal.v_err)


# ==============================================================================
# TEST MASS MODELS
# ==============================================================================

class TestExponentialDisk:
    """Test exponential disk mass model."""

    def test_creation(self):
        """Test disk model creation."""
        disk = ExponentialDisk(total_mass_msun=1e10, scale_length_kpc=3.0)

        assert disk.M_total == 1e10
        assert disk.R_d == 3.0

    def test_enclosed_mass_limits(self):
        """Test enclosed mass at limiting cases."""
        disk = ExponentialDisk(total_mass_msun=1e10, scale_length_kpc=3.0)

        # At r=0, enclosed mass should be 0
        assert disk.enclosed_mass(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-6)

        # At r >> R_d, enclosed mass should approach total mass
        M_far = disk.enclosed_mass(np.array([100.0]))[0]
        assert M_far == pytest.approx(1e10, rel=0.01)

    def test_enclosed_mass_increases(self):
        """Test that enclosed mass increases with radius."""
        disk = ExponentialDisk(total_mass_msun=1e10, scale_length_kpc=3.0)

        r = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        M = disk.enclosed_mass(r)

        assert np.all(np.diff(M) > 0), "Enclosed mass should increase with radius"

    def test_rotation_velocity(self):
        """Test rotation velocity computation."""
        disk = ExponentialDisk(total_mass_msun=1e11, scale_length_kpc=3.0)

        r = np.array([3.0, 6.0, 10.0])
        v = disk.rotation_velocity(r)

        # Velocities should be positive and in reasonable range
        assert np.all(v > 0)
        assert np.all(v < 500)  # Most galaxy rotation < 500 km/s

        print(f"Exponential disk rotation velocities: {v} km/s")


class TestNFWHalo:
    """Test NFW dark matter halo model."""

    def test_creation(self):
        """Test NFW halo creation."""
        halo = NFWHalo(virial_mass_msun=1e12, concentration=10.0, virial_radius_kpc=200.0)

        assert halo.M_vir == 1e12
        assert halo.c == 10.0
        assert halo.R_vir == 200.0
        assert halo.R_s == 20.0  # R_vir / c

    def test_enclosed_mass_limits(self):
        """Test enclosed mass at limiting cases."""
        halo = NFWHalo(virial_mass_msun=1e12, concentration=10.0, virial_radius_kpc=200.0)

        # At small r, enclosed mass should be small (NFW has cusp, not zero)
        M_small = halo.enclosed_mass(np.array([0.001]))[0]
        assert M_small < 1e6, f"Mass at r=0.001 kpc should be small: {M_small:.0f}"

        # At r=R_vir, enclosed mass should be M_vir
        M_vir = halo.enclosed_mass(np.array([200.0]))[0]
        assert M_vir == pytest.approx(1e12, rel=0.01)

    def test_rotation_velocity_profile(self):
        """Test NFW produces rising then falling rotation curve."""
        halo = NFWHalo(virial_mass_msun=1e12, concentration=10.0, virial_radius_kpc=200.0)

        r = np.logspace(0, 2.5, 30)  # 1 to ~300 kpc
        v = halo.rotation_velocity(r)

        # Velocity should peak somewhere and then decline
        v_max_idx = np.argmax(v)
        assert v_max_idx > 0, "Peak should not be at first point"
        assert v_max_idx < len(v) - 1, "Peak should not be at last point"

        print(f"NFW peak velocity: {np.max(v):.1f} km/s at r={r[v_max_idx]:.1f} kpc")


class TestCombinedMassModel:
    """Test combined mass model."""

    def test_combined_model(self):
        """Test disk + halo combined model."""
        disk = ExponentialDisk(total_mass_msun=5e10, scale_length_kpc=3.0)
        halo = NFWHalo(virial_mass_msun=5e11, concentration=10.0, virial_radius_kpc=150.0)

        combined = CombinedMassModel([("disk", disk), ("halo", halo)])

        r = np.array([5.0, 10.0, 20.0])

        # Combined enclosed mass should be sum of components
        M_combined = combined.enclosed_mass(r)
        M_disk = disk.enclosed_mass(r)
        M_halo = halo.enclosed_mass(r)

        np.testing.assert_allclose(M_combined, M_disk + M_halo)

    def test_component_velocities(self):
        """Test individual component velocity retrieval."""
        disk = ExponentialDisk(total_mass_msun=5e10, scale_length_kpc=3.0)
        halo = NFWHalo(virial_mass_msun=5e11, concentration=10.0, virial_radius_kpc=150.0)

        combined = CombinedMassModel([("disk", disk), ("halo", halo)])

        r = np.array([5.0, 10.0])
        v_components = combined.component_velocities(r)

        assert "disk" in v_components
        assert "halo" in v_components
        assert len(v_components["disk"]) == len(r)


# ==============================================================================
# TEST DET MODEL
# ==============================================================================

class TestDETRotationModel:
    """Test DET-based rotation curve model."""

    def test_model_creation(self):
        """Test DET model creation."""
        model = DETRotationModel(kappa=5.0, eta=0.968)

        assert model.kappa == 5.0
        assert model.eta == 0.968
        assert model.G_eff_lattice == pytest.approx(0.968 * 5.0 / (4 * np.pi), rel=1e-6)

    def test_rotation_from_mass(self):
        """Test rotation velocity computation from enclosed mass."""
        model = DETRotationModel()

        r_kpc = np.array([5.0, 10.0, 15.0])
        M_msun = np.array([1e10, 3e10, 5e10])

        v = model.rotation_velocity_from_mass(r_kpc, M_msun)

        # Should match Newtonian prediction
        r_m = r_kpc * KPC
        M_kg = M_msun * M_SUN
        v_expected = np.sqrt(G_SI * M_kg / r_m) / KMS

        np.testing.assert_allclose(v, v_expected, rtol=1e-6)

    def test_fit_galaxy(self):
        """Test fitting to a sample galaxy."""
        model = DETRotationModel()
        galaxies = load_sample_galaxies()

        fit = model.fit_galaxy(galaxies["NGC2403"])

        assert isinstance(fit, RotationCurveFit)
        assert fit.galaxy_name == "NGC2403"
        assert fit.chi_squared >= 0
        assert fit.reduced_chi_squared >= 0
        assert len(fit.v_model) == len(fit.v_obs)

        print(f"NGC2403 fit: chi2={fit.chi_squared:.1f}, red_chi2={fit.reduced_chi_squared:.2f}")

    def test_fit_with_dark_matter(self):
        """Test fitting with dark matter halo."""
        model = DETRotationModel()
        galaxies = load_sample_galaxies()

        fit_no_dm = model.fit_galaxy(galaxies["DDO154"], include_dark_matter=False)
        fit_dm = model.fit_galaxy(galaxies["DDO154"], include_dark_matter=True)

        # Dark matter should improve fit for dwarf galaxy
        print(f"DDO154 without DM: chi2={fit_no_dm.reduced_chi_squared:.2f}")
        print(f"DDO154 with DM: chi2={fit_dm.reduced_chi_squared:.2f}")

        # Both fits should complete
        assert fit_no_dm.chi_squared >= 0
        assert fit_dm.chi_squared >= 0

    def test_optimize_ml_ratio(self):
        """Test M/L ratio optimization."""
        model = DETRotationModel()
        galaxies = load_sample_galaxies()

        opt_ml, fit = model.optimize_ml_ratio(galaxies["NGC2403"])

        assert 0.1 <= opt_ml <= 5.0
        assert fit.parameters['disk_ml_ratio'] == opt_ml

        print(f"NGC2403 optimal M/L = {opt_ml:.2f}")


# ==============================================================================
# TEST ANALYZER
# ==============================================================================

class TestRotationCurveAnalyzer:
    """Test the full analyzer."""

    def test_analyzer_creation(self):
        """Test analyzer creation and galaxy loading."""
        analyzer = RotationCurveAnalyzer()
        analyzer.load_galaxies()

        assert len(analyzer.galaxies) > 0

    def test_single_galaxy_analysis(self):
        """Test analyzing a single galaxy."""
        analyzer = RotationCurveAnalyzer()
        analyzer.load_galaxies()

        results = analyzer.analyze_galaxy("NGC2403", verbose=False)

        assert "DET_baryons" in results
        assert "DET_optimized" in results
        assert "DET_with_DM" in results

    def test_all_galaxy_analysis(self):
        """Test analyzing all galaxies."""
        analyzer = RotationCurveAnalyzer()
        analyzer.load_galaxies()
        results = analyzer.analyze_all(verbose=False)

        assert len(results) == len(analyzer.galaxies)

        for name in analyzer.galaxies:
            assert name in results
            assert "DET_baryons" in results[name]

    def test_summary_report(self):
        """Test summary report generation."""
        analyzer = RotationCurveAnalyzer()
        analyzer.load_galaxies()
        analyzer.analyze_all(verbose=False)

        report = analyzer.summary_report()

        assert len(report) > 0
        assert "DET" in report
        assert "Chi-sq" in report

    def test_needs_dark_matter(self):
        """Test dark matter necessity assessment."""
        analyzer = RotationCurveAnalyzer()
        analyzer.load_galaxies()

        # DDO154 (dwarf) typically needs dark matter
        # NGC2403 (spiral) may or may not need it

        needs_dm_ddo = analyzer.needs_dark_matter("DDO154", threshold=2.0)
        needs_dm_ngc = analyzer.needs_dark_matter("NGC2403", threshold=2.0)

        print(f"DDO154 needs DM: {needs_dm_ddo}")
        print(f"NGC2403 needs DM: {needs_dm_ngc}")

        # Both should return a boolean-like value
        assert needs_dm_ddo in [True, False] or isinstance(needs_dm_ddo, (bool, np.bool_))
        assert needs_dm_ngc in [True, False] or isinstance(needs_dm_ngc, (bool, np.bool_))


# ==============================================================================
# TEST PHYSICS CONSISTENCY
# ==============================================================================

class TestPhysicsConsistency:
    """Test physics consistency of rotation curve calculations."""

    def test_keplerian_point_mass(self):
        """Test that point mass gives Keplerian v ∝ 1/sqrt(r)."""
        model = DETRotationModel()

        M = 1e11  # 10^11 solar masses
        r = np.array([5.0, 10.0, 20.0, 40.0])

        v = model.rotation_velocity_from_mass(r, np.full_like(r, M))

        # For point mass: v ∝ 1/sqrt(r)
        # v1 * sqrt(r1) = v2 * sqrt(r2)
        v_sqrt_r = v * np.sqrt(r)

        # Should all be approximately equal
        assert np.std(v_sqrt_r) / np.mean(v_sqrt_r) < 0.01

    def test_flat_rotation_curve(self):
        """Test conditions for flat rotation curve."""
        # For v = constant, need M(<r) ∝ r
        # This is the isothermal sphere case

        model = DETRotationModel()

        r = np.array([5.0, 10.0, 15.0, 20.0])
        M = r * 1e10  # M proportional to r

        v = model.rotation_velocity_from_mass(r, M)

        # Velocities should be approximately constant
        v_variation = (np.max(v) - np.min(v)) / np.mean(v)
        assert v_variation < 0.05, f"Velocity variation {v_variation:.2%} > 5%"

        print(f"Flat rotation test: v = {v} km/s, variation = {v_variation:.2%}")

    def test_velocity_magnitude_realistic(self):
        """Test that computed velocities are in realistic range."""
        galaxies = load_sample_galaxies()
        model = DETRotationModel()

        for name, galaxy in galaxies.items():
            fit = model.fit_galaxy(galaxy, include_dark_matter=True)

            # All velocities should be between 10 and 500 km/s
            assert np.all(fit.v_model > 0), f"{name}: negative velocities"
            assert np.all(fit.v_model < 500), f"{name}: unrealistically high velocities"


# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

def test_full_sparc_analysis_integration():
    """
    Full integration test of SPARC analysis.

    Runs complete analysis pipeline with all sample galaxies.
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full SPARC Analysis")
    print("="*60)

    analyzer = run_sparc_analysis(verbose=True)

    # Check that all galaxies were analyzed
    assert len(analyzer.results) == len(analyzer.galaxies)

    # Check that all required models were fit
    for name, results in analyzer.results.items():
        assert "DET_baryons" in results
        assert "DET_optimized" in results
        assert "DET_with_DM" in results

        # All fits should have valid statistics
        for model_name, fit in results.items():
            assert fit.chi_squared >= 0
            assert not np.isnan(fit.chi_squared)
            assert not np.isnan(fit.rms_residual)

    print("\n" + "-"*60)
    print("Integration test PASSED")
    print("-"*60)


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    print("Running Galaxy Rotation Curve Test Suite")
    print("="*60)

    pytest.main([__file__, "-v", "--tb=short"])
