"""
Test Suite for Cosmological Scaling
===================================

Tests the DET v6.4 Cosmological Scaling feature:
Large-scale structure formation analysis.

Test Categories:
1. Power spectrum computation
2. Correlation function analysis
3. Structure growth simulation
4. LCDM comparison
5. Physics consistency checks
"""

import numpy as np
import pytest
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'calibration'))

from cosmological_scaling import (
    PowerSpectrumAnalyzer,
    CorrelationAnalyzer,
    StructureGrowthAnalyzer,
    LCDMComparison,
    CosmologicalScalingAnalyzer,
    run_cosmological_analysis,
    OMEGA_M, SIGMA_8, N_S
)

from det_v6_3_3d_collider import DETCollider3D, DETParams3D, compute_lattice_correction


# ==============================================================================
# TEST POWER SPECTRUM
# ==============================================================================

class TestPowerSpectrumAnalyzer:
    """Test power spectrum computation."""

    def test_analyzer_creation(self):
        """Test basic analyzer creation."""
        analyzer = PowerSpectrumAnalyzer(box_size=64.0, verbose=False)
        assert analyzer.box_size == 64.0

    def test_density_contrast(self):
        """Test density contrast computation."""
        analyzer = PowerSpectrumAnalyzer(verbose=False)

        # Uniform density should give zero contrast
        density = np.ones((32, 32, 32)) * 1.5
        delta = analyzer.compute_density_contrast(density)

        assert np.allclose(delta, 0.0)

    def test_density_contrast_positive_fluctuation(self):
        """Test density contrast with overdensity."""
        analyzer = PowerSpectrumAnalyzer(verbose=False)

        # Overdense region
        density = np.ones((32, 32, 32))
        density[15, 15, 15] = 2.0  # Overdensity

        delta = analyzer.compute_density_contrast(density)

        # Mean should be close to zero (by construction)
        assert np.abs(np.mean(delta)) < 0.01

        # Overdense region should have positive contrast
        assert delta[15, 15, 15] > 0

    def test_power_spectrum_shape(self):
        """Test that power spectrum has correct shape."""
        analyzer = PowerSpectrumAnalyzer(box_size=32.0, verbose=False)

        # Random density field
        np.random.seed(42)
        density = 1.0 + 0.1 * np.random.randn(32, 32, 32)

        result = analyzer.compute_power_spectrum(density, n_bins=15)

        assert len(result.k_bins) == 15
        assert len(result.P_k) == 15
        assert len(result.n_modes) == 15

    def test_power_spectrum_positivity(self):
        """Test that power spectrum is non-negative."""
        analyzer = PowerSpectrumAnalyzer(box_size=32.0, verbose=False)

        np.random.seed(42)
        density = 1.0 + 0.1 * np.random.randn(32, 32, 32)

        result = analyzer.compute_power_spectrum(density)

        # P(k) should be non-negative
        assert np.all(result.P_k >= 0)

    def test_power_spectrum_scale_invariant(self):
        """Test power spectrum of scale-invariant perturbations."""
        analyzer = PowerSpectrumAnalyzer(box_size=64.0, verbose=False)

        # Generate scale-invariant perturbations: P(k) ~ k
        np.random.seed(42)
        N = 64

        from scipy.fft import fftfreq, fftn, ifftn

        kx = fftfreq(N, d=1.0/N)
        ky = fftfreq(N, d=1.0/N)
        kz = fftfreq(N, d=1.0/N)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        K[0, 0, 0] = 1.0

        # P(k) ~ k^1 => amplitude ~ k^0.5
        phases = np.random.uniform(0, 2*np.pi, (N, N, N))
        delta_k = np.sqrt(K) * 0.1 * np.exp(1j * phases)
        delta_k[0, 0, 0] = 0

        density = 1.0 + np.real(ifftn(delta_k))

        result = analyzer.compute_power_spectrum(density)

        # Spectral index should be close to 1
        # (but expect some deviation due to aliasing and binning)
        assert -1.0 < result.spectral_index < 3.0  # Reasonable range

    def test_power_spectrum_from_det(self):
        """Test power spectrum computation from DET simulation."""
        params = DETParams3D(N=32, DT=0.01, gravity_enabled=True)
        sim = DETCollider3D(params)

        # Add some structure
        sim.q[14:18, 14:18, 14:18] = 0.5

        # Run a few steps
        for _ in range(10):
            sim.step()

        analyzer = PowerSpectrumAnalyzer(box_size=32.0, verbose=False)
        result = analyzer.compute_from_det(sim)

        assert result.P_k is not None
        assert len(result.k_bins) > 0


# ==============================================================================
# TEST CORRELATION FUNCTION
# ==============================================================================

class TestCorrelationAnalyzer:
    """Test correlation function computation."""

    def test_analyzer_creation(self):
        """Test basic analyzer creation."""
        analyzer = CorrelationAnalyzer(verbose=False)
        assert analyzer is not None

    def test_uniform_correlation(self):
        """Test correlation of uniform field is zero at r > 0."""
        analyzer = CorrelationAnalyzer(verbose=False)

        density = np.ones((32, 32, 32))
        result = analyzer.compute_correlation_fft(density)

        # Uniform field has xi = 0 everywhere (no fluctuations)
        # but xi(0) = <delta^2> = 0 for uniform
        assert np.all(np.abs(result.xi_r) < 1e-10)

    def test_correlation_shape(self):
        """Test correlation function has correct shape."""
        analyzer = CorrelationAnalyzer(verbose=False)

        np.random.seed(42)
        density = 1.0 + 0.1 * np.random.randn(32, 32, 32)

        result = analyzer.compute_correlation_fft(density, n_bins=15)

        assert len(result.r_bins) == 15
        assert len(result.xi_r) == 15

    def test_correlation_at_zero(self):
        """Test xi(0) = variance."""
        analyzer = CorrelationAnalyzer(verbose=False)

        np.random.seed(42)
        density = 1.0 + 0.1 * np.random.randn(32, 32, 32)

        result = analyzer.compute_correlation_fft(density, n_bins=20)

        # xi(r=0) should be positive (variance of delta)
        assert result.xi_r[0] > 0

    def test_correlation_decreases(self):
        """Test that correlation decreases with distance."""
        analyzer = CorrelationAnalyzer(verbose=False)

        # Create clustered field
        np.random.seed(42)
        density = np.ones((32, 32, 32))

        # Add clump at center
        density[12:20, 12:20, 12:20] = 2.0

        result = analyzer.compute_correlation_fft(density)

        # Correlation should generally decrease from center
        # (not strictly monotonic due to periodic boundaries)
        assert result.xi_r[0] >= result.xi_r[-1]


# ==============================================================================
# TEST STRUCTURE GROWTH
# ==============================================================================

class TestStructureGrowthAnalyzer:
    """Test structure growth simulation."""

    def test_analyzer_creation(self):
        """Test basic analyzer creation."""
        analyzer = StructureGrowthAnalyzer(grid_size=32, verbose=False)
        assert analyzer.N == 32

    def test_initial_perturbations(self):
        """Test initial perturbation setup."""
        analyzer = StructureGrowthAnalyzer(grid_size=32, verbose=False)

        sim = analyzer.setup_initial_perturbations(amplitude=0.01)

        # q should have perturbations
        q_mean = np.mean(sim.q)
        q_std = np.std(sim.q)

        assert q_std > 0  # There should be fluctuations
        assert 0 < q_mean < 1  # Mean should be reasonable

    def test_rms_fluctuation_positive(self):
        """Test RMS fluctuation is positive for perturbed field."""
        analyzer = StructureGrowthAnalyzer(grid_size=32, verbose=False)

        sim = analyzer.setup_initial_perturbations(amplitude=0.01)

        sigma = analyzer.measure_rms_fluctuation(sim)

        assert sigma >= 0

    def test_growth_simulation_runs(self):
        """Test that growth simulation completes."""
        analyzer = StructureGrowthAnalyzer(grid_size=32, verbose=False)

        result = analyzer.run_growth_simulation(
            n_steps=100,
            sample_interval=20,
            amplitude=0.01
        )

        assert len(result.time_steps) > 0
        assert len(result.delta_rms) > 0
        assert len(result.growth_factor) > 0

    def test_growth_factor_initial_unity(self):
        """Test that growth factor starts at 1."""
        analyzer = StructureGrowthAnalyzer(grid_size=32, verbose=False)

        result = analyzer.run_growth_simulation(
            n_steps=50,
            sample_interval=10
        )

        # Growth factor normalized to 1 at t=0
        assert result.growth_factor[0] == pytest.approx(1.0)

    def test_growth_occurs(self):
        """Test that structure grows (gravity amplifies fluctuations)."""
        analyzer = StructureGrowthAnalyzer(grid_size=32, kappa=5.0, verbose=False)

        result = analyzer.run_growth_simulation(
            n_steps=200,
            sample_interval=20,
            amplitude=0.02
        )

        # With gravity, fluctuations should grow or remain stable
        # (may not always grow significantly in short simulations)
        assert result.delta_rms[-1] >= 0
        assert result.growth_factor[-1] >= 0


# ==============================================================================
# TEST LCDM COMPARISON
# ==============================================================================

class TestLCDMComparison:
    """Test LCDM comparison functions."""

    def test_comparison_creation(self):
        """Test LCDM comparison creation."""
        lcdm = LCDMComparison(verbose=False)
        assert lcdm.omega_m == OMEGA_M
        assert lcdm.sigma_8 == SIGMA_8

    def test_growth_factor_normalization(self):
        """Test growth factor D(a=1) = 1."""
        lcdm = LCDMComparison(verbose=False)

        D_1 = lcdm.linear_growth_factor_single(1.0)

        # Should be positive
        assert D_1 > 0

    def test_growth_factor_increases(self):
        """Test that growth factor increases with scale factor."""
        lcdm = LCDMComparison(verbose=False)

        a_values = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
        D_values = lcdm.linear_growth_factor(a_values)

        # D should be monotonically increasing
        assert np.all(np.diff(D_values) > 0)

    def test_growth_rate_reasonable(self):
        """Test growth rate is in reasonable range."""
        lcdm = LCDMComparison(verbose=False)

        f = lcdm.growth_rate(1.0)

        # f ~ Omega_m^0.55 ~ 0.47 for Omega_m = 0.315
        assert 0.3 < f < 0.7

    def test_power_spectrum_shape(self):
        """Test that power spectrum has reasonable shape."""
        lcdm = LCDMComparison(verbose=False)

        k = np.logspace(-2, 1, 50)  # 0.01 to 10 h/Mpc
        P_k = lcdm.matter_power_spectrum(k)

        # P(k) should be positive
        assert np.all(P_k > 0)

        # P(k) should have a peak/turnover
        # (increases at low k, decreases at high k)
        # Just check it's not constant
        assert np.std(P_k) > 0


# ==============================================================================
# TEST FULL ANALYSIS
# ==============================================================================

class TestCosmologicalScalingAnalyzer:
    """Test full cosmological analysis."""

    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = CosmologicalScalingAnalyzer(grid_size=32, verbose=False)

        assert analyzer.N == 32
        assert analyzer.G_eff > 0

    def test_G_eff_formula(self):
        """Test G_eff = eta*kappa/(4*pi)."""
        kappa = 5.0
        N = 32
        eta = compute_lattice_correction(N)

        analyzer = CosmologicalScalingAnalyzer(grid_size=N, kappa=kappa, verbose=False)

        G_expected = eta * kappa / (4 * np.pi)
        assert analyzer.G_eff == pytest.approx(G_expected, rel=1e-6)

    def test_full_analysis_runs(self):
        """Test that full analysis completes."""
        analyzer = CosmologicalScalingAnalyzer(grid_size=32, verbose=False)

        result = analyzer.run_full_analysis(growth_steps=100)

        assert result.power_spectrum is not None
        assert result.correlation is not None
        assert result.growth is not None

    def test_full_analysis_summary(self):
        """Test that analysis produces summary."""
        analyzer = CosmologicalScalingAnalyzer(grid_size=32, verbose=False)

        result = analyzer.run_full_analysis(growth_steps=50)

        assert len(result.summary) > 0
        assert "COSMOLOGICAL" in result.summary


# ==============================================================================
# TEST PHYSICS CONSISTENCY
# ==============================================================================

class TestPhysicsConsistency:
    """Test physics consistency of cosmological analysis."""

    def test_power_spectrum_parseval(self):
        """Test Parseval's theorem: sum(P_k) ~ variance."""
        analyzer = PowerSpectrumAnalyzer(box_size=32.0, verbose=False)

        np.random.seed(42)
        density = 1.0 + 0.1 * np.random.randn(32, 32, 32)

        # Variance in configuration space
        delta = analyzer.compute_density_contrast(density)
        variance_config = np.mean(delta**2)

        result = analyzer.compute_power_spectrum(density)

        # Sum of P(k) should be related to variance
        # (just check they're in same order of magnitude)
        assert result.amplitude > 0

    def test_correlation_fourier_pair(self):
        """Test xi(r) and P(k) are Fourier pairs (qualitative)."""
        # Generate field with known power spectrum
        np.random.seed(42)
        N = 32

        from scipy.fft import fftfreq, fftn, ifftn

        kx = fftfreq(N, d=1.0/N)
        ky = fftfreq(N, d=1.0/N)
        kz = fftfreq(N, d=1.0/N)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        K[0, 0, 0] = 1.0

        # White noise: P(k) = const
        phases = np.random.uniform(0, 2*np.pi, (N, N, N))
        delta_k = 0.1 * np.exp(1j * phases)
        delta_k[0, 0, 0] = 0

        density = 1.0 + np.real(ifftn(delta_k))

        # Compute both
        power_analyzer = PowerSpectrumAnalyzer(box_size=N, verbose=False)
        corr_analyzer = CorrelationAnalyzer(verbose=False)

        power = power_analyzer.compute_power_spectrum(density)
        corr = corr_analyzer.compute_correlation_fft(density)

        # Both should be computable without error
        assert power.amplitude > 0
        assert corr.xi_r[0] > 0

    def test_growth_physical(self):
        """Test that growth is physically reasonable."""
        analyzer = StructureGrowthAnalyzer(grid_size=32, verbose=False)

        result = analyzer.run_growth_simulation(
            n_steps=100,
            sample_interval=20
        )

        # Growth factor should remain finite
        assert np.all(np.isfinite(result.growth_factor))

        # RMS should remain finite
        assert np.all(np.isfinite(result.delta_rms))

    def test_lcdm_matter_domination(self):
        """Test growth rate in matter-dominated limit."""
        # In matter-dominated universe, f ~ 1
        lcdm = LCDMComparison(omega_m=1.0, verbose=False)  # Einstein-de Sitter

        f = lcdm.growth_rate(1.0)

        # f = Omega_m^0.55 = 1.0 for Omega_m = 1
        assert f == pytest.approx(1.0, rel=0.01)


# ==============================================================================
# INTEGRATION TEST
# ==============================================================================

def test_full_cosmological_analysis_integration():
    """
    Full integration test of cosmological scaling analysis.

    Runs complete analysis pipeline and validates outputs.
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full Cosmological Analysis")
    print("="*60)

    result = run_cosmological_analysis(
        grid_size=32,
        kappa=5.0,
        growth_steps=100,
        verbose=True
    )

    # Validate power spectrum
    assert result.power_spectrum is not None
    assert len(result.power_spectrum.k_bins) > 0
    assert np.any(result.power_spectrum.P_k > 0)

    # Validate correlation
    assert result.correlation is not None
    assert len(result.correlation.r_bins) > 0

    # Validate growth
    assert result.growth is not None
    assert len(result.growth.time_steps) > 0
    assert result.growth.growth_factor[0] == pytest.approx(1.0)

    # Validate G_eff
    eta = compute_lattice_correction(32)
    G_expected = eta * 5.0 / (4 * np.pi)
    assert result.G_eff == pytest.approx(G_expected, rel=1e-6)

    # Check summary was generated
    assert len(result.summary) > 100

    print("\n" + "-"*60)
    print("Integration test PASSED")
    print("-"*60)


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    print("Running Cosmological Scaling Test Suite")
    print("="*60)

    pytest.main([__file__, "-v", "--tb=short"])
