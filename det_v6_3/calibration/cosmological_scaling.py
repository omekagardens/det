"""
DET v6.4 Cosmological Scaling: Large-Scale Structure Formation
==============================================================

Roadmap Item #4: Study how DET dynamics lead to cosmological structure
formation and compare with standard LCDM predictions.

Theoretical Framework
---------------------
In standard cosmology, structure grows from primordial density fluctuations:
- Power spectrum P(k) characterizes clustering at wavenumber k
- Correlation function xi(r) is Fourier transform of P(k)
- Linear growth: delta(a) = D(a) * delta_initial

In DET, gravity is sourced by structural debt rho = q - b:
- Effective G: G_eff = eta * kappa / (4*pi)
- Structure growth modified by DET dynamics
- Screening from baseline b may affect large-scale behavior

Key Observables
---------------
1. Matter Power Spectrum P(k)
2. Two-Point Correlation Function xi(r)
3. Structure Growth Rate f = d(ln D)/d(ln a)
4. Clustering Amplitude sigma_8

Reference: DET Theory Card v6.3, Section V (Gravity Module)
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D, compute_lattice_correction


# ==============================================================================
# PHYSICAL CONSTANTS (for cosmological context)
# ==============================================================================

# Cosmological parameters (Planck 2018)
H0_SI = 67.4  # km/s/Mpc
OMEGA_M = 0.315  # Matter density parameter
OMEGA_L = 0.685  # Dark energy density parameter
OMEGA_B = 0.0493  # Baryon density parameter
SIGMA_8 = 0.811  # Clustering amplitude at 8 Mpc/h
N_S = 0.965  # Scalar spectral index


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class PowerSpectrumResult:
    """Results from power spectrum computation."""
    k_bins: np.ndarray          # Wavenumber bins
    P_k: np.ndarray             # Power spectrum P(k)
    P_k_err: np.ndarray         # Standard error in each bin
    n_modes: np.ndarray         # Number of modes in each bin
    spectral_index: float       # Fitted spectral index n
    amplitude: float            # Amplitude at reference scale
    box_size: float             # Box size in lattice units


@dataclass
class CorrelationResult:
    """Results from correlation function computation."""
    r_bins: np.ndarray          # Separation bins
    xi_r: np.ndarray            # Correlation function xi(r)
    xi_r_err: np.ndarray        # Standard error
    correlation_length: float   # Characteristic correlation length
    power_law_index: float      # Power law index (xi ~ r^-gamma)


@dataclass
class GrowthResult:
    """Results from structure growth analysis."""
    time_steps: np.ndarray      # Time array
    delta_rms: np.ndarray       # RMS density fluctuation
    growth_factor: np.ndarray   # Normalized growth D(t)/D(0)
    growth_rate: float          # Linear growth rate f
    growth_exponent: float      # Power law exponent


@dataclass
class CosmologicalAnalysis:
    """Complete cosmological scaling analysis."""
    power_spectrum: PowerSpectrumResult
    correlation: CorrelationResult
    growth: GrowthResult
    G_eff: float
    kappa: float
    eta: float
    grid_size: int
    summary: str = ""


# ==============================================================================
# POWER SPECTRUM ANALYSIS
# ==============================================================================

class PowerSpectrumAnalyzer:
    """
    Compute matter power spectrum P(k) from density field.

    The power spectrum is defined as:
        <delta_k delta_k^*> = (2*pi)^3 * P(k) * delta_D(k-k')

    where delta_k is the Fourier transform of the density contrast:
        delta(x) = (rho(x) - rho_mean) / rho_mean
    """

    def __init__(self, box_size: float = 1.0, verbose: bool = True):
        """
        Initialize power spectrum analyzer.

        Parameters
        ----------
        box_size : float
            Physical box size (sets k normalization)
        verbose : bool
            Print progress information
        """
        self.box_size = box_size
        self.verbose = verbose

    def compute_density_contrast(self, density: np.ndarray) -> np.ndarray:
        """
        Compute density contrast field.

        delta = (rho - rho_mean) / rho_mean

        Parameters
        ----------
        density : np.ndarray
            3D density field

        Returns
        -------
        np.ndarray
            Density contrast field
        """
        rho_mean = np.mean(density)
        if rho_mean <= 0:
            rho_mean = 1e-10  # Avoid division by zero
        return (density - rho_mean) / rho_mean

    def compute_power_spectrum(self, density: np.ndarray,
                                n_bins: int = 20) -> PowerSpectrumResult:
        """
        Compute isotropic power spectrum P(k).

        Parameters
        ----------
        density : np.ndarray
            3D density field
        n_bins : int
            Number of k bins

        Returns
        -------
        PowerSpectrumResult
            Power spectrum results
        """
        N = density.shape[0]

        # Compute density contrast
        delta = self.compute_density_contrast(density)

        # FFT to get delta_k
        delta_k = fftn(delta) / N**3

        # Power = |delta_k|^2 * V where V = box_size^3
        power = np.abs(delta_k)**2 * self.box_size**3

        # Create k-grid
        k_fundamental = 2 * np.pi / self.box_size
        kx = fftfreq(N, d=1/N) * k_fundamental
        ky = fftfreq(N, d=1/N) * k_fundamental
        kz = fftfreq(N, d=1/N) * k_fundamental

        kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

        # Bin power spectrum by |k|
        k_min = k_fundamental
        k_max = k_fundamental * N / 2
        k_edges = np.linspace(k_min, k_max, n_bins + 1)
        k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])

        P_k = np.zeros(n_bins)
        P_k_err = np.zeros(n_bins)
        n_modes = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i+1])
            n_modes[i] = np.sum(mask)
            if n_modes[i] > 0:
                P_k[i] = np.mean(power[mask])
                P_k_err[i] = np.std(power[mask]) / np.sqrt(n_modes[i])

        # Fit power law P(k) ~ A * k^n
        valid = (P_k > 0) & (n_modes > 5)
        if np.sum(valid) > 3:
            try:
                log_k = np.log(k_bins[valid])
                log_P = np.log(P_k[valid])
                coeffs = np.polyfit(log_k, log_P, 1)
                spectral_index = coeffs[0]
                amplitude = np.exp(coeffs[1])
            except:
                spectral_index = 0.0
                amplitude = np.mean(P_k[valid])
        else:
            spectral_index = 0.0
            amplitude = np.mean(P_k[P_k > 0]) if np.any(P_k > 0) else 0.0

        if self.verbose:
            print(f"  Power spectrum: n = {spectral_index:.3f}, A = {amplitude:.3e}")

        return PowerSpectrumResult(
            k_bins=k_bins,
            P_k=P_k,
            P_k_err=P_k_err,
            n_modes=n_modes,
            spectral_index=spectral_index,
            amplitude=amplitude,
            box_size=self.box_size
        )

    def compute_from_det(self, sim: DETCollider3D,
                         use_q_minus_b: bool = True) -> PowerSpectrumResult:
        """
        Compute power spectrum from DET simulation.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation
        use_q_minus_b : bool
            If True, use gravitational density (q-b)
            If False, use resource density F

        Returns
        -------
        PowerSpectrumResult
            Power spectrum results
        """
        if use_q_minus_b:
            # Use gravitational source density
            density = sim.q - sim.b
        else:
            # Use resource density
            density = sim.F

        return self.compute_power_spectrum(density)


# ==============================================================================
# CORRELATION FUNCTION ANALYSIS
# ==============================================================================

class CorrelationAnalyzer:
    """
    Compute two-point correlation function xi(r).

    The correlation function is the Fourier transform of P(k):
        xi(r) = (1/2*pi^2) * integral(k^2 * P(k) * sin(kr)/(kr) dk)

    We compute it directly in configuration space for better accuracy
    at small separations.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize correlation analyzer.
        """
        self.verbose = verbose

    def compute_correlation_fft(self, density: np.ndarray,
                                 n_bins: int = 25) -> CorrelationResult:
        """
        Compute correlation function using FFT convolution.

        xi(r) = <delta(x) * delta(x+r)>

        Parameters
        ----------
        density : np.ndarray
            3D density field
        n_bins : int
            Number of radial bins

        Returns
        -------
        CorrelationResult
            Correlation function results
        """
        N = density.shape[0]

        # Compute density contrast
        rho_mean = np.mean(density)
        if rho_mean <= 0:
            rho_mean = 1e-10
        delta = (density - rho_mean) / rho_mean

        # Correlation via FFT: xi = IFFT(|FFT(delta)|^2)
        delta_k = fftn(delta)
        correlation_3d = np.real(ifftn(np.abs(delta_k)**2)) / N**3

        # Create distance grid
        x = np.arange(N)
        x[x > N//2] -= N  # Wrap for periodic boundaries
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2)

        # Bin by radius
        r_max = N // 2
        r_edges = np.linspace(0, r_max, n_bins + 1)
        r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])

        xi_r = np.zeros(n_bins)
        xi_r_err = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (R >= r_edges[i]) & (R < r_edges[i+1])
            n_pairs = np.sum(mask)
            if n_pairs > 0:
                xi_r[i] = np.mean(correlation_3d[mask])
                xi_r_err[i] = np.std(correlation_3d[mask]) / np.sqrt(n_pairs)

        # Find correlation length (where xi = 1/e of xi(0))
        xi_0 = xi_r[0] if xi_r[0] > 0 else 1.0
        threshold = xi_0 / np.e

        # Find where xi crosses threshold
        try:
            cross_idx = np.where(xi_r < threshold)[0]
            if len(cross_idx) > 0:
                correlation_length = r_bins[cross_idx[0]]
            else:
                correlation_length = r_bins[-1]
        except:
            correlation_length = 0.0

        # Fit power law xi ~ r^(-gamma) in intermediate range
        valid = (r_bins > 1) & (r_bins < N//4) & (xi_r > 0)
        if np.sum(valid) > 3:
            try:
                log_r = np.log(r_bins[valid])
                log_xi = np.log(xi_r[valid])
                coeffs = np.polyfit(log_r, log_xi, 1)
                power_law_index = -coeffs[0]  # gamma is positive
            except:
                power_law_index = 0.0
        else:
            power_law_index = 0.0

        if self.verbose:
            print(f"  Correlation length: {correlation_length:.2f}")
            print(f"  Power law index gamma: {power_law_index:.3f}")

        return CorrelationResult(
            r_bins=r_bins,
            xi_r=xi_r,
            xi_r_err=xi_r_err,
            correlation_length=correlation_length,
            power_law_index=power_law_index
        )


# ==============================================================================
# STRUCTURE GROWTH ANALYSIS
# ==============================================================================

class StructureGrowthAnalyzer:
    """
    Analyze structure growth in DET simulations.

    Track the evolution of density fluctuations to measure:
    - Growth factor D(t)
    - Growth rate f = d(ln D)/d(ln a)
    """

    def __init__(self, grid_size: int = 64, kappa: float = 5.0,
                 verbose: bool = True):
        """
        Initialize growth analyzer.

        Parameters
        ----------
        grid_size : int
            Simulation grid size
        kappa : float
            Poisson coupling constant
        verbose : bool
            Print progress
        """
        self.N = grid_size
        self.kappa = kappa
        self.eta = compute_lattice_correction(grid_size)
        self.verbose = verbose

    def setup_initial_perturbations(self, amplitude: float = 0.01,
                                     spectral_index: float = 1.0,
                                     seed: int = 42) -> DETCollider3D:
        """
        Create simulation with initial density perturbations.

        Parameters
        ----------
        amplitude : float
            Initial fluctuation amplitude
        spectral_index : float
            Power spectrum index (n_s)
        seed : int
            Random seed for reproducibility

        Returns
        -------
        DETCollider3D
            Configured simulation with perturbations
        """
        params = DETParams3D(
            N=self.N,
            DT=0.02,
            F_VAC=1.0,  # Higher vacuum for cosmological context
            F_MIN=0.0,

            # Gravity
            gravity_enabled=True,
            kappa_grav=self.kappa,
            alpha_grav=0.02,
            mu_grav=2.0,
            beta_g=5.0,
            eta_lattice=self.eta,

            # Structure
            q_enabled=True,
            alpha_q=0.01,

            # Momentum for dynamics
            momentum_enabled=True,
            alpha_pi=0.1,
            lambda_pi=0.01,
            mu_pi=0.3,

            # Disable interfering features
            angular_momentum_enabled=False,
            floor_enabled=False,
            boundary_enabled=False,
            agency_dynamic=False,
            sigma_dynamic=False,
            coherence_dynamic=False
        )

        sim = DETCollider3D(params)

        # Generate primordial perturbations with P(k) ~ k^n
        np.random.seed(seed)

        # Create perturbation in Fourier space
        kx = fftfreq(self.N, d=1.0/self.N)
        ky = fftfreq(self.N, d=1.0/self.N)
        kz = fftfreq(self.N, d=1.0/self.N)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)
        K[0, 0, 0] = 1.0  # Avoid division by zero

        # P(k) ~ k^n => amplitude ~ k^(n/2)
        power_amplitude = K**(spectral_index / 2)
        power_amplitude[0, 0, 0] = 0  # No k=0 mode

        # Random phases
        phases = np.random.uniform(0, 2*np.pi, (self.N, self.N, self.N))

        # Generate perturbation
        delta_k = power_amplitude * amplitude * np.exp(1j * phases)
        delta_k = (delta_k + np.conj(delta_k[::-1, ::-1, ::-1])) / 2  # Real field

        perturbation = np.real(ifftn(delta_k))

        # Add perturbation to q field (sources gravity)
        sim.q = np.clip(0.1 + perturbation, 0, 1)

        # Initialize F with mean + perturbation
        sim.F = np.maximum(params.F_VAC * (1 + perturbation * 0.1), params.F_MIN)

        return sim

    def measure_rms_fluctuation(self, sim: DETCollider3D) -> float:
        """
        Measure RMS density fluctuation.

        sigma = sqrt(<delta^2>)
        """
        rho = sim.q - sim.b
        rho_mean = np.mean(rho)
        if abs(rho_mean) < 1e-10:
            rho_mean = 1e-10
        delta = (rho - rho_mean) / abs(rho_mean)
        return np.sqrt(np.mean(delta**2))

    def run_growth_simulation(self, n_steps: int = 500,
                               sample_interval: int = 10,
                               amplitude: float = 0.01,
                               spectral_index: float = 1.0) -> GrowthResult:
        """
        Run simulation and track structure growth.

        Parameters
        ----------
        n_steps : int
            Total simulation steps
        sample_interval : int
            Steps between measurements
        amplitude : float
            Initial perturbation amplitude
        spectral_index : float
            Initial power spectrum index

        Returns
        -------
        GrowthResult
            Structure growth results
        """
        if self.verbose:
            print(f"\nStructure Growth Simulation")
            print(f"  Grid: {self.N}^3, Steps: {n_steps}")
            print(f"  Initial amplitude: {amplitude}")
            print("-" * 50)

        # Setup simulation
        sim = self.setup_initial_perturbations(amplitude, spectral_index)

        # Track growth
        time_steps = []
        delta_rms = []

        for t in range(0, n_steps, sample_interval):
            # Measure fluctuation
            sigma = self.measure_rms_fluctuation(sim)
            time_steps.append(t)
            delta_rms.append(sigma)

            # Evolve
            for _ in range(sample_interval):
                sim.step()

        time_steps = np.array(time_steps)
        delta_rms = np.array(delta_rms)

        # Compute growth factor D(t) = delta_rms(t) / delta_rms(0)
        delta_0 = delta_rms[0] if delta_rms[0] > 0 else 1e-10
        growth_factor = delta_rms / delta_0

        # Fit growth: D(t) ~ t^alpha (in matter-dominated, alpha = 2/3)
        valid = time_steps > 0
        if np.sum(valid) > 3 and np.all(growth_factor[valid] > 0):
            try:
                log_t = np.log(time_steps[valid])
                log_D = np.log(growth_factor[valid])
                coeffs = np.polyfit(log_t, log_D, 1)
                growth_exponent = coeffs[0]
            except:
                growth_exponent = 0.0
        else:
            growth_exponent = 0.0

        # Growth rate f = d(ln D)/d(ln t) at late times
        if len(growth_factor) > 10:
            late_idx = len(growth_factor) // 2
            if growth_factor[late_idx] > 0 and growth_factor[-1] > 0:
                d_ln_D = np.log(growth_factor[-1]) - np.log(growth_factor[late_idx])
                d_ln_t = np.log(time_steps[-1]) - np.log(max(time_steps[late_idx], 1))
                growth_rate = d_ln_D / d_ln_t if d_ln_t != 0 else 0.0
            else:
                growth_rate = 0.0
        else:
            growth_rate = growth_exponent

        if self.verbose:
            print(f"  Final sigma = {delta_rms[-1]:.4f}")
            print(f"  Growth factor D = {growth_factor[-1]:.4f}")
            print(f"  Growth exponent = {growth_exponent:.4f}")
            print(f"  Growth rate f = {growth_rate:.4f}")

        return GrowthResult(
            time_steps=time_steps,
            delta_rms=delta_rms,
            growth_factor=growth_factor,
            growth_rate=growth_rate,
            growth_exponent=growth_exponent
        )


# ==============================================================================
# LCDM COMPARISON
# ==============================================================================

class LCDMComparison:
    """
    Compare DET structure formation with standard LCDM predictions.
    """

    def __init__(self, omega_m: float = OMEGA_M, sigma_8: float = SIGMA_8,
                 n_s: float = N_S, verbose: bool = True):
        """
        Initialize LCDM comparison.

        Parameters
        ----------
        omega_m : float
            Matter density parameter
        sigma_8 : float
            Clustering amplitude at 8 Mpc/h
        n_s : float
            Scalar spectral index
        verbose : bool
            Print progress
        """
        self.omega_m = omega_m
        self.sigma_8 = sigma_8
        self.n_s = n_s
        self.verbose = verbose

    def linear_growth_factor(self, a: np.ndarray) -> np.ndarray:
        """
        Compute LCDM linear growth factor D(a).

        Uses approximation: D(a) ~ a * g(a) where g(a) is growth suppression.

        Parameters
        ----------
        a : np.ndarray
            Scale factor

        Returns
        -------
        np.ndarray
            Growth factor D(a), normalized to D(1) = 1
        """
        # Simple fit valid for flat LCDM (Carroll 1992)
        omega_a = self.omega_m / (self.omega_m + (1 - self.omega_m) * a**3)
        omega_l = 1 - omega_a

        # Growth suppression factor
        g = (5/2) * omega_a / (
            omega_a**(4/7) - omega_l + (1 + omega_a/2) * (1 + omega_l/70)
        )

        D = a * g

        # Normalize to D(a=1) = 1
        D_1 = self.linear_growth_factor_single(1.0)
        return D / D_1

    def linear_growth_factor_single(self, a: float) -> float:
        """Growth factor at single scale factor."""
        omega_a = self.omega_m / (self.omega_m + (1 - self.omega_m) * a**3)
        omega_l = 1 - omega_a
        g = (5/2) * omega_a / (
            omega_a**(4/7) - omega_l + (1 + omega_a/2) * (1 + omega_l/70)
        )
        return a * g

    def growth_rate(self, a: float) -> float:
        """
        Compute growth rate f = d(ln D)/d(ln a).

        Uses approximation: f ~ Omega_m(a)^0.55
        """
        omega_a = self.omega_m / (self.omega_m + (1 - self.omega_m) * a**3)
        return omega_a**0.55

    def matter_power_spectrum(self, k: np.ndarray, a: float = 1.0) -> np.ndarray:
        """
        Approximate LCDM matter power spectrum P(k).

        Uses Eisenstein & Hu (1998) transfer function approximation.

        Parameters
        ----------
        k : np.ndarray
            Wavenumber in h/Mpc
        a : float
            Scale factor

        Returns
        -------
        np.ndarray
            Power spectrum P(k) in (Mpc/h)^3
        """
        # Primordial power spectrum
        k_0 = 0.05  # Pivot scale h/Mpc
        P_primordial = (k / k_0)**(self.n_s - 1)

        # Approximate transfer function (simplified Eisenstein-Hu)
        # Characteristic scale from matter-radiation equality
        k_eq = 0.073 * self.omega_m  # h/Mpc

        q = k / k_eq
        T_k = np.log(1 + 2.34*q) / (2.34*q) * (
            1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
        )**(-0.25)

        # Growth factor
        D = self.linear_growth_factor_single(a)

        # Normalize to sigma_8
        # P(k) = A * k^n * T(k)^2 * D(a)^2
        P_unnorm = P_primordial * T_k**2 * D**2

        # Simple normalization (approximate)
        A = self.sigma_8**2 / 100  # Rough normalization

        return A * P_unnorm

    def compare_power_spectrum(self, det_result: PowerSpectrumResult,
                                k_scale: float = 1.0) -> Dict:
        """
        Compare DET power spectrum with LCDM.

        Parameters
        ----------
        det_result : PowerSpectrumResult
            DET power spectrum result
        k_scale : float
            Scaling factor for k (DET units to h/Mpc)

        Returns
        -------
        dict
            Comparison results
        """
        # Scale k to cosmological units (approximate)
        k_cosmo = det_result.k_bins * k_scale

        # LCDM prediction
        P_lcdm = self.matter_power_spectrum(k_cosmo)

        # Spectral index comparison
        n_det = det_result.spectral_index
        n_lcdm = self.n_s - 1  # P(k) ~ k^(n_s-1) on large scales

        if self.verbose:
            print(f"\nPower Spectrum Comparison:")
            print(f"  DET spectral index: {n_det:.3f}")
            print(f"  LCDM spectral index: {n_lcdm:.3f} (primordial)")

        return {
            'k_det': det_result.k_bins,
            'P_det': det_result.P_k,
            'k_cosmo': k_cosmo,
            'P_lcdm': P_lcdm,
            'spectral_index_det': n_det,
            'spectral_index_lcdm': n_lcdm
        }

    def compare_growth(self, det_result: GrowthResult,
                        time_to_scale_factor: Callable = None) -> Dict:
        """
        Compare DET growth with LCDM.

        Parameters
        ----------
        det_result : GrowthResult
            DET growth result
        time_to_scale_factor : Callable
            Function mapping DET time to scale factor a
            If None, uses linear mapping

        Returns
        -------
        dict
            Comparison results
        """
        if time_to_scale_factor is None:
            # Simple linear mapping: t -> a = t/t_max * (1 - a_init) + a_init
            a_init = 0.1
            t_max = det_result.time_steps[-1]
            time_to_scale_factor = lambda t: a_init + (1 - a_init) * t / t_max

        # Map times to scale factors
        a_values = np.array([time_to_scale_factor(t) for t in det_result.time_steps])

        # LCDM growth factor
        D_lcdm = self.linear_growth_factor(a_values)

        # Normalize both to initial
        D_det = det_result.growth_factor

        # Growth rate comparison
        f_lcdm = self.growth_rate(1.0)  # At a=1
        f_det = det_result.growth_rate

        if self.verbose:
            print(f"\nGrowth Comparison:")
            print(f"  DET growth rate: {f_det:.3f}")
            print(f"  LCDM growth rate (a=1): {f_lcdm:.3f}")

        return {
            'time_det': det_result.time_steps,
            'scale_factor': a_values,
            'D_det': D_det,
            'D_lcdm': D_lcdm,
            'growth_rate_det': f_det,
            'growth_rate_lcdm': f_lcdm
        }


# ==============================================================================
# COMPREHENSIVE COSMOLOGICAL ANALYSIS
# ==============================================================================

class CosmologicalScalingAnalyzer:
    """
    Complete cosmological scaling analysis for DET.

    Combines power spectrum, correlation, and growth analyses
    with LCDM comparison.
    """

    def __init__(self, grid_size: int = 64, kappa: float = 5.0,
                 verbose: bool = True):
        """
        Initialize cosmological analyzer.

        Parameters
        ----------
        grid_size : int
            Simulation grid size
        kappa : float
            Poisson coupling constant
        verbose : bool
            Print progress
        """
        self.N = grid_size
        self.kappa = kappa
        self.eta = compute_lattice_correction(grid_size)
        self.G_eff = self.eta * self.kappa / (4 * np.pi)
        self.verbose = verbose

        # Initialize analyzers
        self.power_analyzer = PowerSpectrumAnalyzer(box_size=grid_size, verbose=verbose)
        self.correlation_analyzer = CorrelationAnalyzer(verbose=verbose)
        self.growth_analyzer = StructureGrowthAnalyzer(grid_size, kappa, verbose)
        self.lcdm = LCDMComparison(verbose=verbose)

    def run_full_analysis(self, growth_steps: int = 500,
                           initial_amplitude: float = 0.01) -> CosmologicalAnalysis:
        """
        Run complete cosmological scaling analysis.

        Parameters
        ----------
        growth_steps : int
            Simulation steps for growth analysis
        initial_amplitude : float
            Initial perturbation amplitude

        Returns
        -------
        CosmologicalAnalysis
            Complete analysis results
        """
        if self.verbose:
            print("\n" + "="*70)
            print("DET v6.4 COSMOLOGICAL SCALING ANALYSIS")
            print("="*70)
            print(f"\nGrid size: {self.N}")
            print(f"Kappa: {self.kappa}")
            print(f"Eta: {self.eta:.4f}")
            print(f"G_eff = {self.G_eff:.6f}")

        # 1. Run structure growth simulation
        if self.verbose:
            print("\n" + "-"*50)
            print("1. STRUCTURE GROWTH")
            print("-"*50)

        growth_result = self.growth_analyzer.run_growth_simulation(
            n_steps=growth_steps,
            amplitude=initial_amplitude
        )

        # Setup final state for power spectrum and correlation
        sim = self.growth_analyzer.setup_initial_perturbations(
            amplitude=initial_amplitude
        )

        # Evolve to final state
        for _ in range(growth_steps):
            sim.step()

        # 2. Compute power spectrum
        if self.verbose:
            print("\n" + "-"*50)
            print("2. POWER SPECTRUM")
            print("-"*50)

        power_result = self.power_analyzer.compute_from_det(sim)

        # 3. Compute correlation function
        if self.verbose:
            print("\n" + "-"*50)
            print("3. CORRELATION FUNCTION")
            print("-"*50)

        density = sim.q - sim.b
        correlation_result = self.correlation_analyzer.compute_correlation_fft(density)

        # 4. Compare with LCDM
        if self.verbose:
            print("\n" + "-"*50)
            print("4. LCDM COMPARISON")
            print("-"*50)

        power_comparison = self.lcdm.compare_power_spectrum(power_result)
        growth_comparison = self.lcdm.compare_growth(growth_result)

        # Generate summary
        summary_lines = [
            "",
            "="*70,
            "COSMOLOGICAL SCALING SUMMARY",
            "="*70,
            "",
            f"G_eff = {self.G_eff:.6f}",
            "",
            "Structure Growth:",
            f"  Final fluctuation amplitude: {growth_result.delta_rms[-1]:.4f}",
            f"  Growth factor D: {growth_result.growth_factor[-1]:.4f}",
            f"  Growth exponent: {growth_result.growth_exponent:.4f}",
            f"  Growth rate f: {growth_result.growth_rate:.4f}",
            "",
            "Power Spectrum:",
            f"  Spectral index n: {power_result.spectral_index:.4f}",
            f"  Amplitude: {power_result.amplitude:.4e}",
            "",
            "Correlation Function:",
            f"  Correlation length: {correlation_result.correlation_length:.2f}",
            f"  Power law index gamma: {correlation_result.power_law_index:.4f}",
            "",
            "LCDM Comparison:",
            f"  DET spectral index: {power_result.spectral_index:.3f}",
            f"  LCDM primordial: {N_S - 1:.3f}",
            f"  DET growth rate: {growth_result.growth_rate:.3f}",
            f"  LCDM growth rate: {self.lcdm.growth_rate(1.0):.3f}",
            "",
            "="*70
        ]

        summary = "\n".join(summary_lines)

        if self.verbose:
            print(summary)

        return CosmologicalAnalysis(
            power_spectrum=power_result,
            correlation=correlation_result,
            growth=growth_result,
            G_eff=self.G_eff,
            kappa=self.kappa,
            eta=self.eta,
            grid_size=self.N,
            summary=summary
        )


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_cosmological_analysis(grid_size: int = 64, kappa: float = 5.0,
                               growth_steps: int = 500,
                               verbose: bool = True) -> CosmologicalAnalysis:
    """
    Run complete cosmological scaling analysis.

    This is the main entry point for the v6.4 Cosmological Scaling feature.

    Parameters
    ----------
    grid_size : int
        Simulation grid size
    kappa : float
        Poisson coupling constant
    growth_steps : int
        Simulation steps for growth analysis
    verbose : bool
        Print progress

    Returns
    -------
    CosmologicalAnalysis
        Complete analysis results
    """
    analyzer = CosmologicalScalingAnalyzer(grid_size, kappa, verbose)
    return analyzer.run_full_analysis(growth_steps=growth_steps)


if __name__ == "__main__":
    print("DET v6.4 Cosmological Scaling")
    print("Large-scale structure formation analysis...")
    print()

    # Run analysis with default parameters
    analysis = run_cosmological_analysis(grid_size=64, kappa=5.0, verbose=True)

    # Print final status
    print("\n" + "="*70)
    print("COSMOLOGICAL SCALING ANALYSIS COMPLETE")
    print("="*70)
