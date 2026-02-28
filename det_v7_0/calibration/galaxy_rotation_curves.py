"""
DET v6.4 Galaxy Rotation Curves: Fit SPARC Database Observations
=================================================================

Roadmap Item #2: Implement galaxy rotation curve fitting to test DET
predictions against observed galaxy dynamics.

Theory Overview
---------------
In Newtonian gravity, circular orbital velocity is:
    v(r) = sqrt(G * M(<r) / r)

where M(<r) is the enclosed mass within radius r.

DET predicts gravity via the Poisson equation:
    L*Phi = kappa * rho

where rho = q - b (structural debt). The effective G is:
    G_eff = eta * kappa / (4*pi)

For galaxies, this predicts:
- Rising rotation curves in inner regions (increasing enclosed mass)
- Potentially flat rotation curves in outer regions depending on
  the structural debt distribution

SPARC Database
--------------
SPARC = Spitzer Photometry & Accurate Rotation Curves
~175 galaxies with high-quality rotation curve data.

Reference: Lelli, McGaugh, Schombert (2016), AJ 152, 157
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_si_units import (
    DETUnitSystem, galactic_units, GALACTIC,
    G_SI, M_SUN, PC, C_SI, G_eff
)


# ==============================================================================
# PHYSICAL CONSTANTS FOR GALAXIES
# ==============================================================================

KPC = 1e3 * PC  # kiloparsec in meters
KMS = 1e3       # km/s in m/s


# ==============================================================================
# SPARC-COMPATIBLE DATA STRUCTURES
# ==============================================================================

@dataclass
class GalaxyObservation:
    """
    Observed rotation curve data for a single galaxy.

    Compatible with SPARC database format.
    """
    name: str                          # Galaxy identifier (e.g., "NGC2403")
    radius_kpc: np.ndarray            # Radii of measurements [kpc]
    v_obs: np.ndarray                 # Observed rotation velocity [km/s]
    v_err: np.ndarray                 # Velocity uncertainty [km/s]
    distance_mpc: float = 10.0        # Distance to galaxy [Mpc]
    inclination_deg: float = 60.0     # Disk inclination [degrees]
    morphology: str = "Sb"            # Hubble type

    # Mass components (if available from photometry)
    stellar_mass_msun: float = 0.0    # Total stellar mass [M_sun]
    gas_mass_msun: float = 0.0        # HI gas mass [M_sun]
    disk_scale_kpc: float = 3.0       # Exponential disk scale length [kpc]

    @property
    def n_points(self) -> int:
        """Number of data points."""
        return len(self.radius_kpc)

    @property
    def r_max(self) -> float:
        """Maximum radius in kpc."""
        return np.max(self.radius_kpc)


@dataclass
class RotationCurveFit:
    """Results from fitting a rotation curve model to observations."""
    galaxy_name: str
    model_name: str
    radius_kpc: np.ndarray            # Model evaluation radii
    v_model: np.ndarray               # Model velocities [km/s]
    v_obs: np.ndarray                 # Observed velocities [km/s]
    v_err: np.ndarray                 # Observed uncertainties [km/s]
    chi_squared: float                # Chi-squared statistic
    reduced_chi_squared: float        # Chi-squared / degrees of freedom
    parameters: Dict                  # Fit parameters
    residuals: np.ndarray             # v_obs - v_model
    rms_residual: float               # RMS of residuals [km/s]


# ==============================================================================
# SAMPLE SPARC GALAXY DATA
# ==============================================================================

def load_sample_galaxies() -> Dict[str, GalaxyObservation]:
    """
    Load sample galaxy rotation curves representative of SPARC database.

    These are simplified versions of real SPARC galaxies for testing.
    For production use, load actual SPARC data from files.

    Returns
    -------
    Dict[str, GalaxyObservation]
        Dictionary of galaxy name -> observation data
    """
    galaxies = {}

    # -------------------------------------------------------------------------
    # NGC 2403 - Well-studied spiral galaxy
    # Representative of high surface brightness (HSB) spirals
    # -------------------------------------------------------------------------
    ngc2403_r = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                          10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    ngc2403_v = np.array([35, 55, 85, 105, 115, 120, 125, 130, 132, 133,
                          133, 132, 131, 130, 128, 126])
    ngc2403_err = np.array([5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6])

    galaxies["NGC2403"] = GalaxyObservation(
        name="NGC2403",
        radius_kpc=ngc2403_r,
        v_obs=ngc2403_v,
        v_err=ngc2403_err,
        distance_mpc=3.2,
        inclination_deg=63.0,
        morphology="SABcd",
        stellar_mass_msun=8e9,
        gas_mass_msun=3e9,
        disk_scale_kpc=2.5
    )

    # -------------------------------------------------------------------------
    # UGC 128 - Low surface brightness (LSB) galaxy
    # These are key tests for modified gravity vs dark matter
    # -------------------------------------------------------------------------
    ugc128_r = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                         11.0, 12.0, 13.0, 14.0])
    ugc128_v = np.array([25, 40, 52, 62, 70, 76, 80, 83, 85, 86, 86, 85, 84, 83])
    ugc128_err = np.array([4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7])

    galaxies["UGC128"] = GalaxyObservation(
        name="UGC128",
        radius_kpc=ugc128_r,
        v_obs=ugc128_v,
        v_err=ugc128_err,
        distance_mpc=64.0,
        inclination_deg=57.0,
        morphology="Sdm",
        stellar_mass_msun=1e9,
        gas_mass_msun=2e9,
        disk_scale_kpc=4.0
    )

    # -------------------------------------------------------------------------
    # DDO 154 - Dwarf irregular galaxy
    # Dark matter dominated in standard cosmology
    # -------------------------------------------------------------------------
    ddo154_r = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ddo154_v = np.array([15, 25, 32, 38, 42, 45, 47, 48, 48, 48])
    ddo154_err = np.array([3, 3, 3, 3, 3, 3, 4, 4, 4, 5])

    galaxies["DDO154"] = GalaxyObservation(
        name="DDO154",
        radius_kpc=ddo154_r,
        v_obs=ddo154_v,
        v_err=ddo154_err,
        distance_mpc=4.3,
        inclination_deg=66.0,
        morphology="IBm",
        stellar_mass_msun=3e7,
        gas_mass_msun=4e8,
        disk_scale_kpc=1.0
    )

    # -------------------------------------------------------------------------
    # NGC 6946 - Large spiral galaxy
    # High mass, extended rotation curve
    # -------------------------------------------------------------------------
    ngc6946_r = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                          12.0, 14.0, 16.0, 18.0, 20.0])
    ngc6946_v = np.array([80, 130, 165, 185, 195, 200, 202, 203, 202, 200,
                          195, 190, 185, 180, 175])
    ngc6946_err = np.array([8, 8, 7, 6, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10])

    galaxies["NGC6946"] = GalaxyObservation(
        name="NGC6946",
        radius_kpc=ngc6946_r,
        v_obs=ngc6946_v,
        v_err=ngc6946_err,
        distance_mpc=5.9,
        inclination_deg=33.0,
        morphology="SABcd",
        stellar_mass_msun=4e10,
        gas_mass_msun=8e9,
        disk_scale_kpc=3.5
    )

    return galaxies


# ==============================================================================
# MASS DISTRIBUTION MODELS
# ==============================================================================

class MassModel:
    """Base class for galaxy mass distribution models."""

    def enclosed_mass(self, r_kpc: np.ndarray) -> np.ndarray:
        """
        Compute enclosed mass within radius r.

        Parameters
        ----------
        r_kpc : np.ndarray
            Radii in kpc

        Returns
        -------
        np.ndarray
            Enclosed mass in solar masses
        """
        raise NotImplementedError

    def rotation_velocity(self, r_kpc: np.ndarray) -> np.ndarray:
        """
        Compute circular rotation velocity.

        v(r) = sqrt(G * M(<r) / r)

        Parameters
        ----------
        r_kpc : np.ndarray
            Radii in kpc

        Returns
        -------
        np.ndarray
            Rotation velocity in km/s
        """
        r_m = r_kpc * KPC
        M_kg = self.enclosed_mass(r_kpc) * M_SUN

        # Avoid division by zero
        r_m = np.maximum(r_m, 1e-10)

        v_ms = np.sqrt(G_SI * M_kg / r_m)
        return v_ms / KMS  # Convert to km/s


class ExponentialDisk(MassModel):
    """
    Exponential disk mass distribution.

    Surface density: Sigma(r) = Sigma_0 * exp(-r / R_d)
    """

    def __init__(self, total_mass_msun: float, scale_length_kpc: float):
        """
        Parameters
        ----------
        total_mass_msun : float
            Total disk mass in solar masses
        scale_length_kpc : float
            Disk scale length in kpc
        """
        self.M_total = total_mass_msun
        self.R_d = scale_length_kpc

    def enclosed_mass(self, r_kpc: np.ndarray) -> np.ndarray:
        """
        Enclosed mass for exponential disk.

        M(<r) = M_total * (1 - (1 + r/R_d) * exp(-r/R_d))
        """
        x = r_kpc / self.R_d
        return self.M_total * (1 - (1 + x) * np.exp(-x))


class NFWHalo(MassModel):
    """
    Navarro-Frenk-White (NFW) dark matter halo.

    This is the standard cold dark matter halo profile.
    """

    def __init__(self, virial_mass_msun: float, concentration: float = 10.0,
                 virial_radius_kpc: float = 200.0):
        """
        Parameters
        ----------
        virial_mass_msun : float
            Virial mass in solar masses
        concentration : float
            Concentration parameter c = R_vir / R_s
        virial_radius_kpc : float
            Virial radius in kpc
        """
        self.M_vir = virial_mass_msun
        self.c = concentration
        self.R_vir = virial_radius_kpc
        self.R_s = virial_radius_kpc / concentration

        # Normalization factor
        self.f_c = np.log(1 + concentration) - concentration / (1 + concentration)

    def enclosed_mass(self, r_kpc: np.ndarray) -> np.ndarray:
        """
        Enclosed mass for NFW profile.

        M(<r) = M_vir * f(r/R_s) / f(c)
        where f(x) = ln(1+x) - x/(1+x)
        """
        x = r_kpc / self.R_s
        f_x = np.log(1 + x) - x / (1 + x)
        return self.M_vir * f_x / self.f_c


class CombinedMassModel(MassModel):
    """
    Combined mass model with multiple components.

    Typically: disk + gas + (optional) dark matter halo
    """

    def __init__(self, components: List[Tuple[str, MassModel]]):
        """
        Parameters
        ----------
        components : List[Tuple[str, MassModel]]
            List of (name, model) tuples
        """
        self.components = components

    def enclosed_mass(self, r_kpc: np.ndarray) -> np.ndarray:
        """Sum of enclosed masses from all components."""
        total = np.zeros_like(r_kpc)
        for name, model in self.components:
            total += model.enclosed_mass(r_kpc)
        return total

    def component_velocities(self, r_kpc: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual velocity contributions from each component.

        Returns dict of component name -> velocity array
        """
        result = {}
        for name, model in self.components:
            result[name] = model.rotation_velocity(r_kpc)
        return result


# ==============================================================================
# DET ROTATION CURVE MODEL
# ==============================================================================

class DETRotationModel:
    """
    DET-based rotation curve model.

    In DET, gravity emerges from structural debt (q - b). This model
    computes the rotation curve that would arise from a given mass
    distribution using DET's effective gravitational constant.

    The key prediction is whether DET can explain flat rotation curves
    without requiring dark matter.
    """

    def __init__(self, kappa: float = 5.0, eta: float = 0.968,
                 units: DETUnitSystem = None):
        """
        Parameters
        ----------
        kappa : float
            DET Poisson coupling constant
        eta : float
            Lattice correction factor
        units : DETUnitSystem, optional
            Unit system (defaults to galactic scale)
        """
        self.kappa = kappa
        self.eta = eta
        self.G_eff_lattice = eta * kappa / (4 * np.pi)

        # For galactic scales, G_eff maps to G_SI
        # This is automatic in the unit conversion
        self.units = units or galactic_units()

    def rotation_velocity_from_mass(self, r_kpc: np.ndarray,
                                     enclosed_mass_msun: np.ndarray) -> np.ndarray:
        """
        Compute DET rotation velocity from enclosed mass profile.

        In DET, v(r) = sqrt(G_eff * M(<r) / r) in appropriate units.

        Parameters
        ----------
        r_kpc : np.ndarray
            Radii in kpc
        enclosed_mass_msun : np.ndarray
            Enclosed mass in solar masses at each radius

        Returns
        -------
        np.ndarray
            Rotation velocity in km/s
        """
        r_m = r_kpc * KPC
        M_kg = enclosed_mass_msun * M_SUN

        r_m = np.maximum(r_m, 1e-10)  # Avoid division by zero

        # DET uses same G as Newtonian when calibrated
        v_ms = np.sqrt(G_SI * M_kg / r_m)
        return v_ms / KMS

    def fit_galaxy(self, galaxy: GalaxyObservation,
                   include_dark_matter: bool = False,
                   disk_ml_ratio: float = 1.0) -> RotationCurveFit:
        """
        Fit DET model to observed galaxy rotation curve.

        Parameters
        ----------
        galaxy : GalaxyObservation
            Observed rotation curve data
        include_dark_matter : bool
            If True, include NFW halo component
        disk_ml_ratio : float
            Mass-to-light ratio for stellar disk (adjusts stellar mass)

        Returns
        -------
        RotationCurveFit
            Fit results
        """
        r = galaxy.radius_kpc
        v_obs = galaxy.v_obs
        v_err = galaxy.v_err

        # Build mass model from galaxy properties
        components = []

        # Stellar disk
        if galaxy.stellar_mass_msun > 0:
            stellar_mass = galaxy.stellar_mass_msun * disk_ml_ratio
            disk = ExponentialDisk(stellar_mass, galaxy.disk_scale_kpc)
            components.append(("stars", disk))

        # Gas disk (assume same scale length)
        if galaxy.gas_mass_msun > 0:
            gas = ExponentialDisk(galaxy.gas_mass_msun, galaxy.disk_scale_kpc * 1.5)
            components.append(("gas", gas))

        # Dark matter halo (if requested)
        if include_dark_matter:
            # Estimate halo mass from abundance matching
            halo_mass = 50 * (galaxy.stellar_mass_msun + galaxy.gas_mass_msun)
            halo = NFWHalo(halo_mass, concentration=10.0, virial_radius_kpc=200.0)
            components.append(("halo", halo))

        # Combined model
        if components:
            mass_model = CombinedMassModel(components)
        else:
            # Fallback: point mass estimate
            total_mass = np.max(v_obs)**2 * np.max(r) * KPC / G_SI / M_SUN
            mass_model = ExponentialDisk(total_mass, galaxy.disk_scale_kpc)

        # Compute model velocity
        M_enclosed = mass_model.enclosed_mass(r)
        v_model = self.rotation_velocity_from_mass(r, M_enclosed)

        # Compute fit statistics
        residuals = v_obs - v_model
        chi_sq = np.sum((residuals / v_err)**2)
        dof = len(r) - 1  # Degrees of freedom
        reduced_chi_sq = chi_sq / dof if dof > 0 else chi_sq
        rms = np.sqrt(np.mean(residuals**2))

        model_name = "DET"
        if include_dark_matter:
            model_name += "+NFW"

        return RotationCurveFit(
            galaxy_name=galaxy.name,
            model_name=model_name,
            radius_kpc=r,
            v_model=v_model,
            v_obs=v_obs,
            v_err=v_err,
            chi_squared=chi_sq,
            reduced_chi_squared=reduced_chi_sq,
            parameters={
                'kappa': self.kappa,
                'eta': self.eta,
                'G_eff': self.G_eff_lattice,
                'disk_ml_ratio': disk_ml_ratio,
                'include_dm': include_dark_matter,
                'stellar_mass': galaxy.stellar_mass_msun * disk_ml_ratio,
                'gas_mass': galaxy.gas_mass_msun
            },
            residuals=residuals,
            rms_residual=rms
        )

    def optimize_ml_ratio(self, galaxy: GalaxyObservation,
                          ml_range: Tuple[float, float] = (0.1, 5.0)) -> Tuple[float, RotationCurveFit]:
        """
        Find optimal mass-to-light ratio to minimize residuals.

        Parameters
        ----------
        galaxy : GalaxyObservation
            Galaxy to fit
        ml_range : Tuple[float, float]
            Range of M/L ratios to search

        Returns
        -------
        Tuple[float, RotationCurveFit]
            Optimal M/L ratio and corresponding fit
        """
        def objective(ml):
            fit = self.fit_galaxy(galaxy, include_dark_matter=False, disk_ml_ratio=ml[0])
            return fit.chi_squared

        result = minimize(objective, x0=[1.0], bounds=[ml_range], method='L-BFGS-B')
        optimal_ml = result.x[0]

        best_fit = self.fit_galaxy(galaxy, include_dark_matter=False, disk_ml_ratio=optimal_ml)
        return optimal_ml, best_fit


# ==============================================================================
# ROTATION CURVE ANALYSIS
# ==============================================================================

class RotationCurveAnalyzer:
    """
    Comprehensive analyzer for galaxy rotation curves.

    Compares DET predictions with observations and standard dark matter models.
    """

    def __init__(self, kappa: float = 5.0, eta: float = 0.968):
        self.det_model = DETRotationModel(kappa, eta)
        self.galaxies: Dict[str, GalaxyObservation] = {}
        self.results: Dict[str, Dict[str, RotationCurveFit]] = {}

    def load_galaxies(self, galaxies: Dict[str, GalaxyObservation] = None):
        """Load galaxy data (default: sample SPARC galaxies)."""
        if galaxies is None:
            galaxies = load_sample_galaxies()
        self.galaxies = galaxies

    def analyze_galaxy(self, name: str, verbose: bool = True) -> Dict[str, RotationCurveFit]:
        """
        Perform full analysis on a single galaxy.

        Fits multiple models:
        1. DET (baryons only)
        2. DET with optimized M/L
        3. DET + NFW dark matter halo

        Returns dict of model_name -> fit
        """
        if name not in self.galaxies:
            raise ValueError(f"Galaxy '{name}' not found. Available: {list(self.galaxies.keys())}")

        galaxy = self.galaxies[name]
        results = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {name}")
            print(f"{'='*60}")
            print(f"Distance: {galaxy.distance_mpc:.1f} Mpc")
            print(f"Stellar mass: {galaxy.stellar_mass_msun:.2e} M_sun")
            print(f"Gas mass: {galaxy.gas_mass_msun:.2e} M_sun")
            print(f"Disk scale: {galaxy.disk_scale_kpc:.1f} kpc")
            print(f"Data points: {galaxy.n_points}")

        # Model 1: DET with baryons only (M/L = 1)
        fit_det = self.det_model.fit_galaxy(galaxy, include_dark_matter=False)
        results["DET_baryons"] = fit_det

        if verbose:
            print(f"\nDET (baryons only, M/L=1.0):")
            print(f"  Chi-squared: {fit_det.chi_squared:.1f}")
            print(f"  Reduced chi-squared: {fit_det.reduced_chi_squared:.2f}")
            print(f"  RMS residual: {fit_det.rms_residual:.1f} km/s")

        # Model 2: DET with optimized M/L
        opt_ml, fit_det_opt = self.det_model.optimize_ml_ratio(galaxy)
        results["DET_optimized"] = fit_det_opt

        if verbose:
            print(f"\nDET (optimized M/L={opt_ml:.2f}):")
            print(f"  Chi-squared: {fit_det_opt.chi_squared:.1f}")
            print(f"  Reduced chi-squared: {fit_det_opt.reduced_chi_squared:.2f}")
            print(f"  RMS residual: {fit_det_opt.rms_residual:.1f} km/s")

        # Model 3: DET + dark matter halo
        fit_det_dm = self.det_model.fit_galaxy(galaxy, include_dark_matter=True, disk_ml_ratio=opt_ml)
        results["DET_with_DM"] = fit_det_dm

        if verbose:
            print(f"\nDET + NFW halo (M/L={opt_ml:.2f}):")
            print(f"  Chi-squared: {fit_det_dm.chi_squared:.1f}")
            print(f"  Reduced chi-squared: {fit_det_dm.reduced_chi_squared:.2f}")
            print(f"  RMS residual: {fit_det_dm.rms_residual:.1f} km/s")

        # Store results
        self.results[name] = results

        return results

    def analyze_all(self, verbose: bool = True) -> Dict[str, Dict[str, RotationCurveFit]]:
        """Analyze all loaded galaxies."""
        for name in self.galaxies:
            self.analyze_galaxy(name, verbose=verbose)
        return self.results

    def summary_report(self) -> str:
        """Generate summary report of all analyses."""
        lines = [
            "",
            "=" * 70,
            "DET GALAXY ROTATION CURVE ANALYSIS SUMMARY",
            "=" * 70,
            "",
            f"Galaxies analyzed: {len(self.results)}",
            "",
            "Results by Galaxy:",
            "-" * 70,
            f"{'Galaxy':<12} {'Model':<20} {'Chi-sq':>10} {'Red.Chi-sq':>12} {'RMS (km/s)':>12}",
            "-" * 70
        ]

        for galaxy_name, models in self.results.items():
            for model_name, fit in models.items():
                lines.append(
                    f"{galaxy_name:<12} {model_name:<20} {fit.chi_squared:>10.1f} "
                    f"{fit.reduced_chi_squared:>12.2f} {fit.rms_residual:>12.1f}"
                )
            lines.append("")

        # Summary statistics
        lines.extend([
            "-" * 70,
            "Summary Statistics:",
            "-" * 70
        ])

        # Count how often each model performs best
        best_counts = {}
        for galaxy_name, models in self.results.items():
            best_model = min(models.keys(), key=lambda m: models[m].reduced_chi_squared)
            best_counts[best_model] = best_counts.get(best_model, 0) + 1

        for model, count in sorted(best_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {model}: Best fit for {count}/{len(self.results)} galaxies")

        # DET-only performance
        det_baryons_fits = [r["DET_baryons"] for r in self.results.values() if "DET_baryons" in r]
        if det_baryons_fits:
            mean_chi = np.mean([f.reduced_chi_squared for f in det_baryons_fits])
            lines.append(f"\n  Mean reduced chi-squared (DET baryons only): {mean_chi:.2f}")

        lines.extend(["", "=" * 70])

        return "\n".join(lines)

    def needs_dark_matter(self, galaxy_name: str, threshold: float = 2.0) -> bool:
        """
        Determine if a galaxy requires dark matter in DET framework.

        Returns True if reduced chi-squared for baryons-only model
        exceeds threshold.
        """
        if galaxy_name not in self.results:
            self.analyze_galaxy(galaxy_name, verbose=False)

        fit = self.results[galaxy_name].get("DET_baryons")
        if fit is None:
            return True

        return fit.reduced_chi_squared > threshold


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_sparc_analysis(verbose: bool = True) -> RotationCurveAnalyzer:
    """
    Run full SPARC rotation curve analysis.

    This is the main entry point for the v6.4 Galaxy Rotation Curves feature.

    Returns
    -------
    RotationCurveAnalyzer
        Analyzer with results
    """
    if verbose:
        print("DET v6.4 Galaxy Rotation Curves Analysis")
        print("Fitting SPARC database observations...")
        print()

    analyzer = RotationCurveAnalyzer()
    analyzer.load_galaxies()
    analyzer.analyze_all(verbose=verbose)

    if verbose:
        print(analyzer.summary_report())

    return analyzer


if __name__ == "__main__":
    analyzer = run_sparc_analysis(verbose=True)
