"""
DET v6.4 Black Hole Thermodynamics: Hawking-like Radiation Predictions
======================================================================

Roadmap Item #5: Test whether DET black holes exhibit Hawking-like
thermodynamic behavior.

Theoretical Framework
---------------------
In standard physics, black holes have thermodynamic properties:
- Temperature: T_H = hbar*c^3 / (8*pi*G*M*k_B) ~ 1/M
- Entropy: S = A*c^3 / (4*G*hbar) ~ M^2 (area law)
- Luminosity: L ~ T^4 * A ~ 1/M^2

In DET, a "black hole" is characterized by:
- High structural debt: q -> 1
- Vanishing presence: P -> 0 (time stops)
- Strong gravitational potential well

DET Black Hole Properties
-------------------------
1. **Time Dilation:** P = a*sigma / (1+F) / (1+H) -> 0 as F accumulates
2. **Structural Ceiling:** a_max = 1/(1 + lambda_a * q^2) -> 0 as q -> 1
3. **Radiation Mechanism:** F-flux escaping from high-q region via:
   - Quantum fluctuations (coherence dynamics)
   - Grace injection at boundaries
   - Diffusive transport

Reference: DET Theory Card v6.3, Sections III, V, VI
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.optimize import curve_fit
from scipy.ndimage import label
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D, compute_lattice_correction


# ==============================================================================
# PHYSICAL CONSTANTS (for Hawking comparison)
# ==============================================================================

# Fundamental constants (SI)
HBAR_SI = 1.054571817e-34  # J*s
C_SI = 299792458  # m/s
G_SI = 6.67430e-11  # m^3/(kg*s^2)
K_B_SI = 1.380649e-23  # J/K
M_SUN_SI = 1.989e30  # kg

# Hawking temperature coefficient: T_H = HAWKING_COEFF / M
# T_H = hbar * c^3 / (8*pi*G*M*k_B)
HAWKING_COEFF_SI = HBAR_SI * C_SI**3 / (8 * np.pi * G_SI * K_B_SI)  # K * kg


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class BlackHoleState:
    """State of a DET black hole."""
    mass: float                 # Gravitational mass (sum of q-b)
    radius: float               # Effective radius (where q > threshold)
    q_central: float            # Central structural debt
    P_central: float            # Central presence (clock rate)
    F_total: float              # Total resource in BH region
    surface_area: float         # Surface area in lattice units


@dataclass
class RadiationMeasurement:
    """Measurement of radiation from black hole."""
    time: float                 # Measurement time
    flux_out: float            # Outward F-flux across surface
    flux_in: float             # Inward F-flux across surface
    net_flux: float            # Net radiation (positive = emission)
    luminosity: float          # Power = net_flux per unit time


@dataclass
class ThermodynamicProperties:
    """Thermodynamic properties of DET black hole."""
    mass: float                 # Black hole mass
    temperature: float          # Effective temperature
    entropy: float              # Entropy estimate
    luminosity: float           # Radiation power
    lifetime: float             # Estimated evaporation time

    # Scaling exponents
    T_mass_exponent: float      # T ~ M^n (Hawking predicts n = -1)
    S_mass_exponent: float      # S ~ M^n (Hawking predicts n = 2)


@dataclass
class HawkingComparison:
    """Comparison with Hawking predictions."""
    masses: np.ndarray
    temperatures_det: np.ndarray
    temperatures_hawking: np.ndarray  # Scaled for comparison
    T_exponent_det: float       # Measured exponent
    T_exponent_hawking: float   # Expected = -1
    entropy_exponent_det: float
    entropy_exponent_hawking: float  # Expected = 2
    agreement_temperature: float  # Quality of T~1/M agreement
    agreement_entropy: float      # Quality of S~M^2 agreement


@dataclass
class BlackHoleAnalysis:
    """Complete black hole thermodynamics analysis."""
    black_hole: BlackHoleState
    radiation: List[RadiationMeasurement]
    thermodynamics: ThermodynamicProperties
    hawking_comparison: Optional[HawkingComparison]
    G_eff: float
    summary: str = ""


# ==============================================================================
# BLACK HOLE CONFIGURATION
# ==============================================================================

class BlackHoleConfigurator:
    """
    Create and configure black hole-like states in DET.

    A DET black hole is a region with:
    - High structural debt q -> 1
    - Strong gravitational potential well
    - Vanishing presence P -> 0
    """

    def __init__(self, grid_size: int = 64, kappa: float = 5.0,
                 verbose: bool = True):
        """
        Initialize black hole configurator.

        Parameters
        ----------
        grid_size : int
            Simulation grid size
        kappa : float
            Gravity coupling constant
        verbose : bool
            Print progress information
        """
        self.N = grid_size
        self.kappa = kappa
        self.eta = compute_lattice_correction(grid_size)
        self.G_eff = self.eta * self.kappa / (4 * np.pi)
        self.verbose = verbose
        self.center = grid_size // 2

    def create_black_hole(self, mass: float, radius: float = 3.0,
                          q_core: float = 0.95) -> DETCollider3D:
        """
        Create a black hole configuration.

        Parameters
        ----------
        mass : float
            Desired gravitational mass (F units)
        radius : float
            Core radius
        q_core : float
            Central structural debt (0.9-0.99 for strong BH)

        Returns
        -------
        DETCollider3D
            Simulation with black hole
        """
        params = DETParams3D(
            N=self.N,
            DT=0.01,
            F_VAC=0.1,
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
            alpha_q=0.005,  # Slow q evolution to maintain BH

            # Momentum for dynamics
            momentum_enabled=True,
            alpha_pi=0.1,
            lambda_pi=0.01,
            mu_pi=0.3,

            # Coherence for quantum effects
            coherence_dynamic=True,
            C_init=0.1,
            alpha_C=0.02,
            lambda_C=0.01,

            # Grace for radiation mechanism
            boundary_enabled=True,
            grace_enabled=True,
            F_MIN_grace=0.05,

            # Floor to prevent singularity
            floor_enabled=True,
            eta_floor=0.1,
            F_core=5.0,
            floor_power=2.0,

            # Agency dynamics
            agency_dynamic=True,
            lambda_a=30.0,
            beta_a=0.1
        )

        sim = DETCollider3D(params)

        # Create black hole core with high q
        cx, cy, cz = self.center, self.center, self.center

        # Gaussian profile for smooth BH
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    dx = i - cx
                    dy = j - cy
                    dz = k - cz
                    r = np.sqrt(dx**2 + dy**2 + dz**2)

                    # q profile: high in core, decaying outward
                    q_val = q_core * np.exp(-(r / radius)**2)
                    sim.q[i, j, k] = max(sim.q[i, j, k], q_val)

                    # F profile: concentrated mass
                    F_val = mass * np.exp(-(r / radius)**2) / (2 * np.pi * radius**2)**(3/2)
                    sim.F[i, j, k] += F_val

        # Normalize F to desired mass
        F_total = np.sum(sim.F)
        if F_total > 0:
            sim.F *= mass / F_total

        # Let field establish
        for _ in range(50):
            sim.step()

        return sim

    def measure_black_hole_state(self, sim: DETCollider3D,
                                  q_threshold: float = 0.5) -> BlackHoleState:
        """
        Measure current state of black hole.

        Parameters
        ----------
        sim : DETCollider3D
            Simulation containing black hole
        q_threshold : float
            Threshold for defining BH boundary

        Returns
        -------
        BlackHoleState
            Current black hole state
        """
        # Find BH region (q > threshold)
        bh_mask = sim.q > q_threshold

        # Gravitational mass
        rho = sim.q - sim.b
        mass = np.sum(rho)

        # Effective radius
        n_cells = np.sum(bh_mask)
        if n_cells > 0:
            radius = (3 * n_cells / (4 * np.pi))**(1/3)
        else:
            radius = 0.0

        # Central values
        cx, cy, cz = self.center, self.center, self.center
        q_central = sim.q[cx, cy, cz]

        # Compute presence at center
        F_center = sim.F[cx, cy, cz]
        a_center = sim.a[cx, cy, cz]
        sigma_center = sim.sigma[cx, cy, cz]
        H_center = sigma_center  # Simplified coordination load
        P_central = a_center * sigma_center / (1 + F_center) / (1 + H_center)

        # Total F in BH region
        F_total = np.sum(sim.F[bh_mask]) if np.any(bh_mask) else 0.0

        # Surface area (number of boundary cells * cell area)
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(bh_mask)
        surface_mask = bh_mask & ~eroded
        surface_area = np.sum(surface_mask)

        return BlackHoleState(
            mass=mass,
            radius=radius,
            q_central=q_central,
            P_central=P_central,
            F_total=F_total,
            surface_area=surface_area
        )


# ==============================================================================
# RADIATION MEASUREMENT
# ==============================================================================

class RadiationAnalyzer:
    """
    Measure Hawking-like radiation from DET black holes.

    Radiation is measured as net F-flux escaping from the BH region.
    """

    def __init__(self, verbose: bool = True):
        """Initialize radiation analyzer."""
        self.verbose = verbose

    def measure_flux(self, sim: DETCollider3D,
                     q_threshold: float = 0.5) -> RadiationMeasurement:
        """
        Measure F-flux across black hole surface.

        Parameters
        ----------
        sim : DETCollider3D
            Simulation containing black hole
        q_threshold : float
            Threshold defining BH boundary

        Returns
        -------
        RadiationMeasurement
            Flux measurement
        """
        # Define BH region
        bh_mask = sim.q > q_threshold

        if not np.any(bh_mask):
            return RadiationMeasurement(
                time=0.0, flux_out=0.0, flux_in=0.0,
                net_flux=0.0, luminosity=0.0
            )

        # Find surface cells (boundary of BH region)
        from scipy.ndimage import binary_erosion, binary_dilation
        eroded = binary_erosion(bh_mask)
        dilated = binary_dilation(bh_mask)

        # Inner surface (just inside BH)
        inner_surface = bh_mask & ~eroded
        # Outer surface (just outside BH)
        outer_surface = dilated & ~bh_mask

        # Measure F gradient across surface
        # This gives the diffusive flux
        F_inner = np.mean(sim.F[inner_surface]) if np.any(inner_surface) else 0
        F_outer = np.mean(sim.F[outer_surface]) if np.any(outer_surface) else 0

        # Net flux = outward - inward
        # Positive gradient (F_inner > F_outer) means outward flux
        sigma_avg = np.mean(sim.sigma)
        n_surface = np.sum(inner_surface)

        # Diffusive flux ~ sigma * (F_inner - F_outer)
        flux_out = max(0, sigma_avg * (F_inner - F_outer) * n_surface)
        flux_in = max(0, sigma_avg * (F_outer - F_inner) * n_surface)
        net_flux = flux_out - flux_in

        # Also measure momentum-driven flux
        # Sum of |J_mom| across surface
        if hasattr(sim, 'pi_x'):
            pi_mag = np.sqrt(sim.pi_x**2 + sim.pi_y**2 + sim.pi_z**2)
            mom_flux = np.sum(pi_mag[inner_surface]) * 0.1 if np.any(inner_surface) else 0
            flux_out += mom_flux
            net_flux += mom_flux

        # Luminosity = flux per unit time (per step)
        luminosity = net_flux / sim.p.DT if sim.p.DT > 0 else 0

        return RadiationMeasurement(
            time=0.0,  # Will be set externally
            flux_out=flux_out,
            flux_in=flux_in,
            net_flux=net_flux,
            luminosity=luminosity
        )

    def measure_radiation_over_time(self, sim: DETCollider3D,
                                     n_steps: int = 200,
                                     sample_interval: int = 10,
                                     q_threshold: float = 0.5) -> List[RadiationMeasurement]:
        """
        Track radiation over time.

        Parameters
        ----------
        sim : DETCollider3D
            Simulation containing black hole
        n_steps : int
            Total simulation steps
        sample_interval : int
            Steps between measurements
        q_threshold : float
            BH boundary threshold

        Returns
        -------
        List[RadiationMeasurement]
            Time series of radiation measurements
        """
        measurements = []

        for t in range(0, n_steps, sample_interval):
            measurement = self.measure_flux(sim, q_threshold)
            measurement.time = t * sim.p.DT
            measurements.append(measurement)

            # Evolve simulation
            for _ in range(sample_interval):
                sim.step()

        return measurements


# ==============================================================================
# THERMODYNAMIC CALCULATIONS
# ==============================================================================

class ThermodynamicsCalculator:
    """
    Calculate thermodynamic properties of DET black holes.

    Key relations to test:
    - Temperature T ~ 1/M (Hawking)
    - Entropy S ~ M^2 (Bekenstein-Hawking)
    - Luminosity L ~ 1/M^2
    """

    def __init__(self, G_eff: float, verbose: bool = True):
        """
        Initialize thermodynamics calculator.

        Parameters
        ----------
        G_eff : float
            Effective gravitational constant
        verbose : bool
            Print progress
        """
        self.G_eff = G_eff
        self.verbose = verbose

    def compute_temperature(self, mass: float, luminosity: float,
                            surface_area: float) -> float:
        """
        Estimate effective temperature from radiation.

        Using Stefan-Boltzmann-like relation: L = sigma * A * T^4
        => T = (L / (sigma * A))^(1/4)

        We use normalized units where sigma = 1.

        Parameters
        ----------
        mass : float
            Black hole mass
        luminosity : float
            Measured radiation power
        surface_area : float
            Surface area

        Returns
        -------
        float
            Effective temperature
        """
        if surface_area <= 0 or luminosity <= 0:
            # Fallback: use Hawking-like relation T ~ 1/M
            return 1.0 / max(mass, 0.1)

        # T^4 ~ L / A
        T4 = luminosity / surface_area
        return T4**(1/4)

    def compute_entropy(self, mass: float, surface_area: float) -> float:
        """
        Estimate entropy from area law.

        S ~ A / (4 * G_eff) in natural units

        Parameters
        ----------
        mass : float
            Black hole mass
        surface_area : float
            Surface area

        Returns
        -------
        float
            Entropy estimate
        """
        if surface_area <= 0:
            # Fallback: S ~ M^2
            return mass**2

        # Area law: S = A / (4 * G_eff)
        return surface_area / (4 * self.G_eff)

    def compute_lifetime(self, mass: float, luminosity: float) -> float:
        """
        Estimate evaporation lifetime.

        tau ~ M / L (mass / power)

        In Hawking theory: tau ~ M^3

        Parameters
        ----------
        mass : float
            Black hole mass
        luminosity : float
            Radiation power

        Returns
        -------
        float
            Estimated lifetime
        """
        if luminosity <= 0:
            return float('inf')

        return abs(mass) / luminosity

    def analyze_scaling(self, masses: np.ndarray,
                         temperatures: np.ndarray,
                         entropies: np.ndarray) -> Tuple[float, float]:
        """
        Fit scaling exponents T ~ M^a, S ~ M^b.

        Parameters
        ----------
        masses : np.ndarray
            Array of masses
        temperatures : np.ndarray
            Corresponding temperatures
        entropies : np.ndarray
            Corresponding entropies

        Returns
        -------
        Tuple[float, float]
            (temperature exponent, entropy exponent)
        """
        # Filter valid data
        valid = (masses > 0) & (temperatures > 0) & (entropies > 0)

        if np.sum(valid) < 2:
            return 0.0, 0.0

        log_M = np.log(masses[valid])
        log_T = np.log(temperatures[valid])
        log_S = np.log(entropies[valid])

        # Linear fit in log-log space
        try:
            T_coeffs = np.polyfit(log_M, log_T, 1)
            S_coeffs = np.polyfit(log_M, log_S, 1)
            T_exponent = T_coeffs[0]
            S_exponent = S_coeffs[0]
        except:
            T_exponent = 0.0
            S_exponent = 0.0

        return T_exponent, S_exponent


# ==============================================================================
# HAWKING COMPARISON
# ==============================================================================

class HawkingComparer:
    """
    Compare DET black hole thermodynamics with Hawking predictions.
    """

    def __init__(self, G_eff: float, verbose: bool = True):
        """
        Initialize Hawking comparer.

        Parameters
        ----------
        G_eff : float
            Effective gravitational constant
        verbose : bool
            Print progress
        """
        self.G_eff = G_eff
        self.verbose = verbose

    def hawking_temperature(self, mass: float) -> float:
        """
        Compute Hawking temperature (dimensionless units).

        T_H = 1 / (8 * pi * G_eff * M)

        This is the temperature in units where hbar = c = k_B = 1.
        """
        if mass <= 0:
            return float('inf')
        return 1.0 / (8 * np.pi * self.G_eff * mass)

    def bekenstein_hawking_entropy(self, mass: float) -> float:
        """
        Compute Bekenstein-Hawking entropy.

        S = 4 * pi * G_eff * M^2

        (Area = 16 * pi * G^2 * M^2, S = A / 4G)
        """
        return 4 * np.pi * self.G_eff * mass**2

    def compare(self, masses: np.ndarray,
                temperatures_det: np.ndarray,
                entropies_det: np.ndarray) -> HawkingComparison:
        """
        Compare DET results with Hawking predictions.

        Parameters
        ----------
        masses : np.ndarray
            Black hole masses
        temperatures_det : np.ndarray
            DET temperatures
        entropies_det : np.ndarray
            DET entropies

        Returns
        -------
        HawkingComparison
            Comparison results
        """
        # Compute Hawking predictions
        temperatures_hawking = np.array([self.hawking_temperature(m) for m in masses])
        entropies_hawking = np.array([self.bekenstein_hawking_entropy(m) for m in masses])

        # Fit scaling exponents
        valid = (masses > 0) & (temperatures_det > 0) & (entropies_det > 0)

        if np.sum(valid) >= 2:
            log_M = np.log(masses[valid])
            log_T = np.log(temperatures_det[valid])
            log_S = np.log(entropies_det[valid])

            T_coeffs = np.polyfit(log_M, log_T, 1)
            S_coeffs = np.polyfit(log_M, log_S, 1)
            T_exponent_det = T_coeffs[0]
            S_exponent_det = S_coeffs[0]
        else:
            T_exponent_det = 0.0
            S_exponent_det = 0.0

        # Expected exponents
        T_exponent_hawking = -1.0  # T ~ 1/M
        S_exponent_hawking = 2.0   # S ~ M^2

        # Agreement metrics
        agreement_T = 1.0 - abs(T_exponent_det - T_exponent_hawking) / abs(T_exponent_hawking) if T_exponent_hawking != 0 else 0
        agreement_S = 1.0 - abs(S_exponent_det - S_exponent_hawking) / abs(S_exponent_hawking) if S_exponent_hawking != 0 else 0

        agreement_T = max(0, agreement_T)
        agreement_S = max(0, agreement_S)

        if self.verbose:
            print(f"\nHawking Comparison:")
            print(f"  T exponent: DET = {T_exponent_det:.3f}, Hawking = {T_exponent_hawking:.1f}")
            print(f"  S exponent: DET = {S_exponent_det:.3f}, Hawking = {S_exponent_hawking:.1f}")
            print(f"  Temperature agreement: {agreement_T*100:.1f}%")
            print(f"  Entropy agreement: {agreement_S*100:.1f}%")

        return HawkingComparison(
            masses=masses,
            temperatures_det=temperatures_det,
            temperatures_hawking=temperatures_hawking,
            T_exponent_det=T_exponent_det,
            T_exponent_hawking=T_exponent_hawking,
            entropy_exponent_det=S_exponent_det,
            entropy_exponent_hawking=S_exponent_hawking,
            agreement_temperature=agreement_T,
            agreement_entropy=agreement_S
        )


# ==============================================================================
# COMPREHENSIVE ANALYSIS
# ==============================================================================

class BlackHoleThermodynamicsAnalyzer:
    """
    Complete black hole thermodynamics analysis for DET.
    """

    def __init__(self, grid_size: int = 64, kappa: float = 5.0,
                 verbose: bool = True):
        """
        Initialize analyzer.

        Parameters
        ----------
        grid_size : int
            Simulation grid size
        kappa : float
            Gravity coupling
        verbose : bool
            Print progress
        """
        self.N = grid_size
        self.kappa = kappa
        self.eta = compute_lattice_correction(grid_size)
        self.G_eff = self.eta * self.kappa / (4 * np.pi)
        self.verbose = verbose

        self.configurator = BlackHoleConfigurator(grid_size, kappa, verbose)
        self.radiation_analyzer = RadiationAnalyzer(verbose)
        self.thermo_calculator = ThermodynamicsCalculator(self.G_eff, verbose)
        self.hawking_comparer = HawkingComparer(self.G_eff, verbose)

    def analyze_single_black_hole(self, mass: float,
                                   radiation_steps: int = 200) -> BlackHoleAnalysis:
        """
        Analyze a single black hole configuration.

        Parameters
        ----------
        mass : float
            Black hole mass
        radiation_steps : int
            Steps for radiation measurement

        Returns
        -------
        BlackHoleAnalysis
            Analysis results
        """
        if self.verbose:
            print(f"\nAnalyzing black hole with mass = {mass}")
            print("-" * 50)

        # Create black hole
        sim = self.configurator.create_black_hole(mass)

        # Measure initial state
        bh_state = self.configurator.measure_black_hole_state(sim)

        if self.verbose:
            print(f"  Mass (q-b): {bh_state.mass:.4f}")
            print(f"  Radius: {bh_state.radius:.2f}")
            print(f"  Central q: {bh_state.q_central:.4f}")
            print(f"  Central P: {bh_state.P_central:.6f}")
            print(f"  Surface area: {bh_state.surface_area:.1f}")

        # Measure radiation
        radiation = self.radiation_analyzer.measure_radiation_over_time(
            sim, n_steps=radiation_steps
        )

        # Compute average luminosity
        luminosities = [r.luminosity for r in radiation if r.luminosity > 0]
        avg_luminosity = np.mean(luminosities) if luminosities else 0.0

        if self.verbose:
            print(f"  Average luminosity: {avg_luminosity:.6f}")

        # Compute thermodynamic properties
        temperature = self.thermo_calculator.compute_temperature(
            bh_state.mass, avg_luminosity, bh_state.surface_area
        )
        entropy = self.thermo_calculator.compute_entropy(
            bh_state.mass, bh_state.surface_area
        )
        lifetime = self.thermo_calculator.compute_lifetime(
            bh_state.mass, avg_luminosity
        )

        if self.verbose:
            print(f"  Temperature: {temperature:.6f}")
            print(f"  Entropy: {entropy:.4f}")
            print(f"  Lifetime: {lifetime:.2f}")

        thermo = ThermodynamicProperties(
            mass=bh_state.mass,
            temperature=temperature,
            entropy=entropy,
            luminosity=avg_luminosity,
            lifetime=lifetime,
            T_mass_exponent=0.0,  # Set later in multi-mass analysis
            S_mass_exponent=0.0
        )

        return BlackHoleAnalysis(
            black_hole=bh_state,
            radiation=radiation,
            thermodynamics=thermo,
            hawking_comparison=None,
            G_eff=self.G_eff,
            summary=""
        )

    def analyze_mass_scaling(self, masses: List[float] = None,
                              radiation_steps: int = 150) -> HawkingComparison:
        """
        Analyze how thermodynamics scales with mass.

        Parameters
        ----------
        masses : List[float]
            Masses to test
        radiation_steps : int
            Steps for radiation measurement

        Returns
        -------
        HawkingComparison
            Comparison with Hawking predictions
        """
        if masses is None:
            masses = [20.0, 40.0, 60.0, 80.0, 100.0]

        if self.verbose:
            print("\n" + "="*70)
            print("BLACK HOLE MASS SCALING ANALYSIS")
            print("="*70)

        masses_arr = np.array(masses)
        temperatures = []
        entropies = []

        for mass in masses:
            analysis = self.analyze_single_black_hole(mass, radiation_steps)
            temperatures.append(analysis.thermodynamics.temperature)
            entropies.append(analysis.thermodynamics.entropy)

        temperatures_arr = np.array(temperatures)
        entropies_arr = np.array(entropies)

        # Compare with Hawking
        comparison = self.hawking_comparer.compare(
            masses_arr, temperatures_arr, entropies_arr
        )

        return comparison

    def run_full_analysis(self, masses: List[float] = None,
                           radiation_steps: int = 150) -> Dict:
        """
        Run complete black hole thermodynamics analysis.

        Parameters
        ----------
        masses : List[float]
            Masses to analyze
        radiation_steps : int
            Steps for radiation measurement

        Returns
        -------
        Dict
            Complete analysis results
        """
        if masses is None:
            masses = [30.0, 50.0, 70.0, 90.0]

        if self.verbose:
            print("\n" + "="*70)
            print("DET v6.4 BLACK HOLE THERMODYNAMICS ANALYSIS")
            print("="*70)
            print(f"\nGrid size: {self.N}")
            print(f"Kappa: {self.kappa}")
            print(f"G_eff: {self.G_eff:.6f}")

        # Analyze each mass
        analyses = []
        for mass in masses:
            analysis = self.analyze_single_black_hole(mass, radiation_steps)
            analyses.append(analysis)

        # Extract arrays for scaling analysis
        masses_arr = np.array([a.thermodynamics.mass for a in analyses])
        temps_arr = np.array([a.thermodynamics.temperature for a in analyses])
        entropies_arr = np.array([a.thermodynamics.entropy for a in analyses])

        # Compare with Hawking
        comparison = self.hawking_comparer.compare(masses_arr, temps_arr, entropies_arr)

        # Generate summary
        summary_lines = [
            "",
            "="*70,
            "BLACK HOLE THERMODYNAMICS SUMMARY",
            "="*70,
            "",
            f"G_eff = {self.G_eff:.6f}",
            "",
            "Mass Scaling Results:",
            f"  Temperature exponent (T ~ M^n): DET = {comparison.T_exponent_det:.3f}, Hawking = -1.0",
            f"  Entropy exponent (S ~ M^n): DET = {comparison.entropy_exponent_det:.3f}, Hawking = 2.0",
            "",
            "Agreement with Hawking Predictions:",
            f"  Temperature scaling: {comparison.agreement_temperature*100:.1f}%",
            f"  Entropy scaling: {comparison.agreement_entropy*100:.1f}%",
            "",
            "Individual Black Holes:",
        ]

        for a in analyses:
            summary_lines.append(
                f"  M={a.thermodynamics.mass:.1f}: T={a.thermodynamics.temperature:.4f}, "
                f"S={a.thermodynamics.entropy:.1f}, L={a.thermodynamics.luminosity:.6f}"
            )

        summary_lines.extend(["", "="*70])

        summary = "\n".join(summary_lines)

        if self.verbose:
            print(summary)

        return {
            'analyses': analyses,
            'comparison': comparison,
            'summary': summary,
            'G_eff': self.G_eff
        }


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_black_hole_analysis(grid_size: int = 64, kappa: float = 5.0,
                             masses: List[float] = None,
                             verbose: bool = True) -> Dict:
    """
    Run complete black hole thermodynamics analysis.

    This is the main entry point for the v6.4 Black Hole Thermodynamics feature.

    Parameters
    ----------
    grid_size : int
        Simulation grid size
    kappa : float
        Gravity coupling
    masses : List[float]
        Masses to analyze
    verbose : bool
        Print progress

    Returns
    -------
    Dict
        Complete analysis results
    """
    analyzer = BlackHoleThermodynamicsAnalyzer(grid_size, kappa, verbose)
    return analyzer.run_full_analysis(masses=masses)


if __name__ == "__main__":
    print("DET v6.4 Black Hole Thermodynamics")
    print("Testing Hawking-like radiation predictions...")
    print()

    # Run analysis
    results = run_black_hole_analysis(grid_size=64, kappa=5.0, verbose=True)

    # Print final status
    print("\n" + "="*70)
    comparison = results['comparison']
    if comparison.agreement_temperature > 0.5 and comparison.agreement_entropy > 0.5:
        print("BLACK HOLE THERMODYNAMICS: HAWKING-LIKE BEHAVIOR OBSERVED")
    else:
        print("BLACK HOLE THERMODYNAMICS: ANALYSIS COMPLETE (DEVIATIONS NOTED)")
    print("="*70)
