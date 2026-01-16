"""
DET v6.4 External Calibration: Extract Effective G from Two-Body Simulations
=============================================================================

Roadmap Item #1: Extract effective gravitational constant G_eff from DET
simulations and compare with theoretical prediction.

Theoretical Framework
---------------------
DET's effective gravitational constant emerges from the Poisson equation:
    L*Phi = kappa*rho

In continuum limit, for point mass: Phi(r) = -kappa*M/(4*pi*r)

Comparing with Newtonian gravity Phi(r) = -G*M/r:
    G_eff = eta * kappa / (4*pi)

where eta is the lattice correction factor (0.968 for N=64).

Extraction Methods
------------------
1. **Orbital Method (Kepler):** From two-body orbital dynamics:
   T^2 = (4*pi^2 / G*M) * r^3  =>  G = 4*pi^2 * r^3 / (M * T^2)

2. **Potential Method:** From gravitational potential profile:
   Phi(r) = -G*M/r  =>  G = |Phi(r)| * r / M

Reference: DET Theory Card v6.3, Appendix C
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D, compute_lattice_correction


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class GExtractionResult:
    """Results from a single G extraction measurement."""
    method: str                  # 'orbital' or 'potential'
    G_extracted: float          # Extracted G value (lattice units)
    G_theoretical: float        # Theoretical G_eff = eta*kappa/(4*pi)
    relative_error: float       # |G_extracted - G_theoretical| / G_theoretical
    parameters: Dict            # Measurement parameters
    uncertainty: float = 0.0    # Estimated uncertainty


@dataclass
class CalibrationReport:
    """Complete calibration report with all measurements."""
    orbital_results: List[GExtractionResult]
    potential_results: List[GExtractionResult]
    G_orbital_mean: float
    G_orbital_std: float
    G_potential_mean: float
    G_potential_std: float
    G_theoretical: float
    kappa: float
    eta: float
    grid_size: int
    calibration_passed: bool
    summary: str = ""


# ==============================================================================
# THEORETICAL G CALCULATION
# ==============================================================================

def compute_G_theoretical(kappa: float, eta: float) -> float:
    """
    Compute theoretical G_eff from DET parameters.

    G_eff = eta * kappa / (4*pi)

    Parameters
    ----------
    kappa : float
        Poisson coupling constant
    eta : float
        Lattice correction factor

    Returns
    -------
    float
        Theoretical G_eff in lattice units
    """
    return eta * kappa / (4 * np.pi)


# ==============================================================================
# POTENTIAL PROFILE EXTRACTION
# ==============================================================================

class PotentialProfileExtractor:
    """
    Extract G_eff from gravitational potential profile.

    Method: Place point mass, solve Poisson, fit Phi(r) = -G*M/r + const
    """

    def __init__(self, grid_size: int = 64, kappa: float = 5.0, verbose: bool = True):
        """
        Initialize extractor.

        Parameters
        ----------
        grid_size : int
            Simulation grid size (N x N x N)
        kappa : float
            Poisson coupling constant
        verbose : bool
            Print progress information
        """
        self.N = grid_size
        self.kappa = kappa
        self.eta = compute_lattice_correction(grid_size)
        self.verbose = verbose
        self.center = grid_size // 2

    def setup_point_mass(self, mass: float, width: float = 2.0) -> DETCollider3D:
        """
        Create simulation with central point mass.

        Parameters
        ----------
        mass : float
            Total mass (in F units) - used to scale initial_q
        width : float
            Gaussian width of mass distribution

        Returns
        -------
        DETCollider3D
            Configured simulation
        """
        params = DETParams3D(
            N=self.N,
            DT=0.01,
            F_VAC=0.001,
            F_MIN=0.0,

            # Gravity configuration
            gravity_enabled=True,
            kappa_grav=self.kappa,
            alpha_grav=0.01,
            mu_grav=1.0,
            beta_g=5.0,
            eta_lattice=self.eta,

            # Enable structure for gravity sourcing
            q_enabled=True,
            alpha_q=0.01,

            # Disable features that might interfere
            momentum_enabled=False,
            angular_momentum_enabled=False,
            floor_enabled=False,
            boundary_enabled=False,
            agency_dynamic=False,
            sigma_dynamic=False,
            coherence_dynamic=False
        )

        sim = DETCollider3D(params)

        # Add central mass with high structure q (sources gravity)
        # Note: Gravity is sourced by (q - b), not by F directly
        sim.add_packet(
            (self.center, self.center, self.center),
            mass=mass,
            width=width,
            momentum=(0, 0, 0),
            initial_q=0.9
        )

        return sim

    def measure_gravitational_mass(self, sim: DETCollider3D) -> float:
        """
        Measure the effective gravitational mass from simulation.

        In DET, gravity is sourced by rho = q - b (structural debt),
        not by the F field directly. This computes sum(q - b).

        Parameters
        ----------
        sim : DETCollider3D
            Simulation with established gravitational field

        Returns
        -------
        float
            Effective gravitational mass (sum of q - b)
        """
        # Compute baseline b from Helmholtz solve
        # b is already computed during gravity step, but we need q - b
        rho = sim.q - sim.b
        return np.sum(rho)

    def measure_potential_profile(self, sim: DETCollider3D,
                                   r_min: int = 5, r_max: int = None,
                                   n_samples: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measure gravitational potential as function of radius.

        Parameters
        ----------
        sim : DETCollider3D
            Simulation with established gravitational field
        r_min : int
            Minimum radius to sample (avoid core)
        r_max : int
            Maximum radius to sample
        n_samples : int
            Number of radial samples

        Returns
        -------
        radii : np.ndarray
            Radial distances
        potentials : np.ndarray
            Potential values at each radius
        """
        if r_max is None:
            r_max = self.N // 2 - 2

        # Sample radii
        radii = np.linspace(r_min, r_max, n_samples)
        potentials = []

        center = self.center

        for r in radii:
            r_int = int(round(r))

            # Average potential on spherical shell
            phi_samples = []

            # Sample in 6 cardinal directions
            directions = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1)
            ]

            for dx, dy, dz in directions:
                xi = (center + dx * r_int) % self.N
                yi = (center + dy * r_int) % self.N
                zi = (center + dz * r_int) % self.N
                phi_samples.append(sim.Phi[xi, yi, zi])

            # Also sample along diagonals for better averaging
            diagonals = [
                (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
            ]

            r_diag = int(round(r / np.sqrt(2)))
            for dx, dy, dz in diagonals:
                xi = (center + dx * r_diag) % self.N
                yi = (center + dy * r_diag) % self.N
                zi = (center + dz * r_diag) % self.N
                phi_samples.append(sim.Phi[xi, yi, zi])

            potentials.append(np.mean(phi_samples))

        return np.array(radii), np.array(potentials)

    def fit_potential_profile(self, radii: np.ndarray, potentials: np.ndarray,
                               mass: float) -> Dict:
        """
        Fit potential profile to extract G.

        Fit: Phi(r) = -G*M/r + B (constant offset)

        Parameters
        ----------
        radii : np.ndarray
            Radial distances
        potentials : np.ndarray
            Potential values
        mass : float
            Central mass

        Returns
        -------
        dict
            Fit results including extracted G
        """
        # Define fit function: Phi = A/r + B where A = -G*M
        def phi_fit(r, A, B):
            return A / r + B

        # Initial guess
        p0 = [potentials[0] * radii[0], 0.0]

        try:
            popt, pcov = curve_fit(phi_fit, radii, potentials, p0=p0)
            A_fit, B_fit = popt
            A_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0

            # Extract G: A = -G*M => G = -A/M
            # Note: potentials should be negative for attractive gravity
            G_extracted = -A_fit / mass
            G_uncertainty = A_err / mass

            # Compute fit quality
            phi_predicted = phi_fit(radii, *popt)
            residuals = potentials - phi_predicted
            r_squared = 1 - np.sum(residuals**2) / np.sum((potentials - np.mean(potentials))**2)

            return {
                'G_extracted': G_extracted,
                'G_uncertainty': G_uncertainty,
                'A': A_fit,
                'B': B_fit,
                'r_squared': r_squared,
                'success': True
            }

        except Exception as e:
            if self.verbose:
                print(f"  Fit failed: {e}")
            return {
                'G_extracted': 0.0,
                'G_uncertainty': float('inf'),
                'success': False
            }

    def extract_G(self, mass: float = 50.0, width: float = 2.0,
                  settling_steps: int = 100) -> GExtractionResult:
        """
        Extract G_eff from potential profile.

        Parameters
        ----------
        mass : float
            Central mass (F units) - used to scale initial structure
        width : float
            Mass distribution width
        settling_steps : int
            Steps to let field establish

        Returns
        -------
        GExtractionResult
            Extraction results
        """
        if self.verbose:
            print(f"\nPotential Profile Extraction (M_input={mass}, width={width})")
            print("-" * 50)

        # Setup simulation
        sim = self.setup_point_mass(mass, width)

        # Let gravitational field establish
        if self.verbose:
            print(f"  Establishing field ({settling_steps} steps)...")
        for _ in range(settling_steps):
            sim.step()

        # Measure the ACTUAL gravitational mass (q - b), not the input mass
        # In DET, gravity is sourced by structural debt, not F directly
        M_grav = self.measure_gravitational_mass(sim)
        if self.verbose:
            print(f"  Gravitational mass (sum of q-b) = {M_grav:.4f}")

        # Measure potential profile
        if self.verbose:
            print("  Measuring potential profile...")
        radii, potentials = self.measure_potential_profile(sim)

        # Fit and extract G using the actual gravitational mass
        if self.verbose:
            print("  Fitting Phi(r) = A/r + B...")
        fit_result = self.fit_potential_profile(radii, potentials, M_grav)

        # Theoretical prediction
        G_theoretical = compute_G_theoretical(self.kappa, self.eta)

        if fit_result['success']:
            G_extracted = fit_result['G_extracted']
            rel_error = abs(G_extracted - G_theoretical) / G_theoretical

            if self.verbose:
                print(f"  G_extracted  = {G_extracted:.6f}")
                print(f"  G_theoretical = {G_theoretical:.6f}")
                print(f"  Relative error = {rel_error*100:.2f}%")
                print(f"  R² = {fit_result['r_squared']:.4f}")

            return GExtractionResult(
                method='potential',
                G_extracted=G_extracted,
                G_theoretical=G_theoretical,
                relative_error=rel_error,
                uncertainty=fit_result['G_uncertainty'],
                parameters={
                    'mass_input': mass,
                    'mass_grav': M_grav,
                    'width': width,
                    'kappa': self.kappa,
                    'eta': self.eta,
                    'grid_size': self.N,
                    'r_squared': fit_result['r_squared']
                }
            )
        else:
            return GExtractionResult(
                method='potential',
                G_extracted=0.0,
                G_theoretical=G_theoretical,
                relative_error=1.0,
                uncertainty=float('inf'),
                parameters={
                    'mass_input': mass,
                    'mass_grav': M_grav,
                    'width': width,
                    'success': False
                }
            )


# ==============================================================================
# ORBITAL (KEPLER) EXTRACTION
# ==============================================================================

class OrbitalExtractor:
    """
    Extract G_eff from orbital dynamics using Kepler's Third Law.

    Method: Run two-body simulation, measure T and r, compute G = 4*pi^2*r^3/(M*T^2)
    """

    def __init__(self, grid_size: int = 64, kappa: float = 5.0, verbose: bool = True):
        """
        Initialize orbital extractor.
        """
        self.N = grid_size
        self.kappa = kappa
        self.eta = compute_lattice_correction(grid_size)
        self.verbose = verbose
        self.center = grid_size // 2

    def setup_two_body(self, central_mass: float, test_mass: float,
                       orbital_radius: float, width: float = 2.0) -> Tuple[DETCollider3D, float]:
        """
        Setup two-body problem with circular orbit.

        Parameters
        ----------
        central_mass : float
            Mass of central body
        test_mass : float
            Mass of orbiting body
        orbital_radius : float
            Initial orbital radius
        width : float
            Mass distribution width

        Returns
        -------
        sim : DETCollider3D
            Configured simulation
        v_orbital : float
            Estimated circular orbital velocity
        """
        params = DETParams3D(
            N=self.N,
            DT=0.015,
            F_VAC=0.001,
            F_MIN=0.0,

            # Gravity
            gravity_enabled=True,
            kappa_grav=self.kappa,
            alpha_grav=0.02,
            mu_grav=2.0,
            beta_g=10.0,
            eta_lattice=self.eta,

            # Momentum for dynamics
            momentum_enabled=True,
            alpha_pi=0.1,
            lambda_pi=0.0005,  # Very low decay
            mu_pi=0.5,

            # Structure
            q_enabled=True,
            alpha_q=0.01,

            # Disable interfering features
            angular_momentum_enabled=False,
            floor_enabled=False,
            boundary_enabled=False,
            agency_dynamic=False,
            sigma_dynamic=False,
            coherence_dynamic=False
        )

        sim = DETCollider3D(params)
        center = self.center

        # Add central mass
        sim.add_packet(
            (center, center, center),
            mass=central_mass,
            width=width,
            momentum=(0, 0, 0),
            initial_q=0.9
        )

        # Let field establish
        for _ in range(50):
            sim.step()

        # Estimate orbital velocity from gravitational field
        r_int = min(int(round(orbital_radius)), self.N // 2 - 2)

        # Sample gravitational acceleration
        g_mag = np.sqrt(
            sim.gx[center, center, center + r_int]**2 +
            sim.gy[center, center, center + r_int]**2 +
            sim.gz[center, center, center + r_int]**2
        )

        # v_circular = sqrt(r * g) for circular orbit
        v_orbital = np.sqrt(orbital_radius * g_mag) if g_mag > 0 else 0.1

        # Add test particle with tangential velocity
        sim.add_packet(
            (center, center, center + r_int),
            mass=test_mass,
            width=1.5,
            momentum=(0, v_orbital, 0),  # Tangential in Y
            initial_q=0.05
        )

        return sim, v_orbital

    def measure_orbital_period(self, sim: DETCollider3D,
                                max_steps: int = 15000,
                                target_orbits: float = 3.0) -> Dict:
        """
        Measure orbital period from simulation.

        Parameters
        ----------
        sim : DETCollider3D
            Two-body simulation
        max_steps : int
            Maximum simulation steps
        target_orbits : float
            Target number of orbits to measure

        Returns
        -------
        dict
            Orbital measurements including period
        """
        center = self.center

        # Track orbital motion
        prev_angle = 0.0
        total_angle = 0.0
        distances = []
        times = []

        for t in range(max_steps):
            sim.step()

            # Find blobs
            blobs = sim.find_blobs(threshold_ratio=5.0)

            if len(blobs) >= 2:
                # Identify central and orbiting body
                if blobs[0]['mass'] > blobs[1]['mass']:
                    central_blob = blobs[0]
                    test_blob = blobs[1]
                else:
                    central_blob = blobs[1]
                    test_blob = blobs[0]

                # Relative position (with periodic boundary handling)
                dx = test_blob['x'] - central_blob['x']
                dy = test_blob['y'] - central_blob['y']
                dz = test_blob['z'] - central_blob['z']

                # Handle periodic boundaries
                if dx > self.N/2: dx -= self.N
                if dx < -self.N/2: dx += self.N
                if dy > self.N/2: dy -= self.N
                if dy < -self.N/2: dy += self.N
                if dz > self.N/2: dz -= self.N
                if dz < -self.N/2: dz += self.N

                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                distances.append(distance)
                times.append(t)

                # Track angle in YZ plane (orbit plane)
                angle = np.arctan2(dy, dz)

                d_angle = angle - prev_angle
                if d_angle > np.pi: d_angle -= 2*np.pi
                if d_angle < -np.pi: d_angle += 2*np.pi
                total_angle += d_angle
                prev_angle = angle

                num_orbits = abs(total_angle) / (2 * np.pi)

                if num_orbits >= target_orbits:
                    break
            else:
                if t > 500:
                    break

        if len(distances) < 100:
            return {'success': False, 'reason': 'insufficient_data'}

        distances = np.array(distances)
        times = np.array(times)

        num_orbits = abs(total_angle) / (2 * np.pi)

        if num_orbits < 0.5:
            return {'success': False, 'reason': 'orbit_unstable'}

        # Compute period
        period = len(times) / num_orbits

        # Compute eccentricity
        r_mean = np.mean(distances)
        r_min = np.min(distances)
        r_max = np.max(distances)
        eccentricity = (r_max - r_min) / (r_max + r_min) if (r_max + r_min) > 0 else 1.0

        return {
            'success': True,
            'period': period,
            'num_orbits': num_orbits,
            'r_mean': r_mean,
            'r_min': r_min,
            'r_max': r_max,
            'eccentricity': eccentricity
        }

    def measure_gravitational_mass(self, sim: DETCollider3D) -> float:
        """
        Measure effective gravitational mass from simulation (q - b).
        """
        rho = sim.q - sim.b
        return np.sum(rho)

    def extract_G(self, central_mass: float = 80.0, test_mass: float = 0.5,
                  orbital_radius: float = 10.0) -> GExtractionResult:
        """
        Extract G from Kepler's Third Law.

        G = 4*pi^2 * r^3 / (M * T^2)

        Parameters
        ----------
        central_mass : float
            Input mass for central body (F units)
        test_mass : float
            Mass of orbiting test particle
        orbital_radius : float
            Initial orbital radius

        Returns
        -------
        GExtractionResult
            Extraction results
        """
        if self.verbose:
            print(f"\nOrbital (Kepler) Extraction (M_input={central_mass}, r={orbital_radius})")
            print("-" * 50)

        # Setup two-body problem
        if self.verbose:
            print("  Setting up two-body system...")
        sim, v_orbital = self.setup_two_body(central_mass, test_mass, orbital_radius)

        # Measure the actual gravitational mass BEFORE adding test particle dynamics
        # Since we just added the test particle, measure q-b from central mass region
        M_grav = self.measure_gravitational_mass(sim)
        if self.verbose:
            print(f"  Gravitational mass (sum of q-b) = {M_grav:.4f}")
            print(f"  Initial orbital velocity: {v_orbital:.4f}")
            print("  Running orbital simulation...")

        # Measure orbital period
        result = self.measure_orbital_period(sim)

        # Theoretical prediction
        G_theoretical = compute_G_theoretical(self.kappa, self.eta)

        if result['success']:
            T = result['period']
            r = result['r_mean']  # Use mean radius for better accuracy

            # Use the measured gravitational mass (q - b), not input mass
            M = M_grav

            # Kepler's Third Law: T^2 = (4*pi^2 / G*M) * r^3
            # => G = 4*pi^2 * r^3 / (M * T^2)
            G_extracted = 4 * np.pi**2 * r**3 / (M * T**2)

            rel_error = abs(G_extracted - G_theoretical) / G_theoretical

            # Estimate uncertainty from eccentricity (proxy for orbit stability)
            uncertainty = G_extracted * result['eccentricity']

            if self.verbose:
                print(f"  Period T = {T:.2f} steps")
                print(f"  Mean radius = {r:.2f}")
                print(f"  Eccentricity = {result['eccentricity']:.3f}")
                print(f"  T^2/r^3 = {T**2 / r**3:.4f}")
                print(f"  G_extracted  = {G_extracted:.6f}")
                print(f"  G_theoretical = {G_theoretical:.6f}")
                print(f"  Relative error = {rel_error*100:.2f}%")

            return GExtractionResult(
                method='orbital',
                G_extracted=G_extracted,
                G_theoretical=G_theoretical,
                relative_error=rel_error,
                uncertainty=uncertainty,
                parameters={
                    'central_mass_input': central_mass,
                    'central_mass_grav': M_grav,
                    'test_mass': test_mass,
                    'orbital_radius': orbital_radius,
                    'period': T,
                    'r_mean': r,
                    'eccentricity': result['eccentricity'],
                    'num_orbits': result['num_orbits'],
                    'kappa': self.kappa,
                    'eta': self.eta,
                    'grid_size': self.N
                }
            )
        else:
            if self.verbose:
                print(f"  Orbital measurement failed: {result.get('reason', 'unknown')}")

            return GExtractionResult(
                method='orbital',
                G_extracted=0.0,
                G_theoretical=G_theoretical,
                relative_error=1.0,
                uncertainty=float('inf'),
                parameters={
                    'central_mass_input': central_mass,
                    'orbital_radius': orbital_radius,
                    'success': False,
                    'reason': result.get('reason', 'unknown')
                }
            )


# ==============================================================================
# COMPREHENSIVE CALIBRATION
# ==============================================================================

class GCalibrator:
    """
    Comprehensive G extraction and calibration.

    Runs multiple extraction methods at various parameters and
    computes calibration statistics.
    """

    def __init__(self, grid_size: int = 64, kappa: float = 5.0, verbose: bool = True):
        """
        Initialize calibrator.
        """
        self.N = grid_size
        self.kappa = kappa
        self.eta = compute_lattice_correction(grid_size)
        self.verbose = verbose

        self.potential_extractor = PotentialProfileExtractor(grid_size, kappa, verbose)
        self.orbital_extractor = OrbitalExtractor(grid_size, kappa, verbose)

    def run_potential_calibration(self, masses: List[float] = None) -> List[GExtractionResult]:
        """
        Run potential profile extractions at multiple masses.
        """
        if masses is None:
            masses = [30.0, 50.0, 80.0]

        if self.verbose:
            print("\n" + "="*70)
            print("POTENTIAL PROFILE G EXTRACTION")
            print("="*70)

        results = []
        for mass in masses:
            result = self.potential_extractor.extract_G(mass=mass)
            results.append(result)

        return results

    def run_orbital_calibration(self, radii: List[float] = None,
                                 central_mass: float = 80.0) -> List[GExtractionResult]:
        """
        Run orbital extractions at multiple radii.
        """
        if radii is None:
            radii = [8.0, 10.0, 12.0, 15.0]

        if self.verbose:
            print("\n" + "="*70)
            print("ORBITAL (KEPLER) G EXTRACTION")
            print("="*70)

        results = []
        for r in radii:
            result = self.orbital_extractor.extract_G(
                central_mass=central_mass,
                orbital_radius=r
            )
            results.append(result)

        return results

    def run_full_calibration(self,
                             masses: List[float] = None,
                             radii: List[float] = None,
                             central_mass: float = 80.0) -> CalibrationReport:
        """
        Run complete G calibration using both methods.

        Parameters
        ----------
        masses : List[float]
            Masses for potential profile extraction
        radii : List[float]
            Orbital radii for Kepler extraction
        central_mass : float
            Central mass for orbital tests

        Returns
        -------
        CalibrationReport
            Complete calibration report
        """
        if self.verbose:
            print("\n" + "="*70)
            print("DET v6.4 EXTERNAL CALIBRATION: G EXTRACTION")
            print("="*70)
            print(f"\nGrid size: {self.N}")
            print(f"Kappa: {self.kappa}")
            print(f"Eta (lattice correction): {self.eta}")
            print(f"Theoretical G_eff = eta*kappa/(4*pi) = {compute_G_theoretical(self.kappa, self.eta):.6f}")

        # Run both extraction methods
        potential_results = self.run_potential_calibration(masses)
        orbital_results = self.run_orbital_calibration(radii, central_mass)

        # Compute statistics
        G_theoretical = compute_G_theoretical(self.kappa, self.eta)

        # Potential method statistics
        valid_potential = [r for r in potential_results if r.relative_error < 1.0]
        if valid_potential:
            G_pot_values = [r.G_extracted for r in valid_potential]
            G_pot_mean = np.mean(G_pot_values)
            G_pot_std = np.std(G_pot_values) if len(G_pot_values) > 1 else 0.0
        else:
            G_pot_mean = 0.0
            G_pot_std = float('inf')

        # Orbital method statistics
        valid_orbital = [r for r in orbital_results if r.relative_error < 1.0]
        if valid_orbital:
            G_orb_values = [r.G_extracted for r in valid_orbital]
            G_orb_mean = np.mean(G_orb_values)
            G_orb_std = np.std(G_orb_values) if len(G_orb_values) > 1 else 0.0
        else:
            G_orb_mean = 0.0
            G_orb_std = float('inf')

        # Check calibration success
        pot_error = abs(G_pot_mean - G_theoretical) / G_theoretical if G_pot_mean > 0 else 1.0
        orb_error = abs(G_orb_mean - G_theoretical) / G_theoretical if G_orb_mean > 0 else 1.0

        # Pass if either method agrees within 20%
        calibration_passed = pot_error < 0.20 or orb_error < 0.20

        # Generate summary
        summary_lines = [
            "",
            "=" * 70,
            "CALIBRATION SUMMARY",
            "=" * 70,
            "",
            f"Theoretical G_eff = {G_theoretical:.6f}",
            "",
            "Potential Profile Method:",
            f"  G_mean = {G_pot_mean:.6f} +/- {G_pot_std:.6f}",
            f"  Error from theory: {pot_error*100:.1f}%",
            f"  Valid measurements: {len(valid_potential)}/{len(potential_results)}",
            "",
            "Orbital (Kepler) Method:",
            f"  G_mean = {G_orb_mean:.6f} +/- {G_orb_std:.6f}",
            f"  Error from theory: {orb_error*100:.1f}%",
            f"  Valid measurements: {len(valid_orbital)}/{len(orbital_results)}",
            "",
            "=" * 70,
        ]

        if calibration_passed:
            summary_lines.extend([
                "CALIBRATION PASSED",
                "DET gravity correctly reproduces G_eff = eta*kappa/(4*pi)",
                "="*70
            ])
        else:
            summary_lines.extend([
                "CALIBRATION NEEDS REVIEW",
                "Extracted G deviates >20% from theoretical prediction",
                "="*70
            ])

        summary = "\n".join(summary_lines)

        if self.verbose:
            print(summary)

        return CalibrationReport(
            orbital_results=orbital_results,
            potential_results=potential_results,
            G_orbital_mean=G_orb_mean,
            G_orbital_std=G_orb_std,
            G_potential_mean=G_pot_mean,
            G_potential_std=G_pot_std,
            G_theoretical=G_theoretical,
            kappa=self.kappa,
            eta=self.eta,
            grid_size=self.N,
            calibration_passed=calibration_passed,
            summary=summary
        )


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_g_calibration(grid_size: int = 64, kappa: float = 5.0,
                      verbose: bool = True) -> CalibrationReport:
    """
    Run complete G extraction calibration.

    This is the main entry point for the v6.4 External Calibration feature.

    Parameters
    ----------
    grid_size : int
        Simulation grid size
    kappa : float
        Poisson coupling constant
    verbose : bool
        Print progress

    Returns
    -------
    CalibrationReport
        Complete calibration report
    """
    calibrator = GCalibrator(grid_size, kappa, verbose)
    return calibrator.run_full_calibration()


if __name__ == "__main__":
    print("DET v6.4 External Calibration")
    print("Extracting effective G from two-body simulations...")
    print()

    # Run calibration with default parameters
    report = run_g_calibration(grid_size=64, kappa=5.0, verbose=True)

    # Print final status
    print("\n" + "="*70)
    if report.calibration_passed:
        print("SUCCESS: G extraction calibration PASSED")
    else:
        print("REVIEW NEEDED: G extraction calibration requires attention")
    print("="*70)
