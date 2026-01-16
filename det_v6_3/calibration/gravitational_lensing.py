"""
DET v6.4 Gravitational Lensing: Ray-Tracing Through Φ Field
============================================================

Roadmap Item #3: Implement gravitational lensing via ray-tracing
through the DET gravitational potential field.

Theory Overview
---------------
In the weak-field limit, light follows geodesics described by:
    d²x/dλ² = -∇Φ

where λ is an affine parameter along the ray.

The deflection angle for a light ray passing a mass is:
    α = (2/c²) ∫ ∇⊥Φ dl

For a point mass (Schwarzschild weak-field limit):
    α = 4GM/(c²b)

where b is the impact parameter (closest approach distance).

DET Implementation
------------------
DET computes Φ via the Poisson equation: L*Φ = κ*ρ

The effective gravitational constant is G_eff = ηκ/(4π).

For lensing, we trace rays through the 3D potential field using:
1. Trilinear interpolation for Φ and ∇Φ at arbitrary positions
2. Numerical integration along the ray path
3. Accumulation of deflection angles

Reference: DET Theory Card v6.3, Section V (Gravity Module)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D, compute_lattice_correction
from det_si_units import (
    DETUnitSystem, galactic_units, solar_system_units,
    G_SI, C_SI, M_SUN, PC, AU
)


# ==============================================================================
# PHYSICAL CONSTANTS FOR LENSING
# ==============================================================================

# Speed of light in lattice units (maximum information speed)
C_LATTICE = 1.0  # 1 cell per step


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class RayPath:
    """Traced light ray path through gravitational field."""
    positions: np.ndarray      # [N, 3] array of (x, y, z) positions
    velocities: np.ndarray     # [N, 3] array of velocity vectors
    potentials: np.ndarray     # [N] array of Φ values along path
    n_steps: int               # Number of integration steps
    impact_parameter: float    # Closest approach to center
    total_deflection: float    # Total deflection angle (radians in lattice)


@dataclass
class LensingResult:
    """Results from gravitational lensing analysis."""
    impact_parameters: np.ndarray   # Array of impact parameters tested
    deflection_angles: np.ndarray   # Corresponding deflection angles
    deflection_angles_si: np.ndarray  # Deflection in physical units (radians)
    einstein_radius: float          # Einstein radius (lattice units)
    einstein_radius_si: float       # Einstein radius (physical units)
    mass_lattice: float             # Gravitating mass in lattice units
    mass_si: float                  # Gravitating mass in kg
    theoretical_deflections: np.ndarray  # Schwarzschild predictions
    relative_errors: np.ndarray     # |DET - theory| / theory
    parameters: Dict


# ==============================================================================
# RAY TRACER
# ==============================================================================

class GravitationalRayTracer:
    """
    Trace light rays through DET gravitational potential field.

    Uses numerical integration of geodesic equation in weak-field limit:
        d²x/dt² = -∇Φ

    For light (c = 1 in lattice units), the trajectory bends due to
    the transverse gradient of the potential.
    """

    def __init__(self, collider: DETCollider3D, verbose: bool = True):
        """
        Initialize ray tracer with a DET simulation.

        Parameters
        ----------
        collider : DETCollider3D
            Simulation with computed gravitational potential
        verbose : bool
            Print progress information
        """
        self.collider = collider
        self.N = collider.p.N
        self.verbose = verbose
        self.center = self.N // 2

        # Setup interpolators for smooth field access
        self._setup_interpolators()

    def _setup_interpolators(self):
        """Create interpolators for potential and gravity fields."""
        # Grid coordinates
        x = np.arange(self.N)
        y = np.arange(self.N)
        z = np.arange(self.N)

        # Potential interpolator
        self.phi_interp = RegularGridInterpolator(
            (z, y, x), self.collider.Phi,
            method='linear', bounds_error=False, fill_value=0.0
        )

        # Gravity component interpolators
        self.gx_interp = RegularGridInterpolator(
            (z, y, x), self.collider.gx,
            method='linear', bounds_error=False, fill_value=0.0
        )
        self.gy_interp = RegularGridInterpolator(
            (z, y, x), self.collider.gy,
            method='linear', bounds_error=False, fill_value=0.0
        )
        self.gz_interp = RegularGridInterpolator(
            (z, y, x), self.collider.gz,
            method='linear', bounds_error=False, fill_value=0.0
        )

    def get_potential(self, pos: np.ndarray) -> float:
        """Get interpolated potential at position (x, y, z)."""
        # Handle periodic boundaries
        pos_wrapped = pos % self.N
        # Interpolator expects (z, y, x) ordering
        return float(self.phi_interp((pos_wrapped[2], pos_wrapped[1], pos_wrapped[0])))

    def get_gravity(self, pos: np.ndarray) -> np.ndarray:
        """Get interpolated gravity vector at position (x, y, z)."""
        pos_wrapped = pos % self.N
        point = (pos_wrapped[2], pos_wrapped[1], pos_wrapped[0])

        gx = float(self.gx_interp(point))
        gy = float(self.gy_interp(point))
        gz = float(self.gz_interp(point))

        return np.array([gx, gy, gz])

    def trace_ray(self, start: np.ndarray, direction: np.ndarray,
                  n_steps: int = 1000, dt: float = 0.5) -> RayPath:
        """
        Trace a light ray through the gravitational field.

        Uses velocity Verlet integration for accurate trajectory.

        Parameters
        ----------
        start : np.ndarray
            Starting position [x, y, z] in lattice units
        direction : np.ndarray
            Initial direction (will be normalized to c=1)
        n_steps : int
            Number of integration steps
        dt : float
            Time step (in lattice units)

        Returns
        -------
        RayPath
            Traced ray path with positions, deflection, etc.
        """
        # Normalize direction to speed of light
        direction = np.array(direction, dtype=float)
        direction = direction / np.linalg.norm(direction) * C_LATTICE

        # Initialize arrays
        positions = np.zeros((n_steps + 1, 3))
        velocities = np.zeros((n_steps + 1, 3))
        potentials = np.zeros(n_steps + 1)

        # Initial conditions
        pos = np.array(start, dtype=float)
        vel = direction.copy()

        positions[0] = pos
        velocities[0] = vel
        potentials[0] = self.get_potential(pos)

        # Track minimum distance to center (impact parameter)
        center = np.array([self.center, self.center, self.center], dtype=float)
        min_distance = np.linalg.norm(pos - center)

        # Velocity Verlet integration
        # For light: acceleration = -∇Φ (geodesic in weak field)
        # But we need to be careful: light follows null geodesics
        # In weak field: transverse acceleration = -∇⊥Φ

        acc = -self.get_gravity(pos)

        for i in range(n_steps):
            # Update position (half step)
            pos_half = pos + vel * dt / 2 + acc * dt**2 / 4

            # Get acceleration at new position
            acc_new = -self.get_gravity(pos_half)

            # Full position update
            pos = pos + vel * dt + acc * dt**2 / 2

            # Update velocity
            vel = vel + (acc + acc_new) * dt / 2

            # Renormalize velocity to c (light always travels at c)
            vel = vel / np.linalg.norm(vel) * C_LATTICE

            acc = -self.get_gravity(pos)

            # Store
            positions[i + 1] = pos
            velocities[i + 1] = vel
            potentials[i + 1] = self.get_potential(pos)

            # Track minimum distance
            dist = np.linalg.norm(pos - center)
            if dist < min_distance:
                min_distance = dist

        # Compute total deflection angle
        # Deflection = angle between initial and final velocity
        initial_dir = velocities[0] / np.linalg.norm(velocities[0])
        final_dir = velocities[-1] / np.linalg.norm(velocities[-1])
        cos_angle = np.clip(np.dot(initial_dir, final_dir), -1.0, 1.0)
        total_deflection = np.arccos(cos_angle)

        return RayPath(
            positions=positions,
            velocities=velocities,
            potentials=potentials,
            n_steps=n_steps,
            impact_parameter=min_distance,
            total_deflection=total_deflection
        )

    def compute_deflection(self, impact_parameter: float,
                           n_steps: int = 2000) -> Tuple[float, RayPath]:
        """
        Compute deflection angle for ray with given impact parameter.

        Sets up a ray passing the center at the specified distance
        and measures total deflection.

        Parameters
        ----------
        impact_parameter : float
            Closest approach distance to center (lattice units)
        n_steps : int
            Integration steps

        Returns
        -------
        Tuple[float, RayPath]
            Deflection angle (radians) and full ray path
        """
        # Set up ray parallel to z-axis, offset in x by impact parameter
        # Start far from center, pass through, exit other side
        start = np.array([
            self.center + impact_parameter,
            self.center,
            0.0  # Start at z=0
        ])

        direction = np.array([0.0, 0.0, 1.0])  # Moving in +z direction

        # Trace ray
        path = self.trace_ray(start, direction, n_steps=n_steps)

        return path.total_deflection, path


# ==============================================================================
# LENSING ANALYSIS
# ==============================================================================

class GravitationalLensing:
    """
    Complete gravitational lensing analysis for DET simulations.

    Computes:
    - Deflection angles as function of impact parameter
    - Einstein radius
    - Comparison with Schwarzschild (point mass) predictions
    """

    def __init__(self, grid_size: int = 64, kappa: float = 5.0,
                 units: DETUnitSystem = None, verbose: bool = True):
        """
        Initialize lensing analysis.

        Parameters
        ----------
        grid_size : int
            Simulation grid size
        kappa : float
            Poisson coupling constant
        units : DETUnitSystem
            Unit system for SI conversions
        verbose : bool
            Print progress
        """
        self.N = grid_size
        self.kappa = kappa
        self.eta = compute_lattice_correction(grid_size)
        self.G_eff = self.eta * kappa / (4 * np.pi)
        self.verbose = verbose
        self.center = grid_size // 2

        # Default to solar system units for lensing tests
        self.units = units or solar_system_units(grid_size)

    def setup_point_mass(self, mass: float, width: float = 2.0,
                         settling_steps: int = 100) -> DETCollider3D:
        """
        Set up simulation with central point mass.

        Parameters
        ----------
        mass : float
            Central mass in lattice units
        width : float
            Gaussian width of mass distribution
        settling_steps : int
            Steps to establish gravitational field

        Returns
        -------
        DETCollider3D
            Simulation with established potential field
        """
        params = DETParams3D(
            N=self.N,
            DT=0.01,
            F_VAC=0.001,
            F_MIN=0.0,

            # Gravity
            gravity_enabled=True,
            kappa_grav=self.kappa,
            alpha_grav=0.01,
            eta_lattice=self.eta,

            # Structure for gravity sourcing
            q_enabled=True,
            alpha_q=0.01,

            # Disable interfering dynamics
            momentum_enabled=False,
            angular_momentum_enabled=False,
            floor_enabled=False,
            boundary_enabled=False,
            agency_dynamic=False,
            sigma_dynamic=False,
            coherence_dynamic=False
        )

        sim = DETCollider3D(params)

        # Add central mass
        sim.add_packet(
            (self.center, self.center, self.center),
            mass=mass,
            width=width,
            momentum=(0, 0, 0),
            initial_q=0.9
        )

        # Let field establish
        if self.verbose:
            print(f"Establishing gravitational field ({settling_steps} steps)...")
        for _ in range(settling_steps):
            sim.step()

        return sim

    def measure_gravitational_mass(self, sim: DETCollider3D) -> float:
        """Measure effective gravitational mass (sum of q - b)."""
        rho = sim.q - sim.b
        return np.sum(rho)

    def theoretical_deflection(self, impact_parameter: float,
                                mass_grav: float) -> float:
        """
        Compute theoretical Schwarzschild deflection angle.

        In weak field: α = 4GM/(c²b)

        In lattice units where c=1: α = 4*G_eff*M/b

        Parameters
        ----------
        impact_parameter : float
            Closest approach distance (lattice units)
        mass_grav : float
            Gravitational mass (lattice units)

        Returns
        -------
        float
            Deflection angle in radians
        """
        if impact_parameter <= 0:
            return 0.0

        # α = 4GM/(c²b) with c=1 in lattice units
        return 4 * self.G_eff * mass_grav / impact_parameter

    def compute_einstein_radius(self, mass_grav: float,
                                 d_lens: float = None,
                                 d_source: float = None) -> float:
        """
        Compute Einstein radius.

        For a point mass:
            θ_E = sqrt(4GM/c² × D_LS/(D_L × D_S))

        In lattice units with typical geometry (D_LS ≈ D_S >> D_L):
            R_E ≈ sqrt(4*G_eff*M*D)

        Parameters
        ----------
        mass_grav : float
            Gravitational mass (lattice units)
        d_lens : float, optional
            Distance to lens
        d_source : float, optional
            Distance to source

        Returns
        -------
        float
            Einstein radius in lattice units
        """
        # For a simple case: observer-lens-source geometry
        # R_E = sqrt(4*G_eff*M*D) where D is characteristic distance
        if d_lens is None:
            d_lens = self.N / 4  # Reasonable default

        return np.sqrt(4 * self.G_eff * mass_grav * d_lens)

    def analyze_lensing(self, mass: float = 50.0, width: float = 2.0,
                        impact_range: Tuple[float, float] = (3.0, 20.0),
                        n_samples: int = 10) -> LensingResult:
        """
        Perform complete lensing analysis.

        Parameters
        ----------
        mass : float
            Central mass in lattice units
        width : float
            Mass distribution width
        impact_range : Tuple[float, float]
            Range of impact parameters to test
        n_samples : int
            Number of impact parameters to sample

        Returns
        -------
        LensingResult
            Complete lensing analysis results
        """
        if self.verbose:
            print("\n" + "="*60)
            print("DET GRAVITATIONAL LENSING ANALYSIS")
            print("="*60)
            print(f"Grid size: {self.N}")
            print(f"Kappa: {self.kappa}, Eta: {self.eta:.4f}")
            print(f"G_eff: {self.G_eff:.6f}")

        # Setup simulation
        sim = self.setup_point_mass(mass, width)

        # Measure actual gravitational mass
        M_grav = self.measure_gravitational_mass(sim)
        if self.verbose:
            print(f"Gravitational mass (q-b): {M_grav:.4f}")

        # Create ray tracer
        tracer = GravitationalRayTracer(sim, verbose=False)

        # Sample impact parameters
        b_values = np.linspace(impact_range[0], impact_range[1], n_samples)
        deflections = np.zeros(n_samples)
        theory_deflections = np.zeros(n_samples)

        if self.verbose:
            print(f"\nTracing rays at {n_samples} impact parameters...")
            print("-" * 50)
            print(f"{'b (cells)':<12} {'α_DET (rad)':<14} {'α_theory (rad)':<14} {'Error %':<10}")
            print("-" * 50)

        for i, b in enumerate(b_values):
            # Trace ray
            alpha, path = tracer.compute_deflection(b)
            deflections[i] = alpha

            # Theoretical prediction
            alpha_theory = self.theoretical_deflection(b, M_grav)
            theory_deflections[i] = alpha_theory

            # Relative error
            rel_err = abs(alpha - alpha_theory) / alpha_theory if alpha_theory > 0 else 0

            if self.verbose:
                print(f"{b:<12.2f} {alpha:<14.6f} {alpha_theory:<14.6f} {rel_err*100:<10.1f}")

        # Compute Einstein radius
        R_E = self.compute_einstein_radius(M_grav)

        # Convert to SI units
        M_si = self.units.mass_to_si(M_grav)
        R_E_si = self.units.length_to_si(R_E)

        # Deflection angles in SI (already in radians, dimensionless)
        deflections_si = deflections  # radians are dimensionless

        # Relative errors
        rel_errors = np.abs(deflections - theory_deflections) / np.where(
            theory_deflections > 0, theory_deflections, 1e-10
        )

        if self.verbose:
            print("-" * 50)
            print(f"\nEinstein radius: {R_E:.2f} cells = {R_E_si:.2e} m")
            print(f"Mean relative error: {np.mean(rel_errors)*100:.1f}%")

            # Summary
            print("\n" + "="*60)
            if np.mean(rel_errors) < 0.3:
                print("LENSING TEST PASSED: DET deflections match Schwarzschild prediction")
            else:
                print("LENSING TEST: Significant deviation from point-mass prediction")
            print("="*60)

        return LensingResult(
            impact_parameters=b_values,
            deflection_angles=deflections,
            deflection_angles_si=deflections_si,
            einstein_radius=R_E,
            einstein_radius_si=R_E_si,
            mass_lattice=M_grav,
            mass_si=M_si,
            theoretical_deflections=theory_deflections,
            relative_errors=rel_errors,
            parameters={
                'grid_size': self.N,
                'kappa': self.kappa,
                'eta': self.eta,
                'G_eff': self.G_eff,
                'mass_input': mass,
                'width': width
            }
        )

    def deflection_profile(self, result: LensingResult) -> Dict:
        """
        Analyze deflection profile shape.

        For point mass: α ∝ 1/b

        Returns fit parameters and quality metrics.
        """
        from scipy.optimize import curve_fit

        b = result.impact_parameters
        alpha = result.deflection_angles

        # Fit to α = A/b
        def inverse_law(b, A):
            return A / b

        try:
            popt, pcov = curve_fit(inverse_law, b, alpha, p0=[alpha[0] * b[0]])
            A_fit = popt[0]
            A_err = np.sqrt(pcov[0, 0])

            # Predicted from theory: A = 4*G_eff*M
            A_theory = 4 * self.G_eff * result.mass_lattice

            # Compute R²
            alpha_pred = inverse_law(b, A_fit)
            ss_res = np.sum((alpha - alpha_pred)**2)
            ss_tot = np.sum((alpha - np.mean(alpha))**2)
            r_squared = 1 - ss_res / ss_tot

            return {
                'A_fit': A_fit,
                'A_err': A_err,
                'A_theory': A_theory,
                'relative_error': abs(A_fit - A_theory) / A_theory,
                'r_squared': r_squared,
                'follows_1_over_b': r_squared > 0.95
            }
        except Exception:
            return {
                'A_fit': 0,
                'A_err': float('inf'),
                'A_theory': 4 * self.G_eff * result.mass_lattice,
                'relative_error': 1.0,
                'r_squared': 0,
                'follows_1_over_b': False
            }


# ==============================================================================
# EXTENDED MASS LENSING
# ==============================================================================

class ExtendedMassLensing(GravitationalLensing):
    """
    Gravitational lensing for extended mass distributions.

    Handles galaxy-scale lensing with disk/halo mass profiles.
    """

    def setup_galaxy(self, stellar_mass: float, disk_scale: float,
                     halo_mass: float = 0.0, settling_steps: int = 100) -> DETCollider3D:
        """
        Set up simulation with galaxy-like mass distribution.

        Creates exponential disk with optional NFW-like halo.
        """
        params = DETParams3D(
            N=self.N,
            DT=0.01,
            F_VAC=0.001,
            F_MIN=0.0,
            gravity_enabled=True,
            kappa_grav=self.kappa,
            eta_lattice=self.eta,
            q_enabled=True,
            alpha_q=0.01,
            momentum_enabled=False,
            angular_momentum_enabled=False,
            floor_enabled=False,
            boundary_enabled=False
        )

        sim = DETCollider3D(params)

        # Create exponential disk profile
        center = self.center
        z, y, x = np.mgrid[0:self.N, 0:self.N, 0:self.N]

        # Radial distance in disk plane
        dx = (x - center + self.N/2) % self.N - self.N/2
        dy = (y - center + self.N/2) % self.N - self.N/2
        dz = (z - center + self.N/2) % self.N - self.N/2

        r_disk = np.sqrt(dx**2 + dy**2)

        # Exponential disk: Σ(r) ∝ exp(-r/R_d)
        disk_profile = np.exp(-r_disk / disk_scale) * np.exp(-np.abs(dz) / 1.0)

        # Normalize and add
        disk_profile = disk_profile / np.sum(disk_profile) * stellar_mass
        sim.F += disk_profile
        sim.q += 0.5 * disk_profile / np.max(disk_profile)  # Structure proportional to mass

        # Add halo if requested (spherical)
        if halo_mass > 0:
            r_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            r_s = disk_scale * 3  # Scale radius
            # NFW-like profile
            halo_profile = 1.0 / (r_3d / r_s + 0.1) / (1 + r_3d / r_s)**2
            halo_profile = halo_profile / np.sum(halo_profile) * halo_mass
            sim.F += halo_profile
            sim.q += 0.3 * halo_profile / np.max(halo_profile)

        # Settle
        for _ in range(settling_steps):
            sim.step()

        return sim


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_lensing_analysis(grid_size: int = 64, mass: float = 50.0,
                         verbose: bool = True) -> LensingResult:
    """
    Run complete gravitational lensing analysis.

    This is the main entry point for the v6.4 Gravitational Lensing feature.

    Parameters
    ----------
    grid_size : int
        Simulation grid size
    mass : float
        Central mass in lattice units
    verbose : bool
        Print progress

    Returns
    -------
    LensingResult
        Complete lensing analysis results
    """
    lensing = GravitationalLensing(grid_size=grid_size, verbose=verbose)
    result = lensing.analyze_lensing(mass=mass)

    # Analyze deflection profile
    profile = lensing.deflection_profile(result)

    if verbose:
        print("\nDeflection Profile Analysis:")
        print(f"  Fit: α = {profile['A_fit']:.4f}/b")
        print(f"  Theory: α = {profile['A_theory']:.4f}/b")
        print(f"  R² = {profile['r_squared']:.4f}")
        print(f"  Follows 1/b law: {profile['follows_1_over_b']}")

    return result


if __name__ == "__main__":
    print("DET v6.4 Gravitational Lensing")
    print("Ray-tracing through Φ field...")
    print()

    result = run_lensing_analysis(grid_size=64, mass=50.0, verbose=True)
