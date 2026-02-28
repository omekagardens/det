"""
DET SI Unit Conversion Layer
============================

DET operates in dimensionless "lattice units" where:
- Length: 1 cell
- Time: 1 step (= DT in simulation time)
- Mass: 1 unit of F (resource)

This module provides bidirectional conversion to SI units.

Fundamental Insight
-------------------
DET has two built-in physical constraints:
1. Locality bound: c_DET = 1 cell/step (maximum information speed)
2. Gravity law: G_eff = ηκ/(4π) (effective gravitational constant)

Matching to physical constants c and G requires:
- c = a / τ₀  →  τ₀ = a / c
- G = G_eff × a³/(m₀τ₀²)  →  m₀ = G_eff × a × c² / G

Therefore: Choosing ONE scale (length a) determines ALL conversions.

Usage
-----
    from det_si_units import DETUnitSystem, SOLAR_SYSTEM, LABORATORY

    # Solar system scale
    units = SOLAR_SYSTEM
    print(f"1 cell = {units.a:.2e} m")
    print(f"1 step = {units.tau:.2e} s")
    print(f"1 F = {units.m0:.2e} kg")

    # Convert simulation results
    orbital_period_steps = 1000
    orbital_period_seconds = units.time_to_si(orbital_period_steps)

Reference: DET Theory Card v6.3, Appendix C
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ==============================================================================
# PHYSICAL CONSTANTS (CODATA 2018)
# ==============================================================================

C_SI = 299_792_458.0  # Speed of light [m/s]
G_SI = 6.67430e-11    # Gravitational constant [m³/(kg·s²)]
HBAR_SI = 1.054571817e-34  # Reduced Planck constant [J·s]
K_B_SI = 1.380649e-23      # Boltzmann constant [J/K]

# Derived
PLANCK_LENGTH = np.sqrt(HBAR_SI * G_SI / C_SI**3)  # ~1.616e-35 m
PLANCK_TIME = np.sqrt(HBAR_SI * G_SI / C_SI**5)    # ~5.391e-44 s
PLANCK_MASS = np.sqrt(HBAR_SI * C_SI / G_SI)       # ~2.176e-8 kg

# Astronomical
AU = 1.495978707e11   # Astronomical unit [m]
PC = 3.0857e16        # Parsec [m]
LY = 9.4607e15        # Light year [m]
M_SUN = 1.98892e30    # Solar mass [kg]
M_EARTH = 5.972e24    # Earth mass [kg]
YEAR = 3.15576e7      # Julian year [s]


# ==============================================================================
# DET LATTICE PARAMETERS (from unified schema)
# ==============================================================================

# Default lattice correction factor (N=64)
ETA_DEFAULT = 0.968

# Default Poisson coupling
KAPPA_DEFAULT = 5.0

# Effective G in lattice units: G_eff = η*κ/(4π)
def G_eff(kappa: float = KAPPA_DEFAULT, eta: float = ETA_DEFAULT) -> float:
    """Compute effective gravitational constant in DET lattice units."""
    return eta * kappa / (4 * np.pi)


# ==============================================================================
# UNIT SYSTEM CLASS
# ==============================================================================

@dataclass
class DETUnitSystem:
    """
    Complete SI unit conversion for DET simulations.

    Choose a length scale 'a' (meters per cell) and all other scales
    are determined by matching c and G.

    Parameters
    ----------
    a : float
        Lattice spacing in meters (length scale)
    kappa : float
        DET Poisson coupling (default: 5.0)
    eta : float
        Lattice correction factor (default: 0.968 for N=64)
    name : str
        Descriptive name for this unit system

    Derived Quantities
    ------------------
    tau : float
        Time per step in seconds (from c = a/τ)
    m0 : float
        Mass per F unit in kg (from G constraint)
    """

    a: float  # Length scale: meters per cell
    kappa: float = KAPPA_DEFAULT
    eta: float = ETA_DEFAULT
    name: str = "Custom"

    # Derived (computed post-init)
    tau: float = field(init=False)   # Time scale: seconds per step
    m0: float = field(init=False)    # Mass scale: kg per F unit
    G_lattice: float = field(init=False)  # G_eff in lattice units

    def __post_init__(self):
        """Compute derived scales from a and physical constants."""
        # G_eff in lattice units
        self.G_lattice = G_eff(self.kappa, self.eta)

        # Time scale from speed of light: c = a/τ → τ = a/c
        self.tau = self.a / C_SI

        # Mass scale from gravity: G = G_eff * a³/(m₀τ²)
        # → m₀ = G_eff * a³ / (G * τ²) = G_eff * a³ * c² / (G * a²) = G_eff * a * c² / G
        self.m0 = self.G_lattice * self.a * C_SI**2 / G_SI

    # ==========================================================================
    # BASIC CONVERSIONS: DET → SI
    # ==========================================================================

    def length_to_si(self, x_det: float) -> float:
        """Convert length from cells to meters."""
        return x_det * self.a

    def time_to_si(self, t_det: float) -> float:
        """Convert time from steps to seconds."""
        return t_det * self.tau

    def mass_to_si(self, m_det: float) -> float:
        """Convert mass from F units to kg."""
        return m_det * self.m0

    def velocity_to_si(self, v_det: float) -> float:
        """Convert velocity from cells/step to m/s."""
        return v_det * self.a / self.tau  # = v_det * c

    def acceleration_to_si(self, acc_det: float) -> float:
        """Convert acceleration from cells/step² to m/s²."""
        return acc_det * self.a / self.tau**2

    def force_to_si(self, F_det: float) -> float:
        """Convert force from lattice units to Newtons."""
        return F_det * self.m0 * self.a / self.tau**2

    def energy_to_si(self, E_det: float) -> float:
        """Convert energy from lattice units to Joules."""
        return E_det * self.m0 * self.a**2 / self.tau**2

    def momentum_to_si(self, p_det: float) -> float:
        """Convert momentum from lattice units to kg·m/s."""
        return p_det * self.m0 * self.a / self.tau

    def angular_momentum_to_si(self, L_det: float) -> float:
        """Convert angular momentum from lattice units to kg·m²/s."""
        return L_det * self.m0 * self.a**2 / self.tau

    def density_to_si(self, rho_det: float) -> float:
        """Convert density from F/cell³ to kg/m³."""
        return rho_det * self.m0 / self.a**3

    def potential_to_si(self, phi_det: float) -> float:
        """Convert gravitational potential to J/kg (= m²/s²)."""
        return phi_det * self.a**2 / self.tau**2

    # ==========================================================================
    # BASIC CONVERSIONS: SI → DET
    # ==========================================================================

    def length_to_det(self, x_si: float) -> float:
        """Convert length from meters to cells."""
        return x_si / self.a

    def time_to_det(self, t_si: float) -> float:
        """Convert time from seconds to steps."""
        return t_si / self.tau

    def mass_to_det(self, m_si: float) -> float:
        """Convert mass from kg to F units."""
        return m_si / self.m0

    def velocity_to_det(self, v_si: float) -> float:
        """Convert velocity from m/s to cells/step (units of c)."""
        return v_si * self.tau / self.a  # = v_si / c

    # ==========================================================================
    # CONVENIENCE: Astronomical Units
    # ==========================================================================

    def time_to_years(self, t_det: float) -> float:
        """Convert time from steps to years."""
        return self.time_to_si(t_det) / YEAR

    def length_to_au(self, x_det: float) -> float:
        """Convert length from cells to AU."""
        return self.length_to_si(x_det) / AU

    def length_to_pc(self, x_det: float) -> float:
        """Convert length from cells to parsecs."""
        return self.length_to_si(x_det) / PC

    def mass_to_solar(self, m_det: float) -> float:
        """Convert mass from F units to solar masses."""
        return self.mass_to_si(m_det) / M_SUN

    # ==========================================================================
    # QUANTUM SCALE
    # ==========================================================================

    @property
    def h_bar_lattice(self) -> float:
        """ℏ in lattice units: [m₀ a² / τ]."""
        return HBAR_SI * self.tau / (self.m0 * self.a**2)

    @property
    def coherence_quantum(self) -> float:
        """Characteristic quantum of action in lattice units.

        This may relate to coherence thresholds in DET.
        """
        return self.h_bar_lattice

    # ==========================================================================
    # DIAGNOSTICS
    # ==========================================================================

    def summary(self) -> str:
        """Return formatted summary of unit system."""
        lines = [
            f"DET Unit System: {self.name}",
            "=" * 50,
            "",
            "Fundamental Scales:",
            f"  Length scale a    = {self.a:.4e} m/cell",
            f"  Time scale τ      = {self.tau:.4e} s/step",
            f"  Mass scale m₀     = {self.m0:.4e} kg/F",
            "",
            "Derived Quantities:",
            f"  Velocity (1 c/s)  = {C_SI:.4e} m/s",
            f"  G_lattice         = {self.G_lattice:.6f}",
            f"  ℏ_lattice         = {self.h_bar_lattice:.4e}",
            "",
            "Verification:",
            f"  c_check = a/τ     = {self.a/self.tau:.4e} m/s (should be {C_SI:.4e})",
            f"  G_check           = {self.G_lattice * self.a**3 / (self.m0 * self.tau**2):.4e} m³/kg/s² (should be {G_SI:.4e})",
            "",
            "Physical Scales:",
        ]

        # Add scale-specific info
        if self.a > 1e15:  # Galactic scale
            lines.extend([
                f"  1 cell = {self.a/PC:.2f} pc",
                f"  1 step = {self.tau/YEAR:.2e} yr",
                f"  1 F    = {self.m0/M_SUN:.2e} M☉",
            ])
        elif self.a > 1e9:  # Solar system scale
            lines.extend([
                f"  1 cell = {self.a/AU:.4f} AU",
                f"  1 step = {self.tau/YEAR:.6f} yr = {self.tau/86400:.2f} days",
                f"  1 F    = {self.m0/M_SUN:.4f} M☉",
            ])
        elif self.a > 1e3:  # Planetary scale
            lines.extend([
                f"  1 cell = {self.a/1e3:.2f} km",
                f"  1 step = {self.tau:.4f} s",
                f"  1 F    = {self.m0/M_EARTH:.4e} M⊕",
            ])
        else:  # Laboratory scale
            lines.extend([
                f"  1 cell = {self.a:.4e} m",
                f"  1 step = {self.tau:.4e} s",
                f"  1 F    = {self.m0:.4e} kg",
            ])

        return "\n".join(lines)

    def __repr__(self):
        return f"DETUnitSystem(a={self.a:.2e} m, τ={self.tau:.2e} s, m₀={self.m0:.2e} kg, name='{self.name}')"


# ==============================================================================
# PRE-DEFINED UNIT SYSTEMS
# ==============================================================================

def solar_system_units(N: int = 64) -> DETUnitSystem:
    """
    Unit system for solar system simulations.

    Sets 1 cell = 0.5 AU, suitable for inner solar system.

    Parameters
    ----------
    N : int
        Grid size (affects η correction)

    Returns
    -------
    DETUnitSystem
        Configured for solar system scale
    """
    a = 0.5 * AU  # Half AU per cell
    eta = {32: 0.901, 64: 0.968, 96: 0.975, 128: 0.981}.get(N, 0.968)
    return DETUnitSystem(a=a, eta=eta, name="Solar System (0.5 AU/cell)")


def galactic_units(N: int = 64) -> DETUnitSystem:
    """
    Unit system for galactic simulations.

    Sets 1 cell = 1 kpc, suitable for galaxy rotation curves.
    """
    a = 1e3 * PC  # 1 kpc per cell
    eta = {32: 0.901, 64: 0.968, 96: 0.975, 128: 0.981}.get(N, 0.968)
    return DETUnitSystem(a=a, eta=eta, name="Galactic (1 kpc/cell)")


def laboratory_units() -> DETUnitSystem:
    """
    Unit system for laboratory-scale simulations.

    Sets 1 cell = 1 meter.
    """
    return DETUnitSystem(a=1.0, name="Laboratory (1 m/cell)")


def planck_units() -> DETUnitSystem:
    """
    Unit system with Planck-scale cells.

    At this scale, quantum effects should dominate.
    """
    return DETUnitSystem(a=PLANCK_LENGTH, name="Planck Scale")


def custom_units(a_meters: float, name: str = "Custom") -> DETUnitSystem:
    """Create a custom unit system with specified length scale."""
    return DETUnitSystem(a=a_meters, name=name)


# Pre-instantiated systems
SOLAR_SYSTEM = solar_system_units()
GALACTIC = galactic_units()
LABORATORY = laboratory_units()
PLANCK = planck_units()


# ==============================================================================
# SIMULATION HELPERS
# ==============================================================================

def convert_orbit_to_si(units: DETUnitSystem,
                         radius_cells: float,
                         period_steps: float) -> dict:
    """
    Convert orbital parameters from DET to SI units.

    Parameters
    ----------
    units : DETUnitSystem
        The unit system to use
    radius_cells : float
        Orbital radius in cells
    period_steps : float
        Orbital period in steps

    Returns
    -------
    dict
        Contains radius_m, radius_au, period_s, period_years, velocity_ms
    """
    r_m = units.length_to_si(radius_cells)
    T_s = units.time_to_si(period_steps)

    # Orbital velocity
    v = 2 * np.pi * r_m / T_s

    return {
        'radius_m': r_m,
        'radius_au': r_m / AU,
        'period_s': T_s,
        'period_years': T_s / YEAR,
        'velocity_ms': v,
        'velocity_kms': v / 1000,
    }


def verify_kepler(units: DETUnitSystem,
                  radius_cells: float,
                  period_steps: float,
                  central_mass_F: float) -> dict:
    """
    Verify Kepler's Third Law for an orbit.

    For a circular orbit: T² = (4π²/GM) r³

    Returns measured and theoretical values.
    """
    r_m = units.length_to_si(radius_cells)
    T_s = units.time_to_si(period_steps)
    M_kg = units.mass_to_si(central_mass_F)

    # Measured: T²/r³
    kepler_measured = T_s**2 / r_m**3

    # Theoretical: 4π²/(GM)
    kepler_theory = 4 * np.pi**2 / (G_SI * M_kg)

    error = abs(kepler_measured - kepler_theory) / kepler_theory

    return {
        'T_squared_over_r_cubed': kepler_measured,
        'kepler_constant_theory': kepler_theory,
        'relative_error': error,
        'passes': error < 0.05,  # 5% tolerance
    }


# ==============================================================================
# DIMENSIONAL ANALYSIS
# ==============================================================================

@dataclass
class DETQuantity:
    """
    A quantity with explicit dimensional tracking.

    Dimensions are [L^a M^b T^c] for length, mass, time.
    """
    value: float
    L: int = 0  # Length exponent
    M: int = 0  # Mass exponent
    T: int = 0  # Time exponent

    def to_si(self, units: DETUnitSystem) -> float:
        """Convert to SI units."""
        return self.value * (units.a ** self.L) * (units.m0 ** self.M) * (units.tau ** self.T)

    def __mul__(self, other):
        if isinstance(other, DETQuantity):
            return DETQuantity(
                self.value * other.value,
                self.L + other.L,
                self.M + other.M,
                self.T + other.T
            )
        return DETQuantity(self.value * other, self.L, self.M, self.T)

    def __truediv__(self, other):
        if isinstance(other, DETQuantity):
            return DETQuantity(
                self.value / other.value,
                self.L - other.L,
                self.M - other.M,
                self.T - other.T
            )
        return DETQuantity(self.value / other, self.L, self.M, self.T)

    def __pow__(self, n):
        return DETQuantity(self.value ** n, self.L * n, self.M * n, self.T * n)

    @property
    def dimensions(self) -> str:
        """Human-readable dimensions."""
        parts = []
        if self.L: parts.append(f"L^{self.L}" if self.L != 1 else "L")
        if self.M: parts.append(f"M^{self.M}" if self.M != 1 else "M")
        if self.T: parts.append(f"T^{self.T}" if self.T != 1 else "T")
        return " ".join(parts) if parts else "dimensionless"


# ==============================================================================
# TEST / DEMO
# ==============================================================================

if __name__ == "__main__":
    print("DET SI UNIT CONVERSION LAYER")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("SOLAR SYSTEM SCALE")
    print("=" * 60)
    print(SOLAR_SYSTEM.summary())

    print("\n" + "=" * 60)
    print("GALACTIC SCALE")
    print("=" * 60)
    print(GALACTIC.summary())

    print("\n" + "=" * 60)
    print("LABORATORY SCALE")
    print("=" * 60)
    print(LABORATORY.summary())

    print("\n" + "=" * 60)
    print("PLANCK SCALE")
    print("=" * 60)
    print(PLANCK.summary())

    # Example orbital calculation
    print("\n" + "=" * 60)
    print("EXAMPLE: Earth's Orbit")
    print("=" * 60)

    # Earth orbits at ~1 AU with period ~1 year
    # In solar system units (0.5 AU/cell), r = 2 cells
    orbit = convert_orbit_to_si(SOLAR_SYSTEM, radius_cells=2.0, period_steps=1000)
    print(f"Radius: {orbit['radius_au']:.2f} AU")
    print(f"Period: {orbit['period_years']:.4f} years")
    print(f"Velocity: {orbit['velocity_kms']:.2f} km/s")

    # Verify Kepler
    print("\n" + "=" * 60)
    print("KEPLER VERIFICATION")
    print("=" * 60)

    # What mass would give T = 1000 steps for r = 2 cells?
    # In lattice units: T² = (4π²/G_eff) r³ / M
    G_l = SOLAR_SYSTEM.G_lattice
    r_l = 2.0
    T_l = 1000.0
    M_l = 4 * np.pi**2 * r_l**3 / (G_l * T_l**2)
    print(f"Required central mass: {M_l:.2f} F units = {SOLAR_SYSTEM.mass_to_si(M_l)/M_SUN:.4f} M☉")

    kepler = verify_kepler(SOLAR_SYSTEM, radius_cells=2.0, period_steps=1000, central_mass_F=M_l)
    print(f"Kepler constant (measured): {kepler['T_squared_over_r_cubed']:.4e} s²/m³")
    print(f"Kepler constant (theory):   {kepler['kepler_constant_theory']:.4e} s²/m³")
    print(f"Relative error: {kepler['relative_error']*100:.2f}%")
    print(f"Verification: {'PASS' if kepler['passes'] else 'FAIL'}")
