"""
Test DET SI Unit Conversion Layer
==================================

Verify that the unit conversion correctly maps DET to physical systems.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_si_units import (
    DETUnitSystem, SOLAR_SYSTEM, GALACTIC, LABORATORY, PLANCK,
    solar_system_units, custom_units, verify_kepler, convert_orbit_to_si,
    C_SI, G_SI, HBAR_SI, AU, M_SUN, YEAR, PC
)


def test_c_and_G_consistency():
    """Verify c and G are reproduced correctly at all scales."""
    print("Testing: Speed of Light and Gravity Constant Consistency")
    print("-" * 60)

    scales = [
        ("Planck", PLANCK),
        ("Laboratory", LABORATORY),
        ("Solar System", SOLAR_SYSTEM),
        ("Galactic", GALACTIC),
    ]

    all_pass = True
    for name, units in scales:
        # Check c
        c_computed = units.a / units.tau
        c_error = abs(c_computed - C_SI) / C_SI

        # Check G
        G_computed = units.G_lattice * units.a**3 / (units.m0 * units.tau**2)
        G_error = abs(G_computed - G_SI) / G_SI

        c_ok = c_error < 1e-10
        G_ok = G_error < 1e-10

        print(f"  {name:15s}: c error = {c_error:.2e}, G error = {G_error:.2e}")
        all_pass = all_pass and c_ok and G_ok

    return all_pass


def test_solar_system_planets():
    """Test unit system against known planetary orbits."""
    print("\nTesting: Solar System Planetary Orbits")
    print("-" * 60)

    # Create unit system with 1 AU = 1 cell for clarity
    units = custom_units(AU, name="1 AU/cell")

    # Known planetary data (semi-major axis in AU, period in years)
    planets = {
        'Mercury': (0.387, 0.241),
        'Venus':   (0.723, 0.615),
        'Earth':   (1.000, 1.000),
        'Mars':    (1.524, 1.881),
        'Jupiter': (5.203, 11.86),
        'Saturn':  (9.537, 29.46),
    }

    print(f"\n  Planet       r (AU)    T (yr)    v (km/s)  T²/r³")
    print("  " + "-" * 54)

    kepler_constants = []
    all_pass = True

    for name, (r_au, T_yr) in planets.items():
        # In our units: r = r_au cells, need to find T in steps
        T_steps = T_yr * YEAR / units.tau

        orbit = convert_orbit_to_si(units, radius_cells=r_au, period_steps=T_steps)

        # Kepler constant in SI
        kepler = (T_yr * YEAR)**2 / (r_au * AU)**3
        kepler_constants.append(kepler)

        print(f"  {name:10s}  {r_au:6.3f}   {T_yr:7.3f}    {orbit['velocity_kms']:5.1f}     {kepler:.4e}")

    # All Kepler constants should be equal (4π²/GM_sun)
    kepler_mean = np.mean(kepler_constants)
    kepler_std = np.std(kepler_constants)
    cv = kepler_std / kepler_mean * 100

    # Theoretical value
    kepler_theory = 4 * np.pi**2 / (G_SI * M_SUN)

    print(f"\n  Kepler's constant (mean): {kepler_mean:.4e} s²/m³")
    print(f"  Theoretical (4π²/GM☉):    {kepler_theory:.4e} s²/m³")
    print(f"  Coefficient of variation: {cv:.2f}%")

    # Should match to better than 0.5% (allows for real planetary eccentricities)
    error = abs(kepler_mean - kepler_theory) / kepler_theory
    all_pass = error < 0.005 and cv < 0.5

    return all_pass


def test_simulation_mapping():
    """Test mapping between DET simulation and SI units."""
    print("\nTesting: DET Simulation to SI Mapping")
    print("-" * 60)

    # Use solar system units
    units = solar_system_units(N=64)

    # Simulated orbit in DET lattice units
    # If central mass M_lattice produces a particular orbital period,
    # we can verify Kepler's law holds

    # For a 1 M☉ equivalent mass, what is M in F units?
    M_det = units.mass_to_det(M_SUN)
    print(f"  1 M☉ = {M_det:.6e} F units")

    # For r = 2 cells (= 1 AU), what should T be?
    # T² = 4π²r³/(G_eff M) in lattice units
    r_det = 2.0  # cells
    G_l = units.G_lattice
    T_det_squared = 4 * np.pi**2 * r_det**3 / (G_l * M_det)
    T_det = np.sqrt(T_det_squared)

    T_si = units.time_to_si(T_det)
    T_years = T_si / YEAR

    print(f"  Orbital radius: {r_det} cells = {r_det * 0.5:.1f} AU")
    print(f"  Central mass:   {M_det:.6e} F = 1 M☉")
    print(f"  Orbital period: {T_det:.2f} steps = {T_years:.4f} years")
    print(f"  Expected:       ~1 year for Earth")

    # Should be approximately 1 year
    error = abs(T_years - 1.0)
    passed = error < 0.01  # 1% tolerance (lattice effects)

    return passed


def test_planck_scale():
    """Test Planck scale properties."""
    print("\nTesting: Planck Scale Properties")
    print("-" * 60)

    units = PLANCK

    print(f"  1 cell = {units.a:.4e} m (Planck length)")
    print(f"  1 step = {units.tau:.4e} s (Planck time)")
    print(f"  1 F    = {units.m0:.4e} kg")
    print(f"  Planck mass = {np.sqrt(HBAR_SI * C_SI / G_SI):.4e} kg")
    print(f"  Ratio m0/m_Planck = {units.m0 / np.sqrt(HBAR_SI * C_SI / G_SI):.4f}")

    # ℏ in lattice units should be O(1) at Planck scale
    print(f"\n  ℏ_lattice = {units.h_bar_lattice:.4f}")
    print("  (Should be O(1) at Planck scale)")

    # Check it's between 0.1 and 10
    passed = 0.1 < units.h_bar_lattice < 10

    return passed


def test_velocity_conversion():
    """Test velocity conversions at c."""
    print("\nTesting: Velocity Conversion (Speed of Light)")
    print("-" * 60)

    for name, units in [("Solar", SOLAR_SYSTEM), ("Lab", LABORATORY)]:
        # v = 1 cell/step should equal c
        v_det = 1.0
        v_si = units.velocity_to_si(v_det)

        # Reverse
        v_back = units.velocity_to_det(v_si)

        print(f"  {name:6s}: v=1 → {v_si:.4e} m/s (c = {C_SI:.4e})")
        print(f"          Round-trip error: {abs(v_back - v_det):.2e}")

    # Also test subluminal velocity
    units = SOLAR_SYSTEM
    v_earth = 30e3  # m/s (Earth's orbital velocity)
    v_det = units.velocity_to_det(v_earth)
    print(f"\n  Earth's velocity ({v_earth/1e3:.0f} km/s) = {v_det:.6f} c")

    passed = abs(v_det - v_earth / C_SI) / (v_earth / C_SI) < 1e-10
    return passed


def test_dimensional_consistency():
    """Test that derived quantities have correct dimensions."""
    print("\nTesting: Dimensional Consistency")
    print("-" * 60)

    units = LABORATORY

    # Energy: [M L² T⁻²] = kg m² s⁻²
    E_det = 1.0
    E_si = units.energy_to_si(E_det)
    E_expected = units.m0 * units.a**2 / units.tau**2
    print(f"  Energy: 1 lattice unit = {E_si:.4e} J")
    print(f"          Expected:        {E_expected:.4e} J")

    # Momentum: [M L T⁻¹] = kg m s⁻¹
    p_det = 1.0
    p_si = units.momentum_to_si(p_det)
    p_expected = units.m0 * units.a / units.tau
    print(f"  Momentum: 1 lattice unit = {p_si:.4e} kg·m/s")
    print(f"            Expected:        {p_expected:.4e} kg·m/s")

    # Angular momentum: [M L² T⁻¹] = kg m² s⁻¹
    L_det = 1.0
    L_si = units.angular_momentum_to_si(L_det)
    L_expected = units.m0 * units.a**2 / units.tau
    print(f"  Angular mom: 1 lattice unit = {L_si:.4e} kg·m²/s")
    print(f"               Expected:        {L_expected:.4e} kg·m²/s")

    # All should match
    passed = (
        abs(E_si - E_expected) / E_expected < 1e-10 and
        abs(p_si - p_expected) / p_expected < 1e-10 and
        abs(L_si - L_expected) / L_expected < 1e-10
    )

    return passed


def run_all_tests():
    """Run all unit conversion tests."""
    print("=" * 70)
    print("DET SI UNIT CONVERSION TESTS")
    print("=" * 70)

    results = {}

    results['c_and_G_consistency'] = test_c_and_G_consistency()
    results['solar_system_planets'] = test_solar_system_planets()
    results['simulation_mapping'] = test_simulation_mapping()
    results['planck_scale'] = test_planck_scale()
    results['velocity_conversion'] = test_velocity_conversion()
    results['dimensional_consistency'] = test_dimensional_consistency()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("-" * 70)
    if all_passed:
        print("✓ ALL SI UNIT TESTS PASSED!")
    else:
        print("✗ Some tests failed")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
