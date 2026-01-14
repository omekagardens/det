#!/usr/bin/env python3
"""
Hafele-Keating Experiment Validation
=====================================

Test DET predictions against the Hafele-Keating experiment (1971).

The experiment:
- Flew cesium atomic clocks on commercial aircraft around the world
- Eastward flight: clocks lost time (faster relative to Earth's surface)
- Westward flight: clocks gained time (slower relative to Earth's surface)
- Compared to reference clocks at US Naval Observatory

This tests BOTH gravitational and kinematic time dilation.

Published results (Science, 1972):
- Eastward: -59 ± 10 ns (observed), -40 ± 23 ns (predicted)
- Westward: +273 ± 7 ns (observed), +275 ± 21 ns (predicted)

Reference: Hafele & Keating, Science 177, 166-168 (1972)
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from det_si_units import C_SI, G_SI


# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Earth parameters
GM_EARTH = 3.986004418e14  # m³/s²
R_EARTH = 6.378137e6  # m (equatorial radius)
OMEGA_EARTH = 7.2921159e-5  # rad/s (Earth rotation rate)

# Ground velocity at equator
V_GROUND_EQUATOR = OMEGA_EARTH * R_EARTH  # ~465 m/s


# ============================================================================
# HAFELE-KEATING EXPERIMENT PARAMETERS
# ============================================================================

@dataclass
class FlightParameters:
    """Parameters for a Hafele-Keating style flight."""
    name: str
    direction: str  # "eastward" or "westward"
    altitude_m: float  # Average flight altitude
    ground_speed_m_s: float  # Speed relative to ground
    duration_hours: float  # Total flight time
    latitude_deg: float  # Average latitude

    @property
    def duration_s(self) -> float:
        return self.duration_hours * 3600

    @property
    def latitude_rad(self) -> float:
        return np.radians(self.latitude_deg)

    @property
    def earth_radius_at_latitude(self) -> float:
        """Effective radius for rotation at given latitude."""
        return R_EARTH * np.cos(self.latitude_rad)

    @property
    def ground_velocity_at_latitude(self) -> float:
        """Ground velocity due to Earth rotation at latitude."""
        return OMEGA_EARTH * self.earth_radius_at_latitude

    @property
    def inertial_velocity(self) -> float:
        """Velocity in inertial frame."""
        v_ground = self.ground_velocity_at_latitude

        if self.direction == "eastward":
            # Moving with Earth's rotation
            return v_ground + self.ground_speed_m_s
        else:
            # Moving against Earth's rotation
            return v_ground - self.ground_speed_m_s


# Original Hafele-Keating flight parameters (approximate)
EASTWARD_FLIGHT = FlightParameters(
    name="Eastward (Oct 4-7, 1971)",
    direction="eastward",
    altitude_m=9000,  # ~30,000 ft average
    ground_speed_m_s=250,  # ~900 km/h average
    duration_hours=41.2,
    latitude_deg=35  # Approximate average
)

WESTWARD_FLIGHT = FlightParameters(
    name="Westward (Oct 13-17, 1971)",
    direction="westward",
    altitude_m=9000,
    ground_speed_m_s=250,
    duration_hours=48.6,
    latitude_deg=35
)


# ============================================================================
# DET PREDICTIONS
# ============================================================================

def compute_gravitational_shift(altitude_m: float, duration_s: float) -> float:
    """
    Compute gravitational time shift.

    Clocks at altitude run FASTER.

    Δτ_grav = ∫(ΔΦ/c²)dt ≈ (g*h/c²) * T

    Returns shift in nanoseconds (positive = gained time)
    """
    g = GM_EARTH / R_EARTH**2  # Surface gravity
    frac_shift = g * altitude_m / C_SI**2
    shift_s = frac_shift * duration_s
    return shift_s * 1e9  # Convert to ns


def compute_kinematic_shift(v_inertial: float, v_ground_ref: float,
                           duration_s: float) -> float:
    """
    Compute kinematic (velocity) time shift.

    Moving clocks run SLOWER. The effect depends on velocity in inertial frame.

    Δτ_kin = ∫[(1 - v_ref²/c²)^(1/2) - (1 - v_fly²/c²)^(1/2)]dt
           ≈ (v_ref² - v_fly²)/(2c²) * T  [for v << c]

    Returns shift in nanoseconds (positive = gained time, negative = lost time)
    """
    # Reference clock velocity (ground at given latitude)
    # Flying clock velocity (in inertial frame)

    # Kinematic dilation factors (to first order)
    dilation_ref = -v_ground_ref**2 / (2 * C_SI**2)
    dilation_fly = -v_inertial**2 / (2 * C_SI**2)

    # Relative shift (flying minus reference)
    frac_shift = dilation_fly - dilation_ref
    shift_s = frac_shift * duration_s
    return shift_s * 1e9  # Convert to ns


def compute_det_prediction(flight: FlightParameters) -> Dict:
    """
    Compute DET prediction for Hafele-Keating flight.

    DET formula:
    P = a·σ/(1+F)/(1+H) × γ_v^(-1)

    Gravitational: (1+F)^(-1) gives +45.85 μs/day at GPS altitude
    Kinematic: γ_v^(-1) = √(1 - v²/c²) ≈ 1 - v²/(2c²)
    """
    # Gravitational shift (always positive - flying clocks gain time)
    grav_shift_ns = compute_gravitational_shift(
        flight.altitude_m,
        flight.duration_s
    )

    # Kinematic shift
    v_ground_ref = flight.ground_velocity_at_latitude
    v_inertial = flight.inertial_velocity

    kin_shift_ns = compute_kinematic_shift(
        v_inertial,
        v_ground_ref,
        flight.duration_s
    )

    # Total shift
    total_shift_ns = grav_shift_ns + kin_shift_ns

    return {
        'name': flight.name,
        'direction': flight.direction,
        'altitude_m': flight.altitude_m,
        'duration_h': flight.duration_hours,
        'v_ground_ref': v_ground_ref,
        'v_inertial': v_inertial,
        'grav_shift_ns': grav_shift_ns,
        'kin_shift_ns': kin_shift_ns,
        'total_shift_ns': total_shift_ns
    }


# ============================================================================
# VALIDATION
# ============================================================================

# Published results from Hafele & Keating (1972)
PUBLISHED_RESULTS = {
    'eastward': {
        'predicted_ns': -40,  # GR prediction
        'predicted_err': 23,
        'observed_ns': -59,
        'observed_err': 10
    },
    'westward': {
        'predicted_ns': 275,
        'predicted_err': 21,
        'observed_ns': 273,
        'observed_err': 7
    }
}


def run_hafele_keating_validation(verbose: bool = True) -> Dict:
    """
    Run Hafele-Keating validation.

    Compares DET predictions to both GR predictions and observations.
    """
    if verbose:
        print("=" * 70)
        print("HAFELE-KEATING EXPERIMENT VALIDATION")
        print("=" * 70)
        print()
        print("Testing DET kinematic + gravitational time dilation predictions.")
        print()

    results = {
        'tests': [],
        'summary': {}
    }

    for flight, published in [
        (EASTWARD_FLIGHT, PUBLISHED_RESULTS['eastward']),
        (WESTWARD_FLIGHT, PUBLISHED_RESULTS['westward'])
    ]:
        det = compute_det_prediction(flight)

        if verbose:
            print("-" * 70)
            print(f"Flight: {flight.name}")
            print("-" * 70)
            print(f"  Duration: {flight.duration_hours:.1f} hours")
            print(f"  Altitude: {flight.altitude_m:.0f} m")
            print(f"  Ground speed: {flight.ground_speed_m_s:.0f} m/s")
            print(f"  Latitude: {flight.latitude_deg}°")
            print()
            print(f"  Velocities (inertial frame):")
            print(f"    Ground reference: {det['v_ground_ref']:.1f} m/s")
            print(f"    Flying clock:     {det['v_inertial']:.1f} m/s")
            print()
            print(f"  Time shifts:")
            print(f"    Gravitational: {det['grav_shift_ns']:+.1f} ns")
            print(f"    Kinematic:     {det['kin_shift_ns']:+.1f} ns")
            print(f"    Total (DET):   {det['total_shift_ns']:+.1f} ns")
            print()
            print(f"  Published (H&K 1972):")
            print(f"    GR prediction: {published['predicted_ns']:+.0f} ± {published['predicted_err']} ns")
            print(f"    Observed:      {published['observed_ns']:+.0f} ± {published['observed_err']} ns")
            print()

        # Compare to observed
        obs = published['observed_ns']
        obs_err = published['observed_err']
        det_pred = det['total_shift_ns']

        # Within 2σ of observation?
        within_2sigma = abs(det_pred - obs) < 2 * obs_err

        # Same sign as observation?
        sign_correct = (det_pred * obs) > 0

        # Relative error vs GR prediction
        gr_pred = published['predicted_ns']
        if gr_pred != 0:
            error_vs_gr = abs(det_pred - gr_pred) / abs(gr_pred)
        else:
            error_vs_gr = abs(det_pred)

        test_pass = sign_correct and (within_2sigma or error_vs_gr < 0.5)

        results['tests'].append({
            'name': flight.name,
            'direction': flight.direction,
            'det_prediction_ns': det_pred,
            'gr_prediction_ns': gr_pred,
            'observed_ns': obs,
            'observed_err': obs_err,
            'sign_correct': sign_correct,
            'within_2sigma': within_2sigma,
            'error_vs_gr': error_vs_gr,
            'pass': test_pass,
            'details': det
        })

        if verbose:
            status = "PASS" if test_pass else "FAIL"
            print(f"  [{status}] Sign correct: {sign_correct}")
            print(f"  [{status}] Within 2σ of obs: {within_2sigma}")
            print(f"  [{status}] Error vs GR: {error_vs_gr*100:.0f}%")
            print()

    # Summary
    passed = sum(1 for t in results['tests'] if t['pass'])
    total = len(results['tests'])
    results['summary'] = {
        'passed': passed,
        'total': total,
        'all_pass': passed == total
    }

    if verbose:
        print("=" * 70)
        print(f"TOTAL: {passed}/{total} PASSED")
        if results['summary']['all_pass']:
            print("DET MATCHES HAFELE-KEATING OBSERVATIONS ✓")
        else:
            print("DET PREDICTIONS NEED REFINEMENT")
        print("=" * 70)

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    results = run_hafele_keating_validation(verbose=True)
    sys.exit(0 if results['summary']['all_pass'] else 1)
