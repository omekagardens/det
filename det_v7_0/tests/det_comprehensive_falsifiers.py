"""
DET v6.3 Comprehensive Falsifier Suite
=======================================

Complete falsifier implementation including:
- F1: Locality Violation
- F2: Grace Coercion (a=0 blocks grace)
- F3: Boundary Redundancy
- F4: Regime Transition
- F5: Hidden Global Aggregates
- F6: Gravitational Binding
- F7: Mass Conservation
- F8: Vacuum Momentum
- F9: Symmetry Drift
- F10: Regime Discontinuity
- F_L1: Rotational Flux Conservation
- F_L2: Vacuum Spin No Transport
- F_L3: Orbital Capture
- F_GTD1-2: Gravitational Time Dilation

Reference: DET Theory Card v6.3

Changelog from v6.2:
- Updated to use v6.3 collider with beta_g coupling
- Added lattice correction factor testing
- Enhanced time dilation verification
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.ndimage import label
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


# ============================================================
# FALSIFIER IMPLEMENTATIONS
# ============================================================

def test_F1_locality_violation(verbose: bool = True) -> Dict:
    """
    F1: Locality Violation Test

    Verify that all interactions are strictly local (nearest-neighbor only).
    No information should propagate faster than 1 cell per timestep.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F1: Locality Violation")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02, F_VAC=0.01,
        diff_enabled=True, momentum_enabled=False,
        angular_momentum_enabled=False, floor_enabled=False,
        gravity_enabled=False, boundary_enabled=False,
        q_enabled=False, agency_dynamic=False
    )
    sim = DETCollider3D(params)

    # Place a single-cell perturbation
    center = params.N // 2
    sim.F[center, center, center] = 10.0

    # Track propagation
    propagation_distances = []

    for t in range(20):
        # Find furthest cell with F > threshold
        threshold = params.F_VAC * 2
        above = sim.F > threshold
        if np.any(above):
            z, y, x = np.mgrid[0:params.N, 0:params.N, 0:params.N]
            dists = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
            max_dist = np.max(dists[above])
        else:
            max_dist = 0
        propagation_distances.append(max_dist)
        sim.step()

    # Check that propagation speed <= 1 cell per step
    max_speed = 0
    for i in range(1, len(propagation_distances)):
        speed = propagation_distances[i] - propagation_distances[i-1]
        max_speed = max(max_speed, speed)

    # Allow some tolerance for diffusion spreading
    passed = max_speed <= 2.0  # Conservative bound

    result = {
        'passed': passed,
        'max_speed': max_speed,
        'propagation_distances': propagation_distances
    }

    if verbose:
        print(f"  Max propagation speed: {max_speed:.2f} cells/step")
        print(f"  F1 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F2_grace_coercion(verbose: bool = True) -> Dict:
    """F2: Grace doesn't go to a=0 nodes"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F2: Grace Coercion (a=0 blocks grace)")
        print("="*60)

    params = DETParams3D(
        N=24, boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.15, beta_a=0.0, gravity_enabled=True
    )
    sim = DETCollider3D(params)

    sim.add_packet((12, 12, 6), mass=3.0, width=2.0, momentum=(0, 0, 0.3))
    sim.add_packet((12, 12, 18), mass=3.0, width=2.0, momentum=(0, 0, -0.3))

    # Set sentinel node to a=0
    sz, sy, sx = 12, 12, 12
    sim.a[sz, sy, sx] = 0.0
    sim.F[sz, sy, sx] = 0.01

    for _ in range(200):
        sim.step()

    sentinel_grace = sim.last_grace_injection[sz, sy, sx]
    passed = sentinel_grace == 0.0

    result = {
        'passed': passed,
        'sentinel_a': sim.a[sz, sy, sx],
        'sentinel_grace': sentinel_grace
    }

    if verbose:
        print(f"  Sentinel a = {sim.a[sz, sy, sx]:.4f}")
        print(f"  Sentinel grace = {sentinel_grace:.2e}")
        print(f"  F2 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F3_boundary_redundancy(verbose: bool = True) -> Dict:
    """F3: Boundary ON produces different outcome than OFF"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F3: Boundary Redundancy")
        print("="*60)

    def run_scenario(boundary_on: bool):
        params = DETParams3D(
            N=24, F_VAC=0.02,
            boundary_enabled=boundary_on, grace_enabled=True, F_MIN_grace=0.15,
            gravity_enabled=True
        )
        sim = DETCollider3D(params)
        sim.add_packet((12, 12, 6), mass=2.0, width=2.0, momentum=(0, 0, 0.3))
        sim.add_packet((12, 12, 18), mass=2.0, width=2.0, momentum=(0, 0, -0.3))

        for _ in range(300):
            sim.step()

        return np.mean(sim.F[10:14, 10:14, 10:14]), sim.total_grace_injected

    F_off, grace_off = run_scenario(False)
    F_on, grace_on = run_scenario(True)

    passed = grace_on > grace_off + 0.001

    result = {
        'passed': passed,
        'F_off': F_off,
        'F_on': F_on,
        'grace_off': grace_off,
        'grace_on': grace_on
    }

    if verbose:
        print(f"  OFF: F={F_off:.4f}, grace={grace_off:.4f}")
        print(f"  ON:  F={F_on:.4f}, grace={grace_on:.4f}")
        print(f"  F3 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F4_regime_transition(verbose: bool = True) -> Dict:
    """
    F4: Regime Transition Test

    Verify smooth transition between quantum and classical regimes
    as system parameters change (no discontinuous jumps).
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F4: Regime Transition")
        print("="*60)

    lambda_pi_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = []

    for lp in lambda_pi_values:
        params = DETParams3D(
            N=20, DT=0.02,
            momentum_enabled=True, lambda_pi=lp,
            gravity_enabled=False, boundary_enabled=False
        )
        sim = DETCollider3D(params)
        sim.add_packet((10, 10, 10), mass=5.0, width=3.0, momentum=(0.5, 0.5, 0))

        total_momentum = []
        for _ in range(200):
            px = np.sum(sim.pi_X)
            py = np.sum(sim.pi_Y)
            total_momentum.append(np.sqrt(px**2 + py**2))
            sim.step()

        final_momentum = total_momentum[-1]
        decay_rate = (total_momentum[0] - final_momentum) / total_momentum[0] if total_momentum[0] > 0 else 0
        results.append({
            'lambda_pi': lp,
            'final_momentum': final_momentum,
            'decay_rate': decay_rate
        })

    # Check for monotonic decay rate increase with lambda_pi
    decay_rates = [r['decay_rate'] for r in results]
    monotonic = all(decay_rates[i] <= decay_rates[i+1] for i in range(len(decay_rates)-1))

    # Check for no discontinuous jumps (max ratio between consecutive values)
    max_ratio = 1.0
    for i in range(len(decay_rates)-1):
        if decay_rates[i] > 0.01:
            ratio = decay_rates[i+1] / decay_rates[i]
            max_ratio = max(max_ratio, ratio)

    passed = monotonic and max_ratio < 10.0

    result = {
        'passed': passed,
        'monotonic': monotonic,
        'max_ratio': max_ratio,
        'results': results
    }

    if verbose:
        print(f"  Decay rates: {[f'{r:.3f}' for r in decay_rates]}")
        print(f"  Monotonic: {monotonic}")
        print(f"  Max ratio: {max_ratio:.2f}")
        print(f"  F4 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F5_hidden_global_aggregates(verbose: bool = True) -> Dict:
    """
    F5: Hidden Global Aggregates Test

    Verify that no hidden global state affects local dynamics.
    Two isolated regions should evolve identically if given identical ICs.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F5: Hidden Global Aggregates")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02,
        gravity_enabled=False, boundary_enabled=False
    )

    # Region A: isolated packet in corner
    sim_a = DETCollider3D(params)
    sim_a.add_packet((8, 8, 8), mass=5.0, width=2.0, momentum=(0.3, 0.3, 0))

    # Region B: same packet but with another packet far away
    sim_b = DETCollider3D(params)
    sim_b.add_packet((8, 8, 8), mass=5.0, width=2.0, momentum=(0.3, 0.3, 0))
    sim_b.add_packet((24, 24, 24), mass=10.0, width=2.0, momentum=(-0.3, -0.3, 0))

    # Run both and compare region around first packet
    max_diff = 0
    for _ in range(200):
        sim_a.step()
        sim_b.step()

        # Compare local region
        region_a = sim_a.F[4:12, 4:12, 4:12]
        region_b = sim_b.F[4:12, 4:12, 4:12]
        diff = np.max(np.abs(region_a - region_b))
        max_diff = max(max_diff, diff)

    # Should be identical (within numerical precision)
    passed = max_diff < 1e-10

    result = {
        'passed': passed,
        'max_diff': max_diff
    }

    if verbose:
        print(f"  Max regional difference: {max_diff:.2e}")
        print(f"  F5 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F6_gravitational_binding(verbose: bool = True) -> Dict:
    """F6: Gravitational binding"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F6: Gravitational Binding")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        C_init=0.3, diff_enabled=True,
        momentum_enabled=True, alpha_pi=0.1, lambda_pi=0.002, mu_pi=0.5,
        angular_momentum_enabled=False, floor_enabled=False,
        q_enabled=True, alpha_q=0.02,
        agency_dynamic=True, lambda_a=3.0, beta_a=0.05,
        gravity_enabled=True, alpha_grav=0.01, kappa_grav=10.0, mu_grav=3.0,
        boundary_enabled=True, grace_enabled=True
    )

    sim = DETCollider3D(params)

    initial_sep = 12
    center = params.N // 2
    sim.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, -0.1), initial_q=0.3)

    rec = {'t': [], 'sep': [], 'PE': []}

    for t in range(1500):
        sep, _ = sim.separation()
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['PE'].append(sim.potential_energy())

        if verbose and t % 300 == 0:
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.3f}")

        sim.step()

    initial_sep_m = rec['sep'][0] if rec['sep'][0] > 0 else initial_sep
    final_sep = rec['sep'][-1]
    min_sep = min(rec['sep'])

    sep_decreased = final_sep < initial_sep_m * 0.9
    bound_state = min_sep < initial_sep_m * 0.5

    passed = sep_decreased or bound_state

    result = {
        'passed': passed,
        'initial_sep': initial_sep_m,
        'final_sep': final_sep,
        'min_sep': min_sep,
        'sep_history': rec['sep'],
        'PE_history': rec['PE']
    }

    if verbose:
        print(f"\n  Initial sep: {initial_sep_m:.1f}, Final: {final_sep:.1f}, Min: {min_sep:.1f}")
        print(f"  F6 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F7_mass_conservation(verbose: bool = True) -> Dict:
    """F7: Mass conservation with gravity + boundary"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F7: Mass Conservation")
        print("="*60)

    params = DETParams3D(N=24, F_MIN=0.0, gravity_enabled=True, boundary_enabled=True)
    sim = DETCollider3D(params)
    sim.add_packet((8, 8, 8), mass=10.0, width=3.0, momentum=(0.2, 0.2, 0.2))
    sim.add_packet((16, 16, 16), mass=10.0, width=3.0, momentum=(-0.2, -0.2, -0.2))

    initial_mass = sim.total_mass()

    for t in range(500):
        sim.step()

    final_mass = sim.total_mass()
    grace_added = sim.total_grace_injected
    effective_drift = abs(final_mass - initial_mass - grace_added) / initial_mass

    passed = effective_drift < 0.10

    result = {
        'passed': passed,
        'initial_mass': initial_mass,
        'final_mass': final_mass,
        'grace_added': grace_added,
        'effective_drift': effective_drift
    }

    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Grace added: {grace_added:.4f}")
        print(f"  Effective drift: {effective_drift*100:.4f}%")
        print(f"  F7 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F8_vacuum_momentum(verbose: bool = True) -> Dict:
    """F8: Momentum doesn't push vacuum"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F8: Vacuum Momentum")
        print("="*60)

    params = DETParams3D(
        N=16, momentum_enabled=True, q_enabled=False, floor_enabled=False,
        F_MIN=0.0, gravity_enabled=False, boundary_enabled=False
    )
    sim = DETCollider3D(params)
    sim.F = np.ones_like(sim.F) * params.F_VAC
    sim.pi_X = np.ones_like(sim.pi_X) * 1.0

    initial_mass = sim.total_mass()

    for _ in range(200):
        sim.step()

    final_mass = sim.total_mass()
    drift = abs(final_mass - initial_mass) / initial_mass

    passed = drift < 0.01

    result = {
        'passed': passed,
        'initial_mass': initial_mass,
        'final_mass': final_mass,
        'drift': drift
    }

    if verbose:
        print(f"  Mass drift: {drift*100:.4f}%")
        print(f"  F8 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F9_symmetry_drift(verbose: bool = True) -> Dict:
    """F9: Symmetric IC doesn't drift"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F9: Symmetry Drift")
        print("="*60)

    params = DETParams3D(N=20, momentum_enabled=False, gravity_enabled=False, boundary_enabled=False)
    sim = DETCollider3D(params)

    N = params.N
    sim.add_packet((N//2, N//2, N//4), mass=5.0, width=3.0, momentum=(0, 0, 0))
    sim.add_packet((N//2, N//2, 3*N//4), mass=5.0, width=3.0, momentum=(0, 0, 0))

    initial_com = sim.center_of_mass()

    max_drift = 0
    for _ in range(300):
        com = sim.center_of_mass()
        drift = np.sqrt((com[0] - initial_com[0])**2 +
                       (com[1] - initial_com[1])**2 +
                       (com[2] - initial_com[2])**2)
        max_drift = max(max_drift, drift)
        sim.step()

    passed = max_drift < 1.0

    result = {
        'passed': passed,
        'max_drift': max_drift
    }

    if verbose:
        print(f"  Max COM drift: {max_drift:.4f} cells")
        print(f"  F9 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F10_regime_discontinuity(verbose: bool = True) -> Dict:
    """
    F10: Regime Discontinuity Test

    Sweep lambda_pi and verify no discontinuous behavior in observables.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F10: Regime Discontinuity")
        print("="*60)

    lambda_values = np.linspace(0.001, 0.1, 20)
    final_spreads = []

    for lp in lambda_values:
        params = DETParams3D(
            N=20, DT=0.02,
            momentum_enabled=True, lambda_pi=lp,
            gravity_enabled=False, boundary_enabled=False
        )
        sim = DETCollider3D(params)
        sim.add_packet((10, 10, 10), mass=5.0, width=2.0)

        for _ in range(100):
            sim.step()

        # Measure spatial spread
        z, y, x = np.mgrid[0:params.N, 0:params.N, 0:params.N]
        com = sim.center_of_mass()
        total = np.sum(sim.F) + 1e-9
        spread = np.sqrt(np.sum(((x - com[0])**2 + (y - com[1])**2 + (z - com[2])**2) * sim.F) / total)
        final_spreads.append(spread)

    # Check for discontinuities
    max_jump = 0
    for i in range(1, len(final_spreads)):
        jump = abs(final_spreads[i] - final_spreads[i-1])
        max_jump = max(max_jump, jump)

    passed = max_jump < 2.0  # No jumps larger than 2 cells

    result = {
        'passed': passed,
        'max_jump': max_jump,
        'lambda_values': lambda_values.tolist(),
        'final_spreads': final_spreads
    }

    if verbose:
        print(f"  Max discontinuity: {max_jump:.4f} cells")
        print(f"  F10 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_L1_rotational_conservation(verbose: bool = True) -> Dict:
    """F_L1: Rotational flux conservation (isolated)"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L1: Rotational Flux Conservation")
        print("="*60)

    params = DETParams3D(
        N=24,
        F_MIN=0.0,
        diff_enabled=False,
        momentum_enabled=False,
        floor_enabled=False,
        q_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False,
        coherence_dynamic=False,
        angular_momentum_enabled=True,
        gravity_enabled=False,
        boundary_enabled=False
    )
    sim = DETCollider3D(params)
    center = params.N // 2

    sim.add_packet((center, center, center), mass=15.0, width=4.0)
    sim.add_spin((center, center, center), spin=2.0, width=5.0)

    initial_F = sim.total_mass()
    initial_com = sim.center_of_mass()

    mass_history = []
    com_drift = []

    for t in range(500):
        sim.step()
        mass_history.append(sim.total_mass())
        com = sim.center_of_mass()
        drift = np.sqrt((com[0] - initial_com[0])**2 +
                       (com[1] - initial_com[1])**2 +
                       (com[2] - initial_com[2])**2)
        com_drift.append(drift)

    final_F = sim.total_mass()
    mass_err = abs(final_F - initial_F) / initial_F
    max_drift = max(com_drift)

    passed = mass_err < 1e-10 and max_drift < 0.1

    result = {
        'passed': passed,
        'initial_F': initial_F,
        'final_F': final_F,
        'mass_err': mass_err,
        'max_drift': max_drift
    }

    if verbose:
        print(f"  Mass error: {mass_err:.2e}")
        print(f"  Max COM drift: {max_drift:.6f} cells")
        print(f"  F_L1 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_L2_vacuum_spin(verbose: bool = True) -> Dict:
    """F_L2: Vacuum spin doesn't transport"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L2: Vacuum Spin No Transport")
        print("="*60)

    results = []
    F_vac_values = [0.001, 0.01, 0.1]

    for F_vac in F_vac_values:
        params = DETParams3D(
            N=20,
            F_VAC=F_vac,
            F_MIN=0.0,
            diff_enabled=False,
            momentum_enabled=False,
            floor_enabled=False,
            q_enabled=False,
            agency_dynamic=False,
            sigma_dynamic=False,
            coherence_dynamic=False,
            angular_momentum_enabled=True,
            gravity_enabled=False,
            boundary_enabled=False
        )
        sim = DETCollider3D(params)

        sim.F = np.ones_like(sim.F) * F_vac
        center = params.N // 2
        sim.add_spin((center, center, center), spin=2.0, width=5.0)

        initial_F = sim.total_mass()
        max_J_rot = 0

        for t in range(200):
            max_J_rot = max(max_J_rot, sim.rot_flux_magnitude())
            sim.step()

        final_F = sim.total_mass()
        mass_change = abs(final_F - initial_F)

        results.append({
            'F_vac': F_vac,
            'max_J_rot': max_J_rot,
            'mass_change': mass_change
        })

        if verbose:
            print(f"  F_vac={F_vac}: max|J_rot|={max_J_rot:.6f}, dF={mass_change:.2e}")

    ratio_J = results[2]['max_J_rot'] / (results[0]['max_J_rot'] + 1e-15)
    ratio_F = F_vac_values[2] / F_vac_values[0]
    scaling_ok = 0.5 < ratio_J / ratio_F < 2.0
    mass_ok = all(r['mass_change'] < 1e-10 for r in results)

    passed = scaling_ok and mass_ok

    result = {
        'passed': passed,
        'results': results,
        'scaling_ok': scaling_ok,
        'mass_ok': mass_ok
    }

    if verbose:
        print(f"\n  J_rot scaling ratio: {ratio_J:.2f} (expected ~{ratio_F:.0f})")
        print(f"  F_L2 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_L3_orbital_capture(verbose: bool = True) -> Dict:
    """F_L3: Orbital capture with angular momentum"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F_L3: Orbital Capture")
        print("="*60)

    params = DETParams3D(
        N=50,
        momentum_enabled=True,
        angular_momentum_enabled=True,
        floor_enabled=True,
        gravity_enabled=False,
        boundary_enabled=False
    )
    sim = DETCollider3D(params)
    center = params.N // 2
    sep_init = 15
    b = 3

    sim.add_packet((center - sep_init, center - b, center), mass=10.0, width=2.5, momentum=(2.0, 0, 0))
    sim.add_packet((center + sep_init, center + b, center), mass=10.0, width=2.5, momentum=(-2.0, 0, 0))

    rec = {'t': [], 'sep': [], 'L_z': [], 'angle': []}
    prev_angle = 0
    total_angle = 0

    for t in range(1500):
        sep, num = sim.separation()
        L = sim.total_angular_momentum()

        blobs = sim.find_blobs()
        if len(blobs) >= 2:
            dx = blobs[1]['x'] - blobs[0]['x']
            dy = blobs[1]['y'] - blobs[0]['y']
            angle = np.arctan2(dy, dx)
            d_angle = angle - prev_angle
            if d_angle > np.pi: d_angle -= 2*np.pi
            elif d_angle < -np.pi: d_angle += 2*np.pi
            total_angle += d_angle
            prev_angle = angle

        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['L_z'].append(L[2])
        rec['angle'].append(total_angle)

        sim.step()

    sep_array = np.array(rec['sep'])
    valid_seps = sep_array[sep_array > 0]

    if len(valid_seps) > 10:
        second_half = valid_seps[len(valid_seps)//2:]
        sep_max = np.max(second_half)
        total_revolutions = abs(total_angle) / (2 * np.pi)
        bounded = sep_max < sep_init * 2.5
        orbital_capture = bounded and (total_revolutions > 0.1)
    else:
        orbital_capture = False
        total_revolutions = 0

    passed = orbital_capture

    result = {
        'passed': passed,
        'total_revolutions': total_revolutions,
        'final_L_z': rec['L_z'][-1],
        'sep_history': rec['sep'],
        'angle_history': rec['angle']
    }

    if verbose:
        print(f"  Orbital capture: {orbital_capture}")
        print(f"  Revolutions: {total_revolutions:.2f}")
        print(f"  Final L_z: {rec['L_z'][-1]:.4f}")
        print(f"  F_L3 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_GTD1_time_dilation(verbose: bool = True) -> Dict:
    """
    F_GTD1: Gravitational Time Dilation Test

    Verify presence-based time dilation: P = a*sigma/(1+F)/(1+H)
    High-F regions should have lower presence (slower proper time).
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_GTD1: Gravitational Time Dilation")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False,
        agency_dynamic=False,  # Keep agency uniform
        sigma_dynamic=False    # Keep sigma uniform
    )
    sim = DETCollider3D(params)

    # Create a massive body
    center = params.N // 2
    sim.add_packet((center, center, center), mass=20.0, width=3.0, initial_q=0.5)

    # Let it settle
    for _ in range(100):
        sim.step()

    # Measure P and F at different distances from center
    F_center = sim.F[center, center, center]
    F_edge = sim.F[center, center, center + 10]

    P_center = sim.P[center, center, center]
    P_edge = sim.P[center, center, center + 10]

    # With uniform a=1, sigma=1, H=sigma=1: P = 1/(1+F)/2
    # So P_center/P_edge = (1+F_edge)/(1+F_center)
    a_center = sim.a[center, center, center]
    a_edge = sim.a[center, center, center + 10]
    sigma_center = sim.sigma[center, center, center]
    sigma_edge = sim.sigma[center, center, center + 10]
    H_center = sigma_center
    H_edge = sigma_edge

    # Full formula: P = a*sigma/(1+F)/(1+H)
    predicted_P_center = a_center * sigma_center / (1 + F_center) / (1 + H_center)
    predicted_P_edge = a_edge * sigma_edge / (1 + F_edge) / (1 + H_edge)

    # Key test: higher F should mean lower P (time dilation)
    time_dilated = P_center < P_edge

    # Also check the formula is correctly implemented
    formula_error_center = abs(P_center - predicted_P_center) / (predicted_P_center + 1e-10)
    formula_error_edge = abs(P_edge - predicted_P_edge) / (predicted_P_edge + 1e-10)
    formula_correct = formula_error_center < 0.01 and formula_error_edge < 0.01

    passed = time_dilated and formula_correct

    result = {
        'passed': passed,
        'F_center': F_center,
        'F_edge': F_edge,
        'P_center': P_center,
        'P_edge': P_edge,
        'predicted_P_center': predicted_P_center,
        'predicted_P_edge': predicted_P_edge,
        'time_dilated': time_dilated,
        'formula_correct': formula_correct
    }

    if verbose:
        print(f"  F at center: {F_center:.4f}")
        print(f"  F at edge: {F_edge:.4f}")
        print(f"  P at center: {P_center:.6f}")
        print(f"  P at edge: {P_edge:.6f}")
        print(f"  Time dilation (P_center < P_edge): {time_dilated}")
        print(f"  Formula correct: {formula_correct}")
        print(f"  F_GTD1 {'PASSED' if passed else 'FAILED'}")

    return result


def test_F_GTD2_accumulated_proper_time(verbose: bool = True) -> Dict:
    """
    F_GTD2: Accumulated Proper Time Test

    Verify accumulated proper time differs between high-F and low-F regions.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F_GTD2: Accumulated Proper Time")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False
    )
    sim = DETCollider3D(params)

    center = params.N // 2
    sim.add_packet((center, center, center), mass=30.0, width=3.0, initial_q=0.5)

    # Run simulation
    for _ in range(500):
        sim.step()

    # Compare accumulated proper time
    tau_center = sim.accumulated_proper_time[center, center, center]
    tau_edge = sim.accumulated_proper_time[center, center, center + 12]

    # High-F regions should have less accumulated proper time (time dilation)
    time_dilated = tau_center < tau_edge

    # Calculate dilation factor
    dilation_factor = tau_edge / tau_center if tau_center > 0 else 0

    passed = time_dilated and dilation_factor > 1.01

    result = {
        'passed': passed,
        'tau_center': tau_center,
        'tau_edge': tau_edge,
        'dilation_factor': dilation_factor
    }

    if verbose:
        print(f"  Proper time at center: {tau_center:.4f}")
        print(f"  Proper time at edge: {tau_edge:.4f}")
        print(f"  Dilation factor: {dilation_factor:.4f}")
        print(f"  F_GTD2 {'PASSED' if passed else 'FAILED'}")

    return result


# ============================================================
# OPTION B SPECIFIC TESTS
# ============================================================

def test_option_b_binding(verbose: bool = True) -> Dict:
    """Test gravitational binding with Option B coherence-weighted H."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Option B Gravitational Binding")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        coherence_weighted_H=True  # Enable Option B
    )

    sim = DETCollider3D(params)

    initial_sep = 12
    center = params.N // 2
    sim.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, -0.1), initial_q=0.3)

    min_sep = initial_sep

    for t in range(1000):
        sep, _ = sim.separation()
        min_sep = min(min_sep, sep)

        if verbose and t % 200 == 0:
            H_mean = np.mean(sim._compute_coherence_weighted_H())
            print(f"  t={t}: sep={sep:.1f}, H_mean={H_mean:.4f}, PE={sim.potential_energy():.3f}")

        sim.step()

    passed = min_sep < initial_sep * 0.6

    result = {
        'passed': passed,
        'initial_sep': initial_sep,
        'min_sep': min_sep
    }

    if verbose:
        print(f"\n  Initial sep: {initial_sep:.1f}")
        print(f"  Min sep: {min_sep:.1f}")
        print(f"  Option B Binding {'PASSED' if passed else 'FAILED'}")

    return result


def test_option_b_time_dilation(verbose: bool = True) -> Dict:
    """Test time dilation with Option B coherence-weighted H."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Option B Time Dilation")
        print("="*60)

    params = DETParams3D(
        N=32, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False,
        agency_dynamic=False,
        sigma_dynamic=False,
        coherence_weighted_H=True  # Enable Option B
    )
    sim = DETCollider3D(params)

    center = params.N // 2
    sim.add_packet((center, center, center), mass=20.0, width=3.0, initial_q=0.5)

    for _ in range(100):
        sim.step()

    F_center = sim.F[center, center, center]
    F_edge = sim.F[center, center, center + 10]
    P_center = sim.P[center, center, center]
    P_edge = sim.P[center, center, center + 10]

    time_dilated = P_center < P_edge

    # With Option B, H uses coherence-weighted load
    H_center = sim._compute_coherence_weighted_H()[center, center, center]
    H_edge = sim._compute_coherence_weighted_H()[center, center, center + 10]

    a_center = sim.a[center, center, center]
    sigma_center = sim.sigma[center, center, center]
    predicted_P = a_center * sigma_center / (1 + F_center) / (1 + H_center)
    formula_error = abs(P_center - predicted_P) / (predicted_P + 1e-10)

    passed = time_dilated and formula_error < 0.01

    result = {
        'passed': passed,
        'F_center': F_center,
        'F_edge': F_edge,
        'P_center': P_center,
        'P_edge': P_edge,
        'H_center': H_center,
        'H_edge': H_edge,
        'time_dilated': time_dilated,
        'formula_error': formula_error
    }

    if verbose:
        print(f"  F at center: {F_center:.4f}, H: {H_center:.4f}")
        print(f"  F at edge: {F_edge:.4f}, H: {H_edge:.4f}")
        print(f"  P at center: {P_center:.6f}")
        print(f"  P at edge: {P_edge:.6f}")
        print(f"  Time dilation: {time_dilated}")
        print(f"  Formula error: {formula_error*100:.4f}%")
        print(f"  Option B Time Dilation {'PASSED' if passed else 'FAILED'}")

    return result


def test_option_b_mass_conservation(verbose: bool = True) -> Dict:
    """Test mass conservation with Option B."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Option B Mass Conservation")
        print("="*60)

    params = DETParams3D(
        N=24, F_MIN=0.0,
        gravity_enabled=True, boundary_enabled=True,
        coherence_weighted_H=True
    )
    sim = DETCollider3D(params)
    sim.add_packet((8, 8, 8), mass=10.0, width=3.0, momentum=(0.2, 0.2, 0.2))
    sim.add_packet((16, 16, 16), mass=10.0, width=3.0, momentum=(-0.2, -0.2, -0.2))

    initial_mass = sim.total_mass()

    for t in range(500):
        sim.step()

    final_mass = sim.total_mass()
    grace_added = sim.total_grace_injected
    effective_drift = abs(final_mass - initial_mass - grace_added) / initial_mass

    passed = effective_drift < 0.10

    result = {
        'passed': passed,
        'initial_mass': initial_mass,
        'final_mass': final_mass,
        'grace_added': grace_added,
        'effective_drift': effective_drift
    }

    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Grace added: {grace_added:.4f}")
        print(f"  Effective drift: {effective_drift*100:.4f}%")
        print(f"  Option B Mass Conservation {'PASSED' if passed else 'FAILED'}")

    return result


# ============================================================
# MAIN TEST SUITE
# ============================================================

def run_comprehensive_test_suite(include_option_b: bool = False):
    """Run the complete falsifier test suite for DET v6.3."""
    print("="*70)
    print("DET v6.3 COMPREHENSIVE FALSIFIER SUITE")
    if include_option_b:
        print("(Including Option B: Coherence-Weighted Load Tests)")
    print("="*70)

    start_time = time.time()
    results = {}

    # Core falsifiers
    results['F1'] = test_F1_locality_violation(verbose=True)
    results['F2'] = test_F2_grace_coercion(verbose=True)
    results['F3'] = test_F3_boundary_redundancy(verbose=True)
    results['F4'] = test_F4_regime_transition(verbose=True)
    results['F5'] = test_F5_hidden_global_aggregates(verbose=True)
    results['F6'] = test_F6_gravitational_binding(verbose=True)
    results['F7'] = test_F7_mass_conservation(verbose=True)
    results['F8'] = test_F8_vacuum_momentum(verbose=True)
    results['F9'] = test_F9_symmetry_drift(verbose=True)
    results['F10'] = test_F10_regime_discontinuity(verbose=True)

    # Angular momentum falsifiers
    results['F_L1'] = test_F_L1_rotational_conservation(verbose=True)
    results['F_L2'] = test_F_L2_vacuum_spin(verbose=True)
    results['F_L3'] = test_F_L3_orbital_capture(verbose=True)

    # Gravitational time dilation falsifiers
    results['F_GTD1'] = test_F_GTD1_time_dilation(verbose=True)
    results['F_GTD2'] = test_F_GTD2_accumulated_proper_time(verbose=True)

    # Option B tests
    if include_option_b:
        print("\n" + "="*70)
        print("OPTION B: COHERENCE-WEIGHTED LOAD TESTS")
        print("="*70)
        results['OptionB_Binding'] = test_option_b_binding(verbose=True)
        results['OptionB_TimeDilation'] = test_option_b_time_dilation(verbose=True)
        results['OptionB_MassConservation'] = test_option_b_mass_conservation(verbose=True)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE SUITE SUMMARY")
    print("="*70)

    passed_count = 0
    total_count = 0

    for name, result in results.items():
        if isinstance(result, dict):
            status = result.get('passed', False)
        else:
            status = result
        passed_count += 1 if status else 0
        total_count += 1
        print(f"  {name}: {'PASS' if status else 'FAIL'}")

    print(f"\n  TOTAL: {passed_count}/{total_count} PASSED")
    print(f"  Runtime: {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='DET v6.3 Comprehensive Falsifier Suite')
    parser.add_argument('--option-b', action='store_true', help='Include Option B tests')
    args = parser.parse_args()

    results = run_comprehensive_test_suite(include_option_b=args.option_b)
