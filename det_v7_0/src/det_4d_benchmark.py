"""
DET v7.0 4D Collider Benchmark Suite
=====================================

Runs comparative benchmarks between 3D and 4D colliders to quantify
the effects of 4D adjacency on DET canonical dynamics.

Produces:
- Diffusion rate comparison (3D vs 4D)
- Binding strength comparison
- Presence/drag landscape comparison
- Recovery dynamics comparison
- Projection visualization data
"""

import os
import sys
import json
import time as time_mod
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D
from det_v7_4d_collider import DETCollider4D, DETParams4D


def benchmark_diffusion(verbose: bool = True):
    """Compare diffusion rates between 3D and 4D.

    Places identical Gaussian packets and measures peak decay rate.
    4D should show faster diffusion due to 8 vs 6 neighbors.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK: Diffusion Rate Comparison (3D vs 4D)")
        print("=" * 60)

    # 3D baseline
    p3 = DETParams3D(
        N=16, DT=0.02,
        gravity_enabled=False, q_enabled=False,
        boundary_enabled=False,
        momentum_enabled=False, angular_momentum_enabled=False,
        floor_enabled=False,
        agency_dynamic=False, sigma_dynamic=False,
    )
    sim3 = DETCollider3D(p3)
    c3 = p3.N // 2
    sim3.add_packet((c3, c3, c3), mass=15.0, width=2.0)

    # 4D
    p4 = DETParams4D(
        N=16, DT=0.02,
        gravity_enabled=False, q_enabled=False,
        boundary_enabled=False,
        momentum_enabled=False, angular_momentum_enabled=False,
        floor_enabled=False,
        agency_dynamic=False, sigma_dynamic=False,
    )
    sim4 = DETCollider4D(p4)
    c4 = p4.N // 2
    sim4.add_packet((c4, c4, c4, c4), mass=15.0, width=2.0)

    peaks_3d = [float(np.max(sim3.F))]
    peaks_4d = [float(np.max(sim4.F))]

    steps = 150
    for i in range(steps):
        sim3.step()
        sim4.step()
        peaks_3d.append(float(np.max(sim3.F)))
        peaks_4d.append(float(np.max(sim4.F)))

    peaks_3d = np.array(peaks_3d)
    peaks_4d = np.array(peaks_4d)

    # Compute half-life of peak (steps to reach 50% of initial)
    half_3d = np.argmax(peaks_3d < peaks_3d[0] * 0.5) if np.any(peaks_3d < peaks_3d[0] * 0.5) else steps
    half_4d = np.argmax(peaks_4d < peaks_4d[0] * 0.5) if np.any(peaks_4d < peaks_4d[0] * 0.5) else steps

    # Decay rate (exponential fit over first 50 steps)
    t = np.arange(min(50, steps))
    if len(t) > 2 and peaks_3d[1] > 0 and peaks_4d[1] > 0:
        log_p3 = np.log(np.maximum(peaks_3d[:len(t)], 1e-12))
        log_p4 = np.log(np.maximum(peaks_4d[:len(t)], 1e-12))
        rate_3d = -float(np.polyfit(t, log_p3, 1)[0])
        rate_4d = -float(np.polyfit(t, log_p4, 1)[0])
    else:
        rate_3d = rate_4d = 0.0

    ratio = rate_4d / max(rate_3d, 1e-12)

    results = {
        'peak_initial_3d': float(peaks_3d[0]),
        'peak_initial_4d': float(peaks_4d[0]),
        'peak_final_3d': float(peaks_3d[-1]),
        'peak_final_4d': float(peaks_4d[-1]),
        'half_life_3d': int(half_3d),
        'half_life_4d': int(half_4d),
        'decay_rate_3d': rate_3d,
        'decay_rate_4d': rate_4d,
        'rate_ratio_4d_over_3d': ratio,
    }

    if verbose:
        print(f"  3D: peak {peaks_3d[0]:.2f} -> {peaks_3d[-1]:.2f}, "
              f"half-life={half_3d} steps, rate={rate_3d:.4f}")
        print(f"  4D: peak {peaks_4d[0]:.2f} -> {peaks_4d[-1]:.2f}, "
              f"half-life={half_4d} steps, rate={rate_4d:.4f}")
        print(f"  4D/3D rate ratio: {ratio:.3f}")
        print(f"  Expected: ~1.33 (8/6 neighbor ratio)")

    return results


def benchmark_binding_strength(verbose: bool = True):
    """Compare gravitational binding strength between 3D and 4D.

    Measures potential energy depth for identical mass configurations.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK: Binding Strength Comparison (3D vs 4D)")
        print("=" * 60)

    # 3D
    p3 = DETParams3D(
        N=16, DT=0.015, F_VAC=0.001,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        kappa_grav=7.0, mu_grav=2.5, beta_g=12.5,
    )
    sim3 = DETCollider3D(p3)
    c3 = p3.N // 2
    sim3.add_packet((c3, c3, c3), mass=12.0, width=2.0, initial_q=0.5)

    # 4D
    p4 = DETParams4D(
        N=16, DT=0.015, F_VAC=0.001,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=True, grace_enabled=True,
        kappa_grav=7.0, mu_grav=2.5, beta_g=12.5,
    )
    sim4 = DETCollider4D(p4)
    c4 = p4.N // 2
    sim4.add_packet((c4, c4, c4, c4), mass=12.0, width=2.0, initial_q=0.5)

    pe_3d = []
    pe_4d = []
    steps = 200

    for _ in range(steps):
        sim3.step()
        sim4.step()
        pe_3d.append(sim3.potential_energy())
        pe_4d.append(sim4.potential_energy())

    pe_3d = np.array(pe_3d)
    pe_4d = np.array(pe_4d)

    results = {
        'pe_final_3d': float(pe_3d[-1]),
        'pe_final_4d': float(pe_4d[-1]),
        'pe_min_3d': float(np.min(pe_3d)),
        'pe_min_4d': float(np.min(pe_4d)),
        'pe_ratio': float(pe_4d[-1] / pe_3d[-1]) if pe_3d[-1] != 0 else 0,
    }

    if verbose:
        print(f"  3D: PE final = {pe_3d[-1]:.2f}, PE min = {np.min(pe_3d):.2f}")
        print(f"  4D: PE final = {pe_4d[-1]:.2f}, PE min = {np.min(pe_4d):.2f}")
        print(f"  4D/3D PE ratio: {results['pe_ratio']:.3f}")

    return results


def benchmark_presence_landscape(verbose: bool = True):
    """Compare presence/drag landscapes between 3D and 4D.

    Research Question 3: How does 4D adjacency affect P, drag, M = 1/P?
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK: Presence/Drag Landscape (3D vs 4D)")
        print("=" * 60)

    # 3D
    p3 = DETParams3D(
        N=16, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False,
        agency_dynamic=False, sigma_dynamic=False,
        lambda_P=3.0,
    )
    sim3 = DETCollider3D(p3)
    c3 = p3.N // 2
    sim3.add_packet((c3, c3, c3), mass=15.0, width=2.5, initial_q=0.5)

    # 4D
    p4 = DETParams4D(
        N=16, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False,
        agency_dynamic=False, sigma_dynamic=False,
        lambda_P=3.0,
    )
    sim4 = DETCollider4D(p4)
    c4 = p4.N // 2
    sim4.add_packet((c4, c4, c4, c4), mass=15.0, width=2.5, initial_q=0.5)

    for _ in range(50):
        sim3.step()
        sim4.step()

    # Compare presence statistics
    P3 = sim3.P
    P4 = sim4.P

    # Mass readout
    M3 = 1.0 / np.maximum(P3, 1e-12)
    M4 = 1.0 / np.maximum(P4, 1e-12)

    results = {
        'P_mean_3d': float(np.mean(P3)),
        'P_mean_4d': float(np.mean(P4)),
        'P_std_3d': float(np.std(P3)),
        'P_std_4d': float(np.std(P4)),
        'P_min_3d': float(np.min(P3)),
        'P_min_4d': float(np.min(P4)),
        'M_max_3d': float(np.max(M3)),
        'M_max_4d': float(np.max(M4)),
        'drag_mean_3d': float(np.mean(1.0 / (1.0 + p3.lambda_P * sim3.q))),
        'drag_mean_4d': float(np.mean(1.0 / (1.0 + p4.lambda_P * sim4.q))),
    }

    if verbose:
        print(f"  3D: P_mean={results['P_mean_3d']:.6f}, P_std={results['P_std_3d']:.6f}")
        print(f"  4D: P_mean={results['P_mean_4d']:.6f}, P_std={results['P_std_4d']:.6f}")
        print(f"  3D: M_max={results['M_max_3d']:.2f}, drag_mean={results['drag_mean_3d']:.4f}")
        print(f"  4D: M_max={results['M_max_4d']:.2f}, drag_mean={results['drag_mean_4d']:.4f}")

    return results


def benchmark_recovery_dynamics(verbose: bool = True):
    """Compare Jubilee recovery dynamics between 3D and 4D.

    Research Question 6: Does higher connectivity cause over-annealing?
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK: Recovery Dynamics (3D vs 4D)")
        print("=" * 60)

    # 3D
    p3 = DETParams3D(
        N=16, DT=0.02,
        gravity_enabled=False, floor_enabled=True,
        boundary_enabled=True, grace_enabled=True,
        q_enabled=True, alpha_q=0.002,
        jubilee_enabled=True, delta_q=0.008, n_q=1.0, D_0=0.005,
        lambda_P=3.0, jubilee_energy_coupling=True,
    )
    sim3 = DETCollider3D(p3)
    N3 = p3.N
    z3, y3, x3 = np.mgrid[0:N3, 0:N3, 0:N3]
    c3 = N3 // 2
    r2_3 = (x3 - c3)**2 + (y3 - c3)**2 + (z3 - c3)**2
    sim3.F[:] = 2.0
    sim3.q[:] = 0.1
    sim3.q[r2_3 <= 9] = 0.8
    sim3.C_X[:] = 0.8
    sim3.C_Y[:] = 0.8
    sim3.C_Z[:] = 0.8
    sim3.a[:] = 0.9

    # 4D
    p4 = DETParams4D(
        N=16, DT=0.02,
        gravity_enabled=False, floor_enabled=True,
        boundary_enabled=True, grace_enabled=True,
        q_enabled=True, alpha_q=0.002,
        jubilee_enabled=True, delta_q=0.008, n_q=1.0, D_0=0.005,
        lambda_P=3.0, jubilee_energy_coupling=True,
    )
    sim4 = DETCollider4D(p4)
    N4 = p4.N
    w4, z4, y4, x4 = np.mgrid[0:N4, 0:N4, 0:N4, 0:N4]
    c4 = N4 // 2
    r2_4 = (x4 - c4)**2 + (y4 - c4)**2 + (z4 - c4)**2 + (w4 - c4)**2
    sim4.F[:] = 2.0
    sim4.q[:] = 0.1
    sim4.q[r2_4 <= 9] = 0.8
    sim4.C_X[:] = 0.8
    sim4.C_Y[:] = 0.8
    sim4.C_Z[:] = 0.8
    sim4.C_W[:] = 0.8
    sim4.a[:] = 0.9

    q_mean_3d = [float(np.mean(sim3.q))]
    q_mean_4d = [float(np.mean(sim4.q))]

    steps = 500
    for _ in range(steps):
        sim3.step()
        sim4.step()
        q_mean_3d.append(float(np.mean(sim3.q)))
        q_mean_4d.append(float(np.mean(sim4.q)))

    q3 = np.array(q_mean_3d)
    q4 = np.array(q_mean_4d)

    # Recovery rate: how much q decreased
    recovery_3d = (q3[0] - q3[-1]) / q3[0] if q3[0] > 0 else 0
    recovery_4d = (q4[0] - q4[-1]) / q4[0] if q4[0] > 0 else 0

    results = {
        'q_initial_3d': float(q3[0]),
        'q_final_3d': float(q3[-1]),
        'q_initial_4d': float(q4[0]),
        'q_final_4d': float(q4[-1]),
        'recovery_fraction_3d': recovery_3d,
        'recovery_fraction_4d': recovery_4d,
        'recovery_ratio_4d_over_3d': recovery_4d / max(recovery_3d, 1e-12),
        'q_std_final_3d': float(np.std(sim3.q)),
        'q_std_final_4d': float(np.std(sim4.q)),
    }

    if verbose:
        print(f"  3D: q {q3[0]:.4f} -> {q3[-1]:.4f} (recovery {recovery_3d*100:.1f}%)")
        print(f"  4D: q {q4[0]:.4f} -> {q4[-1]:.4f} (recovery {recovery_4d*100:.1f}%)")
        print(f"  4D/3D recovery ratio: {results['recovery_ratio_4d_over_3d']:.3f}")
        print(f"  3D q_std final: {results['q_std_final_3d']:.4f}")
        print(f"  4D q_std final: {results['q_std_final_4d']:.4f}")

    return results


def benchmark_4d_projection(verbose: bool = True):
    """Generate and validate 4D-to-3D projection data.

    Research Question 7: Can 4D structures be projected into 3D
    observables consistently?
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BENCHMARK: 4D-to-3D Projection")
        print("=" * 60)

    p4 = DETParams4D(
        N=12, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=True,
        boundary_enabled=True, grace_enabled=True,
    )
    sim4 = DETCollider4D(p4)
    c = p4.N // 2
    sim4.add_packet((c, c, c, c), mass=15.0, width=2.0, initial_q=0.5, initial_spin=0.5)

    for _ in range(100):
        sim4.step()

    # Generate projections
    slice_center = sim4.project_3d_slice(c)
    slice_edge = sim4.project_3d_slice(0)
    proj_max = sim4.project_3d_max()
    proj_sum = sim4.project_3d_sum()
    w_prof = sim4.w_profile(c, c, c)
    dim_comp = sim4.dimension_comparison()

    results = {
        'slice_center_F_peak': float(np.max(slice_center['F'])),
        'slice_edge_F_peak': float(np.max(slice_edge['F'])),
        'proj_max_F_peak': float(np.max(proj_max['F'])),
        'proj_sum_F_total': float(np.sum(proj_sum['F'])),
        'total_mass_direct': sim4.total_mass(),
        'w_profile_peak': float(np.max(w_prof['F'])),
        'w_profile_range': float(np.max(w_prof['F']) - np.min(w_prof['F'])),
        'dim_isotropy': dim_comp,
    }

    # Verify sum projection conservation
    sum_err = abs(results['proj_sum_F_total'] - results['total_mass_direct']) / results['total_mass_direct']
    results['sum_projection_error'] = sum_err

    if verbose:
        print(f"  Center slice peak F: {results['slice_center_F_peak']:.4f}")
        print(f"  Edge slice peak F: {results['slice_edge_F_peak']:.4f}")
        print(f"  Max projection peak: {results['proj_max_F_peak']:.4f}")
        print(f"  Sum projection total: {results['proj_sum_F_total']:.4f}")
        print(f"  Direct total mass: {results['total_mass_direct']:.4f}")
        print(f"  Sum projection error: {sum_err:.2e}")
        print(f"  W-profile peak: {results['w_profile_peak']:.4f}")
        print(f"  W-profile range: {results['w_profile_range']:.4f}")
        print(f"  Dimensional gradients: {dim_comp}")

    return results


def run_full_benchmark(verbose: bool = True):
    """Run the complete 3D vs 4D benchmark suite."""
    print("=" * 70)
    print("DET v7.0 4D COLLIDER - FULL BENCHMARK SUITE")
    print("=" * 70)

    all_results = {}

    t0 = time_mod.time()
    all_results['diffusion'] = benchmark_diffusion(verbose)
    t1 = time_mod.time()
    if verbose:
        print(f"  [Diffusion benchmark: {t1-t0:.1f}s]")

    all_results['binding'] = benchmark_binding_strength(verbose)
    t2 = time_mod.time()
    if verbose:
        print(f"  [Binding benchmark: {t2-t1:.1f}s]")

    all_results['presence'] = benchmark_presence_landscape(verbose)
    t3 = time_mod.time()
    if verbose:
        print(f"  [Presence benchmark: {t3-t2:.1f}s]")

    all_results['recovery'] = benchmark_recovery_dynamics(verbose)
    t4 = time_mod.time()
    if verbose:
        print(f"  [Recovery benchmark: {t4-t3:.1f}s]")

    all_results['projection'] = benchmark_4d_projection(verbose)
    t5 = time_mod.time()
    if verbose:
        print(f"  [Projection benchmark: {t5-t4:.1f}s]")

    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total time: {t5-t0:.1f}s")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), '..', 'reports',
                               'benchmark_4d_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert))
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    if verbose:
        print(f"  Results saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    run_full_benchmark()
