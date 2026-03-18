#!/usr/bin/env python3
"""
DET v7.0 — 4D Research Experiment Suite
========================================

Twelve experiments probing what effects a 4D universe would have on
existence under DET canonical dynamics.

Experiments:
  1. Dimensional robustness of canonical laws
  2. Identity persistence phase maps
  3. Orbit and bound-state morphology
  4. Gravity-source law under changed dimensionality
  5. Recovery / Jubilee control theory
  6. Coherence / quantum-classical regime shifts
  7. Information transport and "hidden direction" effects
  8. 3D projection phenomenology
  9. Black-hole / trapped-debt analogs
 10. Cosmological structure formation by dimension
 11. Agency burden and local freedom
 12. Control-policy engineering

Each experiment returns a results dict that is collected into a
single JSON report.
"""

import os, sys, json, time as _time, warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from det_v7_4d_collider import DETCollider4D, DETParams4D
from det_v6_3_3d_collider import DETCollider3D, DETParams3D

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────
def _np(x):
    """Convert numpy types for JSON."""
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_np(v) for v in x]
    return x


def _header(num, title):
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT {num}: {title}")
    print(f"{'='*70}")


# ================================================================
#  EXPERIMENT 1 — Dimensional Robustness of Canonical Laws
# ================================================================
def experiment_01_dimensional_robustness():
    """Run identical canonical updates in 3D and 4D and compare
    conservation, isotropy, and update-order invariants."""
    _header(1, "Dimensional Robustness of Canonical Laws")

    N = 10
    steps = 200

    # ── 3D baseline ──
    p3 = DETParams3D(
        N=N, DT=0.02, gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=True,
        boundary_enabled=False, floor_enabled=True,
    )
    s3 = DETCollider3D(p3)
    c3 = N // 2
    s3.add_packet((c3, c3, c3), mass=10.0, width=2.0, initial_q=0.4)
    m3_init = s3.total_mass()
    for _ in range(steps):
        s3.step()
    m3_final = s3.total_mass()

    # ── 4D ──
    p4 = DETParams4D(
        N=N, DT=0.02, gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=True,
        boundary_enabled=False, floor_enabled=True,
    )
    s4 = DETCollider4D(p4)
    c4 = N // 2
    s4.add_packet((c4, c4, c4, c4), mass=10.0, width=2.0, initial_q=0.4)
    m4_init = s4.total_mass()
    for _ in range(steps):
        s4.step()
    m4_final = s4.total_mass()

    # Isotropy check
    grads_4d = s4.dimension_comparison()
    vals = list(grads_4d.values())
    iso_dev = max(abs(v - np.mean(vals)) / (np.mean(vals) + 1e-12) for v in vals)

    # Conservation
    cons_3d = abs(m3_final - m3_init) / m3_init
    cons_4d = abs(m4_final - m4_init) / m4_init

    # Finite check
    finite_3d = bool(np.isfinite(s3.F).all() and np.isfinite(s3.q).all())
    finite_4d = bool(np.isfinite(s4.F).all() and np.isfinite(s4.q).all())

    res = {
        'conservation_3d': cons_3d,
        'conservation_4d': cons_4d,
        'isotropy_max_deviation': iso_dev,
        'finite_3d': finite_3d,
        'finite_4d': finite_4d,
        'q_mean_3d': float(np.mean(s3.q)),
        'q_mean_4d': float(np.mean(s4.q)),
        'a_mean_3d': float(np.mean(s3.a)),
        'a_mean_4d': float(np.mean(s4.a)),
        'verdict': 'ROBUST' if (cons_4d < 0.01 and iso_dev < 0.2 and finite_4d) else 'FRAGILE',
    }
    print(f"  Conservation 3D: {cons_3d:.2e}  4D: {cons_4d:.2e}")
    print(f"  Isotropy deviation: {iso_dev*100:.1f}%")
    print(f"  Finite 3D: {finite_3d}  4D: {finite_4d}")
    print(f"  Verdict: {res['verdict']}")
    return res


# ================================================================
#  EXPERIMENT 2 — Identity Persistence Phase Maps
# ================================================================
def experiment_02_identity_persistence():
    """Sweep alpha_q × delta_q to map the phase boundary between
    stable identity, slow diffusion, total annealing, and oscillatory
    collapse in 4D."""
    _header(2, "Identity Persistence Phase Maps")

    N = 8
    steps = 600

    alpha_q_vals = [0.001, 0.005, 0.012, 0.025, 0.05]
    delta_q_vals = [0.0, 0.002, 0.005, 0.010, 0.020]

    phase_map = []

    for aq in alpha_q_vals:
        for dq in delta_q_vals:
            p = DETParams4D(
                N=N, DT=0.02,
                gravity_enabled=False, floor_enabled=True,
                boundary_enabled=True, grace_enabled=True,
                q_enabled=True, alpha_q=aq,
                jubilee_enabled=(dq > 0),
                delta_q=dq, n_q=1.0, D_0=0.005,
                lambda_P=3.0,
                momentum_enabled=True, angular_momentum_enabled=False,
            )
            sim = DETCollider4D(p)

            # Localized identity: high q in core, low background
            w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
            c = N // 2
            r2 = (x - c)**2 + (y - c)**2 + (z - c)**2 + (w - c)**2
            sim.F[:] = 1.5
            sim.q[:] = 0.05
            core = r2 <= 4
            far = r2 >= 16
            sim.q[core] = 0.85
            sim.C_X[:] = 0.7; sim.C_Y[:] = 0.7
            sim.C_Z[:] = 0.7; sim.C_W[:] = 0.7
            sim.a[:] = 0.9

            q_core_hist = []
            q_far_hist = []
            for _ in range(steps):
                sim.step()
                q_core_hist.append(float(np.mean(sim.q[core])))
                q_far_hist.append(float(np.mean(sim.q[far])))

            q_core = np.array(q_core_hist)
            q_far = np.array(q_far_hist)

            # Classify phase
            contrast = q_core[-1] - q_far[-1]
            q_core_final = q_core[-1]
            # Check for oscillation in last 200 steps
            tail = q_core[-200:]
            osc_amp = float(np.max(tail) - np.min(tail))

            if osc_amp > 0.15:
                phase = "oscillatory_collapse"
            elif q_core_final < 0.10:
                phase = "total_annealing"
            elif contrast < 0.05:
                phase = "slow_diffusion"
            else:
                phase = "stable_identity"

            phase_map.append({
                'alpha_q': aq, 'delta_q': dq,
                'q_core_final': q_core_final,
                'q_far_final': float(q_far[-1]),
                'contrast': contrast,
                'osc_amplitude': osc_amp,
                'phase': phase,
            })
            print(f"  aq={aq:.3f} dq={dq:.3f} -> {phase} "
                  f"(core={q_core_final:.3f}, contrast={contrast:.3f})")

    # Summary counts
    phases = [p['phase'] for p in phase_map]
    summary = {ph: phases.count(ph) for ph in set(phases)}
    print(f"\n  Phase summary: {summary}")

    return {'phase_map': phase_map, 'summary': summary}


# ================================================================
#  EXPERIMENT 3 — Orbit and Bound-State Morphology
# ================================================================
def experiment_03_orbit_morphology():
    """Study orbit families, precession, capture basins, and multi-body
    stability in 4D."""
    _header(3, "Orbit and Bound-State Morphology")

    N = 12
    steps = 500

    results = {}

    # ── 3a: Two-body orbit families (vary transverse momentum) ──
    orbit_families = []
    for v_trans in [0.05, 0.10, 0.20, 0.35, 0.50]:
        p = DETParams4D(
            N=N, DT=0.015,
            gravity_enabled=True, q_enabled=True,
            momentum_enabled=True, angular_momentum_enabled=True,
            floor_enabled=True, boundary_enabled=False,
            kappa_grav=7.0, mu_grav=2.5, beta_g=12.5,
        )
        sim = DETCollider4D(p)
        c = N // 2
        sim.add_packet((c, c, c, c - 3), mass=10.0, width=1.8,
                       momentum=(0, v_trans, 0, 0), initial_q=0.5)
        sim.add_packet((c, c, c, c + 3), mass=10.0, width=1.8,
                       momentum=(0, -v_trans, 0, 0), initial_q=0.5)

        pe_hist = []
        L_total_hist = []
        for _ in range(steps):
            sim.step()
            pe_hist.append(sim.potential_energy())
            L_all = sim.total_angular_momentum()
            L_total_hist.append(sum(abs(v) for v in L_all.values()))

        pe = np.array(pe_hist)
        bound = float(np.mean(pe[-100:])) < 0
        pe_var = float(np.std(pe[-200:]) / (abs(np.mean(pe[-200:])) + 1e-6))

        if not bound:
            orbit_type = "unbound"
        elif pe_var < 0.05:
            orbit_type = "circular"
        elif pe_var < 0.3:
            orbit_type = "elliptical"
        else:
            orbit_type = "chaotic"

        orbit_families.append({
            'v_trans': v_trans,
            'bound': bound,
            'orbit_type': orbit_type,
            'pe_mean': float(np.mean(pe[-100:])),
            'pe_variability': pe_var,
            'L_final': float(L_total_hist[-1]),
        })
        print(f"  v_trans={v_trans:.2f}: {orbit_type} "
              f"(PE={np.mean(pe[-100:]):.1f}, var={pe_var:.3f})")

    results['orbit_families'] = orbit_families

    # ── 3b: Multi-body stability (3 bodies in 4D) ──
    print("\n  Multi-body (3 bodies):")
    p = DETParams4D(
        N=N, DT=0.015,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=True,
        floor_enabled=True, boundary_enabled=True, grace_enabled=True,
        kappa_grav=7.0, mu_grav=2.5, beta_g=12.5,
    )
    sim = DETCollider4D(p)
    c = N // 2
    # Equilateral triangle in XW plane
    sim.add_packet((c, c, c, c - 3), mass=8.0, width=1.5,
                   momentum=(0, 0.15, 0, 0), initial_q=0.5)
    sim.add_packet((c, c, c, c + 3), mass=8.0, width=1.5,
                   momentum=(0, -0.08, 0, 0.12), initial_q=0.5)
    sim.add_packet((c + 3, c, c, c), mass=8.0, width=1.5,
                   momentum=(0, -0.08, 0, -0.12), initial_q=0.5)

    pe_3body = []
    nblobs_hist = []
    for _ in range(steps):
        sim.step()
        pe_3body.append(sim.potential_energy())
        _, nb = sim.separation()
        nblobs_hist.append(nb)

    pe_3b = np.array(pe_3body)
    multi_bound = float(np.mean(pe_3b[-100:])) < 0
    final_blobs = nblobs_hist[-1]

    results['multi_body'] = {
        'bound': multi_bound,
        'final_blobs': final_blobs,
        'pe_final': float(np.mean(pe_3b[-100:])),
        'merged': final_blobs <= 1,
    }
    print(f"  3-body: bound={multi_bound}, blobs={final_blobs}, "
          f"PE={np.mean(pe_3b[-100:]):.1f}")

    return results


# ================================================================
#  EXPERIMENT 4 — Gravity-Source Law Under Changed Dimensionality
# ================================================================
def experiment_04_gravity_source_scaling():
    """Test whether gravity source rho = q*F and readout M = 1/P
    scale naturally or require renormalization in 4D."""
    _header(4, "Gravity-Source Law Under Changed Dimensionality")

    N = 10
    steps = 100

    # Compare gravity profiles at identical mass/q configurations
    results = {}

    for dim_label, make_sim in [
        ('3D', lambda: _make_grav_sim_3d(N)),
        ('4D', lambda: _make_grav_sim_4d(N)),
    ]:
        sim, center = make_sim()
        for _ in range(steps):
            sim.step()

        # Radial gravity profile
        if dim_label == '3D':
            z, y, x = np.mgrid[0:N, 0:N, 0:N]
            r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
            Phi = sim.Phi
        else:
            w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
            r = np.sqrt((x - center)**2 + (y - center)**2 +
                        (z - center)**2 + (w - center)**2)
            Phi = sim.Phi

        # Bin by radius
        r_flat = r.ravel()
        Phi_flat = Phi.ravel()
        bins = np.arange(0.5, N // 2, 1.0)
        profile = []
        for i in range(len(bins) - 1):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i + 1])
            if np.sum(mask) > 0:
                profile.append({
                    'r': float((bins[i] + bins[i + 1]) / 2),
                    'Phi_mean': float(np.mean(Phi_flat[mask])),
                    'count': int(np.sum(mask)),
                })

        # Presence statistics
        P_center = float(sim.P.ravel()[np.argmin(r_flat)])
        M_center = 1.0 / max(P_center, 1e-12)

        results[dim_label] = {
            'Phi_profile': profile,
            'P_center': P_center,
            'M_center': M_center,
            'Phi_min': float(np.min(Phi)),
            'total_q': float(np.sum(sim.q)),
        }
        print(f"  {dim_label}: Phi_min={np.min(Phi):.3f}, "
              f"P_center={P_center:.6f}, M_center={M_center:.1f}")

    # Compare fall-off rates
    if len(results['3D']['Phi_profile']) > 2 and len(results['4D']['Phi_profile']) > 2:
        r3 = [p['r'] for p in results['3D']['Phi_profile'] if p['Phi_mean'] < 0]
        phi3 = [p['Phi_mean'] for p in results['3D']['Phi_profile'] if p['Phi_mean'] < 0]
        r4 = [p['r'] for p in results['4D']['Phi_profile'] if p['Phi_mean'] < 0]
        phi4 = [p['Phi_mean'] for p in results['4D']['Phi_profile'] if p['Phi_mean'] < 0]

        if len(r3) >= 2 and len(r4) >= 2:
            # Fit log-log slope
            lr3 = np.log(np.array(r3) + 0.1)
            lphi3 = np.log(-np.array(phi3) + 1e-12)
            lr4 = np.log(np.array(r4) + 0.1)
            lphi4 = np.log(-np.array(phi4) + 1e-12)
            slope_3d = float(np.polyfit(lr3, lphi3, 1)[0])
            slope_4d = float(np.polyfit(lr4, lphi4, 1)[0])
            results['falloff_slope_3d'] = slope_3d
            results['falloff_slope_4d'] = slope_4d
            results['requires_renormalization'] = abs(slope_3d - slope_4d) > 0.5
            print(f"  Fall-off slope 3D: {slope_3d:.3f}  4D: {slope_4d:.3f}")
            print(f"  Requires renormalization: {results['requires_renormalization']}")

    return results


def _make_grav_sim_3d(N):
    p = DETParams3D(
        N=N, DT=0.02, gravity_enabled=True, q_enabled=True,
        momentum_enabled=False, angular_momentum_enabled=False,
        boundary_enabled=False, floor_enabled=True,
        kappa_grav=7.0, mu_grav=2.5,
    )
    sim = DETCollider3D(p)
    c = N // 2
    sim.add_packet((c, c, c), mass=12.0, width=2.0, initial_q=0.6)
    return sim, c


def _make_grav_sim_4d(N):
    p = DETParams4D(
        N=N, DT=0.02, gravity_enabled=True, q_enabled=True,
        momentum_enabled=False, angular_momentum_enabled=False,
        boundary_enabled=False, floor_enabled=True,
        kappa_grav=7.0, mu_grav=2.5,
    )
    sim = DETCollider4D(p)
    c = N // 2
    sim.add_packet((c, c, c, c), mass=12.0, width=2.0, initial_q=0.6)
    return sim, c


# ================================================================
#  EXPERIMENT 5 — Recovery / Jubilee Control Theory
# ================================================================
def experiment_05_recovery_jubilee():
    """Sweep Jubilee parameters to map recovery vs washout in 4D
    vs 3D, testing local activation thresholds, energy-coupled vs
    uncapped recovery, and Grace + Jubilee interaction."""
    _header(5, "Recovery / Jubilee Control Theory")

    N = 8
    steps = 800

    configs = [
        ('uncoupled_weak',   {'delta_q': 0.003, 'D_0': 0.01, 'jubilee_energy_coupling': False}),
        ('uncoupled_strong', {'delta_q': 0.010, 'D_0': 0.005, 'jubilee_energy_coupling': False}),
        ('coupled_weak',     {'delta_q': 0.003, 'D_0': 0.01, 'jubilee_energy_coupling': True}),
        ('coupled_strong',   {'delta_q': 0.010, 'D_0': 0.005, 'jubilee_energy_coupling': True}),
        ('grace_only',       {'delta_q': 0.0,   'D_0': 0.01, 'jubilee_energy_coupling': False}),
    ]

    results_3d = {}
    results_4d = {}

    for label, jub_kw in configs:
        for dim, results_dict in [('3D', results_3d), ('4D', results_4d)]:
            jub_on = jub_kw['delta_q'] > 0

            if dim == '3D':
                p = DETParams3D(
                    N=N, DT=0.02,
                    gravity_enabled=False, floor_enabled=True,
                    boundary_enabled=True, grace_enabled=True,
                    q_enabled=True, alpha_q=0.012,
                    jubilee_enabled=jub_on, n_q=1.0, lambda_P=3.0,
                    momentum_enabled=True, angular_momentum_enabled=False,
                    **jub_kw,
                )
                sim = DETCollider3D(p)
                z, y, x = np.mgrid[0:N, 0:N, 0:N]
                c = N // 2
                r2 = (x - c)**2 + (y - c)**2 + (z - c)**2
                sim.F[:] = 2.0
                sim.q[:] = 0.1
                sim.q[r2 <= 4] = 0.8
                sim.C_X[:] = 0.8; sim.C_Y[:] = 0.8; sim.C_Z[:] = 0.8
                sim.a[:] = 0.9
            else:
                p = DETParams4D(
                    N=N, DT=0.02,
                    gravity_enabled=False, floor_enabled=True,
                    boundary_enabled=True, grace_enabled=True,
                    q_enabled=True, alpha_q=0.012,
                    jubilee_enabled=jub_on, n_q=1.0, lambda_P=3.0,
                    momentum_enabled=True, angular_momentum_enabled=False,
                    **jub_kw,
                )
                sim = DETCollider4D(p)
                w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
                c = N // 2
                r2 = (x - c)**2 + (y - c)**2 + (z - c)**2 + (w - c)**2
                sim.F[:] = 2.0
                sim.q[:] = 0.1
                sim.q[r2 <= 4] = 0.8
                sim.C_X[:] = 0.8; sim.C_Y[:] = 0.8
                sim.C_Z[:] = 0.8; sim.C_W[:] = 0.8
                sim.a[:] = 0.9

            q_init = float(np.mean(sim.q))
            q_hist = []
            for _ in range(steps):
                sim.step()
                q_hist.append(float(np.mean(sim.q)))

            q_arr = np.array(q_hist)
            q_final = q_arr[-1]
            recovery_frac = (q_init - q_final) / max(q_init, 1e-6)
            # Check oscillation
            tail = q_arr[-200:]
            osc = float(np.max(tail) - np.min(tail))

            results_dict[label] = {
                'q_initial': q_init,
                'q_final': q_final,
                'recovery_fraction': recovery_frac,
                'oscillation': osc,
                'q_std_final': float(np.std(sim.q)),
            }

    # Print comparison
    for label, _ in configs:
        r3 = results_3d[label]
        r4 = results_4d[label]
        print(f"  {label:20s}  3D: q {r3['q_initial']:.3f}->{r3['q_final']:.3f} "
              f"({r3['recovery_fraction']*100:+.1f}%)  "
              f"4D: q {r4['q_initial']:.3f}->{r4['q_final']:.3f} "
              f"({r4['recovery_fraction']*100:+.1f}%)")

    return {'3D': results_3d, '4D': results_4d}


# ================================================================
#  EXPERIMENT 6 — Coherence / Quantum-Classical Regime Shifts
# ================================================================
def experiment_06_coherence_regimes():
    """Test how 4D adjacency affects coherence dynamics:
    decoherence rates, coherence island persistence, and
    regime transition thresholds."""
    _header(6, "Coherence / Quantum-Classical Regime Shifts")

    N = 10
    steps = 400

    results = {}

    for dim in ['3D', '4D']:
        # ── Decoherence rate measurement ──
        if dim == '3D':
            p = DETParams3D(
                N=N, DT=0.02,
                gravity_enabled=False, floor_enabled=True,
                boundary_enabled=False,
                q_enabled=True, alpha_q=0.005,
                momentum_enabled=True, angular_momentum_enabled=False,
                coherence_dynamic=True, alpha_C=0.04, lambda_C=0.002,
            )
            sim = DETCollider3D(p)
            c = N // 2
            sim.add_packet((c, c, c), mass=10.0, width=2.0, initial_q=0.3)
            # Set high initial coherence
            sim.C_X[:] = 0.95; sim.C_Y[:] = 0.95; sim.C_Z[:] = 0.95
        else:
            p = DETParams4D(
                N=N, DT=0.02,
                gravity_enabled=False, floor_enabled=True,
                boundary_enabled=False,
                q_enabled=True, alpha_q=0.005,
                momentum_enabled=True, angular_momentum_enabled=False,
                coherence_dynamic=True, alpha_C=0.04, lambda_C=0.002,
            )
            sim = DETCollider4D(p)
            c = N // 2
            sim.add_packet((c, c, c, c), mass=10.0, width=2.0, initial_q=0.3)
            sim.C_X[:] = 0.95; sim.C_Y[:] = 0.95
            sim.C_Z[:] = 0.95; sim.C_W[:] = 0.95

        C_mean_hist = []
        C_max_hist = []
        for _ in range(steps):
            sim.step()
            C_all = [getattr(sim, f'C_{ax}') for ax in (['X', 'Y', 'Z'] if dim == '3D'
                                                         else ['X', 'Y', 'Z', 'W'])]
            C_avg = np.mean([np.mean(c) for c in C_all])
            C_mx = np.max([np.max(c) for c in C_all])
            C_mean_hist.append(float(C_avg))
            C_max_hist.append(float(C_mx))

        C_mean = np.array(C_mean_hist)
        C_max = np.array(C_max_hist)

        # Decoherence half-life
        C0 = C_mean[0]
        half_target = (C0 + p.C_init) / 2.0
        half_idx = np.argmax(C_mean < half_target) if np.any(C_mean < half_target) else steps
        
        # Coherence island: fraction of nodes with C > 0.5
        C_all_final = [getattr(sim, f'C_{ax}') for ax in (['X', 'Y', 'Z'] if dim == '3D'
                                                           else ['X', 'Y', 'Z', 'W'])]
        C_avg_final = np.mean(C_all_final, axis=0)
        island_frac = float(np.mean(C_avg_final > 0.5))

        results[dim] = {
            'C_mean_initial': float(C_mean[0]),
            'C_mean_final': float(C_mean[-1]),
            'C_max_final': float(C_max[-1]),
            'decoherence_halflife': int(half_idx),
            'coherence_island_fraction': island_frac,
        }
        print(f"  {dim}: C_mean {C_mean[0]:.3f}->{C_mean[-1]:.3f}, "
              f"half-life={half_idx} steps, island={island_frac*100:.1f}%")

    # Compare
    hl_3d = results['3D']['decoherence_halflife']
    hl_4d = results['4D']['decoherence_halflife']
    results['halflife_ratio'] = hl_4d / max(hl_3d, 1) if hl_3d > 0 else 0
    results['faster_decoherence_in_4d'] = hl_4d < hl_3d
    print(f"  Half-life ratio 4D/3D: {results['halflife_ratio']:.3f}")
    print(f"  Faster decoherence in 4D: {results['faster_decoherence_in_4d']}")

    return results


# ================================================================
#  EXPERIMENT 7 — Information Transport and "Hidden Direction"
# ================================================================
def experiment_07_information_transport():
    """Test whether the 4th dimension acts as faster leakage,
    extra memory, hidden bypass, or stabilizing redundancy."""
    _header(7, "Information Transport and Hidden Direction Effects")

    N = 10
    steps = 300

    results = {}

    # ── 7a: Leakage test — pulse in 3D subspace, measure W-leakage ──
    p = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=False, floor_enabled=False,
        boundary_enabled=False,
        q_enabled=False, momentum_enabled=False,
        angular_momentum_enabled=False,
    )
    sim = DETCollider4D(p)
    c = N // 2
    # Place pulse only in the w=c slice
    sim.F[c, c, c, c] = 20.0

    F_w_profile_init = sim.F[:, c, c, c].copy()
    for _ in range(steps):
        sim.step()
    F_w_profile_final = sim.F[:, c, c, c].copy()

    # How much leaked into W direction
    F_in_w_slice = float(np.sum(sim.F[c]))
    F_total = float(np.sum(sim.F))
    w_leakage = 1.0 - F_in_w_slice / F_total

    results['leakage'] = {
        'w_leakage_fraction': w_leakage,
        'F_w_profile_final': F_w_profile_final.tolist(),
    }
    print(f"  W-leakage fraction: {w_leakage*100:.1f}%")

    # ── 7b: Memory capacity — store pattern in W-layers ──
    p2 = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=False, floor_enabled=True,
        boundary_enabled=False,
        q_enabled=True, alpha_q=0.02,
        momentum_enabled=False, angular_momentum_enabled=False,
    )
    sim2 = DETCollider4D(p2)
    # Store different q patterns in different W layers
    for w_idx in range(N):
        pattern_val = 0.1 * (w_idx + 1) / N
        sim2.q[w_idx, :, :, :] = pattern_val
    sim2.F[:] = 2.0
    sim2.C_X[:] = 0.5; sim2.C_Y[:] = 0.5
    sim2.C_Z[:] = 0.5; sim2.C_W[:] = 0.5
    sim2.a[:] = 0.9

    q_w_means_init = [float(np.mean(sim2.q[w])) for w in range(N)]
    for _ in range(steps):
        sim2.step()
    q_w_means_final = [float(np.mean(sim2.q[w])) for w in range(N)]

    # Measure pattern retention
    corr = float(np.corrcoef(q_w_means_init, q_w_means_final)[0, 1])
    results['memory'] = {
        'pattern_correlation': corr,
        'q_w_init': q_w_means_init,
        'q_w_final': q_w_means_final,
    }
    print(f"  W-layer pattern correlation: {corr:.4f}")

    # ── 7c: Bypass routing — signal propagation speed ──
    p3 = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=False, floor_enabled=False,
        boundary_enabled=False,
        q_enabled=False, momentum_enabled=True,
        angular_momentum_enabled=False,
    )
    sim3 = DETCollider4D(p3)
    # Pulse at one corner
    sim3.F[0, 0, 0, 0] = 30.0
    sim3.pi_X[0, 0, 0, 0] = 1.0
    sim3.pi_W[0, 0, 0, 0] = 1.0

    # Track arrival at opposite corner
    target = (N - 1, N - 1, N - 1, N - 1)
    arrival_hist = []
    for _ in range(steps):
        sim3.step()
        arrival_hist.append(float(sim3.F[target]))

    arr = np.array(arrival_hist)
    arrival_step = np.argmax(arr > p3.F_VAC * 5) if np.any(arr > p3.F_VAC * 5) else steps

    results['bypass'] = {
        'arrival_step': int(arrival_step),
        'arrival_F': float(arr[-1]),
    }
    print(f"  Bypass arrival at opposite corner: step {arrival_step}")

    # Classify W-direction role
    if w_leakage > 0.6:
        role = "faster_leakage"
    elif corr > 0.8:
        role = "extra_memory"
    elif arrival_step < steps * 0.5:
        role = "hidden_bypass"
    else:
        role = "stabilizing_redundancy"

    results['w_role'] = role
    print(f"  W-direction role: {role}")

    return results


# ================================================================
#  EXPERIMENT 8 — 3D Projection Phenomenology
# ================================================================
def experiment_08_projection_phenomenology():
    """Study what 4D processes look like to a 3D observer:
    appearance/disappearance, anomalous forces, shadow orbits,
    identity splitting/merging."""
    _header(8, "3D Projection Phenomenology")

    N = 12
    steps = 400

    results = {}

    # ── 8a: Appearance/disappearance from W-crossing ──
    p = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=False, floor_enabled=True,
    )
    sim = DETCollider4D(p)
    c = N // 2
    # Packet moving through W dimension
    sim.add_packet((2, c, c, c), mass=12.0, width=1.5,
                   momentum=(0.3, 0, 0, 0), initial_q=0.4)

    slice_mass_hist = []  # mass visible in w=c slice
    total_mass_hist = []
    for _ in range(steps):
        sim.step()
        sl = sim.project_3d_slice(c)
        slice_mass_hist.append(float(np.sum(sl['F'])))
        total_mass_hist.append(sim.total_mass())

    sm = np.array(slice_mass_hist)
    tm = np.array(total_mass_hist)

    # Detect appearance/disappearance events
    visible_threshold = np.max(sm) * 0.3
    visible = sm > visible_threshold
    transitions = np.diff(visible.astype(int))
    appearances = int(np.sum(transitions == 1))
    disappearances = int(np.sum(transitions == -1))

    results['appearance'] = {
        'appearances': appearances,
        'disappearances': disappearances,
        'max_slice_mass': float(np.max(sm)),
        'min_slice_mass': float(np.min(sm)),
        'mass_variation': float(np.std(sm) / np.mean(sm)),
    }
    print(f"  Appearances: {appearances}, Disappearances: {disappearances}")
    print(f"  Slice mass range: {np.min(sm):.2f} - {np.max(sm):.2f}")

    # ── 8b: Anomalous force signatures ──
    p2 = DETParams4D(
        N=N, DT=0.02,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=False,
        boundary_enabled=False, floor_enabled=True,
        kappa_grav=7.0, mu_grav=2.5,
    )
    sim2 = DETCollider4D(p2)
    c = N // 2
    # Visible packet in 3D slice
    sim2.add_packet((c, c, c, c), mass=8.0, width=1.5, initial_q=0.4)
    # Hidden packet in different W layer
    sim2.add_packet((c - 4, c, c, c), mass=12.0, width=1.5, initial_q=0.6)

    # Track the visible packet's apparent motion in 3D slice
    com_hist = []
    for _ in range(200):
        sim2.step()
        sl = sim2.project_3d_slice(c)
        F_sl = sl['F']
        total = float(np.sum(F_sl)) + 1e-9
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        cx = float(np.sum(x * F_sl) / total)
        cy = float(np.sum(y * F_sl) / total)
        cz = float(np.sum(z * F_sl) / total)
        com_hist.append((cx, cy, cz))

    com = np.array(com_hist)
    # Anomalous acceleration = deviation from straight-line motion
    if len(com) > 10:
        accel = np.diff(com, n=2, axis=0)
        anomalous_force = float(np.mean(np.linalg.norm(accel, axis=1)))
    else:
        anomalous_force = 0.0

    results['anomalous_force'] = {
        'mean_acceleration': anomalous_force,
        'com_drift': float(np.linalg.norm(com[-1] - com[0])),
    }
    print(f"  Anomalous acceleration: {anomalous_force:.6f}")
    print(f"  COM drift: {results['anomalous_force']['com_drift']:.3f}")

    # ── 8c: Shadow orbit behavior ──
    p3 = DETParams4D(
        N=N, DT=0.015,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=True,
        boundary_enabled=False, floor_enabled=True,
        kappa_grav=7.0, mu_grav=2.5, beta_g=12.5,
    )
    sim3 = DETCollider4D(p3)
    c = N // 2
    # Orbit in XW plane — appears as oscillation in 3D X-slice
    sim3.add_packet((c, c, c, c - 3), mass=10.0, width=1.5,
                    momentum=(0.2, 0, 0, 0), initial_q=0.5)
    sim3.add_packet((c, c, c, c + 3), mass=10.0, width=1.5,
                    momentum=(-0.2, 0, 0, 0), initial_q=0.5)

    shadow_mass = []
    for _ in range(steps):
        sim3.step()
        sl = sim3.project_3d_slice(c)
        shadow_mass.append(float(np.sum(sl['F'])))

    shm = np.array(shadow_mass)
    # Detect oscillatory shadow behavior
    if len(shm) > 50:
        fft_mag = np.abs(np.fft.rfft(shm - np.mean(shm)))
        dominant_freq_idx = np.argmax(fft_mag[1:]) + 1
        shadow_period = len(shm) / dominant_freq_idx if dominant_freq_idx > 0 else 0
    else:
        shadow_period = 0

    results['shadow_orbit'] = {
        'mass_oscillation_amplitude': float(np.std(shm)),
        'dominant_period': shadow_period,
    }
    print(f"  Shadow orbit period: {shadow_period:.1f} steps")
    print(f"  Mass oscillation amplitude: {np.std(shm):.3f}")

    return results


# ================================================================
#  EXPERIMENT 9 — Black-Hole / Trapped-Debt Analogs
# ================================================================
def experiment_09_black_hole_analogs():
    """Test trapped structural debt behavior in 4D:
    debt trapping, horizon sharpness, Hawking-like fluxes,
    identity freeze."""
    _header(9, "Black-Hole / Trapped-Debt Analogs")

    N = 10
    steps = 400

    results = {}

    for dim in ['3D', '4D']:
        if dim == '3D':
            p = DETParams3D(
                N=N, DT=0.015,
                gravity_enabled=True, q_enabled=True,
                momentum_enabled=True, angular_momentum_enabled=False,
                boundary_enabled=True, grace_enabled=True,
                floor_enabled=True,
                kappa_grav=7.0, mu_grav=2.5,
            )
            sim = DETCollider3D(p)
            c = N // 2
            # Create high-q, high-F "trapped debt" region
            sim.add_packet((c, c, c), mass=25.0, width=1.5, initial_q=0.9)
            z, y, x = np.mgrid[0:N, 0:N, 0:N]
            r2 = (x - c)**2 + (y - c)**2 + (z - c)**2
            r = np.sqrt(r2)
        else:
            p = DETParams4D(
                N=N, DT=0.015,
                gravity_enabled=True, q_enabled=True,
                momentum_enabled=True, angular_momentum_enabled=False,
                boundary_enabled=True, grace_enabled=True,
                floor_enabled=True,
                kappa_grav=7.0, mu_grav=2.5,
            )
            sim = DETCollider4D(p)
            c = N // 2
            sim.add_packet((c, c, c, c), mass=25.0, width=1.5, initial_q=0.9)
            w, z, y, x = np.mgrid[0:N, 0:N, 0:N, 0:N]
            r2 = (x - c)**2 + (y - c)**2 + (z - c)**2 + (w - c)**2
            r = np.sqrt(r2)

        # Run and track
        q_core_hist = []
        q_shell_hist = []  # shell at r ~ 3
        P_core_hist = []
        outward_flux_hist = []

        core_mask = r2 <= 4
        shell_mask = (r2 >= 4) & (r2 <= 9)
        far_mask = r2 >= 16

        for _ in range(steps):
            sim.step()
            q_core_hist.append(float(np.mean(sim.q[core_mask])))
            q_shell_hist.append(float(np.mean(sim.q[shell_mask])))
            P_core_hist.append(float(np.mean(sim.P[core_mask])))
            # Outward flux proxy: F in shell region
            outward_flux_hist.append(float(np.mean(sim.F[shell_mask])))

        q_core = np.array(q_core_hist)
        q_shell = np.array(q_shell_hist)
        P_core = np.array(P_core_hist)

        # Horizon sharpness: gradient of q at boundary
        q_gradient = q_core[-1] - q_shell[-1]

        # Identity freeze: how low is P in core?
        P_core_final = P_core[-1]

        # Debt trapping: does core q stay high?
        debt_trapped = q_core[-1] > 0.5

        results[dim] = {
            'q_core_final': float(q_core[-1]),
            'q_shell_final': float(q_shell[-1]),
            'q_gradient': q_gradient,
            'P_core_final': P_core_final,
            'debt_trapped': debt_trapped,
            'outward_flux_final': float(outward_flux_hist[-1]),
        }
        print(f"  {dim}: q_core={q_core[-1]:.3f}, q_shell={q_shell[-1]:.3f}, "
              f"gradient={q_gradient:.3f}, P_core={P_core_final:.6f}")

    # Compare
    results['horizon_sharper_in_4d'] = (results['4D']['q_gradient'] >
                                         results['3D']['q_gradient'])
    results['debt_traps_easier_in_4d'] = (results['4D']['q_core_final'] >
                                           results['3D']['q_core_final'])
    print(f"  Horizon sharper in 4D: {results['horizon_sharper_in_4d']}")
    print(f"  Debt traps easier in 4D: {results['debt_traps_easier_in_4d']}")

    return results


# ================================================================
#  EXPERIMENT 10 — Cosmological Structure Formation
# ================================================================
def experiment_10_cosmological_structure():
    """Compare structure formation rates and morphology between
    3D and 4D: cluster formation, filament morphology, persistence."""
    _header(10, "Cosmological Structure Formation by Dimension")

    N = 10
    steps = 500

    results = {}

    for dim in ['3D', '4D']:
        np.random.seed(42)  # Reproducible initial conditions

        if dim == '3D':
            p = DETParams3D(
                N=N, DT=0.02,
                gravity_enabled=True, q_enabled=True,
                momentum_enabled=True, angular_momentum_enabled=False,
                boundary_enabled=True, grace_enabled=True,
                floor_enabled=True,
                kappa_grav=5.0, mu_grav=2.0,
            )
            sim = DETCollider3D(p)
            # Cosmological initial conditions: uniform + small perturbations
            sim.F[:] = 1.0 + 0.05 * np.random.randn(N, N, N)
            sim.F = np.clip(sim.F, 0.01, 10.0)
            sim.q[:] = 0.05 + 0.02 * np.random.rand(N, N, N)
            sim.C_X[:] = 0.3; sim.C_Y[:] = 0.3; sim.C_Z[:] = 0.3
            sim.a[:] = 0.8
        else:
            p = DETParams4D(
                N=N, DT=0.02,
                gravity_enabled=True, q_enabled=True,
                momentum_enabled=True, angular_momentum_enabled=False,
                boundary_enabled=True, grace_enabled=True,
                floor_enabled=True,
                kappa_grav=5.0, mu_grav=2.0,
            )
            sim = DETCollider4D(p)
            sim.F[:] = 1.0 + 0.05 * np.random.randn(N, N, N, N)
            sim.F = np.clip(sim.F, 0.01, 10.0)
            sim.q[:] = 0.05 + 0.02 * np.random.rand(N, N, N, N)
            sim.C_X[:] = 0.3; sim.C_Y[:] = 0.3
            sim.C_Z[:] = 0.3; sim.C_W[:] = 0.3
            sim.a[:] = 0.8

        F_std_hist = []
        nblobs_hist = []
        for t in range(steps):
            sim.step()
            F_std_hist.append(float(np.std(sim.F)))
            if t % 50 == 0:
                blobs = sim.find_blobs(threshold_ratio=10.0)
                nblobs_hist.append(len(blobs))

        F_std = np.array(F_std_hist)

        # Structure formation: increasing F_std means clustering
        structure_growth = float(F_std[-1] / max(F_std[0], 1e-6))

        # Final cluster count
        blobs_final = sim.find_blobs(threshold_ratio=10.0)

        results[dim] = {
            'F_std_initial': float(F_std[0]),
            'F_std_final': float(F_std[-1]),
            'structure_growth': structure_growth,
            'n_clusters_final': len(blobs_final),
            'largest_cluster_mass': float(blobs_final[0]['mass']) if blobs_final else 0,
            'nblobs_history': nblobs_hist,
        }
        print(f"  {dim}: F_std {F_std[0]:.4f}->{F_std[-1]:.4f} "
              f"(growth={structure_growth:.2f}x), clusters={len(blobs_final)}")

    results['smooths_faster_in_4d'] = (results['4D']['structure_growth'] <
                                        results['3D']['structure_growth'])
    print(f"  4D smooths faster: {results['smooths_faster_in_4d']}")

    return results


# ================================================================
#  EXPERIMENT 11 — Agency Burden and Local Freedom
# ================================================================
def experiment_11_agency_burden():
    """Test whether more adjacency gives more freedom or more burden.
    Measure resource access, coordination load H, and effective
    distinctness."""
    _header(11, "Agency Burden and Local Freedom")

    N = 10
    steps = 300

    results = {}

    for dim in ['3D', '4D']:
        if dim == '3D':
            p = DETParams3D(
                N=N, DT=0.02,
                gravity_enabled=True, q_enabled=True,
                momentum_enabled=True, angular_momentum_enabled=False,
                boundary_enabled=False, floor_enabled=True,
                agency_dynamic=True, sigma_dynamic=True,
                coherence_dynamic=True,
            )
            sim = DETCollider3D(p)
            c = N // 2
            sim.add_packet((c, c, c), mass=12.0, width=2.0, initial_q=0.3)
        else:
            p = DETParams4D(
                N=N, DT=0.02,
                gravity_enabled=True, q_enabled=True,
                momentum_enabled=True, angular_momentum_enabled=False,
                boundary_enabled=False, floor_enabled=True,
                agency_dynamic=True, sigma_dynamic=True,
                coherence_dynamic=True,
            )
            sim = DETCollider4D(p)
            c = N // 2
            sim.add_packet((c, c, c, c), mass=12.0, width=2.0, initial_q=0.3)

        a_hist = []
        P_hist = []
        sigma_hist = []
        for _ in range(steps):
            sim.step()
            a_hist.append(float(np.mean(sim.a)))
            P_hist.append(float(np.mean(sim.P)))
            sigma_hist.append(float(np.mean(sim.sigma)))

        a_arr = np.array(a_hist)
        P_arr = np.array(P_hist)
        sigma_arr = np.array(sigma_hist)

        # Effective distinctness: std(P) / mean(P)
        P_distinctness = float(np.std(sim.P) / (np.mean(sim.P) + 1e-12))

        # Resource access: mean F in neighborhood of high-a nodes
        high_a_mask = sim.a > np.percentile(sim.a, 75)
        if np.any(high_a_mask):
            resource_access = float(np.mean(sim.F[high_a_mask]))
        else:
            resource_access = float(np.mean(sim.F))

        # Coordination load: sigma (higher = more load)
        coord_load = float(np.mean(sim.sigma))

        results[dim] = {
            'a_mean_final': float(a_arr[-1]),
            'P_mean_final': float(P_arr[-1]),
            'sigma_mean_final': float(sigma_arr[-1]),
            'P_distinctness': P_distinctness,
            'resource_access': resource_access,
            'coordination_load': coord_load,
        }
        print(f"  {dim}: a={a_arr[-1]:.4f}, P_distinct={P_distinctness:.4f}, "
              f"resource={resource_access:.3f}, load={coord_load:.4f}")

    # Compare
    results['more_freedom_in_4d'] = (results['4D']['resource_access'] >
                                      results['3D']['resource_access'])
    results['more_burden_in_4d'] = (results['4D']['coordination_load'] >
                                     results['3D']['coordination_load'])
    results['less_distinct_in_4d'] = (results['4D']['P_distinctness'] <
                                       results['3D']['P_distinctness'])
    print(f"  More freedom in 4D: {results['more_freedom_in_4d']}")
    print(f"  More burden in 4D: {results['more_burden_in_4d']}")
    print(f"  Less distinct in 4D: {results['less_distinct_in_4d']}")

    return results


# ================================================================
#  EXPERIMENT 12 — Control-Policy Engineering
# ================================================================
def experiment_12_control_policies():
    """Test intervention policies: local healing, adaptive recovery caps,
    coherence-preserving boundaries, targeted vs uniform Grace,
    dimension-aware stabilization."""
    _header(12, "Control-Policy Engineering")

    N = 8
    steps = 600

    policies = {
        'baseline': {},
        'strong_grace': {'F_MIN_grace': 0.15},
        'adaptive_jubilee': {'jubilee_enabled': True, 'delta_q': 0.006,
                             'D_0': 0.005, 'jubilee_energy_coupling': True},
        'healing_on': {'healing_enabled': True, 'eta_heal': 0.05},
        'coherence_preserving': {'alpha_C': 0.08, 'lambda_C': 0.001},
    }

    results = {}

    for policy_name, overrides in policies.items():
        base_kw = dict(
            N=N, DT=0.02,
            gravity_enabled=True, q_enabled=True,
            momentum_enabled=True, angular_momentum_enabled=False,
            boundary_enabled=True, grace_enabled=True,
            floor_enabled=True,
            kappa_grav=5.0, mu_grav=2.0,
            agency_dynamic=True, sigma_dynamic=True,
            coherence_dynamic=True,
        )
        base_kw.update(overrides)

        p = DETParams4D(**base_kw)
        sim = DETCollider4D(p)
        c = N // 2

        # Two-body scenario
        sim.add_packet((c, c, c, c - 2), mass=10.0, width=1.5,
                       momentum=(0, 0.1, 0, 0), initial_q=0.5)
        sim.add_packet((c, c, c, c + 2), mass=10.0, width=1.5,
                       momentum=(0, -0.1, 0, 0), initial_q=0.5)

        pe_hist = []
        q_hist = []
        a_hist = []
        C_hist = []

        for _ in range(steps):
            sim.step()
            pe_hist.append(sim.potential_energy())
            q_hist.append(float(np.mean(sim.q)))
            a_hist.append(float(np.mean(sim.a)))
            C_all = [getattr(sim, f'C_{ax}') for ax in ['X', 'Y', 'Z', 'W']]
            C_hist.append(float(np.mean([np.mean(c) for c in C_all])))

        pe = np.array(pe_hist)
        q_arr = np.array(q_hist)
        a_arr = np.array(a_hist)
        C_arr = np.array(C_hist)

        # Stability score: low PE variance + high coherence + moderate q
        stability = 1.0 / (1.0 + float(np.std(pe[-200:])) / (abs(float(np.mean(pe[-200:]))) + 1))
        coherence_preserved = float(C_arr[-1])
        identity_score = float(np.std(sim.q))  # spatial variation = identity

        results[policy_name] = {
            'pe_final': float(pe[-1]),
            'pe_stability': stability,
            'q_final': float(q_arr[-1]),
            'a_final': float(a_arr[-1]),
            'C_final': coherence_preserved,
            'identity_score': identity_score,
            'composite_score': stability * coherence_preserved * identity_score,
        }
        print(f"  {policy_name:25s}: PE_stab={stability:.4f}, "
              f"C={coherence_preserved:.3f}, identity={identity_score:.4f}, "
              f"composite={results[policy_name]['composite_score']:.6f}")

    # Rank policies
    ranked = sorted(results.items(), key=lambda x: -x[1]['composite_score'])
    results['ranking'] = [r[0] for r in ranked]
    print(f"\n  Policy ranking: {results['ranking']}")

    return results


# ================================================================
#  MAIN RUNNER
# ================================================================
def run_all_experiments():
    """Run all 12 experiments and save results."""
    print("=" * 70)
    print("  DET v7.0 — 4D RESEARCH EXPERIMENT SUITE")
    print("  12 Experiments on 4D Effects on Existence")
    print("=" * 70)

    all_results = {}
    timings = {}

    experiments = [
        (1,  "Dimensional Robustness",       experiment_01_dimensional_robustness),
        (2,  "Identity Persistence",          experiment_02_identity_persistence),
        (3,  "Orbit Morphology",              experiment_03_orbit_morphology),
        (4,  "Gravity-Source Scaling",         experiment_04_gravity_source_scaling),
        (5,  "Recovery / Jubilee",             experiment_05_recovery_jubilee),
        (6,  "Coherence Regimes",              experiment_06_coherence_regimes),
        (7,  "Information Transport",          experiment_07_information_transport),
        (8,  "Projection Phenomenology",       experiment_08_projection_phenomenology),
        (9,  "Black-Hole Analogs",             experiment_09_black_hole_analogs),
        (10, "Cosmological Structure",         experiment_10_cosmological_structure),
        (11, "Agency Burden",                  experiment_11_agency_burden),
        (12, "Control Policies",               experiment_12_control_policies),
    ]

    for num, name, fn in experiments:
        t0 = _time.time()
        try:
            all_results[f"exp_{num:02d}_{name.lower().replace(' ', '_').replace('/', '_')}"] = fn()
        except Exception as e:
            import traceback
            print(f"\n  EXPERIMENT {num} FAILED: {e}")
            traceback.print_exc()
            all_results[f"exp_{num:02d}_{name.lower().replace(' ', '_').replace('/', '_')}"] = {
                'error': str(e)}
        dt = _time.time() - t0
        timings[f"exp_{num:02d}"] = dt
        print(f"  [{name}: {dt:.1f}s]")

    all_results['timings'] = timings

    # Save
    out_path = os.path.join(REPORT_DIR, "4d_research_results.json")
    with open(out_path, 'w') as f:
        json.dump(_np(all_results), f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print(f"\n{'='*70}")
    print("  ALL 12 EXPERIMENTS COMPLETE")
    print(f"  Total time: {sum(timings.values()):.1f}s")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    run_all_experiments()
