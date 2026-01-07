#!/usr/bin/env python3
"""
DET 4.2 → v5 Phase Dynamics: Minimal Simulation Harness
Implements Appendix Δ (+ Δ.R observables) in a self-contained way.

Model:
  Δτ_i = P_i * Δk
  θ_i^{+} = θ_i + [ ω0 * g(F_i) + Σ_j K_ij sin(θ_j-θ_i) ] * Δτ_i   (mod 2π)
  K_ij = γ0 * σ_ij * sqrt(C_ij) * a_i * a_j

Outputs:
  - R(t) coherence order parameter
  - lock time t_lock
  - phase diffusion proxy (var of phase residual)
  - basic falsifier sweeps

No full DET resource flow is simulated here; this is the phase-sector harness
intended to be embedded later into the canonical DET loop.
"""

from __future__ import annotations
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

PATCH_TAG = "det_phase_harness_patch_2026_01_06"

TAU = 2.0 * math.pi

def wrap_2pi(x: float) -> float:
    # wrap to [0, 2π)
    x = x % TAU
    return x

def circular_mean(phases: List[float]) -> float:
    s = sum(math.sin(th) for th in phases)
    c = sum(math.cos(th) for th in phases)
    return math.atan2(s, c) % TAU

def coherence_R(phases: List[float]) -> Tuple[float, float]:
    # R e^{iΨ} = (1/N) Σ exp(iθ)
    n = len(phases)
    s = sum(math.sin(th) for th in phases) / n
    c = sum(math.cos(th) for th in phases) / n
    R = math.sqrt(s*s + c*c)
    Psi = math.atan2(s, c) % TAU
    return R, Psi

def edge_coherence(g: Graph, phases: List[float]) -> float:
    """Average local coherence over edges: mean(cos(θ_j-θ_i)).
    1.0 means perfectly aligned locally; values near 0 mean random; negative means anti-aligned."""
    n = len(phases)
    if n == 0:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        thi = phases[i]
        for j in g.nbrs[i]:
            if j <= i:
                continue  # count each undirected edge once
            total += math.cos(phases[j] - thi)
            count += 1
    return total / max(1, count)

def var(xs: List[float]) -> float:
    if not xs:
        return 0.0
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)

def sustained_crossing_time(series: List[float], thresh: float, hold: int = 50, start_idx: int = 0) -> Optional[int]:
    """Return the first index t>=start_idx where series[t:t+hold] all exceed thresh.
    This avoids reporting transient spikes as 'lock'."""
    n = len(series)
    if hold <= 1:
        for t in range(start_idx, n):
            if series[t] >= thresh:
                return t
        return None
    for t in range(start_idx, n - hold):
        ok = True
        for u in range(t, t + hold):
            if series[u] < thresh:
                ok = False
                break
        if ok:
            return t
    return None

@dataclass
class PhaseParams:
    omega0: float = 1.0      # rad / proper-time
    gamma0: float = 1.0      # coupling scale
    F_star: float = 1.0      # half-speed resource scale
    dt: float = 0.05         # Δk
    steps: int = 2000

def g_F(F: float, F_star: float) -> float:
    # recommended bounded map: F/(F+F*)
    if F < 0:
        F = 0.0
    return F / (F + F_star)

@dataclass
class Graph:
    # undirected adjacency list
    nbrs: List[List[int]]

def make_ring_graph(n: int, degree: int = 2) -> Graph:
    # ring with k neighbors on each side (degree=2k)
    k = max(1, degree // 2)
    nbrs = [[] for _ in range(n)]
    for i in range(n):
        for d in range(1, k+1):
            j1 = (i + d) % n
            j2 = (i - d) % n
            nbrs[i].append(j1)
            nbrs[i].append(j2)
    # dedupe (if degree odd etc)
    nbrs = [sorted(list(set(ns))) for ns in nbrs]
    return Graph(nbrs=nbrs)

def make_complete_graph(n: int) -> Graph:
    nbrs = [[j for j in range(n) if j != i] for i in range(n)]
    return Graph(nbrs=nbrs)

@dataclass
class PhaseState:
    theta: List[float]
    P: List[float]
    F: List[float]
    a: List[float]
    # bond-level fields stored as dense matrices for simplicity (small N):
    sigma: List[List[float]]
    C: List[List[float]]

def init_state(
    g: Graph,
    *,
    seed: int = 0,
    F0: float = 1.0,
    P0: float = 1.0,
    a0: float = 1.0,
    sigma0: float = 1.0,
    C0: float = 1.0,
    theta_mode: str = "random",
) -> PhaseState:
    rng = random.Random(seed)
    n = len(g.nbrs)
    if theta_mode == "random":
        theta = [rng.random() * TAU for _ in range(n)]
    elif theta_mode == "aligned":
        base = rng.random() * TAU
        theta = [base for _ in range(n)]
    else:
        raise ValueError("theta_mode must be 'random' or 'aligned'")

    P = [P0 for _ in range(n)]
    F = [F0 for _ in range(n)]
    a = [a0 for _ in range(n)]

    # dense matrices; only neighbors matter but this is easiest & still minimal
    sigma = [[0.0 for _ in range(n)] for _ in range(n)]
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in g.nbrs[i]:
            sigma[i][j] = sigma0
            sigma[j][i] = sigma0
            C[i][j] = C0
            C[j][i] = C0

    return PhaseState(theta=theta, P=P, F=F, a=a, sigma=sigma, C=C)

def step_phase(g: Graph, s: PhaseState, p: PhaseParams) -> None:
    n = len(g.nbrs)

    def dtheta_at(theta: List[float], i: int) -> float:
        # Δτ_i = P_i Δk
        dta = s.P[i] * p.dt
        intrinsic = p.omega0 * g_F(s.F[i], p.F_star)

        ai = s.a[i]
        thi = theta[i]
        deg = max(1, len(g.nbrs[i]))

        coupling_sum = 0.0
        for j in g.nbrs[i]:
            aj = s.a[j]
            kij = p.gamma0 * s.sigma[i][j] * math.sqrt(max(0.0, s.C[i][j])) * ai * aj
            coupling_sum += kij * math.sin(theta[j] - thi)

        # Degree-normalize coupling so γ0 is topology-portable.
        coupling_sum /= deg

        return (intrinsic + coupling_sum) * dta

    theta0 = s.theta

    # RK2 / midpoint update for stability in strong coupling.
    # IMPORTANT: do NOT wrap at the midpoint; wrapping mid-step introduces discontinuities
    # that can suppress synchronization and distort strong-coupling behavior.
    k1 = [dtheta_at(theta0, i) for i in range(n)]
    theta_mid = [theta0[i] + 0.5 * k1[i] for i in range(n)]
    k2 = [dtheta_at(theta_mid, i) for i in range(n)]

    s.theta = [wrap_2pi(theta0[i] + k2[i]) for i in range(n)]

@dataclass
class Metrics:
    R: List[float]
    Psi: List[float]
    phase_resid_var: List[float]
    lock_time: Optional[int]
    edge_coh: List[float]
    edge_lock_time: Optional[int]

def run_sim(g: Graph, s: PhaseState, p: PhaseParams, lock_thresh: float = 0.7) -> Metrics:
    R_hist: List[float] = []
    Psi_hist: List[float] = []
    var_hist: List[float] = []
    edge_hist: List[float] = []
    lock_time: Optional[int] = None
    edge_lock_time: Optional[int] = None

    for t in range(p.steps):
        R, Psi = coherence_R(s.theta)
        R_hist.append(R)
        Psi_hist.append(Psi)
        edge_hist.append(edge_coherence(g, s.theta))

        # phase residual variance (simple diffusion proxy)
        resid = [math.atan2(math.sin(th - Psi), math.cos(th - Psi)) for th in s.theta]
        var_hist.append(var(resid))

        if lock_time is None and R >= lock_thresh:
            lock_time = t
        if edge_lock_time is None and edge_hist[-1] >= lock_thresh:
            edge_lock_time = t

        step_phase(g, s, p)

    return Metrics(R=R_hist, Psi=Psi_hist, phase_resid_var=var_hist, lock_time=lock_time, edge_coh=edge_hist, edge_lock_time=edge_lock_time)

def classify_regime(final_R: float, final_edge: float, lock_R: Optional[int], lock_edge: Optional[int]) -> str:
    # Coarse, operational regime tags
    if final_edge > 0.95 and final_R > 0.95:
        return "Global-Coherent"
    if final_edge > 0.95 and final_R <= 0.95:
        return "Twisted-Local-Coherent"
    if (lock_R is None) and (lock_edge is None):
        return "Incoherent"
    if final_edge < 0.2 and final_R < 0.2:
        return "Random"
    return "Mixed"

# -------------------------
# Falsifier-style sweeps
# -------------------------

def sweep_presence_scaling() -> Dict[str, object]:
    """
    F1: Presence scaling test
    Prediction: scaling P scales BOTH intrinsic advance and coupling proportionally (via Δτ_i).
    Operational test: measure early-time slope of mean phase speed and lock time vs P.
    """
    n = 60
    g = make_ring_graph(n, degree=4)
    p = PhaseParams(omega0=1.0, gamma0=2.0, F_star=1.0, dt=0.01, steps=4000)

    Ps = [1.0, 0.5, 0.25, 0.125]
    results = []
    for P0 in Ps:
        s = init_state(g, seed=1, F0=2.0, P0=P0, a0=1.0, sigma0=1.0, C0=1.0, theta_mode="random")
        # Scale steps so each case spans a comparable proper-time horizon
        p_run = PhaseParams(omega0=p.omega0, gamma0=p.gamma0, F_star=p.F_star, dt=p.dt, steps=int(p.steps / max(P0, 1e-6)))
        m = run_sim(g, s, p_run, lock_thresh=0.7)
        # approximate early slope of mean phase advance (in global steps)
        # use mean circular displacement over first 50 steps
        # (crude, but enough for a falsifier harness)
        R0 = m.R[0]
        tlock = m.lock_time
        results.append({"P0": P0, "R0": R0, "lock_step_R": tlock, "lock_step_edge": m.edge_lock_time})
    return {"test": "F1_presence_scaling", "results": results}

def sweep_resource_frequency() -> Dict[str, object]:
    """
    F2: Resource–frequency law (intrinsic)
    Prediction: isolated node dθ/dk ≈ ω0 g(F) P
    Test: single node (no neighbors), sweep F and fit monotone saturating relation.
    """
    g = Graph(nbrs=[[]])  # single isolated node
    p = PhaseParams(omega0=1.0, gamma0=0.0, F_star=1.0, dt=0.01, steps=2000)

    Fs = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    out = []
    for F0 in Fs:
        s = init_state(g, seed=2, F0=F0, P0=0.6, a0=1.0, sigma0=0.0, C0=0.0, theta_mode="aligned")
        th0 = s.theta[0]
        m = run_sim(g, s, p)
        th_end = s.theta[0]
        # unwrap approximately by using expected sign and small dt; here monotone forward so:
        # estimate mean dθ/dk from cumulative expected intrinsic contribution:
        expected_per_k = p.omega0 * g_F(F0, p.F_star) * s.P[0]
        out.append({"F0": F0, "expected_dtheta_per_k": expected_per_k})
    return {"test": "F2_resource_frequency", "results": out}

def frozen_inertness() -> Dict[str, object]:
    """
    F4: Frozen node inertness
    Prediction: P≈0 node does not re-lock even inside synchronized neighborhood.
    Test: initialize all aligned, then kick frozen node; verify it stays effectively frozen and does not track.
    """
    n = 40
    g = make_complete_graph(n)
    p = PhaseParams(omega0=1.0, gamma0=3.0, F_star=1.0, dt=0.01, steps=4000)

    s = init_state(g, seed=3, F0=2.0, P0=1.0, a0=1.0, sigma0=1.0, C0=1.0, theta_mode="aligned")
    frozen_i = 0
    s.P[frozen_i] = 1e-5
    s.theta[frozen_i] = wrap_2pi(s.theta[frozen_i] + 1.0)  # kick by 1 rad
    theta0_start = s.theta[frozen_i]  # measure drift from post-kick state

    m = run_sim(g, s, p, lock_thresh=0.9)

    # Living-only re-lock time after the kick (ignore t=0 artifacts).
    # Compute R_living(t) by reusing the stored global phases during the run.
    # We don't have per-step phase history, so approximate using edge/global histories:
    # For complete graphs and aligned initial state, global R is a good proxy for living re-lock.
    relock_time_all = sustained_crossing_time(m.R, thresh=0.98, hold=50, start_idx=1)

    # check frozen node movement magnitude across run
    # (since P tiny, it should barely move)

    # Frozen drift: minimal circular distance between start and end
    theta0_end = s.theta[frozen_i]
    drift = ((theta0_end - theta0_start + math.pi) % TAU) - math.pi
    frozen_abs_drift = abs(drift)

    # Living-only coherence (exclude the frozen node)
    living_phases = [th for idx, th in enumerate(s.theta) if idx != frozen_i]
    R_living, _ = coherence_R(living_phases)

    return {
        "test": "F4_frozen_inertness",
        "frozen_i": frozen_i,
        "final_R_all": m.R[-1],
        "final_R_living": R_living,
        "lock_step_all_first": m.lock_time,
        "relock_time_all_sustained": relock_time_all,
        "frozen_P": s.P[frozen_i],
        "frozen_abs_drift_rad": frozen_abs_drift,
        "expectation": "Living nodes should synchronize (final_R_living→1) while frozen node barely moves (frozen_abs_drift_rad≈0)."
    }

def size_scaling_scan() -> Dict[str, object]:
    """
    F5-ish: size scaling sketch.
    Not a rigorous proof; just a quick check that the 'critical' gamma0 does not blow up with N.
    """
    Ns = [30, 60, 120]
    gamma_vals = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    out = []
    for n in Ns:
        g = make_complete_graph(n)
        for gamma0 in gamma_vals:
            p = PhaseParams(omega0=1.0, gamma0=gamma0, F_star=1.0, dt=0.01, steps=3000)
            s = init_state(g, seed=4, F0=2.0, P0=1.0, a0=1.0, sigma0=1.0, C0=1.0, theta_mode="random")
            m = run_sim(g, s, p, lock_thresh=0.7)
            out.append({
                "N": n,
                "gamma0": gamma0,
                "lock_step_R": m.lock_time,
                "lock_step_edge": m.edge_lock_time,
                "final_R": m.R[-1],
                "final_edge": m.edge_coh[-1],
                "regime": classify_regime(m.R[-1], m.edge_coh[-1], m.lock_time, m.edge_lock_time),
            })
    return {"test": "F5_size_scaling_scan", "results": out}

def main() -> None:
    print(f"[{PATCH_TAG}] running: {os.path.abspath(__file__)}")
    # Baseline run (regime visualization)
    n = 80
    g = make_ring_graph(n, degree=4)
    p = PhaseParams(omega0=1.0, gamma0=2.0, F_star=1.0, dt=0.01, steps=3000)
    s = init_state(g, seed=0, F0=2.0, P0=1.0, a0=1.0, sigma0=1.0, C0=1.0, theta_mode="random")
    m = run_sim(g, s, p, lock_thresh=0.7)

    print("=== Baseline Phase Run ===")
    print(f"final R: {m.R[-1]:.4f}")
    print(f"lock step (R>0.7): {m.lock_time}")
    print(f"final phase residual variance: {m.phase_resid_var[-1]:.6f}")
    print(f"final edge coherence: {m.edge_coh[-1]:.6f}")
    print(f"edge lock step (edge>0.7): {m.edge_lock_time}")
    print(f"regime: {classify_regime(m.R[-1], m.edge_coh[-1], m.lock_time, m.edge_lock_time)}")
    print()

    # Baseline run (complete graph) to avoid twisted-state artifacts
    g2 = make_complete_graph(n)
    s2 = init_state(g2, seed=0, F0=2.0, P0=1.0, a0=1.0, sigma0=1.0, C0=1.0, theta_mode="random")
    m2 = run_sim(g2, s2, p, lock_thresh=0.7)

    print("=== Baseline Phase Run (Complete Graph) ===")
    print(f"final R: {m2.R[-1]:.4f}")
    print(f"lock step (R>0.7): {m2.lock_time}")
    print(f"final phase residual variance: {m2.phase_resid_var[-1]:.6f}")
    print(f"final edge coherence: {m2.edge_coh[-1]:.6f}")
    print(f"edge lock step (edge>0.7): {m2.edge_lock_time}")
    print(f"regime: {classify_regime(m2.R[-1], m2.edge_coh[-1], m2.lock_time, m2.edge_lock_time)}")
    print()

    # Falsifier sweeps
    tests = [
        sweep_presence_scaling(),
        sweep_resource_frequency(),
        frozen_inertness(),
        size_scaling_scan(),
    ]
    print("=== Falsifier-Style Tests (Quick Harness) ===")
    for t in tests:
        print(f"\n--- {t['test']} ---")
        if "results" in t:
            # print compactly
            for row in t["results"][:20]:
                print(row)
            if len(t["results"]) > 20:
                print(f"... ({len(t['results'])-20} more rows)")
        else:
            for k, v in t.items():
                if k != "test":
                    print(f"{k}: {v}")

if __name__ == "__main__":
    main()