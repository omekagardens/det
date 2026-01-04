"""
DET "Speed of Light" Stability Test in Mesh Networks
===================================================

Goal
----
Show that when you impose a *coherence penalty* for messages that arrive "out of sync"
(relative to an emergent propagation speed), the network self-organizes so that the
only stable, high-coherence propagation speed is a uniform c_*.

Model sketch (software-physics analogy)
---------------------------------------
- Nodes live in 2D space; edges connect nearby nodes (a mesh).
- Each edge has a *physical* latency L_ij (what the network "can do"),
  but routing/forwarding favors *coherent* edges.
- The network maintains an emergent c_* estimate from currently-coherent edges.
- A message hop i->j is "coherent" if its arrival time matches the global expectation:
      t_arr ~ t_send + dist(i,j)/c_*
  Edges that are too fast or too slow (relative to c_*) lose coherence and stop being used.
- Over time: coherence concentrates around edges with speed near c_* and messages propagate
  along those, producing a stable, near-Lorentz-like effective propagation speed.

What to look for
----------------
1) Distribution of effective edge-speeds (dist / latency) narrows around c_*.
2) Wavefront time vs distance from source becomes linear with slope ~ 1/c_*.
3) Artificial "superluminal" shortcuts get rejected unless they match the consensus speed.

Dependencies
------------
- numpy
- scipy (cKDTree)  [optional but strongly recommended for 10k nodes]
- matplotlib (optional; for plots)

Run
---
python det_cstar_mesh_test.py

Tune
----
- N: number of nodes (10_000 recommended)
- k_neighbors: mesh connectivity
- shortcut_frac / shortcut_speedup: add some "too-fast" edges to see them get rejected
- sync_sigma: how tight the coherence window is
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# -----------------------------
# Utility
# -----------------------------

def robust_median(x: np.ndarray, eps: float = 1e-12) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    m = float(np.median(x))
    return m if abs(m) > eps else eps


def clipped_exp_quadratic(residual: np.ndarray, sigma: float) -> np.ndarray:
    # coherence ~ exp(-(res/sigma)^2) but clipped for numerical stability
    z = (residual / max(sigma, 1e-12)) ** 2
    z = np.clip(z, 0.0, 60.0)
    return np.exp(-z)


# -----------------------------
# Mesh construction
# -----------------------------

def build_geometric_mesh(
    N: int = 10_000,
    k_neighbors: int = 8,
    seed: int = 7,
    space_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pos: (N,2) float
      edges_u: (E,) int
      edges_v: (E,) int
    Undirected edges are returned as two directed entries (u->v and v->u).
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, space_size, size=(N, 2)).astype(np.float64)

    if not HAVE_SCIPY:
        # Fallback: NOT recommended for 10k (O(N^2)).
        # Use a smaller N if scipy is unavailable.
        print("WARNING: scipy not found; falling back to slow O(N^2) neighbor search.")
        d2 = (
            (pos[:, None, 0] - pos[None, :, 0]) ** 2
            + (pos[:, None, 1] - pos[None, :, 1]) ** 2
        )
        np.fill_diagonal(d2, np.inf)
        nn = np.argpartition(d2, kth=k_neighbors, axis=1)[:, :k_neighbors]
    else:
        tree = cKDTree(pos)
        # query k+1 because nearest includes self at dist=0
        _, nn = tree.query(pos, k=k_neighbors + 1, workers=-1)
        nn = nn[:, 1:]  # drop self

    # Create directed edges (u->v) for each neighbor
    u = np.repeat(np.arange(N, dtype=np.int32), k_neighbors)
    v = nn.reshape(-1).astype(np.int32)

    # Remove self-loops if any
    mask = u != v
    u = u[mask]
    v = v[mask]

    # Deduplicate directed edges (optional; helps stats)
    # We'll keep directed uniqueness.
    uv = u.astype(np.int64) * np.int64(N) + v.astype(np.int64)
    keep = np.unique(uv, return_index=True)[1]
    u = u[keep]
    v = v[keep]

    return pos, u, v


def add_shortcuts(
    pos: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    shortcut_frac: float = 0.002,      # fraction of N to add as shortcuts
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add random long-range directed shortcuts. (We’ll apply super-fast latency to them later.)
    """
    rng = np.random.default_rng(seed)
    N = pos.shape[0]
    m = int(max(0, round(shortcut_frac * N)))

    if m == 0:
        return edges_u, edges_v

    u_sc = rng.integers(0, N, size=m, dtype=np.int32)
    v_sc = rng.integers(0, N, size=m, dtype=np.int32)
    mask = u_sc != v_sc
    u_sc = u_sc[mask]
    v_sc = v_sc[mask]

    edges_u2 = np.concatenate([edges_u, u_sc], axis=0)
    edges_v2 = np.concatenate([edges_v, v_sc], axis=0)
    return edges_u2, edges_v2


# -----------------------------
# Core experiment
# -----------------------------

@dataclass
class ExperimentConfig:
    N: int = 10_000
    k_neighbors: int = 8

    # Physical latency model
    c_phys_mean: float = 1.0          # baseline physical propagation speed
    c_phys_jitter: float = 0.35       # variability in physical edge speed (heterogeneity)
    base_latency_floor: float = 1e-4  # avoids zero latency

    # Coherence model
    sync_sigma: float = 0.015         # coherence window in time units (tight -> stronger selection)
    coherence_min_forward: float = 0.08
    coherence_learn_rate: float = 0.08
    coherence_decay: float = 0.02

    # Emergent c_* estimation
    cstar_smoothing: float = 0.05     # EMA smoothing of c_*
    cstar_min: float = 0.05
    cstar_max: float = 20.0

    # Messaging / wavefront
    steps: int = 260
    pulse_every: int = 3
    max_hops: int = 28
    source: int = 0

    # Optional "superfast" shortcuts to test rejection
    shortcut_frac: float = 0.002
    shortcut_speedup: float = 12.0    # shortcuts have speed multiplied by this (too fast)


class DetCStarMeshTest:
    def __init__(self, cfg: ExperimentConfig, seed: int = 7):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        pos, eu, ev = build_geometric_mesh(cfg.N, cfg.k_neighbors, seed=seed)
        eu, ev = add_shortcuts(pos, eu, ev, shortcut_frac=cfg.shortcut_frac, seed=seed + 99)

        self.pos = pos
        self.eu = eu
        self.ev = ev
        self.E = eu.size
        self.N = pos.shape[0]

        # Edge distances
        dx = pos[ev, 0] - pos[eu, 0]
        dy = pos[ev, 1] - pos[eu, 1]
        self.dist = np.sqrt(dx * dx + dy * dy).astype(np.float64)

        # Physical edge speeds (heterogeneous)
        # lognormal-ish variability around c_phys_mean
        jitter = self.rng.normal(0.0, cfg.c_phys_jitter, size=self.E)
        c_phys = cfg.c_phys_mean * np.exp(jitter)
        c_phys = np.clip(c_phys, 0.03, 50.0)

        # Make shortcuts "too fast" by multiplying their speed
        # Identify shortcuts as those with unusually long distances (simple heuristic)
        d_med = float(np.median(self.dist))
        is_shortcut = self.dist > (3.5 * d_med)
        c_phys[is_shortcut] *= cfg.shortcut_speedup

        self.is_shortcut = is_shortcut

        self.latency = (self.dist / c_phys) + cfg.base_latency_floor  # "hardware" latency
        self.latency0 = self.latency.copy()

        # Coherence per edge (start moderately high, but not max)
        self.C = np.full(self.E, 0.6, dtype=np.float64)

        # Emergent speed estimate
        # Start from median physical speed on local edges
        initial_speed = self.dist / np.maximum(self.latency, 1e-12)
        self.c_star = float(np.median(initial_speed[~is_shortcut])) if np.any(~is_shortcut) else float(np.median(initial_speed))

        # Build adjacency: per node list of outgoing edge indices
        order = np.argsort(self.eu, kind="mergesort")
        self.eu_sorted = self.eu[order]
        self.ev_sorted = self.ev[order]
        self.dist_sorted = self.dist[order]
        self.lat_sorted = self.latency[order]
        self.lat0_sorted = self.latency0[order]
        self.C_sorted = self.C[order]
        self.is_shortcut_sorted = self.is_shortcut[order]

        # CSR-like indexing
        self.row_ptr = np.zeros(self.N + 1, dtype=np.int64)
        np.add.at(self.row_ptr, self.eu_sorted + 1, 1)
        self.row_ptr = np.cumsum(self.row_ptr)

        # Telemetry
        self.cstar_hist: List[float] = []
        self.speed_spread_hist: List[float] = []
        self.shortcut_usage_hist: List[float] = []
        self.wave_samples: List[Tuple[float, float]] = []  # (distance_from_source, arrival_time)

    def _out_edges(self, u: int) -> slice:
        a = self.row_ptr[u]
        b = self.row_ptr[u + 1]
        return slice(a, b)

    def _estimate_cstar_from_coherent_edges(self) -> float:
        # Use edges weighted by coherence to estimate c_* as robust median speed.
        speed = self.dist_sorted / np.maximum(self.lat_sorted, 1e-12)
        w = np.clip(self.C_sorted, 0.0, 1.0)

        # Keep only reasonably coherent edges to define the "active light cone"
        mask = w > 0.25
        if np.count_nonzero(mask) < 100:
            mask = w > 0.10  # relax early on
        if np.count_nonzero(mask) < 50:
            return self.c_star

        s = speed[mask]
        # Weighted median (approx): sample by probability proportional to w
        prob = w[mask] / np.sum(w[mask])
        idx = self.rng.choice(np.arange(s.size), size=min(4000, s.size), replace=True, p=prob)
        c_hat = float(np.median(s[idx]))
        return float(np.clip(c_hat, self.cfg.cstar_min, self.cfg.cstar_max))

    def _edge_update_coherence(self, residual: np.ndarray) -> None:
        """
        residual: (E,) arrival_time - expected_time
        We update coherence toward exp(-(res/sigma)^2), plus a slow decay to prevent lock-in.
        """
        cfg = self.cfg
        target = clipped_exp_quadratic(residual, cfg.sync_sigma)
        # EMA update
        self.C_sorted = (1.0 - cfg.coherence_learn_rate) * self.C_sorted + cfg.coherence_learn_rate * target
        # small background decay
        self.C_sorted *= (1.0 - cfg.coherence_decay)
        self.C_sorted = np.clip(self.C_sorted, 0.0, 1.0)

    def _maybe_adjust_latencies_toward_consensus(self, residual: np.ndarray) -> None:
        """
        Optional "network adapts" knob:
        If an edge tends to arrive early (negative residual), it can be treated as decohering and
        effectively "slowed" (increased latency) by scheduling/handshake/consensus overhead.
        If late, it can be optimized a little (decreased latency), but bounded by the physical floor.

        This models a protocol layer where coherence requirements reshape effective timing.
        """
        # Small, bounded adjustment
        gain = 0.02
        # push latency in direction that reduces residual (late -> decrease latency, early -> increase latency)
        self.lat_sorted = self.lat_sorted + gain * residual
        # Keep within reasonable bounds (can't beat the physical baseline too much)
        self.lat_sorted = np.clip(self.lat_sorted, 0.85 * self.lat0_sorted, 5.0 * self.lat0_sorted)

    def _run_pulse(self, t0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate one wave/pulse from source using coherence-weighted forwarding.

        Returns:
          arrival_time: (N,) float (inf if not reached)
          hops: (N,) int
        """
        cfg = self.cfg
        N = self.N

        arrival = np.full(N, np.inf, dtype=np.float64)
        hops = np.full(N, -1, dtype=np.int32)
        arrival[cfg.source] = t0
        hops[cfg.source] = 0

        frontier = np.array([cfg.source], dtype=np.int32)

        # Precompute distance to source for wavefront analysis
        dxs = self.pos[:, 0] - self.pos[cfg.source, 0]
        dys = self.pos[:, 1] - self.pos[cfg.source, 1]
        dist_to_source = np.sqrt(dxs * dxs + dys * dys)

        for h in range(cfg.max_hops):
            if frontier.size == 0:
                break

            next_nodes = []
            # Process each node in frontier
            for u in frontier.tolist():
                sl = self._out_edges(u)
                if sl.start == sl.stop:
                    continue

                e_idx = np.arange(sl.start, sl.stop, dtype=np.int64)
                v = self.ev_sorted[sl]
                d = self.dist_sorted[sl]
                L = self.lat_sorted[sl]
                C = self.C_sorted[sl]

                # Only forward along sufficiently coherent edges
                ok = C >= cfg.coherence_min_forward
                if not np.any(ok):
                    continue

                v_ok = v[ok]
                d_ok = d[ok]
                L_ok = L[ok]
                C_ok = C[ok]
                e_ok = e_idx[ok]

                t_send = arrival[u]
                t_arr = t_send + L_ok

                # If this arrival improves, update
                improved = t_arr < arrival[v_ok]
                if np.any(improved):
                    vv = v_ok[improved]
                    tt = t_arr[improved]
                    arrival[vv] = tt
                    hops[vv] = hops[u] + 1
                    next_nodes.append(vv)

                # Coherence penalty is based on *expected* timing from consensus c_*
                # Expected hop time uses dist/c_* (not the physical latency).
                t_expected = t_send + (d_ok / max(self.c_star, 1e-12))
                residual = t_arr - t_expected

                # Update coherence for these edges
                # (Write back into C_sorted in-place for those indices)
                target = clipped_exp_quadratic(residual, cfg.sync_sigma)
                self.C_sorted[e_ok] = (1.0 - cfg.coherence_learn_rate) * self.C_sorted[e_ok] + cfg.coherence_learn_rate * target

                # Optional: adapt effective latencies toward coherence (protocol overhead effect)
                if h % 3 == 0:
                    # push L slightly toward reducing residual
                    self.lat_sorted[e_ok] = np.clip(
                        self.lat_sorted[e_ok] + 0.02 * residual,
                        0.85 * self.lat0_sorted[e_ok],
                        5.0 * self.lat0_sorted[e_ok],
                    )

            if next_nodes:
                frontier = np.unique(np.concatenate(next_nodes).astype(np.int32))
            else:
                frontier = np.array([], dtype=np.int32)

        # Collect wavefront samples (distance vs arrival time) for reached nodes
        reached = np.isfinite(arrival)
        # Sample a subset to keep memory bounded
        idx = np.where(reached)[0]
        if idx.size > 0:
            take = min(2500, idx.size)
            sub = self.rng.choice(idx, size=take, replace=False)
            for n in sub.tolist():
                self.wave_samples.append((float(dist_to_source[n]), float(arrival[n] - t0)))

        return arrival, hops

    def step(self, k: int) -> None:
        cfg = self.cfg

        # Periodically emit a pulse to probe propagation.
        if k % cfg.pulse_every == 0:
            self._run_pulse(t0=float(k))

        # Global decay of coherence
        self.C_sorted *= (1.0 - cfg.coherence_decay)
        self.C_sorted = np.clip(self.C_sorted, 0.0, 1.0)

        # Update c_* from currently coherent edges
        c_hat = self._estimate_cstar_from_coherent_edges()
        self.c_star = (1.0 - cfg.cstar_smoothing) * self.c_star + cfg.cstar_smoothing * c_hat

        # Telemetry: how narrow is speed distribution among coherent edges?
        speed = self.dist_sorted / np.maximum(self.lat_sorted, 1e-12)
        w = np.clip(self.C_sorted, 0.0, 1.0)
        mask = w > 0.25
        if np.count_nonzero(mask) < 100:
            mask = w > 0.10

        if np.any(mask):
            s = speed[mask]
            spread = float(np.percentile(s, 90) - np.percentile(s, 10))
        else:
            spread = float(np.percentile(speed, 90) - np.percentile(speed, 10))

        # Shortcut usage proxy: fraction of coherent edges that are shortcuts
        coherent = w > 0.25
        if np.count_nonzero(coherent) > 0:
            shortcut_usage = float(np.mean(self.is_shortcut_sorted[coherent]))
        else:
            shortcut_usage = float(np.mean(self.is_shortcut_sorted[w > 0.10])) if np.any(w > 0.10) else 0.0

        self.cstar_hist.append(float(self.c_star))
        self.speed_spread_hist.append(spread)
        self.shortcut_usage_hist.append(shortcut_usage)

    def run(self) -> None:
        t_start = time.time()
        for k in range(self.cfg.steps):
            self.step(k)
            if (k + 1) % 20 == 0:
                print(
                    f"k={k+1:4d} | c*={self.c_star:8.4f} | speed_spread(p90-p10)={self.speed_spread_hist[-1]:8.4f} "
                    f"| shortcut_coherent_frac={self.shortcut_usage_hist[-1]:7.4f}"
                )
        print(f"Done in {time.time() - t_start:.2f}s")

    def summarize(self) -> dict:
        """
        Returns key metrics, including a wavefront-derived c_obs estimate.
        """
        # Fit t ≈ d / c_obs using robust median of d/t
        samples = np.array(self.wave_samples, dtype=np.float64)
        if samples.size == 0:
            c_obs = float("nan")
        else:
            d = samples[:, 0]
            t = samples[:, 1]
            mask = (t > 1e-6) & np.isfinite(t) & np.isfinite(d)
            if np.count_nonzero(mask) < 100:
                c_obs = float("nan")
            else:
                c_obs = robust_median(d[mask] / t[mask])

        return {
            "c_star_final": float(self.c_star),
            "c_star_median": float(np.median(np.array(self.cstar_hist))),
            "c_obs_from_wavefront": float(c_obs),
            "speed_spread_final_p90_p10": float(self.speed_spread_hist[-1]) if self.speed_spread_hist else float("nan"),
            "shortcut_coherent_frac_final": float(self.shortcut_usage_hist[-1]) if self.shortcut_usage_hist else float("nan"),
            "num_wave_samples": int(len(self.wave_samples)),
            "N": int(self.N),
            "E_directed": int(self.E),
        }


def main():
    cfg = ExperimentConfig(
        N=10_000,
        k_neighbors=8,
        steps=260,
        pulse_every=3,
        sync_sigma=0.015,
        shortcut_frac=0.002,
        shortcut_speedup=12.0,
    )

    exp = DetCStarMeshTest(cfg, seed=7)
    exp.run()
    out = exp.summarize()

    print("\nSummary")
    for k, v in out.items():
        print(f"  {k:28s}: {v}")

    # Optional: plots
    try:
        import matplotlib.pyplot as plt  # type: ignore

        cstar = np.array(exp.cstar_hist)
        spread = np.array(exp.speed_spread_hist)
        sh = np.array(exp.shortcut_usage_hist)

        plt.figure()
        plt.plot(cstar)
        plt.xlabel("step")
        plt.ylabel("c_* estimate")
        plt.title("Emergent c_* over time")

        plt.figure()
        plt.plot(spread)
        plt.xlabel("step")
        plt.ylabel("p90(speed)-p10(speed) among coherent edges")
        plt.title("Speed distribution tightening")

        plt.figure()
        plt.plot(sh)
        plt.xlabel("step")
        plt.ylabel("fraction of coherent edges that are shortcuts")
        plt.title("Are superfast shortcuts rejected? (should drop)")

        # wavefront scatter (d vs t), optionally with best-fit line
        if len(exp.wave_samples) > 500:
            samp = np.array(exp.wave_samples, dtype=np.float64)
            d = samp[:, 0]
            t = samp[:, 1]
            mask = (t > 1e-6) & np.isfinite(t) & np.isfinite(d)
            d = d[mask]
            t = t[mask]
            # subsample for visualization
            rng = np.random.default_rng(0)
            idx = rng.choice(np.arange(d.size), size=min(4000, d.size), replace=False)
            d2 = d[idx]
            t2 = t[idx]
            plt.figure()
            plt.scatter(d2, t2, s=2, alpha=0.35)
            plt.xlabel("distance from source")
            plt.ylabel("arrival time (relative)")
            plt.title("Wavefront: t vs d (should be ~linear)")

        plt.show()

    except Exception as e:
        print(f"\n(matplotlib plotting skipped: {e})")


if __name__ == "__main__":
    main()