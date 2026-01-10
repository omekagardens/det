# soliton_falsifiers.py
# DET v5 Soliton Falsifier Suite
#
# Core claims tested:
#  - Directed space (C_{i->j} != C_{j->i}) can emerge and sustain motion (soliton)
#  - Phase-energy relation: dtheta/dt = -beta * F
#  - Flow-structure loop: Hebbian growth strengthens forward bonds, decay prunes reverse bonds
#  - Inertia: anisotropy resists direction reversal
#  - Speed limit: terminal soliton speed depends on (alpha, beta, lambda)
#  - Vacuum required: without F_vac, gravity/drag channel collapses (no sustained motion)

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

TAU = 2 * np.pi


def wrap_2pi(x: np.ndarray) -> np.ndarray:
    return np.mod(x, TAU)


def circ_sin(d: np.ndarray) -> np.ndarray:
    # d can be any real; sin works fine
    return np.sin(d)


def center_of_mass(F: np.ndarray, F_vac: float = 0.0) -> float:
    """Center of mass of *excess* resource above vacuum baseline."""
    Fx = np.maximum(F - F_vac, 0.0)
    total = float(np.sum(Fx))
    if total <= 1e-12:
        return float("nan")
    idx = np.arange(F.size, dtype=float)
    return float(np.sum(idx * Fx) / total)



def packet_width(F: np.ndarray, com: float, F_vac: float = 0.0) -> float:
    """RMS width of *excess* resource above vacuum baseline."""
    Fx = np.maximum(F - F_vac, 0.0)
    total = float(np.sum(Fx))
    if total <= 1e-12 or not np.isfinite(com):
        return float("nan")
    idx = np.arange(F.size, dtype=float)
    var = float(np.sum(Fx * (idx - com) ** 2) / total)
    return float(np.sqrt(max(0.0, var)))


# Robust containment metric: smallest symmetric radius about COM containing frac of excess mass.
def containment_radius(F: np.ndarray, com: float, frac: float = 0.80, F_vac: float = 0.0) -> float:
    """Smallest symmetric radius r (in node units) about COM containing `frac` of excess mass.

    Uses excess resource Fx = max(F - F_vac, 0). If no excess mass, returns nan.
    The interval is centered at COM and expanded in integer steps.
    """
    Fx = np.maximum(F - F_vac, 0.0)
    total = float(np.sum(Fx))
    if total <= 1e-12 or (not np.isfinite(com)):
        return float("nan")

    n = Fx.size
    # Use nearest integer center for discrete containment
    c = int(np.clip(int(round(com)), 0, n - 1))
    target = frac * total

    mass = float(Fx[c])
    r = 0
    left = c
    right = c
    while mass < target and (left > 0 or right < n - 1):
        r += 1
        left = max(0, c - r)
        right = min(n - 1, c + r)
        mass = float(np.sum(Fx[left:right + 1]))

    return float(r)

def g_focus(x: np.ndarray, x_star: float) -> np.ndarray:
    """Saturating gain function in [0,1)."""
    xs = np.maximum(x, 0.0)
    return xs / (xs + float(x_star) + 1e-12)

def anisotropy_index(Cf: np.ndarray, Cb: np.ndarray, eps: float = 1e-9) -> float:
    # Measures directedness: (sum forward - sum backward)/(sum forward + sum backward)
    sf = float(np.sum(Cf))
    sb = float(np.sum(Cb))
    return (sf - sb) / (sf + sb + eps)


@dataclass
class Params:
    n: int = 80
    dt: float = 0.05
    steps: int = 2000

    beta: float = 0.45         # phase-energy coupling (like 1/hbar)
    kappa_theta: float = 2.6   # local phase coupling strength (sustains coherent gradients)
    alpha: float = 1.0         # hebbian learning rate
    lam: float = 0.01          # metric decay
    sigma0: float = 3.0        # baseline conductivity
    F_vac: float = 0.10        # vacuum resource floor

    # Transport mix: quantum-like vs classical-like (kept simple but matches your form)
    # Transport mix: 0 => purely phase/quantum term, 1 => include full classical pressure term
    eta_classical: float = 0.0

    # Optional drive saturation (anti-tail). If >0, raw drives are squashed via tanh(raw/sat)
    drive_saturation: float = 0.25

    # Optional self-focusing conductivity: sigma_edge = sigma0 * (1 + kappa * g(F_excess_edge))
    focus_kappa: float = 0.3
    focus_Fstar: float = 0.6

    # Separate scale for *learning* gate (bigger => stricter; tails don't pave roads)
    learn_Fstar: float = 2.0

    # Rectification: diode gating on raw_drive sign
    rectify: bool = True

    # Stability
    C_min: float = 0.0
    C_max: float = 1.0


@dataclass
class Observables:
    com: List[float]
    width: List[float]
    r80: List[float]
    r95: List[float]
    speed: List[float]          # finite difference of COM
    anisotropy: List[float]
    total_F: List[float]
    forward_C_mean: List[float]
    backward_C_mean: List[float]


class DETv5SolitonSim:
    """
    1D chain with directed bonds:
      C_f[i] = C_{i -> i+1}
      C_b[i] = C_{i+1 -> i}
    """
    def __init__(self, p: Params, seed: int = 0):
        self.p = p
        self.rng = np.random.default_rng(seed)

        n = p.n
        self.F = np.full(n, p.F_vac, dtype=float)  # vacuum baseline
        self.theta = self.rng.uniform(0, TAU, size=n).astype(float)

        # directed edge arrays of length n-1
        self.Cf = np.full(n - 1, 0.05, dtype=float)
        self.Cb = np.full(n - 1, 0.05, dtype=float)

        # sigma per node or per bond; keep minimal (constant here, suite can extend)
        self.sigma = np.full(n, p.sigma0, dtype=float)

    def inject_packet(self, idx: int, amount: float, phase_gradient: float = np.pi / 2):
        # Add a localized resource packet and give a directional phase kick (rightward)
        self.F[idx] += amount
        if idx + 1 < self.p.n:
            self.theta[idx] = 0.0
            self.theta[idx + 1] = phase_gradient
        if idx + 2 < self.p.n:
            self.theta[idx + 2] = 2.0 * phase_gradient

    def seed_diode_bias(self, idx: int, forward: float = 0.9, backward: float = 0.01, length: int = 6):
        """Create an initial directed 'road' segment to seed momentum."""
        n = self.p.n
        L = int(max(1, length))
        for k in range(L):
            e = idx + k
            if 0 <= e < n - 1:
                f = forward * (0.85 ** k)
                self.Cf[e] = np.clip(f, self.p.C_min, self.p.C_max)
                self.Cb[e] = np.clip(backward, self.p.C_min, self.p.C_max)

    def _edge_drive(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute directed fluxes on each undirected edge (i, i+1):
          Jf[i] = flux i -> i+1
          Jb[i] = flux i+1 -> i

        Uses your proposed mixed drive:
          raw_drive = sqrt(C_rev)*sin(dtheta) + (1-sqrt(C_rev))*(Fi-Fj)
        with optional rectification via choosing conductance based on sign.
        """
        p = self.p
        Fi = self.F[:-1]
        Fj = self.F[1:]
        ti = self.theta[:-1]
        tj = self.theta[1:]

        dtheta = tj - ti
        # reverse conductance used inside the bracket (matches your structure)
        # for forward i->i+1, reverse is Cb[i] (i+1 -> i)
        sqrt_rev_f = np.sqrt(np.clip(self.Cb, 0.0, 1.0))
        sqrt_rev_b = np.sqrt(np.clip(self.Cf, 0.0, 1.0))

        q_term_f = sqrt_rev_f * circ_sin(dtheta)
        q_term_b = sqrt_rev_b * circ_sin(-dtheta)

        c_term_f = (1.0 - sqrt_rev_f) * (Fi - Fj)
        c_term_b = (1.0 - sqrt_rev_b) * (Fj - Fi)

        # Suppress classical diffusion inside dense packets: allow it mainly in vacuum.
        Fx_i = np.maximum(Fi - p.F_vac, 0.0)
        Fx_j = np.maximum(Fj - p.F_vac, 0.0)
        Fx_e = 0.5 * (Fx_i + Fx_j)
        vacuum_gate = 1.0 - g_focus(Fx_e, p.focus_Fstar)  # ~1 in vacuum, ~0 in dense packet

        eta = float(p.eta_classical)
        raw_f = q_term_f + eta * vacuum_gate * c_term_f
        raw_b = q_term_b + eta * vacuum_gate * c_term_b

        # Optional saturation to suppress low-level tails from accumulating
        sat = float(p.drive_saturation)
        if sat > 0.0:
            raw_f = sat * np.tanh(raw_f / sat)
            raw_b = sat * np.tanh(raw_b / sat)

        # Conductance selection / rectification
        # If rectify: allow only positive raw drive in that direction; otherwise block.
        # Also encode diode nature by using the directed C in that direction.
        sigma_e = self.sigma0_edge()
        if p.rectify:
            Jf = sigma_e * self.Cf * np.maximum(0.0, raw_f)
            Jb = sigma_e * self.Cb * np.maximum(0.0, raw_b)
        else:
            Jf = sigma_e * self.Cf * raw_f
            Jb = sigma_e * self.Cb * raw_b

        return Jf, Jb, raw_f

    def sigma0_edge(self) -> np.ndarray:
        """Per-edge conductivity with optional self-focusing on excess resource."""
        p = self.p
        Fx = np.maximum(self.F - p.F_vac, 0.0)          # excess per node
        Fx_e = 0.5 * (Fx[:-1] + Fx[1:])                 # excess per edge
        gain = 1.0 + float(p.focus_kappa) * g_focus(Fx_e, p.focus_Fstar)
        return float(p.sigma0) * gain



    def step(self):
        p = self.p

        # (I) Phase-energy relation + local phase coupling
        theta = self.theta
        theta = theta + (-p.beta * self.F) * p.dt

        if p.kappa_theta != 0.0:
            k = float(p.kappa_theta)
            # neighbor weights (use sqrt of directed conductances)
            wf = np.sqrt(np.clip(self.Cf, 0.0, 1.0))  # i -> i+1
            wb = np.sqrt(np.clip(self.Cb, 0.0, 1.0))  # i+1 -> i

            dtheta_f = theta[1:] - theta[:-1]
            # coupling contribution per node
            coup = np.zeros_like(theta)
            # from i+1 to i uses wb[i]
            coup[:-1] += wb * np.sin(dtheta_f)
            # from i to i+1 uses wf[i]
            coup[1:]  += wf * np.sin(-dtheta_f)
            theta = theta + k * coup * p.dt

        self.theta = wrap_2pi(theta)

        # (II) Transport (CONSERVATIVE on excess Fx = F - F_vac)
        Jf, Jb, _raw_f = self._edge_drive()

        # Available excess at each node
        Fx = np.maximum(self.F - p.F_vac, 0.0)
        Fx_i = Fx[:-1]
        Fx_j = Fx[1:]

        # Cap outgoing flux by available excess per timestep
        # Forward flux draws from i, backward flux draws from i+1
        cap_f = Fx_i / (p.dt + 1e-12)
        cap_b = Fx_j / (p.dt + 1e-12)
        Jf = np.minimum(Jf, cap_f)
        Jb = np.minimum(Jb, cap_b)

        # Net change on excess (mass-conserving aside from boundary conditions)
        netFx = np.zeros_like(Fx)
        netFx[:-1] += (-Jf + Jb)
        netFx[1:]  += ( Jf - Jb)

        Fx = Fx + netFx * p.dt
        Fx = np.maximum(Fx, 0.0)
        self.F = p.F_vac + Fx

        # (III) Hebbian metric update (directed)
        # Core-gated learning: tails (low excess) should not build strong roads.
        Fx_edge = 0.5 * (Fx[:-1] + Fx[1:])  # excess on each edge (same indexing as Cf/Cb)
        learn_gate = g_focus(Fx_edge, p.learn_Fstar)  # ~0 in vacuum/tail, ~1 in core

        # strengthen only used direction (and only where gate is high); decay always
        self.Cf = self.Cf + (p.alpha * learn_gate * np.maximum(0.0, Jf) - p.lam * self.Cf) * p.dt
        self.Cb = self.Cb + (p.alpha * learn_gate * np.maximum(0.0, Jb) - p.lam * self.Cb) * p.dt
        self.Cf = np.clip(self.Cf, p.C_min, p.C_max)
        self.Cb = np.clip(self.Cb, p.C_min, p.C_max)

    def run(self, record_every: int = 1) -> Observables:
        com_hist: List[float] = []
        width_hist: List[float] = []
        r80_hist: List[float] = []
        r95_hist: List[float] = []
        speed_hist: List[float] = []
        aniso_hist: List[float] = []
        totalF_hist: List[float] = []
        Cf_mean: List[float] = []
        Cb_mean: List[float] = []

        prev_com: Optional[float] = None

        for t in range(self.p.steps):
            self.step()

            if t % record_every == 0:
                com = center_of_mass(self.F, F_vac=self.p.F_vac)
                w = packet_width(self.F, com, F_vac=self.p.F_vac)
                r80 = containment_radius(self.F, com, frac=0.80, F_vac=self.p.F_vac)
                r95 = containment_radius(self.F, com, frac=0.95, F_vac=self.p.F_vac)
                if prev_com is None or not np.isfinite(prev_com) or not np.isfinite(com):
                    v = float("nan")
                else:
                    v = (com - prev_com) / (record_every * self.p.dt)

                prev_com = com

                com_hist.append(com)
                width_hist.append(w)
                r80_hist.append(r80)
                r95_hist.append(r95)
                speed_hist.append(v)
                aniso_hist.append(anisotropy_index(self.Cf, self.Cb))
                totalF_hist.append(float(np.sum(self.F)))
                Cf_mean.append(float(np.mean(self.Cf)))
                Cb_mean.append(float(np.mean(self.Cb)))

        return Observables(
            com=com_hist,
            width=width_hist,
            r80=r80_hist,
            r95=r95_hist,
            speed=speed_hist,
            anisotropy=aniso_hist,
            total_F=totalF_hist,
            forward_C_mean=Cf_mean,
            backward_C_mean=Cb_mean,
        )


# ---------------------------
# FALSIFIER TESTS
# ---------------------------

def _finite_mean(xs: List[float]) -> float:
    arr = np.array(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _finite_median(xs: List[float]) -> float:
    arr = np.array(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def test_F0_soliton_persistence(p: Params) -> Dict[str, object]:
    """
    Claim: a localized packet forms a moving, shape-stable soliton (COM drifts, width bounded).
    Falsifier: either it stalls (no COM drift) or disperses (width grows without bound).
    """
    sim = DETv5SolitonSim(p, seed=1)
    idx0 = 10
    sim.inject_packet(idx0, amount=10.0)
    sim.seed_diode_bias(idx0, forward=0.96, backward=0.008, length=8)

    obs = sim.run(record_every=5)
    com0 = obs.com[0]
    comN = obs.com[-1]
    drift = comN - com0

    w_med = _finite_median(obs.width[len(obs.width)//3:])
    v_med = _finite_median(obs.speed[len(obs.speed)//3:])
    r80_med = _finite_median(obs.r80[len(obs.r80)//3:])

    # thresholds (tuneable but falsifier-like):
    pass_motion = np.isfinite(drift) and abs(drift) > 5.0
    # Soliton criterion: most excess mass stays confined (robust to small tails)
    pass_shape = np.isfinite(r80_med) and r80_med <= 6.0
    pass_speed = np.isfinite(v_med) and (abs(v_med) > 0.015)

    return {
        "test": "F0_soliton_persistence",
        "drift_nodes": drift,
        "median_speed": v_med,
        "median_width": w_med,
        "median_r80": r80_med,
        "pass_motion": pass_motion,
        "pass_shape": pass_shape,
        "pass_speed": pass_speed,
        "PASS": bool(pass_motion and pass_shape and pass_speed),
    }


def test_F1_no_diode_no_motion(p: Params) -> Dict[str, object]:
    """
    Claim: inertia/momentum arises from learned directedness, not from node momentum.
    Falsifier: if you remove directedness/rectification, packet should not sustain directed motion.
    """
    p2 = Params(**{**p.__dict__})
    p2.rectify = False  # remove diode gating
    p2.alpha = 0.0      # disable metric learning (no road-building)
    p2.focus_kappa = 0.0
    p2.kappa_theta = 0.0
    sim = DETv5SolitonSim(p2, seed=2)
    # Force symmetric space (no directedness): C_{i->i+1} == C_{i+1->i}
    sim.Cb[:] = sim.Cf[:]
    idx0 = 10
    # No directional phase kick in counterfactual
    sim.inject_packet(idx0, amount=10.0, phase_gradient=0.0)
    # no diode bias seeded

    # Remove any accidental global phase bias; counterfactual should be phase-flat
    sim.theta[:] = 0.0

    obs = sim.run(record_every=5)
    drift = obs.com[-1] - obs.com[0]
    v_med = _finite_median(obs.speed[len(obs.speed)//3:])
    a_med = _finite_median(obs.anisotropy[len(obs.anisotropy)//2:])

    # Expect small drift / no sustained velocity AND no emergent directedness.
    PASS = ((not np.isfinite(v_med)) or abs(v_med) < 0.02 or abs(drift) < 3.0) and (abs(a_med) < 0.05)
    return {
        "test": "F1_no_diode_no_motion",
        "drift_nodes": drift,
        "median_speed": v_med,
        "median_anisotropy": a_med,
        "expectation": "Without rectification, sustained directed soliton motion should largely vanish and no directedness should emerge.",
        "PASS": bool(PASS),
    }


def test_F2_inertia_resists_reversal(p: Params) -> Dict[str, object]:
    """
    Claim: mass/inertia is structural anisotropy; reversing motion requires undoing learned C.
    Procedure: build a moving soliton, then apply an opposite phase kick; measure reversal delay.
    Falsifier: if it reverses instantly like a frictionless oscillator, anisotropic inertia claim fails.
    """
    sim = DETv5SolitonSim(p, seed=3)
    idx0 = 10
    sim.inject_packet(idx0, amount=10.0)
    sim.seed_diode_bias(idx0, forward=0.9, backward=0.01, length=6)

    # Run to build the road
    sim.p.steps = max(600, p.steps // 3)
    obs1 = sim.run(record_every=5)
    v1 = _finite_median(obs1.speed[len(obs1.speed)//2:])

    # Apply reversal attempt: flip local phase gradient by swapping a few phases
    mid = int(np.clip(center_of_mass(sim.F), 2, sim.p.n - 3))
    sim.theta[mid-1:mid+2] = wrap_2pi(sim.theta[mid-1:mid+2] + np.pi)  # crude opposite kick

    # Continue run
    sim.p.steps = max(800, p.steps // 2)
    obs2 = sim.run(record_every=5)
    v2_series = np.array(obs2.speed, dtype=float)
    v2_series = v2_series[np.isfinite(v2_series)]

    # Define reversal as sustained negative velocity for a window
    reversed_now = False
    if v2_series.size > 50:
        # sustained negative for 30 samples
        for t in range(1, v2_series.size - 30):
            if np.all(v2_series[t:t+30] < -0.01):
                reversed_now = True
                reversal_idx = t
                break
        else:
            reversal_idx = None
    else:
        reversal_idx = None

    # Inertia claim: reversal should be delayed or fail within horizon.
    PASS = (reversal_idx is None) or (reversal_idx > 60)  # >60*5*dt time units
    return {
        "test": "F2_inertia_resists_reversal",
        "median_speed_before": v1,
        "reversal_detected": bool(reversed_now),
        "reversal_index_samples": reversal_idx,
        "PASS": bool(PASS),
    }


def test_F3_speed_limit_scaling(p: Params) -> Dict[str, object]:
    """
    Claim: there is a terminal/saturation soliton speed c = f(alpha, beta, lambda).
    Procedure: scan beta and observe median speed saturates (sublinear growth).
    Falsifier: speed grows unbounded ~beta without saturation.
    """
    betas = [0.05, 0.1, 0.2, 0.4, 0.8]
    speeds = []
    for b in betas:
        p2 = Params(**{**p.__dict__})
        p2.beta = b
        p2.steps = 1600

        v_terms: List[float] = []
        for sd in (10, 11, 12):
            sim = DETv5SolitonSim(p2, seed=sd)
            sim.inject_packet(10, amount=10.0)
            sim.seed_diode_bias(10, forward=0.9, backward=0.01, length=6)
            obs = sim.run(record_every=5)
            v_series = np.array(obs.speed, dtype=float)
            v_series = v_series[np.isfinite(v_series)]
            if v_series.size:
                late = v_series[int(0.7 * v_series.size):]
                v_terms.append(float(np.median(np.abs(late))))
        speeds.append(float(np.median(np.array(v_terms))) if v_terms else float("nan"))

    # Saturation heuristic: later increments diminish
    # Use first positive increment as dv1 to avoid negative early delta
    diffs = [speeds[i+1] - speeds[i] for i in range(len(speeds)-1)]
    diffs = [d for d in diffs if np.isfinite(d)]
    pos_diffs = [d for d in diffs if d > 0]
    dv1 = pos_diffs[0] if pos_diffs else float("nan")
    dv_last = diffs[-1] if diffs else float("nan")

    # Require overall increase and diminishing returns at the top end
    PASS = (
        np.isfinite(speeds[0]) and np.isfinite(speeds[-1]) and speeds[-1] > speeds[0]
        and np.isfinite(dv1) and np.isfinite(dv_last) and (dv_last < 0.7 * dv1)
    )

    return {
        "test": "F3_speed_limit_scaling",
        "betas": betas,
        "median_speeds": speeds,
        "delta_v_early": dv1,
        "delta_v_late": dv_last,
        "expectation": "Speed increases with beta but saturates (diminishing returns).",
        "PASS": bool(PASS),
    }


def test_F4_vacuum_required(p: Params) -> Dict[str, object]:
    """
    Claim: vacuum resource (F_vac > 0) is required as a medium for phase drag / gravity.
    Procedure: set F_vac=0 and see if soliton motion collapses or becomes unstable/nonphysical.
    Falsifier: motion remains equally robust with F_vac=0.
    """
    p2 = Params(**{**p.__dict__})
    p2.F_vac = 0.0
    p2.focus_kappa = 0.0
    sim = DETv5SolitonSim(p2, seed=5)
    sim.inject_packet(10, amount=10.0)
    sim.seed_diode_bias(10, forward=0.9, backward=0.01, length=6)

    obs = sim.run(record_every=5)
    drift = obs.com[-1] - obs.com[0]
    w_med = _finite_median(obs.width[len(obs.width)//2:])
    r80_med = _finite_median(obs.r80[len(obs.r80)//2:])

    # Expectation: without vacuum floor, dynamics should degrade (stall or disperse or hit zeros).
    # We'll call "robust motion" if drift>5 and r80<=6; falsifier expects NOT robust.
    robust = np.isfinite(drift) and drift > 5.0 and np.isfinite(r80_med) and r80_med <= 6.0
    PASS = not robust

    return {
        "test": "F4_vacuum_required",
        "drift_nodes": drift,
        "median_width": w_med,
        "median_r80": r80_med,
        "robust_motion_detected": bool(robust),
        "PASS": bool(PASS),
    }


def test_F5_metric_decay_required(p: Params) -> Dict[str, object]:
    """
    Claim: decay term (-lambda C) is required; otherwise you get permanent roads / absolute frames.
    Procedure: compare mean anisotropy growth with lambda=0 vs lambda>0.
    Falsifier: with lambda=0, anisotropy does NOT diverge / roads don't persist.
    """
    def run_case(lam: float) -> Tuple[float, float]:
        p2 = Params(**{**p.__dict__})
        p2.lam = lam
        p2.steps = 2400
        p2.C_max = 0.7

        # Put F5 in a clear 'road-building' regime so decay vs no-decay becomes measurable.
        p2.drive_saturation = 0.0
        p2.focus_kappa = 0.0
        p2.focus_Fstar = 1.0
        p2.kappa_theta = 0.0
        p2.beta = 0.25
        p2.alpha = 2.0

        sim = DETv5SolitonSim(p2, seed=7)
        sim.inject_packet(10, amount=10.0)
        sim.seed_diode_bias(10, forward=0.98, backward=0.001, length=12)
        obs = sim.run(record_every=5)
        # late anisotropy and mean C levels
        a_late = float(np.median(np.array(obs.anisotropy)[len(obs.anisotropy)//2:]))
        Cf_late = float(np.median(np.array(obs.forward_C_mean)[len(obs.forward_C_mean)//2:]))
        return a_late, Cf_late

    a_decay, Cf_decay = run_case(lam=p.lam)
    a_nodecay, Cf_nodecay = run_case(lam=0.0)

    # Expect nodecay builds stronger, more permanent anisotropy/roads
    PASS = (abs(a_nodecay) > abs(a_decay) + 0.05) or (Cf_nodecay > Cf_decay + 0.05)

    return {
        "test": "F5_metric_decay_required",
        "anisotropy_with_decay": a_decay,
        "anisotropy_no_decay": a_nodecay,
        "Cf_mean_with_decay": Cf_decay,
        "Cf_mean_no_decay": Cf_nodecay,
        "expectation": "Removing decay should produce stronger persistent anisotropy/roads (bad physically).",
        "PASS": bool(PASS),
    }


def run_all(p: Params) -> List[Dict[str, object]]:
    tests = [
        test_F0_soliton_persistence,
        test_F1_no_diode_no_motion,
        test_F2_inertia_resists_reversal,
        test_F3_speed_limit_scaling,
        test_F4_vacuum_required,
        test_F5_metric_decay_required,
    ]
    out = []
    for fn in tests:
        try:
            out.append(fn(p))
        except Exception as e:
            out.append({"test": fn.__name__, "ERROR": repr(e), "PASS": False})
    return out


def print_report(results: List[Dict[str, object]]) -> None:
    print("=== DET v5 Soliton Falsifier Suite ===\n")
    passed = 0
    for r in results:
        name = r.get("test", "UNKNOWN")
        ok = bool(r.get("PASS", False))
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        # print compact key metrics
        keys = [k for k in r.keys() if k not in ("test", "PASS")]
        for k in keys:
            v = r[k]
            if isinstance(v, float):
                print(f"  - {k}: {v:.6g}")
            else:
                print(f"  - {k}: {v}")
        print()
    print(f"Summary: {passed}/{len(results)} tests PASS")


if __name__ == "__main__":
    # Default parameters (tuned to show soliton behavior, similar to your gem5-like setup)
    p = Params(
        n=80,
        dt=0.05,
        steps=2200,
        beta=0.45,
        kappa_theta=2.6,
        alpha=1.0,
        lam=0.01,
        sigma0=6.0,
        F_vac=0.10,
        rectify=True,
        eta_classical=0.0,
        drive_saturation=0.20,
        focus_kappa=0.05,
        focus_Fstar=1.0,
        learn_Fstar=10.0,
    )
    results = run_all(p)
    print_report(results)