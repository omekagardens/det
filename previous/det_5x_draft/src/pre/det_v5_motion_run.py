import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import label

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple


"""
DET v5 COLLIDER - SINGLE-RUN TIME DELAY
=======================================
Fix: Use early-time velocity from SAME run as ghost reference.

Method:
1. Run collision simulation
2. Measure v_in from early trajectory (before collision)
3. Ghost = linear extrapolation of v_in
4. Δt = actual detector crossing time - ghost detector crossing time
5. Δx = actual final position - ghost final position

This avoids the reference/collision mismatch problem.
"""


# --- Multi-trial config and helpers ---
@dataclass
class TrialConfig:
    trials: int = 10
    seed: int = 123
    theta_noise_std: float = 1e-3   # small phase noise
    width_jitter_std: float = 0.05  # relative jitter (fraction of width)
    mass_jitter_std: float = 0.02   # relative jitter (fraction of mass)
    post_dt: float = 5.0            # time after collision to measure early-out shift


def apply_small_perturbations(sim: 'ActiveManifold1D', rng: np.random.Generator,
                              theta_noise_std: float, width_jitter: float = 0.0,
                              mass_jitter_L: float = 0.0, mass_jitter_R: float = 0.0) -> None:
    """Apply tiny perturbations to break numerical degeneracy across trials.

    Notes:
    - We avoid touching agency-like concepts; this is purely measurement noise / prep jitter.
    - We only perturb theta slightly (and optionally masses/widths via caller).
    """
    if theta_noise_std > 0:
        sim.theta += rng.normal(0.0, theta_noise_std, size=sim.N)
        sim.theta = np.mod(sim.theta, 2*np.pi)

class DETConstants:
    def __init__(self, g=1.0):
        self.g = g
        self.BETA = 2.0 * g
        self.NU = 0.5
        self.PHASE_DRAG = 0.2
        self.ALPHA = 0.5 * g
        self.GAMMA = 8.0
        self.LAMBDA = 0.005
        self.K_FUSION = 80.0
        self.F_VAC = 0.01  
        self.C_MIN = 0.05 
        self.DT = 0.05

class ActiveManifold1D:
    def __init__(self, size=200, constants=DETConstants()):
        self.N = size
        self.k = constants
        
        self.F = np.ones(self.N) * self.k.F_VAC
        self.theta = np.zeros(self.N)
        self.sigma = np.ones(self.N)
        self.C_right = np.ones(self.N) * self.k.C_MIN
        self.C_left  = np.ones(self.N) * self.k.C_MIN
        
        self.history_F = []
        self.time = 0.0
        self.start_pos_L = 0
        self.start_pos_R = 0
        
        self.track_L = []
        self.track_R = []
        self.last_pos_L = None
        self.last_pos_R = None

    def initialize_soliton(self, position, width, mass, velocity_kick=0.0):
        x = np.arange(self.N)
        dx = x - position
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        
        abs_dx = np.abs(dx)
        envelope = np.exp(-0.5 * (abs_dx / width)**2)
        
        self.F += mass * envelope
        self.theta += velocity_kick * dx * envelope
        
        if velocity_kick > 0:
            self.C_right += 0.9 * envelope
            self.C_left  += 0.3 * envelope
        elif velocity_kick < 0:
            self.C_right += 0.3 * envelope
            self.C_left  += 0.9 * envelope
        else:
            self.C_right += 0.6 * envelope
            self.C_left  += 0.6 * envelope
            
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)

    def initialize_collider_event(self, offset=40, width=5.0, mass_L=10.0, mass_R=10.0, v_kick=2.0):
        center = self.N // 2
        self.start_pos_L = center - offset
        self.start_pos_R = center + offset
        self.mass_L = mass_L
        self.mass_R = mass_R
        
        self.initialize_soliton(self.start_pos_L, width, mass_L, velocity_kick=v_kick)
        self.initialize_soliton(self.start_pos_R, width, mass_R, velocity_kick=-v_kick)
        
        self.last_pos_L = float(self.start_pos_L)
        self.last_pos_R = float(self.start_pos_R)
        self.track_L = [(0.0, self.last_pos_L)]
        self.track_R = [(0.0, self.last_pos_R)]

    def find_blob_coms(self):
        mass_dist = self.F - self.k.F_VAC
        mass_dist = np.maximum(mass_dist, 0)
        
        threshold = np.max(mass_dist) * 0.15
        binary = (mass_dist > threshold).astype(int)
        
        labeled, n_features = label(binary)
        
        blobs = []
        for i in range(1, n_features + 1):
            mask = (labeled == i)
            blob_mass = np.sum(mass_dist[mask])
            if blob_mass > 0.5:
                indices = np.where(mask)[0]
                masses = mass_dist[mask]
                com = np.sum(indices * masses) / np.sum(masses)
                blobs.append((com, blob_mass))
        
        return blobs

    def unwrap_position(self, new_pos, last_pos):
        if last_pos is None:
            return new_pos
        
        delta = new_pos - (last_pos % self.N)
        
        if delta > self.N / 2:
            delta -= self.N
        elif delta < -self.N / 2:
            delta += self.N
        
        return last_pos + delta

    def update_tracking(self):
        blobs = self.find_blob_coms()
        
        if len(blobs) == 0:
            return
        
        def ring_dist(a, b):
            d = abs((a % self.N) - (b % self.N))
            return min(d, self.N - d)
        
        if len(blobs) == 1:
            com = blobs[0][0]
            new_L = self.unwrap_position(com, self.last_pos_L)
            new_R = self.unwrap_position(com, self.last_pos_R)
            self.track_L.append((self.time, new_L))
            self.track_R.append((self.time, new_R))
            self.last_pos_L = new_L
            self.last_pos_R = new_R
            return
        
        best_L_idx = min(range(len(blobs)), 
                        key=lambda i: ring_dist(blobs[i][0], self.last_pos_L % self.N))
        
        remaining = [i for i in range(len(blobs)) if i != best_L_idx]
        if remaining:
            best_R_idx = min(remaining,
                            key=lambda i: ring_dist(blobs[i][0], self.last_pos_R % self.N))
        else:
            best_R_idx = best_L_idx
        
        new_L = self.unwrap_position(blobs[best_L_idx][0], self.last_pos_L)
        new_R = self.unwrap_position(blobs[best_R_idx][0], self.last_pos_R)
        
        self.track_L.append((self.time, new_L))
        self.track_R.append((self.time, new_R))
        self.last_pos_L = new_L
        self.last_pos_R = new_R

    def step(self):
        idx = np.arange(self.N)
        ip1 = (idx + 1) % self.N
        im1 = (idx - 1) % self.N
        
        d_fwd = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[idx])))
        d_bwd = np.angle(np.exp(1j * (self.theta[im1] - self.theta[idx])))
        laplacian_theta = d_fwd + d_bwd
        grad_theta = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[im1]))) / 2.0
        
        phase_drive = np.minimum(self.k.BETA * self.F * self.k.DT, np.pi/4.0)
        drag_factor = 1.0 / (1.0 + self.k.PHASE_DRAG * (grad_theta**2))
        
        self.theta -= phase_drive * drag_factor
        self.theta += self.k.NU * laplacian_theta * self.k.DT
        self.theta = np.mod(self.theta, 2*np.pi)
        
        recip_R = np.sqrt(self.C_left[ip1])
        recip_L = np.sqrt(self.C_right[im1])
        
        drive_R = recip_R * np.sin(self.theta[ip1]-self.theta[idx]) + (1-recip_R)*(self.F[idx]-self.F[ip1])
        drive_L = recip_L * np.sin(self.theta[im1]-self.theta[idx]) + (1-recip_L)*(self.F[idx]-self.F[im1])
        
        cond_R = self.sigma[idx] * (self.C_right[idx]**2 + 1e-4)
        cond_L = self.sigma[idx] * (self.C_left[idx]**2 + 1e-4)
        
        J_right = cond_R * drive_R
        J_left  = cond_L * drive_L

        max_allowed = 0.40 * self.F / self.k.DT
        total_out = np.abs(J_right) + np.abs(J_left)
        scale = np.minimum(1.0, max_allowed / (total_out + 1e-9))
        J_right *= scale
        J_left *= scale

        dF = (J_right[im1] + J_left[ip1]) - (J_right + J_left)
        self.F += dF * self.k.DT
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        compression = np.maximum(0, dF)
        alpha = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        lam = self.k.LAMBDA / (1.0 + self.k.K_FUSION * compression * 10.0)
        
        J_R_align = np.maximum(0, J_right) + self.k.GAMMA * np.maximum(0, np.roll(J_right, 1))
        J_L_align = np.maximum(0, J_left) + self.k.GAMMA * np.maximum(0, np.roll(J_left, -1))
        
        self.C_right += (alpha * J_R_align - lam * self.C_right) * self.k.DT
        self.C_left  += (alpha * J_L_align - lam * self.C_left) * self.k.DT
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)
        
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_right) + np.abs(J_left))

        self.time += self.k.DT
        self.update_tracking()
        
        if int(self.time/self.k.DT) % 10 == 0:
            self.history_F.append(self.F.copy())


def analyze_single_run(sim, t_collision_window=0.3, post_dt=5.0):
    """
    Analyze collision using single-run ghost reference.

    Method:
    1. Identify collision time (min separation)
    2. Measure incoming velocity from pre-collision phase
    3. Ghost = start_pos + v_in * t
    4. Measure Δx and Δt relative to ghost
    5. Also computes early-out shift at t_coll + post_dt
    """
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)

    if len(track_L) < 20:
        return None

    times = track_L[:, 0]
    pos_L = track_L[:, 1]
    pos_R = track_R[:, 1]

    # Find collision time (minimum separation)
    separation = np.abs(pos_L - pos_R)
    coll_idx = np.argmin(separation)
    t_coll = times[coll_idx]
    min_sep = separation[coll_idx]

    # Initial separation and approach
    sep0 = separation[0]
    approach = sep0 - min_sep

    # Pre-collision window: before collision
    pre_coll_mask = times < t_coll * 0.8

    if np.sum(pre_coll_mask) < 10:
        return None

    # Measure incoming velocities
    t_pre = times[pre_coll_mask]
    p_L_pre = pos_L[pre_coll_mask]
    p_R_pre = pos_R[pre_coll_mask]

    res_L = linregress(t_pre, p_L_pre)
    res_R = linregress(t_pre, p_R_pre)

    v_in_L = res_L.slope
    v_in_R = res_R.slope
    r2_L = res_L.rvalue**2
    r2_R = res_R.rvalue**2

    # Post-collision window: after collision
    post_start = t_coll * (1 + t_collision_window)
    post_coll_mask = times > post_start

    if np.sum(post_coll_mask) < 10:
        # Use final values
        t_final = times[-1]
        x_final_L = pos_L[-1]
        x_final_R = pos_R[-1]
    else:
        # Measure outgoing velocities
        t_post = times[post_coll_mask]
        p_L_post = pos_L[post_coll_mask]
        p_R_post = pos_R[post_coll_mask]

        res_L_out = linregress(t_post, p_L_post)
        res_R_out = linregress(t_post, p_R_post)

        v_out_L = res_L_out.slope
        v_out_R = res_R_out.slope

        t_final = times[-1]
        x_final_L = pos_L[-1]
        x_final_R = pos_R[-1]

    # Ghost positions at t_final
    x_ghost_L = sim.start_pos_L + v_in_L * t_final
    x_ghost_R = sim.start_pos_R + v_in_R * t_final

    # Position shifts
    shift_L = x_final_L - x_ghost_L
    shift_R = x_final_R - x_ghost_R

    # Early-out shift at t_star = t_coll + post_dt
    def interp_at(tq: float, t: np.ndarray, y: np.ndarray) -> float:
        if tq <= t[0]:
            return float(y[0])
        if tq >= t[-1]:
            return float(y[-1])
        return float(np.interp(tq, t, y))

    t_star = t_coll + post_dt
    x_star_L = interp_at(t_star, times, pos_L)
    x_star_R = interp_at(t_star, times, pos_R)
    x_ghost_L_star = sim.start_pos_L + v_in_L * t_star
    x_ghost_R_star = sim.start_pos_R + v_in_R * t_star
    shift_L_star = x_star_L - x_ghost_L_star
    shift_R_star = x_star_R - x_ghost_R_star
    avg_shift_star = (shift_L_star + shift_R_star) / 2

    # Determine outcome
    final_sep = separation[-1]
    if min_sep < 10 and final_sep < 15:
        outcome = "FUSION"
    elif final_sep > 30:
        outcome = "SCATTERING"
    else:
        outcome = "PARTIAL"

    return {
        'v_in_L': v_in_L,
        'v_in_R': v_in_R,
        'v_rel': v_in_L - v_in_R,
        'r2_L': r2_L,
        'r2_R': r2_R,
        't_coll': t_coll,
        'min_sep': min_sep,
        'final_sep': final_sep,
        'shift_L': shift_L,
        'shift_R': shift_R,
        'avg_shift': (shift_L + shift_R) / 2,
        'outcome': outcome,
        'x_ghost_L': x_ghost_L,
        'x_ghost_R': x_ghost_R,
        'x_final_L': x_final_L,
        'x_final_R': x_final_R,
        'times': times,
        'pos_L': pos_L,
        'pos_R': pos_R,
        # Early-out and approach metrics:
        'sep0': sep0,
        'approach': approach,
        't_star': t_star,
        'shift_L_star': shift_L_star,
        'shift_R_star': shift_R_star,
        'avg_shift_star': avg_shift_star,
    }


def run_single_run_experiments():
    print("="*70)
    print("DET v5 COLLIDER - SINGLE-RUN OBSERVABLES")
    print("="*70)
    print("""
METHOD: Use early-time velocity from SAME run as ghost reference.
  - No separate reference run needed
  - Ghost = start_pos + v_in * t (extrapolate pre-collision velocity)
  - Δx = actual final position - ghost final position
  
This avoids reference/collision mismatch and directly measures
the deflection caused by the interaction.
""")

    trial_cfg = TrialConfig()

    # ===== SYMMETRIC CASE =====
    print("\n" + "="*70)
    print("SYMMETRIC CASE (m_L = m_R = 10)")
    print("="*70)
    print(f"\n{'g':<6} | {'v_rel':<10} | {'Δx_avg*':<16} | {'Δx_avg_final':<16} | {'min_sep':<10} | {'approach':<10} | {'Outcome':<10}")
    print("-" * 90)

    sym_results = []
    sym_g_vals = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # For plotting representative trial:
    best = None
    for g in sym_g_vals:
        v_rel_trials = []
        avg_shift_star_trials = []
        avg_shift_final_trials = []
        min_sep_trials = []
        approach_trials = []
        outcome_trials = []
        trial_res = []
        for trial in range(trial_cfg.trials):
            rng = np.random.default_rng(trial_cfg.seed + int(g * 1000) + trial)
            k = DETConstants(g=g)
            sim = ActiveManifold1D(size=200, constants=k)
            sim.initialize_collider_event(offset=40, v_kick=2.0, mass_L=10.0, mass_R=10.0)
            apply_small_perturbations(sim, rng, trial_cfg.theta_noise_std)
            for _ in range(800):
                sim.step()
            res = analyze_single_run(sim, post_dt=trial_cfg.post_dt)
            if res:
                v_rel_trials.append(res['v_rel'])
                avg_shift_star_trials.append(res['avg_shift_star'])
                avg_shift_final_trials.append(res['avg_shift'])
                min_sep_trials.append(res['min_sep'])
                approach_trials.append(res['approach'])
                outcome_trials.append(res['outcome'])
                res['g'] = g
                trial_res.append(res)
        # Aggregate
        if len(trial_res) == 0:
            continue
        # Pick representative for plotting: median trial by avg_shift_star
        idx_best = np.argsort(np.abs(np.array(avg_shift_star_trials) - np.median(avg_shift_star_trials)))[0]
        if g == 1.6 or best is None:
            best = trial_res[idx_best]
        def meanstd(x): return (np.mean(x), np.std(x))
        v_rel_m, v_rel_s = meanstd(v_rel_trials)
        avg_star_m, avg_star_s = meanstd(avg_shift_star_trials)
        avg_final_m, avg_final_s = meanstd(avg_shift_final_trials)
        min_sep_m, min_sep_s = meanstd(min_sep_trials)
        approach_m, approach_s = meanstd(approach_trials)
        # Most common outcome
        from collections import Counter
        mode_outcome = Counter(outcome_trials).most_common(1)[0][0]
        print(f"{g:<6.2f} | {v_rel_m:>7.3f}±{v_rel_s:<5.3f} | {avg_star_m:>7.2f}±{avg_star_s:<7.2f} | {avg_final_m:>7.2f}±{avg_final_s:<7.2f} | {min_sep_m:>6.2f}±{min_sep_s:<4.2f} | {approach_m:>6.2f}±{approach_s:<4.2f} | {mode_outcome:<10}")
        # Save aggregate result for plotting
        sym_results.append({
            'g': g,
            'v_rel_mean': v_rel_m, 'v_rel_std': v_rel_s,
            'avg_shift_star_mean': avg_star_m, 'avg_shift_star_std': avg_star_s,
            'avg_shift_mean': avg_final_m, 'avg_shift_std': avg_final_s,
            'min_sep_mean': min_sep_m, 'min_sep_std': min_sep_s,
            'approach_mean': approach_m, 'approach_std': approach_s,
            'outcome_mode': mode_outcome,
            # For plotting lines:
            'avg_shift_star_trials': avg_shift_star_trials,
            'avg_shift_trials': avg_shift_final_trials,
            'v_rel_trials': v_rel_trials,
            # And for plotting example:
            'example': trial_res[idx_best],
        })

    # ===== ASYMMETRIC CASE =====
    print("\n" + "="*70)
    print("ASYMMETRIC CASE (m_L = 10, m_R = 14)")
    print("="*70)
    print(f"\n{'g':<6} | {'v_rel':<10} | {'Δx_avg*':<16} | {'Δx_avg_final':<16} | {'min_sep':<10} | {'approach':<10} | {'Outcome':<10}")
    print("-" * 90)

    asym_results = []
    asym_g_vals = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    best_a = None
    for g in asym_g_vals:
        v_rel_trials = []
        avg_shift_star_trials = []
        avg_shift_final_trials = []
        shift_L_star_trials = []
        shift_R_star_trials = []
        shift_L_final_trials = []
        shift_R_final_trials = []
        min_sep_trials = []
        approach_trials = []
        outcome_trials = []
        trial_res = []
        for trial in range(trial_cfg.trials):
            rng = np.random.default_rng(trial_cfg.seed + int(g * 1000) + trial)
            k = DETConstants(g=g)
            # Jitter mass and width
            mL = 10.0 * (1 + rng.normal(0, trial_cfg.mass_jitter_std))
            mR = 14.0 * (1 + rng.normal(0, trial_cfg.mass_jitter_std))
            width = 5.0 * (1 + rng.normal(0, trial_cfg.width_jitter_std))
            sim = ActiveManifold1D(size=200, constants=k)
            sim.initialize_collider_event(offset=40, v_kick=2.0, mass_L=mL, mass_R=mR, width=width)
            apply_small_perturbations(sim, rng, trial_cfg.theta_noise_std)
            for _ in range(800):
                sim.step()
            res = analyze_single_run(sim, post_dt=trial_cfg.post_dt)
            if res:
                v_rel_trials.append(res['v_rel'])
                avg_shift_star_trials.append(res['avg_shift_star'])
                avg_shift_final_trials.append(res['avg_shift'])
                shift_L_star_trials.append(res['shift_L_star'])
                shift_R_star_trials.append(res['shift_R_star'])
                shift_L_final_trials.append(res['shift_L'])
                shift_R_final_trials.append(res['shift_R'])
                min_sep_trials.append(res['min_sep'])
                approach_trials.append(res['approach'])
                outcome_trials.append(res['outcome'])
                res['g'] = g
                trial_res.append(res)
        if len(trial_res) == 0:
            continue
        idx_best = np.argsort(np.abs(np.array(avg_shift_star_trials) - np.median(avg_shift_star_trials)))[0]
        if g == 1.6 or best_a is None:
            best_a = trial_res[idx_best]
        def meanstd(x): return (np.mean(x), np.std(x))
        v_rel_m, v_rel_s = meanstd(v_rel_trials)
        avg_star_m, avg_star_s = meanstd(avg_shift_star_trials)
        avg_final_m, avg_final_s = meanstd(avg_shift_final_trials)
        min_sep_m, min_sep_s = meanstd(min_sep_trials)
        approach_m, approach_s = meanstd(approach_trials)
        mode_outcome = Counter(outcome_trials).most_common(1)[0][0]
        print(f"{g:<6.2f} | {v_rel_m:>7.3f}±{v_rel_s:<5.3f} | {avg_star_m:>7.2f}±{avg_star_s:<7.2f} | {avg_final_m:>7.2f}±{avg_final_s:<7.2f} | {min_sep_m:>6.2f}±{min_sep_s:<4.2f} | {approach_m:>6.2f}±{approach_s:<4.2f} | {mode_outcome:<10}")
        asym_results.append({
            'g': g,
            'v_rel_mean': v_rel_m, 'v_rel_std': v_rel_s,
            'avg_shift_star_mean': avg_star_m, 'avg_shift_star_std': avg_star_s,
            'avg_shift_mean': avg_final_m, 'avg_shift_std': avg_final_s,
            'min_sep_mean': min_sep_m, 'min_sep_std': min_sep_s,
            'approach_mean': approach_m, 'approach_std': approach_s,
            'outcome_mode': mode_outcome,
            'shift_L_star_trials': shift_L_star_trials,
            'shift_R_star_trials': shift_R_star_trials,
            'shift_L_final_trials': shift_L_final_trials,
            'shift_R_final_trials': shift_R_final_trials,
            'avg_shift_star_trials': avg_shift_star_trials,
            'avg_shift_trials': avg_shift_final_trials,
            'example': trial_res[idx_best],
        })

    # ===== ENERGY SCAN =====
    print("\n" + "="*70)
    print("ENERGY SCAN (g=1.5, symmetric)")
    print("="*70)
    print(f"\n{'v_kick':<8} | {'v_in_L':<8} | {'v_in_R':<8} | {'v_rel':<8} | {'Outcome':<10} | {'Δx_avg':<8} | {'min_sep':<8}")
    print("-" * 75)

    energy_results = []
    for v_kick in [1.0, 1.5, 2.0, 2.5, 3.0]:
        k = DETConstants(g=1.5)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(offset=40, v_kick=v_kick, mass_L=10.0, mass_R=10.0)
        for _ in range(800):
            sim.step()
        res = analyze_single_run(sim)
        if res:
            res['v_kick'] = v_kick
            energy_results.append(res)
            print(f"{v_kick:<8.2f} | {res['v_in_L']:<8.4f} | {res['v_in_R']:<8.4f} | {res['v_rel']:<8.4f} | "
                  f"{res['outcome']:<10} | {res['avg_shift']:<8.2f} | {res['min_sep']:<8.2f}")

    # ===== VISUALIZATION =====
    print("\n[Creating visualization...]")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Example trajectory (symmetric)
    ax = axes[0, 0]
    best_plot = best['example'] if isinstance(best, dict) and 'example' in best else best
    ax.plot(best_plot['times'], best_plot['pos_L'], 'b-', linewidth=2, label='L actual')
    ax.plot(best_plot['times'], best_plot['pos_R'], 'r-', linewidth=2, label='R actual')
    # Ghost trajectories
    t_arr = np.linspace(0, best_plot['times'][-1], 100)
    start_L = best_plot['pos_L'][0]
    start_R = best_plot['pos_R'][0]
    ax.plot(t_arr, start_L + best_plot['v_in_L']*t_arr, 'b--', alpha=0.5, label=f'L ghost (v={best_plot["v_in_L"]:.3f})')
    ax.plot(t_arr, start_R + best_plot['v_in_R']*t_arr, 'r--', alpha=0.5, label=f'R ghost (v={best_plot["v_in_R"]:.3f})')
    ax.axvline(x=best_plot['t_coll'], color='g', linestyle=':', label='Collision')
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.set_title(f"Symmetric (g={best_plot['g']}, Δx_avg*={best_plot['avg_shift_star']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Example trajectory (asymmetric)
    ax = axes[0, 1]
    best_a_plot = best_a['example'] if isinstance(best_a, dict) and 'example' in best_a else best_a
    ax.plot(best_a_plot['times'], best_a_plot['pos_L'], 'b-', linewidth=2, label='L (m=10)')
    ax.plot(best_a_plot['times'], best_a_plot['pos_R'], 'r-', linewidth=2, label='R (m=14)')
    t_arr = np.linspace(0, best_a_plot['times'][-1], 100)
    start_L = best_a_plot['pos_L'][0]
    start_R = best_a_plot['pos_R'][0]
    ax.plot(t_arr, start_L + best_a_plot['v_in_L']*t_arr, 'b--', alpha=0.5, label='L ghost')
    ax.plot(t_arr, start_R + best_a_plot['v_in_R']*t_arr, 'r--', alpha=0.5, label='R ghost')
    ax.axvline(x=best_a_plot['t_coll'], color='g', linestyle=':', label='Collision')
    ax.set_xlabel("Time")
    ax.set_ylabel("Position")
    ax.set_title(f"Asymmetric (g={best_a_plot['g']}, Δx_L*={best_a_plot['shift_L_star']:.2f}, Δx_R*={best_a_plot['shift_R_star']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Δx vs g (early-out shift as primary)
    ax = axes[1, 0]
    g_sym = [r['g'] for r in sym_results]
    dx_sym_star = [r['avg_shift_star_mean'] for r in sym_results]
    dx_sym_star_std = [r['avg_shift_star_std'] for r in sym_results]
    dx_sym_final = [r['avg_shift_mean'] for r in sym_results]
    dx_sym_final_std = [r['avg_shift_std'] for r in sym_results]
    g_asym = [r['g'] for r in asym_results]
    dx_L_star = [np.mean(r['shift_L_star_trials']) for r in asym_results]
    dx_R_star = [np.mean(r['shift_R_star_trials']) for r in asym_results]
    dx_asym_star = [r['avg_shift_star_mean'] for r in asym_results]
    dx_L_final = [np.mean(r['shift_L_final_trials']) for r in asym_results]
    dx_R_final = [np.mean(r['shift_R_final_trials']) for r in asym_results]
    dx_asym_final = [r['avg_shift_mean'] for r in asym_results]
    # Early-out (solid)
    ax.plot(g_sym, dx_sym_star, 'ko-', markersize=10, linewidth=2, label='Sym Δx_avg*')
    ax.fill_between(g_sym, np.array(dx_sym_star)-np.array(dx_sym_star_std), np.array(dx_sym_star)+np.array(dx_sym_star_std), color='k', alpha=0.08)
    ax.plot(g_asym, dx_L_star, 'b^-', markersize=8, alpha=0.9, label='Asym Δx_L*')
    ax.plot(g_asym, dx_R_star, 'rv-', markersize=8, alpha=0.9, label='Asym Δx_R*')
    ax.plot(g_asym, dx_asym_star, 'gs-', markersize=10, linewidth=2, label='Asym Δx_avg*')
    # Final-time (dashed, faint)
    ax.plot(g_sym, dx_sym_final, 'k--', alpha=0.3, linewidth=1, label='Sym Δx_avg (final)')
    ax.plot(g_asym, dx_L_final, 'b^--', alpha=0.3, linewidth=1)
    ax.plot(g_asym, dx_R_final, 'rv--', alpha=0.3, linewidth=1)
    ax.plot(g_asym, dx_asym_final, 'gs--', alpha=0.3, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Position Shift Δx (lattice units)")
    ax.set_title("Early-out Position Shift vs Coupling")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Δx vs energy
    ax = axes[1, 1]
    v_kicks = [r['v_kick'] for r in energy_results]
    dx_energy = [r['avg_shift'] for r in energy_results]
    v_rels = [r['v_rel'] for r in energy_results]
    ax.plot(v_kicks, dx_energy, 'ko-', markersize=10, linewidth=2, label='Δx_avg')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel("Initial Velocity Kick")
    ax.set_ylabel("Position Shift Δx")
    ax.set_title("Position Shift vs Energy (g=1.5)")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(v_kicks, v_rels, 'rs--', markersize=8, alpha=0.7, label='v_rel')
    ax2.set_ylabel("Measured v_rel", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.savefig('./det_v5_single_run.png', dpi=150)
    print("\nSaved to det_v5_single_run.png")

    return sym_results, asym_results, energy_results


if __name__ == "__main__":
    sym, asym, energy = run_single_run_experiments()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Multi-trial measurement layer:
  - Reports mean±std over 10 trials per g with tiny preparation noise
  - Uses early-out shift at t* = t_coll + post_dt to reduce late-time drift
  - Adds approach metric: approach = sep0 - min_sep

Recommended next experiments (no code changes):
  1) Increase steps/runtime until detector planes are crossed, then add Δt from crossing times.
  2) Run energy scans per g (sweep v_kick) and look for systematic a(g, v) structure.
  3) Sweep mass ratio (m_R/m_L) to separate true regime changes from identity/labeling artifacts.
""")