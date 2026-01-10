import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import label
from collections import defaultdict
import time as clock_time

"""
DET v5 SCATTERING LENGTH - LIGHTWEIGHT VERSION
===============================================
Reduced trials and grid for faster execution.
"""

S_CONTACT = 2.0
S_HOLD = 8.0
PERSIST_FRAC = 0.25
FRAG_THRESHOLD = 3

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
        self.time = 0.0
        self.track_L = []
        self.track_R = []
        self.last_pos_L = None
        self.last_pos_R = None

    def initialize_soliton(self, position, width, mass, velocity_kick=0.0):
        x = np.arange(self.N)
        dx = x - position
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        envelope = np.exp(-0.5 * (np.abs(dx) / width)**2)
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

    def initialize_collider_event(self, offset=15, width=5.0, mass_L=10.0, mass_R=10.0, v_kick=2.0):
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
        mass_dist = np.maximum(self.F - self.k.F_VAC, 0)
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

    def count_peaks(self):
        return len(self.find_blob_coms())

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
            self.last_pos_L = self.unwrap_position(com, self.last_pos_L)
            self.last_pos_R = self.unwrap_position(com, self.last_pos_R)
            self.track_L.append((self.time, self.last_pos_L))
            self.track_R.append((self.time, self.last_pos_R))
            return
        best_L_idx = min(range(len(blobs)), key=lambda i: ring_dist(blobs[i][0], self.last_pos_L % self.N))
        remaining = [i for i in range(len(blobs)) if i != best_L_idx]
        best_R_idx = min(remaining, key=lambda i: ring_dist(blobs[i][0], self.last_pos_R % self.N)) if remaining else best_L_idx
        self.last_pos_L = self.unwrap_position(blobs[best_L_idx][0], self.last_pos_L)
        self.last_pos_R = self.unwrap_position(blobs[best_R_idx][0], self.last_pos_R)
        self.track_L.append((self.time, self.last_pos_L))
        self.track_R.append((self.time, self.last_pos_R))

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


def classify_and_measure(sim):
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)
    if len(track_L) < 20:
        return None, {}
    times = track_L[:, 0]
    pos_L = track_L[:, 1]
    pos_R = track_R[:, 1]
    separation = np.abs(pos_L - pos_R)
    min_sep = np.min(separation)
    final_peaks = sim.count_peaks()
    n_frames = len(separation)
    persist_start = int(n_frames * (1 - PERSIST_FRAC))
    late_sep = separation[persist_start:]
    persistent_binding = np.all(late_sep < S_HOLD)
    made_contact = min_sep <= S_CONTACT
    if made_contact and final_peaks == 1 and persistent_binding:
        outcome = 'BOUND'
    elif final_peaks >= FRAG_THRESHOLD:
        outcome = 'FRAGMENT'
    else:
        outcome = 'SCATTER'
    coll_idx = np.argmin(separation)
    t_coll = times[coll_idx]
    pre_coll_mask = times < t_coll * 0.8
    if np.sum(pre_coll_mask) >= 5:
        t_pre = times[pre_coll_mask]
        res_L = linregress(t_pre, pos_L[pre_coll_mask])
        res_R = linregress(t_pre, pos_R[pre_coll_mask])
        v_in_L = res_L.slope
        v_in_R = res_R.slope
    else:
        v_in_L = v_in_R = 0
    t_star = t_coll + 5.0
    star_idx = min(np.searchsorted(times, t_star), len(times) - 1)
    t_actual = times[star_idx]
    x_star_L = pos_L[star_idx]
    x_star_R = pos_R[star_idx]
    x_ghost_L = pos_L[0] + v_in_L * t_actual
    x_ghost_R = pos_R[0] + v_in_R * t_actual
    shift_L = x_star_L - x_ghost_L
    shift_R = x_star_R - x_ghost_R
    shift_avg = (shift_L + shift_R) / 2
    return outcome, {
        'min_sep': min_sep,
        'made_contact': made_contact,
        'shift_L': shift_L,
        'shift_R': shift_R,
        'shift_avg': shift_avg,
        'scattering_length': -shift_avg,
        'v_rel': v_in_L - v_in_R,
    }


def run_trial(g, v_kick, offset, rng, theta_noise=3e-4, steps=800):
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick)
    if theta_noise > 0:
        sim.theta += rng.normal(0, theta_noise, size=sim.N)
        sim.theta = np.mod(sim.theta, 2*np.pi)
    for _ in range(steps):
        sim.step()
    return classify_and_measure(sim)


def main():
    print("="*70)
    print("DET v5 SCATTERING LENGTH EXTRACTION (LIGHTWEIGHT)")
    print("="*70)
    
    rng = np.random.default_rng(42)
    n_trials = 15  # Reduced
    offset = 15
    
    # Experiment 1: g-scan
    print("\n--- g-scan at v_kick=2.0 ---")
    print(f"{'g':<6} | {'scatter%':<10} | {'a (mean)':<12} | {'a (std)':<10}")
    print("-" * 45)
    
    exp1 = []
    for g in np.arange(0.8, 2.4, 0.2):
        scatter_lengths = []
        outcomes = defaultdict(int)
        for _ in range(n_trials):
            outcome, m = run_trial(g, 2.0, offset, rng)
            if outcome:
                outcomes[outcome] += 1
                if outcome == 'SCATTER' and m['made_contact']:
                    scatter_lengths.append(m['scattering_length'])
        
        total = sum(outcomes.values())
        scatter_frac = outcomes['SCATTER'] / total if total > 0 else 0
        a_mean = np.mean(scatter_lengths) if len(scatter_lengths) >= 2 else np.nan
        a_std = np.std(scatter_lengths) if len(scatter_lengths) >= 2 else np.nan
        
        print(f"{g:<6.1f} | {scatter_frac:<10.2f} | {a_mean:<12.1f} | {a_std:<10.1f}")
        exp1.append({'g': g, 'scatter_frac': scatter_frac, 'a_mean': a_mean, 'a_std': a_std})
    
    # Experiment 2: v_kick scan
    print("\n--- v_kick scan at g=1.5 ---")
    print(f"{'v_kick':<8} | {'scatter%':<10} | {'a (mean)':<12} | {'a (std)':<10}")
    print("-" * 48)
    
    exp2 = []
    for v_kick in np.arange(1.7, 2.3, 0.1):
        scatter_lengths = []
        outcomes = defaultdict(int)
        for _ in range(n_trials):
            outcome, m = run_trial(1.5, v_kick, offset, rng)
            if outcome:
                outcomes[outcome] += 1
                if outcome == 'SCATTER' and m['made_contact']:
                    scatter_lengths.append(m['scattering_length'])
        
        total = sum(outcomes.values())
        scatter_frac = outcomes['SCATTER'] / total if total > 0 else 0
        a_mean = np.mean(scatter_lengths) if len(scatter_lengths) >= 2 else np.nan
        a_std = np.std(scatter_lengths) if len(scatter_lengths) >= 2 else np.nan
        
        print(f"{v_kick:<8.2f} | {scatter_frac:<10.2f} | {a_mean:<12.1f} | {a_std:<10.1f}")
        exp2.append({'v_kick': v_kick, 'scatter_frac': scatter_frac, 'a_mean': a_mean, 'a_std': a_std})
    
    # Visualization
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: a vs g
    ax = axes[0]
    g_vals = [r['g'] for r in exp1]
    a_vals = [r['a_mean'] for r in exp1]
    a_stds = [r['a_std'] for r in exp1]
    valid = ~np.isnan(a_vals)
    
    if np.any(valid):
        g_v = np.array(g_vals)[valid]
        a_v = np.array(a_vals)[valid]
        a_s = np.array(a_stds)[valid]
        ax.errorbar(g_v, a_v, yerr=a_s, fmt='ko-', markersize=10, capsize=5, linewidth=2)
        ax.fill_between(g_v, a_v - a_s, a_v + a_s, alpha=0.2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='a=0')
    ax.set_xlabel("Coupling g", fontsize=12)
    ax.set_ylabel("Scattering Length a", fontsize=12)
    ax.set_title("a(g) at v_kick=2.0", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: a vs v_kick
    ax = axes[1]
    v_vals = [r['v_kick'] for r in exp2]
    a_vals2 = [r['a_mean'] for r in exp2]
    a_stds2 = [r['a_std'] for r in exp2]
    valid2 = ~np.isnan(a_vals2)
    
    if np.any(valid2):
        v_v = np.array(v_vals)[valid2]
        a_v2 = np.array(a_vals2)[valid2]
        a_s2 = np.array(a_stds2)[valid2]
        ax.errorbar(v_v, a_v2, yerr=a_s2, fmt='bs-', markersize=10, capsize=5, linewidth=2)
        ax.fill_between(v_v, a_v2 - a_s2, a_v2 + a_s2, alpha=0.2, color='blue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel("Velocity Kick", fontsize=12)
    ax.set_ylabel("Scattering Length a", fontsize=12)
    ax.set_title("a(v) at g=1.5", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Scatter fraction
    ax = axes[2]
    scatter_fracs = [r['scatter_frac'] for r in exp1]
    ax.bar(g_vals, scatter_fracs, width=0.15, color='green', alpha=0.7, edgecolor='black')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Coupling g", fontsize=12)
    ax.set_ylabel("Scatter Fraction", fontsize=12)
    ax.set_title("Scatter Fraction vs g", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/det_v5_scattering_length.png', dpi=150)
    plt.savefig('/mnt/user-data/outputs/det_v5_scattering_length.png', dpi=150)
    print("Saved to det_v5_scattering_length.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nScattering Length a(g) at v_kick=2.0:")
    for r in exp1:
        if not np.isnan(r['a_mean']):
            print(f"  g={r['g']:.1f}: a = {r['a_mean']:+.1f} ± {r['a_std']:.1f}")
    
    # Check for sign change
    a_valid = [r['a_mean'] for r in exp1 if not np.isnan(r['a_mean'])]
    if len(a_valid) > 1:
        signs = np.sign(a_valid)
        if not np.all(signs == signs[0]):
            print("\n*** SIGN CHANGE DETECTED ***")
            print("This may indicate a bound state threshold!")
    
    return exp1, exp2


if __name__ == "__main__":
    t0 = clock_time.time()
    exp1, exp2 = main()
    print(f"\nCompleted in {clock_time.time() - t0:.1f}s")
