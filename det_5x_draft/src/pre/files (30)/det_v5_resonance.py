import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import label
from collections import defaultdict
import time as clock_time

"""
DET v5 COLLIDER - RESONANCE MAPPING
===================================
Focus on the g = 1.5 to 1.9 region where sign change was detected.

Goals:
1. Fine g-scan to locate resonance precisely
2. More trials to reduce variance
3. Asymmetric mass test for momentum conservation
4. Energy-dependent scattering length a(E)
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


def analyze_collision(sim):
    """Full collision analysis with scattering observables."""
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)
    if len(track_L) < 20:
        return None
    
    times = track_L[:, 0]
    pos_L = track_L[:, 1]
    pos_R = track_R[:, 1]
    separation = np.abs(pos_L - pos_R)
    min_sep = np.min(separation)
    final_peaks = sim.count_peaks()
    
    # Persistence check
    n_frames = len(separation)
    persist_start = int(n_frames * (1 - PERSIST_FRAC))
    late_sep = separation[persist_start:]
    persistent_binding = np.all(late_sep < S_HOLD)
    made_contact = min_sep <= S_CONTACT
    
    # Classification
    if made_contact and final_peaks == 1 and persistent_binding:
        outcome = 'BOUND'
    elif final_peaks >= FRAG_THRESHOLD:
        outcome = 'FRAGMENT'
    else:
        outcome = 'SCATTER'
    
    # Velocity measurement
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
    
    v_rel = v_in_L - v_in_R
    
    # Position shifts at t* = t_coll + 5
    t_star = t_coll + 5.0
    star_idx = min(np.searchsorted(times, t_star), len(times) - 1)
    t_actual = times[star_idx]
    
    x_ghost_L = pos_L[0] + v_in_L * t_actual
    x_ghost_R = pos_R[0] + v_in_R * t_actual
    
    shift_L = pos_L[star_idx] - x_ghost_L
    shift_R = pos_R[star_idx] - x_ghost_R
    shift_avg = (shift_L + shift_R) / 2
    
    # Momentum proxy (for asymmetric masses)
    delta_p = sim.mass_L * shift_L + sim.mass_R * shift_R
    
    return {
        'outcome': outcome,
        'min_sep': min_sep,
        'made_contact': made_contact,
        'v_in_L': v_in_L,
        'v_in_R': v_in_R,
        'v_rel': v_rel,
        'shift_L': shift_L,
        'shift_R': shift_R,
        'shift_avg': shift_avg,
        'scattering_length': -shift_avg,
        'delta_p': delta_p,
    }


def run_trial(g, v_kick, offset, rng, theta_noise=3e-4, steps=800, mass_L=10.0, mass_R=10.0):
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick, mass_L=mass_L, mass_R=mass_R)
    if theta_noise > 0:
        sim.theta += rng.normal(0, theta_noise, size=sim.N)
        sim.theta = np.mod(sim.theta, 2*np.pi)
    for _ in range(steps):
        sim.step()
    return analyze_collision(sim)


def main():
    print("="*70)
    print("DET v5 RESONANCE MAPPING")
    print("="*70)
    
    rng = np.random.default_rng(42)
    offset = 15
    
    # =========================================================================
    # EXPERIMENT 1: Fine g-scan around resonance (g = 1.5 to 1.9)
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Fine g-scan (resonance region)")
    print("="*70)
    
    n_trials = 25
    v_kick = 2.0
    g_fine = np.arange(1.50, 1.95, 0.05)
    
    print(f"g = 1.50 to 1.90 (step 0.05), v_kick={v_kick}, n={n_trials}\n")
    print(f"{'g':<6} | {'scatter%':<10} | {'bound%':<8} | {'a':<12} | {'a_std':<10} | {'v_rel':<8}")
    print("-" * 65)
    
    exp1 = []
    for g in g_fine:
        outcomes = defaultdict(int)
        scatter_a = []
        v_rels = []
        
        for _ in range(n_trials):
            res = run_trial(g, v_kick, offset, rng)
            if res:
                outcomes[res['outcome']] += 1
                if res['outcome'] == 'SCATTER' and res['made_contact']:
                    scatter_a.append(res['scattering_length'])
                    v_rels.append(res['v_rel'])
        
        total = sum(outcomes.values())
        scatter_frac = outcomes['SCATTER'] / total if total > 0 else 0
        bound_frac = outcomes['BOUND'] / total if total > 0 else 0
        
        a_mean = np.mean(scatter_a) if len(scatter_a) >= 3 else np.nan
        a_std = np.std(scatter_a) if len(scatter_a) >= 3 else np.nan
        v_rel_mean = np.mean(v_rels) if v_rels else np.nan
        
        print(f"{g:<6.2f} | {scatter_frac:<10.2f} | {bound_frac:<8.2f} | {a_mean:<12.1f} | {a_std:<10.1f} | {v_rel_mean:<8.3f}")
        
        exp1.append({
            'g': g, 'scatter_frac': scatter_frac, 'bound_frac': bound_frac,
            'a_mean': a_mean, 'a_std': a_std, 'v_rel': v_rel_mean, 'n_scatter': len(scatter_a)
        })
    
    # =========================================================================
    # EXPERIMENT 2: Asymmetric mass test (momentum conservation)
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: Momentum Conservation (m_L=10, m_R=14)")
    print("="*70)
    
    g_test = 1.8  # In scattering regime
    n_trials_mom = 30
    
    print(f"g={g_test}, v_kick={v_kick}, n={n_trials_mom}\n")
    
    shifts_L = []
    shifts_R = []
    delta_ps = []
    
    for _ in range(n_trials_mom):
        res = run_trial(g_test, v_kick, offset, rng, mass_L=10.0, mass_R=14.0)
        if res and res['outcome'] == 'SCATTER' and res['made_contact']:
            shifts_L.append(res['shift_L'])
            shifts_R.append(res['shift_R'])
            delta_ps.append(res['delta_p'])
    
    if shifts_L:
        print(f"Scattering events: {len(shifts_L)}/{n_trials_mom}")
        print(f"Mean Δx_L: {np.mean(shifts_L):+.2f} ± {np.std(shifts_L):.2f}")
        print(f"Mean Δx_R: {np.mean(shifts_R):+.2f} ± {np.std(shifts_R):.2f}")
        print(f"Mean Δp proxy: {np.mean(delta_ps):+.2f} ± {np.std(delta_ps):.2f}")
        print(f"  (Should be ~0 if momentum conserved)")
        
        # Check ratio
        if np.mean(shifts_R) != 0:
            ratio = -np.mean(shifts_L) / np.mean(shifts_R)
            expected_ratio = 14.0 / 10.0  # m_R / m_L
            print(f"\nRatio -Δx_L/Δx_R: {ratio:.2f} (expected {expected_ratio:.2f} for momentum conservation)")
    
    # =========================================================================
    # EXPERIMENT 3: Energy dependence a(v_rel)
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: Energy Dependence a(v_rel)")
    print("="*70)
    
    g_fixed = 1.8
    n_trials_e = 20
    v_kicks_e = [1.5, 1.75, 2.0, 2.25, 2.5]
    
    print(f"g={g_fixed}, n={n_trials_e}\n")
    print(f"{'v_kick':<8} | {'v_rel':<10} | {'scatter%':<10} | {'a':<12} | {'a_std':<10}")
    print("-" * 55)
    
    exp3 = []
    for v_kick_e in v_kicks_e:
        outcomes = defaultdict(int)
        scatter_a = []
        v_rels = []
        
        for _ in range(n_trials_e):
            res = run_trial(g_fixed, v_kick_e, offset, rng)
            if res:
                outcomes[res['outcome']] += 1
                if res['outcome'] == 'SCATTER' and res['made_contact']:
                    scatter_a.append(res['scattering_length'])
                    v_rels.append(res['v_rel'])
        
        total = sum(outcomes.values())
        scatter_frac = outcomes['SCATTER'] / total if total > 0 else 0
        a_mean = np.mean(scatter_a) if len(scatter_a) >= 3 else np.nan
        a_std = np.std(scatter_a) if len(scatter_a) >= 3 else np.nan
        v_rel_mean = np.mean(v_rels) if v_rels else np.nan
        
        print(f"{v_kick_e:<8.2f} | {v_rel_mean:<10.3f} | {scatter_frac:<10.2f} | {a_mean:<12.1f} | {a_std:<10.1f}")
        
        exp3.append({
            'v_kick': v_kick_e, 'v_rel': v_rel_mean, 'scatter_frac': scatter_frac,
            'a_mean': a_mean, 'a_std': a_std
        })
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Fine g-scan: Scattering length
    ax = axes[0, 0]
    g_vals = [r['g'] for r in exp1]
    a_vals = [r['a_mean'] for r in exp1]
    a_stds = [r['a_std'] for r in exp1]
    valid = ~np.isnan(a_vals)
    
    if np.any(valid):
        g_v = np.array(g_vals)[valid]
        a_v = np.array(a_vals)[valid]
        a_s = np.array(a_stds)[valid]
        ax.errorbar(g_v, a_v, yerr=a_s, fmt='ko-', markersize=10, capsize=5, linewidth=2)
        ax.fill_between(g_v, a_v - a_s, a_v + a_s, alpha=0.2, color='blue')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='a=0 (threshold)')
    ax.set_xlabel("Coupling g", fontsize=12)
    ax.set_ylabel("Scattering Length a", fontsize=12)
    ax.set_title("Fine g-scan: Resonance Region", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Phase fractions vs g
    ax = axes[0, 1]
    scatter_fracs = [r['scatter_frac'] for r in exp1]
    bound_fracs = [r['bound_frac'] for r in exp1]
    
    width = 0.02
    ax.bar(np.array(g_vals) - width, scatter_fracs, width=width*1.8, color='blue', alpha=0.7, label='SCATTER')
    ax.bar(np.array(g_vals) + width, bound_fracs, width=width*1.8, color='green', alpha=0.7, label='BOUND')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Coupling g", fontsize=12)
    ax.set_ylabel("Fraction", fontsize=12)
    ax.set_title("Outcome Fractions vs g", fontsize=14)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # 3. Momentum conservation histogram
    ax = axes[1, 0]
    if shifts_L:
        ax.hist(shifts_L, bins=12, alpha=0.5, color='blue', label=f'Δx_L (m=10)', edgecolor='black')
        ax.hist(shifts_R, bins=12, alpha=0.5, color='red', label=f'Δx_R (m=14)', edgecolor='black')
        ax.axvline(x=np.mean(shifts_L), color='blue', linestyle='--', linewidth=2)
        ax.axvline(x=np.mean(shifts_R), color='red', linestyle='--', linewidth=2)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel("Position Shift Δx", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Momentum Test (Δp = {np.mean(delta_ps):.1f}±{np.std(delta_ps):.1f})", fontsize=14)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No scatter events", ha='center', va='center', transform=ax.transAxes)
    
    # 4. Energy dependence
    ax = axes[1, 1]
    v_rels_e = [r['v_rel'] for r in exp3]
    a_vals_e = [r['a_mean'] for r in exp3]
    a_stds_e = [r['a_std'] for r in exp3]
    valid_e = ~np.isnan(a_vals_e)
    
    if np.any(valid_e):
        v_v = np.array(v_rels_e)[valid_e]
        a_v_e = np.array(a_vals_e)[valid_e]
        a_s_e = np.array(a_stds_e)[valid_e]
        ax.errorbar(v_v, a_v_e, yerr=a_s_e, fmt='rs-', markersize=10, capsize=5, linewidth=2)
        ax.fill_between(v_v, a_v_e - a_s_e, a_v_e + a_s_e, alpha=0.2, color='red')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel("Relative Velocity v_rel", fontsize=12)
    ax.set_ylabel("Scattering Length a", fontsize=12)
    ax.set_title(f"Energy Dependence at g={g_fixed}", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./det_v5_resonance.png', dpi=150)
    print("Saved to det_v5_resonance.png")
    
    return exp1, exp3, (shifts_L, shifts_R, delta_ps)


if __name__ == "__main__":
    t0 = clock_time.time()
    exp1, exp3, mom_data = main()
    elapsed = clock_time.time() - t0
    
    print("\n" + "="*70)
    print(f"SUMMARY (completed in {elapsed:.1f}s)")
    print("="*70)
    
    # Find zero crossing
    print("\nResonance search:")
    a_vals = [(r['g'], r['a_mean']) for r in exp1 if not np.isnan(r['a_mean'])]
    if len(a_vals) >= 2:
        for i in range(1, len(a_vals)):
            g_prev, a_prev = a_vals[i-1]
            g_curr, a_curr = a_vals[i]
            if a_prev * a_curr < 0:  # Sign change
                # Linear interpolation for zero crossing
                g_zero = g_prev + (g_curr - g_prev) * (-a_prev) / (a_curr - a_prev)
                print(f"  Zero crossing detected between g={g_prev:.2f} and g={g_curr:.2f}")
                print(f"  Estimated resonance: g* ≈ {g_zero:.3f}")
                break
        else:
            print("  No zero crossing found in scanned range")
    
    # Momentum conservation
    shifts_L, shifts_R, delta_ps = mom_data
    if delta_ps:
        mean_dp = np.mean(delta_ps)
        std_dp = np.std(delta_ps)
        print(f"\nMomentum conservation:")
        print(f"  Δp = {mean_dp:.1f} ± {std_dp:.1f}")
        if abs(mean_dp) < 2 * std_dp:
            print(f"  ✓ Consistent with conservation (|Δp| < 2σ)")
        else:
            print(f"  ✗ Possible violation (|Δp| > 2σ)")
