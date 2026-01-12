import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import label
from collections import Counter

"""
DET v5 COLLIDER - PHASE BOUNDARY EXPERIMENTS
=============================================
Based on forced contact results showing scatter↔fusion transition.

Experiment A: Boundary scan around capture threshold (v_kick sweep)
Experiment B: g scan at two energies (low vs high)
Experiment C: Offset phase boundary (does fusion require overlap?)
"""

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

    def initialize_collider_event(self, offset=15, width=5.0, mass_L=10.0, mass_R=10.0, v_kick=2.0):
        center = self.N // 2
        self.start_pos_L = center - offset
        self.start_pos_R = center + offset
        self.mass_L = mass_L
        self.mass_R = mass_R
        self.initial_sep = 2 * offset
        
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

    def count_peaks(self):
        """Count number of distinct mass peaks."""
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


def analyze_run(sim):
    """Analyze collision outcome."""
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)
    
    if len(track_L) < 20:
        return None
    
    times = track_L[:, 0]
    pos_L = track_L[:, 1]
    pos_R = track_R[:, 1]
    
    sep0 = abs(pos_L[0] - pos_R[0])
    separation = np.abs(pos_L - pos_R)
    
    coll_idx = np.argmin(separation)
    t_coll = times[coll_idx]
    min_sep = separation[coll_idx]
    final_sep = separation[-1]
    
    # Determine if bound (final peak count = 1)
    final_peaks = sim.count_peaks()
    is_bound = (final_peaks == 1)
    
    # Find release time if scattered (when sep starts increasing after min)
    release_time = None
    if not is_bound and coll_idx < len(separation) - 10:
        for i in range(coll_idx, len(separation)):
            if separation[i] > min_sep + 5:  # 5 lattice units threshold
                release_time = times[i] - t_coll
                break
    
    # Measure incoming velocity
    pre_coll_mask = times < t_coll * 0.8
    if np.sum(pre_coll_mask) >= 5:
        t_pre = times[pre_coll_mask]
        res_L = linregress(t_pre, pos_L[pre_coll_mask])
        res_R = linregress(t_pre, pos_R[pre_coll_mask])
        v_in_L = res_L.slope
        v_in_R = res_R.slope
    else:
        v_in_L = v_in_R = 0
    
    return {
        'min_sep': min_sep,
        'final_sep': final_sep,
        'sep0': sep0,
        't_coll': t_coll,
        'is_bound': is_bound,
        'final_peaks': final_peaks,
        'release_time': release_time,
        'v_in_L': v_in_L,
        'v_in_R': v_in_R,
        'v_rel': v_in_L - v_in_R,
    }


def run_trial(g, v_kick, offset, rng, theta_noise=1e-3, steps=800):
    """Run single trial with small perturbation."""
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick, mass_L=10.0, mass_R=10.0)
    
    # Apply small theta noise
    if theta_noise > 0:
        sim.theta += rng.normal(0, theta_noise, size=sim.N)
        sim.theta = np.mod(sim.theta, 2*np.pi)
    
    for _ in range(steps):
        sim.step()
    
    return analyze_run(sim)


def run_experiments():
    print("="*70)
    print("DET v5 PHASE BOUNDARY EXPERIMENTS")
    print("="*70)
    
    rng = np.random.default_rng(42)
    n_trials = 10
    
    # =========================================================================
    # EXPERIMENT A: Boundary scan around capture threshold
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT A: Capture Threshold Scan (g=1.5, offset=15)")
    print("="*70)
    print(f"v_kick = 1.6 to 2.4 (step 0.1), {n_trials} trials each\n")
    
    print(f"{'v_kick':<8} | {'bind_frac':<10} | {'mean_peaks':<12} | {'mean_release_t':<14} | {'mean_min_sep':<12}")
    print("-" * 70)
    
    exp_a_results = []
    v_kicks_a = np.arange(1.6, 2.5, 0.1)
    
    for v_kick in v_kicks_a:
        bind_count = 0
        peaks_list = []
        release_times = []
        min_seps = []
        
        for trial in range(n_trials):
            res = run_trial(g=1.5, v_kick=v_kick, offset=15, rng=rng)
            if res:
                if res['is_bound']:
                    bind_count += 1
                peaks_list.append(res['final_peaks'])
                if res['release_time'] is not None:
                    release_times.append(res['release_time'])
                min_seps.append(res['min_sep'])
        
        bind_frac = bind_count / n_trials
        mean_peaks = np.mean(peaks_list) if peaks_list else 0
        mean_release = np.mean(release_times) if release_times else np.nan
        mean_min_sep = np.mean(min_seps) if min_seps else np.nan
        
        print(f"{v_kick:<8.2f} | {bind_frac:<10.2f} | {mean_peaks:<12.2f} | "
              f"{mean_release:<14.2f}" if not np.isnan(mean_release) else f"{v_kick:<8.2f} | {bind_frac:<10.2f} | {mean_peaks:<12.2f} | {'N/A':<14}",
              f" | {mean_min_sep:<12.2f}" if not np.isnan(mean_min_sep) else "")
        
        exp_a_results.append({
            'v_kick': v_kick,
            'bind_frac': bind_frac,
            'mean_peaks': mean_peaks,
            'mean_release': mean_release,
            'mean_min_sep': mean_min_sep,
        })
    
    # =========================================================================
    # EXPERIMENT B: g scan at two energies
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT B: g Scan at Low vs High Energy (offset=15)")
    print("="*70)
    print(f"Low energy: v_kick=1.0, High energy: v_kick=2.5")
    print(f"g = 0.8 to 3.0 (step 0.2), {n_trials} trials each\n")
    
    print(f"{'g':<6} | {'bind_low':<10} | {'bind_high':<10} | {'peaks_low':<10} | {'peaks_high':<10}")
    print("-" * 55)
    
    exp_b_results = []
    g_vals_b = np.arange(0.8, 3.2, 0.2)
    
    for g in g_vals_b:
        # Low energy
        bind_low = 0
        peaks_low = []
        for trial in range(n_trials):
            res = run_trial(g=g, v_kick=1.0, offset=15, rng=rng)
            if res:
                if res['is_bound']:
                    bind_low += 1
                peaks_low.append(res['final_peaks'])
        
        # High energy
        bind_high = 0
        peaks_high = []
        for trial in range(n_trials):
            res = run_trial(g=g, v_kick=2.5, offset=15, rng=rng)
            if res:
                if res['is_bound']:
                    bind_high += 1
                peaks_high.append(res['final_peaks'])
        
        bind_frac_low = bind_low / n_trials
        bind_frac_high = bind_high / n_trials
        mean_peaks_low = np.mean(peaks_low) if peaks_low else 0
        mean_peaks_high = np.mean(peaks_high) if peaks_high else 0
        
        print(f"{g:<6.2f} | {bind_frac_low:<10.2f} | {bind_frac_high:<10.2f} | "
              f"{mean_peaks_low:<10.2f} | {mean_peaks_high:<10.2f}")
        
        exp_b_results.append({
            'g': g,
            'bind_frac_low': bind_frac_low,
            'bind_frac_high': bind_frac_high,
            'mean_peaks_low': mean_peaks_low,
            'mean_peaks_high': mean_peaks_high,
        })
    
    # =========================================================================
    # EXPERIMENT C: Offset phase boundary
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT C: Offset Phase Boundary (g=1.5, v_kick=2.0)")
    print("="*70)
    print(f"Offsets: 15, 18, 20, 22, 25, 30, {n_trials} trials each\n")
    
    print(f"{'offset':<8} | {'sep0':<6} | {'bind_frac':<10} | {'mean_min_sep':<12} | {'mean_peaks':<10}")
    print("-" * 55)
    
    exp_c_results = []
    offsets_c = [15, 18, 20, 22, 25, 30]
    
    for offset in offsets_c:
        bind_count = 0
        min_seps = []
        peaks_list = []
        
        for trial in range(n_trials):
            res = run_trial(g=1.5, v_kick=2.0, offset=offset, rng=rng)
            if res:
                if res['is_bound']:
                    bind_count += 1
                min_seps.append(res['min_sep'])
                peaks_list.append(res['final_peaks'])
        
        bind_frac = bind_count / n_trials
        mean_min_sep = np.mean(min_seps) if min_seps else np.nan
        mean_peaks = np.mean(peaks_list) if peaks_list else 0
        
        print(f"{offset:<8} | {2*offset:<6} | {bind_frac:<10.2f} | {mean_min_sep:<12.2f} | {mean_peaks:<10.2f}")
        
        exp_c_results.append({
            'offset': offset,
            'sep0': 2 * offset,
            'bind_frac': bind_frac,
            'mean_min_sep': mean_min_sep,
            'mean_peaks': mean_peaks,
        })
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Experiment A: Capture threshold
    ax = axes[0, 0]
    v_kicks = [r['v_kick'] for r in exp_a_results]
    bind_fracs = [r['bind_frac'] for r in exp_a_results]
    
    ax.plot(v_kicks, bind_fracs, 'ko-', markersize=10, linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.fill_between(v_kicks, 0, bind_fracs, alpha=0.3)
    ax.set_xlabel("Velocity Kick (v_kick)")
    ax.set_ylabel("Bind Fraction")
    ax.set_title("Exp A: Capture Threshold (g=1.5, offset=15)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Experiment B: g scan
    ax = axes[0, 1]
    g_vals = [r['g'] for r in exp_b_results]
    bind_low = [r['bind_frac_low'] for r in exp_b_results]
    bind_high = [r['bind_frac_high'] for r in exp_b_results]
    
    ax.plot(g_vals, bind_low, 'bo-', markersize=8, linewidth=2, label='Low E (v=1.0)')
    ax.plot(g_vals, bind_high, 'rs-', markersize=8, linewidth=2, label='High E (v=2.5)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Bind Fraction")
    ax.set_title("Exp B: g Scan at Two Energies (offset=15)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Experiment C: Offset scan
    ax = axes[1, 0]
    offsets = [r['offset'] for r in exp_c_results]
    bind_fracs_c = [r['bind_frac'] for r in exp_c_results]
    min_seps_c = [r['mean_min_sep'] for r in exp_c_results]
    
    ax.plot(offsets, bind_fracs_c, 'ko-', markersize=10, linewidth=2, label='Bind fraction')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Offset (initial sep = 2*offset)")
    ax.set_ylabel("Bind Fraction")
    ax.set_title("Exp C: Offset Phase Boundary (g=1.5, v=2.0)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(offsets, min_seps_c, 'gs--', markersize=8, alpha=0.7, label='min_sep')
    ax2.set_ylabel("Mean min_sep", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')
    
    # 4. Phase diagram summary
    ax = axes[1, 1]
    
    # Create a simple phase diagram from Exp B
    g_mesh = np.array(g_vals)
    v_mesh = np.array([1.0, 2.5])
    bind_matrix = np.array([[r['bind_frac_low'], r['bind_frac_high']] for r in exp_b_results]).T
    
    im = ax.imshow(bind_matrix, aspect='auto', cmap='RdYlGn', origin='lower',
                   extent=[g_mesh.min()-0.1, g_mesh.max()+0.1, 0.5, 3.0],
                   vmin=0, vmax=1)
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Velocity Kick")
    ax.set_title("Phase Diagram: Bind Fraction")
    plt.colorbar(im, ax=ax, label='Bind Fraction')
    
    # Mark the two scan lines
    ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.7, label='Low E scan')
    ax.axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='High E scan')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/det_v5_phase_boundary.png', dpi=150)
    plt.savefig('/mnt/user-data/outputs/det_v5_phase_boundary.png', dpi=150)
    print("\nSaved to det_v5_phase_boundary.png")
    
    return exp_a_results, exp_b_results, exp_c_results


if __name__ == "__main__":
    exp_a, exp_b, exp_c = run_experiments()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Find threshold from Exp A
    for i, r in enumerate(exp_a):
        if r['bind_frac'] >= 0.5:
            print(f"\nExp A: Capture threshold ≈ v_kick = {r['v_kick']:.2f}")
            print(f"  (First point with bind_frac ≥ 50%)")
            break
    
    # Analyze Exp B
    low_e_bind = np.mean([r['bind_frac_low'] for r in exp_b])
    high_e_bind = np.mean([r['bind_frac_high'] for r in exp_b])
    print(f"\nExp B: Low energy (v=1.0) mean bind: {low_e_bind:.2f}")
    print(f"       High energy (v=2.5) mean bind: {high_e_bind:.2f}")
    if high_e_bind > low_e_bind:
        print("  → Higher energy favors binding (surprising!)")
    else:
        print("  → Lower energy favors binding (expected for capture)")
    
    # Analyze Exp C
    for r in exp_c:
        if r['bind_frac'] < 0.5:
            print(f"\nExp C: Fusion disappears around offset = {r['offset']}")
            print(f"  (First point with bind_frac < 50%)")
            break
    else:
        print(f"\nExp C: Fusion persists even at offset = {exp_c[-1]['offset']}")
