import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import label
from collections import Counter

"""
DET v5 COLLIDER - FORCED CONTACT TEST
=====================================
Option 1: Reduce offset from 40 to 15 to force actual collision.

Initial separation = 2 * offset = 30 (instead of 80)
With approach of 10-20 units, particles should actually overlap.
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
        
        if int(self.time/self.k.DT) % 5 == 0:
            self.history_F.append(self.F.copy())


def analyze_single_run(sim, post_dt=5.0):
    """Analyze collision using single-run ghost reference."""
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)
    
    if len(track_L) < 20:
        return None
    
    times = track_L[:, 0]
    pos_L = track_L[:, 1]
    pos_R = track_R[:, 1]
    
    # Initial separation
    sep0 = abs(pos_L[0] - pos_R[0])
    
    # Find collision time (minimum separation)
    separation = np.abs(pos_L - pos_R)
    coll_idx = np.argmin(separation)
    t_coll = times[coll_idx]
    min_sep = separation[coll_idx]
    
    # Approach distance
    approach = sep0 - min_sep
    
    # Pre-collision window
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
    
    # Early-out measurement (at t* = t_coll + post_dt)
    t_star = t_coll + post_dt
    star_mask = times <= t_star
    if np.sum(star_mask) > 0:
        star_idx = np.sum(star_mask) - 1
        x_star_L = pos_L[star_idx]
        x_star_R = pos_R[star_idx]
        t_actual_star = times[star_idx]
    else:
        x_star_L = pos_L[-1]
        x_star_R = pos_R[-1]
        t_actual_star = times[-1]
    
    # Ghost positions at t*
    x_ghost_star_L = pos_L[0] + v_in_L * t_actual_star
    x_ghost_star_R = pos_R[0] + v_in_R * t_actual_star
    
    shift_L_star = x_star_L - x_ghost_star_L
    shift_R_star = x_star_R - x_ghost_star_R
    
    # Final positions
    t_final = times[-1]
    x_final_L = pos_L[-1]
    x_final_R = pos_R[-1]
    
    x_ghost_L = pos_L[0] + v_in_L * t_final
    x_ghost_R = pos_R[0] + v_in_R * t_final
    
    shift_L = x_final_L - x_ghost_L
    shift_R = x_final_R - x_ghost_R
    
    # Determine outcome
    final_sep = separation[-1]
    if min_sep < 5:
        if final_sep < 10:
            outcome = "FUSION"
        else:
            outcome = "SCATTER"
    elif min_sep < 15:
        outcome = "CLOSE"
    else:
        outcome = "FAR"
    
    return {
        'v_in_L': v_in_L,
        'v_in_R': v_in_R,
        'v_rel': v_in_L - v_in_R,
        'r2_L': r2_L,
        'r2_R': r2_R,
        't_coll': t_coll,
        'min_sep': min_sep,
        'final_sep': final_sep,
        'approach': approach,
        'sep0': sep0,
        'shift_L_star': shift_L_star,
        'shift_R_star': shift_R_star,
        'avg_shift_star': (shift_L_star + shift_R_star) / 2,
        'shift_L': shift_L,
        'shift_R': shift_R,
        'avg_shift': (shift_L + shift_R) / 2,
        'outcome': outcome,
        'times': times,
        'pos_L': pos_L,
        'pos_R': pos_R,
    }


def run_forced_contact_experiments():
    print("="*70)
    print("DET v5 COLLIDER - FORCED CONTACT (offset=15)")
    print("="*70)
    print("""
KEY CHANGE: offset = 15 (instead of 40)
  - Initial separation = 30 lattice units (instead of 80)
  - With soliton width=5, particles initially OVERLAP
  - This forces actual interaction regardless of propagation speed
""")
    
    # ===== OFFSET COMPARISON =====
    print("\n" + "="*70)
    print("OFFSET COMPARISON (g=1.5, symmetric)")
    print("="*70)
    print(f"\n{'offset':<8} | {'sep0':<6} | {'min_sep':<8} | {'approach':<10} | {'v_rel':<8} | {'Outcome':<10} | {'Δx_avg':<8}")
    print("-" * 80)
    
    for offset in [40, 30, 20, 15, 10]:
        k = DETConstants(g=1.5)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(offset=offset, v_kick=2.0, mass_L=10.0, mass_R=10.0)
        
        for _ in range(800):
            sim.step()
        
        res = analyze_single_run(sim)
        if res:
            print(f"{offset:<8} | {res['sep0']:<6.0f} | {res['min_sep']:<8.2f} | {res['approach']:<10.2f} | "
                  f"{res['v_rel']:<8.4f} | {res['outcome']:<10} | {res['avg_shift']:<8.2f}")
    
    # ===== SYMMETRIC CASE with offset=15 =====
    print("\n" + "="*70)
    print("SYMMETRIC CASE (m_L = m_R = 10, offset=15)")
    print("="*70)
    
    print(f"\n{'g':<6} | {'v_rel':<8} | {'min_sep':<8} | {'approach':<10} | {'Outcome':<10} | {'Δx_L':<8} | {'Δx_R':<8} | {'Δx_avg':<8}")
    print("-" * 90)
    
    sym_results = []
    for g in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]:
        k = DETConstants(g=g)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(offset=15, v_kick=2.0, mass_L=10.0, mass_R=10.0)
        
        for _ in range(800):
            sim.step()
        
        res = analyze_single_run(sim)
        if res:
            res['g'] = g
            sym_results.append(res)
            print(f"{g:<6.2f} | {res['v_rel']:<8.4f} | {res['min_sep']:<8.2f} | {res['approach']:<10.2f} | "
                  f"{res['outcome']:<10} | {res['shift_L']:<8.2f} | {res['shift_R']:<8.2f} | {res['avg_shift']:<8.2f}")
    
    # ===== ASYMMETRIC CASE with offset=15 =====
    print("\n" + "="*70)
    print("ASYMMETRIC CASE (m_L = 10, m_R = 14, offset=15)")
    print("="*70)
    
    print(f"\n{'g':<6} | {'v_rel':<8} | {'min_sep':<8} | {'approach':<10} | {'Outcome':<10} | {'Δx_L':<8} | {'Δx_R':<8} | {'Δx_avg':<8}")
    print("-" * 90)
    
    asym_results = []
    for g in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]:
        k = DETConstants(g=g)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(offset=15, v_kick=2.0, mass_L=10.0, mass_R=14.0)
        
        for _ in range(800):
            sim.step()
        
        res = analyze_single_run(sim)
        if res:
            res['g'] = g
            asym_results.append(res)
            print(f"{g:<6.2f} | {res['v_rel']:<8.4f} | {res['min_sep']:<8.2f} | {res['approach']:<10.2f} | "
                  f"{res['outcome']:<10} | {res['shift_L']:<8.2f} | {res['shift_R']:<8.2f} | {res['avg_shift']:<8.2f}")
    
    # ===== ENERGY SCAN with offset=15 =====
    print("\n" + "="*70)
    print("ENERGY SCAN (g=1.5, symmetric, offset=15)")
    print("="*70)
    
    print(f"\n{'v_kick':<8} | {'v_rel':<8} | {'min_sep':<8} | {'approach':<10} | {'Outcome':<10} | {'Δx_avg':<8}")
    print("-" * 65)
    
    energy_results = []
    for v_kick in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        k = DETConstants(g=1.5)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(offset=15, v_kick=v_kick, mass_L=10.0, mass_R=10.0)
        
        for _ in range(800):
            sim.step()
        
        res = analyze_single_run(sim)
        if res:
            res['v_kick'] = v_kick
            energy_results.append(res)
            print(f"{v_kick:<8.2f} | {res['v_rel']:<8.4f} | {res['min_sep']:<8.2f} | {res['approach']:<10.2f} | "
                  f"{res['outcome']:<10} | {res['avg_shift']:<8.2f}")
    
    # ===== VISUALIZATION =====
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Space-time diagram (symmetric, g=1.5)
    ax = axes[0, 0]
    k = DETConstants(g=1.5)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=15, v_kick=2.0, mass_L=10.0, mass_R=10.0)
    for _ in range(800):
        sim.step()
    
    if len(sim.history_F) > 0:
        history = np.array(sim.history_F)
        im = ax.imshow(history, aspect='auto', cmap='plasma', origin='lower',
                       extent=[0, 200, 0, sim.time])
        
        res_vis = analyze_single_run(sim)
        if res_vis:
            ax.plot(res_vis['pos_L'] % 200, res_vis['times'], 'c.', markersize=1, alpha=0.8)
            ax.plot(res_vis['pos_R'] % 200, res_vis['times'], 'w.', markersize=1, alpha=0.8)
        
        ax.set_title(f"Symmetric (g=1.5, offset=15)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Time")
        plt.colorbar(im, ax=ax)
    
    # 2. Space-time diagram (asymmetric, g=1.5)
    ax = axes[0, 1]
    k = DETConstants(g=1.5)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=15, v_kick=2.0, mass_L=10.0, mass_R=14.0)
    for _ in range(800):
        sim.step()
    
    if len(sim.history_F) > 0:
        history = np.array(sim.history_F)
        im = ax.imshow(history, aspect='auto', cmap='plasma', origin='lower',
                       extent=[0, 200, 0, sim.time])
        
        res_vis = analyze_single_run(sim)
        if res_vis:
            ax.plot(res_vis['pos_L'] % 200, res_vis['times'], 'c.', markersize=1, alpha=0.8)
            ax.plot(res_vis['pos_R'] % 200, res_vis['times'], 'w.', markersize=1, alpha=0.8)
        
        ax.set_title(f"Asymmetric (g=1.5, offset=15)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Time")
        plt.colorbar(im, ax=ax)
    
    # 3. Trajectory comparison
    ax = axes[0, 2]
    if len(sym_results) > 0:
        best = [r for r in sym_results if r['g'] == 1.5]
        if best:
            best = best[0]
            ax.plot(best['times'], best['pos_L'], 'b-', linewidth=2, label='L')
            ax.plot(best['times'], best['pos_R'], 'r-', linewidth=2, label='R')
            
            # Ghost
            t_arr = np.linspace(0, best['times'][-1], 100)
            ax.plot(t_arr, best['pos_L'][0] + best['v_in_L']*t_arr, 'b--', alpha=0.5)
            ax.plot(t_arr, best['pos_R'][0] + best['v_in_R']*t_arr, 'r--', alpha=0.5)
            
            ax.axvline(x=best['t_coll'], color='g', linestyle=':', label='Collision')
            ax.set_xlabel("Time")
            ax.set_ylabel("Position")
            ax.set_title(f"Symmetric Trajectories (g=1.5)")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 4. Δx vs g
    ax = axes[1, 0]
    g_sym = [r['g'] for r in sym_results]
    dx_sym = [r['avg_shift'] for r in sym_results]
    dx_L_sym = [r['shift_L'] for r in sym_results]
    dx_R_sym = [r['shift_R'] for r in sym_results]
    
    ax.plot(g_sym, dx_sym, 'ko-', markersize=10, linewidth=2, label='Symmetric Δx_avg')
    ax.plot(g_sym, dx_L_sym, 'b^--', markersize=6, alpha=0.5, label='Δx_L')
    ax.plot(g_sym, dx_R_sym, 'rv--', markersize=6, alpha=0.5, label='Δx_R')
    
    if len(asym_results) > 0:
        g_asym = [r['g'] for r in asym_results]
        dx_asym = [r['avg_shift'] for r in asym_results]
        ax.plot(g_asym, dx_asym, 'gs-', markersize=10, linewidth=2, label='Asymmetric Δx_avg')
    
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Position Shift Δx")
    ax.set_title("Position Shift vs Coupling (offset=15)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. min_sep vs g
    ax = axes[1, 1]
    min_sep_sym = [r['min_sep'] for r in sym_results]
    approach_sym = [r['approach'] for r in sym_results]
    
    ax.plot(g_sym, min_sep_sym, 'ko-', markersize=10, linewidth=2, label='min_sep')
    ax.plot(g_sym, approach_sym, 'bs--', markersize=8, label='approach')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Distance (lattice units)")
    ax.set_title("Minimum Separation & Approach")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Δx vs energy
    ax = axes[1, 2]
    if len(energy_results) > 0:
        v_kicks = [r['v_kick'] for r in energy_results]
        dx_energy = [r['avg_shift'] for r in energy_results]
        min_sep_energy = [r['min_sep'] for r in energy_results]
        
        ax.plot(v_kicks, dx_energy, 'ko-', markersize=10, linewidth=2, label='Δx_avg')
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Velocity Kick")
        ax.set_ylabel("Position Shift Δx")
        ax.set_title("Position Shift vs Energy (offset=15)")
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(v_kicks, min_sep_energy, 'rs--', markersize=8, alpha=0.7)
        ax2.set_ylabel("min_sep", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    plt.savefig('/home/claude/det_v5_forced_contact.png', dpi=150)
    plt.savefig('/mnt/user-data/outputs/det_v5_forced_contact.png', dpi=150)
    print("\nSaved to det_v5_forced_contact.png")
    
    return sym_results, asym_results, energy_results


if __name__ == "__main__":
    sym, asym, energy = run_forced_contact_experiments()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
FORCED CONTACT TEST (offset=15):
  - Initial separation = 30 lattice units
  - Soliton width = 5, so particles initially overlap
  - Tests whether DET produces scattering when contact is guaranteed

Key metrics to check:
  - min_sep: Should be much smaller than with offset=40
  - approach: Should be comparable to sep0 (full approach)
  - Outcome: Look for CLOSE or FUSION instead of FAR
  - Δx: Should show systematic variation with g
""")
