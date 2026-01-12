import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import label

"""
DET v5 COLLIDER - MOMENTUM KICK FIX
===================================
The issue: θ += v_kick * x * envelope creates a large absolute phase
but the GRADIENT is what drives flow.

Solution: Create a proper phase gradient within the soliton region.
For a soliton moving right, we want dθ/dx = k > 0 (locally).

New kick formula:
  θ += v_kick * (x - center) * envelope(x)
  
This creates:
  - Zero phase at soliton center
  - Positive gradient (moving right) or negative gradient (moving left)
  - Gradient confined to soliton region
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
        
        # Identity-propagated tracking
        self.track_L = []
        self.track_R = []
        self.last_pos_L = None
        self.last_pos_R = None

    def initialize_soliton(self, position, width, mass, velocity_kick=0.0):
        """
        Initialize soliton with CENTERED phase gradient.
        
        θ += v_kick * (x - center) * envelope(x)
        
        This creates a local phase gradient (momentum) centered on the soliton.
        """
        x = np.arange(self.N)
        
        # Periodic distance from soliton center
        dx = x - position
        # Wrap to [-N/2, N/2]
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        
        # Envelope based on absolute distance
        abs_dx = np.abs(dx)
        envelope = np.exp(-0.5 * (abs_dx / width)**2)
        
        # Add mass
        self.F += mass * envelope
        
        # CENTERED MOMENTUM KICK:
        # Phase gradient proportional to displacement from center
        # This creates dθ/dx = v_kick within the soliton
        self.theta += velocity_kick * dx * envelope
        
        # Pre-seed metric ASYMMETRICALLY for directional motion
        if velocity_kick > 0:
            # Moving right: stronger C_right
            self.C_right += 0.9 * envelope
            self.C_left  += 0.3 * envelope
        elif velocity_kick < 0:
            # Moving left: stronger C_left
            self.C_right += 0.3 * envelope
            self.C_left  += 0.9 * envelope
        else:
            self.C_right += 0.6 * envelope
            self.C_left  += 0.6 * envelope
            
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)

    def initialize_collider_event(self, offset=40, width=5.0, mass=10.0, v_kick=1.0):
        """Initialize two solitons with momentum kicks."""
        center = self.N // 2
        self.start_pos_L = center - offset
        self.start_pos_R = center + offset
        
        # Left particle: positive kick (moving right)
        self.initialize_soliton(self.start_pos_L, width, mass, velocity_kick=v_kick)
        # Right particle: negative kick (moving left)
        self.initialize_soliton(self.start_pos_R, width, mass, velocity_kick=-v_kick)
        
        # Initialize tracking
        self.last_pos_L = float(self.start_pos_L)
        self.last_pos_R = float(self.start_pos_R)
        self.track_L = [(0.0, self.last_pos_L)]
        self.track_R = [(0.0, self.last_pos_R)]

    def find_blob_coms(self):
        """Find center of mass of all blobs above threshold."""
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
        """Position unwrapping for periodic boundary."""
        if last_pos is None:
            return new_pos
        
        delta = new_pos - (last_pos % self.N)
        
        if delta > self.N / 2:
            delta -= self.N
        elif delta < -self.N / 2:
            delta += self.N
        
        return last_pos + delta

    def update_tracking(self):
        """Identity propagation by nearest-neighbor in time."""
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


def analyze(sim):
    """Analyze collision from tracked trajectories."""
    if len(sim.track_L) < 20 or len(sim.track_R) < 20:
        return {"status": "Error"}
    
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)
    
    times = track_L[:, 0]
    pos_L = track_L[:, 1]
    pos_R = track_R[:, 1]
    
    # Early phase: first 25%
    t_early = sim.time * 0.25
    early_mask = times < t_early
    
    if np.sum(early_mask) < 10:
        return {"status": "Error"}
    
    t_early_arr = times[early_mask]
    pL_early = pos_L[early_mask]
    pR_early = pos_R[early_mask]
    
    res_L = linregress(t_early_arr, pL_early)
    res_R = linregress(t_early_arr, pR_early)
    
    v_L = res_L.slope
    v_R = res_R.slope
    v_rel = v_L - v_R
    
    r2_L = res_L.rvalue**2
    r2_R = res_R.rvalue**2
    
    separation = np.abs(pos_L - pos_R)
    coll_idx = np.argmin(separation)
    t_coll = times[coll_idx]
    min_sep = separation[coll_idx]
    
    final_sep = separation[-1]
    if min_sep < 15 and final_sep < 20:
        status = "FUSION"
    elif final_sep > 40:
        status = "SCATTERING"
    else:
        status = "PARTIAL"
    
    t_final = sim.time
    x_ghost_L = sim.start_pos_L + v_L * t_final
    x_ghost_R = sim.start_pos_R + v_R * t_final
    
    shift_L = pos_L[-1] - x_ghost_L
    shift_R = pos_R[-1] - x_ghost_R
    
    return {
        "status": status,
        "v_L": v_L,
        "v_R": v_R,
        "v_rel": v_rel,
        "r2_L": r2_L,
        "r2_R": r2_R,
        "t_coll": t_coll,
        "min_sep": min_sep,
        "final_sep": final_sep,
        "shift_L": shift_L,
        "shift_R": shift_R,
        "avg_shift": (shift_L + shift_R) / 2,
        "times": times,
        "pos_L": pos_L,
        "pos_R": pos_R,
    }


def run_momentum_suite():
    print("="*70)
    print("DET v5 COLLIDER - CENTERED MOMENTUM KICK")
    print("="*70)
    print("""
MOMENTUM KICK FIX:
  θ += v_kick * (x - center) * envelope(x)
  
This creates:
  - dθ/dx = v_kick within the soliton (constant gradient)
  - Zero absolute phase at soliton center
  - Gradient confined to soliton region
  
ALSO: Asymmetric metric seeding (C_right vs C_left)
""")
    
    # --- Single particle calibration ---
    print("\n[Calibration] Single Soliton with Centered Momentum Kick")
    print("-" * 55)
    
    for v_kick in [0.5, 1.0, 2.0, 3.0]:
        k = DETConstants(g=1.5)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_soliton(50, width=5.0, mass=10.0, velocity_kick=v_kick)
        sim.last_pos_L = 50.0
        sim.track_L = [(0.0, 50.0)]
        
        for _ in range(400): sim.step()
        
        track = np.array(sim.track_L)
        if len(track) > 50:
            res = linregress(track[:100, 0], track[:100, 1])
            v_meas = res.slope
            r2 = res.rvalue**2
            disp = track[-1,1] - track[0,1]
            print(f"  v_kick={v_kick:.1f}: v_meas={v_meas:+.4f}, R²={r2:.3f}, Δx={disp:+.1f}")
    
    # --- Collision experiments ---
    print("\n" + "="*70)
    print("[Exp 1] Phase Shift vs Coupling (v_kick=2.0)")
    print(f"{'g':<6} | {'v_L':<8} | {'v_R':<8} | {'v_rel':<8} | {'R²':<6} | {'Status':<10} | {'Shift':<8}")
    print("-" * 75)
    
    results = []
    for g in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]:
        k = DETConstants(g=g)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(v_kick=2.0, offset=40, mass=10.0)
        
        for _ in range(600): sim.step()
        
        res = analyze(sim)
        res['g'] = g
        results.append(res)
        
        if res['status'] != 'Error':
            r2_avg = (res['r2_L'] + res['r2_R']) / 2
            print(f"{g:<6.2f} | {res['v_L']:<8.4f} | {res['v_R']:<8.4f} | "
                  f"{res['v_rel']:<8.4f} | {r2_avg:<6.3f} | {res['status']:<10} | {res['avg_shift']:<8.2f}")
        else:
            print(f"{g:<6.2f} | Error")
    
    # --- Energy scan ---
    print("\n" + "="*70)
    print("[Exp 2] Phase Shift vs Collision Energy (g=1.5)")
    print(f"{'v_kick':<8} | {'v_L':<8} | {'v_R':<8} | {'v_rel':<8} | {'Status':<10} | {'Shift':<8}")
    print("-" * 65)
    
    energy_results = []
    for v_kick in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        k = DETConstants(g=1.5)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(v_kick=v_kick, offset=40, mass=10.0)
        
        for _ in range(600): sim.step()
        
        res = analyze(sim)
        res['v_kick'] = v_kick
        energy_results.append(res)
        
        if res['status'] != 'Error':
            print(f"{v_kick:<8.2f} | {res['v_L']:<8.4f} | {res['v_R']:<8.4f} | "
                  f"{res['v_rel']:<8.4f} | {res['status']:<10} | {res['avg_shift']:<8.2f}")
    
    # --- Visualization ---
    print("\n[Creating visualization...]")
    
    k = DETConstants(g=1.5)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(v_kick=2.0, offset=40, mass=10.0)
    for _ in range(600): sim.step()
    
    res = analyze(sim)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Space-time
    ax = axes[0, 0]
    history = np.array(sim.history_F)
    im = ax.imshow(history, aspect='auto', cmap='plasma', origin='lower',
                   extent=[0, sim.N, 0, sim.time])
    
    pos_L_disp = res['pos_L'] % sim.N
    pos_R_disp = res['pos_R'] % sim.N
    ax.plot(pos_L_disp, res['times'], 'c.', markersize=1, alpha=0.8, label='Track L')
    ax.plot(pos_R_disp, res['times'], 'w.', markersize=1, alpha=0.8, label='Track R')
    
    ax.set_title(f"Space-Time (g={sim.k.g:.1f}, v_kick=2.0)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Time")
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax)
    
    # 2. Unwrapped trajectories
    ax = axes[0, 1]
    ax.plot(res['times'], res['pos_L'], 'b-', linewidth=2, label='Particle L')
    ax.plot(res['times'], res['pos_R'], 'r-', linewidth=2, label='Particle R')
    
    t_arr = np.linspace(0, sim.time, 100)
    ax.plot(t_arr, sim.start_pos_L + res['v_L']*t_arr, 'b--', alpha=0.5, 
            label=f'Ghost L (v={res["v_L"]:.3f})')
    ax.plot(t_arr, sim.start_pos_R + res['v_R']*t_arr, 'r--', alpha=0.5,
            label=f'Ghost R (v={res["v_R"]:.3f})')
    
    ax.axvline(x=res['t_coll'], color='g', linestyle=':', alpha=0.7, label='Collision')
    ax.set_xlabel("Time")
    ax.set_ylabel("Position (unwrapped)")
    ax.set_title(f"v_L={res['v_L']:.4f}, v_R={res['v_R']:.4f}, v_rel={res['v_rel']:.4f}")
    ax.legend()
    
    # 3. Phase shift vs g
    ax = axes[1, 0]
    valid = [r for r in results if r['status'] != 'Error']
    if valid:
        g_vals = [r['g'] for r in valid]
        shift_vals = [r['avg_shift'] for r in valid]
        vrel_vals = [r['v_rel'] for r in valid]
        
        ax.plot(g_vals, shift_vals, 'ko-', markersize=10, linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Coupling g")
        ax.set_ylabel("Phase Shift (lattice units)")
        ax.set_title("Phase Shift vs Coupling")
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(g_vals, vrel_vals, 'rs--', markersize=8)
        ax2.set_ylabel("v_rel", color='red')
    
    # 4. Phase shift vs energy
    ax = axes[1, 1]
    valid_e = [r for r in energy_results if r['status'] != 'Error']
    if valid_e:
        vkick_vals = [r['v_kick'] for r in valid_e]
        shift_vals = [r['avg_shift'] for r in valid_e]
        vrel_vals = [r['v_rel'] for r in valid_e]
        
        ax.plot(vkick_vals, shift_vals, 'ko-', markersize=10, linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.set_xlabel("Velocity Kick")
        ax.set_ylabel("Phase Shift")
        ax.set_title("Phase Shift vs Energy")
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(vkick_vals, vrel_vals, 'rs--', markersize=8)
        ax2.set_ylabel("v_rel", color='red')
    
    plt.tight_layout()
    plt.savefig('./det_v5_momentum.png', dpi=150)
    print("\nSaved to det_v5_momentum.png")
    
    return results, energy_results


if __name__ == "__main__":
    results, energy_results = run_momentum_suite()








