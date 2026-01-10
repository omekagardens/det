import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import label

"""
DET v5 COLLIDER - PROPER SCATTERING OBSERVABLES
===============================================
Fixes based on analysis:

1. TIME DELAY OBSERVABLE (Wigner-style)
   - Place detector planes on each side
   - Measure crossing time t_out after collision
   - Compare to reference (free propagation) t_free
   - Δt = t_out - t_free is symmetry-robust

2. REFERENCE BASELINE
   - Run same dynamics but with no collision
   - Use large separation so particles don't meet
   - This gives model-consistent free propagation

3. SYMMETRY BREAKING
   - Unequal masses: m_L ≠ m_R
   - Makes outgoing packets distinguishable
   - Recovers clean position-phase shift Δx
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
        self.mass_L = 0
        self.mass_R = 0
        
        # Tracking
        self.track_L = []
        self.track_R = []
        self.last_pos_L = None
        self.last_pos_R = None
        
        # Detector planes
        self.detector_L = None  # Position of left detector
        self.detector_R = None  # Position of right detector
        self.crossing_L = None  # Time when L crosses its detector
        self.crossing_R = None  # Time when R crosses its detector

    def initialize_soliton(self, position, width, mass, velocity_kick=0.0):
        """Initialize with centered momentum kick."""
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

    def initialize_collider_event(self, offset=40, width=5.0, mass_L=10.0, mass_R=10.0, 
                                   v_kick=2.0, detector_offset=20):
        """
        Initialize collider with optional mass asymmetry.
        
        detector_offset: how far outside starting positions to place detectors
        """
        center = self.N // 2
        self.start_pos_L = center - offset
        self.start_pos_R = center + offset
        self.mass_L = mass_L
        self.mass_R = mass_R
        
        # Left particle: positive kick (moving right)
        self.initialize_soliton(self.start_pos_L, width, mass_L, velocity_kick=v_kick)
        # Right particle: negative kick (moving left)
        self.initialize_soliton(self.start_pos_R, width, mass_R, velocity_kick=-v_kick)
        
        # Initialize tracking
        self.last_pos_L = float(self.start_pos_L)
        self.last_pos_R = float(self.start_pos_R)
        self.track_L = [(0.0, self.last_pos_L)]
        self.track_R = [(0.0, self.last_pos_R)]
        
        # Set up detector planes
        # After collision, L should exit to the LEFT (bounced back) or continue RIGHT
        # R should exit to the RIGHT (bounced back) or continue LEFT
        # For attractive interaction: they pass through → L ends up on RIGHT side
        # Place detectors where particles would end up after passing through
        self.detector_L = self.start_pos_R + detector_offset  # L exits to right
        self.detector_R = self.start_pos_L - detector_offset  # R exits to left
        
        # Wrap to ring
        self.detector_L = self.detector_L % self.N
        self.detector_R = self.detector_R % self.N

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
        
        # Check detector crossings
        if self.crossing_L is None and self.detector_L is not None:
            # L crosses detector_L (on the right side)?
            if new_L % self.N > self.detector_L or (self.last_pos_L % self.N < self.detector_L and new_L % self.N > self.detector_L):
                self.crossing_L = self.time
        
        if self.crossing_R is None and self.detector_R is not None:
            # R crosses detector_R (on the left side)?
            if new_R % self.N < self.detector_R or (self.last_pos_R % self.N > self.detector_R and new_R % self.N < self.detector_R):
                self.crossing_R = self.time
        
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


def measure_velocities(track, t_max_frac=0.25, t_final=None):
    """Measure velocity from early trajectory."""
    track = np.array(track)
    if len(track) < 10:
        return 0, 0
    
    t_early = track[-1, 0] * t_max_frac if t_final is None else t_final * t_max_frac
    mask = track[:, 0] < t_early
    
    if np.sum(mask) < 5:
        return 0, 0
    
    early = track[mask]
    res = linregress(early[:, 0], early[:, 1])
    return res.slope, res.rvalue**2


def run_reference(g, v_kick, mass_L, mass_R, offset=80, steps=600):
    """
    Run reference simulation with no collision.
    Large offset ensures particles don't meet.
    """
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick, 
                                   mass_L=mass_L, mass_R=mass_R)
    
    for _ in range(steps):
        sim.step()
    
    v_L, r2_L = measure_velocities(sim.track_L, t_final=sim.time)
    v_R, r2_R = measure_velocities(sim.track_R, t_final=sim.time)
    
    return {
        'v_L': v_L,
        'v_R': v_R,
        'r2_L': r2_L,
        'r2_R': r2_R,
        'track_L': np.array(sim.track_L),
        'track_R': np.array(sim.track_R),
    }


def run_collision(g, v_kick, mass_L, mass_R, offset=40, steps=600):
    """Run collision simulation."""
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick,
                                   mass_L=mass_L, mass_R=mass_R)
    
    for _ in range(steps):
        sim.step()
    
    v_L, r2_L = measure_velocities(sim.track_L, t_final=sim.time)
    v_R, r2_R = measure_velocities(sim.track_R, t_final=sim.time)
    
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)
    
    # Find collision time (min separation)
    sep = np.abs(track_L[:, 1] - track_R[:, 1])
    coll_idx = np.argmin(sep)
    t_coll = track_L[coll_idx, 0]
    
    return {
        'v_L': v_L,
        'v_R': v_R,
        'r2_L': r2_L,
        'r2_R': r2_R,
        't_coll': t_coll,
        'track_L': track_L,
        'track_R': track_R,
        'history_F': np.array(sim.history_F),
        'time': sim.time,
        'start_pos_L': sim.start_pos_L,
        'start_pos_R': sim.start_pos_R,
    }


def compute_time_delay(ref, coll, detector_pos_L, detector_pos_R, N=200):
    """
    Compute time delay at detector planes.
    
    detector_pos_L: where L particle should arrive (after passing through)
    detector_pos_R: where R particle should arrive
    """
    def find_crossing_time(track, detector_pos, direction='right'):
        """Find when track crosses detector position."""
        for i in range(1, len(track)):
            t_prev, x_prev = track[i-1]
            t_curr, x_curr = track[i]
            
            # Unwrapped positions
            if direction == 'right':
                # Moving right, looking for crossing detector_pos from below
                if x_prev < detector_pos <= x_curr:
                    # Linear interpolation
                    frac = (detector_pos - x_prev) / (x_curr - x_prev + 1e-9)
                    return t_prev + frac * (t_curr - t_prev)
            else:
                # Moving left, looking for crossing detector_pos from above
                if x_prev > detector_pos >= x_curr:
                    frac = (x_prev - detector_pos) / (x_prev - x_curr + 1e-9)
                    return t_prev + frac * (t_curr - t_prev)
        
        return None
    
    # Reference crossing times (free propagation)
    t_ref_L = find_crossing_time(ref['track_L'], detector_pos_L, 'right')
    t_ref_R = find_crossing_time(ref['track_R'], detector_pos_R, 'left')
    
    # Collision crossing times
    t_coll_L = find_crossing_time(coll['track_L'], detector_pos_L, 'right')
    t_coll_R = find_crossing_time(coll['track_R'], detector_pos_R, 'left')
    
    # Time delays
    dt_L = (t_coll_L - t_ref_L) if (t_coll_L and t_ref_L) else None
    dt_R = (t_coll_R - t_ref_R) if (t_coll_R and t_ref_R) else None
    
    return {
        't_ref_L': t_ref_L,
        't_ref_R': t_ref_R,
        't_coll_L': t_coll_L,
        't_coll_R': t_coll_R,
        'dt_L': dt_L,
        'dt_R': dt_R,
    }


def compute_position_shift(coll, ref):
    """
    Compute position shift relative to reference.
    Uses final position compared to free propagation extrapolation.
    """
    track_L = coll['track_L']
    track_R = coll['track_R']
    
    t_final = coll['time']
    
    # Ghost positions (free propagation at t_final)
    x_ghost_L = coll['start_pos_L'] + ref['v_L'] * t_final
    x_ghost_R = coll['start_pos_R'] + ref['v_R'] * t_final
    
    # Actual final positions
    x_final_L = track_L[-1, 1]
    x_final_R = track_R[-1, 1]
    
    shift_L = x_final_L - x_ghost_L
    shift_R = x_final_R - x_ghost_R
    
    return {
        'shift_L': shift_L,
        'shift_R': shift_R,
        'avg_shift': (shift_L + shift_R) / 2,
        'x_ghost_L': x_ghost_L,
        'x_ghost_R': x_ghost_R,
    }


def run_experiments():
    print("="*70)
    print("DET v5 COLLIDER - PROPER SCATTERING OBSERVABLES")
    print("="*70)
    
    # ===== EXPERIMENT A: TIME DELAY (Symmetric case) =====
    print("\n" + "="*70)
    print("EXPERIMENT A: TIME DELAY (Wigner-style)")
    print("="*70)
    print("Symmetric masses: m_L = m_R = 10.0")
    print("Detector planes at x=160 (L exit) and x=40 (R exit)")
    
    g_vals = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    v_kick = 2.0
    
    print(f"\n{'g':<6} | {'v_L(ref)':<10} | {'v_R(ref)':<10} | {'Δt_L':<10} | {'Δt_R':<10}")
    print("-" * 60)
    
    time_delay_results = []
    for g in g_vals:
        # Reference run (no collision)
        ref = run_reference(g, v_kick, mass_L=10.0, mass_R=10.0, offset=90, steps=800)
        
        # Collision run
        coll = run_collision(g, v_kick, mass_L=10.0, mass_R=10.0, offset=40, steps=800)
        
        # Detector positions (where particles exit after passing through)
        # L starts at 60, moves right, passes R, exits around 160
        # R starts at 140, moves left, passes L, exits around 40
        detector_L = 160
        detector_R = 40
        
        delays = compute_time_delay(ref, coll, detector_L, detector_R)
        
        dt_L_str = f"{delays['dt_L']:.3f}" if delays['dt_L'] else "N/A"
        dt_R_str = f"{delays['dt_R']:.3f}" if delays['dt_R'] else "N/A"
        
        print(f"{g:<6.2f} | {ref['v_L']:<10.4f} | {ref['v_R']:<10.4f} | {dt_L_str:<10} | {dt_R_str:<10}")
        
        time_delay_results.append({
            'g': g,
            'ref': ref,
            'coll': coll,
            'delays': delays,
        })
    
    # ===== EXPERIMENT B: SYMMETRY BREAKING =====
    print("\n" + "="*70)
    print("EXPERIMENT B: SYMMETRY BREAKING (Unequal Masses)")
    print("="*70)
    print("Asymmetric masses: m_L = 10.0, m_R = 14.0")
    print("This breaks identity exchange and recovers clean Δx")
    
    print(f"\n{'g':<6} | {'v_L':<8} | {'v_R':<8} | {'Δx_L':<10} | {'Δx_R':<10} | {'Δx_avg':<10}")
    print("-" * 70)
    
    asymm_results = []
    for g in g_vals:
        # Reference run (asymmetric masses)
        ref = run_reference(g, v_kick, mass_L=10.0, mass_R=14.0, offset=90, steps=800)
        
        # Collision run
        coll = run_collision(g, v_kick, mass_L=10.0, mass_R=14.0, offset=40, steps=800)
        
        # Position shift
        shifts = compute_position_shift(coll, ref)
        
        print(f"{g:<6.2f} | {coll['v_L']:<8.4f} | {coll['v_R']:<8.4f} | "
              f"{shifts['shift_L']:<10.3f} | {shifts['shift_R']:<10.3f} | {shifts['avg_shift']:<10.3f}")
        
        asymm_results.append({
            'g': g,
            'ref': ref,
            'coll': coll,
            'shifts': shifts,
        })
    
    # ===== EXPERIMENT C: VELOCITY CALIBRATION =====
    print("\n" + "="*70)
    print("EXPERIMENT C: VELOCITY CALIBRATION")
    print("="*70)
    print("Mapping v_kick → v_measured")
    
    print(f"\n{'v_kick':<8} | {'v_L(meas)':<10} | {'v_R(meas)':<10} | {'R²_L':<8} | {'R²_R':<8}")
    print("-" * 55)
    
    for v_kick in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        ref = run_reference(g=1.5, v_kick=v_kick, mass_L=10.0, mass_R=10.0, offset=90, steps=600)
        print(f"{v_kick:<8.1f} | {ref['v_L']:<10.4f} | {ref['v_R']:<10.4f} | "
              f"{ref['r2_L']:<8.3f} | {ref['r2_R']:<8.3f}")
    
    # ===== VISUALIZATION =====
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Space-time for asymmetric case
    ax = axes[0, 0]
    best_asymm = asymm_results[3]  # g=1.6
    history = best_asymm['coll']['history_F']
    if len(history) > 0:
        im = ax.imshow(history, aspect='auto', cmap='plasma', origin='lower',
                       extent=[0, 200, 0, best_asymm['coll']['time']])
        
        track_L = best_asymm['coll']['track_L']
        track_R = best_asymm['coll']['track_R']
        ax.plot(track_L[:, 1] % 200, track_L[:, 0], 'c.', markersize=1, label='L (m=10)')
        ax.plot(track_R[:, 1] % 200, track_R[:, 0], 'w.', markersize=1, label='R (m=14)')
        
        ax.set_title(f"Asymmetric Collision (g=1.6, m_L=10, m_R=14)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Time")
        ax.legend(loc='upper right')
        plt.colorbar(im, ax=ax)
    
    # 2. Time delay vs g
    ax = axes[0, 1]
    g_plot = [r['g'] for r in time_delay_results]
    dt_L_plot = [r['delays']['dt_L'] if r['delays']['dt_L'] else 0 for r in time_delay_results]
    dt_R_plot = [r['delays']['dt_R'] if r['delays']['dt_R'] else 0 for r in time_delay_results]
    
    ax.plot(g_plot, dt_L_plot, 'bo-', markersize=8, label='Δt_L')
    ax.plot(g_plot, dt_R_plot, 'rs-', markersize=8, label='Δt_R')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Time Delay Δt")
    ax.set_title("Time Delay (Symmetric Case)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Position shift vs g (asymmetric)
    ax = axes[1, 0]
    g_plot = [r['g'] for r in asymm_results]
    shift_L = [r['shifts']['shift_L'] for r in asymm_results]
    shift_R = [r['shifts']['shift_R'] for r in asymm_results]
    shift_avg = [r['shifts']['avg_shift'] for r in asymm_results]
    
    ax.plot(g_plot, shift_L, 'b^-', markersize=8, label='Δx_L')
    ax.plot(g_plot, shift_R, 'rv-', markersize=8, label='Δx_R')
    ax.plot(g_plot, shift_avg, 'ko-', markersize=10, linewidth=2, label='Δx_avg')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Position Shift Δx (lattice units)")
    ax.set_title("Position Shift (Asymmetric Masses)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Trajectory comparison
    ax = axes[1, 1]
    best = asymm_results[3]  # g=1.6
    
    # Collision tracks
    track_L = best['coll']['track_L']
    track_R = best['coll']['track_R']
    ax.plot(track_L[:, 0], track_L[:, 1], 'b-', linewidth=2, label='L collision')
    ax.plot(track_R[:, 0], track_R[:, 1], 'r-', linewidth=2, label='R collision')
    
    # Reference (ghost) tracks
    ref = best['ref']
    t_arr = np.linspace(0, best['coll']['time'], 100)
    ax.plot(t_arr, best['coll']['start_pos_L'] + ref['v_L']*t_arr, 'b--', alpha=0.5, label='L ghost')
    ax.plot(t_arr, best['coll']['start_pos_R'] + ref['v_R']*t_arr, 'r--', alpha=0.5, label='R ghost')
    
    # Mark collision time
    ax.axvline(x=best['coll']['t_coll'], color='g', linestyle=':', alpha=0.7, label='Collision')
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Position (unwrapped)")
    ax.set_title(f"Trajectories: g=1.6, Δx_avg={best['shifts']['avg_shift']:.2f}")
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/det_v5_scattering.png', dpi=150)
    plt.savefig('/mnt/user-data/outputs/det_v5_scattering.png', dpi=150)
    print("\nSaved to det_v5_scattering.png")
    
    return time_delay_results, asymm_results


if __name__ == "__main__":
    time_results, asymm_results = run_experiments()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
EXPERIMENT A (Time Delay):
  - Symmetric collision with detector planes
  - Measures Δt = t_out - t_free
  - Robust against identity exchange

EXPERIMENT B (Position Shift with Asymmetric Masses):
  - m_L = 10.0, m_R = 14.0
  - Breaks identity degeneracy
  - Recovers clean Δx observable

EXPERIMENT C (Velocity Calibration):
  - Maps v_kick → v_measured
  - Establishes linear regime for momentum injection
  
NEXT: Define scattering length proxy a(g) ≡ -Δx in lattice units
""")
