import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse

"""
DET v5 MOTION LABORATORY & COLLIDER
===================================
Implementation of the "Unified Active Manifold" (v5.0)

Purpose:
--------
1. Demonstrate emergent inertia (Soliton motion).
2. Simulate Soliton-Soliton scattering to extract Phase Shifts & Scattering Lengths.
3. Demonstrate Bound State formation (Fusion).

Operational Upgrade:
--------------------
Now features SELF-CALIBRATING OBSERVABLES. 
Instead of assuming a velocity, the system tracks the pre-collision trajectory,
measures the emergent velocity, and uses that to calculate rigorous phase shifts.

The Five Master Equations Implemented:
1. Time-Energy:     dθ/dt = -β · F + ν · ∇²θ - γ_drag · (∇θ)² (Phase Clamped)
2. Directed Flow:   J_ij  = σ · (C_ij)² · [ ... ] (Flux Limited)
3. Hebbian Metric:  dC/dt = α_eff · max(0, J) - λ_eff · C (Directional)
4. Capacity:        σ     = 1 + ln(1 + usage)
5. Conservation:    dF/dt = -ΣJ
"""

class DETConstants:
    def __init__(self, g=1.0):
        # Primary Scaling Factor (Coupling Strength)
        self.g = g
        
        # Time/Energy Coupling (Inverse Planck)
        self.BETA = 2.0 * g
        
        # Phase Viscosity (Diffusion)
        self.NU = 0.5
        
        # Phase Drag (Fusion Brake)
        self.PHASE_DRAG = 0.2
        
        # Metric Plasticity (Learning Rate)
        self.ALPHA = 0.5 * g
        
        # Forward Plasticity (Lookahead)
        self.GAMMA = 8.0
        
        # Metric Entropy (Forgetting Rate)
        self.LAMBDA = 0.005
        
        # Fusion Trigger (Strong Force)
        # Increased to 90.0 to firm up the bound state at high coupling
        self.K_FUSION = 90.0
        
        # Vacuum Floors
        self.F_VAC = 0.01  
        self.C_MIN = 0.05 
        
        # Simulation scale
        self.DT = 0.05

class ActiveManifold1D:
    def __init__(self, size=200, constants=DETConstants()):
        self.N = size
        self.k = constants
        
        # --- State ---
        self.F = np.ones(self.N) * self.k.F_VAC
        self.theta = np.zeros(self.N)
        self.sigma = np.ones(self.N)
        self.C_right = np.ones(self.N) * self.k.C_MIN
        self.C_left  = np.ones(self.N) * self.k.C_MIN
        
        # --- Analysis Data ---
        self.history_F = []
        self.trajectory_log = [] # Stores list of peak positions per tick
        self.time = 0.0
        self.start_pos_L = 0
        self.start_pos_R = 0

    def initialize_soliton(self, position, width, mass, velocity_kick=0.0):
        """Inject a single packet."""
        x = np.arange(self.N)
        dx = np.minimum(np.abs(x - position), self.N - np.abs(x - position))
        envelope = np.exp(-0.5 * (dx / width)**2)
        
        self.F += mass * envelope
        self.theta += velocity_kick * x 
        
        # Pre-seed metric
        self.C_right += 0.8 * envelope
        self.C_left  += 0.8 * envelope
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)

    def initialize_collider_event(self, offset=40, width=5.0, mass=10.0, v_rel=0.5):
        """Initialize two solitons on a collision course."""
        center = self.N // 2
        self.start_pos_L = center - offset
        self.start_pos_R = center + offset
        
        self.initialize_soliton(self.start_pos_L, width, mass, velocity_kick=v_rel)
        self.initialize_soliton(self.start_pos_R, width, mass, velocity_kick=-v_rel)
        
        print(f"Collider Ready: Mass={mass}, V_kick={2*v_rel:.2f}, g={self.k.g:.2f}")

    def initialize_three_body(self, spacing=20, width=5.0, mass=10.0, v_rel=0.4):
        """Initialize three solitons."""
        center = self.N // 2
        self.initialize_soliton(center - spacing, width, mass, velocity_kick=v_rel)
        self.initialize_soliton(center, width, mass, velocity_kick=0.0)
        self.initialize_soliton(center + spacing, width, mass, velocity_kick=-v_rel)
        print(f"Three-Body Ready: Mass={mass}, V_rel={v_rel:.2f}")

    def find_peaks(self):
        """Robust peak finder for tracking trajectories."""
        mass_dist = self.F - self.k.F_VAC
        peaks = []
        # Smoothing
        smoothed = np.convolve(mass_dist, np.ones(3)/3, mode='same') 
        threshold = np.max(mass_dist) * 0.3
        
        for i in range(1, self.N-1):
            if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > threshold:
                peaks.append(i)
        # Check boundary
        if smoothed[0] > smoothed[-1] and smoothed[0] > smoothed[1] and smoothed[0] > threshold:
            peaks.append(0)
            
        return sorted(peaks)

    def step(self):
        """Execute one cycle of the DET v5 Master Equations."""
        
        # 1. Time-Energy
        idx = np.arange(self.N)
        ip1 = (idx + 1) % self.N
        im1 = (idx - 1) % self.N
        
        # Laplacian & Gradient
        d_fwd = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[idx])))
        d_bwd = np.angle(np.exp(1j * (self.theta[im1] - self.theta[idx])))
        laplacian_theta = d_fwd + d_bwd
        grad_theta = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[im1]))) / 2.0
        
        phase_drive = np.minimum(self.k.BETA * self.F * self.k.DT, np.pi/4.0) # Phase Clamp
        drag_factor = 1.0 / (1.0 + self.k.PHASE_DRAG * (grad_theta**2))
        
        self.theta -= phase_drive * drag_factor
        self.theta += self.k.NU * laplacian_theta * self.k.DT
        self.theta = np.mod(self.theta, 2*np.pi)
        
        # 2. Directed Flow
        recip_R = np.sqrt(self.C_left[ip1])
        recip_L = np.sqrt(self.C_right[im1])
        
        drive_R = recip_R * np.sin(self.theta[ip1]-self.theta[idx]) + (1-recip_R)*(self.F[idx]-self.F[ip1])
        drive_L = recip_L * np.sin(self.theta[im1]-self.theta[idx]) + (1-recip_L)*(self.F[idx]-self.F[im1])
        
        cond_R = self.sigma[idx] * (self.C_right[idx]**2 + 1e-4)
        cond_L = self.sigma[idx] * (self.C_left[idx]**2 + 1e-4)
        
        J_right = cond_R * drive_R
        J_left  = cond_L * drive_L

        # Flux Limit
        max_allowed = 0.40 * self.F / self.k.DT
        total_out = np.abs(J_right) + np.abs(J_left)
        scale = np.minimum(1.0, max_allowed / (total_out + 1e-9))
        J_right *= scale
        J_left *= scale

        # 5. Conservation
        dF = (J_right[im1] + J_left[ip1]) - (J_right + J_left)
        self.F += dF * self.k.DT
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        # 3. Metric
        compression = np.maximum(0, dF)
        alpha = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        lam = self.k.LAMBDA / (1.0 + self.k.K_FUSION * compression * 10.0)
        
        # Directional Plasticity
        J_R_align = np.maximum(0, J_right) + self.k.GAMMA * np.maximum(0, np.roll(J_right, 1))
        J_L_align = np.maximum(0, J_left) + self.k.GAMMA * np.maximum(0, np.roll(J_left, -1))
        
        self.C_right += (alpha * J_R_align - lam * self.C_right) * self.k.DT
        self.C_left  += (alpha * J_L_align - lam * self.C_left) * self.k.DT
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)
        
        # 4. Capacity
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_right) + np.abs(J_left))

        self.time += self.k.DT
        
        # Logging
        if int(self.time/self.k.DT) % 10 == 0:
            self.history_F.append(self.F.copy())
            
        # TRACK TRAJECTORY (Every step for precision)
        self.trajectory_log.append((self.time, self.find_peaks()))

def analyze_experiment(sim):
    """
    Performs post-run analysis to extract measured velocity and phase shift.
    """
    traj = sim.trajectory_log
    
    # 1. Measure Pre-Collision Velocity
    # We look at the first 20% of time steps (approach phase)
    limit_time = sim.time * 0.2
    
    t_vals = []
    x_vals = []
    
    # Track the Left-side particle (Packet 1)
    # It starts at start_pos_L and moves Right (increasing index)
    for t, peaks in traj:
        if t > limit_time: break
        if not peaks: continue
        
        # Find peak closest to expected start
        # Simple heuristic: The leftmost peak is usually Packet 1 in the start phase
        p = min(peaks) 
        
        # Handle ring wrap? In early phase, it shouldn't wrap.
        t_vals.append(t)
        x_vals.append(p)
        
    if len(t_vals) < 5:
        return "Error", 0, 0, 0, 0

    # Linear Regression to get v_meas
    res = linregress(t_vals, x_vals)
    v_meas = res.slope
    r_sq = res.rvalue**2
    
    # 2. Predict Final Position (Ghost Particle)
    # Ghost 1 started at fit_intercept at t=0
    x_start_fit = res.intercept
    x_final_ghost = (x_start_fit + v_meas * sim.time) % sim.N
    
    # 3. Find Actual Final Position
    final_peaks = traj[-1][1]
    num_peaks = len(final_peaks)
    
    status = "Unknown"
    phase_shift = 0.0
    
    if num_peaks == 0:
        status = "Dissipated"
    elif num_peaks == 1:
        # FUSION
        status = "FUSION (Bound)"
        actual_pos = final_peaks[0]
        
        # Shift = Actual - Expected
        # Handle Ring Math
        diff = actual_pos - x_final_ghost
        if diff > sim.N/2: diff -= sim.N
        if diff < -sim.N/2: diff += sim.N
        phase_shift = diff
        
    else:
        status = "SCATTERING"
        # Find peak closest to Ghost
        best_dist = 999
        closest_p = -1
        for p in final_peaks:
            dist = abs(p - x_final_ghost)
            if dist > sim.N/2: dist = sim.N - dist
            if dist < best_dist:
                best_dist = dist
                closest_p = p
        
        diff = closest_p - x_final_ghost
        if diff > sim.N/2: diff -= sim.N
        if diff < -sim.N/2: diff += sim.N
        phase_shift = diff

    # Scattering Length Calculation
    # a = -delta (roughly)
    # We normalize by k if we had it, but raw shift is a good proxy for 'a' in 1D lattice
    scat_len = -phase_shift

    return status, num_peaks, v_meas, phase_shift, scat_len

def run_nuclear_suite():
    print("="*70)
    print("DET v5: SELF-CALIBRATING NUCLEAR SUITE")
    print("="*70)
    
    # --- EXPERIMENT 1: SCATTERING LENGTH (Resonance Scan) ---
    print("\n[Exp 1] Scattering Length vs Coupling (g)")
    print("        Uses Measured Velocity (v_meas) for Phase Shift")
    print(f"{'g':<6} | {'Status':<16} | {'v_meas':<6} | {'Shift':<8} | {'a (fm)':<8}")
    print("-" * 65)
    
    # Scan around resonance
    couplings = [1.3, 1.4, 1.5, 1.6, 1.7]
    
    for g in couplings:
        k = DETConstants(g=g)
        sim = ActiveManifold1D(size=200, constants=k)
        # Use a consistent kick, but we verify it
        sim.initialize_collider_event(v_rel=0.2, offset=30) 
        
        # Run
        for _ in range(1500): sim.step()
        
        stat, n, v, shift, a = analyze_experiment(sim)
        print(f"{g:<6.2f} | {stat:<16} | {v:<6.3f} | {shift:<8.2f} | {a:<8.2f}")

    # --- EXPERIMENT 2: FUSION STABILITY ---
    print("\n[Exp 2] Fusion Stability (High Coupling)")
    print(f"{'g':<6} | {'Status':<16} | {'Peaks':<6} | {'Width':<6}")
    print("-" * 55)
    
    # Focus on the high-g fusion zone where we expect stability
    couplings_scan = [2.4, 2.6, 2.8, 3.0, 3.2]
    
    for g in couplings_scan:
        k = DETConstants(g=g)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(v_rel=0.2, offset=30) 
        
        for _ in range(1500): sim.step()
        
        # Calculate width
        mass_dist = sim.F - sim.k.F_VAC
        total_m = np.sum(mass_dist)
        rms = 0.0
        if total_m > 0:
            com = np.sum(np.arange(sim.N) * mass_dist) / total_m
            rms = np.sqrt(np.sum(mass_dist * (np.arange(sim.N)-com)**2) / total_m)
            
        stat, n, _, _, _ = analyze_experiment(sim)
        print(f"{g:<6.2f} | {stat:<16} | {n:<6} | {rms:<6.1f}")

    # --- EXPERIMENT 3: THREE-BODY ---
    print("\n[Exp 3] Three-Body Interaction")
    sim3 = ActiveManifold1D(size=200, constants=DETConstants(g=2.8)) 
    sim3.initialize_three_body(spacing=20, v_rel=0.4)
    for _ in range(2000): sim3.step()
    
    peaks = sim3.find_peaks()
    print(f"Result: {len(peaks)} peaks found.")
    
    return sim3

def plot_collider(sim):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    history = np.array(sim.history_F)
    im = ax.imshow(history, aspect='auto', cmap='plasma', origin='lower',
                   extent=[0, sim.N, 0, sim.time])
    ax.set_title(f"Nuclear Event Visualization (g={sim.k.g:.1f})")
    ax.set_xlabel("Space (x)")
    ax.set_ylabel("Time (t)")
    plt.colorbar(im, ax=ax, label='Resource Density')
    plt.tight_layout()
    plt.savefig('det_v5_nuclear_suite.png')
    print("\nSaved visualization to 'det_v5_nuclear_suite.png'")

if __name__ == "__main__":
    sim = run_nuclear_suite()
    plot_collider(sim)