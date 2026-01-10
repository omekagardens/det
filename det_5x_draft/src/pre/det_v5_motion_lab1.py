import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

"""
DET v5 MOTION LABORATORY & COLLIDER
===================================
Implementation of the "Unified Active Manifold" (v5.0)

Purpose:
--------
1. Demonstrate emergent inertia (Soliton motion).
2. Simulate Soliton-Soliton scattering to extract Phase Shifts & Scattering Lengths.
3. Demonstrate Bound State formation (Fusion) via "Phase-Locked Metric" & Anti-Aliasing.

The Five Master Equations Implemented:
1. Time-Energy:     dθ/dt = -β · F (Clamped to π/4 per tick)
2. Directed Flow:   J_ij  = σ · (C_ij)² · [ ... ] (Flux Limited)
3. Hebbian Metric:  dC/dt = α_eff · max(0, J) - λ_eff · C (Directional)
4. Capacity:        σ     = 1 + ln(1 + usage)
5. Conservation:    dF/dt = -ΣJ
"""

class DETConstants:
    """
    Fundamental constants of the v5 Active Manifold.
    """
    def __init__(self, g=1.0):
        # Primary Scaling Factor (Coupling Strength)
        self.g = g
        
        # Time/Energy Coupling (Inverse Planck)
        self.BETA = 2.0 * g
        
        # Phase Viscosity (Diffusion)
        # Reduced to 0.5 because the Phase Clamp removes the source of the noise.
        self.NU = 0.5
        
        # Phase Drag (Fusion Brake)
        self.PHASE_DRAG = 0.2
        
        # Metric Plasticity (Learning Rate)
        self.ALPHA = 0.5 * g
        
        # Forward Plasticity (Lookahead)
        self.GAMMA = 8.0
        
        # Metric Entropy (Forgetting Rate)
        self.LAMBDA = 0.005
        
        # Fusion Trigger
        # Tuned to 60.0.
        self.K_FUSION = 60.0
        
        # Vacuum Floors
        self.F_VAC = 0.01  
        self.C_MIN = 0.05 
        
        # Simulation scale
        self.DT = 0.05

class ActiveManifold1D:
    def __init__(self, size=200, constants=DETConstants()):
        self.N = size
        self.k = constants
        
        # --- I. AGENT STATE ---
        self.F = np.ones(self.N) * self.k.F_VAC
        self.theta = np.zeros(self.N)
        self.sigma = np.ones(self.N)
        
        # --- II. NETWORK STATE (The Metric) ---
        self.C_right = np.ones(self.N) * self.k.C_MIN
        self.C_left  = np.ones(self.N) * self.k.C_MIN
        
        # History
        self.history_F = []
        self.time = 0.0
        self.p1_start = 0
        self.p2_start = 0
        self.v_initial = 0
        self.mass_initial = 0

    def initialize_soliton(self, position, width, mass, velocity_kick=0.0):
        """Inject a single packet."""
        x = np.arange(self.N)
        dx = np.minimum(np.abs(x - position), self.N - np.abs(x - position))
        envelope = np.exp(-0.5 * (dx / width)**2)
        
        self.F += mass * envelope
        
        # Phase Gradient = Momentum
        self.theta += velocity_kick * x 
        
        # Pre-seed metric (Gravity Well)
        self.C_right += 0.8 * envelope
        self.C_left  += 0.8 * envelope
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)

    def initialize_collider_event(self, offset=40, width=5.0, mass=10.0, v_rel=0.5):
        """Initialize two solitons on a collision course."""
        center = self.N // 2
        self.initialize_soliton(center - offset, width, mass, velocity_kick=v_rel)
        self.initialize_soliton(center + offset, width, mass, velocity_kick=-v_rel)
        
        self.p1_start = center - offset
        self.p2_start = center + offset
        self.v_initial = v_rel
        self.mass_initial = mass
        
        print(f"Collider Ready: Mass={mass}, V_rel={2*v_rel:.2f}, g={self.k.g:.2f}")

    def initialize_three_body(self, spacing=20, width=5.0, mass=10.0, v_rel=0.4):
        """Initialize three solitons (tight spacing for 3-body)."""
        center = self.N // 2
        self.initialize_soliton(center - spacing, width, mass, velocity_kick=v_rel)
        self.initialize_soliton(center, width, mass, velocity_kick=0.0)
        self.initialize_soliton(center + spacing, width, mass, velocity_kick=-v_rel)
        
        print(f"Three-Body Ready: Mass={mass}, V_rel={v_rel:.2f}")

    def step(self):
        """Execute one cycle of the DET v5 Master Equations."""
        
        # 1. Time-Energy (Clock)
        idx = np.arange(self.N)
        ip1 = (idx + 1) % self.N
        im1 = (idx - 1) % self.N
        
        # Laplacian for Viscosity
        d_fwd = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[idx])))
        d_bwd = np.angle(np.exp(1j * (self.theta[im1] - self.theta[idx])))
        laplacian_theta = d_fwd + d_bwd
        
        # Phase Gradient for Drag
        grad_theta = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[im1]))) / 2.0
        
        phase_drive = self.k.BETA * self.F * self.k.DT
        
        # CRITICAL FIX: PHASE CLAMP (The "Causal Speed Limit")
        # Prevents phase aliasing at the core of the nucleus.
        # If rotation > PI, the force flips sign (Explosion).
        # We clamp to PI/4 to be safe and linear.
        phase_drive = np.minimum(phase_drive, np.pi/4.0)
        
        drag_factor = 1.0 / (1.0 + self.k.PHASE_DRAG * (grad_theta**2))
        
        # Update Theta
        self.theta -= phase_drive * drag_factor
        self.theta += self.k.NU * laplacian_theta * self.k.DT
        self.theta = np.mod(self.theta, 2*np.pi)
        
        # 2. Directed Flow (Engine)
        recip_R = np.sqrt(self.C_left[ip1])
        recip_L = np.sqrt(self.C_right[im1])
        
        phase_term_R = np.sin(self.theta[ip1] - self.theta[idx])
        press_term_R = (self.F[idx] - self.F[ip1])
        drive_R = recip_R * phase_term_R + (1.0 - recip_R) * press_term_R
        cond_R = self.sigma[idx] * (self.C_right[idx]**2 + 1e-4)
        J_right = cond_R * drive_R
        
        phase_term_L = np.sin(self.theta[im1] - self.theta[idx])
        press_term_L = (self.F[idx] - self.F[im1])
        drive_L = recip_L * phase_term_L + (1.0 - recip_L) * press_term_L
        cond_L = self.sigma[idx] * (self.C_left[idx]**2 + 1e-4)
        J_left = cond_L * drive_L

        # FLUX LIMITER (CFL Condition)
        total_out = np.abs(J_right) + np.abs(J_left)
        # Relaxed slightly to 0.40 now that Phase Clamp prevents explosion
        max_allowed = 0.40 * self.F / self.k.DT
        scale_factor = np.minimum(1.0, max_allowed / (total_out + 1e-9))
        
        J_right *= scale_factor
        J_left *= scale_factor

        # 5. Conservation & Imbalance
        in_R = J_right[im1] 
        in_L = J_left[ip1]
        
        dF_flow = (in_R + in_L) - (J_right + J_left)
        
        self.F += dF_flow * self.k.DT
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        # 3. Hebbian Metric (Learning)
        # FUSION LOGIC: Unbalanced Flow -> Hard Metric
        
        compression = np.maximum(0, dF_flow)
        
        # Dynamic Constants
        alpha_eff = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        lambda_eff = self.k.LAMBDA / (1.0 + self.k.K_FUSION * compression * 10.0)
        
        # Directional Plasticity:
        J_R_aligned = np.maximum(0, J_right) 
        J_R_in_aligned = np.maximum(0, np.roll(J_right, 1))
        
        driver_R = J_R_aligned + self.k.GAMMA * J_R_in_aligned
        self.C_right += (alpha_eff * driver_R - lambda_eff * self.C_right) * self.k.DT
        
        J_L_aligned = np.maximum(0, J_left)
        J_L_in_aligned = np.maximum(0, np.roll(J_left, -1))
        
        driver_L = J_L_aligned + self.k.GAMMA * J_L_in_aligned
        self.C_left += (alpha_eff * driver_L - lambda_eff * self.C_left) * self.k.DT
        
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left  = np.clip(self.C_left, self.k.C_MIN, 1.0)
        
        # 4. Capacity
        flux_sum = np.abs(J_right) + np.abs(J_left)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + flux_sum)

        self.time += self.k.DT
        
        if int(self.time/self.k.DT) % 10 == 0:
            self.history_F.append(self.F.copy())

def measure_observables(sim, expected_v):
    """Extract physical quantities."""
    mass_dist = sim.F - sim.k.F_VAC
    
    # 1. Peak Identification
    peaks = []
    smoothed = np.convolve(mass_dist, np.ones(5)/5, mode='same') 
    threshold = np.max(mass_dist) * 0.4
    for i in range(1, sim.N-1):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1] and smoothed[i] > threshold:
            peaks.append(i)
    if smoothed[0] > smoothed[-1] and smoothed[0] > smoothed[1] and smoothed[0] > threshold:
        peaks.append(0)

    num_peaks = len(peaks)
    status = "Unknown"
    phase_shift = 0.0
    scat_len = 0.0
    
    # 2. State Classification
    if num_peaks == 1:
        # FUSION CHECK
        peak_x = peaks[0]
        dx = np.minimum(np.abs(np.arange(sim.N) - peak_x), sim.N - np.abs(np.arange(sim.N) - peak_x))
        rms_width = np.sqrt(np.sum(mass_dist * dx**2) / np.sum(mass_dist))
        
        if rms_width < 25: 
            status = "FUSION (Bound)"
        else:
            status = "Loose Cloud (1 Peak)"
            
    elif num_peaks >= 2:
        status = "SCATTERING"
        # Phase Shift calc
        expected_trans = (sim.p1_start + expected_v * sim.time) % sim.N
        d_min = 9999
        closest_peak = -1
        for p in peaks:
            d = min(abs(p - expected_trans), sim.N - abs(p - expected_trans))
            if d < d_min:
                d_min = d
                closest_peak = p
        
        if d_min > sim.N / 4:
            status += " (Reflected)"
            phase_shift = np.nan
        else:
            dist_signed = closest_peak - expected_trans
            if dist_signed > sim.N/2: dist_signed -= sim.N
            if dist_signed < -sim.N/2: dist_signed += sim.N
            phase_shift = dist_signed
            scat_len = -phase_shift

    return {
        "status": status,
        "peaks": num_peaks,
        "shift": phase_shift,
        "a": scat_len
    }

def run_nuclear_suite():
    print("="*60)
    print("DET v5: NUCLEAR PHYSICS SUITE")
    print("="*60)
    
    # --- EXPERIMENT 1: SCATTERING LENGTH (Low Energy Limit) ---
    print("\n[Exp 1] Scattering Length vs Coupling (g)")
    print(f"{'g':<6} | {'Status':<25} | {'Shift':<8} | {'a (fm)':<8}")
    print("-" * 55)
    
    # Scan window around the established resonance (1.2-1.6)
    couplings = [1.2, 1.3, 1.4, 1.5, 1.6]
    v_low = 0.2
    
    for g in couplings:
        k = DETConstants(g=g)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(v_rel=v_low, offset=30)
        
        # Run
        for _ in range(1500): sim.step()
        
        res = measure_observables(sim, v_low)
        print(f"{g:<6.2f} | {res['status']:<25} | {res['shift']:<8.2f} | {res['a']:<8.2f}")

    # --- EXPERIMENT 2: FUSION WINDOW (Scan Coupling g) ---
    print("\n[Exp 2] Fusion Check vs Coupling (g) [Fixed V=0.2]")
    print(f"{'g':<6} | {'Status':<25} | {'Peaks':<6} | {'Width':<6}")
    print("-" * 55)
    
    # Focus on the high-g fusion zone
    couplings_scan = [1.6, 1.8, 2.0, 2.2, 2.4]
    
    for g in couplings_scan:
        k = DETConstants(g=g)
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_collider_event(v_rel=0.2, offset=30) 
        
        for _ in range(1500): sim.step()
        
        res = measure_observables(sim, 0.2)
        
        # Get width if available
        mass_dist = sim.F - sim.k.F_VAC
        total_m = np.sum(mass_dist)
        if total_m > 0:
            com = np.sum(np.arange(sim.N) * mass_dist) / total_m
            rms = np.sqrt(np.sum(mass_dist * (np.arange(sim.N)-com)**2) / total_m)
        else:
            rms = 0.0
            
        print(f"{g:<6.2f} | {res['status']:<25} | {res['peaks']:<6} | {rms:<6.1f}")

    # --- EXPERIMENT 3: THREE-BODY INTERACTION ---
    print("\n[Exp 3] Three-Body Interaction (Formation of ³He?)")
    # Updated to g=2.0 (Solid Fusion Zone)
    sim3 = ActiveManifold1D(size=200, constants=DETConstants(g=2.0)) 
    sim3.initialize_three_body(spacing=20, v_rel=0.4)
    
    for _ in range(2000): sim3.step()
    
    res3 = measure_observables(sim3, 0.4)
    print(f"Result: {res3['status']} with {res3['peaks']} peaks.")
    
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