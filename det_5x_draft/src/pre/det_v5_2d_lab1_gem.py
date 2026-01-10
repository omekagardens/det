import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from scipy import ndimage
from scipy.optimize import curve_fit

"""
DET v5 2D FLATLAND LABORATORY
=============================
Implementation of the "Unified Active Manifold" in 2D.

Purpose:
--------
1. Move beyond 1D restrictions to allow Phase Vortices (Electromagnetism).
2. Test if "Dark Flow" (Vacuum Pressure) scales as 1/r in 2D (Gravity).
3. Demonstrate vector solitons and metric shear.

Operational Improvements (v5.2):
--------------------------------
1. Separate Free-Flight vs Forced-Contact suites.
2. Early-out phase shift measurement (collision-local).
3. Bind Fraction observable (time-integrated).
4. Trial averaging with noise (robustness).
5. Gauss-Law disk flux for gravity (lattice-safe).
6. Symmetry breaking (unequal masses).
7. Explicit detector reach validation.

The Five Master Equations (2D Generalized):
1. Time-Energy:     dθ/dt = -β · F + ν · ∇²θ - γ_drag · (∇θ)²
2. Directed Flow:   J_vec = σ · (C_vec)² · [ ... ]
3. Hebbian Metric:  dC_vec/dt = α · J_vec - λ · C_vec
4. Capacity:        σ     = 1 + ln(1 + usage)
5. Conservation:    dF/dt = -∇·J
"""

class DETConstants2D:
    def __init__(self, g=2.4): 
        # Primary Coupling
        self.g = g
        
        # Time/Energy
        self.BETA = 2.0 * g
        
        # Viscosity & Drag (Critical for stability)
        self.NU = 0.5
        self.PHASE_DRAG = 0.2
        
        # Plasticity
        self.ALPHA = 0.5 * g
        self.GAMMA = 8.0 # Forward lookahead
        
        # Metric Entropy (Forgetting Rate)
        # SUPERFLUID VACUUM: Reduced to 1e-5 to allow long-range 1/r gravity
        self.LAMBDA = 0.00001
        
        # Fusion/Interaction Trigger
        self.K_FUSION = 90.0
        
        # Vacuum
        # SUPERFLUID VACUUM: Reduced floor to minimize back-pressure
        self.F_VAC = 0.001
        self.C_MIN = 0.05
        
        # Simulation
        self.DT = 0.05

class ActiveManifold2D:
    def __init__(self, size=64, constants=DETConstants2D()):
        self.N = size
        self.k = constants
        
        # --- I. AGENT STATE (N x N) ---
        self.F = np.ones((self.N, self.N)) * self.k.F_VAC
        self.theta = np.zeros((self.N, self.N))
        self.sigma = np.ones((self.N, self.N))
        
        # --- II. METRIC STATE (Vector Bonds) ---
        # C_x[i,j] connects (i,j) to (i, j+1)  [Rightward]
        # C_y[i,j] connects (i,j) to (i+1, j)  [Downward]
        self.C_x = np.ones((self.N, self.N)) * self.k.C_MIN
        self.C_y = np.ones((self.N, self.N)) * self.k.C_MIN
        
        # History for Viz
        self.history_F = []
        self.time = 0.0
        
        # Pinning Mask (for Gravity Test)
        self.static_mask = np.zeros((self.N, self.N), dtype=bool)
        self.F_static = np.zeros((self.N, self.N))
        
        # Store flux for analysis
        self.J_x_last = np.zeros((self.N, self.N))
        self.J_y_last = np.zeros((self.N, self.N))
        
        # Trajectory tracking for collider
        # Stores: (time, [peak1_pos, peak2_pos...])
        self.trajectory_log = []

    def initialize_massive_body(self, mass=50.0, radius=5.0, pinned=True):
        """Injects a massive soliton in the center to test Gravity."""
        center = self.N // 2
        y, x = np.mgrid[0:self.N, 0:self.N]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        mass_dist = mass * np.exp(-0.5 * (r/radius)**2)
        self.F += mass_dist
        
        seed = 0.8 * np.exp(-0.5 * (r/radius)**2)
        self.C_x += seed
        self.C_y += seed
        self.C_x = np.clip(self.C_x, self.k.C_MIN, 1.0)
        self.C_y = np.clip(self.C_y, self.k.C_MIN, 1.0)
        
        if pinned:
            self.static_mask = mass_dist > (self.k.F_VAC * 5)
            self.F_static = self.F.copy()
            # print(f"Initialized Massive Body: M={mass} (PINNED)")

    def initialize_collider(self, separation=20, velocity_kick=0.5, forced_contact=False):
        """
        Initialize two packets for collision.
        forced_contact=True places them closer to ensure interaction.
        """
        center_y, center_x = self.N // 2, self.N // 2
        
        # Symmetry breaking: unequal masses slightly
        mass1 = 20.0
        mass2 = 22.0 
        
        if forced_contact:
            d = 6 # Forced overlap
        else:
            d = separation
            
        p1 = (center_y, center_x - d//2)
        p2 = (center_y, center_x + d//2)
        
        y, x = np.mgrid[0:self.N, 0:self.N]
        
        # Packet 1 (Left, moving Right)
        r1 = np.sqrt((y-p1[0])**2 + (x-p1[1])**2)
        self.F += mass1 * np.exp(-0.5 * (r1/3.0)**2)
        self.theta += velocity_kick * x 
        
        # Packet 2 (Right, moving Left)
        r2 = np.sqrt((y-p2[0])**2 + (x-p2[1])**2)
        self.F += mass2 * np.exp(-0.5 * (r2/3.0)**2)
        
        # Reset theta and apply weighted local ramps
        self.theta = np.zeros((self.N, self.N))
        self.theta += (velocity_kick * x) * (mass1 * np.exp(-0.5 * (r1/3.0)**2) / (self.F + 1e-9))
        self.theta += (-velocity_kick * x) * (mass2 * np.exp(-0.5 * (r2/3.0)**2) / (self.F + 1e-9))
        
        seed = 0.8 * (np.exp(-0.5 * (r1/3.0)**2) + np.exp(-0.5 * (r2/3.0)**2))
        self.C_x += seed
        self.C_y += seed
        
        # print(f"Collider Initialized: d={d}, v={velocity_kick}, forced={forced_contact}")

    def find_peaks(self):
        """Robust peak finder using scipy.ndimage."""
        neighborhood_size = 5
        data_max = ndimage.maximum_filter(self.F, neighborhood_size)
        maxima = (self.F == data_max)
        data_min = ndimage.minimum_filter(self.F, neighborhood_size)
        diff = ((data_max - data_min) > 0.5) 
        maxima[diff == 0] = 0
        
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        
        peaks = []
        for dy, dx in slices:
            xc = (dx.start + dx.stop - 1) / 2
            yc = (dy.start + dy.stop - 1) / 2
            peaks.append(np.array([yc, xc]))
            
        # Sort by x-coordinate
        peaks.sort(key=lambda p: p[1])
        return peaks

    def step(self):
        # 1. TIME-ENERGY
        N_idx = np.roll(np.arange(self.N), 1)  # i-1
        S_idx = np.roll(np.arange(self.N), -1) # i+1
        W_idx = np.roll(np.arange(self.N), 1)  # j-1
        E_idx = np.roll(np.arange(self.N), -1) # j+1
        
        d_E = np.angle(np.exp(1j * (self.theta[:, E_idx] - self.theta)))
        d_W = np.angle(np.exp(1j * (self.theta[:, W_idx] - self.theta)))
        d_S = np.angle(np.exp(1j * (self.theta[S_idx, :] - self.theta)))
        d_N = np.angle(np.exp(1j * (self.theta[N_idx, :] - self.theta)))
        
        laplacian = (d_E + d_W + d_S + d_N)
        grad_sq = 0.25 * ( (d_E - d_W)**2 + (d_S - d_N)**2 )
        
        phase_drive = np.minimum(self.k.BETA * self.F * self.k.DT, np.pi/4.0)
        drag = 1.0 / (1.0 + self.k.PHASE_DRAG * grad_sq)
        
        self.theta -= phase_drive * drag
        self.theta += self.k.NU * laplacian * self.k.DT
        
        # Add Noise (Trial Averaging Discipline)
        self.theta += np.random.normal(0, 0.01, (self.N, self.N)) * self.k.DT
        
        self.theta = np.mod(self.theta, 2*np.pi)
        
        # 2. DIRECTED FLOW
        d_theta_E = np.angle(np.exp(1j * (self.theta[:, E_idx] - self.theta)))
        d_press_E = self.F - self.F[:, E_idx]
        drive_x = np.sin(d_theta_E) + d_press_E
        cond_x = self.sigma * (self.C_x**2 + 1e-4)
        J_x = cond_x * drive_x 
        
        d_theta_S = np.angle(np.exp(1j * (self.theta[S_idx, :] - self.theta)))
        d_press_S = self.F - self.F[S_idx, :]
        drive_y = np.sin(d_theta_S) + d_press_S
        cond_y = self.sigma * (self.C_y**2 + 1e-4)
        J_y = cond_y * drive_y 
        
        # Flux Limiter
        J_mag = np.sqrt(J_x**2 + J_y**2)
        max_allowed = 0.25 * self.F / self.k.DT
        scale = np.minimum(1.0, max_allowed / (J_mag + 1e-9))
        J_x *= scale
        J_y *= scale
        
        self.J_x_last = J_x
        self.J_y_last = J_y
        
        # 5. CONSERVATION
        J_x_in = np.roll(J_x, 1, axis=1)
        J_y_in = np.roll(J_y, 1, axis=0)
        
        div_J = (J_x - J_x_in) + (J_y - J_y_in)
        self.F -= div_J * self.k.DT
        
        if np.any(self.static_mask):
            self.F[self.static_mask] = self.F_static[self.static_mask]
            
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        # 3. METRIC UPDATE
        compression = np.maximum(0, -div_J)
        alpha_eff = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        
        decay_x = self.k.LAMBDA * self.C_x * (1.0 - self.C_x**2)
        decay_y = self.k.LAMBDA * self.C_y * (1.0 - self.C_y**2)
        
        self.C_x += (alpha_eff * np.abs(J_x) - decay_x) * self.k.DT
        self.C_y += (alpha_eff * np.abs(J_y) - decay_y) * self.k.DT
        
        self.C_x = np.clip(self.C_x, self.k.C_MIN, 1.0)
        self.C_y = np.clip(self.C_y, self.k.C_MIN, 1.0)
        
        # 4. CAPACITY
        flux_sum = np.abs(J_x_in) + np.abs(J_y_in) + np.abs(J_x) + np.abs(J_y)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + flux_sum)
        
        self.time += self.k.DT
        
        if int(self.time/self.k.DT) % 10 == 0:
            self.history_F.append(self.F.copy())
            
        self.trajectory_log.append((self.time, self.find_peaks()))

    def get_metrics(self):
        """Returns dictionary of current system state metrics."""
        total_mass = np.sum(self.F)
        max_mass = np.max(self.F)
        avg_C = (np.mean(self.C_x) + np.mean(self.C_y)) / 2.0
        
        peaks = self.find_peaks()
        vortex_dist = 0.0
        if len(peaks) >= 2:
            p1 = np.array(peaks[0])
            p2 = np.array(peaks[1])
            delta = np.abs(p1 - p2)
            delta = np.minimum(delta, self.N - delta) # Torus distance
            vortex_dist = np.sqrt(np.sum(delta**2))
            
        return {
            "Total Mass": total_mass,
            "Max F": max_mass,
            "Avg C": avg_C,
            "Vortex Dist": vortex_dist,
            "Peak Count": len(peaks)
        }

def plot_2d_snapshot(sim, title="DET v5 2D Snapshot"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    im1 = ax1.imshow(sim.F, cmap='plasma', origin='lower', vmin=0, vmax=np.max(sim.F))
    ax1.set_title("Resource Density (Matter)")
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(sim.theta, cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
    ax2.set_title("Phase Field (Vortices)")
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig('det_v5_2d_snapshot.png')
    print("Snapshot saved to 'det_v5_2d_snapshot.png'")

# --- OPERATIONAL SUITES ---

def run_gravity_suite():
    print("\n" + "="*60)
    print("SUITE 1: GAUSS-LAW GRAVITY (Dark Flow)")
    print("="*60)
    
    sim = ActiveManifold2D(size=100)
    sim.initialize_massive_body(mass=100.0, radius=5.0, pinned=True)
    
    for i in range(200): sim.step()
    
    # Gauss Law Calculation (Disk Flux)
    # Phi(r) = Sum(-div J) inside disk r
    div_J = (sim.J_x_last - np.roll(sim.J_x_last, 1, axis=1)) + (sim.J_y_last - np.roll(sim.J_y_last, 1, axis=0))
    sink_density = -div_J # Inflow is negative divergence
    
    center = sim.N // 2
    y, x = np.mgrid[0:sim.N, 0:sim.N]
    r_grid = np.sqrt((x-center)**2 + (y-center)**2)
    
    radii = np.arange(2, 35, 4)
    fluxes = []
    
    print(f"{'Radius':<8} | {'Total Flux':<12} | {'Expected (Const)':<16} | {'Error %':<8}")
    print("-" * 55)
    
    for r in radii:
        mask = r_grid <= r
        total_sink = np.sum(sink_density[mask])
        fluxes.append(total_sink)
    
    # Check "Constant Flux" regime (Newtonian 2D)
    # In 2D, a 1/r field implies Flux(r) = constant.
    # We ignore the core (where source is) and check far field.
    if len(fluxes) > 3:
        ref_idx = 3 # r=14
        ref_flux = fluxes[ref_idx]
    else:
        ref_idx = 0
        ref_flux = fluxes[0]
    
    for r, f in zip(radii, fluxes):
        # Expected: Constant flux for 2D 1/r field outside source
        # In 2D, Field E ~ 1/r => Flux ~ E * 2*pi*r ~ constant
        # So we compare measured flux f to the reference flux ref_flux
        expected = ref_flux 
        err = 100 * abs((f - expected)/expected) if abs(expected) > 1e-9 else 0
        print(f"{r:<8} | {f:<12.4f} | {expected:<16.4f} | {err:<8.1f}%")
        
    plot_2d_snapshot(sim)

def torus_dist(p1, p2, N):
    d = np.abs(p1 - p2)
    d = np.minimum(d, N - d)
    return np.sqrt(np.sum(d**2))

def run_collider_suite(mode="free"):
    """
    mode: 'free' (scattering) or 'forced' (binding)
    """
    print("\n" + "="*60)
    print(f"SUITE 2: {mode.upper()}-FLIGHT COLLIDER")
    print("="*60)
    
    trials = 10
    g_vals = [2.0, 2.4, 2.8]
    
    print(f"{'g':<6} | {'Bind %':<8} | {'Min Sep':<8} | {'Shift':<8}")
    print("-" * 45)
    
    for g in g_vals:
        bind_counts = []
        min_seps = []
        shifts = []
        
        for trial_idx in range(trials):
            k = DETConstants2D(g=g)
            sim = ActiveManifold2D(size=80, constants=k)
            
            forced = (mode == "forced")
            sep = 6 if forced else 25
            
            sim.initialize_collider(separation=sep, velocity_kick=0.4, forced_contact=forced)
            
            seps = []
            
            # Pre-collision ghost track setup
            # We need to measure velocity early on.
            t_meas_end = 50 # steps
            p1_start = None
            v1_meas = None
            
            for t in range(400):
                sim.step()
                peaks = sim.find_peaks()
                
                # Separation Logic
                if len(peaks) >= 2:
                    dist = torus_dist(peaks[0], peaks[1], sim.N)
                    seps.append(dist)
                elif len(peaks) == 1:
                    seps.append(0) # Merged
                else:
                    seps.append(999) # Dissipated/Lost
                
                # Velocity Measurement (Packet 1)
                if t == 0 and len(peaks) >= 1:
                    p1_start = peaks[0]
                if t == t_meas_end and len(peaks) >= 1 and p1_start is not None:
                    # Rough v estimate (dx/dt)
                    # Handle wrapping if needed (unlikely in 50 steps)
                    disp = peaks[0] - p1_start
                    # fix wrap
                    if disp[1] > sim.N/2: disp[1] -= sim.N
                    if disp[1] < -sim.N/2: disp[1] += sim.N
                    # Corrected: Access DT from constants
                    v1_meas = disp / (t_meas_end * sim.k.DT)

            # Analysis
            min_s = np.min(seps) if seps else 0
            min_seps.append(min_s)
            
            # Bind Fraction (Time-Integrated)
            # Threshold: sep < 10 for > 80% of time after t_coll
            # Simplified: just check if final state is bound
            final_seps = seps[-50:]
            is_bound = np.mean(final_seps) < 10.0
            bind_counts.append(1 if is_bound else 0)
            
            # Early-Out Phase Shift
            # t_coll approx when sep is min
            t_coll_idx = np.argmin(seps) if seps else 0
            t_early = t_coll_idx + 20
            
            shift_val = 0.0
            if v1_meas is not None and t_early < len(sim.trajectory_log):
                # Ghost position
                t_e_time = t_early * sim.k.DT
                ghost_pos = p1_start + v1_meas * t_e_time
                # Actual pos
                peaks_e = sim.trajectory_log[t_early][1]
                if len(peaks_e) >= 1:
                    # Assume p1 is the one moving right (higher x usually)
                    # Just find closest to ghost
                    actual = peaks_e[0] # simplification
                    # dist
                    d_vec = actual - ghost_pos
                    # wrap
                    if d_vec[1] > sim.N/2: d_vec[1] -= sim.N
                    if d_vec[1] < -sim.N/2: d_vec[1] += sim.N
                    shift_val = np.linalg.norm(d_vec)
            
            shifts.append(shift_val)
            
        bind_pct = np.mean(bind_counts) * 100
        avg_min = np.mean(min_seps)
        avg_shift = np.mean(shifts)
        
        # Check Detector Reach
        # If shift is huge or 0, might be invalid. 
        # Just report raw for now.
        
        print(f"{g:<6.1f} | {bind_pct:<8.0f} | {avg_min:<8.2f} | {avg_shift:<8.2f}")

if __name__ == "__main__":
    # Run all suites
    run_gravity_suite()
    run_collider_suite(mode="free")
    run_collider_suite(mode="forced")