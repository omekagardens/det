import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import linregress

"""
DET v5 2D FLATLAND LABORATORY
=============================
Implementation of the "Unified Active Manifold" in 2D.

Purpose:
--------
1. Move beyond 1D restrictions to allow Phase Vortices (Electromagnetism).
2. Test if "Dark Flow" (Vacuum Pressure) scales as 1/r in 2D (Gravity).
3. Demonstrate vector solitons and metric shear.

Operational Improvements (v5.8 - "The Fusion Demo"):
---------------------------------------------------
1. Targeted Demo: Added run_fusion_demo() to visualize the confirmed g=1.2 bound state.
2. Parameter Tuning:
   - Shifted Collider Scan to the active window (g=0.8 - 1.6).
   - Confirmed Binding at g ~ 1.0-1.2.
3. Physics:
   - Compression-Dependent Drag ("Crumple Zone") retained.
   - Metric Latching retained.

The Five Master Equations (2D Generalized):
1. Time-Energy:     dθ/dt = -β · F + ν · ∇²θ - γ_drag(comp) · (∇θ)²
2. Directed Flow:   J_vec = σ · (C_vec)² · [ ... ]
3. Hebbian Metric:  dC_vec/dt = α · J_vec - λ · C_vec (Latched)
4. Capacity:        σ     = 1 + ln(1 + usage)
5. Conservation:    dF/dt = -∇·J
"""

class DETConstants2D:
    def __init__(self, g=1.2): # Default to the Fusion Sweet Spot
        self.g = g
        # Time/Energy
        self.BETA = 2.0 * g
        self.NU = 0.6
        
        # Base Phase Drag
        self.PHASE_DRAG_BASE = 0.1
        # Compression Drag Multiplier ("Crumple Zone")
        self.PHASE_DRAG_COMP = 50.0 
        
        # Plasticity
        self.ALPHA = 0.5 * g
        self.GAMMA = 8.0 
        self.LAMBDA = 0.00001 # Superfluid
        
        # Fusion
        self.K_FUSION = 120.0
        
        # Vacuum
        self.F_VAC = 0.001
        self.C_MIN = 0.05
        
        # Simulation
        self.DT = 0.05

class ActiveManifold2D:
    def __init__(self, size=80, constants=DETConstants2D()):
        self.N = size
        self.k = constants
        
        # State
        self.F = np.ones((self.N, self.N)) * self.k.F_VAC
        self.theta = np.zeros((self.N, self.N))
        self.sigma = np.ones((self.N, self.N))
        self.C_x = np.ones((self.N, self.N)) * self.k.C_MIN
        self.C_y = np.ones((self.N, self.N)) * self.k.C_MIN
        
        self.time = 0.0
        
        # History for Viz
        self.history_F = []
        
        # Pinning
        self.static_mask = np.zeros((self.N, self.N), dtype=bool)
        self.F_static = np.zeros((self.N, self.N))
        
        # Flux tracking
        self.J_x_last = np.zeros((self.N, self.N))
        self.J_y_last = np.zeros((self.N, self.N))
        
        # Trajectory tracking
        self.trajectory_log = []

    def initialize_massive_body(self, mass=50.0, radius=5.0, pinned=True):
        center = self.N // 2
        y, x = np.mgrid[0:self.N, 0:self.N]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        mass_dist = mass * np.exp(-0.5 * (r/radius)**2)
        self.F += mass_dist
        
        # Pre-seed metric
        seed = 0.8 * np.exp(-0.5 * (r/radius)**2)
        self.C_x += seed
        self.C_y += seed
        self.C_x = np.clip(self.C_x, self.k.C_MIN, 1.0)
        self.C_y = np.clip(self.C_y, self.k.C_MIN, 1.0)
        
        if pinned:
            self.static_mask = mass_dist > (self.k.F_VAC * 5)
            self.F_static = self.F.copy()

    def initialize_collider(self, separation=20, velocity_kick=0.5, forced_contact=False):
        center_y, center_x = self.N // 2, self.N // 2
        
        # Symmetry breaking (Mass)
        m1 = 20.0
        m2 = 24.0 
        
        d = 10 if forced_contact else separation # Increased forced sep to avoid core repulsion
        
        p1 = (center_y, center_x - d//2)
        p2 = (center_y, center_x + d//2)
        
        y, x = np.mgrid[0:self.N, 0:self.N]
        
        # Packet 1 (Left)
        r1 = np.sqrt((y-p1[0])**2 + (x-p1[1])**2)
        self.F += m1 * np.exp(-0.5 * (r1/3.0)**2)
        
        # Packet 2 (Right)
        r2 = np.sqrt((y-p2[0])**2 + (x-p2[1])**2)
        self.F += m2 * np.exp(-0.5 * (r2/3.0)**2)
        
        # Local Phase Ramps
        # Left moves +x
        self.theta += (velocity_kick * x) * (m1 * np.exp(-0.5 * (r1/3.0)**2) / (self.F + 1e-9))
        # Right moves -x
        self.theta += (-velocity_kick * x) * (m2 * np.exp(-0.5 * (r2/3.0)**2) / (self.F + 1e-9))
        
        # Perturbation
        self.theta += np.random.normal(0, 0.05, (self.N, self.N))

    def find_peaks(self):
        neighborhood = 5
        data_max = ndimage.maximum_filter(self.F, neighborhood)
        maxima = (self.F == data_max)
        data_min = ndimage.minimum_filter(self.F, neighborhood)
        diff = ((data_max - data_min) > 0.5)
        maxima[diff == 0] = 0
        
        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        
        peaks = []
        for dy, dx in slices:
            xc = (dx.start + dx.stop - 1) / 2
            yc = (dy.start + dy.stop - 1) / 2
            peaks.append(np.array([yc, xc]))
            
        # Sort by x
        peaks.sort(key=lambda p: p[1])
        return peaks

    def step(self):
        # 1. TIME-ENERGY (PRE-CALC for Phase Update)
        N_idx = np.roll(np.arange(self.N), 1)
        S_idx = np.roll(np.arange(self.N), -1)
        W_idx = np.roll(np.arange(self.N), 1)
        E_idx = np.roll(np.arange(self.N), -1)
        
        # --- PROVISIONAL FLOW (for Drag) ---
        d_theta_E = np.angle(np.exp(1j * (self.theta[:, E_idx] - self.theta)))
        d_press_E = self.F - self.F[:, E_idx]
        J_x_prov = self.sigma * (self.C_x**2 + 1e-4) * (np.sin(d_theta_E) + d_press_E)
        
        d_theta_S = np.angle(np.exp(1j * (self.theta[S_idx, :] - self.theta)))
        d_press_S = self.F - self.F[S_idx, :]
        J_y_prov = self.sigma * (self.C_y**2 + 1e-4) * (np.sin(d_theta_S) + d_press_S)
        
        J_x_in_p = np.roll(J_x_prov, 1, axis=1)
        J_y_in_p = np.roll(J_y_prov, 1, axis=0)
        div_J_p = (J_x_prov - J_x_in_p) + (J_y_prov - J_y_in_p)
        compression = np.maximum(0, -div_J_p) # Positive where squeezing
        
        # --- THETA UPDATE ---
        d_E = np.angle(np.exp(1j * (self.theta[:, E_idx] - self.theta)))
        d_W = np.angle(np.exp(1j * (self.theta[:, W_idx] - self.theta)))
        d_S = np.angle(np.exp(1j * (self.theta[S_idx, :] - self.theta)))
        d_N = np.angle(np.exp(1j * (self.theta[N_idx, :] - self.theta)))
        
        laplacian = (d_E + d_W + d_S + d_N)
        grad_sq = 0.25 * ((d_E - d_W)**2 + (d_S - d_N)**2)
        
        phase_drive = np.minimum(self.k.BETA * self.F * self.k.DT, np.pi/4.0)
        
        # INELASTIC COLLISION LOGIC:
        effective_drag = self.k.PHASE_DRAG_BASE + self.k.PHASE_DRAG_COMP * compression
        drag_factor = 1.0 / (1.0 + effective_drag * grad_sq)
        
        self.theta -= phase_drive * drag_factor
        self.theta += self.k.NU * laplacian * self.k.DT
        self.theta = np.mod(self.theta, 2*np.pi)
        
        # 2. REAL FLOW UPDATE
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
        
        # Flux Limit
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
        
        # 3. METRIC (Latching)
        real_compression = np.maximum(0, -div_J)
        alpha = self.k.ALPHA * (1.0 + self.k.K_FUSION * real_compression)
        
        decay_x = self.k.LAMBDA * self.C_x * (1.0 - self.C_x**2)
        decay_y = self.k.LAMBDA * self.C_y * (1.0 - self.C_y**2)
        
        self.C_x += (alpha * np.abs(J_x) - decay_x) * self.k.DT
        self.C_y += (alpha * np.abs(J_y) - decay_y) * self.k.DT
        
        self.C_x = np.clip(self.C_x, self.k.C_MIN, 1.0)
        self.C_y = np.clip(self.C_y, self.k.C_MIN, 1.0)
        
        # 4. CAPACITY
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_x) + np.abs(J_y))
        
        self.time += self.k.DT
        
        if int(self.time/self.k.DT) % 10 == 0:
            self.history_F.append(self.F.copy())
            
        self.trajectory_log.append((self.time, self.find_peaks()))

    def get_metrics(self):
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

def run_fusion_demo(g=1.2):
    """
    Run a single, detailed simulation of a successful capture event.
    """
    print("\n" + "="*60)
    print(f"DEMO: FUSION CAPTURE AT g={g}")
    print("="*60)
    
    k = DETConstants2D(g=g)
    sim = ActiveManifold2D(size=100, constants=k)
    
    # Use free-flight conditions that showed binding
    sim.initialize_collider(separation=25, velocity_kick=0.4, forced_contact=False)
    
    # Run longer to see stable orbit/merger
    for t in range(600):
        sim.step()
        if t % 100 == 0:
            peaks = sim.find_peaks()
            print(f"Step {t}: {len(peaks)} peaks")
            
    plot_2d_snapshot(sim)

def get_boundary_flux(sim, radius, center):
    num_samples = int(2 * np.pi * radius * 2)
    angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    flux_sum = 0.0
    for phi in angles:
        px = center + radius * np.cos(phi)
        py = center + radius * np.sin(phi)
        ix, iy = int(px)%sim.N, int(py)%sim.N
        Jx = sim.J_x_last[iy, ix]
        Jy = sim.J_y_last[iy, ix]
        nx, ny = np.cos(phi), np.sin(phi)
        flux_sum += (Jx * nx + Jy * ny)
    dl = (2 * np.pi * radius) / num_samples
    total_flux = flux_sum * dl
    return total_flux

def run_gravity_suite():
    print("\n" + "="*60)
    print("SUITE 1: GAUSS-LAW GRAVITY (Dark Flow)")
    print("="*60)
    sim = ActiveManifold2D(size=100)
    sim.initialize_massive_body(mass=100.0, radius=5.0, pinned=True)
    for i in range(200): sim.step()
    center = sim.N // 2
    radii = np.arange(10, 40, 4)
    fluxes = []
    print(f"{'Radius':<8} | {'Gamma(r)':<12} | {'Const? (Ref)':<12}")
    print("-" * 45)
    ref_val = None
    for r in radii:
        gamma = get_boundary_flux(sim, r, center)
        fluxes.append(gamma)
        if r == 14: ref_val = gamma
    for r, val in zip(radii, fluxes):
        ref = ref_val if ref_val else val
        err = 100 * abs((val - ref)/ref)
        print(f"{r:<8} | {val:<12.4f} | {err:<8.1f}%")
    plot_2d_snapshot(sim)

def torus_dist(p1, p2, N):
    d = np.abs(p1 - p2)
    d = np.minimum(d, N - d)
    return np.sqrt(np.sum(d**2))

def run_collider_suite(mode="free"):
    print("\n" + "="*60)
    print(f"SUITE 2: {mode.upper()}-FLIGHT COLLIDER")
    print("="*60)
    trials = 10
    g_vals = [0.8, 1.0, 1.2, 1.4, 1.6] # Updated scan range
    print(f"{'g':<6} | {'Bind %':<8} | {'Contact %':<10} | {'Final Sep':<10}")
    print("-" * 55)
    for g in g_vals:
        bind_count = 0
        contact_accum = 0.0
        final_sep_accum = 0.0
        for _ in range(trials):
            k = DETConstants2D(g=g)
            sim = ActiveManifold2D(size=80, constants=k)
            forced = (mode == "forced")
            sep = 6 if forced else 25
            sim.initialize_collider(separation=sep, velocity_kick=0.4, forced_contact=forced)
            seps = []
            t_coll = -1
            min_dist = 999
            for t in range(400):
                sim.step()
                peaks = sim.find_peaks()
                if len(peaks) >= 2:
                    p1, p2 = np.array(peaks[0]), np.array(peaks[1])
                    dist = np.linalg.norm(p1 - p2) 
                    seps.append(dist)
                    if dist < min_dist:
                        min_dist = dist
                        t_coll = t
                else:
                    seps.append(0) 
            if seps:
                final_sep = seps[-1]
                final_sep_accum += final_sep
            if t_coll > 0 and t_coll < 350:
                post_coll = np.array(seps[t_coll:])
                contact_frames = np.sum(post_coll < 10.0)
                contact_frac = contact_frames / len(post_coll)
                contact_accum += contact_frac
                if contact_frac > 0.8:
                    bind_count += 1
        bind_pct = (bind_count / trials) * 100
        avg_contact = (contact_accum / trials) * 100
        avg_final = final_sep_accum / trials
        print(f"{g:<6.1f} | {bind_pct:<8.0f} | {avg_contact:<10.1f} | {avg_final:<10.2f}")

if __name__ == "__main__":
    run_gravity_suite()
    run_collider_suite(mode="free")
    run_collider_suite(mode="forced")
    run_fusion_demo(g=1.2)
