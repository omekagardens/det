import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

"""
DET v5 2D - OPTIMIZED STABLE VERSION
====================================

Changes from original:
1. PHASE_DRIVE_MODE = 'none' (removes destabilizing phase winding)
2. NU = 0.15 (reduced from 0.6 - preserves phase gradients for motion)

This gives:
- Stable single packets
- Actual collision/binding dynamics
- Mild fragmentation (2→3-4 peaks) acceptable

Full test suite:
- Single packet stability
- Two-packet collision
- Gravity suite (measuring flux)
- Binding rate statistics
"""

class DETConstants2D:
    def __init__(self, g=1.2):
        self.g = g
        
        # OPTIMIZED PARAMETERS
        self.BETA = 2.0 * g          # Not used when PHASE_DRIVE_MODE='none'
        self.NU = 0.15               # REDUCED: Phase diffusion (was 0.6)
        self.PHASE_DRIVE_MODE = 'none'  # CHANGED: No phase drive
        
        # Phase drag (unchanged)
        self.PHASE_DRAG_BASE = 0.1
        self.PHASE_DRAG_COMP = 50.0 
        
        # Plasticity (unchanged)
        self.ALPHA = 0.5 * g
        self.LAMBDA = 0.00001
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
        
        self.F = np.ones((self.N, self.N)) * self.k.F_VAC
        self.theta = np.zeros((self.N, self.N))
        self.sigma = np.ones((self.N, self.N))
        self.C_x = np.ones((self.N, self.N)) * self.k.C_MIN
        self.C_y = np.ones((self.N, self.N)) * self.k.C_MIN
        
        self.time = 0.0
        self.J_x_last = np.zeros((self.N, self.N))
        self.J_y_last = np.zeros((self.N, self.N))

    def initialize_single_packet(self, mass=20.0, radius=3.0, velocity=0.0, position=None):
        """Initialize single Gaussian packet."""
        if position is None:
            center_y, center_x = self.N // 2, self.N // 2
        else:
            center_y, center_x = position
            
        y, x = np.mgrid[0:self.N, 0:self.N]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        packet = mass * np.exp(-0.5 * (r/radius)**2)
        self.F += packet
        
        if velocity != 0:
            mask = packet / (packet.max() + 1e-9)
            self.theta += velocity * x * mask
            self.theta = np.mod(self.theta, 2*np.pi)

    def initialize_massive_body(self, mass=50.0, radius=5.0, pinned=True):
        """Initialize central massive body for gravity test."""
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
        else:
            self.static_mask = np.zeros((self.N, self.N), dtype=bool)
            self.F_static = np.zeros((self.N, self.N))

    def initialize_two_packets(self, separation=25, velocity_kick=0.4):
        """Two packets moving toward each other."""
        center_y, center_x = self.N // 2, self.N // 2
        
        m1, m2 = 20.0, 24.0
        r_packet = 3.0
        
        p1 = (center_y, center_x - separation//2)
        p2 = (center_y, center_x + separation//2)
        
        y, x = np.mgrid[0:self.N, 0:self.N]
        
        r1 = np.sqrt((y-p1[0])**2 + (x-p1[1])**2)
        packet1 = m1 * np.exp(-0.5 * (r1/r_packet)**2)
        self.F += packet1
        
        r2 = np.sqrt((y-p2[0])**2 + (x-p2[1])**2)
        packet2 = m2 * np.exp(-0.5 * (r2/r_packet)**2)
        self.F += packet2
        
        mask1 = packet1 / (packet1.max() + 1e-9)
        self.theta += velocity_kick * x * mask1
        
        mask2 = packet2 / (packet2.max() + 1e-9)
        self.theta -= velocity_kick * x * mask2
        
        self.theta = np.mod(self.theta, 2*np.pi)

    def find_peaks(self, threshold=0.5):
        """Return number of peaks."""
        neighborhood = 5
        data_max = ndimage.maximum_filter(self.F, neighborhood)
        maxima = (self.F == data_max)
        data_min = ndimage.minimum_filter(self.F, neighborhood)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        return num_objects

    def find_peak_positions(self, threshold=0.5):
        """Return (y, x) positions of peaks."""
        neighborhood = 5
        data_max = ndimage.maximum_filter(self.F, neighborhood)
        maxima = (self.F == data_max)
        data_min = ndimage.minimum_filter(self.F, neighborhood)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        
        positions = []
        for i in range(1, num_objects + 1):
            mask = labeled == i
            total = np.sum(self.F[mask])
            if total > 0:
                y_coords, x_coords = np.where(mask)
                y_cen = np.sum(self.F[mask] * y_coords) / total
                x_cen = np.sum(self.F[mask] * x_coords) / total
                positions.append((y_cen, x_cen))
        return positions

    def step(self):
        # Index arrays
        N_idx = np.roll(np.arange(self.N), 1)
        S_idx = np.roll(np.arange(self.N), -1)
        W_idx = np.roll(np.arange(self.N), 1)
        E_idx = np.roll(np.arange(self.N), -1)
        
        # Phase differences
        d_theta_E = np.angle(np.exp(1j * (self.theta[:, E_idx] - self.theta)))
        d_theta_W = np.angle(np.exp(1j * (self.theta[:, W_idx] - self.theta)))
        d_theta_S = np.angle(np.exp(1j * (self.theta[S_idx, :] - self.theta)))
        d_theta_N = np.angle(np.exp(1j * (self.theta[N_idx, :] - self.theta)))
        
        # Pressure differences
        d_press_E = self.F - self.F[:, E_idx]
        d_press_S = self.F - self.F[S_idx, :]
        
        # Flow
        cond_x = self.sigma * (self.C_x**2 + 1e-4)
        cond_y = self.sigma * (self.C_y**2 + 1e-4)
        
        J_x = cond_x * (np.sin(d_theta_E) + d_press_E)
        J_y = cond_y * (np.sin(d_theta_S) + d_press_S)
        
        # Flux limit
        J_mag = np.sqrt(J_x**2 + J_y**2)
        max_allowed = 0.25 * self.F / self.k.DT
        scale = np.minimum(1.0, max_allowed / (J_mag + 1e-9))
        J_x *= scale
        J_y *= scale
        
        self.J_x_last = J_x.copy()
        self.J_y_last = J_y.copy()
        
        # Divergence
        J_x_in = np.roll(J_x, 1, axis=1)
        J_y_in = np.roll(J_y, 1, axis=0)
        div_J = (J_x - J_x_in) + (J_y - J_y_in)
        compression = np.maximum(0, -div_J)
        
        # Phase update (NO DRIVE, only diffusion)
        laplacian = (d_theta_E + d_theta_W + d_theta_S + d_theta_N)
        self.theta += self.k.NU * laplacian * self.k.DT
        self.theta = np.mod(self.theta, 2*np.pi)
        
        # F update
        self.F -= div_J * self.k.DT
        if hasattr(self, 'static_mask') and np.any(self.static_mask):
            self.F[self.static_mask] = self.F_static[self.static_mask]
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        # Metric update
        alpha = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        decay_x = self.k.LAMBDA * self.C_x * (1.0 - self.C_x**2)
        decay_y = self.k.LAMBDA * self.C_y * (1.0 - self.C_y**2)
        
        self.C_x += (alpha * np.abs(J_x) - decay_x) * self.k.DT
        self.C_y += (alpha * np.abs(J_y) - decay_y) * self.k.DT
        self.C_x = np.clip(self.C_x, self.k.C_MIN, 1.0)
        self.C_y = np.clip(self.C_y, self.k.C_MIN, 1.0)
        
        # Capacity
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_x) + np.abs(J_y))
        
        self.time += self.k.DT

    def get_diagnostics(self):
        total_F = np.sum(self.F)
        phase_order = np.abs(np.mean(np.exp(1j * self.theta)))
        num_peaks = self.find_peaks()
        return {'total_F': total_F, 'phase_coherence': phase_order, 'peaks': num_peaks}


def get_boundary_flux(sim, radius, center):
    """Integrate flux through circular boundary."""
    num_samples = int(2 * np.pi * radius * 2)
    angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    flux_sum = 0.0
    for phi in angles:
        px = center + radius * np.cos(phi)
        py = center + radius * np.sin(phi)
        ix, iy = int(px) % sim.N, int(py) % sim.N
        Jx = sim.J_x_last[iy, ix]
        Jy = sim.J_y_last[iy, ix]
        nx, ny = np.cos(phi), np.sin(phi)
        flux_sum += (Jx * nx + Jy * ny)
    dl = (2 * np.pi * radius) / num_samples
    return flux_sum * dl


def plot_state(sim, title, filename):
    """Plot current state."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    im1 = axes[0].imshow(sim.F, cmap='plasma', origin='lower')
    axes[0].set_title('F (Matter)')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(sim.theta, cmap='hsv', origin='lower', vmin=0, vmax=2*np.pi)
    axes[1].set_title('θ (Phase)')
    plt.colorbar(im2, ax=axes[1])
    
    J_mag = np.sqrt(sim.J_x_last**2 + sim.J_y_last**2)
    im3 = axes[2].imshow(np.log10(J_mag + 1e-10), cmap='inferno', origin='lower')
    axes[2].set_title('log₁₀|J| (Flow)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved: {filename}")
    plt.close()


# ============================================================
# TEST SUITE
# ============================================================

def test_single_packet_stability():
    """Test 1: Single packet stability."""
    print("\n" + "="*60)
    print("TEST 1: SINGLE PACKET STABILITY")
    print("="*60)
    
    sim = ActiveManifold2D(size=80, constants=DETConstants2D())
    sim.initialize_single_packet(mass=20.0, radius=3.0)
    
    initial_F = np.sum(sim.F)
    
    print(f"\n{'Step':<8}{'Peaks':<8}{'Phase Coh':<12}{'Total F':<12}")
    print("-" * 40)
    
    for t in range(301):
        if t % 50 == 0:
            d = sim.get_diagnostics()
            print(f"{t:<8}{d['peaks']:<8}{d['phase_coherence']:<12.3f}{d['total_F']:<12.2f}")
        sim.step()
    
    final = sim.get_diagnostics()
    
    if final['peaks'] <= 2:
        print(f"\n✓ PASS: Single packet stable (peaks={final['peaks']})")
        return True
    else:
        print(f"\n✗ FAIL: Packet fragmented (peaks={final['peaks']})")
        return False


def test_collision_dynamics():
    """Test 2: Two-packet collision."""
    print("\n" + "="*60)
    print("TEST 2: COLLISION DYNAMICS")
    print("="*60)
    
    sim = ActiveManifold2D(size=100, constants=DETConstants2D())
    sim.initialize_two_packets(separation=30, velocity_kick=0.5)
    
    separations = []
    
    print(f"\n{'Step':<8}{'Peaks':<8}{'Separation':<12}{'Phase Coh':<12}")
    print("-" * 44)
    
    for t in range(401):
        positions = sim.find_peak_positions()
        
        if len(positions) >= 2:
            sorted_pos = sorted(positions, key=lambda p: p[1])
            sep = abs(sorted_pos[1][1] - sorted_pos[0][1])
            separations.append(sep)
        
        if t % 50 == 0:
            d = sim.get_diagnostics()
            sep_str = f"{sep:.1f}" if len(positions) >= 2 else "N/A"
            print(f"{t:<8}{d['peaks']:<8}{sep_str:<12}{d['phase_coherence']:<12.3f}")
        
        sim.step()
    
    plot_state(sim, "Collision Test Final State", "optimized_collision.png")
    
    if separations:
        initial = separations[0]
        minimum = min(separations)
        final = separations[-1]
        
        print(f"\nInitial sep: {initial:.1f}")
        print(f"Minimum sep: {minimum:.1f}")
        print(f"Final sep: {final:.1f}")
        
        if minimum < initial * 0.3:
            print(f"\n✓ PASS: Collision occurred (min_sep={minimum:.1f})")
            if final < initial * 0.5:
                print(f"✓ BOUND: Particles bound (final_sep={final:.1f})")
            return True
        else:
            print(f"\n✗ FAIL: No significant collision")
            return False
    return False


def test_gravity_flux():
    """Test 3: Gravity (flux through shells)."""
    print("\n" + "="*60)
    print("TEST 3: GRAVITY (Flux Through Shells)")
    print("="*60)
    
    sim = ActiveManifold2D(size=100, constants=DETConstants2D())
    sim.initialize_massive_body(mass=100.0, radius=5.0, pinned=True)
    
    for i in range(200):
        sim.step()
    
    center = sim.N // 2
    radii = np.arange(10, 40, 4)
    
    print(f"\n{'Radius':<10}{'Γ(r)':<15}{'Deviation %':<12}")
    print("-" * 40)
    
    fluxes = []
    ref_val = None
    
    for r in radii:
        gamma = get_boundary_flux(sim, r, center)
        fluxes.append(gamma)
        if r == 14:
            ref_val = gamma
    
    for r, val in zip(radii, fluxes):
        if ref_val and abs(ref_val) > 1e-10:
            err = 100 * abs((val - ref_val) / ref_val)
        else:
            err = 0
        print(f"{r:<10}{val:<15.4f}{err:<12.1f}")
    
    plot_state(sim, "Gravity Test Final State", "optimized_gravity.png")
    
    # Note: We expect flux to decay (no Gauss law without explicit gravity)
    print("\nNote: Without explicit Poisson gravity, flux decays with distance.")
    print("This is expected - gravity sector needs separate implementation.")


def test_binding_statistics():
    """Test 4: Binding rate statistics."""
    print("\n" + "="*60)
    print("TEST 4: BINDING STATISTICS")
    print("="*60)
    
    trials = 10
    velocities = [0.3, 0.4, 0.5, 0.6]
    
    print(f"\n{'Velocity':<10}{'Bind Rate':<12}{'Avg Min Sep':<14}{'Avg Final':<12}")
    print("-" * 50)
    
    for v in velocities:
        binds = 0
        min_seps = []
        final_seps = []
        
        for _ in range(trials):
            sim = ActiveManifold2D(size=100, constants=DETConstants2D())
            sim.initialize_two_packets(separation=30, velocity_kick=v)
            
            seps = []
            for t in range(400):
                positions = sim.find_peak_positions()
                if len(positions) >= 2:
                    sorted_pos = sorted(positions, key=lambda p: p[1])
                    sep = abs(sorted_pos[1][1] - sorted_pos[0][1])
                    seps.append(sep)
                sim.step()
            
            if seps:
                min_sep = min(seps)
                final_sep = seps[-1]
                min_seps.append(min_sep)
                final_seps.append(final_sep)
                
                if final_sep < 15:  # Bound if final separation < 15
                    binds += 1
        
        bind_rate = (binds / trials) * 100
        avg_min = np.mean(min_seps) if min_seps else 0
        avg_final = np.mean(final_seps) if final_seps else 0
        
        print(f"{v:<10.1f}{bind_rate:<12.0f}%{avg_min:<14.1f}{avg_final:<12.1f}")


def run_full_suite():
    """Run all tests."""
    print("\n" + "="*70)
    print("DET v5 2D OPTIMIZED - FULL TEST SUITE")
    print("="*70)
    print("\nParameters:")
    k = DETConstants2D()
    print(f"  PHASE_DRIVE_MODE = '{k.PHASE_DRIVE_MODE}'")
    print(f"  NU = {k.NU}")
    print(f"  BETA = {k.BETA}")
    
    results = {}
    
    results['stability'] = test_single_packet_stability()
    results['collision'] = test_collision_dynamics()
    test_gravity_flux()
    test_binding_statistics()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nStability Test: {'✓ PASS' if results['stability'] else '✗ FAIL'}")
    print(f"Collision Test: {'✓ PASS' if results['collision'] else '✗ FAIL'}")
    print(f"Gravity Test: Flux decays (expected without Poisson sector)")
    print(f"Binding Test: See statistics above")
    
    return results


if __name__ == "__main__":
    run_full_suite()
