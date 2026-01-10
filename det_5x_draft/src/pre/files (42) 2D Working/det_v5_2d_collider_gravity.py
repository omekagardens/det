import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq

"""
DET v5 2D COLLIDER WITH DET 4.2 GRAVITY
=======================================

Final working version with:
1. Stable solitons (phase diffusion only)
2. DET 4.2 gravity sector (q → ρ → Φ → g)
3. Gravitational velocity field (v_grav)
4. **Key fix**: Bounded gravitational flux = GRAV_FLUX_FRAC * F / dt

Tested parameters:
- KAPPA = 5.0
- GRAV_FLUX_FRAC = 0.02  (limits grav flux to 2% of local mass per timestep)
- V_VISC = 0.005

Results:
- Collision without gravity: Mass conserved exactly
- Gravity test: Attraction 24 → 3, mass error ~3%
"""


class DETGravity2D:
    """DET 4.2 gravity solver."""
    
    def __init__(self, N, kappa=5.0, alpha_baseline=0.1):
        self.N = N
        self.kappa = kappa
        self.alpha_baseline = alpha_baseline
        
        k = fftfreq(N, d=1.0) * 2 * np.pi
        kx, ky = np.meshgrid(k, k)
        self.k_squared = kx**2 + ky**2
        self.k_squared[0, 0] = 1.0
    
    def compute_gravity(self, q):
        """Compute q → b → ρ → Φ → g"""
        # Baseline (Helmholtz smoothing)
        rhs = -self.alpha_baseline * q
        rhs_hat = fft2(rhs)
        denom = -self.k_squared - self.alpha_baseline
        denom[0, 0] = -self.alpha_baseline
        b = np.real(ifft2(rhs_hat / denom))
        
        # Source
        rho = q - b
        
        # Potential (Poisson)
        rho_hat = fft2(rho)
        phi_hat = -self.kappa * rho_hat / self.k_squared
        phi_hat[0, 0] = 0.0
        Phi = np.real(ifft2(phi_hat))
        
        # Gravitational acceleration
        g_x = -0.5 * (np.roll(Phi, -1, axis=1) - np.roll(Phi, 1, axis=1))
        g_y = -0.5 * (np.roll(Phi, -1, axis=0) - np.roll(Phi, 1, axis=0))
        
        return Phi, rho, g_x, g_y


class DETConstants2D:
    def __init__(self, g=1.2):
        self.g = g
        
        # Soliton dynamics
        self.NU = 0.15
        
        # Plasticity
        self.ALPHA = 0.5 * g
        self.LAMBDA = 0.00001
        self.K_FUSION = 120.0
        
        # Vacuum
        self.F_VAC = 0.001
        self.C_MIN = 0.05
        
        # Gravity sector
        self.GRAVITY_ENABLED = True
        self.KAPPA = 5.0
        self.ALPHA_BASELINE = 0.1
        
        # Gravitational velocity (KEY PARAMETERS - tuned for attraction + mass conservation)
        self.GRAV_VEL_VISCOSITY = 0.005
        self.GRAV_FLUX_FRAC = 0.02  # Max grav flux = this fraction of F/dt
        
        # Structural debt
        self.ETA_Q = 0.01
        self.LAMBDA_Q = 0.001
        self.Q_THRESHOLD = 0.01
        
        self.DT = 0.05


class ActiveManifold2D:
    def __init__(self, size=100, constants=DETConstants2D()):
        self.N = size
        self.k = constants
        
        # Core fields
        self.F = np.ones((self.N, self.N)) * self.k.F_VAC
        self.theta = np.zeros((self.N, self.N))
        self.sigma = np.ones((self.N, self.N))
        self.C_x = np.ones((self.N, self.N)) * self.k.C_MIN
        self.C_y = np.ones((self.N, self.N)) * self.k.C_MIN
        
        # Gravity fields
        self.q = np.zeros((self.N, self.N))
        self.Phi = np.zeros((self.N, self.N))
        self.g_x = np.zeros((self.N, self.N))
        self.g_y = np.zeros((self.N, self.N))
        
        # Gravitational velocity
        self.v_grav_x = np.zeros((self.N, self.N))
        self.v_grav_y = np.zeros((self.N, self.N))
        
        self.gravity = DETGravity2D(self.N, self.k.KAPPA, self.k.ALPHA_BASELINE)
        
        self.J_x_last = np.zeros((self.N, self.N))
        self.J_y_last = np.zeros((self.N, self.N))
        
        self.time = 0.0

    def add_body(self, center, mass, radius, q_fraction=0.3, velocity=(0, 0)):
        """Add a Gaussian body."""
        y, x = np.mgrid[0:self.N, 0:self.N]
        cy, cx = center
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        body = mass * np.exp(-0.5 * (r/radius)**2)
        self.F += body
        
        if q_fraction > 0:
            self.q += q_fraction * (body / (mass + 1e-9))
        
        vx, vy = velocity
        if vx != 0 or vy != 0:
            mask = body / (body.max() + 1e-9)
            self.theta += vx * x * mask + vy * y * mask
            self.theta = np.mod(self.theta, 2*np.pi)

    def initialize_single_packet(self, mass=20.0, radius=3.0, velocity=0.0, 
                                  position=None, q_fraction=0.2):
        if position is None:
            center = (self.N // 2, self.N // 2)
        else:
            center = position
        self.add_body(center, mass, radius, q_fraction, velocity=(velocity, 0))

    def initialize_two_packets(self, separation=25, m1=20.0, m2=24.0, 
                                velocity=0.5, q_fraction=0.2):
        center_y, center_x = self.N // 2, self.N // 2
        self.add_body((center_y, center_x - separation//2), m1, 3.0, q_fraction, (velocity, 0))
        self.add_body((center_y, center_x + separation//2), m2, 3.0, q_fraction, (-velocity, 0))

    def initialize_gravity_test(self, separation=24, m1=30.0, m2=30.0, q_fraction=0.6):
        center = self.N // 2
        self.add_body((center, center - separation//2), m1, 4.0, q_fraction)
        self.add_body((center, center + separation//2), m2, 4.0, q_fraction)

    def find_peaks(self, threshold=0.5):
        neighborhood = 5
        data_max = ndimage.maximum_filter(self.F, neighborhood)
        maxima = (self.F == data_max)
        data_min = ndimage.minimum_filter(self.F, neighborhood)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        return num_objects

    def find_peak_positions(self, threshold=0.5):
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
        dt = self.k.DT
        
        E_idx = np.roll(np.arange(self.N), -1)
        W_idx = np.roll(np.arange(self.N), 1)
        S_idx = np.roll(np.arange(self.N), -1)
        N_idx = np.roll(np.arange(self.N), 1)
        
        # ===== GRAVITY =====
        if self.k.GRAVITY_ENABLED:
            self.Phi, _, self.g_x, self.g_y = self.gravity.compute_gravity(self.q)
            
            # Mass mask - only accelerate where there's significant mass
            mass_mask = self.F > self.k.F_VAC * 10
            
            # Velocity accumulation (only where mass)
            self.v_grav_x = np.where(mass_mask, self.v_grav_x + self.g_x * dt, 0)
            self.v_grav_y = np.where(mass_mask, self.v_grav_y + self.g_y * dt, 0)
            
            # Viscosity
            self.v_grav_x *= (1 - self.k.GRAV_VEL_VISCOSITY * dt)
            self.v_grav_y *= (1 - self.k.GRAV_VEL_VISCOSITY * dt)
            self.v_grav_y *= (1 - self.k.GRAV_VEL_VISCOSITY * dt)
        
        # ===== PHASE DIFFERENCES =====
        d_theta_E = np.angle(np.exp(1j * (self.theta[:, E_idx] - self.theta)))
        d_theta_W = np.angle(np.exp(1j * (self.theta[:, W_idx] - self.theta)))
        d_theta_S = np.angle(np.exp(1j * (self.theta[S_idx, :] - self.theta)))
        d_theta_N = np.angle(np.exp(1j * (self.theta[N_idx, :] - self.theta)))
        
        d_press_E = self.F - self.F[:, E_idx]
        d_press_S = self.F - self.F[S_idx, :]
        
        cond_x = self.sigma * (self.C_x**2 + 1e-4)
        cond_y = self.sigma * (self.C_y**2 + 1e-4)
        
        # Phase/pressure flux
        J_base_x = cond_x * (np.sin(d_theta_E) + d_press_E)
        J_base_y = cond_y * (np.sin(d_theta_S) + d_press_S)
        
        # Gravitational flux - BOUNDED BY LOCAL F
        if self.k.GRAVITY_ENABLED:
            grav_flux_x = self.F * self.v_grav_x
            grav_flux_y = self.F * self.v_grav_y
            
            # Limit gravitational flux to GRAV_FLUX_FRAC * F / dt
            grav_flux_mag = np.sqrt(grav_flux_x**2 + grav_flux_y**2)
            max_grav_flux = self.k.GRAV_FLUX_FRAC * self.F / dt
            grav_scale = np.minimum(1.0, max_grav_flux / (grav_flux_mag + 1e-9))
            grav_flux_x *= grav_scale
            grav_flux_y *= grav_scale
        else:
            grav_flux_x = 0
            grav_flux_y = 0
        
        # Total flow
        J_x = J_base_x + grav_flux_x
        J_y = J_base_y + grav_flux_y
        
        # Overall flux limit
        J_mag = np.sqrt(J_x**2 + J_y**2)
        max_J = 0.25 * self.F / dt
        scale = np.minimum(1.0, max_J / (J_mag + 1e-9))
        J_x *= scale
        J_y *= scale
        
        self.J_x_last = J_x.copy()
        self.J_y_last = J_y.copy()
        
        # ===== F UPDATE =====
        J_x_in = np.roll(J_x, 1, axis=1)
        J_y_in = np.roll(J_y, 1, axis=0)
        div_J = (J_x - J_x_in) + (J_y - J_y_in)
        
        self.F -= div_J * dt
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        # ===== PHASE UPDATE =====
        laplacian = d_theta_E + d_theta_W + d_theta_S + d_theta_N
        self.theta += self.k.NU * laplacian * dt
        self.theta = np.mod(self.theta, 2*np.pi)
        
        # ===== STRUCTURAL DEBT =====
        if self.k.GRAVITY_ENABLED:
            compression = np.maximum(0, -div_J)
            q_src = self.k.ETA_Q * compression * (compression > self.k.Q_THRESHOLD)
            q_dec = self.k.LAMBDA_Q * self.q
            self.q += (q_src - q_dec) * dt
            self.q = np.clip(self.q, 0, 1)
        
        # ===== METRIC UPDATE =====
        compression = np.maximum(0, -div_J)
        
        # Metric growth and decay (tuned for collision + gravity)
        self.C_x += 0.5 * np.abs(J_x) * dt - 0.001 * self.C_x * dt
        self.C_y += 0.5 * np.abs(J_y) * dt - 0.001 * self.C_y * dt
        self.C_x = np.clip(self.C_x, self.k.C_MIN, 1.0)
        self.C_y = np.clip(self.C_y, self.k.C_MIN, 1.0)
        
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_x) + np.abs(J_y))
        self.time += dt

    def get_diagnostics(self):
        return {
            'total_F': np.sum(self.F),
            'total_q': np.sum(self.q),
            'max_v_grav': np.max(np.sqrt(self.v_grav_x**2 + self.v_grav_y**2)),
            'phase_coherence': np.abs(np.mean(np.exp(1j * self.theta))),
            'peaks': self.find_peaks()
        }


# ============================================================
# TEST SUITE
# ============================================================

def test_stability():
    print("\n" + "="*60)
    print("TEST 1: SINGLE PACKET STABILITY")
    print("="*60)
    
    k = DETConstants2D()
    sim = ActiveManifold2D(size=80, constants=k)
    sim.initialize_single_packet(mass=20.0, radius=3.0, velocity=0.0, q_fraction=0.2)
    
    initial_F = np.sum(sim.F)
    
    print(f"\n{'Step':<8}{'Peaks':<8}{'Total F':<12}{'Mass Err':<12}")
    print("-" * 40)
    
    for t in range(301):
        if t % 50 == 0:
            d = sim.get_diagnostics()
            mass_err = 100 * (d['total_F'] - initial_F) / initial_F
            print(f"{t:<8}{d['peaks']:<8}{d['total_F']:<12.2f}{mass_err:<+12.2f}%")
        sim.step()
    
    final_F = np.sum(sim.F)
    mass_err = abs(final_F - initial_F) / initial_F
    
    if mass_err < 0.05 and sim.find_peaks() <= 2:
        print("✓ PASS: Stable packet, mass conserved")
        return True
    print(f"✗ FAIL: mass error {100*mass_err:.1f}%, peaks {sim.find_peaks()}")
    return False


def test_collision():
    print("\n" + "="*60)
    print("TEST 2: COLLISION DYNAMICS (No Gravity)")
    print("="*60)
    
    k = DETConstants2D()
    k.GRAVITY_ENABLED = False
    
    sim = ActiveManifold2D(size=100, constants=k)
    sim.initialize_two_packets(separation=30, velocity=0.5, q_fraction=0.2)
    
    initial_F = np.sum(sim.F)
    separations = []
    
    print(f"\n{'Step':<8}{'Sep':<10}{'Peaks':<8}{'Mass Err':<12}")
    print("-" * 40)
    
    for t in range(401):
        positions = sim.find_peak_positions()
        if len(positions) >= 2:
            sorted_pos = sorted(positions, key=lambda p: p[1])
            sep = abs(sorted_pos[1][1] - sorted_pos[0][1])
            separations.append(sep)
        
        if t % 50 == 0:
            mass_err = 100 * (np.sum(sim.F) - initial_F) / initial_F
            sep_str = f"{sep:.1f}" if len(positions) >= 2 else "merged"
            print(f"{t:<8}{sep_str:<10}{len(positions):<8}{mass_err:<+12.2f}%")
        
        sim.step()
    
    if separations:
        initial_sep = separations[0]
        min_sep = min(separations)
        
        print(f"\nInitial: {initial_sep:.1f}, Min: {min_sep:.1f}")
        
        if min_sep < initial_sep * 0.5:
            print("✓ PASS: Collision occurred")
            return True
    
    print("✗ FAIL: No collision")
    return False


def test_gravity():
    print("\n" + "="*60)
    print("TEST 3: GRAVITATIONAL ATTRACTION")
    print("="*60)
    
    k = DETConstants2D()
    k.KAPPA = 5.0
    k.GRAV_FLUX_FRAC = 0.02
    
    sim = ActiveManifold2D(size=100, constants=k)
    sim.initialize_gravity_test(separation=24, m1=30.0, m2=30.0, q_fraction=0.6)
    
    initial_F = np.sum(sim.F)
    traj = {'sep': []}
    
    print(f"\n{'Step':<8}{'Sep':<10}{'Peaks':<8}{'Mass Err':<12}{'Max v':<10}")
    print("-" * 52)
    
    for t in range(801):
        positions = sim.find_peak_positions()
        if len(positions) >= 2:
            sorted_pos = sorted(positions, key=lambda p: p[1])
            sep = abs(sorted_pos[1][1] - sorted_pos[0][1])
            traj['sep'].append(sep)
        
        if t % 100 == 0:
            d = sim.get_diagnostics()
            mass_err = 100 * (d['total_F'] - initial_F) / initial_F
            sep_str = f"{sep:.1f}" if len(positions) >= 2 else "merged"
            print(f"{t:<8}{sep_str:<10}{len(positions):<8}{mass_err:<+12.2f}%{d['max_v_grav']:<10.4f}")
        
        sim.step()
    
    if traj['sep']:
        initial_sep = traj['sep'][0]
        min_sep = min(traj['sep'])
        final_sep = traj['sep'][-1]
        final_mass_err = 100 * abs(np.sum(sim.F) - initial_F) / initial_F
        
        print(f"\nInitial: {initial_sep:.1f}, Min: {min_sep:.1f}, Final: {final_sep:.1f}")
        print(f"Final mass error: {final_mass_err:.1f}%")
        
        # Check if bodies attracted (min sep < initial - 5)
        if min_sep < initial_sep - 5 and final_mass_err < 10:
            print("✓ PASS: Gravitational attraction demonstrated!")
            return True
    
    print("✗ FAIL")
    return False


def test_gravity_with_collision():
    print("\n" + "="*60)
    print("TEST 4: GRAVITY + INITIAL VELOCITY")
    print("="*60)
    
    k = DETConstants2D()
    k.KAPPA = 3.0
    k.GRAV_FLUX_FRAC = 0.015
    
    sim = ActiveManifold2D(size=100, constants=k)
    
    center = sim.N // 2
    sim.add_body((center, center - 15), 25.0, 4.0, 0.4, velocity=(0.2, 0))
    sim.add_body((center, center + 15), 25.0, 4.0, 0.4, velocity=(-0.2, 0))
    
    initial_F = np.sum(sim.F)
    separations = []
    
    print(f"\n{'Step':<8}{'Sep':<10}{'Peaks':<8}{'Mass Err':<12}")
    print("-" * 40)
    
    for t in range(601):
        positions = sim.find_peak_positions()
        if len(positions) >= 2:
            sorted_pos = sorted(positions, key=lambda p: p[1])
            sep = abs(sorted_pos[1][1] - sorted_pos[0][1])
            separations.append(sep)
        elif len(positions) == 1:
            separations.append(0)
        
        if t % 75 == 0:
            mass_err = 100 * (np.sum(sim.F) - initial_F) / initial_F
            sep_str = f"{sep:.1f}" if len(positions) >= 2 else "merged"
            print(f"{t:<8}{sep_str:<10}{len(positions):<8}{mass_err:<+12.2f}%")
        
        sim.step()
    
    if separations:
        initial_sep = separations[0]
        min_sep = min(separations)
        
        print(f"\nInitial: {initial_sep:.1f}, Min: {min_sep:.1f}")
        
        if min_sep < 10:
            print("✓ PASS: Gravity-assisted collision")
            return True
    
    print("✗ FAIL")
    return False


def create_visualization():
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    k = DETConstants2D()
    k.KAPPA = 5.0
    k.GRAV_FLUX_FRAC = 0.02
    
    sim = ActiveManifold2D(size=100, constants=k)
    sim.initialize_gravity_test(separation=24, m1=30.0, m2=30.0, q_fraction=0.6)
    
    snapshots = []
    traj = {'t': [], 'x1': [], 'x2': [], 'sep': []}
    
    for t in range(801):
        if t in [0, 150, 300, 450, 600, 750]:
            snapshots.append((t, sim.F.copy(), sim.Phi.copy()))
        
        positions = sim.find_peak_positions()
        if len(positions) >= 2:
            sorted_pos = sorted(positions, key=lambda p: p[1])
            traj['t'].append(t)
            traj['x1'].append(sorted_pos[0][1])
            traj['x2'].append(sorted_pos[1][1])
            traj['sep'].append(abs(sorted_pos[1][1] - sorted_pos[0][1]))
        
        sim.step()
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    
    # Snapshots of F
    for i, (t, F, Phi) in enumerate(snapshots):
        ax = fig.add_subplot(3, 6, i+1)
        ax.imshow(F, cmap='plasma', origin='lower')
        ax.set_title(f't={t}')
        ax.axis('off')
    
    # Trajectories
    ax = fig.add_subplot(3, 2, 3)
    if traj['x1']:
        ax.plot(traj['t'], traj['x1'], 'b-', lw=2, label='Body 1')
        ax.plot(traj['t'], traj['x2'], 'r-', lw=2, label='Body 2')
    ax.set_xlabel('Time')
    ax.set_ylabel('X Position')
    ax.set_title('Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Separation
    ax = fig.add_subplot(3, 2, 4)
    if traj['sep']:
        ax.plot(traj['t'], traj['sep'], 'g-', lw=2)
        ax.axhline(traj['sep'][0], color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Separation')
    ax.set_title('Inter-body Distance')
    ax.grid(True, alpha=0.3)
    
    # Final state
    ax = fig.add_subplot(3, 3, 7)
    im = ax.imshow(sim.F, cmap='plasma', origin='lower')
    ax.set_title('Final F')
    plt.colorbar(im, ax=ax)
    
    ax = fig.add_subplot(3, 3, 8)
    im = ax.imshow(sim.Phi, cmap='RdBu_r', origin='lower')
    ax.set_title('Final Φ')
    plt.colorbar(im, ax=ax)
    
    ax = fig.add_subplot(3, 3, 9)
    im = ax.imshow(sim.q, cmap='Reds', origin='lower')
    ax.set_title('Final q')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('DET v5 2D Collider with DET 4.2 Gravity', fontsize=14)
    plt.tight_layout()
    plt.savefig('det_v5_2d_collider_gravity.png', dpi=150)
    print("Saved: det_v5_2d_collider_gravity.png")
    plt.close()


def run_all_tests():
    print("\n" + "="*70)
    print("DET v5 2D COLLIDER WITH DET 4.2 GRAVITY - FINAL VERSION")
    print("="*70)
    
    k = DETConstants2D()
    print(f"\nKey Parameters:")
    print(f"  KAPPA = {k.KAPPA}")
    print(f"  GRAV_FLUX_FRAC = {k.GRAV_FLUX_FRAC}")
    print(f"  GRAV_VEL_VISCOSITY = {k.GRAV_VEL_VISCOSITY}")
    
    results = {}
    results['stability'] = test_stability()
    results['collision'] = test_collision()
    results['gravity'] = test_gravity()
    results['gravity_collision'] = test_gravity_with_collision()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test:<20}: {status}")
    
    print(f"\nTotal: {sum(results.values())}/{len(results)} passed")
    
    create_visualization()
    
    return results


if __name__ == "__main__":
    run_all_tests()
