import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq

"""
DET 4.2 GRAVITY - WORKING VERSION
=================================
KAPPA=5.0 showed attraction in the sweep!
"""

class DETGravity2D:
    def __init__(self, N, kappa=5.0, alpha_baseline=0.1):
        self.N = N
        self.kappa = kappa
        self.alpha_baseline = alpha_baseline
        
        k = fftfreq(N, d=1.0) * 2 * np.pi
        kx, ky = np.meshgrid(k, k)
        self.k_squared = kx**2 + ky**2
        self.k_squared[0, 0] = 1.0
    
    def compute_baseline(self, q):
        rhs = -self.alpha_baseline * q
        rhs_hat = fft2(rhs)
        denom = -self.k_squared - self.alpha_baseline
        denom[0, 0] = -self.alpha_baseline
        return np.real(ifft2(rhs_hat / denom))
    
    def compute_potential(self, rho):
        rho_hat = fft2(rho)
        phi_hat = -self.kappa * rho_hat / self.k_squared
        phi_hat[0, 0] = 0.0
        return np.real(ifft2(phi_hat))
    
    def compute_gravity(self, q):
        b = self.compute_baseline(q)
        rho = q - b
        Phi = self.compute_potential(rho)
        g_x = -0.5 * (np.roll(Phi, -1, axis=1) - np.roll(Phi, 1, axis=1))
        g_y = -0.5 * (np.roll(Phi, -1, axis=0) - np.roll(Phi, 1, axis=0))
        return Phi, rho, g_x, g_y


class DETConstants2D:
    def __init__(self):
        self.NU = 0.05
        self.ALPHA = 0.6
        self.LAMBDA = 0.00001
        self.K_FUSION = 120.0
        self.F_VAC = 0.001
        self.C_MIN = 0.05
        self.ETA_Q = 0.02
        self.LAMBDA_Q = 0.001
        self.Q_THRESHOLD = 0.01
        self.KAPPA = 5.0  # Working value!
        self.ALPHA_BASELINE = 0.1
        self.MOM_VISCOSITY = 0.0005
        self.MOM_DIFFUSION = 0.005
        self.DT = 0.05


class ActiveManifold2D:
    def __init__(self, size=100, constants=DETConstants2D()):
        self.N = size
        self.k = constants
        
        self.F = np.ones((self.N, self.N)) * self.k.F_VAC
        self.theta = np.zeros((self.N, self.N))
        self.q = np.zeros((self.N, self.N))
        self.sigma = np.ones((self.N, self.N))
        self.C_x = np.ones((self.N, self.N)) * self.k.C_MIN
        self.C_y = np.ones((self.N, self.N)) * self.k.C_MIN
        self.p_x = np.zeros((self.N, self.N))
        self.p_y = np.zeros((self.N, self.N))
        
        self.gravity = DETGravity2D(self.N, self.k.KAPPA, self.k.ALPHA_BASELINE)
        self.Phi = np.zeros((self.N, self.N))
        self.g_x = np.zeros((self.N, self.N))
        self.g_y = np.zeros((self.N, self.N))
        self.time = 0.0

    def add_body(self, center, mass, radius, q_fraction=0.5):
        y, x = np.mgrid[0:self.N, 0:self.N]
        cy, cx = center
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        body = mass * np.exp(-0.5 * (r/radius)**2)
        self.F += body
        self.q += q_fraction * (body / (mass + 1e-9))

    def initialize_two_bodies(self, separation=30, m1=30.0, m2=30.0, q_frac=0.5):
        center = self.N // 2
        self.add_body((center, center - separation//2), m1, 4.0, q_frac)
        self.add_body((center, center + separation//2), m2, 4.0, q_frac)

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
        
        self.Phi, _, self.g_x, self.g_y = self.gravity.compute_gravity(self.q)
        
        self.p_x += self.F * self.g_x * dt
        self.p_y += self.F * self.g_y * dt
        
        lap_px = (np.roll(self.p_x, 1, axis=1) + np.roll(self.p_x, -1, axis=1) +
                  np.roll(self.p_x, 1, axis=0) + np.roll(self.p_x, -1, axis=0) - 4*self.p_x)
        lap_py = (np.roll(self.p_y, 1, axis=1) + np.roll(self.p_y, -1, axis=1) +
                  np.roll(self.p_y, 1, axis=0) + np.roll(self.p_y, -1, axis=0) - 4*self.p_y)
        self.p_x += self.k.MOM_DIFFUSION * lap_px * dt
        self.p_y += self.k.MOM_DIFFUSION * lap_py * dt
        
        self.p_x *= (1 - self.k.MOM_VISCOSITY * dt)
        self.p_y *= (1 - self.k.MOM_VISCOSITY * dt)
        
        max_p = 10.0
        self.p_x = np.clip(self.p_x, -max_p, max_p)
        self.p_y = np.clip(self.p_y, -max_p, max_p)
        
        E_idx = np.roll(np.arange(self.N), -1)
        W_idx = np.roll(np.arange(self.N), 1)
        S_idx = np.roll(np.arange(self.N), -1)
        N_idx = np.roll(np.arange(self.N), 1)
        
        d_theta_E = np.angle(np.exp(1j * (self.theta[:, E_idx] - self.theta)))
        d_theta_W = np.angle(np.exp(1j * (self.theta[:, W_idx] - self.theta)))
        d_theta_S = np.angle(np.exp(1j * (self.theta[S_idx, :] - self.theta)))
        d_theta_N = np.angle(np.exp(1j * (self.theta[N_idx, :] - self.theta)))
        
        d_press_E = self.F - self.F[:, E_idx]
        d_press_S = self.F - self.F[S_idx, :]
        
        cond_x = self.sigma * (self.C_x**2 + 1e-4)
        cond_y = self.sigma * (self.C_y**2 + 1e-4)
        
        J_x = cond_x * (np.sin(d_theta_E) + d_press_E) + self.p_x
        J_y = cond_y * (np.sin(d_theta_S) + d_press_S) + self.p_y
        
        J_mag = np.sqrt(J_x**2 + J_y**2)
        max_J = 0.25 * self.F / dt
        scale = np.minimum(1.0, max_J / (J_mag + 1e-9))
        J_x *= scale
        J_y *= scale
        
        J_x_in = np.roll(J_x, 1, axis=1)
        J_y_in = np.roll(J_y, 1, axis=0)
        div_J = (J_x - J_x_in) + (J_y - J_y_in)
        
        self.F -= div_J * dt
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        v_x = self.p_x / (self.F + 1e-9)
        v_y = self.p_y / (self.F + 1e-9)
        v_mag = np.sqrt(v_x**2 + v_y**2)
        v_x = np.where(v_mag > 1, v_x / v_mag, v_x)
        v_y = np.where(v_mag > 1, v_y / v_mag, v_y)
        
        dp_x = np.where(v_x > 0, self.p_x - np.roll(self.p_x, 1, axis=1),
                        np.roll(self.p_x, -1, axis=1) - self.p_x)
        dp_y = np.where(v_y > 0, self.p_y - np.roll(self.p_y, 1, axis=0),
                        np.roll(self.p_y, -1, axis=0) - self.p_y)
        self.p_x -= 0.1 * v_x * dp_x * dt
        self.p_y -= 0.1 * v_y * dp_y * dt
        
        laplacian = d_theta_E + d_theta_W + d_theta_S + d_theta_N
        self.theta += self.k.NU * laplacian * dt
        self.theta = np.mod(self.theta, 2*np.pi)
        
        compression = np.maximum(0, -div_J)
        q_src = self.k.ETA_Q * compression * (compression > self.k.Q_THRESHOLD)
        q_dec = self.k.LAMBDA_Q * self.q
        self.q += (q_src - q_dec) * dt
        self.q = np.clip(self.q, 0, 1)
        
        alpha_eff = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        self.C_x += (alpha_eff * np.abs(J_x) - self.k.LAMBDA * self.C_x) * dt
        self.C_y += (alpha_eff * np.abs(J_y) - self.k.LAMBDA * self.C_y) * dt
        self.C_x = np.clip(self.C_x, self.k.C_MIN, 1.0)
        self.C_y = np.clip(self.C_y, self.k.C_MIN, 1.0)
        
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_x) + np.abs(J_y))
        self.time += dt


def run_gravity_demo():
    print("\n" + "="*70)
    print("DET 4.2 GRAVITY DEMONSTRATION (KAPPA=5.0)")
    print("="*70)
    
    k = DETConstants2D()
    sim = ActiveManifold2D(size=100, constants=k)
    sim.initialize_two_bodies(separation=24, m1=30.0, m2=30.0, q_frac=0.6)
    
    traj = {'t': [], 'x1': [], 'x2': [], 'sep': []}
    snapshots = {}
    
    print(f"\n{'Step':<8}{'Sep':<10}{'Peaks':<8}{'Max|g|':<12}")
    print("-" * 40)
    
    for t in range(801):
        if t in [0, 200, 400, 600, 800]:
            snapshots[t] = sim.F.copy()
        
        positions = sim.find_peak_positions()
        
        if t % 50 == 0:
            if len(positions) >= 2:
                sorted_pos = sorted(positions, key=lambda p: p[1])
                sep = abs(sorted_pos[1][1] - sorted_pos[0][1])
                traj['t'].append(t)
                traj['x1'].append(sorted_pos[0][1])
                traj['x2'].append(sorted_pos[1][1])
                traj['sep'].append(sep)
            else:
                sep = 0 if len(positions) <= 1 else traj['sep'][-1]
            
            g_mag = np.sqrt(sim.g_x**2 + sim.g_y**2).max()
            print(f"{t:<8}{sep:<10.1f}{len(positions):<8}{g_mag:<12.5f}")
        
        sim.step()
    
    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Trajectories
    if traj['sep']:
        axes[0,0].plot(traj['t'], traj['x1'], 'b-', lw=2, label='Body 1')
        axes[0,0].plot(traj['t'], traj['x2'], 'r-', lw=2, label='Body 2')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('X')
        axes[0,0].set_title('Trajectories')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(traj['t'], traj['sep'], 'g-', lw=2)
        axes[0,1].axhline(traj['sep'][0], color='k', ls='--', alpha=0.5)
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Separation')
        axes[0,1].set_title('Inter-body Distance')
        axes[0,1].grid(True, alpha=0.3)
    
    # Gravity field
    skip = 5
    Y, X = np.mgrid[0:sim.N:skip, 0:sim.N:skip]
    axes[0,2].quiver(X, Y, sim.g_x[::skip, ::skip], sim.g_y[::skip, ::skip],
                     color='white', alpha=0.8, scale=2.0, scale_units='xy')
    axes[0,2].imshow(sim.Phi, cmap='viridis', origin='lower', alpha=0.6)
    axes[0,2].set_title('g = -∇Φ')
    
    im = axes[0,3].imshow(sim.q, cmap='Reds', origin='lower')
    axes[0,3].set_title('q (Structural Debt)')
    plt.colorbar(im, ax=axes[0,3])
    
    # Snapshots
    snap_times = sorted(snapshots.keys())
    F_max = max(np.max(snapshots[t]) for t in snap_times)
    for i, t in enumerate(snap_times[:4]):
        im = axes[1,i].imshow(snapshots[t], cmap='plasma', origin='lower', vmin=0, vmax=F_max)
        axes[1,i].set_title(f't={t}')
        axes[1,i].axis('off')
    
    plt.suptitle('DET 4.2 Gravity: Two-Body Attraction', fontsize=14)
    plt.tight_layout()
    plt.savefig('det42_gravity_demo.png', dpi=150)
    print("\nSaved: det42_gravity_demo.png")
    plt.close()
    
    # Summary
    if traj['sep']:
        init_sep = traj['sep'][0]
        final_sep = traj['sep'][-1]
        print(f"\n{'='*50}")
        print(f"Initial separation: {init_sep:.1f}")
        print(f"Final separation: {final_sep:.1f}")
        print(f"Change: {final_sep - init_sep:+.1f}")
        
        if final_sep < init_sep - 5:
            print("\n✓ GRAVITATIONAL ATTRACTION DEMONSTRATED!")
            return True
    return False


if __name__ == "__main__":
    success = run_gravity_demo()
    if success:
        print("\n" + "="*70)
        print("SUCCESS: DET 4.2 gravity sector is working!")
        print("="*70)
