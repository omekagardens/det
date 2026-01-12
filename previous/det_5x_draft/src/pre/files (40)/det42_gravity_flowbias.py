import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, maximum_filter, minimum_filter
from scipy.fft import fft, ifft, fftfreq
import time as clock_time

"""
DET 4.2 GRAVITY - SIMPLE FLOW BIAS
==================================

Instead of momentum field (which causes instability),
add gravity directly as a bias to the flow equation:

J = cond * (phase_drive + pressure) + KAPPA * F * g

This is simpler and more stable.
"""


class DETGravity1D:
    def __init__(self, N, kappa=0.1):
        self.N = N
        self.kappa = kappa
        k = fftfreq(N, d=1.0) * 2 * np.pi
        self.k_squared = k**2
        self.k_squared[0] = 1.0
    
    def compute_gravity(self, F):
        rho = F / (np.sum(F) + 1e-9) * self.N
        rho_hat = fft(rho)
        phi_hat = +self.kappa * rho_hat / self.k_squared  # CORRECT sign
        phi_hat[0] = 0.0
        Phi = np.real(ifft(phi_hat))
        g = -0.5 * (np.roll(Phi, -1) - np.roll(Phi, 1))
        return Phi, g


class DETConstants1D:
    def __init__(self):
        self.NU = 0.05
        self.ALPHA = 0.3
        self.LAMBDA = 0.0001
        self.K_FUSION = 50.0
        self.F_VAC = 0.001
        self.C_MIN = 0.05
        self.KAPPA = 0.05  # Gravity strength (small!)
        self.DT = 0.05


class ActiveManifold1D:
    def __init__(self, size=200, constants=DETConstants1D()):
        self.N = size
        self.k = constants
        self.F = np.ones(self.N) * self.k.F_VAC
        self.theta = np.zeros(self.N)
        self.sigma = np.ones(self.N)
        self.C = np.ones(self.N) * self.k.C_MIN
        self.gravity = DETGravity1D(self.N, self.k.KAPPA)
        self.Phi = np.zeros(self.N)
        self.g_field = np.zeros(self.N)
        self.time = 0.0
        self.history = {'t': [], 'x1': [], 'x2': [], 'sep': []}

    def add_body(self, center, mass, radius):
        x = np.arange(self.N)
        dx = x - center
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        self.F += mass * np.exp(-0.5 * (dx/radius)**2)

    def initialize_two_bodies(self, separation=30, m1=30.0, m2=30.0):
        center = self.N // 2
        self.add_body(center - separation//2, m1, 4.0)
        self.add_body(center + separation//2, m2, 4.0)

    def find_peak_positions(self, threshold=0.5):
        neighborhood = 5
        data_max = maximum_filter(self.F, neighborhood)
        maxima = (self.F == data_max)
        data_min = minimum_filter(self.F, neighborhood)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = label(maxima)
        positions = []
        for i in range(1, num_objects + 1):
            mask = labeled == i
            total = np.sum(self.F[mask])
            if total > 0:
                x_coords = np.where(mask)[0]
                positions.append(np.sum(self.F[mask] * x_coords) / total)
        return sorted(positions)

    def step(self):
        dt = self.k.DT
        
        # GRAVITY
        self.Phi, self.g_field = self.gravity.compute_gravity(self.F)
        
        # FLOW with gravity bias
        E_idx = np.roll(np.arange(self.N), -1)
        W_idx = np.roll(np.arange(self.N), 1)
        d_theta_E = np.angle(np.exp(1j * (self.theta[E_idx] - self.theta)))
        d_theta_W = np.angle(np.exp(1j * (self.theta[W_idx] - self.theta)))
        d_press_E = self.F - self.F[E_idx]
        
        cond = self.sigma * (self.C**2 + 1e-4)
        
        # Standard flow + gravity bias
        J = cond * (np.sin(d_theta_E) + d_press_E) + self.F * self.g_field
        
        # Flux limiter
        J_mag = np.abs(J)
        max_J = 0.3 * self.F / dt
        scale = np.minimum(1.0, max_J / (J_mag + 1e-9))
        J *= scale
        
        div_J = J - np.roll(J, 1)
        self.F -= div_J * dt
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        # Phase
        self.theta += self.k.NU * (d_theta_E + d_theta_W) * dt
        self.theta = np.mod(self.theta, 2*np.pi)
        
        # Metric
        compression = np.maximum(0, -div_J)
        alpha_eff = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        self.C += (alpha_eff * np.abs(J) - self.k.LAMBDA * self.C) * dt
        self.C = np.clip(self.C, self.k.C_MIN, 1.0)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J))
        self.time += dt

    def record_positions(self):
        positions = self.find_peak_positions()
        if len(positions) >= 2:
            self.history['t'].append(self.time)
            self.history['x1'].append(positions[0])
            self.history['x2'].append(positions[-1])
            self.history['sep'].append(abs(positions[-1] - positions[0]))


def run_demo():
    print("="*70)
    print("DET 4.2 GRAVITY - SIMPLE FLOW BIAS")
    print("="*70)
    
    k = DETConstants1D()
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_two_bodies(separation=40, m1=30.0, m2=30.0)
    
    # Verify field
    _, g = sim.gravity.compute_gravity(sim.F)
    print(f"\nField check: g[90] = {g[90]:.4f} (should be > 0)")
    
    snapshots = {}
    print(f"\n{'Step':<8}{'Sep':<10}{'Peaks':<8}")
    print("-" * 30)
    
    for t in range(2001):
        if t in [0, 500, 1000, 1500, 2000]:
            snapshots[t] = sim.F.copy()
        
        if t % 200 == 0:
            sim.record_positions()
            positions = sim.find_peak_positions()
            sep = sim.history['sep'][-1] if sim.history['sep'] else 40
            print(f"{t:<8}{sep:<10.1f}{len(positions):<8}")
        
        sim.step()
    
    sim.record_positions()
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    if sim.history['sep']:
        axes[0,0].plot(sim.history['t'], sim.history['x1'], 'b-', lw=2, label='Body 1')
        axes[0,0].plot(sim.history['t'], sim.history['x2'], 'r-', lw=2, label='Body 2')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('X')
        axes[0,0].set_title('Trajectories')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(sim.history['t'], sim.history['sep'], 'g-', lw=2)
        if sim.history['sep']:
            axes[0,1].axhline(sim.history['sep'][0], color='k', ls='--', alpha=0.5)
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Separation')
        axes[0,1].set_title('Inter-body Distance')
        axes[0,1].grid(True, alpha=0.3)
    
    axes[0,2].plot(sim.Phi, 'g-', lw=2)
    axes[0,2].set_title('Final Potential Φ')
    
    snap_times = sorted(snapshots.keys())[:3]
    F_max = max(np.max(snapshots[t]) for t in snap_times)
    for i, t in enumerate(snap_times):
        axes[1,i].plot(snapshots[t], 'b-', lw=2)
        axes[1,i].set_ylim(0, F_max * 1.1)
        axes[1,i].set_title(f't={t}')
    
    plt.suptitle('DET 4.2 Gravity: 1D Flow Bias Method', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/claude/det42_gravity_flowbias.png', dpi=150)
    plt.savefig('/mnt/user-data/outputs/det42_gravity_flowbias.png', dpi=150)
    print("\nSaved: det42_gravity_flowbias.png")
    
    if sim.history['sep']:
        init_sep = sim.history['sep'][0]
        final_sep = sim.history['sep'][-1]
        print(f"\n{'='*50}")
        print(f"Initial separation: {init_sep:.1f}")
        print(f"Final separation: {final_sep:.1f}")
        print(f"Change: {final_sep - init_sep:+.1f}")
        
        if final_sep < init_sep - 2:
            print("\n✓ GRAVITATIONAL ATTRACTION WORKING!")
            return True
    return False


def kappa_sweep():
    """Find optimal KAPPA."""
    print("\n" + "="*70)
    print("KAPPA sweep (flow bias method)")
    print("="*70)
    print(f"{'KAPPA':<10}{'Final sep':<12}{'Peaks':<8}{'Δsep':<10}")
    print("-" * 45)
    
    best = {'kappa': 0, 'delta': 999}
    
    for kappa in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        k = DETConstants1D()
        k.KAPPA = kappa
        
        sim = ActiveManifold1D(size=200, constants=k)
        sim.initialize_two_bodies(separation=40, m1=30.0, m2=30.0)
        
        for _ in range(2000):
            sim.step()
        
        positions = sim.find_peak_positions()
        final_sep = positions[-1] - positions[0] if len(positions) >= 2 else 0
        delta = final_sep - 40.0
        
        print(f"{kappa:<10.2f}{final_sep:<12.1f}{len(positions):<8}{delta:+.1f}")
        
        if abs(delta) < abs(best['delta']) or (delta < 0 and best['delta'] > 0):
            if len(positions) <= 3:  # Not too fragmented
                best = {'kappa': kappa, 'delta': delta}
    
    print(f"\nBest: KAPPA={best['kappa']} (Δsep={best['delta']:+.1f})")
    return best['kappa']


if __name__ == "__main__":
    t0 = clock_time.time()
    
    # Find best KAPPA
    best_kappa = kappa_sweep()
    
    # Run demo with best
    print(f"\nRunning with KAPPA={best_kappa}...")
    
    k = DETConstants1D()
    k.KAPPA = best_kappa
    
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_two_bodies(separation=40, m1=30.0, m2=30.0)
    
    print(f"\n{'Step':<8}{'Sep':<10}{'Peaks':<8}")
    print("-" * 30)
    for t in range(2001):
        if t % 200 == 0:
            sim.record_positions()
            positions = sim.find_peak_positions()
            sep = sim.history['sep'][-1] if sim.history['sep'] else 40
            print(f"{t:<8}{sep:<10.1f}{len(positions):<8}")
        sim.step()
    
    if sim.history['sep']:
        print(f"\nΔsep = {sim.history['sep'][-1] - sim.history['sep'][0]:+.1f}")
    
    elapsed = clock_time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
