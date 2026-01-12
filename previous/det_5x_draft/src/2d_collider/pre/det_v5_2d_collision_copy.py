"""
DET v5 2D Collider - Optimized Final Run
========================================

Tuning for best mass conservation while maintaining collision.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from dataclasses import dataclass
import time


@dataclass
class DETParams:
    N: int = 100
    DT: float = 0.015          # Smaller timestep for stability
    F_VAC: float = 0.01
    R: int = 5
    nu: float = 0.1
    C_init: float = 0.3
    momentum_enabled: bool = True
    alpha_pi: float = 0.08     # Slightly lower accumulation
    lambda_pi: float = 0.015   # Slightly more friction
    mu_pi: float = 0.25        # Slightly lower coupling
    pi_max: float = 2.5        # Lower clip
    q_enabled: bool = True
    alpha_q: float = 0.015
    a_coupling: float = 30.0
    a_rate: float = 0.2
    floor_enabled: bool = True
    eta_floor: float = 0.12    # Stronger floor
    F_core: float = 4.0        # Earlier activation
    floor_power: float = 2.0
    outflow_limit: float = 0.20  # Stricter limiter


def periodic_local_sum_2d(x, radius):
    result = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
    return result


class DETCollider2D:
    def __init__(self, params=None):
        self.p = params or DETParams()
        N = self.p.N
        self.F = np.ones((N, N)) * self.p.F_VAC
        self.q = np.zeros((N, N))
        self.theta = np.zeros((N, N))
        self.a = np.ones((N, N))
        self.pi_E = np.zeros((N, N))
        self.pi_S = np.zeros((N, N))
        self.C_E = np.ones((N, N)) * self.p.C_init
        self.C_S = np.ones((N, N)) * self.p.C_init
        self.sigma = np.ones((N, N))
        self.time = 0.0
        self.step_count = 0
    
    def add_packet(self, center, mass=6.0, width=5.0, momentum=(0, 0)):
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        cy, cx = center
        r2 = (x - cx)**2 + (y - cy)**2
        envelope = np.exp(-0.5 * r2 / width**2)
        self.F += mass * envelope
        self.C_E = np.clip(self.C_E + 0.7 * envelope, self.p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.7 * envelope, self.p.C_init, 1.0)
        py, px = momentum
        if px != 0 or py != 0:
            mom_env = np.exp(-0.5 * r2 / (width*3)**2)
            self.pi_E += px * mom_env
            self.pi_S += py * mom_env
        self._clip()
    
    def _clip(self):
        self.F = np.clip(self.F, self.p.F_VAC, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_E = np.clip(self.pi_E, -self.p.pi_max, self.p.pi_max)
        self.pi_S = np.clip(self.pi_S, -self.p.pi_max, self.p.pi_max)
    
    def step(self):
        p = self.p
        N = p.N
        dt = p.DT
        
        E = lambda x: np.roll(x, -1, axis=1)
        W = lambda x: np.roll(x, 1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)
        Nb = lambda x: np.roll(x, 1, axis=0)
        
        P = self.a / (1.0 + self.F)
        Delta_tau = P * dt
        Delta_tau_E = 0.5 * (Delta_tau + E(Delta_tau))
        Delta_tau_S = 0.5 * (Delta_tau + S(Delta_tau))
        
        F_local = periodic_local_sum_2d(self.F, p.R) + 1e-9
        amp = np.sqrt(np.clip(self.F / F_local, 0, 1))
        psi = amp * np.exp(1j * self.theta)
        
        quantum_E = np.imag(np.conj(psi) * E(psi))
        quantum_W = np.imag(np.conj(psi) * W(psi))
        quantum_S = np.imag(np.conj(psi) * S(psi))
        quantum_N = np.imag(np.conj(psi) * Nb(psi))
        
        classical_E = self.F - E(self.F)
        classical_W = self.F - W(self.F)
        classical_S = self.F - S(self.F)
        classical_N = self.F - Nb(self.F)
        
        sqrt_C_E = np.sqrt(self.C_E)
        sqrt_C_S = np.sqrt(self.C_S)
        sqrt_C_W = np.sqrt(W(self.C_E))
        sqrt_C_N = np.sqrt(Nb(self.C_S))
        
        drive_E = sqrt_C_E * quantum_E + (1 - sqrt_C_E) * classical_E
        drive_W = sqrt_C_W * quantum_W + (1 - sqrt_C_W) * classical_W
        drive_S = sqrt_C_S * quantum_S + (1 - sqrt_C_S) * classical_S
        drive_N = sqrt_C_N * quantum_N + (1 - sqrt_C_N) * classical_N
        
        cond_E = self.sigma * (self.C_E + 1e-4)
        cond_W = self.sigma * (W(self.C_E) + 1e-4)
        cond_S = self.sigma * (self.C_S + 1e-4)
        cond_N = self.sigma * (Nb(self.C_S) + 1e-4)
        
        J_diff_E = cond_E * drive_E
        J_diff_W = cond_W * drive_W
        J_diff_S = cond_S * drive_S
        J_diff_N = cond_N * drive_N
        
        if p.momentum_enabled:
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_W = 0.5 * (self.F + W(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            F_avg_N = 0.5 * (self.F + Nb(self.F))
            J_mom_E = p.mu_pi * self.sigma * self.pi_E * F_avg_E
            J_mom_W = -p.mu_pi * self.sigma * W(self.pi_E) * F_avg_W
            J_mom_S = p.mu_pi * self.sigma * self.pi_S * F_avg_S
            J_mom_N = -p.mu_pi * self.sigma * Nb(self.pi_S) * F_avg_N
        else:
            J_mom_E = J_mom_W = J_mom_S = J_mom_N = 0
        
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_E = p.eta_floor * self.sigma * (s + E(s)) * classical_E
            J_floor_W = p.eta_floor * self.sigma * (s + W(s)) * classical_W
            J_floor_S = p.eta_floor * self.sigma * (s + S(s)) * classical_S
            J_floor_N = p.eta_floor * self.sigma * (s + Nb(s)) * classical_N
        else:
            J_floor_E = J_floor_W = J_floor_S = J_floor_N = 0
        
        J_E = J_diff_E + J_mom_E + J_floor_E
        J_W = J_diff_W + J_mom_W + J_floor_W
        J_S = J_diff_S + J_mom_S + J_floor_S
        J_N = J_diff_N + J_mom_N + J_floor_N
        
        # Conservative limiter
        total_outflow = (np.maximum(0, J_E) + np.maximum(0, J_W) + 
                         np.maximum(0, J_S) + np.maximum(0, J_N))
        max_total_out = p.outflow_limit * self.F / dt
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))
        
        J_E = np.where(J_E > 0, J_E * scale, J_E)
        J_W = np.where(J_W > 0, J_W * scale, J_W)
        J_S = np.where(J_S > 0, J_S * scale, J_S)
        J_N = np.where(J_N > 0, J_N * scale, J_N)
        J_diff_E_scaled = np.where(J_diff_E > 0, J_diff_E * scale, J_diff_E)
        J_diff_S_scaled = np.where(J_diff_S > 0, J_diff_S * scale, J_diff_S)
        
        inflow = W(J_E) + E(J_W) + Nb(J_S) + S(J_N)
        outflow = J_E + J_W + J_S + J_N
        dF = inflow - outflow
        self.F = np.clip(self.F + dF * dt, p.F_VAC, 1000)
        
        if p.momentum_enabled:
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)
            self.pi_E = decay_E * self.pi_E + p.alpha_pi * J_diff_E_scaled * Delta_tau_E
            self.pi_S = decay_S * self.pi_S + p.alpha_pi * J_diff_S_scaled * Delta_tau_S
        
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        d_E = np.angle(np.exp(1j * (E(self.theta) - self.theta)))
        d_W = np.angle(np.exp(1j * (W(self.theta) - self.theta)))
        d_S = np.angle(np.exp(1j * (S(self.theta) - self.theta)))
        d_N = np.angle(np.exp(1j * (Nb(self.theta) - self.theta)))
        self.theta = np.mod(self.theta + p.nu * (d_E + d_W + d_S + d_N) * dt, 2*np.pi)
        
        self.C_E = np.clip(self.C_E + 0.05*np.abs(J_E)*dt - 0.002*self.C_E*dt, p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.05*np.abs(J_S)*dt - 0.002*self.C_S*dt, p.C_init, 1.0)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_E) + np.abs(J_S))
        
        self._clip()
        self.time += dt
        self.step_count += 1
    
    def separation(self):
        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = ndimage.label(above)
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        coms = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) == 0:
                continue
            weights = self.F[mask]
            total_mass = np.sum(weights)
            if total_mass < 0.1:
                continue
            com_x = np.sum(x[mask] * weights) / total_mass
            com_y = np.sum(y[mask] * weights) / total_mass
            coms.append({'x': com_x, 'y': com_y, 'mass': total_mass})
        coms.sort(key=lambda c: -c['mass'])
        if len(coms) < 2:
            return 0.0, len(coms)
        dx = coms[1]['x'] - coms[0]['x']
        dy = coms[1]['y'] - coms[0]['y']
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        return np.sqrt(dx**2 + dy**2), len(coms)


def run_optimized_test():
    """Run optimized collision test."""
    print("="*70)
    print("DET v5 2D COLLIDER - OPTIMIZED TEST")
    print("="*70)
    
    params = DETParams()
    print(f"\nParameters:")
    print(f"  DT={params.DT}, outflow_limit={params.outflow_limit}")
    print(f"  alpha_pi={params.alpha_pi}, lambda_pi={params.lambda_pi}, mu_pi={params.mu_pi}")
    print(f"  eta_floor={params.eta_floor}, F_core={params.F_core}")
    
    sim = DETCollider2D(params)
    sim.add_packet((50, 30), mass=6.0, width=5.0, momentum=(0, 0.5))
    sim.add_packet((50, 70), mass=6.0, width=5.0, momentum=(0, -0.5))
    
    initial_F = np.sum(sim.F)
    sep0, n0 = sim.separation()
    print(f"\nInitial: sep={sep0:.1f}, blobs={n0}, mass={initial_F:.2f}")
    
    rec = {'t': [], 'sep': [], 'mass_err': [], 'blobs': [], 'q_max': [], 'min_a': []}
    snapshots = []
    
    steps = 12000
    for t in range(steps):
        sep, num = sim.separation()
        mass_err = 100 * (np.sum(sim.F) - initial_F) / initial_F
        
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['mass_err'].append(mass_err)
        rec['blobs'].append(num)
        rec['q_max'].append(np.max(sim.q))
        rec['min_a'].append(np.min(sim.a))
        
        if t in [0, steps//6, 2*steps//6, 3*steps//6, 4*steps//6, 5*steps//6]:
            snapshots.append((t, sim.F.copy(), sim.q.copy(), sim.a.copy()))
        
        if t % 2000 == 0:
            print(f"  t={t}: sep={sep:.1f}, blobs={num}, mass_err={mass_err:+.2f}%, q_max={np.max(sim.q):.3f}")
        
        sim.step()
    
    min_sep = min(rec['sep'])
    collision = min_sep < 5
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Min separation: {min_sep:.1f}")
    print(f"  Collision: {'YES' if collision else 'NO'}")
    print(f"  Final blobs: {rec['blobs'][-1]}")
    print(f"  Final mass error: {rec['mass_err'][-1]:+.2f}%")
    print(f"  Max q reached: {max(rec['q_max']):.4f}")
    print(f"  Min a reached: {min(rec['min_a']):.4f}")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 14))
    
    # Row 1: F snapshots
    for i, (t, F, q, a) in enumerate(snapshots):
        ax = fig.add_subplot(4, 6, i + 1)
        ax.imshow(F, cmap='plasma', vmin=0)
        ax.set_title(f'F t={t}', fontsize=9)
        ax.axis('off')
    
    # Row 2: Agency snapshots
    for i, (t, F, q, a) in enumerate(snapshots):
        ax = fig.add_subplot(4, 6, i + 7)
        ax.imshow(a, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'Agency t={t}', fontsize=9)
        ax.axis('off')
    
    # Row 3: Time series
    ax = fig.add_subplot(4, 3, 7)
    ax.plot(rec['t'], rec['sep'], 'b-', lw=1.5)
    ax.axhline(5, color='g', ls='--', alpha=0.5)
    ax.fill_between(rec['t'], 0, rec['sep'], where=[s < 10 for s in rec['sep']], alpha=0.2, color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Separation')
    ax.set_title('Inter-body Separation')
    
    ax = fig.add_subplot(4, 3, 8)
    ax.plot(rec['t'], rec['min_a'], 'g-', lw=1.5, label='min(a)')
    ax.plot(rec['t'], rec['q_max'], 'r-', lw=1.5, label='max(q)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Agency & Structure')
    ax.legend(fontsize=8)
    
    ax = fig.add_subplot(4, 3, 9)
    ax.plot(rec['t'], rec['mass_err'], 'm-', lw=1.5)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mass Error %')
    ax.set_title('Mass Conservation')
    
    # Row 4: Summary
    ax = fig.add_subplot(4, 3, 10)
    ax.plot(rec['t'], rec['blobs'], 'k-', lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('# Blobs')
    ax.set_title('Blob Count')
    
    ax = fig.add_subplot(4, 3, 11)
    ax.axis('off')
    summary = f"""
DET v5 2D Collider Results
==========================

✓ Collision: {'YES' if collision else 'NO'}
✓ Min separation: {min_sep:.1f}
✓ Final blobs: {rec['blobs'][-1]}
✓ Mass error: {rec['mass_err'][-1]:+.2f}%

Agency Gate Mechanism:
- q accumulates on compression
- High q → low a
- Low a → diffusion dies
- Gravity/momentum → binding
"""
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('DET v5 2D Collider - Working Implementation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./det_v5_2d_collider_optimized.png', dpi=150)
    print("\nSaved: det_v5_2d_collider_optimized.png")
    
    return rec


if __name__ == "__main__":
    start = time.time()
    rec = run_optimized_test()
    elapsed = time.time() - start
    print(f"\nRuntime: {elapsed:.1f}s")