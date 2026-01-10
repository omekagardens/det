cat > /home/claude/det_v5_2d_quick_test.py << 'ENDOFFILE'
"""Quick test of the 2D collider from uploaded code."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
from dataclasses import dataclass
from typing import Tuple, List, Dict
import time


@dataclass
class DETParams:
    N: int = 100
    DT: float = 0.02
    F_VAC: float = 0.01
    R: int = 5
    nu: float = 0.1
    C_init: float = 0.3
    momentum_enabled: bool = True
    alpha_pi: float = 0.1
    lambda_pi: float = 0.01
    mu_pi: float = 0.3
    pi_max: float = 3.0
    q_enabled: bool = True
    alpha_q: float = 0.02
    a_coupling: float = 30.0
    a_rate: float = 0.2
    floor_enabled: bool = True
    eta_floor: float = 0.08
    F_core: float = 5.0
    floor_power: float = 2.0
    gravity_enabled: bool = False
    outflow_limit: float = 0.25


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
        dx = x - cx
        dx = np.where(dx > N/2, dx - N, dx)
        dx = np.where(dx < -N/2, dx + N, dx)
        dy = y - cy
        dy = np.where(dy > N/2, dy - N, dy)
        dy = np.where(dy < -N/2, dy + N, dy)
        r2 = dx**2 + dy**2
        envelope = np.exp(-0.5 * r2 / width**2)
        self.F += mass * envelope
        self.C_E = np.clip(self.C_E + 0.7 * envelope, self.p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.7 * envelope, self.p.C_init, 1.0)
        
        py, px = momentum
        if px != 0 or py != 0:
            mom_env = np.exp(-0.5 * r2 / (width*3)**2)
            self.pi_E += px * mom_env
            self.pi_S += py * mom_env
        
        self.F = np.clip(self.F, self.p.F_VAC, 1000)
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
        outflow_E = np.maximum(0, J_E)
        outflow_W = np.maximum(0, J_W)
        outflow_S = np.maximum(0, J_S)
        outflow_N = np.maximum(0, J_N)
        total_outflow = outflow_E + outflow_W + outflow_S + outflow_N
        max_total_out = p.outflow_limit * self.F / dt
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))
        
        J_E = np.where(J_E > 0, J_E * scale, J_E)
        J_W = np.where(J_W > 0, J_W * scale, J_W)
        J_S = np.where(J_S > 0, J_S * scale, J_S)
        J_N = np.where(J_N > 0, J_N * scale, J_N)
        J_diff_E = np.where(J_diff_E > 0, J_diff_E * scale, J_diff_E)
        J_diff_S = np.where(J_diff_S > 0, J_diff_S * scale, J_diff_S)
        
        for arr in [J_E, J_W, J_S, J_N]:
            arr[:] = np.clip(arr, -10, 10)
        
        inflow = W(J_E) + E(J_W) + Nb(J_S) + S(J_N)
        outflow = J_E + J_W + J_S + J_N
        dF = inflow - outflow
        F_new = np.clip(self.F + dF * dt, p.F_VAC, 1000)
        
        if p.momentum_enabled:
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)
            self.pi_E = decay_E * self.pi_E + p.alpha_pi * J_diff_E * Delta_tau_E
            self.pi_S = decay_S * self.pi_S + p.alpha_pi * J_diff_S * Delta_tau_S
        
        if p.q_enabled:
            dq = p.alpha_q * np.maximum(0, -dF)
            self.q = np.clip(self.q + dq, 0, 1)
        
        self.F = F_new
        
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        d_E = np.angle(np.exp(1j * (E(self.theta) - self.theta)))
        d_W = np.angle(np.exp(1j * (W(self.theta) - self.theta)))
        d_S = np.angle(np.exp(1j * (S(self.theta) - self.theta)))
        d_N = np.angle(np.exp(1j * (Nb(self.theta) - self.theta)))
        lap_theta = d_E + d_W + d_S + d_N
        self.theta = self.theta + p.nu * lap_theta * dt
        self.theta = np.mod(self.theta, 2 * np.pi)
        
        self.C_E = np.clip(self.C_E + 0.05 * np.abs(J_E) * dt - 0.002 * self.C_E * dt, p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.05 * np.abs(J_S) * dt - 0.002 * self.C_S * dt, p.C_init, 1.0)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_E) + np.abs(J_S))
        
        self.pi_E = np.clip(self.pi_E, -p.pi_max, p.pi_max)
        self.pi_S = np.clip(self.pi_S, -p.pi_max, p.pi_max)
        
        self.time += dt
        self.step_count += 1
    
    def get_blobs(self, threshold=None):
        if threshold is None:
            threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = ndimage.label(above)
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        blobs = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) == 0:
                continue
            weights = self.F[mask]
            total_mass = np.sum(weights)
            positions_x = x[mask].astype(float)
            positions_y = y[mask].astype(float)
            angles_x = 2 * np.pi * positions_x / N
            angles_y = 2 * np.pi * positions_y / N
            com_x = np.arctan2(np.sum(weights * np.sin(angles_x)), np.sum(weights * np.cos(angles_x)))
            if com_x < 0: com_x += 2 * np.pi
            com_x = com_x * N / (2 * np.pi)
            com_y = np.arctan2(np.sum(weights * np.sin(angles_y)), np.sum(weights * np.cos(angles_y)))
            if com_y < 0: com_y += 2 * np.pi
            com_y = com_y * N / (2 * np.pi)
            blobs.append({'com': (com_y, com_x), 'mass': total_mass})
        blobs.sort(key=lambda b: -b['mass'])
        return blobs
    
    def separation(self):
        blobs = self.get_blobs()
        if len(blobs) < 2:
            return 0.0, False
        com1, com2 = blobs[0]['com'], blobs[1]['com']
        N = self.p.N
        dy = com2[0] - com1[0]
        dx = com2[1] - com1[1]
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        return np.sqrt(dx**2 + dy**2), True
    
    def diagnostics(self):
        sep, valid = self.separation()
        blobs = self.get_blobs()
        return {
            'step': self.step_count,
            'total_F': np.sum(self.F),
            'num_blobs': len(blobs),
            'separation': sep,
            'max_q': np.max(self.q),
            'min_a': np.min(self.a),
            'total_pi': np.sum(np.abs(self.pi_E)) + np.sum(np.abs(self.pi_S)),
        }


def run_test(initial_momentum=0.5, steps=5000):
    print(f"\n{'='*70}")
    print(f"COLLISION TEST: p={initial_momentum}")
    print(f"{'='*70}")
    
    params = DETParams()
    sim = DETCollider2D(params)
    N = params.N
    center = N // 2
    sep = 15
    
    sim.add_packet((center, center - sep), mass=6.0, width=5.0, momentum=(0, initial_momentum))
    sim.add_packet((center, center + sep), mass=6.0, width=5.0, momentum=(0, -initial_momentum))
    
    initial_F = np.sum(sim.F)
    rec = {'t': [], 'sep': [], 'mass_err': [], 'num_blobs': [], 'q_max': [], 'pi': []}
    snapshots = []
    
    for t in range(steps):
        d = sim.diagnostics()
        rec['t'].append(t)
        rec['sep'].append(d['separation'])
        rec['mass_err'].append(100 * (d['total_F'] - initial_F) / initial_F)
        rec['num_blobs'].append(d['num_blobs'])
        rec['q_max'].append(d['max_q'])
        rec['pi'].append(d['total_pi'])
        
        if t in [0, steps//5, 2*steps//5, 3*steps//5, 4*steps//5, steps-1]:
            snapshots.append((t, sim.F.copy(), sim.q.copy()))
        
        if t % 1000 == 0:
            print(f"  t={t}: sep={d['separation']:.1f}, blobs={d['num_blobs']}, "
                  f"mass_err={rec['mass_err'][-1]:+.2f}%, q_max={d['max_q']:.3f}")
        
        sim.step()
    
    min_sep = min(rec['sep'])
    collision = min_sep < 10
    merged = rec['num_blobs'][-1] == 1
    final_mass_err = rec['mass_err'][-1]
    
    print(f"\nResults:")
    print(f"  Min separation: {min_sep:.1f}")
    print(f"  Collision: {'YES' if collision else 'NO'}")
    print(f"  Merged: {'YES' if merged else 'NO'}")
    print(f"  Mass error: {final_mass_err:+.2f}%")
    
    return {'record': rec, 'snapshots': snapshots, 'min_sep': min_sep, 
            'collision': collision, 'merged': merged, 'mass_err': final_mass_err}


def create_visualization(result, save_path=None):
    rec = result['record']
    snapshots = result['snapshots']
    
    fig = plt.figure(figsize=(16, 12))
    
    for i, (t, F, q) in enumerate(snapshots):
        ax = fig.add_subplot(4, 6, i + 1)
        ax.imshow(F, cmap='plasma', vmin=0)
        ax.set_title(f't={t}', fontsize=10)
        ax.axis('off')
    
    for i, (t, F, q) in enumerate(snapshots):
        ax = fig.add_subplot(4, 6, i + 7)
        ax.imshow(q, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'q t={t}', fontsize=10)
        ax.axis('off')
    
    ax = fig.add_subplot(4, 3, 7)
    ax.plot(rec['t'], rec['sep'], 'b-', lw=1.5)
    ax.axhline(10, color='g', ls='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Separation')
    ax.set_title('Separation')
    
    ax = fig.add_subplot(4, 3, 8)
    ax.plot(rec['t'], rec['q_max'], 'r-', lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('q_max')
    ax.set_title('Structure Formation')
    
    ax = fig.add_subplot(4, 3, 9)
    ax.plot(rec['t'], rec['mass_err'], 'm-', lw=1.5)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mass Error %')
    ax.set_title('Mass Conservation')
    
    ax = fig.add_subplot(4, 3, 10)
    ax.plot(rec['t'], rec['num_blobs'], 'k-', lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('# Blobs')
    ax.set_title('Blob Count')
    
    ax = fig.add_subplot(4, 3, 11)
    ax.plot(rec['t'], rec['pi'], 'orange', lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total |π|')
    ax.set_title('Momentum')
    
    plt.suptitle('DET v5 2D Collider (1D-style)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved: {save_path}")
    return fig


if __name__ == "__main__":
    start = time.time()
    result = run_test(initial_momentum=0.5, steps=5000)
    elapsed = time.time() - start
    print(f"\nRuntime: {elapsed:.1f}s")
    
    fig = create_visualization(result, '/mnt/user-data/outputs/det_v5_2d_collider_v2.png')

import time
ENDOFFILE
python3 /home/claude/det_v5_2d_quick_test.py 2>&1
Output


======================================================================
COLLISION TEST: p=0.5
======================================================================
  t=0: sep=0.0, blobs=1, mass_err=+0.00%, q_max=0.000
  t=1000: sep=0.0, blobs=1, mass_err=+0.09%, q_max=1.000
  t=2000: sep=0.0, blobs=1, mass_err=+0.17%, q_max=1.000
  t=3000: sep=18.2, blobs=2, mass_err=+0.23%, q_max=1.000
  t=4000: sep=0.0, blobs=1, mass_err=+0.28%, q_max=1.000

Results:
  Min separation: 0.0
  Collision: YES
  Merged: YES
  Mass error: +1.05%

Runtime: 34.4s

Saved: /mnt/user-data/outputs/det_v5_2d_collider_v2.png