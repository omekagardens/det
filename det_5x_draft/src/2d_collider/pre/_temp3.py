"""
DET v5 2D Collider - Final Working Implementation
=================================================

Based on working 1D collider patterns:
- Bond-based momentum (antisymmetric)
- Conservative limiter (total outflow per node)
- Constant dt for mass conservation
- Momentum accumulates from J^{diff} only
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from dataclasses import dataclass
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
        
        # Non-periodic distance for packet placement
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
        J_diff_E_scaled = np.where(J_diff_E > 0, J_diff_E * scale, J_diff_E)
        J_diff_S_scaled = np.where(J_diff_S > 0, J_diff_S * scale, J_diff_S)
        
        for arr in [J_E, J_W, J_S, J_N]:
            arr[:] = np.clip(arr, -10, 10)
        
        inflow = W(J_E) + E(J_W) + Nb(J_S) + S(J_N)
        outflow = J_E + J_W + J_S + J_N
        dF = inflow - outflow
        F_new = np.clip(self.F + dF * dt, p.F_VAC, 1000)
        
        if p.momentum_enabled:
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)
            self.pi_E = decay_E * self.pi_E + p.alpha_pi * J_diff_E_scaled * Delta_tau_E
            self.pi_S = decay_S * self.pi_S + p.alpha_pi * J_diff_S_scaled * Delta_tau_S
        
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
    
    def find_peaks(self, threshold=0.5, min_dist=5):
        """Find local maxima in F field."""
        data_max = ndimage.maximum_filter(self.F, min_dist)
        peaks = (self.F == data_max) & (self.F > threshold)
        y_peaks, x_peaks = np.where(peaks)
        return list(zip(y_peaks, x_peaks))
    
    def get_blob_coms(self, threshold=None):
        """Find center of mass of each connected region."""
        if threshold is None:
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
            
            # Weighted COM
            com_x = np.sum(x[mask] * weights) / total_mass
            com_y = np.sum(y[mask] * weights) / total_mass
            coms.append({'x': com_x, 'y': com_y, 'mass': total_mass})
        
        coms.sort(key=lambda c: -c['mass'])
        return coms
    
    def separation(self):
        """Compute separation between two largest blobs."""
        coms = self.get_blob_coms()
        if len(coms) < 2:
            return 0.0, len(coms)
        
        dx = coms[1]['x'] - coms[0]['x']
        dy = coms[1]['y'] - coms[0]['y']
        
        # Handle periodic
        N = self.p.N
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        
        return np.sqrt(dx**2 + dy**2), len(coms)
    
    def diagnostics(self):
        sep, num_blobs = self.separation()
        return {
            'step': self.step_count,
            'total_F': np.sum(self.F),
            'num_blobs': num_blobs,
            'separation': sep,
            'max_q': np.max(self.q),
            'min_a': np.min(self.a),
            'total_pi': np.sum(np.abs(self.pi_E)) + np.sum(np.abs(self.pi_S)),
            'max_F': np.max(self.F),
        }


def run_collision_sweep():
    """Run collision tests with different momentum values."""
    print("="*70)
    print("DET v5 2D COLLIDER - MOMENTUM SWEEP")
    print("="*70)
    
    results = []
    
    for p in [0.3, 0.5, 0.8, 1.0]:
        print(f"\n--- Testing momentum p = {p} ---")
        
        params = DETParams()
        sim = DETCollider2D(params)
        
        # Place bodies further apart for clearer separation
        sim.add_packet((50, 30), mass=6.0, width=5.0, momentum=(0, p))
        sim.add_packet((50, 70), mass=6.0, width=5.0, momentum=(0, -p))
        
        initial_F = np.sum(sim.F)
        
        # Record initial state
        sep0, n0 = sim.separation()
        print(f"  Initial: sep={sep0:.1f}, blobs={n0}")
        
        rec = {'t': [], 'sep': [], 'mass_err': [], 'blobs': [], 'q_max': []}
        snapshots = []
        
        steps = 8000
        for t in range(steps):
            d = sim.diagnostics()
            rec['t'].append(t)
            rec['sep'].append(d['separation'])
            rec['mass_err'].append(100 * (d['total_F'] - initial_F) / initial_F)
            rec['blobs'].append(d['num_blobs'])
            rec['q_max'].append(d['max_q'])
            
            if t in [0, steps//4, steps//2, 3*steps//4, steps-1]:
                snapshots.append((t, sim.F.copy(), sim.q.copy(), sim.a.copy()))
            
            sim.step()
        
        min_sep = min(rec['sep'])
        collision = min_sep < 5
        final_blobs = rec['blobs'][-1]
        mass_err = rec['mass_err'][-1]
        
        print(f"  Final: min_sep={min_sep:.1f}, collision={collision}, blobs={final_blobs}, mass_err={mass_err:+.2f}%")
        
        results.append({
            'p': p,
            'min_sep': min_sep,
            'collision': collision,
            'final_blobs': final_blobs,
            'mass_err': mass_err,
            'record': rec,
            'snapshots': snapshots,
        })
    
    return results


def create_summary_plot(results, save_path):
    """Create summary visualization."""
    fig = plt.figure(figsize=(18, 16))
    
    # Find best result (collision with lowest mass error)
    colliding = [r for r in results if r['collision']]
    if colliding:
        best = min(colliding, key=lambda r: abs(r['mass_err']))
    else:
        best = results[-1]
    
    # Row 1-2: Best result snapshots
    snapshots = best['snapshots']
    for i, (t, F, q, a) in enumerate(snapshots):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(F, cmap='plasma', vmin=0)
        ax.set_title(f'F t={t}', fontsize=9)
        ax.axis('off')
        
        ax = fig.add_subplot(5, 5, i + 6)
        ax.imshow(a, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(f'Agency t={t}', fontsize=9)
        ax.axis('off')
    
    # Row 3: Separation curves
    ax = fig.add_subplot(5, 2, 5)
    for r in results:
        label = f"p={r['p']}"
        ax.plot(r['record']['t'], r['record']['sep'], label=label, lw=1.5)
    ax.axhline(5, color='g', ls='--', alpha=0.5, label='Collision')
    ax.set_xlabel('Time')
    ax.set_ylabel('Separation')
    ax.set_title('Separation vs Time')
    ax.legend(fontsize=8)
    
    ax = fig.add_subplot(5, 2, 6)
    for r in results:
        label = f"p={r['p']}"
        ax.plot(r['record']['t'], r['record']['mass_err'], label=label, lw=1.5)
    ax.axhline(0, color='k', ls='-', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Mass Error %')
    ax.set_title('Mass Conservation')
    ax.legend(fontsize=8)
    
    # Row 4: q and blobs
    ax = fig.add_subplot(5, 2, 7)
    for r in results:
        ax.plot(r['record']['t'], r['record']['q_max'], label=f"p={r['p']}", lw=1.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('max(q)')
    ax.set_title('Structure Formation')
    ax.legend(fontsize=8)
    
    ax = fig.add_subplot(5, 2, 8)
    for r in results:
        ax.plot(r['record']['t'], r['record']['blobs'], label=f"p={r['p']}", lw=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('# Blobs')
    ax.set_title('Blob Count')
    ax.legend(fontsize=8)
    
    # Row 5: Summary table
    ax = fig.add_subplot(5, 1, 5)
    ax.axis('off')
    
    table_data = []
    for r in results:
        table_data.append([
            f"{r['p']:.1f}",
            f"{r['min_sep']:.1f}",
            "YES" if r['collision'] else "NO",
            f"{r['final_blobs']}",
            f"{r['mass_err']:+.2f}%"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Momentum', 'Min Sep', 'Collision', 'Final Blobs', 'Mass Error'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    plt.suptitle('DET v5 2D Collider - Momentum Sweep Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    return fig


if __name__ == "__main__":
    start = time.time()
    results = run_collision_sweep()
    elapsed = time.time() - start
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Momentum':<10}{'Min Sep':<10}{'Collision':<12}{'Blobs':<10}{'Mass Err':<12}")
    print("-"*54)
    for r in results:
        print(f"{r['p']:<10.1f}{r['min_sep']:<10.1f}{'YES' if r['collision'] else 'NO':<12}{r['final_blobs']:<10}{r['mass_err']:+.2f}%")
    
    print(f"\nRuntime: {elapsed:.1f}s")
    
    create_summary_plot(results, '/mnt/user-data/outputs/det_v5_2d_collider_final.png')
