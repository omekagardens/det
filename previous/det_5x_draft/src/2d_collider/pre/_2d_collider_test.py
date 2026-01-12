"""
DET v5 2D COLLIDER - Based on Working 1D Implementation
=======================================================

Key principles from 1D collider v2:
1. Bond-based momentum (π_{ij} per bond, antisymmetric)
2. Initial momentum drives approach (not gravity)
3. Conservative limiter (total outflow per node)
4. Constant dt for resource update (mass conservation)
5. Momentum accumulates from J^{diff} only
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
from dataclasses import dataclass
from typing import Tuple, List, Dict
import time


@dataclass
class DETParams:
    """Complete DET v5 parameter set for 2D"""
    N: int = 100
    DT: float = 0.02
    F_VAC: float = 0.01
    R: int = 5  # Local normalization radius
    
    # Phase
    nu: float = 0.1
    
    # Coherence
    C_init: float = 0.3
    
    # Momentum (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.1   # Accumulation from J^diff
    lambda_pi: float = 0.01  # Decay/friction
    mu_pi: float = 0.3      # Flow coupling
    pi_max: float = 3.0     # Clip bound
    
    # q-locking
    q_enabled: bool = True
    alpha_q: float = 0.02
    
    # Agency (target-tracking)
    a_coupling: float = 30.0
    a_rate: float = 0.2
    
    # Floor repulsion (IV.3a)
    floor_enabled: bool = True
    eta_floor: float = 0.08
    F_core: float = 5.0
    floor_power: float = 2.0
    
    # Gravity (optional - for testing)
    gravity_enabled: bool = False
    kappa: float = 5.0
    alpha_baseline: float = 0.1
    mu_grav: float = 0.1
    
    # Numerical
    outflow_limit: float = 0.25


def periodic_local_sum_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Periodic local sum for 2D array using rolls."""
    result = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
    return result


class DETCollider2D:
    """
    DET v5 2D Collider following 1D collider patterns.
    
    Per-node: F, q, θ, a
    Per-bond: π_E (east), π_S (south) - antisymmetric
    """
    
    def __init__(self, params: DETParams = None):
        self.p = params or DETParams()
        N = self.p.N
        
        # Per-node state
        self.F = np.ones((N, N)) * self.p.F_VAC
        self.q = np.zeros((N, N))
        self.theta = np.zeros((N, N))
        self.a = np.ones((N, N))
        
        # Per-bond momentum: π_E[i,j] = momentum from (i,j) to (i,j+1)
        # π_S[i,j] = momentum from (i,j) to (i+1,j)
        self.pi_E = np.zeros((N, N))  # Eastward bond momentum
        self.pi_S = np.zeros((N, N))  # Southward bond momentum
        
        # Coherence (simplified: per-direction)
        self.C_E = np.ones((N, N)) * self.p.C_init
        self.C_S = np.ones((N, N)) * self.p.C_init
        
        # Processing rate
        self.sigma = np.ones((N, N))
        
        # Gravity (optional)
        if self.p.gravity_enabled:
            self._init_gravity()
        
        self.time = 0.0
        self.step_count = 0
    
    def _init_gravity(self):
        """Initialize gravity solver."""
        N = self.p.N
        k = fftfreq(N, d=1.0) * 2 * np.pi
        kx, ky = np.meshgrid(k, k)
        self.k_squared = kx**2 + ky**2
        self.k_squared[0, 0] = 1.0
    
    def _compute_gravity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gravitational potential and gradient."""
        if not self.p.gravity_enabled:
            N = self.p.N
            return np.zeros((N, N)), np.zeros((N, N))
        
        # Baseline field
        rhs_hat = fft2(-self.p.alpha_baseline * self.q)
        denom = -self.k_squared - self.p.alpha_baseline
        denom[0, 0] = -self.p.alpha_baseline
        b = np.real(ifft2(rhs_hat / denom))
        
        # Source and potential
        rho = self.q - b
        phi_hat = -self.p.kappa * fft2(rho) / self.k_squared
        phi_hat[0, 0] = 0.0
        Phi = np.real(ifft2(phi_hat))
        
        return Phi
    
    def add_packet(self, center: Tuple[float, float], mass: float = 6.0,
                   width: float = 5.0, momentum: Tuple[float, float] = (0, 0),
                   C_boost: float = 0.7, use_phase_gradient: bool = False):
        """
        Add Gaussian resource packet with optional momentum.
        
        Args:
            center: (y, x) position
            mass: Peak amplitude
            width: Gaussian width
            momentum: (py, px) - momentum in y and x directions
            C_boost: Coherence boost in packet region
            use_phase_gradient: If True, use phase gradient for momentum (physical)
                               If False, directly inject π (control knob)
        """
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        cy, cx = center
        
        # Periodic distance
        dx = x - cx
        dx = np.where(dx > N/2, dx - N, dx)
        dx = np.where(dx < -N/2, dx + N, dx)
        dy = y - cy
        dy = np.where(dy > N/2, dy - N, dy)
        dy = np.where(dy < -N/2, dy + N, dy)
        
        r2 = dx**2 + dy**2
        envelope = np.exp(-0.5 * r2 / width**2)
        
        self.F += mass * envelope
        self.C_E = np.clip(self.C_E + C_boost * envelope, self.p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + C_boost * envelope, self.p.C_init, 1.0)
        
        # Add momentum
        py, px = momentum
        if px != 0 or py != 0:
            mom_width = width * 3
            mom_env = np.exp(-0.5 * r2 / mom_width**2)
            
            if use_phase_gradient:
                # Physical: set phase gradient that will naturally charge π
                self.theta += (px * dx + py * dy) * mom_env
                self.theta = np.mod(self.theta, 2 * np.pi)
            else:
                # Control knob: directly inject π
                self.pi_E += px * mom_env
                self.pi_S += py * mom_env
        
        self._clip()
    
    def _clip(self):
        """Clip all state variables to valid ranges."""
        self.F = np.clip(self.F, self.p.F_VAC, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_E = np.clip(self.pi_E, -self.p.pi_max, self.p.pi_max)
        self.pi_S = np.clip(self.pi_S, -self.p.pi_max, self.p.pi_max)
    
    def step(self):
        """Execute one DET v5 update step."""
        p = self.p
        N = p.N
        dt = p.DT
        
        # Neighbor indices
        E = lambda x: np.roll(x, -1, axis=1)  # East neighbor
        W = lambda x: np.roll(x, 1, axis=1)   # West neighbor
        S = lambda x: np.roll(x, -1, axis=0)  # South neighbor  
        Nb = lambda x: np.roll(x, 1, axis=0)  # North neighbor
        
        # === STEP 1: Presence and Time ===
        P = self.a / (1.0 + self.F)
        Delta_tau = P * dt
        
        # Bond-local time
        Delta_tau_E = 0.5 * (Delta_tau + E(Delta_tau))
        Delta_tau_S = 0.5 * (Delta_tau + S(Delta_tau))
        
        # === STEP 2a: Wavefunction ===
        F_local = periodic_local_sum_2d(self.F, p.R) + 1e-9
        amp = np.sqrt(np.clip(self.F / F_local, 0, 1))
        psi = amp * np.exp(1j * self.theta)
        
        # === STEP 2b: Diffusive Flow J^{diff} ===
        # Quantum terms
        quantum_E = np.imag(np.conj(psi) * E(psi))
        quantum_W = np.imag(np.conj(psi) * W(psi))
        quantum_S = np.imag(np.conj(psi) * S(psi))
        quantum_N = np.imag(np.conj(psi) * Nb(psi))
        
        # Classical (pressure) terms
        classical_E = self.F - E(self.F)
        classical_W = self.F - W(self.F)
        classical_S = self.F - S(self.F)
        classical_N = self.F - Nb(self.F)
        
        # Interpolated drive
        sqrt_C_E = np.sqrt(self.C_E)
        sqrt_C_S = np.sqrt(self.C_S)
        sqrt_C_W = np.sqrt(W(self.C_E))  # West bond = East bond of west neighbor
        sqrt_C_N = np.sqrt(Nb(self.C_S))  # North bond = South bond of north neighbor
        
        drive_E = sqrt_C_E * quantum_E + (1 - sqrt_C_E) * classical_E
        drive_W = sqrt_C_W * quantum_W + (1 - sqrt_C_W) * classical_W
        drive_S = sqrt_C_S * quantum_S + (1 - sqrt_C_S) * classical_S
        drive_N = sqrt_C_N * quantum_N + (1 - sqrt_C_N) * classical_N
        
        # Conductance
        cond_E = self.sigma * (self.C_E + 1e-4)
        cond_W = self.sigma * (W(self.C_E) + 1e-4)
        cond_S = self.sigma * (self.C_S + 1e-4)
        cond_N = self.sigma * (Nb(self.C_S) + 1e-4)
        
        J_diff_E = cond_E * drive_E
        J_diff_W = cond_W * drive_W
        J_diff_S = cond_S * drive_S
        J_diff_N = cond_N * drive_N
        
        # === STEP 2c: Momentum Flow J^{mom} ===
        if p.momentum_enabled:
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_W = 0.5 * (self.F + W(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            F_avg_N = 0.5 * (self.F + Nb(self.F))
            
            # π is antisymmetric: π_{j←i} = -π_{i→j}
            J_mom_E = p.mu_pi * self.sigma * self.pi_E * F_avg_E
            J_mom_W = -p.mu_pi * self.sigma * W(self.pi_E) * F_avg_W  # Antisymmetric
            J_mom_S = p.mu_pi * self.sigma * self.pi_S * F_avg_S
            J_mom_N = -p.mu_pi * self.sigma * Nb(self.pi_S) * F_avg_N  # Antisymmetric
        else:
            J_mom_E = J_mom_W = J_mom_S = J_mom_N = 0
        
        # === STEP 2d: Floor Repulsion J^{floor} ===
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            s_E = s + E(s)
            s_W = s + W(s)
            s_S = s + S(s)
            s_N = s + Nb(s)
            
            J_floor_E = p.eta_floor * self.sigma * s_E * classical_E
            J_floor_W = p.eta_floor * self.sigma * s_W * classical_W
            J_floor_S = p.eta_floor * self.sigma * s_S * classical_S
            J_floor_N = p.eta_floor * self.sigma * s_N * classical_N
        else:
            J_floor_E = J_floor_W = J_floor_S = J_floor_N = 0
        
        # === STEP 2e: Gravity Flow (optional) ===
        if p.gravity_enabled:
            Phi = self._compute_gravity()
            dPhi_E = Phi - E(Phi)
            dPhi_S = Phi - S(Phi)
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            J_grav_E = p.mu_grav * self.sigma * F_avg_E * dPhi_E
            J_grav_S = p.mu_grav * self.sigma * F_avg_S * dPhi_S
            J_grav_W = -W(J_grav_E)
            J_grav_N = -Nb(J_grav_S)
        else:
            J_grav_E = J_grav_W = J_grav_S = J_grav_N = 0
        
        # === Total Flow ===
        J_E = J_diff_E + J_mom_E + J_floor_E + J_grav_E
        J_W = J_diff_W + J_mom_W + J_floor_W + J_grav_W
        J_S = J_diff_S + J_mom_S + J_floor_S + J_grav_S
        J_N = J_diff_N + J_mom_N + J_floor_N + J_grav_N
        
        # === Conservative Limiter (FIX #5 from 1D) ===
        # Total outflow = sum of positive flows to all neighbors
        outflow_E = np.maximum(0, J_E)
        outflow_W = np.maximum(0, J_W)
        outflow_S = np.maximum(0, J_S)
        outflow_N = np.maximum(0, J_N)
        total_outflow = outflow_E + outflow_W + outflow_S + outflow_N
        
        max_total_out = p.outflow_limit * self.F / dt
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))
        
        # Apply scaling to positive (outgoing) flows only
        J_E = np.where(J_E > 0, J_E * scale, J_E)
        J_W = np.where(J_W > 0, J_W * scale, J_W)
        J_S = np.where(J_S > 0, J_S * scale, J_S)
        J_N = np.where(J_N > 0, J_N * scale, J_N)
        
        # Scale J_diff for momentum consistency
        J_diff_E = np.where(J_diff_E > 0, J_diff_E * scale, J_diff_E)
        J_diff_S = np.where(J_diff_S > 0, J_diff_S * scale, J_diff_S)
        
        # Hard clamp
        for arr in [J_E, J_W, J_S, J_N]:
            arr[:] = np.clip(arr, -10, 10)
        
        # === STEP 5: Resource Update (constant dt for conservation!) ===
        # dF = (inflows) - (outflows)
        # Inflows: J from neighbors pointing toward us
        # J_E flows to east, so inflow from west is J_E from west neighbor = W(J_E)
        inflow = W(J_E) + E(J_W) + Nb(J_S) + S(J_N)
        outflow = J_E + J_W + J_S + J_N
        dF = inflow - outflow
        
        F_new = np.clip(self.F + dF * dt, p.F_VAC, 1000)
        
        # === STEP 5a: Momentum Update ===
        if p.momentum_enabled:
            # Clamp decay factor to prevent sign flip (FIX #3 from 1D)
            decay_E = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_E)
            decay_S = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_S)
            
            self.pi_E = decay_E * self.pi_E + p.alpha_pi * J_diff_E * Delta_tau_E
            self.pi_S = decay_S * self.pi_S + p.alpha_pi * J_diff_S * Delta_tau_S
        
        # === STEP 6: q-locking ===
        if p.q_enabled:
            dq = p.alpha_q * np.maximum(0, -dF)
            self.q = np.clip(self.q + dq, 0, 1)
        
        self.F = F_new
        
        # === STEP 7: Agency Update (target-tracking) ===
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        # === STEP 8: Phase Update ===
        d_E = np.angle(np.exp(1j * (E(self.theta) - self.theta)))
        d_W = np.angle(np.exp(1j * (W(self.theta) - self.theta)))
        d_S = np.angle(np.exp(1j * (S(self.theta) - self.theta)))
        d_N = np.angle(np.exp(1j * (Nb(self.theta) - self.theta)))
        lap_theta = d_E + d_W + d_S + d_N
        
        self.theta = self.theta + p.nu * lap_theta * dt
        self.theta = np.mod(self.theta, 2 * np.pi)
        
        # === Coherence Update ===
        self.C_E = np.clip(self.C_E + 0.05 * np.abs(J_E) * dt - 0.002 * self.C_E * dt, 
                          p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.05 * np.abs(J_S) * dt - 0.002 * self.C_S * dt,
                          p.C_init, 1.0)
        
        # === Conductivity Update ===
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_E) + np.abs(J_S))
        
        self._clip()
        self.time += dt
        self.step_count += 1
    
    def get_blobs(self, threshold: float = None) -> List[Dict]:
        """Find connected regions (blobs) above threshold."""
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
            
            # Periodic COM using circular mean
            positions_x = x[mask].astype(float)
            positions_y = y[mask].astype(float)
            
            angles_x = 2 * np.pi * positions_x / N
            angles_y = 2 * np.pi * positions_y / N
            
            com_x = np.arctan2(
                np.sum(weights * np.sin(angles_x)),
                np.sum(weights * np.cos(angles_x))
            )
            if com_x < 0:
                com_x += 2 * np.pi
            com_x = com_x * N / (2 * np.pi)
            
            com_y = np.arctan2(
                np.sum(weights * np.sin(angles_y)),
                np.sum(weights * np.cos(angles_y))
            )
            if com_y < 0:
                com_y += 2 * np.pi
            com_y = com_y * N / (2 * np.pi)
            
            blobs.append({
                'com': (com_y, com_x),
                'mass': total_mass,
                'size': np.sum(mask),
                'peak': np.max(weights)
            })
        
        blobs.sort(key=lambda b: -b['mass'])
        return blobs
    
    def separation(self) -> Tuple[float, bool]:
        """Compute separation between two largest blobs."""
        blobs = self.get_blobs()
        if len(blobs) < 2:
            return 0.0, False
        
        com1 = blobs[0]['com']
        com2 = blobs[1]['com']
        
        N = self.p.N
        dy = com2[0] - com1[0]
        dx = com2[1] - com1[1]
        
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        
        return np.sqrt(dx**2 + dy**2), True
    
    def diagnostics(self) -> Dict:
        sep, valid = self.separation()
        blobs = self.get_blobs()
        return {
            'time': self.time,
            'step': self.step_count,
            'total_F': np.sum(self.F),
            'num_blobs': len(blobs),
            'separation': sep,
            'sep_valid': valid,
            'max_q': np.max(self.q),
            'min_a': np.min(self.a),
            'total_pi': np.sum(np.abs(self.pi_E)) + np.sum(np.abs(self.pi_S)),
        }


def run_collision_test(initial_momentum: float = 0.5, steps: int = 25000,
                       use_phase_gradient: bool = False):
    """Run two-body collision test."""
    print(f"\n{'='*70}")
    print(f"COLLISION TEST: p={initial_momentum}, phase_grad={use_phase_gradient}")
    print(f"{'='*70}")
    
    params = DETParams()
    sim = DETCollider2D(params)
    
    N = params.N
    center = N // 2
    sep = 15  # Initial separation (half-distance from center)
    
    # Add two packets approaching each other
    sim.add_packet((center, center - sep), mass=6.0, width=5.0,
                   momentum=(0, initial_momentum), use_phase_gradient=use_phase_gradient)
    sim.add_packet((center, center + sep), mass=6.0, width=5.0,
                   momentum=(0, -initial_momentum), use_phase_gradient=use_phase_gradient)
    
    initial_F = np.sum(sim.F)
    
    rec = {'t': [], 'sep': [], 'mass_err': [], 'num_blobs': [], 'q_max': [], 'pi': []}
    snapshots = []
    
    for t in range(steps):
        d = sim.diagnostics()
        
        rec['t'].append(d['time'])
        rec['sep'].append(d['separation'])
        rec['mass_err'].append(100 * (d['total_F'] - initial_F) / initial_F)
        rec['num_blobs'].append(d['num_blobs'])
        rec['q_max'].append(d['max_q'])
        rec['pi'].append(d['total_pi'])
        
        if t in [0, steps//5, 2*steps//5, 3*steps//5, 4*steps//5, steps-1]:
            snapshots.append((t, sim.F.copy(), sim.q.copy()))
        
        if t % 5000 == 0:
            print(f"  t={t}: sep={d['separation']:.1f}, blobs={d['num_blobs']}, "
                  f"mass_err={rec['mass_err'][-1]:+.2f}%, q_max={d['max_q']:.3f}")
        
        sim.step()
    
    # Results
    min_sep = min(rec['sep'])
    collision = min_sep < 10
    merged = rec['num_blobs'][-1] == 1
    final_mass_err = rec['mass_err'][-1]
    
    print(f"\nResults:")
    print(f"  Min separation: {min_sep:.1f}")
    print(f"  Collision: {'YES' if collision else 'NO'}")
    print(f"  Merged: {'YES' if merged else 'NO'}")
    print(f"  Mass error: {final_mass_err:+.2f}%")
    print(f"  Final q_max: {rec['q_max'][-1]:.4f}")
    
    return {
        'record': rec,
        'snapshots': snapshots,
        'min_sep': min_sep,
        'collision': collision,
        'merged': merged,
        'mass_err': final_mass_err,
    }


def create_visualization(result: Dict, save_path: str = None):
    """Create visualization of collision results."""
    rec = result['record']
    snapshots = result['snapshots']
    
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: F snapshots
    for i, (t, F, q) in enumerate(snapshots):
        ax = fig.add_subplot(4, 6, i + 1)
        ax.imshow(F, cmap='plasma', vmin=0)
        ax.set_title(f't={t}', fontsize=10)
        ax.axis('off')
    
    # Row 2: q snapshots
    for i, (t, F, q) in enumerate(snapshots):
        ax = fig.add_subplot(4, 6, i + 7)
        ax.imshow(q, cmap='Reds', vmin=0, vmax=1)
        ax.set_title(f'q t={t}', fontsize=10)
        ax.axis('off')
    
    # Row 3: Time series
    ax = fig.add_subplot(4, 3, 7)
    ax.plot(rec['t'], rec['sep'], 'b-', lw=1.5)
    ax.axhline(10, color='g', ls='--', alpha=0.5, label='Collision threshold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Separation')
    ax.set_title('Separation')
    ax.legend(fontsize=8)
    
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
    
    # Row 4: More diagnostics
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
    
    ax = fig.add_subplot(4, 3, 12)
    ax.axis('off')
    summary = f"""
Results:
  Min separation: {result['min_sep']:.1f}
  Collision: {'YES ✓' if result['collision'] else 'NO'}
  Merged: {'YES' if result['merged'] else 'NO'}
  Mass error: {result['mass_err']:+.2f}%
"""
    ax.text(0.1, 0.7, summary, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('DET v5 2D Collider (1D-style)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("="*70)
    print("DET v5 2D COLLIDER - Based on Working 1D Implementation")
    print("="*70)
    
    # Test 1: Direct momentum injection
    print("\n[1] Direct momentum injection (p=0.5)...")
    result_direct = run_collision_test(initial_momentum=0.5, use_phase_gradient=False)
    
    # Test 2: Phase gradient initialization
    print("\n[2] Phase gradient initialization (v=0.5)...")
    result_phase = run_collision_test(initial_momentum=0.5, use_phase_gradient=True)
    
    # Test 3: Higher momentum
    print("\n[3] Higher momentum (p=1.0)...")
    result_high = run_collision_test(initial_momentum=1.0, use_phase_gradient=False)
    
    # Create visualization for best result
    best = max([result_direct, result_phase, result_high], 
               key=lambda r: -r['min_sep'] if r['collision'] else 1000)
    fig = create_visualization(best, '/mnt/user-data/outputs/det_v5_2d_collider.png')
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Direct π (p=0.5): min_sep={result_direct['min_sep']:.1f}, collision={result_direct['collision']}")
    print(f"Phase grad (v=0.5): min_sep={result_phase['min_sep']:.1f}, collision={result_phase['collision']}")
    print(f"High π (p=1.0): min_sep={result_high['min_sep']:.1f}, collision={result_high['collision']}")