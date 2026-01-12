"""
DET Canonical Collider v1.0
===========================

Implements canonical DET 4.2 flow dynamics for studying dipole collisions.

Key features:
- Canonical flow equation: J = σ[√C Im(ψ*ψ') + (1-√C)(F_i - F_j)]
- NO gravity term in flow (gravity is diagnostic only per DET 4.2)
- Canonical q-locking (Appendix B.3 Step 6)
- Declared extensions: flow-driven coherence, phase diffusion

Reference: DET 4.2 README, Sections IV.2, B.3, X

Author: DET Research
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


@dataclass
class DETParams:
    """Parameters for DET simulation"""
    # Domain
    N: int = 300                  # Grid size
    DT: float = 0.05              # Time step
    
    # Resource
    F_VAC: float = 0.01           # Vacuum level
    
    # Coherence
    C_MIN: float = 0.05           # Minimum coherence
    R: int = 10                   # Local normalization radius
    
    # Phase dynamics (extension)
    BETA: float = 0.01            # Resource-driven phase rotation
    NU: float = 0.2               # Phase diffusion
    PHASE_DRAG: float = 0.2       # Gradient-dependent drag
    
    # Coherence dynamics (extension)
    ALPHA: float = 0.1            # Flow-aligned reinforcement
    GAMMA: float = 4.0            # Neighbor reinforcement
    LAMBDA: float = 0.001         # Coherence decay
    
    # q-locking (canonical)
    ALPHA_Q: float = 0.02         # Structure formation rate
    
    # Dipole structure
    dipole_sep: float = 15.0      # Internal dipole separation
    width: float = 5.0            # Gaussian width
    amplitude: float = 8.0        # Peak amplitude


class DETState:
    """State variables for DET simulation"""
    def __init__(self, N: int, C_MIN: float = 0.05, F_VAC: float = 0.01):
        self.N = N
        self.x = np.arange(N)
        self.F = np.ones(N) * F_VAC          # Resource
        self.theta = np.zeros(N)              # Phase
        self.q = np.zeros(N)                  # Structural debt
        self.C_right = np.ones(N) * C_MIN     # Rightward coherence
        self.C_left = np.ones(N) * C_MIN      # Leftward coherence
        self.sigma = np.ones(N)               # Conductivity
        
    def add_gaussian(self, pos: float, amplitude: float, width: float, 
                     C_init: float = 0.8):
        """Add a Gaussian peak at position pos"""
        dx = self.x - pos
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        envelope = np.exp(-0.5 * (dx / width)**2)
        
        self.F += amplitude * envelope
        self.C_right += C_init * envelope
        self.C_left += C_init * envelope
        
    def add_dipole(self, center: float, params: DETParams, C_init: float = 0.8):
        """Add a dipole (two peaks) centered at position"""
        for offset in [-params.dipole_sep/2, params.dipole_sep/2]:
            self.add_gaussian(center + offset, params.amplitude, 
                            params.width, C_init)
    
    def clip_coherence(self, C_MIN: float):
        """Ensure coherence stays in valid range"""
        self.C_right = np.clip(self.C_right, C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, C_MIN, 1.0)


def local_sum(x: np.ndarray, radius: int = 3) -> np.ndarray:
    """Compute local sum within radius (for normalization)"""
    kernel = np.ones(2 * radius + 1)
    return np.convolve(x, kernel, mode='same')


def count_peaks(F: np.ndarray, F_VAC: float = 0.01, 
                threshold: float = 0.3) -> int:
    """Count number of peaks in F profile"""
    data = np.maximum(F - F_VAC, 0)
    data_max = maximum_filter(data, 7)
    maxima = (data == data_max) & (data > threshold)
    return np.sum(maxima)


def det_step(state: DETState, params: DETParams, 
             with_q_locking: bool = True) -> Dict:
    """
    Perform one canonical DET update step.
    
    Returns dict with diagnostic info.
    """
    N = state.N
    dt = params.DT
    
    # Index arrays for periodic boundary
    idx = np.arange(N)
    ip1 = (idx + 1) % N
    im1 = (idx - 1) % N
    
    # === STEP A: Wavefunction (IV.1) ===
    F_local_sum = local_sum(state.F, params.R) + 1e-9
    amp = np.sqrt(state.F / F_local_sum)
    psi = amp * np.exp(1j * state.theta)
    
    # === STEP B: Phase evolution (extension) ===
    d_fwd = np.angle(np.exp(1j * (state.theta[ip1] - state.theta[idx])))
    d_bwd = np.angle(np.exp(1j * (state.theta[im1] - state.theta[idx])))
    laplacian_theta = d_fwd + d_bwd
    grad_theta = np.angle(np.exp(1j * (state.theta[ip1] - state.theta[im1]))) / 2.0
    
    phase_drive = np.minimum(params.BETA * state.F * dt, np.pi/8.0)
    drag_factor = 1.0 / (1.0 + params.PHASE_DRAG * (grad_theta**2))
    
    state.theta = state.theta - phase_drive * drag_factor + params.NU * laplacian_theta * dt
    state.theta = np.mod(state.theta, 2*np.pi)
    
    # === STEP C: Quantum-Classical Flow (IV.2) - CANONICAL ===
    quantum_R = np.imag(np.conj(psi[idx]) * psi[ip1])
    quantum_L = np.imag(np.conj(psi[idx]) * psi[im1])
    classical_R = state.F[idx] - state.F[ip1]
    classical_L = state.F[idx] - state.F[im1]
    
    sqrt_C_R = np.sqrt(state.C_right)
    sqrt_C_L = np.sqrt(state.C_left)
    
    # Interpolated drive
    drive_R = sqrt_C_R * quantum_R + (1 - sqrt_C_R) * classical_R
    drive_L = sqrt_C_L * quantum_L + (1 - sqrt_C_L) * classical_L
    
    # Conductivity
    cond_R = state.sigma[idx] * (state.C_right[idx] + 1e-4)
    cond_L = state.sigma[idx] * (state.C_left[idx] + 1e-4)
    
    # CANONICAL FLOW - NO GRAVITY TERM!
    J_right = cond_R * drive_R
    J_left = cond_L * drive_L
    
    # === STEP D: Flow limiting ===
    max_allowed = 0.40 * state.F / dt
    total_out = np.abs(J_right) + np.abs(J_left)
    scale = np.minimum(1.0, max_allowed / (total_out + 1e-9))
    J_right = J_right * scale
    J_left = J_left * scale
    
    # === STEP E: Resource update (IV.3) ===
    dF = (J_right[im1] + J_left[ip1]) - (J_right + J_left)
    F_new = np.maximum(state.F + dF * dt, params.F_VAC)
    
    # === STEP F: q-locking (B.3 Step 6) - CANONICAL ===
    if with_q_locking:
        delta_F = F_new - state.F
        state.q = np.clip(state.q + params.ALPHA_Q * np.maximum(0, -delta_F), 0, 1)
    
    state.F = F_new
    
    # === STEP G: Coherence dynamics (extension) ===
    compression = np.maximum(0, dF)
    alpha = params.ALPHA * (1.0 + 40.0 * compression)
    lam = params.LAMBDA / (1.0 + 40.0 * compression * 10.0)
    
    J_R_align = np.maximum(0, J_right) + params.GAMMA * np.maximum(0, np.roll(J_right, 1))
    J_L_align = np.maximum(0, J_left) + params.GAMMA * np.maximum(0, np.roll(J_left, -1))
    
    state.C_right += (alpha * J_R_align - lam * state.C_right) * dt
    state.C_left += (alpha * J_L_align - lam * state.C_left) * dt
    
    state.clip_coherence(params.C_MIN)
    
    # === STEP H: Conductivity update ===
    state.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_right) + np.abs(J_left))
    
    # Return diagnostics
    return {
        'J_right': J_right,
        'J_left': J_left,
        'dF': dF,
        'quantum_R': quantum_R,
        'classical_R': classical_R
    }


def run_collision(overlap: float = 10.0, 
                  params: Optional[DETParams] = None,
                  steps: int = 15000,
                  with_q_locking: bool = True,
                  snapshot_times: Optional[List[float]] = None) -> Dict:
    """
    Run a dipole collision simulation.
    
    Args:
        overlap: How much dipoles overlap (>0 means overlapping)
        params: DET parameters (uses defaults if None)
        steps: Number of simulation steps
        with_q_locking: Whether to enable q-locking
        snapshot_times: Times at which to save snapshots
        
    Returns:
        Dictionary with traces and snapshots
    """
    if params is None:
        params = DETParams()
    
    if snapshot_times is None:
        snapshot_times = [0, 10, 50, 100, 200, 500]
    
    # Initialize state
    state = DETState(params.N, params.C_MIN, params.F_VAC)
    
    # Compute dipole placement
    center = params.N // 2
    spacing = params.dipole_sep * 2 - overlap
    
    # Add two dipoles
    state.add_dipole(center - spacing/2, params)
    state.add_dipole(center + spacing/2, params)
    state.clip_coherence(params.C_MIN)
    
    # Tracking arrays
    peak_trace = []
    q_max_trace = []
    q_sum_trace = []
    total_F_trace = []
    snapshots = []
    
    # Run simulation
    for step in range(steps):
        t = step * params.DT
        
        # Track
        peak_trace.append(count_peaks(state.F, params.F_VAC))
        q_max_trace.append(np.max(state.q))
        q_sum_trace.append(np.sum(state.q))
        total_F_trace.append(np.sum(state.F))
        
        # Save snapshots
        if any(abs(t - st) < params.DT/2 for st in snapshot_times):
            snapshots.append({
                't': t,
                'F': state.F.copy(),
                'q': state.q.copy(),
                'theta': state.theta.copy(),
                'C': (state.C_right + state.C_left).copy() / 2,
                'peaks': peak_trace[-1]
            })
        
        # Step
        det_step(state, params, with_q_locking)
    
    # Final snapshot
    snapshots.append({
        't': step * params.DT,
        'F': state.F.copy(),
        'q': state.q.copy(),
        'theta': state.theta.copy(),
        'C': (state.C_right + state.C_left) / 2,
        'peaks': peak_trace[-1]
    })
    
    return {
        'peaks': np.array(peak_trace),
        'q_max': np.array(q_max_trace),
        'q_sum': np.array(q_sum_trace),
        'total_F': np.array(total_F_trace),
        'snapshots': snapshots,
        'params': params,
        'overlap': overlap,
        'with_q_locking': with_q_locking
    }


def plot_collision_result(result: Dict, save_path: Optional[str] = None):
    """Create visualization of collision result"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    params = result['params']
    t = np.arange(len(result['peaks'])) * params.DT
    
    # Panel 1: Profile evolution
    ax = axes[0, 0]
    snaps = result['snapshots']
    colors = plt.cm.viridis(np.linspace(0, 1, len(snaps)))
    for i, s in enumerate(snaps):
        ax.plot(s['F'], color=colors[i], lw=1.5, alpha=0.8, 
                label=f"t={s['t']:.0f}")
    ax.set_xlabel("Position")
    ax.set_ylabel("F (resource)")
    ax.set_title("Resource Profile Evolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: q evolution
    ax = axes[0, 1]
    for i, s in enumerate(snaps):
        if np.max(s['q']) > 0.001:
            ax.plot(s['q'], color=colors[i], lw=1.5, alpha=0.8)
    ax.set_xlabel("Position")
    ax.set_ylabel("q (structural debt)")
    ax.set_title("Structure Formation")
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Peak count
    ax = axes[0, 2]
    ax.plot(t, result['peaks'], 'b-', lw=2)
    ax.axhline(2, color='g', ls='--', alpha=0.5, label='Single dipole')
    ax.set_xlabel("Time")
    ax.set_ylabel("Peak Count")
    ax.set_title("Peak Count Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 4: q traces
    ax = axes[1, 0]
    ax.plot(t, result['q_max'], 'r-', lw=2, label='max(q)')
    ax.plot(t, result['q_sum'] / params.N, 'r--', lw=1.5, label='mean(q)')
    ax.set_xlabel("Time")
    ax.set_ylabel("q")
    ax.set_title("Structure Accumulation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 5: Final state
    ax = axes[1, 1]
    final = snaps[-1]
    ax.plot(final['F'], 'b-', lw=2, label='F')
    if np.max(final['q']) > 0.001:
        scale = final['F'].max() / max(0.01, final['q'].max())
        ax.plot(final['q'] * scale, 'r--', lw=2, label=f'q (×{scale:.0f})')
    ax.set_xlabel("Position")
    ax.set_ylabel("F / q (scaled)")
    ax.set_title(f"Final State: {final['peaks']} peaks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 6: Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    init_peaks = snaps[0]['peaks']
    final_peaks = snaps[-1]['peaks']
    if final_peaks < init_peaks:
        outcome = "MERGER"
    elif final_peaks > init_peaks:
        outcome = "FRAGMENTATION"
    else:
        outcome = "STABLE"
    
    summary = f"""
    COLLISION SUMMARY
    ─────────────────
    Overlap: {result['overlap']:.1f}
    q-locking: {'ON' if result['with_q_locking'] else 'OFF'}
    
    Initial peaks: {init_peaks}
    Final peaks: {final_peaks}
    Outcome: {outcome}
    
    Final q_max: {result['q_max'][-1]:.4f}
    Final q_sum: {result['q_sum'][-1]:.2f}
    
    Total F conserved: {result['total_F'][-1]/result['total_F'][0]:.4f}
    """
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    q_lock_str = "with" if result['with_q_locking'] else "without"
    plt.suptitle(f"DET Canonical Collision (overlap={result['overlap']}, {q_lock_str} q-locking)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


def scan_overlaps(overlaps: List[float] = None,
                  params: Optional[DETParams] = None,
                  steps: int = 12000,
                  with_q_locking: bool = True) -> Dict:
    """
    Scan collision outcomes across different overlap values.
    
    Returns dictionary with results for each overlap.
    """
    if overlaps is None:
        overlaps = [-10, -5, 0, 5, 10, 15, 20, 25]
    
    if params is None:
        params = DETParams()
    
    results = {}
    
    print("Scanning overlaps...")
    print(f"{'Overlap':<10}|{'Initial':<10}|{'Final':<10}|{'q_max':<10}|{'Outcome':<15}")
    print("-" * 60)
    
    for ov in overlaps:
        r = run_collision(overlap=ov, params=params, steps=steps,
                         with_q_locking=with_q_locking,
                         snapshot_times=[0, steps*params.DT])
        results[ov] = r
        
        init = r['snapshots'][0]['peaks']
        final = r['peaks'][-1]
        q_max = r['q_max'][-1]
        
        if final < init:
            outcome = "MERGER"
        elif final > init:
            outcome = "FRAGMENTATION"
        else:
            outcome = "STABLE"
        
        print(f"{ov:<10}|{init:<10}|{final:<10}|{q_max:<10.4f}|{outcome:<15}")
    
    return results


def plot_scan_results(scan_results: Dict, save_path: Optional[str] = None):
    """Plot results from overlap scan"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    overlaps = sorted(scan_results.keys())
    final_peaks = [scan_results[ov]['peaks'][-1] for ov in overlaps]
    final_q = [scan_results[ov]['q_max'][-1] for ov in overlaps]
    
    # Panel 1: Peak count
    ax = axes[0]
    colors = ['green' if p == 2 else 'blue' if p > 3 else 'orange' 
              for p in final_peaks]
    ax.bar(overlaps, final_peaks, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(2, color='green', ls='--', lw=2, label='Single dipole')
    ax.axhline(4, color='blue', ls=':', lw=2, label='Two dipoles')
    ax.set_xlabel("Overlap (grid units)", fontsize=12)
    ax.set_ylabel("Final Peak Count", fontsize=12)
    ax.set_title("Collision Outcome vs Overlap", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: q accumulation
    ax = axes[1]
    ax.bar(overlaps, final_q, color='red', edgecolor='black', 
           linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Overlap (grid units)", fontsize=12)
    ax.set_ylabel("Final q_max", fontsize=12)
    ax.set_title("Mass Generation vs Overlap", fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate mergers
    for ov, fp, fq in zip(overlaps, final_peaks, final_q):
        if fp == 2 and fq > 0.1:
            ax.annotate('MERGER', (ov, fq), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=10,
                       color='darkgreen', fontweight='bold')
    
    plt.suptitle("DET Collision Phase Diagram", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    return fig


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DET CANONICAL COLLIDER")
    print("=" * 70)
    print("\nFlow equation: J = σ[√C Im(ψ*ψ') + (1-√C)(F_i - F_j)]")
    print("NO gravity term (canonical DET 4.2)")
    print("=" * 70)
    
    # Run overlap scan
    print("\n[1] Running overlap scan...")
    scan = scan_overlaps()
    
    # Plot scan results
    print("\n[2] Plotting scan results...")
    plot_scan_results(scan, save_path='det_collision_scan.png')
    
    # Run detailed merger case
    print("\n[3] Running detailed merger (overlap=10)...")
    merger = run_collision(overlap=10, steps=15000, 
                          snapshot_times=[0, 10, 50, 100, 200, 500, 750])
    
    # Plot merger
    print("\n[4] Plotting merger details...")
    plot_collision_result(merger, save_path='det_merger_detail.png')
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    plt.show()
