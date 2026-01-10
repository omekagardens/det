import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import defaultdict
import time as clock_time

"""
DET v5 COLLIDER - POST-CONTACT DYNAMICS
=======================================

Key question: Why is σ_bind ≈ 0 even when σ_contact ≈ 0.5?

This experiment:
1. Uses offset=22 (guaranteed contact)
2. Tracks collision dynamics in detail
3. Measures energy/mass budget
4. Identifies what determines BIND vs SCATTER vs FRAGMENT

Hypotheses:
H1: Binding requires low collision energy (v_kick)
H2: Binding requires specific g range  
H3: Fragmentation is caused by metric instability
H4: Energy is lost to radiation/excitations
"""

S_CONTACT = 2.0
S_HOLD = 8.0
PERSIST_FRAC = 0.25
FRAG_THRESHOLD = 3

class DETConstants:
    def __init__(self, g=1.0):
        self.g = g
        self.BETA = 2.0 * g
        self.NU = 0.5
        self.PHASE_DRAG = 0.2
        self.ALPHA = 0.5 * g
        self.GAMMA = 8.0
        self.LAMBDA = 0.005
        self.K_FUSION = 80.0
        self.F_VAC = 0.01  
        self.C_MIN = 0.05 
        self.DT = 0.05

class ActiveManifold1D:
    def __init__(self, size=200, constants=DETConstants()):
        self.N = size
        self.k = constants
        self.F = np.ones(self.N) * self.k.F_VAC
        self.theta = np.zeros(self.N)
        self.sigma = np.ones(self.N)
        self.C_right = np.ones(self.N) * self.k.C_MIN
        self.C_left  = np.ones(self.N) * self.k.C_MIN
        self.time = 0.0
        self.track_L = []
        self.track_R = []
        self.last_pos_L = None
        self.last_pos_R = None
        
        # History for analysis
        self.F_history = []
        self.C_history = []
        self.peak_history = []

    def initialize_soliton(self, position, width, mass, velocity_kick=0.0):
        x = np.arange(self.N)
        dx = x - position
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        envelope = np.exp(-0.5 * (np.abs(dx) / width)**2)
        self.F += mass * envelope
        self.theta += velocity_kick * dx * envelope
        if velocity_kick > 0:
            self.C_right += 0.9 * envelope
            self.C_left  += 0.3 * envelope
        elif velocity_kick < 0:
            self.C_right += 0.3 * envelope
            self.C_left  += 0.9 * envelope
        else:
            self.C_right += 0.6 * envelope
            self.C_left  += 0.6 * envelope
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)

    def initialize_collider_event(self, offset=15, width=5.0, mass_L=10.0, mass_R=10.0, v_kick=2.0):
        center = self.N // 2
        self.start_pos_L = center - offset
        self.start_pos_R = center + offset
        self.mass_L = mass_L
        self.mass_R = mass_R
        self.initialize_soliton(self.start_pos_L, width, mass_L, velocity_kick=v_kick)
        self.initialize_soliton(self.start_pos_R, width, mass_R, velocity_kick=-v_kick)
        self.last_pos_L = float(self.start_pos_L)
        self.last_pos_R = float(self.start_pos_R)
        self.track_L = [(0.0, self.last_pos_L)]
        self.track_R = [(0.0, self.last_pos_R)]

    def find_blob_coms(self):
        mass_dist = np.maximum(self.F - self.k.F_VAC, 0)
        threshold = np.max(mass_dist) * 0.15
        binary = (mass_dist > threshold).astype(int)
        labeled, n_features = label(binary)
        blobs = []
        for i in range(1, n_features + 1):
            mask = (labeled == i)
            blob_mass = np.sum(mass_dist[mask])
            if blob_mass > 0.5:
                indices = np.where(mask)[0]
                masses = mass_dist[mask]
                com = np.sum(indices * masses) / np.sum(masses)
                blobs.append((com, blob_mass))
        return blobs

    def count_peaks(self):
        return len(self.find_blob_coms())

    def total_mass(self):
        return np.sum(self.F - self.k.F_VAC)
    
    def total_metric(self):
        return np.sum(self.C_right + self.C_left - 2*self.k.C_MIN)

    def unwrap_position(self, new_pos, last_pos):
        if last_pos is None:
            return new_pos
        delta = new_pos - (last_pos % self.N)
        if delta > self.N / 2:
            delta -= self.N
        elif delta < -self.N / 2:
            delta += self.N
        return last_pos + delta

    def update_tracking(self):
        blobs = self.find_blob_coms()
        if len(blobs) == 0:
            return
        def ring_dist(a, b):
            d = abs((a % self.N) - (b % self.N))
            return min(d, self.N - d)
        if len(blobs) == 1:
            com = blobs[0][0]
            self.last_pos_L = self.unwrap_position(com, self.last_pos_L)
            self.last_pos_R = self.unwrap_position(com, self.last_pos_R)
            self.track_L.append((self.time, self.last_pos_L))
            self.track_R.append((self.time, self.last_pos_R))
            return
        best_L_idx = min(range(len(blobs)), key=lambda i: ring_dist(blobs[i][0], self.last_pos_L % self.N))
        remaining = [i for i in range(len(blobs)) if i != best_L_idx]
        best_R_idx = min(remaining, key=lambda i: ring_dist(blobs[i][0], self.last_pos_R % self.N)) if remaining else best_L_idx
        self.last_pos_L = self.unwrap_position(blobs[best_L_idx][0], self.last_pos_L)
        self.last_pos_R = self.unwrap_position(blobs[best_R_idx][0], self.last_pos_R)
        self.track_L.append((self.time, self.last_pos_L))
        self.track_R.append((self.time, self.last_pos_R))

    def step(self, record_history=False):
        idx = np.arange(self.N)
        ip1 = (idx + 1) % self.N
        im1 = (idx - 1) % self.N
        d_fwd = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[idx])))
        d_bwd = np.angle(np.exp(1j * (self.theta[im1] - self.theta[idx])))
        laplacian_theta = d_fwd + d_bwd
        grad_theta = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[im1]))) / 2.0
        phase_drive = np.minimum(self.k.BETA * self.F * self.k.DT, np.pi/4.0)
        drag_factor = 1.0 / (1.0 + self.k.PHASE_DRAG * (grad_theta**2))
        self.theta -= phase_drive * drag_factor
        self.theta += self.k.NU * laplacian_theta * self.k.DT
        self.theta = np.mod(self.theta, 2*np.pi)
        recip_R = np.sqrt(self.C_left[ip1])
        recip_L = np.sqrt(self.C_right[im1])
        drive_R = recip_R * np.sin(self.theta[ip1]-self.theta[idx]) + (1-recip_R)*(self.F[idx]-self.F[ip1])
        drive_L = recip_L * np.sin(self.theta[im1]-self.theta[idx]) + (1-recip_L)*(self.F[idx]-self.F[im1])
        cond_R = self.sigma[idx] * (self.C_right[idx]**2 + 1e-4)
        cond_L = self.sigma[idx] * (self.C_left[idx]**2 + 1e-4)
        J_right = cond_R * drive_R
        J_left  = cond_L * drive_L
        max_allowed = 0.40 * self.F / self.k.DT
        total_out = np.abs(J_right) + np.abs(J_left)
        scale = np.minimum(1.0, max_allowed / (total_out + 1e-9))
        J_right *= scale
        J_left *= scale
        dF = (J_right[im1] + J_left[ip1]) - (J_right + J_left)
        self.F += dF * self.k.DT
        self.F = np.maximum(self.F, self.k.F_VAC)
        compression = np.maximum(0, dF)
        alpha = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        lam = self.k.LAMBDA / (1.0 + self.k.K_FUSION * compression * 10.0)
        J_R_align = np.maximum(0, J_right) + self.k.GAMMA * np.maximum(0, np.roll(J_right, 1))
        J_L_align = np.maximum(0, J_left) + self.k.GAMMA * np.maximum(0, np.roll(J_left, -1))
        self.C_right += (alpha * J_R_align - lam * self.C_right) * self.k.DT
        self.C_left  += (alpha * J_L_align - lam * self.C_left) * self.k.DT
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_right) + np.abs(J_left))
        self.time += self.k.DT
        self.update_tracking()
        
        if record_history:
            self.F_history.append(self.total_mass())
            self.C_history.append(self.total_metric())
            self.peak_history.append(self.count_peaks())


def classify_outcome(sim):
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)
    if len(track_L) < 20:
        return None, {}
    
    times = track_L[:, 0]
    pos_L = track_L[:, 1]
    pos_R = track_R[:, 1]
    separation = np.abs(pos_L - pos_R)
    min_sep = np.min(separation)
    final_peaks = sim.count_peaks()
    
    n_frames = len(separation)
    persist_start = int(n_frames * (1 - PERSIST_FRAC))
    late_sep = separation[persist_start:]
    persistent_binding = np.all(late_sep < S_HOLD)
    made_contact = min_sep <= S_CONTACT
    
    if made_contact and final_peaks == 1 and persistent_binding:
        outcome = 'BOUND'
    elif final_peaks >= FRAG_THRESHOLD:
        outcome = 'FRAGMENT'
    else:
        outcome = 'SCATTER'
    
    return outcome, {
        'min_sep': min_sep,
        'made_contact': made_contact,
        'final_peaks': final_peaks,
        'final_sep': separation[-1],
    }


def run_detailed_trial(g, v_kick, offset=22, steps=800):
    """Run trial with detailed history tracking."""
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick)
    
    # Record initial state
    M0 = sim.total_mass()
    C0 = sim.total_metric()
    
    for _ in range(steps):
        sim.step(record_history=True)
    
    outcome, metrics = classify_outcome(sim)
    
    # Compute conservation metrics
    M_final = sim.total_mass()
    C_final = sim.total_metric()
    
    metrics['M0'] = M0
    metrics['M_final'] = M_final
    metrics['dM'] = (M_final - M0) / M0 * 100  # percent change
    metrics['C0'] = C0
    metrics['C_final'] = C_final
    metrics['dC'] = (C_final - C0) / (C0 + 1e-9) * 100
    metrics['F_history'] = sim.F_history
    metrics['C_history'] = sim.C_history
    metrics['peak_history'] = sim.peak_history
    
    return outcome, metrics, sim


def main():
    print("="*70)
    print("DET v5 POST-CONTACT DYNAMICS ANALYSIS")
    print("="*70)
    print("""
Question: Why is σ_bind ≈ 0 even when σ_contact ≈ 0.5?

Experiment design:
- Use offset=22 (100% contact guaranteed)
- Track mass and metric conservation
- Analyze what determines outcome after contact
""")
    
    # =========================================================================
    # EXPERIMENT 1: Outcome vs v_kick at fixed g (with conservation tracking)
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: v_kick scan with conservation tracking")
    print("="*70)
    print("g=1.5, offset=22 (guaranteed contact)\n")
    
    print(f"{'v_kick':<8} | {'outcome':<10} | {'peaks':<6} | {'ΔM%':<8} | {'ΔC%':<8}")
    print("-" * 50)
    
    exp1_results = []
    for v_kick in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]:
        outcome, metrics, sim = run_detailed_trial(g=1.5, v_kick=v_kick, offset=22)
        
        print(f"{v_kick:<8.1f} | {outcome:<10} | {metrics['final_peaks']:<6} | "
              f"{metrics['dM']:<8.2f} | {metrics['dC']:<8.1f}")
        
        exp1_results.append({
            'v_kick': v_kick,
            'outcome': outcome,
            'peaks': metrics['final_peaks'],
            'dM': metrics['dM'],
            'dC': metrics['dC'],
            'F_history': metrics['F_history'],
            'C_history': metrics['C_history'],
            'peak_history': metrics['peak_history'],
        })
    
    # =========================================================================
    # EXPERIMENT 2: g scan with conservation tracking
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: g scan with conservation tracking")
    print("="*70)
    print("v_kick=2.0, offset=22 (guaranteed contact)\n")
    
    print(f"{'g':<6} | {'outcome':<10} | {'peaks':<6} | {'ΔM%':<8} | {'ΔC%':<8}")
    print("-" * 48)
    
    exp2_results = []
    for g in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]:
        outcome, metrics, sim = run_detailed_trial(g=g, v_kick=2.0, offset=22)
        
        print(f"{g:<6.1f} | {outcome:<10} | {metrics['final_peaks']:<6} | "
              f"{metrics['dM']:<8.2f} | {metrics['dC']:<8.1f}")
        
        exp2_results.append({
            'g': g,
            'outcome': outcome,
            'peaks': metrics['final_peaks'],
            'dM': metrics['dM'],
            'dC': metrics['dC'],
        })
    
    # =========================================================================
    # EXPERIMENT 3: Detailed time evolution for each outcome type
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: Time evolution by outcome type")
    print("="*70)
    
    # Find one example of each outcome type
    examples = {}
    test_params = [
        (1.5, 1.6),  # Try for BOUND
        (1.5, 2.0),  # Likely SCATTER
        (1.0, 2.0),  # Likely FRAGMENT
    ]
    
    for g, v_kick in test_params:
        outcome, metrics, sim = run_detailed_trial(g=g, v_kick=v_kick, offset=22, steps=1000)
        if outcome not in examples:
            examples[outcome] = {
                'g': g, 'v_kick': v_kick,
                'F_history': metrics['F_history'],
                'C_history': metrics['C_history'],
                'peak_history': metrics['peak_history'],
            }
            print(f"Found {outcome} example at g={g}, v_kick={v_kick}")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Outcome vs v_kick
    ax = axes[0, 0]
    v_kicks = [r['v_kick'] for r in exp1_results]
    outcomes = [r['outcome'] for r in exp1_results]
    peaks = [r['peaks'] for r in exp1_results]
    
    colors = {'BOUND': 'green', 'SCATTER': 'blue', 'FRAGMENT': 'red'}
    for i, (v, out) in enumerate(zip(v_kicks, outcomes)):
        ax.bar(i, 1, color=colors.get(out, 'gray'), alpha=0.8)
    ax.set_xticks(range(len(v_kicks)))
    ax.set_xticklabels([f'{v:.1f}' for v in v_kicks])
    ax.set_xlabel("v_kick")
    ax.set_ylabel("Outcome")
    ax.set_title("Exp 1: Outcome vs v_kick (g=1.5)")
    
    # Add peak count on secondary axis
    ax2 = ax.twinx()
    ax2.plot(range(len(peaks)), peaks, 'ko-', markersize=8)
    ax2.set_ylabel("Final peaks", color='black')
    
    # 2. Conservation: ΔM vs v_kick
    ax = axes[0, 1]
    dMs = [r['dM'] for r in exp1_results]
    dCs = [r['dC'] for r in exp1_results]
    
    ax.bar(np.arange(len(v_kicks))-0.15, dMs, width=0.3, label='ΔM%', color='blue', alpha=0.7)
    ax.bar(np.arange(len(v_kicks))+0.15, dCs, width=0.3, label='ΔC%', color='orange', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(v_kicks)))
    ax.set_xticklabels([f'{v:.1f}' for v in v_kicks])
    ax.set_xlabel("v_kick")
    ax.set_ylabel("% change")
    ax.set_title("Conservation vs v_kick")
    ax.legend()
    
    # 3. Outcome vs g
    ax = axes[0, 2]
    g_vals = [r['g'] for r in exp2_results]
    outcomes_g = [r['outcome'] for r in exp2_results]
    peaks_g = [r['peaks'] for r in exp2_results]
    
    for i, (g, out) in enumerate(zip(g_vals, outcomes_g)):
        ax.bar(i, 1, color=colors.get(out, 'gray'), alpha=0.8)
    ax.set_xticks(range(len(g_vals)))
    ax.set_xticklabels([f'{g:.1f}' for g in g_vals])
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Outcome")
    ax.set_title("Exp 2: Outcome vs g (v_kick=2.0)")
    
    ax2 = ax.twinx()
    ax2.plot(range(len(peaks_g)), peaks_g, 'ko-', markersize=8)
    ax2.set_ylabel("Final peaks", color='black')
    
    # 4-6: Time evolution for each outcome type
    for idx, (outcome, label_pos) in enumerate([('BOUND', (1,0)), ('SCATTER', (1,1)), ('FRAGMENT', (1,2))]):
        ax = axes[label_pos]
        
        if outcome in examples:
            ex = examples[outcome]
            times = np.arange(len(ex['F_history'])) * 0.05
            
            ax.plot(times, ex['F_history'], 'b-', linewidth=2, label='Total M')
            ax.set_xlabel("Time")
            ax.set_ylabel("Total Mass", color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            
            ax2 = ax.twinx()
            ax2.plot(times, ex['peak_history'], 'r--', linewidth=2, label='Peaks')
            ax2.set_ylabel("Peak Count", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title(f"{outcome} (g={ex['g']}, v={ex['v_kick']})")
        else:
            ax.text(0.5, 0.5, f"No {outcome} found", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{outcome} (not found)")
    
    plt.tight_layout()
    plt.savefig('./det_v5_postcontact.png', dpi=150)
    print("Saved to det_v5_postcontact.png")
    
    return exp1_results, exp2_results, examples


if __name__ == "__main__":
    t0 = clock_time.time()
    exp1, exp2, examples = main()
    elapsed = clock_time.time() - t0
    
    print("\n" + "="*70)
    print(f"ANALYSIS (completed in {elapsed:.1f}s)")
    print("="*70)
    
    # Count outcomes
    print("\nOutcome distribution (Exp 1, g=1.5):")
    from collections import Counter
    counts1 = Counter([r['outcome'] for r in exp1])
    for out, count in counts1.items():
        print(f"  {out}: {count}")
    
    print("\nOutcome distribution (Exp 2, v_kick=2.0):")
    counts2 = Counter([r['outcome'] for r in exp2])
    for out, count in counts2.items():
        print(f"  {out}: {count}")
    
    # Check mass conservation
    print("\nMass conservation:")
    dMs = [r['dM'] for r in exp1]
    print(f"  Mean ΔM%: {np.mean(dMs):.2f}")
    print(f"  Max |ΔM%|: {np.max(np.abs(dMs)):.2f}")
    
    # Identify binding window
    print("\nBinding window (v_kick scan):")
    for r in exp1:
        if r['outcome'] == 'BOUND':
            print(f"  BOUND at v_kick = {r['v_kick']}")
