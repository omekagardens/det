import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import defaultdict
import time as clock_time

"""
DET v5 COLLIDER - RIGOROUS TWO-SUITE DESIGN
============================================

SUITE A: BINDING MAP (contact-prepared, offset≤18)
  - Bind/scatter/fragment fractions
  - Capture radius
  - Release times and peak counts
  - Control axis: v_kick (NOT v_rel)

SUITE B: CROSS-SECTION MAP (offset distribution)
  - σ_contact(g, v) ~ P(contact | g, v)
  - σ_bind(g, v) ~ P(bind | g, v)
  - σ_frag(g, v) ~ P(frag | g, v)
  - Uses offset ∈ [18, 40] distribution
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

    def step(self):
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


def classify_outcome(sim):
    """Classify collision outcome. Returns (outcome, metrics)."""
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
    
    # Persistence check
    n_frames = len(separation)
    persist_start = int(n_frames * (1 - PERSIST_FRAC))
    late_sep = separation[persist_start:]
    persistent_binding = np.all(late_sep < S_HOLD)
    made_contact = min_sep <= S_CONTACT
    
    # Classification
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


def run_trial(g, v_kick, offset, rng, theta_noise=3e-4, steps=800):
    """Run single trial."""
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick)
    if theta_noise > 0:
        sim.theta += rng.normal(0, theta_noise, size=sim.N)
        sim.theta = np.mod(sim.theta, 2*np.pi)
    for _ in range(steps):
        sim.step()
    return classify_outcome(sim)


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for proportion k/n."""
    if n == 0:
        return (0, 1)
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - spread), min(1, center + spread))


def main():
    print("="*70)
    print("DET v5 COLLIDER - RIGOROUS TWO-SUITE DESIGN")
    print("="*70)
    
    rng = np.random.default_rng(42)
    
    # =========================================================================
    # SUITE A: BINDING MAP (contact-prepared, offset=15)
    # =========================================================================
    print("\n" + "="*70)
    print("SUITE A: BINDING MAP (contact-prepared, offset=15)")
    print("="*70)
    print("Control axis: v_kick (NOT v_rel)")
    print("Reports: BOUND%, SCATTER%, FRAGMENT% (must sum to 100%)\n")
    
    n_trials_a = 25
    offset_a = 15
    
    # A1: g-scan at fixed v_kick
    print("--- A1: g-scan at v_kick=2.0 ---")
    print(f"{'g':<6} | {'BOUND%':<8} | {'SCATTER%':<10} | {'FRAG%':<8} | {'n':<4}")
    print("-" * 50)
    
    suite_a1 = []
    for g in np.arange(1.0, 2.6, 0.2):
        outcomes = defaultdict(int)
        for _ in range(n_trials_a):
            outcome, _ = run_trial(g, 2.0, offset_a, rng)
            if outcome:
                outcomes[outcome] += 1
        
        total = sum(outcomes.values())
        b = outcomes['BOUND'] / total if total > 0 else 0
        s = outcomes['SCATTER'] / total if total > 0 else 0
        f = outcomes['FRAGMENT'] / total if total > 0 else 0
        
        print(f"{g:<6.1f} | {100*b:<8.0f} | {100*s:<10.0f} | {100*f:<8.0f} | {total:<4}")
        suite_a1.append({'g': g, 'bound': b, 'scatter': s, 'frag': f, 'n': total})
    
    # A2: v_kick scan at fixed g
    print("\n--- A2: v_kick scan at g=1.5 ---")
    print(f"{'v_kick':<8} | {'BOUND%':<8} | {'SCATTER%':<10} | {'FRAG%':<8} | {'n':<4}")
    print("-" * 52)
    
    suite_a2 = []
    for v_kick in np.arange(1.6, 2.6, 0.1):
        outcomes = defaultdict(int)
        for _ in range(n_trials_a):
            outcome, _ = run_trial(1.5, v_kick, offset_a, rng)
            if outcome:
                outcomes[outcome] += 1
        
        total = sum(outcomes.values())
        b = outcomes['BOUND'] / total if total > 0 else 0
        s = outcomes['SCATTER'] / total if total > 0 else 0
        f = outcomes['FRAGMENT'] / total if total > 0 else 0
        
        print(f"{v_kick:<8.2f} | {100*b:<8.0f} | {100*s:<10.0f} | {100*f:<8.0f} | {total:<4}")
        suite_a2.append({'v_kick': v_kick, 'bound': b, 'scatter': s, 'frag': f, 'n': total})
    
    # =========================================================================
    # SUITE B: CROSS-SECTION MAP (offset distribution)
    # =========================================================================
    print("\n" + "="*70)
    print("SUITE B: CROSS-SECTION MAP (offset ∈ [18, 40])")
    print("="*70)
    print("σ_contact ~ P(contact), σ_bind ~ P(bind), σ_frag ~ P(frag)\n")
    
    n_trials_b = 15
    offsets_b = [18, 22, 26, 30, 34, 38]  # Sample of offsets
    steps_b = 1200  # Longer runs for free-flight
    
    # B1: Cross-section vs g at fixed v_kick
    print("--- B1: σ(g) at v_kick=2.5 ---")
    print(f"{'g':<6} | {'σ_contact':<10} | {'σ_bind':<10} | {'σ_frag':<10} | {'n_total':<8}")
    print("-" * 55)
    
    suite_b1 = []
    for g in [1.0, 1.5, 2.0, 2.5]:
        contact_count = 0
        bind_count = 0
        frag_count = 0
        total = 0
        
        for offset in offsets_b:
            for _ in range(n_trials_b):
                outcome, metrics = run_trial(g, 2.5, offset, rng, steps=steps_b)
                if outcome:
                    total += 1
                    if metrics['made_contact']:
                        contact_count += 1
                    if outcome == 'BOUND':
                        bind_count += 1
                    elif outcome == 'FRAGMENT':
                        frag_count += 1
        
        sigma_c = contact_count / total if total > 0 else 0
        sigma_b = bind_count / total if total > 0 else 0
        sigma_f = frag_count / total if total > 0 else 0
        
        print(f"{g:<6.1f} | {sigma_c:<10.2f} | {sigma_b:<10.2f} | {sigma_f:<10.2f} | {total:<8}")
        suite_b1.append({'g': g, 'sigma_c': sigma_c, 'sigma_b': sigma_b, 'sigma_f': sigma_f, 'n': total})
    
    # B2: Cross-section vs v_kick at fixed g
    print("\n--- B2: σ(v_kick) at g=1.5 ---")
    print(f"{'v_kick':<8} | {'σ_contact':<10} | {'σ_bind':<10} | {'σ_frag':<10} | {'n_total':<8}")
    print("-" * 57)
    
    suite_b2 = []
    for v_kick in [1.5, 2.0, 2.5, 3.0]:
        contact_count = 0
        bind_count = 0
        frag_count = 0
        total = 0
        
        for offset in offsets_b:
            for _ in range(n_trials_b):
                outcome, metrics = run_trial(1.5, v_kick, offset, rng, steps=steps_b)
                if outcome:
                    total += 1
                    if metrics['made_contact']:
                        contact_count += 1
                    if outcome == 'BOUND':
                        bind_count += 1
                    elif outcome == 'FRAGMENT':
                        frag_count += 1
        
        sigma_c = contact_count / total if total > 0 else 0
        sigma_b = bind_count / total if total > 0 else 0
        sigma_f = frag_count / total if total > 0 else 0
        
        print(f"{v_kick:<8.1f} | {sigma_c:<10.2f} | {sigma_b:<10.2f} | {sigma_f:<10.2f} | {total:<8}")
        suite_b2.append({'v_kick': v_kick, 'sigma_c': sigma_c, 'sigma_b': sigma_b, 'sigma_f': sigma_f, 'n': total})
    
    # B3: Contact rate vs offset (capture radius measurement)
    print("\n--- B3: Contact rate vs offset (g=1.5, v=2.5) ---")
    print(f"{'offset':<8} | {'sep0':<6} | {'P_contact':<12} | {'n':<4}")
    print("-" * 40)
    
    suite_b3 = []
    for offset in [15, 18, 20, 22, 25, 30, 35, 40]:
        contact_count = 0
        total = 0
        
        for _ in range(n_trials_b):
            outcome, metrics = run_trial(1.5, 2.5, offset, rng, steps=steps_b)
            if outcome:
                total += 1
                if metrics['made_contact']:
                    contact_count += 1
        
        p_contact = contact_count / total if total > 0 else 0
        ci = wilson_ci(contact_count, total)
        
        print(f"{offset:<8} | {2*offset:<6} | {p_contact:<12.2f} | {total:<4}")
        suite_b3.append({'offset': offset, 'sep0': 2*offset, 'p_contact': p_contact, 'ci': ci, 'n': total})
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # A1: Binding map g-scan (stacked bar)
    ax = axes[0, 0]
    g_vals = [r['g'] for r in suite_a1]
    bounds = [r['bound'] for r in suite_a1]
    scatters = [r['scatter'] for r in suite_a1]
    frags = [r['frag'] for r in suite_a1]
    
    width = 0.15
    x = np.arange(len(g_vals))
    ax.bar(x, bounds, width, label='BOUND', color='green', alpha=0.8)
    ax.bar(x, scatters, width, bottom=bounds, label='SCATTER', color='blue', alpha=0.8)
    ax.bar(x, frags, width, bottom=np.array(bounds)+np.array(scatters), label='FRAGMENT', color='red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{g:.1f}' for g in g_vals])
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Fraction")
    ax.set_title("Suite A1: Binding Map (v_kick=2.0, offset=15)")
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    
    # A2: v_kick scan (stacked bar)
    ax = axes[0, 1]
    v_vals = [r['v_kick'] for r in suite_a2]
    bounds2 = [r['bound'] for r in suite_a2]
    scatters2 = [r['scatter'] for r in suite_a2]
    frags2 = [r['frag'] for r in suite_a2]
    
    x2 = np.arange(len(v_vals))
    ax.bar(x2, bounds2, width, label='BOUND', color='green', alpha=0.8)
    ax.bar(x2, scatters2, width, bottom=bounds2, label='SCATTER', color='blue', alpha=0.8)
    ax.bar(x2, frags2, width, bottom=np.array(bounds2)+np.array(scatters2), label='FRAGMENT', color='red', alpha=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels([f'{v:.1f}' for v in v_vals])
    ax.set_xlabel("v_kick")
    ax.set_ylabel("Fraction")
    ax.set_title("Suite A2: Binding Map (g=1.5, offset=15)")
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    
    # B1: Cross-section vs g
    ax = axes[0, 2]
    g_b1 = [r['g'] for r in suite_b1]
    sigma_c = [r['sigma_c'] for r in suite_b1]
    sigma_b = [r['sigma_b'] for r in suite_b1]
    sigma_f = [r['sigma_f'] for r in suite_b1]
    
    ax.plot(g_b1, sigma_c, 'ko-', markersize=10, linewidth=2, label='σ_contact')
    ax.plot(g_b1, sigma_b, 'g^--', markersize=8, linewidth=2, label='σ_bind')
    ax.plot(g_b1, sigma_f, 'rs:', markersize=8, linewidth=2, label='σ_frag')
    ax.set_xlabel("Coupling g")
    ax.set_ylabel("Cross-section proxy")
    ax.set_title("Suite B1: σ(g) at v_kick=2.5")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # B2: Cross-section vs v_kick
    ax = axes[1, 0]
    v_b2 = [r['v_kick'] for r in suite_b2]
    sigma_c2 = [r['sigma_c'] for r in suite_b2]
    sigma_b2 = [r['sigma_b'] for r in suite_b2]
    sigma_f2 = [r['sigma_f'] for r in suite_b2]
    
    ax.plot(v_b2, sigma_c2, 'ko-', markersize=10, linewidth=2, label='σ_contact')
    ax.plot(v_b2, sigma_b2, 'g^--', markersize=8, linewidth=2, label='σ_bind')
    ax.plot(v_b2, sigma_f2, 'rs:', markersize=8, linewidth=2, label='σ_frag')
    ax.set_xlabel("v_kick")
    ax.set_ylabel("Cross-section proxy")
    ax.set_title("Suite B2: σ(v_kick) at g=1.5")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # B3: Contact rate vs offset (capture radius)
    ax = axes[1, 1]
    offsets = [r['offset'] for r in suite_b3]
    p_contacts = [r['p_contact'] for r in suite_b3]
    
    ax.plot(offsets, p_contacts, 'ko-', markersize=10, linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax.fill_between(offsets, 0, p_contacts, alpha=0.2)
    ax.set_xlabel("Offset (sep₀ = 2×offset)")
    ax.set_ylabel("P(contact)")
    ax.set_title("Suite B3: Capture Radius (g=1.5, v=2.5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = """
RIGOROUS SUMMARY

Suite A (Contact-Prepared):
• Regime fractions at fixed offset=15
• Control: v_kick (not v_rel)
• Valid for: phase boundaries

Suite B (Offset Distribution):
• Cross-section proxies σ(g,v)
• Uses offset ∈ [18, 40]
• Valid for: transport properties

Falsifiable Claims:
1. σ_contact decreases with g
2. σ_frag increases with g
3. Capture radius ~ 20-25 offset
"""
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('/home/claude/det_v5_rigorous.png', dpi=150)
    plt.savefig('/mnt/user-data/outputs/det_v5_rigorous.png', dpi=150)
    print("Saved to det_v5_rigorous.png")
    
    return suite_a1, suite_a2, suite_b1, suite_b2, suite_b3


if __name__ == "__main__":
    t0 = clock_time.time()
    a1, a2, b1, b2, b3 = main()
    elapsed = clock_time.time() - t0
    
    print("\n" + "="*70)
    print(f"COMPLETED in {elapsed:.1f}s")
    print("="*70)
    
    # Extract capture radius
    print("\nCapture radius estimate:")
    for r in b3:
        if r['p_contact'] < 0.5:
            print(f"  P(contact) drops below 50% at offset ≈ {r['offset']} (sep₀ = {r['sep0']})")
            break
    
    # Fragmentation onset
    print("\nFragmentation onset:")
    for r in a1:
        if r['frag'] > 0.5:
            print(f"  FRAGMENT > 50% at g ≈ {r['g']}")
            break
