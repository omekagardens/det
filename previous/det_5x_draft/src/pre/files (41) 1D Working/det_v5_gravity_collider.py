import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, maximum_filter, minimum_filter
from scipy.fft import fft, ifft, fftfreq
import time as clock_time

"""
DET v5 COLLIDER WITH WORKING GRAVITY
====================================

Integrates the working 1D gravity (flow bias method, correct sign)
into the collider system.

Key questions:
1. Does gravity extend capture radius?
2. Does gravity help binding (reduce fragmentation)?
3. Can gravity stabilize bound states?
"""

S_CONTACT = 2.0
PERSIST_FRAC = 0.25
FRAG_THRESHOLD = 3


class DETGravity1D:
    """Working 1D gravity with CORRECT sign convention."""
    
    def __init__(self, N, kappa=0.02):
        self.N = N
        self.kappa = kappa
        k = fftfreq(N, d=1.0) * 2 * np.pi
        self.k_squared = k**2
        self.k_squared[0] = 1.0
    
    def compute_gravity(self, F):
        """Compute gravity field from mass distribution F."""
        rho = F / (np.sum(F) + 1e-9) * self.N  # Normalized density
        rho_hat = fft(rho)
        # CORRECT sign for attraction: +kappa (not -kappa)
        phi_hat = +self.kappa * rho_hat / self.k_squared
        phi_hat[0] = 0.0  # Zero mean
        Phi = np.real(ifft(phi_hat))
        g = -0.5 * (np.roll(Phi, -1) - np.roll(Phi, 1))  # g = -∇Φ
        return Phi, g


class DETConstants:
    def __init__(self, g=1.0, kappa=0.0):
        # Coupling parameter
        self.g = g
        
        # Phase dynamics
        self.BETA = 2.0 * g
        self.NU = 0.5
        self.PHASE_DRAG = 0.2
        
        # Metric dynamics
        self.ALPHA = 0.5 * g
        self.GAMMA = 8.0
        self.LAMBDA = 0.005
        self.K_FUSION = 80.0
        
        # Floors
        self.F_VAC = 0.01  
        self.C_MIN = 0.05 
        
        # Gravity (NEW)
        self.KAPPA = kappa  # 0 = no gravity, 0.02 = working value
        
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
        
        # Gravity
        self.gravity = DETGravity1D(self.N, self.k.KAPPA) if self.k.KAPPA > 0 else None
        self.Phi = np.zeros(self.N)
        self.g_field = np.zeros(self.N)
        
        self.time = 0.0
        self.peak_history = []
        self.sep_history = []
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
        self.last_pos_L = float(self.start_pos_L)
        self.last_pos_R = float(self.start_pos_R)
        self.initialize_soliton(self.start_pos_L, width, mass_L, velocity_kick=v_kick)
        self.initialize_soliton(self.start_pos_R, width, mass_R, velocity_kick=-v_kick)

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
        n_peaks = len(blobs)
        self.peak_history.append(n_peaks)
        
        if n_peaks == 0:
            self.sep_history.append(0)
            return
        
        def ring_dist(a, b):
            d = abs((a % self.N) - (b % self.N))
            return min(d, self.N - d)
        
        if n_peaks == 1:
            com = blobs[0][0]
            self.last_pos_L = self.unwrap_position(com, self.last_pos_L)
            self.last_pos_R = self.unwrap_position(com, self.last_pos_R)
            self.sep_history.append(0)
            return
        
        best_L_idx = min(range(len(blobs)), key=lambda i: ring_dist(blobs[i][0], self.last_pos_L % self.N))
        remaining = [i for i in range(len(blobs)) if i != best_L_idx]
        best_R_idx = min(remaining, key=lambda i: ring_dist(blobs[i][0], self.last_pos_R % self.N)) if remaining else best_L_idx
        
        self.last_pos_L = self.unwrap_position(blobs[best_L_idx][0], self.last_pos_L)
        self.last_pos_R = self.unwrap_position(blobs[best_R_idx][0], self.last_pos_R)
        self.sep_history.append(abs(self.last_pos_L - self.last_pos_R))

    def step(self):
        dt = self.k.DT
        idx = np.arange(self.N)
        ip1 = (idx + 1) % self.N
        im1 = (idx - 1) % self.N
        
        # === GRAVITY (if enabled) ===
        if self.gravity is not None:
            self.Phi, self.g_field = self.gravity.compute_gravity(self.F)
        
        # === PHASE DYNAMICS ===
        d_fwd = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[idx])))
        d_bwd = np.angle(np.exp(1j * (self.theta[im1] - self.theta[idx])))
        laplacian_theta = d_fwd + d_bwd
        grad_theta = np.angle(np.exp(1j * (self.theta[ip1] - self.theta[im1]))) / 2.0
        phase_drive = np.minimum(self.k.BETA * self.F * dt, np.pi/4.0)
        drag_factor = 1.0 / (1.0 + self.k.PHASE_DRAG * (grad_theta**2))
        self.theta -= phase_drive * drag_factor
        self.theta += self.k.NU * laplacian_theta * dt
        self.theta = np.mod(self.theta, 2*np.pi)
        
        # === FLOW DYNAMICS (with gravity bias) ===
        recip_R = np.sqrt(self.C_left[ip1])
        recip_L = np.sqrt(self.C_right[im1])
        drive_R = recip_R * np.sin(self.theta[ip1]-self.theta[idx]) + (1-recip_R)*(self.F[idx]-self.F[ip1])
        drive_L = recip_L * np.sin(self.theta[im1]-self.theta[idx]) + (1-recip_L)*(self.F[idx]-self.F[im1])
        cond_R = self.sigma[idx] * (self.C_right[idx]**2 + 1e-4)
        cond_L = self.sigma[idx] * (self.C_left[idx]**2 + 1e-4)
        
        J_right = cond_R * drive_R
        J_left  = cond_L * drive_L
        
        # Add gravity bias to flow (flow bias method)
        if self.gravity is not None:
            J_right += self.F * self.g_field
            J_left  -= self.F * self.g_field  # Opposite direction
        
        max_allowed = 0.40 * self.F / dt
        total_out = np.abs(J_right) + np.abs(J_left)
        scale = np.minimum(1.0, max_allowed / (total_out + 1e-9))
        J_right *= scale
        J_left *= scale
        
        dF = (J_right[im1] + J_left[ip1]) - (J_right + J_left)
        self.F += dF * dt
        self.F = np.maximum(self.F, self.k.F_VAC)
        
        # === METRIC UPDATE ===
        compression = np.maximum(0, dF)
        alpha = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        lam = self.k.LAMBDA / (1.0 + self.k.K_FUSION * compression * 10.0)
        J_R_align = np.maximum(0, J_right) + self.k.GAMMA * np.maximum(0, np.roll(J_right, 1))
        J_L_align = np.maximum(0, J_left) + self.k.GAMMA * np.maximum(0, np.roll(J_left, -1))
        self.C_right += (alpha * J_R_align - lam * self.C_right) * dt
        self.C_left  += (alpha * J_L_align - lam * self.C_left) * dt
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_right) + np.abs(J_left))
        
        self.time += dt
        self.update_tracking()


def corrected_analysis(sim):
    """Corrected classifier with TRANSIENT category."""
    peak_history = np.array(sim.peak_history)
    sep_history = np.array(sim.sep_history)
    
    if len(peak_history) < 20:
        return None
    
    min_sep = np.min(sep_history) if len(sep_history) > 0 else 999
    made_contact = min_sep <= S_CONTACT
    
    n_frames = len(peak_history)
    persist_start = int(n_frames * (1 - PERSIST_FRAC))
    late_peaks = peak_history[persist_start:]
    peak_persistent = np.all(late_peaks == 1)
    
    final_peaks = sim.count_peaks()
    
    if made_contact and final_peaks == 1 and peak_persistent:
        outcome = 'BOUND'
    elif final_peaks >= FRAG_THRESHOLD:
        outcome = 'FRAGMENT'
    elif final_peaks == 2:
        outcome = 'SCATTER'
    else:
        outcome = 'TRANSIENT'
    
    return {
        'outcome': outcome,
        'min_sep': min_sep,
        'made_contact': made_contact,
        'final_peaks': final_peaks,
        'peak_persistent': peak_persistent,
    }


def run_trial(g, v_kick, offset, kappa=0.0, steps=800):
    """Run single trial with optional gravity."""
    k = DETConstants(g=g, kappa=kappa)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick)
    
    sep_trace = []
    for i in range(steps):
        sim.step()
        if i % 10 == 0 and len(sim.sep_history) > 0:
            sep_trace.append(sim.sep_history[-1])
    
    return corrected_analysis(sim), sim, sep_trace


def main():
    print("="*70)
    print("DET v5 COLLIDER WITH GRAVITY")
    print("="*70)
    print("""
Testing the effect of DET 4.2 gravity (KAPPA=0.02) on collider outcomes.

Questions:
1. Does gravity extend capture radius?
2. Does gravity help binding at g=1.5?
3. Does gravity reduce fragmentation?
""")
    
    # =========================================================================
    # TEST 1: Compare with/without gravity at g=1.5
    # =========================================================================
    print("="*70)
    print("TEST 1: Gravity effect at g=1.5, v=2.0 (various offsets)")
    print("="*70)
    print(f"{'offset':<8}|{'KAPPA':<8}|{'outcome':<10}|{'peaks':<6}|{'min_sep':<10}|{'contact'}")
    print("-" * 60)
    
    test1_results = []
    for offset in [15, 18, 20, 22, 25]:
        for kappa in [0.0, 0.02]:
            res, sim, sep = run_trial(g=1.5, v_kick=2.0, offset=offset, kappa=kappa)
            print(f"{offset:<8}|{kappa:<8.2f}|{res['outcome']:<10}|{res['final_peaks']:<6}|"
                  f"{res['min_sep']:<10.1f}|{res['made_contact']}")
            test1_results.append({'offset': offset, 'kappa': kappa, **res})
    
    # =========================================================================
    # TEST 2: Capture radius comparison
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 2: Capture radius (g=1.5, v=2.0)")
    print("="*70)
    
    for kappa in [0.0, 0.02, 0.05]:
        max_contact = 0
        for offset in range(15, 40):
            res, _, _ = run_trial(g=1.5, v_kick=2.0, offset=offset, kappa=kappa, steps=1000)
            if res and res['made_contact']:
                max_contact = offset
        print(f"KAPPA={kappa:.2f}: max contact offset = {max_contact}")
    
    # =========================================================================
    # TEST 3: Does gravity help binding at g=0.8?
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 3: Gravity effect at g=0.8 (binding regime)")
    print("="*70)
    print(f"{'offset':<8}|{'KAPPA':<8}|{'outcome':<10}|{'peaks':<6}|{'min_sep':<8}")
    print("-" * 50)
    
    test3_results = []
    for offset in [18, 20, 22, 25]:
        for kappa in [0.0, 0.02]:
            res, sim, sep = run_trial(g=0.8, v_kick=2.0, offset=offset, kappa=kappa)
            print(f"{offset:<8}|{kappa:<8.2f}|{res['outcome']:<10}|{res['final_peaks']:<6}|"
                  f"{res['min_sep']:<8.1f}")
            test3_results.append({'offset': offset, 'kappa': kappa, **res})
    
    # =========================================================================
    # TEST 4: KAPPA sweep at fixed geometry
    # =========================================================================
    print("\n" + "="*70)
    print("TEST 4: KAPPA sweep (g=1.5, v=2.0, offset=20)")
    print("="*70)
    print(f"{'KAPPA':<10}|{'outcome':<10}|{'peaks':<6}|{'min_sep':<10}|{'contact'}")
    print("-" * 50)
    
    test4_results = []
    for kappa in [0.0, 0.01, 0.02, 0.05, 0.1]:
        res, sim, sep = run_trial(g=1.5, v_kick=2.0, offset=20, kappa=kappa, steps=1200)
        print(f"{kappa:<10.2f}|{res['outcome']:<10}|{res['final_peaks']:<6}|"
              f"{res['min_sep']:<10.1f}|{res['made_contact']}")
        test4_results.append({'kappa': kappa, 'sep_trace': sep, **res})
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {'BOUND': 'green', 'SCATTER': 'blue', 'FRAGMENT': 'red', 'TRANSIENT': 'orange'}
    
    # 1. Test 1: Offset comparison with/without gravity
    ax = axes[0, 0]
    offsets = sorted(set(r['offset'] for r in test1_results))
    x = np.arange(len(offsets))
    width = 0.35
    
    no_grav = [next(r for r in test1_results if r['offset']==o and r['kappa']==0.0) for o in offsets]
    with_grav = [next(r for r in test1_results if r['offset']==o and r['kappa']==0.02) for o in offsets]
    
    for i, (ng, wg) in enumerate(zip(no_grav, with_grav)):
        ax.bar(x[i] - width/2, 1, width, color=colors.get(ng['outcome'], 'gray'), alpha=0.7)
        ax.bar(x[i] + width/2, 1, width, color=colors.get(wg['outcome'], 'gray'), alpha=0.7, hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in offsets])
    ax.set_xlabel("Offset")
    ax.set_title("Test 1: g=1.5 (solid=no grav, hatched=with grav)")
    
    # 2. Test 3: g=0.8 comparison
    ax = axes[0, 1]
    offsets_3 = sorted(set(r['offset'] for r in test3_results))
    x = np.arange(len(offsets_3))
    
    no_grav_3 = [next(r for r in test3_results if r['offset']==o and r['kappa']==0.0) for o in offsets_3]
    with_grav_3 = [next(r for r in test3_results if r['offset']==o and r['kappa']==0.02) for o in offsets_3]
    
    for i, (ng, wg) in enumerate(zip(no_grav_3, with_grav_3)):
        ax.bar(x[i] - width/2, 1, width, color=colors.get(ng['outcome'], 'gray'), alpha=0.7)
        ax.bar(x[i] + width/2, 1, width, color=colors.get(wg['outcome'], 'gray'), alpha=0.7, hatch='//')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in offsets_3])
    ax.set_xlabel("Offset")
    ax.set_title("Test 3: g=0.8 (solid=no grav, hatched=with grav)")
    
    # 3. Test 4: KAPPA sweep
    ax = axes[0, 2]
    for i, r in enumerate(test4_results):
        ax.bar(i, 1, color=colors.get(r['outcome'], 'gray'), alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(test4_results)))
    ax.set_xticklabels([f"{r['kappa']:.2f}" for r in test4_results])
    ax.set_xlabel("KAPPA")
    ax.set_title("Test 4: KAPPA sweep (g=1.5, offset=20)")
    
    # 4. Separation traces for different KAPPA
    ax = axes[1, 0]
    for r in test4_results:
        if r['sep_trace']:
            t = np.arange(len(r['sep_trace'])) * 10 * 0.05
            ax.plot(t, r['sep_trace'], label=f"κ={r['kappa']:.2f}")
    ax.axhline(y=S_CONTACT, color='k', ls='--', alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Separation")
    ax.set_title("Separation vs Time (Test 4)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. min_sep comparison
    ax = axes[1, 1]
    offsets = sorted(set(r['offset'] for r in test1_results))
    ng_sep = [next(r for r in test1_results if r['offset']==o and r['kappa']==0.0)['min_sep'] for o in offsets]
    wg_sep = [next(r for r in test1_results if r['offset']==o and r['kappa']==0.02)['min_sep'] for o in offsets]
    
    x = np.arange(len(offsets))
    ax.bar(x - width/2, ng_sep, width, label='No gravity', alpha=0.7)
    ax.bar(x + width/2, wg_sep, width, label='KAPPA=0.02', alpha=0.7)
    ax.axhline(y=S_CONTACT, color='r', ls='--', label='Contact threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(o) for o in offsets])
    ax.set_xlabel("Offset")
    ax.set_ylabel("min_sep")
    ax.set_title("Minimum Separation (g=1.5)")
    ax.legend()
    
    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Count improvements
    improvements = []
    for o in offsets:
        ng = next(r for r in test1_results if r['offset']==o and r['kappa']==0.0)
        wg = next(r for r in test1_results if r['offset']==o and r['kappa']==0.02)
        if wg['made_contact'] and not ng['made_contact']:
            improvements.append(f"offset={o}: contact enabled")
        elif wg['min_sep'] < ng['min_sep'] - 5:
            improvements.append(f"offset={o}: min_sep reduced by {ng['min_sep']-wg['min_sep']:.0f}")
    
    summary = f"""
════════════════════════════════════════════════
        GRAVITY EFFECT SUMMARY
════════════════════════════════════════════════

Test 1 (g=1.5):
  Gravity (KAPPA=0.02) vs No gravity
  
Improvements detected:
"""
    for imp in improvements[:5]:
        summary += f"  • {imp}\n"
    
    if not improvements:
        summary += "  • No clear improvements\n"
    
    summary += f"""
Test 2 (Capture radius):
  KAPPA=0.00: offset ≤ ?
  KAPPA=0.02: offset ≤ ?
  KAPPA=0.05: offset ≤ ?

Test 4 (KAPPA sweep):
  Best outcome at KAPPA = ?
"""
    
    ax.text(0.02, 0.98, summary, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./det_v5_gravity_collider.png', dpi=150)
    print("Saved to det_v5_gravity_collider.png")
    
    return test1_results, test3_results, test4_results


if __name__ == "__main__":
    t0 = clock_time.time()
    t1, t3, t4 = main()
    elapsed = clock_time.time() - t0
    
    print("\n" + "="*70)
    print(f"COMPLETE ({elapsed:.1f}s)")
    print("="*70)
    
    # Analysis
    print("\nKEY FINDINGS:")
    
    # Did gravity help contact?
    offsets = sorted(set(r['offset'] for r in t1))
    for o in offsets:
        ng = next(r for r in t1 if r['offset']==o and r['kappa']==0.0)
        wg = next(r for r in t1 if r['offset']==o and r['kappa']==0.02)
        if wg['made_contact'] != ng['made_contact']:
            if wg['made_contact']:
                print(f"  ✓ offset={o}: Gravity ENABLED contact")
            else:
                print(f"  ✗ offset={o}: Gravity PREVENTED contact")
        elif wg['outcome'] != ng['outcome']:
            print(f"  ~ offset={o}: Outcome changed {ng['outcome']} → {wg['outcome']}")
