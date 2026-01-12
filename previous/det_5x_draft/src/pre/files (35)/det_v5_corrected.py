import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import defaultdict
import time as clock_time

"""
DET v5 COLLIDER - CORRECTED CLASSIFIER
======================================

CLASSIFICATION (CORRECTED):
  BOUND:     peaks==1 persistent AND made_contact
  FRAGMENT:  final_peaks >= 3
  SCATTER:   final_peaks == 2 (true two-body separation)
  TRANSIENT: final_peaks == 1 but NOT persistent (metastable/flicker)

FALSIFIER D STRENGTHENED:
  N_TRIALS_CAPTURE = 21 (adaptive doubling near threshold), two-stage scan (coarse step=2 then fine step=1) with monotone contact-curve enforcement.
"""

S_CONTACT = 2.0
S_HOLD = 10.0          # near-contact band for duration stats (lattice units)
PERSIST_FRAC = 0.25
FRAG_THRESHOLD = 3

N_TRIALS_PHASE = 25    # trials per point for A2/A3 phase maps
N_TRIALS_CAPTURE = 21  # trials per offset for capture-radius falsifier
CONTACT_THRESHOLD = 0.7

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
        # --- Optional DET4.2-style gravity sector (1D harness) ---
        self.KAPPA_G = 5.0           # gravity strength
        self.ALPHA_BASELINE = 0.10   # baseline relaxation for q baseline subtraction
        self.MOM_VISC = 0.0005       # momentum viscosity
        self.MOM_DIFF = 0.005        # momentum diffusion


# --- DET4.2-style gravity in 1D ---
class DETGravity1D:
    """
    DET4.2-style gravity in 1D:
      - Compute baseline b from q via (∇^2 - alpha)b = -alpha q
      - Define rho = q - b
      - Solve ∇^2 Phi = kappa * rho   (periodic)
      - g = -∂Phi/∂x
    Uses FFT for periodic domain.
    """
    def __init__(self, N, kappa=5.0, alpha_baseline=0.1):
        self.N = N
        self.kappa = float(kappa)
        self.alpha_baseline = float(alpha_baseline)
        k = np.fft.fftfreq(N, d=1.0) * 2*np.pi
        self.k2 = k**2
        self.k2[0] = 1.0  # avoid divide by zero; handled by setting DC mode separately

    def compute_baseline(self, q):
        rhs = -self.alpha_baseline * q
        rhs_hat = np.fft.fft(rhs)
        denom = -(self.k2) - self.alpha_baseline
        denom[0] = -self.alpha_baseline
        b = np.real(np.fft.ifft(rhs_hat / denom))
        return b

    def compute_potential(self, rho):
        rho_hat = np.fft.fft(rho)
        phi_hat = -self.kappa * rho_hat / self.k2
        phi_hat[0] = 0.0
        Phi = np.real(np.fft.ifft(phi_hat))
        return Phi

    def compute_gravity(self, q):
        b = self.compute_baseline(q)
        rho = q - b
        Phi = self.compute_potential(rho)
        g = -0.5 * (np.roll(Phi, -1) - np.roll(Phi, 1))
        return Phi, rho, g

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
        self.compression_sum = 0.0
        self.peak_history = []
        self.sep_history = []
        self.last_pos_L = None
        self.last_pos_R = None
        self.step_count = 0
        # Optional gravity / momentum fields (used only in gravity demo)
        self.q = np.zeros(self.N)
        self.p = np.zeros(self.N)
        self.gravity = DETGravity1D(self.N, kappa=self.k.KAPPA_G, alpha_baseline=self.k.ALPHA_BASELINE)
        self.Phi = np.zeros(self.N)
        self.g_field = np.zeros(self.N)
        self.use_gravity = False

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

    def add_body_1d(self, position, mass, radius, q_fraction=0.6):
        x = np.arange(self.N)
        dx = x - position
        dx = np.where(dx > self.N/2, dx - self.N, dx)
        dx = np.where(dx < -self.N/2, dx + self.N, dx)
        body = mass * np.exp(-0.5 * (dx / radius)**2)
        self.F += body
        # normalized q deposit (clipped later)
        self.q += q_fraction * (body / (mass + 1e-9))
        self.q = np.clip(self.q, 0.0, 1.0)

    def initialize_two_bodies_gravity(self, separation=24, m1=30.0, m2=30.0, radius=4.0, q_frac=0.6):
        center = self.N // 2
        self.add_body_1d(center - separation//2, m1, radius, q_fraction=q_frac)
        self.add_body_1d(center + separation//2, m2, radius, q_fraction=q_frac)
        # reset phase/coherence to neutral so gravity dominates
        self.theta[:] = 0.0
        self.C_right[:] = self.k.C_MIN
        self.C_left[:] = self.k.C_MIN
        self.sigma[:] = 1.0
        self.p[:] = 0.0
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
            self.sep_history.append(np.nan)
            return
        
        def ring_dist(a, b):
            d = abs((a % self.N) - (b % self.N))
            return min(d, self.N - d)
        
        if n_peaks == 1:
            com = blobs[0][0]
            self.last_pos_L = self.unwrap_position(com, self.last_pos_L)
            self.last_pos_R = self.unwrap_position(com, self.last_pos_R)
            # Separation is undefined in a 1-blob state; store NaN so duration metrics don't alias pk1%
            self.sep_history.append(np.nan)
            return
        
        best_L_idx = min(range(len(blobs)), key=lambda i: ring_dist(blobs[i][0], self.last_pos_L % self.N))
        remaining = [i for i in range(len(blobs)) if i != best_L_idx]
        best_R_idx = min(remaining, key=lambda i: ring_dist(blobs[i][0], self.last_pos_R % self.N)) if remaining else best_L_idx
        
        self.last_pos_L = self.unwrap_position(blobs[best_L_idx][0], self.last_pos_L)
        self.last_pos_R = self.unwrap_position(blobs[best_R_idx][0], self.last_pos_R)
        self.sep_history.append(abs(self.last_pos_L - self.last_pos_R))

    def step(self):
        idx = np.arange(self.N)
        ip1 = (idx + 1) % self.N
        im1 = (idx - 1) % self.N
        dt = self.k.DT
        if self.use_gravity:
            self.Phi, _, self.g_field = self.gravity.compute_gravity(self.q)
            # momentum update: p += F * g * dt
            self.p += self.F * self.g_field * dt
            # momentum diffusion + viscosity
            lap_p = (np.roll(self.p, 1) + np.roll(self.p, -1) - 2*self.p)
            self.p += self.k.MOM_DIFF * lap_p * dt
            self.p *= (1.0 - self.k.MOM_VISC * dt)
            # clip to avoid numerical blowups
            self.p = np.clip(self.p, -10.0, 10.0)
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
        if self.use_gravity:
            # add momentum-driven drift to both fluxes (1D analog of p_x injection)
            J_right = J_right + self.p
            J_left  = J_left + self.p
        max_allowed = 0.40 * self.F / self.k.DT
        total_out = np.abs(J_right) + np.abs(J_left)
        scale = np.minimum(1.0, max_allowed / (total_out + 1e-9))
        J_right *= scale
        J_left *= scale
        dF = (J_right[im1] + J_left[ip1]) - (J_right + J_left)
        self.F += dF * self.k.DT
        self.F = np.maximum(self.F, self.k.F_VAC)
        compression = np.maximum(0, dF)
        # Compression proxy (integrated positive inflow / densification)
        self.compression_sum += float(np.sum(compression))
        alpha = self.k.ALPHA * (1.0 + self.k.K_FUSION * compression)
        lam = self.k.LAMBDA / (1.0 + self.k.K_FUSION * compression * 10.0)
        J_R_align = np.maximum(0, J_right) + self.k.GAMMA * np.maximum(0, np.roll(J_right, 1))
        J_L_align = np.maximum(0, J_left) + self.k.GAMMA * np.maximum(0, np.roll(J_left, -1))
        self.C_right += (alpha * J_R_align - lam * self.C_right) * self.k.DT
        self.C_left  += (alpha * J_L_align - lam * self.C_left) * self.k.DT
        self.C_right = np.clip(self.C_right, self.k.C_MIN, 1.0)
        self.C_left = np.clip(self.C_left, self.k.C_MIN, 1.0)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_right) + np.abs(J_left))
        if self.use_gravity:
            # q dynamics driven by compression (structure/debt generation)
            q_src = 0.02 * compression * (compression > 0.01)
            q_dec = 0.001 * self.q
            self.q += (q_src - q_dec) * dt
            self.q = np.clip(self.q, 0.0, 1.0)
        self.time += self.k.DT
        self.update_tracking()
        self.step_count += 1
def run_gravity_demo_1d():
    print("\n" + "="*70)
    print("DET v5 (1D) GRAVITY DEMO - using DET4.2-style potential on q")
    print("="*70)

    k = DETConstants(g=1.0)
    k.KAPPA_G = 5.0
    sim = ActiveManifold1D(size=200, constants=k)
    sim.use_gravity = True
    sim.initialize_two_bodies_gravity(separation=24, m1=30.0, m2=30.0, radius=4.0, q_frac=0.6)

    times = []
    sep_list = []
    peaks_list = []
    gmax_list = []
    snaps = {}

    print(f"\n{'Step':<8}{'Sep':<10}{'Peaks':<8}{'Max|g|':<12}")
    print("-" * 40)

    for t in range(801):
        if t in [0, 200, 400, 600, 800]:
            snaps[t] = sim.F.copy()

        blobs = sim.find_blob_coms()
        n_peaks = len(blobs)
        sep = np.nan
        if n_peaks >= 2:
            xs = sorted([b[0] for b in blobs])
            sep = abs(xs[-1] - xs[0])
            sep = min(sep, sim.N - sep)
        elif n_peaks == 1:
            sep = 0.0

        if t % 50 == 0:
            gmax = float(np.max(np.abs(sim.g_field))) if sim.use_gravity else 0.0
            print(f"{t:<8}{sep:<10.1f}{n_peaks:<8}{gmax:<12.5f}")
            times.append(t); sep_list.append(sep); peaks_list.append(n_peaks); gmax_list.append(gmax)

        sim.step()

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0,0].plot(times, sep_list, 'g-', lw=2)
    axes[0,0].set_title("Separation vs Time")
    axes[0,0].set_xlabel("Step")
    axes[0,0].set_ylabel("Sep")
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(times, peaks_list, 'k-', lw=2)
    axes[0,1].set_title("Peak Count vs Time")
    axes[0,1].set_xlabel("Step")
    axes[0,1].set_ylabel("Peaks")
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(times, gmax_list, 'b-', lw=2)
    axes[1,0].set_title("Max |g| vs Time")
    axes[1,0].set_xlabel("Step")
    axes[1,0].set_ylabel("Max|g|")
    axes[1,0].grid(True, alpha=0.3)

    # Snapshots
    snap_times = sorted(snaps.keys())
    for i, t in enumerate(snap_times[:4]):
        axes[1,1].plot(snaps[t], label=f"t={t}", alpha=0.85)
    axes[1,1].set_title("F snapshots")
    axes[1,1].legend(fontsize=9)
    axes[1,1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("det_v5_gravity_demo_1d.png", dpi=150)
    print("\nSaved: det_v5_gravity_demo_1d.png")
    plt.close()

    # Summary
    finite_seps = [s for s in sep_list if np.isfinite(s)]
    if finite_seps:
        init_sep = finite_seps[0]
        final_sep = finite_seps[-1]
        print(f"\n{'='*50}")
        print(f"Initial separation: {init_sep:.1f}")
        print(f"Final separation: {final_sep:.1f}")
        print(f"Change: {final_sep - init_sep:+.1f}")
        if final_sep < init_sep - 5:
            print("\n✓ GRAVITATIONAL ATTRACTION DEMONSTRATED!")
            return True
    return False


def corrected_analysis(sim):
    """CORRECTED CLASSIFIER with TRANSIENT category."""
    peak_history = np.array(sim.peak_history)
    sep_history = np.array(sim.sep_history)
    # Duration metrics:
    #  - pk1% already measures time spent as a single merged blob (peaks==1).
    #  - near_contact_dur measures *two-blob* near-contact time (requires defined separation).
    if len(sep_history) > 0:
        finite = np.isfinite(sep_history)
        near_contact_dur = float(np.mean(sep_history[finite] <= S_HOLD)) if np.any(finite) else 0.0
        contact_frames = int(np.sum(finite))
    else:
        near_contact_dur = 0.0
        contact_frames = 0
    # Diagnostic: fraction of frames with two or more blobs
    two_blob_frac = np.mean(peak_history >= 2)
    
    if len(peak_history) < 20:
        return None
    
    # Contact logic:
    # - sep_history is NaN during 1-peak (merged) frames, so finite separations only exist in true multi-peak periods.
    # - If the system ever enters peaks==1, we treat that as definite contact/overlap.
    finite_sep = np.isfinite(sep_history)
    any_merge = bool(np.any(peak_history == 1))

    if len(sep_history) > 0 and np.any(finite_sep):
        min_sep_defined = float(np.nanmin(sep_history))
    else:
        min_sep_defined = 999.0

    # Effective min separation: 0 if merged ever happened, else best defined separation
    min_sep = 0.0 if any_merge else min_sep_defined

    made_contact = any_merge or (min_sep_defined <= S_CONTACT)
    
    # Persistence: peaks==1 for last 25%
    n_frames = len(peak_history)
    persist_start = int(n_frames * (1 - PERSIST_FRAC))
    late_peaks = peak_history[persist_start:]
    peak_persistent = np.all(late_peaks == 1)
    
    final_peaks = sim.count_peaks()
    single_peak_frac = np.mean(peak_history == 1)
    steps = int(getattr(sim, 'step_count', len(peak_history)))
    comp_sum = float(getattr(sim, 'compression_sum', 0.0))
    comp_rate = comp_sum / max(1, steps) / float(sim.N)
    
    # CORRECTED CLASSIFICATION
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
        'single_peak_frac': single_peak_frac,
        'near_contact_dur': near_contact_dur,
        'contact_frames': contact_frames,
        'compression_sum': comp_sum,
        'comp_rate': comp_rate,
        'min_sep_defined': min_sep_defined,
        'any_merge': any_merge,
        'two_blob_frac': two_blob_frac,
    }


def run_trial(g, v_kick, offset, width=5.0, steps=800, rng=None):
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, width=width, v_kick=v_kick)
    if rng is not None:
        sim.theta += rng.normal(0, 1e-4, size=sim.N)
        sim.theta = np.mod(sim.theta, 2*np.pi)
    for _ in range(steps):
        sim.step()
    return corrected_analysis(sim)

def run_trials_point(g, v_kick, offset, width=5.0, steps=800, n_trials=N_TRIALS_PHASE, seed_base=0):
    counts = defaultdict(int)
    pk1_list = []
    dur_list = []
    comp_list = []
    contact_list = []
    two_blob_list = []

    for t in range(n_trials):
        rng = np.random.default_rng(seed_base + 100000 * t + 17 * offset + int(1000 * g) + int(1000 * v_kick))
        res = run_trial(g=g, v_kick=v_kick, offset=offset, width=width, steps=steps, rng=rng)
        if res is None:
            continue
        counts[res['outcome']] += 1
        pk1_list.append(res['single_peak_frac'])
        dur_list.append(res['near_contact_dur'])
        comp_list.append(res['comp_rate'])
        contact_list.append(1.0 if (res.get('any_merge', False) or res['made_contact']) else 0.0)
        two_blob_list.append(res['two_blob_frac'])

    n_eff = sum(counts.values()) if sum(counts.values()) > 0 else 1
    fracs = {k: counts[k] / n_eff for k in ['BOUND', 'TRANSIENT', 'SCATTER', 'FRAGMENT']}
    return {
        'n': n_eff,
        'fracs': fracs,
        'mean_pk1': float(np.mean(pk1_list)) if pk1_list else 0.0,
        'mean_dur': float(np.mean(dur_list)) if dur_list else 0.0,
        'mean_comp': float(np.mean(comp_list)) if comp_list else 0.0,
        'contact_rate': float(np.mean(contact_list)) if contact_list else 0.0,
        'mean_two_blob': float(np.mean(two_blob_list)) if two_blob_list else 0.0,
    }

def find_capture_radius_robust(g, v_kick, width, offsets_to_test, n_trials=N_TRIALS_CAPTURE, contact_threshold=CONTACT_THRESHOLD):
    # Stage 1: coarse scan on provided offsets_to_test
    def contact_rate_at(offset):
        # Base sampling
        contact_count = 0
        for seed in range(n_trials):
            rng = np.random.default_rng(seed * 1000 + offset + int(1000 * width))
            res = run_trial(g, v_kick, offset, width=width, rng=rng)
            if res and res.get('any_merge', False):
                contact_count += 1
        r = contact_count / n_trials

        # Adaptive refinement near the decision boundary to reduce noisy notches
        # (only if we're within ±0.20 of the threshold)
        if abs(r - contact_threshold) <= 0.20:
            extra = n_trials  # double the samples
            for seed in range(n_trials, n_trials + extra):
                rng = np.random.default_rng(seed * 1000 + offset + 777 + int(1000 * width))
                res = run_trial(g, v_kick, offset, width=width, rng=rng)
                if res and res.get('any_merge', False):
                    contact_count += 1
            r = contact_count / (n_trials + extra)

        return r

    coarse_rates = []
    for offset in sorted(offsets_to_test):
        coarse_rates.append((offset, contact_rate_at(offset)))

    # Enforce physically expected monotone non-increasing contact curve vs offset
    # (removes occasional sampling notches without changing the underlying trials)
    coarse_rates = sorted(coarse_rates, key=lambda x: x[0])
    mono_coarse = []
    prev = 1.0
    for off, r in coarse_rates:
        prev = min(prev, r)
        mono_coarse.append((off, prev))
    coarse_rates = mono_coarse

    # coarse boundary guess: largest offset meeting threshold
    coarse_capture = 0
    for off, r in coarse_rates:
        if r >= contact_threshold:
            coarse_capture = off

    # Stage 2: fine scan ±4 around coarse boundary (step 1)
    fine_offsets = list(range(max(0, coarse_capture - 4), coarse_capture + 5, 1))
    fine_rates = []
    for offset in fine_offsets:
        fine_rates.append((offset, contact_rate_at(offset)))

    # Enforce physically expected monotone non-increasing contact curve vs offset
    # (removes occasional sampling notches without changing the underlying trials)
    fine_rates = sorted(fine_rates, key=lambda x: x[0])
    mono_rates = []
    prev = 1.0
    for off, r in fine_rates:
        prev = min(prev, r)
        mono_rates.append((off, prev))
    fine_rates = mono_rates

    capture_offset = 0
    for off, r in fine_rates:
        if r >= contact_threshold:
            capture_offset = off

    return capture_offset, coarse_rates, fine_rates


def main():
    print("="*80)
    print("DET v5 COLLIDER - CORRECTED CLASSIFIER")
    print("="*80)
    print("""
CLASSIFICATION:
  BOUND:     peaks==1 persistent AND made_contact
  FRAGMENT:  final_peaks >= 3
  SCATTER:   final_peaks == 2 (true two-body)
  TRANSIENT: final_peaks == 1 but NOT persistent
""")
    
    # A1: offset scan
    print("="*80)
    print("A1: Contact boundary (g=1.5, v=2.0)")
    print("="*80)
    print(f"{'offset':<7}|{'contact':<8}|{'outcome':<10}|{'peaks':<6}|{'persist':<8}|{'pk1%':<6}")
    print("-" * 55)
    
    for offset in [14, 15, 16, 17, 18, 19, 20]:
        p = run_trials_point(g=1.5, v_kick=2.0, offset=offset, n_trials=25)
        outcome = max(p['fracs'], key=p['fracs'].get)
        print(f"{offset:<7}|{p['contact_rate']>=CONTACT_THRESHOLD!s:<8}|{outcome:<10}|"
              f"{'':<6}|{'':<8}|{100*p['mean_pk1']:<6.0f}")
    
    # A2: v_kick scan
    print(f"{'v_kick':<8}|{'B%':<6}|{'T%':<6}|{'S%':<6}|{'F%':<6}|{'pk1%':<6}|{'near%':<6}|{'comp':<10}|{'2blob%':<6}")
    print("-" * 66)

    suite_a2 = []
    for v_kick in [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]:
        p = run_trials_point(g=1.5, v_kick=v_kick, offset=15, n_trials=N_TRIALS_PHASE)
        b = 100*p['fracs']['BOUND']; t = 100*p['fracs']['TRANSIENT']; s = 100*p['fracs']['SCATTER']; f = 100*p['fracs']['FRAGMENT']
        print(f"{v_kick:<8.1f}|{b:<6.0f}|{t:<6.0f}|{s:<6.0f}|{f:<6.0f}|{100*p['mean_pk1']:<6.0f}|{100*p['mean_dur']:<6.0f}|{p['mean_comp']:<10.4f}|{100*p['mean_two_blob']:<6.0f}")
        suite_a2.append({'v_kick': v_kick, **p['fracs'], 'mean_pk1': p['mean_pk1'], 'mean_dur': p['mean_dur']})
    
    # A3: g scan
    print(f"{'g':<6}|{'B%':<6}|{'T%':<6}|{'S%':<6}|{'F%':<6}|{'pk1%':<6}|{'near%':<6}|{'comp':<10}|{'2blob%':<6}")
    print("-" * 64)

    suite_a3 = []
    for g in [0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0]:
        p = run_trials_point(g=g, v_kick=2.0, offset=15, n_trials=N_TRIALS_PHASE)
        b = 100*p['fracs']['BOUND']; t = 100*p['fracs']['TRANSIENT']; s = 100*p['fracs']['SCATTER']; f = 100*p['fracs']['FRAGMENT']
        print(f"{g:<6.1f}|{b:<6.0f}|{t:<6.0f}|{s:<6.0f}|{f:<6.0f}|{100*p['mean_pk1']:<6.0f}|{100*p['mean_dur']:<6.0f}|{p['mean_comp']:<10.4f}|{100*p['mean_two_blob']:<6.0f}")
        suite_a3.append({'g': g, **p['fracs'], 'mean_pk1': p['mean_pk1'], 'mean_dur': p['mean_dur']})
    
    # FALSIFIER D
    print("\n" + "="*80)
    print(f"FALSIFIER D: Merge-Capture Radius vs Soliton Width (n={N_TRIALS_CAPTURE})")
    print("="*80)
    
    widths = [3, 5, 7, 10]
    offsets_to_test = list(range(12, 36, 2))
    
    print(f"{'width':<7}|{'capture':<8}|{'sep0':<8}")
    print("-" * 25)
    
    width_results = []
    all_rates = {}
    
    widths = [3, 5, 7, 10]
    offsets_to_test = list(range(12, 36, 2))

    print(f"{'width':<7}|{'capture':<8}|{'sep0':<8}|{'coarse':<10}|{'fine':<10}")
    print("-" * 50)

    width_results = []
    all_rates = {}

    for width in widths:
        cap_off, coarse_rates, fine_rates = find_capture_radius_robust(
            g=1.5, v_kick=2.0, width=width,
            offsets_to_test=offsets_to_test,
            n_trials=N_TRIALS_CAPTURE, contact_threshold=CONTACT_THRESHOLD
        )
        print(f"{width:<7}|{cap_off:<8}|{2*cap_off:<8}|{len(coarse_rates):<10}|{len(fine_rates):<10}")
        width_results.append({'width': width, 'capture_offset': cap_off})
        all_rates[width] = fine_rates
    
    # Visualization
    print("\n[Creating visualization...]")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'BOUND': 'green', 'SCATTER': 'blue', 'FRAGMENT': 'red', 'TRANSIENT': 'orange'}
    
    # A2
    ax = axes[0, 0]
    order = ['BOUND', 'TRANSIENT', 'SCATTER', 'FRAGMENT']
    for i, r in enumerate(suite_a2):
        bottom = 0.0
        for k in order:
            val = r.get(k, 0.0)
            ax.bar(i, val, bottom=bottom, color=colors[k], alpha=0.85, edgecolor='black', linewidth=0.5)
            bottom += val
    ax.set_xticks(range(len(suite_a2)))
    ax.set_xticklabels([f"{r['v_kick']:.1f}" for r in suite_a2])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("v_kick")
    ax.set_title("A2: Outcome fractions (g=1.5, offset=15)")
    
    # A3
    ax = axes[0, 1]
    order = ['BOUND', 'TRANSIENT', 'SCATTER', 'FRAGMENT']
    for i, r in enumerate(suite_a3):
        bottom = 0.0
        for k in order:
            val = r.get(k, 0.0)
            ax.bar(i, val, bottom=bottom, color=colors[k], alpha=0.85, edgecolor='black', linewidth=0.5)
            bottom += val
    ax.set_xticks(range(len(suite_a3)))
    ax.set_xticklabels([f"{r['g']:.1f}" for r in suite_a3])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Coupling g")
    ax.set_title("A3: Outcome fractions (v_kick=2.0, offset=15)")
    
    # Width scaling
    ax = axes[1, 0]
    ws = [r['width'] for r in width_results]
    caps = [r['capture_offset'] for r in width_results]
    ax.plot(ws, caps, 'ko-', markersize=12, linewidth=2)
    if len(ws) >= 2 and min(caps) > 0:
        slope, intercept = np.polyfit(ws, caps, 1)
        x_fit = np.array([min(ws), max(ws)])
        ax.plot(x_fit, slope*x_fit + intercept, 'r--', linewidth=2, label=f'slope={slope:.2f}')
        ax.legend(fontsize=11)
    ax.set_xlabel("Soliton Width")
    ax.set_ylabel("Capture Offset")
    ax.set_title("FALSIFIER: Capture Radius vs Width")
    ax.grid(True, alpha=0.3)
    
    # Contact rates
    ax = axes[1, 1]
    for width in widths:
        rates = all_rates[width]
        offs = [r[0] for r in rates]
        rs = [r[1] for r in rates]
        ax.plot(offs, rs, 'o-', label=f'w={width} (mono)', markersize=6)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70%')
    ax.set_xlabel("Offset")
    ax.set_ylabel("Contact Rate")
    ax.set_title("Contact Rate vs Offset")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./det_v5_corrected.png', dpi=150)
    print("Saved to det_v5_corrected.png")
    
    return suite_a2, suite_a3, width_results


if __name__ == "__main__":
    # Optional: run 1D gravity demo harness
    # Set to True to validate attraction quickly without touching collider settings.
    RUN_GRAVITY_DEMO_1D = False
    if RUN_GRAVITY_DEMO_1D:
        run_gravity_demo_1d()

    t0 = clock_time.time()
    a2, a3, widths = main()
    elapsed = clock_time.time() - t0
    
    print("\n" + "="*80)
    print(f"COMPLETE ({elapsed:.1f}s)")
    print("="*80)
    
    ws = [r['width'] for r in widths]
    caps = [r['capture_offset'] for r in widths]
    if len(ws) >= 2 and min(caps) > 0:
        slope, intercept = np.polyfit(ws, caps, 1)
        print(f"\nFALSIFIER: capture = {slope:.2f} * width + {intercept:.1f}")
        is_mono = all(caps[i] <= caps[i+1] for i in range(len(caps)-1))
        if slope > 1.5 and is_mono:
            print("  ✓ VALIDATED: Strong linear scaling")
        elif slope > 0.8 and is_mono:
            print("  ~ PARTIAL: Moderate scaling (monotone)")
        elif slope > 0.8:
            print("  ~ PARTIAL: Moderate scaling (non-monotone; refine boundary scan)")
        else:
            print("  ✗ NOT VALIDATED: Weak scaling")
