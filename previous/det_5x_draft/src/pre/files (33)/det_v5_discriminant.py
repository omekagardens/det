import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from collections import defaultdict
import time as clock_time

"""
DET v5 COLLIDER - DISCRIMINANT ANALYSIS
=======================================

Following the critique: print the missing discriminants to understand
what actually determines BOUND vs SCATTER vs FRAGMENT.

Steps:
1. Print min_sep, made_contact, final_sep, persistent_binding, contact_duration
2. Offset scan in guaranteed-contact band (15, 18, 20, 22, 25)
3. Extended runtime test for BOUND seed (g=0.8)
4. Bridging test: (g=1.5, v_kick=2.4) at offset=15 vs 22
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


def full_analysis(sim):
    """Complete analysis with all discriminants."""
    track_L = np.array(sim.track_L)
    track_R = np.array(sim.track_R)
    if len(track_L) < 20:
        return None
    
    times = track_L[:, 0]
    pos_L = track_L[:, 1]
    pos_R = track_R[:, 1]
    separation = np.abs(pos_L - pos_R)
    
    min_sep = np.min(separation)
    final_sep = separation[-1]
    final_peaks = sim.count_peaks()
    made_contact = min_sep <= S_CONTACT
    
    # Persistence check
    n_frames = len(separation)
    persist_start = int(n_frames * (1 - PERSIST_FRAC))
    late_sep = separation[persist_start:]
    persistent_binding = np.all(late_sep < S_HOLD)
    
    # Contact duration: fraction of frames with sep < S_HOLD
    contact_duration = np.mean(separation < S_HOLD)
    
    # Deep contact duration: fraction with sep < S_CONTACT
    deep_contact_duration = np.mean(separation < S_CONTACT)
    
    # Classification
    if made_contact and final_peaks == 1 and persistent_binding:
        outcome = 'BOUND'
    elif final_peaks >= FRAG_THRESHOLD:
        outcome = 'FRAGMENT'
    else:
        outcome = 'SCATTER'
    
    return {
        'outcome': outcome,
        'min_sep': min_sep,
        'final_sep': final_sep,
        'final_peaks': final_peaks,
        'made_contact': made_contact,
        'persistent': persistent_binding,
        'contact_dur': contact_duration,
        'deep_contact_dur': deep_contact_duration,
    }


def run_trial(g, v_kick, offset, steps=800):
    """Run single trial with full analysis."""
    k = DETConstants(g=g)
    sim = ActiveManifold1D(size=200, constants=k)
    sim.initialize_collider_event(offset=offset, v_kick=v_kick)
    for _ in range(steps):
        sim.step()
    return full_analysis(sim), sim


def main():
    print("="*80)
    print("DET v5 DISCRIMINANT ANALYSIS")
    print("="*80)
    
    # =========================================================================
    # STEP 1: Rerun Exp 1 & 2 with all discriminants
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Full discriminants (g=1.5, offset=22)")
    print("="*80)
    print(f"{'v_kick':<7}|{'outcome':<9}|{'min_sep':<8}|{'final_sep':<10}|{'peaks':<6}|{'contact':<8}|{'persist':<8}|{'dur%':<6}")
    print("-" * 80)
    
    for v_kick in [1.0, 1.4, 1.8, 2.0, 2.2, 2.4, 2.6]:
        res, _ = run_trial(g=1.5, v_kick=v_kick, offset=22)
        if res:
            print(f"{v_kick:<7.1f}|{res['outcome']:<9}|{res['min_sep']:<8.1f}|{res['final_sep']:<10.1f}|"
                  f"{res['final_peaks']:<6}|{str(res['made_contact']):<8}|{str(res['persistent']):<8}|{100*res['contact_dur']:<6.0f}")
    
    print("\n--- g-scan at v_kick=2.0, offset=22 ---")
    print(f"{'g':<6}|{'outcome':<9}|{'min_sep':<8}|{'final_sep':<10}|{'peaks':<6}|{'contact':<8}|{'persist':<8}|{'dur%':<6}")
    print("-" * 75)
    
    for g in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        res, _ = run_trial(g=g, v_kick=2.0, offset=22)
        if res:
            print(f"{g:<6.1f}|{res['outcome']:<9}|{res['min_sep']:<8.1f}|{res['final_sep']:<10.1f}|"
                  f"{res['final_peaks']:<6}|{str(res['made_contact']):<8}|{str(res['persistent']):<8}|{100*res['contact_dur']:<6.0f}")
    
    # =========================================================================
    # STEP 2: Offset scan in guaranteed-contact band
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Offset scan (g=1.5, v_kick=2.0)")
    print("="*80)
    print(f"{'offset':<7}|{'sep0':<6}|{'outcome':<9}|{'min_sep':<8}|{'final_sep':<10}|{'peaks':<6}|{'contact':<8}|{'dur%':<6}")
    print("-" * 75)
    
    step2_results = []
    for offset in [15, 16, 17, 18, 20, 22, 25]:
        res, _ = run_trial(g=1.5, v_kick=2.0, offset=offset)
        if res:
            print(f"{offset:<7}|{2*offset:<6}|{res['outcome']:<9}|{res['min_sep']:<8.1f}|{res['final_sep']:<10.1f}|"
                  f"{res['final_peaks']:<6}|{str(res['made_contact']):<8}|{100*res['contact_dur']:<6.0f}")
            step2_results.append({'offset': offset, **res})
    
    # =========================================================================
    # STEP 3: Extended runtime for BOUND seed (g=0.8)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Extended runtime test (g=0.8, v_kick=2.0, offset=22)")
    print("="*80)
    print(f"{'steps':<8}|{'outcome':<9}|{'min_sep':<8}|{'final_sep':<10}|{'peaks':<6}|{'persist':<8}")
    print("-" * 55)
    
    for steps in [800, 1600, 3200]:
        res, _ = run_trial(g=0.8, v_kick=2.0, offset=22, steps=steps)
        if res:
            print(f"{steps:<8}|{res['outcome']:<9}|{res['min_sep']:<8.1f}|{res['final_sep']:<10.1f}|"
                  f"{res['final_peaks']:<6}|{str(res['persistent']):<8}")
    
    # =========================================================================
    # STEP 4: Bridging test (g=1.5, v_kick=2.4 at offset=15 vs 22)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Bridging test (g=1.5, v_kick=2.4)")
    print("="*80)
    print("Suite A2 showed BOUND at v_kick>=2.3 for offset=15. Does it hold at offset=22?\n")
    
    print(f"{'offset':<7}|{'outcome':<9}|{'min_sep':<8}|{'final_sep':<10}|{'peaks':<6}|{'contact':<8}|{'persist':<8}")
    print("-" * 65)
    
    bridge_results = []
    for offset in [15, 18, 20, 22]:
        res, _ = run_trial(g=1.5, v_kick=2.4, offset=offset)
        if res:
            print(f"{offset:<7}|{res['outcome']:<9}|{res['min_sep']:<8.1f}|{res['final_sep']:<10.1f}|"
                  f"{res['final_peaks']:<6}|{str(res['made_contact']):<8}|{str(res['persistent']):<8}")
            bridge_results.append({'offset': offset, **res})
    
    # Also test v_kick=2.3 which was at the boundary
    print("\n--- Also test v_kick=2.3 ---")
    print(f"{'offset':<7}|{'outcome':<9}|{'min_sep':<8}|{'final_sep':<10}|{'peaks':<6}|{'contact':<8}|{'persist':<8}")
    print("-" * 65)
    
    for offset in [15, 18, 22]:
        res, _ = run_trial(g=1.5, v_kick=2.3, offset=offset)
        if res:
            print(f"{offset:<7}|{res['outcome']:<9}|{res['min_sep']:<8.1f}|{res['final_sep']:<10.1f}|"
                  f"{res['final_peaks']:<6}|{str(res['made_contact']):<8}|{str(res['persistent']):<8}")
    
    # =========================================================================
    # STEP 5: Low-g binding region exploration
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Low-g binding region (offset=22, v_kick=2.0)")
    print("="*80)
    print(f"{'g':<6}|{'outcome':<9}|{'min_sep':<8}|{'final_sep':<10}|{'peaks':<6}|{'contact':<8}|{'persist':<8}")
    print("-" * 65)
    
    for g in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        res, _ = run_trial(g=g, v_kick=2.0, offset=22)
        if res:
            print(f"{g:<6.1f}|{res['outcome']:<9}|{res['min_sep']:<8.1f}|{res['final_sep']:<10.1f}|"
                  f"{res['final_peaks']:<6}|{str(res['made_contact']):<8}|{str(res['persistent']):<8}")
    
    return step2_results, bridge_results


if __name__ == "__main__":
    t0 = clock_time.time()
    step2, bridge = main()
    elapsed = clock_time.time() - t0
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE ({elapsed:.1f}s)")
    print("="*80)
    
    print("\nKey questions answered:")
    print("1. Does 'contact' actually occur (min_sep <= 2)?")
    print("2. Does binding require deep overlap (low offset)?")
    print("3. Is g=0.8 BOUND stable over time?")
    print("4. Does v_kick=2.4 binding require offset=15?")
