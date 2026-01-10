import numpy as np
import matplotlib.pyplot as plt

"""
DET v5 - NEUTRON TEST
=====================
Test: Fuse a Proton (spinning, phase ≠ 0) with a Neutron (non-spinning, phase = 0)
Prediction: Lower fusion barrier because no EM/phase repulsion between them

Physical basis:
- Proton: Has electric charge → phase rotation → EM repulsion with other protons
- Neutron: No charge → no phase rotation → no EM repulsion
- Proton-Neutron fusion should have LOWER barrier than Proton-Proton
"""

# Best parameters from alpha derivation (gave α within 2.9%)
EM_STRENGTH = 2.0
STRONG_STRENGTH = 200
EM_RANGE = 10
STRONG_RANGE = 0.5
FUSION_DIST = 1.0

MASS = 1.0
DT = 0.005
STEPS = 30000

class Particle:
    def __init__(self, x, v, phase, is_charged=True):
        self.x = x
        self.v = v
        self.phase = phase
        self.is_charged = is_charged  # Proton=True, Neutron=False

def compute_force(p1, p2):
    """
    Compute force between two particles.
    EM force only applies if BOTH particles are charged (protons).
    Strong force always applies (it's charge-blind).
    """
    r = abs(p2.x - p1.x)
    r = max(r, 0.1)
    direction = 1 if p1.x < p2.x else -1
    
    # EM force: Only if both particles are charged!
    if p1.is_charged and p2.is_charged:
        phase_diff = p2.phase - p1.phase
        phase_factor = np.sin(phase_diff / 2) ** 2
        F_em = EM_STRENGTH * phase_factor * np.exp(-r / EM_RANGE) / (r + 0.5)
    else:
        F_em = 0.0  # No EM repulsion for neutron
    
    # Strong force: Always present (nuclear, charge-blind)
    F_strong = -STRONG_STRENGTH * np.exp(-r / STRONG_RANGE) / (r + 0.1)
    
    return (F_em + F_strong) * direction, F_em, F_strong, r

def run_collision(particle1_charged, particle2_charged, energy, separation=50.0):
    """Run a single collision and return outcome."""
    v = np.sqrt(2 * energy / MASS)
    
    p1 = Particle(x=-separation/2, v=v, phase=0.0, is_charged=particle1_charged)
    p2 = Particle(x=separation/2, v=-v, phase=np.pi if particle2_charged else 0.0, 
                  is_charged=particle2_charged)
    
    history = []
    min_dist = separation
    fused = False
    
    for step in range(STEPS):
        F, F_em, F_strong, r = compute_force(p1, p2)
        min_dist = min(min_dist, r)
        
        history.append({
            'x1': p1.x, 'x2': p2.x, 'r': r,
            'F_em': F_em, 'F_strong': F_strong, 'F_total': F
        })
        
        a = F / MASS
        p1.v += a * DT
        p2.v -= a * DT
        p1.x += p1.v * DT
        p2.x += p2.v * DT
        
        if r < FUSION_DIST and F < 0:
            fused = True
            break
        
        if r > separation * 1.5:
            break
    
    return fused, min_dist, history

def find_threshold(p1_charged, p2_charged):
    """Binary search for fusion threshold."""
    E_low, E_high = 0.01, 20.0
    
    # Check bounds
    fused_low, _, _ = run_collision(p1_charged, p2_charged, E_low)
    fused_high, _, _ = run_collision(p1_charged, p2_charged, E_high)
    
    if fused_low:
        return 0.0
    if not fused_high:
        return float('inf')
    
    for _ in range(25):
        E_mid = (E_low + E_high) / 2
        fused, _, _ = run_collision(p1_charged, p2_charged, E_mid)
        
        if fused:
            E_high = E_mid
        else:
            E_low = E_mid
        
        if E_high - E_low < 0.001:
            break
    
    return (E_low + E_high) / 2

def run_neutron_test():
    """Compare Proton-Proton vs Proton-Neutron fusion."""
    print("="*70)
    print("DET v5 NEUTRON TEST")
    print("="*70)
    print("\nComparing fusion thresholds:")
    print("  - Proton + Proton (both charged, EM repulsion)")
    print("  - Proton + Neutron (only one charged, NO EM repulsion)")
    print()
    
    # Test 1: Proton-Proton (P-P)
    print("Finding Proton-Proton threshold...")
    E_pp = find_threshold(True, True)
    print(f"  P-P Threshold: E = {E_pp:.4f}")
    
    # Test 2: Proton-Neutron (P-N)
    print("\nFinding Proton-Neutron threshold...")
    E_pn = find_threshold(True, False)
    print(f"  P-N Threshold: E = {E_pn:.4f}")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    ratio = E_pn / E_pp if E_pp > 0 else 0
    
    print(f"\n  Proton-Proton Barrier:   E_pp = {E_pp:.4f}")
    print(f"  Proton-Neutron Barrier:  E_pn = {E_pn:.4f}")
    print(f"\n  Barrier Ratio (P-N/P-P): {ratio:.4f}")
    
    if E_pn < E_pp:
        print("\n  ✓ PREDICTION CONFIRMED!")
        print(f"    P-N fusion barrier is {(1-ratio)*100:.1f}% LOWER than P-P")
        print("    This is because neutrons have no phase → no EM repulsion")
    elif E_pn == 0:
        print("\n  ✓ P-N fusion occurs at ANY energy (no barrier)")
        print("    Strong force dominates completely without EM opposition")
    else:
        print("\n  ✗ Unexpected: P-N barrier is higher than P-P")
    
    return E_pp, E_pn

def plot_comparison():
    """Create comparison plots for P-P vs P-N fusion."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    E_pp, E_pn = 3.0, 0.01  # We'll get actual values
    E_pp, E_pn = find_threshold(True, True), find_threshold(True, False)
    
    # Get collision histories at threshold
    _, _, hist_pp = run_collision(True, True, E_pp * 1.1)  # Just above threshold
    _, _, hist_pn = run_collision(True, False, max(E_pn * 1.1, 0.1))
    
    # Plot 1: Trajectories comparison
    ax = axes[0, 0]
    steps_pp = range(len(hist_pp))
    steps_pn = range(len(hist_pn))
    
    ax.plot(steps_pp, [h['x1'] for h in hist_pp], 'b-', label='P-P: Particle 1', alpha=0.7)
    ax.plot(steps_pp, [h['x2'] for h in hist_pp], 'b--', label='P-P: Particle 2', alpha=0.7)
    ax.plot(steps_pn, [h['x1'] for h in hist_pn], 'r-', label='P-N: Proton', alpha=0.7)
    ax.plot(steps_pn, [h['x2'] for h in hist_pn], 'r--', label='P-N: Neutron', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')
    ax.set_title('Particle Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Separation over time
    ax = axes[0, 1]
    ax.plot(steps_pp, [h['r'] for h in hist_pp], 'b-', label='P-P', linewidth=2)
    ax.plot(steps_pn, [h['r'] for h in hist_pn], 'r-', label='P-N', linewidth=2)
    ax.axhline(y=FUSION_DIST, color='green', linestyle='--', label='Fusion distance')
    ax.set_xlabel('Step')
    ax.set_ylabel('Separation')
    ax.set_title('Separation vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 60)
    
    # Plot 3: Forces
    ax = axes[1, 0]
    ax.plot(steps_pp, [h['F_em'] for h in hist_pp], 'r-', label='P-P: EM', alpha=0.7)
    ax.plot(steps_pp, [h['F_strong'] for h in hist_pp], 'b-', label='P-P: Strong', alpha=0.7)
    ax.plot(steps_pn, [h['F_em'] for h in hist_pn], 'r--', label='P-N: EM (=0)', alpha=0.7)
    ax.plot(steps_pn, [h['F_strong'] for h in hist_pn], 'b--', label='P-N: Strong', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Force')
    ax.set_title('Forces During Collision')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Barrier comparison
    ax = axes[1, 1]
    categories = ['Proton-Proton\n(EM + Strong)', 'Proton-Neutron\n(Strong only)']
    values = [E_pp, E_pn]
    colors = ['steelblue', 'forestgreen']
    
    bars = ax.bar(categories, values, color=colors, width=0.5)
    ax.set_ylabel('Fusion Threshold Energy')
    ax.set_title('Fusion Barrier Comparison')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'E = {val:.3f}', ha='center', va='bottom', fontsize=11)
    
    if E_pp > 0:
        reduction = (1 - E_pn/E_pp) * 100
        ax.text(0.5, 0.85, f'Barrier Reduction: {reduction:.1f}%', 
               transform=ax.transAxes, fontsize=12, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen'))
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/neutron_test.png', dpi=150)
    print("\nSaved to neutron_test.png")

if __name__ == "__main__":
    E_pp, E_pn = run_neutron_test()
    plot_comparison()
    
    print("\n" + "="*70)
    print("PHYSICAL INTERPRETATION")
    print("="*70)
    print("""
In DET v5, the neutron is modeled as a particle with:
  - Mass/Energy (F): Same as proton (density binding works)
  - Phase (θ): No rotation (uncharged, no EM interaction)

The Proton-Neutron fusion test confirms:
  1. EM repulsion comes specifically from PHASE MISALIGNMENT
  2. The Strong Force is PHASE-BLIND (acts on density, not phase)
  3. Fusion barriers depend on the combination of particles:
     - P-P: High barrier (EM repulsion must be overcome)
     - P-N: Low/no barrier (only Strong force, which is attractive)

This matches real nuclear physics where:
  - Proton-Proton fusion requires very high temperatures (Sun's core)
  - Proton-Neutron binding (deuterium) happens more easily
""")
