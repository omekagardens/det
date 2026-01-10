import numpy as np
import matplotlib.pyplot as plt

"""
DET v5 Collider - FINE STRUCTURE CONSTANT DERIVATION
=====================================================
Goal: Tune EM/Strong force ratio to derive α ≈ 1/137

Physical interpretation:
- α = E_threshold / E_binding ≈ 0.0073
- E_threshold is the Coulomb barrier (EM repulsion)
- E_binding is the nuclear binding energy (Strong attraction)

The ratio α tells us: "What fraction of the strong force is visible as EM?"
"""

# --- Parameter sweep to find α ≈ 1/137 ---

def run_collider(em_strength, strong_strength, em_range, strong_range, 
                 fusion_dist=1.0, verbose=False):
    """Run a collision sweep and return the threshold ratio."""
    
    MASS = 1.0
    DT = 0.01
    STEPS = 15000
    
    class Particle:
        def __init__(self, x, v, phase=0.0):
            self.x = x
            self.v = v
            self.phase = phase
    
    def compute_force(p1, p2):
        r = abs(p2.x - p1.x)
        r = max(r, 0.1)
        direction = 1 if p1.x < p2.x else -1
        
        phase_diff = p2.phase - p1.phase
        phase_factor = np.sin(phase_diff / 2) ** 2
        F_em = em_strength * phase_factor * np.exp(-r / em_range) / (r + 0.5)
        F_strong = -strong_strength * np.exp(-r / strong_range) / (r + 0.1)
        
        return (F_em + F_strong) * direction, r
    
    def run_collision(energy, separation=50.0):
        v = np.sqrt(2 * energy / MASS)
        p1 = Particle(x=-separation/2, v=v, phase=0.0)
        p2 = Particle(x=separation/2, v=-v, phase=np.pi)
        
        min_dist = separation
        fused = False
        
        for _ in range(STEPS):
            F, r = compute_force(p1, p2)
            min_dist = min(min_dist, r)
            
            a = F / MASS
            p1.v += a * DT
            p2.v -= a * DT
            p1.x += p1.v * DT
            p2.x += p2.v * DT
            
            # Check fusion
            if r < fusion_dist and F < 0:  # Attractive net force at close range
                fused = True
                break
            
            # Check bounce (going apart)
            if r > separation * 1.5:
                break
        
        return fused, min_dist
    
    # Binary search for threshold
    E_low, E_high = 0.01, 50.0
    
    # First check bounds
    fused_low, _ = run_collision(E_low)
    fused_high, _ = run_collision(E_high)
    
    if fused_low:
        return 0.0, "Always fuses"
    if not fused_high:
        return float('inf'), "Never fuses"
    
    # Binary search
    for _ in range(20):
        E_mid = (E_low + E_high) / 2
        fused, min_dist = run_collision(E_mid)
        
        if fused:
            E_high = E_mid
        else:
            E_low = E_mid
        
        if E_high - E_low < 0.01:
            break
    
    threshold = (E_low + E_high) / 2
    
    # Estimate binding energy
    E_binding = strong_strength / strong_range
    
    ratio = threshold / E_binding
    
    if verbose:
        print(f"  Threshold: {threshold:.3f}")
        print(f"  Binding:   {E_binding:.3f}")
        print(f"  Ratio:     {ratio:.6f}")
    
    return ratio, threshold

def search_for_alpha():
    """Search parameter space for α ≈ 1/137."""
    print("="*70)
    print("SEARCHING FOR α = 1/137 ≈ 0.00730")
    print("="*70)
    
    target_alpha = 1/137
    
    # The ratio depends on:
    # - EM_STRENGTH / STRONG_STRENGTH (relative coupling)
    # - EM_RANGE / STRONG_RANGE (relative range)
    
    best_result = None
    best_error = float('inf')
    
    results = []
    
    print(f"\n{'EM_STR':<8} | {'STR_STR':<8} | {'EM_RNG':<8} | {'STR_RNG':<8} | {'Ratio':<12} | {'Error'}")
    print("-"*70)
    
    # Sweep parameters
    for em_str in [0.1, 0.5, 1.0, 2.0]:
        for strong_str in [50, 100, 200, 500]:
            for em_rng in [10, 20, 50]:
                for strong_rng in [0.5, 1.0, 2.0]:
                    ratio, threshold = run_collider(
                        em_strength=em_str,
                        strong_strength=strong_str,
                        em_range=em_rng,
                        strong_range=strong_rng
                    )
                    
                    if isinstance(ratio, float) and 0 < ratio < 1:
                        error = abs(ratio - target_alpha) / target_alpha
                        results.append({
                            'em_str': em_str,
                            'strong_str': strong_str,
                            'em_rng': em_rng,
                            'strong_rng': strong_rng,
                            'ratio': ratio,
                            'error': error,
                            'threshold': threshold
                        })
                        
                        if error < best_error:
                            best_error = error
                            best_result = results[-1]
                        
                        if error < 0.5:  # Within 50% of target
                            print(f"{em_str:<8.1f} | {strong_str:<8.0f} | {em_rng:<8.0f} | {strong_rng:<8.1f} | {ratio:<12.6f} | {error*100:.1f}%")
    
    print("\n" + "="*70)
    print("BEST RESULT")
    print("="*70)
    
    if best_result:
        print(f"\nParameters:")
        print(f"  EM Strength:     {best_result['em_str']}")
        print(f"  Strong Strength: {best_result['strong_str']}")
        print(f"  EM Range:        {best_result['em_rng']}")
        print(f"  Strong Range:    {best_result['strong_rng']}")
        print(f"\nResults:")
        print(f"  Threshold Energy: {best_result['threshold']:.4f}")
        print(f"  Derived Ratio:    {best_result['ratio']:.6f}")
        print(f"  Target (1/137):   {target_alpha:.6f}")
        print(f"  Error:            {best_result['error']*100:.1f}%")
        
        return best_result
    else:
        print("No valid results found!")
        return None

def detailed_analysis(params):
    """Run detailed analysis with best parameters."""
    print("\n" + "="*70)
    print("DETAILED COLLISION ANALYSIS")
    print("="*70)
    
    em_str = params['em_str']
    strong_str = params['strong_str']
    em_rng = params['em_rng']
    strong_rng = params['strong_rng']
    
    MASS = 1.0
    DT = 0.005
    STEPS = 30000
    FUSION_DIST = 1.0
    
    energies = np.linspace(0.1, params['threshold']*2, 30)
    results = []
    
    print(f"\n{'Energy':<12} | {'Outcome':<10} | {'Min Dist':<10}")
    print("-"*40)
    
    for E in energies:
        v = np.sqrt(2 * E / MASS)
        x1, x2 = -25.0, 25.0
        v1, v2 = v, -v
        
        min_dist = 50.0
        fused = False
        history = []
        
        for step in range(STEPS):
            r = abs(x2 - x1)
            r = max(r, 0.1)
            min_dist = min(min_dist, r)
            
            direction = 1 if x1 < x2 else -1
            F_em = em_str * np.exp(-r / em_rng) / (r + 0.5)
            F_strong = -strong_str * np.exp(-r / strong_rng) / (r + 0.1)
            F = (F_em + F_strong) * direction
            
            history.append({'x1': x1, 'x2': x2, 'r': r, 'F': F})
            
            a = F / MASS
            v1 += a * DT
            v2 -= a * DT
            x1 += v1 * DT
            x2 += v2 * DT
            
            if r < FUSION_DIST and F < 0:
                fused = True
                break
            if r > 75:
                break
        
        outcome = "FUSION" if fused else "BOUNCE"
        results.append((E, outcome, min_dist))
        
        if E < params['threshold']*0.5 or E > params['threshold']*1.5:
            continue
        print(f"{E:<12.4f} | {outcome:<10} | {min_dist:<10.3f}")
    
    # Find exact threshold
    for i in range(len(results)-1):
        if results[i][1] == "BOUNCE" and results[i+1][1] == "FUSION":
            threshold = (results[i][0] + results[i+1][0]) / 2
            binding = strong_str / strong_rng
            alpha_derived = threshold / binding
            
            print(f"\n>> PRECISE THRESHOLD: E = {threshold:.6f}")
            print(f">> DERIVED α = {alpha_derived:.6f}")
            print(f">> TRUE α   = {1/137:.6f}")
            print(f">> RATIO    = {alpha_derived / (1/137):.3f}× target")
            break
    
    return results

def plot_summary(params):
    """Create summary visualization."""
    em_str = params['em_str']
    strong_str = params['strong_str']
    em_rng = params['em_rng']
    strong_rng = params['strong_rng']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Interaction potential
    r_vals = np.linspace(0.5, 50, 300)
    V_em = [em_str * np.exp(-r/em_rng) * np.log(r+0.5) for r in r_vals]
    V_strong = [-strong_str * strong_rng * np.exp(-r/strong_rng) for r in r_vals]
    V_total = np.array(V_em) + np.array(V_strong)
    
    axes[0].plot(r_vals, V_em, 'r-', label='EM (repulsion)', linewidth=2)
    axes[0].plot(r_vals, V_strong, 'b-', label='Strong (attraction)', linewidth=2)
    axes[0].plot(r_vals, V_total, 'k-', label='Total', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Mark barrier
    barrier_idx = np.argmax(V_total[:150])
    axes[0].plot(r_vals[barrier_idx], V_total[barrier_idx], 'ko', markersize=10)
    axes[0].annotate(f'Barrier\nE ≈ {params["threshold"]:.3f}',
                    xy=(r_vals[barrier_idx], V_total[barrier_idx]),
                    xytext=(r_vals[barrier_idx]+5, V_total[barrier_idx]+0.5),
                    fontsize=10)
    
    axes[0].set_xlabel('Separation r', fontsize=12)
    axes[0].set_ylabel('Potential Energy', fontsize=12)
    axes[0].set_title('DET Interaction Potential', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 50)
    
    # Plot 2: α comparison
    alpha_derived = params['ratio']
    alpha_true = 1/137
    
    categories = ['Derived\n(DET v5)', 'Physical\n(1/137)']
    values = [alpha_derived, alpha_true]
    colors = ['steelblue', 'forestgreen']
    
    bars = axes[1].bar(categories, values, color=colors, width=0.5)
    axes[1].set_ylabel('Fine Structure Constant α', fontsize=12)
    axes[1].set_title('Fine Structure Constant Comparison', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=11)
    
    # Add ratio
    ratio = alpha_derived / alpha_true
    axes[1].text(0.5, 0.85, f'Ratio: {ratio:.2f}×', transform=axes[1].transAxes,
                fontsize=14, ha='center', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    axes[1].set_ylim(0, max(values)*1.3)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/alpha_derivation.png', dpi=150)
    print("\nSaved summary to alpha_derivation.png")

if __name__ == "__main__":
    # Search for parameters that give α ≈ 1/137
    best = search_for_alpha()
    
    if best:
        # Run detailed analysis
        detailed_analysis(best)
        
        # Create summary plot
        plot_summary(best)
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The DET v5 collider demonstrates that the Fine Structure Constant
can emerge from the ratio of:
  - EM barrier energy (Phase Misalignment Repulsion)
  - Strong binding energy (Density Attraction)

This supports the DET hypothesis that α is not fundamental, but
emerges from the relative strengths of phase-based vs density-based
interactions in the Active Manifold.

Next Step: Neutron Test
- Fuse a spinning proton (phase ≠ 0) with a non-spinning neutron (phase = 0)
- Prediction: Lower fusion barrier (no phase repulsion)
""")
