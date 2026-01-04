"""
DET Falsifier Tests: F1 (Boundary Redundancy) and F2 (Coercion Violation)
=========================================================================
"""

import numpy as np
from det_core import *

def test_F1_boundary_redundancy(n_trials: int = 10, n_nodes: int = 25, steps: int = 500) -> Dict:
    """
    F1: Boundary Redundancy Test
    
    FALSIFIED IF: Systems with boundary operators show NO robust qualitative 
    difference from systems without them.
    
    PASSES IF: Boundary operators produce measurable, significant effects
    on coherence, presence, and structural debt.
    """
    print("\n" + "="*70)
    print("F1: BOUNDARY REDUNDANCY TEST")
    print("="*70)
    
    coherence_with = []
    coherence_without = []
    presence_with = []
    presence_without = []
    
    for trial in range(n_trials):
        # Create identical stressed systems
        sys_with = create_system(n_nodes, conn=0.35, seed=1000+trial)
        sys_without = copy_system(sys_with)
        
        # Stress both systems
        for i in range(n_nodes // 3):
            sys_with.creatures[i].F = 0.05
            sys_with.creatures[i].q = 0.4
            sys_without.creatures[i].F = 0.05
            sys_without.creatures[i].q = 0.4
        
        # Run with boundary operators
        for _ in range(steps):
            step_system(sys_with, dt=0.1, boundary=True)
        
        # Run without boundary operators
        for _ in range(steps):
            step_system(sys_without, dt=0.1, boundary=False)
        
        coherence_with.append(np.mean([b.C_ij for b in sys_with.bonds]))
        coherence_without.append(np.mean([b.C_ij for b in sys_without.bonds]))
        presence_with.append(np.mean([compute_presence(c) for c in sys_with.creatures]))
        presence_without.append(np.mean([compute_presence(c) for c in sys_without.creatures]))
    
    # Compute effect size (Cohen's d)
    C_with_mean, C_with_std = np.mean(coherence_with), np.std(coherence_with)
    C_without_mean, C_without_std = np.mean(coherence_without), np.std(coherence_without)
    pooled_std = np.sqrt((C_with_std**2 + C_without_std**2) / 2)
    cohens_d = (C_with_mean - C_without_mean) / max(pooled_std, 1e-10)
    
    print(f"\nResults ({n_trials} trials, {steps} steps each):")
    print(f"  Coherence WITH boundary:    {C_with_mean:.4f} ± {C_with_std:.4f}")
    print(f"  Coherence WITHOUT boundary: {C_without_mean:.4f} ± {C_without_std:.4f}")
    print(f"  Cohen's d effect size:      {cohens_d:.2f}")
    
    P_with_mean = np.mean(presence_with)
    P_without_mean = np.mean(presence_without)
    print(f"  Presence WITH:    {P_with_mean:.4f}")
    print(f"  Presence WITHOUT: {P_without_mean:.4f}")
    
    # Pass if Cohen's d > 0.5 (medium effect) or > 0.8 (large effect)
    passed = abs(cohens_d) > 0.5
    
    if passed:
        print(f"\n✓ F1 PASSED: Boundary operators produce {'large' if abs(cohens_d) > 0.8 else 'medium'} effect (d={cohens_d:.2f})")
    else:
        print(f"\n✗ F1 FAILED: Boundary operators show no significant effect (d={cohens_d:.2f})")
    
    return {'passed': passed, 'cohens_d': cohens_d, 'C_with': C_with_mean, 'C_without': C_without_mean}


def test_F2_coercion_violation(n_trials: int = 5, steps: int = 300) -> Dict:
    """
    F2: Coercion Violation Test
    
    FALSIFIED IF: Any node with a=0 receives grace injection (I > 0) 
    or bond healing (ΔC > 0).
    
    PASSES IF: Zero violations occur - agency inviolability is maintained.
    """
    print("\n" + "="*70)
    print("F2: COERCION VIOLATION TEST")
    print("="*70)
    
    total_checks = 0
    grace_violations = []
    coherence_violations = []
    
    for trial in range(n_trials):
        sys = create_system(20, conn=0.4, seed=2000+trial)
        
        # Set some nodes to a=0 and make them needy
        zero_agency_nodes = [0, 5, 10]
        for i in zero_agency_nodes:
            sys.creatures[i].a = 0.0
            sys.creatures[i].F = 0.01  # Very needy
        
        for step in range(steps):
            metrics = step_system(sys, dt=0.1, boundary=True)
            
            # Check grace to a=0 nodes
            for i in zero_agency_nodes:
                total_checks += 1
                I_g = metrics['injections'].get(i, 0.0)
                if I_g > 1e-12:
                    grace_violations.append({'trial': trial, 'step': step, 'node': i, 'I_g': I_g})
            
            # Check coherence healing on bonds involving a=0 nodes
            for b in sys.bonds:
                if sys.creatures[b.i].a < 1e-10 or sys.creatures[b.j].a < 1e-10:
                    total_checks += 1
                    # We need to track ΔC, but since reconciliation returns 0 for a=0, this should be fine
                    # Check by looking at the reconciliation value
                    key = (b.i, b.j)
                    delta_C = metrics.get('reconciliations', {}).get(key, 0.0) if 'reconciliations' in metrics else 0.0
                    # Actually metrics doesn't include reconciliations directly, let's recompute
    
    # Re-run with explicit reconciliation tracking
    total_checks = 0
    grace_violations = []
    coherence_violations = []
    
    for trial in range(n_trials):
        sys = create_system(20, conn=0.4, seed=2000+trial)
        
        zero_agency_nodes = [0, 5, 10]
        for i in zero_agency_nodes:
            sys.creatures[i].a = 0.0
            sys.creatures[i].F = 0.01
        
        for step in range(steps):
            # Compute flows and dissipations manually to check reconciliation
            flows = {(min(b.i, b.j), max(b.i, b.j)): compute_flow(sys, b) for b in sys.bonds}
            dissipations = {i: compute_dissipation(sys, i, flows, 0.1) for i in range(sys.n)}
            
            # Check grace
            for i in zero_agency_nodes:
                total_checks += 1
                I_g = compute_grace(sys, i, dissipations)
                if I_g > 1e-12:
                    grace_violations.append({'trial': trial, 'step': step, 'node': i, 'I_g': I_g, 'a': sys.creatures[i].a})
            
            # Check reconciliation on bonds with a=0
            for b in sys.bonds:
                if sys.creatures[b.i].a < 1e-10 or sys.creatures[b.j].a < 1e-10:
                    total_checks += 1
                    delta_C = compute_reconciliation(sys, b, dissipations)
                    if delta_C > 1e-12:
                        coherence_violations.append({'trial': trial, 'step': step, 'bond': (b.i, b.j), 'delta_C': delta_C})
            
            step_system(sys, dt=0.1, boundary=True)
    
    n_grace = len(grace_violations)
    n_coherence = len(coherence_violations)
    
    print(f"\nResults ({n_trials} trials, {steps} steps each):")
    print(f"  Total checks: {total_checks}")
    print(f"  Grace violations (I>0 at a=0): {n_grace}")
    print(f"  Coherence violations (ΔC>0 with a=0): {n_coherence}")
    
    if grace_violations:
        print("\n  Sample grace violations:")
        for v in grace_violations[:3]:
            print(f"    Trial {v['trial']}, Step {v['step']}: node {v['node']}, I_g={v['I_g']:.2e}, a={v['a']:.4f}")
    
    passed = (n_grace == 0) and (n_coherence == 0)
    
    if passed:
        print(f"\n✓ F2 PASSED: 0/{total_checks} coercion violations")
    else:
        print(f"\n✗ F2 FAILED: {n_grace + n_coherence} violations detected")
    
    return {'passed': passed, 'total_checks': total_checks, 'grace_violations': n_grace, 'coherence_violations': n_coherence}


if __name__ == "__main__":
    print("="*70)
    print("DET FALSIFICATION SUITE: F1-F2")
    print("="*70)
    
    results = {}
    results['F1'] = test_F1_boundary_redundancy()
    results['F2'] = test_F2_coercion_violation()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for test, r in results.items():
        status = "✓ PASSED" if r['passed'] else "✗ FAILED"
        print(f"{test}: {status}")
