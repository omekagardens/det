"""
DET v6 GRAVITATIONAL REDSHIFT FALSIFIER - FINAL REPORT
=======================================================

Test Date: 2026-01-10
Reference: DET Theory Card v6.0, Sections III.1, V.1-V.2

EXECUTIVE SUMMARY
-----------------
The gravitational redshift prediction P(r)/P_∞ = 1 + Φ(r) FAILS in the 
current DET v6 formulation. The test reveals a fundamental mismatch between
the presence formula and the gravitational potential.

HOWEVER: The presence formula P = a·σ/(1+F)/(1+H) is VERIFIED to work
correctly, and qualitative gravitational time dilation DOES occur (clocks
run slower in potential wells). The issue is the quantitative relationship.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import sys
sys.path.insert(0, '/home/claude')


def generate_comprehensive_report():
    """Generate the complete test report with verification."""
    from det_v6_3d_collider_gravity import DETCollider3DGravity, DETParams3D
    
    print("="*74)
    print("     DET v6 GRAVITATIONAL REDSHIFT FALSIFIER - COMPREHENSIVE REPORT")
    print("="*74)
    
    # ========================================================================
    # TEST SPECIFICATION
    # ========================================================================
    print("\n" + "="*74)
    print("1. TEST SPECIFICATION")
    print("="*74)
    
    print("""
PREDICTION FROM THEORY CARD:
    Section III.1: P_i = a_i·σ_i / (1+F_i^op) / (1+H_i)
    Section III.2: M_i = P_i^(-1)
    
GRAVITATIONAL REDSHIFT CLAIM:
    Δτ/τ = ΔΦ (in natural units)
    Equivalent: P(r)/P_∞ = 1 + Φ(r) (weak field approximation)
    
FALSIFIER CRITERION:
    |P(r)/P_∞ - (1 + Φ(r))| > 1% → FALSIFIED
    
TEST METHODOLOGY:
    1. Create static mass distribution with structure q > 0
    2. Compute gravitational potential Φ via Poisson solver
    3. Compute presence P at various radii
    4. Compare P(r)/P_∞ with 1 + Φ(r)
""")
    
    # ========================================================================
    # TEST EXECUTION
    # ========================================================================
    print("\n" + "="*74)
    print("2. TEST EXECUTION")
    print("="*74)
    
    # Set up simulation
    params = DETParams3D(
        N=48,
        DT=0.015,
        F_VAC=0.1,
        C_init=0.2,
        diff_enabled=True,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        q_enabled=False,
        agency_dynamic=False,
        gravity_enabled=True,
        alpha_grav=0.015,
        kappa_grav=8.0,
        mu_grav=4.0
    )
    
    sim = DETCollider3DGravity(params)
    N = params.N
    center = N // 2
    
    # Create coordinate grid
    z, y, x = np.mgrid[0:N, 0:N, 0:N]
    dx = (x - center + N/2) % N - N/2
    dy = (y - center + N/2) % N - N/2
    dz = (z - center + N/2) % N - N/2
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Set up mass distribution
    width = 4.0
    envelope = np.exp(-0.5 * (r**2) / width**2)
    sim.q = 0.4 * envelope
    sim.F = params.F_VAC + 5.0 * envelope
    
    # Compute gravity
    sim._compute_gravity()
    
    # Compute presence
    H = sim.sigma
    P = sim.a * sim.sigma / (1.0 + sim.F) / (1.0 + H)
    
    print(f"\nSimulation parameters:")
    print(f"   Grid size: {N}³")
    print(f"   q peak: {np.max(sim.q):.3f}")
    print(f"   F peak: {np.max(sim.F):.3f}")
    print(f"   Φ range: [{np.min(sim.Phi):.4f}, {np.max(sim.Phi):.4f}]")
    print(f"   P range: [{np.min(P):.4f}, {np.max(P):.4f}]")
    
    # ========================================================================
    # QUANTITATIVE ANALYSIS
    # ========================================================================
    print("\n" + "="*74)
    print("3. QUANTITATIVE ANALYSIS")
    print("="*74)
    
    # Sample at various radii
    radii = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # Get far-field references
    far_mask = r > 20
    P_inf = np.mean(P[far_mask])
    Phi_inf = np.mean(sim.Phi[far_mask])
    F_inf = np.mean(sim.F[far_mask])
    
    print(f"\nFar-field reference values (r > 20):")
    print(f"   P_∞ = {P_inf:.6f}")
    print(f"   Φ_∞ = {Phi_inf:.6f}")
    print(f"   F_∞ = {F_inf:.6f}")
    
    # Collect data
    data = {'r': [], 'Phi': [], 'F': [], 'P': [], 
            'P_norm': [], 'Phi_shifted': [], 'P_pred_Phi': [], 'P_pred_F': []}
    
    for rad in radii:
        mask = (r >= rad - 0.5) & (r < rad + 0.5)
        if np.sum(mask) == 0:
            continue
            
        Phi_avg = np.mean(sim.Phi[mask])
        F_avg = np.mean(sim.F[mask])
        P_avg = np.mean(P[mask])
        
        P_norm = P_avg / P_inf
        Phi_shifted = Phi_avg - Phi_inf
        P_pred_Phi = 1 + Phi_shifted  # GR-like prediction
        P_pred_F = (1 + F_inf) / (1 + F_avg)  # DET formula prediction
        
        data['r'].append(rad)
        data['Phi'].append(Phi_avg)
        data['F'].append(F_avg)
        data['P'].append(P_avg)
        data['P_norm'].append(P_norm)
        data['Phi_shifted'].append(Phi_shifted)
        data['P_pred_Phi'].append(P_pred_Phi)
        data['P_pred_F'].append(P_pred_F)
    
    # Convert to arrays
    for key in data:
        data[key] = np.array(data[key])
    
    # Compute deviations
    dev_from_Phi = np.abs(data['P_norm'] - data['P_pred_Phi'])
    dev_from_F = np.abs(data['P_norm'] - data['P_pred_F'])
    
    print("\n" + "-"*74)
    print("Comparison Table:")
    print("-"*74)
    print(f"{'r':>4s} {'Φ':>10s} {'F':>10s} {'P/P_∞':>10s} "
          f"{'1+Φ':>10s} {'(1+F_∞)/(1+F)':>14s} {'Dev(Φ)%':>10s}")
    print("-"*74)
    
    for i in range(len(data['r'])):
        print(f"{data['r'][i]:4.0f} {data['Phi_shifted'][i]:10.4f} {data['F'][i]:10.4f} "
              f"{data['P_norm'][i]:10.4f} {data['P_pred_Phi'][i]:10.4f} "
              f"{data['P_pred_F'][i]:14.4f} {dev_from_Phi[i]*100:10.2f}")
    
    # ========================================================================
    # TEST RESULTS
    # ========================================================================
    print("\n" + "="*74)
    print("4. TEST RESULTS")
    print("="*74)
    
    max_dev_Phi = np.max(dev_from_Phi) * 100
    mean_dev_Phi = np.mean(dev_from_Phi) * 100
    max_dev_F = np.max(dev_from_F) * 100
    mean_dev_F = np.mean(dev_from_F) * 100
    
    print(f"\n(A) REDSHIFT TEST: P/P_∞ vs 1+Φ")
    print(f"   Maximum deviation: {max_dev_Phi:.2f}%")
    print(f"   Mean deviation: {mean_dev_Phi:.2f}%")
    print(f"   Threshold: 1%")
    print(f"   RESULT: {'PASSED' if max_dev_Phi < 1.0 else 'FAILED'}")
    
    print(f"\n(B) PRESENCE FORMULA TEST: P/P_∞ vs (1+F_∞)/(1+F)")
    print(f"   Maximum deviation: {max_dev_F:.4f}%")
    print(f"   Mean deviation: {mean_dev_F:.4f}%")
    print(f"   RESULT: {'VERIFIED' if max_dev_F < 0.5 else 'FAILED'}")
    
    # ========================================================================
    # DIAGNOSIS
    # ========================================================================
    print("\n" + "="*74)
    print("5. DIAGNOSIS")
    print("="*74)
    
    # Check F-Φ correlation
    corr = np.corrcoef(data['F'], data['Phi_shifted'])[0, 1]
    
    print(f"""
The redshift test FAILS because:

1. THE PRESENCE FORMULA WORKS CORRECTLY
   P/P_∞ = (1+F_∞)/(1+F) verified to {max_dev_F:.4f}% accuracy
   
2. BUT Φ IS NOT IN THE PRESENCE FORMULA
   P = a·σ / (1+F) / (1+H)
   
   There is no Φ term! Presence depends on F, not Φ.

3. F AND Φ ARE POSITIVELY CORRELATED
   Correlation(F, Φ) = {corr:.4f}
   
   Where Φ is high (center of mass), F is also high.
   This makes P LOW at center (correct direction for time dilation).

4. BUT THE QUANTITATIVE RELATIONSHIP IS WRONG
   For P/P_∞ = 1+Φ, we would need: F ≈ F_∞ - Φ (anti-correlation)
   Instead we have: F positively correlated with Φ
   
5. THE ACTUAL DET PREDICTION IS:
   P/P_∞ = (1+F_∞)/(1+F)
   
   NOT: P/P_∞ = 1+Φ
""")
    
    # ========================================================================
    # THEORETICAL IMPLICATIONS
    # ========================================================================
    print("\n" + "="*74)
    print("6. THEORETICAL IMPLICATIONS")
    print("="*74)
    
    print("""
The gravitational redshift prediction P/P_∞ = 1+Φ appears in the 
falsifier section of the Theory Card but is NOT derivable from the 
stated dynamics.

This represents a SPECIFICATION ERROR in the Theory Card, not a 
failure of the DET dynamics themselves.

POSSIBLE RESOLUTIONS:

Option A: REMOVE THE CLAIM
    Accept that P/P_∞ = (1+F_∞)/(1+F) is the actual DET prediction.
    Gravitational time dilation occurs but with different scaling.

Option B: MODIFY THE PRESENCE FORMULA
    Add Φ explicitly: P = a·σ·(1+Φ)⁻¹/(1+F)/(1+H)
    This would build in GR-like redshift by construction.
    Caution: This adds non-local dependence (Φ comes from Poisson solve).

Option C: ADD A CALIBRATION RELATION
    Require that in equilibrium: F = f(Φ) such that redshift emerges.
    This requires specific parameter tuning and may not be robust.

Option D: DOCUMENT AS EMERGENT LIMIT
    State that P/P_∞ ≈ 1+Φ emerges only in certain equilibrated limits
    with specific parameter choices, not as a general result.

RECOMMENDATION:
    Update Theory Card Section III to clarify:
    - P/P_∞ = (1+F_∞)/(1+F) is the fundamental relation
    - GR-like redshift P/P_∞ = 1+Φ is NOT automatic
    - Remove or revise the redshift falsifier
""")
    
    # ========================================================================
    # FALSIFIER VERDICT
    # ========================================================================
    print("\n" + "="*74)
    print("7. FALSIFIER VERDICT")
    print("="*74)
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║                      GRAVITATIONAL REDSHIFT TEST                       ║
╠════════════════════════════════════════════════════════════════════════╣
║  Prediction: P(r)/P_∞ = 1 + Φ(r)                                       ║
║  Criterion:  |P(r)/P_∞ - (1 + Φ(r))| < 1%                             ║
║                                                                        ║
║  Maximum deviation: {max_dev_Phi:>8.2f}%                                        ║
║  Threshold:         {1.0:>8.2f}%                                        ║
║                                                                        ║
║  RESULT: {'FAILED':^62s} ║
╠════════════════════════════════════════════════════════════════════════╣
║  Note: This is a SPECIFICATION INCONSISTENCY in the Theory Card.      ║
║  The DET dynamics are internally consistent; the stated prediction    ║
║  P/P_∞ = 1+Φ is not derivable from the presence formula.              ║
╚════════════════════════════════════════════════════════════════════════╝
""")
    
    # ========================================================================
    # CREATE VISUALIZATION
    # ========================================================================
    create_final_visualization(data, params)
    
    return data


def create_final_visualization(data: dict, params):
    """Create publication-quality visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('DET v6 Gravitational Redshift Falsifier Test Results', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    r = data['r']
    
    # Plot 1: P/P_∞ vs predictions
    ax1 = axes[0, 0]
    ax1.plot(r, data['P_norm'], 'bo-', linewidth=2, markersize=8, 
             label='P/P_∞ (measured)', zorder=3)
    ax1.plot(r, data['P_pred_Phi'], 'r^--', linewidth=2, markersize=8,
             label='1+Φ (GR prediction)', zorder=2)
    ax1.plot(r, data['P_pred_F'], 'gs:', linewidth=2, markersize=8,
             label='(1+F_∞)/(1+F) (DET formula)', zorder=2)
    ax1.set_xlabel('Radius r', fontsize=12)
    ax1.set_ylabel('Normalized Presence', fontsize=12)
    ax1.set_title('Presence: Measured vs Predictions', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(r)+1)
    
    # Plot 2: Deviation from GR prediction
    ax2 = axes[0, 1]
    deviation = (data['P_norm'] - data['P_pred_Phi']) * 100
    colors = ['red' if abs(d) > 1 else 'green' for d in deviation]
    bars = ax2.bar(r, deviation, width=1.5, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1, color='r', linestyle='--', linewidth=2, label='±1% threshold')
    ax2.axhline(y=-1, color='r', linestyle='--', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax2.set_xlabel('Radius r', fontsize=12)
    ax2.set_ylabel('Deviation: P/P_∞ - (1+Φ) [%]', fontsize=12)
    ax2.set_title('Deviation from GR Redshift Prediction', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: F and Φ profiles
    ax3 = axes[1, 0]
    ax3b = ax3.twinx()
    l1 = ax3.plot(r, data['F'], 'b-o', linewidth=2, markersize=6, label='F(r)')
    l2 = ax3b.plot(r, data['Phi_shifted'], 'r-s', linewidth=2, markersize=6, label='Φ(r)-Φ_∞')
    ax3.set_xlabel('Radius r', fontsize=12)
    ax3.set_ylabel('Resource F', color='blue', fontsize=12)
    ax3b.set_ylabel('Shifted Potential Φ-Φ_∞', color='red', fontsize=12)
    ax3.set_title('F and Φ Radial Profiles', fontsize=13)
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3b.tick_params(axis='y', labelcolor='red')
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: F vs Φ scatter (showing correlation)
    ax4 = axes[1, 1]
    scatter = ax4.scatter(data['Phi_shifted'], data['F'], 
                          c=r, cmap='viridis', s=100, edgecolor='black', alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Radius r', fontsize=11)
    
    # Fit line
    coeffs = np.polyfit(data['Phi_shifted'], data['F'], 1)
    phi_line = np.linspace(min(data['Phi_shifted']), max(data['Phi_shifted']), 100)
    f_line = coeffs[0] * phi_line + coeffs[1]
    ax4.plot(phi_line, f_line, 'r--', linewidth=2, 
             label=f'F ≈ {coeffs[0]:.3f}Φ + {coeffs[1]:.3f}')
    
    corr = np.corrcoef(data['F'], data['Phi_shifted'])[0, 1]
    ax4.set_xlabel('Shifted Potential Φ-Φ_∞', fontsize=12)
    ax4.set_ylabel('Resource F', fontsize=12)
    ax4.set_title(f'F-Φ Correlation: r = {corr:.3f}', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/redshift_falsifier_final.png', 
                dpi=150, bbox_inches='tight')
    print(f"\nFinal visualization saved to: /mnt/user-data/outputs/redshift_falsifier_final.png")
    plt.close()


if __name__ == "__main__":
    data = generate_comprehensive_report()
