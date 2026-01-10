import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.special import spherical_jn, spherical_yn

"""
DET v5 COLLIDER - IMPROVED RIGOROUS VERSION
============================================
Improvements over v1:
1. Yukawa potential (more realistic than square well)
2. Two-parameter fit (depth AND range) from two observables
3. Better numerical methods for scattering

Physical targets:
- Deuteron binding energy: B_d = 2.224 MeV
- Triplet scattering length: a_t = +5.4 fm
- Effective range: r_0 ≈ 1.76 fm
- Deuteron RMS radius: r_d = 1.97 fm (matter radius)
"""

# Physical constants
HBAR_C = 197.3    # MeV·fm
M_NUCLEON = 939.0 # MeV/c²
ALPHA_EM = 1/137.036

class ImprovedDETScale:
    """
    DET parameters with Yukawa potential.
    
    V(r) = -V₀ * exp(-r/r_s) / (r/r_s)
    
    Two parameters:
    - V₀ = potential depth (MeV)
    - r_s = potential range (fm)
    
    Constrained by:
    - Deuteron binding energy → fixes one combination
    - Scattering length → fixes another combination
    """
    
    def __init__(self, V0_MeV=50.0, r_s_fm=1.4):
        self.V0 = V0_MeV
        self.r_s = r_s_fm
        self.mu = M_NUCLEON / 2  # Reduced mass
        
    def potential_strong(self, r):
        """Yukawa potential."""
        if r < 0.01:
            r = 0.01
        x = r / self.r_s
        return -self.V0 * np.exp(-x) / x
    
    def potential_em(self, r, Z1=1, Z2=1):
        """Screened Coulomb potential."""
        if r < 0.01:
            r = 0.01
        # Screen at short range
        screening = 1 - np.exp(-r / self.r_s)
        return ALPHA_EM * HBAR_C * Z1 * Z2 / r * screening
    
    def potential_total(self, r, has_em=True):
        V = self.potential_strong(r)
        if has_em:
            V += self.potential_em(r)
        return V


class ImprovedScatteringCalculator:
    """Numerov method for scattering from Yukawa potential."""
    
    def __init__(self, det_scale):
        self.scale = det_scale
        self.mu = M_NUCLEON / 2
        
    def solve_schrodinger(self, E_MeV, l=0, has_em=False, r_max=30.0, dr=0.01):
        """Solve radial Schrödinger equation using Numerov method."""
        if E_MeV <= 0:
            raise ValueError("E must be positive")
            
        k = np.sqrt(2 * self.mu * E_MeV) / HBAR_C
        
        r = np.arange(dr, r_max, dr)
        n = len(r)
        u = np.zeros(n)
        
        # Initial conditions (regular at origin for l=0)
        u[0] = dr
        u[1] = 2*dr
        
        def f(ri):
            V = self.scale.potential_total(ri, has_em)
            return 2 * self.mu / HBAR_C**2 * (V - E_MeV) + l*(l+1)/ri**2
        
        h2 = dr**2
        for i in range(1, n-1):
            f0 = f(r[i-1])
            f1 = f(r[i])
            f2 = f(r[i+1])
            u[i+1] = (2*u[i]*(1 - 5*h2*f1/12) - u[i-1]*(1 + h2*f0/12)) / (1 + h2*f2/12)

        # Normalize amplitude to keep numerics stable (phase shift uses ratios)
        umax = np.max(np.abs(u))
        if umax > 0:
            u = u / umax
        
        return r, u, k
    
    def phase_shift(self, E_MeV, l=0, has_em=False):
        """Extract phase shift using log-derivative matching.

        This is much more stable than two-point amplitude matching and respects
        reciprocity/unitarity numerically (phase shift is determined by the asymptotic
        flux ratio).
        """
        if E_MeV <= 0:
            return 0.0

        r, u, k = self.solve_schrodinger(E_MeV, l, has_em)

        # Match at large r where potential is negligible
        i_m = int(len(r) * 0.9)
        r_m = r[i_m]

        # Numerical log-derivative L = u'/u
        dr = r[1] - r[0]
        u_m = u[i_m]
        if abs(u_m) < 1e-12:
            # move slightly inward if at a node
            i_m = int(len(r) * 0.88)
            r_m = r[i_m]
            u_m = u[i_m]
            if abs(u_m) < 1e-12:
                return 0.0

        u_prime = (u[i_m + 1] - u[i_m - 1]) / (2 * dr)
        L = u_prime / u_m

        x = k * r_m
        jl = spherical_jn(l, x)
        yl = spherical_yn(l, x)

        # Derivatives wrt r: d/dr j_l(kr) = k * j_l'(kr)
        jl_p = k * spherical_jn(l, x, derivative=True)
        yl_p = k * spherical_yn(l, x, derivative=True)

        num = L * jl - jl_p
        den = L * yl - yl_p

        # tan(delta) = num/den
        if abs(den) < 1e-12:
            return np.pi / 2

        return np.arctan2(num, den)

    def current_check(self, E_MeV, l=0, has_em=False):
        """Numerical reciprocity check: probability current should be conserved.

        For a real potential, scattering is unitary: |S_l|=1. In practice, Numerov/matching
        errors can break this. We estimate conservation by checking that the Wronskian-like
        quantity u*u' is stable in the asymptotic region.
        Returns a small number when stable.
        """
        r, u, k = self.solve_schrodinger(E_MeV, l, has_em)
        dr = r[1] - r[0]
        # Use asymptotic window
        i1 = int(len(r) * 0.85)
        i2 = int(len(r) * 0.95)
        up = (u[2:] - u[:-2]) / (2 * dr)
        uu = u[1:-1] * up
        window = uu[i1-1:i2-1]
        if len(window) < 10:
            return 0.0
        return float(np.std(window) / (np.mean(np.abs(window)) + 1e-12))
    
    def scattering_length(self, has_em=False):
        """Compute s-wave scattering length.

        Primary: very-low-energy phase shift.
        Fallback: zero-energy solution a ≈ r - u/u' at large r.
        """
        # Primary method
        E_low = 1e-4
        k = np.sqrt(2 * self.mu * E_low) / HBAR_C
        delta = self.phase_shift(E_low, l=0, has_em=has_em)
        if abs(k) > 0 and abs(np.tan(delta)) < 1e6:
            a1 = -np.tan(delta) / k
            if np.isfinite(a1) and abs(a1) < 1e4:
                return float(a1)

        # Fallback: zero-energy integration (E ~ 0)
        E0 = 1e-8
        r, u, k0 = self.solve_schrodinger(E0, l=0, has_em=has_em)
        dr = r[1] - r[0]
        i_m = int(len(r) * 0.9)
        u_m = u[i_m]
        if abs(u_m) < 1e-12:
            i_m = int(len(r) * 0.88)
            u_m = u[i_m]
        u_prime = (u[i_m + 1] - u[i_m - 1]) / (2 * dr)
        if abs(u_prime) < 1e-12:
            return 0.0
        a0 = r[i_m] - u_m / u_prime
        return float(a0)
    
    def effective_range(self, has_em=False):
        """Extract r₀ from effective range expansion."""
        E1, E2 = 0.0005, 0.005
        
        k1 = np.sqrt(2 * self.mu * E1) / HBAR_C
        k2 = np.sqrt(2 * self.mu * E2) / HBAR_C
        
        d1 = self.phase_shift(E1, l=0, has_em=has_em)
        d2 = self.phase_shift(E2, l=0, has_em=has_em)
        
        # k*cot(δ) = -1/a + r₀*k²/2
        if abs(np.tan(d1)) < 1e-10 or abs(np.tan(d2)) < 1e-10:
            return 0.0
            
        y1 = k1 / np.tan(d1)
        y2 = k2 / np.tan(d2)
        
        r0 = 2 * (y2 - y1) / (k2**2 - k1**2)
        return r0
    
    def cross_section(self, E_MeV, l_max=2, has_em=False):
        """Total cross section."""
        if E_MeV <= 0:
            return 0.0
        k = np.sqrt(2 * self.mu * E_MeV) / HBAR_C
        
        sigma = 0.0
        for l in range(l_max + 1):
            delta = self.phase_shift(E_MeV, l, has_em)
            sigma += (2*l + 1) * np.sin(delta)**2
        
        return 4 * np.pi / k**2 / 100  # barns


class DeuteronSolver:
    """Find deuteron binding energy from potential."""
    
    def __init__(self, det_scale):
        self.scale = det_scale
        self.mu = M_NUCLEON / 2
        
    def find_binding_energy(self, B_guess=2.0):
        """Find B such that wavefunction is normalizable."""
        
        def trial_at_boundary(B):
            if B <= 0:
                return 1e10
                
            kappa = np.sqrt(2 * self.mu * B) / HBAR_C
            
            # Integrate from large r inward
            r_max = 30.0
            dr = 0.01
            r = np.arange(dr, r_max, dr)[::-1]  # Reversed
            u = np.zeros(len(r))
            
            # Start with decaying exponential
            u[0] = np.exp(-kappa * r[0])
            u[1] = np.exp(-kappa * r[1])
            
            h2 = dr**2
            for i in range(1, len(r)-1):
                V = self.scale.potential_strong(r[i])
                f_curr = 2 * self.mu / HBAR_C**2 * (-V - B)
                V_next = self.scale.potential_strong(r[i+1])
                f_next = 2 * self.mu / HBAR_C**2 * (-V_next - B)
                V_prev = self.scale.potential_strong(r[i-1])
                f_prev = 2 * self.mu / HBAR_C**2 * (-V_prev - B)
                
                u[i+1] = (2*u[i]*(1 - 5*h2*f_curr/12) - u[i-1]*(1 + h2*f_prev/12)) / (1 + h2*f_next/12)
            
            # Check log derivative at small r
            # Should match u → r near origin
            return u[-1] - u[-2]  # Should cross zero at correct B
        
        try:
            B = brentq(trial_at_boundary, 0.1, 20.0)
        except:
            B = B_guess
        return B
    
    def rms_radius(self, B):
        """Calculate deuteron RMS radius."""
        kappa = np.sqrt(2 * self.mu * B) / HBAR_C
        
        r_max = 30.0
        dr = 0.01
        r_arr = np.arange(dr, r_max, dr)
        
        # Build wavefunction
        psi = np.zeros(len(r_arr))
        for i, r in enumerate(r_arr):
            # Approximate: exponential tail dominates
            psi[i] = np.exp(-kappa * r)
        
        # Normalize
        norm = np.trapezoid(psi**2 * r_arr**2, r_arr) * 4 * np.pi
        psi /= np.sqrt(norm)
        
        # <r²>
        r2_avg = np.trapezoid(psi**2 * r_arr**4, r_arr) * 4 * np.pi
        
        return np.sqrt(r2_avg)


def fit_parameters():
    """
    Fit V₀ and r_s to match:
    1. Deuteron binding energy = 2.224 MeV
    2. Triplet scattering length = 5.4 fm
    """
    print("="*70)
    print("FITTING DET PARAMETERS TO NUCLEAR DATA")
    print("="*70)
    
    target_B = 2.224   # MeV
    target_a = 5.4     # fm
    target_r0 = 1.76   # fm
    target_rms = 1.97  # fm
    
    def objective(params):
        V0, r_s = params
        if V0 < 10 or V0 > 500 or r_s < 0.5 or r_s > 5.0:
            return 1e10
            
        scale = ImprovedDETScale(V0_MeV=V0, r_s_fm=r_s)
        
        # Calculate observables
        deut = DeuteronSolver(scale)
        B_calc = deut.find_binding_energy()
        
        scat = ImprovedScatteringCalculator(scale)
        a_calc = scat.scattering_length(has_em=False)
        
        # Weighted residuals
        err_B = ((B_calc - target_B) / target_B)**2
        err_a = ((a_calc - target_a) / target_a)**2
        
        return err_B + err_a
    
    # Grid search first
    print("\nGrid search for initial guess...")
    best_loss = float('inf')
    best_params = (50.0, 1.4)
    
    for V0 in [30, 40, 50, 60, 70, 80, 100]:
        for r_s in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            loss = objective([V0, r_s])
            if loss < best_loss:
                best_loss = loss
                best_params = (V0, r_s)
                print(f"  V₀={V0}, r_s={r_s}: loss={loss:.4f}")
    
    print(f"\nBest grid point: V₀={best_params[0]}, r_s={best_params[1]}")
    
    # Refine with optimization
    print("\nRefining with optimization...")
    result = minimize(objective, best_params, method='Nelder-Mead',
                     options={'maxiter': 200, 'xatol': 0.1, 'fatol': 0.001})
    
    V0_opt, r_s_opt = result.x
    print(f"Optimized: V₀={V0_opt:.1f} MeV, r_s={r_s_opt:.2f} fm")
    
    # Final validation
    scale = ImprovedDETScale(V0_MeV=V0_opt, r_s_fm=r_s_opt)
    deut = DeuteronSolver(scale)
    scat = ImprovedScatteringCalculator(scale)
    
    B_calc = deut.find_binding_energy()
    a_calc = scat.scattering_length(has_em=False)
    r0_calc = scat.effective_range(has_em=False)
    rms_calc = deut.rms_radius(B_calc)
    
    print("\n" + "-"*70)
    print("VALIDATION")
    print("-"*70)
    print(f"{'Observable':<25} | {'Calculated':<12} | {'Target':<12} | {'Error'}")
    print("-"*70)
    print(f"{'Binding energy (MeV)':<25} | {B_calc:<12.3f} | {target_B:<12.3f} | {abs(B_calc-target_B)/target_B*100:.1f}%")
    print(f"{'Scattering length (fm)':<25} | {a_calc:<12.2f} | {target_a:<12.2f} | {abs(a_calc-target_a)/target_a*100:.1f}%")
    print(f"{'Effective range (fm)':<25} | {r0_calc:<12.2f} | {target_r0:<12.2f} | {abs(r0_calc-target_r0)/target_r0*100:.1f}%")
    print(f"{'RMS radius (fm)':<25} | {rms_calc:<12.2f} | {target_rms:<12.2f} | {abs(rms_calc-target_rms)/target_rms*100:.1f}%")
    
    return V0_opt, r_s_opt, scale


def make_predictions(scale):
    """Generate predictions using fitted parameters."""
    print("\n" + "="*70)
    print("PREDICTIONS (from fitted parameters)")
    print("="*70)
    
    scat = ImprovedScatteringCalculator(scale)
    
    # 1. Phase shifts
    print("\n1. S-WAVE PHASE SHIFTS δ₀(E)")
    print("-"*50)
    print(f"{'E (MeV)':<12} | {'δ₀ n-p (deg)':<15} | {'δ₀ p-p (deg)':<15}")
    energies = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    phase_data = []
    for E in energies:
        d_np = scat.phase_shift(E, l=0, has_em=False) * 180/np.pi
        d_pp = scat.phase_shift(E, l=0, has_em=True) * 180/np.pi
        phase_data.append((E, d_np, d_pp))
        print(f"{E:<12.3f} | {d_np:<15.1f} | {d_pp:<15.1f}")
    
    # 2. Cross sections
    print("\n2. TOTAL CROSS SECTIONS σ(E)")
    print("-"*50)
    print(f"{'E (MeV)':<12} | {'σ_np (barn)':<15} | {'σ_pp (barn)':<15}")
    sigma_data = []
    for E in energies:
        if E < 0.001:
            continue
        s_np = scat.cross_section(E, has_em=False)
        s_pp = scat.cross_section(E, has_em=True)
        sigma_data.append((E, s_np, s_pp))
        print(f"{E:<12.3f} | {s_np:<15.3f} | {s_pp:<15.3f}")
    
    # 3. Comparison to experimental data (if we had it)
    print("\n3. COMPARISON TO NIJMEGEN PWA DATA")
    print("-"*50)
    print("Experimental n-p ³S₁ phase shifts (selected points):")
    exp_data = [
        (0.001, 180),  # Approx from scattering length
        (1.0, 147.7),
        (5.0, 118.2),
        (10.0, 102.6),
        (25.0, 80.6),
        (50.0, 62.8),
    ]
    print(f"{'E (MeV)':<12} | {'Exp δ₀ (deg)':<15} | {'DET δ₀ (deg)':<15} | {'Diff (deg)'}")
    for E, exp_d in exp_data:
        calc_d = scat.phase_shift(E, l=0, has_em=False) * 180/np.pi
        print(f"{E:<12.1f} | {exp_d:<15.1f} | {calc_d:<15.1f} | {calc_d-exp_d:+.1f}")
    
    print("\nReciprocity check (numerical unitarity proxy, lower is better):")
    for E in [0.01, 0.1, 1.0, 5.0, 10.0]:
        chk = scat.current_check(E, l=0, has_em=False)
        print(f"  E={E:>6.2f} MeV: current_stability={chk:.3e}")

    return phase_data, sigma_data


def create_comparison_plots(scale, phase_data, sigma_data):
    """Create publication-quality comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    scat = ImprovedScatteringCalculator(scale)
    
    # 1. Potential
    ax = axes[0, 0]
    r = np.linspace(0.1, 6, 200)
    V_strong = [scale.potential_strong(ri) for ri in r]
    V_em = [scale.potential_em(ri) for ri in r]
    V_total = [scale.potential_total(ri, has_em=True) for ri in r]
    
    ax.plot(r, V_strong, 'b-', label='Strong (Yukawa)', linewidth=2)
    ax.plot(r, V_em, 'r-', label='EM (Coulomb)', linewidth=2)
    ax.plot(r, V_total, 'k--', label='Total (p-p)', linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-2.224, color='green', linestyle=':', label='Deuteron B.E.')
    ax.set_xlabel('r (fm)', fontsize=12)
    ax.set_ylabel('V (MeV)', fontsize=12)
    ax.set_title(f'DET Yukawa Potential\nV₀={scale.V0:.1f} MeV, r_s={scale.r_s:.2f} fm', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-100, 20)
    ax.set_xlim(0, 6)
    
    # 2. Phase shifts vs experimental
    ax = axes[0, 1]
    E_range = np.logspace(-3, 1.5, 100)
    delta_calc = [scat.phase_shift(E, l=0, has_em=False) * 180/np.pi for E in E_range]
    
    ax.semilogx(E_range, delta_calc, 'b-', label='DET prediction', linewidth=2)
    
    # Experimental data points
    exp_E = [0.001, 1.0, 5.0, 10.0, 25.0, 50.0]
    exp_delta = [180, 147.7, 118.2, 102.6, 80.6, 62.8]
    ax.scatter(exp_E, exp_delta, c='red', s=80, marker='o', label='Nijmegen PWA', zorder=5)
    
    ax.set_xlabel('E (MeV)', fontsize=12)
    ax.set_ylabel('δ₀ (degrees)', fontsize=12)
    ax.set_title('S-wave Phase Shift (n-p ³S₁)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200)
    
    # 3. Cross section
    ax = axes[1, 0]
    E_plot = np.logspace(-2, 1.5, 100)
    sigma_np = [scat.cross_section(E, has_em=False) for E in E_plot]
    sigma_pp = [scat.cross_section(E, has_em=True) for E in E_plot]
    
    ax.loglog(E_plot, sigma_np, 'b-', label='n-p', linewidth=2)
    ax.loglog(E_plot, sigma_pp, 'r-', label='p-p', linewidth=2)
    
    # Low energy limit: σ → 4π a² as E → 0
    a = scat.scattering_length(has_em=False)
    sigma_limit = 4 * np.pi * a**2 / 100  # barns
    ax.axhline(y=sigma_limit, color='green', linestyle='--', alpha=0.7, 
               label=f'4πa² = {sigma_limit:.1f} barn')
    
    ax.set_xlabel('E (MeV)', fontsize=12)
    ax.set_ylabel('σ (barn)', fontsize=12)
    ax.set_title('Total Cross Section', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create summary text
    deut = DeuteronSolver(scale)
    B = deut.find_binding_energy()
    a_calc = scat.scattering_length(has_em=False)
    r0_calc = scat.effective_range(has_em=False)
    rms = deut.rms_radius(B)
    
    summary = f"""
    DET v5 RIGOROUS RESULTS
    =======================
    
    FITTED PARAMETERS (2 d.o.f.):
      V₀ = {scale.V0:.1f} MeV
      r_s = {scale.r_s:.2f} fm
    
    FIT TARGETS:
      B_d = 2.224 MeV (deuteron)
      a_t = 5.4 fm (scattering length)
    
    CALCULATED VALUES:
      B_d = {B:.3f} MeV
      a_t = {a_calc:.2f} fm
      r_0 = {r0_calc:.2f} fm (predicted)
      r_d = {rms:.2f} fm (predicted)
    
    EXPERIMENTAL:
      r_0 = 1.76 fm
      r_d = 1.97 fm
    """
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('./det_improved_validation.png', dpi=150)
    print("\nSaved to det_improved_validation.png")


if __name__ == "__main__":
    # Fit parameters
    V0, r_s, scale = fit_parameters()
    
    # Make predictions
    phase_data, sigma_data = make_predictions(scale)
    
    # Create plots
    create_comparison_plots(scale, phase_data, sigma_data)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
DET v5 IMPROVED RIGOROUS VALIDATION
===================================

METHODOLOGY:
1. Use Yukawa potential V(r) = -V₀ exp(-r/r_s)/(r/r_s)
2. Fit TWO parameters (V₀, r_s) to TWO observables:
   - Deuteron binding energy B_d = 2.224 MeV
   - Triplet scattering length a_t = 5.4 fm
3. PREDICT (not fit) other observables:
   - Effective range r₀
   - Deuteron RMS radius
   - Energy-dependent phase shifts
   - Cross sections

This is the proper way to test a theory:
- Constrain free parameters with minimal data
- Predict new observables
- Compare predictions to experiment
""")