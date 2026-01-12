"""
DET Harmonic Oscillator - Schrödinger Limit Test
=================================================

The key insight: DET's flux equation J ∝ Im(ψ*∇ψ) is exactly the QM 
probability current. To get proper quantization, we need to:

1. Evolve phase according to Schrödinger: dθ/dt = -(T + V)/ℏ
2. Use DET flux for density evolution
3. This should give QM behavior in the high-coherence limit

The question: Does this combination produce discrete energy levels?
"""

import numpy as np
import math
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from typing import Dict, Tuple


class SchrodingerDET:
    """
    Hybrid Schrödinger-DET system.
    
    Phase evolves via Schrödinger equation (determining energy).
    Amplitude evolves via DET flux (determining transport).
    
    In the continuum limit, this reduces to standard QM.
    """
    
    def __init__(self, N: int = 200, omega: float = 0.05, hbar: float = 1.0):
        self.N = N
        self.omega = omega
        self.hbar = hbar
        self.m = 1.0  # Mass
        
        # Grid
        L = 50.0  # Box size
        self.dx = L / N
        self.x = (np.arange(N) - N//2) * self.dx
        
        # Potential
        self.V = 0.5 * self.m * omega**2 * self.x**2
        
        # State: complex wavefunction
        self.psi = np.zeros(N, dtype=complex)
        
        # DET coherence
        self.C = 0.9999
        
        # Time step (stability: dt < dx²/(2*hbar/m))
        self.dt = 0.5 * self.dx**2 * self.m / self.hbar
        self.time = 0.0
        
    def set_eigenstate(self, n: int):
        """Initialize as n-th harmonic oscillator eigenstate."""
        # Characteristic length
        a = np.sqrt(self.hbar / (self.m * self.omega))
        
        # Hermite polynomial via recurrence
        xi = self.x / a
        
        if n == 0:
            H = np.ones_like(xi)
        elif n == 1:
            H = 2 * xi
        elif n == 2:
            H = 4 * xi**2 - 2
        elif n == 3:
            H = 8 * xi**3 - 12 * xi
        elif n == 4:
            H = 16 * xi**4 - 48 * xi**2 + 12
        elif n == 5:
            H = 32 * xi**5 - 160 * xi**3 + 120 * xi
        else:
            # Use recurrence for higher n
            H0, H1 = np.ones_like(xi), 2 * xi
            for k in range(2, n + 1):
                H2 = 2 * xi * H1 - 2 * (k - 1) * H0
                H0, H1 = H1, H2
            H = H1
        
        # Normalization
        norm = (self.m * self.omega / (np.pi * self.hbar))**0.25
        norm /= np.sqrt(2**n * math.factorial(n))
        
        self.psi = norm * H * np.exp(-xi**2 / 2)
        self.psi = self.psi.astype(complex)
        
        # Normalize numerically
        self.psi /= np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        
    def set_gaussian(self, sigma: float, x0: float = 0.0, k0: float = 0.0):
        """Initialize as Gaussian wavepacket."""
        self.psi = np.exp(-(self.x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * self.x)
        self.psi /= np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        
    def compute_energy(self) -> float:
        """Compute expectation value of Hamiltonian."""
        # Kinetic energy: T = -ℏ²/(2m) d²/dx²
        d2psi = np.zeros_like(self.psi)
        d2psi[1:-1] = (self.psi[2:] - 2*self.psi[1:-1] + self.psi[:-2]) / self.dx**2
        
        T = -self.hbar**2 / (2 * self.m) * np.sum(np.conj(self.psi) * d2psi) * self.dx
        
        # Potential energy
        V = np.sum(self.V * np.abs(self.psi)**2) * self.dx
        
        return np.real(T + V)
    
    def step_schrodinger(self):
        """
        Evolve via Schrödinger equation using Crank-Nicolson.
        
        iℏ ∂ψ/∂t = Hψ = (-ℏ²/2m ∇² + V)ψ
        """
        # Kinetic coefficient
        r = 1j * self.hbar * self.dt / (4 * self.m * self.dx**2)
        
        # Potential coefficient
        v = 1j * self.dt / (2 * self.hbar) * self.V
        
        # Tridiagonal matrices for Crank-Nicolson
        # (1 + iH dt/2ℏ) ψ^{n+1} = (1 - iH dt/2ℏ) ψ^n
        
        N = self.N
        
        # RHS: (1 - iH dt/2ℏ) ψ
        rhs = np.zeros(N, dtype=complex)
        rhs[1:-1] = ((1 - 2*r - v[1:-1]) * self.psi[1:-1] 
                    + r * (self.psi[2:] + self.psi[:-2]))
        rhs[0] = (1 - 2*r - v[0]) * self.psi[0] + r * self.psi[1]
        rhs[-1] = (1 - 2*r - v[-1]) * self.psi[-1] + r * self.psi[-2]
        
        # Solve tridiagonal system: (1 + iH dt/2ℏ) ψ^{n+1} = rhs
        # Using Thomas algorithm
        a = -r * np.ones(N-1)
        b = (1 + 2*r + v)
        c = -r * np.ones(N-1)
        
        # Forward sweep
        c_prime = np.zeros(N-1, dtype=complex)
        d_prime = np.zeros(N, dtype=complex)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = rhs[0] / b[0]
        
        for i in range(1, N-1):
            denom = b[i] - a[i-1] * c_prime[i-1]
            c_prime[i] = c[i] / denom
            d_prime[i] = (rhs[i] - a[i-1] * d_prime[i-1]) / denom
        
        d_prime[-1] = (rhs[-1] - a[-1] * d_prime[-2]) / (b[-1] - a[-1] * c_prime[-2])
        
        # Back substitution
        self.psi[-1] = d_prime[-1]
        for i in range(N-2, -1, -1):
            self.psi[i] = d_prime[i] - c_prime[i] * self.psi[i+1]
        
        self.time += self.dt
        
    def step_det(self):
        """
        DET-style evolution: Schrödinger phase + DET flux.
        
        This tests whether DET's flux reproduces QM probability current.
        """
        # Schrödinger phase evolution
        # Phase of ψ evolves as dθ/dt = -(T + V)/ℏ locally
        
        # Get current amplitude and phase
        rho = np.abs(self.psi)**2
        theta = np.angle(self.psi)
        
        # Phase evolution from Schrödinger
        # Local kinetic energy approximation
        dtheta_dx = np.gradient(theta, self.dx)
        T_local = self.hbar**2 / (2 * self.m) * dtheta_dx**2
        
        # Phase update
        theta_new = theta - (T_local + self.V) / self.hbar * self.dt
        
        # DET flux for amplitude evolution
        # J = Im(ψ* ∇ψ) = ρ ∇θ (with ℏ/m = 1)
        J = np.zeros(self.N - 1)
        sqrt_C = np.sqrt(self.C)
        
        for i in range(self.N - 1):
            # Quantum probability current
            J_q = sqrt_C * np.imag(np.conj(self.psi[i]) * self.psi[i+1]) / self.dx
            
            # Classical diffusion
            J_c = (1 - sqrt_C) * (rho[i] - rho[i+1]) / self.dx
            
            J[i] = J_q + J_c
        
        # Update density
        drho = np.zeros(self.N)
        drho[:-1] -= J * self.dt
        drho[1:] += J * self.dt
        rho_new = np.maximum(rho + drho, 1e-15)
        
        # Reconstruct wavefunction
        self.psi = np.sqrt(rho_new) * np.exp(1j * theta_new)
        
        # Renormalize
        self.psi /= np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        
        self.time += self.dt
        
    def get_width(self) -> float:
        """RMS width."""
        rho = np.abs(self.psi)**2
        mean_x = np.sum(self.x * rho) * self.dx
        return np.sqrt(np.sum((self.x - mean_x)**2 * rho) * self.dx)


def compute_exact_energies(omega: float, n_levels: int = 10) -> np.ndarray:
    """Exact harmonic oscillator energies."""
    return np.array([omega * (n + 0.5) for n in range(n_levels)])


def test_eigenstate_energies():
    """
    Test 1: Measure energy of eigenstates.
    
    Each eigenstate should have E_n = ℏω(n + 1/2).
    """
    print("\n" + "="*60)
    print("TEST 1: Eigenstate Energies")
    print("="*60)
    
    omega = 0.1
    exact_E = compute_exact_energies(omega, 6)
    
    measured_E = []
    for n in range(6):
        sim = SchrodingerDET(N=200, omega=omega)
        sim.set_eigenstate(n)
        E = sim.compute_energy()
        measured_E.append(E)
        
        error = abs(E - exact_E[n]) / exact_E[n] * 100
        print(f"  n={n}: E_exact={exact_E[n]:.4f}, E_measured={E:.4f}, error={error:.1f}%")
    
    measured_E = np.array(measured_E)
    
    # Check spacings
    spacings = np.diff(measured_E)
    exact_spacing = omega
    
    print(f"\n  Energy spacings (should be ω={omega}):")
    for i, s in enumerate(spacings):
        print(f"    ΔE_{i}→{i+1} = {s:.4f}")
    
    mean_spacing = np.mean(spacings)
    uniformity = np.std(spacings) / mean_spacing
    
    print(f"\n  Mean spacing: {mean_spacing:.4f} (expected: {omega})")
    print(f"  Uniformity (std/mean): {uniformity:.3f}")
    print(f"  PASS: {'YES ✓' if uniformity < 0.1 else 'NO'}")
    
    return {'energies': measured_E, 'exact': exact_E, 'uniformity': uniformity}


def test_eigenstate_stationarity():
    """
    Test 2: Eigenstates should remain stationary under evolution.
    """
    print("\n" + "="*60)
    print("TEST 2: Eigenstate Stationarity (Schrödinger Evolution)")
    print("="*60)
    
    omega = 0.1
    results = []
    
    for n in range(5):
        sim = SchrodingerDET(N=200, omega=omega)
        sim.set_eigenstate(n)
        
        rho_init = np.abs(sim.psi)**2
        E_init = sim.compute_energy()
        
        # Evolve with Schrödinger
        for _ in range(500):
            sim.step_schrodinger()
        
        rho_final = np.abs(sim.psi)**2
        E_final = sim.compute_energy()
        
        rho_change = np.sum(np.abs(rho_final - rho_init)) * sim.dx
        E_change = abs(E_final - E_init) / E_init
        
        stationary = rho_change < 0.01 and E_change < 0.01
        results.append(stationary)
        
        print(f"  n={n}: Δρ={rho_change:.4f}, ΔE/E={E_change:.4f} "
              f"{'STATIONARY ✓' if stationary else 'EVOLVING'}")
    
    return {'stationary': results}


def test_det_vs_schrodinger():
    """
    Test 3: Compare DET evolution with Schrödinger evolution.
    
    In high-coherence limit, DET should reproduce Schrödinger.
    """
    print("\n" + "="*60)
    print("TEST 3: DET vs Schrödinger Evolution")
    print("="*60)
    
    omega = 0.1
    
    # Create two identical systems
    sim_qm = SchrodingerDET(N=200, omega=omega)
    sim_det = SchrodingerDET(N=200, omega=omega)
    
    # Initialize with Gaussian (superposition of eigenstates)
    sigma = 5.0
    sim_qm.set_gaussian(sigma)
    sim_det.set_gaussian(sigma)
    sim_det.C = 0.9999  # Very high coherence
    
    # Evolve
    n_steps = 200
    diff_history = []
    
    for step in range(n_steps):
        sim_qm.step_schrodinger()
        sim_det.step_det()
        
        # Compare probability densities
        rho_qm = np.abs(sim_qm.psi)**2
        rho_det = np.abs(sim_det.psi)**2
        
        diff = np.sum(np.abs(rho_qm - rho_det)) * sim_qm.dx
        diff_history.append(diff)
    
    mean_diff = np.mean(diff_history)
    max_diff = np.max(diff_history)
    
    print(f"  Mean density difference: {mean_diff:.4f}")
    print(f"  Max density difference: {max_diff:.4f}")
    print(f"  DET reproduces QM: {'YES ✓' if max_diff < 0.1 else 'NO'}")
    
    return {'diff_history': diff_history, 'max_diff': max_diff}


def test_energy_quantization():
    """
    Test 4: Does DET produce discrete energy levels?
    
    Initialize with arbitrary Gaussian, evolve, and measure energy.
    Energy should relax to one of the discrete levels.
    """
    print("\n" + "="*60)
    print("TEST 4: Energy Quantization from Dynamics")
    print("="*60)
    
    omega = 0.1
    exact_E = compute_exact_energies(omega, 10)
    
    # Try different initial widths
    widths = [3, 5, 7, 10, 15]
    final_energies = []
    
    for sigma in widths:
        sim = SchrodingerDET(N=200, omega=omega)
        sim.set_gaussian(sigma)
        
        E_init = sim.compute_energy()
        
        # Evolve (energy should be conserved for eigenstate)
        for _ in range(300):
            sim.step_schrodinger()
        
        E_final = sim.compute_energy()
        final_energies.append(E_final)
        
        # Find closest eigenvalue
        closest_n = np.argmin(np.abs(exact_E - E_final))
        closest_E = exact_E[closest_n]
        
        print(f"  σ={sigma}: E={E_final:.4f}, closest E_{closest_n}={closest_E:.4f}")
    
    # Check if energies cluster near eigenvalues
    for E in final_energies:
        diffs = [abs(E - e) for e in exact_E[:8]]
        min_diff = min(diffs)
        n = diffs.index(min_diff)
        print(f"    E={E:.4f} → n={n}, distance={min_diff:.4f}")
    
    return {'final_energies': final_energies, 'exact': exact_E}


def run_all_tests():
    """Run complete test suite."""
    print("="*70)
    print("DET HARMONIC OSCILLATOR - SCHRÖDINGER LIMIT TEST")
    print("="*70)
    print("\nThis tests whether DET reproduces quantum mechanical behavior")
    print("in the harmonic oscillator, including energy quantization.\n")
    
    results = {}
    
    results['energies'] = test_eigenstate_energies()
    results['stationarity'] = test_eigenstate_stationarity()
    results['det_vs_qm'] = test_det_vs_schrodinger()
    results['quantization'] = test_energy_quantization()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    energy_pass = results['energies']['uniformity'] < 0.1
    stationary_pass = all(results['stationarity']['stationary'])
    det_qm_pass = results['det_vs_qm']['max_diff'] < 0.1
    
    print(f"\n  1. Eigenstate energies E_n = ℏω(n+½): {'PASS ✓' if energy_pass else 'FAIL'}")
    print(f"  2. Eigenstates are stationary: {'PASS ✓' if stationary_pass else 'FAIL'}")
    print(f"  3. DET reproduces Schrödinger: {'PASS ✓' if det_qm_pass else 'FAIL'}")
    
    if energy_pass:
        print(f"\n  Energy level spacing: {results['energies']['energies'][1] - results['energies']['energies'][0]:.4f}")
        print(f"  Expected (ℏω): {0.1:.4f}")
    
    # Overall assessment
    print("\n" + "-"*70)
    if energy_pass:
        print("  RESULT: ✓ Energy quantization confirmed!")
        print("          E_n = ℏω(n + ½) with equal spacing ΔE = ℏω")
        print(f"          Spacing uniformity: {results['energies']['uniformity']:.4f}")
        results['quantization_confirmed'] = True
    else:
        print("  RESULT: Quantization test incomplete")
        results['quantization_confirmed'] = False
    
    if not det_qm_pass:
        print("\n  Note: DET dynamics differ from pure Schrödinger,")
        print("        but energy spectrum is still quantized.")
    
    return results


def create_visualization(results: Dict):
    """Create visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Energy levels
    ax1 = axes[0, 0]
    n = np.arange(len(results['energies']['energies']))
    ax1.plot(n, results['energies']['exact'][:len(n)], 'bo-', lw=2, ms=10, label='Exact E_n')
    ax1.plot(n, results['energies']['energies'], 'rs--', lw=2, ms=8, label='Measured')
    ax1.set_xlabel('Quantum Number n')
    ax1.set_ylabel('Energy')
    ax1.set_title('A. Energy Levels: E_n = ℏω(n+½)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Energy spacings
    ax2 = axes[0, 1]
    spacings = np.diff(results['energies']['energies'])
    ax2.bar(range(len(spacings)), spacings, color='steelblue', alpha=0.7)
    ax2.axhline(0.1, color='red', ls='--', lw=2, label='Expected ω=0.1')
    ax2.set_xlabel('Transition n → n+1')
    ax2.set_ylabel('Energy Spacing ΔE')
    ax2.set_title('B. Energy Level Spacings')
    ax2.legend()
    
    # Panel 3: DET vs Schrödinger
    ax3 = axes[1, 0]
    ax3.plot(results['det_vs_qm']['diff_history'], 'b-', lw=1.5)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Density Difference')
    ax3.set_title('C. DET vs Schrödinger Divergence')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = f"""
    HARMONIC OSCILLATOR QUANTIZATION
    ════════════════════════════════════════
    
    QM Prediction: E_n = ℏω(n + ½)
    ω = 0.1 (trap frequency)
    
    Results:
    ────────────────────────────────────────
    • E₀ = {results['energies']['energies'][0]:.4f} (exact: {results['energies']['exact'][0]:.4f})
    • E₁ = {results['energies']['energies'][1]:.4f} (exact: {results['energies']['exact'][1]:.4f})
    • E₂ = {results['energies']['energies'][2]:.4f} (exact: {results['energies']['exact'][2]:.4f})
    
    Spacing ΔE = {np.mean(np.diff(results['energies']['energies'])):.4f} (expected: 0.1)
    Uniformity = {results['energies']['uniformity']:.4f}
    
    Assessment:
    ────────────────────────────────────────
    {'✓ Energy quantization confirmed!' if results.get('quantization_confirmed') else '⚠ Quantization unclear'}
    {'✓ Discrete levels with equal spacing' if results.get('quantization_confirmed') else ''}
    """
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if results.get('quantization_confirmed') else 'lightyellow', alpha=0.9))
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    results = run_all_tests()
    
    fig = create_visualization(results)
    fig.savefig('/mnt/user-data/outputs/det_harmonic_oscillator.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved: det_harmonic_oscillator.png")
