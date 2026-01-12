"""
DET v6_2 Electromagnetism - Clean Implementation
===============================================

Three separate, well-tested modules:
1. Electrostatics: Poisson solver, Coulomb forces
2. Magnetostatics: Current → B field
3. Electrodynamics: Wave propagation (FDTD)

Each module is self-contained and follows DET principles.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import matplotlib.pyplot as plt


# ============================================================
# MODULE 1: ELECTROSTATICS
# ============================================================

class Electrostatics2D:
    """
    Electrostatic solver: ∇²φ = -ρ/ε, E = -∇φ
    
    Uses FFT-based Poisson solver for periodic BC.
    """
    
    def __init__(self, Nx: int = 80, Ny: int = 80, dx: float = 1.0, epsilon: float = 1.0):
        self.Nx, self.Ny = Nx, Ny
        self.dx = dx
        self.epsilon = epsilon
        
        self.rho = np.zeros((Ny, Nx))
        self.phi = np.zeros((Ny, Nx))
        self.Ex = np.zeros((Ny, Nx))
        self.Ey = np.zeros((Ny, Nx))
        
        # Precompute k² for FFT solver
        kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, dx)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # Regularize
        
    def clear(self):
        """Clear all fields."""
        self.rho.fill(0)
        self.phi.fill(0)
        self.Ex.fill(0)
        self.Ey.fill(0)
        
    def add_point_charge(self, x: float, y: float, Q: float):
        """Add point charge (delta function approximation)."""
        ix, iy = int(x) % self.Nx, int(y) % self.Ny
        self.rho[iy, ix] += Q / self.dx**2
        
    def add_gaussian_charge(self, x: float, y: float, Q: float, sigma: float):
        """Add Gaussian charge distribution."""
        Y, X = np.mgrid[0:self.Ny, 0:self.Nx]
        r2 = (X - x)**2 + (Y - y)**2
        # Normalized Gaussian
        gauss = np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        self.rho += Q * gauss
        
    def solve(self):
        """Solve Poisson equation."""
        # FFT solve
        rho_k = np.fft.fft2(self.rho)
        phi_k = rho_k / (self.epsilon * self.K2)
        phi_k[0, 0] = 0  # Zero mean
        self.phi = np.real(np.fft.ifft2(phi_k))
        
        # E = -∇φ (central difference)
        self.Ex = -(np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / (2*self.dx)
        self.Ey = -(np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / (2*self.dx)
        
    def E_at(self, x: float, y: float) -> Tuple[float, float]:
        """Interpolate E field at (x, y)."""
        ix, iy = int(x) % self.Nx, int(y) % self.Ny
        return self.Ex[iy, ix], self.Ey[iy, ix]
    
    def E_magnitude(self) -> np.ndarray:
        return np.sqrt(self.Ex**2 + self.Ey**2)
    
    def total_charge(self) -> float:
        return np.sum(self.rho) * self.dx**2
    
    def energy(self) -> float:
        """Field energy: U = ε/2 ∫ E² dA"""
        return 0.5 * self.epsilon * np.sum(self.Ex**2 + self.Ey**2) * self.dx**2


# ============================================================
# MODULE 2: MAGNETOSTATICS
# ============================================================

class Magnetostatics2D:
    """
    Magnetostatic solver for 2D (currents perpendicular to plane).
    
    Solves: ∇²Az = -μ Jz, B = ∇×A
    """
    
    def __init__(self, Nx: int = 80, Ny: int = 80, dx: float = 1.0, mu: float = 1.0):
        self.Nx, self.Ny = Nx, Ny
        self.dx = dx
        self.mu = mu
        
        self.Jz = np.zeros((Ny, Nx))  # Current (out of plane)
        self.Az = np.zeros((Ny, Nx))  # Vector potential
        self.Bx = np.zeros((Ny, Nx))
        self.By = np.zeros((Ny, Nx))
        
        # Precompute k²
        kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, dx)
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0
        
    def add_wire(self, x: float, y: float, I: float):
        """Add current-carrying wire at (x, y)."""
        ix, iy = int(x) % self.Nx, int(y) % self.Ny
        self.Jz[iy, ix] += I / self.dx**2
        
    def add_gaussian_current(self, x: float, y: float, I: float, sigma: float):
        """Add Gaussian current distribution."""
        Y, X = np.mgrid[0:self.Ny, 0:self.Nx]
        r2 = (X - x)**2 + (Y - y)**2
        gauss = np.exp(-r2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        self.Jz += I * gauss
        
    def solve(self):
        """Solve for A and B."""
        Jz_k = np.fft.fft2(self.Jz)
        Az_k = self.mu * Jz_k / self.K2
        Az_k[0, 0] = 0
        self.Az = np.real(np.fft.ifft2(Az_k))
        
        # B = curl(A): Bx = ∂Az/∂y, By = -∂Az/∂x
        self.Bx = (np.roll(self.Az, -1, axis=0) - np.roll(self.Az, 1, axis=0)) / (2*self.dx)
        self.By = -(np.roll(self.Az, -1, axis=1) - np.roll(self.Az, 1, axis=1)) / (2*self.dx)
        
    def B_magnitude(self) -> np.ndarray:
        return np.sqrt(self.Bx**2 + self.By**2)


# ============================================================
# MODULE 3: ELECTRODYNAMICS (FDTD)
# ============================================================

class Electrodynamics2D:
    """
    FDTD solver for Maxwell equations in 2D (TM mode).
    
    Fields: Ez (out of plane), Hx, Hy (in plane)
    
    Update equations:
    ∂Ez/∂t = (1/ε)(∂Hy/∂x - ∂Hx/∂y - Jz)
    ∂Hx/∂t = -(1/μ) ∂Ez/∂y
    ∂Hy/∂t = (1/μ) ∂Ez/∂x
    """
    
    def __init__(self, Nx: int = 100, Ny: int = 100, dx: float = 1.0, 
                 epsilon: float = 1.0, mu: float = 1.0):
        self.Nx, self.Ny = Nx, Ny
        self.dx = dx
        self.epsilon = epsilon
        self.mu = mu
        self.c = 1.0 / np.sqrt(epsilon * mu)
        
        # Courant-stable dt
        self.dt = 0.5 * dx / (self.c * np.sqrt(2))
        
        # Fields (Yee grid: E at integer, H at half-integer)
        self.Ez = np.zeros((Ny, Nx))
        self.Hx = np.zeros((Ny, Nx))
        self.Hy = np.zeros((Ny, Nx))
        
        # Source
        self.Jz = np.zeros((Ny, Nx))
        
        # Absorbing boundary (PML-like damping)
        self.setup_absorbing_boundary(width=10)
        
        self.time = 0.0
        
    def setup_absorbing_boundary(self, width: int = 10):
        """Setup absorbing boundary layer."""
        self.sigma_e = np.zeros((self.Ny, self.Nx))
        self.sigma_h = np.zeros((self.Ny, self.Nx))
        
        for i in range(width):
            damping = 0.5 * ((width - i) / width)**2
            # Edges
            self.sigma_e[:, i] = damping
            self.sigma_e[:, -(i+1)] = damping
            self.sigma_e[i, :] = np.maximum(self.sigma_e[i, :], damping)
            self.sigma_e[-(i+1), :] = np.maximum(self.sigma_e[-(i+1), :], damping)
            
        self.sigma_h = self.sigma_e.copy()
        
    def step(self):
        """Advance one time step."""
        dt, dx = self.dt, self.dx
        eps, mu = self.epsilon, self.mu
        
        # Update H (half step behind E)
        # Hx^{n+1/2} = Hx^{n-1/2} - (dt/μ) ∂Ez/∂y
        dEz_dy = (np.roll(self.Ez, -1, axis=0) - self.Ez) / dx
        self.Hx = (1 - self.sigma_h * dt) * self.Hx - (dt / mu) * dEz_dy
        
        # Hy^{n+1/2} = Hy^{n-1/2} + (dt/μ) ∂Ez/∂x
        dEz_dx = (np.roll(self.Ez, -1, axis=1) - self.Ez) / dx
        self.Hy = (1 - self.sigma_h * dt) * self.Hy + (dt / mu) * dEz_dx
        
        # Update E
        # Ez^{n+1} = Ez^n + (dt/ε)(∂Hy/∂x - ∂Hx/∂y - Jz)
        dHy_dx = (self.Hy - np.roll(self.Hy, 1, axis=1)) / dx
        dHx_dy = (self.Hx - np.roll(self.Hx, 1, axis=0)) / dx
        curl_H = dHy_dx - dHx_dy
        
        self.Ez = (1 - self.sigma_e * dt) * self.Ez + (dt / eps) * (curl_H - self.Jz)
        
        self.time += dt
        
    def add_point_source(self, x: int, y: int, amplitude: float, frequency: float):
        """Add oscillating point source."""
        self.Jz[y, x] += amplitude * np.sin(2 * np.pi * frequency * self.time)
        
    def add_gaussian_source(self, x: int, y: int, amplitude: float, 
                           frequency: float, width: float = 3.0):
        """Add oscillating Gaussian source."""
        Y, X = np.mgrid[0:self.Ny, 0:self.Nx]
        r2 = (X - x)**2 + (Y - y)**2
        envelope = np.exp(-r2 / (2 * width**2))
        self.Jz = amplitude * np.sin(2 * np.pi * frequency * self.time) * envelope
        
    def electric_energy(self) -> float:
        return 0.5 * self.epsilon * np.sum(self.Ez**2) * self.dx**2
    
    def magnetic_energy(self) -> float:
        return 0.5 * self.mu * np.sum(self.Hx**2 + self.Hy**2) * self.dx**2
    
    def total_energy(self) -> float:
        return self.electric_energy() + self.magnetic_energy()


# ============================================================
# TEST FUNCTIONS
# ============================================================

def test_coulomb():
    """Test Coulomb law: E ∝ 1/r in 2D."""
    print("\n" + "="*60)
    print("TEST 1: Coulomb Law")
    print("="*60)
    
    N = 120
    sim = Electrostatics2D(N, N, dx=1.0)
    
    # Point charge at center
    Q = 1.0
    sim.add_point_charge(N//2, N//2, Q)
    sim.solve()
    
    E = sim.E_magnitude()
    
    # Measure E vs r
    radii = list(range(5, 50, 3))
    E_vals = [E[N//2, N//2 + r] for r in radii]
    
    radii = np.array(radii, dtype=float)
    E_vals = np.array(E_vals)
    
    # Fit E = A/r^n
    log_r = np.log(radii)
    log_E = np.log(E_vals + 1e-12)
    slope, intercept = np.polyfit(log_r, log_E, 1)
    
    print(f"  Point charge Q={Q} at center")
    print(f"  E(r) ∝ r^{slope:.3f}")
    print(f"  2D Coulomb predicts: r^{-1.0}")
    
    passed = abs(slope + 1.0) < 0.15
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed, slope


def test_gauss():
    """Test Gauss's law."""
    print("\n" + "="*60)
    print("TEST 2: Gauss's Law")
    print("="*60)
    
    N = 100
    sim = Electrostatics2D(N, N, dx=1.0, epsilon=1.0)
    
    Q = 10.0
    sim.add_gaussian_charge(N//2, N//2, Q, sigma=3.0)
    sim.solve()
    
    # Compute flux through circles
    radii = [8, 12, 16, 20]
    fluxes = []
    
    for r in radii:
        flux = 0.0
        n_pts = 100
        for i in range(n_pts):
            theta = 2 * np.pi * i / n_pts
            x = N//2 + r * np.cos(theta)
            y = N//2 + r * np.sin(theta)
            
            Ex, Ey = sim.E_at(x, y)
            E_r = Ex * np.cos(theta) + Ey * np.sin(theta)
            dl = 2 * np.pi * r / n_pts
            flux += E_r * dl
            
        fluxes.append(flux)
    
    # In 2D: Φ = Q/ε (line integral)
    expected = Q / sim.epsilon
    
    print(f"  Total charge Q = {Q}")
    print(f"  Expected flux: {expected:.3f}")
    for r, f in zip(radii, fluxes):
        print(f"    r={r}: Φ = {f:.3f} (err: {abs(f-expected)/expected*100:.1f}%)")
    
    mean_flux = np.mean(fluxes)
    error = abs(mean_flux - expected) / expected
    
    passed = error < 0.15
    print(f"  Mean flux: {mean_flux:.3f}, error: {error*100:.1f}%")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed, error


def test_biot_savart():
    """Test B ∝ 1/r from long wire."""
    print("\n" + "="*60)
    print("TEST 3: Biot-Savart Law")
    print("="*60)
    
    N = 100
    sim = Magnetostatics2D(N, N, dx=1.0, mu=1.0)
    
    I = 1.0
    sim.add_wire(N//2, N//2, I)
    sim.solve()
    
    B = sim.B_magnitude()
    
    radii = list(range(3, 40, 3))
    B_vals = [B[N//2, N//2 + r] for r in radii]
    
    radii = np.array(radii, dtype=float)
    B_vals = np.array(B_vals)
    
    log_r = np.log(radii)
    log_B = np.log(B_vals + 1e-12)
    slope, _ = np.polyfit(log_r, log_B, 1)
    
    print(f"  Wire with I={I} at center")
    print(f"  B(r) ∝ r^{slope:.3f}")
    print(f"  Biot-Savart predicts: r^{-1.0}")
    
    passed = abs(slope + 1.0) < 0.15
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed, slope


def test_wave_speed():
    """Test EM wave propagation speed."""
    print("\n" + "="*60)
    print("TEST 4: EM Wave Speed")
    print("="*60)
    
    N = 150
    sim = Electrodynamics2D(N, N, dx=1.0, epsilon=1.0, mu=1.0)
    
    print(f"  c = 1/√(εμ) = {sim.c:.3f}")
    print(f"  dt = {sim.dt:.4f} (Courant stable)")
    
    # Source at left
    source_x, source_y = 30, N//2
    frequency = 0.02
    
    # Detectors
    detectors = [50, 80, 110]
    signals = {d: [] for d in detectors}
    times = []
    
    for step in range(800):
        # Add source
        sim.add_gaussian_source(source_x, source_y, 
                               amplitude=1.0, frequency=frequency, width=5.0)
        sim.step()
        
        times.append(sim.time)
        for d in detectors:
            signals[d].append(sim.Ez[source_y, d])
    
    times = np.array(times)
    
    # Find first significant peak at each detector
    def find_arrival(sig, threshold=0.05):
        for i in range(len(sig)):
            if abs(sig[i]) > threshold:
                return i
        return None
    
    arrivals = {}
    for d in detectors:
        idx = find_arrival(signals[d])
        if idx:
            arrivals[d] = times[idx]
    
    if len(arrivals) >= 2:
        d1, d2 = detectors[0], detectors[1]
        if d1 in arrivals and d2 in arrivals:
            delay = arrivals[d2] - arrivals[d1]
            distance = (d2 - d1) * sim.dx
            c_measured = distance / delay if delay > 0 else 0
            
            print(f"  Wave arrival at x={d1}: t={arrivals[d1]:.2f}")
            print(f"  Wave arrival at x={d2}: t={arrivals[d2]:.2f}")
            print(f"  Measured c = {c_measured:.3f}")
            
            error = abs(c_measured - sim.c) / sim.c
            passed = error < 0.2
        else:
            passed = False
            c_measured = 0
    else:
        print("  Wave did not propagate to detectors")
        passed = False
        c_measured = 0
    
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed, c_measured


def test_superposition():
    """Test superposition principle."""
    print("\n" + "="*60)
    print("TEST 5: Superposition")
    print("="*60)
    
    N = 80
    
    # Charge 1 alone
    sim1 = Electrostatics2D(N, N)
    sim1.add_gaussian_charge(25, 40, 3.0, 2.0)
    sim1.solve()
    
    # Charge 2 alone  
    sim2 = Electrostatics2D(N, N)
    sim2.add_gaussian_charge(55, 40, 5.0, 2.0)
    sim2.solve()
    
    # Both together
    sim_both = Electrostatics2D(N, N)
    sim_both.add_gaussian_charge(25, 40, 3.0, 2.0)
    sim_both.add_gaussian_charge(55, 40, 5.0, 2.0)
    sim_both.solve()
    
    # Test at a point
    test_x, test_y = 60, 40
    phi1 = sim1.phi[test_y, test_x]
    phi2 = sim2.phi[test_y, test_x]
    phi_both = sim_both.phi[test_y, test_x]
    
    error = abs(phi_both - (phi1 + phi2)) / (abs(phi_both) + 1e-9)
    
    print(f"  φ₁ = {phi1:.4f}")
    print(f"  φ₂ = {phi2:.4f}")
    print(f"  φ₁ + φ₂ = {phi1 + phi2:.4f}")
    print(f"  φ_both = {phi_both:.4f}")
    print(f"  Error: {error*100:.3f}%")
    
    passed = error < 0.01
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed, error


def test_dipole():
    """Test dipole field pattern."""
    print("\n" + "="*60)
    print("TEST 6: Dipole Pattern")
    print("="*60)
    
    N = 100
    sim = Electrostatics2D(N, N)
    
    # Dipole along x-axis
    sep = 10
    Q = 5.0
    sim.add_gaussian_charge(N//2 - sep//2, N//2, +Q, 2.0)
    sim.add_gaussian_charge(N//2 + sep//2, N//2, -Q, 2.0)
    sim.solve()
    
    E = sim.E_magnitude()
    
    # E along axis should be ~2x stronger than perpendicular
    r = 20
    E_along = E[N//2, N//2 + r]
    E_perp = E[N//2 + r, N//2]
    ratio = E_along / E_perp
    
    print(f"  Dipole: separation={sep}, Q={Q}")
    print(f"  E along axis (r={r}): {E_along:.4f}")
    print(f"  E perpendicular (r={r}): {E_perp:.4f}")
    print(f"  Ratio: {ratio:.2f} (ideal: 2.0)")
    
    passed = ratio > 1.5
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed, ratio


def test_energy_conservation():
    """Test energy conservation in EM waves."""
    print("\n" + "="*60)
    print("TEST 7: Energy Conservation")
    print("="*60)
    
    N = 80
    sim = Electrodynamics2D(N, N, dx=1.0)
    
    # Initialize with a pulse (no source)
    Y, X = np.mgrid[0:N, 0:N]
    r2 = (X - N//2)**2 + (Y - N//2)**2
    sim.Ez = np.exp(-r2 / 100)
    
    initial_energy = sim.total_energy()
    energies = [initial_energy]
    
    for _ in range(300):
        sim.Jz.fill(0)  # No source
        sim.step()
        energies.append(sim.total_energy())
    
    energies = np.array(energies)
    final_energy = energies[-1]
    
    # Energy should decay due to absorbing BC, but smoothly
    decay = (initial_energy - final_energy) / initial_energy
    
    print(f"  Initial energy: {initial_energy:.4f}")
    print(f"  Final energy: {final_energy:.4f}")
    print(f"  Decay: {decay*100:.1f}%")
    
    # Check for monotonic decay (no blow-up)
    monotonic = all(energies[i] >= energies[i+1] * 0.99 for i in range(len(energies)-1))
    stable = final_energy < initial_energy * 1.1 and final_energy > 0
    
    passed = stable and monotonic
    print(f"  Monotonic decay: {monotonic}")
    print(f"  Stable: {stable}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")
    
    return passed, decay


# ============================================================
# MAIN TEST SUITE
# ============================================================

def run_tests():
    """Run all EM tests."""
    print("="*70)
    print("DET ELECTROMAGNETISM MODULE - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Coulomb Law (1/r)", test_coulomb),
        ("Gauss's Law", test_gauss),
        ("Biot-Savart (1/r)", test_biot_savart),
        ("EM Wave Speed", test_wave_speed),
        ("Superposition", test_superposition),
        ("Dipole Pattern", test_dipole),
        ("Energy Conservation", test_energy_conservation),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed, metric = test_fn()
            results.append((name, passed, metric))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False, None))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed_count = 0
    for name, passed, _ in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name}: {status}")
        if passed:
            passed_count += 1
    
    print(f"\n  Total: {passed_count}/{len(tests)} passed")
    
    return results


if __name__ == "__main__":
    run_tests()
