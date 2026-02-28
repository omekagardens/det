"""
DET Particle Tracker - Discrete Particle Dynamics on DET Fields
================================================================

Purpose: Enable Keplerian orbital dynamics by tracking discrete particles
that interact with DET's gravitational field while maintaining localization.

This is a hybrid approach:
- DET field dynamics generate the gravitational potential Φ
- Discrete particles move according to Newtonian mechanics in that potential
- Particles maintain their identity and don't spread diffusively

This allows testing whether DET's gravity produces correct Keplerian orbits
when the "particle spreading" problem is removed.

Theory Card Reference: Section V (Gravity Module)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


@dataclass
class Particle:
    """A discrete particle with position, velocity, and mass."""
    x: float  # Position
    y: float
    z: float
    vx: float = 0.0  # Velocity
    vy: float = 0.0
    vz: float = 0.0
    mass: float = 1.0
    q: float = 0.0  # Structure (sources gravity)
    fixed: bool = False  # If True, particle doesn't move (central mass)

    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def velocity(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vz)

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * (self.vx**2 + self.vy**2 + self.vz**2)

    def angular_momentum(self, origin: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[float, float, float]:
        """Compute L = r × (m*v) relative to origin."""
        rx = self.x - origin[0]
        ry = self.y - origin[1]
        rz = self.z - origin[2]

        px = self.mass * self.vx
        py = self.mass * self.vy
        pz = self.mass * self.vz

        Lx = ry * pz - rz * py
        Ly = rz * px - rx * pz
        Lz = rx * py - ry * px

        return (Lx, Ly, Lz)


@dataclass
class ParticleTrackerParams:
    """Parameters for particle tracker."""
    # Integration
    dt: float = 0.01  # Time step for particle dynamics
    integrator: str = "leapfrog"  # "euler", "leapfrog", "rk4"

    # Coupling to DET
    gravity_coupling: float = 1.0  # Scale factor for gravity
    use_det_gravity: bool = True  # Use DET's Φ field
    use_newtonian_gravity: bool = False  # Use direct N-body calculation

    # Boundary handling
    periodic: bool = True  # Wrap particles at boundaries

    # Diagnostics
    track_energy: bool = True
    track_angular_momentum: bool = True


class ParticleTracker:
    """
    Discrete particle dynamics coupled to DET gravitational field.

    This hybrid system:
    1. Uses DET to generate gravitational potential Φ from structure q
    2. Moves discrete particles according to a = -∇Φ
    3. Maintains particle identity (no spreading)
    """

    def __init__(self, det_sim: DETCollider3D, params: ParticleTrackerParams = None):
        """
        Initialize particle tracker.

        Args:
            det_sim: DET simulation providing gravitational field
            params: Tracker parameters
        """
        self.sim = det_sim
        self.p = params or ParticleTrackerParams()
        self.particles: List[Particle] = []
        self.N = det_sim.p.N

        # History
        self.time = 0.0
        self.step_count = 0
        self.energy_history = []
        self.L_history = []

    def add_particle(self, x: float, y: float, z: float,
                     vx: float = 0, vy: float = 0, vz: float = 0,
                     mass: float = 1.0, q: float = 0.0,
                     fixed: bool = False) -> int:
        """Add a particle and return its index."""
        particle = Particle(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz,
                           mass=mass, q=q, fixed=fixed)
        self.particles.append(particle)

        # Also add to DET field if particle has structure
        if q > 0:
            self.sim.add_packet(
                (int(z), int(y), int(x)),  # DET uses (z, y, x) indexing
                mass=mass,
                width=2.0,
                momentum=(0, 0, 0),
                initial_q=q
            )

        return len(self.particles) - 1

    def get_gravity_at(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Get gravitational acceleration at position from DET field.

        Uses trilinear interpolation for smooth forces.
        """
        N = self.N

        # Handle periodic boundaries
        x = x % N
        y = y % N
        z = z % N

        # Get integer indices
        ix = int(x)
        iy = int(y)
        iz = int(z)

        # Fractional parts for interpolation
        fx = x - ix
        fy = y - iy
        fz = z - iz

        # Neighbor indices (with wrapping)
        ix1 = (ix + 1) % N
        iy1 = (iy + 1) % N
        iz1 = (iz + 1) % N

        # Trilinear interpolation for each component
        def interp(field):
            c000 = field[iz, iy, ix]
            c001 = field[iz, iy, ix1]
            c010 = field[iz, iy1, ix]
            c011 = field[iz, iy1, ix1]
            c100 = field[iz1, iy, ix]
            c101 = field[iz1, iy, ix1]
            c110 = field[iz1, iy1, ix]
            c111 = field[iz1, iy1, ix1]

            c00 = c000 * (1 - fx) + c001 * fx
            c01 = c010 * (1 - fx) + c011 * fx
            c10 = c100 * (1 - fx) + c101 * fx
            c11 = c110 * (1 - fx) + c111 * fx

            c0 = c00 * (1 - fy) + c01 * fy
            c1 = c10 * (1 - fy) + c11 * fy

            return c0 * (1 - fz) + c1 * fz

        # Note: DET gx, gy, gz are the gravitational field components
        # g = -∇Φ, so acceleration a = g (already includes minus sign)
        gx = interp(self.sim.gx) * self.p.gravity_coupling
        gy = interp(self.sim.gy) * self.p.gravity_coupling
        gz = interp(self.sim.gz) * self.p.gravity_coupling

        return (gx, gy, gz)

    def get_newtonian_gravity_at(self, particle_idx: int) -> Tuple[float, float, float]:
        """
        Compute direct N-body gravitational acceleration on particle.

        a_i = -G Σ_j m_j (r_i - r_j) / |r_i - r_j|³
        """
        p = self.particles[particle_idx]
        ax = ay = az = 0.0
        G = 1.0  # Gravitational constant

        for j, other in enumerate(self.particles):
            if j == particle_idx:
                continue

            dx = p.x - other.x
            dy = p.y - other.y
            dz = p.z - other.z

            # Handle periodic boundaries
            N = self.N
            if abs(dx) > N/2: dx -= np.sign(dx) * N
            if abs(dy) > N/2: dy -= np.sign(dy) * N
            if abs(dz) > N/2: dz -= np.sign(dz) * N

            r2 = dx**2 + dy**2 + dz**2
            if r2 < 1.0:  # Softening
                r2 = 1.0
            r3 = r2 * np.sqrt(r2)

            ax -= G * other.mass * dx / r3
            ay -= G * other.mass * dy / r3
            az -= G * other.mass * dz / r3

        return (ax, ay, az)

    def step_euler(self):
        """Simple Euler integration."""
        dt = self.p.dt

        for i, p in enumerate(self.particles):
            if p.fixed:
                continue

            # Get acceleration
            if self.p.use_det_gravity:
                ax, ay, az = self.get_gravity_at(p.x, p.y, p.z)
            elif self.p.use_newtonian_gravity:
                ax, ay, az = self.get_newtonian_gravity_at(i)
            else:
                ax = ay = az = 0

            # Update velocity
            p.vx += ax * dt
            p.vy += ay * dt
            p.vz += az * dt

            # Update position
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt

            # Periodic boundaries
            if self.p.periodic:
                p.x = p.x % self.N
                p.y = p.y % self.N
                p.z = p.z % self.N

    def step_leapfrog(self):
        """
        Leapfrog (Störmer-Verlet) integration.

        Better energy conservation than Euler.
        """
        dt = self.p.dt

        # Half-step velocity update
        for i, p in enumerate(self.particles):
            if p.fixed:
                continue

            if self.p.use_det_gravity:
                ax, ay, az = self.get_gravity_at(p.x, p.y, p.z)
            elif self.p.use_newtonian_gravity:
                ax, ay, az = self.get_newtonian_gravity_at(i)
            else:
                ax = ay = az = 0

            p.vx += 0.5 * ax * dt
            p.vy += 0.5 * ay * dt
            p.vz += 0.5 * az * dt

        # Full-step position update
        for p in self.particles:
            if p.fixed:
                continue

            p.x += p.vx * dt
            p.y += p.vy * dt
            p.z += p.vz * dt

            if self.p.periodic:
                p.x = p.x % self.N
                p.y = p.y % self.N
                p.z = p.z % self.N

        # Half-step velocity update (with new positions)
        for i, p in enumerate(self.particles):
            if p.fixed:
                continue

            if self.p.use_det_gravity:
                ax, ay, az = self.get_gravity_at(p.x, p.y, p.z)
            elif self.p.use_newtonian_gravity:
                ax, ay, az = self.get_newtonian_gravity_at(i)
            else:
                ax = ay = az = 0

            p.vx += 0.5 * ax * dt
            p.vy += 0.5 * ay * dt
            p.vz += 0.5 * az * dt

    def step(self):
        """Execute one integration step."""
        # First update DET field (for gravity source)
        self.sim.step()

        # Then update particles
        if self.p.integrator == "euler":
            self.step_euler()
        elif self.p.integrator == "leapfrog":
            self.step_leapfrog()
        else:
            self.step_leapfrog()  # Default

        # Track diagnostics
        if self.p.track_energy:
            E = self.total_energy()
            self.energy_history.append(E)

        if self.p.track_angular_momentum:
            L = self.total_angular_momentum()
            self.L_history.append(L)

        self.time += self.p.dt
        self.step_count += 1

    def total_kinetic_energy(self) -> float:
        """Total kinetic energy of all particles."""
        return sum(p.kinetic_energy() for p in self.particles)

    def total_potential_energy(self) -> float:
        """Total potential energy from DET field."""
        E = 0.0
        for p in self.particles:
            if not p.fixed:
                # Φ at particle position
                ix, iy, iz = int(p.x) % self.N, int(p.y) % self.N, int(p.z) % self.N
                Phi = self.sim.Phi[iz, iy, ix]
                E += p.mass * Phi
        return E

    def total_energy(self) -> float:
        """Total mechanical energy."""
        return self.total_kinetic_energy() + self.total_potential_energy()

    def total_angular_momentum(self, origin: Tuple[float, float, float] = None) -> Tuple[float, float, float]:
        """Total angular momentum of all particles about origin."""
        if origin is None:
            # Use center of grid
            origin = (self.N / 2, self.N / 2, self.N / 2)

        Lx = Ly = Lz = 0.0
        for p in self.particles:
            L = p.angular_momentum(origin)
            Lx += L[0]
            Ly += L[1]
            Lz += L[2]

        return (Lx, Ly, Lz)


def test_kepler_with_particle_tracker():
    """
    Test Kepler's Third Law using discrete particle dynamics
    coupled to DET gravity field.
    """
    print("="*70)
    print("KEPLER TEST WITH DET PARTICLE TRACKER")
    print("="*70)

    N = 64
    center = N // 2

    # Create DET simulation with central mass
    det_params = DETParams3D(
        N=N,
        DT=0.01,
        F_VAC=0.001,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=15.0,
        mu_grav=2.0,
        q_enabled=True,
        alpha_q=0.0,  # No further q accumulation
        momentum_enabled=False,  # Don't need DET momentum
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    det_sim = DETCollider3D(det_params)

    # Add central mass to DET field (stationary, sources gravity via q)
    det_sim.add_packet((center, center, center), mass=100.0, width=3.0,
                       momentum=(0, 0, 0), initial_q=0.9)

    # Let gravity field establish
    print("\nEstablishing gravity field...")
    for _ in range(100):
        det_sim.step()

    # Check gravity field
    print("\nGravitational field profile:")
    print("-"*50)
    for r in [6, 8, 10, 12]:
        g_mag = np.sqrt(
            det_sim.gx[center, center, center + r]**2 +
            det_sim.gy[center, center, center + r]**2 +
            det_sim.gz[center, center, center + r]**2
        )
        print(f"  r={r}: |g| = {g_mag:.4f}")

    # Create particle tracker
    tracker_params = ParticleTrackerParams(
        dt=0.02,
        integrator="leapfrog",
        use_det_gravity=True,
        gravity_coupling=1.0,
        track_energy=True,
        track_angular_momentum=True
    )

    tracker = ParticleTracker(det_sim, tracker_params)

    # Add central mass (fixed)
    tracker.add_particle(center, center, center, mass=100.0, q=0.9, fixed=True)

    # Test orbits at different radii
    results = []
    for r_orbit in [8, 10, 12]:
        print(f"\n{'='*50}")
        print(f"Testing orbit at r = {r_orbit}")
        print("="*50)

        # Create fresh tracker for each test
        det_sim2 = DETCollider3D(det_params)
        det_sim2.add_packet((center, center, center), mass=100.0, width=3.0,
                           momentum=(0, 0, 0), initial_q=0.9)
        for _ in range(100):
            det_sim2.step()

        tracker2 = ParticleTracker(det_sim2, tracker_params)
        tracker2.add_particle(center, center, center, mass=100.0, q=0.9, fixed=True)

        # Get circular velocity from DET gravity
        g_at_r = np.sqrt(
            det_sim2.gx[center, center, center + r_orbit]**2 +
            det_sim2.gy[center, center, center + r_orbit]**2 +
            det_sim2.gz[center, center, center + r_orbit]**2
        )
        v_circ = np.sqrt(r_orbit * g_at_r) if g_at_r > 0 else 1.0

        print(f"|g| at r={r_orbit}: {g_at_r:.4f}")
        print(f"Circular velocity: {v_circ:.4f}")

        # Add test particle with tangential velocity
        tracker2.add_particle(
            center, center, center + r_orbit,  # Position: offset in Z
            vx=0, vy=v_circ, vz=0,  # Velocity: tangential in Y
            mass=1.0
        )

        # Track orbit
        initial_L = tracker2.total_angular_momentum()
        initial_L_mag = np.sqrt(initial_L[0]**2 + initial_L[1]**2 + initial_L[2]**2)

        print(f"Initial L = ({initial_L[0]:.2f}, {initial_L[1]:.2f}, {initial_L[2]:.2f})")

        # Run orbit
        angles = []
        radii = []
        prev_angle = 0
        total_angle = 0

        print("\nOrbit evolution:")
        print("-"*50)

        for t in range(3000):
            tracker2.step()

            p = tracker2.particles[1]  # Test particle

            # Compute angle in YZ plane
            dy = p.y - center
            dz = p.z - center
            r_current = np.sqrt((p.x - center)**2 + dy**2 + dz**2)
            angle = np.arctan2(dy, dz)

            # Track cumulative angle
            d_angle = angle - prev_angle
            if d_angle > np.pi: d_angle -= 2*np.pi
            if d_angle < -np.pi: d_angle += 2*np.pi
            total_angle += d_angle
            prev_angle = angle

            angles.append(total_angle)
            radii.append(r_current)

            if t % 500 == 0:
                orbits = abs(total_angle) / (2 * np.pi)
                L = tracker2.total_angular_momentum()
                L_mag = np.sqrt(L[0]**2 + L[1]**2 + L[2]**2)
                L_ratio = L_mag / initial_L_mag if initial_L_mag > 0 else 0

                print(f"  t={t}: r={r_current:.2f}, orbits={orbits:.2f}, L_ratio={L_ratio:.4f}")

        # Analyze results
        num_orbits = abs(angles[-1]) / (2 * np.pi)
        mean_r = np.mean(radii)
        eccentricity = (np.max(radii) - np.min(radii)) / (np.max(radii) + np.min(radii))

        if num_orbits > 0.5:
            period = len(angles) * tracker_params.dt / num_orbits
            t2_r3 = (period ** 2) / (r_orbit ** 3)
        else:
            period = 0
            t2_r3 = 0

        final_L = tracker2.total_angular_momentum()
        final_L_mag = np.sqrt(final_L[0]**2 + final_L[1]**2 + final_L[2]**2)
        L_conservation = final_L_mag / initial_L_mag if initial_L_mag > 0 else 0

        print("\nResults:")
        print(f"  Orbits completed: {num_orbits:.2f}")
        print(f"  Period: {period:.2f}")
        print(f"  Eccentricity: {eccentricity:.4f}")
        print(f"  T²/r³: {t2_r3:.4f}")
        print(f"  L conservation: {L_conservation:.4f}")

        results.append({
            'r': r_orbit,
            'period': period,
            'orbits': num_orbits,
            'ecc': eccentricity,
            't2_r3': t2_r3,
            'L_conservation': L_conservation
        })

    # Summary
    print("\n" + "="*70)
    print("KEPLER'S THIRD LAW ANALYSIS")
    print("="*70)

    print("\n" + "-"*60)
    print(f"{'r':>6} {'Period':>10} {'Orbits':>8} {'Ecc':>8} {'T²/r³':>12}")
    print("-"*60)

    t2_r3_values = []
    for res in results:
        print(f"{res['r']:>6} {res['period']:>10.2f} {res['orbits']:>8.2f} "
              f"{res['ecc']:>8.4f} {res['t2_r3']:>12.4f}")
        if res['orbits'] >= 1.0:
            t2_r3_values.append(res['t2_r3'])

    if len(t2_r3_values) >= 2:
        mean_ratio = np.mean(t2_r3_values)
        std_ratio = np.std(t2_r3_values)
        cv = std_ratio / mean_ratio if mean_ratio > 0 else float('inf')

        print("-"*60)
        print(f"Mean T²/r³ = {mean_ratio:.4f}, CV = {cv:.4f}")

        if cv < 0.20:
            print("\n✓ KEPLER'S THIRD LAW SATISFIED!")
            print("  DET gravity with particle tracking produces Keplerian orbits.")
        else:
            print("\n✗ KEPLER'S THIRD LAW NOT SATISFIED")
            print(f"  T²/r³ varies by {cv*100:.1f}% (threshold: 20%)")


if __name__ == "__main__":
    test_kepler_with_particle_tracker()
