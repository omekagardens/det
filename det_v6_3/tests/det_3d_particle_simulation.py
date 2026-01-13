"""
DET v6.3 3D Particle Simulation
===============================

Demonstrates particle physics emergence from DET dynamics:
1. Particle creation from vacuum fluctuations
2. Particle collision and scattering
3. Orbital dynamics with angular momentum
4. Particle annihilation into wave modes

Reference: DET Theory Card v6.3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D


@dataclass
class ParticleSimConfig:
    """Configuration for particle simulation scenarios."""
    name: str
    description: str
    N: int = 48
    DT: float = 0.02
    duration: int = 1000
    output_interval: int = 50


class DETParticleSimulator:
    """
    DET-based particle simulator demonstrating emergent particle physics.

    Key phenomena:
    - Particle creation: Coherent resource localization
    - Scattering: Momentum exchange via collision
    - Binding: Gravitational and angular momentum capture
    - Annihilation: Resource dispersal into wave modes
    """

    def __init__(self, config: ParticleSimConfig, params: Optional[DETParams3D] = None):
        self.config = config
        self.params = params or DETParams3D(N=config.N, DT=config.DT)
        self.sim = DETCollider3D(self.params)

        # Recording
        self.history = {
            't': [],
            'total_mass': [],
            'num_particles': [],
            'mean_separation': [],
            'total_momentum': [],
            'total_L': [],
            'PE': []
        }

    def add_particle(self, position: Tuple[int, int, int], mass: float,
                     velocity: Tuple[float, float, float] = (0, 0, 0),
                     q: float = 0.0, spin: float = 0.0):
        """Add a particle-like localized excitation."""
        self.sim.add_packet(position, mass=mass, width=2.5, momentum=velocity, initial_q=q)
        if spin != 0:
            self.sim.add_spin(position, spin=spin, width=3.0)

    def record_state(self, t: int):
        """Record current state for analysis."""
        blobs = self.sim.find_blobs()

        self.history['t'].append(t)
        self.history['total_mass'].append(self.sim.total_mass())
        self.history['num_particles'].append(len(blobs))

        # Mean separation
        if len(blobs) >= 2:
            seps = []
            for i in range(len(blobs)):
                for j in range(i+1, len(blobs)):
                    dx = blobs[j]['x'] - blobs[i]['x']
                    dy = blobs[j]['y'] - blobs[i]['y']
                    dz = blobs[j]['z'] - blobs[i]['z']
                    N = self.params.N
                    dx = dx - N if dx > N/2 else (dx + N if dx < -N/2 else dx)
                    dy = dy - N if dy > N/2 else (dy + N if dy < -N/2 else dy)
                    dz = dz - N if dz > N/2 else (dz + N if dz < -N/2 else dz)
                    seps.append(np.sqrt(dx**2 + dy**2 + dz**2))
            self.history['mean_separation'].append(np.mean(seps))
        else:
            self.history['mean_separation'].append(0)

        # Total momentum
        px = np.sum(self.sim.pi_X)
        py = np.sum(self.sim.pi_Y)
        pz = np.sum(self.sim.pi_Z)
        self.history['total_momentum'].append(np.sqrt(px**2 + py**2 + pz**2))

        # Total angular momentum
        L = self.sim.total_angular_momentum()
        self.history['total_L'].append(np.sqrt(L[0]**2 + L[1]**2 + L[2]**2))

        # Potential energy
        self.history['PE'].append(self.sim.potential_energy())

    def run(self, verbose: bool = True):
        """Run the simulation."""
        if verbose:
            print(f"\nRunning: {self.config.name}")
            print(f"  {self.config.description}")
            print("-" * 60)

        for t in range(self.config.duration):
            if t % self.config.output_interval == 0:
                self.record_state(t)
                if verbose:
                    blobs = self.sim.find_blobs()
                    print(f"  t={t:4d}: particles={len(blobs)}, mass={self.sim.total_mass():.2f}")

            self.sim.step()

        # Final record
        self.record_state(self.config.duration)

        if verbose:
            print("-" * 60)
            print(f"  Final: particles={self.history['num_particles'][-1]}, "
                  f"mass={self.history['total_mass'][-1]:.2f}")

        return self.history


def scenario_1_particle_creation():
    """Scenario 1: Particle creation from coherent fluctuations."""
    config = ParticleSimConfig(
        name="Particle Creation",
        description="Creating particle-like excitations from coherent resource localization",
        N=32,
        duration=500
    )

    params = DETParams3D(
        N=config.N, DT=config.DT,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=True,
        boundary_enabled=True, grace_enabled=True
    )

    sim = DETParticleSimulator(config, params)

    # Create three particles at different positions
    center = config.N // 2
    sim.add_particle((center, center, center-8), mass=10.0, q=0.3)
    sim.add_particle((center, center, center), mass=15.0, q=0.4)
    sim.add_particle((center, center, center+8), mass=10.0, q=0.3)

    return sim.run()


def scenario_2_collision():
    """Scenario 2: Two-particle collision."""
    config = ParticleSimConfig(
        name="Particle Collision",
        description="Head-on collision between two particles with momentum exchange",
        N=48,
        duration=800
    )

    params = DETParams3D(
        N=config.N, DT=config.DT,
        gravity_enabled=True, q_enabled=True,
        momentum_enabled=True, angular_momentum_enabled=True,
        floor_enabled=True,
        boundary_enabled=True, grace_enabled=True
    )

    sim = DETParticleSimulator(config, params)

    center = config.N // 2
    # Two particles approaching each other
    sim.add_particle((center-12, center, center), mass=12.0, velocity=(1.5, 0, 0), q=0.3)
    sim.add_particle((center+12, center, center), mass=12.0, velocity=(-1.5, 0, 0), q=0.3)

    return sim.run()


def scenario_3_orbital_dynamics():
    """Scenario 3: Binary system with angular momentum."""
    config = ParticleSimConfig(
        name="Orbital Dynamics",
        description="Binary system forming bound orbit via angular momentum",
        N=50,
        duration=1500
    )

    params = DETParams3D(
        N=config.N, DT=config.DT,
        gravity_enabled=False,  # Pure angular momentum dynamics
        momentum_enabled=True,
        angular_momentum_enabled=True,
        floor_enabled=True,
        boundary_enabled=True, grace_enabled=True
    )

    sim = DETParticleSimulator(config, params)

    center = config.N // 2
    b = 4  # Impact parameter

    # Two particles with offset approach
    sim.add_particle((center-15, center-b, center), mass=10.0, velocity=(2.0, 0, 0))
    sim.add_particle((center+15, center+b, center), mass=10.0, velocity=(-2.0, 0, 0))

    return sim.run()


def scenario_4_annihilation():
    """Scenario 4: Particle annihilation into wave modes."""
    config = ParticleSimConfig(
        name="Particle Annihilation",
        description="Particle-antiparticle collision with resource dispersal",
        N=40,
        duration=600
    )

    params = DETParams3D(
        N=config.N, DT=config.DT,
        gravity_enabled=False, q_enabled=False,
        momentum_enabled=True,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=False
    )

    sim = DETParticleSimulator(config, params)

    center = config.N // 2
    # High-velocity collision (no structure, pure dispersal)
    sim.add_particle((center-10, center, center), mass=8.0, velocity=(3.0, 0, 0))
    sim.add_particle((center+10, center, center), mass=8.0, velocity=(-3.0, 0, 0))

    return sim.run()


def plot_results(results: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot simulation results."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    scenarios = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Mass evolution
    ax1 = fig.add_subplot(gs[0, 0])
    for i, name in enumerate(scenarios):
        data = results[name]
        ax1.plot(data['t'], data['total_mass'], label=name, color=colors[i])
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Conservation')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Particle count
    ax2 = fig.add_subplot(gs[0, 1])
    for i, name in enumerate(scenarios):
        data = results[name]
        ax2.plot(data['t'], data['num_particles'], label=name, color=colors[i])
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Number of Particles')
    ax2.set_title('Particle Count Evolution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Mean separation
    ax3 = fig.add_subplot(gs[1, 0])
    for i, name in enumerate(scenarios):
        data = results[name]
        ax3.plot(data['t'], data['mean_separation'], label=name, color=colors[i])
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Mean Separation')
    ax3.set_title('Particle Separation')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Total momentum
    ax4 = fig.add_subplot(gs[1, 1])
    for i, name in enumerate(scenarios):
        data = results[name]
        ax4.plot(data['t'], data['total_momentum'], label=name, color=colors[i])
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Total Momentum')
    ax4.set_title('Momentum Evolution')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.close()


def run_all_scenarios():
    """Run all particle simulation scenarios."""
    print("="*70)
    print("DET v6.3 3D PARTICLE SIMULATION")
    print("="*70)

    results = {}

    # Run all scenarios
    results["Creation"] = scenario_1_particle_creation()
    results["Collision"] = scenario_2_collision()
    results["Orbital"] = scenario_3_orbital_dynamics()
    results["Annihilation"] = scenario_4_annihilation()

    # Summary
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)

    for name, data in results.items():
        initial_mass = data['total_mass'][0]
        final_mass = data['total_mass'][-1]
        mass_change = abs(final_mass - initial_mass) / initial_mass * 100

        print(f"\n{name}:")
        print(f"  Initial particles: {data['num_particles'][0]}")
        print(f"  Final particles: {data['num_particles'][-1]}")
        print(f"  Mass change: {mass_change:.2f}%")
        print(f"  Final momentum: {data['total_momentum'][-1]:.4f}")

    # Generate plots
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'particle_simulation_v6_3.png')

    try:
        plot_results(results, save_path=plot_path)
    except Exception as e:
        print(f"\nNote: Could not generate plots ({e})")

    return results


if __name__ == "__main__":
    results = run_all_scenarios()
