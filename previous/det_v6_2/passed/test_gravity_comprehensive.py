"""
Comprehensive Gravity Test Suite for DET v6 Colliders
======================================================

Tests the Newtonian 1/r kernel and gravitational binding (F6) across all dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/ubuntu/det_v6_release/src')

from det_v6_1d_collider_gravity import DETCollider1DGravity, DETParams1D
from det_v6_2d_collider_gravity import DETCollider2DGravity, DETParams2D
from det_v6_3d_collider_gravity import DETCollider3DGravity, DETParams3D


def run_1d_binding_test():
    """Run 1D gravitational binding test."""
    print("\n" + "="*60)
    print("1D GRAVITATIONAL BINDING TEST")
    print("="*60)
    
    params = DETParams1D(
        N=200,
        DT=0.02,
        F_VAC=0.001,
        F_MIN=0.0,
        C_init=0.5,
        momentum_enabled=True,
        alpha_pi=0.1,
        lambda_pi=0.002,
        mu_pi=0.5,
        q_enabled=True,
        alpha_q=0.02,
        a_coupling=3.0,
        a_rate=0.05,
        floor_enabled=False,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=10.0,
        mu_grav=3.0
    )
    
    sim = DETCollider1DGravity(params)
    
    initial_sep = 60
    center = params.N // 2
    sim.add_packet(center - initial_sep//2, mass=8.0, width=5.0, momentum=0.1)
    sim.add_packet(center + initial_sep//2, mass=8.0, width=5.0, momentum=-0.1)
    
    x = np.arange(params.N)
    q_left = 0.3 * np.exp(-0.5 * (x - (center - initial_sep//2))**2 / 5**2)
    q_right = 0.3 * np.exp(-0.5 * (x - (center + initial_sep//2))**2 / 5**2)
    sim.q = q_left + q_right
    
    initial_mass = sim.total_mass()
    
    rec = {'t': [], 'sep': [], 'PE': []}
    snapshots = []
    
    n_steps = 3000
    for t in range(n_steps):
        sep = sim.separation()
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['PE'].append(sim.potential_energy())
        
        if t % 500 == 0:
            snapshots.append((t, sim.F.copy(), sim.Phi.copy()))
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.1f}")
        
        sim.step()
    
    min_sep = min(rec['sep'])
    final_sep = rec['sep'][-1]
    passed = min_sep < initial_sep * 0.5
    
    print(f"\n  Initial sep: {initial_sep}, Final sep: {final_sep:.1f}, Min sep: {min_sep:.1f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    return rec, snapshots, passed


def run_2d_binding_test():
    """Run 2D gravitational binding test."""
    print("\n" + "="*60)
    print("2D GRAVITATIONAL BINDING TEST")
    print("="*60)
    
    params = DETParams2D(
        N=80,
        DT=0.02,
        F_VAC=0.001,
        C_init=0.5,
        momentum_enabled=True,
        alpha_pi=0.1,
        lambda_pi=0.002,
        mu_pi=0.5,
        q_enabled=True,
        alpha_q=0.02,
        a_coupling=3.0,
        a_rate=0.05,
        floor_enabled=False,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=10.0,
        mu_grav=3.0
    )
    
    sim = DETCollider2DGravity(params)
    
    initial_sep = 30
    center = params.N // 2
    sim.add_packet((center, center - initial_sep//2), mass=8.0, width=4.0,
                   momentum=(0, 0.1), initial_q=0.3)
    sim.add_packet((center, center + initial_sep//2), mass=8.0, width=4.0,
                   momentum=(0, -0.1), initial_q=0.3)
    
    rec = {'t': [], 'sep': [], 'PE': []}
    snapshots = []
    
    n_steps = 2000
    for t in range(n_steps):
        sep = sim.separation()
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['PE'].append(sim.potential_energy())
        
        if t % 400 == 0:
            snapshots.append((t, sim.F.copy(), sim.Phi.copy()))
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.1f}")
        
        sim.step()
    
    min_sep = min(rec['sep'])
    final_sep = rec['sep'][-1]
    passed = min_sep < initial_sep * 0.5
    
    print(f"\n  Initial sep: {initial_sep}, Final sep: {final_sep:.1f}, Min sep: {min_sep:.1f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    return rec, snapshots, passed


def run_3d_binding_test():
    """Run 3D gravitational binding test."""
    print("\n" + "="*60)
    print("3D GRAVITATIONAL BINDING TEST")
    print("="*60)
    
    params = DETParams3D(
        N=32,
        DT=0.02,
        F_VAC=0.001,
        F_MIN=0.0,
        C_init=0.3,
        diff_enabled=True,
        momentum_enabled=True,
        alpha_pi=0.1,
        lambda_pi=0.002,
        mu_pi=0.5,
        angular_momentum_enabled=False,
        floor_enabled=False,
        q_enabled=True,
        alpha_q=0.02,
        agency_dynamic=True,
        a_coupling=3.0,
        a_rate=0.05,
        gravity_enabled=True,
        alpha_grav=0.01,
        kappa_grav=10.0,
        mu_grav=3.0
    )
    
    sim = DETCollider3DGravity(params)
    
    initial_sep = 12
    center = params.N // 2
    sim.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, -0.1), initial_q=0.3)
    
    rec = {'t': [], 'sep': [], 'PE': []}
    
    n_steps = 1500
    for t in range(n_steps):
        sep = sim.separation()
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['PE'].append(sim.potential_energy())
        
        if t % 300 == 0:
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.1f}")
        
        sim.step()
    
    min_sep = min(rec['sep'])
    final_sep = rec['sep'][-1]
    passed = min_sep < initial_sep * 0.5
    
    print(f"\n  Initial sep: {initial_sep}, Final sep: {final_sep:.1f}, Min sep: {min_sep:.1f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    
    return rec, passed


def create_visualization(rec_1d, snaps_1d, rec_2d, snaps_2d, rec_3d):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: 1D snapshots
    for i, (t, F, Phi) in enumerate(snaps_1d[:4]):
        ax = fig.add_subplot(4, 4, i+1)
        ax.plot(F, 'b-', label='F (mass)', linewidth=1.5)
        ax.plot(Phi * 0.01, 'r--', label='Φ×0.01', linewidth=1)
        ax.set_title(f'1D t={t}')
        ax.set_xlim(0, len(F))
        if i == 0:
            ax.legend(fontsize=8)
    
    # Row 2: 2D snapshots
    for i, (t, F, Phi) in enumerate(snaps_2d[:4]):
        ax = fig.add_subplot(4, 4, 5+i)
        im = ax.imshow(F, cmap='viridis', origin='lower')
        ax.set_title(f'2D F, t={t}')
        ax.axis('off')
    
    # Row 3: Separation vs time
    ax = fig.add_subplot(4, 2, 5)
    ax.plot(rec_1d['t'], rec_1d['sep'], 'b-', label='1D', linewidth=2)
    ax.axhline(y=60*0.5, color='b', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Separation')
    ax.set_title('1D: Separation vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(4, 2, 6)
    ax.plot(rec_2d['t'], rec_2d['sep'], 'g-', label='2D', linewidth=2)
    ax.axhline(y=30*0.5, color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Separation')
    ax.set_title('2D: Separation vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 4: Potential energy and 3D separation
    ax = fig.add_subplot(4, 2, 7)
    ax.plot(rec_1d['t'], rec_1d['PE'], 'b-', alpha=0.7, label='1D PE')
    ax.plot(rec_2d['t'], np.array(rec_2d['PE'])/100, 'g-', alpha=0.7, label='2D PE/100')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Potential Energy')
    ax.set_title('Potential Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(4, 2, 8)
    ax.plot(rec_3d['t'], rec_3d['sep'], 'r-', label='3D', linewidth=2)
    ax.axhline(y=12*0.5, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Separation')
    ax.set_title('3D: Separation vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/det_v6_release/results/gravity_binding_tests.png', dpi=150)
    plt.close()
    print("\nVisualization saved to results/gravity_binding_tests.png")


def main():
    print("="*70)
    print("DET v6 COMPREHENSIVE GRAVITY TEST SUITE")
    print("="*70)
    
    # Run tests
    rec_1d, snaps_1d, pass_1d = run_1d_binding_test()
    rec_2d, snaps_2d, pass_2d = run_2d_binding_test()
    rec_3d, pass_3d = run_3d_binding_test()
    
    # Create visualization
    create_visualization(rec_1d, snaps_1d, rec_2d, snaps_2d, rec_3d)
    
    # Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*70)
    print(f"  1D Gravitational Binding (F6): {'PASS' if pass_1d else 'FAIL'}")
    print(f"  2D Gravitational Binding (F6): {'PASS' if pass_2d else 'FAIL'}")
    print(f"  3D Gravitational Binding (F6): {'PASS' if pass_3d else 'FAIL'}")
    print(f"\n  Overall: {'ALL PASS' if all([pass_1d, pass_2d, pass_3d]) else 'SOME FAILED'}")
    
    return all([pass_1d, pass_2d, pass_3d])


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
