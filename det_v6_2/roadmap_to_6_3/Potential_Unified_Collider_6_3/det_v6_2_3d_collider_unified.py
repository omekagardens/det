        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = label(above)
        
        if num < 2:
            return 0.0
        
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        
        coms = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) == 0:
                continue
            weights = self.F[mask]
            total_mass = np.sum(weights)
            if total_mass < 0.1:
                continue
            cx = np.sum(x[mask] * weights) / total_mass
            cy = np.sum(y[mask] * weights) / total_mass
            cz = np.sum(z[mask] * weights) / total_mass
            coms.append({'x': cx, 'y': cy, 'z': cz, 'mass': total_mass})
        
        coms.sort(key=lambda c: -c['mass'])
        
        if len(coms) < 2:
            return 0.0
        
        dx = coms[1]['x'] - coms[0]['x']
        dy = coms[1]['y'] - coms[0]['y']
        dz = coms[1]['z'] - coms[0]['z']
        
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N
        if dy > N/2: dy -= N
        if dy < -N/2: dy += N
        if dz > N/2: dz -= N
        if dz < -N/2: dz += N
        
        return np.sqrt(dx**2 + dy**2 + dz**2)


# ============================================================
# FULL TEST SUITE
# ============================================================

def test_gravity_vacuum(verbose: bool = True) -> bool:
    """Gravity has no effect in vacuum (q=0)"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravity in Vacuum")
        print("="*60)
    
    params = DETParams3D(N=16, gravity_enabled=True, q_enabled=False, boundary_enabled=False)
    sim = DETCollider3DUnified(params)
    
    for _ in range(200):
        sim.step()
    
    max_g = np.max(np.abs(sim.gx)) + np.max(np.abs(sim.gy)) + np.max(np.abs(sim.gz))
    max_Phi = np.max(np.abs(sim.Phi))
    
    passed = max_g < 1e-10 and max_Phi < 1e-10
    
    if verbose:
        print(f"  Max |g|: {max_g:.2e}")
        print(f"  Max |Φ|: {max_Phi:.2e}")
        print(f"  Vacuum gravity {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F7_mass_conservation(verbose: bool = True) -> bool:
    """F7: Mass conservation with gravity + boundary"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F7: Mass Conservation (Gravity + Boundary)")
        print("="*60)
    
    params = DETParams3D(N=24, F_MIN=0.0, gravity_enabled=True, boundary_enabled=True)
    sim = DETCollider3DUnified(params)
    sim.add_packet((8, 8, 8), mass=10.0, width=3.0, momentum=(0.2, 0.2, 0.2))
    sim.add_packet((16, 16, 16), mass=10.0, width=3.0, momentum=(-0.2, -0.2, -0.2))
    
    initial_mass = sim.total_mass()
    
    for t in range(500):
        sim.step()
    
    final_mass = sim.total_mass()
    grace_added = sim.total_grace_injected
    effective_drift = abs(final_mass - initial_mass - grace_added) / initial_mass
    
    passed = effective_drift < 0.10
    
    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Grace added: {grace_added:.4f}")
        print(f"  Effective drift: {effective_drift*100:.4f}%")
        print(f"  F7 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F6_gravitational_binding(verbose: bool = True) -> Dict:
    """F6: Gravitational binding"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F6: Gravitational Binding (3D)")
        print("="*60)
    
    params = DETParams3D(
        N=32, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        C_init=0.3, diff_enabled=True,
        momentum_enabled=True, alpha_pi=0.1, lambda_pi=0.002, mu_pi=0.5,
        angular_momentum_enabled=False, floor_enabled=False,
        q_enabled=True, alpha_q=0.02,
        agency_dynamic=True, a_coupling=3.0, a_rate=0.05,
        gravity_enabled=True, alpha_grav=0.01, kappa_grav=10.0, mu_grav=3.0,
        boundary_enabled=True, grace_enabled=True
    )
    
    sim = DETCollider3DUnified(params)
    
    initial_sep = 12
    center = params.N // 2
    sim.add_packet((center, center, center - initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, 0.1), initial_q=0.3)
    sim.add_packet((center, center, center + initial_sep//2), mass=8.0, width=2.5,
                   momentum=(0, 0, -0.1), initial_q=0.3)
    
    rec = {'t': [], 'sep': [], 'PE': []}
    
    for t in range(1500):
        sep = sim.separation()
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['PE'].append(sim.potential_energy())
        
        if verbose and t % 300 == 0:
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.3f}")
        
        sim.step()
    
    initial_sep_m = rec['sep'][0] if rec['sep'][0] > 0 else initial_sep
    final_sep = rec['sep'][-1]
    min_sep = min(rec['sep'])
    
    sep_decreased = final_sep < initial_sep_m * 0.9
    bound_state = min_sep < initial_sep_m * 0.5
    
    rec['passed'] = sep_decreased or bound_state
    rec['initial_sep'] = initial_sep_m
    rec['final_sep'] = final_sep
    rec['min_sep'] = min_sep
    
    if verbose:
        print(f"\n  Initial sep: {initial_sep_m:.1f}, Final: {final_sep:.1f}, Min: {min_sep:.1f}")
        print(f"  F6 {'PASSED' if rec['passed'] else 'FAILED'}")
    
    return rec


def test_F2_grace_coercion(verbose: bool = True) -> bool:
    """F2: Grace doesn't go to a=0 nodes"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F2: Grace Coercion (a=0 blocks grace)")
        print("="*60)
    
    params = DETParams3D(
        N=24, boundary_enabled=True, grace_enabled=True,
        F_MIN_grace=0.15, a_rate=0.0, gravity_enabled=True
    )
    sim = DETCollider3DUnified(params)
    
    sim.add_packet((12, 12, 6), mass=3.0, width=2.0, momentum=(0, 0, 0.3))
    sim.add_packet((12, 12, 18), mass=3.0, width=2.0, momentum=(0, 0, -0.3))
    
    sz, sy, sx = 12, 12, 12
    sim.a[sz, sy, sx] = 0.0
    sim.F[sz, sy, sx] = 0.01
    
    for _ in range(200):
        sim.step()
    
    sentinel_grace = sim.last_grace_injection[sz, sy, sx]
    
    passed = sentinel_grace == 0.0
    
    if verbose:
        print(f"  Sentinel a = {sim.a[sz, sy, sx]:.4f}")
        print(f"  Sentinel grace = {sentinel_grace:.2e}")
        print(f"  F2 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F3_boundary_redundancy(verbose: bool = True) -> bool:
    """F3: Boundary ON produces different outcome than OFF"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F3: Boundary Redundancy")
        print("="*60)
    
    def run_scenario(boundary_on: bool):
        params = DETParams3D(
            N=24, F_VAC=0.02,
            boundary_enabled=boundary_on, grace_enabled=True, F_MIN_grace=0.15,
            gravity_enabled=True
        )
        sim = DETCollider3DUnified(params)
        sim.add_packet((12, 12, 6), mass=2.0, width=2.0, momentum=(0, 0, 0.3))
        sim.add_packet((12, 12, 18), mass=2.0, width=2.0, momentum=(0, 0, -0.3))
        
        for _ in range(300):
            sim.step()
        
        return np.mean(sim.F[10:14, 10:14, 10:14]), sim.total_grace_injected
    
    F_off, grace_off = run_scenario(False)
    F_on, grace_on = run_scenario(True)
    
    passed = grace_on > grace_off + 0.001
    
    if verbose:
        print(f"  OFF: F={F_off:.4f}, grace={grace_off:.4f}")
        print(f"  ON:  F={F_on:.4f}, grace={grace_on:.4f}")
        print(f"  F3 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F8_vacuum_momentum(verbose: bool = True) -> bool:
    """F8: Momentum doesn't push vacuum"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F8: Vacuum Momentum")
        print("="*60)
    
    params = DETParams3D(
        N=16, momentum_enabled=True, q_enabled=False, floor_enabled=False,
        F_MIN=0.0, gravity_enabled=False, boundary_enabled=False
    )
    sim = DETCollider3DUnified(params)
    sim.F = np.ones_like(sim.F) * params.F_VAC
    sim.pi_X = np.ones_like(sim.pi_X) * 1.0
    
    initial_mass = sim.total_mass()
    
    for _ in range(200):
        sim.step()
    
    final_mass = sim.total_mass()
    drift = abs(final_mass - initial_mass) / initial_mass
    
    passed = drift < 0.01
    
    if verbose:
        print(f"  Mass drift: {drift*100:.4f}%")
        print(f"  F8 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F9_symmetry_drift(verbose: bool = True) -> bool:
    """F9: Symmetric IC doesn't drift"""
    if verbose:
        print("\n" + "="*60)
        print("TEST F9: Symmetry Drift")
        print("="*60)
    
    params = DETParams3D(N=20, momentum_enabled=False, gravity_enabled=False, boundary_enabled=False)
    sim = DETCollider3DUnified(params)
    
    N = params.N
    sim.add_packet((N//2, N//2, N//4), mass=5.0, width=3.0, momentum=(0, 0, 0))
    sim.add_packet((N//2, N//2, 3*N//4), mass=5.0, width=3.0, momentum=(0, 0, 0))
    
    initial_com = sim.center_of_mass()
    
    max_drift = 0
    for _ in range(300):
        com = sim.center_of_mass()
        drift = np.sqrt((com[0] - initial_com[0])**2 + 
                       (com[1] - initial_com[1])**2 + 
                       (com[2] - initial_com[2])**2)
        max_drift = max(max_drift, drift)
        sim.step()
    
    passed = max_drift < 1.0
    
    if verbose:
        print(f"  Max COM drift: {max_drift:.4f} cells")
        print(f"  F9 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def run_full_test_suite():
    """Run complete 3D test suite."""
    print("="*70)
    print("DET v6.2 3D COLLIDER UNIFIED - FULL TEST SUITE")
    print("Gravity + Boundary Operators")
    print("="*70)
    
    results = {}
    
    results['vacuum_gravity'] = test_gravity_vacuum(verbose=True)
    results['F7'] = test_F7_mass_conservation(verbose=True)
    results['F6'] = test_F6_gravitational_binding(verbose=True)
    results['F2'] = test_F2_grace_coercion(verbose=True)
    results['F3'] = test_F3_boundary_redundancy(verbose=True)
    results['F8'] = test_F8_vacuum_momentum(verbose=True)
    results['F9'] = test_F9_symmetry_drift(verbose=True)
    
    print("\n" + "="*70)
    print("3D SUITE SUMMARY")
    print("="*70)
    print(f"  Vacuum gravity: {'PASS' if results['vacuum_gravity'] else 'FAIL'}")
    print(f"  F7 (Mass conservation): {'PASS' if results['F7'] else 'FAIL'}")
    print(f"  F6 (Gravitational binding): {'PASS' if results['F6']['passed'] else 'FAIL'}")
    print(f"  F2 (Grace coercion): {'PASS' if results['F2'] else 'FAIL'}")
    print(f"  F3 (Boundary redundancy): {'PASS' if results['F3'] else 'FAIL'}")
    print(f"  F8 (Vacuum momentum): {'PASS' if results['F8'] else 'FAIL'}")
    print(f"  F9 (Symmetry drift): {'PASS' if results['F9'] else 'FAIL'}")
    
    all_passed = (results['vacuum_gravity'] and results['F7'] and 
                  results['F6']['passed'] and results['F2'] and 
                  results['F3'] and results['F8'] and results['F9'])
    print(f"\n  OVERALL: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    
    return results


if __name__ == "__main__":
    run_full_test_suite()
