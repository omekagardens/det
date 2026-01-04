"""
Deep Existence Theory - Core Simulation Infrastructure
======================================================
Foundation module for all falsifier tests.
Version: 4.2
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Creature:
    F: float = 1.0; tau: float = 0.0; sigma: float = 1.0; a: float = 0.5
    theta: float = 0.0; k: int = 0; q: float = 0.0; F_op: float = 0.1; H: float = 0.1

@dataclass 
class Bond:
    i: int; j: int; sigma_ij: float = 1.0; C_ij: float = 0.5

@dataclass
class DETSystem:
    creatures: List[Creature]
    bonds: List[Bond]
    epsilon: float = 1e-10
    F_min: float = 0.1
    neighborhood_radius: int = 2
    alpha_q: float = 0.1
    
    def __post_init__(self):
        self.n = len(self.creatures)
        self.adj = {i: [] for i in range(self.n)}
        self.bond_map = {}
        for idx, bond in enumerate(self.bonds):
            self.adj[bond.i].append(bond.j)
            self.adj[bond.j].append(bond.i)
            self.bond_map[(bond.i, bond.j)] = idx
            self.bond_map[(bond.j, bond.i)] = idx
    
    def get_neighborhood(self, i: int, radius: int) -> List[int]:
        visited = {i}; frontier = {i}
        for _ in range(radius):
            new_frontier = set()
            for node in frontier:
                for neighbor in self.adj.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor); new_frontier.add(neighbor)
            frontier = new_frontier
        return list(visited)

def compute_presence(c: Creature, eps: float = 1e-10) -> float:
    return c.a * c.sigma / max((1 + c.F_op) * (1 + c.H), eps)

def compute_psi(sys: DETSystem, i: int) -> complex:
    nbhd = sys.get_neighborhood(i, sys.neighborhood_radius)
    local_F = sum(max(sys.creatures[k].F, 0) for k in nbhd) + sys.epsilon
    return np.sqrt(max(sys.creatures[i].F, 0) / local_F) * np.exp(1j * sys.creatures[i].theta)

def compute_flow(sys: DETSystem, bond: Bond) -> float:
    psi_i, psi_j = compute_psi(sys, bond.i), compute_psi(sys, bond.j)
    quantum = np.sqrt(bond.C_ij) * np.imag(np.conj(psi_i) * psi_j)
    classical = (1 - np.sqrt(bond.C_ij)) * (sys.creatures[bond.i].F - sys.creatures[bond.j].F)
    return np.clip(bond.sigma_ij * (quantum + classical), -sys.creatures[bond.i].F * 0.5, sys.creatures[bond.i].F * 0.5)

def compute_dissipation(sys: DETSystem, i: int, flows: Dict, dt: float) -> float:
    return sum(abs(flows.get((min(i,j), max(i,j)), 0)) for j in sys.adj.get(i, [])) * dt

def compute_grace(sys: DETSystem, i: int, dissipations: Dict) -> float:
    c = sys.creatures[i]
    n_i = max(0.0, sys.F_min - c.F)
    if c.a < sys.epsilon or n_i < sys.epsilon: return 0.0
    nbhd = sys.get_neighborhood(i, sys.neighborhood_radius)
    Z = sum(sys.creatures[l].a * max(0.0, sys.F_min - sys.creatures[l].F) for l in nbhd) + sys.epsilon
    return dissipations.get(i, 0.0) * (c.a * n_i / Z)

def compute_reconciliation(sys: DETSystem, bond: Bond, dissipations: Dict) -> float:
    c_i, c_j = sys.creatures[bond.i], sys.creatures[bond.j]
    if c_i.a < sys.epsilon or c_j.a < sys.epsilon: return 0.0
    u = c_i.a * c_j.a * (1 - bond.C_ij)
    nodes = set(sys.get_neighborhood(bond.i, sys.neighborhood_radius)) | set(sys.get_neighborhood(bond.j, sys.neighborhood_radius))
    Z = sum(sys.creatures[b.i].a * sys.creatures[b.j].a * (1 - b.C_ij) for b in sys.bonds if b.i in nodes and b.j in nodes) + sys.epsilon
    D = 0.5 * (dissipations.get(bond.i, 0.0) + dissipations.get(bond.j, 0.0))
    return D * (u / Z)

def step_system(sys: DETSystem, dt: float = 0.1, boundary: bool = True, noise_a: float = 0.0, noise_q: float = 0.0) -> Dict:
    flows = {(min(b.i, b.j), max(b.i, b.j)): compute_flow(sys, b) for b in sys.bonds}
    dissipations = {i: compute_dissipation(sys, i, flows, dt) for i in range(sys.n)}
    
    injections = {i: compute_grace(sys, i, dissipations) if boundary else 0.0 for i in range(sys.n)}
    reconciliations = {(b.i, b.j): compute_reconciliation(sys, b, dissipations) if boundary else 0.0 for b in sys.bonds}
    
    delta_F, new_F = {}, {}
    for i in range(sys.n):
        net = sum((flows.get((min(i,j), max(i,j)), 0) * (1 if i < j else -1)) for j in sys.adj.get(i, []))
        delta_F[i] = -net * dt + injections[i]
        new_F[i] = max(0.0, sys.creatures[i].F + delta_F[i])
    
    new_q = {i: np.clip(sys.creatures[i].q + sys.alpha_q * max(0, -delta_F[i]) + noise_q * np.random.randn(), 0, 1) for i in range(sys.n)}
    
    new_a = {}
    for i in range(sys.n):
        P_i = compute_presence(sys.creatures[i])
        P_bar = np.mean([compute_presence(sys.creatures[l]) for l in sys.get_neighborhood(i, 1)])
        new_a[i] = np.clip(sys.creatures[i].a + (P_i - P_bar) - sys.creatures[i].q + noise_a * np.random.randn(), 0, 1)
    
    new_C = {(b.i, b.j): np.clip(b.C_ij + reconciliations[(b.i, b.j)], 0, 1) for b in sys.bonds}
    
    for i in range(sys.n):
        sys.creatures[i].theta = (sys.creatures[i].theta + sys.creatures[i].sigma * dt) % (2 * np.pi)
        sys.creatures[i].F, sys.creatures[i].q, sys.creatures[i].a = new_F[i], new_q[i], new_a[i]
        sys.creatures[i].k += 1
        sys.creatures[i].tau += compute_presence(sys.creatures[i]) * dt
    for b in sys.bonds: b.C_ij = new_C[(b.i, b.j)]
    
    return {'mean_P': np.mean([compute_presence(c) for c in sys.creatures]), 'mean_q': np.mean([c.q for c in sys.creatures]),
            'mean_a': np.mean([c.a for c in sys.creatures]), 'mean_C': np.mean([b.C_ij for b in sys.bonds]) if sys.bonds else 0,
            'mean_F': np.mean([c.F for c in sys.creatures]), 'flows': flows, 'injections': injections, 'delta_F': delta_F, 'dissipations': dissipations}

def create_system(n: int, conn: float = 0.3, seed: int = None) -> DETSystem:
    if seed: np.random.seed(seed)
    creatures = [Creature(F=np.random.uniform(0.5,1.5), sigma=np.random.uniform(0.8,1.2), a=np.random.uniform(0.3,0.7),
                         theta=np.random.uniform(0,2*np.pi), q=np.random.uniform(0,0.2), F_op=np.random.uniform(0.05,0.15), H=np.random.uniform(0.05,0.15)) for _ in range(n)]
    bonds = [Bond(i=i, j=j, sigma_ij=np.random.uniform(0.8,1.2), C_ij=np.random.uniform(0.2,0.5)) for i in range(n) for j in range(i+1,n) if np.random.random() < conn]
    if len(bonds) < n-1: bonds += [Bond(i=i, j=i+1, sigma_ij=1.0, C_ij=0.3) for i in range(n-1) if not any((b.i==i and b.j==i+1) for b in bonds)]
    return DETSystem(creatures=creatures, bonds=bonds)

def create_radial(shells: int = 3, per_shell: int = 6) -> DETSystem:
    creatures = [Creature(F=1.0, a=0.5, sigma=1.0, q=0.0, F_op=0.1, H=0.1)]
    for _ in range(shells * per_shell):
        creatures.append(Creature(F=1.0, a=np.random.uniform(0.4,0.7), sigma=1.0, q=np.random.uniform(0,0.2), theta=np.random.uniform(0,2*np.pi)))
    bonds = [Bond(i=0, j=i, sigma_ij=1.0, C_ij=0.5) for i in range(1, per_shell+1)]
    for s in range(shells):
        start = 1 + s * per_shell
        for i in range(start, start + per_shell):
            bonds.append(Bond(i=i, j=start + (i-start+1) % per_shell, sigma_ij=0.8, C_ij=0.6))
            if s < shells-1: bonds.append(Bond(i=i, j=start + per_shell + (i-start) % per_shell, sigma_ij=1.0, C_ij=0.5))
    return DETSystem(creatures=creatures, bonds=bonds)

def copy_system(sys: DETSystem) -> DETSystem:
    return DETSystem([Creature(F=c.F, tau=c.tau, sigma=c.sigma, a=c.a, theta=c.theta, k=c.k, q=c.q, F_op=c.F_op, H=c.H) for c in sys.creatures],
                     [Bond(i=b.i, j=b.j, sigma_ij=b.sigma_ij, C_ij=b.C_ij) for b in sys.bonds], sys.epsilon, sys.F_min, sys.neighborhood_radius, sys.alpha_q)

if __name__ == "__main__":
    sys = create_system(10, seed=42)
    print(f"System: {sys.n} creatures, {len(sys.bonds)} bonds")
    m = step_system(sys)
    print(f"Step 1: P={m['mean_P']:.3f}, C={m['mean_C']:.3f}")
