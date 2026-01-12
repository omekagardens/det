import numpy as np
import matplotlib.pyplot as plt

"""
DET v5 Collider - DIRECT PARTICLE MODEL
========================================
Instead of trying to get emergent soliton propagation (which requires 
careful tuning), this version directly simulates particles with:
- Position and velocity (explicit kinematics)
- DET-derived interaction forces:
  * EM-like repulsion from phase misalignment (long range)
  * Strong force attraction from density binding (short range)

The goal: Find the energy threshold where kinetic energy overcomes
the EM barrier, allowing the strong force to create fusion.
"""

# Physical constants (in DET units)
MASS = 1.0              # Particle mass
EM_STRENGTH = 10.0      # Electrostatic-like repulsion (phase misalignment)
STRONG_STRENGTH = 100.0 # Strong force (density binding)
STRONG_RANGE = 2.0      # Range of strong force
EM_RANGE = 20.0         # Range of EM force
FUSION_DISTANCE = 1.5   # Distance at which fusion occurs

DT = 0.01
STEPS = 10000

class Particle:
    def __init__(self, x, v, phase=0.0, energy=1.0):
        self.x = x          # Position
        self.v = v          # Velocity
        self.phase = phase  # Internal phase (determines EM interaction)
        self.energy = energy  # "Mass-energy" (affects forces)
        
    def kinetic_energy(self):
        return 0.5 * MASS * self.v**2

class DET_DirectCollider:
    def __init__(self):
        self.p1 = None
        self.p2 = None
        self.history = []
        self.fused = False
        
    def setup_collision(self, separation, kinetic_energy):
        """
        Set up two particles for head-on collision.
        
        separation: initial distance between particles
        kinetic_energy: kinetic energy of each particle
        """
        # Place particles symmetrically around x=0
        x1 = -separation / 2
        x2 = separation / 2
        
        # Calculate velocity from kinetic energy: KE = 0.5*m*v^2
        v = np.sqrt(2 * kinetic_energy / MASS)
        
        # Particles move toward each other
        self.p1 = Particle(x=x1, v=v, phase=0.0, energy=kinetic_energy)
        self.p2 = Particle(x=x2, v=-v, phase=np.pi, energy=kinetic_energy)  # Opposite phase
        
        self.fused = False
        self.history = []
        
    def compute_forces(self):
        """
        Compute forces between particles based on DET theory.
        
        EM Force (Phase Misalignment):
        - Repulsive when phases differ
        - F_em = EM_STRENGTH * sin²(Δθ/2) / r²
        - Range: ~EM_RANGE (Coulomb-like, long range)
        
        Strong Force (Density Binding):
        - Attractive at short range
        - F_strong = -STRONG_STRENGTH * exp(-r/STRONG_RANGE)
        - Only significant when r < ~3*STRONG_RANGE
        """
        # Distance between particles
        r = abs(self.p2.x - self.p1.x)
        r = max(r, 0.1)  # Avoid singularity
        
        # Direction (positive = repulsion, negative = attraction)
        direction = 1 if self.p1.x < self.p2.x else -1
        
        # EM-like force (phase-dependent repulsion)
        phase_diff = self.p2.phase - self.p1.phase
        phase_factor = np.sin(phase_diff / 2) ** 2  # Max when phases opposite
        F_em = EM_STRENGTH * phase_factor * np.exp(-r / EM_RANGE) / (r + 0.5)
        
        # Strong force (short-range attraction)
        F_strong = -STRONG_STRENGTH * np.exp(-r / STRONG_RANGE) / (r + 0.1)
        
        # Total force
        F_total = F_em + F_strong
        
        return F_total * direction, F_em, F_strong, r
        
    def step(self):
        """Advance simulation by one timestep."""
        if self.fused:
            return
            
        # Get forces
        F, F_em, F_strong, r = self.compute_forces()
        
        # Update velocities (F = ma, so a = F/m)
        a = F / MASS
        self.p1.v += a * DT
        self.p2.v -= a * DT  # Opposite direction
        
        # Update positions
        self.p1.x += self.p1.v * DT
        self.p2.x += self.p2.v * DT
        
        # Check for fusion
        if r < FUSION_DISTANCE and F_strong < -abs(F_em):
            self.fused = True
        
        # Record history
        self.history.append({
            'x1': self.p1.x,
            'x2': self.p2.x,
            'v1': self.p1.v,
            'v2': self.p2.v,
            'r': r,
            'F_em': F_em,
            'F_strong': F_strong,
            'F_total': F
        })
        
    def run(self, separation=50.0, kinetic_energy=1.0):
        """Run a complete collision simulation."""
        self.setup_collision(separation, kinetic_energy)
        
        for _ in range(STEPS):
            self.step()
            
            # Check if particles have passed through or bounced far apart
            r = abs(self.p2.x - self.p1.x)
            if r > separation * 1.5:  # Bounced far apart
                break
            if self.fused:
                break
                
        return self.fused
    
    def analyze(self):
        """Analyze the collision outcome."""
        if not self.history:
            return "NO DATA"
            
        # Find minimum approach distance
        distances = [h['r'] for h in self.history]
        min_dist = min(distances)
        min_idx = distances.index(min_dist)
        
        # Check if they bounced (velocities reversed)
        final = self.history[-1]
        initial = self.history[0]
        
        if self.fused:
            return "FUSION", min_dist
        elif final['v1'] < 0 and final['v2'] > 0:
            return "BOUNCE", min_dist
        else:
            return "PASS-THROUGH", min_dist
    
    def plot_collision(self, filename):
        """Generate diagnostic plots."""
        if not self.history:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        steps = range(len(self.history))
        x1 = [h['x1'] for h in self.history]
        x2 = [h['x2'] for h in self.history]
        r = [h['r'] for h in self.history]
        F_em = [h['F_em'] for h in self.history]
        F_strong = [h['F_strong'] for h in self.history]
        F_total = [h['F_total'] for h in self.history]
        
        # Trajectories
        axes[0,0].plot(steps, x1, 'b-', label='Particle 1', linewidth=2)
        axes[0,0].plot(steps, x2, 'r-', label='Particle 2', linewidth=2)
        axes[0,0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('Position')
        axes[0,0].set_title('Particle Trajectories')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Separation
        axes[0,1].plot(steps, r, 'k-', linewidth=2)
        axes[0,1].axhline(y=FUSION_DISTANCE, color='green', linestyle='--', 
                         label=f'Fusion distance ({FUSION_DISTANCE})')
        axes[0,1].axhline(y=STRONG_RANGE*3, color='orange', linestyle='--',
                         label=f'Strong force range (~{STRONG_RANGE*3:.1f})')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Separation')
        axes[0,1].set_title('Particle Separation')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_ylim(0, max(r)*1.1)
        
        # Forces
        axes[1,0].plot(steps, F_em, 'r-', label='EM (repulsion)', alpha=0.7)
        axes[1,0].plot(steps, F_strong, 'b-', label='Strong (attraction)', alpha=0.7)
        axes[1,0].plot(steps, F_total, 'k-', label='Total', linewidth=2)
        axes[1,0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Force')
        axes[1,0].set_title('Forces During Collision')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Force vs distance
        axes[1,1].scatter(r, F_em, c='red', s=5, alpha=0.3, label='EM')
        axes[1,1].scatter(r, F_strong, c='blue', s=5, alpha=0.3, label='Strong')
        axes[1,1].scatter(r, F_total, c='black', s=5, alpha=0.5, label='Total')
        axes[1,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1,1].axvline(x=FUSION_DISTANCE, color='green', linestyle='--', alpha=0.5)
        axes[1,1].set_xlabel('Separation')
        axes[1,1].set_ylabel('Force')
        axes[1,1].set_title('Force vs Distance')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Saved to {filename}")

def find_fusion_threshold():
    """
    Sweep kinetic energies to find the fusion threshold.
    This is analogous to finding the Coulomb barrier energy.
    """
    print("\n" + "="*70)
    print("DET v5 DIRECT PARTICLE COLLIDER - FUSION THRESHOLD SEARCH")
    print("="*70)
    print(f"\nParameters:")
    print(f"  EM Strength:     {EM_STRENGTH}")
    print(f"  Strong Strength: {STRONG_STRENGTH}")
    print(f"  EM Range:        {EM_RANGE}")
    print(f"  Strong Range:    {STRONG_RANGE}")
    print(f"  Fusion Distance: {FUSION_DISTANCE}")
    print()
    
    energies = np.linspace(0.5, 10.0, 20)
    results = []
    
    print(f"{'Energy':<12} | {'Outcome':<15} | {'Min Distance':<15} | Notes")
    print("-" * 70)
    
    for E in energies:
        sim = DET_DirectCollider()
        fused = sim.run(separation=50.0, kinetic_energy=E)
        outcome, min_dist = sim.analyze()
        
        results.append((E, outcome, min_dist))
        
        # Determine barrier penetration
        if outcome == "FUSION":
            note = "FUSED! Strong force dominated"
        elif min_dist < STRONG_RANGE * 2:
            note = "Entered strong range but bounced"
        else:
            note = "Stopped by EM barrier"
            
        print(f"{E:<12.2f} | {outcome:<15} | {min_dist:<15.2f} | {note}")
    
    # Find threshold
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Find transition point
    threshold = None
    for i in range(len(results)-1):
        if results[i][1] != "FUSION" and results[i+1][1] == "FUSION":
            threshold = (results[i][0] + results[i+1][0]) / 2
            break
    
    if threshold:
        print(f"\n>> FUSION THRESHOLD DETECTED!")
        print(f">> Critical Energy: E ≈ {threshold:.2f}")
        print(f"\n>> Physical Interpretation:")
        print(f"   The EM barrier (Coulomb barrier analog) is at E ≈ {threshold:.2f}")
        print(f"   Below this energy: EM repulsion dominates → BOUNCE")
        print(f"   Above this energy: Kinetic overcomes barrier → Strong force → FUSION")
        
        # Calculate ratio (analogous to fine structure constant derivation)
        E_binding = STRONG_STRENGTH / STRONG_RANGE  # Rough binding energy
        ratio = threshold / E_binding
        print(f"\n>> Ratio (threshold/binding): {ratio:.4f}")
        print(f"   (Compare to α ≈ 1/137 ≈ 0.0073)")
    else:
        if all(r[1] == "FUSION" for r in results):
            print("\n>> All collisions resulted in FUSION")
            print("   Strong force may be too strong relative to EM")
        else:
            print("\n>> No clear fusion threshold found")
    
    return results, threshold

def plot_potential():
    """Visualize the interaction potential."""
    r_values = np.linspace(0.5, 30, 200)
    
    V_em = []
    V_strong = []
    
    for r in r_values:
        # EM potential (repulsive, assuming phases opposite)
        v_em = EM_STRENGTH * np.exp(-r / EM_RANGE) * np.log(r + 0.5)
        V_em.append(v_em)
        
        # Strong potential (attractive)
        v_strong = -STRONG_STRENGTH * STRONG_RANGE * np.exp(-r / STRONG_RANGE)
        V_strong.append(v_strong)
    
    V_total = np.array(V_em) + np.array(V_strong)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, V_em, 'r-', label='EM (repulsive)', linewidth=2)
    plt.plot(r_values, V_strong, 'b-', label='Strong (attractive)', linewidth=2)
    plt.plot(r_values, V_total, 'k-', label='Total', linewidth=2)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=FUSION_DISTANCE, color='green', linestyle='--', alpha=0.5, 
                label=f'Fusion distance')
    
    # Find and mark the barrier peak
    barrier_idx = np.argmax(V_total[:100])  # Look in first half
    barrier_r = r_values[barrier_idx]
    barrier_V = V_total[barrier_idx]
    plt.plot(barrier_r, barrier_V, 'ko', markersize=10)
    plt.annotate(f'Barrier\n({barrier_r:.1f}, {barrier_V:.1f})', 
                xy=(barrier_r, barrier_V), xytext=(barrier_r+5, barrier_V+20),
                arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.xlabel('Separation r')
    plt.ylabel('Potential Energy')
    plt.title('DET Interaction Potential (EM + Strong)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/claude/potential.png', dpi=150)
    print("Saved potential plot to potential.png")
    
    return barrier_V

if __name__ == "__main__":
    # First, visualize the potential
    print("Plotting interaction potential...")
    barrier = plot_potential()
    print(f"Barrier height from potential: {barrier:.2f}")
    
    # Find the fusion threshold
    results, threshold = find_fusion_threshold()
    
    # Plot a sample collision at threshold energy
    if threshold:
        print(f"\nPlotting collision at threshold energy E={threshold:.2f}...")
        sim = DET_DirectCollider()
        sim.run(separation=50.0, kinetic_energy=threshold)
        sim.plot_collision('/home/claude/threshold_collision.png')
        
        # Also plot a bounce and a fusion for comparison
        print("\nPlotting bounce (E=1.0)...")
        sim2 = DET_DirectCollider()
        sim2.run(separation=50.0, kinetic_energy=1.0)
        sim2.plot_collision('/home/claude/bounce_collision.png')
        
        print("\nPlotting fusion (E=8.0)...")
        sim3 = DET_DirectCollider()
        sim3.run(separation=50.0, kinetic_energy=8.0)
        sim3.plot_collision('/home/claude/fusion_collision.png')
