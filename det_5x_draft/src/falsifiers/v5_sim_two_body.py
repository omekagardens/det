import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration: Two-Body Interaction ---
# Using the "Hyper-Soliton" parameters verified in the Proton Loop test
BETA = -0.15        # Phase Coupling (Suction)
ALPHA_LEARN = 2.0   # Fast Road Building
LAMBDA_DECAY = 0.05 # Moderate Decay
DT = 0.1            
STEPS = 2000        
ZPE = 0.1           # Vacuum Level
MIN_C = 0.001       # Vacuum Permeability Floor

class DET_v5_TwoBody_Sim:
    def __init__(self, run_name="Two-Body Interaction"):
        self.run_name = run_name
        self.num_nodes = 80 # Larger ring to accommodate two particles
        
        # State Variables
        self.F = np.ones(self.num_nodes) * ZPE
        self.theta = np.zeros(self.num_nodes)    
        self.a = np.ones(self.num_nodes)         
        self.k = np.zeros(self.num_nodes)      
        self.sigma = np.ones(self.num_nodes)   
        
        # Bond Matrix (Directed)
        self.C = np.zeros((self.num_nodes, self.num_nodes))
        
        # Initialize Ring Topology
        for i in range(self.num_nodes):
            next_node = (i + 1) % self.num_nodes
            prev_node = (i - 1) % self.num_nodes
            self.C[i, next_node] = MIN_C 
            self.C[i, prev_node] = MIN_C 

        self.history_F = [] 
        self.dist_history = [] # Track distance between peaks

    def get_neighbors(self, i):
        row_nbs = np.where(self.C[i] > 0.0001)[0]
        col_nbs = np.where(self.C[:, i] > 0.0001)[0]
        return np.unique(np.concatenate((row_nbs, col_nbs)))

    def step(self):
        # 1. Phase Update
        d_theta = BETA * self.F
        self.theta += d_theta * DT
        self.theta = np.mod(self.theta, 2 * np.pi)

        # 2. Local Wavefunction
        Psi = np.zeros(self.num_nodes, dtype=complex)
        for i in range(self.num_nodes):
            nbs = self.get_neighbors(i)
            # Normalization with epsilon
            total_local = self.F[i] + np.sum(self.F[nbs])
            norm = 1.0 / (total_local + 1e-9)
            Psi[i] = np.sqrt(self.F[i] * norm) * np.exp(1j * self.theta[i])

        # 3. Flow J
        J_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            targets = [(i + 1) % self.num_nodes] 
            
            for j in targets:
                if self.C[i,j] < 0.0001 and self.C[j,i] < 0.0001: continue
                
                c_avg = 0.5 * (self.C[i,j] + self.C[j,i])
                
                term_Q = np.sqrt(c_avg) * np.imag(np.conj(Psi[i]) * Psi[j])
                term_C = (1.0 - np.sqrt(c_avg)) * (self.F[i] - self.F[j])
                raw_drive = term_Q + term_C
                
                base_cond = self.C[i,j] if raw_drive > 0 else self.C[j,i]
                
                # NON-LINEAR CONDUCTIVITY (Hard Soliton Condition)
                cond = base_cond ** 2
                
                sigma_bond = 0.5 * (self.sigma[i] + self.sigma[j])
                flow = sigma_bond * cond * raw_drive
                
                J_matrix[i,j] = flow
                J_matrix[j,i] = -flow 

        # 4. Resource Update
        self.F -= np.sum(J_matrix, axis=1) * DT
        self.F = np.maximum(self.F, ZPE)
        
        # 5. History
        self.k += np.sum(np.abs(J_matrix), axis=1) * DT
        self.sigma = 1.0 + np.log(1.0 + self.k)

        # 6. Hebbian Update
        for i in range(self.num_nodes):
            for target in [(i + 1) % self.num_nodes, (i - 1) % self.num_nodes]:
                j = target
                growth = ALPHA_LEARN * max(0, J_matrix[i,j])
                decay = LAMBDA_DECAY * self.C[i,j]
                self.C[i,j] += (growth - decay) * DT
                self.C[i,j] = np.clip(self.C[i,j], MIN_C, 1.0)

        self.history_F.append(self.F.copy())
        
        # Track Peak Separation
        peaks = self.find_peaks(self.F)
        if len(peaks) >= 2:
            # Assuming sorted indices, minimal distance on ring
            d1 = abs(peaks[0] - peaks[1])
            d2 = self.num_nodes - d1
            self.dist_history.append(min(d1, d2))
        else:
            self.dist_history.append(0)

    def find_peaks(self, F_arr):
        # Simple threshold-based peak finder for simulation tracking
        # Returns indices of localized masses
        threshold = 5.0
        indices = np.where(F_arr > threshold)[0]
        
        # Group adjacent indices
        if len(indices) == 0: return []
        
        peaks = []
        if len(indices) > 0:
            # Quick clustering
            clusters = []
            current_cluster = [indices[0]]
            for k in range(1, len(indices)):
                if indices[k] == indices[k-1] + 1:
                    current_cluster.append(indices[k])
                else:
                    clusters.append(current_cluster)
                    current_cluster = [indices[k]]
            clusters.append(current_cluster)
            
            # Find center of each cluster
            for c in clusters:
                peaks.append(int(np.mean(c)))
                
        return peaks

    def run(self):
        print(f"\n--- Running DET v5: {self.run_name} ---")
        
        # Initial positions separated by 20 nodes
        pos1 = 20
        pos2 = 40
        
        print(f"Injecting Particle 1 at Node {pos1}...")
        self.inject_particle(pos1)
        
        print(f"Injecting Particle 2 at Node {pos2}...")
        self.inject_particle(pos2)
        
        print("Starting Simulation...")
        for k in range(STEPS):
            self.step()
            
        self.analyze_interaction()
        self.plot_interaction()

    def inject_particle(self, node_idx):
        # Inject Mass
        self.F[node_idx] = 20.0 
        
        # Apply Spin Kick (Rightwards/Clockwise)
        n = self.num_nodes
        self.theta[node_idx] = 0.0
        self.theta[(node_idx+1)%n] = 0.5 * np.pi
        self.theta[(node_idx+2)%n] = 1.0 * np.pi
        
        # Dig Trench
        self.C[node_idx, (node_idx+1)%n] = 0.9
        self.C[(node_idx+1)%n, (node_idx+2)%n] = 0.8
        self.C[node_idx, (node_idx-1)%n] = MIN_C

    def analyze_interaction(self):
        # Check final state
        final_dist = self.dist_history[-1]
        print("\nResults:")
        print(f"  Final Separation Distance: {final_dist:.2f} nodes")
        
        # Interpretation
        if final_dist == 0:
            print("  [RESULT] FUSION (Strong Force): Particles merged.")
        elif final_dist > 15:
            print("  [RESULT] REPULSION (Electrostatics): Particles maintained separation.")
        else:
            print("  [RESULT] INTERACTION COMPLEX: Particles are close but distinct.")

    def plot_interaction(self):
        data = np.array(self.history_F)
        
        plt.figure(figsize=(10, 8))
        
        # Plot 1: Space-Time Heatmap
        plt.subplot(2, 1, 1)
        plt.imshow(data, aspect='auto', cmap='magma', origin='lower',
                   extent=[0, self.num_nodes, 0, STEPS])
        plt.colorbar(label='Resource F')
        plt.title(f"Two-Body Interaction ({self.run_name})")
        plt.xlabel("Ring Position")
        plt.ylabel("Time Step")
        
        # Plot 2: Separation Distance
        plt.subplot(2, 1, 2)
        plt.plot(self.dist_history, color='cyan', linewidth=1.5, label='Separation')
        plt.title("Particle Separation Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Distance (Nodes)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = DET_v5_TwoBody_Sim()
    sim.run()