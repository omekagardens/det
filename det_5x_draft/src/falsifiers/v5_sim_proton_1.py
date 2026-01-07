import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration: Proton Loop ---
# Tuned for "Hard Soliton" stability via Non-Linear Conductivity
BETA = -0.15        # Slower, cleaner phase rotation
ALPHA_LEARN = 2.0   # Fast road building
LAMBDA_DECAY = 0.05 # Moderate decay to maintain the "tail" structure
DT = 0.1            # Standard timestep
STEPS = 2000        
ZPE = 0.1           # Vacuum Level
MIN_C = 0.001       # Lower Vacuum Floor (Less background leakage)

class DET_v5_Proton_Sim:
    def __init__(self, run_name="Proton Loop Stability (Non-Linear)"):
        self.run_name = run_name
        self.num_nodes = 50 
        
        # State Variables
        self.F = np.ones(self.num_nodes) * ZPE
        self.theta = np.zeros(self.num_nodes)    
        self.a = np.ones(self.num_nodes)         
        self.k = np.zeros(self.num_nodes)      
        self.sigma = np.ones(self.num_nodes)   
        
        # Bond Matrix
        self.C = np.zeros((self.num_nodes, self.num_nodes))
        
        # Initialize Ring
        for i in range(self.num_nodes):
            next_node = (i + 1) % self.num_nodes
            prev_node = (i - 1) % self.num_nodes
            self.C[i, next_node] = MIN_C 
            self.C[i, prev_node] = MIN_C 

        self.history_F = [] 

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
            total_local = self.F[i] + np.sum(self.F[nbs])
            norm = 1.0 / (total_local + 1e-9)
            Psi[i] = np.sqrt(self.F[i] * norm) * np.exp(1j * self.theta[i])

        # 3. Flow J
        J_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            targets = [(i + 1) % self.num_nodes] 
            
            for j in targets:
                if self.C[i,j] < 0.0001 and self.C[j,i] < 0.0001: continue
                
                # Metric term for calculation
                c_avg = 0.5 * (self.C[i,j] + self.C[j,i])
                
                term_Q = np.sqrt(c_avg) * np.imag(np.conj(Psi[i]) * Psi[j])
                term_C = (1.0 - np.sqrt(c_avg)) * (self.F[i] - self.F[j])
                raw_drive = term_Q + term_C
                
                # Diode Logic
                base_cond = self.C[i,j] if raw_drive > 0 else self.C[j,i]
                
                # NON-LINEAR SHARPENING: 
                # Conductivity is squared. This suppresses weak bonds (leakage)
                # and enhances strong bonds (wake).
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

    def run(self):
        print(f"\n--- Running DET v5: {self.run_name} ---")
        
        start_node = 10
        print(f"Injecting 'Proton Mass' at Node {start_node}...")
        self.F[start_node] = 20.0 
        
        print("Applying 'Orbital Kick'...")
        self.theta[start_node] = 0.0
        self.theta[(start_node+1)%self.num_nodes] = 0.5 * np.pi
        self.theta[(start_node+2)%self.num_nodes] = 1.0 * np.pi
        
        self.C[start_node, (start_node+1)%self.num_nodes] = 0.9
        self.C[(start_node+1)%self.num_nodes, (start_node+2)%self.num_nodes] = 0.8
        self.C[start_node, (start_node-1)%self.num_nodes] = MIN_C

        for k in range(STEPS):
            self.step()
            
        self.analyze_stability()
        self.plot_ring()

    def analyze_stability(self):
        final_F = self.history_F[-1]
        max_val = np.max(final_F)
        total_F = np.sum(final_F)
        localization = max_val / (total_F / self.num_nodes)
        
        print("\nResults:")
        print(f"  Final Peak Resource: {max_val:.2f}")
        print(f"  Localization Ratio:  {localization:.2f}")
        
        if localization > 10.0:
            print("  [SUCCESS] Particle is Highly Stable! (Solid Soliton)")
        elif localization > 5.0:
            print("  [SUCCESS] Particle is Stable.")
        else:
            print("  [FAIL] Particle Diffused.")

    def plot_ring(self):
        data = np.array(self.history_F)
        
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.imshow(data, aspect='auto', cmap='magma', origin='lower',
                   extent=[0, self.num_nodes, 0, STEPS])
        plt.colorbar(label='Resource F')
        plt.title(f"Proton Loop ({self.run_name})")
        plt.xlabel("Ring Angle")
        plt.ylabel("Time Step")
        
        com_angle = []
        for frame in self.history_F:
            angles = np.linspace(0, 2*np.pi, self.num_nodes, endpoint=False)
            x = np.sum(frame * np.cos(angles))
            y = np.sum(frame * np.sin(angles))
            angle = np.arctan2(y, x)
            if angle < 0: angle += 2*np.pi
            com_angle.append(angle)
            
        plt.subplot(2, 1, 2)
        plt.plot(np.degrees(com_angle), color='orange', label='Phase Angle')
        plt.title("Orbital Trajectory")
        plt.xlabel("Time Step")
        plt.ylabel("Angle (Degrees)")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 360)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = DET_v5_Proton_Sim()
    sim.run()