import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration: Speed Limit Test ---
BETA = -0.2         
ALPHA_LEARN = 1.0   
LAMBDA_DECAY = 0.01 
DT = 0.02           # REDUCED from 0.1 to 0.02 to prevent Phase Aliasing at High E
STEPS = 3000        # INCREASED to compensate for smaller DT

class DET_v5_Sim:
    def __init__(self, energy_level):
        self.energy_level = energy_level
        self.num_nodes = 100 
        
        # State Variables
        self.F = np.zeros(self.num_nodes) 
        self.theta = np.zeros(self.num_nodes)    
        self.a = np.ones(self.num_nodes)         
        self.k = np.zeros(self.num_nodes)      
        self.sigma = np.ones(self.num_nodes)   
        
        # Bond Matrix
        self.C = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes - 1):
            self.C[i, i+1] = 0.05 
            self.C[i+1, i] = 0.05 

        self.com_history = []

    def get_neighbors(self, i):
        row_nbs = np.where(self.C[i] > 0.01)[0]
        col_nbs = np.where(self.C[:, i] > 0.01)[0]
        return np.unique(np.concatenate((row_nbs, col_nbs)))

    def step(self):
        # 1. Phase Update
        d_theta = BETA * self.F
        self.theta += d_theta * DT
        self.theta = np.mod(self.theta, 2 * np.pi)

        # 2. Psi
        Psi = np.zeros(self.num_nodes, dtype=complex)
        for i in range(self.num_nodes):
            nbs = self.get_neighbors(i)
            total = self.F[i] + np.sum(self.F[nbs])
            norm = 1.0 / (total + 1e-9)
            Psi[i] = np.sqrt(self.F[i] * norm) * np.exp(1j * self.theta[i])

        # 3. Flow
        J_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes): 
                if self.C[i,j] < 0.001 and self.C[j,i] < 0.001: continue
                
                c_avg = 0.5 * (self.C[i,j] + self.C[j,i])
                term_Q = np.sqrt(c_avg) * np.imag(np.conj(Psi[i]) * Psi[j])
                term_C = (1.0 - np.sqrt(c_avg)) * (self.F[i] - self.F[j])
                raw_drive = term_Q + term_C
                
                # Diode
                cond = self.C[i,j] if raw_drive > 0 else self.C[j,i]
                sigma_bond = 0.5 * (self.sigma[i] + self.sigma[j])
                flow = sigma_bond * cond * raw_drive
                J_matrix[i,j] = flow
                J_matrix[j,i] = -flow 

        # 4. Resource Update
        self.F -= np.sum(J_matrix, axis=1) * DT
        self.F = np.maximum(self.F, 0)
        
        # 5. History
        self.k += np.sum(np.abs(J_matrix), axis=1) * DT
        self.sigma = 1.0 + np.log(1.0 + self.k)

        # 6. Hebbian
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j: continue
                growth = ALPHA_LEARN * max(0, J_matrix[i,j])
                decay = LAMBDA_DECAY * self.C[i,j]
                self.C[i,j] += (growth - decay) * DT
                self.C[i,j] = np.clip(self.C[i,j], 0.0, 1.0)

        # Track COM
        total_F = np.sum(self.F)
        if total_F > 0.1:
            com = np.sum(np.arange(self.num_nodes) * self.F) / total_F
            self.com_history.append(com)
        else:
            self.com_history.append(np.nan)

    def run(self):
        start_node = 5
        self.F[start_node] = self.energy_level
        
        # Kick
        self.theta[start_node] = 0.0
        self.theta[start_node+1] = 0.5 * np.pi
        self.theta[start_node+2] = 1.0 * np.pi
        self.C[start_node, start_node+1] = 0.9
        self.C[start_node+1, start_node+2] = 0.7
        self.C[start_node+1, start_node] = 0.01 
        self.C[start_node, start_node-1] = 0.01

        steps_run = 0
        for k in range(STEPS):
            self.step()
            steps_run += 1
            if self.com_history[-1] > self.num_nodes - 5:
                break
                
        valid_com = [x for x in self.com_history if not np.isnan(x)]
        
        # RELAXED Constraints for short fast runs
        if len(valid_com) > 20:
            start_idx = int(len(valid_com) * 0.2)
            end_idx = int(len(valid_com) * 0.9)
            dist = valid_com[end_idx] - valid_com[start_idx]
            time = end_idx - start_idx
            velocity = dist / (time * DT)
            return velocity, dist
        return 0.0, 0.0

def run_speed_sweep():
    # Sweep Energy
    energy_levels = [1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0]
    speeds = []
    
    print("--- Running Speed Limit Test (Energy Sweep) ---")
    print(f"{'Energy':<10} | {'Speed (nodes/t)':<18} | {'Dist Traveled':<15}")
    print("-" * 50)
    
    for E in energy_levels:
        sim = DET_v5_Sim(energy_level=E)
        v, d = sim.run()
        speeds.append(v)
        print(f"{E:<10.1f} | {v:<18.4f} | {d:<15.2f}")
        
    # Check for Saturation
    slopes = np.diff(speeds) / np.diff(energy_levels)
    
    # Avoid index error if speeds are all 0
    if len(slopes) > 0:
        final_slope = slopes[-1]
        initial_slope = slopes[0] if slopes[0] > 1e-6 else 1.0
        
        print("\n--- Analysis ---")
        if final_slope < 0.1 * initial_slope and max(speeds) > 0.1:
            print(">> SATURATION DETECTED: Speed has plateaued.")
            print(f">> Emergent 'c' (Speed of Light) approx: {max(speeds):.4f}")
        elif max(speeds) < 0.01:
            print(">> FAILURE: Particles stalled. Check Phase parameters.")
        else:
            print(">> NO SATURATION: Speed is still increasing (or noisy).")
            
        plt.figure(figsize=(8, 6))
        plt.plot(energy_levels, speeds, 'o-', color='crimson', linewidth=2)
        plt.title("DET v5: Energy vs Velocity")
        plt.xlabel("Energy (Resource F)")
        plt.ylabel("Velocity (v)")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(">> ERROR: No valid data points.")

if __name__ == "__main__":
    run_speed_sweep()