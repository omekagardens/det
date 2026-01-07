import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration: Collider (Stable Fusion Regime) ---
# TUNED FOR STABILITY: Couplings are lowered to prevent Phase Aliasing
BETA = -0.2         
ALPHA_LEARN = 1.0   
GAMMA_STRONG = 0.5  # Stable Density Binding
GAMMA_SYNC = 0.05   # Stable Phase Locking (Strong enough to win, weak enough to track)
LAMBDA_DECAY = 0.05 
DT = 0.05           
STEPS = 2000        
ZPE = 0.1           
MIN_C = 0.05        

class DET_v5_Collider:
    def __init__(self, run_name="Collider Sweep"):
        self.run_name = run_name
        self.num_nodes = 100 
        
        # State Variables
        self.F = np.ones(self.num_nodes) * ZPE
        self.theta = np.zeros(self.num_nodes)    
        self.a = np.ones(self.num_nodes)         
        self.k = np.zeros(self.num_nodes)      
        self.sigma = np.ones(self.num_nodes)   
        
        # Bond Matrix
        self.C = np.zeros((self.num_nodes, self.num_nodes))
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
        # 1. Phase Update (Extended Kernel + Stability Clamp)
        sync_drive = np.zeros(self.num_nodes)
        n = self.num_nodes
        
        for i in range(n):
            # EXTENDED KERNEL (2-Hop)
            # Allows particles to 'sense' each other before contact
            targets_1 = [(i + 1) % n, (i - 1) % n]
            targets_2 = [(i + 2) % n, (i - 2) % n]
            
            # Distance 1 (Strong)
            for j in targets_1:
                coupling = GAMMA_SYNC * self.F[i] * self.F[j]
                sync_drive[i] += coupling * np.sin(self.theta[j] - self.theta[i])
            
            # Distance 2 (Weak)
            for j in targets_2:
                coupling = (GAMMA_SYNC * 0.5) * self.F[i] * self.F[j]
                sync_drive[i] += coupling * np.sin(self.theta[j] - self.theta[i])

        d_theta = (BETA * self.F) + sync_drive
        
        # STABILITY CLAMP: Prevent aliasing (max rotation 90 deg per step)
        max_rot = (np.pi / 2.0) / DT
        d_theta = np.clip(d_theta, -max_rot, max_rot)
        
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
                
                c_avg = 0.5 * (self.C[i,j] + self.C[j,i])
                term_Q = np.sqrt(c_avg) * np.imag(np.conj(Psi[i]) * Psi[j])
                term_C = (1.0 - np.sqrt(c_avg)) * (self.F[i] - self.F[j])
                raw_drive = term_Q + term_C
                
                base_cond = self.C[i,j] if raw_drive > 0 else self.C[j,i]
                cond = base_cond ** 2 
                
                sigma_bond = 0.5 * (self.sigma[i] + self.sigma[j])
                flow = sigma_bond * cond * raw_drive
                J_matrix[i,j] = flow
                J_matrix[j,i] = -flow 

        # 4. Resource Update
        self.F -= np.sum(J_matrix, axis=1) * DT
        self.F = np.maximum(self.F, ZPE)
        
        # 5. History & Hebbian
        self.k += np.sum(np.abs(J_matrix), axis=1) * DT
        self.sigma = 1.0 + np.log(1.0 + self.k)

        for i in range(self.num_nodes):
            for target in [(i + 1) % self.num_nodes, (i - 1) % self.num_nodes]:
                j = target
                
                term_flow = ALPHA_LEARN * max(0, J_matrix[i,j])
                term_strong = GAMMA_STRONG * self.F[i] * self.F[j]
                
                growth = term_flow + term_strong
                decay = LAMBDA_DECAY * self.C[i,j]
                
                self.C[i,j] += (growth - decay) * DT
                self.C[i,j] = np.clip(self.C[i,j], MIN_C, 1.0)

        self.history_F.append(self.F.copy())

    def inject_particle(self, node_idx, direction, energy):
        n = self.num_nodes
        self.F[node_idx] = energy
        
        if direction == 1: # Right
            self.theta[node_idx] = 0.0
            self.theta[(node_idx+1)%n] = 0.5 * np.pi
            self.theta[(node_idx+2)%n] = 1.0 * np.pi
            self.C[node_idx, (node_idx+1)%n] = 0.9
            self.C[(node_idx+1)%n, (node_idx+2)%n] = 0.8
            self.C[node_idx, (node_idx-1)%n] = MIN_C
        else: # Left
            self.theta[node_idx] = 0.0
            self.theta[(node_idx-1)%n] = 0.5 * np.pi
            self.theta[(node_idx-2)%n] = 1.0 * np.pi
            self.C[node_idx, (node_idx-1)%n] = 0.9
            self.C[(node_idx-1)%n, (node_idx-2)%n] = 0.8
            self.C[node_idx, (node_idx+1)%n] = MIN_C

    def run_sweep(self):
        # Focus on the transition zone
        energies = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]
        results = []
        
        print(f"\n--- Running DET v5 Fusion Threshold Sweep (Stable Parameters) ---")
        print(f"{'Energy':<10} | {'Outcome':<15} | {'Final Peaks'}")
        print("-" * 45)
        
        for E in energies:
            self.__init__(run_name=f"Run E={E}")
            p1 = 20; p2 = 80
            self.inject_particle(p1, direction=1, energy=E)
            self.inject_particle(p2, direction=-1, energy=E)
            
            for k in range(STEPS):
                self.step()
                
            final_F = self.history_F[-1]
            peaks = []
            for i in range(len(final_F)):
                if final_F[i] > (E * 0.25): 
                    if len(peaks) == 0 or abs(i - peaks[-1]) > 5:
                        peaks.append(i)
            
            outcome = "BOUNCE" if len(peaks) >= 2 else "FUSION"
            if len(peaks) == 0: outcome = "DISINTEGRATE"
            
            results.append((E, outcome))
            print(f"{E:<10.1f} | {outcome:<15} | {len(peaks)}")

        # Find Threshold
        fusion_found = False
        for i in range(len(results)-1):
            if results[i][1] == "BOUNCE" and results[i+1][1] == "FUSION":
                print(f"\n>> FUSION THRESHOLD DETECTED between E={results[i][0]} and E={results[i+1][0]}")
                print(f">> Fine Structure Proxy ~ {results[i+1][0]}")
                fusion_found = True
                break
        
        if not fusion_found:
            print("\n>> No Fusion Transition found in this energy range.")

if __name__ == "__main__":
    sim = DET_v5_Collider()
    sim.run_sweep()