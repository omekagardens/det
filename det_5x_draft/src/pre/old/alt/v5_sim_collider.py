import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration: Collider (Flight Recorder Mode) ---
# FIX: Added INTERACTION THRESHOLD to prevent Strong Force from binding to Vacuum.
# Restored ALPHA_LEARN for mobility.
BETA = -0.2         
ALPHA_LEARN = 2.0   # RESTORED: High mobility
GAMMA_STRONG = 50.0 # High binding
GAMMA_SYNC = 0.5    # Moderate sync
LAMBDA_DECAY = 0.05 
DT = 0.05           
STEPS = 2000        
ZPE = 0.1           
MIN_C = 0.05        
INTERACTION_THRESHOLD = 5.0 # NEW: Forces only engage if F_i * F_j > Threshold

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
            next2_node = (i + 2) % self.num_nodes
            prev2_node = (i - 2) % self.num_nodes
            
            self.C[i, next_node] = MIN_C 
            self.C[i, prev_node] = MIN_C 
            self.C[i, next2_node] = MIN_C 
            self.C[i, prev2_node] = MIN_C 

        self.history_F = [] 
        self.peak_tracks = [] 

    def get_neighbors(self, i):
        row_nbs = np.where(self.C[i] > 0.0001)[0]
        col_nbs = np.where(self.C[:, i] > 0.0001)[0]
        return np.unique(np.concatenate((row_nbs, col_nbs)))

    def step(self):
        # 1. Phase Update (With Thresholded Gluon Synchronization)
        sync_drive = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            targets = [(i + 1) % self.num_nodes, (i - 1) % self.num_nodes]
            for j in targets:
                # INTERACTION THRESHOLD: Only sync if both are matter
                density_prod = self.F[i] * self.F[j]
                if density_prod > INTERACTION_THRESHOLD:
                    coupling = GAMMA_SYNC * density_prod
                    sync_drive[i] += coupling * np.sin(self.theta[j] - self.theta[i])

        d_theta = (BETA * self.F) + sync_drive
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

        # 3. Flow J (Strict Directed)
        J_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            targets = [(i + 1) % self.num_nodes] 
            for j in targets:
                # J(i->j)
                mix_ji = np.sqrt(self.C[j,i])
                term_Q_fwd = mix_ji * np.imag(np.conj(Psi[i]) * Psi[j])
                term_C_fwd = (1.0 - mix_ji) * (self.F[i] - self.F[j])
                drive_fwd = term_Q_fwd + term_C_fwd
                flow_fwd = (self.C[i,j]**2) * max(0, drive_fwd)
                
                # J(j->i)
                mix_ij = np.sqrt(self.C[i,j])
                term_Q_bwd = mix_ij * np.imag(np.conj(Psi[j]) * Psi[i])
                term_C_bwd = (1.0 - mix_ij) * (self.F[j] - self.F[i])
                drive_bwd = term_Q_bwd + term_C_bwd
                flow_bwd = (self.C[j,i]**2) * max(0, drive_bwd)
                
                sigma_bond = 0.5 * (self.sigma[i] + self.sigma[j])
                net_flow = sigma_bond * (flow_fwd - flow_bwd)
                J_matrix[i,j] = net_flow
                J_matrix[j,i] = -net_flow

        # 4. Resource Update
        self.F -= np.sum(J_matrix, axis=1) * DT
        self.F = np.maximum(self.F, ZPE)
        
        # 5. History & Hebbian (Extended Kernel + Threshold)
        self.k += np.sum(np.abs(J_matrix), axis=1) * DT
        self.sigma = 1.0 + np.log(1.0 + self.k)

        for i in range(self.num_nodes):
            # Distance 1
            targets1 = [(i + 1) % self.num_nodes, (i - 1) % self.num_nodes]
            for j in targets1:
                term_flow = ALPHA_LEARN * max(0, J_matrix[i, j])
                
                # Strong Force Threshold Check
                density_prod = self.F[i] * self.F[j]
                term_strong = 0.0
                if density_prod > INTERACTION_THRESHOLD:
                    term_strong = GAMMA_STRONG * density_prod * 1.0
                
                growth = term_flow + term_strong
                decay = LAMBDA_DECAY * self.C[i, j]
                self.C[i, j] += (growth - decay) * DT
                self.C[i, j] = np.clip(self.C[i, j], MIN_C, 1.0)
            
            # Distance 2
            targets2 = [(i + 2) % self.num_nodes, (i - 2) % self.num_nodes]
            for j in targets2:
                term_flow = ALPHA_LEARN * max(0, J_matrix[i, j]) 
                
                # Strong Force Threshold Check
                density_prod = self.F[i] * self.F[j]
                term_strong = 0.0
                if density_prod > INTERACTION_THRESHOLD:
                    term_strong = GAMMA_STRONG * density_prod * 0.5 
                
                growth = term_flow + term_strong
                decay = LAMBDA_DECAY * self.C[i, j]
                self.C[i, j] += (growth - decay) * DT
                self.C[i, j] = np.clip(self.C[i, j], MIN_C, 1.0)

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

    def get_peaks(self):
        # Return indices of the two highest peaks
        # Simple clustering
        peaks = []
        threshold = np.max(self.F) * 0.2
        indices = np.where(self.F > threshold)[0]
        
        if len(indices) == 0: return []
        
        # Cluster
        clusters = []
        curr = [indices[0]]
        for x in indices[1:]:
            if x - curr[-1] < 5: # Group close nodes
                curr.append(x)
            else:
                clusters.append(curr)
                curr = [x]
        clusters.append(curr)
        
        peak_locs = []
        for c in clusters:
            # Weighted center of mass for the cluster
            mass = np.sum(self.F[c])
            pos = np.sum(self.F[c] * c) / mass
            peak_locs.append(pos)
            
        return sorted(peak_locs)

    def run_sweep(self):
        # Single detailed run for debug, then sweep
        print(f"\n--- Running DET v5 Collider FLIGHT RECORDER (Thresholded Forces) ---")
        print(f"{'Energy':<8} | {'T=0':<12} | {'T=500':<12} | {'T=1000':<12} | {'Outcome'}")
        print("-" * 75)
        
        energies = [20.0, 60.0, 100.0, 150.0]
        
        for E in energies:
            self.__init__(run_name=f"Run E={E}")
            p1 = 20; p2 = 80
            self.inject_particle(p1, direction=1, energy=E)
            self.inject_particle(p2, direction=-1, energy=E)
            
            # Record positions at T=0, T=500, T=1000
            p_t0 = self.get_peaks()
            p_t500 = []
            p_t1000 = []
            
            for k in range(STEPS):
                self.step()
                if k == 500: p_t500 = self.get_peaks()
                if k == 1000: p_t1000 = self.get_peaks()
            
            end_peaks = self.get_peaks()
            
            def fmt_peaks(peaks):
                if not peaks: return "None"
                return ",".join([f"{p:.1f}" for p in peaks])

            outcome = "UNKNOWN"
            if len(end_peaks) < 2:
                outcome = "FUSION/LOSS"
            else:
                dist = abs(end_peaks[-1] - end_peaks[0])
                if dist <= 3.0: outcome = "FUSION (Bound)"
                elif dist > 40.0: 
                    # If they barely moved from start (20, 80), it's STASIS, not BOUNCE
                    start_dist = 60.0
                    if abs(dist - start_dist) < 5.0:
                        outcome = "STASIS (Frozen)"
                    else:
                        outcome = "BOUNCE/SEPARATE"
                else:
                    outcome = "STALL/SLOW"

            print(f"{E:<8.1f} | {fmt_peaks(p_t0):<12} | {fmt_peaks(p_t500):<12} | {fmt_peaks(p_t1000):<12} | {outcome}")

if __name__ == "__main__":
    sim = DET_v5_Collider()
    sim.run_sweep()