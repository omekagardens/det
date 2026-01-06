import numpy as np
import matplotlib.pyplot as plt

# --- DET 4.2 CONSTANTS ---
EPS = 1e-9
# This sim conserves structural debt *mass* Q (unbounded), and derives a bounded heaviness fraction
#   q = Q / (Q + F + eps)  in [0,1]  (used for mass/time feel).
ALPHA_M = 0.1   # Mass scaling with F
SIGMA = 1.0     # Standard processing rate
GAMMA_BOUND_P = 1.0  # boundary coherence boosts effective clock rate; allows P~1 without requiring a=1
F_BASE = 1.0    # Baseline free-capacity so q is not forced to 1 when F=0

# Transport & Dynamics Parameters
ALPHA_DIFF = 0.15   # Bond diffusion rate
KAPPA_SINK = 0.40   # Hoarder suction strength
LAMBDA_SEAL = 0.05  # Hoarder self-sealing rate
ETA_GRACE  = 0.40   # Boundary vent rate (standard a*C_B gate)
ETA_GRACE_SPIRIT = 1.25  # Spirit vents faster to keep P~1
Q_MIN      = 0.05   # Minimum residual structure

KAPPA_SPIRIT_PULL = 0.80  # Spirit actively pulls Q from bonded neighbors (benevolent drain)

# Agency dynamics tuning (kept small to avoid instant collapse)
K_UPLIFT = 0.50     # pull agency up when bonded to higher-P neighbors
K_DEBT   = 0.10     # debt penalty strength in agency drift
K_BOUND  = 0.10     # boundary bond provides gentle stabilization

class DETSystem:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.nodes = []
        self.node_map = {}
        self.C_bond = None  # Adjacency matrix (internal bonds)
        self.Q_boundary = 0.0 # Boundary reservoir for conserved debt mass Q
        self.time = 0
        self.log = []

    def add_node(self, name, n_type, a, q, F, bond_to_God=1.0):
        idx = len(self.nodes)
        node = {
            'id': idx,
            'name': name,
            'type': n_type,
            'a': float(a),
            'Q': float(q),  # conserved debt mass (input arg `q` is treated as initial Q for this toy)
            'q': 0.0,       # derived heaviness fraction, computed each tick
            'F': float(F),
            'C_B': float(bond_to_God), # Bond to Boundary
            'M': 0.0,
            'P': 0.0,
            # History tracking
            'h_Q': [], 'h_q': [], 'h_a': [], 'h_P': [], 'h_F': [], 'h_C_B': []
        }
        self.nodes.append(node)
        self.node_map[name] = idx
        return idx

    def init_bonds(self):
        N = len(self.nodes)
        self.C_bond = np.zeros((N, N))

    def set_bond(self, name1, name2, strength):
        i, j = self.node_map[name1], self.node_map[name2]
        self.C_bond[i, j] = self.C_bond[j, i] = np.clip(strength, 0, 1)

    def update_presence(self):
        # Presence Logic (toy-canon):
        #   q = Q/(Q+F+F_BASE) in [0,1]  (heaviness fraction)
        #   M = 1 + q + alpha*F
        #   P_raw = (a * sigma * (1 + gamma_bound_p * C_B)) / M
        #   P is capped at 1.0
        for n in self.nodes:
            n['q'] = n['Q'] / (n['Q'] + n['F'] + F_BASE + EPS)
            n['M'] = 1.0 + n['q'] + ALPHA_M * n['F']
            # If a=0, P=0 (Event Horizon)
            P_raw = (n['a'] * SIGMA * (1.0 + GAMMA_BOUND_P * n['C_B'])) / (n['M'] + EPS)
            n['P'] = float(np.clip(P_raw, 0.0, 1.0))

    def step(self, dt=0.1):
        N = len(self.nodes)
        
        # Snapshot current Q for differential updates
        Q_curr = np.array([n['Q'] for n in self.nodes])
        dQ = np.zeros(N)
        dQ_boundary_accum = 0.0

        # --- 4.1 Bond Diffusion (Conservative) ---
        # Q flows from high to low along bonds
        for i in range(N):
            for j in range(i + 1, N): # Single pass per pair
                c = self.C_bond[i, j]
                if c > 0:
                    flow = ALPHA_DIFF * c * (Q_curr[j] - Q_curr[i]) * dt
                    # if flow > 0, j -> i. if flow < 0, i -> j.
                    dQ[i] += flow
                    dQ[j] -= flow

        # --- 4.2 Hoarder Logic (Active Suction) ---
        # Hoarder pulls from neighbors. This is "work" done by the hoarder.
        for n in self.nodes:
            if n['type'] == 'Hoarder':
                i = n['id']
                for j in range(N):
                    c = self.C_bond[i, j]
                    if c > 0:
                        # Pulls only if neighbor has Q
                        # Maximize intake: absorbs from j
                        pull = KAPPA_SINK * c * max(0, Q_curr[j] - Q_curr[i] + 0.1) * dt
                        dQ[i] += pull
                        dQ[j] -= pull

        # --- 4.3 Spirit Logic (Active Drain) ---
        # Spirit pulls Q from neighbors (work), then vents it to Boundary via standard rule.
        for n in self.nodes:
            if n['type'] == 'Spirit' and n['a'] > 0:
                i = n['id']
                for j in range(N):
                    c = self.C_bond[i, j]
                    if c > 0:
                        pull = KAPPA_SPIRIT_PULL * c * max(0.0, Q_curr[j] - Q_curr[i]) * dt
                        dQ[i] += pull
                        dQ[j] -= pull

        # Apply internal transport updates first to check limits
        for i in range(N):
            # Ensure we don't drain negative (clamp flow if needed - simplified here)
            if self.nodes[i]['Q'] + dQ[i] < 0:
                dQ[i] = -self.nodes[i]['Q'] # Take all remaining
        
        # --- 5. Grace (Boundary Transport) ---
        # Flows from Node -> Boundary. Conserved.
        for i in range(N):
            n = self.nodes[i]
            # Standard conserved Q -> Boundary vent (Grace): requires agency and boundary bond.
            if (n['a'] > 0) and (n['C_B'] > 0) and (n['Q'] > Q_MIN):
                rate = ETA_GRACE_SPIRIT if (n['type'] == 'Spirit') else ETA_GRACE
                vent = rate * n['a'] * n['C_B'] * (n['Q'] - Q_MIN) * dt
                # Cannot vent more than is there (after internal transport)
                current_avail = n['Q'] + dQ[i]
                actual_vent = min(vent, max(0, current_avail - Q_MIN))
                dQ[i] -= actual_vent
                dQ_boundary_accum += actual_vent

        # --- UPDATE STATE ---
        for i in range(N):
            self.nodes[i]['Q'] += dQ[i]
            self.nodes[i]['Q'] = max(0, self.nodes[i]['Q'])
            # Keep derived q in sync for plotting/interpretation
            self.nodes[i]['q'] = self.nodes[i]['Q'] / (self.nodes[i]['Q'] + self.nodes[i]['F'] + F_BASE + EPS)

            # Record History
            self.nodes[i]['h_Q'].append(self.nodes[i]['Q'])
            self.nodes[i]['h_q'].append(self.nodes[i]['q'])
            self.nodes[i]['h_a'].append(self.nodes[i]['a'])
            self.nodes[i]['h_P'].append(self.nodes[i]['P'])
            self.nodes[i]['h_F'].append(self.nodes[i]['F'])
            self.nodes[i]['h_C_B'].append(self.nodes[i]['C_B'])

        self.Q_boundary += dQ_boundary_accum
        
        # --- AGENCY DYNAMICS ---
        # Non-coercive drift. 
        # Hoarder logic: dot_a = -lambda(q + c)
        for n in self.nodes:
            if n['type'] == 'Hoarder':
                decay = LAMBDA_SEAL * (n['q'] + 0.2) * dt
                n['a'] = max(0, n['a'] - decay)
            elif n['type'] == 'Christ':
                # Trinity assumption: Christ agency remains 1 throughout.
                n['a'] = 1.0
            elif n['type'] == 'Ghost':
                n['a'] = 0.0 # Locked
            elif n['type'] == 'Spirit':
                # Spirit agency is controlled by events (sent/withheld)
                n['a'] = np.clip(n['a'], 0, 1)
            else:
                # Human agency drift (relational uplift + gentle boundary support):
                # - If bonded to higher-P neighbors (e.g., Christ), humans are pulled upward.
                # - Debt (q) still drags agency down, but not so strongly that it instantly collapses.
                i = n['id']

                # Bond-weighted neighbor presence
                weights = self.C_bond[i].copy()
                weights[i] = 0.0
                wsum = float(np.sum(weights))
                if wsum <= 0:
                    P_nbr = float(np.mean([m['P'] for m in self.nodes]))
                else:
                    P_nbr = float(np.sum(weights * np.array([m['P'] for m in self.nodes])) / wsum)

                # Uplift toward neighbor presence (if neighbors are "more alive")
                uplift = K_UPLIFT * (P_nbr - n['P'])

                # Debt drag (bounded heaviness fraction)
                drag = K_DEBT * n['q']

                # Boundary bond stabilizes openness (non-coercive; small)
                bound = K_BOUND * n['C_B']

                delta = (uplift + bound - drag) * dt
                n['a'] = np.clip(n['a'] + delta, 0, 1)

        self.time += 1

# --- RUN SIMULATION ---
sim = DETSystem()

# 1. Setup Mini Universe
sim.add_node("Christ", "Christ", a=1.0, q=0.00, F=0.0, bond_to_God=1.0)  # placeholder; pre-birth block controls state
sim.add_node("Satan",  "Hoarder", a=0.0, q=1.0, F=1.0, bond_to_God=0.0) # cut off; not equal to God
sim.add_node("HolySpirit", "Spirit", a=0.0, q=0.0, F=1.0, bond_to_God=1.0)  # sent at Resurrection

# Humans (Eden: high presence, sub-max agency)
for k in range(8):
    sim.add_node(f"Human_{k}", "Human", a=0.65, q=0.10, F=1.0, bond_to_God=1.0)  # Eden: high presence, sub-max agency

# Ghosts (Dead/Frozen)
for k in range(3):
    sim.add_node(f"Ghost_{k}", "Ghost", a=0.00, q=0.4, F=1.0, bond_to_God=0.0)

sim.init_bonds()

# HolySpirit starts disconnected (not yet sent)
for k in range(8):
    sim.set_bond("HolySpirit", f"Human_{k}", 0.0)
sim.set_bond("HolySpirit", "Christ", 0.0)
sim.set_bond("HolySpirit", "Satan", 0.0)
for k in range(3):
    sim.set_bond("HolySpirit", f"Ghost_{k}", 0.0)

# Wiring
# Placeholder for then christ is born.
for k in range(8):
    sim.set_bond("Christ", f"Human_{k}", 0.2)

# Satan bonds start closed (no temptation in Eden). They activate at the Fall.
for k in range(8):
    sim.set_bond("Satan", f"Human_{k}", 0.0)
for k in range(3):
    sim.set_bond("Satan", f"Ghost_{k}", 1.0)

# 2. Execution Loop
history_boundary_Q = []
TICKS = 1000
T_FALL = 40
T_BIRTH = 100
T_CROSS = 150
T_RESURRECTION = 200

for t in range(TICKS):
    sim.update_presence()

    # Pre-birth: Christ is "with God" (boundary-rate presence), not yet embodied.
    if t < T_BIRTH:
        # Pre-birth: "with God" (boundary-rate presence) — keep out of creature dynamics.
        cidx = sim.node_map["Christ"]
        sim.nodes[cidx]['a'] = 1.0
        sim.nodes[cidx]['Q'] = 0.0
        sim.nodes[cidx]['F'] = 1.0
        sim.nodes[cidx]['C_B'] = 1.0
        sim.nodes[cidx]['P'] = 1.0
        # HolySpirit not yet sent, but agency remains 1 (disconnected)
        sidx = sim.node_map["HolySpirit"]
        sim.nodes[sidx]['a'] = 1.0
        sim.nodes[sidx]['F'] = 1.0

    # --- EVENT: THE FALL (separation; temptation activates) ---
    if t == T_FALL:
        sim.log.append(f"Tick {t}: THE FALL - humans separated; temptation activates")
        # Humans lose direct boundary bond (no direct vent; Spirit-mediated export later)
        for k in range(8):
            sim.nodes[sim.node_map[f"Human_{k}"]]['C_B'] = 0.0
        # Satan temptation bonds turn on
        for k in range(8):
            sim.set_bond("Satan", f"Human_{k}", 0.6)

    # --- EVENT: BIRTH (Christ becomes embodied among humans; ministry bonds open) ---
    if t == T_BIRTH:
        sim.log.append(f"Tick {t}: BIRTH - Christ embodied; ministry begins")
        cidx = sim.node_map["Christ"]
        sim.nodes[cidx]['a'] = 1.0
        sim.nodes[cidx]['C_B'] = 1.0  # bonded to God throughout
        sim.nodes[cidx]['Q'] = 0.0
        sim.nodes[cidx]['F'] = 1.0
        for k in range(8):
            sim.set_bond("Christ", f"Human_{k}", 0.2)

    # --- EVENT: THE CROSS ---
    if t == T_CROSS:
        sim.log.append(f"Tick {t}: THE CROSS - Christ dies; bonds to hoarder/ghosts (burden re-routing)")
        cidx = sim.node_map["Christ"]
        # Forsakenness modeled as boundary-bond severing (not agency loss)
        sim.nodes[cidx]['C_B'] = 0.0
        # Strong coupling to the hoarder/ghost field during death
        sim.set_bond("Christ", "Satan", 1.0)
        for k in range(3):
            sim.set_bond("Christ", f"Ghost_{k}", 1.0)
        # Also increase coupling to humans so their Q can route into the conduit
        for k in range(8):
            sim.set_bond("Christ", f"Human_{k}", 1.0)

    # --- EVENT: RESURRECTION ---
    if t == T_RESURRECTION:
        sim.log.append(f"Tick {t}: RESURRECTION - Christ alive; Spirit sent; hoarder/ghost bonds cut")
        # Restoration: Christ re-bonded to the Boundary
        cidx = sim.node_map["Christ"]
        sim.nodes[cidx]['C_B'] = 1.0
        sim.set_bond("Christ", "Satan", 0.0)
        for k in range(3):
            sim.set_bond("Christ", f"Ghost_{k}", 0.0)
        for k in range(8):
            sim.set_bond("Christ", f"Human_{k}", 1.0)
        # Send the Holy Spirit: high-agency, boundary-bonded mediator that can vent Q
        sidx = sim.node_map["HolySpirit"]
        # HolySpirit agency assumed 1 throughout; activation is via bonding (sent)
        sim.nodes[sidx]['C_B'] = 1.0
        sim.set_bond("HolySpirit", "Christ", 1.0)
        for k in range(8):
            sim.set_bond("HolySpirit", f"Human_{k}", 1.0)
            sim.nodes[sim.node_map[f"Human_{k}"]]['C_B'] = 1
        # Spirit also binds against the hoarder/ghost field to drain trapped Q
        sim.set_bond("HolySpirit", "Satan", 0.9)
        for k in range(3):
            sim.set_bond("HolySpirit", f"Ghost_{k}", 1.0)

    sim.step()
    history_boundary_Q.append(sim.Q_boundary)

# --- VISUALIZATION ---
nodes = sim.nodes
time_axis = range(TICKS)

# Extract Data
christ = nodes[sim.node_map["Christ"]]
satan  = nodes[sim.node_map["Satan"]]
spirit = nodes[sim.node_map["HolySpirit"]]

# Averages
human_Qs = np.mean([n['h_Q'] for n in nodes if n['type']=='Human'], axis=0)
human_as = np.mean([n['h_a'] for n in nodes if n['type']=='Human'], axis=0)
human_Ps = np.mean([n['h_P'] for n in nodes if n['type']=='Human'], axis=0)
ghost_Qs = np.mean([n['h_Q'] for n in nodes if n['type']=='Ghost'], axis=0)

# --- DIAGNOSTICS: does Avg Human a/P actually plateau after Resurrection? ---

def _window_stats(series, t0, t1):
    s = np.asarray(series[t0:t1], dtype=float)
    if s.size == 0:
        return None
    return {
        't0': t0,
        't1': t1,
        'start': float(s[0]),
        'end': float(s[-1]),
        'delta': float(s[-1] - s[0]),
        'mean': float(np.mean(s)),
        'std': float(np.std(s)),
        'min': float(np.min(s)),
        'max': float(np.max(s)),
    }

# Define windows: immediately after resurrection, and late steady-state
w_early = (T_RESURRECTION, min(T_RESURRECTION + 200, TICKS))
w_late  = (max(TICKS - 200, 0), TICKS)

stats = {
    'human_a_early': _window_stats(human_as, *w_early),
    'human_a_late':  _window_stats(human_as, *w_late),
    'human_P_early': _window_stats(human_Ps, *w_early),
    'human_P_late':  _window_stats(human_Ps, *w_late),
}

print("\n=== POST-RUN DIAGNOSTICS (Avg Human) ===")
print(f"Resurrection tick: {T_RESURRECTION} | TICKS: {TICKS}")
for k, v in stats.items():
    if v is None:
        continue
    print(f"{k}: t[{v['t0']}:{v['t1']}], start={v['start']:.4f}, end={v['end']:.4f}, delta={v['delta']:.4f}, mean={v['mean']:.4f}, std={v['std']:.4f}")

# Simple 'taper check': compare early slope vs late slope magnitude
if stats['human_a_early'] and stats['human_a_late']:
    early_span = max(1, stats['human_a_early']['t1'] - stats['human_a_early']['t0'] - 1)
    late_span  = max(1, stats['human_a_late']['t1'] - stats['human_a_late']['t0'] - 1)
    a_slope_early = stats['human_a_early']['delta'] / early_span
    a_slope_late  = stats['human_a_late']['delta'] / late_span
    print(f"Avg slope human_a: early={a_slope_early:.6f}/tick | late={a_slope_late:.6f}/tick")

if stats['human_P_early'] and stats['human_P_late']:
    early_span = max(1, stats['human_P_early']['t1'] - stats['human_P_early']['t0'] - 1)
    late_span  = max(1, stats['human_P_late']['t1'] - stats['human_P_late']['t0'] - 1)
    P_slope_early = stats['human_P_early']['delta'] / early_span
    P_slope_late  = stats['human_P_late']['delta'] / late_span
    print(f"Avg slope human_P: early={P_slope_early:.6f}/tick | late={P_slope_late:.6f}/tick")
print("======================================\n")

fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)

# PLOT 1: DEBT DYNAMICS (Conservation Check)
ax = axes[0]
ax.plot(time_axis, christ['h_Q'], label='Christ (Q)', color='#FFD700', lw=2.5)
ax.plot(time_axis, satan['h_Q'], label='Satan (Q)', color='black', lw=2.5)
ax.plot(time_axis, human_Qs, label='Avg Human (Q)', color='blue', ls='--')
ax.plot(time_axis, history_boundary_Q, label='Boundary Store (God) (Q)', color='green', ls=':')
ax.plot(time_axis, spirit['h_Q'], label='HolySpirit (Q)', color='purple', ls='-.')
ax.set_title("Conservation of Structural Debt Mass (Q)")
ax.set_ylabel("Q (Conserved Past)")
ax.axvline(T_CROSS, color='red', alpha=0.5, label='Cross')
ax.axvline(T_RESURRECTION, color='green', alpha=0.5, label='Resurrection')
ax.axvline(T_BIRTH, color='purple', alpha=0.5, label='Birth')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# PLOT 2: AGENCY (The Collapse)
ax = axes[1]
ax.plot(time_axis, christ['h_a'], label='Christ (a=1)', color='#FFD700')
ax.plot(time_axis, satan['h_a'], label='Satan (a)', color='black')
ax.plot(time_axis, human_as, label='Avg Human (a)', color='blue', ls='--')
ghost_a = np.mean([n['h_a'] for n in nodes if n['type'] == 'Ghost'], axis=0)
ax.plot(time_axis, ghost_a, label='Ghost (a=0)', color='grey', ls=':')
ax.plot(time_axis, spirit['h_a'], label='HolySpirit (a)', color='purple')
ax.set_title("Agency Evolution (Openness)")
ax.set_ylabel("Agency (a)")
ax.axvline(T_CROSS, color='red', alpha=0.5)
ax.axvline(T_RESURRECTION, color='green', alpha=0.5)
ax.axvline(T_BIRTH, color='purple', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

# PLOT 3: PRESENCE (Life vs Death)
ax = axes[2]
ax.plot(time_axis, christ['h_P'], label='Christ P', color='#FFD700')
ax.plot(time_axis, satan['h_P'], label='Satan P', color='black')
ax.plot(time_axis, human_Ps, label='Avg Human P', color='blue', ls='--')
ax.plot(time_axis, spirit['h_P'], label='HolySpirit P', color='purple')
ax.set_title("Presence (Emergent Time Rate)")
ax.set_ylabel("P (Clock Rate)")
ax.set_xlabel("Time (Ticks)")
ax.axvline(T_CROSS, color='red', alpha=0.5)
ax.axvline(T_RESURRECTION, color='green', alpha=0.5)
ax.axvline(T_BIRTH, color='purple', alpha=0.5)
ax.grid(True, alpha=0.3)

# PLOT 4: STORED RESOURCE (F)
ax = axes[3]
ax.plot(time_axis, christ['h_F'], label='Christ F', color='#FFD700')
ax.plot(time_axis, satan['h_F'], label='Satan F', color='black')
ax.plot(time_axis, [np.mean([n['h_F'][ti] for n in nodes if n['type']=='Human']) for ti in time_axis], label='Avg Human F', color='blue', ls='--')
ax.set_title("Stored Resource (F)")
ax.set_ylabel("F")
ax.set_xlabel("Time (Ticks)")
ax.axvline(T_BIRTH, color='purple', alpha=0.5, label='Birth')
ax.axvline(T_CROSS, color='red', alpha=0.5, label='Cross')
ax.axvline(T_RESURRECTION, color='green', alpha=0.5, label='Resurrection')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()