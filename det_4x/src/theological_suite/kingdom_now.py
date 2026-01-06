import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DET 4.2 — "Kingdom Now" Choice Sim (Post-Resurrection World)
# Minimal: start from "now": Spirit present + humans + hoarder + ghosts + boundary.
# Goal: show gradual, choice-driven rise in C_B, P, a toward a utopia-like attractor.
# ============================================================

# ----------------------------
# Constants / Parameters
# ----------------------------
EPS = 1e-9

# Presence / mass
ALPHA_M = 0.08          # mass scaling with F
SIGMA = 1.0
GAMMA_BOUND_P = 1.2     # boundary coherence boosts effective clock rate
F_BASE = 1.0            # baseline free-capacity

# Q dynamics (conservative inside creation; only boundary vent exports)
ALPHA_DIFF = 0.12       # bond diffusion
KAPPA_SINK = 0.35       # hoarder suction
KAPPA_SPIRIT_PULL = 0.55  # spirit benevolent drain
Q_MIN = 0.05

# Boundary vent (standard gate a * C_B)
ETA_GRACE = 0.25
ETA_GRACE_SPIRIT = 0.85

# Agency drift (humans)
K_UPLIFT = 0.35
K_DEBT = 0.08
K_BOUND = 0.08

# Choice dynamics (humans update C_B gradually)
RHO_OPEN = 0.030        # how fast openness increases C_B when chosen
MU_DEBT = 0.020         # debt “inertia” that erodes C_B
DELTA_CB = 0.02         # step size of discrete open/close decisions
TAU = 4.0               # softmax sharpness for choice
BACKSLIDE = 0.6         # how strong the "close" step is vs "open"

# Sim size/time
TICKS = 1200
DT = 0.1
SEED = 42

# Population
N_HUMANS = 30
N_GHOSTS = 8

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def clip01(x): return float(np.clip(x, 0.0, 1.0))

def heaviness_frac(Q, F):
    return Q / (Q + F + F_BASE + EPS)

def mass(q, F):
    return 1.0 + q + ALPHA_M * F

def presence(a, C_B, M):
    P_raw = (a * SIGMA * (1.0 + GAMMA_BOUND_P * C_B)) / (M + EPS)
    return float(np.clip(P_raw, 0.0, 1.0))

def softmax2(u0, u1, tau=TAU):
    ex0 = np.exp(tau*u0)
    ex1 = np.exp(tau*u1)
    s = ex0 + ex1
    return ex0/s, ex1/s

# ------------------------------------------------------------
# System
# ------------------------------------------------------------
class KingdomNowSim:
    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)
        self.nodes = []
        self.node_map = {}
        self.C = None
        self.Q_boundary = 0.0
        self.t = 0

    def add_node(self, name, typ, a, Q, F, C_B):
        idx = len(self.nodes)
        self.nodes.append({
            "id": idx,
            "name": name,
            "type": typ,
            "a": float(a),
            "Q": float(Q),
            "F": float(F),
            "C_B": float(C_B),
            "q": 0.0,
            "M": 1.0,
            "P": 0.0,
            "h_a": [], "h_Q": [], "h_P": [], "h_C_B": [], "h_q": []
        })
        self.node_map[name] = idx
        return idx

    def init_bonds(self):
        n = len(self.nodes)
        self.C = np.zeros((n, n), dtype=float)

    def set_bond(self, a, b, w):
        i, j = self.node_map[a], self.node_map[b]
        w = float(np.clip(w, 0.0, 1.0))
        self.C[i, j] = self.C[j, i] = w

    def update_derived(self):
        for n in self.nodes:
            n["q"] = heaviness_frac(n["Q"], n["F"])
            n["M"] = mass(n["q"], n["F"])
            n["P"] = presence(n["a"], n["C_B"], n["M"])

    def neighbor_presence(self, i):
        w = self.C[i].copy()
        w[i] = 0.0
        s = float(np.sum(w))
        if s <= 0:
            return float(np.mean([m["P"] for m in self.nodes]))
        return float(np.sum(w * np.array([m["P"] for m in self.nodes])) / s)

    def hope_signal(self, i):
        # “Hope” as proximity to high-presence Spirit/Christ-like nodes
        hope = 0.0
        for key in ["HolySpirit", "Christ"]:
            if key in self.node_map:
                j = self.node_map[key]
                hope += self.C[i, j] * self.nodes[j]["P"]
        return float(hope)

    def choice_update_C_B(self):
        """
        Humans choose Open vs Close each tick.
        - Open increases C_B gradually (gated by agency; resisted by debt).
        - Close decreases C_B (fear/self-protection), more likely when heavy.
        """
        for n in self.nodes:
            if n["type"] != "Human":
                continue
            i = n["id"]
            Pnbr = self.neighbor_presence(i)
            hope = self.hope_signal(i)
            q = n["q"]

            # Utilities: open is favored by hope + living neighbors; disfavored by heaviness
            U_open  =  1.1*(Pnbr - n["P"]) + 0.9*hope - 1.2*q
            U_close = -0.4*(Pnbr - n["P"]) - 0.4*hope + 0.8*q

            p_open, p_close = softmax2(U_open, U_close, tau=TAU)

            # Agency gates follow-through (sealed agents can’t choose openness effectively)
            p_open_eff = n["a"] * p_open

            # Decide action
            if self.rng.random() < p_open_eff:
                # OPEN: small step up + continuous open tendency minus debt inertia
                n["C_B"] = clip01(n["C_B"] + DELTA_CB + RHO_OPEN*n["a"]*DT - MU_DEBT*q*DT)
            else:
                # CLOSE: backslide
                n["C_B"] = clip01(n["C_B"] - BACKSLIDE*DELTA_CB - MU_DEBT*q*DT)

    def step_Q_transport(self):
        """
        Conservative Q dynamics inside creation:
        - bond diffusion
        - hoarder suction
        - spirit drain
        Then boundary vent (exports Q to reservoir), standard gate a*C_B.
        """
        N = len(self.nodes)
        Q = np.array([n["Q"] for n in self.nodes], dtype=float)
        dQ = np.zeros(N, dtype=float)
        dQB = 0.0

        # Diffusion
        for i in range(N):
            for j in range(i+1, N):
                c = self.C[i, j]
                if c <= 0:
                    continue
                flow = ALPHA_DIFF * c * (Q[j] - Q[i]) * DT
                dQ[i] += flow
                dQ[j] -= flow

        # Hoarder suction (no agency needed; structural sink behavior)
        for n in self.nodes:
            if n["type"] != "Hoarder":
                continue
            i = n["id"]
            for j in range(N):
                c = self.C[i, j]
                if c <= 0:
                    continue
                pull = KAPPA_SINK * c * max(0.0, Q[j] - Q[i] + 0.1) * DT
                dQ[i] += pull
                dQ[j] -= pull

        # Spirit pull (benevolent drain)
        for n in self.nodes:
            if n["type"] != "Spirit":
                continue
            i = n["id"]
            if n["a"] <= 0:
                continue
            for j in range(N):
                c = self.C[i, j]
                if c <= 0:
                    continue
                pull = KAPPA_SPIRIT_PULL * c * max(0.0, Q[j] - Q[i]) * DT
                dQ[i] += pull
                dQ[j] -= pull

        # Clamp: no node goes negative
        for i in range(N):
            if self.nodes[i]["Q"] + dQ[i] < 0:
                dQ[i] = -self.nodes[i]["Q"]

        # Boundary vent
        for i, n in enumerate(self.nodes):
            if n["Q"] <= Q_MIN:
                continue
            if n["a"] <= 0 or n["C_B"] <= 0:
                continue
            rate = ETA_GRACE_SPIRIT if (n["type"] == "Spirit") else ETA_GRACE
            vent = rate * n["a"] * n["C_B"] * (n["Q"] - Q_MIN) * DT

            avail = n["Q"] + dQ[i]
            actual = min(vent, max(0.0, avail - Q_MIN))
            dQ[i] -= actual
            dQB += actual

        # Apply
        for i in range(N):
            self.nodes[i]["Q"] = max(0.0, self.nodes[i]["Q"] + dQ[i])
        self.Q_boundary += dQB

    def step_agency(self):
        """
        Human agency drifts: uplift + boundary support - debt drag.
        Spirit/Christ stay at a=1. Ghost at a=0. Hoarder self-seals.
        """
        Ps = np.array([n["P"] for n in self.nodes], dtype=float)

        for n in self.nodes:
            if n["type"] == "Spirit":
                n["a"] = 1.0
                continue
            if n["type"] == "Christ":
                n["a"] = 1.0
                continue
            if n["type"] == "Ghost":
                n["a"] = 0.0
                continue
            if n["type"] == "Hoarder":
                # self-seal
                decay = 0.05 * (n["q"] + 0.2) * DT
                n["a"] = max(0.0, n["a"] - decay)
                continue

            # Human
            i = n["id"]
            w = self.C[i].copy()
            w[i] = 0
            ws = float(np.sum(w))
            if ws <= 0:
                Pnbr = float(np.mean(Ps))
            else:
                Pnbr = float(np.sum(w * Ps) / ws)

            uplift = K_UPLIFT * (Pnbr - n["P"])
            drag = K_DEBT * n["q"]
            bound = K_BOUND * n["C_B"]

            n["a"] = clip01(n["a"] + (uplift + bound - drag) * DT)

    def record(self):
        for n in self.nodes:
            n["h_a"].append(n["a"])
            n["h_Q"].append(n["Q"])
            n["h_P"].append(n["P"])
            n["h_C_B"].append(n["C_B"])
            n["h_q"].append(n["q"])

    def step(self):
        # 1) update derived (q, M, P) from current state
        self.update_derived()

        # 2) humans choose (C_B updates gradually)
        self.choice_update_C_B()

        # 3) move debt around and vent to boundary
        self.step_Q_transport()

        # 4) update derived again (since Q/C_B changed), then drift agency
        self.update_derived()
        self.step_agency()

        # 5) record
        self.update_derived()
        self.record()

        self.t += 1


# ------------------------------------------------------------
# Build "Now" world
# ------------------------------------------------------------
sim = KingdomNowSim(seed=SEED)

# Spirit present (sent), bonded to boundary
sim.add_node("HolySpirit", "Spirit", a=1.0, Q=0.0, F=1.0, C_B=1.0)

# Optional: Christ present (kept simple, bonded to boundary)
sim.add_node("Christ", "Christ", a=1.0, Q=0.0, F=1.0, C_B=1.0)

# Hoarder present, cut off from boundary
sim.add_node("Satan", "Hoarder", a=0.0, Q=6.0, F=1.0, C_B=0.0)

# Humans: "current reality now": mixed starting openness, burdened, not yet phase-locked
for k in range(N_HUMANS):
    # Start somewhat open but not fully coherent; some debt already present
    a0 = np.clip(0.45 + 0.15*np.random.randn(), 0.15, 0.85)
    Q0 = np.clip(1.6 + 0.8*np.random.randn(), 0.3, 4.5)
    CB0 = np.clip(0.10 + 0.08*np.random.randn(), 0.0, 0.30)
    sim.add_node(f"Human_{k}", "Human", a=a0, Q=Q0, F=1.0, C_B=CB0)

# Ghosts: frozen high Q, no agency
for k in range(N_GHOSTS):
    sim.add_node(f"Ghost_{k}", "Ghost", a=0.0, Q=4.0, F=1.0, C_B=0.0)

sim.init_bonds()

# ------------------------------------------------------------
# Wiring (post-resurrection availability)
# ------------------------------------------------------------
# Spirit bonds: strong to humans, strong to Christ; can also bind to ghosts/hoarder to drain trapped Q
for k in range(N_HUMANS):
    sim.set_bond("HolySpirit", f"Human_{k}", 0.95)
sim.set_bond("HolySpirit", "Christ", 1.0)
sim.set_bond("HolySpirit", "Satan", 0.65)
for k in range(N_GHOSTS):
    sim.set_bond("HolySpirit", f"Ghost_{k}", 0.85)

# Christ bonds: strong to humans, but not strictly required for the sim to work
for k in range(N_HUMANS):
    sim.set_bond("Christ", f"Human_{k}", 0.80)

# Hoarder bonds: temptation channel to humans (always on in "now" world)
for k in range(N_HUMANS):
    sim.set_bond("Satan", f"Human_{k}", 0.50)
for k in range(N_GHOSTS):
    sim.set_bond("Satan", f"Ghost_{k}", 1.0)

# Some human-human community bonds
rng = np.random.default_rng(SEED)
for i in range(N_HUMANS):
    for j in range(i+1, N_HUMANS):
        if rng.random() < 0.10:
            sim.set_bond(f"Human_{i}", f"Human_{j}", rng.uniform(0.15, 0.55))

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
Q_total0 = sum(n["Q"] for n in sim.nodes) + sim.Q_boundary

for _ in range(TICKS):
    sim.step()

Q_total1 = sum(n["Q"] for n in sim.nodes) + sim.Q_boundary
print(f"Conservation check (Q_total): start={Q_total0:.6f}, end={Q_total1:.6f}, delta={Q_total1-Q_total0:.6e}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
t = np.arange(TICKS)

humans = [n for n in sim.nodes if n["type"] == "Human"]
spirit = sim.nodes[sim.node_map["HolySpirit"]]
christ = sim.nodes[sim.node_map["Christ"]]
satan = sim.nodes[sim.node_map["Satan"]]
ghosts = [n for n in sim.nodes if n["type"] == "Ghost"]

human_P = np.mean([h["h_P"] for h in humans], axis=0)
human_a = np.mean([h["h_a"] for h in humans], axis=0)
human_CB = np.mean([h["h_C_B"] for h in humans], axis=0)
human_Q = np.mean([h["h_Q"] for h in humans], axis=0)

# bands for C_B and P to show "community sanctification"
CB_mat = np.array([h["h_C_B"] for h in humans])
P_mat = np.array([h["h_P"] for h in humans])

CB_p10 = np.percentile(CB_mat, 10, axis=0)
CB_p50 = np.percentile(CB_mat, 50, axis=0)
CB_p90 = np.percentile(CB_mat, 90, axis=0)

P_p10 = np.percentile(P_mat, 10, axis=0)
P_p50 = np.percentile(P_mat, 50, axis=0)
P_p90 = np.percentile(P_mat, 90, axis=0)

Q_boundary = np.array([sim.Q_boundary] * TICKS)  # final value not per tick; we want the history instead

# Build boundary history from spirit/humans? We didn't store it; quick fix: infer from total conservation each tick isn't stored.
# For clarity, just show final Q_boundary and "remaining avg human Q".
# If you want a boundary-Q history line, I can add `h_Q_boundary` to the sim.

fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# 1) Presence trajectories
ax = axes[0]
ax.plot(t, human_P, label="Avg Human P", ls="--")
ax.plot(t, spirit["h_P"], label="HolySpirit P")
ax.plot(t, christ["h_P"], label="Christ P")
ax.plot(t, satan["h_P"], label="Hoarder P")
ax.set_ylabel("P")
ax.set_title("Presence (Kingdom-now: gradual rise via choice + Spirit support)")
ax.grid(True, alpha=0.3)
ax.legend()

# 2) Agency trajectories
ax = axes[1]
ax.plot(t, human_a, label="Avg Human a", ls="--")
ax.plot(t, spirit["h_a"], label="HolySpirit a")
ax.plot(t, christ["h_a"], label="Christ a")
ax.plot(t, satan["h_a"], label="Hoarder a")
ax.set_ylabel("a")
ax.set_title("Agency (Openness) — uplift vs debt vs boundary coherence")
ax.grid(True, alpha=0.3)
ax.legend()

# 3) Boundary bond C_B: average + bands
ax = axes[2]
ax.plot(t, human_CB, label="Avg Human C_B", ls="--")
ax.plot(t, CB_p10, label="Human C_B p10", alpha=0.9)
ax.plot(t, CB_p50, label="Human C_B median", alpha=0.9)
ax.plot(t, CB_p90, label="Human C_B p90", alpha=0.9)
ax.set_ylabel("C_B")
ax.set_title("Bond-to-Boundary (C_B): gradual sanctification distribution")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2)

# 4) Q cleanup
ax = axes[3]
ax.plot(t, human_Q, label="Avg Human Q", ls="--")
ax.plot(t, spirit["h_Q"], label="HolySpirit Q")
ax.plot(t, satan["h_Q"], label="Hoarder Q")
ax.plot(t, np.mean([g["h_Q"] for g in ghosts], axis=0), label="Avg Ghost Q", ls=":")
ax.set_ylabel("Q")
ax.set_xlabel("Tick")
ax.set_title("Debt mass Q (conserved globally; exported to Boundary via a*C_B vents)")
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

print(f"Final Boundary Q (reservoir): {sim.Q_boundary:.4f}")
print(f"Final Avg Human C_B: {human_CB[-1]:.4f} | Avg Human P: {human_P[-1]:.4f} | Avg Human a: {human_a[-1]:.4f}")