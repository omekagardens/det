import numpy as np
import matplotlib.pyplot as plt

"""DET5 Creation/Fall/Cross/Resurrection (Utopia-First Replay)

This is a narrative *mapping* built on DET5-style mechanics (strictly local, agency-inviolable, lawful recovery).
We intentionally frame from a future healed regime and "replay" perturbations (Fall/Cross/Resurrection)
inside that regime to test whether the DET5 loop can (a) absorb fracture and (b) recover without coercion.

Trinity mapping (module-level, not person-as-node):
  - God  == I (Information):   lawful local recovery channels (need-weighted injection + bond healing)
  - Christ== A (Agency):       inviolable openness dynamics; reconciliation requires mutual openness
  - Spirit== k (Movement):     event-counted time + directed motion memory (optional momentum module)

IMPORTANT: This is a *toy* — meant to be DET5-shaped, not a literal theology engine.
"""

# -----------------------------
# Core numeric constants
# -----------------------------
EPS = 1e-9

# Local clock / presence
DTK = 1.0  # one global step == one event-count tick (k)
F_MIN = 0.35  # local need threshold for I-channel injection

# Canonical DET5-ish knobs (report if you change them)
ALPHA_Q = 0.020          # q-locking rate (structural debt accumulation from loss)
OMEGA_0 = 0.00           # base phase rotation (set >0 for oscillatory worlds)
GAMMA_PHASE = 0.020      # optional phase coupling gain (used after Resurrection)

# Transport
ALPHA_DIFF = 0.22        # diffusive/phase transport magnitude

# Optional momentum module (Spirit / k enhancement)
MOMENTUM_ENABLED = False
ALPHA_PI = 0.06
LAMBDA_PI = 0.020
MU_PI = 0.22
PI_MAX = 2.5

# "I" channel (lawful recovery): injection + bond healing (local, gated)
I_INJECT_GAIN = 1.00
I_HEAL_GAIN = 0.85

# -----------------------------
# Helpers
# -----------------------------

def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def local_mean(values, weights=None):
    v = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(v))
    w = np.asarray(weights, dtype=float)
    s = float(np.sum(w))
    if s <= 0:
        return float(np.mean(v))
    return float(np.sum(w * v) / s)


# -----------------------------
# DET5-ish Graph System
# -----------------------------

class DET5System:
    """Strictly local graph dynamics. No global state is used for updates.

    State per node i:
      F_i (resource), q_i (structural debt), a_i (agency), theta_i (phase), tau_i (proper time), k_i (event count)

    State per bond (i,j):
      sigma_ij (processing), C_ij (coherence), pi_ij (optional antisymmetric momentum)

    Notes:
      - "I" is modeled as lawful local recovery operators (need-weighted injection + healing).
      - "A" is modeled as inviolable agency and mutual-openness requirements.
      - "k" is modeled as event counted time + optional momentum memory.
    """

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.nodes = []
        self.name_to_id = {}
        self.sigma = None
        self.C = None
        self.pi = None
        self.t = 0
        self.log = []

    def add_node(self, name, kind="Human", F=1.0, q=0.05, a=0.90, theta=None):
        idx = len(self.nodes)
        if theta is None:
            theta = float(self.rng.uniform(0, 2*np.pi))
        node = {
            "id": idx,
            "name": name,
            "kind": kind,
            "F": float(F),
            "q": float(q),
            "a": float(a),
            "theta": float(theta),
            "tau": 0.0,
            "k": 0,
            # derived
            "H": 0.0,
            "P": 0.0,
            "M": 0.0,
            # history
            "h_F": [],
            "h_q": [],
            "h_a": [],
            "h_P": [],
            "h_tau": [],
        }
        self.nodes.append(node)
        self.name_to_id[name] = idx
        return idx

    def init_bonds(self):
        n = len(self.nodes)
        self.sigma = np.zeros((n, n), dtype=float)
        self.C = np.zeros((n, n), dtype=float)
        self.pi = np.zeros((n, n), dtype=float)  # antisymmetric

    def set_bond(self, name1, name2, sigma=1.0, coherence=0.90):
        i, j = self.name_to_id[name1], self.name_to_id[name2]
        if i == j:
            return
        s = float(max(0.0, sigma))
        c = float(np.clip(coherence, 0.0, 1.0))
        self.sigma[i, j] = self.sigma[j, i] = s
        self.C[i, j] = self.C[j, i] = c

    # -----------------------------
    # Local clock / presence
    # -----------------------------

    def update_presence(self):
        """Compute H_i, P_i, M_i strictly locally."""
        n = len(self.nodes)
        for i in range(n):
            # local coordination load
            H = 0.0
            for j in range(n):
                if i == j:
                    continue
                if self.sigma[i, j] > 0:
                    H += np.sqrt(self.C[i, j]) * self.sigma[i, j]

            # local node processing (sigma_i)
            sigma_i = float(np.sum(self.sigma[i]))

            # DET5-ish: P = a * sigma_i / ((1+F) * (1+H))
            F = self.nodes[i]["F"]
            a = self.nodes[i]["a"]
            P = (a * sigma_i) / ((1.0 + F) * (1.0 + H) + EPS)

            # derived mass
            M = 1.0 / (P + EPS)

            self.nodes[i]["H"] = float(H)
            self.nodes[i]["P"] = float(np.clip(P, 0.0, 1.0))
            self.nodes[i]["M"] = float(M)

    # -----------------------------
    # Flow / transport
    # -----------------------------

    def _psi_local(self, i):
        """Local-normalized psi_i = sqrt(F_i / sum_{N(i)} F) * exp(i theta_i)."""
        n = len(self.nodes)
        denom = self.nodes[i]["F"]
        for j in range(n):
            if i != j and self.sigma[i, j] > 0:
                denom += self.nodes[j]["F"]
        denom = denom + EPS
        amp = np.sqrt(self.nodes[i]["F"] / denom)
        return amp * np.exp(1j * self.nodes[i]["theta"])

    def _compute_fluxes(self, dtau, phase_coupling=False):
        """Compute antisymmetric flux matrices J_diff and (optional) J_mom."""
        n = len(self.nodes)
        psi = [self._psi_local(i) for i in range(n)]
        J_diff = np.zeros((n, n), dtype=float)
        J_mom = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i + 1, n):
                s = self.sigma[i, j]
                if s <= 0:
                    continue

                Cij = self.C[i, j]
                aij = np.sqrt(self.nodes[i]["a"] * self.nodes[j]["a"])
                rootC = np.sqrt(Cij)

                # antisymmetric bracket
                phase_term = np.imag(np.conj(psi[i]) * psi[j])
                classical_term = (self.nodes[i]["F"] - self.nodes[j]["F"])
                bracket = (rootC * phase_term) + ((1.0 - rootC) * classical_term)

                # diffusive/phase transport, agency-gated symmetrically
                J = ALPHA_DIFF * aij * s * bracket

                J_diff[i, j] = J
                J_diff[j, i] = -J

                # momentum-driven drift (optional; Spirit/k enhancement)
                if MOMENTUM_ENABLED:
                    Favg = 0.5 * (self.nodes[i]["F"] + self.nodes[j]["F"])
                    Jp = MU_PI * s * self.pi[i, j] * Favg
                    J_mom[i, j] = Jp
                    J_mom[j, i] = -Jp

        # optional phase coupling (kept separate from flux to preserve clarity)
        if phase_coupling:
            # Kuramoto-like neighbor pull, gated by mutual openness and coherence
            for i in range(n):
                dtheta = 0.0
                for j in range(n):
                    if i == j or self.sigma[i, j] <= 0:
                        continue
                    dtheta += (
                        GAMMA_PHASE
                        * self.sigma[i, j]
                        * np.sqrt(self.C[i, j])
                        * self.nodes[i]["a"]
                        * self.nodes[j]["a"]
                        * np.sin(self.nodes[j]["theta"] - self.nodes[i]["theta"])
                    )
                self.nodes[i]["theta"] = float((self.nodes[i]["theta"] + dtheta * dtau[i]) % (2*np.pi))

        return J_diff, J_mom

    # -----------------------------
    # "I" channel: lawful recovery (injection + healing)
    # -----------------------------

    def _i_channel(self, J_total, dtau):
        """Local dissipation -> (a) need-weighted injection, (b) bond healing.

        This is the DET "boundary operator" shape, but here framed as the Information (I) channel:
        lawful, local, and gated by agency (non-coercive).
        """
        n = len(self.nodes)

        # local dissipation
        D = np.zeros(n, dtype=float)
        for i in range(n):
            out = 0.0
            for j in range(n):
                if i == j or self.sigma[i, j] <= 0:
                    continue
                out += abs(J_total[i, j])
            D[i] = out * dtau[i]

        # injection (need-weighted within neighborhood)
        I_inj = np.zeros(n, dtype=float)
        for i in range(n):
            # neighborhood weights (including self)
            nbrs = [i] + [j for j in range(n) if j != i and self.sigma[i, j] > 0]
            needs = []
            wts = []
            for k in nbrs:
                need = max(0.0, F_MIN - self.nodes[k]["F"])
                w = self.nodes[k]["a"] * need
                needs.append(need)
                wts.append(w)
            Z = float(np.sum(wts))
            if Z <= 0:
                continue
            for k, w in zip(nbrs, wts):
                I_inj[k] += I_INJECT_GAIN * D[i] * (w / (Z + EPS))

        # healing (mutual openness)
        dC = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if self.sigma[i, j] <= 0:
                    continue
                u_ij = self.nodes[i]["a"] * self.nodes[j]["a"] * (1.0 - self.C[i, j])
                if u_ij <= 0:
                    continue

                # bond-local neighborhood: edges incident to i or j (toy E_R)
                Zc = 0.0
                for m in range(n):
                    for n2 in range(m + 1, n):
                        if self.sigma[m, n2] <= 0:
                            continue
                        if (m in (i, j)) or (n2 in (i, j)):
                            Zc += (
                                self.nodes[m]["a"]
                                * self.nodes[n2]["a"]
                                * (1.0 - self.C[m, n2])
                            )

                Dij = 0.5 * (D[i] + D[j])
                d = I_HEAL_GAIN * Dij * (u_ij / (Zc + EPS))
                dC[i, j] += d
                dC[j, i] += d

        return D, I_inj, dC

    # -----------------------------
    # Momentum update (optional)
    # -----------------------------

    def _update_momentum(self, J_diff, dtau):
        if not MOMENTUM_ENABLED:
            return
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if self.sigma[i, j] <= 0:
                    continue
                dt_ij = 0.5 * (dtau[i] + dtau[j])
                pij = self.pi[i, j]
                pij = (1.0 - LAMBDA_PI * dt_ij) * pij + (ALPHA_PI * J_diff[i, j] * dt_ij)
                pij = float(np.clip(pij, -PI_MAX, PI_MAX))
                self.pi[i, j] = pij
                self.pi[j, i] = -pij

    # -----------------------------
    # Agency update (A / Christ mapping)
    # -----------------------------

    def _update_agency(self):
        n = len(self.nodes)
        Ps = np.array([self.nodes[i]["P"] for i in range(n)], dtype=float)

        for i in range(n):
            # neighborhood mean presence (strictly local)
            nbrs = [j for j in range(n) if j != i and self.sigma[i, j] > 0]
            if len(nbrs) == 0:
                Pbar = float(np.mean(Ps))
            else:
                weights = np.array([self.sigma[i, j] for j in nbrs], dtype=float)
                Pbar = local_mean(Ps[nbrs], weights=weights)

            # canonical drift: a += (P_i - Pbar) - q
            a = self.nodes[i]["a"]
            q = self.nodes[i]["q"]
            a_new = a + (self.nodes[i]["P"] - Pbar) - q
            self.nodes[i]["a"] = clip01(a_new)

    # -----------------------------
    # q-locking (I / memory)
    # -----------------------------

    def _update_q(self, F_old, F_new):
        n = len(self.nodes)
        for i in range(n):
            loss = max(0.0, F_old[i] - F_new[i])
            self.nodes[i]["q"] = clip01(self.nodes[i]["q"] + ALPHA_Q * loss)

    # -----------------------------
    # Step
    # -----------------------------

    def step(self, phase_coupling=False):
        n = len(self.nodes)

        # (1) presence and local dtau
        self.update_presence()
        dtau = np.array([self.nodes[i]["P"] * DTK for i in range(n)], dtype=float)

        # (2) fluxes
        J_diff, J_mom = self._compute_fluxes(dtau, phase_coupling=phase_coupling)
        J_total = J_diff + J_mom

        # (3) I-channel: dissipation -> injection + healing
        D, I_inj, dC = self._i_channel(J_total, dtau)

        # (4) update F (conservative transport + local injection)
        F_old = np.array([self.nodes[i]["F"] for i in range(n)], dtype=float)
        F_new = F_old.copy()

        for i in range(n):
            # sender-clocked transport
            net = 0.0
            for j in range(n):
                if i == j or self.sigma[i, j] <= 0:
                    continue
                net -= J_total[i, j] * dtau[i]
            F_new[i] = max(0.0, F_new[i] + net + I_inj[i])

        for i in range(n):
            self.nodes[i]["F"] = float(F_new[i])

        # (5) momentum update (uses J_diff)
        self._update_momentum(J_diff, dtau)

        # (6) q-locking from loss
        self._update_q(F_old, F_new)

        # (7) agency update
        self._update_agency()

        # (8) coherence healing (bounded)
        for i in range(n):
            for j in range(n):
                if i == j or self.sigma[i, j] <= 0:
                    continue
                self.C[i, j] = float(np.clip(self.C[i, j] + dC[i, j], 0.0, 1.0))

        # (9) phase baseline (optional)
        if OMEGA_0 != 0.0:
            for i in range(n):
                self.nodes[i]["theta"] = float((self.nodes[i]["theta"] + OMEGA_0 * dtau[i]) % (2*np.pi))

        # advance time
        for i in range(n):
            self.nodes[i]["tau"] += float(dtau[i])
            self.nodes[i]["k"] += 1

            # history
            self.nodes[i]["h_F"].append(self.nodes[i]["F"])
            self.nodes[i]["h_q"].append(self.nodes[i]["q"])
            self.nodes[i]["h_a"].append(self.nodes[i]["a"])
            self.nodes[i]["h_P"].append(self.nodes[i]["P"])
            self.nodes[i]["h_tau"].append(self.nodes[i]["tau"])

        self.t += 1


# -----------------------------
# Narrative Setup (Utopia-first)
# -----------------------------

sim = DET5System(seed=9)

# Humans (future healed baseline)
NUM_HUMANS = 10
for k in range(NUM_HUMANS):
    sim.add_node(f"Human_{k}", kind="Human", F=1.0, q=0.05, a=0.90)

# A "Hoarder" node (the fracture attractor). In utopia-first framing it exists as a *potential*.
sim.add_node("Hoarder", kind="Hoarder", F=0.8, q=0.60, a=0.15)

# An "Anointed" human (role-model for agency openness).
sim.add_node("Anointed", kind="Human", F=1.2, q=0.02, a=0.98)

sim.init_bonds()

# Baseline healed connectivity: humans are coherently bonded (high C)
for i in range(NUM_HUMANS):
    for j in range(i + 1, NUM_HUMANS):
        sim.set_bond(f"Human_{i}", f"Human_{j}", sigma=0.7, coherence=0.92)

# Anointed is gently bonded into the human graph
for i in range(NUM_HUMANS):
    sim.set_bond("Anointed", f"Human_{i}", sigma=0.9, coherence=0.95)

# Hoarder is *weakly* connected in utopia baseline (temptation asleep)
for i in range(NUM_HUMANS):
    sim.set_bond("Hoarder", f"Human_{i}", sigma=0.10, coherence=0.30)

# -----------------------------
# Event schedule (memory replay inside healed regime)
# -----------------------------
TICKS = 900
T_FALL = 120
T_CROSS = 260
T_RES = 330

# We'll enable phase coupling + momentum after Resurrection (Spirit / k widened)

def apply_fall(sim):
    sim.log.append(f"t={sim.t}: FALL (fracture + hoarder coupling + coherence drop)")
    # Fracture: coherence drops, agency pressured, structural debt rises
    for k in range(NUM_HUMANS):
        i = sim.name_to_id[f"Human_{k}"]
        sim.nodes[i]["a"] = clip01(sim.nodes[i]["a"] - 0.25)
        sim.nodes[i]["q"] = clip01(sim.nodes[i]["q"] + 0.20)
        sim.nodes[i]["F"] = max(0.0, sim.nodes[i]["F"] - 0.25)

    # Temptation awakens: strengthen hoarder bonds, lower coherence on those bonds
    for k in range(NUM_HUMANS):
        sim.set_bond("Hoarder", f"Human_{k}", sigma=0.85, coherence=0.18)

    # The human-human bonds also lose coherence unevenly (noise, but local)
    for k in range(NUM_HUMANS):
        for j in range(k + 1, NUM_HUMANS):
            i1, i2 = sim.name_to_id[f"Human_{k}"], sim.name_to_id[f"Human_{j}"]
            if sim.sigma[i1, i2] > 0:
                sim.C[i1, i2] = sim.C[i2, i1] = float(np.clip(sim.C[i1, i2] - 0.35 * sim.rng.uniform(0.6, 1.0), 0.0, 1.0))


def apply_cross(sim):
    sim.log.append(f"t={sim.t}: CROSS (A-centered costly contact with fracture field)")
    # In DET terms: agency remains inviolable, but the Anointed opens maximal mutual-contact
    # with the hoarder field, increasing dissipation locally which powers I-channel healing.
    for k in range(NUM_HUMANS):
        sim.set_bond("Anointed", f"Human_{k}", sigma=1.0, coherence=0.80)

    # Direct bond between Anointed and Hoarder becomes strong (contact with the debt sink)
    sim.set_bond("Anointed", "Hoarder", sigma=1.0, coherence=0.55)

    # Symbolic forsakenness: the Anointed takes a local agency hit (not to zero)
    idx = sim.name_to_id["Anointed"]
    sim.nodes[idx]["a"] = clip01(sim.nodes[idx]["a"] - 0.35)
    sim.nodes[idx]["F"] = max(0.0, sim.nodes[idx]["F"] - 0.55)
    sim.nodes[idx]["q"] = clip01(sim.nodes[idx]["q"] + 0.25)


def apply_resurrection(sim):
    global MOMENTUM_ENABLED
    sim.log.append(f"t={sim.t}: RESURRECTION (k widened: directed motion memory + phase coherence)")

    # Restore Anointed openness (agency is not forced for others; only his own state here)
    idx = sim.name_to_id["Anointed"]
    sim.nodes[idx]["a"] = 0.99
    sim.nodes[idx]["F"] = max(sim.nodes[idx]["F"], 1.0)

    # Hoarder bonds weaken (fracture loses hold)
    for k in range(NUM_HUMANS):
        sim.set_bond("Hoarder", f"Human_{k}", sigma=0.20, coherence=0.25)

    # Spirit (k) module: enable momentum (local motion memory)
    MOMENTUM_ENABLED = True


# -----------------------------
# Run
# -----------------------------

phase_coupling = False

for _ in range(TICKS):
    if sim.t == T_FALL:
        apply_fall(sim)

    if sim.t == T_CROSS:
        apply_cross(sim)

    if sim.t == T_RES:
        apply_resurrection(sim)
        phase_coupling = True

    sim.step(phase_coupling=phase_coupling)


# -----------------------------
# Readouts
# -----------------------------

def mean_series(kind, key):
    arr = [np.array(n[key], dtype=float) for n in sim.nodes if n["kind"] == kind]
    if len(arr) == 0:
        return None
    return np.mean(arr, axis=0)

hum_F = mean_series("Human", "h_F")
hum_q = mean_series("Human", "h_q")
hum_a = mean_series("Human", "h_a")
hum_P = mean_series("Human", "h_P")

an = sim.nodes[sim.name_to_id["Anointed"]]
ho = sim.nodes[sim.name_to_id["Hoarder"]]

print("\n=== EVENT LOG ===")
for line in sim.log:
    print(line)

print("\n=== FINAL SNAPSHOT ===")
print(f"t={sim.t}")
print(f"Avg Human:  a={hum_a[-1]:.3f}  P={hum_P[-1]:.3f}  q={hum_q[-1]:.3f}  F={hum_F[-1]:.3f}")
print(f"Anointed:   a={an['h_a'][-1]:.3f}  P={an['h_P'][-1]:.3f}  q={an['h_q'][-1]:.3f}  F={an['h_F'][-1]:.3f}")
print(f"Hoarder:    a={ho['h_a'][-1]:.3f}  P={ho['h_P'][-1]:.3f}  q={ho['h_q'][-1]:.3f}  F={ho['h_F'][-1]:.3f}")

# Post-event window diagnostics

def window_stats(series, t0, t1):
    s = np.asarray(series[t0:t1], dtype=float)
    if s.size == 0:
        return None
    return dict(
        t0=t0,
        t1=t1,
        start=float(s[0]),
        end=float(s[-1]),
        delta=float(s[-1] - s[0]),
        mean=float(np.mean(s)),
        std=float(np.std(s)),
        min=float(np.min(s)),
        max=float(np.max(s)),
    )

w_pre = (0, T_FALL)
w_post = (T_RES, min(T_RES + 250, TICKS))
w_late = (max(TICKS - 250, 0), TICKS)

stats = {
    "hum_a_pre": window_stats(hum_a, *w_pre),
    "hum_a_post": window_stats(hum_a, *w_post),
    "hum_a_late": window_stats(hum_a, *w_late),
    "hum_P_pre": window_stats(hum_P, *w_pre),
    "hum_P_post": window_stats(hum_P, *w_post),
    "hum_P_late": window_stats(hum_P, *w_late),
}

print("\n=== WINDOW STATS (Avg Human) ===")
for k, v in stats.items():
    print(f"{k}: {v}")

# -----------------------------
# Plot
# -----------------------------

x = np.arange(TICKS)
fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)

# F
ax = axes[0]
ax.plot(x, hum_F, label="Avg Human F")
ax.plot(x, an["h_F"], label="Anointed F", lw=2)
ax.plot(x, ho["h_F"], label="Hoarder F", lw=2)
ax.set_ylabel("F")
ax.set_title("Resource (F)")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")

# q
ax = axes[1]
ax.plot(x, hum_q, label="Avg Human q")
ax.plot(x, an["h_q"], label="Anointed q", lw=2)
ax.plot(x, ho["h_q"], label="Hoarder q", lw=2)
ax.set_ylabel("q")
ax.set_title("Structural Debt (q)")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")

# a
ax = axes[2]
ax.plot(x, hum_a, label="Avg Human a")
ax.plot(x, an["h_a"], label="Anointed a", lw=2)
ax.plot(x, ho["h_a"], label="Hoarder a", lw=2)
ax.set_ylabel("a")
ax.set_title("Agency (a)  — Christ == A")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")

# P
ax = axes[3]
ax.plot(x, hum_P, label="Avg Human P")
ax.plot(x, an["h_P"], label="Anointed P", lw=2)
ax.plot(x, ho["h_P"], label="Hoarder P", lw=2)
ax.set_ylabel("P")
ax.set_xlabel("ticks (k)")
ax.set_title("Presence (P) / Local Clock  — Spirit == k")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")

for ax in axes:
    ax.axvline(T_FALL, color="red", alpha=0.5)
    ax.axvline(T_CROSS, color="purple", alpha=0.5)
    ax.axvline(T_RES, color="green", alpha=0.5)

plt.tight_layout()
plt.show()