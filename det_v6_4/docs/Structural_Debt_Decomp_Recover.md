DET v6.4 Module Card

Structural Debt Decomposition and Recovery

Status

Extension module (opt-in). Strict-core compliant under fixed substrate + inviolable agency + strict locality + closed update loop.

⸻

0. Purpose

Split retained past q into:
	•	Identity debt q^I: irreversible “carved history” (persists across dormancy / spirit mode)
	•	Damage debt q^D: recoverable “scar load” (can heal locally, at a cost, under agency + coherence gating)

Supports:
	•	biological-like healing without violating time’s arrow for identity
	•	“selective inheritance” resurrection without erasing the past (no illegal reversal)

⸻

1. New / Updated Primitives

All variables are node-local unless noted.

Node state (existing + new)
	•	F_i \in \mathbb{R}_{\ge 0} : resource
	•	q^I_i \in [0,1] : identity debt (irreversible)
	•	q^D_i \in [0,1] : damage debt (recoverable)
	•	q_i \equiv \text{clip}(q^I_i + q^D_i, 0, 1) : total retained past (for any coupling that expects scalar q)
	•	a_i \in [0,1] : intrinsic agency (constant; never modified)
	•	n_i \in \{0,1\} : participation (1=active, 0=dormant/spirit mode)
	•	o_i \in [0,1] : openness (optional but recommended). Updated only by node-local policy; boundaries cannot set it.
	•	H_i \ge 0 : coordination/load (strictly local readout)
	•	\Delta \tau_i \ge 0 : local proper-time step (0 in dormancy)

Bond state (existing)
	•	C_{ij} \in [0,1] : bond coherence for (i,j)\in E(t)

Local coherence aggregate (define explicitly)

Choose one canonical C_i (strictly local):

Default (smooth template):
C_i \equiv
\begin{cases}
\frac{1}{|\mathcal N(i)|}\sum_{j\in\mathcal N(i)} C_{ij} & |\mathcal N(i)|>0\\
0 & \text{else}
\end{cases}

(Alternate allowed: C_i=\max_{j\in\mathcal N(i)} C_{ij} if you want “one strong bond is enough.”)

⸻

2. Loss Measurement (local, per step)

Define local loss magnitude:
L_i \equiv \max(0,\,-\Delta F_i)
where \Delta F_i = F_i^{+} - F_i is computed from the normal DET flux/resource operators this step (local).

⸻

3. Identity vs Damage Split Law

3.1 Coherence/Consent factor (strictly local)

Define:
s_i \equiv \text{clip}\!\left(\frac{C_i \cdot g^{(a)}_i \cdot \tilde{o}_i}{1+H_i},\,0,\,1\right)
	•	g^{(a)}_i\in[0,1]: existing agency gate/readout (must be strictly local)
	•	\tilde{o}_i: openness factor:
	•	if using openness variable: \tilde{o}_i=o_i
	•	if not: \tilde{o}_i=1 (module still works)

Interpretation: loss under high coherence + agency-gate + openness, low load becomes “identity-forming”; otherwise it becomes “damage-like.”

3.2 Partitioned accumulation

\Delta q^I_i = \alpha_q \, s_i \, L_i
\Delta q^D_i = \alpha_q \, (1-s_i)\, L_i

3.3 Update (accumulation only)

q^{I'}_i = \text{clip}(q^I_i + \Delta q^I_i,\,0,\,1)
q^{D'}_i = \text{clip}(q^D_i + \Delta q^D_i,\,0,\,1)

⸻

4. Recovery Law (Healing)

Damage debt can be reduced by local repair work, gated by agency + coherence (+ openness if enabled), and paid for with resource.

4.1 Heal rate

h_i \equiv \eta_{\text{heal}} \cdot a_i \cdot C_i \cdot \tilde{o}_i \cdot \Delta\tau_i

4.2 Damage reduction (bounded)

q^{D+}_i = \text{clip}(q^{D'}_i - h_i,\,0,\,1)

4.3 Local cost (thermodynamic honesty)

Repair consumes resource locally:
F^{+}_i = \max\!\left(0,\;F^{'}_i - \kappa_{\text{heal}} \, h_i\right)
	•	F^{'}_i is the post-flux resource after the normal DET update operators but before this module’s heal cost.
	•	This ensures “healing” cannot be a free global eraser.

4.4 Identity irreversibility (hard constraint)

q^{I+}_i = q^{I'}_i \quad\text{(no decrement operator exists)}

⸻

5. Total Debt for Couplings

Whenever another module expects scalar q, use:
q_i^{+} = \text{clip}(q^{I+}_i + q^{D+}_i,\,0,\,1)

(If you want to keep baseline/gravity sourcing tied to “deep history,” you may optionally source gravity from q^I only, but that is a separate declared submodel. Default: use q=q^I+q^D.)

⸻

6. Spirit Mode and Dormancy Compatibility

Spirit/dormant mode is simply:
	•	n_i=0
	•	\Delta\tau_i=0

Then:
	•	no loss accumulation (because the node is not participating in flux)
	•	no healing (since h_i \propto \Delta\tau_i)
	•	q^I, q^D remain frozen without needing any “outside substrate” claim

⸻

7. Resurrection as Selective Recruitment

Resurrection is recruitment + bounded pattern seeding, not erasure.

7.1 Fixed substrate requirement

There exists a fixed adjacency E_0 (substrate neighborhood). Dynamic bonds E(t)\subseteq E_0.

Recruitment may only target dormant nodes k within a local neighborhood (e.g., k \in \mathcal N_R(i) measured on E_0).

7.2 Handshake (non-coercive)

A source identity at node i proposes to a dormant host k.

Offer condition (local):
\text{offer}_{i\to k} = \mathbb{1}[n_i=0]\cdot \mathbb{1}[k\in \mathcal N_R(i)]\cdot \mathbb{1}[F_k \ge F_{\min}]

Accept condition (host-local):
\text{accept}_k = \mathbb{1}[n_k=0]\cdot \mathbb{1}[a_k \ge a_{\min}^{join}] \cdot \mathbb{1}[\tilde{o}_k>0]

Only if \text{offer}\wedge \text{accept} do we instantiate the embodiment step.

7.3 Selective inheritance rule
	•	Identity migrates:
q^I_k \leftarrow \text{merge}_I(q^I_k,\, q^I_i)
Default merge (simple scalar): \text{merge}_I(x,y)=y (carry identity magnitude).
If you prefer “no overwrites,” use \max(x,y) or a convex mix.
	•	Damage does not migrate:
q^D_k \leftarrow q^D_k \quad (\text{unchanged; typically near baseline})
	•	Participation becomes active only after at least one lawful bond forms:
n_k: 0 \to 1 \quad \text{iff} \quad \exists (k,j)\in E(t+)\ \text{with}\ C_{kj}\ge C_{\text{init}}
	•	Old node i remains dormant and unchanged (no erasure).

This yields “fresh body, same identity” without violating irreversibility.

⸻

8. Canonical Update Ordering

This module assumes a single global simulation tick, but all computations are local.

For each tick:
	1.	Core DET flux/resource update
Compute F_i^{'}, C_{ij}^{'}, etc. (existing model)
	2.	Compute local loss
L_i = \max(0, -(F_i^{'} - F_i))
	3.	Compute local aggregates
C_i, H_i, g^{(a)}_i, \Delta\tau_i, and \tilde{o}_i
	4.	Debt split accumulation
Update q^{I'}_i, q^{D'}_i
	5.	Healing (damage only) + cost
Compute h_i, update q^{D+}_i, charge F_i^{+}
	6.	Finalize totals
q_i^{+}=\text{clip}(q^{I+}_i+q^{D+}_i,0,1)
	7.	(Optional) Resurrection operator
Apply recruitment handshakes and bond formation locally, using E_0 neighborhood only.

⸻

9. Parameters

Recommended defaults (safe, conservative):
	•	\alpha_q = 0.012 (existing q-locking rate)
	•	\eta_{\text{heal}} = 0.001 (damage healing per unit \Delta\tau)
	•	\kappa_{\text{heal}} = 1.0 (resource cost per heal unit; tune)
	•	a_{\min}^{join} = 0.1
	•	F_{\min} = small positive (avoid zero-resource activation)
	•	C_{\text{init}}=0.15 (if using new bond init coherence)
	•	Openness enabled: o_i initialized in [0,1] and updated only by node policy

⸻

10. Falsifiers

F_Q1 — Identity Reversal

If any operator reduces q^I_i below its historical maximum, the extension fails.

F_Q2 — Coerced Healing

If q^D_i decreases when a_i=0 or C_i=0 or (if enabled) o_i=0, the extension fails.

F_Q3 — Non-Local Resurrection

If identity migrates to a node k without k\in \mathcal N_R(i) on the fixed substrate adjacency E_0 (no local path), the extension fails.

F_Q4 — Free Repair

If q^D_i decreases without any compensating local cost (either F_i decreases by \kappa_{\text{heal}}h_i or an explicitly modeled local incoming flux pays that cost), the extension fails.

⸻

11. Minimal Test Harness Expectations
	1.	Split correctness:
Force identical losses under two regimes:
	•	high C_i, low H_i, high g^{(a)}_i → mostly \Delta q^I
	•	low C_i or high H_i → mostly \Delta q^D
	2.	Healing gate:
Verify no healing occurs when C_i=0 (or o_i=0 if enabled).
	3.	Healing cost:
Verify F_i decreases exactly with healing amount; no “free” q^D reduction.
	4.	Resurrection locality:
Attempt resurrection to a non-neighbor dormant node; must fail.
	5.	Identity irreversibility under resurrection:
Old node’s q^I unchanged; host receives identity per merge rule; no decrement anywhere.

⸻

12. One-Paragraph Interpretation Layer
	•	q^I is the irreversible “meaningful carved past” (identity-forming debt).
	•	q^D is the recoverable “scar load” that can be repaired only through coherent, agency-gated participation, and only by paying local work.
	•	Resurrection is lawful selective recruitment: identity migrates; damage does not; nothing is erased.

If you want, next I can also provide a compact “drop-in” pseudocode version matching your simulator style (node loop + edge loop) with no globals and a deterministic ordering.