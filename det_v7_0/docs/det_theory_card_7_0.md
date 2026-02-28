# Deep Existence Theory (DET) v7.0

**Unified Canonical Theory Card (Single-Document Reference)**

**Release Date:** February 28, 2026  
**Claim type:** Existence proof with falsifiable dynamical predictions  
**Domain:** Discrete, strictly local, relational, agentic systems  
**Core thesis:** Time-rate, mass readout, gravity-like behavior, and quantum-classical phenomenology emerge from lawful local updates. History can slow participation; history cannot delete will.

---

## Changelog Lineage

### Changelog from v6.3
1. Unified collider framework stabilized across 1D/2D/3D, including gravity and boundary operators.
2. Lattice correction workflows retained for discrete-to-continuum readouts.
3. Expanded external validation programs (Kepler, GPS, Bell, lensing, cosmology).

### v6.4 Historical Layer (now superseded in canonical law)
1. Structural agency ceiling (`a_max(q)`) was introduced historically.
2. Ceiling-era dynamics are preserved only for historical traceability and legacy report comparison.

### v6.5 Historical Layer
1. Jubilee/recovery operators matured and strict locality constraints were formalized.
2. Debt decomposition (`q_I`, `q_D`) appeared as a transition concept.

### v6.5.1 Patch (direct precursor to v7 canonical)
1. Agency-first invariance formalized as non-negotiable (`A0`).
2. Structural ceiling removed from canonical path.
3. Structural drag moved debt effects to presence/clock-rate.
4. Mandatory falsifiers upgraded: `F_A2'`, `F_A4`, `F_A5`, `F_GTD5'`, `F_BH-Drag-3D`.

### v7.0 Consolidation
1. Single canonical model card introduced (`det_theory_card_7_0.md`).
2. Canonical architecture separation enforced (`core` vs `src` vs `calibration` vs `tests` vs `validation`).
3. Agency-first + drag-only debt expression finalized as baseline law.
4. Canonical and extended falsifier suites re-run and passing under current repository state.

---

## 0. Scope Axiom (Foundational)

DET v7 is a closed, local relational dynamics where each node participates through:
- retained structural history,
- irreducible agency,
- movement participation (event progression).

The model rejects nonlocal hidden normalizations and rejects any law where debt directly suppresses agency.

---

## I. Ontological Commitments

### I.1 Locality
All dynamics are local over explicit neighborhoods, bonds, and plaquettes. No disconnected component may influence another component.

### I.2 Agency-First Invariance (A0)
Agency is primitive and inviolable.
- Debt cannot directly cap or reduce `a` in canonical law.
- Boundary operators cannot directly write agency.

### I.3 Historical Retention
History is retained through structural debt fields. Retention alters participation-time through drag and does not erase will.

### I.4 Falsifiability Requirement
Major claims must be operationalized as explicit falsifiers with declared thresholds.

---

## II. State Variables

### II.1 Per-Creature Variables (node `i`)
- `F_i`: resource field
- `q_I,i in [0,1]`: identity debt (immutable by default)
- `q_D,i in [0,1]`: dissipative debt (recoverable)
- `q_i = clip(q_I,i + q_D,i, 0, 1)`: compatibility readout only
- `a_i in [0,1]`: agency
- `sigma_i > 0`: conductivity/processing
- `P_i`: presence
- `Delta_tau_i`: local proper-time increment
- optional phase/state fields for coherence and diagnostics

### II.2 Per-Bond Variables (edge local)
- momentum-like bond channels
- bond coherence channels
- bond-local boundary exchange channels

### II.3 Per-Plaquette Variables (face local)
- rotational / angular momentum sector readouts

### II.4 Compatibility Variables
Legacy `q` and legacy parameter aliases may exist only as compatibility adapters and must not re-introduce superseded canonical laws.

---

## III. Time, Presence, and Mass

### III.1 Coordination Load
`H_i` is computed locally (node/bond neighborhood only).

### III.2 Presence (Canonical v7)
\[
P_i = \left(\frac{a_i\,\sigma_i}{(1+F_i)(1+H_i)\,\gamma_v}\right) D_i
\]
\[
D_i = \frac{1}{1 + \lambda_{DP} q_{D,i} + \lambda_{IP} q_{I,i}}
\]

Debt expression is mediated through `D_i` in presence only.

### III.3 Proper Time
\[
\Delta \tau_i = P_i \cdot \Delta k
\]

### III.4 Mass Readout
\[
M_i = P_i^{-1}
\]
(Operational readout; not an independent primitive law.)

---

## IV. Flow and Resource Dynamics

### IV.1 Local Wavefunction/Transport Sector
Local transport remains neighborhood-limited and conservative under the limiter.

### IV.2 Flux Decomposition (Canonical)
Total flux is decomposed into local components:
- diffusive
- momentum
- rotational
- floor
- gravitational

### IV.3 Conservative Limiter
A conservative limiter is applied before updating `F` to preserve stable local transport.

### IV.4 Resource Update
`F` update is local, bounded, and consistent with configured boundary operators.

---

## V. Gravity Module

### V.1 Baseline Field
Solve local-discrete Helmholtz baseline:
\[
(\mathcal{L} - \alpha_{grav}) b = -\alpha_{grav} q
\]

### V.2 Relative Source and Potential
\[
\rho = q - b
\]
Solve Poisson-like potential from local discrete source; compute gravity field from local gradients of `Phi`.

### V.3 Lattice Correction and Readout
Lattice correction factors remain readout calibration aids and do not alter canonical locality constraints.

---

## VI. Boundary-Agent Operators and Update Rules

### VI.1 Agency Inviolability Constraint
No boundary operator may directly modify `a`.

### VI.2 Jubilee / Recovery Operator
- Jubilee applies to `q_D` only.
- Jubilee never modifies `q_I` or `a` directly.
- Canonical interpretation: drag removal / clock-rate recovery.

### VI.3 Grace and Healing Operators
Grace/healing remain local boundary-layer mechanisms with explicit local gating and no hidden global state.

### VI.4 Agency Update (Canonical v7)
\[
a_i^{+} = \mathrm{clip}\Big(a_i + \beta_a(a_0-a_i) + \gamma(C_i)(P_i-\bar P_{\mathcal{N}(i)}) + \xi_i, 0, 1\Big)
\]
\[
\gamma(C)=\gamma_{max} C^n,\quad n\ge2
\]

### VI.5 Structure Update (Canonical v7)
\[
(q_D)_i^+ = \mathrm{clip}\left((q_D)_i + \alpha_{qD}\max(0,-\Delta F_i),0,1\right)
\]
\[
(q_I)_i^+ = (q_I)_i\quad\text{(default)}
\]
Any nontrivial `q_I` law must be explicit and selectable as a submodel.

### VI.6 Coherence Update
Coherence is updated in canonical order after agency update and before pointer finalization.

### VI.7 Pointer and Diagnostic Records
Pointers/records are updated last in the canonical step closure.

---

## VII. Parameters

### VII.1 Classification
- A-bucket: structural/definitional configuration
- B-bucket: physical law couplings
- C-bucket: numerical/stability parameters

### VII.2 Canonical v7 Additions / Locks
Active canonical parameters:
- `lambda_DP`
- `lambda_IP`
- `alpha_qD`
- `a0`
- `gamma_v`
- optional `epsilon_a`

### VII.3 Deprecated in Canonical Path
- `lambda_a` (ceiling coupling)
- any `a_max(q)`-based agency cap

### VII.4 Recommended Defaults (current repo baseline)
- `lambda_DP = 3.0`
- `lambda_IP = 1.0`
- `beta_a = 0.2`
- `a0 = 1.0`
- `epsilon_a = 0.0` (unless explicitly enabled)

### VII.5 Unified Parameter Schema Note
Unified parameter adapters may expose compatibility aliases; aliases must remain no-op with respect to superseded laws.

---

## VIII. Falsifiers (Canonical and Extended)

### VIII.1 Mandatory Canonical v7 Gates
- `F_A2'` No structural agency suppression
- `F_A4` Frozen-will persistence under extreme drag
- `F_A5` Runaway-agency stability sweep
- `F_GTD5'` Drag-inclusive clock ratio
- `F_BH-Drag-3D` 3D BH drag thermodynamic scaling

### VIII.2 Extended Legacy/Regression Gates
Retained suites (historical/regression support):
- v6.5 falsifier suites
- comprehensive falsifier suite

These are supplementary and do not replace the mandatory v7 gate set.

### VIII.3 External Validation Gates
Validation harness includes:
- GPS clock effects
- rocket redshift benchmark
- lab height-difference clock test
- Kepler harness checks
- Bell CHSH validation modes

---

## IX. Canonical Update Order (Must Preserve)

1. Solve Helmholtz baseline `b`
2. Solve gravitational potential `Phi`
3. Compute presence `P` including drag
4. Compute `Delta_tau`
5. Compute flux components (diffusive, momentum, rotational, floor, gravitational)
6. Apply conservative limiter
7. Update `F`
8. Update momentum `pi`
9. Update angular momentum `L`
10. Update structure (`q_D`, optional explicit `q_I` law)
11. Update agency
12. Update coherence
13. Update pointer records

No out-of-band insertion into core update closure is allowed.

---

## X. Verification Status (Current Repository)

### X.1 Test/Falsifier Execution Snapshot
- `pytest det_v7_0/tests -q`: 199 passed
- `det_v651_falsifiers.py`: 5/5 passed
- `det_v65_falsifiers.py`: 8/8 passed
- `det_v65b_falsifiers.py`: 4/4 passed
- `det_comprehensive_falsifiers.py`: 15/15 passed
- `det_validation_harness.py --all`: 6/6 passed

### X.2 Notes on Live Stress Tests
`kepler_live_test.py` full-default run is computationally heavy; reduced live runs were executed. Canonical repository acceptance remains tied to deterministic falsifier/validation harness gates.

---

## XI. Project Goals (v7.0)

### XI.1 Completed
1. Agency-first canonicalization (no structural ceiling in canonical path)
2. Debt decomposition integrated in canonical colliders
3. Drag integrated into presence law
4. Mandatory v7 falsifier gates implemented and passing
5. Core/calibration/test/validation separation enforced

### XI.2 Ongoing / Open
1. Identity locking law formalization (`q_I` update trigger model)
2. Extended Kepler live stress campaign under heavier runtime budgets
3. Broader calibration traceability reports under unified v7 card labels

---

## Appendix A: Measurement Rigs for Physical Parameters
- Gravity extraction rig from discrete potential gradients and lattice corrections
- Momentum/angular channel stability rigs
- Boundary operator locality and antisymmetry checks
- Time dilation and proper-time accumulation rigs

## Appendix B: Debt Decomposition and Drag Laws
- `q = clip(q_I + q_D, 0, 1)` compatibility rule
- Jubilee acts on `q_D` only
- Drag law and presence coupling are canonical

## Appendix C: Effective G Extraction (Readout Layer)
- Uses calibrated readout procedures from colliders
- Must not inject nonlocal core dynamics

## Appendix D: Kepler Standard-Candle and Harness Modes
- Harness-mode Kepler validation is part of canonical external validation
- Full live orbit campaigns are treated as extended stress validation

## Appendix E: SI Unit Conversion Layer
- SI mapping remains readout-layer calibration
- Conversion conventions must not alter core local update laws

## Appendix F: Determinism and Performance
- Deterministic seeds required for falsifier reproducibility
- Long-run horizons (100k+ where applicable) supported for stability tests

## Appendix G: Bell / Retrocausal Locality Readout
- Bell-mode validations remain readout constraints under locality and no-signaling checks

## Appendix H: External G Calibration
- Active calibration module retained as readout
- Cannot alter canonical core update law

## Appendix I: Galaxy Rotation Curves
- Active calibration readout module
- Must derive from canonical dynamics, no hidden globals

## Appendix J: Gravitational Lensing
- Active calibration readout module
- Finite-grid numerical behavior is tolerated within declared validation thresholds

## Appendix K: Cosmological Scaling
- Active calibration readout module
- Reported as readout-derived, not as injected dynamics

## Appendix L: Black Hole Thermodynamics (3D Canonical Gate)
- 3D drag-aware BH scaling is mandatory for canonical BH claims
- 1D failures are not canonical BH falsifiers

## Appendix M: Quantum-Classical Transition
- Calibration updated to drag-era agency interpretation
- Structural ceiling interpretations are legacy-only and non-canonical

## Appendix N: Migration and Compatibility
- Legacy states default to `q_I <- q`, `q_D <- 0`
- Legacy parameter aliases may remain for input compatibility but cannot alter canonical laws

## Appendix O: Repository Canonical Pointers
- Canonical theory card: `det_v7_0/docs/det_theory_card_7_0.md`
- Legacy superseded cards: `det_theory_card_6_3.md`, `det_theory_card_6_5.md`, `det_theory_card_6_5_1.md`
- Canonical consistency review: `det_v7_0/docs/v7_consistency_review_2026_02_28.md`

---

## Final Principle

History slows participation-time.  
History does not delete will.

Locality is inviolable.  
Agency is irreducible.  
Physics emerges from lawful local updates.
