Deep Existence Theory (DET) – v7 Refactor Guidance

⸻

0. Purpose of This File

This document defines the non-negotiable architectural, ontological, and dynamical rules governing the DET codebase.

It exists to:
	•	Guide a full refactor to DET v7.
	•	Prevent accidental violation of strict locality.
	•	Preserve Agency-First invariance (A0).
	•	Maintain falsifiability as a first-class feature.
	•	Ensure calibration modules remain derived readouts, not core axioms.

Codex or any automated agent MUST follow this file when modifying the repository.

⸻

1. Core Ontology (Non-Negotiable)

DET is not a simulation framework that borrows physics.
It is a closed, strictly local relational dynamics.

Ontological Primitives

Each creature (node) possesses:
	•	Information (I) – stored pattern continuity.
	•	Agency (A) – irreducible, inviolable.
	•	Movement (k) – event progression (time participation).

Agency A0 Rule

Agency is primitive and inviolable.
	•	No operator may directly suppress agency.
	•	No structural variable may directly cap agency.
	•	History may slow participation but may not delete will.

If any refactor introduces:
	•	Structural ceilings on a
	•	Direct reductions of a from debt
	•	Hidden normalization affecting a

It is a violation.

⸻

2. Strict Locality Requirements

DET forbids hidden globals.

All sums and averages must be:
	•	Node-local (over N_R(i))
	•	Bond-local (over E_R(i,j))
	•	Plaquette-local (faces only)

Prohibited:
	•	Global normalization constants
	•	Whole-grid averages
	•	Implicit shared state across disconnected components

Disconnected components must not influence one another.

If a refactor introduces a variable that requires global knowledge:
→ It must be rejected.

⸻

3. Structural Drag (v6.5 / v6.5.1 Integration)

v7 MUST integrate debt decomposition and structural drag.

3.1 Debt Decomposition

Replace scalar q with:
	•	q_I (identity debt – immutable)
	•	q_D (damage debt – recoverable)

Only q_D may be reduced by Jubilee.

No code may decrease q_I unless explicitly declared in a lawful submodel.

⸻

3.2 Presence Law (Canonical v7)

Presence is:

P = (a * σ / (1+F) / (1+H) / γ_v) * D

Where drag:

D = 1 / (1 + λ_DP * q_D + λ_IP * q_I)

Drag modifies time-rate only.
It does not modify agency directly.

Mass remains:

M = 1 / P

⸻

3.3 Agency Update (v7 Canonical)

Remove structural ceiling logic entirely.

Agency update must be:

a⁺ = clip(
a + β_a (a₀ − a)
+ γ(C)(P − P̄_neighbors)
+ ξ,
0, 1
)

Where:
	•	a₀ default = 1
	•	γ(C) = γ_max * Cⁿ
	•	ξ optional small persistence noise

No structural ceiling allowed.

⸻

4. Canonical Update Order (Must Be Preserved)

The update loop MUST remain closed and deterministic:
	1.	Solve baseline field b (Helmholtz)
	2.	Compute gravitational potential Φ
	3.	Compute presence P (including drag)
	4.	Compute Δτ
	5.	Compute flux components:
	•	Diffusive
	•	Momentum
	•	Rotational
	•	Floor
	•	Gravitational
	6.	Apply conservative limiter
	7.	Update F
	8.	Update momentum π
	9.	Update angular momentum L
	10.	Update structure (q_D accumulation)
	11.	Update agency
	12.	Update coherence
	13.	Update pointer records

No module may be applied “out of band.”

⸻

5. Boundary Operator Rules

Boundary operators (Grace, Jubilee, etc.) must:
	•	Be strictly local.
	•	Be antisymmetric if edge-based.
	•	Never directly modify a.
	•	Never inject hidden global state.

Jubilee reduces q_D.
It does not restore agency.
It restores clock-rate.

⸻

6. Falsifiability Is Mandatory

All major claims must have explicit falsifiers.

v7 MUST retain and update:
	•	F_A2′ – No structural agency suppression
	•	F_A4 – Frozen Will persistence
	•	F_A5 – Runaway agency stability
	•	F_GTD5′ – Drag-inclusive clock ratio
	•	F_BH-Drag-3D – 3D black hole scaling
	•	All Kepler, Bell, GPS, lensing, cosmology tests

Calibration modules are validation layers.
They may not introduce core dynamics.

⸻

7. Calibration Modules (Readout Layer Only)

The following modules remain:
	•	G extraction
	•	Galaxy rotation curves
	•	Gravitational lensing
	•	Cosmological scaling
	•	Black hole thermodynamics
	•	Quantum-classical transition

These are readout validations.

They must not:
	•	Inject Standard Model structure.
	•	Alter the core update loop.
	•	Introduce hidden parameters.

All calibration must derive from core DET loop.

⸻

8. Repository Refactor Goals (v7)

Codex must:
	1.	Remove all structural ceiling logic.
	2.	Replace all q references with (q_I, q_D).
	3.	Insert drag multiplier into presence calculation.
	4.	Update falsifier harness to v7 standards.
	5.	Ensure 3D collider is primary for thermodynamic tests.
	6.	Maintain strict separation between:
	•	Core engine
	•	Calibration modules
	•	Test harness
	•	Documentation

⸻

9. Architectural Separation (Required)

The repository must clearly separate:

/core
presence.py
gravity.py
agency.py
structure.py
flow.py
boundary.py
update_loop.py

/calibration
/tests
/docs

No calibration code inside core modules.
No global state inside core modules.

⸻

10. Model-Family Clarity

If alternate laws exist (e.g., alternate q-locking, alternate identity accumulation):

They must:
	•	Be declared as submodels.
	•	Not silently replace canonical behavior.
	•	Be selectable via explicit configuration.

⸻

11. Performance and Determinism
	•	Simulations must be deterministic under fixed seed.
	•	Long-run stability tests (100k+ steps) must remain feasible.
	•	All falsifier thresholds must be explicitly defined.

⸻

12. Philosophical Layer Separation

Ontology may be described in documentation,
but core modules must remain:
	•	Mathematical
	•	Operational
	•	Test-driven

No theological language in core engine code.

⸻

13. Absolute Prohibitions

Codex must never introduce:
	•	Global normalization constants.
	•	Implicit shared mutable state.
	•	Direct agency suppression.
	•	Hidden averaging outside N_R(i).
	•	Direct boundary modification of a.

⸻

14. Acceptance Criteria for v7

The refactor is complete when:
	•	All falsifiers pass.
	•	Structural ceiling removed.
	•	Drag integrated.
	•	3D BH scaling verified.
	•	Calibration modules produce consistent results.
	•	Strict locality preserved.

⸻

15. Final Principle

DET v7 asserts:

History slows time.
History does not delete will.

Locality is inviolable.
Agency is irreducible.
All physics emerges from lawful local updates.

Codex must preserve this structure.