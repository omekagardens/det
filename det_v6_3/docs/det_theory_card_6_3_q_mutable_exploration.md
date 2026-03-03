# DET v6.3 Exploration: Mutable-q via Local Resource/Grace

## Scope

This note defines an **exploration branch variant** based on the v6.3 theory card (`det_theory_card_6_3.md`), focused specifically on making structural debt `q` mutable through strictly local channels.

- v6.5 constructs are intentionally out of scope.
- Canonical v6.3 remains the default behavior.
- This variant is opt-in through collider parameters.

## Motivation

In canonical v6.3, the default q-locking law is:

\[
q_i^+ = \mathrm{clip}(q_i + \alpha_q\max(0,-\Delta F_i),0,1)
\]

This law only accumulates debt from local loss. The exploration asks: can debt also be reduced without violating locality, by allowing local gain pathways?

## Exploration Law (Opt-in)

For the 3D collider only, when `q_mutable_local_enabled=True`:

\[
q_i^+ = \mathrm{clip}(q_i + q_{\text{lock},i} - q_{\text{relief,local},i} - q_{\text{relief,grace},i}, 0, 1)
\]

with:

- `q_lock = alpha_q * max(0, -dF)` (canonical debt accumulation)
- `q_relief_local = alpha_q_local_resource_relief * max(0, dF)`
- `q_relief_grace = alpha_q_grace_relief * I_g`

where `dF` is the local resource change from conservative transport and `I_g` is local grace injection already computed in the boundary operator.

## Locality and Non-Coercion Guardrails

This exploration preserves the following:

1. **No global normalization added** (all terms are point-local).
2. **No direct agency edit** (`a` remains updated by existing agency law only).
3. **Boundary use remains local** (grace term uses already-local `I_g`).
4. **Default behavior unchanged** when `q_mutable_local_enabled=False`.

## New Collider Parameters

Added to `DETParams3D`:

- `q_mutable_local_enabled: bool = False`
- `alpha_q_local_resource_relief: float = 0.0`
- `alpha_q_grace_relief: float = 0.0`

Diagnostic fields are also tracked:

- `last_q_locking`
- `last_q_relief_local`
- `last_q_relief_grace`

## Exploration Test Coverage

New focused tests are added in `tests/test_q_mutable_local_grace.py`:

1. Baseline immutability check (default mode).
2. Mutable-q reduction check using local inflow + grace.
3. Outcome divergence vs immutable control in the same scenario.

## Interpretation

This branch does **not** claim canonical replacement. It provides an explicit, falsifiable submodel for assessing whether local restoration channels improve behavior while respecting v6.3 locality constraints.
