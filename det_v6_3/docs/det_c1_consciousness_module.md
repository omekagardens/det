# DET-C1 (v6.3 branch): Integrated Conscious Regime & Bond-Aware Communication

## Status

Research extension module (non-canonical) on top of:

- `det_theory_card_6_3.md`
- `det_theory_card_6_3_q_mutable_exploration.md`

Core collider laws are unchanged. DET-C1 is readout-only.

## Purpose

Model consciousness as an emergent integrated regime over a local bonded graph,
not as a primitive scalar.

## Regime Readouts

For regime `G` with integration level `U_G`:

- `K_G = alpha_U * U_G * bar_C_G`
- `X_G = beta_U * U_G * (1 - bar_C_G)`
- `P_eff_G ~ bar_P_G * (1 + K_G) / (1 + X_G)`

Where:

- `bar_P_G` is local mean presence over regime nodes.
- `bar_C_G` is local mean coherence over regime nodes.

Also tracked:

- `W_G = V0 / (1 + U_G * bar_C_G)` (symbolic compensation burden)
- `R_G = U_G * bar_C_G` (direct relational bandwidth proxy)

## Path Communication Readout

Between regimes `A` and `B` over strictly local neighbor paths:

- `Gamma_AB = P_AB * C_AB * sqrt(U_A * U_B)`
- `V_AB = V0 / (1 + Gamma_AB)`

Path constraints are local:

- 6-neighbor traversal only.
- Optional thresholds on bond coherence and bond presence.
- No path means `Gamma_AB = 0`.

## Falsifier-Style Tests

Implemented in `tests/test_consciousness_c1.py`:

- `F_C1`: integration benefit in coherent regimes.
- `F_C2`: verbal dependence decline with rising `Gamma`.
- `F_C3`: nonverbal readout gain with rising path coherence.
- `F_C4`: no nonlocal leakage across disconnected components.
- `F_C5`: agency invariance (readout module never writes `a`).

The fixture is explicitly configured with mutable `q` enabled so this module is
evaluated on the q-mutable exploration branch behavior.
