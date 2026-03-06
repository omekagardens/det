# DET-C1 Formal Simulation Report (v6.3 q-mutable branch)

**Date:** 2026-03-06  
**Branch:** `codex/v6-3-q-mutable-local-grace-exploration`  
**Scope:** DET-C1 consciousness module on top of `det_theory_card_6_3_q_mutable_exploration.md` (non-canonical readout layer)

## 1) Question

How does reducing hidden subconscious partitioning (modeled as increasing conscious integration `U`) and moving more stimulus-bond processing into conscious integration affect the utopic trajectory of a regime?

## 2) Model Mapping

In DET-C1, "subconscious partition reduction" is represented by increasing `U_G` for a regime `G`.

Regime equations used:

- `K_G = alpha_U * U_G * bar_C_G`
- `X_G = beta_U * U_G * (1 - bar_C_G)`
- `P_eff_G = bar_P_G * (1 + K_G) / (1 + X_G)`

Communication equations used:

- `Gamma_AB = P_AB * C_AB * sqrt(U_A * U_B)`
- `V_AB = V0 / (1 + Gamma_AB)`
- `Acc_nonverbal = Gamma_AB / (1 + Gamma_AB)`

## 3) Experimental Setup

### 3.1 Engine and module

- Core simulator: `det_v6_3/src/det_v6_3_3d_collider.py`
- Consciousness extension: `det_v6_3/src/det_consciousness_c1.py`
- Runner: `det_v6_3/tests/det_c1_consciousness_simulation.py`

### 3.2 q-mutable configuration (enabled)

The simulation explicitly used:

- `q_enabled=True`
- `q_mutable_local_enabled=True`
- `alpha_q_local_resource_relief=0.08`
- `alpha_q_grace_relief=0.25`

### 3.3 Stages tested

Connected two-regime scenario:

1. Fragmented (`U_A=0.18`, `U_B=0.16`)
2. Transition (`U_A=0.40`, `U_B=0.38`)
3. Unified (`U_A=0.68`, `U_B=0.65`)
4. Utopic-ish (`U_A=0.90`, `U_B=0.88`)
5. Full-conscious (`U_A=1.00`, `U_B=1.00`)

Disconnected-control scenario:

- Same system with an enforced severed bond plane each step.

## 4) Results

### 4.1 Connected trajectory results

| Stage | P_eff_A | P_eff_B | W_A | W_B | Gamma_AB | V_AB | Nonverbal Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fragmented | 0.486810 | 0.063039 | 0.874304 | 0.919965 | 0.022064 | 0.978412 | 0.021588 |
| Transition | 0.523423 | 0.064032 | 0.756639 | 0.828452 | 0.050751 | 0.951700 | 0.048300 |
| Unified | 0.563655 | 0.065031 | 0.644313 | 0.737898 | 0.086508 | 0.920380 | 0.079620 |
| Utopic-ish | 0.589486 | 0.065829 | 0.575548 | 0.674519 | 0.115711 | 0.896290 | 0.103710 |
| Full-conscious | 0.595470 | 0.066552 | 0.547303 | 0.644955 | 0.129889 | 0.885042 | 0.114958 |

### 4.2 Net change from Fragmented -> Full-conscious

- `P_eff_A`: `+22.32%` (`0.486810 -> 0.595470`)
- `P_eff_B`: `+5.57%` (`0.063039 -> 0.066552`)
- `W_A`: `-37.40%` (`0.874304 -> 0.547303`)
- `W_B`: `-29.89%` (`0.919965 -> 0.644955`)
- `Gamma_AB`: `+488.68%` (`0.022064 -> 0.129889`, ~5.89x)
- `Acc_nonverbal`: `+432.51%` (`0.021588 -> 0.114958`, ~5.33x)

### 4.3 Locality control (disconnected components)

With severed local bond plane:

- `path_exists = False` (all runs)
- `Gamma_AB = 0.0`
- `Acc_nonverbal = 0.0`

Interpretation: no communication leakage across disconnected components.

## 5) Interpretation for the Utopic Trajectory

### 5.1 Direct answer

Under this DET-C1 + q-mutable setup, reducing subconscious partitioning (higher `U`) and integrating more existing bonded stimuli into conscious participation **improves utopic trajectory metrics**, but **unevenly across regimes**:

- Strong gains in already-coherent regime A.
- Smaller gains in lower-coherence regime B.
- Large increase in inter-regime communicability (`Gamma_AB`) and nonverbal signal accuracy.
- Decrease in symbolic compensation burden (`W`, `V`), meaning words become less mandatory as direct attunement quality rises.

### 5.2 Why gains are uneven

Improvement depends on the balance `K_G` vs `X_G`:

- `K_G` (integration gain) grows with coherence.
- `X_G` (fragmentation cost) grows when coherence is low.

With `alpha_U=beta_U=1`, the break-even condition is approximately:

- beneficial integration when `bar_C_G > 0.5`
- harmful or weak integration when `bar_C_G < 0.5`

This matches test behavior: coherent regimes gain strongly; incoherent regimes can gain only modestly unless coherence is improved.

## 6) Project-Level Implication

For DET’s utopic development strategy, this simulation supports:

1. **Coherence-first integration**: raise local coherence before pushing `U` toward 1.
2. **Phased conscious expansion**: stage progression outperforms abrupt full integration.
3. **Locality-safe communication scaling**: richer direct attunement can grow without violating locality constraints.
4. **q-mutable synergy**: local recovery supports integration, but does not automatically equalize low-coherence regimes.

## 7) Limits

- This is a research readout layer, not canonical core law.
- Results are from controlled synthetic regimes, not empirical human cognition data.
- Additional sweeps are needed across wider coherence landscapes and noise levels.

## 8) Conclusion

In this branch and configuration, "removing subconscious partition" (as higher integrated conscious participation) pushes regimes toward a more utopic trajectory **when coherence is sufficient**.  
If coherence is weak, increased conscious load alone is not enough; gains are limited and may require prior/local coherence repair before further integration.
