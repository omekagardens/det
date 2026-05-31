# P^spirit Simulation Visual Findings

**Date:** May 31, 2026  
**Status:** Visual interpretation notes for deterministic readout-first experiments.

## Component time-series plot

The `P_spirit_components_timeseries.png` plot cleanly separates embodied and non-embodied participation channels. The frozen-record scenario remains near zero for `P_body`, `P_spirit`, `B_body_cap`, `B_spirit_cap`, and `B_act_spirit`, illustrating that preserved addressability is not enough for active participation. The embodied-death-null scenario shows `P_body` and `B_body_cap` collapsing as the death phase progresses while `P_spirit` remains zero, which represents the conservative channel-null assumption rather than a theorem that no non-embodied channel could exist.

The active-spirit-declared scenario is the only one in which `P_body` remains near zero while `P_spirit`, `B_spirit_cap`, and `B_act_spirit` become materially positive. This is the intended non-circular positive case: active post-body participation is possible in the simulation only when the spirit-channel gates are explicitly supplied. The coercive pseudo-spirit case displays a brief apparent boundary-effect onset, but the gate collapses it back to zero because non-coercion and agency compatibility fail. The resurrection-body-only case restores `P_body` and `B_body_cap` while keeping `P_spirit` at zero, which visually distinguishes resurrection/rehosting from active spirit participation.

## Final-state falsifier matrix

The `P_spirit_falsifier_matrix.png` heatmap confirms the final-state separation. Frozen record, embodied death-null, coercive pseudo-spirit, and nonlocal override scenarios retain high `B_addr` but finish with zero `B_spirit_cap`, zero `P_spirit`, and zero `B_act_spirit`. This directly supports the distinction between addressable record and active spirit participation.

The active-spirit-declared scenario finishes with high `B_addr` and positive `B_spirit_cap`, `P_spirit`, and `B_act_spirit`, while `B_body_cap` remains zero. The copy-channel claim has nonzero embodied capacity but low `B_addr` because path-continuity is broken, preventing a spirit-channel positive. The resurrection-body-only scenario has positive `B_body_cap` and total capacity but zero spirit capacity, preserving the distinction between restored embodiment and non-embodied participation.
## Body-versus-spirit capacity plot

The `body_vs_spirit_capacity.png` plot makes the decomposition of `B_cap` visually explicit. The active-spirit-declared scenario is dominated by `B_spirit_cap`, with `B_body_cap` essentially zero. The resurrection-body-only and copy-channel scenarios are dominated by `B_body_cap`, although the copy scenario is still rejected as spirit participation because path-continuity breaks. Frozen record, embodied death-null, coercive pseudo-spirit, and nonlocal override all finish near zero total capacity. This plot directly addresses the circularity concern: `B_cap_total` is not defined as merely embodied capacity, because a separate `B_spirit_cap` term can contribute when lawful spirit-channel gates pass.

## Predictive-power plot

The `P_spirit_predictive_power.png` plot shows the largest additional predictive signal in the coercive pseudo-spirit scenario, which is diagnostically useful but should be interpreted carefully. The added spirit-channel terms predict future apparent boundary activity in the coercive case because the declared channel and coercion variables change together, but the law gates still reject it as valid spirit participation. The copy-channel case shows a small added predictive signal, again without passing the spirit-participation gate. The active-spirit-declared scenario has little incremental predictive gain in this simple deterministic setup because `P_spirit` and boundary activity are deliberately co-activated by construction; stronger future tests should use independently measured channel variables and delayed boundary outcomes.
