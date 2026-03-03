# Grace-Jubilee Energy Interaction Study

Generated: 2026-03-03T11:46:23.274895

## Scenario Summary

All runs use the same starved initial state (`F=0.01`, `q=0.75`, `a=0.95`, `C=0.92`) and differ by Grace toggle, energy-coupling toggle, and dissipation strength.

## Results

| Scenario | Grace | Energy Coupling | Mom Amp | q_drop | mean F_op | total Grace | total Jubilee |
|---|---:|---:|---:|---:|---:|---:|---:|
| starved-lowD | off | on | 1.0 | -0.000005 | 0.000000 | 0.000000 | 0.000000 |
| starved-lowD | off | off | 1.0 | 0.000792 | 0.000000 | 0.000000 | 0.143446 |
| starved-lowD | on | on | 1.0 | 0.000045 | 0.003503 | 0.783130 | 0.009136 |
| starved-lowD | on | off | 1.0 | 0.001016 | 0.003511 | 0.783336 | 0.184041 |
| starved-highD | off | on | 2.0 | 0.000237 | 0.007891 | 0.000000 | 0.044211 |
| starved-highD | off | off | 2.0 | 0.001080 | 0.007904 | 0.000000 | 0.196020 |
| starved-highD | on | on | 2.0 | 0.000374 | 0.015849 | 1.034365 | 0.069562 |
| starved-highD | on | off | 2.0 | 0.001412 | 0.015866 | 1.034638 | 0.256250 |

## Key Findings

1. With energy coupling ON and Grace OFF in the low-dissipation regime, Jubilee is effectively blocked (`total Jubilee ~ 0`) because `F_op ~ 0`.
2. Turning Grace ON raises `F` and `F_op`, enabling nonzero Jubilee under the same energy-coupled law.
3. With energy coupling OFF, Jubilee proceeds even when `F_op ~ 0`, confirming the cap is the gating mechanism.
4. In higher-dissipation regimes, Grace still increases both `F_op` and total Jubilee, but no longer acts as a hard on/off unlock.

Interpretation: Grace and Jubilee are coupled through local resource availability. Grace is not direct forgiveness; it shifts local budgets so energy-coupled Jubilee can lawfully act.
