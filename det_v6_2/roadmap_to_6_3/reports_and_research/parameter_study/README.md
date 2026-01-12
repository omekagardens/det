# DET Study Package

This package contains all data, code, and reports produced for the Deep Existence Theory (DET) research tasks conducted in this project.

## Folder Structure

```
det_package/
├── README.md                # this file
├── 01_parameter_study/
│   └── param_report.md      # final report summarizing tunable parameters in DET v6.2
└── 02_gravitational_coherence/
    ├── gravitational_coherence_report.md   # combined report on gravitational time dilation and coherence dynamics
    ├── gravitational_redshift_analysis.py   # Python script computing gravitational redshift and GPS corrections
    ├── coherence_dynamics_simulation.py     # Python script simulating DET coherence dynamics
    ├── gravitational_redshift.png           # Plot of fractional frequency shift vs height
    └── coherence_dynamics.png               # Plot of coherence dynamics under flux
```

### 01_parameter_study
The `param_report.md` file contains a comprehensive survey of all tunable parameters in the DET v6.2 theory, classified into unit/scale setters, physical-law constants, and numerical/discretization parameters.  Each parameter is linked to its operational definition, measurement rig, and its role within the DET update loop.

### 02_gravitational_coherence
This folder contains analyses of two key validation domains for DET:

- **Gravitational time dilation** – a Python script (`gravitational_redshift_analysis.py`) computes the fractional frequency gradient near Earth and the combined general‑relativistic and special‑relativistic time dilation for GPS satellites.  The generated plot (`gravitational_redshift.png`) visualises how the frequency shift depends on height.
- **Coherence dynamics** – a Python script (`coherence_dynamics_simulation.py`) implements a simple DET coherence equation.  It simulates coherence growth under a constant flux and exponential decay after the flux is turned off.  The plot (`coherence_dynamics.png`) shows the time evolution of coherence and demonstrates that the simulated decay time matches experimental observations (~17 ms).

The `gravitational_coherence_report.md` file summarises these studies, provides citations to relevant literature and open data, and interprets the implications for DET parameters.