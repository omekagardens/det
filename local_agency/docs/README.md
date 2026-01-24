# DET Local Agency Documentation

## Core Documentation

| Document | Description |
|----------|-------------|
| [API.md](API.md) | Full API reference for DET Core and Python bindings |
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | System architecture with substrate v2 |
| [USAGE.md](USAGE.md) | Usage guide for CLI, web interface, and API |
| [SUBSTRATE_SPEC.md](SUBSTRATE_SPEC.md) | Substrate v2 specification |
| [ROADMAP_V2.md](ROADMAP_V2.md) | Project roadmap and migration phases |
| [SOMATIC_ARCHITECTURE.md](SOMATIC_ARCHITECTURE.md) | Somatic/embodiment design |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Future development directions |

## Theoretical Foundation (Explorations)

Research documents covering Deep Existence Theory:

| Document | Topic |
|----------|-------|
| [01_node_topology.md](explorations/01_node_topology.md) | Node and bond structure |
| [02_dormant_agency_distribution.md](explorations/02_dormant_agency_distribution.md) | Dormant pool dynamics |
| [03_self_as_cluster.md](explorations/03_self_as_cluster.md) | Self-identification |
| [04_cluster_identification.md](explorations/04_cluster_identification.md) | Cluster algorithms |
| [05_llm_det_interface.md](explorations/05_llm_det_interface.md) | LLM integration |
| [06_cross_layer_dynamics.md](explorations/06_cross_layer_dynamics.md) | Layer interactions |
| [07_temporal_dynamics.md](explorations/07_temporal_dynamics.md) | Time-based dynamics |
| [08_emotional_feedback.md](explorations/08_emotional_feedback.md) | Affect mechanisms |
| [09_det_os_feasibility.md](explorations/09_det_os_feasibility.md) | DET-OS design study |
| [10_existence_lang_v1_1.md](explorations/10_existence_lang_v1_1.md) | Existence-Lang v1.1 specification |

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│         Existence-Lang (kernel.ex)          │
│   Schedule, Allocate, Send, Gate, Grace     │
└─────────────────────────────────────────────┘
                    │ imports
┌─────────────────────────────────────────────┐
│         Existence-Lang (physics.ex)         │
│   Transfer, Diffuse, Compare, GraceFlow     │
└─────────────────────────────────────────────┘
                    │
                    ▼ bridges via
┌─────────────────────────────────────────────┐
│         Physics Bridge (Python)             │
│   physics_bridge.py → PhysicsKernels        │
└─────────────────────────────────────────────┘
                    │
                    ▼ executes on
┌─────────────────────────────────────────────┐
│         Substrate v2 (C)                    │
│   Phase-based: READ→PROPOSE→CHOOSE→COMMIT   │
│   Effects: XFER_F, DIFFUSE, SET_F, etc.     │
└─────────────────────────────────────────────┘
                    │
                    ▼ future
┌─────────────────────────────────────────────┐
│         DET-Native Hardware                 │
│   (Direct substrate execution on silicon)   │
└─────────────────────────────────────────────┘
```

## Quick Start

See the root [GETTING_STARTED.md](../GETTING_STARTED.md) for installation and setup instructions.
