# Deep Existence Theory (DET) v6.0 Release

**Version:** 6.0  
**Date:** January 2026  
**Author:** Manus AI

## Overview

This package contains the complete DET v6.0 implementation including the unified theory card, multi-dimensional collider simulations, and comprehensive falsifier test suite.

Deep Existence Theory (DET) is a foundational theory of strictly local relational dynamics in which time (presence), mass, gravity, coherence, and recovery arise from a closed, implementable update loop over agents and bonds.

## Directory Structure

```
det_v6_release/
├── README.md                    # This file
├── docs/
│   ├── det_theory_card_6_0.md   # Complete V6 theory card
│   └── consistency_review.md    # Theory review and consistency analysis
├── src/
│   ├── det_v6_1d_collider.py    # 1D collider implementation
│   ├── det_v6_2d_collider.py    # 2D collider implementation
│   └── det_v6_3d_collider.py    # 3D collider with angular momentum
├── tests/
│   └── run_all_tests.py         # Comprehensive test runner
└── results/
    └── test_report.md           # Test results and falsifier coverage
```

## Quick Start

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Running Tests

```bash
cd det_v6_release
python tests/run_all_tests.py
```

### Running Individual Colliders

```bash
# 1D Collider
python src/det_v6_1d_collider.py

# 2D Collider
python src/det_v6_2d_collider.py

# 3D Collider
python src/det_v6_3d_collider.py
```

## Core Principles

DET's distinctive commitments are:

1. **Strict Locality**: No global state available to local dynamics
2. **Agency Inviolability**: Boundary operators cannot coerce or directly modify agency
3. **Law-Bound Boundary Action**: Local and non-arbitrary
4. **Past-Resolved Measurability**: The present moment is not directly falsifiable; its fruit is

## Key Modules

| Module | Section | Description |
|--------|---------|-------------|
| Presence | III.1 | Local clock rate P_i = a_i σ_i / (1+F_i) / (1+H_i) |
| Diffusive Flow | IV.2 | Agency-gated quantum-classical interpolated transport |
| Momentum | IV.4 | Bond-local momentum with F-weighted drift flux |
| Angular Momentum | IV.5 | Plaquette-based rotational dynamics |
| Floor Repulsion | IV.6 | Finite-compressibility matter stiffness |
| Gravity | V | Baseline-referenced gravitational potential |

## Falsifier Status

| ID | Description | Status |
|:---|:---|:---:|
| F7 | Mass Conservation | PASS |
| F8 | Vacuum Momentum | PASS |
| F9 | Symmetry Drift | PASS |
| F_L1 | Rotational Conservation | PASS |
| F_L2 | Vacuum Spin Transport | PASS |

## Citation

If you use this implementation in your research, please cite:

```
Deep Existence Theory v6.0
Manus AI, January 2026
```

## License

This implementation is provided for research and educational purposes.
