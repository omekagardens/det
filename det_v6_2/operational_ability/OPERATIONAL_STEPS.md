# DET v6.2 Operational Ability Steps

This document outlines the steps to operate the DET v6.2 simulation environment and verify its physical proofs.

## Step 1: Environment Setup
Ensure you have the necessary dependencies installed:
```bash
pip install numpy scipy matplotlib
```

## Step 2: Running the Core Colliders
The colliders demonstrate the fundamental dynamics of DET.
- **1D Collider**: Tests basic collision and mass conservation.
- **2D Collider**: Tests symmetry and momentum in a plane.
- **3D Collider**: Tests angular momentum and rotational stability.

```bash
python src/det_v6_2_1d_collider.py
python src/det_v6_2_2d_collider.py
python src/det_v6_2_3d_collider.py
```

## Step 3: Verification of Physical Proofs
Run the test suite to verify that the current version passes the core falsifiers.
```bash
python tests/run_all_tests.py
```

## Step 4: Quantum Emergence Analysis
Run the quantum proper script to verify topological quantization and tunneling.
```bash
python src/det_quantum_proper.py
```

## Step 5: Gravitational Diagnostic
Use the Cavendish simulation to check the gravitational force law.
```bash
python tests/test_gravity_comprehensive.py
```

## Step 6: Result Interpretation
Check the `results/` folder for generated plots and reports. Key metrics to watch:
- **Mass Drift**: Should be $< 10^{-10}$.
- **COM Drift**: Should be minimal in vacuum.
- **Phase Winding**: Should be integer multiples of $2\pi$.
