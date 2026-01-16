# DET v6 Test 3.1: Newtonian Kernel (1/r Potential)

**Date:** January 2026  
**Status:** PASS

## Executive Summary

This test verifies that the DET gravity module produces a Newtonian 1/r potential in the far-field limit. The test was successful, demonstrating that:

1. **The Poisson solver correctly produces a 1/r far-field potential** with R² = 0.9994
2. **The monopole moment A scales linearly with total source Σq** with R² = 1.0000
3. **The Helmholtz baseline correctly removes the monopole** leaving Σρ = 0

## Test Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid Size | 64³ = 262,144 nodes | 3D cubic lattice |
| q Amplitude | 0.8 | Peak structural debt |
| q Width | 4.0 | Gaussian width (lattice units) |
| Helmholtz α | 0.05 | Baseline screening parameter |
| Gravity κ | 1.0 | Poisson coupling constant |
| Fit Range | r ∈ [8, 28] | Far-field fitting region |

## Test A: Raw q-Sourced Gravity

This test solves the Poisson equation with the raw structural debt q as source:

$$L_\sigma \Phi_{raw} = -\kappa q$$

### Results

| Metric | Value | Criterion | Status |
|--------|-------|-----------|--------|
| A (monopole) | 58.03 ± 0.23 | \|A\| > 0 | PASS |
| B (offset) | -2.27 ± 0.02 | - | - |
| R² | 0.9994 | > 0.95 | PASS |

The fit to Φ(r) = A/r + B is excellent, confirming that the discrete Laplacian/Poisson machinery correctly produces a 1/r far-field potential.

### Interpretation

The positive value of A indicates a repulsive potential (Φ increases near the source). This is expected for a Poisson equation with positive source. For attractive gravity (as in DET), the sign of κ or the source must be adjusted.

## Test B: Baseline-Referenced Gravity

This test implements the full DET gravity module:

1. Solve Helmholtz for baseline: $(L_\sigma - \alpha)b = -\alpha q$
2. Compute relative source: $\rho = q - b$
3. Solve Poisson: $L_\sigma \Phi_{ref} = -\kappa \rho$

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| Total ρ | 0.000000 | Monopole correctly removed |
| Φ_ref range | [-0.06, 4.82] | Much smaller than Φ_raw |

The baseline-referenced potential has a significantly reduced amplitude and does not follow a pure 1/r law. This is expected behavior: the Helmholtz baseline subtracts the monopole moment, leaving only higher-order multipole contributions.

## Scaling Test: A ∝ Σq

To verify that the monopole moment scales correctly with total mass, we ran the test with varying q amplitudes:

| q Amplitude | Total Σq | Monopole A | R² |
|-------------|----------|------------|-----|
| 0.2 | 201.60 | 13.05 | 0.9964 |
| 0.4 | 403.19 | 26.10 | 0.9964 |
| 0.6 | 604.79 | 39.15 | 0.9964 |
| 0.8 | 806.38 | 52.20 | 0.9964 |
| 1.0 | 1007.98 | 65.25 | 0.9964 |

**Linear Fit:** A = 0.0647 × Σq + 0.0000  
**R² (linear):** 1.000000

The monopole moment scales perfectly linearly with total source, as expected for a Newtonian potential.

## Visualizations

The test generated a comprehensive 6-panel visualization showing:

1. **Source Fields:** Radial profiles of q (structural debt), b (baseline), and ρ (relative source)
2. **Raw Potential:** Φ_raw(r) with 1/r fit overlay
3. **Baseline-Referenced Potential:** Φ_ref(r) showing screened behavior
4. **Φ × r Test:** Should be constant for 1/r; shows excellent agreement in far-field
5. **Log-Log Plot:** Slope of -1 confirms 1/r power law
6. **Comparison:** Direct comparison of raw vs baseline-referenced potentials

## Conclusions

1. **Newtonian Kernel Verified:** The DET Poisson solver correctly produces a 1/r far-field potential with excellent fit quality (R² > 0.999).

2. **Scaling Verified:** The monopole moment A scales perfectly linearly with total source Σq (R² = 1.0000).

3. **Baseline Mechanism Confirmed:** The Helmholtz baseline correctly removes the monopole moment from the source, leaving Σρ = 0.

4. **No Falsification:** The test does not falsify DET gravity (G.8.2). The 1/r kernel is correctly implemented.

## Implications for DET

This test confirms that the mathematical machinery for DET gravity is sound. The next steps are:

1. **Implement in Colliders:** Add the gravity module to the 2D and 3D colliders
2. **Test Binding Dynamics:** Verify that two massive bodies form stable bound states (F6)
3. **Extract Effective G:** Calibrate the coupling constant κ against known gravitational phenomena

## Files

- `test_newtonian_kernel_fft.py` - Test implementation
- `newtonian_kernel_fft.png` - Visualization
- `newtonian_kernel_report.md` - This report
