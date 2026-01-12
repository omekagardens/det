# Investigation: The ~0.96 Lattice Correction Factor in DET

## 1. The Observation

A dimensionless factor of approximately **η ≈ 0.9679** (on a 64³ lattice) appears in multiple independent DET contexts:

1. **Gravitational constant extraction:** When extracting G from lattice simulations, a correction factor of ~0.96 is needed to match continuum predictions

2. **Electromagnetic tuning:** The same factor appears when calibrating EM parameters

**Central Question:** Is this coincidence, fundamental, or an artifact?

---

## 2. Theoretical Background: Lattice vs. Continuum

### 2.1 The Core Issue

In any discrete lattice theory, the **discrete Laplacian** differs from the **continuum Laplacian**:

**Continuum (3D):**
$$\nabla^2 G(\mathbf{r}) = -\delta^{(3)}(\mathbf{r}) \implies G(\mathbf{r}) = -\frac{1}{4\pi|\mathbf{r}|}$$

**Discrete (simple cubic):**
$$(L_\sigma \Phi)_i = \sum_{j \in \mathcal{N}(i)} \sigma_{ij}(\Phi_j - \Phi_i)$$

The lattice Green's function (LGF) does **not** exactly equal 1/(4πr) even at large distances. There is a **lattice correction factor**.

### 2.2 Watson Integrals

The fundamental quantity for 3D cubic lattices is the **Watson integral W₃**:

$$W_3 = \frac{1}{\pi^3} \int_0^\pi \int_0^\pi \int_0^\pi \frac{d\theta_1 d\theta_2 d\theta_3}{3 - \cos\theta_1 - \cos\theta_2 - \cos\theta_3}$$

**Known exact value:**
$$W_3 = \frac{\sqrt{6}}{32\pi^3} \Gamma\left(\frac{1}{24}\right) \Gamma\left(\frac{5}{24}\right) \Gamma\left(\frac{7}{24}\right) \Gamma\left(\frac{11}{24}\right) \approx 0.505462...$$

This is related to the lattice Green's function at the origin.

### 2.3 Far-Field Behavior

For large separations |n| → ∞ on a simple cubic lattice:
$$G(n_1, n_2, n_3) \approx \frac{1}{2\pi \cdot 6 |n|} + O(|n|^{-3})$$

The factor of 6 comes from the coordination number of the simple cubic lattice.

Comparing to the continuum G = 1/(4πr):
- Lattice: 1/(12π|n|)
- Continuum: 1/(4πr)

The ratio depends on how lattice spacing maps to continuum distance.

---

## 3. Renormalization in 2D (Instructive Example)

Mamode (2021) provides the clearest treatment for the 2D case:

**Key Finding:** The discrete planar Laplacian's Green function requires a regularization constant ⟨g⟩. In the continuum limit (lattice spacing a → 0), one must perform "appropriate renormalization of ⟨g⟩" to recover the logarithmic Coulomb potential.

**Physical Interpretation:** The self-energy (value at origin) of a point source is divergent in the continuum but regularized (finite) on a lattice. The difference between these contributes to the correction factor.

---

## 4. Possible Sources of the 0.96 Factor

### 4.1 Hypothesis A: Geometric Lattice Factor

The factor could arise from the ratio of:
- Discrete Laplacian eigenvalue spectrum
- Continuum Laplacian eigenvalue spectrum

For a simple cubic lattice with coordination number z = 6:
$$\eta_{\text{geom}} = \frac{\text{effective lattice coupling}}{\text{continuum expectation}}$$

### 4.2 Hypothesis B: Finite-Size Effects

On a finite 64³ lattice:
- Periodic boundary conditions modify the Green's function
- Edge effects create systematic corrections
- The correction should scale as N^(-1/3) or similar

**Test:** Compute η for 32³, 64³, 128³ lattices and check scaling

### 4.3 Hypothesis C: Regularization Constant

The LGF requires removing a divergent self-energy term:
$$G_{\text{regularized}} = G_{\text{raw}} - G(0,0,0)$$

The specific value of G(0,0,0) = W₃/6 ≈ 0.0842... may contribute to the correction.

### 4.4 Hypothesis D: Dispersion Relation Correction

The lattice dispersion relation for the discrete Laplacian is:
$$\omega^2(k) = 4\sum_{i=1}^{3} \sin^2(k_i/2)$$

vs. continuum:
$$\omega^2(k) = k^2$$

At k = 0 they match, but at finite k they differ. This affects propagators and Green's functions.

---

## 5. Numerical Investigation Plan

### 5.1 Direct Calculation

**Step 1:** Compute the LGF for simple cubic lattice at various distances:
```python
import numpy as np
from scipy.fft import fftn, ifftn

def lattice_greens_function_3d(N):
    """Compute 3D LGF on NxNxN periodic lattice"""
    # Eigenvalues of discrete Laplacian
    k = np.fft.fftfreq(N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    eigenvals = 6 - 2*(np.cos(kx) + np.cos(ky) + np.cos(kz))
    eigenvals[0,0,0] = 1  # Regularize zero mode
    
    # Green's function in k-space
    G_k = 1.0 / eigenvals
    G_k[0,0,0] = 0  # Zero mode regularization
    
    # Transform to real space
    G_r = np.real(ifftn(G_k))
    return G_r
```

**Step 2:** Compare LGF to 1/(4πr) at various r
**Step 3:** Extract correction factor η(r) = G_lattice(r) / G_continuum(r)
**Step 4:** Check convergence as N → ∞

### 5.2 Analytic Verification

Use known exact results:
- Watson integral W₃ ≈ 0.5055
- G(1,0,0) = (W₃ - 1)/6 ≈ -0.0824 (lattice units)
- Compare to continuum G(1) = 1/(4π) ≈ 0.0796

Ratio: 0.0824/0.0796 ≈ 1.035... 

This suggests the correction factor might be closer to 1/1.035 ≈ 0.966, which is suspiciously close to the observed 0.9679!

### 5.3 Scale Dependence

Compute η at multiple length scales:
- Near field (r = 1-5 lattice units)
- Mid field (r = 10-50 lattice units)
- Far field (r > 100 lattice units)

If η converges to a constant, it's a fundamental lattice property.
If η varies with r, it's a scale-dependent artifact.

---

## 6. Connection to DET

### 6.1 Why It Matters

If η is a fundamental lattice constant:
- It should be **derived**, not tuned
- It should apply universally to all DET potentials (gravity, EM, etc.)
- It represents the discrete-to-continuum mapping correction

### 6.2 Implication for Parameter Extraction

When extracting physical constants from DET simulations:
$$G_{\text{physical}} = \eta \cdot G_{\text{lattice}}$$

This is analogous to:
- Madelung constant corrections in crystal physics
- Lattice QCD scale setting (a ↔ GeV conversion)
- GPS clock pre-adjustment for relativistic effects

### 6.3 Prediction

If the 0.96 factor is indeed the lattice renormalization constant:

1. It should be **the same** for both gravity and EM (since both use the same discrete Laplacian)

2. It should be **computable** from Watson integral / LGF theory

3. It should **approach 1.0** as the lattice becomes finer (continuum limit)

---

## 7. Literature to Consult

### Primary Sources

1. **Watson (1939):** Original evaluation of cubic lattice integrals
2. **Joyce & Zucker (2001):** "Evaluation of the Watson integral and associated logarithmic integral for the d-dimensional hypercubic lattice"
3. **Glasser & Zucker (1977):** "Extended Watson integrals for the cubic lattices," PNAS 74:1800
4. **Mamode (2021):** "Revisiting the discrete planar Laplacian," Eur. Phys. J. Plus 136

### Related Work

5. **Borwein, Glasser, McPhedran, Wan, Zucker:** "Lattice Sums Then and Now" (Cambridge, 2013) - Comprehensive reference
6. **Joyce (1973):** "On the simple cubic lattice Green function," Phil. Trans. R. Soc. Lond. A
7. **Guttmann (2010):** "Lattice Green's functions in all dimensions," J. Phys. A 43:305205

---

## 8. Immediate Next Steps

1. **Implement LGF calculation** for 3D simple cubic lattice (code above)

2. **Compute η numerically** at multiple scales and lattice sizes

3. **Compare with Watson integral** exact values

4. **Check universality** by computing η for:
   - Gravitational potential (Poisson equation)
   - Electromagnetic potential (same Poisson equation)
   - Diffusion (heat equation)

5. **Document derivation** if η can be expressed analytically

---

## 9. Preliminary Estimate

Based on the analysis above:

**Continuum Green's function at r=1:** G_cont = 1/(4π) ≈ 0.0796

**Lattice Green's function at nearest neighbor:**
Using G(1,0,0) = (W₃ - 1)/6 where W₃ ≈ 0.5055:
G(1,0,0) = (0.5055 - 1)/6 ≈ -0.0824 (magnitude 0.0824)

**Correction factor:**
η ≈ G_cont / |G_lattice| ≈ 0.0796 / 0.0824 ≈ 0.966

This is remarkably close to the observed 0.9679!

**Tentative Conclusion:** The 0.96 factor is likely the lattice-to-continuum renormalization factor arising from the difference between discrete and continuum Laplacian Green's functions at the lattice scale.

---

*Working note for DET research*
*Date: 2026-01-12*
