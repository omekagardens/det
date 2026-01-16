/**
 * Deep Existence Theory (DET) v6.3 Physics Engine
 * JavaScript implementation for real-time 3D simulation
 */

export class DETEngine {
  constructor(config = {}) {
    // Grid configuration
    this.N = config.N || 32;
    this.DT = config.DT || 0.02;
    this.R = config.R || 2;

    // Parameters (unified schema)
    this.params = {
      // Base parameters
      tau_base: 0.02,
      sigma_base: 0.12,
      lambda_base: 0.008,
      mu_base: 2.0,
      kappa_base: 5.0,
      C_0: 0.15,
      phi_L: 0.5,
      lambda_a: 30.0,

      // Momentum
      alpha_pi: 0.12,
      lambda_pi: 0.008,
      mu_pi: 0.35,
      pi_max: 3.0,

      // Angular momentum
      alpha_L: 0.06,
      lambda_L: 0.005,
      mu_L: 0.18,
      L_max: 5.0,

      // Gravity
      alpha_grav: 0.02,
      kappa: 5.0,
      mu_g: 2.0,
      beta_g: 10.0,

      // Coherence
      C_init: 0.15,
      alpha_C: 0.04,
      lambda_C: 0.002,

      // Agency
      alpha_q: 0.012,
      beta_a: 0.2,
      gamma_a_max: 0.15,
      gamma_a_power: 2.0,

      // Floor
      eta_f: 0.12,
      F_core: 5.0,
      floor_power: 2.0,

      // Other
      F_VAC: 0.01,
      F_MIN: 0.0,
    };

    // Override with config
    Object.assign(this.params, config.params || {});

    // Initialize fields
    this.initializeFields();

    // Step counter
    this.step = 0;
    this.time = 0;
  }

  initializeFields() {
    const N = this.N;
    const N3 = N * N * N;

    // Per-node fields
    this.F = new Float32Array(N3).fill(this.params.F_VAC);
    this.q = new Float32Array(N3).fill(0);
    this.a = new Float32Array(N3).fill(1.0);
    this.sigma = new Float32Array(N3).fill(1.0);
    this.P = new Float32Array(N3).fill(1.0);
    this.Phi = new Float32Array(N3).fill(0);

    // Per-bond fields (simplified - 3 directions per node)
    this.pi_x = new Float32Array(N3).fill(0);
    this.pi_y = new Float32Array(N3).fill(0);
    this.pi_z = new Float32Array(N3).fill(0);
    this.C_x = new Float32Array(N3).fill(this.params.C_init);
    this.C_y = new Float32Array(N3).fill(this.params.C_init);
    this.C_z = new Float32Array(N3).fill(this.params.C_init);

    // Angular momentum (plaquettes - simplified)
    this.L_xy = new Float32Array(N3).fill(0);
    this.L_xz = new Float32Array(N3).fill(0);
    this.L_yz = new Float32Array(N3).fill(0);

    // Temporary arrays
    this.F_new = new Float32Array(N3);
    this.rho = new Float32Array(N3);
    this.baseline = new Float32Array(N3);
  }

  // Index conversion
  idx(x, y, z) {
    const N = this.N;
    // Periodic boundary conditions
    x = ((x % N) + N) % N;
    y = ((y % N) + N) % N;
    z = ((z % N) + N) % N;
    return x + y * N + z * N * N;
  }

  coords(i) {
    const N = this.N;
    return {
      x: i % N,
      y: Math.floor(i / N) % N,
      z: Math.floor(i / (N * N))
    };
  }

  // Add a mass blob
  addMass(cx, cy, cz, mass, radius = 3) {
    const N = this.N;
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 <= radius * radius) {
            const i = this.idx(cx + dx, cy + dy, cz + dz);
            const amplitude = mass * Math.exp(-r2 / (2 * radius * radius / 4));
            this.F[i] += amplitude;
            this.q[i] = Math.min(1, this.q[i] + amplitude * 0.1);
          }
        }
      }
    }
  }

  // Add velocity (momentum) to a region
  addVelocity(cx, cy, cz, vx, vy, vz, radius = 3) {
    const N = this.N;
    for (let dz = -radius; dz <= radius; dz++) {
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const r2 = dx * dx + dy * dy + dz * dz;
          if (r2 <= radius * radius) {
            const i = this.idx(cx + dx, cy + dy, cz + dz);
            this.pi_x[i] += vx;
            this.pi_y[i] += vy;
            this.pi_z[i] += vz;
          }
        }
      }
    }
  }

  // Compute presence (local clock rate)
  computePresence() {
    const N3 = this.N * this.N * this.N;
    for (let i = 0; i < N3; i++) {
      const a = this.a[i];
      const sigma = this.sigma[i];
      const F = this.F[i];
      // Simplified H = sigma
      const H = sigma;
      // P = a * sigma / (1 + F) / (1 + H)
      this.P[i] = (a * sigma) / (1 + F) / (1 + H);
    }
  }

  // Solve Poisson for gravity (simplified FFT-like approach)
  computeGravity() {
    const N = this.N;
    const N3 = N * N * N;
    const kappa = this.params.kappa;
    const alpha = this.params.alpha_grav;

    // Compute baseline (simple averaging)
    for (let i = 0; i < N3; i++) {
      this.baseline[i] = this.q[i];
    }

    // Simple smoothing pass
    const temp = new Float32Array(N3);
    for (let iter = 0; iter < 3; iter++) {
      for (let z = 0; z < N; z++) {
        for (let y = 0; y < N; y++) {
          for (let x = 0; x < N; x++) {
            const i = this.idx(x, y, z);
            let sum = this.baseline[i] * 6;
            sum += this.baseline[this.idx(x + 1, y, z)];
            sum += this.baseline[this.idx(x - 1, y, z)];
            sum += this.baseline[this.idx(x, y + 1, z)];
            sum += this.baseline[this.idx(x, y - 1, z)];
            sum += this.baseline[this.idx(x, y, z + 1)];
            sum += this.baseline[this.idx(x, y, z - 1)];
            temp[i] = sum / 12;
          }
        }
      }
      this.baseline.set(temp);
    }

    // Compute rho = q - baseline
    for (let i = 0; i < N3; i++) {
      this.rho[i] = this.q[i] - this.baseline[i];
    }

    // Solve Poisson iteratively (Jacobi)
    const Phi_temp = new Float32Array(N3);
    for (let iter = 0; iter < 20; iter++) {
      for (let z = 0; z < N; z++) {
        for (let y = 0; y < N; y++) {
          for (let x = 0; x < N; x++) {
            const i = this.idx(x, y, z);
            let laplacian =
              this.Phi[this.idx(x + 1, y, z)] +
              this.Phi[this.idx(x - 1, y, z)] +
              this.Phi[this.idx(x, y + 1, z)] +
              this.Phi[this.idx(x, y - 1, z)] +
              this.Phi[this.idx(x, y, z + 1)] +
              this.Phi[this.idx(x, y, z - 1)] -
              6 * this.Phi[i];
            // Poisson: laplacian(Phi) = kappa * rho
            Phi_temp[i] = (laplacian - kappa * this.rho[i]) / (-6);
          }
        }
      }
      this.Phi.set(Phi_temp);
    }
  }

  // Compute diffusive flux
  computeFlux(x, y, z, dx, dy, dz) {
    const i = this.idx(x, y, z);
    const j = this.idx(x + dx, y + dy, z + dz);

    const Fi = this.F[i];
    const Fj = this.F[j];
    const ai = this.a[i];
    const aj = this.a[j];

    // Agency gate
    const g_a = Math.sqrt(ai * aj);

    // Get coherence for this direction
    let C;
    if (dx !== 0) C = this.C_x[i];
    else if (dy !== 0) C = this.C_y[i];
    else C = this.C_z[i];

    // Diffusive flux (classical part for simplicity)
    const sqrtC = Math.sqrt(C);
    const J_diff = g_a * (1 - sqrtC) * (Fi - Fj);

    // Gravitational flux
    const Phi_i = this.Phi[i];
    const Phi_j = this.Phi[j];
    const F_avg = (Fi + Fj) / 2;
    const J_grav = this.params.mu_g * F_avg * (Phi_i - Phi_j);

    // Momentum flux
    let pi;
    if (dx !== 0) pi = this.pi_x[i];
    else if (dy !== 0) pi = this.pi_y[i];
    else pi = this.pi_z[i];
    const J_mom = this.params.mu_pi * pi * F_avg;

    // Floor repulsion
    const F_core = this.params.F_core;
    const si = Math.max(0, (Fi - F_core) / F_core) ** this.params.floor_power;
    const sj = Math.max(0, (Fj - F_core) / F_core) ** this.params.floor_power;
    const J_floor = this.params.eta_f * (si + sj) * (Fi - Fj);

    return J_diff + J_grav + J_mom + J_floor;
  }

  // Main simulation step
  step_simulation() {
    const N = this.N;
    const N3 = N * N * N;
    const DT = this.DT;
    const params = this.params;

    // Step 0: Compute gravity
    this.computeGravity();

    // Step 1: Compute presence
    this.computePresence();

    // Step 2-4: Compute fluxes and update F
    this.F_new.set(this.F);

    for (let z = 0; z < N; z++) {
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const i = this.idx(x, y, z);
          const dt = this.P[i] * DT;

          // Compute fluxes in each direction
          const J_xp = this.computeFlux(x, y, z, 1, 0, 0);
          const J_xm = this.computeFlux(x, y, z, -1, 0, 0);
          const J_yp = this.computeFlux(x, y, z, 0, 1, 0);
          const J_ym = this.computeFlux(x, y, z, 0, -1, 0);
          const J_zp = this.computeFlux(x, y, z, 0, 0, 1);
          const J_zm = this.computeFlux(x, y, z, 0, 0, -1);

          // Net flux
          const total_out = Math.max(0, J_xp) + Math.max(0, -J_xm) +
            Math.max(0, J_yp) + Math.max(0, -J_ym) +
            Math.max(0, J_zp) + Math.max(0, -J_zm);

          // Limiter
          const max_out = this.F[i] * 0.5;
          const scale = total_out > max_out ? max_out / total_out : 1;

          // Apply fluxes
          this.F_new[i] -= (J_xp - J_xm + J_yp - J_ym + J_zp - J_zm) * dt * scale;
          this.F_new[i] = Math.max(params.F_MIN, this.F_new[i]);
        }
      }
    }

    // Copy back
    this.F.set(this.F_new);

    // Step 6: Update momentum
    for (let z = 0; z < N; z++) {
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const i = this.idx(x, y, z);
          const dt = this.P[i] * DT;

          // Gravity gradient
          const gx = (this.Phi[this.idx(x + 1, y, z)] - this.Phi[this.idx(x - 1, y, z)]) / 2;
          const gy = (this.Phi[this.idx(x, y + 1, z)] - this.Phi[this.idx(x, y - 1, z)]) / 2;
          const gz = (this.Phi[this.idx(x, y, z + 1)] - this.Phi[this.idx(x, y, z - 1)]) / 2;

          // Update with decay and gravity coupling
          this.pi_x[i] = (1 - params.lambda_pi * dt) * this.pi_x[i] - params.beta_g * gx * dt;
          this.pi_y[i] = (1 - params.lambda_pi * dt) * this.pi_y[i] - params.beta_g * gy * dt;
          this.pi_z[i] = (1 - params.lambda_pi * dt) * this.pi_z[i] - params.beta_g * gz * dt;

          // Clamp momentum
          const pi_mag = Math.sqrt(this.pi_x[i] ** 2 + this.pi_y[i] ** 2 + this.pi_z[i] ** 2);
          if (pi_mag > params.pi_max) {
            const scale = params.pi_max / pi_mag;
            this.pi_x[i] *= scale;
            this.pi_y[i] *= scale;
            this.pi_z[i] *= scale;
          }
        }
      }
    }

    // Step 8: Update structure q
    for (let i = 0; i < N3; i++) {
      const dF = this.F_new[i] - this.F[i];
      if (dF < 0) {
        this.q[i] = Math.min(1, this.q[i] + params.alpha_q * Math.abs(dF));
      }
    }

    // Step 9: Update agency
    for (let z = 0; z < N; z++) {
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const i = this.idx(x, y, z);
          const q = this.q[i];

          // Structural ceiling
          const a_max = 1 / (1 + params.lambda_a * q * q);

          // Relax toward ceiling
          this.a[i] = this.a[i] + params.beta_a * (a_max - this.a[i]) * DT;
          this.a[i] = Math.max(0, Math.min(a_max, this.a[i]));
        }
      }
    }

    // Step 10: Update coherence (simplified)
    for (let i = 0; i < N3; i++) {
      const decay = params.lambda_C * DT;
      this.C_x[i] = Math.max(0.01, this.C_x[i] - decay * this.C_x[i]);
      this.C_y[i] = Math.max(0.01, this.C_y[i] - decay * this.C_y[i]);
      this.C_z[i] = Math.max(0.01, this.C_z[i] - decay * this.C_z[i]);
    }

    this.step++;
    this.time += DT;
  }

  // Get statistics
  getStats() {
    const N3 = this.N * this.N * this.N;
    let totalF = 0;
    let maxF = 0;
    let totalQ = 0;
    let maxQ = 0;
    let totalP = 0;
    let minP = Infinity;

    for (let i = 0; i < N3; i++) {
      totalF += this.F[i];
      maxF = Math.max(maxF, this.F[i]);
      totalQ += this.q[i];
      maxQ = Math.max(maxQ, this.q[i]);
      totalP += this.P[i];
      minP = Math.min(minP, this.P[i]);
    }

    return {
      step: this.step,
      time: this.time,
      totalF,
      maxF,
      avgF: totalF / N3,
      totalQ,
      maxQ,
      avgP: totalP / N3,
      minP
    };
  }

  // Get field data for visualization
  getFieldData(field = 'F', threshold = 0.1) {
    const N = this.N;
    const data = [];
    let fieldArray;

    switch (field) {
      case 'F': fieldArray = this.F; break;
      case 'q': fieldArray = this.q; break;
      case 'P': fieldArray = this.P; break;
      case 'Phi': fieldArray = this.Phi; break;
      default: fieldArray = this.F;
    }

    // Find max for normalization
    let maxVal = 0;
    for (let i = 0; i < fieldArray.length; i++) {
      maxVal = Math.max(maxVal, Math.abs(fieldArray[i]));
    }
    if (maxVal === 0) maxVal = 1;

    for (let z = 0; z < N; z++) {
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const i = this.idx(x, y, z);
          const value = fieldArray[i];
          const normalized = value / maxVal;

          if (Math.abs(normalized) > threshold) {
            data.push({
              x: x - N / 2,
              y: y - N / 2,
              z: z - N / 2,
              value,
              normalized,
              F: this.F[i],
              q: this.q[i],
              a: this.a[i],
              P: this.P[i]
            });
          }
        }
      }
    }

    return data;
  }
}

export default DETEngine;
