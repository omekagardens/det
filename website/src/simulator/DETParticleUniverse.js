/**
 * Deep Existence Theory (DET) v6.3/v6.4 Particle Universe Engine
 * Complete implementation with bonding, collision, fusion, boundary, and grace
 *
 * Each particle ("creature") has:
 * - F: Resource (stored energy/mass)
 * - q: Structural debt (retained past, creates gravity, enables binding)
 * - a: Agency (decision-making capacity)
 * - σ: Processing rate
 * - θ: Phase (for quantum coherence)
 * - π: Momentum vector
 * - L: Angular momentum (for orbital dynamics)
 * - isBoundary: Whether this is a boundary agent
 *
 * Bonds between particles have:
 * - C: Coherence (quantum correlation strength)
 * - π_bond: Bond momentum (memory of flow)
 * - L: Bond angular momentum
 */

export class DETParticleUniverse {
  constructor(config = {}) {
    this.numParticles = config.numParticles || 200;
    this.numBoundary = config.numBoundary || 60;
    this.width = config.width || 800;
    this.height = config.height || 800;
    this.DT = config.DT || 0.02;

    // Feature toggles
    this.boundaryEnabled = config.boundaryEnabled !== false;
    this.graceEnabled = config.graceEnabled !== false;
    this.replicationEnabled = config.replicationEnabled || false;

    // DET Parameters (from unified schema v6.3/v6.4)
    this.params = {
      // Gravity
      kappa: 5.0,           // Gravitational coupling (κ)
      mu_g: 2.0,            // Gravity mobility
      alpha_grav: 0.02,     // Screening parameter
      beta_g: 10.0,         // Gravity-momentum coupling (5 × μ_g)

      // Momentum
      alpha_pi: 0.12,       // Momentum charging gain
      lambda_pi: 0.008,     // Momentum decay rate
      mu_pi: 0.35,          // Momentum mobility
      pi_max: 3.0,          // Maximum momentum

      // Angular Momentum (for orbital capture)
      alpha_L: 0.06,        // Angular momentum charging
      lambda_L: 0.005,      // Angular momentum decay
      mu_L: 0.18,           // Rotational mobility
      L_max: 5.0,           // Maximum angular momentum

      // Agency
      lambda_a: 30.0,       // Structural ceiling coupling
      beta_a: 0.2,          // Agency relaxation rate

      // Coherence (quantum bonds)
      C_init: 0.15,         // Initial coherence for new bonds
      C_min: 0.01,          // Minimum coherence
      alpha_C: 0.04,        // Coherence growth rate (from flow)
      lambda_C: 0.002,      // Coherence decay rate

      // Structure (q-locking for binding)
      alpha_q: 0.012,       // Q-locking rate (binding during collision)
      q_bind_threshold: 0.3, // q threshold for stable binding

      // Floor repulsion (prevent infinite compression)
      eta_f: 0.12,          // Floor stiffness
      F_core: 5.0,          // Core onset density
      floor_power: 2.0,     // Floor exponent

      // Phase
      omega_0: 0.1,         // Base phase evolution rate

      // Fusion
      fusion_distance: 15,  // Distance for potential fusion
      fusion_C_threshold: 0.6, // Coherence needed for fusion

      // Boundary
      R_boundary: 50,       // Connection distance to boundary
      boundary_F: 3.0,      // Resource level of boundary agents
      boundary_a: 0.8,      // Agency of boundary agents
      initial_boundary_C: 0.3, // Initial coherence to boundary

      // Grace (v6.4)
      eta_g: 0.5,           // Grace flux coefficient
      beta_grace: 0.5,      // Grace threshold factor (F_thresh = β × ⟨F⟩)
      C_quantum: 0.85,      // Quantum gate threshold (high C blocks grace)
      F_min_grace: 0.05,    // Minimum F below which grace activates

      // Visuals
      trailLength: 20,
      bond_display_threshold: 0.05,
      interaction_range: 150,

      // Replication (from DET subdivision theory v3)
      a_min_division: 0.2,      // Minimum agency to initiate division
      F_min_division: 0.8,      // Minimum resource to divide
      drive_threshold: 0.15,    // Local drive threshold for fork eligibility
      lambda_fork: 0.15,        // Rate of coherence reduction per step
      C_open_threshold: 0.1,    // Bond considered "open" below this
      kappa_break: 0.05,        // Cost per |ΔC| during unzipping
      kappa_form: 0.08,         // Cost per C_init when forming new bonds
      replication_cooldown: 50, // Steps between replications for a particle
      max_particles: 400,       // Maximum particles (prevent runaway replication)
    };

    Object.assign(this.params, config.params || {});

    this.particles = [];
    this.bonds = new Map();
    this.step = 0;
    this.time = 0;
    this.graceFlowTotal = 0; // Track total grace flow for stats

    // Replication tracking
    this.activeForks = new Map(); // parentId -> fork state
    this.replicationCount = 0;    // Total replications this session
    this.dormantPool = [];        // Pre-allocated dormant particles for recruitment

    this.initializeParticles();
  }

  initializeParticles() {
    this.particles = [];
    this.bonds.clear();
    this.activeForks.clear();
    this.dormantPool = [];
    const cx = this.width / 2;
    const cy = this.height / 2;

    // Create boundary particles first (if enabled)
    if (this.boundaryEnabled) {
      const boundaryRadius = Math.min(this.width, this.height) * 0.45;
      for (let i = 0; i < this.numBoundary; i++) {
        const angle = (i / this.numBoundary) * Math.PI * 2;
        this.particles.push({
          id: i,
          x: cx + Math.cos(angle) * boundaryRadius,
          y: cy + Math.sin(angle) * boundaryRadius,
          F: this.params.boundary_F,
          q: 0.1,  // Low structural debt
          a: this.params.boundary_a,
          sigma: 1.0,
          theta: angle,  // Phase aligned with position
          px: 0,
          py: 0,
          L: 0,
          P: 1.0,
          trail: [],
          alive: true,
          isBoundary: true,
          isDormant: false,
          replicationCooldown: 0,
          recentFlux: 0,
        });
      }
    }

    // Create interior particles
    const startId = this.boundaryEnabled ? this.numBoundary : 0;
    for (let i = 0; i < this.numParticles; i++) {
      const angle = Math.random() * Math.PI * 2;
      const maxR = Math.min(this.width, this.height) * (this.boundaryEnabled ? 0.35 : 0.4);
      const r = Math.random() * maxR;

      this.particles.push({
        id: startId + i,
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,
        F: 0.5 + Math.random() * 1.5,
        q: Math.random() * 0.4,
        a: 0.7 + Math.random() * 0.3,
        sigma: 0.8 + Math.random() * 0.4,
        theta: Math.random() * Math.PI * 2,
        px: (Math.random() - 0.5) * 2,
        py: (Math.random() - 0.5) * 2,
        L: 0,
        P: 1.0,
        trail: [],
        alive: true,
        isBoundary: false,
        isDormant: false,
        replicationCooldown: 0,
        recentFlux: 0,
      });
    }

    // Create initial bonds to boundary
    if (this.boundaryEnabled) {
      this.createBoundaryBonds();
    }

    // Initialize dormant pool for replication
    this.initializeDormantPool();
  }

  // Create pool of dormant particles that can be recruited during replication
  initializeDormantPool() {
    const cx = this.width / 2;
    const cy = this.height / 2;
    const numDormant = 50; // Pre-allocate dormant particles
    const startId = this.particles.length;

    for (let i = 0; i < numDormant; i++) {
      // Dormant particles are scattered but not visible/active
      const angle = Math.random() * Math.PI * 2;
      const maxR = Math.min(this.width, this.height) * 0.4;
      const r = Math.random() * maxR;

      const dormant = {
        id: startId + i,
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,
        F: 0.2 + Math.random() * 0.3, // Low initial resource
        q: 0,
        a: 0.3 + Math.random() * 0.4, // Moderate agency potential
        sigma: 0.8 + Math.random() * 0.4,
        theta: Math.random() * Math.PI * 2,
        px: 0,
        py: 0,
        L: 0,
        P: 0, // Dormant have no presence
        trail: [],
        alive: true,
        isBoundary: false,
        isDormant: true, // Key: this particle is dormant
        replicationCooldown: 0,
        recentFlux: 0,
      };

      this.particles.push(dormant);
      this.dormantPool.push(dormant.id);
    }
  }

  // Create initial bonds connecting interior particles to nearby boundary agents
  createBoundaryBonds() {
    const interiorParticles = this.particles.filter(p => !p.isBoundary);
    const boundaryParticles = this.particles.filter(p => p.isBoundary);

    for (const interior of interiorParticles) {
      // Find closest boundary particles
      const distances = boundaryParticles.map(bp => {
        const dx = bp.x - interior.x;
        const dy = bp.y - interior.y;
        return { bp, dist: Math.sqrt(dx * dx + dy * dy) };
      }).sort((a, b) => a.dist - b.dist);

      // Connect to 2-3 nearest boundary particles
      const numConnections = 2 + Math.floor(Math.random() * 2);
      for (let i = 0; i < Math.min(numConnections, distances.length); i++) {
        if (distances[i].dist < this.params.R_boundary * 8) {
          const bond = this.getBond(interior.id, distances[i].bp.id);
          bond.C = this.params.initial_boundary_C * (1 - distances[i].dist / (this.params.R_boundary * 8));
        }
      }
    }
  }

  bondKey(i, j) {
    return i < j ? `${i}-${j}` : `${j}-${i}`;
  }

  getBond(i, j) {
    const key = this.bondKey(i, j);
    if (!this.bonds.has(key)) {
      this.bonds.set(key, {
        C: 0,
        pi: 0,
        L: 0,
      });
    }
    return this.bonds.get(key);
  }

  // DET Presence formula: P = a·σ·(1+F)⁻¹·(1+H)⁻¹
  computePresence(particle, numBonds) {
    const { a, sigma, F } = particle;
    const H = numBonds * 0.1;
    return (a * sigma) / (1 + F) / (1 + H);
  }

  // Agency ceiling: a_max = 1/(1 + λ_a·q²)
  computeAgencyCeiling(q) {
    return 1 / (1 + this.params.lambda_a * q * q);
  }

  // Gravitational force between particles
  computeGravity(p1, p2, r, dx, dy) {
    const q_source = p2.q - this.params.alpha_grav;
    if (q_source <= 0) return { fx: 0, fy: 0 };

    const strength = this.params.kappa * q_source * p1.F / (r * r + 10);
    return {
      fx: strength * dx / r,
      fy: strength * dy / r
    };
  }

  // Floor repulsion
  computeFloorRepulsion(p1, p2, r, dx, dy) {
    const minDist = Math.sqrt(p1.F + p2.F) * 8;
    if (r > minDist) return { fx: 0, fy: 0 };

    const overlap = Math.pow((minDist - r) / minDist, this.params.floor_power);
    const strength = this.params.eta_f * overlap * (p1.F + p2.F) * 50;

    return {
      fx: -strength * dx / (r + 0.1),
      fy: -strength * dy / (r + 0.1)
    };
  }

  // Diffusive flow (quantum-classical interpolation)
  computeDiffusiveFlow(p1, p2, bond, r) {
    const g_a = Math.sqrt(p1.a * p2.a);
    const sqrtC = Math.sqrt(bond.C);
    const phaseTerm = Math.sin(p1.theta - p2.theta);
    const pressureTerm = p1.F - p2.F;

    return g_a * (sqrtC * phaseTerm * 0.5 + (1 - sqrtC) * pressureTerm * 0.1);
  }

  // Grace mechanism (v6.4)
  computeGrace(particles, particleIndex) {
    if (!this.graceEnabled) return 0;

    const p = particles[particleIndex];
    if (!p.alive) return 0;

    // Find neighbors within interaction range
    const neighbors = [];
    let totalNeighborF = 0;
    let totalNeighborR = 0; // Total recipient need in neighborhood

    for (let j = 0; j < particles.length; j++) {
      if (j === particleIndex || !particles[j].alive) continue;

      const other = particles[j];
      const dx = other.x - p.x;
      const dy = other.y - p.y;
      const r = Math.sqrt(dx * dx + dy * dy);

      if (r < this.params.interaction_range) {
        neighbors.push({ particle: other, index: j, dist: r });
        totalNeighborF += other.F;
      }
    }

    if (neighbors.length === 0) return 0;

    // Compute threshold: F_thresh = β × ⟨F⟩_neighbors
    const avgNeighborF = totalNeighborF / neighbors.length;
    const F_thresh = this.params.beta_grace * avgNeighborF;

    // Need and Excess
    const need_i = Math.max(0, F_thresh - p.F);
    const excess_i = Math.max(0, p.F - F_thresh);

    // Donor capacity and recipient need (agency-gated)
    const d_i = p.a * excess_i;
    const r_i = p.a * need_i;

    // Compute total recipient need in neighborhood
    for (const neighbor of neighbors) {
      const other = neighbor.particle;
      const other_F_thresh = this.params.beta_grace * avgNeighborF;
      const other_need = Math.max(0, other_F_thresh - other.F);
      const other_r = other.a * other_need;
      totalNeighborR += other_r;
    }

    // Compute grace flow
    let graceIn = 0;

    for (const neighbor of neighbors) {
      const other = neighbor.particle;
      const bond = this.getBond(p.id, other.id);

      // Quantum gate: Q = max(0, 1 - √C / C_quantum)
      // High coherence BLOCKS grace (quantum regime doesn't need it)
      const Q = Math.max(0, 1 - Math.sqrt(bond.C) / this.params.C_quantum);

      if (Q < 0.01) continue; // Skip if quantum gate blocks

      // Agency gate
      const g_a = Math.sqrt(p.a * other.a);

      // Other's need and excess
      const other_F_thresh = this.params.beta_grace * avgNeighborF;
      const other_need = Math.max(0, other_F_thresh - other.F);
      const other_excess = Math.max(0, other.F - other_F_thresh);
      const d_j = other.a * other_excess;
      const r_j = other.a * other_need;

      // Compute other's neighborhood recipient need (simplified: use same total)
      const epsilon = 0.001;

      // Grace flux (antisymmetric)
      // G_{i→j} = η_g × g^a × Q × (d_i × r_j/Σr_k - d_j × r_i/Σr_k)
      const grace_out = d_i * (r_j / (totalNeighborR + epsilon));
      const grace_in_from_j = d_j * (r_i / (totalNeighborR + epsilon));

      const G_ij = this.params.eta_g * g_a * Q * (grace_in_from_j - grace_out);

      graceIn += G_ij;
    }

    return graceIn;
  }

  step_simulation() {
    const dt = this.DT;
    const particles = this.particles.filter(p => p.alive);
    const n = particles.length;

    if (n === 0) return;

    const bondCounts = new Array(n).fill(0);
    const forces = particles.map(() => ({ fx: 0, fy: 0 }));
    const flowIn = particles.map(() => 0);
    const qLockIn = particles.map(() => 0);
    const graceIn = particles.map(() => 0);
    const activeBonds = new Set();

    // Create index map for quick lookup
    const indexMap = new Map();
    particles.forEach((p, idx) => indexMap.set(p.id, idx));

    // Compute grace for each particle
    if (this.graceEnabled) {
      for (let i = 0; i < n; i++) {
        graceIn[i] = this.computeGrace(particles, i);
      }
      this.graceFlowTotal = graceIn.reduce((sum, g) => sum + Math.abs(g), 0);
    }

    // Pairwise interactions
    for (let i = 0; i < n; i++) {
      const p1 = particles[i];

      for (let j = i + 1; j < n; j++) {
        const p2 = particles[j];

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const r = Math.sqrt(dx * dx + dy * dy);

        if (r > this.params.interaction_range) continue;

        const bond = this.getBond(p1.id, p2.id);
        activeBonds.add(this.bondKey(p1.id, p2.id));

        // Update coherence
        const phaseAlign = 0.5 + 0.5 * Math.cos(p1.theta - p2.theta);
        const distFactor = Math.max(0, 1 - r / this.params.interaction_range);
        const agencyGate = Math.sqrt(p1.a * p2.a);

        // Boundary bonds decay slower
        const isBoundaryBond = p1.isBoundary || p2.isBoundary;
        const decayMultiplier = isBoundaryBond ? 0.5 : 1.0;

        const C_growth = this.params.alpha_C * phaseAlign * distFactor * agencyGate * dt;
        const C_decay = this.params.lambda_C * bond.C * dt * decayMultiplier;

        bond.C = Math.min(1, Math.max(this.params.C_min, bond.C + C_growth - C_decay));

        if (bond.C > this.params.bond_display_threshold) {
          bondCounts[i]++;
          bondCounts[j]++;
        }

        // Diffusive flow
        const J_diff = this.computeDiffusiveFlow(p1, p2, bond, r);

        // Update bond momentum
        const avgDt = dt * 0.5 * (p1.P + p2.P);
        bond.pi = (1 - this.params.lambda_pi * avgDt) * bond.pi
                  + this.params.alpha_pi * J_diff * avgDt;
        bond.pi = Math.max(-this.params.pi_max, Math.min(this.params.pi_max, bond.pi));

        // Momentum-driven drift flux
        const J_mom = this.params.mu_pi * bond.pi * (p1.F + p2.F) / 2;

        // Total flow
        const J_total = J_diff + J_mom;
        flowIn[i] -= J_total;
        flowIn[j] += J_total;

        // Skip physics for boundary particles (they don't move)
        if (!p1.isBoundary && !p2.isBoundary) {
          // Gravity
          const grav1 = this.computeGravity(p1, p2, r, dx, dy);
          const grav2 = this.computeGravity(p2, p1, r, -dx, -dy);
          forces[i].fx += grav1.fx;
          forces[i].fy += grav1.fy;
          forces[j].fx += grav2.fx;
          forces[j].fy += grav2.fy;

          // Floor repulsion
          const floor = this.computeFloorRepulsion(p1, p2, r, dx, dy);
          forces[i].fx += floor.fx;
          forces[i].fy += floor.fy;
          forces[j].fx -= floor.fx;
          forces[j].fy -= floor.fy;

          // Q-locking during close approach
          const collisionDist = Math.sqrt(p1.F + p2.F) * 6;
          if (r < collisionDist && bond.C > 0.1) {
            const qLock = this.params.alpha_q * bond.C * (collisionDist - r) / collisionDist;
            qLockIn[i] += qLock;
            qLockIn[j] += qLock;
          }

          // Angular momentum
          const cross = dx * p1.py - dy * p1.px;
          const torque = this.params.alpha_L * cross * bond.C * 0.01;
          bond.L = (1 - this.params.lambda_L * avgDt) * bond.L + torque * avgDt;
          bond.L = Math.max(-this.params.L_max, Math.min(this.params.L_max, bond.L));

          if (Math.abs(bond.L) > 0.01 && r > 10) {
            const rotStrength = this.params.mu_L * bond.L / (r + 10);
            forces[i].fx += -dy * rotStrength * 0.5;
            forces[i].fy += dx * rotStrength * 0.5;
            forces[j].fx += dy * rotStrength * 0.5;
            forces[j].fy += -dx * rotStrength * 0.5;
          }

          // Fusion check
          if (r < this.params.fusion_distance && bond.C > this.params.fusion_C_threshold) {
            this.fuseParticles(p1, p2, i, j, particles);
          }
        } else if (p1.isBoundary !== p2.isBoundary) {
          // Interior particle near boundary - apply soft attraction to stay in bounds
          const interior = p1.isBoundary ? p2 : p1;
          const interiorIdx = p1.isBoundary ? j : i;
          const boundary = p1.isBoundary ? p1 : p2;

          // Compute distance from center
          const cx = this.width / 2;
          const cy = this.height / 2;
          const distFromCenter = Math.sqrt(
            (interior.x - cx) ** 2 + (interior.y - cy) ** 2
          );
          const maxRadius = Math.min(this.width, this.height) * 0.42;

          // Soft repulsion if too close to boundary
          if (distFromCenter > maxRadius * 0.9) {
            const pushStrength = (distFromCenter - maxRadius * 0.9) / maxRadius * 2;
            forces[interiorIdx].fx -= (interior.x - cx) / distFromCenter * pushStrength;
            forces[interiorIdx].fy -= (interior.y - cy) / distFromCenter * pushStrength;
          }
        }
      }
    }

    // Update each particle
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      if (!p.alive) continue;

      // Boundary particles only update resources, not position
      if (p.isBoundary) {
        // Boundary agents slowly restore their resources
        p.F = Math.max(this.params.boundary_F * 0.5,
              Math.min(this.params.boundary_F * 1.5, p.F + flowIn[i] * dt * 0.5));

        // Boundary maintains phase stability
        p.theta = (p.theta + this.params.omega_0 * dt * 0.5) % (Math.PI * 2);
        continue;
      }

      const f = forces[i];

      // Compute presence
      p.P = this.computePresence(p, bondCounts[i]);
      const localDt = dt * p.P;

      // Update momentum
      p.px = (1 - this.params.lambda_pi * localDt) * p.px + this.params.beta_g * f.fx * localDt;
      p.py = (1 - this.params.lambda_pi * localDt) * p.py + this.params.beta_g * f.fy * localDt;

      const pMag = Math.sqrt(p.px * p.px + p.py * p.py);
      if (pMag > this.params.pi_max * 3) {
        p.px *= this.params.pi_max * 3 / pMag;
        p.py *= this.params.pi_max * 3 / pMag;
      }

      // Update position
      p.x += p.px * this.params.mu_pi * localDt * 30;
      p.y += p.py * this.params.mu_pi * localDt * 30;

      // Boundary wrapping (only if no boundary enabled)
      if (!this.boundaryEnabled) {
        if (p.x < 0) p.x += this.width;
        if (p.x > this.width) p.x -= this.width;
        if (p.y < 0) p.y += this.height;
        if (p.y > this.height) p.y -= this.height;
      } else {
        // Soft clamp within boundary
        const cx = this.width / 2;
        const cy = this.height / 2;
        const maxR = Math.min(this.width, this.height) * 0.43;
        const dx = p.x - cx;
        const dy = p.y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > maxR) {
          p.x = cx + dx * maxR / dist;
          p.y = cy + dy * maxR / dist;
          // Reflect momentum
          const dot = (p.px * dx + p.py * dy) / dist;
          if (dot > 0) {
            p.px -= 1.5 * dot * dx / dist;
            p.py -= 1.5 * dot * dy / dist;
          }
        }
      }

      // Update resource F (including grace)
      p.F = Math.max(0.1, p.F + flowIn[i] * localDt + graceIn[i] * localDt);

      // Update structural debt q
      p.q = Math.min(1, p.q + qLockIn[i] * localDt);

      // Update agency toward ceiling
      const a_max = this.computeAgencyCeiling(p.q);
      p.a = p.a + this.params.beta_a * (a_max - p.a) * localDt;
      p.a = Math.max(0.01, Math.min(a_max, p.a));

      // Update phase
      p.theta = (p.theta + this.params.omega_0 * localDt) % (Math.PI * 2);

      // Update trail
      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > this.params.trailLength) {
        p.trail.shift();
      }
    }

    // Clean up inactive bonds
    for (const key of this.bonds.keys()) {
      if (!activeBonds.has(key)) {
        const bond = this.bonds.get(key);
        bond.C *= 0.95;
        if (bond.C < this.params.C_min) {
          this.bonds.delete(key);
        }
      }
    }

    // Process replication (if enabled)
    if (this.replicationEnabled && this.step % 3 === 0) {
      this.processReplicationStep();
    }

    this.step++;
    this.time += dt;
  }

  fuseParticles(p1, p2, i1, i2, particles) {
    // Don't fuse boundary particles
    if (p1.isBoundary || p2.isBoundary) return;

    const totalF = p1.F + p2.F;
    const totalQ = p1.q + p2.q;
    const totalPx = p1.px * p1.F + p2.px * p2.F;
    const totalPy = p1.py * p1.F + p2.py * p2.F;

    const survivor = p1.F >= p2.F ? p1 : p2;
    const absorbed = p1.F >= p2.F ? p2 : p1;

    survivor.F = totalF;
    survivor.q = Math.min(1, (totalQ / 2) + 0.1);
    survivor.px = totalPx / totalF;
    survivor.py = totalPy / totalF;
    survivor.x = (p1.x * p1.F + p2.x * p2.F) / totalF;
    survivor.y = (p1.y * p1.F + p2.y * p2.F) / totalF;

    absorbed.alive = false;
    absorbed.F = 0;

    const absorbedId = absorbed.id;
    for (const key of this.bonds.keys()) {
      if (key.includes(`${absorbedId}-`) || key.includes(`-${absorbedId}`)) {
        this.bonds.delete(key);
      }
    }
  }

  getBonds() {
    const bonds = [];
    for (const [key, bond] of this.bonds) {
      if (bond.C > this.params.bond_display_threshold) {
        const [i, j] = key.split('-').map(Number);
        const p1 = this.particles.find(p => p.id === i && p.alive);
        const p2 = this.particles.find(p => p.id === j && p.alive);
        if (p1 && p2) {
          bonds.push({
            p1, p2,
            C: bond.C,
            L: bond.L,
            isBoundaryBond: p1.isBoundary || p2.isBoundary
          });
        }
      }
    }
    return bonds;
  }

  getStats() {
    const alive = this.particles.filter(p => p.alive && !p.isBoundary);
    const boundary = this.particles.filter(p => p.alive && p.isBoundary);
    let totalF = 0, totalQ = 0, totalP = 0;
    let maxF = 0, maxQ = 0;

    for (const p of alive) {
      totalF += p.F;
      totalQ += p.q;
      totalP += p.P;
      maxF = Math.max(maxF, p.F);
      maxQ = Math.max(maxQ, p.q);
    }

    const n = alive.length;
    const initialCount = this.numParticles;
    const dormantCount = this.dormantPool.length;

    return {
      step: this.step,
      time: this.time,
      particles: n,
      boundary: boundary.length,
      totalF,
      avgF: n > 0 ? totalF / n : 0,
      maxF,
      avgQ: n > 0 ? totalQ / n : 0,
      maxQ,
      avgP: n > 0 ? totalP / n : 0,
      bonds: this.getBonds().length,
      boundaryBonds: this.getBonds().filter(b => b.isBoundaryBond).length,
      fusions: initialCount - n,
      graceFlow: this.graceFlowTotal,
      replications: this.replicationCount,
      dormant: dormantCount,
      activeForks: this.activeForks.size
    };
  }

  addParticle(x, y, vx = 0, vy = 0) {
    const newId = this.particles.length;
    const p = {
      id: newId,
      x, y,
      F: 0.5 + Math.random() * 0.5,
      q: Math.random() * 0.2,
      a: 0.8 + Math.random() * 0.2,
      sigma: 0.8 + Math.random() * 0.4,
      theta: Math.random() * Math.PI * 2,
      px: vx,
      py: vy,
      L: 0,
      P: 1.0,
      trail: [],
      alive: true,
      isBoundary: false,
      isDormant: false,
      replicationCooldown: 0,
      recentFlux: 0,
    };
    this.particles.push(p);

    // Connect to nearby boundary if enabled
    if (this.boundaryEnabled) {
      const boundaryParticles = this.particles.filter(bp => bp.isBoundary);
      for (const bp of boundaryParticles) {
        const dx = bp.x - p.x;
        const dy = bp.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < this.params.R_boundary * 5) {
          const bond = this.getBond(p.id, bp.id);
          bond.C = this.params.initial_boundary_C * 0.5;
        }
      }
    }

    return p;
  }

  // Toggle boundary
  setBoundaryEnabled(enabled) {
    if (this.boundaryEnabled === enabled) return;

    this.boundaryEnabled = enabled;
    this.initializeParticles();
  }

  // Toggle grace
  setGraceEnabled(enabled) {
    this.graceEnabled = enabled;
  }

  // Toggle replication
  setReplicationEnabled(enabled) {
    this.replicationEnabled = enabled;
    if (enabled && this.dormantPool.length === 0) {
      this.initializeDormantPool();
    }
  }

  // ========================================================================
  // REPLICATION SYSTEM (Based on DET Subdivision Theory v3)
  // ========================================================================

  // Compute local drive/stress that makes a particle want to replicate
  computeLocalDrive(particle, neighbors) {
    // Factor 1: Recent flux magnitude
    const fluxDrive = Math.abs(particle.recentFlux || 0);

    // Factor 2: Resource gradient (am I a local maximum?)
    let gradientDrive = 0;
    if (neighbors.length > 0) {
      const neighborF = neighbors.reduce((sum, n) => sum + n.particle.F, 0) / neighbors.length;
      gradientDrive = Math.max(0, particle.F - neighborF);
    }

    // Factor 3: Agency * resource (high agency + high resource = drive)
    const agencyDrive = particle.a * particle.F * 0.1;

    return fluxDrive + gradientDrive + agencyDrive;
  }

  // Compute stress on a bond (high stress = good candidate for forking)
  computeBondStress(p1, p2, bond) {
    // Resource differential across bond
    const F_diff = Math.abs(p1.F - p2.F);

    // Phase misalignment
    const theta_diff = Math.abs(Math.sin(p1.theta - p2.theta));

    // Combined stress
    return F_diff * 0.5 + theta_diff * 0.3;
  }

  // Check if particle can initiate a replication fork
  checkForkEligibility(particle, neighbors) {
    const params = this.params;

    // Must be active and alive
    if (!particle.alive || particle.isDormant || particle.isBoundary) {
      return { eligible: false, reason: 'Not active' };
    }

    // Check agency gate
    if (particle.a < params.a_min_division) {
      return { eligible: false, reason: 'Agency too low' };
    }

    // Check resource
    if (particle.F < params.F_min_division) {
      return { eligible: false, reason: 'Resource too low' };
    }

    // Check cooldown
    if (particle.replicationCooldown > 0) {
      return { eligible: false, reason: 'On cooldown' };
    }

    // Check local drive
    const drive = this.computeLocalDrive(particle, neighbors);
    if (drive < params.drive_threshold) {
      return { eligible: false, reason: 'Drive too low' };
    }

    // Find bonds with neighbors
    const bondedNeighbors = neighbors.filter(n => {
      const bond = this.bonds.get(this.bondKey(particle.id, n.particle.id));
      return bond && bond.C > params.bond_display_threshold;
    });

    if (bondedNeighbors.length === 0) {
      return { eligible: false, reason: 'No bonds to fork' };
    }

    // Find highest stress bond
    let bestBond = null;
    let maxStress = 0;
    for (const n of bondedNeighbors) {
      const bond = this.bonds.get(this.bondKey(particle.id, n.particle.id));
      const stress = this.computeBondStress(particle, n.particle, bond);
      if (stress > maxStress) {
        maxStress = stress;
        bestBond = { neighbor: n.particle, bond, stress };
      }
    }

    // Find nearest dormant particle
    const nearestDormant = this.findNearestDormant(particle);
    if (!nearestDormant) {
      return { eligible: false, reason: 'No dormant to recruit' };
    }

    return {
      eligible: true,
      bestBond,
      recruit: nearestDormant,
      drive
    };
  }

  // Find nearest dormant particle that can be recruited
  findNearestDormant(particle) {
    let nearest = null;
    let minDist = Infinity;

    for (const dormantId of this.dormantPool) {
      const dormant = this.particles.find(p => p.id === dormantId);
      if (!dormant || !dormant.isDormant) continue;

      // Check agency gate for recruitment
      if (dormant.a < this.params.a_min_division * 0.5) continue;

      const dx = dormant.x - particle.x;
      const dy = dormant.y - particle.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < minDist && dist < this.params.interaction_range * 1.5) {
        minDist = dist;
        nearest = { particle: dormant, dist };
      }
    }

    return nearest;
  }

  // Process one replication step for all eligible particles
  processReplicationStep() {
    if (!this.replicationEnabled) return;

    const params = this.params;
    const activeParticles = this.particles.filter(p =>
      p.alive && !p.isDormant && !p.isBoundary
    );

    // Check max particles limit
    const currentActive = activeParticles.length;
    if (currentActive >= params.max_particles) return;

    // Build neighbor lists for active particles
    const neighborLists = new Map();
    for (const p of activeParticles) {
      const neighbors = [];
      for (const other of activeParticles) {
        if (other.id === p.id) continue;
        const dx = other.x - p.x;
        const dy = other.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < params.interaction_range) {
          neighbors.push({ particle: other, dist });
        }
      }
      neighborLists.set(p.id, neighbors);
    }

    // Process existing forks
    for (const [parentId, fork] of this.activeForks.entries()) {
      const parent = this.particles.find(p => p.id === parentId);
      if (!parent || !parent.alive) {
        this.activeForks.delete(parentId);
        continue;
      }

      this.processFork(parent, fork, neighborLists.get(parentId) || []);
    }

    // Check for new fork initiations (limit per step)
    let newForks = 0;
    const maxNewForks = 2;

    for (const particle of activeParticles) {
      if (newForks >= maxNewForks) break;
      if (this.activeForks.has(particle.id)) continue;

      const eligibility = this.checkForkEligibility(
        particle,
        neighborLists.get(particle.id) || []
      );

      if (eligibility.eligible) {
        // Initiate new fork
        this.activeForks.set(particle.id, {
          phase: 'opening',
          neighborId: eligibility.bestBond.neighbor.id,
          recruitId: eligibility.recruit.particle.id,
          C_start: eligibility.bestBond.bond.C,
          F_spent: 0,
          stepsOpening: 0
        });
        newForks++;
      }
    }

    // Decrement cooldowns
    for (const p of this.particles) {
      if (p.replicationCooldown > 0) {
        p.replicationCooldown--;
      }
    }
  }

  // Process a single replication fork through its phases
  processFork(parent, fork, neighbors) {
    const params = this.params;

    switch (fork.phase) {
      case 'opening': {
        // Gradually reduce coherence on the fork bond
        const bondKey = this.bondKey(parent.id, fork.neighborId);
        const bond = this.bonds.get(bondKey);

        if (!bond) {
          this.activeForks.delete(parent.id);
          return;
        }

        // Gradual reduction: ΔC = -λ_fork * C
        const deltaC = params.lambda_fork * bond.C;
        const C_new = Math.max(0.01, bond.C - deltaC);

        // Cost proportional to |ΔC|
        const cost = params.kappa_break * deltaC;
        if (parent.F < cost) {
          // Not enough resource, abort fork
          this.activeForks.delete(parent.id);
          return;
        }

        // Apply changes
        parent.F -= cost;
        fork.F_spent += cost;
        bond.C = C_new;
        fork.stepsOpening++;

        // Check if bond is now "open"
        if (C_new <= params.C_open_threshold) {
          fork.phase = 'recruiting';
        }
        break;
      }

      case 'recruiting': {
        // Activate the dormant recruit
        const recruit = this.particles.find(p => p.id === fork.recruitId);
        if (!recruit || !recruit.isDormant) {
          this.activeForks.delete(parent.id);
          return;
        }

        // Cost to form new bonds
        const cost = params.kappa_form * params.C_init * 2;
        if (parent.F < cost) {
          this.activeForks.delete(parent.id);
          return;
        }

        // Pay cost
        parent.F -= cost;
        fork.F_spent += cost;

        // ACTIVATE RECRUIT
        recruit.isDormant = false;
        recruit.P = 1.0;
        recruit.replicationCooldown = params.replication_cooldown;

        // Position recruit near parent
        const angle = Math.random() * Math.PI * 2;
        const dist = 20 + Math.random() * 20;
        recruit.x = parent.x + Math.cos(angle) * dist;
        recruit.y = parent.y + Math.sin(angle) * dist;

        // Template phase alignment (lawful coupling, not direct assignment)
        const alpha_theta = 0.1;
        const delta_theta = alpha_theta * Math.sin(parent.theta - recruit.theta);
        recruit.theta += delta_theta;

        // Give recruit some initial momentum
        recruit.px = parent.px * 0.3 + (Math.random() - 0.5) * 0.5;
        recruit.py = parent.py * 0.3 + (Math.random() - 0.5) * 0.5;

        // Transfer some resource to recruit
        const transfer = parent.F * 0.2;
        parent.F -= transfer;
        recruit.F += transfer;

        // Remove from dormant pool
        const poolIdx = this.dormantPool.indexOf(recruit.id);
        if (poolIdx !== -1) {
          this.dormantPool.splice(poolIdx, 1);
        }

        fork.phase = 'rebonding';
        break;
      }

      case 'rebonding': {
        // Form new bonds: parent-recruit and recruit-oldNeighbor
        const recruit = this.particles.find(p => p.id === fork.recruitId);
        const oldNeighbor = this.particles.find(p => p.id === fork.neighborId);

        if (!recruit || !oldNeighbor) {
          this.activeForks.delete(parent.id);
          return;
        }

        // Create bond: parent -> recruit
        const bond1Key = this.bondKey(parent.id, recruit.id);
        if (!this.bonds.has(bond1Key)) {
          this.bonds.set(bond1Key, { C: params.C_init, pi: 0, L: 0 });
        } else {
          this.bonds.get(bond1Key).C = Math.max(
            this.bonds.get(bond1Key).C,
            params.C_init
          );
        }

        // Create bond: recruit -> oldNeighbor (preserves topology)
        const bond2Key = this.bondKey(recruit.id, oldNeighbor.id);
        if (!this.bonds.has(bond2Key)) {
          this.bonds.set(bond2Key, { C: params.C_init, pi: 0, L: 0 });
        } else {
          this.bonds.get(bond2Key).C = Math.max(
            this.bonds.get(bond2Key).C,
            params.C_init
          );
        }

        // Set cooldown on parent
        parent.replicationCooldown = params.replication_cooldown;

        // Complete the fork
        fork.phase = 'complete';
        this.replicationCount++;
        break;
      }

      case 'complete': {
        // Clean up
        this.activeForks.delete(parent.id);
        break;
      }
    }
  }

  setupScenario(name) {
    const cx = this.width / 2;
    const cy = this.height / 2;

    this.bonds.clear();

    // Temporarily disable boundary for some scenarios
    const oldBoundaryEnabled = this.boundaryEnabled;

    switch (name) {
      case 'orbiting':
        this.initializeParticles();
        // Find first interior particle for central mass
        const interiorStart = this.boundaryEnabled ? this.numBoundary : 0;
        this.particles[interiorStart].x = cx;
        this.particles[interiorStart].y = cy;
        this.particles[interiorStart].q = 0.9;
        this.particles[interiorStart].F = 5.0;
        this.particles[interiorStart].px = 0;
        this.particles[interiorStart].py = 0;

        for (let i = interiorStart + 1; i < this.particles.length; i++) {
          if (this.particles[i].isBoundary) continue;
          const angle = ((i - interiorStart) / (this.particles.length - interiorStart)) * Math.PI * 2;
          const r = 60 + Math.random() * 120;
          this.particles[i].x = cx + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].q = 0.15 + Math.random() * 0.15;
          this.particles[i].theta = angle;
          const v = 2.5 / Math.sqrt(r / 60);
          this.particles[i].px = -Math.sin(angle) * v;
          this.particles[i].py = Math.cos(angle) * v;
        }
        break;

      case 'collision':
        this.initializeParticles();
        const startIdx = this.boundaryEnabled ? this.numBoundary : 0;
        const interiorCount = this.particles.length - startIdx;
        const half = Math.floor(interiorCount / 2);

        for (let i = 0; i < half; i++) {
          const p = this.particles[startIdx + i];
          if (p.isBoundary) continue;
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 50;
          p.x = cx - 120 + Math.cos(angle) * r;
          p.y = cy + Math.sin(angle) * r;
          p.px = 3;
          p.py = (Math.random() - 0.5) * 0.5;
          p.q = 0.35;
          p.theta = 0;
        }
        for (let i = half; i < interiorCount; i++) {
          const p = this.particles[startIdx + i];
          if (p.isBoundary) continue;
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 50;
          p.x = cx + 120 + Math.cos(angle) * r;
          p.y = cy + Math.sin(angle) * r;
          p.px = -3;
          p.py = (Math.random() - 0.5) * 0.5;
          p.q = 0.35;
          p.theta = Math.PI;
        }
        break;

      case 'expansion':
        this.initializeParticles();
        for (const p of this.particles) {
          if (p.isBoundary) continue;
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 30;
          p.x = cx + Math.cos(angle) * r;
          p.y = cy + Math.sin(angle) * r;
          p.q = 0.5 + Math.random() * 0.4;
          p.F = 1.5 + Math.random();
          p.theta = angle;
          const v = 1.0 + r / 15;
          p.px = Math.cos(angle) * v;
          p.py = Math.sin(angle) * v;
        }
        break;

      case 'quantum':
        this.initializeParticles();
        for (const p of this.particles) {
          if (p.isBoundary) continue;
          p.theta = 0;
          p.a = 0.95;
          p.q = 0.1;
          p.px *= 0.2;
          p.py *= 0.2;
        }
        this.params.alpha_C = 0.1;
        break;

      case 'galaxy':
        this.initializeParticles();
        const gStart = this.boundaryEnabled ? this.numBoundary : 0;
        this.particles[gStart].x = cx;
        this.particles[gStart].y = cy;
        this.particles[gStart].q = 0.95;
        this.particles[gStart].F = 8.0;
        this.particles[gStart].px = 0;
        this.particles[gStart].py = 0;

        for (let i = gStart + 1; i < this.particles.length; i++) {
          const p = this.particles[i];
          if (p.isBoundary) continue;
          const arm = (i - gStart) % 2;
          const t = ((i - gStart) / (this.particles.length - gStart)) * 3;
          const r = 50 + t * 80;
          const angle = t * Math.PI * 0.8 + arm * Math.PI + (Math.random() - 0.5) * 0.3;

          p.x = cx + Math.cos(angle) * r;
          p.y = cy + Math.sin(angle) * r;
          p.q = 0.2 + Math.random() * 0.15;
          p.theta = angle;

          const v = 2.5 / Math.sqrt(r / 50);
          p.px = -Math.sin(angle) * v;
          p.py = Math.cos(angle) * v;
        }
        break;

      case 'grace-demo':
        // Scenario specifically to demonstrate grace mechanism
        this.initializeParticles();
        for (const p of this.particles) {
          if (p.isBoundary) continue;
          // Create depleted particles in one region, rich in another
          const angle = Math.atan2(p.y - cy, p.x - cx);
          if (angle > 0 && angle < Math.PI) {
            // Upper half: depleted
            p.F = 0.1 + Math.random() * 0.2;
            p.a = 0.8; // High agency to receive grace
          } else {
            // Lower half: rich
            p.F = 2.0 + Math.random() * 1.0;
            p.a = 0.8; // High agency to donate
          }
          p.px *= 0.3;
          p.py *= 0.3;
          p.theta = angle; // Phase aligned for coherence
        }
        this.graceEnabled = true;
        break;

      case 'replication-demo':
        // Scenario to demonstrate node replication (subdivision)
        // Start with fewer particles, high resources, lots of dormant potential
        this.initializeParticles();

        // Give particles high resources and agency for replication
        for (const p of this.particles) {
          if (p.isBoundary || p.isDormant) continue;
          const angle = Math.random() * Math.PI * 2;
          const r = 50 + Math.random() * 100;
          p.x = cx + Math.cos(angle) * r;
          p.y = cy + Math.sin(angle) * r;
          p.F = 1.5 + Math.random() * 1.5; // High resource for division
          p.a = 0.6 + Math.random() * 0.3; // Good agency
          p.q = 0.2 + Math.random() * 0.2; // Some structure
          p.px = (Math.random() - 0.5) * 0.5;
          p.py = (Math.random() - 0.5) * 0.5;
          p.theta = angle;
          p.replicationCooldown = 0; // Ready to replicate
          p.recentFlux = 0.2 + Math.random() * 0.1; // Drive for replication
        }

        // Enable replication
        this.replicationEnabled = true;
        this.replicationCount = 0;
        break;

      default:
        this.initializeParticles();
    }

    for (const p of this.particles) {
      p.trail = [];
      if (!p.isBoundary) p.alive = true;
    }

    // Recreate boundary bonds
    if (this.boundaryEnabled) {
      this.createBoundaryBonds();
    }

    // Reset replication state (unless replication-demo keeps it enabled)
    if (name !== 'replication-demo') {
      this.activeForks.clear();
      this.replicationCount = 0;
    }

    this.step = 0;
    this.time = 0;
  }
}

export default DETParticleUniverse;
