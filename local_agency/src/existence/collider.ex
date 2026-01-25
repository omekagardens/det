/**
 * ColliderCreature - Native DET Physics Simulation
 * ================================================
 *
 * A creature that manages a DET lattice collider for physics experimentation.
 * Uses the lattice substrate primitives to run DET v6.3/v6.4 physics.
 *
 * This creature validates that DET-OS can natively express the full DET
 * physics theory using agency-first semantics and phase-based execution.
 *
 * Physics Implementation:
 *   The underlying physics kernels are defined in physics.ex (Tier 4):
 *   - ComputePresenceV63: P = aσ/(1+F)/(1+H) - gravitational time dilation
 *   - DiffusiveFlux: J_diff = g·σ·(C+ε)·(1-√C)·∇F - quantum-gated diffusion
 *   - MomentumFlux: J_mom = g·π - momentum-driven transport
 *   - GravityFlux: J_grav = -g·F·∇Φ - gravitational attraction
 *   - MomentumUpdate: π += α_π·J_diff - λ_π·π + β_g·∇Φ
 *   - StructureUpdate: q += α_q·outflow - γ_q·q
 *   - AgencyUpdateV64: a → min(1-q, a_mean) - structural ceiling
 *   - GraceInjection: Emergency resource from dissipation pool
 *
 * Default Parameters (Theory Card VII.2):
 *   kappa_grav = 5.0    -- Helmholtz screening
 *   mu_grav = 2.0       -- Gravity potential scale
 *   beta_g = 10.0       -- Momentum-gravity coupling (5.0 × μ_g)
 *   lambda_pi = 0.008   -- Momentum decay
 *   alpha_pi = 0.12     -- Momentum amplification
 *   C_init = 0.15       -- Initial bond coherence
 *
 * Usage from det_os_boot.py:
 *   det> use collider
 *   det> send collider Init {dim: 1, N: 200}
 *   det> send collider AddPacket {pos: [50], mass: 8, width: 5, q: 0.3}
 *   det> send collider AddPacket {pos: [150], mass: 8, width: 5, q: 0.3}
 *   det> send collider Step {n: 100}
 *   det> send collider Query
 *   det> send collider Render
 *
 * For gravitational binding demo (requires stronger coupling):
 *   det> send collider Demo
 */

// Import DET v6.3 physics kernels
import "physics.ex";

creature ColliderCreature {
    // Lattice handle (0 = not initialized)
    var lattice_id: Register = 0;

    // Configuration
    var dim: Register = 1;
    var grid_size: Register = 100;
    var initialized: Register = 0;

    // Cached stats
    var last_mass: Register = 0;
    var last_separation: Register = 0;
    var last_step: Register = 0;

    /**
     * Init - Create a new lattice collider
     *
     * Parameters:
     *   dim: Dimensionality (1, 2, or 3)
     *   N: Grid size per dimension
     *   gravity_enabled: Enable gravity (default true)
     *   momentum_enabled: Enable momentum (default true)
     */
    kernel Init {
        in dim_param: Register;
        in size_param: Register;
        out result: TokenReg;

        phase READ {
            current_id ::= witness(self.lattice_id);
        }

        phase PROPOSE {
            // Destroy old lattice if exists
            proposal InitNew {
                score = 1.0;
                effect {
                    // Store parameters
                    self.dim = dim_param;
                    self.grid_size = size_param;

                    // Create new lattice via primitive
                    new_id ::= primitive("lattice_create", dim_param, size_param);
                    self.lattice_id = new_id;
                    self.initialized = 1;
                    self.last_step = 0;

                    result ::= witness(OK, new_id);
                }
            }
        }

        phase CHOOSE {
            choice := choose({InitNew});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * AddPacket - Add a resource packet to the lattice
     *
     * Parameters:
     *   pos: Position tuple [x] or [y,x] or [z,y,x]
     *   mass: Total resource to add
     *   width: Gaussian width
     *   momentum: Optional momentum tuple
     *   q: Initial structure (0-1)
     */
    kernel AddPacket {
        in pos: TokenReg;
        in mass: Register;
        in width: Register;
        in momentum: TokenReg;
        in initial_q: Register;
        out result: TokenReg;

        phase READ {
            lid ::= witness(self.lattice_id);
            is_init ::= witness(self.initialized);
        }

        phase PROPOSE {
            proposal AddIfInit {
                score = if_past(is_init > 0) then 1.0 else 0.0;
                effect {
                    success ::= primitive("lattice_add_packet", lid, pos, mass, width, momentum, initial_q);
                    result ::= witness(OK, success);
                }
            }

            proposal NotInit {
                score = if_past(is_init <= 0) then 1.0 else 0.0;
                effect {
                    result ::= witness(FAIL, "Lattice not initialized");
                }
            }
        }

        phase CHOOSE {
            choice := choose({AddIfInit, NotInit});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Step - Execute physics timesteps
     *
     * Parameters:
     *   n: Number of steps to execute
     */
    kernel Step {
        in num_steps: Register;
        out step_count: Register;
        out result: TokenReg;

        phase READ {
            lid ::= witness(self.lattice_id);
            is_init ::= witness(self.initialized);
        }

        phase PROPOSE {
            proposal StepIfInit {
                score = if_past(is_init > 0) then 1.0 else 0.0;
                effect {
                    count ::= primitive("lattice_step", lid, num_steps);
                    self.last_step = count;
                    step_count = count;
                    result ::= witness(OK, count);
                }
            }

            proposal NotInit {
                score = if_past(is_init <= 0) then 1.0 else 0.0;
                effect {
                    step_count = 0;
                    result ::= witness(FAIL, "Lattice not initialized");
                }
            }
        }

        phase CHOOSE {
            choice := choose({StepIfInit, NotInit});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Query - Get current lattice statistics
     *
     * Returns: stats dict with mass, separation, energy, etc.
     */
    kernel Query {
        out stats: TokenReg;
        out result: TokenReg;

        phase READ {
            lid ::= witness(self.lattice_id);
            is_init ::= witness(self.initialized);
        }

        phase PROPOSE {
            proposal QueryIfInit {
                score = if_past(is_init > 0) then 1.0 else 0.0;
                effect {
                    stats_dict ::= primitive("lattice_get_stats", lid);

                    // Cache some values
                    mass ::= primitive("lattice_total_mass", lid);
                    sep ::= primitive("lattice_separation", lid);
                    self.last_mass = mass;
                    self.last_separation = sep;

                    stats = stats_dict;
                    result ::= witness(OK);
                }
            }

            proposal NotInit {
                score = if_past(is_init <= 0) then 1.0 else 0.0;
                effect {
                    stats = "{}";
                    result ::= witness(FAIL, "Lattice not initialized");
                }
            }
        }

        phase CHOOSE {
            choice := choose({QueryIfInit, NotInit});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Render - Generate ASCII visualization
     *
     * Parameters:
     *   field: Field to render (F, q, a, P)
     *   width: Output width in characters
     */
    kernel Render {
        in field: TokenReg;
        in width: Register;
        out ascii: TokenReg;
        out result: TokenReg;

        phase READ {
            lid ::= witness(self.lattice_id);
            is_init ::= witness(self.initialized);
        }

        phase PROPOSE {
            proposal RenderIfInit {
                score = if_past(is_init > 0) then 1.0 else 0.0;
                effect {
                    rendered ::= primitive("lattice_render", lid, field, width);
                    ascii = rendered;
                    result ::= witness(OK);
                }
            }

            proposal NotInit {
                score = if_past(is_init <= 0) then 1.0 else 0.0;
                effect {
                    ascii = "[Lattice not initialized]";
                    result ::= witness(FAIL);
                }
            }
        }

        phase CHOOSE {
            choice := choose({RenderIfInit, NotInit});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * SetParam - Update a physics parameter
     *
     * Parameters:
     *   param: Parameter name (e.g., "beta_g", "gravity_enabled")
     *   value: New value
     */
    kernel SetParam {
        in param: TokenReg;
        in value: Register;
        out result: TokenReg;

        phase READ {
            lid ::= witness(self.lattice_id);
            is_init ::= witness(self.initialized);
        }

        phase PROPOSE {
            proposal SetIfInit {
                score = if_past(is_init > 0) then 1.0 else 0.0;
                effect {
                    success ::= primitive("lattice_set_param", lid, param, value);
                    result ::= witness(OK, success);
                }
            }

            proposal NotInit {
                score = if_past(is_init <= 0) then 1.0 else 0.0;
                effect {
                    result ::= witness(FAIL, "Lattice not initialized");
                }
            }
        }

        phase CHOOSE {
            choice := choose({SetIfInit, NotInit});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Destroy - Clean up lattice resources
     */
    kernel Destroy {
        out result: TokenReg;

        phase READ {
            lid ::= witness(self.lattice_id);
        }

        phase PROPOSE {
            proposal DestroyIfExists {
                score = if_past(lid > 0) then 1.0 else 0.0;
                effect {
                    success ::= primitive("lattice_destroy", lid);
                    self.lattice_id = 0;
                    self.initialized = 0;
                    result ::= witness(OK);
                }
            }

            proposal NothingToDestroy {
                score = if_past(lid <= 0) then 1.0 else 0.0;
                effect {
                    result ::= witness(OK, "No lattice to destroy");
                }
            }
        }

        phase CHOOSE {
            choice := choose({DestroyIfExists, NothingToDestroy});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Demo - Run a demonstration simulation
     *
     * Creates a 1D lattice with two packets and shows gravitational binding.
     * Uses strong coupling (beta_g=30) to demonstrate attractive dynamics.
     *
     * Physics:
     *   - Two packets with initial inward momentum
     *   - Structure (q=0.5) enables gravitational mass
     *   - Strong beta_g overcomes local gravity wells
     *   - Packets should move closer over time
     */
    kernel Demo {
        out result: TokenReg;

        phase PROPOSE {
            proposal RunDemo {
                score = 1.0;
                effect {
                    // Create lattice
                    lid ::= primitive("lattice_create", 1, 200);
                    self.lattice_id = lid;
                    self.initialized = 1;
                    self.dim = 1;
                    self.grid_size = 200;

                    // Configure for strong gravitational binding
                    // (Default beta_g=10 is too weak for visible binding)
                    primitive("lattice_set_param", lid, "beta_g", 30.0);
                    primitive("lattice_set_param", lid, "mu_grav", 3.0);

                    // Add two packets closer together with higher structure
                    // Position: 70 and 130 (separation = 60)
                    // Mass: 10 each
                    // Width: 5 (Gaussian spread)
                    // Momentum: 0.15 inward (toward center)
                    // Structure: 0.5 (high enough for strong gravity)
                    primitive("lattice_add_packet", lid, [70], 10.0, 5.0, [0.15], 0.5);
                    primitive("lattice_add_packet", lid, [130], 10.0, 5.0, [-0.15], 0.5);

                    // Get initial separation
                    sep_init ::= primitive("lattice_separation", lid);
                    mass_init ::= primitive("lattice_total_mass", lid);

                    // Run 300 steps for visible dynamics
                    primitive("lattice_step", lid, 300);

                    // Get final stats
                    sep_final ::= primitive("lattice_separation", lid);
                    mass_final ::= primitive("lattice_total_mass", lid);

                    // Render resource field
                    ascii ::= primitive("lattice_render", lid, "F", 60);

                    // Output results
                    primitive("terminal_write", "╔═══════════════════════════════════════════════════╗\n");
                    primitive("terminal_write", "║           DET v6.3 Collider Demo                  ║\n");
                    primitive("terminal_write", "╠═══════════════════════════════════════════════════╣\n");
                    primitive("terminal_write", "║ Parameters: beta_g=30, mu_grav=3, q=0.5          ║\n");
                    primitive("terminal_write", "╠═══════════════════════════════════════════════════╣\n");
                    primitive("terminal_write", "║ Initial separation: ");
                    primitive("terminal_write", sep_init);
                    primitive("terminal_write", "\n");
                    primitive("terminal_write", "║ Final separation:   ");
                    primitive("terminal_write", sep_final);
                    primitive("terminal_write", "\n");
                    primitive("terminal_write", "║ Initial mass:       ");
                    primitive("terminal_write", mass_init);
                    primitive("terminal_write", "\n");
                    primitive("terminal_write", "║ Final mass:         ");
                    primitive("terminal_write", mass_final);
                    primitive("terminal_write", "\n");
                    primitive("terminal_write", "╠═══════════════════════════════════════════════════╣\n");
                    primitive("terminal_write", "║ Resource field (F):                               ║\n");
                    primitive("terminal_write", "╚═══════════════════════════════════════════════════╝\n");
                    primitive("terminal_write", ascii);
                    primitive("terminal_write", "\n");

                    // Binding check
                    if_past(sep_final < sep_init) {
                        primitive("terminal_write", "✓ Gravitational binding: packets moved closer\n");
                    } else {
                        primitive("terminal_write", "✗ No binding (increase beta_g or decrease separation)\n");
                    }

                    self.last_separation = sep_final;
                    self.last_mass = mass_final;

                    result ::= witness(OK, "Demo complete");
                }
            }
        }

        phase CHOOSE {
            choice := choose({RunDemo});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Benchmark - Run performance benchmark
     *
     * Tests physics performance with configurable parameters.
     */
    kernel Benchmark {
        in grid_n: Register;        // Grid size (default 500)
        in num_steps: Register;     // Steps to run (default 1000)
        out ticks_per_sec: Register;
        out result: TokenReg;

        phase PROPOSE {
            proposal RunBenchmark {
                score = 1.0;
                effect {
                    // Create large lattice
                    n := if_past(grid_n > 0) then grid_n else 500;
                    lid ::= primitive("lattice_create", 1, n);
                    self.lattice_id = lid;
                    self.initialized = 1;
                    self.dim = 1;
                    self.grid_size = n;

                    // Add some packets
                    primitive("lattice_add_packet", lid, [n/4], 10.0, 10.0, [0.0], 0.3);
                    primitive("lattice_add_packet", lid, [3*n/4], 10.0, 10.0, [0.0], 0.3);

                    // Time the physics steps
                    steps := if_past(num_steps > 0) then num_steps else 1000;
                    start_time ::= primitive("time_now");

                    primitive("lattice_step", lid, steps);

                    end_time ::= primitive("time_now");
                    elapsed := end_time - start_time;

                    // Calculate performance
                    tps := steps / (elapsed + 0.001);
                    ticks_per_sec = tps;

                    // Output
                    primitive("terminal_write", "=== Benchmark Results ===\n");
                    primitive("terminal_write", "Grid size: ");
                    primitive("terminal_write", n);
                    primitive("terminal_write", "\nSteps: ");
                    primitive("terminal_write", steps);
                    primitive("terminal_write", "\nTime: ");
                    primitive("terminal_write", elapsed);
                    primitive("terminal_write", " sec\n");
                    primitive("terminal_write", "Ticks/sec: ");
                    primitive("terminal_write", tps);
                    primitive("terminal_write", "\n");

                    result ::= witness(OK, tps);
                }
            }
        }

        phase CHOOSE {
            choice := choose({RunBenchmark});
        }

        phase COMMIT {
            commit choice;
        }
    }
}

// =============================================================================
// CONSTANTS
// =============================================================================

const OK = 0x0100;
const FAIL = 0x0101;
