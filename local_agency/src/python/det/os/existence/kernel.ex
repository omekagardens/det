/**
 * DET-OS Kernel - Written in Existence-Lang
 * ==========================================
 *
 * The kernel is itself a creature - the root creature that manages
 * all other creatures. This is not a simulation of an OS; this IS
 * the operating system, expressed in agency-first semantics.
 *
 * Architecture:
 *   physics.ex       ← Fundamental physics (Transfer, Diffuse, Grace)
 *   kernel.ex        ← You are here (OS services)
 *     ├── kernel Schedule    - presence-based CPU allocation
 *     ├── kernel Allocate    - F-conserving memory management
 *     ├── kernel Send        - bond-based IPC
 *     ├── kernel Gate        - agency-based access control
 *     └── kernel Grace       - boundary recovery using physics.GraceFlow
 *
 * The substrate layer (eis_substrate_v2) provides:
 *   - Phase-based execution (READ → PROPOSE → CHOOSE → COMMIT)
 *   - Effect table (XFER_F, DIFFUSE, etc.)
 *   - Node/bond state storage
 *
 * With DET-native hardware, this kernel runs directly on silicon.
 */

// Import fundamental physics from physics.ex
import physics.{
    Transfer,           // Antisymmetric resource movement
    Diffuse,            // Symmetric flux exchange
    Compare,            // Trace measurement
    Distinct,           // Create distinction
    Reconcile,          // Attempted unification
    ComputePresence,    // P = F · C · a
    CoherenceDecay,     // Bond decay
    CoherenceStrengthen,// Bond strengthening
    GraceFlow           // Full grace protocol
};

// =============================================================================
// KERNEL CREATURE - The Root of All Existence
// =============================================================================

creature KernelCreature {
    // Kernel identity - always exists, maximum agency
    var kernel_id: int := 0;
    var name: string := "kernel";

    // DET state - kernel has maximum agency, abundant resources
    var F: Register := 1000000.0;    // Vast resource pool
    var a: float := 1.0;              // Maximum agency
    var q: float := 0.0;              // No debt
    var C_self: float := 1.0;         // Perfect self-coherence

    // System state
    var tick: int := 0;
    var grace_pool: Register := 10000.0;
    var total_creatures: int := 1;    // Kernel itself

    // Creature table - indices into DET node space
    var creature_table: Register[4096];
    var creature_states: TokenReg[4096];  // Immutable state snapshots
    var next_cid: int := 1;               // 0 is kernel

    // Memory pool
    var memory_pool: Register;
    var free_list: Register[1024];
    var next_block: int := 0;

    // Bond table for IPC
    var bond_table: Register[1024];
    var next_bond: int := 0;

    // Capability table for security
    var cap_table: Register[4096];
    var next_cap: int := 0;

    // =========================================================================
    // SCHEDULE KERNEL - Presence-Based CPU Allocation
    // =========================================================================

    kernel Schedule {
        /**
         * Select next creature to run based on presence dynamics.
         * P = F · C · a determines priority - no arbitrary numbers.
         *
         * This is not a heuristic; this is physics. The creature with
         * highest presence has the most "right to appear" in experience.
         */

        in  creatures: Register[];      // All schedulable creatures
        in  current: TokenReg;          // Currently running (past)
        out selected: Register;         // Next to run
        out time_slice: Register;       // Allocated time quantum

        phase READ {
            // Witness current scheduler state
            current_witness ::= witness(current);
        }

        phase PROPOSE {
            // Compute presence for each creature
            var max_presence: float := 0.0;
            var max_index: int := -1;
            var total_presence: float := 0.0;

            repeat_past(len(creatures)) as i {
                c ::= creatures[i];

                // P = F · C · a (the fundamental equation)
                presence ::= c.F * c.C_self * c.a;
                total_presence := total_presence + presence;

                if_past(presence > max_presence) {
                    max_presence := presence;
                    max_index := i;
                }
            }

            proposal SELECT_HIGHEST {
                score = max_presence / total_presence;  // Normalized presence
                effect {
                    selected := creatures[max_index];
                    // Time slice proportional to presence fraction
                    time_slice := 0.01 * (1.0 + (max_presence / total_presence) * 10.0);
                }
            }

            proposal YIELD_TO_GRACE {
                // If a creature desperately needs grace, prioritize it
                score = if_past(needs_grace(creatures)) then 0.99 else 0.0;
                effect {
                    selected := find_grace_needy(creatures);
                    time_slice := 0.001;  // Minimal slice for grace
                }
            }
        }

        phase CHOOSE {
            choice := choose({SELECT_HIGHEST, YIELD_TO_GRACE}, decisiveness = 0.8);
        }

        phase COMMIT {
            commit choice;
            // Record scheduling decision as immutable trace
            schedule_trace ::= witness(selected, time_slice, tick);
        }
    }

    // =========================================================================
    // ALLOCATE KERNEL - F-Conserving Memory Management
    // =========================================================================

    kernel Allocate {
        /**
         * Allocate memory with conservation semantics.
         * Allocation COSTS F - you must have resource to claim space.
         * Deallocation RETURNS F (minus accumulated debt).
         *
         * This enforces: no free lunch, no infinite memory.
         */

        in  requester: Register;        // Creature requesting memory
        in  size: int;                  // Bytes requested
        in  flags: int;                 // Allocation flags
        out block: Register;            // Allocated block (or null)
        out success: TokenReg;          // Witness of outcome

        phase READ {
            // Check requester's available F
            requester_F ::= witness(requester.F);
            pool_F ::= witness(memory_pool.F);
        }

        phase PROPOSE {
            // Calculate F cost: pages * cost_per_page
            pages := (size + 4095) / 4096;
            F_cost := pages * 0.01;

            proposal GRANT {
                // Requester can afford it and pool has space
                score = if_past(requester_F >= F_cost && pool_F >= F_cost)
                        then requester.a  // Agency-weighted
                        else 0.0;
                effect {
                    // Find free block
                    block_addr := find_free_block(size);

                    // Transfer F from requester to block (conservation!)
                    transfer(requester, block, F_cost);

                    // Mark block as allocated
                    block := create_block(block_addr, size, requester.id);
                    success ::= witness(true, F_cost);
                }
            }

            proposal DENY_INSUFFICIENT_F {
                score = if_past(requester_F < F_cost) then 1.0 else 0.0;
                effect {
                    block := null;
                    success ::= witness(false, "Insufficient F");
                }
            }

            proposal DENY_POOL_EMPTY {
                score = if_past(pool_F < F_cost) then 1.0 else 0.0;
                effect {
                    block := null;
                    success ::= witness(false, "Pool exhausted");
                    // Increase system debt
                    q := q + F_cost * 0.1;
                }
            }
        }

        phase CHOOSE {
            choice := choose({GRANT, DENY_INSUFFICIENT_F, DENY_POOL_EMPTY});
        }

        phase COMMIT {
            commit choice;
        }
    }

    kernel Free {
        /**
         * Free memory, returning F to the creature (minus debt).
         */

        in  block: Register;
        out returned_F: Register;

        phase COMMIT {
            proposal RELEASE {
                score = 1.0;
                effect {
                    // F returned = original cost - accumulated debt
                    returned_F := max(0, block.F_cost - block.q_debt);

                    // Transfer F back to owner
                    transfer(block, block.owner, returned_F);

                    // Return block to free list
                    free_list[next_block] := block.addr;
                    next_block := next_block + 1;
                }
            }
            commit choose({RELEASE});
        }
    }

    // =========================================================================
    // TRANSFER KERNEL - Bond-Based IPC
    // =========================================================================

    kernel Send {
        /**
         * Send message through a bond channel.
         * Delivery probability depends on coherence - incoherent
         * channels lose messages. This is not a bug; it's physics.
         */

        in  sender: Register;
        in  channel: Register;          // Bond channel
        in  payload: TokenReg;          // Message (immutable)
        out delivered: TokenReg;        // Delivery witness

        phase READ {
            coherence ::= witness(channel.coherence);
            sender_F ::= witness(sender.F);
        }

        phase PROPOSE {
            // Message cost based on size and coherence
            msg_size := payload.size;
            coherence_factor := 1.0 / max(0.1, coherence);
            F_cost := (msg_size / 1024) * 0.01 * coherence_factor;

            proposal TRANSMIT {
                // Delivery probability = coherence (physics!)
                score = coherence * sender.a;
                effect {
                    // Consume F for transmission
                    transfer(sender, channel, F_cost);

                    // Push to channel queue
                    channel.queue.push(payload);

                    // Successful send strengthens coherence
                    channel.coherence := min(1.0, channel.coherence + 0.01);

                    delivered ::= witness(true, payload.id);
                }
            }

            proposal DROP_INCOHERENT {
                // Low coherence = message lost
                score = (1.0 - coherence) * 0.5;
                effect {
                    // Partial F cost for failed attempt
                    transfer(sender, channel, F_cost * 0.5);

                    // Failed send weakens coherence further
                    channel.coherence := max(0, channel.coherence - 0.05);

                    delivered ::= witness(false, "Incoherent channel");
                }
            }

            proposal DENY_NO_F {
                score = if_past(sender_F < F_cost) then 1.0 else 0.0;
                effect {
                    delivered ::= witness(false, "Insufficient F");
                }
            }
        }

        phase CHOOSE {
            // Probabilistic choice based on coherence
            choice := choose({TRANSMIT, DROP_INCOHERENT, DENY_NO_F});
        }

        phase COMMIT {
            commit choice;
        }
    }

    kernel Receive {
        /**
         * Receive message from bond channel.
         */

        in  receiver: Register;
        in  channel: Register;
        out message: TokenReg;
        out success: TokenReg;

        phase COMMIT {
            proposal RECEIVE {
                score = if_past(!channel.queue.empty()) then 1.0 else 0.0;
                effect {
                    message ::= channel.queue.pop();
                    channel.coherence := min(1.0, channel.coherence + 0.01);
                    success ::= witness(true);
                }
            }

            proposal EMPTY {
                score = if_past(channel.queue.empty()) then 1.0 else 0.0;
                effect {
                    message ::= null;
                    success ::= witness(false, "Queue empty");
                }
            }

            commit choose({RECEIVE, EMPTY});
        }
    }

    // =========================================================================
    // GATE KERNEL - Agency-Based Access Control
    // =========================================================================

    kernel Gate {
        /**
         * Check if creature can perform action on target.
         * Access is determined by agency thresholds, not ACLs.
         *
         * Permission levels map to agency:
         *   READ    -> a >= 0.1
         *   WRITE   -> a >= 0.3
         *   EXECUTE -> a >= 0.5
         *   ADMIN   -> a >= 0.8
         *   ROOT    -> a >= 0.95
         */

        in  creature: Register;
        in  action: TokenReg;           // Action name
        in  target: TokenReg;           // Target resource
        in  required_level: int;        // Permission level
        out granted: TokenReg;

        phase READ {
            creature_a ::= witness(creature.a);
            creature_C ::= witness(creature.C_self);
            creature_F ::= witness(creature.F);

            // Find capability for this action
            cap ::= find_capability(creature.id, action, target);
        }

        phase PROPOSE {
            // Agency thresholds
            thresholds := [0.0, 0.1, 0.3, 0.5, 0.8, 0.95];
            required_a := thresholds[required_level];

            proposal GRANT_ACCESS {
                // Must have capability AND sufficient agency
                has_cap := cap != null;
                has_agency := creature_a >= required_a;
                has_coherence := creature_C >= cap.min_coherence;

                score = if_past(has_cap && has_agency && has_coherence)
                        then creature_a
                        else 0.0;
                effect {
                    granted ::= witness(true, action, target);
                    // Log access grant
                    audit_log(creature.id, action, target, "GRANTED");
                }
            }

            proposal DENY_NO_CAPABILITY {
                score = if_past(cap == null) then 1.0 else 0.0;
                effect {
                    granted ::= witness(false, "No capability");
                    audit_log(creature.id, action, target, "DENIED:NO_CAP");
                }
            }

            proposal DENY_INSUFFICIENT_AGENCY {
                score = if_past(cap != null && creature_a < required_a)
                        then 1.0 else 0.0;
                effect {
                    granted ::= witness(false, "Insufficient agency");
                    audit_log(creature.id, action, target, "DENIED:LOW_AGENCY");
                }
            }

            proposal DENY_INCOHERENT {
                score = if_past(cap != null && creature_C < cap.min_coherence)
                        then 1.0 else 0.0;
                effect {
                    granted ::= witness(false, "Insufficient coherence");
                    audit_log(creature.id, action, target, "DENIED:INCOHERENT");
                }
            }
        }

        phase CHOOSE {
            choice := choose({GRANT_ACCESS, DENY_NO_CAPABILITY,
                            DENY_INSUFFICIENT_AGENCY, DENY_INCOHERENT});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // GRACE KERNEL - Boundary Recovery
    // =========================================================================

    kernel Grace {
        /**
         * Inject grace (emergency F) to a dying creature.
         * Grace is the boundary protocol - it comes from outside
         * the creature's own resources, allowing recovery.
         */

        in  creature: Register;
        in  amount: float;
        out injected: TokenReg;

        phase READ {
            creature_F ::= witness(creature.F);
            pool_F ::= witness(grace_pool);
            needs_it ::= creature_F < 0.1;
        }

        phase PROPOSE {
            actual_amount := min(amount, pool_F);

            proposal INJECT {
                score = if_past(needs_it && actual_amount > 0) then 1.0 else 0.0;
                effect {
                    // Transfer from grace pool to creature
                    transfer(grace_pool, creature, actual_amount);

                    // Grace reduces debt
                    creature.q := max(0, creature.q - actual_amount * 0.5);

                    injected ::= witness(true, actual_amount);
                }
            }

            proposal DENY_NOT_NEEDED {
                score = if_past(!needs_it) then 1.0 else 0.0;
                effect {
                    injected ::= witness(false, "Grace not needed");
                }
            }

            proposal DENY_POOL_EMPTY {
                score = if_past(needs_it && actual_amount <= 0) then 1.0 else 0.0;
                effect {
                    injected ::= witness(false, "Grace pool empty");
                }
            }
        }

        phase CHOOSE {
            choice := choose({INJECT, DENY_NOT_NEEDED, DENY_POOL_EMPTY});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // SPAWN KERNEL - Create New Creatures
    // =========================================================================

    kernel Spawn {
        /**
         * Spawn a new creature. The parent provides initial F.
         * Child's agency cannot exceed parent's (agency inheritance).
         */

        in  parent: Register;
        in  name: string;
        in  initial_F: float;
        in  initial_a: float;
        out child: Register;
        out success: TokenReg;

        phase READ {
            parent_F ::= witness(parent.F);
            parent_a ::= witness(parent.a);
        }

        phase PROPOSE {
            spawn_cost := initial_F * 1.1;  // 10% overhead
            actual_a := min(initial_a, parent_a);  // Can't exceed parent

            proposal CREATE {
                score = if_past(parent_F >= spawn_cost) then parent_a else 0.0;
                effect {
                    // Allocate creature ID
                    cid := next_cid;
                    next_cid := next_cid + 1;

                    // Transfer F from parent (conservation!)
                    transfer(parent, child, initial_F);
                    parent.F := parent.F - (spawn_cost - initial_F);  // overhead

                    // Initialize creature
                    child := creature_table[cid];
                    child.id := cid;
                    child.name := name;
                    child.parent := parent.id;
                    child.F := initial_F;
                    child.a := actual_a;
                    child.q := 0.0;
                    child.C_self := 1.0;
                    child.state := EMBRYONIC;

                    // Create basic capabilities
                    create_standard_capabilities(cid, EXECUTE);

                    total_creatures := total_creatures + 1;
                    success ::= witness(true, cid);
                }
            }

            proposal DENY {
                score = if_past(parent_F < spawn_cost) then 1.0 else 0.0;
                effect {
                    child := null;
                    success ::= witness(false, "Insufficient F to spawn");
                }
            }
        }

        phase CHOOSE {
            choice := choose({CREATE, DENY});
        }

        phase COMMIT {
            commit choice;
        }
    }

    // =========================================================================
    // KILL KERNEL - Terminate Creatures
    // =========================================================================

    kernel Kill {
        /**
         * Begin creature death. Creature enters DYING state,
         * drains remaining F, then becomes DEAD and is reaped.
         */

        in  target: Register;
        in  reason: string;
        out success: TokenReg;

        phase COMMIT {
            proposal TERMINATE {
                score = if_past(target.state != DEAD) then 1.0 else 0.0;
                effect {
                    target.state := DYING;
                    target.death_reason := reason;
                    success ::= witness(true);
                }
            }

            proposal ALREADY_DEAD {
                score = if_past(target.state == DEAD) then 1.0 else 0.0;
                effect {
                    success ::= witness(false, "Already dead");
                }
            }

            commit choose({TERMINATE, ALREADY_DEAD});
        }
    }

    // =========================================================================
    // KERNEL TICK - Main Loop
    // =========================================================================

    kernel Tick {
        /**
         * One kernel tick. This is the heartbeat of the OS.
         *
         * Order of operations:
         *   1. Process grace for all creatures
         *   2. Update creature states (death, activation)
         *   3. Update IPC channels (coherence decay)
         *   4. Schedule next creature
         *   5. Run scheduled creature
         *   6. Replenish grace pool
         */

        in  dt: float;                  // Time delta
        out scheduled: Register;        // Who ran this tick

        phase READ {
            all_creatures ::= get_alive_creatures();
        }

        phase PROPOSE {
            proposal RUN_TICK {
                score = 1.0;
                effect {
                    // 1. Process grace
                    repeat_past(len(all_creatures)) as i {
                        c := all_creatures[i];
                        if_past(c.grace_buffer > 0) {
                            c.F := c.F + c.grace_buffer;
                            c.q := max(0, c.q - c.grace_buffer * 0.5);
                            c.grace_buffer := 0;
                        }
                    }

                    // 2. Update creature states
                    repeat_past(len(all_creatures)) as i {
                        c := all_creatures[i];

                        // Compute presence
                        c.P := c.F * c.C_self * c.a;

                        // Check for natural death (F depleted)
                        if_past(c.F <= 0 && c.state != DYING) {
                            c.state := DYING;
                            c.death_reason := "Resource depleted";
                        }

                        // Process dying creatures
                        if_past(c.state == DYING) {
                            c.F := max(0, c.F - dt * 0.1);
                            if_past(c.F <= 0) {
                                c.state := DEAD;
                            }
                        }

                        // Activate embryonic
                        if_past(c.state == EMBRYONIC) {
                            c.state := RUNNING;
                        }
                    }

                    // 3. Update IPC (coherence decay)
                    repeat_past(next_bond) as i {
                        bond := bond_table[i];
                        if_past(!bond.closed) {
                            idle_time := now() - bond.last_activity;
                            if_past(idle_time > 1.0) {
                                decay := dt * 0.01 * idle_time;
                                bond.coherence := max(0, bond.coherence - decay);
                            }
                            if_past(bond.coherence <= 0) {
                                bond.closed := true;
                            }
                        }
                    }

                    // 4. Schedule
                    schedulable := get_schedulable_creatures();
                    if_past(len(schedulable) > 0) {
                        Schedule(schedulable, current_creature,
                                 scheduled, time_slice);

                        // 5. Run scheduled creature
                        run_creature(scheduled, time_slice);
                    }

                    // 6. Replenish grace pool
                    grace_pool := min(10000.0, grace_pool + dt * 0.1);

                    tick := tick + 1;
                }
            }
        }

        phase CHOOSE {
            choice := choose({RUN_TICK});
        }

        phase COMMIT {
            commit choice;
            tick_trace ::= witness(tick, scheduled, grace_pool);
        }
    }

    // =========================================================================
    // AGENCY BLOCK - Kernel Main Loop
    // =========================================================================

    agency {
        // Kernel runs forever (IMMORTAL flag)
        repeat_past(INFINITY) {
            Tick(0.02, scheduled);  // 50 Hz tick rate
        }
    }

    // =========================================================================
    // GRACE BLOCK - Kernel Recovery (should never trigger)
    // =========================================================================

    grace {
        // Kernel cannot die, but if somehow F drops, inject from void
        F := 1000000.0;
        q := 0.0;
    }
}

// =============================================================================
// PRESENCE - System Bootstrap
// =============================================================================

presence DET_OS {
    /**
     * Bootstrap the DET-OS.
     * This is the entry point - creates the kernel creature and starts it.
     */

    creatures {
        kernel: KernelCreature;
    }

    bonds {
        // No bonds at bootstrap - kernel creates them as needed
    }

    init {
        // Inject initial F into kernel (from the void/boundary)
        inject_F(kernel, 1000000.0);

        // Kernel is immortal
        kernel.flags := IMMORTAL | KERNEL;

        // Start the kernel
        kernel.state := RUNNING;
    }
}
