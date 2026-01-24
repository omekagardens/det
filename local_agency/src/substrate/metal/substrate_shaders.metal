/**
 * EIS Substrate v2 - Metal Compute Shaders
 * =========================================
 *
 * GPU implementation of the substrate execution model.
 * Phase-based execution maps to kernel dispatch:
 *   - phase_read:    Load past trace, compute derived values
 *   - phase_propose: Emit proposals with scores
 *   - phase_choose:  Deterministic selection
 *   - phase_commit:  Apply effects, emit witnesses
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/* ==========================================================================
 * CONSTANTS (match substrate_types.h)
 * ========================================================================== */

constant uint NUM_SCALAR_REGS = 64;
constant uint NUM_REF_REGS = 32;
constant uint NUM_TOKEN_REGS = 32;
constant uint MAX_PROPOSALS = 8;
constant uint MAX_EFFECT_ARGS = 4;

/* ==========================================================================
 * TOKEN VALUES
 * ========================================================================== */

constant uint TOK_VOID = 0x0000;
constant uint TOK_ERR = 0xFFFF;
constant uint TOK_FALSE = 0x0001;
constant uint TOK_TRUE = 0x0002;
constant uint TOK_LT = 0x0010;
constant uint TOK_EQ = 0x0011;
constant uint TOK_GT = 0x0012;
constant uint TOK_OK = 0x0100;
constant uint TOK_FAIL = 0x0101;
constant uint TOK_XFER_OK = 0x0200;
constant uint TOK_XFER_PARTIAL = 0x0202;
constant uint TOK_DIFFUSE_OK = 0x0210;

/* ==========================================================================
 * REFERENCE TYPES (upper 4 bits of ref)
 * ========================================================================== */

constant uint REF_TYPE_NODE = 0x0;
constant uint REF_TYPE_BOND = 0x1;
constant uint REF_TYPE_FIELD = 0x2;
constant uint REF_TYPE_PROP = 0x3;
constant uint REF_TYPE_BUF = 0x4;
constant uint REF_TYPE_CHOICE = 0x5;

inline uint ref_make(uint type, uint id) {
    return (type << 28) | (id & 0x0FFFFFFF);
}

inline uint ref_type(uint ref) {
    return ref >> 28;
}

inline uint ref_id(uint ref) {
    return ref & 0x0FFFFFFF;
}

/* ==========================================================================
 * NODE FIELD IDs
 * ========================================================================== */

constant uint NODE_FIELD_F = 0x00;
constant uint NODE_FIELD_Q = 0x01;
constant uint NODE_FIELD_A = 0x02;
constant uint NODE_FIELD_SIGMA = 0x03;
constant uint NODE_FIELD_P = 0x04;
constant uint NODE_FIELD_TAU = 0x05;
constant uint NODE_FIELD_COS_THETA = 0x06;
constant uint NODE_FIELD_SIN_THETA = 0x07;
constant uint NODE_FIELD_K = 0x08;
constant uint NODE_FIELD_R = 0x09;
constant uint NODE_FIELD_FLAGS = 0x0A;

/* ==========================================================================
 * BOND FIELD IDs
 * ========================================================================== */

constant uint BOND_FIELD_NODE_I = 0x00;
constant uint BOND_FIELD_NODE_J = 0x01;
constant uint BOND_FIELD_C = 0x02;
constant uint BOND_FIELD_PI = 0x03;
constant uint BOND_FIELD_SIGMA = 0x04;
constant uint BOND_FIELD_FLAGS = 0x05;

/* ==========================================================================
 * EFFECT IDs
 * ========================================================================== */

constant uint EFFECT_NONE = 0x00;
constant uint EFFECT_XFER_F = 0x01;
constant uint EFFECT_DIFFUSE = 0x02;
constant uint EFFECT_SET_F = 0x03;
constant uint EFFECT_ADD_F = 0x04;
constant uint EFFECT_SET_C = 0x10;
constant uint EFFECT_ADD_C = 0x11;
constant uint EFFECT_SET_PI = 0x12;
constant uint EFFECT_INC_K = 0x0B;

/* ==========================================================================
 * OPCODE DEFINITIONS (match eis_substrate_v2.h)
 * ========================================================================== */

constant uint OP_NOP = 0x00;
constant uint OP_HALT = 0x01;
constant uint OP_YIELD = 0x02;
constant uint OP_TICK = 0x03;
constant uint OP_PHASE_R = 0x04;
constant uint OP_PHASE_P = 0x05;
constant uint OP_PHASE_C = 0x06;
constant uint OP_PHASE_X = 0x07;

constant uint OP_LDN = 0x10;
constant uint OP_LDB = 0x11;
constant uint OP_LDNB = 0x12;
constant uint OP_LDI = 0x13;
constant uint OP_LDI_F = 0x14;

constant uint OP_MOV = 0x20;
constant uint OP_MOVR = 0x21;
constant uint OP_MOVT = 0x22;
constant uint OP_TSET = 0x23;
constant uint OP_TGET = 0x24;

constant uint OP_ADD = 0x30;
constant uint OP_SUB = 0x31;
constant uint OP_MUL = 0x32;
constant uint OP_DIV = 0x33;
constant uint OP_MAD = 0x34;
constant uint OP_NEG = 0x35;
constant uint OP_ABS = 0x36;
constant uint OP_SQRT = 0x37;
constant uint OP_MIN = 0x38;
constant uint OP_MAX = 0x39;
constant uint OP_RELU = 0x3A;
constant uint OP_CLAMP = 0x3B;

constant uint OP_CMP = 0x40;
constant uint OP_CMPE = 0x41;
constant uint OP_TEQ = 0x42;
constant uint OP_TNE = 0x43;

constant uint OP_PROP_NEW = 0x50;
constant uint OP_PROP_SCORE = 0x51;
constant uint OP_PROP_EFFECT = 0x52;
constant uint OP_PROP_ARG = 0x53;
constant uint OP_PROP_END = 0x54;

constant uint OP_CHOOSE = 0x60;
constant uint OP_COMMIT = 0x61;
constant uint OP_WITNESS = 0x62;

constant uint OP_STN = 0x70;
constant uint OP_STB = 0x71;
constant uint OP_STT = 0x72;

constant uint OP_IN = 0x80;
constant uint OP_OUT = 0x81;

constant uint OP_RAND = 0xF0;
constant uint OP_SEED = 0xF1;
constant uint OP_LANE = 0xF2;
constant uint OP_TIME = 0xF3;

/* ==========================================================================
 * DATA STRUCTURES
 * ========================================================================== */

/** Decoded instruction (packed for GPU efficiency) */
struct Instruction {
    uint opcode;
    uint dst;
    uint src0;
    uint src1;
    int imm;
    uint ext;
    uint has_ext;
    uint _pad;
};

/** Proposal (matches C structure) */
struct Proposal {
    float score;
    uint effect_id;
    uint arg_count;
    uint args[MAX_EFFECT_ARGS];
    uint valid;
    uint _pad[2];
};

/** Per-lane register file */
struct LaneRegisters {
    float scalars[NUM_SCALAR_REGS];
    uint refs[NUM_REF_REGS];
    uint tokens[NUM_TOKEN_REGS];
};

/** Per-lane proposal buffer */
struct ProposalBuffer {
    Proposal proposals[MAX_PROPOSALS];
    uint count;
    uint chosen;
    uint _pad[2];
};

/** Per-lane execution state */
struct LaneState {
    uint lane_id;
    uint phase;
    uint pc;
    uint state;
    ulong seed;
    uint _pad[2];
};

/** Global execution parameters */
struct ExecutionParams {
    uint num_nodes;
    uint num_bonds;
    uint num_lanes;
    uint num_instructions;
    uint current_phase;
    uint tick;
    uint lane_mode;
    uint _pad;
};

/* ==========================================================================
 * NODE/BOND BUFFER POINTERS
 * ========================================================================== */

/** Node arrays (SoA layout for GPU efficiency) */
struct NodeBuffers {
    device float* F;
    device float* q;
    device float* a;
    device float* sigma;
    device float* P;
    device float* tau;
    device float* cos_theta;
    device float* sin_theta;
    device atomic_uint* k;
    device atomic_uint* r;
    device atomic_uint* flags;
};

/** Bond arrays (SoA layout) */
struct BondBuffers {
    device uint* node_i;
    device uint* node_j;
    device float* C;
    device float* pi;
    device float* sigma;
    device atomic_uint* flags;
};

/* ==========================================================================
 * RANDOM NUMBER GENERATOR
 * ========================================================================== */

/** MurmurHash3 32-bit finalizer */
inline uint hash_mix32(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

/** Combine values into hash */
inline uint hash_combine(uint a, uint b, uint c, uint d) {
    uint h = a;
    h = hash_mix32(h ^ b);
    h = hash_mix32(h ^ c);
    h = hash_mix32(h ^ d);
    return h;
}

/** XorShift64 PRNG */
inline ulong xorshift64(thread ulong* state) {
    ulong x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/** Get random float [0,1) */
inline float random_float(thread ulong* seed) {
    ulong r = xorshift64(seed);
    return float(r & 0xFFFFFFFF) / 4294967296.0f;
}

/** Derive lane seed from trace state */
inline ulong derive_lane_seed(uint lane_id, uint r, uint k, uint tick) {
    uint h = hash_combine(r, k, tick, lane_id);
    return ulong(h) | (ulong(hash_mix32(h + 1)) << 32);
}

/* ==========================================================================
 * FIELD ACCESS HELPERS
 * ========================================================================== */

inline float load_node_field(
    device float* F,
    device float* q,
    device float* a,
    device float* sigma,
    device float* P,
    device float* tau,
    device float* cos_theta,
    device float* sin_theta,
    device atomic_uint* k,
    device atomic_uint* r,
    uint node_id,
    uint field,
    uint num_nodes
) {
    if (node_id >= num_nodes) return 0.0f;

    switch (field) {
        case NODE_FIELD_F: return F[node_id];
        case NODE_FIELD_Q: return q[node_id];
        case NODE_FIELD_A: return a[node_id];
        case NODE_FIELD_SIGMA: return sigma[node_id];
        case NODE_FIELD_P: return P[node_id];
        case NODE_FIELD_TAU: return tau[node_id];
        case NODE_FIELD_COS_THETA: return cos_theta[node_id];
        case NODE_FIELD_SIN_THETA: return sin_theta[node_id];
        case NODE_FIELD_K: return float(atomic_load_explicit(k + node_id, memory_order_relaxed));
        case NODE_FIELD_R: return float(atomic_load_explicit(r + node_id, memory_order_relaxed));
        default: return 0.0f;
    }
}

inline void store_node_field(
    device float* F,
    device float* q,
    device float* a,
    device float* sigma,
    device float* P,
    device float* tau,
    device float* cos_theta,
    device float* sin_theta,
    device atomic_uint* k,
    uint node_id,
    uint field,
    float value,
    uint num_nodes
) {
    if (node_id >= num_nodes) return;

    switch (field) {
        case NODE_FIELD_F: F[node_id] = value; break;
        case NODE_FIELD_Q: q[node_id] = value; break;
        case NODE_FIELD_A: a[node_id] = clamp(value, 0.0f, 1.0f); break;
        case NODE_FIELD_SIGMA: sigma[node_id] = value; break;
        case NODE_FIELD_P: P[node_id] = value; break;
        case NODE_FIELD_TAU: tau[node_id] = value; break;
        case NODE_FIELD_COS_THETA: cos_theta[node_id] = value; break;
        case NODE_FIELD_SIN_THETA: sin_theta[node_id] = value; break;
        case NODE_FIELD_K: atomic_store_explicit(k + node_id, uint(value), memory_order_relaxed); break;
        default: break;
    }
}

inline float load_bond_field(
    device uint* node_i,
    device uint* node_j,
    device float* C,
    device float* pi,
    device float* sigma,
    uint bond_id,
    uint field,
    uint num_bonds
) {
    if (bond_id >= num_bonds) return 0.0f;

    switch (field) {
        case BOND_FIELD_NODE_I: return float(node_i[bond_id]);
        case BOND_FIELD_NODE_J: return float(node_j[bond_id]);
        case BOND_FIELD_C: return C[bond_id];
        case BOND_FIELD_PI: return pi[bond_id];
        case BOND_FIELD_SIGMA: return sigma[bond_id];
        default: return 0.0f;
    }
}

inline void store_bond_field(
    device float* C,
    device float* pi,
    device float* sigma,
    uint bond_id,
    uint field,
    float value,
    uint num_bonds
) {
    if (bond_id >= num_bonds) return;

    switch (field) {
        case BOND_FIELD_C: C[bond_id] = clamp(value, 0.0f, 1.0f); break;
        case BOND_FIELD_PI: pi[bond_id] = value; break;
        case BOND_FIELD_SIGMA: sigma[bond_id] = value; break;
        default: break;
    }
}

/* ==========================================================================
 * EFFECT APPLICATION (with atomic operations)
 * ========================================================================== */

/** Unpack float from uint32 */
inline float effect_unpack_float(uint u) {
    return as_type<float>(u);
}

/** Apply XFER_F effect atomically */
inline uint effect_xfer_f(
    device float* F,
    uint src_node,
    uint dst_node,
    float amount,
    uint num_nodes,
    uint lane_id,
    uint lane_mode
) {
    if (src_node >= num_nodes || dst_node >= num_nodes) return TOK_FAIL;

    // Ownership check for node-lane model
    if (lane_mode == 1) { // LANE_OWNER_NODE
        uint owner = min(src_node, dst_node);
        if (lane_id != owner) return TOK_OK; // Skip, another lane will handle
    }

    // Non-atomic for now (single lane owns)
    float available = F[src_node];
    float actual = min(amount, available);
    if (actual < 0.0f) actual = 0.0f;

    F[src_node] -= actual;
    F[dst_node] += actual;

    return (actual > 0.0f) ? TOK_XFER_OK : TOK_XFER_PARTIAL;
}

/** Apply DIFFUSE effect */
inline uint effect_diffuse(
    device float* F,
    device uint* node_i,
    device uint* node_j,
    uint bond_id,
    float delta,
    uint num_nodes,
    uint num_bonds
) {
    if (bond_id >= num_bonds) return TOK_FAIL;

    uint i = node_i[bond_id];
    uint j = node_j[bond_id];

    if (i >= num_nodes || j >= num_nodes) return TOK_FAIL;

    F[i] -= delta;
    F[j] += delta;

    return TOK_DIFFUSE_OK;
}

/* ==========================================================================
 * KERNEL: Phase Initialization
 * ========================================================================== */

kernel void init_lanes(
    device LaneState* lane_states [[buffer(0)]],
    device LaneRegisters* lane_regs [[buffer(1)]],
    device ProposalBuffer* prop_bufs [[buffer(2)]],
    constant ExecutionParams& params [[buffer(3)]],
    uint lane_id [[thread_position_in_grid]]
) {
    if (lane_id >= params.num_lanes) return;

    // Initialize lane state
    lane_states[lane_id].lane_id = lane_id;
    lane_states[lane_id].phase = params.current_phase;
    lane_states[lane_id].pc = 0;
    lane_states[lane_id].state = 0; // RUNNING
    lane_states[lane_id].seed = derive_lane_seed(lane_id, 0, 0, params.tick);

    // Clear proposal buffer
    prop_bufs[lane_id].count = 0;
    prop_bufs[lane_id].chosen = 0xFFFFFFFF;

    // Initialize refs (H0 = lane_id as node ref)
    lane_regs[lane_id].refs[0] = ref_make(REF_TYPE_NODE, lane_id);
}

/* ==========================================================================
 * KERNEL: Phase READ
 * ========================================================================== */

kernel void phase_read(
    device float* node_F [[buffer(0)]],
    device float* node_q [[buffer(1)]],
    device float* node_a [[buffer(2)]],
    device float* node_sigma [[buffer(3)]],
    device float* node_P [[buffer(4)]],
    device float* node_tau [[buffer(5)]],
    device float* node_cos_theta [[buffer(6)]],
    device float* node_sin_theta [[buffer(7)]],
    device atomic_uint* node_k [[buffer(8)]],
    device atomic_uint* node_r [[buffer(9)]],
    device uint* bond_node_i [[buffer(10)]],
    device uint* bond_node_j [[buffer(11)]],
    device float* bond_C [[buffer(12)]],
    device float* bond_pi [[buffer(13)]],
    device float* bond_sigma [[buffer(14)]],
    device LaneRegisters* lane_regs [[buffer(15)]],
    device LaneState* lane_states [[buffer(16)]],
    constant Instruction* program [[buffer(17)]],
    constant ExecutionParams& params [[buffer(18)]],
    uint lane_id [[thread_position_in_grid]]
) {
    if (lane_id >= params.num_lanes) return;

    device LaneRegisters& regs = lane_regs[lane_id];
    device LaneState& state = lane_states[lane_id];

    // Derive seed from trace state
    uint r_val = 0;
    uint k_val = 0;
    if (lane_id < params.num_nodes) {
        r_val = atomic_load_explicit(node_r + lane_id, memory_order_relaxed);
        k_val = atomic_load_explicit(node_k + lane_id, memory_order_relaxed);
    }
    state.seed = derive_lane_seed(lane_id, r_val, k_val, params.tick);

    // Execute READ phase instructions
    uint pc = 0;
    uint max_instrs = min(params.num_instructions, 1000u);

    while (pc < max_instrs) {
        constant Instruction& instr = program[pc];

        if (instr.opcode == OP_HALT || instr.opcode == OP_YIELD) break;
        if (instr.opcode == OP_PHASE_P || instr.opcode == OP_PHASE_C || instr.opcode == OP_PHASE_X) break;

        switch (instr.opcode) {
            case OP_NOP:
                break;

            case OP_LDN: {
                uint ref = regs.refs[instr.src0 & 0x1F];
                uint node_id = ref_id(ref);
                regs.scalars[instr.dst & 0x3F] = load_node_field(
                    node_F, node_q, node_a, node_sigma, node_P,
                    node_tau, node_cos_theta, node_sin_theta,
                    node_k, node_r, node_id, instr.src1, params.num_nodes);
                break;
            }

            case OP_LDB: {
                uint ref = regs.refs[instr.src0 & 0x1F];
                uint bond_id = ref_id(ref);
                regs.scalars[instr.dst & 0x3F] = load_bond_field(
                    bond_node_i, bond_node_j, bond_C, bond_pi, bond_sigma,
                    bond_id, instr.src1, params.num_bonds);
                break;
            }

            case OP_LDI:
                regs.scalars[instr.dst & 0x3F] = float(instr.imm);
                break;

            case OP_LDI_F:
                regs.scalars[instr.dst & 0x3F] = as_type<float>(instr.ext);
                break;

            case OP_MOV:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F];
                break;

            case OP_MOVR:
                if (instr.imm == 0) {
                    regs.refs[instr.dst & 0x1F] = uint(regs.scalars[instr.src0 & 0x3F]);
                } else {
                    regs.scalars[instr.dst & 0x3F] = float(regs.refs[instr.src0 & 0x1F]);
                }
                break;

            case OP_ADD:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] + regs.scalars[instr.src1 & 0x3F];
                break;

            case OP_SUB:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] - regs.scalars[instr.src1 & 0x3F];
                break;

            case OP_MUL:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] * regs.scalars[instr.src1 & 0x3F];
                break;

            case OP_DIV: {
                float b = regs.scalars[instr.src1 & 0x3F];
                regs.scalars[instr.dst & 0x3F] = (b != 0.0f) ? (regs.scalars[instr.src0 & 0x3F] / b) : 0.0f;
                break;
            }

            case OP_MAD:
                regs.scalars[instr.dst & 0x3F] += regs.scalars[instr.src0 & 0x3F] * regs.scalars[instr.src1 & 0x3F];
                break;

            case OP_NEG:
                regs.scalars[instr.dst & 0x3F] = -regs.scalars[instr.src0 & 0x3F];
                break;

            case OP_ABS:
                regs.scalars[instr.dst & 0x3F] = abs(regs.scalars[instr.src0 & 0x3F]);
                break;

            case OP_SQRT:
                regs.scalars[instr.dst & 0x3F] = sqrt(max(0.0f, regs.scalars[instr.src0 & 0x3F]));
                break;

            case OP_MIN:
                regs.scalars[instr.dst & 0x3F] = min(regs.scalars[instr.src0 & 0x3F], regs.scalars[instr.src1 & 0x3F]);
                break;

            case OP_MAX:
                regs.scalars[instr.dst & 0x3F] = max(regs.scalars[instr.src0 & 0x3F], regs.scalars[instr.src1 & 0x3F]);
                break;

            case OP_RELU:
                regs.scalars[instr.dst & 0x3F] = max(0.0f, regs.scalars[instr.src0 & 0x3F]);
                break;

            case OP_CLAMP:
                regs.scalars[instr.dst & 0x3F] = clamp(regs.scalars[instr.src0 & 0x3F], 0.0f, 1.0f);
                break;

            case OP_CMP: {
                float a = regs.scalars[instr.src0 & 0x3F];
                float b = regs.scalars[instr.src1 & 0x3F];
                regs.tokens[instr.dst & 0x1F] = (a < b) ? TOK_LT : ((a > b) ? TOK_GT : TOK_EQ);
                break;
            }

            case OP_TEQ:
                regs.tokens[instr.dst & 0x1F] = (regs.tokens[instr.src0 & 0x1F] == regs.tokens[instr.src1 & 0x1F]) ? TOK_TRUE : TOK_FALSE;
                break;

            case OP_TNE:
                regs.tokens[instr.dst & 0x1F] = (regs.tokens[instr.src0 & 0x1F] != regs.tokens[instr.src1 & 0x1F]) ? TOK_TRUE : TOK_FALSE;
                break;

            case OP_RAND: {
                ulong seed = state.seed;
                regs.scalars[instr.dst & 0x3F] = random_float(&seed);
                state.seed = seed;
                break;
            }

            case OP_LANE:
                regs.scalars[instr.dst & 0x3F] = float(lane_id);
                break;

            case OP_TIME:
                regs.scalars[instr.dst & 0x3F] = float(params.tick);
                break;

            default:
                break;
        }

        pc++;
    }

    state.pc = pc;
}

/* ==========================================================================
 * KERNEL: Phase PROPOSE
 * ========================================================================== */

kernel void phase_propose(
    device LaneRegisters* lane_regs [[buffer(0)]],
    device LaneState* lane_states [[buffer(1)]],
    device ProposalBuffer* prop_bufs [[buffer(2)]],
    constant Instruction* program [[buffer(3)]],
    constant ExecutionParams& params [[buffer(4)]],
    uint lane_id [[thread_position_in_grid]]
) {
    if (lane_id >= params.num_lanes) return;

    device LaneRegisters& regs = lane_regs[lane_id];
    device LaneState& state = lane_states[lane_id];
    device ProposalBuffer& prop_buf = prop_bufs[lane_id];

    // Clear proposal buffer
    prop_buf.count = 0;
    prop_buf.chosen = 0xFFFFFFFF;

    // Find start of PROPOSE section
    uint pc = 0;
    while (pc < params.num_instructions) {
        if (program[pc].opcode == OP_PHASE_P) {
            pc++;
            break;
        }
        pc++;
    }

    // Execute PROPOSE phase instructions
    uint max_pc = min(pc + 1000u, params.num_instructions);

    while (pc < max_pc) {
        constant Instruction& instr = program[pc];

        if (instr.opcode == OP_HALT || instr.opcode == OP_YIELD) break;
        if (instr.opcode == OP_PHASE_C || instr.opcode == OP_PHASE_X) break;

        switch (instr.opcode) {
            case OP_NOP:
                break;

            case OP_PROP_NEW: {
                if (prop_buf.count < MAX_PROPOSALS) {
                    uint idx = prop_buf.count++;
                    prop_buf.proposals[idx].score = 0.0f;
                    prop_buf.proposals[idx].effect_id = EFFECT_NONE;
                    prop_buf.proposals[idx].arg_count = 0;
                    prop_buf.proposals[idx].valid = 1;
                    regs.refs[instr.dst & 0x1F] = ref_make(REF_TYPE_PROP, idx);
                }
                break;
            }

            case OP_PROP_SCORE: {
                uint ref = regs.refs[instr.dst & 0x1F];
                uint idx = ref_id(ref);
                if (idx < prop_buf.count) {
                    prop_buf.proposals[idx].score = regs.scalars[instr.src0 & 0x3F];
                }
                break;
            }

            case OP_PROP_EFFECT: {
                uint ref = regs.refs[instr.dst & 0x1F];
                uint idx = ref_id(ref);
                if (idx < prop_buf.count) {
                    prop_buf.proposals[idx].effect_id = instr.src0;
                    if (instr.has_ext) {
                        prop_buf.proposals[idx].args[0] = instr.ext;
                        prop_buf.proposals[idx].arg_count = 1;
                    }
                }
                break;
            }

            case OP_PROP_ARG: {
                uint ref = regs.refs[instr.dst & 0x1F];
                uint idx = ref_id(ref);
                if (idx < prop_buf.count) {
                    uint arg_idx = prop_buf.proposals[idx].arg_count;
                    if (arg_idx < MAX_EFFECT_ARGS) {
                        prop_buf.proposals[idx].args[arg_idx] = as_type<uint>(regs.scalars[instr.src0 & 0x3F]);
                        prop_buf.proposals[idx].arg_count = arg_idx + 1;
                    }
                }
                break;
            }

            // Arithmetic ops (same as READ phase)
            case OP_ADD:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] + regs.scalars[instr.src1 & 0x3F];
                break;
            case OP_SUB:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] - regs.scalars[instr.src1 & 0x3F];
                break;
            case OP_MUL:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] * regs.scalars[instr.src1 & 0x3F];
                break;
            case OP_MOV:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F];
                break;
            case OP_LDI:
                regs.scalars[instr.dst & 0x3F] = float(instr.imm);
                break;
            case OP_LDI_F:
                regs.scalars[instr.dst & 0x3F] = as_type<float>(instr.ext);
                break;
            case OP_RAND: {
                ulong seed = state.seed;
                regs.scalars[instr.dst & 0x3F] = random_float(&seed);
                state.seed = seed;
                break;
            }
            case OP_CMP: {
                float a = regs.scalars[instr.src0 & 0x3F];
                float b = regs.scalars[instr.src1 & 0x3F];
                regs.tokens[instr.dst & 0x1F] = (a < b) ? TOK_LT : ((a > b) ? TOK_GT : TOK_EQ);
                break;
            }

            default:
                break;
        }

        pc++;
    }

    state.pc = pc;
}

/* ==========================================================================
 * KERNEL: Phase CHOOSE
 * ========================================================================== */

kernel void phase_choose(
    device LaneRegisters* lane_regs [[buffer(0)]],
    device LaneState* lane_states [[buffer(1)]],
    device ProposalBuffer* prop_bufs [[buffer(2)]],
    constant Instruction* program [[buffer(3)]],
    constant ExecutionParams& params [[buffer(4)]],
    uint lane_id [[thread_position_in_grid]]
) {
    if (lane_id >= params.num_lanes) return;

    device LaneRegisters& regs = lane_regs[lane_id];
    device LaneState& state = lane_states[lane_id];
    device ProposalBuffer& prop_buf = prop_bufs[lane_id];

    // Find start of CHOOSE section
    uint pc = 0;
    while (pc < params.num_instructions) {
        if (program[pc].opcode == OP_PHASE_C) {
            pc++;
            break;
        }
        pc++;
    }

    // Execute CHOOSE phase instructions
    uint max_pc = min(pc + 100u, params.num_instructions);

    while (pc < max_pc) {
        constant Instruction& instr = program[pc];

        if (instr.opcode == OP_HALT || instr.opcode == OP_YIELD) break;
        if (instr.opcode == OP_PHASE_X) break;

        switch (instr.opcode) {
            case OP_NOP:
                break;

            case OP_CHOOSE: {
                float decisiveness = regs.scalars[instr.src0 & 0x3F];

                if (prop_buf.count == 0) {
                    prop_buf.chosen = 0xFFFFFFFF;
                    regs.refs[instr.dst & 0x1F] = ref_make(REF_TYPE_CHOICE, 0xFFFFFFFF);
                    break;
                }

                // Find highest score and total
                float total_score = 0.0f;
                float max_score = -1.0f;
                uint max_idx = 0;

                for (uint i = 0; i < prop_buf.count; i++) {
                    if (prop_buf.proposals[i].valid) {
                        float score = prop_buf.proposals[i].score;
                        total_score += score;
                        if (score > max_score) {
                            max_score = score;
                            max_idx = i;
                        }
                    }
                }

                uint choice = max_idx;

                // If not fully decisive, may use weighted random
                if (decisiveness < 0.99f && total_score > 0.0f) {
                    ulong seed = state.seed;
                    float r = random_float(&seed);

                    if (r > decisiveness) {
                        float target = random_float(&seed);
                        float cumulative = 0.0f;

                        for (uint i = 0; i < prop_buf.count; i++) {
                            if (prop_buf.proposals[i].valid) {
                                cumulative += prop_buf.proposals[i].score / total_score;
                                if (target <= cumulative) {
                                    choice = i;
                                    break;
                                }
                            }
                        }
                    }

                    state.seed = seed;
                }

                prop_buf.chosen = choice;
                regs.refs[instr.dst & 0x1F] = ref_make(REF_TYPE_CHOICE, choice);
                break;
            }

            case OP_LDI:
                regs.scalars[instr.dst & 0x3F] = float(instr.imm);
                break;

            default:
                break;
        }

        pc++;
    }

    state.pc = pc;
}

/* ==========================================================================
 * KERNEL: Phase COMMIT
 * ========================================================================== */

kernel void phase_commit(
    device float* node_F [[buffer(0)]],
    device float* node_q [[buffer(1)]],
    device float* node_a [[buffer(2)]],
    device float* node_sigma [[buffer(3)]],
    device float* node_P [[buffer(4)]],
    device float* node_tau [[buffer(5)]],
    device float* node_cos_theta [[buffer(6)]],
    device float* node_sin_theta [[buffer(7)]],
    device atomic_uint* node_k [[buffer(8)]],
    device atomic_uint* node_r [[buffer(9)]],
    device uint* bond_node_i [[buffer(10)]],
    device uint* bond_node_j [[buffer(11)]],
    device float* bond_C [[buffer(12)]],
    device float* bond_pi [[buffer(13)]],
    device float* bond_sigma [[buffer(14)]],
    device LaneRegisters* lane_regs [[buffer(15)]],
    device LaneState* lane_states [[buffer(16)]],
    device ProposalBuffer* prop_bufs [[buffer(17)]],
    constant Instruction* program [[buffer(18)]],
    constant ExecutionParams& params [[buffer(19)]],
    uint lane_id [[thread_position_in_grid]]
) {
    if (lane_id >= params.num_lanes) return;

    device LaneRegisters& regs = lane_regs[lane_id];
    device LaneState& state = lane_states[lane_id];
    device ProposalBuffer& prop_buf = prop_bufs[lane_id];

    // Find start of COMMIT section
    uint pc = 0;
    while (pc < params.num_instructions) {
        if (program[pc].opcode == OP_PHASE_X) {
            pc++;
            break;
        }
        pc++;
    }

    // Execute COMMIT phase instructions
    uint max_pc = min(pc + 1000u, params.num_instructions);

    while (pc < max_pc) {
        constant Instruction& instr = program[pc];

        if (instr.opcode == OP_HALT || instr.opcode == OP_YIELD) break;

        switch (instr.opcode) {
            case OP_NOP:
                break;

            case OP_COMMIT: {
                if (prop_buf.chosen < prop_buf.count && prop_buf.proposals[prop_buf.chosen].valid) {
                    device Proposal& chosen = prop_buf.proposals[prop_buf.chosen];
                    uint result = TOK_OK;

                    switch (chosen.effect_id) {
                        case EFFECT_NONE:
                            result = TOK_OK;
                            break;

                        case EFFECT_XFER_F:
                            if (chosen.arg_count >= 3) {
                                result = effect_xfer_f(
                                    node_F,
                                    chosen.args[0],
                                    chosen.args[1],
                                    effect_unpack_float(chosen.args[2]),
                                    params.num_nodes,
                                    lane_id,
                                    params.lane_mode);
                            }
                            break;

                        case EFFECT_DIFFUSE:
                            if (chosen.arg_count >= 2) {
                                result = effect_diffuse(
                                    node_F,
                                    bond_node_i,
                                    bond_node_j,
                                    chosen.args[0],
                                    effect_unpack_float(chosen.args[1]),
                                    params.num_nodes,
                                    params.num_bonds);
                            }
                            break;

                        case EFFECT_SET_F:
                            if (chosen.arg_count >= 2 && chosen.args[0] < params.num_nodes) {
                                node_F[chosen.args[0]] = effect_unpack_float(chosen.args[1]);
                                result = TOK_OK;
                            }
                            break;

                        case EFFECT_ADD_F:
                            if (chosen.arg_count >= 2 && chosen.args[0] < params.num_nodes) {
                                node_F[chosen.args[0]] += effect_unpack_float(chosen.args[1]);
                                result = TOK_OK;
                            }
                            break;

                        case EFFECT_SET_C:
                            if (chosen.arg_count >= 2 && chosen.args[0] < params.num_bonds) {
                                bond_C[chosen.args[0]] = clamp(effect_unpack_float(chosen.args[1]), 0.0f, 1.0f);
                                result = TOK_OK;
                            }
                            break;

                        case EFFECT_INC_K:
                            if (chosen.arg_count >= 1 && chosen.args[0] < params.num_nodes) {
                                atomic_fetch_add_explicit(node_k + chosen.args[0], 1u, memory_order_relaxed);
                                result = TOK_OK;
                            }
                            break;

                        default:
                            result = TOK_FAIL;
                            break;
                    }

                    regs.tokens[instr.dst & 0x1F] = result;
                } else {
                    regs.tokens[instr.dst & 0x1F] = TOK_FAIL;
                }

                // Clear proposal buffer
                prop_buf.count = 0;
                prop_buf.chosen = 0xFFFFFFFF;
                break;
            }

            case OP_STN: {
                uint ref = regs.refs[instr.dst & 0x1F];
                uint node_id = ref_id(ref);
                store_node_field(
                    node_F, node_q, node_a, node_sigma, node_P,
                    node_tau, node_cos_theta, node_sin_theta, node_k,
                    node_id, instr.src0, regs.scalars[instr.src1 & 0x3F],
                    params.num_nodes);
                break;
            }

            case OP_STB: {
                uint ref = regs.refs[instr.dst & 0x1F];
                uint bond_id = ref_id(ref);
                store_bond_field(
                    bond_C, bond_pi, bond_sigma,
                    bond_id, instr.src0, regs.scalars[instr.src1 & 0x3F],
                    params.num_bonds);
                break;
            }

            case OP_WITNESS:
                regs.tokens[instr.dst & 0x1F] = uint(instr.imm);
                break;

            // Arithmetic (same as other phases)
            case OP_ADD:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] + regs.scalars[instr.src1 & 0x3F];
                break;
            case OP_SUB:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] - regs.scalars[instr.src1 & 0x3F];
                break;
            case OP_MUL:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F] * regs.scalars[instr.src1 & 0x3F];
                break;
            case OP_MOV:
                regs.scalars[instr.dst & 0x3F] = regs.scalars[instr.src0 & 0x3F];
                break;
            case OP_LDI:
                regs.scalars[instr.dst & 0x3F] = float(instr.imm);
                break;

            default:
                break;
        }

        pc++;
    }

    state.pc = pc;
}

/* ==========================================================================
 * KERNEL: Compute Derived Values (Presence)
 * ========================================================================== */

kernel void compute_presence(
    device float* node_F [[buffer(0)]],
    device float* node_q [[buffer(1)]],
    device float* node_P [[buffer(2)]],
    constant ExecutionParams& params [[buffer(3)]],
    uint node_id [[thread_position_in_grid]]
) {
    if (node_id >= params.num_nodes) return;

    // P = sqrt(F² + q²) - Presence magnitude
    float F = node_F[node_id];
    float q = node_q[node_id];
    node_P[node_id] = sqrt(F * F + q * q);
}
