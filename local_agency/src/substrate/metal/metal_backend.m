/**
 * EIS Substrate v2 - Metal Backend Implementation
 * ================================================
 *
 * Objective-C implementation of GPU acceleration using Metal.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/substrate_metal.h"
#include "../include/substrate_types.h"
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* ==========================================================================
 * GPU BUFFER LAYOUT STRUCTURES (must match Metal shaders)
 * ========================================================================== */

typedef struct __attribute__((packed)) {
    uint32_t opcode;
    uint32_t dst;
    uint32_t src0;
    uint32_t src1;
    int32_t imm;
    uint32_t ext;
    uint32_t has_ext;
    uint32_t _pad;
} GPUInstruction;

typedef struct __attribute__((packed)) {
    float score;
    uint32_t effect_id;
    uint32_t arg_count;
    uint32_t args[SUB_MAX_EFFECT_ARGS];
    uint32_t valid;
    uint32_t _pad[2];
} GPUProposal;

typedef struct __attribute__((packed)) {
    float scalars[SUB_NUM_SCALAR_REGS];
    uint32_t refs[SUB_NUM_REF_REGS];
    uint32_t tokens[SUB_NUM_TOKEN_REGS];
} GPULaneRegisters;

typedef struct __attribute__((packed)) {
    GPUProposal proposals[SUB_MAX_PROPOSALS];
    uint32_t count;
    uint32_t chosen;
    uint32_t _pad[2];
} GPUProposalBuffer;

typedef struct __attribute__((packed)) {
    uint32_t lane_id;
    uint32_t phase;
    uint32_t pc;
    uint32_t state;
    uint64_t seed;
    uint32_t _pad[2];
} GPULaneState;

typedef struct __attribute__((packed)) {
    uint32_t num_nodes;
    uint32_t num_bonds;
    uint32_t num_lanes;
    uint32_t num_instructions;
    uint32_t current_phase;
    uint32_t tick;
    uint32_t lane_mode;
    uint32_t _pad;
} GPUExecutionParams;

/* ==========================================================================
 * INTERNAL CONTEXT STRUCTURE
 * ========================================================================== */

@interface SubstrateMetal : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLLibrary> library;

// Compute pipelines
@property (nonatomic, strong) id<MTLComputePipelineState> initLanesPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> phaseReadPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> phaseProposePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> phaseChoosePipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> phaseCommitPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> computePresencePipeline;

// Node buffers (SoA layout)
@property (nonatomic, strong) id<MTLBuffer> nodeF;
@property (nonatomic, strong) id<MTLBuffer> nodeQ;
@property (nonatomic, strong) id<MTLBuffer> nodeA;
@property (nonatomic, strong) id<MTLBuffer> nodeSigma;
@property (nonatomic, strong) id<MTLBuffer> nodeP;
@property (nonatomic, strong) id<MTLBuffer> nodeTau;
@property (nonatomic, strong) id<MTLBuffer> nodeCosTheta;
@property (nonatomic, strong) id<MTLBuffer> nodeSinTheta;
@property (nonatomic, strong) id<MTLBuffer> nodeK;
@property (nonatomic, strong) id<MTLBuffer> nodeR;
@property (nonatomic, strong) id<MTLBuffer> nodeFlags;

// Bond buffers (SoA layout)
@property (nonatomic, strong) id<MTLBuffer> bondNodeI;
@property (nonatomic, strong) id<MTLBuffer> bondNodeJ;
@property (nonatomic, strong) id<MTLBuffer> bondC;
@property (nonatomic, strong) id<MTLBuffer> bondPi;
@property (nonatomic, strong) id<MTLBuffer> bondSigma;
@property (nonatomic, strong) id<MTLBuffer> bondFlags;

// Per-lane buffers
@property (nonatomic, strong) id<MTLBuffer> laneRegs;
@property (nonatomic, strong) id<MTLBuffer> laneStates;
@property (nonatomic, strong) id<MTLBuffer> proposalBuffers;

// Program buffer
@property (nonatomic, strong) id<MTLBuffer> programBuffer;

// Parameters buffer
@property (nonatomic, strong) id<MTLBuffer> paramsBuffer;

// Configuration
@property (nonatomic) SubstrateMetalConfig config;
@property (nonatomic) uint32_t currentNumNodes;
@property (nonatomic) uint32_t currentNumBonds;
@property (nonatomic) uint32_t currentNumInstructions;
@property (nonatomic) uint64_t currentTick;
@property (nonatomic) LaneOwnershipMode laneMode;
@property (nonatomic) uint64_t baseSeed;

// Statistics
@property (nonatomic) uint64_t totalTicks;
@property (nonatomic) uint64_t totalGpuTimeNs;

// Error state
@property (nonatomic, strong) NSString* lastError;
@property (nonatomic, strong) NSString* deviceName;

- (instancetype)initWithConfig:(const SubstrateMetalConfig*)config;
- (BOOL)createPipelines;
- (BOOL)allocateBuffers;

@end

@implementation SubstrateMetal

- (instancetype)initWithConfig:(const SubstrateMetalConfig*)config {
    self = [super init];
    if (!self) return nil;

    // Get Metal device
    if (config && config->prefer_discrete_gpu) {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        for (id<MTLDevice> dev in devices) {
            if (!dev.isLowPower) {
                _device = dev;
                break;
            }
        }
    }

    if (!_device) {
        _device = MTLCreateSystemDefaultDevice();
    }

    if (!_device) {
        _lastError = @"No Metal device available";
        return nil;
    }

    _deviceName = _device.name;

    // Store configuration
    if (config) {
        _config = *config;
    } else {
        _config = sub_metal_default_config();
    }

    // Create command queue
    _commandQueue = [_device newCommandQueue];
    if (!_commandQueue) {
        _lastError = @"Failed to create command queue";
        return nil;
    }

    // Create pipelines and buffers
    if (![self createPipelines]) {
        return nil;
    }

    if (![self allocateBuffers]) {
        return nil;
    }

    _laneMode = LANE_OWNER_NONE;
    _baseSeed = 12345678901234567ULL;

    return self;
}

- (BOOL)createPipelines {
    NSError* error = nil;
    NSFileManager* fm = [NSFileManager defaultManager];

    // Get path to this library using a function pointer
    Dl_info info;
    NSString* libraryDir = nil;
    if (dladdr((void*)sub_metal_is_available, &info) && info.dli_fname) {
        NSString* libPath = [NSString stringWithUTF8String:info.dli_fname];
        libraryDir = [libPath stringByDeletingLastPathComponent];
    }

    // Try to load precompiled metallib first
    NSArray* metallibPaths = @[
        @"substrate_shaders.metallib",
        @"../metal/substrate_shaders.metallib"
    ];

    for (NSString* relPath in metallibPaths) {
        NSString* fullPath = nil;
        if (libraryDir) {
            fullPath = [libraryDir stringByAppendingPathComponent:relPath];
        } else {
            fullPath = relPath;
        }
        if ([fm fileExistsAtPath:fullPath]) {
            NSURL* libURL = [NSURL fileURLWithPath:fullPath];
            _library = [_device newLibraryWithURL:libURL error:&error];
            if (_library) break;
        }
    }

    // Also try main bundle
    if (!_library) {
        NSString* bundlePath = [[NSBundle mainBundle] pathForResource:@"substrate_shaders" ofType:@"metallib"];
        if (bundlePath) {
            NSURL* libURL = [NSURL fileURLWithPath:bundlePath];
            _library = [_device newLibraryWithURL:libURL error:&error];
        }
    }

    // If no precompiled library, compile from source
    if (!_library) {
        NSString* shaderPath = nil;

        // Search paths for shader source
        NSArray* searchPaths = @[
            @"substrate_shaders.metal",
            @"../metal/substrate_shaders.metal",
            @"metal/substrate_shaders.metal",
            @"../substrate_shaders.metal"
        ];

        // Try relative to this library's location first (most reliable)
        if (libraryDir) {
            for (NSString* path in searchPaths) {
                NSString* fullPath = [libraryDir stringByAppendingPathComponent:path];
                if ([fm fileExistsAtPath:fullPath]) {
                    shaderPath = fullPath;
                    break;
                }
            }
        }

        // Try current working directory
        if (!shaderPath) {
            for (NSString* path in searchPaths) {
                if ([fm fileExistsAtPath:path]) {
                    shaderPath = path;
                    break;
                }
            }
        }

        // Try relative to executable
        if (!shaderPath) {
            NSString* execPath = [[NSBundle mainBundle] executablePath];
            if (execPath) {
                NSString* execDir = [execPath stringByDeletingLastPathComponent];
                for (NSString* path in searchPaths) {
                    NSString* fullPath = [execDir stringByAppendingPathComponent:path];
                    if ([fm fileExistsAtPath:fullPath]) {
                        shaderPath = fullPath;
                        break;
                    }
                }
            }
        }

        if (shaderPath) {
            NSString* source = [NSString stringWithContentsOfFile:shaderPath
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (source) {
                MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                if (@available(macOS 15.0, *)) {
                    options.mathMode = MTLMathModeFast;
                } else {
                    #pragma clang diagnostic push
                    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
                    options.fastMathEnabled = YES;
                    #pragma clang diagnostic pop
                }
                _library = [_device newLibraryWithSource:source options:options error:&error];
                if (!_library) {
                    _lastError = [NSString stringWithFormat:@"Failed to compile shaders from %@: %@",
                                  shaderPath, error.localizedDescription];
                    return NO;
                }
            } else {
                _lastError = [NSString stringWithFormat:@"Failed to read shader source from %@: %@",
                              shaderPath, error.localizedDescription];
                return NO;
            }
        } else {
            // Build a helpful error message listing where we looked
            NSMutableString* searchedPaths = [NSMutableString string];
            if (libraryDir) {
                for (NSString* path in searchPaths) {
                    [searchedPaths appendFormat:@"\n  - %@", [libraryDir stringByAppendingPathComponent:path]];
                }
            }
            for (NSString* path in searchPaths) {
                [searchedPaths appendFormat:@"\n  - %@ (cwd)", path];
            }
            _lastError = [NSString stringWithFormat:@"Shader source not found. Searched:%@", searchedPaths];
            return NO;
        }
    }

    if (!_library) {
        _lastError = [NSString stringWithFormat:@"Failed to load/compile shaders: %@",
                      error ? error.localizedDescription : @"unknown error"];
        return NO;
    }

    // Create compute pipelines
    id<MTLFunction> func;

    func = [_library newFunctionWithName:@"init_lanes"];
    if (func) {
        _initLanesPipeline = [_device newComputePipelineStateWithFunction:func error:&error];
    }

    func = [_library newFunctionWithName:@"phase_read"];
    if (func) {
        _phaseReadPipeline = [_device newComputePipelineStateWithFunction:func error:&error];
    }

    func = [_library newFunctionWithName:@"phase_propose"];
    if (func) {
        _phaseProposePipeline = [_device newComputePipelineStateWithFunction:func error:&error];
    }

    func = [_library newFunctionWithName:@"phase_choose"];
    if (func) {
        _phaseChoosePipeline = [_device newComputePipelineStateWithFunction:func error:&error];
    }

    func = [_library newFunctionWithName:@"phase_commit"];
    if (func) {
        _phaseCommitPipeline = [_device newComputePipelineStateWithFunction:func error:&error];
    }

    func = [_library newFunctionWithName:@"compute_presence"];
    if (func) {
        _computePresencePipeline = [_device newComputePipelineStateWithFunction:func error:&error];
    }

    if (!_phaseReadPipeline || !_phaseCommitPipeline) {
        _lastError = [NSString stringWithFormat:@"Failed to create pipelines: %@", error.localizedDescription];
        return NO;
    }

    return YES;
}

- (BOOL)allocateBuffers {
    MTLResourceOptions options = MTLResourceStorageModeShared;

    uint32_t maxNodes = _config.max_nodes;
    uint32_t maxBonds = _config.max_bonds;
    uint32_t maxLanes = _config.max_lanes;
    uint32_t maxInstructions = _config.max_instructions;

    // Node buffers
    _nodeF = [_device newBufferWithLength:maxNodes * sizeof(float) options:options];
    _nodeQ = [_device newBufferWithLength:maxNodes * sizeof(float) options:options];
    _nodeA = [_device newBufferWithLength:maxNodes * sizeof(float) options:options];
    _nodeSigma = [_device newBufferWithLength:maxNodes * sizeof(float) options:options];
    _nodeP = [_device newBufferWithLength:maxNodes * sizeof(float) options:options];
    _nodeTau = [_device newBufferWithLength:maxNodes * sizeof(float) options:options];
    _nodeCosTheta = [_device newBufferWithLength:maxNodes * sizeof(float) options:options];
    _nodeSinTheta = [_device newBufferWithLength:maxNodes * sizeof(float) options:options];
    _nodeK = [_device newBufferWithLength:maxNodes * sizeof(uint32_t) options:options];
    _nodeR = [_device newBufferWithLength:maxNodes * sizeof(uint32_t) options:options];
    _nodeFlags = [_device newBufferWithLength:maxNodes * sizeof(uint32_t) options:options];

    // Bond buffers
    _bondNodeI = [_device newBufferWithLength:maxBonds * sizeof(uint32_t) options:options];
    _bondNodeJ = [_device newBufferWithLength:maxBonds * sizeof(uint32_t) options:options];
    _bondC = [_device newBufferWithLength:maxBonds * sizeof(float) options:options];
    _bondPi = [_device newBufferWithLength:maxBonds * sizeof(float) options:options];
    _bondSigma = [_device newBufferWithLength:maxBonds * sizeof(float) options:options];
    _bondFlags = [_device newBufferWithLength:maxBonds * sizeof(uint32_t) options:options];

    // Per-lane buffers
    _laneRegs = [_device newBufferWithLength:maxLanes * sizeof(GPULaneRegisters) options:options];
    _laneStates = [_device newBufferWithLength:maxLanes * sizeof(GPULaneState) options:options];
    _proposalBuffers = [_device newBufferWithLength:maxLanes * sizeof(GPUProposalBuffer) options:options];

    // Program buffer
    _programBuffer = [_device newBufferWithLength:maxInstructions * sizeof(GPUInstruction) options:options];

    // Parameters buffer
    _paramsBuffer = [_device newBufferWithLength:sizeof(GPUExecutionParams) options:options];

    // Verify all allocations
    if (!_nodeF || !_nodeQ || !_nodeA || !_nodeSigma || !_nodeP ||
        !_nodeTau || !_nodeCosTheta || !_nodeSinTheta || !_nodeK || !_nodeR ||
        !_bondNodeI || !_bondNodeJ || !_bondC || !_bondPi || !_bondSigma ||
        !_laneRegs || !_laneStates || !_proposalBuffers ||
        !_programBuffer || !_paramsBuffer) {
        _lastError = @"Failed to allocate GPU buffers";
        return NO;
    }

    return YES;
}

- (void)updateParams:(uint32_t)numLanes phase:(uint32_t)phase {
    GPUExecutionParams* params = (GPUExecutionParams*)[_paramsBuffer contents];
    params->num_nodes = _currentNumNodes;
    params->num_bonds = _currentNumBonds;
    params->num_lanes = numLanes;
    params->num_instructions = _currentNumInstructions;
    params->current_phase = phase;
    params->tick = (uint32_t)_currentTick;
    params->lane_mode = _laneMode;
}

- (void)dispatchKernel:(id<MTLComputePipelineState>)pipeline
         commandBuffer:(id<MTLCommandBuffer>)cmdBuf
              numLanes:(uint32_t)numLanes {

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];

    // Calculate thread groups
    NSUInteger threadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup > 256) threadsPerGroup = 256;

    MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize gridSize = MTLSizeMake(numLanes, 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

@end

/* ==========================================================================
 * CONTEXT WRAPPER
 *
 * Important: Since this is a C struct, ARC doesn't manage the impl pointer.
 * We use CFBridgingRetain/CFBridgingRelease to manually manage the reference.
 * ========================================================================== */

struct SubstrateMetalContext {
    void* impl;  // Actually SubstrateMetal*, but stored as void* for ARC safety
};

// Helper to get the impl pointer with proper bridging
static inline SubstrateMetal* GetImpl(SubstrateMetalHandle ctx) {
    return (__bridge SubstrateMetal*)(ctx->impl);
}

/* ==========================================================================
 * C API IMPLEMENTATION
 * ========================================================================== */

SubstrateMetalConfig sub_metal_default_config(void) {
    SubstrateMetalConfig config = {0};
    config.max_nodes = 65536;
    config.max_bonds = 131072;
    config.max_lanes = 65536;
    config.max_proposals = SUB_MAX_PROPOSALS;
    config.max_instructions = 4096;
    config.enable_timestamps = false;
    config.prefer_discrete_gpu = false;
    return config;
}

int sub_metal_is_available(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil ? 1 : 0;
}

SubstrateMetalHandle sub_metal_create(void) {
    return sub_metal_create_with_config(NULL);
}

SubstrateMetalHandle sub_metal_create_with_config(const SubstrateMetalConfig* config) {
    @autoreleasepool {
        struct SubstrateMetalContext* ctx = (struct SubstrateMetalContext*)calloc(1, sizeof(struct SubstrateMetalContext));
        if (!ctx) return NULL;

        SubstrateMetal* impl = [[SubstrateMetal alloc] initWithConfig:config];
        if (!impl) {
            free(ctx);
            return NULL;
        }

        // Retain the object manually since we're storing it in a C struct
        ctx->impl = (void*)CFBridgingRetain(impl);

        return ctx;
    }
}

void sub_metal_destroy(SubstrateMetalHandle ctx) {
    if (!ctx) return;
    @autoreleasepool {
        if (ctx->impl) {
            // Ensure all Metal operations are complete before releasing
            SubstrateMetal* impl = GetImpl(ctx);
            if (impl.commandQueue) {
                // Submit an empty command buffer and wait for it to complete
                // This ensures all pending GPU work is done
                id<MTLCommandBuffer> syncBuf = [impl.commandQueue commandBuffer];
                [syncBuf commit];
                [syncBuf waitUntilCompleted];
            }
            CFBridgingRelease(ctx->impl);
            ctx->impl = NULL;
        }
        free(ctx);
    }
}

int sub_metal_upload_nodes(SubstrateMetalHandle ctx, const NodeArrays* nodes, uint32_t num_nodes) {
    if (!ctx || !ctx->impl || !nodes) return SUB_METAL_ERR_INVALID;
    SubstrateMetal* impl = GetImpl(ctx);
    if (num_nodes > impl.config.max_nodes) return SUB_METAL_ERR_OVERFLOW;

    @autoreleasepool {
        size_t fsize = num_nodes * sizeof(float);
        size_t usize = num_nodes * sizeof(uint32_t);

        if (nodes->F) memcpy([impl.nodeF contents], nodes->F, fsize);
        if (nodes->q) memcpy([impl.nodeQ contents], nodes->q, fsize);
        if (nodes->a) memcpy([impl.nodeA contents], nodes->a, fsize);
        if (nodes->sigma) memcpy([impl.nodeSigma contents], nodes->sigma, fsize);
        if (nodes->P) memcpy([impl.nodeP contents], nodes->P, fsize);
        if (nodes->tau) memcpy([impl.nodeTau contents], nodes->tau, fsize);
        if (nodes->cos_theta) memcpy([impl.nodeCosTheta contents], nodes->cos_theta, fsize);
        if (nodes->sin_theta) memcpy([impl.nodeSinTheta contents], nodes->sin_theta, fsize);
        if (nodes->k) memcpy([impl.nodeK contents], nodes->k, usize);
        if (nodes->r) memcpy([impl.nodeR contents], nodes->r, usize);
        if (nodes->flags) memcpy([impl.nodeFlags contents], nodes->flags, usize);

        impl.currentNumNodes = num_nodes;
        return SUB_METAL_OK;
    }
}

int sub_metal_upload_bonds(SubstrateMetalHandle ctx, const BondArrays* bonds, uint32_t num_bonds) {
    if (!ctx || !ctx->impl || !bonds) return SUB_METAL_ERR_INVALID;
    SubstrateMetal* impl = GetImpl(ctx);
    if (num_bonds > impl.config.max_bonds) return SUB_METAL_ERR_OVERFLOW;

    @autoreleasepool {
        size_t fsize = num_bonds * sizeof(float);
        size_t usize = num_bonds * sizeof(uint32_t);

        if (bonds->node_i) memcpy([impl.bondNodeI contents], bonds->node_i, usize);
        if (bonds->node_j) memcpy([impl.bondNodeJ contents], bonds->node_j, usize);
        if (bonds->C) memcpy([impl.bondC contents], bonds->C, fsize);
        if (bonds->pi) memcpy([impl.bondPi contents], bonds->pi, fsize);
        if (bonds->sigma) memcpy([impl.bondSigma contents], bonds->sigma, fsize);
        if (bonds->flags) memcpy([impl.bondFlags contents], bonds->flags, usize);

        impl.currentNumBonds = num_bonds;
        return SUB_METAL_OK;
    }
}

int sub_metal_upload_program(SubstrateMetalHandle ctx, const PredecodedProgram* program) {
    if (!ctx || !ctx->impl || !program || !program->instrs) return SUB_METAL_ERR_INVALID;
    SubstrateMetal* impl = GetImpl(ctx);
    if (program->count > impl.config.max_instructions) return SUB_METAL_ERR_OVERFLOW;

    @autoreleasepool {

        // Convert instructions to GPU format
        GPUInstruction* gpu_instrs = (GPUInstruction*)[impl.programBuffer contents];

        for (uint32_t i = 0; i < program->count; i++) {
            const SubstrateInstr* src = &program->instrs[i];
            GPUInstruction* dst = &gpu_instrs[i];

            dst->opcode = src->opcode;
            dst->dst = src->dst;
            dst->src0 = src->src0;
            dst->src1 = src->src1;
            dst->imm = src->imm;
            dst->ext = src->ext;
            dst->has_ext = src->has_ext ? 1 : 0;
            dst->_pad = 0;
        }

        impl.currentNumInstructions = program->count;
        return SUB_METAL_OK;
    }
}

int sub_metal_download_nodes(SubstrateMetalHandle ctx, NodeArrays* nodes, uint32_t num_nodes) {
    if (!ctx || !ctx->impl || !nodes) return SUB_METAL_ERR_INVALID;

    @autoreleasepool {
        SubstrateMetal* impl = GetImpl(ctx);
        size_t fsize = num_nodes * sizeof(float);
        size_t usize = num_nodes * sizeof(uint32_t);

        if (nodes->F) memcpy(nodes->F, [impl.nodeF contents], fsize);
        if (nodes->q) memcpy(nodes->q, [impl.nodeQ contents], fsize);
        if (nodes->a) memcpy(nodes->a, [impl.nodeA contents], fsize);
        if (nodes->sigma) memcpy(nodes->sigma, [impl.nodeSigma contents], fsize);
        if (nodes->P) memcpy(nodes->P, [impl.nodeP contents], fsize);
        if (nodes->tau) memcpy(nodes->tau, [impl.nodeTau contents], fsize);
        if (nodes->cos_theta) memcpy(nodes->cos_theta, [impl.nodeCosTheta contents], fsize);
        if (nodes->sin_theta) memcpy(nodes->sin_theta, [impl.nodeSinTheta contents], fsize);
        if (nodes->k) memcpy(nodes->k, [impl.nodeK contents], usize);
        if (nodes->r) memcpy(nodes->r, [impl.nodeR contents], usize);
        if (nodes->flags) memcpy(nodes->flags, [impl.nodeFlags contents], usize);

        return SUB_METAL_OK;
    }
}

int sub_metal_download_bonds(SubstrateMetalHandle ctx, BondArrays* bonds, uint32_t num_bonds) {
    if (!ctx || !ctx->impl || !bonds) return SUB_METAL_ERR_INVALID;

    @autoreleasepool {
        SubstrateMetal* impl = GetImpl(ctx);
        size_t fsize = num_bonds * sizeof(float);
        size_t usize = num_bonds * sizeof(uint32_t);

        if (bonds->node_i) memcpy(bonds->node_i, [impl.bondNodeI contents], usize);
        if (bonds->node_j) memcpy(bonds->node_j, [impl.bondNodeJ contents], usize);
        if (bonds->C) memcpy(bonds->C, [impl.bondC contents], fsize);
        if (bonds->pi) memcpy(bonds->pi, [impl.bondPi contents], fsize);
        if (bonds->sigma) memcpy(bonds->sigma, [impl.bondSigma contents], fsize);
        if (bonds->flags) memcpy(bonds->flags, [impl.bondFlags contents], usize);

        return SUB_METAL_OK;
    }
}

int sub_metal_execute_phase(SubstrateMetalHandle ctx, SubstratePhase phase, uint32_t num_lanes) {
    if (!ctx || !ctx->impl) return SUB_METAL_ERR_INVALID;
    SubstrateMetal* impl = GetImpl(ctx);
    if (num_lanes > impl.config.max_lanes) return SUB_METAL_ERR_OVERFLOW;

    @autoreleasepool {

        id<MTLComputePipelineState> pipeline = nil;

        switch (phase) {
            case PHASE_READ:
                pipeline = impl.phaseReadPipeline;
                break;
            case PHASE_PROPOSE:
                pipeline = impl.phaseProposePipeline;
                break;
            case PHASE_CHOOSE:
                pipeline = impl.phaseChoosePipeline;
                break;
            case PHASE_COMMIT:
                pipeline = impl.phaseCommitPipeline;
                break;
            default:
                return SUB_METAL_ERR_INVALID;
        }

        if (!pipeline) {
            impl.lastError = @"Pipeline not available for this phase";
            return SUB_METAL_ERR_EXECUTE;
        }

        // Update parameters
        [impl updateParams:num_lanes phase:phase];

        // Create command buffer
        id<MTLCommandBuffer> cmdBuf = [impl.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Set buffers based on phase
        if (phase == PHASE_READ || phase == PHASE_COMMIT) {
            // Node buffers
            [encoder setBuffer:impl.nodeF offset:0 atIndex:0];
            [encoder setBuffer:impl.nodeQ offset:0 atIndex:1];
            [encoder setBuffer:impl.nodeA offset:0 atIndex:2];
            [encoder setBuffer:impl.nodeSigma offset:0 atIndex:3];
            [encoder setBuffer:impl.nodeP offset:0 atIndex:4];
            [encoder setBuffer:impl.nodeTau offset:0 atIndex:5];
            [encoder setBuffer:impl.nodeCosTheta offset:0 atIndex:6];
            [encoder setBuffer:impl.nodeSinTheta offset:0 atIndex:7];
            [encoder setBuffer:impl.nodeK offset:0 atIndex:8];
            [encoder setBuffer:impl.nodeR offset:0 atIndex:9];

            // Bond buffers
            [encoder setBuffer:impl.bondNodeI offset:0 atIndex:10];
            [encoder setBuffer:impl.bondNodeJ offset:0 atIndex:11];
            [encoder setBuffer:impl.bondC offset:0 atIndex:12];
            [encoder setBuffer:impl.bondPi offset:0 atIndex:13];
            [encoder setBuffer:impl.bondSigma offset:0 atIndex:14];

            // Lane state
            [encoder setBuffer:impl.laneRegs offset:0 atIndex:15];
            [encoder setBuffer:impl.laneStates offset:0 atIndex:16];

            if (phase == PHASE_COMMIT) {
                [encoder setBuffer:impl.proposalBuffers offset:0 atIndex:17];
                [encoder setBuffer:impl.programBuffer offset:0 atIndex:18];
                [encoder setBuffer:impl.paramsBuffer offset:0 atIndex:19];
            } else {
                [encoder setBuffer:impl.programBuffer offset:0 atIndex:17];
                [encoder setBuffer:impl.paramsBuffer offset:0 atIndex:18];
            }
        } else {
            // PROPOSE and CHOOSE phases
            [encoder setBuffer:impl.laneRegs offset:0 atIndex:0];
            [encoder setBuffer:impl.laneStates offset:0 atIndex:1];
            [encoder setBuffer:impl.proposalBuffers offset:0 atIndex:2];
            [encoder setBuffer:impl.programBuffer offset:0 atIndex:3];
            [encoder setBuffer:impl.paramsBuffer offset:0 atIndex:4];
        }

        // Dispatch
        NSUInteger threadsPerGroup = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256u);
        MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize gridSize = MTLSizeMake(num_lanes, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.error) {
            impl.lastError = [NSString stringWithFormat:@"Execute failed: %@", cmdBuf.error.localizedDescription];
            return SUB_METAL_ERR_EXECUTE;
        }

        return SUB_METAL_OK;
    }
}

int sub_metal_execute_tick(SubstrateMetalHandle ctx, uint32_t num_lanes) {
    if (!ctx || !ctx->impl) return SUB_METAL_ERR_INVALID;
    if (num_lanes == 0) return SUB_METAL_OK;

    // Execute all four phases (each phase handles its own lane state initialization)
    SubstrateMetal* impl = GetImpl(ctx);
    int result;
    result = sub_metal_execute_phase(ctx, PHASE_READ, num_lanes);
    if (result != SUB_METAL_OK) return result;

    result = sub_metal_execute_phase(ctx, PHASE_PROPOSE, num_lanes);
    if (result != SUB_METAL_OK) return result;

    result = sub_metal_execute_phase(ctx, PHASE_CHOOSE, num_lanes);
    if (result != SUB_METAL_OK) return result;

    result = sub_metal_execute_phase(ctx, PHASE_COMMIT, num_lanes);
    if (result != SUB_METAL_OK) return result;

    // Increment tick counters
    impl.currentTick++;
    impl.totalTicks++;

    return SUB_METAL_OK;
}

int sub_metal_execute_ticks(SubstrateMetalHandle ctx, uint32_t num_lanes, uint32_t num_ticks) {
    for (uint32_t i = 0; i < num_ticks; i++) {
        int result = sub_metal_execute_tick(ctx, num_lanes);
        if (result != SUB_METAL_OK) return result;
    }
    return SUB_METAL_OK;
}

void sub_metal_synchronize(SubstrateMetalHandle ctx) {
    if (!ctx || !ctx->impl) return;
    // All our operations are already synchronous with waitUntilCompleted
}

void sub_metal_set_lane_mode(SubstrateMetalHandle ctx, LaneOwnershipMode mode) {
    if (!ctx || !ctx->impl) return;
    GetImpl(ctx).laneMode = mode;
}

void sub_metal_set_seed(SubstrateMetalHandle ctx, uint64_t seed) {
    if (!ctx || !ctx->impl) return;
    GetImpl(ctx).baseSeed = seed;
}

void sub_metal_set_tick(SubstrateMetalHandle ctx, uint64_t tick) {
    if (!ctx || !ctx->impl) return;
    GetImpl(ctx).currentTick = tick;
}

const char* sub_metal_device_name(SubstrateMetalHandle ctx) {
    if (!ctx || !ctx->impl) return "Unknown";
    return [GetImpl(ctx).deviceName UTF8String];
}

size_t sub_metal_memory_usage(SubstrateMetalHandle ctx) {
    if (!ctx || !ctx->impl) return 0;

    SubstrateMetal* impl = GetImpl(ctx);

    size_t total = 0;
    total += impl.nodeF.length + impl.nodeQ.length + impl.nodeA.length;
    total += impl.nodeSigma.length + impl.nodeP.length + impl.nodeTau.length;
    total += impl.nodeCosTheta.length + impl.nodeSinTheta.length;
    total += impl.nodeK.length + impl.nodeR.length + impl.nodeFlags.length;
    total += impl.bondNodeI.length + impl.bondNodeJ.length;
    total += impl.bondC.length + impl.bondPi.length + impl.bondSigma.length + impl.bondFlags.length;
    total += impl.laneRegs.length + impl.laneStates.length + impl.proposalBuffers.length;
    total += impl.programBuffer.length + impl.paramsBuffer.length;

    return total;
}

void sub_metal_get_stats(SubstrateMetalHandle ctx, uint64_t* out_ticks, uint64_t* out_gpu_time_ns) {
    if (!ctx || !ctx->impl) return;
    SubstrateMetal* impl = GetImpl(ctx);
    if (out_ticks) *out_ticks = impl.totalTicks;
    if (out_gpu_time_ns) *out_gpu_time_ns = impl.totalGpuTimeNs;
}

const char* sub_metal_get_error(SubstrateMetalHandle ctx) {
    if (!ctx || !ctx->impl) return "";
    SubstrateMetal* impl = GetImpl(ctx);
    if (!impl.lastError) return "";
    return [impl.lastError UTF8String];
}
