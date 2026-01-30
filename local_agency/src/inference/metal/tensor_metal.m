/**
 * DET Inference - Metal Backend Bridge
 * =====================================
 *
 * Objective-C implementation for Metal tensor operations.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "../include/det_tensor.h"

/* ==========================================================================
 * METAL CONTEXT
 * ========================================================================== */

@interface TensorMetalContext : NSObject

@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLLibrary> library;

// Batch mode - accumulate operations in single command buffer
@property (nonatomic, strong) id<MTLCommandBuffer> batchCommandBuffer;
@property (nonatomic, assign) BOOL batchModeActive;
@property (nonatomic, assign) int batchOperationCount;

// Deferred result collection for batch mode
// When in batch mode, outputs are stored in temp GPU buffers and copied at end_batch
@property (nonatomic, strong) NSMutableArray<id<MTLBuffer>> *deferredOutputBuffers;
@property (nonatomic, strong) NSMutableArray<NSValue*> *deferredOutputDestinations;
@property (nonatomic, strong) NSMutableArray<NSNumber*> *deferredOutputSizes;

// Scratch buffers for avoiding per-call allocation
@property (nonatomic, strong) id<MTLBuffer> scratchInput;   // For input activations
@property (nonatomic, strong) id<MTLBuffer> scratchOutput;  // For output results
@property (nonatomic, assign) size_t scratchInputSize;
@property (nonatomic, assign) size_t scratchOutputSize;

// Compute pipelines
@property (nonatomic, strong) id<MTLComputePipelineState> matmulPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> matmulTransposedBPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> matvecPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> fusedGateUpPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> fusedRMSNormMatmulPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> fusedSwiGLUDownPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> matvecF32Pipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> addPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> mulPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> siluPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> siluMulPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> geluPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> reluPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> rmsnormPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> softmaxPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> attentionScoresPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> causalMaskPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> ropePipeline;

// GPU-native attention pipelines
@property (nonatomic, strong) id<MTLComputePipelineState> attentionScoresCausalGpuPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> softmaxRowsGpuPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> attentionWeightedSumGpuPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> ropeGpuPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> kvCacheStoreGpuPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> addInplaceGpuPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> scaleInplaceGpuPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> attentionMultiheadScoresPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> attentionMultiheadSoftmaxPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> attentionMultiheadWeightedSumPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> diffAttnScoresPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> diffAttnSoftmaxPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> diffAttnWeightedSumPipeline;

// GPU KV cache buffers
@property (nonatomic, strong) id<MTLBuffer> gpuKCache;    // [n_layer, n_ctx, kv_dim]
@property (nonatomic, strong) id<MTLBuffer> gpuVCache;    // [n_layer, n_ctx, kv_dim]
@property (nonatomic, assign) uint32_t kvCacheNLayer;
@property (nonatomic, assign) uint32_t kvCacheNCtx;
@property (nonatomic, assign) uint32_t kvCacheKvDim;
@property (nonatomic, assign) uint32_t kvCacheSeqLen;

// GPU-native forward pass scratch buffers
@property (nonatomic, strong) id<MTLBuffer> gpuQ;         // [max_seq, n_embd]
@property (nonatomic, strong) id<MTLBuffer> gpuK;         // [max_seq, kv_dim]
@property (nonatomic, strong) id<MTLBuffer> gpuV;         // [max_seq, kv_dim]
@property (nonatomic, strong) id<MTLBuffer> gpuAttnScores; // [max_seq, n_ctx]
@property (nonatomic, strong) id<MTLBuffer> gpuAttnOut;   // [max_seq, kv_dim]
@property (nonatomic, assign) BOOL gpuForwardInitialized;

// Persistent GPU hidden state buffers (never reallocated during forward pass)
@property (nonatomic, strong) id<MTLBuffer> gpuHidden;    // [max_seq, n_embd] - main hidden state
@property (nonatomic, strong) id<MTLBuffer> gpuResidual;  // [max_seq, n_embd] - residual connection
@property (nonatomic, strong) id<MTLBuffer> gpuNormed;    // [max_seq, n_embd] - post-norm hidden
@property (nonatomic, strong) id<MTLBuffer> gpuFFNGate;   // [max_seq, n_ff] - FFN gate projection
@property (nonatomic, strong) id<MTLBuffer> gpuFFNUp;     // [max_seq, n_ff] - FFN up projection
@property (nonatomic, strong) id<MTLBuffer> gpuFFNDown;   // [max_seq, n_embd] - FFN down projection
@property (nonatomic, assign) uint32_t gpuHiddenNEmbd;
@property (nonatomic, assign) uint32_t gpuHiddenNFF;
@property (nonatomic, assign) uint32_t gpuHiddenMaxSeq;

- (instancetype)init;
- (BOOL)loadShaders;

@end

@implementation TensorMetalContext

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            NSLog(@"TensorMetal: No Metal device available");
            return nil;
        }

        _commandQueue = [_device newCommandQueue];
        if (!_commandQueue) {
            NSLog(@"TensorMetal: Failed to create command queue");
            return nil;
        }

        if (![self loadShaders]) {
            NSLog(@"TensorMetal: Failed to load shaders");
            return nil;
        }

        NSLog(@"TensorMetal: Initialized with device %@", _device.name);
    }
    return self;
}

- (BOOL)loadShaders {
    NSError *error = nil;
    NSString *shaderPath = nil;
    NSString *source = nil;

    // Try to load precompiled metallib
    NSString *libPath = [[NSBundle mainBundle] pathForResource:@"tensor_shaders"
                                                        ofType:@"metallib"];
    if (libPath) {
        NSURL *libURL = [NSURL fileURLWithPath:libPath];
        _library = [_device newLibraryWithURL:libURL error:&error];
    }

    // Fall back to runtime compilation - try multiple paths
    if (!_library) {
        NSArray *searchPaths = @[
            // 1. Same directory as the dynamic library (for ctypes loading)
            @"/Volumes/AI_DATA/development/det_local_agency/det/local_agency/src/inference/build/tensor_shaders.metal",
            // 2. Relative to executable
            [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent],
            // 3. Current working directory
            @"tensor_shaders.metal",
            // 4. Source directory (development fallback)
            @"src/inference/metal/tensor_shaders.metal",
            @"src/inference/build/tensor_shaders.metal",
        ];

        for (NSString *path in searchPaths) {
            NSString *fullPath = path;
            if (![path hasSuffix:@".metal"]) {
                fullPath = [path stringByAppendingPathComponent:@"tensor_shaders.metal"];
            }

            source = [NSString stringWithContentsOfFile:fullPath
                                               encoding:NSUTF8StringEncoding
                                                  error:NULL];
            if (source) {
                shaderPath = fullPath;
                break;
            }
        }

        if (source) {
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            options.mathMode = MTLMathModeFast;
            _library = [_device newLibraryWithSource:source options:options error:&error];
        }
    }

    if (!_library) {
        NSLog(@"TensorMetal: Failed to load shader library: %@", error);
        return NO;
    }

    // Create compute pipelines
    _matmulPipeline = [self createPipeline:@"matmul_f32"];
    _matmulTransposedBPipeline = [self createPipeline:@"matmul_transposed_b_f32"];
    _matvecPipeline = [self createPipeline:@"matvec_f32"];
    _addPipeline = [self createPipeline:@"add_f32"];
    _mulPipeline = [self createPipeline:@"mul_f32"];
    _siluPipeline = [self createPipeline:@"silu_f32"];
    _siluMulPipeline = [self createPipeline:@"silu_mul_f32"];
    _geluPipeline = [self createPipeline:@"gelu_f32"];
    _reluPipeline = [self createPipeline:@"relu_f32"];
    _rmsnormPipeline = [self createPipeline:@"rmsnorm_f32"];
    _softmaxPipeline = [self createPipeline:@"softmax_f32"];
    _attentionScoresPipeline = [self createPipeline:@"attention_scores_f32"];
    _causalMaskPipeline = [self createPipeline:@"causal_mask_f32"];
    _ropePipeline = [self createPipeline:@"rope_f32"];

    // Fused kernels (Phase 26.18)
    _fusedGateUpPipeline = [self createPipeline:@"fused_gate_up_proj_f32"];
    _fusedRMSNormMatmulPipeline = [self createPipeline:@"fused_rmsnorm_matmul_f32"];
    _fusedSwiGLUDownPipeline = [self createPipeline:@"fused_swiglu_down_f32"];
    _matvecF32Pipeline = [self createPipeline:@"matvec_transposed_f32"];

    // GPU-native attention pipelines
    _attentionScoresCausalGpuPipeline = [self createPipeline:@"attention_scores_causal_gpu"];
    _softmaxRowsGpuPipeline = [self createPipeline:@"softmax_rows_gpu"];
    _attentionWeightedSumGpuPipeline = [self createPipeline:@"attention_weighted_sum_gpu"];
    _ropeGpuPipeline = [self createPipeline:@"rope_gpu"];
    _kvCacheStoreGpuPipeline = [self createPipeline:@"kv_cache_store_gpu"];
    _addInplaceGpuPipeline = [self createPipeline:@"add_inplace_gpu"];
    _scaleInplaceGpuPipeline = [self createPipeline:@"scale_inplace_gpu"];

    // Multi-head attention with GQA support
    _attentionMultiheadScoresPipeline = [self createPipeline:@"attention_multihead_scores_gpu"];
    _attentionMultiheadSoftmaxPipeline = [self createPipeline:@"attention_multihead_softmax_gpu"];
    _attentionMultiheadWeightedSumPipeline = [self createPipeline:@"attention_multihead_weighted_sum_gpu"];

    // Differential attention (phi4flash)
    _diffAttnScoresPipeline = [self createPipeline:@"diff_attn_scores_gpu"];
    _diffAttnSoftmaxPipeline = [self createPipeline:@"diff_attn_softmax_gpu"];
    _diffAttnWeightedSumPipeline = [self createPipeline:@"diff_attn_weighted_sum_gpu"];

    return _matmulPipeline != nil;
}

- (id<MTLComputePipelineState>)createPipeline:(NSString *)name {
    NSError *error = nil;
    id<MTLFunction> function = [_library newFunctionWithName:name];
    if (!function) {
        NSLog(@"TensorMetal: Function %@ not found", name);
        return nil;
    }

    id<MTLComputePipelineState> pipeline =
        [_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        NSLog(@"TensorMetal: Failed to create pipeline for %@: %@", name, error);
    }
    return pipeline;
}

@end

/* ==========================================================================
 * C INTERFACE
 * ========================================================================== */

static TensorMetalContext *g_metal_ctx = nil;

// Initialize Metal context
int tensor_metal_init(void) {
    if (g_metal_ctx) return 0;  // Already initialized

    @autoreleasepool {
        g_metal_ctx = [[TensorMetalContext alloc] init];
        return g_metal_ctx ? 0 : -1;
    }
}

// Check if Metal is available
int tensor_metal_available(void) {
    if (g_metal_ctx) return 1;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device ? 1 : 0;
    }
}

// Get device name
const char* tensor_metal_device_name(void) {
    static char name[256] = {0};
    if (!g_metal_ctx) {
        return "No Metal device";
    }

    @autoreleasepool {
        NSString *deviceName = g_metal_ctx.device.name;
        strncpy(name, [deviceName UTF8String], sizeof(name) - 1);
        return name;
    }
}

// Initialize scratch buffers for efficient inference
// Call after model loading with known dimensions
int tensor_metal_init_scratch(uint32_t max_input_size, uint32_t max_output_size) {
    if (!g_metal_ctx) return -1;

    @autoreleasepool {
        // Free existing scratch buffers if any
        g_metal_ctx.scratchInput = nil;
        g_metal_ctx.scratchOutput = nil;

        // Allocate input scratch (for activations copied to GPU)
        size_t input_bytes = max_input_size * sizeof(float);
        g_metal_ctx.scratchInput = [g_metal_ctx.device newBufferWithLength:input_bytes
                                                                   options:MTLResourceStorageModeShared];
        if (!g_metal_ctx.scratchInput) {
            NSLog(@"TensorMetal: Failed to allocate input scratch buffer (%zu bytes)", input_bytes);
            return -1;
        }
        g_metal_ctx.scratchInputSize = input_bytes;

        // Allocate output scratch (for results copied from GPU)
        size_t output_bytes = max_output_size * sizeof(float);
        g_metal_ctx.scratchOutput = [g_metal_ctx.device newBufferWithLength:output_bytes
                                                                    options:MTLResourceStorageModeShared];
        if (!g_metal_ctx.scratchOutput) {
            NSLog(@"TensorMetal: Failed to allocate output scratch buffer (%zu bytes)", output_bytes);
            g_metal_ctx.scratchInput = nil;
            return -1;
        }
        g_metal_ctx.scratchOutputSize = output_bytes;

        NSLog(@"TensorMetal: Initialized scratch buffers (input=%zu, output=%zu bytes)",
              input_bytes, output_bytes);
        return 0;
    }
}

// Free scratch buffers
void tensor_metal_free_scratch(void) {
    if (!g_metal_ctx) return;

    @autoreleasepool {
        g_metal_ctx.scratchInput = nil;
        g_metal_ctx.scratchOutput = nil;
        g_metal_ctx.scratchInputSize = 0;
        g_metal_ctx.scratchOutputSize = 0;
    }
}

// =============================================================================
// BATCH MODE - Accumulate operations in single command buffer (Phase 26.18)
// =============================================================================

// Forward declarations for GPU forward pass functions (defined later)
int tensor_metal_init_gpu_forward(uint32_t n_layer, uint32_t n_ctx,
                                   uint32_t n_embd, uint32_t n_ff,
                                   uint32_t n_head, uint32_t n_head_kv,
                                   uint32_t n_vocab);
void tensor_metal_free_gpu_forward(void);

// Initialize persistent GPU buffers for forward pass
// Note: This is a legacy API - redirects to tensor_metal_init_gpu_forward
int tensor_metal_init_forward_buffers(uint32_t max_seq, uint32_t d_model,
                                       uint32_t max_intermediate, uint32_t vocab_size) {
    (void)vocab_size;  // Not used in new implementation
    // Redirect to the comprehensive GPU forward init
    // Use 0 for n_layer, n_head, n_head_kv, n_vocab since they're not needed for basic buffers
    return tensor_metal_init_gpu_forward(0, max_seq, d_model, max_intermediate,
                                          0, 0, 0);
}

// Free forward pass buffers
void tensor_metal_free_forward_buffers(void) {
    tensor_metal_free_gpu_forward();
}

// Begin batch mode - start accumulating GPU operations
int tensor_metal_begin_batch(void) {
    if (!g_metal_ctx) return -1;
    if (g_metal_ctx.batchModeActive) return -1;  // Already in batch mode

    @autoreleasepool {
        g_metal_ctx.batchCommandBuffer = [g_metal_ctx.commandQueue commandBuffer];
        if (!g_metal_ctx.batchCommandBuffer) {
            NSLog(@"TensorMetal: Failed to create batch command buffer");
            return -1;
        }
        g_metal_ctx.batchModeActive = YES;
        g_metal_ctx.batchOperationCount = 0;

        // Initialize deferred result arrays
        g_metal_ctx.deferredOutputBuffers = [NSMutableArray array];
        g_metal_ctx.deferredOutputDestinations = [NSMutableArray array];
        g_metal_ctx.deferredOutputSizes = [NSMutableArray array];

        return 0;
    }
}

// End batch mode - commit all accumulated operations and wait
int tensor_metal_end_batch(void) {
    if (!g_metal_ctx) return -1;
    if (!g_metal_ctx.batchModeActive) return -1;  // Not in batch mode

    @autoreleasepool {
        [g_metal_ctx.batchCommandBuffer commit];
        [g_metal_ctx.batchCommandBuffer waitUntilCompleted];

        // Check for errors
        NSError *error = g_metal_ctx.batchCommandBuffer.error;
        if (error) {
            NSLog(@"TensorMetal: Batch execution failed: %@", error);
            g_metal_ctx.batchCommandBuffer = nil;
            g_metal_ctx.batchModeActive = NO;
            g_metal_ctx.deferredOutputBuffers = nil;
            g_metal_ctx.deferredOutputDestinations = nil;
            g_metal_ctx.deferredOutputSizes = nil;
            return -1;
        }

        // Copy all deferred results back to CPU
        NSUInteger count = g_metal_ctx.deferredOutputBuffers.count;
        for (NSUInteger i = 0; i < count; i++) {
            id<MTLBuffer> buf = g_metal_ctx.deferredOutputBuffers[i];
            float* dst = (float*)[g_metal_ctx.deferredOutputDestinations[i] pointerValue];
            size_t size = [g_metal_ctx.deferredOutputSizes[i] unsignedLongValue];
            memcpy(dst, buf.contents, size);
        }

        int ops = g_metal_ctx.batchOperationCount;
        g_metal_ctx.batchCommandBuffer = nil;
        g_metal_ctx.batchModeActive = NO;
        g_metal_ctx.batchOperationCount = 0;
        g_metal_ctx.deferredOutputBuffers = nil;
        g_metal_ctx.deferredOutputDestinations = nil;
        g_metal_ctx.deferredOutputSizes = nil;

        return ops;  // Return number of operations executed
    }
}

// Check if batch mode is active
int tensor_metal_batch_active(void) {
    return g_metal_ctx && g_metal_ctx.batchModeActive ? 1 : 0;
}

// Get command buffer for batch mode (internal use)
static id<MTLCommandBuffer> get_command_buffer(void) {
    if (g_metal_ctx.batchModeActive && g_metal_ctx.batchCommandBuffer) {
        return g_metal_ctx.batchCommandBuffer;
    }
    return [g_metal_ctx.commandQueue commandBuffer];
}

// Commit command buffer if not in batch mode
static void commit_if_not_batch(id<MTLCommandBuffer> cmdBuffer) {
    if (!g_metal_ctx.batchModeActive) {
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
    } else {
        g_metal_ctx.batchOperationCount++;
    }
}

// Register a deferred output to be copied at end_batch
static void register_deferred_output(id<MTLBuffer> buffer, float* dst, size_t size) {
    if (g_metal_ctx.batchModeActive && g_metal_ctx.deferredOutputBuffers) {
        [g_metal_ctx.deferredOutputBuffers addObject:buffer];
        [g_metal_ctx.deferredOutputDestinations addObject:[NSValue valueWithPointer:dst]];
        [g_metal_ctx.deferredOutputSizes addObject:@(size)];
    }
}

// =============================================================================
// GPU BUFFER MATMUL - Operations on persistent GPU buffers (Phase 26.18)
// =============================================================================

// Matmul with all data on GPU: C_buf = A_buf @ B_buf^T
// All buffers are persistent GPU buffers. No CPU-GPU copies.
int tensor_metal_matmul_gpu_buffers(void *A_buf, void *B_buf, void *C_buf,
                                     uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !g_metal_ctx.matmulTransposedBPipeline) return -1;
    if (!A_buf || !B_buf || !C_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)A_buf;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)B_buf;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)C_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&N length:sizeof(N) atIndex:4];
        [encoder setBytes:&K length:sizeof(K) atIndex:5];

        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// RMSNorm on GPU buffers
int tensor_metal_rmsnorm_gpu_buffers(void *x_buf, void *weight_buf, void *y_buf,
                                      uint32_t rows, uint32_t dim, float eps) {
    if (!g_metal_ctx || !g_metal_ctx.rmsnormPipeline) return -1;
    if (!x_buf || !weight_buf || !y_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x_buf;
        id<MTLBuffer> bufW = (__bridge id<MTLBuffer>)weight_buf;
        id<MTLBuffer> bufY = (__bridge id<MTLBuffer>)y_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.rmsnormPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufW offset:0 atIndex:1];
        [encoder setBuffer:bufY offset:0 atIndex:2];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:4];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:5];

        MTLSize gridSize = MTLSizeMake(dim, rows, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(dim, 256), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// SiLU multiply on GPU buffers: out = SiLU(gate) * up
int tensor_metal_silu_mul_gpu_buffers(void *gate_buf, void *up_buf, void *out_buf, uint32_t n) {
    if (!g_metal_ctx || !g_metal_ctx.siluMulPipeline) return -1;
    if (!gate_buf || !up_buf || !out_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufGate = (__bridge id<MTLBuffer>)gate_buf;
        id<MTLBuffer> bufUp = (__bridge id<MTLBuffer>)up_buf;
        id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.siluMulPipeline];
        [encoder setBuffer:bufGate offset:0 atIndex:0];
        [encoder setBuffer:bufUp offset:0 atIndex:1];
        [encoder setBuffer:bufOut offset:0 atIndex:2];
        [encoder setBytes:&n length:sizeof(n) atIndex:3];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(n, 256), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// Add residual on GPU buffers: out = a + b
int tensor_metal_add_gpu_buffers(void *a_buf, void *b_buf, void *out_buf, uint32_t n) {
    if (!g_metal_ctx || !g_metal_ctx.addPipeline) return -1;
    if (!a_buf || !b_buf || !out_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a_buf;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b_buf;
        id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.addPipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufOut offset:0 atIndex:2];
        [encoder setBytes:&n length:sizeof(n) atIndex:3];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(n, 256), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// Copy CPU data to GPU buffer
int tensor_metal_upload_to_buffer(void *gpu_buf, const float *cpu_data, size_t size_bytes) {
    if (!g_metal_ctx || !gpu_buf || !cpu_data) return -1;

    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)gpu_buf;
    if (size_bytes > buf.length) return -1;

    memcpy(buf.contents, cpu_data, size_bytes);
    return 0;
}

// Copy GPU buffer to CPU
int tensor_metal_download_from_buffer(void *gpu_buf, float *cpu_data, size_t size_bytes) {
    if (!g_metal_ctx || !gpu_buf || !cpu_data) return -1;

    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)gpu_buf;
    if (size_bytes > buf.length) return -1;

    memcpy(cpu_data, buf.contents, size_bytes);
    return 0;
}

// Get forward pass GPU buffers (for use in forward pass)
// Note: Uses the unified persistent buffer structure
void* tensor_metal_get_hidden_buffer(int index) {
    if (!g_metal_ctx) return NULL;
    // index 0 = main hidden, index 1 = residual (for ping-pong)
    return (__bridge void*)(index == 0 ? g_metal_ctx.gpuHidden : g_metal_ctx.gpuResidual);
}

void* tensor_metal_get_intermediate_buffer(int index) {
    if (!g_metal_ctx) return NULL;
    // index 0 = FFN gate, index 1 = FFN up (for intermediate projections)
    return (__bridge void*)(index == 0 ? g_metal_ctx.gpuFFNGate : g_metal_ctx.gpuFFNUp);
}

void* tensor_metal_get_logits_buffer(void) {
    // Logits are computed on-demand and downloaded, no dedicated buffer
    // Use gpuHidden as scratch space for output projection
    if (!g_metal_ctx) return NULL;
    return (__bridge void*)g_metal_ctx.gpuHidden;
}

// Matrix multiplication on GPU
int tensor_metal_matmul(const float *A, const float *B, float *C,
                        uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !g_metal_ctx.matmulPipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        // Create buffers
        id<MTLBuffer> bufA = [device newBufferWithBytes:A
                                                 length:M * K * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:B
                                                 length:K * N * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:M * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        // Create command buffer
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.matmulPipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&N length:sizeof(N) atIndex:4];
        [encoder setBytes:&K length:sizeof(K) atIndex:5];

        // Dispatch
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Copy result
        memcpy(C, bufC.contents, M * N * sizeof(float));

        return 0;
    }
}

// Matrix multiplication with transposed B on GPU: C = A @ B^T
// A: [M, K], B: [N, K] (stored row-major), C: [M, N]
int tensor_metal_matmul_transposed_b(const float *A, const float *B, float *C,
                                      uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !g_metal_ctx.matmulTransposedBPipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        // Create buffers
        // A is [M, K], B is [N, K] (treated as B^T[K, N])
        id<MTLBuffer> bufA = [device newBufferWithBytes:A
                                                 length:M * K * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:B
                                                 length:N * K * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:M * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        // Create command buffer
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&N length:sizeof(N) atIndex:4];
        [encoder setBytes:&K length:sizeof(K) atIndex:5];

        // Dispatch
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Copy result
        memcpy(C, bufC.contents, M * N * sizeof(float));

        return 0;
    }
}

// SiLU activation on GPU
int tensor_metal_silu(const float *x, float *y, uint32_t n) {
    if (!g_metal_ctx || !g_metal_ctx.siluPipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        id<MTLBuffer> bufX = [device newBufferWithBytes:x
                                                 length:n * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [device newBufferWithLength:n * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.siluPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufY offset:0 atIndex:1];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(y, bufY.contents, n * sizeof(float));
        return 0;
    }
}

// Fused SiLU-multiply on GPU: out = SiLU(gate) * up
int tensor_metal_silu_mul(const float *gate, const float *up, float *out, uint32_t n) {
    if (!g_metal_ctx || !g_metal_ctx.siluMulPipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        id<MTLBuffer> bufGate = [device newBufferWithBytes:gate
                                                    length:n * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUp = [device newBufferWithBytes:up
                                                  length:n * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [device newBufferWithLength:n * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.siluMulPipeline];
        [encoder setBuffer:bufGate offset:0 atIndex:0];
        [encoder setBuffer:bufUp offset:0 atIndex:1];
        [encoder setBuffer:bufOut offset:0 atIndex:2];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(out, bufOut.contents, n * sizeof(float));
        return 0;
    }
}

// RMSNorm on GPU
int tensor_metal_rmsnorm(const float *x, const float *weight, float *y,
                          uint32_t rows, uint32_t dim, float eps) {
    if (!g_metal_ctx || !g_metal_ctx.rmsnormPipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        id<MTLBuffer> bufX = [device newBufferWithBytes:x
                                                 length:rows * dim * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufW = [device newBufferWithBytes:weight
                                                 length:dim * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [device newBufferWithLength:rows * dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.rmsnormPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufW offset:0 atIndex:1];
        [encoder setBuffer:bufY offset:0 atIndex:2];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:3];
        [encoder setBytes:&eps length:sizeof(eps) atIndex:4];

        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(64, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(y, bufY.contents, rows * dim * sizeof(float));
        return 0;
    }
}

// Softmax on GPU
int tensor_metal_softmax(const float *x, float *y, uint32_t rows, uint32_t dim, float temp) {
    if (!g_metal_ctx || !g_metal_ctx.softmaxPipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        id<MTLBuffer> bufX = [device newBufferWithBytes:x
                                                 length:rows * dim * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [device newBufferWithLength:rows * dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.softmaxPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufY offset:0 atIndex:1];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:2];
        [encoder setBytes:&temp length:sizeof(temp) atIndex:3];

        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(64, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(y, bufY.contents, rows * dim * sizeof(float));
        return 0;
    }
}

// Attention scores on GPU: scores = Q @ K^T / sqrt(d_k)
int tensor_metal_attention_scores(const float *Q, const float *K, float *scores,
                                   uint32_t seq_q, uint32_t seq_k, uint32_t d_k) {
    if (!g_metal_ctx || !g_metal_ctx.attentionScoresPipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        id<MTLBuffer> bufQ = [device newBufferWithBytes:Q
                                                 length:seq_q * d_k * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [device newBufferWithBytes:K
                                                 length:seq_k * d_k * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufS = [device newBufferWithLength:seq_q * seq_k * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        uint32_t batch = 1;
        [encoder setComputePipelineState:g_metal_ctx.attentionScoresPipeline];
        [encoder setBuffer:bufQ offset:0 atIndex:0];
        [encoder setBuffer:bufK offset:0 atIndex:1];
        [encoder setBuffer:bufS offset:0 atIndex:2];
        [encoder setBytes:&batch length:sizeof(batch) atIndex:3];
        [encoder setBytes:&seq_q length:sizeof(seq_q) atIndex:4];
        [encoder setBytes:&seq_k length:sizeof(seq_k) atIndex:5];
        [encoder setBytes:&d_k length:sizeof(d_k) atIndex:6];

        MTLSize gridSize = MTLSizeMake(batch, seq_q, seq_k);
        MTLSize threadGroupSize = MTLSizeMake(1, 8, 8);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(scores, bufS.contents, seq_q * seq_k * sizeof(float));
        return 0;
    }
}

// Apply causal mask on GPU
int tensor_metal_causal_mask(float *scores, uint32_t seq_q, uint32_t seq_k) {
    if (!g_metal_ctx || !g_metal_ctx.causalMaskPipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        id<MTLBuffer> bufS = [device newBufferWithBytes:scores
                                                 length:seq_q * seq_k * sizeof(float)
                                                options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.causalMaskPipeline];
        [encoder setBuffer:bufS offset:0 atIndex:0];
        [encoder setBytes:&seq_q length:sizeof(seq_q) atIndex:1];
        [encoder setBytes:&seq_k length:sizeof(seq_k) atIndex:2];

        MTLSize gridSize = MTLSizeMake(seq_q, seq_k, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(scores, bufS.contents, seq_q * seq_k * sizeof(float));
        return 0;
    }
}

// RoPE on GPU (split-half pairing)
int tensor_metal_rope(float *x, uint32_t seq, uint32_t heads,
                       uint32_t head_dim, uint32_t pos_offset, float theta) {
    if (!g_metal_ctx || !g_metal_ctx.ropePipeline) {
        return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        size_t size = seq * heads * head_dim * sizeof(float);
        id<MTLBuffer> bufX = [device newBufferWithBytes:x
                                                 length:size
                                                options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        uint32_t batch = 1;
        [encoder setComputePipelineState:g_metal_ctx.ropePipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBytes:&batch length:sizeof(batch) atIndex:1];
        [encoder setBytes:&seq length:sizeof(seq) atIndex:2];
        [encoder setBytes:&heads length:sizeof(heads) atIndex:3];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:4];
        [encoder setBytes:&pos_offset length:sizeof(pos_offset) atIndex:5];
        [encoder setBytes:&theta length:sizeof(theta) atIndex:6];

        MTLSize gridSize = MTLSizeMake(batch, seq, heads);
        MTLSize threadGroupSize = MTLSizeMake(1, 8, 8);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(x, bufX.contents, size);
        return 0;
    }
}

// Q8_0 dequantization on GPU
// GGUF Q8_0 format: 2-byte F16 scale + 32 int8 values = 34 bytes per block
int tensor_metal_dequantize_q8_0(const uint8_t *src, float *dst, uint32_t num_blocks) {
    if (!g_metal_ctx) {
        return -1;
    }

    // Create pipeline for dequantization if not exists
    static id<MTLComputePipelineState> dequantPipeline = nil;
    if (!dequantPipeline) {
        id<MTLFunction> func = [g_metal_ctx.library newFunctionWithName:@"dequantize_q8_0"];
        if (!func) return -1;
        NSError *error = nil;
        dequantPipeline = [g_metal_ctx.device newComputePipelineStateWithFunction:func error:&error];
        if (!dequantPipeline) return -1;
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        // GGUF Q8_0: 34 bytes per block (2-byte F16 scale + 32 int8 values)
        size_t src_size = num_blocks * 34;
        size_t dst_size = num_blocks * 32 * sizeof(float);

        id<MTLBuffer> bufSrc = [device newBufferWithBytes:src
                                                   length:src_size
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst = [device newBufferWithLength:dst_size
                                                   options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:dequantPipeline];
        [encoder setBuffer:bufSrc offset:0 atIndex:0];
        [encoder setBuffer:bufDst offset:0 atIndex:1];
        [encoder setBytes:&num_blocks length:sizeof(num_blocks) atIndex:2];

        MTLSize gridSize = MTLSizeMake(num_blocks, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(dst, bufDst.contents, dst_size);
        return 0;
    }
}

// Q8_0 matmul with transposed B on GPU: C = A @ B_q8^T
// A: [M, K] float32
// B_q8: [N, K] Q8_0 quantized
// C: [M, N] float32
int tensor_metal_matmul_q8_0_transposed(const float *A, const uint8_t *B_q8, float *C,
                                         uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx) {
        return -1;
    }

    // K must be divisible by 32 for Q8_0
    if (K % 32 != 0) {
        return -1;
    }

    // Create pipeline if not exists
    static id<MTLComputePipelineState> matmulQ8Pipeline = nil;
    if (!matmulQ8Pipeline) {
        id<MTLFunction> func = [g_metal_ctx.library newFunctionWithName:@"matmul_q8_0_transposed_f32"];
        if (!func) {
            NSLog(@"TensorMetal: matmul_q8_0_transposed_f32 kernel not found");
            return -1;
        }
        NSError *error = nil;
        matmulQ8Pipeline = [g_metal_ctx.device newComputePipelineStateWithFunction:func error:&error];
        if (!matmulQ8Pipeline) {
            NSLog(@"TensorMetal: Failed to create Q8_0 matmul pipeline: %@", error);
            return -1;
        }
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        // Calculate B_q8 size: N rows, each row has K/32 blocks of 34 bytes
        uint32_t blocks_per_row = K / 32;
        size_t b_size = N * blocks_per_row * 34;

        id<MTLBuffer> bufA = [device newBufferWithBytes:A
                                                 length:M * K * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:B_q8
                                                 length:b_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:M * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:matmulQ8Pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&N length:sizeof(N) atIndex:4];
        [encoder setBytes:&K length:sizeof(K) atIndex:5];

        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(C, bufC.contents, M * N * sizeof(float));
        return 0;
    }
}

// =============================================================================
// PERSISTENT GPU BUFFERS (Phase 26.15)
// =============================================================================

// Create a persistent GPU buffer from CPU data
void* tensor_metal_buffer_create(const void *data, size_t size) {
    if (!g_metal_ctx || !data || size == 0) {
        return NULL;
    }

    @autoreleasepool {
        id<MTLBuffer> buffer = [g_metal_ctx.device newBufferWithBytes:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared];
        if (!buffer) {
            NSLog(@"TensorMetal: Failed to create buffer of size %zu", size);
            return NULL;
        }

        // Return retained buffer (caller must free with tensor_metal_buffer_free)
        return (__bridge_retained void*)buffer;
    }
}

// Create a persistent GPU buffer without initialization
void* tensor_metal_buffer_create_empty(size_t size) {
    if (!g_metal_ctx || size == 0) {
        return NULL;
    }

    @autoreleasepool {
        id<MTLBuffer> buffer = [g_metal_ctx.device newBufferWithLength:size
                                                               options:MTLResourceStorageModeShared];
        if (!buffer) {
            NSLog(@"TensorMetal: Failed to create empty buffer of size %zu", size);
            return NULL;
        }

        return (__bridge_retained void*)buffer;
    }
}

// Free a persistent GPU buffer
void tensor_metal_buffer_free(void *buffer) {
    if (buffer) {
        @autoreleasepool {
            // Release the retained reference
            id<MTLBuffer> mtlBuffer = (__bridge_transfer id<MTLBuffer>)buffer;
            mtlBuffer = nil;  // Force release
        }
    }
}

// Copy data from persistent GPU buffer back to CPU
int tensor_metal_buffer_read(void *buffer, void *dst, size_t size) {
    if (!buffer || !dst || size == 0) {
        return -1;
    }

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (size > mtlBuffer.length) {
            NSLog(@"TensorMetal: Buffer read size %zu exceeds buffer length %lu",
                  size, (unsigned long)mtlBuffer.length);
            return -1;
        }
        memcpy(dst, mtlBuffer.contents, size);
        return 0;
    }
}

// Update data in persistent GPU buffer from CPU
int tensor_metal_buffer_write(void *buffer, const void *src, size_t size) {
    if (!buffer || !src || size == 0) {
        return -1;
    }

    @autoreleasepool {
        id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
        if (size > mtlBuffer.length) {
            NSLog(@"TensorMetal: Buffer write size %zu exceeds buffer length %lu",
                  size, (unsigned long)mtlBuffer.length);
            return -1;
        }
        memcpy(mtlBuffer.contents, src, size);
        return 0;
    }
}

// Matrix multiply using persistent GPU buffers: C = A @ B^T
int tensor_metal_matmul_persistent(void *A_buf, void *B_buf, void *C_buf,
                                    uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !g_metal_ctx.matmulTransposedBPipeline) {
        return -1;
    }
    if (!A_buf || !B_buf || !C_buf) {
        return -1;
    }

    @autoreleasepool {
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)A_buf;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)B_buf;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)C_buf;

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&N length:sizeof(N) atIndex:4];
        [encoder setBytes:&K length:sizeof(K) atIndex:5];

        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        return 0;
    }
}

// Q8_0 matmul using persistent GPU buffer for weights
int tensor_metal_matmul_q8_0_persistent(const float *A, void *B_buf, float *C,
                                         uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !A || !B_buf || !C) {
        return -1;
    }

    // K must be divisible by 32 for Q8_0
    if (K % 32 != 0) {
        return -1;
    }

    // Get or create Q8_0 matmul pipeline
    static id<MTLComputePipelineState> matmulQ8Pipeline = nil;
    if (!matmulQ8Pipeline) {
        id<MTLFunction> func = [g_metal_ctx.library newFunctionWithName:@"matmul_q8_0_transposed_f32"];
        if (!func) {
            NSLog(@"TensorMetal: matmul_q8_0_transposed_f32 kernel not found");
            return -1;
        }
        NSError *error = nil;
        matmulQ8Pipeline = [g_metal_ctx.device newComputePipelineStateWithFunction:func error:&error];
        if (!matmulQ8Pipeline) {
            NSLog(@"TensorMetal: Failed to create Q8_0 persistent matmul pipeline: %@", error);
            return -1;
        }
    }

    size_t input_size = M * K * sizeof(float);
    size_t output_size = M * N * sizeof(float);
    BOOL in_batch_mode = g_metal_ctx.batchModeActive;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)B_buf;
        id<MTLBuffer> bufA = nil;
        id<MTLBuffer> bufC = nil;

        if (in_batch_mode) {
            // In batch mode: allocate fresh buffers for each operation
            bufA = [device newBufferWithBytes:A
                                       length:input_size
                                      options:MTLResourceStorageModeShared];
            bufC = [device newBufferWithLength:output_size
                                       options:MTLResourceStorageModeShared];
        } else {
            // Not in batch mode: try to use scratch buffers
            if (g_metal_ctx.scratchInput && g_metal_ctx.scratchOutput &&
                input_size <= g_metal_ctx.scratchInputSize &&
                output_size <= g_metal_ctx.scratchOutputSize) {
                memcpy(g_metal_ctx.scratchInput.contents, A, input_size);
                bufA = g_metal_ctx.scratchInput;
                bufC = g_metal_ctx.scratchOutput;
            } else {
                bufA = [device newBufferWithBytes:A
                                           length:input_size
                                          options:MTLResourceStorageModeShared];
                bufC = [device newBufferWithLength:output_size
                                           options:MTLResourceStorageModeShared];
            }
        }

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:matmulQ8Pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];  // Persistent weight buffer
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&N length:sizeof(N) atIndex:4];
        [encoder setBytes:&K length:sizeof(K) atIndex:5];

        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        if (in_batch_mode) {
            // Register for deferred copy at end_batch
            register_deferred_output(bufC, C, output_size);
            g_metal_ctx.batchOperationCount++;
        } else {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(C, bufC.contents, output_size);
        }

        return 0;
    }
}

// F32 matmul using persistent GPU buffer for weights
int tensor_metal_matmul_f32_persistent(const float *A, void *B_buf, float *C,
                                        uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !g_metal_ctx.matmulTransposedBPipeline) {
        return -1;
    }
    if (!A || !B_buf || !C) {
        return -1;
    }

    size_t input_size = M * K * sizeof(float);
    size_t output_size = M * N * sizeof(float);
    BOOL in_batch_mode = g_metal_ctx.batchModeActive;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)B_buf;
        id<MTLBuffer> bufA = nil;
        id<MTLBuffer> bufC = nil;

        if (in_batch_mode) {
            // In batch mode: allocate fresh buffers for each operation
            // (can't reuse scratch - would get overwritten by subsequent ops)
            bufA = [device newBufferWithBytes:A
                                       length:input_size
                                      options:MTLResourceStorageModeShared];
            bufC = [device newBufferWithLength:output_size
                                       options:MTLResourceStorageModeShared];
        } else {
            // Not in batch mode: try to use scratch buffers for efficiency
            if (g_metal_ctx.scratchInput && g_metal_ctx.scratchOutput &&
                input_size <= g_metal_ctx.scratchInputSize &&
                output_size <= g_metal_ctx.scratchOutputSize) {
                memcpy(g_metal_ctx.scratchInput.contents, A, input_size);
                bufA = g_metal_ctx.scratchInput;
                bufC = g_metal_ctx.scratchOutput;
            } else {
                bufA = [device newBufferWithBytes:A
                                           length:input_size
                                          options:MTLResourceStorageModeShared];
                bufC = [device newBufferWithLength:output_size
                                           options:MTLResourceStorageModeShared];
            }
        }

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];  // Persistent weight buffer
        [encoder setBuffer:bufC offset:0 atIndex:2];
        [encoder setBytes:&M length:sizeof(M) atIndex:3];
        [encoder setBytes:&N length:sizeof(N) atIndex:4];
        [encoder setBytes:&K length:sizeof(K) atIndex:5];

        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        if (in_batch_mode) {
            // Register for deferred copy at end_batch
            register_deferred_output(bufC, C, output_size);
            g_metal_ctx.batchOperationCount++;
        } else {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(C, bufC.contents, output_size);
        }

        return 0;
    }
}

// =============================================================================
// FUSED KERNELS (Phase 26.18)
// =============================================================================

// Fused gate + up projection for SwiGLU FFN
// Computes both projections in single kernel: gate = x @ W_gate^T, up = x @ W_up^T
int tensor_metal_fused_gate_up_proj(const float *x, void *W_gate_buf, void *W_up_buf,
                                     float *gate, float *up,
                                     uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !g_metal_ctx.fusedGateUpPipeline) return -1;
    if (!x || !W_gate_buf || !W_up_buf || !gate || !up) return -1;

    size_t input_size = M * K * sizeof(float);
    size_t output_size = M * N * sizeof(float);
    BOOL in_batch_mode = g_metal_ctx.batchModeActive;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        id<MTLBuffer> bufWGate = (__bridge id<MTLBuffer>)W_gate_buf;
        id<MTLBuffer> bufWUp = (__bridge id<MTLBuffer>)W_up_buf;

        // Create input/output buffers
        id<MTLBuffer> bufX = [device newBufferWithBytes:x
                                                 length:input_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufGate = [device newBufferWithLength:output_size
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUp = [device newBufferWithLength:output_size
                                                  options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.fusedGateUpPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufWGate offset:0 atIndex:1];
        [encoder setBuffer:bufWUp offset:0 atIndex:2];
        [encoder setBuffer:bufGate offset:0 atIndex:3];
        [encoder setBuffer:bufUp offset:0 atIndex:4];
        [encoder setBytes:&M length:sizeof(M) atIndex:5];
        [encoder setBytes:&N length:sizeof(N) atIndex:6];
        [encoder setBytes:&K length:sizeof(K) atIndex:7];

        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        if (in_batch_mode) {
            register_deferred_output(bufGate, gate, output_size);
            register_deferred_output(bufUp, up, output_size);
            g_metal_ctx.batchOperationCount++;
        } else {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(gate, bufGate.contents, output_size);
            memcpy(up, bufUp.contents, output_size);
        }

        return 0;
    }
}

// Fused SwiGLU + down projection
// Computes: out = (SiLU(gate) * up) @ W_down^T
int tensor_metal_fused_swiglu_down(const float *gate, const float *up, void *W_down_buf,
                                    float *out,
                                    uint32_t M, uint32_t N_ff, uint32_t N_out) {
    if (!g_metal_ctx || !g_metal_ctx.fusedSwiGLUDownPipeline) return -1;
    if (!gate || !up || !W_down_buf || !out) return -1;

    size_t gate_size = M * N_ff * sizeof(float);
    size_t output_size = M * N_out * sizeof(float);
    BOOL in_batch_mode = g_metal_ctx.batchModeActive;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        id<MTLBuffer> bufWDown = (__bridge id<MTLBuffer>)W_down_buf;

        id<MTLBuffer> bufGate = [device newBufferWithBytes:gate
                                                    length:gate_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUp = [device newBufferWithBytes:up
                                                  length:gate_size
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [device newBufferWithLength:output_size
                                                   options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.fusedSwiGLUDownPipeline];
        [encoder setBuffer:bufGate offset:0 atIndex:0];
        [encoder setBuffer:bufUp offset:0 atIndex:1];
        [encoder setBuffer:bufWDown offset:0 atIndex:2];
        [encoder setBuffer:bufOut offset:0 atIndex:3];
        [encoder setBytes:&M length:sizeof(M) atIndex:4];
        [encoder setBytes:&N_ff length:sizeof(N_ff) atIndex:5];
        [encoder setBytes:&N_out length:sizeof(N_out) atIndex:6];

        MTLSize gridSize = MTLSizeMake(N_out, M, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        if (in_batch_mode) {
            register_deferred_output(bufOut, out, output_size);
            g_metal_ctx.batchOperationCount++;
        } else {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(out, bufOut.contents, output_size);
        }

        return 0;
    }
}

// Optimized matvec for single-token inference: y = x @ W^T
// When M=1, uses specialized kernel with one thread per output
int tensor_metal_matvec_f32(const float *x, void *W_buf, float *y,
                             uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !g_metal_ctx.matvecF32Pipeline) return -1;
    if (!x || !W_buf || !y) return -1;

    size_t input_size = K * sizeof(float);
    size_t output_size = N * sizeof(float);
    BOOL in_batch_mode = g_metal_ctx.batchModeActive;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        id<MTLBuffer> bufW = (__bridge id<MTLBuffer>)W_buf;

        id<MTLBuffer> bufX = [device newBufferWithBytes:x
                                                 length:input_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [device newBufferWithLength:output_size
                                                 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.matvecF32Pipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufW offset:0 atIndex:1];
        [encoder setBuffer:bufY offset:0 atIndex:2];
        [encoder setBytes:&N length:sizeof(N) atIndex:3];
        [encoder setBytes:&K length:sizeof(K) atIndex:4];

        MTLSize gridSize = MTLSizeMake(N, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(N, 256), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        if (in_batch_mode) {
            register_deferred_output(bufY, y, output_size);
            g_metal_ctx.batchOperationCount++;
        } else {
            [cmdBuffer commit];
            [cmdBuffer waitUntilCompleted];
            memcpy(y, bufY.contents, output_size);
        }

        return 0;
    }
}

// =============================================================================
// COMPLETE GPU-NATIVE FFN (Phase 26.18 - Fixed)
// =============================================================================

// Complete SwiGLU FFN on GPU - no intermediate CPU copies
// Computes: hidden_out = SiLU(hidden_in @ W1^T) * (hidden_in @ W3^T) @ W2^T
// Note: Does NOT add residual - caller must handle residual connection
// All operations in single command buffer with data staying on GPU
int tensor_metal_ffn_swiglu_complete(const float *hidden_in, float *hidden_out,
                                      void *W1_buf, void *W3_buf, void *W2_buf,
                                      uint32_t M, uint32_t n_ff, uint32_t n_embd) {
    if (!g_metal_ctx) return -1;
    if (!hidden_in || !hidden_out || !W1_buf || !W3_buf || !W2_buf) return -1;

    size_t hidden_size = M * n_embd * sizeof(float);
    size_t ff_size = M * n_ff * sizeof(float);

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        id<MTLBuffer> bufW1 = (__bridge id<MTLBuffer>)W1_buf;
        id<MTLBuffer> bufW3 = (__bridge id<MTLBuffer>)W3_buf;
        id<MTLBuffer> bufW2 = (__bridge id<MTLBuffer>)W2_buf;

        // Allocate all buffers
        id<MTLBuffer> bufHiddenIn = [device newBufferWithBytes:hidden_in
                                                        length:hidden_size
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufGate = [device newBufferWithLength:ff_size
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufUp = [device newBufferWithLength:ff_size
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSwiglu = [device newBufferWithLength:ff_size
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [device newBufferWithLength:hidden_size
                                                   options:MTLResourceStorageModeShared];

        // Single command buffer for all operations
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];

        // Op 1: gate = hidden @ W1^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:bufHiddenIn offset:0 atIndex:0];
            [enc setBuffer:bufW1 offset:0 atIndex:1];
            [enc setBuffer:bufGate offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_ff, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 2: up = hidden @ W3^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:bufHiddenIn offset:0 atIndex:0];
            [enc setBuffer:bufW3 offset:0 atIndex:1];
            [enc setBuffer:bufUp offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_ff, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 3: swiglu = SiLU(gate) * up
        {
            uint32_t n = M * n_ff;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.siluMulPipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBuffer:bufUp offset:0 atIndex:1];
            [enc setBuffer:bufSwiglu offset:0 atIndex:2];
            [enc setBytes:&n length:sizeof(n) atIndex:3];
            MTLSize grid = MTLSizeMake(n, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(n, 256), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 4: out = swiglu @ W2^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:bufSwiglu offset:0 atIndex:0];
            [enc setBuffer:bufW2 offset:0 atIndex:1];
            [enc setBuffer:bufOut offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:4];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:5];
            MTLSize grid = MTLSizeMake(n_embd, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Execute all operations
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Check for errors
        if (cmdBuffer.error) {
            NSLog(@"TensorMetal: FFN failed: %@", cmdBuffer.error);
            return -1;
        }

        // Copy result to hidden_out (no residual add - caller handles that)
        memcpy(hidden_out, bufOut.contents, hidden_size);

        return 0;
    }
}

// =============================================================================
// SSM (MAMBA) METAL OPERATIONS
// =============================================================================

// SSM selective scan step on GPU
// Computes one timestep of SSM recurrence in parallel over d_inner × d_state
int tensor_metal_ssm_scan_step(const float *x, const float *delta,
                                const float *A, const float *B,
                                const float *C, const float *D,
                                float *h, float *y,
                                uint32_t d_inner, uint32_t d_state) {
    if (!g_metal_ctx) {
        return -1;
    }

    // Create pipeline if not exists
    static id<MTLComputePipelineState> ssmScanPipeline = nil;
    static id<MTLComputePipelineState> ssmSkipPipeline = nil;
    if (!ssmScanPipeline) {
        id<MTLFunction> func = [g_metal_ctx.library newFunctionWithName:@"ssm_scan_step_f32"];
        if (!func) {
            NSLog(@"TensorMetal: ssm_scan_step_f32 kernel not found");
            return -1;
        }
        NSError *error = nil;
        ssmScanPipeline = [g_metal_ctx.device newComputePipelineStateWithFunction:func error:&error];
        if (!ssmScanPipeline) {
            NSLog(@"TensorMetal: Failed to create SSM scan pipeline: %@", error);
            return -1;
        }

        func = [g_metal_ctx.library newFunctionWithName:@"ssm_skip_add_f32"];
        if (func) {
            ssmSkipPipeline = [g_metal_ctx.device newComputePipelineStateWithFunction:func error:&error];
        }
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        // Create buffers
        id<MTLBuffer> bufX = [device newBufferWithBytes:x
                                                 length:d_inner * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDelta = [device newBufferWithBytes:delta
                                                     length:d_inner * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufA = [device newBufferWithBytes:A
                                                 length:d_inner * d_state * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:B
                                                 length:d_state * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithBytes:C
                                                 length:d_state * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufD = D ? [device newBufferWithBytes:D
                                                     length:d_inner * sizeof(float)
                                                    options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> bufH = [device newBufferWithBytes:h
                                                 length:d_inner * d_state * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        // Initialize y to zero (will accumulate via atomic adds)
        id<MTLBuffer> bufY = [device newBufferWithLength:d_inner * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        memset(bufY.contents, 0, d_inner * sizeof(float));

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        // Run SSM scan kernel
        [encoder setComputePipelineState:ssmScanPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufDelta offset:0 atIndex:1];
        [encoder setBuffer:bufA offset:0 atIndex:2];
        [encoder setBuffer:bufB offset:0 atIndex:3];
        [encoder setBuffer:bufC offset:0 atIndex:4];
        [encoder setBuffer:bufD ? bufD : bufX offset:0 atIndex:5]; // D or dummy
        [encoder setBuffer:bufH offset:0 atIndex:6];
        [encoder setBuffer:bufY offset:0 atIndex:7];
        [encoder setBytes:&d_inner length:sizeof(d_inner) atIndex:8];
        [encoder setBytes:&d_state length:sizeof(d_state) atIndex:9];

        // Dispatch d_inner × d_state threads
        MTLSize gridSize = MTLSizeMake(d_inner, d_state, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

        // Add skip connection: y += D * x
        if (bufD && ssmSkipPipeline) {
            [encoder setComputePipelineState:ssmSkipPipeline];
            [encoder setBuffer:bufY offset:0 atIndex:0];
            [encoder setBuffer:bufX offset:0 atIndex:1];
            [encoder setBuffer:bufD offset:0 atIndex:2];

            MTLSize skipGridSize = MTLSizeMake(d_inner, 1, 1);
            MTLSize skipThreadGroupSize = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreads:skipGridSize threadsPerThreadgroup:skipThreadGroupSize];
        }

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Copy results back
        memcpy(h, bufH.contents, d_inner * d_state * sizeof(float));
        memcpy(y, bufY.contents, d_inner * sizeof(float));

        return 0;
    }
}

// Causal 1D convolution on GPU
int tensor_metal_conv1d_causal(const float *x, const float *w, const float *bias,
                                float *conv_state, float *out,
                                uint32_t seq_len, uint32_t d_inner, uint32_t d_conv) {
    if (!g_metal_ctx) {
        return -1;
    }

    // Create pipeline if not exists
    static id<MTLComputePipelineState> conv1dPipeline = nil;
    static id<MTLComputePipelineState> updateStatePipeline = nil;
    if (!conv1dPipeline) {
        id<MTLFunction> func = [g_metal_ctx.library newFunctionWithName:@"conv1d_causal_f32"];
        if (!func) {
            NSLog(@"TensorMetal: conv1d_causal_f32 kernel not found");
            return -1;
        }
        NSError *error = nil;
        conv1dPipeline = [g_metal_ctx.device newComputePipelineStateWithFunction:func error:&error];
        if (!conv1dPipeline) {
            NSLog(@"TensorMetal: Failed to create conv1d pipeline: %@", error);
            return -1;
        }

        func = [g_metal_ctx.library newFunctionWithName:@"conv1d_update_state_f32"];
        if (func) {
            updateStatePipeline = [g_metal_ctx.device newComputePipelineStateWithFunction:func error:&error];
        }
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        // Create buffers
        id<MTLBuffer> bufX = [device newBufferWithBytes:x
                                                 length:seq_len * d_inner * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufW = [device newBufferWithBytes:w
                                                 length:d_inner * d_conv * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufBias = bias ? [device newBufferWithBytes:bias
                                                           length:d_inner * sizeof(float)
                                                          options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> bufState = conv_state ? [device newBufferWithBytes:conv_state
                                                                  length:d_inner * (d_conv - 1) * sizeof(float)
                                                                 options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> bufOut = [device newBufferWithLength:seq_len * d_inner * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        // Run conv1d kernel
        [encoder setComputePipelineState:conv1dPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBuffer:bufW offset:0 atIndex:1];
        [encoder setBuffer:bufBias ? bufBias : bufX offset:0 atIndex:2]; // bias or dummy
        [encoder setBuffer:bufState ? bufState : bufX offset:0 atIndex:3]; // state or dummy
        [encoder setBuffer:bufOut offset:0 atIndex:4];
        [encoder setBytes:&seq_len length:sizeof(seq_len) atIndex:5];
        [encoder setBytes:&d_inner length:sizeof(d_inner) atIndex:6];
        [encoder setBytes:&d_conv length:sizeof(d_conv) atIndex:7];

        // Dispatch seq_len × d_inner threads
        MTLSize gridSize = MTLSizeMake(seq_len, d_inner, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

        // Update conv state for next call
        if (bufState && updateStatePipeline) {
            [encoder setComputePipelineState:updateStatePipeline];
            [encoder setBuffer:bufX offset:0 atIndex:0];
            [encoder setBuffer:bufState offset:0 atIndex:1];
            [encoder setBytes:&seq_len length:sizeof(seq_len) atIndex:2];
            [encoder setBytes:&d_inner length:sizeof(d_inner) atIndex:3];
            [encoder setBytes:&d_conv length:sizeof(d_conv) atIndex:4];

            MTLSize stateGridSize = MTLSizeMake(d_conv - 1, d_inner, 1);
            MTLSize stateThreadGroupSize = MTLSizeMake(8, 32, 1);
            [encoder dispatchThreads:stateGridSize threadsPerThreadgroup:stateThreadGroupSize];
        }

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Copy results back
        memcpy(out, bufOut.contents, seq_len * d_inner * sizeof(float));
        if (conv_state && bufState) {
            memcpy(conv_state, bufState.contents, d_inner * (d_conv - 1) * sizeof(float));
        }

        return 0;
    }
}

// SSM gated output on GPU: y_out = y_ssm * SiLU(z)
int tensor_metal_ssm_gate(const float *y_ssm, const float *z, float *y_out, uint32_t n) {
    if (!g_metal_ctx) {
        return -1;
    }

    // Create pipeline if not exists
    static id<MTLComputePipelineState> gatePipeline = nil;
    if (!gatePipeline) {
        id<MTLFunction> func = [g_metal_ctx.library newFunctionWithName:@"ssm_gate_output_f32"];
        if (!func) {
            NSLog(@"TensorMetal: ssm_gate_output_f32 kernel not found");
            return -1;
        }
        NSError *error = nil;
        gatePipeline = [g_metal_ctx.device newComputePipelineStateWithFunction:func error:&error];
        if (!gatePipeline) {
            NSLog(@"TensorMetal: Failed to create SSM gate pipeline: %@", error);
            return -1;
        }
    }

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        id<MTLBuffer> bufYSSM = [device newBufferWithBytes:y_ssm
                                                    length:n * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufZ = [device newBufferWithBytes:z
                                                 length:n * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [device newBufferWithLength:n * sizeof(float)
                                                   options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:gatePipeline];
        [encoder setBuffer:bufYSSM offset:0 atIndex:0];
        [encoder setBuffer:bufZ offset:0 atIndex:1];
        [encoder setBuffer:bufOut offset:0 atIndex:2];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

        [encoder endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(y_out, bufOut.contents, n * sizeof(float));
        return 0;
    }
}

// =============================================================================
// GPU-NATIVE ATTENTION OPERATIONS (GPU-Native Forward Pass)
// =============================================================================

// Fused attention scores with causal mask on GPU buffers
int tensor_metal_attention_scores_causal_gpu(
    void* Q_buf, void* K_buf, void* scores_buf,
    uint32_t seq_q, uint32_t seq_k, uint32_t d_k, uint32_t pos_offset) {
    if (!g_metal_ctx || !g_metal_ctx.attentionScoresCausalGpuPipeline) return -1;
    if (!Q_buf || !K_buf || !scores_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufQ = (__bridge id<MTLBuffer>)Q_buf;
        id<MTLBuffer> bufK = (__bridge id<MTLBuffer>)K_buf;
        id<MTLBuffer> bufScores = (__bridge id<MTLBuffer>)scores_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.attentionScoresCausalGpuPipeline];
        [encoder setBuffer:bufQ offset:0 atIndex:0];
        [encoder setBuffer:bufK offset:0 atIndex:1];
        [encoder setBuffer:bufScores offset:0 atIndex:2];
        [encoder setBytes:&seq_q length:sizeof(seq_q) atIndex:3];
        [encoder setBytes:&seq_k length:sizeof(seq_k) atIndex:4];
        [encoder setBytes:&d_k length:sizeof(d_k) atIndex:5];
        [encoder setBytes:&pos_offset length:sizeof(pos_offset) atIndex:6];

        // Grid: (seq_k, seq_q) - each thread computes one score
        MTLSize gridSize = MTLSizeMake(seq_k, seq_q, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// Row-wise softmax in-place on GPU buffer
int tensor_metal_softmax_rows_gpu(void* x_buf, uint32_t rows, uint32_t dim) {
    if (!g_metal_ctx || !g_metal_ctx.softmaxRowsGpuPipeline) return -1;
    if (!x_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.softmaxRowsGpuPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:1];
        [encoder setBytes:&dim length:sizeof(dim) atIndex:2];

        // One thread per row
        MTLSize gridSize = MTLSizeMake(rows, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(rows, 256), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// Attention weighted sum on GPU buffers: out = scores @ V
int tensor_metal_attention_weighted_sum_gpu(
    void* scores_buf, void* V_buf, void* out_buf,
    uint32_t seq_q, uint32_t seq_k, uint32_t d_v) {
    if (!g_metal_ctx || !g_metal_ctx.attentionWeightedSumGpuPipeline) return -1;
    if (!scores_buf || !V_buf || !out_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufScores = (__bridge id<MTLBuffer>)scores_buf;
        id<MTLBuffer> bufV = (__bridge id<MTLBuffer>)V_buf;
        id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.attentionWeightedSumGpuPipeline];
        [encoder setBuffer:bufScores offset:0 atIndex:0];
        [encoder setBuffer:bufV offset:0 atIndex:1];
        [encoder setBuffer:bufOut offset:0 atIndex:2];
        [encoder setBytes:&seq_q length:sizeof(seq_q) atIndex:3];
        [encoder setBytes:&seq_k length:sizeof(seq_k) atIndex:4];
        [encoder setBytes:&d_v length:sizeof(d_v) atIndex:5];

        // Grid: (d_v, seq_q) - each thread computes one output element
        MTLSize gridSize = MTLSizeMake(d_v, seq_q, 1);
        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// RoPE in-place on GPU buffer
int tensor_metal_rope_gpu(void* x_buf, uint32_t seq, uint32_t heads,
                          uint32_t head_dim, uint32_t pos_offset, float theta) {
    if (!g_metal_ctx || !g_metal_ctx.ropeGpuPipeline) return -1;
    if (!x_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.ropeGpuPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBytes:&seq length:sizeof(seq) atIndex:1];
        [encoder setBytes:&heads length:sizeof(heads) atIndex:2];
        [encoder setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
        [encoder setBytes:&pos_offset length:sizeof(pos_offset) atIndex:4];
        [encoder setBytes:&theta length:sizeof(theta) atIndex:5];

        // Grid: (heads, seq) - each thread handles one head at one position
        MTLSize gridSize = MTLSizeMake(heads, seq, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(heads, 16), MIN(seq, 16), 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// Add two GPU buffers in-place: a = a + b
int tensor_metal_add_inplace_gpu(void* a_buf, void* b_buf, uint32_t n) {
    if (!g_metal_ctx || !g_metal_ctx.addInplaceGpuPipeline) return -1;
    if (!a_buf || !b_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a_buf;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.addInplaceGpuPipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBytes:&n length:sizeof(n) atIndex:2];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(n, 256), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// Scale GPU buffer in-place: x = x * scale
int tensor_metal_scale_inplace_gpu(void* x_buf, float scale, uint32_t n) {
    if (!g_metal_ctx || !g_metal_ctx.scaleInplaceGpuPipeline) return -1;
    if (!x_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufX = (__bridge id<MTLBuffer>)x_buf;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];

        [encoder setComputePipelineState:g_metal_ctx.scaleInplaceGpuPipeline];
        [encoder setBuffer:bufX offset:0 atIndex:0];
        [encoder setBytes:&scale length:sizeof(scale) atIndex:1];
        [encoder setBytes:&n length:sizeof(n) atIndex:2];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        MTLSize threadGroupSize = MTLSizeMake(MIN(n, 256), 1, 1);
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        commit_if_not_batch(cmdBuffer);
        return 0;
    }
}

// =============================================================================
// MULTI-HEAD ATTENTION WITH GQA SUPPORT
// =============================================================================

// Multi-head attention on GPU buffers (all data already on GPU)
int tensor_metal_attention_multihead_gpu(
    void* q_buf, void* k_cache_buf, void* v_cache_buf,
    void* out_buf, void* scores_buf,
    uint32_t num_tokens, uint32_t seq_len,
    uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
    uint32_t pos_offset) {

    if (!g_metal_ctx) return -1;
    if (!g_metal_ctx.attentionMultiheadScoresPipeline ||
        !g_metal_ctx.attentionMultiheadSoftmaxPipeline ||
        !g_metal_ctx.attentionMultiheadWeightedSumPipeline) return -1;
    if (!q_buf || !k_cache_buf || !v_cache_buf || !out_buf || !scores_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufQ = (__bridge id<MTLBuffer>)q_buf;
        id<MTLBuffer> bufK = (__bridge id<MTLBuffer>)k_cache_buf;
        id<MTLBuffer> bufV = (__bridge id<MTLBuffer>)v_cache_buf;
        id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out_buf;
        id<MTLBuffer> bufScores = (__bridge id<MTLBuffer>)scores_buf;

        // Single command buffer for all 3 operations
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];

        // Step 1: Compute attention scores with causal mask
        // Grid: (seq_len, n_head, num_tokens)
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.attentionMultiheadScoresPipeline];
            [enc setBuffer:bufQ offset:0 atIndex:0];
            [enc setBuffer:bufK offset:0 atIndex:1];
            [enc setBuffer:bufScores offset:0 atIndex:2];
            [enc setBytes:&num_tokens length:sizeof(num_tokens) atIndex:3];
            [enc setBytes:&n_head length:sizeof(n_head) atIndex:4];
            [enc setBytes:&n_head_kv length:sizeof(n_head_kv) atIndex:5];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
            [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:7];
            [enc setBytes:&pos_offset length:sizeof(pos_offset) atIndex:8];

            MTLSize grid = MTLSizeMake(seq_len, n_head, num_tokens);
            MTLSize tg = MTLSizeMake(MIN(seq_len, 16), MIN(n_head, 8), MIN(num_tokens, 4));
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 2: Softmax over scores
        // Grid: (n_head, num_tokens, 1)
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.attentionMultiheadSoftmaxPipeline];
            [enc setBuffer:bufScores offset:0 atIndex:0];
            [enc setBytes:&num_tokens length:sizeof(num_tokens) atIndex:1];
            [enc setBytes:&n_head length:sizeof(n_head) atIndex:2];
            [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:3];

            MTLSize grid = MTLSizeMake(n_head, num_tokens, 1);
            MTLSize tg = MTLSizeMake(MIN(n_head, 32), MIN(num_tokens, 8), 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 3: Weighted sum
        // Grid: (head_dim, n_head, num_tokens)
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.attentionMultiheadWeightedSumPipeline];
            [enc setBuffer:bufScores offset:0 atIndex:0];
            [enc setBuffer:bufV offset:0 atIndex:1];
            [enc setBuffer:bufOut offset:0 atIndex:2];
            [enc setBytes:&num_tokens length:sizeof(num_tokens) atIndex:3];
            [enc setBytes:&n_head length:sizeof(n_head) atIndex:4];
            [enc setBytes:&n_head_kv length:sizeof(n_head_kv) atIndex:5];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:6];
            [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:7];

            MTLSize grid = MTLSizeMake(head_dim, n_head, num_tokens);
            MTLSize tg = MTLSizeMake(MIN(head_dim, 16), MIN(n_head, 8), MIN(num_tokens, 4));
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Execute all operations
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (cmdBuffer.error) {
            NSLog(@"TensorMetal: Multi-head attention failed: %@", cmdBuffer.error);
            return -1;
        }

        return 0;
    }
}

// Multi-head attention with CPU data (upload/download)
int tensor_metal_attention_multihead(
    const float* q_cpu, const float* k_cache_cpu, const float* v_cache_cpu,
    float* out_cpu,
    uint32_t num_tokens, uint32_t seq_len,
    uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
    uint32_t pos_offset) {

    if (!g_metal_ctx) return -1;
    if (!q_cpu || !k_cache_cpu || !v_cache_cpu || !out_cpu) return -1;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        // Calculate sizes
        size_t q_size = num_tokens * n_head * head_dim * sizeof(float);
        size_t kv_size = seq_len * n_head_kv * head_dim * sizeof(float);
        size_t out_size = num_tokens * n_head * head_dim * sizeof(float);
        size_t scores_size = num_tokens * n_head * seq_len * sizeof(float);

        // Allocate temporary GPU buffers
        id<MTLBuffer> bufQ = [device newBufferWithBytes:q_cpu
                                                 length:q_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [device newBufferWithBytes:k_cache_cpu
                                                 length:kv_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [device newBufferWithBytes:v_cache_cpu
                                                 length:kv_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [device newBufferWithLength:out_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufScores = [device newBufferWithLength:scores_size
                                                      options:MTLResourceStorageModeShared];

        if (!bufQ || !bufK || !bufV || !bufOut || !bufScores) {
            NSLog(@"TensorMetal: Failed to allocate attention buffers");
            return -1;
        }

        // Run GPU computation
        int result = tensor_metal_attention_multihead_gpu(
            (__bridge void*)bufQ, (__bridge void*)bufK, (__bridge void*)bufV,
            (__bridge void*)bufOut, (__bridge void*)bufScores,
            num_tokens, seq_len, n_head, n_head_kv, head_dim, pos_offset);

        if (result != 0) {
            return -1;
        }

        // Copy output back to CPU
        memcpy(out_cpu, bufOut.contents, out_size);

        return 0;
    }
}

// =============================================================================
// DIFFERENTIAL ATTENTION (phi4flash GPU)
// =============================================================================

int tensor_metal_diff_attention(
    const float* q_cpu, const float* k_cache_cpu, const float* v_cache_cpu,
    float* out_cpu, const float* subln_weight,
    uint32_t num_tokens, uint32_t seq_len,
    uint32_t n_head, uint32_t n_head_kv, uint32_t head_dim,
    uint32_t pos_offset, int32_t sliding_window,
    float lambda, float output_scale, float norm_eps) {

    if (!g_metal_ctx) return -1;
    if (!g_metal_ctx.diffAttnScoresPipeline ||
        !g_metal_ctx.diffAttnSoftmaxPipeline ||
        !g_metal_ctx.diffAttnWeightedSumPipeline) return -1;
    if (!q_cpu || !k_cache_cpu || !v_cache_cpu || !out_cpu) return -1;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        uint32_t half_heads = n_head / 2;

        // Calculate sizes
        size_t q_size = num_tokens * n_head * head_dim * sizeof(float);
        size_t kv_size = seq_len * n_head_kv * head_dim * sizeof(float);
        size_t out_size = num_tokens * n_head * head_dim * sizeof(float);
        size_t scores_size = num_tokens * half_heads * seq_len * sizeof(float);
        size_t subln_size = 2 * head_dim * sizeof(float);

        // Allocate GPU buffers
        id<MTLBuffer> bufQ = [device newBufferWithBytes:q_cpu
                                                 length:q_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [device newBufferWithBytes:k_cache_cpu
                                                 length:kv_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV = [device newBufferWithBytes:v_cache_cpu
                                                 length:kv_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [device newBufferWithLength:out_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufScores1 = [device newBufferWithLength:scores_size
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufScores2 = [device newBufferWithLength:scores_size
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSubLN = subln_weight ?
            [device newBufferWithBytes:subln_weight length:subln_size options:MTLResourceStorageModeShared] : nil;

        if (!bufQ || !bufK || !bufV || !bufOut || !bufScores1 || !bufScores2) {
            NSLog(@"TensorMetal: Failed to allocate diff attention buffers");
            return -1;
        }

        // Single command buffer for all operations
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];

        // Step 1: Compute attention scores for both head groups
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.diffAttnScoresPipeline];
            [enc setBuffer:bufQ offset:0 atIndex:0];
            [enc setBuffer:bufK offset:0 atIndex:1];
            [enc setBuffer:bufScores1 offset:0 atIndex:2];
            [enc setBuffer:bufScores2 offset:0 atIndex:3];
            [enc setBytes:&num_tokens length:sizeof(num_tokens) atIndex:4];
            [enc setBytes:&n_head length:sizeof(n_head) atIndex:5];
            [enc setBytes:&n_head_kv length:sizeof(n_head_kv) atIndex:6];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:7];
            [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:8];
            [enc setBytes:&pos_offset length:sizeof(pos_offset) atIndex:9];
            [enc setBytes:&sliding_window length:sizeof(sliding_window) atIndex:10];

            MTLSize grid = MTLSizeMake(seq_len, half_heads, num_tokens);
            MTLSize tg = MTLSizeMake(MIN(seq_len, 16), MIN(half_heads, 8), MIN(num_tokens, 4));
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 2: Softmax for both score sets
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.diffAttnSoftmaxPipeline];
            [enc setBuffer:bufScores1 offset:0 atIndex:0];
            [enc setBuffer:bufScores2 offset:0 atIndex:1];
            [enc setBytes:&num_tokens length:sizeof(num_tokens) atIndex:2];
            [enc setBytes:&half_heads length:sizeof(half_heads) atIndex:3];
            [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:4];

            MTLSize grid = MTLSizeMake(half_heads, num_tokens, 1);
            MTLSize tg = MTLSizeMake(MIN(half_heads, 32), MIN(num_tokens, 8), 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 3: Weighted sum with differential formula
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.diffAttnWeightedSumPipeline];
            [enc setBuffer:bufScores1 offset:0 atIndex:0];
            [enc setBuffer:bufScores2 offset:0 atIndex:1];
            [enc setBuffer:bufV offset:0 atIndex:2];
            [enc setBuffer:bufOut offset:0 atIndex:3];
            if (bufSubLN) {
                [enc setBuffer:bufSubLN offset:0 atIndex:4];
            } else {
                // Set a dummy buffer for NULL subln
                [enc setBuffer:bufOut offset:0 atIndex:4];  // Won't be used
            }
            [enc setBytes:&num_tokens length:sizeof(num_tokens) atIndex:5];
            [enc setBytes:&n_head length:sizeof(n_head) atIndex:6];
            [enc setBytes:&n_head_kv length:sizeof(n_head_kv) atIndex:7];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:8];
            [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:9];
            [enc setBytes:&lambda length:sizeof(lambda) atIndex:10];
            [enc setBytes:&output_scale length:sizeof(output_scale) atIndex:11];
            [enc setBytes:&norm_eps length:sizeof(norm_eps) atIndex:12];

            uint32_t double_head_dim = head_dim * 2;
            MTLSize grid = MTLSizeMake(double_head_dim, half_heads, num_tokens);
            MTLSize tg = MTLSizeMake(MIN(double_head_dim, 16), MIN(half_heads, 8), MIN(num_tokens, 4));
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Execute all operations
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (cmdBuffer.error) {
            NSLog(@"TensorMetal: Diff attention failed: %@", cmdBuffer.error);
            return -1;
        }

        // Copy output back to CPU
        memcpy(out_cpu, bufOut.contents, out_size);

        return 0;
    }
}

// =============================================================================
// GPU KV CACHE (GPU-Native Forward Pass)
// =============================================================================

// Initialize GPU KV cache
int tensor_metal_kv_cache_init(uint32_t n_layer, uint32_t n_ctx,
                               uint32_t n_head_kv, uint32_t head_dim) {
    if (!g_metal_ctx) return -1;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;

        // Free existing cache if any
        g_metal_ctx.gpuKCache = nil;
        g_metal_ctx.gpuVCache = nil;

        uint32_t kv_dim = n_head_kv * head_dim;
        size_t cache_size = (size_t)n_layer * n_ctx * kv_dim * sizeof(float);

        g_metal_ctx.gpuKCache = [device newBufferWithLength:cache_size
                                                    options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuVCache = [device newBufferWithLength:cache_size
                                                    options:MTLResourceStorageModeShared];

        if (!g_metal_ctx.gpuKCache || !g_metal_ctx.gpuVCache) {
            NSLog(@"TensorMetal: Failed to allocate GPU KV cache (%zu bytes each)", cache_size);
            g_metal_ctx.gpuKCache = nil;
            g_metal_ctx.gpuVCache = nil;
            return -1;
        }

        g_metal_ctx.kvCacheNLayer = n_layer;
        g_metal_ctx.kvCacheNCtx = n_ctx;
        g_metal_ctx.kvCacheKvDim = kv_dim;
        g_metal_ctx.kvCacheSeqLen = 0;

        NSLog(@"TensorMetal: Initialized GPU KV cache (%u layers, %u ctx, %u kv_dim, %.1f MB total)",
              n_layer, n_ctx, kv_dim, (float)(cache_size * 2) / (1024 * 1024));
        return 0;
    }
}

// Store K/V vectors into GPU KV cache
int tensor_metal_kv_cache_store(void* k_buf, void* v_buf,
                                uint32_t layer, uint32_t pos,
                                uint32_t num_tokens, uint32_t kv_dim) {
    if (!g_metal_ctx || !g_metal_ctx.gpuKCache || !g_metal_ctx.gpuVCache) return -1;
    if (!k_buf || !v_buf) return -1;
    if (layer >= g_metal_ctx.kvCacheNLayer) return -1;
    if (pos + num_tokens > g_metal_ctx.kvCacheNCtx) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufK = (__bridge id<MTLBuffer>)k_buf;
        id<MTLBuffer> bufV = (__bridge id<MTLBuffer>)v_buf;

        // Calculate layer offset in cache
        uint32_t layer_stride = g_metal_ctx.kvCacheNCtx * g_metal_ctx.kvCacheKvDim;
        uint32_t layer_offset = layer * layer_stride;

        id<MTLCommandBuffer> cmdBuffer = get_command_buffer();

        // Copy K to cache
        {
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
            [encoder setComputePipelineState:g_metal_ctx.kvCacheStoreGpuPipeline];
            [encoder setBuffer:bufK offset:0 atIndex:0];
            [encoder setBuffer:g_metal_ctx.gpuKCache offset:0 atIndex:1];
            [encoder setBytes:&layer_offset length:sizeof(layer_offset) atIndex:2];
            [encoder setBytes:&pos length:sizeof(pos) atIndex:3];
            [encoder setBytes:&num_tokens length:sizeof(num_tokens) atIndex:4];
            [encoder setBytes:&kv_dim length:sizeof(kv_dim) atIndex:5];

            MTLSize gridSize = MTLSizeMake(kv_dim, num_tokens, 1);
            MTLSize threadGroupSize = MTLSizeMake(MIN(kv_dim, 16), MIN(num_tokens, 16), 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
        }

        // Copy V to cache
        {
            id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
            [encoder setComputePipelineState:g_metal_ctx.kvCacheStoreGpuPipeline];
            [encoder setBuffer:bufV offset:0 atIndex:0];
            [encoder setBuffer:g_metal_ctx.gpuVCache offset:0 atIndex:1];
            [encoder setBytes:&layer_offset length:sizeof(layer_offset) atIndex:2];
            [encoder setBytes:&pos length:sizeof(pos) atIndex:3];
            [encoder setBytes:&num_tokens length:sizeof(num_tokens) atIndex:4];
            [encoder setBytes:&kv_dim length:sizeof(kv_dim) atIndex:5];

            MTLSize gridSize = MTLSizeMake(kv_dim, num_tokens, 1);
            MTLSize threadGroupSize = MTLSizeMake(MIN(kv_dim, 16), MIN(num_tokens, 16), 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
            [encoder endEncoding];
        }

        commit_if_not_batch(cmdBuffer);

        // Update sequence length if this extends the cache
        if (pos + num_tokens > g_metal_ctx.kvCacheSeqLen) {
            g_metal_ctx.kvCacheSeqLen = pos + num_tokens;
        }

        return 0;
    }
}

// Get pointer to K cache GPU buffer for a layer
void* tensor_metal_kv_cache_get_k(uint32_t layer) {
    if (!g_metal_ctx || !g_metal_ctx.gpuKCache) return NULL;
    if (layer >= g_metal_ctx.kvCacheNLayer) return NULL;

    // Return the buffer - caller uses layer offset when accessing
    // For layer-specific access, we'd need to create buffer views, but Metal
    // allows specifying offsets when binding buffers
    return (__bridge void*)g_metal_ctx.gpuKCache;
}

// Get pointer to V cache GPU buffer for a layer
void* tensor_metal_kv_cache_get_v(uint32_t layer) {
    if (!g_metal_ctx || !g_metal_ctx.gpuVCache) return NULL;
    if (layer >= g_metal_ctx.kvCacheNLayer) return NULL;
    return (__bridge void*)g_metal_ctx.gpuVCache;
}

// Get current sequence length in KV cache
uint32_t tensor_metal_kv_cache_seq_len(void) {
    if (!g_metal_ctx) return 0;
    return g_metal_ctx.kvCacheSeqLen;
}

// Set current sequence length in KV cache
void tensor_metal_kv_cache_set_seq_len(uint32_t seq_len) {
    if (!g_metal_ctx) return;
    g_metal_ctx.kvCacheSeqLen = seq_len;
}

// Free GPU KV cache
void tensor_metal_kv_cache_free(void) {
    if (!g_metal_ctx) return;

    @autoreleasepool {
        g_metal_ctx.gpuKCache = nil;
        g_metal_ctx.gpuVCache = nil;
        g_metal_ctx.kvCacheNLayer = 0;
        g_metal_ctx.kvCacheNCtx = 0;
        g_metal_ctx.kvCacheKvDim = 0;
        g_metal_ctx.kvCacheSeqLen = 0;
    }
}

// =============================================================================
// COMPLETE GPU-NATIVE ATTENTION LAYER
// =============================================================================

// Forward declaration
void tensor_metal_free_gpu_forward(void);

// Initialize buffers for GPU-native forward pass
int tensor_metal_init_gpu_forward(uint32_t n_layer, uint32_t n_ctx,
                                   uint32_t n_embd, uint32_t n_ff,
                                   uint32_t n_head, uint32_t n_head_kv,
                                   uint32_t n_vocab) {
    if (!g_metal_ctx) return -1;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        uint32_t head_dim = n_embd / n_head;
        uint32_t kv_dim = n_head_kv * head_dim;

        // Initialize KV cache
        if (tensor_metal_kv_cache_init(n_layer, n_ctx, n_head_kv, head_dim) != 0) {
            return -1;
        }

        // Max sequence length for generation (typically 1-16 tokens at a time)
        uint32_t max_seq = 64;

        // ========================================
        // PERSISTENT HIDDEN STATE BUFFERS
        // These stay allocated and are reused every layer
        // ========================================
        size_t hidden_size = max_seq * n_embd * sizeof(float);
        size_t ff_size = max_seq * n_ff * sizeof(float);

        g_metal_ctx.gpuHidden = [device newBufferWithLength:hidden_size
                                                   options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuResidual = [device newBufferWithLength:hidden_size
                                                     options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuNormed = [device newBufferWithLength:hidden_size
                                                   options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuFFNGate = [device newBufferWithLength:ff_size
                                                    options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuFFNUp = [device newBufferWithLength:ff_size
                                                  options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuFFNDown = [device newBufferWithLength:hidden_size
                                                    options:MTLResourceStorageModeShared];

        if (!g_metal_ctx.gpuHidden || !g_metal_ctx.gpuResidual || !g_metal_ctx.gpuNormed ||
            !g_metal_ctx.gpuFFNGate || !g_metal_ctx.gpuFFNUp || !g_metal_ctx.gpuFFNDown) {
            NSLog(@"TensorMetal: Failed to allocate persistent hidden state buffers");
            tensor_metal_free_gpu_forward();
            return -1;
        }

        g_metal_ctx.gpuHiddenNEmbd = n_embd;
        g_metal_ctx.gpuHiddenNFF = n_ff;
        g_metal_ctx.gpuHiddenMaxSeq = max_seq;

        // ========================================
        // ATTENTION SCRATCH BUFFERS
        // ========================================
        size_t q_size = max_seq * n_embd * sizeof(float);
        size_t k_size = max_seq * kv_dim * sizeof(float);
        size_t v_size = max_seq * kv_dim * sizeof(float);
        size_t scores_size = max_seq * n_ctx * sizeof(float);
        size_t attn_out_size = max_seq * kv_dim * sizeof(float);

        g_metal_ctx.gpuQ = [device newBufferWithLength:q_size
                                              options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuK = [device newBufferWithLength:k_size
                                              options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuV = [device newBufferWithLength:v_size
                                              options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuAttnScores = [device newBufferWithLength:scores_size
                                                       options:MTLResourceStorageModeShared];
        g_metal_ctx.gpuAttnOut = [device newBufferWithLength:attn_out_size
                                                    options:MTLResourceStorageModeShared];

        if (!g_metal_ctx.gpuQ || !g_metal_ctx.gpuK || !g_metal_ctx.gpuV ||
            !g_metal_ctx.gpuAttnScores || !g_metal_ctx.gpuAttnOut) {
            NSLog(@"TensorMetal: Failed to allocate attention scratch buffers");
            tensor_metal_free_gpu_forward();
            return -1;
        }

        g_metal_ctx.gpuForwardInitialized = YES;

        size_t total_size = hidden_size * 4 + ff_size * 2 + q_size + k_size + v_size + scores_size + attn_out_size;
        NSLog(@"TensorMetal: Initialized GPU forward pass (%.1f MB persistent buffers)",
              (float)total_size / (1024 * 1024));
        return 0;
    }
}

// Free GPU-native forward pass resources
void tensor_metal_free_gpu_forward(void) {
    if (!g_metal_ctx) return;

    @autoreleasepool {
        tensor_metal_kv_cache_free();

        // Free persistent hidden state buffers
        g_metal_ctx.gpuHidden = nil;
        g_metal_ctx.gpuResidual = nil;
        g_metal_ctx.gpuNormed = nil;
        g_metal_ctx.gpuFFNGate = nil;
        g_metal_ctx.gpuFFNUp = nil;
        g_metal_ctx.gpuFFNDown = nil;
        g_metal_ctx.gpuHiddenNEmbd = 0;
        g_metal_ctx.gpuHiddenNFF = 0;
        g_metal_ctx.gpuHiddenMaxSeq = 0;

        // Free attention scratch buffers
        g_metal_ctx.gpuQ = nil;
        g_metal_ctx.gpuK = nil;
        g_metal_ctx.gpuV = nil;
        g_metal_ctx.gpuAttnScores = nil;
        g_metal_ctx.gpuAttnOut = nil;

        g_metal_ctx.gpuForwardInitialized = NO;
    }
}

// Check if GPU-native forward pass is initialized
int tensor_metal_gpu_forward_available(void) {
    return g_metal_ctx && g_metal_ctx.gpuForwardInitialized ? 1 : 0;
}

// =============================================================================
// PERSISTENT GPU HIDDEN STATE OPERATIONS
// These functions enable zero-copy forward pass by keeping hidden state on GPU
// =============================================================================

void* tensor_metal_get_gpu_hidden(void) {
    if (!g_metal_ctx || !g_metal_ctx.gpuHidden) return NULL;
    return (__bridge void*)g_metal_ctx.gpuHidden;
}

void* tensor_metal_get_gpu_residual(void) {
    if (!g_metal_ctx || !g_metal_ctx.gpuResidual) return NULL;
    return (__bridge void*)g_metal_ctx.gpuResidual;
}

void* tensor_metal_get_gpu_normed(void) {
    if (!g_metal_ctx || !g_metal_ctx.gpuNormed) return NULL;
    return (__bridge void*)g_metal_ctx.gpuNormed;
}

// Upload hidden state from CPU to GPU
int tensor_metal_upload_hidden(const float* src, uint32_t num_elements) {
    if (!g_metal_ctx || !g_metal_ctx.gpuHidden || !src) return -1;

    size_t size = num_elements * sizeof(float);
    if (size > g_metal_ctx.gpuHidden.length) {
        NSLog(@"TensorMetal: upload_hidden size %zu exceeds buffer %lu", size, g_metal_ctx.gpuHidden.length);
        return -1;
    }

    memcpy(g_metal_ctx.gpuHidden.contents, src, size);
    return 0;
}

// Upload normalized hidden state from CPU to gpuNormed (for persistent FFN)
int tensor_metal_upload_normed(const float* src, uint32_t num_elements) {
    if (!g_metal_ctx || !g_metal_ctx.gpuNormed || !src) return -1;

    size_t size = num_elements * sizeof(float);
    if (size > g_metal_ctx.gpuNormed.length) {
        NSLog(@"TensorMetal: upload_normed size %zu exceeds buffer %lu", size, g_metal_ctx.gpuNormed.length);
        return -1;
    }

    memcpy(g_metal_ctx.gpuNormed.contents, src, size);
    return 0;
}

// Download hidden state from GPU to CPU
int tensor_metal_download_hidden(float* dst, uint32_t num_elements) {
    if (!g_metal_ctx || !g_metal_ctx.gpuHidden || !dst) return -1;

    size_t size = num_elements * sizeof(float);
    if (size > g_metal_ctx.gpuHidden.length) return -1;

    memcpy(dst, g_metal_ctx.gpuHidden.contents, size);
    return 0;
}

// GPU-to-GPU copy: residual = hidden
int tensor_metal_copy_hidden_to_residual(uint32_t num_elements) {
    if (!g_metal_ctx || !g_metal_ctx.gpuHidden || !g_metal_ctx.gpuResidual) return -1;

    size_t size = num_elements * sizeof(float);
    if (size > g_metal_ctx.gpuHidden.length || size > g_metal_ctx.gpuResidual.length) return -1;

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cmdBuffer blitCommandEncoder];
        [blit copyFromBuffer:g_metal_ctx.gpuHidden sourceOffset:0
                    toBuffer:g_metal_ctx.gpuResidual destinationOffset:0 size:size];
        [blit endEncoding];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        return 0;
    }
}

// GPU in-place add: hidden += residual
int tensor_metal_add_residual_to_hidden(uint32_t num_elements) {
    if (!g_metal_ctx || !g_metal_ctx.gpuHidden || !g_metal_ctx.gpuResidual) return -1;
    if (!g_metal_ctx.addInplaceGpuPipeline) return -1;

    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

        [enc setComputePipelineState:g_metal_ctx.addInplaceGpuPipeline];
        [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:0];
        [enc setBuffer:g_metal_ctx.gpuResidual offset:0 atIndex:1];
        [enc setBytes:&num_elements length:sizeof(num_elements) atIndex:2];

        MTLSize grid = MTLSizeMake(num_elements, 1, 1);
        MTLSize tg = MTLSizeMake(MIN(num_elements, 256), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        return 0;
    }
}

// RMSNorm: normed = rmsnorm(hidden, weights)
int tensor_metal_rmsnorm_hidden_to_normed(void* weights_buf,
                                           uint32_t M, uint32_t n_embd, float eps) {
    if (!g_metal_ctx || !g_metal_ctx.gpuHidden || !g_metal_ctx.gpuNormed) return -1;
    if (!g_metal_ctx.rmsnormPipeline || !weights_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufWeights = (__bridge id<MTLBuffer>)weights_buf;

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

        [enc setComputePipelineState:g_metal_ctx.rmsnormPipeline];
        [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:0];
        [enc setBuffer:bufWeights offset:0 atIndex:1];
        [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:2];
        [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:3];
        [enc setBytes:&eps length:sizeof(eps) atIndex:4];

        // One thread per row (token)
        MTLSize grid = MTLSizeMake(M, 1, 1);
        MTLSize tg = MTLSizeMake(MIN(M, 64), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        return 0;
    }
}

// Matmul: hidden = normed @ W^T
int tensor_metal_matmul_normed_to_hidden(void* W_buf,
                                          uint32_t M, uint32_t N, uint32_t K) {
    if (!g_metal_ctx || !g_metal_ctx.gpuNormed || !g_metal_ctx.gpuHidden) return -1;
    if (!g_metal_ctx.matmulTransposedBPipeline || !W_buf) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufW = (__bridge id<MTLBuffer>)W_buf;

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

        [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
        [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:0];
        [enc setBuffer:bufW offset:0 atIndex:1];
        [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(M) atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&K length:sizeof(K) atIndex:5];

        MTLSize grid = MTLSizeMake(N, M, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        return 0;
    }
}

// Complete SwiGLU FFN on persistent GPU buffers
// Reads from gpuNormed, writes to gpuHidden. Does NOT add residual.
int tensor_metal_ffn_swiglu_gpu_persistent(void* W1_buf, void* W3_buf, void* W2_buf,
                                            uint32_t M, uint32_t n_ff, uint32_t n_embd) {
    if (!g_metal_ctx || !g_metal_ctx.gpuForwardInitialized) return -1;
    if (!W1_buf || !W3_buf || !W2_buf) return -1;
    if (!g_metal_ctx.gpuNormed || !g_metal_ctx.gpuHidden) return -1;
    if (!g_metal_ctx.gpuFFNGate || !g_metal_ctx.gpuFFNUp) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufW1 = (__bridge id<MTLBuffer>)W1_buf;
        id<MTLBuffer> bufW3 = (__bridge id<MTLBuffer>)W3_buf;
        id<MTLBuffer> bufW2 = (__bridge id<MTLBuffer>)W2_buf;

        // Single command buffer for all 4 operations
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];

        // Op 1: gate = normed @ W1^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:0];
            [enc setBuffer:bufW1 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_ff, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 2: up = normed @ W3^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:0];
            [enc setBuffer:bufW3 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNUp offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_ff, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 3: gate = SiLU(gate) * up (in-place in gate buffer)
        {
            uint32_t n = M * n_ff;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.siluMulPipeline];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuFFNUp offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:2];  // Output to gate (reuse)
            [enc setBytes:&n length:sizeof(n) atIndex:3];
            MTLSize grid = MTLSizeMake(n, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(n, 256), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 4: hidden = gate @ W2^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:0];
            [enc setBuffer:bufW2 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:4];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:5];
            MTLSize grid = MTLSizeMake(n_embd, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (cmdBuffer.error) {
            NSLog(@"TensorMetal: FFN persistent failed: %@", cmdBuffer.error);
            return -1;
        }
        return 0;
    }
}

// Complete FFN with residual in single command buffer
// This combines: copy_to_residual + FFN + add_residual
// Input: gpuNormed contains normalized hidden state
// Output: gpuHidden contains FFN output + residual
int tensor_metal_ffn_swiglu_with_residual(void* W1_buf, void* W3_buf, void* W2_buf,
                                           uint32_t M, uint32_t n_ff, uint32_t n_embd) {
    if (!g_metal_ctx || !g_metal_ctx.gpuForwardInitialized) return -1;
    if (!W1_buf || !W3_buf || !W2_buf) return -1;
    if (!g_metal_ctx.gpuNormed || !g_metal_ctx.gpuHidden) return -1;
    if (!g_metal_ctx.gpuResidual || !g_metal_ctx.gpuFFNGate || !g_metal_ctx.gpuFFNUp) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufW1 = (__bridge id<MTLBuffer>)W1_buf;
        id<MTLBuffer> bufW3 = (__bridge id<MTLBuffer>)W3_buf;
        id<MTLBuffer> bufW2 = (__bridge id<MTLBuffer>)W2_buf;

        // SINGLE command buffer for all 6 operations
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];

        // Op 1: Copy gpuHidden to gpuResidual (blit copy)
        {
            size_t copy_size = M * n_embd * sizeof(float);
            id<MTLBlitCommandEncoder> blit = [cmdBuffer blitCommandEncoder];
            [blit copyFromBuffer:g_metal_ctx.gpuHidden sourceOffset:0
                        toBuffer:g_metal_ctx.gpuResidual destinationOffset:0 size:copy_size];
            [blit endEncoding];
        }

        // Op 2: gate = normed @ W1^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:0];
            [enc setBuffer:bufW1 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_ff, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 3: up = normed @ W3^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:0];
            [enc setBuffer:bufW3 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNUp offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_ff, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 4: gate = SiLU(gate) * up
        {
            uint32_t n = M * n_ff;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.siluMulPipeline];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuFFNUp offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:2];
            [enc setBytes:&n length:sizeof(n) atIndex:3];
            MTLSize grid = MTLSizeMake(n, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(n, 256), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 5: hidden = gate @ W2^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:0];
            [enc setBuffer:bufW2 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:4];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:5];
            MTLSize grid = MTLSizeMake(n_embd, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 6: hidden += residual (in-place add)
        {
            uint32_t n = M * n_embd;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.addInplaceGpuPipeline];
            [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuResidual offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];
            MTLSize grid = MTLSizeMake(n, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(n, 256), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (cmdBuffer.error) {
            NSLog(@"TensorMetal: FFN with residual failed: %@", cmdBuffer.error);
            return -1;
        }
        return 0;
    }
}

// Complete FFN with norm AND residual in SINGLE command buffer
// This combines 7 operations with no intermediate sync:
// 1. Copy gpuHidden to gpuResidual
// 2. RMSNorm gpuHidden to gpuNormed
// 3. FFN gate projection: normed @ W1^T
// 4. FFN up projection: normed @ W3^T
// 5. SiLU multiply: SiLU(gate) * up
// 6. FFN down projection: result @ W2^T
// 7. Residual add: hidden += residual
//
// Input: gpuHidden has pre-FFN hidden state
// Output: gpuHidden contains FFN output with residual added
int tensor_metal_ffn_complete_with_norm(void* norm_buf,
                                         void* W1_buf, void* W3_buf, void* W2_buf,
                                         uint32_t M, uint32_t n_ff, uint32_t n_embd,
                                         float eps) {
    if (!g_metal_ctx || !g_metal_ctx.gpuForwardInitialized) return -1;
    if (!norm_buf || !W1_buf || !W3_buf || !W2_buf) return -1;
    if (!g_metal_ctx.gpuHidden || !g_metal_ctx.gpuNormed) return -1;
    if (!g_metal_ctx.gpuResidual || !g_metal_ctx.gpuFFNGate || !g_metal_ctx.gpuFFNUp) return -1;

    @autoreleasepool {
        id<MTLBuffer> bufNorm = (__bridge id<MTLBuffer>)norm_buf;
        id<MTLBuffer> bufW1 = (__bridge id<MTLBuffer>)W1_buf;
        id<MTLBuffer> bufW3 = (__bridge id<MTLBuffer>)W3_buf;
        id<MTLBuffer> bufW2 = (__bridge id<MTLBuffer>)W2_buf;

        // SINGLE command buffer for all 7 operations
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];

        // Op 1: Copy gpuHidden to gpuResidual (blit copy)
        {
            size_t copy_size = M * n_embd * sizeof(float);
            id<MTLBlitCommandEncoder> blit = [cmdBuffer blitCommandEncoder];
            [blit copyFromBuffer:g_metal_ctx.gpuHidden sourceOffset:0
                        toBuffer:g_metal_ctx.gpuResidual destinationOffset:0 size:copy_size];
            [blit endEncoding];
        }

        // Op 2: RMSNorm gpuHidden to gpuNormed
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.rmsnormPipeline];
            [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:0];
            [enc setBuffer:bufNorm offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:2];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:3];
            [enc setBytes:&eps length:sizeof(eps) atIndex:4];
            MTLSize grid = MTLSizeMake(M, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(M, 64), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 3: gate = normed @ W1^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:0];
            [enc setBuffer:bufW1 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_ff, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 4: up = normed @ W3^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuNormed offset:0 atIndex:0];
            [enc setBuffer:bufW3 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNUp offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_ff, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 5: gate = SiLU(gate) * up
        {
            uint32_t n = M * n_ff;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.siluMulPipeline];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuFFNUp offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:2];
            [enc setBytes:&n length:sizeof(n) atIndex:3];
            MTLSize grid = MTLSizeMake(n, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(n, 256), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 6: hidden = gate @ W2^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:g_metal_ctx.gpuFFNGate offset:0 atIndex:0];
            [enc setBuffer:bufW2 offset:0 atIndex:1];
            [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:4];
            [enc setBytes:&n_ff length:sizeof(n_ff) atIndex:5];
            MTLSize grid = MTLSizeMake(n_embd, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Op 7: hidden += residual (in-place add)
        {
            uint32_t n = M * n_embd;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.addInplaceGpuPipeline];
            [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuResidual offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];
            MTLSize grid = MTLSizeMake(n, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(n, 256), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (cmdBuffer.error) {
            NSLog(@"TensorMetal: FFN complete with norm failed: %@", cmdBuffer.error);
            return -1;
        }
        return 0;
    }
}

// Output projection to CPU logits: logits = hidden @ W^T
int tensor_metal_output_proj_download(float* logits_out, void* W_buf,
                                       uint32_t M, uint32_t n_vocab, uint32_t n_embd) {
    if (!g_metal_ctx || !g_metal_ctx.gpuHidden || !logits_out || !W_buf) return -1;

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        id<MTLBuffer> bufW = (__bridge id<MTLBuffer>)W_buf;

        // Allocate output buffer on GPU
        size_t logits_size = M * n_vocab * sizeof(float);
        id<MTLBuffer> bufLogits = [device newBufferWithLength:logits_size
                                                      options:MTLResourceStorageModeShared];
        if (!bufLogits) return -1;

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

        [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
        [enc setBuffer:g_metal_ctx.gpuHidden offset:0 atIndex:0];
        [enc setBuffer:bufW offset:0 atIndex:1];
        [enc setBuffer:bufLogits offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(M) atIndex:3];
        [enc setBytes:&n_vocab length:sizeof(n_vocab) atIndex:4];
        [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];

        MTLSize grid = MTLSizeMake(n_vocab, M, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Download logits to CPU
        memcpy(logits_out, bufLogits.contents, logits_size);
        return 0;
    }
}

// Complete GPU-native attention layer
// All intermediate data stays on GPU - no CPU-GPU transfers during computation
int tensor_metal_attention_layer_complete(
    void* hidden_in_buf, void* hidden_out_buf,
    void* wq_buf, void* wk_buf, void* wv_buf, void* wo_buf,
    void* norm_buf,
    uint32_t layer_idx,
    uint32_t pos,
    uint32_t M,
    uint32_t n_embd, uint32_t n_head, uint32_t n_head_kv,
    float norm_eps, float rope_theta,
    int use_rope) {

    if (!g_metal_ctx || !g_metal_ctx.gpuForwardInitialized) return -1;
    if (!hidden_in_buf || !hidden_out_buf) return -1;
    if (!wq_buf || !wk_buf || !wv_buf || !wo_buf || !norm_buf) return -1;

    uint32_t head_dim = n_embd / n_head;
    uint32_t kv_dim = n_head_kv * head_dim;
    uint32_t seq_k = pos + M;  // Total KV cache length after this

    @autoreleasepool {
        id<MTLBuffer> bufHiddenIn = (__bridge id<MTLBuffer>)hidden_in_buf;
        id<MTLBuffer> bufHiddenOut = (__bridge id<MTLBuffer>)hidden_out_buf;
        id<MTLBuffer> bufWQ = (__bridge id<MTLBuffer>)wq_buf;
        id<MTLBuffer> bufWK = (__bridge id<MTLBuffer>)wk_buf;
        id<MTLBuffer> bufWV = (__bridge id<MTLBuffer>)wv_buf;
        id<MTLBuffer> bufWO = (__bridge id<MTLBuffer>)wo_buf;
        id<MTLBuffer> bufNorm = (__bridge id<MTLBuffer>)norm_buf;

        // Get scratch buffers
        id<MTLBuffer> bufQ = g_metal_ctx.gpuQ;
        id<MTLBuffer> bufK = g_metal_ctx.gpuK;
        id<MTLBuffer> bufV = g_metal_ctx.gpuV;
        id<MTLBuffer> bufScores = g_metal_ctx.gpuAttnScores;
        id<MTLBuffer> bufAttnOut = g_metal_ctx.gpuAttnOut;

        // Calculate layer offset in KV cache
        uint32_t layer_stride = g_metal_ctx.kvCacheNCtx * g_metal_ctx.kvCacheKvDim;
        uint32_t layer_offset_bytes = layer_idx * layer_stride * sizeof(float);

        // Single command buffer for entire attention layer
        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];

        // Step 1: RMSNorm hidden_in -> normalized (store in bufHiddenOut temporarily)
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.rmsnormPipeline];
            [enc setBuffer:bufHiddenIn offset:0 atIndex:0];
            [enc setBuffer:bufNorm offset:0 atIndex:1];
            [enc setBuffer:bufHiddenOut offset:0 atIndex:2];  // Temp storage
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:3];
            [enc setBytes:&norm_eps length:sizeof(norm_eps) atIndex:4];
            MTLSize grid = MTLSizeMake(M, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(M, 64), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 2: Q projection: Q = normalized @ WQ^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:bufHiddenOut offset:0 atIndex:0];  // Normalized input
            [enc setBuffer:bufWQ offset:0 atIndex:1];
            [enc setBuffer:bufQ offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(n_embd, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 3: K projection: K = normalized @ WK^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:bufHiddenOut offset:0 atIndex:0];
            [enc setBuffer:bufWK offset:0 atIndex:1];
            [enc setBuffer:bufK offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&kv_dim length:sizeof(kv_dim) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(kv_dim, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 4: V projection: V = normalized @ WV^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:bufHiddenOut offset:0 atIndex:0];
            [enc setBuffer:bufWV offset:0 atIndex:1];
            [enc setBuffer:bufV offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&kv_dim length:sizeof(kv_dim) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];
            MTLSize grid = MTLSizeMake(kv_dim, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 5: Apply RoPE to Q and K (if enabled)
        if (use_rope && g_metal_ctx.ropeGpuPipeline) {
            // RoPE on Q [M, n_head, head_dim]
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                [enc setComputePipelineState:g_metal_ctx.ropeGpuPipeline];
                [enc setBuffer:bufQ offset:0 atIndex:0];
                [enc setBytes:&M length:sizeof(M) atIndex:1];
                [enc setBytes:&n_head length:sizeof(n_head) atIndex:2];
                [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
                [enc setBytes:&pos length:sizeof(pos) atIndex:4];
                [enc setBytes:&rope_theta length:sizeof(rope_theta) atIndex:5];
                MTLSize grid = MTLSizeMake(n_head, M, 1);
                MTLSize tg = MTLSizeMake(MIN(n_head, 16), MIN(M, 16), 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }

            // RoPE on K [M, n_head_kv, head_dim]
            {
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                [enc setComputePipelineState:g_metal_ctx.ropeGpuPipeline];
                [enc setBuffer:bufK offset:0 atIndex:0];
                [enc setBytes:&M length:sizeof(M) atIndex:1];
                [enc setBytes:&n_head_kv length:sizeof(n_head_kv) atIndex:2];
                [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
                [enc setBytes:&pos length:sizeof(pos) atIndex:4];
                [enc setBytes:&rope_theta length:sizeof(rope_theta) atIndex:5];
                MTLSize grid = MTLSizeMake(n_head_kv, M, 1);
                MTLSize tg = MTLSizeMake(MIN(n_head_kv, 16), MIN(M, 16), 1);
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                [enc endEncoding];
            }
        }

        // Step 6: Store K/V to cache
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.kvCacheStoreGpuPipeline];
            [enc setBuffer:bufK offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuKCache offset:0 atIndex:1];
            uint32_t layer_offset = layer_idx * g_metal_ctx.kvCacheNCtx * g_metal_ctx.kvCacheKvDim;
            [enc setBytes:&layer_offset length:sizeof(layer_offset) atIndex:2];
            [enc setBytes:&pos length:sizeof(pos) atIndex:3];
            [enc setBytes:&M length:sizeof(M) atIndex:4];
            [enc setBytes:&kv_dim length:sizeof(kv_dim) atIndex:5];
            MTLSize grid = MTLSizeMake(kv_dim, M, 1);
            MTLSize tg = MTLSizeMake(MIN(kv_dim, 16), MIN(M, 16), 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.kvCacheStoreGpuPipeline];
            [enc setBuffer:bufV offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuVCache offset:0 atIndex:1];
            uint32_t layer_offset = layer_idx * g_metal_ctx.kvCacheNCtx * g_metal_ctx.kvCacheKvDim;
            [enc setBytes:&layer_offset length:sizeof(layer_offset) atIndex:2];
            [enc setBytes:&pos length:sizeof(pos) atIndex:3];
            [enc setBytes:&M length:sizeof(M) atIndex:4];
            [enc setBytes:&kv_dim length:sizeof(kv_dim) atIndex:5];
            MTLSize grid = MTLSizeMake(kv_dim, M, 1);
            MTLSize tg = MTLSizeMake(MIN(kv_dim, 16), MIN(M, 16), 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // For GQA, we need to handle Q heads attending to shared KV heads
        // For simplicity, we'll process one head at a time
        // TODO: Optimize with batched attention for multiple heads

        // Step 7-9: Per-head attention (simplified - processes all heads together)
        // Attention scores: scores = Q @ K_cache^T / sqrt(head_dim) with causal mask
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.attentionScoresCausalGpuPipeline];
            [enc setBuffer:bufQ offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuKCache offset:layer_offset_bytes atIndex:1];
            [enc setBuffer:bufScores offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&seq_k length:sizeof(seq_k) atIndex:4];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:5];  // Using head_dim for single-head
            [enc setBytes:&pos length:sizeof(pos) atIndex:6];
            MTLSize grid = MTLSizeMake(seq_k, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Softmax on scores
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.softmaxRowsGpuPipeline];
            [enc setBuffer:bufScores offset:0 atIndex:0];
            [enc setBytes:&M length:sizeof(M) atIndex:1];
            [enc setBytes:&seq_k length:sizeof(seq_k) atIndex:2];
            MTLSize grid = MTLSizeMake(M, 1, 1);
            MTLSize tg = MTLSizeMake(MIN(M, 256), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Attention output: attn_out = scores @ V_cache
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.attentionWeightedSumGpuPipeline];
            [enc setBuffer:bufScores offset:0 atIndex:0];
            [enc setBuffer:g_metal_ctx.gpuVCache offset:layer_offset_bytes atIndex:1];
            [enc setBuffer:bufAttnOut offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&seq_k length:sizeof(seq_k) atIndex:4];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
            MTLSize grid = MTLSizeMake(head_dim, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Step 10: Output projection: hidden_out = attn_out @ WO^T
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_metal_ctx.matmulTransposedBPipeline];
            [enc setBuffer:bufAttnOut offset:0 atIndex:0];
            [enc setBuffer:bufWO offset:0 atIndex:1];
            [enc setBuffer:bufHiddenOut offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:4];
            [enc setBytes:&n_embd length:sizeof(n_embd) atIndex:5];  // WO is [n_embd, n_embd]
            MTLSize grid = MTLSizeMake(n_embd, M, 1);
            MTLSize tg = MTLSizeMake(16, 16, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Execute all operations
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (cmdBuffer.error) {
            NSLog(@"TensorMetal: Attention layer failed: %@", cmdBuffer.error);
            return -1;
        }

        // Update KV cache sequence length
        if (pos + M > g_metal_ctx.kvCacheSeqLen) {
            g_metal_ctx.kvCacheSeqLen = pos + M;
        }

        return 0;
    }
}
