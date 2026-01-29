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

// Compute pipelines
@property (nonatomic, strong) id<MTLComputePipelineState> matmulPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> matmulTransposedBPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> matvecPipeline;
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

    @autoreleasepool {
        id<MTLDevice> device = g_metal_ctx.device;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)B_buf;

        // Create temporary buffers for A and C (small - just activations)
        id<MTLBuffer> bufA = [device newBufferWithBytes:A
                                                 length:M * K * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [device newBufferWithLength:M * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuffer = [g_metal_ctx.commandQueue commandBuffer];
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

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        // Copy result back
        memcpy(C, bufC.contents, M * N * sizeof(float));

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
