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
@property (nonatomic, strong) id<MTLComputePipelineState> matvecPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> addPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> mulPipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> siluPipeline;
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

    // Try to load precompiled metallib
    NSString *libPath = [[NSBundle mainBundle] pathForResource:@"tensor_shaders"
                                                        ofType:@"metallib"];
    if (libPath) {
        NSURL *libURL = [NSURL fileURLWithPath:libPath];
        _library = [_device newLibraryWithURL:libURL error:&error];
    }

    // Fall back to runtime compilation
    if (!_library) {
        // Look for .metal source in same directory as executable
        NSString *execPath = [[NSBundle mainBundle] executablePath];
        NSString *execDir = [execPath stringByDeletingLastPathComponent];
        NSString *shaderPath = [execDir stringByAppendingPathComponent:@"tensor_shaders.metal"];

        NSString *source = [NSString stringWithContentsOfFile:shaderPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            // Try relative path from current directory
            shaderPath = @"tensor_shaders.metal";
            source = [NSString stringWithContentsOfFile:shaderPath
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
        }

        if (source) {
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = YES;
            _library = [_device newLibraryWithSource:source options:options error:&error];
        }
    }

    if (!_library) {
        NSLog(@"TensorMetal: Failed to load shader library: %@", error);
        return NO;
    }

    // Create compute pipelines
    _matmulPipeline = [self createPipeline:@"matmul_f32"];
    _matvecPipeline = [self createPipeline:@"matvec_f32"];
    _addPipeline = [self createPipeline:@"add_f32"];
    _mulPipeline = [self createPipeline:@"mul_f32"];
    _siluPipeline = [self createPipeline:@"silu_f32"];
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
