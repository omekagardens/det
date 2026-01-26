# DET Inference Library - Phase 26
# =================================
#
# Native model inference primitives for DET-OS.
# Provides tensor operations, model loading, and sampling.

cmake_minimum_required(VERSION 3.10)

project(det_inference VERSION 0.1.0 LANGUAGES C OBJC)

# =============================================================================
# Configuration
# =============================================================================

option(DET_INFERENCE_USE_METAL "Enable Metal GPU acceleration" ON)
option(DET_INFERENCE_BUILD_TESTS "Build inference tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# =============================================================================
# Source Files
# =============================================================================

set(INFERENCE_SOURCES
    src/det_tensor.c
    src/det_gguf.c
    src/det_tokenizer.c
    src/det_model.c
)

set(INFERENCE_HEADERS
    include/det_tensor.h
    include/det_gguf.h
    include/det_tokenizer.h
    include/det_model.h
    include/det_tensor_metal.h
)

# Metal sources (Apple only)
if(APPLE AND DET_INFERENCE_USE_METAL)
    set(METAL_SOURCES
        metal/tensor_metal.m
    )
    set(METAL_SHADERS
        metal/tensor_shaders.metal
    )
endif()

# =============================================================================
# Platform Detection
# =============================================================================

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        message(STATUS "DET Inference: Accelerate framework found")
        set(PLATFORM_LIBS ${ACCELERATE_FRAMEWORK})
        # Use new Accelerate LAPACK API (ILP64)
        add_definitions(-DACCELERATE_NEW_LAPACK)
    endif()

    if(DET_INFERENCE_USE_METAL)
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            message(STATUS "DET Inference: Metal GPU acceleration enabled")
            list(APPEND PLATFORM_LIBS ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
            add_definitions(-DDET_USE_METAL=1)
        endif()
    endif()
else()
    # Try to find OpenBLAS on non-Apple platforms
    find_package(BLAS)
    if(BLAS_FOUND)
        message(STATUS "DET Inference: BLAS found")
        set(PLATFORM_LIBS ${BLAS_LIBRARIES})
    endif()
endif()

# =============================================================================
# Libraries
# =============================================================================

# Shared library
add_library(det_inference SHARED ${INFERENCE_SOURCES} ${METAL_SOURCES})
target_include_directories(det_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(det_inference PRIVATE ${PLATFORM_LIBS} m)

# Static library
add_library(det_inference_static STATIC ${INFERENCE_SOURCES} ${METAL_SOURCES})
target_include_directories(det_inference_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(det_inference_static PRIVATE ${PLATFORM_LIBS} m)

# Objective-C settings for Metal
if(APPLE AND DET_INFERENCE_USE_METAL)
    set_source_files_properties(${METAL_SOURCES} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    # Copy Metal shaders to build directory
    add_custom_command(TARGET det_inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/metal/tensor_shaders.metal
                $<TARGET_FILE_DIR:det_inference>/tensor_shaders.metal
        COMMENT "Copying Metal shaders"
    )
endif()

# Compiler options
target_compile_options(det_inference PRIVATE -Wall -Wextra -O2)
target_compile_options(det_inference_static PRIVATE -Wall -Wextra -O2)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(det_inference PRIVATE -g -O0)
    target_compile_options(det_inference_static PRIVATE -g -O0)
endif()

# =============================================================================
# Tests
# =============================================================================

if(DET_INFERENCE_BUILD_TESTS)
    enable_testing()

    add_executable(test_tensor tests/test_tensor.c)
    target_link_libraries(test_tensor det_inference_static ${PLATFORM_LIBS} m)
    target_include_directories(test_tensor PRIVATE include)
    add_test(NAME tensor_tests COMMAND test_tensor)

    add_executable(test_gguf tests/test_gguf.c)
    target_link_libraries(test_gguf det_inference_static ${PLATFORM_LIBS} m)
    target_include_directories(test_gguf PRIVATE include)
    add_test(NAME gguf_tests COMMAND test_gguf)
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS det_inference det_inference_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${INFERENCE_HEADERS}
    DESTINATION include/det
)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "DET Inference Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Metal GPU: ${DET_INFERENCE_USE_METAL}")
message(STATUS "  Platform libs: ${PLATFORM_LIBS}")
