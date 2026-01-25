# DET Inference Library - Phase 26
# =================================
#
# Native model inference primitives for DET-OS.
# Provides tensor operations, model loading, and sampling.

cmake_minimum_required(VERSION 3.10)

project(det_inference VERSION 0.1.0 LANGUAGES C)

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
)

set(INFERENCE_HEADERS
    include/det_tensor.h
)

# =============================================================================
# Platform Detection
# =============================================================================

if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        message(STATUS "DET Inference: Accelerate framework found")
        set(PLATFORM_LIBS ${ACCELERATE_FRAMEWORK})
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
add_library(det_inference SHARED ${INFERENCE_SOURCES})
target_include_directories(det_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(det_inference PRIVATE ${PLATFORM_LIBS} m)

# Static library
add_library(det_inference_static STATIC ${INFERENCE_SOURCES})
target_include_directories(det_inference_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(det_inference_static PRIVATE ${PLATFORM_LIBS} m)

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
