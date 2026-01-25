# EIS Substrate - Metal GPU Backend
# ==================================
#
# Build configuration for Metal compute shaders on macOS.

cmake_minimum_required(VERSION 3.10)

# Metal backend is only available on Apple platforms
if(NOT APPLE)
    message(STATUS "Metal backend: Skipped (non-Apple platform)")
    return()
endif()

message(STATUS "Metal backend: Building for macOS")

# Find Metal and Foundation frameworks
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)

if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
    message(WARNING "Metal backend: Metal or Foundation framework not found")
    return()
endif()

# =============================================================================
# Compile Metal Shaders (optional - can compile at runtime)
# =============================================================================

option(METAL_PRECOMPILE_SHADERS "Precompile Metal shaders to .metallib" OFF)

set(METAL_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/substrate_shaders.metal")
set(METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/substrate_shaders.air")
set(METAL_LIB "${CMAKE_CURRENT_BINARY_DIR}/substrate_shaders.metallib")

if(METAL_PRECOMPILE_SHADERS)
    # Custom command to compile Metal shader to .air
    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND xcrun -sdk macosx metal
                -std=metal2.0
                -O2
                -c ${METAL_SHADER_SOURCE}
                -o ${METAL_AIR}
        DEPENDS ${METAL_SHADER_SOURCE}
        COMMENT "Compiling Metal shaders to AIR"
        VERBATIM
    )

    # Custom command to link .air to .metallib
    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metallib
                ${METAL_AIR}
                -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking Metal library"
        VERBATIM
    )

    # Custom target for Metal shaders
    add_custom_target(substrate_shaders ALL DEPENDS ${METAL_LIB})
else()
    message(STATUS "Metal shaders will be compiled at runtime from source")
    # Copy shader source to build directory for runtime compilation
    add_custom_target(substrate_shaders ALL
        COMMAND ${CMAKE_COMMAND} -E copy
                ${METAL_SHADER_SOURCE}
                ${CMAKE_CURRENT_BINARY_DIR}/substrate_shaders.metal
        DEPENDS ${METAL_SHADER_SOURCE}
        COMMENT "Copying Metal shader source for runtime compilation"
    )
endif()

# =============================================================================
# Metal Backend Library
# =============================================================================

set(METAL_BACKEND_SOURCES
    metal_backend.m
    ../src/substrate_metal.c
)

# Shared library
add_library(substrate_metal SHARED ${METAL_BACKEND_SOURCES})

target_include_directories(substrate_metal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(substrate_metal
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    eis_substrate_v2_static
)

# Enable ARC for Objective-C
target_compile_options(substrate_metal PRIVATE
    -fobjc-arc
    -fmodules
)

# Add dependency on shader compilation/copying
add_dependencies(substrate_metal substrate_shaders)

if(METAL_PRECOMPILE_SHADERS)
    # Copy metallib to library directory
    add_custom_command(TARGET substrate_metal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${METAL_LIB}
                $<TARGET_FILE_DIR:substrate_metal>/substrate_shaders.metallib
        COMMENT "Copying Metal library"
    )
else()
    # Copy shader source to library directory for runtime compilation
    add_custom_command(TARGET substrate_metal POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/substrate_shaders.metal
                $<TARGET_FILE_DIR:substrate_metal>/substrate_shaders.metal
        COMMENT "Copying Metal shader source"
    )
endif()

# Static library (for embedding)
add_library(substrate_metal_static STATIC ${METAL_BACKEND_SOURCES})

target_include_directories(substrate_metal_static PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(substrate_metal_static
    eis_substrate_v2_static
)

target_compile_options(substrate_metal_static PRIVATE
    -fobjc-arc
    -fmodules
)

add_dependencies(substrate_metal_static substrate_shaders)

# =============================================================================
# Test executable
# =============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_metal.c")
    add_executable(test_metal ../tests/test_metal.c)
    target_link_libraries(test_metal
        substrate_metal_static
        eis_substrate_v2_static
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        m
    )
    target_include_directories(test_metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
endif()

# =============================================================================
# Install
# =============================================================================

install(TARGETS substrate_metal substrate_metal_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${METAL_LIB}
    DESTINATION lib
)

install(FILES ../include/substrate_metal.h
    DESTINATION include/eis
)

# =============================================================================
# Output info
# =============================================================================

message(STATUS "Metal backend configured:")
message(STATUS "  Metal framework: ${METAL_FRAMEWORK}")
message(STATUS "  Foundation framework: ${FOUNDATION_FRAMEWORK}")
message(STATUS "  Shader source: ${METAL_SHADER_SOURCE}")
message(STATUS "  Output metallib: ${METAL_LIB}")
