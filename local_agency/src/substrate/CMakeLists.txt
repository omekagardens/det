# EIS Substrate - Minimal Execution Layer
# =======================================
#
# Substrate v2: DET-aware with phases, proposals, typed refs.
# NO DET physics - those are in physics.ex (Existence-Lang).

cmake_minimum_required(VERSION 3.10)
project(eis_substrate C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")
    set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Include directories
include_directories(include)

# =============================================================================
# Substrate v2 (DET-aware, GPU-ready)
# =============================================================================

set(SUBSTRATE_V2_SOURCES
    src/eis_substrate_v2.c
)

# Shared library
add_library(eis_substrate_v2 SHARED ${SUBSTRATE_V2_SOURCES})
target_link_libraries(eis_substrate_v2 m)

# Static library
add_library(eis_substrate_v2_static STATIC ${SUBSTRATE_V2_SOURCES})
target_link_libraries(eis_substrate_v2_static m)

# Test executable
add_executable(test_substrate_v2 tests/test_substrate_v2.c)
target_link_libraries(test_substrate_v2 eis_substrate_v2_static m)

# =============================================================================
# Substrate v1 (Legacy, pure execution)
# =============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/eis_substrate.c")
    set(SUBSTRATE_V1_SOURCES
        src/eis_substrate.c
    )

    add_library(eis_substrate SHARED ${SUBSTRATE_V1_SOURCES})
    target_link_libraries(eis_substrate m)

    add_library(eis_substrate_static STATIC ${SUBSTRATE_V1_SOURCES})
    target_link_libraries(eis_substrate_static m)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_substrate.c")
        add_executable(test_substrate tests/test_substrate.c)
        target_link_libraries(test_substrate eis_substrate_static m)
    endif()
endif()

# =============================================================================
# Install
# =============================================================================

install(TARGETS eis_substrate_v2 eis_substrate_v2_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    include/eis_substrate_v2.h
    include/substrate_types.h
    include/effect_table.h
    DESTINATION include/eis
)

# =============================================================================
# Output info
# =============================================================================

message(STATUS "EIS Substrate v2 - DET-Aware Execution Layer")
message(STATUS "  Phase-based execution (READ → PROPOSE → CHOOSE → COMMIT)")
message(STATUS "  Proposal buffers and effect table")
message(STATUS "  GPU-ready SoA memory layout")
message(STATUS "  NO DET physics - those are in physics.ex")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
