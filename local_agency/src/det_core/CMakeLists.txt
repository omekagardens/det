cmake_minimum_required(VERSION 3.16)
project(det_core C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# Source files
set(DET_CORE_SOURCES
    src/det_core.c
)

# Include directories
include_directories(include)

# Shared library
add_library(det_core SHARED ${DET_CORE_SOURCES})

# Set library properties
set_target_properties(det_core PROPERTIES
    VERSION 0.1.0
    SOVERSION 0
    PUBLIC_HEADER "include/det_core.h"
)

# Math library (for sin, cos, etc.)
target_link_libraries(det_core m)

# Installation
install(TARGETS det_core
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# Optional: Build tests
option(BUILD_TESTS "Build test executables" ON)

if(BUILD_TESTS)
    enable_testing()

    # Simple test executable
    add_executable(det_core_test tests/test_det_core.c)
    target_link_libraries(det_core_test det_core m)

    add_test(NAME basic_test COMMAND det_core_test)
endif()
