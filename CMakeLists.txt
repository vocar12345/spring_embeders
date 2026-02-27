cmake_minimum_required(VERSION 3.21)
project(FruchtermanReingold LANGUAGES CXX)

# ── C++20 ────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── FetchContent: GLM ────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# ── Shared compiler options (applied to all targets) ─────────────────────────
set(FR_COMPILE_OPTIONS
    $<$<CXX_COMPILER_ID:GNU,Clang>: -Wall -Wextra -Wpedantic -O2>
    $<$<CXX_COMPILER_ID:MSVC>:      /W4 /O2>
)

set(FR_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# ── Executable 1: fr_layout ──────────────────────────────────────────────────
add_executable(fr_layout src/main.cpp)

target_include_directories(fr_layout PRIVATE ${FR_INCLUDE_DIR})
target_link_libraries(fr_layout PRIVATE glm::glm)
target_compile_options(fr_layout PRIVATE ${FR_COMPILE_OPTIONS})

# ── Executable 2: fr_benchmark ───────────────────────────────────────────────
add_executable(fr_benchmark src/benchmark.cpp)

target_include_directories(fr_benchmark PRIVATE ${FR_INCLUDE_DIR})
target_link_libraries(fr_benchmark PRIVATE glm::glm)
target_compile_options(fr_benchmark PRIVATE ${FR_COMPILE_OPTIONS})
