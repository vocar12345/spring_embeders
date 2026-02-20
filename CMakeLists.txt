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

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(fr_layout
    src/main.cpp
)

target_include_directories(fr_layout PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(fr_layout PRIVATE glm::glm)

# ── Compiler warnings ────────────────────────────────────────────────────────
target_compile_options(fr_layout PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>: -Wall -Wextra -Wpedantic -O2>
    $<$<CXX_COMPILER_ID:MSVC>:      /W4 /O2>
)