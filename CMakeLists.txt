cmake_minimum_required(VERSION 3.28)
project(lewm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LEWM_BUILD_TESTS "Build LeWM tests" ON)

# ggml: pinned submodule. Backends are opt-in through ggml's own options (GGML_CUDA, ...).
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(ggml)

add_subdirectory(src)

if(LEWM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
