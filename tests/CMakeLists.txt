include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.12
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(doctest)

# Model files and reference dumps produced by tools/export_model.py and tools/dump_reference.py.
set(LEWM_TESTDATA_DIR "${CMAKE_SOURCE_DIR}/testdata" CACHE PATH "directory with *.gguf test fixtures")

add_executable(lewm-tests
    test_main.cpp
    test_host_tensor.cpp
    test_parity.cpp
    test_ggml_ops.cpp
    test_rng.cpp
)
target_link_libraries(lewm-tests PRIVATE lewm doctest::doctest)
target_compile_definitions(lewm-tests PRIVATE LEWM_TESTDATA_DIR="${LEWM_TESTDATA_DIR}")

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(lewm-tests)
