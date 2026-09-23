include(FetchContent)
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.12
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(doctest)

# Test executables are added here as they are written, e.g.:
# add_executable(test_encoder test_encoder.cpp)
# target_link_libraries(test_encoder PRIVATE lewm doctest::doctest)
# add_test(NAME encoder COMMAND test_encoder)
