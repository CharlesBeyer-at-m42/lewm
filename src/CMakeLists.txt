# The lewm library and its binaries are added here as they are written.
add_library(lewm INTERFACE)
target_link_libraries(lewm INTERFACE ggml)
target_include_directories(lewm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
