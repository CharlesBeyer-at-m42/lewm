add_library(lewm STATIC
    lewm/host_tensor.cpp
    lewm/gguf_store.cpp
    lewm/hparams.cpp
    lewm/session.cpp
    lewm/world_model.cpp
    lewm/parity.cpp
)
target_include_directories(lewm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lewm PUBLIC ggml)
target_compile_options(lewm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(lewm-parity bin/lewm_parity.cpp)
target_link_libraries(lewm-parity PRIVATE lewm)

add_executable(lewm-rollout bin/lewm_rollout.cpp)
target_link_libraries(lewm-rollout PRIVATE lewm)
