add_library(lewm STATIC
    lewm/host_tensor.cpp
    lewm/gguf_store.cpp
    lewm/hparams.cpp
    lewm/session.cpp
    lewm/rng.cpp
    lewm/graph_builder.cpp
    lewm/world_model.cpp
    lewm/train_graph.cpp
    lewm/dataset.cpp
    lewm/parity.cpp
)
target_include_directories(lewm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lewm PUBLIC ggml)
target_compile_options(lewm PRIVATE -Wall -Wextra -Wpedantic)

add_executable(lewm-parity bin/lewm_parity.cpp)
target_link_libraries(lewm-parity PRIVATE lewm)

add_executable(lewm-rollout bin/lewm_rollout.cpp)
target_link_libraries(lewm-rollout PRIVATE lewm)

add_executable(lewm-gradcheck bin/lewm_gradcheck.cpp)
target_link_libraries(lewm-gradcheck PRIVATE lewm)

add_executable(lewm-train bin/lewm_train.cpp)
target_link_libraries(lewm-train PRIVATE lewm)
