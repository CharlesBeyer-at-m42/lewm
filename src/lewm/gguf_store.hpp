#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "lewm/host_tensor.hpp"

namespace lewm {

/// A GGUF file loaded onto a backend: metadata plus every tensor, allocated and filled.
/// Used for both model files (weights) and reference dumps (inputs and expected outputs).
class GgufStore {
public:
    GgufStore(const std::string& path, ggml_backend_t backend);
    ~GgufStore();
    GgufStore(const GgufStore&) = delete;
    GgufStore& operator=(const GgufStore&) = delete;

    /// Tensor by name; throws std::runtime_error if absent.
    ggml_tensor* tensor(const std::string& name) const;
    ggml_tensor* find(const std::string& name) const;  // nullptr if absent
    std::vector<std::string> tensor_names() const;

    /// Copy a tensor back to host memory.
    HostTensor read(const std::string& name) const;

    bool has_key(const std::string& key) const;
    uint32_t get_u32(const std::string& key) const;
    float get_f32(const std::string& key) const;
    bool get_bool(const std::string& key) const;
    std::string get_str(const std::string& key) const;

    const std::string& path() const { return path_; }
    const gguf_context* gguf() const { return gguf_; }

private:
    std::string path_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;

    int64_t key_id(const std::string& key) const;
};

}  // namespace lewm
