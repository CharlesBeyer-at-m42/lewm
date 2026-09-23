#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "lewm/host_tensor.hpp"

namespace lewm {

/// Collects the graph's named inputs (with host data to upload) and named outputs (to download).
class Binder {
public:
    /// Create an input tensor in `ctx` shaped like `h` and remember to upload `h.data` after allocation.
    ggml_tensor* input(ggml_context* ctx, const std::string& name, const HostTensor& h);
    /// Mark `t` as an output to fetch under `name` after compute.
    void output(const std::string& name, ggml_tensor* t);

private:
    friend class Session;
    std::vector<std::pair<ggml_tensor*, const HostTensor*>> inputs_;
    std::vector<std::pair<std::string, ggml_tensor*>> outputs_;
};

using Outputs = std::map<std::string, HostTensor>;
using BuildFn = std::function<void(ggml_context* ctx, ggml_cgraph* gf, Binder& b)>;

/// Owns a backend and a graph allocator. Each `run` builds a fresh graph, allocates it,
/// uploads inputs, computes, and downloads outputs. Graph shapes may differ between runs.
class Session {
public:
    /// device: "cpu" or "gpu" (the first GPU device ggml registered, e.g. CUDA0).
    explicit Session(int n_threads = 0, const std::string& device = "cpu");
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    ggml_backend_t backend() const { return backend_; }
    const std::string& device() const { return device_; }

    Outputs run(const BuildFn& build, size_t max_nodes = 8192);

private:
    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    std::string device_;
};

}  // namespace lewm
