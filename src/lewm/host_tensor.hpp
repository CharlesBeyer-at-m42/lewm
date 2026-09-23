#pragma once

#include <array>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace lewm {

/// A dense fp32 tensor in host memory using ggml's dimension convention:
/// ne[0] is the fastest-varying (innermost) dimension. A numpy array of shape
/// (T, P, D) therefore has ne = {D, P, T}.
struct HostTensor {
    std::array<int64_t, 4> ne{1, 1, 1, 1};
    int n_dims = 0;
    std::vector<float> data;

    int64_t numel() const { return ne[0] * ne[1] * ne[2] * ne[3]; }

    std::string shape_str() const;  // numpy order, e.g. "(4, 257, 384)"

    static HostTensor zeros(std::initializer_list<int64_t> ne_ggml);
};

}  // namespace lewm
