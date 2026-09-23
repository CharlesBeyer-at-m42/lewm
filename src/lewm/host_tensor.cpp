#include "lewm/host_tensor.hpp"

#include <sstream>

namespace lewm {

std::string HostTensor::shape_str() const {
    std::ostringstream os;
    os << "(";
    for (int i = n_dims - 1; i >= 0; --i) {
        os << ne[i];
        if (i > 0) os << ", ";
    }
    os << ")";
    return os.str();
}

HostTensor HostTensor::zeros(std::initializer_list<int64_t> ne_ggml) {
    HostTensor t;
    t.n_dims = static_cast<int>(ne_ggml.size());
    int i = 0;
    for (int64_t n : ne_ggml) t.ne[i++] = n;
    t.data.assign(static_cast<size_t>(t.numel()), 0.0f);
    return t;
}

}  // namespace lewm
