#include "lewm/session.hpp"

#include <stdexcept>
#include <thread>

#include "ggml-cpu.h"

namespace lewm {

ggml_tensor* Binder::input(ggml_context* ctx, const std::string& name, const HostTensor& h) {
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, h.n_dims, h.ne.data());
    ggml_set_name(t, name.c_str());
    ggml_set_input(t);
    inputs_.emplace_back(t, &h);
    return t;
}

void Binder::output(const std::string& name, ggml_tensor* t) {
    ggml_set_name(t, name.c_str());
    ggml_set_output(t);
    outputs_.emplace_back(name, t);
}

Session::Session(int n_threads) {
    backend_ = ggml_backend_cpu_init();
    if (!backend_) throw std::runtime_error("ggml_backend_cpu_init failed");
    if (n_threads <= 0) n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0) n_threads = 4;
    ggml_backend_cpu_set_n_threads(backend_, n_threads);
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Session::~Session() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (backend_) ggml_backend_free(backend_);
}

Outputs Session::run(const BuildFn& build, size_t max_nodes) {
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_init failed");

    Outputs out;
    try {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, max_nodes, false);
        Binder b;
        build(ctx, gf, b);
        for (auto& [name, t] : b.outputs_) ggml_build_forward_expand(gf, t);

        // Every run may have a different topology or shape. The allocator's automatic
        // re-planning only triggers on node-count or size growth, so always re-plan.
        if (!ggml_gallocr_reserve(galloc_, gf)) throw std::runtime_error("graph reservation failed");
        if (!ggml_gallocr_alloc_graph(galloc_, gf)) throw std::runtime_error("graph allocation failed");

        for (auto& [t, h] : b.inputs_) {
            if (static_cast<int64_t>(h->data.size()) != ggml_nelements(t)) {
                throw std::runtime_error(std::string("input size mismatch for ") + ggml_get_name(t));
            }
            ggml_backend_tensor_set(t, h->data.data(), 0, ggml_nbytes(t));
        }

        if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed");
        }

        for (auto& [name, t] : b.outputs_) {
            HostTensor h;
            h.n_dims = ggml_n_dims(t);
            for (int i = 0; i < 4; ++i) h.ne[i] = t->ne[i];
            h.data.resize(static_cast<size_t>(ggml_nelements(t)));
            ggml_backend_tensor_get(t, h.data.data(), 0, ggml_nbytes(t));
            out.emplace(name, std::move(h));
        }
    } catch (...) {
        ggml_free(ctx);
        throw;
    }
    ggml_free(ctx);
    return out;
}

}  // namespace lewm
