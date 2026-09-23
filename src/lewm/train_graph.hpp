#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "lewm/gguf_store.hpp"
#include "lewm/graph_builder.hpp"
#include "lewm/hparams.hpp"
#include "lewm/host_tensor.hpp"

namespace lewm {

struct TrainConfig {
    int64_t batch = 16;        // B, samples per optimizer step (RDMReg sorts across the whole batch)
    float lr = 1e-4f;          // muP base learning rate
    float weight_decay = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float reg_weight = 0.5f;   // RDMReg weight
    int64_t num_projections = 8192;
    size_t max_nodes = 32768;  // graph capacity for forward + backward
};

struct Losses {
    float z = 0;      // MSE between predicted and target codes
    float reg = 0;    // sliced-Wasserstein RDMReg term (unweighted)
    float total = 0;  // z + reg_weight * reg
};

/// One static training graph on ggml: forward, backward into persistent gradient accumulators,
/// and a separate AdamW graph with per-tensor muP learning rates. See docs/adr/0004.
///
/// Inputs per step, in ggml shape order:
///   visual        {W, H, 3, T*B}   normalised frames, frame-major within each sample: index t + T*b
///   actions       {A, T, B}        chunked normalised actions
///   proj          {C, K}           unit-norm random projections for RDMReg
///   target_sorted {B, K, TP}       projected link(GN_p) samples, sorted along B per (k, tp)
class TrainGraph {
public:
    TrainGraph(GgufStore& weights, const Hparams& hp, ggml_backend_t backend, TrainConfig cfg);
    ~TrainGraph();
    TrainGraph(const TrainGraph&) = delete;
    TrainGraph& operator=(const TrainGraph&) = delete;

    const TrainConfig& config() const { return cfg_; }
    int64_t frames_per_sample() const { return T_; }

    /// Zero the accumulators, run forward and backward once. Returns the losses.
    Losses forward_backward(const HostTensor& visual, const HostTensor& actions, const HostTensor& proj,
                            const HostTensor& target_sorted);

    /// Apply one AdamW step with the accumulated gradients (iteration counter starts at 1).
    void optimizer_step();

    /// Read a parameter's accumulated gradient or current value.
    HostTensor grad(const std::string& name) const;
    /// Overwrite a parameter's accumulated gradient (used to test the optimizer in isolation).
    void set_grad(const std::string& name, const HostTensor& g);
    HostTensor param(const std::string& name) const;
    void set_param(const std::string& name, const HostTensor& value);
    std::vector<std::string> param_names() const;

    /// Zero AdamW moments and the iteration counter.
    void reset_optimizer();

    /// Codes of the current batch after forward: {C, P, T, B}.
    HostTensor codes() const;

    int64_t iteration() const { return iter_; }
    void set_iteration(int64_t it) { iter_ = it; }

    /// Optimizer state access for checkpoint sidecars.
    HostTensor moment_m(const std::string& name) const;
    HostTensor moment_v(const std::string& name) const;
    void set_moments(const std::string& name, const HostTensor& m, const HostTensor& v);

private:
    struct Param {
        std::string name;
        ggml_tensor* tensor = nullptr;
        ggml_tensor* grad_acc = nullptr;
        ggml_tensor* m = nullptr;
        ggml_tensor* v = nullptr;
        ggml_tensor* adamw_params = nullptr;  // 7 floats
        float lr_mult = 1.0f;
        bool weight_decay = true;
    };

    GgufStore& weights_;
    Hparams hp_;
    ggml_backend_t backend_;
    TrainConfig cfg_;
    GraphBuilder gb_;
    int64_t T_ = 0;  // frames per sample = num_hist + 1
    int64_t iter_ = 0;

    std::vector<Param> params_;
    std::map<std::string, size_t> param_index_;

    ggml_context* ctx_static_ = nullptr;   // grad accumulators, moments, adamw params
    ggml_backend_buffer_t buf_static_ = nullptr;
    ggml_context* ctx_compute_ = nullptr;  // forward + backward graph tensors
    ggml_cgraph* gf_ = nullptr;
    ggml_gallocr_t galloc_ = nullptr;
    ggml_context* ctx_opt_ = nullptr;
    ggml_cgraph* gopt_ = nullptr;
    ggml_gallocr_t galloc_opt_ = nullptr;

    // graph inputs and outputs
    ggml_tensor* in_visual_ = nullptr;
    ggml_tensor* in_actions_ = nullptr;
    ggml_tensor* in_proj_ = nullptr;
    ggml_tensor* in_target_sorted_ = nullptr;
    ggml_tensor* in_mask_ = nullptr;
    ggml_tensor* in_token_zeros_ = nullptr;
    ggml_tensor* out_z_ = nullptr;
    ggml_tensor* out_loss_z_ = nullptr;
    ggml_tensor* out_loss_reg_ = nullptr;
    ggml_tensor* out_loss_ = nullptr;
    HostTensor mask_host_;
    HostTensor token_zeros_host_;

    void collect_params();
    void build_forward_backward();
    void build_optimizer();
    const Param& find(const std::string& name) const;
};

}  // namespace lewm
