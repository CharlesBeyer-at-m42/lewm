#include "lewm/train_graph.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "gguf.h"

namespace lewm {

namespace {

HostTensor read_tensor(const ggml_tensor* t) {
    HostTensor h;
    h.n_dims = ggml_n_dims(t);
    for (int i = 0; i < 4; ++i) h.ne[i] = t->ne[i];
    h.data.resize(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, h.data.data(), 0, ggml_nbytes(t));
    return h;
}

void upload(ggml_tensor* t, const HostTensor& h, const char* what) {
    if (static_cast<int64_t>(h.data.size()) != ggml_nelements(t)) {
        throw std::runtime_error(std::string("size mismatch for ") + what + ": got " + h.shape_str());
    }
    ggml_backend_tensor_set(t, h.data.data(), 0, ggml_nbytes(t));
}

/// Write named device tensors to a GGUF file, copying through host memory.
void write_gguf(const std::string& path, const std::vector<std::pair<std::string, ggml_tensor*>>& tensors,
                const gguf_context* copy_kv_from, const std::map<std::string, uint32_t>& u32_meta) {
    size_t bytes = 1024;
    for (auto& [name, t] : tensors) bytes += ggml_nbytes(t) + ggml_tensor_overhead();
    ggml_init_params ip{};
    ip.mem_size = bytes;
    ip.no_alloc = false;
    ggml_context* host = ggml_init(ip);
    if (!host) throw std::runtime_error("failed to allocate host context for saving");

    gguf_context* g = gguf_init_empty();
    if (copy_kv_from) gguf_set_kv(g, copy_kv_from);
    gguf_set_val_str(g, "general.architecture", "lewm");
    for (auto& [k, v] : u32_meta) gguf_set_val_u32(g, k.c_str(), v);
    for (auto& [name, t] : tensors) {
        ggml_tensor* h = ggml_new_tensor(host, t->type, ggml_n_dims(t), t->ne);
        ggml_set_name(h, name.c_str());
        ggml_backend_tensor_get(t, h->data, 0, ggml_nbytes(t));
        gguf_add_tensor(g, h);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(host);
    if (!ok) throw std::runtime_error("failed to write " + path);
}

}  // namespace

TrainGraph::TrainGraph(GgufStore& weights, const Hparams& hp, ggml_backend_t backend, TrainConfig cfg)
    : weights_(weights), hp_(hp), backend_(backend), cfg_(cfg), gb_(weights, hp_, /*training=*/cfg.backward) {
    T_ = static_cast<int64_t>(hp_.predictor_num_frames) + 1;
    if (cfg_.backward) collect_params();
    build_graph();
    if (cfg_.backward) build_optimizer();
}

TrainGraph::~TrainGraph() {
    if (galloc_opt_) ggml_gallocr_free(galloc_opt_);
    if (ctx_opt_) ggml_free(ctx_opt_);
    if (galloc_) ggml_gallocr_free(galloc_);
    if (ctx_compute_) ggml_free(ctx_compute_);
    if (buf_static_) ggml_backend_buffer_free(buf_static_);
    if (ctx_static_) ggml_free(ctx_static_);
}

void TrainGraph::collect_params() {
    // Every weight tensor is trainable. muP metadata comes from the model file.
    std::vector<std::string> names = weights_.tensor_names();
    const size_t n = names.size();

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (4 * n + 8);
    ip.no_alloc = true;
    ctx_static_ = ggml_init(ip);

    for (const std::string& name : names) {
        Param p;
        p.name = name;
        p.tensor = weights_.tensor(name);
        ggml_set_param(p.tensor);
        p.grad_acc = ggml_dup_tensor(ctx_static_, p.tensor);
        ggml_format_name(p.grad_acc, "grad.%s", name.c_str());
        p.m = ggml_dup_tensor(ctx_static_, p.tensor);
        ggml_format_name(p.m, "m.%s", name.c_str());
        p.v = ggml_dup_tensor(ctx_static_, p.tensor);
        ggml_format_name(p.v, "v.%s", name.c_str());
        p.adamw_params = ggml_new_tensor_1d(ctx_static_, GGML_TYPE_F32, 7);
        ggml_format_name(p.adamw_params, "adamw.%s", name.c_str());
        const std::string k_mult = "lewm.mup.lr_mult." + name;
        const std::string k_wd = "lewm.mup.wd." + name;
        if (weights_.has_key(k_mult)) p.lr_mult = weights_.get_f32(k_mult);
        if (weights_.has_key(k_wd)) p.weight_decay = weights_.get_bool(k_wd);
        param_index_[name] = params_.size();
        params_.push_back(p);
    }
    buf_static_ = ggml_backend_alloc_ctx_tensors(ctx_static_, backend_);
    if (!buf_static_) throw std::runtime_error("failed to allocate optimizer state");
    for (Param& p : params_) {
        ggml_set_zero(p.grad_acc);
        ggml_set_zero(p.m);
        ggml_set_zero(p.v);
        ggml_set_zero(p.adamw_params);
    }
}

void TrainGraph::build_graph() {
    const int64_t B = cfg_.batch;
    const int64_t T = T_;
    const int64_t C = hp_.code_dim;
    const int64_t P = hp_.predictor_num_patches;
    const int64_t K = cfg_.num_projections;
    const int64_t TP = T * P;
    const int64_t H = hp_.predictor_num_frames;  // = T - 1

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * cfg_.max_nodes + ggml_graph_overhead_custom(cfg_.max_nodes, cfg_.backward);
    ip.no_alloc = true;
    ctx_compute_ = ggml_init(ip);
    ggml_context* ctx = ctx_compute_;

    auto input = [&](const char* name, ggml_tensor* t) {
        ggml_set_name(t, name);
        ggml_set_input(t);
        return t;
    };
    in_visual_ = input("in.visual", ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hp_.image_size, hp_.image_size, 3, T * B));
    in_actions_ = input("in.actions", ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp_.action_dim, T, B));
    in_proj_ = input("rdm.proj", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, K));
    if (cfg_.target_from_base) {
        in_base_ = input("rdm.base", ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, TP, B));
    } else {
        in_target_sorted_ = input("rdm.target_sorted", ggml_new_tensor_3d(ctx, GGML_TYPE_F32, B, K, TP));
    }
    mask_host_ = GraphBuilder::causal_mask(H, P);
    in_mask_ = input("predictor.mask", ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H * P, H * P));
    token_zeros_host_ = gb_.token_zeros_host(T * B);
    in_token_zeros_ = input("encoder.token_zeros", ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hp_.dim, hp_.encoder_tokens(), T * B));

    // dropout masks become inputs, created on demand while the predictor graph is built
    mask_specs_.clear();
    in_dropout_.clear();
    if (cfg_.dropout > 0.0f) {
        gb_.set_dropout_masks([&](const std::string& site, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
            MaskSpec spec;
            spec.site = site;
            spec.ne = {n0, n1, n2, n3};
            ggml_tensor* m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, n2, n3);
            ggml_set_name(m, ("dropout." + site).c_str());
            ggml_set_input(m);
            mask_specs_.push_back(spec);
            in_dropout_.push_back(m);
            return m;
        });
    }

    // ---- forward: encoder over all frames, link, action codes, teacher-forced prediction ----
    ggml_tensor* u = gb_.encoder(ctx, in_visual_, in_token_zeros_, nullptr);  // {C, P, T*B}
    ggml_tensor* z = gb_.link(ctx, u);
    z = ggml_reshape_4d(ctx, z, C, P, T, B);
    out_z_ = z;
    ggml_set_name(out_z_, "fwd.z");
    ggml_set_output(out_z_);

    ggml_tensor* a = gb_.action_codes(ctx, in_actions_);  // {C, T, B}

    ggml_tensor* z_src = ggml_cont(ctx, ggml_view_4d(ctx, z, C, P, H, B, z->nb[1], z->nb[2], z->nb[3], 0));
    ggml_tensor* z_tgt = ggml_cont(ctx, ggml_view_4d(ctx, z, C, P, H, B, z->nb[1], z->nb[2], z->nb[3], z->nb[2]));
    ggml_tensor* a_src = ggml_cont(ctx, ggml_view_3d(ctx, a, C, H, B, a->nb[1], a->nb[2], 0));

    ggml_tensor* u_pred = gb_.predictor(ctx, z_src, a_src, in_mask_, nullptr);  // {C, P, H, B}
    ggml_tensor* z_pred = gb_.link(ctx, u_pred);

    ggml_tensor* diff = ggml_sub(ctx, z_pred, z_tgt);
    ggml_tensor* z_loss = ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, diff)), 1.0f / static_cast<float>(ggml_nelements(diff)));
    out_loss_z_ = z_loss;
    ggml_set_name(out_loss_z_, "loss.z");
    ggml_set_output(out_loss_z_);

    // ---- RDMReg, agg "b": samples along B, one SWD per (t, p) token slot ----
    ggml_tensor* z3 = ggml_reshape_3d(ctx, z, C, TP, B);
    ggml_tensor* pz = ggml_mul_mat(ctx, in_proj_, z3);                      // {K, TP, B}
    ggml_tensor* pz_t = ggml_cont(ctx, ggml_permute(ctx, pz, 1, 2, 0, 3));  // {B, K, TP}
    ggml_tensor* idx = ggml_argsort(ctx, pz_t, GGML_SORT_ORDER_ASC);        // {B, K, TP} i32
    ggml_tensor* idx2 = ggml_reshape_2d(ctx, idx, B, K * TP);

    ggml_tensor* target_sorted;  // {B, K, TP}, constant
    if (cfg_.target_from_base) {
        // target = link(base) projected, then sorted along B (a gather by argsort; no gradients flow here)
        ggml_tensor* tgt = gb_.link(ctx, in_base_);                                 // {C, TP, B}
        ggml_tensor* pt = ggml_mul_mat(ctx, in_proj_, tgt);                          // {K, TP, B}
        ggml_tensor* pt_t = ggml_cont(ctx, ggml_permute(ctx, pt, 1, 2, 0, 3));      // {B, K, TP}
        ggml_tensor* tidx = ggml_argsort(ctx, pt_t, GGML_SORT_ORDER_ASC);
        ggml_tensor* gathered = ggml_get_rows(ctx, ggml_reshape_3d(ctx, pt_t, 1, B, K * TP),
                                              ggml_reshape_2d(ctx, tidx, B, K * TP));  // {1, B, K*TP}
        target_sorted = ggml_reshape_3d(ctx, gathered, B, K, TP);
    } else {
        target_sorted = in_target_sorted_;
    }
    ggml_tensor* tgt_rows = ggml_reshape_3d(ctx, target_sorted, 1, B, K * TP);
    ggml_tensor* aligned_dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, B, K * TP);
    ggml_set_name(aligned_dst, "rdm.aligned_dst");
    ggml_tensor* aligned = ggml_set_rows(ctx, aligned_dst, tgt_rows, idx2);  // aligned[idx[i]] = target_sorted[i]
    aligned = ggml_reshape_3d(ctx, aligned, B, K, TP);
    ggml_tensor* rdiff = ggml_sub(ctx, pz_t, aligned);
    ggml_tensor* reg = ggml_scale(ctx, ggml_sum(ctx, ggml_sqr(ctx, rdiff)), 1.0f / static_cast<float>(B * K * TP));
    out_loss_reg_ = reg;
    ggml_set_name(out_loss_reg_, "loss.reg");
    ggml_set_output(out_loss_reg_);

    out_loss_ = ggml_add(ctx, z_loss, ggml_scale(ctx, reg, cfg_.reg_weight));
    ggml_set_name(out_loss_, "loss.total");
    ggml_set_output(out_loss_);
    if (cfg_.backward) ggml_set_loss(out_loss_);

    gf_ = ggml_new_graph_custom(ctx, cfg_.max_nodes, cfg_.backward);
    ggml_build_forward_expand(gf_, out_loss_);
    ggml_build_forward_expand(gf_, out_z_);
    ggml_build_forward_expand(gf_, out_loss_z_);
    ggml_build_forward_expand(gf_, out_loss_reg_);

    if (cfg_.backward) {
        const int n_nodes = ggml_graph_n_nodes(gf_);
        std::vector<ggml_tensor*> grad_accs(static_cast<size_t>(n_nodes), nullptr);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = ggml_graph_node(gf_, i);
            if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                auto it = param_index_.find(ggml_get_name(node));
                if (it == param_index_.end()) throw std::runtime_error(std::string("param without accumulator: ") + ggml_get_name(node));
                grad_accs[static_cast<size_t>(i)] = params_[it->second].grad_acc;
            }
        }
        ggml_build_backward_expand(ctx, gf_, grad_accs.data());
    }

    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!ggml_gallocr_reserve(galloc_, gf_) || !ggml_gallocr_alloc_graph(galloc_, gf_)) {
        throw std::runtime_error("failed to allocate the training graph");
    }
}

void TrainGraph::build_optimizer() {
    ggml_init_params ip{};
    const size_t n = params_.size();
    const size_t graph_size = 8 * n + 16;  // each step node references five tensors, all graph leaves
    ip.mem_size = ggml_tensor_overhead() * (2 * n + 8) + ggml_graph_overhead_custom(graph_size, false);
    ip.no_alloc = true;
    ctx_opt_ = ggml_init(ip);
    gopt_ = ggml_new_graph_custom(ctx_opt_, graph_size, false);
    for (Param& p : params_) {
        ggml_tensor* step = ggml_opt_step_adamw(ctx_opt_, p.tensor, p.grad_acc, p.m, p.v, p.adamw_params);
        ggml_build_forward_expand(gopt_, step);
    }
    galloc_opt_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!ggml_gallocr_reserve(galloc_opt_, gopt_) || !ggml_gallocr_alloc_graph(galloc_opt_, gopt_)) {
        throw std::runtime_error("failed to allocate the optimizer graph");
    }
}

Losses TrainGraph::step(const StepInputs& in) {
    upload(in_visual_, in.visual, "in.visual");
    upload(in_actions_, in.actions, "in.actions");
    upload(in_proj_, in.proj, "rdm.proj");
    if (cfg_.target_from_base) upload(in_base_, in.base, "rdm.base");
    else upload(in_target_sorted_, in.target_sorted, "rdm.target_sorted");
    upload(in_mask_, mask_host_, "predictor.mask");
    upload(in_token_zeros_, token_zeros_host_, "encoder.token_zeros");
    if (in.dropout_masks.size() != in_dropout_.size()) {
        throw std::runtime_error("step: expected " + std::to_string(in_dropout_.size()) + " dropout masks, got " +
                                 std::to_string(in.dropout_masks.size()));
    }
    for (size_t i = 0; i < in_dropout_.size(); ++i) upload(in_dropout_[i], in.dropout_masks[i], mask_specs_[i].site.c_str());

    if (cfg_.backward) ggml_graph_reset(gf_);  // zero gradient accumulators, seed d(loss)/d(loss) = 1
    if (ggml_backend_graph_compute(backend_, gf_) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("training graph compute failed");
    }
    Losses l;
    ggml_backend_tensor_get(out_loss_z_, &l.z, 0, sizeof(float));
    ggml_backend_tensor_get(out_loss_reg_, &l.reg, 0, sizeof(float));
    ggml_backend_tensor_get(out_loss_, &l.total, 0, sizeof(float));
    return l;
}

void TrainGraph::optimizer_step() {
    if (!cfg_.backward) throw std::runtime_error("optimizer_step on a forward-only graph");
    iter_ += 1;
    const float beta1h = 1.0f / (1.0f - std::pow(cfg_.beta1, static_cast<float>(iter_)));
    const float beta2h = 1.0f / (1.0f - std::pow(cfg_.beta2, static_cast<float>(iter_)));
    for (Param& p : params_) {
        const float pars[7] = {cfg_.lr * p.lr_mult, cfg_.beta1, cfg_.beta2, cfg_.eps,
                               p.weight_decay ? cfg_.weight_decay : 0.0f, beta1h, beta2h};
        ggml_backend_tensor_set(p.adamw_params, pars, 0, sizeof(pars));
    }
    if (ggml_backend_graph_compute(backend_, gopt_) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("optimizer graph compute failed");
    }
}

const TrainGraph::Param& TrainGraph::find(const std::string& name) const {
    auto it = param_index_.find(name);
    if (it == param_index_.end()) throw std::runtime_error("unknown parameter: " + name);
    return params_[it->second];
}

HostTensor TrainGraph::grad(const std::string& name) const { return read_tensor(find(name).grad_acc); }
HostTensor TrainGraph::param(const std::string& name) const { return read_tensor(weights_.tensor(name)); }
HostTensor TrainGraph::codes() const { return read_tensor(out_z_); }
void TrainGraph::set_grad(const std::string& name, const HostTensor& g) { upload(find(name).grad_acc, g, "gradient"); }
void TrainGraph::set_param(const std::string& name, const HostTensor& value) { upload(weights_.tensor(name), value, "parameter"); }

void TrainGraph::reset_optimizer() {
    for (Param& p : params_) {
        ggml_set_zero(p.m);
        ggml_set_zero(p.v);
    }
    iter_ = 0;
}

std::vector<std::string> TrainGraph::param_names() const {
    if (!params_.empty()) {
        std::vector<std::string> out;
        out.reserve(params_.size());
        for (const Param& p : params_) out.push_back(p.name);
        return out;
    }
    return weights_.tensor_names();
}

void TrainGraph::save_model(const std::string& path) const {
    std::vector<std::pair<std::string, ggml_tensor*>> tensors;
    for (const std::string& name : weights_.tensor_names()) tensors.emplace_back(name, weights_.tensor(name));
    write_gguf(path, tensors, weights_.gguf(), {});
}

void TrainGraph::save_optimizer(const std::string& path) const {
    if (!cfg_.backward) throw std::runtime_error("save_optimizer on a forward-only graph");
    std::vector<std::pair<std::string, ggml_tensor*>> tensors;
    for (const Param& p : params_) {
        tensors.emplace_back("m." + p.name, p.m);
        tensors.emplace_back("v." + p.name, p.v);
    }
    write_gguf(path, tensors, nullptr, {{"lewm.opt.iter", static_cast<uint32_t>(iter_)}});
}

void TrainGraph::load_optimizer(const std::string& path) {
    if (!cfg_.backward) throw std::runtime_error("load_optimizer on a forward-only graph");
    GgufStore side(path, backend_);
    for (Param& p : params_) {
        ggml_tensor* m = side.tensor("m." + p.name);
        ggml_tensor* v = side.tensor("v." + p.name);
        if (ggml_nelements(m) != ggml_nelements(p.m)) throw std::runtime_error("optimizer sidecar shape mismatch: " + p.name);
        ggml_backend_tensor_copy(m, p.m);
        ggml_backend_tensor_copy(v, p.v);
    }
    iter_ = side.get_u32("lewm.opt.iter");
}

}  // namespace lewm
