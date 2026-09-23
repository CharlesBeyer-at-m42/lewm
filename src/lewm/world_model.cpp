#include "lewm/world_model.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace lewm {

namespace {

// The reference stacks two LayerNorms per sublayer: a non-affine one in the block (eps 1e-6)
// and an affine one inside Attention / FeedForward (PyTorch default eps 1e-5).
constexpr float kBlockNormEps = 1e-6f;
constexpr float kAffineNormEps = 1e-5f;

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias) {
    // weight from numpy (out, in) has ne = {in, out}; mul_mat(weight, x) -> {out, ...}
    ggml_tensor* y = ggml_mul_mat(ctx, weight, x);
    if (bias) y = ggml_add(ctx, y, bias);
    return y;
}

}  // namespace

WorldModel::WorldModel(const std::string& model_path, int n_threads)
    : session_(std::make_unique<Session>(n_threads)),
      weights_(std::make_unique<GgufStore>(model_path, session_->backend())) {
    hp_ = Hparams::from_store(*weights_);
}

ggml_tensor* WorldModel::layer_norm(ggml_context* ctx, ggml_tensor* x, const std::string& prefix, float eps) const {
    ggml_tensor* y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w(prefix + ".weight"));
    y = ggml_add(ctx, y, w(prefix + ".bias"));
    return y;
}

ggml_tensor* WorldModel::attention(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mask, const std::string& prefix,
                                   uint32_t heads, uint32_t dim_head) const {
    // x: {D, L, N}
    const int64_t L = x->ne[1];
    const int64_t N = x->ne[2];
    const int64_t inner = static_cast<int64_t>(heads) * dim_head;

    ggml_tensor* h = layer_norm(ctx, x, prefix + ".norm", kAffineNormEps);
    ggml_tensor* qkv = linear(ctx, h, w(prefix + ".to_qkv.weight"), nullptr);  // {3*inner, L, N}

    auto split = [&](int64_t idx) {
        ggml_tensor* v = ggml_view_3d(ctx, qkv, inner, L, N, qkv->nb[1], qkv->nb[2], idx * inner * ggml_element_size(qkv));
        v = ggml_cont(ctx, v);
        v = ggml_reshape_4d(ctx, v, dim_head, heads, L, N);  // {d, h, L, N}
        return ggml_permute(ctx, v, 0, 2, 1, 3);             // {d, L, h, N}
    };
    ggml_tensor* q = split(0);
    ggml_tensor* k = split(1);
    ggml_tensor* v = split(2);

    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);  // {L_k, L_q, h, N}
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim_head));
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);

    ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // {L, d, h, N}
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq);                        // {d, L_q, h, N}
    kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));            // {d, h, L_q, N}
    kqv = ggml_reshape_3d(ctx, kqv, inner, L, N);                         // {inner, L, N}

    return linear(ctx, kqv, w(prefix + ".to_out.0.weight"), w(prefix + ".to_out.0.bias"));
}

ggml_tensor* WorldModel::feed_forward(ggml_context* ctx, ggml_tensor* x, const std::string& prefix) const {
    ggml_tensor* h = layer_norm(ctx, x, prefix + ".net.0", kAffineNormEps);
    h = linear(ctx, h, w(prefix + ".net.1.weight"), w(prefix + ".net.1.bias"));
    h = ggml_gelu_erf(ctx, h);
    h = linear(ctx, h, w(prefix + ".net.4.weight"), w(prefix + ".net.4.bias"));
    return h;
}

ggml_tensor* WorldModel::mlp_head(ggml_context* ctx, ggml_tensor* x, const std::string& prefix) const {
    // MLP: Linear -> LayerNorm(hidden) -> GELU -> Linear
    ggml_tensor* h = linear(ctx, x, w(prefix + ".net.0.weight"), w(prefix + ".net.0.bias"));
    h = layer_norm(ctx, h, prefix + ".net.1", kAffineNormEps);
    h = ggml_gelu_erf(ctx, h);
    h = linear(ctx, h, w(prefix + ".net.3.weight"), w(prefix + ".net.3.bias"));
    return h;
}

ggml_tensor* WorldModel::link(ggml_context* ctx, ggml_tensor* u) const {
    if (hp_.link == "identity") return u;
    return ggml_relu(ctx, u);  // relu and reprelu share the forward value
}

HostTensor WorldModel::causal_mask(int64_t T, int64_t P) const {
    // Block-causal, frame-major: query token i attends key token j iff frame(j) <= frame(i).
    // With P == 1 this is the plain causal mask. ne = {L_k, L_q}.
    const int64_t L = T * P;
    HostTensor m = HostTensor::zeros({L, L});
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = 0; j < L; ++j) {
            if (j / P > i / P) m.data[static_cast<size_t>(i * L + j)] = neg_inf;
        }
    }
    return m;
}

Outputs WorldModel::encode(const HostTensor& images, bool taps) {
    const Hparams& hp = hp_;
    if (images.ne[0] != hp.image_size || images.ne[1] != hp.image_size || images.ne[2] != 3) {
        throw std::runtime_error("encode: images must be {W, H, 3, N} at image_size, got " + images.shape_str());
    }
    const int64_t N = images.ne[3];
    const int64_t G = hp.grid();
    const int64_t D = hp.dim;
    const int64_t n_tok = G * G + 1;

    return session_->run([&](ggml_context* ctx, ggml_cgraph*, Binder& b) {
        ggml_tensor* img = b.input(ctx, "in.visual", images);

        // Patchify as the reference Conv2d(3, D, k=p, s=p), in fp32 (ggml_conv_2d would use f16 im2col).
        ggml_tensor* kernel = w("encoder.patch_embed.weight");  // {p, p, 3, D}
        const int p = static_cast<int>(hp.patch_size);
        ggml_tensor* cols = ggml_im2col(ctx, kernel, img, p, p, 0, 0, 1, 1, true, GGML_TYPE_F32);  // {3*p*p, G, G, N}
        ggml_tensor* cols2d = ggml_reshape_2d(ctx, cols, cols->ne[0], cols->ne[1] * cols->ne[2] * cols->ne[3]);
        ggml_tensor* k2d = ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]);
        ggml_tensor* patches = ggml_mul_mat(ctx, k2d, cols2d);  // {D, G*G*N}: token index = w + G*h (matches "(h w)")
        patches = ggml_reshape_3d(ctx, patches, D, G * G, N);
        patches = ggml_add(ctx, patches, w("encoder.patch_embed.bias"));

        ggml_tensor* cls = ggml_reshape_3d(ctx, w("encoder.cls_token"), D, 1, 1);
        cls = ggml_repeat_4d(ctx, cls, D, 1, N, 1);
        ggml_tensor* x = ggml_concat(ctx, cls, patches, 1);  // {D, 1+G*G, N}
        x = ggml_add(ctx, x, w("encoder.pos_embedding"));   // {D, 1+G*G, 1} broadcast over N
        if (taps) b.output("encoder.tokens_in", x);

        for (uint32_t i = 0; i < hp.encoder_depth; ++i) {
            const std::string pre = "encoder.transformer.layers." + std::to_string(i);
            ggml_tensor* h = ggml_norm(ctx, x, kBlockNormEps);
            x = ggml_add(ctx, x, attention(ctx, h, nullptr, pre + ".attn", hp.encoder_heads, hp.encoder_dim_head));
            h = ggml_norm(ctx, x, kBlockNormEps);
            x = ggml_add(ctx, x, feed_forward(ctx, h, pre + ".mlp"));
            if (taps) b.output("encoder.block." + std::to_string(i) + ".out", x);
        }
        x = layer_norm(ctx, x, "encoder.transformer.norm", kAffineNormEps);
        if (taps) b.output("encoder.norm.out", x);

        ggml_tensor* tokens;
        if (hp.cls_mode()) {
            tokens = ggml_view_3d(ctx, x, D, 1, N, x->nb[1], x->nb[2], 0);
        } else {
            tokens = ggml_view_3d(ctx, x, D, n_tok - 1, N, x->nb[1], x->nb[2], x->nb[1]);
        }
        tokens = ggml_cont(ctx, tokens);  // {D, P, N}
        ggml_tensor* u = mlp_head(ctx, tokens, "encoder.projector");  // {C, P, N}
        b.output("encoder.u", u);
        b.output("encoder.z", link(ctx, u));
    });
}

Outputs WorldModel::encode_actions(const HostTensor& actions) {
    if (actions.ne[0] != hp_.action_dim) {
        throw std::runtime_error("encode_actions: expected {A, T}, got " + actions.shape_str());
    }
    return session_->run([&](ggml_context* ctx, ggml_cgraph*, Binder& b) {
        ggml_tensor* a = b.input(ctx, "in.actions", actions);  // {A, T}
        // Conv1d(k=1) is a per-step linear map; weight from numpy (out, in, 1) has ne = {1, in, out}.
        ggml_tensor* cw = w("action_encoder.patch_embed.weight");
        cw = ggml_reshape_2d(ctx, cw, cw->ne[1], cw->ne[2]);  // {in, out}
        ggml_tensor* h = linear(ctx, a, cw, w("action_encoder.patch_embed.bias"));
        h = linear(ctx, h, w("action_encoder.embed.0.weight"), w("action_encoder.embed.0.bias"));
        h = ggml_silu(ctx, h);
        h = linear(ctx, h, w("action_encoder.embed.2.weight"), w("action_encoder.embed.2.bias"));
        b.output("action_encoder.out", h);  // {C, T}
    });
}

Outputs WorldModel::predict(const HostTensor& codes, const HostTensor& action_codes, bool taps) {
    const Hparams& hp = hp_;
    const int64_t C = hp.code_dim;
    const int64_t P = hp.predictor_num_patches;
    const int64_t T = codes.ne[2];
    if (codes.ne[0] != C || codes.ne[1] != P || T < 1 || T > hp.predictor_num_frames) {
        throw std::runtime_error("predict: codes must be {C, P, T<=num_frames}, got " + codes.shape_str());
    }
    if (action_codes.ne[0] != C || action_codes.ne[1] != T) {
        throw std::runtime_error("predict: action_codes must be {C, T}, got " + action_codes.shape_str());
    }
    const int64_t L = T * P;
    const HostTensor mask_h = causal_mask(T, P);

    return session_->run([&](ggml_context* ctx, ggml_cgraph*, Binder& b) {
        ggml_tensor* x = b.input(ctx, "predictor.codes", codes);           // {C, P, T}
        ggml_tensor* a = b.input(ctx, "predictor.action_codes", action_codes);  // {C, T}
        ggml_tensor* mask = b.input(ctx, "predictor.mask", mask_h);        // {L, L}

        ggml_tensor* tpos = w("predictor.temporal_pos");  // {C, 1, num_frames, 1}
        tpos = ggml_view_3d(ctx, tpos, C, 1, T, tpos->nb[1], tpos->nb[2], 0);
        x = ggml_add(ctx, x, tpos);
        if (P > 1) x = ggml_add(ctx, x, w("predictor.spatial_pos"));  // {C, P, 1, 1}
        x = ggml_reshape_2d(ctx, ggml_cont(ctx, x), C, L);             // frame-major (t p)

        ggml_tensor* c = ggml_reshape_3d(ctx, a, C, 1, T);
        c = ggml_repeat_4d(ctx, c, C, P, T, 1);
        c = ggml_reshape_2d(ctx, c, C, L);  // {C, L}: action of token (t, p) is a[t]
        if (taps) b.output("predictor.tokens_in", x);

        for (uint32_t i = 0; i < hp.predictor_depth; ++i) {
            const std::string pre = "predictor.transformer.layers." + std::to_string(i);
            // adaLN_modulation: SiLU -> Linear(C, 6C) -> chunk into shift/scale/gate for attn and mlp
            ggml_tensor* mod = linear(ctx, ggml_silu(ctx, c), w(pre + ".adaLN_modulation.1.weight"),
                                      w(pre + ".adaLN_modulation.1.bias"));  // {6C, L}
            auto chunk = [&](int64_t k) {
                return ggml_cont(ctx, ggml_view_2d(ctx, mod, C, L, mod->nb[1], k * C * ggml_element_size(mod)));
            };
            ggml_tensor* shift_msa = chunk(0);
            ggml_tensor* scale_msa = chunk(1);
            ggml_tensor* gate_msa = chunk(2);
            ggml_tensor* shift_mlp = chunk(3);
            ggml_tensor* scale_mlp = chunk(4);
            ggml_tensor* gate_mlp = chunk(5);

            auto modulate = [&](ggml_tensor* h, ggml_tensor* shift, ggml_tensor* scale) {
                // h * (1 + scale) + shift
                return ggml_add(ctx, ggml_add(ctx, h, ggml_mul(ctx, h, scale)), shift);
            };

            ggml_tensor* h = modulate(ggml_norm(ctx, x, kBlockNormEps), shift_msa, scale_msa);
            h = attention(ctx, h, mask, pre + ".attn", hp.predictor_heads, hp.predictor_dim_head);
            x = ggml_add(ctx, x, ggml_mul(ctx, gate_msa, h));

            h = modulate(ggml_norm(ctx, x, kBlockNormEps), shift_mlp, scale_mlp);
            h = feed_forward(ctx, h, pre + ".mlp");
            x = ggml_add(ctx, x, ggml_mul(ctx, gate_mlp, h));
            if (taps) b.output("predictor.block." + std::to_string(i) + ".out", x);
        }
        x = layer_norm(ctx, x, "predictor.transformer.norm", kAffineNormEps);
        if (taps) b.output("predictor.norm.out", x);

        ggml_tensor* u = mlp_head(ctx, x, "predictor.output_proj");  // {C, L}
        u = ggml_reshape_3d(ctx, u, C, P, T);
        b.output("predictor.u_pred", u);
        b.output("predictor.z_pred", link(ctx, u));
    });
}

HostTensor WorldModel::rollout(const HostTensor& z0, const HostTensor& action_codes) {
    const int64_t C = hp_.code_dim;
    const int64_t P = hp_.predictor_num_patches;
    const int64_t T_act = action_codes.ne[1];
    const int64_t H = hp_.predictor_num_frames;
    if (z0.ne[0] != C || z0.ne[1] != P) throw std::runtime_error("rollout: z0 must be {C, P, n}, got " + z0.shape_str());
    if (action_codes.ne[0] != C) throw std::runtime_error("rollout: action_codes must be {C, T_act}");

    const size_t frame = static_cast<size_t>(C * P);
    HostTensor emb = z0;  // grows one frame per step
    emb.n_dims = 3;

    auto predict_next = [&]() {
        const int64_t Lc = emb.ne[2];
        const int64_t h = std::min(H, Lc);
        if (Lc > T_act) throw std::runtime_error("rollout: more frames than actions");
        HostTensor win = HostTensor::zeros({C, P, h});
        std::copy(emb.data.begin() + static_cast<std::ptrdiff_t>((Lc - h) * frame), emb.data.end(), win.data.begin());
        HostTensor awin = HostTensor::zeros({C, h});
        std::copy(action_codes.data.begin() + static_cast<std::ptrdiff_t>((Lc - h) * C),
                  action_codes.data.begin() + static_cast<std::ptrdiff_t>(Lc * C), awin.data.begin());
        Outputs o = predict(win, awin, false);
        const HostTensor& zp = o.at("predictor.z_pred");  // {C, P, h}
        emb.data.insert(emb.data.end(), zp.data.end() - static_cast<std::ptrdiff_t>(frame), zp.data.end());
        emb.ne[2] += 1;
    };

    while (emb.ne[2] < T_act) predict_next();
    predict_next();
    return emb;  // {C, P, T_act + 1}
}

}  // namespace lewm
