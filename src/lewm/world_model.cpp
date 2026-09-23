#include "lewm/world_model.hpp"

#include <algorithm>
#include <stdexcept>

#include "lewm/graph_builder.hpp"

namespace lewm {

WorldModel::WorldModel(const std::string& model_path, int n_threads, const std::string& device)
    : session_(std::make_unique<Session>(n_threads, device)),
      weights_(std::make_unique<GgufStore>(model_path, session_->backend())) {
    hp_ = Hparams::from_store(*weights_);
}

Outputs WorldModel::encode(const HostTensor& images, bool taps) {
    if (images.ne[0] != hp_.image_size || images.ne[1] != hp_.image_size || images.ne[2] != 3) {
        throw std::runtime_error("encode: images must be {W, H, 3, N} at image_size, got " + images.shape_str());
    }
    GraphBuilder gb(*weights_, hp_, false);
    const HostTensor zeros = gb.token_zeros_host(images.ne[3]);
    return session_->run([&](ggml_context* ctx, ggml_cgraph*, Binder& b) {
        ggml_tensor* img = b.input(ctx, "in.visual", images);
        ggml_tensor* tz = b.input(ctx, "encoder.token_zeros", zeros);
        TapFn tap;
        if (taps) tap = [&](const std::string& name, ggml_tensor* t) { b.output(name, t); };
        ggml_tensor* u = gb.encoder(ctx, img, tz, tap);
        b.output("encoder.u", u);
        b.output("encoder.z", gb.link(ctx, u));
    });
}

Outputs WorldModel::encode_actions(const HostTensor& actions) {
    if (actions.ne[0] != hp_.action_dim) {
        throw std::runtime_error("encode_actions: expected {A, T}, got " + actions.shape_str());
    }
    GraphBuilder gb(*weights_, hp_, false);
    return session_->run([&](ggml_context* ctx, ggml_cgraph*, Binder& b) {
        ggml_tensor* a = b.input(ctx, "in.actions", actions);  // {A, T}
        b.output("action_encoder.out", gb.action_codes(ctx, a));  // {C, T}
    });
}

Outputs WorldModel::predict(const HostTensor& codes, const HostTensor& action_codes, bool taps) {
    const int64_t C = hp_.code_dim;
    const int64_t P = hp_.predictor_num_patches;
    const int64_t T = codes.ne[2];
    if (codes.ne[0] != C || codes.ne[1] != P || T < 1 || T > hp_.predictor_num_frames) {
        throw std::runtime_error("predict: codes must be {C, P, T<=num_frames}, got " + codes.shape_str());
    }
    if (action_codes.ne[0] != C || action_codes.ne[1] != T) {
        throw std::runtime_error("predict: action_codes must be {C, T}, got " + action_codes.shape_str());
    }
    GraphBuilder gb(*weights_, hp_, false);
    const HostTensor mask_h = GraphBuilder::causal_mask(T, P);

    return session_->run([&](ggml_context* ctx, ggml_cgraph*, Binder& b) {
        ggml_tensor* x = b.input(ctx, "predictor.codes", codes);                // {C, P, T}
        ggml_tensor* a = b.input(ctx, "predictor.action_codes", action_codes);  // {C, T}
        ggml_tensor* mask = b.input(ctx, "predictor.mask", mask_h);             // {L, L}
        x = ggml_reshape_4d(ctx, x, C, P, T, 1);
        a = ggml_reshape_3d(ctx, a, C, T, 1);
        TapFn tap;
        if (taps) tap = [&](const std::string& name, ggml_tensor* t) { b.output(name, t); };
        ggml_tensor* u = gb.predictor(ctx, x, a, mask, tap);  // {C, P, T, 1}
        u = ggml_reshape_3d(ctx, u, C, P, T);
        b.output("predictor.u_pred", u);
        b.output("predictor.z_pred", gb.link(ctx, u));
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
