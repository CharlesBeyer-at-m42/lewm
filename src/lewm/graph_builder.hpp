#pragma once

#include <functional>
#include <string>

#include "ggml.h"
#include "lewm/gguf_store.hpp"
#include "lewm/hparams.hpp"
#include "lewm/host_tensor.hpp"

namespace lewm {

/// Called with (tap name, tensor) for intermediate outputs a caller wants to keep.
using TapFn = std::function<void(const std::string&, ggml_tensor*)>;

/// Returns an inverted-dropout mask tensor (values 0 or 1/(1-p)) for a named site, or nullptr for
/// no dropout at that site. Shapes are ggml order.
using MaskFn = std::function<ggml_tensor*(const std::string& site, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)>;

/// Builds the LpWM graphs on ggml from a weight store. Shared by inference (WorldModel) and
/// training (TrainGraph). Shapes are in ggml order: C = code width, P = tokens per frame.
///
/// With `training` set, the link uses the straight-through reprelu op and every op chosen has a
/// backward pass in ggml (the cls token is placed with ggml_acc rather than ggml_concat).
class GraphBuilder {
public:
    GraphBuilder(const GgufStore& weights, const Hparams& hp, bool training = false)
        : w_(weights), hp_(hp), training_(training) {}

    const Hparams& hparams() const { return hp_; }

    /// Dropout sites are multiplied by the mask this returns (predictor only; the encoder trains
    /// without dropout in the reference configuration). Unset means no dropout.
    void set_dropout_masks(MaskFn fn) { mask_ = std::move(fn); }

    /// img {W, H, 3, N} normalised -> u {C, P, N} (projector output, pre-link).
    /// `token_zeros` is a zero tensor {dim, 1+G*G, N} the cls/patch tokens are accumulated into.
    ggml_tensor* encoder(ggml_context* ctx, ggml_tensor* img, ggml_tensor* token_zeros, const TapFn& tap) const;

    /// actions {A, T, N} -> action codes {C, T, N}.
    ggml_tensor* action_codes(ggml_context* ctx, ggml_tensor* actions) const;

    /// codes {C, P, T, N} (linked) + action codes {C, T, N} + mask {T*P, T*P} -> u_pred {C, P, T, N}.
    ggml_tensor* predictor(ggml_context* ctx, ggml_tensor* codes, ggml_tensor* action_codes, ggml_tensor* mask,
                           const TapFn& tap) const;

    /// The link h(.): identity, relu, or reprelu (relu forward; gelu gradient when training).
    ggml_tensor* link(ggml_context* ctx, ggml_tensor* u) const;

    /// Block-causal mask {L_k, L_q} for T frames of P tokens, 0 = attend, -inf = masked.
    static HostTensor causal_mask(int64_t T, int64_t P);

    /// Zero tensor host data for `encoder`'s token_zeros argument.
    HostTensor token_zeros_host(int64_t N) const;

private:
    const GgufStore& w_;
    const Hparams& hp_;
    bool training_;

    ggml_tensor* w(const std::string& name) const { return w_.tensor(name); }
    ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, const std::string& prefix, float eps) const;
    ggml_tensor* attention(ggml_context* ctx, ggml_tensor* x, ggml_tensor* mask, const std::string& prefix,
                           uint32_t heads, uint32_t dim_head, bool dropout = false) const;
    ggml_tensor* feed_forward(ggml_context* ctx, ggml_tensor* x, const std::string& prefix, bool dropout = false) const;
    ggml_tensor* apply_mask(ggml_context* ctx, ggml_tensor* x, const std::string& site) const;
    MaskFn mask_;
    ggml_tensor* mlp_head(ggml_context* ctx, ggml_tensor* x, const std::string& prefix) const;
};

}  // namespace lewm
