#pragma once

#include <memory>
#include <string>

#include "lewm/gguf_store.hpp"
#include "lewm/hparams.hpp"
#include "lewm/host_tensor.hpp"
#include "lewm/session.hpp"

namespace lewm {

/// The LpWM world model on ggml, inference side: encoder, action encoder, predictor, link.
/// Tensor shapes below use ggml order (innermost first). C = code_dim, P = tokens per frame.
///
/// Output names match tools/dump_reference.py so the same map can be compared to a dump.
class WorldModel {
public:
    WorldModel(const std::string& model_path, int n_threads = 0, const std::string& device = "cpu");

    const Hparams& hparams() const { return hp_; }
    const GgufStore& weights() const { return *weights_; }
    GgufStore& weights() { return *weights_; }
    ggml_backend_t backend() const { return session_->backend(); }

    /// images: {W, H, 3, N} normalised as the reference transform does.
    /// Returns encoder.u {C, P, N}, encoder.z {C, P, N} and, with taps, every block output.
    Outputs encode(const HostTensor& images, bool taps = false);

    /// actions: {A, T}. Returns action_encoder.out {C, T}.
    Outputs encode_actions(const HostTensor& actions);

    /// codes: {C, P, T} linked codes of a history window (T <= num_frames);
    /// action_codes: {C, T} aligned with codes. Returns predictor.u_pred and predictor.z_pred {C, P, T}
    /// and, with taps, every block output {C, P*T}.
    Outputs predict(const HostTensor& codes, const HostTensor& action_codes, bool taps = false);

    /// Open-loop rollout mirroring VWorldModel._rollout_adaln: z0 {C, P, n} linked codes of the
    /// starting frames, action_codes {C, T_act}. Returns {C, P, T_act + 1}.
    HostTensor rollout(const HostTensor& z0, const HostTensor& action_codes);

private:
    Hparams hp_;
    std::unique_ptr<Session> session_;
    std::unique_ptr<GgufStore> weights_;
};

}  // namespace lewm
