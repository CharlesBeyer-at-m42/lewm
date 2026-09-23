// lewm-rollout: open-loop prediction from the first observation of an input file.
//
//   lewm-rollout <model.gguf> <inputs.gguf> [--out rollout.gguf] [--threads N]
//
// <inputs.gguf> needs `in.visual` {W, H, 3, T} (normalised) and `in.actions` {A, T_act}; a reference
// dump qualifies. Prints per-frame code statistics and, with --out, writes `rollout.z` {C, P, T_act+1}.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "gguf.h"
#include "lewm/world_model.hpp"

namespace {

void write_rollout(const std::string& path, const lewm::HostTensor& z) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + z.data.size() * sizeof(float) + 1024;
    ip.no_alloc = false;
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t = ggml_new_tensor(ctx, GGML_TYPE_F32, z.n_dims, z.ne.data());
    ggml_set_name(t, "rollout.z");
    std::copy(z.data.begin(), z.data.end(), static_cast<float*>(t->data));

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "lewm");
    gguf_add_tensor(g, t);
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    if (!ok) throw std::runtime_error("failed to write " + path);
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, inputs_path, out_path;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (model_path.empty()) {
            model_path = a;
        } else if (inputs_path.empty()) {
            inputs_path = a;
        } else {
            std::fprintf(stderr, "unexpected argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (model_path.empty() || inputs_path.empty()) {
        std::fprintf(stderr, "usage: lewm-rollout <model.gguf> <inputs.gguf> [--out rollout.gguf] [--threads N]\n");
        return 2;
    }
    try {
        lewm::WorldModel wm(model_path, threads);
        lewm::GgufStore in(inputs_path, wm.backend());
        lewm::HostTensor visual = in.read("in.visual");
        const lewm::HostTensor actions = in.read("in.actions");

        // first observation only
        const int64_t per_img = visual.ne[0] * visual.ne[1] * visual.ne[2];
        visual.data.resize(static_cast<size_t>(per_img));
        visual.ne[3] = 1;
        visual.n_dims = 4;

        lewm::Outputs enc = wm.encode(visual);
        lewm::Outputs act = wm.encode_actions(actions);
        lewm::HostTensor z = wm.rollout(enc.at("encoder.z"), act.at("action_encoder.out"));

        const int64_t C = z.ne[0], P = z.ne[1], T = z.ne[2];
        std::printf("rollout: %lld frames of %lld x %lld codes (frame 0 encoded, %lld predicted)\n",
                    static_cast<long long>(T), static_cast<long long>(P), static_cast<long long>(C),
                    static_cast<long long>(T - 1));
        std::printf("%6s %10s %10s %10s\n", "frame", "nonzero", "mean", "max");
        for (int64_t t = 0; t < T; ++t) {
            const float* f = z.data.data() + t * C * P;
            int64_t nz = 0;
            double sum = 0, mx = 0;
            for (int64_t i = 0; i < C * P; ++i) {
                nz += f[i] != 0.0f;
                sum += f[i];
                mx = std::max(mx, static_cast<double>(std::fabs(f[i])));
            }
            std::printf("%6lld %9.1f%% %10.4f %10.4f\n", static_cast<long long>(t),
                        100.0 * static_cast<double>(nz) / static_cast<double>(C * P), sum / static_cast<double>(C * P), mx);
        }
        if (!out_path.empty()) {
            write_rollout(out_path, z);
            std::printf("wrote %s\n", out_path.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
