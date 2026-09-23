// lewm-gradcheck: gradient parity against a dump from tools/dump_gradients.py.
//
//   lewm-gradcheck <model.gguf> <grad.gguf> [--atol 1e-6] [--rtol 1e-3] [--threads N] [--device cpu|gpu] [--show 12]
//
// Runs one forward/backward on the dump's batch with the dump's RDMReg randomness, compares the
// losses and every parameter gradient, then applies one AdamW step with the dump's optimizer
// settings and compares the updated parameters. A tensor passes when every element satisfies
// |lewm - ref| <= atol + rtol * |ref|. Exit 0 on full pass, 1 on any failure, 2 on errors.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "lewm/gguf_store.hpp"
#include "lewm/session.hpp"
#include "lewm/train_graph.hpp"

namespace {

struct Row {
    std::string name;
    double worst_ratio = 0;  // max over elements of |d| / (atol + rtol*|ref|)
    double max_abs_diff = 0;
    double ref_max_abs = 0;
    bool pass = false;
};

Row compare(const std::string& name, const lewm::HostTensor& ref, const lewm::HostTensor& got, double atol, double rtol) {
    Row r;
    r.name = name;
    if (ref.data.size() != got.data.size()) {
        r.worst_ratio = INFINITY;
        return r;
    }
    for (size_t i = 0; i < ref.data.size(); ++i) {
        const double a = ref.data[i], b = got.data[i];
        const double d = std::fabs(a - b);
        if (std::isnan(d)) {
            r.worst_ratio = NAN;
            return r;
        }
        r.max_abs_diff = std::max(r.max_abs_diff, d);
        r.ref_max_abs = std::max(r.ref_max_abs, std::fabs(a));
        r.worst_ratio = std::max(r.worst_ratio, d / (atol + rtol * std::fabs(a)));
    }
    r.pass = r.worst_ratio <= 1.0;
    return r;
}

void print_rows(const char* title, std::vector<Row> rows, size_t show) {
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.worst_ratio > b.worst_ratio; });
    size_t n_fail = 0;
    for (const Row& r : rows) n_fail += !r.pass;
    std::printf("\n%s: %zu tensors, %zu failing. Worst %zu:\n", title, rows.size(), n_fail, std::min(show, rows.size()));
    std::printf("  %-52s %10s %12s %10s  %s\n", "tensor", "ratio", "max|diff|", "ref max", "result");
    for (size_t i = 0; i < std::min(show, rows.size()); ++i) {
        const Row& r = rows[i];
        std::printf("  %-52s %10.3f %12.3e %10.3g  %s\n", r.name.c_str(), r.worst_ratio, r.max_abs_diff, r.ref_max_abs,
                    r.pass ? "ok" : "FAIL");
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, dump_path, device = "cpu";
    double atol = 1e-6, rtol = 1e-3;
    int threads = 0;
    size_t show = 12;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--atol" && i + 1 < argc) atol = std::atof(argv[++i]);
        else if (a == "--rtol" && i + 1 < argc) rtol = std::atof(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--device" && i + 1 < argc) device = argv[++i];
        else if (a == "--show" && i + 1 < argc) show = static_cast<size_t>(std::atoi(argv[++i]));
        else if (model_path.empty()) model_path = a;
        else if (dump_path.empty()) dump_path = a;
        else { std::fprintf(stderr, "unexpected argument: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || dump_path.empty()) {
        std::fprintf(stderr, "usage: lewm-gradcheck <model.gguf> <grad.gguf> [--atol 1e-6] [--rtol 1e-3] [--threads N]\n");
        return 2;
    }
    try {
        lewm::Session session(threads, device);
        std::printf("device: %s (%s)\n", device.c_str(), ggml_backend_name(session.backend()));
        lewm::GgufStore weights(model_path, session.backend());
        const lewm::Hparams hp = lewm::Hparams::from_store(weights);
        lewm::GgufStore dump(dump_path, session.backend());

        lewm::TrainConfig cfg;
        cfg.batch = dump.get_u32("lewm.gradcheck.batch_size");
        cfg.lr = dump.get_f32("lewm.gradcheck.lr");
        cfg.weight_decay = dump.get_f32("lewm.gradcheck.wd");
        cfg.beta1 = dump.get_f32("lewm.gradcheck.beta1");
        cfg.beta2 = dump.get_f32("lewm.gradcheck.beta2");
        cfg.eps = dump.get_f32("lewm.gradcheck.eps");
        cfg.reg_weight = dump.get_f32("lewm.gradcheck.reg_weight");
        cfg.num_projections = dump.get_u32("lewm.gradcheck.num_projections");
        std::printf("model: %s\n  %s\ndump:  %s\n  batch %lld, lr %g, wd %g, reg_weight %g, K %lld\n", model_path.c_str(),
                    hp.describe().c_str(), dump_path.c_str(), static_cast<long long>(cfg.batch), cfg.lr, cfg.weight_decay,
                    cfg.reg_weight, static_cast<long long>(cfg.num_projections));

        lewm::TrainGraph tg(weights, hp, session.backend(), cfg);

        // Inputs. The dump stores frames as (B*T, 3, H, W) -> ggml {W, H, 3, T*B}, frame-major per sample.
        lewm::HostTensor visual = dump.read("in.visual");
        lewm::HostTensor actions = dump.read("in.actions");  // {A, T, B}
        lewm::HostTensor proj = dump.read("rdm.proj");       // {C, K}
        lewm::HostTensor tsorted = dump.read("rdm.target_sorted");  // {B, K, TP}

        lewm::Losses l = tg.forward_backward(visual, actions, proj, tsorted);
        const float ref_z = dump.read("loss.z").data[0];
        const float ref_reg = dump.read("loss.reg").data[0];
        const float ref_total = dump.read("loss.total").data[0];
        std::printf("\nlosses            %12s %12s %12s\n", "lewm", "reference", "|diff|");
        std::printf("  z_loss          %12.7f %12.7f %12.3e\n", l.z, ref_z, std::fabs(l.z - ref_z));
        std::printf("  reg_loss        %12.7f %12.7f %12.3e\n", l.reg, ref_reg, std::fabs(l.reg - ref_reg));
        std::printf("  total           %12.7f %12.7f %12.3e\n", l.total, ref_total, std::fabs(l.total - ref_total));

        std::vector<Row> fwd_rows, grad_rows, post_rows;
        // activations are O(1) where gradients are O(1e-2): give them a proportionally larger floor
        fwd_rows.push_back(compare("fwd.z", dump.read("fwd.z"), tg.codes(), 1e-5, rtol));
        for (const std::string& name : tg.param_names()) {
            grad_rows.push_back(compare(name, dump.read("grad." + name), tg.grad(name), atol, rtol));
        }
        print_rows("forward codes", fwd_rows, show);
        print_rows("gradients", grad_rows, show);

        // Optimizer check 1 (informational): step from LeWM's own gradients. Adam normalises each
        // element by its gradient magnitude, so 1e-8 gradient noise on near-zero entries becomes
        // update noise; compared at a looser absolute tolerance.
        std::vector<lewm::HostTensor> original;
        for (const std::string& name : tg.param_names()) original.push_back(tg.param(name));
        tg.optimizer_step();
        {
            std::vector<Row> e2e;
            for (const std::string& name : tg.param_names()) {
                e2e.push_back(compare(name, dump.read("post." + name), tg.param(name), 1e-5, rtol));
            }
            print_rows("end-to-end step from LeWM gradients (informational, atol 1e-5)", e2e, 5);
        }

        // Optimizer check 2 (gate): restore the weights, step from the reference gradients, compare strictly.
        {
            size_t i = 0;
            for (const std::string& name : tg.param_names()) tg.set_param(name, original[i++]);
            tg.reset_optimizer();
            for (const std::string& name : tg.param_names()) tg.set_grad(name, dump.read("grad." + name));
        }
        tg.optimizer_step();
        for (const std::string& name : tg.param_names()) {
            post_rows.push_back(compare(name, dump.read("post." + name), tg.param(name), atol, rtol));
        }
        print_rows("AdamW step from reference gradients", post_rows, show);

        auto all = [](const std::vector<Row>& rows) { return std::all_of(rows.begin(), rows.end(), [](const Row& r) { return r.pass; }); };
        const bool loss_ok = std::fabs(l.total - ref_total) <= atol + rtol * std::fabs(ref_total);
        const bool ok = loss_ok && all(fwd_rows) && all(grad_rows) && all(post_rows);
        std::printf("\n%s (atol %g, rtol %g)\n", ok ? "GRADIENT PARITY OK" : "GRADIENT PARITY FAILED", atol, rtol);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
