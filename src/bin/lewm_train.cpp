// lewm-train: train the LpWM world model on ggml.
//
//   lewm-train --model init.gguf --data <exported dataset dir> --out <run dir> [options]
//
// Defaults reproduce the reference Wall run (lewm_ref_wall): batch 16, muP base lr 1e-4, weight
// decay 0.01, RDMReg weight 0.5 with 8192 projections and a Laplace (p=1) target, predictor
// dropout 0.1, 5 epochs. Batches follow the dataset's batch plan (the reference's fixed order).
//
// Options:
//   --epochs N            (5)      --batch B            (16)     --lr X          (1e-4)
//   --wd X                (0.01)   --reg-weight X       (0.5)    --dropout X     (0.1)
//   --projections K       (8192)   --seed S             (0)      --device cpu|gpu (cpu)
//   --threads N           (auto)   --max-steps N        (all)    --val-batches N (all)
//   --no-val                       --save-every E       (1)      --resume <run dir/epoch_N>
//   --log-every N         (10)
//
// Writes <out>/train.jsonl (one record per step and per epoch), <out>/epoch_N/model.gguf and
// optimizer.gguf, and <out>/model_latest.gguf.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "lewm/dataset.hpp"
#include "lewm/gguf_store.hpp"
#include "lewm/rng.hpp"
#include "lewm/session.hpp"
#include "lewm/train_graph.hpp"

namespace {

// RNG stream ids shared with tools/lewm_rng.py
constexpr uint64_t kStreamProj = 1, kStreamTarget = 2, kStreamDropout = 3, kStreamVal = 4;
constexpr double kLaplaceScale = 0.70710678118654752440;  // unit-variance Laplace: 1/sqrt(2)

struct Args {
    std::string model, data, out, device = "cpu", resume;
    int epochs = 5;
    int64_t batch = 16;
    float lr = 1e-4f, wd = 0.01f, reg_weight = 0.5f, dropout = 0.1f;
    int64_t projections = 8192;
    uint64_t seed = 0;
    int threads = 0;
    int64_t max_steps = -1, val_batches = -1;
    bool do_val = true;
    int save_every = 1, log_every = 10;
};

Args parse(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--model") a.model = need(i);
        else if (k == "--data") a.data = need(i);
        else if (k == "--out") a.out = need(i);
        else if (k == "--device") a.device = need(i);
        else if (k == "--resume") a.resume = need(i);
        else if (k == "--epochs") a.epochs = std::atoi(need(i));
        else if (k == "--batch") a.batch = std::atoll(need(i));
        else if (k == "--lr") a.lr = std::atof(need(i));
        else if (k == "--wd") a.wd = std::atof(need(i));
        else if (k == "--reg-weight") a.reg_weight = std::atof(need(i));
        else if (k == "--dropout") a.dropout = std::atof(need(i));
        else if (k == "--projections") a.projections = std::atoll(need(i));
        else if (k == "--seed") a.seed = std::strtoull(need(i), nullptr, 10);
        else if (k == "--threads") a.threads = std::atoi(need(i));
        else if (k == "--max-steps") a.max_steps = std::atoll(need(i));
        else if (k == "--val-batches") a.val_batches = std::atoll(need(i));
        else if (k == "--no-val") a.do_val = false;
        else if (k == "--save-every") a.save_every = std::atoi(need(i));
        else if (k == "--log-every") a.log_every = std::atoi(need(i));
        else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.model.empty() || a.data.empty() || a.out.empty()) throw std::runtime_error("--model, --data and --out are required");
    return a;
}

/// Randomness for one step, identical to tools/lewm_rng.py given (seed, step).
struct StepRandom {
    lewm::Rng proj, target, dropout;
    StepRandom(uint64_t seed, uint64_t stream_offset)
        : proj(seed, kStreamProj + stream_offset), target(seed, kStreamTarget + stream_offset), dropout(seed, kStreamDropout + stream_offset) {}
};

void fill_random(lewm::StepInputs& in, const lewm::TrainGraph& tg, lewm::Rng& proj_rng, lewm::Rng& target_rng,
                 lewm::Rng& dropout_rng, int64_t step, int64_t C, int64_t K, int64_t B, int64_t TP, float dropout) {
    // projections: K*C normals in (k, c) order, each row normalised
    proj_rng.seek(static_cast<uint64_t>(step) * static_cast<uint64_t>(K * C));
    in.proj.n_dims = 2;
    in.proj.ne = {C, K, 1, 1};
    proj_rng.normal(in.proj.data, static_cast<size_t>(K * C));
    for (int64_t k = 0; k < K; ++k) {
        float* row = in.proj.data.data() + k * C;
        double ss = 0;
        for (int64_t c = 0; c < C; ++c) ss += static_cast<double>(row[c]) * row[c];
        const float inv = static_cast<float>(1.0 / std::sqrt(ss));
        for (int64_t c = 0; c < C; ++c) row[c] *= inv;
    }
    // target base samples: Laplace(0, 1/sqrt 2), shape (B, TP, C) numpy == {C, TP, B} ggml
    const size_t n_base = static_cast<size_t>(B * TP * C);
    target_rng.seek(static_cast<uint64_t>(step) * n_base);
    in.base.n_dims = 3;
    in.base.ne = {C, TP, B, 1};
    target_rng.laplace(in.base.data, n_base, kLaplaceScale);
    // dropout masks, in mask_specs order
    in.dropout_masks.clear();
    if (dropout > 0.0f) {
        size_t total = 0;
        for (const lewm::MaskSpec& s : tg.mask_specs()) total += static_cast<size_t>(s.numel());
        dropout_rng.seek(static_cast<uint64_t>(step) * total);
        for (const lewm::MaskSpec& s : tg.mask_specs()) {
            lewm::HostTensor m;
            m.n_dims = 4;
            m.ne = s.ne;
            dropout_rng.dropout_mask(m.data, static_cast<size_t>(s.numel()), dropout);
            in.dropout_masks.push_back(std::move(m));
        }
    }
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse(argc, argv);
        std::filesystem::create_directories(args.out);
        std::ofstream log(args.out + "/train.jsonl", std::ios::app);

        lewm::Session session(args.threads, args.device);
        lewm::GgufStore weights(args.model, session.backend());
        const lewm::Hparams hp = lewm::Hparams::from_store(weights);
        lewm::Dataset data(args.data);
        const auto& meta = data.meta();
        if (meta.num_frames != static_cast<int64_t>(hp.predictor_num_frames) + 1) {
            throw std::runtime_error("dataset num_frames does not match the model's history window + 1");
        }
        if (data.chunked_action_dim() != hp.action_dim) throw std::runtime_error("dataset chunked action width != model action_dim");

        lewm::TrainConfig cfg;
        cfg.batch = args.batch;
        cfg.lr = args.lr;
        cfg.weight_decay = args.wd;
        cfg.reg_weight = args.reg_weight;
        cfg.num_projections = args.projections;
        cfg.dropout = args.dropout;
        cfg.backward = true;
        cfg.target_from_base = true;
        lewm::TrainGraph train(weights, hp, session.backend(), cfg);

        lewm::TrainConfig vcfg = cfg;
        vcfg.dropout = 0.0f;
        vcfg.backward = false;
        std::unique_ptr<lewm::TrainGraph> val;
        if (args.do_val) val = std::make_unique<lewm::TrainGraph>(weights, hp, session.backend(), vcfg);

        int64_t global_step = 0;
        int first_epoch = 0;
        if (!args.resume.empty()) {
            train.load_optimizer(args.resume + "/optimizer.gguf");
            global_step = train.iteration();
            const std::string ep_file = args.resume + "/epoch.txt";
            std::ifstream ef(ep_file);
            if (ef) ef >> first_epoch;
            std::printf("resumed from %s: iteration %lld, starting epoch %d\n", args.resume.c_str(),
                        static_cast<long long>(global_step), first_epoch);
        }

        const int64_t B = args.batch;
        const int64_t T = meta.num_frames;
        const int64_t C = hp.code_dim, P = hp.predictor_num_patches, K = args.projections, TP = T * P;
        const int64_t train_batches = static_cast<int64_t>(data.plan_train().size()) / B;
        int64_t val_batches = static_cast<int64_t>(data.plan_val().size()) / B;
        if (args.val_batches > 0) val_batches = std::min(val_batches, args.val_batches);

        std::printf("model:   %s\n  %s\ndata:    %s (%lld train / %lld val batches of %lld)\ndevice:  %s\n"
                    "config:  lr %g wd %g reg_weight %g K %lld dropout %g epochs %d seed %llu\nout:     %s\n\n",
                    args.model.c_str(), hp.describe().c_str(), args.data.c_str(), static_cast<long long>(train_batches),
                    static_cast<long long>(val_batches), static_cast<long long>(B), ggml_backend_name(session.backend()),
                    args.lr, args.wd, args.reg_weight, static_cast<long long>(K), args.dropout, args.epochs,
                    static_cast<unsigned long long>(args.seed), args.out.c_str());

        StepRandom rnd(args.seed, 0);
        lewm::Rng val_proj(args.seed, kStreamVal), val_target(args.seed, kStreamVal + 1), val_dummy(args.seed, kStreamVal + 2);
        lewm::StepInputs in;
        const auto t_start = std::chrono::steady_clock::now();

        for (int epoch = first_epoch; epoch < args.epochs; ++epoch) {
            double sum_total = 0, sum_z = 0, sum_reg = 0;
            int64_t n = 0;
            const auto t_epoch = std::chrono::steady_clock::now();
            for (int64_t b = 0; b < train_batches; ++b) {
                if (args.max_steps >= 0 && global_step >= args.max_steps) break;
                data.make_batch(data.plan_train(), static_cast<size_t>(b * B), B, in.visual, in.actions);
                fill_random(in, train, rnd.proj, rnd.target, rnd.dropout, global_step, C, K, B, TP, args.dropout);
                const lewm::Losses l = train.step(in);
                train.optimizer_step();
                ++global_step;
                sum_total += l.total;
                sum_z += l.z;
                sum_reg += l.reg;
                ++n;
                if (!std::isfinite(l.total)) throw std::runtime_error("non-finite loss at step " + std::to_string(global_step));
                log << "{\"kind\":\"step\",\"step\":" << global_step << ",\"epoch\":" << epoch << ",\"batch\":" << b
                    << ",\"loss\":" << l.total << ",\"z_loss\":" << l.z << ",\"reg_loss\":" << l.reg
                    << ",\"time\":" << seconds_since(t_start) << "}\n";
                if (global_step % args.log_every == 0 || b == train_batches - 1) {
                    std::printf("\repoch %d  step %lld/%lld  loss %.5f (z %.5f reg %.5f)  %.2f s/step   ", epoch,
                                static_cast<long long>(b + 1), static_cast<long long>(train_batches), l.total, l.z, l.reg,
                                seconds_since(t_epoch) / static_cast<double>(n));
                    std::fflush(stdout);
                    log.flush();
                }
            }
            std::printf("\n");

            double val_total = 0, val_z = 0, val_reg = 0;
            int64_t nv = 0;
            if (val && val_batches > 0) {
                for (int64_t b = 0; b < val_batches; ++b) {
                    data.make_batch(data.plan_val(), static_cast<size_t>(b * B), B, in.visual, in.actions);
                    fill_random(in, *val, val_proj, val_target, val_dummy, static_cast<int64_t>(epoch) * val_batches + b, C, K, B, TP, 0.0f);
                    const lewm::Losses l = val->step(in);
                    val_total += l.total;
                    val_z += l.z;
                    val_reg += l.reg;
                    ++nv;
                }
            }
            const double tr = n ? sum_total / static_cast<double>(n) : NAN;
            const double vl = nv ? val_total / static_cast<double>(nv) : NAN;
            std::printf("epoch %d done in %.0f s: train loss %.5f (z %.5f reg %.5f)  val loss %.5f (z %.5f reg %.5f)\n", epoch,
                        seconds_since(t_epoch), tr, n ? sum_z / n : NAN, n ? sum_reg / n : NAN, vl, nv ? val_z / nv : NAN,
                        nv ? val_reg / nv : NAN);
            log << "{\"kind\":\"epoch\",\"epoch\":" << epoch << ",\"step\":" << global_step << ",\"train_loss\":" << tr
                << ",\"train_z_loss\":" << (n ? sum_z / n : NAN) << ",\"train_reg_loss\":" << (n ? sum_reg / n : NAN)
                << ",\"val_loss\":" << vl << ",\"val_z_loss\":" << (nv ? val_z / nv : NAN) << ",\"val_reg_loss\":" << (nv ? val_reg / nv : NAN)
                << ",\"epoch_seconds\":" << seconds_since(t_epoch) << ",\"time\":" << seconds_since(t_start) << "}\n";
            log.flush();

            if (args.save_every > 0 && ((epoch + 1) % args.save_every == 0 || epoch + 1 == args.epochs)) {
                const std::string dir = args.out + "/epoch_" + std::to_string(epoch + 1);
                std::filesystem::create_directories(dir);
                train.save_model(dir + "/model.gguf");
                train.save_optimizer(dir + "/optimizer.gguf");
                std::ofstream(dir + "/epoch.txt") << (epoch + 1) << "\n";
                train.save_model(args.out + "/model_latest.gguf");
                std::printf("saved %s\n", dir.c_str());
            }
            if (args.max_steps >= 0 && global_step >= args.max_steps) break;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nerror: %s\n", e.what());
        return 2;
    }
}
