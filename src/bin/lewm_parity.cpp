// lewm-parity: run LeWM on a reference dump's inputs and compare every tensor the dump holds.
//
//   lewm-parity <model.gguf> <dump.gguf> [--tol 1e-4] [--threads N]
//
// Exit code 0 when every tensor is within tolerance, 1 otherwise, 2 on usage or load errors.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "lewm/parity.hpp"

int main(int argc, char** argv) {
    std::string model_path, dump_path;
    double tol = 1e-4;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc) {
            tol = std::atof(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (model_path.empty()) {
            model_path = a;
        } else if (dump_path.empty()) {
            dump_path = a;
        } else {
            std::fprintf(stderr, "unexpected argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (model_path.empty() || dump_path.empty()) {
        std::fprintf(stderr, "usage: lewm-parity <model.gguf> <dump.gguf> [--tol 1e-4] [--threads N]\n");
        return 2;
    }
    try {
        lewm::WorldModel wm(model_path, threads);
        std::printf("model: %s\n  %s\n", model_path.c_str(), wm.hparams().describe().c_str());
        lewm::GgufStore dump(dump_path, wm.backend());
        std::printf("dump:  %s\n\n", dump_path.c_str());
        lewm::ParityReport rep = lewm::run_parity(wm, dump, tol);
        std::fputs(rep.table().c_str(), stdout);
        return rep.all_pass() ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
