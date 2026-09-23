// Parity tests: LeWM against reference dumps produced by tools/dump_reference.py.
// Fixtures live in LEWM_TESTDATA_DIR (default <repo>/testdata). A missing fixture skips its case
// rather than failing, so a fresh checkout without model files still builds and runs the rest.

#include <doctest/doctest.h>

#include <filesystem>
#include <string>

#include "lewm/parity.hpp"

namespace {

const std::string kDir = LEWM_TESTDATA_DIR;

bool fixture(const std::string& model, const std::string& dump) {
    return std::filesystem::exists(kDir + "/" + model) && std::filesystem::exists(kDir + "/" + dump);
}

void check_parity(const std::string& model, const std::string& dump, double tol) {
    if (!fixture(model, dump)) {
        MESSAGE("fixture missing, skipping: ", model, " / ", dump);
        return;
    }
    lewm::WorldModel wm(kDir + "/" + model);
    lewm::GgufStore d(kDir + "/" + dump, wm.backend());
    lewm::ParityReport rep = lewm::run_parity(wm, d, tol);
    INFO(rep.table());
    CHECK(rep.all_pass());
    CHECK(!rep.rows.empty());
}

}  // namespace

TEST_CASE("parity: seeded random-init model (CLS mode)") {
    check_parity("random_seed0.gguf", "random_seed0.dump.gguf", 1e-4);
}

TEST_CASE("parity: trained Wall checkpoint (CLS mode)") {
    check_parity("wall_ref.gguf", "wall_ref.dump.gguf", 1e-4);
}

TEST_CASE("parity: seeded random-init model (patch mode)") {
    check_parity("random_patch_seed0.gguf", "random_patch_seed0.dump.gguf", 1e-4);
}

TEST_CASE("hparams round-trip from a model file") {
    if (!std::filesystem::exists(kDir + "/random_seed0.gguf")) {
        MESSAGE("fixture missing, skipping");
        return;
    }
    lewm::WorldModel wm(kDir + "/random_seed0.gguf");
    const lewm::Hparams& hp = wm.hparams();
    CHECK(hp.image_size == 224);
    CHECK(hp.patch_size == 14);
    CHECK(hp.grid() == 16);
    CHECK(hp.encoder_tokens() == 257);
    CHECK(hp.cls_mode());
    CHECK(hp.predictor_num_patches == 1);
    CHECK(hp.code_dim == 384);
}
