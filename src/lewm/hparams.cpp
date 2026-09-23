#include "lewm/hparams.hpp"

#include <sstream>
#include <stdexcept>

#include "lewm/gguf_store.hpp"

namespace lewm {

Hparams Hparams::from_store(const GgufStore& s) {
    Hparams h;
    auto u = [&](const char* k) { return s.get_u32(std::string("lewm.") + k); };
    h.image_size = u("image_size");
    h.patch_size = u("patch_size");
    h.dim = u("dim");
    h.encoder_depth = u("encoder_depth");
    h.encoder_heads = u("encoder_heads");
    h.encoder_dim_head = u("encoder_dim_head");
    h.encoder_mlp_dim = u("encoder_mlp_dim");
    h.encoder_feature = s.get_str("lewm.encoder_feature");
    h.encoder_proj_hidden = u("encoder_proj_hidden");
    h.code_dim = u("code_dim");
    h.action_dim = u("action_dim");
    h.action_smoothed_dim = u("action_smoothed_dim");
    h.action_mlp_hidden = u("action_mlp_hidden");
    h.predictor_depth = u("predictor_depth");
    h.predictor_heads = u("predictor_heads");
    h.predictor_dim_head = u("predictor_dim_head");
    h.predictor_mlp_dim = u("predictor_mlp_dim");
    h.predictor_num_frames = u("predictor_num_frames");
    h.predictor_num_patches = u("predictor_num_patches");
    h.predictor_proj_hidden = u("predictor_proj_hidden");
    h.link = s.get_str("lewm.link");

    if (h.image_size % h.patch_size != 0) throw std::runtime_error("image_size not divisible by patch_size");
    if (h.encoder_feature != "cls" && h.encoder_feature != "patch") {
        throw std::runtime_error("unknown encoder_feature: " + h.encoder_feature);
    }
    const uint32_t expect_p = h.cls_mode() ? 1 : h.grid() * h.grid();
    if (h.predictor_num_patches != expect_p) {
        throw std::runtime_error("predictor_num_patches does not match encoder_feature");
    }
    if (h.link != "identity" && h.link != "relu" && h.link != "reprelu") {
        throw std::runtime_error("unknown link: " + h.link);
    }
    return h;
}

std::string Hparams::describe() const {
    std::ostringstream os;
    os << "image " << image_size << "/" << patch_size << " grid " << grid() << " | encoder dim " << dim << " depth "
       << encoder_depth << " heads " << encoder_heads << "x" << encoder_dim_head << " mlp " << encoder_mlp_dim
       << " feature " << encoder_feature << " proj " << encoder_proj_hidden << "->" << code_dim
       << " | action " << action_dim << "->" << action_smoothed_dim << "->" << action_mlp_hidden << "->" << code_dim
       << " | predictor depth " << predictor_depth << " heads " << predictor_heads << "x" << predictor_dim_head
       << " mlp " << predictor_mlp_dim << " frames " << predictor_num_frames << " patches " << predictor_num_patches
       << " proj " << predictor_proj_hidden << " | link " << link;
    return os.str();
}

}  // namespace lewm
