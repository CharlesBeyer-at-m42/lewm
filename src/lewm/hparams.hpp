#pragma once

#include <cstdint>
#include <string>

namespace lewm {

class GgufStore;

/// Hyperparameters of the LpWM path, read from a model file's `lewm.*` metadata.
/// Field names match tools/lewm_ref.py::Hparams one to one.
struct Hparams {
    uint32_t image_size = 224;
    uint32_t patch_size = 14;
    uint32_t dim = 384;  // encoder residual width
    uint32_t encoder_depth = 12;
    uint32_t encoder_heads = 3;
    uint32_t encoder_dim_head = 64;
    uint32_t encoder_mlp_dim = 768;
    std::string encoder_feature = "cls";  // "cls" | "patch"
    uint32_t encoder_proj_hidden = 2048;
    uint32_t code_dim = 384;  // width of a code
    uint32_t action_dim = 10;  // chunked action width
    uint32_t action_smoothed_dim = 10;
    uint32_t action_mlp_hidden = 1536;
    uint32_t predictor_depth = 6;
    uint32_t predictor_heads = 16;
    uint32_t predictor_dim_head = 64;
    uint32_t predictor_mlp_dim = 2048;
    uint32_t predictor_num_frames = 3;  // history window
    uint32_t predictor_num_patches = 1;  // tokens per frame (P)
    uint32_t predictor_proj_hidden = 2048;
    std::string link = "reprelu";  // "identity" | "relu" | "reprelu"

    uint32_t grid() const { return image_size / patch_size; }
    uint32_t encoder_tokens() const { return grid() * grid() + 1; }  // patches + cls
    uint32_t tokens_per_frame() const { return predictor_num_patches; }
    bool cls_mode() const { return encoder_feature == "cls"; }

    static Hparams from_store(const GgufStore& s);
    std::string describe() const;
};

}  // namespace lewm
