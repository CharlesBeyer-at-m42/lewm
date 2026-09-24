#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "lewm/host_tensor.hpp"

namespace lewm {

/// A dataset exported by tools/export_dataset.py: a memory-mapped uint8 frame file plus a GGUF with
/// normalised actions, the train/validation split, and the reference's batch plan.
class Dataset {
public:
    struct Meta {
        int64_t n_episodes = 0;
        int64_t frames_per_episode = 0;
        int64_t steps = 0;  // raw actions per episode
        int64_t height = 0, width = 0;
        int64_t action_dim = 0;  // raw action width (before chunking)
        int64_t frameskip = 0;
        int64_t num_frames = 0;  // frames per sample, = num_hist + num_pred
    };
    using PlanRow = std::array<int32_t, 2>;  // (episode, start)

    explicit Dataset(const std::string& dir);
    ~Dataset();
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    const Meta& meta() const { return meta_; }
    const std::vector<PlanRow>& plan_train() const { return plan_train_; }
    const std::vector<PlanRow>& plan_val() const { return plan_val_; }
    int64_t chunked_action_dim() const { return meta_.action_dim * meta_.frameskip; }

    /// Fill one batch from plan rows [first, first + B).
    /// visual: {W, H, 3, T*B} normalised like the reference transform; actions: {A*frameskip, T, B}.
    void make_batch(const std::vector<PlanRow>& plan, size_t first, int64_t B, HostTensor& visual, HostTensor& actions) const;

private:
    Meta meta_;
    std::vector<float> actions_;  // (n_episodes, steps, action_dim), normalised
    std::vector<PlanRow> plan_train_, plan_val_;
    const uint8_t* frames_ = nullptr;
    size_t frames_bytes_ = 0;
    int fd_ = -1;

    const uint8_t* frame(int64_t episode, int64_t index) const;
};

}  // namespace lewm
