#include "lewm/dataset.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdexcept>

#include "ggml-cpu.h"
#include "lewm/gguf_store.hpp"

namespace lewm {

Dataset::Dataset(const std::string& dir) {
    // metadata and small tensors through a throwaway CPU backend
    ggml_backend_t cpu = ggml_backend_cpu_init();
    {
        GgufStore g(dir + "/dataset.gguf", cpu);
        meta_.n_episodes = g.get_u32("lewm.data.n_episodes");
        meta_.frames_per_episode = g.get_u32("lewm.data.frames_per_episode");
        meta_.steps = g.get_u32("lewm.data.steps");
        meta_.height = g.get_u32("lewm.data.height");
        meta_.width = g.get_u32("lewm.data.width");
        meta_.action_dim = g.get_u32("lewm.data.action_dim");
        meta_.frameskip = g.get_u32("lewm.data.frameskip");
        meta_.num_frames = g.get_u32("lewm.data.num_frames");

        actions_ = g.read("actions").data;  // {A, steps, n_episodes} ggml order == (n_ep, steps, A) numpy
        if (static_cast<int64_t>(actions_.size()) != meta_.n_episodes * meta_.steps * meta_.action_dim) {
            throw std::runtime_error("dataset: actions tensor has unexpected size");
        }
        auto read_plan = [&](const char* name, std::vector<PlanRow>& out) {
            ggml_tensor* t = g.tensor(name);  // {2, n} i32
            if (t->type != GGML_TYPE_I32 || t->ne[0] != 2) throw std::runtime_error(std::string("dataset: bad plan tensor ") + name);
            const int64_t n = t->ne[1];
            std::vector<int32_t> buf(static_cast<size_t>(2 * n));
            ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
            out.resize(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = {buf[static_cast<size_t>(2 * i)], buf[static_cast<size_t>(2 * i + 1)]};
        };
        read_plan("plan.train", plan_train_);
        read_plan("plan.val", plan_val_);
    }
    ggml_backend_free(cpu);

    const std::string frames_path = dir + "/frames.u8";
    fd_ = ::open(frames_path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("dataset: cannot open " + frames_path);
    struct stat st{};
    if (::fstat(fd_, &st) != 0) throw std::runtime_error("dataset: fstat failed");
    frames_bytes_ = static_cast<size_t>(st.st_size);
    const size_t expect = static_cast<size_t>(meta_.n_episodes * meta_.frames_per_episode * 3 * meta_.height * meta_.width);
    if (frames_bytes_ != expect) throw std::runtime_error("dataset: frames.u8 size does not match metadata");
    void* p = ::mmap(nullptr, frames_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error("dataset: mmap failed");
    frames_ = static_cast<const uint8_t*>(p);
}

Dataset::~Dataset() {
    if (frames_) ::munmap(const_cast<uint8_t*>(frames_), frames_bytes_);
    if (fd_ >= 0) ::close(fd_);
}

const uint8_t* Dataset::frame(int64_t episode, int64_t index) const {
    const size_t frame_bytes = static_cast<size_t>(3 * meta_.height * meta_.width);
    return frames_ + (static_cast<size_t>(episode) * static_cast<size_t>(meta_.frames_per_episode) + static_cast<size_t>(index)) * frame_bytes;
}

void Dataset::make_batch(const std::vector<PlanRow>& plan, size_t first, int64_t B, HostTensor& visual, HostTensor& actions) const {
    const int64_t T = meta_.num_frames;
    const int64_t fs = meta_.frameskip;
    const int64_t A = meta_.action_dim;
    const int64_t H = meta_.height, W = meta_.width;
    if (first + static_cast<size_t>(B) > plan.size()) throw std::runtime_error("dataset: batch past the end of the plan");

    visual = HostTensor::zeros({W, H, 3, T * B});
    actions = HostTensor::zeros({A * fs, T, B});
    const size_t frame_elems = static_cast<size_t>(3 * H * W);

    for (int64_t b = 0; b < B; ++b) {
        const PlanRow row = plan[first + static_cast<size_t>(b)];
        const int64_t ep = row[0], start = row[1];
        if (ep < 0 || ep >= meta_.n_episodes || start < 0 || start + fs * T > meta_.steps) {
            throw std::runtime_error("dataset: plan row out of range");
        }
        for (int64_t t = 0; t < T; ++t) {
            const uint8_t* src = frame(ep, start + fs * t);
            float* dst = visual.data.data() + static_cast<size_t>(t + T * b) * frame_elems;
            for (size_t i = 0; i < frame_elems; ++i) {
                // exactly the reference transform: x/255, then Normalize(0.5, 0.5)
                const float v = static_cast<float>(src[i]) / 255.0f;
                dst[i] = (v - 0.5f) / 0.5f;
            }
            // chunked action for frame t: raw steps start+fs*t .. start+fs*t+fs-1, laid out (f d)
            float* adst = actions.data.data() + static_cast<size_t>((t + T * b) * A * fs);
            for (int64_t f = 0; f < fs; ++f) {
                const float* a = actions_.data() + static_cast<size_t>((ep * meta_.steps + start + fs * t + f) * A);
                for (int64_t d = 0; d < A; ++d) adst[f * A + d] = a[d];
            }
        }
    }
}

}  // namespace lewm
