#include "lewm/rng.hpp"

#include <cmath>

namespace lewm {

namespace {
constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

uint64_t splitmix64(uint64_t x) {
    x += kGolden;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
}  // namespace

uint64_t Rng::mix(uint64_t seed, uint64_t stream, uint64_t counter) {
    // three rounds keep seed/stream/counter from interacting linearly
    return splitmix64(splitmix64(splitmix64(seed) ^ stream) ^ counter);
}

uint64_t Rng::next_u64() { return mix(seed_, stream_, counter_++); }

double Rng::uniform() {
    // 53 random bits -> (0, 1): (bits + 0.5) / 2^53 is never 0 or 1
    const uint64_t bits = next_u64() >> 11;
    return (static_cast<double>(bits) + 0.5) * (1.0 / 9007199254740992.0);
}

void Rng::normal(std::vector<float>& out, size_t n) {
    out.resize(n);
    for (size_t i = 0; i < n; i += 2) {
        const double u1 = uniform();
        const double u2 = uniform();
        const double r = std::sqrt(-2.0 * std::log(u1));
        const double theta = 2.0 * M_PI * u2;
        out[i] = static_cast<float>(r * std::cos(theta));
        if (i + 1 < n) out[i + 1] = static_cast<float>(r * std::sin(theta));
    }
}

void Rng::laplace(std::vector<float>& out, size_t n, double scale) {
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const double u = uniform() - 0.5;  // (-0.5, 0.5)
        const double v = -scale * std::copysign(std::log(1.0 - 2.0 * std::fabs(u)), u);
        out[i] = static_cast<float>(v);
    }
}

void Rng::dropout_mask(std::vector<float>& out, size_t n, double p) {
    out.resize(n);
    const float keep = static_cast<float>(1.0 / (1.0 - p));
    for (size_t i = 0; i < n; ++i) out[i] = uniform() >= p ? keep : 0.0f;
}

}  // namespace lewm
