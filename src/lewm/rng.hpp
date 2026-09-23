#pragma once

#include <cstdint>
#include <vector>

namespace lewm {

/// Portable counter-based random numbers, mirrored bit for bit by tools/lewm_rng.py, so LeWM and
/// the Python reference can consume identical randomness for trajectory parity.
///
/// Stream: splitmix64 over (seed, stream, counter) -> uint64 -> uniform double in (0, 1).
/// Normals by Box-Muller in double precision, Laplace by inverse CDF, then rounded to float.
class Rng {
public:
    Rng(uint64_t seed, uint64_t stream) : seed_(seed), stream_(stream) {}

    /// Advance to a specific position (e.g. per training step) so draws are reproducible.
    void seek(uint64_t counter) { counter_ = counter; }
    uint64_t position() const { return counter_; }

    uint64_t next_u64();
    double uniform();  // in (0, 1), never 0

    /// n standard normals via Box-Muller (pairs; the second of each pair is used next).
    void normal(std::vector<float>& out, size_t n);
    /// n Laplace(0, scale) samples.
    void laplace(std::vector<float>& out, size_t n, double scale);
    /// n values in {0, 1/(1-p)} with P(nonzero) = 1-p  (inverted dropout mask).
    void dropout_mask(std::vector<float>& out, size_t n, double p);

    static uint64_t mix(uint64_t seed, uint64_t stream, uint64_t counter);

private:
    uint64_t seed_, stream_, counter_ = 0;
};

}  // namespace lewm
