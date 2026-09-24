// The C++ RNG must match tools/lewm_rng.py bit for bit; expected values below were printed by the
// Python mirror (see the commit that added this file for the exact snippet).

#include <doctest/doctest.h>

#include <cmath>
#include <vector>

#include "lewm/rng.hpp"

TEST_CASE("Rng: raw stream matches the Python mirror") {
    lewm::Rng r(0, 1);
    CHECK(r.next_u64() == 4964578127960768432ULL);
    CHECK(r.next_u64() == 5067554077270220563ULL);
    CHECK(r.next_u64() == 6768782832058643234ULL);
}

TEST_CASE("Rng: uniforms match") {
    lewm::Rng r(0, 1);
    CHECK(r.uniform() == doctest::Approx(0.26913031959045414).epsilon(1e-15));
    CHECK(r.uniform() == doctest::Approx(0.2747126569882074).epsilon(1e-15));
    CHECK(r.uniform() == doctest::Approx(0.36693645258002833).epsilon(1e-15));
}

TEST_CASE("Rng: normals, laplace and dropout masks match") {
    std::vector<float> v;
    lewm::Rng r(0, 1);
    r.normal(v, 5);
    const float expect_n[5] = {-0.250568956f, 1.60072935f, 1.37393737f, 0.342679679f, -0.600541711f};
    for (int i = 0; i < 5; ++i) CHECK(v[i] == doctest::Approx(expect_n[i]).epsilon(1e-6));

    lewm::Rng l(0, 2);
    l.laplace(v, 4, 0.70710678118654752440);
    const float expect_l[4] = {-0.786958873f, 0.317070425f, 0.341866344f, -0.695340693f};
    for (int i = 0; i < 4; ++i) CHECK(v[i] == doctest::Approx(expect_l[i]).epsilon(1e-6));

    lewm::Rng d(7, 3);
    d.seek(1000);
    d.dropout_mask(v, 12, 0.1);
    const float keep = 1.1111111640930176f;
    const float expect_d[12] = {keep, keep, keep, keep, keep, 0, keep, 0, keep, keep, keep, keep};
    for (int i = 0; i < 12; ++i) CHECK(v[i] == expect_d[i]);

    lewm::Rng s(0, 1);
    s.seek(5);
    s.normal(v, 2);
    CHECK(v[0] == doctest::Approx(-0.171534896f).epsilon(1e-6));
    CHECK(v[1] == doctest::Approx(-1.43429828f).epsilon(1e-6));
}
