#include <doctest/doctest.h>

#include "lewm/host_tensor.hpp"

TEST_CASE("HostTensor shape string is numpy order") {
    lewm::HostTensor t = lewm::HostTensor::zeros({384, 257, 4});
    CHECK(t.n_dims == 3);
    CHECK(t.numel() == 384 * 257 * 4);
    CHECK(t.shape_str() == "(4, 257, 384)");
    CHECK(t.data.size() == static_cast<size_t>(t.numel()));
}
