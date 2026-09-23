#pragma once

#include <string>
#include <vector>

#include "lewm/host_tensor.hpp"
#include "lewm/world_model.hpp"

namespace lewm {

struct ParityRow {
    std::string name;
    std::string shape;
    double max_abs_diff = 0;
    double mean_abs_diff = 0;
    double ref_max_abs = 0;
    bool present = true;  // false when LeWM produced no tensor of that name
    bool pass = false;
};

struct ParityReport {
    std::vector<ParityRow> rows;
    double tolerance = 1e-4;
    bool all_pass() const;
    std::string table() const;
};

/// Run every stage LeWM has against a reference dump and compare each tensor the dump contains.
ParityReport run_parity(WorldModel& wm, const GgufStore& dump, double tolerance);

}  // namespace lewm
