#include "lewm/parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace lewm {

namespace {

ParityRow compare(const std::string& name, const HostTensor& ref, const HostTensor* got, double tol) {
    ParityRow r;
    r.name = name;
    r.shape = ref.shape_str();
    for (float v : ref.data) r.ref_max_abs = std::max(r.ref_max_abs, static_cast<double>(std::fabs(v)));
    if (!got) {
        r.present = false;
        return r;
    }
    if (got->data.size() != ref.data.size()) {
        r.shape += " vs " + got->shape_str();
        r.max_abs_diff = std::numeric_limits<double>::infinity();
        return r;
    }
    double sum = 0;
    for (size_t i = 0; i < ref.data.size(); ++i) {
        const double d = std::fabs(static_cast<double>(ref.data[i]) - static_cast<double>(got->data[i]));
        if (std::isnan(d)) {
            r.max_abs_diff = std::numeric_limits<double>::quiet_NaN();
            return r;
        }
        r.max_abs_diff = std::max(r.max_abs_diff, d);
        sum += d;
    }
    r.mean_abs_diff = ref.data.empty() ? 0 : sum / static_cast<double>(ref.data.size());
    r.pass = r.max_abs_diff <= tol;
    return r;
}

}  // namespace

bool ParityReport::all_pass() const {
    return std::all_of(rows.begin(), rows.end(), [](const ParityRow& r) { return r.present && r.pass; });
}

std::string ParityReport::table() const {
    std::ostringstream os;
    char line[256];
    std::snprintf(line, sizeof line, "%-28s %-18s %12s %12s %10s  %s\n", "tensor", "shape", "max|diff|", "mean|diff|",
                  "ref max", "result");
    os << line;
    for (const ParityRow& r : rows) {
        if (!r.present) {
            std::snprintf(line, sizeof line, "%-28s %-18s %12s %12s %10.3g  %s\n", r.name.c_str(), r.shape.c_str(), "-",
                          "-", r.ref_max_abs, "MISSING");
        } else {
            std::snprintf(line, sizeof line, "%-28s %-18s %12.3e %12.3e %10.3g  %s\n", r.name.c_str(), r.shape.c_str(),
                          r.max_abs_diff, r.mean_abs_diff, r.ref_max_abs, r.pass ? "ok" : "FAIL");
        }
        os << line;
    }
    os << (all_pass() ? "PARITY OK" : "PARITY FAILED") << " (tolerance " << tolerance << ")\n";
    return os.str();
}

ParityReport run_parity(WorldModel& wm, const GgufStore& dump, double tolerance) {
    ParityReport rep;
    rep.tolerance = tolerance;

    const HostTensor visual = dump.read("in.visual");    // {W, H, 3, T}
    const HostTensor actions = dump.read("in.actions");  // {A, T_act}
    const uint32_t num_hist = dump.get_u32("lewm.dump.num_hist");

    Outputs got = wm.encode(visual, true);
    Outputs acts = wm.encode_actions(actions);
    got.insert(acts.begin(), acts.end());

    // Teacher-forced prediction on the first num_hist frames, exactly as the dump did.
    const HostTensor& z = got.at("encoder.z");  // {C, P, T}
    const HostTensor& a = acts.at("action_encoder.out");  // {C, T_act}
    const int64_t C = z.ne[0], P = z.ne[1];
    HostTensor z_hist = HostTensor::zeros({C, P, num_hist});
    std::copy(z.data.begin(), z.data.begin() + static_cast<std::ptrdiff_t>(C * P * num_hist), z_hist.data.begin());
    HostTensor a_hist = HostTensor::zeros({C, num_hist});
    std::copy(a.data.begin(), a.data.begin() + static_cast<std::ptrdiff_t>(C * num_hist), a_hist.data.begin());
    Outputs pred = wm.predict(z_hist, a_hist, true);
    got.insert(pred.begin(), pred.end());

    // Rollout from frame 0 over every action.
    HostTensor z0 = HostTensor::zeros({C, P, 1});
    std::copy(z.data.begin(), z.data.begin() + static_cast<std::ptrdiff_t>(C * P), z0.data.begin());
    got.emplace("rollout.z", wm.rollout(z0, a));

    for (const std::string& name : dump.tensor_names()) {
        if (name.rfind("in.", 0) == 0) continue;
        const HostTensor ref = dump.read(name);
        auto it = got.find(name);
        rep.rows.push_back(compare(name, ref, it == got.end() ? nullptr : &it->second, tolerance));
    }
    return rep;
}

}  // namespace lewm
